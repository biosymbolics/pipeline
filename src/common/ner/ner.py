"""
Named-entity recognition using spacy

No hardware acceleration: see https://github.com/explosion/spaCy/issues/10783#issuecomment-1132523032
"""
from typing import Any, Optional
import spacy
from scispacy.linking import EntityLinker  # required to use 'scispacy_linker' pipeline
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy_llm.util import assemble
from pydash import flatten
import logging
import warnings
import spacy_llm

from .cleaning import clean_entities
from .debugging import debug_pipeline
from .linking import enrich_with_canonical
from .patterns import INDICATION_SPACY_PATTERNS, INTERVENTION_SPACY_PATTERNS
from .types import GetTokenizer, SpacyPatterns


warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.amp.autocast_mode"
)
spacy_llm.logger.addHandler(logging.StreamHandler())
spacy_llm.logger.setLevel(logging.DEBUG)

common_nlp = spacy.load("en_core_web_sm", disable=["ner"])


def get_default_tokenizer(nlp: Language):
    return Tokenizer(nlp.vocab)


LINKER_CONFIG = {
    "resolve_abbreviations": True,
    "linker_name": "umls",
    "threshold": 0.7,
    "filter_for_definitions": False,
    "no_definition_threshold": 0.7,
}


class NerTagger:
    def __init__(
        self,
        use_llm: Optional[bool] = False,
        # alt models: en_core_sci_scibert, en_ner_bionlp13cg_md, en_ner_bc5cdr_md
        model: Optional[str] = "en_core_sci_scibert",
        rule_sets: Optional[list[SpacyPatterns]] = None,
        get_tokenizer: Optional[GetTokenizer] = None,
    ):
        """
        Named-entity recognition using spacy

        Args:
            use_llm (Optional[bool], optional): Use LLM model. Defaults to False. if true, nothing below is used.
            model (str, optional): SpaCy model. Defaults to "en_core_sci_scibert".
            rule_sets (Optional[list[SpacyPatterns]], optional): SpaCy patterns. Defaults to None.
            get_tokenizer (Optional[GetTokenizer], optional): SpaCy tokenizer. Defaults to None.
        """
        self.model = model
        self.use_llm = use_llm
        self.rule_sets = (
            [
                INDICATION_SPACY_PATTERNS,
                INTERVENTION_SPACY_PATTERNS,
            ]
            if rule_sets is None
            else rule_sets
        )

        self.get_tokenizer = (
            get_default_tokenizer if get_tokenizer is None else get_tokenizer
        )

        self.__init_tagger()

    def __init_tagger(self):
        nlp: Language = (
            spacy.blank("en")
            if self.use_llm or not self.model
            else spacy.load(self.model)
        )

        if self.use_llm:
            nlp = assemble("configs/config.cfg")
        else:
            nlp.tokenizer = self.get_tokenizer(nlp)
            nlp.add_pipe("merge_entities", after="ner")
            ruler = nlp.add_pipe(
                "entity_ruler",
                config={"validate": True, "overwrite_ents": True},
                after="merge_entities",
            )
            for set in self.rule_sets:
                ruler.add_patterns(set)  # type: ignore

        nlp.add_pipe("scispacy_linker", config=LINKER_CONFIG)

        logging.info("Setting NER pipeline: %s", nlp)
        self.nlp = nlp

    def extract(self, content: list[str]):
        """
        Extract named entities from a list of content
        - basic SpaCy pipeline
        - applies rule_sets
        - applies scispacy_linker (canonical mapping to UMLS)

        Args:
            content (list[str]): list of content on which to do NER

        Examples:
            >>> tagger.extract("SMALL MOLECULE INHIBITORS OF NF-kB INDUCING KINASE")
            >>> tagger.extract("Interferon alpha and omega antibody antagonists")
            >>> tagger.extract("Inhibitors of beta secretase")
            >>> tagger.extract("Antibodies specifically binding hla-dr/colii_259 complex and their uses")
            >>> tagger.extract("Use of small molecules to enhance mafa expression in pancreatic endocrine cells")
            >>> tagger.extract("Macrocyclic 2-amino-3-fluoro-but-3-enamides as inhibitors of mcl-1")
            >>> tagger.extract("Inhibitors of antigen presentation by hla-dr")
            >>> tagger.extract("Antibodies specifically binding tim-3 and their uses")
            >>> tagger.extract("Antibodies and antigen binding peptides for factor xia inhibitors and uses thereof")
            >>> tagger.extract("P2x7 modulating n-acyl-triazolopyrazines")
            >>> tagger.extract("Small molecule inhibitors of the jak family of kinases")
            >>> tagger.extract("Inhibitors of keap1-nrf2 protein-protein interaction")
        """
        if not isinstance(content, list):
            content = [content]

        docs = [self.nlp(batch) for batch in content]
        entities = flatten([doc.ents for doc in docs])

        if not self.nlp:
            logging.error("NER tagger not initialized")
            return []
        enriched = enrich_with_canonical(entities, nlp=self.nlp)
        entity_names = clean_entities(list(enriched.keys()), common_nlp)

        logging.info("Entity names: %s", entity_names)
        # debug_pipeline(docs, nlp)

        return entity_names

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.nlp:
            return self.extract(*args, **kwds)
