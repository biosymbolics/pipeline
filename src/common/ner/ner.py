"""
Named-entity recognition using spacy

No hardware acceleration: see https://github.com/explosion/spaCy/issues/10783#issuecomment-1132523032
"""
import time
from typing import Any, Optional, Union
import spacy
from scispacy.linking import EntityLinker  # required to use 'scispacy_linker' pipeline
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy_llm.util import assemble
from pydash import flatten
import logging
import warnings
import spacy_llm

from clients.spacy import Spacy

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
    _instances: dict[tuple, Any] = {}

    def __init__(
        self,
        use_llm: Optional[bool] = True,
        # alt models: en_core_sci_scibert, en_ner_bionlp13cg_md, en_ner_bc5cdr_md
        model: Optional[str] = "en_core_sci_lg",
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

        self.common_nlp = Spacy.get_instance("en_core_web_sm", disable=["ner"])

        self.__init_tagger()

    def __init_tagger(self):
        start_time = time.time()
        nlp = (
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

        # nlp.add_pipe("scispacy_linker", config=LINKER_CONFIG)

        logging.info(
            "Init NER pipeline took %s seconds",
            round(time.time() - start_time, 2),
        )
        self.nlp = nlp

    def extract(
        self, content: list[str], flatten_results: bool = True
    ) -> Union[list[str], list[list[str]]]:
        """
        Extract named entities from a list of content
        - basic SpaCy pipeline
        - applies rule_sets
        - applies scispacy_linker (canonical mapping to UMLS)

        Args:
            content (list[str]): list of content on which to do NER
            flatten (bool, optional): flatten results.
                Defaults to True, which means result is list[str].
                Otherwise, returns list[list[str]].

        Examples:
            >>> tagger.extract("SMALL MOLECULE INHIBITORS OF NF-kB INDUCING KINASE")
            >>> tagger.extract("Interferon alpha and omega antibody antagonists")
            >>> tagger.extract("Inhibitors of beta secretase")
        """
        if not isinstance(content, list):
            content = [content]

        if not self.nlp:
            logging.error("NER tagger not initialized")
            raise Exception("NER tagger not initialized")

        docs = self.nlp.pipe(content)

        if flatten_results:
            entities = flatten([doc.ents for doc in docs])
            entity_names = clean_entities(
                [e.lemma_ or e.text for e in entities], self.common_nlp
            )
        else:
            entities = [doc.ents for doc in docs]
            entity_names = [
                clean_entities([e.lemma_ or e.text for e in ent], self.common_nlp)  # type: ignore
                for ent in entities
            ]

        # enriched = enrich_with_canonical(entities, nlp=self.nlp)

        logging.info("Entity names: %s", flatten(entity_names))
        # debug_pipeline(docs, nlp)

        return entity_names

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.nlp:
            return self.extract(*args, **kwds)

    @classmethod
    def get_instance(cls, **kwargs) -> "NerTagger":
        # Convert kwargs to a hashable type
        kwargs_tuple = tuple(sorted(kwargs.items()))
        if kwargs_tuple not in cls._instances:
            cls._instances[kwargs_tuple] = cls(**kwargs)
        return cls._instances[kwargs_tuple]
