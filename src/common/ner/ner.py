"""
Named-entity recognition using spacy

No hardware acceleration: see https://github.com/explosion/spaCy/issues/10783#issuecomment-1132523032
"""
from functools import partial
import time
from typing import Any, Optional, TypeVar, Union
from pydash import flatten
import spacy
from spacy.language import Language
from spacy.tokens import Span
from spacy.tokenizer import Tokenizer
import spacy_llm
from spacy_llm.util import assemble
import logging
import warnings

from clients.spacy import Spacy
from common.ner.normalizer import TermNormalizer
from common.utils.functional import compose

from .cleaning import sanitize_entities
from .debugging import debug_pipeline
from .patterns import INDICATION_SPACY_PATTERNS, INTERVENTION_SPACY_PATTERNS
from .types import GetTokenizer, SpacyPatterns

T = TypeVar("T", bound=Union[Span, str])

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.amp.autocast_mode"
)
spacy_llm.logger.addHandler(logging.StreamHandler())
spacy_llm.logger.setLevel(logging.INFO)

DocEntities = list[tuple[str, str]]


def get_default_tokenizer(nlp: Language):
    return Tokenizer(nlp.vocab)


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
            use_llm (Optional[bool], optional): Use LLM model. Defaults to False. If true, no rules or anything are used.
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
        self.normalizer = TermNormalizer()

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

        logging.info(
            "Init NER pipeline took %s seconds",
            round(time.time() - start_time, 2),
        )
        self.nlp = nlp

    def __get_entities(self, docs) -> list[DocEntities]:
        """
        Get normalized entities from a list of docs
        """
        entity_names = [e.lemma_ or e.text for doc in docs for e in doc.ents]
        normalization_map = self.normalizer.generate_map(entity_names)

        def __get_canonical(e):
            term = e.lemma_ or e.text
            entry = normalization_map.get(term) or None
            return entry.canonical_name if entry else term

        get_entities = compose(
            lambda span: [(__get_canonical(e), e.label_) for e in span],
            partial(sanitize_entities, nlp=self.common_nlp),
        )

        ents_by_doc = [get_entities([span for span in doc.ents]) for doc in docs]

        logging.info("Entities: %s", ents_by_doc)

        return ents_by_doc

    def extract(
        self, content: list[str], flatten_results: bool = True
    ) -> Union[list[str], list[DocEntities]]:
        """
        Extract named entities from a list of content
        - basic SpaCy pipeline
        - applies rule_sets
        - normalizes terms

        Args:
            content (list[str]): list of content on which to do NER
            flatten (bool, optional): flatten results.
                Defaults to True and result is returned as list[str].
                Otherwise, returns list[list[tuple[str, str]]] (entity and its type/label per doc).

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

        logging.info("Starting NER pipeline with %s docs", len(content))
        docs = list(self.nlp.pipe(content))

        ents_by_doc = self.__get_entities(docs)

        if flatten_results:
            return [e[0] for e in flatten(ents_by_doc)]

        return ents_by_doc

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if self.nlp:
            return self.extract(*args, **kwds)

    @classmethod
    def get_instance(cls, **kwargs) -> "NerTagger":
        # Convert kwargs to a hashable type
        kwargs_tuple = tuple(sorted(kwargs.items()))
        if kwargs_tuple not in cls._instances:
            logging.info("Creating new instance of %s", cls)
            cls._instances[kwargs_tuple] = cls(**kwargs)
        else:
            logging.info("Using existing instance of %s", cls)
        return cls._instances[kwargs_tuple]
