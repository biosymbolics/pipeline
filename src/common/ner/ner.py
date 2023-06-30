"""
Named-entity recognition using spacy

No hardware acceleration: see https://github.com/explosion/spaCy/issues/10783#issuecomment-1132523032
"""
from functools import reduce
import time
from typing import Any, Literal, Optional, TypeVar, Union
from bs4 import BeautifulSoup
from pydash import flatten
import spacy
from spacy.language import Language
from spacy.tokens import Span, Doc
from spacy.tokenizer import Tokenizer
import spacy_llm
from spacy_llm.util import assemble
import logging
import warnings

from common.ner.linker import TermLinker
from common.utils.string import chunk_list

from .cleaning import sanitize_entities
from .patterns import INDICATION_SPACY_PATTERNS, INTERVENTION_SPACY_PATTERNS
from .types import GetTokenizer, CanonicalEntity, SpacyPatterns

T = TypeVar("T", bound=Union[Span, str])
DocEntities = list[tuple[str, str]]
LinkedDocEntities = list[tuple[str, str, Optional[CanonicalEntity]]]
ContentType = Literal["text", "html"]
CHUNK_SIZE = 10000

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.amp.autocast_mode"
)
spacy_llm.logger.addHandler(logging.StreamHandler())
spacy_llm.logger.setLevel(logging.DEBUG)


def get_default_tokenizer(nlp: Language):
    return Tokenizer(nlp.vocab)


class NerTagger:
    """
    Named-entity recognition using spacy and other means
    """

    _instances: dict[tuple, Any] = {}

    def __init__(
        self,
        use_llm: Optional[bool] = True,
        model: Optional[str] = "en_core_sci_lg",
        rule_sets: list[SpacyPatterns] = [
            INDICATION_SPACY_PATTERNS,
            INTERVENTION_SPACY_PATTERNS,
        ],
        get_tokenizer: Optional[GetTokenizer] = None,
        content_type: Optional[ContentType] = "text",
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
        self.rule_sets = rule_sets

        self.get_tokenizer = (
            get_default_tokenizer if get_tokenizer is None else get_tokenizer
        )

        self.content_type = content_type
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

        self.linker = TermLinker()
        self.nlp = nlp

        logging.info(
            "Init NER pipeline took %s seconds",
            round(time.time() - start_time, 2),
        )

    def __normalize_and_link(
        self, doc: Doc, entity_types: Optional[list[str]] = None
    ) -> LinkedDocEntities:
        """
        Normalize entities for a single doc

        - santize entities (filtering, removing of undesired characters, lemmatization)
        - normalization (canonicalization, synonymization)

        Args:
            doc (Doc): SpaCy doc
            entity_types (Optional[list[str]], optional): filter by entity types. Defaults to None (all types permitted)
        """
        entity_set = [(span.text, span.label_) for span in doc.ents]

        # basic filtering, character removal, lemmatization
        sanitized = sanitize_entities(entity_set)
        linked_entity_map = dict(self.linker([tup[0] for tup in sanitized]))
        logging.info("linked_entity_map %s", linked_entity_map)

        # canonicalization, synonymization
        normalized = [(e[0], e[1], linked_entity_map.get(e[0])) for e in sanitized]

        # filter by entity types, if provided
        if entity_types:
            return [e for e in normalized if e[1] in entity_types]

        return normalized

    def __prep_doc(self, content: list[str]) -> list[str]:
        """
        Prepares a list of content for NER
        """
        _content = content.copy()

        # if use_llm, no tokenization
        if self.use_llm:
            if self.content_type == "html":
                # strip out all the HTML tags
                _content = [
                    " ".join(BeautifulSoup(c).get_text(separator=" ") for c in _content)
                ]

            # chunk it up (spacy-llm doesn't use langchain for chaining, i guess?)
            _content = flatten(chunk_list(_content, CHUNK_SIZE))

        return _content

    def extract(
        self,
        content: list[str],
        flatten_results: bool = True,
        entity_types: Optional[list[str]] = None,
    ) -> Union[list[str], list[LinkedDocEntities]]:
        """
        Extract named entities from a list of content
        - basic SpaCy pipeline
        - applies rule_sets
        - normalizes terms

        Args:
            content (list[str]): list of content on which to do NER
            flatten_results (bool, optional): flatten results.
                Defaults to True and result is returned as list[str].
                Otherwise, returns list[list[tuple[str, str]]] (entity and its type/label per doc).
            entity_types (Optional[list[str]], optional): filter by entity types. Defaults to None (all types permitted)

        Examples:
            >>> tagger.extract(["SMALL MOLECULE INHIBITORS OF NF-kB INDUCING KINASE"])
            >>> tagger.extract(["Interferon alpha and omega antibody antagonists"])
            >>> tagger.extract(["Inhibitors of beta secretase"])
        """
        if not self.nlp:
            raise Exception("NER tagger not initialized")

        if not isinstance(content, list):
            raise Exception("Content must be a list")

        logging.info("Starting NER pipeline with %s docs", len(content))

        steps = [
            self.__prep_doc,
            self.nlp.pipe,
            lambda docs: [
                self.__normalize_and_link(doc, entity_types) for doc in docs
            ],  # TODO: linking would be faster if done in batch
        ]
        ents_by_doc = reduce(lambda x, func: func(x), steps, content.copy())

        logging.info("Entities: %s", ents_by_doc)

        # return just the names if flatten_results is True
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
