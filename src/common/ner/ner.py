"""
Named-entity recognition using spacy

No hardware acceleration: see https://github.com/explosion/spaCy/issues/10783#issuecomment-1132523032
"""
from functools import reduce
import time
from typing import Any, Literal, Optional, TypeVar, Union
from pydash import flatten
from spacy.tokens import Span, Doc

import spacy_llm
from spacy_llm.util import assemble
import logging
import warnings
from common.ner.binder.binder import BinderNlp

from common.ner.linker import TermLinker
from common.utils.extraction.html import extract_text
from common.utils.string import chunk_list

from .cleaning import sanitize_entities
from .types import DocEntities

T = TypeVar("T", bound=Union[Span, str])
ContentType = Literal["text", "html"]
CHUNK_SIZE = 10000

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.amp.autocast_mode"
)
spacy_llm.logger.addHandler(logging.StreamHandler())
spacy_llm.logger.setLevel(logging.INFO)


class NerTagger:
    """
    Named-entity recognition using spacy and other means
    """

    _instances: dict[tuple, Any] = {}

    def __init__(
        self,
        use_llm: Optional[bool] = True,
        llm_config: Optional[str] = "configs/patents/config.cfg",
        model: Optional[str] = "model.pt",  # ignored if use_llm is True
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
        self.content_type = content_type
        self.llm_config = llm_config
        self.linker: Optional[TermLinker] = None  # lazy initialization
        start_time = time.time()

        if self.use_llm:
            if not self.llm_config:
                raise ValueError("Must provide llm_config if use_llm is True")
            self.nlp = assemble(self.llm_config)
        elif self.model:
            if not self.model.endswith(".pt"):
                raise ValueError("Model must be torch")
            self.nlp = BinderNlp(self.model)
        else:
            raise ValueError("Must provide either use_llm or model")

        logging.info(
            "Init NER pipeline took %s seconds",
            round(time.time() - start_time, 2),
        )

    def __normalize(
        self, doc: Doc, entity_types: Optional[list[str]] = None
    ) -> DocEntities:
        entity_set: DocEntities = [(span.text, span.label_, None) for span in doc.ents]

        # basic filtering, character removal, lemmatization
        normalized = sanitize_entities(entity_set)

        # filter by entity types, if provided
        if entity_types:
            return [e for e in normalized if e[1] in entity_types]

        return normalized

    def __link(self, entities: DocEntities) -> DocEntities:
        """
        Link entities for a single doc
        Args:
            entities (DocEntities): Entities to link
        """
        if self.linker is None:
            logging.info("Lazy-loading linker...")
            self.linker = TermLinker()

        linked_entity_map = dict(self.linker([tup[0] for tup in entities]))

        # canonicalization, synonymization
        linked = [(e[0], e[1], linked_entity_map.get(e[0])) for e in entities]

        return linked

    def __normalize_and_maybe_link(
        self, doc: Doc, link: bool = True, entity_types: Optional[list[str]] = None
    ) -> DocEntities:
        """
        Normalize and maybe link entities for a single doc

        Args:
            doc (Doc): SpaCy doc
            link (bool, optional): Whether to link entities. Defaults to True.
            entity_types (list[str], optional): Entity types to filter by. Defaults to None.
        """
        normalized = self.__normalize(doc, entity_types)
        if link:
            return self.__link(normalized)
        return normalized

    def __prep_doc(self, content: list[str]) -> list[str]:
        """
        Prepares a list of content for NER
        (only if use_llm is True)
        """
        _content = content.copy()

        # if use_llm, no tokenization
        if self.use_llm:
            if self.content_type == "html":
                _content = [extract_text(c) for c in _content]

            # chunk it up (spacy-llm doesn't use langchain for chaining, i guess?)
            _content = flatten(chunk_list(_content, CHUNK_SIZE))

        return _content

    def extract(
        self,
        content: list[str],
        link: bool = True,
        entity_types: Optional[list[str]] = None,
    ) -> list[DocEntities]:
        """
        Extract named entities from a list of content
        - basic SpaCy pipeline
        - applies rule_sets
        - normalizes terms

        Args:
            content (list[str]): list of content on which to do NER
            link (bool, optional): whether to link entities. Defaults to True.
            entity_types (Optional[list[str]], optional): filter by entity types. Defaults to None (all types permitted)

        Examples:
            >>> tagger.extract(["SMALL MOLECULE INHIBITORS OF NF-kB INDUCING KINASE"])
            >>> tagger.extract(["Interferon alpha and omega antibody antagonists"])
            >>> tagger.extract(["Inhibitors of beta secretase"], link=False)
            >>> tagger.extract(["This patent is about novel anti-ab monoclonal antibodies"], link=False)
        """
        if not self.nlp:
            raise Exception("NER tagger not initialized")

        if not isinstance(content, list):
            raise Exception("Content must be a list")

        logging.info("Starting NER pipeline with %s docs", len(content))

        steps = [
            self.__prep_doc,
            self.nlp.pipe,
            # TODO: linking would be faster if done in batch
            lambda docs: [
                self.__normalize_and_maybe_link(doc, link, entity_types) for doc in docs
            ],
        ]
        ents_by_doc = reduce(lambda x, func: func(x), steps, content.copy())

        logging.info("Entities found: %s", ents_by_doc)

        return ents_by_doc  # type: ignore

    def __call__(self, *args: Any, **kwds: Any) -> Any:
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
