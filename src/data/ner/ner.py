"""
Named-entity recognition using spacy

No hardware acceleration: see https://github.com/explosion/spaCy/issues/10783#issuecomment-1132523032
"""
from functools import reduce
from itertools import groupby
import time
from typing import Any, Literal, Optional, TypeVar, Union
from pydash import flatten
import logging
import warnings
import html
import spacy
from spacy.tokens import Span, Doc
import spacy_llm
from spacy_llm.util import assemble

from utils.extraction.html import extract_text
from utils.re import remove_extra_spaces
from utils.string import chunk_list

from .binder import BinderNlp
from .cleaning import EntityCleaner
from .linker import TermLinker
from .patterns import (
    INDICATION_SPACY_PATTERNS,
    INTERVENTION_SPACY_PATTERNS,
    MECHANISM_SPACY_PATTERNS,
)
from .types import DocEntities, DocEntity, SpacyPatterns
from .utils import spans_to_doc_entities

T = TypeVar("T", bound=Union[Span, str])
ContentType = Literal["text", "html"]
CHUNK_SIZE = 10000

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.amp.autocast_mode"
)
spacy_llm.logger.addHandler(logging.StreamHandler())
spacy_llm.logger.setLevel(logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NerTagger:
    """
    Named-entity recognition using spacy and other means
    """

    _instances: dict[tuple, Any] = {}

    def __init__(
        self,
        use_llm: Optional[bool] = False,
        llm_config: Optional[str] = "configs/patents/config.cfg",
        model: Optional[str] = "model.pt",  # ignored if use_llm is True
        content_type: Optional[ContentType] = "text",
        entity_types: Optional[frozenset[str]] = None,
        rule_sets: list[SpacyPatterns] = list(
            [
                INDICATION_SPACY_PATTERNS,
                INTERVENTION_SPACY_PATTERNS,
                MECHANISM_SPACY_PATTERNS,
            ]
        ),
        parallelize: bool = True,
    ):
        """
        Named-entity recognition using spacy

        Args:
            use_llm (Optional[bool], optional): Use LLM model. Defaults to False. If true, no rules or anything are used.
            llm_config (Optional[str], optional): LLM config file. Defaults to "configs/patents/config.cfg".
            model (str, optional): torch NER model. Defaults to "model.pt".
            content_type (Optional[ContentType], optional): Content type. Defaults to "text".
            rule_sets (Optional[list[SpacyPatterns]], optional): SpaCy patterns. Defaults to None.
            parallelize (bool, optional): Parallelize. Defaults to True. Even if true, will only parallelize for sufficiently large batches.
        """
        # prefer_gpu()
        # set_gpu_allocator("pytorch")

        self.model = model
        self.use_llm = use_llm
        self.content_type = content_type
        self.llm_config = llm_config
        self.rule_sets = rule_sets
        self.entity_types = entity_types
        self.linker: Optional[TermLinker] = None  # lazy initialization
        self.cleaner = EntityCleaner(parallelize=parallelize)
        start_time = time.time()

        if entity_types is not None and not isinstance(entity_types, frozenset):
            raise ValueError("entity_types must be a frozenset")

        if self.use_llm:
            if not self.llm_config:
                raise ValueError("Must provide llm_config if use_llm is True")
            self.nlp = assemble(self.llm_config)

        elif self.model:
            if not self.model.endswith(".pt"):
                raise ValueError("Model must be torch")
            self.nlp = BinderNlp(self.model)

            if len(self.rule_sets) > 0:
                # rules catch a few things the binder model misses
                rule_nlp = spacy.load("en_core_sci_lg")
                rule_nlp.add_pipe("merge_entities", after="ner")
                ruler = rule_nlp.add_pipe(
                    "entity_ruler",
                    config={"validate": True, "overwrite_ents": True},
                    after="merge_entities",
                )
                for rules in self.rule_sets:
                    ruler.add_patterns(rules)  # type: ignore

                self.rule_nlp = rule_nlp
            else:
                self.rule_nlp = None
        else:
            raise ValueError("Must provide either use_llm or model")

        logger.info(
            "Init NER pipeline took %s seconds",
            round(time.time() - start_time, 2),
        )

    def __link(self, entities: DocEntities) -> DocEntities:
        """
        Link entities for a single doc
        Args:
            entities (DocEntities): Entities to link
        """
        if self.linker is None:
            logger.info("Lazy-loading linker...")
            self.linker = TermLinker()

        linked_entity_map = dict(self.linker([tup[0] for tup in entities]))

        # canonicalization, synonymization
        linked = [DocEntity(*e[0:5], linked_entity_map.get(e[0])) for e in entities]

        return linked

    @staticmethod
    def __get_string_entity(entity: DocEntity) -> str:
        """
        Get the string representation of an entity
        (linked name otherwise normalized term otherwise term)
        """
        if entity.linked_entity:
            return entity.linked_entity.name
        return entity.normalized_term or entity.term

    @staticmethod
    def __get_string_entities(entities: DocEntities) -> list[str]:
        """
        Get the string representation of a list of entities
        """
        return [NerTagger.__get_string_entity(entity) for entity in entities]

    def __prep_for_extract(self, content: list[str]) -> list[str]:
        """
        Prepares a list of content for NER
        """
        steps = [
            lambda string: [extract_text(s) for s in string]
            if self.content_type == "html"
            else string,
            lambda string: flatten(chunk_list(string, CHUNK_SIZE))
            if self.use_llm
            else string,
            lambda string: [html.unescape(s) for s in string],
            remove_extra_spaces,  # important; model gets confused by weird spacing
        ]

        _content = list(reduce(lambda c, f: f(c), steps, content))

        return _content

    @staticmethod
    def __combine_ents(doc1: Doc, doc2: Doc) -> DocEntities:
        """
        Extract entities from two docs (produced by two different nlp pipelines)
        and combine them.

        - If the same entity is found in both docs, use the longest one.
        - Hacks: entit
        """
        # Create a new list combining entities from both docs
        all_ents = [*doc1.ents, *doc2.ents]

        indices = sorted(
            [(ent.start_char, ent.end_char, ent) for ent in all_ents],
            key=lambda k: (k[0], k[1]),
            reverse=True,
        )

        grouped_ind = groupby(indices, key=lambda i: i[0])
        # due to the sort order, the longest entity will be first (e.g. [(1, 3), (1, 1)])
        deduped = [list(g)[0][2] for _, g in grouped_ind]

        # turn into doc entities
        entity_set = spans_to_doc_entities(sorted(deduped, key=lambda e: e.start_char))
        return entity_set

    def __extract(self, content: list[str]) -> list[DocEntities]:
        """
        Run both NLP pipelines (rule_nlp and binder)

        To avoid reconstructing the SpaCy doc, just return DocEntities
        """
        binder_docs = self.nlp.pipe(content)
        if self.rule_nlp:
            rule_docs = self.rule_nlp.pipe(content)
            return [
                self.__combine_ents(d1, d2) for d1, d2 in zip(binder_docs, rule_docs)
            ]
        return [spans_to_doc_entities(doc.ents) for doc in binder_docs]

    def __normalize(
        self,
        entity_set: DocEntities,
        link: bool = True,
    ) -> DocEntities:
        """
        Normalize entity set

        Args:
            entity_set (DocEntities): Entities to normalize
            link (bool, optional): Whether to link entities. Defaults to True.
        """
        normalized = self.cleaner.clean(entity_set, remove_suppressed=True)

        # filter by entity types, if provided
        if self.entity_types:
            normalized = [e for e in normalized if e[1] in self.entity_types]

        if link and len(normalized) > 0:
            return self.__link(normalized)

        return normalized

    def extract(
        self,
        content: list[str],
        link: bool = True,
    ) -> list[DocEntities]:
        """
        Extract named entities from a list of content
        - basic SpaCy pipeline
        - applies rule_sets
        - normalizes terms

        Note: for bulk processing, linking is better done in a separate step, batched

        Args:
            content (list[str]): list of content on which to do NER
            link (bool, optional): whether to link entities. Defaults to True.

        Examples:
            >>> tagger.extract(["Inhibitors of beta secretase"], link=False)
            >>> tagger.extract(["This patent is about novel anti-ab monoclonal antibodies"], link=False)
            >>> tagger.extract(["commercialize biosimilar BAT1806, a anti-interleukin-6 (IL-6) receptor monoclonal antibody"], link=False)
            >>> tagger.extract(["mannose-1-phosphate guanylyltransferase (GDP) activity"], link=False)
            >>> tagger.extract(["zuranolone for treatment of major depression, of BIIB124 (SAGE-324) for essential tremor, and BIIB111 (timrepigene emparvovec)"], link=False)
        """
        if not self.nlp:
            raise Exception("NER tagger not initialized")

        if not isinstance(content, list):
            raise Exception("Content must be a list")

        start_time = time.time()
        logger.info("Starting NER pipeline with %s docs", len(content))

        prepped_content = self.__prep_for_extract(content.copy())
        entity_sets = self.__extract(prepped_content)
        normalized_entity_sets = [
            self.__normalize(e_set, link) for e_set in entity_sets
        ]

        logger.info(
            "Full entity extraction took %s seconds, yielded %s",
            round(time.time() - start_time, 2),
            normalized_entity_sets,
        )

        return normalized_entity_sets

    def extract_strings(self, content: list[str], **kwargs) -> list[list[str]]:
        """
        Extract named entities from a list of content, returning a list of strings
        """
        ents_by_doc = self.extract(content, **kwargs)
        return [self.__get_string_entities(e) for e in ents_by_doc]

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.extract(*args, **kwds)

    @classmethod
    def get_instance(cls, **kwargs) -> "NerTagger":
        # Convert kwargs to a hashable type
        kwargs_tuple = tuple(sorted(kwargs.items()))
        if kwargs_tuple not in cls._instances:
            logger.info("Creating new instance of %s", cls)
            cls._instances[kwargs_tuple] = cls(**kwargs)
        else:
            logger.info("Using existing instance of %s", cls)
        return cls._instances[kwargs_tuple]
