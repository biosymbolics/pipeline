"""
Named-entity recognition using spacy

No hardware acceleration: see https://github.com/explosion/spaCy/issues/10783#issuecomment-1132523032
"""
from functools import reduce
from itertools import groupby
import time
from typing import Any, Literal, Optional, Sequence, TypeVar
from pydash import compact
import logging
import html
from spacy.tokens import Span, Doc

from core.ner.cleaning import CleanFunction
from core.ner.normalizer import TermNormalizer
from core.ner.spacy import Spacy
from utils.args import make_hashable
from utils.model import get_model_path
from utils.re import remove_extra_spaces

from .binder import BinderNlp
from .patterns import (
    INDICATION_SPACY_PATTERNS,
    INTERVENTION_SPACY_PATTERNS,
    MECHANISM_SPACY_PATTERNS,
)
from .types import CanonicalEntity, DocEntities, DocEntity, SpacyPatterns
from .utils import spans_to_doc_entities

T = TypeVar("T", bound=Span | str)

ContentType = Literal["text", "html"]
CHUNK_SIZE = 10000


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NerTagger:
    """
    Named-entity recognition using spacy and other means

    TODO: add abbr resolution
    """

    _instances: dict[str, Any] = {}

    def __init__(
        self,
        model: str = "binder.pt",
        entity_types: Optional[frozenset[str]] = None,
        rule_sets: list[SpacyPatterns] = list(
            [
                INDICATION_SPACY_PATTERNS,
                INTERVENTION_SPACY_PATTERNS,
                MECHANISM_SPACY_PATTERNS,
            ]
        ),
        additional_cleaners: list[CleanFunction] = [],
        link: bool = True,
        normalize: bool = True,
    ):
        """
        Named-entity recognition using spacy

        NOTE: if using binder, requires binder model class be in PYTHONPATH. (TODO: fix this)

        Args:
            model (str, optional): torch NER model. Defaults to "binder.pt".
            rule_sets (Optional[list[SpacyPatterns]], optional): SpaCy patterns. Defaults to None.
            additional_cleaners (list[Callable[[Sequence[str]], Sequence[str]]], optional): Additional cleaners funs. Defaults to [].
            link (bool, optional): Whether to link entities. Defaults to True.
            normalize (bool, optional): Whether to normalize entities. Defaults to True.
        """
        start_time = time.time()

        self.model = model
        self.rule_sets = rule_sets
        self.entity_types = entity_types
        self.normalizer = (
            TermNormalizer(link, additional_cleaners) if normalize else None
        )

        if link and not normalize:
            raise ValueError("Cannot link entities without normalizing")

        if entity_types is not None and not isinstance(entity_types, frozenset):
            raise ValueError("entity_types must be a frozenset")

        if not normalize:
            logger.warning("Normalization is disabled")
        if not link:
            logger.warning("Linking is disabled")

        if not self.model.endswith(".pt"):
            raise ValueError("Model must be torch")

        model_filename = get_model_path(self.model)
        self.nlp = BinderNlp(model_filename)

        if len(self.rule_sets) > 0:
            logger.info("Adding rule sets to NER pipeline")
            # rules catch a few things the binder model misses
            rule_nlp = Spacy.get_instance(
                model="en_core_sci_lg",
                additional_pipelines={
                    "merge_entities": {"after": "ner"},
                    "entity_ruler": {
                        "config": {"validate": True, "overwrite_ents": True},
                        "after": "merge_entities",
                    },
                },
            )
            ruler = rule_nlp.get_pipe("entity_ruler")

            for rules in self.rule_sets:
                ruler.add_patterns(rules)  # type: ignore

            self.rule_nlp = rule_nlp
        else:
            self.rule_nlp = None

        logger.info(
            "Init NER pipeline took %s seconds",
            round(time.time() - start_time, 2),
        )

    def _prep_for_extract(self, content: Sequence[str]) -> Sequence[str]:
        """
        Prepares a list of content for NER
        """
        steps = [
            lambda _content: [html.unescape(c) for c in _content],
            remove_extra_spaces,
            # TODO: doing this for binder, which due to some off-by-one can't find ents at the start of a string
            lambda _content: [" " + c for c in _content],
        ]

        return list(reduce(lambda c, f: f(c), steps, content))  # type: ignore

    @staticmethod
    def _combine_ents(doc1: Doc, doc2: Doc) -> DocEntities:
        """
        Extract entities from two docs (produced by two different nlp pipelines)
        and combine them.

        - If the same entity is found in both docs, use the longest one.
        """
        indices = sorted(
            [(ent.start_char, ent.end_char, ent) for ent in [*doc1.ents, *doc2.ents]],
            key=lambda k: (k[0], k[1]),
            reverse=True,
        )

        grouped_ind = groupby(indices, key=lambda i: i[0])
        # due to the sort order, the longest entity will be first (e.g. [(1, 3), (1, 1)])
        deduped = [list(g)[0][2] for _, g in grouped_ind]

        # turn into doc entities
        entity_set = spans_to_doc_entities(sorted(deduped, key=lambda e: e.start_char))
        return entity_set

    def _dual_model_extract(self, content: Sequence[str]) -> list[DocEntities]:
        """
        Run both NLP pipelines (rule_nlp and binder)

        To avoid reconstructing the SpaCy doc, just return DocEntities
        """

        binder_docs = self.nlp.pipe(content)
        if self.rule_nlp:
            rule_docs = self.rule_nlp.pipe(content)
            return [
                self._combine_ents(d1, d2) for d1, d2 in zip(binder_docs, rule_docs)
            ]
        return [spans_to_doc_entities(doc.ents) for doc in binder_docs]

    def _normalize(
        self,
        entity_sets: list[DocEntities],
    ) -> list[DocEntities]:
        """
        Normalize entity set

        Args:
            entity_set (DocEntities): Entities to normalize
        """
        if not self.normalizer:
            logger.debug("Skipping normalization step")
            return entity_sets

        def get_doc_entity(
            e: DocEntity, norm_entity: CanonicalEntity
        ) -> DocEntity | None:
            if len(norm_entity.name) == 0:
                return None
            return DocEntity(
                *e[0:5],
                normalized_term=norm_entity.name,
                linked_entity=norm_entity,
            )

        def get_doc_entities(entity_set: Sequence[DocEntity]) -> list[DocEntity]:
            if not self.normalizer:
                return list(entity_set)
            normalizations = self.normalizer.normalize(entity_set)
            return compact(
                [
                    get_doc_entity(es, norm)
                    for es, norm in zip(entity_set, normalizations)
                    if len(es.term) > 0
                    and ((self.entity_types is None) or (es.type in self.entity_types))
                ]
            )

        # filter by entity types (if provided) and remove empty names
        norm_entity_sets = [get_doc_entities(es) for es in entity_sets]

        if len(norm_entity_sets) != len(entity_sets):
            raise ValueError("Normalization changed number of entities")

        return norm_entity_sets

    def extract(
        self,
        content: Sequence[str],
    ) -> Sequence[DocEntities]:
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
            >>> tagger.extract(["Inhibitors of beta secretase"])
            >>> tagger.extract(["This patent is about novel anti-ab monoclonal antibodies"])
            >>> tagger.extract(["commercialize biosimilar BAT1806, a anti-interleukin-6 (IL-6) receptor monoclonal antibody"])
            >>> tagger.extract(["mannose-1-phosphate guanylyltransferase (GDP) activity"])

        NOTE: As of 08/14/2023, start_chars and end_chars are not to be trusted in context with 2+ spaces or html-encoded chars
        """

        start_time = time.time()

        if not self.nlp:
            raise Exception("NER tagger not initialized")

        if not isinstance(content, list):
            raise Exception("Content must be a list")

        prepped_content = self._prep_for_extract(content)
        entity_sets = self._dual_model_extract(prepped_content)
        norm_entity_sets = self._normalize(entity_sets)

        logger.info(
            "Full entity extraction took %s seconds for %s docs, yielded %s",
            round(time.time() - start_time, 2),
            len(content),
            norm_entity_sets,
        )
        return norm_entity_sets

    def extract_string_map(self, content: list[str], **kwargs) -> dict[str, list[str]]:
        """
        Extract named entities from a list of content, returning a list of strings
        """

        def as_string(entity: DocEntity) -> str:
            """
            Get the string representation of an entity
            """
            if entity.linked_entity:
                return entity.linked_entity.name
            return entity.normalized_term or entity.term

        ents_by_doc = self.extract(content, **kwargs)
        map = {orig: [as_string(e) for e in v] for orig, v in zip(content, ents_by_doc)}

        return map

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.extract(*args, **kwds)

    @classmethod
    def get_instance(cls, **kwargs) -> "NerTagger":
        args = sorted(kwargs.items())
        args_hash = make_hashable(args)  # Convert args/kwargs to a hashable type
        if args_hash not in cls._instances:
            logger.info("Creating new instance of %s", cls)
            cls._instances[args_hash] = cls(**kwargs)
        else:
            logger.info("Using existing instance of %s", cls)
        return cls._instances[args_hash]
