"""
Named-entity recognition using spacy

No hardware acceleration: see https://github.com/explosion/spaCy/issues/10783#issuecomment-1132523032
"""
from functools import reduce
from itertools import groupby
import time
from typing import Any, Literal, Optional, Sequence, TypeVar
from pydash import flatten
import logging
import warnings
import html
import spacy
from spacy.tokens import Span, Doc
from core.ner.cleaning import CleanFunction
from core.ner.normalizer import TermNormalizer
from typings.core import is_string_list, is_string_list_list

from utils.args import make_hashable
from utils.model import get_model_path
from utils.re import remove_extra_spaces
from utils.string import chunk_list

from .binder import BinderNlp
from .patterns import (
    INDICATION_SPACY_PATTERNS,
    INTERVENTION_SPACY_PATTERNS,
    MECHANISM_SPACY_PATTERNS,
)
from .types import DocEntities, DocEntity, SpacyPatterns
from .utils import spans_to_doc_entities

T = TypeVar("T", bound=Span | str)

ContentType = Literal["text", "html"]
CHUNK_SIZE = 10000


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NerTagger:
    """
    Named-entity recognition using spacy and other means
    """

    _instances: dict[str, Any] = {}

    def __init__(
        self,
        use_llm: Optional[bool] = False,
        llm_config: Optional[str] = "configs/patents/config.cfg",
        model: Optional[str] = "binder.pt",  # ignored if use_llm is True
        entity_types: Optional[frozenset[str]] = None,
        rule_sets: list[SpacyPatterns] = list(
            [
                INDICATION_SPACY_PATTERNS,
                INTERVENTION_SPACY_PATTERNS,
                MECHANISM_SPACY_PATTERNS,
            ]
        ),
        additional_cleaners: list[CleanFunction] = [],
        additional_removal_terms: list[str] = [],
        link: bool = True,
    ):
        """
        Named-entity recognition using spacy

        NOTE: if using binder, requires binder model class be in PYTHONPATH. (TODO: fix this)

        Args:
            use_llm (Optional[bool], optional): Use LLM model. Defaults to False. If true, no rules or anything are used.
            llm_config (Optional[str], optional): LLM config file. Defaults to "configs/patents/config.cfg".
            model (str, optional): torch NER model. Defaults to "binder.pt".
            rule_sets (Optional[list[SpacyPatterns]], optional): SpaCy patterns. Defaults to None.
            additional_cleaners (list[Callable[[Sequence[str]], Sequence[str]]], optional): Additional cleaners funs. Defaults to [].
            link (bool, optional): Whether to link entities. Defaults to True.
        """
        # prefer_gpu()
        # set_gpu_allocator("pytorch")

        self.model = model
        self.use_llm = use_llm
        self.llm_config = llm_config
        self.rule_sets = rule_sets
        self.additional_cleaners = additional_cleaners
        self.entity_types = entity_types
        self.normalizer = TermNormalizer(link, additional_cleaners)
        start_time = time.time()

        if entity_types is not None and not isinstance(entity_types, frozenset):
            raise ValueError("entity_types must be a frozenset")

        if self.use_llm:
            # lazy imports / inits
            import spacy_llm
            from spacy_llm.util import assemble

            spacy_llm.logger.addHandler(logging.StreamHandler())
            spacy_llm.logger.setLevel(logging.INFO)
            warnings.filterwarnings(
                "ignore", category=UserWarning, module="torch.amp.autocast_mode"
            )

            if not self.llm_config:
                raise ValueError("Must provide llm_config if use_llm is True")
            self.nlp = assemble(self.llm_config)

        elif self.model:
            if not self.model.endswith(".pt"):
                raise ValueError("Model must be torch")

            model_filename = get_model_path(self.model)
            self.nlp = BinderNlp(model_filename)

            if len(self.rule_sets) > 0:
                logger.info("Adding rule sets to NER pipeline")
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

    def __prep_for_extract(self, content: Sequence[str]) -> Sequence[str]:
        """
        Prepares a list of content for NER
        """
        steps = [
            lambda c: flatten(chunk_list(c, CHUNK_SIZE)) if self.use_llm else c,
            lambda _content: [html.unescape(c) for c in _content],
            remove_extra_spaces,  # important; model gets confused by weird spacing
        ]

        return list(reduce(lambda c, f: f(c), steps, content))  # type: ignore

    @staticmethod
    def __combine_ents(doc1: Doc, doc2: Doc) -> DocEntities:
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

    def __dual_model_extract(self, content: Sequence[str]) -> list[DocEntities]:
        """
        Run both NLP pipelines (rule_nlp and binder)

        To avoid reconstructing the SpaCy doc, just return DocEntities
        """

        if is_string_list_list(content):
            ents = flatten([self.__dual_model_extract(c) for c in content])
            return ents
        elif is_string_list(content):
            binder_docs = self.nlp.pipe(content)
            if self.rule_nlp:
                rule_docs = self.rule_nlp.pipe(content)
                return [
                    self.__combine_ents(d1, d2)
                    for d1, d2 in zip(binder_docs, rule_docs)
                ]
            return [spans_to_doc_entities(doc.ents) for doc in binder_docs]
        else:
            raise Exception("Bad content type")

    def __normalize(
        self,
        entity_sets: list[DocEntities],
    ) -> list[DocEntities]:
        """
        Normalize entity set

        Args:
            entity_set (DocEntities): Entities to normalize
        """
        terms: list[str] = [e[0] for e in flatten(entity_sets)]
        normalization_map = dict(self.normalizer.normalize(terms))

        def get_doc_entity(e: DocEntity) -> DocEntity:
            linked_entity = normalization_map.get(e.term)
            return DocEntity(
                *e[0:4],
                normalized_term=linked_entity.name if linked_entity else e.term,
                linked_entity=linked_entity,
            )

        # filter by entity types (if provided) and remove empty chars
        norm_entity_sets = [
            [
                get_doc_entity(e)
                for e in es
                if len(e[0]) > 0
                and ((self.entity_types is None) or (e[1] in self.entity_types))
            ]
            for es in entity_sets
        ]

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

        prepped_content = self.__prep_for_extract(content)
        entity_sets = self.__dual_model_extract(prepped_content)
        norm_entity_sets = self.__normalize(entity_sets)

        if len(norm_entity_sets) != len(entity_sets):
            raise ValueError("Normalization changed number of entities")

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

        def as_strings(entities: DocEntities) -> list[str]:
            return [as_string(e) for e in entities]

        ents_by_doc = self.extract(content, **kwargs)
        map = {orig: as_strings(v) for orig, v in zip(content, ents_by_doc)}

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
