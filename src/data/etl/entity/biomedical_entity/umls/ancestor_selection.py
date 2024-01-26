from typing import Sequence
import logging
from prisma.enums import OntologyLevel

from clients.low_level.prisma import prisma_client
from clients.umls.graph import UmlsGraph
from clients.umls.types import EdgeRecord, NodeRecord
from constants.umls import (
    MOST_PREFERRED_UMLS_TYPES,
    UMLS_CUI_SUPPRESSIONS,
    UMLS_NAME_SUPPRESSIONS,
)
from typings.documents.common import DocType
from utils.classes import overrides
from utils.re import get_or_re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_DEPTH = 2
DEFAULT_UMLS_TO_UMLS_RELATIONSHIPS = (
    "isa",
    "inverse_isa",
    "mapped_from",
    "mapped_to",
)


class AncestorUmlsGraph(UmlsGraph):
    """
    Extends abstract UmlsGraph class, to make this suitable for ancestor selection
    """

    def __init__(self, doc_type: DocType = DocType.patent):
        super().__init__()
        self.doc_type = doc_type

    @staticmethod
    async def create(doc_type: DocType = DocType.patent) -> "AncestorUmlsGraph":
        """
        Factory for AncestorUmlsGraph
        """
        aug = AncestorUmlsGraph(doc_type)
        await aug.load()
        return aug

    @overrides(UmlsGraph)
    async def get_nodes(
        self,
        considered_tuis: list[str] = list(MOST_PREFERRED_UMLS_TYPES.keys()),
    ) -> list[NodeRecord]:
        """
        Query nodes from umls
        """
        query = rf"""
            SELECT umls.id as id, count(*) as count
            FROM biomedical_entity, intervenable, _entity_to_umls AS etu, umls
            WHERE biomedical_entity.id=intervenable.entity_id
            AND etu."A"=intervenable.id
            AND umls.id=etu."B"
            AND umls.type_ids && $1
            GROUP BY umls.id

            UNION

            -- indication-mapped UMLS terms associated with docs, if preferred type
            SELECT umls.id as id, count(*) as count
            FROM biomedical_entity, indicatable, _entity_to_umls AS etu, umls
            WHERE biomedical_entity.id=indicatable.entity_id
            AND etu."A"=indicatable.id
            AND umls.id=etu."B"
            AND umls.type_ids && $1
            GROUP BY umls.id
            """

        client = await prisma_client(300)
        results = await client.query_raw(query, considered_tuis)
        return [NodeRecord(**r) for r in results]

    @overrides(UmlsGraph)
    async def get_edges(
        self,
        considered_relationships: tuple[str, ...] = DEFAULT_UMLS_TO_UMLS_RELATIONSHIPS,
        considered_tuis: tuple[str, ...] = tuple(MOST_PREFERRED_UMLS_TYPES.keys()),
        name_suppressions: Sequence[str] = UMLS_NAME_SUPPRESSIONS,
        cui_suppressions: Sequence[str] = tuple(UMLS_CUI_SUPPRESSIONS.keys()),
    ) -> list[EdgeRecord]:
        """
        Query edges from umls
        - non-null hierarchy (good idea??)
        - only isa/inverse_isa/mapped_from/mapped_to or no relationship
        - limits types to biomedical
        - applies some naming restrictions (via 'suppressions')
        - suppresses entities whose name is also a type (indicates overly general)

        Query took 4.51 minutes (depth < 3)
        """
        # head == parent, tail == child
        query = rf"""
            -- patent to umls edges
            WITH RECURSIVE working_terms AS (
                SELECT
                    distinct umls.id as head,
                    'leaf' as tail,
                    1 AS depth
                FROM biomedical_entity, intervenable, _entity_to_umls AS etu, umls
                WHERE biomedical_entity.id=intervenable.entity_id
                AND etu."A"=intervenable.id
                AND umls.id=etu."B"
                AND umls.type_ids && $1

                UNION

                -- indication-mapped UMLS terms associated with docs, if preferred type
                SELECT
                    distinct umls.id as head,
                    'leaf' as tail,
                    1 AS depth
                FROM biomedical_entity, indicatable, _entity_to_umls AS etu, umls
                WHERE biomedical_entity.id=indicatable.entity_id
                AND etu."A"=indicatable.id
                AND umls.id=etu."B"
                AND umls.type_ids && $1

                UNION

                -- UMLS to UMLS relationships
                -- e.g. C12345 & C67890
                SELECT
                    head_id as head,
                    tail_id as tail,
                    ut.depth + 1
                FROM umls_graph
                JOIN working_terms ut ON ut.tail = head_id
                JOIN umls as head_entity on head_entity.id = umls_graph.head_id
                JOIN umls as tail_entity on tail_entity.id = umls_graph.tail_id
                where ut.depth <= {MAX_DEPTH}
                and (
                    relationship is null
                    OR relationship in {considered_relationships}
                )
                and head_entity.type_ids && $1
                and tail_entity.type_ids && $1
                and head_id not in {cui_suppressions}
                and tail_id not in {cui_suppressions}
                and ts_lexize('english_stem', head_entity.type_names[1]) <> ts_lexize('english_stem', head_name)  -- exclude entities with a name that is also the type
                and ts_lexize('english_stem', tail_entity.type_names[1]) <> ts_lexize('english_stem', tail_name)
                and not head_entity.name ~* '\y{get_or_re(name_suppressions, permit_plural=False)}\y'
                and not tail_entity.name ~* '\y{get_or_re(name_suppressions, permit_plural=False)}\y'
            )
            SELECT DISTINCT head, tail
            FROM working_terms;
            """

        client = await prisma_client(300)
        results = await client.query_raw(query, considered_tuis)
        return [EdgeRecord(**r) for r in results]

    def get_ontology_level(self, cui: str) -> OntologyLevel:
        """
        Get ontology level for cui
        """
        if not cui in self.nodes:
            return OntologyLevel.UNKNOWN

        node = NodeRecord(**self.nodes[cui])

        return node.level
