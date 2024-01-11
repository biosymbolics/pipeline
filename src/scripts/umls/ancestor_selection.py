from typing import Sequence
import logging

from clients.umls.graph import BETWEENNESS_FILE, UmlsGraph
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


class AncestorUmlsGraph(UmlsGraph):
    """
    Extends abstract UmlsGraph class, to make this suitable for ancestor selection
    """

    def __init__(
        self, doc_type: DocType = DocType.patent, file_name: str = BETWEENNESS_FILE
    ):
        super().__init__(file_name)
        self.doc_type = doc_type

    @staticmethod
    async def create(
        doc_type: DocType = DocType.patent, file_name: str = BETWEENNESS_FILE
    ) -> "AncestorUmlsGraph":
        """
        Factory for AncestorUmlsGraph
        """
        aug = AncestorUmlsGraph(doc_type, file_name)
        await aug.load()
        return aug

    @overrides(UmlsGraph)
    def edge_query(self, suppressions: Sequence[str] = UMLS_NAME_SUPPRESSIONS) -> str:
        """
        Query edges from umls
        - non-null hierarchy (good idea??)
        - only ancestor relationships RN, RB, CHD, PAR (narrower, broader, child, parent)
        - only isa/inverse_isa/mapped_from/mapped_to or no relationship
        - limits types to biomedical
        - applies some naming restrictions (via 'suppressions')
        - suppresses entities whose name is also a type (indicates overly general)

        Query took 4.51 minutes (depth < 3)
        """
        return rf"""
            -- patent to umls edges
            WITH RECURSIVE working_terms AS (
                -- Start with UMLS terms associated directly to patents
                SELECT
                    {self.doc_type.name}_id as head,
                    etu."B" as tail, -- B is cui
                    1 AS depth
                FROM biomedical_entity, intervenable, _entity_to_umls AS etu
                WHERE biomedical_entity.id=intervenable.entity_id
                AND etu."A"=intervenable.id

                UNION

                SELECT
                    {self.doc_type.name}_id as head,
                    etu."B" as tail, -- B is cui
                    1 AS depth
                FROM biomedical_entity, indicatable, _entity_to_umls AS etu
                WHERE biomedical_entity.id=indicatable.entity_id
                AND etu."A"=indicatable.id

                UNION

                SELECT
                    head_id as head,
                    tail_id as tail,
                    ut.depth + 1
                FROM umls_graph
                JOIN working_terms ut ON ut.tail = head_id
                JOIN umls as head_entity on head_entity.id = umls_graph.head_id
                JOIN umls as tail_entity on tail_entity.id = umls_graph.tail_id
                where ut.depth < 3
                and (
                    relationship is null
                    OR relationship in (
                        'isa', 'inverse_isa', 'mapped_from', 'mapped_to'
                    )
                )
                and head_entity.type_ids && ARRAY{list(MOST_PREFERRED_UMLS_TYPES.keys())}
                and tail_entity.type_ids && ARRAY{list(MOST_PREFERRED_UMLS_TYPES.keys())}
                and head_id not in {tuple(UMLS_CUI_SUPPRESSIONS.keys())}
                and tail_id not in {tuple(UMLS_CUI_SUPPRESSIONS.keys())}
                and ts_lexize('english_stem', head_entity.type_names[1]) <> ts_lexize('english_stem', head_name)  -- exclude entities with a name that is also the type
                and ts_lexize('english_stem', tail_entity.type_names[1]) <> ts_lexize('english_stem', tail_name)
                and not head_entity.name ~* '\y{get_or_re(suppressions, permit_plural=False)}\y'
                and not tail_entity.name ~* '\y{get_or_re(suppressions, permit_plural=False)}\y'
            )
            SELECT DISTINCT head, tail
            FROM working_terms;
            """

    def get_umls_centrality(self, id: str) -> float:
        """
        Get centrality for UMLS id

        Returns betweeness centrality if calculated (only top k nodes)
        Else, return 0 if in graph
        Else, return -1 (which will be the case for entities we excluded from ancestor consideration)
        """
        if id in self.betweenness_map:
            return self.betweenness_map[id]

        elif id in self.nodes:
            return 0

        return -1
