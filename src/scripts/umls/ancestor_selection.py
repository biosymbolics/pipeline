from typing import Sequence
import logging

from clients.umls.graph import UmlsGraph
from constants.umls import (
    MOST_PREFERRED_UMLS_TYPES,
    UMLS_CUI_SUPPRESSIONS,
    UMLS_NAME_SUPPRESSIONS,
)
from utils.classes import overrides
from utils.re import get_or_re

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AncestorUmlsGraph(UmlsGraph):
    """
    Extends abstract UmlsGraph class, to make this suitable for ancestor selection
    """

    def __init__(self):
        super().__init__()

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
        """
        return rf"""
            -- patent to umls edges
            WITH RECURSIVE working_terms AS (
                -- Start with UMLS terms associated directly to patents
                SELECT
                    publication_number as head,
                    term_ids.cid as tail,
                    1 AS depth
                FROM annotations, term_ids
                WHERE annotations.id=term_ids.id

                UNION

                SELECT
                    head_id as head,
                    tail_id as tail,
                    ut.depth + 1
                FROM umls_graph
                JOIN working_terms ut ON ut.tail = head_id
                JOIN umls_lookup as head_entity on head_entity.id = umls_graph.head_id
                JOIN umls_lookup as tail_entity on tail_entity.id = umls_graph.tail_id
                where ut.depth < 9
                and (
                    relationship is null
                    OR relationship in (
                        'isa', 'inverse_isa', 'mapped_from', 'mapped_to'
                    )
                )
                and head_entity.type_ids::text[] && ARRAY{list(MOST_PREFERRED_UMLS_TYPES.keys())}
                and tail_entity.type_ids::text[] && ARRAY{list(MOST_PREFERRED_UMLS_TYPES.keys())}
                and head_id not in {tuple(UMLS_CUI_SUPPRESSIONS.keys())}
                and tail_id not in {tuple(UMLS_CUI_SUPPRESSIONS.keys())}
                and not head_entity.canonical_name ~* '\y{get_or_re(suppressions, permit_plural=False)}\y'
                and not tail_entity.canonical_name ~* '\y{get_or_re(suppressions, permit_plural=False)}\y'
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
