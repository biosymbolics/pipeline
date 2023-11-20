from typing import Sequence
import logging

from clients.umls.graph import UmlsGraph
from constants.umls import (
    UMLS_CUI_SUPPRESSIONS,
    UMLS_NAME_SUPPRESSIONS,
    BIOMEDICAL_GRAPH_UMLS_TYPES,
)
from utils.classes import overrides

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

BETWEENNESS_FILE = "umls_betweenness.json"


class AncestorUmlsGraph(UmlsGraph):
    """
    Extends abstract UmlsGraph class, to make this suitable for ancestor selection
    """

    @staticmethod
    def _format_name_sql(table: str, suppressions: Sequence[str] | set[str]) -> str:
        name_filter = (
            " ".join([rf"and not {table}.str ~* '\y{s}\y'" for s in suppressions])
            if len(suppressions) > 0
            else ""
        )
        lang_filter = f"""
            and {table}.lat='ENG' -- english
            and {table}.ts='P' -- preferred terms
            and {table}.ispref='Y' -- preferred term
        """
        return name_filter + lang_filter

    @overrides(UmlsGraph)
    def edge_query(
        self, suppressions: Sequence[str] | set[str] = UMLS_NAME_SUPPRESSIONS
    ) -> str:
        """
        Query edges from umls
        - non-null hierarchy (good idea??)
        - only ancestor relationships RN, RB, CHD, PAR (narrower, broader, child, parent)
        - only isa/inverse_isa/mapped_from/mapped_to or no relationship
        - limits types to biomedical
        - applies some naming restrictions (via 'suppressions')
        - suppresses entities whose name is also a type (indicates overly general)
        """
        head_name_sql = AncestorUmlsGraph._format_name_sql("head_entity", suppressions)
        tail_name_sql = AncestorUmlsGraph._format_name_sql("tail_entity", suppressions)
        name_sql = head_name_sql + "\n" + tail_name_sql
        return f"""
            SELECT cui1 as head, cui2 as tail
            FROM mrrel as relationship,
            mrhier as hierarchy,
            mrsty as head_semantic_type,
            mrsty as tail_semantic_type,
            mrconso as head_entity,
            mrconso as tail_entity
            where head_entity.cui = cui1
            and tail_entity.cui = cui2
            and relationship.cui1 = head_semantic_type.cui
            and relationship.cui2 = tail_semantic_type.cui
            and hierarchy.cui = relationship.cui1
            and hierarchy.ptr is not null  -- suppress entities wo parent (otherwise overly general)
            and relationship.rel in ('RN', 'RB', 'CHD', 'PAR')  -- narrower, broader, child, parent
            and (
                relationship.rela is null
                OR relationship.rela in (
                    'isa', 'inverse_isa', 'mapped_from', 'mapped_to'
                )
            )
            and head_semantic_type.tui in {tuple(BIOMEDICAL_GRAPH_UMLS_TYPES.keys())}
            and tail_semantic_type.tui in {tuple(BIOMEDICAL_GRAPH_UMLS_TYPES.keys())}
            and ts_lexize('english_stem', head_semantic_type.sty) <> ts_lexize('english_stem', head_entity.str)  -- exclude entities with a name that is also the type
            and ts_lexize('english_stem', tail_semantic_type.sty) <> ts_lexize('english_stem', tail_entity.str)
            and cui1 not in {tuple(UMLS_CUI_SUPPRESSIONS.keys())}
            and cui2 not in {tuple(UMLS_CUI_SUPPRESSIONS.keys())}
            {name_sql}  -- applies lang and name filters
            group by cui1, cui2
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
