import time
from typing import Sequence
import logging
from pydash import omit

from data.domain.biomedical.umls import clean_umls_name
from typings.umls import OntologyLevel, UmlsRecord

from .ancestor_selection import AncestorUmlsGraph
from .constants import MAX_DENORMALIZED_ANCESTORS


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class UmlsTransformer:
    """
    Class for transforming UMLS records
    """

    def __init__(self, aui_lookup: dict[str, str]):
        # aui -> cui lookup (since hierarchy is stored as aui)
        self.aui_lookup: dict[str, str] = aui_lookup
        self.lookup_dict: dict[str, UmlsRecord] | None = None

        # graph w/ betweenness centrality to find ancestors
        self.umls_graph = AncestorUmlsGraph()

    def _create_lookup(self, records: Sequence[UmlsRecord]) -> dict[str, UmlsRecord]:
        def _enrich(r: UmlsRecord) -> UmlsRecord:
            """
            Enrich UMLS record with ancestor info, level and preferred name
            """
            if r.hierarchy is not None:
                # reverse to get nearest ancestor first
                ancestors = r.hierarchy.split(".")[::-1]
            else:
                ancestors = []

            ancestor_cuis = [self.aui_lookup.get(aui, "") for aui in ancestors]
            level = OntologyLevel.find(r.id, self.umls_graph.get_umls_centrality)
            preferred_name = clean_umls_name(r.id, r.canonical_name, r.synonyms, False)

            return UmlsRecord(
                **{
                    **r,
                    **{
                        f"l{i}_ancestor": ancestor_cuis[i]
                        if i < len(ancestor_cuis)
                        else None
                        for i in range(MAX_DENORMALIZED_ANCESTORS)
                    },
                    "level": level,
                    "preferred_name": preferred_name,
                }
            )

        start = time.monotonic()
        logger.info("Initializing UMLS lookup dict with %s records", len(records))
        lookup_records = [_enrich(r) for r in records]
        lookup_map = {r["id"]: r for r in lookup_records}
        logger.info(
            "Finished initializing UMLS lookup dict in %s",
            round(time.monotonic() - start, 2),
        )

        return lookup_map

    @staticmethod
    def find_level_ancestor(
        record: UmlsRecord,
        levels: Sequence[OntologyLevel],
        ancestors: tuple[UmlsRecord, ...],
    ) -> str:
        """
        Find first ancestor at the specified level

        Args:
            record (UmlsRecord): UMLS record
            level (OntologyLevel): level to find
            ancestors (tuple[UmlsRecord]): ordered list of ancestors

        Returns (str): ancestor id, or "" if none found
        """
        level_ancestors = [a for a in ancestors if a["level"] in levels]

        if len(level_ancestors) == 0:
            # return self as instance ancestor if no ancestors
            if record["level"] in levels:
                return record["id"]

            return ""

        # for instance level, use the last "INSTANCE" ancestor
        if OntologyLevel.INSTANCE in levels:
            return level_ancestors[-1]["id"]

        # else, use self as ancestor if matching or exceeding level
        if all(record["level"] >= level for level in levels):
            return record["id"]

        # otherwise, use the first matching ancestor
        return level_ancestors[0]["id"]

    def transform_record(self, r: UmlsRecord) -> UmlsRecord:
        """
        Transform a single UMLS record, intended to be persisted
        """
        if self.lookup_dict is None:
            raise ValueError("Lookup dict is not initialized")

        cuis = [r[f"l{i}_ancestor"] for i in range(MAX_DENORMALIZED_ANCESTORS)]
        ancestors = tuple(
            [self.lookup_dict[cui] for cui in cuis if cui in self.lookup_dict]
        )
        return UmlsRecord(
            **omit(r.__dict__, ["instance_rollup", "category_rollup"]),
            instance_rollup=UmlsTransformer.find_level_ancestor(
                r, [OntologyLevel.INSTANCE], ancestors
            ),
            category_rollup=UmlsTransformer.find_level_ancestor(
                r, [OntologyLevel.L1_CATEGORY, OntologyLevel.L2_CATEGORY], ancestors
            ),
        )

    def __call__(
        self,
        batch: Sequence[dict],
        all_records: Sequence[dict],
    ) -> list[UmlsRecord]:
        """
        Transform umls relationship

        Args:
            batch (Sequence[dict]): batch of records to transform
            all_records (Sequence[dict]): all UMLs records
        """
        if self.lookup_dict is None:
            as_umls_records = [UmlsRecord(**r) for r in all_records]
            self.lookup_dict = self._create_lookup(as_umls_records)
            assert self.lookup_dict is not None

        batch_records = [self.lookup_dict[r["id"]] for r in batch]

        return [self.transform_record(r) for r in batch_records]
