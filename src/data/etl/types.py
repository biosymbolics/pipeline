from dataclasses import dataclass, field, fields
from typing import Callable, Literal, Sequence
from prisma.enums import BiomedicalEntityType, Source
from prisma.types import (
    BiomedicalEntityCreateManyNestedWithoutRelationsInput as BiomedicalEntityRelationInput,
)

from core.ner.cleaning import CleanFunction
from core.ner.linker.types import CandidateSelectorType
from core.ner.types import CanonicalEntity
from typings.core import Dataclass


@dataclass(frozen=True)
class RelationConnectInfo(Dataclass):
    source_field: str
    dest_field: Literal["canonical_id", "name", "term"]
    connect_type: Literal["connect", "create"]

    def _get_relation_val(
        self, val: str, canonical_map: dict[str, CanonicalEntity]
    ) -> str | None:
        """
        Helper to get value appropriate for dest_field

        Typically:
            - canonical_id if dest_field is canonical_id
            - val otherwise, which can be an id or something like "term" in the case of synonyms
        """
        if self.dest_field == "canonical_id":
            if val in canonical_map:
                return canonical_map[val].id
            return None
        return val

    def form_prisma_relation(
        self, source_rec: dict, canonical_map: dict[str, CanonicalEntity]
    ) -> BiomedicalEntityRelationInput:
        """
        Form prisma relations from source record

        Example:
            { "create": [{ "term": "foo" }] }
            { "connect": [{ "canonical_id": "foo" }] }
        """
        vals = [
            self._get_relation_val(val, canonical_map)
            for val in source_rec.get(self.source_field) or []
        ]
        if vals is None:
            return {}

        relation_inputs = [{self.dest_field: str(val)} for val in vals]

        if self.connect_type == "connect":
            return BiomedicalEntityRelationInput(connect=relation_inputs)  # type: ignore
        elif self.connect_type == "create":
            return BiomedicalEntityRelationInput(create=relation_inputs)  # type: ignore
        else:
            raise ValueError(f"Unknown connect_type: {self.connect_type}")


@dataclass(frozen=True)
class RelationIdFieldMap(Dataclass):
    def items(self) -> list[tuple[str, RelationConnectInfo]]:
        return [(field.name, getattr(self, field.name)) for field in fields(self)]

    comprised_of: RelationConnectInfo | None = None
    parents: RelationConnectInfo | None = None
    synonyms: RelationConnectInfo | None = None
    umls_entities: RelationConnectInfo | None = None


def default_get_source_map(records: Sequence[dict]) -> dict:
    """
    Default source map for records
    (mostly as an example since it isn't strongly typed)
    """
    return {
        rec["term"]: {
            "synonyms": [rec["term"]],
            "type": BiomedicalEntityType.OTHER,
        }
        for rec in records
    }


@dataclass(frozen=True)
class BiomedicalEntityLoadSpec(Dataclass):
    database: str
    candidate_selector: CandidateSelectorType
    sql: str
    # get_source_map must yield map of term -> fields, for at least all fields mentioned in relation_id_field_map
    get_source_map: Callable[[Sequence[dict]], dict] = default_get_source_map
    additional_cleaners: Sequence[CleanFunction] = field(default_factory=list)
    get_terms: Callable[[dict], Sequence[str]] = lambda sm: list(sm.keys())
    get_terms_to_canonicalize: Callable[[dict], Sequence[str]] = lambda sm: list(
        sm.keys()
    )
    non_canonical_source: Source = Source.BIOSYM
    # operates on source_map
    relation_id_field_map: RelationIdFieldMap = RelationIdFieldMap(
        synonyms=RelationConnectInfo(
            source_field="synonyms", dest_field="term", connect_type="create"
        ),
    )
