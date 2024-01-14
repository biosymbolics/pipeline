from dataclasses import dataclass, fields
from typing import Callable, Literal, Sequence, TypedDict
from core.ner.types import CanonicalEntity
from typings.core import Dataclass
from prisma.types import (
    BiomedicalEntityCreateWithoutRelationsInput,
    OwnerCreateWithoutRelationsInput,
)
from prisma.enums import Source

from core.ner.linker.types import CandidateSelectorType
from core.ner.cleaning import CleanFunction


class BiomedicalEntityCreateInputWithRelationIds(
    BiomedicalEntityCreateWithoutRelationsInput
):
    comprised_of: list[str]
    parents: list[str]
    synonyms: list[str]


class OwnerCreateWithSynonymsInput(OwnerCreateWithoutRelationsInput):
    synonyms: list[str]


@dataclass(frozen=True)
class RelationConnectInfo(Dataclass):
    source_field: str
    dest_field: str
    input_type: Literal["set", "create"]

    def get_value(self, value: str, canonical_map: dict[str, CanonicalEntity]):
        # helper to get value appropriate for dest_field - canonical mapping if dest is canonical_id
        if self.dest_field == "canonical_id":
            if value in canonical_map:
                return canonical_map[value].id
            return None
        return value

    def form_prisma_relation(
        self, rec: BiomedicalEntityCreateInputWithRelationIds
    ) -> dict:
        if rec.get(self.source_field) is None:
            return {}
        return {
            self.input_type: [
                {self.dest_field: str(v)} for v in rec[self.source_field] or []  # type: ignore
            ]
        }


@dataclass(frozen=True)
class RelationIdFieldMap(Dataclass):
    def items(self) -> list[tuple[str, RelationConnectInfo]]:
        return [(field.name, getattr(self, field.name)) for field in fields(self)]

    comprised_of: RelationConnectInfo | None = None
    parents: RelationConnectInfo | None = None
    synonyms: RelationConnectInfo | None = None


@dataclass(frozen=True)
class BiomedicalEntityLoadSpec(Dataclass):
    sql: str
    candidate_selector: CandidateSelectorType
    get_source_map: Callable[[list[dict]], dict]
    additional_cleaners: Sequence[CleanFunction] = []
    get_terms: Callable[[dict], Sequence[str]] = lambda sm: list(sm.keys())
    get_terms_to_canonicalize: Callable[[dict], Sequence[str]] = lambda sm: list(
        sm.keys()
    )
    non_canonical_source: Source = Source.BIOSYM
    relation_id_field_map: RelationIdFieldMap = RelationIdFieldMap(
        synonyms=RelationConnectInfo(
            source_field="synonyms", dest_field="term", input_type="create"
        ),
    )
