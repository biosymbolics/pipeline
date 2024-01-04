from dataclasses import dataclass, fields
from typing import Literal
from core.ner.types import CanonicalEntity
from typings.core import Dataclass
from prisma.types import (
    BiomedicalEntityCreateWithoutRelationsInput,
)


class BiomedicalEntityCreateInputWithRelationIds(
    BiomedicalEntityCreateWithoutRelationsInput
):
    comprised_of: list[str]
    parents: list[str]
    synonyms: list[str]


@dataclass(frozen=True)
class RelationConnectInfo(Dataclass):
    source_field: str
    dest_field: str
    input_type: Literal["connect", "create"]

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
                {self.dest_field: v} for v in rec[self.source_field] or []
            ]
        }


@dataclass(frozen=True)
class RelationIdFieldMap(Dataclass):
    def items(self) -> list[tuple[str, RelationConnectInfo]]:
        return [(field.name, getattr(self, field.name)) for field in fields(self)]

    comprised_of: RelationConnectInfo | None = None
    parents: RelationConnectInfo | None = None
    synonyms: RelationConnectInfo | None = None
