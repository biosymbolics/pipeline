from dataclasses import dataclass, field, fields
from typing import Callable, Literal, Sequence
from prisma.enums import BiomedicalEntityType, Source

from core.ner.cleaning import CleanFunction
from core.ner.linker.types import CandidateSelectorType
from core.ner.types import CanonicalEntity
from typings.core import Dataclass
from typings.prisma import BiomedicalEntityCreateInputWithRelationIds


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
            source_field="synonyms", dest_field="term", input_type="create"
        ),
    )
