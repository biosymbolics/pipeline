from dataclasses import dataclass
from typings.core import Dataclass
from prisma.types import (
    BiomedicalEntityCreateWithoutRelationsInput,
)


@dataclass(frozen=True)
class RelationIdFieldMap(Dataclass):
    comprised_of: str
    parents: str


class BiomedicalEntityCreateInputWithRelationIds(
    BiomedicalEntityCreateWithoutRelationsInput
):
    comprised_of: list[str]
    parents: list[str]
