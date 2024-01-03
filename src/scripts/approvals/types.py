from dataclasses import dataclass
from prisma.types import (
    BiomedicalEntityCreateWithoutRelationsInput,
)

from typings.core import Dataclass


@dataclass(frozen=True)
class RelationIdFieldMap(Dataclass):
    comprised_of: str
    parents: str


@dataclass(frozen=True)
class InterventionIntermediate(Dataclass):
    generic_name: str
    brand_name: str
    active_ingredients: list[str]
    pharmacologic_classes: list[str]


class BiomedicalEntityCreateInputWithRelationIds(
    BiomedicalEntityCreateWithoutRelationsInput
):
    comprised_of: list[str]
    parents: list[str]
