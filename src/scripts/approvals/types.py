from dataclasses import dataclass

from typings.core import Dataclass


@dataclass(frozen=True)
class InterventionIntermediate(Dataclass):
    generic_name: str
    brand_name: str
    active_ingredients: list[str]
    pharmacologic_classes: list[str]
