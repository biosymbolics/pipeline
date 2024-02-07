"""
Patent types
"""

from dataclasses import dataclass
from typing import Literal, TypedDict

from typings.core import Dataclass


AutocompleteMode = Literal["id", "term"]
AutocompleteResult = TypedDict("AutocompleteResult", {"id": str, "label": str})


RelevancyThreshold = Literal["very low", "low", "medium", "high", "very high"]
