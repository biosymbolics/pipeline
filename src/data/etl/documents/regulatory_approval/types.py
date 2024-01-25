import re
from pydantic import BaseModel
from pydash import compact

from constants.patterns.intervention import PRIMARY_MECHANISM_BASE_TERMS
from utils.re import get_or_re


class PharmaClass(BaseModel):
    name: str
    type: str

    def __init__(self, **kwargs):
        """
        Init with only name and type (both then lowered)
        """
        super().__init__(
            **{"name": kwargs["name"].lower(), "type": kwargs["type"].lower()}
        )

    @staticmethod
    def sort(pharma_classes: list["PharmaClass"]) -> list["PharmaClass"]:
        """
        Temporary/hack solution for getting the preferred pharmacologic class
        """

        def get_priority(pc: "PharmaClass") -> int:
            score = 0
            if (
                re.match(
                    f".*{get_or_re(list(PRIMARY_MECHANISM_BASE_TERMS.values()))}.*",
                    pc.name,
                    flags=re.IGNORECASE,
                )
                is not None
            ):
                score += 10
            if pc.type == "moa":
                score += 3
            elif pc.type == "epc":
                score += 2
            elif pc.type == "mesh":
                score += 1

            return score

        prioritized = sorted(
            [pa for pa in pharma_classes if pa.name is not None],
            key=get_priority,
            reverse=True,
        )
        return prioritized


class InterventionIntermediate(BaseModel):
    generic_name: str
    brand_name: str
    active_ingredients: list[str]
    pharmacologic_classes: list[PharmaClass]

    @property
    def intervention(self) -> str:
        return self.generic_name or self.brand_name

    def __init__(self, pharmacologic_classes, **kwargs):
        """
        Init with only name and type (both then lowered)
        """
        super().__init__(pharmacologic_classes=compact(pharmacologic_classes), **kwargs)
