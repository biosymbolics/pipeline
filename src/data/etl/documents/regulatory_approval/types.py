import re
from pydantic import BaseModel
from pydash import compact

from constants.patterns.intervention import PRIMARY_MECHANISM_BASE_TERMS
from utils.re import get_or_re

IS_MECHANISM_RE = re.compile(
    ".*" + get_or_re(list(PRIMARY_MECHANISM_BASE_TERMS.values())) + ".*"
)


class PharmaClass(BaseModel):
    name: str
    type: str

    @staticmethod
    def sort(pharma_classes: list["PharmaClass"]) -> list["PharmaClass"]:
        """
        Temporary/hack solution for getting the preferred pharmacologic class
        """

        def get_priority(pc: "PharmaClass") -> int:
            score = 0
            if IS_MECHANISM_RE.match(pc.name.lower()) is not None:
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
        pcs = [
            PharmaClass(name=pa["name"], type=pa["type"])
            for pa in compact(pharmacologic_classes)
        ]
        super().__init__(pharmacologic_classes=pcs, **kwargs)
