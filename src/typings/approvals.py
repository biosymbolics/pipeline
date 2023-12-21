from dataclasses import dataclass
from pydantic import validate_arguments
from datetime import date
from typings.core import Dataclass


@validate_arguments  # type: ignore
@dataclass(frozen=True)
class RegulatoryApproval(Dataclass):
    """
    Base approval info
    """

    @property
    def count(self) -> int:
        return len(self.approval_dates)

    @property
    def instance_rollup(self) -> str:
        return (self.pharmacologic_class or self.generic_name).lower()

    # applicant: str
    application_types: list[str]
    approval_dates: list[date]
    # application_number: str
    brand_name: str
    generic_name: str
    ndc_code: str
    # indications: list[str]
    label_url: str
    pharmacologic_class: str | None
    # routes: list[str]
    score: float = 0.0
