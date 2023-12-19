from dataclasses import dataclass
from datetime import date
from typings.core import Dataclass


@dataclass(frozen=True)
class RegulatoryApproval(Dataclass):
    """
    Base approval info
    """

    applicant: str
    application_types: list[str]
    approval_dates: list[date]
    application_number: str
    brand_name: str
    generic_name: str
    # indications: list[str]
    label_url: str
    routes: list[str]
    score: float = 0.0
