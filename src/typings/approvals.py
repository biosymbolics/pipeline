from dataclasses import dataclass
from pydantic import validate_arguments
from datetime import date
from typings.core import Dataclass


# @validate_arguments  # type: ignore
# @dataclass(frozen=True)
# class RegulatoryApproval(Dataclass):
#     """
#     Base approval info
#     """

#     @property
#     def count(self) -> int:
#         return 1

#     @property
#     def instance_rollup(self) -> str:
#         return (self.pharmacologic_class or self.generic_name).lower()

#     # applicant: str
#     application_type: str
#     approval_date: date
#     # application_number: str
#     brand_name: str | None
#     generic_name: str
#     ndc_code: str
#     indications: list[str]
#     label_url: str
#     pharmacologic_class: str | None
#     pharmacologic_classes: list[str]
#     regulatory_agency: str | None
#     # routes: list[str]
#     score: float = 0.0
