"""
SEC-related types
"""
from typing import Literal, TypedDict


class SecFilingSection(TypedDict):
    """
    SEC filing section
    """

    sequence: str  # number
    description: str  # e.g. EX-31.2
    documentUrl: str
    type: str  # e.g. EX-32
    size: int


class SecFiling(TypedDict):
    """
    SEC filing
    """

    id: str
    ticker: str
    companyName: str
    companyNameLong: str
    formType: str
    description: str
    filedAt: str  # iso date
    linkToText: str  # url
    linkToHtml: str  # url
    linkToXbrl: str  # url
    linkToFilingDetails: str  # url
    documentFormatFiles: list[SecFilingSection]
    dataFiles: list[SecFilingSection]
    periodOfReport: str  # of date


SecProductQueryStrategy = Literal["TABLE"]


# there may be others
ExtractReturnType = Literal["html", "text"]
