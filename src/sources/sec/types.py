"""
SEC-related types
"""
from typing import Literal, TypedDict


class SecFilingSection(TypedDict):
    sequence: str  # number
    description: str  # e.g. EX-31.2
    documentUrl: str
    type: str  # e.g. EX-32
    size: int


class SecFiling(TypedDict):
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


class SecFilingResponse(TypedDict):
    # ...other stuff
    filings: list[SecFiling]


SecProductQueryStrategy = Literal["TABLE", "SEARCH"]


# there may be others
ExtractReturnType = Literal["html", "text", "stripped-html"]
