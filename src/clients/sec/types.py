from typing import Literal
from pydantic import BaseModel


class SecFilingSection(BaseModel):
    """
    SEC filing section
    """

    sequence: str  # number
    description: str  # e.g. EX-31.2
    documentUrl: str
    type: str  # e.g. EX-32
    size: int


class SecFiling(BaseModel):
    """
    SEC filing
    """

    id: str
    ticker: str
    accessionNo: str
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


SecProductQueryStrategy = Literal["TABLE", "SEARCH"]


# there may be others
ExtractReturnType = Literal["html", "text"]
