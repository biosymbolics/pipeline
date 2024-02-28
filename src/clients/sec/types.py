from typing import Literal
from pydantic import BaseModel


class SecFilingSection(BaseModel):
    """
    SEC filing section
    """

    sequence: str | None = None  # number
    description: str | None = None  # e.g. EX-31.2
    documentUrl: str | None = None
    type: str | None = None  # e.g. EX-32
    size: int | None = None


class SecFiling(BaseModel):
    """
    SEC filing
    """

    id: str
    ticker: str
    accessionNo: str | None = None
    companyName: str
    companyNameLong: str | None = None
    formType: str | None = None
    description: str | None = None
    filedAt: str  # iso date
    linkToTxt: str  # url
    linkToHtml: str | None = None
    linkToXbrl: str | None = None
    linkToFilingDetails: str | None = None
    documentFormatFiles: list[SecFilingSection]
    dataFiles: list[SecFilingSection]


SecProductQueryStrategy = Literal["TABLE", "SEARCH"]


# there may be others
ExtractReturnType = Literal["html", "text"]
