from dataclasses import asdict, dataclass
from typing import Any, Optional

from typings.core import Dataclass


@dataclass(frozen=True)
class Company(Dataclass):
    """
    Company dataclass
    """

    id: str
    name: str
    symbol: str
    current_ratio: float | None
    debt_equity_ratio: float | None
    is_troubled: bool | None
    market_cap: float | None
    net_debt: float | None
    total_debt: float | None
    parent_company_id: Optional[str]  # id
    synonyms: Optional[list[str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
