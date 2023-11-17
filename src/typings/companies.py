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
    market_cap: float
    net_debt: float
    is_trading_below_cash: bool
    synonyms: Optional[list[str]]
    parent_company_id: Optional[str]  # id

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
