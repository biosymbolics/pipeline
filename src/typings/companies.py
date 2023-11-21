from dataclasses import asdict, dataclass
from typing import Any, Optional

from typings.core import Dataclass

COMPANY_STR_KEYS = [
    "symbol",
    "market_cap",
    "total_debt",
    "net_debt",
    "current_ratio",
    "debt_equity_ratio",
    "is_bad_current_ratio",
    "is_trading_below_cash",
    "is_bad_debt_equity_ratio",
]


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
    is_trading_below_cash: bool | None
    is_bad_current_ratio: bool | None
    is_bad_debt_equity_ratio: bool | None
    market_cap: float | None
    net_debt: float | None
    total_debt: float | None
    parent_company_id: Optional[str]  # id
    synonyms: Optional[list[str]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self):
        return ", ".join([f"{k}={getattr(self, k)}" for k in COMPANY_STR_KEYS])

    def __repr__(self):
        return self.__str__()
