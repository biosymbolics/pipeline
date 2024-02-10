from dataclasses import dataclass
from typing import TypedDict
from prisma.models import FinancialSnapshot
import logging

from typings.core import ResultBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

CompanyInfo = TypedDict("CompanyInfo", {"name": str, "symbol": str})


class CompanyFinancials(FinancialSnapshot, ResultBase):
    @property
    def is_bad_current_ratio(self) -> bool:
        """
        Current ratio should be > 1
        """
        if self.current_ratio is None:
            logger.warning("Unable to fetch current ratio for %s", self.symbol)
            return False
        return self.current_ratio < 1

    @property
    def is_bad_debt_equity_ratio(self, max_ratio: float = 1.5) -> bool:
        """
        Debt equity ratio should be < 1.5
        """
        if self.debt_equity_ratio is None:
            logger.warning("Unable to fetch d/e ratio for %s", self.symbol)
            return False
        return self.debt_equity_ratio > max_ratio

    @property
    def is_low_return_on_equity(self, min_roe: float = 0.15) -> bool:
        """
        Return on equity should be > 15%
        """
        if self.return_on_equity is None:
            return False
        return self.return_on_equity < min_roe

    @property
    def is_trading_below_cash(self) -> bool:
        if self.market_cap is None or self.net_debt is None:
            logger.warning(
                "Unable to fetch market cap and/or net debt for %s", self.symbol
            )
            return False
        return self.market_cap < self.net_debt

    @property
    def is_troubled(self) -> bool:
        """
        Returns true if any of the following are true:

        - is trading below cash
        - has a bad current ratio
        - has a bad debt equity ratio

        This indicates a company may be at risk of bankruptcy, being scrapped,
        or interested in liquidation or buyout.

        TODO: include change over time
        """
        return (
            self.is_trading_below_cash
            or self.is_bad_current_ratio
            or self.is_bad_debt_equity_ratio
            or self.is_low_return_on_equity
        )
