from typing import Literal
import polars as pl
from pydash import compact
import yfinance as yf
import logging
import requests_cache

from utils.list import contains

from .stock import StockPerformance

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


CCE = "CashAndCashEquivalents"
STD = "ShortLongTermDebt"
LTD = "LongTermDebt"
ND = "NetDebt"
TD = "TotalDebt"
CA = "CurrentAssets"
TSE = "StockholdersEquity"
CL = "CurrentLiabilities"

SIG_DIGITS = 2


class CompanyFinancials(StockPerformance):
    """
    Company financial performance client

    Currently shows the most recent financials
    (either annual or quarterly, though market_cap is realish-time)
    """

    def __init__(
        self, ticker: str, timeframe: Literal["quarterly", "annual"] = "quarterly"
    ):
        """
        Initialize company financial performance client

        Args:
            ticker (str): company ticker
        """
        self.ticker = ticker
        session = requests_cache.CachedSession("yfinance.cache")
        session.headers["User-agent"] = "my-program/1.0"
        self.client = yf.Ticker(ticker, session=session)
        self.timeframe = timeframe

    @property
    def market_cap(self) -> float | None:
        try:
            mc = self.client.fast_info["marketCap"]
            if mc is None:
                return None

            return round(mc, SIG_DIGITS)
        except Exception as e:
            logger.error("Unable to fetch market cap for %s: %s", self.ticker, e)
            return None

    @property
    def balance_sheet(self) -> pl.DataFrame | None:
        try:
            balance_sheet = pl.DataFrame(self.client.get_balance_sheet().reset_index())  # type: ignore
            return balance_sheet
        except Exception as e:
            logger.error("Unable to fetch balance sheet for %s: %s", self.ticker, e)
            return None

    def get_balance_sheet_value(
        self, key: str, suppress_warning: bool = False
    ) -> float | None:
        if self.balance_sheet is None:
            return None

        if key not in self.balance_sheet_keys:
            if not suppress_warning:
                logger.warning(
                    "Unable to fetch balance sheet value %s for %s", key, self.ticker
                )
            return None

        return self.balance_sheet.row(by_predicate=(pl.col("index") == key))[1]

    @property
    def balance_sheet_keys(self) -> list[str]:
        if self.balance_sheet is None:
            return []
        return self.balance_sheet.select(pl.col("index")).to_series().to_list()

    @property
    def total_debt(self) -> float | None:
        if self.balance_sheet is None:
            return None

        calc_td = None
        td = self.get_balance_sheet_value(TD)

        if contains([STD, LTD], self.balance_sheet_keys):
            std = self.get_balance_sheet_value(STD) or 0
            ltd = self.get_balance_sheet_value(LTD) or 0
            calc_td = std + ltd

        tds = compact([td, calc_td])
        if len(tds) == 2 and td != calc_td:
            logger.warning(
                "Total debt mismatch for %s: td %s != calc_td %s",
                self.ticker,
                td,
                calc_td,
            )

        if len(tds) == 0:
            return None

        return tds[0]

    @property
    def cash_and_cash_equivalents(self) -> float | None:
        if self.balance_sheet is None:
            return None

        return self.get_balance_sheet_value(CCE)

    @property
    def net_debt(self) -> float | None:
        """
        "Net debt shows how much cash would remain if all debts were paid off and if a company has enough
        liquidity to meet its debt obligations." -- https://www.investopedia.com/terms/n/netdebt.asp

        NET_DEBT = STD + LTD - CASH

        Notes:
            - some balance sheets don't include NetDebt, in which case we calculate it.
            - the calculated and reported NetDebt values can differ,
              in which case we return the reported value and log a warning
        """
        if self.balance_sheet is None:
            return None

        nd = self.get_balance_sheet_value(ND, suppress_warning=True)
        calc_nd = None

        if self.total_debt is not None and self.cash_and_cash_equivalents is not None:
            calc_nd = self.total_debt - self.cash_and_cash_equivalents

        nds = compact([nd, calc_nd])

        if len(nds) == 2 and nd != calc_nd:
            logger.warning(
                "Net debt mismatch for %s: nd %s != calc_nd %s",
                self.ticker,
                nd,
                calc_nd,
            )

        if len(nds) == 0:
            return None

        return nds[0]

    @property
    def is_trading_below_cash(self) -> bool | None:
        if self.market_cap is None or self.net_debt is None:
            logger.warning("Unable to fetch market cap or net debt for %s", self.ticker)
            return None
        return self.market_cap < self.net_debt

    @property
    def current_ratio(self) -> float | None:
        """
        CR = current_assets / current_liabilities
        """
        if self.balance_sheet is None:
            return None

        current_assets = self.get_balance_sheet_value(CA)
        current_liabilities = self.get_balance_sheet_value(CL)

        if (
            current_assets is None
            or current_liabilities is None
            or current_liabilities == 0
        ):
            return None

        return round(current_assets / current_liabilities, SIG_DIGITS)

    @property
    def is_bad_current_ratio(self) -> bool:
        """
        Current ratio should be > 1
        """
        if self.current_ratio is None:
            logger.warning("Unable to fetch current ratio for %s", self.ticker)
            return False
        return self.current_ratio < 1

    @property
    def debt_equity_ratio(self) -> float | None:
        """
        DER = total_debt / total_equity
        """
        if self.balance_sheet is None:
            return None

        total_debt = self.total_debt
        total_equity = self.get_balance_sheet_value(TSE)

        if total_debt is None or total_equity is None or total_equity == 0:
            return None

        return round(total_debt / total_equity, SIG_DIGITS)

    @property
    def is_bad_debt_equity_ratio(self, max_ratio: float = 1.5) -> bool:
        """
        Debt equity ratio should be < 1.5
        """
        if self.debt_equity_ratio is None:
            logger.warning("Unable to fetch d/e ratio for %s", self.ticker)
            return False
        return self.debt_equity_ratio > max_ratio

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
        )
