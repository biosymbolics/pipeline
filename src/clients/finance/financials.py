from dataclasses import dataclass
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
TA = "TotalAssets"
TSE = "StockholdersEquity"
CL = "CurrentLiabilities"
TL = "TotalLiabilities"

SIG_DIGITS = 2


class CompanyFinancialExtractor(StockPerformance):
    """
    Company financial performance client

    Currently shows the most recent financials
    (either annual or quarterly, though market_cap is realish-time)

    Usage:
    ```
    from clients.finance.financials import CompanyFinancials
    c = CompanyFinancials('BAYRY')
    ```
    """

    def __init__(
        self, symbol: str, timeframe: Literal["quarterly", "yearly"] = "yearly"
    ):
        """
        Initialize company financial performance client

        Args:
            symbol (str): company symbol
        """
        logger.info("Loading CompanyFinancials for %s", symbol)
        self.symbol = symbol
        session = requests_cache.CachedSession("yfinance.cache")
        session.headers["User-agent"] = "my-program/1.0"
        self.client = yf.Ticker(symbol, session=session)
        self.timeframe = timeframe

    def __str__(self):
        return ", ".join(
            [f"{k}={getattr(self, k)}" for k in self.__class__.__dict__.keys()]
        )

    def __repr__(self):
        return self.__str__()

    def get_value(
        self, parent: Literal["balance_sheet", "income_statement"], key: str
    ) -> float | None:
        if parent not in dir(self):
            return None

        if key not in getattr(self, f"{parent}_keys"):
            return None

        value = getattr(self, parent).row(by_predicate=(pl.col("index") == key))[1]

        if value is None:
            logger.warning(
                "Unable to fetch %s for %s, timeframe %s",
                key,
                self.symbol,
                self.timeframe,
            )
            return None

        return value

    @property
    def market_cap(self) -> float | None:
        try:
            mc = self.client.fast_info["marketCap"]
            if mc is None:
                return None

            return round(mc, SIG_DIGITS)
        except Exception as e:
            logger.error("Unable to fetch market cap for %s: %s", self.symbol, e)
            return None

    @property
    def shareholders_equity(self) -> float | None:
        """
        Shareholders equity  = total assets - total liabilities
        """
        se = self.get_value("balance_sheet", TSE)
        calc_se = None

        total_assets = self.get_value("balance_sheet", TA)
        total_liabilities = self.get_value("balance_sheet", TL)

        if total_assets is not None and total_liabilities is not None:
            calc_se = total_assets - total_liabilities
            if se is not None and se != calc_se:
                logger.warning(
                    "Shareholders equity mismatch for %s: se %s != calc_se %s",
                    self.symbol,
                    se,
                    calc_se,
                )

        return se or calc_se

    @property
    def income_statement(self) -> pl.DataFrame | None:
        try:
            income_stmt = pl.DataFrame(
                self.client.get_income_stmt(freq=self.timeframe).reset_index()  # type: ignore
            )
            return income_stmt
        except Exception as e:
            logger.error("Unable to fetch income stmt for %s: %s", self.symbol, e)
            return None

    @property
    def income_statement_keys(self) -> list[str]:
        if self.income_statement is None:
            return []
        return self.income_statement.select(pl.col("index")).to_series().to_list()

    @property
    def return_on_equity(self) -> float | None:
        """
        ROE = net_income / avg_equity
        https://www.investopedia.com/terms/r/returnonequity.asp
        "a gauge of a corporation's profitability and how efficiently it generates those profits"

        TODO: should use annual net_income regardless of timeframe?
        TODO: move to DuPont Calculation of ROE?
        """

        net_income = self.get_value("income_statement", "NetIncome")

        if (
            net_income is None
            or self.shareholders_equity is None
            or self.shareholders_equity == 0
        ):
            return None

        return round(net_income / self.shareholders_equity, SIG_DIGITS)

    @property
    def gross_profit(self) -> float | None:
        """
        Gross profit = revenue - cost of goods sold
        """
        gp = self.get_value("income_statement", "GrossProfit")
        return gp

    @property
    def ebitda(self) -> float | None:
        """
        EBITDA = earnings before interest, taxes, depreciation, and amortization
        """
        ebitda = self.get_value("income_statement", "EBITDA")
        return ebitda

    @property
    def return_on_research_capital(self):
        """
        return on research capital ratio = net income / research capital

        RORC = net_income / research_capital

        TODO: is 'ResearchAndDevelopment' the right key?
        """
        net_income = self.get_value("income_statement", "NetIncome")
        research_capital = self.get_value("income_statement", "ResearchAndDevelopment")

        if net_income is None or research_capital is None or research_capital == 0:
            return None

        return round(net_income / research_capital, SIG_DIGITS)

    @property
    def balance_sheet(self) -> pl.DataFrame | None:
        try:
            balance_sheet = pl.DataFrame(
                self.client.get_balance_sheet(freq=self.timeframe).reset_index()  # type: ignore
            )
            return balance_sheet
        except Exception as e:
            logger.error("Unable to fetch balance sheet for %s: %s", self.symbol, e)
            return None

    @property
    def balance_sheet_keys(self) -> list[str]:
        if self.balance_sheet is None:
            return []
        return self.balance_sheet.select(pl.col("index")).to_series().to_list()

    @property
    def total_debt(self) -> float | None:
        calc_td = None
        td = self.get_value("balance_sheet", TD)

        if contains([STD, LTD], self.balance_sheet_keys):
            std = self.get_value("balance_sheet", STD) or 0
            ltd = self.get_value("balance_sheet", LTD) or 0
            calc_td = std + ltd

        tds = compact([td, calc_td])
        if len(tds) == 2 and td != calc_td:
            logger.warning(
                "Total debt mismatch for %s: td %s != calc_td %s",
                self.symbol,
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

        return self.get_value("balance_sheet", CCE)

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
        nd = self.get_value("balance_sheet", ND)
        calc_nd = None

        if self.total_debt is not None and self.cash_and_cash_equivalents is not None:
            calc_nd = self.total_debt - self.cash_and_cash_equivalents

        nds = compact([nd, calc_nd])

        if len(nds) == 2 and nd != calc_nd:
            logger.warning(
                "Net debt mismatch for %s: nd %s != calc_nd %s",
                self.symbol,
                nd,
                calc_nd,
            )

        if len(nds) == 0:
            return None

        return nds[0]

    @property
    def current_ratio(self) -> float | None:
        """
        CR = current_assets / current_liabilities
        """
        current_assets = self.get_value("balance_sheet", CA)
        current_liabilities = self.get_value("balance_sheet", CL)

        if (
            current_assets is None
            or current_liabilities is None
            or current_liabilities == 0
        ):
            return None

        return round(current_assets / current_liabilities, SIG_DIGITS)

    @property
    def debt_equity_ratio(self) -> float | None:
        """
        DER = total_debt / total_equity
        """
        total_debt = self.total_debt
        total_equity = self.get_value("balance_sheet", TSE)

        if total_debt is None or total_equity is None or total_equity == 0:
            return None

        return round(total_debt / total_equity, SIG_DIGITS)
