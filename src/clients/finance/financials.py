import polars as pl
import yfinance as yf

from .stock import StockPerformance


class CompanyFinancials(StockPerformance):
    """
    Company financial performance client
    """

    def __init__(self, ticker: str):
        self.ticker = ticker
        self.client = yf.Ticker(ticker)

    def market_cap(self) -> float:
        return self.client.fast_info["marketCap"]

    def net_debt(self) -> float:
        """
        "Net debt shows how much cash would remain if all debts were paid off and if a company has enough
        liquidity to meet its debt obligations." -- https://www.investopedia.com/terms/n/netdebt.asp

        NET_DEBT = STD + LTD - CASH
        """
        bs_df = pl.DataFrame(self.client.get_balance_sheet().reset_index())  # type: ignore
        net_debt = bs_df.row(by_predicate=(pl.col("index") == "NetDebt"))
        return net_debt[1]

    def is_trading_below_cash(self) -> bool:
        return self.market_cap() < self.net_debt()
