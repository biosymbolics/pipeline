"""
stock performance client
"""
from datetime import date
import yfinance as yf
import logging

from .types import StockPrice

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class StockPerformance:
    @staticmethod
    def _normalize_stock_data(data) -> list[StockPrice]:
        try:
            data = data.reset_index().rename(columns={"index": "Date"})
            data["date"] = data["Date"].dt.strftime("%Y-%m-%d")
            data["close"] = data["Close"]
            data["open"] = data["Open"]
            data["volume"] = data["Volume"]
            df = data[["date", "close", "open", "volume"]]
            return df.to_dict("records")
        except Exception as e:
            logger.warning("Error normalizing data: %s (data: %s)", e, data)
            raise e

    @staticmethod
    def fetch_stock_over_time(
        ticker: str, start_date: date, end_date: date
    ) -> list[StockPrice]:
        """
        Get stock performance over time

        Args:
            ticker (str): ticker to get data for
            start_date (date): start date
            end_date (date): end date
        """
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        logging.info(
            "Fetching yfinance data for %s, %s-%s", ticker, start_date_str, end_date_str
        )
        data = yf.download(ticker, start=start_date_str, end=end_date_str)

        if len(data) > 0:
            return StockPerformance._normalize_stock_data(data)

        return []
