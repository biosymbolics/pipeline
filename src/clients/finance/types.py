from typing import TypedDict


StockPrice = TypedDict(
    "StockPrice", {"date": str, "close": float, "open": float, "volume": int}
)
