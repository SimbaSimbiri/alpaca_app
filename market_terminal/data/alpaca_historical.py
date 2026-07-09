from __future__ import annotations

from datetime import datetime

import pandas as pd

from market_terminal.core.settings import get_alpaca_credentials


def download_daily_ohlcv_from_alpaca(
    symbol: str,
    start: datetime,
    end: datetime,
    feed: str = "iex",
) -> pd.DataFrame:
    """
    Downloads daily OHLCV bars from Alpaca.

    feed options:
    - "iex": free IEX feed
    - "sip": full-market SIP feed, but recent data may require a paid subscription
    """

    try:
        from alpaca.data.enums import DataFeed
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError as exc:
        raise ImportError(
            "alpaca-py is not installed. Install it with:\n"
            "pip install alpaca-py"
        ) from exc

    api_key, api_secret = get_alpaca_credentials()

    client = StockHistoricalDataClient(
        api_key=api_key,
        secret_key=api_secret,
    )

    feed = feed.lower().strip()

    if feed == "iex":
        data_feed = DataFeed.IEX
    elif feed == "sip":
        data_feed = DataFeed.SIP
    else:
        raise ValueError("feed must be either 'iex' or 'sip'.")

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=start,
        end=end,
        feed=data_feed,
    )

    try:
        bars = client.get_stock_bars(request).df
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download Alpaca data for {symbol} using feed='{feed}'.\n\n"
            "Most common fix:\n"
            "1. Use the free IEX feed:\n"
            f"   python run_ml_backtest.py --symbol {symbol} --feed iex\n\n"
            "2. Or use SIP with an end time at least 15–20 minutes behind current time:\n"
            f"   python run_ml_backtest.py --symbol {symbol} --feed sip --data-delay-minutes 20\n\n"
            "Original Alpaca error:\n"
            f"{exc}"
        ) from exc

    if bars.empty:
        raise ValueError(
            f"No daily bars returned for {symbol}. "
            "Check the symbol, date range, and Alpaca data access."
        )

    if isinstance(bars.index, pd.MultiIndex):
        index_names = list(bars.index.names)

        if "symbol" in index_names:
            bars = bars.xs(symbol, level="symbol")
        else:
            bars = bars.loc[symbol]

    bars = bars.sort_index()
    bars.index.name = "timestamp"

    required_columns = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required_columns if col not in bars.columns]

    if missing:
        raise ValueError(f"Downloaded Alpaca data is missing columns: {missing}")

    ohlcv = bars[required_columns].copy()
    ohlcv = ohlcv.dropna()

    return ohlcv