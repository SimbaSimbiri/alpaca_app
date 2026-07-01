from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import ceil
from typing import Any

import pandas as pd
from alpaca.data import TimeFrameUnit
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from pandas import DataFrame, Series

from market_terminal.config import load_settings
from market_terminal.constants import HISTORICAL_DAYS, DATA_FEED_IEX, DATA_FEED_SIP, BACKTEST_YEARS

FEED_MAP = {
    DATA_FEED_IEX: DataFeed.IEX,  # Investor’s exchange data feed
    DATA_FEED_SIP: DataFeed.SIP,  # Securities' Information Processor feed
}


class AlpacaDataConnector:
    """
    Handles Alpaca authentication and historical market data retrieval.
    """

    def __init__(self) -> None:
        settings = load_settings()

        self.api_key = settings.api_key
        self.secret_key = settings.secret_key
        self.feed = FEED_MAP.get(settings.data_feed, DataFeed.IEX)

        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
            paper=True,
        )

        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.secret_key,
        )

    def validate_paper_account(self) -> str:
        """
        Confirms paper-trading credentials are valid.
        """
        account = self.trading_client.get_account()
        return f"Paper account status: {account.status}"

    def get_historical_bars(
            self,
            symbol: str,
            time_frame: TimeFrame,
            days: int = HISTORICAL_DAYS,
    ) -> DataFrame | Series[Any]:
        """
        Downloads historical OHLCV stock bars using the user-selected timeframe.
        eg:
            TimeFrame(15, TimeFrameUnit.Minute) -> 15-minute bars
            TimeFrame(1, TimeFrameUnit.Hour) -> 1-hour bars
            TimeFrame(1, TimeFrameUnit.Day) -> daily bars
        """
        clean_symbol = symbol.upper().strip()

        if not clean_symbol:
            raise ValueError("Symbol cannot be empty.")

        # we allow users to specify how far back they want to look
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)

        return self._request_bars(clean_symbol, time_frame, start, end)

    def get_daily_ohlcv(self, symbol: str, years: int = BACKTEST_YEARS) -> DataFrame:
        """
        Downloads by default 5 years of histroical daily OHLCV data.
        Returns a DataFrame indexed by timestamp and contains Open, High, Low, Close, Volume.
        """
        clean_symbol = symbol.upper().strip()

        if not clean_symbol:
            raise ValueError("Symbol cannot be empty.")

        if years <= 0:
            raise ValueError("Years must be greater than zero.")

        calendar_days = ceil(years * 365.25) + 10
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=calendar_days)

        bars = self._request_bars(clean_symbol, TimeFrame(1, TimeFrameUnit.Day), start, end)

        if bars.empty:
            return bars

        return bars[["open", "high", "low", "close", "volume"]].copy()

    def _request_bars(self, symbol: str, timeframe: TimeFrame, start: datetime, end: datetime) -> DataFrame:
        """ Helper function to request ohlcv bars."""
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=self.feed,
        )

        bars = self.data_client.get_stock_bars(request).df

        if bars.empty:
            return bars

        # we only want to extract the symbol data if
        # df has multi symbols on the row index
        if isinstance(bars.index, pd.MultiIndex):
            bars = bars.xs(symbol, level=0)

        # xs returns a df indexed with the date and time, but we convert all
        # just for safety and consistent formating
        bars.index = pd.to_datetime(bars.index)
        bars = bars.sort_index()

        expected_columns = ["open", "high", "low", "close", "volume"]
        optional_columns = [
            col for col in ["trade_count", "vwap"] if col in bars.columns
        ]

        return bars[expected_columns + optional_columns]
