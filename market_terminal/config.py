from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_key: str
    secret_key: str
    data_feed: str


def load_settings() -> Settings:
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    data_feed = os.getenv("ALPACA_DATA_FEED", "iex").lower().strip()

    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing Alpaca credentials. Add ALPACA_API_KEY and "
            "ALPACA_SECRET_KEY to your local .env file."
        )

    return Settings(
        api_key=api_key,
        secret_key=secret_key,
        data_feed=data_feed,
    )