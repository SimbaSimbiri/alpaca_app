from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from market_terminal.core.constants import DATA_FEED_IEX


load_dotenv()


@dataclass(frozen=True)
class Settings:
    api_key: str
    secret_key: str
    data_feed: str


def get_env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def load_settings() -> Settings:
    api_key = get_env_value(
        "ALPACA_API_KEY",
        "ALPACA_API_KEY_ID",
        "APCA_API_KEY_ID",
    )

    secret_key = get_env_value(
        "ALPACA_SECRET_KEY",
        "ALPACA_API_SECRET",
        "ALPACA_API_SECRET_KEY",
        "APCA_API_SECRET_KEY",
    )

    data_feed = os.getenv("ALPACA_DATA_FEED", DATA_FEED_IEX).lower().strip()

    if not api_key or not secret_key:
        raise RuntimeError(
            "Missing Alpaca credentials. Add your paper-trading credentials to .env using either:\n\n"
            "ALPACA_API_KEY=your_key\n"
            "ALPACA_SECRET_KEY=your_secret\n\n"
            "or:\n"
            "APCA_API_KEY_ID=your_key\n"
            "APCA_API_SECRET_KEY=your_secret"
        )

    return Settings(
        api_key=api_key,
        secret_key=secret_key,
        data_feed=data_feed,
    )


def get_alpaca_credentials() -> tuple[str, str]:
    settings = load_settings()
    return settings.api_key, settings.secret_key