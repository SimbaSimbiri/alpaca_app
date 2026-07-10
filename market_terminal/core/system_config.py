from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class UniverseConfig:
    symbols: list[str] = field(default_factory=lambda: ["SPY"])


@dataclass(frozen=True)
class StrategyRuntimeConfig:
    probability_threshold: float | None = None
    years: int = 5
    feed: str = "sip"
    data_delay_minutes: int = 20


@dataclass(frozen=True)
class RiskRuntimeConfig:
    max_order_qty: float = 10.0
    min_buying_power_after_order: float = 0.0
    allow_short_selling: bool = False


@dataclass(frozen=True)
class EngineRuntimeConfig:
    model_bundles: list[str] = field(default_factory=list)
    qty: float = 1.0
    polling_interval_seconds: float = 300.0
    cycles: int = 1
    continuous: bool = False
    execute_orders: bool = False
    order_snapshot_limit: int = 50


@dataclass(frozen=True)
class SystemConfig:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    strategy: StrategyRuntimeConfig = field(default_factory=StrategyRuntimeConfig)
    risk: RiskRuntimeConfig = field(default_factory=RiskRuntimeConfig)
    engine: EngineRuntimeConfig = field(default_factory=EngineRuntimeConfig)


def _as_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if not isinstance(value, dict):
        raise ValueError("Config sections must be YAML objects.")

    return value


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError("Expected a list of strings.")

    return [str(item) for item in value]


def load_system_config(path: str | Path | None = None) -> SystemConfig:
    """
    Loads project runtime configuration from YAML.

    If no path is provided, returns safe defaults.
    """

    if path is None:
        return SystemConfig()

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if not isinstance(raw, dict):
        raise ValueError("Top-level config must be a YAML object.")

    universe_raw = _as_dict(raw.get("universe"))
    strategy_raw = _as_dict(raw.get("strategy"))
    risk_raw = _as_dict(raw.get("risk"))
    engine_raw = _as_dict(raw.get("engine"))

    universe = UniverseConfig(
        symbols=_as_string_list(universe_raw.get("symbols")) or ["SPY"],
    )

    strategy = StrategyRuntimeConfig(
        probability_threshold=strategy_raw.get("probability_threshold"),
        years=int(strategy_raw.get("years", 5)),
        feed=str(strategy_raw.get("feed", "sip")).lower(),
        data_delay_minutes=int(strategy_raw.get("data_delay_minutes", 20)),
    )

    risk = RiskRuntimeConfig(
        max_order_qty=float(risk_raw.get("max_order_qty", 10.0)),
        min_buying_power_after_order=float(
            risk_raw.get("min_buying_power_after_order", 0.0)
        ),
        allow_short_selling=bool(risk_raw.get("allow_short_selling", False)),
    )

    engine = EngineRuntimeConfig(
        model_bundles=_as_string_list(engine_raw.get("model_bundles")),
        qty=float(engine_raw.get("qty", 1.0)),
        polling_interval_seconds=float(
            engine_raw.get("polling_interval_seconds", 300.0)
        ),
        cycles=int(engine_raw.get("cycles", 1)),
        continuous=bool(engine_raw.get("continuous", False)),
        execute_orders=bool(engine_raw.get("execute_orders", False)),
        order_snapshot_limit=int(engine_raw.get("order_snapshot_limit", 50)),
    )

    return SystemConfig(
        universe=universe,
        strategy=strategy,
        risk=risk,
        engine=engine,
    )