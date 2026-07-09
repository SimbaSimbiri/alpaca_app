from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from market_terminal.backtest.engine import (
    BacktestConfig,
    backtest_buy_and_hold,
    backtest_long_only_strategy,
)
from market_terminal.core.constants import BACKTEST_YEARS, INITIAL_CAPITAL, RISK_FREE_RATE
from market_terminal.features.indicators import add_all_indicators
from market_terminal.backtest.metrics import build_metrics_table, format_metrics_for_console
from market_terminal.reporting.report import create_pdf_report
from market_terminal.reporting.visualizations import save_all_charts
from market_terminal.strategy.technical_strategies import add_all_strategy_signals


def make_sample_ohlcv(years: int = BACKTEST_YEARS, seed: int = 7) -> pd.DataFrame:
    """
    Development-only sample data. Use Alpaca data for the actual assignment.
    """
    rng = np.random.default_rng(seed)
    periods = int(252 * years)
    index = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)

    drift = 0.00035
    shocks = rng.normal(loc=drift, scale=0.018, size=periods)
    close = 100 * np.exp(np.cumsum(shocks))
    open_ = close * (1 + rng.normal(0, 0.004, periods))
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.018, periods))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.018, periods))
    volume = rng.integers(1_000_000, 8_000_000, periods)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def load_market_data(ticker: str, years: int, use_sample_data: bool) -> pd.DataFrame:
    if use_sample_data:
        print("Using generated sample data. Use real Alpaca data for final submission.")
        return make_sample_ohlcv(years=years)

    from market_terminal.data_connector import AlpacaDataConnector

    connector = AlpacaDataConnector()
    return connector.get_daily_ohlcv(symbol=ticker, years=years)


def run_backtest(
    ticker: str,
    years: int,
    initial_capital: float,
    commission: float,
    fractional: bool,
    output_root: Path,
    use_sample_data: bool = False,
) -> dict[str, Path | pd.DataFrame]:
    ticker = ticker.upper().strip()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{ticker}_{timestamp}"
    chart_dir = run_dir / "charts"
    data_dir = run_dir / "data"
    report_dir = run_dir / "reports"

    chart_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_market_data(ticker=ticker, years=years, use_sample_data=use_sample_data)

    if raw_data.empty:
        raise RuntimeError(f"No historical data returned for {ticker}.")

    data = add_all_indicators(raw_data)
    data, strategy_specs = add_all_strategy_signals(data)

    config = BacktestConfig(
        initial_capital=initial_capital,
        commission_per_trade=commission,
        allow_fractional_shares=fractional,
    )

    results = [backtest_buy_and_hold(data, config=config)]

    for spec in strategy_specs:
        results.append(
            backtest_long_only_strategy(
                df=data,
                signal=data[spec.signal_column],
                name=spec.name,
                config=config,
            )
        )

    metrics = build_metrics_table(
        results=results,
        initial_capital=initial_capital,
        risk_free_rate=RISK_FREE_RATE,
    )

    # Persist data, trades, portfolio paths for reproducibility.
    data.to_csv(data_dir / f"{ticker}_daily_ohlcv_indicators_signals.csv")
    metrics.to_csv(data_dir / f"{ticker}_performance_metrics.csv")

    for result in results:
        safe_name = (
            result.name.lower()
            .replace("&", "and")
            .replace(":", "")
            .replace(" ", "_")
            .replace("/", "_")
        )
        result.portfolio.to_csv(data_dir / f"{safe_name}_portfolio.csv")
        result.trades.to_csv(data_dir / f"{safe_name}_trades.csv")

    chart_paths = save_all_charts(data, results, chart_dir)

    report_path = create_pdf_report(
        ticker=ticker,
        years=years,
        data=data,
        metrics=metrics,
        strategy_specs=strategy_specs,
        chart_paths=chart_paths,
        output_path=report_dir / f"{ticker}_final_report.pdf",
    )

    print("\nPerformance metrics:")
    print(format_metrics_for_console(metrics).to_string())
    print(f"\nOutput folder: {run_dir}")
    print(f"Final report: {report_path}")

    return {
        "run_dir": run_dir,
        "report_path": report_path,
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run technical-indicator strategy backtests using Alpaca historical daily OHLCV data."
    )
    parser.add_argument("--ticker", default="MSFT", help="Ticker symbol, for example AAPL, MSFT, SPY, QQQ, NVDA.")
    parser.add_argument("--years", type=int, default=BACKTEST_YEARS, help="Number of calendar years of daily OHLCV data.")
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL, help="Initial capital for each strategy.")
    parser.add_argument("--commission", type=float, default=0.0, help="Flat commission/slippage cost per executed trade.")
    parser.add_argument("--fractional", action="store_true", help="Allow fractional shares. Default uses whole shares.")
    parser.add_argument("--output-root", default="outputs", help="Folder where charts, data, and PDF report are saved.")
    parser.add_argument("--sample", action="store_true", help="Use generated sample data for local testing without Alpaca credentials.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_backtest(
        ticker=args.ticker,
        years=args.years,
        initial_capital=args.initial_capital,
        commission=args.commission,
        fractional=args.fractional,
        output_root=Path(args.output_root),
        use_sample_data=args.sample,
    )


if __name__ == "__main__":
    main()
