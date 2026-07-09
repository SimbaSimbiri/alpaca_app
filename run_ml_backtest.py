from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, time, timezone, timedelta
from pathlib import Path

import joblib
import pandas as pd

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from market_terminal.features.feature_engineering import get_clean_ml_dataset, FEATURE_COLUMNS
from market_terminal.features.pca import fit_pca, transform_pca, print_pca_summary
from market_terminal.strategy.ml_model import (
    time_train_test_split,
    train_random_forest,
    predict_up_probability,
    probability_to_signal,
    print_model_summary,
)
from market_terminal.reporting.visualizations import save_hw3_charts
from market_terminal.backtest.engine import (
    BacktestConfig,
    backtest_ml_long_only_signal,
    backtest_buy_and_hold,
    extract_round_trips_from_result,
    build_backtest_comparison_frame,
)
from market_terminal.backtest.metrics import (
    build_hw3_performance_table,
    print_hw3_performance_summary,
    format_hw3_metrics_for_console,
)


def get_env_value(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


def get_alpaca_credentials() -> tuple[str, str]:
    if load_dotenv is not None:
        load_dotenv()

    api_key = get_env_value(
        "ALPACA_API_KEY",
        "ALPACA_API_KEY_ID",
        "APCA_API_KEY_ID",
    )

    api_secret = get_env_value(
        "ALPACA_API_SECRET",
        "ALPACA_SECRET_KEY",
        "ALPACA_API_SECRET_KEY",
        "APCA_API_SECRET_KEY",
    )

    if not api_key or not api_secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Add them to .env using one of these formats:\n"
            "ALPACA_API_KEY=your_key\n"
            "ALPACA_API_SECRET=your_secret\n\n"
            "or:\n"
            "APCA_API_KEY_ID=your_key\n"
            "APCA_API_SECRET_KEY=your_secret"
        )

    return api_key, api_secret


def years_ago(dt: datetime, years: int) -> datetime:
    try:
        return dt.replace(year=dt.year - years)
    except ValueError:
        return dt.replace(year=dt.year - years, month=2, day=28)


def parse_date_arg(date_string: str | None, default: datetime, end_of_day: bool = False) -> datetime:
    if date_string is None:
        return default

    parsed_date = datetime.fromisoformat(date_string)

    if parsed_date.tzinfo is None:
        if "T" not in date_string and end_of_day:
            parsed_date = datetime.combine(
                parsed_date.date(),
                time(hour=23, minute=59, second=59),
                tzinfo=timezone.utc,
            )
        else:
            parsed_date = parsed_date.replace(tzinfo=timezone.utc)

    return parsed_date


def download_daily_ohlcv_from_alpaca(
    symbol: str,
    start: datetime,
    end: datetime,
    feed: str = "iex",
) -> pd.DataFrame:
    """
    Downloads fresh daily OHLCV data from Alpaca.

    feed options:
    - "iex": free real-time IEX feed
    - "sip": full-market SIP feed, but recent data may require a paid subscription

    Output columns:
    - open
    - high
    - low
    - close
    - volume
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
            f"   python run_hw3_ml_backtest.py --symbol {symbol} --feed iex\n\n"
            "2. Or use SIP with an end time at least 15–20 minutes behind current time:\n"
            f"   python run_hw3_ml_backtest.py --symbol {symbol} --feed sip --data-delay-minutes 20\n\n"
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

def save_pca_summary(fitted_pca, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for i, explained in enumerate(fitted_pca.explained_variance_ratio, start=1):
        rows.append(
            {
                "component": f"PC{i}",
                "explained_variance_ratio": explained,
                "cumulative_variance_ratio": fitted_pca.cumulative_variance_ratio[i - 1],
            }
        )

    pd.DataFrame(rows).to_csv(output_path, index=False)


def run_hw3_pipeline(args: argparse.Namespace) -> None:
    symbol = args.symbol.upper()
    safe_symbol = symbol.lower()

    now_utc = datetime.now(timezone.utc)

    # Avoid Alpaca recent SIP restriction by default.
    # The Basic/free plan can error if we request SIP data too close to real time.
    safe_default_end = now_utc - timedelta(minutes=args.data_delay_minutes)

    end = parse_date_arg(
        args.end,
        default=safe_default_end,
        end_of_day=True,
    )

    # If using SIP, automatically cap end time so it is not too recent.
    if args.feed.lower() == "sip":
        latest_allowed_end = now_utc - timedelta(minutes=args.data_delay_minutes)

        if end > latest_allowed_end:
            print(
                "\nRequested SIP end time is too recent for many Alpaca accounts."
            )
            print(f"Original end: {end}")
            print(f"Adjusted end: {latest_allowed_end}")
            end = latest_allowed_end

    start = parse_date_arg(
        args.start,
        default=years_ago(end, args.years),
    )

    if start >= end:
        raise ValueError("Start date must be before end date.")

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(args.output_root) / f"{symbol}_{run_timestamp}"
    data_dir = output_dir / "data"
    chart_dir = output_dir / "charts"
    report_dir = output_dir / "reports"
    artifact_dir = output_dir / "artifacts"

    data_dir.mkdir(parents=True, exist_ok=True)
    chart_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print("\nHW3 ML Backtest")
    print("-" * 40)
    print(f"Symbol: {symbol}")
    print(f"Start: {start}")
    print(f"End: {end}")
    print(f"Output directory: {output_dir}")

    # Download fresh OHLCV data

    df = download_daily_ohlcv_from_alpaca(
        symbol=symbol,
        start=start,
        end=end,
        feed=args.feed
    )

    raw_data_path = data_dir / f"{safe_symbol}_daily_ohlcv.csv"
    df.to_csv(raw_data_path)

    print("\nDownloaded data:")
    print(df.head())
    print("\nDownloaded data shape:")
    print(df.shape)

    print("\nSaved downloaded OHLCV data to:")
    print(raw_data_path)

    # Build ML dataset

    ml_data = get_clean_ml_dataset(df)

    if len(ml_data) < 300:
        raise ValueError(
            f"Only {len(ml_data)} ML-ready rows after feature engineering. "
            "Use a longer date range."
        )

    ml_dataset_path = data_dir / f"hw3_{safe_symbol}_ml_dataset.csv"
    ml_data.to_csv(ml_dataset_path)

    X = ml_data[FEATURE_COLUMNS]
    y = ml_data["target"]

    print("\nML dataset shape:")
    print(ml_data.shape)

    print("\nFeature count:")
    print(len(FEATURE_COLUMNS))

    print("\nSaved ML dataset to:")
    print(ml_dataset_path)

    feature_columns_path = report_dir / f"hw3_{safe_symbol}_feature_columns.csv"
    pd.Series(FEATURE_COLUMNS, name="feature").to_csv(feature_columns_path, index=False)

    # Time-series split

    X_train, X_test, y_train, y_test = time_train_test_split(
        X=X,
        y=y,
        test_size=args.test_size,
    )

    print("\nTrain shape:")
    print(X_train.shape)

    print("\nTest shape:")
    print(X_test.shape)

    # PCA

    X_train_pca, fitted_pca = fit_pca(
        X_train,
        variance_threshold=args.pca_variance,
    )

    X_test_pca = transform_pca(
        X_test,
        fitted_pca,
    )

    print_pca_summary(fitted_pca)

    train_pca_path = data_dir / f"hw3_{safe_symbol}_train_pca.csv"
    test_pca_path = data_dir / f"hw3_{safe_symbol}_test_pca.csv"
    pca_summary_path = report_dir / f"hw3_{safe_symbol}_pca_summary.csv"

    X_train_pca.to_csv(train_pca_path)
    X_test_pca.to_csv(test_pca_path)
    save_pca_summary(fitted_pca, pca_summary_path)

    print("\nSaved PCA summary to:")
    print(pca_summary_path)

    # Train model

    model = train_random_forest(
        X_train=X_train_pca,
        y_train=y_train,
    )

    print_model_summary(
        model=model,
        X_test=X_test_pca,
        y_test=y_test,
        symbol=symbol,
    )

    # Generate probabilities and signals

    probabilities = predict_up_probability(
        model=model,
        X=X_test_pca,
    )

    signals = probability_to_signal(
        probabilities=probabilities,
        threshold=args.threshold,
    )

    test_data = ml_data.loc[X_test.index].copy()
    test_data["ml_probability"] = probabilities
    test_data["ml_signal"] = signals

    print("\nTest data preview:")
    print(test_data[["close", "ml_probability", "ml_signal"]].head())

    print("\nSignal counts:")
    print(test_data["ml_signal"].value_counts())

    test_data_path = data_dir / f"hw3_{safe_symbol}_test_data_with_ml_signals.csv"
    test_data.to_csv(test_data_path)

    # Backtest ML Signal and Buy & Hold

    config = BacktestConfig(
        initial_capital=args.initial_capital,
        commission_per_trade=args.commission,
        allow_fractional_shares=args.allow_fractional_shares,
    )

    ml_result = backtest_ml_long_only_signal(
        df=test_data,
        signal_col="ml_signal",
        name="ML Signal",
        config=config,
    )

    buy_hold_result = backtest_buy_and_hold(
        df=test_data,
        name="Buy & Hold",
        config=config,
    )

    round_trips = extract_round_trips_from_result(
        result=ml_result,
    )

    comparison_df = build_backtest_comparison_frame(
        ml_result=ml_result,
        buy_hold_result=buy_hold_result,
    )

    comparison_path = data_dir / f"hw3_{safe_symbol}_backtest_comparison.csv"
    round_trips_path = data_dir / f"hw3_{safe_symbol}_round_trips.csv"
    ml_trades_path = data_dir / f"hw3_{safe_symbol}_ml_raw_trades.csv"
    buy_hold_trades_path = data_dir / f"hw3_{safe_symbol}_buy_hold_raw_trades.csv"

    comparison_df.to_csv(comparison_path)
    round_trips.to_csv(round_trips_path, index=False)
    ml_result.trades.to_csv(ml_trades_path)
    buy_hold_result.trades.to_csv(buy_hold_trades_path)

    print("\nRound trips preview:")
    print(round_trips.head())

    # Performance metrics

    metrics = build_hw3_performance_table(
        comparison_df=comparison_df,
        round_trips=round_trips,
        initial_capital=args.initial_capital,
    )

    print_hw3_performance_summary(metrics)

    formatted_metrics = format_hw3_metrics_for_console(metrics)

    metrics_path = report_dir / f"hw3_{safe_symbol}_performance_metrics.csv"
    formatted_metrics_path = report_dir / f"hw3_{safe_symbol}_performance_metrics_formatted.csv"

    metrics.to_csv(metrics_path)
    formatted_metrics.to_csv(formatted_metrics_path)

    print("\nSaved raw metrics to:")
    print(metrics_path)

    print("\nSaved formatted metrics to:")
    print(formatted_metrics_path)

    # Save charts

    chart_paths = save_hw3_charts(
        test_data=test_data,
        comparison_df=comparison_df,
        fitted_pca=fitted_pca,
        chart_dir=chart_dir,
        symbol=symbol,
        threshold=args.threshold,
    )

    print("\nSaved HW3 charts:")
    for path in chart_paths:
        print(path)

    # Save model/PCA artifact bundle for paper trading

    model_bundle = {
        "symbol": symbol,
        "model": model,
        "fitted_pca": fitted_pca,
        "feature_columns": FEATURE_COLUMNS,
        "probability_threshold": args.threshold,
        "pca_variance_threshold": args.pca_variance,
        "train_start": str(X_train.index.min()),
        "train_end": str(X_train.index.max()),
        "test_start": str(X_test.index.min()),
        "test_end": str(X_test.index.max()),
    }

    model_bundle_path = artifact_dir / f"hw3_{safe_symbol}_model_bundle.joblib"
    joblib.dump(model_bundle, model_bundle_path)

    run_config = {
        "symbol": symbol,
        "start": str(start),
        "end": str(end),
        "feed": args.feed,
        "data_delay_minutes": args.data_delay_minutes,
        "years": args.years,
        "test_size": args.test_size,
        "pca_variance": args.pca_variance,
        "threshold": args.threshold,
        "initial_capital": args.initial_capital,
        "commission": args.commission,
        "allow_fractional_shares": args.allow_fractional_shares,
        "output_dir": str(output_dir),
        "model_bundle_path": str(model_bundle_path),
    }

    run_config_path = report_dir / f"hw3_{safe_symbol}_run_config.json"

    with open(run_config_path, "w", encoding="utf-8") as file:
        json.dump(run_config, file, indent=2)

    print("\nSaved model/PCA bundle to:")
    print(model_bundle_path)

    print("\nSaved run config to:")
    print(run_config_path)

    # summary

    print("\nFinal Summary")
    print("-" * 40)
    print(f"Symbol: {symbol}")
    print(f"Selected PCA components: {fitted_pca.n_components}")
    print(f"Total PCA explained variance: {fitted_pca.explained_variance_ratio.sum():.2%}")
    print(f"ML final value: ${comparison_df['portfolio_value'].iloc[-1]:,.2f}")
    print(f"Buy & Hold final value: ${comparison_df['buy_hold_value'].iloc[-1]:,.2f}")
    print(f"ML round trips: {len(round_trips)}")
    print(f"ML long-signal days: {int(test_data['ml_signal'].sum())}")
    print(f"Output directory: {output_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run HW3 ML trading pipeline with fresh Alpaca OHLCV data.",
    )

    parser.add_argument(
        "--symbol",
        type=str,
        required=True,
        help="Ticker symbol to backtest, for example AAPL, MSFT, NVDA, SPY.",
    )

    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Optional start date, for example 2021-07-01. Defaults to --years before end.",
    )

    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="Optional end date, for example 2026-07-01. Defaults to current UTC time.",
    )

    parser.add_argument(
        "--feed",
        type=str,
        default="sip",
        choices=["iex", "sip"],
        help="Alpaca market data feed. Use 'iex' for free real-time IEX data or 'sip' for delayed/full-market SIP data.",
    )

    parser.add_argument(
        "--data-delay-minutes",
        type=int,
        default=20,
        help="Minutes to subtract from current time to avoid recent SIP data restrictions.",
    )

    parser.add_argument(
        "--years",
        type=int,
        default=5,
        help="Number of years of data to download when --start is not provided.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Fraction of ML-ready data used for testing.",
    )

    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.80,
        help="Minimum cumulative PCA explained variance threshold.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Long signal threshold. Long if probability > threshold.",
    )

    parser.add_argument(
        "--initial-capital",
        type=float,
        default=100_000,
        help="Initial capital for the backtest.",
    )

    parser.add_argument(
        "--commission",
        type=float,
        default=0.0,
        help="Commission per trade.",
    )

    parser.add_argument(
        "--allow-fractional-shares",
        action="store_true",
        help="Allow fractional share sizing in the backtest.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root folder where run outputs are saved.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_hw3_pipeline(args)


if __name__ == "__main__":
    main()