from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import joblib
import pandas as pd

from market_terminal.backtest.engine import (
    BacktestConfig,
    backtest_buy_and_hold,
    backtest_ml_long_only_signal,
    build_backtest_comparison_frame,
    extract_round_trips_from_result,
)
from market_terminal.backtest.metrics import (
    build_ml_performance_table,
    format_ml_metrics_for_console,
    print_ml_performance_summary,
)
from market_terminal.core.time_utils import parse_date_arg, years_ago
from market_terminal.data.alpaca_historical import download_daily_ohlcv_from_alpaca
from market_terminal.features.feature_engineering import FEATURE_COLUMNS, get_clean_ml_dataset
from market_terminal.features.pca import fit_pca, print_pca_summary, transform_pca
from market_terminal.reporting.visualizations import save_ml_strategy_charts
from market_terminal.strategy.ml_model import (
    predict_up_close_probability,
    print_model_summary,
    probability_to_signal,
    time_train_test_split,
    train_random_forest,
)


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


def run_ml_backtest_pipeline(args: argparse.Namespace) -> None:
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

    print("\nMachine-Learning Strategy Backtest")
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

    ml_dataset_path = data_dir / f"ml_{safe_symbol}_dataset.csv"
    ml_data.to_csv(ml_dataset_path)

    X = ml_data[FEATURE_COLUMNS]
    y = ml_data["target"]

    print("\nML dataset shape:")
    print(ml_data.shape)

    print("\nFeature count:")
    print(len(FEATURE_COLUMNS))

    print("\nSaved ML dataset to:")
    print(ml_dataset_path)

    feature_columns_path = report_dir / f"ml_{safe_symbol}_feature_columns.csv"
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

    train_pca_path = data_dir / f"ml_{safe_symbol}_train_pca.csv"
    test_pca_path = data_dir / f"ml_{safe_symbol}_test_pca.csv"
    pca_summary_path = report_dir / f"ml_{safe_symbol}_pca_summary.csv"

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

    probabilities = predict_up_close_probability(model=model, X=X_test_pca)

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

    test_data_path = data_dir / f"ml_{safe_symbol}_test_data_with_signals.csv"
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

    comparison_path = data_dir / f"ml_{safe_symbol}_backtest_comparison.csv"
    round_trips_path = data_dir / f"ml_{safe_symbol}_round_trips.csv"
    ml_trades_path = data_dir / f"ml_{safe_symbol}_strategy_raw_trades.csv"
    buy_hold_trades_path = data_dir / f"ml_{safe_symbol}_buy_hold_raw_trades.csv"

    comparison_df.to_csv(comparison_path)
    round_trips.to_csv(round_trips_path, index=False)
    ml_result.trades.to_csv(ml_trades_path)
    buy_hold_result.trades.to_csv(buy_hold_trades_path)

    print("\nRound trips preview:")
    print(round_trips.head())

    # Performance metrics

    metrics = build_ml_performance_table(comparison_df=comparison_df, round_trips=round_trips,
                                         initial_capital=args.initial_capital)

    print_ml_performance_summary(metrics)

    formatted_metrics = format_ml_metrics_for_console(metrics)

    metrics_path = report_dir / f"ml_{safe_symbol}_performance_metrics.csv"
    formatted_metrics_path = report_dir / f"ml_{safe_symbol}_performance_metrics_formatted.csv"

    metrics.to_csv(metrics_path)
    formatted_metrics.to_csv(formatted_metrics_path)

    print("\nSaved raw metrics to:")
    print(metrics_path)

    print("\nSaved formatted metrics to:")
    print(formatted_metrics_path)

    # Save charts

    chart_paths = save_ml_strategy_charts(test_data=test_data, comparison_df=comparison_df, fitted_pca=fitted_pca,
                                          chart_dir=chart_dir, symbol=symbol, threshold=args.threshold)

    print("\nSaved ML charts:")
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

    model_bundle_path = artifact_dir / f"ml_{safe_symbol}_model_bundle.joblib"
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

    run_config_path = report_dir / f"ml_{safe_symbol}_run_config.json"

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
