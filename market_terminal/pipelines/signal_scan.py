from __future__ import annotations

import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from market_terminal.core.time_utils import years_ago
from market_terminal.data.alpaca_historical import download_daily_ohlcv_from_alpaca
from market_terminal.features.feature_engineering import (
    FEATURE_COLUMNS,
    add_ml_features,
    get_clean_ml_dataset,
)
from market_terminal.features.pca import fit_pca, transform_pca
from market_terminal.strategy.ml_model import (
    predict_up_probability,
    time_train_test_split,
    train_random_forest,
)

DEFAULT_SYMBOLS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMD",
    "TSLA",
    "META",
    "AMZN",
    "GOOGL",
    "AVGO",
    "QQQ",
    "SPY",
    "SMH",
    "NFLX",
    "CRM",
    "ORCL",
    "JPM",
]


def scan_symbol(
    symbol: str,
    years: int,
    feed: str,
    data_delay_minutes: int,
    threshold: float,
    test_size: float,
    pca_variance: float,
) -> dict:
    now_utc = datetime.now(timezone.utc)
    end = now_utc - timedelta(minutes=data_delay_minutes)
    start = years_ago(end, years)

    df = download_daily_ohlcv_from_alpaca(
        symbol=symbol,
        start=start,
        end=end,
        feed=feed,
    )

    ml_data = get_clean_ml_dataset(df)

    if len(ml_data) < 300:
        raise ValueError(f"Not enough ML-ready rows for {symbol}: {len(ml_data)}")

    X = ml_data[FEATURE_COLUMNS]
    y = ml_data["target"]

    X_train, X_test, y_train, y_test = time_train_test_split(
        X=X,
        y=y,
        test_size=test_size,
    )

    X_train_pca, fitted_pca = fit_pca(
        X_train,
        variance_threshold=pca_variance,
    )

    X_test_pca = transform_pca(
        X_test,
        fitted_pca,
    )

    model = train_random_forest(
        X_train=X_train_pca,
        y_train=y_train,
    )

    test_predictions = model.predict(X_test_pca)
    test_accuracy = accuracy_score(y_test, test_predictions)

    # Build the latest inference row.
    # Use add_ml_features instead of get_clean_ml_dataset because the latest row
    # does not need a known target.
    feature_data = add_ml_features(df)
    feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
    feature_data = feature_data.dropna(subset=FEATURE_COLUMNS).copy()

    latest_row = feature_data.tail(1).copy()

    X_latest = latest_row[FEATURE_COLUMNS]
    X_latest_pca = transform_pca(
        X_latest,
        fitted_pca,
    )

    latest_probability = float(
        predict_up_probability(
            model=model,
            X=X_latest_pca,
        ).iloc[-1]
    )

    latest_signal = int(latest_probability > threshold)

    latest_timestamp = latest_row.index[-1]
    latest_close = float(latest_row["close"].iloc[-1])

    return {
        "symbol": symbol,
        "latest_timestamp": latest_timestamp,
        "latest_close": latest_close,
        "latest_probability": latest_probability,
        "threshold": threshold,
        "latest_signal": latest_signal,
        "desired_state": "LONG" if latest_signal == 1 else "FLAT",
        "test_accuracy": test_accuracy,
        "pca_components": fitted_pca.n_components,
        "pca_explained_variance": float(fitted_pca.explained_variance_ratio.sum()),
        "ml_rows": len(ml_data),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }


def run_signal_scan_pipeline(args: argparse.Namespace) -> None:
    symbols = [symbol.upper().strip() for symbol in args.symbols]

    print("\nHW3 Latest Signal Scan")
    print("-" * 40)
    print(f"Feed: {args.feed}")
    print(f"Data delay minutes: {args.data_delay_minutes}")
    print(f"Threshold: {args.threshold}")
    print(f"Symbols: {', '.join(symbols)}")

    rows = []

    for symbol in symbols:
        print(f"\nScanning {symbol}...")

        try:
            result = scan_symbol(
                symbol=symbol,
                years=args.years,
                feed=args.feed,
                data_delay_minutes=args.data_delay_minutes,
                threshold=args.threshold,
                test_size=args.test_size,
                pca_variance=args.pca_variance,
            )

            rows.append(result)

            print(
                f"{symbol}: probability={result['latest_probability']:.4f}, "
                f"signal={result['latest_signal']}, "
                f"state={result['desired_state']}, "
                f"accuracy={result['test_accuracy']:.4f}"
            )

        except Exception as exc:
            print(f"{symbol}: FAILED")
            print(exc)

            rows.append(
                {
                    "symbol": symbol,
                    "latest_timestamp": None,
                    "latest_close": None,
                    "latest_probability": None,
                    "threshold": args.threshold,
                    "latest_signal": None,
                    "desired_state": "ERROR",
                    "test_accuracy": None,
                    "pca_components": None,
                    "pca_explained_variance": None,
                    "ml_rows": None,
                    "train_rows": None,
                    "test_rows": None,
                    "error": str(exc),
                }
            )

    results = pd.DataFrame(rows)

    if "latest_probability" in results.columns:
        results = results.sort_values(
            by="latest_probability",
            ascending=False,
            na_position="last",
        )

    output_dir = Path(args.output_root) / "signal_scans"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"hw3_signal_scan_{timestamp}.csv"

    results.to_csv(output_path, index=False)

    print("\nRanked Signal Scan Results")
    print("-" * 40)

    display_columns = [
        "symbol",
        "latest_probability",
        "latest_signal",
        "desired_state",
        "latest_close",
        "test_accuracy",
        "pca_components",
        "pca_explained_variance",
    ]

    print(results[display_columns].to_string(index=False))

    long_candidates = results[results["latest_signal"] == 1].copy()

    print("\nLong Candidates")
    print("-" * 40)

    if long_candidates.empty:
        print("No symbols crossed the 0.60 long threshold.")
        print("Do not lower the threshold just to force a trade.")
        print("Better options: scan a broader universe or wait for a valid long signal.")
    else:
        print(
            long_candidates[
                [
                    "symbol",
                    "latest_probability",
                    "latest_close",
                    "test_accuracy",
                    "pca_explained_variance",
                ]
            ].to_string(index=False)
        )

        best_symbol = long_candidates.iloc[0]["symbol"]

        print("\nSuggested next step")
        print("-" * 40)
        print(f"Run a full backtest/model-bundle generation for {best_symbol}:")
        print(f"python run_hw3_ml_backtest.py --symbol {best_symbol}")
        print()
        print("Then run the paper-trading dry run using the new model bundle.")

    print("\nSaved scan results to:")
    print(output_path)

