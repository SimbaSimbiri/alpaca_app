# Market Data Terminal + Systematic Trading Platform

A Python market data, backtesting, machine-learning, Alpaca paper-trading, and monitoring platform built with Alpaca, Pandas, NumPy, scikit-learn, Matplotlib, PyYAML, and Tkinter.

The project combines seven connected workflows:

1. **Market Data Terminal**  
   A desktop Tkinter application that connects to Alpaca, downloads historical OHLCV data, streams live bid/ask quotes and last trade prices, logs live quote/trade events, and displays candlestick-style charts.

2. **Technical Indicator Backtesting Engine**  
   A reusable long-only backtesting workflow that downloads historical daily OHLCV data from Alpaca, computes technical indicators, compares multiple strategy rules, generates charts, calculates performance metrics, and produces a PDF report.

3. **Machine-Learning Strategy Pipeline**  
   A machine-learning workflow that downloads fresh Alpaca daily OHLCV data, engineers technical-indicator features, applies PCA, trains a Random Forest classifier, generates probability-based long/flat signals, backtests the ML strategy, compares it against Buy & Hold, and saves reusable model artifacts.

4. **Signal Scanner**  
   A multi-symbol scanner that trains a fresh ML model per ticker, computes the latest probability of a positive next-day return, ranks symbols, and identifies current long candidates.

5. **One-Shot Alpaca Paper-Trading Decision Tool**  
   A paper-trading workflow that loads a saved ML model bundle, rebuilds the latest feature row from fresh market data, generates the current ML signal, checks the current Alpaca paper position, applies risk controls, optionally submits a paper market order, and saves a lifecycle log.

6. **Live Paper-Trading Engine**  
   A repeated engine loop that evaluates one or more saved ML model bundles, generates fresh signals, turns them into paper-trading decisions, applies the risk manager, optionally submits approved paper orders, and logs account/order/position snapshots.

7. **System Monitor UI**  
   A lightweight Tkinter monitor that reads generated runtime logs, displays the latest signal, action, risk result, order state, position snapshot paths, and provides start/stop controls for the live paper-trading engine.

This is a research and educational trading system. It uses Alpaca paper trading only. It is not financial advice and is not intended for live-money trading.

---

## Core System Flow

```text
Market Data
   ↓
Feature Engineering / Technical Indicators
   ↓
Strategy Logic
   ↓
Signal Generation
   ↓
Backtest or Paper-Trading Decision
   ↓
Risk Management
   ↓
Order Simulation or Alpaca Paper Order
   ↓
Lifecycle Logs, Account Snapshots, Metrics, Charts, Reports, UI Monitoring
```

The package structure separates concerns so that data access, features, strategy logic, backtesting, execution, risk checks, reporting, runtime pipelines, and UI code live in different modules.

---

## Features

### Market Data Terminal

- Authenticates with Alpaca paper-trading credentials
- Loads API keys securely from a local `.env` file
- Downloads historical OHLCV bars
- Supports user-selected timeframes such as minute, hour, day, week, and month
- Displays candlestick-style OHLCV charts with volume bars
- Shows recent OHLCV rows in a table
- Streams live bid/ask quotes
- Streams last trade prices
- Updates the UI through an event queue
- Logs incoming live quote and trade events to structured CSV files under `outputs/live_data/`
- Uses a thin root launcher with the actual UI implementation inside `market_terminal/ui/`

### Technical Indicator Backtesting

- Downloads 5+ years of daily OHLCV data from Alpaca
- Includes a sample-data mode for local testing without credentials
- Computes trend, momentum, volatility, and volume indicators
- Builds long-only strategy signals
- Executes signals at the next day’s open to reduce close-price look-ahead bias
- Simulates cash, shares, holdings, equity, trades, and drawdown
- Compares strategies against Buy & Hold
- Exports trades, portfolios, metrics, charts, and PDF reports

### Machine-Learning Strategy Pipeline

- Downloads fresh Alpaca daily OHLCV data for any user-selected symbol
- Supports Alpaca IEX or SIP data feeds
- Uses a delayed end time by default to avoid requesting too-recent SIP data
- Engineers 22 normalized market features
- Standardizes features before PCA
- Keeps the minimum number of principal components needed to explain at least 80% of feature variance
- Trains a Random Forest classifier
- Predicts whether the next daily return will be positive
- Converts predicted probability into a long/flat signal
- Uses the rule: **Long if probability > 0.60, otherwise Flat**
- Backtests the ML signal against Buy & Hold
- Saves model bundles for later paper-trading inference

### Signal Scanner

- Scans a universe of tickers
- Trains a fresh model per symbol
- Computes the latest probability of a positive next-day return
- Ranks symbols by model probability
- Identifies current long candidates based on the configured probability threshold
- Saves scan results to CSV

### Paper Trading

- Loads a saved ML model bundle
- Downloads fresh Alpaca market data
- Rebuilds the latest feature row
- Applies the saved scaler and PCA transformer
- Generates the latest probability and long/flat signal
- Converts the model signal into a typed `PaperTradeDecision`
- Checks the current Alpaca paper position
- Decides whether the account should BUY, SELL, or HOLD
- Applies a risk-management gate before paper orders are submitted
- Blocks oversized orders based on `--max-order-qty`
- Blocks sell orders that would create a short position
- Can enforce minimum buying power after a buy order
- Submits a paper order only when explicitly run with `--execute`
- Records structured lifecycle events for signal generation, decision building, risk checks, account snapshots, and order submission outcomes
- Saves decision logs for review and auditing

Each paper-trading log includes a `lifecycle_events` array that shows how the system moved from model signal to paper-trading decision, risk approval, account snapshotting, and final order outcome.

### Live Paper-Trading Engine

- Runs repeated signal-to-risk-to-order cycles for one or more saved ML model bundles
- Supports dry-run mode by default
- Submits paper orders only when `--execute` is explicitly provided
- Supports finite cycles with `--cycles`
- Supports continuous polling with `--continuous`
- Supports configurable polling interval with `--polling-interval-seconds`
- Uses the same typed decision objects, risk manager, Alpaca broker wrapper, and lifecycle logging as the one-shot paper-trading workflow
- Saves order and position snapshots after each cycle

### Order, Position, and Account Monitoring

- Wraps Alpaca paper account access in `AlpacaBroker`
- Retrieves current positions
- Retrieves recent paper orders
- Serializes order states such as submitted, filled, canceled, expired, failed, and replaced when available from Alpaca
- Saves recent-order snapshots to `outputs/paper_trading/paper_orders_YYYYMMDD.csv`
- Saves current-position snapshots to `outputs/paper_trading/paper_positions_YYYYMMDD.csv`
- Keeps generated account snapshots out of Git

### Configuration

- Uses `.env` for Alpaca credentials
- Supports YAML configuration through `config/config.example.yaml`
- Keeps local `config/config.yaml` out of Git
- Supports configurable ticker universe, strategy parameters, risk limits, engine polling settings, model bundle paths, order size, and execution mode

### System Monitor UI

- Provides a second Tkinter interface for runtime monitoring
- Reads the latest paper-trading JSON logs
- Reads the latest order and position snapshot paths
- Displays latest symbol, close, probability, threshold, signal, desired state, action, quantity, risk approval, and lifecycle stage
- Includes refresh and auto-refresh controls
- Includes a command box for launching the live paper-trading engine
- Includes Start Engine and Stop Engine controls
- Writes UI-launched engine process logs under `outputs/ui_engine_control/`

---

## Tech Stack

- Python
- Alpaca-py
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- Tkinter
- python-dotenv
- PyYAML
- joblib

---

## Project Structure

```text
alpaca_app/
│
├── app.py                         # Thin launcher for the Tkinter market terminal
├── run_backtest.py                # CLI entrypoint for technical-indicator backtests
├── run_ml_backtest.py             # CLI entrypoint for ML strategy backtests
├── run_signal_scan.py             # CLI entrypoint for latest-signal scanning
├── run_paper_trade.py             # CLI entrypoint for one-shot Alpaca paper-trading decisions
├── run_live_paper_engine.py       # CLI entrypoint for repeated live paper-trading cycles
├── run_system_monitor.py          # CLI entrypoint for the system monitor UI
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   └── config.example.yaml        # Example runtime config; copy to config/config.yaml locally
│
├── market_terminal/
│   ├── __init__.py
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── constants.py           # Shared constants and defaults
│   │   ├── settings.py            # Environment variable and Alpaca credential loading
│   │   ├── system_config.py       # YAML runtime config loading and typed config objects
│   │   ├── types.py               # Shared model signal, decision, and lifecycle event types
│   │   └── time_utils.py          # Shared date/time helpers
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── alpaca_historical.py   # Shared Alpaca historical OHLCV downloader
│   │   ├── alpaca_connector.py    # Alpaca historical data connector used by UI/backtests
│   │   ├── alpaca_live.py         # Alpaca websocket quote/trade stream for the UI
│   │   └── quote_logger.py        # CSV logger for incoming live quote/trade events
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── indicators.py          # SMA, EMA, MACD, ADX, RSI, Bollinger, ATR, OBV, CMF
│   │   ├── feature_engineering.py # Machine-learning feature engineering
│   │   └── pca.py                 # StandardScaler + PCA fitting/transform logic
│   │
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── technical_strategies.py # Technical-indicator strategy signal rules
│   │   └── ml_model.py             # Random Forest model + probability signal logic
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py              # Reusable long-only backtesting engine
│   │   └── metrics.py             # Risk/return metrics and performance summaries
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── alpaca_broker.py       # Alpaca paper account, position, and order wrapper
│   │   └── account_snapshot_logger.py # CSV logger for order and position snapshots
│   │
│   ├── risk/
│   │   ├── __init__.py
│   │   └── risk_manager.py        # Paper-trading risk checks before order submission
│   │
│   ├── reporting/
│   │   ├── __init__.py
│   │   ├── visualizations.py      # Backtest and ML charts
│   │   └── report.py              # PDF report generator
│   │
│   ├── pipelines/
│   │   ├── __init__.py
│   │   ├── indicator_backtest.py  # Technical-indicator backtest orchestration
│   │   ├── ml_backtest.py         # ML strategy pipeline orchestration
│   │   ├── signal_scan.py         # Multi-symbol signal scan orchestration
│   │   ├── paper_trade.py         # One-shot paper-trading decision orchestration
│   │   └── live_paper_engine.py   # Repeated signal-to-risk-to-order engine loop
│   │
│   └── ui/
│       ├── __init__.py
│       ├── dashboard.py           # Tkinter market terminal UI
│       ├── monitor_state.py       # Reads latest generated logs/snapshots for monitoring
│       ├── system_monitor.py      # Tkinter system monitor UI
│       └── engine_process.py      # Starts/stops the live engine as a subprocess
│
├── data/
│   └── examples/                  # Small sanitized examples, if included
├── screenshots/                   # UI screenshots
├── charts/                        # Committed chart examples
├── reports/                       # Optional manually copied reports
└── outputs/                       # Auto-generated runtime outputs; usually ignored by Git
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/SimbaSimbiri/alpaca_app.git
cd alpaca_app
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## API Key Setup

This project requires Alpaca API credentials for real Alpaca data access and paper trading.

Create a local `.env` file in the project root. Do not commit `.env` to GitHub.

```env
ALPACA_API_KEY=replace_me
ALPACA_SECRET_KEY=replace_me
ALPACA_DATA_FEED=sip
```

The settings loader also accepts common Alpaca environment variable names:

```env
APCA_API_KEY_ID=replace_me
APCA_API_SECRET_KEY=replace_me
```

Recommended `.gitignore` entries:

```text
.env
config/config.yaml
outputs/
outputs/live_data/
outputs/paper_trading/
outputs/ui_engine_control/
__pycache__/
*.pyc
```

The ML backtest and paper-trading workflows default to Alpaca SIP data with a delayed end time. This avoids requesting very recent SIP data that may not be available on some Alpaca subscriptions.

---

## Runtime Configuration

The project supports YAML-based runtime configuration for the live paper-trading engine.

The committed example file is:

```text
config/config.example.yaml
```

For local runs, copy it to:

```text
config/config.yaml
```

`config/config.yaml` is ignored by Git and should be used for local/private settings such as model bundle paths, execution mode, ticker universe, risk limits, and engine polling settings.

Example config structure:

```yaml
universe:
  symbols:
    - SPY
    - AAPL
    - MSFT
    - QQQ
    - NVDA

strategy:
  probability_threshold: 0.60
  years: 5
  feed: sip
  data_delay_minutes: 20

risk:
  max_order_qty: 10
  min_buying_power_after_order: 0
  allow_short_selling: false

engine:
  model_bundles:
    - outputs/SPY_<timestamp>/artifacts/ml_spy_model_bundle.joblib

  qty: 1
  polling_interval_seconds: 300
  cycles: 1
  continuous: false
  execute_orders: false
  order_snapshot_limit: 50
```

CLI arguments will override config values when both are provided.

---

## Running the Market Terminal UI

Launch the terminal:

```bash
python app.py
```

Recommended test tickers:

```text
MSFT
AAPL
SPY
QQQ
NVDA
TSLA
```

### How to Use the Terminal

1. Start the app with `python app.py`.
2. Enter a ticker symbol.
3. Select a timeframe.
4. Click **Load Historical Data**.
5. Review the OHLCV candlestick chart and recent-bars table.
6. Click **Start Live Stream**.
7. Watch the bid, ask, last trade, quote time, and trade time update.
8. Click **Stop Stream** to stop the websocket stream.

### Market Hours Note

Historical bars can load outside regular market hours.

Live quote and trade updates are easiest to observe during regular U.S. market hours. Outside market hours, the websocket may connect successfully but receive few or no events depending on ticker, feed, and market activity.

---

## Running the Technical Indicator Backtest

Run with real Alpaca historical daily data:

```bash
python run_backtest.py --ticker MSFT --years 5
```

Other examples:

```bash
python run_backtest.py --ticker AAPL --years 5
python run_backtest.py --ticker SPY --years 5
python run_backtest.py --ticker QQQ --years 5
python run_backtest.py --ticker NVDA --years 5
```

Run with sample data and no Alpaca credentials:

```bash
python run_backtest.py --ticker SAMPLE --years 5 --sample
```

Add a commission or slippage assumption:

```bash
python run_backtest.py --ticker MSFT --years 5 --commission 1.00
```

Allow fractional shares:

```bash
python run_backtest.py --ticker MSFT --years 5 --fractional
```

### Technical Backtest Outputs

Each run creates a timestamped folder:

```text
outputs/MSFT_YYYYMMDD_HHMMSS/
├── data/
│   ├── MSFT_daily_ohlcv_indicators_signals.csv
│   ├── MSFT_performance_metrics.csv
│   ├── trend_following_portfolio.csv
│   ├── trend_following_trades.csv
│   └── ...
├── charts/
│   ├── trend_following_price_signals.png
│   ├── mean_reversion_price_signals.png
│   ├── custom_volume-confirmed_trend_pullback_price_signals.png
│   ├── equity_curve_comparison.png
│   └── drawdown_comparison.png
└── reports/
    └── MSFT_final_report.pdf
```

---

## Running the Machine-Learning Backtest

Run the ML strategy pipeline:

```bash
python run_ml_backtest.py --symbol SPY
```

The command downloads fresh daily OHLCV data, builds ML features, applies PCA, trains a Random Forest classifier, generates long/flat signals, backtests the ML strategy, compares it against Buy & Hold, saves charts, saves metrics, and saves a model/PCA bundle for paper trading.

The default feed is SIP. The default data delay is 20 minutes.

Equivalent explicit command:

```bash
python run_ml_backtest.py --symbol SPY --feed sip --data-delay-minutes 20
```

Use the IEX feed:

```bash
python run_ml_backtest.py --symbol SPY --feed iex
```

Run another symbol:

```bash
python run_ml_backtest.py --symbol AAPL
python run_ml_backtest.py --symbol MSFT
python run_ml_backtest.py --symbol QQQ
```

Use custom dates:

```bash
python run_ml_backtest.py --symbol SPY --start 2021-07-01 --end 2026-07-01
```

Allow fractional shares in the backtest:

```bash
python run_ml_backtest.py --symbol SPY --allow-fractional-shares
```

Adjust the probability threshold:

```bash
python run_ml_backtest.py --symbol SPY --threshold 0.60
```

### Machine-Learning Backtest Output

Each ML run creates a timestamped output folder:

```text
outputs/SPY_YYYYMMDD_HHMMSS/
│
├── data/
│   ├── spy_daily_ohlcv.csv
│   ├── ml_spy_train_pca.csv
│   ├── ml_spy_test_pca.csv
│   ├── ml_spy_test_data_with_ml_signals.csv
│   ├── ml_spy_backtest_comparison.csv
│   ├── ml_spy_round_trips.csv
│   ├── ml_spy_ml_raw_trades.csv
│   └── ml_spy_buy_hold_raw_trades.csv
│
├── charts/
│   ├── ml_spy_equity_curve.png
│   ├── ml_spy_drawdown.png
│   ├── ml_spy_pca_variance.png
│   ├── ml_spy_signals.png
│   └── ml_spy_probability_signal.png
│
├── reports/
│   ├── ml_spy_performance_metrics.csv
│   ├── ml_spy_performance_metrics_formatted.csv
│   ├── ml_spy_pca_summary.csv
│   ├── ml_spy_feature_columns.csv
│   └── ml_spy_run_config.json
│
└── artifacts/
    └── ml_spy_model_bundle.joblib
```

The internal output filenames now use the `ml_` prefix.

---

## Running the Signal Scanner

Run the default scan:

```bash
python run_signal_scan.py
```

Scan a custom list:

```bash
python run_signal_scan.py --symbols SPY AAPL MSFT QQQ NVDA TSLA META AMZN GOOGL
```

Scan a broader ETF universe:

```bash
python run_signal_scan.py --symbols SPY VOO IVV SPLG SPYM VTI ITOT SCHB IWB VV DIA RSP XLK XLF XLY XLV XLI XLC XLP XLE XLB XLU XLRE
```

The scanner prints:

- latest probability
- latest signal
- desired state: LONG or FLAT
- model test accuracy
- number of PCA components
- total explained variance

A symbol becomes a long candidate when:

```text
latest_probability > 0.60
```

Signal scan results are saved under:

```text
outputs/signal_scans/
```

---

## Running the Paper-Trading Workflow

The paper-trading workflow loads a saved model bundle and generates the latest trading decision.

### 1. Dry run first

A dry run is the default behavior. It prints the decision but does not submit an order.

```bash
python run_paper_trade.py --model-bundle outputs\SPY_<timestamp>\artifacts\ml_spy_model_bundle.joblib
```

Expected dry-run output includes:

```text
This is paper trading only — no real money is used.
Predicted probability of positive next-day return: ...
ML signal: 1
Desired state: LONG
Action: BUY
Dry Run: No paper order was submitted.
```

### 2. Execute a paper order

Only use `--execute` after reviewing the dry run.

```bash
python run_paper_trade.py --model-bundle outputs\SPY_<timestamp>\artifacts\ml_spy_model_bundle.joblib --qty 1 --execute
```

This submits a paper market order only. No real money is used.

### 3. Confirm the paper position

Run the same command again after the order has been submitted:

```bash
python run_paper_trade.py --model-bundle outputs\SPY_<timestamp>\artifacts\ml_spy_model_bundle.joblib --qty 1 --execute
```

When the model still wants LONG and the paper account is already long, the script should print:

```text
Action: HOLD
Reason: Model wants LONG and the paper account is already long.
```

### Paper-Trading Risk and Snapshot Options

Limit order size:

```bash
python run_paper_trade.py --model-bundle outputs\SPY_<timestamp>\artifacts\ml_spy_model_bundle.joblib --qty 1 --max-order-qty 10
```

Require minimum buying power after a buy order:

```bash
python run_paper_trade.py --model-bundle outputs\SPY_<timestamp>\artifacts\ml_spy_model_bundle.joblib --min-buying-power-after-order 1000
```

Control how many recent paper orders are included in the order snapshot:

```bash
python run_paper_trade.py --model-bundle outputs\SPY_<timestamp>\artifacts\ml_spy_model_bundle.joblib --order-snapshot-limit 50
```

After each run, the workflow attempts to save:

```text
outputs/paper_trading/paper_orders_YYYYMMDD.csv
outputs/paper_trading/paper_positions_YYYYMMDD.csv
```

### Paper-Trading Decision Logic

```text
Signal = 1 and no current position   -> BUY
Signal = 1 and already long          -> HOLD
Signal = 0 and currently long        -> SELL
Signal = 0 and no current position   -> HOLD
```

The paper-trading script is intentionally conservative. It does not submit an order unless `--execute` is provided.

---

## Running the Live Paper-Trading Engine

The live paper-trading engine repeatedly evaluates one or more saved ML model bundles, downloads fresh Alpaca data, generates latest signals, checks current paper positions, applies risk controls, optionally submits approved paper orders, and writes lifecycle logs.

Dry-run one cycle with a model bundle:

```bash
python run_live_paper_engine.py --model-bundles outputs\SPY_<timestamp>\artifacts\ml_spy_model_bundle.joblib --cycles 1
```

Run multiple model bundles:

```bash
python run_live_paper_engine.py --model-bundles outputs\SPY_<timestamp>\artifacts\ml_spy_model_bundle.joblib outputs\AAPL_<timestamp>\artifacts\ml_aapl_model_bundle.joblib --cycles 1
```

Run continuously in dry-run mode:

```bash
python run_live_paper_engine.py --model-bundles outputs\SPY_<timestamp>\artifacts\ml_spy_model_bundle.joblib --continuous --polling-interval-seconds 300
```

Run from YAML config:

```bash
python run_live_paper_engine.py --config config/config.yaml
```

Submit approved paper orders:

```bash
python run_live_paper_engine.py --model-bundles outputs\SPY_<timestamp>\artifacts\ml_spy_model_bundle.joblib --cycles 1 --execute
```

The engine is conservative by default. Without `--execute`, it generates signals, builds decisions, applies risk checks, saves account snapshots, and writes logs, but does not submit orders.

---

## Running the System Monitor UI

Launch the system monitor:

```bash
python run_system_monitor.py
```

The monitor reads generated files from `outputs/` and displays:

- latest paper-trading log
- latest symbol
- latest close
- latest ML probability
- latest signal and desired state
- latest BUY/SELL/HOLD action
- risk approval result
- latest lifecycle stage
- order snapshot path
- position snapshot path

The monitor also includes live-engine controls:

- command entry box
- Start Engine button
- Stop Engine button
- engine status label
- engine process log path

The default command is:

```bash
python run_live_paper_engine.py --config config/config.yaml --continuous
```

Only run this command after creating a valid local `config/config.yaml` with real model bundle paths. The UI starts the engine as a local subprocess and writes process logs under:

```text
outputs/ui_engine_control/
```

---

## Runtime Outputs and Logs

Generated runtime files are kept under `outputs/` and are usually ignored by Git.

Important output folders:

```text
outputs/live_data/
outputs/paper_trading/
outputs/ui_engine_control/
outputs/signal_scans/
outputs/<SYMBOL>_<timestamp>/
```

Live data logs:

```text
outputs/live_data/live_market_events_YYYYMMDD.csv
```

Paper account snapshots:

```text
outputs/paper_trading/paper_orders_YYYYMMDD.csv
outputs/paper_trading/paper_positions_YYYYMMDD.csv
```

Paper-trading lifecycle logs are written near the model bundle output folder and include:

```text
signal_generated
decision_built
risk_checked
account_snapshot_saved
dry_run / order_submitted / order_rejected / no_order_needed
```

These logs show the full path from model output to risk check to paper order decision.

---

## Indicators Implemented

### Trend Indicators

- Simple Moving Average, SMA
- Exponential Moving Average, EMA
- Moving Average Convergence Divergence, MACD
- Average Directional Index, ADX

### Momentum Indicators

- Momentum
- Relative Strength Index, RSI
- Stochastic Oscillator
- Williams %R

### Volatility Indicators

- Bollinger Bands
- Average True Range, ATR

### Volume Indicators

- On-Balance Volume, OBV
- Chaikin Money Flow, CMF

---

## Technical Indicator Strategies

### Buy & Hold

Baseline benchmark.

Entry:

- Buy at the first available open.

Exit:

- Hold until the end of the sample.

Purpose:

- Provides a passive benchmark so the active strategies can be judged against simply owning the asset.

---

### Strategy 1: Trend Following

This strategy attempts to participate when price is already trending upward and the trend has enough strength.

Entry:

- MACD > MACD signal
- ADX(14) > 25
- SMA(50) > SMA(200)
- Close > SMA(50)

Exit:

- MACD < MACD signal, or
- Close < SMA(50), or
- ADX(14) < 18

Main idea:

- MACD confirms bullish momentum.
- ADX filters for trend strength.
- SMA(50) and SMA(200) filter for broader trend direction.

---

### Strategy 2: Mean Reversion

This strategy looks for short-term oversold conditions where price may revert back toward its average.

Entry:

- RSI(14) < 30
- Close < lower Bollinger Band(20, 2)

Exit:

- RSI(14) > 70 and close > upper Bollinger Band, or
- Close reverts above the Bollinger middle band, or
- RSI(14) rises above 55

Main idea:

- RSI identifies oversold or overbought momentum.
- Bollinger Bands identify price stretched far from its recent average.
- The strategy buys weakness and exits after a bounce or recovery.

---

### Strategy 3: Custom Volume-Confirmed Trend Pullback

This custom strategy combines trend, momentum, volatility/location, and volume indicators.

Entry:

- Trend: close > SMA(200)
- Momentum: Momentum(10) > 0
- Volatility/location: close > Bollinger middle band
- Volume: OBV > OBV SMA(20)
- Volume: CMF(20) > 0

Exit:

- Close < SMA(50), or
- Momentum(10) < 0, or
- CMF(20) < 0, or
- RSI(14) > 75

Main idea:

- Trade only in a broad uptrend.
- Require positive momentum.
- Require price to be above its Bollinger middle band.
- Require volume confirmation using OBV and CMF.
- Exit when trend, momentum, or money flow weakens.

---

## Machine-Learning Model Design

### Target

The target is binary:

```text
1 = next-day return > 0
0 = next-day return <= 0
```

### Feature Set

The ML feature set includes 22 features across:

- returns and rolling statistics
- trend indicators
- momentum indicators
- volatility indicators
- volume indicators

### PCA

Features are standardized before PCA. The PCA step keeps the minimum number of components needed to explain at least 80% of feature variance.

### Classifier

The classifier is a Random Forest model trained on PCA-transformed features.

### Signal Rule

```text
Long if predicted probability > 0.60
Flat otherwise
```

The strategy is long-only. It does not short, does not use leverage, and does not trade unless the probability threshold is crossed.

---

## Backtesting Assumptions

- Initial capital: `$100,000` by default
- Long-only
- No leverage
- No short selling
- Whole-share position sizing by default
- Fractional share sizing is optional
- Close-based signals execute at the next day’s open to reduce look-ahead bias
- Default commission is zero
- Commission can be changed with the `--commission` flag
- Daily returns are computed from portfolio value changes
- Buy & Hold is included as the benchmark

---

## Performance Metrics

The reports and CSV metrics include:

- Ending Value
- Total Return
- CAGR
- Annualized Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Round Trips
- Number of trades
- Market exposure

Example comparison table:

```text
Strategy      Ending Value   Total Return   CAGR   Volatility   Sharpe   Sortino   Max Drawdown   Win Rate   Exposure
Buy & Hold
ML Signal
```

---

## Visualizations

The technical-indicator backtesting workflow generates:

- price and signal charts
- equity curve comparison
- drawdown comparison

The machine-learning workflow generates:

- equity curve comparison
- drawdown comparison
- PCA explained variance chart
- close price with ML long signals
- model probability signal chart

---

## SPY Example Evidence

SPY was used as the paper-trading example ticker. The committed SPY files include charts, metrics, and sanitized paper-trading logs.

### SPY Performance Metrics

```text
data/SPY/spy_performance_metrics.csv
```

### SPY Charts

```text
charts/SPY/spy_equity_curve.png
charts/SPY/spy_drawdown.png
charts/SPY/spy_ml_signals.png
charts/SPY/spy_pca_variance.png
charts/SPY/spy_probability_signal.png
```

### SPY Paper-Trade Transition Logs

```text
data/SPY/paper_trade_logs/spy_paper_trade_01_dry_run_buy_decision.json
data/SPY/paper_trade_logs/spy_paper_trade_02_paper_buy_order_submitted.json
data/SPY/paper_trade_logs/spy_paper_trade_03_position_confirmed_hold.json
data/SPY/paper_trade_logs/spy_paper_trade_transition_summary.json
```

The transition shown by these logs is:

```text
1. Dry run: the SPY model wanted LONG and the action would be BUY.
2. Execute run: the SPY model wanted LONG and a paper BUY order was submitted.
3. Follow-up run: the account was already long SPY, so the action changed to HOLD.
```
---

## Screenshots

### MSFT 4H Chart for the Last Two Years

![MSFT 4H OHLCV Chart for the last two years](screenshots/ui_running_3.png)

### MSFT Most Recent OHLCV Data

![MSFT Most Recent Data](screenshots/ui_running_4.png)

### NVDA 30m Chart Outside Trading Hours

![NVDA Chart Outside Trading Hours](screenshots/ui_running_1.png)

### NVDA Most Recent OHLCV Data

![NVDA Most Recent Data](screenshots/ui_running_2.png)

---

## Safety and Limitations

This project is for educational research and paper trading.

Limitations:

- No live-money trading is used.
- No tax modeling.
- No market impact modeling.
- No partial-fill modeling.
- No bid/ask spread modeling unless represented through a commission/slippage assumption.
- No leverage.
- No short selling.
- Paper-trading results may differ from live execution.
- Model performance may degrade out of sample.
- Backtest results are not guarantees of future performance.

---

## Quick Command Reference

```bash
# Launch market data terminal UI
python app.py

# Launch paper-trading system monitor UI
python run_system_monitor.py

# Technical-indicator backtest with sample data
python run_backtest.py --ticker SAMPLE --years 1 --sample

# Technical-indicator backtest with Alpaca data
python run_backtest.py --ticker MSFT --years 5

# Machine-learning backtest
python run_ml_backtest.py --symbol SPY

# Signal scan
python run_signal_scan.py --symbols SPY AAPL MSFT QQQ NVDA TSLA

# One-shot paper-trading dry run
python run_paper_trade.py --model-bundle outputs\SPY_<timestamp>\artifacts\ml_spy_model_bundle.joblib

# One-shot paper-trading execution
python run_paper_trade.py --model-bundle outputs\SPY_<timestamp>\artifacts\ml_spy_model_bundle.joblib --qty 1 --execute

# Live paper-trading engine dry run
python run_live_paper_engine.py --model-bundles outputs\SPY_<timestamp>\artifacts\ml_spy_model_bundle.joblib --cycles 1

# Live paper-trading engine using local config
python run_live_paper_engine.py --config config/config.yaml

# Live paper-trading engine continuous dry run
python run_live_paper_engine.py --model-bundles outputs\SPY_<timestamp>\artifacts\ml_spy_model_bundle.joblib --continuous --polling-interval-seconds 300

# Live paper-trading engine with paper order execution
python run_live_paper_engine.py --model-bundles outputs\SPY_<timestamp>\artifacts\ml_spy_model_bundle.joblib --cycles 1 --execute
```