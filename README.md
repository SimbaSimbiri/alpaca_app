# Market Data Terminal + Strategy Backtesting Platform

A Python market data and algorithmic trading research project built with Alpaca, Pandas, Matplotlib, and Tkinter.

This project has two main parts:

1. **Market Data Terminal**  
   A desktop Tkinter application that connects to Alpaca, downloads historical OHLCV data, streams real-time bid/ask quotes and last trade prices, and displays candlestick-style charts.

2. **Technical Indicator Strategy Backtester**  
   A reusable backtesting workflow for the FINM-25000 assignment. It downloads historical daily OHLCV data from Alpaca, computes technical indicators, compares multiple strategies, generates charts, calculates performance metrics, and produces a final PDF report.

---

## Features

### Market Data Terminal

- Authenticates with Alpaca paper-trading credentials
- Loads API keys securely from a local `.env` file
- Downloads historical OHLCV bars
- Supports user-selected timeframes such as minute, hour, day, week, and month
- Displays the most recent chart window for readability
- Shows candlestick-style OHLCV data with volume bars
- Colors bullish candles green and bearish candles black
- Adds daily separators to make session boundaries easier to read
- Streams real-time bid/ask quotes
- Streams last trade prices
- Updates the UI automatically when new live market data arrives
- Organizes API, configuration, streaming, and UI logic into separate modules

### Backtesting Platform

- Downloads 5+ years of daily OHLCV data from Alpaca
- Computes technical indicators
- Builds strategy entry and exit signals
- Runs long-only portfolio simulations
- Compares strategies against Buy & Hold
- Exports trades, portfolios, metrics, charts, and reports
- Includes a sample-data mode for testing without Alpaca credentials

---

## Tech Stack

- Python
- Alpaca-py
- Pandas
- NumPy
- Matplotlib
- Tkinter
- python-dotenv
- Git / GitHub

---

## Project Structure

```text
mini-market-data-terminal/
│
├── app.py                         # Tkinter market terminal
├── run_backtest.py                # Assignment CLI backtest runner
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── market_terminal/
│   ├── __init__.py
│   ├── config.py                  # Loads Alpaca credentials from .env
│   ├── constants.py               # Shared constants and defaults
│   ├── data_connector.py          # Alpaca historical data + paper account validation
│   ├── live_stream.py             # Alpaca websocket stream for terminal UI
│   ├── indicators.py              # SMA, EMA, MACD, ADX, RSI, Bollinger, ATR, OBV, CMF.
│   ├── strategies.py              # Strategy signal rules
│   ├── backtester.py              # Reusable long-only backtesting engine
│   ├── performance.py             # Risk/return metrics
│   ├── visualizations.py          # Charts
│   └── report.py                  # report generator
│
├── screenshots/                   # Terminal UI screenshots
│   ├── ui_running_1.png
│   ├── ui_running_2.png
│   ├── ui_running_3.png
│   └── ui_running_4.png
│
├── charts/                        # Optional manual chart folder
├── data/                          # Optional manual data folder
├── reports/                       # Optional manual reports folder
└── outputs/                       # Created automatically by run_backtest.py
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

This project requires Alpaca API credentials.

Create a local `.env` file in the project root with the following contents.

```env
ALPACA_API_KEY=replace_me
ALPACA_SECRET_KEY=replace_me
ALPACA_DATA_FEED=iex
```

---

## Running the Existing Market Terminal UI

After setting up the virtual environment and `.env` file, run:

```bash
python app.py
```

Recommended tickers for testing:

- MSFT
- AAPL
- SPY
- QQQ
- NVDA
- TSLA

### How to Use the Terminal

1. Open the app with:

   ```bash
   python app.py
   ```

2. Enter a ticker symbol, such as `MSFT`, `AAPL`, or `NVDA`.
3. Select a timeframe.
4. Click **Load Historical Data**.
5. The chart displays recent OHLCV candlestick bars with volume.
6. Click **Start Live Stream**.
7. The real-time quote panel waits for live bid, ask, and last trade updates.
8. Click **Stop Stream** to stop the websocket stream.

### Historical Data Viewer

The terminal chart displays:

- Open, high, low, and close candles
- Green bullish candles where close is greater than or equal to open
- Black bearish candles where close is below open
- Volume bars beneath the candles
- Day separators at session boundaries
- A recent OHLCV table below the chart

The app downloads more data than it displays so the backend can work with historical data while keeping the UI readable.

### Real-Time Quote UI

The real-time quote panel displays:

- Current bid
- Current ask
- Last trade price
- Quote timestamp
- Trade timestamp

The live stream is event-driven, so the fields update when Alpaca sends new quote or trade events.

### Market Hours Note

Historical data can load outside regular market hours.

Real-time quotes and trades are easiest to observe during regular U.S. market hours. Outside market hours, the websocket may connect successfully but receive few or no quote/trade updates depending on the ticker, feed, and market activity.

---

## Running Backtest

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

Optional commission/slippage assumption:

```bash
python run_backtest.py --ticker MSFT --years 5 --commission 1.00
```

Local test without Alpaca credentials:

```bash
python run_backtest.py --ticker SAMPLE --years 5 --sample
```

Use the `--sample` flag only to verify that the code runs. Use Alpaca data for the final assignment submission.

---

## Generated Backtest Outputs

Each backtest run creates a timestamped folder:

```text
outputs/MSFT_YYYYMMDD_HHMMSS/
│
├── charts/
│   ├── trend_following_price_signals.png
│   ├── mean_reversion_price_signals.png
│   ├── custom_volume-confirmed_trend_pullback_price_signals.png
│   ├── equity_curve_comparison.png
│   └── drawdown_comparison.png
│
├── data/
│   ├── MSFT_daily_ohlcv_indicators_signals.csv
│   ├── MSFT_performance_metrics.csv
│   ├── buy_and_hold_portfolio.csv
│   ├── trend_following_portfolio.csv
│   ├── trend_following_trades.csv
│   ├── mean_reversion_portfolio.csv
│   ├── mean_reversion_trades.csv
│   ├── custom_volume-confirmed_trend_pullback_portfolio.csv
│   └── custom_volume-confirmed_trend_pullback_trades.csv
│
└── reports/
    └── MSFT_final_report.pdf
```

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

## Strategies

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

This strategy tries to participate when price is already trending upward and the trend has enough strength.

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

## Backtesting Assumptions

- Initial capital: `$100,000` by default
- Long-only
- No leverage
- No short selling
- Whole-share position sizing by default
- Close-based signals execute at the next day's open to reduce look-ahead bias
- Default commission is zero
- Commission can be changed with the `--commission` flag
- Daily returns are computed from portfolio value changes
- Buy & Hold is included as the benchmark

---

## Performance Metrics

The final report and CSV metrics include:

- Total Return
- CAGR
- Annualized Volatility
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate

Example comparison table format:

```text
Strategy                         Total Return    Sharpe    Sortino    Max Drawdown
Buy & Hold
Trend Following
Mean Reversion
Custom Volume-Confirmed Trend Pullback
```

---

## Visualizations

The backtesting workflow generates:

### Price and Signal Charts

Each strategy has a price chart showing:

- Close price
- Relevant indicators
- Buy signals
- Sell signals

### Equity Curve Comparison

Compares:

- Buy & Hold
- Trend Following
- Mean Reversion
- Custom Strategy

### Drawdown Chart

Compares drawdowns for:

- Buy & Hold
- Trend Following
- Mean Reversion
- Custom Strategy

---

## Final Report

The generated PDF report includes:

- Ticker and backtest period
- Strategy descriptions
- Entry and exit rules
- Performance comparison table
- Price/signal charts
- Equity curve comparison
- Drawdown comparison
- Discussion of results

The report is saved under:

```text
outputs/<TICKER>_<TIMESTAMP>/reports/<TICKER>_final_report.pdf
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
