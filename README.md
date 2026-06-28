# Market Data Terminal

A Python desktop market data terminal that connects to Alpaca, downloads historical OHLCV stock data, streams real-time bid/ask quotes and last trade prices, and displays everything in a simple Tkinter UI.

This project simulates the core components of a lightweight trading terminal: authentication, historical market data retrieval, real-time quote streaming, charting, and UI updates.

## Features

* Authenticates with Alpaca paper-trading credentials
* Loads API keys securely from a local `.env` file
* Downloads 730 calendar days of historical OHLCV data
* Uses 15-minute stock bars by default, user can now toggle to desired timeframe.
* Displays the most recent 70 candle timeframe bars in the chart for readability
* Shows candlestick-style OHLCV data with volume bars
* Colors bullish candles green and bearish candles black
* Adds daily separators to make session boundaries easier to read
* Streams real-time bid/ask quotes
* Streams last trade prices
* Updates the UI automatically when new live market data arrives
* Organizes API, configuration, streaming, and UI logic into separate modules

## Tech Stack

* Python
* Alpaca-py
* Tkinter
* Matplotlib
* Pandas
* python-dotenv
* Git / GitHub

## Project Structure

```text
mini-market-data-terminal/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── market_terminal/
│   ├── __init__.py
│   ├── config.py
│   ├── constants.py
│   ├── data_connector.py
│   └── live_stream.py
│
└── screenshots/
    └── ui-running.png
```

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/SimbaSimbiri/mini-market-data-terminal.git
cd market_terminal
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

## API Key Setup

This project requires Alpaca API credentials.

Create a local `.env` file in the project root:

```env
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_DATA_FEED=iex
```

The real `.env` file is ignored by Git and should never be committed.

The repository includes `.env.example` as a safe template:

```env
ALPACA_API_KEY=replace_me
ALPACA_SECRET_KEY=replace_me
ALPACA_DATA_FEED=iex
```

## Running the App

After setting up the virtual environment and `.env` file, run:

```bash
python app.py
```

Recommended tickers for testing:

* MSFT
* AAPL
* SPY
* NVDA
* TSLA

## How to Use

1. Open the app with `python app.py`.
2. Enter a ticker symbol, such as `TSLA`.
3. Click **Load Historical Data**.
4. The app downloads 730 calendar days of 15-minute OHLCV bars.
5. The chart displays the most recent 70 candle bars for readability.
6. Click **Start Live Stream**.
7. The real-time quote panel waits for live bid, ask, and last trade updates.
8. Click **Stop Stream** to stop the websocket stream.

## Historical Data Viewer

The chart displays:

* Open, high, low, and close candles
* Green bullish candles where close is greater than or equal to open
* Black bearish candles where close is below open
* Volume bars beneath the candles
* Day separators at session boundaries
* A recent OHLCV table below the chart

The app downloads more data than it displays so the backend satisfies the historical data requirement while keeping the UI readable.

## Real-Time Quote UI

The real-time quote panel displays:

* Current bid
* Current ask
* Last trade price
* Quote timestamp
* Trade timestamp

The live stream is event-driven, so the fields update when Alpaca sends new quote or trade events.

## Market Hours Note

Historical data can load outside regular market hours.

Real-time quotes and trades are easiest to observe during regular US market hours. Outside market hours, the websocket may connect successfully but receive few or no quote/trade updates depending on the ticker, feed, and market activity.

## Screenshots
MSFT 4H Chart for the last two years
![MSFT 4H OHLCV Chart for the last two years](screenshots/ui_running_3.png)

MSFT Most Recent OHLCV Data
![MSFT Most Recent Data](screenshots/ui_running_4.png)

NVDA 30m Chart Outside Trading Hours

![NVDA Chart Outside Trading Hours](screenshots/ui_running_1.png)

NVDA Most Recent OHLCV Data

![NVDA Most Recent Data](screenshots/ui_running_2.png)

##
