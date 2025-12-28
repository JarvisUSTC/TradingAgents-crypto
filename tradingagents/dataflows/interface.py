from typing import Annotated, Dict
from .reddit_utils import fetch_top_from_category
from .yfin_utils import *
from .stockstats_utils import *
from .googlenews_utils import *
from .finnhub_utils import get_data_in_range
from .coingecko_utils import (
    get_crypto_price_data,
    get_crypto_market_data,
    get_crypto_news,
    get_crypto_technical_indicators
)
from dateutil.relativedelta import relativedelta
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
import pandas as pd
from tqdm import tqdm
import yfinance as yf
from openai import OpenAI
from .config import get_config, set_config, DATA_DIR
import asyncio
import os
import threading


def get_finnhub_news(
    ticker: Annotated[
        str,
        "Search query of a company's, e.g. 'AAPL, TSM, etc.",
    ],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
):
    """
    Retrieve news about a company within a time frame

    Args
        ticker (str): ticker for the company you are interested in
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns
        str: dataframe containing the news of the company in the time frame

    """

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    result = get_data_in_range(ticker, before, curr_date, "news_data", DATA_DIR)

    if len(result) == 0:
        return ""

    combined_result = ""
    for day, data in result.items():
        if len(data) == 0:
            continue
        for entry in data:
            current_news = (
                "### " + entry["headline"] + f" ({day})" + "\n" + entry["summary"]
            )
            combined_result += current_news + "\n\n"

    return f"## {ticker} News, from {before} to {curr_date}:\n" + str(combined_result)


def get_finnhub_company_insider_sentiment(
    ticker: Annotated[str, "ticker symbol for the company"],
    curr_date: Annotated[
        str,
        "current date of you are trading at, yyyy-mm-dd",
    ],
    look_back_days: Annotated[int, "number of days to look back"],
):
    """
    Retrieve insider sentiment about a company (retrieved from public SEC information) for the past 15 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading on, yyyy-mm-dd
    Returns:
        str: a report of the sentiment in the past 15 days starting at curr_date
    """

    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    data = get_data_in_range(ticker, before, curr_date, "insider_senti", DATA_DIR)

    if len(data) == 0:
        return ""

    result_str = ""
    seen_dicts = []
    for date, senti_list in data.items():
        for entry in senti_list:
            if entry not in seen_dicts:
                result_str += f"### {entry['year']}-{entry['month']}:\nChange: {entry['change']}\nMonthly Share Purchase Ratio: {entry['mspr']}\n\n"
                seen_dicts.append(entry)

    return (
        f"## {ticker} Insider Sentiment Data for {before} to {curr_date}:\n"
        + result_str
        + "The change field refers to the net buying/selling from all insiders' transactions. The mspr field refers to monthly share purchase ratio."
    )


def get_finnhub_company_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[
        str,
        "current date you are trading at, yyyy-mm-dd",
    ],
    look_back_days: Annotated[int, "how many days to look back"],
):
    """
    Retrieve insider transcaction information about a company (retrieved from public SEC information) for the past 15 days
    Args:
        ticker (str): ticker symbol of the company
        curr_date (str): current date you are trading at, yyyy-mm-dd
    Returns:
        str: a report of the company's insider transaction/trading informtaion in the past 15 days
    """

    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    data = get_data_in_range(ticker, before, curr_date, "insider_trans", DATA_DIR)

    if len(data) == 0:
        return ""

    result_str = ""

    seen_dicts = []
    for date, senti_list in data.items():
        for entry in senti_list:
            if entry not in seen_dicts:
                result_str += f"### Filing Date: {entry['filingDate']}, {entry['name']}:\nChange:{entry['change']}\nShares: {entry['share']}\nTransaction Price: {entry['transactionPrice']}\nTransaction Code: {entry['transactionCode']}\n\n"
                seen_dicts.append(entry)

    return (
        f"## {ticker} insider transactions from {before} to {curr_date}:\n"
        + result_str
        + "The change field reflects the variation in share count—here a negative number indicates a reduction in holdings—while share specifies the total number of shares involved. The transactionPrice denotes the per-share price at which the trade was executed, and transactionDate marks when the transaction occurred. The name field identifies the insider making the trade, and transactionCode (e.g., S for sale) clarifies the nature of the transaction. FilingDate records when the transaction was officially reported, and the unique id links to the specific SEC filing, as indicated by the source. Additionally, the symbol ties the transaction to a particular company, isDerivative flags whether the trade involves derivative securities, and currency notes the currency context of the transaction."
    )


def get_simfin_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "balance_sheet",
        "companies",
        "us",
        f"us-balance-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No balance sheet available before the given current date.")
        return ""

    # Get the most recent balance sheet by selecting the row with the latest Publish Date
    latest_balance_sheet = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_balance_sheet = latest_balance_sheet.drop("SimFinId")

    return (
        f"## {freq} balance sheet for {ticker} released on {str(latest_balance_sheet['Publish Date'])[0:10]}: \n"
        + str(latest_balance_sheet)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of assets, liabilities, and equity. Assets are grouped as current (liquid items like cash and receivables) and noncurrent (long-term investments and property). Liabilities are split between short-term obligations and long-term debts, while equity reflects shareholder funds such as paid-in capital and retained earnings. Together, these components ensure that total assets equal the sum of liabilities and equity."
    )


def get_simfin_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "cash_flow",
        "companies",
        "us",
        f"us-cashflow-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No cash flow statement available before the given current date.")
        return ""

    # Get the most recent cash flow statement by selecting the row with the latest Publish Date
    latest_cash_flow = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_cash_flow = latest_cash_flow.drop("SimFinId")

    return (
        f"## {freq} cash flow statement for {ticker} released on {str(latest_cash_flow['Publish Date'])[0:10]}: \n"
        + str(latest_cash_flow)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a breakdown of cash movements. Operating activities show cash generated from core business operations, including net income adjustments for non-cash items and working capital changes. Investing activities cover asset acquisitions/disposals and investments. Financing activities include debt transactions, equity issuances/repurchases, and dividend payments. The net change in cash represents the overall increase or decrease in the company's cash position during the reporting period."
    )


def get_simfin_income_statements(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[
        str,
        "reporting frequency of the company's financial history: annual / quarterly",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
):
    data_path = os.path.join(
        DATA_DIR,
        "fundamental_data",
        "simfin_data_all",
        "income_statements",
        "companies",
        "us",
        f"us-income-{freq}.csv",
    )
    df = pd.read_csv(data_path, sep=";")

    # Convert date strings to datetime objects and remove any time components
    df["Report Date"] = pd.to_datetime(df["Report Date"], utc=True).dt.normalize()
    df["Publish Date"] = pd.to_datetime(df["Publish Date"], utc=True).dt.normalize()

    # Convert the current date to datetime and normalize
    curr_date_dt = pd.to_datetime(curr_date, utc=True).normalize()

    # Filter the DataFrame for the given ticker and for reports that were published on or before the current date
    filtered_df = df[(df["Ticker"] == ticker) & (df["Publish Date"] <= curr_date_dt)]

    # Check if there are any available reports; if not, return a notification
    if filtered_df.empty:
        print("No income statement available before the given current date.")
        return ""

    # Get the most recent income statement by selecting the row with the latest Publish Date
    latest_income = filtered_df.loc[filtered_df["Publish Date"].idxmax()]

    # drop the SimFinID column
    latest_income = latest_income.drop("SimFinId")

    return (
        f"## {freq} income statement for {ticker} released on {str(latest_income['Publish Date'])[0:10]}: \n"
        + str(latest_income)
        + "\n\nThis includes metadata like reporting dates and currency, share details, and a comprehensive breakdown of the company's financial performance. Starting with Revenue, it shows Cost of Revenue and resulting Gross Profit. Operating Expenses are detailed, including SG&A, R&D, and Depreciation. The statement then shows Operating Income, followed by non-operating items and Interest Expense, leading to Pretax Income. After accounting for Income Tax and any Extraordinary items, it concludes with Net Income, representing the company's bottom-line profit or loss for the period."
    )


def get_google_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    query = query.replace(" ", "+")

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    news_results = getNewsData(query, before, curr_date)

    news_str = ""

    for news in news_results:
        news_str += (
            f"### {news['title']} (source: {news['source']}) \n\n{news['snippet']}\n\n"
        )

    if len(news_results) == 0:
        return ""

    return f"## {query} Google News, from {before} to {curr_date}:\n\n{news_str}"


def get_reddit_global_news(
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
    max_limit_per_day: Annotated[int, "Maximum number of news per day"],
) -> str:
    """
    Retrieve the latest top reddit news
    Args:
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the latest news articles posts on reddit and meta information in these columns: "created_utc", "id", "title", "selftext", "score", "num_comments", "url"
    """

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    posts = []
    # iterate from start_date to end_date
    curr_date = datetime.strptime(before, "%Y-%m-%d")

    total_iterations = (start_date - curr_date).days + 1
    pbar = tqdm(desc=f"Getting Global News on {start_date}", total=total_iterations)

    while curr_date <= start_date:
        curr_date_str = curr_date.strftime("%Y-%m-%d")
        fetch_result = fetch_top_from_category(
            "global_news",
            curr_date_str,
            max_limit_per_day,
            data_path=os.path.join(DATA_DIR, "reddit_data"),
        )
        posts.extend(fetch_result)
        curr_date += relativedelta(days=1)
        pbar.update(1)

    pbar.close()

    if len(posts) == 0:
        return ""

    news_str = ""
    for post in posts:
        if post["content"] == "":
            news_str += f"### {post['title']}\n\n"
        else:
            news_str += f"### {post['title']}\n\n{post['content']}\n\n"

    return f"## Global News Reddit, from {before} to {curr_date}:\n{news_str}"


def get_reddit_company_news(
    ticker: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
    max_limit_per_day: Annotated[int, "Maximum number of news per day"],
) -> str:
    """
    Retrieve the latest top reddit news
    Args:
        ticker: ticker symbol of the company
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the latest news articles posts on reddit and meta information in these columns: "created_utc", "id", "title", "selftext", "score", "num_comments", "url"
    """

    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    posts = []
    # iterate from start_date to end_date
    curr_date = datetime.strptime(before, "%Y-%m-%d")

    total_iterations = (start_date - curr_date).days + 1
    pbar = tqdm(
        desc=f"Getting Company News for {ticker} on {start_date}",
        total=total_iterations,
    )

    while curr_date <= start_date:
        curr_date_str = curr_date.strftime("%Y-%m-%d")
        fetch_result = fetch_top_from_category(
            "company_news",
            curr_date_str,
            max_limit_per_day,
            ticker,
            data_path=os.path.join(DATA_DIR, "reddit_data"),
        )
        posts.extend(fetch_result)
        curr_date += relativedelta(days=1)

        pbar.update(1)

    pbar.close()

    if len(posts) == 0:
        return ""

    news_str = ""
    for post in posts:
        if post["content"] == "":
            news_str += f"### {post['title']}\n\n"
        else:
            news_str += f"### {post['title']}\n\n{post['content']}\n\n"

    return f"##{ticker} News Reddit, from {before} to {curr_date}:\n\n{news_str}"


def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
    online: Annotated[bool, "to fetch data online or offline"],
) -> str:

    best_ind_params = {
        # Moving Averages
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        # MACD Related
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes. "
            "Tips: Confirm with other indicators in low-volatility or sideways markets."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades. "
            "Tips: Should be part of a broader strategy to avoid false positives."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early. "
            "Tips: Can be volatile; complement with additional filters in fast-moving markets."
        ),
        # Momentum Indicators
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        # Volatility Indicators
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        # Volume-Based Indicators
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date - relativedelta(days=look_back_days)

    if not online:
        # read from YFin data
        data = pd.read_csv(
            os.path.join(
                DATA_DIR,
                f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
            )
        )
        data["Date"] = pd.to_datetime(data["Date"], utc=True)
        dates_in_df = data["Date"].astype(str).str[:10]

        ind_string = ""
        while curr_date >= before:
            # only do the trading dates
            if curr_date.strftime("%Y-%m-%d") in dates_in_df.values:
                indicator_value = get_stockstats_indicator(
                    symbol, indicator, curr_date.strftime("%Y-%m-%d"), online
                )

                ind_string += f"{curr_date.strftime('%Y-%m-%d')}: {indicator_value}\n"

            curr_date = curr_date - relativedelta(days=1)
    else:
        # online gathering
        ind_string = ""
        while curr_date >= before:
            indicator_value = get_stockstats_indicator(
                symbol, indicator, curr_date.strftime("%Y-%m-%d"), online
            )

            ind_string += f"{curr_date.strftime('%Y-%m-%d')}: {indicator_value}\n"

            curr_date = curr_date - relativedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    online: Annotated[bool, "to fetch data online or offline"],
) -> str:

    curr_date = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = curr_date.strftime("%Y-%m-%d")

    try:
        indicator_value = StockstatsUtils.get_stock_stats(
            symbol,
            indicator,
            curr_date,
            os.path.join(DATA_DIR, "market_data", "price_data"),
            online=online,
        )
    except Exception as e:
        print(
            f"Error getting stockstats indicator data for indicator {indicator} on {curr_date}: {e}"
        )
        return ""

    return str(indicator_value)


def get_YFin_data_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    # calculate past days
    date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    before = date_obj - relativedelta(days=look_back_days)
    start_date = before.strftime("%Y-%m-%d")

    # read in data
    data = pd.read_csv(
        os.path.join(
            DATA_DIR,
            f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
        )
    )

    # Extract just the date part for comparison
    data["DateOnly"] = data["Date"].str[:10]

    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["DateOnly"] >= start_date) & (data["DateOnly"] <= curr_date)
    ]

    # Drop the temporary column we created
    filtered_data = filtered_data.drop("DateOnly", axis=1)

    # Set pandas display options to show the full DataFrame
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None, "display.width", None
    ):
        df_string = filtered_data.to_string()

    return (
        f"## Raw Market Data for {symbol} from {start_date} to {curr_date}:\n\n"
        + df_string
    )


def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):

    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    # Create ticker object
    ticker = yf.Ticker(symbol.upper())

    # Fetch historical data for the specified date range
    data = ticker.history(start=start_date, end=end_date)

    # Check if data is empty
    if data.empty:
        return (
            f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        )

    # Remove timezone info from index for cleaner output
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Round numerical values to 2 decimal places for cleaner display
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    # Convert DataFrame to CSV string
    csv_string = data.to_csv()

    # Add header information
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string


def get_YFin_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    # read in data
    data = pd.read_csv(
        os.path.join(
            DATA_DIR,
            f"market_data/price_data/{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
        )
    )

    if end_date > "2025-03-25":
        raise Exception(
            f"Get_YFin_Data: {end_date} is outside of the data range of 2015-01-01 to 2025-03-25"
        )

    # Extract just the date part for comparison
    data["DateOnly"] = data["Date"].str[:10]

    # Filter data between the start and end dates (inclusive)
    filtered_data = data[
        (data["DateOnly"] >= start_date) & (data["DateOnly"] <= end_date)
    ]

    # Drop the temporary column we created
    filtered_data = filtered_data.drop("DateOnly", axis=1)

    # remove the index from the dataframe
    filtered_data = filtered_data.reset_index(drop=True)

    return filtered_data


def get_stock_news_openai(ticker, curr_date):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"], api_key=config["api_key"])

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search Social Media for {ticker} from 7 days before {curr_date} to {curr_date}? Make sure you only get the data posted during that period.",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text


def get_global_news_openai(curr_date):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"], api_key=config["api_key"])

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search global or macroeconomics news from 7 days before {curr_date} to {curr_date} that would be informative for trading purposes? Make sure you only get the data posted during that period.",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text


def get_fundamentals_openai(ticker, curr_date):
    config = get_config()
    client = OpenAI(base_url=config["backend_url"], api_key=config["api_key"])

    response = client.responses.create(
        model=config["quick_think_llm"],
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Can you search Fundamental for discussions on {ticker} during of the month before {curr_date} to the month of {curr_date}. Make sure you only get the data posted during that period. List as a table, with PE/PS/Cash flow/ etc",
                    }
                ],
            }
        ],
        text={"format": {"type": "text"}},
        reasoning={},
        tools=[
            {
                "type": "web_search_preview",
                "user_location": {"type": "approximate"},
                "search_context_size": "low",
            }
        ],
        temperature=1,
        max_output_tokens=4096,
        top_p=1,
        store=True,
    )

    return response.output[1].content[0].text


# ===== CRYPTO TRADING FUNCTIONS =====

def _get_hb_md_provider():
    """Helper to get or create a Hummingbot MarketDataProvider (singleton-like for this session)"""
    if not hasattr(_get_hb_md_provider, "_provider"):
        # Import Hummingbot lazily. The caller is responsible for providing a working environment.
        from hummingbot.data_feed.market_data_provider import MarketDataProvider

        # MarketDataProvider requires a dict of connectors.
        # For public candles access in research mode, we don't need authenticated trading connectors.
        _get_hb_md_provider._provider = MarketDataProvider(connectors={})
    return _get_hb_md_provider._provider


def _run_coro_sync(coro):
    """Run an async coroutine from sync code safely (even if an event loop is already running)."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop is None or not loop.is_running():
        return asyncio.run(coro)

    # If there's already a running loop (e.g., called from an async context), run in a new thread.
    result_container = {"value": None, "error": None}

    def _worker():
        try:
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)
            result_container["value"] = new_loop.run_until_complete(coro)
        except Exception as e:
            result_container["error"] = e
        finally:
            try:
                new_loop.close()
            except Exception:
                pass

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()
    if result_container["error"] is not None:
        raise result_container["error"]
    return result_container["value"]

def _format_hb_candles_to_report(symbol: str, df: pd.DataFrame, start_date: str, end_date: str) -> str:
    """Helper to format Hummingbot candles DataFrame into the string report expected by the LLM"""
    if df is None or df.empty:
        return f"No price data available for {symbol} via Hummingbot."

    # Hummingbot candles columns: ["timestamp", "open", "high", "low", "close", "volume", ...]
    # Convert timestamp (ms) to YYYY-MM-DD
    df = df.copy()
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y-%m-%d")
    
    result_str = f"## {symbol.upper()} Price Data (via Hummingbot) from {start_date} to {end_date}:\n\n"
    
    # Only take the relevant range and last few points if too many (similar to original logic)
    # The df returned by get_historical_candles_df is already filtered by time usually.
    # We'll show up to 30 records to match original behavior.
    records_to_show = df.tail(30)
    
    for _, row in records_to_show.iterrows():
        result_str += f"Date: {row['date']}\n"
        result_str += f"Price: ${row['close']:,.2f}\n"
        result_str += f"Volume: ${row['volume']:,.0f}\n"
        result_str += f"High: ${row['high']:,.2f} | Low: ${row['low']:,.2f}\n\n"
    
    return result_str

def _fetch_hb_historical_data(symbol: str, start_date: str, end_date: str, interval: str):
    """Sync wrapper for the async HB historical fetch"""
    config = get_config()
    hb_cfg = config.get("hummingbot_market_data", {})
    connector = hb_cfg.get("connector", "binance")
    quote = hb_cfg.get("quote_asset", "USDT")
    trading_pair = f"{symbol.upper()}-{quote.upper()}"
    max_cache = hb_cfg.get("max_cache_records", 10000)

    # Convert YYYY-MM-DD to timestamp (seconds)
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    provider = _get_hb_md_provider()

    # Optional cache: persist candles to disk for research reproducibility and faster reruns.
    hb_cfg = config.get("hummingbot_market_data", {})
    cache_dir = hb_cfg.get("cache_dir")
    if isinstance(cache_dir, str) and cache_dir.strip() == "":
        cache_dir = None
    if cache_dir:
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            cache_dir = None

    cache_path = None
    if cache_dir:
        safe_pair = trading_pair.replace("/", "_")
        cache_path = os.path.join(cache_dir, f"hb_candles_{connector}_{safe_pair}_{interval}_{start_date}_{end_date}.csv")
        if os.path.exists(cache_path):
            try:
                cached = pd.read_csv(cache_path)
                if not cached.empty and "timestamp" in cached.columns:
                    return cached
            except Exception:
                pass
    
    df = _run_coro_sync(provider.get_historical_candles_df(
        connector_name=connector,
        trading_pair=trading_pair,
        interval=interval,
        start_time=start_ts,
        end_time=end_ts,
        max_cache_records=max_cache
    ))

    if cache_path and df is not None and not df.empty:
        try:
            df.to_csv(cache_path, index=False)
        except Exception:
            pass
    return df

def get_crypto_market_analysis(
    symbol: Annotated[str, "Cryptocurrency symbol like BTC, ETH, ADA"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """
    Get comprehensive market analysis for a cryptocurrency
    
    Args:
        symbol: Crypto symbol (e.g., 'BTC', 'ETH', 'ADA')
        curr_date: Current date in yyyy-mm-dd format
    
    Returns:
        String containing market data and analysis
    """
    return get_crypto_market_data(symbol)


def get_crypto_price_history(
    symbol: Annotated[str, "Cryptocurrency symbol like BTC, ETH, ADA"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "How many days to look back"] = 30,
) -> str:
    """
    Get historical price data for a cryptocurrency
    
    Args:
        symbol: Crypto symbol (e.g., 'BTC', 'ETH', 'ADA')
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back
    
    Returns:
        String containing historical price data
    """
    from datetime import datetime, timedelta
    
    curr_date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
    start_date_obj = curr_date_obj - timedelta(days=look_back_days)
    start_date = start_date_obj.strftime("%Y-%m-%d")
    
    config = get_config()
    hb_cfg = config.get("hummingbot_market_data", {})
    
    if hb_cfg.get("enabled", False):
        try:
            interval = hb_cfg.get("interval", "1d")
            df = _fetch_hb_historical_data(symbol, start_date, curr_date, interval)
            return _format_hb_candles_to_report(symbol, df, start_date, curr_date)
        except Exception as e:
            # Fallback to the existing CoinGecko implementation.
            _ = e

    return get_crypto_price_data(symbol, start_date, curr_date)


def get_crypto_technical_analysis(
    symbol: Annotated[str, "Cryptocurrency symbol like BTC, ETH, ADA"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "How many days to look back"] = 30,
) -> str:
    """
    Get technical analysis for a cryptocurrency
    
    Args:
        symbol: Crypto symbol (e.g., 'BTC', 'ETH', 'ADA')
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to analyze
    
    Returns:
        String containing technical analysis
    """
    config = get_config()
    hb_cfg = config.get("hummingbot_market_data", {})
    
    if hb_cfg.get("enabled", False):
        try:
            from datetime import datetime, timedelta
            curr_date_obj = datetime.strptime(curr_date, "%Y-%m-%d")
            start_date_obj = curr_date_obj - timedelta(days=look_back_days)
            start_date = start_date_obj.strftime("%Y-%m-%d")
            
            interval = hb_cfg.get("interval", "1d")
            df = _fetch_hb_historical_data(symbol, start_date, curr_date, interval)
            
            if df is not None and not df.empty:
                # Compute simple indicators from HB DataFrame instead of calling CoinGecko
                current_price = float(df['close'].iloc[-1])
                avg_7d = float(df['close'].tail(min(len(df), 7)).mean())
                avg_30d = float(df['close'].tail(min(len(df), 30)).mean())
                high_30 = float(df['high'].tail(min(len(df), 30)).max())
                low_30 = float(df['low'].tail(min(len(df), 30)).min())
                vol_7d = float(df['volume'].tail(min(len(df), 7)).mean())
                
                trend_7d = "Bullish" if current_price > avg_7d else "Bearish"
                trend_30d = "Bullish" if current_price > avg_30d else "Bearish"
                
                report = f"## {symbol.upper()} Technical Analysis (via Hummingbot - {look_back_days} days):\n\n"
                report += f"**Price Levels:**\n"
                report += f"- Current Price: ${current_price:,.2f}\n"
                report += f"- 7-day Average: ${avg_7d:,.2f}\n"
                report += f"- 30-day Average: ${avg_30d:,.2f}\n"
                report += f"- 30-day High: ${high_30:,.2f}\n"
                report += f"- 30-day Low: ${low_30:,.2f}\n\n"
                report += f"**Volume Analysis:**\n"
                report += f"- 7-day Average Volume: ${vol_7d:,.0f}\n\n"
                report += f"**Trend Analysis:**\n"
                report += f"- 7-day Trend: {trend_7d}\n"
                report += f"- 30-day Trend: {trend_30d}\n"
                report += f"- Distance from 30d High: {((current_price - high_30) / high_30 * 100):+.1f}%\n"
                report += f"- Distance from 30d Low: {((current_price - low_30) / low_30 * 100):+.1f}%\n"
                return report
        except Exception as e:
            _ = e

    return get_crypto_technical_indicators(symbol, curr_date, look_back_days)


def get_crypto_factor_report(
    symbol: Annotated[str, "Cryptocurrency symbol like BTC, ETH, ADA"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "How many days to look back"] = 180,
) -> str:
    """Generate a research-oriented factor report from OHLCV.

    - Prefer Hummingbot candles if enabled.
    - Fallback: use CoinGecko technical indicators (price-only) when OHLCV is not available.
    """
    from datetime import datetime, timedelta

    config = get_config()
    hb_cfg = config.get("hummingbot_market_data", {})

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    df = None
    used_source = "coingecko"
    if hb_cfg.get("enabled", False):
        try:
            interval = hb_cfg.get("interval", "1d")
            df = _fetch_hb_historical_data(symbol, start_date, curr_date, interval)
            used_source = f"hummingbot({hb_cfg.get('connector', '')},{interval})"
        except Exception:
            df = None

    if df is None or df.empty:
        # No structured OHLCV available; return existing price-based technical summary as fallback.
        fallback = get_crypto_technical_indicators(symbol, curr_date, look_back_days)
        return (
            f"## {symbol.upper()} Factor Report (fallback: CoinGecko price-only)\n\n"
            "无法获取结构化 OHLCV（用于因子计算）。以下返回价格层面的技术摘要作为替代：\n\n"
            + (fallback or "(no data)")
        )

    # Normalize columns and time
    work = df.copy()
    if "timestamp" in work.columns:
        work["date"] = pd.to_datetime(work["timestamp"], unit="ms").dt.date.astype(str)
    else:
        work["date"] = ""

    # Ensure numeric
    for c in ("open", "high", "low", "close", "volume"):
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.dropna(subset=["close"]).reset_index(drop=True)
    if work.empty:
        return f"## {symbol.upper()} Factor Report\n\nNo valid close prices available."

    close = work["close"]
    ret_1 = close.pct_change()
    # Log returns (optional; not all factors use it)
    try:
        import numpy as np
        logret_1 = np.log(close / close.shift(1))
    except Exception:
        logret_1 = None

    # Common factor-like summaries
    mom_7 = close.pct_change(7)
    mom_30 = close.pct_change(30)
    vol_30 = ret_1.rolling(30).std() * (30 ** 0.5)
    vol_90 = ret_1.rolling(90).std() * (90 ** 0.5)
    ma_20 = close.rolling(20).mean()
    ma_60 = close.rolling(60).mean()
    ma_ratio_20 = close / ma_20 - 1
    ma_ratio_60 = close / ma_60 - 1

    # RSI(14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi_14 = 100 - (100 / (1 + rs))

    # Volume zscore (20)
    vol = work["volume"] if "volume" in work.columns else pd.Series([pd.NA] * len(work))
    vol_z20 = (vol - vol.rolling(20).mean()) / vol.rolling(20).std()

    # Latest snapshot
    last_idx = len(work) - 1
    def _fmt(x, pct=False, digits=4):
        if x is None or (isinstance(x, float) and (pd.isna(x))):
            return "N/A"
        try:
            if pct:
                return f"{float(x) * 100:.2f}%"
            return f"{float(x):.{digits}f}"
        except Exception:
            return "N/A"

    last_date = work.loc[last_idx, "date"] if "date" in work.columns else curr_date
    last_close = close.iloc[last_idx]

    report = f"## {symbol.upper()} Factor Report (source: {used_source})\n\n"
    report += f"**As of:** {last_date}\n"
    report += f"**Close:** ${float(last_close):,.4f}\n\n"

    report += "**Momentum:**\n"
    report += f"- 7D momentum: {_fmt(mom_7.iloc[last_idx], pct=True)}\n"
    report += f"- 30D momentum: {_fmt(mom_30.iloc[last_idx], pct=True)}\n\n"

    report += "**Volatility (annualized-ish over window):**\n"
    report += f"- 30D vol: {_fmt(vol_30.iloc[last_idx], digits=4)}\n"
    report += f"- 90D vol: {_fmt(vol_90.iloc[last_idx], digits=4)}\n\n"

    report += "**Trend (MA distance):**\n"
    report += f"- price vs MA20: {_fmt(ma_ratio_20.iloc[last_idx], pct=True)}\n"
    report += f"- price vs MA60: {_fmt(ma_ratio_60.iloc[last_idx], pct=True)}\n\n"

    report += "**Mean-reversion / Overbought:**\n"
    report += f"- RSI(14): {_fmt(rsi_14.iloc[last_idx], digits=2)}\n\n"

    report += "**Liquidity / Activity:**\n"
    report += f"- Volume zscore(20): {_fmt(vol_z20.iloc[last_idx], digits=2)}\n"
    return report


def get_crypto_news_analysis(
    symbol: Annotated[str, "Cryptocurrency symbol like BTC, ETH, ADA"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "How many days to look back"] = 7,
) -> str:
    """
    Get news and market trends for cryptocurrency
    
    Args:
        symbol: Crypto symbol (e.g., 'BTC', 'ETH', 'ADA')
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back
    
    Returns:
        String containing news and trends
    """
    return get_crypto_news(symbol, curr_date, look_back_days)


def get_crypto_fundamentals_analysis(
    symbol: Annotated[str, "Cryptocurrency symbol like BTC, ETH, ADA"],
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
) -> str:
    """
    Get fundamental analysis for a cryptocurrency (different from traditional stocks)
    
    Args:
        symbol: Crypto symbol (e.g., 'BTC', 'ETH', 'ADA')
        curr_date: Current date in yyyy-mm-dd format
    
    Returns:
        String containing fundamental metrics like market cap, supply, etc.
    """
    # For crypto, fundamentals include market cap, supply metrics, dominance, etc.
    market_data = get_crypto_market_data(symbol)
    
    # Add some additional context for crypto fundamentals
    additional_context = f"""

## Cryptocurrency Fundamental Analysis Context for {symbol.upper()}:

**Key Differences from Traditional Stocks:**
- No earnings reports or P/E ratios
- Focus on adoption, technology, and network metrics
- Token economics and supply mechanics are crucial
- Market sentiment and community strength matter significantly

**Important Metrics to Consider:**
- Market capitalization and rank
- Circulating vs total supply
- Trading volume and liquidity
- Network activity and adoption
- Developer activity and updates
- Regulatory environment and compliance
"""
    
    return market_data + additional_context
