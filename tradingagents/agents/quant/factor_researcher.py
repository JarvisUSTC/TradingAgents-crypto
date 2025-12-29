from typing import Dict, Any
from datetime import datetime, timedelta
import json

import pandas as pd
from langchain_core.prompts import ChatPromptTemplate

from tradingagents.agents.utils.agent_states import AgentState


def _to_trading_pair(symbol: str) -> str:
    """Convert a ticker like BTC to a trading pair like BTC-USDT."""
    if "-" in symbol:
        return symbol
    if "/" in symbol:
        return symbol.replace("/", "-")
    return f"{symbol}-USDT"


def create_factor_researcher(llm, toolkit):
    """
    Create a factor researcher node that:
    1) Calls hummingbot-api tools (candles / order-book) to gather raw data
    2) Summarizes key quantitative factors and writes them into AgentState
    """

    system = """You are a quantitative factor researcher.
You design and compute trading factors using historical candles and order book data.

Your responsibilities:
- Decide what factors to use for the current symbol and date (e.g., trend, volatility, volume imbalance, order book imbalance).
- Use the provided candle and order book data (pulled via hummingbot-api) to infer factor values.
- Based on the fetched data, compute and summarize factor values in a concise, machine-readable way.

Output requirements:
- First, describe in natural language which factors you are using and why.
- Then, provide a concise summary of the factor values and signals that downstream strategy designers can consume.
Keep the output strictly focused on quantitative information; do not make final trade decisions.
"""

    # Prompt for interpreting factor values and producing a human-readable specification
    interpret_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """You are analyzing {company_of_interest} on {trade_date}.
You have access to market analysis, sentiment, news, and fundamentals reports:

Market report:
{market_report}

Sentiment report:
{sentiment_report}

News report:
{news_report}

Fundamentals report:
{fundamentals_report}

You also have quantitative data retrieved from hummingbot-api:

- Recent candle data:
{candles_data}

- Current order book snapshot:
{order_book_data}

The following JSON contains pre-computed quantitative factor values derived from candles and order book:

{factor_values_json}

Design a factor specification and interpret these factors together with the research reports.
Return:
- A clear factor specification describing which factors are used and how.
- A concise summary of the current factor signals and what they imply for trading.""",
            ),
        ]
    )

    # Prompt for proposing additional factor combinations in a constrained JSON DSL
    spec_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a quantitative factor engineer.
You design NEW factors as combinations or transformations of existing base factors, for use in systematic trading.

You MUST:
- Only use the allowed operator types: 'momentum', 'ma_ratio', 'vol_ratio', 'volume_zscore'.
- Express your design as a JSON array of objects, each with:
  - name: string, a short identifier for the new factor
  - type: one of ['momentum', 'ma_ratio', 'vol_ratio', 'volume_zscore']
  - params: an object with parameters described below

Operator parameter schemas:
- momentum:
  - source: one of ['close']
  - horizon: integer, number of 1h bars to look back (e.g. 6, 12, 24)
- ma_ratio:
  - fast: integer, fast moving average window in bars (e.g. 10, 20)
  - slow: integer, slow moving average window in bars (e.g. 50, 100)
- vol_ratio:
  - short_window: integer, short volatility window in bars (e.g. 24)
  - long_window: integer, long volatility window in bars (e.g. 72)
- volume_zscore:
  - window: integer, lookback window for average volume (e.g. 24)
  - norm_window: integer, window for historical volume distribution (e.g. 72)

Rules:
- Only produce factors that can be computed from 1h OHLCV candles.
- Do NOT include free-form text in the JSON. The entire response MUST be valid JSON.""",
            ),
            (
                "human",
                """Base factor values JSON:
{factor_values_json}

Design 2-5 additional factors that could enrich this set, using the allowed operator types and parameter schemas.
Return ONLY the JSON array, with no extra text.""",
            ),
        ]
    )

    def node(state: AgentState) -> Dict[str, Any]:
        # Determine a reasonable lookback window for candles based on trade_date
        symbol = state["company_of_interest"]
        trading_pair = _to_trading_pair(symbol)
        trade_date = state["trade_date"]
        try:
            dt = datetime.strptime(trade_date, "%Y-%m-%d")
            # We use a fixed window in number of records (max_records)
            _ = dt  # kept for potential future date alignment
        except Exception:
            pass

        # Fetch recent candles and order book via hummingbot-api tools
        try:
            # hb_get_candles is exposed as a LangChain StructuredTool via @tool.
            # When calling it directly from code (outside LangGraph), we need
            # to call its underlying Python function via the .func attribute.
            candles_raw = toolkit.hb_get_candles.func(trading_pair, "1h", 500)
        except Exception as e:
            candles_raw = json.dumps({"error": f"Failed to fetch candles: {e}"})

        try:
            order_book_raw = toolkit.hb_get_order_book.func(trading_pair, 50)
        except Exception as e:
            order_book_raw = json.dumps({"error": f"Failed to fetch order book: {e}"})

        # Compute simple technical and order book factors
        factors: Dict[str, Any] = {}
        candles_overview = ""
        order_book_overview = ""
        closes = None
        volumes = None

        try:
            candles_json = json.loads(candles_raw)
            if isinstance(candles_json, dict) and "error" in candles_json:
                candles_overview = str(candles_json)
            else:
                df = pd.DataFrame(candles_json)
                if not df.empty:
                    df = df.sort_values("timestamp")
                    closes = df["close"].astype(float)
                    volumes = df["volume"].astype(float)

                    # 1h return
                    if len(closes) >= 2:
                        factors["ret_1h"] = float(closes.iloc[-1] / closes.iloc[-2] - 1.0)

                    # 24h return (approx 24 * 1h candles)
                    if len(closes) >= 24:
                        factors["ret_24h"] = float(
                            closes.iloc[-1] / closes.iloc[-24] - 1.0
                        )

                    # Moving averages
                    if len(closes) >= 20:
                        ma_fast = float(closes.tail(20).mean())
                        factors["ma_fast_20"] = ma_fast
                    else:
                        ma_fast = None

                    if len(closes) >= 50:
                        ma_slow = float(closes.tail(50).mean())
                        factors["ma_slow_50"] = ma_slow
                    else:
                        ma_slow = None

                    if ma_fast is not None and ma_slow is not None:
                        if ma_fast > ma_slow:
                            factors["trend_state"] = "uptrend"
                        elif ma_fast < ma_slow:
                            factors["trend_state"] = "downtrend"
                        else:
                            factors["trend_state"] = "sideways"

                    # Volatility over last 24 hours
                    if len(closes) >= 24:
                        rets_24h = closes.pct_change().tail(24).dropna()
                        if not rets_24h.empty:
                            factors["vol_24h"] = float(rets_24h.std())

                    # Volume features
                    if len(volumes) >= 24:
                        vol_24h = volumes.tail(24)
                        factors["avg_volume_24h"] = float(vol_24h.mean())
                        if len(volumes) >= 72:
                            vol_hist = volumes.tail(72)
                            mu = float(vol_hist.mean())
                            sigma = float(vol_hist.std()) or 1e-9
                            factors["volume_zscore_24h"] = float(
                                (float(vol_24h.mean()) - mu) / sigma
                            )

                    # Keep a small textual overview of the last few candles
                    candles_overview = df.tail(5).to_string(index=False)
        except Exception as e:
            candles_overview = f"Error parsing candles data: {e}"

        try:
            ob_json = json.loads(order_book_raw)
            if isinstance(ob_json, dict) and "error" in ob_json:
                order_book_overview = str(ob_json)
            else:
                bids = ob_json.get("bids", [])
                asks = ob_json.get("asks", [])
                if bids and asks:
                    best_bid = float(bids[0]["price"])
                    best_ask = float(asks[0]["price"])
                    mid = (best_bid + best_ask) / 2.0
                    spread = best_ask - best_bid
                    factors["best_bid"] = best_bid
                    factors["best_ask"] = best_ask
                    if mid > 0:
                        factors["spread_bps"] = float(spread / mid * 1e4)

                    # Depth imbalance on top 10 levels
                    bid_volume = float(
                        sum(level["amount"] for level in bids[:10])
                    )
                    ask_volume = float(
                        sum(level["amount"] for level in asks[:10])
                    )
                    total_vol = bid_volume + ask_volume
                    if total_vol > 0:
                        factors["depth_imbalance_top10"] = float(
                            (bid_volume - ask_volume) / total_vol
                        )

                    order_book_overview = (
                        f"best_bid={best_bid}, best_ask={best_ask}, "
                        f"bid_vol_top10={bid_volume}, ask_vol_top10={ask_volume}"
                    )
        except Exception as e:
            order_book_overview = f"Error parsing order book data: {e}"

        # Serialize base factor values as JSON
        factor_values_json = json.dumps(factors)

        # Ask LLM to propose additional factor specs in the constrained DSL
        extra_specs = []
        try:
            spec_chain = spec_prompt | llm
            spec_result = spec_chain.invoke({"factor_values_json": factor_values_json})
            spec_content = (
                spec_result.content
                if hasattr(spec_result, "content")
                else str(spec_result)
            )
            parsed = json.loads(spec_content)
            if isinstance(parsed, list):
                extra_specs = parsed
        except Exception:
            extra_specs = []

        # Apply additional factor specs using code, to extend the factor set
        if closes is not None:
            for spec in extra_specs:
                try:
                    name = spec.get("name")
                    ftype = spec.get("type")
                    params = spec.get("params", {})
                    if not name or not isinstance(params, dict):
                        continue

                    if ftype == "momentum":
                        horizon = int(params.get("horizon", 0))
                        if horizon > 0 and len(closes) > horizon:
                            value = float(
                                closes.iloc[-1] / closes.iloc[-horizon] - 1.0
                            )
                            factors[name] = value

                    elif ftype == "ma_ratio":
                        fast = int(params.get("fast", 0))
                        slow = int(params.get("slow", 0))
                        if fast > 0 and slow > 0 and len(closes) >= max(fast, slow):
                            ma_fast = float(closes.tail(fast).mean())
                            ma_slow = float(closes.tail(slow).mean())
                            if ma_slow != 0:
                                factors[name] = float(ma_fast / ma_slow)

                    elif ftype == "vol_ratio":
                        short_window = int(params.get("short_window", 0))
                        long_window = int(params.get("long_window", 0))
                        if (
                            short_window > 1
                            and long_window > 1
                            and len(closes) >= max(short_window, long_window)
                        ):
                            rets = closes.pct_change().dropna()
                            short_vol = float(
                                rets.tail(short_window).std()
                            ) or 0.0
                            long_vol = float(
                                rets.tail(long_window).std()
                            ) or 0.0
                            if long_vol != 0:
                                factors[name] = float(short_vol / long_vol)

                    elif ftype == "volume_zscore" and volumes is not None:
                        window = int(params.get("window", 0))
                        norm_window = int(params.get("norm_window", 0))
                        if (
                            window > 0
                            and norm_window > 1
                            and len(volumes) >= max(window, norm_window)
                        ):
                            vol_short = volumes.tail(window)
                            vol_hist = volumes.tail(norm_window)
                            mu = float(vol_hist.mean())
                            sigma = float(vol_hist.std()) or 1e-9
                            factors[name] = float(
                                (float(vol_short.mean()) - mu) / sigma
                            )
                except Exception:
                    # Skip malformed or non-computable specs
                    continue

        # Serialize final factor values (base + dynamic) as JSON for downstream agents
        factor_values_json = json.dumps(factors)

        interpret_chain = interpret_prompt | llm
        result = interpret_chain.invoke(
            {
                "company_of_interest": state["company_of_interest"],
                "trade_date": state["trade_date"],
                "market_report": state.get("market_report", ""),
                "sentiment_report": state.get("sentiment_report", ""),
                "news_report": state.get("news_report", ""),
                "fundamentals_report": state.get("fundamentals_report", ""),
                "candles_data": candles_overview,
                "order_book_data": order_book_overview,
                "factor_values_json": factor_values_json,
            }
        )
        content = result.content if hasattr(result, "content") else str(result)
        return {
            "factors_spec": content,
            "factor_values": factor_values_json,
            "sender": "Factor Researcher",
            "messages": state["messages"] + [result],
        }

    return node
