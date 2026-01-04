import json
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from tradingagents.agents.utils.agent_states import AgentState


class AlphaLib:
    """
    Built-in factor library for LLM to use.
    Functions are designed to work on pandas Series or DataFrames with
    DatetimeIndex and one or more asset columns.
    """

    @staticmethod
    def MA(series, window):
        """Moving Average"""
        return series.rolling(window).mean()

    @staticmethod
    def STD(series, window):
        """Standard Deviation"""
        return series.rolling(window).std()

    @staticmethod
    def ROC(series, window):
        """Rate of Change: (Price_t - Price_{t-n}) / Price_{t-n}"""
        return series.pct_change(window)

    @staticmethod
    def RESI(close, window):
        """Residual: Price - MA(Price)"""
        return close - close.rolling(window).mean()

    @staticmethod
    def WVMA(close, volume, window):
        """Weighted Volume Moving Average (rolling VWAP approximation)"""
        pv = close * volume
        return pv.rolling(window).sum() / volume.rolling(window).sum()

    @staticmethod
    def VSTD(volume, window):
        """Volume Standard Deviation"""
        return volume.rolling(window).std()

    @staticmethod
    def KLEN(high, low, open_price):
        """K-Line Length: (High - Low) / Open"""
        return (high - low) / open_price.replace(0, np.nan)

    @staticmethod
    def KLOW(close, high, low):
        """Position of Close within High-Low Range"""
        denom = (high - low).replace(0, np.nan)
        return (close - low) / denom

    @staticmethod
    def CORR(series, window):
        """Auto-Correlation (Simple Trend Strength)"""
        return series.rolling(window).corr(series.shift(1))

    @staticmethod
    def RSQR(series, window):
        """R-Squared (Trend Strength Squared)"""
        return series.rolling(window).corr(series.shift(1)) ** 2

    @staticmethod
    def CORD(series, window):
        """Volatility of Changes (Stability)"""
        return series.diff().rolling(window).std()


alpha_lib = AlphaLib()


def _to_trading_pair(symbol: str) -> str:
    """Convert a ticker like BTC to a trading pair like BTC-USDT."""
    if "-" in symbol:
        return symbol
    if "/" in symbol:
        return symbol.replace("/", "-")
    return f"{symbol}-USDT"


def _safe_exec_factor_code(df: pd.DataFrame, code_snippet: str) -> Dict[str, Any]:
    """
    Executes LLM-generated pandas code in a restricted local scope.
    The code is expected to assume 'df' exists and return a pd.Series named 'factor'.
    """
    local_scope = {
        "df": df.copy(),
        "np": np,
        "pd": pd,
        "pdSeries": pd.Series,
        "lib": alpha_lib,
    }

    # Normalize and clean the snippet so it can live under a try: block
    snippet = code_snippet.strip()
    if "```" in snippet:
        snippet = snippet.replace("```python", "").replace("```", "").strip()

    # Drop any import lines (e.g. "import numpy as np", "import pandas as pd")
    cleaned_lines: List[str] = []
    for line in snippet.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        cleaned_lines.append(line)
    snippet = "\n".join(cleaned_lines).strip()

    indented_snippet = "\n".join(
        ["        " + line for line in snippet.split("\n")]
    )

    wrapper = (
        "def compute(df):\n"
        "    # Default return if calculation fails within logic\n"
        "    factor = pd.Series(index=df.index, data=0.0)\n"
        "    try:\n"
        f"{indented_snippet}\n"
        "    except Exception:\n"
        "        pass\n"
        "    return factor\n"
        "\n"
        "result_series = compute(df)\n"
    )

    try:
        # Execute wrapper in a controlled global namespace
        exec(wrapper, local_scope)

        factor_series = local_scope.get("result_series")

        if not isinstance(factor_series, (pd.Series, np.ndarray)):
            return {"success": False, "error": "Code did not return a pandas Series or Array."}

        # Basic Validation
        factor_series = pd.Series(factor_series, index=df.index).fillna(0).replace(
            [np.inf, -np.inf], 0
        )
        latest_value = float(factor_series.iloc[-1])

        return {
            "success": True,
            "latest_value": latest_value,
            "series": factor_series,
            "series_sample": factor_series.tail(24).tolist(),
            "code": code_snippet,
        }

    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {str(e)}"}


def _simple_factor_backtest(df: pd.DataFrame, factor_series: pd.Series) -> Dict[str, Any]:
    """
    Run a very simple long/short backtest on the factor series over the df['close'] returns.
    Used only locally inside this node as feedback for factor refinement.
    """
    try:
        if df.empty or len(df) < 10:
            return {"success": False, "error": "Not enough data for backtest."}

        closes = df["close"].astype(float)
        rets = closes.pct_change().fillna(0.0)

        factor_aligned = pd.Series(factor_series, index=df.index).fillna(0.0)
        positions = np.sign(factor_aligned).shift(1).fillna(0.0)

        strat_rets = positions * rets

        equity = (1.0 + strat_rets).cumprod()
        peak = equity.cummax()
        drawdown = equity / peak - 1.0
        max_dd = float(drawdown.min())

        mean_ret = float(strat_rets.mean())
        std_ret = float(strat_rets.std() or 0.0)
        sharpe = float(mean_ret / std_ret *
                       np.sqrt(len(strat_rets))) if std_ret > 0 else 0.0

        total_ret = float(equity.iloc[-1] - 1.0)
        hit_rate = float((strat_rets > 0).mean())

        return {
            "success": True,
            "total_return": total_ret,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "hit_rate": hit_rate,
            "avg_return": mean_ret,
            "std_return": std_ret,
            "n_trades": int((positions != 0).sum()),
        }
    except Exception as e:
        return {"success": False, "error": f"{type(e).__name__}: {str(e)}"}


def create_factor_researcher(llm, toolkit):
    """
    An advanced Factor Miner that generates Python code to discover trading signals.

    Design goals:
    - Use STRICTLY historical candles up to the given trade_date (no look-ahead).
    - Let the LLM propose a single alpha factor as Python/pandas code.
    - Execute and auto-fix the code in a guarded sandbox for sanity checks.
    - Pass downstream the FACTOR CODE (and hypothesis), not the numeric values.
    """

    # GENERATOR PROMPT: Think trading idea first, then code using AlphaLib
    generator_system = """You are a Senior Quantitative Researcher.
Your goal is to mine a robust alpha factor for the given market conditions.

DATA INPUT:
- You have a Pandas DataFrame `df` with columns: ['open', 'high', 'low', 'close', 'volume'].
- Index is Datetime for a single asset (or a small set of assets).

BUILT-IN FACTOR LIBRARY (global object `lib`):
You MUST PREFER using these functions over writing raw pandas rolling logic where possible:
- lib.MA(series, win), lib.STD(series, win), lib.ROC(series, win)
- lib.RESI(close, win): Residuals (Price - MA)
- lib.WVMA(close, vol, win): Volume Weighted MA (rolling VWAP approximation)
- lib.VSTD(vol, win): Volume Std Dev
- lib.KLEN(high, low, open): (High - Low) / Open (K-line length)
- lib.KLOW(close, high, low): (Close - Low) / (High - Low)
- lib.CORR(series, win), lib.RSQR(series, win): Trend strength
- lib.CORD(series, win): Volatility/Stability of changes

PROCESS:
1. Based on the provided market context (research reports, news), articulate ONE clear factor idea:
   - e.g., mean reversion with volume confirmation, trend following with stability filter, etc.
2. Then write Python code that implements this idea as a single factor.

CONSTRAINTS FOR CODE:
- Assume `df` exists and contains OHLCV columns.
- You may use the global `lib` object (AlphaLib) as described above.
- You MUST assign the final result to a variable named `factor`.
- `factor` must be a pd.Series with the same index as `df`.
- Use purely vectorized operations (no for-loops).
- Handle NaNs and division by zero robustly using fillna/replace/clip, etc.

RESPONSE FORMAT:
- First, briefly write your factor idea in natural language (1-3 sentences).
- Then provide ONLY a Python code block wrapped in ```python ... ``` that defines the calculation of `factor`.
"""

    def node(state: AgentState) -> Dict[str, Any]:
        symbol = state["company_of_interest"]
        trade_date_str = state["trade_date"]

        # --- 1. Data Fetching (STRICT, no look-ahead) ---
        trading_pair = _to_trading_pair(symbol)

        try:
            # Use the same 30-day window convention as backtesting:
            # [trade_date - 30 days, trade_date], at 1h resolution.
            end_dt = datetime.strptime(trade_date_str, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=30)
            start_ts = int(start_dt.timestamp())
            end_ts = int(end_dt.timestamp())

            candles_raw = toolkit.hb_get_historical_candles.func(
                trading_pair,
                "1h",
                start_ts,
                end_ts,
            )
            df = pd.DataFrame(json.loads(candles_raw))

            if not df.empty and "timestamp" in df.columns:
                df = df.sort_values("timestamp")
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = df[col].astype(float)

            if df.empty or len(df) < 50:
                return {
                    "factors_spec": "Insufficient historical data to mine factors.",
                    "factor_values": json.dumps({}),
                    "sender": "Factor Researcher",
                    "messages": state["messages"],
                }

        except Exception as e:
            return {
                "factors_spec": f"Data Error while fetching historical candles: {e}",
                "factor_values": json.dumps({}),
                "sender": "Factor Researcher",
                "messages": state["messages"],
            }

        # --- 2. Context Preparation ---
        context_str = (
            "Market Report: "
            + state.get("market_report", "N/A")
            + "\n\nNews: "
            + state.get("news_report", "N/A")
        )

        # --- 3. Factor Mining + Multi-Turn Improvement Loop (with feedback) ---
        # We run a short internal conversation with the LLM:
        #   - First message: factor research mission with market context.
        #   - Each iteration: LLM proposes code -> we execute + backtest ->
        #     we append a feedback "user" message with backtest results and ask for refinement.
        initial_user = (
            f"You are mining an alpha factor for {symbol} on {trade_date_str}.\n\n"
            f"Market context:\n{context_str}\n\n"
            "You are given a pandas DataFrame `df` with 1h OHLCV data for the last 30 days "
            f"ending at {trade_date_str}. Using the instructions in the system prompt, first think of ONE clear "
            "factor idea (trend following, mean reversion, volatility/volume-based, etc.), then provide the "
            "Python code that computes a pd.Series named `factor` using `df` and the `lib` factor library.\n"
            "Return your explanation followed by a ```python ... ``` block with the factor code."
        )

        convo: List[Any] = [
            SystemMessage(content=generator_system),
            HumanMessage(content=initial_user),
        ]

        best_code = ""
        best_result: Dict[str, Any] = {
            "success": False, "error": "No code tried."}
        best_metrics: Dict[str, Any] = {}
        best_explanation = ""
        last_ai_msg: AIMessage | None = None

        max_rounds = 5
        for _ in range(max_rounds):
            ai_msg = llm.invoke(convo)
            last_ai_msg = ai_msg
            convo.append(ai_msg)

            content = ai_msg.content if hasattr(
                ai_msg, "content") else str(ai_msg)

            # Extract python code block
            code_block = ""
            if "```python" in content:
                code_block = content.split("```python")[
                    1].split("```")[0].strip()
            elif "```" in content:
                code_block = content.split("```")[1].split("```")[0].strip()

            if not code_block:
                # Ask LLM to provide code
                convo.append(
                    HumanMessage(
                        content=(
                            "System: No valid Python code block was found. "
                            "Please provide the factor implementation in a ```python ... ``` block, "
                            "assigning the result to a pd.Series named `factor`."
                        )
                    )
                )
                continue

            # Execute code
            exec_result = _safe_exec_factor_code(df, code_block)
            if not exec_result.get("success"):
                error_msg = exec_result.get(
                    "error", "Unknown execution error.")
                convo.append(
                    HumanMessage(
                        content=(
                            "System: Your factor code raised an error during execution:\n"
                            f"{error_msg}\n"
                            "Please fix the code while keeping the same high-level idea. "
                            "Use only vectorized operations and ensure `factor` is a pd.Series "
                            "aligned with df.index."
                        )
                    )
                )
                continue

            # Run simple backtest on this factor
            metrics = {}
            if "series" in exec_result:
                metrics = _simple_factor_backtest(df, exec_result["series"])

            if not metrics.get("success"):
                convo.append(
                    HumanMessage(
                        content=(
                            "System: Your factor code executed, but the simple backtest could not be run properly. "
                            "Please make sure the factor is not constant and has enough variation, "
                            "then send an improved version of the factor code."
                        )
                    )
                )
                continue

            # Update best if better Sharpe / return
            if not best_metrics.get("success"):
                is_better = True
            else:
                old_sharpe = best_metrics.get("sharpe", 0.0)
                new_sharpe = metrics.get("sharpe", 0.0)
                old_ret = best_metrics.get("total_return", 0.0)
                new_ret = metrics.get("total_return", 0.0)
                is_better = (new_sharpe > old_sharpe) or (
                    np.isclose(new_sharpe, old_sharpe) and new_ret > old_ret
                )

            if is_better:
                best_code = code_block
                best_result = exec_result
                best_metrics = metrics
                best_explanation = content.split("```")[0].strip()

            # If Sharpe reasonably high, we can stop early
            if metrics.get("sharpe", 0.0) >= 1.5:
                break

            # Otherwise, feed back metrics and ask for refinement
            convo.append(
                HumanMessage(
                    content=(
                        "System: Simple backtest results for your factor over the last 30 days (1h bars):\n"
                        f"- total_return: {metrics.get('total_return', 0.0):.4f}\n"
                        f"- sharpe: {metrics.get('sharpe', 0.0):.3f}\n"
                        f"- max_drawdown: {metrics.get('max_drawdown', 0.0):.3f}\n"
                        f"- hit_rate: {metrics.get('hit_rate', 0.0):.3f}\n\n"
                        "Please refine the factor (parameters or functional form) to improve Sharpe and reduce drawdown, "
                        "without overfitting. Keep using the `lib` functions and return ONLY updated Python code in a "
                        "```python ... ``` block."
                    )
                )
            )

        execution_result = best_result

        # --- 5. Result Formatting ---
        # We deliberately pass downstream FACTOR DEFINITIONS (code/hypothesis),
        # plus a compact summary of the simple backtest metrics used for refinement.
        if best_explanation:
            hypothesis_text = best_explanation
        else:
            last_content = (
                last_ai_msg.content
                if last_ai_msg is not None and hasattr(last_ai_msg, "content")
                else ""
            )
            hypothesis_text = last_content.split("```")[0].strip()

        metrics_summary = ""
        if best_metrics.get("success"):
            tr = best_metrics.get("total_return", 0.0)
            sh = best_metrics.get("sharpe", 0.0)
            dd = best_metrics.get("max_drawdown", 0.0)
            hr = best_metrics.get("hit_rate", 0.0)
            metrics_summary = (
                "\n\n[Simple Factor Backtest] "
                f"total_return={tr:.4f}, sharpe={sh:.3f}, "
                f"max_drawdown={dd:.3f}, hit_rate={hr:.3f}"
            )

        factors_spec_text = hypothesis_text + metrics_summary

        factor_definitions = {
            "base_factors": [
                {
                    "name": "close",
                    "kind": "builtin",
                    "formula": "df['close'].iloc[-1]",
                    "description": "Latest close price on 1h bars over the past 30 days.",
                },
                {
                    "name": "ret_24h",
                    "kind": "builtin",
                    "formula": "df['close'].pct_change(24).iloc[-1]",
                    "description": "24-hour close-to-close return computed from 1h bars.",
                },
            ],
            "mined_factors": [
                {
                    "name": "llm_mined_alpha",
                    "kind": "llm_generated",
                    "valid": execution_result.get("success", False),
                    "code": execution_result.get("code", ""),
                    "hypothesis": hypothesis_text,
                }
            ],
        }

        if best_metrics:
            factor_definitions["mined_factor_backtest"] = best_metrics

        return {
            "factors_spec": factors_spec_text,
            "factor_values": json.dumps(factor_definitions),
            "sender": "Factor Researcher",
            "messages": state["messages"]
            + ([last_ai_msg] if last_ai_msg is not None else []),
        }

    return node
