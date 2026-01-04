import json
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tradingagents.agents.utils.agent_states import AgentState


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
    local_scope = {"df": df.copy(), "np": np, "pd": pd}

    # Wrap code to ensure it's robust
    wrapper = f"""
def compute(df):
    # Default return if calculation fails within logic
    factor = pd.Series(index=df.index, data=0.0) 
    try:
{code_snippet}
    except Exception:
        pass
    return factor

result_series = compute(df)
"""

    try:
        # Indent the snippet for the wrapper
        indented_snippet = "\n".join(
            ["        " + line for line in code_snippet.split('\n')])
        full_code = wrapper.replace(f"{{code_snippet}}", indented_snippet)

        exec(full_code, {}, local_scope)

        factor_series = local_scope.get("result_series")

        if not isinstance(factor_series, (pd.Series, np.ndarray)):
            return {"success": False, "error": "Code did not return a pandas Series or Array."}

        # Basic Validation
        factor_series = factor_series.fillna(0).replace([np.inf, -np.inf], 0)
        latest_value = float(factor_series.iloc[-1])

        return {
            "success": True,
            "latest_value": latest_value,
            # For visualization if needed
            "series_sample": factor_series.tail(24).tolist(),
            "code": code_snippet
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

    # 1. GENERATOR PROMPT: Asks for Logic + Code
    generator_system = """You are a Senior Quantitative Researcher. 
Your goal is to "mine" a specific alpha factor for the given market conditions.

You have access to a Pandas DataFrame `df` with columns: ['open', 'high', 'low', 'close', 'volume'].
Index is Datetime.

Process:
1. Analyze the provided Market Reports (Sentiment, Macro).
2. Hypothesize: Is the market trending? Mean reverting? Volatile?
3. Design ONE robust factor based on this hypothesis.
4. Write Python Pandas code to calculate this factor.

Constraints:
- The code must assume `df` exists.
- The code MUST assign the final result to a variable named `factor`.
- `factor` must be a pd.Series with the same index as `df`.
- Use purely vectorized operations (no for-loops).
- Handle division by zero using .replace() or fillna.

Example Code:
```python
# Hypothesis: RSI Mean Reversion
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss
factor = 100 - (100 / (1 + rs))
```
"""

    generator_prompt = ChatPromptTemplate.from_messages([
        ("system", generator_system),
        ("human", """
Current Date: {trade_date}
Asset: {symbol}

Market Context:
{market_context}

Task:
1. Explain your hypothesis (Why this factor fits the current regime).
2. Provide the Python code block wrapped in ```python ... ```.
""")
    ])

    # 2. REFLECTION PROMPT: Fixes code if execution fails
    reflection_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Python Code Debugger for Quant Finance."),
        ("human", """
The following factor calculation code failed:

Code:
```python
{code}
```

Error:
{error}

Please fix the code. Ensure it is valid Pandas syntax and defines a variable `factor` (pd.Series).
Return ONLY the python code block.
""")
    ])

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

        # --- 3. Factor Mining (Generation Loop) ---
        chain = generator_prompt | llm
        response = chain.invoke({
            "trade_date": trade_date_str,
            "symbol": symbol,
            "market_context": context_str
        })

        content = response.content

        # simple parsing to extract python code block
        code_block = ""
        if "```python" in content:
            code_block = content.split("```python")[1].split("```")[0].strip()
        elif "```" in content:
            code_block = content.split("```")[1].split("```")[0].strip()

        # --- 4. Execution & Reflection Loop ---
        execution_result = _safe_exec_factor_code(df, code_block)

        # If failed, try to fix ONCE
        if not execution_result["success"]:
            retry_chain = reflection_prompt | llm
            fix_response = retry_chain.invoke({
                "code": code_block,
                "error": execution_result["error"]
            })
            fixed_code = fix_response.content.replace(
                "```python", "").replace("```", "").strip()
            execution_result = _safe_exec_factor_code(df, fixed_code)

        # --- 5. Result Formatting ---
        # We deliberately pass downstream FACTOR DEFINITIONS (code/hypothesis),
        # not the numeric factor time series values.
        hypothesis_text = content.split("```")[0]

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
                    "hypothesis": hypothesis_text.strip(),
                }
            ],
        }

        return {
            "factors_spec": hypothesis_text.strip(),
            "factor_values": json.dumps(factor_definitions),
            "sender": "Factor Researcher",
            "messages": state["messages"] + [response],
        }

    return node
