from typing import Dict, Any
import json
from langchain_core.prompts import ChatPromptTemplate

from tradingagents.agents.utils.agent_states import AgentState


def create_backtest_runner(llm, toolkit):
    """
    Create a backtest runner node that:
    - Sends the current strategy_config to hummingbot-api backtesting endpoint
    - Parses and summarizes key performance metrics
    """

    system = """You are a backtest analyst.
You receive raw backtest results from hummingbot-api and summarize the key performance metrics.

Focus on:
- Sharpe ratio
- Annualized return
- Maximum drawdown
- Win rate and trade statistics if available

Your output should:
- Start with a short bullet list of key metrics.
- Then provide a concise prose summary of the strategy's performance.
"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """You are evaluating the performance of a strategy for {company_of_interest} on {trade_date}.

The raw backtest results JSON from hummingbot-api are:
{raw_results}

Summarize the key metrics and provide a concise evaluation of this strategy's performance.""",
            ),
        ]
    )

    def node(state: AgentState) -> Dict[str, Any]:
        config_json = state.get("strategy_config", "")
        if not config_json:
            return {
                "backtest_results": "No strategy_config available to run backtest.",
                "sender": "Backtest Runner",
            }

        # hb_run_backtest is a StructuredTool; use .func for direct call here.
        # Use a 30-day window ending at trade_date with default resolution and trade cost.
        trade_date = state.get("trade_date", "")
        raw_results = toolkit.hb_run_backtest.func(
            config_json,
            trade_date,
        )

        chain = prompt | llm
        result = chain.invoke(
            {
                "company_of_interest": state["company_of_interest"],
                "trade_date": state["trade_date"],
                "raw_results": raw_results,
            }
        )
        content = result.content if hasattr(result, "content") else str(result)

        return {
            "backtest_results": raw_results,
            "selected_strategy_summary": content,
            "sender": "Backtest Runner",
            "messages": state["messages"] + [result],
        }

    return node
