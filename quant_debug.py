import os
import sys
import json
from datetime import datetime

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents import (
    create_factor_researcher,
    create_strategy_designer,
    create_backtest_runner,
)


def run_short_backtest(ticker: str, trade_date: str) -> None:
    """
    Run a minimal quant pipeline:
    FactorResearcher -> StrategyDesigner -> BacktestRunner
    and print backtest results for quick iteration.
    """
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = "openai"  # Use a different model
    # Use a different backend
    config["backend_url"] = "https://cloud-router-dev.lobehub.com/v1"
    config["deep_think_llm"] = "o4-mini-fallback"  # Use a different model
    config["quick_think_llm"] = "gpt-5-mini-fallback"  # Use a different model
    config["max_debate_rounds"] = 1  # Increase debate rounds
    config["online_tools"] = True  # Increase debate rounds
    config["api_key"] = os.environ.get("OPENAI_API_KEY", "")

    graph = TradingAgentsGraph(
        # We do not use the main graph here, but TradingAgentsGraph
        # sets up LLMs, toolkit, and dataflow config for us.
        selected_analysts=["market"],
        debug=False,
        config=config,
    )

    llm = graph.quick_thinking_llm
    toolkit = graph.toolkit

    factor_node = create_factor_researcher(llm, toolkit)
    strategy_node = create_strategy_designer(llm, toolkit)
    backtest_node = create_backtest_runner(llm, toolkit)

    state = graph.propagator.create_initial_state(ticker, trade_date)

    # Run only the quant steps, in-process, without going through the full LangGraph.
    for node_fn in (factor_node, strategy_node, backtest_node):
        updates = node_fn(state)
        if updates:
            state.update(updates)

    strategy_config = state.get("strategy_config")
    backtest_results = state.get("backtest_results")

    print("\n=== Short Quant Pipeline Result ===")
    print(f"Ticker: {ticker}, Trade date: {trade_date}")

    print("\n[Strategy Config]")
    print(strategy_config)

    print("\n[Backtest Results]")
    print(backtest_results)

    # Try to extract a few key metrics if JSON-shaped
    try:
        results_json = json.loads(backtest_results)
        sharpe = results_json.get("sharpe_ratio")
        max_dd = results_json.get("max_drawdown")
        print("\n[Parsed Metrics]")
        print(f"Sharpe ratio: {sharpe}")
        print(f"Max drawdown: {max_dd}")
    except Exception:
        pass


if __name__ == "__main__":
    # Usage:
    #   python quant_debug.py BTC-USDT 2024-05-10
    # ticker can also be a bare symbol like BTC; it will be mapped to BTC-USDT.
    if len(sys.argv) < 3:
        print(
            "Usage: python quant_debug.py <TICKER_OR_PAIR> <TRADE_DATE_YYYY-MM-DD>\n"
            "Example: python quant_debug.py BTC-USDT 2024-05-10"
        )
        sys.exit(1)

    ticker_arg = sys.argv[1]
    date_arg = sys.argv[2]

    # Basic sanity check on date format; let downstream raise if invalid
    try:
        datetime.strptime(date_arg, "%Y-%m-%d")
    except ValueError:
        print("TRADE_DATE must be in YYYY-MM-DD format.")
        sys.exit(1)

    run_short_backtest(ticker_arg, date_arg)
