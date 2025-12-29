from typing import Dict, Any
import json

from tradingagents.agents.utils.agent_states import AgentState


def create_strategy_evaluator(llm):
    """
    Evaluate whether the current strategy/backtest is acceptable based on metrics.
    This node inspects backtest_results (JSON string) and sets strategy_acceptable.
    """

    def node(state: AgentState) -> Dict[str, Any]:
        raw = state.get("backtest_results", "")
        acceptable = False
        sharpe = None
        max_drawdown = None

        try:
            data = json.loads(raw)
            sharpe = data.get("sharpe_ratio")
            max_drawdown = data.get("max_drawdown")
        except Exception:
            data = None

        if isinstance(sharpe, (int, float)):
            if sharpe is not None and sharpe >= 1.0:
                acceptable = True
        if acceptable and isinstance(max_drawdown, (int, float)):
            if max_drawdown is not None and max_drawdown > 0.5:
                acceptable = False

        return {
            "strategy_acceptable": acceptable,
            "sender": "Strategy Evaluator",
        }

    return node

