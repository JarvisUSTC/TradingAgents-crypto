# TradingAgents/graph/conditional_logic.py

from tradingagents.agents.utils.agent_states import AgentState


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(
        self,
        max_debate_rounds: int = 1,
        max_risk_discuss_rounds: int = 1,
        max_backtest_rounds: int = 3,
    ):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds
        self.max_backtest_rounds = max_backtest_rounds

    def should_continue_market(self, state: AgentState):
        """Determine if market analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_market"
        return "Msg Clear Market"

    def should_continue_social(self, state: AgentState):
        """Determine if social media analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_social"
        return "Msg Clear Social"

    def should_continue_news(self, state: AgentState):
        """Determine if news analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_news"
        return "Msg Clear News"

    def should_continue_fundamentals(self, state: AgentState):
        """Determine if fundamentals analysis should continue."""
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools_fundamentals"
        return "Msg Clear Fundamentals"

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 3 rounds of back-and-forth between 2 agents
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 rounds of back-and-forth between 3 agents
            return "Risk Judge"
        if state["risk_debate_state"]["latest_speaker"].startswith("Risky"):
            return "Safe Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Safe"):
            return "Neutral Analyst"
        return "Risky Analyst"

    def should_continue_backtesting(self, state: AgentState) -> str:
        """
        Determine if backtesting and strategy refinement should continue.

        If strategy_acceptable is True, or the maximum number of
        backtest rounds has been reached, stop the graph (END).
        Otherwise, loop back to Factor Researcher so that both factors
        and strategy can be refined using the latest backtest results.
        """
        # Track how many backtest iterations have been run
        current_round = state.get("backtest_round", 0) + 1
        state["backtest_round"] = current_round

        if state.get("strategy_acceptable"):
            return "END"
        if current_round >= self.max_backtest_rounds:
            return "END"
        return "Factor Researcher"

    def should_apply_risk_adjustment(self, state: AgentState) -> str:
        """
        Decide whether to send the flow back to Strategy Designer for a
        risk-driven adjustment pass, or end the graph.

        We only allow a single risk-adjustment pass. If `risk_adjustment_pass`
        is already True, or the risk decision is not ADJUST, we return End.
        """
        # If we've already done a risk adjustment pass, do not loop again.
        if state.get("risk_adjustment_pass"):
            return "End"

        decision = (state.get("risk_decision") or "").strip().upper()
        if decision == "ADJUST":
            return "Adjust"
        return "End"
