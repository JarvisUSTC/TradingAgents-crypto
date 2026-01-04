import json


def create_risk_manager(llm, memory):
    """
    Quantitative risk manager that evaluates the final strategy and backtest
    results, and produces a risk-focused decision and control plan.

    This version is designed to sit at the end of the quantitative loop:
    Factor Researcher -> Strategy Designer -> Backtest Runner -> Strategy Evaluator -> Risk Manager.
    """

    def risk_manager_node(state) -> dict:
        company_name = state.get("company_of_interest", "")
        trade_date = state.get("trade_date", "")

        investment_plan = state.get("investment_plan", "")
        market_report = state.get("market_report", "")
        news_report = state.get("news_report", "")
        fundamentals_report = state.get("fundamentals_report", "")
        sentiment_report = state.get("sentiment_report", "")
        research_summary = state.get("research_summary", "")

        factors_spec = state.get("factors_spec", "")
        factor_values_raw = state.get("factor_values", "")
        strategy_explanation = state.get("strategy_template", "")
        strategy_config_raw = state.get("strategy_config", "")
        backtest_results_raw = state.get("backtest_results", "")
        strategy_eval_flag = state.get("strategy_acceptable", False)
        strategy_summary = state.get("selected_strategy_summary", "")

        # Parse JSON payloads where possible for metrics / compact summaries.
        try:
            factor_values = json.loads(factor_values_raw) if factor_values_raw else {}
        except Exception:
            factor_values = {}

        try:
            strategy_config = json.loads(strategy_config_raw) if strategy_config_raw else {}
        except Exception:
            strategy_config = {}

        try:
            backtest_results = json.loads(backtest_results_raw) if backtest_results_raw else {}
        except Exception:
            backtest_results = {}

        # Extract key metrics for prompt readability.
        factor_bt = factor_values.get("mined_factor_backtest", {}) if isinstance(
            factor_values, dict
        ) else {}

        sharpe = backtest_results.get("sharpe_ratio")
        max_dd = backtest_results.get("max_drawdown")
        total_return = backtest_results.get("total_return")

        factor_sharpe = factor_bt.get("sharpe")
        factor_max_dd = factor_bt.get("max_drawdown")

        risk_context = f"""
Company: {company_name}, Trade Date: {trade_date}

High-level investment plan:
{investment_plan}

Qualitative research context:
- Compressed summary: {research_summary}
- Market: {market_report}
- News: {news_report}
- Fundamentals: {fundamentals_report}
- Sentiment: {sentiment_report}

Factor research summary:
{factors_spec}

Factor-level simple backtest (if available):
- factor_sharpe: {factor_sharpe}
- factor_max_drawdown: {factor_max_dd}

Strategy design and explanation (from Strategy Designer / Backtest Runner):
{strategy_explanation}

Strategy backtest summary (from Backtest Runner):
{strategy_summary}

Key backtest metrics (strategy-level):
- sharpe_ratio: {sharpe}
- max_drawdown: {max_dd}
- total_return: {total_return}

StrategyEvaluator acceptable flag: {strategy_eval_flag}
"""

        prompt = f"""You are a quantitative risk manager reviewing a Hummingbot V2 trading strategy.
Your job is to make a risk-focused decision about whether this strategy is acceptable to deploy, and if not,
what risk controls or parameter changes should be applied.

CONTEXT:
{risk_context}

The strategy is implemented as a Hummingbot controller + config (JSON). You may not see the full code here,
but you can infer its behavior from the factor description, strategy explanation, and backtest results.

You MUST:
1. Assess the risk profile of this strategy (concentration, leverage/position sizing if visible, drawdown behavior,
   tail risk, reliance on a single factor, etc.).
2. Decide one of: ACCEPT, ADJUST, or REJECT from a risk perspective:
   - ACCEPT: metrics and risk profile are reasonable for deployment.
   - ADJUST: conceptually acceptable but requires tighter risk parameters (position size, stop-loss/take-profit,
     max_drawdown thresholds, etc.).
   - REJECT: strategy is too risky or unstable given this context.
3. If you choose ADJUST, clearly specify a *risk control plan* that can be implemented in the controller config, such as:
   - max_quote_exposure or total_amount_quote limits
   - per-trade risk caps
   - tighter max_drawdown thresholds
   - disabling short side or leverage, etc.
4. Briefly justify your decision using the factor backtest and strategy backtest metrics.

RESPONSE FORMAT (STRICT):
Start with a single line:
  FINAL_RISK_DECISION: <ACCEPT|ADJUST|REJECT>

Then, on following lines, provide a short markdown-style bullet list under a heading 'RISK_CONTROL_PLAN',
describing concrete risk controls or reasons for rejection/acceptance.
You may reference configuration fields conceptually (e.g. total_amount_quote, max_global_drawdown_quote),
but do NOT output raw JSON for the config here.
"""

        response = llm.invoke(prompt)
        decision_text = response.content if hasattr(response, "content") else str(response)

        # Parse FINAL_RISK_DECISION from the first relevant line
        risk_decision_value = ""
        for line in decision_text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("FINAL_RISK_DECISION:"):
                parts = stripped.split(":", 1)
                if len(parts) == 2:
                    risk_decision_value = parts[1].strip().upper()
                break

        # Derive a simple boolean flag indicating whether a risk-driven
        # adjustment pass should be performed.
        prior_adjustment = state.get("risk_adjustment_pass", False)
        needs_adjustment = risk_decision_value == "ADJUST"
        risk_adjustment_pass = prior_adjustment or needs_adjustment

        # Store the raw decision text as both the final_trade_decision (for UI)
        # and into the risk_debate_state["judge_decision"] for consistency with
        # the existing AgentState schema.
        new_risk_debate_state = state.get("risk_debate_state", {}) or {}
        new_risk_debate_state["judge_decision"] = decision_text

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": decision_text,
            "risk_control_plan": decision_text,
            "risk_decision": risk_decision_value,
            "risk_adjustment_pass": risk_adjustment_pass,
        }

    return risk_manager_node
