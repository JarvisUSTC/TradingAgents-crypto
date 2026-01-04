from typing import Dict, Any, List
import json
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

from tradingagents.agents.utils.agent_states import AgentState


def create_strategy_designer(llm, toolkit):
    """
    Create a strategy designer node that:
    - Reads factor specification and values
    - Uses Hummingbot controller tools via a tool-enabled agent
    - Prefers creating a NEW controller (code + config) in Hummingbot
    - Produces a concrete controller config JSON string for backtesting
    """

    controller_tools = [
        toolkit.hb_list_controllers,
        toolkit.hb_list_controller_configs,
        toolkit.hb_get_controller_template,
        toolkit.hb_get_controller,
        toolkit.hb_get_controller_config,
        toolkit.hb_validate_controller_config,
        toolkit.hb_create_or_update_controller,
        toolkit.hb_create_or_update_controller_config,
    ]

    system = """You are a quantitative strategy designer for Hummingbot V2.
You receive research outputs (investment plan, quantitative factors, and optional risk preferences)
and must design a FULL Hummingbot controller strategy that can be saved and backtested.

You have access to CONTROLLER tools that let you:
- Inspect existing controllers and configs
- Create or update controller PYTHON code
- Validate and save controller CONFIGS

Design principles:
- By default, PREFER creating a NEW controller with a NEW controller_name that does not clash
  with existing controllers. You may read existing controllers for inspiration, but do not just
  tweak parameters of an existing one unless explicitly instructed.
- Encode risk management directly into the strategy (position sizing, leverage limits,
  stop loss / take profit logic, max drawdown parameters, etc.), consistent with any risk context.
- If the input includes a prior risk manager decision or a 'Risk manager final decision and control plan',
  treat this as a risk-adjustment pass: keep the core factor/strategy idea, but MODIFY the controller
  config and (if necessary) controller code to implement those risk controls explicitly.
- IMPORTANT ABOUT CONFIG INHERITANCE: If you choose to implement a new config class that inherits from
  an existing Hummingbot config base class, you MUST NOT add any new configuration fields beyond what
  the base class defines, because the base class schema forbids unknown fields. If you truly need new
  configuration parameters, DO NOT use inheritance from that base class; instead, define an appropriate
  config class that matches the fields you intend to use.
- Make sure the config is consistent with the controller_type and controller_name, and is
  suitable for the backtest horizon used in this system (recent 30 days by default).

Expected tool usage (you may adapt as needed):
1) Optionally call hb_list_controllers and hb_list_controller_configs to understand what exists.
2) Propose a NEW controller_type and controller_name and generate full controller_code.
3) Call hb_create_or_update_controller with the code so Hummingbot can load it.
4) Design a full config object for this controller, then call hb_validate_controller_config.
5) Call hb_create_or_update_controller_config to save the config.

When you are completely done (all necessary tools have been called), respond with exactly ONE line:
STRATEGY_RESULT_JSON: { ... }

The JSON object MUST have these keys:
- "controller_type": string, one of ["directional_trading", "market_making", "generic"].
- "controller_name": string, usually a NEW controller name you defined for this strategy.
- "config_name": string, slug identifier for this specific strategy config (no spaces, e.g. "btc_bollinger_factors_v1").
- "config": object, the full config for this strategy, including at least:
    - controller_name
    - controller_type
    - connector_name
    - trading_pair
    - any other required fields for the controller to run
- "explanation": brief natural-language summary of the strategy logic and key risk controls.
You may optionally include:
- "controller_code": string with the Python source code you created or updated.

The JSON MUST be valid, with no comments or trailing commas."""

    def node(state: AgentState) -> Dict[str, Any]:
        # Build risk context if any is present
        risk_context_parts: List[str] = []
        final_trade_decision = state.get("final_trade_decision", "")
        if final_trade_decision:
            risk_context_parts.append(
                f"Final trade decision from risk manager:\n{final_trade_decision}"
            )
        risk_state = state.get("risk_debate_state")
        if isinstance(risk_state, dict) and risk_state.get("judge_decision"):
            risk_context_parts.append(
                f"Risk debate judge decision:\n{risk_state.get('judge_decision')}"
            )
        # Include any explicit risk control plan produced by the quantitative
        # Risk Manager on a prior pass.
        risk_control_plan = state.get("risk_control_plan", "")
        if risk_control_plan:
            risk_context_parts.append(
                "Risk manager final decision and control plan:\n" + risk_control_plan
            )

        risk_context = (
            "\n\n".join(risk_context_parts)
            if risk_context_parts
            else "No explicit risk context available in this run."
        )

        company = state["company_of_interest"]
        trade_date = state["trade_date"]

        human = HumanMessage(
            content=(
                f"You are designing a Hummingbot controller strategy for {company} on {trade_date}.\n\n"
                f"Investment plan:\n{state.get('investment_plan', '')}\n\n"
                f"Compressed research summary (for context):\n{state.get('research_summary', '')}\n\n"
                f"Quantitative factors specification / hypotheses:\n{state.get('factors_spec', '')}\n\n"
                f"Quantitative factor DEFINITIONS / code specs (JSON string):\n{state.get('factor_values', '')}\n\n"
                f"Backtest results (may be empty on first run):\n{state.get('backtest_results', '')}\n\n"
                f"Risk management context (if any):\n{risk_context}\n\n"
                "You may assume the trading universe is crypto, and you can use symbols like BTC-USDT, ETH-USDT, etc., "
                "consistent with the company_of_interest.\n\n"
                "Use the available controller tools to create or update a NEW controller and its config, "
                "then return STRATEGY_RESULT_JSON as specified in the system message."
            )
        )

        messages: List[Any] = [SystemMessage(content=system), human]

        tool_llm = llm.bind_tools(controller_tools)

        last_ai: AIMessage | None = None
        max_iterations = 8

        for _ in range(max_iterations):
            ai_msg = tool_llm.invoke(messages)
            last_ai = ai_msg
            messages.append(ai_msg)

            tool_calls = getattr(ai_msg, "tool_calls", None) or []
            if tool_calls:
                for call in tool_calls:
                    name = call.get("name")
                    args = call.get("args") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    tool_obj = None
                    for t in controller_tools:
                        if t.name == name:
                            tool_obj = t
                            break
                    if tool_obj is None:
                        result_text = f"Tool '{name}' is not available."
                    else:
                        try:
                            result_text = tool_obj.func(**args)
                        except Exception as exc:
                            result_text = f"Error while executing tool '{name}': {exc}"
                    messages.append(
                        ToolMessage(
                            content=str(result_text),
                            tool_call_id=call.get("id", ""),
                        )
                    )
                continue

            content_text = (
                ai_msg.content if hasattr(ai_msg, "content") else str(ai_msg)
            )
            if "STRATEGY_RESULT_JSON:" in content_text:
                break
            # No tools and no final JSON marker: stop to avoid infinite loop
            break

        if last_ai is None:
            return {
                "strategy_template": "Strategy designer did not produce any output.",
                "strategy_config": "",
                "sender": "Strategy Designer",
                "messages": state["messages"],
            }

        final_content = (
            last_ai.content if hasattr(last_ai, "content") else str(last_ai)
        )
        marker = "STRATEGY_RESULT_JSON:"
        spec_json: Dict[str, Any] | None = None
        if marker in final_content:
            _, tail = final_content.split(marker, 1)
            spec_str = tail.strip()
            try:
                spec_json = json.loads(spec_str)
            except Exception:
                spec_json = None

        controller_type = ""
        controller_name = ""
        config_name = ""
        config_obj: Dict[str, Any] = {}
        controller_code: str | None = None
        explanation = final_content

        if isinstance(spec_json, dict):
            controller_type = spec_json.get("controller_type", "") or ""
            controller_name = spec_json.get("controller_name", "") or ""
            config_name = spec_json.get("config_name", "") or ""
            raw_config = spec_json.get("config", {})
            if isinstance(raw_config, dict):
                config_obj = raw_config
            explanation = spec_json.get("explanation", explanation) or explanation
            code_val = spec_json.get("controller_code")
            if isinstance(code_val, str) and code_val.strip():
                controller_code = code_val

        if not config_obj:
            try:
                config_obj = json.loads(state.get("factor_values", "{}"))
            except Exception:
                config_obj = {}

        if controller_type:
            config_obj.setdefault("controller_type", controller_type)
        if controller_name:
            config_obj.setdefault("controller_name", controller_name)
        if not config_name and controller_name:
            config_name = f"{controller_name}_auto"
        if config_name:
            config_obj.setdefault("id", config_name)

        config_json = json.dumps(config_obj)

        validation_info = ""
        upsert_info = ""
        controller_upsert_info = ""

        # Prefer to create or update controller code first if provided
        if controller_code and controller_type and controller_name:
            try:
                controller_upsert_info = toolkit.hb_create_or_update_controller.func(
                    controller_type,
                    controller_name,
                    controller_code,
                )
            except Exception as exc:
                controller_upsert_info = (
                    f"Failed to create/update controller code via hummingbot-api: {exc}"
                )

        if controller_type and controller_name and config_json:
            try:
                validation_info = toolkit.hb_validate_controller_config.func(
                    controller_type, controller_name, config_json
                )
            except Exception as exc:
                validation_info = f"Validation failed or unavailable: {exc}"

        if config_name and config_json:
            try:
                upsert_info = toolkit.hb_create_or_update_controller_config.func(
                    config_name, config_json
                )
            except Exception as exc:
                upsert_info = (
                    f"Failed to create/update controller config via hummingbot-api: {exc}"
                )

        technical_lines: List[str] = []
        technical_lines.append(
            f"Strategy controller: type={controller_type}, name={controller_name}, config_name={config_name}"
        )
        if controller_upsert_info:
            technical_lines.append(
                f"Controller code upsert result: {controller_upsert_info}"
            )
        if validation_info:
            technical_lines.append(f"Config validation result: {validation_info}")
        if upsert_info:
            technical_lines.append(f"Config upsert result: {upsert_info}")

        technical_summary = ""
        if technical_lines:
            technical_summary = "\n\n[HUMMINGBOT_API_ACTIONS]\n" + "\n".join(
                technical_lines
            )

        narrative = explanation + technical_summary

        return {
            "strategy_template": narrative,
            "strategy_config": config_json or narrative,
            "sender": "Strategy Designer",
            "messages": state["messages"] + [last_ai],
        }

    return node
