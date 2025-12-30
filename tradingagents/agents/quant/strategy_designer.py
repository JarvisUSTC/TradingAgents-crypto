from typing import Dict, Any, List, Tuple
import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from tradingagents.agents.utils.agent_states import AgentState


def create_strategy_designer(llm, toolkit):
    """
    Create a strategy designer node that:
    - Reads factor specification and values
    - Calls hummingbot controller template/config endpoints
    - Produces a concrete controller config JSON string for backtesting
    """

    system = """You are a quantitative strategy designer for algorithmic trading.
You design concrete strategy configurations for a hummingbot-based execution engine.

You MUST:
- First, choose an appropriate controller type and name based on the factors and investment plan.
- Then, use the controller's configuration TEMPLATE from hummingbot-api as the base for the final config.
- Only adjust parameter VALUES; do not invent new fields that are not in the template unless clearly optional.
- When previous backtest results are available, use them to refine or adjust the strategy parameters to improve performance.
- Produce a final JSON configuration string that hummingbot-api can accept for backtesting.

Final output structure (for the whole node):
1) Brief natural language explanation of the chosen controller and parameters.
2) A JSON configuration object on a separate line, starting with 'CONFIG_JSON:'.
The JSON must be valid, compact, and self-contained."""

    # Step 2: design config using the selected template
    config_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """You are designing a strategy for {company_of_interest} on {trade_date}.

Investment plan (human research team decision):
{investment_plan}

Quantitative factors specification:
{factors_spec}

Quantitative factor values and signals:
{factor_values}

Backtest results and performance summary (may be empty on first run):
{backtest_results}

You have selected the following hummingbot controller:
controller_type = {controller_type}
controller_name = {controller_name}

The configuration TEMPLATE for this controller from hummingbot-api is:
{controller_template}

Use this TEMPLATE as the basis for your design instead of inventing new schema.
You may adjust parameter values, thresholds, and toggles, but should not remove required fields.

Design ONE concrete strategy configuration suitable for backtesting.
Explain your choice briefly, then output the final JSON config on a separate line starting with:
CONFIG_JSON: <json>""",
            ),
        ]
    )

    def node(state: AgentState) -> Dict[str, Any]:
        # Step 0: fetch available controllers from hummingbot-api as context
        try:
            # hb_list_controllers is a StructuredTool; use .func for direct call.
            controller_index = toolkit.hb_list_controllers.func()
        except Exception as e:
            controller_index = f"Failed to fetch controller list from hummingbot-api: {e}"

        # Helper: parse available controller (type, name) pairs from index
        available_pairs: List[Tuple[str, str]] = []
        try:
            parsed_index = json.loads(controller_index)
            if isinstance(parsed_index, dict):
                for ctype, controllers in parsed_index.items():
                    if not isinstance(controllers, list):
                        continue
                    for entry in controllers:
                        if isinstance(entry, str):
                            available_pairs.append((ctype, entry))
                        elif isinstance(entry, dict) and "name" in entry:
                            name_val = entry.get("name")
                            if isinstance(name_val, str):
                                available_pairs.append((ctype, name_val))
        except Exception:
            parsed_index = None

        # Step 1: multi-turn selection of controller_type and controller_name.
        # We keep a small conversational loop with explicit feedback when
        # the chosen controller is not present in the controller index.
        selection_messages = [
            SystemMessage(
                content=(
                    "You are selecting the most suitable hummingbot controller "
                    "(type and name) for a trading strategy based on factors "
                    "and an investment plan.\n\n"
                    "You MUST:\n"
                    "- Choose controller_type and controller_name ONLY from the available list below.\n"
                    "- NEVER invent new controller names.\n"
                    "- Reply EXACTLY in this format:\n"
                    "  CONTROLLER_TYPE: <type>\n"
                    "  CONTROLLER_NAME: <name>"
                )
            ),
            HumanMessage(
                content=(
                    f"You are designing a strategy for {state['company_of_interest']} "
                    f"on {state['trade_date']}.\n\n"
                    f"Investment plan:\n{state.get('investment_plan', '')}\n\n"
                    f"Quantitative factors specification:\n{state.get('factors_spec', '')}\n\n"
                    f"Quantitative factor values and signals:\n{state.get('factor_values', '')}\n\n"
                    f"Backtest results (may be empty on first run):\n{state.get('backtest_results', '')}\n\n"
                    "Available controllers from hummingbot-api:\n"
                    f"{controller_index}\n\n"
                    "Now choose ONE controller_type and ONE controller_name from the list above.\n"
                    "Remember: do not invent new names."
                )
            ),
        ]

        controller_type = ""
        controller_name = ""

        max_selection_attempts = 3
        for attempt in range(max_selection_attempts):
            select_result = llm.invoke(selection_messages)
            selection_messages.append(
                AIMessage(
                    content=select_result.content
                    if hasattr(select_result, "content")
                    else str(select_result)
                )
            )

            # Parse controller_type / controller_name from the latest reply
            select_content = (
                select_result.content
                if hasattr(select_result, "content")
                else str(select_result)
            )
            c_type = ""
            c_name = ""
            for line in select_content.splitlines():
                line_stripped = line.strip()
                if line_stripped.upper().startswith("CONTROLLER_TYPE:"):
                    c_type = line_stripped.split(":", 1)[1].strip()
                if line_stripped.upper().startswith("CONTROLLER_NAME:"):
                    c_name = line_stripped.split(":", 1)[1].strip()

            # If we could not parse the index, accept the first parsed values.
            if not available_pairs:
                controller_type, controller_name = c_type, c_name
                break

            if (c_type, c_name) in available_pairs:
                controller_type, controller_name = c_type, c_name
                break

            # Invalid selection: append feedback and ask again.
            feedback = (
                "Your previous selection was INVALID.\n"
                f"controller_type='{c_type}' and controller_name='{c_name}' "
                "do NOT exist in the available controller list.\n\n"
                "Please carefully re-read the available controllers shown above "
                "and answer again using ONLY a valid pair.\n"
                "Remember to respond exactly in the required format."
            )
            selection_messages.append(HumanMessage(content=feedback))

        # As a final safeguard, if still invalid and we know valid pairs,
        # fall back to the first available combination.
        if available_pairs and (controller_type, controller_name) not in available_pairs:
            controller_type, controller_name = available_pairs[0]

        # Step 2: fetch the specific controller template using the selected type/name
        if controller_type and controller_name:
            try:
                controller_template = toolkit.hb_get_controller_template.func(
                    controller_type, controller_name
                )
            except Exception as e:
                controller_template = (
                    f"Failed to fetch controller template for "
                    f"type={controller_type}, name={controller_name}: {e}"
                )
        else:
            controller_template = (
                "Controller type/name could not be parsed. Use defaults inferred "
                "from the available configs above."
            )

        # Step 3: design final config using the template
        config_chain = config_prompt | llm
        result = config_chain.invoke(
            {
                "company_of_interest": state["company_of_interest"],
                "trade_date": state["trade_date"],
                "investment_plan": state.get("investment_plan", ""),
                "factors_spec": state.get("factors_spec", ""),
                "factor_values": state.get("factor_values", ""),
                "backtest_results": state.get("backtest_results", ""),
                "controller_type": controller_type or "UNKNOWN",
                "controller_name": controller_name or "UNKNOWN",
                "controller_template": controller_template,
            }
        )
        content = result.content if hasattr(result, "content") else str(result)
        config_json = ""
        marker = "CONFIG_JSON:"
        if marker in content:
            _, tail = content.split(marker, 1)
            config_json = tail.strip()

        return {
            "strategy_template": content,
            "strategy_config": config_json or content,
            "sender": "Strategy Designer",
            "messages": state["messages"] + [result],
        }

    return node
