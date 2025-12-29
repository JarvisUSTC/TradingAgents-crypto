from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate

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
- Produce a final JSON configuration string that hummingbot-api can accept for backtesting.

Final output structure (for the whole node):
1) Brief natural language explanation of the chosen controller and parameters.
2) A JSON configuration object on a separate line, starting with 'CONFIG_JSON:'.
The JSON must be valid, compact, and self-contained."""

    # Step 1: choose controller_type and controller_name
    select_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are selecting the most suitable hummingbot controller (type and name) "
                "for a trading strategy based on factors and an investment plan. "
                "Only choose from the provided available controllers.",
            ),
            (
                "human",
                """You are designing a strategy for {company_of_interest} on {trade_date}.

Investment plan (human research team decision):
{investment_plan}

Quantitative factors specification:
{factors_spec}

Quantitative factor values and signals:
{factor_values}

Available controllers from hummingbot-api:
{controller_configs}

From these controllers, select ONE controller_type and ONE controller_name that best match the plan and factors.

Output EXACTLY in the following format:
CONTROLLER_TYPE: <type>
CONTROLLER_NAME: <name>""",
            ),
        ]
    )

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

        # Step 1: ask LLM to choose controller_type and controller_name
        select_chain = select_prompt | llm
        select_result = select_chain.invoke(
            {
                "company_of_interest": state["company_of_interest"],
                "trade_date": state["trade_date"],
                "investment_plan": state.get("investment_plan", ""),
                "factors_spec": state.get("factors_spec", ""),
                "factor_values": state.get("factor_values", ""),
                "controller_configs": controller_index,
            }
        )
        select_content = (
            select_result.content
            if hasattr(select_result, "content")
            else str(select_result)
        )

        controller_type = ""
        controller_name = ""
        for line in select_content.splitlines():
            line_stripped = line.strip()
            if line_stripped.upper().startswith("CONTROLLER_TYPE:"):
                controller_type = line_stripped.split(":", 1)[1].strip()
            if line_stripped.upper().startswith("CONTROLLER_NAME:"):
                controller_name = line_stripped.split(":", 1)[1].strip()

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
