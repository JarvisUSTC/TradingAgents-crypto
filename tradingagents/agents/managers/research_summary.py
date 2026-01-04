from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from tradingagents.agents.utils.agent_states import AgentState


def create_research_summary(llm):
    """
    Create a node that compresses the four long-form research reports
    (market, news, fundamentals, sentiment) plus the investment plan
    into a single concise summary for downstream quantitative agents.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior research summarizer for a quantitative trading team. "
                "Your task is to compress multiple long-form research reports into a short, "
                "actionable summary that preserves only the most important signals for "
                "quantitative modelling and risk management.",
            ),
            (
                "human",
                """
Company: {company}
Trade Date: {trade_date}

High-level investment plan:
{investment_plan}

Market research report:
{market_report}

News report:
{news_report}

Fundamentals report:
{fundamentals_report}

Sentiment report:
{sentiment_report}

Please produce a concise summary (no more than ~300 words) that captures:
- The core directional view (bullish/bearish/neutral) and main drivers.
- Any major risks or scenario dependencies that are relevant for factor design and risk management.
- Any important time horizon considerations (short-term vs long-term).

Focus on information that will directly influence quantitative factor design and risk limits.
Return only the summary text (no bullet points or headings needed).
""",
            ),
        ]
    )

    def node(state: AgentState) -> Dict[str, Any]:
        chain = prompt | llm
        result = chain.invoke(
            {
                "company": state["company_of_interest"],
                "trade_date": state["trade_date"],
                "investment_plan": state.get("investment_plan", ""),
                "market_report": state.get("market_report", ""),
                "news_report": state.get("news_report", ""),
                "fundamentals_report": state.get("fundamentals_report", ""),
                "sentiment_report": state.get("sentiment_report", ""),
            }
        )
        content = result.content if hasattr(result, "content") else str(result)

        return {
            "research_summary": content.strip(),
            "sender": "Research Summary",
            "messages": state["messages"] + [result],
        }

    return node

