from tradingagents.graph.trading_graph import TradingAgentsGraph

g = TradingAgentsGraph(debug=False)

# 1) LangGraph 自带 Mermaid 文本
print(g.export_langgraph_mermaid())

# 2) LangGraph 自带 PNG
g.export_langgraph_png("./docs/langgraph_flow.png")