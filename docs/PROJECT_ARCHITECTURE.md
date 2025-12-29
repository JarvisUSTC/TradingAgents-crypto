# TradingAgents-crypto 项目架构文档

> 本文档基于仓库当前实现整理，目标是给出“入口 → LangGraph 流程 → Agents → Tools/Dataflows → 扩展点”的清晰全景。

## 1. 项目概览

TradingAgents-crypto 是一个基于多智能体（Multi-Agent）协作的研究/分析框架。核心是用 LangGraph 把多个角色（行情分析、新闻/情绪、基本面、多空研究辩论、交易计划、风险辩论）组织为一个有状态的工作流，最终输出长文本决策与可选的 `BUY/SELL/HOLD` 信号。

关键特点：

- 多入口：脚本、CLI、Web。
- 核心编排：`TradingAgentsGraph` + `GraphSetup`（LangGraph StateGraph）。
- 工具与数据：`Toolkit` 封装工具，`dataflows/interface.py` 统一数据接口。
- 研究扩展：新增确定性 `Research Backtest` 节点与因子报告工具（不依赖 LLM）。

## 2. 入口层与调用链

### 2.1 最小脚本入口

- `main.py:14`
  - 直接创建 `TradingAgentsGraph(debug=True)` 并调用 `propagate("BTC", "2024-05-10")`。

### 2.2 Web 入口（Flask + SocketIO）

- `run_web.py:14`
  - 启动 SocketIO 服务器。
- `web_app.py:178`
  - 后台线程中创建 `TradingAgentsGraph(...)`，通过 `graph.graph.stream(...)` 持续输出中间 state。

### 2.3 CLI 入口（Typer + Rich）

- `cli/main.py:734`
  - 交互式选择参数后，同样通过 `graph.graph.stream(...)` 流式跑完整图。

## 3. 核心编排层（LangGraph）

### 3.1 TradingAgentsGraph

- `tradingagents/graph/trading_graph.py:32`
  - 初始化 LLM（OpenAI/Anthropic/Google）。
  - 初始化 `Toolkit`（工具集合）与 Memory。
  - 创建 ToolNodes：`_create_tool_nodes()`（按 analyst 类型分组）。
  - 通过 `GraphSetup.setup_graph()` 编译 LangGraph。

### 3.2 GraphSetup（节点与边）

- `tradingagents/graph/setup.py:43`
  - 将 analyst 节点按 `selected_analysts` 串联。
  - 通过 `ConditionalLogic.should_continue_*` 决定“继续调用工具”还是“清理消息并进入下一 analyst”。
  - 在 analyst 阶段结束后（最后一个 analyst 的 Msg Clear 后），可选插入 `Research Backtest` 节点。

### 3.3 ConditionalLogic（条件跳转）

- `tradingagents/graph/conditional_logic.py:14`
  - `should_continue_market/social/news/fundamentals`：根据 tool_calls 决定路由。
  - `should_continue_debate`：控制 Bull/Bear 辩论轮数。
  - `should_continue_risk_analysis`：控制风险辩论轮数。

### 3.4 State 初始化

- `tradingagents/graph/propagation.py:18`
  - `create_initial_state()` 初始化 `AgentState` 中的字段，包括新增的 `backtest_report/backtest_metrics`。

## 4. 状态模型（AgentState）

- `tradingagents/agents/utils/agent_states.py:50`

核心字段：

- 基础：`company_of_interest`, `trade_date`, `messages`
- Analyst 输出：`market_report`, `sentiment_report`, `news_report`, `fundamentals_report`
- 研究辩论：`investment_debate_state`, `investment_plan`
- Trader：`trader_investment_plan`
- 风控辩论：`risk_debate_state`, `final_trade_decision`
- 新增研究字段：`backtest_report`, `backtest_metrics`

## 5. 智能体层（Agents）

入口聚合：`tradingagents/agents/__init__.py:1`

### 5.1 Analysts

- 行情分析：`tradingagents/agents/analysts/market_analyst.py:52`
- 新闻：`tradingagents/agents/analysts/news_analyst.py:52`
- 基本面：`tradingagents/agents/analysts/fundamentals_analyst.py:52`
- 社交情绪：`tradingagents/agents/analysts/social_media_analyst.py:6`

### 5.2 Researchers（多空辩论）

- Bull：`tradingagents/agents/researchers/bull_researcher.py:6`
- Bear：`tradingagents/agents/researchers/bear_researcher.py:6`

### 5.3 Research Backtest（确定性研究回测节点）

- `tradingagents/agents/researchers/research_backtest.py:70`
  - 不依赖 LLM
  - 通过历史价格生成一个简化动量策略的回测结果，输出：
    - `backtest_report`（面向人读的摘要）
    - `backtest_metrics`（结构化指标，如 total_return/max_drawdown/sharpe）

### 5.4 Managers / Trader / Risk

- 研究经理：`tradingagents/agents/managers/research_manager.py:5`
- Trader：`tradingagents/agents/trader/trader.py`
- 风控：`tradingagents/agents/managers/risk_manager.py` + `tradingagents/agents/risk_mgmt/*`

## 6. 工具层与数据层（Toolkit → dataflows）

### 6.1 Toolkit（LangChain Tools）

- `tradingagents/agents/utils/agent_utils.py:34`
  - `Toolkit` 把数据函数封装为 tool，供 ToolNode 执行。

### 6.2 dataflows 接口

- `tradingagents/dataflows/interface.py:1`
  - 对外提供 crypto/stock 的统一数据接口。
  - 包含新增 `get_crypto_factor_report(...)`（因子摘要）。

### 6.3 配置注入

- `tradingagents/dataflows/config.py:17`
  - `set_config/get_config` 让入口将 config 写入 dataflows（工具行为由 config 控制）。
- `tradingagents/default_config.py:3`
  - 默认配置，包含 backtest/factor/hummingbot_market_data 等段落。

## 7. LangGraph 自带流程图导出（官方接口）

本项目编译后的 LangGraph 在 `TradingAgentsGraph.graph` 中。LangGraph 自带可视化能力通常通过：

- `compiled_graph.get_graph().draw_mermaid()`：导出 Mermaid 文本
- `compiled_graph.get_graph().draw_mermaid_png()`：导出 PNG bytes

为方便直接使用，已在 `tradingagents/graph/trading_graph.py:226` 增加两个便捷方法：

- `TradingAgentsGraph.export_langgraph_mermaid()`
- `TradingAgentsGraph.export_langgraph_png(output_path)`

示例（在你的开发机正确环境上运行）：

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph

g = TradingAgentsGraph(debug=False)
print(g.export_langgraph_mermaid())

g.export_langgraph_png("./docs/langgraph_flow.png")
```
