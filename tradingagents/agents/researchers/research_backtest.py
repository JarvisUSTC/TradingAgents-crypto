from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Any

import pandas as pd


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _compute_drawdown(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return dd


def _simple_momentum_strategy(prices: pd.Series, lookback: int, long_only: bool = True) -> pd.Series:
    """Return target position series in {0,1} (or {-1,0,1} if not long_only)."""
    mom = prices.pct_change(lookback)
    if long_only:
        pos = (mom > 0).astype(float)
    else:
        pos = pd.Series(0.0, index=prices.index)
        pos[mom > 0] = 1.0
        pos[mom < 0] = -1.0
    return pos.fillna(0.0)


def _backtest_from_close(close: pd.Series, position: pd.Series, fee_bps: float = 0.0) -> Dict[str, Any]:
    """Vectorized backtest with simple fee model based on turnover."""
    close = close.astype(float)
    ret = close.pct_change().fillna(0.0)

    # Use prior day's position (trade at close, hold next period)
    pos = position.shift(1).fillna(0.0)

    # Turnover and fees
    turnover = (pos - pos.shift(1).fillna(0.0)).abs()
    fee = turnover * (fee_bps / 10000.0)

    strat_ret = pos * ret - fee
    equity = (1.0 + strat_ret).cumprod()

    dd = _compute_drawdown(equity)
    max_dd = float(dd.min()) if len(dd) else 0.0

    # Simple Sharpe (daily) assuming input is daily-ish; caller controls interval.
    vol = float(strat_ret.std()) if len(strat_ret) > 1 else 0.0
    mean = float(strat_ret.mean()) if len(strat_ret) else 0.0
    sharpe = (mean / vol) * (252 ** 0.5) if vol and vol > 0 else 0.0

    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0

    return {
        "equity": equity,
        "strategy_returns": strat_ret,
        "position": pos,
        "total_return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "turnover_mean": float(turnover.mean()) if len(turnover) else 0.0,
    }


def create_research_backtest(toolkit):
    """Create a research-only backtest node.

    设计目标：
    - 不依赖 LLM（确定性、可复现）
    - 只使用现有工具层（优先 Hummingbot candles；失败则走原数据源）
    - 输出 backtest_report/backtest_metrics，供 bull/bear/manager 引用
    """

    def node(state) -> dict:
        cfg = getattr(toolkit, "config", {}) or {}
        bt_cfg = cfg.get("backtest", {}) or {}
        if bt_cfg.get("enabled", True) is False:
            return {"backtest_report": "", "backtest_metrics": {}}

        ticker = state["company_of_interest"]
        curr_date = state["trade_date"]

        factor_cfg = cfg.get("factor", {}) or {}

        lookback_days = int(bt_cfg.get("look_back_days", factor_cfg.get("look_back_days", 180)))
        strategy = str(bt_cfg.get("strategy", "momentum"))
        mom_lookback = int(bt_cfg.get("momentum_lookback", 20))
        fee_bps = float(bt_cfg.get("fee_bps", 0.0))
        long_only = bool(bt_cfg.get("long_only", True))

        # For now, we backtest using daily-ish close extracted from factor report text.
        # Prefer a structured candles path if available in your environment; this keeps the node lightweight.
        # We rely on get_crypto_price_history (which can be backed by Hummingbot candles when enabled).
        # The function returns a text report; we parse close prices if present.
        start_dt = datetime.strptime(curr_date, "%Y-%m-%d") - timedelta(days=lookback_days)
        start_date = start_dt.strftime("%Y-%m-%d")

        # Use Toolkit tools directly to avoid coupling to interface implementation details.
        try:
            price_text = toolkit.get_crypto_price_history(symbol=ticker, curr_date=curr_date, look_back_days=lookback_days)
        except Exception as e:
            return {
                "backtest_report": f"## Research Backtest\n\n无法获取价格历史：{e}",
                "backtest_metrics": {},
            }

        # Parse close prices from the text (expects lines like 'Price: $12,345.67')
        dates = []
        closes = []
        current_date_line = None
        for line in str(price_text).splitlines():
            line = line.strip()
            if line.startswith("Date:"):
                current_date_line = line.replace("Date:", "").strip()
            elif line.startswith("Price:"):
                raw = line.replace("Price:", "").strip()
                raw = raw.replace("$", "").replace(",", "")
                px = _safe_float(raw)
                if current_date_line and px is not None:
                    dates.append(current_date_line)
                    closes.append(px)
                    current_date_line = None

        if len(closes) < max(30, mom_lookback + 5):
            return {
                "backtest_report": (
                    "## Research Backtest\n\n"
                    "价格样本不足，无法进行回测。"
                    f"\n- symbol: {ticker}\n- start: {start_date}\n- end: {curr_date}\n"
                ),
                "backtest_metrics": {},
            }

        df = pd.DataFrame({"date": dates, "close": closes})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "close"]).sort_values("date")
        close = df.set_index("date")["close"]

        if strategy == "momentum":
            position = _simple_momentum_strategy(close, lookback=mom_lookback, long_only=long_only)
        else:
            # default fallback
            position = _simple_momentum_strategy(close, lookback=mom_lookback, long_only=long_only)

        res = _backtest_from_close(close, position, fee_bps=fee_bps)

        metrics = {
            "strategy": strategy,
            "look_back_days": lookback_days,
            "momentum_lookback": mom_lookback,
            "fee_bps": fee_bps,
            "long_only": long_only,
            "total_return": res["total_return"],
            "max_drawdown": res["max_drawdown"],
            "sharpe": res["sharpe"],
            "turnover_mean": res["turnover_mean"],
            "n_bars": int(len(close)),
        }

        report = "## Research Backtest (deterministic)\n\n"
        report += f"- 标的: {ticker}\n"
        report += f"- 时间区间: {close.index.min().date()} ~ {close.index.max().date()}\n"
        report += f"- 策略: {strategy} (momentum_lookback={mom_lookback}, long_only={long_only})\n"
        report += f"- 费用: {fee_bps} bps（按换手近似）\n\n"
        report += "**结果摘要：**\n"
        report += f"- 总收益: {metrics['total_return'] * 100:.2f}%\n"
        report += f"- 最大回撤: {metrics['max_drawdown'] * 100:.2f}%\n"
        report += f"- Sharpe(年化，假设日频): {metrics['sharpe']:.2f}\n"
        report += f"- 平均换手(绝对仓位变化): {metrics['turnover_mean']:.4f}\n\n"
        report += "**注意：**该回测节点用于科研对比与证据补充，不输出买卖指令，也不等价于可交易策略。\n"

        return {
            "backtest_report": report,
            "backtest_metrics": metrics,
        }

    return node
