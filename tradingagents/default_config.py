import os

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", "./results"),
    "data_dir": os.getenv("TRADINGAGENTS_DATA_DIR", "./data"),
    "data_cache_dir": os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
        "dataflows/data_cache",
    ),
    # LLM settings
    "llm_provider": "openai",
    "deep_think_llm": "o4-mini",
    "quick_think_llm": "gpt-4o-mini",
    "backend_url": "https://cloud-router-dev.lobehub.com/v1/messages",
    "api_key": "sk-MTOQIyECB0mEPpzletZvrw",
    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,
    # Tool settings
    "online_tools": True,

    # ===== Hummingbot market data (research only, no live trading) =====
    # When enabled, crypto price/technical tools will use Hummingbot candles feeds
    # (public endpoints) instead of CoinGecko for OHLCV/time-series.
    "hummingbot_market_data": {
        # Default to disabled because the vendored Hummingbot requires Python>=3.10.
        "enabled": os.getenv("TRADINGAGENTS_HB_MD_ENABLED", "0").lower() in ("1", "true", "yes", "y"),
        # Spot connector name used by Hummingbot candles feeds, e.g. "binance", "okx", "bybit".
        "connector": os.getenv("TRADINGAGENTS_HB_MD_CONNECTOR", "binance"),
        # Quote asset used to map symbol -> trading_pair, e.g. BTC -> BTC-USDT.
        "quote_asset": os.getenv("TRADINGAGENTS_HB_MD_QUOTE", "USDT"),
        # Candles interval, must be supported by Hummingbot candles feeds (e.g. "1m", "5m", "1h", "1d").
        "interval": os.getenv("TRADINGAGENTS_HB_MD_INTERVAL", "1d"),
        # For historical fetch/cache in Hummingbot MarketDataProvider.
        "max_cache_records": int(os.getenv("TRADINGAGENTS_HB_MD_MAX_CACHE", "10000")),
        # Prefer historical REST fetch instead of realtime WS feed (better for research/backtest).
        "prefer_historical": os.getenv("TRADINGAGENTS_HB_MD_PREFER_HIST", "1").lower() in ("1", "true", "yes", "y"),

        # Optional: persist fetched candles to disk (CSV) for research reproducibility.
        # Set to empty/None to disable.
        "cache_dir": os.getenv("TRADINGAGENTS_HB_MD_CACHE_DIR", ""),
    },

    # ===== Research factor defaults =====
    "factor": {
        "look_back_days": int(os.getenv("TRADINGAGENTS_FACTOR_LOOKBACK_DAYS", "180")),
    },

    # ===== Research backtest defaults =====
    # Research-only, deterministic backtest node (no live trading).
    "backtest": {
        "enabled": os.getenv("TRADINGAGENTS_BACKTEST_ENABLED", "1").lower() in ("1", "true", "yes", "y"),
        "look_back_days": int(os.getenv("TRADINGAGENTS_BACKTEST_LOOKBACK_DAYS", "365")),
        "strategy": os.getenv("TRADINGAGENTS_BACKTEST_STRATEGY", "momentum"),
        "momentum_lookback": int(os.getenv("TRADINGAGENTS_BACKTEST_MOM_LB", "20")),
        "fee_bps": float(os.getenv("TRADINGAGENTS_BACKTEST_FEE_BPS", "0")),
        "long_only": os.getenv("TRADINGAGENTS_BACKTEST_LONG_ONLY", "1").lower() in ("1", "true", "yes", "y"),
    },
}
