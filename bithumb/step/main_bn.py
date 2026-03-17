import argparse
import base64
import builtins
import hashlib
import hmac
import json
import os
import sqlite3
import ssl
import subprocess
import sys
import time
import uuid
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from email.message import EmailMessage
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

KST = timezone(timedelta(hours=9))


def now_kst() -> datetime:
    return datetime.now(tz=KST)


def log(message: str) -> None:
    builtins.print(f"[{now_kst().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def asbool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


KNOWN_QUOTE_ASSETS = [
    "USDT", "USDC", "FDUSD", "BUSD", "TUSD", "USDP", "DAI", "BTC", "ETH", "BNB", "TRY", "EUR",
]


def normalize_market(market: str, default_quote: str = "USDT") -> str:
    raw = (market or "").strip().upper().replace("/", "-").replace("_", "-")
    if not raw:
        return raw
    if "-" in raw:
        left, right = [p.strip().upper() for p in raw.split("-", 1)]
        if left in KNOWN_QUOTE_ASSETS and right:
            return f"{left}-{right}"
        if right in KNOWN_QUOTE_ASSETS and left:
            return f"{right}-{left}"
        return f"{left}-{right}"
    quotes = sorted(set([default_quote.upper(), *KNOWN_QUOTE_ASSETS]), key=len, reverse=True)
    for quote in quotes:
        if raw.endswith(quote) and len(raw) > len(quote):
            return f"{quote}-{raw[:-len(quote)]}"
    return f"{default_quote.upper()}-{raw}"


def parse_markets(markets: str) -> List[str]:
    default_quote = os.environ.get("MARKET_QUOTE_CURRENCY", "USDT").upper()
    return [normalize_market(m, default_quote) for m in markets.split(",") if m.strip()]


def parse_symbols(symbols: str) -> List[str]:
    return [s.strip().upper() for s in symbols.split(",") if s.strip()]


def parse_csv_list(values: str) -> List[str]:
    return [v.strip() for v in values.replace(";", ",").split(",") if v.strip()]


def normalize_deploy_strategy(value: Optional[str]) -> str:
    strategy = (value or "auto").strip().lower().replace("-", "_")
    allowed = {"auto", "safetensors_adapter", "gguf_adapter"}
    return strategy if strategy in allowed else "auto"


def make_compact_reason(action: str, technical: dict, sentiment: dict, onchain: dict, rl_action: str, position_qty: Decimal, source: str) -> str:
    tags: List[str] = [source.upper(), f"A={str(action or 'HOLD').upper()}"]
    tags.append(f"T={'UP' if technical.get('trend_up') else 'DOWN'}")
    if technical.get('rebound_signal'):
        tags.append("REB")
    fear = int((sentiment.get('fear_greed') or {}).get('value', 50) or 50)
    if fear <= 25:
        tags.append("FEAR")
    elif fear >= 75:
        tags.append("GREED")
    score = float(sentiment.get('sentiment_score', 0.0) or 0.0)
    if score <= -0.3:
        tags.append("SNEG")
    elif score >= 0.3:
        tags.append("SPOS")
    pressure = float(onchain.get('combined_score', 0.0) or 0.0)
    if pressure >= 0.25:
        tags.append("CHAINHOT")
    elif pressure <= -0.25:
        tags.append("CHAINCOLD")
    if position_qty > 0:
        tags.append("POS")
    if rl_action:
        tags.append(f"RL={str(rl_action).upper()}")
    return "|".join(tags)


def split_market(market: str) -> Tuple[str, str]:
    normalized = normalize_market(market)
    quote, base = normalized.split("-", 1)
    return quote, base


def market_to_symbol(market: str) -> str:
    quote, base = split_market(market)
    return f"{base}{quote}"


def symbol_to_market(symbol: str, default_quote: str = "USDT") -> str:
    return normalize_market(symbol, default_quote)


def floor_to_step(value: Decimal, step: Decimal) -> Decimal:
    value = to_decimal(value)
    step = to_decimal(step)
    if step <= 0:
        return value
    return (value / step).to_integral_value(rounding=ROUND_DOWN) * step


def merge_symbols(*groups: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for group in groups:
        for value in group:
            v = (value or "").strip().upper()
            if not v or v in seen:
                continue
            seen.add(v)
            out.append(v)
    return out


def filter_markets(markets: List[str], excluded_symbols: List[str]) -> List[str]:
    excluded = {s.upper() for s in excluded_symbols}
    out: List[str] = []
    for market in markets:
        try:
            _, base = split_market(market)
        except ValueError:
            continue
        if base.upper() in excluded:
            continue
        out.append(market)
    return out


def clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        value = float(value)
    except Exception:
        return default
    return max(low, min(high, value))


def clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        value = int(value)
    except Exception:
        return default
    return max(low, min(high, value))


def to_decimal(value: Any) -> Decimal:
    return Decimal(str(value or 0))


def quantize_down(value: Decimal, digits: int = 8) -> Decimal:
    q = Decimal("1") / (Decimal(10) ** digits)
    return value.quantize(q, rounding=ROUND_DOWN)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [_to_jsonable(r) for r in value.to_dict(orient="records")]
    return value


def safe_json_dumps(value: Any) -> str:
    return json.dumps(_to_jsonable(value), ensure_ascii=False)


def collect_ollama_models() -> List[str]:
    models: List[str] = []
    primary = os.environ.get("OLLAMA_MODEL", "qwen3:4b").strip()
    if primary:
        models.append(primary)

    model_list = os.environ.get("OLLAMA_MODEL_LIST", "").strip() or os.environ.get("OLLAMA_MODEL_NAMES", "").strip()
    if model_list:
        for model in [m.strip() for m in model_list.replace(";", ",").split(",") if m.strip()]:
            if model not in models:
                models.append(model)

    for key in sorted(os.environ):
        if key.startswith("OLLAMA_MODEL_") and key[13:].isdigit():
            model = os.environ.get(key, "").strip()
            if model and model not in models:
                models.append(model)
    return models


@dataclass
class Config:
    coinone_access_token: str
    coinone_secret_key: str
    gmail_user: str
    gmail_password: str
    target_email: str
    db_path: str
    ollama_url: str
    ollama_models: List[str]
    ollama_ensemble_enabled: bool
    etherscan_api_key: Optional[str]
    dry_run: bool
    target_markets: List[str]
    excluded_symbols: List[str]
    stablecoin_symbols: List[str]
    allow_stablecoin_trading: bool
    ai_override_enabled: bool
    loop_interval_seconds: int
    report_interval_seconds: int
    max_exposure_per_market: Decimal
    max_total_exposure: Decimal
    stop_loss_pct: Decimal
    take_profit_pct: Decimal
    take_profit_close_fraction: Decimal
    highest_take_profit_pct: Decimal
    trailing_take_profit_gap_pct: Decimal
    emergency_exit_price_offset_pct: Decimal
    rescue_addon_trigger_pct: Decimal
    rescue_min_notional_buffer_pct: Decimal
    grid_levels: int
    grid_step_pct: Decimal
    min_trade_krw: Decimal
    exchange_min_trade_krw: Decimal
    buy_spend_safety_buffer: Decimal
    market_quote_currency: str
    auto_discover_markets: bool
    market_universe_size: int
    market_refresh_seconds: int
    coinone_post_only: bool
    buy_book_level: int
    sell_book_level: int
    gmail_smtp_host: str
    gmail_smtp_port: int
    request_timeout: int
    buy_confidence_threshold: float
    strong_hold_buy_enabled: bool
    strong_hold_buy_confidence: float
    min_new_position_krw: Decimal
    suppress_unheld_sell_logs: bool
    order_timeout_seconds: int
    order_max_attempts: int
    order_poll_interval_seconds: int
    pending_sync_interval_seconds: int
    markets_per_cycle: int
    balance_refresh_every_markets: int
    macro_refresh_every_markets: int
    completed_order_lookup_limit: int
    decision_reward_horizon_minutes: int
    binance_futures_base_url: str
    binance_recv_window: int
    binance_leverage: int
    binance_margin_type: str
    binance_hedge_mode: bool
    futures_short_enabled: bool
    env_file: str
    sft_output_jsonl: str
    sft_max_rows: int
    sft_good_reward_threshold: float
    sft_bad_reward_threshold: float
    sft_hold_band: float
    sft_base_model: str
    sft_train_output_dir: str
    sft_num_train_epochs: float
    sft_learning_rate: float
    sft_per_device_batch_size: int
    sft_gradient_accumulation_steps: int
    sft_max_seq_length: int
    sft_lora_r: int
    sft_lora_alpha: int
    sft_lora_dropout: float
    sft_target_modules: List[str]
    sft_use_bf16: bool
    ollama_deploy_strategy: str
    llama_cpp_dir: str
    llama_cpp_convert_lora_to_gguf: str
    ollama_gguf_adapter_path: str
    ollama_deploy_base_model: str
    ollama_deploy_model_name: str
    ollama_modelfile_path: str
    ollama_update_env_after_deploy: bool
    compare_input_jsonl: str
    compare_output_jsonl: str
    compare_max_cases: int

    @classmethod
    def from_env(cls) -> "Config":
        load_dotenv()
        excluded_symbols = parse_symbols(os.environ.get("EXCLUDED_SYMBOLS", "BTC,ETH"))
        stablecoin_symbols = parse_symbols(
            os.environ.get(
                "STABLECOIN_SYMBOLS",
                "USDT,USDC,DAI,FDUSD,TUSD,USDP,USDE,PYUSD,USDS,USDD,USD1,STABLE",
            )
        )
        allow_stablecoin_trading = asbool(os.environ.get("ALLOW_STABLECOIN_TRADING"), False)
        blocked = merge_symbols(excluded_symbols, [] if allow_stablecoin_trading else stablecoin_symbols)
        quote = os.environ.get("MARKET_QUOTE_CURRENCY", "USDT").upper()
        target_markets = filter_markets(
            parse_markets(os.environ.get("TARGET_MARKETS", f"{quote}-BTC,{quote}-ETH,{quote}-SOL")),
            blocked,
        )
        return cls(
            coinone_access_token=os.environ.get("BINANCE_API_KEY") or os.environ.get("COINONE_ACCESS_TOKEN", ""),
            coinone_secret_key=os.environ.get("BINANCE_SECRET_KEY") or os.environ.get("COINONE_SECRET_KEY", ""),
            gmail_user=os.environ.get("GMAIL_USER", ""),
            gmail_password=os.environ.get("GMAIL_PASSWORD", ""),
            target_email=os.environ.get("TARGET_EMAIL", ""),
            db_path=os.environ.get("DB_PATH", "./ai_binance_bot.db"),
            ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            ollama_models=collect_ollama_models(),
            ollama_ensemble_enabled=asbool(os.environ.get("OLLAMA_ENSEMBLE_ENABLED"), True),
            etherscan_api_key=os.environ.get("ETHERSCAN_API_KEY") or None,
            dry_run=asbool(os.environ.get("DRY_RUN"), True),
            target_markets=target_markets,
            excluded_symbols=excluded_symbols,
            stablecoin_symbols=stablecoin_symbols,
            allow_stablecoin_trading=allow_stablecoin_trading,
            ai_override_enabled=asbool(os.environ.get("AI_OVERRIDE_ENABLED"), True),
            loop_interval_seconds=int(os.environ.get("LOOP_INTERVAL_SECONDS", "300")),
            report_interval_seconds=int(os.environ.get("REPORT_INTERVAL_SECONDS", str(3 * 60 * 60))),
            max_exposure_per_market=to_decimal(os.environ.get("MAX_EXPOSURE_PER_MARKET", "0.12")),
            max_total_exposure=to_decimal(os.environ.get("MAX_TOTAL_EXPOSURE", "0.35")),
            stop_loss_pct=to_decimal(os.environ.get("STOP_LOSS_PCT", "0.025")),
            take_profit_pct=to_decimal(os.environ.get("TAKE_PROFIT_PCT", "0.022")),
            take_profit_close_fraction=to_decimal(os.environ.get("TAKE_PROFIT_CLOSE_FRACTION", "0.55")),
            highest_take_profit_pct=to_decimal(os.environ.get("HIGHEST_TAKE_PROFIT_PCT", "0.035")),
            trailing_take_profit_gap_pct=to_decimal(os.environ.get("TRAILING_TAKE_PROFIT_GAP_PCT", "0.008")),
            emergency_exit_price_offset_pct=to_decimal(os.environ.get("EMERGENCY_EXIT_PRICE_OFFSET_PCT", "0.0007")),
            rescue_addon_trigger_pct=to_decimal(os.environ.get("RESCUE_ADDON_TRIGGER_PCT", "0.045")),
            rescue_min_notional_buffer_pct=to_decimal(os.environ.get("RESCUE_MIN_NOTIONAL_BUFFER_PCT", "0.15")),
            grid_levels=int(os.environ.get("GRID_LEVELS", "4")),
            grid_step_pct=to_decimal(os.environ.get("GRID_STEP_PCT", "0.012")),
            min_trade_krw=to_decimal(os.environ.get("MIN_TRADE_KRW", "25")),
            exchange_min_trade_krw=to_decimal(os.environ.get("EXCHANGE_MIN_TRADE_KRW", os.environ.get("BINANCE_MIN_NOTIONAL", "10"))),
            buy_spend_safety_buffer=to_decimal(os.environ.get("BUY_SPEND_SAFETY_BUFFER", "0.995")),
            market_quote_currency=quote,
            auto_discover_markets=asbool(os.environ.get("AUTO_DISCOVER_MARKETS"), True),
            market_universe_size=max(1, int(os.environ.get("MARKET_UNIVERSE_SIZE", "150"))),
            market_refresh_seconds=max(60, int(os.environ.get("MARKET_REFRESH_SECONDS", str(6 * 60 * 60)))),
            coinone_post_only=asbool(os.environ.get("BINANCE_POST_ONLY", os.environ.get("COINONE_POST_ONLY", "False")), False),
            buy_book_level=max(1, int(os.environ.get("BUY_BOOK_LEVEL", "1"))),
            sell_book_level=max(1, int(os.environ.get("SELL_BOOK_LEVEL", "1"))),
            gmail_smtp_host=os.environ.get("GMAIL_SMTP_HOST", "smtp.gmail.com"),
            gmail_smtp_port=int(os.environ.get("GMAIL_SMTP_PORT", "465")),
            request_timeout=int(os.environ.get("REQUEST_TIMEOUT", "15")),
            buy_confidence_threshold=clamp_float(os.environ.get("BUY_CONFIDENCE_THRESHOLD", 0.60), 0.0, 1.0, 0.60),
            strong_hold_buy_enabled=asbool(os.environ.get("STRONG_HOLD_BUY_ENABLED"), False),
            strong_hold_buy_confidence=clamp_float(os.environ.get("STRONG_HOLD_BUY_CONFIDENCE", 0.80), 0.0, 1.0, 0.80),
            min_new_position_krw=to_decimal(os.environ.get("MIN_NEW_POSITION_KRW", os.environ.get("MIN_TRADE_KRW", "25"))),
            suppress_unheld_sell_logs=asbool(os.environ.get("SUPPRESS_UNHELD_SELL_LOGS"), True),
            order_timeout_seconds=max(5, int(os.environ.get("ORDER_TIMEOUT_SECONDS", os.environ.get("BUY_ORDER_TIMEOUT_SECONDS", "45")))),
            order_max_attempts=max(1, int(os.environ.get("ORDER_MAX_ATTEMPTS", "4"))),
            order_poll_interval_seconds=max(1, int(os.environ.get("ORDER_POLL_INTERVAL_SECONDS", "2"))),
            pending_sync_interval_seconds=max(5, int(os.environ.get("PENDING_SYNC_INTERVAL_SECONDS", "15"))),
            markets_per_cycle=max(1, int(os.environ.get("MARKETS_PER_CYCLE", "12"))),
            balance_refresh_every_markets=max(1, int(os.environ.get("BALANCE_REFRESH_EVERY_MARKETS", "3"))),
            macro_refresh_every_markets=max(1, int(os.environ.get("MACRO_REFRESH_EVERY_MARKETS", "5"))),
            completed_order_lookup_limit=max(1, int(os.environ.get("COMPLETED_ORDER_LOOKUP_LIMIT", "50"))),
            decision_reward_horizon_minutes=max(30, int(os.environ.get("DECISION_REWARD_HORIZON_MINUTES", "120"))),
            binance_futures_base_url=os.environ.get("BINANCE_FUTURES_BASE_URL", "https://fapi.binance.com").rstrip("/"),
            binance_recv_window=max(1000, int(os.environ.get("BINANCE_RECV_WINDOW", "5000"))),
            binance_leverage=max(1, min(125, int(os.environ.get("BINANCE_LEVERAGE", "3")))),
            binance_margin_type=(os.environ.get("BINANCE_MARGIN_TYPE", "ISOLATED").strip().upper() or "ISOLATED"),
            binance_hedge_mode=asbool(os.environ.get("BINANCE_HEDGE_MODE"), False),
            futures_short_enabled=asbool(os.environ.get("FUTURES_SHORT_ENABLED", "true"), True),
            env_file=os.environ.get("ENV_FILE", ".env"),
            sft_output_jsonl=os.environ.get("SFT_OUTPUT_JSONL", "./artifacts/binance_sft_cases.jsonl"),
            sft_max_rows=max(10, int(os.environ.get("SFT_MAX_ROWS", "2000"))),
            sft_good_reward_threshold=clamp_float(os.environ.get("SFT_GOOD_REWARD_THRESHOLD", 0.03), -1.0, 1.0, 0.03),
            sft_bad_reward_threshold=clamp_float(os.environ.get("SFT_BAD_REWARD_THRESHOLD", -0.03), -1.0, 1.0, -0.03),
            sft_hold_band=clamp_float(os.environ.get("SFT_HOLD_BAND", 0.01), 0.0, 1.0, 0.01),
            sft_base_model=os.environ.get("SFT_BASE_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
            sft_train_output_dir=os.environ.get("SFT_TRAIN_OUTPUT_DIR", "./artifacts/lora_adapter"),
            sft_num_train_epochs=float(os.environ.get("SFT_NUM_TRAIN_EPOCHS", "3")),
            sft_learning_rate=float(os.environ.get("SFT_LEARNING_RATE", "2e-4")),
            sft_per_device_batch_size=max(1, int(os.environ.get("SFT_PER_DEVICE_BATCH_SIZE", "1"))),
            sft_gradient_accumulation_steps=max(1, int(os.environ.get("SFT_GRADIENT_ACCUMULATION_STEPS", "8"))),
            sft_max_seq_length=max(512, int(os.environ.get("SFT_MAX_SEQ_LENGTH", "2048"))),
            sft_lora_r=max(1, int(os.environ.get("SFT_LORA_R", "16"))),
            sft_lora_alpha=max(1, int(os.environ.get("SFT_LORA_ALPHA", "32"))),
            sft_lora_dropout=clamp_float(os.environ.get("SFT_LORA_DROPOUT", 0.05), 0.0, 0.5, 0.05),
            sft_target_modules=parse_csv_list(os.environ.get("SFT_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")),
            sft_use_bf16=asbool(os.environ.get("SFT_USE_BF16"), True),
            ollama_deploy_strategy=normalize_deploy_strategy(os.environ.get("OLLAMA_DEPLOY_STRATEGY", "auto")),
            llama_cpp_dir=os.environ.get("LLAMA_CPP_DIR", "./llama.cpp"),
            llama_cpp_convert_lora_to_gguf=os.environ.get("LLAMA_CPP_CONVERT_LORA_TO_GGUF", ""),
            ollama_gguf_adapter_path=os.environ.get("OLLAMA_GGUF_ADAPTER_PATH", "./artifacts/binance-qwen3-lora.gguf"),
            ollama_deploy_base_model=os.environ.get("OLLAMA_DEPLOY_BASE_MODEL", "qwen3:4b-instruct-2507-q4_K_M"),
            ollama_deploy_model_name=os.environ.get("OLLAMA_DEPLOY_MODEL_NAME", "binance-qwen3-sft"),
            ollama_modelfile_path=os.environ.get("OLLAMA_MODELFILE_PATH", "./artifacts/Modelfile.binance-qwen3-sft"),
            ollama_update_env_after_deploy=asbool(os.environ.get("OLLAMA_UPDATE_ENV_AFTER_DEPLOY"), True),
            compare_input_jsonl=os.environ.get("COMPARE_INPUT_JSONL", os.environ.get("SFT_OUTPUT_JSONL", "./artifacts/binance_sft_cases.jsonl")),
            compare_output_jsonl=os.environ.get("COMPARE_OUTPUT_JSONL", "./artifacts/model_compare_results.jsonl"),
            compare_max_cases=max(1, int(os.environ.get("COMPARE_MAX_CASES", "200"))),
        )


class BotDB:
    def __init__(self, path: str) -> None:
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trade_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                exchange TEXT NOT NULL,
                market TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                price REAL,
                qty REAL,
                notional_krw REAL,
                status TEXT NOT NULL,
                reason TEXT,
                order_id TEXT,
                raw_json TEXT
            );
            CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                exchange TEXT NOT NULL,
                total_equity_krw REAL NOT NULL,
                cash_krw REAL NOT NULL,
                invested_krw REAL NOT NULL,
                raw_json TEXT
            );
            CREATE TABLE IF NOT EXISTS bandit_stats (
                context TEXT NOT NULL,
                action TEXT NOT NULL,
                alpha REAL NOT NULL DEFAULT 1.0,
                beta REAL NOT NULL DEFAULT 1.0,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (context, action)
            );
            CREATE TABLE IF NOT EXISTS report_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                summary_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS pending_buy_orders (
                order_id TEXT PRIMARY KEY,
                market TEXT NOT NULL,
                price REAL,
                qty REAL,
                spend_krw REAL,
                submitted_at TEXT NOT NULL,
                last_checked_at TEXT,
                status TEXT NOT NULL,
                cancel_requested_at TEXT,
                canceled_at TEXT,
                raw_json TEXT
            );
            CREATE TABLE IF NOT EXISTS pending_orders (
                order_id TEXT PRIMARY KEY,
                market TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL,
                qty REAL,
                notional_krw REAL,
                submitted_at TEXT NOT NULL,
                last_checked_at TEXT,
                status TEXT NOT NULL,
                cancel_requested_at TEXT,
                canceled_at TEXT,
                raw_json TEXT
            );
            CREATE TABLE IF NOT EXISTS order_attempt_state (
                market TEXT NOT NULL,
                side TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_reason TEXT,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (market, side)
            );
            CREATE TABLE IF NOT EXISTS ai_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                due_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                exchange TEXT NOT NULL,
                market TEXT NOT NULL,
                context TEXT NOT NULL,
                model_name TEXT,
                rl_hint TEXT,
                prompt_json TEXT NOT NULL,
                decision_json TEXT NOT NULL,
                selected_action TEXT NOT NULL,
                confidence REAL,
                target_risk REAL,
                position_qty REAL,
                entry_price REAL,
                entry_equity_krw REAL,
                reward_horizon_minutes INTEGER NOT NULL DEFAULT 360,
                execution_status TEXT NOT NULL DEFAULT 'PENDING',
                execution_meta_json TEXT,
                reward_status TEXT NOT NULL DEFAULT 'PENDING',
                reward REAL,
                reward_price REAL,
                reward_meta_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ai_decisions_due ON ai_decisions (reward_status, due_at);
            """
        )
        self._migrate_pending_buy_orders()
        self.conn.commit()

    def _migrate_pending_buy_orders(self) -> None:
        tables = {
            row["name"]
            for row in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        if "pending_buy_orders" not in tables or "pending_orders" not in tables:
            return
        self.conn.execute(
            """
            INSERT OR IGNORE INTO pending_orders
            (order_id, market, side, price, qty, notional_krw, submitted_at, last_checked_at, status, cancel_requested_at, canceled_at, raw_json)
            SELECT
                order_id,
                market,
                'BUY' AS side,
                price,
                qty,
                spend_krw AS notional_krw,
                submitted_at,
                last_checked_at,
                status,
                cancel_requested_at,
                canceled_at,
                raw_json
            FROM pending_buy_orders
            """
        )

    def log_trade(self, exchange: str, market: str, side: str, order_type: str, price: Optional[Decimal], qty: Optional[Decimal], notional_krw: Optional[Decimal], status: str, reason: str, order_id: Optional[str] = None, raw: Optional[dict] = None) -> None:
        self.conn.execute(
            """
            INSERT INTO trade_logs
            (ts, exchange, market, side, order_type, price, qty, notional_krw, status, reason, order_id, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_kst().isoformat(), exchange, market, side, order_type,
                float(price) if price is not None else None,
                float(qty) if qty is not None else None,
                float(notional_krw) if notional_krw is not None else None,
                status, reason, order_id, safe_json_dumps(raw or {}),
            ),
        )
        self.conn.commit()

    def save_snapshot(self, exchange: str, total_equity_krw: Decimal, cash_krw: Decimal, invested_krw: Decimal, raw: dict) -> None:
        self.conn.execute(
            "INSERT INTO portfolio_snapshots (ts, exchange, total_equity_krw, cash_krw, invested_krw, raw_json) VALUES (?, ?, ?, ?, ?, ?)",
            (now_kst().isoformat(), exchange, float(total_equity_krw), float(cash_krw), float(invested_krw), safe_json_dumps(raw)),
        )
        self.conn.commit()

    def get_first_snapshot(self, exchange: str) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM portfolio_snapshots WHERE exchange = ? ORDER BY id ASC LIMIT 1", (exchange,)).fetchone()

    def get_snapshot_before(self, exchange: str, cutoff: datetime) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM portfolio_snapshots WHERE exchange = ? AND ts <= ? ORDER BY id DESC LIMIT 1",
            (exchange, cutoff.isoformat()),
        ).fetchone()

    def get_recent_trades(self, since: datetime) -> List[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM trade_logs WHERE ts >= ? ORDER BY id DESC", (since.isoformat(),)).fetchall()

    def save_report(self, summary: dict) -> None:
        self.conn.execute("INSERT INTO report_history (ts, summary_json) VALUES (?, ?)", (now_kst().isoformat(), safe_json_dumps(summary)))
        self.conn.commit()

    def log_ai_decision(
        self,
        exchange: str,
        market: str,
        context: str,
        model_name: str,
        rl_hint: str,
        prompt_payload: dict,
        decision: dict,
        position_qty: Decimal,
        entry_price: Decimal,
        entry_equity_krw: Decimal,
        reward_horizon_minutes: int,
    ) -> int:
        now = now_kst()
        due_at = now + timedelta(minutes=max(1, int(reward_horizon_minutes)))
        cur = self.conn.execute(
            """
            INSERT INTO ai_decisions
            (ts, due_at, updated_at, exchange, market, context, model_name, rl_hint, prompt_json, decision_json,
             selected_action, confidence, target_risk, position_qty, entry_price, entry_equity_krw,
             reward_horizon_minutes, execution_status, execution_meta_json, reward_status, reward, reward_price, reward_meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, 'PENDING', NULL, NULL, NULL)
            """,
            (
                now.isoformat(),
                due_at.isoformat(),
                now.isoformat(),
                exchange,
                market,
                context,
                model_name,
                rl_hint,
                safe_json_dumps(prompt_payload),
                safe_json_dumps(decision),
                str(decision.get("action", "HOLD")).upper(),
                float(decision.get("confidence", 0.0)),
                float(decision.get("target_risk", 0.0)),
                float(position_qty),
                float(entry_price),
                float(entry_equity_krw),
                max(1, int(reward_horizon_minutes)),
                safe_json_dumps({}),
            ),
        )
        self.conn.commit()
        return int(cur.lastrowid or 0)

    def update_ai_decision_execution(self, decision_id: int, execution_status: str, execution_meta: Optional[dict] = None) -> None:
        self.conn.execute(
            """
            UPDATE ai_decisions
            SET execution_status = ?, execution_meta_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (str(execution_status or "PENDING").upper(), safe_json_dumps(execution_meta or {}), now_kst().isoformat(), int(decision_id)),
        )
        self.conn.commit()

    def get_due_ai_decisions(self, as_of: datetime, limit: int = 500) -> List[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT * FROM ai_decisions
            WHERE reward_status = 'PENDING' AND due_at <= ?
            ORDER BY due_at ASC, id ASC
            LIMIT ?
            """,
            (as_of.isoformat(), max(1, int(limit))),
        ).fetchall()

    def complete_ai_decision_reward(self, decision_id: int, reward: float, reward_price: Decimal, reward_meta: Optional[dict] = None) -> None:
        self.conn.execute(
            """
            UPDATE ai_decisions
            SET reward_status = 'COMPLETED', reward = ?, reward_price = ?, reward_meta_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (float(reward), float(reward_price), safe_json_dumps(reward_meta or {}), now_kst().isoformat(), int(decision_id)),
        )
        self.conn.commit()


    def get_reward_labeled_ai_decisions(self, good_threshold: float, bad_threshold: float, limit: int = 2000) -> List[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT *
            FROM ai_decisions
            WHERE reward_status = 'COMPLETED'
              AND reward IS NOT NULL
              AND (reward >= ? OR reward <= ?)
            ORDER BY updated_at DESC, id DESC
            LIMIT ?
            """,
            (float(good_threshold), float(bad_threshold), max(1, int(limit))),
        ).fetchall()

    def get_bandit_stat(self, context: str, action: str) -> Tuple[float, float]:
        row = self.conn.execute("SELECT alpha, beta FROM bandit_stats WHERE context = ? AND action = ?", (context, action)).fetchone()
        if row is None:
            self.conn.execute(
                "INSERT OR IGNORE INTO bandit_stats (context, action, alpha, beta, updated_at) VALUES (?, ?, 1.0, 1.0, ?)",
                (context, action, now_kst().isoformat()),
            )
            self.conn.commit()
            return 1.0, 1.0
        return float(row["alpha"]), float(row["beta"])

    def update_bandit_stat(self, context: str, action: str, reward: float) -> None:
        alpha, beta = self.get_bandit_stat(context, action)
        if reward >= 0:
            alpha += min(1.0, reward + 0.5)
        else:
            beta += min(1.0, abs(reward) + 0.5)
        self.conn.execute(
            """
            INSERT INTO bandit_stats (context, action, alpha, beta, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(context, action) DO UPDATE SET alpha=excluded.alpha, beta=excluded.beta, updated_at=excluded.updated_at
            """,
            (context, action, alpha, beta, now_kst().isoformat()),
        )
        self.conn.commit()

    def add_pending_order(self, market: str, side: str, price: Decimal, qty: Decimal, notional_krw: Decimal, order_id: str, raw: Optional[dict] = None) -> None:
        self.conn.execute(
            """
            INSERT OR REPLACE INTO pending_orders
            (order_id, market, side, price, qty, notional_krw, submitted_at, last_checked_at, status, cancel_requested_at, canceled_at, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order_id, market, side.upper(),
                float(price) if price is not None else None,
                float(qty) if qty is not None else None,
                float(notional_krw) if notional_krw is not None else None,
                now_kst().isoformat(), None, "SUBMITTED", None, None, safe_json_dumps(raw or {}),
            ),
        )
        self.conn.commit()

    def get_open_pending_orders(self) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM pending_orders WHERE status IN ('SUBMITTED', 'PARTIALLY_FILLED', 'CANCEL_REQUESTED', 'LIVE', 'TRIGGERED', 'NOT_TRIGGERED') ORDER BY submitted_at ASC"
        ).fetchall()

    def update_pending_order_status(self, order_id: str, status: str, raw: Optional[dict] = None, canceled: bool = False) -> None:
        now = now_kst().isoformat()
        self.conn.execute(
            """
            UPDATE pending_orders
            SET status = ?, last_checked_at = ?, canceled_at = CASE WHEN ? THEN ? ELSE canceled_at END, raw_json = ?
            WHERE order_id = ?
            """,
            (status, now, bool(canceled), now if canceled else None, safe_json_dumps(raw or {}), order_id),
        )
        self.conn.commit()

    def mark_pending_order_cancel_requested(self, order_id: str, raw: Optional[dict] = None) -> None:
        now = now_kst().isoformat()
        self.conn.execute(
            """
            UPDATE pending_orders
            SET status = 'CANCEL_REQUESTED', last_checked_at = ?, cancel_requested_at = ?, raw_json = ?
            WHERE order_id = ?
            """,
            (now, now, safe_json_dumps(raw or {}), order_id),
        )
        self.conn.commit()

    def get_order_attempt_count(self, market: str, side: str) -> int:
        row = self.conn.execute(
            "SELECT attempt_count FROM order_attempt_state WHERE market = ? AND side = ?",
            (market.upper(), side.upper()),
        ).fetchone()
        return int(row["attempt_count"]) if row is not None else 0

    def set_order_attempt_count(self, market: str, side: str, attempt_count: int, reason: Optional[str] = None) -> None:
        self.conn.execute(
            """
            INSERT INTO order_attempt_state (market, side, attempt_count, last_reason, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(market, side) DO UPDATE SET
                attempt_count = excluded.attempt_count,
                last_reason = excluded.last_reason,
                updated_at = excluded.updated_at
            """,
            (market.upper(), side.upper(), max(0, int(attempt_count)), reason, now_kst().isoformat()),
        )
        self.conn.commit()

    def increment_order_attempt_count(self, market: str, side: str, reason: Optional[str] = None) -> int:
        next_count = self.get_order_attempt_count(market, side) + 1
        self.set_order_attempt_count(market, side, next_count, reason)
        return next_count

    def reset_order_attempt_count(self, market: str, side: str, reason: Optional[str] = None) -> None:
        self.set_order_attempt_count(market, side, 0, reason)


class HttpMixin:
    def __init__(self, timeout: int = 15) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ai-coinone-bot/1.0"})

    def _request(self, method: str, url: str, **kwargs: Any) -> dict:
        suppress_log_codes = {str(code) for code in (kwargs.pop("_suppress_log_codes", None) or [])}
        for attempt in range(3):
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
                if response.status_code >= 400:
                    response_code = ""
                    try:
                        body_json = response.json()
                        response_code = str(body_json.get("code") or "") if isinstance(body_json, dict) else ""
                    except Exception:
                        response_code = ""
                    if response_code not in suppress_log_codes:
                        log(f"[HTTP {response.status_code}] url={url} body={response.text}")
                response.raise_for_status()
                if not response.text.strip():
                    return {}
                return response.json()
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1.2 * (attempt + 1))
        return {}



class CoinoneClient(HttpMixin):
    base_url = "https://api.coinone.co.kr"

    def __init__(self, access_token: str, secret_key: str, timeout: int = 15) -> None:
        super().__init__(timeout=timeout)
        self.access_token = access_token
        self.secret_key = secret_key.encode("utf-8")

    def _signed_post(self, path: str, payload: dict) -> dict:
        payload = dict(payload)
        payload["access_token"] = self.access_token
        payload["nonce"] = str(uuid.uuid4())
        dumped = json.dumps(payload, separators=(",", ":"))
        encoded = base64.b64encode(dumped.encode("utf-8"))
        signature = hmac.new(self.secret_key, encoded, hashlib.sha512).hexdigest()
        headers = {
            "Content-Type": "application/json",
            "X-COINONE-PAYLOAD": encoded.decode("utf-8"),
            "X-COINONE-SIGNATURE": signature,
        }
        return self._request("POST", f"{self.base_url}{path}", headers=headers, data=dumped)

    def get_balances(self) -> List[dict]:
        response = self._signed_post("/v2.1/account/balance/all", {})
        return response.get("balances", []) or []

    def get_ticker(self, market: str) -> dict:
        quote, base = split_market(market)
        return self._request("GET", f"{self.base_url}/public/v2/ticker_new/{quote}/{base}")

    def get_all_tickers(self, quote_currency: str = "KRW") -> List[dict]:
        response = self._request("GET", f"{self.base_url}/public/v2/ticker_new/{quote_currency.upper()}")
        if isinstance(response, list):
            return response
        return response.get("tickers") or response.get("markets") or response.get("data") or []

    def get_markets(self, quote_currency: str = "KRW") -> List[str]:
        quote = quote_currency.upper()
        response = self._request("GET", f"{self.base_url}/public/v2/markets/{quote}")
        markets = response.get("markets") or response.get("data") or response.get("result") or []
        out: List[str] = []
        for item in markets:
            if isinstance(item, str):
                symbol = item.upper()
                out.append(symbol if "-" in symbol else f"{quote}-{symbol}")
            else:
                base = str(item.get("target_currency") or item.get("currency") or item.get("base_currency") or "").upper()
                if base:
                    out.append(f"{quote}-{base}")
        return out

    def get_orderbook(self, market: str) -> dict:
        quote, base = split_market(market)
        return self._request("GET", f"{self.base_url}/public/v2/orderbook/{quote}/{base}", params={"size": 15})

    def get_candles(self, market: str, interval: str = "1h", size: int = 200) -> pd.DataFrame:
        quote, base = split_market(market)
        response = self._request("GET", f"{self.base_url}/public/v2/chart/{quote}/{base}", params={"interval": interval, "size": size})
        rows = response.get("chart", []) or response.get("candles", []) or []
        norm = []
        for r in rows:
            ts = r.get("timestamp") or r.get("time") or r.get("t")
            norm.append({
                "timestamp": pd.to_datetime(int(ts), unit="ms", utc=True) if ts else pd.NaT,
                "open": float(r.get("open") or r.get("opening_price") or r.get("o") or 0),
                "high": float(r.get("high") or r.get("high_price") or r.get("h") or 0),
                "low": float(r.get("low") or r.get("low_price") or r.get("l") or 0),
                "close": float(r.get("close") or r.get("trade_price") or r.get("c") or 0),
                "volume": float(r.get("target_volume") or r.get("volume") or r.get("v") or 0),
            })
        if not norm:
            return pd.DataFrame()
        return pd.DataFrame(norm).sort_values("timestamp").reset_index(drop=True)

    def place_limit_order(self, market: str, side: str, price: Decimal, qty: Decimal, post_only: bool = True) -> dict:
        quote, base = split_market(market)
        payload = {
            "quote_currency": quote,
            "target_currency": base,
            "side": side.upper(),
            "type": "LIMIT",
            "qty": str(qty),
            "price": str(price),
            "post_only": bool(post_only),
        }
        return self._signed_post("/v2.1/order", payload)

    def get_order_detail(self, market: str, order_id: str) -> dict:
        quote, base = split_market(market)
        return self._signed_post("/v2.1/order/detail", {
            "quote_currency": quote,
            "target_currency": base,
            "order_id": order_id,
        })

    def get_completed_orders(self, market: str, size: int = 50) -> dict:
        quote, base = split_market(market)
        return self._signed_post("/v2.1/order/completed_orders", {
            "quote_currency": quote,
            "target_currency": base,
            "size": max(1, int(size)),
        })

    def get_active_orders(self, market: Optional[str] = None, order_types: Optional[List[str]] = None) -> dict:
        payload: Dict[str, Any] = {}
        if market:
            quote, base = split_market(market)
            payload["quote_currency"] = quote
            payload["target_currency"] = base
        if order_types is not None:
            payload["order_type"] = order_types
        return self._signed_post("/v2.1/order/active_orders", payload)

    def cancel_order(self, market: str, order_id: str) -> dict:
        quote, base = split_market(market)
        return self._signed_post("/v2.1/order/cancel", {
            "quote_currency": quote,
            "target_currency": base,
            "order_id": order_id,
        })


class BinanceFuturesClient(HttpMixin):
    def __init__(self, api_key: str, secret_key: str, base_url: str, timeout: int = 15, recv_window: int = 5000, default_leverage: int = 3, default_margin_type: str = "ISOLATED", hedge_mode: bool = False) -> None:
        super().__init__(timeout=timeout)
        self.api_key = api_key
        self.secret_key = secret_key.encode("utf-8")
        self.base_url = base_url.rstrip("/")
        self.recv_window = int(recv_window)
        self.default_leverage = int(default_leverage)
        self.default_margin_type = str(default_margin_type or "ISOLATED").upper()
        self.hedge_mode = bool(hedge_mode)
        self._exchange_info_cache: Optional[dict] = None
        self._symbol_info_cache: Dict[str, dict] = {}
        self._configured_symbols: set[str] = set()
        self._position_mode_synced = False

    def _signed_request(self, method: str, path: str, params: Optional[dict] = None, suppress_log_codes: Optional[set[str]] = None) -> Any:
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                payload = dict(params or {})
                payload.setdefault("recvWindow", self.recv_window)
                payload["timestamp"] = int(time.time() * 1000)
                query = urlencode(payload, doseq=True)
                signature = hmac.new(self.secret_key, query.encode("utf-8"), hashlib.sha256).hexdigest()
                url = f"{self.base_url}{path}?{query}&signature={signature}"
                headers = {"X-MBX-APIKEY": self.api_key}
                return self._request(method, url, headers=headers, _suppress_log_codes=suppress_log_codes)
            except Exception as exc:
                last_exc = exc
                if attempt == 2:
                    break
                time.sleep(1.0 * (attempt + 1))
        raise last_exc or RuntimeError(f"Signed request failed: {path}")

    def _public_get(self, path: str, params: Optional[dict] = None) -> Any:
        return self._request("GET", f"{self.base_url}{path}", params=params or {})

    def _handle_known_config_error(self, exc: Exception, allowed_codes: set[str]) -> bool:
        response = getattr(exc, "response", None)
        text = ""
        if response is not None:
            try:
                text = response.text or ""
            except Exception:
                text = ""
        for code in allowed_codes:
            if code in text:
                return True
        return False

    def sync_position_mode(self) -> None:
        if self._position_mode_synced:
            return
        try:
            self._signed_request("POST", "/fapi/v1/positionSide/dual", {"dualSidePosition": "true" if self.hedge_mode else "false"}, suppress_log_codes={"-4059"})
        except Exception as exc:
            if not self._handle_known_config_error(exc, {"-4059"}):
                log(f"[BINANCE CONFIG WARN] position mode sync failed: {exc}")
        self._position_mode_synced = True

    def get_exchange_info(self) -> dict:
        if self._exchange_info_cache is None:
            self._exchange_info_cache = self._public_get("/fapi/v1/exchangeInfo")
        return self._exchange_info_cache or {}

    def get_symbol_info(self, market: str) -> dict:
        symbol = market_to_symbol(market)
        if symbol in self._symbol_info_cache:
            return self._symbol_info_cache[symbol]
        info = self.get_exchange_info()
        for item in info.get("symbols", []) or []:
            if str(item.get("symbol") or "").upper() == symbol.upper():
                self._symbol_info_cache[symbol] = item
                return item
        raise KeyError(f"Binance symbol not found for market={market}")

    def _symbol_filters(self, market: str) -> dict:
        info = self.get_symbol_info(market)
        out: Dict[str, dict] = {}
        for filt in info.get("filters", []) or []:
            ftype = str(filt.get("filterType") or "")
            if ftype:
                out[ftype] = filt
        return out

    def normalize_price_qty(self, market: str, price: Decimal, qty: Decimal) -> Tuple[Decimal, Decimal, Decimal]:
        filters = self._symbol_filters(market)
        price_filter = filters.get("PRICE_FILTER", {})
        lot_filter = filters.get("LOT_SIZE", {})
        tick_size = to_decimal(price_filter.get("tickSize") or 0)
        step_size = to_decimal(lot_filter.get("stepSize") or 0)
        norm_price = floor_to_step(to_decimal(price), tick_size) if tick_size > 0 else to_decimal(price)
        norm_qty = floor_to_step(to_decimal(qty), step_size) if step_size > 0 else to_decimal(qty)
        min_notional = Decimal("0")
        if "MIN_NOTIONAL" in filters:
            min_notional = max(min_notional, to_decimal(filters["MIN_NOTIONAL"].get("notional") or filters["MIN_NOTIONAL"].get("minNotional") or 0))
        if "NOTIONAL" in filters:
            min_notional = max(min_notional, to_decimal(filters["NOTIONAL"].get("minNotional") or 0))
        return norm_price, norm_qty, min_notional

    def ensure_symbol_config(self, market: str) -> None:
        self.sync_position_mode()
        symbol = market_to_symbol(market)
        if symbol in self._configured_symbols:
            return
        try:
            self._signed_request("POST", "/fapi/v1/marginType", {"symbol": symbol, "marginType": self.default_margin_type}, suppress_log_codes={"-4046"})
        except Exception as exc:
            if not self._handle_known_config_error(exc, {"-4046", "No need to change margin type"}):
                log(f"[BINANCE CONFIG WARN] margin type {symbol}: {exc}")
        try:
            self._signed_request("POST", "/fapi/v1/leverage", {"symbol": symbol, "leverage": self.default_leverage})
        except Exception as exc:
            log(f"[BINANCE CONFIG WARN] leverage {symbol}: {exc}")
        self._configured_symbols.add(symbol)

    def get_balances(self) -> dict:
        account = self._signed_request("GET", "/fapi/v3/account", {})
        positions = self._signed_request("GET", "/fapi/v3/positionRisk", {})
        if isinstance(account, dict):
            account["positions"] = positions if isinstance(positions, list) else []
        return account

    def get_ticker(self, market: str) -> dict:
        symbol = market_to_symbol(market)
        data = self._public_get("/fapi/v1/ticker/24hr", {"symbol": symbol})
        return {
            "last": data.get("lastPrice") or data.get("last") or data.get("closePrice") or 0,
            "close": data.get("lastPrice") or data.get("closePrice") or 0,
            "quote_volume": data.get("quoteVolume") or 0,
            "symbol": symbol,
        }

    def get_all_tickers(self, quote_currency: str = "USDT") -> List[dict]:
        quote = quote_currency.upper()
        rows = self._public_get("/fapi/v1/ticker/24hr", {})
        if not isinstance(rows, list):
            rows = [rows]
        out: List[dict] = []
        for item in rows:
            symbol = str(item.get("symbol") or "").upper()
            if not symbol.endswith(quote):
                continue
            base = symbol[:-len(quote)]
            if not base:
                continue
            out.append({
                "target_currency": base,
                "base_currency": base,
                "quote_currency": quote,
                "last": item.get("lastPrice") or 0,
                "close": item.get("lastPrice") or 0,
                "quote_volume": item.get("quoteVolume") or 0,
            })
        return out

    def get_markets(self, quote_currency: str = "USDT") -> List[str]:
        quote = quote_currency.upper()
        info = self.get_exchange_info()
        out: List[str] = []
        for item in info.get("symbols", []) or []:
            if str(item.get("contractType") or "") != "PERPETUAL":
                continue
            if str(item.get("status") or "") != "TRADING":
                continue
            if str(item.get("quoteAsset") or "").upper() != quote:
                continue
            base = str(item.get("baseAsset") or "").upper()
            if base:
                out.append(f"{quote}-{base}")
        return out

    def get_orderbook(self, market: str) -> dict:
        symbol = market_to_symbol(market)
        data = self._public_get("/fapi/v1/depth", {"symbol": symbol, "limit": 20})
        return {
            "bids": [{"price": p, "qty": q} for p, q in (data.get("bids") or [])],
            "asks": [{"price": p, "qty": q} for p, q in (data.get("asks") or [])],
        }

    def get_candles(self, market: str, interval: str = "1h", size: int = 200) -> pd.DataFrame:
        symbol = market_to_symbol(market)
        rows = self._public_get("/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": max(1, int(size))})
        if not isinstance(rows, list) or not rows:
            return pd.DataFrame()
        norm = []
        for row in rows:
            norm.append({
                "timestamp": pd.to_datetime(int(row[0]), unit="ms", utc=True),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            })
        return pd.DataFrame(norm).sort_values("timestamp").reset_index(drop=True)

    def place_limit_order(self, market: str, side: str, price: Decimal, qty: Decimal, post_only: bool = True, reduce_only: Optional[bool] = None, position_side: Optional[str] = None, time_in_force: Optional[str] = None, allow_below_min_notional: bool = False) -> dict:
        self.ensure_symbol_config(market)
        symbol = market_to_symbol(market)
        norm_price, norm_qty, min_notional = self.normalize_price_qty(market, price, qty)
        actual_notional = norm_price * norm_qty
        if norm_price <= 0:
            raise ValueError(f"Invalid Binance price after normalization: market={market} price={price} normalized={norm_price}")
        if norm_qty <= 0:
            raise ValueError(f"Invalid Binance quantity after normalization: market={market} qty={qty} normalized={norm_qty}")
        if min_notional > 0 and actual_notional < min_notional and not allow_below_min_notional:
            raise ValueError(
                f"Binance min notional not met after normalization: market={market} notional={actual_notional} min_notional={min_notional}"
            )
        params: Dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "LIMIT",
            "timeInForce": str(time_in_force or ("GTX" if post_only else "GTC")).upper(),
            "price": str(norm_price),
            "quantity": str(norm_qty),
        }
        if position_side:
            params["positionSide"] = str(position_side).upper()
        if reduce_only is not None and not self.hedge_mode:
            params["reduceOnly"] = "true" if reduce_only else "false"
        data = self._signed_request("POST", "/fapi/v1/order", params)
        return {"order_id": str(data.get("orderId") or ""), **(data if isinstance(data, dict) else {})}

    def get_order_detail(self, market: str, order_id: str) -> dict:
        symbol = market_to_symbol(market)
        data = self._signed_request("GET", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id})
        if not isinstance(data, dict):
            data = {}
        orig_qty = to_decimal(data.get("origQty") or 0)
        executed_qty = to_decimal(data.get("executedQty") or 0)
        remain_qty = max(Decimal("0"), orig_qty - executed_qty)
        return {
            "order": {
                "order_id": str(data.get("orderId") or order_id),
                "status": str(data.get("status") or "UNKNOWN").upper(),
                "remain_qty": str(remain_qty),
                "executed_qty": str(executed_qty),
                "price": data.get("price") or data.get("avgPrice") or 0,
                "side": data.get("side") or "",
            },
            "raw": data,
        }

    def get_completed_orders(self, market: str, size: int = 50) -> dict:
        symbol = market_to_symbol(market)
        rows = self._signed_request("GET", "/fapi/v1/allOrders", {"symbol": symbol, "limit": max(1, int(size))})
        if not isinstance(rows, list):
            rows = []
        completed = []
        for item in rows:
            status = str(item.get("status") or "").upper()
            if status == "FILLED":
                completed.append({"order_id": str(item.get("orderId") or ""), "status": status})
        return {"completed_orders": completed, "orders": rows}

    def get_active_orders(self, market: Optional[str] = None, order_types: Optional[List[str]] = None) -> dict:
        params: Dict[str, Any] = {}
        if market:
            params["symbol"] = market_to_symbol(market)
        rows = self._signed_request("GET", "/fapi/v1/openOrders", params)
        if not isinstance(rows, list):
            rows = []
        return {"orders": rows}

    def cancel_order(self, market: str, order_id: str) -> dict:
        symbol = market_to_symbol(market)
        data = self._signed_request("DELETE", "/fapi/v1/order", {"symbol": symbol, "orderId": order_id})
        return data if isinstance(data, dict) else {"order_id": order_id}


class SentimentProvider(HttpMixin):
    DEFAULT_FEEDS = [
        "https://www.coindesk.com/arc/outboundfeeds/rss/",
        "https://cointelegraph.com/rss",
    ]

    def __init__(self, ollama: "OllamaClient", timeout: int = 15) -> None:
        super().__init__(timeout=timeout)
        self.ollama = ollama

    def fear_greed(self) -> dict:
        response = self._request("GET", "https://api.alternative.me/fng/", params={"limit": 1, "format": "json"})
        data = response.get("data", [{}])[0] if response.get("data") else {}
        return {"value": int(data.get("value", 50)), "classification": data.get("value_classification", "Neutral")}

    def recent_headlines(self, limit_per_feed: int = 4) -> List[str]:
        headlines: List[str] = []
        for url in self.DEFAULT_FEEDS:
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                items = response.text.split("<item>")[: limit_per_feed + 1]
                for item in items[1:]:
                    if "<title>" in item:
                        title = item.split("<title>", 1)[1].split("</title>", 1)[0]
                        title = title.replace("<![CDATA[", "").replace("]]>", "").strip()
                        if title:
                            headlines.append(title)
            except Exception:
                continue
        return headlines[: limit_per_feed * len(self.DEFAULT_FEEDS)]

    def score(self) -> dict:
        fg = self.fear_greed()
        headlines = self.recent_headlines()
        if headlines:
            prompt = (
                "You are scoring crypto market sentiment. Return strict JSON with keys sentiment_score (-1 to 1), summary, risk_flag (true/false).\n"
                f"FearGreed={fg['value']} ({fg['classification']})\nHeadlines={headlines}\n"
            )
            try:
                parsed = self.ollama.generate_json(prompt)
                if isinstance(parsed, dict) and "sentiment_score" in parsed:
                    return {
                        "fear_greed": fg,
                        "headlines": headlines,
                        "sentiment_score": float(parsed.get("sentiment_score", 0)),
                        "summary": parsed.get("summary", ""),
                        "risk_flag": bool(parsed.get("risk_flag", False)),
                    }
            except Exception:
                pass
        fallback_score = (fg["value"] - 50) / 50.0
        return {
            "fear_greed": fg,
            "headlines": headlines,
            "sentiment_score": float(max(-1.0, min(1.0, fallback_score))),
            "summary": fg["classification"],
            "risk_flag": fg["value"] < 18,
        }


class OnChainProvider(HttpMixin):
    def __init__(self, etherscan_api_key: Optional[str], timeout: int = 15) -> None:
        super().__init__(timeout=timeout)
        self.etherscan_api_key = etherscan_api_key

    def bitcoin_pressure(self) -> dict:
        recommended = self._request("GET", "https://mempool.space/api/v1/fees/recommended")
        fastest = float(recommended.get("fastestFee", 0))
        hour = float(recommended.get("hourFee", 0))
        score = min(1.0, fastest / 100.0) - min(1.0, hour / 100.0) * 0.5
        return {"fastest_fee_sat_vb": fastest, "hour_fee_sat_vb": hour, "pressure_score": float(max(-1.0, min(1.0, score)))}

    def ethereum_gas(self) -> dict:
        if not self.etherscan_api_key:
            return {"gas_score": 0.0, "status": "skipped"}
        response = self._request(
            "GET", "https://api.etherscan.io/v2/api",
            params={"chainid": 1, "module": "gastracker", "action": "gasoracle", "apikey": self.etherscan_api_key},
        )
        result = response.get("result", {}) or {}
        safe = float(result.get("SafeGasPrice", 0) or 0)
        propose = float(result.get("ProposeGasPrice", 0) or 0)
        score = min(1.0, propose / 100.0) - min(1.0, safe / 100.0) * 0.3
        return {"safe_gwei": safe, "propose_gwei": propose, "gas_score": float(max(-1.0, min(1.0, score)))}

    def combined(self) -> dict:
        btc = self.bitcoin_pressure()
        eth = self.ethereum_gas()
        combined_score = float(max(-1.0, min(1.0, (btc.get("pressure_score", 0.0) + eth.get("gas_score", 0.0)) / 2)))
        return {"bitcoin": btc, "ethereum": eth, "combined_score": combined_score}


class OllamaClient(HttpMixin):
    def __init__(self, base_url: str, models: List[str], timeout: int = 30, ensemble_enabled: bool = True) -> None:
        super().__init__(timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.models = [m for m in models if m]
        self.ensemble_enabled = bool(ensemble_enabled)
        if not self.models:
            raise ValueError("At least one Ollama model is required")

    def _generate(self, model: str, prompt: str, options: Optional[dict] = None) -> str:
        payload = {"model": model, "prompt": prompt, "stream": False, "format": "json"}
        if options:
            payload["options"] = options
        response = self._request("POST", f"{self.base_url}/api/generate", json=payload)
        return response.get("response", "{}")

    def _merge_values(self, values: List[Any]) -> Any:
        non_null = [v for v in values if v is not None]
        if not non_null:
            return None
        if all(isinstance(v, bool) for v in non_null):
            return sum(1 for v in non_null if v) >= (len(non_null) / 2)
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
            return sum(float(v) for v in non_null) / len(non_null)
        if all(isinstance(v, dict) for v in non_null):
            merged: Dict[str, Any] = {}
            for key in set().union(*(v.keys() for v in non_null)):
                merged[key] = self._merge_values([v.get(key) for v in non_null])
            return merged
        for value in non_null:
            if value not in ("", {}, []):
                return value
        return non_null[0]

    def generate_json(self, prompt: str) -> dict:
        outputs: List[dict] = []
        last_error: Optional[Exception] = None
        models = self.models if self.ensemble_enabled else self.models[:1]
        for model in models:
            try:
                outputs.append(json.loads(self._generate(model, prompt)))
            except Exception as exc:
                last_error = exc
        if not outputs:
            raise last_error or RuntimeError("No Ollama models returned valid JSON")
        if len(outputs) == 1:
            return outputs[0]
        merged = self._merge_values(outputs)
        return merged if isinstance(merged, dict) else outputs[0]

    def generate_json_by_model(self, prompt: str, options: Optional[dict] = None) -> Dict[str, dict]:
        outputs: Dict[str, dict] = {}
        last_error: Optional[Exception] = None
        for model in self.models:
            try:
                outputs[model] = json.loads(self._generate(model, prompt, options=options))
            except Exception as exc:
                last_error = exc
                outputs[model] = {"_error": str(exc)}
        if not outputs:
            raise last_error or RuntimeError("No Ollama models available for comparison")
        return outputs


class BanditRLPolicy:
    ACTIONS = ["BUY", "HOLD", "SELL"]

    def __init__(self, db: BotDB) -> None:
        self.db = db

    def choose(self, context: str) -> str:
        samples = {}
        for action in self.ACTIONS:
            alpha, beta = self.db.get_bandit_stat(context, action)
            samples[action] = np.random.beta(alpha, beta)
        return max(samples, key=samples.get)


TRADE_DECISION_SYSTEM_PROMPT = (
    "You are a crypto trading model. Return strict JSON with keys: action (BUY/HOLD/SELL), confidence (0..1), target_risk (0..1), reason, use_grid (true/false), use_rebound (true/false), enabled_signals, signal_weights, parameter_overrides.\n"
    "Interpret position_qty in the exchange-specific context. Favor capital preservation in downtrends."
)


def build_trade_decision_prompt(payload: dict) -> str:
    return (
        f"market={payload.get('market')}\n"
        f"position_qty={payload.get('position_qty')}\n"
        f"technical={safe_json_dumps(payload.get('technical', {}))}\n"
        f"sentiment={safe_json_dumps(payload.get('sentiment', {}))}\n"
        f"onchain={safe_json_dumps(payload.get('onchain', {}))}\n"
        f"rl_hint={payload.get('rl_hint', '')}\n"
    )


class StrategyEngine:
    SIGNAL_KEYS = ["technical", "sentiment", "onchain", "arbitrage", "rebound", "rl"]

    def __init__(self, cfg: Config, ollama: OllamaClient) -> None:
        self.cfg = cfg
        self.ollama = ollama

    @staticmethod
    def indicators(df: pd.DataFrame) -> dict:
        if df.empty or len(df) < 60:
            raise ValueError("Not enough candle data")
        close = df["close"].astype(float)
        low = df["low"].astype(float)
        high = df["high"].astype(float)
        volume = df["volume"].astype(float)
        ema_fast = close.ewm(span=20, adjust=False).mean()
        ema_slow = close.ewm(span=50, adjust=False).mean()
        delta = close.diff().fillna(0)
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean().replace(0, 1e-9)
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        mid = close.rolling(20).mean()
        std = close.rolling(20).std().fillna(0)
        lower = mid - 2 * std
        upper = mid + 2 * std
        atr = (high - low).rolling(14).mean()
        return {
            "last_close": float(close.iloc[-1]),
            "ema_fast": float(ema_fast.iloc[-1]),
            "ema_slow": float(ema_slow.iloc[-1]),
            "rsi": float(rsi.iloc[-1]),
            "bollinger_lower": float(lower.iloc[-1]),
            "bollinger_upper": float(upper.iloc[-1]),
            "atr": float(atr.iloc[-1]),
            "recent_return": float((close.iloc[-1] / close.iloc[-8]) - 1),
            "volume_zscore": float(((volume.iloc[-20:] - volume.iloc[-20:].mean()) / (volume.iloc[-20:].std() or 1)).iloc[-1]),
            "trend_up": bool(ema_fast.iloc[-1] > ema_slow.iloc[-1]),
            "rebound_signal": bool(rsi.iloc[-1] < 35 and close.iloc[-1] > close.iloc[-2] and close.iloc[-1] <= lower.iloc[-1] * 1.03),
        }

    def context_key(self, technical: dict, sentiment: dict, onchain: dict) -> str:
        regime = "bull" if technical["trend_up"] else "bear"
        sentiment_bucket = "fear" if sentiment["fear_greed"]["value"] < 30 else "greed" if sentiment["fear_greed"]["value"] > 65 else "neutral"
        chain_bucket = "hot" if onchain["combined_score"] > 0.25 else "cool" if onchain["combined_score"] < -0.25 else "flat"
        return f"{regime}:{sentiment_bucket}:{chain_bucket}"

    def default_enabled_signals(self) -> dict:
        return {key: True for key in self.SIGNAL_KEYS}

    def default_signal_weights(self) -> dict:
        return {"technical": 1.0, "sentiment": 0.9, "onchain": 0.9, "arbitrage": 0.0, "rebound": 0.8, "rl": 0.6}

    def normalize_enabled_signals(self, raw: Any) -> dict:
        defaults = self.default_enabled_signals()
        if not isinstance(raw, dict):
            return defaults
        return {key: bool(raw.get(key, value)) for key, value in defaults.items()}

    def normalize_signal_weights(self, raw: Any) -> dict:
        defaults = self.default_signal_weights()
        if not isinstance(raw, dict):
            return defaults
        return {key: clamp_float(raw.get(key, value), 0.0, 2.0, value) for key, value in defaults.items()}

    def normalize_parameter_overrides(self, raw: Any) -> dict:
        if not isinstance(raw, dict):
            return {}
        out: Dict[str, Any] = {}
        if "stop_loss_pct" in raw:
            out["stop_loss_pct"] = clamp_float(raw.get("stop_loss_pct"), 0.01, 0.10, float(self.cfg.stop_loss_pct))
        if "take_profit_pct" in raw:
            out["take_profit_pct"] = clamp_float(raw.get("take_profit_pct"), 0.01, 0.20, float(self.cfg.take_profit_pct))
        if "highest_take_profit_pct" in raw:
            out["highest_take_profit_pct"] = clamp_float(raw.get("highest_take_profit_pct"), 0.015, 0.30, float(self.cfg.highest_take_profit_pct))
        if "trailing_take_profit_gap_pct" in raw:
            out["trailing_take_profit_gap_pct"] = clamp_float(raw.get("trailing_take_profit_gap_pct"), 0.002, 0.05, float(self.cfg.trailing_take_profit_gap_pct))
        if "grid_step_pct" in raw:
            out["grid_step_pct"] = clamp_float(raw.get("grid_step_pct"), 0.005, 0.05, float(self.cfg.grid_step_pct))
        if "grid_levels" in raw:
            out["grid_levels"] = clamp_int(raw.get("grid_levels"), 1, 10, self.cfg.grid_levels)
        if "buy_budget_fraction" in raw:
            out["buy_budget_fraction"] = clamp_float(raw.get("buy_budget_fraction"), 0.05, 0.95, 0.95)
        if "rebound_buy_fraction" in raw:
            out["rebound_buy_fraction"] = clamp_float(raw.get("rebound_buy_fraction"), 0.02, 0.30, 0.10)
        if "max_exposure_multiplier" in raw:
            out["max_exposure_multiplier"] = clamp_float(raw.get("max_exposure_multiplier"), 0.25, 1.50, 1.0)
        return out

    def effective_params(self, decision: dict) -> dict:
        overrides = self.normalize_parameter_overrides(decision.get("parameter_overrides", {}))
        return {
            "stop_loss_pct": Decimal(str(overrides.get("stop_loss_pct", float(self.cfg.stop_loss_pct)))),
            "take_profit_pct": Decimal(str(overrides.get("take_profit_pct", float(self.cfg.take_profit_pct)))),
            "highest_take_profit_pct": Decimal(str(overrides.get("highest_take_profit_pct", float(self.cfg.highest_take_profit_pct)))),
            "trailing_take_profit_gap_pct": Decimal(str(overrides.get("trailing_take_profit_gap_pct", float(self.cfg.trailing_take_profit_gap_pct)))),
            "grid_step_pct": Decimal(str(overrides.get("grid_step_pct", float(self.cfg.grid_step_pct)))),
            "grid_levels": int(overrides.get("grid_levels", self.cfg.grid_levels)),
            "buy_budget_fraction": Decimal(str(overrides.get("buy_budget_fraction", 0.95))),
            "rebound_buy_fraction": Decimal(str(overrides.get("rebound_buy_fraction", 0.10))),
            "max_exposure_multiplier": Decimal(str(overrides.get("max_exposure_multiplier", 1.00))),
        }

    def ai_decision(self, market: str, technical: dict, sentiment: dict, onchain: dict, rl_action: str, position_qty: Decimal) -> dict:
        prompt = TRADE_DECISION_SYSTEM_PROMPT + "\n" + build_trade_decision_prompt({
            "market": market,
            "position_qty": position_qty,
            "technical": technical,
            "sentiment": sentiment,
            "onchain": onchain,
            "rl_hint": rl_action,
        })
        try:
            decision = self.ollama.generate_json(prompt) if self.cfg.ai_override_enabled else {}
            action = str(decision.get("action", "HOLD")).upper()
            if action not in {"BUY", "HOLD", "SELL"}:
                action = "HOLD"
            if position_qty <= 0 and action == "SELL":
                action = "HOLD"
            raw_reason = str(decision.get("reason", "ollama"))
            normalized = {
                "action": action,
                "confidence": clamp_float(decision.get("confidence", 0.5), 0.0, 1.0, 0.5),
                "target_risk": clamp_float(decision.get("target_risk", 0.3), 0.0, 1.0, 0.3),
                "reason_raw": raw_reason,
                "reason": make_compact_reason(action, technical, sentiment, onchain, rl_action, position_qty, "model"),
                "use_grid": bool(decision.get("use_grid", False)),
                "use_rebound": bool(decision.get("use_rebound", technical.get("rebound_signal", False))),
                "enabled_signals": self.normalize_enabled_signals(decision.get("enabled_signals")),
                "signal_weights": self.normalize_signal_weights(decision.get("signal_weights")),
                "parameter_overrides": self.normalize_parameter_overrides(decision.get("parameter_overrides")),
            }
            normalized["effective_params"] = self.effective_params(normalized)
            return normalized
        except Exception as exc:
            if technical["trend_up"] and sentiment["sentiment_score"] > -0.1:
                action, conf = "BUY", 0.60
            elif position_qty > 0 and (not technical["trend_up"]) and sentiment["fear_greed"]["value"] < 25:
                action, conf = "SELL", 0.68
            elif technical["rebound_signal"] and sentiment["fear_greed"]["value"] <= 25:
                action, conf = "BUY", 0.52
            else:
                action, conf = "HOLD", 0.50
            fallback = {
                "action": action,
                "confidence": conf,
                "target_risk": 0.25,
                "reason_raw": f"heuristic-fallback:{type(exc).__name__}:{exc}",
                "reason": make_compact_reason(action, technical, sentiment, onchain, rl_action, position_qty, "fallback"),
                "use_grid": action == "HOLD",
                "use_rebound": technical.get("rebound_signal", False),
                "enabled_signals": self.default_enabled_signals(),
                "signal_weights": self.default_signal_weights(),
                "parameter_overrides": {},
            }
            fallback["effective_params"] = self.effective_params(fallback)
            return fallback




class BinanceStrategyEngine(StrategyEngine):
    def ai_decision(self, market: str, technical: dict, sentiment: dict, onchain: dict, rl_action: str, position_qty: Decimal) -> dict:
        prompt = (
            "You are a Binance USDⓈ-M futures trading model. Return strict JSON with keys: "
            "action (BUY/HOLD/SELL), confidence (0..1), target_risk (0..1), reason, use_grid (true/false), "
            "use_rebound (true/false), enabled_signals, signal_weights, parameter_overrides. "
            "position_qty > 0 means long, position_qty < 0 means short, position_qty = 0 means flat. "
            "BUY can open/increase long or reduce/close short. SELL can open/increase short or reduce/close long."
            " Favor capital preservation in downtrends and reduce unnecessary flips.\n"
        ) + build_trade_decision_prompt({
            "market": market,
            "position_qty": position_qty,
            "technical": technical,
            "sentiment": sentiment,
            "onchain": onchain,
            "rl_hint": rl_action,
        })
        try:
            decision = self.ollama.generate_json(prompt) if self.cfg.ai_override_enabled else {}
            action = str(decision.get("action", "HOLD")).upper()
            if action not in {"BUY", "HOLD", "SELL"}:
                action = "HOLD"
            if action == "SELL" and position_qty == 0 and not getattr(self.cfg, "futures_short_enabled", True):
                action = "HOLD"
            raw_reason = str(decision.get("reason", "ollama"))
            normalized = {
                "action": action,
                "confidence": clamp_float(decision.get("confidence", 0.5), 0.0, 1.0, 0.5),
                "target_risk": clamp_float(decision.get("target_risk", 0.3), 0.0, 1.0, 0.3),
                "reason_raw": raw_reason,
                "reason": make_compact_reason(action, technical, sentiment, onchain, rl_action, position_qty, "model"),
                "use_grid": bool(decision.get("use_grid", False)),
                "use_rebound": bool(decision.get("use_rebound", technical.get("rebound_signal", False))),
                "enabled_signals": self.normalize_enabled_signals(decision.get("enabled_signals")),
                "signal_weights": self.normalize_signal_weights(decision.get("signal_weights")),
                "parameter_overrides": self.normalize_parameter_overrides(decision.get("parameter_overrides")),
            }
            normalized["effective_params"] = self.effective_params(normalized)
            return normalized
        except Exception as exc:
            if technical["trend_up"] and sentiment["sentiment_score"] > -0.1:
                action, conf = "BUY", 0.60
            elif (not technical["trend_up"]) and sentiment["fear_greed"]["value"] < 40:
                action = "SELL" if getattr(self.cfg, "futures_short_enabled", True) or position_qty > 0 else "HOLD"
                conf = 0.62 if action == "SELL" else 0.50
            elif technical["rebound_signal"] and sentiment["fear_greed"]["value"] <= 25:
                action, conf = "BUY", 0.52
            else:
                action, conf = "HOLD", 0.50
            fallback = {
                "action": action,
                "confidence": conf,
                "target_risk": 0.25,
                "reason_raw": f"heuristic-fallback:{type(exc).__name__}:{exc}",
                "reason": make_compact_reason(action, technical, sentiment, onchain, rl_action, position_qty, "fallback"),
                "use_grid": False,
                "use_rebound": technical.get("rebound_signal", False),
                "enabled_signals": self.default_enabled_signals(),
                "signal_weights": self.default_signal_weights(),
                "parameter_overrides": {},
            }
            fallback["effective_params"] = self.effective_params(fallback)
            return fallback


class Mailer:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def send(self, subject: str, body: str) -> None:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.cfg.gmail_user
        msg["To"] = self.cfg.target_email
        msg.set_content(body)
        with __import__("smtplib").SMTP_SSL(self.cfg.gmail_smtp_host, self.cfg.gmail_smtp_port, context=ssl.create_default_context()) as smtp:
            smtp.login(self.cfg.gmail_user, self.cfg.gmail_password)
            smtp.send_message(msg)


class CoinoneOnlyAIBot:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.db = BotDB(cfg.db_path)
        self.ollama = OllamaClient(cfg.ollama_url, cfg.ollama_models, timeout=cfg.request_timeout * 2, ensemble_enabled=cfg.ollama_ensemble_enabled)
        self.coinone = CoinoneClient(cfg.coinone_access_token, cfg.coinone_secret_key, timeout=cfg.request_timeout)
        self.sentiment = SentimentProvider(self.ollama, timeout=cfg.request_timeout)
        self.onchain = OnChainProvider(cfg.etherscan_api_key, timeout=cfg.request_timeout)
        self.strategy = StrategyEngine(cfg, self.ollama)
        self.rl = BanditRLPolicy(self.db)
        self.mailer = Mailer(cfg)
        self.last_report_at = 0.0
        self.last_market_refresh_at = 0.0
        self.last_pending_sync_at = 0.0
        self.cycle_market_cursor = 0
        self.position_peak_state: Dict[str, Dict[str, Decimal]] = {}

    def current_market_attempt(self, market: str, side: str) -> int:
        return self.db.get_order_attempt_count(market, side)

    def reset_market_attempt(self, market: str, side: str, reason: str = "") -> None:
        self.db.reset_order_attempt_count(market, side, reason)

    def fail_market_attempt(self, market: str, side: str, reason: str = "") -> int:
        return self.db.increment_order_attempt_count(market, side, reason)

    def trading_blocked_symbols(self) -> set[str]:
        blocked = {s.upper() for s in self.cfg.excluded_symbols}
        if not self.cfg.allow_stablecoin_trading:
            blocked.update(s.upper() for s in self.cfg.stablecoin_symbols)
        return blocked

    def is_trading_blocked_symbol(self, symbol: str) -> bool:
        return symbol.upper() in self.trading_blocked_symbols()

    def book_prices(self, orderbook: dict) -> Tuple[List[Decimal], List[Decimal]]:
        bids = sorted({to_decimal(x.get("price") or x.get("bid_price") or 0) for x in (orderbook.get("bids") or []) if to_decimal(x.get("price") or x.get("bid_price") or 0) > 0}, reverse=True)
        asks = sorted({to_decimal(x.get("price") or x.get("ask_price") or 0) for x in (orderbook.get("asks") or []) if to_decimal(x.get("price") or x.get("ask_price") or 0) > 0})
        return bids, asks

    def select_limit_price(self, side: str, orderbook: dict, preferred_level: int) -> Decimal:
        bids, asks = self.book_prices(orderbook)
        idx = max(0, preferred_level - 1)
        is_post_only = bool(getattr(self.cfg, "coinone_post_only", True))
        side_upper = side.upper()

        if side_upper == "BUY":
            primary = bids if is_post_only else asks
            secondary = asks if is_post_only else bids
        else:
            primary = asks if is_post_only else bids
            secondary = bids if is_post_only else asks

        if primary:
            return primary[min(idx, len(primary) - 1)]
        return secondary[0] if secondary else Decimal("0")

    def mark_price_from_orderbook(self, orderbook: dict) -> Decimal:
        bids, asks = self.book_prices(orderbook)
        if bids and asks:
            return (bids[0] + asks[0]) / Decimal("2")
        return bids[0] if bids else (asks[0] if asks else Decimal("0"))

    def refresh_target_markets(self, force: bool = False) -> None:
        if not self.cfg.auto_discover_markets:
            self.cfg.target_markets = filter_markets(self.cfg.target_markets, list(self.trading_blocked_symbols()))
            return
        if not force and (time.time() - self.last_market_refresh_at) < self.cfg.market_refresh_seconds:
            return
        quote = self.cfg.market_quote_currency.upper()
        markets = filter_markets(self.coinone.get_markets(quote), list(self.trading_blocked_symbols()))
        volume_map: Dict[str, Decimal] = {}
        try:
            for ticker in self.coinone.get_all_tickers(quote):
                base = str(ticker.get("target_currency") or "").upper()
                if base:
                    volume_map[f"{quote}-{base}"] = to_decimal(ticker.get("quote_volume") or 0)
        except Exception as exc:
            log(f"[MARKET DISCOVERY WARN] ticker ranking unavailable: {exc}")
        if volume_map:
            markets = sorted(markets, key=lambda m: (-float(volume_map.get(m, Decimal("0"))), m))
        self.cfg.target_markets = markets[: self.cfg.market_universe_size]
        self.last_market_refresh_at = time.time()
        preview = ", ".join(self.cfg.target_markets[: min(10, len(self.cfg.target_markets))])
        suffix = " ..." if len(self.cfg.target_markets) > 10 else ""
        log(f"[MARKETS] refreshed count={len(self.cfg.target_markets)} quote={quote} -> {preview}{suffix}")

    def build_position_map(self, balances: List[dict]) -> Tuple[Decimal, Dict[str, Dict[str, Decimal]]]:
        cash = Decimal("0")
        positions: Dict[str, Dict[str, Decimal]] = {}
        excluded = {s.upper() for s in self.cfg.excluded_symbols}
        for item in balances:
            currency = (item.get("currency") or item.get("unit_currency") or "").upper()
            available = to_decimal(item.get("available") or item.get("balance") or 0)
            locked = to_decimal(item.get("limit") or item.get("locked") or 0)
            qty = available + locked
            avg_buy = to_decimal(item.get("average_price") or item.get("avg_buy_price") or 0)
            if currency == self.cfg.market_quote_currency.upper():
                cash = qty
            elif qty > 0 and currency not in excluded:
                positions[currency] = {"qty": qty, "avg_buy": avg_buy}
        return cash, positions

    def available_buy_budget_krw(self, cash_krw: Decimal) -> Decimal:
        return max(Decimal("0"), cash_krw * self.cfg.buy_spend_safety_buffer)

    def can_open_new_position_without_ai(self, cash_krw: Decimal, equity_krw: Decimal) -> Tuple[bool, dict]:
        min_order_krw = self.cfg.exchange_min_trade_krw
        budget_krw = self.available_buy_budget_krw(cash_krw)
        max_risk_target_krw = equity_krw * self.cfg.max_exposure_per_market * Decimal("1.5")
        max_openable_krw = min(budget_krw, max_risk_target_krw)
        return max_openable_krw >= min_order_krw, {
            "budget_krw": budget_krw,
            "max_risk_target_krw": max_risk_target_krw,
            "max_openable_krw": max_openable_krw,
            "min_order_krw": min_order_krw,
        }

    def estimate_equity(self, balances: List[dict]) -> Tuple[Decimal, Decimal, Decimal, dict]:
        total = Decimal("0")
        cash = Decimal("0")
        invested = Decimal("0")
        assets = []
        excluded_assets = []
        excluded = {s.upper() for s in self.cfg.excluded_symbols}
        for item in balances:
            currency = (item.get("currency") or item.get("unit_currency") or "").upper()
            available = to_decimal(item.get("available") or item.get("balance") or 0)
            locked = to_decimal(item.get("limit") or item.get("locked") or 0)
            qty = available + locked
            if qty <= 0:
                continue
            if currency == self.cfg.market_quote_currency.upper():
                total += qty
                cash += qty
                assets.append({"currency": currency, "qty": float(qty), "krw_value": float(qty)})
                continue
            if currency in excluded:
                excluded_assets.append({"currency": currency, "qty": float(qty)})
                continue
            market = f"{self.cfg.market_quote_currency}-{currency}"
            try:
                ob = self.coinone.get_orderbook(market)
                price = self.mark_price_from_orderbook(ob)
                if price <= 0:
                    ticker = self.coinone.get_ticker(market)
                    price = to_decimal(ticker.get("last") or ticker.get("close") or 0)
            except Exception:
                price = Decimal("0")
            krw_value = qty * price
            total += krw_value
            invested += krw_value
            assets.append({"currency": currency, "qty": float(qty), "krw_value": float(krw_value)})
        return total, cash, invested, {"assets": assets, "excluded_assets": excluded_assets, "excluded_symbols": list(excluded)}

    def capture_snapshot(self, balances: List[dict]) -> dict:
        total, cash, invested, raw = self.estimate_equity(balances)
        self.db.save_snapshot("coinone", total, cash, invested, raw)
        return {"exchange": "coinone", "total_equity_krw": total, "cash_krw": cash, "invested_krw": invested}

    def refresh_cycle_state(self) -> dict:
        balances = self.coinone.get_balances()
        cash, positions = self.build_position_map(balances)
        equity, _, _, _ = self.estimate_equity(balances)
        return {"balances": balances, "cash_krw": cash, "positions": positions, "equity_krw": equity}

    def select_cycle_markets(self) -> List[str]:
        markets = list(self.cfg.target_markets)
        if not markets:
            return []
        batch = max(1, min(self.cfg.markets_per_cycle, len(markets)))
        if batch >= len(markets):
            self.cycle_market_cursor = 0
            return markets
        start = self.cycle_market_cursor % len(markets)
        selected = [markets[(start + idx) % len(markets)] for idx in range(batch)]
        self.cycle_market_cursor = (start + batch) % len(markets)
        return selected

    def order_terminal_statuses(self) -> set[str]:
        return {
            "FILLED",
            "CANCELED",
            "PARTIALLY_CANCELED",
            "NOT_TRIGGERED_CANCELED",
            "NOT_TRIGGERED_PARTIALLY_CANCELED",
            "CANCELED_NO_ORDER",
            "CANCELED_LIMIT_PRICE_EXCEED",
            "CANCELED_UNDER_PRODUCT_UNIT",
        }

    def order_live_statuses(self) -> set[str]:
        return {"LIVE", "PARTIALLY_FILLED", "TRIGGERED", "NOT_TRIGGERED", "SUBMITTED", "CANCEL_REQUESTED"}

    def _extract_completed_order_rows(self, payload: dict) -> List[dict]:
        if not isinstance(payload, dict):
            return []
        for key in ("completed_orders", "complete_orders", "orders", "data"):
            rows = payload.get(key)
            if isinstance(rows, list):
                return rows
        return []

    def _lookup_completed_order(self, market: str, order_id: str) -> Optional[dict]:
        try:
            payload = self.coinone.get_completed_orders(market, size=self.cfg.completed_order_lookup_limit)
            for row in self._extract_completed_order_rows(payload):
                if str(row.get("order_id") or row.get("id") or "") == str(order_id):
                    return {"status": "FILLED", "payload": payload, "row": row}
        except Exception as exc:
            log(f"[COMPLETED ORDER LOOKUP WARN] {getattr(self, "exchange_name", "coinone")} {market} order_id={order_id}: {exc}")
        return None

    def send_order_retry_failure_mail(self, market: str, side: str, price: Decimal, requested_qty: Decimal, remaining_qty: Decimal, attempts: int, reason: str) -> None:
        filled_qty = max(Decimal("0"), requested_qty - remaining_qty)
        subject = f"[AI COIN BOT] {market} {side} 체결 {attempts}회 실패"
        body = "\n".join([
            "AI 코인 투자 주문 실패 알림",
            f"발생 시각(KST): {now_kst().strftime('%Y-%m-%d %H:%M:%S')}",
            f"마켓: {market}",
            f"주문 방향: {side}",
            f"고정 주문가: {price}",
            f"최초 주문 수량: {requested_qty}",
            f"체결 수량: {filled_qty}",
            f"미체결 잔량: {remaining_qty}",
            f"재시도 횟수: {attempts}",
            f"사유: {reason}",
        ])
        try:
            self.mailer.send(subject, body)
            log(f"[ORDER FAILURE MAIL] sent market={market} side={side} attempts={attempts}")
        except Exception as exc:
            log(f"[ORDER FAILURE MAIL FAILED] market={market} side={side}: {exc}")

    def wait_for_order_attempt(self, market: str, side: str, order_id: str, price: Decimal, submitted_qty: Decimal, submitted_notional: Decimal, reason: str, attempt_no: int, timeout_override_seconds: Optional[int] = None) -> dict:
        terminal_statuses = self.order_terminal_statuses()
        live_statuses = self.order_live_statuses()
        timeout_seconds = max(1, int(timeout_override_seconds or self.cfg.order_timeout_seconds))
        poll_seconds = max(1, self.cfg.order_poll_interval_seconds)
        deadline = time.time() + timeout_seconds
        last_payload: dict = {}
        last_status = "SUBMITTED"
        remain_qty = submitted_qty
        executed_qty = Decimal("0")
        submitted_monotonic = time.time()
        next_progress_log_at = submitted_monotonic + min(10, timeout_seconds)
        exchange_name = getattr(self, "exchange_name", "coinone")

        while time.time() < deadline:
            try:
                detail = self.coinone.get_order_detail(market, order_id)
                last_payload = detail
                order = detail.get("order") or {}
                last_status = str(order.get("status") or last_status).upper()
                remain_qty = quantize_down(to_decimal(order.get("remain_qty") or 0), 8)
                executed_qty = quantize_down(to_decimal(order.get("executed_qty") or 0), 8)
                if last_status in terminal_statuses or (executed_qty > 0 and remain_qty <= 0):
                    final_status = "FILLED" if executed_qty > 0 and remain_qty <= 0 else last_status
                    self.db.update_pending_order_status(order_id, final_status, detail, canceled=("CANCEL" in final_status))
                    return {
                        "status": final_status,
                        "executed_qty": executed_qty,
                        "remain_qty": max(Decimal("0"), remain_qty),
                        "payload": detail,
                    }
                tracked_status = last_status if last_status in live_statuses else "SUBMITTED"
                self.db.update_pending_order_status(order_id, tracked_status, detail)
                now_monotonic = time.time()
                if now_monotonic >= next_progress_log_at:
                    elapsed = int(now_monotonic - submitted_monotonic)
                    log(f"[ORDER WAIT] {exchange_name} {market} {side} order_id={order_id} attempt={attempt_no} elapsed={elapsed}s status={tracked_status} remain_qty={remain_qty}")
                    next_progress_log_at = now_monotonic + min(10, timeout_seconds)
            except Exception as exc:
                log(f"[ORDER POLL WARN] {exchange_name} {market} {side} order_id={order_id} attempt={attempt_no}: {exc}")
            time.sleep(poll_seconds)

        completed = self._lookup_completed_order(market, order_id)
        if completed:
            self.db.update_pending_order_status(order_id, "FILLED", completed["payload"], canceled=False)
            return {
                "status": "FILLED",
                "executed_qty": submitted_qty,
                "remain_qty": Decimal("0"),
                "payload": completed["payload"],
            }

        try:
            cancel_result = self.coinone.cancel_order(market, order_id)
            self.db.mark_pending_order_cancel_requested(order_id, cancel_result)
            last_payload = cancel_result or last_payload
            log(f"[{side} CANCEL REQUESTED] {exchange_name} {market} order_id={order_id} attempt={attempt_no} timeout={timeout_seconds}s")
        except Exception as exc:
            log(f"[{side} CANCEL FAILED] {exchange_name} {market} order_id={order_id} attempt={attempt_no}: {exc}")

        cancel_deadline = time.time() + max(3, poll_seconds * 3)
        while time.time() < cancel_deadline:
            try:
                detail = self.coinone.get_order_detail(market, order_id)
                last_payload = detail
                order = detail.get("order") or {}
                last_status = str(order.get("status") or last_status).upper()
                remain_qty = quantize_down(to_decimal(order.get("remain_qty") or 0), 8)
                executed_qty = quantize_down(to_decimal(order.get("executed_qty") or 0), 8)
                if last_status in terminal_statuses or last_status == "CANCEL_REQUESTED":
                    final_status = "FILLED" if executed_qty > 0 and remain_qty <= 0 else last_status
                    self.db.update_pending_order_status(order_id, final_status, detail, canceled=("CANCEL" in final_status))
                    return {
                        "status": final_status,
                        "executed_qty": executed_qty,
                        "remain_qty": max(Decimal("0"), remain_qty),
                        "payload": detail,
                    }
            except Exception as exc:
                log(f"[ORDER CANCEL POLL WARN] {exchange_name} {market} {side} order_id={order_id} attempt={attempt_no}: {exc}")
            completed = self._lookup_completed_order(market, order_id)
            if completed:
                self.db.update_pending_order_status(order_id, "FILLED", completed["payload"], canceled=False)
                return {
                    "status": "FILLED",
                    "executed_qty": submitted_qty,
                    "remain_qty": Decimal("0"),
                    "payload": completed["payload"],
                }
            time.sleep(1)

        self.db.update_pending_order_status(order_id, "CANCEL_REQUESTED", last_payload, canceled=False)
        return {
            "status": "CANCEL_REQUESTED",
            "executed_qty": executed_qty,
            "remain_qty": max(Decimal("0"), remain_qty),
            "payload": last_payload,
        }

    def execute_limit_order_with_retry(self, market: str, side: str, price: Decimal, qty: Decimal, notional_krw: Decimal, reason: str, order_kwargs: Optional[dict] = None, timeout_override_seconds: Optional[int] = None, allow_below_min_notional: bool = False, price_adjustment_pct: Decimal = Decimal("0")) -> bool:
        qty = quantize_down(qty, 8)
        initial_price = quantize_down(price, 0) if price == price.to_integral_value() else price
        last_price = initial_price
        requested_qty = qty
        remaining_qty = qty
        max_attempts = max(1, self.cfg.order_max_attempts)
        failure_note = ""
        consecutive_attempt = self.current_market_attempt(market, side)
        exchange_name = getattr(self, "exchange_name", "coinone")
        side_upper = side.upper()
        side_level = self.cfg.buy_book_level if side_upper == "BUY" else self.cfg.sell_book_level
        order_kwargs = dict(order_kwargs or {})
        requested_time_in_force = str(order_kwargs.get("time_in_force") or "").upper()
        post_only = self.cfg.coinone_post_only and requested_time_in_force not in {"IOC", "FOK"}

        if requested_qty <= 0 or initial_price <= 0:
            log(f"[SKIP {side}] {exchange_name} {market} invalid fixed order params price={initial_price} qty={requested_qty}")
            return False

        if self.cfg.dry_run:
            self.db.log_trade(exchange_name, market, side, "LIMIT", initial_price, requested_qty, notional_krw, "DRY_RUN", reason)
            log(f"[DRY {side}] {exchange_name} {market} price={initial_price} qty={requested_qty} reason={reason}")
            return True

        for local_attempt in range(1, max_attempts + 1):
            remaining_qty = quantize_down(remaining_qty, 8)
            attempt_price = last_price if last_price > 0 else initial_price
            try:
                refreshed_orderbook = self.coinone.get_orderbook(market)
                refreshed_price = self.select_limit_price(side, refreshed_orderbook, side_level)
                if refreshed_price > 0:
                    adjusted_price = refreshed_price
                    if price_adjustment_pct > 0:
                        adjusted_price = refreshed_price * (Decimal("1") - price_adjustment_pct if side_upper == "SELL" else Decimal("1") + price_adjustment_pct)
                    attempt_price = quantize_down(adjusted_price, 0) if adjusted_price == adjusted_price.to_integral_value() else adjusted_price
            except Exception as exc:
                log(f"[ORDERBOOK REFRESH WARN] {exchange_name} {market} {side}: {exc}")
            if attempt_price <= 0:
                attempt_price = initial_price
            last_price = attempt_price
            remaining_notional = attempt_price * remaining_qty
            display_attempt = consecutive_attempt + 1
            attempt_tag = f"attempt={display_attempt} local_attempt={local_attempt}/{max_attempts}"

            if remaining_qty <= 0:
                self.reset_market_attempt(market, side, f"{reason} | success before submit")
                return True
            if remaining_notional < self.cfg.exchange_min_trade_krw and not allow_below_min_notional:
                if remaining_qty < requested_qty:
                    self.reset_market_attempt(market, side, f"{reason} | partial fill success; dust remain")
                    self.db.log_trade(
                        exchange_name, market, side, "LIMIT", attempt_price, remaining_qty, remaining_notional,
                        "PARTIAL_DUST_REMAIN", f"{reason} | remain below minimum after partial fills | last_attempt={display_attempt}", None, {"attempt": display_attempt, "local_attempt": local_attempt, "remaining_qty": str(remaining_qty)},
                    )
                    log(f"[{side} PARTIAL COMPLETE] {exchange_name} {market} remaining_notional={remaining_notional} below minimum; stop retry")
                    return True
                log(f"[SKIP {side}] {exchange_name} {market} below min notional={remaining_notional}")
                return False
            try:
                result = self.coinone.place_limit_order(market, side, attempt_price, remaining_qty, post_only=post_only, allow_below_min_notional=allow_below_min_notional, **order_kwargs)
                order_id = result.get("order_id")
                self.db.log_trade(
                    exchange_name, market, side, "LIMIT", attempt_price, remaining_qty, remaining_notional,
                    "SUBMITTED", f"{reason} | {attempt_tag}", order_id, result,
                )
                if order_id:
                    self.db.add_pending_order(market, side, attempt_price, remaining_qty, remaining_notional, order_id, result)
                log(f"[{side}] {exchange_name} {market} price={attempt_price} qty={remaining_qty} order_id={order_id} {attempt_tag}")
                if not order_id:
                    raise RuntimeError(f"{exchange_name} response missing order_id")

                outcome = self.wait_for_order_attempt(market, side, order_id, attempt_price, remaining_qty, remaining_notional, reason, display_attempt, timeout_override_seconds=timeout_override_seconds)
                executed_qty = quantize_down(outcome.get("executed_qty") or Decimal("0"), 8)
                remaining_qty = quantize_down(outcome.get("remain_qty") or Decimal("0"), 8)
                status = str(outcome.get("status") or "UNKNOWN").upper()
                payload = outcome.get("payload") or {}

                if remaining_qty <= 0:
                    self.reset_market_attempt(market, side, f"{reason} | filled")
                    self.db.log_trade(
                        exchange_name, market, side, "LIMIT", attempt_price, requested_qty, attempt_price * requested_qty,
                        "FILLED", f"{reason} | completed attempt={display_attempt} local_attempt={local_attempt}/{max_attempts}", order_id, payload,
                    )
                    log(f"[{side} FILLED] {exchange_name} {market} attempts={display_attempt} local_attempt={local_attempt}/{max_attempts}")
                    return True

                if status == "CANCEL_REQUESTED":
                    consecutive_attempt = self.fail_market_attempt(market, side, f"{reason} | cancel not confirmed")
                    failure_note = f"attempt={consecutive_attempt} cancel not confirmed"
                    self.db.log_trade(
                        exchange_name, market, side, "LIMIT", attempt_price, remaining_qty, attempt_price * remaining_qty,
                        "FAILED_CANCEL_UNCONFIRMED", f"{reason} | {failure_note} | local_attempt={local_attempt}/{max_attempts}", order_id, payload,
                    )
                    log(f"[{side} ABORT] {exchange_name} {market} {failure_note} remaining_qty={remaining_qty}")
                    break

                if executed_qty > 0:
                    self.reset_market_attempt(market, side, f"{reason} | partial fill success")
                    consecutive_attempt = 0
                    self.db.log_trade(
                        exchange_name, market, side, "LIMIT", attempt_price, executed_qty, attempt_price * executed_qty,
                        "PARTIAL_RETRY", f"{reason} | partial fill reset attempt | status={status} remaining_qty={remaining_qty} | last_attempt={display_attempt} local_attempt={local_attempt}/{max_attempts}", order_id, payload,
                    )
                    log(f"[{side} PARTIAL] {exchange_name} {market} attempt_reset=0 last_attempt={display_attempt} local_attempt={local_attempt}/{max_attempts} executed_qty={executed_qty} remaining_qty={remaining_qty} status={status}")
                else:
                    consecutive_attempt = self.fail_market_attempt(market, side, f"{reason} | retry pending status={status}")
                    self.db.log_trade(
                        exchange_name, market, side, "LIMIT", attempt_price, remaining_qty, attempt_price * remaining_qty,
                        "RETRY_PENDING", f"{reason} | attempt={consecutive_attempt} status={status} local_attempt={local_attempt}/{max_attempts}", order_id, payload,
                    )
                    log(f"[{side} RETRY] {exchange_name} {market} attempt={consecutive_attempt} local_attempt={local_attempt}/{max_attempts} remaining_qty={remaining_qty} status={status} repriced={attempt_price}")
            except ValueError as exc:
                consecutive_attempt = self.fail_market_attempt(market, side, f"{reason} | invalid order params | {exc}")
                self.db.log_trade(
                    exchange_name, market, side, "LIMIT", attempt_price, remaining_qty, attempt_price * remaining_qty,
                    "FAILED_INVALID", f"{reason} | attempt={consecutive_attempt} local_attempt={local_attempt}/{max_attempts}: {exc}", None, {},
                )
                log(f"[{side} ABORT] {exchange_name} {market} attempt={consecutive_attempt} local_attempt={local_attempt}/{max_attempts}: {exc}")
                break
            except Exception as exc:
                error_text = str(exc)
                is_margin_error = ("-2019" in error_text) or ("Margin is insufficient" in error_text) or ("insufficient balance" in error_text.lower())
                consecutive_attempt = self.fail_market_attempt(market, side, f"{reason} | {exc}")
                self.db.log_trade(
                    exchange_name, market, side, "LIMIT", attempt_price, remaining_qty, attempt_price * remaining_qty,
                    "FAILED_NO_MARGIN" if is_margin_error else "FAILED", f"{reason} | attempt={consecutive_attempt} local_attempt={local_attempt}/{max_attempts}: {exc}", None, {},
                )
                log(f"[{side} FAILED] {exchange_name} {market} attempt={consecutive_attempt} local_attempt={local_attempt}/{max_attempts}: {exc}")
                if is_margin_error:
                    failure_note = f"attempt={consecutive_attempt} margin insufficient"
                    break

        final_attempts = self.current_market_attempt(market, side)
        failure_reason = f"{reason} | {failure_note}" if failure_note else f"{reason} | max attempts exceeded (local={max_attempts}, cumulative={final_attempts})"
        self.db.log_trade(
            exchange_name, market, side, "LIMIT", last_price, remaining_qty, last_price * remaining_qty,
            "FAILED_MAX_RETRY", failure_reason, None, {"remaining_qty": str(remaining_qty), "attempt": final_attempts},
        )
        self.send_order_retry_failure_mail(market, side, last_price, requested_qty, remaining_qty, final_attempts, failure_reason)
        return False

    def sync_pending_orders(self) -> None:
        self.last_pending_sync_at = time.time()
        terminal_statuses = self.order_terminal_statuses()
        cancellable_statuses = {"LIVE", "PARTIALLY_FILLED", "TRIGGERED", "NOT_TRIGGERED"}
        timeout_seconds = max(5, self.cfg.order_timeout_seconds)
        now = now_kst()
        for row in self.db.get_open_pending_orders():
            order_id = row["order_id"]
            market = row["market"]
            side = str(row["side"] or "BUY").upper()
            submitted_at = datetime.fromisoformat(row["submitted_at"])
            age_seconds = (now - submitted_at).total_seconds()
            try:
                detail = self.coinone.get_order_detail(market, order_id)
                order = detail.get("order") or {}
                status = str(order.get("status") or row["status"] or "UNKNOWN").upper()
                remain_qty = to_decimal(order.get("remain_qty") or 0)
                executed_qty = to_decimal(order.get("executed_qty") or 0)
                if status in terminal_statuses or remain_qty <= 0:
                    final_status = "FILLED" if remain_qty <= 0 and executed_qty > 0 else status
                    self.db.update_pending_order_status(order_id, final_status, detail, canceled=("CANCEL" in final_status))
                    continue
                completed = self._lookup_completed_order(market, order_id)
                if completed:
                    self.db.update_pending_order_status(order_id, "FILLED", completed["payload"], canceled=False)
                    continue
                if age_seconds >= timeout_seconds and status in cancellable_statuses:
                    if str(row["status"] or "").upper() != "CANCEL_REQUESTED":
                        cancel_result = self.coinone.cancel_order(market, order_id)
                        self.db.mark_pending_order_cancel_requested(order_id, cancel_result)
                        self.db.log_trade(
                            getattr(self, "exchange_name", "coinone"), market, side, "LIMIT",
                            to_decimal(row["price"]), to_decimal(row["qty"]), to_decimal(row["notional_krw"]),
                            "CANCEL_REQUESTED", f"timeout>{timeout_seconds}s unfilled/partial", order_id, cancel_result,
                        )
                        log(f"[{side} CANCEL REQUESTED] {getattr(self, "exchange_name", "coinone")} {market} order_id={order_id} age={int(age_seconds)}s status={status} remain_qty={remain_qty}")
                    continue
                tracked_status = status if status in {"LIVE", "PARTIALLY_FILLED", "TRIGGERED", "NOT_TRIGGERED"} else "SUBMITTED"
                self.db.update_pending_order_status(order_id, tracked_status, detail)
            except Exception as exc:
                log(f"[PENDING ORDER CHECK WARN] {getattr(self, "exchange_name", "coinone")} {market} {side} order_id={order_id}: {exc}")

    def maybe_sync_pending_orders(self, force: bool = False) -> None:
        if force or (time.time() - self.last_pending_sync_at) >= self.cfg.pending_sync_interval_seconds:
            self.sync_pending_orders()

    def build_reward_price_map(self) -> Dict[str, Decimal]:
        quote = self.cfg.market_quote_currency.upper()
        out: Dict[str, Decimal] = {}
        try:
            for ticker in self.coinone.get_all_tickers(quote):
                base = str(ticker.get("target_currency") or ticker.get("currency") or ticker.get("base_currency") or "").upper()
                if not base:
                    continue
                price = to_decimal(
                    ticker.get("last")
                    or ticker.get("close")
                    or ticker.get("trade_price")
                    or ticker.get("last_price")
                    or ticker.get("price")
                    or 0
                )
                if price > 0:
                    out[f"{quote}-{base}"] = price
        except Exception as exc:
            log(f"[REWARD PRICE MAP WARN] {exc}")
        return out

    def market_price_for_reward(self, market: str, price_map: Optional[Dict[str, Decimal]] = None) -> Decimal:
        cached = (price_map or {}).get(market)
        if cached and cached > 0:
            return cached
        try:
            ticker = self.coinone.get_ticker(market)
            price = to_decimal(ticker.get("last") or ticker.get("close") or ticker.get("trade_price") or 0)
            if price > 0:
                return price
        except Exception:
            pass
        try:
            orderbook = self.coinone.get_orderbook(market)
            return self.mark_price_from_orderbook(orderbook)
        except Exception:
            return Decimal("0")

    def compute_decision_reward(self, action: str, entry_price: Decimal, current_price: Decimal) -> Tuple[float, dict]:
        if entry_price <= 0 or current_price <= 0:
            raise ValueError("entry_price/current_price must be positive")
        raw_return = float((current_price - entry_price) / entry_price)
        action = str(action or "HOLD").upper()
        if action == "BUY":
            reward = raw_return
            formula = "BUY=return"
        elif action == "SELL":
            reward = -raw_return
            formula = "SELL=-return"
        else:
            reward = 0.01 - abs(raw_return)
            formula = "HOLD=0.01-abs(return)"
        reward = max(-1.0, min(1.0, reward))
        return reward, {
            "formula": formula,
            "raw_return": raw_return,
            "entry_price": float(entry_price),
            "current_price": float(current_price),
        }

    def bandit_rewardable_statuses(self) -> set[str]:
        return {
            "BUY_SUBMITTED",
            "SELL_SUBMITTED",
            "NO_ORDER",
            "HOLD_GRID_PLACED",
            "HOLD_REBOUND_BUY_SUBMITTED",
            "STRONG_HOLD_BUY_SUBMITTED",
        }

    def evaluate_due_ai_decisions(self) -> None:
        rows = self.db.get_due_ai_decisions(now_kst())
        if not rows:
            return
        price_map = self.build_reward_price_map()
        completed = 0
        pending_retry = 0
        bandit_updates = 0
        for row in rows:
            decision_id = int(row["id"])
            market = str(row["market"])
            current_price = self.market_price_for_reward(market, price_map)
            if current_price <= 0:
                pending_retry += 1
                continue
            entry_price = to_decimal(row["entry_price"] or 0)
            if entry_price <= 0:
                pending_retry += 1
                continue
            action = str(row["selected_action"] or "HOLD").upper()
            execution_status = str(row["execution_status"] or "PENDING").upper()
            reward, meta = self.compute_decision_reward(action, entry_price, current_price)
            meta["execution_status"] = execution_status
            meta["due_at"] = row["due_at"]
            self.db.complete_ai_decision_reward(decision_id, reward, current_price, meta)
            completed += 1
            if execution_status in self.bandit_rewardable_statuses() and row["context"]:
                self.db.update_bandit_stat(str(row["context"]), action, reward)
                bandit_updates += 1
        if completed or pending_retry:
            log(f"[AI REWARD] completed={completed} bandit_updates={bandit_updates} pending_price_retry={pending_retry}")

    def _normalize_futures_order(self, market: str, side: str, price: Decimal, qty: Decimal, allow_below_min_notional: bool = False) -> Optional[Tuple[Decimal, Decimal, Decimal]]:
        norm_price, norm_qty, min_notional = self.coinone.normalize_price_qty(market, price, abs(qty))
        actual_notional = norm_price * norm_qty
        min_trade = max(self.cfg.exchange_min_trade_krw, min_notional)
        if norm_price <= 0:
            log(f"[SKIP {side}] binance_futures {market} invalid normalized price raw={price} normalized={norm_price}")
            return None
        if norm_qty <= 0:
            log(f"[SKIP {side}] binance_futures {market} invalid normalized qty raw={qty} normalized={norm_qty}")
            return None
        if actual_notional < min_trade and not allow_below_min_notional:
            log(
                f"[SKIP {side}] binance_futures {market} below normalized min notional={actual_notional} min_required={min_trade} raw_qty={qty} normalized_qty={norm_qty}"
            )
            return None
        return norm_price, norm_qty, actual_notional

    def submit_buy(self, market: str, price: Decimal, spend_krw: Decimal, reason: str) -> bool:
        _, base = split_market(market)
        if self.is_trading_blocked_symbol(base):
            return False
        qty = quantize_down(spend_krw / price, 8) if price > 0 else Decimal("0")
        notional = price * qty
        if spend_krw < self.cfg.exchange_min_trade_krw:
            log(f"[SKIP BUY] coinone {market} below min spend={spend_krw}")
            return False
        if qty <= 0:
            log(f"[SKIP BUY] coinone {market} invalid qty at price={price}")
            return False
        return self.execute_limit_order_with_retry(market, "BUY", price, qty, notional, reason)

    def submit_sell(self, market: str, price: Decimal, qty: Decimal, reason: str) -> bool:
        _, base = split_market(market)
        if self.is_trading_blocked_symbol(base):
            return False
        qty = quantize_down(qty, 8)
        notional = price * qty
        if qty <= 0:
            return False
        if notional < self.cfg.exchange_min_trade_krw:
            log(f"[SKIP SELL] binance_futures {market} below min notional={notional}")
            return False
        return self.execute_limit_order_with_retry(market, "SELL", price, qty, notional, reason)

    def rebalance_buy(self, market: str, mark_price: Decimal, buy_limit_price: Decimal, current_qty: Decimal, target_notional: Decimal, cash_krw: Decimal, decision: dict, stats: dict) -> bool:
        effective = decision.get("effective_params") or self.strategy.effective_params(decision)
        if current_qty <= 0 and decision["confidence"] < self.cfg.buy_confidence_threshold:
            stats["buy_low_conf"] += 1
            log(f"[SKIP BUY] coinone {market} low confidence={decision['confidence']:.2f}")
            return False
        current_notional = current_qty * (mark_price if mark_price > 0 else buy_limit_price)
        desired_target = target_notional if current_qty > 0 else max(target_notional, self.cfg.min_new_position_krw)
        gap = max(Decimal("0"), desired_target - current_notional)
        budget_cap = cash_krw * effective["buy_budget_fraction"] * self.cfg.buy_spend_safety_buffer
        spend = min(gap, budget_cap)
        if current_qty <= 0:
            spend = max(spend, min(self.cfg.min_new_position_krw, budget_cap))
        if spend < self.cfg.exchange_min_trade_krw:
            stats["buy_below_min"] += 1
            log(f"[SKIP BUY] coinone {market} below min spend={spend} target={desired_target} cash={cash_krw}")
            return False
        spend = min(spend, budget_cap)
        ok = self.submit_buy(market, buy_limit_price, spend, decision["reason"])
        if ok:
            stats["buy_submitted"] += 1
        return ok

    def rebound_buy(self, market: str, buy_limit_price: Decimal, cash_krw: Decimal, decision: dict) -> bool:
        effective = decision.get("effective_params") or self.strategy.effective_params(decision)
        spend = min(cash_krw * effective["rebound_buy_fraction"] * self.cfg.buy_spend_safety_buffer, self.cfg.min_trade_krw * Decimal("2"))
        if spend < self.cfg.exchange_min_trade_krw:
            log(f"[SKIP REBOUND BUY] coinone {market} below min spend={spend}")
            return False
        return self.submit_buy(market, buy_limit_price, spend, "rebound")

    def place_grid(self, market: str, orderbook: dict, cash_krw: Decimal, coin_qty: Decimal, decision: dict) -> None:
        effective = decision.get("effective_params") or self.strategy.effective_params(decision)
        step = effective["grid_step_pct"]
        levels = effective["grid_levels"]
        bids, asks = self.book_prices(orderbook)
        if not bids and not asks:
            return
        fallback_bid = bids[-1] if bids else (asks[0] if asks else Decimal("0"))
        fallback_ask = asks[-1] if asks else (bids[0] if bids else Decimal("0"))
        for level in range(1, levels + 1):
            buy_price = bids[level - 1] if bids and level <= len(bids) else quantize_down(fallback_bid * (Decimal("1") - step * max(1, level - len(bids))), 0)
            sell_price = asks[level - 1] if asks and level <= len(asks) else quantize_down(fallback_ask * (Decimal("1") + step * max(1, level - len(asks))), 0)
            buy_budget = cash_krw * Decimal("0.02")
            if buy_budget >= self.cfg.exchange_min_trade_krw and buy_price > 0:
                self.submit_buy(market, buy_price, buy_budget, f"grid-buy-L{level}")
            sell_qty = coin_qty * Decimal("0.05")
            if sell_qty > 0 and sell_price > 0:
                self.submit_sell(market, sell_price, sell_qty, f"grid-sell-L{level}")

    def build_report_summary(self, snapshot: dict) -> dict:
        cutoff = now_kst() - timedelta(seconds=self.cfg.report_interval_seconds)
        trades = self.db.get_recent_trades(cutoff)
        first = self.db.get_first_snapshot("coinone")
        old = self.db.get_snapshot_before("coinone", cutoff)
        current = snapshot["total_equity_krw"]
        initial = Decimal(str(first["total_equity_krw"])) if first else current
        previous = Decimal(str(old["total_equity_krw"])) if old else current
        return {
            "generated_at": now_kst().isoformat(),
            "exchange": {
                "total_equity_krw": float(current),
                "cash_krw": float(snapshot["cash_krw"]),
                "invested_krw": float(snapshot["invested_krw"]),
                "return_vs_initial_pct": float(((current - initial) / initial * 100) if initial > 0 else Decimal("0")),
                "return_vs_3h_pct": float(((current - previous) / previous * 100) if previous > 0 else Decimal("0")),
            },
            "recent_trades": [
                {"ts": row["ts"], "exchange": row["exchange"], "market": row["market"], "side": row["side"], "price": row["price"], "qty": row["qty"], "status": row["status"], "reason": row["reason"]}
                for row in trades
            ],
        }

    def render_report(self, summary: dict) -> str:
        info = summary["exchange"]
        lines = [
            f"AI 코인 투자 리포트 ({summary['generated_at']})",
            f"거래소: coinone",
            f"제외 자산: {', '.join(self.cfg.excluded_symbols)}",
            f"사용 모델: {', '.join(self.cfg.ollama_models)}",
            "",
            f"총 평가금액: {info['total_equity_krw']:,.0f} KRW",
            f"현금성 자산: {info['cash_krw']:,.0f} KRW",
            f"투자 자산: {info['invested_krw']:,.0f} KRW",
            f"초기 금액 대비 수익률: {info['return_vs_initial_pct']:.2f}%",
            f"3시간 전 대비 수익률: {info['return_vs_3h_pct']:.2f}%",
            "",
            "[최근 3시간 거래 내역]",
        ]
        if not summary["recent_trades"]:
            lines.append("- 거래 없음")
        else:
            for t in summary["recent_trades"][:50]:
                lines.append(f"- {t['ts']} | {t['market']} | {t['side']} | price={t['price']} | qty={t['qty']} | {t['status']} | {t['reason']}")
        return "\n".join(lines)

    def handle_market(self, market: str, sentiment: dict, onchain: dict, cycle_state: dict, stats: dict) -> None:
        _, base = split_market(market)
        if self.is_trading_blocked_symbol(base):
            return
        pos = cycle_state["positions"].get(base, {"qty": Decimal("0"), "avg_buy": Decimal("0")})
        cash_krw = cycle_state["cash_krw"]
        qty = pos["qty"]
        avg_buy = pos["avg_buy"]
        equity = cycle_state["equity_krw"]
        has_position = qty > 0
        if not has_position:
            can_open, buy_power = self.can_open_new_position_without_ai(cash_krw, equity)
            if not can_open:
                stats["buy_skipped_no_cash"] = stats.get("buy_skipped_no_cash", 0) + 1
                log(
                    f"[SKIP MARKET] {market} insufficient orderable KRW before AI "
                    f"budget={buy_power['budget_krw']} max_openable={buy_power['max_openable_krw']} min_order={buy_power['min_order_krw']}"
                )
                return
        candles = self.coinone.get_candles(market, interval="1h", size=200)
        if candles.empty or len(candles) < 60:
            stats["market_skipped_short_history"] = stats.get("market_skipped_short_history", 0) + 1
            log(f"[SKIP MARKET] {market} insufficient candle data rows={len(candles)} required=60")
            return
        technical = self.strategy.indicators(candles)
        orderbook = self.coinone.get_orderbook(market)
        best_bid, best_ask = self.book_prices(orderbook)
        best_bid_v = float(best_bid[0]) if best_bid else 0.0
        buy_limit = self.select_limit_price("BUY", orderbook, self.cfg.buy_book_level)
        sell_limit = self.select_limit_price("SELL", orderbook, self.cfg.sell_book_level)
        mark_price = self.mark_price_from_orderbook(orderbook)
        context = self.strategy.context_key(technical, sentiment, onchain)
        rl_hint = self.rl.choose(context)
        decision = self.strategy.ai_decision(market, technical, sentiment, onchain, rl_hint, qty)
        effective = decision["effective_params"]
        entry_price = mark_price if mark_price > 0 else (buy_limit if buy_limit > 0 else sell_limit)
        decision_id = self.db.log_ai_decision(
            exchange="coinone",
            market=market,
            context=context,
            model_name=",".join(self.cfg.ollama_models),
            rl_hint=rl_hint,
            prompt_payload={
                "market": market,
                "position_qty": qty,
                "technical": technical,
                "sentiment": sentiment,
                "onchain": onchain,
                "rl_hint": rl_hint,
            },
            decision=decision,
            position_qty=qty,
            entry_price=entry_price,
            entry_equity_krw=equity,
            reward_horizon_minutes=self.cfg.decision_reward_horizon_minutes,
        )
        log(f"[DECISION] {market} action={decision['action']} conf={decision['confidence']:.2f} reason={decision['reason']} raw_reason={decision.get('reason_raw','')} enabled={decision['enabled_signals']} overrides={decision['parameter_overrides']}")
        if has_position and avg_buy > 0 and best_bid_v <= float(avg_buy * (Decimal("1") - effective["stop_loss_pct"])):
            ok = self.submit_sell(market, sell_limit, qty, "stop-loss")
            self.db.update_ai_decision_execution(
                decision_id,
                "RISK_OVERRIDE_STOP_LOSS_SUBMITTED" if ok else "RISK_OVERRIDE_STOP_LOSS_SKIPPED",
                {"risk_rule": "stop_loss", "avg_buy": float(avg_buy), "best_bid": best_bid_v},
            )
            if ok:
                cycle_state.update(self.refresh_cycle_state())
                log(f"[POST TRADE REFRESH] {market} side=SELL cash={cycle_state['cash_krw']}")
            else:
                stats["sell_dust"] += 1
            return
        if has_position and avg_buy > 0 and best_bid_v >= float(avg_buy * (Decimal("1") + effective["take_profit_pct"])):
            partial_qty = qty * self.cfg.take_profit_close_fraction
            partial_ok = self.submit_sell(market, sell_limit, partial_qty, "take-profit-partial")
            if partial_ok:
                cycle_state.update(self.refresh_cycle_state())
                log(f"[POST TRADE REFRESH] {market} side=SELL cash={cycle_state['cash_krw']}")
            else:
                stats["sell_dust"] += 1
        target_notional = equity * self.cfg.max_exposure_per_market * Decimal(str(decision["target_risk"])) * effective["max_exposure_multiplier"]
        if decision["action"] == "BUY":
            stats["buy_signal"] += 1
            ok = self.rebalance_buy(market, mark_price or buy_limit, buy_limit, qty, target_notional, cash_krw, decision, stats)
            self.db.update_ai_decision_execution(
                decision_id,
                "BUY_SUBMITTED" if ok else "BUY_SKIPPED",
                {"target_notional": float(target_notional), "cash_krw": float(cash_krw), "entry_price": float(entry_price)},
            )
            if ok:
                cycle_state.update(self.refresh_cycle_state())
                log(f"[POST TRADE REFRESH] {market} side=BUY cash={cycle_state['cash_krw']}")
            return
        if decision["action"] == "SELL":
            if has_position:
                ok = self.submit_sell(market, sell_limit, qty, decision["reason"])
                self.db.update_ai_decision_execution(
                    decision_id,
                    "SELL_SUBMITTED" if ok else "SELL_SKIPPED",
                    {"qty": float(qty), "sell_limit": float(sell_limit), "entry_price": float(entry_price)},
                )
                if ok:
                    cycle_state.update(self.refresh_cycle_state())
                    log(f"[POST TRADE REFRESH] {market} side=SELL cash={cycle_state['cash_krw']}")
                else:
                    stats["sell_dust"] += 1
            else:
                stats["sell_unheld"] += 1
                self.db.update_ai_decision_execution(
                    decision_id,
                    "SELL_UNHELD",
                    {"qty": float(qty), "has_position": False},
                )
                if not self.cfg.suppress_unheld_sell_logs:
                    log(f"[SKIP SELL] binance_futures {market} no position")
            return
        stats["buy_nonbuy_action"] += 1
        if self.cfg.strong_hold_buy_enabled and not has_position:
            strong_hold = decision["confidence"] >= self.cfg.strong_hold_buy_confidence and (technical.get("trend_up") or technical.get("rebound_signal"))
            if strong_hold:
                synthetic = dict(decision)
                synthetic["reason"] = f"{decision['reason']} | strong-hold-buy"
                ok = self.rebalance_buy(market, mark_price or buy_limit, buy_limit, qty, target_notional, cash_krw, synthetic, stats)
                self.db.update_ai_decision_execution(
                    decision_id,
                    "STRONG_HOLD_BUY_SUBMITTED" if ok else "STRONG_HOLD_BUY_SKIPPED",
                    {"target_notional": float(target_notional), "cash_krw": float(cash_krw), "entry_price": float(entry_price)},
                )
                if ok:
                    cycle_state.update(self.refresh_cycle_state())
                    log(f"[POST TRADE REFRESH] {market} side=BUY cash={cycle_state['cash_krw']}")
                return
        if has_position and decision["use_grid"] and decision["enabled_signals"].get("technical", True):
            self.place_grid(market, orderbook, cash_krw, qty, decision)
            self.db.update_ai_decision_execution(
                decision_id,
                "HOLD_GRID_PLACED",
                {"cash_krw": float(cash_krw), "qty": float(qty), "entry_price": float(entry_price)},
            )
        elif (not has_position) and decision["use_rebound"] and decision["enabled_signals"].get("rebound", True):
            ok = self.rebound_buy(market, buy_limit, cash_krw, decision)
            self.db.update_ai_decision_execution(
                decision_id,
                "HOLD_REBOUND_BUY_SUBMITTED" if ok else "HOLD_REBOUND_BUY_SKIPPED",
                {"cash_krw": float(cash_krw), "buy_limit": float(buy_limit), "entry_price": float(entry_price)},
            )
        else:
            self.db.update_ai_decision_execution(
                decision_id,
                "NO_ORDER",
                {"has_position": has_position, "entry_price": float(entry_price)},
            )

    def run_cycle(self) -> None:
        self.refresh_target_markets(force=False)
        self.sync_pending_orders()
        self.evaluate_due_ai_decisions()
        sentiment = self.sentiment.score()
        onchain = self.onchain.combined()
        cycle_state = self.refresh_cycle_state()
        markets = self.select_cycle_markets()
        stats = {"sell_unheld": 0, "sell_dust": 0, "buy_signal": 0, "buy_submitted": 0, "buy_below_min": 0, "buy_low_conf": 0, "buy_nonbuy_action": 0, "buy_skipped_no_cash": 0, "market_skipped_short_history": 0}
        log(f"[CYCLE START] sentiment={sentiment['summary']} onchain={onchain['combined_score']:.3f} markets={len(markets)}/{len(self.cfg.target_markets)}")
        for idx, market in enumerate(markets, start=1):
            try:
                if idx > 1 and idx % self.cfg.balance_refresh_every_markets == 0:
                    cycle_state.update(self.refresh_cycle_state())
                    log(f"[MID CYCLE REFRESH] idx={idx} cash={cycle_state['cash_krw']} positions={len(cycle_state['positions'])}")
                if idx > 1 and idx % self.cfg.macro_refresh_every_markets == 0:
                    sentiment = self.sentiment.score()
                    onchain = self.onchain.combined()
                    log(f"[MID CYCLE MACRO REFRESH] idx={idx} onchain={onchain['combined_score']:.3f}")
                self.handle_market(market, sentiment, onchain, cycle_state, stats)
            except Exception as exc:
                log(f"[MARKET ERROR] {market}: {type(exc).__name__}: {exc}")
            finally:
                self.maybe_sync_pending_orders()
        self.maybe_sync_pending_orders(force=True)
        snapshot = self.capture_snapshot(cycle_state["balances"])
        log("[CYCLE SUMMARY] "
              f"sell_unheld_suppressed={stats['sell_unheld']} sell_dust_skipped={stats['sell_dust']} "
              f"buy_signals={stats['buy_signal']} buy_submitted={stats['buy_submitted']} "
              f"buy_below_min={stats['buy_below_min']} buy_low_conf={stats['buy_low_conf']} "
              f"buy_skipped_no_cash={stats['buy_skipped_no_cash']} nonbuy_decisions={stats['buy_nonbuy_action']} "
              f"short_history_skipped={stats['market_skipped_short_history']}")
        if time.time() - self.last_report_at >= self.cfg.report_interval_seconds:
            summary = self.build_report_summary(snapshot)
            self.mailer.send(f"[AI COIN BOT] 3시간 투자 리포트 {now_kst().strftime('%Y-%m-%d %H:%M')}", self.render_report(summary))
            self.db.save_report(summary)
            self.last_report_at = time.time()
            log("[REPORT] sent")

    def run_forever(self) -> None:
        self.refresh_target_markets(force=True)
        log(f"[START] exchange=coinone DRY_RUN={self.cfg.dry_run} markets={self.cfg.target_markets} excluded={self.cfg.excluded_symbols} stable_trading={self.cfg.allow_stablecoin_trading} models={self.cfg.ollama_models}")
        while True:
            started = time.time()
            try:
                self.run_cycle()
            except Exception as exc:
                log(f"[ERROR] {type(exc).__name__}: {exc}")
            time.sleep(max(5, self.cfg.loop_interval_seconds - int(time.time() - started)))




class BinanceFuturesAIBot(CoinoneOnlyAIBot):
    exchange_name = "binance_futures"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.db = BotDB(cfg.db_path)
        self.ollama = OllamaClient(cfg.ollama_url, cfg.ollama_models, timeout=cfg.request_timeout * 2, ensemble_enabled=cfg.ollama_ensemble_enabled)
        self.coinone = BinanceFuturesClient(
            api_key=cfg.coinone_access_token,
            secret_key=cfg.coinone_secret_key,
            base_url=cfg.binance_futures_base_url,
            timeout=cfg.request_timeout,
            recv_window=cfg.binance_recv_window,
            default_leverage=cfg.binance_leverage,
            default_margin_type=cfg.binance_margin_type,
            hedge_mode=cfg.binance_hedge_mode,
        )
        self.sentiment = SentimentProvider(self.ollama, timeout=cfg.request_timeout)
        self.onchain = OnChainProvider(cfg.etherscan_api_key, timeout=cfg.request_timeout)
        self.strategy = BinanceStrategyEngine(cfg, self.ollama)
        self.rl = BanditRLPolicy(self.db)
        self.mailer = Mailer(cfg)
        self.last_report_at = 0.0
        self.last_market_refresh_at = 0.0
        self.last_pending_sync_at = 0.0
        self.cycle_market_cursor = 0
        self.position_peak_state: Dict[str, Dict[str, Decimal]] = {}

    def _binance_open_order_kwargs(self, side: str) -> dict:
        if not self.cfg.binance_hedge_mode:
            return {}
        return {"position_side": "LONG" if side.upper() == "BUY" else "SHORT"}

    def _binance_close_order_kwargs(self, close_side: str) -> dict:
        if self.cfg.binance_hedge_mode:
            return {"position_side": "LONG" if close_side.upper() == "SELL" else "SHORT"}
        return {"reduce_only": True}

    def _is_urgent_close_reason(self, reason: str) -> bool:
        reason_lower = str(reason or "").lower()
        return any(token in reason_lower for token in ("stop-loss", "close-long", "close-short", "take-profit", "trailing", "risk_override", "rescue"))

    def _close_timeout_override_seconds(self) -> int:
        return min(max(3, self.cfg.order_poll_interval_seconds * 3), max(3, self.cfg.order_timeout_seconds // 3))

    def _position_state_key(self, market: str, direction: int) -> str:
        return f"{market}:{'LONG' if direction > 0 else 'SHORT'}"

    def _reset_position_peak_state(self, market: str) -> None:
        self.position_peak_state.pop(f"{market}:LONG", None)
        self.position_peak_state.pop(f"{market}:SHORT", None)

    def _position_pnl_pct(self, direction: int, avg_buy: Decimal, best_bid: Decimal, best_ask: Decimal, mark_price: Decimal) -> Decimal:
        if direction == 0 or avg_buy <= 0:
            return Decimal("0")
        exit_price = best_bid if direction > 0 else best_ask
        if exit_price <= 0:
            exit_price = mark_price
        if exit_price <= 0:
            return Decimal("0")
        return (exit_price - avg_buy) / avg_buy if direction > 0 else (avg_buy - exit_price) / avg_buy

    def _record_position_peak(self, market: str, direction: int, avg_buy: Decimal, qty: Decimal, pnl_pct: Decimal) -> dict:
        key = self._position_state_key(market, direction)
        state = self.position_peak_state.get(key)
        if state is None or to_decimal(state.get("entry_price") or 0) != avg_buy or int(state.get("direction") or 0) != direction:
            state = {
                "direction": direction,
                "entry_price": avg_buy,
                "qty": abs(qty),
                "peak_pnl_pct": pnl_pct,
                "last_pnl_pct": pnl_pct,
            }
        else:
            state["qty"] = abs(qty)
            state["last_pnl_pct"] = pnl_pct
            state["peak_pnl_pct"] = max(to_decimal(state.get("peak_pnl_pct") or pnl_pct), pnl_pct)
        self.position_peak_state[key] = state
        stale_key = self._position_state_key(market, -direction)
        self.position_peak_state.pop(stale_key, None)
        return state

    def _market_min_notional(self, market: str) -> Decimal:
        try:
            _, _, min_notional = self.coinone.normalize_price_qty(market, Decimal("1"), Decimal("1"))
            return max(self.cfg.exchange_min_trade_krw, min_notional)
        except Exception:
            return self.cfg.exchange_min_trade_krw

    def _aggressive_exit_offset_pct(self) -> Decimal:
        return max(Decimal("0"), self.cfg.emergency_exit_price_offset_pct)

    def _close_position_with_fallback(self, market: str, direction: int, qty: Decimal, base_reason: str, buy_limit: Decimal, sell_limit: Decimal, mark_price: Decimal, best_bid: Decimal, best_ask: Decimal, avg_buy: Decimal, cash_krw: Decimal, equity_krw: Decimal, invested_krw: Decimal, decision: dict, cycle_state: dict, stats: dict, allow_rescue: bool = False, aggressive: bool = False) -> bool:
        qty = abs(quantize_down(qty, 8))
        if qty <= 0:
            return False
        price_adjustment_pct = self._aggressive_exit_offset_pct() if aggressive else Decimal("0")
        close_price = sell_limit if direction > 0 else buy_limit
        direct_ok = (
            self.close_long(market, close_price, qty, base_reason, allow_below_min_notional=True, price_adjustment_pct=price_adjustment_pct)
            if direction > 0
            else self.close_short(market, close_price, qty, base_reason, allow_below_min_notional=True, price_adjustment_pct=price_adjustment_pct)
        )
        if direct_ok:
            self._reset_position_peak_state(market)
            cycle_state.update(self.refresh_cycle_state())
            return True

        ref_price = best_bid if direction > 0 else best_ask
        if ref_price <= 0:
            ref_price = mark_price
        current_notional = abs(qty) * ref_price if ref_price > 0 else Decimal("0")
        min_notional = self._market_min_notional(market)
        pnl_pct = self._position_pnl_pct(direction, avg_buy, best_bid, best_ask, mark_price)
        rescue_trigger = max(self.cfg.stop_loss_pct, self.cfg.rescue_addon_trigger_pct)
        if (not allow_rescue) or current_notional >= min_notional or pnl_pct > -rescue_trigger:
            return False

        effective = decision.get("effective_params") or self.strategy.effective_params(decision)
        remaining_total_cap = self.remaining_total_exposure_capacity_krw(equity_krw, invested_krw)
        budget_cap = cash_krw * effective["buy_budget_fraction"] * self.cfg.buy_spend_safety_buffer * Decimal(str(self.cfg.binance_leverage))
        addon_target_total = max(min_notional * (Decimal("1") + self.cfg.rescue_min_notional_buffer_pct), current_notional)
        addon_spend = max(min_notional, addon_target_total - current_notional)
        addon_spend = quantize_down(addon_spend, 8)
        allowed_spend = min(budget_cap, remaining_total_cap)
        if addon_spend <= 0 or addon_spend > allowed_spend:
            log(f"[RESCUE SKIP] {market} direction={direction} addon_spend={addon_spend} allowed={allowed_spend} pnl_pct={float(pnl_pct):.4f} current_notional={current_notional} min_notional={min_notional}")
            return False

        stats["rescue_attempted"] = stats.get("rescue_attempted", 0) + 1
        if direction > 0:
            rescue_ok = self.submit_buy(market, buy_limit if buy_limit > 0 else mark_price, addon_spend, f"{base_reason} | rescue-addon-buy")
        else:
            sell_price = sell_limit if sell_limit > 0 else mark_price
            rescue_qty = quantize_down(addon_spend / sell_price, 8) if sell_price > 0 else Decimal("0")
            rescue_ok = rescue_qty > 0 and self.submit_sell(market, sell_price, rescue_qty, f"{base_reason} | rescue-addon-short")
        if not rescue_ok:
            return False

        stats["rescue_submitted"] = stats.get("rescue_submitted", 0) + 1
        cycle_state.update(self.refresh_cycle_state())
        _, base = split_market(market)
        refreshed_pos = cycle_state["positions"].get(base, {})
        refreshed_qty = abs(to_decimal(refreshed_pos.get("qty") or 0))
        if refreshed_qty <= 0:
            self._reset_position_peak_state(market)
            return True
        refreshed_orderbook = self.coinone.get_orderbook(market)
        refreshed_buy_limit = self.select_limit_price("BUY", refreshed_orderbook, self.cfg.buy_book_level)
        refreshed_sell_limit = self.select_limit_price("SELL", refreshed_orderbook, self.cfg.sell_book_level)
        final_ok = (
            self.close_long(market, refreshed_sell_limit, refreshed_qty, f"{base_reason} | rescue-close", allow_below_min_notional=True, price_adjustment_pct=self._aggressive_exit_offset_pct())
            if direction > 0
            else self.close_short(market, refreshed_buy_limit, refreshed_qty, f"{base_reason} | rescue-close", allow_below_min_notional=True, price_adjustment_pct=self._aggressive_exit_offset_pct())
        )
        if final_ok:
            stats["rescue_close_submitted"] = stats.get("rescue_close_submitted", 0) + 1
            self._reset_position_peak_state(market)
            cycle_state.update(self.refresh_cycle_state())
        return final_ok

    def order_terminal_statuses(self) -> set[str]:
        return {"FILLED", "CANCELED", "EXPIRED", "EXPIRED_IN_MATCH", "REJECTED"}

    def order_live_statuses(self) -> set[str]:
        return {"NEW", "PARTIALLY_FILLED", "PENDING_CANCEL", "SUBMITTED", "CANCEL_REQUESTED"}

    def build_position_map(self, balances: Any) -> Tuple[Decimal, Dict[str, Dict[str, Decimal]]]:
        account = balances if isinstance(balances, dict) else self.coinone.get_balances()
        cash = to_decimal(account.get("availableBalance") or 0)
        positions: Dict[str, Dict[str, Decimal]] = {}
        excluded = {s.upper() for s in self.cfg.excluded_symbols}
        for item in account.get("positions", []) or []:
            symbol = str(item.get("symbol") or "").upper()
            if not symbol:
                continue
            market = symbol_to_market(symbol, self.cfg.market_quote_currency)
            _, base = split_market(market)
            if base in excluded:
                continue
            qty = quantize_down(to_decimal(item.get("positionAmt") or 0), 8)
            if qty == 0:
                continue
            positions[base] = {
                "qty": qty,
                "avg_buy": to_decimal(item.get("entryPrice") or 0),
                "mark_price": to_decimal(item.get("markPrice") or 0),
                "notional": to_decimal(item.get("notional") or 0),
            }
        return cash, positions

    def estimate_equity(self, balances: Any) -> Tuple[Decimal, Decimal, Decimal, dict]:
        account = balances if isinstance(balances, dict) else self.coinone.get_balances()
        total = to_decimal(account.get("totalMarginBalance") or account.get("totalWalletBalance") or 0)
        cash = to_decimal(account.get("availableBalance") or 0)
        invested = Decimal("0")
        positions = []
        for item in account.get("positions", []) or []:
            qty = to_decimal(item.get("positionAmt") or 0)
            if qty == 0:
                continue
            notional = abs(to_decimal(item.get("notional") or 0))
            invested += notional
            positions.append({
                "symbol": item.get("symbol"),
                "positionAmt": float(qty),
                "entryPrice": float(to_decimal(item.get("entryPrice") or 0)),
                "markPrice": float(to_decimal(item.get("markPrice") or 0)),
                "notional": float(notional),
                "unRealizedProfit": float(to_decimal(item.get("unRealizedProfit") or 0)),
            })
        raw = {
            "assets": account.get("assets", []),
            "positions": positions,
            "totalMarginBalance": float(total),
            "availableBalance": float(cash),
        }
        return total, cash, invested, raw

    def capture_snapshot(self, balances: Any) -> dict:
        total, cash, invested, raw = self.estimate_equity(balances)
        self.db.save_snapshot(self.exchange_name, total, cash, invested, raw)
        return {"exchange": self.exchange_name, "total_equity_krw": total, "cash_krw": cash, "invested_krw": invested}

    def refresh_cycle_state(self) -> dict:
        balances = self.coinone.get_balances()
        cash, positions = self.build_position_map(balances)
        equity, _, invested, _ = self.estimate_equity(balances)
        return {"balances": balances, "cash_krw": cash, "positions": positions, "equity_krw": equity, "invested_krw": invested}

    def total_exposure_limit_krw(self, equity_krw: Decimal) -> Decimal:
        return max(Decimal("0"), equity_krw * self.cfg.max_total_exposure * Decimal(str(self.cfg.binance_leverage)))

    def remaining_total_exposure_capacity_krw(self, equity_krw: Decimal, invested_krw: Decimal) -> Decimal:
        return max(Decimal("0"), self.total_exposure_limit_krw(equity_krw) - max(Decimal("0"), invested_krw))

    def can_open_new_position_without_ai(self, cash_krw: Decimal, equity_krw: Decimal, invested_krw: Decimal = Decimal("0")) -> Tuple[bool, dict]:
        min_order_krw = self.cfg.exchange_min_trade_krw
        budget_krw = self.available_buy_budget_krw(cash_krw) * Decimal(str(self.cfg.binance_leverage))
        max_risk_target_krw = equity_krw * self.cfg.max_exposure_per_market * Decimal(str(self.cfg.binance_leverage))
        total_exposure_limit_krw = self.total_exposure_limit_krw(equity_krw)
        remaining_total_cap_krw = self.remaining_total_exposure_capacity_krw(equity_krw, invested_krw)
        max_openable_krw = min(budget_krw, max_risk_target_krw, remaining_total_cap_krw)
        return max_openable_krw >= min_order_krw, {
            "budget_krw": budget_krw,
            "max_risk_target_krw": max_risk_target_krw,
            "total_exposure_limit_krw": total_exposure_limit_krw,
            "remaining_total_cap_krw": remaining_total_cap_krw,
            "max_openable_krw": max_openable_krw,
            "min_order_krw": min_order_krw,
        }

    def submit_buy(self, market: str, price: Decimal, spend_krw: Decimal, reason: str) -> bool:
        _, base = split_market(market)
        if self.is_trading_blocked_symbol(base):
            return False
        qty = quantize_down(spend_krw / price, 8) if price > 0 else Decimal("0")
        if spend_krw < self.cfg.exchange_min_trade_krw:
            log(f"[SKIP BUY] binance_futures {market} below min spend={spend_krw}")
            return False
        if qty <= 0:
            log(f"[SKIP BUY] binance_futures {market} invalid qty at price={price}")
            return False
        normalized = self._normalize_futures_order(market, "BUY", price, qty)
        if not normalized:
            return False
        norm_price, norm_qty, notional = normalized
        return self.execute_limit_order_with_retry(market, "BUY", norm_price, norm_qty, notional, reason, order_kwargs=self._binance_open_order_kwargs("BUY"))

    def submit_sell(self, market: str, price: Decimal, qty: Decimal, reason: str) -> bool:
        _, base = split_market(market)
        if self.is_trading_blocked_symbol(base):
            return False
        qty = quantize_down(abs(qty), 8)
        if qty <= 0:
            return False
        normalized = self._normalize_futures_order(market, "SELL", price, qty)
        if not normalized:
            return False
        norm_price, norm_qty, notional = normalized
        return self.execute_limit_order_with_retry(market, "SELL", norm_price, norm_qty, notional, reason, order_kwargs=self._binance_open_order_kwargs("SELL"))

    def close_long(self, market: str, price: Decimal, qty: Decimal, reason: str, allow_below_min_notional: bool = False, price_adjustment_pct: Decimal = Decimal("0")) -> bool:
        qty = quantize_down(abs(qty), 8)
        if qty <= 0:
            return False
        normalized = self._normalize_futures_order(market, "SELL", price, qty, allow_below_min_notional=allow_below_min_notional)
        if not normalized:
            return False
        norm_price, norm_qty, notional = normalized
        order_kwargs = self._binance_close_order_kwargs("SELL")
        timeout_override = None
        if self._is_urgent_close_reason(reason):
            order_kwargs["time_in_force"] = "IOC"
            timeout_override = self._close_timeout_override_seconds()
        return self.execute_limit_order_with_retry(market, "SELL", norm_price, norm_qty, notional, reason, order_kwargs=order_kwargs, timeout_override_seconds=timeout_override, allow_below_min_notional=allow_below_min_notional, price_adjustment_pct=price_adjustment_pct)

    def close_short(self, market: str, price: Decimal, qty: Decimal, reason: str, allow_below_min_notional: bool = False, price_adjustment_pct: Decimal = Decimal("0")) -> bool:
        qty = quantize_down(abs(qty), 8)
        if qty <= 0:
            return False
        normalized = self._normalize_futures_order(market, "BUY", price, qty, allow_below_min_notional=allow_below_min_notional)
        if not normalized:
            return False
        norm_price, norm_qty, notional = normalized
        order_kwargs = self._binance_close_order_kwargs("BUY")
        timeout_override = None
        if self._is_urgent_close_reason(reason):
            order_kwargs["time_in_force"] = "IOC"
            timeout_override = self._close_timeout_override_seconds()
        return self.execute_limit_order_with_retry(market, "BUY", norm_price, norm_qty, notional, reason, order_kwargs=order_kwargs, timeout_override_seconds=timeout_override, allow_below_min_notional=allow_below_min_notional, price_adjustment_pct=price_adjustment_pct)

    def rebalance_buy(self, market: str, mark_price: Decimal, buy_limit_price: Decimal, current_qty: Decimal, target_notional: Decimal, cash_krw: Decimal, equity_krw: Decimal, invested_krw: Decimal, decision: dict, stats: dict) -> bool:
        effective = decision.get("effective_params") or self.strategy.effective_params(decision)
        if current_qty == 0 and decision["confidence"] < self.cfg.buy_confidence_threshold:
            stats["buy_low_conf"] += 1
            log(f"[SKIP BUY] binance_futures {market} low confidence={decision['confidence']:.2f}")
            return False
        current_notional = abs(current_qty) * (mark_price if mark_price > 0 else buy_limit_price) if current_qty > 0 else Decimal("0")
        desired_target = target_notional if current_qty > 0 else max(target_notional, self.cfg.min_new_position_krw)
        gap = max(Decimal("0"), desired_target - current_notional)
        budget_cap = cash_krw * effective["buy_budget_fraction"] * self.cfg.buy_spend_safety_buffer * Decimal(str(self.cfg.binance_leverage))
        remaining_total_cap = self.remaining_total_exposure_capacity_krw(equity_krw, invested_krw)
        spend = min(gap, budget_cap, remaining_total_cap)
        if current_qty <= 0:
            spend = max(spend, min(self.cfg.min_new_position_krw, budget_cap, remaining_total_cap))
        if remaining_total_cap < self.cfg.exchange_min_trade_krw:
            stats["buy_skipped_no_cash"] += 1
            log(f"[SKIP BUY] binance_futures {market} total exposure cap reached invested={invested_krw} limit={self.total_exposure_limit_krw(equity_krw)}")
            return False
        if spend < self.cfg.exchange_min_trade_krw:
            stats["buy_below_min"] += 1
            log(f"[SKIP BUY] binance_futures {market} below min spend={spend} target={desired_target} cash={cash_krw} remaining_total_cap={remaining_total_cap}")
            return False
        ok = self.submit_buy(market, buy_limit_price, min(spend, budget_cap, remaining_total_cap), decision["reason"])
        if ok:
            stats["buy_submitted"] += 1
        return ok

    def rebalance_short(self, market: str, mark_price: Decimal, sell_limit_price: Decimal, current_qty: Decimal, target_notional: Decimal, cash_krw: Decimal, equity_krw: Decimal, invested_krw: Decimal, decision: dict, stats: dict) -> bool:
        effective = decision.get("effective_params") or self.strategy.effective_params(decision)
        if current_qty == 0 and decision["confidence"] < self.cfg.buy_confidence_threshold:
            stats["buy_low_conf"] += 1
            log(f"[SKIP SHORT] binance_futures {market} low confidence={decision['confidence']:.2f}")
            return False
        current_notional = abs(current_qty) * (mark_price if mark_price > 0 else sell_limit_price) if current_qty < 0 else Decimal("0")
        desired_target = target_notional if current_qty < 0 else max(target_notional, self.cfg.min_new_position_krw)
        gap = max(Decimal("0"), desired_target - current_notional)
        budget_cap = cash_krw * effective["buy_budget_fraction"] * self.cfg.buy_spend_safety_buffer * Decimal(str(self.cfg.binance_leverage))
        remaining_total_cap = self.remaining_total_exposure_capacity_krw(equity_krw, invested_krw)
        spend = min(gap, budget_cap, remaining_total_cap)
        if current_qty >= 0:
            spend = max(spend, min(self.cfg.min_new_position_krw, budget_cap, remaining_total_cap))
        if remaining_total_cap < self.cfg.exchange_min_trade_krw:
            stats["buy_skipped_no_cash"] += 1
            log(f"[SKIP SHORT] binance_futures {market} total exposure cap reached invested={invested_krw} limit={self.total_exposure_limit_krw(equity_krw)}")
            return False
        if spend < self.cfg.exchange_min_trade_krw:
            stats["buy_below_min"] += 1
            log(f"[SKIP SHORT] binance_futures {market} below min spend={spend} target={desired_target} cash={cash_krw} remaining_total_cap={remaining_total_cap}")
            return False
        qty = quantize_down(min(spend, budget_cap, remaining_total_cap) / sell_limit_price, 8) if sell_limit_price > 0 else Decimal("0")
        if qty <= 0:
            return False
        ok = self.submit_sell(market, sell_limit_price, qty, f"{decision['reason']} | open-short")
        if ok:
            stats["buy_submitted"] += 1
        return ok

    def rebound_buy(self, market: str, buy_limit_price: Decimal, cash_krw: Decimal, equity_krw: Decimal, invested_krw: Decimal, decision: dict) -> bool:
        effective = decision.get("effective_params") or self.strategy.effective_params(decision)
        remaining_total_cap = self.remaining_total_exposure_capacity_krw(equity_krw, invested_krw)
        spend = min(
            cash_krw * effective["rebound_buy_fraction"] * self.cfg.buy_spend_safety_buffer * Decimal(str(self.cfg.binance_leverage)),
            self.cfg.min_trade_krw * Decimal("2"),
            remaining_total_cap,
        )
        if spend < self.cfg.exchange_min_trade_krw:
            log(f"[SKIP REBOUND BUY] binance_futures {market} below min spend={spend} remaining_total_cap={remaining_total_cap}")
            return False
        return self.submit_buy(market, buy_limit_price, spend, "rebound")

    def place_grid(self, market: str, orderbook: dict, cash_krw: Decimal, coin_qty: Decimal, decision: dict) -> None:
        log(f"[GRID SKIPPED] futures mode market={market} current_qty={coin_qty}")

    def build_report_summary(self, snapshot: dict) -> dict:
        cutoff = now_kst() - timedelta(seconds=self.cfg.report_interval_seconds)
        trades = self.db.get_recent_trades(cutoff)
        first = self.db.get_first_snapshot(self.exchange_name)
        old = self.db.get_snapshot_before(self.exchange_name, cutoff)
        current = snapshot["total_equity_krw"]
        initial = Decimal(str(first["total_equity_krw"])) if first else current
        previous = Decimal(str(old["total_equity_krw"])) if old else current
        return {
            "generated_at": now_kst().isoformat(),
            "exchange": {
                "total_equity_krw": float(current),
                "cash_krw": float(snapshot["cash_krw"]),
                "invested_krw": float(snapshot["invested_krw"]),
                "return_vs_initial_pct": float(((current - initial) / initial * 100) if initial > 0 else Decimal("0")),
                "return_vs_3h_pct": float(((current - previous) / previous * 100) if previous > 0 else Decimal("0")),
            },
            "recent_trades": [
                {"ts": row["ts"], "exchange": row["exchange"], "market": row["market"], "side": row["side"], "price": row["price"], "qty": row["qty"], "status": row["status"], "reason": row["reason"]}
                for row in trades
            ],
        }

    def render_report(self, summary: dict) -> str:
        info = summary["exchange"]
        quote = self.cfg.market_quote_currency.upper()
        lines = [
            f"AI 바이낸스 선물 리포트 ({summary['generated_at']})",
            f"거래소: Binance USDⓈ-M Futures",
            f"기준 통화: {quote}",
            f"제외 자산: {', '.join(self.cfg.excluded_symbols)}",
            f"사용 모델: {', '.join(self.cfg.ollama_models)}",
            "",
            f"총 평가금액: {info['total_equity_krw']:,.4f} {quote}",
            f"가용 잔고: {info['cash_krw']:,.4f} {quote}",
            f"포지션 노출(절대합): {info['invested_krw']:,.4f} {quote}",
            f"초기 금액 대비 수익률: {info['return_vs_initial_pct']:.2f}%",
            f"3시간 전 대비 수익률: {info['return_vs_3h_pct']:.2f}%",
            "",
            "[최근 3시간 거래 내역]",
        ]
        if not summary["recent_trades"]:
            lines.append("- 거래 없음")
        else:
            for t in summary["recent_trades"][:50]:
                lines.append(f"- {t['ts']} | {t['market']} | {t['side']} | price={t['price']} | qty={t['qty']} | {t['status']} | {t['reason']}")
        return "\n".join(lines)

    def handle_market(self, market: str, sentiment: dict, onchain: dict, cycle_state: dict, stats: dict) -> None:
        _, base = split_market(market)
        if self.is_trading_blocked_symbol(base):
            return
        pos = cycle_state["positions"].get(base, {"qty": Decimal("0"), "avg_buy": Decimal("0")})
        cash_krw = cycle_state["cash_krw"]
        qty = to_decimal(pos.get("qty") or 0)
        avg_buy = to_decimal(pos.get("avg_buy") or 0)
        equity = cycle_state["equity_krw"]
        invested_krw = to_decimal(cycle_state.get("invested_krw") or 0)
        direction = 1 if qty > 0 else -1 if qty < 0 else 0
        has_position = direction != 0
        if not has_position:
            self._reset_position_peak_state(market)
            can_open, buy_power = self.can_open_new_position_without_ai(cash_krw, equity, invested_krw)
            if not can_open:
                stats["buy_skipped_no_cash"] = stats.get("buy_skipped_no_cash", 0) + 1
                log(
                    f"[SKIP MARKET] {market} insufficient orderable margin before AI "
                    f"budget={buy_power['budget_krw']} remaining_total_cap={buy_power.get('remaining_total_cap_krw')} max_openable={buy_power['max_openable_krw']} min_order={buy_power['min_order_krw']}"
                )
                return
        candles = self.coinone.get_candles(market, interval="1h", size=200)
        if candles.empty or len(candles) < 60:
            stats["market_skipped_short_history"] = stats.get("market_skipped_short_history", 0) + 1
            log(f"[SKIP MARKET] {market} insufficient candle data rows={len(candles)} required=60")
            return
        technical = self.strategy.indicators(candles)
        orderbook = self.coinone.get_orderbook(market)
        best_bid, best_ask = self.book_prices(orderbook)
        best_bid_price = best_bid[0] if best_bid else Decimal("0")
        best_ask_price = best_ask[0] if best_ask else Decimal("0")
        best_bid_v = float(best_bid_price) if best_bid_price > 0 else 0.0
        best_ask_v = float(best_ask_price) if best_ask_price > 0 else 0.0
        buy_limit = self.select_limit_price("BUY", orderbook, self.cfg.buy_book_level)
        sell_limit = self.select_limit_price("SELL", orderbook, self.cfg.sell_book_level)
        mark_price = self.mark_price_from_orderbook(orderbook)
        context = self.strategy.context_key(technical, sentiment, onchain)
        rl_hint = self.rl.choose(context)
        decision = self.strategy.ai_decision(market, technical, sentiment, onchain, rl_hint, qty)
        effective = decision["effective_params"]
        entry_price = mark_price if mark_price > 0 else (buy_limit if buy_limit > 0 else sell_limit)
        decision_id = self.db.log_ai_decision(
            exchange=self.exchange_name,
            market=market,
            context=context,
            model_name=",".join(self.cfg.ollama_models),
            rl_hint=rl_hint,
            prompt_payload={
                "market": market,
                "position_qty": float(qty),
                "technical": technical,
                "sentiment": sentiment,
                "onchain": onchain,
                "rl_hint": rl_hint,
            },
            decision=decision,
            position_qty=qty,
            entry_price=entry_price,
            entry_equity_krw=equity,
            reward_horizon_minutes=self.cfg.decision_reward_horizon_minutes,
        )
        log(f"[DECISION] {market} qty={qty} action={decision['action']} conf={decision['confidence']:.2f} reason={decision['reason']} raw_reason={decision.get('reason_raw','')} enabled={decision['enabled_signals']} overrides={decision['parameter_overrides']}")

        if has_position and avg_buy > 0:
            pnl_pct = self._position_pnl_pct(direction, avg_buy, best_bid_price, best_ask_price, mark_price)
            peak_state = self._record_position_peak(market, direction, avg_buy, qty, pnl_pct)
            peak_pnl_pct = to_decimal(peak_state.get("peak_pnl_pct") or pnl_pct)
            trailing_gap = peak_pnl_pct - pnl_pct
            if peak_pnl_pct >= effective["highest_take_profit_pct"] and trailing_gap >= effective["trailing_take_profit_gap_pct"]:
                ok = self._close_position_with_fallback(
                    market=market,
                    direction=direction,
                    qty=qty,
                    base_reason="take-profit-trailing-long" if direction > 0 else "take-profit-trailing-short",
                    buy_limit=buy_limit,
                    sell_limit=sell_limit,
                    mark_price=mark_price,
                    best_bid=best_bid_price,
                    best_ask=best_ask_price,
                    avg_buy=avg_buy,
                    cash_krw=cash_krw,
                    equity_krw=equity,
                    invested_krw=invested_krw,
                    decision=decision,
                    cycle_state=cycle_state,
                    stats=stats,
                    allow_rescue=False,
                    aggressive=True,
                )
                self.db.update_ai_decision_execution(
                    decision_id,
                    "RISK_OVERRIDE_TRAILING_TP_SUBMITTED" if ok else "RISK_OVERRIDE_TRAILING_TP_SKIPPED",
                    {
                        "risk_rule": "trailing_take_profit",
                        "avg_buy": float(avg_buy),
                        "pnl_pct": float(pnl_pct),
                        "peak_pnl_pct": float(peak_pnl_pct),
                        "trailing_gap_pct": float(trailing_gap),
                    },
                )
                if ok:
                    return

        if direction > 0 and avg_buy > 0 and best_bid_v <= float(avg_buy * (Decimal("1") - effective["stop_loss_pct"])):
            ok = self._close_position_with_fallback(
                market=market,
                direction=direction,
                qty=qty,
                base_reason="stop-loss-long",
                buy_limit=buy_limit,
                sell_limit=sell_limit,
                mark_price=mark_price,
                best_bid=best_bid_price,
                best_ask=best_ask_price,
                avg_buy=avg_buy,
                cash_krw=cash_krw,
                equity_krw=equity,
                invested_krw=invested_krw,
                decision=decision,
                cycle_state=cycle_state,
                stats=stats,
                allow_rescue=True,
                aggressive=True,
            )
            self.db.update_ai_decision_execution(decision_id, "RISK_OVERRIDE_STOP_LOSS_SUBMITTED" if ok else "RISK_OVERRIDE_STOP_LOSS_SKIPPED", {"risk_rule": "stop_loss_long", "avg_buy": float(avg_buy), "best_bid": best_bid_v})
            if ok:
                return
        if direction < 0 and avg_buy > 0 and best_ask_v >= float(avg_buy * (Decimal("1") + effective["stop_loss_pct"])):
            ok = self._close_position_with_fallback(
                market=market,
                direction=direction,
                qty=qty,
                base_reason="stop-loss-short",
                buy_limit=buy_limit,
                sell_limit=sell_limit,
                mark_price=mark_price,
                best_bid=best_bid_price,
                best_ask=best_ask_price,
                avg_buy=avg_buy,
                cash_krw=cash_krw,
                equity_krw=equity,
                invested_krw=invested_krw,
                decision=decision,
                cycle_state=cycle_state,
                stats=stats,
                allow_rescue=True,
                aggressive=True,
            )
            self.db.update_ai_decision_execution(decision_id, "RISK_OVERRIDE_STOP_LOSS_SUBMITTED" if ok else "RISK_OVERRIDE_STOP_LOSS_SKIPPED", {"risk_rule": "stop_loss_short", "avg_buy": float(avg_buy), "best_ask": best_ask_v})
            if ok:
                return
        if direction > 0 and avg_buy > 0 and best_bid_v >= float(avg_buy * (Decimal("1") + effective["take_profit_pct"])):
            partial_qty = abs(qty) * self.cfg.take_profit_close_fraction
            partial_ok = self.close_long(market, sell_limit, partial_qty, "take-profit-partial-long", allow_below_min_notional=True, price_adjustment_pct=self._aggressive_exit_offset_pct() / Decimal("2"))
            if partial_ok:
                cycle_state.update(self.refresh_cycle_state())
        if direction < 0 and avg_buy > 0 and best_ask_v <= float(avg_buy * (Decimal("1") - effective["take_profit_pct"])):
            partial_qty = abs(qty) * self.cfg.take_profit_close_fraction
            partial_ok = self.close_short(market, buy_limit, partial_qty, "take-profit-partial-short", allow_below_min_notional=True, price_adjustment_pct=self._aggressive_exit_offset_pct() / Decimal("2"))
            if partial_ok:
                cycle_state.update(self.refresh_cycle_state())

        target_notional = equity * self.cfg.max_exposure_per_market * Decimal(str(decision["target_risk"])) * effective["max_exposure_multiplier"] * Decimal(str(self.cfg.binance_leverage))
        if decision["action"] == "BUY":
            stats["buy_signal"] += 1
            if direction < 0:
                ok = self._close_position_with_fallback(
                    market=market,
                    direction=direction,
                    qty=qty,
                    base_reason=f"{decision['reason']} | close-short",
                    buy_limit=buy_limit,
                    sell_limit=sell_limit,
                    mark_price=mark_price,
                    best_bid=best_bid_price,
                    best_ask=best_ask_price,
                    avg_buy=avg_buy,
                    cash_krw=cash_krw,
                    equity_krw=equity,
                    invested_krw=invested_krw,
                    decision=decision,
                    cycle_state=cycle_state,
                    stats=stats,
                    allow_rescue=False,
                    aggressive=True,
                )
                self.db.update_ai_decision_execution(decision_id, "BUY_CLOSE_SHORT_SUBMITTED" if ok else "BUY_CLOSE_SHORT_SKIPPED", {"qty": float(abs(qty)), "buy_limit": float(buy_limit), "entry_price": float(entry_price)})
                if ok:
                    return
            ok = self.rebalance_buy(market, mark_price or buy_limit, buy_limit, qty, target_notional, cash_krw, equity, invested_krw, decision, stats)
            self.db.update_ai_decision_execution(decision_id, "BUY_SUBMITTED" if ok else "BUY_SKIPPED", {"target_notional": float(target_notional), "cash_krw": float(cash_krw), "entry_price": float(entry_price)})
            if ok:
                cycle_state.update(self.refresh_cycle_state())
            return
        if decision["action"] == "SELL":
            if direction > 0:
                ok = self._close_position_with_fallback(
                    market=market,
                    direction=direction,
                    qty=qty,
                    base_reason=f"{decision['reason']} | close-long",
                    buy_limit=buy_limit,
                    sell_limit=sell_limit,
                    mark_price=mark_price,
                    best_bid=best_bid_price,
                    best_ask=best_ask_price,
                    avg_buy=avg_buy,
                    cash_krw=cash_krw,
                    equity_krw=equity,
                    invested_krw=invested_krw,
                    decision=decision,
                    cycle_state=cycle_state,
                    stats=stats,
                    allow_rescue=False,
                    aggressive=True,
                )
                self.db.update_ai_decision_execution(decision_id, "SELL_CLOSE_LONG_SUBMITTED" if ok else "SELL_CLOSE_LONG_SKIPPED", {"qty": float(abs(qty)), "sell_limit": float(sell_limit), "entry_price": float(entry_price)})
                if ok:
                    return
            if not self.cfg.futures_short_enabled:
                self.db.update_ai_decision_execution(decision_id, "SELL_DISABLED", {"futures_short_enabled": False})
                return
            ok = self.rebalance_short(market, mark_price or sell_limit, sell_limit, qty, target_notional, cash_krw, equity, invested_krw, decision, stats)
            self.db.update_ai_decision_execution(decision_id, "SELL_SHORT_SUBMITTED" if ok else "SELL_SHORT_SKIPPED", {"target_notional": float(target_notional), "cash_krw": float(cash_krw), "entry_price": float(entry_price)})
            if ok:
                cycle_state.update(self.refresh_cycle_state())
            return
        stats["buy_nonbuy_action"] += 1
        if (not has_position) and decision["use_rebound"] and decision["enabled_signals"].get("rebound", True):
            ok = self.rebound_buy(market, buy_limit, cash_krw, equity, invested_krw, decision)
            self.db.update_ai_decision_execution(decision_id, "HOLD_REBOUND_BUY_SUBMITTED" if ok else "HOLD_REBOUND_BUY_SKIPPED", {"cash_krw": float(cash_krw), "buy_limit": float(buy_limit), "entry_price": float(entry_price)})
        else:
            self.db.update_ai_decision_execution(decision_id, "NO_ORDER", {"has_position": has_position, "entry_price": float(entry_price)})

    def run_forever(self) -> None:
        self.refresh_target_markets(force=True)
        log(f"[START] exchange={self.exchange_name} DRY_RUN={self.cfg.dry_run} markets={self.cfg.target_markets} excluded={self.cfg.excluded_symbols} stable_trading={self.cfg.allow_stablecoin_trading} models={self.cfg.ollama_models} leverage={self.cfg.binance_leverage} margin_type={self.cfg.binance_margin_type} short_enabled={self.cfg.futures_short_enabled}")
        while True:
            started = time.time()
            try:
                self.run_cycle()
            except Exception as exc:
                log(f"[ERROR] {type(exc).__name__}: {exc}")
            time.sleep(max(5, self.cfg.loop_interval_seconds - int(time.time() - started)))


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _read_json_or_default(raw: Any, default: Any) -> Any:
    if raw in (None, ""):
        return default
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return default


def _upsert_env_file(env_path: Path, updates: Dict[str, str]) -> None:
    existing = []
    if env_path.exists():
        existing = env_path.read_text(encoding="utf-8").splitlines()
    seen = set()
    out: List[str] = []
    for line in existing:
        stripped = line.strip()
        replaced = False
        for key, value in updates.items():
            if stripped.startswith(f"{key}="):
                out.append(f"{key}={value}")
                seen.add(key)
                replaced = True
                break
        if not replaced:
            out.append(line)
    for key, value in updates.items():
        if key not in seen:
            out.append(f"{key}={value}")
    env_path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


def _infer_target_action_from_return(price_return: float, position_qty: Decimal, hold_band: float) -> str:
    if price_return >= hold_band:
        return "BUY"
    if price_return <= -hold_band:
        return "SELL" if position_qty > 0 else "HOLD"
    return "HOLD"


def _normalize_export_decision(cfg: Config, decision: dict, position_qty: Decimal, fallback_action: str) -> dict:
    action = str(decision.get("action", fallback_action or "HOLD")).upper()
    if position_qty <= 0 and action == "SELL":
        action = "HOLD"
    enabled = decision.get("enabled_signals") if isinstance(decision.get("enabled_signals"), dict) else {key: True for key in StrategyEngine.SIGNAL_KEYS}
    weights = decision.get("signal_weights") if isinstance(decision.get("signal_weights"), dict) else {"technical": 1.0, "sentiment": 0.9, "onchain": 0.9, "arbitrage": 0.0, "rebound": 0.8, "rl": 0.6}
    return {
        "action": action,
        "confidence": clamp_float(decision.get("confidence", 0.5), 0.0, 1.0, 0.5),
        "target_risk": clamp_float(decision.get("target_risk", 0.25), 0.0, 1.0, 0.25),
        "reason": str(decision.get("reason", "sft-label")),
        "reason_raw": str(decision.get("reason_raw", decision.get("reason", "sft-label"))),
        "use_grid": bool(decision.get("use_grid", False)),
        "use_rebound": bool(decision.get("use_rebound", False)),
        "enabled_signals": {key: bool(enabled.get(key, True)) for key in StrategyEngine.SIGNAL_KEYS},
        "signal_weights": {key: clamp_float(weights.get(key, 1.0), 0.0, 2.0, 1.0) for key in ["technical", "sentiment", "onchain", "arbitrage", "rebound", "rl"]},
        "parameter_overrides": decision.get("parameter_overrides", {}) if isinstance(decision.get("parameter_overrides", {}), dict) else {},
    }


def _build_corrective_decision(cfg: Config, original: dict, position_qty: Decimal, price_return: float) -> dict:
    target_action = _infer_target_action_from_return(price_return, position_qty, cfg.sft_hold_band)
    corrected = dict(original)
    corrected.update({
        "action": target_action,
        "confidence": clamp_float(max(float(original.get("confidence", 0.5)), min(0.95, abs(price_return) * 12 + 0.55)), 0.0, 1.0, 0.65),
        "target_risk": 0.2 if target_action == "HOLD" else clamp_float(original.get("target_risk", 0.3), 0.0, 1.0, 0.3),
        "reason": f"reward-corrected:{target_action}:future_return={price_return:.4f}",
        "use_grid": target_action == "HOLD",
        "use_rebound": target_action == "BUY" and price_return > cfg.sft_hold_band,
    })
    return _normalize_export_decision(cfg, corrected, position_qty, target_action)


def _score_prediction(label: dict, pred: dict) -> dict:
    label_action = str(label.get("action", "HOLD")).upper()
    pred_action = str(pred.get("action", "HOLD")).upper()
    action_match = label_action == pred_action
    confidence_gap = abs(float(label.get("confidence", 0.5)) - float(pred.get("confidence", 0.5)))
    risk_gap = abs(float(label.get("target_risk", 0.25)) - float(pred.get("target_risk", 0.25)))
    return {
        "action_match": action_match,
        "confidence_gap": confidence_gap,
        "risk_gap": risk_gap,
        "score": (1.0 if action_match else 0.0) + max(0.0, 0.5 - confidence_gap) + max(0.0, 0.25 - risk_gap),
    }


class LLMOpsWorkflow:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.db = BotDB(cfg.db_path)
        self.ollama = OllamaClient(cfg.ollama_url, cfg.ollama_models, timeout=cfg.request_timeout * 2, ensemble_enabled=cfg.ollama_ensemble_enabled)

    def export_reward_labeled_jsonl(self, output_path: Optional[str] = None) -> Path:
        output = Path(output_path or self.cfg.sft_output_jsonl)
        _ensure_parent_dir(output)
        rows = self.db.get_reward_labeled_ai_decisions(
            good_threshold=self.cfg.sft_good_reward_threshold,
            bad_threshold=self.cfg.sft_bad_reward_threshold,
            limit=self.cfg.sft_max_rows,
        )
        written = 0
        good = 0
        bad = 0
        with output.open("w", encoding="utf-8") as f:
            for row in rows:
                prompt_payload = _read_json_or_default(row["prompt_json"], {})
                original = _read_json_or_default(row["decision_json"], {})
                reward_meta = _read_json_or_default(row["reward_meta_json"], {})
                position_qty = to_decimal(row["position_qty"] or 0)
                prompt_text = TRADE_DECISION_SYSTEM_PROMPT + "\n" + build_trade_decision_prompt(prompt_payload)
                raw_return = float(reward_meta.get("raw_return", 0.0))
                reward = float(row["reward"] or 0.0)
                if reward >= self.cfg.sft_good_reward_threshold:
                    target = _normalize_export_decision(self.cfg, original, position_qty, str(original.get("action", "HOLD")))
                    label_source = "good"
                    good += 1
                else:
                    target = _build_corrective_decision(self.cfg, original, position_qty, raw_return)
                    label_source = "bad"
                    bad += 1
                target_text = safe_json_dumps(target)
                record = {
                    "messages": [
                        {"role": "system", "content": TRADE_DECISION_SYSTEM_PROMPT},
                        {"role": "user", "content": build_trade_decision_prompt(prompt_payload)},
                        {"role": "assistant", "content": target_text},
                    ],
                    "prompt": prompt_text,
                    "completion": target_text,
                    "label_decision": target,
                    "metadata": {
                        "decision_id": int(row["id"]),
                        "market": row["market"],
                        "ts": row["ts"],
                        "context": row["context"],
                        "model_name": row["model_name"],
                        "execution_status": row["execution_status"],
                        "reward": reward,
                        "raw_return": raw_return,
                        "label_source": label_source,
                    },
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
        log(f"[SFT EXPORT] output={output} written={written} good={good} bad={bad}")
        return output

    def train_lora_sft(self, dataset_path: Optional[str] = None, output_dir: Optional[str] = None) -> Path:
        dataset_file = Path(dataset_path or self.cfg.sft_output_jsonl)
        output = Path(output_dir or self.cfg.sft_train_output_dir)
        _ensure_parent_dir(output / "dummy")
        try:
            import torch
            from datasets import load_dataset
            from peft import LoraConfig
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import SFTConfig, SFTTrainer
        except Exception as exc:
            raise RuntimeError("train-lora requires torch, datasets, transformers, peft, trl packages") from exc

        if not dataset_file.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_file}")

        dataset = load_dataset("json", data_files=str(dataset_file), split="train")
        if "completion" not in dataset.column_names:
            def _ensure_prompt_completion(example: dict) -> dict:
                prompt = example.get("prompt")
                completion = example.get("completion")
                if not completion:
                    label_decision = example.get("label_decision")
                    if isinstance(label_decision, dict):
                        completion = safe_json_dumps(label_decision)
                    elif label_decision not in (None, ""):
                        completion = str(label_decision)
                messages = example.get("messages")
                if not completion and isinstance(messages, list):
                    for msg in reversed(messages):
                        if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "assistant":
                            completion = str(msg.get("content", ""))
                            if completion:
                                break
                if not prompt and isinstance(messages, list):
                    prompt_parts: List[str] = []
                    for msg in messages:
                        if not isinstance(msg, dict):
                            continue
                        role = str(msg.get("role", "")).lower()
                        if role == "assistant":
                            break
                        content = str(msg.get("content", "") or "").strip()
                        if content:
                            prompt_parts.append(content)
                    prompt = "\n".join(prompt_parts).strip()
                return {
                    "prompt": str(prompt or ""),
                    "completion": str(completion or ""),
                }

            dataset = dataset.map(_ensure_prompt_completion)
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.sft_base_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        dtype = torch.bfloat16 if self.cfg.sft_use_bf16 and torch.cuda.is_available() else (torch.float16 if torch.cuda.is_available() else torch.float32)
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.sft_base_model,
            dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
        )
        model.config.use_cache = False
        if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
            model.config.pad_token_id = tokenizer.pad_token_id
        peft_config = LoraConfig(
            r=self.cfg.sft_lora_r,
            lora_alpha=self.cfg.sft_lora_alpha,
            lora_dropout=self.cfg.sft_lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=self.cfg.sft_target_modules,
        )
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            peft_config=peft_config,
            args=SFTConfig(
                output_dir=str(output),
                learning_rate=self.cfg.sft_learning_rate,
                num_train_epochs=self.cfg.sft_num_train_epochs,
                per_device_train_batch_size=self.cfg.sft_per_device_batch_size,
                gradient_accumulation_steps=self.cfg.sft_gradient_accumulation_steps,
                max_length=self.cfg.sft_max_seq_length,
                completion_only_loss=True,
                logging_steps=10,
                save_strategy="epoch",
                bf16=self.cfg.sft_use_bf16 and torch.cuda.is_available(),
                fp16=(not self.cfg.sft_use_bf16) and torch.cuda.is_available(),
                report_to="none",
                dataloader_pin_memory=torch.cuda.is_available(),
                use_cpu=not torch.cuda.is_available(),
            ),
        )
        trainer.train()
        trainer.model.save_pretrained(str(output))
        tokenizer.save_pretrained(str(output))
        log(f"[SFT TRAIN] base_model={self.cfg.sft_base_model} dataset={dataset_file} output={output}")
        return output

    def _resolve_llama_cpp_convert_script(self) -> Path:
        configured = (self.cfg.llama_cpp_convert_lora_to_gguf or "").strip()
        if configured:
            candidate = Path(configured).expanduser().resolve()
            if candidate.exists():
                return candidate
            raise FileNotFoundError(f"LLAMA_CPP_CONVERT_LORA_TO_GGUF not found: {candidate}")
        candidate = (Path(self.cfg.llama_cpp_dir).expanduser().resolve() / "convert_lora_to_gguf.py")
        if candidate.exists():
            return candidate
        raise FileNotFoundError(
            "convert_lora_to_gguf.py not found. Set LLAMA_CPP_DIR or LLAMA_CPP_CONVERT_LORA_TO_GGUF to your llama.cpp checkout."
        )

    def _effective_deploy_strategy(self) -> str:
        if self.cfg.ollama_deploy_strategy != "auto":
            return self.cfg.ollama_deploy_strategy
        base_hint = f"{self.cfg.sft_base_model} {self.cfg.ollama_deploy_base_model}".lower()
        if "qwen" in base_hint:
            return "gguf_adapter"
        if any(tag in base_hint for tag in ("llama", "mistral", "gemma")):
            return "safetensors_adapter"
        return "gguf_adapter"

    def convert_lora_adapter_to_gguf(self, adapter_dir: Optional[str] = None, output_path: Optional[str] = None) -> Path:
        adapter_path = Path(adapter_dir or self.cfg.sft_train_output_dir).resolve()
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter dir not found: {adapter_path}")
        convert_script = self._resolve_llama_cpp_convert_script()
        gguf_path = Path(output_path or self.cfg.ollama_gguf_adapter_path).expanduser().resolve()
        _ensure_parent_dir(gguf_path)
        cmd = [
            sys.executable,
            str(convert_script),
            str(adapter_path),
            "--base-model-id",
            self.cfg.sft_base_model,
            "--outfile",
            str(gguf_path),
            "--outtype",
            "f16",
        ]
        log(f"[LORA->GGUF] adapter={adapter_path} outfile={gguf_path} base_model_id={self.cfg.sft_base_model}")
        subprocess.run(cmd, check=True)
        return gguf_path

    def deploy_ollama_adapter(self, adapter_dir: Optional[str] = None) -> Path:
        adapter_path = Path(adapter_dir or self.cfg.sft_train_output_dir).resolve()
        if not adapter_path.exists():
            raise FileNotFoundError(f"Adapter dir not found: {adapter_path}")
        strategy = self._effective_deploy_strategy()
        deploy_adapter_path = adapter_path
        if strategy == "safetensors_adapter":
            supported = any(tag in self.cfg.ollama_deploy_base_model.lower() or tag in self.cfg.sft_base_model.lower() for tag in ("llama", "mistral", "gemma"))
            if not supported:
                raise ValueError("Safetensors adapter deployment to Ollama is only documented for Llama/Mistral/Gemma base families. Use OLLAMA_DEPLOY_STRATEGY=gguf_adapter for Qwen-based deployment.")
        elif strategy == "gguf_adapter":
            deploy_adapter_path = self.convert_lora_adapter_to_gguf(str(adapter_path))
        else:
            raise ValueError(f"Unsupported deploy strategy: {strategy}")

        modelfile = Path(self.cfg.ollama_modelfile_path)
        _ensure_parent_dir(modelfile)
        modelfile.write_text(
            "\n".join([
                f"FROM {self.cfg.ollama_deploy_base_model}",
                f"ADAPTER {deploy_adapter_path.as_posix()}",
                "PARAMETER temperature 0",
                "PARAMETER num_ctx 4096",
            ]) + "\n",
            encoding="utf-8",
        )
        subprocess.run(["ollama", "create", self.cfg.ollama_deploy_model_name, "-f", str(modelfile)], check=True)
        if self.cfg.ollama_update_env_after_deploy:
            env_path = Path(self.cfg.env_file)
            compare_models = []
            if env_path.exists():
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    if line.startswith("OLLAMA_MODEL_LIST="):
                        compare_models = parse_csv_list(line.split("=", 1)[1])
                        break
            if not compare_models:
                compare_models = [m for m in self.cfg.ollama_models if m != self.cfg.ollama_deploy_model_name]
            if self.cfg.ollama_deploy_model_name not in compare_models:
                compare_models.append(self.cfg.ollama_deploy_model_name)
            _upsert_env_file(env_path, {
                "OLLAMA_MODEL": self.cfg.ollama_deploy_model_name,
                "OLLAMA_MODEL_LIST": ",".join(compare_models),
                "OLLAMA_ENSEMBLE_ENABLED": "false",
            })
            log(f"[ENV UPDATE] env={env_path} OLLAMA_MODEL={self.cfg.ollama_deploy_model_name} OLLAMA_MODEL_LIST={','.join(compare_models)}")
        log(
            f"[OLLAMA DEPLOY] strategy={strategy} model={self.cfg.ollama_deploy_model_name} base={self.cfg.ollama_deploy_base_model} adapter={deploy_adapter_path} modelfile={modelfile}"
        )
        return modelfile


    def compare_models(self, input_jsonl: Optional[str] = None, output_jsonl: Optional[str] = None, max_cases: Optional[int] = None) -> Path:
        input_path = Path(input_jsonl or self.cfg.compare_input_jsonl)
        output_path = Path(output_jsonl or self.cfg.compare_output_jsonl)
        _ensure_parent_dir(output_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Compare dataset not found: {input_path}")
        records = []
        with input_path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= (max_cases or self.cfg.compare_max_cases):
                    break
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        aggregate = {model: {"cases": 0, "action_match": 0, "score_sum": 0.0} for model in self.cfg.ollama_models}
        with output_path.open("w", encoding="utf-8") as out:
            for record in records:
                prompt = record.get("prompt") or (TRADE_DECISION_SYSTEM_PROMPT + "\n" + record["messages"][1]["content"])
                label = record.get("label_decision") or _read_json_or_default(record["messages"][-1]["content"], {})
                preds = self.ollama.generate_json_by_model(prompt, options={"temperature": 0, "seed": 42})
                scored = {}
                for model, pred in preds.items():
                    if isinstance(pred, dict) and pred.get("_error"):
                        scored[model] = {"prediction": pred, "metrics": {"action_match": False, "confidence_gap": 1.0, "risk_gap": 1.0, "score": 0.0}}
                    else:
                        metrics = _score_prediction(label, pred if isinstance(pred, dict) else {})
                        scored[model] = {"prediction": pred, "metrics": metrics}
                        aggregate.setdefault(model, {"cases": 0, "action_match": 0, "score_sum": 0.0})
                        aggregate[model]["cases"] += 1
                        aggregate[model]["action_match"] += 1 if metrics["action_match"] else 0
                        aggregate[model]["score_sum"] += float(metrics["score"])
                out.write(json.dumps({
                    "decision_id": record.get("metadata", {}).get("decision_id"),
                    "market": record.get("metadata", {}).get("market"),
                    "label_decision": label,
                    "predictions": scored,
                }, ensure_ascii=False) + "\n")
        summary_parts = []
        for model, stats in aggregate.items():
            if stats["cases"] <= 0:
                continue
            summary_parts.append(
                f"{model}: action_match={stats['action_match']}/{stats['cases']} avg_score={(stats['score_sum'] / stats['cases']):.3f}"
            )
        log(f"[MODEL COMPARE] input={input_path} output={output_path} | {' | '.join(summary_parts)}")
        return output_path

    def run_step234(self) -> None:
        dataset_path = self.export_reward_labeled_jsonl()
        adapter_dir = self.train_lora_sft(str(dataset_path))
        self.deploy_ollama_adapter(str(adapter_dir))
        self.compare_models(str(dataset_path))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Binance Futures AI bot + SFT/LoRA/Ollama workflow")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("run")
    sub.add_parser("export-sft-jsonl")
    sub.add_parser("train-lora")
    sub.add_parser("deploy-ollama-adapter")
    sub.add_parser("compare-models")
    sub.add_parser("run-step234")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    command = args.command or "run"
    cfg = Config.from_env()
    if command == "run":
        bot = BinanceFuturesAIBot(cfg)
        bot.run_forever()
        return
    workflow = LLMOpsWorkflow(cfg)
    if command == "export-sft-jsonl":
        workflow.export_reward_labeled_jsonl()
    elif command == "train-lora":
        workflow.train_lora_sft()
    elif command == "deploy-ollama-adapter":
        workflow.deploy_ollama_adapter()
    elif command == "compare-models":
        workflow.compare_models()
    elif command == "run-step234":
        workflow.run_step234()
    else:
        parser.error(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
