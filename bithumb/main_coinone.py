import base64
import builtins
import hashlib
import hmac
import json
import os
import sqlite3
import ssl
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from email.message import EmailMessage
from typing import Any, Dict, List, Optional, Tuple

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


def parse_markets(markets: str) -> List[str]:
    return [m.strip().upper() for m in markets.split(",") if m.strip()]


def parse_symbols(symbols: str) -> List[str]:
    return [s.strip().upper() for s in symbols.split(",") if s.strip()]


def split_market(market: str) -> Tuple[str, str]:
    quote, base = market.split("-")
    return quote, base


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
    grid_levels: int
    grid_step_pct: Decimal
    min_trade_krw: Decimal
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
        target_markets = filter_markets(
            parse_markets(os.environ.get("TARGET_MARKETS", "KRW-XRP,KRW-SOL,KRW-DOGE")),
            blocked,
        )
        return cls(
            coinone_access_token=os.environ["COINONE_ACCESS_TOKEN"],
            coinone_secret_key=os.environ["COINONE_SECRET_KEY"],
            gmail_user=os.environ["GMAIL_USER"],
            gmail_password=os.environ["GMAIL_PASSWORD"],
            target_email=os.environ["TARGET_EMAIL"],
            db_path=os.environ.get("DB_PATH", "./ai_coin_bot.db"),
            ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
            ollama_models=collect_ollama_models(),
            etherscan_api_key=os.environ.get("ETHERSCAN_API_KEY") or None,
            dry_run=asbool(os.environ.get("DRY_RUN"), True),
            target_markets=target_markets,
            excluded_symbols=excluded_symbols,
            stablecoin_symbols=stablecoin_symbols,
            allow_stablecoin_trading=allow_stablecoin_trading,
            ai_override_enabled=asbool(os.environ.get("AI_OVERRIDE_ENABLED"), True),
            loop_interval_seconds=int(os.environ.get("LOOP_INTERVAL_SECONDS", "300")),
            report_interval_seconds=int(os.environ.get("REPORT_INTERVAL_SECONDS", str(3 * 60 * 60))),
            max_exposure_per_market=to_decimal(os.environ.get("MAX_EXPOSURE_PER_MARKET", "0.20")),
            max_total_exposure=to_decimal(os.environ.get("MAX_TOTAL_EXPOSURE", "0.60")),
            stop_loss_pct=to_decimal(os.environ.get("STOP_LOSS_PCT", "0.035")),
            take_profit_pct=to_decimal(os.environ.get("TAKE_PROFIT_PCT", "0.060")),
            grid_levels=int(os.environ.get("GRID_LEVELS", "4")),
            grid_step_pct=to_decimal(os.environ.get("GRID_STEP_PCT", "0.012")),
            min_trade_krw=to_decimal(os.environ.get("MIN_TRADE_KRW", "20000")),
            market_quote_currency=os.environ.get("MARKET_QUOTE_CURRENCY", "KRW").upper(),
            auto_discover_markets=asbool(os.environ.get("AUTO_DISCOVER_MARKETS"), True),
            market_universe_size=max(1, int(os.environ.get("MARKET_UNIVERSE_SIZE", "350"))),
            market_refresh_seconds=max(60, int(os.environ.get("MARKET_REFRESH_SECONDS", str(6 * 60 * 60)))),
            coinone_post_only=asbool(os.environ.get("COINONE_POST_ONLY"), True),
            buy_book_level=max(1, int(os.environ.get("BUY_BOOK_LEVEL", "1"))),
            sell_book_level=max(1, int(os.environ.get("SELL_BOOK_LEVEL", "1"))),
            gmail_smtp_host=os.environ.get("GMAIL_SMTP_HOST", "smtp.gmail.com"),
            gmail_smtp_port=int(os.environ.get("GMAIL_SMTP_PORT", "465")),
            request_timeout=int(os.environ.get("REQUEST_TIMEOUT", "15")),
            buy_confidence_threshold=clamp_float(os.environ.get("BUY_CONFIDENCE_THRESHOLD", 0.60), 0.0, 1.0, 0.60),
            strong_hold_buy_enabled=asbool(os.environ.get("STRONG_HOLD_BUY_ENABLED"), False),
            strong_hold_buy_confidence=clamp_float(os.environ.get("STRONG_HOLD_BUY_CONFIDENCE", 0.80), 0.0, 1.0, 0.80),
            min_new_position_krw=to_decimal(os.environ.get("MIN_NEW_POSITION_KRW", os.environ.get("MIN_TRADE_KRW", "20000"))),
            suppress_unheld_sell_logs=asbool(os.environ.get("SUPPRESS_UNHELD_SELL_LOGS"), True),
            order_timeout_seconds=max(5, int(os.environ.get("ORDER_TIMEOUT_SECONDS", os.environ.get("BUY_ORDER_TIMEOUT_SECONDS", "30")))),
            order_max_attempts=max(1, int(os.environ.get("ORDER_MAX_ATTEMPTS", "10"))),
            order_poll_interval_seconds=max(1, int(os.environ.get("ORDER_POLL_INTERVAL_SECONDS", "2"))),
            pending_sync_interval_seconds=max(5, int(os.environ.get("PENDING_SYNC_INTERVAL_SECONDS", "15"))),
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
        for attempt in range(3):
            try:
                response = self.session.request(method, url, timeout=self.timeout, **kwargs)
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
    def __init__(self, base_url: str, models: List[str], timeout: int = 30) -> None:
        super().__init__(timeout=timeout)
        self.base_url = base_url.rstrip("/")
        self.models = [m for m in models if m]
        if not self.models:
            raise ValueError("At least one Ollama model is required")

    def _generate(self, model: str, prompt: str) -> str:
        response = self._request("POST", f"{self.base_url}/api/generate", json={"model": model, "prompt": prompt, "stream": False, "format": "json"})
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
        for model in self.models:
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
            out["take_profit_pct"] = clamp_float(raw.get("take_profit_pct"), 0.02, 0.20, float(self.cfg.take_profit_pct))
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
            "grid_step_pct": Decimal(str(overrides.get("grid_step_pct", float(self.cfg.grid_step_pct)))),
            "grid_levels": int(overrides.get("grid_levels", self.cfg.grid_levels)),
            "buy_budget_fraction": Decimal(str(overrides.get("buy_budget_fraction", 0.95))),
            "rebound_buy_fraction": Decimal(str(overrides.get("rebound_buy_fraction", 0.10))),
            "max_exposure_multiplier": Decimal(str(overrides.get("max_exposure_multiplier", 1.00))),
        }

    def ai_decision(self, market: str, technical: dict, sentiment: dict, onchain: dict, rl_action: str, position_qty: Decimal) -> dict:
        prompt = (
            "You are a crypto spot trading model for Korean KRW markets. Return strict JSON with keys: action (BUY/HOLD/SELL), confidence (0..1), target_risk (0..1), reason, use_grid (true/false), use_rebound (true/false), enabled_signals, signal_weights, parameter_overrides.\n"
            "SELL is only allowed when position_qty > 0. If position_qty == 0, choose BUY or HOLD only. Favor capital preservation in downtrends.\n"
            f"market={market}\nposition_qty={position_qty}\ntechnical={safe_json_dumps(technical)}\nsentiment={safe_json_dumps(sentiment)}\nonchain={safe_json_dumps(onchain)}\nrl_hint={rl_action}\n"
        )
        try:
            decision = self.ollama.generate_json(prompt) if self.cfg.ai_override_enabled else {}
            action = str(decision.get("action", "HOLD")).upper()
            if action not in {"BUY", "HOLD", "SELL"}:
                action = "HOLD"
            if position_qty <= 0 and action == "SELL":
                action = "HOLD"
            normalized = {
                "action": action,
                "confidence": clamp_float(decision.get("confidence", 0.5), 0.0, 1.0, 0.5),
                "target_risk": clamp_float(decision.get("target_risk", 0.3), 0.0, 1.0, 0.3),
                "reason": str(decision.get("reason", "ollama")),
                "use_grid": bool(decision.get("use_grid", False)),
                "use_rebound": bool(decision.get("use_rebound", technical.get("rebound_signal", False))),
                "enabled_signals": self.normalize_enabled_signals(decision.get("enabled_signals")),
                "signal_weights": self.normalize_signal_weights(decision.get("signal_weights")),
                "parameter_overrides": self.normalize_parameter_overrides(decision.get("parameter_overrides")),
            }
            normalized["effective_params"] = self.effective_params(normalized)
            return normalized
        except Exception:
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
                "reason": "heuristic-fallback",
                "use_grid": action == "HOLD",
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
        self.ollama = OllamaClient(cfg.ollama_url, cfg.ollama_models, timeout=cfg.request_timeout * 2)
        self.coinone = CoinoneClient(cfg.coinone_access_token, cfg.coinone_secret_key, timeout=cfg.request_timeout)
        self.sentiment = SentimentProvider(self.ollama, timeout=cfg.request_timeout)
        self.onchain = OnChainProvider(cfg.etherscan_api_key, timeout=cfg.request_timeout)
        self.strategy = StrategyEngine(cfg, self.ollama)
        self.rl = BanditRLPolicy(self.db)
        self.mailer = Mailer(cfg)
        self.last_report_at = 0.0
        self.last_market_refresh_at = 0.0
        self.last_pending_sync_at = 0.0

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
        if side.upper() == "BUY":
            if bids:
                return bids[min(idx, len(bids) - 1)]
            return asks[0] if asks else Decimal("0")
        if asks:
            return asks[min(idx, len(asks) - 1)]
        return bids[0] if bids else Decimal("0")

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

    def wait_for_order_attempt(self, market: str, side: str, order_id: str, price: Decimal, submitted_qty: Decimal, submitted_notional: Decimal, reason: str, attempt_no: int) -> dict:
        terminal_statuses = self.order_terminal_statuses()
        live_statuses = self.order_live_statuses()
        timeout_seconds = max(1, self.cfg.order_timeout_seconds)
        poll_seconds = max(1, self.cfg.order_poll_interval_seconds)
        deadline = time.time() + timeout_seconds
        last_payload: dict = {}
        last_status = "SUBMITTED"
        remain_qty = submitted_qty
        executed_qty = Decimal("0")
        submitted_monotonic = time.time()
        next_progress_log_at = submitted_monotonic + min(10, timeout_seconds)

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
                    log(f"[ORDER WAIT] coinone {market} {side} order_id={order_id} attempt={attempt_no} elapsed={elapsed}s status={tracked_status} remain_qty={remain_qty}")
                    next_progress_log_at = now_monotonic + min(10, timeout_seconds)
            except Exception as exc:
                log(f"[ORDER POLL WARN] coinone {market} {side} order_id={order_id} attempt={attempt_no}: {exc}")
            time.sleep(poll_seconds)

        try:
            cancel_result = self.coinone.cancel_order(market, order_id)
            self.db.mark_pending_order_cancel_requested(order_id, cancel_result)
            last_payload = cancel_result or last_payload
            log(f"[{side} CANCEL REQUESTED] coinone {market} order_id={order_id} attempt={attempt_no} timeout={timeout_seconds}s")
        except Exception as exc:
            log(f"[{side} CANCEL FAILED] coinone {market} order_id={order_id} attempt={attempt_no}: {exc}")

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
                log(f"[ORDER CANCEL POLL WARN] coinone {market} {side} order_id={order_id} attempt={attempt_no}: {exc}")
            time.sleep(1)

        self.db.update_pending_order_status(order_id, "CANCEL_REQUESTED", last_payload, canceled=False)
        return {
            "status": "CANCEL_REQUESTED",
            "executed_qty": executed_qty,
            "remain_qty": max(Decimal("0"), remain_qty),
            "payload": last_payload,
        }

    def execute_limit_order_with_retry(self, market: str, side: str, price: Decimal, qty: Decimal, notional_krw: Decimal, reason: str) -> bool:
        qty = quantize_down(qty, 8)
        fixed_price = quantize_down(price, 0) if price == price.to_integral_value() else price
        requested_qty = qty
        remaining_qty = qty
        max_attempts = max(1, self.cfg.order_max_attempts)
        failure_note = ""
        consecutive_attempt = self.current_market_attempt(market, side)

        if requested_qty <= 0 or fixed_price <= 0:
            log(f"[SKIP {side}] coinone {market} invalid fixed order params price={fixed_price} qty={requested_qty}")
            return False

        if self.cfg.dry_run:
            self.db.log_trade("coinone", market, side, "LIMIT", fixed_price, requested_qty, notional_krw, "DRY_RUN", reason)
            log(f"[DRY {side}] coinone {market} price={fixed_price} qty={requested_qty} reason={reason}")
            return True

        for local_attempt in range(1, max_attempts + 1):
            remaining_qty = quantize_down(remaining_qty, 8)
            remaining_notional = fixed_price * remaining_qty
            display_attempt = consecutive_attempt + 1
            attempt_tag = f"attempt={display_attempt} local_attempt={local_attempt}/{max_attempts}"

            if remaining_qty <= 0:
                self.reset_market_attempt(market, side, f"{reason} | success before submit")
                return True
            if remaining_notional < self.cfg.min_trade_krw:
                if remaining_qty < requested_qty:
                    self.reset_market_attempt(market, side, f"{reason} | partial fill success; dust remain")
                    self.db.log_trade(
                        "coinone", market, side, "LIMIT", fixed_price, remaining_qty, remaining_notional,
                        "PARTIAL_DUST_REMAIN", f"{reason} | remain below minimum after partial fills | last_attempt={display_attempt}", None, {"attempt": display_attempt, "local_attempt": local_attempt, "remaining_qty": str(remaining_qty)},
                    )
                    log(f"[{side} PARTIAL COMPLETE] coinone {market} remaining_notional={remaining_notional} below minimum; stop retry")
                    return True
                log(f"[SKIP {side}] coinone {market} below min notional={remaining_notional}")
                return False
            try:
                result = self.coinone.place_limit_order(market, side, fixed_price, remaining_qty, post_only=self.cfg.coinone_post_only)
                order_id = result.get("order_id")
                self.db.log_trade(
                    "coinone", market, side, "LIMIT", fixed_price, remaining_qty, remaining_notional,
                    "SUBMITTED", f"{reason} | {attempt_tag}", order_id, result,
                )
                if order_id:
                    self.db.add_pending_order(market, side, fixed_price, remaining_qty, remaining_notional, order_id, result)
                log(f"[{side}] coinone {market} price={fixed_price} qty={remaining_qty} order_id={order_id} {attempt_tag}")
                if not order_id:
                    raise RuntimeError("Coinone response missing order_id")

                outcome = self.wait_for_order_attempt(market, side, order_id, fixed_price, remaining_qty, remaining_notional, reason, display_attempt)
                executed_qty = quantize_down(outcome.get("executed_qty") or Decimal("0"), 8)
                remaining_qty = quantize_down(outcome.get("remain_qty") or Decimal("0"), 8)
                status = str(outcome.get("status") or "UNKNOWN").upper()
                payload = outcome.get("payload") or {}

                if remaining_qty <= 0:
                    self.reset_market_attempt(market, side, f"{reason} | filled")
                    self.db.log_trade(
                        "coinone", market, side, "LIMIT", fixed_price, requested_qty, fixed_price * requested_qty,
                        "FILLED", f"{reason} | completed attempt={display_attempt} local_attempt={local_attempt}/{max_attempts}", order_id, payload,
                    )
                    log(f"[{side} FILLED] coinone {market} attempts={display_attempt} local_attempt={local_attempt}/{max_attempts}")
                    return True

                if status == "CANCEL_REQUESTED":
                    consecutive_attempt = self.fail_market_attempt(market, side, f"{reason} | cancel not confirmed")
                    failure_note = f"attempt={consecutive_attempt} cancel not confirmed"
                    self.db.log_trade(
                        "coinone", market, side, "LIMIT", fixed_price, remaining_qty, fixed_price * remaining_qty,
                        "FAILED_CANCEL_UNCONFIRMED", f"{reason} | {failure_note} | local_attempt={local_attempt}/{max_attempts}", order_id, payload,
                    )
                    log(f"[{side} ABORT] coinone {market} {failure_note} remaining_qty={remaining_qty}")
                    break

                if executed_qty > 0:
                    self.reset_market_attempt(market, side, f"{reason} | partial fill success")
                    consecutive_attempt = 0
                    self.db.log_trade(
                        "coinone", market, side, "LIMIT", fixed_price, executed_qty, fixed_price * executed_qty,
                        "PARTIAL_RETRY", f"{reason} | partial fill reset attempt | status={status} remaining_qty={remaining_qty} | last_attempt={display_attempt} local_attempt={local_attempt}/{max_attempts}", order_id, payload,
                    )
                    log(f"[{side} PARTIAL] coinone {market} attempt_reset=0 last_attempt={display_attempt} local_attempt={local_attempt}/{max_attempts} executed_qty={executed_qty} remaining_qty={remaining_qty} status={status}")
                else:
                    consecutive_attempt = self.fail_market_attempt(market, side, f"{reason} | retry pending status={status}")
                    self.db.log_trade(
                        "coinone", market, side, "LIMIT", fixed_price, remaining_qty, fixed_price * remaining_qty,
                        "RETRY_PENDING", f"{reason} | attempt={consecutive_attempt} status={status} local_attempt={local_attempt}/{max_attempts}", order_id, payload,
                    )
                    log(f"[{side} RETRY] coinone {market} attempt={consecutive_attempt} local_attempt={local_attempt}/{max_attempts} remaining_qty={remaining_qty} status={status}")
            except Exception as exc:
                consecutive_attempt = self.fail_market_attempt(market, side, f"{reason} | {exc}")
                self.db.log_trade(
                    "coinone", market, side, "LIMIT", fixed_price, remaining_qty, fixed_price * remaining_qty,
                    "FAILED", f"{reason} | attempt={consecutive_attempt} local_attempt={local_attempt}/{max_attempts}: {exc}", None, {},
                )
                log(f"[{side} FAILED] coinone {market} attempt={consecutive_attempt} local_attempt={local_attempt}/{max_attempts}: {exc}")

        final_attempts = self.current_market_attempt(market, side)
        failure_reason = f"{reason} | {failure_note}" if failure_note else f"{reason} | max attempts exceeded (local={max_attempts}, cumulative={final_attempts})"
        self.db.log_trade(
            "coinone", market, side, "LIMIT", fixed_price, remaining_qty, fixed_price * remaining_qty,
            "FAILED_MAX_RETRY", failure_reason, None, {"remaining_qty": str(remaining_qty), "attempt": final_attempts},
        )
        self.send_order_retry_failure_mail(market, side, fixed_price, requested_qty, remaining_qty, final_attempts, failure_reason)
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
                if age_seconds >= timeout_seconds and status in cancellable_statuses:
                    if str(row["status"] or "").upper() != "CANCEL_REQUESTED":
                        cancel_result = self.coinone.cancel_order(market, order_id)
                        self.db.mark_pending_order_cancel_requested(order_id, cancel_result)
                        self.db.log_trade(
                            "coinone", market, side, "LIMIT",
                            to_decimal(row["price"]), to_decimal(row["qty"]), to_decimal(row["notional_krw"]),
                            "CANCEL_REQUESTED", f"timeout>{timeout_seconds}s unfilled/partial", order_id, cancel_result,
                        )
                        log(f"[{side} CANCEL REQUESTED] coinone {market} order_id={order_id} age={int(age_seconds)}s status={status} remain_qty={remain_qty}")
                    continue
                tracked_status = status if status in {"LIVE", "PARTIALLY_FILLED", "TRIGGERED", "NOT_TRIGGERED"} else "SUBMITTED"
                self.db.update_pending_order_status(order_id, tracked_status, detail)
            except Exception as exc:
                log(f"[PENDING ORDER CHECK WARN] coinone {market} {side} order_id={order_id}: {exc}")

    def maybe_sync_pending_orders(self, force: bool = False) -> None:
        if force or (time.time() - self.last_pending_sync_at) >= self.cfg.pending_sync_interval_seconds:
            self.sync_pending_orders()

    def submit_buy(self, market: str, price: Decimal, spend_krw: Decimal, reason: str) -> bool:
        _, base = split_market(market)
        if self.is_trading_blocked_symbol(base):
            return False
        qty = quantize_down(spend_krw / price, 8) if price > 0 else Decimal("0")
        notional = price * qty
        if spend_krw < self.cfg.min_trade_krw:
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
        if notional < self.cfg.min_trade_krw:
            log(f"[SKIP SELL] coinone {market} below min notional={notional}")
            return False
        return self.execute_limit_order_with_retry(market, "SELL", price, qty, notional, reason)

    def rebalance_buy(self, market: str, mark_price: Decimal, buy_limit_price: Decimal, current_qty: Decimal, target_notional: Decimal, cash_krw: Decimal, decision: dict, stats: dict) -> bool:
        effective = decision.get("effective_params") or self.strategy.effective_params(decision)
        if current_qty <= 0 and decision["confidence"] < self.cfg.buy_confidence_threshold:
            stats["buy_low_conf"] += 1
            log(f"[SKIP BUY] coinone {market} low confidence={decision['confidence']:.2f}")
            return False
        current_notional = current_qty * (mark_price if mark_price > 0 else buy_limit_price)
        required_target = target_notional if current_qty > 0 else max(target_notional, self.cfg.min_new_position_krw)
        gap = max(Decimal("0"), required_target - current_notional)
        spend = min(gap, cash_krw * effective["buy_budget_fraction"])
        if current_qty <= 0 and cash_krw >= self.cfg.min_new_position_krw:
            spend = max(spend, min(self.cfg.min_new_position_krw, cash_krw))
        if spend < self.cfg.min_trade_krw:
            stats["buy_below_min"] += 1
            log(f"[SKIP BUY] coinone {market} below min spend={spend} target={required_target} cash={cash_krw}")
            return False
        ok = self.submit_buy(market, buy_limit_price, spend, decision["reason"])
        if ok:
            stats["buy_submitted"] += 1
        return ok

    def rebound_buy(self, market: str, buy_limit_price: Decimal, cash_krw: Decimal, decision: dict) -> bool:
        effective = decision.get("effective_params") or self.strategy.effective_params(decision)
        spend = min(cash_krw * effective["rebound_buy_fraction"], self.cfg.min_trade_krw * Decimal("2"))
        if spend < self.cfg.min_trade_krw:
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
            if buy_budget >= self.cfg.min_trade_krw and buy_price > 0:
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
        pos = cycle_state["positions"].get(base, {"qty": Decimal("0"), "avg_buy": Decimal("0")})
        cash_krw = cycle_state["cash_krw"]
        qty = pos["qty"]
        avg_buy = pos["avg_buy"]
        equity = cycle_state["equity_krw"]
        has_position = qty > 0
        decision = self.strategy.ai_decision(market, technical, sentiment, onchain, self.rl.choose(self.strategy.context_key(technical, sentiment, onchain)), qty)
        effective = decision["effective_params"]
        log(f"[DECISION] {market} action={decision['action']} conf={decision['confidence']:.2f} reason={decision['reason']} enabled={decision['enabled_signals']} overrides={decision['parameter_overrides']}")
        if has_position and avg_buy > 0 and best_bid_v <= float(avg_buy * (Decimal("1") - effective["stop_loss_pct"])):
            if not self.submit_sell(market, sell_limit, qty, "stop-loss"):
                stats["sell_dust"] += 1
            return
        if has_position and avg_buy > 0 and best_bid_v >= float(avg_buy * (Decimal("1") + effective["take_profit_pct"])):
            partial_qty = qty / Decimal("2")
            if not self.submit_sell(market, sell_limit, partial_qty, "take-profit-partial"):
                stats["sell_dust"] += 1
        target_notional = equity * self.cfg.max_exposure_per_market * Decimal(str(decision["target_risk"])) * effective["max_exposure_multiplier"]
        if decision["action"] == "BUY":
            stats["buy_signal"] += 1
            self.rebalance_buy(market, mark_price or buy_limit, buy_limit, qty, target_notional, cash_krw, decision, stats)
            return
        if decision["action"] == "SELL":
            if has_position:
                if not self.submit_sell(market, sell_limit, qty, decision["reason"]):
                    stats["sell_dust"] += 1
            else:
                stats["sell_unheld"] += 1
                if not self.cfg.suppress_unheld_sell_logs:
                    log(f"[SKIP SELL] coinone {market} no position")
            return
        stats["buy_nonbuy_action"] += 1
        if self.cfg.strong_hold_buy_enabled and not has_position:
            strong_hold = decision["confidence"] >= self.cfg.strong_hold_buy_confidence and (technical.get("trend_up") or technical.get("rebound_signal"))
            if strong_hold:
                synthetic = dict(decision)
                synthetic["reason"] = f"{decision['reason']} | strong-hold-buy"
                self.rebalance_buy(market, mark_price or buy_limit, buy_limit, qty, target_notional, cash_krw, synthetic, stats)
                return
        if has_position and decision["use_grid"] and decision["enabled_signals"].get("technical", True):
            self.place_grid(market, orderbook, cash_krw, qty, decision)
        elif (not has_position) and decision["use_rebound"] and decision["enabled_signals"].get("rebound", True):
            self.rebound_buy(market, buy_limit, cash_krw, decision)

    def run_cycle(self) -> None:
        self.refresh_target_markets(force=False)
        self.sync_pending_orders()
        sentiment = self.sentiment.score()
        onchain = self.onchain.combined()
        balances = self.coinone.get_balances()
        cash, positions = self.build_position_map(balances)
        equity, _, _, _ = self.estimate_equity(balances)
        cycle_state = {"balances": balances, "cash_krw": cash, "positions": positions, "equity_krw": equity}
        stats = {"sell_unheld": 0, "sell_dust": 0, "buy_signal": 0, "buy_submitted": 0, "buy_below_min": 0, "buy_low_conf": 0, "buy_nonbuy_action": 0, "market_skipped_short_history": 0}
        log(f"[CYCLE START] sentiment={sentiment['summary']} onchain={onchain['combined_score']:.3f} markets={len(self.cfg.target_markets)}")
        for market in self.cfg.target_markets:
            try:
                self.handle_market(market, sentiment, onchain, cycle_state, stats)
            except Exception as exc:
                log(f"[MARKET ERROR] {market}: {type(exc).__name__}: {exc}")
            finally:
                self.maybe_sync_pending_orders()
        self.maybe_sync_pending_orders(force=True)
        snapshot = self.capture_snapshot(balances)
        log("[CYCLE SUMMARY] "
              f"sell_unheld_suppressed={stats['sell_unheld']} sell_dust_skipped={stats['sell_dust']} "
              f"buy_signals={stats['buy_signal']} buy_submitted={stats['buy_submitted']} "
              f"buy_below_min={stats['buy_below_min']} buy_low_conf={stats['buy_low_conf']} nonbuy_decisions={stats['buy_nonbuy_action']} short_history_skipped={stats['market_skipped_short_history']}")
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


def main() -> None:
    cfg = Config.from_env()
    bot = CoinoneOnlyAIBot(cfg)
    bot.run_forever()


if __name__ == "__main__":
    main()
