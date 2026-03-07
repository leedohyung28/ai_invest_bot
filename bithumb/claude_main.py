#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bithumb Spot Trading Bot (single-file, refactored)

Goal of this rewrite:
- Make the bot *actually trade* instead of getting stuck in HOLD loops.
- Turn previously "hard rejects" (stop_too_tight / no_pattern_match / ML filter) into *soft scoring* where possible.
- Replace unrealistic liquidity thresholds with a universe-selection + small, sane hard checks.
- Keep safety: position sizing by stop distance, global kill switch, per-coin cooldown.

Notes from your logs (2026-03-05):
- HOLD reasons dominated by: stop_too_tight, low_volume, no_pattern_match
- GATE rejects dominated by: ML_filter_pullback_ema20 and hard_block low_volume / low_trade_value
So we address those explicitly.
Run:
    python main.py --dry-run
    python main.py --live
"""
from __future__ import annotations

import argparse
import base64
import copy
import dataclasses
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
import difflib
import hashlib
import json
import logging
import math
import os
import random
import sqlite3
import sys
import threading
import time
import uuid
import urllib.parse
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
import jwt
from dotenv import load_dotenv
try:
    from urllib3.util.retry import Retry
except Exception:
    Retry = None  # type: ignore[assignment]

load_dotenv()

# ============================================================
# Logging
# ============================================================

LOG = logging.getLogger("TradingBot")
LOG.setLevel(logging.DEBUG)

_FMT = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

_console = logging.StreamHandler(sys.stdout)
_console.setLevel(logging.INFO)
_console.setFormatter(_FMT)
LOG.addHandler(_console)

_current_log_file: Optional[str] = None
_file_handler: Optional[logging.FileHandler] = None
_EMAIL_SEND_LOCK = threading.Lock()

GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")
TARGET_EMAIL = os.getenv("TARGET_EMAIL")
EMAIL_INTERVAL_SEC = int(os.getenv("EMAIL_INTERVAL_SEC", str(3 * 60 * 60)))
PAPER_INITIAL_KRW = float(os.getenv("PAPER_INITIAL_KRW") or os.getenv("INITIAL_KRW") or "200000")
INITIAL_CAPITAL_KRW = float(os.getenv("INITIAL_CAPITAL_KRW") or os.getenv("PAPER_INITIAL_KRW") or os.getenv("INITIAL_KRW") or "200000")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_MODEL_FAST = os.getenv("OLLAMA_MODEL_FAST", OLLAMA_MODEL)
OLLAMA_TIMEOUT_SEC = int(os.getenv("OLLAMA_TIMEOUT_SEC", "20"))
OLLAMA_EMAIL_SUMMARY = os.getenv("OLLAMA_EMAIL_SUMMARY", "1").strip().lower() in {"1", "true", "yes", "on"}

DEFAULT_EXCLUDED_SYMBOLS = {"BTC", "ETH"}

def normalize_symbol(value: Any) -> str:
    s = str(value or "").strip().upper()
    if not s:
        return ""
    if "-" in s:
        s = s.split("-")[-1]
    return s

def normalize_excluded_symbols(values: Optional[Iterable[Any]] = None) -> set:
    out = set(DEFAULT_EXCLUDED_SYMBOLS)
    if values:
        for v in values:
            sym = normalize_symbol(v)
            if sym:
                out.add(sym)
    return out

def market_is_excluded(market: Any, excluded_symbols: Optional[Iterable[Any]] = None) -> bool:
    return normalize_symbol(market) in normalize_excluded_symbols(excluded_symbols)


def _new_log_filename() -> str:
    return datetime.now().strftime("%Y_%m_%d_%H_%M.log")

def setup_logger_file() -> str:
    global _current_log_file, _file_handler
    newf = _new_log_filename()
    if _file_handler:
        try:
            LOG.removeHandler(_file_handler)
        except Exception:
            pass
        try:
            _file_handler.close()
        except Exception:
            pass
    _current_log_file = newf
    _file_handler = logging.FileHandler(newf, encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(_FMT)
    LOG.addHandler(_file_handler)
    LOG.info(f"새 로그 파일 생성: {newf}")
    return newf

setup_logger_file()

# ============================================================
# Email (optional)
# ============================================================

def _send_email_with_attachment(subject: str, body: str, file_to_send: str):
    if not (GMAIL_USER and GMAIL_PASSWORD and TARGET_EMAIL):
        LOG.debug("이메일 환경변수가 없어 전송을 건너뜁니다.")
        return

    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.mime.base import MIMEBase
    from email import encoders

    msg = MIMEMultipart()
    msg["From"] = GMAIL_USER
    msg["To"] = TARGET_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with open(file_to_send, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(file_to_send)}")
        msg.attach(part)

    server = smtplib.SMTP("smtp.gmail.com", 587)
    try:
        server.starttls()
        server.login(GMAIL_USER, GMAIL_PASSWORD)
        server.send_message(msg)
    finally:
        server.quit()

def send_log_report_once(get_report_fn, *, rotate_log: bool = False, subject_suffix: str = "") -> bool:
    if not (GMAIL_USER and GMAIL_PASSWORD and TARGET_EMAIL):
        return False
    with _EMAIL_SEND_LOCK:
        if not _current_log_file or not os.path.exists(_current_log_file):
            LOG.debug("[EMAIL] 전송할 로그 파일이 없습니다.")
            return False
        subject, body = get_report_fn()
        if subject_suffix:
            subject = f"{subject} | {subject_suffix}"
        _send_email_with_attachment(subject, body, _current_log_file)
        if rotate_log:
            setup_logger_file()
        return True

def email_log_rotation_scheduler(get_report_fn):
    """
    Every EMAIL_INTERVAL_SEC:
      - email the current log
      - rotate to a new log file
    """
    while True:
        time.sleep(EMAIL_INTERVAL_SEC)
        try:
            send_log_report_once(get_report_fn, rotate_log=True, subject_suffix="scheduled")
        except Exception as e:
            LOG.error(f"[EMAIL] 전송/로테이션 실패: {e}")

# ============================================================
# Config + DB persistence (fixes AIConfig.load_from_db issue)
# ============================================================

@dataclass
class BotParams:
    # portfolio / sizing
    max_positions: int = 5
    max_coin_weight: float = 0.35         # per-position max allocation (fraction of equity)
    max_total_allocation: float = 0.95    # total allocation cap
    risk_per_trade: float = 0.01          # 1% of equity risked per trade (by stop distance)
    min_order_krw: int = 5000             # Bithumb KRW min order; see Bithumb notice (2024-02-19)
    fee_rate: float = 0.0004              # estimate, adjust to your tier

    # stop / take-profit
    atr_period: int = 14
    atr_mult_stop: float = 2.2
    min_stop_pct: float = 0.004           # 0.4% minimum stop distance
    max_stop_pct: float = 0.10            # skip if stop would be too wide
    max_stop_pct_relaxed: float = 0.14    # relaxed mode widens stop cap instead of rejecting volatile leaders
    take_profit_pct: float = 0.02
    trailing_start_pct: float = 0.012
    trailing_atr_mult: float = 1.8
    time_stop_hours: float = 6.0

    # cycle
    cycle_seconds: int = 300
    candle_timeframe: str = "15m"         # Bithumb public candlestick supports 15m

    # universe selection
    universe_size: int = 25               # top N by 24h trade value
    exclude: List[str] = field(default_factory=lambda: ["BTC", "ETH"])  # always exclude manual assets from bot universe/reporting
    manage_unmanaged_live_holdings: bool = True  # adopt non-excluded live holdings into bot management
    adopt_holdings_min_value_krw: int = 5000       # ignore dust below practical sell threshold
    adopt_take_profit_immediate: bool = True       # if adopted holding already exceeds TP, sell immediately

@dataclass
class FilterParams:
    # liquidity / sanity checks
    min_trade_value_ma_krw: float = 1_200_000.0  # avg trade value per 15m candle
    min_vol_ratio_hard: float = 0.06             # only reject truly dead candles
    min_vol_ratio_soft: float = 0.50             # soft target; affects score

    # gating
    min_signal_score: float = 0.58               # base min score to enter
    min_signal_score_relaxed: float = 0.52       # when stagnation breaker triggers
    stagnation_streak_to_relax: int = 20         # cycles without fills -> relax
    defensive_loss_trigger: int = 7              # losses in last 10 trades -> defensive mode

@dataclass
class MLParams:
    enabled: bool = True
    hard_reject_below: float = 0.12      # never trade if predicted win prob < this
    soft_floor: float = 0.35             # below this, apply penalty but not hard reject
    min_trades_to_train: int = 50
    min_validation_accuracy_for_hard_reject: float = 0.64
    min_samples_for_hard_reject: int = 600

@dataclass
class AIConfig:
    params: BotParams = field(default_factory=BotParams)
    filters: FilterParams = field(default_factory=FilterParams)
    ml: MLParams = field(default_factory=MLParams)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AIConfig":
        d = d or {}
        return cls(
            params=BotParams(**(d.get("params") or {})),
            filters=FilterParams(**(d.get("filters") or {})),
            ml=MLParams(**(d.get("ml") or {})),
        )

    @classmethod
    def load_from_db(cls, db_path: str, key: str = "ai_config") -> "AIConfig":
        """Fix: the original crash came from missing load_from_db; this implementation is robust."""
        try:
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
            con.commit()
            cur.execute("SELECT v FROM meta WHERE k=?", (key,))
            row = cur.fetchone()
            con.close()
            if not row or not row[0]:
                return cls()
            return cls.from_dict(json.loads(row[0]))
        except Exception as e:
            LOG.warning(f"[CFG] DB 로드 실패 → 기본값 사용: {e}")
            return cls()

    def save_to_db(self, db_path: str, key: str = "ai_config"):
        try:
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
            cur.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)", (key, json.dumps(self.to_dict(), ensure_ascii=False)))
            con.commit()
            con.close()
        except Exception as e:
            LOG.error(f"[CFG] DB 저장 실패: {e}")

# ============================================================
# SQLite DB (candles/trades/positions/meta)
# ============================================================

class DB:
    def __init__(self, path: str):
        self.path = path
        self._init()

    @staticmethod
    def _table_columns(cur: sqlite3.Cursor, table: str) -> set:
        return {r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()}

    def _ensure_columns(self, cur: sqlite3.Cursor, table: str, cols: Dict[str, str]):
        existing = self._table_columns(cur, table)
        for col, ddl in cols.items():
            if col not in existing:
                cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {ddl}")

    def _init(self):
        con = sqlite3.connect(self.path)
        cur = con.cursor()

        cur.execute("""
        CREATE TABLE IF NOT EXISTS candles (
            ts INTEGER,
            market TEXT,
            timeframe TEXT,
            open REAL, high REAL, low REAL, close REAL,
            volume REAL,
            trade_value REAL,
            PRIMARY KEY (ts, market, timeframe)
        )
        """)
        cur.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
        cur.execute("""
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            ts INTEGER PRIMARY KEY,
            equity_krw REAL NOT NULL,
            available_krw REAL NOT NULL,
            position_count INTEGER NOT NULL,
            note TEXT
        )
        """)

        cur.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            market TEXT PRIMARY KEY,
            entry_price REAL,
            entry_time INTEGER,
            size REAL,
            stop_loss REAL,
            direction TEXT,
            entry_fee REAL,
            signal_type TEXT,
            score REAL,
            order_id TEXT,
            rsi_at_entry REAL,
            vol_ratio_at_entry REAL,
            tv_ma_at_entry REAL,
            ml_prob REAL
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market TEXT,
            direction TEXT,
            entry_price REAL,
            exit_price REAL,
            size REAL,
            entry_time INTEGER,
            exit_time INTEGER,
            pnl REAL,
            total_fee REAL,
            holding_hours REAL,
            exit_reason TEXT,
            signal_type TEXT,
            score REAL,
            rsi_at_entry REAL,
            vol_ratio_at_entry REAL,
            tv_ma_at_entry REAL,
            ml_prob REAL
        )
        """)

        # Auto-migrate older DB schema safely (e.g., legacy bithumb_swing.db).
        self._ensure_columns(cur, "positions", {
            "signal_type": "TEXT DEFAULT 'unknown'",
            "score": "REAL DEFAULT 0",
            "order_id": "TEXT",
            "rsi_at_entry": "REAL DEFAULT 50.0",
            "vol_ratio_at_entry": "REAL DEFAULT 1.0",
            "tv_ma_at_entry": "REAL DEFAULT 0.0",
            "ml_prob": "REAL",
        })
        self._ensure_columns(cur, "trades", {
            "signal_type": "TEXT DEFAULT 'unknown'",
            "score": "REAL DEFAULT 0",
            "rsi_at_entry": "REAL DEFAULT 50.0",
            "vol_ratio_at_entry": "REAL DEFAULT 1.0",
            "tv_ma_at_entry": "REAL DEFAULT 0.0",
            "ml_prob": "REAL",
        })
        pos_cols = self._table_columns(cur, "positions")
        if "confidence" in pos_cols and "score" in pos_cols:
            cur.execute("""
                UPDATE positions
                SET score = confidence
                WHERE confidence IS NOT NULL AND (score IS NULL OR score = 0)
            """)
        trade_cols = self._table_columns(cur, "trades")
        if "confidence" in trade_cols and "score" in trade_cols:
            cur.execute("""
                UPDATE trades
                SET score = confidence
                WHERE confidence IS NOT NULL AND (score IS NULL OR score = 0)
            """)
        con.commit()
        con.close()

    def put_candles(self, rows: List[Tuple]):
        if not rows:
            return
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.executemany("""
            INSERT OR REPLACE INTO candles
            (ts, market, timeframe, open, high, low, close, volume, trade_value)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)
        con.commit()
        con.close()

    def get_candles(self, market: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
        con = sqlite3.connect(self.path)
        df = pd.read_sql_query("""
            SELECT * FROM candles
            WHERE market=? AND timeframe=?
            ORDER BY ts DESC LIMIT ?
        """, con, params=(market, timeframe, limit))
        con.close()
        if df.empty:
            return df
        return df.sort_values("ts").reset_index(drop=True)

    def set_meta(self, k: str, v: str):
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute("INSERT OR REPLACE INTO meta (k,v) VALUES (?,?)", (k, v))
        con.commit()
        con.close()

    def get_meta(self, k: str, default: Optional[str] = None) -> Optional[str]:
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute("SELECT v FROM meta WHERE k=?", (k,))
        row = cur.fetchone()
        con.close()
        return row[0] if row else default

    def upsert_position(self, pos: dict):
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute("""
        INSERT OR REPLACE INTO positions
        (market, entry_price, entry_time, size, stop_loss, direction, entry_fee, signal_type, score,
         order_id, rsi_at_entry, vol_ratio_at_entry, tv_ma_at_entry, ml_prob)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pos["market"], pos["entry_price"], pos["entry_time"], pos["size"], pos["stop_loss"],
            pos["direction"], pos.get("entry_fee", 0.0), pos.get("signal_type", "unknown"), pos.get("score", 0.0),
            pos.get("order_id", ""),
            pos.get("rsi_at_entry", 50.0),
            pos.get("vol_ratio_at_entry", 1.0),
            pos.get("tv_ma_at_entry", 0.0),
            pos.get("ml_prob", None),
        ))
        con.commit()
        con.close()

    def delete_position(self, market: str):
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute("DELETE FROM positions WHERE market=?", (market,))
        con.commit()
        con.close()

    def get_positions(self) -> pd.DataFrame:
        con = sqlite3.connect(self.path)
        df = pd.read_sql_query("SELECT * FROM positions", con)
        con.close()
        return df

    def insert_trade(self, t: dict):
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute("""
        INSERT INTO trades
        (market, direction, entry_price, exit_price, size, entry_time, exit_time, pnl, total_fee, holding_hours,
         exit_reason, signal_type, score, rsi_at_entry, vol_ratio_at_entry, tv_ma_at_entry, ml_prob)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            t["market"], t["direction"], t["entry_price"], t["exit_price"], t["size"],
            t["entry_time"], t["exit_time"], t["pnl"], t["total_fee"], t["holding_hours"],
            t.get("exit_reason",""), t.get("signal_type","unknown"), t.get("score",0.0),
            t.get("rsi_at_entry",50.0), t.get("vol_ratio_at_entry",1.0), t.get("tv_ma_at_entry",0.0),
            t.get("ml_prob", None)
        ))
        con.commit()
        con.close()

    def recent_trades(self, limit: int = 200) -> pd.DataFrame:
        con = sqlite3.connect(self.path)
        df = pd.read_sql_query("""
            SELECT * FROM trades
            ORDER BY id DESC LIMIT ?
        """, con, params=(limit,))
        con.close()
        if df.empty:
            return df
        return df.sort_values("id").reset_index(drop=True)

    def insert_equity_snapshot(self, ts: int, equity_krw: float, available_krw: float, position_count: int, note: str = ""):
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO equity_snapshots
            (ts, equity_krw, available_krw, position_count, note)
            VALUES (?, ?, ?, ?, ?)
        """, (int(ts), float(equity_krw), float(available_krw), int(position_count), note))
        con.commit()
        con.close()

    def get_last_equity_snapshot_before(self, ts: int) -> Optional[dict]:
        con = sqlite3.connect(self.path)
        cur = con.cursor()
        cur.execute("""
            SELECT ts, equity_krw, available_krw, position_count, note
            FROM equity_snapshots
            WHERE ts <= ?
            ORDER BY ts DESC
            LIMIT 1
        """, (int(ts),))
        row = cur.fetchone()
        con.close()
        if not row:
            return None
        return {
            "ts": int(row[0]),
            "equity_krw": float(row[1]),
            "available_krw": float(row[2]),
            "position_count": int(row[3]),
            "note": row[4] or "",
        }

    def recent_equity_snapshots(self, limit: int = 36) -> pd.DataFrame:
        con = sqlite3.connect(self.path)
        df = pd.read_sql_query("""
            SELECT * FROM equity_snapshots
            ORDER BY ts DESC LIMIT ?
        """, con, params=(limit,))
        con.close()
        if df.empty:
            return df
        return df.sort_values("ts").reset_index(drop=True)

# ============================================================
# Bithumb API (Public + Private JWT)
# ============================================================

def build_http_session() -> requests.Session:
    sess = requests.Session()
    if Retry is None:
        return sess
    retry_kwargs = dict(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.35,
        status_forcelist=(429, 500, 502, 503, 504),
        raise_on_status=False,
    )
    try:
        retry = Retry(allowed_methods=frozenset({"GET", "POST", "DELETE"}), **retry_kwargs)
    except TypeError:
        # urllib3<1.26 compatibility
        retry = Retry(method_whitelist=frozenset({"GET", "POST", "DELETE"}), **retry_kwargs)  # type: ignore[call-arg]
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

class BithumbPublic:
    def __init__(self, sess: Optional[requests.Session] = None):
        self.base = "https://api.bithumb.com"
        self.sess = sess or build_http_session()

    def ticker_all_krw(self) -> Optional[dict]:
        url = f"{self.base}/public/ticker/ALL_KRW"
        try:
            r = self.sess.get(url, timeout=10)
            r.raise_for_status()
            j = r.json()
            if j.get("status") != "0000":
                return None
            return j.get("data")
        except Exception as e:
            LOG.error(f"[PUBLIC] ticker ALL failed: {e}")
            return None

    def ticker_one(self, symbol: str, payment: str = "KRW") -> Optional[dict]:
        url = f"{self.base}/public/ticker/{symbol}_{payment}"
        try:
            r = self.sess.get(url, timeout=10)
            r.raise_for_status()
            j = r.json()
            if j.get("status") != "0000":
                return None
            return j.get("data")
        except Exception as e:
            LOG.error(f"[PUBLIC] ticker {symbol} failed: {e}")
            return None

    def candlestick(self, symbol: str, interval: str = "15m", payment: str = "KRW") -> Optional[List[list]]:
        # Official endpoint: /public/candlestick/{order_currency}_{payment_currency}/{chart_intervals}
        # Supports: 1m,3m,5m,10m,15m,30m,1h,4h,6h,12h,24h,1w,1mm (per Bithumb API docs search snippet)
        url = f"{self.base}/public/candlestick/{symbol}_{payment}/{interval}"
        try:
            r = self.sess.get(url, timeout=10)
            r.raise_for_status()
            j = r.json()
            if j.get("status") != "0000":
                return None
            return j.get("data")
        except Exception as e:
            LOG.error(f"[PUBLIC] candle {symbol} {interval} failed: {e}")
            return None

class BithumbPrivate:
    """
    Bithumb OpenAPI (JWT Authorization Bearer) implementation.
    Docs: '인증 헤더 만들기' shows payload fields and query_hash method.
    """
    def __init__(self, api_key: str, secret_key: str, sess: Optional[requests.Session] = None):
        self.base = "https://api.bithumb.com"
        self.api_key = api_key
        self.secret_key = secret_key
        self.sess = sess or build_http_session()

    def _jwt(self, params: Optional[dict] = None) -> str:
        payload = {
            "access_key": self.api_key,
            "nonce": str(uuid.uuid4()),
            "timestamp": round(time.time() * 1000),
        }
        if params:
            query = urllib.parse.urlencode(params, doseq=True).encode("utf-8")
            query_hash = hashlib.sha512(query).hexdigest()
            payload["query_hash"] = query_hash
            payload["query_hash_alg"] = "SHA512"
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        return f"Bearer {token}"

    def _request(self, method: str, path: str, params: Optional[dict] = None, query: bool = False) -> Optional[dict]:
        url = f"{self.base}{path}"
        headers = {"Authorization": self._jwt(params), "Content-Type": "application/json"}
        try:
            if method.upper() == "GET":
                r = self.sess.get(url, params=params if query else None, headers=headers, timeout=10)
            elif method.upper() == "DELETE":
                r = self.sess.delete(url, params=params if query else None, headers=headers, timeout=10)
            else:  # POST/PUT
                r = self.sess.request(method.upper(), url, json=params or {}, headers=headers, timeout=10)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            LOG.error(f"[PRIVATE] {method} {path} failed: {e} | resp={getattr(getattr(e, 'response', None), 'text', '')}")
            return None

    def accounts(self) -> Optional[list]:
        # GET /v1/accounts (docs)
        return self._request("GET", "/v1/accounts", None, query=True)

    def order_chance(self, market: str) -> Optional[dict]:
        return self._request("GET", "/v1/orders/chance", {"market": market}, query=True)

    def order_get(self, uuid_: str) -> Optional[dict]:
        return self._request("GET", "/v1/order", {"uuid": uuid_}, query=True)

    def orders_list(self, market: str, state: str = "wait", limit: int = 20, page: int = 1, order_by: str = "desc") -> Optional[list]:
        return self._request("GET", "/v1/orders", {"market": market, "state": state, "limit": limit, "page": page, "order_by": order_by}, query=True)

    def place_order_v1(self, body: dict) -> Optional[dict]:
        return self._request("POST", "/v1/orders", body, query=False)

    def cancel_order_v1(self, uuid_: str) -> Optional[dict]:
        return self._request("DELETE", "/v1/order", {"uuid": uuid_}, query=True)

    def place_order_beta(self, body: dict) -> Optional[dict]:
        # POST /v2/orders (docs)
        return self._request("POST", "/v2/orders", body, query=False)

    def cancel_order_beta(self, order_id: str) -> Optional[dict]:
        # DELETE /v2/order?order_id=... (docs)
        return self._request("DELETE", "/v2/order", {"order_id": order_id}, query=True)

# ============================================================
# Helpers: indicators / tick size / rounding
# ============================================================

def ema(series: pd.Series, span: int) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    delta = s.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, min_periods=period).mean()
    ma_down = down.ewm(alpha=1/period, min_periods=period).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period).mean()

def format_price(p: float) -> str:
    try:
        p = float(p)
    except Exception:
        return str(p)
    if p < 1:
        return f"{p:,.6f}"
    if p < 10:
        return f"{p:,.4f}"
    if p < 100:
        return f"{p:,.2f}"
    return f"{p:,.0f}"

def normalize_krw_amount(krw: float) -> int:
    return int(max(0, math.floor(float(krw))))

def format_signed_krw(v: float) -> str:
    v = float(v)
    sign = "+" if v > 0 else ""
    return f"{sign}{v:,.0f} KRW"

def format_signed_pct(v: float) -> str:
    v = float(v)
    sign = "+" if v > 0 else ""
    return f"{sign}{v * 100:.2f}%"

def safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return float(default)
        return float(v)
    except Exception:
        return float(default)

def format_order_volume(volume: float, max_dp: int = 8) -> str:
    v = max(0.0, float(volume))
    s = f"{v:.{max_dp}f}".rstrip("0").rstrip(".")
    return s if s else "0"

def get_bithumb_tick_size_krw(price: float) -> float:
    """
    Conservative tick size table.
    Note: Bithumb announced that for 100~1000 KRW range, tick changed from 0.1 to 1 KRW (2023-11-30).
    """
    p = float(price)
    if p >= 2_000_000: return 1000.0
    if p >= 1_000_000: return 500.0
    if p >= 500_000:   return 200.0
    if p >= 100_000:   return 100.0
    if p >= 10_000:    return 50.0
    if p >= 1_000:     return 10.0
    if p >= 100:       return 1.0    # changed from 0.1 -> 1 in that band
    if p >= 10:        return 1.0
    if p >= 1:         return 0.1
    if p >= 0.1:       return 0.01
    if p >= 0.01:      return 0.001
    return 0.0001

def round_to_tick(price: float) -> float:
    tick = get_bithumb_tick_size_krw(price)
    if tick <= 0:
        return float(price)
    return round(round(price / tick) * tick, 10)

def latest_vol_ratio(df: pd.DataFrame, window: int = 20) -> float:
    if df is None or df.empty:
        return 1.0
    vol = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    ma = vol.rolling(window).mean()
    v = float(vol.iloc[-1] or 0.0)
    m = float(ma.iloc[-1] or 0.0)
    if m <= 0:
        return 1.0
    return float(max(0.0, min(10.0, v / m)))

def trade_value_ma(df: pd.DataFrame, window: int = 20) -> float:
    if df is None or df.empty:
        return 0.0
    tv = pd.to_numeric(df["trade_value"], errors="coerce").fillna(0.0)
    return float(tv.rolling(window).mean().iloc[-1] or 0.0)

# ============================================================
# ML filter (optional, rewritten to be soft instead of hard SKIP at 35%)
# ============================================================

ML_FEATURE_COLS = ["signal_type_enc", "score", "rsi", "vol_ratio", "tv_ma", "hour"]

SIGNAL_TYPE_MAP = {
    "pullback_ema20": 0,
    "breakout": 1,
    "rsi_bounce": 2,
    "unknown": 3,
}

def _build_ml_df(trades: pd.DataFrame) -> Optional[pd.DataFrame]:
    if trades is None or trades.empty:
        return None
    df = trades.copy()
    df["label"] = (pd.to_numeric(df["pnl"], errors="coerce").fillna(0.0) > 0).astype(int)
    df["signal_type_enc"] = df["signal_type"].map(SIGNAL_TYPE_MAP).fillna(3).astype(int)
    df["score"] = pd.to_numeric(df.get("score", 0.0), errors="coerce").fillna(0.0)
    df["rsi"] = pd.to_numeric(df.get("rsi_at_entry", 50.0), errors="coerce").fillna(50.0)
    df["vol_ratio"] = pd.to_numeric(df.get("vol_ratio_at_entry", 1.0), errors="coerce").fillna(1.0)
    df["tv_ma"] = pd.to_numeric(df.get("tv_ma_at_entry", 0.0), errors="coerce").fillna(0.0)
    df["hour"] = pd.to_datetime(df["entry_time"], unit="s").dt.hour.astype(int)
    keep = df.dropna(subset=["label"])
    if keep.empty:
        return None
    return keep

def train_ml_model(db: DB, min_trades: int = 50) -> Optional[str]:
    """
    Train a simple model and store it in meta as base64(pickle).
    Uses time-aware split (last 20% as validation) to reduce leakage.
    """
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        import pickle

        trades = db.recent_trades(limit=1000)
        if trades.empty or len(trades) < min_trades:
            return None
        df = _build_ml_df(trades)
        if df is None or len(df) < min_trades:
            return None

        df = df.sort_values("entry_time")
        X = df[ML_FEATURE_COLS].values
        y = df["label"].values

        split = int(len(df) * 0.8)
        X_train, y_train = X[:split], y[:split]
        X_val, y_val = X[split:], y[split:]

        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)
        val_acc = float(model.score(X_val, y_val)) if len(y_val) >= 10 else float(model.score(X_train, y_train))

        payload = {
            "model": base64.b64encode(pickle.dumps(model)).decode("utf-8"),
            "trained_at": datetime.now().isoformat(),
            "val_acc": val_acc,
            "n": int(len(df)),
        }
        db.set_meta("ml_model", json.dumps(payload))
        LOG.info(f"[ML] 모델 학습 완료: n={len(df)}, val_acc={val_acc:.3f}")
        return json.dumps(payload)
    except Exception as e:
        LOG.debug(f"[ML] 학습 실패(무시): {e}")
        return None

def predict_ml_prob(db: DB, signal_type: str, score: float, rsi_v: float, vol_ratio: float, tv_ma: float) -> float:
    try:
        import pickle
        raw = db.get_meta("ml_model")
        if not raw:
            return 0.5
        payload = json.loads(raw)
        model = pickle.loads(base64.b64decode(payload["model"]))
        hour = datetime.now().hour
        X = np.array([[SIGNAL_TYPE_MAP.get(signal_type, 3), score, rsi_v, vol_ratio, tv_ma, hour]], dtype=float)
        p = float(model.predict_proba(X)[0][1])
        return max(0.0, min(1.0, p))
    except Exception:
        return 0.5

def get_ml_model_meta(db: DB) -> dict:
    try:
        raw = db.get_meta("ml_model")
        if not raw:
            return {}
        payload = json.loads(raw)
        return {
            "trained_at": payload.get("trained_at"),
            "val_acc": safe_float(payload.get("val_acc"), 0.0),
            "n": int(safe_float(payload.get("n"), 0.0)),
        }
    except Exception:
        return {}

# ============================================================
# Strategy: signal scoring (no hard "no_pattern_match")
# ============================================================

@dataclass
class Signal:
    market: str
    symbol: str
    price: float
    signal_type: str
    score: float
    rsi: float
    vol_ratio: float
    tv_ma: float
    stop_loss: float
    take_profit: float
    meta: dict = field(default_factory=dict)

def compute_stop_loss(
    entry: float,
    atr_value: float,
    cfg: AIConfig,
    relaxed: bool = False,
    max_stop_pct_override: Optional[float] = None,
    clamp_when_wide: bool = False,
) -> Tuple[Optional[float], str]:
    p = float(entry)
    atr_mult = float(cfg.params.atr_mult_stop)
    raw_dist = max(0.0, float(atr_value) * atr_mult)
    min_dist = max(p * cfg.params.min_stop_pct, 3.0 * get_bithumb_tick_size_krw(p))
    dist = max(raw_dist, min_dist)

    base_cap = cfg.params.max_stop_pct_relaxed if relaxed else cfg.params.max_stop_pct
    hard_cap = float(max_stop_pct_override if max_stop_pct_override is not None else base_cap)
    reject_cap = hard_cap * (1.25 if relaxed else 1.10)

    width = dist / max(p, 1e-12)
    if width > reject_cap:
        return None, f"stop_too_wide({width:.1%})"

    if width > hard_cap:
        if clamp_when_wide or relaxed:
            dist = p * hard_cap
            reason = f"stop_clamped({width:.1%}->{hard_cap:.1%})"
        else:
            return None, f"stop_too_wide({width:.1%})"
    else:
        reason = "ok"

    sl = round_to_tick(p - dist)
    if sl >= p:
        sl = round_to_tick(p - min_dist)
    if sl >= p * (1.0 - cfg.params.min_stop_pct * 0.5):
        sl = round_to_tick(sl - get_bithumb_tick_size_krw(sl) * 2)

    return float(sl), reason
def score_signal(df: pd.DataFrame, cfg: AIConfig) -> Optional[Tuple[str, float, dict]]:
    """
    Returns (signal_type, score, meta) or None.
    Score is 0~1.
    """
    if df is None or df.empty or len(df) < 80:
        return None
    d = df.copy()
    close = pd.to_numeric(d["close"], errors="coerce")
    high = pd.to_numeric(d["high"], errors="coerce")
    low = pd.to_numeric(d["low"], errors="coerce")
    vol = pd.to_numeric(d["volume"], errors="coerce").fillna(0.0)

    e20 = ema(close, 20)
    e60 = ema(close, 60)
    r = rsi(close, 14)
    a = atr(d, cfg.params.atr_period)

    # last values
    p = float(close.iloc[-1])
    p_prev = float(close.iloc[-2])
    e20v = float(e20.iloc[-1])
    e60v = float(e60.iloc[-1])
    rsiv = float(r.iloc[-1])
    atrv = float(a.iloc[-1] if not math.isnan(float(a.iloc[-1])) else 0.0)

    # volume ratio and trade value
    d["trade_value"] = pd.to_numeric(d.get("trade_value", close * vol), errors="coerce").fillna(close * vol)
    vr = latest_vol_ratio(d, 20)
    tv = trade_value_ma(d, 20)

    # basic trend score: prefer p > e60 and e20 > e60
    trend = 0.0
    if p > e60v:
        trend += 0.20
    if e20v > e60v:
        trend += 0.20
    if (p - e20v) / max(1e-9, e20v) > 0:
        trend += 0.05

    # volume score (soft)
    vol_score = 0.0
    if vr >= cfg.filters.min_vol_ratio_soft:
        vol_score = 0.15
    elif vr >= cfg.filters.min_vol_ratio_hard:
        vol_score = 0.08
    else:
        vol_score = 0.0

    # --- Candidate 1: Pullback EMA20 rebound ---
    # conditions: in uptrend-ish, price near/under ema20 and bouncing
    pullback_score = 0.0
    pullback_ok = False
    dist_to_e20 = abs(p - e20v) / max(1e-9, e20v)
    if e20v > 0 and dist_to_e20 <= 0.008:  # within 0.8%
        pullback_score += 0.25
        pullback_ok = True
    if p > p_prev:
        pullback_score += 0.08
    if rsiv < 60:
        pullback_score += 0.05

    # --- Candidate 2: Breakout ---
    breakout_score = 0.0
    lookback = 20
    recent_high = float(high.tail(lookback).max())
    if recent_high > 0 and p >= recent_high * 0.998:
        breakout_score += 0.25
        if p > p_prev:
            breakout_score += 0.05
        if vr >= 1.0:
            breakout_score += 0.10

    # --- Candidate 3: RSI bounce (mean reversion) ---
    rsi_score = 0.0
    if rsiv <= 34:
        rsi_score += 0.22
        if p > p_prev:
            rsi_score += 0.08
        if p > e20v * 0.995:
            rsi_score += 0.05

    # Pick best
    candidates = [
        ("pullback_ema20", pullback_score),
        ("breakout", breakout_score),
        ("rsi_bounce", rsi_score),
    ]
    best_type, best_pat = max(candidates, key=lambda x: x[1])

    # If none of patterns "fire", still allow a small trend+volume setup
    base = trend + vol_score
    score = base + best_pat

    # mild penalties
    # avoid very overbought
    if rsiv > 78:
        score -= 0.15
    # avoid very weak trend unless it is rsi bounce
    if best_type != "rsi_bounce" and p < e60v * 0.995:
        score -= 0.12

    meta = {
        "trend_score": trend,
        "vol_score": vol_score,
        "pattern_score": best_pat,
        "rsiv": rsiv,
        "atr": atrv,
        "e20": e20v,
        "e60": e60v,
        "vr": vr,
        "tv_ma": tv,
        "recent_high": recent_high,
    }

    score = float(max(0.0, min(1.0, score)))
    return best_type, score, meta

# ============================================================
# Risk state & cooldown
# ============================================================

def now_ts() -> int:
    return int(time.time())

def parse_json(s: Optional[str], default: Any):
    if not s:
        return default
    try:
        return json.loads(s)
    except Exception:
        return default

def coin_block_key(market: str) -> str:
    return f"block_until:{market}"

def get_block_until(db: DB, market: str) -> int:
    return int(float(db.get_meta(coin_block_key(market), "0") or 0))

def set_block_until(db: DB, market: str, seconds: int, reason: str):
    until = now_ts() + int(seconds)
    db.set_meta(coin_block_key(market), str(until))
    LOG.info(f"HARD_BLOCK {market} {seconds//60}min | {reason}")

def is_blocked(db: DB, market: str) -> Tuple[bool, int]:
    until = get_block_until(db, market)
    if until > now_ts():
        return True, until - now_ts()
    return False, 0

def compute_loss_stats(trades: pd.DataFrame) -> Tuple[int, int]:
    if trades is None or trades.empty:
        return 0, 0
    last10 = trades.tail(10)
    losses = int((pd.to_numeric(last10["pnl"], errors="coerce").fillna(0.0) <= 0).sum())
    return losses, int(len(last10))

# ============================================================
# Exchange abstraction: live vs dry-run (paper)
# ============================================================

class Exchange:
    def get_balances(self) -> Dict[str, float]:
        raise NotImplementedError
    def get_price_krw(self, symbol: str) -> Optional[float]:
        raise NotImplementedError
    def place_market_buy_krw(self, market: str, krw_amount: float) -> Tuple[bool, str, float]:
        """returns (ok, order_id, fee_paid_krw)"""
        raise NotImplementedError
    def place_market_sell_all(self, market: str, volume: float) -> Tuple[bool, str, float]:
        raise NotImplementedError

class PaperExchange(Exchange):
    def __init__(self, public: BithumbPublic, db: DB, fee_rate: float):
        self.public = public
        self.db = db
        self.fee_rate = fee_rate
        # paper balances in meta
        st = parse_json(db.get_meta("paper_balances"), None)
        if not st:
            st = {"KRW": float(PAPER_INITIAL_KRW)}  # default paper cash
            db.set_meta("paper_balances", json.dumps(st))

    def _load(self) -> dict:
        return parse_json(self.db.get_meta("paper_balances"), {"KRW": 0.0})
    def _save(self, st: dict):
        self.db.set_meta("paper_balances", json.dumps(st))

    def get_balances(self) -> Dict[str, float]:
        return self._load()

    def get_price_krw(self, symbol: str) -> Optional[float]:
        t = self.public.ticker_one(symbol, "KRW")
        if not t:
            return None
        try:
            return float(t["closing_price"])
        except Exception:
            return None

    def place_market_buy_krw(self, market: str, krw_amount: float) -> Tuple[bool, str, float]:
        # market like "KRW-ETH"
        symbol = market.split("-")[1]
        if market_is_excluded(market):
            return False, "", 0.0
        price = self.get_price_krw(symbol)
        if not price or price <= 0:
            return False, "", 0.0
        st = self._load()
        if st.get("KRW", 0.0) < krw_amount:
            return False, "", 0.0
        fee = krw_amount * self.fee_rate
        net = max(0.0, krw_amount - fee)
        vol = net / price
        st["KRW"] -= krw_amount
        st[symbol] = st.get(symbol, 0.0) + vol
        self._save(st)
        return True, f"paper-buy-{uuid.uuid4()}", fee

    def place_market_sell_all(self, market: str, volume: float) -> Tuple[bool, str, float]:
        symbol = market.split("-")[1]
        if market_is_excluded(market):
            return False, "", 0.0
        price = self.get_price_krw(symbol)
        if not price or price <= 0:
            return False, "", 0.0
        st = self._load()
        have = st.get(symbol, 0.0)
        if have <= 0:
            return False, "", 0.0
        vol = min(have, volume)
        gross = vol * price
        fee = gross * self.fee_rate
        net = gross - fee
        st[symbol] = have - vol
        st["KRW"] = st.get("KRW", 0.0) + net
        self._save(st)
        return True, f"paper-sell-{uuid.uuid4()}", fee

class LiveExchange(Exchange):
    def __init__(self, public: BithumbPublic, private: BithumbPrivate, fee_rate: float):
        self.public = public
        self.private = private
        self.fee_rate = fee_rate

    def get_account_rows(self) -> List[dict]:
        acc = self.private.accounts()
        if not acc:
            return []
        if isinstance(acc, dict):
            acc = acc.get("data", [])
        if not isinstance(acc, list):
            return []
        return [a for a in acc if isinstance(a, dict)]

    def get_balances(self) -> Dict[str, float]:
        acc = self.private.accounts()
        out: Dict[str, float] = {}
        if not acc:
            return out
        if isinstance(acc, dict):
            acc = acc.get("data", [])
        if not isinstance(acc, list):
            return out
        # expected shape: list of dicts with currency, balance, locked ...
        for a in acc:
            try:
                cur = a.get("currency")
                balance = float(a.get("balance", 0.0) or 0.0)
                locked = float(a.get("locked", 0.0) or 0.0)
                available = a.get("available", None)
                if available is None:
                    bal = max(0.0, balance - locked if locked > 0 else balance)
                else:
                    bal = max(0.0, float(available or 0.0))
                if cur:
                    out[cur.upper()] = bal
            except Exception:
                continue
        return out

    def get_price_krw(self, symbol: str) -> Optional[float]:
        t = self.public.ticker_one(symbol, "KRW")
        if not t:
            return None
        try:
            return float(t["closing_price"])
        except Exception:
            return None

    def place_market_buy_krw(self, market: str, krw_amount: float) -> Tuple[bool, str, float]:
        if market_is_excluded(market):
            return False, "", 0.0
        spend_krw = normalize_krw_amount(krw_amount)
        if spend_krw <= 0:
            return False, "", 0.0

        available_krw = float(self.get_balances().get("KRW", 0.0))
        if available_krw > 0:
            # Keep a small buffer for fee/latency/slippage to reduce insufficient_funds.
            live_cap = normalize_krw_amount(available_krw * (1.0 - self.fee_rate - 0.002))
            if live_cap <= 0:
                return False, "", 0.0
            spend_krw = min(spend_krw, live_cap)
        if spend_krw <= 0:
            return False, "", 0.0

        body_v1 = {
            "market": market,
            "side": "bid",
            "price": str(spend_krw),
            "ord_type": "price",
        }
        # /v1 is currently more stable for KRW market-buy by price.
        resp = self.private.place_order_v1(body_v1)
        if not resp:
            body_v2 = {
                "market": market,
                "side": "bid",
                "price": str(spend_krw),
                "order_type": "price",
            }
            resp = self.private.place_order_beta(body_v2)
        if not resp:
            return False, "", 0.0
        order_id = resp.get("uuid") or resp.get("order_id") or resp.get("id") or ""
        fee = spend_krw * self.fee_rate  # estimate; real fee needs fills endpoint
        return True, str(order_id), fee

    def place_market_sell_all(self, market: str, volume: float) -> Tuple[bool, str, float]:
        if market_is_excluded(market):
            return False, "", 0.0
        vol = max(0.0, float(volume))
        if vol <= 0:
            return False, "", 0.0
        body_v1 = {
            "market": market,
            "side": "ask",
            "volume": format_order_volume(vol),
            "ord_type": "market",
        }
        resp = self.private.place_order_v1(body_v1)
        if not resp:
            body_v2 = {
                "market": market,
                "side": "ask",
                "volume": format_order_volume(vol),
                "order_type": "market",
            }
            resp = self.private.place_order_beta(body_v2)
        if not resp:
            return False, "", 0.0
        order_id = resp.get("uuid") or resp.get("order_id") or resp.get("id") or ""
        # fee estimate from price unknown until filled; we approximate 0 for now
        return True, str(order_id), 0.0

# ============================================================
# Ollama helper (optional)
# ============================================================

class OllamaClient:
    def __init__(self, url: str, default_model: str, fast_model: str, timeout: int = 20):
        self.url = url.rstrip("/")
        self.default_model = default_model
        self.fast_model = fast_model
        self.timeout = timeout
        self.sess = build_http_session()

    def chat(self, messages: List[dict], fast: bool = False) -> Optional[str]:
        model = self.fast_model if fast else self.default_model
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": "15m",
        }
        try:
            r = self.sess.post(self.url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            j = r.json()
            msg = j.get("message") or {}
            content = (msg.get("content") or "").strip()
            return content or None
        except Exception as e:
            LOG.debug(f"[OLLAMA] 호출 실패(무시): {e}")
            return None

# ============================================================
# Core bot
# ============================================================

class TradingBot:
    def __init__(self, db_path: str, dry_run: bool = True):
        self.db = DB(db_path)
        self.cfg = AIConfig.load_from_db(db_path)
        self.public = BithumbPublic()
        self.dry_run = dry_run

        api_key = os.getenv("BITHUMB_API_KEY")
        secret = os.getenv("BITHUMB_SECRET_KEY")

        if not dry_run and api_key and secret:
            self.private = BithumbPrivate(api_key, secret)
            self.ex = LiveExchange(self.public, self.private, self.cfg.params.fee_rate)
            LOG.info("[MODE] LIVE trading enabled")
        else:
            self.private = None
            self.ex = PaperExchange(self.public, self.db, self.cfg.params.fee_rate)
            LOG.info("[MODE] DRY_RUN (paper) enabled")

        self.hold_counter = Counter()
        self.gate_reject_counter = Counter()
        self.no_fill_streak = int(self.db.get_meta("no_fill_streak", "0") or 0)
        self._shutdown_email_sent = False
        self._last_audit_signature = ""
        self.ollama = OllamaClient(OLLAMA_URL, OLLAMA_MODEL, OLLAMA_MODEL_FAST, OLLAMA_TIMEOUT_SEC)
        self._normalize_runtime_config()
        self._ensure_reporting_baseline()
        self.record_equity_snapshot(note="startup")

    def _normalize_runtime_config(self):
        cfg = self.cfg
        p = cfg.params
        f = cfg.filters
        m = cfg.ml

        p.max_positions = 4
        p.max_coin_weight = 0.28
        p.max_total_allocation = 0.82
        p.risk_per_trade = 0.0085
        p.atr_mult_stop = 2.0
        p.max_stop_pct = 0.085
        p.max_stop_pct_relaxed = 0.115
        p.take_profit_pct = 0.022
        p.trailing_start_pct = 0.010
        p.trailing_atr_mult = 1.6
        p.time_stop_hours = 5.0
        p.exclude = sorted(normalize_excluded_symbols(getattr(p, "exclude", [])))

        f.min_trade_value_ma_krw = 900_000.0
        f.min_vol_ratio_hard = 0.05
        f.min_vol_ratio_soft = 0.35
        f.min_signal_score = 0.56
        f.min_signal_score_relaxed = 0.50
        f.stagnation_streak_to_relax = 16

        m.hard_reject_below = 0.10
        m.soft_floor = 0.33
        m.min_validation_accuracy_for_hard_reject = 0.63
        m.min_samples_for_hard_reject = 500

        LOG.info(
            "[CFG] runtime tuning 적용: max_stop_pct=%.3f, max_stop_pct_relaxed=%.3f, min_trade_value_ma_krw=%.1fK, min_vol_ratio_hard=%.2f, min_signal_score=%.2f, min_signal_score_relaxed=%.2f, ml.hard_reject_below=%.2f, exclude=%s"
            % (
                p.max_stop_pct,
                p.max_stop_pct_relaxed,
                f.min_trade_value_ma_krw / 1000.0,
                f.min_vol_ratio_hard,
                f.min_signal_score,
                f.min_signal_score_relaxed,
                m.hard_reject_below,
                ",".join(sorted(p.exclude)),
            )
        )

    def _ensure_reporting_baseline(self):
        if self.db.get_meta("initial_capital_krw") is None:
            self.db.set_meta("initial_capital_krw", str(float(INITIAL_CAPITAL_KRW)))
        if self.db.get_meta("report_baseline_ts") is None:
            self.db.set_meta("report_baseline_ts", str(now_ts()))

    def excluded_symbols(self) -> set:
        return normalize_excluded_symbols(getattr(self.cfg.params, "exclude", []))

    def is_excluded_symbol(self, symbol: Any) -> bool:
        return normalize_symbol(symbol) in self.excluded_symbols()

    def is_excluded_market(self, market: Any) -> bool:
        return market_is_excluded(market, self.excluded_symbols())

    def _filter_positions_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "market" not in df.columns:
            return df
        excluded = self.excluded_symbols()
        mask = ~df["market"].astype(str).map(lambda m: market_is_excluded(m, excluded))
        return df.loc[mask].copy()

    def _filter_trades_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty or "market" not in df.columns:
            return df
        excluded = self.excluded_symbols()
        mask = ~df["market"].astype(str).map(lambda m: market_is_excluded(m, excluded))
        return df.loc[mask].copy()

    def _balance_asset_rows(self, min_value_krw: float = 3000.0) -> List[dict]:
        balances = self.ex.get_balances()
        rows: List[dict] = []
        for cur, amt in balances.items():
            symbol = normalize_symbol(cur)
            amount = safe_float(amt, 0.0)
            if symbol == "KRW" or amount <= 0 or self.is_excluded_symbol(symbol):
                continue
            price = safe_float(self.ex.get_price_krw(symbol), 0.0)
            value = amount * price
            if value < float(min_value_krw):
                continue
            rows.append({
                "symbol": symbol,
                "market": f"KRW-{symbol}",
                "amount": amount,
                "price": price,
                "value_krw": value,
            })
        rows.sort(key=lambda x: x["value_krw"], reverse=True)
        return rows

    def get_unmanaged_live_holdings(self, min_value_krw: float = 3000.0) -> List[dict]:
        live_rows = self._balance_asset_rows(min_value_krw=min_value_krw)
        managed_df = self._filter_positions_df(self.db.get_positions())
        managed_markets = set()
        if managed_df is not None and not managed_df.empty:
            managed_markets = {str(x).upper() for x in managed_df["market"].astype(str).tolist()}
        return [row for row in live_rows if row["market"].upper() not in managed_markets]

    def _current_stop_for_adopted_holding(self, market: str, price: float) -> float:
        df = self.fetch_and_store_candles(market, self.cfg.params.candle_timeframe)
        if df is not None and len(df) >= self.cfg.params.atr_period + 2:
            try:
                atrv = float(atr(df, self.cfg.params.atr_period).iloc[-1] or 0.0)
            except Exception:
                atrv = 0.0
            if atrv > 0:
                sl, _ = compute_stop_loss(price, atrv, self.cfg, relaxed=True)
                if sl is not None:
                    return float(sl)
        fallback_pct = min(max(self.cfg.params.min_stop_pct * 4.0, 0.03), max(0.03, self.cfg.params.max_stop_pct_relaxed * 0.70))
        return float(round_to_tick(price * (1.0 - fallback_pct)))

    def sync_unmanaged_live_holdings(self):
        if not isinstance(self.ex, LiveExchange):
            return
        if not getattr(self.cfg.params, "manage_unmanaged_live_holdings", True):
            return

        min_value_krw = float(getattr(self.cfg.params, "adopt_holdings_min_value_krw", self.cfg.params.min_order_krw) or self.cfg.params.min_order_krw)
        managed_df = self._filter_positions_df(self.db.get_positions())
        managed_markets = set()
        if managed_df is not None and not managed_df.empty:
            managed_markets = {str(x).upper() for x in managed_df["market"].astype(str).tolist()}

        adopted_count = 0
        exited_count = 0
        for row in self.ex.get_account_rows():
            symbol = normalize_symbol(row.get("currency"))
            if not symbol or symbol == "KRW" or self.is_excluded_symbol(symbol):
                continue

            market = f"KRW-{symbol}"
            if market.upper() in managed_markets:
                continue

            available = row.get("available", None)
            if available is None:
                amount = max(0.0, safe_float(row.get("balance"), 0.0) - safe_float(row.get("locked"), 0.0))
            else:
                amount = max(0.0, safe_float(available, 0.0))
            if amount <= 0:
                continue

            price = safe_float(self.ex.get_price_krw(symbol), 0.0)
            if price <= 0:
                continue
            value_krw = amount * price
            if value_krw < min_value_krw:
                continue

            avg_buy = safe_float(row.get("avg_buy_price"), 0.0)
            entry_price = avg_buy if avg_buy > 0 else price
            pnl_pct = ((price - entry_price) / entry_price) if entry_price > 0 else 0.0

            if getattr(self.cfg.params, "adopt_take_profit_immediate", True) and entry_price > 0 and pnl_pct >= self.cfg.params.take_profit_pct:
                LOG.warning(
                    f"[ADOPT_EXIT_TRY] {market} unmanaged holding immediate take_profit pnl_pct={pnl_pct:.2%} amount={amount:.8f} entry={format_price(entry_price)} price={format_price(price)}"
                )
                ok, oid, fee = self.ex.place_market_sell_all(market, amount)
                if ok:
                    self._close_position(market, entry_price, price, amount, fee, 0.0, "adopt_take_profit")
                    exited_count += 1
                else:
                    LOG.warning(
                        f"[SELL_FAIL] {market} adopt_take_profit pnl_pct={pnl_pct:.2%} size={amount:.8f}{self._order_chance_brief(market)}"
                    )
                continue

            stop_loss = self._current_stop_for_adopted_holding(market, price)
            self.db.upsert_position({
                "market": market,
                "entry_price": entry_price,
                "entry_time": now_ts(),
                "size": amount,
                "stop_loss": stop_loss,
                "direction": "long",
                "entry_fee": 0.0,
                "signal_type": "adopted_live_holding",
                "score": 0.0,
                "order_id": f"adopt-{uuid.uuid4()}",
                "rsi_at_entry": 50.0,
                "vol_ratio_at_entry": 1.0,
                "tv_ma_at_entry": 0.0,
                "ml_prob": None,
            })
            managed_markets.add(market.upper())
            adopted_count += 1
            LOG.warning(
                f"[ADOPT] {market} unmanaged holding -> managed size={amount:.8f} entry={format_price(entry_price)} price={format_price(price)} pnl_pct={pnl_pct:.2%} sl={format_price(stop_loss)}"
            )

        if adopted_count or exited_count:
            LOG.info(f"[SYNC] unmanaged holdings adopted={adopted_count} immediate_exits={exited_count}")

    def _order_chance_brief(self, market: str) -> str:
        if not isinstance(self.ex, LiveExchange) or self.private is None:
            return ""
        try:
            chance = self.private.order_chance(market) or {}
            ask = chance.get("ask_account") or {}
            bid = chance.get("bid_account") or {}
            ask_balance = ask.get("balance", "?")
            ask_locked = ask.get("locked", "?")
            bid_balance = bid.get("balance", "?")
            return f" ask_balance={ask_balance} ask_locked={ask_locked} bid_balance={bid_balance}"
        except Exception:
            return ""

    def log_position_audit(self):
        managed_df = self._filter_positions_df(self.db.get_positions())
        managed_count = 0 if managed_df.empty else len(managed_df)
        unmanaged = self.get_unmanaged_live_holdings(min_value_krw=3000.0)
        balances = self.ex.get_balances()
        available_krw = safe_float(balances.get("KRW", 0.0), 0.0)
        signature = json.dumps({
            "managed_count": managed_count,
            "available_krw": round(available_krw, 4),
            "unmanaged": [(u["market"], round(u["value_krw"], 2)) for u in unmanaged[:8]],
        }, ensure_ascii=False, sort_keys=True)
        if signature != self._last_audit_signature:
            self._last_audit_signature = signature
            if unmanaged:
                summary = [(u["market"], round(u["value_krw"])) for u in unmanaged[:8]]
                LOG.warning(
                    f"[AUDIT] managed_positions={managed_count} unmanaged_live_holdings={len(unmanaged)} available_krw={available_krw:,.0f} top={summary}"
                )
            else:
                LOG.info(f"[AUDIT] managed_positions={managed_count} unmanaged_live_holdings=0 available_krw={available_krw:,.0f}")

    def finalize_shutdown(self, reason: str = "ctrl_c"):
        if self._shutdown_email_sent:
            return
        self._shutdown_email_sent = True
        try:
            self.record_equity_snapshot(note=f"shutdown:{reason}")
        except Exception as e:
            LOG.debug(f"[SHUTDOWN] 스냅샷 기록 실패(무시): {e}")
        try:
            sent = send_log_report_once(self.email_report, rotate_log=False, subject_suffix=f"shutdown:{reason}")
            if sent:
                LOG.info(f"[EMAIL] 종료 전 마지막 리포트 전송 완료 ({reason})")
            else:
                LOG.info(f"[EMAIL] 종료 전 마지막 리포트 전송 건너뜀 ({reason})")
        except Exception as e:
            LOG.error(f"[EMAIL] 종료 전 마지막 리포트 전송 실패: {e}")

    # ---------- portfolio ----------
    def get_equity_krw(self) -> float:
        bal = self.ex.get_balances()
        krw = float(bal.get("KRW", 0.0))
        total = krw
        for cur, amt in bal.items():
            if cur == "KRW":
                continue
            if amt <= 0 or self.is_excluded_symbol(cur):
                continue
            p = self.ex.get_price_krw(cur)
            if p:
                total += amt * p
        return float(total)

    def get_allocated_krw(self) -> float:
        pos = self._filter_positions_df(self.db.get_positions())
        if pos.empty:
            return 0.0
        allocated = 0.0
        for _, row in pos.iterrows():
            allocated += float(row["entry_price"]) * float(row["size"])
        return allocated

    # ---------- universe ----------
    def universe(self) -> List[str]:
        data = self.public.ticker_all_krw()
        if not data:
            return []
        excluded = self.excluded_symbols()
        items = []
        for sym, v in data.items():
            if sym == "date":
                continue
            if normalize_symbol(sym) in excluded:
                continue
            try:
                tv24 = float(v.get("acc_trade_value_24H", 0.0))
                items.append((sym.upper(), tv24))
            except Exception:
                continue
        items.sort(key=lambda x: x[1], reverse=True)
        top = [f"KRW-{sym}" for sym, _ in items[: self.cfg.params.universe_size]]
        return top

    # ---------- data ingestion ----------
    def fetch_and_store_candles(self, market: str, timeframe: str) -> Optional[pd.DataFrame]:
        sym = market.split("-")[1]
        raw = self.public.candlestick(sym, timeframe, "KRW")
        # (symbol, interval, payment) already includes payment
        if not raw:
            return None
        rows = []
        for x in raw:
            # expected: [timestamp, open, close, high, low, volume]
            if not isinstance(x, (list, tuple)) or len(x) < 6:
                continue
            ts_ms = int(float(x[0]))
            ts = ts_ms // 1000
            o = float(x[1]); c = float(x[2]); h = float(x[3]); l = float(x[4]); v = float(x[5])
            tv = c * v
            rows.append((ts, market, timeframe, o, h, l, c, v, tv))
        if not rows:
            return None
        self.db.put_candles(rows)
        return self.db.get_candles(market, timeframe, limit=300)

    # ---------- position management ----------
    def maybe_exit_positions(self):
        pos_df = self._filter_positions_df(self.db.get_positions())
        if pos_df.empty:
            return
        now = now_ts()
        live_balances = self.ex.get_balances() if isinstance(self.ex, LiveExchange) else {}
        for _, pos in pos_df.iterrows():
            market = str(pos["market"])
            symbol = market.split("-")[1].upper()
            price = self.ex.get_price_krw(symbol)
            if not price:
                LOG.warning(f"[EXIT_SKIP] {market} 현재가 조회 실패")
                continue
            entry = float(pos["entry_price"])
            size = float(pos["size"])
            sl = float(pos["stop_loss"])
            entry_time = int(pos["entry_time"])
            direction = pos.get("direction", "long")
            sell_size = size

            if isinstance(self.ex, LiveExchange):
                have = float(live_balances.get(symbol, 0.0))
                if have <= 0:
                    LOG.warning(f"[POS_SYNC] {market} 잔고가 없어 포지션을 DB에서 정리합니다.")
                    self.db.delete_position(market)
                    continue
                if have + 1e-12 < size:
                    sell_size = have
                    LOG.warning(f"[POS_SYNC] {market} size 보정 DB={size:.8f} -> balance={sell_size:.8f}")

            holding_hours = (now - entry_time) / 3600.0
            pnl_pct = (price - entry) / entry if direction == "long" else (entry - price) / entry

            if direction == "long" and price <= sl:
                LOG.info(f"[EXIT_TRY] {market} reason=stop_loss price={format_price(price)} sl={format_price(sl)} size={sell_size:.8f}")
                ok, oid, fee = self.ex.place_market_sell_all(market, sell_size)
                if ok:
                    self._close_position(market, entry, price, sell_size, fee, holding_hours, "stop_loss")
                else:
                    LOG.warning(f"[SELL_FAIL] {market} stop_loss price={format_price(price)} sl={format_price(sl)} size={sell_size:.8f}{self._order_chance_brief(market)}")
                continue

            if pnl_pct >= self.cfg.params.take_profit_pct:
                LOG.info(f"[EXIT_TRY] {market} reason=take_profit pnl_pct={pnl_pct:.2%} threshold={self.cfg.params.take_profit_pct:.2%} size={sell_size:.8f}")
                ok, oid, fee = self.ex.place_market_sell_all(market, sell_size)
                if ok:
                    self._close_position(market, entry, price, sell_size, fee, holding_hours, "take_profit")
                else:
                    LOG.warning(f"[SELL_FAIL] {market} take_profit pnl_pct={pnl_pct:.2%} size={sell_size:.8f}{self._order_chance_brief(market)}")
                continue

            if holding_hours >= self.cfg.params.time_stop_hours:
                LOG.info(f"[EXIT_TRY] {market} reason=time_stop holding_hours={holding_hours:.2f} limit={self.cfg.params.time_stop_hours:.2f} size={sell_size:.8f}")
                ok, oid, fee = self.ex.place_market_sell_all(market, sell_size)
                if ok:
                    self._close_position(market, entry, price, sell_size, fee, holding_hours, "time_stop")
                else:
                    LOG.warning(f"[SELL_FAIL] {market} time_stop holding_hours={holding_hours:.2f} size={sell_size:.8f}{self._order_chance_brief(market)}")
                continue

            if pnl_pct >= self.cfg.params.trailing_start_pct:
                df = self.db.get_candles(market, self.cfg.params.candle_timeframe, limit=120)
                if len(df) >= self.cfg.params.atr_period + 2:
                    a = atr(df, self.cfg.params.atr_period)
                    atrv = float(a.iloc[-1] or 0.0)
                    if atrv > 0:
                        new_sl = round_to_tick(price - atrv * self.cfg.params.trailing_atr_mult)
                        if new_sl > sl:
                            self.db.upsert_position({
                                "market": market,
                                "entry_price": entry,
                                "entry_time": entry_time,
                                "size": size,
                                "stop_loss": new_sl,
                                "direction": direction,
                                "entry_fee": float(pos.get("entry_fee", 0.0)),
                                "signal_type": pos.get("signal_type", "unknown"),
                                "score": float(pos.get("score", 0.0)),
                                "order_id": pos.get("order_id", ""),
                                "rsi_at_entry": float(pos.get("rsi_at_entry", 50.0)),
                                "vol_ratio_at_entry": float(pos.get("vol_ratio_at_entry", 1.0)),
                                "tv_ma_at_entry": float(pos.get("tv_ma_at_entry", 0.0)),
                                "ml_prob": pos.get("ml_prob", None),
                            })
                            LOG.info(f"[TRAIL] {market} SL {format_price(sl)} → {format_price(new_sl)}")

    def _close_position(self, market: str, entry: float, exit_p: float, size: float, exit_fee: float, holding_hours: float, reason: str):
        # compute total fee approximately: entry fee is stored in position table
        pos_df = self.db.get_positions()
        entry_fee = 0.0
        rsi_at_entry = 50.0
        vol_ratio_at_entry = 1.0
        tv_ma_at_entry = 0.0
        ml_prob = None
        if not pos_df.empty:
            r = pos_df[pos_df["market"] == market]
            if not r.empty:
                entry_fee = float(r.iloc[0].get("entry_fee", 0.0) or 0.0)
                sig_type = r.iloc[0].get("signal_type", "unknown")
                score = float(r.iloc[0].get("score", 0.0) or 0.0)
                rsi_at_entry = float(r.iloc[0].get("rsi_at_entry", 50.0) or 50.0)
                vol_ratio_at_entry = float(r.iloc[0].get("vol_ratio_at_entry", 1.0) or 1.0)
                tv_ma_at_entry = float(r.iloc[0].get("tv_ma_at_entry", 0.0) or 0.0)
                ml_prob = r.iloc[0].get("ml_prob", None)
            else:
                sig_type = "unknown"
                score = 0.0
        else:
            sig_type = "unknown"
            score = 0.0

        pnl = (exit_p - entry) * size
        total_fee = float(entry_fee) + float(exit_fee)
        pnl_net = pnl - total_fee

        t = {
            "market": market,
            "direction": "long",
            "entry_price": entry,
            "exit_price": exit_p,
            "size": size,
            "entry_time": int(now_ts() - holding_hours * 3600),
            "exit_time": now_ts(),
            "pnl": pnl_net,
            "total_fee": total_fee,
            "holding_hours": holding_hours,
            "exit_reason": reason,
            "signal_type": sig_type,
            "score": score,
            "rsi_at_entry": rsi_at_entry,
            "vol_ratio_at_entry": vol_ratio_at_entry,
            "tv_ma_at_entry": tv_ma_at_entry,
            "ml_prob": ml_prob,
        }
        self.db.insert_trade(t)
        self.db.delete_position(market)
        LOG.info(f"[EXIT] {market} reason={reason} pnl={pnl_net:,.0f}krw (gross={pnl:,.0f}, fee={total_fee:,.0f})")

        # per-coin cooldown on loss
        if pnl_net <= 0:
            set_block_until(self.db, market, seconds=60*30, reason="loss_cooldown")

    # ---------- entry ----------
    def decide_and_enter(self):
        cfg = self.cfg
        equity = self.get_equity_krw()
        balances = self.ex.get_balances()
        krw = float(balances.get("KRW", 0.0))

        pos_df = self._filter_positions_df(self.db.get_positions())
        open_n = 0 if pos_df.empty else len(pos_df)
        if open_n >= cfg.params.max_positions:
            return

        allocated = self.get_allocated_krw()
        if equity > 0 and allocated / equity >= cfg.params.max_total_allocation:
            return

        trades = self._filter_trades_df(self.db.recent_trades(limit=200))
        losses10, n10 = compute_loss_stats(trades)
        defensive = (n10 >= 8 and losses10 >= cfg.filters.defensive_loss_trigger)
        relaxed = (self.no_fill_streak >= cfg.filters.stagnation_streak_to_relax)

        min_score = cfg.filters.min_signal_score_relaxed if relaxed else cfg.filters.min_signal_score
        if self.no_fill_streak >= 40:
            min_score -= 0.02
        if self.no_fill_streak >= 80:
            min_score -= 0.04
        if self.no_fill_streak >= 120:
            min_score -= 0.04
        min_score = max(0.44, min_score)

        risk_mult = 0.5 if defensive else 1.0
        if relaxed:
            risk_mult *= 0.6

        stop_cap = cfg.params.max_stop_pct_relaxed if relaxed else cfg.params.max_stop_pct
        ml_meta = get_ml_model_meta(self.db)
        ml_hard_reject_enabled = (
            cfg.ml.enabled
            and ml_meta.get("n", 0) >= getattr(cfg.ml, "min_samples_for_hard_reject", 600)
            and ml_meta.get("val_acc", 0.0) >= getattr(cfg.ml, "min_validation_accuracy_for_hard_reject", 0.64)
        )

        markets = self.universe()
        candidates: List[Signal] = []

        for market in markets:
            blocked, remain = is_blocked(self.db, market)
            if blocked:
                continue
            if not pos_df.empty and (pos_df["market"] == market).any():
                continue

            df = self.fetch_and_store_candles(market, cfg.params.candle_timeframe)
            if df is None or df.empty:
                self.gate_reject_counter["no_candles"] += 1
                continue

            vr = latest_vol_ratio(df, 20)
            tv = trade_value_ma(df, 20)
            liquidity_penalty = 0.0

            if tv < cfg.filters.min_trade_value_ma_krw:
                if tv < cfg.filters.min_trade_value_ma_krw * 0.35:
                    self.gate_reject_counter["hard_block: low_trade_value"] += 1
                    continue
                liquidity_penalty += 0.10

            if vr < cfg.filters.min_vol_ratio_hard:
                if vr < max(0.010, cfg.filters.min_vol_ratio_hard * 0.35):
                    self.gate_reject_counter["hard_block: low_volume"] += 1
                    continue
                liquidity_penalty += 0.06

            res = score_signal(df, cfg)
            if not res:
                self.hold_counter["no_signal"] += 1
                continue
            signal_type, score, meta = res

            if liquidity_penalty > 0:
                score = max(0.0, score - liquidity_penalty)
                meta["liquidity_penalty"] = liquidity_penalty

            symbol = market.split("-")[1].upper()
            if self.is_excluded_symbol(symbol):
                continue
            price = self.ex.get_price_krw(symbol)
            if not price:
                continue
            price = float(price)

            clamp_wide_stop = relaxed or self.no_fill_streak >= 100
            sl, sl_reason = compute_stop_loss(
                price,
                meta.get("atr", 0.0),
                cfg,
                relaxed=(relaxed or self.no_fill_streak >= 60),
                max_stop_pct_override=stop_cap,
                clamp_when_wide=clamp_wide_stop,
            )
            if sl is None:
                self.hold_counter[sl_reason] += 1
                continue
            if sl_reason.startswith("stop_clamped"):
                meta["stop_clamped"] = sl_reason
                score *= 0.97

            tp = round_to_tick(price * (1 + cfg.params.take_profit_pct))

            ml_prob = 0.5
            if cfg.ml.enabled:
                ml_prob = predict_ml_prob(
                    self.db,
                    signal_type,
                    score,
                    float(meta.get("rsiv", 50.0)),
                    float(meta.get("vr", 1.0)),
                    float(meta.get("tv_ma", 0.0)),
                )
                if ml_hard_reject_enabled and ml_prob < cfg.ml.hard_reject_below:
                    self.gate_reject_counter["ML_hard_reject"] += 1
                    continue
                if ml_prob < cfg.ml.soft_floor:
                    score *= 0.90

            if score < min_score:
                self.gate_reject_counter["score_below_min"] += 1
                continue

            candidates.append(Signal(
                market=market,
                symbol=symbol,
                price=price,
                signal_type=signal_type,
                score=score,
                rsi=float(meta.get("rsiv", 50.0)),
                vol_ratio=float(meta.get("vr", 1.0)),
                tv_ma=float(meta.get("tv_ma", 0.0)),
                stop_loss=float(sl),
                take_profit=float(tp),
                meta={"ml_prob": ml_prob, **meta},
            ))

        if not candidates:
            return

        candidates.sort(key=lambda s: (s.score, s.meta.get("ml_prob", 0.5), s.tv_ma), reverse=True)
        chosen = candidates[0]

        stop_dist = max(1e-9, chosen.price - chosen.stop_loss)
        risk_budget = equity * cfg.params.risk_per_trade * risk_mult
        raw_size = risk_budget / stop_dist

        max_pos_krw = equity * cfg.params.max_coin_weight
        size_by_alloc = max_pos_krw / chosen.price
        size = min(raw_size, size_by_alloc)

        order_krw = size * chosen.price
        if order_krw < cfg.params.min_order_krw:
            need = cfg.params.min_order_krw / chosen.price
            size = min(size_by_alloc, max(size, need))
            order_krw = size * chosen.price

        cash_cap = krw * (1.0 if relaxed else 0.95)
        if order_krw > cash_cap:
            size = cash_cap / chosen.price
            order_krw = size * chosen.price

        order_krw = float(normalize_krw_amount(order_krw))
        if isinstance(self.ex, LiveExchange):
            live_cash_cap = normalize_krw_amount(krw * (1.0 - cfg.params.fee_rate - 0.002))
            order_krw = float(min(order_krw, live_cash_cap))
            size = order_krw / chosen.price if chosen.price > 0 else 0.0

        if order_krw < cfg.params.min_order_krw or size <= 0:
            self.gate_reject_counter["insufficient_krw"] += 1
            LOG.info(f"[ORDER_SKIP] insufficient_krw usable={krw:,.0f} order={order_krw:,.0f} equity={equity:,.0f} excluded={sorted(self.excluded_symbols())}")
            return

        pre_balances = self.ex.get_balances() if isinstance(self.ex, LiveExchange) else balances
        pre_asset = float(pre_balances.get(chosen.symbol.upper(), 0.0))
        ok, oid, fee = self.ex.place_market_buy_krw(chosen.market, float(order_krw))
        if not ok:
            self.gate_reject_counter["buy_failed"] += 1
            return

        filled_size = size
        if isinstance(self.ex, LiveExchange):
            time.sleep(0.35)
            post_balances = self.ex.get_balances()
            post_asset = float(post_balances.get(chosen.symbol.upper(), pre_asset))
            delta = max(0.0, post_asset - pre_asset)
            if delta > 0:
                filled_size = delta
            else:
                LOG.warning(f"[ENTRY_SYNC] {chosen.market} 체결수량 동기화 실패, 계산수량 사용")

        self.db.upsert_position({
            "market": chosen.market,
            "entry_price": chosen.price,
            "entry_time": now_ts(),
            "size": filled_size,
            "stop_loss": chosen.stop_loss,
            "direction": "long",
            "entry_fee": fee,
            "signal_type": chosen.signal_type,
            "score": chosen.score,
            "order_id": oid,
            "rsi_at_entry": chosen.rsi,
            "vol_ratio_at_entry": chosen.vol_ratio,
            "tv_ma_at_entry": chosen.tv_ma,
            "ml_prob": chosen.meta.get("ml_prob", 0.5),
        })
        LOG.info(
            f"[ENTER] {chosen.market} type={chosen.signal_type} score={chosen.score:.2f} "
            f"ml={chosen.meta.get('ml_prob',0.5):.2f} entry={format_price(chosen.price)} "
            f"sl={format_price(chosen.stop_loss)} stop_cap={stop_cap:.1%} size={filled_size:.6f} cost≈{order_krw:,.0f}krw"
        )

        self.no_fill_streak = 0
        self.db.set_meta("no_fill_streak", str(self.no_fill_streak))

    def record_equity_snapshot(self, note: str = "cycle") -> dict:
        equity = self.get_equity_krw()
        balances = self.ex.get_balances()
        available_krw = safe_float(balances.get("KRW", 0.0), 0.0)
        positions_df = self._filter_positions_df(self.db.get_positions())
        position_count = 0 if positions_df.empty else int(len(positions_df))
        ts = now_ts()
        self.db.insert_equity_snapshot(ts, equity, available_krw, position_count, note)
        return {
            "ts": ts,
            "equity_krw": float(equity),
            "available_krw": float(available_krw),
            "position_count": int(position_count),
            "note": note,
        }

    def equity_change_report(self, current_equity: Optional[float] = None) -> dict:
        equity = float(current_equity if current_equity is not None else self.get_equity_krw())
        initial_capital = safe_float(self.db.get_meta("initial_capital_krw", str(INITIAL_CAPITAL_KRW)), INITIAL_CAPITAL_KRW)
        snap_3h = self.db.get_last_equity_snapshot_before(now_ts() - 3 * 3600)
        equity_3h = safe_float((snap_3h or {}).get("equity_krw"), equity)
        change_3h = equity - equity_3h
        change_initial = equity - initial_capital
        pct_3h = (change_3h / equity_3h) if equity_3h > 0 else 0.0
        pct_initial = (change_initial / initial_capital) if initial_capital > 0 else 0.0
        return {
            "current_equity": equity,
            "initial_capital": initial_capital,
            "equity_3h_ago": equity_3h,
            "change_3h": change_3h,
            "change_3h_pct": pct_3h,
            "change_initial": change_initial,
            "change_initial_pct": pct_initial,
            "snapshot_3h": snap_3h,
        }

    def maybe_generate_ai_email_summary(self, perf: dict, eq: dict) -> Optional[str]:
        if not OLLAMA_EMAIL_SUMMARY:
            return None
        prompt = {
            "equity": round(eq.get("current_equity", 0.0), 2),
            "change_3h": round(eq.get("change_3h", 0.0), 2),
            "change_3h_pct": round(eq.get("change_3h_pct", 0.0) * 100, 4),
            "change_initial": round(eq.get("change_initial", 0.0), 2),
            "change_initial_pct": round(eq.get("change_initial_pct", 0.0) * 100, 4),
            "win_rate": round(perf.get("win_rate", 0.0) * 100, 2),
            "recent_win_rate": round(perf.get("recent_win_rate", 0.0) * 100, 2),
            "no_fill_streak": self.no_fill_streak,
            "hold_top": self.hold_counter.most_common(5),
            "gate_reject_top": self.gate_reject_counter.most_common(5),
        }
        messages = [
            {"role": "system", "content": "너는 자동매매 봇 운영 리포터다. 한국어로만 답하고, 3줄 이내로 핵심 진단만 작성한다. 과장 없이 수치 기반으로 적는다."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ]
        return self.ollama.chat(messages, fast=True)

    # ---------- reporting ----------
    def performance_summary(self) -> dict:
        trades = self._filter_trades_df(self.db.recent_trades(limit=200))
        if trades.empty:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "recent20_win_rate": 0.0,
                "recent_win_rate": 0.0,
                "total_pnl": 0.0,
                "recent20_pnl": 0.0,
                "best": 0.0,
                "worst": 0.0,
            }
        pnl = pd.to_numeric(trades["pnl"], errors="coerce").fillna(0.0)
        wins = int((pnl > 0).sum())
        total = int(len(trades))
        recent20 = pnl.tail(20)
        recent20_wins = int((recent20 > 0).sum())
        return {
            "total_trades": total,
            "win_rate": wins / total if total else 0.0,
            "recent20_win_rate": recent20_wins / len(recent20) if len(recent20) else 0.0,
            "recent_win_rate": recent20_wins / len(recent20) if len(recent20) else 0.0,
            "total_pnl": float(pnl.sum()),
            "recent20_pnl": float(recent20.sum()) if len(recent20) else 0.0,
            "best": float(pnl.max()),
            "worst": float(pnl.min()),
        }

    def email_report(self) -> Tuple[str, str]:
        snap = self.record_equity_snapshot(note="email")
        equity = float(snap["equity_krw"])
        perf = self.performance_summary()
        eq = self.equity_change_report(equity)
        positions_df = self._filter_positions_df(self.db.get_positions())
        positions = positions_df.to_dict("records") if not positions_df.empty else []
        unmanaged = self.get_unmanaged_live_holdings(min_value_krw=3000.0)
        balances = self.ex.get_balances()
        available_krw = safe_float(balances.get("KRW", 0.0), 0.0)
        ai_summary = self.maybe_generate_ai_email_summary(perf, eq)

        subject = (
            f"[BOT] 총액 {equity:,.0f} KRW | 3h {format_signed_krw(eq['change_3h'])} ({format_signed_pct(eq['change_3h_pct'])}) "
            f"| 초기 {format_signed_krw(eq['change_initial'])} ({format_signed_pct(eq['change_initial_pct'])}) "
            f"| 승률 {perf.get('win_rate', 0.0) * 100:.1f}%"
        )

        lines = [
            f"리포트 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "[핵심 지표]",
            f"- 총액: {equity:,.0f} KRW",
            f"- 3시간 전 대비: {format_signed_krw(eq['change_3h'])} ({format_signed_pct(eq['change_3h_pct'])}) | 기준 총액 {eq['equity_3h_ago']:,.0f} KRW",
            f"- 초기 금액 200,000 대비: {format_signed_krw(eq['change_initial'])} ({format_signed_pct(eq['change_initial_pct'])})",
            f"- 승률: 전체 {perf.get('win_rate', 0.0) * 100:.2f}% | 최근 20건 {perf.get('recent20_win_rate', 0.0) * 100:.2f}%",
            f"- 누적 손익: {format_signed_krw(perf.get('total_pnl', 0.0))}",
            f"- 최근 20건 손익: {format_signed_krw(perf.get('recent20_pnl', 0.0))}",
            f"- 거래 수: 총 {perf.get('total_trades', 0)}건",
            f"- no_fill_streak: {self.no_fill_streak}",
            f"- 사용 가능 KRW: {available_krw:,.0f}",
            f"- 제외 자산: {', '.join(sorted(self.excluded_symbols()))}",
            "",
            "[병목 요약]",
            f"- HOLD 상위: {self.hold_counter.most_common(8)}",
            f"- GATE reject 상위: {self.gate_reject_counter.most_common(8)}",
            "",
            "[보유 포지션]",
        ]

        if positions:
            for p in positions:
                lines.append(
                    f"- {p.get('market')} | 진입가 {format_price(safe_float(p.get('entry_price')))} | "
                    f"손절 {format_price(safe_float(p.get('stop_loss')))} | size {safe_float(p.get('size')):.8f} | "
                    f"score {safe_float(p.get('score')):.2f} | ml {safe_float(p.get('ml_prob'), 0.5):.2f}"
                )
        else:
            lines.append("- 없음")

        lines += ["", "[비관리 보유자산]"]
        if unmanaged:
            for row in unmanaged:
                lines.append(
                    f"- {row['market']} | 수량 {safe_float(row.get('amount')):.8f} | 현재가 {format_price(safe_float(row.get('price')))} | 평가액 {safe_float(row.get('value_krw')):,.0f} KRW"
                )
        else:
            lines.append("- 없음")

        if ai_summary:
            lines += ["", "[AI 요약]", ai_summary]

        lines += [
            "",
            "[JSON]",
            json.dumps({
                "time": datetime.now().isoformat(),
                "equity_krw": equity,
                "equity_change": eq,
                "perf": perf,
                "no_fill_streak": self.no_fill_streak,
                "hold_top": self.hold_counter.most_common(8),
                "gate_reject_top": self.gate_reject_counter.most_common(8),
                "positions": positions,
                "unmanaged_live_holdings": unmanaged,
                "available_krw": available_krw,
                "ai_summary": ai_summary,
            }, ensure_ascii=False, indent=2),
        ]
        body = "\n".join(lines)
        return subject, body

    def cycle(self):
        LOG.info("="*60)
        LOG.info(f"[CYCLE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        LOG.info("="*60)

        if self.cfg.ml.enabled:
            last_train = float(self.db.get_meta("ml_last_train", "0") or 0)
            if time.time() - last_train > 2 * 3600:
                train_ml_model(self.db, self.cfg.ml.min_trades_to_train)
                self.db.set_meta("ml_last_train", str(time.time()))

        self.sync_unmanaged_live_holdings()
        self.maybe_exit_positions()
        self.log_position_audit()

        before_positions = len(self._filter_positions_df(self.db.get_positions()))
        self.decide_and_enter()
        after_positions = len(self._filter_positions_df(self.db.get_positions()))

        balances = self.ex.get_balances()
        available_krw = safe_float(balances.get("KRW", 0.0), 0.0)
        equity = self.get_equity_krw()
        capital_locked = available_krw < self.cfg.params.min_order_krw and equity > available_krw + 1.0

        if after_positions <= before_positions:
            if capital_locked:
                LOG.info(
                    f"[STREAK] no_fill_streak 유지: available_krw={available_krw:,.0f} < min_order={self.cfg.params.min_order_krw:,.0f} (capital_locked)"
                )
            else:
                self.no_fill_streak += 1
        else:
            self.no_fill_streak = 0

        self.db.set_meta("no_fill_streak", str(self.no_fill_streak))
        self.record_equity_snapshot(note="cycle")

        LOG.info(f"[STATS] HOLD top5: {self.hold_counter.most_common(5)}")
        LOG.info(f"[STATS] GATE reject top5: {self.gate_reject_counter.most_common(5)}")
        LOG.info(f"[SLEEP] {self.cfg.params.cycle_seconds}s (no_fill_streak={self.no_fill_streak})")

    def run_forever(self):
        if GMAIL_USER and GMAIL_PASSWORD and TARGET_EMAIL:
            t = threading.Thread(target=email_log_rotation_scheduler, args=(self.email_report,), daemon=True)
            t.start()
            LOG.info("[EMAIL] 로그 메일 전송 스케줄러 시작")

        while True:
            try:
                self.cycle()
            except Exception as e:
                LOG.error(f"[CYCLE] 예외: {e}")
            time.sleep(self.cfg.params.cycle_seconds)

# ============================================================
# Entrypoint
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    default_db = os.getenv("BOT_DB_PATH") or os.getenv("DB_PATH") or "bot.db"
    ap.add_argument("--db", default=default_db, help="sqlite db path")
    ap.add_argument("--dry-run", action="store_true", help="paper trading (default)")
    ap.add_argument("--live", action="store_true", help="live trading (needs API keys)")
    ap.add_argument("--once", action="store_true", help="run one cycle and exit")
    args = ap.parse_args()

    dry = True
    if args.live:
        dry = False
    elif args.dry_run:
        dry = True

    bot = TradingBot(db_path=args.db, dry_run=dry)
    LOG.info(f"[BOOT] DB_PATH={os.path.abspath(args.db)}")
    LOG.info(f"[BOOT] OLLAMA_URL={OLLAMA_URL} model={OLLAMA_MODEL} fast_model={OLLAMA_MODEL_FAST} email_summary={'on' if OLLAMA_EMAIL_SUMMARY else 'off'}")
    LOG.info(f"[BOOT] excluded_symbols={sorted(normalize_excluded_symbols(bot.cfg.params.exclude))} initial_capital={INITIAL_CAPITAL_KRW:,.0f}")

    bot.cfg.save_to_db(args.db)

    interrupted = False
    try:
        if args.once:
            bot.cycle()
            return
        bot.run_forever()
    except KeyboardInterrupt:
        interrupted = True
        LOG.info("[SHUTDOWN] Ctrl+C 감지")
    finally:
        if interrupted:
            bot.finalize_shutdown(reason="ctrl_c")

if __name__ == "__main__":
    main()
