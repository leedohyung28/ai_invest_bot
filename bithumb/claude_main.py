# ============================================================
# claude_main.py — AI 전략 완전 제어 리팩토링 버전
# ============================================================

import time, json, sqlite3, math
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import List
import requests
import pandas as pd
import numpy as np
import os, hmac, hashlib, base64, urllib.parse, jwt
from dotenv import load_dotenv
import re, logging, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import threading, sys

load_dotenv()

# ============================================================
# 로깅 설정
# ============================================================
current_log_file = None
file_handler = None
logger = logging.getLogger("TradingBot")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")
TARGET_EMAIL = os.getenv("TARGET_EMAIL")
EMAIL_INTERVAL = 3 * 60 * 60

def get_new_log_filename():
    """현재 시간 기준으로 새 로그 파일명 생성"""
    return datetime.now().strftime("%Y_%m_%d_%H_%M.log")

def setup_logger():
    """새로운 로그 파일을 생성하고 핸들러를 교체함"""
    global current_log_file, file_handler, logger
    new_log_file = get_new_log_filename()
    if file_handler:
        logger.removeHandler(file_handler)
        file_handler.close()
    current_log_file = new_log_file
    file_handler = logging.FileHandler(current_log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"새로운 로그 파일 생성됨: {current_log_file}")
    return new_log_file

def send_email_and_rotate_log():
    """현재 로그 파일을 전송하고, 새로운 로그 파일로 교체"""
    global current_log_file, last_portfolio_value, last_email_time
    file_to_send = current_log_file
    if not file_to_send or not os.path.exists(file_to_send):
        logger.warning("전송할 로그 파일이 없습니다.")
        return
    
    current_portfolio = get_total_portfolio_value()
    current_value = current_portfolio['total_krw']
    
    if last_portfolio_value is None:
        last_portfolio_value = current_value
    diff_krw = current_value - last_portfolio_value
    diff_pct = (diff_krw / last_portfolio_value * 100) if last_portfolio_value > 0 else 0
    
    perf = analyze_trading_performance()
    change_raw = db_get_meta("last_ai_config_change", None)
    change_text = "(기록 없음)"
    change_ts = None
    if change_raw:
        try:
            obj = json.loads(change_raw)
            change_text = obj.get("text", "(변경 없음)")
            change_ts = obj.get("ts", None)
        except Exception:
            change_text = change_raw
    
    coins_lines = []
    coins = current_portfolio.get("coins", {})
    if coins:
        for coin, info in sorted(coins.items(), key=lambda x: x[1]['value_krw'], reverse=True):
            coins_lines.append(f" - {coin}: {info['balance']:.6f}개, 평가 {info['value_krw']:,.0f}원 (@ {info['price']:,.0f})")
    else:
        coins_lines.append(" - (보유 코인 없음)")
    
    now = datetime.now()
    subject = f"Trading Bot : {now.strftime('%Y/%m/%d %H:%M')} (3시간 리포트)"
    change_time_str = ""
    if change_ts:
        change_time_str = datetime.fromtimestamp(change_ts).strftime("%Y/%m/%d %H:%M")
    
    body = f"""
=== 트레이딩 봇 3시간 리포트 ===

1) 3시간 전 대비 금액 증감
 - 현재 총 자산: {current_value:,.0f}원
 - 3시간 전(직전 리포트): {last_portfolio_value:,.0f}원
 - 증감: {diff_krw:+,.0f}원 ({diff_pct:+.2f}%)

2) AI Config 변동 내역
 - 마지막 변경 시각: {change_time_str if change_time_str else "(알 수 없음)"}
{change_text}

3) 승률 / 최근 100거래 성과
 - 총 거래 수: {perf['total_trades']}
 - 승률: {perf['win_rate']*100:.1f}%
 - 총 손익: {perf['total_pnl']:+,.0f}원
 - 평균 이익: {perf['avg_profit']:+,.0f}원
 - 평균 손실: {-perf['avg_loss']:,.0f}원
 - 총 수수료: {perf['total_fees']:,.0f}원

4) 포트폴리오 구성 (보유코인 포함)
 - KRW 잔고: {current_portfolio['krw']:,.0f}원
 - 코인 평가액: {current_portfolio['total_coin_value']:,.0f}원
 - 보유 코인 수: {len(coins)}
{chr(10).join(coins_lines)}

(상세 로그는 첨부 파일 참고)
"""
    
    try:
        if GMAIL_USER and GMAIL_PASSWORD:
            msg = MIMEMultipart()
            msg['From'] = GMAIL_USER
            msg['To'] = TARGET_EMAIL
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            with open(file_to_send, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={file_to_send}')
                msg.attach(part)
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(GMAIL_USER, GMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            logger.info(f"이메일 전송 성공: {file_to_send} -> {TARGET_EMAIL}")
            last_portfolio_value = current_value
            last_email_time = time.time()
        else:
            logger.warning("이메일 설정이 없어 전송을 건너뜁니다.")
    except Exception as e:
        logger.error(f"이메일 전송 중 오류 발생: {e}")
    setup_logger()

def log_rotation_scheduler():
    """30분마다 로그 파일을 전송하고 교체하는 스케줄러"""
    while True:
        time.sleep(EMAIL_INTERVAL)
        send_email_and_rotate_log()

setup_logger()
last_portfolio_value = None
last_email_time = 0

# ============================================================
# 설정 데이터클래스
# ============================================================

@dataclass
class Params:
    """거래 전략 기본 파라미터"""
    max_positions: int = 4
    max_coin_weight: float = 0.6
    risk_per_trade: float = 0.035
    position_allocation_pct: float = 0.65
    atr_mult_stop: float = 3.0
    breakout_lookback: int = 2
    trend_ma_fast: int = 20
    trend_ma_slow: int = 50
    cooldown_minutes: int = 90
    min_volume_mult: float = 1.2

@dataclass
class StrategyConfig:
    """전략 종료 조건 파라미터"""
    use_trend_filter: bool = True
    use_volume_filter: bool = True
    use_volatility_filter: bool = False
    min_volume_mult: float = 1.3
    volatility_mult: float = 0.9
    trailing_stop_profit_threshold: float = 0.018
    max_loss_per_trade: float = 0.022
    take_profit_pct: float = 0.040
    time_stop_hours: float = 48.0

@dataclass
class SignalConfig:
    """신호 생성 파라미터"""
    # RSI
    rsi_min: float = 30.0
    rsi_max: float = 65.0
    rsi_oversold: float = 35.0
    rsi_overbought: float = 70.0
    # 가격 포지션
    price_pos_min: float = 0.2
    price_pos_max: float = 0.4
    price_pos_block: float = 0.8
    price_pos_lookback: int = 10
    # 추세
    trend_strong_pct: float = 0.02
    # 거래량
    volume_confirm_mult: float = 0.9
    volume_strong_mult: float = 1.5
    # BB
    bb_lower_touch_pct: float = 0.02
    # 캔들 강도
    candle_strength_mult: float = 0.5
    big_candle_mult: float = 1.0
    # ATR 손절 배수
    atr_stop_p1: float = 1.5
    atr_stop_p2: float = 1.2
    atr_stop_p3: float = 1.0
    atr_stop_p4: float = 1.0
    atr_stop_p5: float = 1.0
    # 손절 안전 마진
    stop_margin_p1: float = 0.98
    stop_margin_p2: float = 0.97
    stop_margin_p3: float = 0.97
    stop_margin_p4: float = 0.96
    stop_margin_p5: float = 0.96
    # 패턴 활성화
    use_pattern_1: bool = True
    use_pattern_2: bool = True
    use_pattern_3: bool = True
    use_pattern_4: bool = True
    use_pattern_5: bool = True
    # 패턴 우선순위
    pattern_priority: List[int] = field(default_factory=lambda: [3, 1, 2, 4, 5])

@dataclass
class RiskConfig:
    """리스크/주문 실행 파라미터"""
    # 트레일링
    trail_atr_mult: float = 2.0
    pullback_exit_pct: float = 0.02
    pullback_lookback: int = 10
    # 쿨다운
    cooldown_after_loss: int = 90
    cooldown_after_win: int = 15
    cooldown_default: int = 5
    # 포지션 사이징
    invest_capital_pct: float = 0.55
    risk_multiplier: float = 8.0
    # 주문 실행
    order_wait_sec: int = 30
    market_order_buffer: float = 1.1
    available_krw_safety: float = 0.95
    min_order_mult: float = 1.0

@dataclass
class Config:
    """시스템 설정"""
    db_path: str = os.getenv("DB_PATH", "trading_bot.db")
    ollama_url: str = os.getenv("OLLAMA_URL")
    ollama_model: str = os.getenv("OLLAMA_MODEL")
    quote: str = "KRW"
    exclude: tuple = ("BTC", "ETH")
    universe_size: int = 300
    collect_interval_sec: int = 300
    ai_refresh_min: int = 30
    fee_rate: float = 0.0004
    min_order_krw: float = 7000

@dataclass
class AIConfig:
    """AI가 제어하는 모든 설정의 통합 컨테이너"""
    params: Params = field(default_factory=Params)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    
    def to_dict(self) -> dict:
        return {
            "params": asdict(self.params),
            "strategy": asdict(self.strategy),
            "signal": asdict(self.signal),
            "risk": asdict(self.risk),
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: dict) -> "AIConfig":
        cfg = cls()
        def safe_set(obj, kv: dict):
            for k, v in kv.items():
                if hasattr(obj, k):
                    try: 
                        setattr(obj, k, v)
                    except: 
                        pass
        if "params" in data: safe_set(cfg.params, data["params"])
        if "strategy" in data: safe_set(cfg.strategy, data["strategy"])
        if "signal" in data: safe_set(cfg.signal, data["signal"])
        if "risk" in data: safe_set(cfg.risk, data["risk"])
        return cfg
    
    @classmethod
    def from_json(cls, json_str: str) -> "AIConfig":
        return cls.from_dict(json.loads(json_str))
    
    def save_to_db(self):
        db_set_meta("ai_config_v2", self.to_json())
        logger.info("[AIConfig] 설정 DB 저장 완료")
    
    @classmethod
    def load_from_db(cls) -> "AIConfig":
        raw = db_get_meta("ai_config_v2")
        if raw:
            try:
                cfg = cls.from_json(raw)
                logger.info("[AIConfig] 설정 DB 로드 완료")
                return cfg
            except Exception as e:
                logger.warning(f"[AIConfig] 로드 실패, 기본값 사용: {e}")
        return cls()

CFG = Config()
AI_CFG = AIConfig()

# ============================================================
# AI Config 범위 검증 함수
# ============================================================

def validate_and_clamp_config(cfg: AIConfig) -> AIConfig:
    """AI가 설정한 값의 범위 검증 및 자동 보정"""
    p = cfg.params
    s = cfg.strategy
    sg = cfg.signal
    r = cfg.risk
    
    # ── Params ──
    p.risk_per_trade = max(0.010, min(0.080, p.risk_per_trade))
    p.max_positions = max(1, min(5, int(p.max_positions)))
    p.trend_ma_fast = max(5, min(30, int(p.trend_ma_fast)))
    p.trend_ma_slow = max(30, min(120, int(p.trend_ma_slow)))
    if p.trend_ma_fast >= p.trend_ma_slow:
        p.trend_ma_slow = p.trend_ma_fast + 20
    p.cooldown_minutes = max(15, min(240, int(p.cooldown_minutes)))
    p.atr_mult_stop = max(1.0, min(5.0, p.atr_mult_stop))
    
    # ── StrategyConfig ──
    s.max_loss_per_trade = max(0.008, min(0.050, s.max_loss_per_trade))
    s.take_profit_pct = max(0.010, min(0.100, s.take_profit_pct))
    if s.take_profit_pct <= s.max_loss_per_trade:
        s.take_profit_pct = s.max_loss_per_trade * 1.6
    s.time_stop_hours = max(6.0, min(120.0, s.time_stop_hours))
    s.trailing_stop_profit_threshold = max(0.008, min(0.050, s.trailing_stop_profit_threshold))
    s.min_volume_mult = max(0.5, min(3.0, s.min_volume_mult))
    
    # ── SignalConfig ──
    sg.rsi_min = max(15.0, min(45.0, sg.rsi_min))
    sg.rsi_max = max(50.0, min(80.0, sg.rsi_max))
    sg.rsi_oversold = max(20.0, min(45.0, sg.rsi_oversold))
    sg.rsi_overbought = max(60.0, min(85.0, sg.rsi_overbought))
    if sg.rsi_min >= sg.rsi_max: 
        sg.rsi_max = sg.rsi_min + 20
    sg.price_pos_min = max(0.05, min(0.40, sg.price_pos_min))
    sg.price_pos_max = max(sg.price_pos_min + 0.05, min(0.70, sg.price_pos_max))
    sg.price_pos_block = max(0.60, min(0.98, sg.price_pos_block))
    sg.price_pos_lookback = max(5, min(30, int(sg.price_pos_lookback)))
    sg.trend_strong_pct = max(0.003, min(0.08, sg.trend_strong_pct))
    sg.volume_confirm_mult = max(0.3, min(1.5, sg.volume_confirm_mult))
    sg.volume_strong_mult = max(1.0, min(4.0, sg.volume_strong_mult))
    sg.bb_lower_touch_pct = max(0.00, min(0.05, sg.bb_lower_touch_pct))
    sg.candle_strength_mult = max(0.1, min(2.0, sg.candle_strength_mult))
    sg.big_candle_mult = max(0.3, min(3.0, sg.big_candle_mult))
    
    for attr in ["atr_stop_p1", "atr_stop_p2", "atr_stop_p3", "atr_stop_p4", "atr_stop_p5"]:
        setattr(sg, attr, max(0.3, min(4.0, getattr(sg, attr))))
    for attr in ["stop_margin_p1", "stop_margin_p2", "stop_margin_p3", "stop_margin_p4", "stop_margin_p5"]:
        setattr(sg, attr, max(0.90, min(0.99, getattr(sg, attr))))
    
    valid = set(range(1, 6))
    if not (isinstance(sg.pattern_priority, list) and 
            set(sg.pattern_priority) == valid and 
            len(sg.pattern_priority) == 5):
        sg.pattern_priority = [3, 1, 2, 4, 5]
    
    # ── RiskConfig ──
    r.trail_atr_mult = max(1.0, min(5.0, r.trail_atr_mult))
    r.pullback_exit_pct = max(0.005, min(0.08, r.pullback_exit_pct))
    r.pullback_lookback = max(3, min(30, int(r.pullback_lookback)))
    r.cooldown_after_loss = max(30, min(360, int(r.cooldown_after_loss)))
    r.cooldown_after_win = max(5, min(120, int(r.cooldown_after_win)))
    r.cooldown_default = max(1, min(30, int(r.cooldown_default)))
    r.invest_capital_pct = max(0.20, min(0.90, r.invest_capital_pct))
    r.risk_multiplier = max(2.0, min(20.0, r.risk_multiplier))
    r.order_wait_sec = max(5, min(120, int(r.order_wait_sec)))
    r.market_order_buffer = max(1.00, min(1.30, r.market_order_buffer))
    r.available_krw_safety = max(0.70, min(0.99, r.available_krw_safety))
    r.min_order_mult = max(0.5, min(5.0, r.min_order_mult))
    
    return cfg

# ============================================================
# DB 함수들
# ============================================================

def init_db():
    con = sqlite3.connect(CFG.db_path)
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
    cur.execute("""
        CREATE TABLE IF NOT EXISTS meta (
            k TEXT PRIMARY KEY,
            v TEXT
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
            entry_fee REAL
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
            exit_reason TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ai_learning (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER,
            win_rate REAL,
            avg_profit REAL,
            avg_loss REAL,
            total_trades INTEGER,
            params_json TEXT,
            strategy_json TEXT,
            performance_score REAL
        )
    """)
    con.commit()
    con.close()

def db_put_candles(rows):
    con = sqlite3.connect(CFG.db_path)
    cur = con.cursor()
    cur.executemany("""
        INSERT OR REPLACE INTO candles
        (ts, market, timeframe, open, high, low, close, volume, trade_value)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    con.commit()
    con.close()

def db_get_candles(market, timeframe, limit=300):
    con = sqlite3.connect(CFG.db_path)
    df = pd.read_sql_query("""
        SELECT * FROM candles
        WHERE market=? AND timeframe=?
        ORDER BY ts DESC LIMIT ?
    """, con, params=(market, timeframe, limit))
    con.close()
    return df.sort_values('ts').reset_index(drop=True) if not df.empty else df

def db_set_meta(k, v):
    con = sqlite3.connect(CFG.db_path)
    cur = con.cursor()
    cur.execute("INSERT OR REPLACE INTO meta (k, v) VALUES (?, ?)", (k, v))
    con.commit()
    con.close()

def save_ai_change_summary(changes_text: str):
    if not changes_text:
        changes_text = "(변경 없음)"
    payload = {
        "ts": int(time.time()),
        "text": changes_text
    }
    db_set_meta("last_ai_config_change", json.dumps(payload, ensure_ascii=False))

def db_get_meta(k, default=None):
    con = sqlite3.connect(CFG.db_path)
    cur = con.cursor()
    cur.execute("SELECT v FROM meta WHERE k=?", (k,))
    r = cur.fetchone()
    con.close()
    return r[0] if r else default

def db_get_positions():
    con = sqlite3.connect(CFG.db_path)
    df = pd.read_sql_query("SELECT * FROM positions", con)
    con.close()
    return df

def db_add_position(market, entry_price, size, stop_loss, direction, entry_fee):
    con = sqlite3.connect(CFG.db_path)
    cur = con.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO positions
        (market, entry_price, entry_time, size, stop_loss, direction, entry_fee)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (market, entry_price, int(time.time()), size, stop_loss, direction, entry_fee))
    con.commit()
    con.close()

def db_remove_position(market):
    con = sqlite3.connect(CFG.db_path)
    cur = con.cursor()
    cur.execute("DELETE FROM positions WHERE market=?", (market,))
    con.commit()
    con.close()

def db_get_recent_trades(limit=100):
    con = sqlite3.connect(CFG.db_path)
    df = pd.read_sql_query("""
        SELECT * FROM trades
        ORDER BY exit_time DESC
        LIMIT ?
    """, con, params=(limit,))
    con.close()
    return df

def db_get_recent_trades_by_market(market, limit=5):
    con = sqlite3.connect(CFG.db_path)
    df = pd.read_sql_query("""
        SELECT * FROM trades
        WHERE market=?
        ORDER BY exit_time DESC
        LIMIT ?
    """, con, params=(market, limit))
    con.close()
    return df

# ============================================================
# 유틸리티 함수들
# ============================================================

def is_excluded_coin(market):
    if not market or '-' not in market:
        return True
    _, currency = market.split('-')
    if currency.upper() in [x.upper() for x in CFG.exclude]:
        return True
    if len(currency) < 2:
        return True
    return False

def analyze_trading_performance():
    trades = db_get_recent_trades(limit=100)
    if trades.empty:
        return {
            "total_trades": 0,
            "win_rate": 0,
            "avg_profit": 0,
            "avg_loss": 0,
            "profit_factor": 0,
            "avg_holding_hours": 0,
            "best_trade": 0,
            "worst_trade": 0,
            "total_pnl": 0,
            "total_fees": 0
        }
    
    winning_trades = trades[trades['pnl'] > 0]
    losing_trades = trades[trades['pnl'] <= 0]
    total_trades = len(trades)
    win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
    avg_profit = winning_trades['pnl'].mean() if not winning_trades.empty else 0
    avg_loss = abs(losing_trades['pnl'].mean()) if not losing_trades.empty else 0
    total_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
    total_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    avg_holding_hours = trades['holding_hours'].mean() if 'holding_hours' in trades.columns else 0
    
    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "avg_profit": avg_profit,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_holding_hours": avg_holding_hours,
        "best_trade": trades['pnl'].max(),
        "worst_trade": trades['pnl'].min(),
        "total_pnl": trades['pnl'].sum(),
        "total_fees": trades['total_fee'].sum()
    }

# ============================================================
# Bithumb API
# ============================================================

class BithumbPrivateAPI:
    def __init__(self):
        self.api_url = "https://api.bithumb.com"
        self.api_key = os.getenv("BITHUMB_API_KEY")
        self.api_secret = os.getenv("BITHUMB_SECRET_KEY")
        if not self.api_key or not self.api_secret:
            raise ValueError("[ERROR] 환경변수에 BITHUMB_API_KEY와 BITHUMB_SECRET_KEY를 설정해주세요")
        self.sess = requests.Session()
    
    def _create_jwt_token(self, query_params=None):
        import uuid
        payload = {
            'access_key': self.api_key,
            'nonce': str(uuid.uuid4()),
            'timestamp': round(time.time() * 1000)
        }
        if query_params:
            query_string = urllib.parse.urlencode(query_params, doseq=True).encode('utf-8')
            query_hash = hashlib.sha512(query_string).hexdigest()
            payload['query_hash'] = query_hash
            payload['query_hash_alg'] = 'SHA512'
        token = jwt.encode(payload, self.api_secret, algorithm='HS256')
        if isinstance(token, bytes):
            token = token.decode('utf-8')
        return f'Bearer {token}'
    
    def _api_call(self, method, endpoint, params=None, is_query=False):
        try:
            url = self.api_url + endpoint
            if is_query and params:
                authorization = self._create_jwt_token(params)
                headers = {
                    'Authorization': authorization,
                    'Content-Type': 'application/json'
                }
                response = self.sess.request(method, url, params=params, headers=headers, timeout=10)
            else:
                authorization = self._create_jwt_token(params)
                headers = {
                    'Authorization': authorization,
                    'Content-Type': 'application/json'
                }
                response = self.sess.request(method, url, json=params, headers=headers, timeout=10)
            
            try:
                response.raise_for_status()
                result = response.json()
            except requests.HTTPError as e:
                logger.error(f"[ERROR] HTTP {response.status_code} {response.text}")
                raise
            
            if 'error' in result:
                logger.error(f"[ERROR] API 에러: {result['error']['message']}")
                return None
            return result
        except requests.exceptions.Timeout:
            logger.error(f"[ERROR] API 타임아웃: {endpoint}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"[ERROR] API 요청 실패: {e}")
            return None
        except Exception as e:
            logger.error(f"[ERROR] 예상치 못한 오류: {e}")
            return None
    
    def get_accounts(self):
        endpoint = "/v1/accounts"
        return self._api_call("GET", endpoint)
    
    def get_order_chance(self, market):
        endpoint = "/v1/orders/chance"
        params = {"market": market}
        return self._api_call("GET", endpoint, params, is_query=True)
    
    def place_order(self, market, side, volume=None, price=None, ord_type="limit"):
        endpoint = "/v1/orders"
        params = {
            "market": market,
            "side": side,
            "ord_type": ord_type
        }
        if volume:
            params["volume"] = str(volume)
        if price:
            params["price"] = str(price)
        return self._api_call("POST", endpoint, params)
    
    def cancel_order(self, uuid):
        endpoint = "/v1/order"
        params = {"uuid": uuid}
        return self._api_call("DELETE", endpoint, params, is_query=True)
    
    def get_order(self, uuid):
        endpoint = "/v1/order"
        params = {"uuid": uuid}
        return self._api_call("GET", endpoint, params, is_query=True)

class BithumbPublic:
    def __init__(self):
        self.sess = requests.Session()
        self.base_url = "https://api.bithumb.com"
    
    def get_markets(self):
        url = f"{self.base_url}/v1/market/all"
        try:
            resp = self.sess.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"[ERROR] 마켓 코드 조회 실패: {e}")
            return []
    
    def get_day_candles(self, market: str, count: int = 300, to: str = None):
        url = f"{self.base_url}/v1/candles/minutes/30"
        params = {
            "market": market,
            "count": min(count, 300)
        }
        if to:
            params["to"] = to
        try:
            resp = self.sess.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"[ERROR] {market} 캔들 조회 실패: {e}")
            return []
    
    def get_current_price(self, market: str):
        if is_excluded_coin(market):
            return None
        url = f"{self.base_url}/v1/ticker"
        params = {"markets": market}
        try:
            resp = self.sess.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data and len(data) > 0:
                return float(data[0]['trade_price'])
            return None
        except Exception as e:
            logger.error(f"[ERROR] {market} 현재가 조회 실패: {e}")
            return None

# ============================================================
# 수수료 및 잔고 함수
# ============================================================

def calculate_buy_fee_and_total(price, size):
    amount = price * size
    fee = amount * CFG.fee_rate
    total = amount + fee
    return {
        "amount": amount,
        "fee": fee,
        "total": total
    }

def calculate_sell_fee_and_net(price, size):
    amount = price * size
    fee = amount * CFG.fee_rate
    net = amount - fee
    return {
        "amount": amount,
        "fee": fee,
        "net": net
    }

def get_available_krw_balance():
    try:
        api = BithumbPrivateAPI()
        result = api.get_accounts()
        if result:
            for account in result:
                if account['currency'] == 'KRW':
                    return float(account['balance'])
        return 0
    except Exception as e:
        logger.error(f"[ERROR] KRW 잔고 조회 실패: {e}")
        return 0

def get_coin_balance(market):
    if is_excluded_coin(market):
        return 0
    try:
        api = BithumbPrivateAPI()
        result = api.get_accounts()
        currency = market.split('-')[1]
        if result:
            for account in result:
                if account['currency'].upper() == currency.upper():
                    return float(account['balance'])
        return 0
    except Exception as e:
        logger.error(f"[ERROR] {market} 잔고 조회 실패: {e}")
        return 0

def get_total_portfolio_value(exclude_coins=None):
    if exclude_coins is None:
        exclude_coins = CFG.exclude
    try:
        api = BithumbPrivateAPI()
        pub = BithumbPublic()
        accounts = api.get_accounts()
        if not accounts:
            return {"krw": 0, "coins": {}, "total_coin_value": 0, "total_krw": 0}
        
        krw_balance = 0
        coins_value = {}
        total_coin_value = 0
        
        for account in accounts:
            currency = account['currency']
            balance = float(account['balance'])
            if currency == 'KRW':
                krw_balance = balance
                continue
            if currency.upper() in [x.upper() for x in exclude_coins]:
                continue
            if balance > 0:
                market = f"KRW-{currency}"
                current_price = pub.get_current_price(market)
                if current_price:
                    value_krw = balance * current_price
                    coins_value[currency] = {
                        "balance": balance,
                        "price": current_price,
                        "value_krw": value_krw
                    }
                    total_coin_value += value_krw
        
        total_value = krw_balance + total_coin_value
        return {
            "krw": krw_balance,
            "coins": coins_value,
            "total_coin_value": total_coin_value,
            "total_krw": total_value
        }
    except Exception as e:
        logger.error(f"[ERROR] 포트폴리오 가치 계산 실패: {e}")
        return {"krw": 0, "coins": {}, "total_coin_value": 0, "total_krw": 0}

def print_portfolio_summary():
    portfolio = get_total_portfolio_value()
    logger.info(f"{'='*60}")
    logger.info(f"[PORTFOLIO] AI 관리 자산 요약")
    logger.info(f" (BTC, ETH 등 수동 투자 자산 제외)")
    logger.info(f"{'='*60}")
    logger.info(f"KRW 잔고: {portfolio['krw']:,.0f}원")
    logger.info(f"코인 평가액: {portfolio['total_coin_value']:,.0f}원")
    logger.info(f"총 자산: {portfolio['total_krw']:,.0f}원")
    if portfolio['coins']:
        logger.info(f"AI 관리 코인 ({len(portfolio['coins'])}개):")
        for coin, info in sorted(portfolio['coins'].items(), 
                                key=lambda x: x[1]['value_krw'], 
                                reverse=True):
            logger.info(f" - {coin}: {info['balance']:.6f} "
                       f"= {info['value_krw']:,.0f}원 (@ {info['price']:,.0f})")
    logger.info(f"{'='*60}\n")

# ============================================================
# 기술적 지표
# ============================================================

def calculate_atr(df, period=14):
    if len(df) < period + 1:
        return pd.Series([0] * len(df))
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr.fillna(0)

def calculate_ma(df, period):
    if len(df) < period:
        return pd.Series([0] * len(df))
    return df['close'].rolling(period).mean().fillna(0)

def parse_candles_to_rows(candles_data, market, timeframe="30m"):
    rows = []
    for candle in candles_data:
        try:
            ts = candle.get("timestamp", 0) // 1000
            o = float(candle.get("opening_price", 0))
            h = float(candle.get("high_price", 0))
            l = float(candle.get("low_price", 0))
            c = float(candle.get("trade_price", 0))
            v = float(candle.get("candle_acc_trade_volume", 0))
            tv = float(candle.get("candle_acc_trade_price", 0))
            rows.append((ts, market, timeframe, o, h, l, c, v, tv))
        except (ValueError, TypeError, KeyError):
            continue
    return rows

def collect_candles(markets):
    pub = BithumbPublic()
    for market in markets:
        if is_excluded_coin(market):
            continue
        try:
            candles = pub.get_day_candles(market, count=300)
            if candles:
                rows = parse_candles_to_rows(candles, market, "30m")
                if rows:
                    db_put_candles(rows)
        except Exception as e:
            logger.error(f"[ERROR] {market} 캔들 수집 실패: {e}")
        time.sleep(0.1)

# ============================================================
# 신호 생성 (SignalConfig 적용)
# ============================================================

def generate_swing_signals(market, params: Params, strategy: StrategyConfig, sig_cfg: SignalConfig) -> dict:
    """스윙 트레이딩 신호 생성"""
    if is_excluded_coin(market):
        return {"signal": "hold", "reason": "excluded_coin"}
    
    df = db_get_candles(market, "30m", limit=300)
    if len(df) < max(params.trend_ma_slow, params.breakout_lookback) + 5:
        return {"signal": "hold", "reason": "insufficient_data"}
    
    # 지표 계산
    df["atr"] = calculate_atr(df, 14)
    df["ma_fast"] = calculate_ma(df, params.trend_ma_fast)
    df["ma_slow"] = calculate_ma(df, params.trend_ma_slow)
    df["volume_ma"] = df["volume"].rolling(20).mean()
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, 1e-10)))
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + bb_std * 2
    df["bb_lower"] = df["bb_mid"] - bb_std * 2
    
    lb = sig_cfg.price_pos_lookback
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    current_price = last["close"]
    atr = last["atr"]
    ma_fast = last["ma_fast"]
    ma_slow = last["ma_slow"]
    rsi = last["rsi"]
    
    if current_price is None or current_price <= 0 or atr <= 0 or pd.isna(atr):
        return {"signal": "hold", "reason": "invalid_data"}
    
    # 공통 지표
    trend_up = ma_fast > ma_slow
    trend_strong = ma_fast > ma_slow * (1 + sig_cfg.trend_strong_pct)
    recent_high = df["high"].tail(lb).max()
    recent_low = df["low"].tail(lb).min()
    price_range = recent_high - recent_low
    price_pos = (current_price - recent_low) / price_range if price_range > 0 else 0.5
    
    rsi_ok = sig_cfg.rsi_min < rsi < sig_cfg.rsi_max
    rsi_os = rsi < sig_cfg.rsi_oversold
    
    vol_ok = last["volume"] > last["volume_ma"] * sig_cfg.volume_confirm_mult
    vol_strong = last["volume"] > last["volume_ma"] * sig_cfg.volume_strong_mult
    
    bb_lo_touch = current_price <= last["bb_lower"] * (1 + sig_cfg.bb_lower_touch_pct)
    bb_mid_abv = current_price > last["bb_mid"]
    
    # 진입 금지 조건
    if price_pos > sig_cfg.price_pos_block:
        return {"signal": "hold", "reason": "price_too_high"}
    if rsi > sig_cfg.rsi_overbought:
        return {"signal": "hold", "reason": "rsi_overbought"}
    if not trend_up:
        return {"signal": "hold", "reason": "downtrend"}
    if not vol_ok:
        return {"signal": "hold", "reason": "low_volume"}
    
    # 패턴 체크
    for pat_id in sig_cfg.pattern_priority:
        # Pattern 1: MA Support Bounce
        if pat_id == 1 and sig_cfg.use_pattern_1:
            if trend_up and not trend_strong:
                ma_touch = (
                    prev["low"] <= prev["ma_fast"] <= prev["high"] and
                    last["close"] > last["ma_fast"] and
                    last["close"] > last["open"]
                )
                if ma_touch and rsi_ok and vol_ok:
                    sl = ma_fast - atr * sig_cfg.atr_stop_p1
                    if sl > 0 and sl < current_price * sig_cfg.stop_margin_p1:
                        return {"signal": "buy", "price": float(current_price),
                               "stop": float(sl), "atr": float(atr),
                               "reason": "ma_support_bounce", "confidence": 0.9}
        
        # Pattern 2: Oversold Reversal
        elif pat_id == 2 and sig_cfg.use_pattern_2:
            if trend_up and rsi_os and bb_lo_touch:
                strong_candle = (last["close"] - last["open"]) > atr * sig_cfg.candle_strength_mult
                if strong_candle and vol_strong:
                    sl = last["low"] - atr * sig_cfg.atr_stop_p2
                    if sl > 0 and sl < current_price * sig_cfg.stop_margin_p2:
                        return {"signal": "buy", "price": float(current_price),
                               "stop": float(sl), "atr": float(atr),
                               "reason": "oversold_reversal", "confidence": 0.85}
        
        # Pattern 3: Trend Dip Buy
        elif pat_id == 3 and sig_cfg.use_pattern_3:
            if (trend_up and
                sig_cfg.price_pos_min <= price_pos <= sig_cfg.price_pos_max and
                rsi_ok and last["close"] > last["open"] and vol_ok):
                sl = recent_low - atr * sig_cfg.atr_stop_p3
                if sl > 0 and sl < current_price * sig_cfg.stop_margin_p3:
                    return {"signal": "buy", "price": float(current_price),
                           "stop": float(sl), "atr": float(atr),
                           "reason": "trend_dip_buy", "confidence": 0.95}
        
        # Pattern 4: Golden Cross
        elif pat_id == 4 and sig_cfg.use_pattern_4:
            ma_cross = (
                ma_fast > ma_slow and
                prev["ma_fast"] <= prev["ma_slow"] and
                last["close"] > last["ma_fast"]
            )
            if ma_cross and rsi_ok and bb_mid_abv and vol_strong:
                sl = ma_slow - atr * sig_cfg.atr_stop_p4
                if sl > 0 and sl < current_price * sig_cfg.stop_margin_p4:
                    return {"signal": "buy", "price": float(current_price),
                           "stop": float(sl), "atr": float(atr),
                           "reason": "golden_cross", "confidence": 0.8}
        
        # Pattern 5: V Reversal
        elif pat_id == 5 and sig_cfg.use_pattern_5:
            if len(df) >= 5:
                three_down = (
                    prev["close"] < prev2["close"] and
                    prev2["close"] < df.iloc[-4]["close"] and
                    prev["close"] < prev["open"]
                )
                big_green = (last["close"] - last["open"]) > atr * sig_cfg.big_candle_mult
                if trend_up and three_down and big_green and vol_strong:
                    sl = min(prev["low"], prev2["low"]) - atr * sig_cfg.atr_stop_p5
                    if sl > 0 and sl < current_price * sig_cfg.stop_margin_p5:
                        return {"signal": "buy", "price": float(current_price),
                               "stop": float(sl), "atr": float(atr),
                               "reason": "v_reversal", "confidence": 0.75}
    
    return {"signal": "hold", "reason": "no_setup"}

# ============================================================
# 포지션 청산 (RiskConfig 적용)
# ============================================================

def check_position_exit(market: str, current_price: float, position_row, 
                       params: Params, strategy: StrategyConfig, risk_cfg: RiskConfig):
    """포지션 청산 판단"""
    if is_excluded_coin(market):
        return False, None
    
    try:
        entry_price = float(position_row["entry_price"])
        stop_loss = float(position_row["stop_loss"])
        direction = position_row.get("direction", "long")
        entry_time = float(position_row["entry_time"])
    except Exception:
        return False, None
    
    if entry_price <= 0 or current_price is None or current_price <= 0:
        return False, None
    
    profit_pct = ((current_price - entry_price) / entry_price
                 if direction == "long"
                 else (entry_price - current_price) / entry_price)
    
    max_loss = float(strategy.max_loss_per_trade)
    take_profit = float(strategy.take_profit_pct)
    trail_start = float(strategy.trailing_stop_profit_threshold)
    time_stop_h = float(strategy.time_stop_hours)
    trail_mult = float(risk_cfg.trail_atr_mult)
    pullback_pct = float(risk_cfg.pullback_exit_pct)
    pb_lookback = int(risk_cfg.pullback_lookback)
    
    # 1) Stop-loss
    if direction == "long" and current_price <= stop_loss:
        logger.info(f" [STOP_LOSS] {market} {current_price:,.0f} <= {stop_loss:,.0f}")
        return True, "stop_loss"
    if direction == "short" and current_price >= stop_loss:
        logger.info(f" [STOP_LOSS] {market} {current_price:,.0f} >= {stop_loss:,.0f}")
        return True, "stop_loss"
    
    # 2) 최대 손실
    if profit_pct <= -max_loss:
        logger.info(f" [MAX_LOSS] {market} {profit_pct*100:.2f}% (limit {max_loss*100:.2f}%)")
        return True, "max_loss"
    
    # 3) 목표 수익
    if profit_pct >= take_profit:
        logger.info(f" [TAKE_PROFIT] {market} {profit_pct*100:.2f}% (target {take_profit*100:.2f}%)")
        return True, "take_profit"
    
    # 4) 시간 기반 청산
    holding_h = (time.time() - entry_time) / 3600.0
    if holding_h >= time_stop_h and profit_pct <= 0:
        logger.info(f" [TIME_STOP] {market} {holding_h:.1f}h {profit_pct*100:.2f}%")
        return True, "time_stop"
    
    # 5) Pullback 청산
    if profit_pct >= max(trail_start, 0.008):
        df = db_get_candles(market, "30m", limit=pb_lookback + 5)
        if len(df) >= pb_lookback:
            recent_high = float(df["high"].tail(pb_lookback).max())
            if recent_high > 0:
                pb = (recent_high - current_price) / recent_high
                if pb >= pullback_pct:
                    logger.info(f" [PULLBACK] {market} -{pb*100:.2f}%")
                    return True, "pullback_from_profit"
    
    # 6) 트레일링 스톱 업데이트
    if direction == "long" and profit_pct >= trail_start:
        df = db_get_candles(market, "30m", limit=150)
        if len(df) >= 14:
            atr_s = calculate_atr(df, 14)
            atr = float(atr_s.iloc[-1]) if len(atr_s) > 0 else 0.0
            if atr > 0:
                trailing = current_price - atr * trail_mult
                if trailing > stop_loss:
                    try:
                        con = sqlite3.connect(CFG.db_path)
                        cur = con.cursor()
                        cur.execute("UPDATE positions SET stop_loss=? WHERE market=?",
                                   (trailing, market))
                        con.commit()
                        con.close()
                        logger.info(f" [TRAILING] {market} {stop_loss:,.0f}→{trailing:,.0f}")
                    except Exception as e:
                        logger.error(f"[ERROR] trailing stop 업데이트 실패: {e}")
    
    return False, None

# ============================================================
# 주문 실행 (RiskConfig 적용)
# ============================================================
def execute_order(market: str, direction: str, price: float, size: float, risk_cfg: RiskConfig = None):
    """주문 실행"""
    if risk_cfg is None:
        risk_cfg = RiskConfig()

    if is_excluded_coin(market):
        return {"success": False, "order_id": None, "message": "Excluded coin", "fee": 0}

    if price is None or price <= 0:
        logger.error(f"[ERROR] {market} 잘못된 가격: {price}")
        return {"success": False, "order_id": None, "message": "Invalid price", "fee": 0}

    if size is None or size <= 0:
        logger.error(f"[ERROR] {market} 잘못된 수량: {size}")
        return {"success": False, "order_id": None, "message": "Invalid size", "fee": 0}

    min_order = CFG.min_order_krw * risk_cfg.min_order_mult

    if price * size < min_order:
        logger.error(f"[ERROR] {market} 최소 주문 미달: {price*size:,.0f}원")
        return {"success": False, "order_id": None, "message": "Below minimum order", "fee": 0}

    api = BithumbPrivateAPI()
    wait_sec = risk_cfg.order_wait_sec

    if direction == "buy":
        result = api.place_order(market=market, side="bid",
                                 price=str(int(price)), volume=str(size), ord_type="limit")

        if not result or "uuid" not in result:
            logger.error(f"[ERROR] {market} 매수 주문 접수 실패")
            return {"success": False, "order_id": None, "message": "Order failed", "fee": 0}

        order_id = result["uuid"]
        logger.info(f"[BUY ORDER] {market} 지정가 매수 접수 | 가격: {price:,.0f} | 수량: {size:.6f} | 대기: {wait_sec}초")

        # 체결 대기 루프
        start = time.time()
        while time.time() - start < wait_sec:
            order = api.get_order(order_id)
            if order and order.get("state") == "done":
                fee = calculate_buy_fee_and_total(price, size)["fee"]
                logger.info(f"[BUY FILLED] {market} 완전 체결 | 수량: {size:.6f}")
                return {"success": True, "order_id": order_id, "message": "Filled", "fee": fee}
            time.sleep(1)

        # 타임아웃 — 미체결 주문 취소, 시장가 재주문 없음
        order = api.get_order(order_id)
        exec_vol = 0.0
        if order:
            exec_vol = float(order.get("executed_volume", 0))

        try:
            api.cancel_order(order_id)
            logger.info(f"[CANCEL] {market} {wait_sec}초 타임아웃 → 주문 취소 "
                        f"(체결: {exec_vol:.6f} / 미체결: {size - exec_vol:.6f})")
        except Exception as e:
            logger.warning(f"[CANCEL] {market} 주문 취소 중 오류: {e}")

        # 부분 체결이 있으면 부분 성공으로 반환
        if exec_vol > 0:
            fee = calculate_buy_fee_and_total(price, exec_vol)["fee"]
            logger.info(f"[PARTIAL] {market} 부분 체결 포지션 등록 | 체결량: {exec_vol:.6f}")
            return {"success": True, "order_id": order_id,
                    "message": f"Partial filled ({exec_vol:.6f}), rest cancelled", "fee": fee}

        # 완전 미체결 → 실패
        return {"success": False, "order_id": None,
                "message": f"Timeout after {wait_sec}s, order cancelled", "fee": 0}

    else:  # sell
        pub = BithumbPublic()
        cur_mkt_price = pub.get_current_price(market) or price
        result = api.place_order(market=market, side="ask",
                                 volume=str(size), ord_type="market")
        if result and "uuid" in result:
            fee = calculate_sell_fee_and_net(cur_mkt_price, size)["fee"]
            return {"success": True, "order_id": result["uuid"], "message": "Market sell", "fee": fee}
        return {"success": False, "order_id": None, "message": "Sell failed", "fee": 0}


# ============================================================
# AI Config 갱신 (영어 프롬프트)
# ============================================================

def ai_refresh_all_configs(ai_cfg: AIConfig, perf: dict) -> tuple:
    """Ollama AI를 통해 모든 설정을 통합 업데이트 (영어 프롬프트)"""
    if not CFG.ollama_url or not CFG.ollama_model:
        logger.warning("[AI] Ollama not configured — skipping config update")
        return ai_cfg, "(Ollama not configured)"
    
    current_json = json.dumps(ai_cfg.to_dict(), ensure_ascii=False, indent=2)
    
    prompt = f"""
You are an AI cryptocurrency swing trading bot parameter optimization expert.
You are trading on Bithumb Exchange using 30-minute candle data for KRW markets.

## Current Performance Metrics
- Total Trades: {perf.get("total_trades", 0)}
- Win Rate: {perf.get("win_rate", 0)*100:.1f}%
- Profit Factor: {perf.get("profit_factor", 0):.2f}
- Total PnL: {perf.get("total_pnl", 0):+,.0f} KRW
- Average Holding Time: {perf.get("avg_holding_hours", 0):.1f} hours
- Average Profit: {perf.get("avg_profit", 0):+,.0f} KRW
- Average Loss: {abs(perf.get("avg_loss", 0)):,.0f} KRW
- Total Fees: {perf.get("total_fees", 0):,.0f} KRW

## Current Configuration (JSON)
{current_json}

## Adjustable Parameter Ranges

### params
risk_per_trade: 0.010~0.080 | max_positions: 1~5
trend_ma_fast: 5~30 | trend_ma_slow: 30~120
cooldown_minutes: 15~240 | atr_mult_stop: 1.0~5.0

### strategy
max_loss_per_trade: 0.008~0.050 | take_profit_pct: 0.010~0.100
time_stop_hours: 6~120 | trailing_stop_profit_threshold: 0.008~0.050
min_volume_mult: 0.5~3.0

### signal (AI-controlled)
rsi_min: 15~45 | rsi_max: 50~80 | rsi_oversold: 20~45 | rsi_overbought: 60~85
price_pos_min: 0.05~0.40 | price_pos_max: [min+0.05]~0.70 | price_pos_block: 0.60~0.98
price_pos_lookback: 5~30 | trend_strong_pct: 0.003~0.08
volume_confirm_mult: 0.3~1.5 | volume_strong_mult: 1.0~4.0
bb_lower_touch_pct: 0.00~0.05 | candle_strength_mult: 0.1~2.0 | big_candle_mult: 0.3~3.0
atr_stop_p1~p5: 0.3~4.0 (ATR stop multiplier per pattern)
stop_margin_p1~p5: 0.90~0.99 (stop safety margin per pattern)
use_pattern_1~5: true/false (pattern activation)
pattern_priority: [1~5 permutation] (check priority order, e.g., [3,1,2,4,5])

### risk (AI-controlled)
trail_atr_mult: 1.0~5.0 | pullback_exit_pct: 0.005~0.08 | pullback_lookback: 3~30
cooldown_after_loss: 30~360 | cooldown_after_win: 5~120 | cooldown_default: 1~30
invest_capital_pct: 0.20~0.90 | risk_multiplier: 2.0~20.0
order_wait_sec: 5~120 | market_order_buffer: 1.00~1.30 | available_krw_safety: 0.70~0.99

## Analysis and Adjustment Guidelines

1. Win Rate < 40%: Strengthen RSI filter (lower rsi_max), narrow price_pos range
2. Win Rate > 60%: Relax entry conditions, consider increasing max_positions
3. Profit Factor < 1.5: Expand take_profit or reduce max_loss
4. Avg Holding < 10h: Increase time_stop_hours, expand trail_atr_mult
5. Trade Count < 3: Relax entry conditions (raise price_pos_block, raise rsi_max)
6. Trade Count > 20: Tighten conditions, increase cooldown
7. If losses exceed profits: Reduce atr_stop multipliers, adjust stop_margin
8. pattern_priority: Place better-performing patterns first
9. If fees exceed 20% of PnL: Set minimum take_profit_pct to at least 0.02

You MUST respond ONLY in the following JSON format (explanations only in changes_summary):

{{
  "params": {{ ... }},
  "strategy": {{ ... }},
  "signal": {{ ... }},
  "risk": {{ ... }},
  "changes_summary": "Brief summary of changes in English"
}}
"""
    
    try:
        resp = requests.post(
            CFG.ollama_url,
            json={"model": CFG.ollama_model, "prompt": prompt, "stream": False},
            timeout=180
        )
        response_text = resp.json().get("response", "")
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if not json_match:
            return ai_cfg, "(AI response parsing failed)"
        
        parsed = json.loads(json_match.group())
        summary = parsed.pop("changes_summary", "(No changes)")
        new_cfg = AIConfig.from_dict(parsed)
        new_cfg = validate_and_clamp_config(new_cfg)
        new_cfg.save_to_db()
        save_ai_change_summary(summary)
        logger.info(f"[AI] Config updated: {summary[:100]}")
        return new_cfg, summary
    except Exception as e:
        logger.error(f"[AI] Config update error: {e}")
        return ai_cfg, f"(Error: {e})"

# ============================================================
# 메인 루프
# ============================================================

def run():
    global AI_CFG
    logger.info("=" * 60)
    logger.info("[STARTUP] Trading Bot Started")
    logger.info("=" * 60)
    
    init_db()
    threading.Thread(target=log_rotation_scheduler, daemon=True).start()
    
    # DB에서 AI 설정 로드
    AI_CFG = AIConfig.load_from_db()
    AI_CFG = validate_and_clamp_config(AI_CFG)
    
    pub = BithumbPublic()
    last_ai_refresh = 0
    print_portfolio_summary()
    
    while True:
        try:
            logger.info("=" * 60)
            logger.info(f"[CYCLE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 60)
            
            params = AI_CFG.params
            strategy = AI_CFG.strategy
            sig_cfg = AI_CFG.signal
            risk_cfg = AI_CFG.risk
            
            # 1. 캔들 수집
            markets = [m["market"] for m in pub.get_markets()
                      if m["market"].startswith(CFG.quote + "-")][:CFG.universe_size]
            collect_candles(markets)
            
            # 2. AI 설정 갱신
            elapsed_ai = (time.time() - last_ai_refresh) / 60
            if elapsed_ai >= CFG.ai_refresh_min:
                perf = analyze_trading_performance()
                AI_CFG, summary = ai_refresh_all_configs(AI_CFG, perf)
                params = AI_CFG.params
                strategy = AI_CFG.strategy
                sig_cfg = AI_CFG.signal
                risk_cfg = AI_CFG.risk
                last_ai_refresh = time.time()
            
            # 3. 포지션 관리 (청산 체크)
            positions = db_get_positions()
            if not positions.empty:
                for _, pos in positions.iterrows():
                    market = pos["market"]
                    if is_excluded_coin(market):
                        continue
                    
                    actual_bal = get_coin_balance(market)
                    if actual_bal <= 0:
                        logger.info(f"SYNC: {market} 실제 잔고 없음. DB에서 포지션을 제거합니다.")
                        db_remove_position(market)
                        continue  # 이미 팔렸으므로 이후 로직 건너뜀
                
                    current_price = pub.get_current_price(market)
                    if current_price is None:
                        continue
                    
                    should_exit, reason = check_position_exit(
                        market, current_price, pos, params, strategy, risk_cfg)
                    
                    if should_exit:
                        actual_bal = get_coin_balance(market)
                        if actual_bal <= 0:
                            continue
                        
                        logger.info(f"[EXIT] {market} {current_price:,.0f} ({reason})")
                        result = execute_order(market, "sell", current_price, actual_bal, risk_cfg)
                        
                        if result["success"]:
                            pnl = (current_price - pos["entry_price"]) * pos["size"]
                            pnl -= (pos.get("entry_fee", 0) + result["fee"])
                            holding_h = (time.time() - pos["entry_time"]) / 3600
                            
                            con = sqlite3.connect(CFG.db_path)
                            cur = con.cursor()
                            cur.execute("""INSERT INTO trades
                                (market, direction, entry_price, exit_price, size,
                                 entry_time, exit_time, pnl, total_fee, holding_hours, exit_reason)
                                VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                                (market, pos.get("direction", "long"),
                                 pos["entry_price"], current_price, pos["size"],
                                 pos["entry_time"], int(time.time()),
                                 pnl, result["fee"], holding_h, reason))
                            con.commit()
                            con.close()
                            db_remove_position(market)
            
            # 4. 매수 신호 스캔
            positions = db_get_positions()
            ai_pos_count = sum(1 for _, p in positions.iterrows()
                             if not is_excluded_coin(p["market"]))
            
            if ai_pos_count >= params.max_positions:
                logger.info(f"[INFO] Max positions reached: {ai_pos_count}/{params.max_positions}")
            else:
                signals = []
                for market in markets:
                    if is_excluded_coin(market):
                        continue
                    if not positions.empty and market in positions["market"].values:
                        continue
                    
                    sig = generate_swing_signals(market, params, strategy, sig_cfg)
                    if sig["signal"] == "buy":
                        signals.append((market, sig))
                
                if not signals:
                    logger.info("[INFO] No buy signals")
                else:
                    logger.info(f"[SIGNAL] {len(signals)} buy signals found")
                    avail_krw = get_available_krw_balance()
                    slots = params.max_positions - ai_pos_count
                    
                    # 쿨다운 체크
                    last_entries_raw = db_get_meta("last_entries", "{}")
                    last_entries = json.loads(last_entries_raw)
                    current_time = time.time()
                    
                    for market, signal in signals[:slots]:
                        if market in last_entries:
                            elapsed = (current_time - last_entries[market]) / 60
                            recent = db_get_recent_trades_by_market(market, limit=1)
                            if not recent.empty:
                                last_exit = recent.iloc[0]["exit_reason"]
                                if last_exit in ("max_loss", "stop_loss"):
                                    min_wait = risk_cfg.cooldown_after_loss
                                elif last_exit == "take_profit":
                                    min_wait = risk_cfg.cooldown_after_win
                                else:
                                    min_wait = risk_cfg.cooldown_default
                            else:
                                min_wait = risk_cfg.cooldown_default
                            
                            if elapsed < min_wait:
                                logger.info(f"[COOLDOWN] {market} {elapsed:.0f}/{min_wait}min")
                                continue
                        
                        entry_price = signal["price"]
                        total_cap = avail_krw
                        
                        invest = min(
                            total_cap * params.risk_per_trade * risk_cfg.risk_multiplier,
                            avail_krw * risk_cfg.invest_capital_pct
                        )
                        
                        buy_info = calculate_buy_fee_and_total(entry_price, invest / entry_price)
                        if buy_info["total"] > avail_krw:
                            continue
                        if buy_info["total"] < CFG.min_order_krw * risk_cfg.min_order_mult:
                            continue
                        
                        size = invest / entry_price
                        result = execute_order(market, "buy", entry_price, size, risk_cfg)
                        
                        if result["success"]:
                            db_add_position(market, entry_price, size,
                                          signal["stop"], "long", result["fee"])
                            last_entries[market] = current_time
                            avail_krw -= buy_info["total"]
                            logger.info(f"[SUCCESS] {market} Buy completed @ {entry_price:,.0f}")
                    
                    db_set_meta("last_entries", json.dumps(last_entries))
            
            logger.info(f"\n[SLEEP] Waiting {CFG.collect_interval_sec} seconds...\n")
            time.sleep(CFG.collect_interval_sec)
        
        except KeyboardInterrupt:
            logger.info("\n[SHUTDOWN] Bot stopped")
            break
        except Exception as e:
            logger.error(f"[ERROR] Main loop exception: {e}", exc_info=True)
            time.sleep(30)

if __name__ == "__main__":
    run()