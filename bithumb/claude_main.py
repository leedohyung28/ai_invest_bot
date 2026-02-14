import time, json, sqlite3, math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import requests
import pandas as pd
import numpy as np
import os
import hmac
import hashlib
import base64
import urllib.parse
import jwt
from dotenv import load_dotenv
import re
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import threading
import sys



load_dotenv()

# ---------------------------
# 전역 변수 및 설정
# ---------------------------
current_log_file = None
file_handler = None
logger = logging.getLogger("TradingBot")
logger.setLevel(logging.DEBUG)

# 콘솔 핸들러는 한 번만 설정
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ---------------------------
# Gmail 설정
# ---------------------------
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")
TARGET_EMAIL = os.getenv("TARGET_EMAIL")
EMAIL_INTERVAL = 3 * 60 * 60  # 3시간

def get_new_log_filename():
    """현재 시간 기준으로 새 로그 파일명 생성 (예: 2026_02_11_21_30.log)"""
    return datetime.now().strftime("%Y_%m_%d_%H_%M.log")

def setup_logger():
    """새로운 로그 파일을 생성하고 핸들러를 교체함"""
    global current_log_file, file_handler, logger

    # 새 파일명 생성
    new_log_file = get_new_log_filename()

    # 기존 핸들러 제거
    if file_handler:
        logger.removeHandler(file_handler)
        file_handler.close()

    # 새 핸들러 설정
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

    # 1) 포트폴리오
    current_portfolio = get_total_portfolio_value()
    current_value = current_portfolio['total_krw']

    # 2) 3시간 전 대비(=직전 이메일 대비)
    if last_portfolio_value is None:
        last_portfolio_value = current_value

    diff_krw = current_value - last_portfolio_value
    diff_pct = (diff_krw / last_portfolio_value * 100) if last_portfolio_value > 0 else 0

    # 3) 승률/성과
    perf = analyze_trading_performance()

    # 4) AI Config 변경 내역(meta에서 읽기)
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

    # 5) 보유 코인 상세 문자열
    coins_lines = []
    coins = current_portfolio.get("coins", {})
    if coins:
        for coin, info in sorted(coins.items(), key=lambda x: x[1]['value_krw'], reverse=True):
            coins_lines.append(
                f"- {coin}: {info['balance']:.6f}개, 평가 {info['value_krw']:,.0f}원 (@ {info['price']:,.0f})"
            )
    else:
        coins_lines.append("- (보유 코인 없음)")

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

            # 직전 리포트 값 업데이트 (다음 리포트에서 '3시간 전 대비'로 사용)
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

# 초기 로거 설정
setup_logger()

# 백그라운드 스레드 시작 (메인 코드 실행 전 호출 필요)
# threading.Thread(target=log_rotation_scheduler, daemon=True).start()

# ---------------------------
# 전역 변수 추가
# ---------------------------
last_portfolio_value = None  # 이전 포트폴리오 가치
last_email_time = 0  # 마지막 이메일 전송 시각

# ---------------------------
# Config / Parameters
# ---------------------------
@dataclass
class Params:
    max_positions: int = 5
    max_coin_weight: float = 0.5
    risk_per_trade: float = 0.08
    position_allocation_pct: float = 0.7
    atr_mult_stop: float = 2.5
    breakout_lookback: int = 2
    trend_ma_fast: int = 10
    trend_ma_slow: int = 30
    cooldown_minutes: int = 10
    min_volume_mult: float = 0.7


@dataclass
class StrategyConfig:
    """AI가 수정 가능한 전략 설정"""
    use_trend_filter: bool = True
    use_volume_filter: bool = False
    use_volatility_filter: bool = False
    min_volume_mult: float = 0.8
    volatility_mult: float = 0.8
    trailing_stop_profit_threshold: float = 0.012
    max_loss_per_trade: float = 0.025


@dataclass
class Config:
    db_path: str = os.getenv("DB_PATH")
    ollama_url: str = os.getenv("OLLAMA_URL")
    ollama_model: str = os.getenv("OLLAMA_MODEL")
    quote: str = "KRW"
    exclude: tuple = ("BTC", "ETH")  # AI 관리 대상에서 제외
    universe_size: int = 200
    collect_interval_sec: int = 300
    ai_refresh_min: int = 30
    fee_rate: float = 0  # 빗썸 수수료 0.25%
    min_order_krw: float = 7000  # 최소 주문 금액


CFG = Config()
PARAMS = Params()
STRATEGY = StrategyConfig()

def log_print(*args, level="info", sep=" "):
    # (선택) 과거 스타일: log_print("msg", "error")도 허용
    if len(args) == 2 and isinstance(args[1], str) and args[1] in ("debug", "info", "warning", "error"):
        msg, level2 = args
        args = (msg,)
        level = level2

    msg = sep.join(str(a) for a in args)

    if level == "debug":
        logger.debug(msg)
    elif level == "info":
        logger.info(msg)
    elif level == "warning":
        logger.warning(msg)
    elif level == "error":
        logger.error(msg)
    else:
        logger.info(msg)

# ---------------------------
# DB
# ---------------------------
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


def db_get_candles(market, timeframe, limit=200):
    """특정 마켓의 캔들 데이터 조회"""
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
    """최근 AI 설정 변경 내역을 meta에 저장"""
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
    """현재 포지션 조회"""
    con = sqlite3.connect(CFG.db_path)
    df = pd.read_sql_query("SELECT * FROM positions", con)
    con.close()
    return df


def db_add_position(market, entry_price, size, stop_loss, direction, entry_fee):
    """포지션 추가"""
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
    """포지션 제거"""
    con = sqlite3.connect(CFG.db_path)
    cur = con.cursor()
    cur.execute("DELETE FROM positions WHERE market=?", (market,))
    con.commit()
    con.close()


def db_get_recent_trades(limit=100):
    """최근 거래 내역 조회"""
    con = sqlite3.connect(CFG.db_path)
    df = pd.read_sql_query("""
        SELECT * FROM trades 
        ORDER BY exit_time DESC 
        LIMIT ?
    """, con, params=(limit,))
    con.close()
    return df


def db_save_ai_learning(win_rate, avg_profit, avg_loss, total_trades, params_json, strategy_json, performance_score):
    """AI 학습 결과 저장"""
    con = sqlite3.connect(CFG.db_path)
    cur = con.cursor()
    cur.execute("""
        INSERT INTO ai_learning 
        (timestamp, win_rate, avg_profit, avg_loss, total_trades, params_json, strategy_json, performance_score)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (int(time.time()), win_rate, avg_profit, avg_loss, total_trades, params_json, strategy_json, performance_score))
    con.commit()
    con.close()


def db_get_ai_learning_history(limit=10):
    """AI 학습 히스토리 조회"""
    con = sqlite3.connect(CFG.db_path)
    df = pd.read_sql_query("""
        SELECT * FROM ai_learning 
        ORDER BY timestamp DESC 
        LIMIT ?
    """, con, params=(limit,))
    con.close()
    return df


# ---------------------------
# 유틸리티
# ---------------------------
def is_excluded_coin(market):
    """해당 마켓이 제외 대상인지 확인 및 유효성 검사"""
    if not market or '-' not in market:
        return True # 잘못된 형식은 제외
    
    _, currency = market.split('-')
    
    # 1. 사용자 설정 제외 리스트 체크
    if currency.upper() in [x.upper() for x in CFG.exclude]:
        return True
    
    # 2. 티커 길이 체크 (일반적인 코인은 2~5자 이상)
    if len(currency) < 2:
        return True
        
    return False


def analyze_trading_performance():
    """거래 성과 분석"""
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


# ---------------------------
# Bithumb API 2.0 - Private API
# ---------------------------
class BithumbPrivateAPI:
    """빗썸 Private API 2.0 클래스"""
    
    def __init__(self):
        self.api_url = "https://api.bithumb.com"
        self.api_key = os.getenv("BITHUMB_API_KEY")
        self.api_secret = os.getenv("BITHUMB_SECRET_KEY")
        
        if not self.api_key or not self.api_secret:
            raise ValueError("[ERROR] 환경변수에 BITHUMB_API_KEY와 BITHUMB_SECRET_KEY를 설정해주세요")
        
        self.sess = requests.Session()
    
    def _create_jwt_token(self, query_params=None):
        """JWT 토큰 생성 (API 2.0)"""
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

        # pyjwt 버전 차이: bytes면 decode
        if isinstance(token, bytes):
            token = token.decode('utf-8')

        return f'Bearer {token}'
    
    def _api_call(self, method, endpoint, params=None, is_query=False):
        """
        API 2.0 호출
        
        Args:
            method: GET, POST 등
            endpoint: API 엔드포인트
            params: 파라미터
            is_query: True면 쿼리 파라미터, False면 body 파라미터
        """
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
                log_print("[ERROR] HTTP", response.status_code, response.text)
                raise
            
            # API 2.0 에러 체크
            if 'error' in result:
                log_print(f"[ERROR] API 에러: {result['error']['message']}")
                return None
            
            return result
        
        except requests.exceptions.Timeout:
            log_print(f"[ERROR] API 타임아웃: {endpoint}")
            return None
        except requests.exceptions.RequestException as e:
            log_print(f"[ERROR] API 요청 실패: {e}")
            return None
        except Exception as e:
            log_print(f"[ERROR] 예상치 못한 오류: {e}")
            return None
    
    def get_accounts(self):
        """전체 계좌 조회 (API 2.0)"""
        endpoint = "/v1/accounts"
        return self._api_call("GET", endpoint)
    
    def get_order_chance(self, market):
        """
        주문 가능 정보 조회 (API 2.0)
        - 최소 주문 금액
        - 주문 가능 여부
        - 수수료 등
        """
        endpoint = "/v1/orders/chance"
        params = {"market": market}
        return self._api_call("GET", endpoint, params, is_query=True)
    
    def place_order(self, market, side, volume=None, price=None, ord_type="limit"):
        """
        주문하기 (API 2.0)
        
        Args:
            market: 마켓 ID (예: KRW-BTC)
            side: 'bid' (매수) or 'ask' (매도)
            volume: 주문량
            price: 주문 가격 (지정가 주문 시)
            ord_type: 'limit' (지정가) or 'price' (시장가 매수) or 'market' (시장가 매도)
        """
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
        """주문 취소 (API 2.0)"""
        endpoint = "/v1/order"
        params = {"uuid": uuid}
        return self._api_call("DELETE", endpoint, params, is_query=True)
    
    def get_order(self, uuid):
        """개별 주문 조회 (API 2.0)"""
        endpoint = "/v1/order"
        params = {"uuid": uuid}
        return self._api_call("GET", endpoint, params, is_query=True)


# ---------------------------
# Bithumb API 2.0 - Public API
# ---------------------------
class BithumbPublic:
    """빗썸 Public API 2.0 클래스"""
    
    def __init__(self):
        self.sess = requests.Session()
        self.base_url = "https://api.bithumb.com"
    
    def get_markets(self):
        """전체 마켓 코드 조회 (API 2.0)"""
        url = f"{self.base_url}/v1/market/all"
        
        try:
            resp = self.sess.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log_print(f"[ERROR] 마켓 코드 조회 실패: {e}")
            return []
    
    def get_day_candles(self, market: str, count: int = 200, to: str = None):
        """
        일봉 캔들 조회 (API 2.0)
        
        Args:
            market: 마켓 코드 (예: KRW-BTC)
            count: 캔들 개수 (최대 200)
            to: 마지막 캔들 시각 (ISO 8601 형식, 예: 2023-01-01T00:00:00Z)
        """
        url = f"{self.base_url}/v1/candles/minutes/30"
        params = {
            "market": market,
            "count": min(count, 200)
        }
        
        if to:
            params["to"] = to
        
        try:
            resp = self.sess.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            log_print(f"[ERROR] {market} 캔들 조회 실패: {e}")
            return []
    
    def get_current_price(self, market: str):
        """현재가 조회 (API 2.0 - ticker)"""
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
            log_print(f"[ERROR] {market} 현재가 조회 실패: {e}")
            return None


# ---------------------------
# 수수료 계산
# ---------------------------
def calculate_buy_fee_and_total(price, size):
    """매수 시 필요한 총 금액 계산"""
    amount = price * size
    fee = amount * CFG.fee_rate
    total = amount + fee
    
    return {
        "amount": amount,
        "fee": fee,
        "total": total
    }


def calculate_sell_fee_and_net(price, size):
    """매도 시 실제 받을 금액 계산"""
    amount = price * size
    fee = amount * CFG.fee_rate
    net = amount - fee
    
    return {
        "amount": amount,
        "fee": fee,
        "net": net
    }


# ---------------------------
# 잔고 조회
# ---------------------------
def get_available_krw_balance():
    """사용 가능한 KRW 잔고 조회 (API 2.0)"""
    try:
        api = BithumbPrivateAPI()
        result = api.get_accounts()
        
        if result:
            for account in result:
                if account['currency'] == 'KRW':
                    return float(account['balance'])
        return 0
    except Exception as e:
        log_print(f"[ERROR] KRW 잔고 조회 실패: {e}")
        return 0


def get_coin_balance(market):
    """특정 코인 잔고 조회 (API 2.0)"""
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
        log_print(f"[ERROR] {market} 잔고 조회 실패: {e}")
        return 0


# ---------------------------
# 포트폴리오 가치 계산
# ---------------------------
def get_total_portfolio_value(exclude_coins=None):
    """AI 관리 자산의 포트폴리오 가치 계산"""
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
            
            # 제외 코인 스킵
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
        log_print(f"[ERROR] 포트폴리오 가치 계산 실패: {e}")
        return {"krw": 0, "coins": {}, "total_coin_value": 0, "total_krw": 0}


def print_portfolio_summary():
    """AI 관리 자산 포트폴리오 요약"""
    portfolio = get_total_portfolio_value()
    
    log_print(f"{'='*60}")
    log_print(f"[PORTFOLIO] AI 관리 자산 요약")
    log_print(f"  (BTC, ETH 등 수동 투자 자산 제외)")
    log_print(f"{'='*60}")
    log_print(f"KRW 잔고: {portfolio['krw']:,.0f}원")
    log_print(f"코인 평가액: {portfolio['total_coin_value']:,.0f}원")
    log_print(f"총 자산: {portfolio['total_krw']:,.0f}원")
    
    if portfolio['coins']:
        log_print(f"AI 관리 코인 ({len(portfolio['coins'])}개):")
        for coin, info in sorted(portfolio['coins'].items(), 
                                key=lambda x: x[1]['value_krw'], 
                                reverse=True):
            log_print(f"  - {coin}: {info['balance']:.6f} "
                  f"= {info['value_krw']:,.0f}원 (@ {info['price']:,.0f})")
    
    log_print(f"{'='*60}\n")


# ---------------------------
# 캔들 데이터 변환
# ---------------------------
def parse_candles_to_rows(candles_data, market, timeframe="30m"):
    """API 2.0 캔들 응답을 DB 저장용 rows로 변환"""
    rows = []
    
    for candle in candles_data:
        try:
            # API 2.0 응답 구조
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


# ---------------------------
# 기술적 지표 계산
# ---------------------------
def calculate_atr(df, period=14):
    """ATR (Average True Range) 계산"""
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
    """이동평균 계산"""
    if len(df) < period:
        return pd.Series([0] * len(df))
    return df['close'].rolling(period).mean().fillna(0)


def calculate_swing_high_low(df, lookback):
    """스윙 고점/저점 계산"""
    if len(df) < lookback:
        return pd.Series([0] * len(df)), pd.Series([0] * len(df))
    
    swing_high = df['high'].rolling(lookback).max()
    swing_low = df['low'].rolling(lookback).min()
    
    return swing_high.fillna(0), swing_low.fillna(0)


# ---------------------------
# 스윙 전략 시그널 생성
# ---------------------------
def generate_swing_signals(market, params: Params, strategy: StrategyConfig):
    """스윙 트레이딩 시그널 생성"""
    if is_excluded_coin(market):
        return {"signal": "hold", "reason": "excluded_coin"}
    
    df = db_get_candles(market, "30m", limit=200)
    if len(df) < max(params.trend_ma_slow, params.breakout_lookback) + 5:
        return {"signal": "hold", "reason": "insufficient_data"}
    
    # 지표 계산
    df['atr'] = calculate_atr(df, 14)
    df['ma_fast'] = calculate_ma(df, params.trend_ma_fast)
    df['ma_slow'] = calculate_ma(df, params.trend_ma_slow)
    df['swing_high'], df['swing_low'] = calculate_swing_high_low(df, params.breakout_lookback)
    df['volume_ma'] = df['volume'].rolling(20).mean()
    
    # RSI 추가 (과매도 구간 탐지)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    last = df.iloc[-1]
    prev = df.iloc[-2]
    current_price = last['close']
    atr = last['atr']
    
    # ===== 완화된 진입 조건 =====
    
    # 1) 추세 조건 완화: MA 교차 임박 or 정배열
    trend_up = (
        last['ma_fast'] > last['ma_slow'] or  # 기존 조건
        (last['ma_fast'] > prev['ma_fast'] and last['close'] > last['ma_fast'])  # 상승 추세 시작
    ) if strategy.use_trend_filter else True
    
    # 2) 브레이크아웃 조건 완화
    # 기존: 6봉 고점 돌파 (너무 엄격)
    # 개선: 3~4봉 고점 근접 or 직전 고점 돌파
    recent_high_3 = df['high'].tail(3).max()  # 최근 3봉 고점
    recent_high_4 = df['high'].tail(4).max()  # 최근 4봉 고점
    
    breakout_up = (
        last['high'] > df['swing_high'].iloc[-2] or  # 기존 조건 유지
        current_price > recent_high_3 * 0.998 or  # 3봉 고점 근접 (0.2% 이내)
        (current_price > prev['high'] and last['close'] > last['open'])  # 직전 고점 돌파 + 양봉
    )
    
    # 3) 거래량 조건 완화
    volume_confirm = (
        last['volume'] > (last['volume_ma'] * strategy.min_volume_mult * 0.8)  # 80%로 완화
    ) if strategy.use_volume_filter else True
    
    # 4) 변동성 필터는 선택적으로만
    atr_avg = df['atr'].tail(20).mean()
    volatility_ok = (atr > atr_avg * strategy.volatility_mult * 0.7) if strategy.use_volatility_filter else True
    
    # 5) RSI 과매도 조건 추가 (선택적 진입)
    rsi = last['rsi']
    rsi_oversold = rsi < 35  # RSI 35 이하면 과매도
    rsi_buy = 30 < rsi < 50  # RSI 30~50 구간에서 매수
    
    # ===== 매수 시그널 (다양한 진입 패턴) =====
    
    # 패턴 1: 기존 브레이크아웃
    if trend_up and breakout_up and volume_confirm and volatility_ok:
        stop_loss = current_price - (atr * params.atr_mult_stop)
        return {
            "signal": "buy",
            "price": current_price,
            "stop": stop_loss,
            "atr": atr,
            "reason": "breakout_up"
        }
    
    # 패턴 2: RSI 과매도 반등 (추가)
    if rsi_oversold and last['close'] > last['open'] and last['volume'] > last['volume_ma'] * 1.2:
        stop_loss = current_price - (atr * 2.5)  # 더 타이트한 손절
        return {
            "signal": "buy",
            "price": current_price,
            "stop": stop_loss,
            "atr": atr,
            "reason": "rsi_oversold_bounce"
        }
    
    # 패턴 3: 이평선 지지 반등 (추가)
    ma_support = (
        last['low'] <= last['ma_fast'] <= last['high'] and  # MA가 봉 내부
        last['close'] > last['ma_fast'] and  # 종가는 MA 위
        last['close'] > last['open']  # 양봉
    )
    if ma_support and rsi_buy and volume_confirm:
        stop_loss = last['ma_fast'] - (atr * 1.5)  # MA 아래 손절
        return {
            "signal": "buy",
            "price": current_price,
            "stop": stop_loss,
            "atr": atr,
            "reason": "ma_support"
        }
    
    # 패턴 4: 단순 상승 모멘텀 (가장 완화된 조건)
    momentum_up = (
        last['close'] > prev['close'] and
        last['close'] > last['open'] and
        last['volume'] > last['volume_ma'] * 0.9 and
        rsi_buy
    )
    if momentum_up and not strategy.use_trend_filter:  # 필터 꺼져있을 때만
        stop_loss = current_price - (atr * 2.0)
        return {
            "signal": "buy",
            "price": current_price,
            "stop": stop_loss,
            "atr": atr,
            "reason": "momentum"
        }
    
    return {"signal": "hold", "reason": "no_setup"}


# ---------------------------
# 포지션 관리
# ---------------------------
def check_position_exit(market, current_price, position_row, params: Params, strategy: StrategyConfig):
    if is_excluded_coin(market):
        return False, None
    
    entry_price = position_row['entry_price']
    stop_loss = position_row['stop_loss']
    direction = position_row['direction']
    
    # 수익률 계산
    if direction == "long":
        profit_pct = (current_price - entry_price) / entry_price
    else:  # short
        profit_pct = (entry_price - current_price) / entry_price
    
    # 데이터 수집
    df = db_get_candles(market, "30m", limit=150)
    if len(df) < 14:
        return False, None
    
    atr = calculate_atr(df, 14).iloc[-1]
    
    # ===== 익절 조건 추가 =====
    # 1) 목표 수익률 도달 시 익절 (1.5% 이익)
    target_profit = 0.015  # 1.5%
    if profit_pct >= target_profit:
        log_print(f" [TAKE_PROFIT] {market} 목표 수익 달성: {profit_pct*100:.2f}%")
        return True, "take_profit"
    
    # 2) 중간 익절: 0.8% 이상 수익 시 절반 익절 (부분 실현)
    # 이 로직은 execute에서 처리하거나, 여기서는 전체 청산만 다룸
    
    # 3) 고점 대비 하락 시 익절 (이익 구간에서의 추세 반전)
    if profit_pct > 0.005:  # 0.5% 이상 수익 구간
        recent_high = df['high'].tail(10).max()
        pullback_pct = (recent_high - current_price) / recent_high
        
        # 고점 대비 1% 이상 하락하면 익절
        if pullback_pct > 0.01:
            log_print(f" [PULLBACK_EXIT] {market} 고점 대비 하락: {pullback_pct*100:.2f}%")
            return True, "pullback_from_profit"
    
    # ===== 개선된 손절 조건 =====
    # 1) 기존 손절가 도달
    if direction == "long" and current_price <= stop_loss:
        # 하지만 손실이 너무 작으면 (< -0.5%) 대기
        if profit_pct < -0.005:
            return True, "stop_loss"
        else:
            return False, None  # 미세 손실은 무시
    
    if direction == "short" and current_price >= stop_loss:
        if profit_pct < -0.005:
            return True, "stop_loss"
        else:
            return False, None
    
    # 2) 최대 손실 제한 (strategy 설정 기반)
    max_loss = strategy.max_loss_per_trade  # 기본 2%
    if profit_pct <= -max_loss:
        log_print(f" [MAX_LOSS] {market} 최대 손실 도달: {profit_pct*100:.2f}%")
        return True, "max_loss"
    
    # ===== 트레일링 스톱 (수익 보호) =====
    if direction == "long":
        # 수익이 1.5% 이상일 때만 트레일링 스톱 활성화
        if profit_pct > strategy.trailing_stop_profit_threshold:
            # 현재가 기준 아래 ATR 2배 위치로 스톱 상향
            trailing_stop = current_price - (atr * 2.0)  # 기존 3배 → 2배로 타이트하게
            
            if trailing_stop > stop_loss:
                con = sqlite3.connect(CFG.db_path)
                cur = con.cursor()
                cur.execute("UPDATE positions SET stop_loss=? WHERE market=?",
                            (trailing_stop, market))
                con.commit()
                con.close()
                log_print(f" [TRAILING] {market} 스톱 업데이트: {stop_loss:,.0f} → {trailing_stop:,.0f}원 (수익: {profit_pct*100:.2f}%)")
    
    # 3) 장기 보유 시 자동 청산 (48시간 이상)
    holding_hours = (time.time() - position_row['entry_time']) / 3600
    if holding_hours > 48:
        if profit_pct > 0:  # 수익이라면 청산
            log_print(f" [LONG_HOLD] {market} 장기 보유 익절: {holding_hours:.1f}시간")
            return True, "long_hold_profit"
        elif profit_pct < -0.015:  # 손실 1.5% 이상이면 청산
            log_print(f" [LONG_HOLD] {market} 장기 보유 손절: {holding_hours:.1f}시간")
            return True, "long_hold_loss"
    
    return False, None


# ---------------------------
# 주문 실행 (API 2.0)
# ---------------------------
def execute_order(market, direction, price, size, wait_sec=30):
    """
    주문 실행 함수
    - 매수: 지정가 실패 → 시장가 매수
    - 매도: 3회 단계적 가격 하향(1/3씩) → 시장가 매도
    """
    if is_excluded_coin(market):
        return {"success": False, "order_id": None, "message": "Excluded coin", "fee": 0}

    api = BithumbPrivateAPI()

    # ===========================
    # [매수 Logic] 기존 유지
    # ===========================
    if direction == "buy":
        # 1. 지정가 주문
        result = api.place_order(
            market=market,
            side="bid",
            price=str(int(price)),
            volume=str(size),
            ord_type="limit"
        )

        if result and "uuid" in result:
            order_id = result["uuid"]
            # 체결 대기
            start_time = time.time()
            while time.time() - start_time < wait_sec:
                order = api.get_order(order_id)
                if order and order.get("state") == "done":
                    fee_info = calculate_buy_fee_and_total(price, size)
                    return {
                        "success": True, 
                        "order_id": order_id, 
                        "message": "Filled", 
                        "fee": fee_info["fee"]
                    }
                time.sleep(1)
            # 미체결 취소
            api.cancel_order(order_id)
        
        # 2. 시장가 매수
        log_print(f"[MARKET BUY] 지정가 실패 → 시장가 매수 실행")
        market_result = api.place_order(
            market=market,
            side="bid",
            price=str(int(price * size)), # KRW 총액
            ord_type="price" # 시장가 매수(KRW)
        )
        
        if market_result and "uuid" in market_result:
            fee_info = calculate_buy_fee_and_total(price, size)
            return {
                "success": True, 
                "order_id": market_result["uuid"], 
                "message": "Market buy", 
                "fee": fee_info["fee"]
            }
        return {"success": False, "order_id": None, "message": "Market buy failed", "fee": 0}

    # ===========================
    # [매도 Logic] 3단계 하향 조정
    # ===========================
    else: # direction == "sell"
        pub = BithumbPublic()
        original_price = price
        
        # 3회 시도 (0, 1, 2)
        for i in range(3):
            try_price = original_price
            
            # 2회차, 3회차는 가격 조정
            if i > 0:
                current_market_price = pub.get_current_price(market)
                if current_market_price:
                    gap = original_price - current_market_price
                    if gap > 0:
                        # 1/3씩 하향 (Gap * 1/3, Gap * 2/3)
                        decrement = gap * (i / 3.0)
                        try_price = original_price - decrement
                        try_price = adjust_price_to_tick(try_price) # 호가 단위 맞춤
                        
                        log_print(f"[RETRY SELL] {i+1}/3회차: 목표가 하향 {original_price:,.0f} -> {try_price:,.0f} (Gap: {gap:,.0f})")
            
            # 지정가 주문
            result = api.place_order(
                market=market,
                side="ask",
                price=str(int(try_price)),
                volume=str(size),
                ord_type="limit"
            )
            
            if not result or "uuid" not in result:
                log_print(f"[ERROR] 매도 주문 에러 (Attempt {i+1})")
                time.sleep(1)
                continue
                
            order_id = result["uuid"]
            
            # 체결 대기
            start_time = time.time()
            filled = False
            while time.time() - start_time < wait_sec:
                order = api.get_order(order_id)
                if order and order.get("state") == "done":
                    filled = True
                    break
                time.sleep(1)
                
            if filled:
                fee_info = calculate_sell_fee_and_net(try_price, size)
                return {
                    "success": True, 
                    "order_id": order_id, 
                    "message": "Filled", 
                    "fee": fee_info["fee"]
                }
            
            # 미체결 시 취소하고 다음 루프(가격 낮춰서 재주문)
            api.cancel_order(order_id)
            
        # 3. 시장가 매도 (3번 실패 시)
        log_print(f"[MARKET SELL] 3회 지정가 실패 → 시장가 매도")
        market_result = api.place_order(
            market=market,
            side="ask",
            volume=str(size),
            ord_type="market" # 시장가 매도(수량)
        )
        
        if market_result and "uuid" in market_result:
            fee_info = calculate_sell_fee_and_net(price, size)
            return {
                "success": True, 
                "order_id": market_result["uuid"], 
                "message": "Market sell", 
                "fee": fee_info["fee"]
            }
            
        return {"success": False, "order_id": None, "message": "Market sell failed", "fee": 0}

# ---------------------------
# 소액 잔고 정리(Dusting) 헬퍼 함수
# ---------------------------
def clear_dust_position(market, current_price, actual_balance, api):
    """
    최소 주문 금액 미달 포지션 청산 로직
    - 전략: 최소 금액만큼 추가 매수 -> 합쳐서 즉시 전량 매도
    """
    needed_krw = 6000  # 최소 주문(5000원)보다 넉넉하게 잡음
    available_krw = get_available_krw_balance()
    
    if available_krw < needed_krw:
        log_print(f"[DUST_FAIL] {market} 소액 정리 실패: KRW 잔고 부족 ({available_krw:,.0f}원)")
        return False, 0
    
    log_print(f"[DUST_ACTION] {market} 소액 청산 시도: {needed_krw}원 추가 매수 후 전량 매도")
    
    # 1. 시장가 매수 (덩치 키우기)
    buy_res = api.place_order(
        market=market,
        side="bid",
        price=str(needed_krw),
        ord_type="price" # 시장가 매수
    )
    
    if not buy_res or "uuid" not in buy_res:
        log_print(f"[DUST_ERROR] 추가 매수 실패")
        return False, 0
        
    time.sleep(1.5) # 체결 대기
    
    # 2. 잔고 재조회 (합쳐진 수량 확인)
    total_balance = get_coin_balance(market)
    
    # 3. 시장가 전량 매도
    sell_res = api.place_order(
        market=market,
        side="ask",
        volume=str(total_balance),
        ord_type="market" # 시장가 매도
    )
    
    if sell_res and "uuid" in sell_res:
        log_print(f"[DUST_SUCCESS] 소액 청산 완료")
        # 수수료 대략 계산 (매수+매도)
        dust_fee = (needed_krw * 0.0025) + (total_balance * current_price * 0.0025)
        return True, dust_fee
    else:
        log_print(f"[DUST_ERROR] 전량 매도 실패")
        return False, 0


# ---------------------------
# 리밸런싱
# ---------------------------
def rebalance_portfolio(universe, params: Params, strategy: StrategyConfig):
    """포트폴리오 리밸런싱 (수정됨: 잔고 동기화 및 투자금액 제한 해제)"""
    positions = db_get_positions()
    pub = BithumbPublic()
    priv_api = BithumbPrivateAPI()

    # 시작 전 DB 포지션과 실제 잔고 강제 동기화 (Sync)
    # 앱에서 매도했거나 수량이 바뀐 경우를 봇이 매번 인지하도록 함
    if not positions.empty:
        log_print("[SYNC] 보유 포지션 실제 잔고 동기화 중...")
        for _, pos in positions.iterrows():
            market = pos['market']
            # 실제 잔고 조회
            real_balance = get_coin_balance(market)
            
            # 잔고가 없으면(이미 팔았으면) DB에서 삭제
            if real_balance <= 0.0001:  # 0에 가까우면 삭제
                log_print(f"[SYNC] {market} 실제 잔고 없음 -> DB에서 포지션 삭제")
                db_remove_position(market)
            # 잔고가 DB와 다르면 업데이트 (부분 매도 등)
            elif abs(real_balance - pos['size']) / pos['size'] > 0.05: # 5% 이상 차이나면
                log_print(f"[SYNC] {market} 수량 불일치 보정 ({pos['size']} -> {real_balance})")
                con = sqlite3.connect(CFG.db_path)
                cur = con.cursor()
                cur.execute("UPDATE positions SET size=? WHERE market=?", (real_balance, market))
                con.commit()
                con.close()

    # 동기화 후 포지션 다시 로드
    positions = db_get_positions()

    # 1) 기존 포지션 관리 (청산 로직)
    for _, pos in positions.iterrows():
        market = pos['market']
        
        # 제외 코인 무시
        if is_excluded_coin(market):
            continue

        current_price = pub.get_current_price(market)
        if current_price is None:
            continue

        should_exit, reason = check_position_exit(market, current_price, pos, params, strategy)

        if should_exit:
            # 실시간 잔고 재조회
            actual_balance = get_coin_balance(market)
            
            # (이미 위에서 동기화 했지만, 찰나의 순간을 위해 다시 체크)
            if actual_balance <= 0:
                db_remove_position(market)
                continue

            # 최소 주문 금액 미달 처리 로직
            sell_value = current_price * actual_balance
            if sell_value < CFG.min_order_krw:
                log_print(f"[INFO] {market} 청산 금액({sell_value:,.0f}원)이 최소 주문금액 미달")
                
                # AI 판단 및 소액 정리 로직 (기존 유지)
                analysis = generate_swing_signals(market, params, strategy)
                is_bullish = (analysis["signal"] == "buy")
                is_urgent_exit = (reason in ["stop_loss", "max_loss", "long_hold_loss"])

                if is_bullish and not is_urgent_exit:
                    log_print(f" └ [AI 판단] 📈 상승/반등 가능성 있음 (Reason: {reason}) -> 홀딩 및 대기")
                    continue
                else:
                    log_print(f" └ [AI 판단] 📉 하락 리스크 큼 (Reason: {reason}) -> '물타기 후 탈출' 시도")
                    success, fee = clear_dust_position(market, current_price, actual_balance, priv_api)
                    if success:
                        db_remove_position(market)
                    continue

            log_print(f"[EXIT] 청산: {market} @ {current_price:,.0f}원 (수량: {actual_balance}) 사유: {reason}")

            # 매도 실행
            exit_result = execute_order(market, "sell", current_price, actual_balance)

            if exit_result["success"]:
                pnl_gross = (current_price - pos['entry_price']) * pos['size']
                total_fee = pos.get('entry_fee', 0) + exit_result['fee']
                pnl_net = pnl_gross - total_fee
                if pos['direction'] == "short": # 현물에선 없지만 로직 유지
                    pnl_net = -pnl_net
                
                holding_hours = (time.time() - pos['entry_time']) / 3600
                
                con = sqlite3.connect(CFG.db_path)
                cur = con.cursor()
                cur.execute("""
                    INSERT INTO trades 
                    (market, direction, entry_price, exit_price, size, entry_time, exit_time, pnl, total_fee, holding_hours, exit_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (market, pos['direction'], pos['entry_price'], current_price, 
                      pos['size'], pos['entry_time'], int(time.time()), pnl_net, total_fee, holding_hours, reason))
                con.commit()
                con.close()
                
                log_print(f" 순손익: {pnl_net:+,.0f}원 (보유: {holding_hours:.1f}시간)")
                db_remove_position(market)

    # 2) 신규 진입 탐색
    # 현재 관리 중인 포지션 수 재확인
    current_positions_df = db_get_positions()
    ai_managed_positions = 0
    if not current_positions_df.empty:
        for _, pos in current_positions_df.iterrows():
            if not is_excluded_coin(pos['market']):
                ai_managed_positions += 1

    if ai_managed_positions >= params.max_positions:
        log_print(f"[INFO] 최대 포지션 수 도달 ({ai_managed_positions}/{params.max_positions})")
        return

    # 잔고 조회
    available_krw = get_available_krw_balance()
    if available_krw < CFG.min_order_krw:
        log_print(f"[INFO] 잔고 부족 ({available_krw:,.0f}원)")
        return

    log_print(f"[INFO] 사용 가능 잔고: {available_krw:,.0f}원")

    # 진입 쿨타임 체크 등 기존 로직
    last_entries = db_get_meta("last_entries", "{}")
    last_entries = json.loads(last_entries)
    current_time = time.time()
    
    signals = []
    for market in universe:
        if is_excluded_coin(market): continue
        
        # 쿨타임 체크
        if market in last_entries:
            elapsed_minutes = (current_time - last_entries[market]) / 60
            if market in current_positions_df['market'].values:
                if elapsed_minutes < params.cooldown_minutes: continue
            else:
                if elapsed_minutes < 5: continue
        
        # 이미 보유중이면 패스
        if not current_positions_df.empty and market in current_positions_df['market'].values:
            continue

        signal = generate_swing_signals(market, params, strategy)
        if signal["signal"] in ["buy"]: # 매수 시그널만
            signals.append((market, signal))

    if not signals:
        log_print("[INFO] 진입 가능한 시그널 없음")
        return

    log_print(f"[SIGNAL] 감지된 시그널: {len(signals)}개")

    # 3) 포지션 크기 계산 및 주문 [수정 2: 투자 비중 확대]
    available_slots = params.max_positions - ai_managed_positions
    if available_slots <= 0: return

    # 균등 분할 방식: (가용 잔고 * 투자비중) / 남은 자리 수
    # 예: 잔고 1000만원, 비중 100%(1.0), 남은 자리 5개 -> 종목당 200만원
    # 안전을 위해 한 번에 전액을 다 쓰지 않도록 가용 잔고의 99%만 계산에 사용
    
    allocatable_krw = available_krw * params.position_allocation_pct # 투자 할당 비중 (0.7 등)
    position_size_krw = allocatable_krw / available_slots
    
    # 너무 작은 금액 방지 (최소 1만원 이상일 때만)
    if position_size_krw < 10000:
        position_size_krw = available_krw  # 잔액이 적으면 그냥 잔액 전부 투입 (단, min_order_krw 체크는 뒤에서 함)

    log_print(f"[DEBUG] 1종목당 투자 예정 금액: {position_size_krw:,.0f}원 (남은 슬롯: {available_slots}개)")

    for i, (market, signal) in enumerate(signals[:available_slots]):
        if signal["signal"] == "sell": continue # 매도 시그널은 진입 안함
        
        # 리스크 기반 수량 계산 (기존 로직 활용하되, 한도 제한 해제)
        risk_krw = position_size_krw * params.risk_per_trade
        price_risk = abs(signal["price"] - signal["stop"])
        
        size = 0
        if price_risk > 0:
            size = risk_krw / price_risk
        
        # 다만, 계산된 size가 할당된 금액(position_size_krw)을 넘지 않도록 캡
        max_size_by_capital = position_size_krw / signal["price"]
        
        # 코인별 최대 비중(max_coin_weight) 체크
        # (전체 자산 대비 너무 큰 비중 방지)
        # 하지만 여기선 position_size_krw 자체가 이미 분할된 금액이므로, 
        # size가 max_size_by_capital보다 크지만 않게 하면 됨.
        
        if size == 0 or size > max_size_by_capital:
             size = max_size_by_capital # 리스크 계산 무시하고 자금력만큼 매수 (공격적)

        # 최종 수량 및 금액 계산
        buy_info = calculate_buy_fee_and_total(signal["price"], size)

        # 최소 주문 금액 체크
        if buy_info["total"] < CFG.min_order_krw:
            min_size = adjust_min_order_size(signal["price"])
            min_buy_info = calculate_buy_fee_and_total(signal["price"], min_size)
            
            if min_buy_info["total"] <= available_krw:
                log_print(f"[ADJUST] 최소 주문금액 보정 → 수량 {size:.6f} → {min_size:.6f}")
                size = min_size
                buy_info = min_buy_info
            else:
                log_print(f"[SKIP] 최소 주문금액 맞추기엔 잔고 부족")
                continue

        # 실제 잔고 재확인 (루프 돌면서 잔고 줄어듦)
        if buy_info["total"] > available_krw:
            log_print(f"[SKIP] {market} 잔고 부족 (필요: {buy_info['total']:,.0f}, 잔고: {available_krw:,.0f})")
            continue

        log_print(f"[ENTRY] 진입 준비: {market}")
        log_print(f" 가격: {signal['price']:,.0f}원")
        log_print(f" 수량: {size:.6f}")
        log_print(f" 총액: {buy_info['total']:,.0f}원")
        log_print(f" 손절가: {signal['stop']:,.0f}원")

        entry_price = get_entry_price(signal["price"], "buy")
        
        order_result = execute_order(market, "buy", entry_price, size)

        if order_result["success"]:
            log_print(f"[SUCCESS] 진입 완료: {market}")
            db_add_position(
                market, 
                signal["price"], # 진입가는 시그널 가격 기준 or 체결가
                size, 
                signal["stop"], 
                "long", 
                order_result['fee']
            )
            
            last_entries[market] = current_time
            db_set_meta("last_entries", json.dumps(last_entries))
            
            # 잔고 차감 반영
            available_krw -= buy_info["total"]
            log_print(f" 남은 잔고: {available_krw:,.0f}원")
            
            time.sleep(0.5)

def get_entry_price(price, direction):
    """
    체결 우선 지정가
    """
    if direction == "buy":
        return adjust_price_to_tick(price * 1.001)
    else:
        return adjust_price_to_tick(price * 0.999)

def adjust_min_order_size(price, min_krw=7000, buffer=1.02):
    """
    빗썸 지정가 최소주문 보정
    """
    return math.ceil((min_krw * buffer) / price * 1e8) / 1e8

def adjust_price_to_tick(price):
    """빗썸 호가 단위에 맞춰 가격 조정"""
    if price < 100:
        return round(price, 1)  # 0.1원 단위
    elif price < 1000:
        return round(price)  # 1원 단위
    elif price < 10000:
        return round(price / 5) * 5  # 5원 단위
    elif price < 100000:
        return round(price / 10) * 10  # 10원 단위
    else:
        return round(price / 100) * 100  # 100원 단위
    
# ---------------------------
# Market List
# ---------------------------
def build_universe():
    """유니버스 구성 (API 2.0)"""
    try:
        pub = BithumbPublic()
        markets = pub.get_markets()
        
        if not markets:
            return []
        
        # KRW 마켓만 필터링, BTC/ETH 제외
        krw_markets = []
        for market in markets:
            market_id = market['market']
            if market_id.startswith('KRW-'):
                currency = market_id.split('-')[1]
                if currency.upper() not in [x.upper() for x in CFG.exclude]:
                    krw_markets.append(market_id)
        
        log_print(f"[UNIVERSE] AI 관리 유니버스: {len(krw_markets)}개")
        log_print(f"  제외 (사용자 직접 투자): {', '.join(CFG.exclude)}")
        
        return krw_markets[:CFG.universe_size]
    
    except Exception as e:
        log_print(f"[ERROR] 유니버스 구성 실패: {e}")
        return []

def filter_fields(cls, data: dict):
    return {k: v for k, v in data.items() if k in cls.__dataclass_fields__}

# ---------------------------
# AI 리밸런싱 로그 추가
# ---------------------------
def log_param_changes(old_params, new_params, old_strategy, new_strategy):
    """파라미터 및 전략 변경 사항을 상세 로그로 기록"""
    changes = []

    oldp = asdict(old_params)
    newp = asdict(new_params)
    for key in oldp:
        if oldp[key] != newp[key]:
            changes.append(f"PARAM {key}: {oldp[key]} -> {newp[key]}")

    olds = asdict(old_strategy)
    news = asdict(new_strategy)
    for key in olds:
        if olds[key] != news[key]:
            changes.append(f"STRATEGY {key}: {olds[key]} -> {news[key]}")

    if changes:
        log_print("[AI_REBALANCE] Config 변경 감지:")
        for c in changes:
            log_print(" - " + c)
        text = "\n".join(changes)
    else:
        log_print("[AI_REBALANCE] Config 변경 없음")
        text = "(변경 없음)"

    return text
# ---------------------------
# AI Controller
# ---------------------------
def ollama_update_params_and_strategy(performance: dict, learning_history: pd.DataFrame, current_params: Params, current_strategy: StrategyConfig):
    """AI 파라미터 및 전략 동적 조정"""
    sys = (
        "You are a quantitative crypto trading AI.\n"
        "Your task is to update trading parameters.\n\n"
        "STRICT RULES:\n"
        "1. Output ONLY valid JSON\n"
        "2. Do NOT add explanations\n"
        "3. Do NOT add comments\n"
        "4. Do NOT wrap JSON in markdown\n"
        "5. If total_trades == 0, return EMPTY objects\n\n"
        "JSON SCHEMA:\n"
        "{\n"
        '  "params": { OPTIONAL PARAM UPDATES },\n'
        '  "strategy": { OPTIONAL STRATEGY UPDATES }\n'
        "}\n\n"
        "VALID EXAMPLE:\n"
        "{\n"
        '  "params": {"risk_per_trade": 0.004},\n'
        '  "strategy": {"min_volume_mult": 1.2}\n'
        "}\n\n"
        "EMPTY EXAMPLE:\n"
        "{\n"
        '  "params": {},\n'
        '  "strategy": {}\n'
        "}"
    )
    
    learning_summary = []
    if not learning_history.empty:
        for _, row in learning_history.head(5).iterrows():
            learning_summary.append({
                "timestamp": row['timestamp'],
                "win_rate": row['win_rate'],
                "performance_score": row['performance_score']
            })
    
    user = {
        "task": "과거 거래 성과를 분석하고 파라미터와 전략을 개선하라.",
        "current_performance": performance,
        "learning_history": learning_summary,
        "current_params": asdict(current_params),
        "current_strategy": asdict(current_strategy)
    }

    payload = {
        "model": CFG.ollama_model,
        "messages": [
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "temperature": 0.1,
        "stream": False
    }
    
    try:
        resp = requests.post(CFG.ollama_url, json=payload, timeout=60)
        resp.raise_for_status()
        content = resp.json()["message"]["content"].strip()
        
        match = re.search(r'\{[\s\S]*\}', content)
        if not match:
            raise ValueError(f"JSON not found: {content}")
        
        obj = json.loads(match.group())
        
        new_params = Params(**filter_fields(
            Params,
            {**asdict(current_params), **obj.get("params", {})}
        ))
        
        new_strategy = StrategyConfig(**filter_fields(
            StrategyConfig,
            {**asdict(current_strategy), **obj.get("strategy", {})}
        ))
        
        # ✨ 변경 사항 로그 추가
        change_text = log_param_changes(PARAMS, new_params, STRATEGY, new_strategy)
        save_ai_change_summary(change_text)

        performance_score = (
            performance["win_rate"] * performance["profit_factor"]
            if performance["total_trades"] > 0 else 0
        )
        
        db_save_ai_learning(
            win_rate=performance["win_rate"],
            avg_profit=performance["avg_profit"],
            avg_loss=performance["avg_loss"],
            total_trades=performance["total_trades"],
            params_json=json.dumps(asdict(new_params)),
            strategy_json=json.dumps(asdict(new_strategy)),
            performance_score=performance_score
        )
        
        return new_params, new_strategy
    
    except Exception as e:
        log_print(f"[ERROR] AI 갱신 실패: {e}")
        return current_params, current_strategy


# ---------------------------
# Main Loop
# ---------------------------
def run():
    log_print("="*60)
    log_print("[START] 빗썸 AI 스윙 트레이딩 봇 (API 2.0)")
    log_print("  - AI 관리: BTC, ETH 제외 알트코인")
    log_print("  - API 2.0 사용")
    log_print("="*60)
    
    init_db()
    pub = BithumbPublic()

    log_print("[LOADING] 마켓 데이터 로딩...")
    universe = build_universe()
    
    if not universe:
        log_print("[ERROR] 마켓 리스트 조회 실패")
        return
    
    log_print(f"[SUCCESS] AI 관리 유니버스: {len(universe)}개")
    log_print(f"  상위 5개: {', '.join(universe[:5])}")
    
    print_portfolio_summary()

    last_ai = 0
    cycle = 0
    
    global PARAMS, STRATEGY
    
    while True:
        t0 = time.time()
        cycle += 1
        log_print(f"{'='*60}")
        log_print(f"[CYCLE #{cycle}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log_print(f"{'='*60}")

        # 1) 캔들 수집
        log_print("[DATA] 캔들 수집...")
        collected = 0
        
        for market in universe:
            if is_excluded_coin(market):
                continue
            
            try:
                data = pub.get_day_candles(market, count=200)
                
                if data:
                    rows = parse_candles_to_rows(data, market, "30m")
                    
                    if rows:
                        db_put_candles(rows)
                        collected += 1
                
                time.sleep(0.1)
                
            except Exception as e:
                log_print(f"[ERROR] {market} 수집 실패: {e}")
        
        log_print(f"[SUCCESS] {collected}개 수집 완료")

        # 2) AI 학습
        if time.time() - last_ai > CFG.ai_refresh_min * 60:
            log_print("[AI] 학습 시작...")
            
            performance = analyze_trading_performance()
            
            log_print(f"[PERFORMANCE]")
            log_print(f"  총 거래: {performance['total_trades']}회")
            log_print(f"  승률: {performance['win_rate']*100:.1f}%")
            log_print(f"  순손익: {performance['total_pnl']:+,.0f}원")
            
            learning_history = db_get_ai_learning_history(limit=10)
            
            PARAMS, STRATEGY = ollama_update_params_and_strategy(
                performance=performance,
                learning_history=learning_history,
                current_params=PARAMS,
                current_strategy=STRATEGY
            )
            
            db_set_meta("params", json.dumps(asdict(PARAMS)))
            db_set_meta("strategy", json.dumps(asdict(STRATEGY)))
            last_ai = time.time()
            
            log_print(f"[AI] 학습 완료")

        # 3) 리밸런싱
        log_print("[REBALANCE] 리밸런싱...")
        rebalance_portfolio(universe, PARAMS, STRATEGY)
        
        # 4) 포지션 요약
        positions = db_get_positions()
        if not positions.empty:
            log_print(f"[POSITION] 현재 포지션:")
            
            ai_positions = []
            excluded_positions = []
            
            for _, pos in positions.iterrows():
                if is_excluded_coin(pos['market']):
                    excluded_positions.append(pos['market'])
                else:
                    ai_positions.append(pos)
            
            if excluded_positions:
                log_print(f"  제외 (사용자 직접 투자): {', '.join(excluded_positions)}")
            
            if ai_positions:
                log_print(f"  AI 관리: {len(ai_positions)}개")
                total_pnl = 0
                
                for pos in ai_positions:
                    current_price = pub.get_current_price(pos['market'])
                    if current_price:
                        pnl_krw = (current_price - pos['entry_price']) * pos['size']
                        total_pnl += pnl_krw
                        log_print(f"    {pos['market']}: {pnl_krw:+,.0f}원")
                
                log_print(f"  미실현 손익: {total_pnl:+,.0f}원")
        else:
            log_print("[POSITION] 포지션 없음")
        
        # 5) 포트폴리오 요약
        print_portfolio_summary()

        # Sleep
        elapsed = time.time() - t0
        sleep_time = max(1, CFG.collect_interval_sec - elapsed)
        log_print(f"[WAIT] {sleep_time:.0f}초 대기...")
        time.sleep(sleep_time)


if __name__ == "__main__":
    # 스케줄러 스레드 시작
    t = threading.Thread(target=log_rotation_scheduler, daemon=True)
    t.start()

    logger.info("트레이딩 봇 시작 및 로그 로테이션 스케줄러 가동")
    try:
        run()
    except KeyboardInterrupt:
        log_print("\n\n[STOP] 프로그램 종료")
    except Exception as e:
        log_print(f"\n[ERROR] 심각한 오류: {e}")
        import traceback
        traceback.print_exc()
