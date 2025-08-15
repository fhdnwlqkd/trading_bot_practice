import ccxt, pandas as pd, time, logging
import numpy as np
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import execute_values

# -------- 설정 --------
SYMBOL = "XRP/KRW"
TF = "1m"
LOOKBACK_DAYS = 30                 # ✅ 최근 30일
LOOKBACK_MINUTES = LOOKBACK_DAYS * 24 * 60
BATCH = 200                        # Upbit 1m OHLCV 페이지당 최대치(일반적으로 200)
CSV_PATH = "results/xrp_1m.csv"

# -------- 로깅 --------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("ingest.log"), logging.StreamHandler()]
)

def fetch_ohlcv_range_days():
    """Upbit에서 최근 LOOKBACK_DAYS 만큼 1분봉 OHLCV를 끌어와 연속 시계열로 반환."""
    ex = ccxt.upbit()
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = end_ms - LOOKBACK_MINUTES * 60_000

    since = start_ms
    last_ts = None
    all_rows = []

    while since < end_ms and len(all_rows) < LOOKBACK_MINUTES + 500:
        rows = ex.fetch_ohlcv(SYMBOL, timeframe=TF, since=since, limit=BATCH)
        if not rows:
            logging.info("no more rows from exchange; breaking.")
            break

        rows = sorted(rows, key=lambda r: r[0])  # ts 기준 오름차순
        # 새 데이터의 마지막 ts
        new_last = rows[-1][0]

        # 수집 버퍼에 추가
        all_rows.extend(rows)

        # 다음 페이지 기준점: ✅ 마지막 캔들 + 1분 (겹치지도, 건너뛰지도 않게)
        # (중복은 최종 DataFrame에서 drop_duplicates로 정리)
        if last_ts is not None and new_last <= last_ts:
            logging.info("since not moving forward; breaking to avoid loop.")
            break
        last_ts = new_last
        since = last_ts + 60_000  # 1분

        # 레이트리밋 여유
        time.sleep(0.20)

        # 안전 가드: end를 초과하면 중단
        if since > end_ms:
            break

    # DataFrame 구성
    df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","volume"])
    df = df.drop_duplicates("ts").sort_values("ts")
    # ms -> UTC ts
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()

    # 1분 격자에 맞춰 정렬(결측은 NaN으로 보이게)
    df = df.asfreq("1min")
    return df

def quality_check(df: pd.DataFrame):
    """결측과 중복 진단. asfreq 이후라 diff 기반 대신 NaN 카운트를 사용."""
    # asfreq 후라 index 간격은 1분 고정 → 결측은 값 NaN으로 나타남
    # open/close 기준 어느 한 컬럼으로 집계(둘 다 동일하게 NaN 뜸)
    missing_rows = int(df["close"].isna().sum())
    # 중복은 asfreq 후엔 거의 0이지만, 원시 단계에서의 중복은 이미 drop됨
    dups = 0
    logging.info(f"quality: missing_rows={missing_rows}, duplicates={dups}, rows={len(df)}")
    return missing_rows, dups

def save_csv(df: pd.DataFrame):
    import os
    os.makedirs("results", exist_ok=True)
    out = df.copy()
    out["trade_count"] = 0
    out.reset_index().rename(columns={"ts":"timestamp"}).to_csv(CSV_PATH, index=False)
    logging.info(f"saved CSV -> {CSV_PATH}")

def upsert_db(df: pd.DataFrame):
    import os
    DB_NAME = os.getenv("POSTGRES_DB", "market")
    DB_USER = os.getenv("POSTGRES_USER", "postgres")
    DB_PASS = os.getenv("POSTGRES_PASSWORD", "pgpass")
    DB_HOST = os.getenv("PGHOST", "localhost")
    DB_PORT = int(os.getenv("PGPORT", "5432"))

    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()

    # numpy 스칼라 안전 캐스팅
    def cast(x):
        return None if pd.isna(x) else float(x)

    rows = []
    for ts, r in df.iterrows():
        rows.append((
            ts.to_pydatetime(),             # timestamptz
            cast(r.open), cast(r.high), cast(r.low), cast(r.close), cast(r.volume),
            0  # trade_count 미수집 → 0
        ))

    sql = """
      INSERT INTO ohlcv_xrp_krw_1m(ts, open, high, low, close, volume, trade_count)
      VALUES %s
      ON CONFLICT (ts) DO UPDATE SET
        open=EXCLUDED.open,
        high=EXCLUDED.high,
        low=EXCLUDED.low,
        close=EXCLUDED.close,
        volume=EXCLUDED.volume,
        trade_count=COALESCE(ohlcv_xrp_krw_1m.trade_count, 0);
    """
    execute_values(cur, sql, rows, page_size=1000)
    conn.commit()
    cur.close(); conn.close()
    logging.info(f"upserted {len(rows)} rows into Postgres")

if __name__ == "__main__":
    df = fetch_ohlcv_range_days()
    quality_check(df)
    save_csv(df)
    upsert_db(df)
    logging.info("done.")