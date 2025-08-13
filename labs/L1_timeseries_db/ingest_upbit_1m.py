import ccxt, pandas as pd, time, logging
import numpy as np
from datetime import datetime, timezone
import psycopg2
from psycopg2.extras import execute_values

# -------- 설정 --------
SYMBOL = "XRP/KRW"
TF = "1m"
TARGET_MINUTES = 24 * 60    # 최근 24시간
BATCH = 200                  # 페이지당 요청 수
CSV_PATH = "results/xrp_1m.csv"

# -------- 로깅 --------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("ingest.log"), logging.StreamHandler()]
)

def fetch_ohlcv_24h():
    ex = ccxt.upbit()
    end = int(datetime.now(timezone.utc).timestamp() * 1000)
    start = end - TARGET_MINUTES * 60_000
    since = start
    last_ts = None
    all_rows = []
    while since < end and len(all_rows) < TARGET_MINUTES + 10:
        rows = ex.fetch_ohlcv(SYMBOL, timeframe=TF, since=since, limit=BATCH)
        if not rows:
            break
        rows = sorted(rows, key=lambda r: r[0])
        if last_ts is not None and rows[-1][0] <= last_ts:
            break
        all_rows += rows
        last_ts = rows[-1][0]
        since = last_ts + 60_000
        time.sleep(0.15)  # 레이트리밋 여유
    df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","volume"]).drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    df = df.asfreq("1min")  # 1분 격자 (결측은 NaN으로 보이게)
    return df

def quality_check(df: pd.DataFrame):
    missing = int(df.index.to_series().diff().gt(pd.Timedelta(minutes=1)).sum())
    dups = int(df.index.duplicated().sum())
    logging.info(f"quality: missing_gaps={missing}, duplicates={dups}, rows={len(df)}")
    return missing, dups

def save_csv(df: pd.DataFrame):
    import os
    os.makedirs("results", exist_ok=True)
    out = df.copy()
    out["trade_count"] = 0
    out.reset_index().rename(columns={"ts":"timestamp"}).to_csv(CSV_PATH, index=False)
    logging.info(f"saved CSV -> {CSV_PATH}")

def upsert_db(df: pd.DataFrame):
    import os, math
    import psycopg2
    from psycopg2.extras import execute_values

    # (1) 접속 정보
    DB_NAME = os.getenv("POSTGRES_DB", "market")
    DB_USER = os.getenv("POSTGRES_USER", "postgres")
    DB_PASS = os.getenv("POSTGRES_PASSWORD", "pgpass")
    DB_HOST = os.getenv("PGHOST", "localhost")
    DB_PORT = int(os.getenv("PGPORT", "5432"))

    conn = psycopg2.connect(
        dbname=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST, port=DB_PORT
    )
    cur = conn.cursor()

    # (2) numpy 스칼라 → 파이썬 내장형으로 캐스팅 (NaN은 None으로)
    def f(x):
        return None if pd.isna(x) else float(x)

    rows = []
    for ts, r in df.iterrows():
        rows.append((
            ts.to_pydatetime(),  # tz-aware OK (timestamptz)
            f(r.open), f(r.high), f(r.low), f(r.close), f(r.volume),
            0  # trade_count 미수집 → 0
        ))

    sql = """
      INSERT INTO ohlcv_xrp_krw_1m(ts, open, high, low, close, volume, trade_count)
      VALUES %s
      ON CONFLICT (ts) DO UPDATE SET
        open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low,
        close=EXCLUDED.close, volume=EXCLUDED.volume,
        trade_count=COALESCE(ohlcv_xrp_krw_1m.trade_count, 0);
    """
    execute_values(cur, sql, rows, page_size=1000)
    conn.commit()
    cur.close(); conn.close()
    logging.info(f"upserted {len(rows)} rows into Postgres")

if __name__ == "__main__":
    df = fetch_ohlcv_24h()
    quality_check(df)
    save_csv(df)
    upsert_db(df)
    logging.info("done.")