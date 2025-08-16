#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Upbit OHLCV 수집 (XRP/KRW 1m, 최근 N일) → CSV 저장(+품질 로그)
- --source ccxt        : OHLCV 6열
- --source upbit-native: (미구현, 확장 자리만)
- --with-ticks         : (미사용; 확장 자리만)
"""
import os, time, argparse, logging, requests  # requests는 추후 확장 대비로만 사용
from typing import List
import pandas as pd
from datetime import datetime, timezone
import ccxt

# ----------------------
# 로깅
# ----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("ingest.log")]
)

# ----------------------
# 품질/저장 유틸
# ----------------------
def quality_check(df: pd.DataFrame, label: str = "candles") -> None:
    """데이터프레임 품질 로그(행수/범위/중복/결측)"""
    rows = len(df)
    if rows == 0:
        logging.warning(f"[QC][{label}] No data (0 rows).")
        return
    span = (df.index.min(), df.index.max())
    dups = int(df.index.duplicated().sum())
    miss = int(df.isna().any(axis=1).sum())
    logging.info(f"[QC][{label}] rows={rows}  span={span[0]}→{span[1]}  dup_idx={dups}  missing_rows={miss}")

def save_csv(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out = df.reset_index().rename(columns={"ts": "timestamp"})
    out.to_csv(out_path, index=False)
    logging.info(f"Saved CSV -> {out_path}")
    
    
#-----------------------
# 결측 감지 및 다시 채우기
#-----------------------
# ==== (1) 유틸: 결측 구간 찾기 ====
def find_missing_segments(df: pd.DataFrame, col: str = "close") -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """close가 NaN인 연속 구간을 [(start_ts, end_ts), ...]로 반환 (모두 UTC, 1min 격자 가정)"""
    mask = df[col].isna()
    if mask.sum() == 0:
        return []
    idx = df.index[mask]
    # 연속 구간으로 묶기
    groups = (idx.to_series().diff() != pd.Timedelta("1min")).cumsum()
    segs = []
    for _, g in idx.to_series().groupby(groups):
        segs.append((g.iloc[0], g.iloc[-1]))
    return segs

# ==== (1) 유틸: 범위 재수집(CCXT) ====
def fetch_range_ccxt(symbol: str, timeframe: str,
                     start_ts: pd.Timestamp, end_ts: pd.Timestamp,
                     batch: int = 200, sleep_s: float = 0.20) -> pd.DataFrame:
    """start~end 범위를 CCXT로 좁게 재수집 → 1분 격자 리턴(UTC index)"""
    ex = ccxt.upbit()
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms   = int(end_ts.timestamp()   * 1000)
    all_rows = []
    last_ts = None
    since = start_ms
    while since <= end_ms + 60_000:  # 끝 분 포함 여유
        rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=batch)
        if not rows:
            break
        rows = sorted(rows, key=lambda r: r[0])
        new_last = rows[-1][0]
        if last_ts is not None and new_last <= last_ts:
            break
        all_rows.extend(rows)
        last_ts = new_last
        if last_ts >= end_ms:
            break
        since = last_ts + 60_000
        time.sleep(sleep_s)

    if not all_rows:
        return pd.DataFrame(columns=["open","high","low","close","volume"],
                            index=pd.DatetimeIndex([], tz="UTC", name="ts"))

    df_new = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","volume"]).drop_duplicates("ts")
    df_new["ts"] = pd.to_datetime(df_new["ts"], unit="ms", utc=True)
    df_new = df_new.set_index("ts").sort_index()

    # 요청범위를 1분 격자로 맞춘 뷰만 반환
    grid = pd.date_range(start=start_ts.floor("min"), end=end_ts.floor("min"), freq="1min", tz="UTC")
    df_new = df_new.reindex(grid)  # 결측은 NaN (원자료 보존)
    return df_new

# ==== (1) 백필 본체 ====
def backfill_missing_ccxt(df: pd.DataFrame, symbol: str, timeframe: str,
                          batch: int = 200, sleep_s: float = 0.20,
                          max_passes: int = 2, pad_minutes: int = 3) -> pd.DataFrame:
    """
    결측 구간만 CCXT로 재수집해 덮어씀. max_passes 회 반복하며 줄어들면 계속.
    pad_minutes: 각 구간 앞뒤 버퍼(분)
    """
    base_cols = ["open","high","low","close","volume"]
    df_out = df.copy()
    for p in range(1, max_passes+1):
        segs = find_missing_segments(df_out, col="close")
        if not segs:
            logging.info(f"[backfill] no missing segments; stop (pass #{p})")
            break
        logging.info(f"[backfill] pass #{p} segments={len(segs)}")
        before = int(df_out["close"].isna().sum())

        for (s, e) in segs:
            s_pad = (s - pd.Timedelta(minutes=pad_minutes)).floor("min")
            e_pad = (e + pd.Timedelta(minutes=pad_minutes)).ceil("min")
            try:
                patch = fetch_range_ccxt(symbol, timeframe, s_pad, e_pad, batch=batch, sleep_s=sleep_s)
            except Exception as ex:
                logging.warning(f"[backfill] fetch error {s}~{e}: {ex}")
                continue
            # 해당 구간 교집합에 대해 덮어쓰기
            idx = patch.index
            if len(idx) == 0:
                continue
            df_out.loc[idx, base_cols] = patch[base_cols].values

        after = int(df_out["close"].isna().sum())
        logging.info(f"[backfill] pass #{p} missing_rows {before} -> {after}")
        if after >= before:
            logging.info("[backfill] no improvement; stop")
            break
    return df_out

# ----------------------
# CCXT 수집 (OHLCV만)
# ----------------------
def fetch_ohlcv_ccxt(symbol: str, timeframe: str, minutes: int, batch: int = 200, sleep_s: float = 0.20) -> pd.DataFrame:
    ex = ccxt.upbit()
    end_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since = end_ms - minutes * 60_000
    all_rows: List[List[float]] = []
    last_ts = None

    while since < end_ms:
        try:
            rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=batch)
            if not rows:
                logging.info("No more data from exchange; break.")
                break
        except ccxt.NetworkError as e:
            logging.warning(f"NetworkError: {e}; retry in 1s")
            time.sleep(1.0); continue
        except ccxt.ExchangeError as e:
            logging.error(f"ExchangeError: {e}; stop"); break

        rows = sorted(rows, key=lambda r: r[0])
        new_last_ts = rows[-1][0]
        if last_ts is not None and new_last_ts <= last_ts:
            logging.warning("Timestamp not advancing; break to avoid loop.")
            break

        all_rows.extend(rows)
        last_ts = new_last_ts
        since = last_ts + 60_000  # 다음 1분
        time.sleep(sleep_s)

    df = pd.DataFrame(all_rows, columns=["ts","open","high","low","close","volume"])
    df = df.drop_duplicates("ts").copy()
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()

    # 수집 범위를 1분 격자로 리인덱스(결측은 NaN 유지)
    if not df.empty:
        start_dt, end_dt = df.index.min().floor("min"), df.index.max().floor("min")
        full = pd.date_range(start=start_dt, end=end_dt, freq="1min", tz="UTC")
        df = df.reindex(full)
    return df

# ----------------------
# (확장 자리) 업비트 네이티브 — 아직 미구현
# ----------------------
def fetch_ohlcv_upbit_native(symbol: str, minutes: int, batch: int = 200, sleep_s: float = 0.12) -> pd.DataFrame:
    logging.warning("[upbit-native] not implemented yet. Returning empty DataFrame.")
    # 빈 DataFrame을 반환해도 run()에서 안전 종료되도록 설계
    return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"), columns=["open","high","low","close","volume"])

# ----------------------
# 메인 파이프라인
# ----------------------
def run(source: str, symbol: str, timeframe: str, days: int, out_csv: str, batch: int,
        with_ticks: bool, backfill: bool):
    minutes = days * 24 * 60
    logging.info(f"Start ingest: src={source}, symbol={symbol}, tf={timeframe}, days={days}, minutes={minutes}")

    if source == "upbit-native":
        df = fetch_ohlcv_upbit_native(symbol, minutes, batch=batch)  # 현재 미구현 placeholder
    else:
        df = fetch_ohlcv_ccxt(symbol, timeframe, minutes, batch=batch)

    quality_check(df, label=source)
    if df.empty:
        logging.error("No candle data fetched. Aborting.")
        return

    # 🔁 선택: 결측 구간 백필 (CCXT 경로에서만 동작 권장)
    if backfill and source == "ccxt":
        df = backfill_missing_ccxt(df, symbol, timeframe, batch=batch)
        quality_check(df, label=f"{source}+backfilled")

    save_csv(df, out_csv)
    logging.info("Done.")

# ----------------------
# CLI
# ----------------------
def main():
    ap = argparse.ArgumentParser(description="Fetch Upbit OHLCV to CSV")
    ap.add_argument("--source", choices=["ccxt","upbit-native"], default="ccxt", help="Data source (ccxt recommended for now)")
    ap.add_argument("--symbol", default="XRP/KRW")
    ap.add_argument("--timeframe", default="1m")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--csv", default="/Users/m2nsteel/trading_bot_practice/labs/L2_modeling/data/raw/xrp_1m_30d.csv")
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--with-ticks", action="store_true", help="(reserved) include trade counts from ticks")
    ap.add_argument("--backfill", action="store_true", help="결측 구간을 CCXT로 재수집해 보정")
    args = ap.parse_args()
    run(args.source, args.symbol, args.timeframe, args.days, args.csv, args.batch, args.with_ticks, args.backfill)
if __name__ == "__main__":
    main()