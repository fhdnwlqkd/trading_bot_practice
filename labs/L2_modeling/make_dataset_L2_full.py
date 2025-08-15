import pandas as pd, numpy as np, psycopg2

DB_DSN = "dbname=market user=postgres password=pgpass host=localhost port=5432"
LOOKBACK_DAYS = 3    # 연습: 최근 3일 (가능하면 더 키우면 좋아요)
AS_FREQ = "1min"

def load_from_db():
    conn = psycopg2.connect(DB_DSN)
    q = f"""
    SELECT ts AS timestamp, open, high, low, close, volume
    FROM ohlcv_xrp_krw_1m
    WHERE ts >= now() - interval '{LOOKBACK_DAYS} days'
    ORDER BY ts;
    """
    df = pd.read_sql(q, conn, parse_dates=['timestamp']).set_index('timestamp').sort_index()
    conn.close()
    return df.asfreq(AS_FREQ)

def make_features(df: pd.DataFrame):
    out = df.copy()
    out['close_ffill'] = out['close'].ffill()

    # 리턴(모멘텀)
    for h in (1,5,15):
        out[f'ret_{h}'] = out['close'].pct_change(h)

    # MA/변동성/표준화
    out['ma_20']  = out['close_ffill'].rolling(20, min_periods=20).mean()
    out['vol_20'] = out['ret_1'].rolling(20, min_periods=20).std()
    out['ret_mean_20'] = out['ret_1'].rolling(20, min_periods=20).mean()
    out['z_ret_20']    = (out['ret_1'] - out['ret_mean_20']) / out['vol_20'].replace({0: np.nan})

    # EMA & MACD
    out['ema_12'] = out['close_ffill'].ewm(span=12, adjust=False).mean()
    out['ema_26'] = out['close_ffill'].ewm(span=26, adjust=False).mean()
    out['macd']   = out['ema_12'] - out['ema_26']

    # RSI(14)
    delta = out['close_ffill'].diff()
    up = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean().replace(0, 1e-12)
    rs = up / down
    out['rsi_14'] = 100 - (100 / (1 + rs))

    # 볼린저 밴드 폭(20)
    std20 = out['close_ffill'].rolling(20, min_periods=20).std()
    out['bb_width_20'] = (2 * std20) / out['ma_20']

    # ATR(14)
    prev_close = out['close_ffill'].shift(1)
    tr = pd.concat([
        (out['high'] - out['low']).abs(),
        (out['high'] - prev_close).abs(),
        (out['low']  - prev_close).abs()
    ], axis=1).max(axis=1)
    out['atr_14'] = tr.rolling(14, min_periods=14).mean()

    # ROC(10)
    out['roc_10'] = out['close_ffill'].pct_change(10)

    return out

if __name__ == "__main__":
    base = load_from_db()
    feats = make_features(base)
    keep = [
        'open','high','low','close','volume',
        'ret_1','ret_5','ret_15','ma_20','vol_20','z_ret_20',
        'ema_12','ema_26','macd','rsi_14','bb_width_20','atr_14','roc_10'
    ]
    feats[keep].dropna().to_csv('dataset_L2_full.csv')
    print("Saved dataset_L2_full.csv", feats[keep].dropna().shape)