import pandas as pd, numpy as np, psycopg2

DB_DSN = "dbname=market user=postgres password=pgpass host=localhost port=5432"
LOOKBACK_DAYS = 2   # 연습: 최근 2일
H = 5               # 예측 지평: 5분 뒤 수익률

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
    return df.asfreq('1min')

def make_features(df: pd.DataFrame):
    out = df.copy()
    out['close_ffill'] = out['close'].ffill()
    for h in (1,5,15):
        out[f'ret_{h}'] = out['close'].pct_change(h)
    out['ma_20']  = out['close_ffill'].rolling(20, min_periods=20).mean()
    out['vol_20'] = out['ret_1'].rolling(20, min_periods=20).std()
    out['ret_mean_20'] = out['ret_1'].rolling(20, min_periods=20).mean()
    out['z_ret_20']    = (out['ret_1'] - out['ret_mean_20']) / out['vol_20'].replace({0: np.nan})
    return out

def make_labels(df: pd.DataFrame, H: int):
    out = df.copy()
    out['y_ret_H'] = out['close'].shift(-H) / out['close'] - 1.0  # 회귀 라벨
    return out

if __name__ == "__main__":
    base = load_from_db()
    feats = make_features(base)
    Xy = make_labels(feats, H)
    feature_cols = ['ret_1','ret_5','ret_15','ma_20','vol_20','z_ret_20','close']
    keep = feature_cols + ['y_ret_H']
    Xy = Xy[keep].dropna()
    Xy.to_csv('dataset_1m_h5_reg.csv')
    print("Saved dataset_1m_h5_reg.csv", Xy.shape)
    print(Xy.tail(3))