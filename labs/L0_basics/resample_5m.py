import pandas as pd
df = pd.read_csv('results/xrp_1m.csv', parse_dates=['timestamp']).set_index('timestamp').sort_index()
df = df.asfreq('1T')  # 1분 격자(결측을 NaN으로 노출)
ohlcv_5m = df.resample('5T').agg({
    'open':'first','high':'max','low':'min','close':'last','volume':'sum'
}).dropna(subset=['open','high','low','close'])
# 무결성 체크
assert (ohlcv_5m['high'] >= ohlcv_5m[['open','close','low']].max(axis=1)).all()
assert (ohlcv_5m['low']  <= ohlcv_5m[['open','close','high']].min(axis=1)).all()
print(ohlcv_5m.head(10))
ohlcv_5m.to_csv('results/xrp_1m.csv'); print('saved xrp_1m.csv')