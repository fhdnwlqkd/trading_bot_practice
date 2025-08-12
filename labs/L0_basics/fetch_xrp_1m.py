import ccxt, pandas as pd

ex = ccxt.upbit()
symbol, tf, limit = 'XRP/KRW', '1m', 1000
ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)  # [ts, o, h, l, c, v]

df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Seoul')
df = df.sort_values('timestamp').reset_index(drop=True)
df.to_csv('results/xrp_1m.csv', index=False)
print('saved xrp_1m.csv'); print(df.head(3)); print(df.tail(3))