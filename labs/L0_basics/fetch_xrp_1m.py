import ccxt, pandas as pd

# upbit XRP 1분 봉 데이터를 가져와서 CSV로 저장합니다.
# timestamp, open, high, low, close, volume 순서로 저장됩니다.
# open: 첫 거래 가격, high: 최고가, low: 최저가, close: 마지막 거래 가격, volume: 거래량
# 결과는 'results/xrp_1m.csv'에 저장됩니다.
ex = ccxt.upbit()
symbol, tf, limit = 'XRP/KRW', '1m', 1000
ohlcv = ex.fetch_ohlcv(symbol, timeframe=tf, limit=limit)  # [ts, o, h, l, c, v]

df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('Asia/Seoul')
df = df.sort_values('timestamp').reset_index(drop=True)
df.to_csv('results/xrp_1m.csv', index=False)
print('saved xrp_1m.csv'); print(df.head(3)); print(df.tail(3))