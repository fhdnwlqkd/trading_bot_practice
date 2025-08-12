import pandas as pd

# upbit XRP 1분 봉 데이터를 가져와서 5분 봉으로 리샘플링하여 CSV로 저장합니다.
# timestamp, open, high, low, close, volume 순서로 저장됩니다.
# open: 첫 거래 가격, high: 최고가, low: 최저가, close: 마지막 거래 가격, volume: 거래량
# 결과는 'results/xrp_5m.csv'에 저장됩니다.
# resample을 하는 이유는 1분 봉 데이터는 노이즈가 많을 수 있고, 모델 입력 간격을 맞추기 위함입니다.
df = pd.read_csv('results/xrp_1m.csv', parse_dates=['timestamp']).set_index('timestamp').sort_index()
df = df.asfreq('1T')  # 1분 격자(결측을 NaN으로 노출)
ohlcv_5m = df.resample('5T').agg({
    'open':'first','high':'max','low':'min','close':'last','volume':'sum'
}).dropna(subset=['open','high','low','close'])
# 무결성 체크
assert (ohlcv_5m['high'] >= ohlcv_5m[['open','close','low']].max(axis=1)).all()
assert (ohlcv_5m['low']  <= ohlcv_5m[['open','close','high']].min(axis=1)).all()
print(ohlcv_5m.head(10))
ohlcv_5m.to_csv('results/xrp_5m.csv'); print('saved xrp_5m.csv')