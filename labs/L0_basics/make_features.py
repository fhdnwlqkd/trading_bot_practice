# make_features_1m.py
import pandas as pd, numpy as np

# upbit XRP 1분 봉 데이터를 가져와서 특성(feature)을 생성합니다.
# timestamp, open, high, low, close, volume 순서로 저장됩니다.
# open: 첫 거래 가격, high: 최고가, low: 최저가, close: 마지막 거래 가격, volume: 거래량
# 결과는 'results/xrp_1m_features.csv'에 저장됩니다.
# 기존의 1분 봉 데이터를 기반으로 모멘텀, 추세, 변동성 등의 특성을 생성합니다.
# 기존의 데이터를 정상성을 갖춘 시계열로 변환하고, 다양한 지표를 추가합니다.

# 1) 로드 & 시간 정렬(1분 격자 권장)
df = pd.read_csv('results/xrp_1m.csv', parse_dates=['timestamp']).set_index('timestamp').sort_index()
df = df.asfreq('1T')  # 누락 분봉은 NaN으로 노출(결측 확인용)
df['close_ffill'] = df['close'].ffill()  # 지표 연속성용 (OHLC 본문은 채우지 않음)

# 2) 모멘텀: n개 '봉' 기준 수익률(1분봉이면 n분)
for h in (1, 5, 15):
    df[f'ret_{h}'] = df['close'].pct_change(h)

# 3) 추세/변동성: 20개 창(=20분)
df['ma_20']  = df['close_ffill'].rolling(20, min_periods=20).mean()
df['vol_20'] = df['ret_1'].rolling(20, min_periods=20).std()  # 수익률 기반 변동성(단위 일치 👍)

# 4) 수익률-기반 z-score (단위 맞춤)
df['ret_mean_20'] = df['ret_1'].rolling(20, min_periods=20).mean()
df['vol_20_safe'] = df['vol_20'].replace({0: np.nan})  # 0으로 나눔 방지
df['z_ret_20']    = (df['ret_1'] - df['ret_mean_20']) / df['vol_20_safe']

# (선택) 보조 컬럼 정리
df = df.drop(columns=['vol_20_safe'])

# 5) 저장 & 확인
out_cols = ['close', 'ret_1', 'ret_5', 'ret_15', 'ma_20', 'vol_20', 'z_ret_20']
df.to_csv('results/xrp_1m_features.csv')
print(df[out_cols].head(25))