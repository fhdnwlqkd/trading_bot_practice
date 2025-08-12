import pandas as pd, numpy as np
df = pd.read_csv('results/xrp_1m.csv', parse_dates=['timestamp']).set_index('timestamp').sort_index()

# 결측 탐지(선택)
gaps = df.index.to_series().diff().gt(pd.Timedelta('1T')).sum()
print('missing gaps:', gaps)

# 지표 연속성용으로만 ffill 사용(가격 자체는 채우지 않음)
df['close_ffill'] = df['close'].ffill()

# 수익률
df['ret_1']  = df['close'].pct_change(1)
df['ret_5']  = df['close'].pct_change(5)
df['ret_15'] = df['close'].pct_change(15)

# 롤링 통계(고정 길이, trailing window)
df['ma_20']  = df['close_ffill'].rolling(20, min_periods=20).mean()
df['std_20'] = df['close_ffill'].rolling(20, min_periods=20).std()

print(df[['close','ret_1','ret_5','ret_15','ma_20','std_20']].head(25))
df.to_csv('results/xrp_1m_features.csv'); print('saved xrp_1m_features.csv')