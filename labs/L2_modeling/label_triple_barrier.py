import pandas as pd, numpy as np

H = 5          # 시간 배리어(5분 뒤까지 관찰)
PT_K = 1.5     # 이익배리어(atr 비율의 배수)
SL_K = 1.0     # 손절배리어(atr 비율의 배수)

df = pd.read_csv('dataset_L2_full.csv', parse_dates=['timestamp']).set_index('timestamp').sort_index()

# ATR 비율(가격 대비)
atr_pct = (df['atr_14'] / df['close']).fillna(method='ffill')
upper = df['close'] * (1 + PT_K * atr_pct)
lower = df['close'] * (1 - SL_K * atr_pct)

labels = np.zeros(len(df), dtype=int)   # -1,0,1 중 0 기본
hit_time = np.full(len(df), np.nan)

high = df['high'].values
low  = df['low'].values
close= df['close'].values
up   = upper.values
lo   = lower.values

n = len(df)
for i in range(n - H):
    hi = high[i+1:i+H+1]
    lw = low[i+1:i+H+1]
    # 각 배리어 충족 여부의 최초 시점
    up_hit_idx = np.where(hi >= up[i])[0]
    lo_hit_idx = np.where(lw <= lo[i])[0]

    t_up = i + 1 + up_hit_idx[0] if len(up_hit_idx) else np.inf
    t_lo = i + 1 + lo_hit_idx[0] if len(lo_hit_idx) else np.inf
    t_exp= i + H

    # 먼저 맞은 이벤트 결정
    t_first = min(t_up, t_lo, t_exp)
    hit_time[i] = t_first

    if t_first == t_up:
        labels[i] = 1
    elif t_first == t_lo:
        labels[i] = -1
    else:
        # 만기까지 배리어 미도달 → 만기 수익률 부호 사용
        y = close[i+H]/close[i] - 1.0
        labels[i] = 1 if y > 0 else (-1 if y < 0 else 0)

df['tb_label_3'] = labels  # -1,0,1
# 이진 분류용(중립 0은 제외할 예정)
df['tb_label_bin'] = df['tb_label_3'].map({-1:0, 1:1})

# 회귀/백테스트용 H분 뒤 수익률도 같이 저장
df['y_ret_H'] = df['close'].shift(-H)/df['close'] - 1.0

out = df.dropna(subset=['tb_label_bin','y_ret_H']).copy()
out.to_csv('dataset_L2_tb.csv')
print("Saved dataset_L2_tb.csv", out.shape, "  labels(3) counts:", out['tb_label_3'].value_counts().to_dict())