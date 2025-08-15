# labs/L2_modeling/bt_reg_threshold.py
import pandas as pd, numpy as np

H = 15
FEE = 0.0005
SLIPPAGE = 0.0005
ROUND_TRIP = FEE + SLIPPAGE

VAL = pd.read_csv('predictions_xgb_reg_val.csv', parse_dates=['timestamp']).sort_values('timestamp')
TE  = pd.read_csv('predictions_xgb_reg_te.csv',  parse_dates=['timestamp']).sort_values('timestamp')
DATA= pd.read_csv('dataset_L2_tb.csv', parse_dates=['timestamp']).sort_values('timestamp')[['timestamp','y_ret_H']]

def align(df): 
    return pd.merge_asof(df, DATA, on='timestamp', direction='nearest', tolerance=pd.Timedelta('30s')).dropna()

val = align(VAL); te = align(TE)

def bt_reg(df, tau):
    # 진입 조건: 예측수익 >= 비용+tau
    sig = (df['y_hat'] >= (ROUND_TRIP + tau)).astype(int).to_numpy()
    net = sig * (df['y_ret_H'].to_numpy() - ROUND_TRIP)
    if net.size==0 or sig.sum()==0:
        return dict(tau=tau, trades=int(sig.sum()), cagr=0., sharpe=0., mdd=0., win=0.)
    eq = np.cumprod(1+net); peak=np.maximum.accumulate(eq)
    mdd=float((eq/peak-1).min()); ann=(365*24*60)/max(H,1)
    return dict(tau=tau, trades=int(sig.sum()),
                cagr=float(eq[-1]**(ann/len(eq))-1),
                sharpe=float((net.mean()/(net.std()+1e-12))*np.sqrt(ann)),
                mdd=mdd, win=float((net>0).mean()))

grid = np.round(np.linspace(0.0, 0.0020, 21), 5)   # 0~20bp 추가 마진
val_res = pd.DataFrame([bt_reg(val, t) for t in grid])
val_res_nz = val_res[val_res.trades>0]
if val_res_nz.empty:
    print("[VAL] 모든 임계치에서 거래 0건 → H↑ 또는 데이터/피처 확장 필요"); exit(0)

best = val_res_nz.sort_values('sharpe', ascending=False).iloc[0]
print(f"[VAL][REG] best tau={best.tau:.5f} trades={best.trades} Sharpe={best.sharpe:.2f}")

test = bt_reg(te, float(best.tau))
print(f"[TEST][REG] tau={best.tau:.5f} Trades={test['trades']}  CAGR={test['cagr']:.3f}  Sharpe={test['sharpe']:.2f}  MDD={test['mdd']:.2%}  Hit={test['win']:.2%}")