# labs/L2_modeling/wf_reg_tau.py
# 워크포워드(증분 재학습): 각 단계에서 (train→val에서 τ 선정) → 바로 다음 test 구간에 적용
import pandas as pd, numpy as np, xgboost as xgb
from sklearn.metrics import mean_absolute_error

DATA = 'dataset_L2_tb.csv'
FEATS = ['ret_1','ret_5','ret_15','ma_20','vol_20','z_ret_20','ema_12','ema_26','macd','rsi_14','bb_width_20','atr_14','roc_10','close']
TARGET = 'y_ret_H'

# ---- 하이퍼 ----
H = 15                 # 라벨 생성 스크립트와 맞춰주세요(권장: 10~15)
FEE = 0.0005
SLIPPAGE = 0.0005
ROUND_TRIP = FEE + SLIPPAGE
VAL_FRAC = 0.10        # 각 step에서 train 내부 마지막 10%를 검증으로 사용
TEST_WINDOW = 24*60    # 1일=1440 포인트를 한 step의 테스트 구간으로
TAU_GRID = np.round(np.linspace(0.0, 0.0020, 21), 5)  # 0~20bp 추가 마진
MIN_TRADES_VAL = 5     # 검증에서 최소 거래 수(과최적화 방지)
PARAMS = dict(objective='reg:squarederror', eval_metric='rmse',
              eta=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8,
              lambda_=1.0, seed=42, verbosity=0)
NUM_BOOST = 1500
ES_ROUNDS = 100

def predict_best(booster, dmat):
    bi = getattr(booster, 'best_iteration', None)
    return booster.predict(dmat, iteration_range=(0, bi+1)) if bi is not None else booster.predict(dmat)

def pnl_metrics(y_ret, signal):
    net = signal * (y_ret - ROUND_TRIP)
    if net.size == 0 or signal.sum() == 0:
        return dict(trades=int(signal.sum()), cagr=0., sharpe=0., mdd=0., hit=0.)
    eq = np.cumprod(1+net)
    peak = np.maximum.accumulate(eq); mdd = float((eq/peak - 1).min())
    ann = (365*24*60)/max(H,1)
    cagr = float(eq[-1]**(ann/len(eq)) - 1)
    sharpe = float((net.mean()/ (net.std()+1e-12))*np.sqrt(ann))
    hit = float((net>0).mean())
    return dict(trades=int(signal.sum()), cagr=cagr, sharpe=sharpe, mdd=mdd, hit=hit)

def choose_tau_on_val(y_hat_val, y_val):
    rows = []
    for tau in TAU_GRID:
        sig = (y_hat_val >= (ROUND_TRIP + tau)).astype(int)
        m = pnl_metrics(y_val.values, sig.values)
        rows.append(dict(tau=float(tau), **m))
    df = pd.DataFrame(rows)
    df = df[df.trades >= MIN_TRADES_VAL]
    if df.empty:
        return float(TAU_GRID[0]), None  # 거래 너무 적으면 가장 낮은 τ
    best = df.sort_values('sharpe', ascending=False).iloc[0]
    return float(best.tau), df

# ---- 데이터 로드 ----
df = pd.read_csv(DATA, parse_dates=['timestamp']).set_index('timestamp').sort_index()
df = df.dropna(subset=[TARGET])
X, y = df[FEATS], df[TARGET]
N = len(df)
print(f"[INFO] total={N}, start={df.index[0]}, end={df.index[-1]}")

# ---- 워크포워드 루프 ----
results = []
start_train_min = max(20_000, 7*24*60)  # 최소 학습 길이: 7일(또는 데이터 상황에 맞게 14~30일 권장)
i = start_train_min
step = TEST_WINDOW

while i + step <= N:
    X_train, y_train = X.iloc[:i], y.iloc[:i]
    X_test,  y_test  = X.iloc[i:i+step], y.iloc[i:i+step]

    # train → val 분리(+ purge H 반영)
    val_size = max(int(len(X_train)*VAL_FRAC), 200)
    core_end = max(len(X_train) - val_size - H, 1)
    X_core, y_core = X_train.iloc[:core_end], y_train.iloc[:core_end]
    X_val,  y_val  = X_train.iloc[-val_size:], y_train.iloc[-val_size:]

    dcore = xgb.DMatrix(X_core, label=y_core, feature_names=FEATS)
    dval  = xgb.DMatrix(X_val,  label=y_val,  feature_names=FEATS)
    dtest = xgb.DMatrix(X_test,              feature_names=FEATS)

    booster = xgb.train(PARAMS, dcore, NUM_BOOST, evals=[(dval, 'valid')],
                        early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
    y_hat_val = pd.Series(predict_best(booster, dval), index=X_val.index, name='y_hat_val')
    y_hat_te  = pd.Series(predict_best(booster, dtest), index=X_test.index, name='y_hat_te')

    tau, val_table = choose_tau_on_val(y_hat_val, y_val)
    sig_te = (y_hat_te >= (ROUND_TRIP + tau)).astype(int)

    m_te = pnl_metrics(y_test.values, sig_te.values)
    m_te.update(dict(tau=tau, start=str(X_test.index[0]), end=str(X_test.index[-1])))
    results.append(m_te)

    print(f"[WF] {X_test.index[0]}→{X_test.index[-1]}  tau={tau:.5f}  "
          f"trades={m_te['trades']}  Sharpe={m_te['sharpe']:.2f}  CAGR={m_te['cagr']:.3f}")

    i += step  # 다음 윈도우로 이동

wf = pd.DataFrame(results)
if not wf.empty:
    agg = dict(
        windows=len(wf),
        trades=int(wf['trades'].sum()),
        sharpe=float((wf['sharpe']*wf['trades']).sum()/max(wf['trades'].sum(),1)),  # 거래수 가중 평균
        cagr=float(np.prod(1+wf['cagr'])**(1/len(wf)) - 1) if (wf['cagr']> -1).all() else -1.0,
        mdd=float(wf['mdd'].min()),
        hit=float((wf['hit']*wf['trades']).sum()/max(wf['trades'].sum(),1))
    )
    print("\n[WF][AGG] ", agg)
    wf.to_csv("wf_reg_tau_results.csv", index=False)
    print("Saved: wf_reg_tau_results.csv")
else:
    print("[WF] no windows evaluated — increase data or lower TEST_WINDOW.")