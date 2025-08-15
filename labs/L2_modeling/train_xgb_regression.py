import pandas as pd, numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

DATA = 'dataset_1m_h5_reg.csv'
FEATS = ['ret_1','ret_5','ret_15','ma_20','vol_20','z_ret_20','close']
TARGET = 'y_ret_H'
H = 5  # 라벨 지평(누설 방지 purge 크기)

def rmse(y_true, y_pred):
    return float(root_mean_squared_error(y_true, y_pred))

# 0) 데이터 로드
df = pd.read_csv(DATA, parse_dates=['timestamp']).set_index('timestamp').sort_index()

# 1) 시간순 홀드아웃(80/20)
split = int(len(df)*0.8)
train, test = df.iloc[:split], df.iloc[split:]
Xtr, ytr = train[FEATS], train[TARGET]
Xte, yte = test[FEATS],  test[TARGET]

# 2) 베이스 모델
base_params = dict(
    n_estimators=1000,            # 조기종료 전제로 넉넉히
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=0
)

model = XGBRegressor(**base_params)

# 3) Train 내부 교차검증(TimeSeriesSplit) + purge(H)
tscv = TimeSeriesSplit(n_splits=5)
cv_r2 = []
for fold, (tr_idx, va_idx) in enumerate(tscv.split(Xtr), 1):
    # purge: train의 가장 최근 H개는 제거(경계 누설 방지)
    if len(tr_idx) > H:
        tr_idx = tr_idx[:-H]
    X_tr, y_tr = Xtr.iloc[tr_idx], ytr.iloc[tr_idx]
    X_va, y_va = Xtr.iloc[va_idx], ytr.iloc[va_idx]

    m = XGBRegressor(**base_params)
    # fold마다 조기종료(검증셋: 해당 fold의 val)
    m.fit(X_tr, y_tr,
          eval_set=[(X_va, y_va)],
          verbose=False)
    pred_va = m.predict(X_va)
    cv_r2.append(r2_score(y_va, pred_va))

print(f"[CV] R2(mean±std) = {np.mean(cv_r2):.3f} ± {np.std(cv_r2):.3f}")

# 4) 최종 학습: Train의 마지막 10%를 검증으로 분리 + purge(H)
val_size = max(int(len(Xtr)*0.10), 100) if len(Xtr) > 100 else max(int(len(Xtr)*0.1), 10)
core_end = max(len(Xtr) - val_size - H, 1)  # purge 반영
X_core, y_core = Xtr.iloc[:core_end], ytr.iloc[:core_end]
X_val,  y_val  = Xtr.iloc[-val_size:], ytr.iloc[-val_size:]

model.fit(X_core, y_core,
          eval_set=[(X_val, y_val)],
          verbose=False)

# 5) 테스트 평가
pred = model.predict(Xte)
r2   = r2_score(yte, pred)
mae  = mean_absolute_error(yte, pred)
rmse_val = rmse(yte, pred)
print(f"[TEST] R2={r2:.3f}  MAE={mae:.6f}  RMSE={rmse_val:.6f}  N={len(yte)}")

# 6) 중요도/예측 저장
imp = pd.Series(model.feature_importances_, index=FEATS).sort_values(ascending=False)
imp.to_csv("feature_importance_xgb_reg.csv")

out = pd.DataFrame({
    'timestamp': Xte.index.astype(str),
    'y_true': yte.values,
    'y_pred': pred
})
out.to_csv("predictions_xgb_reg.csv", index=False)

print("\nTop importances:\n", imp)
print("\nSaved: feature_importance_xgb_reg.csv, predictions_xgb_reg.csv")