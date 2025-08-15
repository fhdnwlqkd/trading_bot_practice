# labs/L2_modeling/train_baselines_tb.py
import pandas as pd, numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ----------------------------
# Config
# ----------------------------
DATA = 'dataset_L2_tb.csv'
FEATS = [
    'ret_1','ret_5','ret_15','ma_20','vol_20','z_ret_20',
    'ema_12','ema_26','macd','rsi_14','bb_width_20','atr_14','roc_10','close'
]
TARGET = 'tb_label_bin'
H = 5                 # purge 크기 = 라벨 지평
VAL_FRAC = 0.10       # 최종 학습 시 train의 마지막 10%를 검증으로 사용
NUM_BOOST = 1000
ES_ROUNDS = 50
RANDOM_SEED = 42

# XGBoost params (원시 API)
xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.05,               # learning_rate
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1.0,             # L2
    "seed": RANDOM_SEED,
    "verbosity": 0
}

# ----------------------------
# Utils
# ----------------------------
def compute_spw(y: pd.Series) -> float:
    """scale_pos_weight = negative / positive (최소 1.0)."""
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos <= 0:  # 안전 가드
        return 1.0
    return max(1.0, neg / pos)

def predict_best(booster: xgb.Booster, dmat: xgb.DMatrix) -> np.ndarray:
    """버전 독립 안전 예측: iteration_range 우선, 미지원이면 ntree_limit fallback."""
    bi = getattr(booster, "best_iteration", None)
    try:
        # XGBoost 2.x/3.x 권장
        return booster.predict(dmat, iteration_range=(0, bi + 1)) if bi is not None else booster.predict(dmat)
    except TypeError:
        # 구버전 fallback
        bntl = getattr(booster, "best_ntree_limit", 0) or (bi + 1 if bi is not None else 0)
        return booster.predict(dmat, ntree_limit=bntl) if bntl else booster.predict(dmat)

def purged_cv_auc_xgb(X: pd.DataFrame, y: pd.Series, n_splits=5, purge=H):
    """TimeSeriesSplit + purge(H)로 XGBoost AUC CV. 각 fold에서 scale_pos_weight 동적 적용."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), 1):
        if len(tr_idx) > purge:
            tr_idx = tr_idx[:-purge]  # 경계 누설 방지

        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        spw = compute_spw(y_tr)
        params = dict(xgb_params, scale_pos_weight=spw)

        dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=X.columns.tolist())
        dva = xgb.DMatrix(X_va, label=y_va, feature_names=X.columns.tolist())

        booster = xgb.train(
            params=params,
            dtrain=dtr,
            num_boost_round=NUM_BOOST,
            evals=[(dva, "valid")],
            early_stopping_rounds=ES_ROUNDS,
            verbose_eval=False
        )
        proba = predict_best(booster, dva)
        aucs.append(roc_auc_score(y_va, proba))
    return float(np.mean(aucs)), float(np.std(aucs))

def purged_cv_auc_rf(X: pd.DataFrame, y: pd.Series, n_splits=5, purge=H):
    """TimeSeriesSplit + purge(H)로 RandomForest AUC CV."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    aucs = []
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=8, min_samples_leaf=5,
        max_features='sqrt', random_state=RANDOM_SEED, n_jobs=-1
    )
    for fold,(tr_idx, va_idx) in enumerate(tscv.split(X),1):
        if len(tr_idx) > purge:
            tr_idx = tr_idx[:-purge]
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]
        rf.fit(X_tr, y_tr)
        proba = rf.predict_proba(X_va)[:,1]
        aucs.append(roc_auc_score(y_va, proba))
    return float(np.mean(aucs)), float(np.std(aucs))

def save_feature_importance(booster: xgb.Booster, feature_names, path_csv="feature_importance_xgb.csv"):
    """gain 기반 중요도 저장."""
    score = booster.get_score(importance_type='gain')
    # booster는 f0,f1,... 키로 줄 수 있음 → feature_names와 매핑
    # DMatrix에 feature_names를 지정했으므로 f0 순서 = feature_names 순서
    mapping = {}
    for i, name in enumerate(feature_names):
        mapping[f"f{i}"] = name
    series = pd.Series({mapping.get(k, k): v for k, v in score.items()})
    series = series.reindex(feature_names).fillna(0).sort_values(ascending=False)
    series.to_csv(path_csv)

# ----------------------------
# 0) Load data
# ----------------------------
df = pd.read_csv(DATA, parse_dates=['timestamp']).set_index('timestamp').sort_index()
df = df.dropna(subset=[TARGET])
df = df[df[TARGET].isin([0,1])]
X, y = df[FEATS], df[TARGET].astype(int)

# 라벨 분포 로그
pos_rate_all = float((y == 1).mean())
print(f"[INFO] Dataset size={len(df)}  PosRate(all)={pos_rate_all:.3f}")

# Train/Test split (최근 20%를 테스트)
split = int(len(df)*0.8)
Xtr, ytr = X.iloc[:split], y.iloc[:split]
Xte, yte = X.iloc[split:], y.iloc[split:]
print(f"[INFO] Train={len(Xtr)}  Test={len(Xte)}  PosRate(train)={float((ytr==1).mean()):.3f}  PosRate(test)={float((yte==1).mean()):.3f}")

# ----------------------------
# 1) CV (XGB / RF)
# ----------------------------
xgb_cv_mean, xgb_cv_std = purged_cv_auc_xgb(Xtr, ytr)
rf_cv_mean,  rf_cv_std  = purged_cv_auc_rf(Xtr, ytr)
print(f"[CV][XGB] AUC = {xgb_cv_mean:.3f} ± {xgb_cv_std:.3f}")
print(f"[CV][RF ] AUC = {rf_cv_mean:.3f} ± {rf_cv_std:.3f}")

# ----------------------------
# 2) Final XGB training (train 내 마지막 10%를 val, 경계 purge)
# ----------------------------
val_size = max(int(len(Xtr)*VAL_FRAC), 100) if len(Xtr) > 100 else max(int(len(Xtr)*VAL_FRAC), 10)
core_end = max(len(Xtr) - val_size - H, 1)   # purge(H) 반영
X_core, y_core = Xtr.iloc[:core_end], ytr.iloc[:core_end]
X_val,  y_val  = Xtr.iloc[-val_size:], ytr.iloc[-val_size:]

# ... (기존 코드 동일)

# --- 최종 학습 ---
spw_core = compute_spw(y_core)
params_final = dict(xgb_params, scale_pos_weight=spw_core)

dcore = xgb.DMatrix(X_core, label=y_core, feature_names=FEATS)
dval  = xgb.DMatrix(X_val,  label=y_val,  feature_names=FEATS)
dtest = xgb.DMatrix(Xte,                 feature_names=FEATS)

final_booster = xgb.train(
    params=params_final,
    dtrain=dcore,
    num_boost_round=NUM_BOOST,
    evals=[(dval, "valid")],
    early_stopping_rounds=ES_ROUNDS,
    verbose_eval=False
)

# ❶ 검증/테스트 예측 저장 (임계치 튜닝용)
proba_val = predict_best(final_booster, dval)
proba_te  = predict_best(final_booster, dtest)

pd.DataFrame({
    'timestamp': X_val.index.astype(str),
    'y_true': y_val.values,
    'proba': proba_val
}).to_csv("predictions_xgb_val.csv", index=False)

pd.DataFrame({
    'timestamp': Xte.index.astype(str),
    'y_true': yte.values,
    'proba': proba_te
}).to_csv("predictions_xgb_cls.csv", index=False)

print(f"Saved: predictions_xgb_val.csv, predictions_xgb_cls.csv  | "
      f"scale_pos_weight(final)={spw_core:.2f}  best_iteration={getattr(final_booster,'best_iteration', None)}")

# (선택) 중요도 저장 유지
save_feature_importance(final_booster, FEATS, "feature_importance_xgb.csv")