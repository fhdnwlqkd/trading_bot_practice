# labs/L2_modeling/train_xgb_reg_pnl.py
import pandas as pd, numpy as np, xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

DATA = 'dataset_L2_tb.csv'
FEATS = ['ret_1','ret_5','ret_15','ma_20','vol_20','z_ret_20','ema_12','ema_26','macd','rsi_14','bb_width_20','atr_14','roc_10','close']
TARGET = 'y_ret_H'
H = 15    # 라벨 스크립트와 맞추세요
VAL_FRAC = 0.10
NUM_BOOST = 1500
ES_ROUNDS = 100
PARAMS = dict(objective='reg:squarederror', eval_metric='rmse', eta=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, lambda_=1.0, seed=42, verbosity=0)

df = pd.read_csv(DATA, parse_dates=['timestamp']).set_index('timestamp').sort_index()
df = df.dropna(subset=[TARGET])
X, y = df[FEATS], df[TARGET]

split = int(len(df)*0.8)
Xtr, ytr = X.iloc[:split], y.iloc[:split]
Xte, yte = X.iloc[split:], y.iloc[split:]

val_size = max(int(len(Xtr)*VAL_FRAC), 100) if len(Xtr) > 100 else max(int(len(Xtr)*VAL_FRAC), 10)
core_end = max(len(Xtr) - val_size - H, 1)
X_core, y_core = Xtr.iloc[:core_end], ytr.iloc[:core_end]
X_val,  y_val  = Xtr.iloc[-val_size:], ytr.iloc[-val_size:]

dcore = xgb.DMatrix(X_core, label=y_core, feature_names=FEATS)
dval  = xgb.DMatrix(X_val,  label=y_val,  feature_names=FEATS)
dtest = xgb.DMatrix(Xte,                 feature_names=FEATS)

booster = xgb.train(PARAMS, dcore, NUM_BOOST, evals=[(dval,'valid')], early_stopping_rounds=ES_ROUNDS, verbose_eval=False)
bi = getattr(booster,'best_iteration', None)
def predict_best(b,d): 
    return b.predict(d, iteration_range=(0, bi+1)) if bi is not None else b.predict(d)

pred_val = predict_best(booster, dval)
pred_te  = predict_best(booster, dtest)
print("[VAL] MAE=", mean_absolute_error(y_val, pred_val))

pd.DataFrame({'timestamp': X_val.index.astype(str), 'y_true': y_val.values, 'y_hat': pred_val}).to_csv('predictions_xgb_reg_val.csv', index=False)
pd.DataFrame({'timestamp': Xte.index.astype(str),  'y_true': yte.values,  'y_hat': pred_te }).to_csv('predictions_xgb_reg_te.csv', index=False)
print("Saved: predictions_xgb_reg_val.csv, predictions_xgb_reg_te.csv")