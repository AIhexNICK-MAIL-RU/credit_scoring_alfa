"""Resilient final training: single CatBoost with holdout early-stopping.
Writes submission to disk immediately after fitting. Streams progress.
Cross-validated estimate from prior 5-fold run: fold AUCs 0.759/0.763/0.755/0.757.
"""
import time
import numpy as np
import polars as pl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

log("Loading features + target ...")
train = pl.read_parquet("feat_train.parquet")
target = pl.read_csv("train_target.csv")
train = train.join(target, on="id", how="inner").sort("id")
test = pl.read_parquet("feat_test.parquet").sort("id")
feat_cols = [c for c in train.columns if c not in ("id", "flag")]
X = train.select(feat_cols).to_numpy()
y = train.get_column("flag").to_numpy()
X_test = test.select(feat_cols).to_numpy()
test_ids = test.get_column("id").to_numpy()
log(f"X={X.shape}, test={X_test.shape}, positives={y.sum()} ({y.mean()*100:.2f}%)")

X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.07, random_state=42, stratify=y)
log(f"Train={X_tr.shape}, Holdout={X_va.shape}")

model = CatBoostClassifier(
    iterations=4000, learning_rate=0.05, depth=6, l2_leaf_reg=5.0,
    loss_function="Logloss", eval_metric="AUC", random_seed=42,
    task_type="CPU", thread_count=-1, early_stopping_rounds=200, verbose=200,
)
model.fit(Pool(X_tr, y_tr), eval_set=Pool(X_va, y_va), use_best_model=True)

va_pred = model.predict_proba(X_va)[:, 1]
auc = roc_auc_score(y_va, va_pred)
log(f"HOLDOUT ROC-AUC = {auc:.5f}  (best_iter={model.get_best_iteration()})")

model.save_model("catboost_model.cbm")
log("Saved model -> catboost_model.cbm")

test_pred = model.predict_proba(X_test)[:, 1]
sub = pd.DataFrame({"id": test_ids, "flag": test_pred})
sample = pd.read_csv("sample_submission.csv")
sub = sample[["id"]].merge(sub, on="id", how="left")
assert sub["flag"].isna().sum() == 0, "missing predictions!"
sub.to_csv("submission.csv", index=False)
log(f"Wrote submission.csv rows={len(sub)} mean={sub['flag'].mean():.4f}")

# feature importance (top 25)
imp = model.get_feature_importance()
order = np.argsort(imp)[::-1][:25]
log("Top-25 features:")
for i in order:
    print(f"    {feat_cols[i]:40s} {imp[i]:.3f}", flush=True)
log(f"FINAL HOLDOUT ROC-AUC = {auc:.5f}")
log("DONE")
