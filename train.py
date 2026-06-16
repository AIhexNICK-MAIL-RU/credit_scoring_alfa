"""Train CatBoost on engineered features with 5-fold CV, predict test, build submission.
Metric: ROC-AUC. Progress streamed to terminal in real time.
"""
import time
import numpy as np
import polars as pl
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

# ---- Load ----
log("Loading features + target ...")
train = pl.read_parquet("feat_train.parquet")
target = pl.read_csv("train_target.csv")
train = train.join(target, on="id", how="inner").sort("id")
log(f"Train joined: {train.shape}")

test = pl.read_parquet("feat_test.parquet").sort("id")
log(f"Test: {test.shape}")

feat_cols = [c for c in train.columns if c not in ("id", "flag")]
X = train.select(feat_cols).to_numpy()
y = train.get_column("flag").to_numpy()
X_test = test.select(feat_cols).to_numpy()
test_ids = test.get_column("id").to_numpy()
log(f"X={X.shape}, positives={y.sum()} ({y.mean()*100:.2f}%)")

# ---- CV ----
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
oof = np.zeros(len(y))
test_pred = np.zeros(len(test_ids))
fold_aucs = []

params = dict(
    iterations=3000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=5.0,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    task_type="CPU",
    thread_count=-1,
    early_stopping_rounds=150,
    verbose=200,
)

for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
    log(f"===== FOLD {fold+1}/{N_FOLDS} =====")
    model = CatBoostClassifier(**params)
    model.fit(
        Pool(X[tr_idx], y[tr_idx]),
        eval_set=Pool(X[va_idx], y[va_idx]),
        use_best_model=True,
    )
    va_pred = model.predict_proba(X[va_idx])[:, 1]
    oof[va_idx] = va_pred
    auc = roc_auc_score(y[va_idx], va_pred)
    fold_aucs.append(auc)
    log(f"FOLD {fold+1} AUC = {auc:.5f}  (best_iter={model.get_best_iteration()})")
    test_pred += model.predict_proba(X_test)[:, 1] / N_FOLDS

oof_auc = roc_auc_score(y, oof)
log(f"===== OOF ROC-AUC = {oof_auc:.5f}  folds={[f'{a:.5f}' for a in fold_aucs]} =====")

# ---- Submission ----
sub = pd.DataFrame({"id": test_ids, "flag": test_pred})
# match sample order
sample = pd.read_csv("sample_submission.csv")
sub = sample[["id"]].merge(sub, on="id", how="left")
assert sub["flag"].isna().sum() == 0, "missing predictions!"
sub.to_csv("submission.csv", index=False)
log(f"Wrote submission.csv  rows={len(sub)}  mean_pred={sub['flag'].mean():.4f}")
log(f"FINAL OOF ROC-AUC = {oof_auc:.5f}")
log("DONE")
