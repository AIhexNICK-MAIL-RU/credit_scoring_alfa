"""CatBoost on v2 features (fractions + stats), shared 5-fold. Saves OOF + test preds."""
import time, numpy as np, polars as pl
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool

T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

folds = pl.read_parquet("folds.parquet").sort("id")
y = folds.get_column("flag").to_numpy(); fold = folds.get_column("fold").to_numpy()

tr = pl.read_parquet("feat2_train.parquet").sort("id")
# keep fractions (_f), stats (_mean/_std/_last/_first), counts of small-card cols, n_records, rn_max
keep = [c for c in tr.columns if c!="id" and (
        c in ("n_records","rn_max") or "_f" in c or c.endswith(("_mean","_std","_last","_first")))]
log(f"{len(keep)} features selected")
X = tr.select(keep).to_numpy().astype(np.float32); del tr
te = pl.read_parquet("feat2_test.parquet").sort("id")
test_ids = te.get_column("id").to_numpy()
Xt = te.select(keep).to_numpy().astype(np.float32); del te
log(f"X={X.shape} Xt={Xt.shape}")

NF=3
oof=np.zeros(len(y)); tpred=np.zeros(len(test_ids)); aucs=[]
params=dict(iterations=2200, learning_rate=0.06, depth=6, l2_leaf_reg=6.0,
    loss_function="Logloss", eval_metric="AUC", random_seed=42, task_type="CPU",
    thread_count=-1, early_stopping_rounds=120, verbose=300)
for f in range(NF):
    log(f"=== FOLD {f} ===")
    tri=fold!=f; vai=fold==f
    m=CatBoostClassifier(**params)
    m.fit(Pool(X[tri],y[tri]), eval_set=Pool(X[vai],y[vai]), use_best_model=True)
    oof[vai]=m.predict_proba(X[vai])[:,1]
    a=roc_auc_score(y[vai],oof[vai]); aucs.append(a)
    log(f"FOLD {f} AUC={a:.5f} best_iter={m.get_best_iteration()}")
    tpred+=m.predict_proba(Xt)[:,1]/NF
    np.save("oof_cat.npy",oof); np.save("test_cat.npy",tpred)  # checkpoint each fold
oof_auc=roc_auc_score(y[np.isin(fold,range(NF))],oof[np.isin(fold,range(NF))])
log(f"=== CatBoost OOF AUC = {oof_auc:.5f}  folds={[round(a,5) for a in aucs]} ===")
np.save("oof_cat.npy",oof); np.save("test_cat.npy",tpred); np.save("test_ids.npy",test_ids)
log("DONE")
