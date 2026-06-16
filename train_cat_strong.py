"""Stronger CatBoost: depth 7, lower lr, more iterations (v2 best_iter hit the cap →
undertrained). Single split train(fold!=0)/val(fold0). Saves test preds + fold0 OOF."""
import time, numpy as np, polars as pl
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

folds=pl.read_parquet("folds.parquet").sort("id")
y=folds.get_column("flag").to_numpy(); fold=folds.get_column("fold").to_numpy()
tr=pl.read_parquet("feat2_train.parquet").sort("id")
keep=[c for c in tr.columns if c!="id" and (
      c in ("n_records","rn_max") or "_f" in c or c.endswith(("_mean","_std","_last","_first")))]
log(f"{len(keep)} features")
X=tr.select(keep).to_numpy().astype(np.float32); del tr
tri=fold!=0; vai=fold==0
ptr=Pool(X[tri],y[tri]); pva=Pool(X[vai],y[vai]); del X
log("pools built, training ...")
m=CatBoostClassifier(iterations=6000, learning_rate=0.035, depth=7, l2_leaf_reg=8.0,
    loss_function="Logloss", eval_metric="AUC", random_seed=42, task_type="CPU",
    thread_count=-1, early_stopping_rounds=250, verbose=300)
m.fit(ptr, eval_set=pva, use_best_model=True)
vp=m.predict_proba(pva)[:,1]; a=roc_auc_score(y[vai],vp)
log(f"=== STRONG CatBoost fold0 AUC = {a:.5f} best_iter={m.get_best_iteration()} ===")
del ptr, pva
oof=np.full(len(y),np.nan); oof[vai]=vp; np.save("oof_catS.npy",oof)
m.save_model("catS.cbm")
te=pl.read_parquet("feat2_test.parquet").sort("id")
test_ids=te.get_column("id").to_numpy()
Xt=te.select(keep).to_numpy().astype(np.float32); del te
tp=m.predict_proba(Xt)[:,1]; np.save("test_catS.npy",tp); np.save("test_ids.npy",test_ids)
log(f"test preds saved mean={tp.mean():.4f}"); log("DONE")
