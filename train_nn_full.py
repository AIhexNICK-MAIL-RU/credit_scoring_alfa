"""Full-data NN for k-fold bagging (run AFTER reboot — needs ~8GB free RAM).
Full int16 in-place offset shift (6.2GB resident), plain nn.Embedding (multi-core),
Conv1d + masked mean/last pooling. Validates on fold=K, trains on the rest (full data).
Usage: python train_nn_full.py K SEED
Saves oof_nn_fK.npy, test_nn_fK.npy, nn_fK_state.pt."""
import sys, time, json, copy, numpy as np, polars as pl, torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
torch.set_num_threads(10)
K=int(sys.argv[1]); SEED=int(sys.argv[2]) if len(sys.argv)>2 else 42
torch.manual_seed(SEED); np.random.seed(SEED)
T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s][f{K}] {m}", flush=True)

meta=json.load(open("seq_meta.json")); cols=meta["cols"]; card=meta["card"]; L=meta["L"]; F=len(cols)
offs=np.zeros(F,dtype=np.int16); tot=0
for i,c in enumerate(cols): offs[i]=tot; tot+=card[c]
TOTAL=int(tot)
def load_shift(path):
    raw=np.load(path); padpos=(raw==0); X=raw.astype(np.int16); del raw
    X+=offs; X[padpos]=0; del padpos; return X
Xtr=load_shift("seq_train.npy"); Mtr=(Xtr!=0).any(-1).astype(np.float32)
folds=pl.read_parquet("folds.parquet").sort("id")
y=folds.get_column("flag").to_numpy().astype(np.float32); fold=folds.get_column("fold").to_numpy()
log(f"Xtr={Xtr.shape} emb_total={TOTAL} seed={SEED}")

D=16
class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb=nn.Embedding(TOTAL+1,D,padding_idx=0); s.proj=nn.Linear(F*D,160); s.act=nn.ReLU()
        s.c1=nn.Conv1d(160,160,3,padding=1); s.c2=nn.Conv1d(160,160,3,padding=1)
        s.head=nn.Sequential(nn.Linear(160*2,256),nn.ReLU(),nn.Dropout(0.2),
                             nn.Linear(256,64),nn.ReLU(),nn.Linear(64,1))
    def forward(s,x,m):
        e=s.emb(x).reshape(x.size(0),L,F*D)
        h=s.act(s.proj(e)).transpose(1,2); h=s.act(s.c1(h)); h=s.act(s.c2(h)).transpose(1,2)
        h=h*m; mean=h.sum(1)/m.sum(1).clamp(min=1)
        li=(m.squeeze(-1).sum(1)-1).clamp(min=0).long(); last=h[torch.arange(x.size(0)),li]
        return s.head(torch.cat([mean,last],-1)).squeeze(-1)

crit=nn.BCEWithLogitsLoss()
def train_epoch(net,idx,opt,bs,cap):
    net.train(); sel=np.random.permutation(idx)[:cap]; tl=0
    for s in range(0,len(sel),bs):
        b=sel[s:s+bs]
        xb=torch.from_numpy(Xtr[b].astype(np.int64)); mb=torch.from_numpy(Mtr[b]).unsqueeze(-1); yb=torch.from_numpy(y[b])
        out=net(xb,mb); loss=crit(out,yb); opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item()*len(b)
    return tl/len(sel)
def predict(net,Xs,Ms,bs=16384):
    net.eval(); out=np.zeros(len(Xs))
    with torch.no_grad():
        for s in range(0,len(Xs),bs):
            xb=torch.from_numpy(Xs[s:s+bs].astype(np.int64)); mb=torch.from_numpy(Ms[s:s+bs]).unsqueeze(-1)
            out[s:s+bs]=torch.sigmoid(net(xb,mb)).numpy()
    return out

tri=np.where(fold!=K)[0]; vai=np.where(fold==K)[0]
net=Net(); opt=torch.optim.Adam(net.parameters(),lr=2e-3,weight_decay=1e-5)
EPOCHS=14; BS=4096; CAP=1100000
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
best=0; best_state=None; bad=0
for ep in range(EPOCHS):
    t0=time.time(); tl=train_epoch(net,tri,opt,BS,CAP)
    vp=predict(net,Xtr[vai],Mtr[vai]); a=roc_auc_score(y[vai],vp); sched.step()
    log(f"ep{ep} loss={tl:.4f} val_auc={a:.5f} ({time.time()-t0:.0f}s)")
    if a>best:
        best=a; best_state=copy.deepcopy(net.state_dict()); best_val=vp; bad=0
        oof=np.full(len(y),np.nan); oof[vai]=best_val; np.save(f"oof_nn_f{K}.npy",oof)
        torch.save(best_state,f"nn_f{K}_state.pt")
    else:
        bad+=1
        if bad>=3: log("early stop"); break
log(f"=== fold{K} best val AUC = {best:.5f} ===")
net.load_state_dict(best_state); oof=np.full(len(y),np.nan); oof[vai]=best_val; np.save(f"oof_nn_f{K}.npy",oof)
del Xtr, Mtr
Xte=load_shift("seq_test.npy"); Mte=(Xte!=0).any(-1).astype(np.float32)
np.save(f"test_nn_f{K}.npy",predict(net,Xte,Mte)); log("DONE")
