"""Sequence NN, memory-safe IN-PLACE int16 offset shift (no int64 transient — that was the
swap bug). Full shifted array resident at 6.2GB -> plain nn.Embedding runs multi-core.
Conv1d + masked mean/last pooling. Train fold!=0, val fold0. OOF(fold0)+test preds."""
import time, json, copy, numpy as np, polars as pl, torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
torch.set_num_threads(10); torch.manual_seed(42); np.random.seed(42)
T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

meta=json.load(open("seq_meta.json")); cols=meta["cols"]; card=meta["card"]; L=meta["L"]; F=len(cols)
offsets=np.zeros(F,dtype=np.int16); tot=0
for i,c in enumerate(cols): offsets[i]=tot; tot+=card[c]
TOTAL=int(tot)

def load_shift(path):
    raw=np.load(path)                 # int8 (3GB train)
    padpos=(raw==0)                   # bool, pads
    X=raw.astype(np.int16); del raw   # 6.2GB
    X+=offsets                        # in-place broadcast over last axis (int16, no big temp)
    X[padpos]=0; del padpos           # restore pad=0
    return X

Xtr=load_shift("seq_train.npy")
Mtr=(Xtr!=0).any(-1).astype(np.float32)
folds=pl.read_parquet("folds.parquet").sort("id")
y=folds.get_column("flag").to_numpy().astype(np.float32); fold=folds.get_column("fold").to_numpy()
log(f"Xtr={Xtr.shape} {Xtr.dtype} emb_total={TOTAL}")

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
        last_idx=(m.squeeze(-1).sum(1)-1).clamp(min=0).long()
        last=h[torch.arange(x.size(0)),last_idx]
        return s.head(torch.cat([mean,last],-1)).squeeze(-1)

crit=nn.BCEWithLogitsLoss()
def train_epoch(net,idx,opt,bs,cap):
    net.train(); sel=np.random.permutation(idx)[:cap]; tl=0
    for s in range(0,len(sel),bs):
        b=sel[s:s+bs]
        xb=torch.from_numpy(Xtr[b].astype(np.int64)); mb=torch.from_numpy(Mtr[b]).unsqueeze(-1); yb=torch.from_numpy(y[b])
        out=net(xb,mb); loss=crit(out,yb); opt.zero_grad(); loss.backward(); opt.step()
        tl+=loss.item()*len(b)
    return tl/len(sel)

def predict(net,Xs,Ms,bs=16384):
    net.eval(); out=np.zeros(len(Xs))
    with torch.no_grad():
        for s in range(0,len(Xs),bs):
            xb=torch.from_numpy(Xs[s:s+bs].astype(np.int64)); mb=torch.from_numpy(Ms[s:s+bs]).unsqueeze(-1)
            out[s:s+bs]=torch.sigmoid(net(xb,mb)).numpy()
    return out

tri=np.where(fold!=0)[0]; vai=np.where(fold==0)[0]
net=Net(); opt=torch.optim.Adam(net.parameters(),lr=2e-3,weight_decay=1e-5)
EPOCHS=14; BS=4096; CAP=1000000
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
best=0; best_state=None; bad=0
for ep in range(EPOCHS):
    t0=time.time(); tl=train_epoch(net,tri,opt,BS,CAP)
    vp=predict(net,Xtr[vai],Mtr[vai]); a=roc_auc_score(y[vai],vp); sched.step()
    log(f"ep{ep} loss={tl:.4f} val_auc={a:.5f} ({time.time()-t0:.0f}s)")
    if a>best:
        best=a; best_state=copy.deepcopy(net.state_dict()); best_val=vp; bad=0
        oof=np.full(len(y),np.nan); oof[vai]=best_val; np.save("oof_nn.npy",oof)
    else:
        bad+=1
        if bad>=3: log("early stop"); break
log(f"=== NN best val(fold0) AUC = {best:.5f} ===")
net.load_state_dict(best_state)
oof=np.full(len(y),np.nan); oof[vai]=best_val; np.save("oof_nn.npy",oof)
del Xtr, Mtr
Xte=load_shift("seq_test.npy"); Mte=(Xte!=0).any(-1).astype(np.float32)
tpred=predict(net,Xte,Mte); np.save("test_nn.npy",tpred)
log(f"test preds saved mean={tpred.mean():.4f}"); log("DONE")
