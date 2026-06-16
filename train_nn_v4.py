"""CPU sequence model — memory-safe: keep raw int8 sequences in RAM (~4GB total), apply
the per-feature offset shift PER BATCH (tiny). Plain nn.Embedding (multi-core) + Conv1d +
masked mean/last pool. Train fold!=0, val fold0. Saves OOF(fold0)+test preds."""
import time, json, copy, numpy as np, polars as pl, torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
torch.set_num_threads(10); torch.manual_seed(42); np.random.seed(42)
T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

meta=json.load(open("seq_meta.json")); cols=meta["cols"]; card=meta["card"]; L=meta["L"]; F=len(cols)
offsets=np.zeros(F,dtype=np.int64); tot=0
for i,c in enumerate(cols): offsets[i]=tot; tot+=card[c]
TOTAL=tot
Xtr=np.load("seq_train.npy")           # int8, raw (0=pad, values 1..card-shifted-by-1)
folds=pl.read_parquet("folds.parquet").sort("id")
y=folds.get_column("flag").to_numpy().astype(np.float32); fold=folds.get_column("fold").to_numpy()
log(f"Xtr={Xtr.shape} dtype={Xtr.dtype} emb_total={TOTAL}")

def to_batch(arr, idx):
    """gather rows (int8), shift into per-feature embedding range on the fly -> int64 + mask"""
    xb=arr[idx].astype(np.int64)           # [b,L,F] small
    m=(xb!=0)
    xb=np.where(m, xb+offsets, 0)
    return torch.from_numpy(xb), torch.from_numpy(m.any(-1).astype(np.float32)).unsqueeze(-1)

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
        b=sel[s:s+bs]; xb,mb=to_batch(Xtr,b); yb=torch.from_numpy(y[b])
        out=net(xb,mb); loss=crit(out,yb); opt.zero_grad(); loss.backward(); opt.step()
        tl+=loss.item()*len(b)
    return tl/len(sel)

def predict_idx(net,arr,idx,bs=16384):
    net.eval(); out=np.zeros(len(idx))
    with torch.no_grad():
        for s in range(0,len(idx),bs):
            xb,mb=to_batch(arr,idx[s:s+bs]); out[s:s+bs]=torch.sigmoid(net(xb,mb)).numpy()
    return out

tri=np.where(fold!=0)[0]; vai=np.where(fold==0)[0]
net=Net(); opt=torch.optim.Adam(net.parameters(),lr=2e-3,weight_decay=1e-5)
EPOCHS=12; BS=4096; CAP=800000
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
best=0; best_state=None
for ep in range(EPOCHS):
    t0=time.time(); tl=train_epoch(net,tri,opt,BS,CAP)
    vp=predict_idx(net,Xtr,vai); a=roc_auc_score(y[vai],vp); sched.step()
    log(f"ep{ep} loss={tl:.4f} val_auc={a:.5f} ({time.time()-t0:.0f}s)")
    if a>best:
        best=a; best_state=copy.deepcopy(net.state_dict()); best_val=vp
        oof=np.full(len(y),np.nan); oof[vai]=best_val; np.save("oof_nn.npy",oof)
log(f"=== NN best val(fold0) AUC = {best:.5f} ===")
net.load_state_dict(best_state)
oof=np.full(len(y),np.nan); oof[vai]=best_val; np.save("oof_nn.npy",oof)
del Xtr
Xte=np.load("seq_test.npy"); test_ids=np.load("seqid_test.npy")
tpred=predict_idx(net,Xte,np.arange(len(Xte))); np.save("test_nn.npy",tpred)
log(f"test preds saved mean={tpred.mean():.4f}"); log("DONE")
