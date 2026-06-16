"""Memory-light NN for k-fold bagging. seq arrays via MMAP (no big RAM load -> no swap,
survives system memory pressure). Per-batch int8->int64 offset shift. Validates on fold=K.
Usage: python train_nn_fold.py K SEED
Saves oof_nn_fK.npy, test_nn_fK.npy, nn_fK_state.pt."""
import sys, time, json, copy, numpy as np, polars as pl, torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
torch.set_num_threads(10)
K=int(sys.argv[1]); SEED=int(sys.argv[2]) if len(sys.argv)>2 else 42
torch.manual_seed(SEED); np.random.seed(SEED)
T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s][fold{K}] {m}", flush=True)

meta=json.load(open("seq_meta.json")); cols=meta["cols"]; card=meta["card"]; L=meta["L"]; F=len(cols)
offsets=np.zeros(F,dtype=np.int64); tot=0
for i,c in enumerate(cols): offsets[i]=tot; tot+=card[c]
TOTAL=int(tot)
Xtr=np.load("seq_train.npy")   # int8 fully in RAM (3GB, no swap)
folds=pl.read_parquet("folds.parquet").sort("id")
y=folds.get_column("flag").to_numpy().astype(np.float32); fold=folds.get_column("fold").to_numpy()
log(f"mmap Xtr={Xtr.shape} emb_total={TOTAL} seed={SEED}")

def to_batch(arr, idx):
    xb=np.asarray(arr[idx]).astype(np.int64); m=(xb!=0)
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
        b=sel[s:s+bs]
        xb,mb=to_batch(Xtr,b); yb=torch.from_numpy(y[b])
        out=net(xb,mb); loss=crit(out,yb); opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item()*len(b)
    return tl/len(sel)
def predict_idx(net,arr,idx,bs=16384):
    net.eval(); out=np.zeros(len(idx))
    with torch.no_grad():
        for s in range(0,len(idx),bs):
            xb,mb=to_batch(arr,idx[s:s+bs]); out[s:s+bs]=torch.sigmoid(net(xb,mb)).numpy()
    return out

tri=np.where(fold!=K)[0]; vai=np.where(fold==K)[0]
net=Net(); opt=torch.optim.Adam(net.parameters(),lr=2e-3,weight_decay=1e-5)
EPOCHS=12; BS=4096; CAP=1100000
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
best=0; best_state=None; bad=0
for ep in range(EPOCHS):
    t0=time.time(); tl=train_epoch(net,tri,opt,BS,CAP)
    vp=predict_idx(net,Xtr,vai); a=roc_auc_score(y[vai],vp); sched.step()
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
Xte=np.load("seq_test.npy")
tp=predict_idx(net,Xte,np.arange(Xte.shape[0])); np.save(f"test_nn_f{K}.npy",tp)
log(f"test preds saved mean={tp.mean():.4f}"); log("DONE")
