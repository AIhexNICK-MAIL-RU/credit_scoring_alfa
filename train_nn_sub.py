"""Ultra-low-memory NN for k-fold bagging under severe RAM pressure. Loads ONLY a
subsample of needed rows into a compact int8 RAM array (~1.5GB). Per-batch offset shift.
Usage: python train_nn_sub.py K SEED [NTRAIN]
Saves oof_nn_fK.npy, test_nn_fK.npy, nn_fK_state.pt."""
import sys, time, json, copy, numpy as np, polars as pl, torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
torch.set_num_threads(10)
K=int(sys.argv[1]); SEED=int(sys.argv[2]) if len(sys.argv)>2 else 42
NTRAIN=int(sys.argv[3]) if len(sys.argv)>3 else 750000
torch.manual_seed(SEED); np.random.seed(SEED)
T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s][f{K}] {m}", flush=True)

meta=json.load(open("seq_meta.json")); cols=meta["cols"]; card=meta["card"]; L=meta["L"]; F=len(cols)
offsets=np.zeros(F,dtype=np.int64); tot=0
for i,c in enumerate(cols): offsets[i]=tot; tot+=card[c]
TOTAL=int(tot)
folds=pl.read_parquet("folds.parquet").sort("id")
y=folds.get_column("flag").to_numpy().astype(np.float32); fold=folds.get_column("fold").to_numpy()

mm=np.load("seq_train.npy", mmap_mode="r")
tri_all=np.where(fold!=K)[0]; vai=np.where(fold==K)[0]
rng=np.random.default_rng(SEED)
tri_sub=np.sort(rng.choice(tri_all, size=min(NTRAIN,len(tri_all)), replace=False))
keep=np.sort(np.concatenate([tri_sub, vai]))
log(f"gathering {len(keep)} rows into RAM ...")
Xc=np.ascontiguousarray(mm[keep])          # compact int8 in RAM (~1.4GB)
del mm
pos={int(g):i for i,g in enumerate(keep)}  # global idx -> local pos
tri_l=np.fromiter((pos[int(g)] for g in tri_sub), dtype=np.int64)
vai_l=np.fromiter((pos[int(g)] for g in vai), dtype=np.int64)
yc=y[keep]
log(f"Xc={Xc.shape} {Xc.dtype} train={len(tri_l)} val={len(vai_l)}")

def to_batch(idx):
    xb=Xc[idx].astype(np.int64); m=(xb!=0); xb=np.where(m,xb+offsets,0)
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
        li=(m.squeeze(-1).sum(1)-1).clamp(min=0).long(); last=h[torch.arange(x.size(0)),li]
        return s.head(torch.cat([mean,last],-1)).squeeze(-1)

crit=nn.BCEWithLogitsLoss()
def train_epoch(net,idx,opt,bs):
    net.train(); order=np.random.permutation(idx); tl=0
    for s in range(0,len(order),bs):
        b=order[s:s+bs]; xb,mb=to_batch(b); yb=torch.from_numpy(yc[b])
        out=net(xb,mb); loss=crit(out,yb); opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item()*len(b)
    return tl/len(order)
def predict_local(net,idx,bs=16384):
    net.eval(); out=np.zeros(len(idx))
    with torch.no_grad():
        for s in range(0,len(idx),bs):
            xb,mb=to_batch(idx[s:s+bs]); out[s:s+bs]=torch.sigmoid(net(xb,mb)).numpy()
    return out

net=Net(); opt=torch.optim.Adam(net.parameters(),lr=2e-3,weight_decay=1e-5)
EPOCHS=12; BS=4096
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
best=0; best_state=None; bad=0
for ep in range(EPOCHS):
    t0=time.time(); tl=train_epoch(net,tri_l,opt,BS)
    vp=predict_local(net,vai_l); a=roc_auc_score(yc[vai_l],vp); sched.step()
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
del Xc
# predict test via mmap (sequential batches)
mt=np.load("seq_test.npy", mmap_mode="r"); nt=mt.shape[0]; out=np.zeros(nt); bs=16384
net.eval()
with torch.no_grad():
    for s in range(0,nt,bs):
        xb8=np.ascontiguousarray(mt[s:s+bs]).astype(np.int64); m=(xb8!=0)
        xb=torch.from_numpy(np.where(m,xb8+offsets,0)); mb=torch.from_numpy(m.any(-1).astype(np.float32)).unsqueeze(-1)
        out[s:s+bs]=torch.sigmoid(net(xb,mb)).numpy()
np.save(f"test_nn_f{K}.npy",out); log(f"test saved mean={out.mean():.4f}"); log("DONE")
