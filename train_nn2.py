"""MPS-friendly sequence model: single offset embedding (1 lookup) + Conv1d over the
credit-history sequence + masked global pooling + MLP. Avoids GRU (broken/slow on MPS).
Shared 3-fold split; saves OOF + test preds. Streams val AUC per epoch."""
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")
import time, json, copy, numpy as np, polars as pl, torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42)

meta=json.load(open("seq_meta.json")); cols=meta["cols"]; card=meta["card"]; L=meta["L"]
F=len(cols)
offsets=np.zeros(F,dtype=np.int64); tot=0
for i,c in enumerate(cols): offsets[i]=tot; tot+=card[c]
TOTAL=tot
Xtr=np.load("seq_train.npy"); Xte=np.load("seq_test.npy")
seqid_tr=np.load("seqid_train.npy"); test_ids=np.load("seqid_test.npy")
folds=pl.read_parquet("folds.parquet").sort("id")
assert np.array_equal(folds.get_column("id").to_numpy(), seqid_tr)
y=folds.get_column("flag").to_numpy().astype(np.float32); fold=folds.get_column("fold").to_numpy()
off_t=torch.from_numpy(offsets).to(dev)
log(f"dev={dev} Xtr={Xtr.shape} F={F} emb_total={TOTAL}")

D=24  # embedding dim per feature
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb=nn.Embedding(TOTAL+1, D, padding_idx=0)  # 0 stays pad (offset of pads handled by mask)
        self.proj=nn.Linear(F*D,192)
        self.pos=nn.Parameter(torch.zeros(1,L,192))
        self.c1=nn.Conv1d(192,192,3,padding=1); self.c2=nn.Conv1d(192,192,3,padding=1)
        self.act=nn.ReLU()
        self.head=nn.Sequential(nn.Linear(192*2,256),nn.ReLU(),nn.Dropout(0.2),
                                nn.Linear(256,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x):           # x [B,L,F] long (1..card, 0=pad)
        mask=(x!=0).any(-1).float()                 # [B,L]
        xo=torch.where(x>0, x+off_t, torch.zeros_like(x))  # shift into per-feature range; pads ->0
        e=self.emb(xo).reshape(x.size(0),L,F*D)      # [B,L,F*D]
        h=self.act(self.proj(e))+self.pos            # [B,L,192]
        m=mask.unsqueeze(-1)
        hc=h.transpose(1,2)                          # [B,192,L]
        hc=self.act(self.c1(hc)); hc=self.act(self.c2(hc))
        h=hc.transpose(1,2)*m                        # masked
        summ=h.sum(1); cnt=m.sum(1).clamp(min=1); mean=summ/cnt
        mx=(h.masked_fill(m==0,-1e9)).max(1).values
        return self.head(torch.cat([mean,mx],-1)).squeeze(-1)

crit=nn.BCEWithLogitsLoss()
def train_epoch(net,idx,opt,bs):
    net.train(); order=np.random.permutation(len(idx))
    for s in range(0,len(idx),bs):
        b=idx[order[s:s+bs]]
        xb=torch.from_numpy(Xtr[b].astype(np.int64)).to(dev)
        yb=torch.from_numpy(y[b]).to(dev)
        out=net(xb); loss=crit(out,yb)
        opt.zero_grad(); loss.backward(); opt.step()

def predict(net,Xsrc,rows,bs=8192):
    net.eval(); out=np.zeros(len(rows))
    with torch.no_grad():
        for s in range(0,len(rows),bs):
            xb=torch.from_numpy(Xsrc[rows[s:s+bs]].astype(np.int64)).to(dev)
            out[s:s+bs]=torch.sigmoid(net(xb)).cpu().numpy()
    return out

NF=3; EPOCHS=6; BS=4096
allrows_te=np.arange(len(Xte))
oof=np.zeros(len(y)); tpred=np.zeros(len(test_ids)); aucs=[]
for f in range(NF):
    log(f"=== FOLD {f} ===")
    tri=np.where(fold!=f)[0]; vai=np.where(fold==f)[0]
    net=Net().to(dev); opt=torch.optim.Adam(net.parameters(),lr=2e-3,weight_decay=1e-5)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
    best=0; best_state=None
    for ep in range(EPOCHS):
        t0=time.time(); train_epoch(net,tri,opt,BS)
        vp=predict(net,Xtr,vai); a=roc_auc_score(y[vai],vp); sched.step()
        log(f"  fold{f} ep{ep} val_auc={a:.5f} ({time.time()-t0:.0f}s)")
        if a>best: best=a; best_state=copy.deepcopy(net.state_dict()); best_val=vp
    net.load_state_dict(best_state)
    oof[vai]=best_val; tpred+=predict(net,Xte,allrows_te)/NF; aucs.append(best)
    log(f"FOLD {f} best AUC={best:.5f}")
    np.save("oof_nn.npy",oof); np.save("test_nn.npy",tpred)
m=np.isin(fold,range(NF))
log(f"=== NN OOF AUC = {roc_auc_score(y[m],oof[m]):.5f} folds={[round(a,5) for a in aucs]} ===")
np.save("oof_nn.npy",oof); np.save("test_nn.npy",tpred)
log("DONE")
