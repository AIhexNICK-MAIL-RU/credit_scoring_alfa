"""GRU sequence model with per-feature categorical embeddings on MPS GPU.
5-fold (shared split), saves OOF + averaged test predictions. Streams val AUC live."""
import time, json, numpy as np, polars as pl, torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
torch.manual_seed(42); np.random.seed(42)

meta=json.load(open("seq_meta.json")); cols=meta["cols"]; card=meta["card"]; L=meta["L"]
Xtr=np.load("seq_train.npy"); Xte=np.load("seq_test.npy")
seqid_tr=np.load("seqid_train.npy"); test_ids=np.load("seqid_test.npy")
folds=pl.read_parquet("folds.parquet").sort("id")
assert np.array_equal(folds.get_column("id").to_numpy(), seqid_tr)
y=folds.get_column("flag").to_numpy().astype(np.float32); fold=folds.get_column("fold").to_numpy()
F=len(cols)
log(f"dev={dev} Xtr={Xtr.shape} F={F}")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embs=nn.ModuleList([nn.Embedding(card[c], min(8,max(2,card[c]//2)), padding_idx=0) for c in cols])
        din=sum(e.embedding_dim for e in self.embs)
        self.gru=nn.GRU(din, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.head=nn.Sequential(nn.Linear(128*2*3,256), nn.ReLU(), nn.Dropout(0.2),
                                nn.Linear(256,64), nn.ReLU(), nn.Linear(64,1))
    def forward(self,x):  # x [B,L,F] long
        e=torch.cat([emb(x[...,i]) for i,emb in enumerate(self.embs)],dim=-1)
        mask=(x!=0).any(-1).float()            # [B,L]
        o,_=self.gru(e)                         # [B,L,256]
        m=mask.unsqueeze(-1)
        summ=(o*m).sum(1); cnt=m.sum(1).clamp(min=1)
        mean=summ/cnt
        mx=(o.masked_fill(m==0,-1e9)).max(1).values
        last_idx=(mask.sum(1)-1).clamp(min=0).long()
        last=o[torch.arange(o.size(0),device=o.device),last_idx]
        h=torch.cat([mean,mx,last],-1)
        return self.head(h).squeeze(-1)

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

import copy
NF=3; EPOCHS=6; BS=4096
allrows_te=np.arange(len(Xte))
oof=np.zeros(len(y)); tpred=np.zeros(len(test_ids)); aucs=[]
for f in range(NF):
    log(f"=== FOLD {f} ===")
    tri=np.where(fold!=f)[0]; vai=np.where(fold==f)[0]
    net=Net().to(dev); opt=torch.optim.Adam(net.parameters(),lr=1.5e-3)
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
