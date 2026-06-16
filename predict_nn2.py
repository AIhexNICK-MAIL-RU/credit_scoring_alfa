"""Predict test for NN2 from saved weights (nn2_state.pt). Memory-safe: int8 + per-batch shift."""
import json, numpy as np, torch, torch.nn as nn
torch.set_num_threads(10)
meta=json.load(open("seq_meta.json")); cols=meta["cols"]; card=meta["card"]; L=meta["L"]; F=len(cols)
offsets=np.zeros(F,dtype=np.int64); tot=0
for i,c in enumerate(cols): offsets[i]=tot; tot+=card[c]
TOTAL=int(tot)
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
net=Net(); net.load_state_dict(torch.load("nn2_state.pt")); net.eval()
Xte=np.load("seq_test.npy")   # int8
out=np.zeros(len(Xte)); bs=8192
with torch.no_grad():
    for sidx in range(0,len(Xte),bs):
        xb8=Xte[sidx:sidx+bs].astype(np.int64); m=(xb8!=0)
        xb=torch.from_numpy(np.where(m,xb8+offsets,0))
        mb=torch.from_numpy(m.any(-1).astype(np.float32)).unsqueeze(-1)
        out[sidx:sidx+bs]=torch.sigmoid(net(xb,mb)).numpy()
np.save("test_nn2.npy",out)
print("test_nn2 saved mean",round(out.mean(),4),"n",len(out))
