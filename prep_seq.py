"""Build padded sequence tensors [N, L, F] of categorical codes (last L credit products
per id, ordered by rn). 0 = padding; real values shifted +1. int8."""
import time, json, numpy as np, polars as pl
T0=time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
L=25

def build(src, tag):
    log(f"loading {src}")
    df = pl.read_parquet(src).sort(["id","rn"])
    cols = [c for c in df.columns if c not in ("id","rn")]
    ids = df.get_column("id").to_numpy()
    F = df.select(cols).to_numpy().astype(np.int16)  # values >=0
    maxv = F.max(axis=0)
    Fs = (F+1).astype(np.int16)  # shift, 0=pad
    n = len(ids)
    uniq, first_idx, counts = np.unique(ids, return_index=True, return_counts=True)
    num = len(uniq)
    start = np.repeat(first_idx, counts)
    within = np.arange(n) - start
    size = np.repeat(counts, counts)
    from_end = size-1-within
    keep = from_end < L
    pos = (L-1-from_end)
    g = np.repeat(np.arange(num), counts)
    out = np.zeros((num, L, len(cols)), dtype=np.int8)
    out[g[keep], pos[keep]] = Fs[keep].astype(np.int8)
    np.save(f"seq_{tag}.npy", out)
    np.save(f"seqid_{tag}.npy", uniq)
    log(f"{tag}: seq {out.shape} ids {num}")
    return cols, maxv

cols, maxv_tr = build("train_data.parquet", "train")
_, maxv_te = build("test_data.parquet", "test")
card = {c:int(max(maxv_tr[i], maxv_te[i]))+2 for i,c in enumerate(cols)}  # +1 shift +1 pad
json.dump({"cols":cols,"card":card,"L":L}, open("seq_meta.json","w"))
log(f"cards saved. total emb rows {sum(card.values())}")
log("DONE")
