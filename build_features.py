"""Feature engineering for Alfa-Bank credit scoring.
Aggregates per-id transaction/credit-product history into a feature table.
All input features are categorical-coded ordinal integers.
"""
import sys
import time
import polars as pl

T0 = time.time()
def log(msg):
    print(f"[{time.time()-T0:7.1f}s] {msg}", flush=True)

def build(src_parquet, out_parquet):
    log(f"Scanning {src_parquet} ...")
    lf = pl.scan_parquet(src_parquet)
    cols = lf.collect_schema().names()
    feat_cols = [c for c in cols if c not in ("id", "rn")]
    log(f"{len(feat_cols)} feature columns")

    aggs = [pl.len().alias("n_records"),
            pl.col("rn").max().alias("rn_max")]

    # Statistical aggregates for every feature column
    for c in feat_cols:
        aggs.append(pl.col(c).mean().alias(f"{c}_mean"))
        aggs.append(pl.col(c).max().alias(f"{c}_max"))
        aggs.append(pl.col(c).min().alias(f"{c}_min"))
        aggs.append(pl.col(c).std().alias(f"{c}_std"))
        aggs.append(pl.col(c).sum().alias(f"{c}_sum"))
        # last value = value at most recent credit product (max rn)
        aggs.append(pl.col(c).sort_by("rn").last().alias(f"{c}_last"))

    # nunique for key categorical encodings
    for c in ["enc_loans_credit_type", "enc_loans_credit_status",
              "enc_loans_account_holder_type", "enc_loans_account_cur"]:
        aggs.append(pl.col(c).n_unique().alias(f"{c}_nuniq"))

    log("Running group_by aggregation ...")
    out = lf.group_by("id").agg(aggs).sort("id").collect(engine="streaming")
    log(f"Aggregated: {out.shape[0]} ids x {out.shape[1]} cols")

    out = out.fill_null(0).fill_nan(0)
    out.write_parquet(out_parquet)
    log(f"Wrote {out_parquet}")
    return out

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("train", "both"):
        build("train_data.parquet", "feat_train.parquet")
    if which in ("test", "both"):
        build("test_data.parquet", "feat_test.parquet")
    log("DONE")
