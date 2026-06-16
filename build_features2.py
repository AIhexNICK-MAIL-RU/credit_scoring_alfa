"""Feature engineering v2: value-count / fraction encoding per id + stat aggregates.
For each categorical column, count occurrences of each observed value within an id's
history (and its fraction). This preserves the distribution that mean/min/max discard.
"""
import sys, time, json
import polars as pl

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

# observed unique values per column (from train) — computed once, reused for test
def get_levels():
    lf = pl.scan_parquet("train_data.parquet")
    cols = [c for c in lf.collect_schema().names() if c not in ("id", "rn")]
    uniq = {}
    for c in cols:
        vals = lf.select(pl.col(c).unique()).collect().get_column(c).to_list()
        uniq[c] = sorted(v for v in vals if v is not None)
    return cols, uniq

def build(src, out, cols, uniq):
    log(f"Scanning {src} ...")
    lf = pl.scan_parquet(src)
    aggs = [pl.len().alias("n_records"), pl.col("rn").max().alias("rn_max")]
    # value counts + fractions
    for c in cols:
        for v in uniq[c]:
            aggs.append((pl.col(c) == v).sum().cast(pl.Int32).alias(f"{c}_c{v}"))
    # recency / shape stats for ordinal columns
    for c in cols:
        aggs.append(pl.col(c).mean().alias(f"{c}_mean"))
        aggs.append(pl.col(c).std().alias(f"{c}_std"))
        aggs.append(pl.col(c).sort_by("rn").last().alias(f"{c}_last"))
        aggs.append(pl.col(c).sort_by("rn").first().alias(f"{c}_first"))
    log(f"{len(aggs)} aggregations")
    out_df = lf.group_by("id").agg(aggs).sort("id").collect(engine="streaming")
    log(f"Aggregated {out_df.shape}")
    # add fractions = count / n_records for the count columns
    n = pl.col("n_records")
    frac_exprs = []
    for c in cols:
        for v in uniq[c]:
            frac_exprs.append((pl.col(f"{c}_c{v}") / n).cast(pl.Float32).alias(f"{c}_f{v}"))
    out_df = out_df.with_columns(frac_exprs)
    out_df = out_df.fill_null(0).fill_nan(0)
    log(f"With fractions {out_df.shape}")
    out_df.write_parquet(out)
    log(f"Wrote {out}")

if __name__ == "__main__":
    cols, uniq = get_levels()
    json.dump(uniq, open("levels.json", "w"))
    log(f"levels: {sum(len(v) for v in uniq.values())} total")
    build("train_data.parquet", "feat2_train.parquet", cols, uniq)
    build("test_data.parquet", "feat2_test.parquet", cols, uniq)
    log("DONE")
