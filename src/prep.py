# Build baskets from transactions

import pandas as pd
from pathlib import Path
import argparse

def main(input_csv, out_parquet, out_csv):
    df = pd.read_csv(input_csv, low_memory=False)

    if "date_time" in df.columns:
        df["date_time"] = pd.to_datetime(df["date_time"], dayfirst=True, errors="coerce")

    df["Item"] = df["Item"].astype(str).str.strip()

    sort_cols = ["date_time", "Transaction"] if "date_time" in df.columns else ["Transaction"]
    df = df.sort_values(sort_cols)

    baskets = (df.groupby("Transaction")["Item"]
                 .apply(lambda s: sorted(set(s)))
                 .reset_index(name="items"))

    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    baskets.to_parquet(out_parquet, index=False)
    baskets.to_csv(out_csv, index=False)

    # Preview
    print(f"✅ Baskets: {len(baskets)}")
    print(f"➡ Saved: {out_parquet}")
    print(f"➡ Also saved (CSV): {out_csv}")
    print(baskets.head(5).to_string(index=False))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/raw/bread_basket.csv")
    ap.add_argument("--out_parquet", default="../data/processed/baskets.parquet")
    ap.add_argument("--out_csv", default="../data/processed/baskets.csv")
    args = ap.parse_args()
    main(args.input, args.out_parquet, args.out_csv)
