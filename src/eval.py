import argparse
import ast
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


# This method loads baskets from a Parquet or CSV file and returns a list of baskets
def load_baskets(path="data/processed/baskets.csv"):
    if path.endswith(".csv"):
        df = pd.read_csv(path, converters={
            "items": lambda x: ast.literal_eval(x)
        })
    else:
        df = pd.read_parquet(path)

    # Auto-detect item column ("items", "Item", etc.)
    item_col = None
    for c in df.columns:
        if any(s in c.lower() for s in ["item", "items", "basket"]):
            item_col = c
            break

    if item_col is None:
        raise ValueError(f"⚠️ No items column found in {df.columns}")

    # Convert to Python lists
    baskets = df[item_col].apply(lambda x: x if isinstance(x, list) else eval(x)).tolist()
    return baskets



# This method mines association rules from the given baskets using the apriori algorithm
def mine_rules_from_baskets(
    baskets, min_support=0.01, min_conf=0.2, allow_multi_ante=True
):
    """
    Returns a rule_table (dict: antecedent tuple -> list of (consequent, score))
    and the full rules dataframe.
    """
    te = TransactionEncoder()
    X = pd.DataFrame(te.fit(baskets).transform(baskets), columns=te.columns_)

    freq = apriori(X, min_support=min_support, use_colnames=True)
    if len(freq) == 0:
        return {}, pd.DataFrame()

    rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    if len(rules) == 0:
        return {}, pd.DataFrame()

    # Normalize antecedents/consequents to tuples and keep single-consequent rules
    rules["antecedents"] = rules["antecedents"].apply(lambda s: tuple(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: tuple(sorted(s)))
    rules = rules[rules["consequents"].apply(len) == 1].copy()
    if len(rules) == 0:
        return {}, pd.DataFrame()

    # Optional: only single-item antecedents (simpler) vs allow multi (better when available)
    if not allow_multi_ante:
        rules = rules[rules["antecedents"].apply(len) == 1].copy()

    # (Recommended) Keep only positively associated rules
    rules = rules[rules["lift"] > 1].copy()
    if len(rules) == 0:
        return {}, pd.DataFrame()

    rules["conseq"] = rules["consequents"].apply(lambda t: t[0])
    rules = rules.sort_values(["lift", "confidence"], ascending=False)

    # Score blend: emphasize lift (usefulness) + confidence (reliability)
    def score_row(r):
        return 0.7 * r["lift"] + 0.3 * r["confidence"]

    table = defaultdict(list)
    for _, r in rules.iterrows():
        a = r["antecedents"]
        s = score_row(r)
        table[a].append((r["conseq"], s))

    # Keep top-N per antecedent to avoid bloat
    for a in list(table.keys()):
        table[a] = sorted(table[a], key=lambda x: -x[1])[:50]

    return dict(table), rules


# This method recommends items based on the given context using the mined rules
def recommend_rules(context_items, rule_table, k=10):
    """
    Recommend using subset match: any rule whose antecedent ⊆ context.
    """
    basket_set = set(context_items)
    cand = defaultdict(float)
    for antecedent, conseqs in rule_table.items():
        if set(antecedent).issubset(basket_set):
            for c, s in conseqs:
                if c not in basket_set:
                    cand[c] = max(cand[c], s)
    if not cand:
        return []
    return [c for c, _ in sorted(cand.items(), key=lambda x: -x[1])[:k]]


# This method computes the top-k popular items from the training baskets
def popularity_topk(train_baskets, k=10):
    cnt = Counter(i for b in train_baskets for i in set(b))
    return [i for i, _ in cnt.most_common(k)]


# This method recommends items based on popularity, excluding context items
def recommend_pop(context_items, pop_list, k=10):
    context = set(context_items)
    recs = [i for i in pop_list if i not in context]
    return recs[:k]


# This method computes HitRate@k
def hitrate_at_k(held_out, recs, k):
    return 1.0 if held_out in set(recs[:k]) else 0.0


# This method computes Average Precision at k (APK)
def apk(held_out, recs, k):
    # Average precision at k for single held-out item simplifies to 1/rank if found
    for idx, item in enumerate(recs[:k], start=1):
        if item == held_out:
            return 1.0 / idx
    return 0.0


def main(
    baskets_path="data/processed/baskets.csv",
    min_support=0.01,
    min_conf=0.2,
    k=10,
    allow_multi_ante=True,
    out_dir="outputs/tables",
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    baskets = load_baskets(baskets_path)
    n = len(baskets)
    cut = int(0.8 * n)
    train, test = baskets[:cut], baskets[cut:]

    # Mine on TRAIN only
    rule_table, rules_df = mine_rules_from_baskets(
        train, min_support=min_support, min_conf=min_conf, allow_multi_ante=allow_multi_ante
    )

    pop_list = popularity_topk(train, k*3)  # take a little extra then filter per context

    records = []
    total_cases_rules = 0
    total_cases_pop = 0

    for b in test:
        # leave-one-out over items in basket (only for baskets with at least 2 items)
        if len(b) < 2:
            continue
        for held_out in b:
            context = [x for x in b if x != held_out]

            # Rules
            rec_rules = recommend_rules(context, rule_table, k=k)
            if rec_rules:
                total_cases_rules += 1
                records.append(
                    {
                        "model": "rules",
                        "hit": hitrate_at_k(held_out, rec_rules, k),
                        "apk": apk(held_out, rec_rules, k),
                        "context_size": len(context),
                    }
                )

            # Popularity baseline
            rec_pop = recommend_pop(context, pop_list, k=k)
            if rec_pop:
                total_cases_pop += 1
                records.append(
                    {
                        "model": "popularity",
                        "hit": hitrate_at_k(held_out, rec_pop, k),
                        "apk": apk(held_out, rec_pop, k),
                        "context_size": len(context),
                    }
                )

    if not records:
        print("No evaluation cases collected. Check rules mining thresholds or data.")
        return

    res = pd.DataFrame(records)
    summary = res.groupby("model").agg(hit_rate=("hit", "mean"), MAP_at_k=("apk", "mean")).reset_index()

    # Coverage: fraction of LOO cases where model produced candidates
    # (Note: same number of potential cases for both models)
    potential_cases = sum(max(len(b) - 1, 0) for b in test)  # leave-one-out per item
    cov_rules = total_cases_rules / potential_cases if potential_cases else 0.0
    cov_pop = total_cases_pop / potential_cases if potential_cases else 0.0

    print("\n=== Evaluation (leave-one-out on time split) ===")
    print(f"Train baskets: {len(train)} | Test baskets: {len(test)} | Potential LOO cases: {potential_cases}")
    print(f"Rules mined: {0 if rules_df is None or len(rules_df)==0 else len(rules_df)}")
    print("\nMetrics (k = %d):" % k)
    for _, row in summary.iterrows():
        model = row["model"]
        print(f"{model:10s} — HitRate@{k}: {row['hit_rate']:.3f} | MAP@{k}: {row['MAP_at_k']:.3f}")
    print(f"\nCoverage — rules: {cov_rules:.3f} | popularity: {cov_pop:.3f}")

    # Save detailed results
    res.to_csv(f"{out_dir}/eval_cases.csv", index=False)
    summary.to_csv(f"{out_dir}/eval_summary.csv", index=False)
    print(f"\nSaved: {out_dir}/eval_cases.csv and {out_dir}/eval_summary.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baskets_path", type=str, default="data/processed/baskets.csv")
    ap.add_argument("--min_support", type=float, default=0.01)
    ap.add_argument("--min_conf", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--allow_multi_ante", action="store_true", help="use multi-item antecedents if present")
    ap.add_argument("--out_dir", type=str, default="outputs/tables")
    args = ap.parse_args()

    main(
        baskets_path=args.baskets_path,
        min_support=args.min_support,
        min_conf=args.min_conf,
        k=args.k,
        allow_multi_ante=args.allow_multi_ante,
        out_dir=args.out_dir,
    )
