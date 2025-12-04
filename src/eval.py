import argparse
import ast
from collections import Counter, defaultdict
from pathlib import Path
import random
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from keras import layers, models, optimizers
import warnings
import os

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# This method loads baskets from a Parquet or CSV file and returns a list of baskets
def load_baskets(path="data/processed/baskets.csv"):
    if path.endswith(".csv"):
        df = pd.read_csv(path, converters={"items": lambda x: ast.literal_eval(x)})
    else:
        df = pd.read_parquet(path)

    item_col = None
    for c in df.columns:
        if any(s in c.lower() for s in ["item", "items", "basket"]):
            item_col = c
            break

    if item_col is None:
        raise ValueError(f"⚠️ No items column found in {df.columns}")

    baskets = df[item_col].apply(lambda x: x if isinstance(x, list) else eval(x)).tolist()
    return baskets



# Mine association rules from the given baskets using the apriori algorithm
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

    rules["antecedents"] = rules["antecedents"].apply(lambda s: tuple(sorted(s)))
    rules["consequents"] = rules["consequents"].apply(lambda s: tuple(sorted(s)))
    rules = rules[rules["consequents"].apply(len) == 1].copy()
    if len(rules) == 0:
        return {}, pd.DataFrame()

    if not allow_multi_ante:
        rules = rules[rules["antecedents"].apply(len) == 1].copy()

    # Keep only positively associated rules
    rules = rules[rules["lift"] > 1].copy()
    if len(rules) == 0:
        return {}, pd.DataFrame()

    rules["conseq"] = rules["consequents"].apply(lambda t: t[0])
    rules = rules.sort_values(["lift", "confidence"], ascending=False)

    def score_row(r):
        return 0.7 * r["lift"] + 0.3 * r["confidence"]

    table = defaultdict(list)
    for _, r in rules.iterrows():
        a = r["antecedents"]
        s = score_row(r)
        table[a].append((r["conseq"], s))

    for a in list(table.keys()):
        table[a] = sorted(table[a], key=lambda x: -x[1])[:50]

    return dict(table), rules



# Recommend items based on the given context using the mined rules
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



# Compute the top-k popular items from the training baskets
def popularity_topk(train_baskets, k=10):
    cnt = Counter(i for b in train_baskets for i in set(b))
    return [i for i, _ in cnt.most_common(k)]



# Recommend items based on popularity, excluding context items
def recommend_pop(context_items, pop_list, k=10):
    context = set(context_items)
    recs = [i for i in pop_list if i not in context]
    return recs[:k]



# Compute HitRate@k
def hitrate_at_k(held_out, recs, k):
    return 1.0 if held_out in set(recs[:k]) else 0.0



# Compute Average Precision at k (APK)
def apk(held_out, recs, k):
    # Average precision at k for single held-out item simplifies to 1/rank if found
    for idx, item in enumerate(recs[:k], start=1):
        if item == held_out:
            return 1.0 / idx
    return 0.0



def extract_rule_features(context, candidate, rule_table, rule_df, pop_dict, time_dict=None):
    ctx = set(context)
    feats = {}

    matching = rule_df[rule_df["conseq"] == candidate]
    matching = matching[matching["antecedents"].apply(lambda a: set(a).issubset(ctx))]

    if len(matching) == 0:
        feats["max_lift"] = 0.0
        feats["max_conf"] = 0.0
        feats["max_support"] = 0.0
        feats["num_rules"] = 0.0
        feats["antecedent_size"] = 0.0
        feats["rule_score"] = 0.0
    else:
        feats["max_lift"] = float(matching["lift"].max())
        feats["max_conf"] = float(matching["confidence"].max())
        feats["max_support"] = float(matching["support"].max())
        feats["num_rules"] = float(len(matching))
        feats["antecedent_size"] = float(matching["antecedents"].apply(len).max())
        feats["rule_score"] = float(0.7 * matching["lift"].max() + 0.3 * matching["confidence"].max())

    feats["context_size"] = float(len(context))

    # === NEW TIME FEATURES ===
    if time_dict is not None:
        period, weekend = time_dict.get(candidate, ("unknown", "unknown"))

        feats["is_morning"] = 1.0 if period == "morning" else 0.0
        feats["is_afternoon"] = 1.0 if period == "afternoon" else 0.0
        feats["is_night"] = 1.0 if period == "night" else 0.0

        feats["is_weekend"] = 1.0 if weekend == "weekend" else 0.0
    else:
        feats["is_morning"] = 0.0
        feats["is_afternoon"] = 0.0
        feats["is_night"] = 0.0
        feats["is_weekend"] = 0.0

    return feats

def build_training_matrix(train, rule_table, rule_df, pop_dict, time_dict, num_neg=3):
    X = []
    y = []
    all_items = list(pop_dict.keys())

    for basket in train:
        if len(basket) < 2:
            continue

        for held_out in basket:
            context = [i for i in basket if i != held_out]

            feats = extract_rule_features(context, held_out, rule_table, rule_df, pop_dict, time_dict)
            feat_order = list(feats.keys())
            X.append([feats[f] for f in feat_order])
            y.append(1)

            # === OPTION A NEGATIVE SAMPLING (your request) ===
            for _ in range(num_neg):
                neg = random.choice(all_items)
                if neg == held_out or neg in context:
                    continue
                feats_neg = extract_rule_features(context, neg, rule_table, rule_df, pop_dict, time_dict)
                X.append([feats_neg[f] for f in feat_order])
                y.append(0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), feat_order

def build_rule_dense_model(num_features):
    inp = layers.Input(shape=(num_features,))
    x = layers.Dense(16, activation="relu")(inp)
    x = layers.Dense(8, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inp, out)
    opt = optimizers.Adam(learning_rate=0.00005)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def recommend_dense(context, model, rule_table, rule_df, pop_dict, feat_names, time_dict, k=10):
    candidates = [item for item in pop_dict.keys() if item not in context]
    if not candidates:
        return []

    X_rows = []
    for item in candidates:
        feats = extract_rule_features(context, item, rule_table, rule_df, pop_dict)
        X_rows.append([feats[f] for f in feat_names])

    X = np.array(X_rows, dtype=np.float32)
    scores = model.predict(X, verbose=0).flatten()

    scored_items = list(zip(candidates, scores))
    scored_items.sort(key=lambda x: -x[1])

    topk = [item for item, _ in scored_items[:k]]
    return topk

def load_time_features(raw_path="../data/raw/bread_basket.csv"):
    df = pd.read_csv(raw_path)

    df["period_day"] = df["period_day"].astype(str)
    df["weekday_weekend"] = df["weekday_weekend"].astype(str)

    mapping = {}
    for _, row in df.iterrows():
        item = row["Item"]
        mapping[item] = (row["period_day"], row["weekday_weekend"])

    return mapping

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
    time_dict = load_time_features()
    n = len(baskets)
    cut = int(0.8 * n)
    train, test = baskets[:cut], baskets[cut:]

    rule_table, rules_df = mine_rules_from_baskets(
        train, min_support=min_support, min_conf=min_conf, allow_multi_ante=allow_multi_ante
    )

    pop_list = popularity_topk(train, k * 3)
    pop_dict = Counter(i for b in train for i in set(b))

    X_train, y_train, feat_names = build_training_matrix(train, rule_table, rules_df, pop_dict, time_dict)

    model = build_rule_dense_model(X_train.shape[1])

    model.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )

    print("\n=== Example Dense Predictions ===")

    for _ in range(5):
        basket = random.choice(test)
        if len(basket) < 2:
            continue

        held_out = random.choice(basket)
        context = [x for x in basket if x != held_out]

        # Compute probability for ALL items
        scores = {}
        for item in pop_dict.keys():
            if item in context:
                continue
            feats = extract_rule_features(context, item, rule_table, rules_df, pop_dict)
            x = np.array([[feats[f] for f in feat_names]], dtype=np.float32)
            score = float(model.predict(x, verbose=0)[0][0])
            scores[item] = score

        top5 = sorted(scores.items(), key=lambda x: -x[1])[:5]

        print(f"\nContext: {context}")
        print(f"Held-out (true): {held_out}")
        print("Dense model top-5:")
        for item, sc in top5:
            print(f"  {item:20s}  score={sc:.4f}")

    print("\n=== Feature Importance (Dense Model) ===")
    weights = model.layers[1].get_weights()[0].flatten()
    feat_importance = sorted(zip(feat_names, weights), key=lambda x: -abs(x[1]))

    for name, w in feat_importance:
        print(f"{name:20s}  weight={w:.4f}")

    records = []
    total_cases_rules = 0
    total_cases_pop = 0

    # Print learned feature importance
    weights = model.layers[1].get_weights()[0].flatten()
    feat_importance = sorted(zip(feat_names, weights), key=lambda x: -abs(x[1]))

    print("\n=== Feature Importance (Dense Model) ===")
    for name, w in feat_importance:
        print(f"{name:20s}  weight = {w:.4f}")

    for b in test:
        if len(b) < 2:
            continue
        for held_out in b:
            context = [x for x in b if x != held_out]

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

            rec_dense = recommend_dense(context, model, rule_table, rules_df, pop_dict, feat_names, time_dict, k=k)
            records.append(
                {
                    "model": "dense",
                    "hit": hitrate_at_k(held_out, rec_dense, k),
                    "apk": apk(held_out, rec_dense, k),
                    "context_size": len(context),
                }
            )

    if not records:
        print("No evaluation cases collected. Check rules mining thresholds or data.")
        return

    res = pd.DataFrame(records)
    summary = res.groupby("model").agg(hit_rate=("hit", "mean"), MAP_at_k=("apk", "mean")).reset_index()

    potential_cases = sum(max(len(b) - 1, 0) for b in test)
    cov_rules = total_cases_rules / potential_cases if potential_cases else 0.0
    cov_pop = total_cases_pop / potential_cases if potential_cases else 0.0

    print("\n=== Evaluation (leave-one-out on time split) ===")
    print(f"Train baskets: {len(train)} | Test baskets: {len(test)} | Potential LOO cases: {potential_cases}")
    print(f"Rules mined: {0 if rules_df is None or len(rules_df)==0 else len(rules_df)}")
    print("\nMetrics (k = %d):" % k)
    for _, row in summary.iterrows():
        model_name = row["model"]
        print(f"{model_name:10s} — HitRate@{k}: {row['hit_rate']:.3f} | MAP@{k}: {row['MAP_at_k']:.3f}")
    print(f"\nCoverage — rules: {cov_rules:.3f} | popularity: {cov_pop:.3f}")

    res.to_csv(f"{out_dir}/eval_cases.csv", index=False)
    summary.to_csv(f"{out_dir}/eval_summary.csv", index=False)
    print(f"\nSaved: {out_dir}/eval_cases.csv and {out_dir}/eval_summary.csv")



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baskets_path", type=str, default="../data/processed/baskets.csv")
    ap.add_argument("--min_support", type=float, default=0.01)
    ap.add_argument("--min_conf", type=float, default=0.2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--allow_multi_ante", action="store_true", help="use multi-item antecedents if present")
    ap.add_argument("--out_dir", type=str, default="../outputs/tables")
    args = ap.parse_args()

    main(
        baskets_path=args.baskets_path,
        min_support=args.min_support,
        min_conf=args.min_conf,
        k=args.k,
        allow_multi_ante=args.allow_multi_ante,
        out_dir=args.out_dir,
    )
