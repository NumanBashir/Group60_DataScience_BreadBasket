import pandas as pd
from ast import literal_eval
from collections import defaultdict


def load_rules(path="outputs/rules.csv", top_per_ante=50):

    def parse_set(x):
        try:
            val = literal_eval(x)
            if isinstance(val, str):
                val = (val,)
            elif isinstance(val, set):
                val = tuple(val)
            return val
        except Exception:
            return ()

    rules = pd.read_csv(path, converters={
        "antecedents": parse_set,
        "consequents": parse_set
    })

    rules["conseq"] = rules["consequents"].apply(
        lambda t: list(t)[0] if len(t) == 1 else None
    )
    rules = rules.dropna(subset=["conseq"])

    rules = rules.sort_values(["lift", "confidence"], ascending=False)

    table = defaultdict(list)
    for _, r in rules.iterrows():
        a = tuple(r["antecedents"])  # ✅ multi-item compatible
        score = 0.7 * r["lift"] + 0.3 * r["confidence"]
        table[a].append((r["conseq"], score))

    return {
        a: sorted(v, key=lambda x: -x[1])[:top_per_ante]
        for a, v in table.items()
    }, rules

# Recommend items with detailed info for logging/debugging
def recommend_verbose(basket, rules, k=5):
    recs = []

    for _, r in rules.iterrows():
        antecedent_items = list(r["antecedents"])
        if set(antecedent_items).issubset(set(basket)):
            c = list(r["consequents"])[0]
            if c in basket:
                continue

            lift, conf = r["lift"], r["confidence"]
            label = (
                "✅ Strongly Recommended" if lift > 1
                else "⚠️ Weakly Associated" if 0.9 <= lift <= 1
                else "🚫 Not Recommended"
            )

            recs.append({
                "antecedent": " + ".join(antecedent_items),
                "consequent": c,
                "confidence": round(conf, 3),
                "lift": round(lift, 3),
                "label": label
            })
    
    recs = sorted(recs, key=lambda x: -x["lift"])[:k]
    return recs


if __name__ == "__main__":
    rule_table, rules = load_rules("outputs/rules.csv")

    # Insert basket items here for testing -- Make sure they exist in the rules
    # For example: ["Cake", "Tea"]
    basket = ["Cake", "Tea"]
    recommendations = recommend_verbose(basket, rules)

    print(f"\nRecommendations for basket: {basket}")
    if len(recommendations) == 0:
        print("No matching rules for this basket.")
    else:
        for r in recommendations:
            print(
                f"{r['antecedent']} → {r['consequent']} | "
                f"Confidence: {r['confidence']}, Lift: {r['lift']} | {r['label']}"
            )

        if not any(r["lift"] > 1 for r in recommendations):
            print("\nNo strong recommendations for this basket.")

