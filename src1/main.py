# -*- coding: utf-8 -*-
"""
BreadBasket analytics pipeline (report-ready, prints & plots only):
- LOAD & CLEAN
- DESCRIPTIVE STATS (item frequencies, basket sizes)
- W5: Frequent itemsets & Association rules (+ weekday/weekend, morning splits)
- W4: Similar items (MinHash + LSH) tuned to B=64, r=2 (t≈0.125)
- W6: Clustering (PPMI+TFIDF -> PCA(95%) -> k-means / spherical / spectral / DBSCAN)
- W7: Graph communities (Lift graph) [original + filtered/strong]
- ML1: Add-on propensity models (one-vs-all per popular item) [leakage-safe context]
- ML2: Next-item recommendation (time split, train-only stats) [leakage-safe]

Plots -> ../data/plots/
(NO CSV/TXT/GEXF WRITES)

Author: you ✨
"""

import os, warnings, random
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML / Stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import (
    davies_bouldin_score, silhouette_score,
    roc_auc_score, average_precision_score,
    normalized_mutual_info_score
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

# Assoc rules
from mlxtend.frequent_patterns import apriori, association_rules

# Graph
import networkx as nx
try:
    from networkx.algorithms.community import greedy_modularity_communities
except Exception:
    greedy_modularity_communities = None

# Louvain (optional)
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except Exception:
    HAS_LOUVAIN = False

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(ROOT, "..", "data"))
RAW  = os.path.join(DATA_DIR, "raw")
PLOTS = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

# -------------------------
# Plot utils
# -------------------------

def figsav(path, tight=True):
    if tight:
        plt.tight_layout()
    plt.savefig(path, dpi=140)
    print(f"[PLOT] {os.path.relpath(path, DATA_DIR)}")
    plt.close()

def figpath(fname):
    return os.path.join(PLOTS, fname)

# -------------------------
# Load + basic utils
# -------------------------

def load_breadbasket():
    candidates = [
        os.path.join(RAW, "bread_basket.csv"),
        os.path.join(RAW, "BreadBasket_DMS.csv"),
        os.path.join(RAW, "BreadBasket.csv"),
        os.path.join(RAW, "breadbasket.csv"),
    ]
    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p
            break
    if path is None:
        raise FileNotFoundError(f"Could not find dataset in {RAW}.")
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "date_time" not in df.columns:
        if "date" in df.columns and "time" in df.columns:
            df["date_time"] = pd.to_datetime(df["date"] + " " + df["time"])
        else:
            for c in df.columns:
                if "date" in c or "time" in c:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            df["date_time"] = pd.to_datetime(df[c])
                            break
                        except Exception:
                            pass
    if "date_time" not in df.columns:
        raise ValueError("Could not infer date_time column.")

    df = df.dropna(subset=["item"])
    df = df[df["item"].astype(str).str.upper() != "NONE"]
    df["item"] = df["item"].astype(str).str.strip().str.title()

    df["date_time"] = pd.to_datetime(df["date_time"])
    hour = df["date_time"].dt.hour
    df["period_day"] = np.where(hour < 12, "morning", "not_morning")
    df["weekday_weekend"] = np.where(df["date_time"].dt.dayofweek < 5, "weekday", "weekend")

    if "transaction" not in df.columns:
        df = df.sort_values("date_time").copy()
        df["transaction"] = (df.groupby(df["date_time"].dt.floor("min")).cumcount() + 1).astype(int)

    return df

def basket_summary_and_plots(df):
    print(f"Loaded dataset: {len(df):,} rows, {df['item'].nunique()} unique items")

    # Basket size stats
    tx_counts = df.groupby("transaction")["item"].nunique()
    desc = tx_counts.describe()
    print("\nBasket size stats:\n", desc.to_string())

    # Plot: histogram of basket sizes
    plt.figure(figsize=(6.8,3.6))
    tx_counts.plot(kind="hist", bins=10)
    plt.xlabel("Unique items per basket")
    plt.ylabel("Frequency")
    plt.title("Basket size distribution")
    figsav(figpath("D0_basket_size_hist.png"))

    # Item frequencies (rows) — print top 20
    top_rows_df = df["item"].value_counts().reset_index()
    top_rows_df.columns = ["item", "rows"]
    print("\nTop-20 items by rows:\n", top_rows_df.head(20).to_string(index=False))

    # Plot: top-20 items by rows
    plt.figure(figsize=(8.5,4.2))
    (top_rows_df.head(20)
        .sort_values("rows")
        .set_index("item")["rows"]
        .plot(kind="barh"))
    plt.xlabel("Row count")
    plt.title("Top-20 items by rows")
    figsav(figpath("D0_top20_items_rows.png"))

    # Basket support (unique transactions per item) — print top 20
    top_baskets_df = (
        df.groupby("item")["transaction"]
          .nunique()
          .sort_values(ascending=False)
          .reset_index()
    )
    top_baskets_df.columns = ["item", "baskets"]
    print("\nTop-20 items by basket support:\n", top_baskets_df.head(20).to_string(index=False))

    # Plot: top-20 items by basket support
    plt.figure(figsize=(8.5,4.2))
    (top_baskets_df.head(20)
        .sort_values("baskets")
        .set_index("item")["baskets"]
        .plot(kind="barh"))
    plt.xlabel("Unique baskets")
    plt.title("Top-20 items by basket support")
    figsav(figpath("D0_top20_items_baskets.png"))

    print("\nPreview:\n", df[["transaction","item","date_time","period_day","weekday_weekend"]].head(5).to_string(index=False))

def baskets_ohe(df):
    items = sorted(df["item"].unique())
    item_to_idx = {it:i for i,it in enumerate(items)}
    tx_ids = sorted(df["transaction"].unique())
    tx_to_idx = {t:i for i,t in enumerate(tx_ids)}

    mat = np.zeros((len(tx_ids), len(items)), dtype=np.uint8)
    for t, grp in df.groupby("transaction"):
        i = tx_to_idx[t]
        for it in grp["item"].unique():
            mat[i, item_to_idx[it]] = 1
    X = pd.DataFrame(mat, index=tx_ids, columns=items)
    return X

def split_weekday_weekend(df):
    return df[df["weekday_weekend"]=="weekday"], df[df["weekday_weekend"]=="weekend"]

def split_morning(df):
    return df[df["period_day"]=="morning"], df[df["period_day"]!="morning"]

# -------------------------
# Category mapping
# -------------------------

def map_category(item: str) -> str:
    it = str(item).lower()
    if any(k in it for k in ["coffee", "tea", "hot chocolate", "coke", "juice", "smoothie", "mineral water", "water", "drinking chocolate"]):
        return "beverage"
    if any(k in it for k in ["bread", "toast", "farm house", "baguette", "focaccia", "scandinavian"]):
        return "bread"
    if any(k in it for k in ["cake", "brownie", "muffin", "cookie", "cookies", "alfajores", "truffles", "scone", "pastry", "bakewell", "tiffin", "fudge", "lemon and coconut", "victorian sponge", "dulce de leche", "jammie dodgers", "panatone"]):
        return "sweet"
    if any(k in it for k in ["sandwich", "soup", "empanadas", "muesli", "granola", "salad", "bowl", "spanish brunch", "chicken", "bacon", "eggs", "frittata", "tartine", "pintxos", "polenta"]):
        return "savory"
    return "other"

# -------------------------
# Context features (leakage-safe)
# -------------------------

def build_context_features_masked(df: pd.DataFrame, target: str) -> pd.DataFrame:
    d = df[df["item"] != target].copy()
    meta = df.drop_duplicates("transaction").set_index("transaction")
    hour  = meta["date_time"].dt.hour
    dow   = meta["date_time"].dt.dayofweek
    month = meta["date_time"].dt.month

    hour_sin = np.sin(2*np.pi*hour/24).rename("hour_sin")
    hour_cos = np.cos(2*np.pi*hour/24).rename("hour_cos")
    dow_sin  = np.sin(2*np.pi*dow/7).rename("dow_sin")
    dow_cos  = np.cos(2*np.pi*dow/7).rename("dow_cos")
    mon_sin  = np.sin(2*np.pi*month/12).rename("month_sin")
    mon_cos  = np.cos(2*np.pi*month/12).rename("month_cos")

    is_morning = (meta["period_day"] == "morning").astype(int).rename("is_morning")
    is_weekend = (meta["weekday_weekend"] == "weekend").astype(int).rename("is_weekend")

    basket_size = d.groupby("transaction")["item"].nunique().rename("basket_size")

    d["category"] = d["item"].map(map_category)
    cat_counts = d.pivot_table(index="transaction", columns="category", values="item",
                               aggfunc="count", fill_value=0)
    cat_counts = cat_counts.add_prefix("cnt_cat_")

    feats = pd.concat(
        [basket_size, is_morning, is_weekend,
         hour_sin, hour_cos, dow_sin, dow_cos, mon_sin, mon_cos,
         cat_counts],
        axis=1
    ).fillna(0)

    for c in cat_counts.columns:
        feats[c.replace("cnt_", "has_")] = (feats[c] > 0).astype(int)

    feats = feats.reindex(sorted(feats.columns), axis=1)
    return feats.astype(np.float32)

# -------------------------
# W5: Frequent itemsets & rules
# -------------------------

def frequent_and_rules(X, min_support=0.01):
    X_bool = X.astype(bool)
    freq = apriori(X_bool, min_support=min_support, use_colnames=True)
    freq["k"] = freq["itemsets"].apply(len)
    freq["count"] = (freq["support"] * len(X_bool)).round().astype(int)
    rules = association_rules(freq, metric="lift", min_threshold=1.0)

    def tup_to_str(t):
        return ", ".join(sorted(list(t)))
    out = rules.copy()
    out["antecedent"] = out["antecedents"].apply(tup_to_str)
    out["consequent"]  = out["consequents"].apply(tup_to_str)
    sel = out[["antecedent","consequent","support","confidence","lift"]].sort_values(
        ["lift","confidence","support"], ascending=False
    ).reset_index(drop=True)
    return freq, sel

def plot_top_rules(rules_df, title, fname, top=10):
    topdf = rules_df.head(top).copy()
    if topdf.empty:
        return
    plt.figure(figsize=(7.8,4.2))
    y = [f"{a} → {b}" for a,b in zip(topdf["antecedent"], topdf["consequent"])]
    plt.barh(range(len(topdf)), topdf["lift"].values)
    plt.yticks(range(len(topdf)), y)
    plt.xlabel("Lift")
    plt.title(title)
    plt.gca().invert_yaxis()
    figsav(figpath(fname))

# -------------------------
# W4: MinHash + LSH
# -------------------------

def jaccard_sets(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))

def minhash_signatures(item_txsets, n_hashes=128, seed=42):
    random.seed(seed)
    tx_universe = sorted(set().union(*item_txsets.values()))
    tx_to_idx = {t:i for i,t in enumerate(tx_universe)}

    P = 2_147_483_647
    rng = np.random.default_rng(seed)
    A = rng.integers(1, P-1, size=n_hashes, dtype=np.int64)
    B = rng.integers(0, P-1, size=n_hashes, dtype=np.int64)

    sigs = {}
    for it, txs in item_txsets.items():
        idxs = np.array([tx_to_idx[t] for t in txs], dtype=np.int64)
        if idxs.size == 0:
            sigs[it] = np.full(n_hashes, P, dtype=np.int64)
            continue
        vals = (A[:,None]*idxs[None,:] + B[:,None]) % P
        sigs[it] = vals.min(axis=1)
    return sigs

def lsh_candidates(sigs, bands=64, rows_per_band=2):
    example = next(iter(sigs.values()))
    n_hashes = example.shape[0]
    assert bands * rows_per_band == n_hashes, "BANDS * ROWS_PER_BAND must equal N_HASHES"

    buckets = [defaultdict(list) for _ in range(bands)]
    items = list(sigs.keys())
    for it in items:
        sig = sigs[it]
        for b in range(bands):
            start = b*rows_per_band
            end = start + rows_per_band
            key = tuple(sig[start:end].tolist())
            buckets[b][key].append(it)

    cand = defaultdict(set)
    for b in range(bands):
        for _, bucket_items in buckets[b].items():
            if len(bucket_items) < 2:
                continue
            for i in range(len(bucket_items)):
                for j in range(i+1, len(bucket_items)):
                    a, c = bucket_items[i], bucket_items[j]
                    cand[a].add(c)
                    cand[c].add(a)
    return cand

def lsh_eval_versions(item_txsets,
                      focus=("Coffee","Bread","Tea","Cake","Pastry"),
                      n_hashes=128, vA_bands=64, vA_r=2):
    print("\nW4 — SIMILAR ITEMS (MinHash + LSH, tuned)")
    print("=========================================")
    print("[LSH] Computing signatures...")
    sigs = minhash_signatures(item_txsets, n_hashes=n_hashes, seed=RANDOM_STATE)
    cand = lsh_candidates(sigs, bands=vA_bands, rows_per_band=vA_r)
    approx_t = (1/vA_bands)**(1/vA_r)
    items_with_cand = sum(len(v)>0 for v in cand.values())
    print(f"[LSH] Candidate neighbor sets for {items_with_cand} items.")
    print(f"[LSH] Approx collision threshold t ≈ (1/B)^1/r = {approx_t:.3f}\n")

    all_items = list(item_txsets.keys())
    def exact_top(it, K=10):
        scores = []
        for jt in all_items:
            if jt == it:
                continue
            scores.append((jt, jaccard_sets(item_txsets[it], item_txsets[jt])))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:K]

    rows = []
    bars = []
    for it in focus:
        gold_pairs = exact_top(it, K=10)
        gold = [g for g,_ in gold_pairs]
        cand_list = list(cand.get(it, set()))
        scored = [(jt, jaccard_sets(item_txsets[it], item_txsets[jt])) for jt in cand_list]
        pred = [p for p,_ in sorted(scored, key=lambda x: x[1], reverse=True)[:10]]
        inter = len(set(gold) & set(pred)) if gold else 0
        p10 = inter / max(1, len(pred)) if pred else 0.0
        r10 = inter / len(gold) if gold else 0.0
        rows.append([it, p10, r10, len(cand_list),
                     ", ".join([a for a,_ in gold_pairs]),
                     ", ".join(pred)])
        bars.append((it, p10, r10))

    if rows:
        df_eval = pd.DataFrame(rows, columns=["item","P@10","R@10","candidates","ExactTop10","LSH_Top10"])
        print(df_eval.to_string(index=False))

    # Plots: precision and recall
    if bars:
        items = [b[0] for b in bars]
        precs = [b[1] for b in bars]
        recs  = [b[2] for b in bars]

        plt.figure(figsize=(6.8,3.4))
        plt.bar(range(len(items)), recs)
        plt.xticks(range(len(items)), items)
        plt.ylabel("Recall@10")
        plt.title("LSH recall (tuned 64x2)")
        figsav(figpath("W4_LSH_recall_tuned.png"))

        plt.figure(figsize=(6.8,3.4))
        plt.bar(range(len(items)), precs)
        plt.xticks(range(len(items)), items)
        plt.ylabel("Precision@10")
        plt.title("LSH precision (tuned 64x2)")
        figsav(figpath("W4_LSH_precision_tuned.png"))

# -------------------------
# Co-occurrence / Lift
# -------------------------

def cooc_and_lift(X):
    item_names = list(X.columns)
    nB = len(X)
    supp = X.sum(axis=0).astype(int)
    counts = X.T.dot(X).astype(int).values
    counts = np.triu(counts,1) + np.tril(counts,-1) + np.diag(np.diag(counts))
    P_i = supp.values / nB
    P_ij = counts / nB
    denom = (P_i[:,None] * P_i[None,:]) + 1e-12
    lift = P_ij / denom
    np.fill_diagonal(lift, 0.0)
    return item_names, counts, lift, supp

# -------------------------
# W6: Clustering
# -------------------------

def build_ppmi(X):
    B = X.values
    nB = B.shape[0]
    f_i = B.sum(axis=0)
    C = (B.T @ B).astype(float)
    np.fill_diagonal(C, 0.0)
    p_i = f_i / nB
    p_ij = C / nB
    denom = (p_i[:,None] * p_i[None,:]) + 1e-12
    PMI = np.log((p_ij + 1e-12) / denom)
    PPMI = np.maximum(PMI, 0.0)
    return PPMI

def kmeans_sweep(Z, item_names, metric="euclidean", k_grid=None, label="Euclidean k-means"):
    if k_grid is None:
        k_grid = [3,4,5,8,9,10,11,12,13,14]
    results = []
    for k in k_grid:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init="auto")
        labels = km.fit_predict(Z)
        try:
            dbi = davies_bouldin_score(Z, labels)
        except Exception:
            dbi = np.nan
        try:
            sil = silhouette_score(Z, labels, metric="euclidean")
        except Exception:
            sil = np.nan
        results.append((k, dbi, sil, km, labels))
    results_sorted = sorted(results, key=lambda x: (np.isnan(x[1]), x[1], -(x[2] if not np.isnan(x[2]) else -1)))
    print(f"\n[{label}] — top 5 by DBI/Sil:")
    for (k,dbi,sil,_,_) in results_sorted[:5]:
        print(f"  k={k:2d}  DBI={dbi:.3f}  Sil={sil:.3f}")

    # Plot silhouette for top-5 k
    plt.figure(figsize=(6.2,3.1))
    plt.plot([k for (k,_,_,_,_) in results_sorted[:5]], [sil for (_,_,sil,_,_) in results_sorted[:5]], marker='o')
    plt.xlabel('k'); plt.ylabel('Silhouette'); plt.title(f'{label}: top-5 Silhouette')
    figsav(figpath(f"W6_{label.replace(' ','_')}_silhouette.png"))

    # Best clustering listing
    k, dbi, sil, km, labels = results_sorted[0]
    clusters = defaultdict(list)
    for name, lab in zip(item_names, labels):
        clusters[lab].append(name)
    print(f"[{label}] Best: k={k}  DBI={dbi:.3f}  Sil={sil:.3f}")
    for cid in sorted(clusters.keys()):
        members = ", ".join(sorted(clusters[cid]))
        print(f"  Cluster {cid} (size={len(clusters[cid])}): {members}")
    # Plot cluster sizes
    plt.figure(figsize=(6.2,3.1))
    sizes = [len(clusters[c]) for c in sorted(clusters.keys())]
    plt.bar(range(len(sizes)), sizes)
    plt.title(f"{label} best (k={k}) cluster sizes")
    plt.xlabel("Cluster id"); plt.ylabel("Size")
    figsav(figpath(f"W6_{label.replace(' ','_')}_cluster_sizes.png"))
    return km, labels

def spherical_kmeans(Z, item_names, k_grid=None):
    Z_norm = normalize(Z, norm="l2", axis=1)
    return kmeans_sweep(Z_norm, item_names, metric="cosine", k_grid=k_grid, label="Spherical k-means")

def spectral_try(affinity, item_names):
    try:
        from sklearn.cluster import SpectralClustering
    except Exception:
        print("[Spectral] Not available; skipping.")
        return None
    k_grid = [3,4,5,6,7]
    best = None
    for k in k_grid:
        try:
            sc = SpectralClustering(n_clusters=k, affinity='precomputed', assign_labels='kmeans',
                                    random_state=RANDOM_STATE)
            labels = sc.fit_predict(affinity)
            dbi = davies_bouldin_score(affinity, labels) if affinity.shape[0] > k else np.nan
            sil = silhouette_score(affinity, labels, metric="cosine") if affinity.shape[0] > k else np.nan
            if (best is None) or (dbi < best[0]):
                best = (dbi, sil, k, labels)
        except Exception:
            continue
    if best is None:
        print("[Spectral] Skipped (no fit).")
        return None
    dbi, sil, k, labels = best
    clusters = defaultdict(list)
    for name, lab in zip(item_names, labels):
        clusters[lab].append(name)
    print(f"[Spectral] Best: k={k}  DBI={dbi:.3f}  Sil(cos)={sil:.3f}")
    for cid in sorted(clusters.keys()):
        members = ", ".join(sorted(clusters[cid]))
        print(f"  Cluster {cid} (size={len(clusters[cid])}): {members}")
    # Plot sizes
    plt.figure(figsize=(6.2,3.1))
    sizes = [len(clusters[c]) for c in sorted(clusters.keys())]
    plt.bar(range(len(sizes)), sizes)
    plt.title(f"Spectral clustering (k={k}) cluster sizes")
    plt.xlabel("Cluster id"); plt.ylabel("Size")
    figsav(figpath("W6_Spectral_cluster_sizes.png"))
    return labels

# -------------------------
# W7: Graph communities
# -------------------------

def graph_from_lift(item_names, lift, supp, nB, min_lift=1.3, min_count=40):
    G = nx.Graph()
    for it in item_names:
        G.add_node(it)
    n = len(item_names)
    for i in range(n):
        for j in range(i+1,n):
            c_ij = int(round(lift[i,j] * (supp.values[i]/nB) * (supp.values[j]/nB) * nB))
            if lift[i,j] >= min_lift or c_ij >= min_count:
                G.add_edge(item_names[i], item_names[j], weight=float(lift[i,j]))
    return G

def summarize_communities(G, tag):
    print(f"\nW7 — GRAPH COMMUNITIES ({tag})")
    print("====================================")
    # Greedy
    if greedy_modularity_communities is not None:
        comms = list(greedy_modularity_communities(G, weight='weight'))
        mod = nx.algorithms.community.quality.modularity(G, comms, weight='weight')
        print(f"Greedy modularity: {mod:.3f} | #communities={len(comms)} | edges={G.number_of_edges()}")
        # bar of sizes
        plt.figure(figsize=(6.2,3.1))
        sizes = sorted([len(c) for c in comms], reverse=True)
        plt.bar(range(len(sizes)), sizes)
        plt.title(f"Greedy communities: sizes ({tag})")
        plt.xlabel("Community id (sorted)"); plt.ylabel("Size")
        figsav(figpath(f"W7_{tag}_greedy_sizes.png"))
    else:
        print("Greedy modularity not available; upgrade networkx.")

    # Louvain
    if HAS_LOUVAIN:
        part = community_louvain.best_partition(G, weight='weight', random_state=RANDOM_STATE, resolution=1.0)
        groups = defaultdict(list)
        for node, cid in part.items():
            groups[cid].append(node)
        mod = community_louvain.modularity(part, G, weight='weight')
        print(f"Louvain modularity: {mod:.3f} | #communities={len(groups)}")
        plt.figure(figsize=(6.2,3.1))
        sizes = sorted([len(v) for v in groups.values()], reverse=True)
        plt.bar(range(len(sizes)), sizes)
        plt.title(f"Louvain communities: sizes ({tag})")
        plt.xlabel("Community id (sorted)"); plt.ylabel("Size")
        figsav(figpath(f"W7_{tag}_louvain_sizes.png"))

        # small resolution sweep — print only (no files)
        base_part = part
        nodes = sorted(G.nodes())
        base_labels = np.array([base_part[n] for n in nodes])
        print("\n[Louvain resolution sweep]")
        for gamma in [0.50, 0.80, 1.00, 1.20, 1.50]:
            part_g = community_louvain.best_partition(G, weight='weight', random_state=RANDOM_STATE, resolution=gamma)
            groups_g = defaultdict(list)
            for node, cid in part_g.items():
                groups_g[cid].append(node)
            mod_g = community_louvain.modularity(part_g, G, weight='weight')
            y2 = np.array([part_g[n] for n in nodes])
            try:
                nmi = normalized_mutual_info_score(base_labels, y2)
            except Exception:
                nmi = np.nan
            print(f"  γ={gamma:.2f}: modularity={mod_g:.3f} | #communities={len(groups_g):2d} | NMI_vs_γ=1.0={nmi:.3f}")
    else:
        print("Louvain not installed (pip install python-louvain).")

# -------------------------
# ML1: Propensity models
# -------------------------

def ml1_propensity(X, df, top_k_targets=10):
    item_names = list(X.columns)
    item_freq = X.sum(axis=0).sort_values(ascending=False)
    targets = list(item_freq.head(top_k_targets).index)

    rows = []
    print("\nML1 — Propensity models for add-ons")
    print("====================================")
    print(f"{'target':>12}      AUC       AP")
    aucs = []; aps = []; labels = []
    for tgt in targets:
        y = X[tgt].values.astype(int)
        cols = [c for c in item_names if c != tgt]
        Zt = X[cols].values.astype(np.uint8)
        CTX_t = build_context_features_masked(df, tgt).reindex(index=X.index).fillna(0).astype(np.float32)
        ctx_mat = CTX_t.values

        idx = np.arange(len(y))
        idx_tr, idx_te = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
        Xtr = np.hstack([Zt[idx_tr], ctx_mat[idx_tr]])
        Xte = np.hstack([Zt[idx_te], ctx_mat[idx_te]])
        ytr, yte = y[idx_tr], y[idx_te]
        if ytr.sum() == 0 or yte.sum() == 0:
            auc = np.nan; ap = np.nan
        else:
            clf = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
            clf.fit(Xtr, ytr)
            p = clf.predict_proba(Xte)[:,1]
            auc = roc_auc_score(yte, p)
            ap = average_precision_score(yte, p)
        rows.append((tgt, auc, ap))
        labels.append(tgt); aucs.append(0.0 if pd.isna(auc) else float(auc)); aps.append(0.0 if pd.isna(ap) else float(ap))
        print(f"{tgt:>12} {auc:8.6f} {ap:8.6f}")

    # Plot: AUC + AP bars
    if labels:
        x = np.arange(len(labels))
        plt.figure(figsize=(8.4,4.0))
        plt.bar(x-0.2, aucs, width=0.4, label="AUC")
        plt.bar(x+0.2, aps,  width=0.4, label="AP")
        plt.xticks(x, labels, rotation=30, ha="right")
        plt.ylabel("Score")
        plt.title("ML1: Propensity models performance")
        plt.legend(frameon=False)
        figsav(figpath("ML1_performance.png"))

    # Demo: Basket={'Coffee','Tea'}
    demo_ctx = {"Coffee","Tea"}
    suggestions = []
    for tgt in targets:
        if tgt in demo_ctx:
            continue
        cols = [c for c in item_names if c != tgt]
        Zt = X[cols].values.astype(np.uint8)
        y = X[tgt].values.astype(int)
        CTX_t = build_context_features_masked(df, tgt).reindex(index=X.index).fillna(0).astype(np.float32)
        Xt = np.hstack([Zt, CTX_t.values])
        clf = LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")
        clf.fit(Xt, y)
        ctx_neutral = CTX_t.mean(axis=0).values.reshape(1, -1).astype(np.float32)
        ctx_proj = np.zeros((1, len(cols)), dtype=np.uint8)
        for j, it in enumerate(cols):
            if it in demo_ctx:
                ctx_proj[0, j] = 1
        x_demo = np.hstack([ctx_proj, ctx_neutral])
        prob = clf.predict_proba(x_demo)[:, 1].item()
        suggestions.append((tgt, prob))

    suggestions = sorted(suggestions, key=lambda x: -x[1])[:5]
    if suggestions:
        print("\n[Propensity demo] Basket={'Coffee','Tea'} → top add-ons:")
        for it, p in suggestions:
            print(f"  {it:15s} {float(p):.3f}")
        # Plot demo
        items = [a for a,_ in suggestions]
        probs = [float(b) for _,b in suggestions]
        plt.figure(figsize=(6.8,3.4))
        plt.barh(range(len(items)), probs)
        plt.yticks(range(len(items)), items)
        plt.xlabel("Predicted add-on probability")
        plt.title("Demo: Basket={'Coffee','Tea'} → top add-ons")
        plt.gca().invert_yaxis()
        figsav(figpath("ML1_demo_addons_plot.png"))

# -------------------------
# ML2: Next-item recommendation (leakage-safe)
# -------------------------

def tx_sequences(df):
    seqs = []
    for tx, grp in df.sort_values(["transaction","date_time"]).groupby("transaction"):
        items = list(dict.fromkeys(grp["item"].tolist()))
        if len(items) >= 2:
            seqs.append(items)
    return seqs

def markov_from_sequences(seqs, items=None):
    trans = defaultdict(Counter)
    unig = Counter()
    for items_seq in seqs:
        for i in range(len(items_seq)-1):
            a, b = items_seq[i], items_seq[i+1]
            trans[a][b] += 1
            unig[a] += 1
    P = {}
    for a, cnts in trans.items():
        tot = sum(cnts.values())
        if tot == 0:
            continue
        P[a] = {b: cnt/tot for b, cnt in cnts.items()}
    return trans, unig, P

def ml2_next_item_leak_safe(df, X):
    print("\nML2 — Next-item recommendation (Markov + ML re-rank)")
    print("=====================================================")
    df_sorted = df.sort_values("date_time")
    cut_idx = int(0.8 * len(df_sorted))
    t_cut = df_sorted.iloc[cut_idx]["date_time"]
    df_train = df[df["date_time"] <= t_cut].copy()
    df_test  = df[df["date_time"] >  t_cut].copy()

    seq_train = tx_sequences(df_train)
    seq_test  = tx_sequences(df_test)

    trans, unig, P = markov_from_sequences(seqs=seq_train)
    X_train = baskets_ohe(df_train)
    X_train = X_train.reindex(columns=X.columns, fill_value=0)
    item_names = list(X.columns)

    _, counts_tr, lift_tr, supp_tr = cooc_and_lift(X_train)
    item_to_idx = {it:i for i,it in enumerate(item_names)}

    Xpairs, ypairs = [], []
    for seq in seq_train:
        for i in range(len(seq)-1):
            a, b = seq[i], seq[i+1]
            Xpairs.append((a,b)); ypairs.append(1)
            for _ in range(3):
                c = random.choice(item_names)
                if c != b:
                    Xpairs.append((a,c)); ypairs.append(0)

    def pair_features(a, b):
        ia, ib = item_to_idx.get(a, None), item_to_idx.get(b, None)
        if ia is None or ib is None:
            return np.array([0,0,0,0], dtype=float)
        lift_ab = float(lift_tr[ia, ib])
        cooc_ab = float(counts_tr[ia, ib])
        pop_b = float(supp_tr.values[ib])
        p_ab = float(P.get(a, {}).get(b, 0.0))
        return np.array([lift_ab, cooc_ab, pop_b, p_ab], dtype=float)

    if sum(ypairs) == 0:
        print("[ML2] No positive pairs — skipping.")
        return

    Xf = np.vstack([pair_features(a,b) for (a,b) in Xpairs])
    yf = np.array(ypairs, dtype=int)
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=RANDOM_STATE)
    clf.fit(Xf, yf)

    test_pairs, y_true = [], []
    for seq in seq_test:
        for i in range(len(seq)-1):
            a, b = seq[i], seq[i+1]
            test_pairs.append((a,b)); y_true.append(1)
            neg_pool = [c for c in item_names if c != b]
            for _ in range(3):
                c = random.choice(neg_pool)
                test_pairs.append((a,c)); y_true.append(0)

    Xtest_feat = np.vstack([pair_features(a,b) for (a,b) in test_pairs])
    y_true = np.array(y_true, dtype=int)
    pscore = clf.predict_proba(Xtest_feat)[:,1]
    ap = average_precision_score(y_true, pscore)
    print(f"[ML2] Next-item Average Precision: {float(ap):.3f}")

    # Demo ranking for last='Coffee'
    last = "Coffee" if "Coffee" in item_names else item_names[0]
    cand = list(P.get(last, {}).keys()) or [it for it in item_names if it != last][:10]
    scored = []
    for b in cand:
        x = pair_features(last, b).reshape(1,-1)
        prob = clf.predict_proba(x)[:,1].item()
        scored.append((b, prob))
    scored = sorted(scored, key=lambda x: -x[1])[:6]
    print(f"\n[Next-item demo] last='{last}' →")
    for it, sc in scored:
        print(f"  {it:15s} {float(sc):.3f}")

    # Plot demo
    if scored:
        items = [a for a,_ in scored]
        probs = [float(b) for _,b in scored]
        plt.figure(figsize=(6.8,3.4))
        plt.barh(range(len(items)), probs)
        plt.yticks(range(len(items)), items)
        plt.xlabel("Predicted relevance")
        plt.title(f"Next-item demo from '{last}'")
        plt.gca().invert_yaxis()
        figsav(figpath("ML2_demo_next_from_coffee.png"))

# -------------------------
# MAIN
# -------------------------

def main():
    print("\nLOAD & CLEAN\n============")
    df = load_breadbasket()
    basket_summary_and_plots(df)

    X = baskets_ohe(df)

    # ----------------- W5 -----------------
    print("\nW5 — FREQUENT ITEMSETS & ASSOCIATION RULES\n==========================================")
    min_support = 0.01
    freq, rules = frequent_and_rules(X, min_support=min_support)
    min_count = int(round(min_support * X.shape[0]))
    k1 = (freq["k"]==1).sum(); k2 = (freq["k"]==2).sum(); k3 = (freq["k"]==3).sum()
    print(f"Using min_support={min_support:.2%} => min_count={min_count} of {X.shape[0]} baskets")
    print(f"Frequent 1-itemsets: {k1} | 2-itemsets: {k2} | 3-itemsets: {k3}\n")

    # Print top rules table (all data)
    print("Top rules (by lift):")
    print(rules.head(12).to_string(index=False, float_format=lambda v: f"{float(v):.6f}"))
    # Plots for W5
    plot_top_rules(rules, "Top rules (all data, by lift)", "W5_top_rules_all.png", top=12)

    # Weekday/Weekend splits
    df_wk, df_we = split_weekday_weekend(df)
    X_wk, X_we = baskets_ohe(df_wk), baskets_ohe(df_we)
    _, r_wk = frequent_and_rules(X_wk, min_support=min_support)
    _, r_we = frequent_and_rules(X_we, min_support=min_support)
    print("\nWEEKDAY — Top rules:\n", r_wk.head(12).to_string(index=False, float_format=lambda v: f"{float(v):.6f}"))
    print("\nWEEKEND — Top rules:\n", r_we.head(12).to_string(index=False, float_format=lambda v: f"{float(v):.6f}"))
    plot_top_rules(r_wk, "Weekday: top rules (by lift)", "W5_top_rules_weekday.png", top=12)
    plot_top_rules(r_we, "Weekend: top rules (by lift)", "W5_top_rules_weekend.png", top=12)

    # Morning split
    df_morn, df_notm = split_morning(df)
    X_m, X_nm = baskets_ohe(df_morn), baskets_ohe(df_notm)
    _, r_m = frequent_and_rules(X_m, min_support=min_support)
    _, r_nm = frequent_and_rules(X_nm, min_support=min_support)
    print("\nMORNING — Top rules:\n", r_m.head(12).to_string(index=False, float_format=lambda v: f"{float(v):.6f}"))
    print("\nNOT MORNING — Top rules:\n", r_nm.head(12).to_string(index=False, float_format=lambda v: f"{float(v):.6f}"))
    plot_top_rules(r_m, "Morning: top rules", "W5_top_rules_morning.png", top=12)
    plot_top_rules(r_nm, "Not morning: top rules", "W5_top_rules_not_morning.png", top=12)

    # ----------------- W4: LSH -----------------
    item_txsets = {it: set(df.loc[df["item"]==it, "transaction"].unique()) for it in df["item"].unique()}
    lsh_eval_versions(item_txsets)

    # ----------------- W6: Clustering -----------------
    print("\nW6 — CLUSTERING (PPMI + TF-IDF → PCA(95%) → variants)")
    print("====================================================")
    top_keep = 22
    item_freq = df["item"].value_counts()
    keep_items = list(item_freq.head(top_keep).index)
    print(f"[W6] Pre-filter: keeping {len(keep_items)} items, dropping {df['item'].nunique()-len(keep_items)} rare/merch for clustering.")
    Xk = X[keep_items].copy()

    PPMI = build_ppmi(Xk)
    PPMI_tf = normalize(PPMI, axis=1, norm="l2")

    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
    Z = pca.fit_transform(PPMI_tf)
    print(f"PCA components chosen: {Z.shape[1]} (explained variance ≥ 95%)")

    # Euclidean k-means
    _km, _labels = kmeans_sweep(Z, keep_items, metric="euclidean", label="Euclidean k-means")
    # Spherical k-means
    _km2, _labels2 = spherical_kmeans(Z, keep_items)

    # Spectral (cosine affinity)
    PPMI_cos = 1 - np.clip(1 - (PPMI_tf @ PPMI_tf.T), 0, 1)
    spectral_try(PPMI_cos, keep_items)

    # DBSCAN sweep (print+plots only)
    print("\n[DBSCAN sweep]")
    eps_list = [0.24, 0.26, 0.28, 0.30, 0.32]
    rows = []
    for eps in eps_list:
        db = DBSCAN(eps=eps, min_samples=3, metric="cosine")
        labs = db.fit_predict(PPMI_tf)
        n_clusters = len(set(l for l in labs if l != -1))
        noise = (labs == -1).sum()
        rows.append((eps, n_clusters, noise))
    print("eps, #clusters, noise")
    for eps, c, n in rows:
        print(f"  {eps:.2f}, {c}, {n}")
    plt.figure(figsize=(6.2,3.1))
    plt.plot([r[0] for r in rows], [r[1] for r in rows], marker='o')
    plt.xlabel("eps"); plt.ylabel("#clusters"); plt.title("DBSCAN sweep: clusters vs eps")
    figsav(figpath("W6_DBSCAN_clusters_vs_eps.png"))

    eps = 0.28
    db = DBSCAN(eps=eps, min_samples=3, metric="cosine")
    labs = db.fit_predict(PPMI_tf)
    sizes = pd.Series(labs[labs!=-1]).value_counts().sort_index()
    print(f"[DBSCAN] Chosen eps={eps:.2f}, cluster sizes (excl. noise):")
    if not sizes.empty:
        for cid, val in sizes.items():
            print(f"  Cluster {cid}: {val}")
    plt.figure(figsize=(6.2,3.1))
    if not sizes.empty:
        plt.bar(sizes.index.astype(str), sizes.values)
    plt.title(f"DBSCAN (eps={eps}) cluster sizes (no noise)")
    plt.xlabel("Cluster id"); plt.ylabel("Size")
    figsav(figpath("W6_DBSCAN_cluster_sizes.png"))

    # ----------------- W7: Graph communities -----------------
    item_names, counts_mat, lift_mat, supp = cooc_and_lift(X)

    # A) Original (broad)
    G_A = graph_from_lift(item_names, lift_mat, supp, X.shape[0], min_lift=1.3, min_count=40)
    summarize_communities(G_A, tag="original")

    # B) Food-only + stronger edges
    food_items = [it for it in item_names if map_category(it) in {"beverage","bread","sweet","savory"}]
    idx = [item_names.index(it) for it in food_items]
    lift_food = lift_mat[np.ix_(idx, idx)]
    supp_food = supp.iloc[idx]
    G_B = graph_from_lift(food_items, lift_food, supp_food, X.shape[0], min_lift=1.4, min_count=60)
    summarize_communities(G_B, tag="food_strong")

    # ----------------- ML1 -----------------
    ml1_propensity(X, df, top_k_targets=10)

    # ----------------- ML2 -----------------
    ml2_next_item_leak_safe(df, X)

    print("\n[DELIVERABLES] Plots saved in ../data/plots/ (no files written to processed/)\nDone ✅")

if __name__ == "__main__":
    main()
