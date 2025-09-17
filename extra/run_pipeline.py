#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
import sys
import unicodedata
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple

import numpy as np

try:
    import polars as pl
except ImportError as e:
    print("polars is required. Install with: pip install polars", file=sys.stderr)
    raise

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from joblib import dump

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None

# -------------------------
# Cleaning utilities
# -------------------------
URL_RE = re.compile(r'https?://\S+|www\.\S+')
CODE_RE = re.compile(r'`{1,3}.*?`{1,3}', re.S)
NON_ALPHA = re.compile(r"[^a-zA-Z'\s]")


def build_stopwords(extra: List[str]) -> set:
    base = set(ENGLISH_STOP_WORDS)
    base |= {s.lower() for s in extra}
    return base




def clean(text: str, stops: set) -> str:
    if not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFC", text)
    t = URL_RE.sub(" ", t)
    t = CODE_RE.sub(" ", t)
    t = t.lower()
    t = NON_ALPHA.sub(" ", t)
    toks = [w for w in t.split() if len(w) > 2 and w not in stops]
    return " ".join(toks)


def sentence_split(text: str) -> List[str]:
    if not text:
        return []
    # Simple sentence splitter; preserves apostrophes
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in parts if s and len(s.strip()) > 2]


# -------------------------
# Aspect seeds by category
# -------------------------
CATEGORY_SUBS = {
    "headphones": {"r/headphones", "r/BudgetAudiophile"},
    "laptops_mac": {"r/GamingLaptops", "r/laptops", "r/macbookpro", "r/mac"},
    "phones_apple": {"r/iphone", "r/AppleWatch"},
    "monitors": {"r/Monitors"},
    "home_av_iot": {"r/hometheater", "r/HomeDecorating", "r/SmartThings"},
    "photography": {"r/photography"},
    "pc_homelab_keyboards": {"r/PcBuild", "r/homelab", "r/ErgoMechKeyboards"},
}

CATEGORY_SEEDS = {
    "headphones": {
        "comfort": ["comfort", "clamp", "pad", "pads", "headband", "weight"],
        "build": ["build", "quality", "materials", "cable", "hinge"],
        "bass": ["bass", "subbass", "low end"],
        "mids": ["mids", "midrange", "vocals"],
        "treble": ["treble", "sibilance", "bright", "sparkle"],
        "soundstage/imaging": ["soundstage", "imaging", "separation", "stage"],
        "power/amp": ["amp", "drive", "dac", "impedance", "power"],
    },
    "laptops_mac": {
        "battery": ["battery", "charge", "drain", "screen on", "sot"],
        "thermals": ["thermal", "thermals", "heat", "hot", "fan"],
        "keyboard": ["keyboard", "keys", "keycap"],
        "trackpad": ["trackpad", "touchpad"],
        "display": ["display", "brightness", "pwm", "oled", "color", "panel"],
        "performance": ["performance", "fps", "benchmark", "slow", "lag"],
        "ports": ["port", "ports", "thunderbolt", "usb", "hdmi", "sd"],
        "weight/noise": ["weight", "noise", "loud", "quiet"],
    },
    "phones_apple": {
        "battery": ["battery", "charge", "drain", "screen on", "sot"],
        "camera": ["camera", "photos", "hdr", "low light", "ultrawide", "telephoto"],
        "display": ["display", "brightness", "pwm", "oled", "color"],
        "performance": ["performance", "lag", "smooth", "slow"],
        "charging": ["charging", "charge", "fast charge", "mag safe", "magsafe"],
        "build": ["build", "design", "aluminum", "titanium", "glass"],
        "health/fitness": ["health", "fitness", "workout", "heart", "sleep"],
    },
    "monitors": {
        "panel": ["panel", "ips", "va", "tn", "oled"],
        "brightness": ["brightness", "nits"],
        "color": ["color", "gamut", "delta", "calibration"],
        "contrast": ["contrast", "black", "hdr"],
        "refresh": ["refresh", "hz"],
        "response": ["response", "gtg", "ms"],
        "HDR": ["hdr", "dolby vision"],
        "uniformity": ["uniformity", "bleed", "ips glow"],
    },
    "home_av_iot": {
        "sound": ["sound", "audio", "bass", "dialogue"],
        "placement": ["placement", "mount", "stand"],
        "calibration": ["calibration", "room", "eq", "auto"],
        "latency": ["latency", "lag"],
        "automation": ["automation", "routine", "scene"],
        "reliability": ["reliability", "disconnect", "bug", "stable"],
        "aesthetics": ["aesthetics", "look", "design"],
    },
    "photography": {
        "autofocus": ["autofocus", "af", "focus"],
        "low-light": ["low light", "noise", "iso"],
        "dynamic range": ["dynamic range", "dr"],
        "stabilization": ["stabilization", "ois", "ibis"],
        "lenses": ["lens", "lenses", "prime", "zoom"],
        "color science": ["color science", "color"],
    },
    "pc_homelab_keyboards": {
        "cpu/gpu/psu": ["cpu", "gpu", "psu"],
        "airflow/thermals": ["airflow", "thermals", "fan"],
        "noise": ["noise", "quiet", "loud"],
        "expandability": ["expand", "slots", "bays"],
        "switches/stabs/keycaps": ["switch", "stab", "keycap"],
        "ergonomics": ["ergonomic", "wrist", "angle"],
    },
}


def detect_categories(subs: List[str]) -> List[str]:
    cats = set()
    for s in subs:
        key = s if s.startswith("r/") else f"r/{s}"
        for cat, sset in CATEGORY_SUBS.items():
            if key in sset:
                cats.add(cat)
    return sorted(cats)


def merge_aspect_seeds(subs_present: List[str]) -> Dict[str, List[str]]:
    cats = detect_categories(subs_present)
    merged: Dict[str, List[str]] = {}
    for c in cats:
        for asp, words in CATEGORY_SEEDS[c].items():
            merged.setdefault(asp, [])
            merged[asp].extend(words)
    # Deduplicate and lowercase
    return {a: sorted(set(w.lower() for w in ws)) for a, ws in merged.items()}


# -------------------------
# Data loading and filtering
# -------------------------
def read_dataset(path: str) -> pl.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".parquet", ".pq"):
        return pl.read_parquet(path)
    if ext in (".csv", ".tsv"):
        sep = "," if ext == ".csv" else "\t"
        return pl.read_csv(path, separator=sep)
    if ext in (".jsonl", ".json"):
        return pl.read_ndjson(path) if ext == ".jsonl" else pl.read_json(path)
    # Fallback try parquet then csv
    try:
        return pl.read_parquet(path)
    except Exception:
        return pl.read_csv(path)


def filter_rows(df: pl.DataFrame, aliases: List[str], subs: List[str] | None) -> pl.DataFrame:
    # Require 'body'; optionally filter by subreddit list
    if "body" not in df.columns:
        raise ValueError("Dataset missing required column: 'body'")

    # Build alias filter (case-insensitive contains)
    safe_aliases = [a.strip().lower() for a in aliases if a and a.strip()]
    if not safe_aliases:
        raise ValueError("At least one alias is required for filtering")
    pattern = "|".join(re.escape(a) for a in safe_aliases)
    body_mask = pl.col("body").str.to_lowercase().str.contains(pattern, literal=False, strict=False)

    # Optional subreddit restriction if provided and column exists
    if subs and "subreddit" in df.columns:
        subs_norm = [s if s.startswith("r/") else f"r/{s}" for s in subs]
        sub_mask = pl.col("subreddit").is_in(subs_norm)
        df = df.filter(body_mask & sub_mask)
    else:
        df = df.filter(body_mask)

    # Keep identifying metadata if present
    keep_cols = [c for c in ["name", "subreddit", "body"] if c in df.columns]
    # If name/subreddit missing, they will be filled later
    return df.select(keep_cols)  # type: ignore


# -------------------------
# Vectorize, SVD, KMeans
# -------------------------
def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def compute_auto_params(N: int) -> Tuple[int, int]:
    # Returns (min_df, svd_components)
    min_df = 3 if N >= 300 else 2
    svd_components = clamp(int(round(0.25 * N)), 50, 200)
    return min_df, svd_components


def vectorize_texts(texts: List[str], ngram_range=(1, 2), min_df=3, max_df=0.95, max_features=2000) -> Tuple[TfidfVectorizer, Any]:
    vec = TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_df=max_df, max_features=max_features)
    try:
        X = vec.fit_transform(texts)
    except ValueError as e:
        if "empty vocabulary" in str(e).lower() and min_df > 1:
            # Retry with min_df=1 as requested
            vec = TfidfVectorizer(ngram_range=ngram_range, min_df=1, max_df=max_df, max_features=max_features)
            X = vec.fit_transform(texts)
        else:
            raise
    return vec, X


def run_svd(X, n_components: int, random_state: int = 42, n_iter: int = 7) -> Tuple[TruncatedSVD, np.ndarray, Dict[str, Any]]:
    svd = TruncatedSVD(n_components=n_components, random_state=random_state, n_iter=n_iter)
    X_red = svd.fit_transform(X)
    var_ratio = svd.explained_variance_ratio_.tolist()
    cumulative = float(np.sum(svd.explained_variance_ratio_))
    report = {"explained_variance_ratio": var_ratio, "cumulative": cumulative}
    return svd, X_red, report


def kmeans_sweep(X_red: np.ndarray, k_min: int, k_max: int, random_state: int = 42, n_init: int = 10) -> Tuple[KMeans, int, float, np.ndarray]:
    best_model = None
    best_k = None
    best_score = -1.0
    best_labels = None
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X_red)
        if len(set(labels)) < 2:
            score = -1.0
        else:
            score = silhouette_score(X_red, labels)
        if score > best_score:
            best_score = score
            best_model = model
            best_k = k
            best_labels = labels
    assert best_model is not None and best_labels is not None and best_k is not None
    return best_model, best_k, float(best_score), best_labels


def top_terms_by_cluster(X_tfidf, labels: np.ndarray, feature_names: List[str], top_n: int = 15) -> Dict[int, List[Tuple[str, float]]]:
    clusters = {}
    for cid in sorted(set(labels)):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            clusters[cid] = []
            continue
        # Mean TF-IDF across docs in cluster
        sub = X_tfidf[idx]
        mean_vec = np.asarray(sub.mean(axis=0)).ravel()
        top_idx = np.argsort(mean_vec)[::-1][:top_n]
        clusters[cid] = [(feature_names[i], float(mean_vec[i])) for i in top_idx]
    return clusters


def representatives_by_cluster(X_red: np.ndarray, labels: np.ndarray, centers: np.ndarray, k: int, top_m: int = 5) -> Dict[int, List[int]]:
    reps = {}
    for cid in range(k):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            reps[cid] = []
            continue
        center = centers[cid]
        d = np.linalg.norm(X_red[idx] - center, axis=1)
        order = np.argsort(d)
        reps[cid] = idx[order[:top_m]].tolist()
    return reps


# -------------------------
# Aspect assignment and sentiment
# -------------------------
def build_aspect_matchers(aspect_seeds: Dict[str, List[str]]) -> Dict[str, List[re.Pattern]]:
    matchers = {}
    for asp, seeds in aspect_seeds.items():
        pats = []
        for s in seeds:
            s_esc = re.escape(s)
            # Word boundary if single token, else substring
            if " " in s:
                pats.append(re.compile(s_esc, re.IGNORECASE))
            else:
                pats.append(re.compile(rf"\b{s_esc}\b", re.IGNORECASE))
        matchers[asp] = pats
    return matchers


def expand_aspect_seeds_from_tfidf(aspect_seeds: Dict[str, List[str]], feature_names: List[str], top_k: int, exclude: set[str]) -> Dict[str, List[str]]:
    # Use top_k features (by order given) and assign to aspect if substring match with any seed
    additions: Dict[str, set[str]] = {a: set() for a in aspect_seeds}
    for term in feature_names[:top_k]:
        t = term.lower()
        if t in exclude:
            continue
        for asp, seeds in aspect_seeds.items():
            for s in seeds:
                if t == s:
                    continue
                if t.find(s) != -1 or s.find(t) != -1:
                    additions[asp].add(t)
                    break
    # Merge
    merged = {a: sorted(set(ws) | additions[a]) for a, ws in aspect_seeds.items()}
    return merged


def assign_aspects_and_sentences(rows: List[Dict[str, Any]], aspect_matchers: Dict[str, List[re.Pattern]], analyzer: Any) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    # Returns (overall_aspect_summary, per_subreddit_summary, assignments)
    overall = defaultdict(lambda: {"mentions": 0, "sum_sent": 0.0, "top_positive_quotes": [], "top_negative_quotes": []})
    per_sub = defaultdict(lambda: defaultdict(lambda: {"mentions": 0, "sum_sent": 0.0}))
    assignments = []

    def add_quote(store_list, quote_obj, pos=True, limit=5):
        store_list.append(quote_obj)
        # keep top by sentiment
        key = (lambda x: x["sentiment"]) if pos else (lambda x: -x["sentiment"])  # type: ignore
        store_list.sort(key=key, reverse=True)
        if len(store_list) > limit:
            del store_list[limit:]

    for row in rows:
        doc_name = row["name"]
        sub = row["subreddit"]
        body = row.get("body", "")
        sentences = sentence_split(body)
        doc_aspects = defaultdict(lambda: {"mentions": 0, "sum_sent": 0.0})
        for sent in sentences:
            hits = []
            for asp, pats in aspect_matchers.items():
                if any(p.search(sent) for p in pats):
                    hits.append(asp)
            if not hits:
                continue
            score = analyzer.polarity_scores(sent).get("compound", 0.0)
            for asp in set(hits):
                overall[asp]["mentions"] += 1
                overall[asp]["sum_sent"] += score
                per_sub[sub][asp]["mentions"] += 1
                per_sub[sub][asp]["sum_sent"] += score
                doc_aspects[asp]["mentions"] += 1
                doc_aspects[asp]["sum_sent"] += score
                quote_obj = {"name": doc_name, "subreddit": sub, "sentence": sent.strip(), "sentiment": score}
                if score >= 0.05:
                    add_quote(overall[asp]["top_positive_quotes"], quote_obj, pos=True)
                if score <= -0.05:
                    add_quote(overall[asp]["top_negative_quotes"], quote_obj, pos=False)

        # Finalize doc aspects
        doc_aspects_final = {a: {"mentions": v["mentions"], "avg_sentiment": (v["sum_sent"] / v["mentions"]) if v["mentions"] else 0.0} for a, v in doc_aspects.items()}
        assignments.append({"name": doc_name, "subreddit": sub, "aspects": doc_aspects_final})

    # Finalize overall and per_sub
    overall_final = {a: {
        "mentions": v["mentions"],
        "avg_sentiment": (v["sum_sent"] / v["mentions"]) if v["mentions"] else 0.0,
        "top_positive_quotes": v["top_positive_quotes"],
        "top_negative_quotes": v["top_negative_quotes"],
    } for a, v in overall.items()}

    per_sub_final = {}
    for sub, d in per_sub.items():
        per_sub_final[sub] = {a: {
            "mentions": v["mentions"],
            "avg_sentiment": (v["sum_sent"] / v["mentions"]) if v["mentions"] else 0.0,
        } for a, v in d.items()}

    return overall_final, per_sub_final, assignments


# -------------------------
# Plotting helpers
# -------------------------
def plot_cluster_sizes(sizes: Dict[int, int], out_path: str):
    import matplotlib.pyplot as plt
    ks = sorted(sizes)
    vals = [sizes[k] for k in ks]
    plt.figure(figsize=(6, 4))
    plt.bar([str(k) for k in ks], vals, color="#4C78A8")
    plt.xlabel("Cluster")
    plt.ylabel("Count")
    plt.title("Cluster Sizes")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_aspects_pos_neg(aspect_quotes: Dict[str, Any], out_path: str):
    import matplotlib.pyplot as plt
    aspects = sorted(aspect_quotes.keys())
    pos_counts, neg_counts = [], []
    for a in aspects:
        pos = sum(1 for q in aspect_quotes[a].get("top_positive_quotes", []) if q["sentiment"] >= 0.05)
        neg = sum(1 for q in aspect_quotes[a].get("top_negative_quotes", []) if q["sentiment"] <= -0.05)
        pos_counts.append(pos)
        neg_counts.append(neg)

    x = np.arange(len(aspects))
    width = 0.35
    plt.figure(figsize=(max(6, len(aspects) * 0.6), 4))
    plt.bar(x - width/2, pos_counts, width, label='Positive', color="#59A14F")
    plt.bar(x + width/2, neg_counts, width, label='Negative', color="#E15759")
    plt.xticks(x, aspects, rotation=45, ha='right')
    plt.ylabel('Top quote counts')
    plt.title('Aspects: Positive vs Negative Quotes')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -------------------------
# Main CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Config-driven Reddit product insights pipeline")
    parser.add_argument("--data", required=True, help="Path to dataset (parquet/csv/jsonl)")
    parser.add_argument("--product", required=True, help="Product name string")
    parser.add_argument("--aliases", required=True, help="Comma-separated alias strings")
    parser.add_argument("--out", required=True, help="Artifacts output directory")
    parser.add_argument("--subs", default="", help="Optional comma-separated subreddits to restrict")

    # Vectorize/FS
    parser.add_argument("--min-df", default="auto", help="min_df or 'auto' (3 if N>=300 else 2; retry 1 on empty vocab)")
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--max-feat", default="auto", help="max_features or 'auto' = min(2000, 3*N)")
    parser.add_argument("--ngram-min", type=int, default=1)
    parser.add_argument("--ngram-max", type=int, default=2)

    # SVD
    parser.add_argument("--svd", default="auto", help="n_components or 'auto' (clamp(0.25N,50,200) and <= min(V-1,N-1)) or 'skip'")
    parser.add_argument("--svd-random-state", type=int, default=42)
    parser.add_argument("--svd-n-iter", type=int, default=7)

    # Unsupervised
    parser.add_argument("--method", choices=["kmeans", "nmf"], default="kmeans")
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=8)
    parser.add_argument("--kmeans-random-state", type=int, default=42)
    parser.add_argument("--kmeans-n-init", type=int, default=10)
    # NMF params (not implemented fully; placeholder for interface)
    parser.add_argument("--nmf-k-min", type=int, default=10)
    parser.add_argument("--nmf-k-max", type=int, default=15)
    parser.add_argument("--nmf-random-state", type=int, default=42)
    parser.add_argument("--nmf-init", default="nndsvda")
    parser.add_argument("--nmf-alpha", type=float, default=0.0)
    parser.add_argument("--nmf-l1-ratio", type=float, default=0.0)

    # Aspects
    parser.add_argument(
        "--aspect-category",
        default="auto",
        help=(
            "Aspect category: 'auto' (infer from subreddit) or one/more of "
            + ", ".join(sorted(CATEGORY_SEEDS.keys()))
            + " (comma-separated)"
        ),
    )
    parser.add_argument("--expand-seeds", action="store_true", help="Expand aspect seeds from TF-IDF top features")
    parser.add_argument("--expand-top-k", type=int, default=50)
    parser.add_argument("--min-aspect-freq", default="auto", help="min mentions to keep aspect in summary: 'auto' max(5,round(0.01N))")

    parser.add_argument("--save-plots", action="store_true")

    args = parser.parse_args()

    # Repro log
    echo_cmd = "python " + " ".join([re.escape(arg) if " " in arg else arg for arg in sys.argv])
    print("Repro command:")
    print(echo_cmd)

    # Version log
    versions = {
        "python": sys.version.split(" ")[0],
        "polars": getattr(pl, "__version__", "unknown"),
        "sklearn": __import__("sklearn").__version__,
        "numpy": np.__version__,
    }

    # Ensure VADER
    if SentimentIntensityAnalyzer is None:
        print("VADER not available. Install with: pip install vaderSentiment", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # Load and filter
    df = read_dataset(args.data)
    aliases = [a.strip() for a in args.aliases.split(",") if a.strip()]
    subs = [s.strip() for s in args.subs.split(",") if s.strip()] if args.subs else None
    df_f = filter_rows(df, aliases, subs)

    # Convert to rows list and ensure defaults for name/subreddit
    rows = df_f.to_dicts()
    for i, r in enumerate(rows):
        if "name" not in r:
            r["name"] = f"doc_{i}"
        if "subreddit" not in r:
            r["subreddit"] = "r/unknown"
    N = len(rows)
    print(f"Filtered N={N} rows")
    if N == 0:
        print("No rows after filtering; exiting.")
        sys.exit(0)

    # Cleaning
    brand_terms = set()
    for s in [args.product] + aliases:
        for w in re.split(r"\W+", s.lower()):
            if w:
                brand_terms.add(w)
    stops = build_stopwords(extra=list(brand_terms))
    cleaned = [clean(r.get("body", ""), stops) for r in rows]

    # Auto params
    auto_min_df, auto_svd = compute_auto_params(N)
    min_df = auto_min_df if args.min_df == "auto" else int(args.min_df)
    max_features = min(2000, 3 * N) if args.max_feat == "auto" else int(args.max_feat)

    # Vectorize with retry-on-empty handled in helper
    vec, X_tfidf = vectorize_texts(
        cleaned,
        ngram_range=(args.ngram_min, args.ngram_max),
        min_df=min_df,
        max_df=args.max_df,
        max_features=max_features,
    )
    V = len(vec.get_feature_names_out())

    # SVD params respecting bounds
    if args.svd == "skip":
        X_red = None
        svd = None
        svd_report = None
        n_components = None
    else:
        base_n = auto_svd if args.svd == "auto" else int(args.svd)
        max_allowed = max(2, min(V - 1, N - 1))
        n_components = int(min(base_n, max_allowed))
        svd, X_red, svd_report = run_svd(X_tfidf, n_components=n_components, random_state=args.svd_random_state, n_iter=args.svd_n_iter)

    # Save vectorizer and svd artifacts
    dump(vec, os.path.join(out_dir, "tfidf_vectorizer.joblib"))
    if svd is not None:
        dump(svd, os.path.join(out_dir, "svd_model.joblib"))
        with open(os.path.join(out_dir, "svd_explained_variance.json"), "w", encoding="utf-8") as f:
            json.dump(svd_report, f, ensure_ascii=False, indent=2)

    # Unsupervised (default KMeans on SVD)
    method_used = args.method
    method_info: Dict[str, Any] = {
        "method": method_used,
        "k_search": [args.k_min, args.k_max],
        "kmeans_random_state": args.kmeans_random_state,
        "kmeans_n_init": args.kmeans_n_init,
        "params": {
            "vectorizer": {
                "ngram_range": [args.ngram_min, args.ngram_max],
                "min_df": min_df,
                "max_df": args.max_df,
                "max_features": max_features,
            },
            "svd": {
                "n_components": n_components,
                "random_state": args.svd_random_state,
                "n_iter": args.svd_n_iter,
            },
        },
        "versions": versions,
    }

    if method_used == "kmeans":
        if X_red is None:
            # If SVD skipped, use TF-IDF directly (dense risk); better to require SVD
            print("SVD is recommended before KMeans; proceeding on TF-IDF which may be slow.")
            X_use = X_tfidf
        else:
            X_use = X_red
        model, best_k, best_sil, labels = kmeans_sweep(
            X_use, args.k_min, args.k_max, random_state=args.kmeans_random_state, n_init=args.kmeans_n_init
        )
        method_info.update({"chosen_k": int(best_k), "silhouette": float(best_sil)})

        # Cluster sizes and top terms by averaging TF-IDF within each cluster
        sizes = {int(c): int(n) for c, n in Counter(labels).items()}
        feature_names = vec.get_feature_names_out().tolist()
        cluster_terms = top_terms_by_cluster(X_tfidf, labels, feature_names, top_n=15)
        reps_idx = representatives_by_cluster(X_use, labels, model.cluster_centers_, best_k, top_m=5)

        # Build cluster JSON
        clusters_json = []
        for cid in range(best_k):
            rep_rows = [rows[i] for i in reps_idx.get(cid, [])]
            rep_examples = [{"name": r["name"], "subreddit": r["subreddit"], "body": r["body"]} for r in rep_rows]
            clusters_json.append({
                "cluster": int(cid),
                "size": int(sizes.get(cid, 0)),
                "top_terms": cluster_terms.get(cid, []),
                "representatives": rep_examples,
            })

        with open(os.path.join(out_dir, "kmeans_clusters.json"), "w", encoding="utf-8") as f:
            json.dump({"clusters": clusters_json}, f, ensure_ascii=False, indent=2)

        if args.save_plots:
            plot_cluster_sizes({int(k): int(v) for k, v in sizes.items()}, os.path.join(out_dir, "cluster_sizes.png"))

        # Add per-doc cluster to assignments later
        per_doc_clusters = {i: int(lbl) for i, lbl in enumerate(labels)}
    else:
        # NMF pathway not fully implemented in this script; placeholder
        print("NMF method requested, but KMeans is preferred and implemented here.")
        per_doc_clusters = {i: -1 for i in range(N)}

    # Aspects
    subs_present = sorted(set(r["subreddit"] for r in rows))
    # Resolve aspect categories
    if args.aspect_category == "auto":
        aspect_seeds = merge_aspect_seeds(subs_present)
        # Fallback: if no known subreddits present, merge all categories to keep aspects working
        if not aspect_seeds:
            merged_all: Dict[str, List[str]] = {}
            for cat in CATEGORY_SEEDS.values():
                for asp, words in cat.items():
                    merged_all.setdefault(asp, [])
                    merged_all[asp].extend(words)
            aspect_seeds = {a: sorted(set(w.lower() for w in ws)) for a, ws in merged_all.items()}
    else:
        # Allow explicit category selection (comma-separated)
        chosen = [c.strip() for c in args.aspect_category.split(",") if c.strip()]
        merged: Dict[str, List[str]] = {}
        for cat in chosen:
            if cat not in CATEGORY_SEEDS:
                print(f"[warn] Unknown aspect category '{cat}'. Options: {sorted(CATEGORY_SEEDS.keys())}", file=sys.stderr)
                continue
            for asp, words in CATEGORY_SEEDS[cat].items():
                merged.setdefault(asp, [])
                merged[asp].extend(words)
        aspect_seeds = {a: sorted(set(w.lower() for w in ws)) for a, ws in merged.items()}
        if not aspect_seeds:
            print("[warn] No valid aspect categories resolved; falling back to 'auto' across all.", file=sys.stderr)
            merged_all: Dict[str, List[str]] = {}
            for cat in CATEGORY_SEEDS.values():
                for asp, words in cat.items():
                    merged_all.setdefault(asp, [])
                    merged_all[asp].extend(words)
            aspect_seeds = {a: sorted(set(w.lower() for w in ws)) for a, ws in merged_all.items()}

    # Expansion from TF-IDF
    if args.expand_seeds and aspect_seeds:
        # Feature ranking: use global mean tfidf
        mean_tfidf = np.asarray(X_tfidf.mean(axis=0)).ravel()
        order = np.argsort(mean_tfidf)[::-1]
        feat_sorted = [vec.get_feature_names_out()[i] for i in order]
        exclude_terms = set(brand_terms)
        # also exclude aliases (as strings and tokens)
        for a in aliases:
            exclude_terms.add(a.lower())
            for w in a.lower().split():
                exclude_terms.add(w)
        aspect_seeds = expand_aspect_seeds_from_tfidf(aspect_seeds, feat_sorted, args.expand_top_k, exclude_terms)

    # Sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    # Assign aspects and compute sentiment
    matchers = build_aspect_matchers(aspect_seeds)
    overall_aspects, per_sub_aspects, aspect_assignments = assign_aspects_and_sentences(rows, matchers, analyzer)

    # Apply min aspect freq
    min_aspect_freq = max(5, int(round(0.01 * N))) if args.min_aspect_freq == "auto" else int(args.min_aspect_freq)
    overall_aspects = {a: v for a, v in overall_aspects.items() if v.get("mentions", 0) >= min_aspect_freq}
    per_sub_aspects = {
        sub: {a: v for a, v in d.items() if a in overall_aspects}
        for sub, d in per_sub_aspects.items()
    }

    # Save assignments.jsonl with per-doc cluster and aspects summary
    with open(os.path.join(out_dir, "assignments.jsonl"), "w", encoding="utf-8") as f:
        for i, doc in enumerate(rows):
            doc_aspects = next((a["aspects"] for a in aspect_assignments if a["name"] == doc["name"]), {})
            out_obj = {
                "name": doc["name"],
                "subreddit": doc["subreddit"],
                "cluster": per_doc_clusters.get(i, -1),
                "aspects": doc_aspects,
            }
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

    # Save aspect summaries
    with open(os.path.join(out_dir, "aspect_summary.json"), "w", encoding="utf-8") as f:
        json.dump(overall_aspects, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "aspect_summary_by_subreddit.json"), "w", encoding="utf-8") as f:
        json.dump(per_sub_aspects, f, ensure_ascii=False, indent=2)

    # Save method.json
    method_path = os.path.join(out_dir, "method.json")
    with open(method_path, "w", encoding="utf-8") as f:
        json.dump(method_info, f, ensure_ascii=False, indent=2)

    # Save aspects plot
    if args.save_plots:
        plot_aspects_pos_neg(overall_aspects, os.path.join(out_dir, "aspects_pos_neg.png"))

    # Console summary
    print("--- Summary ---")
    print(f"N={N}, V={V}, min_df={min_df}, max_df={args.max_df}, max_features={max_features}")
    if svd is not None:
        print(f"SVD components={n_components}, cumulative explained variance={svd_report['cumulative']:.3f}")
    if method_used == "kmeans":
        print(f"Chosen K={method_info['chosen_k']} with silhouette={method_info['silhouette']:.3f}")
    # Print 1 example quote per aspect polarity (if present)
    for a, v in overall_aspects.items():
        pos_q = v.get("top_positive_quotes", [])
        neg_q = v.get("top_negative_quotes", [])
        if pos_q:
            print(f"Aspect {a} POS: {pos_q[0]['sentence']} ({pos_q[0]['sentiment']:.2f})")
        if neg_q:
            print(f"Aspect {a} NEG: {neg_q[0]['sentence']} ({neg_q[0]['sentiment']:.2f})")


if __name__ == "__main__":
    main()
