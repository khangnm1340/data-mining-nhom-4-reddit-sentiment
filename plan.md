
# Analyzing Public Discussions for Product Insights

*(Outline — Markdown → Typst → PDF)*

## Title & Team

* **Project:** Mining Reddit to extract product issues, pros/cons, and topic trends
* **Deliverable:** CLI + datasets + analyses + slides
* **Team:** \[add names/roles]

## Objectives

* Identify **common issues** across consumer assets (laptops, phones, home gear, etc.)
* Produce **sentiment scores** per comment and **topics** per subreddit
* Deliver a **CLI** for repeatable analysis + export

---

## Why Reddit? (+ Tiki for diversity)

* **Reddit:** rich, threaded discussions; public API; text-first; strong English NLP support
* **Excluded:**

  * Facebook → heavy bots, limited Vietnamese NLP support
  * TikTok → media-first, no practical API, scraping slow/inefficient
* **Tiki (e-commerce reviews):** complements Reddit with **structured, purchase-verified** user feedback and **Vietnam market** signal

---

## Data Collection — Initial vs. Now

* **Initial (PRAW)**

  * Pros: simple API, good for prototyping
  * Cons: \~1,000 post cap per subreddit (hot/new/top/rising), limited time filtering, rate limits
* **Now (hybrid)**

  * **Reddit historical** via downloaded archives (e.g., Academic Torrents) to bypass caps
  * **Tiki** review dumps (where available) for cross-source validation
  * Result: broader time windows, more volume, better representativeness

---

## How PRAW Traverses Comments (and why it’s not “just save to JSON”)

* **CommentForest structure:** each submission has `submission.comments` (a tree)
* **MoreComments placeholders:** large threads include `MoreComments` nodes; they **require extra API calls**
* **Exhausting the tree:** call `submission.comments.replace_more(limit=None)` → recursively fetch remaining branches
* **Traversal:** iterate via `for c in submission.comments.list():` or DFS/BFS over `comment.replies`
* **Not automatic export:** PRAW returns **Python objects** lazily; you must **walk the tree**, handle rate limits, deleted/removed entries, pagination, and **serialize** yourself (e.g., to JSONL/Parquet)

---

## Alternatives Considered (to bypass API limits)

* **Pushshift (mod-gated now):** powerful time filters; access constraints → not feasible
* **Academic Torrents:** downloadable Reddit snapshots for specific periods → used for scale & history
* **Personal archive:** long-running collector; impractical (hardware/time), misses older content

---

## Subreddits Chosen (assets & ecosystems)

* r/macbookpro, r/GamingLaptops, r/HomeDecorating, r/photography, r/iphone, r/mac, r/AppleWatch, r/Monitors, r/SmartThings, r/PcBuild, r/laptops, r/hometheater, r/headphones, r/ErgoMechKeyboards, r/homelab, r/BudgetAudiophile

---

## EDA & Visualization **\[TODO]**

* Volume over time (posts/comments) per subreddit
* Comment length distribution; user activity distribution
* Top products / models mentioned (NER or keyword heuristics)
* Sentiment by subreddit/product; variance & outliers
* Co-occurrence networks (issue terms ↔ products)

---

## Preprocessing Pipeline

* **Format normalization**

  * Built a **Polars + Nushell wrapper** to ingest **JSONL → Parquet** (`posts.parquet`, `comments.parquet`)
  * JSONL has variable schemas → enforced typed schema in Parquet
  * Identified one malformed file → **removed problematic column**, re-created empty field to align schema
* **Cleaning**

  * Remove URLs; strip markup; collapse whitespace
  * **Language filter:** drop non-English for core run (keeps model assumptions cleaner)
* **Context experiment (hierarchy)**

  * Hypothesis: including parent context improves SA
  * Trial: concatenated parent chains (up to \~512 tokens)
  * Result: **worse** performance (context pollution + truncation); chose **per-comment** modeling
* **Sampling**

  * Initially planned **top-100** per subreddit → biased to virality
  * Switched to **random 100 posts** per subreddit for representativeness (set random seed)

---

## Sentiment Analysis (SA)

* **Baselines (considered)**

  * **VADER** (rule-based): valence lexicon + heuristics

    * Handles intensifiers, negations, punctuation/caps; outputs **compound** ∈ \[−1, 1] + pos/neu/neg
    * Fast, interpretable; **English-centric**, brittle on domain jargon/emojis outside lexicon
  * **TextBlob:** simple polarity/subjectivity; similar limitations
* **Pretrained transformers (chosen)**

  * Tried **twitter-roberta** → too heavy for local CPU
  * Selected **lxyuan/distilbert-base-multilingual-cased-sentiments-student**

    * Pros: multilingual coverage, lighter footprint, good zero-shot behavior
    * Cons: still needs GPU for throughput → **rented GPU on vast.ai**
* **Planned fine-tuning**

  * Goal: domain adapt to product/support discourse
  * Approach: label a **stratified sample** (by subreddit/product/sentiment)
  * Tooling: use **gemini-cli** to assist labeling/QA; track **IAA** (inter-annotator agreement)
  * Metrics: accuracy/F1 on held-out; calibration of thresholds for {neg, neu, pos}
* **Output**

  * Per-comment: `sent_score` ∈ \[−1,1] + label; attach `subreddit`, `product`, `timestamp`

Great summary. Let’s turn it into a **clean, modular pipeline** that (1) satisfies the *Feature Selection / Dimensionality Reduction* requirement, (2) adds **sentiment** + **aspects**, and (3) stays **adaptable across products** and your 16 subreddits.

I’ll give you the flow first, then tiny code stubs only where you asked (basic cleaning + TF-IDF→SVD).

---

# 🧭 End-to-End Flow (adaptable across products)

## 0) Select data

* **Query** your 1M-row Reddit corpus for product mentions (case-insensitive, include model and synonyms):

  * e.g., `{"iphone 16 pro max", "iphone16 pro", "a3106"}` or `{"hd600", "hd 600", "hd6xx", "sennheiser hd600"}`.
* Optionally **stratify** by subreddit (you have 16) so each has representation in downstream steps.

> Output: a filtered frame with at least `name` (id), `subreddit`, `body`, and optional metadata.

---

## 1) Basic cleaning (lightweight, *before* vectorization)

Goals: normalize, remove obvious noise, **preserve sentiment cues**.

* Lowercase; NFC unicode normalize.
* Remove URLs, HTML, code fences.
* Strip non-letters but **keep apostrophes** to preserve negation (“don’t”).
* Tokenize; drop very short tokens (len<3) and stopwords.
* **Extend stopwords with domain terms** (brand/model words you don’t want to dominate: `{"iphone", "apple", "hd600", "sennheiser", "6xx"}` etc).
* Optional: keep a **whitelist** of sentiment shifters: `{"not","never","no","but","however"}`.

### Tiny code stub (cleaning)

```python
import re, unicodedata
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_RE = re.compile(r'https?://\S+|www\.\S+')
CODE_RE = re.compile(r'`{1,3}.*?`{1,3}', re.S)
NON_ALPHA = re.compile(r"[^a-zA-Z'\s]")

def build_stopwords(extra):
    base = set(ENGLISH_STOP_WORDS)
    base |= {s.lower() for s in extra}
    return base

def clean(text, stops):
    if not isinstance(text, str): return ""
    t = unicodedata.normalize("NFC", text)
    t = URL_RE.sub(" ", t)
    t = CODE_RE.sub(" ", t)
    t = t.lower()
    t = NON_ALPHA.sub(" ", t)
    toks = [w for w in t.split() if len(w) > 2 and w not in stops]
    return " ".join(toks)
```

---

## 2) Vectorize → **Feature Selection** → **Dimensionality Reduction**

This is the part your graders care about.

* **Vectorize** with TF-IDF (uni/bi-grams).
* **Feature selection**: `min_df`, `max_df`, and `max_features` (this is explicit, measurable).
* **Dimensionality reduction**: **TruncatedSVD (LSA)** to, say, 100–300 dims. (You can report explained variance.)

### Tiny code stub (FS + DR)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

vec = TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.95, max_features=8000)
X_tfidf = vec.fit_transform(cleaned_texts)         # (N, V_selected)

svd = TruncatedSVD(n_components=200, random_state=42)
X_reduced = svd.fit_transform(X_tfidf)             # (N, 200)  ← DR step
```

Artifacts you can save here:

* `tfidf_vectorizer.joblib`
* `svd_model.joblib`
* `svd_explained_variance_.json` (variance ratio, cumulative)

---

## 3) **Unsupervised structure**: choose KMeans vs LDA vs NMF

Pick based on the *kind* of structure you want:

* **KMeans on SVD** (crisp groups; fast; great for dashboards & cluster labels via top TF-IDF words).

  * Use **silhouette** or **Davies–Bouldin** to select K.
* **NMF on TF-IDF** (soft topics; deterministic; interpretable top-words; docs get topic mixtures).

  * Evaluate with **topic coherence** (e.g., NPMI) and **sparsity**.
* **LDA on counts** (classic probabilistic topics; soft mixtures; slower; requires tuning priors).

  * Evaluate with coherence + held-out likelihood.

> Practical default for product EDA: **NMF** if you want “topic mixtures” per comment, or **KMeans(+SVD)** if you want crisp buckets and speed. For presentations, KMeans + SVD usually yields cleaner visuals; NMF yields richer per-doc topic bars.

---

## 4) **Aspect setup** (adaptable across products)

Make aspects **config-driven** so you can reuse the pipeline.

* Define a small **YAML/JSON config** per *category* (headphones, phones, laptops…), each with:

  * `seeds`: aspect → keywords (e.g., *battery, camera, display* for phones; *comfort, bass, treble* for headphones).
  * `extra_stopwords`: brand/model aliases to ignore in features.
  * optional `expand: true` to enable **auto-expansion** from TF-IDF top n-grams (add candidates to closest aspect bucket by substring or embedding neighbors).

Example (concept):

```yaml
headphones:
  extra_stopwords: [hd600, 6xx, sennheiser]
  aspects:
    comfort: [comfort, clamp, pad, headband, weight]
    bass: [bass, subbass, low end]
    treble: [treble, sibilance, bright]
    soundstage: [soundstage, imaging, separation]
    power: [amp, drive, dac, impedance]
phones:
  extra_stopwords: [iphone, apple, pro, max, 16]
  aspects:
    battery: [battery, charge, drain, screen-on, sot]
    camera: [camera, photos, hdr, low light, ultrawide, telephoto]
    display: [display, brightness, pwm, oled, color]
```

---

## 5) **Aspect extraction + Sentiment**

* **Sentence-level** scan: for each sentence, check if it hits any aspect’s keywords (seeded + expanded).
* Score sentence **sentiment** using:

  * **Multilingual SA model** you already used (preferred, consistent results), or
  * **VADER** fallback when GPU is unavailable.
* Aggregate per aspect:

  * `mentions`, `avg_sentiment`, `top_positive_quotes`, `top_negative_quotes`.
* **Per-subreddit breakdowns** (since you have 16): compute the same metrics by subreddit to show community differences.

Artifacts:

* `aspect_summary.json` (overall)
* `aspect_summary_by_subreddit.json`

---

## 6) **Topic outputs & examples**

* For **KMeans**: show cluster sizes, top terms (mean TF-IDF per cluster), and 5–10 representative comments.
* For **NMF/LDA**: show per-topic top words and top comments (highest topic weight), plus **per-doc topic mixtures** if needed.

Artifacts:

* `cluster_topics.json` / `topics.json`
* `cluster_representatives.json`
* PNGs: **cluster sizes**, **aspects pos/neg bars**.

---

## 7) **Model/Method selection heuristics** (quick decision guide)

* Want **fast, crisp groupings** for slides/demos → **KMeans on SVD** (K via silhouette).
* Want **overlapping topics** that reflect mixed discussions → **NMF on TF-IDF** (k=10–20; tune by coherence).
* Want **probabilistic topics** + priors (e.g., short docs with noise) → **LDA on counts** (slower to tune).

You can also do a **hybrid**:

* Use **KMeans(+SVD)** to create broad buckets → within each cluster, run **aspect+sentiment** for sharper pros/cons.
* Or run **NMF**, then use highest-weight topic as “primary bucket” and still allow secondary topics for nuanced analysis.

---

## 8) **Evaluation & reporting**

* **Clustering**: silhouette (KMeans), stability under resampling, human inspection of top terms & samples.
* **Topics**: UMass/NPMI coherence, sparsity of W/H (NMF), interpretability checks.
* **Sentiment**: sanity checks on polarity distribution per aspect & subreddit; a few manual spot-labels.

---

## 9) **Adaptation knobs** (per product, no code changes)

* `category`: picks the seed/stopword set.
* `product_aliases`: synonyms/sku strings for filtering.
* `min_df`, `max_df`, `max_features`: tighten/loosen vocabulary.
* `svd_components`: 100–300 depending on doc count.
* `topic_k` (or K range): e.g., 5–12 for mixed subreddits; fewer for very tight niches.
* `min_aspect_freq`: filter out rare aspects in summaries.

---

# 🧪 A concrete “default” that works well

* **Vectorize**: TF-IDF (1–2 grams), `min_df=3`, `max_df=0.95`, `max_features=8k`.
* **DR**: TruncatedSVD `n_components=200`.
* **Method**: Start with **KMeans** (K from 4–10; pick by silhouette).
* **Aspects**: category seeds + TF-IDF expansion (top 50 n-grams assigned by substring rules).
* **Sentiment**: your multilingual SA model; fallback to VADER when not available.
* **Outputs**: overall + per-subreddit aspect sentiment, cluster sizes, top terms, quoted examples.

This gives you a clean story:

> “We used TF-IDF with frequency-based feature selection and LSA to reduce dimensionality, then applied KMeans for unsupervised grouping. In parallel, we ran aspect-based sentiment with category-specific seeds (auto-expanded from corpus n-grams). We reported results overall and across 16 subreddits.”

---

If you want, I can turn this into a **single config-driven CLI** layout (folders, `config.yaml` schema, artifact names) that drops into your repo — no heavy code dump, just the scaffolding and a few hooks you can fill in.
