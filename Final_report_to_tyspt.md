> # Analyzing Public Discussions for Product Insights

  ## Executive Overview

  Our project mines Reddit (plus supplemental Tiki e-commerce reviews)
  to surface recurring product issues, pros/cons, and emerging themes
  across 16 hardware-oriented communities. We collected posts and comments
  via the Arctic Shift harvesting workflow to overcome public API rate
  limits, engineered a reusable Parquet-based corpus, delivered exploratory
  dashboards, and built a configurable topic-modeling CLI that fuses
  clustering with aspect-level sentiment. The intended sentiment fine-tuning
  stage was documented but ultimately descoped after a cost/benefit review.

  ## Data Sources & Collection

  - Reddit subreddits span laptops, phones, audio, smart home, photography,
  and DIY communities, ensuring a broad coverage of enthusiast and
  troubleshooting discussions (plan.md:63).
  - Arctic Shift (an internal wrapper around AcademicTorrents archives plus
  Tiki review dumps) replaced API scraping so we could ingest multi-year
  histories without the 1 000-post ceiling of PRAW (plan.md:31).
  - The pipeline normalizes heterogeneous JSONL exports into two typed
  Parquet tables, posts.parquet and comments.parquet, with schema
  harmonization to accommodate malformed dumps (plan.md:79).
  - Each Parquet file retains metadata such as author, timestamps,
  score, and subreddit identifiers, enabling downstream stratification
  (eda/00_summary.txt:1).

  ## Data Engineering & Integration

  - Posts (134 121 rows) and comments (1 300 190 rows) were stored
  separately, then merged on demand for workloads that require both text
  levels (eda/00_summary.txt:1).
  - For modeling, we collapse the relevant subset into a single Polars
  frame filtered by user-defined product aliases and optional subreddit
  constraints (run_pipeline.py:517).
  - Brand and product aliases (e.g., “iphone16 pro”, “hd 600”) drive
  inclusion while being down-weighted in vocabulary by adding them to the
  stopword list before vectorization (run_pipeline.py:537).

  ## Exploratory Data Analysis

  - Volume trends highlight that r/PcBuild, r/iphone, and r/homelab dominate
  both post and comment activity, underscoring heavy builder and Apple
  chatter (eda/posts_by_subreddit.csv:1, eda/comments_by_subreddit.csv:1).
  - Average comment length peaks in r/macbookpro and r/AppleWatch, signaling
  deep technical exchanges, whereas décor and gaming threads stay punchier
  (eda/avg_comment_length_by_subreddit.csv:1).
  - Engagement vs. verbosity shows near-zero correlation, suggesting
  long replies do not automatically earn higher karma (eda/
  score_vs_length_corr.csv:1).
  - Temporal heatmaps and CSV exports capture daily/weekly posting rhythms
  to guide crawl refresh schedules (eda/weekday_hour_heatmap.csv:1).
  - Visual artifacts (bar charts, histograms, word clouds) reside in eda/
  for slide-ready storytelling.

  ## Sentiment Analysis (Initial Plan & Rationale for Dropping)

  - Planned sentiment pipeline: start with multilingual DistilBERT
  (plan.md:111) to align with mixed English/Vietnamese content, then
  fine-tune on a stratified human-labeled sample with Gemini-assisted QA
  (plan.md:118).
  - Required sentiment labels, GPU time, and calibration work were deemed
  disproportionate to the incremental insight; we documented the workflow
  but deferred execution.
  - The operational CLI therefore defaults to VADER for sentence-level
  scoring when deriving aspect sentiment summaries (run_pipeline.py:26,
  run_pipeline.py:694).

  ## Topic Modeling & Aspect Pipeline

  1. Pre-processing
      - Unicode normalization, URL/code stripping, case folding, and token
  pruning keep sentiment cues intact (run_pipeline.py:47).
      - Stopword expansion removes product-brand leakage while preserving
  interrogatives and negations.
  2. Vectorization & Dimensionality Reduction
      - TF-IDF on uni/bi-grams with adaptive min_df and cap on vocabulary
  (run_pipeline.py:545).
      - Optional TruncatedSVD (auto-clamped between 50–200 components)
  provides low-rank representations and explains retained variance
  (run_pipeline.py:560).
  3. Clustering
      - KMeans sweeps candidate K in [k_min, k_max], selecting the
  silhouette-optimal model; TF-IDF means per cluster yield interpretable top
  terms (run_pipeline.py:603).
      - Representative comments are chosen by proximity to cluster centroids
  for qualitative review (run_pipeline.py:286).
  4. Aspect-Based Layer
      - Subreddit membership auto-selects relevant aspect seed dictionaries
  (battery, thermals, comfort, etc.), extensible via TF-IDF-driven expansion
  (run_pipeline.py:666, run_pipeline.py:680).
      - Sentence-level matching assigns quotes to aspects, while VADER
  scores determine polarity summaries and curated positive/negative
  exemplars (run_pipeline.py:337).
  5. Artifacts
      - Persisted outputs include TF-IDF/SVD joblibs, kmeans_clusters.json,
  assignments.jsonl, aspect_summary.json, aspect_summary_by_subreddit.json,
  optional PNG dashboards, and a reproducibility log capturing command-
  line args plus package versions (run_pipeline.py:572, run_pipeline.py:720,
  run_pipeline.py:735).

  ## Deliverables & Usage

  - CLI entry point: run_pipeline.py, invoked with dataset path, product
  label, aliases, and output folder (run_pipeline.py:445).
  - End users can rerun analyses for new products by adjusting aliases/
  subreddits, while keeping evaluation heuristics and aspect taxonomies
  consistent (plan.md:317).
  - Exploratory visualizations and CSVs supply quick wins for reports and
  presentations; JSONL assignments support further downstream analytics.

  ## Limitations & Future Work

  - Sentiment remains rule-based; migrating to the documented transformer
  fine-tuning plan would improve nuance for sarcasm and domain jargon once
  labeled data is available.
  - Current topic exploration emphasizes crisp clusters; adding the NMF
  branch (already scaffolded) would unlock mixed-topic perspectives.
  - Incorporating Tiki datasets more deeply (beyond alias filtering) and
  extending to other locales will broaden coverage.
  - Additional evaluation—coherence, stability checks, human-in-the-loop
  validation—should accompany future iterations (plan.md:309).

  ## Appendix: Key Assets

  - plan.md – overarching roadmap, design trade-offs, and evaluation
  criteria.
  - eda/ – plots and CSV summaries supporting the exploratory analysis.
  - run_pipeline.py – end-to-end automation script for filtering,
  clustering, and aspect sentiment extraction.
