# Netflix Recommendation System 🎬

> A production-inspired collaborative filtering engine built from scratch in Python — featuring both **User-Based** and **Item-Based CF**, real MovieLens data, an interactive Streamlit dashboard, and a full evaluation suite.

🚀 **[Try it Live!](PASTE_YOUR_STREAMLIT_URL_HERE)** 

---

## Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Evaluation Metrics](#-evaluation-metrics)
- [How It Works](#-how-it-works)
- [Docker](#-docker)
- [Testing](#-testing)
- [Configuration](#-configuration)
- [FAQ / Interview Prep](#-faq--interview-prep)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Item-Based CF** | Netflix's approach — stable, scalable, pre-computable offline |
| **User-Based CF** | Classic nearest-neighbour method for educational comparison |
| **Real Data** | Auto-downloads [MovieLens Small](https://grouplens.org/datasets/movielens/) (~100 k ratings) |
| **Modern Titles** | Injects 2020–2024 blockbusters (Oppenheimer, Dune 2, etc.) |
| **Cold-Start Handling** | Popularity-based fallback for unknown users |
| **Full Evaluation** | Precision@K · Recall@K · F1@K · NDCG@K · RMSE |
| **Interactive Dashboard** | Streamlit UI with model selection & recommendation generation |
| **Containerised** | Dockerfile included for one-command deployment |
| **Tested** | `pytest` suite covering metrics and preprocessing |

---

## 🏗 Architecture

```
User Interactions (Ratings)
        │
        ▼
  ┌──────────────┐
  │  Data Loader  │  ← auto-downloads MovieLens or generates synthetic data
  └──────┬───────┘
         ▼
  ┌──────────────────┐
  │  Preprocessing    │  ← pivot to User × Item matrix, mean-center
  └──────┬───────────┘
         ▼
  ┌──────────────────┐
  │  Model Selection  │
  ├────────┬─────────┤
  │ User CF│ Item CF │  ← cosine similarity on rows / columns
  └────┬───┴────┬────┘
       ▼        ▼
  ┌──────────────────┐
  │  Top-K Ranking    │  ← weighted aggregation → sort → filter seen
  └──────┬───────────┘
         ▼
  ┌──────────────────┐
  │  Evaluation       │  ← Precision · Recall · F1 · NDCG
  └──────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- `pip` (or a virtualenv manager of your choice)

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd netflix-recommendation-system

# Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 🎯 Usage

### 1. CLI Pipeline

Runs data loading → training → evaluation → sample recommendations:

```bash
python main.py
```

### 2. Interactive Dashboard

Launches a Streamlit web UI:

```bash
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
netflix-recommendation-system/
├── config.py                        # Centralised constants & hyperparameters
├── main.py                          # CLI entry-point (full pipeline)
├── app.py                           # Streamlit dashboard
├── data/
│   ├── __init__.py
│   ├── loader.py                    # MovieLens download, synthetic generation
│   └── ml-latest-small/             # (auto-downloaded at runtime)
├── models/
│   ├── __init__.py
│   └── collaborative_filtering.py   # UserBasedCF, ItemBasedCF
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                   # RMSE, Precision, Recall, F1, NDCG
├── utils/
│   ├── __init__.py
│   └── preprocessing.py             # Matrix creation & normalisation
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py
│   └── test_preprocessing.py
├── requirements.txt                 # Pinned dependencies
├── pyproject.toml                   # PEP 621 project metadata
├── Dockerfile                       # Container build
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 📊 Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| **Precision@K** | Fraction of top-K recs that are relevant |
| **Recall@K** | Fraction of all relevant items captured in top-K |
| **F1@K** | Harmonic mean of Precision and Recall |
| **NDCG@K** | Ranking quality — rewards relevant items ranked higher |
| **RMSE** | Rating prediction accuracy (lower is better) |

---

## 🧠 How It Works

### Item-Based CF (Netflix Style)

1. Transpose the User × Item matrix to Item × User.
2. Compute **cosine similarity** between every pair of items.
3. For a target user, score each unseen item as: `score(i) = Σ sim(i,j) × rating(u,j)`.
4. Return the top-K highest-scored items.

**Why Netflix prefers this:**
- Items are more stable than user preferences.
- N items ≪ M users → the N×N matrix is tractable.
- Similarity can be pre-computed offline and cached.

### User-Based CF

1. Compute cosine similarity between users on the mean-centered matrix.
2. Find *k* most similar neighbours for the target user.
3. Predict each unseen item's rating as a weighted average of neighbour ratings, plus the user's mean.

---

## 🐳 Docker

```bash
# Build the image
docker build -t netflix-rec .

# Run the CLI pipeline
docker run netflix-rec

# Run the Streamlit dashboard
docker run -p 8501:8501 netflix-rec \
  streamlit run app.py --server.port=8501 --server.address=0.0.0.0
```

---

## ✅ Testing

```bash
# Run the full test suite
pytest

# With verbose output
pytest -v
```

---

## ⚙ Configuration

All tuneable parameters live in **`config.py`**:

| Constant | Default | Purpose |
|----------|---------|---------|
| `DEFAULT_K_NEIGHBORS_USER` | 50 | Neighbours for User-Based CF |
| `DEFAULT_K_NEIGHBORS_ITEM` | 20 | Neighbours for Item-Based CF |
| `DEFAULT_N_RECOMMENDATIONS` | 10 | Items returned per prediction |
| `RELEVANCE_THRESHOLD` | 4.0 | Rating ≥ this counts as "relevant" |
| `TEST_RATIO` | 0.2 | Fraction held out for evaluation |
| `EVAL_USER_SAMPLE_SIZE` | 50 | Users sampled for metric computation |

---

## ❓ FAQ / Interview Prep

<details>
<summary><b>How do you handle matrix sparsity?</b></summary>

Real-world data is 99%+ sparse. We use dense Pandas DataFrames here for readability, but in production you'd switch to `scipy.sparse.csr_matrix` and compute similarity only on non-zero vectors.
</details>

<details>
<summary><b>Why cosine similarity instead of Euclidean distance?</b></summary>

Cosine measures the *angle* (direction of preference) rather than magnitude, so a "strict rater" (max 4/5) and a "lenient rater" (everything 5/5) are treated similarly as long as their preference patterns align.
</details>

<details>
<summary><b>How would you scale to 100M users?</b></summary>

1. **Offline pre-computation** of item-item similarity.
2. **ANN search** (FAISS / HNSW) for sub-linear neighbour lookups.
3. **MapReduce / Spark** for distributed matrix multiplication.
4. **Two-tower neural models** for real-time candidate generation.
</details>

<details>
<summary><b>What about cold-start users?</b></summary>

When a user has no history, collaborative filtering produces no signal. We fall back to a **popularity-based** ranking (most-rated / highest-rated titles) until enough interactions accumulate.
</details>

---

### Resume Bullet Points

- **Designed and implemented a scalable recommendation engine** in Python mimicking Netflix's Item-Based Collaborative Filtering architecture with full evaluation suite (Precision, Recall, F1, NDCG).
- **Optimised for sparsity** using dense matrix operations and Top-K retrieval to handle high-dimensional User–Item interactions.
- **Engineered an evaluation pipeline** integrating RMSE and ranking metrics to validate model performance against historical interaction logs.
- **Solved cold-start problems** by implementing hybrid failover strategies using popularity-based signals for new users.

---

*Built with ❤️ using Python, NumPy, Pandas, scikit-learn, and Streamlit.*
