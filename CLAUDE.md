# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Netflix genre prediction using multi-label text classification. Predicts movie/TV show genres from descriptions using semantic embeddings (Ollama) and scikit-learn.

## Development Commands

This project uses **uv** for dependency management and **Make** for automation:

```bash
# Setup
uv sync                # Install dependencies
ollama pull embeddinggemma  # Download embedding model (622MB, one-time)

# Development workflow
make preprocess        # Generate embeddings and features (~3-5 min)
make train            # Train LR + RF models (~3 min)
make all              # Run full pipeline (preprocess + train)

# Code quality
make lint             # Check with ruff
make format           # Auto-format with ruff

# Cleanup
make clean            # Remove processed_data/ and cache
```

**Ruff configuration:** Line length 88, rules: E/F/W/UP/B/I/A/RUF (see pyproject.toml)

## Core Architecture

**Two-Stage Pipeline:**

1. **Preprocessing** (`netflix_preprocessing.py`):
   - Loads `NetFlix.csv` (7,787 titles)
   - Generates 768D embeddings via Ollama embeddinggemma
   - Encodes categorical features: type (Movie/TV), rating (17 one-hot)
   - Creates 785D feature matrix (768 embeddings + 17 categorical)
   - 80/20 train/test split
   - Saves to `processed_data/` with checkpointing

2. **Training** (`netflix_training.py`):
   - Trains two models:
     - Logistic Regression (baseline, ~30 sec)
     - Random Forest with class_weight='balanced_subsample' (~3 min)
   - CV-based threshold tuning (5-fold, no test leakage)
   - Evaluates on 42 genres using F1 Macro (primary metric)

**Key Design Decisions:**

- **Multi-label classification:** Uses `MultiOutputClassifier` wrapper (42 binary classifiers)
- **Class imbalance handling:** 203:1 ratio → class weighting + threshold tuning (0.15)
- **Feature engineering:** Type (40% of genres are type-specific), rating (age-appropriate content), text embeddings (semantic patterns)
- **Checkpointing:** Embeddings cached (`embeddings.npy`) to avoid regeneration

**Data Flow:**
```
NetFlix.csv → [Preprocessing] → processed_data/ → [Training] → Models
                                 ├── X_train.npy (6229 × 785)
                                 ├── X_test.npy (1558 × 785)
                                 ├── y_train.npy (6229 × 42)
                                 ├── y_test.npy (1558 × 42)
                                 ├── embeddings.npy (cached)
                                 ├── config.json (genre names, metadata)
                                 └── preprocessor.pkl (sklearn encoders)
```

## Critical Implementation Details

**Embedding Generation:**
- Requires Ollama service running: `ollama serve`
- Uses embeddinggemma model (Google, 768D vectors)
- Batch processing with checkpointing (saves `embeddings.npy`)
- If embeddings exist, loads from cache instead of regenerating

**Model Training:**
- F1 Macro prioritized (treats rare genres equally)
- Random Forest: 100 trees, max_depth=20, class_weight='balanced_subsample'
- Threshold optimization: 5-fold CV on training set finds optimal=0.15 (vs default 0.5)
- Final performance: RF F1=0.38 (45% better than LR baseline)

**Genre Prediction Patterns:**
- Easy genres (F1 >0.70): Stand-Up Comedy, Kids' TV, Crime TV Shows (distinct features)
- Ambiguous genres (F1 <0.30): Classic/Cult Movies, LGBTQ, Teen TV (low support or subjective)
- Common confusions: Dramas ↔ International Movies (legitimately overlap 320 times)

**Loading Preprocessed Data:**
```python
from netflix_training import load_data
X_train, X_test, y_train, y_test, config, sklearn_obj = load_data('processed_data')
# config['genre_names'] → list of 42 genres
# sklearn_obj → {'cat_encoder': OneHotEncoder, ...}
```

## External Dependencies

**Ollama Service (required for preprocessing only):**
- Start: `ollama serve`
- Model: embeddinggemma (300M parameters, 622MB download)
- Test connection: `curl http://localhost:11434/api/embed -d '{"model": "embeddinggemma", "input": "test"}'`
- Not needed for training if `processed_data/` already exists

**Documentation:**
- `README.md`: User-facing guide with installation, results, experimental findings
- `EXPERIMENTS.md`: Local-only (gitignored), detailed experimental log with diagnostics