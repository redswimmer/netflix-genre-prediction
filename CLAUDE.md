# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Netflix genre prediction project that uses multi-label text classification to predict movie/TV show genres from descriptions. The system leverages semantic embeddings via Ollama and scikit-learn for machine learning.

## Development Environment Setup

This project uses **uv** for dependency management. Essential setup commands:

```bash
# Install dependencies from pyproject.toml
uv sync

# Activate virtual environment  
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Verify Ollama connection (required for embeddings)
ollama serve
ollama pull embeddinggemma
curl http://localhost:11434/api/embed -d '{"model": "embeddinggemma", "input": "test"}'
```

## Core Architecture

**Two-Stage Pipeline:**
1. **Preprocessing** (`netflix_preprocessing.py`): Generates 768-dimensional embeddings using Ollama's embeddinggemma model, encodes categorical features (type, rating), creates train/test splits, saves to `processed_data/`
2. **Training** (`netflix_training.py`): Loads preprocessed data, trains multi-label classifiers (Logistic Regression baseline + Random Forest), evaluates with comprehensive metrics, performs error analysis

**Data Flow:**
- Input: `NetFlix.csv` (7,787 titles with descriptions, genres, metadata)
- Processing: Text → embeddings (768D) + categorical features (17D) = 785D feature matrix
- Output: Multi-label predictions across 42 genre categories

**Key Dependencies:**
- `ollama` - Required for generating semantic embeddings (embeddinggemma model)
- `scikit-learn` - Multi-label classification with MultiOutputClassifier
- `pandas/numpy` - Data manipulation and matrix operations
- `tqdm` - Progress tracking during embedding generation

## Running the Project

```bash
# Full pipeline (run in order)
python netflix_preprocessing.py  # ~3-5 minutes for embedding generation
python netflix_training.py       # ~3 minutes for Random Forest training

# Load preprocessed data for custom models
from netflix_training import load_data
X_train, X_test, y_train, y_test, config, sklearn_obj = load_data('processed_data')
```

## Important Implementation Details

**Multi-label Classification:**
- Uses `MultiOutputClassifier` wrapper for binary classification per genre
- Primary metric: F1 Macro (treats rare genres equally with common ones)
- Handles class imbalance (203:1 ratio between most/least common genres)

**Feature Engineering:**
- Semantic embeddings capture contextual meaning and synonyms
- Categorical features handle type-specific genres ("TV Dramas" vs "Dramas")
- Rating correlates with content appropriateness (Kids vs Horror)

**Error Analysis:**
- Identifies genre confusion patterns (e.g., "International Movies" vs "Dramas")
- Analyzes exact vs partial prediction errors
- Provides example misclassifications with descriptions

**Checkpointing:**
- All processed data saved to `processed_data/` directory
- Includes feature matrices, target arrays, configuration, and sklearn objects
- Enables iterative model development without re-preprocessing

## External Dependencies

**Ollama Service:**
- Must be running (`ollama serve`) for embedding generation
- Uses embeddinggemma model (622MB download)
- Essential for preprocessing; not needed for training with existing data