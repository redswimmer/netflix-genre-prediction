# Netflix Genre Prediction

A multi-label text classification project that predicts Netflix content genres using semantic embeddings from descriptions and categorical features.

---

## 🎯 Project Overview

This data mining project tackles the challenge of predicting multiple genre labels for Netflix movies and TV shows using text descriptions combined with categorical metadata features. The system uses semantic embeddings to capture meaning from descriptions, plus categorical features (type: Movie/TV, rating: PG/TV-MA/etc.) to train multi-label classifiers that handle content spanning multiple genres.

**Key Features:**
- 🤖 Semantic text embeddings using Ollama's embeddinggemma (768 dimensions)
- 🏷️ Multi-label classification supporting 42 different genres
- 📊 Comprehensive evaluation metrics for multi-label tasks
- 🔍 Error analysis revealing genre ambiguity and confusion patterns
- 🚀 Production-ready preprocessing pipeline with checkpointing
- 📈 Model comparison (Logistic Regression vs Random Forest)

**Dataset:** 7,787 Netflix titles with descriptions, metadata (type, rating), and genre labels

---

## 🛠️ Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.

### Prerequisites

1. **Install uv** (if not already installed):
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

2. **Install Ollama** (for embeddings):
```bash
# Visit https://ollama.com/download for installation
# Or on macOS:
brew install ollama
```

### Project Setup

1. **Clone the repository:**
```bash
git clone https://github.com/redswimmer/netflix-genre-prediction.git
cd netflix-genre-prediction
```

2. **Install Python dependencies:**
```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

uv sync
```

3. **Pull the embedding model:**
```bash
# Start Ollama service (if not running)
ollama serve

# Pull embeddinggemma model (622MB, one-time download)
ollama pull embeddinggemma
```

4. **Verify installation:**
```bash
# Test Ollama connection
curl http://localhost:11434/api/embed -d '{
  "model": "embeddinggemma",
  "input": "test"
}'
```

---

## 📁 Project Structure

```
netflix-genre-prediction/
├── NetFlix.csv                    # Dataset
├── netflix_preprocessing.py       # Preprocessing pipeline
├── netflix_training.py            # Model training & evaluation
├── Makefile                       # Build automation commands
├── pyproject.toml                 # Project dependencies & ruff config
├── CLAUDE.md                      # AI assistant instructions
└── processed_data/                # Generated during preprocessing
    ├── X.npy                      # Feature matrix (7787, 785)
    ├── y.npy                      # Target matrix (7787, 42)
    ├── X_train.npy                # Training features
    ├── X_test.npy                 # Test features
    ├── y_train.npy                # Training targets
    ├── y_test.npy                 # Test targets
    ├── embeddings.npy             # Text embeddings (7787, 768)
    ├── config.json                # Configuration metadata
    ├── preprocessor.pkl           # Sklearn encoders
    └── processed_df.csv           # Full processed dataframe
```

---

## 🚀 Quick Start

### Step 1: Preprocess the Data

```bash
make preprocess 
```

**What happens:**
- Loads `NetFlix.csv`
- Parses 42 unique genres from multi-label format
- Generates 768-dimensional embeddings using Ollama
- Encodes categorical features (type: Movie/TV Show, rating: TV-MA/PG/etc.)
- Combines features into final matrix (785 dimensions)
- Creates 80/20 train/test split
- Saves all processed data to `processed_data/`

### Step 2: Train and Evaluate Models

```bash
make train
```

**What happens:**
- Trains Logistic Regression (baseline)
- Trains Random Forest with class weighting and optimized threshold
- Evaluates using multi-label metrics
- Uses cross-validation for threshold tuning

**Results:**
```
FINAL MODEL: Random Forest (class_weight + CV-tuned threshold)
  F1 Score (macro):    0.38
  F1 Score (micro):    0.53
  Training time:       ~3 minutes

Performance by Genre Frequency:
  High-frequency genres (>100 samples):  F1 = 0.60-0.87  ✓ Works well
  Medium-frequency (50-100 samples):     F1 = 0.30-0.60  ⚠ Mixed
  Low-frequency (<50 samples):           F1 = 0.00-0.30  ✗ Struggles

Top Performing Genres:
  Stand-Up Comedy:     F1 = 0.87
  Kids' TV:            F1 = 0.81
  Crime TV Shows:      F1 = 0.72
  International Movies: F1 = 0.58
  Documentaries:       F1 = 0.61
```

**Key Findings:**
- Model works reliably for 15-20 common genres (60-70% of use cases)
- Class imbalance (203:1 ratio) makes rare genres difficult to predict
- For multi-label classification with 42 classes, F1=0.38 is reasonable performance

---

## 🔬 Experimental Results

### Model Comparison: Logistic Regression vs Random Forest

We trained and evaluated two models on the Netflix genre prediction task:

| Model | F1 Macro | F1 Micro | Exact Match | Training Time | Prediction Time |
|-------|----------|----------|-------------|---------------|-----------------|
| **Logistic Regression** | 0.2604 | 0.5696 | 19.26% | ~30 seconds | Fast |
| **Random Forest (optimized)** | **0.3783** | 0.5329 | 1.60% | ~3 minutes | Medium |

**Winner: Random Forest** with class weighting and CV-tuned threshold (0.15)
- **+45.3% improvement** in F1 Macro over baseline
- Optimization: 5-fold CV on training set (no test set peeking)
- Configuration: 100 trees, max_depth=20, class_weight='balanced_subsample'

### Efficiency Analysis

**Training Time:**
- Logistic Regression: ~30 seconds (fast iteration)
- Random Forest: ~3 minutes (including CV threshold tuning: ~2 minutes)

**Prediction Time:**
- Both models provide real-time predictions (<1 second for 1,558 test samples)

**Memory:**
- Feature matrix: 785 dimensions (768 embeddings + 17 categorical)
- Models fit in memory on standard laptop (8GB+ RAM recommended)

**Trade-offs:**
- LR: Faster training, better exact match, good for prototyping
- RF: Better F1 scores, handles non-linear patterns, production choice

### Genre-Level Insights

#### ✅ Easy to Predict (F1 > 0.70)

These genres have **clear distinguishing features** and sufficient training data:

| Genre | F1 Score | Why It's Easy | Key Features |
|-------|----------|---------------|--------------|
| Stand-Up Comedy | 0.87 | Unique vocabulary ("comedian", "jokes", "stage") | Text embeddings + type |
| Kids' TV | 0.81 | Distinct rating (TV-Y, TV-Y7) + keywords | Rating + text |
| Crime TV Shows | 0.72 | Type-specific + crime vocabulary | Type + text |
| Children & Family Movies | 0.71 | Clear ratings (G, PG) + family themes | Rating + text |

**Common characteristics:**
- **High support** (>70 training samples)
- **Distinct features** (unique ratings, keywords, or type)
- **Clear semantic patterns** in descriptions

#### ⚠️ Ambiguous Genres (F1 < 0.30)

These genres are **hard to distinguish** due to overlap and limited data:

| Genre | F1 Score | Why It's Ambiguous | Confusion With |
|-------|----------|-------------------|----------------|
| Teen TV Shows | 0.15 | Overlaps with Kids' TV and TV Dramas | Kids' TV, TV Dramas |
| Classic Movies | 0.00 | Temporal concept, not in descriptions | Dramas, International |
| Cult Movies | 0.00 | Subjective label, no clear features | Horror, Thrillers |
| LGBTQ Movies | 0.00 | Low support (20 samples) + subtle themes | Dramas, Romantic |
| TV Thrillers | 0.00 | Extremely low support (10 samples) | Crime TV, TV Dramas |

**Common characteristics:**
- **Low support** (<20 training samples) → insufficient data
- **Subjective labels** (e.g., "Cult", "Classic") → no semantic markers
- **High overlap** with other genres → model confusion
- **Temporal/cultural concepts** not expressed in descriptions

#### 🔀 Most Confused Genre Pairs

Our error analysis reveals systematic confusion patterns:

| Predicted | Actually Was | Count | Explanation |
|-----------|-------------|-------|-------------|
| Dramas | International Movies | 81 | Often co-occur (foreign films are dramatic) |
| Dramas | Comedies | 71 | Dramedy overlap, subtle tone differences |
| International Movies | Dramas | 54 | Reverse confusion, same root cause |
| International Movies | Comedies | 41 | Foreign comedies mislabeled as just "International" |
| International TV Shows | TV Dramas | 29 | Type correctly identified, genre ambiguous |

**Key insight:** "Dramas" and "International Movies" are frequently **co-labels** (appear together in 320 titles), making them legitimately overlapping rather than errors.

### Feature Importance

While Random Forest doesn't provide direct feature importance for multi-label tasks, we can infer from performance:

#### 🎯 Most Valuable Features

1. **Type (Movie/TV Show)** - Nearly perfect for type-specific genres
   - "Kids' TV" vs "Children & Family Movies" perfectly separated
   - "Crime TV Shows" vs "Crime Movies" distinction
   - Impact: ~40% of genres are type-specific

2. **Rating** - Strong signal for age-appropriate content
   - TV-Y/TV-Y7 → Kids' TV (F1=0.81)
   - TV-MA → distinguishes adult content
   - G/PG → Family content (F1=0.71)
   - Impact: Crucial for 5-10 genres

3. **Text Embeddings (768D)** - Captures semantic meaning
   - Comedy vs Drama tone detection
   - Crime/thriller vocabulary ("detective", "murder", "investigation")
   - Genre-specific keywords ("stand-up", "documentary")
   - Impact: Essential for all genres

#### 📊 Feature Impact by Genre Type

**Type-dominant genres** (type feature is decisive):
- TV Dramas, TV Comedies, Kids' TV, Crime TV Shows
- These achieve high F1 when `type` matches

**Rating-dominant genres** (rating feature is decisive):
- Kids' TV (TV-Y/TV-Y7), Children & Family Movies (G/PG)
- Horror Movies (R/TV-MA) vs Kids content

**Text-dominant genres** (embeddings are decisive):
- Stand-Up Comedy (vocabulary: "comedian", "jokes", "laugh")
- Documentaries (vocabulary: "explores", "history", "real")
- Sports Movies (vocabulary: "team", "championship", "athlete")

**Multi-feature genres** (need all three):
- Most genres require combination of type + rating + text
- E.g., "Romantic TV Shows" needs type=TV + romantic keywords

### Key Learnings

1. **Class imbalance is the main challenge**
   - 203:1 ratio between most/least common genres
   - Genres with <50 samples achieve F1 ≈ 0.00-0.30
   - Solution: class weighting + threshold tuning improved F1 by 337%

2. **Threshold optimization is critical**
   - Default 0.5 threshold: F1 Macro = 0.09
   - CV-tuned 0.15 threshold: F1 Macro = 0.38
   - Proper CV methodology avoids test set leakage

3. **Genre labels are subjective**
   - "Dramas" + "International Movies" co-occur 320 times
   - Model "confusion" often reflects legitimate ambiguity
   - Perfect prediction is impossible due to labeling subjectivity

4. **Feature engineering matters**
   - Text alone insufficient (many genres need type/rating)
   - Embeddings capture semantic nuance better than TF-IDF
   - Categorical features provide strong structural signals

---

## ⚡ Using the Makefile

For convenience, a Makefile is provided with common commands:

```bash
# View all available commands
make help

# Install dependencies (includes dev tools like ruff)
make install

# Run the pipeline
make preprocess       # Generate embeddings and features
make train            # Train and evaluate models
make all              # Run full pipeline (preprocess + train)

# Code quality
make lint             # Check code with ruff
make format           # Auto-format code with ruff

# Cleanup
make clean            # Remove processed_data/ and cache files
```

**Quick workflow:**
```bash
# First time setup
make install
ollama pull embeddinggemma

# Run full pipeline
make all

# Clean and re-run
make clean && make all
```

---

## 📊 Understanding the Metrics

### Multi-Label Classification Metrics

- **Hamming Loss** (lower is better): Fraction of labels incorrectly predicted
- **Exact Match Ratio**: Percentage where ALL genres are perfectly predicted
- **F1 Macro**: Average F1 across all genres (primary metric)
- **F1 Micro**: Overall F1 across all predictions
- **F1 Samples**: Average F1 per sample

**For this problem, F1 Macro is most meaningful** as it treats rare genres equally with common ones.

---

## 💡 Key Insights & Findings

### What Makes This Problem Interesting

1. **Genre Ambiguity**: Many titles legitimately belong to multiple genres
   - "Dramas" + "International Movies" appears together 320 times
   - Model confusion reveals subjective labeling decisions

2. **Class Imbalance**: Real-world challenge
   - Most common: "International Movies" (2,437 items)
   - Least common: "TV Shows" (12 items)
   - 203:1 imbalance ratio

3. **Feature Impact**:
   - `type` (Movie vs TV) is nearly perfect for some genres (e.g., "TV Dramas")
   - Embeddings capture semantic similarity ("zombie apocalypse" ≈ "undead outbreak")
   - `rating` helps distinguish Kids content from Horror

4. **Model Performance**:
   - Best approach: Random Forest with class weighting + CV-tuned threshold (0.15)
   - High-frequency genres: F1 = 0.60-0.87 (Stand-Up Comedy, Kids' TV, Crime Shows)
   - Low-frequency genres: F1 = 0.00-0.30 (insufficient training data)
   - Overall F1 Macro: 0.38 (reasonable for 42-class imbalanced multi-label)
   - Final model outperforms Logistic Regression baseline by 45%

---

## 🎓 Technical Details

### Feature Engineering

**Text Features (768 dimensions):**
- Semantic embeddings from Google's embeddinggemma
- Captures contextual meaning and synonyms

**Categorical Features (17 dimensions):**
- `type`: One-hot encoded (Movie, TV Show)
- `rating`: One-hot encoded (TV-MA, PG, TV-Y7, etc.)

**Why these features?**
- Genre labels are often type-specific ("TV Dramas" vs "Dramas")
- Rating correlates with content type (Kids content vs Horror)
- Country and duration left out to avoid noise (high cardinality)

### Model Architecture

**Baseline: Logistic Regression**
- OneVsRest strategy (42 binary classifiers)
- Fast training
- Good interpretability

**Advanced: Random Forest**
- Handles non-linear patterns
- Better performance

---

## 🐛 Troubleshooting

### Ollama Connection Issues
```bash
# Ensure Ollama is running
ollama serve

# Test connection
ollama list
```

### Memory Issues
```python
# Use smaller batches in preprocessing
preprocessor.generate_embeddings(texts, batch_size=10)

# Or use fewer trees in Random Forest
RandomForestClassifier(n_estimators=50)
```

---

## 📖 References

### Dataset
- Netflix Movies and TV Shows dataset from Kaggle
- Source: [kaggle.com/datasets/imtkaggleteam/netflix](https://www.kaggle.com/datasets/imtkaggleteam/netflix)

### Embedding Model
- embeddinggemma by Google (300M parameters)
- Deployed via Ollama
- [ollama.com/library/embeddinggemma](https://ollama.com/library/embeddinggemma)

### AI Assistance
Assistance provided by Claude (Anthropic) for: (1) Netflix dataset feasibility analysis and problem identification, (2) feature engineering guidance comparing TF-IDF and embedding approaches, (3) complete Python implementation of preprocessing pipeline using Ollama embeddinggemma for text embeddings, and (4) multi-label classification training framework, 10/25/2025.

