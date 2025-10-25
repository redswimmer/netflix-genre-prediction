# Netflix Genre Prediction

A multi-label text classification project that predicts Netflix content genres from descriptions using semantic embeddings and machine learning.

---

## 🎯 Project Overview

This data mining project tackles the challenge of predicting multiple genre labels for Netflix movies and TV shows based solely on their text descriptions. The system uses state-of-the-art embedding models to capture semantic meaning and trains multi-label classifiers to handle the complexity of content that spans multiple genres.

**Key Features:**
- 🤖 Semantic text embeddings using Ollama's embeddinggemma (768 dimensions)
- 🏷️ Multi-label classification supporting 42 different genres
- 📊 Comprehensive evaluation metrics for multi-label tasks
- 🔍 Error analysis revealing genre ambiguity and confusion patterns
- 🚀 Production-ready preprocessing pipeline with checkpointing
- 📈 Model comparison (Logistic Regression vs Random Forest)

**Dataset:** 7,787 Netflix titles with descriptions and genre labels

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
python netflix_preprocessing.py
```

**What happens:**
- Loads `NetFlix.csv`
- Parses 42 unique genres from multi-label format
- Generates 768-dimensional embeddings using Ollama (~3-5 minutes)
- Encodes categorical features (type: Movie/TV Show, rating: TV-MA/PG/etc.)
- Combines features into final matrix (785 dimensions)
- Creates 80/20 train/test split
- Saves all processed data to `processed_data/`

**Output:**
```
Loading data from NetFlix.csv...
Loaded 7787 rows

Parsing genres...
Average genres per item: 2.19
Total unique genres: 42

Generating embeddings using embeddinggemma...
Processing 7787 descriptions...
Embedding batches: 100%|████████| 156/156

Final feature matrix shape: (7787, 785)
  - Embeddings: 768 dimensions
  - Categorical: 17 dimensions

✅ Preprocessing complete!
```

### Step 2: Train and Evaluate Models

```bash
python netflix_training.py
```

**What happens:**
- Trains Logistic Regression (baseline, ~30 seconds)
- Trains Random Forest (advanced, ~3 minutes)
- Evaluates using multi-label metrics
- Performs error analysis
- Identifies genre confusion patterns

**Expected Results:**
```
BASELINE MODEL: Logistic Regression
  F1 Score (macro):    0.6234
  Exact Match Ratio:   0.3721

RANDOM FOREST MODEL
  F1 Score (macro):    0.6589
  Exact Match Ratio:   0.4112

Top Performing Genres:
  International Movies: F1 = 0.89
  TV Dramas:           F1 = 0.84
  Documentaries:       F1 = 0.78

Most Confused Genre Pairs:
  Predicted: International Movies → Actual: Dramas (47 cases)
  Predicted: Comedies → Actual: Romantic Movies (32 cases)
```

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
   - Easy genres: International Movies (F1 ~0.89), TV Dramas (~0.84)
   - Hard genres: Rare categories with <100 examples
   - Random Forest outperforms Logistic Regression by ~5-7% F1

---

## 🔬 Advanced Usage

### Load Preprocessed Data

```python
from netflix_training import load_data

X_train, X_test, y_train, y_test, config, sklearn_obj = load_data('processed_data')

print(f"Training samples: {X_train.shape[0]}")
print(f"Features: {X_train.shape[1]}")
print(f"Genres: {config['genre_names']}")
```

### Train Custom Model

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Create custom model
model = MultiOutputClassifier(
    GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )
)

# Train
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
```

### Predict Genres for New Content

```python
import ollama
import numpy as np

# Generate embedding for new description
new_description = "A gripping thriller about corporate espionage"
response = ollama.embed(model='embeddinggemma', input=new_description)
embedding = np.array(response['embeddings'][0])

# Encode categorical features (example: Movie, TV-MA)
# ... use saved cat_encoder from sklearn_obj ...

# Combine features and predict
combined_features = np.hstack([embedding, categorical_features])
prediction = model.predict([combined_features])

# Get predicted genres
genre_names = config['genre_names']
predicted_genres = [genre_names[i] for i in range(len(genre_names)) 
                    if prediction[0, i] == 1]
print(f"Predicted genres: {predicted_genres}")
```

---

## 🎓 Technical Details

### Feature Engineering

**Text Features (768 dimensions):**
- Semantic embeddings from Google's embeddinggemma
- Trained on 320B tokens across 100+ languages
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
- Fast training (~30 seconds)
- Good interpretability

**Advanced: Random Forest**
- 100 trees, max_depth=20
- Handles non-linear patterns
- Better performance (+5-7% F1)

---

## 🚧 Future Improvements

- [ ] Handle class imbalance with oversampling (SMOTE)
- [ ] Add country features (US vs non-US binary)
- [ ] Experiment with different embedding dimensions (512, 256)
- [ ] Try neural networks (MLPClassifier)
- [ ] Implement cross-validation for hyperparameter tuning
- [ ] Add feature importance analysis
- [ ] Create web interface for genre prediction
- [ ] Explore label correlation (which genres co-occur?)

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

