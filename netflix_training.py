"""
Usage Guide and Training Example for Netflix Genre Prediction
Shows how to use the preprocessed data for model training
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
)
from sklearn.multioutput import MultiOutputClassifier

# ============================================================================
# 1. LOADING PREPROCESSED DATA
# ============================================================================


def load_data(data_dir="processed_data"):
    """Load all preprocessed data"""
    data_dir = Path(data_dir)

    # Load feature and target matrices
    X_train = np.load(data_dir / "X_train.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_train = np.load(data_dir / "y_train.npy")
    y_test = np.load(data_dir / "y_test.npy")

    # Load configuration
    with open(data_dir / "config.json") as f:
        config = json.load(f)

    # Load sklearn objects
    with open(data_dir / "preprocessor.pkl", "rb") as f:
        sklearn_objects = pickle.load(f)

    return X_train, X_test, y_train, y_test, config, sklearn_objects


# ============================================================================
# 2. MULTI-LABEL CLASSIFICATION METRICS
# ============================================================================


def evaluate_multilabel(y_true, y_pred, genre_names):
    """Comprehensive evaluation for multi-label classification"""

    results = {}

    # Overall metrics
    results["hamming_loss"] = hamming_loss(y_true, y_pred)
    results["exact_match_ratio"] = accuracy_score(y_true, y_pred)
    results["f1_micro"] = f1_score(y_true, y_pred, average="micro")
    results["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    results["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")
    results["f1_samples"] = f1_score(y_true, y_pred, average="samples")

    # Per-genre metrics
    per_genre_f1 = f1_score(y_true, y_pred, average=None)

    genre_performance = pd.DataFrame(
        {"genre": genre_names, "f1_score": per_genre_f1, "support": y_true.sum(axis=0)}
    ).sort_values("f1_score", ascending=False)

    results["per_genre"] = genre_performance

    return results


def print_evaluation(results):
    """Pretty print evaluation results"""
    print("\n" + "=" * 60)
    print("MULTI-LABEL CLASSIFICATION RESULTS")
    print("=" * 60)

    print("\n📊 Overall Metrics:")
    print(f"  Hamming Loss:        {results['hamming_loss']:.4f} (lower is better)")
    print(f"  Exact Match Ratio:   {results['exact_match_ratio']:.4f}")
    print(f"  F1 Score (micro):    {results['f1_micro']:.4f}")
    print(f"  F1 Score (macro):    {results['f1_macro']:.4f}")
    print(f"  F1 Score (weighted): {results['f1_weighted']:.4f}")
    print(f"  F1 Score (samples):  {results['f1_samples']:.4f}")

    print("\n📈 Top 10 Best-Performing Genres:")
    print(results["per_genre"].head(10).to_string(index=False))

    print("\n📉 Bottom 10 Worst-Performing Genres:")
    print(results["per_genre"].tail(10).to_string(index=False))


# ============================================================================
# 3. BASELINE MODEL: LOGISTIC REGRESSION
# ============================================================================


def train_baseline_model(X_train, y_train, X_test, y_test, genre_names):
    """
    Train a baseline logistic regression model
    Uses OneVsRest strategy for multi-label classification
    """
    print("\n" + "=" * 60)
    print("TRAINING BASELINE MODEL: Logistic Regression")
    print("=" * 60)

    # Create multi-label classifier
    model = MultiOutputClassifier(
        LogisticRegression(max_iter=1000, random_state=42, C=1.0, solver="lbfgs"),
        n_jobs=-1,  # Use all CPU cores
    )

    print("Training...")
    model.fit(X_train, y_train)

    print("Predicting on test set...")
    y_pred = model.predict(X_test)

    # Evaluate
    results = evaluate_multilabel(y_test, y_pred, genre_names)
    print_evaluation(results)

    return model, results


# ============================================================================
# 4. ADVANCED MODEL: RANDOM FOREST
# ============================================================================


def train_random_forest(X_train, y_train, X_test, y_test, genre_names):
    """
    Train a Random Forest model
    Generally performs better than logistic regression for complex patterns
    """
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST MODEL")
    print("=" * 60)

    model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        ),
        n_jobs=1,  # RandomForest already uses n_jobs
    )

    print("Training (this may take a few minutes)...")
    model.fit(X_train, y_train)

    print("Predicting on test set...")
    y_pred = model.predict(X_test)

    # Evaluate
    results = evaluate_multilabel(y_test, y_pred, genre_names)
    print_evaluation(results)

    return model, results


# ============================================================================
# 5. ERROR ANALYSIS
# ============================================================================


def analyze_errors(X_test, y_test, y_pred, df_test, genre_names):
    """
    Analyze where the model fails to understand genre ambiguity
    """
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    # Find exact mismatches (predictions that don't match ground truth at all)
    exact_errors = (y_test != y_pred).all(axis=1)
    n_exact_errors = exact_errors.sum()

    print(
        f"\nTotal exact mismatches: {n_exact_errors}/{len(y_test)} ({n_exact_errors / len(y_test) * 100:.1f}%)"
    )

    # Find partial errors (some genres right, some wrong)
    partial_errors = (y_test != y_pred).any(axis=1) & ~exact_errors
    n_partial_errors = partial_errors.sum()

    print(
        f"Partial errors: {n_partial_errors}/{len(y_test)} ({n_partial_errors / len(y_test) * 100:.1f}%)"
    )

    # Most confused genre pairs
    print("\n🔀 Most Confused Genre Pairs:")

    confusion_pairs = []
    for i, genre in enumerate(genre_names):
        # False positives: predicted but not true
        fp_idx = (y_pred[:, i] == 1) & (y_test[:, i] == 0)

        if fp_idx.sum() > 0:
            # What genres were actually true when this was falsely predicted?
            actual_genres = y_test[fp_idx]
            for j, other_genre in enumerate(genre_names):
                if i != j:
                    overlap = (actual_genres[:, j] == 1).sum()
                    if overlap > 5:  # At least 5 occurrences
                        confusion_pairs.append(
                            {
                                "predicted": genre,
                                "actual": other_genre,
                                "count": overlap,
                            }
                        )

    confusion_df = pd.DataFrame(confusion_pairs).sort_values("count", ascending=False)
    print(confusion_df.head(20).to_string(index=False))

    # Show example errors
    print("\n📋 Example Misclassified Items:")
    error_indices = np.where(exact_errors)[0][:5]

    for idx in error_indices:
        true_genres = [
            genre_names[i] for i in range(len(genre_names)) if y_test[idx, i] == 1
        ]
        pred_genres = [
            genre_names[i] for i in range(len(genre_names)) if y_pred[idx, i] == 1
        ]

        # Get description from test dataframe
        test_idx = df_test.index[idx]
        desc = df_test.loc[test_idx, "description"]
        title = df_test.loc[test_idx, "title"]

        print(f"\nTitle: {title}")
        print(f"Description: {desc[:100]}...")
        print(f"True genres: {', '.join(true_genres)}")
        print(f"Predicted genres: {', '.join(pred_genres)}")


# ============================================================================
# 6. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Load preprocessed data
    print("Loading preprocessed data...")
    X_train, X_test, y_train, y_test, config, sklearn_objects = load_data(
        "processed_data"
    )
    genre_names = config["genre_names"]

    # Load test dataframe for error analysis
    df_test = pd.read_csv("processed_data/processed_df.csv")
    # Get test indices (assuming same random state for split)
    from sklearn.model_selection import train_test_split

    _, df_test_subset = train_test_split(df_test, test_size=0.2, random_state=42)
    df_test_subset = df_test_subset.reset_index(drop=True)

    print("\nDataset Info:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Genres: {y_train.shape[1]}")

    # Train baseline model
    lr_model, lr_results = train_baseline_model(
        X_train, y_train, X_test, y_test, genre_names
    )

    # Train random forest model
    rf_model, rf_results = train_random_forest(
        X_train, y_train, X_test, y_test, genre_names
    )

    # Compare models
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"Logistic Regression F1 (macro): {lr_results['f1_macro']:.4f}")
    print(f"Random Forest F1 (macro):       {rf_results['f1_macro']:.4f}")

    # Error analysis on best model
    best_model = (
        rf_model if rf_results["f1_macro"] > lr_results["f1_macro"] else lr_model
    )
    best_y_pred = best_model.predict(X_test)

    analyze_errors(X_test, y_test, best_y_pred, df_test_subset, genre_names)

    print("\n✅ Analysis complete!")


# ============================================================================
# 7. QUICK START EXAMPLE
# ============================================================================

"""
QUICK START GUIDE
=================

1. First, run the preprocessing pipeline:
   python netflix_preprocessing.py

   This will:
   - Generate embeddings using Ollama
   - Encode genres and categorical features
   - Save processed data to processed_data/

2. Then train and evaluate models:
   python netflix_training.py

   This will:
   - Load preprocessed data
   - Train baseline (Logistic Regression)
   - Train Random Forest
   - Compare results
   - Analyze errors

3. Load and use a trained model:

   ```python
   # Load data
   X_train, X_test, y_train, y_test, config, sklearn_obj = load_data()

   # Train your model
   model = MultiOutputClassifier(LogisticRegression())
   model.fit(X_train, y_train)

   # Make predictions
   predictions = model.predict(X_test)

   # Evaluate
   results = evaluate_multilabel(y_test, predictions, config['genre_names'])
   print_evaluation(results)
   ```

4. Predict genres for new descriptions:

   ```python
   import ollama

   # Get embedding for new description
   new_desc = "A thrilling zombie apocalypse movie"
   response = ollama.embed(model='embeddinggemma', input=new_desc)
   embedding = np.array(response['embeddings'][0])

   # Add categorical features (e.g., Movie, TV-MA)
   # ... encode type and rating using saved cat_encoder ...

   # Predict
   prediction = model.predict([combined_features])
   predicted_genres = [genre_names[i] for i in range(len(genre_names))
                       if prediction[0, i] == 1]
   print(f"Predicted genres: {predicted_genres}")
   ```

KEY METRICS EXPLAINED
=====================

- Hamming Loss: Fraction of labels incorrectly predicted (lower is better)
- Exact Match Ratio: Percentage where ALL genres are correct
- F1 Micro: Overall F1 treating all genre predictions equally
- F1 Macro: Average F1 across genres (treats rare genres equally)
- F1 Weighted: Average F1 weighted by genre frequency
- F1 Samples: Average F1 per sample (how well we predict all genres for each item)

For multi-label classification, F1 Macro is often most meaningful as it
shows how well the model works across ALL genres, not just common ones.
"""
