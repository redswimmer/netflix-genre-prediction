"""
Netflix Genre Prediction - Training Script
Trains and evaluates two models:
1. Logistic Regression (baseline)
2. Random Forest with class weighting and CV-tuned threshold
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, hamming_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.multioutput import MultiOutputClassifier

# ============================================================================
# 1. DATA LOADING
# ============================================================================


def load_data(data_dir="processed_data"):
    """Load preprocessed training and test data"""
    data_dir = Path(data_dir)

    print("Loading preprocessed data...")
    X_train = np.load(data_dir / "X_train.npy")
    X_test = np.load(data_dir / "X_test.npy")
    y_train = np.load(data_dir / "y_train.npy")
    y_test = np.load(data_dir / "y_test.npy")

    with open(data_dir / "config.json") as f:
        config = json.load(f)

    with open(data_dir / "preprocessor.pkl", "rb") as f:
        sklearn_objects = pickle.load(f)

    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Genres: {len(config['genre_names'])}")

    return X_train, X_test, y_train, y_test, config, sklearn_objects


# ============================================================================
# 2. EVALUATION
# ============================================================================


def evaluate_model(y_true, y_pred, genre_names):
    """Evaluate multi-label classification performance"""
    results = {}

    # Overall metrics
    results["hamming_loss"] = hamming_loss(y_true, y_pred)
    results["exact_match"] = accuracy_score(y_true, y_pred)
    results["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    results["f1_micro"] = f1_score(y_true, y_pred, average="micro")
    results["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")

    # Per-genre performance
    per_genre_f1 = f1_score(y_true, y_pred, average=None)
    genre_performance = pd.DataFrame(
        {"genre": genre_names, "f1_score": per_genre_f1, "support": y_true.sum(axis=0)}
    ).sort_values("f1_score", ascending=False)

    results["per_genre"] = genre_performance

    return results


def print_evaluation(results, model_name):
    """Pretty print evaluation results"""
    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS: {model_name}")
    print("=" * 60)

    print("\n📊 Overall Metrics:")
    print(f"  Hamming Loss:      {results['hamming_loss']:.4f}")
    print(f"  Exact Match Ratio: {results['exact_match']:.4f}")
    print(f"  F1 Macro:          {results['f1_macro']:.4f}")
    print(f"  F1 Micro:          {results['f1_micro']:.4f}")
    print(f"  F1 Weighted:       {results['f1_weighted']:.4f}")

    print("\n🏆 Top 10 Best-Performing Genres:")
    print(results["per_genre"].head(10).to_string(index=False))

    print("\n⚠️  Bottom 10 Worst-Performing Genres:")
    print(results["per_genre"].tail(10).to_string(index=False))


# ============================================================================
# 3. BASELINE MODEL: LOGISTIC REGRESSION
# ============================================================================


def train_logistic_regression(X_train, y_train):
    """
    Train baseline Logistic Regression model
    Uses OneVsRest strategy for multi-label classification
    """
    print("\n" + "=" * 60)
    print("TRAINING BASELINE: Logistic Regression")
    print("=" * 60)

    model = MultiOutputClassifier(
        LogisticRegression(max_iter=1000, random_state=42, C=1.0, solver="lbfgs"),
        n_jobs=-1,
    )

    print("\nTraining...")
    model.fit(X_train, y_train)
    print("✓ Training complete")

    return model


# ============================================================================
# 4. ADVANCED MODEL: RANDOM FOREST WITH CV-TUNED THRESHOLD
# ============================================================================


def train_random_forest(X_train, y_train):
    """
    Train Random Forest with class weighting to handle imbalance

    Configuration based on experimental results:
    - class_weight='balanced_subsample': Addresses 203:1 class imbalance
    - n_estimators=100: Good balance of performance and speed
    - max_depth=20: Allows complex patterns without severe overfitting
    """
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
    print("=" * 60)
    print("\nModel: Random Forest with class weighting")
    print("  - n_estimators: 100")
    print("  - max_depth: 20")
    print("  - class_weight: balanced_subsample")

    model = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_split=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    )

    print("\nTraining...")
    model.fit(X_train, y_train)
    print("✓ Training complete")

    return model


def tune_threshold_cv(model, X_train, y_train, cv_folds=5):
    """
    Find optimal prediction threshold using cross-validation

    Uses CV on training set only (no test set peeking) to find threshold
    that maximizes F1 Macro score. This is proper Kaggle methodology.

    Returns:
        optimal_threshold: Best threshold (typically 0.10-0.20 for imbalanced data)
        threshold_results: DataFrame with performance at each threshold
    """
    print("\n" + "=" * 60)
    print("THRESHOLD OPTIMIZATION (5-Fold Cross-Validation)")
    print("=" * 60)

    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    print(f"\nTesting thresholds: {thresholds}")
    print("Optimization metric: F1 Macro")

    # Stratify by total label count to maintain genre distribution
    stratify_labels = y_train.sum(axis=1)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    threshold_scores = {t: [] for t in thresholds}

    print("\nRunning cross-validation...")
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train, stratify_labels)):
        print(f"  Fold {fold_idx + 1}/{cv_folds}...", end=" ", flush=True)

        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

        # Train model on this fold
        model_params = model.get_params()
        model_fold = MultiOutputClassifier(
            RandomForestClassifier(**model_params["estimator"].get_params())
        )
        model_fold.fit(X_fold_train, y_fold_train)

        # Get probabilities
        proba_list = []
        for estimator in model_fold.estimators_:
            proba = estimator.predict_proba(X_fold_val)[:, 1]
            proba_list.append(proba)
        y_proba = np.column_stack(proba_list)

        # Evaluate each threshold
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            f1_macro = f1_score(y_fold_val, y_pred, average="macro", zero_division=0)
            threshold_scores[threshold].append(f1_macro)

        print("✓")

    # Calculate mean performance
    threshold_results = pd.DataFrame(
        {
            "threshold": thresholds,
            "f1_macro": [np.mean(threshold_scores[t]) for t in thresholds],
            "std": [np.std(threshold_scores[t]) for t in thresholds],
        }
    )

    optimal_idx = threshold_results["f1_macro"].idxmax()
    optimal_threshold = threshold_results.loc[optimal_idx, "threshold"]
    optimal_f1 = threshold_results.loc[optimal_idx, "f1_macro"]

    print("\nCV Results:")
    print(threshold_results.to_string(index=False))
    print(f"\n🎯 Optimal Threshold: {optimal_threshold}")
    print(
        f"   Expected F1 Macro: {optimal_f1:.4f} (±{threshold_results.loc[optimal_idx, 'std']:.4f})"
    )

    return optimal_threshold, threshold_results


def apply_threshold(model, X, threshold):
    """Apply custom threshold to model predictions"""
    proba_list = []
    for estimator in model.estimators_:
        proba = estimator.predict_proba(X)[:, 1]
        proba_list.append(proba)
    y_proba = np.column_stack(proba_list)

    return (y_proba >= threshold).astype(int)


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("NETFLIX GENRE PREDICTION - TRAINING")
    print("=" * 60)

    # Load data
    X_train, X_test, y_train, y_test, config, sklearn_objects = load_data()
    genre_names = config["genre_names"]

    # ========================================================================
    # EXPERIMENT 1: Logistic Regression Baseline
    # ========================================================================

    lr_model = train_logistic_regression(X_train, y_train)
    lr_y_pred = lr_model.predict(X_test)
    lr_results = evaluate_model(y_test, lr_y_pred, genre_names)
    print_evaluation(lr_results, "Logistic Regression")

    # ========================================================================
    # EXPERIMENT 2: Random Forest with Class Weighting + CV-Tuned Threshold
    # ========================================================================

    # Train model with class weighting
    rf_model = train_random_forest(X_train, y_train)

    # Find optimal threshold via CV (on training set only!)
    optimal_threshold, threshold_results = tune_threshold_cv(rf_model, X_train, y_train)

    # Evaluate on test set with optimal threshold
    print("\n" + "=" * 60)
    print(f"FINAL EVALUATION (threshold={optimal_threshold})")
    print("=" * 60)

    rf_y_pred = apply_threshold(rf_model, X_test, optimal_threshold)
    rf_results = evaluate_model(y_test, rf_y_pred, genre_names)
    print_evaluation(rf_results, "Random Forest")

    # ========================================================================
    # FINAL COMPARISON
    # ========================================================================

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print("\nLogistic Regression (Baseline):")
    print(f"  F1 Macro:     {lr_results['f1_macro']:.4f}")
    print(f"  F1 Micro:     {lr_results['f1_micro']:.4f}")
    print(f"  Exact Match:  {lr_results['exact_match']:.2%}")

    print(f"\nRandom Forest (class_weight + threshold={optimal_threshold}):")
    print(f"  F1 Macro:     {rf_results['f1_macro']:.4f}")
    print(f"  F1 Micro:     {rf_results['f1_micro']:.4f}")
    print(f"  Exact Match:  {rf_results['exact_match']:.2%}")

    improvement = (
        (rf_results["f1_macro"] - lr_results["f1_macro"]) / lr_results["f1_macro"]
    ) * 100
    print(f"\n📊 RF improves over LR by: {improvement:+.1f}%")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("\nFinal Model Performance:")
    print(f"  F1 Macro:     {rf_results['f1_macro']:.4f}")
    print(f"  F1 Micro:     {rf_results['f1_micro']:.4f}")
    print(f"  Exact Match:  {rf_results['exact_match']:.2%}")
    print(f"\nOptimal Threshold: {optimal_threshold}")
    print("\nModel: RandomForestClassifier(class_weight='balanced_subsample')")
    print("Methodology: 5-fold CV threshold tuning on training set")
