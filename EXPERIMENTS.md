# Experimental Log - Netflix Genre Prediction

This document tracks experiments, findings, and model improvements for the Netflix multi-label genre classification project.

---

## Experiment 1: Baseline Models (Initial Implementation)

**Date:** 2025-10-25

**Goal:** Establish baseline performance with Logistic Regression and Random Forest

### Configuration

**Logistic Regression:**
```python
LogisticRegression(max_iter=1000, random_state=42, C=1.0, solver='lbfgs')
wrapped in MultiOutputClassifier
```

**Random Forest:**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
wrapped in MultiOutputClassifier
```

### Results

| Model | F1 Macro | F1 Micro | Exact Match | Hamming Loss |
|-------|----------|----------|-------------|--------------|
| Logistic Regression | 0.2604 | 0.5696 | 19.26% | 0.0357 |
| **Random Forest** | **0.0433** | **0.1655** | **1.93%** | **0.0484** |

### Diagnostic Findings

**Critical Issue:** Random Forest severe underprediction due to class imbalance

**Prediction Distribution:**
- True: 2.19 labels/sample
- Logistic Regression: 1.30 labels/sample (59.6% of true)
- **Random Forest: 0.25 labels/sample (11.4% of true)** ⚠️

**Zero Predictions:**
- LR: 224/1558 samples (14%)
- **RF: 1250/1558 samples (80%)** ⚠️

**Probability Analysis:**
- Mean RF probability: 0.057 (vs 0.5 threshold)
- Median: 0.020
- Most predictions cluster at 0.01-0.10, well below 0.5 threshold

**Threshold Analysis (on test set - for diagnostic purposes only):**
```
Threshold 0.1: 6.91 labels/sample
Threshold 0.2: 3.12 labels/sample
Threshold 0.3: 1.68 labels/sample ← closest to true 2.19
Threshold 0.4: 0.77 labels/sample
Threshold 0.5: 0.25 labels/sample ← current
```

### Root Cause Analysis

1. **Class Imbalance (203:1 ratio)**
   - Most common genre: "International Movies" (2,437 instances)
   - Least common genre: "TV Shows" (12 instances)
   - RF optimizes for accuracy → predicts mostly zeros

2. **No Class Weighting**
   - RF treats all samples equally
   - Predicting "no genre X" is correct >95% of the time for rare genres
   - Model learns to be overly conservative

3. **Probability Calibration**
   - RF probabilities are poorly calibrated for imbalanced data
   - Mean probability 0.057 << 0.5 threshold
   - Even common genres rarely exceed 0.5

### Conclusions

- ✅ Logistic Regression provides reasonable baseline (F1 macro: 0.26)
- ❌ Random Forest fails catastrophically without class weighting
- 🎯 **Next step:** Add class_weight='balanced_subsample' to RF
- ⚠️ **Note:** Threshold tuning observed on test set - DO NOT use this for model selection (data leakage)

---

## Experiment 2: Random Forest with Class Weighting

**Date:** 2025-10-25

**Goal:** Address class imbalance using sklearn's built-in class weighting

### Hypothesis
Adding `class_weight='balanced_subsample'` will:
- Increase prediction probabilities for rare classes
- Reduce zero predictions from 80% to <20%
- Improve F1 macro from 0.04 to 0.30-0.40

### Configuration Changes

**Random Forest (Modified):**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced_subsample',  # ← NEW
    random_state=42,
    n_jobs=-1
)
```

**What `balanced_subsample` does:**
- Reweights samples in each tree's bootstrap sample
- Rare classes get higher weight → model penalizes errors on them more
- Each tree sees different class distribution
- Increases diversity in ensemble

### Validation Approach

✅ **No data leakage concerns:**
- Class weights computed from training data only
- Test set never touched during training
- Threshold remains at default 0.5
- Valid comparison to Experiment 1

### Results

| Metric | Exp 1 (No Weighting) | Exp 2 (Class Weighted) | Change |
|--------|---------------------|------------------------|--------|
| **F1 Macro** | 0.0433 | **0.0865** | **+100% (2x)** ✅ |
| **F1 Micro** | 0.1655 | **0.2444** | **+48%** ✅ |
| **Exact Match** | 1.93% | **5.20%** | **+169%** ✅ |
| **Hamming Loss** | 0.0484 | **0.0463** | **-4%** ✅ |
| **Avg Labels/Sample** | 0.25 | **0.39** | **+56%** ✅ |
| **Zero Predictions** | 1250/1558 (80%) | **1122/1558 (72%)** | **-8%** ✅ |

**Top Performing Genres (Exp 2):**
1. International Movies: F1 = 0.548 (was 0.477 in Exp 1)
2. Stand-Up Comedy: F1 = 0.474 (was 0.130 in Exp 1) 🎯
3. Dramas: F1 = 0.401 (was 0.312 in Exp 1)
4. Kids' TV: F1 = 0.400 (was 0.333 in Exp 1)

**Probability Distribution (Exp 2):**
- Mean: 0.056 (unchanged from Exp 1: 0.057)
- Median: 0.020 (same as Exp 1)
- Still clustering far below 0.5 threshold

**Threshold Analysis (Diagnostic - NOT for model selection):**
- Threshold 0.5: 0.39 labels/sample (current)
- **Threshold 0.3: 1.91 labels/sample** ← closer to true 2.19
- Threshold 0.2: 3.28 labels/sample (too high)

### Analysis

**✅ Improvements Achieved:**
1. **F1 Macro doubled** (0.043 → 0.087) - class weighting working as intended
2. **All metrics improved** - no regressions
3. **Zero predictions reduced** by 128 samples (8% improvement)
4. **Genre-specific gains**: Stand-Up Comedy improved 3.6x (0.13 → 0.47)

**⚠️ Fundamental Issues Remain:**
1. **Still severely underpredicting**: 0.39 vs 2.19 true labels/sample (**18% of expected**)
2. **72% of samples get zero predictions** (down from 80%, but still too high)
3. **Probability calibration unchanged**: Mean 0.056 << 0.5 threshold
4. **Still worse than Logistic Regression**: RF F1 0.09 vs LR F1 0.26

**🔍 Root Cause:**
Class weighting increased the *relative* probabilities for rare classes, but the *absolute* probabilities remain very low (mean 0.056). The 0.5 threshold is fundamentally too high for this imbalanced dataset with Random Forest.

**📊 Evidence:**
- Threshold analysis shows 0.3 would yield 1.91 labels/sample (87% of true)
- But we cannot use test set to tune threshold (data leakage)

### Conclusions

**What worked:**
- ✅ Class weighting improved all metrics
- ✅ Validates our diagnostic approach
- ✅ No data leakage - proper experimental design

**What didn't work:**
- ❌ Class weighting alone insufficient to fix RF underprediction
- ❌ Probability calibration remains poor

**Next Steps:**
1. **Implement cross-validation threshold tuning** (Option B from plan)
   - Use CV on training set to find optimal threshold
   - Avoid test set peeking
2. **Consider hyperparameter tuning**
   - Reduce max_depth to prevent overfitting
   - Try more trees (n_estimators=200-300)
3. **Alternative: Use Logistic Regression**
   - Already achieving F1=0.26 (3x better than RF)
   - Simpler, faster, more interpretable

---

## Experiment 3: CV-Based Threshold Tuning (Kaggle Best Practice)

**Date:** 2025-10-25

**Goal:** Optimize prediction threshold using cross-validation on training set WITHOUT test set peeking

### Motivation

Experiment 2 showed:
- Class weighting improved metrics but insufficient
- Mean probability (0.056) << default threshold (0.5)
- Diagnostic analysis suggests threshold 0.3 would be better
- **But cannot use test set for threshold tuning (data leakage)**

### Hypothesis

Using cross-validation on training set to find optimal threshold will:
- Maintain valid experimental design (no test set peeking)
- Find optimal threshold that balances precision/recall
- Improve F1 macro from 0.09 → 0.20-0.30

### Approach: Cross-Validation Threshold Optimization

**Method:**
```python
# Step 1: Train RF with class_weight on FULL training set
model.fit(X_train, y_train)

# Step 2: Use CV on training set to find optimal threshold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for fold in cv.split(X_train, y_train):
    # Get probabilities on validation fold
    y_proba = model.predict_proba(X_val_fold)
    
    # Try thresholds from 0.1 to 0.5
    # Find threshold that maximizes F1 macro on this fold
    
# Average optimal threshold across folds

# Step 3: Apply optimal threshold to test set predictions
# This is the ONLY time we touch test set
y_test_proba = model.predict_proba(X_test)
y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
```

**Global vs Per-Genre Threshold:**
- Start with **global threshold** (same for all genres) - simpler
- If insufficient, try per-genre thresholds in future experiment

**Validation Strategy:**
✅ **No data leakage:**
- Threshold tuned using CV on training set only
- Test set never seen during threshold selection
- Valid Kaggle competition approach

### Configuration

**Random Forest (Same as Exp 2):**
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
```

**CV Parameters:**
- Method: 5-fold cross-validation
- Stratification: By total genre count per sample
- Threshold search space: [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
- Optimization metric: F1 Macro (to treat all genres equally)

### Results

| Metric | Exp 2 (threshold=0.5) | Exp 3 (CV-tuned threshold) | Change |
|--------|----------------------|---------------------------|--------|
| **F1 Macro** | 0.0865 | **0.3783** | **+337%** 🎯 |
| **F1 Micro** | 0.2444 | **0.5329** | **+118%** ✅ |
| **Exact Match** | 5.20% | 1.60% | -69% ⚠️ |
| **Hamming Loss** | 0.0463 | 0.0731 | +58% ⚠️ |
| **Avg Labels/Sample** | 0.39 | 4.39 | +1025% ⚠️ |
| **Optimal Threshold** | 0.5 (default) | **0.15** | N/A |

**CV Threshold Search Results (on training set):**
```
Threshold    F1 Macro (CV)    Std Dev
0.10         0.3369          ±0.0057
0.15         0.3513 ← BEST   ±0.0113
0.20         0.3262          ±0.0077
0.25         0.2911          ±0.0065
0.30         0.2475          ±0.0106
0.50         0.0738          ±0.0059
```

**Top Performing Genres (Exp 3):**
1. Stand-Up Comedy: F1 = 0.868 (was 0.474 in Exp 2)
2. Kids' TV: F1 = 0.815 (was 0.400 in Exp 2)
3. Crime TV Shows: F1 = 0.723 (was 0.178 in Exp 2)
4. Children & Family Movies: F1 = 0.707 (was 0.0 in Exp 2!)
5. International TV Shows: F1 = 0.688 (was 0.191 in Exp 2)

**Comparison to Logistic Regression:**
- LR F1 Macro: 0.2604
- **Exp 3 F1 Macro: 0.3783** ← RF now BEATS LR! ✅

### Analysis

**✅ Major Breakthrough:**
1. **F1 Macro improved 4.4x** (0.0865 → 0.3783)
2. **Now outperforms Logistic Regression** by 45% (0.378 vs 0.260)
3. **CV approach is valid** - no test set peeking, proper Kaggle methodology
4. **Threshold 0.15 found via 5-fold CV** on training set

**⚠️ Trade-offs Observed:**
1. **Now overpredicting**: 4.39 labels/sample vs true 2.19 (201% of expected)
2. **Exact match decreased**: 5.2% → 1.6% (predicting too many genres)
3. **Hamming loss increased**: More label errors overall
4. **Precision vs Recall shift**: Higher recall, lower precision

**🔍 Root Cause of Trade-off:**
- Threshold 0.15 optimizes for F1 Macro (treats all genres equally)
- This favors recall for rare genres → predicts more labels
- F1 Macro doesn't penalize overprediction as heavily as exact match
- Trade-off between different metrics based on optimization goal

**📊 Validation of Approach:**
- CV predicted F1=0.3513, actual test F1=0.3783 ← **good generalization!**
- Only 7.7% difference between CV estimate and test performance
- Proves CV-based tuning is working correctly

### Conclusions

**What worked exceptionally well:**
- ✅ CV-based threshold tuning is the correct approach
- ✅ Achieved 337% improvement in F1 Macro
- ✅ RF with class_weight + optimal threshold beats LR baseline
- ✅ No data leakage - proper experimental methodology
- ✅ CV estimates aligned with test performance

**What needs consideration:**
- ⚠️ Overprediction (4.39 vs 2.19 labels/sample)
- ⚠️ May want to optimize for different metric (F1 samples instead of F1 macro)
- ⚠️ Could try threshold between 0.15-0.25 for better balance

**Key Learning:**
**The choice of optimization metric matters!**
- F1 Macro → Optimizes for rare classes → More predictions
- F1 Samples → Optimizes per-sample accuracy → Fewer predictions
- Exact Match → Optimizes for perfect predictions → Conservative

**Final Model Performance:**
```
Model: Random Forest (class_weight='balanced_subsample', threshold=0.15)
F1 Macro: 0.3783
F1 Micro: 0.5329
F1 Samples: 0.5242
```

**Next Steps:**
1. **Try different CV optimization metrics:**
   - Optimize for F1 Samples instead of F1 Macro
   - Might find threshold closer to 0.2-0.25
   - Better balance precision/recall

2. **Per-genre threshold tuning:**
   - Different optimal threshold for each genre
   - More complex but could improve further

3. **Hyperparameter tuning:**
   - n_estimators, max_depth with GridSearchCV
   - Combined with threshold tuning

4. **Accept current model as final:**
   - 0.378 F1 Macro is strong for this imbalanced dataset
   - 45% better than LR baseline
   - Proper methodology validates results

---

## Future Experiments (Planned)

### Experiment 3: Hyperparameter Tuning with Cross-Validation

**Approach:** GridSearchCV or RandomizedSearchCV on training set only

**Parameters to tune:**
```python
param_grid = {
    'estimator__n_estimators': [100, 200, 300],
    'estimator__max_depth': [10, 15, 20],
    'estimator__min_samples_split': [5, 10, 15],
    'estimator__min_samples_leaf': [2, 5, 10],
}
```

**Validation strategy:**
- 5-fold cross-validation on training set
- Stratified split to maintain genre distribution
- Optimize for F1 macro score

### Experiment 4: Threshold Tuning (If Needed)

**Approach:** Cross-validation to find optimal threshold WITHOUT test set peeking

**Method:**
```python
# Option A: Create validation split
X_train_new, X_val, y_train_new, y_val = train_test_split(
    X_train, y_train, test_size=0.25, random_state=42
)
# Tune threshold on validation set
# Evaluate on original test set

# Option B: Cross-validation threshold tuning
# Use CV folds on training set to find optimal threshold per genre
# Never touch test set until final evaluation
```

**Per-genre vs global threshold:**
- Try global threshold first (simpler)
- If needed, optimize per-genre thresholds for rare classes

### Experiment 5: Alternative Models

**To try:**
- Gradient Boosting (XGBoost, LightGBM)
- Neural network (MLPClassifier)
- Ensemble combining LR + RF predictions

---

## Best Practices Checklist

✅ **Data Splitting:**
- [ ] Train/validation/test split created
- [x] Same random_state for reproducibility
- [x] Test set never used for model selection

✅ **Class Imbalance:**
- [x] Diagnostic analysis performed
- [ ] Class weighting implemented
- [ ] Results documented and compared

✅ **Model Selection:**
- [x] Baseline established
- [ ] Cross-validation for hyperparameters
- [ ] Validation set for threshold tuning
- [ ] Final evaluation on held-out test set

✅ **Documentation:**
- [x] Experimental log maintained
- [x] Hypotheses stated before experiments
- [ ] Results documented after each experiment
- [ ] Reproducible with random seeds

---

## Key Learnings

1. **Always check prediction distributions** - F1 score alone didn't reveal that RF was predicting almost nothing
2. **Class imbalance requires class weighting** - Standard RF fails on imbalanced multi-label data
3. **Probability calibration matters** - RF probabilities on imbalanced data cluster far below 0.5
4. **Avoid test set peeking** - Use validation set or CV for threshold tuning, not test set

---

## References

- Scikit-learn class_weight documentation: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
- Multi-label classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#multilabel-ranking-metrics
- Kaggle best practices for imbalanced data
