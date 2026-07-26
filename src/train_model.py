import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from src.features import generate_features

# Regressor hyperparameters — tuned via 5-fold CV grid search (see
# PROGRESS.md "Regressor tuning" for the search log). The untuned defaults
# combined with a single 80/20 split gave a misleadingly noisy R2 (as low as
# 0.106 on an unlucky split, with only ~43 test rows) — 5-fold CV showed the
# untuned model was really closer to R2=0.25, and this shallower/more
# regularized config pushes that to a more stable ~0.30. Not shallower out of
# guesswork: with only 217 training rows and 44 features, the untuned
# max_depth=3 config was mildly overfitting, and CV variance dropped from
# +/-0.16 to +/-0.08 with these settings.
REGRESSOR_PARAMS = dict(n_estimators=120, max_depth=2, learning_rate=0.04, subsample=0.8, random_state=42)
CLASSIFIER_PARAMS = dict(n_estimators=100, random_state=42)


def train_and_evaluate():
    """
    Train and save the Classifier (off-spec) and Regressor (stabilization time).
    Uses GradientBoosting models per the architecture constraint.

    Reports 5-fold cross-validated metrics rather than a single train/test
    split: with only ~217 valid training rows, a single 80/20 split leaves
    just ~43 test rows, and the resulting R2/accuracy can swing wildly
    depending on which rows happen to land in the test set (verified
    empirically — single-split R2 for the regressor ranged from 0.08 to 0.53
    across different folds of the same data). CV reports the honest,
    stable picture instead of one lucky or unlucky split.

    The final saved models are fit on ALL valid rows (not held-out from an
    80/20 split), since with a dataset this small, throwing away 20% of it
    for the deployed model is a real cost — the CV score above is what
    tells you how good that final model actually is, not a held-out test set.
    """
    df = generate_features()
    
    # Filter out early_breach events for training
    df_valid = df[df['early_breach'] == False].copy()
    print(f"Total events: {len(df)}, Valid for training (no early breach): {len(df_valid)}")
    
    # Define targets
    y_class = df_valid['offspec_label']
    y_reg = df_valid['stabilization_time_seconds']
    
    # Define features
    X = df_valid.drop(columns=[
        'event_id', 
        'offspec_label', 
        'stabilization_time_seconds', 
        'early_breach'
    ])

    print("\n--- Classifier (Offspec Risk): 5-fold CV ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_cv = GradientBoostingClassifier(**CLASSIFIER_PARAMS)
    acc_scores = cross_val_score(clf_cv, X, y_class, cv=skf, scoring='accuracy')
    auc_scores = cross_val_score(clf_cv, X, y_class, cv=skf, scoring='roc_auc')
    print(f"Accuracy: {acc_scores.mean():.3f} +/- {acc_scores.std():.3f}  (folds: {acc_scores.round(3)})")
    print(f"ROC-AUC:  {auc_scores.mean():.3f} +/- {auc_scores.std():.3f}  (folds: {auc_scores.round(3)})")

    print("\n--- Regressor (Stabilization Time): 5-fold CV ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    reg_cv = GradientBoostingRegressor(**REGRESSOR_PARAMS)
    r2_scores = cross_val_score(reg_cv, X, y_reg, cv=kf, scoring='r2')
    mae_scores = -cross_val_score(reg_cv, X, y_reg, cv=kf, scoring='neg_mean_absolute_error')
    print(f"R2:  {r2_scores.mean():.3f} +/- {r2_scores.std():.3f}  (folds: {r2_scores.round(3)})")
    print(f"MAE: {mae_scores.mean():.1f}s +/- {mae_scores.std():.1f}s")

    # Fit FINAL models on all valid data for deployment
    print("\n--- Fitting final models on all valid data for deployment ---")
    clf = GradientBoostingClassifier(**CLASSIFIER_PARAMS)
    clf.fit(X, y_class)

    reg = GradientBoostingRegressor(**REGRESSOR_PARAMS)
    reg.fit(X, y_reg)
    
    # Save models
    os.makedirs('models', exist_ok=True)
    clf_path = os.path.join('models', 'offspec_classifier.pkl')
    reg_path = os.path.join('models', 'stabilization_regressor.pkl')
    
    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)
    
    print(f"Models saved to:\n- {clf_path}\n- {reg_path}")
    
    # Print top features (just for debugging/visibility)
    clf_importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    reg_importances = pd.Series(reg.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    print("\nTop 5 Features for Classifier:")
    print(clf_importances.head(5))
    
    print("\nTop 5 Features for Regressor:")
    print(reg_importances.head(5))


if __name__ == "__main__":
    train_and_evaluate()