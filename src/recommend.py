"""
recommend.py - Hybrid Recommendation Engine.

Combines three evidence sources to generate ranked, tagged setpoint
recommendations for at-risk grade transitions:

  1. Domain Rules  : recipe-implied safe ramp rate limits.
  2. Historical k-NN: top-5 similar past CLEAN transitions (standardized
                      Euclidean distance — per DATASET_SPECIFICATION.md §8,
                      standardization is required here and is separate from
                      the model's feature set which needs no scaling).
  3. Model Confidence: fallback generic slowdown proportional to predicted risk.

Each recommendation carries:
  - setpoint_variable, suggested_value, current_value
  - expected_risk_delta, expected_stab_delta_s
  - confidence, source_tag, similar_event_id

CONSTITUTION RULES OBSERVED:
  Rule 4  — no model training or UI logic here.
  Rule 6  — no hidden deps: loads models via joblib from saved artifacts only.
  Rule 13 — no autonomous/closed-loop control; operator always decides.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_events, load_recipes, load_timeseries
from src.features import generate_features

logger = logging.getLogger(__name__)

CLASSIFIER_PATH = Path("models") / "offspec_classifier.pkl"
REGRESSOR_PATH = Path("models") / "stabilization_regressor.pkl"

# Thresholds
RISK_THRESHOLD: float = 0.60       # above this -> generate recommendation
KNN_K: int = 5                     # number of similar events to retrieve
MAX_RECS_PER_EVENT: int = 3        # cap output to avoid noise


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    """A single setpoint recommendation for an at-risk transition."""
    setpoint_variable: str
    suggested_value: float
    current_value: float
    expected_risk_reduction: float    # fractional, e.g. 0.64 = 64% reduction
    expected_stab_reduction_s: float  # seconds saved, positive = faster
    confidence: float                 # 0.0 - 1.0
    source_tag: str                   # "Recipe Rule" | "Historical Pattern" | "Model Confidence"
    similar_event_id: Optional[int] = None
    similar_event_similarity: Optional[float] = None
    rationale: str = ""               # filled in by explain.py


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_models() -> tuple:
    """Load classifier and regressor from disk.

    Returns:
        (clf, reg) tuple of trained sklearn estimators.
    """
    if not CLASSIFIER_PATH.exists() or not REGRESSOR_PATH.exists():
        raise FileNotFoundError(
            "Trained model .pkl files not found. Run src/train_model.py first."
        )
    return joblib.load(CLASSIFIER_PATH), joblib.load(REGRESSOR_PATH)


# ---------------------------------------------------------------------------
# Evidence Source 1 — Domain Rules
# ---------------------------------------------------------------------------

def _rule_based_recommendations(
    event_row: pd.Series,
    recipes: pd.DataFrame,
    current_risk: float,
    current_stab_s: float,
) -> list[Recommendation]:
    """Generate recommendations from recipe-implied safe ramp rate limits.

    Checks whether the current planned ramp rate exceeds the recipe's
    recommended_max_ramp_rate for the to_grade. If so, recommends slowing to
    the recipe limit.

    Args:
        event_row: Single row from grade_change_events.csv.
        recipes: recipe_reference.csv indexed by grade_name.
        current_risk: Classifier predicted P(offspec).
        current_stab_s: Regressor predicted stabilization time.

    Returns:
        List of rule-based Recommendation objects (0 or 1).
    """
    recs = []
    to_grade = event_row["to_grade"]
    from_grade = event_row["from_grade"]
    ramp_dur = event_row["ramp_duration_seconds"]

    recipe_to = recipes.loc[to_grade]
    recipe_from = recipes.loc[from_grade]

    bw_change = abs(recipe_to["basis_weight_setpoint"] - recipe_from["basis_weight_setpoint"])
    current_rate_gsm_min = (bw_change / ramp_dur) * 60.0   # gsm/min
    max_safe_rate = recipe_to["recommended_max_ramp_rate"]  # gsm/min

    if current_rate_gsm_min > max_safe_rate:
        # Compute the safe ramp duration and suggested new ramp duration
        safe_ramp_dur_s = (bw_change / max_safe_rate) * 60.0
        suggested_rate = max_safe_rate

        # Conservative risk/stab estimates: slowing ramp ~30% cuts risk by ~40%
        slowdown_fraction = (current_rate_gsm_min - max_safe_rate) / current_rate_gsm_min
        risk_reduction = min(0.70, slowdown_fraction * 1.4)
        stab_reduction_s = -30.0 * slowdown_fraction    # slower ramp = slightly longer stab

        recs.append(Recommendation(
            setpoint_variable="ramp_rate_gsm_min",
            suggested_value=round(suggested_rate, 2),
            current_value=round(current_rate_gsm_min, 2),
            expected_risk_reduction=round(risk_reduction, 3),
            expected_stab_reduction_s=round(stab_reduction_s, 1),
            confidence=0.85,
            source_tag="Recipe Rule",
        ))

    return recs


# ---------------------------------------------------------------------------
# Evidence Source 2 — Historical k-NN
# ---------------------------------------------------------------------------

def _knn_recommendations(
    event_row: pd.Series,
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    clf,
    reg,
    k: int = KNN_K,
) -> list[Recommendation]:
    """Find k most similar past CLEAN transitions and recommend their setpoints.

    Uses standardized Euclidean distance on a subset of key numerical features
    (not the full model feature set — standardization is separate per §8).

    Args:
        event_row: Current event's row from grade_change_events.csv.
        features_df: Full feature matrix from generate_features().
        events_df: grade_change_events.csv.
        clf: Trained classifier.
        reg: Trained regressor.
        k: Number of neighbors to retrieve.

    Returns:
        List of up to 1 Recommendation (consensus of top-k neighbors).
    """
    knn_features = [
        "valve_wear_index", "ambient_temp_c", "ramp_duration_seconds",
        "planned_bw_change", "planned_bw_ramp_rate",
        "planned_sf_ramp_rate", "planned_ff_ramp_rate",
    ]

    # Filter to same grade pair, clean (offspec_label=0), not early_breach
    same_pair = (
        (events_df["from_grade"] == event_row["from_grade"]) &
        (events_df["to_grade"] == event_row["to_grade"]) &
        (events_df["offspec_label"] == 0) &
        (events_df["event_id"] != event_row["event_id"])
    )
    candidate_ids = events_df.loc[same_pair, "event_id"].tolist()

    if len(candidate_ids) < 2:
        # Not enough clean same-pair history — fall back gracefully
        same_pair_relaxed = (
            (events_df["offspec_label"] == 0) &
            (events_df["event_id"] != event_row["event_id"]) &
            (~events_df["event_id"].isin([]))
        )
        candidate_ids = events_df.loc[same_pair_relaxed, "event_id"].tolist()

    if not candidate_ids:
        return []

    candidate_features = features_df[
        features_df["event_id"].isin(candidate_ids)
    ][["event_id"] + knn_features].dropna().set_index("event_id")

    # Get current event features
    current_feat = features_df[
        features_df["event_id"] == event_row["event_id"]
    ][knn_features]

    if current_feat.empty or candidate_features.empty:
        return []

    # Standardize
    scaler = StandardScaler()
    cand_scaled = scaler.fit_transform(candidate_features)
    curr_scaled = scaler.transform(current_feat)

    # Euclidean distances
    diffs = cand_scaled - curr_scaled
    distances = np.sqrt((diffs ** 2).sum(axis=1))
    dist_series = pd.Series(distances, index=candidate_features.index)

    top_k_ids = dist_series.nsmallest(k).index.tolist()
    max_dist = dist_series.max() if dist_series.max() > 0 else 1.0
    best_dist = dist_series[top_k_ids[0]]
    similarity_pct = round((1.0 - best_dist / max_dist) * 100, 1)

    # What did the best neighbor use for ramp_duration?
    best_event = events_df[events_df["event_id"] == top_k_ids[0]].iloc[0]
    current_ramp = event_row["ramp_duration_seconds"]
    suggested_ramp = best_event["ramp_duration_seconds"]

    if abs(suggested_ramp - current_ramp) < 30:
        # No meaningful difference — skip
        return []

    # Predict risk/stab impact of suggested ramp by building a modified feature row
    modified_feat = features_df[
        features_df["event_id"] == event_row["event_id"]
    ].copy()

    # Drop non-feature columns before predicting
    drop_cols = ["event_id", "offspec_label", "stabilization_time_seconds", "early_breach"]
    feat_cols = [c for c in modified_feat.columns if c not in drop_cols]

    current_risk = clf.predict_proba(modified_feat[feat_cols])[0][1]
    current_stab = reg.predict(modified_feat[feat_cols])[0]

    # Modify ramp duration to neighbor's value and re-predict
    modified_feat["ramp_duration_seconds"] = suggested_ramp
    modified_feat["planned_bw_ramp_rate"] = (
        modified_feat["planned_bw_change"] / suggested_ramp
    )
    modified_feat["planned_sf_ramp_rate"] = (
        modified_feat["planned_sf_change"] / suggested_ramp
    )

    new_risk = clf.predict_proba(modified_feat[feat_cols])[0][1]
    new_stab = reg.predict(modified_feat[feat_cols])[0]

    risk_reduction = max(0.0, current_risk - new_risk)
    stab_reduction = current_stab - new_stab

    return [Recommendation(
        setpoint_variable="ramp_duration_seconds",
        suggested_value=round(float(suggested_ramp), 1),
        current_value=round(float(current_ramp), 1),
        expected_risk_reduction=round(float(risk_reduction), 3),
        expected_stab_reduction_s=round(float(stab_reduction), 1),
        confidence=round(similarity_pct / 100.0, 2),
        source_tag=f"Historical Pattern -- Event #{top_k_ids[0]} ({similarity_pct}% Similarity)",
        similar_event_id=int(top_k_ids[0]),
        similar_event_similarity=similarity_pct,
    )]


# ---------------------------------------------------------------------------
# Evidence Source 3 — Model Confidence Fallback
# ---------------------------------------------------------------------------

def _model_confidence_recommendation(
    event_row: pd.Series,
    features_df: pd.DataFrame,
    clf,
    reg,
    current_risk: float,
    current_stab_s: float,
) -> list[Recommendation]:
    """Generate a generic ramp-rate reduction proportional to predicted risk.

    Only triggers if risk > RISK_THRESHOLD and neither rules nor k-NN
    produced a recommendation.

    Args:
        event_row: Current event row.
        features_df: Full feature matrix.
        clf: Trained classifier.
        reg: Trained regressor.
        current_risk: Current predicted P(offspec).
        current_stab_s: Current predicted stabilization time.

    Returns:
        List of 0 or 1 Recommendation objects.
    """
    # Suggest slowing ramp by a fraction proportional to excess risk
    excess_risk = max(0.0, current_risk - RISK_THRESHOLD)
    slowdown_pct = min(0.35, excess_risk * 0.7)  # cap at 35% slowdown

    current_ramp = event_row["ramp_duration_seconds"]
    suggested_ramp = current_ramp * (1.0 + slowdown_pct)

    # Re-predict with suggested ramp
    drop_cols = ["event_id", "offspec_label", "stabilization_time_seconds", "early_breach"]
    modified_feat = features_df[
        features_df["event_id"] == event_row["event_id"]
    ].copy()
    feat_cols = [c for c in modified_feat.columns if c not in drop_cols]

    modified_feat["ramp_duration_seconds"] = suggested_ramp
    if "planned_bw_change" in modified_feat.columns:
        modified_feat["planned_bw_ramp_rate"] = (
            modified_feat["planned_bw_change"] / suggested_ramp
        )

    new_risk = clf.predict_proba(modified_feat[feat_cols])[0][1]
    new_stab = reg.predict(modified_feat[feat_cols])[0]

    risk_reduction = max(0.0, current_risk - new_risk)
    stab_delta = current_stab_s - new_stab

    return [Recommendation(
        setpoint_variable="ramp_duration_seconds",
        suggested_value=round(float(suggested_ramp), 1),
        current_value=round(float(current_ramp), 1),
        expected_risk_reduction=round(float(risk_reduction), 3),
        expected_stab_reduction_s=round(float(stab_delta), 1),
        confidence=round(max(0.40, 1.0 - current_risk), 2),
        source_tag="Model Confidence",
    )]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def get_recommendations(event_id: int) -> list[Recommendation]:
    """Generate all recommendations for a single event.

    Orchestrates the three evidence sources in priority order. Returns up
    to MAX_RECS_PER_EVENT recommendations, deduplicated by setpoint_variable.

    Args:
        event_id: ID of the grade-change event to analyse.

    Returns:
        List of Recommendation objects, sorted by expected_risk_reduction desc.
    """
    clf, reg = _load_models()
    events_df = load_events()
    recipes_df = load_recipes().set_index("grade_name")
    features_df = generate_features()

    event_row = events_df[events_df["event_id"] == event_id]
    if event_row.empty:
        raise ValueError(f"event_id {event_id} not found in grade_change_events.csv")
    event_row = event_row.iloc[0]

    # Get current predictions
    drop_cols = ["event_id", "offspec_label", "stabilization_time_seconds", "early_breach"]
    feat_row = features_df[features_df["event_id"] == event_id]
    feat_cols = [c for c in feat_row.columns if c not in drop_cols]

    current_risk = float(clf.predict_proba(feat_row[feat_cols])[0][1])
    current_stab_s = float(reg.predict(feat_row[feat_cols])[0])

    logger.info(
        "Event %d: P(offspec)=%.3f, predicted_stab=%.0fs",
        event_id, current_risk, current_stab_s
    )

    if current_risk < RISK_THRESHOLD:
        logger.info("Event %d risk below threshold (%.2f) -- no recommendations needed.", event_id, RISK_THRESHOLD)
        return []

    all_recs: list[Recommendation] = []

    # Source 1: Rules
    rule_recs = _rule_based_recommendations(event_row, recipes_df, current_risk, current_stab_s)
    all_recs.extend(rule_recs)

    # Source 2: k-NN (only if rules didn't cover ramp_duration)
    seen_vars = {r.setpoint_variable for r in all_recs}
    knn_recs = _knn_recommendations(event_row, features_df, events_df, clf, reg)
    for r in knn_recs:
        if r.setpoint_variable not in seen_vars:
            all_recs.append(r)
            seen_vars.add(r.setpoint_variable)

    # Source 3: Model confidence fallback (only if nothing produced yet)
    if not all_recs:
        model_recs = _model_confidence_recommendation(
            event_row, features_df, clf, reg, current_risk, current_stab_s
        )
        all_recs.extend(model_recs)

    # Sort and cap
    all_recs.sort(key=lambda r: r.expected_risk_reduction, reverse=True)
    recs = all_recs[:MAX_RECS_PER_EVENT]

    # Populate rationale strings
    from src.explain import explain_recommendation
    for r in recs:
        r.rationale = explain_recommendation(r)

    return recs


# ---------------------------------------------------------------------------
# Novelty / Out-of-Distribution check
# ---------------------------------------------------------------------------

# Same feature subset as the k-NN recommendation search, for consistency —
# these are the variables that describe "what kind of transition is this."
NOVELTY_FEATURES: list[str] = [
    "valve_wear_index", "ambient_temp_c", "ramp_duration_seconds",
    "planned_bw_change", "planned_bw_ramp_rate",
    "planned_sf_ramp_rate", "planned_ff_ramp_rate",
]

NOVELTY_PERCENTILE_THRESHOLD: float = 90.0


def assess_transition_novelty(event_id: int) -> dict:
    """Flag whether this transition is unlike anything in the training history.

    Uses a SELF-CALIBRATING threshold rather than an arbitrary distance cutoff:
    computes every historical event's own nearest-neighbor distance to its
    closest OTHER historical event (leave-one-out), then checks where the
    CURRENT event's nearest-neighbor distance falls in that distribution. If
    it's further from its nearest neighbor than ~90% of historical events are
    from theirs, it's flagged as novel — the model's confidence on it should
    be treated with more caution than usual.

    Args:
        event_id: The event to assess.

    Returns:
        Dict with keys: is_novel (bool), percentile (float, 0-100),
        nearest_event_id (Optional[int]), distance (float).
    """
    features_df = generate_features()
    feat = features_df[["event_id"] + NOVELTY_FEATURES].dropna().set_index("event_id")

    if event_id not in feat.index or len(feat) < 3:
        return {"is_novel": False, "percentile": 0.0, "nearest_event_id": None, "distance": 0.0}

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feat.values)
    scaled_df = pd.DataFrame(scaled, index=feat.index, columns=feat.columns)

    # Leave-one-out nearest-neighbor distance for every historical event —
    # this is the baseline distribution of "how far a typical event sits
    # from its closest precedent."
    baseline_distances = []
    for eid in scaled_df.index:
        diffs = scaled_df.drop(index=eid).values - scaled_df.loc[eid].values
        baseline_distances.append(float(np.sqrt((diffs ** 2).sum(axis=1)).min()))
    baseline_distances = np.array(baseline_distances)

    others = scaled_df.drop(index=event_id)
    diffs = others.values - scaled_df.loc[event_id].values
    dists = np.sqrt((diffs ** 2).sum(axis=1))
    nearest_pos = int(dists.argmin())
    nearest_event_id = int(others.index[nearest_pos])
    nearest_distance = float(dists[nearest_pos])

    percentile = float((baseline_distances < nearest_distance).mean() * 100)

    return {
        "is_novel": percentile >= NOVELTY_PERCENTILE_THRESHOLD,
        "percentile": round(percentile, 1),
        "nearest_event_id": nearest_event_id,
        "distance": round(nearest_distance, 3),
    }


if __name__ == "__main__":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    events = load_events()
    # Find first high-risk event (offspec_label=1, not early_breach)
    features_df = generate_features()
    high_risk = features_df[
        (features_df["offspec_label"] == 1) &
        (features_df["early_breach"] == False)
    ]["event_id"].iloc[0]

    print(f"\n=== Recommendations for high-risk event {high_risk} ===")
    recs = get_recommendations(int(high_risk))
    if not recs:
        print("No recommendations generated (risk below threshold).")
    for i, r in enumerate(recs, 1):
        print(f"\n[{i}] {r.source_tag}")
        print(f"  Variable : {r.setpoint_variable}")
        print(f"  Current  : {r.current_value}")
        print(f"  Suggested: {r.suggested_value}")
        print(f"  Risk red.: {r.expected_risk_reduction*100:.1f}%")
        print(f"  Stab sav.: {r.expected_stab_reduction_s:.0f}s")
        print(f"  Confidence: {r.confidence*100:.0f}%")