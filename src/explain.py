"""
explain.py — SHAP-driven Explainability Module.

Converts model feature importances / SHAP values and recommendation evidence
into plain-language, deterministic rationale strings for both predictions
and setpoint recommendations — the single "Why?" logic reused across both.

CONSTITUTION RULES OBSERVED:
  Rule 13 — NO Large Language Models or generative text APIs inside the product.
            Uses deterministic SHAP-driven templates and python f-strings only.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import shap

if TYPE_CHECKING:
    from src.recommend import Recommendation

logger = logging.getLogger(__name__)

# User-friendly feature name translations for the "Why?" text
FEATURE_DISPLAY_NAMES = {
    "ambient_temp_c": "high ambient plant temperature",
    "valve_wear_index": "elevated filler valve wear/stiction",
    "ramp_duration_seconds": "short planned ramp duration",
    "planned_bw_ramp_rate": "aggressive planned Basis Weight ramp rate",
    "planned_sf_ramp_rate": "fast Stock Flow ramp rate",
    "planned_ff_ramp_rate": "fast Filler Flow ramp rate",
    "planned_sp_ramp_rate": "fast Steam Pressure ramp rate",
    "planned_bw_change": "large Basis Weight delta",
    "planned_sf_change": "large Stock Flow delta",
    "planned_ff_change": "large Filler Flow delta",
    "planned_sp_change": "large Steam Pressure delta",
    "bw_roc_300s": "rapid initial Basis Weight deviation",
    "bw_pct_dev_roc_300s": "steep initial deviation growth rate",
    "bw_dev_at_300s": "current Basis Weight offset from setpoint",
    "sf_dev_at_300s": "Stock Flow setpoint tracking offset",
    "ff_dev_at_300s": "Filler Flow setpoint tracking offset",
    "sp_dev_at_300s": "Steam Pressure tracking offset",
    "ms_dev_at_300s": "Machine Speed tracking offset",
    "basis_weight_mean_300s": "elevated early Basis Weight average",
    "basis_weight_std_300s": "early Basis Weight fluctuation",
    "moisture_mean_300s": "early Moisture average",
    "moisture_std_300s": "early Moisture fluctuation",
    "ash_mean_300s": "early Ash average",
    "ash_std_300s": "early Ash fluctuation",
    "caliper_mean_300s": "early Caliper average",
    "caliper_std_300s": "early Caliper fluctuation",
    "stock_flow_mean_300s": "high initial Stock Flow",
    "stock_flow_std_300s": "early Stock Flow fluctuation",
    "filler_flow_mean_300s": "high initial Filler Flow",
    "filler_flow_std_300s": "early Filler Flow fluctuation",
    "steam_pressure_mean_300s": "high Steam Pressure baseline",
    "steam_pressure_std_300s": "early Steam Pressure fluctuation",
    "machine_speed_mean_300s": "high Machine Speed",
    "machine_speed_std_300s": "early Machine Speed fluctuation",
}

# Fixed: the 10 one-hot grade columns (from_/to_ x 5 grades) had no display
# name at all — a SHAP explanation citing one would fall back to the raw
# column name (e.g. "from G3 HeavyBond"), readable but not polished.
# Generated rather than hardcoded so it stays correct if grades ever change.
_GRADE_NAMES = ["G1_LightBond", "G2_StandardBond", "G3_HeavyBond", "G4_Kraft", "G5_Newsprint"]
for _g in _GRADE_NAMES:
    FEATURE_DISPLAY_NAMES[f"from_{_g}"] = f"transitioning FROM {_g}"
    FEATURE_DISPLAY_NAMES[f"to_{_g}"] = f"transitioning TO {_g}"


def explain_prediction(
    feat_row: pd.DataFrame,
    clf,
    top_n: int = 3
) -> str:
    """
    Generate a plain-language rationale sentence explaining why an event is at risk.

    Uses SHAP TreeExplainer to find the top positive-impact features for this
    specific event.

    Args:
        feat_row: DataFrame containing 1 row of features (matching model inputs).
        clf: Trained GradientBoostingClassifier instance.
        top_n: Number of top driving features to cite in the text (default 3).

    Returns:
        A human-readable rationale sentence string.
    """
    drop_cols = ["event_id", "offspec_label", "stabilization_time_seconds", "early_breach"]
    X = feat_row[[c for c in feat_row.columns if c not in drop_cols]]

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X)

        # For binary classifier, shap_values might be 2D array [1, n_features] or list
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 2:
            sv = shap_values[0]
        else:
            sv = shap_values

        feature_names = X.columns.tolist()
        shap_series = pd.Series(sv, index=feature_names).sort_values(ascending=False)

        # Select top positive features (pushing risk UP)
        top_positive = shap_series[shap_series > 0].head(top_n)

        if top_positive.empty:
            # Fallback if no positive SHAP values (e.g. low risk)
            top_positive = shap_series.head(top_n)

        reasons = []
        for feat, val in top_positive.items():
            display_name = FEATURE_DISPLAY_NAMES.get(str(feat), str(feat).replace("_", " "))
            reasons.append(display_name)

        if len(reasons) == 1:
            reasons_text = reasons[0]
        elif len(reasons) == 2:
            reasons_text = f"{reasons[0]} and {reasons[1]}"
        else:
            reasons_text = f"{', '.join(reasons[:-1])}, and {reasons[-1]}"

        return f"Off-spec risk is primarily driven by {reasons_text}."

    except Exception as e:
        logger.warning("SHAP calculation failed, falling back to feature importances: %s", e)
        # Fallback to global feature importances
        imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False).head(top_n)
        reasons = [FEATURE_DISPLAY_NAMES.get(f, str(f)) for f in imp.index]
        return f"Off-spec risk is primarily driven by {', '.join(reasons)}."


def explain_recommendation(rec: Recommendation) -> str:
    """
    Generate a plain-language explanation string for why a recommendation is made.

    Args:
        rec: A Recommendation object from src.recommend.

    Returns:
        Human-readable explanation string.
    """
    var_name = rec.setpoint_variable.replace("_", " ")

    if "Recipe Rule" in rec.source_tag:
        return (
            f"The planned ramp rate ({rec.current_value} gsm/min) exceeds the recipe's recommended "
            f"safe limit ({rec.suggested_value} gsm/min). Adjusting to the recipe threshold keeps the "
            f"transition within safe control bounds."
        )

    elif "Historical Pattern" in rec.source_tag:
        match_info = f"Event #{rec.similar_event_id}" if rec.similar_event_id else "a similar past event"
        sim_pct = f"{rec.similar_event_similarity:.0f}%" if rec.similar_event_similarity else "high"
        return (
            f"Historical analysis identified {match_info} ({sim_pct} match under similar ambient/wear conditions) "
            f"which stayed on-spec by using a {var_name} of {rec.suggested_value}. "
            f"Adopting this setpoint is expected to lower off-spec risk by {rec.expected_risk_reduction*100:.0f}%."
        )

    else:  # Model Confidence fallback
        return (
            f"Model analysis indicates an elevated risk of deviation. Easing the {var_name} "
            f"from {rec.current_value} to {rec.suggested_value} provides process margin, reducing "
            f"predicted off-spec probability by {rec.expected_risk_reduction*100:.0f}%."
        )


if __name__ == "__main__":
    import joblib
    from src.data_loader import load_events
    from src.features import generate_features
    from src.recommend import get_recommendations

    clf = joblib.load("models/offspec_classifier.pkl")
    features_df = generate_features()

    # Find a high-risk event
    high_risk_event_id = features_df[
        (features_df["offspec_label"] == 1) & (features_df["early_breach"] == False)
    ]["event_id"].iloc[0]

    feat_row = features_df[features_df["event_id"] == high_risk_event_id]

    print(f"\n=== Explainability Test for Event #{high_risk_event_id} ===")
    pred_explanation = explain_prediction(feat_row, clf)
    print(f"Prediction Explanation:\n  {pred_explanation}\n")

    recs = get_recommendations(int(high_risk_event_id))
    for i, r in enumerate(recs, 1):
        r.rationale = explain_recommendation(r)
        print(f"Recommendation #{i} [{r.source_tag}]:")
        print(f"  Rationale: {r.rationale}\n")