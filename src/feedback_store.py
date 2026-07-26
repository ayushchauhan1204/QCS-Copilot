"""
feedback_store.py - SQLite interface for logging operator actions.

Responsible for persisting operator decisions to accept/reject setpoint recommendations,
and computing calibration accuracy statistics for the Trust Ledger dashboard panel.

CONSTITUTION RULES OBSERVED:
  Rule 11 - Never skip accept/reject feedback logging. It is a core deliverable.
  Rule 4  - No UI or prediction logic here.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Anchored to this file's location (src/feedback_store.py -> project root),
# NOT to the current working directory. Fixes: the db silently ending up in
# the wrong place (or an empty file with no table ever getting created)
# depending on how/where `streamlit run` was launched from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = str(_PROJECT_ROOT / "db")
DB_PATH = os.path.join(DB_DIR, "feedback.db")


def init_db() -> None:
    """Initialize the SQLite database and create the feedback table if not exists.

    Safe to call multiple times — uses CREATE TABLE IF NOT EXISTS. Called
    once at module import AND defensively at the top of every public
    function below, so a missing/corrupted table (e.g. from a sync tool
    interfering with the .db file, or the file being deleted mid-session)
    self-heals on the next call instead of raising DatabaseError.
    """
    try:
        os.makedirs(DB_DIR, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS recommendations_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id INTEGER NOT NULL,
                setpoint_variable TEXT NOT NULL,
                suggested_value REAL NOT NULL,
                current_value REAL NOT NULL,
                confidence REAL NOT NULL,
                source_tag TEXT NOT NULL,
                operator_action TEXT NOT NULL CHECK(operator_action IN ('accepted', 'rejected')),
                realized_outcome TEXT NOT NULL CHECK(realized_outcome IN ('clean', 'offspec')),
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()
        logger.info("Feedback database ready at %s", DB_PATH)
    except sqlite3.Error:
        logger.exception("Failed to initialize feedback database at %s", DB_PATH)
        raise


# Initialize on module import — self-healing calls below make this a
# best-effort head start, not the only line of defense.
init_db()


def log_feedback(
    event_id: int,
    setpoint_variable: str,
    suggested_value: float,
    current_value: float,
    confidence: float,
    source_tag: str,
    operator_action: str,
    realized_outcome: str
) -> None:
    """
    Log a single recommendation action and realized outcome to SQLite.

    Args:
        event_id: ID of the grade-change event.
        setpoint_variable: Name of the setpoint variable adjusted.
        suggested_value: Recommended setpoint target.
        current_value: Original planned setpoint.
        confidence: Stated confidence of recommendation.
        source_tag: Tag indicating source (Recipe, k-NN, etc.).
        operator_action: 'accepted' or 'rejected'.
        realized_outcome: 'clean' or 'offspec'.
    """
    init_db()  # self-healing: recreate the table if it's ever missing
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO recommendations_feedback (
            event_id, setpoint_variable, suggested_value, current_value, 
            confidence, source_tag, operator_action, realized_outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event_id, setpoint_variable, suggested_value, current_value,
        confidence, source_tag, operator_action, realized_outcome
    ))
    
    conn.commit()
    conn.close()
    logger.info("Logged operator %s for event %d", operator_action, event_id)


def get_calibration_stats() -> dict:
    """
    Compute calibration stats comparing stated confidence to actual outcomes.

    Returns:
        Dict containing total counts, accept rates, and detailed stats DataFrame.
    """
    init_db()  # self-healing: recreate the table if it's ever missing
    conn = sqlite3.connect(DB_PATH)
    
    query = "SELECT * FROM recommendations_feedback"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    if df.empty:
        return {
            "total_count": 0,
            "accept_count": 0,
            "reject_count": 0,
            "accept_rate": 0.0,
            "history_df": df
        }
        
    total = len(df)
    accepts = len(df[df["operator_action"] == "accepted"])
    rejects = len(df[df["operator_action"] == "rejected"])
    
    return {
        "total_count": total,
        "accept_count": accepts,
        "reject_count": rejects,
        "accept_rate": round(accepts / total, 3) if total > 0 else 0.0,
        "history_df": df
    }


def get_calibration_curve_data() -> pd.DataFrame:
    """
    Retrieve bin-level calibration confidence vs realization rate.

    Returns:
        DataFrame with confidence bins, expected accuracy, and actual accuracy.
    """
    stats_data = get_calibration_stats()
    df = stats_data["history_df"]
    
    if df.empty:
        return pd.DataFrame(columns=["confidence_bin", "avg_confidence", "actual_clean_rate", "count"])
        
    # Group confidence into bins (e.g. 40-60%, 60-80%, 80-100%)
    bins = [0.0, 0.60, 0.80, 1.0]
    labels = ["Low (<60%)", "Medium (60-80%)", "High (80-100%)"]
    df["confidence_bin"] = pd.cut(df["confidence"], bins=bins, labels=labels, include_lowest=True)
    
    # Calculate % clean outcomes per bin
    # If accepted -> target is clean. If rejected -> it depends on if the process actually went off-spec
    df["is_clean_outcome"] = (df["realized_outcome"] == "clean").astype(int)
    
    grouped = df.groupby("confidence_bin").agg(
        avg_confidence=("confidence", "mean"),
        actual_clean_rate=("is_clean_outcome", "mean"),
        count=("id", "count")
    ).reset_index()
    
    # Fill NAs
    grouped["avg_confidence"] = grouped["avg_confidence"].fillna(0.0)
    grouped["actual_clean_rate"] = grouped["actual_clean_rate"].fillna(0.0)
    
    return grouped


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_db()
    
    # Run test inserts
    print("\n--- Running Test Inserts ---")
    log_feedback(5, "ramp_duration_seconds", 1250.0, 1155.0, 0.66, "Historical Pattern", "accepted", "clean")
    log_feedback(10, "ramp_rate_gsm_min", 4.5, 6.2, 0.85, "Recipe Rule", "rejected", "offspec")
    
    stats_data = get_calibration_stats()
    print(f"Total entries: {stats_data['total_count']}")
    print(f"Accept rate  : {stats_data['accept_rate']*100:.1f}%")
    
    curve = get_calibration_curve_data()
    print("\nCalibration Curve Data:")
    print(curve.to_string(index=False))