import os
import pandas as pd

# Canonical output directory — all generated CSVs live here.
# Run `python -m data.generate_data --output outputs/` to populate.
OUTPUT_DIR = "outputs" if os.path.exists(os.path.join("outputs", "grade_change_events.csv")) else os.path.join("data", "output")
EVENTS_PATH = os.path.join(OUTPUT_DIR, "grade_change_events.csv")
TIMESERIES_PATH = os.path.join(OUTPUT_DIR, "historian_timeseries.csv")
RECIPES_PATH = os.path.join(OUTPUT_DIR, "recipe_reference.csv")


def load_events() -> pd.DataFrame:
    """Load the grade_change_events.csv file.
    
    Returns:
        pd.DataFrame containing event-level metadata and labels.
    """
    if not os.path.exists(EVENTS_PATH):
        raise FileNotFoundError(f"Could not find {EVENTS_PATH}. Run data/generate_data.py first.")
    
    # Parse timestamps
    df = pd.read_csv(EVENTS_PATH)
    df["event_start_timestamp"] = pd.to_datetime(df["event_start_timestamp"])
    return df


def load_timeseries() -> pd.DataFrame:
    """Load the historian_timeseries.csv file.
    
    Returns:
        pd.DataFrame containing the high-resolution event timeseries.
    """
    if not os.path.exists(TIMESERIES_PATH):
        raise FileNotFoundError(f"Could not find {TIMESERIES_PATH}. Run data/generate_data.py first.")
    
    # Parse timestamps
    df = pd.read_csv(TIMESERIES_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_recipes() -> pd.DataFrame:
    """Load the recipe_reference.csv file.
    
    Returns:
        pd.DataFrame containing the recipe definitions, keyed by grade_name.
    """
    if not os.path.exists(RECIPES_PATH):
        raise FileNotFoundError(f"Could not find {RECIPES_PATH}. Run data/generate_data.py first.")
    
    return pd.read_csv(RECIPES_PATH)
