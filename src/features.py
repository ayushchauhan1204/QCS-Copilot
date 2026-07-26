import pandas as pd
import numpy as np
from src.data_loader import load_events, load_timeseries, load_recipes

FEATURE_WINDOW_S = 300

def generate_features() -> pd.DataFrame:
    """
    Generate the feature matrix for all events according to DATASET_SPECIFICATION.md.
    
    Includes:
    - `early_breach` flag to filter training data later.
    - One-hot encoded `from_grade` and `to_grade`.
    - Planned setpoint features (known at t=0).
    - Windowed actual/measured features (only t <= 300s).
    
    Returns:
        pd.DataFrame: Feature matrix keyed by event_id, including labels.
    """
    events = load_events()
    ts = load_timeseries()
    recipes = load_recipes()
    recipes.set_index("grade_name", inplace=True)
    
    # 1. Start with the event base table
    df = events[['event_id', 'from_grade', 'to_grade', 'valve_wear_index', 'ambient_temp_c', 
                 'ramp_duration_seconds', 'offspec_label', 'stabilization_time_seconds', 
                 'time_to_offspec_s']].copy()
                 
    # 2. Add early_breach flag (events where threshold was breached before the window closed)
    df['early_breach'] = df['time_to_offspec_s'] < FEATURE_WINDOW_S
    
    # 3. One-hot encode grades
    # Using specific prefix to keep it clean
    grades = ['G1_LightBond', 'G2_StandardBond', 'G3_HeavyBond', 'G4_Kraft', 'G5_Newsprint']
    for g in grades:
        df[f'from_{g}'] = (df['from_grade'] == g).astype(int)
        df[f'to_{g}'] = (df['to_grade'] == g).astype(int)
        
    # 4. Planned / Setpoint features (known at t=0)
    # We can get these directly by looking up the recipes
    df['planned_bw_change'] = df.apply(lambda row: recipes.loc[row['to_grade'], 'basis_weight_setpoint'] - recipes.loc[row['from_grade'], 'basis_weight_setpoint'], axis=1)
    df['planned_bw_ramp_rate'] = df['planned_bw_change'] / df['ramp_duration_seconds']
    
    df['planned_sf_change'] = df.apply(lambda row: recipes.loc[row['to_grade'], 'stock_flow_setpoint'] - recipes.loc[row['from_grade'], 'stock_flow_setpoint'], axis=1)
    df['planned_sf_ramp_rate'] = df['planned_sf_change'] / df['ramp_duration_seconds']
    
    df['planned_ff_change'] = df.apply(lambda row: recipes.loc[row['to_grade'], 'filler_flow_setpoint'] - recipes.loc[row['from_grade'], 'filler_flow_setpoint'], axis=1)
    df['planned_ff_ramp_rate'] = df['planned_ff_change'] / df['ramp_duration_seconds']
    
    df['planned_sp_change'] = df.apply(lambda row: recipes.loc[row['to_grade'], 'steam_pressure_setpoint'] - recipes.loc[row['from_grade'], 'steam_pressure_setpoint'], axis=1)
    df['planned_sp_ramp_rate'] = df['planned_sp_change'] / df['ramp_duration_seconds']
    
    # 5. Windowed actual/measured features (t <= 300)
    ts_window = ts[ts['t_seconds'] <= FEATURE_WINDOW_S].copy()
    
    # Aggregate stats
    agg_funcs = {
        'basis_weight': ['mean', 'std'],
        'moisture': ['mean', 'std'],
        'ash': ['mean', 'std'],
        'caliper': ['mean', 'std'],
        'stock_flow': ['mean', 'std'],
        'filler_flow': ['mean', 'std'],
        'steam_pressure': ['mean', 'std'],
        'machine_speed': ['mean', 'std']
    }
    
    # Compute aggregations
    ts_agg = ts_window.groupby('event_id').agg(agg_funcs)
    # Flatten multi-index columns
    ts_agg.columns = [f"{col}_{stat}_300s" for col, stat in ts_agg.columns]
    
    # Compute rate-of-change and current deviation (value at t=300)
    # Sort just to be safe
    ts_window = ts_window.sort_values(['event_id', 't_seconds'])
    
    # Get first (t=0) and last (t=300) rows per event
    ts_first = ts_window.groupby('event_id').first()
    ts_last = ts_window.groupby('event_id').last()
    
    # Rate of change over the window
    roc_bw = ts_last['basis_weight'] - ts_first['basis_weight']
    roc_bw_pct = ts_last['basis_weight_pct_dev'] - ts_first['basis_weight_pct_dev']
    
    # Current deviation at t=300 (measured vs setpoint)
    dev_bw = ts_last['basis_weight'] - ts_last['basis_weight_setpoint']
    dev_sf = ts_last['stock_flow'] - ts_last['stock_flow_setpoint']
    dev_ff = ts_last['filler_flow'] - ts_last['filler_flow_setpoint']
    dev_sp = ts_last['steam_pressure'] - ts_last['steam_pressure_setpoint']
    dev_ms = ts_last['machine_speed'] - ts_last['machine_speed_setpoint']
    
    # Combine window features
    window_features = pd.DataFrame({
        'bw_roc_300s': roc_bw,
        'bw_pct_dev_roc_300s': roc_bw_pct,
        'bw_dev_at_300s': dev_bw,
        'sf_dev_at_300s': dev_sf,
        'ff_dev_at_300s': dev_ff,
        'sp_dev_at_300s': dev_sp,
        'ms_dev_at_300s': dev_ms
    })
    
    # Join everything together
    df = df.join(ts_agg, on='event_id')
    df = df.join(window_features, on='event_id')
    
    # Drop categorical columns we one-hot encoded and leakage columns
    # We keep labels in the returned dataframe, but downstream train_model will separate them
    df.drop(columns=['from_grade', 'to_grade', 'time_to_offspec_s'], inplace=True)
    
    # Fill NA for std where only one sample might exist (though 60 samples exist, so it shouldn't)
    df.fillna(0, inplace=True)
    
    return df

if __name__ == "__main__":
    df = generate_features()
    print(f"Generated feature matrix with shape {df.shape}")
    print(f"Number of events with early_breach == True: {df['early_breach'].sum()}")
    print("Columns:", df.columns.tolist()[:15], "...")
