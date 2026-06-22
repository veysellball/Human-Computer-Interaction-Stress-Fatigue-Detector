import os
import pandas as pd
from config import CSV_DATA_PATH
from ml_pipeline import run_prediction, train_local_model, ALL_FEATURES

def run_test():
    print("--- Test 1: Prediction before Local Model ---")
    mock_item = {feat: 1.0 for feat in ALL_FEATURES}
    res1 = run_prediction(mock_item)
    print(f"Prediction result: {res1}")
    
    print("\n--- Test 2: Simulating Local Data Accumulation ---")
    # Take 60 rows from features_v5.csv to ensure we pass the 50 threshold
    df = pd.read_csv("features_v5.csv").head(60)
    
    # Needs a mix of Stressed and Not_Stressed. Let's explicitly manipulate some labels to ensure variety.
    mapped_col = 'stress_val'
    if 'stress_val_mapped' in df.columns:
        mapped_col = 'stress_val_mapped'
    
    # Even if they don't have mapped col, we just inject it so mapping evaluates to different classes
    df.loc[0:20, 'stress_val'] = 'F_Great'    # Maps to Not_Stressed
    df.loc[21:, 'stress_val'] = 'V_Stressed'  # Maps to Stressed
    
    # Do the same for fatigue_val to ensure local training doesn't skip due to single-class
    df.loc[0:20, 'fatigue_val'] = 'No'         # Maps to Not_Fatigued
    df.loc[21:, 'fatigue_val'] = 'V_High'      # Maps to Fatigued
    
    # It hasn't been mapped yet in the local data probably, wait: local data logger should write the target column as it sees fit.
    # We will write the unmapped data, train_local_model assumes 'stress_val_mapped' OR 'stress_val' is already in classes...? 
    # Wait, train_local_model expects target classes (e.g., Stressed, Not_Stressed). The backend should prepare the mapping. 
    # Oh! My `train_local_model` looks for `stress_val_mapped`, and if it's missing, it checks `stress_val`, without mapping it!
    # Let me fix ml_pipeline.py: `train_local_model` should call `apply_label_mapping` first.
    
    df.to_csv(CSV_DATA_PATH, index=False)
    print(f"Saved dummy data to {CSV_DATA_PATH}")
    
    print("\n--- Test 3: Training Local Model ---")
    # I should edit ml_pipeline.py before I run this script to ensure local training maps labels.
    success = train_local_model(CSV_DATA_PATH)
    print(f"Local training success: {success}")
    
    print("\n--- Test 4: Prediction after Local Model ---")
    res2 = run_prediction(mock_item)
    print(f"Prediction result: {res2}")

if __name__ == "__main__":
    run_test()
