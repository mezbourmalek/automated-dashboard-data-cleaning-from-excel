import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import os
import time

# --- CONFIGURATION ---
# INPUT_FILE = 'data.xlsx'
# OUTPUT_FILE = 'laptops_cleaned_automated.xlsx'
# CHECK_INTERVAL = 5  # Check for updates every 5 seconds

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(BASE_DIR, 'data.xlsx')
OUTPUT_FILE = os.path.join(BASE_DIR, 'laptops_cleaned_automated.xlsx')
CHECK_INTERVAL = 5

def smart_cleaning_pipeline(df_input):
    """Full cleaning logic from the notebook[cite: 518, 576]."""
    df_temp = df_input.copy()
    
    # 1. Regex Cleaning for Numeric Columns [cite: 521, 527]
    cols_to_fix = ['Price', 'Total Sales', 'cpu_speed', 'screen_size', 
                   'harddisk', 'ram', 'Available Stock', 'Sale Product Count']
    
    for col in cols_to_fix:
        if col in df_temp.columns:
            df_temp[col] = df_temp[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce')
    
    # 2. Categorical Imputation [cite: 528, 529]
    cat_cols = df_temp.select_dtypes(include=['object']).columns
    df_temp[cat_cols] = df_temp[cat_cols].fillna("Unknown")
    
    # 3. Numeric Imputation Logic [cite: 530, 564]
    for col in cols_to_fix:
        if col not in df_temp.columns: 
            continue
            
        missing_pct = df_temp[col].isnull().mean()
        
        if missing_pct > 0.60: # Drop if > 60% missing 
            df_temp = df_temp.drop(columns=[col])
            continue
            
        num_only = df_temp.select_dtypes(include=[np.number])
        correlations = num_only.corr().abs()[col].drop(col, errors='ignore')
        max_corr = correlations.max() if not correlations.empty else 0
        
        # Apply Median or KNN based on thresholds [cite: 544, 555, 573]
        if missing_pct < 0.05:
            df_temp[col] = df_temp[col].fillna(df_temp[col].median())
        elif 0.05 <= missing_pct <= 0.30:
            if max_corr > 0.3:
                imputer = KNNImputer(n_neighbors=5)
                df_temp[[col]] = imputer.fit_transform(df_temp[[col]])
            else:
                df_temp[col] = df_temp[col].fillna(df_temp[col].median())
        elif 0.30 < missing_pct <= 0.60:
            df_temp[col] = df_temp[col].fillna(df_temp[col].median())
            
    # 4. Outlier Handling (Total Sales) 
    if 'Total Sales' in df_temp.columns:
        Q1 = df_temp['Total Sales'].quantile(0.25)
        Q3 = df_temp['Total Sales'].quantile(0.75)
        IQR = Q3 - Q1
        upper_limit = Q3 + 5 * IQR
        df_temp = df_temp[df_temp['Total Sales'] <= upper_limit]
        
    # 5. String Normalization [cite: 626]
    for text_col in ['brand', 'model']:
        if text_col in df_temp.columns:
            df_temp[text_col] = df_temp[text_col].astype(str).str.lower().str.strip()
            
    return df_temp

def monitor_and_clean():
    last_modified = 0
    print(f"--- AUTOMATION ACTIVE ---")
    print(f"Script folder: {BASE_DIR}")
    print(f"Watching for: {INPUT_FILE}")
    
    while True:
        if os.path.exists(INPUT_FILE):
            current_modified = os.path.getmtime(INPUT_FILE)
            if current_modified > last_modified:
                try:
                    print(f"\nChange detected! Cleaning at {time.ctime()}...")
                    
                    # Load and process
                    df_raw = pd.read_excel(INPUT_FILE)
                    df_cleaned = smart_cleaning_pipeline(df_raw)
                    
                    # Save result
                    df_cleaned.to_excel(OUTPUT_FILE, index=False)
                    
                    print(f"SUCCESS: Created '{OUTPUT_FILE}'")
                    last_modified = current_modified
                except Exception as e:
                    print(f"Error: {e}")
        else:
            print(f"Waiting... Please put 'data.xlsx' inside: {BASE_DIR}")
            
        time.sleep(CHECK_INTERVAL)
if __name__ == "__main__":
    monitor_and_clean()