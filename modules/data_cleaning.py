# modules/data_cleaning.py

import pandas as pd
import numpy as np
import os

def load_and_clean_data():
    """
    Loads raw dataset, cleans it by filling missing numeric values,
    and saves the cleaned version to a new CSV file.
    """
    # Define file paths
    FILE_PATH = r"D:\Telecom_customer_churn\dataset\telecom_churn_dataset.csv"
    OUTPUT_FILE_PATH = r"D:\Telecom_customer_churn\dataset\telecom_churn_dataset_cleaned.csv"

    # Load the raw dataset
    print(f"📂 Loading raw dataset from: {FILE_PATH}")
    df_raw = pd.read_csv(FILE_PATH)
    print("\n🔍 Missing Values Before Cleaning:\n", df_raw.isnull().sum())

    # Data Cleaning Process
    df_final = df_raw.copy()
    df_final.fillna(df_final.mean(numeric_only=True), inplace=True)

    # Verify cleaning
    print("\n✅ Missing Values After Cleaning:\n", df_final.isnull().sum())

    # Save the final cleaned data to a new CSV file
    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)
    df_final.to_csv(OUTPUT_FILE_PATH, index=False, float_format='%.2f')

    print(f"\n💾 Cleaned dataset saved successfully at: {OUTPUT_FILE_PATH}")
    print("\n🎯 Data Cleaning Completed Successfully!")

    return df_final
