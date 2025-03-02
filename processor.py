import os
import pandas as pd
import numpy as np
import streamlit as st
from paths import *


def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Fall"

def clean_data(df, log):
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    log.append("Missing data percentage per column:")
    missing_pct = df.isna().mean() * 100
    log.append(missing_pct.to_string())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in numeric_cols:
        if col != "temperature_2m" and df[col].isna().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            log.append(f"Imputed missing values in numeric column '{col}' with median: {median_val}")

    for col in categorical_cols:
        if df[col].isna().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            log.append(f"Imputed missing values in categorical column '{col}' with mode: {mode_val}")

    for col in df.columns:
        if col.lower() in ['period', 'date'] or ("date" in col.lower() or "time" in col.lower()):
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if df[col].isna().sum() > 0 and df[col].notna().sum() > 0:
                    mode_date = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_date)
                    log.append(f"Imputed missing values in date column '{col}' with mode: {mode_date}")
                df[col] = pd.to_datetime(df[col], errors='coerce')
                log.append(f"Converted column '{col}' to datetime.")

                prefix = "extracted_" + col
                if df[col].notna().sum() > 0:
                    df[prefix + "_hour"] = df[col].dt.hour
                    df[prefix + "_day"] = df[col].dt.day
                    df[prefix + "_month"] = df[col].dt.month
                    df[prefix + "_dayofweek"] = df[col].dt.dayofweek
                    for feat, rng in [("_hour", (0,23)), ("_day", (1,28)), ("_month", (1,12)), ("_dayofweek", (0,6))]:
                        feat_col = prefix + feat
                        if df[feat_col].isna().sum() > 0 and df[feat_col].notna().sum() > 0:
                            median_val = int(round(df[feat_col].median()))
                            df[feat_col] = df[feat_col].fillna(median_val)
                            log.append(f"Imputed missing values in {feat_col} with median: {median_val}")
                    df[prefix + "_is_weekend"] = df[prefix + "_dayofweek"].apply(lambda x: x >= 5)
                    df[prefix + "_season"] = df[prefix + "_month"].apply(get_season)
                else:
                    log.append(f"Column '{col}' has no valid dates; no temporal features extracted.")
            except Exception as e:
                log.append(f"Error processing date column '{col}': {e}")

    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - df.shape[0]
    if duplicates_removed > 0:
        log.append(f"Removed {duplicates_removed} duplicate rows.")

    constant_columns = [col for col in df.columns if df[col].nunique() <= 1 and col.lower() != "value"]
    if constant_columns:
        df.drop(columns=constant_columns, inplace=True)
        log.append(f"Dropped constant columns (excluding 'value'): {constant_columns}")

    outlier_summary = {}
    for col in numeric_cols:
        if col in df.columns and col != "temperature_2m":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            before_rows = df.shape[0]
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            after_rows = df.shape[0]
            removed = before_rows - after_rows
            if removed > 0:
                outlier_summary[col] = f"Removed {removed} outliers"
                log.append(f"Removed {removed} outliers from column '{col}'.")
            else:
                outlier_summary[col] = "No outliers"
                log.append(f"No outliers found in column '{col}'.")

    if "temperature_2m" in df.columns:
        if df["temperature_2m"].nunique() > 1:
            Q1 = df["temperature_2m"].quantile(0.25)
            Q3 = df["temperature_2m"].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df["temperature_anomaly"] = ~df["temperature_2m"].between(lower_bound, upper_bound)
            n_anomalies = df["temperature_anomaly"].sum()
            outlier_summary["temperature_2m"] = f"Marked {n_anomalies} anomalies"
            log.append(f"Marked {n_anomalies} anomalies in 'temperature_2m'.")
            df = df.sort_values(by="temperature_anomaly", ascending=True).reset_index(drop=True)
        else:
            outlier_summary["temperature_2m"] = "No outliers (constant)"
            log.append("Temperature column 'temperature_2m' is constant; no anomaly marking applied.")

    for col in numeric_cols:
        if col in df.columns:
            std_col = col + "_std"
            df[std_col] = (df[col] - df[col].mean()) / df[col].std()
            log.append(f"Created standardized feature '{std_col}'.")

    log.append("\n***** Outlier Summary *****")
    for col, summary in outlier_summary.items():
        log.append(f"{col}: {summary}")

    return df, outlier_summary
def cleaner():
    log = []

    input_file = merged_csv_output
    output_file = os.path.join(output_dir, "cleaned.csv")

    # Verify input file exists
    if not os.path.exists(input_file):
        st.write(f"Input file does not exist: {input_file}")
        return
    
    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the input CSV
    try:
        df = pd.read_csv(input_file)
    except Exception as e:
        st.write(f"Error reading CSV: {e}")
        return
    
    log.append(f"Original DataFrame shape: {df.shape}")
    
    # Clean the data
    cleaned_df, outlier_summary = clean_data(df, log)
    log.append(f"Cleaned DataFrame shape: {cleaned_df.shape}")
    
    # Save the cleaned DataFrame
    try:
        cleaned_df.to_csv(output_file, index=False)
        log.append(f"Cleaned CSV saved as {output_file}")
        st.write(f"Cleaned CSV saved as {output_file}")
    except Exception as e:
        log.append(f"Error saving cleaned CSV: {e}")
        st.write(f"Error saving cleaned CSV: {e}")
    
    # Generate summary statistics and print log
    try:
        summary_stats = cleaned_df.describe(include='all')
        st.write("Summary Statistics for Cleaned Data")
        st.write(summary_stats)
        st.write("\n***** Processing Log *****")
        st.write(pd.DataFrame(log))
    except Exception as e:
        st.write(f"Error generating summary statistics: {e}")
