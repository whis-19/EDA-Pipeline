import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import streamlit as st

def feature_engineering(data):
    # Feature Engineering
    data["hour"] = data["datetime"].dt.hour
    data["day"] = data["datetime"].dt.day
    data["month"] = data["datetime"].dt.month
    data['year'] = data["datetime"].dt.year
    data["day_of_week"] = data["datetime"].dt.dayofweek
    data["is_weekend"] = data["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    data["season"] = data["month"].apply(lambda x: 'Winter' if x in [12, 1, 2] else 'Spring' if x in [3, 4, 5] else 'Summer' if x in [6, 7, 8] else 'Fall')
    data = pd.get_dummies(data, columns=['season'], drop_first=True)

    return data

def data_type_conversions(data):
    # Ensure demand_mwh is numeric
    data["demand_mwh"] = pd.to_numeric(data["demand_mwh"], errors='coerce')

    # Sort data by datetime
    data.sort_values(by="datetime", inplace=True)

    return data

def test_mcar(df):
    df_miss = df.isnull().astype(int)
    chi2, p, _, _ = chi2_contingency(df_miss.corr())
    return p  # If p > 0.05, data is MCAR

def data_cleaning_and_consistency(df):
    # 1. Identify missing values per column
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    # 2. Display missing values summary
    missing_summary = pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage})
    st.write("Missing Values Summary:")
    st.write(missing_summary[missing_summary['Missing Values'] > 0])
    
    p_value = test_mcar(df)
    if p_value > 0.05:
        st.write("Missing data is likely MCAR (Missing Completely at Random)")
    else:
        st.write("Missing data is likely MAR (Missing at Random) or MNAR (Not Missing at Random)")

    # 5. Handling missing data
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == 'object':  # Categorical columns
                df[col].fillna(df[col].mode()[0], inplace=True)  # Impute with mode
            else:  # Numerical columns
                df[col].fillna(df[col].median(), inplace=True)  # Impute with median

    st.write("Missing values handled using mode (categorical) or median (numerical)")

    return df

def handle_duplicates_and_anomalies(df):
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Identify numerical columns
    num_cols = df.select_dtypes(include=['number']).columns

    # Detect outliers using IQR method
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Mark outliers
        outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        st.write(f"Outliers detected in {col}: {outliers.sum()} rows")

    return df

# def normalize_data(data):
#     # Identify numerical columns for normalization/standardization
#     num_cols = data.select_dtypes(include=['number']).columns

#     # Choose either StandardScaler (Z-score normalization) or MinMaxScaler (scales to [0,1])
#     scaler = StandardScaler()
#     data[num_cols] = scaler.fit_transform(data[num_cols])
#     return data

def normalize_data(data):
    # Normalize only the 'value' column
    if 'value' in data.columns:
        scaler = StandardScaler()
        data['value'] = scaler.fit_transform(data[['value']])
    return data


def process_data(df):
    df = data_type_conversions(df)
    df = data_cleaning_and_consistency(df)
    df = handle_duplicates_and_anomalies(df)
    df = feature_engineering(df)

    return df