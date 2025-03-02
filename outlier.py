import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import streamlit as st

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
    return outliers

def detect_outliers_zscore(data, column, threshold=3):
    z_scores = np.abs(zscore(data[column]))
    return z_scores > threshold

def detect_and_handle_outliers(df, column, method="remove"):
    """
    Detects and handles outliers in the specified column of the dataset.

    Parameters:
        df (pd.DataFrame): The dataset containing the column.
        column (str): The column name to analyze.
        method (str): Strategy to handle outliers - "remove", "cap", or "transform".

    Returns:
        pd.DataFrame: The dataset after applying the outlier handling strategy.
    """

    # Handle Outliers Based on the Selected Method
    df_cleaned = df.copy()
    
    if method == "remove":
        iqr_outliers = detect_outliers_iqr(df, column)
        df_cleaned = df_cleaned[~iqr_outliers]  # Remove outliers
    elif method == "cap":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_cleaned[column] = np.clip(df_cleaned[column], lower_bound, upper_bound)  # Cap values
    elif method == "transform":
        df_cleaned[column] = np.log1p(df_cleaned[column])  # Log transformation

    # Plot Before and After
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    axs[0].hist(df[column], bins=50, color="blue", alpha=0.7)
    axs[0].set_title(f"Original {column} Distribution")

    axs[1].hist(df_cleaned[column], bins=50, color="red", alpha=0.7)
    axs[1].set_title(f"After Applying {method.capitalize()} Method")

    st.pyplot(fig)
    
    return df_cleaned

def outliers(df):

    # Apply to demand_mwh column
    iqr_outliers = detect_outliers_iqr(df, "demand_mwh")
    st.write("IQR Outliers (demand_mwh):", iqr_outliers.sum())
    zscore_outliers = detect_outliers_zscore(df, "demand_mwh")
    st.write("Z-score Outliers (demand_mwh):", zscore_outliers.sum())
    
    # Apply to temperature column
    iqr_outliers = detect_outliers_iqr(df, "temperature")
    st.write("IQR Outliers (temperature):", iqr_outliers.sum())
    zscore_outliers = detect_outliers_zscore(df, "temperature")
    st.write("Z-score Outliers (temperature):", zscore_outliers.sum())

    # Run Outlier Detection & Handling
    cleaned_data = detect_and_handle_outliers(df, "demand_mwh", method="transform")
    cleaned_data = detect_and_handle_outliers(cleaned_data, "temperature", method="remove")

    return cleaned_data