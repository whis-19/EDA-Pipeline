import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from paths import *

def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (data[column] < lower_bound) | (data[column] > upper_bound), lower_bound, upper_bound

def detect_outliers_zscore(data, column, threshold=3):
    return np.abs(zscore(data[column])) > threshold

def plot_before_after(df_original, df_capped, column):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Before
    axes[0].boxplot(df_original[column].dropna(), vert=True)
    axes[0].set_title(f"Before Capping - {column}")
    
    # After
    axes[1].boxplot(df_capped[column].dropna(), vert=True)
    axes[1].set_title(f"After Capping - {column}")
    
    plt.tight_layout()
    st.pyplot(fig)
    

def handle_outliers(df, num_cols):
    outlier_summary = []
    total_outliers = 0

    for col in num_cols:
        if col in df.columns:
            # IQR-based detection
            iqr_mask, lower_bound, upper_bound = detect_outliers_iqr(df, col)
            
            # Z-score detection
            z_mask = detect_outliers_zscore(df, col)
            
            # Combine both outlier masks (logical OR)
            combined_mask = iqr_mask | z_mask
            
            # Count outliers before capping
            outlier_count = combined_mask.sum()
            total_outliers += outlier_count
            
            # Record summary
            outlier_summary.append({
                "column": col,
                "iqr_outliers": iqr_mask.sum(),
                "zscore_outliers": z_mask.sum(),
                "combined_outliers": outlier_count,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            })
            
            # cap outliers
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound

    return df, outlier_summary, total_outliers

def generate_report(outlier_summary, total_outliers):
    st.write("Outlier Handling Report")

    st.write("Column-by-Column Summary:")
    for summary in outlier_summary:
        st.write(
            f"- {summary['column']}:\n"
            f"   IQR outliers: {summary['iqr_outliers']}\n"
            f"   Z-score outliers: {summary['zscore_outliers']}\n"
            f"   Combined outliers: {summary['combined_outliers']}\n"
            f"   Capping range: [{summary['lower_bound']:.2f}, {summary['upper_bound']:.2f}]\n"
        )

    st.write(f"Total Outliers (across all columns): {total_outliers}\n")

    st.write("Technical Rationale:")
    st.write("- Outliers can distort mean/variance, affect model performance, and skew visualizations.")
    st.write("- By capping rather than removing, we retain data size while mitigating extreme skew but this cause data to be biased.\n")


    st.write("Decision:")
    st.write("  We applied capping based on IQR boundaries.\n")
def outlier():
    num_cols = ["value", "temperature_2m", "value_std", "temperature_2m_std"]
    
    df = pd.read_csv(input_file)
    
    df_original = df.copy()
    
    df, outlier_summary, total_outliers = handle_outliers(df, num_cols)
    
    for col in num_cols:
        if col in df.columns:
            plot_before_after(df_original, df, col)
    
    generate_report(outlier_summary, total_outliers)
    st.write("Outlier detection, capping, and reporting complete.")


