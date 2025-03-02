import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import streamlit as st
from paths import *

def statistical_summary(df):
    """
    Compute key statistical metrics for selected numerical features and save to "Statistical_Summary.txt".
    """
    key_features = ['value', 'temperature_2m', 'extracted_period_hour',
                    'extracted_period_day', 'extracted_period_month', 'extracted_period_dayofweek']
    existing_features = [col for col in key_features if col in df.columns]
    
    summary = df[existing_features].describe()
    extra_stats = pd.DataFrame({
        'skewness': df[existing_features].skew(),
        'kurtosis': df[existing_features].kurtosis()
    })
    summary = pd.concat([summary, extra_stats])
    
    st.write(summary)

def time_series_analysis(df):
    """
    Plot electricity demand over time using a plasma colormap.
    """
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df_sorted = df.sort_values(by='date').dropna(subset=['date', 'value'])

    plt.figure(figsize=(12,6))

    # Use scatter plot with plasma colormap for gradient effect
    colors = plt.cm.plasma(np.linspace(0, 1, len(df_sorted)))
    plt.scatter(df_sorted['date'], df_sorted['value'], c=np.arange(len(df_sorted)), cmap='plasma', edgecolors='none')

    plt.xlabel("Time")
    plt.ylabel("Electricity Demand")
    plt.title("Electricity Demand Over Time")
    plt.colorbar(label="Time Progression")
    plt.grid(True)

    plt.tight_layout()
    st.pyplot(plt)

def univariate_analysis(df):
    """
    Generate histograms, boxplots, and density plots for key numerical features using plasma colormap.
    """
    key_features = [
        "extracted_period_hour", "extracted_period_day", "extracted_period_month", 
        "extracted_period_dayofweek", "temperature_2m"
    ]
    df = df[key_features].dropna()
    
    fig, axes = plt.subplots(len(key_features), 3, figsize=(15, 5*len(key_features)))
    
    for i, col in enumerate(key_features):
        col_data = df[col]
        
        # Histogram
        sns.histplot(col_data, bins=30, kde=True, ax=axes[i, 0], color=plt.cm.plasma(0.2))
        axes[i, 0].set_title(f"Histogram of {col}")
        
        # Boxplot
        sns.boxplot(x=col_data, ax=axes[i, 1], color=plt.cm.plasma(0.5))
        axes[i, 1].set_title(f"Boxplot of {col}")
        
        # KDE (Density Plot)
        sns.kdeplot(col_data, ax=axes[i, 2], color=plt.cm.plasma(0.8))
        axes[i, 2].set_title(f"Density Plot of {col}")
    
    plt.tight_layout()
    st.pyplot(fig)

def correlation_analysis(df):
    """
    Compute correlation matrix and visualize using a heatmap with plasma colormap.
    """
    key_features = ['value', 'temperature_2m', 'extracted_period_hour',
                    'extracted_period_day', 'extracted_period_month', 'extracted_period_dayofweek']
    df = df[key_features].dropna()
    
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="plasma", square=True)
    plt.title("Correlation Matrix")

    st.pyplot(plt)
    
def advanced_time_series_techniques(df):
    """
    Perform time series decomposition and ADF test.
    """
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df_sorted = df.sort_values(by='date').dropna(subset=['date', 'value'])
    ts = df_sorted.set_index('date')['value']
    
    decomposition = seasonal_decompose(ts, model='additive', period=24)
    fig = decomposition.plot()
    fig.set_size_inches(12, 8)
    plt.tight_layout()

    st.pyplot(fig)

    adf_result = adfuller(ts.dropna())
    adf_output = (f"ADF Statistic: {adf_result[0]:.4f}\n"
                    f"p-value: {adf_result[1]:.4f}\n"
                    f"Critical Values: {adf_result[4]}")

    st.write(adf_output)

def run_eda(input_csv):
    """
    Load dataset, run all EDA functions, and save outputs.
    """
    df = pd.read_csv(input_csv)
    st.write(f"Loaded dataset with shape: {df.shape}")
    
    statistical_summary(df)
    time_series_analysis(df)
    univariate_analysis(df)
    correlation_analysis(df)
    advanced_time_series_techniques(df)
