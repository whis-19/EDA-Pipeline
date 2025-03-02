import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import streamlit as st

def statistical_summary(df):
    """
    Computes statistical metrics for each numerical variable in the dataframe and returns
    a prettified DataFrame.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with numerical variables
    
    Returns:
    pd.DataFrame: A DataFrame containing the statistical summaries
    """
    # Select numerical columns
    num_cols = df.select_dtypes(include=['number']).columns

    if len(num_cols) == 0:
        st.write("No numerical columns found in the DataFrame.")
        return

    # Dictionary to store summaries for each variable
    summary_dict = {}
    for col in num_cols:
        summary_dict[col] = {
            "Mean": df[col].mean(),
            "Median": df[col].median(),
            "Standard Deviation": df[col].std(),
            "Variance": df[col].var(),
            "Skewness": df[col].skew(),
            "Kurtosis": stats.kurtosis(df[col], fisher=True),
            "Min": df[col].min(),
            "Max": df[col].max(),
            "25th Percentile": df[col].quantile(0.25),
            "50th Percentile (Median)": df[col].quantile(0.50),
            "75th Percentile": df[col].quantile(0.75)
        }

    # Convert dictionary to DataFrame for pretty display
    summary_df = pd.DataFrame(summary_dict).T
    st.write(summary_df)
    
    return summary_df

def plot_time_series(df, datetime_col="datetime", demand_col="demand_mwh", sample_rate=1000):
    """
    Plots a clean time series line chart without unwanted fill effects.

    :param df: Pandas DataFrame containing time series data
    :param datetime_col: Column name representing timestamps
    :param demand_col: Column name representing electricity demand
    :param sample_rate: Interval to downsample data for better visualization
    """
    # Convert datetime column to proper format
    df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Sort data by time
    df = df.sort_values(by=datetime_col)

    # Downsample data (optional but improves clarity)
    df_sampled = df

    # Set figure size
    plt.figure(figsize=(14, 6))

    # Plot as a thin line (no fill)
    plt.plot(df_sampled[datetime_col], df_sampled[demand_col], color='blue', linewidth=1, linestyle='-')

    # Titles and labels
    plt.title("Electricity Demand Over Time", fontsize=14)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Electricity Demand", fontsize=12)

    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45)

    # Grid for better visualization
    plt.grid(True, linestyle="--", alpha=0.6)

    # Show plot
    st.pyplot(plt)

def univariate_analysis(df):
    """
    Perform univariate analysis: Histogram, Boxplot, Density plot.
    
    :param df: Pandas DataFrame containing the data
    """
    # Convert datetime column to actual datetime format
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Selecting only numerical columns
    numerical_cols = df.select_dtypes(include=["number"]).columns

    # Iterate over each numerical column
    for col in numerical_cols:
        plt.figure(figsize=(18, 5))

        # Histogram
        plt.subplot(1, 3, 1)
        sns.histplot(df[col], bins=30, kde=True, color='blue')
        plt.title(f"Histogram of {col}")

        # Boxplot
        plt.subplot(1, 3, 2)
        sns.boxplot(y=df[col], color='green')
        plt.title(f"Boxplot of {col}")

        # Density Plot
        plt.subplot(1, 3, 3)
        sns.kdeplot(df[col], fill=True, color='red')
        plt.title(f"Density Plot of {col}")

        st.pyplot(plt)

        # Statistical Summary
        stats = df[col].describe()
        skewness = df[col].skew()
        kurtosis = df[col].kurtosis()

        st.write(f"\n🔹 Statistical Summary for {col}:\n")
        st.write(stats)
        st.write(f"Skewness: {skewness}")
        st.write(f"Kurtosis: {kurtosis}\n")
        st.write("-" * 50)

def correlation_analysis(df, threshold=0.75):
    """
    Computes and visualizes the correlation matrix for numerical features.
    Identifies multicollinearity issues by flagging highly correlated features.

    Parameters:
    df (pd.DataFrame): Input DataFrame
    threshold (float): Correlation threshold for identifying multicollinearity (default = 0.75)

    Returns:
    high_corr_pairs (list): List of highly correlated feature pairs
    """
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns

    # Compute correlation matrix
    corr_matrix = df[numerical_cols].corr()

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix Heatmap")
    st.pyplot(plt)

    # Identifying Multicollinearity (correlation > threshold)
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

    # Print highly correlated features
    if high_corr_pairs:
        st.write("\n🔹 Highly Correlated Feature Pairs (|correlation| > {}):".format(threshold))
        for feature1, feature2, correlation in high_corr_pairs:
            st.write(f"{feature1} ↔ {feature2} | Correlation: {correlation:.2f}")
    else:
        st.write("\n✅ No strong multicollinearity detected (correlation > {}).".format(threshold))
    
    return high_corr_pairs

def time_series_analysis(df, date_col, target_col, period=24):
    """
    Performs time series decomposition and stationarity testing (ADF Test).

    Parameters:
    df (pd.DataFrame): Input DataFrame with time series data
    date_col (str): Column name containing datetime information
    target_col (str): Column name for the time series variable (e.g., demand)
    period (int): Seasonal period for decomposition (default = 24 for hourly data)

    Returns:
    None
    """
    # Convert date column to datetime
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)

    # Time Series Decomposition
    decomposition = sm.tsa.seasonal_decompose(df[target_col], model='additive', period=period)

    # Plot decomposition
    fig, axes = plt.subplots(4, 1, figsize=(12, 8))
    decomposition.observed.plot(ax=axes[0], title="Observed")
    decomposition.trend.plot(ax=axes[1], title="Trend")
    decomposition.seasonal.plot(ax=axes[2], title="Seasonality")
    decomposition.resid.plot(ax=axes[3], title="Residuals", linestyle='dashed')
    for ax in axes:
        ax.set_xlabel("Date")
    
    plt.tight_layout()
    st.pyplot(fig)

    # Stationarity Test: Augmented Dickey-Fuller Test
    st.write("\n📉 Augmented Dickey-Fuller Test Results:")
    adf_test = adfuller(df[target_col].dropna())
    results = pd.Series(adf_test[:4], index=['Test Statistic', 'p-value', '# Lags Used', '# Observations Used'])
    for key, value in adf_test[4].items():
        results[f'Critical Value ({key})'] = value

    st.write(results)

    # Interpretation
    if adf_test[1] < 0.05:
        st.write("\n✅ The time series is stationary (p-value < 0.05).")
    else:
        st.write("\n⚠️ The time series is non-stationary (p-value >= 0.05). Consider differencing or detrending.")

    # Reset index after analysis
    df.reset_index(inplace=True)

def check_demand_in_each_city(data):
    # Extract province columns (all columns starting with "Province_")
    province_columns = [col for col in data.columns if col.startswith("Province_")]
    
    # Multiply each one-hot encoded province column by the demand to distribute demand correctly
    province_demand = data[province_columns].multiply(data["demand_mwh"], axis=0).sum()

    # Sort values
    province_demand = province_demand.sort_values()

    # Plot
    plt.figure(figsize=(12, 6))
    sns.barplot(y=province_demand.index.str.replace("Province_", ""), x=province_demand.values, palette="coolwarm", orient='h')
    plt.ylabel("Province")
    plt.xlabel("Total Electricity Demand (MWh)")
    plt.title("Electricity Demand by Province")
    st.pyplot(plt)

    return province_demand

def perform_eda(df):
    check_demand_in_each_city(df)

    statistical_summary_result = statistical_summary(df)

    plot_time_series(df)

    univariate_analysis(df)

    high_corr_pairs = correlation_analysis(df)

    time_series_analysis(df, "datetime", "demand_mwh")