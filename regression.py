import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

def regression_model(df, target, time_column, test_size=0.2):
    """
    Builds and evaluates a regression model to predict electricity demand.

    Parameters:
        df (pd.DataFrame): The preprocessed dataset.
        target (str): The column name of the target variable (e.g., electricity demand).
        time_column (str): The column representing timestamps.
        test_size (float): Proportion of data used for testing (default = 0.2).

    Returns:
        model (LinearRegression): Trained regression model.
        predictions (np.array): Predictions on test data.
    """


    # Define Features (Excluding Non-Numeric Columns)
    feature_cols = ["hour", "day", "month", "year", "day_of_week", "is_weekend", "season_Spring", "season_Summer", "season_Winter"]
    if "temperature" in df.columns:
        feature_cols.append("temperature")  # Include temperature if available

    X = df[feature_cols]
    y = df[target]

    # Split into Training & Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluate Model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    st.write("Model Performance Metrics:")
    st.write(f"MSE: {mse:.2f}")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R² Score: {r2:.2f}")
    st.write(f"Model Score: {model.score(X_test, y_test):.2f}")

    # Plot Actual vs Predicted
    plt.figure(figsize=(10, 5))
    plt.scatter(y_test, y_pred, alpha=0.5, color="blue")
    plt.plot([y.min(), y.max()], [y.min(), y.max()], '--r', linewidth=2)  # Perfect fit line
    plt.xlabel("Actual Demand")
    plt.ylabel("Predicted Demand")
    plt.title("Actual vs Predicted Electricity Demand")
    st.pyplot(plt)

    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, bins=30, kde=True, color="purple")
    plt.axvline(0, color='red', linestyle='dashed', linewidth=2)
    plt.xlabel("Residuals")
    plt.title("Residual Analysis")
    st.pyplot(plt)

    return model, y_pred