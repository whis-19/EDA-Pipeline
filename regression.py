import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st
from paths import *  

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test) 
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return y_pred, mse, rmse, r2

def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
    plt.xlabel("Actual Electricity Demand")
    plt.ylabel("Predicted Electricity Demand")
    plt.title("Actual vs. Predicted Electricity Demand")
    st.pyplot(plt)

def residual_analysis(y_test, y_pred):
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, bins=30, kde=True)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel("Residuals")
    plt.ylabel("Frequency")
    plt.title("Residual Analysis")
    st.pyplot(plt)

def plot_model_performance(mse, rmse, r2):
    metrics = [mse, rmse, r2]
    metric_names = ['Mean Squared Error', 'Root Mean Squared Error', 'R² Score']
    
    plt.figure(figsize=(8, 5))
    sns.barplot(x=metric_names, y=metrics, palette='viridis')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics')
    plt.ylim(0, max(metrics) * 1.1)  
    st.pyplot(plt)

def regression():
    features = [
        "extracted_period_hour", "extracted_period_day", "extracted_period_month", 
        "extracted_period_dayofweek", "temperature_2m"
    ]
    target = "value"  
    # Load and preprocess data
    df = pd.read_csv(input_file)
    df[features] = df[features].fillna(df[features].median())
    df[target] = df[target].fillna(df[target].median())
    x,y = df[features],df[target]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)


    # Evaluate model
    y_pred, mse, rmse, r2 = evaluate_model(model, X_test, y_test)

    # Print evaluation metrics
    st.write(f"Mean Squared Error (MSE): {mse}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")
    st.write(f"R² Score: {r2}")
    st.write("Model R^2 Score:", model.score(X_test, y_test))

    # Plot results
    plot_actual_vs_predicted(y_test, y_pred)
    residual_analysis(y_test, y_pred)
    plot_model_performance(mse, rmse, r2)
