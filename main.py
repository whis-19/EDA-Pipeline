from eda import perform_eda
from loader import merge_data
from processor import process_data,normalize_data
from outlier import outliers
from regression import regression_model
import streamlit as st

if __name__ == "__main__":
    st.title("Exploratory Data Analysis and Regression Model")

    # Data Loading
    st.header("Data Loading")
    raw_df = merge_data()
    raw_df.to_csv("raw_merged_data.csv", index=False)
    st.write("Raw data loaded and saved as 'raw_merged_data.csv'.")

    # Data Processing
    st.header("Data Processing")
    processed_df = process_data(raw_df)
    processed_df.to_csv("processed_data.csv", index=False)
    st.write("Processed data saved as 'processed_data.csv'.")

    # EDA
    st.header("Exploratory Data Analysis (EDA)")
    perform_eda(processed_df)
    st.write("EDA performed on processed data.")

    # Outlier Detection
    st.header("Outlier Detection")
    cleaned_df = outliers(processed_df)
    cleaned_df.to_csv("cleaned_data.csv", index=False)
    st.write("Outliers detected and cleaned data saved as 'cleaned_data.csv'.")

    # Normalization
    st.header("Normalization")
    normalized_df = normalize_data(cleaned_df)
    normalized_df.to_csv("normalized_data.csv", index=False)
    st.write("Normalized data saved as 'normalized_data.csv'.")

    # Regression Model
    st.header("Regression Model")
    model, predictions = regression_model(normalized_df, target="demand_mwh", time_column="datetime")
    st.write("Regression model built and evaluated.")