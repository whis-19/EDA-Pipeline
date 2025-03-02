from eda import run_eda
from loader import merger
from outlier import outlier
from processor import cleaner
from regression import regression
from paths import input_file
import streamlit as st

if __name__ == "__main__":
    st.title("Electricity Demand Forecasting")
    st.header("Introduction")
    st.write("This application performs data preprocessing, exploratory data analysis, outlier detection, and regression analysis on electricity demand data.")
    st.header("Data Loading")
    merger()
    st.header("Data Preprocessing")
    cleaner()
    st.header("Exploratory Data Analysis")
    run_eda(input_file)
    st.header("Outlier Detection")
    outlier()
    st.header("Regression Analysis")
    regression()
    
