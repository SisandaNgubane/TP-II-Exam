import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

st.set_page_config(page_title="SA Crime Analysis Dashboard", layout="wide")
st.title("Crime Hotspot & Crime Trend Dashboard")

@st.cache_data
def load_data():
    crime = pd.read_csv('SouthAfricaCrimeStats_v2.csv')
    sample1 = pd.read_csv('Census2022sample_F18.csv')
    sample2 = pd.read_csv('Census2022sample_F19.csv')
    sample3 = pd.read_csv('Census2022sample_F21.csv')
    return crime, sample1, sample2, sample3

CrimeStat, Sample1, Sample2, Sample3 = load_data()

# Multi-relational merge
census_merged = Sample1.merge(Sample2, on='Precinct', how='outer').merge(Sample3, on='Precinct', how='outer')
precinct_stats = CrimeStat.groupby('Precinct')['Incident_Count'].sum().reset_index()
features = precinct_stats.merge(census_merged, on='Precinct', how='left')
threshold = features['Incident_Count'].quantile(0.75)
features['Hotspot'] = (features['Incident_Count'] >= threshold).astype(int)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "EDA", "Hotspot Classification", "Forecasting"])

if page == "Overview":
    st.header("Project Overview")
    st.markdown("""
    This dashboard allows interactive analysis of South African crime data.
    - **Hotspot Classification:** Identifies top 25% crime precincts to help with patrol allocation.
    - **Crime Forecasting:** Predicts future crime trends for major categories.
    - All data sourced from SAPS Crime Stats and Census 2022.
    """)

if page == "EDA":
    st.header("Exploratory Data Analysis")
    st.subheader("Crime Category Distribution")
    fig1, ax1 = plt.subplots(figsize=(10,4))
    sns.countplot(x='Crime_Category', data=CrimeStat, ax=ax1)
    plt.xticks(rotation=90)
    st.pyplot(fig1)

    st.subheader("Incidents per Precinct (Top 20)")
    top_precincts = CrimeStat.groupby('Precinct')['Incident_Count'].sum().sort_values(ascending=False).head(20)
    st.bar_chart(top_precincts)

if page == "Hotspot Classification":
    st.header("Crime Hotspot Classification")
    st.write("Hotspots (top 25% precincts by incident count):")
    st.dataframe(features[features['Hotspot']==1][['Precinct', 'Incident_Count']].sort_values('Incident_Count', ascending=False))
    st.write(f"Threshold for hotspot: {threshold} total incidents (top 25%).")
    st.write("All features used for classification are shown in the table below:")
    st.dataframe(features.head())

if page == "Forecasting":
    st.header("Crime Trend Forecasting")
    category_option = st.selectbox("Select crime category for forecast", CrimeStat['Crime_Category'].unique())
    crime_cat_df = CrimeStat[CrimeStat['Crime_Category'] == category_option]
    crime_cat_df = crime_cat_df.dropna(subset=['Date'])
    crime_cat_df['Date'] = pd.to_datetime(crime_cat_df['Date'], errors='coerce')
    monthly_crime = crime_cat_df.groupby(pd.Grouper(key='Date', freq='M'))['Incident_Count'].sum().reset_index()
    monthly_crime = monthly_crime.rename(columns={'Date
