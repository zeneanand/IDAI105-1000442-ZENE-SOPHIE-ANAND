import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# --- PAGE SETUP ---
st.set_page_config(page_title="FinTrust ATM Planner", page_icon="🏦", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    h1, h2, h3 {color: #1e3d59;}
    </style>
    """, unsafe_allow_html=True)

st.title("🏦 FinTrust Bank: ATM Intelligence Dashboard")
st.markdown("Explore ATM usage trends, identify demand clusters, and flag unusual withdrawal spikes.")
st.markdown("---")

# --- DATA LOADING ---
@st.cache_data
def load_data():
    np.random.seed(42)
    data_size = 1000
    df = pd.DataFrame({
        'ATM_ID': np.random.randint(1, 51, data_size),
        'Date': pd.date_range(start='2023-01-01', periods=data_size),
        'Day_of_Week': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], size=data_size),
        'Total_Withdrawals': np.random.normal(50000, 15000, data_size),
        'Total_Deposits': np.random.normal(20000, 5000, data_size),
        'Previous_Day_Cash_Level': np.random.normal(100000, 20000, data_size),
        'Location_Type': np.random.choice(['Urban', 'Semi-Urban', 'Rural'], size=data_size), 
        'Holiday_Flag': np.random.choice([0, 1], p=[0.9, 0.1], size=data_size),
        'Weather_Condition': np.random.choice(['Clear', 'Rain', 'Extreme'], size=data_size)
    })
    # Add artificial spikes for holidays
    df.loc[df['Holiday_Flag'] == 1, 'Total_Withdrawals'] += np.random.normal(40000, 10000, sum(df['Holiday_Flag'] == 1))
    return df

df = load_data()

# --- SIDEBAR & INTERACTIVITY ---
st.sidebar.title("Navigation & Filters")

# Module Selection
menu = st.sidebar.radio("Select Analysis Module:", 
                        ["📊 1. Exploratory Data Analysis", 
                         "🎯 2. ATM Clustering", 
                         "🚨 3. Anomaly Detection"])

st.sidebar.markdown("---")

# Global Filters
st.sidebar.subheader("Filter Data:")
location_filter = st.sidebar.selectbox("Location Type:", ["All", "Urban", "Semi-Urban", "Rural"])
weather_filter = st.sidebar.selectbox("Weather Condition:", ["All", "Clear", "Rain", "Extreme"])

# Apply Filters
filtered_df = df.copy()
if location_filter != "All":
    filtered_df = filtered_df[filtered_df['Location_Type'] == location_filter]
if weather_filter != "All":
    filtered_df = filtered_df[filtered_df['Weather_Condition'] == weather_filter]

st.sidebar.success(f"Showing {len(filtered_df)} transactions.")

# --- TOP KPI METRICS ---
col1, col2, col3 = st.columns(3)
col1.metric("Total ATMs Monitored", filtered_df['ATM_ID'].nunique())
col2.metric("Avg Daily Withdrawals", f"${filtered_df['Total_Withdrawals'].mean():,.0f}")
col3.metric("Holiday Spikes Recorded", filtered_df[filtered_df['Holiday_Flag'] == 1].shape[0])
st.markdown("---")

# --- MODULE 1: EDA ---
if menu == "📊 1. Exploratory Data Analysis":
    st.header("Exploratory Data Analysis (EDA)")
    sns.set_theme(style="whitegrid")
    
    # Row 1 of charts
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribution of Withdrawals")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.histplot(filtered_df['Total_Withdrawals'], kde=True, ax=ax1, color='#3498db')
        st.pyplot(fig1)
        st.info("Insight: Most withdrawals hover around the baseline, with a tail of high-demand days.")

    with c2:
        st.subheader("Withdrawals by Day of Week")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.boxplot(x='Day_of_Week', y='Total_Withdrawals', data=filtered_df, ax=ax2, palette='Set2')
        st.pyplot(fig2)
        st.info("Insight: Notice the volatility and spread changes depending on the day of the week.")

    # Row 2 of charts
    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Impact of Holidays")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.barplot(x='Holiday_Flag', y='Total_Withdrawals', data=filtered_df, ax=ax3, palette='pastel')
        ax3.set_xticklabels(['Normal Day', 'Holiday'])
        st.pyplot(fig3)

    with c4:
        st.subheader("Correlation Heatmap")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        numeric_cols = filtered_df[['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level']]
        sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)
        st.pyplot(fig4)

# --- MODULE 2: CLUSTERING ---
elif menu == "🎯 2. ATM Clustering":
    st.header("Clustering Analysis of ATMs")
    st.write("Using K-Means to group ATMs based on their Withdrawal and Deposit behaviors.")
    
    # Preprocessing for Clustering
    features = ['Total_Withdrawals', 'Total_Deposits']
    X = filtered_df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means Model
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    filtered_df['Cluster'] = kmeans.fit_predict(X_scaled)
    cluster_mapping = {0: 'Steady-Demand', 1: 'High-Demand', 2: 'Low-Demand'}
    filtered_df['Cluster_Label'] = filtered_df['Cluster'].map(cluster_mapping)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='Total_Withdrawals', y='Total_Deposits', hue='Cluster_Label', 
                    data=filtered_df, palette=['#3498db', '#e74c3c', '#2ecc71'], alpha=0.7, ax=ax)
    plt.title('ATM Clustering: Grouping by Withdrawal & Deposit Behavior', fontweight='bold')
    st.pyplot(fig)
    
    st.success("Insight: High-Demand ATMs (Red) require more frequent cash replenishment, whereas Low-Demand ATMs (Green) risk holding idle cash.")

# --- MODULE 3: ANOMALY DETECTION ---
elif menu == "🚨 3. Anomaly Detection":
    st.header("Anomaly Detection on Holidays/Events")
    st.write("Using Isolation Forest to detect unusual withdrawal spikes that could lead to cash-outs.")
    
    # Isolation Forest Model
    features = ['Total_Withdrawals', 'Holiday_Flag']
    X = filtered_df[features]
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    filtered_df['Anomaly_Score'] = iso_forest.fit_predict(X)
    filtered_df['Behavior'] = filtered_df['Anomaly_Score'].map({1: 'Normal', -1: 'Anomaly'})

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(x=filtered_df.index, y='Total_Withdrawals', hue='Behavior', 
                    palette={'Normal': '#95a5a6', 'Anomaly': '#c0392b'}, 
                    data=filtered_df, alpha=0.8, ax=ax)
    
    plt.title('Anomaly Detection: Flagging Unusual Withdrawal Spikes', fontweight='bold')
    plt.xlabel('Transaction Index (Time)')
    st.pyplot(fig)
    
    anomaly_count = filtered_df[filtered_df['Behavior'] == 'Anomaly'].shape[0]
    st.warning(f"**Insight:** Detected **{anomaly_count}** unusual transaction spikes. Notice how the red anomalies sit distinctly above the standard transaction baseline.")
