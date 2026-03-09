import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# --- PAGE SETUP ---
st.set_page_config(page_title="ATM Intelligence Planner", layout="wide")
st.title("🏦 FA-2: ATM Intelligence Demand Forecasting")
st.markdown("Interactive dashboard for FinTrust Bank Ltd. to explore ATM usage, clusters, and anomalies.")

# --- DATA LOADING (Cached so it doesn't reload every click) ---
@st.cache_data
def load_data():
    np.random.seed(42)
    data_size = 1000
    df = pd.DataFrame({
        'ATM_ID': np.random.randint(1, 51, data_size),
        'Date': pd.date_range(start='2023-01-01', periods=data_size),
        'Day_of_Week': np.random.randint(1, 8, data_size),
        'Total_Withdrawals': np.random.normal(50000, 15000, data_size),
        'Total_Deposits': np.random.normal(20000, 5000, data_size),
        'Previous_Day_Cash_Level': np.random.normal(100000, 20000, data_size),
        'Location_Type': np.random.choice(['Urban', 'Semi-Urban', 'Rural'], size=data_size), 
        'Holiday_Flag': np.random.choice([0, 1], p=[0.9, 0.1], size=data_size),
    })
    # Add artificial anomalies for holidays
    df.loc[df['Holiday_Flag'] == 1, 'Total_Withdrawals'] += np.random.normal(40000, 10000, sum(df['Holiday_Flag'] == 1))
    return df

df = load_data()

# --- SIDEBAR & INTERACTIVITY ---
st.sidebar.header("Navigation & Filters")
menu = st.sidebar.radio("Select Analysis Module:", 
                        ["1. Exploratory Data Analysis (EDA)", 
                         "2. ATM Clustering", 
                         "3. Anomaly Detection"])

# Add the required interactive filter
location_filter = st.sidebar.selectbox("Filter by Location Type:", ["All", "Urban", "Semi-Urban", "Rural"])
if location_filter != "All":
    df = df[df['Location_Type'] == location_filter]
    st.write(f"**Currently viewing data for: {location_filter} ATMs**")

# --- MODULE 1: EDA ---
if menu == "1. Exploratory Data Analysis (EDA)":
    st.header("Exploratory Data Analysis (EDA)")
    st.write("Uncovering trends and patterns in the dataset before applying advanced analysis.")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sns.histplot(df['Total_Withdrawals'], kde=True, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Distribution of Total Withdrawals')
    
    sns.boxplot(x='Day_of_Week', y='Total_Withdrawals', data=df, ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_title('Withdrawal Patterns by Day of Week')
    
    sns.barplot(x='Holiday_Flag', y='Total_Withdrawals', data=df, ax=axes[1, 0], palette='pastel')
    axes[1, 0].set_title('Impact of Holidays on Withdrawals')
    
    # Correlation needs numeric columns only
    numeric_df = df[['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level']]
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1, 1])
    axes[1, 1].set_title('Correlation Matrix')
    
    plt.tight_layout()
    st.pyplot(fig) # This is how Streamlit draws graphs!

# --- MODULE 2: CLUSTERING ---
elif menu == "2. ATM Clustering":
    st.header("Clustering Analysis of ATMs")
    st.write("Grouping ATMs into clusters to categorize them based on demand behavior.")
    
    features = ['Total_Withdrawals', 'Total_Deposits']
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    cluster_mapping = {0: 'Steady-Demand', 1: 'High-Demand', 2: 'Low-Demand'}
    df['Cluster_Label'] = df['Cluster'].map(cluster_mapping)

    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Total_Withdrawals', y='Total_Deposits', hue='Cluster_Label', 
                    data=df, palette=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    plt.title('ATM Clustering: Grouping by Withdrawal & Deposit Behavior')
    st.pyplot(fig)

# --- MODULE 3: ANOMALY DETECTION ---
elif menu == "3. Anomaly Detection":
    st.header("Anomaly Detection on Holidays/Events")
    st.write("Detecting unusual or unexpected behaviors to ensure cash shortages are avoided.")
    
    features = ['Total_Withdrawals', 'Holiday_Flag']
    X = df[features]
    
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly_Score'] = iso_forest.fit_predict(X)
    df['Is_Anomaly'] = df['Anomaly_Score'].map({1: 'Normal', -1: 'Anomaly'})

    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df.index, y='Total_Withdrawals', hue='Is_Anomaly', 
                    palette={'Normal': '#aec7e8', 'Anomaly': '#d62728'}, 
                    data=df, alpha=0.8)
    plt.title('Anomaly Detection: Flagging Unusual Withdrawal Spikes')
    st.pyplot(fig)
    
    anomaly_count = df[df['Is_Anomaly'] == 'Anomaly'].shape[0]
    st.warning(f"Insight: Detected **{anomaly_count}** unusual transaction spikes. Notice how the red anomalies sit higher than the standard transaction baseline.")
