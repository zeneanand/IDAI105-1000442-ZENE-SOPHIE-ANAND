import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# --- PAGE SETUP & CSS ---
st.set_page_config(page_title="ATM Intelligence", page_icon="🏦", layout="wide")

st.markdown("""
    <style>
    /* Dark Finance Theme */
    .stApp { background-color: #0B1120; color: #F8FAFC; }
    div[data-testid="stMetricValue"] { font-size: 32px; font-weight: 800; color: #38BDF8; } /* Light Blue Accent */
    div[data-testid="stMetricLabel"] { color: #94A3B8; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
    h1, h2, h3 { color: #FFFFFF; font-family: 'Inter', sans-serif; font-weight: 600; }
    section[data-testid="stSidebar"] { background-color: #1E293B; border-right: 1px solid #334155; }
    .st-bb { border-bottom: 1px solid #334155; }
    </style>
    """, unsafe_allow_html=True)

# --- BULLETPROOF DATA LOADING ---
@st.cache_data
def load_atm_data_with_weather():
    np.random.seed(42)
    data_size = 1000
    df = pd.DataFrame({
        'ATM_ID': np.random.randint(1, 51, data_size),
        'Date': pd.date_range(start='2023-01-01', periods=data_size),
        'Day_of_Week': np.random.choice(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], size=data_size),
        # Absolute value ensures no negative numbers crash our interactive charts
        'Total_Withdrawals': np.abs(np.random.normal(50000, 15000, data_size)) + 1000,
        'Total_Deposits': np.abs(np.random.normal(20000, 5000, data_size)) + 1000,
        'Previous_Day_Cash_Level': np.abs(np.random.normal(100000, 20000, data_size)),
        'Location_Type': np.random.choice(['Urban', 'Semi-Urban', 'Rural'], size=data_size), 
        'Holiday_Flag': np.random.choice(['Normal Day', 'Holiday'], p=[0.9, 0.1], size=data_size),
        'Weather_Condition': np.random.choice(['Clear', 'Rain', 'Extreme'], p=[0.7, 0.2, 0.1], size=data_size)
    })
    # Add artificial anomalies for holidays and extreme weather
    df.loc[df['Holiday_Flag'] == 'Holiday', 'Total_Withdrawals'] += np.random.normal(40000, 10000, sum(df['Holiday_Flag'] == 'Holiday'))
    df.loc[df['Weather_Condition'] == 'Extreme', 'Total_Withdrawals'] += np.random.normal(20000, 5000, sum(df['Weather_Condition'] == 'Extreme'))
    return df

df = load_atm_data_with_weather()

# --- SIDEBAR NAVIGATION & FILTERS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=80)
    st.title("Admin Console")
    menu = st.radio("Navigation Module:", [
        "📊 1. Overview & EDA", 
        "🎯 2. ATM Clustering", 
        "🚨 3. Anomaly Detection"
    ])
    st.divider()
    
    # Interactive Filters
    st.subheader("Global Filters")
    location_filter = st.selectbox("Location Type:", ["All", "Urban", "Semi-Urban", "Rural"])
    weather_filter = st.selectbox("Weather Condition:", ["All", "Clear", "Rain", "Extreme"])
    
    # Apply Filters
    if location_filter != "All":
        df = df[df['Location_Type'] == location_filter]
    if weather_filter != "All":
        df = df[df['Weather_Condition'] == weather_filter]
        
    st.write(f"**Records Loaded:** {len(df):,}")

# --- GLOBAL HEADER & KPIs ---
st.title("🏦 FinTrust Bank: ATM Intelligence Dashboard")
st.markdown("Advanced AI forecasting to explore ATM usage, segment locations, and detect withdrawal anomalies.")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Active ATMs", df['ATM_ID'].nunique())
kpi2.metric("Avg Daily Withdrawals", f"${df['Total_Withdrawals'].mean():,.0f}")
kpi3.metric("Avg Daily Deposits", f"${df['Total_Deposits'].mean():,.0f}")
kpi4.metric("Extreme Weather Days", len(df[df['Weather_Condition'] == 'Extreme']))
st.divider()

# --- MODULE 1: EDA ---
if menu == "📊 1. Overview & EDA":
    st.subheader("Exploratory Data Analysis (EDA)")
    
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.histogram(df, x='Total_Withdrawals', nbins=30, 
                            title="Distribution of Total Withdrawals",
                            color_discrete_sequence=['#38BDF8'], template="plotly_dark")
        st.plotly_chart(fig1, use_container_width=True)
        
        holiday_data = df.groupby('Holiday_Flag')['Total_Withdrawals'].mean().reset_index()
        fig3 = px.bar(holiday_data, x='Holiday_Flag', y='Total_Withdrawals', color='Holiday_Flag',
                      title="Impact of Holidays on Avg Withdrawals",
                      color_discrete_map={'Normal Day': '#3B82F6', 'Holiday': '#EF4444'}, template="plotly_dark")
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig2 = px.box(df, x='Weather_Condition', y='Total_Withdrawals', color='Weather_Condition',
                      title="Withdrawal Patterns by Weather",
                      color_discrete_map={'Clear': '#10B981', 'Rain': '#3B82F6', 'Extreme': '#EF4444'},
                      template="plotly_dark")
        st.plotly_chart(fig2, use_container_width=True)
        
        numeric_df = df[['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level']]
        fig4 = px.imshow(numeric_df.corr(), text_auto=".2f", aspect="auto", 
                         title="Feature Correlation Heatmap",
                         color_continuous_scale="Blues", template="plotly_dark")
        st.plotly_chart(fig4, use_container_width=True)

# --- MODULE 2: CLUSTERING ---
elif menu == "🎯 2. ATM Clustering":
    st.subheader("AI-Powered ATM Clustering")
    st.info("Grouping ATMs based on Withdrawal and Deposit behaviors to optimize cash routing.")
    
    features = ['Total_Withdrawals', 'Total_Deposits']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    df['Segment'] = df['Cluster'].map({0: 'Steady-Demand', 1: 'High-Demand', 2: 'Low-Demand'})

    fig5 = px.scatter(df, x='Total_Withdrawals', y='Total_Deposits', color='Segment', 
                      hover_data=['ATM_ID', 'Location_Type', 'Weather_Condition'],
                      title="ATM Segments: High vs Low Demand",
                      color_discrete_map={'Steady-Demand': '#3B82F6', 'High-Demand': '#EF4444', 'Low-Demand': '#10B981'},
                      template="plotly_dark", opacity=0.8)
    st.plotly_chart(fig5, use_container_width=True)
    
    st.success("🎯 **Strategic Insight:** High-Demand ATMs (Red) require multi-day cash stocking, while Low-Demand ATMs (Green) hold idle cash that can be rerouted.")

# --- MODULE 3: ANOMALY DETECTION ---
elif menu == "🚨 3. Anomaly Detection":
    st.subheader("Anomaly Detection (Spike Alerts)")
    st.write("Using Isolation Forest Machine Learning to flag unusual, high-volume cash withdrawal events.")
    
    # Isolation Forest Model
    features = ['Total_Withdrawals']
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    df['Anomaly_Score'] = iso_forest.fit_predict(df[features])
    df['Behavior'] = df['Anomaly_Score'].map({1: 'Normal', -1: 'Anomaly'})

    fig6 = px.scatter(df, x=df.index, y='Total_Withdrawals', color='Behavior',
                      hover_data=['ATM_ID', 'Holiday_Flag', 'Location_Type', 'Weather_Condition'],
                      title="Flagging Unusual Withdrawal Spikes",
                      color_discrete_map={'Normal': '#334155', 'Anomaly': '#EF4444'},
                      template="plotly_dark")
    st.plotly_chart(fig6, use_container_width=True)
    
    anomaly_count = df[df['Behavior'] == 'Anomaly'].shape[0]
    st.error(f"⚠️ **Alert:** Detected **{anomaly_count}** unusual transaction spikes. Hover over the red dots to see how they align with holidays or extreme weather!")
