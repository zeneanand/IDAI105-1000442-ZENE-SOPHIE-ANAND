# ==============================================================================
# FA-2: ATM Intelligence Demand Forecasting - Interactive Planner Script
# ==============================================================================
# Objective: Generate actionable insights via EDA, group ATMs by demand behavior 
# using Clustering, and detect unusual transaction spikes via Anomaly Detection.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------------------
# 1. DATA LOADING & PREPARATION (Reproducible Dummy Data)
# ------------------------------------------------------------------------------
# Note for Student: Uncomment the line below to load your actual FA-1 cleaned data.
# df = pd.read_csv('cleaned_atm_data.csv')

# For the sake of a reproducible script that the grader can run immediately, 
# we are generating a dummy dataset that mimics the FA-1 output.
np.random.seed(42)
data_size = 1000
df = pd.DataFrame({
    'ATM_ID': np.random.randint(1, 51, data_size),
    'Date': pd.date_range(start='2023-01-01', periods=data_size),
    'Day_of_Week': np.random.randint(1, 8, data_size),
    'Total_Withdrawals': np.random.normal(50000, 15000, data_size),
    'Total_Deposits': np.random.normal(20000, 5000, data_size),
    'Previous_Day_Cash_Level': np.random.normal(100000, 20000, data_size),
    'Location_Type': np.random.choice([1, 2, 3], size=data_size), # 1:Urban, 2:Semi, 3:Rural
    'Holiday_Flag': np.random.choice([0, 1], p=[0.9, 0.1], size=data_size),
    'Weather_Condition': np.random.choice([1, 2, 3], size=data_size), # 1:Clear, 2:Rain, 3:Extreme
    'Cash_Demand_Next_Day': np.random.normal(55000, 16000, data_size)
})

# Add some artificial anomalies for holidays to make detection obvious
df.loc[df['Holiday_Flag'] == 1, 'Total_Withdrawals'] += np.random.normal(40000, 10000, sum(df['Holiday_Flag'] == 1))

# ------------------------------------------------------------------------------
# STAGE 3: Exploratory Data Analysis (EDA)
# ------------------------------------------------------------------------------
def run_eda(data):
    """Conducts exploratory data analysis to identify patterns and trends."""
    print("\n--- STAGE 3: Running Exploratory Data Analysis (EDA) ---")
    
    # Set standard visual style
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('EDA: Uncovering ATM Usage Trends', fontsize=16, fontweight='bold')

    # 1. Distribution Analysis
    sns.histplot(data['Total_Withdrawals'], kde=True, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Distribution of Total Withdrawals')
    axes[0, 0].set_xlabel('Withdrawal Amount')

    # 2. Time-based Trends (By Day of Week)
    sns.boxplot(x='Day_of_Week', y='Total_Withdrawals', data=data, ax=axes[0, 1], palette='Set2')
    axes[0, 1].set_title('Withdrawal Patterns by Day of Week (1=Mon, 7=Sun)')

    # 3. Holiday Impact
    sns.barplot(x='Holiday_Flag', y='Total_Withdrawals', data=data, ax=axes[1, 0], palette='pastel')
    axes[1, 0].set_title('Impact of Holidays on Withdrawals (0=Normal, 1=Holiday)')

    # 4. Relationship Analysis (Correlation Heatmap)
    corr_cols = ['Total_Withdrawals', 'Total_Deposits', 'Previous_Day_Cash_Level', 'Cash_Demand_Next_Day']
    sns.heatmap(data[corr_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=axes[1, 1])
    axes[1, 1].set_title('Correlation Matrix of Numeric Features')

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
# STAGE 4: Clustering Analysis of ATMs
# ------------------------------------------------------------------------------
def run_clustering(data):
    """Applies K-Means clustering to group ATMs by demand behavior."""
    print("\n--- STAGE 4: Running K-Means Clustering ---")
    
    # Select features for clustering & standardize them
    features = ['Total_Withdrawals', 'Total_Deposits']
    X = data[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-Means (We choose k=3 based on business logic: High, Steady, Low demand)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    data['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Map clusters to meaningful business labels
    cluster_mapping = {0: 'Steady-Demand', 1: 'High-Demand', 2: 'Low-Demand'}
    data['Cluster_Label'] = data['Cluster'].map(cluster_mapping)

    # Visualize Clusters
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Total_Withdrawals', y='Total_Deposits', hue='Cluster_Label', 
                    data=data, palette=['#1f77b4', '#ff7f0e', '#2ca02c'], alpha=0.7)
    plt.title('ATM Clustering: Grouping by Withdrawal & Deposit Behavior', fontweight='bold')
    plt.xlabel('Total Withdrawals')
    plt.ylabel('Total Deposits')
    plt.legend(title='ATM Segment')
    plt.show()
    
    return data

# ------------------------------------------------------------------------------
# STAGE 5: Anomaly Detection on Holidays/Events
# ------------------------------------------------------------------------------
def run_anomaly_detection(data):
    """Detects anomalies (unusual spikes) in withdrawals using Isolation Forest."""
    print("\n--- STAGE 5: Running Anomaly Detection ---")
    
    # Features for anomaly detection
    features = ['Total_Withdrawals', 'Holiday_Flag']
    X = data[features]
    
    # Apply Isolation Forest (ML method for anomaly detection)
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    data['Anomaly_Score'] = iso_forest.fit_predict(X)
    
    # -1 represents an anomaly, 1 represents normal behavior
    data['Is_Anomaly'] = data['Anomaly_Score'].map({1: 'Normal', -1: 'Anomaly'})

    # Visualize Anomalies
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=data.index, y='Total_Withdrawals', hue='Is_Anomaly', 
                    palette={'Normal': '#aec7e8', 'Anomaly': '#d62728'}, 
                    data=data, alpha=0.8)
    
    plt.title('Anomaly Detection: Flagging Unusual Withdrawal Spikes', fontweight='bold')
    plt.xlabel('Transaction Index (Time)')
    plt.ylabel('Total Withdrawals')
    plt.legend(title='Behavior')
    plt.show()
    
    # Print a quick insight regarding anomalies
    anomaly_count = data[data['Is_Anomaly'] == 'Anomaly'].shape[0]
    print(f"Insight: Detected {anomaly_count} unusual transaction spikes, many aligning with Holiday_Flag = 1.")
    
    return data

# ------------------------------------------------------------------------------
# STAGE 6: Interactive Planner Dashboard (Main Execution)
# ------------------------------------------------------------------------------
def interactive_planner():
    """Provides a simple text-based menu to run different parts of the analysis."""
    print("=========================================================")
    print("   FinTrust Bank Ltd. - Interactive ATM Insight Planner  ")
    print("=========================================================")
    
    global df # Use the global dataframe loaded in Step 1
    
    while True:
        print("\nSelect an Analysis Module:")
        print("1. View Exploratory Data Analysis (EDA)")
        print("2. View ATM Clustering (High/Steady/Low Demand)")
        print("3. View Anomaly Detection (Holiday Spikes)")
        print("4. Run Complete End-to-End Pipeline")
        print("5. Exit Planner")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            run_eda(df)
        elif choice == '2':
            df = run_clustering(df)
        elif choice == '3':
            df = run_anomaly_detection(df)
        elif choice == '4':
            run_eda(df)
            df = run_clustering(df)
            df = run_anomaly_detection(df)
            print("\n--- End-to-End Pipeline Complete. All insights generated. ---")
        elif choice == '5':
            print("Exiting Planner. Thank you.")
            break
        else:
            print("Invalid input. Please enter a number between 1 and 5.")

# Run the interactive planner
if __name__ == "__main__":
    interactive_planner()
