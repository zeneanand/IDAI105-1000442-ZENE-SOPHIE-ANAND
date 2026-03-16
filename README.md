
# 🏦 FA-2: ATM Intelligence Demand Forecasting

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://idai105-1000442-zene-sophie-anand-fa2.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)]()

An interactive, AI-driven Business Intelligence dashboard built with **Streamlit**. This project applies Advanced Data Mining techniques to analyze ATM cash demand, segment locations based on withdrawal behaviors, and detect highly unusual transaction spikes.

---

## 🎓 **Academic Details**
* **Developer:** Zene-Sophie-Anand
* **WACP No:** 1000442
* **CRS:** Python / Artificial Intelligence
* **Course:** IBCP (AI)
* **Institution:** Aspen Nutan Academy

---

## 🚀 **Project Scope & Objectives**
The primary goal of this project is to optimize cash replenishment strategies for FinTrust Bank Ltd. by analyzing historical ATM transaction data. By deploying an interactive web application, stakeholders can explore data visually and make data-driven decisions regarding cash logistics.

### **Core Objectives:**
1. **Explore Trends:** Understand baseline cash demand across different days, locations, and weather conditions.
2. **Segment ATMs:** Group ATMs by their deposit and withdrawal volumes to prioritize high-traffic machines.
3. **Flag Anomalies:** Detect sudden, extreme spikes in cash withdrawals to prevent ATM cash-outs.

---

## 🧠 **Advanced Analytics & AI Methodology**

### 📊 **1. Exploratory Data Analysis (EDA)**
* Visualized the distribution of Total Withdrawals and Deposits.
* Analyzed the impact of **Holidays** and **Extreme Weather** on cash demand.
* Generated a Feature Correlation Heatmap to identify relationships between previous day cash levels and current day withdrawals.

### 🎯 **2. ATM Clustering (K-Means)**
* **Algorithm:** K-Means Clustering (`scikit-learn`).
* **Process:** Standardized the `Total_Withdrawals` and `Total_Deposits` features and grouped the ATMs into **3 distinct clusters**.
* **Business Labels Applied:** 
  * 🔴 *High-Demand:* Requires multi-day or frequent cash stocking.
  * 🔵 *Steady-Demand:* Predictable, baseline cash requirements.
  * 🟢 *Low-Demand:* Holds idle cash that can be optimized or rerouted.

### 🚨 **3. Anomaly Detection (Isolation Forest)**
* **Algorithm:** Isolation Forest (`scikit-learn`).
* **Process:** Analyzed transaction volumes to isolate statistical outliers without relying on rigid, hard-coded thresholds.
* **Business Value:** Automatically flags highly unusual withdrawal events (often correlating with holidays or local events) that deviate from the standard baseline, allowing the bank to proactively prevent cash shortages.

---

## 📸 UI Screenshots & Dashboard Views

<table>
  <tr>
    <td align="center">
      <img src="docs/screenshot1.png" width="250"/><br>
      <b>1. Main Dashboard </b>
    </td>
    <td align="center">
      <img src="docs/screenshot2.png" width="250"/><br>
      <b>2. Global Filter Location Type (EDA)</b>
    </td>
    <td align="center">
      <img src="docs/screenshot3.png" width="250"/><br>
      <b>3. Global Filter Weather Condition </b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="docs/screenshot4.png" width="250"/><br>
      <b>4. Exploring Data Analytics</b>
    </td>
    <td align="center">
      <img src="docs/screenshot5.png" width="250"/><br>
      <b>5. Impact Of Holidays On Average Withdrawals</b>
    </td>
    <td align="center">
      <img src="docs/screenshot6.png" width="250"/><br>
      <b>6. AI Powered ATM Clustering</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="docs/screenshot7.png" width="250"/><br>
      <b>7. Anomaly & Spike Detection</b>
    </td>
    <td></td>
    <td></td>
  </tr>
</table>

---
## 💡 **Key Business Insights Generated**
1. **Holiday Volatility:** Withdrawals show massive demand spikes on and immediately preceding recorded holidays.
2. **Location-Based Strategy:** Urban and heavy commercial zones correlate strongly with the 'High-Demand' K-Means cluster.
3. **Weather Impacts:** Extreme weather conditions occasionally suppress baseline withdrawals but can cause localized panic-withdrawal anomalies.

---

## 📂 **Repository Structure**

```text
IDAI105(1000442)-zene-sophie-anand/
│
├── app.py                 # Main Streamlit dashboard and ML pipeline
├── requirements.txt       # Python library dependencies
├── data/
│   └── dummy_atm_data.csv # Dataset (Generated in-script for reproducibility)
│
└── README.md              # Project documentation and rubric evidence

```

---

## 🛠️ **Installation & Deployment**

### **Live Application**

The project is fully deployed on Streamlit Community Cloud. You can interact with the live dashboard here:
👉 **[Insert Your Streamlit App Link Here]**

### **Local Setup**

To run this project on your local machine:

1. Clone the repository:
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)

```


2. Install the required dependencies:
```bash
pip install -r requirements.txt

```


3. Run the Streamlit server:
```bash
streamlit run app.py

```

### 👥 Collaborators

| Name | WACP NO |
|------|---------|
| ZENE SOPHIE ANAND | 1000442 |
| NAMAN OM SHRESHTA | 1000432 |
| NISHTHA PRIYESH SHAH  | 1000436 |
