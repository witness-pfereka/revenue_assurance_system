import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Revenue Assurance System", layout="wide")

# Title
st.title("📡 Revenue Assurance Anomaly Detection")

# File uploader
uploaded_file = st.file_uploader("Upload Revenue CSV File", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)

    st.success("✅ File uploaded successfully!")

    # --- Rule-based alert
    def rule_based_flags(df):
        return ((df['Usage_MB'] > 0) & (df['Billed_MB'] == 0)) | ((df['Revenue'] == 0) & (df['Usage_MB'] > 0))

    data['rule_flag'] = rule_based_flags(data).astype(int)

    # Simulated labels for training
    data['label'] = 0
    anomaly_idx = data.sample(frac=0.1, random_state=42).index
    data.loc[anomaly_idx, 'label'] = 1

    # Encode categorical
    encoded = pd.get_dummies(data.drop(['Customer', 'label'], axis=1))

    # Isolation Forest
    iso = IsolationForest(contamination=0.1, random_state=42)
    data['iso_flag'] = (iso.fit_predict(encoded) == -1).astype(int)

    # XGBoost
    X = encoded
    y = data['label']
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', base_score=0.5)
    xgb.fit(X, y)
    data['xgb_flag'] = xgb.predict(X)

    # Final score
    data['final_score'] = data[['rule_flag', 'iso_flag', 'xgb_flag']].sum(axis=1)
    data['final_alert'] = (data['final_score'] >= 2).astype(int)

    # Show summary
    st.subheader("🚨 Final Alert Summary")
    st.write(data['final_alert'].value_counts().rename({0: "Normal", 1: "High Risk"}))

    # Show high-risk transactions
    st.subheader("🔍 High-Risk Transactions")
    st.dataframe(data[data['final_alert'] == 1].head(10))

    # Visualizations
    st.subheader("📊 Visualizations")

    fig1, ax1 = plt.subplots()
    sns.countplot(x='final_alert', data=data, palette='Set2', ax=ax1)
    ax1.set_title("Final Alerts Distribution")
    ax1.set_xticklabels(['Normal', 'High Risk'])
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.boxplot(x='final_alert', y='Revenue', data=data, palette='coolwarm', ax=ax2)
    ax2.set_title("Revenue Distribution: Normal vs High Risk")
    ax2.set_xticklabels(['Normal', 'High Risk'])
    st.pyplot(fig2)

    # Download results
    st.subheader("⬇️ Download Flagged Results")
    @st.cache_data
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df(data)
    st.download_button(
        label="Download Results as CSV",
        data=csv,
        file_name='RA_Flagged_Results.csv',
        mime='text/csv',
    )
else:
    st.info("📂 Upload a .csv file to start analysis.")

