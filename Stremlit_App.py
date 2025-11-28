import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

# ---------------------------
# Load Dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("credit_fraud_data.csv")
    return df

df = load_data()

st.title("💳 Credit Card Fraud Detection Dashboard")
st.write("This dashboard helps analyze imbalanced credit card fraud data and run ML predictions.")

# ---------------------------
# Show Data
# ---------------------------
st.header("📌 Dataset Preview")
st.dataframe(df.head())

# ---------------------------
# EDA
# ---------------------------
st.header("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Fraud Count")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x="IsFraud", ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Transaction Amount Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df["TransactionAmount"], kde=True, ax=ax)
    st.pyplot(fig)

# ---------------------------
# Preprocessing
# ---------------------------
st.header("⚙️ Data Preprocessing")

df_processed = df.copy()

label_cols = ["MerchantCategory", "TransactionMode", "Location", "DayOfWeek"]
label_encoder = LabelEncoder()

for col in label_cols:
    df_processed[col] = label_encoder.fit_transform(df_processed[col])

# Features & Target
X = df_processed.drop(columns=["IsFraud", "TransactionID", "CustomerID", "TransactionTime"])
y = df_processed["IsFraud"]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle Imbalance with SMOTE
sm = SMOTE()
X_res, y_res = sm.fit_resample(X_scaled, y)

st.write("SMOTE Applied: Class Distribution After Oversampling")
st.write(pd.Series(y_res).value_counts())

# ---------------------------
# Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.25, random_state=42
)

# ---------------------------
# Model Training
# ---------------------------
st.header("🤖 Model Training")

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
st.subheader("📈 Model Results")
st.write("Accuracy:", accuracy_score(y_test, y_pred))

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
st.pyplot(fig)

# ---------------------------
# Live Prediction Form
# ---------------------------
st.header("🔮 Predict Fraud for New Transaction")

with st.form("prediction_form"):
    TransactionAmount = st.number_input("Transaction Amount", min_value=0.0)
    AccountAgeDays = st.number_input("Account Age (Days)", min_value=0)
    RiskScore = st.number_input("Risk Score", min_value=0)
    Hour = st.slider("Hour of Transaction", 0, 23)
    DayOfWeek = st.selectbox("Day of Week", df["DayOfWeek"].unique())
    MerchantCategory = st.selectbox("Merchant Category", df["MerchantCategory"].unique())
    TransactionMode = st.selectbox("Transaction Mode", df["TransactionMode"].unique())
    Location = st.selectbox("Location", df["Location"].unique())
    
    submitted = st.form_submit_button("Predict Fraud")

if submitted:
    # Encode inputs
    DayOfWeek_enc = label_encoder.fit_transform(df["DayOfWeek"])[
        list(df["DayOfWeek"].unique()).index(DayOfWeek)
    ]
    MerchantCategory_enc = label_encoder.fit_transform(df["MerchantCategory"])[
        list(df["MerchantCategory"].unique()).index(MerchantCategory)
    ]
    TransactionMode_enc = label_encoder.fit_transform(df["TransactionMode"])[
        list(df["TransactionMode"].unique()).index(TransactionMode)
    ]
    Location_enc = label_encoder.fit_transform(df["Location"])[
        list(df["Location"].unique()).index(Location)
    ]

    # Prepare input row
    new_data = np.array([
        TransactionAmount, MerchantCategory_enc, TransactionMode_enc,
        Location_enc, AccountAgeDays, RiskScore, Hour, DayOfWeek_enc
    ]).reshape(1, -1)

    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)[0]

    st.subheader("🟩 Prediction Result")
    st.success("Fraud Detected 🚨" if prediction == 1 else "Normal Transaction ✔️")

