
# Step 1: Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Step 2: Load Your Dataset
df = pd.read_csv("credit_fraud_data.csv")

print(df.head())
print(df.info())
print(df.columns)

# Step 3: Define Features (X) and Target (y)
X = df.drop(columns=['IsFraud'])
y = df['IsFraud']

# Step 4: Identify Numerical & Categorical Columns
X = X.drop(columns=['TransactionID', 'CustomerID', 'TransactionTime'])

numeric_features = ['TransactionAmount', 'AccountAgeDays', 'RiskScore', 'Hour']
categorical_features = ['MerchantCategory', 'TransactionMode', 'Location', 'DayOfWeek']

# Step 5: Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


# Step 6: Build Final ML Pipeline
model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# Step 8: Train the Model
model.fit(X_train, y_train)

# Step 9: Predictions
y_pred = model.predict(X_test)

# Step 10: Evaluation Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))