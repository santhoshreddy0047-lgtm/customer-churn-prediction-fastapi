"""
Customer Churn Prediction Model Training Script
Trains a RandomForest classifier and saves it as a pickle file
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/customer_subscription_churn_dataset.csv')
print(f"Dataset shape: {df.shape}")

# Display basic info
print("\nColumn types:")
print(df.dtypes)

# ============================================
# DATA CLEANING
# ============================================
print("\n" + "="*50)
print("DATA CLEANING")
print("="*50)

# Drop columns not needed for prediction
columns_to_drop = ['customer_id', 'signup_date']
df = df.drop(columns=columns_to_drop)
print(f"Dropped columns: {columns_to_drop}")

# Standardize categorical columns (fix case inconsistencies)
# plan_type has: Basic, basic, Pro, PRO, Premium
df['plan_type'] = df['plan_type'].str.lower().str.capitalize()
print("Standardized plan_type values:", df['plan_type'].unique())

# Handle missing values - fill numerical with median
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove('churn')  # Don't include target

for col in numerical_cols:
    missing_count = df[col].isnull().sum()
    if missing_count > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Filled {missing_count} missing values in '{col}' with median: {median_val}")

# ============================================
# FEATURE ENGINEERING
# ============================================
print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Encode categorical variables
categorical_cols = ['plan_type', 'region', 'payment_method', 'discount_applied']

# Store label encoders for later use
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"Encoded '{col}': {dict(zip(le.classes_, le.transform(le.classes_)))}")

# ============================================
# MODEL TRAINING
# ============================================
print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)

# Prepare features and target
X = df.drop('churn', axis=1)
y = df['churn']

# Store feature names
feature_names = X.columns.tolist()
print(f"Features: {feature_names}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Train RandomForest model
print("\nTraining RandomForest classifier...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

# Feature importance
print("\nFeature Importance:")
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df.to_string(index=False))

# ============================================
# SAVE MODEL AND ARTIFACTS
# ============================================
print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save model and label encoders together
model_artifacts = {
    'model': model,
    'label_encoders': label_encoders,
    'feature_names': feature_names,
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols
}

with open('models/churn_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

print("Model saved to 'models/churn_model.pkl'")

# Print the expected input format for reference
print("\n" + "="*50)
print("EXPECTED INPUT FORMAT FOR PREDICTIONS")
print("="*50)
print("\nCategorical features and their valid values:")
for col in categorical_cols:
    print(f"  {col}: {list(label_encoders[col].classes_)}")

print("\nNumerical features and their ranges (from training data):")
for col in numerical_cols:
    print(f"  {col}: min={df[col].min():.2f}, max={df[col].max():.2f}")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
