"""
FastAPI Customer Churn Prediction API
Provides endpoints for single and batch predictions
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import pickle
import io
import uvicorn

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for predicting customer churn using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
print("Loading model...")
with open('models/churn_model.pkl', 'rb') as f:
    model_artifacts = pickle.load(f)

model = model_artifacts['model']
label_encoders = model_artifacts['label_encoders']
feature_names = model_artifacts['feature_names']
categorical_cols = model_artifacts['categorical_cols']
numerical_cols = model_artifacts['numerical_cols']
print("Model loaded successfully!")


# Pydantic model for single prediction input
class CustomerData(BaseModel):
    tenure_months: float
    plan_type: str
    region: str
    avg_sessions_per_week: float
    avg_session_duration_minutes: float
    feature_usage_score: float
    days_since_last_login: float
    monthly_fee: float
    payment_method: str
    failed_payments_last_3_months: float
    discount_applied: str
    support_tickets_count: float
    avg_ticket_resolution_time_hours: float
    customer_satisfaction_score: float
    marketing_email_opens_last_30_days: float


def preprocess_input(data: dict) -> pd.DataFrame:
    """Preprocess input data for prediction"""
    # Create a dataframe
    df = pd.DataFrame([data])

    # Standardize plan_type (fix case issues)
    df['plan_type'] = df['plan_type'].str.lower().str.capitalize()

    # Encode categorical variables
    for col in categorical_cols:
        if col in df.columns:
            le = label_encoders[col]
            try:
                df[col] = le.transform(df[col])
            except ValueError as e:
                valid_values = list(le.classes_)
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid value for '{col}'. Valid values are: {valid_values}"
                )

    # Ensure correct column order
    df = df[feature_names]

    return df


def preprocess_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess batch data for prediction"""
    # Create a copy
    df_processed = df.copy()

    # Standardize plan_type
    if 'plan_type' in df_processed.columns:
        df_processed['plan_type'] = df_processed['plan_type'].str.lower().str.capitalize()

    # Encode categorical variables
    for col in categorical_cols:
        if col in df_processed.columns:
            le = label_encoders[col]
            # Handle unknown categories by mapping to most common
            df_processed[col] = df_processed[col].apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df_processed[col] = le.transform(df_processed[col])

    # Select only required features
    df_processed = df_processed[feature_names]

    return df_processed


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the frontend HTML page"""
    with open('frontend/index.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/predict")
async def predict_single(customer: CustomerData):
    """Make a single prediction for one customer"""
    try:
        # Convert to dict and preprocess
        data = customer.model_dump()
        df = preprocess_input(data)

        # Make prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0]

        # Prepare response
        result = {
            "prediction": int(prediction),
            "prediction_label": "Churn" if prediction == 1 else "No Churn",
            "probability_no_churn": round(float(probability[0]), 4),
            "probability_churn": round(float(probability[1]), 4),
            "confidence": round(float(max(probability)) * 100, 2)
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-predict")
async def predict_batch(file: UploadFile = File(...)):
    """Make batch predictions from uploaded CSV file"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(
                status_code=400,
                detail="Please upload a CSV file"
            )

        # Read the uploaded file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        # Keep original data for output
        df_original = df.copy()

        # Check if required columns exist
        missing_cols = set(feature_names) - set(df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required columns: {list(missing_cols)}"
            )

        # Preprocess and predict
        df_processed = preprocess_batch(df)
        predictions = model.predict(df_processed)
        probabilities = model.predict_proba(df_processed)

        # Add predictions to original data
        df_original['churn_prediction'] = predictions
        df_original['churn_label'] = ['Churn' if p == 1 else 'No Churn' for p in predictions]
        df_original['probability_no_churn'] = probabilities[:, 0].round(4)
        df_original['probability_churn'] = probabilities[:, 1].round(4)
        df_original['confidence'] = (np.max(probabilities, axis=1) * 100).round(2)

        # Convert to CSV
        output = io.StringIO()
        df_original.to_csv(output, index=False)
        output.seek(0)

        # Return as streaming response
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=churn_predictions.csv"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def get_model_info():
    """Get information about the model and valid input values"""
    return {
        "features": feature_names,
        "categorical_features": {
            col: list(label_encoders[col].classes_)
            for col in categorical_cols
        },
        "numerical_features": numerical_cols
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
