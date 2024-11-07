from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import joblib
import numpy as np
from pydantic import BaseModel
import logging
import traceback

# Initialize the FastAPI application
app = FastAPI(
    title="Battery Aging Prediction API",
    description="An API to predict the aging stage of lithium-ion batteries using a trained Random Forest model",
    version="1.0.0"
)

# Load the saved model and scaler
try:
    model = joblib.load('battery_aging_model.pkl')
    scaler = joblib.load('scaler.pkl')
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model or scaler: {str(e)}")
    raise RuntimeError("Failed to load the model and scaler.")

# Define the input data format
class BatteryData(BaseModel):
    voltage: float
    capacity: float

# Root endpoint to check the API status
@app.get("/")
async def root():
    return {"message": "Battery Aging Prediction API is up and running."}

# Endpoint to predict the battery aging stage
@app.post('/predict', summary="Predict Battery Aging Stage")
async def predict_aging_stage(data: BatteryData):
    try:
        # Extract features from request
        features = np.array([[data.voltage, data.capacity]])

        # Scale features
        scaled_features = scaler.transform(features)

        # Predict using the trained model
        prediction = model.predict(scaled_features)

        # Map encoded prediction back to class label
        label_mapping = {0: 'Aged', 1: 'Healthy', 2: 'Moderate Aging'}
        result = label_mapping[prediction[0]]

        # Logging the prediction request and response
        logging.info(f"Prediction for input {data.dict()}: {result}")

        return {"prediction": result}

    except Exception as e:
        # Logging the exception details
        logging.error(f"Error while predicting: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

# Endpoint to provide information about the model
@app.get('/model-info', summary="Get Model Information")
async def get_model_info():
    """
    Provides information about the model, such as:
    - Algorithm used
    - Number of features
    - Expected input features
    """
    try:
        model_info = {
            "model_name": "Random Forest Classifier",
            "n_estimators": model.n_estimators,
            "features_used": ["Voltage [V]", "Capacity [Ah]"],
            "description": "A Random Forest Classifier trained on battery aging data to predict the aging stage."
        }
        return model_info

    except Exception as e:
        logging.error(f"Error fetching model info: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred while fetching model information.")

# Endpoint to check input scaling (for debugging)
@app.post('/scale-features', summary="Scale Input Features")
async def scale_input_features(data: BatteryData):
    """
    Accepts input data and returns the scaled values.
    This endpoint is intended for debugging purposes, to see the transformed feature values.
    """
    try:
        # Extract features from request
        features = np.array([[data.voltage, data.capacity]])
        
        # Scale features
        scaled_features = scaler.transform(features)

        # Logging the scaled features
        logging.info(f"Scaled features for input {data.dict()}: {scaled_features.tolist()}")

        return {"scaled_features": scaled_features.tolist()}

    except Exception as e:
        # Logging the exception details
        logging.error(f"Error while scaling features: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="An error occurred during feature scaling.")

# Error handling for 404 errors (not found)
@app.exception_handler(404)
async def custom_404_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "The resource you are looking for is not available. Please check the endpoint."}
    )