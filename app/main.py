from fastapi import FastAPI
import joblib
import pandas as pd

# Create the FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load("src/model/model.pkl")


# Define the prediction endpoint
@app.post("/predict")
def predict(data: dict):
    """
    This endpoint takes a dictionary of features as input and returns a prediction.
    """
    # Convert the input data to a pandas DataFrame
    df = pd.DataFrame([data])

    # Make a prediction
    prediction = model.predict(df)

    # Return the prediction
    return {"prediction": prediction.tolist()}
