from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime


app = FastAPI()


data = pd.read_csv('CurSolData.csv')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)


models = {}  # Train the ARIMA models for each currency pair
target_currencies = ['GBP to PKR', 'GBP to INR', 'GBP to NGN']

for currency in target_currencies:
    ts = data[currency]
    p, d, q = 1, 2, 1  # Example parameters, you may want to tune these for each currency
    model = ARIMA(ts, order=(p, d, q))
    models[currency] = model


class PredictionRequest(BaseModel): #input for the request
    currency: str
    target_date: str


def predict_on_date(model, ts, target_date):
    last_date = ts.index[-1]
    target_date = pd.to_datetime(target_date)
    days_to_predict = (target_date - last_date).days

    if days_to_predict < 0:
        raise ValueError("Target date must be after the last date in the dataset.")
    
    forecast, conf_int = model.predict(n_periods=days_to_predict, return_conf_int=True)
    predicted_value = forecast[-1]
    confidence_interval = conf_int[-1]
    
    return predicted_value, confidence_interval

PORT: int  = 8000


@app.on_event("startup")
def startup_event():
    print("The port used for this app is", PORT)

@app.get("/")
def root():
    return {"message": "Hello, World!"}



@app.post("/predict") #api endpoint
def get_prediction(request: PredictionRequest):
    currency = request.currency
    target_date = request.target_date

    # Validate the input currency
    if currency not in models:
        raise HTTPException(status_code=400, detail="Currency not supported.")
    
    # Get the corresponding model and time series
    model = models[currency]
    ts = data[currency]
    
    # Predict the value on the target date
    try:
        predicted_value, confidence_interval = predict_on_date(model, ts, target_date)
        return {
            "currency": currency,
            "target_date": target_date,
            "predicted_value": predicted_value,
            "confidence_interval": confidence_interval.tolist()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run the server using uvicorn
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
