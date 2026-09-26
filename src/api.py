from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(
    title="Olist Customer Churn Prediction API",
    description="API RESTful para prever o risco de churn de clientes da Olist.",
    version="1.0.0"
)

# Carregar o modelo treinado
MODEL_PATH = "models/model.joblib"

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# Estrutura do JSON de entrada (Payload)
class CustomerData(BaseModel):
    total_spent: float
    avg_item_price: float
    avg_freight_value: float
    total_items: int
    avg_review_score: float

@app.get("/")
def read_root():
    return {"message": "API de Previsão de Churn da Olist está online!"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo não encontrado. Treine o modelo primeiro.")

    # Converter entrada para DataFrame
    input_data = pd.DataFrame([{
        'total_spent': customer.total_spent,
        'avg_item_price': customer.avg_item_price,
        'avg_freight_value': customer.avg_freight_value,
        'total_items': customer.total_items,
        'avg_review_score': customer.avg_review_score
    }])

    # Previsão e Probabilidade
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    return {
        "is_churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "status": "Risco Alto de Churn" if prediction == 1 else "Cliente Ativo"
    }