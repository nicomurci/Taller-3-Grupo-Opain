from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import pandas as pd
import joblib
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin
from custom_transformers import StringToNumericTransformer, ConvertToStringTransformer
import shap
import numpy as np

app = FastAPI()

# Cargar los modelos
model_v1 = joblib.load("C:/Users/Sebastián/OneDrive/Desktop/churn-v1.0.joblib")  # Modelo Baseline
model_v2 = joblib.load("C:/Users/Sebastián/OneDrive/Desktop/churn-v2.0.joblib")  # Mejor modelo de GridSearch

explainer_v1 = joblib.load("C:/Users/Sebastián/OneDrive/Desktop/explainer-v1.0.joblib")  # Modelo Baseline
explainer_v2 = joblib.load("C:/Users/Sebastián/OneDrive/Desktop/explainer-v2.0.joblib")  # Mejor modelo de GridSearch

prep=joblib.load("C:/Users/Sebastián/OneDrive/Desktop/prep.joblib")

# Clase para entrada de datos
class DataInput(BaseModel):
    customerID:str
    gender: str
    SeniorCitizen: bool
    Partner: str
    Dependents: str
    tenure: float
    PhoneService: str
    MultipleLines : str
    InternetService: str
    OnlineSecurity : str
    OnlineBackup :str 
    DeviceProtection:str 
    TechSupport:str 
    StreamingTV:str 
    StreamingMovies:str 
    Contract:str
    PaperlessBilling:object 
    PaymentMethod:str 
    MonthlyCharges:float
    TotalCharges:float 
    Churn: str

    @validator('TotalCharges', pre=True, always=True)
    def parse_total_charges(cls, v):
        if v == "":
            return 0.0  
        return v

@app.post("/{model_version}/predict")
async def predict(model_version: str, data: List[DataInput]):
    # Convertir la lista de objetos DataInput en un DataFrame
    input_data = pd.DataFrame([item.dict() for item in data])

    # Seleccionar el modelo en función de la versión
    if model_version == 'v1':
        model = model_v1
    elif model_version == 'v2':
        model = model_v2
    else:
        raise HTTPException(status_code=404, detail="Modelo no disponible")

    # Realizar predicciones
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    return {
        "predicciones": predictions.tolist(),
        "probabilidades": probabilities.tolist()
    }


@app.post("/{model_version}/explain")
async def explain(model_version: str, data: List[DataInput]):
    # Convertir la lista de objetos DataInput en un DataFrame
    input_data = pd.DataFrame([item.dict() for item in data])

     # Seleccionar el modelo en función de la versión
    if model_version == 'v1':
        model = model_v1
        explainer = explainer_v1
    elif model_version == 'v2':
        model = model_v2
        explainer = explainer_v2
    else:
        raise HTTPException(status_code=404, detail="Modelo no disponible")

    # Transformar la data de input
    transformed_input_data = prep.transform(input_data)

    # Obtener Predicciones
    predi = model.predict(input_data).tolist()

    # Obtener nombres de features
    feature_names = prep[2:].get_feature_names_out()
    
    # Calcular SHAP Values
    shap_values = explainer.shap_values(transformed_input_data)

    explanations = []
    for i in range(transformed_input_data.shape[0]):
        var = abs(shap_values[predi[i]][i])
        important_feature_indices = np.argsort(var)[-3:]
        important_feature_names = [feature_names[idx] for idx in important_feature_indices]
        

        # Create explanation with original feature values
        explanation = {
            "prediction": "El cliente pertenece a la clase: " + str(predi[i]),
            "important_features": important_feature_names,
            "shap_values": shap_values[predi[i]][i][important_feature_indices].tolist()
        }
        explanations.append(explanation)

    return {"explanations": explanations}
