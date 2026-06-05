from fastapi import FastAPI
from pydantic import BaseModel

from backend.predictor import predict

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0"
)

#input schema matching training features
class HeartDiseasePredict(BaseModel):
    sex:int
    age:int
    cp:int
    trestbps:float
    chol:float
    fbs:int
    restecg:int
    thalach:float
    exang:int
    oldpeak:float
    slope:int
    ca:int
    thal:int

@app.get("/health")
def health_check():
    return {"status": "ok"}

#heart disease prediction endpoint
@app.post("/predict-heart-disease")
def predict_heart_disease(input_data : HeartDiseasePredict):
    input_data=input_data.model_dump()
    result = predict(input_data=input_data)
    return {
        "prediction": result["prediction"],
        "probability": result["probability"],
         "Diagnosis": (
             "Heart Disease Detected"
             if result["prediction"]==1
             else "No Heart Disease Detected"
         )
    }

    
    