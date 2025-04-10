import sklearn
print(f"[DEBUG] Version de scikit-learn dans Docker : {sklearn.__version__}")

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("classifier.pkl")
scaler = joblib.load("scaler.pkl")

class PatientData(BaseModel):
    age: int
    motif_asthenie: int
    motif_alt_fonction: int
    motif_hta: int
    motif_oedeme: int
    motif_diabete: int
    pm_hta: int
    pm_diabete1: int
    pm_diabete2: int
    pm_cardiovasculaire: int
    symptome_anemie: int
    symptome_hta: int
    symptome_asthenie: int
    symptome_insomnie: int
    symptome_perte_poids: int
    etat_general: int
    uree: float
    creatinine: float
    anemie: int

@app.post("/predict")
def predict(data: PatientData):
    values = list(data.dict().values())
    X_scaled = scaler.transform([values])
    prediction = model.predict(X_scaled)[0]
    stade_dict = {
        1: "CKD 1", 2: "CKD 2", 3: "CKD 3a",
        4: "CKD 3b", 5: "CKD 4", 6: "CKD 5"
    }
    return {"result": f"Le patient est au stade de l'IRC {stade_dict[prediction]}"}  
