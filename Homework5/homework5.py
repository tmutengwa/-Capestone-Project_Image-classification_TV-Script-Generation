from fastapi import FastAPI
from pydantic import BaseModel
import pickle

# Load the model and vectorizer
with open("pipeline_v1.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)

# Define the input data model
class Client(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

# Create the FastAPI app
app = FastAPI()

@app.post("/predict")
def predict(client: Client):
    record = client.model_dump()
    X = dv.transform([record])
    y_pred = model.predict_proba(X)[:, 1]
    return {"prediction_probability": y_pred[0]}