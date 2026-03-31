from fastapi import FastAPI , HTTPException
from pydantic import BaseModel,Field
from typing import Annotated
import joblib
import pandas as pd

model = joblib.load("Random_forest.pkl")

class item(BaseModel):
    #age	sex	bmi	children	smoker	region	charges
    age : Annotated[int,Field(...,gt=0,lt=120,description="age of user")]
    sex : Annotated[str,Field(...,description="gender of use")]
    bmi : Annotated[float,Field(...)]
    children : Annotated[int,Field(...,description="how manu childern user have")]
    smoker : Annotated[str,Field(...,description="user is smoker or not")]
    region : Annotated[str,Field(...,description="where user lives")]


app = FastAPI()

@app.get("/")
def home():
    return {"message": "API is running 🚀"}

@app.post("/predict")
def predict(item : item):
    
    df = {
        "age":item.age,
        "sex":item.sex,
        "bmi":item.bmi,
        "children":item.children,
        "smoker":item.smoker,
        "region":item.region
    }
          
    try:
      data = pd.DataFrame([df]) 
    except Exception as e:
      raise HTTPException(
        status_code=500,
        detail=f"Error: {str(e)}"
      )

    

    prediction = float(model.predict(data)[0])
   
   # Optional interpretation
    if prediction < 10000:
        category = "Low Insurance Cost"
    elif prediction < 30000:
        category = "Medium Insurance Cost"
    else:
        category = "High Insurance Cost"

    return {
        "status": "success",
        "prediction": {
            "estimated_charges": round(prediction, 2),
            "category": category
        },
        "input_summary": df
    }

