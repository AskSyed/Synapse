from fastapi import FastAPI
import joblib
model = joblib.load('logistic_regression_model.pkl')

app = FastAPI() 

@app.get("/")
def predict_iris(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return {"predicted_class": int(prediction[0])}