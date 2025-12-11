import joblib

model = joblib.load('logistic_regression_model.pkl')
print(model.predict([[2.1, 3.2, 4.5, 6.1]]))  # Example input for prediction