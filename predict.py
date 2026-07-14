import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model/best_model.pkl")
scaler = joblib.load("model/scaler.pkl")

print("Enter the following details:")

age = float(input("Age: "))
sex = float(input("Sex (0=Female, 1=Male): "))
cp = float(input("Chest Pain Type (0-3): "))
trestbps = float(input("Resting Blood Pressure: "))
chol = float(input("Cholesterol: "))
fbs = float(input("Fasting Blood Sugar (0/1): "))
restecg = float(input("Rest ECG (0-2): "))
thalach = float(input("Maximum Heart Rate: "))
exang = float(input("Exercise Induced Angina (0/1): "))
oldpeak = float(input("Oldpeak: "))
slope = float(input("Slope (0-2): "))
ca = float(input("Number of Major Vessels (0-4): "))
thal = float(input("Thal (1-3): "))

import pandas as pd

columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs,
                      restecg, thalach, exang, oldpeak,
                      slope, ca, thal]], columns=columns)

data = scaler.transform(data)

prediction = model.predict(data)

probability = model.predict_proba(data)[0][1] * 100

print("\n========== RESULT ==========")

if prediction[0] == 1:
    print("Prediction : Heart Disease Detected")
else:
    print("Prediction : No Heart Disease")

print(f"Risk Score : {probability:.2f}%")
print("============================")