# Disease Prediction using Machine Learning

## 📌 Project Overview

This project predicts the likelihood of heart disease based on a patient's medical information using Machine Learning classification algorithms. Multiple models are trained and evaluated, and the best-performing model is saved for making predictions on new patient data.

---

## 🎯 Objective

To build a machine learning model that predicts whether a patient is likely to have heart disease using structured medical data.

---

## 📂 Dataset

- **Dataset:** Heart Disease Dataset
- **Total Records:** 3235
- **Features:** 13
- **Target Variable:**
  - `0` → No Heart Disease
  - `1` → Heart Disease

### Features Used

- Age
- Sex
- Chest Pain Type (cp)
- Resting Blood Pressure (trestbps)
- Cholesterol (chol)
- Fasting Blood Sugar (fbs)
- Resting ECG (restecg)
- Maximum Heart Rate (thalach)
- Exercise Induced Angina (exang)
- Oldpeak
- Slope
- Number of Major Vessels (ca)
- Thal

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib

---

## 🤖 Machine Learning Algorithms

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVM)

---

## 📊 Model Performance

| Model | Accuracy |
|--------|----------|
| Logistic Regression | **95.67%** |
| Random Forest | **97.99%** |
| Support Vector Machine | **97.99%** |

### Best Model

**Random Forest Classifier**

---

## 📈 Data Analysis

The following visualizations were generated:

- Correlation Heatmap
- Heart Disease Distribution Chart

---

## 📁 Project Structure

```
CodeAlpha_DiseasePrediction/
│
├── dataset/
│   └── heart_disease_dataset.csv
│
├── model/
│   ├── best_model.pkl
│   └── scaler.pkl
│
├── results/
│   ├── correlation_heatmap.png
│   └── target_distribution.png
│
├── train.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/AnushkaNegi27/CodeAlpha_DiseasePrediction.git
```

Move into the project folder

```bash
cd CodeAlpha_DiseasePrediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train.py
```

This will:

- Load the dataset
- Perform preprocessing
- Scale features
- Train multiple ML models
- Compare model performance
- Save the best model

---

## 🔍 Make Predictions

Run:

```bash
python predict.py
```

Enter the required patient details when prompted.

### Example Output

```
========== RESULT ==========

Prediction : No Heart Disease
Risk Score : 22.00%

============================
```

---

## 📌 Results

- Dataset successfully preprocessed.
- Three classification models were trained and compared.
- Random Forest achieved the highest accuracy of **97.99%**.
- The trained model can predict heart disease risk for new patient data.

---

## 🚀 Future Improvements

- Add XGBoost classifier
- Build a Flask/Django web application
- Deploy the model using Streamlit or Render
- Add feature importance visualization

---

## 📜 License

This project was developed as part of the **CodeAlpha Machine Learning Internship**.
