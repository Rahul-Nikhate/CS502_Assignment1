# Diabetes Prediction System

A machine learning system that predicts diabetes risk based on health metrics using logistic regression.

## 📊 Dataset

The system uses the Pima Indian Diabetes Dataset with the following features:
- **Pregnancies**: Number of pregnancies
- **Glucose**: Glucose level (mg/dL)
- **BloodPressure**: Blood pressure (mmHg)
- **SkinThickness**: Skin thickness (mm)
- **Insulin**: Insulin level (μU/mL)
- **BMI**: Body Mass Index
- **DiabetesPedigreeFunction**: Diabetes pedigree function (0-2)
- **Age**: Age in years

## 🚀 Quick Start

### Requirements
```bash
pip3 install --user pandas numpy matplotlib seaborn scikit-learn
```

### Running the Complete Analysis
```bash
python3 diabetes_prediction.py
```

This will:
- Load and analyze the dataset
- Perform data preprocessing
- Create visualizations
- Train the logistic regression model
- Evaluate model performance
- Show example predictions

### Interactive Prediction Tool
```bash
python3 predict_diabetes.py
```

This provides an interactive interface to predict diabetes risk for individual patients.

## 📈 Model Performance

- **Accuracy**: 70.8%
- **AUC Score**: 81.3%
- **Precision**: 60% (Diabetes class)
- **Recall**: 50% (Diabetes class)

### Feature Importance
1. **Glucose** (highest impact)
2. **BMI**
3. **Pregnancies**
4. **Diabetes Pedigree Function**
5. **Age**

## 🔬 Technical Details

### Data Preprocessing
- Replaces zero values in key features with median values
- Features scaled using StandardScaler
- Stratified train-test split (80/20)

### Model
- **Algorithm**: Logistic Regression
- **Features**: 8 health metrics
- **Target**: Binary classification (Diabetes/No Diabetes)

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-score
- ROC-AUC score
- Confusion Matrix
- Feature importance analysis

## 📊 Visualizations

The system generates:
1. **Exploratory Data Analysis plots**:
   - Target distribution
   - Feature correlation heatmap
   - Age vs Glucose scatter plot
   - BMI distribution by diabetes status

2. **Model Performance plots**:
   - Confusion matrix
   - ROC curve
   - Feature importance chart

## 🎯 Example Usage

```python
from diabetes_prediction import predict_diabetes_risk

# Example patient data
result = predict_diabetes_risk(
    pregnancies=2,
    glucose=120,
    blood_pressure=70,
    skin_thickness=30,
    insulin=135,
    bmi=28.5,
    diabetes_pedigree=0.4,
    age=35
)

print(f"Prediction: {result['prediction']}")
print(f"Risk Level: {result['risk_level']}")
print(f"Probability: {result['probability_percent']}%")
```

## ⚠️ Important Notes

- This is a screening tool and should not replace professional medical advice
- Always consult healthcare professionals for medical decisions
- The model is trained on a specific population (Pima Indians) and may not generalize perfectly to all populations
- Regular model retraining with new data is recommended

## 📁 Files

- `diabetes.csv`: Original dataset
- `diabetes_prediction.py`: Complete analysis and model training
- `predict_diabetes.py`: Interactive prediction tool
- `requirements.txt`: Python dependencies
- `README.md`: This documentation

## 🏥 Risk Level Guidelines

- **Low Risk** (<30%): Continue healthy lifestyle
- **Medium Risk** (30-60%): Preventive measures recommended
- **High Risk** (>60%): Medical consultation recommended

---

*Disclaimer: This tool is for educational and screening purposes only. Always seek professional medical advice for health-related decisions.*
