#!/usr/bin/env python3
"""
Diabetes Prediction Model using Logistic Regression

This script builds a machine learning model to predict diabetes based on health metrics
including BMI, glucose level, age, and other factors.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data(filepath):
    """Load the diabetes dataset and perform basic exploration."""
    print("=" * 60)
    print("DIABETES PREDICTION MODEL")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(filepath)
    
    print(f"\nDataset Shape: {df.shape}")
    print(f"Features: {list(df.columns[:-1])}")
    print(f"Target: {df.columns[-1]}")
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nDataset Info:")
    print(df.info())
    
    print("\nBasic Statistics:")
    print(df.describe())
    
    print("\nTarget Distribution:")
    target_counts = df['Outcome'].value_counts()
    print(target_counts)
    print(f"Diabetes rate: {target_counts[1] / len(df) * 100:.1f}%")
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    return df

def data_preprocessing(df):
    """Clean and preprocess the data."""
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING")
    print("=" * 60)
    
    # Some features have 0 values that are likely missing data
    # These are medically impossible/unlikely to be 0
    zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    print("\nZero values in key features (likely missing data):")
    for feature in zero_features:
        zero_count = (df[feature] == 0).sum()
        print(f"{feature}: {zero_count} ({zero_count/len(df)*100:.1f}%)")
    
    # Replace 0s with median values for these features
    df_clean = df.copy()
    for feature in zero_features:
        median_val = df_clean[df_clean[feature] != 0][feature].median()
        df_clean[feature] = df_clean[feature].replace(0, median_val)
        print(f"Replaced 0s in {feature} with median: {median_val:.1f}")
    
    return df_clean

def exploratory_data_analysis(df):
    """Perform exploratory data analysis."""
    print("\n" + "=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Feature importance analysis
    print("Feature correlation with diabetes outcome:")
    correlations = df.corr()['Outcome'].abs().sort_values(ascending=False)
    for feature, corr in correlations.items():
        if feature != 'Outcome':
            print(f"{feature}: {corr:.3f}")
    
    # Basic statistics by outcome
    print("\nBasic statistics by diabetes outcome:")
    for outcome in [0, 1]:
        outcome_label = "No Diabetes" if outcome == 0 else "Diabetes"
        print(f"\n{outcome_label} group:")
        subset = df[df['Outcome'] == outcome]
        key_features = ['Glucose', 'BMI', 'Age', 'Pregnancies']
        for feature in key_features:
            mean_val = subset[feature].mean()
            std_val = subset[feature].std()
            print(f"  {feature}: {mean_val:.1f} ± {std_val:.1f}")
    
    return df

def train_model(df):
    """Train the logistic regression model."""
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    # Prepare features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    return model, scaler, X_test, y_test, y_pred, y_pred_proba, X.columns

def evaluate_model(y_test, y_pred, y_pred_proba, feature_names, model):
    """Evaluate the model performance."""
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"AUC Score: {auc_score:.3f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"True Negatives: {cm[0,0]}")
    print(f"False Positives: {cm[0,1]}")
    print(f"False Negatives: {cm[1,0]}")
    print(f"True Positives: {cm[1,1]}")
    
    # Calculate additional metrics
    sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])  # True Positive Rate
    specificity = cm[0,0] / (cm[0,0] + cm[0,1])  # True Negative Rate
    
    print(f"\nAdditional Metrics:")
    print(f"Sensitivity (Recall): {sensitivity:.3f} ({sensitivity*100:.1f}%)")
    print(f"Specificity: {specificity:.3f} ({specificity*100:.1f}%)")
    
    # Feature importance
    print("\nFeature Importance (Logistic Regression Coefficients):")
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0],
        'abs_coefficient': np.abs(model.coef_[0])
    }).sort_values('abs_coefficient', ascending=False)
    
    for _, row in feature_importance.iterrows():
        direction = "increases" if row['coefficient'] > 0 else "decreases"
        print(f"{row['feature']}: {row['coefficient']:.3f} ({direction} diabetes risk)")
    
    return accuracy, auc_score

def predict_diabetes(model, scaler, feature_names):
    """Function to predict diabetes for new patients."""
    print("\n" + "=" * 60)
    print("DIABETES PREDICTION FUNCTION")
    print("=" * 60)
    
    def make_prediction(pregnancies, glucose, blood_pressure, skin_thickness, 
                       insulin, bmi, diabetes_pedigree, age):
        """
        Predict diabetes probability for a new patient.
        
        Parameters:
        - pregnancies: Number of pregnancies
        - glucose: Glucose level
        - blood_pressure: Blood pressure
        - skin_thickness: Skin thickness
        - insulin: Insulin level
        - bmi: Body Mass Index
        - diabetes_pedigree: Diabetes pedigree function
        - age: Age in years
        
        Returns:
        - prediction: 0 (No Diabetes) or 1 (Diabetes)
        - probability: Probability of having diabetes (0-1)
        """
        
        # Create input array
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                               insulin, bmi, diabetes_pedigree, age]])
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0, 1]
        
        return prediction, probability
    
    # Example predictions
    print("Example Predictions:")
    print("-" * 40)
    
    # Example 1: Low risk patient
    pred1, prob1 = make_prediction(1, 85, 66, 29, 94, 26.6, 0.351, 31)
    print(f"Patient 1 (Low risk profile):")
    print(f"  Prediction: {'Diabetes' if pred1 == 1 else 'No Diabetes'}")
    print(f"  Probability: {prob1:.3f} ({prob1*100:.1f}%)")
    
    # Example 2: High risk patient
    pred2, prob2 = make_prediction(6, 148, 72, 35, 155, 33.6, 0.627, 50)
    print(f"\nPatient 2 (High risk profile):")
    print(f"  Prediction: {'Diabetes' if pred2 == 1 else 'No Diabetes'}")
    print(f"  Probability: {prob2:.3f} ({prob2*100:.1f}%)")
    
    # Example 3: Medium risk patient
    pred3, prob3 = make_prediction(3, 120, 70, 30, 135, 28.5, 0.4, 35)
    print(f"\nPatient 3 (Medium risk profile):")
    print(f"  Prediction: {'Diabetes' if pred3 == 1 else 'No Diabetes'}")
    print(f"  Probability: {prob3:.3f} ({prob3*100:.1f}%)")
    
    return make_prediction

def main():
    """Main function to run the complete diabetes prediction pipeline."""
    try:
        # Load and explore data
        df = load_and_explore_data('/Users/devrev/Documents/Rahul/tp/HW/diabetes.csv')
        
        # Preprocess data
        df_clean = data_preprocessing(df)
        
        # Exploratory data analysis
        exploratory_data_analysis(df_clean)
        
        # Train model
        model, scaler, X_test, y_test, y_pred, y_pred_proba, feature_names = train_model(df_clean)
        
        # Evaluate model
        accuracy, auc_score = evaluate_model(y_test, y_pred, y_pred_proba, feature_names, model)
        
        # Create prediction function
        prediction_function = predict_diabetes(model, scaler, feature_names)
        
        print("\n" + "=" * 60)
        print("MODEL TRAINING COMPLETED!")
        print("=" * 60)
        print(f"Final Model Performance:")
        print(f"• Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"• AUC Score: {auc_score:.3f}")
        print("Model is ready for making predictions on new diabetes cases.")
        
        return model, scaler, prediction_function
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    model, scaler, predict_function = main()
