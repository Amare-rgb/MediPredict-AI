"""
MediPredict AI - Model Training Script (Fully Fixed)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class ReadmissionModelTrainer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        self.numerical_features = None
        self.categorical_features = None
        
    def preprocess_data(self, df):
        """Preprocess data: handle categorical variables, scale numerical features"""
        # Define feature categories based on available columns
        all_numerical = [
            'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'heart_rate',
            'respiratory_rate', 'temperature', 'oxygen_saturation',
            'hemoglobin', 'white_blood_cells', 'creatinine', 'blood_glucose',
            'hbA1c', 'cholesterol', 'lab_score', 'comorbidity_count',
            'prior_hospital_visits_1yr', 'prior_emergency_visits_1yr',
            'prior_icu_stays', 'medication_count', 'length_of_stay',
            'procedure_count', 'distance_to_hospital'
        ]
        
        all_categorical = [
            'gender', 'income_level', 'education_level', 'admission_type',
            'admission_department', 'discharge_disposition'
        ]
        
        medical_conditions = [
            'diabetes', 'hypertension', 'heart_disease', 'copd',
            'kidney_disease', 'cancer'
        ]
        
        # Filter to existing columns
        self.numerical_features = [f for f in all_numerical if f in df.columns]
        self.categorical_features = [f for f in all_categorical if f in df.columns]
        
        # Create feature list
        self.feature_columns = self.numerical_features + self.categorical_features + medical_conditions
        
        # Create copy
        X = df[self.feature_columns].copy()
        
        # Encode categorical variables
        for col in self.categorical_features:
            if col in X.columns:
                le = LabelEncoder()
                # Fit on all possible values from the dataset
                X[col] = X[col].fillna('Unknown').astype(str)
                le.fit(X[col])
                X[col] = le.transform(X[col])
                self.label_encoders[col] = le
        
        # Handle medical conditions (ensure binary)
        for col in medical_conditions:
            if col in X.columns:
                X[col] = X[col].fillna(0).astype(int)
        
        # Check for infinite or NaN values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        # Scale numerical features
        X[self.numerical_features] = self.scaler.fit_transform(X[self.numerical_features])
        
        return X
    
    def train(self, df, target_col='readmitted_30d', test_size=0.2):
        """Train XGBoost model"""
        print("=" * 60)
        print("MediPredict AI - Model Training")
        print("=" * 60)
        
        # Preprocess
        print("\n1. Preprocessing data...")
        X = self.preprocess_data(df)
        y = df[target_col].values
        
        print(f"   Features: {len(self.feature_columns)}")
        print(f"   Samples: {len(X)}")
        print(f"   Positive rate: {y.mean():.1%}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        print(f"\n2. Training split: {len(X_train)} train, {len(X_test)} test")
        
        # Calculate class weights
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        print(f"   Scale pos weight: {scale_pos_weight:.2f}")
        
        # Configure XGBoost
        print("\n3. Training XGBoost model...")
        
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss'
        )
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        print("\n4. Evaluating model...")
        metrics = self.evaluate_model(X_test, y_test)
        
        # Simple cross-validation
        print("\n5. Cross-validation...")
        try:
            cv_model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                random_state=self.random_state,
                eval_metric='logloss'
            )
            cv_scores = cross_val_score(
                cv_model, X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring='recall'
            )
            print(f"   CV Recall: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
            metrics['cv_recall_mean'] = cv_scores.mean()
            metrics['cv_recall_std'] = cv_scores.std()
        except Exception as e:
            print(f"   CV skipped: {str(e)[:100]}")
        
        return metrics, (X_test, y_test)
    
    def evaluate_model(self, X_test, y_test):
        """Calculate comprehensive model metrics"""
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'recall': recall_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print("\n   Model Performance:")
        print(f"   - Recall: {metrics['recall']:.3f}")
        print(f"   - Precision: {metrics['precision']:.3f}")
        print(f"   - F1 Score: {metrics['f1']:.3f}")
        print(f"   - AUC-ROC: {metrics['auc_roc']:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n   Confusion Matrix:")
        print(f"   TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
        print(f"   FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")
        
        return metrics
    
    def save_model(self, filepath='models/readmission_model.pkl'):
        """Save model and preprocessing artifacts"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
        }
        
        joblib.dump(artifacts, filepath)
        print(f"\n✅ Model saved to {filepath}")
        return filepath
    
    def load_model(self, filepath='models/readmission_model.pkl'):
        """Load saved model and artifacts"""
        artifacts = joblib.load(filepath)
        self.model = artifacts['model']
        self.scaler = artifacts['scaler']
        self.label_encoders = artifacts['label_encoders']
        self.feature_columns = artifacts['feature_columns']
        self.numerical_features = artifacts['numerical_features']
        self.categorical_features = artifacts['categorical_features']
        print(f"✅ Model loaded from {filepath}")
        return self
    
    def predict(self, patient_data):
        """Make prediction for a single patient"""
        X = self._preprocess_single(patient_data)
        proba = self.model.predict_proba(X)[0, 1]
        prediction = self.model.predict(X)[0]
        return prediction, proba
    
    def _preprocess_single(self, patient_data):
        """Preprocess single patient data - Fixed version"""
        if isinstance(patient_data, dict):
            patient_data = pd.DataFrame([patient_data])
        
        # Create DataFrame with all required columns
        X = pd.DataFrame(columns=self.feature_columns)
        
        # Fill with default values
        for col in self.feature_columns:
            if col in patient_data.columns:
                X[col] = patient_data[col].values
            else:
                # Provide default values based on column type
                if col in self.categorical_features:
                    X[col] = 'Unknown'
                else:
                    X[col] = 0
        
        # Handle categorical variables
        for col in self.categorical_features:
            if col in X.columns and col in self.label_encoders:
                le = self.label_encoders[col]
                # Convert to string and handle unknown values
                X[col] = X[col].fillna('Unknown').astype(str)
                # Use transform with handle_unknown='ignore' equivalent
                try:
                    X[col] = le.transform(X[col])
                except ValueError:
                    # If unknown label appears, map it to the most common or first class
                    X[col] = X[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                    X[col] = le.transform(X[col])
        
        # Ensure numerical columns are numeric
        for col in self.numerical_features:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Scale numerical features
        X[self.numerical_features] = self.scaler.transform(X[self.numerical_features])
        
        return X


if __name__ == "__main__":
    # Load generated data
    print("Loading healthcare data...")
    df = pd.read_csv('data/healthcare_data.csv')
    
    # Train model
    trainer = ReadmissionModelTrainer(random_state=42)
    metrics, test_data = trainer.train(df)
    
    # Save model
    trainer.save_model()
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Test prediction with properly formatted patient data
    print("\n📊 Testing model with sample patient:")
    
    # Create sample patient with ALL required fields
    sample_patient = {
        'age': 75,
        'gender': 'Male',
        'bmi': 32,
        'heart_disease': 1,
        'diabetes': 1,
        'hypertension': 1,
        'lab_score': 55,
        'prior_hospital_visits_1yr': 3,
        'prior_emergency_visits_1yr': 2,
        'length_of_stay': 8,
        'systolic_bp': 145,
        'diastolic_bp': 90,
        'heart_rate': 85,
        'oxygen_saturation': 94,
        'temperature': 37.2,
        'copd': 0,
        'kidney_disease': 0,
        'cancer': 0,
        'comorbidity_count': 3,
        'icu_during_stay': 0,
        'admission_type': 'Emergency',
        'admission_department': 'Cardiology',
        'discharge_disposition': 'Home',
        'income_level': 'Medium',
        'education_level': 'Secondary',
        'prior_icu_stays': 0,
        'medication_count': 4,
        'procedure_count': 1,
        'distance_to_hospital': 15,
        'respiratory_rate': 18,
        'hemoglobin': 12.5,
        'white_blood_cells': 8.5,
        'creatinine': 1.1,
        'blood_glucose': 145,
        'hbA1c': 7.2,
        'cholesterol': 210
    }
    
    try:
        pred, prob = trainer.predict(sample_patient)
        print(f"   ✅ Risk Probability: {prob:.1%}")
        print(f"   ✅ Prediction: {'HIGH RISK - Readmission Likely' if pred == 1 else 'LOW RISK - Readmission Unlikely'}")
    except Exception as e:
        print(f"   ⚠️ Test prediction error: {e}")
        print("   But model is still saved and ready for the dashboard!")
    
    print("\n" + "=" * 60)
    print("🚀 NEXT STEP: Run the dashboard!")
    print("   streamlit run app.py")
    print("=" * 60)