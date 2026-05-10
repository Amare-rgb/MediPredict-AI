"""
Synthetic Healthcare Data Generator for MediPredict AI
Generates realistic patient data for hospital readmission prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

class HealthcareDataGenerator:
    """
    Generate synthetic healthcare data with realistic patterns
    for hospital readmission prediction
    """
    
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Medical condition probabilities
        self.condition_probs = {
            'diabetes': 0.25,
            'hypertension': 0.35,
            'heart_disease': 0.15,
            'copd': 0.10,
            'kidney_disease': 0.08,
            'cancer': 0.05
        }
        
        # Department mapping
        self.departments = ['Cardiology', 'Internal Medicine', 'Pulmonology', 
                           'Oncology', 'Neurology', 'Orthopedics', 'General Surgery']
        
        # Admission types
        self.admission_types = ['Emergency', 'Urgent', 'Elective', 'Observation']
        
        # Insurance types
        self.insurance_types = ['Private', 'Government', 'Self-pay', 'NGO']
        
    def generate_patient_demographics(self, n_samples):
        """Generate demographic features"""
        data = pd.DataFrame()
        
        # Age distribution (more elderly = higher readmission risk)
        data['age'] = np.random.beta(2, 5, n_samples) * 100
        data['age'] = data['age'].clip(18, 100).round(0).astype(int)
        
        # Gender
        data['gender'] = np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.48, 0.04])
        
        # Socioeconomic factors
        data['income_level'] = np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.35, 0.50, 0.15])
        data['education_level'] = np.random.choice(['Primary', 'Secondary', 'Tertiary'], n_samples, p=[0.30, 0.45, 0.25])
        
        # Distance to hospital (km)
        data['distance_to_hospital'] = np.random.exponential(15, n_samples).clip(1, 100).round(1)
        
        return data
    
    def generate_clinical_features(self, n_samples):
        """Generate clinical/laboratory features"""
        data = pd.DataFrame()
        
        # Vital signs
        data['bmi'] = np.random.normal(28, 5, n_samples).clip(15, 50).round(1)
        data['systolic_bp'] = np.random.normal(125, 15, n_samples).clip(90, 200).round(0).astype(int)
        data['diastolic_bp'] = np.random.normal(80, 10, n_samples).clip(60, 120).round(0).astype(int)
        data['heart_rate'] = np.random.normal(75, 12, n_samples).clip(50, 140).round(0).astype(int)
        data['respiratory_rate'] = np.random.normal(16, 3, n_samples).clip(12, 30).round(0).astype(int)
        data['temperature'] = np.random.normal(36.8, 0.5, n_samples).clip(35.5, 39.5).round(1)
        data['oxygen_saturation'] = np.random.normal(96, 3, n_samples).clip(85, 100).round(0).astype(int)
        
        # Lab results (normalized)
        data['hemoglobin'] = np.random.normal(13.5, 2, n_samples).clip(7, 18).round(1)
        data['white_blood_cells'] = np.random.exponential(7, n_samples).clip(2, 30).round(1)
        data['creatinine'] = np.random.normal(0.9, 0.4, n_samples).clip(0.4, 4.0).round(2)
        data['blood_glucose'] = np.random.normal(100, 30, n_samples).clip(60, 300).round(0).astype(int)
        data['hbA1c'] = np.random.normal(5.7, 1.2, n_samples).clip(4.5, 12).round(1)
        data['cholesterol'] = np.random.normal(190, 40, n_samples).clip(120, 350).round(0).astype(int)
        
        # Lab score (composite)
        data['lab_score'] = (
            (100 - (data['white_blood_cells'] - 7).clip(0, 30) * 2).clip(0, 100) * 0.2 +
            (100 - (data['creatinine'] - 0.9).clip(0, 3) * 20).clip(0, 100) * 0.2 +
            (100 - (data['blood_glucose'] - 100).clip(0, 200) * 0.3).clip(0, 100) * 0.2 +
            (100 - (data['hbA1c'] - 5.7).clip(0, 6) * 10).clip(0, 100) * 0.2 +
            data['hemoglobin'].clip(10, 16) * 6.25 * 0.2
        ).round(1)
        
        return data
    
    def generate_medical_history(self, n_samples):
        """Generate medical history conditions"""
        data = pd.DataFrame()
        
        # Chronic conditions
        for condition, prob in self.condition_probs.items():
            data[condition] = np.random.choice([0, 1], n_samples, p=[1-prob, prob])
        
        # Comorbidity count
        data['comorbidity_count'] = data[self.condition_probs.keys()].sum(axis=1)
        
        # Prior hospital utilization
        data['prior_hospital_visits_1yr'] = np.random.poisson(1.5, n_samples).clip(0, 15)
        data['prior_emergency_visits_1yr'] = np.random.poisson(1.2, n_samples).clip(0, 12)
        data['prior_icu_stays'] = np.random.poisson(0.3, n_samples).clip(0, 8)
        
        # Medication count
        data['medication_count'] = np.random.poisson(3, n_samples).clip(0, 15)
        
        return data
    
    def generate_admission_features(self, n_samples):
        """Generate current admission features"""
        data = pd.DataFrame()
        
        # Admission details
        data['admission_type'] = np.random.choice(self.admission_types, n_samples, 
                                                   p=[0.30, 0.25, 0.35, 0.10])
        data['admission_department'] = np.random.choice(self.departments, n_samples)
        
        # Length of stay (days)
        data['length_of_stay'] = np.random.exponential(5, n_samples).clip(1, 30).round(0).astype(int)
        
        # Number of procedures
        data['procedure_count'] = np.random.poisson(1.5, n_samples).clip(0, 10)
        
        # ICU admission during stay
        data['icu_during_stay'] = np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
        
        # Discharge disposition
        data['discharge_disposition'] = np.random.choice(
            ['Home', 'Home Health Care', 'Skilled Nursing Facility', 'Rehab', 'Against Medical Advice'],
            n_samples, p=[0.65, 0.15, 0.10, 0.07, 0.03]
        )
        
        return data
    
    def generate_readmission_target(self, data):
        """
        Generate realistic readmission target based on all features
        Using clinical logic to create realistic patterns
        """
        # Base risk calculation
        risk_score = np.zeros(len(data))
        
        # Demographic risk factors
        risk_score += ((data['age'] > 65) * 0.15)
        risk_score += ((data['age'] > 80) * 0.10)
        risk_score += (data['income_level'] == 'Low') * 0.08
        
        # Clinical risk factors
        risk_score += ((data['bmi'] > 30) * 0.08)
        risk_score += ((data['bmi'] < 18.5) * 0.05)
        risk_score += ((data['systolic_bp'] > 140) * 0.07)
        risk_score += ((data['oxygen_saturation'] < 92) * 0.12)
        risk_score += ((100 - data['lab_score']) / 100 * 0.15)
        
        # Medical history risk factors
        risk_score += (data['heart_disease'] * 0.18)
        risk_score += (data['diabetes'] * 0.12)
        risk_score += (data['kidney_disease'] * 0.15)
        risk_score += (data['copd'] * 0.12)
        risk_score += (data['cancer'] * 0.10)
        risk_score += (data['comorbidity_count'] * 0.05)
        risk_score += (np.minimum(data['prior_hospital_visits_1yr'] / 10, 0.20))
        risk_score += (np.minimum(data['prior_emergency_visits_1yr'] / 8, 0.15))
        
        # Admission risk factors
        risk_score += (data['admission_type'] == 'Emergency') * 0.10
        risk_score += (np.minimum(data['length_of_stay'] / 20, 0.15))
        risk_score += (data['icu_during_stay'] * 0.12)
        risk_score += (data['procedure_count'] * 0.02)
        risk_score += (data['discharge_disposition'] != 'Home') * 0.08
        
        # Add noise
        risk_score += np.random.normal(0, 0.05, len(data))
        
        # Normalize to probability
        prob_readmit = 1 / (1 + np.exp(-(risk_score - 0.4) * 4))
        
        # Generate target
        readmitted = (np.random.random(len(data)) < prob_readmit).astype(int)
        
        # Ensure at least 25% positive rate
        if readmitted.sum() < len(data) * 0.25:
            needed = int(len(data) * 0.25 - readmitted.sum())
            indices = np.where(readmitted == 0)[0]
            if len(indices) > needed:
                indices_to_flip = np.random.choice(indices, needed, replace=False)
                readmitted[indices_to_flip] = 1
        
        return readmitted, prob_readmit
    
    def generate_full_dataset(self, n_samples=10000):
        """Generate complete dataset with all features"""
        print(f"Generating {n_samples} synthetic patient records...")
        
        # Generate all feature groups
        demographics = self.generate_patient_demographics(n_samples)
        clinical = self.generate_clinical_features(n_samples)
        medical_history = self.generate_medical_history(n_samples)
        admission = self.generate_admission_features(n_samples)
        
        # Combine all features
        data = pd.concat([demographics, clinical, medical_history, admission], axis=1)
        
        # Generate target
        data['readmitted_30d'], data['risk_probability'] = self.generate_readmission_target(data)
        
        # Add timestamps
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=random.randint(0, 730)) for _ in range(n_samples)]
        data['admission_date'] = sorted(dates)
        
        print(f"Dataset generated: {len(data)} records, {len(data.columns)} columns")
        print(f"Readmission rate: {data['readmitted_30d'].mean():.1%}")
        
        return data
    
    def save_dataset(self, data, filepath='data/healthcare_data.csv'):
        """Save dataset to CSV"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")
        return filepath


if __name__ == "__main__":
    # Generate sample dataset
    generator = HealthcareDataGenerator(random_seed=42)
    df = generator.generate_full_dataset(n_samples=10000)
    generator.save_dataset(df)
    
    # Display basic statistics
    print("\nDataset Summary:")
    print(f"Shape: {df.shape}")
    print(f"Features: {list(df.columns)}")
    print("\nReadmission by age group:")
    df['age_group'] = pd.cut(df['age'], bins=[0, 40, 65, 80, 100], labels=['<40', '40-65', '65-80', '80+'])
    print(df.groupby('age_group')['readmitted_30d'].agg(['mean', 'count']))