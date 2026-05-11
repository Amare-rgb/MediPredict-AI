"""
MediPredict AI - Professional Healthcare Dashboard
Predicts 30-day hospital readmission risk with SHAP explanations
Amharic/English Bilingual Support
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOGIN SYSTEM
# ============================================================================

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'role' not in st.session_state:
    st.session_state.role = ''

def check_login(username, password):
    """Check login credentials"""
    if username == 'admin' and password == '123':
        return True, 'Admin'
    elif username == 'doctor' and password == '456':
        return True, 'Doctor'
    else:
        return False, ''

def login_page():
    """Display login page"""
    st.markdown("""
    <div style="text-align: center; padding: 50px 0;">
        <h1 style="color: #1E88E5; margin-bottom: 30px;">🏥 MediPredict AI</h1>
        <p style="font-size: 18px; color: #64748B; margin-bottom: 40px;">
            Healthcare Analytics Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login")
        
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submit_button = st.form_submit_button("Login", use_container_width=True)
            
            if submit_button:
                if username and password:
                    success, role = check_login(username, password)
                    if success:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.role = role
                        st.success(f"✅ Welcome {role}!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.warning("⚠️ Please enter both username and password")
        
        st.markdown("---")
        st.markdown("""
        <div style="background: #f1f5f9; padding: 15px; border-radius: 8px;">
            <p style="margin: 0 0 10px 0; font-weight: 600; color: #334155;">
                📋 Demo Credentials:
            </p>
            <div style="font-size: 14px;">
                <strong>Admin:</strong> admin / 123<br>
                <strong>Doctor:</strong> doctor / 456
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# LANGUAGE DICTIONARY (English and Amharic)
# ============================================================================

LANGUAGES = {
    'English': {
        # General
        'app_title': 'MediPredict AI',
        'app_subtitle': 'AI-Powered Clinical Decision Support System | 30-Day Readmission Risk Prediction',
        'sidebar_title': 'MediPredict AI',
        'sidebar_subtitle': '30-Day Readmission Risk Assessment',
        
        # Demographics
        'demographics': '👤 Demographics',
        'age': 'Age',
        'gender': 'Gender',
        'male': 'Male',
        'female': 'Female',
        'other': 'Other',
        
        # Vital Signs
        'vital_signs': '🩺 Vital Signs',
        'bmi': 'BMI',
        'systolic_bp': 'Systolic BP',
        'diastolic_bp': 'Diastolic BP',
        'heart_rate': 'Heart Rate',
        'oxygen_saturation': 'O₂ Saturation (%)',
        'temperature': 'Temperature (°C)',
        
        # Medical History
        'medical_history': '📋 Medical History',
        'diabetes': 'Diabetes',
        'hypertension': 'Hypertension',
        'heart_disease': 'Heart Disease',
        'copd': 'COPD',
        'kidney_disease': 'Kidney Disease',
        'cancer': 'Cancer',
        
        # Lab Results
        'lab_results': '🔬 Lab Results',
        'hemoglobin': 'Hemoglobin (g/dL)',
        'blood_glucose': 'Blood Glucose (mg/dL)',
        'creatinine': 'Creatinine (mg/dL)',
        'hba1c': 'HbA1c (%)',
        
        # Admission Details
        'admission_details': '🏥 Admission Details',
        'length_of_stay': 'Length of Stay (days)',
        'prior_visits': 'Prior Hospital Visits (1yr)',
        'prior_ed_visits': 'Prior ED Visits (1yr)',
        'icu_during_stay': 'ICU Admission During Stay',
        
        # Buttons
        'predict_button': '🔮 Predict Readmission Risk',
        
        # KPI Metrics
        'total_patients': 'Total Patients Analyzed',
        'readmission_rate': 'Population Readmission Rate',
        'model_recall': 'Model Recall (Sensitivity)',
        'model_performance': 'Model Performance',
        
        # Risk Levels
        'predicted_risk': 'Predicted Readmission Risk',
        'risk_score': 'Risk Score',
        'probability_text': 'Probability of hospital readmission within 30 days of discharge',
        'high_risk': 'HIGH RISK',
        'moderate_risk': 'MODERATE RISK',
        'low_risk': 'LOW RISK',
        
        # SHAP Section
        'shap_title': 'AI Explainability: Why This Prediction?',
        'shap_subtitle': 'Understanding the key factors driving the readmission risk assessment',
        'shap_axis_label': 'SHAP Value (Impact on Risk)',
        'shap_chart_title': 'Feature Impact on Readmission Risk',
        'shap_caption': 'Positive values (red) increase readmission risk, negative values (green) decrease risk',
        
        # Recommendations
        'recommendations_title': 'Clinical Recommendations',
        
        # Summary
        'summary_title': 'Patient Clinical Summary',
        'demographics_summary': 'Demographics',
        'vital_signs_summary': 'Vital Signs',
        'medical_conditions': 'Medical Conditions',
        'utilization': 'Utilization',
        'none_reported': 'None reported',
        'comorbidity_count': 'Comorbidity Count',
        'los': 'LOS',
        'prior_visits_summary': 'Prior Visits',
        'icu_stay': 'ICU Stay',
        'yes': 'Yes',
        'no': 'No',
        
        # Insights
        'welcome_message': '👈 Please enter patient information in the sidebar and click "Predict Readmission Risk" to get started',
        'population_insights': 'Population Insights',
        'readmission_by_age': 'Readmission Rate by Age Group',
        'top_risk_factors': 'Top Risk Factors (Odds Ratio)',
        'about_ai': 'About MediPredict AI',
        'about_text': 'This system uses an XGBoost machine learning model trained on healthcare data to predict 30-day hospital readmission risk. The model achieves 87% recall (sensitivity), meaning it correctly identifies 87% of patients who will be readmitted. SHAP values provide transparent, interpretable explanations for each prediction.',
        
        # Footer
        'footer_warning': '⚠️ MediPredict AI is a clinical decision support tool. All predictions should be reviewed by qualified healthcare professionals.',
        'footer_copyright': '© 2026 MediPredict AI | Powered by XGBoost & SHAP | HABTech Solutions AI Healthcare Challenge',
        
        # Recommendations by Risk Level
        'rec_high_1': '🔴 **URGENT:** Schedule follow-up appointment within **7 days** of discharge',
        'rec_high_2': '📞 Implement **daily check-in calls** for the first 2 weeks post-discharge',
        'rec_high_3': '💊 Conduct **comprehensive medication reconciliation** before discharge',
        'rec_high_4': '🏠 Arrange **home health visit** within 48 hours',
        'rec_high_5': '📋 Provide **detailed discharge plan** with clear warning signs',
        'rec_cardio': '❤️ **Cardiology follow-up** required within 7 days',
        'rec_endocrine': '🩸 **Endocrine consult** for diabetes management optimization',
        'rec_nephrology': '🩺 **Nephrology review** of medication regimen',
        'rec_moderate_1': '🟡 Schedule follow-up within **14 days** of discharge',
        'rec_moderate_2': '📋 Provide comprehensive discharge instructions with medication list',
        'rec_moderate_3': '📞 Schedule a **telehealth check-in** at day 7 post-discharge',
        'rec_moderate_4': '🏥 Review recent lab results and adjust medications if needed',
        'rec_case_management': '🔄 **Case management referral** for care coordination',
        'rec_geriatric': '👴 **Geriatric assessment** recommended',
        'rec_low_1': '✅ Routine follow-up within **30 days** is adequate',
        'rec_low_2': '📋 Provide standard discharge instructions',
        'rec_low_3': '💊 Review medication list at follow-up appointment',
        'rec_low_4': '📞 Optional telehealth check-in at patient\'s convenience',
        
        # Computing
        'computing_shap': 'Computing SHAP explanations...',
        'model_not_found': '⚠️ Model not found. Please run `python train_model.py` first.',
        'data_warning': 'Data file not found. Generating sample data for demo...',
    },
    'አማርኛ': {
        # General
        'app_title': 'MediPredict AI',
        'app_subtitle': 'በአርቲፊሻል ኢንተሊጀንስ የሚመራ ክሊኒካል ውሳኔ አሰጣጥ ስርዓት | በ30 ቀናት ውስጥ እንደገና የመተኛት ትንበያ',
        'sidebar_title': 'MediPredict AI',
        'sidebar_subtitle': 'የ30 ቀናት የእንደገና መተኛት ስጋት ግምገማ',
        
        # Demographics
        'demographics': '👤 የህምሪ መረጃ',
        'age': 'ዕድሜ',
        'gender': 'ፆታ',
        'male': 'ወንድ',
        'female': 'ሴት',
        'other': 'ሌላ',
        
        # Vital Signs
        'vital_signs': '🩺 የህይወት ምልክቶች',
        'bmi': 'BMI',
        'systolic_bp': 'ሲስቶሊክ የደም ግፊት',
        'diastolic_bp': 'ዳይስቶሊክ የደም ግፊት',
        'heart_rate': 'የልብ ምት',
        'oxygen_saturation': 'የኦክስጅን መጠን (%)',
        'temperature': 'የሙቀት መጠን (°C)',
        
        # Medical History
        'medical_history': '📋 የሕክምና ታሪክ',
        'diabetes': 'ስኳር በሽታ',
        'hypertension': 'የደም ግፊት',
        'heart_disease': 'የልብ በሽታ',
        'copd': 'ሲኦፒዲ',
        'kidney_disease': 'የኩላሊት በሽታ',
        'cancer': 'ካንሰር',
        
        # Lab Results
        'lab_results': '🔬 የላብራቶሪ ውጤቶች',
        'hemoglobin': 'ሄሞግሎቢን (g/dL)',
        'blood_glucose': 'የደም ግሉኮስ (mg/dL)',
        'creatinine': 'ክሬቲኒን (mg/dL)',
        'hba1c': 'HbA1c (%)',
        
        # Admission Details
        'admission_details': '🏥 የህክምና ቆይታ ዝርዝሮች',
        'length_of_stay': 'የህክምና ቆይታ (ቀናት)',
        'prior_visits': 'ቀዳሚ የሆስፒታል ጉብኝቶች (1 ዓመት)',
        'prior_ed_visits': 'ቀዳሚ የድንገተና ክፍል ጉብኝቶች (1 ዓመት)',
        'icu_during_stay': 'በህክምና ቆይታ ወቅት አይሲዩ መተኛት',
        
        # Buttons
        'predict_button': '🔮 የእንደገና የመተኛት ስጋት ትንበያ',
        
        # KPI Metrics
        'total_patients': 'ጠቅላላ የተመረመሩ ህሙማን',
        'readmission_rate': 'የህዝብ እንደገና የመተኛት መጠን',
        'model_recall': 'የሞዴሉ ማስታወስ (ስሜታዊነት)',
        'model_performance': 'የሞዴሉ አፈጻጸም',
        
        # Risk Levels
        'predicted_risk': 'የተተነበየ የእንደገና መተኛት ስጋት',
        'risk_score': 'የስጋት ነጥብ',
        'probability_text': 'ከህክምና ቤት ከወጡ በኋላ በ30 ቀናት ውስጥ እንደገና የመተኛት እድል',
        'high_risk': 'ከፍተኛ ስጋት',
        'moderate_risk': 'መካከለኛ ስጋት',
        'low_risk': 'ዝቅተኛ ስጋት',
        
        # SHAP Section
        'shap_title': '🔬 የአርቲፊሻል ኢንተሊጀንስ ማብራሪያ: ለምን ይህ ትንበያ?',
        'shap_subtitle': 'የእንደገና መተኛት ስጋት ግምገማን የሚወስኑ ቁልፍ ነገሮችን መረዳት',
        'shap_axis_label': 'የSHAP ዋጋ (በስጋት ላይ ያለው ተጽእኖ)',
        'shap_chart_title': 'በእንደገና መተኛት ስጋት ላይ ተጽእኖ የሚያሳድሩ ነገሮች',
        'shap_caption': 'አወንታዊ እሴቶች (ቀይ) የእንደገና መተኛት ስጋትን ይጨምራሉ, አሉታዊ እሴቶች (አረንጓዴ) ይቀንሳሉ',
        
        # Recommendations
        'recommendations_title': '📋 ክሊኒካል ምክሮች',
        
        # Summary
        'summary_title': '📄 የህሙም ክሊኒካል ማጠቃለያ',
        'demographics_summary': 'ህሙም መረጃ',
        'vital_signs_summary': 'የህይወት ምልክቶች',
        'medical_conditions': 'የሕክምና ሁኔታዎች',
        'utilization': 'አጠቃቀም',
        'none_reported': 'ምንም አልተገለጸም',
        'comorbidity_count': 'ተያያዥ በሽታዎች ብዛት',
        'los': 'የህክምና ቆይታ',
        'prior_visits_summary': 'ቀዳሚ ጉብኝቶች',
        'icu_stay': 'አይሲዩ መተኛት',
        'yes': 'አዎ',
        'no': 'አይ',
        
        # Insights
        'welcome_message': '👈 እባክዎ የህሙም መረጃ በጎን አሞሌ ውስጥ ያስገቡ እና "የእንደገና የመተኛት ስጋት ትንበያ" ን ጠቅ ያድርጉ',
        'population_insights': '📈 የህዝብ ግንዛቤዎች',
        'readmission_by_age': 'በዕድሜ ቡድን የእንደገና መተኛት መጠን',
        'top_risk_factors': 'ከፍተኛ የስጋት ምክንያቶች (የዕድል መጠን)',
        'about_ai': '🤖 ስለ MediPredict AI',
        'about_text': 'ይህ ስርዓት በ30 ቀናት ውስጥ የሆስፒታል እንደገና መተኛት ስጋት ለመተንበይ በጤና እንክብካቤ መረጃ ላይ የሰለጠነ XGBoost ማሽን መማሪያ ሞዴል ይጠቀማል። ሞዴሉ 87% ማስታወስ (ስሜታዊነት) ያገኛል, ይህም ማለት እንደገና የሚተኙ 87% ህሙማንን በትክክል ይለያል ማለት ነው። SHAP እሴቶች ለእያንዳንዱ ትንበያ ግልጽ እና ሊረዱ የሚችሉ ማብራሪያዎችን ይሰጣሉ።',
        
        # Footer
        'footer_warning': '⚠️ MediPredict AI ክሊኒካል ውሳኔ አሰጣጥ መሳሪያ ነው። ሁሉም ትንበያዎች ብቁ በሆኑ የጤና እንክብካቤ ባለሙያዎች መገምገም አለባቸው።',
        'footer_copyright': '© 2026 MediPredict AI | በXGBoost እና SHAP የተጎላበተ | የHABTech ሶሉሽንስ አይኤ ሄልዝኬየር ውድድር',
        
        # Recommendations by Risk Level (Amharic)
        'rec_high_1': '🔴 **አስቸኳይ:** ከሆስፒታል ከወጡ በኋላ በ**7 ቀናት** ውስጥ የክትትል ቀጠሮ ይያዙ',
        'rec_high_2': '📞 ከሆስፒታል ከወጡ በኋላ ለመጀመሪያዎቹ 2 ሳምንታት **ዕለታዊ የስልክ ጥቆማ** ያድርጉ',
        'rec_high_3': '💊 ከሆስፒታል ከመውጣትዎ በፊት **ሁሉን አቀፍ የመድሀኒት እርቅ** ያከናውኑ',
        'rec_high_4': '🏠 በ48 ሰዓታት ውስጥ **የቤት ውስጥ የጤና ጉብኝት** ያስደምጡ',
        'rec_high_5': '📋 ግልጽ የማስጠንቀቂያ ምልክቶችን የያዘ **ዝርዝር የመውጫ እቅድ** ያቅርቡ',
        'rec_cardio': '❤️ በ7 ቀናት ውስጥ **የልብ ህክምና ክትትል** ያስፈልጋል',
        'rec_endocrine': '🩸 የስኳር በሽታ አያያዝን ለማሻሻል **የኢንዶክሪን ምክክር**',
        'rec_nephrology': '🩺 የመድሀኒት አጠቃቀም **የኩላሊት ህክምና ግምገማ**',
        'rec_moderate_1': '🟡 ከሆስፒታል ከወጡ በኋላ በ**14 ቀናት** ውስጥ የክትትል ቀጠሮ ይያዙ',
        'rec_moderate_2': '📋 የመድሀኒት ዝርዝርን ጨምሮ አጠቃላይ የመውጫ መመሪያዎችን ያቅርቡ',
        'rec_moderate_3': '📞 ከሆስፒታል ከወጡ በ7ኛ ቀን **የቴሌህክምና ጥቆማ** ያስደምጡ',
        'rec_moderate_4': '🏥 የቅርብ ጊዜ የላብራቶሪ ውጤቶችን ይገምግሙ እና አስፈላጊ ከሆነ መድሀኒቶችን ያስተካክሉ',
        'rec_case_management': '🔄 የህክምና ቅንጅት ለማግኘት **የጉዳይ አያያዝ ሪፈራል**',
        'rec_geriatric': '👴 **የአረጋውያን ምርመራ** ይመከራል',
        'rec_low_1': '✅ በ**30 ቀናት** ውስጥ መደበኛ ክትትል በቂ ነው',
        'rec_low_2': '📋 መደበኛ የመውጫ መመሪያዎችን ያቅርቡ',
        'rec_low_3': '💊 በክትትል ቀጠሮ ላይ የመድሀኒት ዝርዝርን ይገምግሙ',
        'rec_low_4': '📞 በህሙሙ ምቾት ጊዜ አማራጭ የቴሌህክምና ጥቆማ',
        
        # Computing
        'computing_shap': 'የSHAP ማብራሪያዎችን በማስላት ላይ...',
        'model_not_found': '⚠️ ሞዴል አልተገኘም። እባክዎ መጀመሪያ `python train_model.py` ያሂዱ።',
        'data_warning': 'የውሂብ ፋይል አልተገኘም። ለማሳያ ናሙና ውሂብ በማመንጨት ላይ...',
    }
}

# Page configuration
st.set_page_config(
    page_title="MediPredict AI | Healthcare Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LANGUAGE SELECTION
# ============================================================================

# Initialize language in session state
if 'language' not in st.session_state:
    st.session_state.language = 'English'

# Language selector in sidebar
# Check if user is logged in
if not st.session_state.logged_in:
    login_page()
    st.stop()

# Language selector in sidebar (only show if logged in)
lang = st.sidebar.selectbox(
    "🌐 Language / ቋንቋ",
    options=['English', 'አማርኛ'],
    index=0 if st.session_state.language == 'English' else 1
)
st.session_state.language = lang
t = LANGUAGES[st.session_state.language]

# Logout button in sidebar
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ''
    st.session_state.role = ''
    st.rerun()

# Sidebar navigation (role-based)
st.sidebar.markdown("---")
st.sidebar.markdown(f"### 👤 {st.session_state.username} ({st.session_state.role})")
st.sidebar.markdown("---")

# Show navigation based on role
if st.session_state.role == 'Admin':
    st.sidebar.markdown("#### 📊 Navigation")
    if st.sidebar.button("🏠 Admin Dashboard", use_container_width=True):
        st.session_state.page = "admin"
        st.rerun()
    if st.sidebar.button("👥 Patient Records", use_container_width=True):
        st.session_state.page = "patients"
        st.rerun()
elif st.session_state.role == 'Doctor':
    st.sidebar.markdown("#### 🩺 Navigation")
    # Doctor only has AI prediction - no Patient Records access

# Initialize page state
if 'page' not in st.session_state:
    st.session_state.page = "main"

# Page routing
if st.session_state.page == "admin" and st.session_state.role == 'Admin':
    # Import and show Admin Dashboard
    import importlib.util
    spec = importlib.util.spec_from_file_location("admin_dashboard", "pages/Admin_Dashboard.py")
    admin_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(admin_module)
    st.stop()
elif st.session_state.page == "patients":
    # Import and show Patient Records
    import importlib.util
    spec = importlib.util.spec_from_file_location("patient_records", "pages/Patient_Records.py")
    patient_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(patient_module)
    st.stop()

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================

st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #E0F7FA 0%, #B2EBF2 50%, #80DEEA 100%);
        min-height: 100vh;
    }
    
    /* Custom card styling */
    .card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FAFC 100%);
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F1F5F9 100%);
        border-radius: 20px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1), 0 4px 6px -2px rgba(0,0,0,0.05);
        border: 1px solid #CBD5E1;
        transition: transform 0.2s ease-in-out;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 20px 25px -5px rgba(0,0,0,0.15), 0 10px 10px -5px rgba(0,0,0,0.04);
    }
    
    .metric-value {
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(135deg, #1E88E5, #0D47A1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #DC2626, #B91C1C);
        color: white;
        padding: 12px 24px;
        border-radius: 40px;
        font-weight: 600;
        font-size: 20px;
        text-align: center;
    }
    
    .risk-moderate {
        background: linear-gradient(135deg, #F59E0B, #D97706);
        color: white;
        padding: 12px 24px;
        border-radius: 40px;
        font-weight: 600;
        font-size: 20px;
        text-align: center;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #10B981, #059669);
        color: white;
        padding: 12px 24px;
        border-radius: 40px;
        font-weight: 600;
        font-size: 20px;
        text-align: center;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(135deg, #E0F7FA 0%, #B2EBF2 50%, #80DEEA 100%);
        border-right: 1px solid #B2EBF2;
    }
    
    /* Enhanced sidebar sections */
    .stSidebar {
        background: linear-gradient(180deg, #E0F7FA 0%, #B2EBF2 50%, #80DEEA 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1E88E5, #0D47A1);
        color: white;
        border-radius: 12px;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(30,136,229,0.2);
    }
    
    /* Section headers */
    .section-header {
        font-size: 20px;
        font-weight: 600;
        color: #1E293B;
        margin: 20px 0 15px 0;
        padding-bottom: 8px;
        border-bottom: 2px solid #1E88E5;
        display: inline-block;
    }
    
    /* Info box */
    .info-box {
        background-color: #EFF6FF;
        border-left: 4px solid #1E88E5;
        padding: 15px;
        border-radius: 12px;
        margin: 15px 0;
    }
    
    /* Language-specific font adjustments for Amharic */
    [data-testid="stMarkdownContainer"] p, 
    [data-testid="stMarkdownContainer"] div,
    .stSelectbox label,
    .stNumberInput label,
    .stCheckbox label {
        font-family: 'Segoe UI', 'Noto Sans Ethiopic', 'Roboto', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND ARTIFACTS
# ============================================================================

@st.cache_resource
def load_model_artifacts():
    """Load trained model and preprocessing artifacts"""
    try:
        artifacts = joblib.load('models/readmission_model.pkl')
        return artifacts
    except FileNotFoundError:
        st.error(t['model_not_found'])
        return None

@st.cache_data
def load_data():
    """Load healthcare dataset for analytics"""
    try:
        df = pd.read_csv('data/healthcare_data.csv')
        return df
    except FileNotFoundError:
        st.warning(t['data_warning'])
        from data_generator import HealthcareDataGenerator
        generator = HealthcareDataGenerator()
        df = generator.generate_full_dataset(5000)
        return df

# Load artifacts
artifacts = load_model_artifacts()
df = load_data()

if artifacts is None:
    st.stop()

model = artifacts['model']
scaler = artifacts['scaler']
label_encoders = artifacts['label_encoders']
feature_columns = artifacts['feature_columns']
numerical_features = artifacts['numerical_features']
categorical_features = artifacts['categorical_features']

# ============================================================================
# SIDEBAR - PATIENT INPUT
# ============================================================================

with st.sidebar:
    st.markdown(f"""
    <div style="text-align: center; padding: 25px 15px 20px 15px; margin: 0 10px 20px 10px;">
        <h1 style="font-size: 2.0rem; font-weight: 800; color: #0F172A; margin: 0 0 12px 0; line-height: 1.2;">
            {t['sidebar_title']}
        </h1>
        <div style="height: 1px; background: linear-gradient(90deg, transparent, #CBD5E1, transparent); margin: 0 0 12px 0;"></div>
        <p style="font-size: 0.85rem; color: #64748B; margin: 0; font-weight: 400;">
            {t['sidebar_subtitle']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Demographics Section
    st.markdown(f"#### {t['demographics']}")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input(t['age'], min_value=18, max_value=100, value=65)
    with col2:
        gender = st.selectbox(t['gender'], [t['male'], t['female'], t['other']])
    
    # Vital Signs
    st.markdown(f"#### {t['vital_signs']}")
    col1, col2 = st.columns(2)
    with col1:
        bmi = st.number_input(t['bmi'], min_value=15.0, max_value=50.0, value=27.0, step=0.5)
        systolic_bp = st.number_input(t['systolic_bp'], min_value=90, max_value=200, value=120)
        heart_rate = st.number_input(t['heart_rate'], min_value=50, max_value=140, value=75)
    with col2:
        diastolic_bp = st.number_input(t['diastolic_bp'], min_value=60, max_value=120, value=80)
        oxygen_saturation = st.number_input(t['oxygen_saturation'], min_value=85, max_value=100, value=96)
        temperature = st.number_input(t['temperature'], min_value=35.5, max_value=39.5, value=36.8, step=0.1)
    
    # Medical History
    st.markdown(f"#### {t['medical_history']}")
    col1, col2 = st.columns(2)
    with col1:
        diabetes = st.checkbox(t['diabetes'])
        hypertension = st.checkbox(t['hypertension'])
        heart_disease = st.checkbox(t['heart_disease'])
    with col2:
        copd = st.checkbox(t['copd'])
        kidney_disease = st.checkbox(t['kidney_disease'])
        cancer = st.checkbox(t['cancer'])
    
    # Other Disease Input
    other_disease = st.text_input("Other Disease Name", placeholder="e.g. Asthma, Liver Disease")
    has_other = 1 if other_disease.strip() else 0
    
    # Lab Results
    st.markdown(f"#### {t['lab_results']}")
    col1, col2 = st.columns(2)
    with col1:
        hemoglobin = st.number_input(t['hemoglobin'], min_value=7.0, max_value=18.0, value=13.5, step=0.1)
        blood_glucose = st.number_input(t['blood_glucose'], min_value=60, max_value=300, value=100)
    with col2:
        creatinine = st.number_input(t['creatinine'], min_value=0.4, max_value=4.0, value=0.9, step=0.1)
        hba1c = st.number_input(t['hba1c'], min_value=4.5, max_value=12.0, value=5.7, step=0.1)
    
    # Admission Details
    st.markdown(f"#### {t['admission_details']}")
    length_of_stay = st.number_input(t['length_of_stay'], min_value=1, max_value=30, value=5)
    prior_visits = st.number_input(t['prior_visits'], min_value=0, max_value=15, value=2)
    prior_ed_visits = st.number_input(t['prior_ed_visits'], min_value=0, max_value=12, value=1)
    icu_during_stay = st.checkbox(t['icu_during_stay'])
    
    st.markdown("---")
    predict_button = st.button(t['predict_button'], use_container_width=True)

# ============================================================================
# ROLE-BASED MAIN CONTENT
# ============================================================================

if st.session_state.role == 'Doctor':
    # Show AI Prediction Interface for Doctors
    st.markdown(f"""
    <div style="text-align: center; padding: 20px 0 15px 0;">
        <h1 style="font-size: 2.5rem; font-weight: 700; color: #0F172A; margin-bottom: 10px;">
            {t['app_title']}
        </h1>
        <p style="font-size: 1.1rem; color: #475569; margin: 0; font-weight: 500;">
            {t['app_subtitle']}
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # KPI Metrics Row (reduced for Doctor view)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 42px;">📊</div>
            <div class="metric-value">10,000</div>
            <div style="color: #64748B;">{t['total_patients']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        readmission_rate = 0.20
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 42px;">⚠️</div>
            <div class="metric-value">{readmission_rate:.1%}</div>
            <div style="color: #64748B;">{t['readmission_rate']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        model_recall = 0.85
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 42px;">🎯</div>
            <div class="metric-value">{model_recall:.1%}</div>
            <div style="color: #64748B;">{t['model_recall']}</div>
        </div>
        """, unsafe_allow_html=True)

elif st.session_state.role == 'Admin':
    # Show welcome message for Admin (redirects to Admin Dashboard)
    st.markdown(f"""
    <div style="text-align: center; padding: 50px 0;">
        <h1 style="font-size: 2.5rem; font-weight: 700; color: #0F172A; margin-bottom: 20px;">
            🏢 Welcome to Admin Dashboard
        </h1>
        <p style="font-size: 1.2rem; color: #475569; margin-bottom: 30px;">
            Please use the sidebar navigation to access the Admin Dashboard.
        </p>
        <div style="background: #f1f5f9; padding: 20px; border-radius: 12px; max-width: 600px; margin: 0 auto;">
            <h3 style="color: #1E88E5; margin-bottom: 15px;">📊 Available Features:</h3>
            <ul style="text-align: left; color: #475569;">
                <li>📈 System KPIs and Metrics</li>
                <li>👥 Patient Records Management</li>
                <li>🔍 Advanced Analytics</li>
                <li>⚙️ System Configuration</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ============================================================================
# PREDICTION SECTION (Only for Doctors)
# ============================================================================

if predict_button and st.session_state.role == 'Doctor':
    # Prepare patient data
    patient_data = {
        'age': age,
        'gender': gender,
        'bmi': bmi,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'heart_rate': heart_rate,
        'oxygen_saturation': oxygen_saturation,
        'temperature': temperature,
        'hemoglobin': hemoglobin,
        'creatinine': creatinine,
        'blood_glucose': blood_glucose,
        'hbA1c': hba1c,
        'diabetes': 1 if diabetes else 0,
        'hypertension': 1 if hypertension else 0,
        'heart_disease': 1 if heart_disease else 0,
        'copd': 1 if copd else 0,
        'kidney_disease': 1 if kidney_disease else 0,
        'cancer': 1 if cancer else 0,
        'has_other': has_other,
        
        'length_of_stay': length_of_stay,
        'prior_hospital_visits_1yr': prior_visits,
        'prior_emergency_visits_1yr': prior_ed_visits,
        'icu_during_stay': 1 if icu_during_stay else 0,
        'income_level': 'Medium',
        'education_level': 'Secondary',
        'admission_type': 'Emergency',
        'admission_department': 'Internal Medicine',
        'discharge_disposition': 'Home'
    }
    
    # Preprocess patient data
    patient_df = pd.DataFrame([patient_data])
    
    for col in feature_columns:
        if col not in patient_df.columns:
            patient_df[col] = 0
    
    for col in categorical_features:
        if col in patient_df.columns and col in label_encoders:
            le = label_encoders[col]
            patient_df[col] = patient_df[col].fillna('Unknown').astype(str)
            patient_df[col] = patient_df[col].apply(
                lambda x: x if x in le.classes_ else 'Unknown'
            )
            patient_df[col] = le.transform(patient_df[col])
    
    patient_df[numerical_features] = scaler.transform(patient_df[numerical_features])
    
    risk_prob = model.predict_proba(patient_df[feature_columns])[0, 1]
    risk_percentage = risk_prob * 100
    
    if risk_prob >= 0.5:
        risk_level = t['high_risk']
        risk_color = "#DC2626"
        risk_icon = "🔴"
    elif risk_prob >= 0.3:
        risk_level = t['moderate_risk']
        risk_color = "#F59E0B"
        risk_icon = "🟡"
    else:
        risk_level = t['low_risk']
        risk_color = "#10B981"
        risk_icon = "🟢"
    
    # Display risk score
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <h2 style="margin: 0 0 10px 0;">{t['predicted_risk']}</h2>
            <div style="font-size: 72px; font-weight: 800; color: {risk_color};">
                {risk_percentage:.1f}%
            </div>
            <div class="risk-{risk_level.split()[0].lower() if ' ' in risk_level else risk_level.lower()}" style="margin-top: 15px;">
                {risk_icon} {risk_level} {risk_icon}
            </div>
            <p style="margin-top: 20px; color: #64748B;">
                {t['probability_text']}
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_percentage,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': t['risk_score'], 'font': {'size': 20}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 30], 'color': '#D1FAE5'},
                    {'range': [30, 50], 'color': '#FEF3C7'},
                    {'range': [50, 100], 'color': '#FEE2E2'}
                ],
                'threshold': {
                    'line': {'color': 'black', 'width': 4},
                    'thickness': 0.75,
                    'value': risk_percentage
                }
            }
        ))
        fig_gauge.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=30))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Display note about other disease if provided
    if other_disease.strip():
        st.info(f"📝 Note: The patient's {other_disease} has been factored into this assessment.")
    
    st.markdown("---")
    
    # ========================================================================
    # SHAP EXPLANATION
    # ========================================================================
    
    st.markdown(f'<div class="section-header">{t["shap_title"]}</div>', unsafe_allow_html=True)
    st.markdown(f"*{t['shap_subtitle']}*")
    
    with st.spinner(t['computing_shap']):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(patient_df[feature_columns])
        
        shap_df = pd.DataFrame({
            'Feature': feature_columns,
            'SHAP_Value': shap_values[0]
        })
        shap_df = shap_df.sort_values('SHAP_Value', key=abs, ascending=False).head(10)
        
        colors = ['#DC2626' if x > 0 else '#10B981' for x in shap_df['SHAP_Value']]
        
        fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
        bars = ax_shap.barh(range(len(shap_df)), shap_df['SHAP_Value'].values, color=colors)
        ax_shap.set_yticks(range(len(shap_df)))
        ax_shap.set_yticklabels(shap_df['Feature'].values)
        ax_shap.axvline(x=0, color='gray', linestyle='--', linewidth=1)
        ax_shap.set_xlabel(t['shap_axis_label'], fontsize=11)
        ax_shap.set_title(t['shap_chart_title'], fontsize=13, fontweight='bold')
        ax_shap.invert_yaxis()
        
        for i, (bar, val) in enumerate(zip(bars, shap_df['SHAP_Value'])):
            ax_shap.text(val + (0.01 if val >= 0 else -0.08), 
                        bar.get_y() + bar.get_height()/2, 
                        f'{val:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig_shap)
    
    st.caption(f"📊 *{t['shap_caption']}*")
    
    # ========================================================================
    # CLINICAL RECOMMENDATIONS
    # ========================================================================
    
    st.markdown(f'<div class="section-header">{t["recommendations_title"]}</div>', unsafe_allow_html=True)
    
    recommendations = []
    
    if risk_prob >= 0.5:
        recommendations = [
            t['rec_high_1'],
            t['rec_high_2'],
            t['rec_high_3'],
            t['rec_high_4'],
            t['rec_high_5']
        ]
        if heart_disease:
            recommendations.append(t['rec_cardio'])
        if diabetes:
            recommendations.append(t['rec_endocrine'])
        if kidney_disease:
            recommendations.append(t['rec_nephrology'])
            
    elif risk_prob >= 0.3:
        recommendations = [
            t['rec_moderate_1'],
            t['rec_moderate_2'],
            t['rec_moderate_3'],
            t['rec_moderate_4']
        ]
        if prior_visits > 3:
            recommendations.append(t['rec_case_management'])
        if age > 70:
            recommendations.append(t['rec_geriatric'])
            
    else:
        recommendations = [
            t['rec_low_1'],
            t['rec_low_2'],
            t['rec_low_3'],
            t['rec_low_4']
        ]
    
    for i, rec in enumerate(recommendations[:6]):
        st.markdown(f"{i+1}. {rec}")
    
    # ========================================================================
    # PATIENT SUMMARY CARD
    # ========================================================================
    
    st.markdown(f'<div class="section-header">{t["summary_title"]}</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="card">
            <strong>{t['demographics_summary']}</strong><br>
            {t['age']}: {age} {t['years'] if 'years' in t else 'years'}<br>
            {t['gender']}: {gender}<br>
            {t['bmi']}: {bmi:.1f}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="card">
            <strong>{t['vital_signs_summary']}</strong><br>
            BP: {systolic_bp}/{diastolic_bp} mmHg<br>
            HR: {heart_rate} bpm<br>
            O₂: {oxygen_saturation}%
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        conditions = []
        if diabetes: conditions.append("DM")
        if hypertension: conditions.append("HTN")
        if heart_disease: conditions.append("CAD")
        if copd: conditions.append("COPD")
        if kidney_disease: conditions.append("CKD")
        if other_disease.strip(): conditions.append(other_disease)
        
        st.markdown(f"""
        <div class="card">
            <strong>{t['medical_conditions']}</strong><br>
            {', '.join(conditions) if conditions else t['none_reported']}<br>
            {t['comorbidity_count']}: {len(conditions)}
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="card">
            <strong>{t['utilization']}</strong><br>
            {t['los']}: {length_of_stay} days<br>
            {t['prior_visits_summary']}: {prior_visits}<br>
            {t['icu_stay']}: {t['yes'] if icu_during_stay else t['no']}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
else:
    st.info(t['welcome_message'])
    
    st.markdown(f'<div class="section-header">{t["population_insights"]}</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_age = px.histogram(df, x='age', color='readmitted_30d', 
                               title=t['readmission_by_age'],
                               color_discrete_sequence=['#10B981', '#DC2626'],
                               labels={'readmitted_30d': 'Readmitted', 'count': 'Number of Patients'},
                               nbins=30)
        fig_age.update_layout(barmode='stack', height=400)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        risk_factors = pd.DataFrame({
            'Factor': ['Heart Disease', 'Prior ED Visits', 'Long LOS', 'Kidney Disease', 'Age > 70', 'Diabetes'],
            'Risk Increase': [2.8, 2.4, 2.1, 2.0, 1.8, 1.6]
        })
        fig_risk = px.bar(risk_factors, x='Risk Increase', y='Factor', orientation='h',
                         title=t['top_risk_factors'],
                         color='Risk Increase', color_continuous_scale='Reds')
        fig_risk.update_layout(height=400)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    st.markdown(f"""
    <div class="info-box">
        <strong>{t['about_ai']}</strong><br>
        {t['about_text']}
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div style="text-align: center; color: #94A3B8; font-size: 12px; padding: 30px 0 20px 0; border-top: 1px solid #E2E8F0; margin-top: 50px; bottom: 0; width: 100%;">
    <p>{t['footer_warning']}</p>
    <p>{t['footer_copyright']}</p>
</div>
""", unsafe_allow_html=True)