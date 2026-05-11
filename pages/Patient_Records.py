"""
Patient Records - MediPredict AI
Accessible to both Admin and Doctor users
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
from io import BytesIO

# Check if user is logged in
if not st.session_state.get('logged_in', False):
    st.error("🚫 Access denied. Please login to view patient records.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Patient Records | MediPredict AI",
    page_icon="👥",
    layout="wide"
)

# Initialize session state for patient records if not exists
if 'patient_df' not in st.session_state:
    # Generate sample patient data
    np.random.seed(42)
    patients = []
    for i in range(100):
        patient_id = f"P{1000 + i}"
        names = ["John Smith", "Mary Johnson", "Robert Williams", "Patricia Brown", 
                "Michael Davis", "Linda Miller", "James Wilson", "Elizabeth Moore",
                "David Taylor", "Barbara Anderson", "Richard Thomas", "Jennifer Jackson",
                "Joseph White", "Susan Harris", "Charles Martin", "Margaret Thompson",
                "Thomas Garcia", "Dorothy Martinez", "Christopher Robinson", "Lisa Clark"]
        
        patient = {
            'patient_id': patient_id,
            'name': names[i % len(names)] + f" {i}",
            'age': np.random.randint(18, 95),
            'gender': np.random.choice(['Male', 'Female']),
            'admission_date': datetime.now() - timedelta(days=np.random.randint(1, 365)),
            'discharge_date': datetime.now() - timedelta(days=np.random.randint(0, 30)),
            'length_of_stay': np.random.randint(1, 15),
            'diagnosis': np.random.choice(['Heart Failure', 'Diabetes', 'Pneumonia', 'COPD', 'Kidney Disease', 'Cancer']),
            'readmission_risk': np.random.choice(['High', 'Moderate', 'Low'], p=[0.2, 0.3, 0.5]),
            'risk_score': round(np.random.uniform(0.1, 0.9), 2),
            'predicted_by': 'AI Model v2.1',
            'status': np.random.choice(['Discharged', 'In Treatment', 'Follow-up Required'])
        }
        patients.append(patient)
    
    st.session_state.patient_df = pd.DataFrame(patients)

# Custom CSS (minimized)
st.markdown("""
<style>
    .header {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .search-box {
        background: #f8fafc;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-bottom: 15px;
    }
    .patient-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 8px;
        transition: all 0.2s;
    }
    .risk-high {
        background: #FEE2E2;
        border-left: 3px solid #DC2626;
    }
    .risk-moderate {
        background: #FEF3C7;
        border-left: 3px solid #F59E0B;
    }
    .risk-low {
        background: #D1FAE5;
        border-left: 3px solid #10B981;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.empty()

# Home button - goes to doctor dashboard
if st.sidebar.button("🏠 Home", use_container_width=True):
    st.session_state.page = "app"
    st.rerun()

# Show user info
st.sidebar.markdown("---")
st.sidebar.markdown(f"*Logged in as: {st.session_state.get('username', 'User')}*")
st.sidebar.markdown(f"*Role: {st.session_state.get('role', 'User')}*")

# Patient Input Form (for Doctors only)
if st.session_state.get('role', '') == 'Doctor':
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 Add New Patient Record")
    
    with st.sidebar.form("patient_form", clear_on_submit=True):
        # Patient Information
        st.markdown("**Patient Info**")
        patient_name = st.text_input("Patient Name*", placeholder="Enter patient name")
        patient_id = st.text_input("Patient ID", placeholder="Leave blank for auto-generate")
        age = st.number_input("Age*", min_value=0, max_value=120, value=50)
        gender = st.selectbox("Gender*", ["Male", "Female"])
        
        # Medical Information
        st.markdown("**Medical Info**")
        diagnosis = st.text_input("Diagnosis*", placeholder="Primary diagnosis")
        risk_level = st.selectbox("Risk Level*", ["Low", "Moderate", "High"])
        
        # Visit Information
        st.markdown("**Visit Info**")
        admission_date = st.date_input("Admission Date*", datetime.now())
        discharge_date = st.date_input("Discharge Date*", datetime.now() + timedelta(days=5))
        length_of_stay = st.number_input("Length of Stay (days)*", min_value=0, value=5)
        
        # Calculate risk score based on risk level
        risk_score_map = {"Low": 0.25, "Moderate": 0.55, "High": 0.85}
        risk_score = risk_score_map[risk_level]
        
        # Calculate status
        if discharge_date > admission_date:
            status = "Discharged"
        elif discharge_date == admission_date:
            status = "In Treatment"
        else:
            status = "In Treatment"
        
        # Submit button
        submit_patient = st.form_submit_button("➕ Add Patient Record", use_container_width=True)
        
        if submit_patient:
            if not patient_name:
                st.sidebar.error("❌ Patient Name is required!")
            elif not diagnosis:
                st.sidebar.error("❌ Diagnosis is required!")
            else:
                # Auto-generate patient ID if not provided
                if not patient_id:
                    next_id = len(st.session_state.patient_df) + 1000
                    patient_id = f"P{next_id}"
                
                # Create new patient record
                new_patient = {
                    'patient_id': patient_id,
                    'name': patient_name,
                    'age': age,
                    'gender': gender,
                    'admission_date': pd.to_datetime(admission_date),
                    'discharge_date': pd.to_datetime(discharge_date),
                    'length_of_stay': length_of_stay,
                    'diagnosis': diagnosis,
                    'readmission_risk': risk_level,
                    'risk_score': risk_score,
                    'predicted_by': f"Dr. {st.session_state.get('username', 'Doctor')}",
                    'status': status
                }
                
                # Add to dataframe in session state
                st.session_state.patient_df = pd.concat(
                    [st.session_state.patient_df, pd.DataFrame([new_patient])], 
                    ignore_index=True
                )
                st.sidebar.success(f"✅ Patient '{patient_name}' added successfully!")
                st.rerun()

# Header (minimized)
st.markdown("## 👥 Patient Records")

# Search and Filter Section
st.markdown('<div class="search-box">', unsafe_allow_html=True)
st.markdown("### 🔍 Search & Filter")

col1, col2, col3, col4 = st.columns([2, 2, 2, 1])

with col1:
    search_name = st.text_input("Search by Patient Name or ID", placeholder="Enter name or ID...")

with col2:
    filter_risk = st.selectbox("Filter by Risk Level", ["All", "High", "Moderate", "Low"])

with col3:
    filter_status = st.selectbox("Filter by Status", ["All", "Discharged", "In Treatment", "Follow-up Required"])

with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    reset_button = st.button("🔄 Reset Filters", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# Reset filters if reset button is clicked
if reset_button:
    search_name = ""
    filter_risk = "All"
    filter_status = "All"
    st.rerun()

# Apply filters
filtered_df = st.session_state.patient_df.copy()

if search_name:
    filtered_df = filtered_df[
        filtered_df['name'].str.contains(search_name, case=False, na=False) |
        filtered_df['patient_id'].str.contains(search_name, case=False, na=False)
    ]

if filter_risk != "All":
    filtered_df = filtered_df[filtered_df['readmission_risk'] == filter_risk]

if filter_status != "All":
    filtered_df = filtered_df[filtered_df['status'] == filter_status]

# Statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Patients", len(filtered_df))

with col2:
    high_risk_count = len(filtered_df[filtered_df['readmission_risk'] == 'High'])
    risk_percentage = (high_risk_count/len(filtered_df)*100) if len(filtered_df) > 0 else 0
    st.metric("High Risk Patients", high_risk_count, delta=f"{risk_percentage:.1f}%")

with col3:
    avg_los = filtered_df['length_of_stay'].mean() if len(filtered_df) > 0 else 0
    st.metric("Avg. Length of Stay", f"{avg_los:.1f} days")

with col4:
    follow_up = len(filtered_df[filtered_df['status'] == 'Follow-up Required'])
    st.metric("Follow-up Required", follow_up)

st.markdown("---")

# View Options
view_option = st.radio("View Options", ["📋 Table View", "🃏 Card View", "📊 Analytics"], horizontal=True)

if view_option == "📋 Table View":
    # Table View
    st.markdown("### 📋 Patient Records Table")
    
    # Format the data for display
    display_df = filtered_df.copy()
    if 'admission_date' in display_df.columns:
        display_df['Admission Date'] = pd.to_datetime(display_df['admission_date']).dt.strftime('%Y-%m-%d')
    if 'discharge_date' in display_df.columns:
        display_df['Discharge Date'] = pd.to_datetime(display_df['discharge_date']).dt.strftime('%Y-%m-%d')
    display_df['Risk Score'] = display_df['risk_score'].apply(lambda x: f"{x:.0%}")
    
    # Select columns for display
    display_columns = ['patient_id', 'name', 'age', 'gender', 'diagnosis', 
                      'readmission_risk', 'risk_score', 'status']
    
    # Add date columns if they exist
    if 'Admission Date' in display_df.columns:
        display_columns.append('Admission Date')
    if 'Discharge Date' in display_df.columns:
        display_columns.append('Discharge Date')
    display_columns.append('length_of_stay')
    
    display_df = display_df[display_columns]
    
    # Show total records count
    st.info(f"📊 Showing {len(display_df)} patient records")
    
    # Display dataframe with color coding for risk levels
    st.dataframe(
        display_df,
        use_container_width=True,
        height=500,
        column_config={
            "readmission_risk": st.column_config.Column(
                "Readmission Risk",
                help="Patient readmission risk level"
            ),
            "risk_score": st.column_config.Column(
                "Risk Score",
                help="Predicted readmission probability"
            ),
            "status": st.column_config.Column(
                "Status",
                help="Current patient status"
            )
        }
    )

elif view_option == "🃏 Card View":
    # Card View
    st.markdown("### 🃏 Patient Cards")
    
    # Display patients in cards (limit to first 20 for performance)
    for idx, patient in filtered_df.head(20).iterrows():
        risk_class = f"risk-{patient['readmission_risk'].lower()}"
        risk_emoji = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}[patient['readmission_risk']]
        
        # Format dates
        admission_str = patient['admission_date'].strftime('%Y-%m-%d') if hasattr(patient['admission_date'], 'strftime') else str(patient['admission_date'])
        
        st.markdown(f"""
        <div class="patient-card {risk_class}">
            <div style="display: flex; justify-content: space-between; align-items: start;">
                <div>
                    <h4 style="margin: 0; color: #1E293B;">{patient['name']}</h4>
                    <p style="margin: 5px 0; color: #64748B; font-size: 14px;">
                        ID: {patient['patient_id']} | Age: {patient['age']} | {patient['gender']}
                    </p>
                    <p style="margin: 5px 0; color: #475569;">
                        <strong>Diagnosis:</strong> {patient['diagnosis']}
                    </p>
                    <p style="margin: 5px 0; color: #475569;">
                        <strong>Status:</strong> {patient['status']} | 
                        <strong>LOS:</strong> {patient['length_of_stay']} days
                    </p>
                    <p style="margin: 5px 0; color: #475569; font-size: 12px;">
                        <strong>Admitted:</strong> {admission_str}
                    </p>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 24px;">{risk_emoji}</div>
                    <div style="font-weight: bold; color: #1E293B;">{patient['readmission_risk']} Risk</div>
                    <div style="color: #64748B; font-size: 18px;">{patient['risk_score']:.0%}</div>
                    <div style="color: #64748B; font-size: 12px;">Predicted by:<br>{patient['predicted_by']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if len(filtered_df) > 20:
        st.info(f"Showing first 20 of {len(filtered_df)} patients. Use Table View to see all records.")

else:  # Analytics View
    # Analytics View
    st.markdown("### 📊 Patient Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk Distribution
        risk_counts = filtered_df['readmission_risk'].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Distribution",
            color_discrete_map={
                'High': '#DC2626',
                'Moderate': '#F59E0B',
                'Low': '#10B981'
            }
        )
        fig_risk.update_layout(height=400)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with col2:
        # Age Distribution
        fig_age = px.histogram(
            filtered_df, 
            x='age', 
            color='readmission_risk',
            title="Age Distribution by Risk Level",
            nbins=20,
            color_discrete_map={
                'High': '#DC2626',
                'Moderate': '#F59E0B',
                'Low': '#10B981'
            },
            barmode='group'
        )
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)
    
    # Diagnosis Analysis
    st.markdown("### 🏥 Diagnosis Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        diagnosis_counts = filtered_df['diagnosis'].value_counts()
        fig_diagnosis = px.bar(
            x=diagnosis_counts.values,
            y=diagnosis_counts.index,
            orientation='h',
            title="Diagnosis Distribution",
            color=diagnosis_counts.values,
            color_continuous_scale='Viridis'
        )
        fig_diagnosis.update_layout(height=400)
        st.plotly_chart(fig_diagnosis, use_container_width=True)
    
    with col2:
        # Length of Stay by Risk
        fig_los = px.box(
            filtered_df, 
            x='readmission_risk', 
            y='length_of_stay',
            title="Length of Stay by Risk Level",
            color='readmission_risk',
            color_discrete_map={
                'High': '#DC2626',
                'Moderate': '#F59E0B',
                'Low': '#10B981'
            }
        )
        fig_los.update_layout(height=400)
        st.plotly_chart(fig_los, use_container_width=True)

# Export functionality (Admin only)
if st.session_state.get('role', '') == 'Admin':
    st.markdown("---")
    st.markdown("### 📤 Export Data (Admin Only)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export to CSV", use_container_width=True):
            # Create CSV
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"patient_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="csv_download"
            )
    
    with col2:
        if st.button("📋 Export to Excel", use_container_width=True):
            # Create Excel file
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered_df.to_excel(writer, index=False, sheet_name='Patient Records')
            excel_data = output.getvalue()
            st.download_button(
                label="📥 Download Excel",
                data=excel_data,
                file_name=f"patient_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="excel_download"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; padding: 20px;">
    <p>© 2026 MediPredict AI | Patient Records System | HIPAA Compliant</p>
    <p style="font-size: 12px;">Total patients in database: {}</p>
</div>
""".format(len(st.session_state.patient_df)), unsafe_allow_html=True)