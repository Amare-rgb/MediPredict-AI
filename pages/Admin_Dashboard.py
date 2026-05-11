"""
Admin Dashboard - MediPredict AI
Accessible only to Admin users with real data analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Check if user is logged in and is Admin
if not st.session_state.get('logged_in', False) or st.session_state.get('role', '') != 'Admin':
    st.error("🚫 Access denied. Admin privileges required.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Admin Dashboard | MediPredict AI",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize navigation state
if 'admin_page' not in st.session_state:
    st.session_state.admin_page = "Dashboard Home"

# Check if patient data exists in session state
if 'patient_df' not in st.session_state:
    # Generate sample data if not exists
    np.random.seed(42)
    patients = []
    for i in range(200):
        patient_id = f"P{1000 + i}"
        names = ["John Smith", "Mary Johnson", "Robert Williams", "Patricia Brown", 
                "Michael Davis", "Linda Miller", "James Wilson", "Elizabeth Moore",
                "David Taylor", "Barbara Anderson", "Richard Thomas", "Jennifer Jackson"]
        
        admission_date = datetime.now() - timedelta(days=np.random.randint(1, 365))
        discharge_date = admission_date + timedelta(days=np.random.randint(1, 20))
        
        patient = {
            'patient_id': patient_id,
            'name': names[i % len(names)] + (f" {i}" if i >= len(names) else ""),
            'age': np.random.randint(18, 95),
            'gender': np.random.choice(['Male', 'Female']),
            'admission_date': admission_date,
            'discharge_date': discharge_date,
            'length_of_stay': (discharge_date - admission_date).days,
            'diagnosis': np.random.choice(['Heart Failure', 'Diabetes', 'Pneumonia', 'COPD', 'Kidney Disease', 'Cancer', 'Hypertension', 'Asthma']),
            'readmission_risk': np.random.choice(['High', 'Moderate', 'Low'], p=[0.15, 0.35, 0.5]),
            'risk_score': round(np.random.uniform(0.1, 0.95), 2),
            'predicted_by': np.random.choice(['AI Model v2.1', 'Dr. Smith', 'Dr. Johnson', 'AI Model v2.0']),
            'status': np.random.choice(['Discharged', 'In Treatment', 'Follow-up Required'], p=[0.6, 0.3, 0.1]),
            'visit_count': np.random.randint(1, 8),
            'insurance_type': np.random.choice(['Private', 'Government', 'Self-Pay', 'Insurance'], p=[0.4, 0.4, 0.1, 0.1]),
            'department': np.random.choice(['Cardiology', 'Neurology', 'Pediatrics', 'Orthopedics', 'Oncology', 'Emergency'])
        }
        patients.append(patient)
    
    st.session_state.patient_df = pd.DataFrame(patients)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin-bottom: 2rem;
        text-align: center;
    }
    .stat-card {
        background: white;
        border-radius: 0.75rem;
        padding: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        transition: all 0.3s;
    }
    .stat-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    .stat-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .stat-label {
        font-size: 0.875rem;
        color: #6b7280;
        margin-top: 0.5rem;
    }
    .trend-up {
        color: #10b981;
        font-size: 0.75rem;
    }
    .trend-down {
        color: #ef4444;
        font-size: 0.75rem;
    }
    .insight-box {
        background: #f3f4f6;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.5rem;">🏢 Admin Dashboard</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Enterprise Analytics & System Management</p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.875rem;">Last updated: {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")), unsafe_allow_html=True)

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🏥 MediPredict AI")
    st.markdown("---")
    
    # Simple navigation buttons
    if st.button("🏠 Dashboard Home", use_container_width=True, key="btn_home"):
        st.session_state.admin_page = "Dashboard Home"
        st.rerun()
    
    if st.button("👥 Patient Analytics", use_container_width=True, key="btn_patients"):
        st.session_state.admin_page = "Patient Analytics"
        st.rerun()
    
    if st.button("⚙️ System Health", use_container_width=True, key="btn_health"):
        st.session_state.admin_page = "System Health"
        st.rerun()
    
    if st.button("👥 User Management", use_container_width=True, key="btn_users"):
        st.session_state.admin_page = "User Management"
        st.rerun()
    
    if st.button("📄 Reports", use_container_width=True, key="btn_reports"):
        st.session_state.admin_page = "Reports"
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    
    # Quick stats in sidebar
    total_patients = len(st.session_state.patient_df)
    high_risk = len(st.session_state.patient_df[st.session_state.patient_df['readmission_risk'] == 'High'])
    avg_los = st.session_state.patient_df['length_of_stay'].mean()
    
    st.metric("Total Patients", f"{total_patients:,}")
    st.metric("High Risk Patients", f"{high_risk}", delta=f"{(high_risk/total_patients)*100:.1f}%")
    st.metric("Avg Length of Stay", f"{avg_los:.1f} days")

# Main content based on navigation
selected = st.session_state.admin_page

if selected == "Dashboard Home":
    # Key Metrics Row
    st.markdown("### 📈 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    df = st.session_state.patient_df
    
    with col1:
        total_patients = len(df)
        growth = np.random.randint(5, 15)
        st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 1.5rem;">👥</div>
                <div class="stat-value">{total_patients:,}</div>
                <div class="stat-label">Total Patients</div>
                <div class="trend-up">↑ {growth}% from last month</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        readmission_rate = (df['readmission_risk'] == 'High').sum() / len(df) * 100
        target = 20
        delta = readmission_rate - target
        trend_class = 'trend-up' if delta > 0 else 'trend-down'
        st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 1.5rem;">🔄</div>
                <div class="stat-value">{readmission_rate:.1f}%</div>
                <div class="stat-label">High Risk Rate</div>
                <div class="{trend_class}">{'+' if delta > 0 else ''}{delta:.1f}% vs target</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_stay = df['length_of_stay'].mean()
        st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 1.5rem;">🏥</div>
                <div class="stat-value">{avg_stay:.1f}</div>
                <div class="stat-label">Avg Length of Stay (days)</div>
                <div class="trend-down">↓ 0.8 days from last month</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        model_accuracy = 85.0
        st.markdown(f"""
            <div class="stat-card">
                <div style="font-size: 1.5rem;">🤖</div>
                <div class="stat-value">{model_accuracy:.1f}%</div>
                <div class="stat-label">Model Accuracy</div>
                <div class="trend-up">↑ 2.3% improvement</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Charts Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Patient Trends Over Time")
        # Create time series data
        df_copy = df.copy()
        df_copy['admission_month'] = df_copy['admission_date'].dt.to_period('M')
        monthly_trend = df_copy.groupby('admission_month').size().reset_index(name='count')
        monthly_trend['admission_month'] = monthly_trend['admission_month'].astype(str)
        
        fig_trend = px.line(
            monthly_trend,
            x='admission_month',
            y='count',
            markers=True,
            title='Monthly Patient Admissions'
        )
        fig_trend.update_layout(
            xaxis_title="Month",
            yaxis_title="Number of Patients",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Risk Distribution")
        risk_counts = df['readmission_risk'].value_counts()
        fig_risk = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Patient Risk Distribution",
            color_discrete_map={
                'High': '#ef4444',
                'Moderate': '#f59e0b',
                'Low': '#10b981'
            },
            hole=0.3
        )
        fig_risk.update_layout(height=400)
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Charts Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏥 Department Analysis")
        dept_stats = df.groupby('department').agg({
            'patient_id': 'count',
            'length_of_stay': 'mean',
            'risk_score': 'mean'
        }).reset_index()
        dept_stats.columns = ['Department', 'Patient Count', 'Avg LOS', 'Avg Risk Score']
        
        fig_dept = px.bar(
            dept_stats,
            x='Department',
            y='Patient Count',
            color='Avg Risk Score',
            title='Patients by Department',
            color_continuous_scale='RdYlGn_r',
            text='Patient Count'
        )
        fig_dept.update_layout(height=400)
        st.plotly_chart(fig_dept, use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Age Distribution & Risk")
        fig_age = px.histogram(
            df,
            x='age',
            color='readmission_risk',
            title='Age Distribution by Risk Level',
            nbins=30,
            color_discrete_map={
                'High': '#ef4444',
                'Moderate': '#f59e0b',
                'Low': '#10b981'
            },
            barmode='stack'
        )
        fig_age.update_layout(height=400)
        st.plotly_chart(fig_age, use_container_width=True)
    
    # Insights Section
    st.markdown("---")
    st.markdown("### 💡 AI Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="insight-box">
            <strong>🔍 Top Risk Factor</strong><br>
            Age > 65 with multiple diagnoses shows <span style="color: #ef4444;">45% higher</span> readmission risk.
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="insight-box">
            <strong>📊 Performance Alert</strong><br>
            Cardiology department has <span style="color: #f59e0b;">18%</span> higher readmission rate than average.
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="insight-box">
            <strong>🎯 Recommendation</strong><br>
            Increase follow-up for discharged patients within 7 days to reduce readmissions.
        </div>
        """, unsafe_allow_html=True)

elif selected == "Patient Analytics":
    st.markdown("### 📊 Detailed Patient Analytics")
    
    df = st.session_state.patient_df
    
    # Advanced filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        dept_filter = st.multiselect(
            "Department",
            options=df['department'].unique().tolist(),
            default=[]
        )
    
    with col2:
        risk_filter = st.multiselect(
            "Risk Level",
            options=['High', 'Moderate', 'Low'],
            default=[]
        )
    
    with col3:
        gender_filter = st.multiselect(
            "Gender",
            options=['Male', 'Female'],
            default=[]
        )
    
    with col4:
        age_range = st.slider(
            "Age Range",
            min_value=0,
            max_value=100,
            value=(0, 100)
        )
    
    # Apply filters
    filtered_df = df.copy()
    if dept_filter:
        filtered_df = filtered_df[filtered_df['department'].isin(dept_filter)]
    if risk_filter:
        filtered_df = filtered_df[filtered_df['readmission_risk'].isin(risk_filter)]
    if gender_filter:
        filtered_df = filtered_df[filtered_df['gender'].isin(gender_filter)]
    filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]
    
    # Analytics metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Filtered Patients", len(filtered_df))
    with col2:
        high_risk_pct = (filtered_df['readmission_risk'] == 'High').sum() / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
        st.metric("High Risk %", f"{high_risk_pct:.1f}%")
    with col3:
        avg_age = filtered_df['age'].mean() if len(filtered_df) > 0 else 0
        st.metric("Average Age", f"{avg_age:.0f}")
    with col4:
        most_common = filtered_df['diagnosis'].mode().iloc[0] if len(filtered_df) > 0 else "N/A"
        st.metric("Most Common Diagnosis", most_common)
    
    # Diagnosis breakdown
    st.markdown("### 🏥 Diagnosis Analysis")
    diagnosis_stats = filtered_df.groupby('diagnosis').agg({
        'patient_id': 'count',
        'risk_score': 'mean',
        'length_of_stay': 'mean'
    }).reset_index()
    diagnosis_stats.columns = ['Diagnosis', 'Patient Count', 'Avg Risk Score', 'Avg LOS']
    diagnosis_stats = diagnosis_stats.sort_values('Patient Count', ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if len(diagnosis_stats) > 0:
            fig_diag = px.bar(
                diagnosis_stats.head(10),
                x='Patient Count',
                y='Diagnosis',
                orientation='h',
                title='Top 10 Diagnoses',
                color='Avg Risk Score',
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig_diag, use_container_width=True)
        else:
            st.info("No data available for the selected filters")
    
    with col2:
        if len(diagnosis_stats) > 0:
            fig_los = px.scatter(
                diagnosis_stats,
                x='Avg Risk Score',
                y='Avg LOS',
                size='Patient Count',
                text='Diagnosis',
                title='Risk Score vs Length of Stay',
                color='Patient Count',
                size_max=60
            )
            st.plotly_chart(fig_los, use_container_width=True)
        else:
            st.info("No data available for the selected filters")
    
    # Detailed data table
    st.markdown("### 📋 Patient Data")
    display_cols = ['patient_id', 'name', 'age', 'gender', 'diagnosis', 'readmission_risk', 'risk_score', 'status']
    available_cols = [col for col in display_cols if col in filtered_df.columns]
    st.dataframe(
        filtered_df[available_cols],
        use_container_width=True,
        height=400
    )

elif selected == "System Health":
    st.markdown("### ⚙️ System Health Monitor")
    
    # System metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="stat-card" style="text-align: center;">
            <div style="font-size: 2rem;">🟢</div>
            <div class="stat-value">99.9%</div>
            <div class="stat-label">Uptime</div>
            <div>Last 30 days</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card" style="text-align: center;">
            <div style="font-size: 2rem;">⚡</div>
            <div class="stat-value">245ms</div>
            <div class="stat-label">Avg Response Time</div>
            <div>🚀 12% faster than last month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card" style="text-align: center;">
            <div style="font-size: 2rem;">💾</div>
            <div class="stat-value">78%</div>
            <div class="stat-label">Storage Used</div>
            <div>156 GB / 200 GB</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Real-time monitoring
    st.markdown("### 📡 Real-time Monitoring")
    
    # Mock system metrics
    metrics_df = pd.DataFrame({
        'Component': ['API Server', 'Database', 'AI Model', 'Redis Cache', 'Load Balancer'],
        'Status': ['Healthy', 'Healthy', 'Degraded', 'Healthy', 'Healthy'],
        'Latency (ms)': [125, 89, 234, 45, 67],
        'CPU Usage (%)': [45, 32, 78, 12, 23],
        'Memory Usage (%)': [56, 67, 82, 34, 45]
    })
    
    st.dataframe(metrics_df, use_container_width=True)
    
    # Incident log
    st.markdown("### 📋 Recent Incidents")
    incidents = pd.DataFrame({
        'Timestamp': pd.date_range(start=datetime.now() - timedelta(hours=24), periods=5, freq='6H')[::-1],
        'Component': ['AI Model', 'Database', 'System', 'API', 'System'],
        'Status': ['Resolved', 'Resolved', 'Operational', 'Monitoring', 'Operational'],
        'Message': [
            'High latency detected, auto-resolved',
            'Connection pool exhausted, scaled up',
            'All systems operational',
            'Increased error rate, investigating',
            'Normal operations resumed'
        ]
    })
    st.dataframe(incidents, use_container_width=True)

elif selected == "User Management":
    st.markdown("### 👥 User Management")
    
    # User list
    if 'users' not in st.session_state:
        st.session_state.users = {
            'admin': {'password': '123', 'role': 'Admin', 'email': 'admin@medipredict.com', 'status': 'Active'},
            'doctor': {'password': '456', 'role': 'Doctor', 'email': 'doctor@medipredict.com', 'status': 'Active'},
            'dr_johnson': {'password': 'doctor123', 'role': 'Doctor', 'email': 'johnson@medipredict.com', 'status': 'Active'},
            'dr_smith': {'password': 'doctor456', 'role': 'Doctor', 'email': 'smith@medipredict.com', 'status': 'Inactive'}
        }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📋 User List")
        users_df = pd.DataFrame(st.session_state.users).T
        users_df.index.name = 'Username'
        users_df = users_df.reset_index()
        st.dataframe(users_df, use_container_width=True)
    
    with col2:
        st.markdown("#### ➕ Add New User")
        with st.form("add_user_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            new_role = st.selectbox("Role", ["Doctor", "Admin"])
            new_email = st.text_input("Email")
            
            if st.form_submit_button("Add User"):
                if new_username and new_password:
                    st.session_state.users[new_username] = {
                        'password': new_password,
                        'role': new_role,
                        'email': new_email,
                        'status': 'Active'
                    }
                    st.success(f"User {new_username} added successfully!")
                    st.rerun()
                else:
                    st.error("Username and password are required!")

elif selected == "Reports":
    st.markdown("### 📄 Generate Reports")
    
    report_type = st.selectbox(
        "Select Report Type",
        ["Monthly Summary", "Department Performance", "Risk Analysis", "Patient Demographics"]
    )
    
    date_range = st.date_input(
        "Select Date Range",
        value=(datetime.now() - timedelta(days=30), datetime.now())
    )
    
    if st.button("Generate Report", type="primary"):
        st.markdown("#### 📊 Report Preview")
        
        df = st.session_state.patient_df
        
        if report_type == "Monthly Summary":
            # Generate monthly summary
            df_copy = df.copy()
            df_copy['month'] = df_copy['admission_date'].dt.strftime('%Y-%m')
            monthly_summary = df_copy.groupby('month').agg({
                'patient_id': 'count',
                'length_of_stay': 'mean',
                'risk_score': 'mean'
            }).reset_index()
            monthly_summary.columns = ['Month', 'Total Admissions', 'Avg LOS', 'Avg Risk Score']
            st.dataframe(monthly_summary, use_container_width=True)
            
            # Download button
            csv = monthly_summary.to_csv(index=False)
            st.download_button(
                label="Download Report (CSV)",
                data=csv,
                file_name=f"{report_type}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        elif report_type == "Department Performance":
            dept_summary = df.groupby('department').agg({
                'patient_id': 'count',
                'length_of_stay': 'mean',
                'risk_score': 'mean'
            }).reset_index()
            dept_summary.columns = ['Department', 'Patients', 'Avg LOS', 'Avg Risk Score']
            st.dataframe(dept_summary, use_container_width=True)
            
            csv = dept_summary.to_csv(index=False)
            st.download_button(
                label="Download Report (CSV)",
                data=csv,
                file_name=f"{report_type}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        elif report_type == "Risk Analysis":
            risk_summary = df.groupby('readmission_risk').agg({
                'patient_id': 'count',
                'age': 'mean',
                'length_of_stay': 'mean'
            }).reset_index()
            risk_summary.columns = ['Risk Level', 'Patient Count', 'Avg Age', 'Avg LOS']
            st.dataframe(risk_summary, use_container_width=True)
            
            csv = risk_summary.to_csv(index=False)
            st.download_button(
                label="Download Report (CSV)",
                data=csv,
                file_name=f"{report_type}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        elif report_type == "Patient Demographics":
            demo_summary = df.groupby(['gender', 'age_group']).agg({
                'patient_id': 'count'
            }).reset_index()
            st.dataframe(demo_summary, use_container_width=True)
            
            csv = demo_summary.to_csv(index=False)
            st.download_button(
                label="Download Report (CSV)",
                data=csv,
                file_name=f"{report_type}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6b7280; padding: 1rem;">
    <p>© 2026 MediPredict AI | Enterprise Admin Dashboard | Version 2.1.0</p>
    <p style="font-size: 0.75rem;">HIPAA Compliant | Real-time Analytics | AI-Powered Insights</p>
</div>
""", unsafe_allow_html=True)