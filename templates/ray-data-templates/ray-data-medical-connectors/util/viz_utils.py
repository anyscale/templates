"""Visualization utilities for medical data processing templates."""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_medical_dashboard(patient_df):
    """Create medical analytics dashboard."""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Age Distribution', 'Diagnosis', 'Outcomes'),
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "pie"}]]
    )
    
    # Age distribution
    if 'age' in patient_df.columns:
        age_groups = pd.cut(patient_df['age'], bins=[0, 18, 35, 50, 65, 100],
                           labels=['<18', '18-35', '36-50', '51-65', '65+'])
        age_counts = age_groups.value_counts().reset_index()
        fig.add_trace(
            go.Bar(x=age_counts['age'], y=age_counts['count'],
                  marker_color='lightblue', name="Age"),
            row=1, col=1
        )
    
    # Top diagnoses
    if 'diagnosis' in patient_df.columns:
        diagnosis_counts = patient_df['diagnosis'].value_counts().head(8).reset_index()
        fig.add_trace(
            go.Bar(x=diagnosis_counts['diagnosis'], y=diagnosis_counts['count'],
                  marker_color='lightcoral', name="Diagnosis"),
            row=1, col=2
        )
    
    # Treatment outcomes
    if 'treatment_outcome' in patient_df.columns:
        fig.add_trace(
            go.Pie(labels=patient_df['treatment_outcome'].value_counts().index,
                  values=patient_df['treatment_outcome'].value_counts().values,
                  name="Outcomes"),
            row=1, col=3
        )
    
    fig.update_layout(height=500, showlegend=False,
                     title_text="Medical Analytics Dashboard")
    
    return fig


def create_imaging_dashboard(imaging_df):
    """Create medical imaging analytics dashboard."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Imaging Modality Distribution', 'Studies by Institution'),
        specs=[[{"type": "pie"}, {"type": "bar"}]]
    )
    
    # Modality distribution
    if 'modality' in imaging_df.columns:
        fig.add_trace(
            go.Pie(labels=imaging_df['modality'].value_counts().index,
                  values=imaging_df['modality'].value_counts().values,
                  name="Modality"),
            row=1, col=1
        )
    
    # Studies by institution
    if 'institution' in imaging_df.columns:
        institution_counts = imaging_df['institution'].value_counts().head(10).reset_index()
        fig.add_trace(
            go.Bar(x=institution_counts['institution'], y=institution_counts['count'],
                  marker_color='lightgreen', name="Studies"),
            row=1, col=2
        )
    
    fig.update_layout(height=500, title_text="Medical Imaging Analytics")
    
    return fig


def create_patient_timeline(patient_df):
    """Create patient visit timeline visualization."""
    if 'visit_date' in patient_df.columns and 'patient_id' in patient_df.columns:
        # Get top patients with most visits
        top_patients = patient_df['patient_id'].value_counts().head(10).index
        timeline_df = patient_df[patient_df['patient_id'].isin(top_patients)]
        
        fig = px.timeline(timeline_df, x_start='visit_date', x_end='discharge_date',
                         y='patient_id', color='diagnosis',
                         title='Patient Visit Timeline (Top 10 Patients)')
        fig.update_layout(height=500)
        return fig
    return None


def create_lab_results_trends(lab_df):
    """Create laboratory results trends visualization."""
    if 'test_date' in lab_df.columns and 'test_value' in lab_df.columns:
        fig = px.line(lab_df, x='test_date', y='test_value', color='test_type',
                     title='Laboratory Results Trends',
                     labels={'test_value': 'Test Value', 'test_date': 'Date'})
        fig.update_layout(height=500)
        return fig
    return None


def create_medical_analytics_dashboard():
    """Generate healthcare analytics dashboard with multiple insights."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Healthcare Analytics Dashboard - Ray Data Processing', fontsize=16, fontweight='bold')
    
    # 1. Patient Age Distribution by Department
    departments = ['CARDIOLOGY', 'EMERGENCY', 'ORTHOPEDICS', 'NEUROLOGY', 'ONCOLOGY', 'PEDIATRICS']
    age_data = {
        'CARDIOLOGY': np.random.normal(65, 15, 1000),
        'EMERGENCY': np.random.normal(45, 20, 1500),
        'ORTHOPEDICS': np.random.normal(55, 18, 800),
        'NEUROLOGY': np.random.normal(60, 16, 600),
        'ONCOLOGY': np.random.normal(58, 14, 700),
        'PEDIATRICS': np.random.normal(8, 5, 500)
    }
    
    ax1 = axes[0, 0]
    for dept, ages in age_data.items():
        ages = np.clip(ages, 0, 100)
        ax1.hist(ages, alpha=0.6, label=dept, bins=20)
    ax1.set_title('Patient Age Distribution by Department', fontweight='bold')
    ax1.set_xlabel('Patient Age')
    ax1.set_ylabel('Number of Patients')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Lab Result Processing Volume
    ax2 = axes[0, 1]
    lab_tests = ['Glucose', 'Cholesterol', 'Blood Count', 'Liver Panel', 'Kidney Panel', 'Thyroid']
    daily_volumes = [2500, 1800, 3200, 1200, 1100, 900]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']
    
    bars = ax2.bar(lab_tests, daily_volumes, color=colors)
    ax2.set_title('Daily Lab Test Processing Volume', fontweight='bold')
    ax2.set_ylabel('Tests Processed')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, volume in zip(bars, daily_volumes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{volume:,}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Critical Alert Distribution
    ax3 = axes[0, 2]
    alert_types = ['Blood Pressure', 'Heart Rate', 'Glucose Level', 'Oxygen Saturation', 'Temperature']
    alert_counts = [145, 89, 156, 67, 23]
    severity_colors = ['#FF4757', '#FF6348', '#FFA502', '#F0932B', '#6C5CE7']
    
    wedges, texts, autotexts = ax3.pie(alert_counts, labels=alert_types, autopct='%1.1f%%',
                                      colors=severity_colors, startangle=90)
    ax3.set_title('Critical Alert Distribution (Last 24 Hours)', fontweight='bold')
    
    # 4. DICOM Image Processing Performance
    ax4 = axes[1, 0]
    modalities = ['CT', 'MRI', 'X-Ray', 'Ultrasound', 'Mammography']
    processing_times = [3.2, 8.5, 1.1, 2.3, 4.7]
    throughput = [450, 180, 800, 650, 320]
    
    ax4_twin = ax4.twinx()
    line1 = ax4.plot(modalities, processing_times, 'bo-', linewidth=3, markersize=8, label='Processing Time')
    line2 = ax4_twin.plot(modalities, throughput, 'rs-', linewidth=3, markersize=8, label='Throughput')
    
    ax4.set_title('DICOM Processing Performance by Modality', fontweight='bold')
    ax4.set_ylabel('Avg Processing Time (min)', color='blue')
    ax4_twin.set_ylabel('Images/Hour', color='red')
    ax4.tick_params(axis='x', rotation=45)
    
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # 5. Patient Flow Analysis  
    ax5 = axes[1, 1]
    hours = list(range(24))
    admissions = [12, 8, 5, 3, 2, 4, 8, 15, 22, 28, 32, 35, 38, 42, 45, 48, 52, 48, 45, 38, 32, 28, 22, 18]
    discharges = [8, 5, 3, 2, 1, 2, 5, 12, 18, 25, 30, 32, 35, 38, 40, 42, 38, 35, 30, 25, 20, 15, 12, 10]
    
    ax5.fill_between(hours, admissions, alpha=0.6, color='lightcoral', label='Admissions')
    ax5.fill_between(hours, discharges, alpha=0.6, color='lightblue', label='Discharges')
    ax5.plot(hours, admissions, 'r-', linewidth=2)
    ax5.plot(hours, discharges, 'b-', linewidth=2)
    ax5.set_title('24-Hour Patient Flow Pattern', fontweight='bold')
    ax5.set_xlabel('Hour of Day')
    ax5.set_ylabel('Number of Patients')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Data Quality Metrics
    ax6 = axes[1, 2]
    quality_metrics = ['Completeness', 'Accuracy', 'Consistency', 'Timeliness', 'Validity']
    scores = [94.5, 98.2, 91.8, 96.7, 93.4]
    colors = ['#2ECC71' if score >= 95 else '#F39C12' if score >= 90 else '#E74C3C' for score in scores]
    
    bars = ax6.barh(quality_metrics, scores, color=colors)
    ax6.set_title('Medical Data Quality Assessment', fontweight='bold')
    ax6.set_xlabel('Quality Score (%)')
    ax6.set_xlim(0, 100)
    
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax6.text(width + 1, bar.get_y() + bar.get_height()/2,
                f'{score}%', ha='left', va='center', fontweight='bold')
    
    ax6.axvline(x=95, color='green', linestyle='--', alpha=0.7, label='Excellent (95%+)')
    ax6.axvline(x=90, color='orange', linestyle='--', alpha=0.7, label='Good (90%+)')
    ax6.legend()
    
    plt.tight_layout()
    
    return fig

