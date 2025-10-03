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

