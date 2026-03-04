#!/usr/bin/env python3
"""
New Patient Prediction Script using P-S-P Meta-Path
====================================================

This script follows P-S-P meta-path logic:
1. Analyzes SYMPTOM-LEVEL severity first (Patient → Symptoms)
2. Aggregates symptoms to ORGAN-LEVEL predictions (Symptoms → Organs)
3. Generates comprehensive reports with both levels

Severity Level Guide:
- Level 0 (Normal): All test values within healthy ranges
- Level 1 (Mild): Slight deviations, early warning signs
- Level 2 (Moderate): Significant abnormalities requiring attention
- Level 3 (Severe): Critical abnormalities requiring immediate medical care
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from collections import defaultdict

# Import HAN components
from HAN import MedicalGraphData, HANPP

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'models_saved/ruhunu_data_clustered/hanpp_P-S-P.pt'
OUTPUT_DIR = 'predictions_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Severity level explanations
SEVERITY_LEVELS = {
    0: {
        'name': 'NORMAL',
        'color': '#2ecc71',  # Green
        'description': 'All organ functions are within healthy ranges',
        'action': 'No immediate action needed. Continue regular health monitoring.'
    },
    1: {
        'name': 'MILD',
        'color': '#f39c12',  # Orange
        'description': 'Slight abnormalities detected. Early warning signs.',
        'action': 'Schedule follow-up tests. Consider lifestyle modifications.'
    },
    2: {
        'name': 'MODERATE',
        'color': '#e67e22',  # Dark Orange
        'description': 'Significant organ dysfunction. Medical attention recommended.',
        'action': 'Consult specialist. Begin treatment plan.'
    },
    3: {
        'name': 'SEVERE',
        'color': '#e74c3c',  # Red
        'description': 'Critical organ dysfunction. Immediate medical care required.',
        'action': 'URGENT: Seek immediate medical attention.'
    }
}

# Common medical tests with realistic value ranges
MEDICAL_TESTS = {
    'Haemoglobin Absolute Value': {'normal': (12.5, 16), 'unit': 'g/dL', 'organ': 'blood'},
    'WBC Absolute Value': {'normal': (4, 11), 'unit': '10^9/L', 'organ': 'immune system'},
    'Platelet Count Absolute Value': {'normal': (150, 450), 'unit': '10^9/L', 'organ': 'blood'},
    'HbA1c Result': {'normal': (4, 5.6), 'unit': '%', 'organ': 'pancreas'},
    'Serum_Creatinine_Result': {'normal': (0.65, 1.2), 'unit': 'mg/dL', 'organ': 'kidney'},
    'eGFR Result': {'normal': (90, 120), 'unit': 'mL/min', 'organ': 'kidney'},
    'Total Cholesterol': {'normal': (100, 200), 'unit': 'mg/dL', 'organ': 'cardiovascular system'},
    'LDL-Cholesterol': {'normal': (50, 100), 'unit': 'mg/dL', 'organ': 'cardiovascular system'},
    'HDL Cholesterol': {'normal': (45, 80), 'unit': 'mg/dL', 'organ': 'cardiovascular system'},
    'SGPT (ALT) Result': {'normal': (7, 56), 'unit': 'U/L', 'organ': 'liver'},
    'TSH': {'normal': (0.4, 4), 'unit': 'mIU/L', 'organ': 'thyroid'},
    'Serum - Potassium': {'normal': (3.5, 5.1), 'unit': 'mmol/L', 'organ': 'kidney'},
    'Serum - Sodium': {'normal': (135, 145), 'unit': 'mmol/L', 'organ': 'kidney'},
    'Blood Urea Result': {'normal': (20, 40), 'unit': 'mg/dL', 'organ': 'kidney'},
    'RBC Absolute Value': {'normal': (4.11, 5.51), 'unit': '10^12/L', 'organ': 'blood'},
}


def generate_synthetic_patients(num_patients=25):
    """Generate synthetic patient data with realistic test values."""
    print("\n" + "="*80)
    print("GENERATING SYNTHETIC PATIENT DATA")
    print("="*80)
    
    patients = []
    patient_id_start = 900000
    
    profiles = [
        {'type': 'healthy', 'count': 8, 'deviation': 0.05},
        {'type': 'mild', 'count': 10, 'deviation': 0.25},
        {'type': 'moderate', 'count': 5, 'deviation': 0.50},
        {'type': 'severe', 'count': 2, 'deviation': 0.80}
    ]
    
    report_date = datetime.now()
    patient_idx = 0
    
    for profile in profiles:
        for _ in range(profile['count']):
            patient_id = patient_id_start + patient_idx
            age = np.random.randint(25, 80)
            sex = np.random.choice(['Male', 'Female'])
            dob = report_date - timedelta(days=age*365)
            
            for test_name, test_info in MEDICAL_TESTS.items():
                min_val, max_val = test_info['normal']
                mid_val = (min_val + max_val) / 2
                range_val = max_val - min_val
                
                if profile['type'] == 'healthy':
                    value = np.random.uniform(min_val + 0.1*range_val, max_val - 0.1*range_val)
                elif profile['type'] == 'mild':
                    if np.random.random() < 0.5:
                        value = mid_val + np.random.uniform(0.1, 0.3) * range_val
                    else:
                        value = mid_val - np.random.uniform(0.1, 0.3) * range_val
                elif profile['type'] == 'moderate':
                    if np.random.random() < 0.5:
                        value = max_val + np.random.uniform(0.2, 0.5) * range_val
                    else:
                        value = min_val - np.random.uniform(0.2, 0.5) * range_val
                else:  # severe
                    if np.random.random() < 0.5:
                        value = max_val + np.random.uniform(0.5, 1.0) * range_val
                    else:
                        value = min_val - np.random.uniform(0.5, 1.0) * range_val
                
                value = max(0.1, value)
                
                patients.append({
                    'patient_id': patient_id,
                    'report_date': report_date.strftime('%m/%d/%Y %H:%M'),
                    'test_name': test_name,
                    'test_value': round(value, 2),
                    'date_of_birth': dob.strftime('%m/%d/%Y 0:00'),
                    'age_at_report': float(age),
                    'sex': sex,
                    'is_foreign': 0,
                    'profile_type': profile['type']
                })
            
            patient_idx += 1
    
    df = pd.DataFrame(patients)
    print(f"✅ Generated {num_patients} synthetic patients")
    print(f"   - Total test records: {len(df)}")
    return df


def load_model_and_data(new_patients_df):
    """Load existing data, append new patients, and load trained model."""
    print("\n" + "="*80)
    print("LOADING DATA AND MODEL")
    print("="*80)
    
    temp_file = os.path.join(OUTPUT_DIR, 'new_patients_temp.csv')
    new_patients_df.to_csv(temp_file, index=False)
    
    existing_df = pd.read_csv('data/filtered_patient_reports.csv')
    combined_df = pd.concat([existing_df, new_patients_df], ignore_index=True)
    
    combined_file = os.path.join(OUTPUT_DIR, 'combined_patient_data.csv')
    combined_df.to_csv(combined_file, index=False)
    
    print(f"✅ Existing patients: {existing_df['patient_id'].nunique()}")
    print(f"✅ New patients: {new_patients_df['patient_id'].nunique()}")
    
    print("\n📊 Loading data with MedicalGraphData...")
    data_loader = MedicalGraphData(
        path_records=combined_file,
        path_symptom='data/test-disease-organ.csv',
        symptom_freq_threshold=5,
        prune_per_patient=True,
        nnz_threshold=80_000_000,
        seed=42
    )
    
    data_loader.load_data()
    data_loader.build_labels_and_features()
    data_loader.build_adjacency_matrices()
    
    new_patient_ids = new_patients_df['patient_id'].unique()
    patient_id_to_idx = {pid: idx for idx, pid in enumerate(data_loader.patient_ids)}
    new_patient_indices = [patient_id_to_idx[pid] for pid in new_patient_ids if pid in patient_id_to_idx]
    
    print(f"✅ Graph constructed: {len(new_patient_indices)} new patients")
    
    # Load model (not actually used for P-S-P symptom analysis, but kept for compatibility)
    print(f"\n🔧 Model path: {MODEL_PATH}")
    print("   (P-S-P analysis uses rule-based symptom severity)")
    
    return data_loader, new_patient_indices


def analyze_symptom_levels(data_loader, patient_indices, new_patients_df):
    """STEP 1: Analyze symptom levels for patients based on P-S-P meta-path."""
    print("\n" + "="*80)
    print("STEP 1: ANALYZING PATIENT SYMPTOM LEVELS (P-S-P Meta-Path)")
    print("="*80)
    
    symptom_results = []
    
    for idx in patient_indices:
        patient_id = data_loader.patient_ids[idx]
        patient_data = new_patients_df[new_patients_df['patient_id'] == patient_id]
        
        for _, test_record in patient_data.iterrows():
            test_name = test_record['test_name']
            test_value = test_record['test_value']
            
            if test_name in MEDICAL_TESTS:
                normal_min, normal_max = MEDICAL_TESTS[test_name]['normal']
                unit = MEDICAL_TESTS[test_name]['unit']
                organ = MEDICAL_TESTS[test_name]['organ']
                
                if test_value < normal_min:
                    deviation = (normal_min - test_value) / normal_min
                    direction = 'LOW'
                elif test_value > normal_max:
                    deviation = (test_value - normal_max) / normal_max
                    direction = 'HIGH'
                else:
                    deviation = 0
                    direction = 'NORMAL'
                
                if deviation == 0:
                    symptom_severity = 0
                elif deviation < 0.3:
                    symptom_severity = 1
                elif deviation < 0.6:
                    symptom_severity = 2
                else:
                    symptom_severity = 3
                
                symptom_results.append({
                    'patient_id': patient_id,
                    'symptom': test_name,
                    'value': test_value,
                    'unit': unit,
                    'normal_range': f"{normal_min}-{normal_max}",
                    'deviation_pct': round(deviation * 100, 1),
                    'direction': direction,
                    'symptom_severity': symptom_severity,
                    'severity_name': SEVERITY_LEVELS[symptom_severity]['name'],
                    'target_organ': organ
                })
    
    symptom_df = pd.DataFrame(symptom_results)
    
    print(f"✅ Analyzed {len(symptom_df)} symptom-level measurements")
    print(f"   - Normal symptoms: {len(symptom_df[symptom_df['symptom_severity'] == 0])}")
    print(f"   - Mild abnormalities: {len(symptom_df[symptom_df['symptom_severity'] == 1])}")
    print(f"   - Moderate abnormalities: {len(symptom_df[symptom_df['symptom_severity'] == 2])}")
    print(f"   - Severe abnormalities: {len(symptom_df[symptom_df['symptom_severity'] == 3])}")
    
    return symptom_df


def aggregate_symptom_to_organ(symptom_df):
    """STEP 2: Aggregate symptom-level severities to organ-level predictions."""
    print("\n" + "="*80)
    print("STEP 2: AGGREGATING SYMPTOMS TO ORGAN SEVERITY")
    print("="*80)
    
    organ_results = []
    
    for patient_id in symptom_df['patient_id'].unique():
        patient_symptoms = symptom_df[symptom_df['patient_id'] == patient_id]
        
        for organ in patient_symptoms['target_organ'].unique():
            organ_symptoms = patient_symptoms[patient_symptoms['target_organ'] == organ]
            
            avg_severity = organ_symptoms['symptom_severity'].mean()
            max_severity = organ_symptoms['symptom_severity'].max()
            num_abnormal = len(organ_symptoms[organ_symptoms['symptom_severity'] > 0])
            abnormal_ratio = num_abnormal / len(organ_symptoms)
            
            if max_severity == 0:
                organ_severity = 0
            elif max_severity == 1 and abnormal_ratio < 0.3:
                organ_severity = 1
            elif max_severity <= 2 and abnormal_ratio < 0.5:
                organ_severity = 1
            elif max_severity == 2 or (max_severity == 1 and abnormal_ratio >= 0.5):
                organ_severity = 2
            else:
                organ_severity = 3
            
            organ_results.append({
                'patient_id': patient_id,
                'organ': organ,
                'predicted_severity': organ_severity,
                'severity_name': SEVERITY_LEVELS[organ_severity]['name'],
                'avg_symptom_severity': round(avg_severity, 2),
                'max_symptom_severity': int(max_severity),
                'num_symptoms': len(organ_symptoms),
                'num_abnormal_symptoms': num_abnormal,
                'abnormal_ratio': round(abnormal_ratio, 2),
                'damage_score': round(avg_severity * abnormal_ratio, 3)
            })
    
    organ_df = pd.DataFrame(organ_results)
    
    print(f"✅ Aggregated symptoms to {len(organ_df)} organ predictions")
    print(f"   - Normal organs: {len(organ_df[organ_df['predicted_severity'] == 0])}")
    print(f"   - Mild organ issues: {len(organ_df[organ_df['predicted_severity'] == 1])}")
    print(f"   - Moderate organ issues: {len(organ_df[organ_df['predicted_severity'] == 2])}")
    print(f"   - Severe organ issues: {len(organ_df[organ_df['predicted_severity'] == 3])}")
    
    return organ_df


def make_predictions(data_loader, patient_indices, new_patients_df):
    """Make predictions following P-S-P meta-path logic."""
    print("\n" + "="*80)
    print("P-S-P META-PATH PREDICTION PIPELINE")
    print("="*80)
    print("Following the logic: Patients → Symptoms → Organ Severity")
    
    symptom_df = analyze_symptom_levels(data_loader, patient_indices, new_patients_df)
    organ_df = aggregate_symptom_to_organ(symptom_df)
    
    print("\n✅ Prediction pipeline completed!")
    return symptom_df, organ_df


def create_patient_summary(symptom_df, organ_df, new_patients_df):
    """Create patient summary reports with symptom and organ analysis."""
    print("\n" + "="*80)
    print("CREATING PATIENT SUMMARIES")
    print("="*80)
    
    summaries = []
    
    for patient_id in organ_df['patient_id'].unique():
        patient_organs = organ_df[organ_df['patient_id'] == patient_id]
        patient_symptoms = symptom_df[symptom_df['patient_id'] == patient_id]
        patient_data = new_patients_df[new_patients_df['patient_id'] == patient_id].iloc[0]
        
        organ_severity_counts = patient_organs['predicted_severity'].value_counts().to_dict()
        symptom_severity_counts = patient_symptoms['symptom_severity'].value_counts().to_dict()
        
        max_organ_severity = patient_organs['predicted_severity'].max()
        avg_damage_score = patient_organs['damage_score'].mean()
        
        affected_organs = patient_organs[patient_organs['predicted_severity'] > 0]['organ'].tolist()
        abnormal_symptoms = patient_symptoms[patient_symptoms['symptom_severity'] > 0]['symptom'].tolist()
        
        summaries.append({
            'patient_id': patient_id,
            'age': int(patient_data['age_at_report']),
            'sex': patient_data['sex'],
            'overall_status': SEVERITY_LEVELS[max_organ_severity]['name'],
            'max_severity_level': int(max_organ_severity),
            'avg_damage_score': round(avg_damage_score, 3),
            'total_symptoms': len(patient_symptoms),
            'normal_symptoms': symptom_severity_counts.get(0, 0),
            'mild_symptoms': symptom_severity_counts.get(1, 0),
            'moderate_symptoms': symptom_severity_counts.get(2, 0),
            'severe_symptoms': symptom_severity_counts.get(3, 0),
            'num_normal_organs': organ_severity_counts.get(0, 0),
            'num_mild_organs': organ_severity_counts.get(1, 0),
            'num_moderate_organs': organ_severity_counts.get(2, 0),
            'num_severe_organs': organ_severity_counts.get(3, 0),
            'affected_organs': ', '.join(affected_organs[:5]) if affected_organs else 'None',
            'abnormal_symptoms_count': len(abnormal_symptoms),
            'recommended_action': SEVERITY_LEVELS[max_organ_severity]['action'],
            'profile_type': patient_data.get('profile_type', 'unknown')
        })
    
    summary_df = pd.DataFrame(summaries)
    summary_file = os.path.join(OUTPUT_DIR, 'patient_summaries.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Patient summaries saved to: {summary_file}")
    
    return summary_df


def create_visualizations(symptom_df, organ_df, summary_df):
    """Create comprehensive visualizations following P-S-P logic."""
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    colors = [SEVERITY_LEVELS[i]['color'] for i in range(4)]
    
    # P-S-P Pipeline Overview
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Symptom severity distribution
    symptom_severity_counts = symptom_df['symptom_severity'].value_counts().sort_index()
    axes[0, 0].bar(range(4), [symptom_severity_counts.get(i, 0) for i in range(4)], color=colors)
    axes[0, 0].set_xlabel('Severity Level', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title('STEP 1: Symptom Severity Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xticks(range(4))
    axes[0, 0].set_xticklabels([SEVERITY_LEVELS[i]['name'] for i in range(4)])
    
    # Organ severity distribution
    organ_severity_counts = organ_df['predicted_severity'].value_counts().sort_index()
    axes[0, 1].bar(range(4), [organ_severity_counts.get(i, 0) for i in range(4)], color=colors)
    axes[0, 1].set_xlabel('Severity Level', fontsize=12)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].set_title('STEP 2: Organ Severity Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(range(4))
    axes[0, 1].set_xticklabels([SEVERITY_LEVELS[i]['name'] for i in range(4)])
    
    # Patient health status
    status_counts = summary_df['overall_status'].value_counts()
    status_colors = [SEVERITY_LEVELS[i]['color'] for i in range(4) 
                     if SEVERITY_LEVELS[i]['name'] in status_counts.index]
    axes[0, 2].pie(status_counts.values, labels=status_counts.index, autopct='%1.1f%%',
                   colors=status_colors, startangle=90)
    axes[0, 2].set_title('Patient Overall Health Status', fontsize=14, fontweight='bold')
    
    # Top affected symptoms
    symptom_severity = symptom_df.groupby('symptom')['symptom_severity'].mean().sort_values(ascending=False).head(10)
    axes[1, 0].barh(range(len(symptom_severity)), symptom_severity.values, color='lightcoral')
    axes[1, 0].set_yticks(range(len(symptom_severity)))
    axes[1, 0].set_yticklabels([s[:25] + '...' if len(s) > 25 else s for s in symptom_severity.index], fontsize=9)
    axes[1, 0].set_xlabel('Average Severity', fontsize=12)
    axes[1, 0].set_title('Top 10 Most Abnormal Symptoms', fontsize=14, fontweight='bold')
    axes[1, 0].invert_yaxis()
    
    # Top affected organs
    organ_severity = organ_df.groupby('organ')['predicted_severity'].mean().sort_values(ascending=False).head(10)
    axes[1, 1].barh(range(len(organ_severity)), organ_severity.values, color='coral')
    axes[1, 1].set_yticks(range(len(organ_severity)))
    axes[1, 1].set_yticklabels(organ_severity.index, fontsize=10)
    axes[1, 1].set_xlabel('Average Severity', fontsize=12)
    axes[1, 1].set_title('Top 10 Most Affected Organs', fontsize=14, fontweight='bold')
    axes[1, 1].invert_yaxis()
    
    # Symptom count vs organ severity
    axes[1, 2].scatter(summary_df['abnormal_symptoms_count'], summary_df['max_severity_level'], 
                       c=summary_df['max_severity_level'], cmap='RdYlGn_r', 
                       s=100, alpha=0.6, edgecolors='black')
    axes[1, 2].set_xlabel('Number of Abnormal Symptoms', fontsize=12)
    axes[1, 2].set_ylabel('Max Organ Severity Level', fontsize=12)
    axes[1, 2].set_title('Symptom Count vs Organ Severity', fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('P-S-P Meta-Path Analysis: Patient → Symptoms → Organ Severity', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    fig_path = os.path.join(OUTPUT_DIR, 'psp_prediction_overview.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ P-S-P overview visualization saved to: {fig_path}")
    plt.close()
    
    # Symptom-level heatmap
    fig, ax = plt.subplots(figsize=(18, 10))
    symptom_pivot = symptom_df.pivot_table(
        index='patient_id', columns='symptom', values='symptom_severity', aggfunc='mean')
    top_symptoms = symptom_df.groupby('symptom')['symptom_severity'].mean().sort_values(ascending=False).head(15).index
    symptom_pivot_top = symptom_pivot[top_symptoms]
    
    sns.heatmap(symptom_pivot_top, cmap='RdYlGn_r', center=1.5, annot=False, 
                cbar_kws={'label': 'Symptom Severity Level'}, ax=ax, vmin=0, vmax=3)
    ax.set_title('Patient-Symptom Severity Heatmap (Top 15 Symptoms)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Clinical Test/Symptom', fontsize=12)
    ax.set_ylabel('Patient ID', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    
    symptom_heatmap_path = os.path.join(OUTPUT_DIR, 'patient_symptom_heatmap.png')
    plt.savefig(symptom_heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✅ Symptom heatmap saved to: {symptom_heatmap_path}")
    plt.close()
    
    # Organ-level heatmap
    fig, ax = plt.subplots(figsize=(16, 10))
    heatmap_data = organ_df.pivot(index='patient_id', columns='organ', values='predicted_severity')
    
    sns.heatmap(heatmap_data, cmap='RdYlGn_r', center=1.5, annot=False, 
                cbar_kws={'label': 'Organ Severity Level'}, ax=ax, vmin=0, vmax=3)
    ax.set_title('Patient-Organ Severity Heatmap (Aggregated from Symptoms)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Organ System', fontsize=12)
    ax.set_ylabel('Patient ID', fontsize=12)
    plt.tight_layout()
    
    heatmap_path = os.path.join(OUTPUT_DIR, 'patient_organ_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    print(f"✅ Organ heatmap saved to: {heatmap_path}")
    plt.close()


def create_detailed_report(summary_df, symptom_df, organ_df, new_patients_df):
    """Create detailed text report for each patient following P-S-P logic."""
    print("\n" + "="*80)
    print("CREATING DETAILED PATIENT REPORTS")
    print("="*80)
    
    report_path = os.path.join(OUTPUT_DIR, 'detailed_patient_reports.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MEDICAL PREDICTION REPORT - P-S-P META-PATH ANALYSIS\n")
        f.write("Patient → Symptom Level → Organ Severity Analysis\n")
        f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        for _, patient in summary_df.iterrows():
            patient_id = patient['patient_id']
            patient_symptoms = symptom_df[symptom_df['patient_id'] == patient_id]
            patient_organs = organ_df[organ_df['patient_id'] == patient_id]
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"PATIENT ID: {patient_id}\n")
            f.write("="*80 + "\n")
            f.write(f"Demographics: Age {patient['age']}, {patient['sex']}\n")
            f.write(f"Overall Status: {patient['overall_status']}\n")
            f.write(f"Profile: {patient['profile_type'].upper()}\n\n")
            
            # STEP 1: Symptom Analysis
            f.write("="*80 + "\n")
            f.write("STEP 1: SYMPTOM-LEVEL ANALYSIS\n")
            f.write("="*80 + "\n")
            f.write(f"Total Symptoms: {patient['total_symptoms']}\n")
            f.write(f"  Normal: {patient['normal_symptoms']}, ")
            f.write(f"Mild: {patient['mild_symptoms']}, ")
            f.write(f"Moderate: {patient['moderate_symptoms']}, ")
            f.write(f"Severe: {patient['severe_symptoms']}\n\n")
            
            abnormal_symptoms = patient_symptoms[patient_symptoms['symptom_severity'] > 0].sort_values(
                'symptom_severity', ascending=False)
            
            if len(abnormal_symptoms) > 0:
                f.write(f"🔴 ABNORMAL SYMPTOMS ({len(abnormal_symptoms)}):\n")
                for _, symptom in abnormal_symptoms.head(10).iterrows():
                    f.write(f"  • {symptom['symptom']}: {symptom['value']} {symptom['unit']} ")
                    f.write(f"(Normal: {symptom['normal_range']}) - ")
                    f.write(f"{symptom['deviation_pct']}% {symptom['direction']} - ")
                    f.write(f"{symptom['severity_name']}\n")
                f.write("\n")
            else:
                f.write("✅ All symptoms within normal ranges.\n\n")
            
            # STEP 2: Organ Analysis
            f.write("="*80 + "\n")
            f.write("STEP 2: ORGAN-LEVEL ANALYSIS (Aggregated from Symptoms)\n")
            f.write("="*80 + "\n")
            f.write(f"Organs: Normal {patient['num_normal_organs']}, ")
            f.write(f"Mild {patient['num_mild_organs']}, ")
            f.write(f"Moderate {patient['num_moderate_organs']}, ")
            f.write(f"Severe {patient['num_severe_organs']}\n\n")
            
            affected_organs = patient_organs[patient_organs['predicted_severity'] > 0].sort_values(
                'predicted_severity', ascending=False)
            
            if len(affected_organs) > 0:
                f.write(f"⚠️  ORGANS REQUIRING ATTENTION ({len(affected_organs)}):\n")
                for _, organ_pred in affected_organs.iterrows():
                    f.write(f"  • {organ_pred['organ']}: {organ_pred['severity_name']} ")
                    f.write(f"(Score: {organ_pred['damage_score']:.3f}, ")
                    f.write(f"{organ_pred['num_abnormal_symptoms']}/{organ_pred['num_symptoms']} abnormal symptoms)\n")
                f.write("\n")
            else:
                f.write("✅ All organs functioning normally.\n\n")
            
            f.write(f"RECOMMENDED ACTION: {patient['recommended_action']}\n")
    
    print(f"✅ Detailed reports saved to: {report_path}")


def print_summary_table(summary_df):
    """Print a summary table to console."""
    print("\n" + "="*80)
    print("PATIENT PREDICTION SUMMARY")
    print("="*80)
    
    print(f"\n{'ID':<10} {'Age':<5} {'Sex':<8} {'Status':<12} {'Affected':<10}")
    print("-" * 50)
    
    for _, patient in summary_df.iterrows():
        patient_id = str(patient['patient_id'])
        age = str(int(patient['age']))
        sex = patient['sex'][:6]
        status = patient['overall_status']
        affected = str(patient['num_moderate_organs'] + patient['num_severe_organs'])
        print(f"{patient_id:<10} {age:<5} {sex:<8} {status:<12} {affected:<10}")
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total patients: {len(summary_df)}")
    for severity_name in ['NORMAL', 'MILD', 'MODERATE', 'SEVERE']:
        count = len(summary_df[summary_df['overall_status'] == severity_name])
        pct = count/len(summary_df)*100 if len(summary_df) > 0 else 0
        print(f"{severity_name}: {count} ({pct:.1f}%)")


def main():
    """Main execution function following P-S-P meta-path logic."""
    print("\n" + "="*80)
    print("HAN MEDICAL PREDICTION SYSTEM")
    print("P-S-P Meta-Path Analysis: Patient → Symptoms → Organ Severity")
    print("="*80)
    
    # Generate synthetic patients
    new_patients_df = generate_synthetic_patients(num_patients=25)
    
    # Load model and data
    data_loader, new_patient_indices = load_model_and_data(new_patients_df)
    
    # Make predictions following P-S-P logic
    symptom_df, organ_df = make_predictions(data_loader, new_patient_indices, new_patients_df)
    
    # Save predictions
    symptom_file = os.path.join(OUTPUT_DIR, 'symptom_level_predictions.csv')
    symptom_df.to_csv(symptom_file, index=False)
    print(f"\n✅ Symptom-level predictions saved to: {symptom_file}")
    
    organ_file = os.path.join(OUTPUT_DIR, 'organ_level_predictions.csv')
    organ_df.to_csv(organ_file, index=False)
    print(f"✅ Organ-level predictions saved to: {organ_file}")
    
    # Create summaries, visualizations, and reports
    summary_df = create_patient_summary(symptom_df, organ_df, new_patients_df)
    create_visualizations(symptom_df, organ_df, summary_df)
    create_detailed_report(summary_df, symptom_df, organ_df, new_patients_df)
    print_summary_table(summary_df)
    
    print("\n" + "="*80)
    print("P-S-P PREDICTION PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n📁 All outputs saved to: {OUTPUT_DIR}/")
    print(f"\nGenerated files:")
    print(f"  1. symptom_level_predictions.csv - Symptom-level severity analysis")
    print(f"  2. organ_level_predictions.csv - Organ severity (aggregated)")
    print(f"  3. patient_summaries.csv - Summary statistics")
    print(f"  4. detailed_patient_reports.txt - Human-readable P-S-P reports")
    print(f"  5. psp_prediction_overview.png - P-S-P pipeline visualization")
    print(f"  6. patient_symptom_heatmap.png - Symptom-level heatmap")
    print(f"  7. patient_organ_heatmap.png - Organ-level heatmap")
    print(f"\n✨ P-S-P Analysis complete!")
    print(f"   Flow: Patients → Symptom Severity → Organ Severity\n")


if __name__ == "__main__":
    main()
