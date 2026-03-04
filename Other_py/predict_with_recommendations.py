#!/usr/bin/env python3
"""
Predictive Medical AI Assistant with Test Recommendations
==========================================================

Clinical Workflow:
1. Patient provides initial lab test results
2. Model predicts likely organ issues and severity
3. System recommends additional confirmatory tests
4. Doctor validates predictions and orders recommended tests

This mimics a real AI-assisted diagnostic workflow.
"""

import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from HAN import MedicalGraphData, HANPP

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'models_saved/ruhunu_data_clustered/hanpp_P-S-P.pt'
OUTPUT_DIR = 'predictions_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Severity definitions
SEVERITY_LEVELS = {
    0: {'name': 'NORMAL', 'description': 'No significant abnormalities detected'},
    1: {'name': 'MILD', 'description': 'Minor abnormalities - monitor closely'},
    2: {'name': 'MODERATE', 'description': 'Significant concern - further investigation needed'},
    3: {'name': 'SEVERE', 'description': 'Critical - immediate medical attention required'}
}

# Confirmatory test recommendations by organ system
CONFIRMATORY_TESTS = {
    'kidney': {
        'initial_tests': ['Serum_Creatinine_Result', 'eGFR Result', 'Blood Urea Result'],
        'confirmatory_tests': [
            'Urine Albumin-to-Creatinine Ratio',
            '24-hour Urine Collection',
            'Renal Ultrasound',
            'Cystatin C',
            'Kidney Biopsy (if severe)'
        ],
        'monitoring_tests': ['Serum - Potassium', 'Serum - Sodium', 'Serum Bicarbonate']
    },
    'liver': {
        'initial_tests': ['SGPT (ALT) Result', 'SGOT (AST) Result', 'Bilirubin'],
        'confirmatory_tests': [
            'Liver Function Panel (comprehensive)',
            'Gamma-GT',
            'Alkaline Phosphatase',
            'Prothrombin Time (PT/INR)',
            'Hepatitis Panel',
            'Liver Ultrasound',
            'FibroScan / Liver Elastography'
        ],
        'monitoring_tests': ['Albumin', 'Total Protein']
    },
    'cardiovascular system': {
        'initial_tests': ['Total Cholesterol', 'LDL-Cholesterol', 'HDL Cholesterol', 'Triglycerides'],
        'confirmatory_tests': [
            'Apolipoprotein B',
            'Lipoprotein(a)',
            'hs-CRP (inflammatory marker)',
            'Cardiac Troponin',
            'ECG',
            'Echocardiogram',
            'Coronary Calcium Score CT'
        ],
        'monitoring_tests': ['Blood Pressure', 'Heart Rate Variability']
    },
    'pancreas': {
        'initial_tests': ['HbA1c Result', 'Fasting Glucose'],
        'confirmatory_tests': [
            'Oral Glucose Tolerance Test (OGTT)',
            'C-Peptide',
            'Insulin Level',
            'Fructosamine',
            'Continuous Glucose Monitoring (CGM)',
            'Pancreatic Ultrasound',
            'Anti-GAD Antibodies (Type 1 screening)'
        ],
        'monitoring_tests': ['Postprandial Glucose', 'Urine Ketones']
    },
    'thyroid': {
        'initial_tests': ['TSH'],
        'confirmatory_tests': [
            'Free T4',
            'Free T3',
            'Thyroid Antibodies (TPO, TgAb)',
            'Thyroid Ultrasound',
            'Thyroid Uptake Scan',
            'Fine Needle Aspiration (if nodules)'
        ],
        'monitoring_tests': ['Reverse T3', 'Thyroglobulin']
    },
    'blood': {
        'initial_tests': ['Haemoglobin Absolute Value', 'WBC Absolute Value', 'Platelet Count Absolute Value', 'RBC Absolute Value'],
        'confirmatory_tests': [
            'Complete Blood Count with Differential',
            'Peripheral Blood Smear',
            'Reticulocyte Count',
            'Iron Panel (Iron, Ferritin, TIBC)',
            'Vitamin B12 and Folate',
            'Bone Marrow Biopsy (if severe)',
            'Hemoglobin Electrophoresis'
        ],
        'monitoring_tests': ['ESR', 'Haptoglobin']
    },
    'immune system': {
        'initial_tests': ['WBC Absolute Value', 'Lymphocyte Count'],
        'confirmatory_tests': [
            'Immunoglobulin Panel (IgG, IgA, IgM, IgE)',
            'Complement Levels (C3, C4)',
            'Autoimmune Panel (ANA, RF, Anti-CCP)',
            'T-cell and B-cell Subsets',
            'HIV Test',
            'CMV, EBV serologies'
        ],
        'monitoring_tests': ['Neutrophil Count', 'CD4/CD8 Ratio']
    }
}


def load_trained_model(data_loader):
    """Load the trained HAN model."""
    print(f"\n🔧 Loading trained model: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        return None
    
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    # Infer model dimensions
    in_dim = checkpoint['project.weight'].shape[1]
    hidden_dim = checkpoint['project.weight'].shape[0]
    out_dim = checkpoint['out_proj.weight'].shape[0]
    num_organs = len(data_loader.organs) if hasattr(data_loader, 'organs') else 25
    
    model = HANPP(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        metapath_names=['P-S-P'],
        num_heads=4,
        num_organs=num_organs,
        num_severity=4,
        dropout=0.3
    )
    
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    
    # Handle feature mismatch
    if data_loader.patient_feats.shape[1] != in_dim:
        current_feats = data_loader.patient_feats
        if current_feats.shape[1] < in_dim:
            padding = np.zeros((current_feats.shape[0], in_dim - current_feats.shape[1]))
            data_loader.patient_feats = np.hstack([current_feats, padding])
        else:
            data_loader.patient_feats = current_feats[:, :in_dim]
    
    print(f"✅ Model loaded successfully!")
    print(f"   - Input features: {in_dim}")
    print(f"   - Hidden dimension: {hidden_dim}")
    print(f"   - Output classes: {out_dim}")
    print(f"   - Organ systems: {num_organs}")
    
    return model


def predict_organ_severity(model, data_loader, patient_indices):
    """Use trained model to predict organ severity."""
    print(f"\n📊 Running model inference on {len(patient_indices)} patients...")
    
    # Prepare features
    if isinstance(data_loader.patient_feats, np.ndarray):
        features = torch.from_numpy(data_loader.patient_feats).float().to(DEVICE)
    else:
        features = data_loader.patient_feats.to(DEVICE)
    
    # Build P-S-P meta-path if needed
    if not hasattr(data_loader, 'metapath_matrices') or len(data_loader.metapath_matrices) == 0:
        print("   Building P-S-P meta-path adjacency...")
        data_loader.build_metapaths(['P-S-P'])
    
    neighbor_dicts = data_loader.metapath_matrices
    
    # Make predictions
    with torch.no_grad():
        organ_logits, organ_scores, embeddings, attention_weights = model(features, neighbor_dicts)
        predictions = torch.argmax(organ_logits, dim=2).cpu().numpy()
        scores = organ_scores.cpu().numpy()
        confidences = torch.softmax(organ_logits, dim=2).max(dim=2)[0].cpu().numpy()
    
    # Get organ names
    if hasattr(data_loader, 'organs'):
        organ_names = data_loader.organs
    else:
        organ_names = [f"Organ_{i+1}" for i in range(predictions.shape[1])]
    
    print(f"✅ Predictions completed!")
    
    return predictions, scores, confidences, organ_names


def identify_affected_organs(predictions, scores, confidences, organ_names, patient_idx):
    """Identify organs with predicted abnormalities."""
    affected = []
    
    for organ_idx, organ_name in enumerate(organ_names):
        if organ_idx < predictions.shape[1]:
            severity = int(predictions[patient_idx, organ_idx])
            if severity > 0:  # Any abnormality
                affected.append({
                    'organ': organ_name,
                    'severity': severity,
                    'severity_name': SEVERITY_LEVELS[severity]['name'],
                    'description': SEVERITY_LEVELS[severity]['description'],
                    'damage_score': float(scores[patient_idx, organ_idx]),
                    'confidence': float(confidences[patient_idx, organ_idx])
                })
    
    # Sort by severity (descending)
    affected.sort(key=lambda x: (x['severity'], x['damage_score']), reverse=True)
    
    return affected


def recommend_confirmatory_tests(affected_organs, existing_tests):
    """Recommend additional tests to confirm predicted organ issues."""
    recommendations = []
    
    for organ_info in affected_organs:
        organ = organ_info['organ'].lower()
        severity = organ_info['severity']
        
        # Find matching organ system
        for organ_system, test_info in CONFIRMATORY_TESTS.items():
            if organ_system in organ or organ in organ_system:
                # Identify which initial tests patient already has
                tests_done = [test for test in test_info['initial_tests'] if test in existing_tests]
                tests_missing = [test for test in test_info['initial_tests'] if test not in existing_tests]
                
                # Recommend confirmatory tests
                confirmatory = test_info['confirmatory_tests']
                monitoring = test_info['monitoring_tests']
                
                # Priority based on severity
                if severity >= 3:  # SEVERE
                    priority = 'URGENT'
                    recommended_tests = confirmatory[:4] + monitoring[:2]  # Top tests
                elif severity == 2:  # MODERATE
                    priority = 'HIGH'
                    recommended_tests = confirmatory[:3] + monitoring[:1]
                else:  # MILD
                    priority = 'ROUTINE'
                    recommended_tests = confirmatory[:2]
                
                recommendations.append({
                    'organ': organ_info['organ'],
                    'severity': organ_info['severity_name'],
                    'priority': priority,
                    'initial_tests_done': tests_done,
                    'initial_tests_missing': tests_missing,
                    'recommended_confirmatory': recommended_tests,
                    'rationale': f"To confirm {organ_info['severity_name']} {organ_info['organ']} dysfunction (Confidence: {organ_info['confidence']*100:.1f}%)"
                })
                break
    
    return recommendations


def generate_clinical_report(patient_id, patient_data, affected_organs, recommendations):
    """Generate a clinical report with predictions and recommendations."""
    report = []
    report.append("="*80)
    report.append(f"CLINICAL AI PREDICTION REPORT")
    report.append("="*80)
    report.append(f"Patient ID: {patient_id}")
    report.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Model: HAN++ with P-S-P Meta-Path")
    report.append("")
    
    # Patient demographics
    if 'age_at_report' in patient_data.columns:
        age = int(patient_data.iloc[0]['age_at_report'])
        sex = patient_data.iloc[0]['sex']
        report.append(f"Patient Demographics: {age} years old, {sex}")
        report.append("")
    
    # Available test results
    report.append("="*80)
    report.append("STEP 1: AVAILABLE CLINICAL TEST RESULTS")
    report.append("="*80)
    tests_available = patient_data['test_name'].unique()
    report.append(f"Number of tests performed: {len(tests_available)}")
    report.append("")
    for i, test in enumerate(tests_available[:10], 1):
        test_row = patient_data[patient_data['test_name'] == test].iloc[0]
        report.append(f"  {i:2d}. {test}: {test_row['test_value']}")
    if len(tests_available) > 10:
        report.append(f"  ... and {len(tests_available) - 10} more tests")
    report.append("")
    
    # Model predictions
    report.append("="*80)
    report.append("STEP 2: AI MODEL PREDICTIONS")
    report.append("="*80)
    
    if len(affected_organs) == 0:
        report.append("✅ NO SIGNIFICANT ABNORMALITIES DETECTED")
        report.append("   All organ systems appear to be functioning normally.")
        report.append("   Continue routine health monitoring.")
    else:
        report.append(f"⚠️  {len(affected_organs)} ORGAN SYSTEM(S) SHOW ABNORMALITIES")
        report.append("")
        
        for i, organ_info in enumerate(affected_organs, 1):
            report.append(f"{i}. {organ_info['organ'].upper()}")
            report.append(f"   Predicted Severity: {organ_info['severity_name']} (Level {organ_info['severity']})")
            report.append(f"   Description: {organ_info['description']}")
            report.append(f"   Damage Score: {organ_info['damage_score']:.3f}")
            report.append(f"   Model Confidence: {organ_info['confidence']*100:.1f}%")
            report.append("")
    
    # Recommendations
    if len(recommendations) > 0:
        report.append("="*80)
        report.append("STEP 3: RECOMMENDED CONFIRMATORY TESTS")
        report.append("="*80)
        report.append("To validate AI predictions, consider ordering these additional tests:")
        report.append("")
        
        for i, rec in enumerate(recommendations, 1):
            report.append(f"{i}. {rec['organ']} ({rec['severity']}) - Priority: {rec['priority']}")
            report.append(f"   Rationale: {rec['rationale']}")
            report.append("")
            
            if rec['initial_tests_missing']:
                report.append(f"   ⚠️  Missing basic tests:")
                for test in rec['initial_tests_missing']:
                    report.append(f"      • {test}")
                report.append("")
            
            report.append(f"   📋 Recommended confirmatory tests:")
            for test in rec['recommended_confirmatory']:
                report.append(f"      • {test}")
            report.append("")
    
    # Medical disclaimer
    report.append("="*80)
    report.append("IMPORTANT MEDICAL DISCLAIMER")
    report.append("="*80)
    report.append("This report is generated by an AI system for clinical decision support.")
    report.append("ALL PREDICTIONS MUST BE VALIDATED BY A QUALIFIED PHYSICIAN.")
    report.append("Do not use this report alone for medical diagnosis or treatment decisions.")
    report.append("Recommended tests should be ordered at the discretion of the attending physician.")
    report.append("="*80)
    
    return "\n".join(report)


def main():
    """Main workflow: Predict and recommend tests."""
    print("\n" + "="*80)
    print("PREDICTIVE MEDICAL AI ASSISTANT")
    print("Clinical Workflow: Predict → Validate → Recommend")
    print("="*80)
    
    # Step 1: Load patient data
    print("\n📂 Loading patient data...")
    
    # Use synthetic patients from previous run, or create new ones
    if os.path.exists(os.path.join(OUTPUT_DIR, 'new_patients_temp.csv')):
        new_patients_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'new_patients_temp.csv'))
        print(f"✅ Loaded {new_patients_df['patient_id'].nunique()} patients from previous run")
    else:
        print("⚠️  No patient data found. Run predict_psp_new_patients.py first.")
        return
    
    combined_file = os.path.join(OUTPUT_DIR, 'combined_patient_data.csv')
    if not os.path.exists(combined_file):
        print("⚠️  Combined data not found. Run predict_psp_new_patients.py first.")
        return
    
    # Step 2: Load data with graph structure
    print("\n📊 Building medical knowledge graph...")
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
    
    # Find new patient indices
    new_patient_ids = new_patients_df['patient_id'].unique()
    patient_id_to_idx = {pid: idx for idx, pid in enumerate(data_loader.patient_ids)}
    new_patient_indices = [patient_id_to_idx[pid] for pid in new_patient_ids if pid in patient_id_to_idx]
    
    print(f"✅ Graph constructed: {len(new_patient_indices)} new patients")
    
    # Step 3: Load trained model
    model = load_trained_model(data_loader)
    if model is None:
        return
    
    # Step 4: Make predictions
    predictions, scores, confidences, organ_names = predict_organ_severity(
        model, data_loader, new_patient_indices)
    
    # Step 5: Generate reports with recommendations
    print("\n" + "="*80)
    print("GENERATING CLINICAL REPORTS WITH RECOMMENDATIONS")
    print("="*80)
    
    reports_dir = os.path.join(OUTPUT_DIR, 'clinical_reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    all_recommendations = []
    
    for i, patient_idx in enumerate(new_patient_indices[:5]):  # First 5 patients
        patient_id = data_loader.patient_ids[patient_idx]
        patient_data = new_patients_df[new_patients_df['patient_id'] == patient_id]
        
        # Identify affected organs
        affected_organs = identify_affected_organs(
            predictions, scores, confidences, organ_names, patient_idx)
        
        # Get test recommendations
        existing_tests = patient_data['test_name'].tolist()
        recommendations = recommend_confirmatory_tests(affected_organs, existing_tests)
        
        # Generate clinical report
        report_text = generate_clinical_report(
            patient_id, patient_data, affected_organs, recommendations)
        
        # Save individual report
        report_file = os.path.join(reports_dir, f'patient_{patient_id}_clinical_report.txt')
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"✅ Report generated for Patient {patient_id}")
        
        # Store recommendations
        for rec in recommendations:
            rec['patient_id'] = patient_id
            all_recommendations.append(rec)
    
    # Save recommendations summary
    if all_recommendations:
        rec_df = pd.DataFrame(all_recommendations)
        rec_file = os.path.join(OUTPUT_DIR, 'test_recommendations.csv')
        rec_df.to_csv(rec_file, index=False)
        print(f"\n✅ Test recommendations saved to: {rec_file}")
    
    print("\n" + "="*80)
    print("CLINICAL WORKFLOW COMPLETE")
    print("="*80)
    print(f"\n📁 Reports saved to: {reports_dir}/")
    print(f"\nWorkflow Summary:")
    print(f"  1. ✅ Loaded patient test results")
    print(f"  2. ✅ Trained model made predictions")
    print(f"  3. ✅ Identified organs requiring attention")
    print(f"  4. ✅ Recommended confirmatory tests")
    print(f"  5. ⏳ NEXT STEP: Doctor reviews and validates predictions")
    print(f"  6. ⏳ THEN: Order recommended tests for confirmation")
    print(f"\n💡 This workflow mimics real AI-assisted clinical decision making.\n")


if __name__ == "__main__":
    main()
