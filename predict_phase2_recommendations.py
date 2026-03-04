#!/usr/bin/env python3
"""
Phase 2: Confirmatory Test Recommendations
===========================================

This phase generates test recommendations AFTER doctor has validated Phase 1 predictions.

Workflow:
1. Doctor reviews Phase 1 AI predictions
2. Doctor VALIDATES predictions (confirms they are clinically plausible)
3. THIS SCRIPT: Generate specific confirmatory test recommendations
4. Doctor orders recommended tests
5. New test results validate or refute AI predictions

This phase should ONLY run after physician validation of Phase 1.
"""

import os
import pandas as pd
from datetime import datetime

OUTPUT_DIR = 'predictions_output'

# Confirmatory test recommendations by organ system
CONFIRMATORY_TESTS = {
    'kidney': {
        'initial_tests': ['Serum_Creatinine_Result', 'eGFR Result', 'Blood Urea Result', 'Serum - Potassium', 'Serum - Sodium'],
        'confirmatory_tests': {
            'SEVERE': [
                'Urine Albumin-to-Creatinine Ratio (UACR)',
                '24-hour Urine Collection (Creatinine Clearance)',
                'Renal Ultrasound with Doppler',
                'Cystatin C Blood Test',
                'Kidney Biopsy (if indicated)',
                'Renal Angiography (if vascular cause suspected)'
            ],
            'MODERATE': [
                'Urine Albumin-to-Creatinine Ratio (UACR)',
                'Renal Ultrasound',
                'Cystatin C Blood Test',
                'Urine Electrolytes Panel'
            ],
            'MILD': [
                'Urine Albumin-to-Creatinine Ratio (UACR)',
                'Repeat Creatinine and eGFR in 3 months'
            ]
        }
    },
    'liver': {
        'initial_tests': ['SGPT (ALT) Result', 'SGOT (AST) Result'],
        'confirmatory_tests': {
            'SEVERE': [
                'Comprehensive Liver Function Panel',
                'Gamma-GT (GGT)',
                'Alkaline Phosphatase',
                'Total and Direct Bilirubin',
                'Prothrombin Time/INR',
                'Hepatitis Panel (A, B, C)',
                'Liver Ultrasound or CT',
                'FibroScan or Liver Elastography',
                'Consider Liver Biopsy'
            ],
            'MODERATE': [
                'Comprehensive Liver Function Panel',
                'Gamma-GT (GGT)',
                'Hepatitis Panel (A, B, C)',
                'Liver Ultrasound',
                'FibroScan'
            ],
            'MILD': [
                'Repeat ALT/AST in 6 weeks',
                'Gamma-GT',
                'Liver Ultrasound'
            ]
        }
    },
    'cardiovascular system': {
        'initial_tests': ['Total Cholesterol', 'LDL-Cholesterol', 'HDL Cholesterol', 'Triglycerides'],
        'confirmatory_tests': {
            'SEVERE': [
                'Apolipoprotein B (ApoB)',
                'Lipoprotein(a) [Lp(a)]',
                'hs-CRP (high-sensitivity C-Reactive Protein)',
                'Cardiac Troponin I or T',
                'NT-proBNP or BNP',
                'ECG (Electrocardiogram)',
                'Echocardiogram',
                'Coronary Calcium Score CT',
                'Stress Test or Cardiac Catheterization if indicated'
            ],
            'MODERATE': [
                'Apolipoprotein B (ApoB)',
                'Lipoprotein(a)',
                'hs-CRP',
                'ECG',
                'Coronary Calcium Score CT'
            ],
            'MILD': [
                'Repeat Lipid Panel in 3 months',
                'hs-CRP',
                'ECG'
            ]
        }
    },
    'pancreas': {
        'initial_tests': ['HbA1c Result', 'Fasting Glucose'],
        'confirmatory_tests': {
            'SEVERE': [
                '2-hour Oral Glucose Tolerance Test (OGTT)',
                'Fasting and Postprandial C-Peptide',
                'Fasting Insulin Level',
                'Fructosamine',
                'Continuous Glucose Monitoring (7-14 days)',
                'Anti-GAD Antibodies (Type 1 screening)',
                'Pancreatic CT or MRI',
                'Urine Microalbumin'
            ],
            'MODERATE': [
                'Oral Glucose Tolerance Test (OGTT)',
                'C-Peptide',
                'Fasting Insulin',
                'Continuous Glucose Monitoring'
            ],
            'MILD': [
                'Repeat HbA1c in 3 months',
                'Fasting Glucose monitoring',
                'Consider OGTT'
            ]
        }
    },
    'thyroid': {
        'initial_tests': ['TSH'],
        'confirmatory_tests': {
            'SEVERE': [
                'Free T4 (Thyroxine)',
                'Free T3 (Triiodothyronine)',
                'Thyroid Peroxidase Antibodies (TPO)',
                'Thyroglobulin Antibodies (TgAb)',
                'Thyroid Receptor Antibodies (TRAb)',
                'Thyroid Ultrasound',
                'Thyroid Uptake and Scan',
                'Fine Needle Aspiration if nodules present'
            ],
            'MODERATE': [
                'Free T4',
                'Free T3',
                'TPO Antibodies',
                'Thyroid Ultrasound'
            ],
            'MILD': [
                'Repeat TSH in 6-8 weeks',
                'Free T4'
            ]
        }
    },
    'blood': {
        'initial_tests': ['Haemoglobin Absolute Value', 'WBC Absolute Value', 'Platelet Count Absolute Value', 'RBC Absolute Value'],
        'confirmatory_tests': {
            'SEVERE': [
                'Complete Blood Count with Differential',
                'Peripheral Blood Smear',
                'Reticulocyte Count',
                'Comprehensive Iron Panel (Iron, Ferritin, TIBC, % Saturation)',
                'Vitamin B12 and Folate Levels',
                'Hemoglobin Electrophoresis',
                'Direct Antiglobulin Test (Coombs)',
                'Bone Marrow Aspiration and Biopsy',
                'Flow Cytometry if malignancy suspected'
            ],
            'MODERATE': [
                'CBC with Differential',
                'Peripheral Blood Smear',
                'Iron Panel',
                'Vitamin B12 and Folate',
                'Reticulocyte Count'
            ],
            'MILD': [
                'Repeat CBC in 4-6 weeks',
                'Iron Panel',
                'Vitamin B12 and Folate'
            ]
        }
    },
    'immune system': {
        'initial_tests': ['WBC Absolute Value', 'Lymphocyte Count'],
        'confirmatory_tests': {
            'SEVERE': [
                'Immunoglobulin Panel (IgG, IgA, IgM, IgE)',
                'Complement Levels (C3, C4, CH50)',
                'Comprehensive Autoimmune Panel (ANA, RF, Anti-CCP, dsDNA)',
                'T-cell and B-cell Subsets (Flow Cytometry)',
                'HIV-1/2 Antibody and Antigen Test',
                'CMV, EBV, Toxoplasma Serologies',
                'Lymph Node Biopsy if indicated'
            ],
            'MODERATE': [
                'Immunoglobulin Panel',
                'Complement Levels',
                'ANA and RF',
                'HIV Test'
            ],
            'MILD': [
                'Repeat WBC with Differential',
                'Consider Immunoglobulin Panel'
            ]
        }
    }
}


def load_phase1_predictions():
    """Load validated predictions from Phase 1."""
    pred_file = os.path.join(OUTPUT_DIR, 'phase1_predictions_summary.csv')
    
    if not os.path.exists(pred_file):
        print("❌ Error: Phase 1 predictions not found.")
        print("   Run predict_phase1_diagnosis.py first.")
        return None
    
    df = pd.read_csv(pred_file)
    print(f"✅ Loaded {len(df)} predicted abnormalities from Phase 1")
    return df


def match_organ_to_tests(organ_name):
    """Match predicted organ to test recommendation category."""
    organ_lower = organ_name.lower()
    
    # Direct matches
    for key in CONFIRMATORY_TESTS.keys():
        if key in organ_lower or organ_lower in key:
            return key
    
    # Special mappings
    if 'heart' in organ_lower or 'cardiac' in organ_lower or 'vascular' in organ_lower:
        return 'cardiovascular system'
    elif 'diabete' in organ_lower or 'glucose' in organ_lower or 'insulin' in organ_lower:
        return 'pancreas'
    elif 'renal' in organ_lower or 'nephro' in organ_lower:
        return 'kidney'
    elif 'hepat' in organ_lower:
        return 'liver'
    elif 'hematolog' in organ_lower or 'anemia' in organ_lower:
        return 'blood'
    
    return None


def generate_test_recommendations(predictions_df, patient_data):
    """Generate specific test recommendations based on validated predictions."""
    
    recommendations = []
    
    for patient_id in predictions_df['patient_id'].unique():
        patient_preds = predictions_df[predictions_df['patient_id'] == patient_id]
        patient_tests = patient_data[patient_data['patient_id'] == patient_id]['test_name'].tolist()
        
        for _, pred in patient_preds.iterrows():
            organ = pred['organ']
            severity = pred['severity_name']
            confidence = pred['confidence']
            
            # Match to test category
            test_category = match_organ_to_tests(organ)
            
            if test_category and test_category in CONFIRMATORY_TESTS:
                test_info = CONFIRMATORY_TESTS[test_category]
                
                # Check which initial tests are done
                initial_done = [t for t in test_info['initial_tests'] if t in patient_tests]
                initial_missing = [t for t in test_info['initial_tests'] if t not in patient_tests]
                
                # Get confirmatory tests based on severity
                if severity in test_info['confirmatory_tests']:
                    confirmatory = test_info['confirmatory_tests'][severity]
                else:
                    confirmatory = test_info['confirmatory_tests'].get('MODERATE', [])
                
                # Priority
                if severity == 'SEVERE':
                    priority = 'URGENT (within 24-48 hours)'
                elif severity == 'MODERATE':
                    priority = 'HIGH (within 1-2 weeks)'
                else:
                    priority = 'ROUTINE (within 1 month)'
                
                recommendations.append({
                    'patient_id': patient_id,
                    'organ_system': organ,
                    'predicted_severity': severity,
                    'model_confidence': f"{confidence:.1f}%",
                    'priority': priority,
                    'initial_tests_completed': ', '.join(initial_done) if initial_done else 'None',
                    'initial_tests_missing': ', '.join(initial_missing) if initial_missing else 'None',
                    'recommended_tests': ' | '.join(confirmatory),
                    'test_count': len(confirmatory)
                })
    
    return pd.DataFrame(recommendations)


def generate_phase2_report(patient_id, patient_recs, patient_pred):
    """Generate Phase 2 test recommendation report for a patient."""
    
    report = []
    report.append("="*80)
    report.append("PHASE 2: CONFIRMATORY TEST RECOMMENDATIONS")
    report.append("="*80)
    report.append(f"Patient ID: {patient_id}")
    report.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    report.append("STATUS: Phase 1 AI predictions have been VALIDATED by physician")
    report.append("PURPOSE: Recommend confirmatory tests to validate AI predictions")
    report.append("")
    
    report.append("="*80)
    report.append("VALIDATED AI PREDICTIONS REQUIRING CONFIRMATION")
    report.append("="*80)
    report.append("")
    
    for _, pred in patient_pred.iterrows():
        report.append(f"• {pred['organ']}: {pred['severity_name']} (Confidence: {pred['confidence']:.1f}%)")
    report.append("")
    
    report.append("="*80)
    report.append("RECOMMENDED CONFIRMATORY TESTS")
    report.append("="*80)
    report.append("")
    
    for i, rec in patient_recs.iterrows():
        report.append(f"ORGAN SYSTEM: {rec['organ_system']}")
        report.append(f"Predicted Severity: {rec['predicted_severity']}")
        report.append(f"Priority: {rec['priority']}")
        report.append("")
        
        if rec['initial_tests_missing'] != 'None':
            report.append("⚠️  MISSING BASELINE TESTS (Order first):")
            for test in rec['initial_tests_missing'].split(', '):
                report.append(f"  • {test}")
            report.append("")
        
        report.append("📋 CONFIRMATORY TESTS TO ORDER:")
        tests = rec['recommended_tests'].split(' | ')
        for j, test in enumerate(tests, 1):
            report.append(f"  {j}. {test}")
        report.append("")
        
        report.append(f"CLINICAL RATIONALE:")
        report.append(f"  These tests will help confirm the {rec['predicted_severity']} {rec['organ_system']}")
        report.append(f"  dysfunction predicted by the AI model with {rec['model_confidence']} confidence.")
        report.append("")
        report.append("-"*80)
        report.append("")
    
    report.append("="*80)
    report.append("NEXT STEPS")
    report.append("="*80)
    report.append("")
    report.append("1. Physician reviews and approves test orders")
    report.append("2. Order approved confirmatory tests")
    report.append("3. Patient completes tests")
    report.append("4. Compare new test results with AI predictions")
    report.append("5. Validate or refute AI predictions")
    report.append("6. Proceed with treatment if confirmed")
    report.append("")
    report.append("Physician Approval: _______________________  Date: __________")
    report.append("")
    report.append("="*80)
    
    return "\n".join(report)


def main():
    """Phase 2: Generate test recommendations after doctor validation."""
    print("\n" + "="*80)
    print("PHASE 2: CONFIRMATORY TEST RECOMMENDATIONS")
    print("Post-Validation Test Ordering Protocol")
    print("="*80)
    
    print("\n⚠️  IMPORTANT: This phase should only run AFTER:")
    print("   1. Phase 1 predictions have been generated")
    print("   2. A physician has reviewed and VALIDATED the predictions")
    print("")
    
    input_confirm = input("Have Phase 1 predictions been validated by a physician? (yes/no): ")
    if input_confirm.lower() not in ['yes', 'y']:
        print("\n⚠️  Aborted: Phase 1 predictions must be validated first.")
        print("   Please have a physician review Phase 1 reports before proceeding.")
        return
    
    # Load Phase 1 predictions
    print("\n📂 Step 1: Loading validated predictions from Phase 1...")
    predictions_df = load_phase1_predictions()
    if predictions_df is None:
        return
    
    # Load patient data
    print("\n📂 Step 2: Loading patient clinical data...")
    if os.path.exists(os.path.join(OUTPUT_DIR, 'new_patients_temp.csv')):
        patient_data = pd.read_csv(os.path.join(OUTPUT_DIR, 'new_patients_temp.csv'))
        print(f"✅ Loaded data for {patient_data['patient_id'].nunique()} patients")
    else:
        print("❌ Patient data not found.")
        return
    
    # Generate recommendations
    print("\n🔬 Step 3: Generating confirmatory test recommendations...")
    recommendations_df = generate_test_recommendations(predictions_df, patient_data)
    
    if len(recommendations_df) == 0:
        print("✅ No test recommendations needed - all patients appear normal.")
        return
    
    print(f"✅ Generated recommendations for {recommendations_df['patient_id'].nunique()} patients")
    
    # Save recommendations
    rec_file = os.path.join(OUTPUT_DIR, 'phase2_test_recommendations.csv')
    recommendations_df.to_csv(rec_file, index=False)
    print(f"✅ Recommendations saved to: {rec_file}")
    
    # Generate detailed reports
    print("\n📄 Step 4: Creating detailed recommendation reports...")
    phase2_dir = os.path.join(OUTPUT_DIR, 'phase2_recommendations')
    os.makedirs(phase2_dir, exist_ok=True)
    
    for patient_id in recommendations_df['patient_id'].unique()[:5]:  # First 5
        patient_recs = recommendations_df[recommendations_df['patient_id'] == patient_id]
        patient_pred = predictions_df[predictions_df['patient_id'] == patient_id]
        
        report_text = generate_phase2_report(patient_id, patient_recs, patient_pred)
        
        report_file = os.path.join(phase2_dir, f'phase2_patient_{patient_id}_tests.txt')
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"  ✅ Patient {patient_id}: {len(patient_recs)} test recommendation sets")
    
    print("\n" + "="*80)
    print("PHASE 2 COMPLETE")
    print("="*80)
    print(f"\n📁 Reports saved to: {phase2_dir}/")
    print(f"\n✅ Generated:")
    print(f"   • Confirmatory test recommendations for validated predictions")
    print(f"   • Priority-based ordering protocol")
    print(f"   • Clinical rationale for each test")
    print(f"\n⏭️  NEXT STEPS:")
    print(f"   1. Physician reviews and approves test orders")
    print(f"   2. Order confirmatory tests")
    print(f"   3. Patient completes tests")
    print(f"   4. Compare results with AI predictions")
    print(f"   5. Validate accuracy of AI model")
    print("")


if __name__ == "__main__":
    main()
