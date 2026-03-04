#!/usr/bin/env python3
"""
Phase 1: AI-Assisted Diagnosis using Trained Model
===================================================

This phase uses the TRAINED HAN model to predict organ issues from patient data.
NO test recommendations yet - just predictions for doctor review.

Workflow:
1. Patient provides initial lab results
2. Trained model predicts organ severity
3. Generate prediction report
4. Doctor reviews predictions → VALIDATES or REJECTS
5. If validated → proceed to Phase 2 (test recommendations)
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
    0: {'name': 'NORMAL', 'color': '#2ecc71', 'description': 'No significant abnormalities detected'},
    1: {'name': 'MILD', 'color': '#f39c12', 'description': 'Minor abnormalities - monitor closely'},
    2: {'name': 'MODERATE', 'color': '#e67e22', 'description': 'Significant concern - investigation needed'},
    3: {'name': 'SEVERE', 'color': '#e74c3c', 'description': 'Critical - immediate attention required'}
}


def load_trained_model(data_loader):
    """Load the trained HAN model."""
    print(f"\n🔧 Loading trained model from: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model not found at {MODEL_PATH}")
        print("   Train the model first using: cd Other_py && python train_complete.py")
        return None
    
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    # Infer model architecture from checkpoint
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
    
    # Handle feature dimension mismatch
    if data_loader.patient_feats.shape[1] != in_dim:
        print(f"\n⚠️  Feature mismatch detected:")
        print(f"   Model expects: {in_dim} features")
        print(f"   Data contains: {data_loader.patient_feats.shape[1]} features")
        
        current_feats = data_loader.patient_feats
        if current_feats.shape[1] < in_dim:
            # Pad with zeros
            padding = np.zeros((current_feats.shape[0], in_dim - current_feats.shape[1]))
            data_loader.patient_feats = np.hstack([current_feats, padding])
            print(f"   → Padded features to match model input")
        else:
            # Truncate
            data_loader.patient_feats = current_feats[:, :in_dim]
            print(f"   → Truncated {current_feats.shape[1] - in_dim} extra features")
    
    print(f"✅ Model loaded successfully!")
    print(f"   Model architecture:")
    print(f"     • Input features: {in_dim}")
    print(f"     • Hidden dimensions: {hidden_dim}")
    print(f"     • Output classes: {out_dim}")
    print(f"     • Organ systems tracked: {num_organs}")
    print(f"     • Meta-path: P-S-P (Patient-Symptom-Patient)")
    
    return model


def make_predictions(model, data_loader, patient_indices):
    """Use trained model to predict organ severity for new patients."""
    print(f"\n📊 Making predictions for {len(patient_indices)} patients...")
    
    # Prepare features
    if isinstance(data_loader.patient_feats, np.ndarray):
        features = torch.from_numpy(data_loader.patient_feats).float().to(DEVICE)
    else:
        features = data_loader.patient_feats.to(DEVICE)
    
    # Build P-S-P meta-path if needed
    if not hasattr(data_loader, 'metapath_matrices') or len(data_loader.metapath_matrices) == 0:
        print("   Building P-S-P meta-path relations...")
        data_loader.build_metapaths(['P-S-P'])
    
    neighbor_dicts = data_loader.metapath_matrices
    
    # Run model inference
    with torch.no_grad():
        organ_logits, organ_scores, embeddings, attention_weights = model(features, neighbor_dicts)
        
        # Get predictions
        predictions = torch.argmax(organ_logits, dim=2).cpu().numpy()  # [N, num_organs]
        scores = organ_scores.cpu().numpy()
        
        # Calculate confidence scores
        probs = torch.softmax(organ_logits, dim=2)  # [N, num_organs, num_classes]
        confidences = probs.max(dim=2)[0].cpu().numpy()  # Max probability per organ
    
    # Get organ names
    organ_names = data_loader.organs if hasattr(data_loader, 'organs') else [f"Organ_{i+1}" for i in range(predictions.shape[1])]
    
    print(f"✅ Predictions completed using trained P-S-P model!")
    
    return predictions, scores, confidences, organ_names


def generate_prediction_report(patient_id, patient_data, predictions, scores, confidences, organ_names, patient_idx):
    """Generate Phase 1 prediction report for doctor review."""
    
    # Identify affected organs
    affected_organs = []
    normal_organs = []
    
    for organ_idx, organ_name in enumerate(organ_names):
        if organ_idx < predictions.shape[1]:
            severity = int(predictions[patient_idx, organ_idx])
            organ_info = {
                'organ': organ_name,
                'severity': severity,
                'severity_name': SEVERITY_LEVELS[severity]['name'],
                'description': SEVERITY_LEVELS[severity]['description'],
                'damage_score': float(scores[patient_idx, organ_idx]),
                'confidence': float(confidences[patient_idx, organ_idx]) * 100  # Convert to percentage
            }
            
            if severity > 0:
                affected_organs.append(organ_info)
            else:
                normal_organs.append(organ_info)
    
    # Sort by severity and confidence
    affected_organs.sort(key=lambda x: (x['severity'], x['damage_score']), reverse=True)
    
    # Generate report
    report = []
    report.append("="*80)
    report.append("PHASE 1: AI DIAGNOSTIC PREDICTION REPORT")
    report.append("="*80)
    report.append(f"Patient ID: {patient_id}")
    report.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"AI Model: HAN++ with P-S-P Meta-Path")
    report.append(f"Model Path: {MODEL_PATH}")
    report.append("")
    
    # Patient info
    if 'age_at_report' in patient_data.columns:
        age = int(patient_data.iloc[0]['age_at_report'])
        sex = patient_data.iloc[0]['sex']
        report.append(f"Patient: {age} years old, {sex}")
        report.append("")
    
    # Available tests
    report.append("="*80)
    report.append("AVAILABLE CLINICAL DATA")
    report.append("="*80)
    tests_available = patient_data['test_name'].unique()
    report.append(f"Laboratory tests performed: {len(tests_available)}")
    report.append("")
    
    for i, test in enumerate(tests_available, 1):
        test_row = patient_data[patient_data['test_name'] == test].iloc[0]
        report.append(f"  {i:2d}. {test}: {test_row['test_value']}")
    report.append("")
    
    # AI Predictions
    report.append("="*80)
    report.append("AI MODEL PREDICTIONS")
    report.append("="*80)
    report.append("")
    
    if len(affected_organs) == 0:
        report.append("✅ NO SIGNIFICANT ABNORMALITIES DETECTED")
        report.append("")
        report.append("According to the AI model, all organ systems appear to be")
        report.append("functioning within normal parameters based on available data.")
        report.append("")
        report.append("Recommendation: Continue routine health monitoring.")
    else:
        report.append(f"⚠️  ATTENTION: {len(affected_organs)} ORGAN SYSTEM(S) SHOW PREDICTED ABNORMALITIES")
        report.append("")
        
        for i, organ in enumerate(affected_organs, 1):
            report.append(f"{i}. {organ['organ'].upper()}")
            report.append(f"   ├─ Predicted Severity: {organ['severity_name']} (Level {organ['severity']}/3)")
            report.append(f"   ├─ Clinical Significance: {organ['description']}")
            report.append(f"   ├─ Damage Score: {organ['damage_score']:.3f}")
            report.append(f"   └─ Model Confidence: {organ['confidence']:.1f}%")
            report.append("")
    
    # Normal organs summary
    if len(normal_organs) > 0:
        report.append(f"✅ NORMAL ORGAN SYSTEMS: {len(normal_organs)}")
        organ_list = ", ".join([o['organ'] for o in normal_organs[:10]])
        if len(normal_organs) > 10:
            organ_list += f" (and {len(normal_organs) - 10} more)"
        report.append(f"   {organ_list}")
        report.append("")
    
    # Doctor validation section
    report.append("="*80)
    report.append("REQUIRED: PHYSICIAN VALIDATION")
    report.append("="*80)
    report.append("")
    report.append("This report contains AI-generated predictions that MUST be validated")
    report.append("by a qualified physician before any clinical action is taken.")
    report.append("")
    report.append("PHYSICIAN REVIEW CHECKLIST:")
    report.append("  [ ] Review patient history and symptoms")
    report.append("  [ ] Verify available test results")
    report.append("  [ ] Assess AI predictions for clinical plausibility")
    report.append("  [ ] Compare with differential diagnosis")
    report.append("  [ ] Decide: VALIDATE or REJECT AI predictions")
    report.append("")
    report.append("If predictions are VALIDATED:")
    report.append("  → Proceed to Phase 2 for confirmatory test recommendations")
    report.append("")
    report.append("If predictions are REJECTED:")
    report.append("  → Document reasons for rejection")
    report.append("  → Follow standard diagnostic protocol")
    report.append("")
    report.append("Physician Name: _______________________  Date: __________")
    report.append("")
    report.append("Validation Decision:  [ ] VALIDATED    [ ] REJECTED")
    report.append("")
    report.append("="*80)
    
    # Medical disclaimer
    report.append("")
    report.append("IMPORTANT MEDICAL DISCLAIMER")
    report.append("="*80)
    report.append("This AI system is a DECISION SUPPORT TOOL, not a replacement for")
    report.append("clinical judgment. All predictions must be interpreted by a qualified")
    report.append("healthcare professional in the context of the patient's complete")
    report.append("clinical picture, medical history, and physical examination.")
    report.append("")
    report.append("Do NOT use this report alone for diagnosis or treatment decisions.")
    report.append("="*80)
    
    return "\n".join(report), affected_organs


def main():
    """Phase 1: Generate AI predictions using trained model."""
    print("\n" + "="*80)
    print("PHASE 1: AI-ASSISTED DIAGNOSIS")
    print("Using Trained Model for Organ Severity Prediction")
    print("="*80)
    
    # Load patient data
    print("\n📂 Step 1: Loading patient data...")
    
    if os.path.exists(os.path.join(OUTPUT_DIR, 'new_patients_temp.csv')):
        new_patients_df = pd.read_csv(os.path.join(OUTPUT_DIR, 'new_patients_temp.csv'))
        print(f"✅ Loaded {new_patients_df['patient_id'].nunique()} patients")
    else:
        print("⚠️  No patient data found.")
        print("   Run predict_psp_new_patients.py first to generate synthetic patients.")
        return
    
    combined_file = os.path.join(OUTPUT_DIR, 'combined_patient_data.csv')
    if not os.path.exists(combined_file):
        print("⚠️  Combined data file not found.")
        print("   Run predict_psp_new_patients.py first.")
        return
    
    # Build medical knowledge graph
    print("\n📊 Step 2: Building medical knowledge graph...")
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
    
    # Map patient IDs to indices
    new_patient_ids = new_patients_df['patient_id'].unique()
    patient_id_to_idx = {pid: idx for idx, pid in enumerate(data_loader.patient_ids)}
    new_patient_indices = [patient_id_to_idx[pid] for pid in new_patient_ids if pid in patient_id_to_idx]
    
    print(f"✅ Knowledge graph constructed: {len(new_patient_indices)} patients ready for prediction")
    
    # Load trained model
    print("\n🤖 Step 3: Loading trained AI model...")
    model = load_trained_model(data_loader)
    if model is None:
        return
    
    # Make predictions
    print("\n🔮 Step 4: Generating predictions...")
    predictions, scores, confidences, organ_names = make_predictions(
        model, data_loader, new_patient_indices)
    
    # Generate reports
    print("\n📄 Step 5: Creating physician review reports...")
    
    phase1_dir = os.path.join(OUTPUT_DIR, 'phase1_predictions')
    os.makedirs(phase1_dir, exist_ok=True)
    
    all_predictions = []
    
    for i, patient_idx in enumerate(new_patient_indices[:10]):  # First 10 patients
        patient_id = data_loader.patient_ids[patient_idx]
        patient_data = new_patients_df[new_patients_df['patient_id'] == patient_id]
        
        # Generate report
        report_text, affected_organs = generate_prediction_report(
            patient_id, patient_data, predictions, scores, confidences, 
            organ_names, patient_idx)
        
        # Save report
        report_file = os.path.join(phase1_dir, f'phase1_patient_{patient_id}.txt')
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        print(f"  ✅ Patient {patient_id}: {len(affected_organs)} organ(s) with predicted abnormalities")
        
        # Store for summary
        for organ in affected_organs:
            organ['patient_id'] = patient_id
            all_predictions.append(organ)
    
    # Save predictions summary
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        pred_file = os.path.join(OUTPUT_DIR, 'phase1_predictions_summary.csv')
        pred_df.to_csv(pred_file, index=False)
        print(f"\n✅ Predictions summary saved to: {pred_file}")
    
    print("\n" + "="*80)
    print("PHASE 1 COMPLETE")
    print("="*80)
    print(f"\n📁 Reports saved to: {phase1_dir}/")
    print(f"\n⏭️  NEXT STEPS:")
    print(f"   1. Physician reviews prediction reports")
    print(f"   2. Physician validates or rejects AI predictions")
    print(f"   3. If VALIDATED → Run Phase 2 (test recommendations)")
    print(f"   4. If REJECTED → Follow standard diagnostic protocol")
    print(f"\n💡 To proceed to Phase 2:")
    print(f"   → After doctor validation, run: predict_phase2_recommendations.py")
    print(f"   → This will generate confirmatory test recommendations")
    print("")


if __name__ == "__main__":
    main()
