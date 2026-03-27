import os
import json
from flask import Flask, request, jsonify

# Import the new v6 inference logic
from inference_v6 import (
    predict_new_patient_v6, 
    extract_abnormal_features,
    TEST_INFO_PATH,
    generate_early_warning
)

app = Flask(__name__)
PORT = 5001

# ------------------------------------------------------------
# Prediction API
# ------------------------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():
    try:
        req = request.json
        if not req:
            return jsonify({"error": "Invalid JSON payload"}), 400

        patient_id = req.get("patient_id", "Unknown")
        age = req.get("age", "Unknown")
        sex = req.get("sex", "Unknown")
        lab_results = req.get("lab_results", [])

        # Attach time_since_test (days) from each lab entry (optional field)
        for lr in lab_results:
            if "time_since_test" not in lr:
                lr["time_since_test"] = None  # mark as missing if not provided

        if not lab_results:
            return jsonify({"error": "No lab results provided"}), 400

        print(f"\n[API] Received prediction request for Patient {patient_id} (Labs: {len(lab_results)})")

        # --------------------------------------------------------
        # 1. Get Abnormal Features
        # --------------------------------------------------------
        # We compute abnormal features directly so we can include them in the response
        abnormal_features = extract_abnormal_features(lab_results, TEST_INFO_PATH)

        # Annotate each abnormal feature with time_since_test from the matching lab entry
        lab_time_map = {
            str(lr.get("test_name", "")).strip(): lr.get("time_since_test")
            for lr in lab_results
        }
        for feat in abnormal_features:
            feat["time_since_test"] = lab_time_map.get(feat["test"])

        # --------------------------------------------------------
        # 2. Run model inference (v6 logic)
        # --------------------------------------------------------
        # This will load the graph/model and perform inductive MC-dropout scoring.
        # (Note: For extreme high-traffic production, graph and model loading 
        # inside predict(...) should be preloaded globally, but this works perfectly for now).
        v6_output = predict_new_patient_v6(lab_results)

        # --------------------------------------------------------
        # 3. Format Response
        # --------------------------------------------------------
        
        # Format predictions list
        predictions_formatted = []
        for p in v6_output.get("disease_link_scores", []):
            predictions_formatted.append({
                "disease": p["disease"],
                "gnn_score": p["gnn_score"],
                "rule_score": p["rule_score"],
                "final_score": p["final_score"],
                "uncertainty": p["uncertainty"]
            })

        # Format recommended tests mapping
        # Flattening the recommended tests from {disease: [tests]} into a unified list
        # or keeping it grouped. We'll group them for better frontend usage.
        recommended_tests = v6_output.get("recommended_tests", {})

        early_warnings = generate_early_warning(
            predictions_formatted,
            abnormal_features,
            top_k=3
        )

        response = {
            "patient_info": {
                "patient_id": patient_id,
                "age": age,
                "sex": sex
            },
            "abnormal_labs": abnormal_features,
            "predictions": predictions_formatted[:10], # Top 10 predictions
            "early_warnings": early_warnings,
            #"recommended_tests": recommended_tests,
            "recommended_tests": v6_output.get("recommended_tests", {}),
            "inference_details": {
                "method": v6_output.get("method"),
                "neighbor_count": v6_output.get("neighbor_count"),
                "status": "success",
                "fusion_weights": v6_output.get("fusion")
            }
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "status": "failed"}), 500


# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "server running", "version": "v6"})


# ------------------------------------------------------------
# Run server
# ------------------------------------------------------------

if __name__ == '__main__':
    print("="*60)
    print(" CareAI Neuro-Symbolic Medical API Server v6 Starting...")
    print(f" Server running at http://localhost:{PORT}")
    print("="*60)

    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=True,
        use_reloader=False # Disable reloader to prevent duplicate graph building if testing
    )
