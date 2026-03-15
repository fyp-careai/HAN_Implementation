from flask import Flask, request, jsonify
import torch
from inference import predict_new_patient, recommend_tests_from_diseases
from dataset_pyg import load_pyg_data

app = Flask(__name__)
PORT = 5001


# ------------------------------------------------------------
# Load graph once at server startup
# ------------------------------------------------------------

data, patient_to_idx, test_to_idx, organ_to_idx, disease_to_idx, norm_stats = load_pyg_data()

idx_to_test = {v: k for k, v in test_to_idx.items()}
idx_to_disease = {v: k for k, v in disease_to_idx.items()}


# ------------------------------------------------------------
# Graph-based test recommendation
# ------------------------------------------------------------




# ------------------------------------------------------------
# Prediction API
# ------------------------------------------------------------

@app.route('/predict', methods=['POST'])
def predict():

    req = request.json

    patient_id = req.get("patient_id")
    age = req.get("age")
    sex = req.get("sex")
    lab_results = req.get("lab_results", [])

    if not lab_results:
        return jsonify({"error": "No lab results provided"}), 400

    # --------------------------------------------------------
    # Run model inference
    # --------------------------------------------------------

    predictions, abnormal_features = predict_new_patient(lab_results)

    # Top predicted diseases
    recommended_tests = recommend_tests_from_diseases(
        data,
        predictions,
        disease_to_idx,
        organ_to_idx,
        test_to_idx
    )

    # --------------------------------------------------------
    # Graph-based test recommendation
    # --------------------------------------------------------

    #recommended_tests = recommend_tests_from_graph(top_diseases)

    # --------------------------------------------------------
    # Build response
    # --------------------------------------------------------

    response = {

        "patient_info": {
            "patient_id": patient_id,
            "age": age,
            "sex": sex
        },

        "abnormal_labs": abnormal_features,

        "predictions": [
            {
                "disease": d,
                "gnn_score": float(s),
                "final_score": float(z)
            }
            for d, s, z in predictions[:10]
        ],
        "recommended_tests": [
            {
                "test_name": test_name,
                "score": float(score)
            }
            for test_name, score in recommended_tests
        ]
        #"recommended_tests": recommended_tests
    }

    return jsonify(response)


# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "server running"})


# ------------------------------------------------------------
# Run server
# ------------------------------------------------------------

if __name__ == '__main__':

    print("Neuro-Symbolic Medical AI Server Starting...")
    print(f"Server running at http://localhost:{PORT}")

    app.run(
        host='0.0.0.0',
        port=PORT,
        debug=True
    )