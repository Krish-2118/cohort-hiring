from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

class EnhancedPrescriptionGenerator:
    def __init__(self):
        self.is_trained = False

    def train_enhanced_model(self, use_ensemble=True):
        print("Training the model...")
        # Dummy training logic
        self.is_trained = True

    def generate_prescription(self, patient_data, doctor_id, consider_cost=True):
        if not self.is_trained:
            raise Exception("Model not trained yet.")
        return {
            "prescription": f"Prescription for patient by Dr.{doctor_id} based on {patient_data}"
        }

prescription_generator = EnhancedPrescriptionGenerator()

@app.route('/')
def home():
    return jsonify({"message": "ML backend running"}), 200

@app.route('/health')
def health():
    return jsonify({
        "status": "ok",
        "trained": prescription_generator.is_trained
    })

@app.route('/train', methods=['POST', 'OPTIONS'])
def train_model():
    try:
        prescription_generator.train_enhanced_model(use_ensemble=True)
        return jsonify({"message": "Model trained successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/generate_prescription', methods=['POST', 'OPTIONS'])
def generate_prescription():
    try:
        if not prescription_generator.is_trained:
            return jsonify({"status": "error", "message": "Model not trained"}), 500
        data = request.get_json()
        patient_data = data.get("patient_data", {})
        doctor_id = data.get("doctor_id", "default")
        consider_cost = data.get("consider_cost", True)
        result = prescription_generator.generate_prescription(patient_data, doctor_id, consider_cost)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    global prescription_generator
    prescription_generator = EnhancedPrescriptionGenerator()
    prescription_generator.train_enhanced_model(use_ensemble=True)
    print("Model trained and server starting...")
    app.run(debug=True, host='0.0.0.0', port=5000)
