from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import json
from typing import List, Dict, Tuple, Optional
import warnings
import logging
from datetime import datetime
import traceback

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

class EnhancedPrescriptionGenerator:
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.encoders = {}
        self.scalers = {}
        self.medicine_database = self._load_medicine_database()
        self.doctor_preferences = {}
        self.interaction_matrix = self._load_drug_interactions()
        self.contraindications = self._load_contraindications()
        self.dosage_calculator = DosageCalculator()
        self.audit_log = []

    # ... [rest of the class definition, unchanged, as in your code] ...

    # Add this method to make the endpoints work
    def _get_medicine_details(self, medicine: str) -> dict:
        for condition, meds in self.medicine_database.items():
            for med in meds:
                if med['name'] == medicine:
                    return med
        return {}

    def update_doctor_preferences(self, doctor_id: str, prescription_data: dict):
        # Example: Update doctor preferences with provided data
        if doctor_id not in self.doctor_preferences:
            self.doctor_preferences[doctor_id] = {}
        self.doctor_preferences[doctor_id].update(prescription_data)

    def save_model(self, filepath: str):
        joblib.dump({
            'models': self.models,
            'vectorizers': self.vectorizers,
            'encoders': self.encoders,
            'scalers': self.scalers,
        }, filepath)

    def load_model(self, filepath: str):
        data = joblib.load(filepath)
        self.models = data['models']
        self.vectorizers = data['vectorizers']
        self.encoders = data['encoders']
        self.scalers = data['scalers']

# DosageCalculator class here (unchanged)
class DosageCalculator:
    def __init__(self):
        self.dosage_adjustments = {
            'kidney_impairment': {
                'Metformin': 0.5,  # Reduce by 50%
                'Lisinopril': 0.75  # Reduce by 25%
            },
            'elderly': {
                'Lorazepam': 0.5,
                'Digoxin': 0.75
            }
        }

    def calculate_dose(self, medicine: str, age: int, weight: float, 
                      kidney_function: str = 'normal') -> Optional[str]:
        base_dose = self._get_base_dose(medicine)
        if not base_dose:
            return None

        adjustment_factor = 1.0

        # Age adjustments
        if age >= 65 and medicine in self.dosage_adjustments.get('elderly', {}):
            adjustment_factor *= self.dosage_adjustments['elderly'][medicine]

        # Kidney function adjustments
        if kidney_function != 'normal' and medicine in self.dosage_adjustments.get('kidney_impairment', {}):
            adjustment_factor *= self.dosage_adjustments['kidney_impairment'][medicine]

        # Weight-based adjustments (simplified)
        if weight < 50:
            adjustment_factor *= 0.8
        elif weight > 100:
            adjustment_factor *= 1.2

        adjusted_dose = self._apply_dose_adjustment(base_dose, adjustment_factor)
        return adjusted_dose

    def _get_base_dose(self, medicine: str) -> Optional[str]:
        dose_map = {
            'Lisinopril': '10mg',
            'Metformin': '500mg',
            'Amlodipine': '5mg'
        }
        return dose_map.get(medicine)

    def _apply_dose_adjustment(self, base_dose: str, factor: float) -> str:
        import re
        match = re.match(r'(\d+(?:\.\d+)?)(.*)', base_dose)
        if not match:
            return base_dose
        dose_value = float(match.group(1))
        unit = match.group(2)
        adjusted_value = dose_value * factor
        if adjusted_value < 1:
            adjusted_value = round(adjusted_value, 2)
        else:
            adjusted_value = round(adjusted_value, 1)
        return f"{adjusted_value}{unit}"

# Initialize the enhanced prescription generator
prescription_generator = EnhancedPrescriptionGenerator()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/train', methods=['POST'])
def train_model():
    try:
        data = request.json or {}
        use_ensemble = data.get('use_ensemble', True)
        n_samples = data.get('n_samples', 5000)
        if hasattr(prescription_generator, 'create_enhanced_training_data'):
            original_method = prescription_generator.create_enhanced_training_data
            prescription_generator.create_enhanced_training_data = lambda: original_method(n_samples)
        prescription_generator.train_enhanced_model(use_ensemble=use_ensemble)
        return jsonify({
            'status': 'success',
            'message': f'Enhanced model trained successfully with {len(prescription_generator.models)} medicines',
            'model_count': len(prescription_generator.models)
        })
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e), 'traceback': traceback.format_exc()}), 500

@app.route('/generate_prescription', methods=['POST'])
def generate_prescription():
    try:
        data = request.json
        patient_data = data.get('patient_data', {})
        doctor_id = data.get('doctor_id', 'default')
        consider_cost = data.get('consider_cost', True)
        prescription = prescription_generator.generate_enhanced_prescription(
            patient_data, doctor_id, consider_cost
        )
        patient_id = patient_data.get('patient_id', 'unknown')
        prescription_generator.log_prescription(patient_id, doctor_id, prescription)
        return jsonify({
            'status': 'success',
            'prescription': prescription,
            'patient_id': patient_id,
            'generated_at': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Prescription generation error: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/check_interactions', methods=['POST'])
def check_interactions():
    try:
        data = request.json
        medicines = data.get('medicines', [])
        interactions = []
        for i, med1 in enumerate(medicines):
            for med2 in medicines[i+1:]:
                interaction = prescription_generator._check_detailed_interactions(med1, [med2])
                if interaction:
                    interactions.extend(interaction)
        return jsonify({
            'status': 'success',
            'interactions': interactions,
            'interaction_count': len(interactions)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/calculate_dosage', methods=['POST'])
def calculate_dosage():
    try:
        data = request.json
        medicine = data.get('medicine')
        age = data.get('age', 50)
        weight = data.get('weight', 70)
        kidney_function = data.get('kidney_function', 'normal')
        adjusted_dose = prescription_generator.dosage_calculator.calculate_dose(
            medicine, age, weight, kidney_function
        )
        return jsonify({
            'status': 'success',
            'medicine': medicine,
            'adjusted_dose': adjusted_dose,
            'patient_factors': {
                'age': age,
                'weight': weight,
                'kidney_function': kidney_function
            }
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_medicine_info', methods=['GET'])
def get_medicine_info():
    try:
        medicine = request.args.get('medicine')
        if not medicine:
            return jsonify({'status': 'error', 'message': 'Medicine name required'}), 400
        medicine_info = prescription_generator._get_medicine_details(medicine)
        if not medicine_info:
            return jsonify({'status': 'error', 'message': 'Medicine not found'}), 404
        return jsonify({
            'status': 'success',
            'medicine_info': medicine_info
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/update_preferences', methods=['POST'])
def update_preferences():
    try:
        data = request.json
        doctor_id = data.get('doctor_id', 'default')
        prescription_data = data.get('prescription_data', {})
        prescription_generator.update_doctor_preferences(doctor_id, prescription_data)
        return jsonify({
            'status': 'success',
            'message': 'Preferences updated',
            'doctor_id': doctor_id
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_audit_log', methods=['GET'])
def get_audit_log():
    try:
        limit = int(request.args.get('limit', 100))
        doctor_id = request.args.get('doctor_id')
        audit_entries = prescription_generator.audit_log[-limit:]
        if doctor_id:
            audit_entries = [entry for entry in audit_entries if entry.get('doctor_id') == doctor_id]
        return jsonify({
            'status': 'success',
            'audit_entries': audit_entries,
            'total_entries': len(audit_entries)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/get_statistics', methods=['GET'])
def get_statistics():
    try:
        stats = {
            'total_medicines': len(prescription_generator.medicine_database),
            'trained_models': len(prescription_generator.models),
            'doctors_with_preferences': len(prescription_generator.doctor_preferences),
            'total_prescriptions_logged': len(prescription_generator.audit_log),
            'medicine_categories': {}
        }
        for condition, medicines in prescription_generator.medicine_database.items():
            for med in medicines:
                category = med.get('category', 'Unknown')
                stats['medicine_categories'][category] = stats['medicine_categories'].get(category, 0) + 1
        return jsonify({
            'status': 'success',
            'statistics': stats
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/validate_prescription', methods=['POST'])
def validate_prescription():
    try:
        data = request.json
        prescription = data.get('prescription', [])
        patient_data = data.get('patient_data', {})
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': []
        }
        medicine_names = [med.get('medicine') for med in prescription if 'medicine' in med]
        for medicine_info in prescription:
            medicine = medicine_info.get('medicine')
            interactions = prescription_generator._check_detailed_interactions(medicine, medicine_names)
            if interactions:
                for interaction in interactions:
                    if interaction['severity'] == 'major':
                        validation_results['errors'].append(f"Major interaction: {medicine} with {interaction['drug']}")
                        validation_results['is_valid'] = False
                    else:
                        validation_results['warnings'].append(f"Moderate interaction: {medicine} with {interaction['drug']}")
            contraindications = prescription_generator._check_contraindications(
                medicine, patient_data.get('contraindications', [])
            )
            if contraindications:
                validation_results['errors'].append(f"Contraindication: {medicine} with {', '.join(contraindications)}")
                validation_results['is_valid'] = False
            age_warnings = prescription_generator._check_age_warnings(medicine, patient_data.get('age', 50))
            if age_warnings:
                validation_results['warnings'].extend(age_warnings)
        return jsonify({
            'status': 'success',
            'validation': validation_results
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/save_model', methods=['POST'])
def save_model():
    try:
        filepath = request.json.get('filepath', 'enhanced_prescription_model.pkl')
        prescription_generator.save_model(filepath)
        return jsonify({'status': 'success', 'message': f'Model saved to {filepath}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/load_model', methods=['POST'])
def load_model():
    try:
        filepath = request.json.get('filepath', 'enhanced_prescription_model.pkl')
        prescription_generator.load_model(filepath)
        return jsonify({'status': 'success', 'message': f'Model loaded from {filepath}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

@app.route('/')
def root():
    return jsonify({'message': 'ML backend is running'})

if __name__ == '__main__':
    print("Initializing Enhanced Prescription Generator...")
    prescription_generator = EnhancedPrescriptionGenerator()
    print("Training model on startup (this may take a few minutes)...")
    try:
        prescription_generator.train_enhanced_model(use_ensemble=True)
        print("Model training completed successfully!")
    except Exception as e:
        print(f"Warning: Model training failed: {e}")
        print("Server will start without pre-trained model.")
    print("Starting Flask server on port 5000...")
    app.run(debug=True, port=5000, host='0.0.0.0')
