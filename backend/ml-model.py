
from flask import Flask, jsonify, request

# Global model instance
prescription_generator = None

# Global model instance
prescription_generator = None
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
from flask_cors import CORS
CORS(app)


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
import joblib
import json
from typing import List, Dict, Tuple, Optional
import warnings
import logging
from datetime import datetime
warnings.filterwarnings('ignore')
from flask_cors import CORS


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPrescriptionGenerator:
    def __init__(self):
        self.is_trained = False
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
        
    def _load_medicine_database(self) -> Dict:
        """Enhanced medicine database with more details"""
        return {
            'Hypertension': [
                {
                    'name': 'Lisinopril', 'dosage': '10mg', 'frequency': 'once daily', 
                    'category': 'ACE Inhibitor', 'max_dose': '40mg',
                    'contraindications': ['pregnancy', 'hyperkalemia', 'bilateral_renal_artery_stenosis'],
                    'monitoring': ['kidney_function', 'potassium_levels'],
                    'cost_tier': 1
                },
                {
                    'name': 'Amlodipine', 'dosage': '5mg', 'frequency': 'once daily', 
                    'category': 'CCB', 'max_dose': '10mg',
                    'contraindications': ['severe_aortic_stenosis'],
                    'monitoring': ['blood_pressure', 'edema'],
                    'cost_tier': 1
                },
                {
                    'name': 'Metoprolol', 'dosage': '50mg', 'frequency': 'twice daily', 
                    'category': 'Beta Blocker', 'max_dose': '200mg',
                    'contraindications': ['asthma', 'severe_bradycardia', 'heart_block'],
                    'monitoring': ['heart_rate', 'blood_pressure'],
                    'cost_tier': 1
                }
            ],
            'Type 2 Diabetes': [
                {
                    'name': 'Metformin', 'dosage': '500mg', 'frequency': 'twice daily', 
                    'category': 'Biguanide', 'max_dose': '2000mg',
                    'contraindications': ['kidney_disease', 'heart_failure', 'liver_disease'],
                    'monitoring': ['kidney_function', 'hba1c', 'vitamin_b12'],
                    'cost_tier': 1
                },
                {
                    'name': 'Glipizide', 'dosage': '5mg', 'frequency': 'twice daily', 
                    'category': 'Sulfonylurea', 'max_dose': '20mg',
                    'contraindications': ['type_1_diabetes', 'diabetic_ketoacidosis'],
                    'monitoring': ['blood_glucose', 'hba1c'],
                    'cost_tier': 1
                }
            ],
            'Asthma': [
                {
                    'name': 'Albuterol', 'dosage': '90mcg', 'frequency': 'as needed', 
                    'category': 'SABA', 'max_dose': '720mcg/day',
                    'contraindications': ['hypersensitivity'],
                    'monitoring': ['peak_flow', 'heart_rate'],
                    'cost_tier': 2
                },
                {
                    'name': 'Fluticasone', 'dosage': '110mcg', 'frequency': 'twice daily', 
                    'category': 'ICS', 'max_dose': '880mcg',
                    'contraindications': ['fungal_infections'],
                    'monitoring': ['peak_flow', 'growth_in_children'],
                    'cost_tier': 2
                }
            ]
        }
    
    def _load_drug_interactions(self) -> Dict:
        """Enhanced drug interaction matrix with severity levels"""
        return {
            'Lisinopril': [
                {'drug': 'Potassium supplements', 'severity': 'major', 'mechanism': 'hyperkalemia'},
                {'drug': 'Spironolactone', 'severity': 'major', 'mechanism': 'hyperkalemia'}
            ],
            'Metformin': [
                {'drug': 'Iodinated contrast', 'severity': 'major', 'mechanism': 'lactic_acidosis'},
                {'drug': 'Alcohol', 'severity': 'moderate', 'mechanism': 'lactic_acidosis'}
            ],
            'Warfarin': [
                {'drug': 'Aspirin', 'severity': 'major', 'mechanism': 'bleeding_risk'},
                {'drug': 'Ibuprofen', 'severity': 'major', 'mechanism': 'bleeding_risk'}
            ]
        }
    
    def _load_contraindications(self) -> Dict:
        """Load contraindications for conditions and medicines"""
        return {
            'pregnancy': ['Lisinopril', 'Warfarin', 'Methotrexate'],
            'kidney_disease': ['Metformin', 'NSAIDs'],
            'liver_disease': ['Metformin', 'Statins'],
            'asthma': ['Beta Blockers', 'Aspirin']
        }
    
    def create_enhanced_training_data(self, n_samples: int = 2000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create more realistic synthetic training data"""
        np.random.seed(42)
        
        conditions_list = list(self.medicine_database.keys())
        training_data = []
        
        # Age and condition probability distributions
        age_condition_probs = {
            'Hypertension': {'young': 0.1, 'middle': 0.3, 'old': 0.6},
            'Type 2 Diabetes': {'young': 0.05, 'middle': 0.25, 'old': 0.4},
            'Asthma': {'young': 0.15, 'middle': 0.1, 'old': 0.08}
        }
        
        for _ in range(n_samples):
            # Generate realistic patient profile
            age = np.random.randint(18, 90)
            age_group = 'young' if age < 40 else 'middle' if age < 65 else 'old'
            
            weight = np.random.normal(75, 20)
            weight = max(40, min(150, weight))  # Realistic weight bounds
            
            # BMI calculation
            height = np.random.normal(170, 15)  # cm
            bmi = weight / ((height/100) ** 2)
            
            # Gender affects some conditions
            gender = np.random.choice(['M', 'F'])
            
            # Smoking status
            smoking = np.random.choice([0, 1], p=[0.7, 0.3])
            
            # Generate conditions based on age and demographics
            conditions = []
            for condition in conditions_list:
                prob = age_condition_probs.get(condition, {'young': 0.1, 'middle': 0.2, 'old': 0.3})[age_group]
                if np.random.random() < prob:
                    conditions.append(condition)
            
            if not conditions:  # Ensure at least one condition
                conditions = [np.random.choice(conditions_list)]
            
            # Generate symptoms with noise
            all_symptoms = self._generate_symptoms_for_conditions(conditions, age, gender)
            
            # Generate lab values
            lab_values = self._generate_lab_values(conditions, age)
            
            # Generate prescriptions with more realistic logic
            prescriptions = self._generate_realistic_prescriptions(conditions, age, weight, lab_values)
            
            training_data.append({
                'age': age,
                'weight': weight,
                'bmi': bmi,
                'gender': gender,
                'smoking': smoking,
                'conditions': '|'.join(conditions),
                'symptoms': '|'.join(all_symptoms),
                'lab_values': json.dumps(lab_values),
                'prescriptions': '|'.join(prescriptions)
            })
        
        df = pd.DataFrame(training_data)
        
        # Enhanced features
        features_df = df[['age', 'weight', 'bmi', 'gender', 'smoking', 
                         'conditions', 'symptoms', 'lab_values']].copy()
        targets_df = df[['prescriptions']].copy()
        
        return features_df, targets_df
    
    def _generate_symptoms_for_conditions(self, conditions: List[str], age: int, gender: str) -> List[str]:
        """Generate realistic symptoms based on conditions and demographics"""
        symptom_mapping = {
            'Hypertension': {
                'common': ['headache', 'dizziness', 'fatigue'],
                'severe': ['chest_pain', 'shortness_of_breath', 'visual_changes'],
                'elderly': ['confusion', 'falls']
            },
            'Type 2 Diabetes': {
                'common': ['excessive_thirst', 'frequent_urination', 'fatigue'],
                'severe': ['blurred_vision', 'slow_healing_wounds', 'numbness'],
                'elderly': ['confusion', 'infections']
            },
            'Asthma': {
                'common': ['shortness_of_breath', 'wheezing', 'chest_tightness'],
                'severe': ['difficulty_speaking', 'blue_lips', 'severe_wheezing'],
                'elderly': ['reduced_exercise_tolerance']
            }
        }
        
        all_symptoms = []
        for condition in conditions:
            if condition in symptom_mapping:
                # Common symptoms (high probability)
                common_symptoms = np.random.choice(
                    symptom_mapping[condition]['common'], 
                    np.random.randint(1, 3), 
                    replace=False
                )
                all_symptoms.extend(common_symptoms)
                
                # Age-specific symptoms
                if age > 65 and 'elderly' in symptom_mapping[condition]:
                    if np.random.random() < 0.3:
                        elderly_symptom = np.random.choice(symptom_mapping[condition]['elderly'])
                        all_symptoms.append(elderly_symptom)
                
                # Severe symptoms (lower probability)
                if np.random.random() < 0.2:
                    severe_symptom = np.random.choice(symptom_mapping[condition]['severe'])
                    all_symptoms.append(severe_symptom)
        
        return list(set(all_symptoms))  # Remove duplicates
    
    def _generate_lab_values(self, conditions: List[str], age: int) -> Dict:
        """Generate realistic lab values based on conditions"""
        lab_values = {}
        
        # Default normal ranges
        lab_values['glucose'] = np.random.normal(90, 10)
        lab_values['hba1c'] = np.random.normal(5.2, 0.3)
        lab_values['cholesterol'] = np.random.normal(180, 30)
        lab_values['creatinine'] = np.random.normal(1.0, 0.2)
        
        # Modify based on conditions
        if 'Type 2 Diabetes' in conditions:
            lab_values['glucose'] = np.random.normal(150, 40)
            lab_values['hba1c'] = np.random.normal(8.0, 1.5)
        
        if 'High Cholesterol' in conditions:
            lab_values['cholesterol'] = np.random.normal(250, 50)
        
        # Age adjustments
        if age > 65:
            lab_values['creatinine'] = np.random.normal(1.2, 0.3)
        
        return lab_values
    
    def _generate_realistic_prescriptions(self, conditions: List[str], age: int, 
                                        weight: float, lab_values: Dict) -> List[str]:
        """Generate more realistic prescriptions based on guidelines"""
        prescriptions = []
        
        for condition in conditions:
            if condition in self.medicine_database:
                # First-line treatments
                first_line = self._get_first_line_treatment(condition, age, lab_values)
                prescriptions.extend(first_line)
                
                # Add second-line if needed (based on severity)
                if np.random.random() < 0.3:  # 30% chance of needing second-line
                    second_line = self._get_second_line_treatment(condition, age)
                    prescriptions.extend(second_line)
        
        return list(set(prescriptions))  # Remove duplicates
    
    def _get_first_line_treatment(self, condition: str, age: int, lab_values: Dict) -> List[str]:
        """Get first-line treatment based on guidelines"""
        first_line_map = {
            'Hypertension': ['Lisinopril'] if age < 65 else ['Amlodipine'],
            'Type 2 Diabetes': ['Metformin'] if lab_values.get('creatinine', 1.0) < 1.5 else ['Glipizide'],
            'Asthma': ['Albuterol']
        }
        
        return first_line_map.get(condition, [])
    
    def _get_second_line_treatment(self, condition: str, age: int) -> List[str]:
        """Get second-line treatment options"""
        second_line_map = {
            'Hypertension': ['Metoprolol', 'Hydrochlorothiazide'],
            'Type 2 Diabetes': ['Glipizide', 'Sitagliptin'],
            'Asthma': ['Fluticasone']
        }
        
        options = second_line_map.get(condition, [])
        return [np.random.choice(options)] if options else []
    
    def preprocess_enhanced_features(self, features_df: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """Enhanced feature preprocessing"""
        # Text features
        conditions_text = features_df['conditions'].fillna('')
        symptoms_text = features_df['symptoms'].fillna('')
        
        if fit:
            self.vectorizers['conditions'] = TfidfVectorizer(max_features=50, ngram_range=(1, 2))
            self.vectorizers['symptoms'] = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
            self.encoders['gender'] = LabelEncoder()
            self.scalers['numerical'] = StandardScaler()
            
            conditions_features = self.vectorizers['conditions'].fit_transform(conditions_text)
            symptoms_features = self.vectorizers['symptoms'].fit_transform(symptoms_text)
            
            # Encode categorical features
            gender_encoded = self.encoders['gender'].fit_transform(features_df['gender'])
        else:
            conditions_features = self.vectorizers['conditions'].transform(conditions_text)
            symptoms_features = self.vectorizers['symptoms'].transform(symptoms_text)
            gender_encoded = self.encoders['gender'].transform(features_df['gender'])
        
        # Process lab values
        lab_features = []
        for _, row in features_df.iterrows():
            lab_data = json.loads(row['lab_values']) if pd.notna(row['lab_values']) else {}
            lab_vector = [
                lab_data.get('glucose', 90),
                lab_data.get('hba1c', 5.2),
                lab_data.get('cholesterol', 180),
                lab_data.get('creatinine', 1.0)
            ]
            lab_features.append(lab_vector)
        
        lab_features = np.array(lab_features)
        
        # Numerical features
        numerical_features = np.column_stack([
            features_df['age'].fillna(50),
            features_df['weight'].fillna(70),
            features_df['bmi'].fillna(25),
            features_df['smoking'].fillna(0),
            gender_encoded,
            lab_features
        ])
        
        if fit:
            numerical_features = self.scalers['numerical'].fit_transform(numerical_features)
        else:
            numerical_features = self.scalers['numerical'].transform(numerical_features)
        
        # Combine all features
        combined_features = np.hstack([
            numerical_features,
            conditions_features.toarray(),
            symptoms_features.toarray()
        ])
        
        return combined_features
    
    def train_enhanced_model(self, use_ensemble: bool = True):
        """Train enhanced model with better algorithms"""
        logger.info("Generating enhanced training data...")
        features_df, targets_df = self.create_enhanced_training_data(n_samples=5000)
        
        logger.info("Preprocessing features...")
        X = self.preprocess_enhanced_features(features_df, fit=True)
        
        # Get all unique medicines
        all_medicines = set()
        for prescriptions in targets_df['prescriptions']:
            medicines = prescriptions.split('|')
            all_medicines.update(medicines)
        
        logger.info(f"Training models for {len(all_medicines)} medicines...")
        
        # Train models
        for medicine in all_medicines:
            y = targets_df['prescriptions'].apply(lambda x: 1 if medicine in x.split('|') else 0)
            
            # Skip if not enough positive samples
            if y.sum() < 10:
                logger.warning(f"Skipping {medicine} - insufficient positive samples")
                continue
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            if use_ensemble:
                # Use ensemble of models
                rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                
                # Train both models
                rf_model.fit(X_train, y_train)
                gb_model.fit(X_train, y_train)
                
                # Simple ensemble averaging
                rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
                gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]
                ensemble_proba = (rf_pred_proba + gb_pred_proba) / 2
                
                # Store both models
                self.models[medicine] = {
                    'rf': rf_model,
                    'gb': gb_model,
                    'type': 'ensemble'
                }
                
                # Evaluate ensemble
                auc_score = roc_auc_score(y_test, ensemble_proba)
                logger.info(f"Model for {medicine}: AUC = {auc_score:.3f}")
            else:
                # Single model
                model = RandomForestClassifier(
                    n_estimators=200, 
                    random_state=42, 
                    class_weight='balanced',
                    max_depth=10
                )
                model.fit(X_train, y_train)
                
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_pred_proba)
                
                self.models[medicine] = {
                    'model': model,
                    'type': 'single'
                }
                
                logger.info(f"Model for {medicine}: AUC = {auc_score:.3f}")
        
        logger.info("Enhanced training completed!")
    
    def generate_enhanced_prescription(self, patient_data: Dict, doctor_id: str = None, 
                                     consider_cost: bool = True) -> List[Dict]:
        """Generate enhanced prescription with safety checks"""
        if not self.models:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Enhanced patient data processing
        enhanced_patient_data = self._enhance_patient_data(patient_data)
        
        # Convert to DataFrame
        patient_df = pd.DataFrame([enhanced_patient_data])
        
        # Preprocess
        X = self.preprocess_enhanced_features(patient_df, fit=False)
        
        # Get predictions
        predictions = []
        for medicine, model_info in self.models.items():
            if model_info['type'] == 'ensemble':
                rf_proba = model_info['rf'].predict_proba(X)[0][1]
                gb_proba = model_info['gb'].predict_proba(X)[0][1]
                probability = (rf_proba + gb_proba) / 2
            else:
                probability = model_info['model'].predict_proba(X)[0][1]
            
            # Apply doctor preferences
            if doctor_id and doctor_id in self.doctor_preferences:
                doctor_pref = self.doctor_preferences[doctor_id].get(medicine, 1.0)
                probability *= doctor_pref
            
            if probability > 0.2:  # Lower threshold for better recall
                medicine_details = self._get_medicine_details(medicine)
                
                # Calculate appropriate dosage
                adjusted_dosage = self.dosage_calculator.calculate_dose(
                    medicine, patient_data.get('age', 50), patient_data.get('weight', 70),
                    patient_data.get('kidney_function', 'normal')
                )
                
                predictions.append({
                    'medicine': medicine,
                    'probability': probability,
                    'dosage': adjusted_dosage or medicine_details.get('dosage', 'N/A'),
                    'frequency': medicine_details.get('frequency', 'N/A'),
                    'category': medicine_details.get('category', 'N/A'),
                    'cost_tier': medicine_details.get('cost_tier', 2),
                    'monitoring_required': medicine_details.get('monitoring', [])
                })
        
        # Sort by probability
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        # Enhanced safety checks
        safe_prescriptions = self._comprehensive_safety_check(predictions, patient_data)
        
        # Consider cost if requested
        if consider_cost:
            safe_prescriptions = self._optimize_for_cost(safe_prescriptions)
        
        self.is_trained = True
        return safe_prescriptions[:5]
    
    def _enhance_patient_data(self, patient_data: Dict) -> Dict:
        """Enhance patient data with calculated fields"""
        enhanced = patient_data.copy()
        
        # Calculate BMI if height and weight available
        if 'height' in patient_data and 'weight' in patient_data:
            height_m = patient_data['height'] / 100
            enhanced['bmi'] = patient_data['weight'] / (height_m ** 2)
        
        # Default values for missing fields
        enhanced.setdefault('gender', 'M')
        enhanced.setdefault('smoking', 0)
        enhanced.setdefault('lab_values', json.dumps({
            'glucose': 90, 'hba1c': 5.2, 'cholesterol': 180, 'creatinine': 1.0
        }))
        
        return enhanced
    
    def _comprehensive_safety_check(self, prescriptions: List[Dict], patient_data: Dict) -> List[Dict]:
        """Comprehensive safety checking"""
        safe_prescriptions = []
        prescribed_medicines = [p['medicine'] for p in prescriptions]
        
        # Check contraindications
        patient_conditions = patient_data.get('contraindications', [])
        
        for prescription in prescriptions:
            medicine = prescription['medicine']
            safety_flags = []
            
            # Drug interactions
            interactions = self._check_detailed_interactions(medicine, prescribed_medicines)
            
            # Contraindications
            contraindications = self._check_contraindications(medicine, patient_conditions)
            
            # Age-related warnings
            age_warnings = self._check_age_warnings(medicine, patient_data.get('age', 50))
            
            prescription.update({
                'interactions': interactions,
                'contraindications': contraindications,
                'age_warnings': age_warnings,
                'safety_score': self._calculate_safety_score(interactions, contraindications, age_warnings)
            })
            
            # Only include if safety score is acceptable
            if prescription['safety_score'] >= 0.5:  # 0.5 threshold for safety
                safe_prescriptions.append(prescription)
        
        return safe_prescriptions
    
    def _check_detailed_interactions(self, medicine: str, other_medicines: List[str]) -> List[Dict]:
        """Check for detailed drug interactions"""
        interactions = []
        
        if medicine in self.interaction_matrix:
            for interaction in self.interaction_matrix[medicine]:
                if interaction['drug'] in other_medicines:
                    interactions.append(interaction)
        
        return interactions
    
    def _check_contraindications(self, medicine: str, patient_conditions: List[str]) -> List[str]:
        """Check contraindications"""
        contraindications = []
        medicine_details = self._get_medicine_details(medicine)
        
        med_contraindications = medicine_details.get('contraindications', [])
        for condition in patient_conditions:
            if condition in med_contraindications:
                contraindications.append(condition)
        
        return contraindications
    
    def _check_age_warnings(self, medicine: str, age: int) -> List[str]:
        """Check age-related warnings"""
        warnings = []
        
        # Beers criteria for elderly (simplified)
        if age >= 65:
            high_risk_elderly = ['Lorazepam', 'Diphenhydramine', 'Amitriptyline']
            if medicine in high_risk_elderly:
                warnings.append('Potentially inappropriate for elderly')
        
        # Pediatric warnings
        if age < 18:
            avoid_pediatric = ['Aspirin', 'Tetracycline', 'Quinolones']
            if medicine in avoid_pediatric:
                warnings.append('Not recommended for pediatric use')
        
        return warnings
    
    def _calculate_safety_score(self, interactions: List[Dict], 
                               contraindications: List[str], age_warnings: List[str]) -> float:
        """Calculate overall safety score (0-1)"""
        score = 1.0
        
        # Penalize based on interactions
        for interaction in interactions:
            if interaction['severity'] == 'major':
                score -= 0.3
            elif interaction['severity'] == 'moderate':
                score -= 0.1
        
        # Penalize contraindications
        score -= len(contraindications) * 0.2
        
        # Penalize age warnings
        score -= len(age_warnings) * 0.1
        
        return max(0.0, score)
    
    def _optimize_for_cost(self, prescriptions: List[Dict]) -> List[Dict]:
        """Optimize prescriptions considering cost"""
        # Sort by cost tier (lower is cheaper) and then by probability
        prescriptions.sort(key=lambda x: (x['cost_tier'], -x['probability']))
        return prescriptions
    
    def log_prescription(self, patient_id: str, doctor_id: str, 
                        prescription: List[Dict], outcome: str = None):
        """Log prescription for audit trail"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'patient_id': patient_id,
            'doctor_id': doctor_id,
            'prescription': prescription,
            'outcome': outcome
        }
        self.audit_log.append(log_entry)


class DosageCalculator:
    """Helper class for dosage calculations"""
    
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
        """Calculate appropriate dose based on patient factors"""
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
        
        # Apply adjustment
        adjusted_dose = self._apply_dose_adjustment(base_dose, adjustment_factor)
        return adjusted_dose
    
    def _get_base_dose(self, medicine: str) -> Optional[str]:
        """Get base dose for medicine"""
        dose_map = {
            'Lisinopril': '10mg',
            'Metformin': '500mg',
            'Amlodipine': '5mg'
        }
        return dose_map.get(medicine)
    
    def _apply_dose_adjustment(self, base_dose: str, factor: float) -> str:
        """Apply dose adjustment factor"""
        import re
        
        # Extract numeric value and unit
        match = re.match(r'(\d+(?:\.\d+)?)(.*)', base_dose)
        if not match:
            return base_dose
        
        dose_value = float(match.group(1))
        unit = match.group(2)
        
        # Apply adjustment
        adjusted_value = dose_value * factor
        
        # Round to reasonable precision
        if adjusted_value < 1:
            adjusted_value = round(adjusted_value, 2)
        else:
            adjusted_value = round(adjusted_value, 1)
        
        return f"{adjusted_value}{unit}"


# Enhanced Flask API with additional endpoints
from flask import Flask, request, jsonify
from flask_cors import CORS
import traceback


# Initialize the enhanced prescription generator
prescription_generator = EnhancedPrescriptionGenerator()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/train', methods=['POST'])
def train_model():
    """Train the enhanced prescription model"""
    try:
        data = request.json or {}
        use_ensemble = data.get('use_ensemble', True)
        n_samples = data.get('n_samples', 5000)
        
        # Update training data size if specified
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

@app.route('/generate_prescription', methods=['POST', 'OPTIONS'])
def generate_prescription():
    global prescription_generator
    global prescription_generator
    """Generate enhanced prescription for a patient"""
    try:
        data = request.json
        patient_data = data.get('patient_data', {})
        doctor_id = data.get('doctor_id', 'default')
        consider_cost = data.get('consider_cost', True)
        
        prescription = prescription_generator.generate_enhanced_prescription(
            patient_data, doctor_id, consider_cost
        )
        
        # Log the prescription
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
    """Check drug interactions for a list of medicines"""
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
    """Calculate adjusted dosage for a patient"""
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
    """Get detailed information about a medicine"""
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
    """Update doctor preferences based on modifications"""
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
    """Get audit log entries"""
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
    """Get system statistics"""
    try:
        stats = {
            'total_medicines': len(prescription_generator.medicine_database),
            'trained_models': len(prescription_generator.models),
            'doctors_with_preferences': len(prescription_generator.doctor_preferences),
            'total_prescriptions_logged': len(prescription_generator.audit_log),
            'medicine_categories': {}
        }
        
        # Count medicines by category
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
    """Validate a prescription for safety and appropriateness"""
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
        
        # Check each medicine in prescription
        medicine_names = [med.get('medicine') for med in prescription if 'medicine' in med]
        
        for medicine_info in prescription:
            medicine = medicine_info.get('medicine')
            
            # Check interactions
            interactions = prescription_generator._check_detailed_interactions(medicine, medicine_names)
            if interactions:
                for interaction in interactions:
                    if interaction['severity'] == 'major':
                        validation_results['errors'].append(f"Major interaction: {medicine} with {interaction['drug']}")
                        validation_results['is_valid'] = False
                    else:
                        validation_results['warnings'].append(f"Moderate interaction: {medicine} with {interaction['drug']}")
            
            # Check contraindications
            contraindications = prescription_generator._check_contraindications(
                medicine, patient_data.get('contraindications', [])
            )
            if contraindications:
                validation_results['errors'].append(f"Contraindication: {medicine} with {', '.join(contraindications)}")
                validation_results['is_valid'] = False
            
            # Check age warnings
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
    """Save the trained model"""
    try:
        filepath = request.json.get('filepath', 'enhanced_prescription_model.pkl')
        prescription_generator.save_model(filepath)
        return jsonify({'status': 'success', 'message': f'Model saved to {filepath}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/load_model', methods=['POST'])
def load_model():
    """Load a pre-trained model"""
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
    global prescription_generator
    global prescription_generator
    prescription_generator = EnhancedPrescriptionGenerator()
    print("Training model on startup...")
    try:
        prescription_generator.train_enhanced_model(use_ensemble=True)
        print("Model training complete.")
    except Exception as e:
        print(f"Startup training failed: {e}")
    prescription_generator = EnhancedPrescriptionGenerator()
    print("Training model on startup...")
    try:
        prescription_generator.train_enhanced_model(use_ensemble=True)
        print("Model training complete.")
    except Exception as e:
        print(f"Startup training failed: {e}")
    prescription_generator = EnhancedPrescriptionGenerator()

    # Train the enhanced model on startup
    print("Starting Enhanced Prescription Generator...")
    print("Training enhanced model (this may take a few minutes)...")
    
    try:
        
        print("Model training completed successfully!")
    except Exception as e:
        print(f"Warning: Model training failed: {e}")
        print("Server will start without pre-trained model.")
    
    # Start the Flask server
    print("Starting Flask server on port 5000...")
    app.run(debug=True, port=5000, host='0.0.0.0')

@app.route('/train', methods=['POST'])
def train_model():
    try:
        prescription_generator.train_enhanced_model(use_ensemble=True)
        return jsonify({'message': 'Model trained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server on port 5000...")
    app.run(debug=True, port=5000, host='0.0.0.0')
