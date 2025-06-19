import React, { useState, useMemo, useEffect } from 'react';
import { 
  User, Calendar, FileText, Heart, Pill, AlertCircle, Search, Edit3, 
  Phone, Mail, MapPin, Activity, TrendingUp, Shield, AlertTriangle, 
  Zap, Clock, Filter, Plus, Bell, Settings, Trash2, ChevronDown, ChevronUp, X,
  Printer, Menu, Loader2, CheckCircle2, AlertOctagon
} from 'lucide-react';

const MedicalInterface = () => {
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [riskFilter, setRiskFilter] = useState('all');
  const [sortBy, setSortBy] = useState('name');
  const [isAddingPatient, setIsAddingPatient] = useState(false);
  const [isEditingPatient, setIsEditingPatient] = useState(false);
  const [newPatient, setNewPatient] = useState(getBlankPatient());
  const [showPatientForm, setShowPatientForm] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [showPrescription, setShowPrescription] = useState(false);
  const [isGeneratingPrescription, setIsGeneratingPrescription] = useState(false);
  const [prescriptionError, setPrescriptionError] = useState(null);
  const [generatedPrescription, setGeneratedPrescription] = useState([]);
  const [interactionWarnings, setInteractionWarnings] = useState([]);
  const [contraindicationWarnings, setContraindicationWarnings] = useState([]);
  
  const [patients, setPatients] = useState([
    {
      id: 1,
      name: "Sarah Johnson",
      age: 34,
      gender: "Female",
      phone: "+1 (555) 123-4567",
      email: "sarah.johnson@email.com",
      address: "123 Main St, New York, NY 10001",
      diagnosis: "Hypertension, Type 2 Diabetes",
      bloodGroup: "A+",
      allergies: ["Penicillin", "Shellfish"],
      contraindications: ["pregnancy", "kidney_disease"],
      lastVisit: "2024-05-15",
      nextAppointment: "2024-06-20",
      riskLevel: "high",
      riskScore: 85,
      riskFactors: ["Uncontrolled BP", "Diabetes complications", "Medication non-compliance"],
      criticalAlerts: ["BP spike last week", "Missed 2 appointments"],
      vitals: {
        bp: "160/95 mmHg",
        pulse: "88 bpm",
        temp: "98.6°F",
        weight: "175 lbs",
        height: "5'6\"",
        lastUpdated: "2024-06-10"
      },
      labResults: {
        hba1c: "8.2%",
        cholesterol: "245 mg/dL",
        creatinine: "1.4 mg/dL",
        status: "concerning"
      },
      medicalHistory: [
        { date: "2024-05-15", condition: "Emergency Visit", notes: "Severe hypertension episode, adjusted medications", severity: "high" },
        { date: "2024-03-10", condition: "Diabetes Follow-up", notes: "HbA1c elevated, lifestyle counseling provided", severity: "medium" },
        { date: "2024-01-22", condition: "Annual Physical", notes: "Multiple risk factors identified", severity: "medium" }
      ],
      currentMedications: [
        { name: "Metformin", dosage: "1000mg", frequency: "Twice daily", prescribed: "2024-01-15", adherence: "poor" },
        { name: "Amlodipine", dosage: "10mg", frequency: "Once daily", prescribed: "2024-05-15", adherence: "good" },
        { name: "Atorvastatin", dosage: "40mg", frequency: "Once daily", prescribed: "2024-01-15", adherence: "fair" }
      ],
      compliance: 65,
      priority: "urgent"
    },
    {
      id: 2,
      name: "Michael Chen",
      age: 42,
      gender: "Male",
      phone: "+1 (555) 987-6543",
      email: "m.chen@email.com",
      address: "456 Oak Ave, Los Angeles, CA 90210",
      diagnosis: "Asthma, Anxiety Disorder",
      bloodGroup: "O-",
      allergies: ["Aspirin"],
      contraindications: ["asthma"],
      lastVisit: "2024-05-20",
      nextAppointment: "2024-07-15",
      riskLevel: "moderate",
      riskScore: 45,
      riskFactors: ["Stress-induced asthma", "Work-related anxiety"],
      criticalAlerts: [],
      vitals: {
        bp: "125/82 mmHg",
        pulse: "75 bpm",
        temp: "98.4°F",
        weight: "180 lbs",
        height: "5'10\"",
        lastUpdated: "2024-06-08"
      },
      labResults: {
        ige: "150 IU/mL",
        cortisol: "12 μg/dL",
        status: "stable"
      },
      medicalHistory: [
        { date: "2024-05-20", condition: "Asthma Management", notes: "Symptoms well controlled with current regimen", severity: "low" },
        { date: "2024-04-05", condition: "Anxiety Consultation", notes: "Therapy showing positive results", severity: "medium" },
        { date: "2024-02-18", condition: "Respiratory Issues", notes: "Acute exacerbation resolved", severity: "medium" }
      ],
      currentMedications: [
        { name: "Albuterol Inhaler", dosage: "90mcg", frequency: "As needed", prescribed: "2024-02-18", adherence: "excellent" },
        { name: "Sertraline", dosage: "50mg", frequency: "Once daily", prescribed: "2024-04-05", adherence: "good" },
        { name: "Fluticasone", dosage: "110mcg", frequency: "Twice daily", prescribed: "2024-02-18", adherence: "good" }
      ],
      compliance: 85,
      priority: "routine"
    }
  ]);

  function getBlankPatient() {
    return {
      id: Date.now(),
      name: "",
      age: "",
      gender: "",
      phone: "",
      email: "",
      address: "",
      diagnosis: "",
      bloodGroup: "",
      allergies: [],
      contraindications: [],
      lastVisit: new Date().toISOString().split('T')[0],
      nextAppointment: "",
      riskLevel: "low",
      riskScore: 0,
      riskFactors: [],
      criticalAlerts: [],
      vitals: {
        bp: "",
        pulse: "",
        temp: "",
        weight: "",
        height: "",
        lastUpdated: new Date().toISOString().split('T')[0]
      },
      labResults: {
        status: "stable"
      },
      medicalHistory: [],
      currentMedications: [],
      compliance: 0,
      priority: "routine"
    };
  }

  useEffect(() => {
  const checkBackendHealth = async () => {
    try {
      const response = await fetch('https://cohort-hiring-3.onrender.com/health');
      if (!response.ok) {
        console.error('Backend health check failed');
      }
    } catch (error) {
      console.error('Error checking backend health:', error);
    }
  };
  
  checkBackendHealth();
}, []);

  const generatePrescription = async () => {
  if (!selectedPatient) return;
  
  setIsGeneratingPrescription(true);
  setPrescriptionError(null);
  setGeneratedPrescription([]);
  setInteractionWarnings([]);
  setContraindicationWarnings([]);
  
  try {
    // Prepare patient data for the ML model
    const patientData = {
      patient_id: selectedPatient.id,
      age: selectedPatient.age,
      weight: parseFloat(selectedPatient.vitals.weight),
      height: parseFloat(selectedPatient.vitals.height.replace(/[^\d.]/g, '')), // Ensure height is parsed correctly
      gender: selectedPatient.gender.toLowerCase(),
      conditions: selectedPatient.diagnosis.split(',').map(d => d.trim()),
      contraindications: selectedPatient.contraindications || [],
      lab_values: {
        glucose: selectedPatient.labResults.hba1c ? parseFloat(selectedPatient.labResults.hba1c) * 10 : 90,
        hba1c: selectedPatient.labResults.hba1c ? parseFloat(selectedPatient.labResults.hba1c) : 5.2,
        cholesterol: selectedPatient.labResults.cholesterol ? parseFloat(selectedPatient.labResults.cholesterol) : 180,
        creatinine: selectedPatient.labResults.creatinine ? parseFloat(selectedPatient.labResults.creatinine) : 1.0
      },
      kidney_function: selectedPatient.labResults.creatinine > 1.5 ? "impaired" : "normal",
      symptoms: [] // Add empty symptoms array if not provided
    };

    // Add debug logging
    console.log("Sending patient data:", patientData);
    // In your generatePrescription function, before the fetch call:
if (!patientData.conditions || patientData.conditions.length === 0) {
  setPrescriptionError('Patient must have at least one condition');
  setIsGeneratingPrescription(false);
  return;
}

if (isNaN(patientData.weight)) {
  setPrescriptionError('Invalid weight value');
  setIsGeneratingPrescription(false);
  return;
}

    // Call the Flask API to generate prescription
    const response = await fetch('https://cohort-hiring-3.onrender.com/generate_prescription', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        patient_data: patientData,
        doctor_id: "default",
        consider_cost: true
      })
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log("Received response:", data);
    
    if (data.status === 'success') {
      setGeneratedPrescription(data.prescription);
      
      // Check for interactions and contraindications
      checkPrescriptionSafety(data.prescription, patientData);
      
      setShowPrescription(true);
    } else {
      throw new Error(data.message || 'Failed to generate prescription');
    }
  } catch (error) {
    console.error('Error generating prescription:', error);
    setPrescriptionError(error.message);
  } finally {
    setIsGeneratingPrescription(false);
  }
};

  const checkPrescriptionSafety = async (prescription, patientData) => {
    const medicineNames = prescription.map(med => med.medicine);
    
    // Check interactions
    try {
      const interactionsResponse = await fetch('https://cohort-hiring-3.onrender.com/check_interactions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          medicines: medicineNames
        })
      });
      
      if (interactionsResponse.ok) {
        const interactionsData = await interactionsResponse.json();
        setInteractionWarnings(interactionsData.interactions || []);
      }
    } catch (error) {
      console.error('Error checking interactions:', error);
    }
    
    // Check contraindications
    const contraindications = [];
    prescription.forEach(med => {
      patientData.contraindications.forEach(condition => {
        if (med.contraindications && med.contraindications.includes(condition)) {
          contraindications.push({
            medicine: med.medicine,
            condition: condition
          });
        }
      });
    });
    
    setContraindicationWarnings(contraindications);
  };

  const calculateDosage = async (medicine, age, weight, kidneyFunction) => {
    try {
      const response = await fetch('https://cohort-hiring-3.onrender.com/calculate_dosage', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          medicine,
          age,
          weight,
          kidney_function: kidneyFunction
        })
      });

      if (response.ok) {
        const data = await response.json();
        return data.adjusted_dose;
      }
    } catch (error) {
      console.error('Error calculating dosage:', error);
    }
    return null;
  };

  const handleAddPatient = () => {
    setIsAddingPatient(true);
    setIsEditingPatient(false);
    setNewPatient(getBlankPatient());
    setShowPatientForm(true);
    setSelectedPatient(null);
    setIsSidebarOpen(false);
  };

  const handleEditPatient = () => {
    if (!selectedPatient) return;
    setIsEditingPatient(true);
    setIsAddingPatient(false);
    setNewPatient({
      ...selectedPatient,
      vitals: {...selectedPatient.vitals},
      labResults: {...selectedPatient.labResults},
      allergies: [...selectedPatient.allergies],
      contraindications: [...selectedPatient.contraindications || []],
      riskFactors: [...selectedPatient.riskFactors],
      criticalAlerts: [...selectedPatient.criticalAlerts],
      medicalHistory: selectedPatient.medicalHistory.map(item => ({...item})),
      currentMedications: selectedPatient.currentMedications.map(med => ({...med}))
    });
    setShowPatientForm(true);
  };

  const handleDeletePatient = (id) => {
    if (window.confirm("Are you sure you want to delete this patient?")) {
      setPatients(patients.filter(patient => patient.id !== id));
      if (selectedPatient && selectedPatient.id === id) {
        setSelectedPatient(null);
      }
    }
  };

  const handleSavePatient = () => {
    if (!newPatient.name || !newPatient.age || !newPatient.gender) {
      alert('Please fill in all required fields');
      return;
    }

    if (isAddingPatient) {
      const newPatientWithId = {
        ...newPatient,
        id: Date.now()
      };
      setPatients([...patients, newPatientWithId]);
      setSelectedPatient(newPatientWithId);
    } else if (isEditingPatient) {
      setPatients(patients.map(patient => 
        patient.id === newPatient.id ? newPatient : patient
      ));
      setSelectedPatient(newPatient);
    }
    setShowPatientForm(false);
  };

  const printPrescription = () => {
    window.print();
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setNewPatient({
      ...newPatient,
      [name]: value
    });
  };

  const handleNestedInputChange = (parent, e) => {
    const { name, value } = e.target;
    setNewPatient({
      ...newPatient,
      [parent]: {
        ...newPatient[parent],
        [name]: value
      }
    });
  };

  const handleArrayChange = (arrayName, index, field, value) => {
    const updatedArray = [...newPatient[arrayName]];
    updatedArray[index][field] = value;
    setNewPatient({
      ...newPatient,
      [arrayName]: updatedArray
    });
  };

  const handleAddArrayItem = (arrayName, item) => {
    setNewPatient({
      ...newPatient,
      [arrayName]: [...newPatient[arrayName], item]
    });
  };

  const handleRemoveArrayItem = (arrayName, index) => {
    const updatedArray = [...newPatient[arrayName]];
    updatedArray.splice(index, 1);
    setNewPatient({
      ...newPatient,
      [arrayName]: updatedArray
    });
  };

  const getRiskColor = (level) => {
    switch (level) {
      case 'critical': return 'text-red-400 bg-red-900';
      case 'high': return 'text-orange-400 bg-orange-900';
      case 'moderate': return 'text-yellow-400 bg-yellow-900';
      case 'low': return 'text-green-400 bg-green-900';
      default: return 'text-gray-400 bg-gray-700';
    }
  };

  const getRiskIcon = (level) => {
    switch (level) {
      case 'critical': return <Zap className="w-4 h-4" />;
      case 'high': return <AlertTriangle className="w-4 h-4" />;
      case 'moderate': return <TrendingUp className="w-4 h-4" />;
      case 'low': return <Shield className="w-4 h-4" />;
      default: return <Activity className="w-4 h-4" />;
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'critical': return 'border-l-red-500 bg-red-900/20';
      case 'urgent': return 'border-l-orange-500 bg-orange-900/20';
      case 'routine': return 'border-l-blue-500 bg-blue-900/20';
      default: return 'border-l-gray-500 bg-gray-800';
    }
  };

  const getComplianceColor = (compliance) => {
    if (compliance >= 90) return 'text-green-400';
    if (compliance >= 70) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getGradientColor = (score) => {
    if (score >= 80) return 'from-red-500 to-red-700';
    if (score >= 60) return 'from-orange-500 to-orange-700';
    if (score >= 40) return 'from-yellow-500 to-yellow-700';
    return 'from-green-500 to-green-700';
  };

  const filteredAndSortedPatients = useMemo(() => {
    let filtered = patients.filter(patient => {
      const matchesSearch = patient.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                           patient.diagnosis.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesRisk = riskFilter === 'all' || patient.riskLevel === riskFilter;
      return matchesSearch && matchesRisk;
    });

    return filtered.sort((a, b) => {
      switch (sortBy) {
        case 'risk':
          const riskOrder = { 'critical': 4, 'high': 3, 'moderate': 2, 'low': 1 };
          return riskOrder[b.riskLevel] - riskOrder[a.riskLevel];
        case 'lastVisit':
          return new Date(b.lastVisit) - new Date(a.lastVisit);
        case 'nextAppt':
          return new Date(a.nextAppointment) - new Date(b.nextAppointment);
        case 'compliance':
          return b.compliance - a.compliance;
        default:
          return a.name.localeCompare(b.name);
      }
    });
  }, [patients, searchTerm, riskFilter, sortBy]);

  const formatDate = (dateString) => {
    if (!dateString) return "Not scheduled";
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    });
  };

  const riskSummary = useMemo(() => {
    const summary = patients.reduce((acc, patient) => {
      acc[patient.riskLevel] = (acc[patient.riskLevel] || 0) + 1;
      return acc;
    }, {});
    return summary;
  }, [patients]);

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <header className="bg-gray-800 border-b border-gray-700 px-4 sm:px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-2 sm:space-x-4">
            <button 
              className="sm:hidden p-2 mr-1 text-gray-400 hover:text-white"
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            >
              {isSidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
            <div className="flex items-center justify-center w-10 h-10 sm:w-12 sm:h-12 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl">
              <Heart className="w-5 h-5 sm:w-7 sm:h-7 text-white" />
            </div>
            <div>
              <h1 className="text-lg sm:text-2xl font-bold text-white">MedCare Portal Pro</h1>
              <p className="text-xs sm:text-sm text-gray-400">AI-Powered Prescription System</p>
            </div>
          </div>
          
          <div className="flex items-center space-x-3 sm:space-x-6">
            <div className="hidden sm:flex items-center space-x-4">
              <div className="text-center">
                <div className="flex items-center space-x-1">
                  <Zap className="w-4 h-4 text-red-400" />
                  <span className="text-red-400 font-bold">{riskSummary.critical || 0}</span>
                </div>
                <p className="text-xs text-gray-400">Critical</p>
              </div>
              <div className="text-center">
                <div className="flex items-center space-x-1">
                  <AlertTriangle className="w-4 h-4 text-orange-400" />
                  <span className="text-orange-400 font-bold">{riskSummary.high || 0}</span>
                </div>
                <p className="text-xs text-gray-400">High</p>
              </div>
              <div className="text-center">
                <div className="flex items-center space-x-1">
                  <TrendingUp className="w-4 h-4 text-yellow-400" />
                  <span className="text-yellow-400 font-bold">{riskSummary.moderate || 0}</span>
                </div>
                <p className="text-xs text-gray-400">Moderate</p>
              </div>
              <div className="text-center">
                <div className="flex items-center space-x-1">
                  <Shield className="w-4 h-4 text-green-400" />
                  <span className="text-green-400 font-bold">{riskSummary.low || 0}</span>
                </div>
                <p className="text-xs text-gray-400">Low</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-2 sm:space-x-3">
              <button className="p-1 sm:p-2 hover:bg-gray-700 rounded-lg transition-colors">
                <Bell className="w-4 h-4 sm:w-5 sm:h-5 text-gray-400" />
              </button>
              <button className="p-1 sm:p-2 hover:bg-gray-700 rounded-lg transition-colors">
                <Settings className="w-4 h-4 sm:w-5 sm:h-5 text-gray-400" />
              </button>
              <div className="flex items-center space-x-2 sm:space-x-3">
                <div className="hidden sm:block text-right">
                  <p className="text-sm font-medium text-white">Dr. Amanda Smith</p>
                  <p className="text-xs text-gray-400">Internal Medicine • Risk Specialist</p>
                </div>
                <div className="w-8 h-8 sm:w-10 sm:h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                  <User className="w-4 h-4 sm:w-6 sm:h-6 text-white" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="flex flex-col sm:flex-row h-[calc(100vh-64px)]">
        {/* Sidebar */}
        <div className={`${isSidebarOpen ? 'block' : 'hidden'} sm:block w-full sm:w-96 bg-gray-800 border-r border-gray-700 flex flex-col`}>
          <div className="p-4 border-b border-gray-700 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">Patients ({patients.length})</h2>
              <button 
                onClick={handleAddPatient}
                className="flex items-center space-x-1 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 rounded-lg text-sm transition-colors"
              >
                <Plus className="w-4 h-4" />
                <span>Add Patient</span>
              </button>
            </div>
            
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <input
                type="text"
                placeholder="Search patients..."
                className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            
            <div className="flex space-x-2">
              <select
                value={riskFilter}
                onChange={(e) => setRiskFilter(e.target.value)}
                className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="all">All Risk Levels</option>
                <option value="critical">Critical</option>
                <option value="high">High</option>
                <option value="moderate">Moderate</option>
                <option value="low">Low</option>
              </select>
              
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value)}
                className="flex-1 px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="name">Name</option>
                <option value="risk">Risk Level</option>
                <option value="lastVisit">Last Visit</option>
                <option value="nextAppt">Next Appointment</option>
                <option value="compliance">Compliance</option>
              </select>
            </div>
          </div>
          
          <div className="flex-1 overflow-y-auto">
            {filteredAndSortedPatients.length === 0 ? (
              <div className="p-4 text-center text-gray-400">
                No patients found matching your criteria
              </div>
            ) : (
              filteredAndSortedPatients.map((patient) => (
                <div
                  key={patient.id}
                  onClick={() => {
                    setSelectedPatient(patient);
                    setIsSidebarOpen(false);
                  }}
                  className={`p-4 border-b border-gray-700 cursor-pointer transition-all duration-200 ${
                    selectedPatient?.id === patient.id
                      ? 'bg-blue-900 border-blue-600 shadow-lg'
                      : 'hover:bg-gray-700'
                  } ${getPriorityColor(patient.priority)} border-l-4`}
                >
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2">
                          <h3 className="font-semibold text-white">{patient.name}</h3>
                          {patient.criticalAlerts.length > 0 && (
                            <div className="flex items-center space-x-1">
                              <AlertCircle className="w-4 h-4 text-red-400" />
                              <span className="text-xs text-red-400">{patient.criticalAlerts.length}</span>
                            </div>
                          )}
                        </div>
                        <p className="text-sm text-gray-400">{patient.age} years • {patient.gender}</p>
                      </div>
                      <div className={`px-2 py-1 rounded-full text-xs font-medium flex items-center space-x-1 ${getRiskColor(patient.riskLevel)}`}>
                        {getRiskIcon(patient.riskLevel)}
                        <span className="capitalize">{patient.riskLevel}</span>
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-gray-400">Risk Score:</span>
                        <span className={`text-xs font-bold ${getRiskColor(patient.riskLevel).split(' ')[0]}`}>
                          {patient.riskScore}
                        </span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <span className="text-xs text-gray-400">Compliance:</span>
                        <span className={`text-xs font-bold ${getComplianceColor(patient.compliance)}`}>
                          {patient.compliance}%
                        </span>
                      </div>
                    </div>
                    
                    <p className="text-xs text-gray-500 truncate">{patient.diagnosis}</p>
                    
                    <div className="flex items-center justify-between">
                      <div className="text-right">
                        <p className="text-xs text-gray-400">Last visit</p>
                        <p className="text-xs text-gray-300">{formatDate(patient.lastVisit)}</p>
                      </div>
                      <div className="text-right">
                        <p className="text-xs text-gray-400">Next appt</p>
                        <p className="text-xs text-gray-300">{formatDate(patient.nextAppointment)}</p>
                      </div>
                    </div>
                    
                    {patient.criticalAlerts.length > 0 && (
                      <div className="mt-2 p-2 bg-red-900/30 rounded border-l-2 border-red-500">
                        <p className="text-xs text-red-300 font-medium">Critical Alerts:</p>
                        <p className="text-xs text-red-400">{patient.criticalAlerts[0]}</p>
                        {patient.criticalAlerts.length > 1 && (
                          <p className="text-xs text-red-400">+{patient.criticalAlerts.length - 1} more</p>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 overflow-hidden">
          {showPrescription ? (
            <div className="h-full overflow-y-auto p-4 sm:p-6 bg-white text-gray-800 print:p-0">
              <div className="max-w-4xl mx-auto p-6 border-2 border-gray-200 rounded-lg print:border-0 print:rounded-none">
                <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6">
                  <div>
                    <h2 className="text-2xl font-bold">Medical Prescription</h2>
                    <p className="text-gray-600">{new Date().toLocaleDateString()}</p>
                  </div>
                  <div className="flex space-x-2 mt-4 sm:mt-0">
                    <button 
                      onClick={() => setShowPrescription(false)}
                      className="px-4 py-2 bg-gray-200 hover:bg-gray-300 rounded"
                    >
                      Back
                    </button>
                    <button 
                      onClick={printPrescription}
                      className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded flex items-center"
                    >
                      <Printer className="w-4 h-4 mr-2" />
                      Print
                    </button>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                  <div>
                    <h3 className="text-lg font-semibold mb-2">Patient Information</h3>
                    <p className="font-medium">{selectedPatient.name}</p>
                    <p>{selectedPatient.age} years, {selectedPatient.gender}</p>
                    <p>Blood Group: {selectedPatient.bloodGroup}</p>
                    <p>Diagnosis: {selectedPatient.diagnosis}</p>
                    {selectedPatient.allergies.length > 0 && (
                      <p className="text-red-600 mt-2">
                        <strong>Allergies:</strong> {selectedPatient.allergies.join(', ')}
                      </p>
                    )}
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold mb-2">Prescribing Physician</h3>
                    <p className="font-medium">Dr. Amanda Smith</p>
                    <p>Internal Medicine Specialist</p>
                    <p>License: MED12345678</p>
                    <p>Contact: clinic@medcare.example.com</p>
                  </div>
                </div>
                
                {/* AI-Generated Prescription Section */}
                <div className="mb-8">
                  <h3 className="text-lg font-semibold mb-4 border-b pb-2">AI-Generated Prescription</h3>
                  
                  {isGeneratingPrescription ? (
                    <div className="flex items-center justify-center py-8">
                      <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
                      <span className="ml-2">Generating prescription...</span>
                    </div>
                  ) : prescriptionError ? (
                    <div className="bg-red-100 border-l-4 border-red-500 text-red-700 p-4 mb-4">
                      <p>Error generating prescription: {prescriptionError}</p>
                    </div>
                  ) : generatedPrescription.length > 0 ? (
                    <>
                      <div className="overflow-x-auto mb-6">
                        <table className="w-full border-collapse">
                          <thead>
                            <tr className="bg-gray-100">
                              <th className="p-2 text-left border">Medication</th>
                              <th className="p-2 text-left border">Dosage</th>
                              <th className="p-2 text-left border">Frequency</th>
                              <th className="p-2 text-left border">Category</th>
                              <th className="p-2 text-left border">Confidence</th>
                            </tr>
                          </thead>
                          <tbody>
                            {generatedPrescription.map((med, index) => (
                              <tr key={index}>
                                <td className="p-2 border">{med.medicine}</td>
                                <td className="p-2 border">{med.dosage}</td>
                                <td className="p-2 border">{med.frequency}</td>
                                <td className="p-2 border">{med.category}</td>
                                <td className="p-2 border">
                                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                                    <div 
                                      className="bg-blue-600 h-2.5 rounded-full" 
                                      style={{ width: `${Math.round(med.probability * 100)}%` }}
                                    ></div>
                                  </div>
                                  <span className="text-xs text-gray-600">{Math.round(med.probability * 100)}%</span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>

                      {/* Safety Warnings */}
                      {(interactionWarnings.length > 0 || contraindicationWarnings.length > 0) && (
                        <div className="mb-6">
                          <h4 className="text-md font-semibold mb-2 text-red-600">Safety Warnings</h4>
                          
                          {interactionWarnings.length > 0 && (
                            <div className="mb-4">
                              <h5 className="font-medium mb-1 flex items-center">
                                <AlertOctagon className="w-4 h-4 mr-1" /> Drug Interactions
                              </h5>
                              <ul className="list-disc pl-5 space-y-1 text-sm">
                                {interactionWarnings.map((interaction, idx) => (
                                  <li key={idx} className="text-red-600">
                                    <strong>{interaction.drug}</strong> with <strong>{interaction.drug2 || interaction.drug}</strong>: {interaction.mechanism} ({interaction.severity} severity)
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          {contraindicationWarnings.length > 0 && (
                            <div>
                              <h5 className="font-medium mb-1 flex items-center">
                                <AlertOctagon className="w-4 h-4 mr-1" /> Contraindications
                              </h5>
                              <ul className="list-disc pl-5 space-y-1 text-sm">
                                {contraindicationWarnings.map((contra, idx) => (
                                  <li key={idx} className="text-red-600">
                                    <strong>{contra.medicine}</strong> is contraindicated for <strong>{contra.condition}</strong>
                                  </li>
                                ))}
                              </ul>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Monitoring Requirements */}
                      <div className="mb-6">
                        <h4 className="text-md font-semibold mb-2 text-blue-600">Monitoring Requirements</h4>
                        <ul className="list-disc pl-5 space-y-1 text-sm">
                          {generatedPrescription
                            .filter(med => med.monitoring_required && med.monitoring_required.length > 0)
                            .flatMap(med => 
                              med.monitoring_required.map((item, idx) => (
                                <li key={`${med.medicine}-${idx}`}>
                                  Monitor <strong>{item}</strong> for {med.medicine}
                                </li>
                              ))
                            )}
                          {generatedPrescription.filter(med => med.monitoring_required && med.monitoring_required.length > 0).length === 0 && (
                            <li className="text-gray-500">No special monitoring required for these medications</li>
                          )}
                        </ul>
                      </div>
                    </>
                  ) : (
                    <div className="bg-yellow-100 border-l-4 border-yellow-500 text-yellow-700 p-4">
                      <p>No prescription generated yet. Click "Generate Prescription" to create one.</p>
                    </div>
                  )}
                </div>
                
                <div className="mb-8">
                  <h3 className="text-lg font-semibold mb-2">Instructions</h3>
                  <ul className="list-disc pl-5 space-y-1">
                    <li>Take medications as prescribed with food unless otherwise directed</li>
                    <li>Follow up in 30 days or if symptoms worsen</li>
                    <li>Maintain a healthy diet and exercise routine</li>
                    <li>Monitor blood pressure weekly and record readings</li>
                    {selectedPatient.allergies.length > 0 && (
                      <li className="text-red-600">Avoid allergens: {selectedPatient.allergies.join(', ')}</li>
                    )}
                    <li>Contact clinic immediately if experiencing severe side effects</li>
                  </ul>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                  <div>
                    <h3 className="text-lg font-semibold mb-2">Additional Notes</h3>
                    <p className="text-gray-700">
                      Patient is advised to maintain regular follow-ups and adhere to the prescribed treatment plan. 
                      Lifestyle modifications including diet and exercise are strongly recommended.
                    </p>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold mb-2">Next Appointment</h3>
                    <p className="text-gray-700">
                      {selectedPatient.nextAppointment 
                        ? formatDate(selectedPatient.nextAppointment)
                        : "To be scheduled"}
                    </p>
                  </div>
                </div>
                
                <div className="flex justify-end mt-8 pt-4 border-t">
                  <div className="text-right">
                    <p className="font-medium">Dr. Amanda Smith</p>
                    <p className="text-sm text-gray-600">Signature</p>
                    <div className="mt-4 h-16 w-48 border-t border-gray-400"></div>
                  </div>
                </div>
              </div>
            </div>
          ) : showPatientForm ? (
            <div className="h-full overflow-y-auto p-4 sm:p-6">
              <div className="bg-gray-800 rounded-lg p-4 sm:p-6 max-w-4xl mx-auto">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl sm:text-2xl font-bold">
                    {isAddingPatient ? "Add New Patient" : "Edit Patient"}
                  </h2>
                  <button 
                    onClick={() => setShowPatientForm(false)}
                    className="p-2 hover:bg-gray-700 rounded-lg"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
                
                <div className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">
                        Full Name <span className="text-red-500">*</span>
                      </label>
                      <input
                        type="text"
                        name="name"
                        value={newPatient.name}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="Patient's full name"
                        required
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">
                        Age <span className="text-red-500">*</span>
                      </label>
                      <input
                        type="number"
                        name="age"
                        value={newPatient.age}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="Patient's age"
                        required
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">
                        Gender <span className="text-red-500">*</span>
                      </label>
                      <select
                        name="gender"
                        value={newPatient.gender}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        required
                      >
                        <option value="">Select Gender</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                        <option value="Other">Other</option>
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Phone</label>
                      <input
                        type="tel"
                        name="phone"
                        value={newPatient.phone}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="Patient's phone number"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Email</label>
                      <input
                        type="email"
                        name="email"
                        value={newPatient.email}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="Patient's email"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Address</label>
                      <input
                        type="text"
                        name="address"
                        value={newPatient.address}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="Patient's address"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Diagnosis</label>
                      <input
                        type="text"
                        name="diagnosis"
                        value={newPatient.diagnosis}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="Primary diagnosis"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Blood Group</label>
                      <select
                        name="bloodGroup"
                        value={newPatient.bloodGroup}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                      >
                        <option value="">Select Blood Group</option>
                        <option value="A+">A+</option>
                        <option value="A-">A-</option>
                        <option value="B+">B+</option>
                        <option value="B-">B-</option>
                        <option value="AB+">AB+</option>
                        <option value="AB-">AB-</option>
                        <option value="O+">O+</option>
                        <option value="O-">O-</option>
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Risk Level</label>
                      <select
                        name="riskLevel"
                        value={newPatient.riskLevel}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                      >
                        <option value="low">Low</option>
                        <option value="moderate">Moderate</option>
                        <option value="high">High</option>
                        <option value="critical">Critical</option>
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Risk Score</label>
                      <input
                        type="number"
                        name="riskScore"
                        value={newPatient.riskScore}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="0-100"
                        min="0"
                        max="100"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Compliance (%)</label>
                      <input
                        type="number"
                        name="compliance"
                        value={newPatient.compliance}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        placeholder="0-100"
                        min="0"
                        max="100"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Priority</label>
                      <select
                        name="priority"
                        value={newPatient.priority}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                      >
                        <option value="routine">Routine</option>
                        <option value="urgent">Urgent</option>
                        <option value="critical">Critical</option>
                      </select>
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Last Visit</label>
                      <input
                        type="date"
                        name="lastVisit"
                        value={newPatient.lastVisit}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                    
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Next Appointment</label>
                      <input
                        type="date"
                        name="nextAppointment"
                        value={newPatient.nextAppointment}
                        onChange={handleInputChange}
                        className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                  </div>
                  
                  <div className="bg-gray-700 p-4 rounded-lg">
                    <h3 className="text-lg font-semibold mb-3 flex items-center justify-between">
                      <span>Allergies</span>
                      <button 
                        onClick={() => handleAddArrayItem('allergies', '')}
                        className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                      >
                        <Plus className="w-3 h-3" />
                        <span>Add Allergy</span>
                      </button>
                    </h3>
                    {newPatient.allergies.length === 0 ? (
                      <p className="text-gray-400 text-sm">No allergies recorded</p>
                    ) : (
                      <div className="space-y-2">
                        {newPatient.allergies.map((allergy, index) => (
                          <div key={index} className="flex items-center space-x-2">
                            <input
                              type="text"
                              value={allergy}
                              onChange={(e) => {
                                const updatedAllergies = [...newPatient.allergies];
                                updatedAllergies[index] = e.target.value;
                                setNewPatient({
                                  ...newPatient,
                                  allergies: updatedAllergies
                                });
                              }}
                              className="flex-1 px-3 py-1.5 bg-gray-600 border border-gray-500 rounded text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                              placeholder="Allergy name"
                            />
                            <button 
                              onClick={() => handleRemoveArrayItem('allergies', index)}
                              className="p-1.5 text-red-400 hover:bg-red-900/30 rounded"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  
                  <div className="bg-gray-700 p-4 rounded-lg">
                    <h3 className="text-lg font-semibold mb-3 flex items-center justify-between">
                      <span>Contraindications</span>
                      <button 
                        onClick={() => handleAddArrayItem('contraindications', '')}
                        className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                      >
                        <Plus className="w-3 h-3" />
                        <span>Add Contraindication</span>
                      </button>
                    </h3>
                    {newPatient.contraindications.length === 0 ? (
                      <p className="text-gray-400 text-sm">No contraindications recorded</p>
                    ) : (
                      <div className="space-y-2">
                        {newPatient.contraindications.map((contra, index) => (
                          <div key={index} className="flex items-center space-x-2">
                            <input
                              type="text"
                              value={contra}
                              onChange={(e) => {
                                const updatedContras = [...newPatient.contraindications];
                                updatedContras[index] = e.target.value;
                                setNewPatient({
                                  ...newPatient,
                                  contraindications: updatedContras
                                });
                              }}
                              className="flex-1 px-3 py-1.5 bg-gray-600 border border-gray-500 rounded text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                              placeholder="Contraindication"
                            />
                            <button 
                              onClick={() => handleRemoveArrayItem('contraindications', index)}
                              className="p-1.5 text-red-400 hover:bg-red-900/30 rounded"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  
                  <div className="bg-gray-700 p-4 rounded-lg">
                    <h3 className="text-lg font-semibold mb-3 flex items-center justify-between">
                      <span>Risk Factors</span>
                      <button 
                        onClick={() => handleAddArrayItem('riskFactors', '')}
                        className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                      >
                        <Plus className="w-3 h-3" />
                        <span>Add Risk Factor</span>
                      </button>
                    </h3>
                    {newPatient.riskFactors.length === 0 ? (
                      <p className="text-gray-400 text-sm">No risk factors recorded</p>
                    ) : (
                      <div className="space-y-2">
                        {newPatient.riskFactors.map((factor, index) => (
                          <div key={index} className="flex items-center space-x-2">
                            <input
                              type="text"
                              value={factor}
                              onChange={(e) => {
                                const updatedFactors = [...newPatient.riskFactors];
                                updatedFactors[index] = e.target.value;
                                setNewPatient({
                                  ...newPatient,
                                  riskFactors: updatedFactors
                                });
                              }}
                              className="flex-1 px-3 py-1.5 bg-gray-600 border border-gray-500 rounded text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                              placeholder="Risk factor"
                            />
                            <button 
                              onClick={() => handleRemoveArrayItem('riskFactors', index)}
                              className="p-1.5 text-red-400 hover:bg-red-900/30 rounded"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  
                  <div className="bg-gray-700 p-4 rounded-lg">
                    <h3 className="text-lg font-semibold mb-3 flex items-center justify-between">
                      <span>Critical Alerts</span>
                      <button 
                        onClick={() => handleAddArrayItem('criticalAlerts', '')}
                        className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                      >
                        <Plus className="w-3 h-3" />
                        <span>Add Alert</span>
                      </button>
                    </h3>
                    {newPatient.criticalAlerts.length === 0 ? (
                      <p className="text-gray-400 text-sm">No critical alerts</p>
                    ) : (
                      <div className="space-y-2">
                        {newPatient.criticalAlerts.map((alert, index) => (
                          <div key={index} className="flex items-center space-x-2">
                            <input
                              type="text"
                              value={alert}
                              onChange={(e) => {
                                const updatedAlerts = [...newPatient.criticalAlerts];
                                updatedAlerts[index] = e.target.value;
                                setNewPatient({
                                  ...newPatient,
                                  criticalAlerts: updatedAlerts
                                });
                              }}
                              className="flex-1 px-3 py-1.5 bg-gray-600 border border-gray-500 rounded text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                              placeholder="Critical alert"
                            />
                            <button 
                              onClick={() => handleRemoveArrayItem('criticalAlerts', index)}
                              className="p-1.5 text-red-400 hover:bg-red-900/30 rounded"
                            >
                              <Trash2 className="w-4 h-4" />
                            </button>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  
                  <div className="bg-gray-700 p-4 rounded-lg">
                    <h3 className="text-lg font-semibold mb-3 flex items-center justify-between">
                      <span>Vitals</span>
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">Blood Pressure</label>
                        <input
                          type="text"
                          name="bp"
                          value={newPatient.vitals.bp}
                          onChange={(e) => handleNestedInputChange('vitals', e)}
                          className="w-full px-3 py-1.5 bg-gray-600 border border-gray-500 rounded text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                          placeholder="e.g. 120/80 mmHg"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">Pulse</label>
                        <input
                          type="text"
                          name="pulse"
                          value={newPatient.vitals.pulse}
                          onChange={(e) => handleNestedInputChange('vitals', e)}
                          className="w-full px-3 py-1.5 bg-gray-600 border border-gray-500 rounded text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                          placeholder="e.g. 72 bpm"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">Temperature</label>
                        <input
                          type="text"
                          name="temp"
                          value={newPatient.vitals.temp}
                          onChange={(e) => handleNestedInputChange('vitals', e)}
                          className="w-full px-3 py-1.5 bg-gray-600 border border-gray-500 rounded text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                          placeholder="e.g. 98.6°F"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">Weight</label>
                        <input
                          type="text"
                          name="weight"
                          value={newPatient.vitals.weight}
                          onChange={(e) => handleNestedInputChange('vitals', e)}
                          className="w-full px-3 py-1.5 bg-gray-600 border border-gray-500 rounded text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                          placeholder="e.g. 150 lbs"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">Height</label>
                        <input
                          type="text"
                          name="height"
                          value={newPatient.vitals.height}
                          onChange={(e) => handleNestedInputChange('vitals', e)}
                          className="w-full px-3 py-1.5 bg-gray-600 border border-gray-500 rounded text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                          placeholder="e.g. 5'8&quot;"
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">Last Updated</label>
                        <input
                          type="date"
                          name="lastUpdated"
                          value={newPatient.vitals.lastUpdated}
                          onChange={(e) => handleNestedInputChange('vitals', e)}
                          className="w-full px-3 py-1.5 bg-gray-600 border border-gray-500 rounded text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                        />
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-gray-700 p-4 rounded-lg">
                    <h3 className="text-lg font-semibold mb-3 flex items-center justify-between">
                      <span>Lab Results</span>
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {Object.entries(newPatient.labResults).map(([key, value]) => {
                        if (key === 'status') return null;
                        return (
                          <div key={key}>
                            <label className="block text-sm font-medium text-gray-300 mb-1 capitalize">{key}</label>
                            <input
                              type="text"
                              name={key}
                              value={value}
                              onChange={(e) => handleNestedInputChange('labResults', e)}
                              className="w-full px-3 py-1.5 bg-gray-600 border border-gray-500 rounded text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                            />
                          </div>
                        );
                      })}
                      <div>
                        <label className="block text-sm font-medium text-gray-300 mb-1">Status</label>
                        <select
                          name="status"
                          value={newPatient.labResults.status}
                          onChange={(e) => handleNestedInputChange('labResults', e)}
                          className="w-full px-3 py-1.5 bg-gray-600 border border-gray-500 rounded text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                        >
                          <option value="stable">Stable</option>
                          <option value="improving">Improving</option>
                          <option value="concerning">Concerning</option>
                          <option value="critical">Critical</option>
                        </select>
                      </div>
                    </div>
                  </div>
                  
                  <div className="bg-gray-700 p-4 rounded-lg">
                    <h3 className="text-lg font-semibold mb-3 flex items-center justify-between">
                      <span>Medical History</span>
                      <button 
                        onClick={() => handleAddArrayItem('medicalHistory', {
                          date: new Date().toISOString().split('T')[0],
                          condition: "",
                          notes: "",
                          severity: "medium"
                        })}
                        className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                      >
                        <Plus className="w-3 h-3" />
                        <span>Add Entry</span>
                      </button>
                    </h3>
                    {newPatient.medicalHistory.length === 0 ? (
                      <p className="text-gray-400 text-sm">No medical history recorded</p>
                    ) : (
                      <div className="space-y-4">
                        {newPatient.medicalHistory.map((entry, index) => (
                          <div key={index} className="border-l-2 border-blue-500 pl-4 py-2">
                            <div className="flex justify-between items-start">
                              <div className="flex-1 space-y-2">
                                <input
                                  type="text"
                                  value={entry.condition}
                                  onChange={(e) => handleArrayChange('medicalHistory', index, 'condition', e.target.value)}
                                  className="w-full px-3 py-1.5 bg-gray-600 border border-gray-500 rounded text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                                  placeholder="Condition"
                                />
                                <textarea
                                  value={entry.notes}
                                  onChange={(e) => handleArrayChange('medicalHistory', index, 'notes', e.target.value)}
                                  className="w-full px-3 py-1.5 bg-gray-600 border border-gray-500 rounded text-white focus:outline-none focus:ring-1 focus:ring-blue-500"
                                  placeholder="Notes"
                                  rows="2"
                                />
                              </div>
                              <button 
                                onClick={() => handleRemoveArrayItem('medicalHistory', index)}
                                className="p-1.5 text-red-400 hover:bg-red-900/30 rounded ml-2"
                              >
                                <Trash2 className="w-4 h-4" />
                              </button>
                            </div>
                            <div className="flex items-center space-x-4 mt-2">
                              <div>
                                <label className="block text-xs text-gray-400 mb-1">Date</label>
                                <input
                                  type="date"
                                  value={entry.date}
                                  onChange={(e) => handleArrayChange('medicalHistory', index, 'date', e.target.value)}
                                  className="px-2 py-1 bg-gray-600 border border-gray-500 rounded text-white text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                                />
                              </div>
                              <div>
                                <label className="block text-xs text-gray-400 mb-1">Severity</label>
                                <select
                                  value={entry.severity}
                                  onChange={(e) => handleArrayChange('medicalHistory', index, 'severity', e.target.value)}
                                  className="px-2 py-1 bg-gray-600 border border-gray-500 rounded text-white text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                                >
                                  <option value="low">Low</option>
                                  <option value="medium">Medium</option>
                                  <option value="high">High</option>
                                </select>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  
                  <div className="bg-gray-700 p-4 rounded-lg">
                    <h3 className="text-lg font-semibold mb-3 flex items-center justify-between">
                      <span>Current Medications</span>
                      <button 
                        onClick={() => handleAddArrayItem('currentMedications', {
                          name: "",
                          dosage: "",
                          frequency: "",
                          prescribed: new Date().toISOString().split('T')[0],
                          adherence: "good"
                        })}
                        className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm"
                      >
                        <Plus className="w-3 h-3" />
                        <span>Add Medication</span>
                      </button>
                    </h3>
                    {newPatient.currentMedications.length === 0 ? (
                      <p className="text-gray-400 text-sm">No current medications</p>
                    ) : (
                      <div className="space-y-4">
                        {newPatient.currentMedications.map((med, index) => (
                          <div key={index} className="border-l-2 border-blue-500 pl-4 py-2">
                            <div className="flex justify-between items-start">
                              <div className="grid grid-cols-1 md:grid-cols-4 gap-2 flex-1">
                                <div>
                                  <label className="block text-xs text-gray-400 mb-1">Name</label>
                                  <input
                                    type="text"
                                    value={med.name}
                                    onChange={(e) => handleArrayChange('currentMedications', index, 'name', e.target.value)}
                                    className="w-full px-2 py-1 bg-gray-600 border border-gray-500 rounded text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                                    placeholder="Medication name"
                                  />
                                </div>
                                <div>
                                  <label className="block text-xs text-gray-400 mb-1">Dosage</label>
                                  <input
                                    type="text"
                                    value={med.dosage}
                                    onChange={(e) => handleArrayChange('currentMedications', index, 'dosage', e.target.value)}
                                    className="w-full px-2 py-1 bg-gray-600 border border-gray-500 rounded text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                                    placeholder="Dosage"
                                  />
                                </div>
                                <div>
                                  <label className="block text-xs text-gray-400 mb-1">Frequency</label>
                                  <input
                                    type="text"
                                    value={med.frequency}
                                    onChange={(e) => handleArrayChange('currentMedications', index, 'frequency', e.target.value)}
                                    className="w-full px-2 py-1 bg-gray-600 border border-gray-500 rounded text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                                    placeholder="Frequency"
                                  />
                                </div>
                                <div>
                                  <label className="block text-xs text-gray-400 mb-1">Adherence</label>
                                  <select
                                    value={med.adherence}
                                    onChange={(e) => handleArrayChange('currentMedications', index, 'adherence', e.target.value)}
                                    className="w-full px-2 py-1 bg-gray-600 border border-gray-500 rounded text-white text-sm focus:outline-none focus:ring-1 focus:ring-blue-500"
                                  >
                                    <option value="excellent">Excellent</option>
                                    <option value="good">Good</option>
                                    <option value="fair">Fair</option>
                                    <option value="poor">Poor</option>
                                  </select>
                                </div>
                              </div>
                              <button 
                                onClick={() => handleRemoveArrayItem('currentMedications', index)}
                                className="p-1.5 text-red-400 hover:bg-red-900/30 rounded ml-2"
                              >
                                <Trash2 className="w-4 h-4" />
                              </button>
                            </div>
                            <div className="mt-2">
                              <label className="block text-xs text-gray-400 mb-1">Prescribed Date</label>
                              <input
                                type="date"
                                value={med.prescribed}
                                onChange={(e) => handleArrayChange('currentMedications', index, 'prescribed', e.target.value)}
                                className="px-2 py-1 bg-gray-600 border border-gray-500 rounded text-white text-xs focus:outline-none focus:ring-1 focus:ring-blue-500"
                              />
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                  
                  <div className="flex justify-end space-x-3 pt-4">
                    <button
                      onClick={() => setShowPatientForm(false)}
                      className="px-6 py-2 bg-gray-600 hover:bg-gray-500 rounded-lg transition-colors"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={handleSavePatient}
                      className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                    >
                      {isAddingPatient ? "Add Patient" : "Save Changes"}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          ) : selectedPatient ? (
            <div className="h-full overflow-y-auto">
              <div className="bg-gradient-to-r from-gray-800 to-gray-700 p-4 sm:p-6 border-b border-gray-700">
                <div className="flex flex-col sm:flex-row sm:items-center justify-between">
                  <div className="flex items-center space-x-3 sm:space-x-4 mb-4 sm:mb-0">
                    <div className="w-16 h-16 sm:w-20 sm:h-20 bg-gradient-to-br from-blue-600 to-purple-600 rounded-full flex items-center justify-center">
                      <User className="w-8 h-8 sm:w-10 sm:h-10 text-white" />
                    </div>
                    <div>
                      <div className="flex flex-col sm:flex-row sm:items-center space-y-1 sm:space-y-0 sm:space-x-3">
                        <h2 className="text-xl sm:text-3xl font-bold text-white">{selectedPatient.name}</h2>
                        <div className={`px-2 sm:px-3 py-1 rounded-full text-xs sm:text-sm font-medium flex items-center space-x-1 ${getRiskColor(selectedPatient.riskLevel)}`}>
                          {getRiskIcon(selectedPatient.riskLevel)}
                          <span className="capitalize">{selectedPatient.riskLevel} Risk</span>
                        </div>
                      </div>
                      <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs sm:text-sm">
                        <span className="text-gray-300">{selectedPatient.age} years old</span>
                        <span className="text-gray-400 hidden sm:inline">•</span>
                        <span className="text-gray-300">{selectedPatient.gender}</span>
                        <span className="text-gray-400 hidden sm:inline">•</span>
                        <span className="text-gray-300">Blood Group: {selectedPatient.bloodGroup}</span>
                        <span className="text-gray-400 hidden sm:inline">•</span>
                        <span className={`font-bold ${getComplianceColor(selectedPatient.compliance)}`}>
                          {selectedPatient.compliance}% Compliance
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button 
                      onClick={generatePrescription}
                      disabled={isGeneratingPrescription}
                      className={`flex items-center space-x-2 px-3 sm:px-4 py-2 rounded-lg transition-colors ${
                        isGeneratingPrescription 
                          ? 'bg-gray-600 cursor-not-allowed' 
                          : 'bg-green-600 hover:bg-green-700'
                      }`}
                    >
                      {isGeneratingPrescription ? (
                        <Loader2 className="w-4 h-4 sm:w-5 sm:h-5 animate-spin" />
                      ) : (
                        <Printer className="w-4 h-4 sm:w-5 sm:h-5" />
                      )}
                      <span>Generate Prescription</span>
                    </button>
                    <button 
                      onClick={() => handleDeletePatient(selectedPatient.id)}
                      className="flex items-center space-x-2 px-3 sm:px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
                    >
                      <Trash2 className="w-4 h-4 sm:w-5 sm:h-5" />
                      <span>Delete</span>
                    </button>
                    <button 
                      onClick={handleEditPatient}
                      className="flex items-center space-x-2 px-3 sm:px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                    >
                      <Edit3 className="w-4 h-4 sm:w-5 sm:h-5" />
                      <span>Edit</span>
                    </button>
                  </div>
                </div>
                
                {selectedPatient.criticalAlerts.length > 0 && (
                  <div className="mt-4 p-4 bg-red-900/20 border border-red-500 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <AlertCircle className="w-5 h-5 text-red-400" />
                      <h3 className="text-red-400 font-semibold">Critical Alerts</h3>
                    </div>
                    <div className="space-y-1">
                      {selectedPatient.criticalAlerts.map((alert, index) => (
                        <p key={index} className="text-red-300 text-sm">• {alert}</p>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              <div className="p-4 sm:p-6 space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
                  <div className="lg:col-span-2 bg-gray-800 rounded-lg p-4 sm:p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                      <Activity className="w-5 h-5 mr-2 text-blue-400" />
                      Risk Assessment
                    </h3>
                    <div className="space-y-4">
                      <div className="flex flex-col sm:flex-row items-center justify-between">
                        <div>
                          <span className="text-gray-400">Risk Score</span>
                          <div className="text-3xl font-bold" style={{ color: getRiskColor(selectedPatient.riskLevel).split(' ')[0] }}>
                            {selectedPatient.riskScore}
                            <span className="text-xl text-gray-400">/100</span>
                          </div>
                        </div>
                        <div className="w-full sm:max-w-md mt-4 sm:mt-0 sm:ml-4">
                          <div className="h-4 bg-gray-700 rounded-full overflow-hidden">
                            <div 
                              className={`h-full rounded-full bg-gradient-to-r ${getGradientColor(selectedPatient.riskScore)}`}
                              style={{ width: `${selectedPatient.riskScore}%` }}
                            ></div>
                          </div>
                          <div className="flex justify-between mt-1 text-xs text-gray-400">
                            <span>Low</span>
                            <span>Moderate</span>
                            <span>High</span>
                            <span>Critical</span>
                          </div>
                        </div>
                      </div>
                      
                      <div>
                        <h4 className="text-md font-medium text-gray-300 mb-2">Key Risk Factors</h4>
                        <div className="space-y-2">
                          {selectedPatient.riskFactors.map((factor, index) => (
                            <div key={index} className="flex items-center">
                              <div className={`w-3 h-3 rounded-full mr-3 ${index < 2 ? 'bg-red-500' : 'bg-yellow-500'}`}></div>
                              <span className="text-gray-300">{factor}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="bg-gray-800 rounded-lg p-4 sm:p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                      <TrendingUp className="w-5 h-5 mr-2 text-green-400" />
                      Risk Trend
                    </h3>
                    <div className="h-48 sm:h-64 flex items-end justify-between pt-4">
                      {[30, 45, 60, 75, 90, selectedPatient.riskScore].map((value, index) => (
                        <div key={index} className="flex flex-col items-center w-8 sm:w-10">
                          <div 
                            className={`w-6 sm:w-8 rounded-t-md ${
                              value >= 80 ? 'bg-gradient-to-t from-red-500 to-red-700' :
                              value >= 60 ? 'bg-gradient-to-t from-orange-500 to-orange-700' :
                              value >= 40 ? 'bg-gradient-to-t from-yellow-500 to-yellow-700' : 
                              'bg-gradient-to-t from-green-500 to-green-700'
                            }`}
                            style={{ height: `${value}%` }}
                          ></div>
                          <span className="text-xs text-gray-400 mt-2">{['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'][index]}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
                  <div className="bg-gray-800 rounded-lg p-4 sm:p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                      <User className="w-5 h-5 mr-2 text-blue-400" />
                      Contact Information
                    </h3>
                    <div className="space-y-3">
                      <div className="flex items-center">
                        <Phone className="w-4 h-4 mr-3 text-gray-400" />
                        <span className="text-gray-300">{selectedPatient.phone}</span>
                      </div>
                      <div className="flex items-center">
                        <Mail className="w-4 h-4 mr-3 text-gray-400" />
                        <span className="text-gray-300">{selectedPatient.email}</span>
                      </div>
                      <div className="flex items-start">
                        <MapPin className="w-4 h-4 mr-3 text-gray-400 mt-1" />
                        <span className="text-gray-300">{selectedPatient.address}</span>
                      </div>
                    </div>
                  </div>

                  <div className="lg:col-span-2 bg-gray-800 rounded-lg p-4 sm:p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                      <FileText className="w-5 h-5 mr-2 text-blue-400" />
                      Medical History
                    </h3>
                    <div className="space-y-4">
                      {selectedPatient.medicalHistory.map((item, index) => (
                        <div key={index} className="border-l-2 border-blue-500 pl-4 py-1">
                          <div className="flex flex-col sm:flex-row sm:justify-between">
                            <span className="text-gray-300 font-medium">{item.condition}</span>
                            <span className="text-gray-400 text-sm">{formatDate(item.date)}</span>
                          </div>
                          <p className="text-gray-400 text-sm">{item.notes}</p>
                          <div className="flex items-center mt-1">
                            <span className={`text-xs px-2 py-1 rounded-full ${
                              item.severity === 'high' ? 'bg-red-900 text-red-400' :
                              item.severity === 'medium' ? 'bg-yellow-900 text-yellow-400' :
                              'bg-green-900 text-green-400'
                            }`}>
                              Severity: {item.severity}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
                  <div className="lg:col-span-2 bg-gray-800 rounded-lg p-4 sm:p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                      <Pill className="w-5 h-5 mr-2 text-blue-400" />
                      Current Medications
                    </h3>
                    <div className="overflow-x-auto">
                      <table className="w-full">
                        <thead>
                          <tr className="text-left text-gray-400 text-sm">
                            <th className="pb-2">Medication</th>
                            <th className="pb-2">Dosage</th>
                            <th className="pb-2">Frequency</th>
                            <th className="pb-2">Prescribed</th>
                            <th className="pb-2">Adherence</th>
                          </tr>
                        </thead>
                        <tbody>
                          {selectedPatient.currentMedications.map((med, index) => (
                            <tr key={index} className="border-b border-gray-700">
                              <td className="py-3 text-gray-300">{med.name}</td>
                              <td className="py-3 text-gray-400">{med.dosage}</td>
                              <td className="py-3 text-gray-400">{med.frequency}</td>
                              <td className="py-3 text-gray-400">{formatDate(med.prescribed)}</td>
                              <td className="py-3">
                                <span className={`px-2 py-1 rounded-full text-xs ${
                                  med.adherence === 'excellent' ? 'bg-green-900 text-green-400' :
                                  med.adherence === 'good' ? 'bg-blue-900 text-blue-400' :
                                  med.adherence === 'fair' ? 'bg-yellow-900 text-yellow-400' : 'bg-red-900 text-red-400'
                                }`}>
                                  {med.adherence}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  <div className="bg-gray-800 rounded-lg p-4 sm:p-6">
                    <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                      <Activity className="w-5 h-5 mr-2 text-blue-400" />
                      Lab Results
                    </h3>
                    <div className="space-y-4">
                      {Object.entries(selectedPatient.labResults).map(([key, value]) => {
                        if (key === 'status') return null;
                        return (
                          <div key={key} className="flex justify-between items-center border-b border-gray-700 pb-2">
                            <span className="text-gray-400 capitalize">{key}</span>
                            <span className="text-gray-300 font-medium">{value}</span>
                          </div>
                        );
                      })}
                      <div className="mt-4 p-3 rounded-lg bg-gray-700">
                        <span className="text-sm text-gray-300">Status: </span>
                        <span className={`text-sm font-medium ${
                          selectedPatient.labResults.status === 'critical' ? 'text-red-400' :
                          selectedPatient.labResults.status === 'concerning' ? 'text-yellow-400' :
                          selectedPatient.labResults.status === 'stable' ? 'text-green-400' : 'text-blue-400'
                        }`}>
                          {selectedPatient.labResults.status}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-800 rounded-lg p-4 sm:p-6">
                  <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
                    <Heart className="w-5 h-5 mr-2 text-blue-400" />
                    Vitals
                  </h3>
                  <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2 sm:gap-4">
                    {Object.entries(selectedPatient.vitals).map(([key, value]) => {
                      if (key === 'lastUpdated') return null;
                      return (
                        <div key={key} className="bg-gray-700 p-2 sm:p-4 rounded-lg">
                          <div className="text-gray-400 text-xs sm:text-sm capitalize">{key}</div>
                          <div className="text-white text-lg sm:text-xl font-semibold">{value}</div>
                        </div>
                      );
                    })}
                  </div>
                  <div className="mt-4 text-xs sm:text-sm text-gray-400">
                    Last updated: {selectedPatient.vitals.lastUpdated}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <div className="bg-gray-800 p-6 sm:p-8 rounded-xl inline-block">
                  <User className="w-12 h-12 sm:w-16 sm:h-16 text-gray-400 mx-auto" />
                  <h3 className="text-lg sm:text-xl text-gray-300 mt-4">Select a patient to view details</h3>
                  <p className="text-gray-500 text-sm sm:text-base mt-2">Choose a patient from the list to see their medical information</p>
                  <button 
                    onClick={handleAddPatient}
                    className="mt-4 px-4 sm:px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors"
                  >
                    <Plus className="w-4 h-4 sm:w-5 sm:h-5 inline mr-2" />
                    Add New Patient
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default MedicalInterface;
