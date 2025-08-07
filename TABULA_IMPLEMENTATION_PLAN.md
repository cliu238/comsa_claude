# Tabula-8B VA Classification Implementation Plan

## Executive Summary

This plan outlines the implementation of Tabula-8B for Verbal Autopsy (VA) cause-of-death classification, leveraging the PHMRC adult dataset. Based on comprehensive analysis, we propose a phased approach integrating Tabula-8B with existing high-performing methods (XGBoost, InSilicoVA) to achieve state-of-the-art performance while maintaining domain expertise.

## Project Objectives

### Primary Goals
- Deploy Tabula-8B for VA classification with >75% CSMF accuracy
- Create hybrid ensemble combining domain expertise with LLM capabilities
- Establish production-ready pipeline with site-stratified validation
- Provide interpretable results for medical practitioners

### Success Metrics
- **CSMF Accuracy**: Target >75% (baseline: XGBoost 74.84%)
- **Individual COD Accuracy**: Target >60% (baseline: XGBoost 56.92%)
- **Site Generalization**: Consistent performance across PHMRC sites
- **Inference Latency**: <2 seconds per prediction
- **Memory Efficiency**: Peak GPU usage <20GB

## Data Analysis Summary

### Dataset Characteristics
- **Size**: 7,841 adult VA records
- **Features**: 946 variables (195 symptoms, 680 word counts, demographics)
- **Target**: 34 cause categories (gs_text34)
- **Challenge**: Severe class imbalance (top cause: 19.4%, tail causes: <1%)

### Current Performance Baseline
```
Method          CSMF Acc   COD Acc   Strengths
XGBoost         74.84%     56.92%    Best overall performance
InSilicoVA      62.03%     44.21%    Domain expertise, interpretability
Random Forest   73.12%     54.83%    Robust, feature importance
Naive Bayes     65.27%     48.15%    Fast, probabilistic
```

## VA Codebook Semantic Mapping

### PHMRC Dataset Structure
The PHMRC adult dataset contains 1,649 variables organized into functional categories. Understanding the semantic meaning of these coded columns is critical for Tabula-8B's language understanding capabilities.

### Core Variable Categories & Mappings

#### Demographics (g-series variables)
```python
demographic_mappings = {
    'g1_05': 'patient_sex',  # 1=Male, 2=Female
    'g1_07a': 'age_at_death_years',
    'g1_07b': 'age_at_death_months',
    'g1_07c': 'age_at_death_days',
    'g1_08': 'marital_status',  # 1=Never married, 2=Married, 3=Widowed
    'g1_09': 'education_level',  # 1=No schooling, 2=Primary, 3=Secondary
    'g4_02': 'respondent_sex',
    'g4_03a': 'respondent_relationship_to_deceased',
    'g4_04': 'respondent_age'
}
```

#### Medical History (a1-series) - Pre-existing Conditions
```python
medical_history_mappings = {
    'a1_01_1': 'history_of_asthma',
    'a1_01_2': 'history_of_arthritis',
    'a1_01_3': 'history_of_cancer',
    'a1_01_4': 'history_of_copd',
    'a1_01_7': 'history_of_diabetes',
    'a1_01_8': 'history_of_epilepsy',
    'a1_01_9': 'history_of_heart_disease',
    'a1_01_10': 'history_of_hypertension',
    'a1_01_11': 'history_of_kidney_disease',
    'a1_01_12': 'history_of_liver_disease',
    'a1_01_13': 'history_of_stroke',
    'a1_01_14': 'history_of_aids'
}
```

#### Primary Symptom Indicators (a2-series) - 195 Clinical Features
```python
symptom_mappings = {
    # Constitutional Symptoms
    'a2_02': 'patient_had_fever',
    'a2_04': 'fever_severity',  # 1=Mild, 2=Moderate, 3=Severe
    'a2_05': 'fever_pattern',  # 1=Continuous, 2=On and off
    'a2_18': 'experienced_weight_loss',
    'a2_20': 'had_pallor_or_looked_pale',
    
    # Respiratory Symptoms
    'a2_32': 'patient_had_cough',
    'a2_34': 'cough_was_productive_with_sputum',
    'a2_35': 'coughed_blood_hemoptysis',
    'a2_36': 'had_difficulty_breathing_dyspnea',
    'a2_38': 'breathing_difficulty_pattern',  # 1=Continuous, 2=On and off
    'a2_40': 'had_fast_breathing_tachypnea',
    'a2_42': 'had_wheezing',
    
    # Cardiovascular Symptoms
    'a2_43': 'experienced_chest_pain',
    'a2_44': 'chest_pain_duration',  # 1=<30min, 2=0.5-24hrs, 3=>24hrs
    'a2_17': 'had_cyanosis_blue_lips',
    'a2_23': 'had_ankle_swelling_edema',
    'a2_25': 'had_facial_puffiness',
    
    # Gastrointestinal Symptoms
    'a2_47': 'had_diarrhea_loose_stools',
    'a2_50': 'had_blood_in_stool',
    'a2_53': 'experienced_vomiting',
    'a2_55': 'vomited_blood_hematemesis',
    'a2_57': 'had_difficulty_swallowing_dysphagia',
    'a2_61': 'had_abdominal_pain',
    'a2_63_1': 'abdominal_pain_in_upper_region',
    'a2_63_2': 'abdominal_pain_in_lower_region',
    
    # Neurological Symptoms
    'a2_69': 'had_headaches',
    'a2_72': 'had_neck_stiffness',
    'a2_74': 'lost_consciousness',
    'a2_78': 'experienced_confusion',
    'a2_81': 'had_memory_loss',
    'a2_82': 'had_convulsions_or_seizures',
    'a2_85': 'experienced_paralysis',
    'a2_87_1': 'paralysis_affected_right_arm',
    'a2_87_2': 'paralysis_affected_left_arm',
    'a2_87_3': 'paralysis_affected_right_leg',
    'a2_87_4': 'paralysis_affected_left_leg',
    
    # Dermatological Symptoms
    'a2_07': 'had_skin_rash',
    'a2_09_1a': 'rash_location_on_body',
    'a2_10': 'had_skin_sores_or_ulcers',
    'a2_12': 'experienced_itching'
}
```

#### Female-Specific Variables (a3-series)
```python
female_specific_mappings = {
    'a3_01': 'had_breast_lumps',
    'a3_10': 'was_pregnant_at_death',
    'a3_13': 'had_bleeding_during_pregnancy',
    'a3_15': 'died_during_labor_or_delivery',
    'a3_18': 'died_within_6_weeks_of_childbirth'
}
```

#### Risk Factors (a4-series)
```python
risk_factor_mappings = {
    'a4_01': 'used_tobacco',
    'a4_02_1': 'smoked_cigarettes',
    'a4_02_2': 'smoked_pipe',
    'a4_02_3': 'chewed_tobacco',
    'a4_05': 'consumed_alcohol',
    'a4_06': 'alcohol_consumption_level'  # 1=Low, 2=Moderate, 3=High
}
```

#### Injury Categories (a5-series)
```python
injury_mappings = {
    'a5_01_1': 'had_road_traffic_injury',
    'a5_01_2': 'had_fall_injury',
    'a5_01_3': 'experienced_drowning',
    'a5_01_4': 'had_poisoning',
    'a5_01_5': 'had_bite_or_sting',
    'a5_01_6': 'had_burn_injury',
    'a5_01_7': 'experienced_violence'
}
```

#### Healthcare Utilization (a6-series)
```python
healthcare_mappings = {
    'a6_01': 'sought_care_outside_home',
    'a6_02_1': 'visited_traditional_healer',
    'a6_02_2': 'visited_hospital',
    'a6_02_3': 'visited_health_clinic',
    'a6_02_4': 'visited_private_doctor'
}
```

#### Narrative Word Features (word-series) - Top 100 Most Informative
```python
# Select top features based on mutual information with cause of death
top_narrative_features = [
    'word_fever', 'word_cough', 'word_pain', 'word_chest',
    'word_breath', 'word_heart', 'word_cancer', 'word_accident',
    'word_blood', 'word_vomit', 'word_headache', 'word_pregnant',
    'word_deliver', 'word_hospital', 'word_medicine', 'word_sudden',
    # ... up to 100 most informative features
]
```

### Symptom Clustering by Medical Systems

```python
medical_system_clusters = {
    'respiratory_symptoms': [
        'a2_32',  # cough
        'a2_36',  # breathing difficulty
        'a2_40',  # fast breathing
        'a2_42',  # wheezing
    ],
    'cardiovascular_symptoms': [
        'a2_43',  # chest pain
        'a2_17',  # cyanosis
        'a2_23',  # ankle swelling
        'a2_25',  # facial puffiness
    ],
    'neurological_symptoms': [
        'a2_69',  # headaches
        'a2_72',  # neck stiffness
        'a2_74',  # loss of consciousness
        'a2_82',  # seizures
        'a2_85',  # paralysis
    ],
    'gastrointestinal_symptoms': [
        'a2_47',  # diarrhea
        'a2_53',  # vomiting
        'a2_61',  # abdominal pain
    ],
    'infectious_disease_indicators': [
        'a2_02',  # fever
        'a2_07',  # rash
        'a2_32',  # cough
        'a2_20',  # pallor
    ]
}
```

### Handling Special Cases

#### "Don't Know" Responses
```python
def handle_uncertain_response(column, value):
    """Handle DK/Don't Know responses - exclude from prompts"""
    if value in ["Don't Know", "DK", "Refused"]:
        return None  # Exclude from prompt
    return semantic_mappings[column]
```

#### Temporal Pattern Encoding
```python
temporal_patterns = {
    'acute': 'symptom_duration_under_72_hours',
    'subacute': 'symptom_duration_2_weeks',
    'chronic': 'symptom_duration_over_month'
}
```

## Technical Architecture

### Infrastructure Requirements

#### Hardware Specifications
```yaml
GPU Requirements:
  - Minimum: 1x A100 40GB or equivalent
  - Recommended: 1x A100 80GB for fine-tuning
  - Memory: 19.2GB+ for inference, 40GB+ for training

CPU Requirements:
  - 16+ cores for data preprocessing
  - 64GB+ RAM for large batch processing

Storage:
  - 500GB+ SSD for model weights and datasets
  - Network storage for result archiving
```

#### Software Stack
```python
# Core Dependencies
torch >= 2.0.0
transformers >= 4.35.0
accelerate >= 0.24.0
peft >= 0.6.0  # For LoRA fine-tuning
bitsandbytes >= 0.41.0  # For quantization

# Data & ML
pandas >= 2.0.0
scikit-learn >= 1.3.0
xgboost >= 1.7.0
numpy >= 1.24.0

# Monitoring & Logging
wandb >= 0.15.0
tensorboard >= 2.14.0
tqdm >= 4.65.0
```

## Implementation Phases

### Phase 1: Foundation Setup (Weeks 1-2)

#### 1.1 Environment & Infrastructure
```bash
# Setup virtual environment
poetry install
poetry add torch transformers accelerate peft

# Initialize tracking
wandb init --project="tabula-va-classification"
```

#### 1.2 Data Pipeline Enhancement
```python
# File: src/data/tabula_preprocessor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

class TabulaVAPreprocessor:
    """Specialized preprocessor for Tabula-8B VA classification with PHMRC codebook mapping"""
    
    def __init__(self, semantic_mapping: bool = True):
        self.semantic_mapping = semantic_mapping
        self.initialize_mappings()
        self.feature_groups = self._create_feature_groups()
    
    def initialize_mappings(self):
        """Initialize comprehensive PHMRC column mappings"""
        # Demographics
        self.demographic_mappings = {
            'g1_05': 'patient_sex',
            'g1_07a': 'age_at_death_years',
            'g1_08': 'marital_status',
            'g1_09': 'education_level'
        }
        
        # Primary symptoms (a2 series - 195 features)
        self.symptom_mappings = {
            'a2_02': 'patient_had_fever',
            'a2_04': 'fever_severity',
            'a2_32': 'patient_had_cough',
            'a2_36': 'had_difficulty_breathing_dyspnea',
            'a2_43': 'experienced_chest_pain',
            'a2_47': 'had_diarrhea_loose_stools',
            'a2_53': 'experienced_vomiting',
            'a2_61': 'had_abdominal_pain',
            'a2_69': 'had_headaches',
            'a2_82': 'had_convulsions_or_seizures',
            'a2_85': 'experienced_paralysis',
            # ... complete mapping of all 195 symptoms
        }
        
        # Medical history (a1 series)
        self.medical_history_mappings = {
            'a1_01_1': 'history_of_asthma',
            'a1_01_3': 'history_of_cancer',
            'a1_01_7': 'history_of_diabetes',
            'a1_01_9': 'history_of_heart_disease',
            'a1_01_14': 'history_of_aids',
            # ... complete medical history
        }
        
        # Combine all mappings
        self.all_mappings = {
            **self.demographic_mappings,
            **self.symptom_mappings,
            **self.medical_history_mappings
        }
    
    def _create_feature_groups(self) -> Dict[str, List[str]]:
        """Create tiered feature groups for context window management"""
        return {
            'tier1_core_symptoms': [col for col in self.symptom_mappings.keys() 
                                   if col.startswith('a2_')],
            'tier2_medical_history': list(self.medical_history_mappings.keys()),
            'tier3_demographics': list(self.demographic_mappings.keys()),
            'tier4_narrative': ['word_fever', 'word_cough', 'word_pain', 
                              'word_chest', 'word_breath', 'word_heart'],
        }
    
    def create_semantic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert PHMRC coded features to semantic descriptions"""
        df_semantic = df.copy()
        
        # Apply semantic mappings
        df_semantic = df_semantic.rename(columns=self.all_mappings)
        
        # Handle categorical encodings
        df_semantic = self._encode_categorical_values(df_semantic)
        
        # Add medical system clusters
        df_semantic = self._add_symptom_clusters(df_semantic)
        
        return df_semantic
    
    def _encode_categorical_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical values to semantic descriptions"""
        # Fever severity encoding
        if 'fever_severity' in df.columns:
            severity_map = {1: 'mild_fever', 2: 'moderate_fever', 3: 'severe_fever'}
            df['fever_severity'] = df['fever_severity'].map(severity_map)
        
        # Yes/No encoding
        binary_cols = [col for col in df.columns if col.startswith('patient_had_') 
                      or col.startswith('experienced_')]
        for col in binary_cols:
            df[col] = df[col].map({'Yes': f'{col}_present', 
                                  'No': f'{col}_absent',
                                  'Don\'t Know': None})  # Exclude uncertain
        
        return df
    
    def _add_symptom_clusters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add medical system-based symptom clusters"""
        # Respiratory syndrome detection
        respiratory_cols = ['patient_had_cough', 'had_difficulty_breathing_dyspnea', 
                          'had_fast_breathing_tachypnea', 'had_wheezing']
        df['respiratory_syndrome'] = (
            df[respiratory_cols].notna().sum(axis=1) >= 2
        ).astype(int)
        
        # Cardiovascular syndrome
        cardiac_cols = ['experienced_chest_pain', 'had_ankle_swelling_edema', 
                       'had_cyanosis_blue_lips']
        df['cardiovascular_syndrome'] = (
            df[cardiac_cols].notna().sum(axis=1) >= 2
        ).astype(int)
        
        return df
    
    def format_for_tabula(self, row: pd.Series) -> str:
        """Format row data for Tabula-8B input with semantic features"""
        # Extract demographics
        demographics = {
            'age': row.get('age_at_death_years', 'Unknown'),
            'sex': row.get('patient_sex', 'Unknown'),
            'marital_status': row.get('marital_status', 'Unknown'),
            'education': row.get('education_level', 'Unknown')
        }
        
        # Extract core symptoms (Tier 1)
        symptoms = []
        for symptom_col in self.feature_groups['tier1_core_symptoms']:
            semantic_name = self.symptom_mappings.get(symptom_col)
            if semantic_name and row.get(semantic_name):
                value = row[semantic_name]
                if value and value != 'None':  # Exclude Don't Know
                    symptoms.append(f"- {semantic_name}: {value}")
        
        # Extract medical history (Tier 2)
        medical_history = []
        for history_col in self.feature_groups['tier2_medical_history']:
            semantic_name = self.medical_history_mappings.get(history_col)
            if semantic_name and row.get(semantic_name) == 'Yes':
                medical_history.append(f"- {semantic_name}")
        
        # Extract top narrative features (Tier 4)
        narrative_features = []
        for word_col in self.feature_groups['tier4_narrative']:
            if word_col in row and row[word_col] > 0:
                count = row[word_col]
                feature = word_col.replace('word_', '')
                if count > 5:
                    narrative_features.append(f"frequently mentioned {feature}")
                else:
                    narrative_features.append(f"mentioned {feature}")
        
        # Create structured prompt
        prompt = f"""Medical Case Presentation for Verbal Autopsy Classification:

PATIENT DEMOGRAPHICS:
- Age at death: {demographics['age']} years
- Sex: {demographics['sex']}
- Marital status: {demographics['marital_status']}
- Education: {demographics['education']}

PRIMARY SYMPTOMS PRESENT:
{chr(10).join(symptoms) if symptoms else '- No specific symptoms recorded'}

MEDICAL HISTORY:
{chr(10).join(medical_history) if medical_history else '- No significant medical history'}

CLINICAL NARRATIVE INDICATORS:
{', '.join(narrative_features) if narrative_features else 'No specific narrative indicators'}

TASK: Based on this clinical presentation, determine the most likely cause of death from the following 34 categories:
Stroke, Pneumonia, AIDS, Acute Myocardial Infarction, Diabetes, COPD, Maternal, 
Road Traffic Accident, Falls, Drowning, Homicide, Suicide, Other Injuries, 
Tuberculosis, Malaria, Diarrhea/Dysentery, Hemorrhagic fever, Meningitis, 
Encephalitis, Sepsis, Lung Cancer, Digestive Cancer, Other Cancers, 
Leukemia/Lymphomas, Epilepsy, Kidney Disease, Liver Disease, Congenital malformation,
Preterm Delivery, Birth asphyxia, Stillbirth, Poisonings, Bite of Venomous Animal, Other

Provide your classification as: "Cause of Death: [exact_cause_name]"
"""
        return prompt
    
    def create_tiered_prompts(self, row: pd.Series) -> List[str]:
        """Create multiple prompts for different feature tiers to handle token limits"""
        prompts = []
        
        # Tier 1: Core symptoms only (highest signal, fits in context)
        tier1_prompt = self._create_tier1_prompt(row)
        prompts.append(tier1_prompt)
        
        # Tier 2: Add medical history if needed
        if self._needs_additional_context(row):
            tier2_prompt = self._create_tier2_prompt(row)
            prompts.append(tier2_prompt)
        
        return prompts
    
    def _create_tier1_prompt(self, row: pd.Series) -> str:
        """Create prompt with only core symptoms (Tier 1)"""
        # Focus on a2_* series symptoms
        symptoms = []
        for col in ['a2_02', 'a2_32', 'a2_36', 'a2_43', 'a2_47', 'a2_53', 
                   'a2_61', 'a2_69', 'a2_82', 'a2_85']:
            if col in self.symptom_mappings:
                semantic_name = self.symptom_mappings[col]
                if semantic_name in row and row[semantic_name]:
                    symptoms.append(f"{semantic_name}: {row[semantic_name]}")
        
        return f"Core symptoms: {', '.join(symptoms)}"
```

class AdvancedTabulaPreprocessor(TabulaVAPreprocessor):
    """Advanced preprocessing with feature selection and optimization"""
    
    def select_informative_features(self, df: pd.DataFrame, 
                                   target: str = 'gs_text34',
                                   max_features: int = 200) -> pd.DataFrame:
        """Select most informative features using mutual information"""
        from sklearn.feature_selection import mutual_info_classif
        
        # Separate features and target
        X = df.drop(columns=[target])
        y = df[target]
        
        # Calculate mutual information
        mi_scores = mutual_info_classif(X, y)
        
        # Select top features
        top_indices = np.argsort(mi_scores)[-max_features:]
        selected_features = X.columns[top_indices].tolist()
        
        return df[selected_features + [target]]
    
    def handle_class_imbalance(self, df: pd.DataFrame, 
                              target: str = 'gs_text34') -> pd.DataFrame:
        """Handle class imbalance using SMOTE and undersampling"""
        from imblearn.combine import SMOTEENN
        
        X = df.drop(columns=[target])
        y = df[target]
        
        # Apply hybrid resampling
        smote_enn = SMOTEENN(random_state=42)
        X_resampled, y_resampled = smote_enn.fit_resample(X, y)
        
        # Combine back into dataframe
        df_resampled = pd.concat([
            pd.DataFrame(X_resampled, columns=X.columns),
            pd.Series(y_resampled, name=target)
        ], axis=1)
        
        return df_resampled
```

#### 1.3 Model Loading & Configuration
```python
# File: src/models/tabula_model.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

class TabulaVAClassifier:
    def __init__(self, model_name: str = "mlfoundations/tabula-8b"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
    def setup_lora(self, rank: int = 64, alpha: int = 16):
        """Configure LoRA for efficient fine-tuning"""
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
```

**Deliverables:**
- [ ] Tabula-8B model successfully loaded and tested
- [ ] Data preprocessing pipeline for semantic feature mapping
- [ ] GPU memory optimization confirmed <20GB
- [ ] Basic inference functionality validated

### Phase 1.5: Tiered Feature Processing Strategy

#### Managing 1,649 Features Within Context Limits
```python
# File: src/data/feature_tier_manager.py
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np

class FeatureTierManager:
    """Manages tiered processing of 1,649 PHMRC features for Tabula-8B's 8,192 token limit"""
    
    def __init__(self, max_tokens_per_prompt: int = 2048):
        self.max_tokens = max_tokens_per_prompt
        self.feature_tiers = self._define_feature_tiers()
        
    def _define_feature_tiers(self) -> Dict[str, Dict]:
        """Define feature tiers based on predictive importance and token usage"""
        return {
            'tier_1_critical': {
                'features': [
                    # Core symptoms (a2 series) - highest signal
                    'a2_02', 'a2_32', 'a2_36', 'a2_43', 'a2_47',
                    'a2_53', 'a2_61', 'a2_69', 'a2_82', 'a2_85'
                ],
                'max_features': 50,
                'token_budget': 500,
                'description': 'Critical symptoms with highest predictive value'
            },
            'tier_2_medical_history': {
                'features': [
                    # Medical history (a1 series)
                    'a1_01_1', 'a1_01_3', 'a1_01_7', 'a1_01_9', 'a1_01_14'
                ],
                'max_features': 30,
                'token_budget': 300,
                'description': 'Pre-existing conditions and medical history'
            },
            'tier_3_demographics': {
                'features': [
                    # Demographics (g series)
                    'g1_05', 'g1_07a', 'g1_08', 'g1_09'
                ],
                'max_features': 20,
                'token_budget': 200,
                'description': 'Age, sex, education, marital status'
            },
            'tier_4_narrative': {
                'features': [
                    # Top narrative features (word_* series)
                    'word_fever', 'word_cough', 'word_pain', 'word_chest'
                ],
                'max_features': 100,
                'token_budget': 400,
                'description': 'Most informative narrative word counts'
            },
            'tier_5_supplementary': {
                'features': [],  # Remaining features
                'max_features': 200,
                'token_budget': 600,
                'description': 'Additional symptoms and risk factors'
            }
        }
    
    def create_multi_pass_strategy(self, patient_data: pd.Series) -> List[Dict]:
        """Create multiple prompts for multi-pass processing"""
        passes = []
        
        # Pass 1: Critical features only (fast, high confidence)
        pass1 = {
            'pass_id': 1,
            'name': 'critical_symptoms',
            'features': self._extract_tier_features(patient_data, 'tier_1_critical'),
            'purpose': 'Initial high-confidence classification',
            'expected_tokens': 500
        }
        passes.append(pass1)
        
        # Pass 2: Add medical history (for complex cases)
        pass2 = {
            'pass_id': 2,
            'name': 'with_medical_context',
            'features': self._extract_tier_features(patient_data, 
                                                   ['tier_1_critical', 'tier_2_medical_history']),
            'purpose': 'Refined classification with medical context',
            'expected_tokens': 800
        }
        passes.append(pass2)
        
        # Pass 3: Full context (for uncertain cases)
        pass3 = {
            'pass_id': 3,
            'name': 'comprehensive_analysis',
            'features': self._extract_all_tiers(patient_data),
            'purpose': 'Complete analysis for edge cases',
            'expected_tokens': 1800
        }
        passes.append(pass3)
        
        return passes
    
    def optimize_feature_selection(self, df: pd.DataFrame, 
                                  target: str = 'gs_text34') -> Dict[str, List[str]]:
        """Optimize feature selection per tier using information gain"""
        from sklearn.feature_selection import mutual_info_classif
        
        optimized_tiers = {}
        
        for tier_name, tier_config in self.feature_tiers.items():
            # Get features for this tier
            tier_features = [f for f in tier_config['features'] if f in df.columns]
            
            if not tier_features:
                continue
            
            # Calculate mutual information
            X_tier = df[tier_features]
            y = df[target]
            mi_scores = mutual_info_classif(X_tier, y, random_state=42)
            
            # Select top features within token budget
            n_features = min(len(tier_features), tier_config['max_features'])
            top_indices = np.argsort(mi_scores)[-n_features:]
            selected_features = [tier_features[i] for i in top_indices]
            
            optimized_tiers[tier_name] = selected_features
        
        return optimized_tiers
    
    def estimate_token_usage(self, features: Dict) -> int:
        """Estimate token usage for a set of features"""
        # Rough estimation: feature_name + value = ~10 tokens per feature
        tokens_per_feature = 10
        total_features = sum(len(v) for v in features.values() if isinstance(v, list))
        
        # Add overhead for prompt structure
        structure_overhead = 200
        
        return (total_features * tokens_per_feature) + structure_overhead
```

#### Adaptive Context Window Management
```python
# File: src/data/adaptive_context_manager.py
class AdaptiveContextManager:
    """Dynamically manages context window usage based on case complexity"""
    
    def __init__(self, base_window: int = 8192):
        self.base_window = base_window
        self.reserved_tokens = 1000  # Reserve for model response
        self.available_tokens = base_window - reserved_tokens
        
    def assess_case_complexity(self, patient_data: pd.Series) -> str:
        """Assess complexity to determine feature tier usage"""
        complexity_score = 0
        
        # Check symptom count
        symptom_count = sum(1 for col in patient_data.index 
                          if col.startswith('a2_') and patient_data[col] == 'Yes')
        
        if symptom_count > 20:
            complexity_score += 3
        elif symptom_count > 10:
            complexity_score += 2
        else:
            complexity_score += 1
        
        # Check for rare symptoms
        rare_symptoms = ['a2_85', 'a2_82']  # paralysis, seizures
        if any(patient_data.get(s) == 'Yes' for s in rare_symptoms):
            complexity_score += 2
        
        # Check narrative richness
        word_features = [col for col in patient_data.index if col.startswith('word_')]
        narrative_richness = sum(patient_data[col] for col in word_features if col in patient_data)
        
        if narrative_richness > 50:
            complexity_score += 2
        
        # Determine complexity level
        if complexity_score >= 6:
            return 'complex'
        elif complexity_score >= 3:
            return 'moderate'
        else:
            return 'simple'
    
    def select_features_for_complexity(self, patient_data: pd.Series, 
                                      complexity: str) -> Dict[str, List]:
        """Select appropriate features based on complexity"""
        feature_selection = {
            'simple': ['tier_1_critical'],
            'moderate': ['tier_1_critical', 'tier_2_medical_history', 'tier_3_demographics'],
            'complex': ['tier_1_critical', 'tier_2_medical_history', 
                       'tier_3_demographics', 'tier_4_narrative']
        }
        
        return feature_selection.get(complexity, ['tier_1_critical'])
```

### Phase 2: Baseline Implementation (Weeks 3-4)

#### 2.1 Zero-Shot Classification
```python
# File: src/experiments/zero_shot_tabula.py
class ZeroShotVAClassifier:
    def __init__(self, model: TabulaVAClassifier):
        self.model = model
        self.cause_descriptions = self._load_cause_descriptions()
    
    def classify(self, patient_data: dict) -> dict:
        """Zero-shot classification with cause descriptions"""
        prompt = self._create_classification_prompt(patient_data)
        
        # Generate classification
        inputs = self.model.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model.model.generate(
                inputs.input_ids.to(self.model.device),
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
        
        response = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return self._parse_classification(response)
    
    def _create_classification_prompt(self, data: dict) -> str:
        """Create structured medical prompt with PHMRC semantic features"""
        return f"""
        VERBAL AUTOPSY CLASSIFICATION TASK
        
        Patient Case Presentation:
        
        DEMOGRAPHICS:
        - Age: {data.get('age_at_death_years', 'Unknown')} years
        - Sex: {data.get('patient_sex', 'Unknown')}
        - Education: {data.get('education_level', 'Unknown')}
        - Marital Status: {data.get('marital_status', 'Unknown')}
        
        PRIMARY SYMPTOMS (Present):
        {self._format_symptoms(data)}
        
        MEDICAL HISTORY:
        {self._format_medical_history(data)}
        
        CLINICAL NARRATIVE INDICATORS:
        {self._format_narrative_features(data)}
        
        TASK: Classify the cause of death into one of these 34 categories:
        {', '.join(self.cause_descriptions.keys())}
        
        RESPONSE FORMAT:
        Cause: [exact_cause_name]
        Confidence: [0-100]%
        Key Evidence: [list 2-3 most relevant symptoms/features]
        """
    
    def _format_symptoms(self, data: dict) -> str:
        """Format symptom data using semantic names"""
        symptoms = []
        
        # Constitutional symptoms
        if data.get('patient_had_fever') == 'Yes':
            severity = data.get('fever_severity', 'unspecified')
            symptoms.append(f"- Fever ({severity})")
        
        # Respiratory symptoms
        if data.get('patient_had_cough') == 'Yes':
            symptoms.append("- Persistent cough")
            if data.get('cough_was_productive_with_sputum') == 'Yes':
                symptoms.append("  • Productive with sputum")
            if data.get('coughed_blood_hemoptysis') == 'Yes':
                symptoms.append("  • With blood (hemoptysis)")
        
        if data.get('had_difficulty_breathing_dyspnea') == 'Yes':
            symptoms.append("- Difficulty breathing (dyspnea)")
        
        # Cardiovascular symptoms
        if data.get('experienced_chest_pain') == 'Yes':
            duration = data.get('chest_pain_duration', 'unspecified')
            symptoms.append(f"- Chest pain (duration: {duration})")
        
        # Neurological symptoms
        if data.get('had_convulsions_or_seizures') == 'Yes':
            symptoms.append("- Convulsions/seizures")
        
        if data.get('experienced_paralysis') == 'Yes':
            symptoms.append("- Paralysis")
        
        return '\n'.join(symptoms) if symptoms else "No significant symptoms reported"
    
    def _format_medical_history(self, data: dict) -> str:
        """Format medical history using semantic names"""
        conditions = []
        
        history_mapping = {
            'history_of_diabetes': 'Diabetes',
            'history_of_hypertension': 'Hypertension',
            'history_of_heart_disease': 'Heart disease',
            'history_of_aids': 'AIDS/HIV',
            'history_of_cancer': 'Cancer',
            'history_of_stroke': 'Previous stroke',
            'history_of_copd': 'COPD',
            'history_of_asthma': 'Asthma'
        }
        
        for key, condition in history_mapping.items():
            if data.get(key) == 'Yes':
                conditions.append(f"- {condition}")
        
        return '\n'.join(conditions) if conditions else "No significant medical history"
    
    def _format_narrative_features(self, data: dict) -> str:
        """Format narrative word features"""
        narrative_indicators = []
        
        # Check top narrative features
        high_freq_words = []
        moderate_freq_words = []
        
        narrative_words = ['fever', 'cough', 'pain', 'chest', 'breath', 
                          'heart', 'accident', 'fall', 'pregnant']
        
        for word in narrative_words:
            word_key = f'word_{word}'
            if word_key in data:
                count = data[word_key]
                if count > 5:
                    high_freq_words.append(word)
                elif count > 0:
                    moderate_freq_words.append(word)
        
        if high_freq_words:
            narrative_indicators.append(f"Frequently mentioned: {', '.join(high_freq_words)}")
        if moderate_freq_words:
            narrative_indicators.append(f"Also mentioned: {', '.join(moderate_freq_words)}")
        
        return '\n'.join(narrative_indicators) if narrative_indicators else "No specific narrative patterns"
```

#### 2.2 Few-Shot Learning Setup
```python
# File: src/experiments/few_shot_tabula.py
class FewShotVAClassifier(ZeroShotVAClassifier):
    def __init__(self, model: TabulaVAClassifier, examples_per_class: int = 3):
        super().__init__(model)
        self.examples_per_class = examples_per_class
        self.exemplars = self._create_balanced_exemplars()
    
    def _create_balanced_exemplars(self) -> dict:
        """Create representative examples for each cause"""
        exemplars = {}
        for cause in self.cause_descriptions.keys():
            # Select diverse, high-quality examples
            examples = self._select_exemplars_for_cause(cause)
            exemplars[cause] = examples
        return exemplars
    
    def classify_with_examples(self, patient_data: dict) -> dict:
        """Few-shot classification with contextual examples"""
        # Select most relevant examples using similarity
        relevant_examples = self._select_relevant_examples(patient_data)
        
        prompt = self._create_few_shot_prompt(patient_data, relevant_examples)
        return self._generate_classification(prompt)
```

**Deliverables:**
- [ ] Zero-shot classification achieving >60% CSMF accuracy
- [ ] Few-shot implementation with example selection
- [ ] Comprehensive evaluation on PHMRC test set
- [ ] Performance comparison with existing baselines

### Phase 3: Fine-Tuning & Optimization (Weeks 5-7)

#### 3.1 LoRA Fine-Tuning Implementation
```python
# File: src/training/tabula_trainer.py
from transformers import TrainingArguments, Trainer
import wandb

class TabulaVATrainer:
    def __init__(self, model: TabulaVAClassifier, train_dataset, val_dataset):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
    def setup_training(self):
        """Configure training parameters for VA classification"""
        training_args = TrainingArguments(
            output_dir="./results/tabula_va_finetuning",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="csmf_accuracy",
            greater_is_better=True,
            report_to="wandb",
            fp16=True,
            dataloader_num_workers=8,
        )
        return training_args
    
    def train(self):
        """Execute fine-tuning with custom metrics"""
        training_args = self.setup_training()
        
        trainer = Trainer(
            model=self.model.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=self._compute_va_metrics,
            data_collator=self._va_data_collator,
        )
        
        trainer.train()
        return trainer
    
    def _compute_va_metrics(self, eval_pred):
        """Custom metrics for VA evaluation"""
        predictions, labels = eval_pred
        
        # Convert to cause predictions
        predicted_causes = self._decode_predictions(predictions)
        true_causes = self._decode_labels(labels)
        
        # Calculate VA-specific metrics
        csmf_accuracy = self._calculate_csmf_accuracy(predicted_causes, true_causes)
        cod_accuracy = accuracy_score(true_causes, predicted_causes)
        
        return {
            "csmf_accuracy": csmf_accuracy,
            "cod_accuracy": cod_accuracy,
            "per_cause_f1": self._calculate_per_cause_f1(predicted_causes, true_causes)
        }
```

#### 3.2 Advanced Training Strategies
```python
# File: src/training/advanced_strategies.py
class AdvancedVATraining:
    def __init__(self, model: TabulaVAClassifier):
        self.model = model
        
    def curriculum_learning(self, train_data: pd.DataFrame):
        """Implement curriculum learning for VA classification"""
        # Start with clear-cut cases, gradually add ambiguous ones
        easy_cases = self._identify_easy_cases(train_data)
        medium_cases = self._identify_medium_cases(train_data)
        hard_cases = self._identify_hard_cases(train_data)
        
        # Progressive training phases
        for phase, cases in enumerate([easy_cases, medium_cases, hard_cases]):
            print(f"Training Phase {phase + 1}: {len(cases)} cases")
            self._train_on_subset(cases, phase)
    
    def class_balanced_sampling(self, train_data: pd.DataFrame):
        """Implement balanced sampling for imbalanced classes"""
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        
        # Hybrid resampling strategy
        oversample = SMOTE(sampling_strategy='minority', k_neighbors=3)
        undersample = RandomUnderSampler(sampling_strategy='majority')
        
        # Apply to feature space only
        features = train_data.drop(['gs_text34'], axis=1)
        labels = train_data['gs_text34']
        
        X_resampled, y_resampled = oversample.fit_resample(features, labels)
        X_final, y_final = undersample.fit_resample(X_resampled, y_resampled)
        
        return pd.concat([pd.DataFrame(X_final), pd.Series(y_final, name='gs_text34')], axis=1)
    
    def multi_task_learning(self):
        """Implement multi-task learning for VA classification"""
        # Auxiliary tasks: symptom prediction, demographic prediction
        # Main task: cause classification
        pass
```

**Deliverables:**
- [ ] LoRA fine-tuned model achieving >72% CSMF accuracy
- [ ] Curriculum learning implementation
- [ ] Class imbalance mitigation strategies
- [ ] Comprehensive hyperparameter optimization

### Phase 4: Ensemble & Hybrid Methods (Weeks 8-10)

#### 4.1 Intelligent Ensemble Architecture
```python
# File: src/ensemble/hybrid_classifier.py
class HybridVAEnsemble:
    """Intelligent ensemble combining domain expertise with LLM capabilities"""
    
    def __init__(self, tabula_model, xgboost_model, insilico_model):
        self.tabula_model = tabula_model
        self.xgboost_model = xgboost_model
        self.insilico_model = insilico_model
        self.confidence_threshold = 0.8
        self.ensemble_weights = self._learn_ensemble_weights()
    
    def predict(self, patient_data: dict) -> dict:
        """Intelligent routing and ensemble prediction"""
        # Get predictions from all models
        tabula_pred = self.tabula_model.predict(patient_data)
        xgboost_pred = self.xgboost_model.predict(patient_data)
        insilico_pred = self.insilico_model.predict(patient_data)
        
        # Route based on confidence and case complexity
        if self._is_clear_cut_case(patient_data):
            # Use fastest, most confident model
            return self._select_highest_confidence([tabula_pred, xgboost_pred])
        
        elif self._requires_domain_expertise(patient_data):
            # Weight domain-expert models higher
            return self._weighted_ensemble(
                [tabula_pred, xgboost_pred, insilico_pred],
                weights=[0.4, 0.3, 0.3]
            )
        
        else:
            # Full ensemble for ambiguous cases
            return self._adaptive_ensemble([tabula_pred, xgboost_pred, insilico_pred])
    
    def _adaptive_ensemble(self, predictions: list) -> dict:
        """Adaptive ensemble based on prediction agreement"""
        agreements = self._calculate_prediction_agreement(predictions)
        
        if agreements > 0.7:  # High agreement
            return self._simple_average(predictions)
        else:  # Low agreement - use learned weights
            return self._weighted_ensemble(predictions, self.ensemble_weights)
    
    def _learn_ensemble_weights(self) -> np.ndarray:
        """Learn optimal ensemble weights through cross-validation"""
        # Implement Bayesian optimization for weight learning
        from skopt import gp_minimize
        from skopt.space import Real
        
        space = [Real(0, 1) for _ in range(3)]  # Weights for 3 models
        
        def objective(weights):
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            
            # Cross-validation performance with these weights
            cv_score = self._evaluate_ensemble_weights(weights)
            return -cv_score  # Minimize negative score
        
        result = gp_minimize(objective, space, n_calls=50)
        optimal_weights = np.array(result.x)
        return optimal_weights / optimal_weights.sum()
```

#### 4.2 Case Routing Logic
```python
# File: src/ensemble/case_router.py
class IntelligentCaseRouter:
    """Routes cases to optimal prediction strategy"""
    
    def __init__(self):
        self.complexity_classifier = self._train_complexity_classifier()
        self.confidence_calibrator = self._train_confidence_calibrator()
    
    def route_case(self, patient_data: dict) -> str:
        """Determine optimal prediction strategy"""
        complexity = self._assess_case_complexity(patient_data)
        symptom_clarity = self._assess_symptom_clarity(patient_data)
        narrative_quality = self._assess_narrative_quality(patient_data)
        
        if complexity == "simple" and symptom_clarity > 0.8:
            return "xgboost_only"
        elif narrative_quality > 0.7 and self._has_rare_symptoms(patient_data):
            return "tabula_primary"
        elif self._requires_medical_expertise(patient_data):
            return "insilico_primary"
        else:
            return "full_ensemble"
    
    def _assess_case_complexity(self, patient_data: dict) -> str:
        """Assess case complexity using learned features"""
        # Feature engineering for complexity assessment
        features = {
            'symptom_count': sum(patient_data.get('symptoms', {}).values()),
            'narrative_length': len(patient_data.get('narrative', '')),
            'contradictory_symptoms': self._count_contradictions(patient_data),
            'rare_symptom_combinations': self._detect_rare_combinations(patient_data)
        }
        
        complexity_score = self.complexity_classifier.predict_proba([list(features.values())])[0][1]
        
        if complexity_score < 0.3:
            return "simple"
        elif complexity_score < 0.7:
            return "moderate"
        else:
            return "complex"
```

**Deliverables:**
- [ ] Hybrid ensemble achieving >76% CSMF accuracy
- [ ] Intelligent case routing system
- [ ] Confidence calibration for clinical decision support
- [ ] Comprehensive ensemble evaluation

### Phase 5: Production Deployment (Weeks 11-12)

#### 5.1 Production API
```python
# File: src/api/va_classification_api.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import asyncio
import redis
from typing import Dict, List, Optional

app = FastAPI(title="Tabula VA Classification API", version="1.0.0")

class VAClassificationRequest(BaseModel):
    patient_id: str
    demographics: Dict[str, any]
    symptoms: Dict[str, bool]
    narrative: str
    site: Optional[str] = None
    priority: str = "normal"  # normal, high, urgent

class VAClassificationResponse(BaseModel):
    patient_id: str
    predicted_cause: str
    confidence: float
    alternative_causes: List[Dict[str, float]]
    reasoning: str
    processing_time: float
    model_version: str

@app.post("/classify", response_model=VAClassificationResponse)
async def classify_va(request: VAClassificationRequest, background_tasks: BackgroundTasks):
    """Classify VA case with production-grade error handling"""
    try:
        # Input validation
        validated_data = validate_va_input(request)
        
        # Route to appropriate model ensemble
        router = IntelligentCaseRouter()
        strategy = router.route_case(validated_data.dict())
        
        # Execute prediction
        start_time = time.time()
        result = await predict_with_strategy(validated_data, strategy)
        processing_time = time.time() - start_time
        
        # Log for monitoring
        background_tasks.add_task(log_prediction, request, result, processing_time)
        
        return VAClassificationResponse(
            patient_id=request.patient_id,
            predicted_cause=result['cause'],
            confidence=result['confidence'],
            alternative_causes=result['alternatives'],
            reasoning=result['reasoning'],
            processing_time=processing_time,
            model_version="tabula-ensemble-v1.0"
        )
        
    except Exception as e:
        logger.error(f"Classification error for patient {request.patient_id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Classification failed")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "gpu_memory": get_gpu_memory_usage(),
        "model_loaded": check_model_status(),
        "timestamp": datetime.now().isoformat()
    }
```

#### 5.2 Monitoring & Observability
```python
# File: src/monitoring/va_monitor.py
class VAClassificationMonitor:
    """Production monitoring for VA classification system"""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.drift_detector = DriftDetector()
        self.performance_tracker = PerformanceTracker()
    
    def monitor_prediction(self, request: dict, response: dict, processing_time: float):
        """Monitor individual predictions"""
        # Performance metrics
        self.metrics_collector.record_latency(processing_time)
        self.metrics_collector.record_confidence(response['confidence'])
        
        # Data drift detection
        input_features = self._extract_features(request)
        drift_score = self.drift_detector.detect_drift(input_features)
        
        if drift_score > 0.1:
            self._alert_drift_detected(drift_score, request['patient_id'])
        
        # Model performance tracking
        self.performance_tracker.update_metrics(response)
    
    def generate_daily_report(self) -> dict:
        """Generate daily performance report"""
        return {
            'total_predictions': self.metrics_collector.get_daily_count(),
            'average_latency': self.metrics_collector.get_average_latency(),
            'confidence_distribution': self.metrics_collector.get_confidence_distribution(),
            'cause_distribution': self.metrics_collector.get_cause_distribution(),
            'drift_alerts': self.drift_detector.get_daily_alerts(),
            'performance_degradation': self.performance_tracker.check_degradation()
        }
```

**Deliverables:**
- [ ] Production-ready API with <2s latency
- [ ] Comprehensive monitoring and alerting
- [ ] Automated model health checks
- [ ] Deployment documentation and runbooks

## Risk Mitigation Strategies

### Technical Risks

#### 1. GPU Memory Constraints
**Risk**: Tabula-8B requires significant GPU memory (19.2GB+)
**Mitigation**: 
- Implement model quantization (INT8/FP16)
- Use gradient checkpointing during training
- Implement dynamic batching with memory monitoring
- Fallback to smaller models if memory issues persist

#### 2. Model Performance Gap
**Risk**: Tabula-8B underperforms compared to XGBoost baseline
**Mitigation**:
- Comprehensive hyperparameter optimization
- Domain-specific fine-tuning with medical corpora
- Ensemble approach as primary strategy
- Gradual rollout with A/B testing

#### 3. Inference Latency
**Risk**: LLM inference too slow for production use
**Mitigation**:
- Model optimization with TensorRT/ONNX
- Caching for similar cases
- Asynchronous processing for non-urgent cases
- Tiered service levels (urgent vs. routine)

### Operational Risks

#### 1. Data Quality Degradation
**Risk**: Real-world data differs from PHMRC training data
**Mitigation**:
- Continuous data quality monitoring
- Drift detection with automated alerts
- Regular model retraining schedule
- Human-in-the-loop validation for low-confidence cases

#### 2. Regulatory Compliance
**Risk**: Model decisions lack medical interpretability
**Mitigation**:
- Maintain ensemble with interpretable models (InSilicoVA)
- Implement SHAP/LIME explanations
- Clinical validation studies
- Clear model limitations documentation

## Quality Assurance Framework

### Testing Strategy

#### 1. Unit Tests
```python
# File: tests/test_tabula_classifier.py
import pytest
import pandas as pd
import numpy as np
from src.models.tabula_model import TabulaVAClassifier
from src.data.tabula_preprocessor import TabulaVAPreprocessor

class TestTabulaVAClassifier:
    @pytest.fixture
    def classifier(self):
        return TabulaVAClassifier()
    
    def test_model_loading(self, classifier):
        """Test model loads successfully"""
        assert classifier.model is not None
        assert classifier.tokenizer is not None
    
    def test_inference_output_format(self, classifier):
        """Test inference returns expected format"""
        sample_input = {
            'age_at_death_years': 65,
            'patient_sex': 'Male',
            'patient_had_fever': 'Yes',
            'patient_had_cough': 'Yes',
            'word_fever': 3,
            'word_cough': 2
        }
        result = classifier.predict(sample_input)
        
        assert 'cause' in result
        assert 'confidence' in result
        assert 0 <= result['confidence'] <= 1
    
    def test_memory_usage(self, classifier):
        """Test GPU memory usage within limits"""
        import torch
        torch.cuda.empty_cache()
        
        # Run inference with semantic features
        sample_batch = [
            {'patient_had_fever': 'Yes', 'fever_severity': 'moderate'} 
            for _ in range(10)
        ]
        results = classifier.predict_batch(sample_batch)
        
        memory_used = torch.cuda.max_memory_allocated() / 1024**3  # GB
        assert memory_used < 20.0, f"Memory usage {memory_used:.1f}GB exceeds 20GB limit"

class TestSemanticTransformation:
    """Test suite for PHMRC semantic feature transformation"""
    
    @pytest.fixture
    def preprocessor(self):
        return TabulaVAPreprocessor(semantic_mapping=True)
    
    @pytest.fixture
    def sample_phmrc_data(self):
        """Create sample PHMRC data with original column codes"""
        return pd.DataFrame({
            # Demographics
            'g1_05': ['Male', 'Female'],
            'g1_07a': [65, 45],
            'g1_08': [2, 1],  # Married, Never married
            'g1_09': [2, 3],  # Primary, Secondary education
            
            # Symptoms (a2 series)
            'a2_02': ['Yes', 'No'],  # Fever
            'a2_04': [2, np.nan],  # Fever severity (moderate)
            'a2_32': ['Yes', 'Yes'],  # Cough
            'a2_36': ['No', 'Yes'],  # Breathing difficulty
            'a2_43': ['Yes', 'No'],  # Chest pain
            'a2_82': ['No', 'Yes'],  # Seizures
            'a2_85': ['No', 'No'],  # Paralysis
            
            # Medical history (a1 series)
            'a1_01_7': ['Yes', 'No'],  # Diabetes
            'a1_01_9': ['No', 'Yes'],  # Heart disease
            'a1_01_14': ['No', 'No'],  # AIDS
            
            # Narrative features
            'word_fever': [5, 0],
            'word_cough': [3, 8],
            'word_pain': [2, 1],
            
            # Target
            'gs_text34': ['Pneumonia', 'Stroke']
        })
    
    def test_demographic_mapping(self, preprocessor, sample_phmrc_data):
        """Test demographic column mapping"""
        df_semantic = preprocessor.create_semantic_features(sample_phmrc_data)
        
        # Check demographic mappings
        assert 'patient_sex' in df_semantic.columns
        assert 'age_at_death_years' in df_semantic.columns
        assert 'marital_status' in df_semantic.columns
        assert 'education_level' in df_semantic.columns
        
        # Verify values preserved
        assert df_semantic['patient_sex'].iloc[0] == 'Male'
        assert df_semantic['age_at_death_years'].iloc[0] == 65
    
    def test_symptom_mapping(self, preprocessor, sample_phmrc_data):
        """Test symptom column mapping"""
        df_semantic = preprocessor.create_semantic_features(sample_phmrc_data)
        
        # Check symptom mappings
        assert 'patient_had_fever' in df_semantic.columns
        assert 'patient_had_cough' in df_semantic.columns
        assert 'had_difficulty_breathing_dyspnea' in df_semantic.columns
        assert 'experienced_chest_pain' in df_semantic.columns
        assert 'had_convulsions_or_seizures' in df_semantic.columns
        
        # Original columns should be replaced
        assert 'a2_02' not in df_semantic.columns
        assert 'a2_32' not in df_semantic.columns
    
    def test_medical_history_mapping(self, preprocessor, sample_phmrc_data):
        """Test medical history mapping"""
        df_semantic = preprocessor.create_semantic_features(sample_phmrc_data)
        
        # Check medical history mappings
        assert 'history_of_diabetes' in df_semantic.columns
        assert 'history_of_heart_disease' in df_semantic.columns
        assert 'history_of_aids' in df_semantic.columns
        
        # Original columns should be replaced
        assert 'a1_01_7' not in df_semantic.columns
    
    def test_dont_know_handling(self, preprocessor):
        """Test that Don't Know responses are handled correctly"""
        data_with_dk = pd.DataFrame({
            'a2_02': ['Yes', 'No', "Don't Know"],
            'a2_32': ['Yes', "Don't Know", 'No'],
            'gs_text34': ['Pneumonia', 'Unknown', 'Stroke']
        })
        
        df_semantic = preprocessor.create_semantic_features(data_with_dk)
        
        # Don't Know should be mapped to None (excluded)
        assert df_semantic['patient_had_fever'].iloc[2] is None
        assert df_semantic['patient_had_cough'].iloc[1] is None
    
    def test_severity_encoding(self, preprocessor, sample_phmrc_data):
        """Test ordinal severity encoding"""
        df_semantic = preprocessor.create_semantic_features(sample_phmrc_data)
        
        # Check fever severity is properly encoded
        if 'fever_severity' in df_semantic.columns:
            assert df_semantic['fever_severity'].iloc[0] == 'moderate_fever'
    
    def test_symptom_clustering(self, preprocessor, sample_phmrc_data):
        """Test medical system clustering"""
        df_semantic = preprocessor.create_semantic_features(sample_phmrc_data)
        
        # Check that syndrome flags are created
        assert 'respiratory_syndrome' in df_semantic.columns
        assert 'cardiovascular_syndrome' in df_semantic.columns
    
    def test_prompt_generation(self, preprocessor, sample_phmrc_data):
        """Test prompt generation with semantic features"""
        df_semantic = preprocessor.create_semantic_features(sample_phmrc_data)
        
        # Generate prompt for first patient
        prompt = preprocessor.format_for_tabula(df_semantic.iloc[0])
        
        # Check prompt contains semantic names, not codes
        assert 'patient_had_fever' in prompt or 'Fever' in prompt
        assert 'a2_02' not in prompt  # Original code should not appear
        assert 'Age at death: 65 years' in prompt
        assert 'Sex: Male' in prompt
    
    def test_feature_tier_grouping(self, preprocessor):
        """Test feature tier grouping for context management"""
        feature_groups = preprocessor._create_feature_groups()
        
        # Check tier structure
        assert 'tier1_core_symptoms' in feature_groups
        assert 'tier2_medical_history' in feature_groups
        assert 'tier3_demographics' in feature_groups
        assert 'tier4_narrative' in feature_groups
        
        # Verify core symptoms include critical features
        core_symptoms = feature_groups['tier1_core_symptoms']
        assert 'a2_02' in core_symptoms  # Fever
        assert 'a2_32' in core_symptoms  # Cough
        assert 'a2_82' in core_symptoms  # Seizures
    
    def test_tiered_prompt_creation(self, preprocessor, sample_phmrc_data):
        """Test multi-tier prompt creation"""
        df_semantic = preprocessor.create_semantic_features(sample_phmrc_data)
        
        # Create tiered prompts
        prompts = preprocessor.create_tiered_prompts(df_semantic.iloc[0])
        
        # Should have multiple prompts for different tiers
        assert len(prompts) >= 1
        
        # First prompt should be concise (tier 1 only)
        assert len(prompts[0]) < 1000  # Characters
    
    def test_all_1649_features_mapped(self, preprocessor):
        """Test that mapping covers all expected PHMRC features"""
        all_mappings = preprocessor.all_mappings
        
        # Check we have mappings for major categories
        demographic_cols = [k for k in all_mappings.keys() if k.startswith('g')]
        symptom_cols = [k for k in all_mappings.keys() if k.startswith('a2_')]
        history_cols = [k for k in all_mappings.keys() if k.startswith('a1_')]
        
        assert len(demographic_cols) >= 4  # At least core demographics
        assert len(symptom_cols) >= 10  # At least core symptoms
        assert len(history_cols) >= 5  # At least major conditions
```

#### 2. Integration Tests
```python
# File: tests/test_ensemble_integration.py
class TestHybridEnsemble:
    def test_ensemble_consistency(self):
        """Test ensemble produces consistent results"""
        ensemble = HybridVAEnsemble()
        
        # Same input should produce same output
        sample_input = self._create_sample_patient()
        
        result1 = ensemble.predict(sample_input)
        result2 = ensemble.predict(sample_input)
        
        assert result1['cause'] == result2['cause']
        assert abs(result1['confidence'] - result2['confidence']) < 0.01
    
    def test_performance_vs_baseline(self):
        """Test ensemble outperforms individual models"""
        ensemble = HybridVAEnsemble()
        test_data = load_test_dataset()
        
        ensemble_accuracy = evaluate_model(ensemble, test_data)
        xgboost_accuracy = evaluate_model(ensemble.xgboost_model, test_data)
        
        assert ensemble_accuracy > xgboost_accuracy
```

#### 3. Performance Tests
```python
# File: tests/test_performance.py
import time
import pytest

class TestPerformance:
    def test_inference_latency(self):
        """Test inference meets latency requirements"""
        classifier = TabulaVAClassifier()
        sample_input = self._create_sample_patient()
        
        start_time = time.time()
        result = classifier.predict(sample_input)
        inference_time = time.time() - start_time
        
        assert inference_time < 2.0, f"Inference took {inference_time:.2f}s, exceeds 2s limit"
    
    def test_batch_throughput(self):
        """Test batch processing throughput"""
        classifier = TabulaVAClassifier()
        batch_size = 100
        batch = [self._create_sample_patient() for _ in range(batch_size)]
        
        start_time = time.time()
        results = classifier.predict_batch(batch)
        total_time = time.time() - start_time
        
        throughput = batch_size / total_time
        assert throughput > 10, f"Throughput {throughput:.1f} samples/sec below target"
```

### Validation Framework

#### 1. Cross-Site Validation
```python
# File: src/validation/cross_site_validation.py
class CrossSiteValidator:
    """Validate model performance across different sites"""
    
    def __init__(self, model, phmrc_data):
        self.model = model
        self.phmrc_data = phmrc_data
        self.sites = phmrc_data['site'].unique()
    
    def validate_cross_site_performance(self):
        """Validate model performance across all PHMRC sites"""
        results = {}
        
        for site in self.sites:
            site_data = self.phmrc_data[self.phmrc_data['site'] == site]
            site_results = self._evaluate_on_site(site_data)
            results[site] = site_results
        
        # Check for significant performance drops
        mean_csmf = np.mean([r['csmf_accuracy'] for r in results.values()])
        min_csmf = np.min([r['csmf_accuracy'] for r in results.values()])
        
        if (mean_csmf - min_csmf) > 0.1:  # More than 10% drop
            self._alert_site_performance_issue(results)
        
        return results
    
    def _evaluate_on_site(self, site_data: pd.DataFrame) -> dict:
        """Evaluate model on specific site data"""
        X_test = site_data.drop(['gs_text34'], axis=1)
        y_test = site_data['gs_text34']
        
        predictions = self.model.predict(X_test)
        
        return {
            'csmf_accuracy': calculate_csmf_accuracy(y_test, predictions),
            'cod_accuracy': accuracy_score(y_test, predictions),
            'per_cause_performance': classification_report(y_test, predictions, output_dict=True),
            'sample_size': len(site_data)
        }
```

## Deployment Strategy

### Staging Environment
1. **Model Validation**: Comprehensive testing on held-out validation sets
2. **Performance Benchmarking**: Latency, throughput, and memory usage validation
3. **Integration Testing**: API endpoint and monitoring system validation
4. **Load Testing**: Stress testing with concurrent requests
5. **Clinical Review**: Medical expert validation of classification results

### Production Rollout
1. **Shadow Mode** (Week 11): Run alongside existing system without affecting decisions
2. **Canary Deployment** (Week 12): Route 5% of traffic to new system
3. **Gradual Rollout**: Increase to 25%, 50%, 100% based on performance metrics
4. **Monitoring**: Continuous monitoring of all key metrics
5. **Rollback Plan**: Immediate rollback capability if performance degrades

### Success Criteria

#### Technical Metrics
- **CSMF Accuracy**: >75% (vs. XGBoost baseline 74.84%)
- **COD Accuracy**: >58% (vs. XGBoost baseline 56.92%)
- **Inference Latency**: <2 seconds per prediction
- **System Uptime**: >99.5%
- **GPU Memory Usage**: <20GB peak

#### Clinical Metrics
- **Medical Expert Agreement**: >85% for high-confidence predictions
- **Interpretability Score**: >4.0/5.0 from clinical users
- **False Positive Rate**: <5% for critical causes (maternal, child health)

#### Operational Metrics
- **Processing Throughput**: >1000 cases/hour
- **System Reliability**: <0.1% error rate
- **Data Drift Detection**: <24 hour alert response time

## Long-Term Roadmap

### Phase 6: Advanced Features (Months 4-6)
- **Multi-language Support**: Extend to non-English VA narratives
- **Temporal Analysis**: Incorporate disease trend analysis
- **Uncertainty Quantification**: Bayesian approaches for confidence estimation
- **Active Learning**: Continuous improvement with expert feedback

### Phase 7: Research Extensions (Months 7-12)
- **Causal Inference**: Move beyond classification to causal reasoning
- **Multi-modal Integration**: Incorporate additional data types (lab results, imaging)
- **Federated Learning**: Enable multi-site training while preserving privacy
- **Explainable AI**: Advanced interpretability for clinical decision support

## Resource Requirements

### Team Structure
- **ML Engineer** (1.0 FTE): Model implementation and optimization
- **Data Scientist** (0.5 FTE): Evaluation and validation
- **DevOps Engineer** (0.5 FTE): Infrastructure and deployment
- **Clinical Advisor** (0.25 FTE): Medical validation and requirements
- **Product Manager** (0.25 FTE): Coordination and stakeholder management

### Budget Estimates
- **Infrastructure**: $3,000/month (GPU compute, storage, monitoring)
- **Development Tools**: $1,000/month (MLflow, Weights & Biases, monitoring)
- **Personnel**: $50,000/month (based on team structure)
- **Total 3-month implementation**: ~$162,000

## Conclusion

This comprehensive plan provides a structured approach to implementing Tabula-8B for VA classification while maintaining high performance standards and production reliability. The phased approach allows for iterative improvement and risk mitigation, while the hybrid ensemble strategy leverages both domain expertise and modern LLM capabilities.

The success of this implementation will establish a new state-of-the-art for automated VA classification and provide a foundation for future AI-driven public health applications.

---

**Next Steps**: 
1. Secure computational resources and team assignments
2. Begin Phase 1 implementation
3. Establish monitoring and evaluation frameworks
4. Initiate clinical partnership for validation studies