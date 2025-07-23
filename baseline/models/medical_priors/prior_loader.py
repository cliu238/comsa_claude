"""Loader for medical prior probability data from InSilicoVA.

This module handles extraction and parsing of conditional probability tables
and cause priors from InSilicoVA's data files.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MedicalPriors:
    """Container for medical prior probabilities.
    
    Attributes:
        conditional_probs: Mapping of (symptom, cause) to probability
        cause_priors: Population-level cause probabilities
        symptom_names: Ordered list of symptom names
        cause_names: Ordered list of cause names
        implausible_patterns: List of medically impossible combinations
        conditional_matrix: Numpy array of shape (n_symptoms, n_causes)
    """
    conditional_probs: Dict[Tuple[str, str], float]
    cause_priors: Dict[str, float]
    symptom_names: List[str]
    cause_names: List[str]
    implausible_patterns: List[Tuple[str, str]]
    conditional_matrix: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Create conditional probability matrix for efficient computation."""
        if self.conditional_matrix is None:
            n_symptoms = len(self.symptom_names)
            n_causes = len(self.cause_names)
            self.conditional_matrix = np.zeros((n_symptoms, n_causes))
            
            for i, symptom in enumerate(self.symptom_names):
                for j, cause in enumerate(self.cause_names):
                    key = (symptom, cause)
                    if key in self.conditional_probs:
                        self.conditional_matrix[i, j] = self.conditional_probs[key]


class PriorLoader:
    """Loads and processes medical prior data."""
    
    def __init__(self, prior_data_path: Optional[Path] = None):
        """Initialize prior loader.
        
        Args:
            prior_data_path: Path to directory containing prior data files.
                           If None, uses default simulated priors.
        """
        self.prior_data_path = prior_data_path
        self._priors: Optional[MedicalPriors] = None
        
    def load_priors(self) -> MedicalPriors:
        """Load medical priors from data files.
        
        Returns:
            MedicalPriors object containing all prior information
        """
        if self._priors is not None:
            return self._priors
            
        if self.prior_data_path is None:
            logger.info("No prior data path specified, using simulated priors")
            self._priors = self._create_simulated_priors()
        else:
            logger.info(f"Loading priors from {self.prior_data_path}")
            self._priors = self._load_from_files()
            
        return self._priors
    
    def _create_simulated_priors(self) -> MedicalPriors:
        """Create simulated medical priors for testing.
        
        This creates realistic-looking priors based on common VA patterns.
        """
        # Common symptoms in VA data
        symptoms = [
            "fever", "cough", "diarrhea", "vomiting", "headache",
            "chest_pain", "difficulty_breathing", "abdominal_pain",
            "skin_rash", "weight_loss", "fatigue", "convulsions",
            "injury", "bleeding", "swelling"
        ]
        
        # Common causes of death
        causes = [
            "tuberculosis", "aids", "malaria", "pneumonia", "diarrheal_diseases",
            "road_traffic", "stroke", "ischemic_heart", "diabetes", "copd",
            "maternal", "suicide", "homicide", "drowning", "fires"
        ]
        
        # Create conditional probabilities based on medical knowledge
        conditional_probs = {}
        
        # Define symptom-cause associations (simplified medical knowledge)
        associations = {
            ("fever", "malaria"): 0.9,
            ("fever", "tuberculosis"): 0.7,
            ("fever", "pneumonia"): 0.8,
            ("fever", "aids"): 0.6,
            ("cough", "tuberculosis"): 0.9,
            ("cough", "pneumonia"): 0.85,
            ("cough", "copd"): 0.95,
            ("diarrhea", "diarrheal_diseases"): 0.95,
            ("diarrhea", "aids"): 0.5,
            ("chest_pain", "ischemic_heart"): 0.8,
            ("chest_pain", "pneumonia"): 0.6,
            ("difficulty_breathing", "pneumonia"): 0.85,
            ("difficulty_breathing", "copd"): 0.9,
            ("difficulty_breathing", "tuberculosis"): 0.6,
            ("injury", "road_traffic"): 0.95,
            ("injury", "homicide"): 0.9,
            ("injury", "suicide"): 0.7,
            ("bleeding", "maternal"): 0.7,
            ("bleeding", "road_traffic"): 0.8,
            ("convulsions", "malaria"): 0.6,
            ("convulsions", "stroke"): 0.7,
            ("weight_loss", "tuberculosis"): 0.8,
            ("weight_loss", "aids"): 0.85,
            ("weight_loss", "diabetes"): 0.5,
        }
        
        # Fill in base probabilities
        for symptom in symptoms:
            for cause in causes:
                key = (symptom, cause)
                if key in associations:
                    conditional_probs[key] = associations[key]
                else:
                    # Low base probability for unassociated symptoms
                    conditional_probs[key] = 0.1
        
        # Create cause priors (rough global mortality fractions)
        cause_priors = {
            "tuberculosis": 0.03,
            "aids": 0.02,
            "malaria": 0.015,
            "pneumonia": 0.04,
            "diarrheal_diseases": 0.03,
            "road_traffic": 0.025,
            "stroke": 0.11,
            "ischemic_heart": 0.16,
            "diabetes": 0.03,
            "copd": 0.05,
            "maternal": 0.005,
            "suicide": 0.015,
            "homicide": 0.008,
            "drowning": 0.007,
            "fires": 0.005,
        }
        
        # Normalize to sum to 1
        total = sum(cause_priors.values())
        cause_priors = {k: v/total for k, v in cause_priors.items()}
        
        # Define implausible patterns
        implausible_patterns = [
            ("injury", "diabetes"),  # Injury doesn't cause diabetes
            ("injury", "tuberculosis"),  # Injury doesn't cause TB
            ("maternal", "road_traffic"),  # Maternal symptoms incompatible with traffic
            ("bleeding", "copd"),  # Bleeding not typical for COPD
        ]
        
        return MedicalPriors(
            conditional_probs=conditional_probs,
            cause_priors=cause_priors,
            symptom_names=symptoms,
            cause_names=causes,
            implausible_patterns=implausible_patterns
        )
    
    def _load_from_files(self) -> MedicalPriors:
        """Load priors from actual InSilicoVA data files.
        
        Loads the conditional probability matrices extracted from InSilicoVA R package.
        """
        try:
            # Check for required files
            symptoms_file = self.prior_data_path / "symptoms.csv"
            causes_file = self.prior_data_path / "causes.csv"
            
            # Try numeric probabilities first, then alphabetic
            condprobnum_file = self.prior_data_path / "condprobnum.csv"
            condprobnum_converted = self.prior_data_path / "condprobnum_converted.csv"
            condprob_file = self.prior_data_path / "condprob.csv"
            
            # Load symptoms and causes
            if not symptoms_file.exists() or not causes_file.exists():
                logger.warning(f"Required files not found in {self.prior_data_path}")
                return self._create_simulated_priors()
                
            symptoms_df = pd.read_csv(symptoms_file)
            causes_df = pd.read_csv(causes_file)
            
            symptom_names = symptoms_df['symptom'].tolist()
            cause_names = causes_df['cause'].tolist()
            
            # Load conditional probabilities
            if condprobnum_file.exists():
                logger.info("Loading numeric conditional probabilities")
                prob_df = pd.read_csv(condprobnum_file)
            elif condprobnum_converted.exists():
                logger.info("Loading converted numeric probabilities")
                prob_df = pd.read_csv(condprobnum_converted)
            elif condprob_file.exists():
                logger.info("Loading and converting alphabetic probabilities")
                prob_df = pd.read_csv(condprob_file)
                prob_df = self._convert_alphabetic_probs(prob_df)
            else:
                logger.warning("No probability files found")
                return self._create_simulated_priors()
            
            # Build conditional probability dictionary
            conditional_probs = {}
            symptom_col = 'symptom'
            
            for _, row in prob_df.iterrows():
                symptom = row[symptom_col]
                for cause in cause_names:
                    if cause in row:
                        prob = row[cause]
                        if pd.notna(prob):
                            conditional_probs[(symptom, cause)] = float(prob)
                        else:
                            conditional_probs[(symptom, cause)] = 0.1  # Default
                            
            # Create cause priors (uniform if not available)
            # In real InSilicoVA, these would come from population data
            n_causes = len(cause_names)
            cause_priors = {cause: 1.0/n_causes for cause in cause_names}
            
            # Define implausible patterns based on medical knowledge
            # These would ideally come from InSilicoVA's constraints
            implausible_patterns = self._get_insilico_implausible_patterns(symptom_names, cause_names)
            
            logger.info(f"Loaded {len(symptom_names)} symptoms and {len(cause_names)} causes")
            
            return MedicalPriors(
                conditional_probs=conditional_probs,
                cause_priors=cause_priors,
                symptom_names=symptom_names,
                cause_names=cause_names,
                implausible_patterns=implausible_patterns
            )
            
        except Exception as e:
            logger.error(f"Error loading InSilicoVA data: {e}")
            logger.warning("Falling back to simulated priors")
            return self._create_simulated_priors()
    
    def _convert_alphabetic_probs(self, prob_df: pd.DataFrame) -> pd.DataFrame:
        """Convert InSilicoVA alphabetic probability codes to numeric.
        
        Args:
            prob_df: DataFrame with alphabetic codes
            
        Returns:
            DataFrame with numeric probabilities
        """
        # InSilicoVA probability mapping
        prob_mapping = {
            'A': 0.9,    # Very likely
            'B': 0.65,   # Likely
            'C': 0.35,   # Possible
            'D': 0.125,  # Unlikely
            'E': 0.03,   # Very unlikely
            'N': 0.0,    # Never
            'Y': 1.0,    # Always
            '': 0.1,     # Default/missing
            '-': 0.1,    # Not applicable
        }
        
        numeric_df = prob_df.copy()
        symptom_col = 'symptom'
        
        for col in numeric_df.columns:
            if col != symptom_col:
                numeric_df[col] = numeric_df[col].apply(
                    lambda x: prob_mapping.get(str(x).upper().strip(), 0.1)
                )
                
        return numeric_df
    
    def _get_insilico_implausible_patterns(
        self, 
        symptom_names: List[str], 
        cause_names: List[str]
    ) -> List[Tuple[str, str]]:
        """Get medically implausible symptom-cause patterns.
        
        Based on medical knowledge encoded in InSilicoVA.
        
        Args:
            symptom_names: List of available symptoms
            cause_names: List of available causes
            
        Returns:
            List of (symptom, cause) tuples that are implausible
        """
        implausible = []
        
        # Define some known implausible patterns
        # These would be extracted from InSilicoVA's constraints
        implausible_rules = [
            # Maternal symptoms incompatible with non-maternal causes
            ("pregnant", "Road traffic"),
            ("delivery", "Suicide"),
            # Age-specific incompatibilities
            ("baby", "Stroke"),
            ("elder", "Maternal"),
            # Gender-specific incompatibilities (if gender info available)
            ("pregnant", "Prostate"),
        ]
        
        # Add patterns that exist in our data
        for symptom, cause in implausible_rules:
            # Check if both symptom and cause exist in our data
            symptom_matches = [s for s in symptom_names if symptom.lower() in s.lower()]
            cause_matches = [c for c in cause_names if cause.lower() in c.lower()]
            
            for s in symptom_matches:
                for c in cause_matches:
                    implausible.append((s, c))
                    
        return implausible
    
    def save_priors(self, priors: MedicalPriors, output_path: Path) -> None:
        """Save priors to CSV files for inspection.
        
        Args:
            priors: MedicalPriors object to save
            output_path: Directory to save files to
        """
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save conditional probabilities
        cond_data = []
        for (symptom, cause), prob in priors.conditional_probs.items():
            cond_data.append({"symptom": symptom, "cause": cause, "probability": prob})
        pd.DataFrame(cond_data).to_csv(output_path / "conditional_probs.csv", index=False)
        
        # Save cause priors
        cause_df = pd.DataFrame(
            list(priors.cause_priors.items()),
            columns=["cause", "prior_probability"]
        )
        cause_df.to_csv(output_path / "cause_priors.csv", index=False)
        
        # Save implausible patterns
        imp_df = pd.DataFrame(
            priors.implausible_patterns,
            columns=["symptom", "cause"]
        )
        imp_df.to_csv(output_path / "implausible_patterns.csv", index=False)
        
        logger.info(f"Saved priors to {output_path}")