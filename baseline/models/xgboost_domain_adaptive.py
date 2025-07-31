"""Domain-adaptive XGBoost model for better cross-site generalization.

This module implements advanced domain adaptation techniques including:
- Multi-task learning with site-specific heads
- Feature alignment across domains
- Adaptive instance weighting
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from baseline.models.xgboost_advanced_model import XGBoostAdvancedModel
from baseline.models.xgboost_enhanced_config import XGBoostEnhancedConfig

logger = logging.getLogger(__name__)


class XGBoostDomainAdaptive:
    """Domain-adaptive XGBoost using multi-task learning.
    
    This model trains separate models for each domain/site while sharing
    information through a common feature representation, improving both
    in-domain and out-domain performance.
    """
    
    def __init__(
        self,
        base_config: Optional[XGBoostEnhancedConfig] = None,
        adaptation_strategy: str = "multi_task",
        feature_alignment: bool = True,
        instance_weighting: bool = True,
        shared_layers_ratio: float = 0.7,
    ):
        """Initialize domain-adaptive model.
        
        Args:
            base_config: Base configuration for XGBoost models
            adaptation_strategy: Strategy for domain adaptation
                                ("multi_task", "feature_align", "instance_weight")
            feature_alignment: Whether to align features across domains
            instance_weighting: Whether to use adaptive instance weighting
            shared_layers_ratio: Ratio of trees shared across domains
        """
        self.base_config = base_config or XGBoostEnhancedConfig()
        self.adaptation_strategy = adaptation_strategy
        self.feature_alignment = feature_alignment
        self.instance_weighting = instance_weighting
        self.shared_layers_ratio = shared_layers_ratio
        
        # Storage for domain-specific models
        self.domain_models_: Dict[str, XGBoostAdvancedModel] = {}
        self.shared_model_: Optional[XGBoostAdvancedModel] = None
        self.feature_aligners_: Dict[str, StandardScaler] = {}
        self.domain_weights_: Dict[str, np.ndarray] = {}
        
        # Global feature statistics
        self.global_feature_stats_: Optional[Dict[str, float]] = None
        self.domains_: List[str] = []
    
    def fit(
        self,
        data_by_domain: Dict[str, Tuple[pd.DataFrame, pd.Series]],
        validation_data: Optional[Dict[str, Tuple[pd.DataFrame, pd.Series]]] = None,
    ) -> "XGBoostDomainAdaptive":
        """Fit domain-adaptive model.
        
        Args:
            data_by_domain: Dictionary mapping domain names to (X, y) tuples
            validation_data: Optional validation data by domain
            
        Returns:
            Self: Fitted model
        """
        self.domains_ = list(data_by_domain.keys())
        logger.info(f"Training domain-adaptive model for domains: {self.domains_}")
        
        # Step 1: Compute global feature statistics
        self._compute_global_statistics(data_by_domain)
        
        # Step 2: Align features across domains if enabled
        if self.feature_alignment:
            aligned_data = self._align_features(data_by_domain)
        else:
            aligned_data = data_by_domain
        
        # Step 3: Compute instance weights if enabled
        if self.instance_weighting:
            self._compute_instance_weights(aligned_data)
        
        # Step 4: Train models based on strategy
        if self.adaptation_strategy == "multi_task":
            self._train_multi_task(aligned_data, validation_data)
        elif self.adaptation_strategy == "feature_align":
            self._train_with_feature_alignment(aligned_data, validation_data)
        elif self.adaptation_strategy == "instance_weight":
            self._train_with_instance_weighting(aligned_data, validation_data)
        else:
            raise ValueError(f"Unknown adaptation strategy: {self.adaptation_strategy}")
        
        return self
    
    def predict(
        self,
        X: pd.DataFrame,
        source_domain: Optional[str] = None,
        target_domain: Optional[str] = None,
    ) -> np.ndarray:
        """Make predictions with domain adaptation.
        
        Args:
            X: Features to predict
            source_domain: Source domain (for domain-specific model)
            target_domain: Target domain (for adaptation)
            
        Returns:
            Predictions
        """
        # Apply feature alignment if used during training
        if self.feature_alignment and target_domain in self.feature_aligners_:
            X_aligned = self.feature_aligners_[target_domain].transform(X)
            X = pd.DataFrame(X_aligned, columns=X.columns, index=X.index)
        
        if self.adaptation_strategy == "multi_task":
            return self._predict_multi_task(X, source_domain, target_domain)
        else:
            # Use shared model for other strategies
            return self.shared_model_.predict(X)
    
    def _compute_global_statistics(
        self, data_by_domain: Dict[str, Tuple[pd.DataFrame, pd.Series]]
    ) -> None:
        """Compute global feature statistics across all domains."""
        all_features = []
        
        for domain, (X, _) in data_by_domain.items():
            all_features.append(X)
        
        # Concatenate all features
        all_X = pd.concat(all_features, ignore_index=True)
        
        # Compute statistics
        self.global_feature_stats_ = {
            "mean": all_X.mean(),
            "std": all_X.std(),
            "min": all_X.min(),
            "max": all_X.max(),
        }
    
    def _align_features(
        self, data_by_domain: Dict[str, Tuple[pd.DataFrame, pd.Series]]
    ) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
        """Align features across domains to reduce domain shift."""
        aligned_data = {}
        
        # Fit scalers for each domain
        for domain, (X, y) in data_by_domain.items():
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Store scaler for prediction time
            self.feature_aligners_[domain] = scaler
            
            # Apply global statistics adjustment
            X_aligned = X_scaled * self.global_feature_stats_["std"].values
            X_aligned = X_aligned + self.global_feature_stats_["mean"].values
            
            X_aligned_df = pd.DataFrame(X_aligned, columns=X.columns, index=X.index)
            aligned_data[domain] = (X_aligned_df, y)
        
        return aligned_data
    
    def _compute_instance_weights(
        self, data_by_domain: Dict[str, Tuple[pd.DataFrame, pd.Series]]
    ) -> None:
        """Compute instance weights based on domain similarity."""
        # For each domain, compute weights based on similarity to other domains
        for target_domain in self.domains_:
            X_target, _ = data_by_domain[target_domain]
            weights = np.ones(len(X_target))
            
            # Compute similarity to other domains
            for source_domain in self.domains_:
                if source_domain == target_domain:
                    continue
                
                X_source, _ = data_by_domain[source_domain]
                
                # Simple density-based weighting
                # Higher weight for instances similar to other domains
                for i in range(len(X_target)):
                    # Find nearest neighbors in source domain
                    distances = np.linalg.norm(
                        X_source.values - X_target.iloc[i].values, axis=1
                    )
                    min_dist = np.min(distances)
                    
                    # Weight based on distance (closer = higher weight)
                    weight = np.exp(-min_dist / X_target.shape[1])
                    weights[i] *= weight
            
            # Normalize weights
            weights = weights / weights.mean()
            self.domain_weights_[target_domain] = weights
    
    def _train_multi_task(
        self,
        data_by_domain: Dict[str, Tuple[pd.DataFrame, pd.Series]],
        validation_data: Optional[Dict[str, Tuple[pd.DataFrame, pd.Series]]],
    ) -> None:
        """Train multi-task model with shared and domain-specific components."""
        # Step 1: Train shared model on all data
        logger.info("Training shared model on combined data")
        
        all_X = []
        all_y = []
        all_domains = []
        
        for domain, (X, y) in data_by_domain.items():
            all_X.append(X)
            all_y.append(y)
            all_domains.extend([domain] * len(X))
        
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        domain_labels = pd.Series(all_domains)
        
        # Configure shared model with fewer trees
        shared_config = XGBoostEnhancedConfig(
            **self.base_config.model_dump(),
            n_estimators=int(self.base_config.n_estimators * self.shared_layers_ratio),
        )
        
        self.shared_model_ = XGBoostAdvancedModel(
            config=shared_config,
            use_custom_objective=True,
            objective_type="domain_adversarial",
            domain_adaptation=True,
        )
        
        # Add validation set if available
        eval_set = None
        if validation_data:
            val_X = []
            val_y = []
            for domain, (X_val, y_val) in validation_data.items():
                val_X.append(X_val)
                val_y.append(y_val)
            
            if val_X:
                eval_set = [(pd.concat(val_X), pd.concat(val_y))]
        
        self.shared_model_.fit(
            X_combined,
            y_combined,
            domain_labels=domain_labels,
            eval_set=eval_set,
        )
        
        # Step 2: Train domain-specific models
        logger.info("Training domain-specific models")
        
        remaining_trees = int(
            self.base_config.n_estimators * (1 - self.shared_layers_ratio)
        )
        
        for domain in self.domains_:
            logger.info(f"Training model for domain: {domain}")
            
            X_domain, y_domain = data_by_domain[domain]
            
            # Get instance weights if computed
            sample_weight = None
            if domain in self.domain_weights_:
                sample_weight = self.domain_weights_[domain]
            
            # Configure domain-specific model
            domain_config = XGBoostEnhancedConfig(
                **self.base_config.model_dump(),
                n_estimators=remaining_trees,
            )
            
            domain_model = XGBoostAdvancedModel(
                config=domain_config,
                use_custom_objective=True,
                objective_type="csmf_weighted",
            )
            
            # Add validation set for this domain
            eval_set = None
            if validation_data and domain in validation_data:
                X_val, y_val = validation_data[domain]
                eval_set = [(X_val, y_val)]
            
            domain_model.fit(
                X_domain,
                y_domain,
                sample_weight=sample_weight,
                eval_set=eval_set,
            )
            
            self.domain_models_[domain] = domain_model
    
    def _train_with_feature_alignment(
        self,
        data_by_domain: Dict[str, Tuple[pd.DataFrame, pd.Series]],
        validation_data: Optional[Dict[str, Tuple[pd.DataFrame, pd.Series]]],
    ) -> None:
        """Train single model with feature-aligned data."""
        # Combine all aligned data
        all_X = []
        all_y = []
        
        for domain, (X, y) in data_by_domain.items():
            all_X.append(X)
            all_y.append(y)
        
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        
        # Train single model
        self.shared_model_ = XGBoostAdvancedModel(
            config=self.base_config,
            use_custom_objective=True,
            objective_type="csmf_weighted",
        )
        
        self.shared_model_.fit(X_combined, y_combined)
    
    def _train_with_instance_weighting(
        self,
        data_by_domain: Dict[str, Tuple[pd.DataFrame, pd.Series]],
        validation_data: Optional[Dict[str, Tuple[pd.DataFrame, pd.Series]]],
    ) -> None:
        """Train single model with instance weighting."""
        # Combine all data with weights
        all_X = []
        all_y = []
        all_weights = []
        
        for domain, (X, y) in data_by_domain.items():
            all_X.append(X)
            all_y.append(y)
            
            if domain in self.domain_weights_:
                all_weights.append(self.domain_weights_[domain])
            else:
                all_weights.append(np.ones(len(X)))
        
        X_combined = pd.concat(all_X, ignore_index=True)
        y_combined = pd.concat(all_y, ignore_index=True)
        weights_combined = np.concatenate(all_weights)
        
        # Train single model with weights
        self.shared_model_ = XGBoostAdvancedModel(
            config=self.base_config,
            use_custom_objective=True,
            objective_type="csmf_weighted",
        )
        
        self.shared_model_.fit(
            X_combined,
            y_combined,
            sample_weight=weights_combined,
        )
    
    def _predict_multi_task(
        self,
        X: pd.DataFrame,
        source_domain: Optional[str],
        target_domain: Optional[str],
    ) -> np.ndarray:
        """Make predictions using multi-task model."""
        # Get shared predictions
        shared_probs = self.shared_model_.predict_proba(X)
        
        # If we have a domain-specific model, combine predictions
        if source_domain and source_domain in self.domain_models_:
            domain_probs = self.domain_models_[source_domain].predict_proba(X)
            
            # Weighted combination based on shared ratio
            combined_probs = (
                self.shared_layers_ratio * shared_probs +
                (1 - self.shared_layers_ratio) * domain_probs
            )
            
            # Get class predictions
            pred_indices = np.argmax(combined_probs, axis=1)
            return self.shared_model_.label_encoder_.inverse_transform(pred_indices)
        else:
            # Use only shared model
            return self.shared_model_.predict(X)
    
    def cross_domain_evaluate(
        self,
        test_data_by_domain: Dict[str, Tuple[pd.DataFrame, pd.Series]]
    ) -> pd.DataFrame:
        """Evaluate model performance across all domain pairs.
        
        Args:
            test_data_by_domain: Test data for each domain
            
        Returns:
            DataFrame with cross-domain evaluation results
        """
        results = []
        
        for source_domain in self.domains_:
            for target_domain in test_data_by_domain.keys():
                X_test, y_test = test_data_by_domain[target_domain]
                
                # Make predictions
                y_pred = self.predict(
                    X_test,
                    source_domain=source_domain,
                    target_domain=target_domain,
                )
                
                # Calculate metrics
                csmf_acc = self.shared_model_.calculate_csmf_accuracy(y_test, y_pred)
                cod_acc = (y_test == y_pred).mean()
                
                results.append({
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "csmf_accuracy": csmf_acc,
                    "cod_accuracy": cod_acc,
                    "is_in_domain": source_domain == target_domain,
                })
        
        return pd.DataFrame(results)