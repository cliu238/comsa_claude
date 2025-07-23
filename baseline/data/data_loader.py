"""Simple data loader for integration testing."""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


class VADataProcessor:
    """Simplified VADataProcessor for integration testing."""
    
    def __init__(self):
        """Initialize data processor."""
        self.label_encoders = {}
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        return pd.read_csv(filepath)
    
    def to_numeric(self, df: pd.DataFrame, target_column: str = 'cause', 
                   encoding_type: str = 'standard'):
        """Convert categorical features to numeric."""
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode categorical features
        for col in X.columns:
            if X[col].dtype == 'object':
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    X[col] = self.label_encoders[col].fit_transform(X[col])
                else:
                    X[col] = self.label_encoders[col].transform(X[col])
        
        return X, y