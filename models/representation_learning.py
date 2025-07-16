from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


class BaseRepresentationLearner(ABC):
    """Base class for representation learning models."""

    @abstractmethod
    def fit(self, data: np.ndarray) -> None:
        """Fit the representation learning model to the data."""
        pass

    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform the data using the learned representation."""
        pass

    @abstractmethod
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Convert the transformed data back to original space."""
        pass


class AutoEncoder(nn.Module, BaseRepresentationLearner):
    """AutoEncoder for converting between WHO2016 and PHMRC questionnaire formats."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [256, 128, 64],
        latent_dim: int = 32,
        learning_rate: float = 0.001,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.learning_rate = learning_rate

        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)]
            )
            prev_dim = hidden_dim

        # Latent layer
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        # Decoder layers
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)]
            )
            prev_dim = hidden_dim

        # Output layer
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.to(device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the autoencoder."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def fit(
        self,
        data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.1,
    ) -> Dict[str, list]:
        """Train the autoencoder."""
        self.train()
        data = torch.FloatTensor(data).to(self.device)

        # Split into train and validation
        val_size = int(len(data) * validation_split)
        train_data = data[:-val_size]
        val_data = data[-val_size:]

        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False
        )

        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Training
            train_loss = 0
            for batch in train_loader:
                self.optimizer.zero_grad()
                _, decoded = self(batch)
                loss = self.criterion(decoded, batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            # Validation
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    _, decoded = self(batch)
                    val_loss += self.criterion(decoded, batch).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f} - "
                    f"Val Loss: {val_loss:.4f}"
                )

        return history

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data to latent space."""
        self.eval()
        with torch.no_grad():
            data = torch.FloatTensor(data).to(self.device)
            encoded, _ = self(data)
            return encoded.cpu().numpy()

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data from latent space back to original space."""
        self.eval()
        with torch.no_grad():
            data = torch.FloatTensor(data).to(self.device)
            decoded = self.decoder(data)
            return decoded.cpu().numpy()


class PCARepresentationLearner(BaseRepresentationLearner):
    """PCA-based representation learning."""

    def __init__(self, n_components: int = 32):
        self.pca = PCA(n_components=n_components)

    def fit(self, data: np.ndarray) -> None:
        """Fit PCA to the data."""
        self.pca.fit(data)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using PCA."""
        return self.pca.transform(data)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data back to original space."""
        return self.pca.inverse_transform(data)
