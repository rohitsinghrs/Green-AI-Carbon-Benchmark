import torch
import torch.nn as nn


# ============================================================
# MODEL 1: LINEAR BASELINE
# ============================================================

class LinearModel(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


# ============================================================
# MODEL 2: SMALL MLP
# ============================================================

class SmallMLP(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


# ============================================================
# MODEL 3: DEEP MLP
# ============================================================

class DeepMLP(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


# ============================================================
# MODEL 4: 1D CNN
# ============================================================

class CNN1D(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1)
        )

        self.classifier = nn.Linear(64, 1)

    def forward(self, x):

        # Shape:
        # [batch, features]
        # →
        # [batch, 1, features]

        x = x.unsqueeze(1)

        x = self.features(x)

        x = x.squeeze(-1)

        x = self.classifier(x)

        return x.squeeze(1)


# ============================================================
# MODEL FACTORY
# ============================================================

def create_model(model_name, input_dim):

    if model_name == "Linear":

        return LinearModel(input_dim)

    elif model_name == "SmallMLP":

        return SmallMLP(input_dim)

    elif model_name == "DeepMLP":

        return DeepMLP(input_dim)

    elif model_name == "CNN1D":

        return CNN1D(input_dim)

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )