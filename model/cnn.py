"""
Shared CNN model for MNIST classification.
Both clients and server use this exact architecture so weights are compatible.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    """A small CNN for MNIST. Light enough to train fast on CPU."""

    def __init__(self):
        super().__init__()
        # Input: 1x28x28 (grayscale MNIST images)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)   # -> 16x28x28
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)  # -> 32x28x28
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 digit classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)          # -> 16x14x14
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)          # -> 32x7x7
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return self.fc2(x)              # raw logits (CrossEntropyLoss handles softmax)


def get_model_weights(model):
    """Extract weights as a dict of {layer_name: list} — JSON-serializable for REST."""
    return {name: param.detach().cpu().numpy().tolist()
            for name, param in model.state_dict().items()}


def set_model_weights(model, weights_dict):
    """Load weights back into a model from the dict format above."""
    state_dict = {name: torch.tensor(values)
                  for name, values in weights_dict.items()}
    model.load_state_dict(state_dict)