"""
Local training logic for a federated client.

Kept separate from networking (client.py) so we can unit-test it in isolation.
Each call to train_one_epoch() does one local epoch — that's the "local update"
that gets sent back to the coordinator in FedAvg.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def train_one_epoch(model, dataset, batch_size=32, lr=0.01, device="cpu"):
    """
    Train the model for one full pass over the local dataset.

    Args:
        model: a MNIST_CNN instance with current global weights loaded
        dataset: a torch Subset (this client's local data shard)
        batch_size: SGD batch size
        lr: learning rate
        device: "cpu" or "cuda"

    Returns:
        (trained_model, num_samples, avg_loss)
    """
    model.to(device)
    model.train()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    num_batches = 0

    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return model, len(dataset), avg_loss


def evaluate(model, test_dataset, batch_size=128, device="cpu"):
    """
    Evaluate model accuracy on a test set.
    Used by clients to log how well the global model is performing.
    """
    model.to(device)
    model.eval()

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

    return correct / total if total > 0 else 0.0