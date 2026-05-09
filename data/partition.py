"""
Splits MNIST across N clients.

IID:     each client gets a random uniform sample (easy case)
non-IID: each client gets only 2-3 digit classes (realistic, harder for FL)

The non-IID case is what makes federated learning interesting — and it's what
your report should highlight as the key distributed challenge.
"""
import numpy as np
import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms


def load_mnist(data_dir="./mnist_data"):
    """Download MNIST once, reused by all clients."""
    import ssl
    # Bypass SSL verification on Mac (common issue with Python's bundled certs)
    ssl._create_default_https_context = ssl._create_unverified_context

    # Use a reliable MNIST mirror (the original yann.lecun.com is often down)
    new_mirror = "https://ossci-datasets.s3.amazonaws.com/mnist"
    datasets.MNIST.mirrors = [new_mirror + "/"]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # standard MNIST normalization
    ])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    return train, test


def partition_iid(dataset, num_clients, seed=42):
    """Split dataset into num_clients equal random shards."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    shards = np.array_split(indices, num_clients)
    return [Subset(dataset, shard.tolist()) for shard in shards]


def partition_non_iid(dataset, num_clients, classes_per_client=2, seed=42):
    """
    Each client gets data from only `classes_per_client` digit classes.
    This is realistic (e.g., a hospital only sees certain patient types)
    and exposes the weaknesses of naive FedAvg.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(dataset.targets)
    num_classes = 10

    # group indices by class
    class_indices = {c: np.where(labels == c)[0] for c in range(num_classes)}
    for c in class_indices:
        rng.shuffle(class_indices[c])

    client_indices = [[] for _ in range(num_clients)]
    # assign each client a few classes
    for client_id in range(num_clients):
        chosen_classes = rng.choice(num_classes, classes_per_client, replace=False)
        for c in chosen_classes:
            # take a chunk of that class's indices
            chunk_size = len(class_indices[c]) // num_clients
            start = client_id * chunk_size
            client_indices[client_id].extend(
                class_indices[c][start:start + chunk_size].tolist()
            )

    return [Subset(dataset, idx) for idx in client_indices]


def get_client_data(client_id, num_clients, mode="iid"):
    """Convenience wrapper — returns this client's local train+test data."""
    train, test = load_mnist()
    if mode == "iid":
        train_shards = partition_iid(train, num_clients)
    elif mode == "non_iid":
        train_shards = partition_non_iid(train, num_clients)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return train_shards[client_id], test  # test set shared for evaluation


if __name__ == "__main__":
    # quick sanity check
    train, _ = load_mnist()
    shards = partition_non_iid(train, num_clients=5)
    for i, shard in enumerate(shards):
        labels = [train.targets[idx].item() for idx in shard.indices[:200]]
        print(f"Client {i}: {len(shard)} samples, sample labels: {set(labels)}")