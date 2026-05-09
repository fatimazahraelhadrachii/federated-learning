"""
FedAvg (Federated Averaging) — McMahan et al., 2017.

The core formula:

    w_global = sum over k of (n_k / n_total) * w_k

where:
    w_k     = weights from client k
    n_k     = number of training samples client k used
    n_total = total samples across all participating clients

Clients with more data have more influence on the global model.
This is the simplest aggregation strategy — there are many extensions
(FedProx, FedNova, etc.) you can mention in the report.
"""
import numpy as np


def federated_average(client_updates):
    """
    Aggregate weights from multiple clients using weighted average.

    Args:
        client_updates: list of dicts, each with keys:
            - "weights": dict {layer_name: list (nested)} from get_model_weights()
            - "num_samples": int, how many samples this client trained on

    Returns:
        dict in the same format as "weights" — the new global weights.
    """
    if not client_updates:
        raise ValueError("Cannot aggregate: no client updates received.")

    total_samples = sum(c["num_samples"] for c in client_updates)
    if total_samples == 0:
        raise ValueError("Total samples is zero — clients sent empty updates.")

    # Initialize with zeros, same shape as first client's weights
    layer_names = client_updates[0]["weights"].keys()
    aggregated = {}

    for layer_name in layer_names:
        # Stack same-layer weights across clients, weighted by sample count
        weighted_sum = None
        for client in client_updates:
            weight_array = np.array(client["weights"][layer_name], dtype=np.float32)
            contribution = (client["num_samples"] / total_samples) * weight_array
            if weighted_sum is None:
                weighted_sum = contribution
            else:
                weighted_sum = weighted_sum + contribution
        aggregated[layer_name] = weighted_sum.tolist()

    return aggregated


def krum_aggregate(client_updates, num_byzantine=1):
    """
    BONUS: Krum aggregation — Byzantine-fault-tolerant alternative to FedAvg.

    Picks the client whose weights are closest to its neighbors. Resists
    malicious clients sending poisoned updates. Mention this in the report
    even if you only implement vanilla FedAvg — shows you understand the
    Byzantine fault tolerance problem.
    """
    n = len(client_updates)
    if n <= 2 * num_byzantine + 2:
        # Not enough clients for Krum to be meaningful — fall back
        return federated_average(client_updates)

    # Flatten each client's weights into a single vector
    def flatten(weights):
        return np.concatenate([np.array(v).flatten() for v in weights.values()])

    vectors = [flatten(c["weights"]) for c in client_updates]
    scores = []
    # for each client, sum distances to its (n - num_byzantine - 2) nearest neighbors
    for i in range(n):
        dists = sorted(np.linalg.norm(vectors[i] - vectors[j])**2
                       for j in range(n) if j != i)
        scores.append(sum(dists[:n - num_byzantine - 2]))

    selected_idx = int(np.argmin(scores))
    return client_updates[selected_idx]["weights"]