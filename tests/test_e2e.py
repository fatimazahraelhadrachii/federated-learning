"""
End-to-end test: spin up server in a thread, run 3 clients sequentially
(using fake data so we don't need internet for MNIST download).
"""
import sys
import os
import threading
import time
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ----- Patch data.partition.get_client_data to use fake data -----
import torch
from torch.utils.data import TensorDataset

def fake_get_client_data(client_idx, num_clients, mode="iid"):
    torch.manual_seed(client_idx)
    # 200 fake MNIST-like samples per client
    x = torch.randn(200, 1, 28, 28)
    y = torch.randint(0, 10, (200,))
    test_x = torch.randn(100, 1, 28, 28)
    test_y = torch.randint(0, 10, (100,))
    return TensorDataset(x, y), TensorDataset(test_x, test_y)

import data.partition
data.partition.get_client_data = fake_get_client_data

# Also patch in client.client (since it imports the function directly)
import client.client
client.client.get_client_data = fake_get_client_data

# ----- Reduce rounds for a fast test -----
import server.coordinator as coord
coord.MAX_ROUNDS = 3
coord.MIN_CLIENTS_PER_ROUND = 3

# ----- Start the server in a background thread -----
def run_server():
    coord.app.run(host="127.0.0.1", port=5000, threaded=True, debug=False, use_reloader=False)

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(2)  # let Flask boot

# ----- Health check -----
r = requests.get("http://127.0.0.1:5000/status")
print("Initial status:", r.json())

# ----- Launch 3 clients in parallel threads -----
from client.client import FederatedClient

def run_client(idx):
    c = FederatedClient("http://127.0.0.1:5000", idx, num_clients=3, mode="iid")
    c.run()

client_threads = []
for i in range(3):
    t = threading.Thread(target=run_client, args=(i,), daemon=True)
    t.start()
    client_threads.append(t)
    time.sleep(0.5)  # stagger registration

# Wait for all clients to finish
for t in client_threads:
    t.join(timeout=120)

# ----- Final status -----
r = requests.get("http://127.0.0.1:5000/status")
final = r.json()
print("\n========== FINAL STATUS ==========")
print(f"Current round: {final['current_round']}")
print(f"Training complete: {final['training_complete']}")
print(f"Registered clients: {final['registered_clients']}")
print(f"\nRound history:")
for h in final['round_history']:
    print(f"  Round {h['round']}: {h['num_clients']} clients, "
          f"agg time: {h['aggregation_time_sec']}s")

assert final['training_complete'], "Training didn't complete!"
assert len(final['round_history']) == 3, f"Expected 3 rounds, got {len(final['round_history'])}"
print("\n🎉 END-TO-END TEST PASSED!")