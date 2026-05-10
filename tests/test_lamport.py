"""
E2E test: verify Lamport clocks work across server + clients.

Expected: at the end, the server's clock should reflect events from all clients,
and the round_history should contain client_clocks + server_clock_at_aggregation.
"""
import sys
import os
import threading
import time
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from torch.utils.data import TensorDataset

def fake_get_client_data(client_idx, num_clients, mode="iid"):
    torch.manual_seed(client_idx)
    return (TensorDataset(torch.randn(200, 1, 28, 28), torch.randint(0, 10, (200,))),
            TensorDataset(torch.randn(50, 1, 28, 28), torch.randint(0, 10, (50,))))

import data.partition
data.partition.get_client_data = fake_get_client_data
import client.client
client.client.get_client_data = fake_get_client_data

import server.coordinator as coord
coord.MAX_ROUNDS = 3
coord.MIN_CLIENTS_PER_ROUND = 2

def run_server():
    coord.app.run(host="127.0.0.1", port=5052, threaded=True, debug=False, use_reloader=False)

threading.Thread(target=run_server, daemon=True).start()
time.sleep(2)

from client.client import FederatedClient

def run_client(idx):
    c = FederatedClient("http://127.0.0.1:5052", idx, num_clients=2, mode="iid")
    c.run()
    print(f"\n>>> Client {idx} final Lamport clock: {c.clock.value}")

threads = []
for i in range(2):
    t = threading.Thread(target=run_client, args=(i,), daemon=True)
    t.start()
    threads.append(t)
    time.sleep(0.3)

for t in threads:
    t.join(timeout=60)

r = requests.get("http://127.0.0.1:5052/status")
final = r.json()
print("\n========== FINAL STATUS ==========")
print(f"Server Lamport clock: {final['lamport_clock']}")
print(f"Training complete: {final['training_complete']}")
print(f"\nRound history (with Lamport timestamps):")
for h in final['round_history']:
    print(f"  Round {h['round']}: server_L={h['server_clock_at_aggregation']}, client_clocks={h['client_clocks']}")

assert final['training_complete']
assert final['lamport_clock'] > 0, "Server clock should have advanced!"
for h in final['round_history']:
    assert 'server_clock_at_aggregation' in h
    assert 'client_clocks' in h
    assert len(h['client_clocks']) == 2

print("\n🎉 LAMPORT CLOCKS WORKING END-TO-END!")