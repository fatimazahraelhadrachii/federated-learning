"""
E2E test: 3 clients, one of them crashes after a couple rounds.
The system should recover — server keeps progressing once 2 of 3 still submit.
"""
import sys
import os
import threading
import time
import requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Use fake data (no network needed)
import torch
from torch.utils.data import TensorDataset

def fake_get_client_data(client_idx, num_clients, mode="iid"):
    torch.manual_seed(client_idx)
    x = torch.randn(200, 1, 28, 28)
    y = torch.randint(0, 10, (200,))
    return TensorDataset(x, y), TensorDataset(torch.randn(50, 1, 28, 28), torch.randint(0, 10, (50,)))

import data.partition
data.partition.get_client_data = fake_get_client_data
import client.client
client.client.get_client_data = fake_get_client_data

import server.coordinator as coord
coord.MAX_ROUNDS = 4
coord.MIN_CLIENTS_PER_ROUND = 2  # only need 2 of 3 to advance

def run_server():
    coord.app.run(host="127.0.0.1", port=5051, threaded=True, debug=False, use_reloader=False)

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
time.sleep(2)

from client.client import FederatedClient

def run_normal_client(idx):
    c = FederatedClient("http://127.0.0.1:5051", idx, num_clients=3, mode="iid")
    c.run()

def run_crashy_client(idx):
    # 100% crash rate -> dies on first round attempt
    c = FederatedClient("http://127.0.0.1:5051", idx, num_clients=3, mode="iid",
                        failure_mode="crash", failure_rate=1.0)
    try:
        c.run()
    except SystemExit:
        print(f"[TEST] Client {idx} crashed as expected")

threads = [
    threading.Thread(target=run_normal_client, args=(0,), daemon=True),
    threading.Thread(target=run_normal_client, args=(1,), daemon=True),
    threading.Thread(target=run_crashy_client, args=(2,), daemon=True),
]

for t in threads:
    t.start()
    time.sleep(0.5)

for t in threads:
    t.join(timeout=120)

r = requests.get("http://127.0.0.1:5051/status")
final = r.json()
print("\n========== FINAL STATUS ==========")
print(f"Current round: {final['current_round']}")
print(f"Training complete: {final['training_complete']}")
print(f"Round history:")
for h in final['round_history']:
    print(f"  Round {h['round']}: clients={h['client_ids']}")

assert final['training_complete'], "Training should still complete despite the crash!"
print("\n🎉 SYSTEM RECOVERED FROM CRASHED CLIENT!")