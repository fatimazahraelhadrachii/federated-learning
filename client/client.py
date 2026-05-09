"""
Federated Learning Client.

Lifecycle:
  1. Register with the coordinator -> get a client_id
  2. Loop:
       a. Fetch the current global model + round number
       b. Train locally for one epoch
       c. Submit updated weights to the coordinator
       d. Wait for the round to advance, then repeat
  3. Stop when the coordinator says training_complete = True

Run:
    python client/client.py --client-idx 0 --num-clients 3
    python client/client.py --client-idx 1 --num-clients 3
    python client/client.py --client-idx 2 --num-clients 3
"""
import argparse
import logging
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.cnn import MNIST_CNN, get_model_weights, set_model_weights
from data.partition import get_client_data
from client.trainer import train_one_epoch, evaluate


# ---------- Configuration ----------
HEARTBEAT_INTERVAL = 5     # seconds between heartbeats during idle wait
POLL_INTERVAL = 2          # seconds between checks for round advancement
MAX_WAIT_PER_ROUND = 120   # if we wait this long for a round to advance, give up


def setup_logging(client_idx):
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [CLIENT-{client_idx}] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


class FederatedClient:
    def __init__(self, server_url, client_idx, num_clients, mode="iid"):
        self.server_url = server_url.rstrip("/")
        self.client_idx = client_idx
        self.num_clients = num_clients
        self.client_id = None             # assigned by server
        self.log = setup_logging(client_idx)

        # Load this client's local data shard (they don't share data!)
        self.log.info(f"Loading local data ({mode} split, shard {client_idx}/{num_clients})...")
        self.train_data, self.test_data = get_client_data(client_idx, num_clients, mode=mode)
        self.log.info(f"Loaded {len(self.train_data)} local training samples.")

        self.model = MNIST_CNN()

    # ---------- HTTP wrappers (with simple retry) ----------

    def _post(self, path, payload, retries=3):
        for attempt in range(retries):
            try:
                r = requests.post(f"{self.server_url}{path}", json=payload, timeout=30)
                return r
            except requests.RequestException as e:
                self.log.warning(f"POST {path} failed (attempt {attempt + 1}/{retries}): {e}")
                time.sleep(2)
        raise ConnectionError(f"Server unreachable after {retries} attempts")

    def _get(self, path, retries=3):
        for attempt in range(retries):
            try:
                r = requests.get(f"{self.server_url}{path}", timeout=30)
                return r
            except requests.RequestException as e:
                self.log.warning(f"GET {path} failed (attempt {attempt + 1}/{retries}): {e}")
                time.sleep(2)
        raise ConnectionError(f"Server unreachable after {retries} attempts")

    # ---------- Federated protocol ----------

    def register(self):
        r = self._post("/register", {})
        data = r.json()
        self.client_id = data["client_id"]
        self.log.info(f"Registered as {self.client_id} (server is on round {data['current_round']})")

    def fetch_global_model(self):
        """Returns (round_number, training_complete)."""
        r = self._get("/get_model")
        data = r.json()
        set_model_weights(self.model, data["weights"])
        return data["round"], data["training_complete"]

    def submit_update(self, round_num, num_samples):
        weights = get_model_weights(self.model)
        payload = {
            "client_id": self.client_id,
            "round": round_num,
            "weights": weights,
            "num_samples": num_samples,
        }
        r = self._post("/submit_update", payload)
        if r.status_code == 409:
            self.log.warning(f"Update for round {round_num} was rejected (stale).")
            return False
        return True

    def heartbeat(self):
        try:
            self._post("/heartbeat", {"client_id": self.client_id}, retries=1)
        except ConnectionError:
            self.log.warning("Heartbeat failed (server may be down).")

    def wait_for_next_round(self, current_round):
        """Poll until the server moves to a round > current_round."""
        waited = 0
        while waited < MAX_WAIT_PER_ROUND:
            time.sleep(POLL_INTERVAL)
            waited += POLL_INTERVAL
            try:
                round_num, done = self.fetch_global_model()
                if done:
                    return round_num, True
                if round_num > current_round:
                    return round_num, False
            except ConnectionError:
                self.log.warning("Couldn't reach server, will retry...")
            if waited % HEARTBEAT_INTERVAL == 0:
                self.heartbeat()
        self.log.error(f"Round {current_round} didn't advance in {MAX_WAIT_PER_ROUND}s — giving up.")
        return current_round, True

    # ---------- Main loop ----------

    def run(self):
        self.register()

        while True:
            round_num, done = self.fetch_global_model()
            if done:
                self.log.info("Server says training is complete.")
                break

            self.log.info(f"=== Round {round_num} ===")

            # Evaluate the global model on local test data BEFORE training
            # (gives us a sense of how good the global model is)
            acc_before = evaluate(self.model, self.test_data)
            self.log.info(f"Global model accuracy on test set: {acc_before:.4f}")

            # Local training
            self.log.info("Training locally for 1 epoch...")
            t0 = time.time()
            self.model, num_samples, avg_loss = train_one_epoch(self.model, self.train_data)
            t_train = time.time() - t0
            self.log.info(f"Trained in {t_train:.1f}s. Loss={avg_loss:.4f}")

            # Submit
            ok = self.submit_update(round_num, num_samples)
            if not ok:
                # Stale -> re-fetch and retry next iteration
                continue

            # Wait until the server aggregates and starts the next round
            self.log.info("Waiting for other clients...")
            new_round, done = self.wait_for_next_round(round_num)
            if done:
                break

        self.log.info("Client shutting down.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:5050")
    parser.add_argument("--client-idx", type=int, required=True,
                        help="This client's index (0 to num-clients-1)")
    parser.add_argument("--num-clients", type=int, required=True,
                        help="Total number of clients in the federation")
    parser.add_argument("--mode", default="iid", choices=["iid", "non_iid"],
                        help="How data is split across clients")
    args = parser.parse_args()

    client = FederatedClient(
        server_url=args.server_url,
        client_idx=args.client_idx,
        num_clients=args.num_clients,
        mode=args.mode,
    )
    client.run()


if __name__ == "__main__":
    main()