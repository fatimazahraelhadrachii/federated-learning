"""
Federated Learning Client (v3) — with leader discovery for Bully algorithm.

NEW: instead of one server URL, accepts a list of coordinator URLs.
The client asks any of them "who's the leader?", then talks to the leader.
If a request gets HTTP 421 (not_leader), the client re-discovers and retries.
"""
import argparse
import logging
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.cnn import MNIST_CNN, get_model_weights, set_model_weights
from model.lamport_clock import LamportClock
from data.partition import get_client_data
from client.trainer import train_one_epoch, evaluate
from client.failure_injector import FailureInjector


HEARTBEAT_INTERVAL = 5
POLL_INTERVAL = 2
MAX_WAIT_PER_ROUND = 120
LEADER_DISCOVERY_RETRIES = 5


def setup_logging(client_idx):
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [CLIENT-{client_idx}] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


class FederatedClient:
    def __init__(self, coordinator_urls, client_idx, num_clients, mode="iid",
                 failure_mode="none", failure_rate=0.0):
        # NEW: list of all coordinator URLs (not just one server)
        self.coordinator_urls = [u.rstrip("/") for u in coordinator_urls]
        self.current_leader_url = None  # cached leader URL

        self.client_idx = client_idx
        self.num_clients = num_clients
        self.client_id = None
        self.log = setup_logging(client_idx)

        self.log.info(f"Loading local data ({mode} split, shard {client_idx}/{num_clients})...")
        self.train_data, self.test_data = get_client_data(client_idx, num_clients, mode=mode)
        self.log.info(f"Loaded {len(self.train_data)} local training samples.")

        self.model = MNIST_CNN()
        self.clock = LamportClock(f"client_{client_idx}")
        self.failure = FailureInjector(
            mode=failure_mode, rate=failure_rate, seed=42 + client_idx,
        )

    # ---------- Leader discovery ----------

    def discover_leader(self):
        """
        Ask each coordinator who the leader is. Cache the answer.
        Returns True if a leader was found, False otherwise.
        """
        for url in self.coordinator_urls:
            try:
                r = requests.get(f"{url}/bully/whoisleader", timeout=3)
                if r.status_code == 200:
                    data = r.json()
                    leader_id = data.get("leader_id")
                    leader_url = data.get("leader_url")
                    if leader_id is not None and leader_url is not None:
                        if leader_url != self.current_leader_url:
                            self.log.info(
                                f"[L={self.clock.value}] 👑 Leader is coordinator "
                                f"{leader_id} at {leader_url}"
                            )
                        self.current_leader_url = leader_url
                        return True
            except requests.RequestException:
                continue
        self.log.warning(f"[L={self.clock.value}] No coordinator responded — election may be in progress.")
        return False

    def _ensure_leader(self):
        """Discover leader with retries (e.g. during an election)."""
        for attempt in range(LEADER_DISCOVERY_RETRIES):
            if self.discover_leader():
                return True
            self.log.info(f"Retrying leader discovery in 3s... (attempt {attempt + 1})")
            time.sleep(3)
        return False

    # ---------- HTTP wrappers ----------

    def _post(self, path, payload, retries=3):
        if self.failure.maybe_partition():
            raise ConnectionError("Network partition (simulated)")
        self.failure.maybe_drop_message()

        if not self.current_leader_url:
            if not self._ensure_leader():
                raise ConnectionError("No leader available")

        payload = dict(payload)
        payload["clock"] = self.clock.send_event()

        for attempt in range(retries):
            try:
                r = requests.post(f"{self.current_leader_url}{path}", json=payload, timeout=30)
                if r.status_code == 421:
                    # We hit a non-leader; refresh and retry
                    data = r.json()
                    new_leader_url = data.get("leader_url")
                    if new_leader_url and new_leader_url != self.current_leader_url:
                        self.log.info(f"Redirected to new leader: {new_leader_url}")
                        self.current_leader_url = new_leader_url
                        continue
                    else:
                        # Re-discover from scratch
                        self._ensure_leader()
                        continue
                try:
                    server_clock = r.json().get("clock", 0)
                    self.clock.receive_event(server_clock)
                except Exception:
                    pass
                return r
            except requests.RequestException as e:
                self.log.warning(f"POST {path} failed (attempt {attempt + 1}/{retries}): {e}")
                # Maybe the leader died — try to find a new one
                self._ensure_leader()
                time.sleep(2)
        raise ConnectionError(f"Could not reach leader after {retries} attempts")

    def _get(self, path, retries=3):
        if self.failure.maybe_partition():
            raise ConnectionError("Network partition (simulated)")

        if not self.current_leader_url:
            if not self._ensure_leader():
                raise ConnectionError("No leader available")

        sep = "&" if "?" in path else "?"
        clock_ts = self.clock.send_event()
        full_path = f"{path}{sep}clock={clock_ts}"

        for attempt in range(retries):
            try:
                r = requests.get(f"{self.current_leader_url}{full_path}", timeout=30)
                if r.status_code == 421:
                    data = r.json()
                    new_leader_url = data.get("leader_url")
                    if new_leader_url and new_leader_url != self.current_leader_url:
                        self.log.info(f"Redirected to new leader: {new_leader_url}")
                        self.current_leader_url = new_leader_url
                        continue
                    else:
                        self._ensure_leader()
                        continue
                try:
                    server_clock = r.json().get("clock", 0)
                    self.clock.receive_event(server_clock)
                except Exception:
                    pass
                return r
            except requests.RequestException as e:
                self.log.warning(f"GET {path} failed (attempt {attempt + 1}/{retries}): {e}")
                self._ensure_leader()
                time.sleep(2)
        raise ConnectionError(f"Could not reach leader after {retries} attempts")

    # ---------- Federated protocol ----------

    def register(self):
        if not self._ensure_leader():
            raise ConnectionError("No leader available for registration")
        r = self._post("/register", {})
        data = r.json()
        self.client_id = data["client_id"]
        self.log.info(
            f"[L={self.clock.value}] Registered as {self.client_id} "
            f"(round {data['current_round']}, leader_L={data['clock']})"
        )

    def fetch_global_model(self):
        r = self._get("/get_model")
        data = r.json()
        set_model_weights(self.model, data["weights"])
        return data["round"], data["training_complete"]

    def submit_update(self, round_num, num_samples):
        weights = get_model_weights(self.model)
        weights = self.failure.maybe_corrupt_weights(weights)
        payload = {
            "client_id": self.client_id,
            "round": round_num,
            "weights": weights,
            "num_samples": num_samples,
        }
        r = self._post("/submit_update", payload)
        if r.status_code == 409:
            self.log.warning(f"[L={self.clock.value}] Update for round {round_num} rejected (stale).")
            return False
        return True

    def heartbeat(self):
        try:
            self._post("/heartbeat", {"client_id": self.client_id}, retries=1)
        except ConnectionError:
            self.log.warning(f"[L={self.clock.value}] Heartbeat failed.")

    def wait_for_next_round(self, current_round):
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
                self.log.warning(f"[L={self.clock.value}] Couldn't reach leader, will retry...")
            if waited % HEARTBEAT_INTERVAL == 0:
                self.heartbeat()
        return current_round, True

    # ---------- Main loop ----------

    def run(self):
        try:
            self.register()
        except ConnectionError as e:
            self.log.error(f"Failed to register: {e}")
            return

        while True:
            self.failure.maybe_crash()

            try:
                round_num, done = self.fetch_global_model()
            except ConnectionError:
                self.log.warning("Can't fetch model — retrying...")
                time.sleep(5)
                continue

            if done:
                self.log.info(f"[L={self.clock.value}] Training complete.")
                break

            self.log.info(f"[L={self.clock.value}] === Round {round_num} ===")
            acc_before = evaluate(self.model, self.test_data)
            self.log.info(f"[L={self.clock.value}] Global model accuracy: {acc_before:.4f}")

            self.failure.maybe_straggle()

            self.log.info(f"[L={self.clock.value}] Training locally for 1 epoch...")
            t0 = time.time()
            self.model, num_samples, avg_loss = train_one_epoch(self.model, self.train_data)
            t_train = time.time() - t0
            self.clock.tick()
            self.log.info(f"[L={self.clock.value}] Trained in {t_train:.1f}s. Loss={avg_loss:.4f}")

            try:
                ok = self.submit_update(round_num, num_samples)
            except ConnectionError as e:
                self.log.warning(f"Couldn't submit: {e}")
                time.sleep(3)
                continue

            if not ok:
                continue

            self.log.info(f"[L={self.clock.value}] Waiting for other clients...")
            new_round, done = self.wait_for_next_round(round_num)
            if done:
                break

        self.log.info(f"[L={self.clock.value}] Client shutting down.")


def main():
    parser = argparse.ArgumentParser()
    # NEW: --coordinator-urls (multiple) instead of single --server-url
    parser.add_argument("--coordinator-urls", required=True,
                        help="Comma-separated list of coordinator URLs, "
                             "e.g. http://coord-0:5050,http://coord-1:5051,http://coord-2:5052")
    parser.add_argument("--client-idx", type=int, required=True)
    parser.add_argument("--num-clients", type=int, required=True)
    parser.add_argument("--mode", default="iid", choices=["iid", "non_iid"])
    parser.add_argument("--failure-mode", default="none",
                        choices=["none", "crash", "straggler", "byzantine",
                                 "partition", "message_loss"])
    parser.add_argument("--failure-rate", type=float, default=0.0)
    args = parser.parse_args()

    coord_urls = [u.strip() for u in args.coordinator_urls.split(",") if u.strip()]
    client = FederatedClient(
        coordinator_urls=coord_urls,
        client_idx=args.client_idx,
        num_clients=args.num_clients,
        mode=args.mode,
        failure_mode=args.failure_mode,
        failure_rate=args.failure_rate,
    )
    client.run()


if __name__ == "__main__":
    main()