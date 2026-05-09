"""
Coordinator Server — orchestrates federated learning rounds.

Responsibilities:
  1. Hold the global model
  2. Accept client registrations
  3. Serve the current global model to clients on request
  4. Receive trained weights from clients
  5. Once enough clients have submitted, run FedAvg and start a new round
  6. Track client liveness via heartbeats

REST API:
  POST /register        -> client announces itself, gets a client_id
  GET  /get_model       -> returns current global weights + round number
  POST /submit_update   -> client submits its trained weights for the round
  POST /heartbeat       -> client signals it's still alive
  GET  /status          -> debug endpoint, returns server state
"""
import logging
import threading
import time
from collections import defaultdict
from datetime import datetime

from flask import Flask, jsonify, request

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.cnn import MNIST_CNN, get_model_weights, set_model_weights
from server.aggregator import federated_average


# ---------- Configuration ----------
HOST = "0.0.0.0"
PORT = 5050
MIN_CLIENTS_PER_ROUND = 2     # how many clients must submit before we aggregate
MAX_ROUNDS = 10               # total federated rounds before we stop
HEARTBEAT_TIMEOUT = 30        # seconds before a client is considered dead

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SERVER] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
# silence Flask's noisy default request logging
logging.getLogger("werkzeug").setLevel(logging.WARNING)


# ---------- Server state (in-memory, protected by a lock) ----------
class ServerState:
    """All mutable state lives here, behind a single lock for thread-safety."""

    def __init__(self):
        self.lock = threading.Lock()
        self.global_model = MNIST_CNN()
        self.current_round = 0
        self.clients = {}                  # client_id -> {"last_seen": timestamp, "address": str}
        self.round_updates = defaultdict(list)  # round -> list of {weights, num_samples, client_id}
        self.next_client_id = 0
        self.training_complete = False
        self.round_history = []            # for the report: track timing, num clients, etc.

    def register_client(self, address):
        cid = f"client_{self.next_client_id}"
        self.next_client_id += 1
        self.clients[cid] = {"last_seen": time.time(), "address": address}
        return cid

    def update_heartbeat(self, client_id):
        if client_id in self.clients:
            self.clients[client_id]["last_seen"] = time.time()

    def alive_clients(self):
        now = time.time()
        return [cid for cid, info in self.clients.items()
                if now - info["last_seen"] < HEARTBEAT_TIMEOUT]


state = ServerState()
app = Flask(__name__)


# ---------- Endpoints ----------

@app.route("/register", methods=["POST"])
def register():
    """A client announces itself. Returns its assigned client_id."""
    address = request.remote_addr
    with state.lock:
        cid = state.register_client(address)
    log.info(f"Registered {cid} from {address}")
    return jsonify({"client_id": cid, "current_round": state.current_round})


@app.route("/get_model", methods=["GET"])
def get_model():
    """Returns the current global model weights + round number."""
    with state.lock:
        weights = get_model_weights(state.global_model)
        return jsonify({
            "round": state.current_round,
            "weights": weights,
            "training_complete": state.training_complete,
        })


@app.route("/submit_update", methods=["POST"])
def submit_update():
    """
    A client submits its locally-trained weights.
    When enough updates arrive for the current round, we aggregate.
    """
    payload = request.get_json()
    client_id = payload["client_id"]
    submitted_round = payload["round"]
    weights = payload["weights"]
    num_samples = payload["num_samples"]

    with state.lock:
        # Reject stale updates from a previous round (a slow client we already moved past)
        if submitted_round != state.current_round:
            log.warning(
                f"{client_id} submitted for round {submitted_round} "
                f"but we're on round {state.current_round} — dropping."
            )
            return jsonify({"status": "stale", "current_round": state.current_round}), 409

        state.update_heartbeat(client_id)
        state.round_updates[submitted_round].append({
            "client_id": client_id,
            "weights": weights,
            "num_samples": num_samples,
        })
        num_received = len(state.round_updates[submitted_round])
        log.info(
            f"Round {submitted_round}: received update from {client_id} "
            f"({num_samples} samples). Total this round: {num_received}/{MIN_CLIENTS_PER_ROUND}"
        )

        # If we have enough updates, aggregate and advance the round
        if num_received >= MIN_CLIENTS_PER_ROUND:
            _aggregate_and_advance()

    return jsonify({"status": "accepted", "round": submitted_round})


@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    """Client signals it's still alive."""
    payload = request.get_json()
    client_id = payload.get("client_id")
    with state.lock:
        state.update_heartbeat(client_id)
    return jsonify({"status": "ok", "current_round": state.current_round})


@app.route("/status", methods=["GET"])
def status():
    """Debug endpoint — see what the server is doing."""
    with state.lock:
        return jsonify({
            "current_round": state.current_round,
            "training_complete": state.training_complete,
            "registered_clients": list(state.clients.keys()),
            "alive_clients": state.alive_clients(),
            "updates_this_round": len(state.round_updates[state.current_round]),
            "round_history": state.round_history,
        })


# ---------- Aggregation ----------

def _aggregate_and_advance():
    """Called while holding state.lock. Runs FedAvg and starts the next round."""
    round_num = state.current_round
    updates = state.round_updates[round_num]

    log.info(f"--- Aggregating round {round_num} with {len(updates)} client updates ---")
    start = time.time()

    new_weights = federated_average(updates)
    set_model_weights(state.global_model, new_weights)

    elapsed = time.time() - start
    state.round_history.append({
        "round": round_num,
        "num_clients": len(updates),
        "client_ids": [u["client_id"] for u in updates],
        "aggregation_time_sec": round(elapsed, 3),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })
    log.info(f"Round {round_num} complete in {elapsed:.2f}s. Advancing to round {round_num + 1}.")

    state.current_round += 1
    if state.current_round >= MAX_ROUNDS:
        state.training_complete = True
        log.info(f"🎉 Training complete after {MAX_ROUNDS} rounds!")


# ---------- Background reaper for dead clients ----------

def _reap_dead_clients():
    """Logs clients that haven't sent heartbeats — useful for failure detection."""
    while True:
        time.sleep(10)
        with state.lock:
            now = time.time()
            dead = [cid for cid, info in state.clients.items()
                    if now - info["last_seen"] > HEARTBEAT_TIMEOUT]
            if dead:
                log.warning(f"💀 Clients with no recent heartbeat: {dead}")


# ---------- Entry point ----------

if __name__ == "__main__":
    log.info(f"Starting coordinator on {HOST}:{PORT}")
    log.info(f"Config: MIN_CLIENTS_PER_ROUND={MIN_CLIENTS_PER_ROUND}, MAX_ROUNDS={MAX_ROUNDS}")

    reaper = threading.Thread(target=_reap_dead_clients, daemon=True)
    reaper.start()

    # threaded=True so multiple clients can hit the server in parallel
    app.run(host=HOST, port=PORT, threaded=True, debug=False)