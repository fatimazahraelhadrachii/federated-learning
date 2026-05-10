"""
Coordinator Server (v3) — with Bully leader election.

The system runs N coordinator processes. Only the LEADER processes client
requests; followers reject them with the leader's URL. If the leader dies,
the surviving coordinators automatically elect a new one (Bully algorithm).

Configuration via environment variables:
    COORD_ID        — this coordinator's unique ID (e.g. 0, 1, 2)
    COORD_PORT      — port to listen on (e.g. 5050)
    COORD_PEERS     — comma-separated "id=url" pairs for ALL coordinators
                      e.g. "0=http://coordinator-0:5050,1=http://coordinator-1:5051,2=http://coordinator-2:5052"

REST API (existing FL endpoints work only on the leader):
  POST /register
  GET  /get_model
  POST /submit_update
  POST /heartbeat
  GET  /status

NEW endpoints (always served, by leader and followers):
  POST /bully/election       — peer is starting an election
  POST /bully/coordinator    — peer is announcing itself as leader
  GET  /bully/whoisleader    — debug: ask any node who's leader
"""
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime

from flask import Flask, jsonify, request

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.cnn import MNIST_CNN, get_model_weights, set_model_weights
from model.lamport_clock import LamportClock
from server.aggregator import federated_average
from server.bully import BullyElection


# ---------- Configuration ----------
HOST = "0.0.0.0"
PORT = int(os.environ.get("COORD_PORT", "5050"))
COORD_ID = int(os.environ.get("COORD_ID", "0"))

# Parse peer list. Format: "0=http://host:port,1=http://host:port,..."
_peer_str = os.environ.get("COORD_PEERS", f"{COORD_ID}=http://localhost:{PORT}")
PEER_URLS = {}
for entry in _peer_str.split(","):
    entry = entry.strip()
    if not entry:
        continue
    pid, url = entry.split("=", 1)
    PEER_URLS[int(pid)] = url

MY_URL = PEER_URLS.get(COORD_ID, f"http://localhost:{PORT}")

MIN_CLIENTS_PER_ROUND = int(os.environ.get("MIN_CLIENTS_PER_ROUND", "2"))
MAX_ROUNDS = int(os.environ.get("MAX_ROUNDS", "10"))
HEARTBEAT_TIMEOUT = 30

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [COORD-{COORD_ID}] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)


# ---------- Server state ----------
class ServerState:
    def __init__(self):
        self.lock = threading.Lock()
        self.global_model = MNIST_CNN()
        self.current_round = 0
        self.clients = {}
        self.round_updates = defaultdict(list)
        self.next_client_id = 0
        self.training_complete = False
        self.round_history = []
        self.clock = LamportClock(f"coord_{COORD_ID}")

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
bully = BullyElection(my_id=COORD_ID, my_url=MY_URL, peer_urls=PEER_URLS)
app = Flask(__name__)


# ---------- Leader-only guard ----------

def _require_leader():
    """
    Returns None if we ARE the leader, or a JSON response redirecting
    to the actual leader if we are not.
    """
    if bully.is_leader():
        return None
    leader_id = bully.get_leader_id()
    leader_url = PEER_URLS.get(leader_id) if leader_id is not None else None
    return jsonify({
        "status": "not_leader",
        "leader_id": leader_id,
        "leader_url": leader_url,
        "this_coordinator": COORD_ID,
    }), 421  # 421 Misdirected Request


# ---------- Bully endpoints (always served) ----------

@app.route("/bully/election", methods=["POST"])
def bully_election():
    payload = request.get_json()
    return jsonify(bully.handle_election_message(payload["from_id"]))


@app.route("/bully/coordinator", methods=["POST"])
def bully_coordinator():
    payload = request.get_json()
    return jsonify(bully.handle_coordinator_message(
        payload["leader_id"], payload["leader_url"]
    ))


@app.route("/bully/whoisleader", methods=["GET"])
def bully_whoisleader():
    return jsonify({
        "leader_id": bully.get_leader_id(),
        "leader_url": PEER_URLS.get(bully.get_leader_id()) if bully.get_leader_id() is not None else None,
        "this_coordinator": COORD_ID,
        "this_role": bully.role.value,
    })


# ---------- FL endpoints (leader-only) ----------

@app.route("/register", methods=["POST"])
def register():
    redirect = _require_leader()
    if redirect is not None:
        return redirect

    address = request.remote_addr
    payload = request.get_json(silent=True) or {}
    client_clock = payload.get("clock", 0)

    with state.lock:
        new_clock = state.clock.receive_event(client_clock)
        cid = state.register_client(address)

    log.info(f"[L={new_clock}] Registered {cid} from {address}")
    return jsonify({
        "client_id": cid,
        "current_round": state.current_round,
        "clock": new_clock,
    })


@app.route("/get_model", methods=["GET"])
def get_model():
    redirect = _require_leader()
    if redirect is not None:
        return redirect

    client_clock = int(request.args.get("clock", 0))
    with state.lock:
        new_clock = state.clock.receive_event(client_clock)
        weights = get_model_weights(state.global_model)
        return jsonify({
            "round": state.current_round,
            "weights": weights,
            "training_complete": state.training_complete,
            "clock": new_clock,
        })


@app.route("/submit_update", methods=["POST"])
def submit_update():
    redirect = _require_leader()
    if redirect is not None:
        return redirect

    payload = request.get_json()
    client_id = payload["client_id"]
    submitted_round = payload["round"]
    weights = payload["weights"]
    num_samples = payload["num_samples"]
    client_clock = payload.get("clock", 0)

    with state.lock:
        new_clock = state.clock.receive_event(client_clock)

        if submitted_round != state.current_round:
            log.warning(
                f"[L={new_clock}] {client_id} submitted for round {submitted_round} "
                f"but we're on round {state.current_round} — dropping."
            )
            return jsonify({
                "status": "stale",
                "current_round": state.current_round,
                "clock": new_clock,
            }), 409

        state.update_heartbeat(client_id)
        state.round_updates[submitted_round].append({
            "client_id": client_id,
            "weights": weights,
            "num_samples": num_samples,
            "client_clock": client_clock,
        })
        num_received = len(state.round_updates[submitted_round])
        log.info(
            f"[L={new_clock}] Round {submitted_round}: received update from {client_id} "
            f"({num_samples} samples). Total: {num_received}/{MIN_CLIENTS_PER_ROUND}"
        )

        if num_received >= MIN_CLIENTS_PER_ROUND:
            _aggregate_and_advance()

    return jsonify({"status": "accepted", "round": submitted_round, "clock": state.clock.value})


@app.route("/heartbeat", methods=["POST"])
def heartbeat():
    redirect = _require_leader()
    if redirect is not None:
        return redirect

    payload = request.get_json()
    client_id = payload.get("client_id")
    client_clock = payload.get("clock", 0)
    with state.lock:
        new_clock = state.clock.receive_event(client_clock)
        state.update_heartbeat(client_id)
    return jsonify({"status": "ok", "current_round": state.current_round, "clock": new_clock})


@app.route("/status", methods=["GET"])
def status():
    with state.lock:
        return jsonify({
            "coordinator_id": COORD_ID,
            "role": bully.role.value,
            "is_leader": bully.is_leader(),
            "leader_id": bully.get_leader_id(),
            "current_round": state.current_round,
            "training_complete": state.training_complete,
            "registered_clients": list(state.clients.keys()),
            "alive_clients": state.alive_clients(),
            "updates_this_round": len(state.round_updates[state.current_round]),
            "round_history": state.round_history,
            "lamport_clock": state.clock.value,
        })


# ---------- Aggregation ----------

def _aggregate_and_advance():
    round_num = state.current_round
    updates = state.round_updates[round_num]

    state.clock.tick()
    log.info(f"[L={state.clock.value}] --- Aggregating round {round_num} with {len(updates)} updates ---")
    start = time.time()

    new_weights = federated_average(updates)
    set_model_weights(state.global_model, new_weights)

    elapsed = time.time() - start
    state.round_history.append({
        "round": round_num,
        "num_clients": len(updates),
        "client_ids": [u["client_id"] for u in updates],
        "client_clocks": [u["client_clock"] for u in updates],
        "server_clock_at_aggregation": state.clock.value,
        "aggregation_time_sec": round(elapsed, 3),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "leader_coordinator_id": COORD_ID,
    })
    log.info(f"[L={state.clock.value}] Round {round_num} complete in {elapsed:.2f}s. -> round {round_num + 1}.")

    state.current_round += 1
    if state.current_round >= MAX_ROUNDS:
        state.training_complete = True
        log.info(f"[L={state.clock.value}] 🎉 Training complete after {MAX_ROUNDS} rounds!")


# ---------- Background reaper ----------

def _reap_dead_clients():
    while True:
        time.sleep(10)
        if not bully.is_leader():
            continue  # only the leader tracks clients
        with state.lock:
            now = time.time()
            dead = [cid for cid, info in state.clients.items()
                    if now - info["last_seen"] > HEARTBEAT_TIMEOUT]
            if dead:
                log.warning(f"[L={state.clock.value}] 💀 Clients silent: {dead}")


# ---------- Entry point ----------

if __name__ == "__main__":
    log.info(f"=" * 60)
    log.info(f"Starting coordinator {COORD_ID} on {HOST}:{PORT}")
    log.info(f"My URL: {MY_URL}")
    log.info(f"Peers: {PEER_URLS}")
    log.info(f"=" * 60)

    # Start the bully election monitor
    bully.start_background_thread()

    # Start the dead-client reaper (only does work if we're leader)
    threading.Thread(target=_reap_dead_clients, daemon=True).start()

    app.run(host=HOST, port=PORT, threaded=True, debug=False)