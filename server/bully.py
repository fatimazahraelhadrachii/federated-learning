"""
Bully Leader Election Algorithm (Garcia-Molina, 1982).

Why we need this:
  Our original system has ONE coordinator. If it crashes, the entire
  federated learning system stops. This is a "single point of failure" —
  exactly what distributed systems are supposed to prevent.

  With Bully, we run multiple coordinator processes. One is the active
  leader; the others are passive standbys. When the leader dies, the
  surviving coordinator with the HIGHEST ID becomes the new leader,
  automatically and without human intervention.

The algorithm in 4 rules:
  1. Every coordinator has a unique integer ID.
  2. The highest-ID coordinator should be the leader.
  3. When a coordinator notices the leader is silent, it starts an
     election by sending ELECTION to all higher-ID peers.
       - If anyone replies (OK), it backs off — they'll handle it.
       - If nobody replies, it declares itself leader and broadcasts
         COORDINATOR to everyone.
  4. When a previously-dead coordinator comes back, it checks whether
     it has the highest ID among live nodes. If so, it starts an election
     to "bully" the current leader out of office (hence the name).

Trade-offs (discuss in report):
  + Simple to implement, well-studied
  + No external dependency (no Zookeeper / etcd needed)
  - O(N²) messages in the worst case
  - Assumes reliable failure detection (heartbeats) — false positives
    can cause split-brain
  - Doesn't replicate state. Each coordinator has its own model.
    Real systems (Raft, Paxos) replicate the log of operations so any
    coordinator can take over with the same state.
"""
import logging
import threading
import time
from enum import Enum

import requests

log = logging.getLogger(__name__)


class Role(Enum):
    LEADER = "leader"
    FOLLOWER = "follower"
    CANDIDATE = "candidate"  # transient — only during an election


class BullyElection:
    """
    Manages leader election for a single coordinator node.

    Each coordinator instantiates this with:
      - its own ID (small int, must be unique)
      - the URLs of all peer coordinators (including itself)

    The coordinator is responsible for serving HTTP endpoints that this
    class reads/writes:
      POST /bully/election    -> peer is starting an election
      POST /bully/coordinator -> peer is declaring itself leader
      GET  /bully/whoisleader -> ask any node who the current leader is
    """

    # How long to wait for OK replies before declaring victory
    ELECTION_TIMEOUT = 5.0
    # How often to ping the current leader to make sure it's alive
    LEADER_HEARTBEAT_INTERVAL = 3.0
    # If we don't hear from the leader in this long, start an election
    LEADER_TIMEOUT = 10.0
    # HTTP timeouts (must be short — we expect dead nodes)
    HTTP_TIMEOUT = 2.0

    def __init__(self, my_id, my_url, peer_urls):
        """
        Args:
            my_id: int, unique ID for this coordinator (higher = more priority)
            my_url: str, my own URL (e.g. "http://coordinator-2:5052")
            peer_urls: dict {peer_id: url} — should include all coordinators
                       (including ourselves)
        """
        self.my_id = my_id
        self.my_url = my_url
        self.peer_urls = peer_urls  # {id: url}

        # State
        self.role = Role.FOLLOWER
        self.current_leader_id = None
        self.last_heard_from_leader = time.time()
        self.election_in_progress = False

        self.lock = threading.Lock()
        log.info(f"Bully initialized: my_id={my_id}, peers={list(peer_urls.keys())}")

    # ---------- Public API for the coordinator to call ----------

    def is_leader(self):
        with self.lock:
            return self.role == Role.LEADER

    def get_leader_id(self):
        with self.lock:
            return self.current_leader_id

    def start_background_thread(self):
        """Run the election monitor in the background."""
        t = threading.Thread(target=self._monitor_loop, daemon=True)
        t.start()
        return t

    # ---------- Election logic ----------

    def start_election(self):
        """
        Initiate an election. Called when this node thinks the leader is dead.
        """
        with self.lock:
            if self.election_in_progress:
                log.debug(f"[bully] Election already in progress, skipping.")
                return
            self.election_in_progress = True
            self.role = Role.CANDIDATE

        log.info(f"[bully] Node {self.my_id} starting election")

        # Find peers with HIGHER ID
        higher_peers = {pid: url for pid, url in self.peer_urls.items()
                        if pid > self.my_id}

        if not higher_peers:
            # We have the highest ID — we win automatically
            log.info(f"[bully] No higher-ID peers exist. Node {self.my_id} wins by default.")
            self._become_leader()
            return

        # Send ELECTION to all higher-ID peers
        got_ok = False
        for pid, url in higher_peers.items():
            try:
                r = requests.post(
                    f"{url}/bully/election",
                    json={"from_id": self.my_id},
                    timeout=self.HTTP_TIMEOUT,
                )
                if r.status_code == 200 and r.json().get("ok"):
                    log.info(f"[bully] Got OK from higher peer {pid}, backing off.")
                    got_ok = True
            except requests.RequestException:
                log.debug(f"[bully] Peer {pid} didn't respond (likely down).")

        if got_ok:
            # A higher peer is alive — they'll handle the election. Wait.
            with self.lock:
                self.role = Role.FOLLOWER
                self.election_in_progress = False
            log.info(f"[bully] Stepping back; waiting for higher peer to announce leadership.")
        else:
            # No higher peer responded — we win!
            log.info(f"[bully] No higher peers responded. Node {self.my_id} declaring victory.")
            self._become_leader()

    def _become_leader(self):
        """Declare ourselves leader and broadcast to all peers."""
        with self.lock:
            self.role = Role.LEADER
            self.current_leader_id = self.my_id
            self.election_in_progress = False
            self.last_heard_from_leader = time.time()

        log.info(f"[bully] 👑 Node {self.my_id} is now LEADER.")

        # Tell everyone (except ourselves)
        for pid, url in self.peer_urls.items():
            if pid == self.my_id:
                continue
            try:
                requests.post(
                    f"{url}/bully/coordinator",
                    json={"leader_id": self.my_id, "leader_url": self.my_url},
                    timeout=self.HTTP_TIMEOUT,
                )
            except requests.RequestException:
                log.debug(f"[bully] Couldn't notify peer {pid} (it may be down).")

    # ---------- Endpoints (called by Flask handlers) ----------

    def handle_election_message(self, from_id):
        """
        A peer sent us ELECTION. If we have a higher ID than them, reply OK
        and start our own election to make sure WE become leader.
        """
        log.info(f"[bully] Received ELECTION from node {from_id}")
        if from_id < self.my_id:
            # Reply OK to suppress them, then start our own election
            threading.Thread(target=self.start_election, daemon=True).start()
            return {"ok": True, "from_id": self.my_id}
        else:
            # They have a higher (or equal) ID — they should win, not us
            return {"ok": False}

    def handle_coordinator_message(self, leader_id, leader_url):
        """A peer declared itself the new leader."""
        with self.lock:
            self.current_leader_id = leader_id
            self.last_heard_from_leader = time.time()
            self.election_in_progress = False
            if leader_id != self.my_id:
                self.role = Role.FOLLOWER
        log.info(f"[bully] Acknowledged new leader: node {leader_id}")
        return {"ok": True}

    def heard_from_leader(self):
        """Called by the coordinator when it hears any message from the leader."""
        with self.lock:
            self.last_heard_from_leader = time.time()

    # ---------- Background monitor ----------

    def _monitor_loop(self):
        """
        Background thread that:
          - If we're a follower: check that the leader is still alive.
            If silent for too long, start an election.
          - If we're the leader: just track that we're up.
        """
        # Initial pause so all coordinators boot before we start
        time.sleep(2)

        # On startup, immediately try to find or become leader
        self._initial_election()

        while True:
            time.sleep(self.LEADER_HEARTBEAT_INTERVAL)
            with self.lock:
                role = self.role
                leader_id = self.current_leader_id
                last_heard = self.last_heard_from_leader

            if role == Role.LEADER:
                # We're the leader, nothing to monitor
                continue

            # We're a follower — is the leader alive?
            if leader_id is None:
                # No known leader — start an election
                log.info(f"[bully] No known leader. Starting election.")
                self.start_election()
                continue

            # Ping the leader
            leader_url = self.peer_urls.get(leader_id)
            if not leader_url:
                # Unknown leader URL — shouldn't happen, but trigger election
                self.start_election()
                continue

            try:
                r = requests.get(f"{leader_url}/bully/whoisleader",
                                 timeout=self.HTTP_TIMEOUT)
                if r.status_code == 200:
                    self.heard_from_leader()
                else:
                    raise requests.RequestException("non-200 status")
            except requests.RequestException:
                # Leader is silent
                if time.time() - last_heard > self.LEADER_TIMEOUT:
                    log.warning(f"[bully] Leader {leader_id} unreachable for "
                                f"{time.time() - last_heard:.0f}s. Starting election.")
                    self.start_election()

    def _initial_election(self):
        """On startup, try to discover the current leader. If none, elect one."""
        # Ask every peer who they think the leader is
        for pid, url in self.peer_urls.items():
            if pid == self.my_id:
                continue
            try:
                r = requests.get(f"{url}/bully/whoisleader",
                                 timeout=self.HTTP_TIMEOUT)
                if r.status_code == 200:
                    data = r.json()
                    if data.get("leader_id") is not None:
                        with self.lock:
                            self.current_leader_id = data["leader_id"]
                            self.last_heard_from_leader = time.time()
                        log.info(f"[bully] Discovered existing leader: {data['leader_id']}")
                        return
            except requests.RequestException:
                continue

        # No existing leader found — start an election
        log.info(f"[bully] No existing leader found on startup. Starting election.")
        self.start_election()