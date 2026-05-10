"""
Lamport Logical Clock (Lamport, 1978).

Solves the problem of ordering events in a distributed system without
relying on synchronized physical clocks. Each node maintains a counter
that follows three rules:

    1. Local event:      counter += 1
    2. Send a message:   counter += 1, attach counter to message
    3. Receive a message: counter = max(counter, msg_counter) + 1

This guarantees: if event A causally caused event B, then clock(A) < clock(B).
The reverse is NOT true (two concurrent events can have any ordering of clocks)
— that's a known limitation of Lamport clocks (vector clocks fix it).

Used in our FL system to:
  - Order events across clients and the server
  - Tag log messages with logical time
  - Track causal dependencies (a client's update is causally after the
    /get_model call that fetched the global weights it trained on)
"""
import threading


class LamportClock:
    """Thread-safe Lamport logical clock for a distributed node."""

    def __init__(self, node_id="node"):
        self._counter = 0
        self._lock = threading.Lock()
        self.node_id = node_id

    @property
    def value(self):
        """Read the current logical time."""
        with self._lock:
            return self._counter

    def tick(self):
        """
        Rule 1: a local event happened (e.g. trained a model, aggregated weights).
        Returns the new clock value.
        """
        with self._lock:
            self._counter += 1
            return self._counter

    def send_event(self):
        """
        Rule 2: about to send a message — increment then return the timestamp
        to attach to the outgoing message.
        """
        with self._lock:
            self._counter += 1
            return self._counter

    def receive_event(self, msg_timestamp):
        """
        Rule 3: received a message with the given timestamp.
        Update our clock to max(ours, theirs) + 1.
        Returns the new clock value.
        """
        with self._lock:
            self._counter = max(self._counter, int(msg_timestamp)) + 1
            return self._counter

    def __repr__(self):
        return f"LamportClock({self.node_id}={self._counter})"


if __name__ == "__main__":
    # Quick demo replicating the diagram in the report
    server = LamportClock("server")
    client_a = LamportClock("client_A")
    client_b = LamportClock("client_B")

    print("=== Lamport Clock Demo ===")
    print(f"Server local event:           {server.tick()}     -> {server}")
    print(f"Client A local event:         {client_a.tick()}     -> {client_a}")
    print(f"Client B local event:         {client_b.tick()}     -> {client_b}")

    ts = server.send_event()
    print(f"Server sends msg (ts={ts}) to A")
    print(f"  Client A receives (ts={ts}):    {client_a.receive_event(ts)}     -> {client_a}")

    print(f"Client A local event:         {client_a.tick()}     -> {client_a}")

    ts = client_a.send_event()
    print(f"Client A sends msg (ts={ts}) to server")
    print(f"  Server receives (ts={ts}):       {server.receive_event(ts)}     -> {server}")