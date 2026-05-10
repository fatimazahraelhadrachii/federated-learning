"""
Failure injection for federated learning clients.

The project brief requires "simulation des pannes ou des problèmes réseau
éventuels (partitionnement, perte de messages)". This module provides
controllable failure injection so we can demonstrate the system's
fault tolerance in our report.

Failure modes implemented:
    - crash:      client dies permanently mid-training
    - straggler:  client trains very slowly (random delays)
    - byzantine:  client sends corrupted weights (poisoning attack)
    - partition:  client temporarily can't reach the server, then recovers
    - message_loss: random network errors when submitting updates
"""
import random
import time
import logging

log = logging.getLogger(__name__)


class FailureInjector:
    """
    Simulates faults in a federated client.

    Configured via --failure-mode and --failure-rate command-line args.
    The probability check happens once per round, so you'll see the
    failure trigger at various points during training.
    """

    def __init__(self, mode="none", rate=0.0, seed=None):
        """
        Args:
            mode: one of {none, crash, straggler, byzantine, partition, message_loss}
            rate: probability (0.0 - 1.0) that the failure occurs each round
            seed: random seed for reproducibility (good for the report!)
        """
        self.mode = mode
        self.rate = float(rate)
        self.rng = random.Random(seed)
        self.has_crashed = False  # crash is permanent — once dead, stay dead
        self.is_partitioned = False
        self.partition_until = 0
        log.info(f"FailureInjector initialized: mode={mode}, rate={rate}")

    def should_trigger(self):
        """Roll the dice — returns True if we should inject a failure this round."""
        return self.rng.random() < self.rate

    # ---------- The 5 failure modes ----------

    def maybe_crash(self):
        """Crash failure — raise SystemExit so the process actually dies."""
        if self.mode != "crash" or self.has_crashed:
            return
        if self.should_trigger():
            self.has_crashed = True
            log.error("💥 FAILURE INJECTED: crash — client is dying now!")
            raise SystemExit(1)

    def maybe_straggle(self):
        """Straggler — sleep for a long random duration before training."""
        if self.mode != "straggler":
            return
        if self.should_trigger():
            delay = self.rng.uniform(15, 30)
            log.warning(f"🐌 FAILURE INJECTED: straggler — sleeping {delay:.1f}s before training")
            time.sleep(delay)

    def maybe_corrupt_weights(self, weights_dict):
        """Byzantine — replace some weights with garbage to poison the global model."""
        if self.mode != "byzantine":
            return weights_dict
        if self.should_trigger():
            log.error("☠️  FAILURE INJECTED: byzantine — sending corrupted weights")
            corrupted = {}
            for layer_name, values in weights_dict.items():
                # Replace with random noise of similar magnitude
                corrupted[layer_name] = self._random_like(values, scale=10.0)
            return corrupted
        return weights_dict

    def maybe_partition(self):
        """
        Partition — refuse to make any network call for the next 10-30 seconds.
        Returns True if we're currently in a partition.
        """
        if self.mode != "partition":
            return False

        # Check if a previous partition has expired
        if self.is_partitioned and time.time() > self.partition_until:
            log.info("🔌 Partition healed — client can reach the server again.")
            self.is_partitioned = False

        # Maybe start a new partition
        if not self.is_partitioned and self.should_trigger():
            duration = self.rng.uniform(10, 30)
            self.is_partitioned = True
            self.partition_until = time.time() + duration
            log.warning(f"🔌 FAILURE INJECTED: network partition — isolated for {duration:.1f}s")

        return self.is_partitioned

    def maybe_drop_message(self):
        """Message loss — randomly raise a connection error to simulate packet drop."""
        if self.mode != "message_loss":
            return
        if self.should_trigger():
            log.warning("📨 FAILURE INJECTED: message dropped (connection error)")
            raise ConnectionError("Simulated packet drop")

    # ---------- Helpers ----------

    @staticmethod
    def _random_like(nested_list, scale=1.0):
        """Replace a nested list of floats with random values of the same shape."""
        if isinstance(nested_list, list):
            return [FailureInjector._random_like(item, scale) for item in nested_list]
        else:
            return random.uniform(-scale, scale)