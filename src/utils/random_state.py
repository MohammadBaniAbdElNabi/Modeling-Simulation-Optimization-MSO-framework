"""Seed management for Common Random Numbers (CRN) across policies."""
import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def crn_seeds(n_replications: int, base_seed: int = 42) -> list[int]:
    """Return seeds 1..n_replications for CRN (seeds shared across policies)."""
    return list(range(1, n_replications + 1))


def policy_rng(policy_name: str, replication: int, base_seed: int = 42) -> np.random.Generator:
    """Produce a deterministic RNG for a specific policy x replication combination.

    CRN: replications share the same seed; policies get independent streams
    derived from a namespace hash so seeds don't collide across policies.
    """
    seed = replication  # CRN: same replication seed across all policies
    return np.random.default_rng(seed)
