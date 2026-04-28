"""Distance matrix loader and flight time computation."""
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.config_loader import load_config


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def load_distance_matrix(network_cfg: dict[str, Any]) -> np.ndarray:
    """Return distance matrix d[b, h] in km (shape: B x H) from config.

    Uses the precomputed distance_matrix if available, otherwise falls back
    to haversine computation from coordinates.
    """
    if "distance_matrix" in network_cfg:
        banks = network_cfg["blood_banks"]
        rows = []
        for bank in banks:
            bid = bank["id"]
            rows.append(network_cfg["distance_matrix"][bid])
        return np.array(rows, dtype=float)

    # Fallback: compute from coordinates
    banks = network_cfg["blood_banks"]
    hospitals = network_cfg["hospitals"]
    B, H = len(banks), len(hospitals)
    d = np.zeros((B, H))
    for b_idx, bank in enumerate(banks):
        for h_idx, hosp in enumerate(hospitals):
            d[b_idx, h_idx] = haversine_km(
                bank["lat"], bank["lon"], hosp["lat"], hosp["lon"]
            )
    return d


def flight_time_minutes(d_km: float, speed_kmh: float) -> float:
    """One-way flight time in minutes."""
    return d_km / speed_kmh * 60.0


def flight_time_matrix(d_matrix: np.ndarray, speed_kmh: float) -> np.ndarray:
    """Convert distance matrix (km) to flight time matrix (minutes)."""
    return d_matrix / speed_kmh * 60.0
