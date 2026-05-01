"""Four-condition specification (R3): isolate the value of *planning sophistication*.

All LP arms use the same Factor forecast. Variation is in planning method only:
  - Random:       no information, lower bound
  - Greedy:       reactive nearest-feasible, no plan
  - Static-LP:    24-hour plan committed at t=0
  - MPC-LP:       rolling-horizon LP re-solved hourly with current state
"""
from __future__ import annotations

CONDITIONS: list[dict] = [
    {
        "id": "random",
        "label": "Random",
        "demand_model": None,
        "policy": "random",
        "role": "reference",
    },
    {
        "id": "greedy",
        "label": "Greedy",
        "demand_model": None,
        "policy": "greedy",
        "role": "reference",
    },
    {
        "id": "lp_static",
        "label": "Static-LP",
        "demand_model": "factor",
        "policy": "lp_static",
        "role": "primary",
    },
    {
        "id": "lp_mpc",
        "label": "MPC-LP",
        "demand_model": "factor",
        "policy": "lp_mpc",
        "role": "primary",
    },
]

PRIMARY_IDS:   list[str] = [c["id"] for c in CONDITIONS if c["role"] == "primary"]
REFERENCE_IDS: list[str] = [c["id"] for c in CONDITIONS if c["role"] == "reference"]
ALL_IDS:       list[str] = [c["id"] for c in CONDITIONS]

CONDITION_LABELS: dict[str, str] = {c["id"]: c["label"] for c in CONDITIONS}
CONDITION_COLORS: dict[str, str] = {
    "random":    "#9467bd",
    "greedy":    "#ff7f0e",
    "lp_static": "#1a6faf",
    "lp_mpc":    "#d62728",
}
