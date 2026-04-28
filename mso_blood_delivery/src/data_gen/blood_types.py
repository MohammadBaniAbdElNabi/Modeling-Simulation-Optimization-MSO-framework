"""Blood type distribution constants."""
from typing import Final

BLOOD_TYPES: Final[list[str]] = ["O_neg", "O_pos", "A_pos", "B_pos"]
BLOOD_TYPE_PROBS: Final[list[float]] = [0.10, 0.40, 0.35, 0.15]

PRIORITY_CLASSES: Final[list[str]] = ["NORMAL", "URGENT", "EMERGENCY"]
PRIORITY_PROBS: Final[list[float]] = [0.50, 0.35, 0.15]

PRIORITY_VALUES: Final[dict[str, int]] = {
    "NORMAL": 1,
    "URGENT": 2,
    "EMERGENCY": 3,
}
