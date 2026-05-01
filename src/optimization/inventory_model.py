"""InventoryTracker: maintains blood bank inventory state between LP windows."""
from __future__ import annotations

import numpy as np


class InventoryTracker:
    """Track total inventory per blood bank across time windows.

    Inventory is treated as aggregate (all blood types combined) for LP
    constraint purposes.
    """

    def __init__(self, I_init: np.ndarray) -> None:
        """
        Parameters
        ----------
        I_init : shape (B,) -- initial total inventory per bank
        """
        self.I = I_init.astype(float).copy()

    def update(self, assignments: dict[tuple[int, int], int], B: int, H: int) -> None:
        """Deduct assigned units from inventory after solving window t.

        Parameters
        ----------
        assignments : {(b, h): count} for the just-solved window
        B, H        : index bounds
        """
        for b in range(B):
            used = sum(assignments.get((b, h), 0) for h in range(H))
            self.I[b] = max(0.0, self.I[b] - used)

    def snapshot(self) -> np.ndarray:
        return self.I.copy()
