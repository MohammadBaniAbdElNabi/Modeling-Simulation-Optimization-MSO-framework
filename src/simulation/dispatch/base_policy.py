"""Abstract BaseDispatch interface for all dispatch policies."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulation.entities import BloodBank, Drone, Hospital, Request


class BaseDispatch(ABC):
    """Abstract base class for dispatch policies."""

    @abstractmethod
    def dispatch(
        self,
        env_now: float,
        hospitals: list["Hospital"],
        drones: list["Drone"],
        banks: list["BloodBank"],
        d_matrix: "object",  # np.ndarray (B, H)
        metrics: "object",
    ) -> list[tuple["Drone", "Request", "BloodBank"]]:
        """Select (drone, request, bank) assignment pairs for this dispatch cycle.

        Parameters
        ----------
        env_now   : current SimPy clock time in seconds
        hospitals : list of Hospital entities
        drones    : list of Drone entities
        banks     : list of BloodBank entities
        d_matrix  : shape (B, H) distance matrix in km
        metrics   : MetricsCollector

        Returns
        -------
        List of (drone, request, bank) tuples to dispatch this cycle.
        """
