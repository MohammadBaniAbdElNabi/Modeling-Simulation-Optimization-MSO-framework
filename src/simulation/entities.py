"""Request, Drone, Hospital, BloodBank dataclasses for the SimPy DES."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import ClassVar


class DroneState(IntEnum):
    IDLE = 0
    LOADING = 1
    TRANSIT_TO_HOSPITAL = 2
    DELIVERING = 3
    RETURNING = 4
    RECHARGING = 5
    UNAVAILABLE = 6


class Priority(IntEnum):
    NORMAL = 1
    URGENT = 2
    EMERGENCY = 3


_req_counter: int = 0


@dataclass
class Request:
    """A single blood delivery request."""
    hospital_id: int
    blood_type: str
    units_needed: int
    priority: Priority
    arrival_time: float           # seconds
    expiration_time: float        # seconds

    # Set after assignment / delivery
    assignment_time: float = field(default=float("nan"))
    delivery_time: float = field(default=float("nan"))
    assigned_drone_id: int = field(default=-1)
    assigned_bank_id: int = field(default=-1)

    # Status flags
    is_fulfilled: bool = False
    is_expired: bool = False

    # Unique id
    request_id: int = field(default_factory=lambda: Request._next_id())

    _counter: ClassVar[int] = 0

    @staticmethod
    def _next_id() -> int:
        Request._counter += 1
        return Request._counter

    @staticmethod
    def reset_counter() -> None:
        Request._counter = 0


@dataclass
class Drone:
    """A drone in the delivery fleet."""
    drone_id: int
    home_bank: int
    speed_kmh: float = 50.0
    battery_drain: float = 1.5    # % per km
    battery_min: float = 30.0     # %
    recharge_rate: float = 20.0   # % per minute
    max_payload: int = 4

    battery: float = 100.0
    state: DroneState = DroneState.IDLE


@dataclass
class BloodBank:
    """A blood bank with typed inventory."""
    bank_id: int
    blood_types: list[str] = field(default_factory=lambda: ["O_neg", "O_pos", "A_pos", "B_pos"])
    initial_per_type: int = 50

    inventory: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.inventory:
            self.inventory = {bt: self.initial_per_type for bt in self.blood_types}

    def total_inventory(self) -> int:
        return sum(self.inventory.values())

    def has_stock(self, blood_type: str, units: int) -> bool:
        return self.inventory.get(blood_type, 0) >= units

    def deduct(self, blood_type: str, units: int) -> None:
        self.inventory[blood_type] = max(0, self.inventory[blood_type] - units)


@dataclass
class Hospital:
    """A hospital that generates blood requests."""
    hospital_id: int
    pending_requests: list[Request] = field(default_factory=list)
