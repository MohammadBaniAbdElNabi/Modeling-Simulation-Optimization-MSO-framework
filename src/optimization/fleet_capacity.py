"""C_fleet computation: maximum drone throughput per hourly window."""
import math


def compute_c_fleet(
    fleet_size: int,
    window_minutes: int = 60,
    t_avg_mission_min: float = 20.0,
) -> int:
    """Compute fleet throughput cap per window.

    C_fleet = floor(c * T_window / t_avg_mission)
    Default: floor(8 * 60 / 20) = 24
    """
    return math.floor(fleet_size * window_minutes / t_avg_mission_min)
