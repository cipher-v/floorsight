"""Display measured quantities without turning unknown values into zero."""
import math


def floor_error_text(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "unavailable"
    return f"{value:.2f} m"
