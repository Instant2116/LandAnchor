import math
import datetime
from typing import Dict, Optional


def format_timestamp() -> str:
    """
    Generates a standardized UTC timestamp string.
    Format adheres to the database schema: YYYY-MM-DD HH:MM:SS
    """
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def calculate_azimuth(
    current_point: Dict[str, float], previous_point: Optional[Dict[str, float]] = None
) -> float:
    """
    Calculates the azimuth (heading) in degrees normalized to 0-360.

    If 'yaw' is provided in current_point, it normalizes it.
    If 'yaw' is missing, it calculates the bearing based on the
    Haversine forward azimuth using previous_point GPS coordinates.
    """
    if "yaw" in current_point and current_point["yaw"] is not None:
        # Normalizes any degree-based yaw to strict 0-360 range
        return (current_point["yaw"] + 360.0) % 360.0

    if not previous_point:
        raise ValueError(
            "Missing 'yaw' data. A previous_point containing 'lat' and 'lon' "
            "is required to calculate bearing."
        )

    lat1 = math.radians(previous_point.get("lat", 0.0))
    lon1 = math.radians(previous_point.get("lon", 0.0))
    lat2 = math.radians(current_point.get("lat", 0.0))
    lon2 = math.radians(current_point.get("lon", 0.0))

    d_lon = lon2 - lon1

    x = math.sin(d_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (
        math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    )

    initial_bearing = math.atan2(x, y)

    # Convert from radians to degrees and normalize to 0-360
    initial_bearing = math.degrees(initial_bearing)
    azimuth = (initial_bearing + 360.0) % 360.0

    return round(azimuth, 6)
