import math

EARTH_RADIUS = 6371e3  # Earth's radius in meters
MIN_ELEVATION_ANGLE = 25  # Minimum elevation angle in degrees
LIGHT_SPEED = 3e8  # Speed of light in m/s
FREQUENCY = 12e9  # Ku-band frequency in Hz (12 GHz)
RECEIVER_SENSITIVITY = -80  # Receiver sensitivity in dBm


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate the great-circle distance between two points on the Earth."""
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS * c


def free_space_loss(slant_range, frequency):
    """
    Calculate the free-space path loss (Lfs) in dB.
    - slant_range: Distance between satellite and ground station (meters)
    - frequency: Signal frequency (Hz)
    """
    return 20 * math.log10(slant_range) + 20 * math.log10(frequency) + 20 * math.log10(4 * math.pi / LIGHT_SPEED)


def link_budget_cost(slant_range, frequency, pt, gt, gr, lm, lsys):
    """
    Calculate the received power (Pr) in dBm based on the link budget.
    - slant_range: Distance between satellite and ground station (meters)
    - frequency: Signal frequency (Hz)
    - pt: Transmit power (dBm)
    - gt: Transmitter antenna gain (dBi)
    - gr: Receiver antenna gain (dBi)
    - lm: Miscellaneous losses (dB)
    - lsys: System losses (dB)
    """
    lfs = free_space_loss(slant_range, frequency)
    pr = pt + gt + gr - lfs - lm - lsys
    return pr
