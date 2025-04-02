import math

# Constants
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


def satellite_visible(ground_lat, ground_lon, sat_lat, sat_lon, sat_alt):
    """
    Determine if the satellite is visible to the ground station.
    - ground_lat, ground_lon: Ground station latitude and longitude (degrees)
    - sat_lat, sat_lon: Satellite latitude and longitude (degrees)
    - sat_alt: Satellite altitude above the Earth's surface (meters)
    """
    # Calculate distance between the ground station and the satellite
    distance_ground_sat = haversine_distance(ground_lat, ground_lon, sat_lat, sat_lon)
    # Calculate slant range (distance between ground station and satellite in 3D)
    slant_range = math.sqrt(distance_ground_sat ** 2 + sat_alt ** 2)
    # Calculate the elevation angle
    elevation_angle = math.degrees(math.asin(sat_alt / slant_range))
    return elevation_angle >= MIN_ELEVATION_ANGLE, slant_range


def free_space_loss(slant_range, frequency):
    """
    Calculate the free-space path loss (Lfs) in dB.
    - slant_range: Distance between satellite and ground station (meters)
    - frequency: Signal frequency (Hz)
    """
    return 20 * math.log10(slant_range) + 20 * math.log10(frequency) + 20 * math.log10(4 * math.pi / LIGHT_SPEED)


def link_budget(slant_range, frequency, pt, gt, gr, lm, lsys):
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


# Example inputs
ground_station = {"lat": 40.7128, "lon": -74.0060}  # New York City (latitude, longitude)
satellite = {"lat": 41.0, "lon": -75.0, "alt": 550e3}  # Satellite position (latitude, longitude, altitude in meters)
transmit_power = 30  # Transmit power (dBm)
transmitter_gain = 25  # Satellite antenna gain (dBi)
receiver_gain = 30  # Ground station antenna gain (dBi)
misc_losses = 2  # Miscellaneous losses (dB)
system_losses = 2  # System losses (dB)

# Check visibility
visible, slant_range = satellite_visible(
    ground_station["lat"], ground_station["lon"],
    satellite["lat"], satellite["lon"],
    satellite["alt"]
)

if visible:
    print("Satellite is visible!")
    print(f"Slant range: {slant_range / 1e3:.2f} km")
    # Calculate received power
    received_power = link_budget(
        slant_range, FREQUENCY,
        transmit_power, transmitter_gain,
        receiver_gain, misc_losses,
        system_losses
    )
    print(f"Received power: {received_power:.2f} dBm")
    if received_power >= RECEIVER_SENSITIVITY:
        print("Signal is strong enough to be received.")
    else:
        print("Signal is too weak to be received.")
else:
    print("Satellite is not visible.")
