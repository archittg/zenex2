# priority.py
# ----------------------------------------------------
# Maps a detected object "type" -> an importance label.
# This is the DEFAULT table (mission-agnostic).
# mission_modes.py will later override these per mission.
# ----------------------------------------------------

DEFAULT_PRIORITY_TABLE = {
    # --- Disaster / hazard events ---
    "wildfire": "CRITICAL",
    "flood": "CRITICAL",
    "volcanic_eruption": "CRITICAL",
    "landslide": "CRITICAL",
    "earthquake_damage": "CRITICAL",
    "explosion": "CRITICAL",
    "smoke_plume": "HIGH",
    "oil_spill": "HIGH",
    "storm_system": "HIGH",
    "drought_area": "MEDIUM",

    # --- Maritime ---
    "unknown_ship": "HIGH",
    "military_vessel": "HIGH",
    "ship": "MEDIUM",
    "fishing_boat": "MEDIUM",
    "tanker": "MEDIUM",
    "small_boat": "LOW",
    "buoy": "LOW",
    "wake_trail": "LOW",

    # --- Aircraft ---
    "unknown_aircraft": "HIGH",
    "military_aircraft": "HIGH",
    "commercial_aircraft": "LOW",
    "drone": "MEDIUM",

    # --- Infrastructure / human activity ---
    "building": "LOW",
    "airport": "LOW",
    "bridge": "LOW",
    "power_plant": "MEDIUM",
    "military_base": "HIGH",
    "construction_site": "LOW",
    "road": "LOW",
    "vehicle_convoy": "MEDIUM",
    "vehicle": "LOW",
    "railway": "LOW",

    # --- Environment / land ---
    "deforestation": "MEDIUM",
    "crop_field": "LOW",
    "glacier": "LOW",
    "coastline_erosion": "MEDIUM",

    # --- People / crowds ---
    "crowd_gathering": "MEDIUM",
    "refugee_camp": "HIGH",

    # --- Noise / irrelevant detections ---
    "cloud": "DISCARD",
    "shadow": "DISCARD",
    "ocean_glare": "DISCARD",
    "sensor_noise": "DISCARD",

    # --- Fallback ---
    "unknown_thing": "LOW",
}


def get_priority(object_type: str, table: dict = None) -> str:
    """
    Look up the priority label for a given object type.
    Falls back to 'LOW' if the type isn't in the table at all
    (safer default than crashing or silently discarding).
    """
    table = table or DEFAULT_PRIORITY_TABLE
    return table.get(object_type, "LOW")


if __name__ == "__main__":
    test_types = ["ship", "wildfire", "building", "unknown_thing", "military_base", "cloud"]

    for t in test_types:
        print(f"{t:15} -> {get_priority(t)}")