# mission_modes.py
# ----------------------------------------------------
# Same object, different priority, depending on the
# CURRENT MISSION the satellite is running.
#
# This overrides priority.py's default table when a
# specific mission mode is active.
# ----------------------------------------------------

DISASTER_RESPONSE = {
    "wildfire": "CRITICAL",
    "flood": "CRITICAL",
    "landslide": "CRITICAL",
    "smoke_plume": "HIGH",
    "refugee_camp": "HIGH",
    "building": "MEDIUM",
    "road": "MEDIUM",
    "ship": "LOW",
    "vehicle": "LOW",
    "cloud": "DISCARD",
}

MARITIME_SURVEILLANCE = {
    "unknown_ship": "CRITICAL",
    "military_vessel": "CRITICAL",
    "ship": "MEDIUM",
    "tanker": "MEDIUM",
    "fishing_boat": "MEDIUM",
    "wildfire": "MEDIUM",
    "oil_spill": "HIGH",
    "small_boat": "LOW",
    "buoy": "LOW",
    "cloud": "DISCARD",
}

URBAN_MONITORING = {
    "crowd_gathering": "HIGH",
    "vehicle_convoy": "MEDIUM",
    "construction_site": "LOW",
    "building": "LOW",
    "road": "LOW",
    "wildfire": "CRITICAL",
    "ship": "LOW",
    "cloud": "DISCARD",
}

# Map mode name -> its table, so decision.py can pick one easily
MISSION_MODES = {
    "DISASTER_RESPONSE": DISASTER_RESPONSE,
    "MARITIME_SURVEILLANCE": MARITIME_SURVEILLANCE,
    "URBAN_MONITORING": URBAN_MONITORING,
}


def get_priority_for_mode(object_type: str, mode: str) -> str:
    """
    Look up priority for an object type under a specific mission mode.
    Falls back to 'LOW' if the type isn't listed for that mode.
    """
    table = MISSION_MODES.get(mode, {})
    return table.get(object_type, "LOW")


if __name__ == "__main__":
    for mode_name, table in MISSION_MODES.items():
        print(f"--- Mode: {mode_name} ---")
        for object_type, label in table.items():
            print(f"{object_type:20} -> {label}")
        print()