# confidence_policy.py
# ----------------------------------------------------
# Decides HOW MUCH DATA to send based on how confident
# the AI (Person 1) was in its detection.
#
# Logic: confident detections need less proof (just metadata).
# Unsure detections need more evidence sent down so a human
# on Earth can visually double-check.
# ----------------------------------------------------

# Each option: (package_name, size_in_kb)
DATA_PACKAGES = {
    "METADATA_ONLY": 1,
    "METADATA_PLUS_THUMBNAIL": 4,
    "METADATA_PLUS_IMAGE_CROP": 8,
    "METADATA_PLUS_FULL_IMAGE": 20,   # extra tier, for very unsure/critical cases
}


def get_data_package(confidence: float) -> dict:
    """
    Given a confidence score (0.0 - 1.0), decide what data
    package to send and how much it costs in KB.

    Returns a dict: {"package": str, "cost_kb": int}
    """
    if confidence >= 0.9:
        package = "METADATA_ONLY"
    elif confidence >= 0.6:
        package = "METADATA_PLUS_THUMBNAIL"
    elif confidence >= 0.3:
        package = "METADATA_PLUS_IMAGE_CROP"
    else:
        # Very unsure — send everything, let a human decide
        package = "METADATA_PLUS_FULL_IMAGE"

    return {
        "package": package,
        "cost_kb": DATA_PACKAGES[package],
    }


if __name__ == "__main__":
    test_confidences = [0.95, 0.75, 0.4, 0.15]

    for c in test_confidences:
        result = get_data_package(c)
        print(f"confidence={c} -> {result['package']} ({result['cost_kb']} KB)")