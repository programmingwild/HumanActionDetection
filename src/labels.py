"""Human-readable display names for action labels.

Models train on folder names (calling, listening_to_music, ...).
Everything user-facing should use pretty() instead.
"""

DISPLAY_NAMES = {
    "calling": "Calling",
    "clapping": "Clapping",
    "cycling": "Cycling",
    "dancing": "Dancing",
    "drinking": "Drinking",
    "eating": "Eating",
    "fighting": "Fighting",
    "hugging": "Hugging",
    "laughing": "Laughing",
    "listening_to_music": "Listening to Music",
    "running": "Running",
    "sitting": "Sitting",
    "sleeping": "Sleeping",
    "texting": "Texting",
    "using_laptop": "Using Laptop",
    # KTH video branch
    "boxing": "Boxing",
    "handclapping": "Hand Clapping",
    "handwaving": "Hand Waving",
    "jogging": "Jogging",
    "walking": "Walking",
}


def pretty(name: str) -> str:
    """Raw class name -> display name (falls back to title-cased)."""
    if name in DISPLAY_NAMES:
        return DISPLAY_NAMES[name]
    return name.replace("_", " ").title()
