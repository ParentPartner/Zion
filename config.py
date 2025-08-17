# config.py - Configuration and Constants

import pytz

# === CONFIGURATION ===
tz = pytz.timezone("US/Eastern")
TARGET_DAILY = 200
ORDER_TYPES = ["Delivery", "Shop", "Pickup"]
PERFORMANCE_LEVELS = {
    "Excellent": {"min_epm": 3.0, "min_eph": 30},
    "Good": {"min_epm": 2.0, "min_eph": 25},
    "Fair": {"min_epm": 1.5, "min_eph": 20},
    "Poor": {"min_epm": 0, "min_eph": 0}
}