from datetime import time
from typing import Tuple, List, Dict
from firebase_helpers import get_user
from config import ORDER_TYPES

def apply_incentives(order_type: str, order_time: time, base_amount: float, username: str) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Apply any matching incentives to the order."""
    user_data = get_user(username)
    incentives = user_data.get("incentives", []) if user_data else []
    bonus_amount = 0.0
    applied_incentives = []
    
    for incentive in incentives:
        if incentive.get("active", True) and order_type in incentive.get("applies_to", []):
            start_time = datetime.strptime(incentive["start_time"], "%H:%M").time()
            end_time = datetime.strptime(incentive["end_time"], "%H:%M").time()
            
            if start_time <= order_time <= end_time:
                bonus_amount += incentive["amount"]
                applied_incentives.append({
                    "name": incentive["name"],
                    "amount": incentive["amount"]
                })
    
    total_amount = base_amount + bonus_amount
    return total_amount, bonus_amount, applied_incentives