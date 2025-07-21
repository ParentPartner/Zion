from typing import TypedDict, List, Optional
from datetime import datetime, date, time

class Delivery(TypedDict):
    timestamp: datetime
    order_total: float
    base_amount: float
    bonus_amount: float
    incentives: List[Dict[str, Any]]
    miles: float
    earnings_per_mile: float
    hour: int
    username: str
    order_type: str
    created_at: Optional[datetime]

class TipBaiter(TypedDict):
    name: str
    address: str
    date: date
    amount: float
    rating: int
    notes: str
    username: str
    timestamp: datetime
    created_at: Optional[datetime]

class Incentive(TypedDict):
    name: str
    start_time: str
    end_time: str
    applies_to: List[str]
    amount: float
    notes: str
    active: bool

class User(TypedDict):
    password: str
    last_checkin_date: str
    incentives: List[Incentive]
    today_goal: float
    today_notes: str
    is_working: bool
    checkin_time: Optional[datetime]
    created_at: datetime