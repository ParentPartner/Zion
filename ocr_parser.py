import re
from datetime import datetime, time
import easyocr
import streamlit as st
from typing import List, Tuple
from config import tz

def extract_text_from_image(image_file) -> List[str]:
    """Extract text from an image using OCR."""
    try:
        reader = easyocr.Reader(["en"], gpu=False)
        img_bytes = image_file.read()
        image_file.seek(0)
        return reader.readtext(img_bytes, detail=0)
    except Exception as e:
        st.error(f"OCR processing failed: {e}")
        return []

def parse_screenshot_text(text_list: List[str]) -> Tuple[datetime, float, float, str]:
    """Parse OCR text to extract delivery information."""
    joined = " ".join(text_list).lower()
    
    # Default values
    ts = datetime.now(tz)
    total = 0.0
    ml = 0.0
    order_type = "Delivery"
    
    try:
        # Improved amount detection with multiple patterns
        amount_patterns = [
            r"\$?(\d{1,3}(?:,\d{3})*\.\d{2})",  # $1,234.56 or 1,234.56
            r"\$?(\d+\.\d{2})\b",               # $12.34 or 12.34
            r"\$(\d+)\b",                        # $12 (whole dollar amounts)
            r"total.*?(\d+\.\d{2})"              # Total: 12.34
        ]
        
        for pattern in amount_patterns:
            dollar_matches = re.findall(pattern, joined)
            if dollar_matches:
                amounts = [float(amt.replace(',', '')) for amt in dollar_matches]
                total = max(amounts)
                break
        
        # Miles detection with multiple patterns
        mile_patterns = [
            r"(\d+(?:\.\d)?)\s?mi(?:les)?",     # 1.2 mi or 1.2 miles
            r"distance.*?(\d+(?:\.\d)?)",        # Distance: 1.2
            r"(\d+(?:\.\d)?)\s?miles?\b"        # 1.2 mile or 1.2 miles
        ]
        
        for pattern in mile_patterns:
            miles = re.findall(pattern, joined)
            if miles:
                ml = float(miles[0])
                break
        
        # Enhanced time parsing
        time_patterns = [
            r"\b(\d{1,2}):(\d{2})\s?([ap]m)?\b",  # 12:30 PM or 12:30
            r"\b(\d{1,2})\s?([ap]m)\b",           # 12 PM
            r"time.*?(\d{1,2}):(\d{2})"           # Time: 12:30
        ]
        
        for pattern in time_patterns:
            time_match = re.search(pattern, joined, re.IGNORECASE)
            if time_match:
                groups = time_match.groups()
                if len(groups) >= 2:
                    hour = int(groups[0])
                    minute = int(groups[1]) if len(groups) > 1 and groups[1].isdigit() else 0
                    period = groups[-1].lower() if len(groups) > 2 and groups[-1] in ['am', 'pm'] else None
                    
                    if period:
                        if period == "pm" and hour < 12:
                            hour += 12
                        elif period == "am" and hour == 12:
                            hour = 0
                    elif hour < 6 or hour > 21:
                        current_hour = ts.hour
                        if current_hour < 12:  # AM
                            if hour > 11:
                                hour -= 12
                        else:  # PM
                            if hour < 12:
                                hour += 12
                    
                    try:
                        ts = ts.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    except ValueError:
                        pass
                break
        
        # Enhanced order type detection
        type_keywords = {
            "Shop": ["shop", "s&d", "shopping", "scan", "item", "shopping order"],
            "Pickup": ["pickup", "curbside", "pick up", "store pickup", "pick-up"]
        }
        
        for t, keywords in type_keywords.items():
            if any(kw in joined for kw in keywords):
                order_type = t
                break
    
    except Exception as e:
        st.error(f"Error parsing OCR text: {e}")
    
    return ts, total, ml, order_type