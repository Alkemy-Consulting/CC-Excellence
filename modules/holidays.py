import pandas as pd

def get_holidays(country: str = "Italy") -> pd.DataFrame:
    """
    Fetch holidays for the specified country.

    Args:
        country (str): Country name for which holidays are to be fetched.

    Returns:
        pd.DataFrame: DataFrame containing holiday dates and names.
    """
    try:
        # Example implementation: Replace with actual holiday fetching logic
        if country.lower() == "italy":
            data = {
                "date": ["2025-01-01", "2025-04-25", "2025-12-25"],
                "name": ["New Year's Day", "Liberation Day", "Christmas Day"]
            }
            return pd.DataFrame(data)
        else:
            return pd.DataFrame(columns=["date", "name"])
    except Exception as e:
        raise ValueError(f"Error fetching holidays for {country}: {str(e)}")