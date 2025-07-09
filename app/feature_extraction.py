import pandas as pd 

def create_features(data):
    # Normalize Video_Call_Minutes to a 0-1 scale
    data["Video_Call_Score"] = data["screen_time_minutes"] / data["screen_time_minutes"].max()

    # Breaks per hour of screen time (Avoid division by zero)
    data["Break_Efficiency"] = data["breaks_taken"] / (data["screen_time_minutes"] + 1e-3)

    # Calculate the average screen time in hours
    data["Average_Screen_Time_Hours"] = data["screen_time_minutes"] / 60

    # Composite Burnout Indicator (just for experimentation)
    data["Cognitive_Load_Index"] = (
        0.4 * data["screen_time_minutes"] +
        0.3 * data["Video_Call_Score"] -
        0.2 * data["Break_Efficiency"]
    )

    return data