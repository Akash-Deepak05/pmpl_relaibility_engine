# create_mock_data.py
import pandas as pd
import numpy as np
import random

print("Generating mock master_data.csv...")

# Define depots and bus types
DEPOTS = ["Swargate", "Kothrud", "Bhosari", "Hinjewadi", "Pimpri", "Nigdi"]
BUS_TYPES = ["Diesel", "CNG", "e-Bus"]
OWNER_TYPES = ["PMPML", "Private"]

# Create a date range (3 years of monthly data)
dates = pd.date_range(start="2022-01-01", end="2024-12-31", freq="MS")

records = []

# Simulate a fleet of 500 buses
for i in range(500):
    bus_id = f"BUS-{1000 + i}"
    depot = random.choice(DEPOTS)
    bus_type = random.choice(BUS_TYPES)
    # Make 'Private' buses more common, as they are a known issue
    owner = "Private" if random.random() < 0.6 else "PMPML"

    # Give each bus a base age (in months) at start of 2022
    bus_age_months_base = random.randint(1, 120)

    # Randomize slight propensity per bus for heterogeneity
    bus_effect = random.uniform(-0.02, 0.02)

    for date in dates:
        # Age grows with time
        months_since_2022 = (date.year - 2022) * 12 + (date.month - 1)
        bus_age_months = bus_age_months_base + months_since_2022

        # Simulate monthly KMs run
        kms_run = random.randint(3000, 8000)

        # Base breakdown probability
        breakdown_prob = 0.05
        # Older buses break down more (cap contribution)
        breakdown_prob += min(bus_age_months / 120.0, 1.0) * 0.10
        # Private buses break down more
        if owner == "Private":
            breakdown_prob += 0.10
        # e-Buses might have different issues (e.g., charging logistics)
        if bus_type == "e-Bus":
            breakdown_prob += 0.05

        # Per-bus effect and mild seasonal noise
        month_season = 0.02 if date.month in (5, 6) else 0.0  # example: hotter months
        breakdown_prob += bus_effect + month_season

        # Clamp to reasonable range
        breakdown_prob = max(0.01, min(0.8, breakdown_prob))

        # Simulate breakdowns
        total_breakdowns = 0
        if random.random() < breakdown_prob:
            total_breakdowns = random.randint(1, 3)

        records.append({
            "Bus_ID": bus_id,
            "Date": date,
            "Year": date.year,
            "Month": date.month,
            "Depot_Name": depot,
            "Bus_Type": bus_type,
            "Owner_Type": owner,
            "Bus_Age_Months": bus_age_months,
            "KMs_Run_Monthly": kms_run,
            "Total_Breakdowns": total_breakdowns,
        })

# Assemble and save
df = pd.DataFrame(records)
df.to_csv("master_data.csv", index=False)
print(f"Successfully created master_data.csv with {len(df)} rows.")
