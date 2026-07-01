import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
n = 1200

locations = ["Downtown", "Suburb", "Rural", "Urban Fringe"]
location_premium = {"Downtown": 55000, "Suburb": 20000, "Urban Fringe": 8000, "Rural": -15000}
location_probs = [0.2, 0.4, 0.15, 0.25]

df = pd.DataFrame({
    "area_sqft": rng.normal(1800, 650, n).clip(400, 6000).round(0),
    "location": rng.choice(locations, size=n, p=location_probs),
    "bedrooms": rng.integers(1, 7, n),
    "bathrooms": rng.integers(1, 5, n),
    "age_years": rng.integers(0, 90, n),
    "garage_spaces": rng.integers(0, 4, n),
    "lot_size_sqft": rng.normal(6000, 2500, n).clip(1000, 25000).round(0),
    "has_basement": rng.choice(["Yes", "No"], size=n, p=[0.55, 0.45]),
    "overall_quality": rng.integers(1, 11, n),  # 1-10 subjective quality score
    "distance_to_city_center_km": rng.exponential(8, n).clip(0.2, 60).round(2),
})

# Bedrooms shouldn't wildly exceed what area supports -- add mild coupling for realism
df["bedrooms"] = np.clip(df["bedrooms"], 1, (df["area_sqft"] // 350).astype(int).clip(1, 7))

# --- Price generation: a "true" underlying linear-ish process + noise ---
base_price = 40000
price = (
    base_price
    + df["area_sqft"] * 95
    + df["bedrooms"] * 6000
    + df["bathrooms"] * 9000
    + df["garage_spaces"] * 5000
    + df["overall_quality"] * 8500
    + df["lot_size_sqft"] * 2.2
    - df["age_years"] * 550
    - df["distance_to_city_center_km"] * 900
    + df["has_basement"].map({"Yes": 12000, "No": 0})
    + df["location"].map(location_premium)
)

# Add heteroskedastic noise (bigger homes -> bigger absolute noise) for realism
noise = rng.normal(0, 1, n) * (12000 + df["area_sqft"] * 8)
price = (price + noise).clip(35000, None).round(0)
df["price"] = price

# Introduce some missing values, like real-world messy data
for col, frac in [("lot_size_sqft", 0.04), ("garage_spaces", 0.02), ("bathrooms", 0.015)]:
    idx = rng.choice(df.index, size=int(frac * n), replace=False)
    df.loc[idx, col] = np.nan

# Shuffle column order a bit and reset index
df = df[[
    "area_sqft", "location", "bedrooms", "bathrooms", "age_years",
    "garage_spaces", "lot_size_sqft", "has_basement", "overall_quality",
    "distance_to_city_center_km", "price"
]]

output_dir = "data"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "house_prices.csv")
df.to_csv(output_path, index=False)
print(f"Saved dataset to {output_path}")
print(df.shape)
print(df.head())
print(df.isnull().sum())
