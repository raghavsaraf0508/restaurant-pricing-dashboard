"""
Restaurant Daily Pricing Optimization — prototype ML script

Inputs:
  - City
  - Restaurant/outlet name
  - Product name or Product_ID (optional; blank = all products)
  - Day of week

Outputs:
  - Recommended price (one price for the whole day)
  - Last-week price proxy and sales proxy
  - Predicted units and contribution at the recommended price
  - Daily competitor benchmark and data-quality notes

IMPORTANT:
The current workbook contains synthetic, representative weekly data. Its "last week" values
are therefore historical-week proxies, not verified POS results from a dated calendar week.
For production, replace synthetic sales/price/cost/stockout fields with observed records.
The current dataset does not identify causal price elasticity. The script therefore uses an explicit configurable elasticity assumption as a temporary demand-response adjustment.
"""

from pathlib import Path
import math
import sys
import warnings
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

DEFAULT_WORKBOOK = "Restaurant_Pricing_Weekly_28_Slots_Enhanced.xlsx"
PRICE_ELASTICITY = -1.5  # Provisional assumption; estimate from real pricing experiments when possible.
SLOTS = [
    "12:00–15:00 Lunch",
    "15:00–18:00 Afternoon",
    "18:00–21:00 Dinner",
    "21:00–23:00 Late Evening",
]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def read_workbook(path):
    """Load required sheets and validate their presence."""
    xls = pd.ExcelFile(path)
    required = {
        "Model_Features_28_Slots",
        "Product_Cost_Constraints",
        "Outlet_Master",
        "Product_Master",
    }
    missing = required - set(xls.sheet_names)
    if missing:
        raise ValueError(f"Workbook is missing required sheets: {sorted(missing)}")
    data = {
        "features": pd.read_excel(path, sheet_name="Model_Features_28_Slots"),
        "costs": pd.read_excel(path, sheet_name="Product_Cost_Constraints"),
        "outlets": pd.read_excel(path, sheet_name="Outlet_Master"),
        "products": pd.read_excel(path, sheet_name="Product_Master"),
    }
    return data


def normalize_text(value):
    return str(value).strip().casefold()


def select_outlet(outlets, city, restaurant):
    """Select an outlet by city and outlet name, with a useful error on ambiguity."""
    filtered = outlets[
        outlets["City_Name"].astype(str).map(normalize_text).eq(normalize_text(city))
    ]
    if filtered.empty:
        options = sorted(outlets["City_Name"].dropna().astype(str).unique())
        raise ValueError(
            f"No city matched '{city}'. Available cities: {', '.join(options)}"
        )

    matched = filtered[
        filtered["Outlet_Name"].astype(str).map(normalize_text).eq(normalize_text(restaurant))
    ]
    if matched.empty:
        options = sorted(filtered["Outlet_Name"].dropna().astype(str).unique())
        raise ValueError(
            f"No restaurant/outlet '{restaurant}' found in '{city}'. "
            f"Available outlets in this city: {', '.join(options)}"
        )
    if len(matched) > 1:
        ids = ", ".join(matched["Outlet_ID"].astype(str))
        raise ValueError(f"Restaurant name is ambiguous; matching Outlet_IDs: {ids}")
    return matched.iloc[0]


def select_products(product_master, product_query):
    """Blank product query means all products; otherwise match ID or name."""
    if product_query is None or not str(product_query).strip():
        return product_master.copy()

    q = normalize_text(product_query)
    match = product_master[
        product_master["Product_ID"].astype(str).map(normalize_text).eq(q)
        | product_master["Product_Name"].astype(str).map(normalize_text).eq(q)
    ]
    if match.empty:
        # Allow a unique partial product-name match.
        partial = product_master[
            product_master["Product_Name"].astype(str).str.casefold().str.contains(
                str(product_query).strip(), regex=False, na=False
            )
        ]
        if len(partial) == 1:
            match = partial
        elif len(partial) > 1:
            names = ", ".join(
                f"{r.Product_Name} ({r.Product_ID})" for r in partial.itertuples()
            )
            raise ValueError(f"Product query is ambiguous. Matches: {names}")
    if match.empty:
        names = ", ".join(product_master["Product_Name"].dropna().astype(str).tolist())
        raise ValueError(f"No product matched '{product_query}'. Available: {names}")
    return match


def build_demand_model(features):
    """
    Fit a pooled prototype demand model.
    The target and several inputs in the provided workbook are synthetic estimates.
    """
    df = features.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    target = "Estimated_Units_Sold"

    numeric_features = [
        "Historical_Node_Price_Proxy_INR",
        "Competitor_Price_Mean_INR",
        "Competitor_Price_Median_INR",
        "Competitor_Price_Min_INR",
        "Competitor_Price_Max_INR",
        "Total_Variable_Cost_Per_Unit_INR",
        "Discount_Percent",
        "Promotion_Flag",
        "Stockout_Flag",
        "Stockout_Duration_Minutes",
        "Is_Weekend",
        "Is_Peak_Slot",
        "Local_Event_Flag",
        "Estimated_Lost_Demand_Units",
    ]
    categorical_features = [
        "City_Name",
        "Outlet_ID",
        "Location_Type",
        "Product_ID",
        "Category",
        "Day_of_Week",
        "Time_Slot",
        "Promotion_Type",
    ]
    feature_cols = numeric_features + categorical_features
    for c in numeric_features + [target]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[target] = df[target].fillna(0).clip(lower=0)

    # Ensure all expected columns exist, allowing future workbook revisions.
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan if c in numeric_features else "Unknown"

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:  # compatibility with older scikit-learn
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=True)

    prep = ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", encoder),
            ]), categorical_features),
        ],
        remainder="drop",
    )
    model = Pipeline([
        ("prep", prep),
        ("regressor", RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=3,
            max_features=0.9,
            random_state=42,
            n_jobs=-1,
        )),
    ])
    model.fit(df[feature_cols], df[target])
    return model, feature_cols, numeric_features, categorical_features


def get_slot_rows(features, outlet_id, product_id, day):
    rows = features[
        features["Outlet_ID"].astype(str).eq(str(outlet_id))
        & features["Product_ID"].astype(str).eq(str(product_id))
        & features["Day_of_Week"].astype(str).str.casefold().eq(day.casefold())
    ].copy()
    if rows.empty:
        return rows
    # Use the canonical four slots in the intended order.
    rows["_slot_order"] = rows["Time_Slot"].apply(
        lambda x: SLOTS.index(x) if x in SLOTS else 99
    )
    return rows.sort_values("_slot_order").drop(columns="_slot_order")


def make_candidate_row(slot_row, price, feature_cols):
    """Copy a slot's features and set the candidate price to the same price for every slot."""
    row = slot_row.copy()
    row["Historical_Node_Price_Proxy_INR"] = float(price)
    # The price paid is the candidate daily price. Promo discount is kept as an independent feature.
    return pd.DataFrame([{c: row.get(c, np.nan) for c in feature_cols}])


def get_last_week_summary(slot_rows):
    """Return last-week proxy summary for a weekday from the representative-week workbook."""
    if slot_rows.empty:
        return np.nan, np.nan, np.nan
    units = pd.to_numeric(slot_rows["Estimated_Units_Sold"], errors="coerce").fillna(0)
    prices = pd.to_numeric(slot_rows["Historical_Node_Price_Proxy_INR"], errors="coerce")
    total_units = float(units.sum())
    if total_units > 0:
        weighted_price = float((prices.fillna(0) * units).sum() / total_units)
    else:
        weighted_price = float(prices.mean()) if prices.notna().any() else np.nan
    orders = pd.to_numeric(slot_rows["Estimated_Orders"], errors="coerce").fillna(0).sum()
    return weighted_price, total_units, float(orders)


def recommend_for_product(model, feature_cols, slot_rows, cost_row, day, increment=1.0):
    """Optimize one daily price across all four slots."""
    if slot_rows.empty:
        return None

    # Use the cost sheet's unit cost and illustrative bounds.
    unit_cost = float(cost_row["Total_Variable_Cost_Per_Unit_INR"])
    min_price = float(cost_row["Calculated_Minimum_Allowed_Price_INR"])
    max_price = float(cost_row["Illustrative_Maximum_Allowed_Price_INR"])

    # Last-week weighted price is a sensible anchor, but candidates remain inside approved bounds.
    last_price, last_sales, last_orders = get_last_week_summary(slot_rows)
    if not np.isfinite(last_price):
        last_price = float(cost_row["Menu_Base_Price_INR"])

    # Search prices in ₹1 increments by default. Include rounded boundaries.
    low = max(0.0, math.ceil(min_price / increment) * increment)
    high = math.floor(max_price / increment) * increment
    if high < low:
        low = high = round(max(min_price, min(max_price, last_price)) / increment) * increment
    candidates = np.arange(low, high + increment * 0.5, increment, dtype=float)
    if len(candidates) == 0:
        candidates = np.array([round(last_price / increment) * increment], dtype=float)

    # Price-independent row context is kept per slot; candidate price is changed consistently.
    best = None
    competitor_values = pd.to_numeric(slot_rows["Competitor_Price_Mean_INR"], errors="coerce").dropna()
    competitor_benchmark = float(competitor_values.mean()) if len(competitor_values) else np.nan
    competitor_count = (
        pd.to_numeric(slot_rows["Competitor_Count"], errors="coerce").fillna(0).sum()
        if "Competitor_Count" in slot_rows else 0
    )

    for price in candidates:
        daily_units = 0.0
        slot_predictions = {}
        for _, slot_row in slot_rows.iterrows():
            X = make_candidate_row(slot_row, price, feature_cols)
            # The ML model estimates baseline demand from context, while a transparent
            # provisional elasticity adjusts demand as the candidate price changes.
            # Price is deliberately excluded from model training because this synthetic
            # one-week data cannot identify a causal price-response curve.
            base_pred = max(0.0, float(model.predict(X)[0]))
            reference_price = max(float(last_price), 0.01)
            pred = base_pred * (float(price) / reference_price) ** PRICE_ELASTICITY
            pred = max(0.0, pred)
            slot_predictions[str(slot_row["Time_Slot"])] = pred
            daily_units += pred

        contribution = (price - unit_cost) * daily_units
        if best is None or contribution > best["Predicted_Contribution_INR"]:
            best = {
                "Recommended_Price_INR": round(float(price), 2),
                "Predicted_Daily_Units": round(float(daily_units), 2),
                "Predicted_Contribution_INR": round(float(contribution), 2),
                "Competitor_Benchmark_INR": round(competitor_benchmark, 2) if np.isfinite(competitor_benchmark) else np.nan,
                "Competitor_Count_Summed_Across_Slots": int(competitor_count),
                "Price_Floor_INR": round(min_price, 2),
                "Price_Ceiling_INR": round(max_price, 2),
                "Predicted_Units_Lunch": round(slot_predictions.get(SLOTS[0], np.nan), 2),
                "Predicted_Units_Afternoon": round(slot_predictions.get(SLOTS[1], np.nan), 2),
                "Predicted_Units_Dinner": round(slot_predictions.get(SLOTS[2], np.nan), 2),
                "Predicted_Units_Late_Evening": round(slot_predictions.get(SLOTS[3], np.nan), 2),
                "Last_Week_Price_Proxy_INR": round(last_price, 2) if np.isfinite(last_price) else np.nan,
                "Last_Week_Sales_Proxy_Units": round(last_sales, 2) if np.isfinite(last_sales) else np.nan,
                "Last_Week_Orders_Proxy": round(last_orders, 2) if np.isfinite(last_orders) else np.nan,
                "Unit_Cost_INR": round(unit_cost, 2),
                "Price_Elasticity_Assumption": PRICE_ELASTICITY,
                "Data_Status": "Prototype: synthetic demand data plus provisional price elasticity; validate with actual POS/experiments before use",
            }
    return best


def run_pricing(workbook, city, restaurant, product_query, day, output_csv="pricing_recommendations.csv"):
    data = read_workbook(workbook)
    features = data["features"]
    costs = data["costs"]
    outlets = data["outlets"]
    products = data["products"]

    outlet = select_outlet(outlets, city, restaurant)
    selected_products = select_products(products, product_query)
    model, feature_cols, _, _ = build_demand_model(features)

    all_results = []
    for _, product in selected_products.iterrows():
        slot_rows = get_slot_rows(features, outlet["Outlet_ID"], product["Product_ID"], day)
        if slot_rows.empty:
            warnings.warn(
                f"No weekly slot rows for {outlet['Outlet_ID']} / {product['Product_ID']} / {day}; skipping."
            )
            continue

        cost_rows = costs[
            costs["Outlet_ID"].astype(str).eq(str(outlet["Outlet_ID"]))
            & costs["Product_ID"].astype(str).eq(str(product["Product_ID"]))
        ]
        if cost_rows.empty:
            warnings.warn(f"No cost row for {outlet['Outlet_ID']} / {product['Product_ID']}; skipping.")
            continue

        result = recommend_for_product(model, feature_cols, slot_rows, cost_rows.iloc[0], day)
        if result:
            result.update({
                "City_Name": outlet["City_Name"],
                "Outlet_ID": outlet["Outlet_ID"],
                "Outlet_Name": outlet["Outlet_Name"],
                "Product_ID": product["Product_ID"],
                "Product_Name": product["Product_Name"],
                "Day_of_Week": day,
            })
            all_results.append(result)

    results = pd.DataFrame(all_results)
    if results.empty:
        raise ValueError("No recommendations generated. Check city, outlet, product and weekday inputs.")

    # Organize the key user-requested fields first.
    first = [
        "City_Name", "Outlet_ID", "Outlet_Name", "Product_ID", "Product_Name", "Day_of_Week",
        "Recommended_Price_INR", "Last_Week_Price_Proxy_INR", "Last_Week_Sales_Proxy_Units",
        "Predicted_Daily_Units", "Predicted_Contribution_INR", "Competitor_Benchmark_INR",
        "Competitor_Count_Summed_Across_Slots", "Price_Floor_INR", "Price_Ceiling_INR", "Price_Elasticity_Assumption",
        "Last_Week_Orders_Proxy", "Unit_Cost_INR", "Predicted_Units_Lunch",
        "Predicted_Units_Afternoon", "Predicted_Units_Dinner", "Predicted_Units_Late_Evening",
        "Data_Status",
    ]
    results = results[[c for c in first if c in results.columns] + [c for c in results.columns if c not in first]]
    results.to_csv(output_csv, index=False)
    return results


def main():
    print("\nRESTAURANT DAILY PRICING OPTIMIZATION — PROTOTYPE")
    workbook = input(f"Excel workbook path [{DEFAULT_WORKBOOK}]: ").strip() or DEFAULT_WORKBOOK
    if not Path(workbook).exists():
        # Try alongside this script.
        local = Path(__file__).resolve().parent / workbook
        if local.exists():
            workbook = str(local)
        else:
            raise FileNotFoundError(
                f"Workbook not found: {workbook}\n"
                "Place the workbook next to this script or enter its full path."
            )

    data = read_workbook(workbook)
    cities = sorted(data["outlets"]["City_Name"].dropna().astype(str).unique())
    print("\nAvailable cities:")
    print(", ".join(cities))
    city = input("City: ").strip()

    city_outlets = data["outlets"][
        data["outlets"]["City_Name"].astype(str).map(normalize_text).eq(normalize_text(city))
    ]
    if city_outlets.empty:
        raise ValueError("City not found. Please use one of the available city names above.")
    print("\nAvailable outlets in selected city:")
    print(", ".join(sorted(city_outlets["Outlet_Name"].dropna().astype(str).unique())))
    restaurant = input("Restaurant/outlet name: ").strip()

    print("\nEnter a product name or Product_ID. Leave blank for ALL products.")
    print("Example product names:", ", ".join(data["products"]["Product_Name"].dropna().astype(str).head(8)))
    product_query = input("Product (optional): ").strip()

    print("\nDays:", ", ".join(DAYS))
    day_input = input("Day of week: ").strip()
    day_match = next((d for d in DAYS if d.casefold() == day_input.casefold()), None)
    if day_match is None:
        raise ValueError(f"Invalid weekday '{day_input}'. Use one of: {', '.join(DAYS)}")

    out_name = input("Output CSV path [pricing_recommendations.csv]: ").strip() or "pricing_recommendations.csv"
    results = run_pricing(workbook, city, restaurant, product_query, day_match, out_name)
    print("\nPRICE RECOMMENDATIONS")
    print(results.to_string(index=False))
    print(f"\nSaved results to: {Path(out_name).resolve()}")


if __name__ == "__main__":
    main()
