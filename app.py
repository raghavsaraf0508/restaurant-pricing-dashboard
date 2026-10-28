from pathlib import Path
import io
import pandas as pd
import streamlit as st

from restaurant_daily_pricing_model import (
    DAYS,
    read_workbook,
    run_pricing,
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_WORKBOOK = APP_DIR / "Restaurant_Pricing_Weekly_28_Slots_Enhanced.xlsx"

st.set_page_config(
    page_title="Restaurant Pricing Intelligence",
    page_icon="🍽️",
    layout="wide",
)

st.title("🍽️ Restaurant Pricing Intelligence")
st.caption(
    "Prototype dashboard for daily menu-price recommendations using synthetic data. "
    "Recommendations are for demonstration only and must not be applied to live menus."
)

@st.cache_data(show_spinner=False)
def load_reference_data(workbook_path: str):
    data = read_workbook(workbook_path)
    outlets = data["outlets"].copy()
    products = data["products"].copy()
    return outlets, products

if not DEFAULT_WORKBOOK.exists():
    st.error(
        "Workbook not found. Place "
        "'Restaurant_Pricing_Weekly_28_Slots_Enhanced.xlsx' in the same folder as app.py."
    )
    st.stop()

try:
    outlets, products = load_reference_data(str(DEFAULT_WORKBOOK))
except Exception as exc:
    st.error(f"Could not load the workbook: {exc}")
    st.stop()

with st.sidebar:
    st.header("Recommendation inputs")
    city_options = sorted(outlets["City_Name"].dropna().astype(str).unique().tolist())
    city = st.selectbox("City", city_options)

    city_outlets = outlets[
        outlets["City_Name"].astype(str).str.casefold() == city.casefold()
    ]
    outlet_options = sorted(city_outlets["Outlet_Name"].dropna().astype(str).unique().tolist())
    restaurant = st.selectbox("Restaurant / outlet", outlet_options)

    product_choice = st.selectbox(
        "Product",
        ["All products"] + sorted(products["Product_Name"].dropna().astype(str).unique().tolist()),
    )
    day = st.selectbox("Day of week", DAYS)

    run_button = st.button("Recommend price", type="primary", use_container_width=True)

st.info(
    "Model note: sales, price history and competitor benchmarks in this workbook are synthetic "
    "or estimated. The price elasticity of -1.5 is a provisional assumption, not a causal estimate."
)

if run_button:
    product_query = "" if product_choice == "All products" else product_choice
    with st.spinner("Training the prototype model and calculating recommendations..."):
        try:
            results = run_pricing(
                workbook=str(DEFAULT_WORKBOOK),
                city=city,
                restaurant=restaurant,
                product_query=product_query,
                day=day,
                output_csv=str(APP_DIR / "pricing_recommendations.csv"),
            )
            st.session_state["pricing_results"] = results
            st.session_state["pricing_inputs"] = {
                "city": city,
                "restaurant": restaurant,
                "product": product_choice,
                "day": day,
            }
        except Exception as exc:
            st.error(f"Could not generate recommendations: {exc}")

results = st.session_state.get("pricing_results")
inputs = st.session_state.get("pricing_inputs")

if results is not None and inputs is not None:
    st.subheader(f"Results for {inputs['restaurant']} · {inputs['day']}")
    st.caption(f"City: {inputs['city']} | Product selection: {inputs['product']}")

    if len(results) == 1:
        r = results.iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Recommended daily price", f"₹{r['Recommended_Price_INR']:,.0f}")
        c2.metric("Last-week price proxy", f"₹{r['Last_Week_Price_Proxy_INR']:,.2f}")
        c3.metric("Last-week sales proxy", f"{r['Last_Week_Sales_Proxy_Units']:,.0f} units")
        c4.metric("Predicted daily units", f"{r['Predicted_Daily_Units']:,.0f}")
        c5.metric("Predicted contribution", f"₹{r['Predicted_Contribution_INR']:,.0f}")

        st.subheader("Price and operating constraints")
        a, b, c = st.columns(3)
        a.metric("Competitor benchmark", f"₹{r['Competitor_Benchmark_INR']:,.2f}" if pd.notna(r["Competitor_Benchmark_INR"]) else "Not available")
        b.metric("Price floor", f"₹{r['Price_Floor_INR']:,.2f}")
        c.metric("Price ceiling", f"₹{r['Price_Ceiling_INR']:,.2f}")

        slot_df = pd.DataFrame({
            "Time slot": ["Lunch", "Afternoon", "Dinner", "Late evening"],
            "Predicted units": [
                r["Predicted_Units_Lunch"],
                r["Predicted_Units_Afternoon"],
                r["Predicted_Units_Dinner"],
                r["Predicted_Units_Late_Evening"],
            ],
        })
        st.subheader("Predicted demand by time slot")
        st.bar_chart(slot_df.set_index("Time slot"))

        compare_df = pd.DataFrame({
            "Price type": ["Last-week price proxy", "Recommended price"],
            "Price (₹)": [r["Last_Week_Price_Proxy_INR"], r["Recommended_Price_INR"]],
        })
        st.subheader("Price comparison")
        st.bar_chart(compare_df.set_index("Price type"))
    else:
        st.subheader("Product-level recommendations")
        display_cols = [
            "Product_Name",
            "Recommended_Price_INR",
            "Last_Week_Price_Proxy_INR",
            "Last_Week_Sales_Proxy_Units",
            "Predicted_Daily_Units",
            "Predicted_Contribution_INR",
            "Competitor_Benchmark_INR",
            "Price_Floor_INR",
            "Price_Ceiling_INR",
        ]
        st.dataframe(
            results[display_cols].rename(columns={
                "Product_Name": "Product",
                "Recommended_Price_INR": "Recommended price (₹)",
                "Last_Week_Price_Proxy_INR": "Last-week price proxy (₹)",
                "Last_Week_Sales_Proxy_Units": "Last-week sales proxy (units)",
                "Predicted_Daily_Units": "Predicted daily units",
                "Predicted_Contribution_INR": "Predicted contribution (₹)",
                "Competitor_Benchmark_INR": "Competitor benchmark (₹)",
                "Price_Floor_INR": "Price floor (₹)",
                "Price_Ceiling_INR": "Price ceiling (₹)",
            }),
            use_container_width=True,
            hide_index=True,
        )
        chart_df = results[["Product_Name", "Last_Week_Price_Proxy_INR", "Recommended_Price_INR"]].set_index("Product_Name")
        chart_df.columns = ["Last-week price proxy (₹)", "Recommended price (₹)"]
        st.subheader("Price comparison by product")
        st.bar_chart(chart_df)
        sales_df = results[["Product_Name", "Last_Week_Sales_Proxy_Units", "Predicted_Daily_Units"]].set_index("Product_Name")
        sales_df.columns = ["Last-week sales proxy", "Predicted daily units"]
        st.subheader("Sales comparison by product")
        st.bar_chart(sales_df)

    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download recommendations as CSV",
        data=csv_bytes,
        file_name="pricing_recommendations.csv",
        mime="text/csv",
    )

    with st.expander("Model assumptions and limitations"):
        st.markdown(
            """
            - One recommended price is used for the entire business day and all four time slots.
            - The workbook uses synthetic or estimated sales and competitor-price values.
            - Last-week values are representative-week proxies, not verified dated POS history.
            - Price elasticity is set provisionally to **-1.5**; it has not been causally estimated.
            - Predicted contribution is a prototype estimate based on the model's unit-cost fields.
            - Review and validate with real POS data and controlled price tests before any operational use.
            """
        )
else:
    st.markdown("### How to use the dashboard")
    st.write("1. Select a city, restaurant, product and weekday in the sidebar.")
    st.write("2. Click **Recommend price**.")
    st.write("3. Review the daily price recommendation, demand estimates, contribution and competitor benchmark.")
    st.write("4. Download the results as a CSV if needed.")
