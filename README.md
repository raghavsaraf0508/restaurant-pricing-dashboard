# Restaurant Pricing Intelligence — Prototype

## Files
- `app.py`: Streamlit dashboard interface.
- `restaurant_daily_pricing_model.py`: existing recommendation model.
- `Restaurant_Pricing_Weekly_28_Slots_Enhanced.xlsx`: synthetic input workbook.
- `requirements.txt`: Python dependencies.

## Run locally
1. Install Python 3.10 or later.
2. Open a terminal in this folder.
3. Run: `python -m pip install -r requirements.txt`
4. Run: `python -m streamlit run app.py`
5. Open the local URL printed in the terminal (usually http://localhost:8501).

## Deploy on Streamlit Community Cloud
1. Upload all four files to a GitHub repository.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the main file path to `app.py`.
4. Deploy.

## Important limitations
This is a presentation prototype. Sales, price history, and competitor benchmarks in the workbook are synthetic or estimated. Last-week values are representative-week proxies. The elasticity assumption is provisional (-1.5), not causally estimated. Do not use recommendations to change live menu prices without validation.
