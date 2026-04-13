from datetime import datetime, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio CAGR Simulator", layout="wide")

today = datetime.today()
today_str = today.strftime("%Y-%m-%d")

st.title("Portfolio CAGR Simulator")
st.caption(f"As of {today_str}")
st.caption(
    "Enter a ticker and a lookback period. The app automatically calculates CAGR from price history and projects future portfolio value."
)
st.caption(
    "Examples: AAPL, MSFT, SPY, 005930.KS, 000660.KS, 035420.KS, 091990.KQ"
)


@st.cache_data(ttl=60 * 60 * 6)
def load_price_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    return df


def calculate_cagr(start_price: float, end_price: float, years: float) -> float:
    if start_price <= 0 or end_price <= 0 or years <= 0:
        raise ValueError("Invalid values for CAGR calculation.")
    return (end_price / start_price) ** (1 / years) - 1


def annual_to_daily_return(cagr: float, trading_days: int = 252) -> float:
    return (1 + cagr) ** (1 / trading_days) - 1


def annual_to_monthly_return(cagr: float) -> float:
    return (1 + cagr) ** (1 / 12) - 1


def future_value_with_monthly_investment(
    initial_investment: float,
    monthly_investment_amount: float,
    annual_return: float,
    projection_days: int,
):
    months = projection_days / 30.4375
    full_months = int(months)
    remaining_fraction = months - full_months

    monthly_r = annual_to_monthly_return(annual_return)

    fv_initial = initial_investment * ((1 + monthly_r) ** months)

    if abs(monthly_r) < 1e-12:
        fv_monthly = monthly_investment_amount * full_months
        fv_monthly *= (1 + monthly_r) ** remaining_fraction
    else:
        fv_monthly = monthly_investment_amount * (((1 + monthly_r) ** full_months - 1) / monthly_r)
        fv_monthly *= (1 + monthly_r) ** remaining_fraction

    total_invested = initial_investment + monthly_investment_amount * full_months
    future_value = fv_initial + fv_monthly
    profit = future_value - total_invested

    return {
        "months_equiv": months,
        "future_value": future_value,
        "total_invested": total_invested,
        "profit": profit,
        "monthly_return": monthly_r,
    }


default_df = pd.DataFrame(
    [
        {
            "Ticker": "AAPL",
            "CAGR Lookback (Years)": 10,
            "Projection (Days)": 252,
            "Initial Investment": 10000000,
            "Monthly Investment Amount": 500000,
        }
    ]
)

st.subheader("Assets")
edited_df = st.data_editor(
    default_df,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker"),
        "CAGR Lookback (Years)": st.column_config.NumberColumn(
            "CAGR Lookback (Years)", min_value=1, max_value=30, step=1
        ),
        "Projection (Days)": st.column_config.NumberColumn(
            "Projection (Days)", min_value=1, max_value=10000, step=1
        ),
        "Initial Investment": st.column_config.NumberColumn(
            "Initial Investment", min_value=0, step=100000
        ),
        "Monthly Investment Amount": st.column_config.NumberColumn(
            "Monthly Investment Amount", min_value=0, step=10000
        ),
    },
)

run = st.button("Run Simulation", type="primary")

if run:
    if edited_df.empty:
        st.warning("Please add at least one asset.")
        st.stop()

    results = []
    errors = []

    progress = st.progress(0)
    total_rows = len(edited_df)

    for i, row in edited_df.iterrows():
        try:
            ticker = str(row["Ticker"]).strip().upper()
            lookback_years = int(row["CAGR Lookback (Years)"])
            projection_days = int(row["Projection (Days)"])
            initial_investment = float(row["Initial Investment"])
            monthly_investment_amount = float(row["Monthly Investment Amount"])

            if not ticker:
                raise ValueError("Ticker is empty.")

            start_date = today - timedelta(days=int(lookback_years * 365.25) + 20)
            df = load_price_data(ticker, start_date, today)

            if df.empty:
                raise ValueError("No price data found.")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            if "Close" not in df.columns:
                raise ValueError("Close column not found.")

            price_df = df[["Close"]].dropna().copy()
            if len(price_df) < 2:
                raise ValueError("Not enough data.")

            target_start = today - timedelta(days=int(lookback_years * 365.25))
            valid_start_rows = price_df[price_df.index >= pd.Timestamp(target_start)]

            if valid_start_rows.empty:
                start_price = float(price_df["Close"].iloc[0])
                start_used = pd.to_datetime(price_df.index[0]).to_pydatetime()
            else:
                start_price = float(valid_start_rows["Close"].iloc[0])
                start_used = pd.to_datetime(valid_start_rows.index[0]).to_pydatetime()

            current_price = float(price_df["Close"].iloc[-1])
            end_used = pd.to_datetime(price_df.index[-1]).to_pydatetime()
            actual_years = (end_used - start_used).days / 365.25

            cagr = calculate_cagr(start_price, current_price, actual_years)
            daily_return = annual_to_daily_return(cagr)
            projected_return = (1 + daily_return) ** projection_days - 1
            projected_price = current_price * (1 + projected_return)

            fv = future_value_with_monthly_investment(
                initial_investment=initial_investment,
                monthly_investment_amount=monthly_investment_amount,
                annual_return=cagr,
                projection_days=projection_days,
            )

            results.append(
                {
                    "Ticker": ticker,
                    "Lookback Years": lookback_years,
                    "Projection Days": projection_days,
                    "Start Date Used": start_used.strftime("%Y-%m-%d"),
                    "End Date Used": end_used.strftime("%Y-%m-%d"),
                    "Start Price": start_price,
                    "Current Price": current_price,
                    "CAGR (%)": cagr * 100,
                    "Projected Return (%)": projected_return * 100,
                    "Projected Price": projected_price,
                    "Initial Investment": initial_investment,
                    "Monthly Investment Amount": monthly_investment_amount,
                    "Total Invested": fv["total_invested"],
                    "Projected Portfolio Value": fv["future_value"],
                    "Projected Profit": fv["profit"],
                }
            )

        except Exception as e:
            errors.append(f"Row {i + 1}: {e}")

        progress.progress((i + 1) / total_rows)

    progress.empty()

    if errors:
        st.error("Some rows could not be calculated.")
        for err in errors:
            st.write(f"- {err}")

    if not results:
        st.stop()

    result_df = pd.DataFrame(results)

    total_invested = result_df["Total Invested"].sum()
    total_future_value = result_df["Projected Portfolio Value"].sum()
    total_profit = result_df["Projected Profit"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Invested", f"{total_invested:,.0f}")
    c2.metric("Projected Portfolio Value", f"{total_future_value:,.0f}")
    c3.metric("Projected Profit", f"{total_profit:,.0f}")

    st.subheader("Asset Results")
    st.dataframe(
        result_df.style.format(
            {
                "Start Price": "{:,.2f}",
                "Current Price": "{:,.2f}",
                "CAGR (%)": "{:,.2f}",
                "Projected Return (%)": "{:,.2f}",
                "Projected Price": "{:,.2f}",
                "Initial Investment": "{:,.0f}",
                "Monthly Investment Amount": "{:,.0f}",
                "Total Invested": "{:,.0f}",
                "Projected Portfolio Value": "{:,.0f}",
                "Projected Profit": "{:,.0f}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Projected Portfolio Value by Ticker")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(result_df["Ticker"], result_df["Projected Portfolio Value"])
    ax1.set_xlabel("Ticker")
    ax1.set_ylabel("Value")
    ax1.set_title("Projected Portfolio Value by Ticker")
    st.pyplot(fig1)

    st.subheader("Total Invested vs Projected Value")
    compare_df = pd.DataFrame(
        {
            "Category": ["Total Invested", "Projected Value"],
            "Amount": [total_invested, total_future_value],
        }
    )
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(compare_df["Category"], compare_df["Amount"])
    ax2.set_ylabel("Amount")
    ax2.set_title("Total Invested vs Projected Value")
    st.pyplot(fig2)
