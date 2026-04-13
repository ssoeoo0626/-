from datetime import datetime, timedelta

import numpy as np
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


def normalize_ticker(ticker: str) -> str:
    ticker = str(ticker).strip().upper()
    if ticker.isdigit():
        if len(ticker) == 6:
            return f"{ticker}.KS"
    return ticker


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


def get_period_cagr(price_df: pd.DataFrame, end_date: datetime, years: int):
    target_start = end_date - timedelta(days=int(years * 365.25))
    valid_rows = price_df[price_df.index >= pd.Timestamp(target_start)]

    if valid_rows.empty:
        start_price = float(price_df["Close"].iloc[0])
        start_used = pd.to_datetime(price_df.index[0]).to_pydatetime()
    else:
        start_price = float(valid_rows["Close"].iloc[0])
        start_used = pd.to_datetime(valid_rows.index[0]).to_pydatetime()

    end_price = float(price_df["Close"].iloc[-1])
    end_used = pd.to_datetime(price_df.index[-1]).to_pydatetime()
    actual_years = (end_used - start_used).days / 365.25

    if actual_years <= 0:
        return np.nan

    return calculate_cagr(start_price, end_price, actual_years)


def monte_carlo_price_paths(
    current_price: float,
    daily_returns: pd.Series,
    projection_days: int,
    n_sims: int = 1000,
):
    if len(daily_returns) < 30:
        raise ValueError("Not enough daily return history for Monte Carlo simulation.")

    mu = daily_returns.mean()
    sigma = daily_returns.std()

    random_returns = np.random.normal(
        loc=mu,
        scale=sigma,
        size=(projection_days, n_sims)
    )

    growth_paths = np.cumprod(1 + random_returns, axis=0)
    final_prices = current_price * growth_paths[-1]

    return {
        "mu": mu,
        "sigma": sigma,
        "final_prices": final_prices,
    }


def monte_carlo_portfolio_values(
    initial_investment: float,
    monthly_investment_amount: float,
    final_price_paths: np.ndarray,
    current_price: float,
    projection_days: int,
):
    months = projection_days / 30.4375
    full_months = int(months)

    price_ratio = final_price_paths / current_price

    future_initial_values = initial_investment * price_ratio

    # 월 적립금은 평균적으로 절반 기간 정도 투자되었다고 보는 근사
    # 조금 단순화했지만 UI용으로는 충분히 직관적
    avg_months_invested = max(full_months / 2, 0)
    monthly_growth_factor = price_ratio ** (avg_months_invested / max(months, 1e-9))
    future_monthly_values = monthly_investment_amount * full_months * monthly_growth_factor

    total_invested = initial_investment + monthly_investment_amount * full_months
    future_values = future_initial_values + future_monthly_values

    return {
        "future_values": future_values,
        "total_invested": total_invested,
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

left_top, right_top = st.columns([3, 1])

with left_top:
    st.subheader("Assets")
with right_top:
    n_sims = st.number_input(
        "Monte Carlo Simulations",
        min_value=100,
        max_value=10000,
        value=1000,
        step=100,
    )

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
    mc_detail = {}

    progress = st.progress(0)
    total_rows = len(edited_df)

    for i, row in edited_df.iterrows():
        try:
            ticker = normalize_ticker(row["Ticker"])
            lookback_years = int(row["CAGR Lookback (Years)"])
            projection_days = int(row["Projection (Days)"])
            initial_investment = float(row["Initial Investment"])
            monthly_investment_amount = float(row["Monthly Investment Amount"])

            if not ticker:
                raise ValueError("Ticker is empty.")

            longest_years = max(10, lookback_years)
            start_date = today - timedelta(days=int(longest_years * 365.25) + 40)
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
            cagr_3y = get_period_cagr(price_df, today, 3)
            cagr_5y = get_period_cagr(price_df, today, 5)
            cagr_10y = get_period_cagr(price_df, today, 10)

            daily_return = annual_to_daily_return(cagr)
            projected_return = (1 + daily_return) ** projection_days - 1
            projected_price = current_price * (1 + projected_return)

            fv = future_value_with_monthly_investment(
                initial_investment=initial_investment,
                monthly_investment_amount=monthly_investment_amount,
                annual_return=cagr,
                projection_days=projection_days,
            )

            historical_daily_returns = price_df["Close"].pct_change().dropna()

            mc = monte_carlo_price_paths(
                current_price=current_price,
                daily_returns=historical_daily_returns,
                projection_days=projection_days,
                n_sims=n_sims,
            )

            mc_portfolio = monte_carlo_portfolio_values(
                initial_investment=initial_investment,
                monthly_investment_amount=monthly_investment_amount,
                final_price_paths=mc["final_prices"],
                current_price=current_price,
                projection_days=projection_days,
            )

            mc_values = mc_portfolio["future_values"]

            mc_mean = float(np.mean(mc_values))
            mc_median = float(np.median(mc_values))
            mc_p10 = float(np.percentile(mc_values, 10))
            mc_p90 = float(np.percentile(mc_values, 90))

            results.append(
                {
                    "Ticker": ticker,
                    "Lookback Years": lookback_years,
                    "Projection Days": projection_days,
                    "Start Date Used": start_used.strftime("%Y-%m-%d"),
                    "End Date Used": end_used.strftime("%Y-%m-%d"),
                    "Start Price": start_price,
                    "Current Price": current_price,
                    "Selected CAGR (%)": cagr * 100,
                    "3Y CAGR (%)": cagr_3y * 100 if pd.notna(cagr_3y) else np.nan,
                    "5Y CAGR (%)": cagr_5y * 100 if pd.notna(cagr_5y) else np.nan,
                    "10Y CAGR (%)": cagr_10y * 100 if pd.notna(cagr_10y) else np.nan,
                    "Projected Return (%)": projected_return * 100,
                    "Projected Price": projected_price,
                    "Initial Investment": initial_investment,
                    "Monthly Investment Amount": monthly_investment_amount,
                    "Total Principal Invested": fv["total_invested"],
                    "Projected Future Value": fv["future_value"],
                    "Projected Profit": fv["profit"],
                    "MC Mean Value": mc_mean,
                    "MC Median Value": mc_median,
                    "MC P10 Value": mc_p10,
                    "MC P90 Value": mc_p90,
                    "MC Daily Mean Return (%)": mc["mu"] * 100,
                    "MC Daily Volatility (%)": mc["sigma"] * 100,
                }
            )

            mc_detail[ticker] = {
                "distribution": mc_values,
                "projected_value": fv["future_value"],
                "principal": fv["total_invested"],
            }

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

    total_principal = result_df["Total Principal Invested"].sum()
    total_future_value = result_df["Projected Future Value"].sum()
    total_profit = result_df["Projected Profit"].sum()

    st.subheader("Portfolio Summary")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Principal Invested", f"{total_principal:,.0f}")
    c2.metric("Projected Future Value", f"{total_future_value:,.0f}")
    c3.metric("Projected Profit", f"{total_profit:,.0f}")

    st.subheader("Asset Results")
    st.dataframe(
        result_df.style.format(
            {
                "Start Price": "{:,.2f}",
                "Current Price": "{:,.2f}",
                "Selected CAGR (%)": "{:,.2f}",
                "3Y CAGR (%)": "{:,.2f}",
                "5Y CAGR (%)": "{:,.2f}",
                "10Y CAGR (%)": "{:,.2f}",
                "Projected Return (%)": "{:,.2f}",
                "Projected Price": "{:,.2f}",
                "Initial Investment": "{:,.0f}",
                "Monthly Investment Amount": "{:,.0f}",
                "Total Principal Invested": "{:,.0f}",
                "Projected Future Value": "{:,.0f}",
                "Projected Profit": "{:,.0f}",
                "MC Mean Value": "{:,.0f}",
                "MC Median Value": "{:,.0f}",
                "MC P10 Value": "{:,.0f}",
                "MC P90 Value": "{:,.0f}",
                "MC Daily Mean Return (%)": "{:,.4f}",
                "MC Daily Volatility (%)": "{:,.4f}",
            }
        ),
        use_container_width=True,
    )

    st.subheader("Projected Future Value by Ticker")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(result_df["Ticker"], result_df["Projected Future Value"])
    ax1.set_xlabel("Ticker")
    ax1.set_ylabel("Value")
    ax1.set_title("Projected Future Value by Ticker")
    st.pyplot(fig1)

    st.subheader("Total Principal vs Projected Future Value")
    compare_df = pd.DataFrame(
        {
            "Category": ["Total Principal", "Projected Future Value"],
            "Amount": [total_principal, total_future_value],
        }
    )
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(compare_df["Category"], compare_df["Amount"])
    ax2.set_ylabel("Amount")
    ax2.set_title("Total Principal vs Projected Future Value")
    st.pyplot(fig2)

    st.subheader("Monte Carlo Summary by Ticker")
    mc_summary_cols = [
        "Ticker",
        "MC Mean Value",
        "MC Median Value",
        "MC P10 Value",
        "MC P90 Value",
        "MC Daily Mean Return (%)",
        "MC Daily Volatility (%)",
    ]
    st.dataframe(
        result_df[mc_summary_cols].style.format(
            {
                "MC Mean Value": "{:,.0f}",
                "MC Median Value": "{:,.0f}",
                "MC P10 Value": "{:,.0f}",
                "MC P90 Value": "{:,.0f}",
                "MC Daily Mean Return (%)": "{:,.4f}",
                "MC Daily Volatility (%)": "{:,.4f}",
            }
        ),
        use_container_width=True,
    )

    selected_ticker = st.selectbox(
        "Select a ticker for Monte Carlo distribution",
        options=result_df["Ticker"].tolist(),
    )

    if selected_ticker:
        dist = mc_detail[selected_ticker]["distribution"]
        projected_value = mc_detail[selected_ticker]["projected_value"]
        principal = mc_detail[selected_ticker]["principal"]

        st.subheader(f"Monte Carlo Distribution: {selected_ticker}")
        fig3, ax3 = plt.subplots(figsize=(10, 5))
        ax3.hist(dist, bins=40)
        ax3.axvline(principal, linestyle="--", label="Principal Invested")
        ax3.axvline(projected_value, linestyle="--", label="CAGR-Based Projected Value")
        ax3.set_xlabel("Future Portfolio Value")
        ax3.set_ylabel("Frequency")
        ax3.set_title(f"Monte Carlo Distribution - {selected_ticker}")
        ax3.legend()
        st.pyplot(fig3)
