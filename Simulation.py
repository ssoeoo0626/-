import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(page_title="포트폴리오 CAGR 시뮬레이터", layout="wide")
st.title("포트폴리오 CAGR 미래자산 시뮬레이터")

st.caption(
    "티커별 과거 CAGR을 계산하고, 초기 투자금 + 매월 적립금 기준으로 n일 후 미래자산을 추정합니다."
)

# -----------------------------
# 유틸
# -----------------------------
@st.cache_data(ttl=60 * 60 * 6)
def load_price_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    auto_adjust=True:
    분할/배당 반영 가격 기준 계산
    """
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
        raise ValueError("CAGR 계산값이 올바르지 않습니다.")
    return (end_price / start_price) ** (1 / years) - 1


def annual_to_monthly_return(cagr: float) -> float:
    return (1 + cagr) ** (1 / 12) - 1


def annual_to_daily_return(cagr: float, trading_days: int = 252) -> float:
    return (1 + cagr) ** (1 / trading_days) - 1


def future_value_with_monthly_contribution(
    initial_investment: float,
    monthly_contribution: float,
    annual_return: float,
    n_days: int,
):
    """
    n일 후를 월 단위 기준으로 근사 계산
    - 초기 투자금: 월복리
    - 매월 적립금: 월말 납입 가정
    """
    months = n_days / 30.4375
    full_months = int(months)
    remaining_fraction = months - full_months

    monthly_r = annual_to_monthly_return(annual_return)

    # 초기 투자금 성장
    fv_initial = initial_investment * ((1 + monthly_r) ** months)

    # 적립식 미래가치
    if abs(monthly_r) < 1e-12:
        fv_contrib = monthly_contribution * full_months
        fv_contrib *= (1 + monthly_r) ** remaining_fraction
    else:
        fv_contrib = monthly_contribution * (((1 + monthly_r) ** full_months - 1) / monthly_r)
        fv_contrib *= (1 + monthly_r) ** remaining_fraction

    total_invested = initial_investment + monthly_contribution * full_months
    future_value = fv_initial + fv_contrib
    profit = future_value - total_invested

    return {
        "months_equiv": months,
        "future_value": future_value,
        "total_invested": total_invested,
        "profit": profit,
        "monthly_return": monthly_r,
    }


# -----------------------------
# 기본 입력표
# -----------------------------
default_df = pd.DataFrame(
    [
        {
            "Ticker": "AAPL",
            "CAGR_기간(년)": 10,
            "예측기간(n일)": 252,
            "초기투자금(원)": 10000000,
            "월적립금(원)": 500000,
        }
    ]
)

st.subheader("자산 입력")
edited_df = st.data_editor(
    default_df,
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Ticker": st.column_config.TextColumn("Ticker"),
        "CAGR_기간(년)": st.column_config.NumberColumn("CAGR 기간(년)", min_value=1, max_value=30, step=1),
        "예측기간(n일)": st.column_config.NumberColumn("예측기간(n일)", min_value=1, max_value=10000, step=1),
        "초기투자금(원)": st.column_config.NumberColumn("초기 투자금(원)", min_value=0, step=100000),
        "월적립금(원)": st.column_config.NumberColumn("월 적립금(원)", min_value=0, step=10000),
    },
)

run = st.button("시뮬레이션 실행", type="primary")

# -----------------------------
# 계산
# -----------------------------
if run:
    if edited_df.empty:
        st.warning("최소 1개 자산은 입력해줘.")
        st.stop()

    results = []
    errors = []
    today = datetime.today()

    progress = st.progress(0)
    total_rows = len(edited_df)

    for i, row in edited_df.iterrows():
        try:
            ticker = str(row["Ticker"]).strip().upper()
            cagr_years = int(row["CAGR_기간(년)"])
            n_days = int(row["예측기간(n일)"])
            initial_investment = float(row["초기투자금(원)"])
            monthly_contribution = float(row["월적립금(원)"])

            if not ticker:
                raise ValueError("Ticker가 비어 있습니다.")

            start_date = today - timedelta(days=int(cagr_years * 365.25) + 20)
            df = load_price_data(ticker, start_date, today)

            if df.empty:
                raise ValueError("가격 데이터를 불러오지 못했습니다.")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]

            if "Close" not in df.columns:
                raise ValueError("Close 데이터가 없습니다.")

            price_df = df[["Close"]].dropna().copy()
            if len(price_df) < 2:
                raise ValueError("데이터가 부족합니다.")

            target_start = today - timedelta(days=int(cagr_years * 365.25))
            valid_start_rows = price_df[price_df.index >= pd.Timestamp(target_start)]

            if valid_start_rows.empty:
                actual_start_price = float(price_df["Close"].iloc[0])
                actual_start_date = pd.to_datetime(price_df.index[0]).to_pydatetime()
            else:
                actual_start_price = float(valid_start_rows["Close"].iloc[0])
                actual_start_date = pd.to_datetime(valid_start_rows.index[0]).to_pydatetime()

            current_price = float(price_df["Close"].iloc[-1])
            actual_end_date = pd.to_datetime(price_df.index[-1]).to_pydatetime()
            actual_years = (actual_end_date - actual_start_date).days / 365.25

            cagr = calculate_cagr(actual_start_price, current_price, actual_years)
            daily_return = annual_to_daily_return(cagr)
            expected_return_n_days = (1 + daily_return) ** n_days - 1
            expected_price_n_days = current_price * (1 + expected_return_n_days)

            fv = future_value_with_monthly_contribution(
                initial_investment=initial_investment,
                monthly_contribution=monthly_contribution,
                annual_return=cagr,
                n_days=n_days,
            )

            results.append(
                {
                    "Ticker": ticker,
                    "CAGR 기간(년)": cagr_years,
                    "예측기간(n일)": n_days,
                    "시작일": actual_start_date.strftime("%Y-%m-%d"),
                    "종료일": actual_end_date.strftime("%Y-%m-%d"),
                    "시작가": actual_start_price,
                    "현재가": current_price,
                    "CAGR(%)": cagr * 100,
                    "n일 기대수익률(%)": expected_return_n_days * 100,
                    "n일 후 예상가": expected_price_n_days,
                    "초기투자금(원)": initial_investment,
                    "월적립금(원)": monthly_contribution,
                    "총투입원금(원)": fv["total_invested"],
                    "예상미래자산(원)": fv["future_value"],
                    "예상수익(원)": fv["profit"],
                    "월복리 환산수익률(%)": fv["monthly_return"] * 100,
                }
            )

        except Exception as e:
            errors.append(f"{i+1}행 오류: {e}")

        progress.progress((i + 1) / total_rows)

    progress.empty()

    if errors:
        st.error("일부 자산은 계산되지 않았어.")
        for err in errors:
            st.write(f"- {err}")

    if not results:
        st.stop()

    result_df = pd.DataFrame(results)

    # -----------------------------
    # 요약 카드
    # -----------------------------
    total_invested = result_df["총투입원금(원)"].sum()
    total_future_value = result_df["예상미래자산(원)"].sum()
    total_profit = result_df["예상수익(원)"].sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("총 투입원금", f"{total_invested:,.0f}원")
    c2.metric("총 예상 미래자산", f"{total_future_value:,.0f}원")
    c3.metric("총 예상 수익", f"{total_profit:,.0f}원")

    st.divider()

    # -----------------------------
    # 자산별 상세 결과
    # -----------------------------
    st.subheader("자산별 결과")
    display_cols = [
        "Ticker",
        "CAGR 기간(년)",
        "예측기간(n일)",
        "현재가",
        "CAGR(%)",
        "n일 기대수익률(%)",
        "n일 후 예상가",
        "초기투자금(원)",
        "월적립금(원)",
        "총투입원금(원)",
        "예상미래자산(원)",
        "예상수익(원)",
    ]
    st.dataframe(
        result_df[display_cols].style.format(
            {
                "현재가": "{:,.2f}",
                "CAGR(%)": "{:,.2f}",
                "n일 기대수익률(%)": "{:,.2f}",
                "n일 후 예상가": "{:,.2f}",
                "초기투자금(원)": "{:,.0f}",
                "월적립금(원)": "{:,.0f}",
                "총투입원금(원)": "{:,.0f}",
                "예상미래자산(원)": "{:,.0f}",
                "예상수익(원)": "{:,.0f}",
            }
        ),
        use_container_width=True,
    )

    st.divider()

    # -----------------------------
    # 그래프 1: 자산별 미래자산
    # -----------------------------
    st.subheader("자산별 예상 미래자산")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(result_df["Ticker"], result_df["예상미래자산(원)"])
    ax1.set_ylabel("원")
    ax1.set_xlabel("Ticker")
    ax1.set_title("자산별 예상 미래자산")
    st.pyplot(fig1)

    # -----------------------------
    # 그래프 2: 총투입원금 vs 총미래자산
    # -----------------------------
    st.subheader("총계 비교")
    compare_df = pd.DataFrame(
        {
            "구분": ["총투입원금", "총예상미래자산"],
            "금액": [total_invested, total_future_value],
        }
    )
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.bar(compare_df["구분"], compare_df["금액"])
    ax2.set_ylabel("원")
    ax2.set_title("총투입원금 vs 총예상미래자산")
    st.pyplot(fig2)

    # -----------------------------
    # CSV 다운로드
    # -----------------------------
    csv = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "결과 CSV 다운로드",
        data=csv,
        file_name="portfolio_cagr_simulation_result.csv",
        mime="text/csv",
    )
