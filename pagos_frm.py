import io
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from utils import (
    aggregate_monthly,
    calculate_correlations,
    calculate_descriptive_stats,
    calculate_monthly_stats,
    calculate_seasonal_indices,
    calculate_yearly_stats,
    clean_dataframe,
    create_correlation_heatmap,
    create_eda_charts,
    create_monthly_pattern_chart,
    create_seasonal_decomposition_chart,
    create_trend_analysis,
    create_year_growth_chart,
    detect_outliers_iqr,
    find_amount_column,
    get_dataframe_info,
    process_date_columns,
    test_normality,
)

try:
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import (
        ExtraTreesRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    HAS_ML_LIBS = True
except ImportError:
    HAS_ML_LIBS = False


def show_pagos_frm_view():
    """Display PAGOS_FRM view for data analysis and download."""
    uploaded_file = st.file_uploader(
        "Upload Excel file", type=["xlsx", "xls"], key="pagos_frm_uploader"
    )

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)

        df_clean = clean_dataframe(df)

        rename_dict = {
            "CLIENTE": "CLIENTE",
            "CONTRATO": "CONTRATO",
            "MANDANTE": "MANDANTE",
            "ESTADO": "ESTADO",
            "ESTADO 2": "ESTADO2",
            "FECHA DE PAGO": "FECHA_PAGO",
            "MONTO PAGADO": "MONTO",
            "TIPO DE PAGO": "TIPO_PAGO",
            "Saldo capital": "SALDO_CAPITAL",
        }
        df_clean = df_clean.rename(columns=rename_dict)

        df_clean, date_columns = process_date_columns(
            df_clean, known_date_col="FECHA_PAGO"
        )

        info = get_dataframe_info(df_clean)

        st.write("**Column Information:**")
        st.text(info["info"])

        st.write("**Data Types:**")
        st.write(info["dtypes"])

        st.write("**Missing Values:**")
        missing_display = info["missing"][info["missing"]["Missing Count"] > 0]
        if len(missing_display) > 0:
            st.dataframe(missing_display)
        else:
            st.write("No missing values found.")

        if "year_month" in df_clean.columns and date_columns:
            st.write("---")

            amount_col = find_amount_column(df_clean)

            if (
                "year" in df_clean.columns
                and "month_num" in df_clean.columns
                and "MONTO" in df_clean.columns
            ):
                try:
                    monthly = aggregate_monthly(df_clean, "MONTO")

                    if monthly is not None and len(monthly) > 0:
                        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
                            [
                                "Overview",
                                "Statistics",
                                "Time Series",
                                "Correlations",
                                "Executive",
                                "Comparative",
                                "Prediction",
                            ]
                        )

                        with tab1:
                            _show_monthly_metrics(monthly)
                            _show_eda_charts(monthly, df_clean, amount_col)

                        with tab2:
                            _show_descriptive_stats(monthly)

                        with tab3:
                            _show_seasonal_analysis(monthly)
                            _show_trend_analysis(monthly)
                            _show_patterns_analysis(monthly)

                        with tab4:
                            _show_correlation_analysis(monthly)

                        with tab5:
                            _show_analysis_by_executive(df_clean, monthly)

                        with tab6:
                            _show_comparative_monthly_analysis(monthly, df_clean)

                        with tab7:
                            _show_prediction_analysis(monthly, df_clean)
                    elif monthly is not None:
                        st.warning(
                            "⚠️ No historical data available for detailed analysis"
                        )
                    else:
                        st.warning("⚠️ No valid data found after cleaning")
                except Exception as e:
                    st.error(f"Error in monthly aggregation: {e}")
                    _show_basic_stats_fallback(df_clean, amount_col)
            else:
                st.warning("⚠️ Not enough valid data for monthly aggregation")
        else:
            st.info(
                "ℹ️ For detailed temporal analysis, ensure the file has a date column"
            )

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)

        st.download_button(
            label="Download Original File",
            data=output.getvalue(),
            file_name="frm_2023_2026.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


def _show_monthly_metrics(monthly: pd.DataFrame):
    """Display monthly metrics summary with additional context."""
    CURRENT_MONTH = monthly.iloc[-1]
    historical = monthly.iloc[:-1].copy().reset_index(drop=True)

    if len(historical) > 0:
        avg_monthly_amount = historical["total_amount"].mean()
        avg_monthly_payments = historical["payment_count"].mean()
        amount_change = (
            (
                (CURRENT_MONTH["total_amount"] - avg_monthly_amount)
                / avg_monthly_amount
                * 100
            )
            if avg_monthly_amount != 0
            else 0
        )
        payments_change = (
            (
                (CURRENT_MONTH["payment_count"] - avg_monthly_payments)
                / avg_monthly_payments
                * 100
            )
            if avg_monthly_payments != 0
            else 0
        )
    else:
        avg_monthly_amount = avg_monthly_payments = amount_change = payments_change = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Complete historical months", len(historical))

    with col2:
        st.metric(
            "Current partial month",
            f"{CURRENT_MONTH['YEAR_MONTH']}",
            f"{CURRENT_MONTH['payment_count']:.0f} payments, ${CURRENT_MONTH['total_amount']:,.0f}",
        )

    with col3:
        if len(historical) > 0:
            st.metric(
                "Amount Change vs Historical Average",
                f"{amount_change:+.1f}%",
                f"${CURRENT_MONTH['total_amount']:,.0f} vs ${avg_monthly_amount:,.0f}",
            )
        else:
            st.metric("Historical Average Amount", "No data")

    with col4:
        if len(historical) > 0:
            st.metric(
                "Payment Change vs Historical Average",
                f"{payments_change:+.1f}%",
                f"{CURRENT_MONTH['payment_count']:.0f} vs {avg_monthly_payments:.0f}",
            )
        else:
            st.metric("Historical Average Payments", "No data")


def _show_descriptive_stats(monthly: pd.DataFrame):
    """Display descriptive statistics with confidence intervals."""
    st.write("---")
    st.subheader("📈 Detailed Descriptive Statistics")

    historical = monthly.iloc[:-1].copy().reset_index(drop=True)
    y = historical["total_amount"].astype(float).values

    if len(y) < 2:
        st.warning("⚠️ At least 2 data points are needed for descriptive statistics")
        return

    stats_data = calculate_descriptive_stats(y)

    from scipy import stats as scipy_stats

    n = len(y)
    mean = stats_data["mean"]
    std = stats_data["std"]
    se = std / np.sqrt(n) if n > 0 else 0
    confidence_level = 0.95
    t_critical = scipy_stats.t.ppf((1 + confidence_level) / 2, n - 1) if n > 1 else 0
    margin_of_error = t_critical * se
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error

    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

    with metrics_col1:
        st.write("**Measures of Central Tendency:**")
        st.write(f"• Mean: ${mean:,.0f}")
        st.write(f"• Median: ${stats_data['median']:,.0f}")
        st.write(f"• 95% CI: [${ci_lower:,.0f}, ${ci_upper:,.0f}]")

    with metrics_col2:
        st.write("**Measures of Dispersion:**")
        st.write(f"• Std. Deviation: ${std:,.0f}")
        st.write(f"• CV: {stats_data['cv']:.1f}%")
        st.write(f"• Range: ${stats_data['range']:,.0f}")

    with metrics_col3:
        st.write("**Measures of Shape:**")
        st.write(f"• Skewness: {stats_data['skew']:.3f}")
        st.write(f"• Kurtosis: {stats_data['kurtosis']:.3f}")

        skew_interp = (
            "Approximately symmetric"
            if abs(stats_data["skew"]) < 0.5
            else (
                "Positive skew (right tail)"
                if stats_data["skew"] > 0.5
                else "Negative skew (left tail)"
            )
        )
        st.write(f"→ {skew_interp}")

        kurtosis_interp = (
            "Mesokurtic"
            if abs(stats_data["kurtosis"]) < 0.5
            else "Leptokurtic" if stats_data["kurtosis"] > 0.5 else "Platykurtic"
        )
        st.write(f"• Kurtosis: {kurtosis_interp}")

    st.write("**Percentiles:**")
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    pct_cols = st.columns(len(percentiles))
    for i, p in enumerate(percentiles):
        with pct_cols[i]:
            st.metric(f"P{p}", f"${stats_data['percentiles'][p]:,.0f}")

    _show_outlier_detection(y)
    _show_normality_test(y)


def _show_outlier_detection(data: np.ndarray):
    """Display outlier detection results."""
    st.write("**Outlier Detection (IQR ×1.5):**")

    outlier_info = detect_outliers_iqr(data)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lower bound", f"${outlier_info['lower_bound']:,.0f}")
    with col2:
        st.metric("Upper bound", f"${outlier_info['upper_bound']:,.0f}")
    with col3:
        st.metric("Outliers detected", outlier_info["outlier_count"])

    if outlier_info["outlier_count"] == 0:
        st.success("✅ No outliers detected using IQR method")


def _show_normality_test(data: np.ndarray):
    """Display normality test results."""
    st.write("**Normality Test (Shapiro-Wilk):**")

    normality = test_normality(data)

    if normality["statistic"] is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("W Statistic", f"{normality['statistic']:.4f}")
        with col2:
            st.metric("p-value", f"{normality['p_value']:.4f}")

        if normality["is_normal"]:
            st.success("✅ NORMAL distribution (p > 0.05)")
        else:
            st.warning("⚠️ NON-normal distribution (p ≤ 0.05)")
    else:
        st.warning("⚠️ Could not perform normality test")


def _show_eda_charts(
    monthly: pd.DataFrame, df_original: pd.DataFrame, amount_col: Optional[str]
):
    """Display exploratory data analysis charts."""
    st.write("---")
    st.subheader("📊 Exploratory Data Analysis")
    st.info("""
    **This set of charts shows:**
    1. **Monthly Recovery**: Bars showing total amount recovered each month (in millions)
    2. **Distribution by Month**: Box plots showing amount variability by month
    3. **Inter-annual Comparison**: Lines comparing monthly behavior across different years
    4. **Distribution by Payment Type**: Horizontal bars showing total amount by payment category
    """)
    if amount_col is not None:
        create_eda_charts(monthly.iloc[:-1].copy(), df_original, amount_col)
    else:
        st.warning("⚠️ No amount column found for EDA charts")


def _show_seasonal_analysis(monthly: pd.DataFrame):
    """Display seasonal decomposition analysis with enhanced insights."""
    st.write("---")
    st.subheader("🔄 Time Series Decomposition")
    st.info("""
    **This analysis shows the decomposition of the time series into:**
    1. **Trend**: General long-term direction of the data
    2. **Seasonality**: Patterns that repeat at regular periods (monthly)
    3. **Residual**: Remaining variation after removing trend and seasonality
    """)

    historical = monthly.iloc[:-1].copy()

    decomp = create_seasonal_decomposition_chart(historical)

    seasonal_df = calculate_seasonal_indices(historical)
    if seasonal_df is not None:
        st.write("**Seasonal Indices:**")
        st.dataframe(seasonal_df.round(4))
        st.write(
            "**Interpretation:** Values > 1 indicate months with above-average recovery"
        )

        if decomp is not None:
            try:
                seasonal_var = np.var(decomp.seasonal)
                residual_var = np.var(decomp.resid)
                if seasonal_var + residual_var > 0:
                    Fs = seasonal_var / (seasonal_var + residual_var)
                    Fs = max(0, min(1, Fs))
                    st.write(f"**Strength of Seasonality (Fs):** {Fs:.3f}")
                    if Fs > 0.6:
                        st.write("→ High seasonality detected")
                    elif Fs > 0.3:
                        st.write("→ Moderate seasonality detected")
                    else:
                        st.write("→ Low seasonality detected")
            except Exception:
                pass
    else:
        st.info(
            "ℹ️ Could not calculate seasonal decomposition. At least 2 years of monthly data are needed."
        )


def _show_trend_analysis(monthly: pd.DataFrame):
    """Display trend analysis with additional metrics."""
    st.write("---")
    st.subheader("📈 Trend Analysis")
    st.info("""
    **This analysis shows the data trend through:**
    1. **Trend Line**: Best-fit line showing the general direction
    2. **Moving Average**: Average of the last 3 months to smooth fluctuations
    3. **Distribution**: Histogram showing the frequency of different recovery values
    4. **Cumulative Recovery**: Progressive sum of amounts over time
    """)

    historical = monthly.iloc[:-1].copy()

    if len(historical) < 2:
        st.warning("⚠️ At least 2 data points are needed for trend analysis")
        return

    from scipy import stats as scipy_stats

    y = historical["total_amount"].values
    x = np.arange(len(y))

    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)

    r_squared = r_value**2

    trend_significant = p_value < 0.05

    if len(y) >= 2 and y[0] != 0:
        pct_change = ((y[-1] - y[0]) / y[0]) * 100
    else:
        pct_change = 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Linear Trend", f"{slope:,.0f} per month", f"R² = {r_squared:.3f}")

    with col2:
        st.metric(
            "Trend Significance",
            "Significant" if trend_significant else "Not significant",
            f"p-value = {p_value:.4f}",
        )

    with col3:
        st.metric("Total Change", f"{pct_change:+.1f}%", "From first to last period")

    with col4:
        if len(y) >= 2 and y[0] > 0:
            cmgr = (y[-1] / y[0]) ** (1 / len(y)) - 1
            st.metric("Compound Monthly Growth Rate", f"{cmgr*100:+.2f}%", "Per period")
        else:
            st.metric("Compound Monthly Growth Rate", "N/A")

    create_trend_analysis(historical)


def _show_patterns_analysis(monthly: pd.DataFrame):
    """Display yearly and monthly patterns with enhanced insights."""
    st.write("---")
    st.subheader("📅 Annual and Monthly Patterns")
    st.info("""
    **This analysis shows recurring patterns in the data:**
    1. **Statistics by Year**: Average, standard deviation, min, max, and year-over-year growth
    2. **Year-over-Year Growth**: Percentage change year to year in average recovery
    3. **Monthly Pattern**: Typical behavior of each month across years (seasonality)
    """)

    historical = monthly.iloc[:-1].copy().reset_index(drop=True)

    if len(historical) < 2:
        st.warning("⚠️ At least 2 data points are needed for pattern analysis")
        return

    yearly = calculate_yearly_stats(historical)
    st.write("**Statistics by Year:**")
    st.dataframe(yearly.round(2))

    create_year_growth_chart(yearly)

    if len(yearly) >= 2:
        latest_year = yearly.iloc[-1]
        prev_year = yearly.iloc[-2] if len(yearly) >= 2 else None

        if prev_year is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Average Annual Growth",
                    (
                        f"{latest_year['growth']:.1f}%"
                        if not pd.isna(latest_year["growth"])
                        else "N/A"
                    ),
                )
            with col2:
                st.metric(
                    "Annual Volatility (CV)",
                    (
                        f"{latest_year['cv']:.1f}%"
                        if not pd.isna(latest_year["cv"])
                        else "N/A"
                    ),
                )
            with col3:
                st.metric(
                    "Total Recovered Current Year",
                    (
                        f"${latest_year['sum']:,.0f}"
                        if not pd.isna(latest_year["sum"])
                        else "N/A"
                    ),
                )

    monthly_stats = calculate_monthly_stats(historical)
    st.write("**Statistics by Month (Seasonal Pattern):**")
    st.dataframe(monthly_stats.round(2))

    create_monthly_pattern_chart(historical)

    if len(monthly_stats) > 0:
        peak_month = monthly_stats["mean"].idxmax()
        low_month = monthly_stats["mean"].idxmin()

        peak_value = monthly_stats.loc[peak_month, "mean"]
        low_value = monthly_stats.loc[low_month, "mean"]

        if peak_value > 0 and low_value > 0:
            seasonal_ratio = peak_value / low_value
            st.write(f"**Seasonal Ratio (Peak/Valley):** {seasonal_ratio:.2f}x")
            if seasonal_ratio > 2:
                st.write("→ Strong seasonal variability detected")
            elif seasonal_ratio > 1.5:
                st.write("→ Moderate seasonal variability detected")
            else:
                st.write("→ Low seasonal variability")


def _show_correlation_analysis(monthly: pd.DataFrame):
    """Display correlation analysis with significance testing."""
    st.write("---")
    st.subheader("🔗 Correlation Analysis")
    st.info("""
    **This analysis shows the relationships between different variables:**
    1. **Correlation Heatmap**: Color visualization showing the strength and direction of relationships between variables
    2. **Correlations with Total Amount**: Numerical values indicating how each variable relates to total recovered amount
    """)

    if monthly.shape[1] < 2:
        st.info("ℹ️ Not enough numeric columns for correlation analysis")
        return

    amount_col_monthly = "total_amount"

    if amount_col_monthly not in monthly.columns:
        st.warning(f"⚠️ Column '{amount_col_monthly}' is not available for correlation")
        return

    create_correlation_heatmap(monthly)

    corr_df = calculate_correlations(monthly)

    if corr_df is not None:
        st.write("**Correlations with Total Amount:**")

        try:
            from scipy import stats as scipy_stats

            numeric_cols = [
                col
                for col in monthly.columns
                if col != amount_col_monthly
                and pd.api.types.is_numeric_dtype(monthly[col])
            ]

            if len(numeric_cols) > 0 and len(monthly) > 2:
                corr_details = []
                for col in numeric_cols:
                    valid_data = monthly[[amount_col_monthly, col]].dropna()
                    if len(valid_data) > 2:
                        corr_val, p_val = scipy_stats.pearsonr(
                            valid_data[amount_col_monthly], valid_data[col]
                        )
                        corr_details.append(
                            {
                                "Variable": col,
                                "Correlation": f"{corr_val:.3f}",
                                "p-value": f"{p_val:.4f}",
                                "Significant": "Yes" if p_val < 0.05 else "No",
                            }
                        )

                if corr_details:
                    detail_df = pd.DataFrame(corr_details)
                    st.dataframe(detail_df)
                else:
                    st.dataframe(corr_df)
            else:
                st.dataframe(corr_df)
        except Exception:
            st.dataframe(corr_df)
    else:
        st.warning("⚠️ Could not calculate correlations")


def _show_analysis_by_executive(df_original: pd.DataFrame, monthly: pd.DataFrame):
    """Display analysis by executive."""
    st.write("---")
    st.subheader("👥 Analysis by Executive")

    if "EJECUTIVA" not in df_original.columns:
        st.info("ℹ️ For executive analysis, ensure the file has an 'EJECUTIVA' column")
        return

    try:
        import matplotlib.pyplot as plt

        if len(monthly) == 0:
            st.warning("⚠️ No monthly data available")
            return

        current_period = monthly.iloc[-1]["YEAR_MONTH"]

        try:
            current_year = int(str(current_period)[:4])
            current_month_num = int(str(current_period)[5:7])
        except (ValueError, IndexError):
            if "year" not in monthly.columns or "month" not in monthly.columns:
                st.error("⚠️ Could not extract year and month from data")
                return
            current_year = int(monthly.iloc[-1]["year"])
            current_month_num = int(monthly.iloc[-1]["month"])

        current_day_data = df_original[
            (df_original["year"] == current_year)
            & (df_original["month_num"] == current_month_num)
        ]
        if len(current_day_data) > 0:
            current_day_of_month = int(current_day_data["day"].max())
        else:
            current_day_of_month = (
                monthly.iloc[-1]["days_in_month"]
                if "days_in_month" in monthly.columns
                else 30
            )

        st.write(
            f"**Comparing up to day {current_day_of_month} of month {current_month_num}**"
        )

        current_month_data = df_original[
            (df_original["year"] == current_year)
            & (df_original["month_num"] == current_month_num)
            & (df_original["day"] <= current_day_of_month)
        ]

        historical_data = df_original[
            (df_original["month_num"] == current_month_num)
            & (df_original["day"] <= current_day_of_month)
            & (df_original["year"] < current_year)
        ]

        if len(current_month_data) == 0 and len(historical_data) == 0:
            st.warning("⚠️ No data available for analysis.")
            return

        if len(current_month_data) > 0:
            ejecutivo_current = (
                current_month_data.groupby("EJECUTIVA")
                .agg(total_amount=("MONTO", "sum"), payment_count=("MONTO", "count"))
                .reset_index()
            )

            ejecutivo_current = ejecutivo_current.sort_values(
                "total_amount", ascending=False
            )
        else:
            ejecutivo_current = pd.DataFrame(
                columns=["EJECUTIVA", "total_amount", "payment_count"]
            )

        if len(historical_data) > 0:
            ejecutivo_historical = (
                historical_data.groupby(["year", "EJECUTIVA"])
                .agg(total_amount=("MONTO", "sum"), payment_count=("MONTO", "count"))
                .reset_index()
            )

            ejecutivo_historical_avg = (
                ejecutivo_historical.groupby("EJECUTIVA")
                .agg(
                    total_amount=("total_amount", "mean"),
                    payment_count=("payment_count", "mean"),
                )
                .reset_index()
            )

            ejecutivo_historical_avg = ejecutivo_historical_avg.sort_values(
                "total_amount", ascending=False
            )
        else:
            ejecutivo_historical_avg = pd.DataFrame(
                columns=["EJECUTIVA", "total_amount", "payment_count"]
            )

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Top 5 Executives - Current Month (Amount)**")
            if len(ejecutivo_current) > 0:
                top_5_current = ejecutivo_current.head(5)[
                    ["EJECUTIVA", "total_amount", "payment_count"]
                ]
                top_5_current["total_amount"] = top_5_current["total_amount"].apply(
                    lambda x: f"${x:,.0f}"
                )
                st.dataframe(top_5_current, hide_index=True)
            else:
                st.write("No data for current month")

        with col2:
            st.write("**Top 5 Executives - Historical Average (Amount)**")
            if len(ejecutivo_historical_avg) > 0:
                top_5_historical = ejecutivo_historical_avg.head(5)[
                    ["EJECUTIVA", "total_amount", "payment_count"]
                ]
                top_5_historical["total_amount"] = top_5_historical[
                    "total_amount"
                ].apply(lambda x: f"${x:,.0f}")
                st.dataframe(top_5_historical, hide_index=True)
            else:
                st.write("No historical data")

        if len(ejecutivo_current) > 0 and len(ejecutivo_historical_avg) > 0:
            ejecutivo_merged = pd.merge(
                ejecutivo_current[["EJECUTIVA", "total_amount"]],
                ejecutivo_historical_avg[["EJECUTIVA", "total_amount"]],
                on="EJECUTIVA",
                suffixes=("_current", "_historical"),
            )

            if len(ejecutivo_merged) > 0:
                ejecutivo_merged["growth_pct"] = (
                    (
                        ejecutivo_merged["total_amount_current"]
                        - ejecutivo_merged["total_amount_historical"]
                    )
                    / ejecutivo_merged["total_amount_historical"]
                    * 100
                )

                st.write("**Growth by Executive (vs Historical Average)**")
                growth_data = ejecutivo_merged[
                    [
                        "EJECUTIVA",
                        "total_amount_current",
                        "total_amount_historical",
                        "growth_pct",
                    ]
                ].copy()
                growth_data["total_amount_current"] = growth_data[
                    "total_amount_current"
                ].apply(lambda x: f"${x:,.0f}")
                growth_data["total_amount_historical"] = growth_data[
                    "total_amount_historical"
                ].apply(lambda x: f"${x:,.0f}")
                growth_data["growth_pct"] = growth_data["growth_pct"].apply(
                    lambda x: f"{x:+.1f}%"
                )
                st.dataframe(
                    growth_data.sort_values("growth_pct", ascending=False),
                    hide_index=True,
                )
            else:
                st.info(
                    "ℹ️ No common executives between current and historical periods for growth comparison."
                )
        else:
            if len(ejecutivo_current) == 0:
                st.warning("⚠️ No data for current month up to today.")
            if len(ejecutivo_historical_avg) == 0:
                st.warning(
                    "⚠️ No historical data for the same period in previous years."
                )

        if len(ejecutivo_current) > 0 and len(ejecutivo_historical_avg) > 0:
            st.write("---")
            st.subheader("📊 Visual Comparison by Executive")

            ejecutivos_current = ejecutivo_current["EJECUTIVA"].tolist()
            ejecutivos_all = list(ejecutivos_current)

            comparison_data = pd.DataFrame(
                {
                    "Executive": ejecutivos_all,
                    "Current Month": [
                        ejecutivo_current.set_index("EJECUTIVA").loc[ej, "total_amount"]
                        for ej in ejecutivos_all
                    ],
                    "Historical Average": [
                        (
                            ejecutivo_historical_avg.set_index("EJECUTIVA").loc[
                                ej, "total_amount"
                            ]
                            if ej in ejecutivo_historical_avg["EJECUTIVA"].values
                            else 0
                        )
                        for ej in ejecutivos_all
                    ],
                }
            )

            comparison_data = comparison_data.sort_values(
                "Current Month", ascending=False
            )

            fig, ax = plt.subplots(figsize=(12, 6))

            x = np.arange(len(comparison_data))
            width = 0.35

            bars1 = ax.bar(
                x - width / 2,
                comparison_data["Current Month"],
                width,
                label="Current Month",
                color="skyblue",
                edgecolor="navy",
            )
            bars2 = ax.bar(
                x + width / 2,
                comparison_data["Historical Average"],
                width,
                label="Historical Average",
                color="lightcoral",
                edgecolor="darkred",
            )

            ax.set_xlabel("Executive", fontsize=12)
            ax.set_ylabel("Total Amount ($)", fontsize=12)
            ax.set_title(
                "Payment Comparison by Executive: Current Month vs Historical Average",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_data["Executive"], rotation=45, ha="right")
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

            def add_value_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"${height:,.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

            add_value_labels(bars1)
            add_value_labels(bars2)

            plt.tight_layout()
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error in executive analysis: {e}")
        st.info("ℹ️ Verify that the data has the correct format")


def _show_comparative_monthly_analysis(
    monthly: pd.DataFrame, df_original: pd.DataFrame
):
    """Display comparative analysis of current month vs historical data for the same month up to the same day."""
    st.write("---")
    st.subheader("📊 Comparative Monthly Analysis (Up to Current Day)")
    st.info("""
    **This analysis compares the current month up to today with the same period in previous years:**
    1. **Current values**: Total amount and number of payments for the current month up to today
    2. **Historical average**: Average of the same period (up to the same day) in previous years
    3. **Percentage variation**: Percentage change relative to historical average
    4. **Trend**: Evolution of the same period across years
    """)

    try:
        import matplotlib.pyplot as plt

        if len(monthly) == 0:
            st.warning("⚠️ No monthly data available")
            return

        current_month_data = monthly.iloc[-1]
        current_period = current_month_data["YEAR_MONTH"]
        st.write(f"**Analyzing data for period: {current_period}**")

        try:
            current_year = int(str(current_period)[:4])
            current_month_num = int(str(current_period)[5:7])
        except (ValueError, IndexError):
            if "year" not in monthly.columns or "month" not in monthly.columns:
                st.error("⚠️ Could not extract year and month from data")
                return
            current_year = int(monthly.iloc[-1]["year"])
            current_month_num = int(monthly.iloc[-1]["month"])

        current_day_data = df_original[
            (df_original["year"] == current_year)
            & (df_original["month_num"] == current_month_num)
        ]
        if len(current_day_data) > 0:
            current_day_of_month = int(current_day_data["day"].max())
        else:
            current_day_of_month = (
                monthly.iloc[-1]["days_in_month"]
                if "days_in_month" in monthly.columns
                else 30
            )

        st.write(
            f"**Comparing up to day {current_day_of_month} of month {current_month_num}**"
        )

        historical_mask = (
            (df_original["month_num"] == current_month_num)
            & (df_original["day"] <= current_day_of_month)
            & (df_original["year"] < current_year)
        )

        historical_data = df_original[historical_mask]

        if len(historical_data) == 0:
            st.warning(
                "⚠️ No historical data available for the same period (up to current day) in previous years."
            )
            _show_current_month_only(current_month_data)
            return

        historical_yearly = (
            historical_data.groupby("year")
            .agg(total_amount=("MONTO", "sum"), payment_count=("MONTO", "count"))
            .reset_index()
        )

        if len(historical_yearly) == 0:
            st.warning("⚠️ Could not calculate historical yearly aggregates.")
            _show_current_month_only(current_month_data)
            return

        current_amount = current_month_data["total_amount"]
        current_payments = current_month_data["payment_count"]

        historical_avg_amount = historical_yearly["total_amount"].mean()
        historical_avg_payments = historical_yearly["payment_count"].mean()

        previous_year_data = historical_yearly[
            historical_yearly["year"] == current_year - 1
        ]
        yoy_change_amount = None
        yoy_change_payments = None

        if len(previous_year_data) > 0:
            prev_year_amount = previous_year_data.iloc[0]["total_amount"]
            prev_year_payments = previous_year_data.iloc[0]["payment_count"]

            if prev_year_amount != 0:
                yoy_change_amount = (
                    (current_amount - prev_year_amount) / prev_year_amount
                ) * 100
            if prev_year_payments != 0:
                yoy_change_payments = (
                    (current_payments - prev_year_payments) / prev_year_payments
                ) * 100

        pct_change_amount = (
            ((current_amount - historical_avg_amount) / historical_avg_amount * 100)
            if historical_avg_amount != 0
            else 0
        )
        pct_change_payments = (
            (
                (current_payments - historical_avg_payments)
                / historical_avg_payments
                * 100
            )
            if historical_avg_payments != 0
            else 0
        )

        try:
            from scipy import stats as scipy_stats

            if len(historical_yearly) > 1:
                t_stat_amount, p_val_amount = scipy_stats.ttest_1samp(
                    historical_yearly["total_amount"].values, current_amount
                )
                t_stat_payments, p_val_payments = scipy_stats.ttest_1samp(
                    historical_yearly["payment_count"].values, current_payments
                )

                sig_amount = p_val_amount < 0.05
                sig_payments = p_val_payments < 0.05
            else:
                p_val_amount = p_val_payments = 1.0
                sig_amount = sig_payments = False
        except Exception:
            p_val_amount = p_val_payments = 1.0
            sig_amount = sig_payments = False

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Current Amount",
                f"${current_amount:,.0f}",
                f"{pct_change_amount:+.1f}% vs historical average",
            )
            if len(historical_yearly) > 1:
                sig_text = " (significant)" if sig_amount else " (not significant)"
                st.caption(f"p-value: {p_val_amount:.4f}{sig_text}")

        with col2:
            st.metric(
                "Current Payments",
                f"{current_payments:,.0f}",
                f"{pct_change_payments:+.1f}% vs historical average",
            )
            if len(historical_yearly) > 1:
                sig_text = " (significant)" if sig_payments else " (not significant)"
                st.caption(f"p-value: {p_val_payments:.4f}{sig_text}")

        with col3:
            if yoy_change_amount is not None:
                st.metric(
                    "Annual Amount Change",
                    f"{yoy_change_amount:+.1f}%",
                    f"vs {current_year-1}",
                )
            else:
                st.metric("Historical Average Amount", f"${historical_avg_amount:,.0f}")

        with col4:
            if yoy_change_payments is not None:
                st.metric(
                    "Annual Payment Change",
                    f"{yoy_change_payments:+.1f}%",
                    f"vs {current_year-1}",
                )
            else:
                st.metric(
                    "Historical Average Payments", f"{historical_avg_payments:,.0f}"
                )

        fig = plt.figure(figsize=(16, 12))

        ax1 = plt.subplot(3, 2, 1)
        comparison_data = [current_amount, historical_avg_amount]
        comparison_labels = ["Current", "Historical Average"]
        bars1 = ax1.bar(
            comparison_labels,
            comparison_data,
            color=["skyblue", "lightcoral"],
            alpha=0.8,
            edgecolor="black",
        )
        ax1.set_ylabel("Total Amount ($)", fontsize=12)
        ax1.set_title(
            "Total Amount: Current vs Historical Average",
            fontweight="bold",
            fontsize=14,
        )
        ax1.grid(True, alpha=0.3, axis="y")

        for bar, value in zip(bars1, comparison_data):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 * max(comparison_data),
                f"${value:,.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax2 = plt.subplot(3, 2, 2)
        comparison_data_p = [current_payments, historical_avg_payments]
        bars2 = ax2.bar(
            comparison_labels,
            comparison_data_p,
            color=["lightgreen", "orange"],
            alpha=0.8,
            edgecolor="black",
        )
        ax2.set_ylabel("Number of Payments", fontsize=12)
        ax2.set_title(
            "Number of Payments: Current vs Historical Average",
            fontweight="bold",
            fontsize=14,
        )
        ax2.grid(True, alpha=0.3, axis="y")

        for bar, value in zip(bars2, comparison_data_p):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01 * max(comparison_data_p),
                f"{value:,.0f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax3 = plt.subplot(3, 2, 3)
        if len(historical_yearly) > 0:
            historical_yearly_sorted = historical_yearly.sort_values("year")
            years = historical_yearly_sorted["year"].tolist()
            amounts = historical_yearly_sorted["total_amount"].tolist()

            ax3.plot(
                years, amounts, "b-o", linewidth=2.5, markersize=6, label="Total Amount"
            )
            ax3.set_xlabel("Year", fontsize=12)
            ax3.set_ylabel("Total Amount ($)", fontsize=12, color="b")
            ax3.tick_params(axis="y", labelcolor="b")
            ax3.set_title(
                f"Historical Amount Trend - Month {current_month_num}",
                fontweight="bold",
                fontsize=14,
            )
            ax3.set_xticks(years)
            ax3.tick_params(axis="x", rotation=45)
            ax3.grid(True, alpha=0.3)

            current_in_historical = historical_yearly[
                historical_yearly["year"] == current_year
            ]
            if len(current_in_historical) > 0:
                ax3.plot(
                    current_year,
                    current_in_historical.iloc[0]["total_amount"],
                    "b*",
                    markersize=15,
                    label="Current",
                )

            ax3.legend(loc="best")
        else:
            ax3.text(
                0.5,
                0.5,
                "Not enough historical data\nto show trend",
                ha="center",
                va="center",
                transform=ax3.transAxes,
                fontsize=12,
            )
            ax3.set_title(
                f"Historical Amount Trend - Month {current_month_num}",
                fontweight="bold",
                fontsize=14,
            )

        ax4 = plt.subplot(3, 2, 4)
        if len(historical_yearly) > 0:
            historical_yearly_sorted = historical_yearly.sort_values("year")
            years = historical_yearly_sorted["year"].tolist()
            payments = historical_yearly_sorted["payment_count"].tolist()

            ax4.plot(
                years,
                payments,
                "r-s",
                linewidth=2.5,
                markersize=6,
                label="Number of Payments",
            )
            ax4.set_xlabel("Year", fontsize=12)
            ax4.set_ylabel("Number of Payments", fontsize=12, color="r")
            ax4.tick_params(axis="y", labelcolor="r")
            ax4.set_title(
                f"Historical Payment Trend - Month {current_month_num}",
                fontweight="bold",
                fontsize=14,
            )
            ax4.set_xticks(years)
            ax4.tick_params(axis="x", rotation=45)
            ax4.grid(True, alpha=0.3)

            current_in_historical = historical_yearly[
                historical_yearly["year"] == current_year
            ]
            if len(current_in_historical) > 0:
                ax4.plot(
                    current_year,
                    current_in_historical.iloc[0]["payment_count"],
                    "r*",
                    markersize=15,
                    label="Current",
                )

            ax4.legend(loc="best")
        else:
            ax4.text(
                0.5,
                0.5,
                "Not enough historical data\nto show trend",
                ha="center",
                va="center",
                transform=ax4.transAxes,
                fontsize=12,
            )
            ax4.set_title(
                f"Historical Payment Trend - Month {current_month_num}",
                fontweight="bold",
                fontsize=14,
            )

        ax5 = plt.subplot(3, 2, 5)
        change_amount = current_amount - historical_avg_amount

        categories_waterfall_a = ["Historical Average", "Change", "Current Value"]
        values_waterfall_a = [historical_avg_amount, change_amount, current_amount]

        cumulative_a = [0]
        for i in range(len(values_waterfall_a) - 1):
            cumulative_a.append(cumulative_a[-1] + values_waterfall_a[i])

        colors_waterfall_a = ["gray"]
        for i in range(1, len(values_waterfall_a) - 1):
            colors_waterfall_a.append("green" if values_waterfall_a[i] >= 0 else "red")
        colors_waterfall_a.append("blue")

        bars5 = ax5.bar(
            range(len(categories_waterfall_a)),
            values_waterfall_a,
            color=colors_waterfall_a,
            alpha=0.8,
            edgecolor="black",
        )

        for i in range(1, len(bars5) - 1):
            ax5.plot(
                [i - 1 + 0.4, i - 1 + 0.6],
                [cumulative_a[i], cumulative_a[i]],
                "k--",
                alpha=0.5,
            )
            ax5.plot(
                [i - 0.4, i - 0.4],
                [cumulative_a[i], cumulative_a[i + 1]],
                "k--",
                alpha=0.5,
            )

        ax5.set_ylabel("Total Amount ($)", fontsize=12)
        ax5.set_title("Total Amount Change Breakdown", fontweight="bold", fontsize=14)
        ax5.set_xticks(range(len(categories_waterfall_a)))
        ax5.set_xticklabels(categories_waterfall_a, rotation=15)

        for i, (bar, value) in enumerate(zip(bars5, values_waterfall_a)):
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (0.01 * abs(height) if height >= 0 else -0.01 * abs(height)),
                f"${value:,.0f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

        ax6 = plt.subplot(3, 2, 6)
        change_payments = current_payments - historical_avg_payments

        categories_waterfall_p = ["Historical Average", "Change", "Current Value"]
        values_waterfall_p = [
            historical_avg_payments,
            change_payments,
            current_payments,
        ]

        cumulative_p = [0]
        for i in range(len(values_waterfall_p) - 1):
            cumulative_p.append(cumulative_p[-1] + values_waterfall_p[i])

        colors_waterfall_p = ["gray"]
        for i in range(1, len(values_waterfall_p) - 1):
            colors_waterfall_p.append("green" if values_waterfall_p[i] >= 0 else "red")
        colors_waterfall_p.append("blue")

        bars6 = ax6.bar(
            range(len(categories_waterfall_p)),
            values_waterfall_p,
            color=colors_waterfall_p,
            alpha=0.8,
            edgecolor="black",
        )

        for i in range(1, len(bars6) - 1):
            ax6.plot(
                [i - 1 + 0.4, i - 1 + 0.6],
                [cumulative_p[i], cumulative_p[i]],
                "k--",
                alpha=0.5,
            )
            ax6.plot(
                [i - 0.4, i - 0.4],
                [cumulative_p[i], cumulative_p[i + 1]],
                "k--",
                alpha=0.5,
            )

        ax6.set_ylabel("Number of Payments", fontsize=12)
        ax6.set_title("Payment Count Change Breakdown", fontweight="bold", fontsize=14)
        ax6.set_xticks(range(len(categories_waterfall_p)))
        ax6.set_xticklabels(categories_waterfall_p, rotation=15)

        for i, (bar, value) in enumerate(zip(bars6, values_waterfall_p)):
            height = bar.get_height()
            ax6.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (0.01 * abs(height) if height >= 0 else -0.01 * abs(height)),
                f"{value:,.0f}",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=10,
                fontweight="bold",
            )

        plt.tight_layout(pad=3.0)
        st.pyplot(fig)

        with st.expander("View detailed historical data"):
            if len(historical_yearly) > 0:
                display_data = historical_yearly[
                    ["year", "total_amount", "payment_count"]
                ].copy()
                if "total_amount" in display_data.columns:
                    display_data["total_amount"] = display_data["total_amount"].apply(
                        lambda x: f"${x:,.0f}" if pd.notnull(x) else "$0"
                    )
                display_data = display_data.sort_values("year", ascending=False)
                st.dataframe(display_data)
            else:
                st.write("No historical data available for this period.")

    except Exception as e:
        st.error(f"Error in comparative monthly analysis: {e}")
        st.info("ℹ️ Verify that the data has the correct format")


def _show_current_month_only(current_month_data):
    """Helper function to show only current month data when no historical data is available."""
    st.write("### Current Month Data (No historical data for comparison)")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Amount", f"${current_month_data['total_amount']:,.0f}")
    with col2:
        st.metric("Current Payments", f"{current_month_data['payment_count']:,.0f}")

    st.info("ℹ️ Data from previous years is needed for historical comparisons.")


def _show_basic_stats_fallback(df: pd.DataFrame, amount_col: str):
    """Show basic statistics when monthly aggregation fails."""
    st.info("ℹ️ Falling back to basic statistics")

    numeric_series = pd.to_numeric(df[amount_col], errors="coerce").dropna()

    if len(numeric_series) > 0:
        st.write("**Basic Statistics:**")
        st.write(f"• Count: {len(numeric_series)}")
        st.write(f"• Sum: ${numeric_series.sum():,.0f}")
        st.write(f"• Mean: ${numeric_series.mean():,.0f}")
        st.write(f"• Median: ${numeric_series.median():,.0f}")
        st.write(f"• Std Dev: ${numeric_series.std():,.0f}")
        st.write(f"• Min: ${numeric_series.min():,.0f}")
        st.write(f"• Max: ${numeric_series.max():,.0f}")
    else:
        st.warning("⚠️ No valid numeric data found in amount column")


def create_features(df_hist):
    """Create lagged features for prediction models."""
    feat = pd.DataFrame(index=df_hist.index)

    feat["year"] = df_hist["year"] if "year" in df_hist.columns else 0
    feat["month"] = df_hist["month"] if "month" in df_hist.columns else 1

    for lag in [1, 2, 3, 6, 12]:
        feat[f"amount_lag_{lag}"] = df_hist["total_amount"].shift(lag)
        feat[f"payment_count_lag_{lag}"] = df_hist["payment_count"].shift(lag)

    for window in [3, 6, 12]:
        feat[f"amount_ma_{window}"] = df_hist["total_amount"].rolling(window).mean()
        feat[f"payment_count_ma_{window}"] = (
            df_hist["payment_count"].rolling(window).mean()
        )

    feat["amount_std_6"] = df_hist["total_amount"].rolling(6).std()
    feat["payment_count_std_6"] = df_hist["payment_count"].rolling(6).std()

    feat["amount_diff_1"] = df_hist["total_amount"].diff(1)
    feat["amount_diff_12"] = df_hist["total_amount"].diff(12)

    feat["pct_judicial"] = (
        df_hist["pct_judicial"] if "pct_judicial" in df_hist.columns else 0
    )
    feat["pct_castigo"] = (
        df_hist["pct_castigo"] if "pct_castigo" in df_hist.columns else 0
    )

    feat["days_in_month"] = (
        df_hist["days_in_month"] if "days_in_month" in df_hist.columns else 30
    )

    return feat


def _show_prediction_analysis(monthly: pd.DataFrame, df_original: pd.DataFrame):
    """Display ML-based prediction for current month payments."""
    if not HAS_ML_LIBS:
        st.warning("⚠️ Machine Learning libraries not available for predictions")
        return

    st.write("---")
    st.subheader("🔮 Payment Prediction for Current Month")
    st.info("""
    **This analysis uses Machine Learning models to predict:**
    1. **Expected payments for the total month**: Based on historical patterns and trends
    2. **Expected amount for the total month**: Prediction of total value to be recovered
    3. **Estimated remaining payments**: Difference between expected and already registered
    4. **Estimated remaining amount**: Difference between expected and already registered
    """)

    try:
        if len(monthly) == 0:
            st.warning("⚠️ No monthly data available")
            return

        current_month_period = monthly.iloc[-1]["YEAR_MONTH"]

        st.write(f"**Predicting for month: {current_month_period}**")

        if len(monthly) < 2:
            st.warning("⚠️ Not enough historical data to generate reliable predictions")
            return

        df_hist = monthly.iloc[:-1].copy()

        if len(df_hist) < 15:
            st.warning("⚠️ Not enough historical data to generate reliable predictions")
            return

        required_columns = ["total_amount", "payment_count"]
        for col in required_columns:
            if col not in df_hist.columns:
                st.warning(f"⚠️ Column '{col}' not found for predictive analysis")
                return

        df_hist["total_amount"] = pd.to_numeric(
            df_hist["total_amount"], errors="coerce"
        )
        df_hist["payment_count"] = pd.to_numeric(
            df_hist["payment_count"], errors="coerce"
        )

        df_hist = df_hist.dropna(subset=["total_amount", "payment_count"])

        if len(df_hist) < 15:
            st.warning("⚠️ Not enough valid data after cleaning for predictions")
            return

        X_hist = create_features(df_hist)

        y_pagos = df_hist["payment_count"].values
        y_monto = df_hist["total_amount"].values

        valid_idx = X_hist.dropna().index
        if len(valid_idx) < 10:
            st.warning(
                "⚠️ Not enough valid data after creating features for predictions"
            )
            return

        X_clean = X_hist.loc[valid_idx].values
        y_pagos_clean = y_pagos[valid_idx]
        y_monto_clean = y_monto[valid_idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)

        pred_features_dict = {
            "year": int(df_hist["year"].iloc[-1]),
            "month": int(df_hist["month"].iloc[-1]),
        }

        amount_series = df_hist["total_amount"].values
        payment_count_series = df_hist["payment_count"].values

        for lag in [1, 2, 3, 6, 12]:
            if len(amount_series) >= lag:
                pred_features_dict[f"amount_lag_{lag}"] = amount_series[-lag]
                pred_features_dict[f"payment_count_lag_{lag}"] = payment_count_series[
                    -lag
                ]
            else:
                pred_features_dict[f"amount_lag_{lag}"] = np.nan
                pred_features_dict[f"payment_count_lag_{lag}"] = np.nan

        for window in [3, 6, 12]:
            if len(amount_series) >= window:
                pred_features_dict[f"amount_ma_{window}"] = np.mean(
                    amount_series[-window:]
                )
                pred_features_dict[f"payment_count_ma_{window}"] = np.mean(
                    payment_count_series[-window:]
                )
            else:
                pred_features_dict[f"amount_ma_{window}"] = np.nan
                pred_features_dict[f"payment_count_ma_{window}"] = np.nan

        if len(amount_series) >= 6:
            pred_features_dict["amount_std_6"] = np.std(amount_series[-6:])
            pred_features_dict["payment_count_std_6"] = np.std(
                payment_count_series[-6:]
            )
        else:
            pred_features_dict["amount_std_6"] = np.nan
            pred_features_dict["payment_count_std_6"] = np.nan

        if len(amount_series) >= 2:
            pred_features_dict["amount_diff_1"] = amount_series[-1] - amount_series[-2]
        else:
            pred_features_dict["amount_diff_1"] = 0

        if len(amount_series) >= 13:
            pred_features_dict["amount_diff_12"] = (
                amount_series[-1] - amount_series[-13]
            )
        else:
            pred_features_dict["amount_diff_12"] = 0

        pred_features_dict["pct_judicial"] = 0.0
        pred_features_dict["pct_castigo"] = 0.0
        pred_features_dict["days_in_month"] = (
            df_hist["days_in_month"].iloc[-1]
            if "days_in_month" in df_hist.columns
            else 30
        )

        X_pred = pd.DataFrame([pred_features_dict])

        for col in X_pred.columns:
            if pd.isna(X_pred[col].iloc[0]):
                hist_col = col
                if hist_col in X_hist.columns:
                    X_pred[col] = X_hist[hist_col].mean()
                else:
                    X_pred[col] = 0

        X_hist_features = create_features(df_hist)
        feature_names = X_hist_features.columns.tolist()
        feature_names = [
            f for f in feature_names if f not in ["total_amount", "payment_count"]
        ]

        X_pred = X_pred[feature_names]

        X_pred_scaled = scaler.transform(X_pred)

        modelos_pagos = {
            "XGBoost": xgb.XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                random_state=42,
                verbosity=0,
            ),
            "LightGBM": lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                random_state=42,
                verbose=-1,
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=42
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.03, random_state=42
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=42
            ),
        }

        modelos_monto = {
            "XGBoost": xgb.XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                random_state=42,
                verbosity=0,
            ),
            "LightGBM": lgb.LGBMRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                random_state=42,
                verbose=-1,
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=42
            ),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=300, max_depth=4, learning_rate=0.03, random_state=42
            ),
            "ExtraTrees": ExtraTreesRegressor(
                n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=42
            ),
        }

        pred_pagos = {}
        pred_monto = {}
        resultados_pagos = {}
        resultados_monto = {}

        tscv = TimeSeriesSplit(n_splits=3)
        for nombre, modelo in modelos_pagos.items():
            mae_scores = []
            mape_scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_pagos_clean[train_idx], y_pagos_clean[val_idx]
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_val)
                mae_scores.append(mean_absolute_error(y_val, y_pred))
                mape_scores.append(mean_absolute_percentage_error(y_val, y_pred) * 100)
            resultados_pagos[nombre] = {
                "MAE": np.mean(mae_scores),
                "MAPE": np.mean(mape_scores),
                "weights": 1 / (np.mean(mape_scores) + 1),
            }

        for nombre, modelo in modelos_monto.items():
            mae_scores = []
            mape_scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y_monto_clean[train_idx], y_monto_clean[val_idx]
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_val)
                mae_scores.append(mean_absolute_error(y_val, y_pred))
                mape_scores.append(mean_absolute_percentage_error(y_val, y_pred) * 100)
            resultados_monto[nombre] = {
                "MAE": np.mean(mae_scores),
                "MAPE": np.mean(mape_scores),
                "weights": 1 / (np.mean(mape_scores) + 1),
            }

        total_weight_pagos = sum(r["weights"] for r in resultados_pagos.values())
        total_weight_monto = sum(r["weights"] for r in resultados_monto.values())

        for nombre in resultados_pagos:
            resultados_pagos[nombre]["normalized_weight"] = (
                resultados_pagos[nombre]["weights"] / total_weight_pagos
            )
        for nombre in resultados_monto:
            resultados_monto[nombre]["normalized_weight"] = (
                resultados_monto[nombre]["weights"] / total_weight_monto
            )

        feature_importance_pagos = {}
        feature_importance_monto = {}

        for nombre, modelo in modelos_pagos.items():
            modelo.fit(X_scaled, y_pagos_clean)
            pred = modelo.predict(X_pred_scaled)[0]
            pred_pagos[nombre] = max(0, pred)
            if hasattr(modelo, "feature_importances_"):
                feature_importance_pagos[nombre] = modelo.feature_importances_

        for nombre, modelo in modelos_monto.items():
            modelo.fit(X_scaled, y_monto_clean)
            pred = modelo.predict(X_pred_scaled)[0]
            pred_monto[nombre] = max(0, pred)
            if hasattr(modelo, "feature_importances_"):
                feature_importance_monto[nombre] = modelo.feature_importances_

        pred_pagos_ensemble = sum(
            pred_pagos[n] * resultados_pagos[n]["normalized_weight"] for n in pred_pagos
        )
        pred_monto_ensemble = sum(
            pred_monto[n] * resultados_monto[n]["normalized_weight"] for n in pred_monto
        )

        payments_so_far = monthly.iloc[-1]["payment_count"]
        amount_so_far = monthly.iloc[-1]["total_amount"]

        remaining_pagos = pred_pagos_ensemble - payments_so_far
        remaining_monto = pred_monto_ensemble - amount_so_far

        st.write("### 📊 Prediction Results")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Registered Payments",
                f"{payments_so_far:,.0f}",
                help="Payments already registered in the current month",
            )

        with col2:
            st.metric(
                "Registered Amount",
                f"${amount_so_far:,.0f}",
                help="Amount already registered in the current month",
            )

        with col3:
            st.metric(
                "Expected Payments (Total)",
                f"{pred_pagos_ensemble:,.0f}",
                f"{remaining_pagos:,.0f} remaining",
                help="Total payment prediction for the month",
            )

        with col4:
            st.metric(
                "Expected Amount (Total)",
                f"${pred_monto_ensemble:,.0f}",
                f"${remaining_monto:,.0f} remaining",
                help="Total amount prediction for the month",
            )

        with st.expander("View model details"):
            st.write("**Payment Count Prediction:**")
            payments_df = pd.DataFrame(
                {
                    "Model": list(pred_pagos.keys()),
                    "Prediction": [f"{pred_pagos[n]:,.0f}" for n in pred_pagos.keys()],
                    "Weight": [
                        f"{resultados_pagos[n]['normalized_weight']:.1%}"
                        for n in pred_pagos.keys()
                    ],
                    "MAE": [
                        f"{resultados_pagos[n]['MAE']:,.2f}" for n in pred_pagos.keys()
                    ],
                    "MAPE": [
                        f"{resultados_pagos[n]['MAPE']:.2f}%" for n in pred_pagos.keys()
                    ],
                }
            )
            st.dataframe(payments_df)

            st.info("""
            **Understanding the metrics:**
            - **MAE (Mean Absolute Error)**: Average of absolute differences between predictions and actual values. Indicates how much the model is wrong on average, in the same units as the target variable (e.g., number of payments or amount in pesos).
            - **MAPE (Mean Absolute Percentage Error)**: Average of absolute percentage errors. Shows the error as a percentage of the actual value, allowing comparison of accuracy across different scales.
            """)

            if feature_importance_pagos:
                st.write("**Feature Importance (Payments):**")
                feature_names = (
                    create_features(df_hist).dropna(axis=1, how="all").columns.tolist()
                )
                feature_names = [
                    f
                    for f in feature_names
                    if f not in ["total_amount", "payment_count"]
                ]

                avg_importance = np.mean(
                    [
                        feature_importance_pagos[model]
                        for model in feature_importance_pagos.keys()
                    ],
                    axis=0,
                )

                importance_df_pagos = pd.DataFrame(
                    {"Feature": feature_names, "Importance": avg_importance}
                ).sort_values("Importance", ascending=False)

                top_features = importance_df_pagos.head(5)

                feature_descriptions = {
                    "amount_lag_1": "Previous month amount - Indicates immediate recovery trend",
                    "amount_lag_2": "Amount from 2 months ago - Short-term pattern",
                    "amount_lag_3": "Amount from 3 months ago - Quarterly influence",
                    "amount_lag_6": "Amount from 6 months ago - Semi-annual pattern",
                    "amount_lag_12": "Amount from 12 months ago - Annual seasonality",
                    "amount_ma_3": "3-month moving average - Smoothed short-term trend",
                    "amount_ma_6": "6-month moving average - Intermediate trend",
                    "amount_ma_12": "12-month moving average - Annual trend",
                    "amount_std_6": "6-month standard deviation - Recent volatility",
                    "amount_diff_1": "Monthly change in amount - Immediate momentum",
                    "amount_diff_12": "Annual change in amount - Seasonal momentum",
                    "payment_count_lag_1": "Previous month payment count - Immediate volume",
                    "payment_count_lag_2": "Payment count from 2 months ago - Short-term volume",
                    "payment_count_lag_3": "Payment count from 3 months ago - Quarterly volume",
                    "payment_count_lag_6": "Payment count from 6 months ago - Semi-annual volume",
                    "payment_count_lag_12": "Payment count from 12 months ago - Annual volume",
                    "payment_count_ma_3": "3-month payment moving average - Volume trend",
                    "payment_count_ma_6": "6-month payment moving average - Intermediate volume trend",
                    "payment_count_ma_12": "12-month payment moving average - Annual volume trend",
                    "payment_count_std_6": "6-month payment std deviation - Volume volatility",
                    "payment_count_diff_1": "Monthly change in payment count - Volume momentum",
                    "payment_count_diff_12": "Annual change in payment count - Seasonal volume momentum",
                    "year": "Current year - Temporal trend factor",
                    "month": "Current month - Seasonal factor",
                    "pct_judicial": "Percentage of judicial cases - Legal proceedings impact",
                    "pct_castigo": "Percentage of penalty cases - Punitive measures impact",
                    "days_in_month": "Number of days in month - Recovery opportunity factor",
                }

                display_data = []
                for _, row in top_features.iterrows():
                    feature = row["Feature"]
                    importance = row["Importance"]
                    description = feature_descriptions.get(
                        feature, "Feature derived from historical data"
                    )
                    display_data.append(
                        {
                            "Feature": feature,
                            "Importance": f"{importance:.3f}",
                            "Description": description,
                        }
                    )

                display_df = pd.DataFrame(display_data)
                st.dataframe(display_df, hide_index=True)

                st.caption(
                    """Features with higher importance have more influence on predictions.
                Lags show historical values, moving averages show smoothed trends,
                and differences show momentary changes."""
                )

            st.write("**Total Amount Prediction:**")
            amount_df = pd.DataFrame(
                {
                    "Model": list(pred_monto.keys()),
                    "Prediction": [f"${pred_monto[n]:,.0f}" for n in pred_monto.keys()],
                    "Weight": [
                        f"{resultados_monto[n]['normalized_weight']:.1%}"
                        for n in pred_monto.keys()
                    ],
                    "MAE": [
                        f"{resultados_monto[n]['MAE']:,.2f}" for n in pred_monto.keys()
                    ],
                    "MAPE": [
                        f"{resultados_monto[n]['MAPE']:.2f}%" for n in pred_monto.keys()
                    ],
                }
            )
            st.dataframe(amount_df)

            st.info("""
            **Understanding the metrics:**
            - **MAE (Mean Absolute Error)**: Average of absolute differences between predictions and actual values. Indicates how much the model is wrong on average, in the same units as the target variable (e.g., number of payments or amount in pesos).
            - **MAPE (Mean Absolute Percentage Error)**: Average of absolute percentage errors. Shows the error as a percentage of the actual value, allowing comparison of accuracy across different scales.
            """)

            if feature_importance_monto:
                st.write("**Feature Importance (Amount):**")
                feature_names = (
                    create_features(df_hist).dropna(axis=1, how="all").columns.tolist()
                )
                feature_names = [
                    f
                    for f in feature_names
                    if f not in ["total_amount", "payment_count"]
                ]

                avg_importance = np.mean(
                    [
                        feature_importance_monto[model]
                        for model in feature_importance_monto.keys()
                    ],
                    axis=0,
                )

                importance_df_monto = pd.DataFrame(
                    {"Feature": feature_names, "Importance": avg_importance}
                ).sort_values("Importance", ascending=False)

                top_features = importance_df_monto.head(5)

                feature_descriptions = {
                    "amount_lag_1": "Previous month amount - Indicates immediate recovery trend",
                    "amount_lag_2": "Amount from 2 months ago - Short-term pattern",
                    "amount_lag_3": "Amount from 3 months ago - Quarterly influence",
                    "amount_lag_6": "Amount from 6 months ago - Semi-annual pattern",
                    "amount_lag_12": "Amount from 12 months ago - Annual seasonality",
                    "amount_ma_3": "3-month moving average - Smoothed short-term trend",
                    "amount_ma_6": "6-month moving average - Intermediate trend",
                    "amount_ma_12": "12-month moving average - Annual trend",
                    "amount_std_6": "6-month standard deviation - Recent volatility",
                    "amount_diff_1": "Monthly change in amount - Immediate momentum",
                    "amount_diff_12": "Annual change in amount - Seasonal momentum",
                    "payment_count_lag_1": "Previous month payment count - Immediate volume",
                    "payment_count_lag_2": "Payment count from 2 months ago - Short-term volume",
                    "payment_count_lag_3": "Payment count from 3 months ago - Quarterly volume",
                    "payment_count_lag_6": "Payment count from 6 months ago - Semi-annual volume",
                    "payment_count_lag_12": "Payment count from 12 months ago - Annual volume",
                    "payment_count_ma_3": "3-month payment moving average - Volume trend",
                    "payment_count_ma_6": "6-month payment moving average - Intermediate volume trend",
                    "payment_count_ma_12": "12-month payment moving average - Annual volume trend",
                    "payment_count_std_6": "6-month payment std deviation - Volume volatility",
                    "payment_count_diff_1": "Monthly change in payment count - Volume momentum",
                    "payment_count_diff_12": "Annual change in payment count - Seasonal volume momentum",
                    "year": "Current year - Temporal trend factor",
                    "month": "Current month - Seasonal factor",
                    "pct_judicial": "Percentage of judicial cases - Legal proceedings impact",
                    "pct_castigo": "Percentage of penalty cases - Punitive measures impact",
                    "days_in_month": "Number of days in month - Recovery opportunity factor",
                }

                display_data = []
                for _, row in top_features.iterrows():
                    feature = row["Feature"]
                    importance = row["Importance"]
                    description = feature_descriptions.get(
                        feature, "Feature derived from historical data"
                    )
                    display_data.append(
                        {
                            "Feature": feature,
                            "Importance": f"{importance:.3f}",
                            "Description": description,
                        }
                    )

                display_df = pd.DataFrame(display_data)
                st.dataframe(display_df, hide_index=True)

                st.caption(
                    """Features with higher importance have more influence on predictions.
                Lags show historical values, moving averages show smoothed trends,
                and differences show momentary changes."""
                )

        if len(monthly) > 1:
            estimated_progress = (
                min(95.0, (payments_so_far / pred_pagos_ensemble) * 100)
                if pred_pagos_ensemble > 0
                else 0
            )
            st.progress(min(1.0, estimated_progress / 100))
            st.caption(f"Estimated progress: {estimated_progress:.1f}% completed")

    except Exception as e:
        st.error(f"Error in predictive analysis: {e}")
        st.info("ℹ️ Verify that the data is sufficient and has the correct format")
