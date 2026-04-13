"""Utility functions for the Excel transformer application."""

import io
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

try:
    from statsmodels.tsa.seasonal import seasonal_decompose

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def normalize_column_name(col_name: Any) -> str:
    """Normalize column name for comparison: lowercase and remove underscores.

    Args:
        col_name: Column name to normalize

    Returns:
        Normalized column name (lowercase, underscores removed)
    """
    if not isinstance(col_name, str):
        col_name = str(col_name)
    return re.sub(r"_+", "", col_name.lower())


def find_matching_column(
    df_columns: Union[pd.Index, List[str]], target_col: str
) -> Optional[str]:
    """Find actual column name that matches target column (case-insensitive, underscore-insensitive).

    Args:
        df_columns: Column names from the dataframe (Index or list-like)
        target_col: Target column name to match

    Returns:
        Actual column name from df_columns that matches target_col, or None if not found
    """
    normalized_target = normalize_column_name(target_col)
    for col in df_columns:
        if normalize_column_name(col) == normalized_target:
            return col
    return None


def validate_required_columns(
    df_columns: Union[pd.Index, List[str]], columns_to_keep: List[str]
) -> Tuple[List[str], Dict[str, str]]:
    """Check if all required columns exist and create mapping.

    Args:
        df_columns: Column names from the dataframe (Index or list-like)
        columns_to_keep: List of expected column names

    Returns:
        Tuple of (missing_columns, column_mapping)
        missing_columns: List of expected columns not found in df_columns
        column_mapping: Dict mapping expected column names to actual column names
    """
    missing_columns = []
    column_mapping = {}

    for expected_col in columns_to_keep:
        actual_col = find_matching_column(df_columns, expected_col)
        if actual_col is None:
            missing_columns.append(expected_col)
        else:
            column_mapping[expected_col] = actual_col

    return missing_columns, column_mapping


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataframe column names and prepare for analysis."""
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip().str.rstrip(".")
    return df_clean


def find_date_columns(df: pd.DataFrame) -> List[str]:
    """Find columns that contain date information."""
    return [
        col for col in df.columns if "fecha" in col.lower() or "date" in col.lower()
    ]


def find_amount_column(df: pd.DataFrame) -> Optional[str]:
    """Find the amount/payment column in the dataframe."""
    amount_keywords = ["monto", "amount", "valor"]
    amount_cols = [
        col
        for col in df.columns
        if any(keyword in col.lower() for keyword in amount_keywords)
    ]

    if amount_cols:
        return amount_cols[0]

    payment_cols = [
        col
        for col in df.columns
        if "pago" in col.lower()
        and (
            "monto" in col.lower() or "amount" in col.lower() or "valor" in col.lower()
        )
    ]
    if payment_cols:
        return payment_cols[0]

    return None


def process_date_columns(
    df: pd.DataFrame, known_date_col: Optional[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract date-based features from dataframe."""
    df_processed = df.copy()

    if known_date_col and known_date_col in df_processed.columns:
        date_columns = [known_date_col]
    else:
        date_columns = find_date_columns(df_processed)

    if date_columns:
        date_col = date_columns[0]
        try:
            df_processed[date_col] = pd.to_datetime(df_processed[date_col])
            df_processed["year"] = df_processed[date_col].dt.year
            df_processed["month_num"] = df_processed[date_col].dt.month
            df_processed["day"] = df_processed[date_col].dt.day
            df_processed["day_of_week"] = df_processed[date_col].dt.dayofweek
            df_processed["year_month"] = df_processed[date_col].dt.strftime("%Y-%m")
        except Exception:
            pass

    return df_processed, date_columns


def get_dataframe_info(df: pd.DataFrame) -> Dict:
    """Get basic information about a dataframe."""
    buffer = io.StringIO()
    df.info(buf=buffer)

    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    return {
        "info": buffer.getvalue(),
        "dtypes": df.dtypes.astype(str),
        "missing": pd.DataFrame(
            {
                "Missing Count": missing_values,
                "Percentage (%)": missing_percentage.round(2),
            }
        ),
    }


def aggregate_monthly(df: pd.DataFrame, amount_col: str) -> Optional[pd.DataFrame]:
    """Aggregate data by year-month."""
    if "year" not in df.columns or "month_num" not in df.columns:
        return None

    df_copy = df.copy()
    df_copy["month_num"] = pd.to_numeric(df_copy["month_num"], errors="coerce")

    df_copy[amount_col] = pd.to_numeric(df_copy[amount_col], errors="coerce")

    valid_df = df_copy.dropna(subset=["month_num", amount_col]).copy()

    if len(valid_df) == 0:
        return None

    valid_df["YEAR_MONTH"] = (
        valid_df["year"].astype(str)
        + "-"
        + valid_df["month_num"].astype(int).astype(str).str.zfill(2)
    )

    monthly = (
        valid_df.groupby("YEAR_MONTH")
        .agg({amount_col: ["sum", "count", "mean", "median", "std"]})
        .reset_index()
    )

    monthly.columns = [
        "YEAR_MONTH",
        "total_amount",
        "payment_count",
        "avg_amount",
        "median_amount",
        "std_amount",
    ]
    monthly["year"] = monthly["YEAR_MONTH"].str[:4].astype(int)
    monthly["month"] = monthly["YEAR_MONTH"].str[5:].astype(int)
    monthly["days_in_month"] = monthly.apply(
        lambda row: pd.Period(
            f"{int(row['year'])}-{int(row['month']):02d}"
        ).days_in_month,
        axis=1,
    )

    return monthly


def calculate_descriptive_stats(data: np.ndarray) -> Dict:
    """Calculate descriptive statistics for the data."""
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data, ddof=1),
        "cv": (np.std(data, ddof=1) / np.mean(data)) * 100 if np.mean(data) != 0 else 0,
        "range": np.max(data) - np.min(data),
        "skew": stats.skew(data),
        "kurtosis": stats.kurtosis(data),
        "percentiles": {p: np.percentile(data, p) for p in [5, 10, 25, 50, 75, 90, 95]},
    }


def detect_outliers_iqr(data: np.ndarray) -> Dict:
    """Detect outliers using IQR method."""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    return {
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outliers": outliers,
        "outlier_count": len(outliers),
    }


def test_normality(data: np.ndarray) -> Dict:
    """Test normality using Shapiro-Wilk test."""
    try:
        stat, p_value = stats.shapiro(data)
        return {"statistic": stat, "p_value": p_value, "is_normal": p_value > 0.05}
    except Exception:
        return {"statistic": None, "p_value": None, "is_normal": None}


def calculate_yearly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate yearly statistics."""
    if "executives_count" in df.columns:
        yearly = df.groupby("year").agg(
            {
                "total_amount": ["mean", "std", "min", "max", "sum"],
                "executives_count": "mean",
            }
        )
        yearly.columns = [
            "total_amount_mean",
            "total_amount_std",
            "total_amount_min",
            "total_amount_max",
            "total_amount_sum",
            "executives_count_mean",
        ]
        yearly = yearly.rename(
            columns={
                "total_amount_mean": "mean",
                "total_amount_std": "std",
                "total_amount_min": "min",
                "total_amount_max": "max",
                "total_amount_sum": "sum",
                "executives_count_mean": "executives_count",
            }
        )
    else:
        yearly = df.groupby("year")["total_amount"].agg(
            ["mean", "std", "min", "max", "sum"]
        )
        yearly["executives_count"] = 0

    yearly = yearly.assign(
        cv=lambda x: (x["std"] / x["mean"] * 100) if x["mean"].ne(0).all() else 0,
        growth=lambda x: x["mean"].pct_change() * 100,
    )
    return yearly


def calculate_monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly (seasonal) statistics."""
    if "executives_count" in df.columns:
        monthly_stats = df.groupby("month").agg(
            {"total_amount": ["mean", "std", "min", "max"], "executives_count": "mean"}
        )
        monthly_stats.columns = [
            "total_amount_mean",
            "total_amount_std",
            "total_amount_min",
            "total_amount_max",
            "executives_count_mean",
        ]
        monthly_stats = monthly_stats.rename(
            columns={
                "total_amount_mean": "mean",
                "total_amount_std": "std",
                "total_amount_min": "min",
                "total_amount_max": "max",
                "executives_count_mean": "executives_count",
            }
        )
    else:
        monthly_stats = df.groupby("month")["total_amount"].agg(
            ["mean", "std", "min", "max"]
        )
        monthly_stats["executives_count"] = 0

    monthly_stats["cv"] = monthly_stats["std"] / monthly_stats["mean"] * 100
    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    monthly_stats.index = [month_labels[m - 1] for m in monthly_stats.index]
    return monthly_stats


def calculate_seasonal_indices(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Calculate seasonal indices using statsmodels."""
    if not HAS_STATSMODELS or len(df) < 12:
        return None

    try:
        ts = pd.Series(
            df["total_amount"].values,
            index=pd.date_range(start="2023-01", periods=len(df), freq="MS"),
        )

        if (ts <= 0).any():
            decomp = seasonal_decompose(ts, model="additive", period=12)
            seasonal_stl = decomp.seasonal[:12].values
            seasonal_stl = seasonal_stl / seasonal_stl.mean()
        else:
            decomp = seasonal_decompose(ts, model="multiplicative", period=12)
            seasonal_stl = decomp.seasonal[:12].values

        return pd.DataFrame(
            {
                "Month": [
                    "Jan",
                    "Feb",
                    "Mar",
                    "Apr",
                    "May",
                    "Jun",
                    "Jul",
                    "Aug",
                    "Sep",
                    "Oct",
                    "Nov",
                    "Dec",
                ],
                "Seasonal Index": seasonal_stl,
            }
        )
    except Exception:
        return None


def calculate_correlations(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Calculate correlations between numeric variables."""
    numeric_cols = [
        "total_amount",
        "payment_count",
        "avg_amount",
        "median_amount",
        "std_amount",
        "days_in_month",
    ]
    available_cols = [col for col in numeric_cols if col in df.columns]

    if len(available_cols) < 2:
        return None

    corr = df[available_cols].corr()

    if "total_amount" not in corr.columns:
        return None

    corr_with_total = corr["total_amount"].drop("total_amount")

    if len(corr_with_total) == 0:
        return None

    return pd.DataFrame(
        [
            {"Variable": col, "Correlation": f"{val:.3f}"}
            for col, val in corr_with_total.items()
        ]
    )


def create_eda_charts(
    historical: pd.DataFrame, df_original: pd.DataFrame, amount_col: str
) -> None:
    """Create exploratory data analysis charts using seaborn with enhanced annotations.

    Args:
        historical: Monthly aggregated historical data
        df_original: Original dataframe with all payment data
        amount_col: Name of the amount column
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        if not pd.api.types.is_numeric_dtype(historical["total_amount"]):
            historical = historical.copy()
            historical["total_amount"] = pd.to_numeric(
                historical["total_amount"], errors="coerce"
            )
            if (
                len(historical["total_amount"]) > 0
                and historical["total_amount"].isnull().values.all()
            ):
                st.warning(
                    "⚠️ The amount column does not contain valid numeric values after conversion."
                )
                return

        if (historical["total_amount"] < 0).any():
            neg_count = (historical["total_amount"] < 0).sum()
            st.warning(
                f"⚠️ Found {neg_count} negative values in the amount column. Absolute values will be used for charts."
            )
            historical_chart = historical.copy()
            historical_chart["total_amount"] = historical_chart["total_amount"].abs()
        else:
            historical_chart = historical

        month_labels = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        unique_years = sorted(historical_chart["year"].unique())
        palette = sns.color_palette("mako_r", n_colors=len(unique_years))
        year_colors = {year: palette[i] for i, year in enumerate(unique_years)}
        colors = [year_colors[y] for y in historical_chart["year"]]

        ax1 = axes[0, 0]
        bars = ax1.bar(
            range(len(historical_chart)),
            historical_chart["total_amount"] / 1e6,
            color=colors,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        ax1.axhline(
            historical_chart["total_amount"].mean() / 1e6,
            color="#f59e0b",
            ls="--",
            lw=2,
            label=f'Mean: ${historical_chart["total_amount"].mean()/1e6:.1f}M',
        )
        ax1.set_title(
            "Monthly Recovery (Millions $)", fontweight="bold", fontsize=12, pad=10
        )
        ax1.set_ylabel("Millions $", fontsize=10)
        ax1.set_xlabel("Month", fontsize=10)
        ax1.legend(loc="upper right")

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}M",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax2 = axes[0, 1]
        box_data = [
            historical_chart[historical_chart["month"] == m]["total_amount"].values
            / 1e6
            for m in range(1, 13)
        ]
        bp = ax2.boxplot(
            box_data, tick_labels=month_labels, patch_artist=True, notch=True
        )
        colors_box = sns.color_palette("mako", n_colors=12)
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_title("Distribution by Month", fontweight="bold", fontsize=12, pad=10)
        ax2.set_ylabel("Millions $", fontsize=10)

        ax3 = axes[1, 0]
        markers = ["o", "s", "D", "^"]
        for i, yr in enumerate(unique_years):
            sub = historical_chart[historical_chart["year"] == yr]
            if len(sub) > 0:
                ax3.plot(
                    sub["month"].values,
                    sub["total_amount"].values / 1e6,
                    marker=markers[i % len(markers)],
                    color=palette[i],
                    label=str(yr),
                    linewidth=2.5,
                    markersize=8,
                )
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(month_labels)
        ax3.set_title(
            "Year-over-Year Comparison", fontweight="bold", fontsize=12, pad=10
        )
        ax3.set_ylabel("Millions $", fontsize=10)
        ax3.set_xlabel("Month", fontsize=10)
        ax3.legend(title="Year", loc="upper right")

        ax4 = axes[1, 1]
        if amount_col and amount_col in df_original.columns:
            tipo_pago_cols = [
                col for col in df_original.columns if "tipo" in col.lower()
            ]
            if tipo_pago_cols:
                tp_col = tipo_pago_cols[0]
                numeric_amount = pd.to_numeric(df_original[amount_col], errors="coerce")
                if not pd.isna(numeric_amount).all():
                    temp_df = df_original.copy()
                    temp_df["_numeric_amount"] = numeric_amount
                    tp_grouped = temp_df.groupby(tp_col)["_numeric_amount"].sum()
                    tp_sorted = tp_grouped.sort_values(ascending=True)
                    tp = tp_sorted / 1e6
                    colors_bar = sns.color_palette("mako", n_colors=len(tp))
                    tp.plot(
                        kind="barh",
                        ax=ax4,
                        color=colors_bar,
                        alpha=0.8,
                        edgecolor="white",
                    )
                    ax4.set_title(
                        "Amount by Payment Type (Millions $)",
                        fontweight="bold",
                        fontsize=12,
                        pad=10,
                    )
                    ax4.set_xlabel("Millions $", fontsize=10)
                    del temp_df["_numeric_amount"]
                else:
                    ax4.text(
                        0.5,
                        0.5,
                        "Cannot sum amount values (non-numeric)",
                        ha="center",
                        va="center",
                        transform=ax4.transAxes,
                    )
                    ax4.set_title(
                        "Distribution by Type", fontweight="bold", fontsize=12, pad=10
                    )
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "No payment type column found",
                    ha="center",
                    va="center",
                    transform=ax4.transAxes,
                )
                ax4.set_title(
                    "Distribution by Type", fontweight="bold", fontsize=12, pad=10
                )
        else:
            ax4.text(
                0.5,
                0.5,
                "No amount column found",
                ha="center",
                va="center",
                transform=ax4.transAxes,
            )
            ax4.set_title(
                "Distribution by Type", fontweight="bold", fontsize=12, pad=10
            )

        plt.tight_layout()
        st.pyplot(fig)

    except ImportError:
        st.warning("⚠️ Seaborn is not installed. Charts cannot be displayed.")
    except Exception as e:
        st.warning(f"⚠️ Error generating charts: {e}")


def create_seasonal_decomposition_chart(
    historical: pd.DataFrame,
) -> Optional[Any]:
    """Create seasonal decomposition chart.

    Args:
        historical: Monthly aggregated historical data
    """
    try:
        import matplotlib.pyplot as plt

        if not HAS_STATSMODELS:
            st.info("ℹ️ statsmodels not available for seasonal decomposition")
            return None

        if len(historical) < 12:
            st.info(
                "ℹ️ At least 12 months of historical data are needed for seasonal decomposition"
            )
            return None

        ts = pd.Series(
            historical["total_amount"].values,
            index=pd.date_range(start="2023-01", periods=len(historical), freq="MS"),
        )

        if (ts <= 0).any():
            decomp = seasonal_decompose(ts, model="additive", period=12)
        else:
            decomp = seasonal_decompose(ts, model="multiplicative", period=12)

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

        colors = ["#6366f1", "#f59e0b", "#10b981", "#ec4899"]
        titles = ["Original", "Trend", "Seasonality", "Residual"]

        for ax, data, color, title in zip(
            axes,
            [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid],
            colors,
            titles,
        ):
            ax.plot(data, color=color, linewidth=2)
            ax.set_title(title, fontweight="bold", fontsize=11, loc="left")
            ax.set_ylabel("")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Date", fontsize=10)

        plt.tight_layout()
        st.pyplot(fig)

        return decomp

    except Exception as e:
        st.warning(f"⚠️ Error in seasonal decomposition: {e}")
        return None


def create_correlation_heatmap(df: pd.DataFrame) -> None:
    """Create correlation heatmap.

    Args:
        df: Monthly aggregated data
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        numeric_cols = [
            "total_amount",
            "payment_count",
            "avg_amount",
            "median_amount",
            "std_amount",
            "days_in_month",
        ]
        available_cols = [col for col in numeric_cols if col in df.columns]

        if len(available_cols) < 2:
            return None

        corr = df[available_cols].corr()

        labels = [
            "total\namount",
            "payment\ncount",
            "avg\namount",
            "median\namount",
            "std\namount",
            "days in\nmonth",
        ]

        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={"label": "Correlation", "shrink": 0.8},
        )

        ax.set_title("Correlation Matrix", fontweight="bold", fontsize=14, pad=15)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)

        plt.tight_layout()
        st.pyplot(plt.gcf())

    except ImportError:
        st.warning("⚠️ Seaborn is not installed.")
    except Exception as e:
        st.warning(f"⚠️ Error generating heatmap: {e}")


def create_year_growth_chart(yearly_stats: pd.DataFrame) -> None:
    """Create yearly growth chart.

    Args:
        yearly_stats: Yearly statistics dataframe
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        yearly_stats["mean"].plot(
            kind="bar", ax=ax1, color=sns.color_palette("mako", len(yearly_stats))
        )
        ax1.set_title("Average Recovery by Year", fontweight="bold")
        ax1.set_ylabel("Millions $")
        ax1.set_xlabel("Year")
        for i, v in enumerate(yearly_stats["mean"]):
            ax1.text(i, v / 1e6, f"${v/1e6:.1f}M", ha="center", va="bottom", fontsize=9)

        ax2 = axes[1]
        growth_data = yearly_stats["growth"].dropna()
        colors = ["green" if x > 0 else "red" for x in growth_data]
        growth_data.plot(kind="bar", ax=ax2, color=colors)
        ax2.set_title("Year-over-Year Growth Rate (%)", fontweight="bold")
        ax2.set_ylabel("Percentage (%)")
        ax2.set_xlabel("Year")
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"⚠️ Error generating growth chart: {e}")


def create_monthly_pattern_chart(historical: pd.DataFrame) -> None:
    """Create monthly pattern visualization.

    Args:
        historical: Monthly aggregated historical data
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        month_labels = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax1 = axes[0]
        monthly_avg = historical.groupby("month")["total_amount"].mean()
        sns.barplot(
            x=monthly_avg.index, y=monthly_avg.values / 1e6, ax=ax1, palette="mako"
        )
        ax1.set_xticklabels(month_labels)
        ax1.set_title("Average Monthly Pattern", fontweight="bold")
        ax1.set_ylabel("Millions $")
        ax1.set_xlabel("Month")

        ax2 = axes[1]
        seasonal_idx = historical.groupby("month")["total_amount"].mean()
        seasonal_idx = seasonal_idx / seasonal_idx.mean()
        colors = ["green" if x > 1 else "red" for x in seasonal_idx.values]
        sns.barplot(x=seasonal_idx.index, y=seasonal_idx.values, ax=ax2, palette=colors)
        ax2.axhline(y=1, color="black", linestyle="--", linewidth=1)
        ax2.set_xticklabels(month_labels)
        ax2.set_title("Seasonal Index (Average = 1)", fontweight="bold")
        ax2.set_ylabel("Index")
        ax2.set_xlabel("Month")

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"⚠️ Error generating pattern chart: {e}")


def create_trend_analysis(historical: pd.DataFrame) -> None:
    """Create trend analysis visualization.

    Args:
        historical: Monthly aggregated historical data
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scipy import stats as scipy_stats

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax1 = axes[0, 0]
        ax1.plot(
            range(len(historical)),
            historical["total_amount"].values / 1e6,
            marker="o",
            linewidth=2,
            label="Data",
        )

        x = np.arange(len(historical))
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
            x, historical["total_amount"].values
        )
        trend_line = slope * x + intercept
        ax1.plot(
            x,
            trend_line / 1e6,
            "r--",
            linewidth=2,
            label=f"Trend (R²={r_value**2:.3f})",
        )
        ax1.set_title("Trend Analysis", fontweight="bold")
        ax1.set_xlabel("Months since start")
        ax1.set_ylabel("Millions $")
        ax1.legend()

        ax2 = axes[0, 1]
        rolling_mean = historical["total_amount"].rolling(window=3).mean()
        rolling_std = historical["total_amount"].rolling(window=3).std()
        ax2.plot(historical["total_amount"].values / 1e6, label="Original", alpha=0.5)
        ax2.plot(
            rolling_mean.values / 1e6, label="Moving Average (3 months)", linewidth=2
        )
        ax2.fill_between(
            range(len(historical)),
            (rolling_mean - rolling_std).values / 1e6,
            (rolling_mean + rolling_std).values / 1e6,
            alpha=0.2,
            label="±1 Std. Dev.",
        )
        ax2.set_title("Moving Average (3 months)", fontweight="bold")
        ax2.set_xlabel("Months")
        ax2.set_ylabel("Millions $")
        ax2.legend()

        ax3 = axes[1, 0]
        sns.histplot(
            historical["total_amount"].values / 1e6,
            kde=True,
            ax=ax3,
            color=sns.color_palette("mako")[0],
        )
        ax3.axvline(
            historical["total_amount"].mean() / 1e6,
            color="red",
            linestyle="--",
            label="Mean",
        )
        ax3.axvline(
            historical["total_amount"].median() / 1e6,
            color="green",
            linestyle="--",
            label="Median",
        )
        ax3.set_title("Monthly Recovery Distribution", fontweight="bold")
        ax3.set_xlabel("Millions $")
        ax3.legend()

        ax4 = axes[1, 1]
        cumulative = historical["total_amount"].cumsum()
        ax4.fill_between(range(len(cumulative)), cumulative.values / 1e6, alpha=0.3)
        ax4.plot(cumulative.values / 1e6, linewidth=2)
        ax4.set_title("Cumulative Recovery", fontweight="bold")
        ax4.set_xlabel("Months")
        ax4.set_ylabel("Millions $")

        plt.tight_layout()
        st.pyplot(fig)

    except Exception as e:
        st.warning(f"⚠️ Error generating trend analysis: {e}")
