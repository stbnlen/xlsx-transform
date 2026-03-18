"""Utility functions for the Excel transformer application."""

import re
import io
from typing import List, Optional, Tuple, Dict

import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def normalize_column_name(col_name) -> str:
    """Normalize column name for comparison: lowercase and remove underscores.
    
    Args:
        col_name: Column name to normalize
        
    Returns:
        Normalized column name (lowercase, underscores removed)
    """
    if not isinstance(col_name, str):
        col_name = str(col_name)
    return re.sub(r'_+', '', col_name.lower())


def find_matching_column(df_columns, target_col: str) -> Optional[str]:
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


def validate_required_columns(df_columns, 
                            columns_to_keep: List[str]) -> Tuple[List[str], Dict[str, str]]:
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
    df_clean.columns = df_clean.columns.str.strip().str.rstrip('.')
    return df_clean


def find_date_columns(df: pd.DataFrame) -> List[str]:
    """Find columns that contain date information."""
    return [col for col in df.columns if 'fecha' in col.lower() or 'date' in col.lower()]


def find_amount_column(df: pd.DataFrame) -> Optional[str]:
    """Find the amount/payment column in the dataframe."""
    # Prioritize columns that are more specifically about amounts
    amount_keywords = ['monto', 'amount', 'valor']  # Removed 'pago' as it's too generic
    amount_cols = [col for col in df.columns 
                  if any(keyword in col.lower() for keyword in amount_keywords)]
    
    # If we found specific amount columns, return the first one
    if amount_cols:
        return amount_cols[0]
    
    # Fallback: look for payment-related columns but be more specific
    payment_cols = [col for col in df.columns 
                   if 'pago' in col.lower() and ('monto' in col.lower() or 'amount' in col.lower() or 'valor' in col.lower())]
    if payment_cols:
        return payment_cols[0]
        
    return None


def process_date_columns(df: pd.DataFrame, known_date_col: str = None) -> Tuple[pd.DataFrame, List[str]]:
    """Extract date-based features from dataframe."""
    df_processed = df.copy()
    
    # If a known date column is provided, use it
    if known_date_col and known_date_col in df_processed.columns:
        date_columns = [known_date_col]
    else:
        # Otherwise, find date columns automatically
        date_columns = find_date_columns(df_processed)
    
    if date_columns:
        date_col = date_columns[0]
        try:
            df_processed[date_col] = pd.to_datetime(df_processed[date_col])
            df_processed['AÑO'] = df_processed[date_col].dt.year
            df_processed['MES_NUM'] = df_processed[date_col].dt.month
            df_processed['DIA'] = df_processed[date_col].dt.day
            df_processed['DIA_SEMANA'] = df_processed[date_col].dt.dayofweek
            df_processed['AÑO_MES'] = df_processed[date_col].dt.strftime('%Y-%m')
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
        'info': buffer.getvalue(),
        'dtypes': df.dtypes.astype(str),
        'missing': pd.DataFrame({
            'Missing Count': missing_values,
            'Percentage (%)': missing_percentage.round(2)
        })
    }


def aggregate_monthly(df: pd.DataFrame, amount_col: str) -> Optional[pd.DataFrame]:
    """Aggregate data by year-month."""
    if 'AÑO' not in df.columns or 'MES_NUM' not in df.columns:
        return None
    
    df_copy = df.copy()
    df_copy['MES_NUM'] = pd.to_numeric(df_copy['MES_NUM'], errors='coerce')
    
    # Convert amount column to numeric first
    df_copy[amount_col] = pd.to_numeric(df_copy[amount_col], errors='coerce')
    
    # Drop rows where either MES_NUM or amount_col is NaN
    valid_df = df_copy.dropna(subset=['MES_NUM', amount_col]).copy()
    
    if len(valid_df) == 0:
        return None
    
    valid_df['YEAR_MONTH'] = (valid_df['AÑO'].astype(str) + '-' + 
                               valid_df['MES_NUM'].astype(int).astype(str).str.zfill(2))
    
    monthly = valid_df.groupby('YEAR_MONTH').agg({
        amount_col: ['sum', 'count', 'mean', 'median', 'std']
    }).reset_index()
    
    monthly.columns = ['AÑO_MES', 'monto_total', 'num_pagos', 'monto_prom', 'monto_mediana', 'monto_std']
    monthly['año'] = monthly['AÑO_MES'].str[:4].astype(int)
    monthly['mes'] = monthly['AÑO_MES'].str[5:].astype(int)
    monthly['dias_mes'] = monthly.apply(
        lambda row: pd.Period(f"{int(row['año'])}-{int(row['mes']):02d}").days_in_month, 
        axis=1
    )
    
    return monthly


def calculate_descriptive_stats(data: np.ndarray) -> Dict:
    """Calculate descriptive statistics for the data."""
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data, ddof=1),
        'cv': (np.std(data, ddof=1) / np.mean(data)) * 100 if np.mean(data) != 0 else 0,
        'range': np.max(data) - np.min(data),
        'skew': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'percentiles': {p: np.percentile(data, p) for p in [5, 10, 25, 50, 75, 90, 95]}
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
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'outliers': outliers,
        'outlier_count': len(outliers)
    }


def test_normality(data: np.ndarray) -> Dict:
    """Test normality using Shapiro-Wilk test."""
    try:
        stat, p_value = stats.shapiro(data)
        return {
            'statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
    except Exception:
        return {'statistic': None, 'p_value': None, 'is_normal': None}


def calculate_yearly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate yearly statistics."""
    yearly = df.groupby('año')['monto_total'].agg(['mean', 'std', 'min', 'max', 'sum'])
    yearly = yearly.assign(
        cv=lambda x: (x['std'] / x['mean'] * 100) if x['mean'].ne(0).all() else 0,
        crecimiento=lambda x: x['mean'].pct_change() * 100
    )
    return yearly


def calculate_monthly_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate monthly (seasonal) statistics."""
    monthly_stats = df.groupby('mes')['monto_total'].agg(['mean', 'std', 'min', 'max'])
    monthly_stats['cv'] = monthly_stats['std'] / monthly_stats['mean'] * 100
    meses_lbl = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
    monthly_stats.index = [meses_lbl[m-1] for m in monthly_stats.index]
    return monthly_stats


def calculate_seasonal_indices(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Calculate seasonal indices using statsmodels."""
    if not HAS_STATSMODELS or len(df) < 12:
        return None
    
    try:
        ts = pd.Series(
            df['monto_total'].values,
            index=pd.date_range(start='2023-01', periods=len(df), freq='MS')
        )
        
        # Use additive model if there are zero or negative values
        if (ts <= 0).any():
            decomp = seasonal_decompose(ts, model='additive', period=12)
            seasonal_stl = decomp.seasonal[:12].values
            # Normalize to mean = 1 for consistency
            seasonal_stl = seasonal_stl / seasonal_stl.mean()
        else:
            decomp = seasonal_decompose(ts, model='multiplicative', period=12)
            seasonal_stl = decomp.seasonal[:12].values
        
        return pd.DataFrame({
            'Mes': ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'],
            'Índice Estacional': seasonal_stl
        })
    except Exception:
        return None


def calculate_correlations(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Calculate correlations between numeric variables."""
    numeric_cols = ['monto_total', 'num_pagos', 'monto_prom', 'monto_mediana', 
                    'monto_std', 'dias_mes']
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    if len(available_cols) < 2:
        return None
    
    corr = df[available_cols].corr()
    
    if 'monto_total' not in corr.columns:
        return None
    
    corr_with_total = corr['monto_total'].drop('monto_total')
    
    if len(corr_with_total) == 0:
        return None
    
    return pd.DataFrame([
        {'Variable': col, 'Correlación': f"{val:.3f}"}
        for col, val in corr_with_total.items()
    ])


def create_eda_charts(historico: pd.DataFrame, df_original: pd.DataFrame, amount_col: str):
    """Create exploratory data analysis charts using seaborn with enhanced annotations.
    
    Args:
        historico: Monthly aggregated historical data
        df_original: Original dataframe with all payment data
        amount_col: Name of the amount column
    """
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Ensure historico['monto_total'] is numeric
        if not pd.api.types.is_numeric_dtype(historico['monto_total']):
            # Try to convert to numeric
            historico = historico.copy()
            historico['monto_total'] = pd.to_numeric(historico['monto_total'], errors='coerce')
            # If all values became NaN, show warning
            if len(historico['monto_total']) > 0 and historico['monto_total'].isnull().values.all():
                st.warning("⚠️ La columna de monto no contiene valores numéricos válidos después de la conversión.")
                return
        
        # Check for negative values in monto_total (which don't make sense for recovery amounts)
        if (historico['monto_total'] < 0).any():
            neg_count = (historico['monto_total'] < 0).sum()
            st.warning(f"⚠️ Se encontraron {neg_count} valores negativos en la columna de monto. Se usarán valores absolutos para los gráficos.")
            # Use absolute values for charts where negative amounts don't make sense
            historico_chart = historico.copy()
            historico_chart['monto_total'] = historico_chart['monto_total'].abs()
        else:
            historico_chart = historico
        
        meses_lbl = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Color palette by year
        unique_years = sorted(historico_chart['año'].unique())
        palette = sns.color_palette("mako_r", n_colors=len(unique_years))
        year_colors = {year: palette[i] for i, year in enumerate(unique_years)}
        colors = [year_colors[y] for y in historico_chart['año']]
        
        # 1. Monthly recovery bar chart
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(historico_chart)), historico_chart['monto_total'] / 1e6, 
                       color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
        ax1.axhline(historico_chart['monto_total'].mean() / 1e6, color='#f59e0b', 
                   ls='--', lw=2, label=f'Media: ${historico_chart["monto_total"].mean()/1e6:.1f}M')
        ax1.set_title('Recupero Mensual (Millones $)', fontweight='bold', fontsize=12, pad=10)
        ax1.set_ylabel('Millones $', fontsize=10)
        ax1.set_xlabel('Mes', fontsize=10)
        ax1.legend(loc='upper right')
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}M', ha='center', va='bottom', fontsize=9)
        
        # 2. Boxplot by month
        ax2 = axes[0, 1]
        box_data = [historico_chart[historico_chart['mes'] == m]['monto_total'].values / 1e6 for m in range(1, 13)]
        bp = ax2.boxplot(box_data, tick_labels=meses_lbl, patch_artist=True, notch=True)
        colors_box = sns.color_palette("mako", n_colors=12)
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax2.set_title('Distribución por Mes', fontweight='bold', fontsize=12, pad=10)
        ax2.set_ylabel('Millones $', fontsize=10)
        
        # 3. Inter-year comparison
        ax3 = axes[1, 0]
        markers = ['o', 's', 'D', '^']
        for i, yr in enumerate(unique_years):
            sub = historico_chart[historico_chart['año'] == yr]
            if len(sub) > 0:
                ax3.plot(sub['mes'].values, sub['monto_total'].values / 1e6,
                        marker=markers[i % len(markers)], color=palette[i], 
                        label=str(yr), linewidth=2.5, markersize=8)
        ax3.set_xticks(range(1, 13))
        ax3.set_xticklabels(meses_lbl)
        ax3.set_title('Comparación Interanual', fontweight='bold', fontsize=12, pad=10)
        ax3.set_ylabel('Millones $', fontsize=10)
        ax3.set_xlabel('Mes', fontsize=10)
        ax3.legend(title='Año', loc='upper right')
        
        # 4. Payment type distribution (if available)
        ax4 = axes[1, 1]
        if amount_col and amount_col in df_original.columns:
            tipo_pago_cols = [col for col in df_original.columns if 'tipo' in col.lower()]
            if tipo_pago_cols:
                tp_col = tipo_pago_cols[0]
                # Convert amount column to numeric, coercing errors to NaN
                numeric_amount = pd.to_numeric(df_original[amount_col], errors='coerce')
                # Check if we have any valid numeric values
                if not pd.isna(numeric_amount).all():
                    # Create a temporary dataframe with the numeric column for grouping
                    temp_df = df_original.copy()
                    temp_df['_numeric_amount'] = numeric_amount
                    # Group by payment type and sum the numeric amounts
                    tp_grouped = temp_df.groupby(tp_col)['_numeric_amount'].sum()
                    # Sort values and convert to millions
                    tp_sorted = tp_grouped.sort_values(ascending=True)
                    tp = tp_sorted / 1e6
                    colors_bar = sns.color_palette("mako", n_colors=len(tp))
                    tp.plot(kind='barh', ax=ax4, color=colors_bar, alpha=0.8, edgecolor='white')
                    ax4.set_title('Monto por Tipo de Pago (Millones $)', fontweight='bold', fontsize=12, pad=10)
                    ax4.set_xlabel('Millones $', fontsize=10)
                    # Clean up temporary column
                    del temp_df['_numeric_amount']
                else:
                    ax4.text(0.5, 0.5, 'No se pueden sumar los valores de monto (no numéricos)', 
                            ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Distribución por Tipo', fontweight='bold', fontsize=12, pad=10)
            else:
                ax4.text(0.5, 0.5, 'No hay columna de tipo de pago', 
                        ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Distribución por Tipo', fontweight='bold', fontsize=12, pad=10)
        else:
            ax4.text(0.5, 0.5, 'No hay columna de monto', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Distribución por Tipo', fontweight='bold', fontsize=12, pad=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except ImportError:
        st.warning("⚠️ Seaborn no está instalado. Los gráficos no se pueden mostrar.")
    except Exception as e:
        st.warning(f"⚠️ Error al generar gráficos: {e}")


def create_seasonal_decomposition_chart(historico: pd.DataFrame):
    """Create seasonal decomposition chart.
    
    Args:
        historico: Monthly aggregated historical data
    """
    try:
        import matplotlib.pyplot as plt
        
        if not HAS_STATSMODELS:
            st.info("ℹ️ statsmodels no disponible para descomposición estacional")
            return None
        
        if len(historico) < 12:
            st.info("ℹ️ Se necesitan al menos 12 meses históricos para descomposición estacional")
            return None
        
        ts = pd.Series(
            historico['monto_total'].values,
            index=pd.date_range(start='2023-01', periods=len(historico), freq='MS')
        )
        
        # Use additive model if there are zero or negative values
        if (ts <= 0).any():
            decomp = seasonal_decompose(ts, model='additive', period=12)
        else:
            decomp = seasonal_decompose(ts, model='multiplicative', period=12)
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
        
        colors = ['#6366f1', '#f59e0b', '#10b981', '#ec4899']
        titles = ['Original', 'Tendencia', 'Estacionalidad', 'Residuo']
        
        for ax, data, color, title in zip(axes, 
            [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid],
            colors, titles):
            ax.plot(data, color=color, linewidth=2)
            ax.set_title(title, fontweight='bold', fontsize=11, loc='left')
            ax.set_ylabel('')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Fecha', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        return decomp
        
    except Exception as e:
        st.warning(f"⚠️ Error en descomposición estacional: {e}")
        return None


def create_correlation_heatmap(df: pd.DataFrame):
    """Create correlation heatmap.
    
    Args:
        df: Monthly aggregated data
    """
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        numeric_cols = ['monto_total', 'num_pagos', 'monto_prom', 'monto_mediana', 
                       'monto_std', 'dias_mes']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) < 2:
            return None
        
        corr = df[available_cols].corr()
        
        labels = ['monto\ntotal', 'num\npagos', 'monto\nprom', 'monto\nmedian', 
                  'monto\nstd', 'dias\nmes']
        
        plt.figure(figsize=(10, 8))
        ax = sns.heatmap(corr, 
            annot=True, 
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1, 
            vmax=1,
            square=True,
            linewidths=0.5,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Correlación', 'shrink': 0.8})
        
        ax.set_title('Matriz de Correlación', fontweight='bold', fontsize=14, pad=15)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        
        plt.tight_layout()
        st.pyplot(plt.gcf())
        
    except ImportError:
        st.warning("⚠️ Seaborn no está instalado.")
    except Exception as e:
        st.warning(f"⚠️ Error al generar heatmap: {e}")


def create_year_growth_chart(yearly_stats: pd.DataFrame):
    """Create yearly growth chart.
    
    Args:
        yearly_stats: Yearly statistics dataframe
    """
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Mean by year
        ax1 = axes[0]
        yearly_stats['mean'].plot(kind='bar', ax=ax1, color=sns.color_palette("mako", len(yearly_stats)))
        ax1.set_title('Promedio de Recuperación por Año', fontweight='bold')
        ax1.set_ylabel('Millones $')
        ax1.set_xlabel('Año')
        for i, v in enumerate(yearly_stats['mean']):
            ax1.text(i, v/1e6, f'${v/1e6:.1f}M', ha='center', va='bottom', fontsize=9)
        
        # Growth rate
        ax2 = axes[1]
        growth_data = yearly_stats['crecimiento'].dropna()
        colors = ['green' if x > 0 else 'red' for x in growth_data]
        growth_data.plot(kind='bar', ax=ax2, color=colors)
        ax2.set_title('Tasa de Crecimiento Interanual (%)', fontweight='bold')
        ax2.set_ylabel('Porcentaje (%)')
        ax2.set_xlabel('Año')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"⚠️ Error al generar gráfico de crecimiento: {e}")


def create_monthly_pattern_chart(historico: pd.DataFrame):
    """Create monthly pattern visualization.
    
    Args:
        historico: Monthly aggregated historical data
    """
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        meses_lbl = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic']
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Monthly average pattern
        ax1 = axes[0]
        monthly_avg = historico.groupby('mes')['monto_total'].mean()
        sns.barplot(x=monthly_avg.index, y=monthly_avg.values/1e6, ax=ax1, 
                   palette="mako")
        ax1.set_xticklabels(meses_lbl)
        ax1.set_title('Patrón Mensual Promedio', fontweight='bold')
        ax1.set_ylabel('Millones $')
        ax1.set_xlabel('Mes')
        
        # Seasonal index
        ax2 = axes[1]
        seasonal_idx = historico.groupby('mes')['monto_total'].mean()
        seasonal_idx = seasonal_idx / seasonal_idx.mean()
        colors = ['green' if x > 1 else 'red' for x in seasonal_idx.values]
        sns.barplot(x=seasonal_idx.index, y=seasonal_idx.values, ax=ax2, palette=colors)
        ax2.axhline(y=1, color='black', linestyle='--', linewidth=1)
        ax2.set_xticklabels(meses_lbl)
        ax2.set_title('Índice Estacional (Promedio = 1)', fontweight='bold')
        ax2.set_ylabel('Índice')
        ax2.set_xlabel('Mes')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"⚠️ Error al generar gráfico de patrones: {e}")


def create_trend_analysis(historico: pd.DataFrame):
    """Create trend analysis visualization.
    
    Args:
        historico: Monthly aggregated historical data
    """
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        from scipy import stats as scipy_stats
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Line plot with trend
        ax1 = axes[0, 0]
        ax1.plot(range(len(historico)), historico['monto_total'].values / 1e6, 
                marker='o', linewidth=2, label='Datos')
        
        # Linear trend
        x = np.arange(len(historico))
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, historico['monto_total'].values)
        trend_line = slope * x + intercept
        ax1.plot(x, trend_line / 1e6, 'r--', linewidth=2, label=f'Tendencia (R²={r_value**2:.3f})')
        ax1.set_title('Análisis de Tendencia', fontweight='bold')
        ax1.set_xlabel('Meses desde inicio')
        ax1.set_ylabel('Millones $')
        ax1.legend()
        
        # 2. Rolling mean
        ax2 = axes[0, 1]
        rolling_mean = historico['monto_total'].rolling(window=3).mean()
        rolling_std = historico['monto_total'].rolling(window=3).std()
        ax2.plot(historico['monto_total'].values / 1e6, label='Original', alpha=0.5)
        ax2.plot(rolling_mean.values / 1e6, label='Media móvil (3 meses)', linewidth=2)
        ax2.fill_between(range(len(historico)), 
                        (rolling_mean - rolling_std).values / 1e6,
                        (rolling_mean + rolling_std).values / 1e6, 
                        alpha=0.2, label='±1 Desv. Est.')
        ax2.set_title('Media Móvil (3 meses)', fontweight='bold')
        ax2.set_xlabel('Meses')
        ax2.set_ylabel('Millones $')
        ax2.legend()
        
        # 3. Distribution histogram
        ax3 = axes[1, 0]
        sns.histplot(historico['monto_total'].values / 1e6, kde=True, ax=ax3, 
                    color=sns.color_palette("mako")[0])
        ax3.axvline(historico['monto_total'].mean() / 1e6, color='red', 
                   linestyle='--', label='Media')
        ax3.axvline(historico['monto_total'].median() / 1e6, color='green', 
                   linestyle='--', label='Mediana')
        ax3.set_title('Distribución de Recuperación Mensual', fontweight='bold')
        ax3.set_xlabel('Millones $')
        ax3.legend()
        
        # 4. Cumulative recovery
        ax4 = axes[1, 1]
        cumulative = historico['monto_total'].cumsum()
        ax4.fill_between(range(len(cumulative)), cumulative.values / 1e6, alpha=0.3)
        ax4.plot(cumulative.values / 1e6, linewidth=2)
        ax4.set_title('Recuperación Acumulada', fontweight='bold')
        ax4.set_xlabel('Meses')
        ax4.set_ylabel('Millones $')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"⚠️ Error al generar análisis de tendencia: {e}")
