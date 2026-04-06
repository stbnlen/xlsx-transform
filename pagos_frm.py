import streamlit as st
import pandas as pd
import numpy as np
import io
from typing import Optional

from utils import (
    clean_dataframe,
    process_date_columns,
    get_dataframe_info,
    find_amount_column,
    aggregate_monthly,
    calculate_descriptive_stats,
    detect_outliers_iqr,
    test_normality,
    calculate_yearly_stats,
    calculate_monthly_stats,
    calculate_seasonal_indices,
    calculate_correlations,
    create_eda_charts,
    create_seasonal_decomposition_chart,
    create_correlation_heatmap,
    create_year_growth_chart,
    create_monthly_pattern_chart,
    create_trend_analysis,
    HAS_STATSMODELS,
)

# ML imports for prediction
import importlib
import sys
import os

# Check if we're in the venv and add its site-packages to path if needed
venv_path = os.path.join(os.path.dirname(__file__), 'venv', 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
if os.path.exists(venv_path) and venv_path not in sys.path:
    sys.path.insert(0, venv_path)

# ML imports for prediction
try:
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
    HAS_ML_LIBS = True
except ImportError as e:
    # Fallback: try to import without venv path modification
    HAS_ML_LIBS = False
    # st.warning(f"ML libraries not available: {e}")  # Commented out to avoid spam


def show_pagos_frm_view():
    """Display PAGOS_FRM view for data analysis and download."""
    uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx", "xls"], key="pagos_frm_uploader")
    
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        
        # Clean column names
        df_clean = clean_dataframe(df)
        
        # Rename columns to match notebook expectations
        rename_dict = {
            'CLIENTE': 'CLIENTE',
            'CONTRATO': 'CONTRATO',
            'MANDANTE': 'MANDANTE',
            'ESTADO': 'ESTADO',
            'ESTADO 2': 'ESTADO2',
            'FECHA DE PAGO': 'FECHA_PAGO',
            'MONTO PAGADO': 'MONTO',
            'TIPO DE PAGO': 'TIPO_PAGO',
            'Saldo capital': 'SALDO_CAPITAL',
        }
        df_clean = df_clean.rename(columns=rename_dict)
        
        # Process date columns using the known date column name
        df_clean, date_columns = process_date_columns(df_clean, known_date_col='FECHA_PAGO')
        
        info = get_dataframe_info(df_clean)
        
        st.write("**Column Information:**")
        st.text(info['info'])
        
        st.write("**Data Types:**")
        st.write(info['dtypes'])
        
        st.write("**Missing Values:**")
        missing_display = info['missing'][info['missing']['Missing Count'] > 0]
        if len(missing_display) > 0:
            st.dataframe(missing_display)
        else:
            st.write("No missing values found.")
         
        if 'AÑO_MES' in df_clean.columns and date_columns:
            st.write("---")
              
            amount_col = find_amount_column(df_clean)
                
            if 'AÑO' in df_clean.columns and 'MES_NUM' in df_clean.columns and 'MONTO' in df_clean.columns:
                 try:
                     monthly = aggregate_monthly(df_clean, 'MONTO')
                        
                     if monthly is not None and len(monthly) > 0:
                         tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                             "Vista General",
                             "Estadísticas",
                             "Serie Temporal",
                             "Correlaciones",
                             "Ejecutiva",
                             "Comparativo",
                             "Predicción",
                         ])
                         
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
                             _show_analisis_por_ejecutiva(df_clean, monthly)
                         
                         with tab6:
                             _show_analisis_mensual_comparativo(monthly, df_clean)
                         
                         with tab7:
                             _show_prediction_analysis(monthly, df_clean)
                     elif monthly is not None:
                         st.warning("⚠️ No historical data available for detailed analysis")
                     else:
                         st.warning("⚠️ No valid data found after cleaning")
                 except Exception as e:
                     st.error(f"Error in monthly aggregation: {e}")
                     _show_basic_stats_fallback(df_clean, amount_col)
            else:
                st.warning("⚠️ Not enough valid data for monthly aggregation")
        else:
            st.info("ℹ️ Para análisis temporal detallado, asegúrese de que el archivo tenga una columna de fecha")
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False)
        
        st.download_button(
            label="Descargar Archivo Original",
            data=output.getvalue(),
            file_name="frm_2023_2026.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


def _show_monthly_metrics(monthly: pd.DataFrame):
    """Display monthly metrics summary with additional context."""
    MES_ACTUAL = monthly.iloc[-1]
    historico = monthly.iloc[:-1].copy().reset_index(drop=True)
    
    # Calculate additional metrics for context
    if len(historico) > 0:
        avg_monthly_amount = historico['monto_total'].mean()
        avg_monthly_payments = historico['num_pagos'].mean()
        amount_change = ((MES_ACTUAL['monto_total'] - avg_monthly_amount) / avg_monthly_amount * 100) if avg_monthly_amount != 0 else 0
        payments_change = ((MES_ACTUAL['num_pagos'] - avg_monthly_payments) / avg_monthly_payments * 100) if avg_monthly_payments != 0 else 0
    else:
        avg_monthly_amount = avg_monthly_payments = amount_change = payments_change = 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Meses históricos completos", len(historico))
        
    with col2:
        st.metric(
            "Mes parcial actual", 
            f"{MES_ACTUAL['AÑO_MES']}", 
            f"{MES_ACTUAL['num_pagos']:.0f} pagos, ${MES_ACTUAL['monto_total']:,.0f}"
        )
        
    with col3:
        if len(historico) > 0:
            st.metric(
                "Cambio Monto vs Promedio Histórico", 
                f"{amount_change:+.1f}%",
                f"${MES_ACTUAL['monto_total']:,.0f} vs ${avg_monthly_amount:,.0f}"
            )
        else:
            st.metric("Promedio Histórico Monto", "Sin datos")
            
    with col4:
        if len(historico) > 0:
            st.metric(
                "Cambio Pagos vs Promedio Histórico", 
                f"{payments_change:+.1f}%",
                f"{MES_ACTUAL['num_pagos']:.0f} vs {avg_monthly_payments:.0f}"
            )
        else:
            st.metric("Promedio Histórico Pagos", "Sin datos")


def _show_descriptive_stats(monthly: pd.DataFrame):
    """Display descriptive statistics with confidence intervals."""
    st.write("---")
    st.subheader("📈 Estadísticas Descriptivas Detalladas")
    
    historico = monthly.iloc[:-1].copy().reset_index(drop=True)
    y = historico['monto_total'].astype(float).values
    
    if len(y) < 2:
        st.warning("⚠️ Se necesitan al menos 2 puntos de datos para estadísticas descriptivas")
        return
    
    stats_data = calculate_descriptive_stats(y)
    
    # Calculate confidence interval for the mean
    from scipy import stats as scipy_stats
    n = len(y)
    mean = stats_data['mean']
    std = stats_data['std']
    se = std / np.sqrt(n) if n > 0 else 0
    confidence_level = 0.95
    t_critical = scipy_stats.t.ppf((1 + confidence_level) / 2, n - 1) if n > 1 else 0
    margin_of_error = t_critical * se
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.write("**Medidas de Tendencia Central:**")
        st.write(f"• Media: ${mean:,.0f}")
        st.write(f"• Mediana: ${stats_data['median']:,.0f}")
        st.write(f"• IC 95%: [${ci_lower:,.0f}, ${ci_upper:,.0f}]")
    
    with metrics_col2:
        st.write("**Medidas de Dispersión:**")
        st.write(f"• Desv. Estándar: ${std:,.0f}")
        st.write(f"• Coef. Variación: {stats_data['cv']:.1f}%")
        st.write(f"• Rango: ${stats_data['range']:,.0f}")
    
    with metrics_col3:
        st.write("**Medidas de Forma:**")
        st.write(f"• Asimetría: {stats_data['skew']:.3f}")
        st.write(f"• Curtosis: {stats_data['kurtosis']:.3f}")
        
        skew_interp = (
            "Aproximadamente simétrica" if abs(stats_data['skew']) < 0.5
            else "Asimétrica positiva (cola a la derecha)" if stats_data['skew'] > 0.5
            else "Asimétrica negativa (cola a la izquierda)"
        )
        st.write(f"→ {skew_interp}")
        
        # Add kurtosis interpretation
        kurtosis_interp = (
            "Mesocúrtica" if abs(stats_data['kurtosis']) < 0.5
            else "Leptocúrtica" if stats_data['kurtosis'] > 0.5
            else "Platicúrtica"
        )
        st.write(f"• Curtosis: {kurtosis_interp}")
    
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
    st.write("**Detección de Outliers (IQR ×1.5):**")
    
    outlier_info = detect_outliers_iqr(data)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Límite inferior", f"${outlier_info['lower_bound']:,.0f}")
    with col2:
        st.metric("Límite superior", f"${outlier_info['upper_bound']:,.0f}")
    with col3:
        st.metric("Outliers detectados", outlier_info['outlier_count'])
    
    if outlier_info['outlier_count'] == 0:
        st.success("✅ No se detectaron outliers según el método IQR")


def _show_normality_test(data: np.ndarray):
    """Display normality test results."""
    st.write("**Test de Normalidad (Shapiro-Wilk):**")
    
    normality = test_normality(data)
    
    if normality['statistic'] is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estadístico W", f"{normality['statistic']:.4f}")
        with col2:
            st.metric("Valor p", f"{normality['p_value']:.4f}")
        
        if normality['is_normal']:
            st.success("✅ Distribución NORMAL (p > 0.05)")
        else:
            st.warning("⚠️ Distribución NO normal (p ≤ 0.05)")
    else:
        st.warning("⚠️ No se pudo realizar test de normalidad")


def _show_eda_charts(monthly: pd.DataFrame, df_original: pd.DataFrame, amount_col: Optional[str]):
    """Display exploratory data analysis charts."""
    st.write("---")
    st.subheader("📊 Análisis Exploratorio de Datos")
    st.info("""
    **Este conjunto de gráficos muestra:**
    1. **Recupero Mensual**: Barras que muestran el monto total recuperado cada mes (en millones)
    2. **Distribución por Mes**: Diagramas de caja que muestran la variabilidad de los montos por mes
    3. **Comparación Interanual**: Líneas que comparan el comportamiento mensual entre diferentes años
    4. **Distribución por Tipo de Pago**: Barras horizontales que muestran el monto total por categoría de pago
    """)
    if amount_col is not None:
        create_eda_charts(monthly.iloc[:-1].copy(), df_original, amount_col)
    else:
        st.warning("⚠️ No se encontró columna de monto para los gráficos EDA")


def _show_seasonal_analysis(monthly: pd.DataFrame):
    """Display seasonal decomposition analysis with enhanced insights."""
    st.write("---")
    st.subheader("🔄 Descomposición de la Serie Temporal")
    st.info("""
    **Este análisis muestra la descomposición de la serie temporal en:**
    1. **Tendencia**: Dirección general a largo plazo de los datos
    2. **Estacionalidad**: Patrones que se repiten en períodos regulares (mensuales)
    3. **Residuo**: Variación restante después de eliminar tendencia y estacionalidad
    """)
    
    historico = monthly.iloc[:-1].copy()
    
    # Create seasonal decomposition chart
    decomp = create_seasonal_decomposition_chart(historico)
    
    seasonal_df = calculate_seasonal_indices(historico)
    if seasonal_df is not None:
        st.write("**Índices Estacionales:**")
        st.dataframe(seasonal_df.round(4))
        st.write("**Interpretación:** Valores > 1 indican meses con recuperación superior al promedio")
        
        # Add seasonal strength metrics
        if decomp is not None:
            # Calculate strength of seasonality
            try:
                detrended = decomp.observed - decomp.trend
                seasonal_var = np.var(decomp.seasonal)
                residual_var = np.var(decomp.resid)
                if seasonal_var + residual_var > 0:
                    Fs = seasonal_var / (seasonal_var + residual_var)
                    Fs = max(0, min(1, Fs))  # Bound between 0 and 1
                    st.write(f"**Fuerza de la Estacionalidad (Fs):** {Fs:.3f}")
                    if Fs > 0.6:
                        st.write("→ Alta estacionalidad detectada")
                    elif Fs > 0.3:
                        st.write("→ Estacionalidad moderada detectada")
                    else:
                        st.write("→ Baja estacionalidad detectada")
            except:
                pass  # Skip if calculation fails
    else:
        st.info("ℹ️ No se pudo calcular el descomposición estacional. Se necesitan al menos 2 años de datos mensuales.")


def _show_trend_analysis(monthly: pd.DataFrame):
    """Display trend analysis with additional metrics."""
    st.write("---")
    st.subheader("📈 Análisis de Tendencia")
    st.info("""
    **Este análisis muestra la tendencia de los datos a través de:**
    1. **Línea de Tendencia**: Recta que mejor ajusta los datos para mostrar dirección general
    2. **Media Móvil**: Promedio de los últimos 3 meses para suavizar fluctuaciones
    3. **Distribución**: Histograma que muestra la frecuencia de diferentes valores de recuperación
    4. **Recuperación Acumulada**: Suma progresiva de los montos a lo largo del tiempo
    """)
    
    historico = monthly.iloc[:-1].copy()
    
    if len(historico) < 2:
        st.warning("⚠️ Se necesitan al menos 2 puntos de datos para análisis de tendencia")
        return
    
    # Calculate additional trend metrics
    from scipy import stats as scipy_stats
    import numpy as np
    
    # Prepare data
    y = historico['monto_total'].values
    x = np.arange(len(y))
    
    # Calculate linear trend
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, y)
    
    # Calculate R-squared
    r_squared = r_value ** 2
    
    # Calculate trend significance
    trend_significant = p_value < 0.05
    
    # Calculate percentage change from first to last period
    if len(y) >= 2 and y[0] != 0:
        pct_change = ((y[-1] - y[0]) / y[0]) * 100
    else:
        pct_change = 0
    
    # Display trend metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Tendencia Lineal", 
            f"{slope:,.0f} por mes",
            f"R² = {r_squared:.3f}"
        )
    
    with col2:
        st.metric(
            "Significancia de Tendencia", 
            "Significativa" if trend_significant else "No significativa",
            f"p-value = {p_value:.4f}"
        )
    
    with col3:
        st.metric(
            "Cambio Total", 
            f"{pct_change:+.1f}%",
            f"Del primer al último período"
        )
    
    with col4:
        # Calculate compound monthly growth rate
        if len(y) >= 2 and y[0] > 0:
            cmgr = (y[-1] / y[0]) ** (1/len(y)) - 1
            st.metric(
                "Tasa Crec. Mensual Compuesta", 
                f"{cmgr*100:+.2f}%",
                f"Por período"
            )
        else:
            st.metric("Tasa Crec. Mensual Compuesta", "N/A")
    
    # Create the trend visualization
    create_trend_analysis(historico)


def _show_patterns_analysis(monthly: pd.DataFrame):
    """Display yearly and monthly patterns with enhanced insights."""
    st.write("---")
    st.subheader("📅 Patrones Anuales y Mensuales")
    st.info("""
    **Este análisis muestra los patrones recurrentes en los datos:**
    1. **Estadísticas por Año**: Promedio, desviación estándar, mínimo, máximo y crecimiento interanual
    2. **Crecimiento Interanual**: Porcentaje de cambio año a año en la recuperación promedio
    3. **Patrón Mensual**: Comportamiento típico de cada mes a lo largo de los años (estacionalidad)
    """)
    
    historico = monthly.iloc[:-1].copy().reset_index(drop=True)
    
    if len(historico) < 2:
        st.warning("⚠️ Se necesitan al menos 2 puntos de datos para análisis de patrones")
        return
    
    yearly = calculate_yearly_stats(historico)
    st.write("**Estadísticas por Año:**")
    st.dataframe(yearly.round(2))
    
    create_year_growth_chart(yearly)
    
    # Add yearly insights
    if len(yearly) >= 2:
        latest_year = yearly.iloc[-1]
        prev_year = yearly.iloc[-2] if len(yearly) >= 2 else None
        
        if prev_year is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Crecimiento Anual Promedio", 
                    f"{latest_year['crecimiento']:.1f}%" if not pd.isna(latest_year['crecimiento']) else "N/A"
                )
            with col2:
                st.metric(
                    "Volatilidad Anual (CV)", 
                    f"{latest_year['cv']:.1f}%" if not pd.isna(latest_year['cv']) else "N/A"
                )
            with col3:
                st.metric(
                    "Total Recuperado Año Actual", 
                    f"${latest_year['sum']:,.0f}" if not pd.isna(latest_year['sum']) else "N/A"
                )
    
    monthly_stats = calculate_monthly_stats(historico)
    st.write("**Estadísticas por Mes (Patrón Estacional):**")
    st.dataframe(monthly_stats.round(2))
    
    create_monthly_pattern_chart(historico)
    
    # Add seasonal insights
    if len(monthly_stats) > 0:
        # Find peak and low months
        peak_month = monthly_stats['mean'].idxmax()
        low_month = monthly_stats['mean'].idxmin()
        
        peak_value = monthly_stats.loc[peak_month, 'mean']
        low_value = monthly_stats.loc[low_month, 'mean']
        
        if peak_value > 0 and low_value > 0:
            seasonal_ratio = peak_value / low_value
            st.write(f"**Ratio Estacional (Pico/Valle):** {seasonal_ratio:.2f}x")
            if seasonal_ratio > 2:
                st.write("→ Fuerte variabilidad estacional detectada")
            elif seasonal_ratio > 1.5:
                st.write("→ Variabilidad estacional moderada detectada")
            else:
                st.write("→ Baja variabilidad estacional")


def _show_correlation_analysis(monthly: pd.DataFrame):
    """Display correlation analysis with significance testing."""
    st.write("---")
    st.subheader("🔗 Análisis de Correlaciones")
    st.info("""
    **Este análisis muestra las relaciones entre diferentes variables:**
    1. **Heatmap de Correlación**: Visualización de color que muestra la fuerza y dirección de las relaciones entre variables
    2. **Correlaciones con Monto Total**: Valores numéricos que indican cómo cada variable se relaciona con el monto total recuperado
    """)
    
    if monthly.shape[1] < 2:
        st.info("ℹ️ No hay suficientes columnas numéricas para análisis de correlaciones")
        return
    
    # Find the amount column in monthly data (should be 'monto_total' after aggregation)
    amount_col_monthly = 'monto_total'  # This is the standard name used in aggregate_monthly
    
    if amount_col_monthly not in monthly.columns:
        st.warning(f"⚠️ La columna '{amount_col_monthly}' no está disponible para correlación")
        return
    
    # Create correlation heatmap
    create_correlation_heatmap(monthly)
    
    # Calculate correlations with significance
    corr_df = calculate_correlations(monthly)
    
    if corr_df is not None:
        st.write("**Correlaciones con Monto Total:**")
        
        # Add significance indicators if we can calculate them
        try:
            from scipy import stats as scipy_stats
            import numpy as np
            
            # Prepare data for correlation testing
            numeric_cols = [col for col in monthly.columns if col != amount_col_monthly and 
                            pd.api.types.is_numeric_dtype(monthly[col])]
            
            if len(numeric_cols) > 0 and len(monthly) > 2:
                # Calculate correlations with p-values
                corr_details = []
                for col in numeric_cols:
                    # Remove NaN values for correlation calculation
                    valid_data = monthly[[amount_col_monthly, col]].dropna()
                    if len(valid_data) > 2:
                        corr_val, p_val = scipy_stats.pearsonr(valid_data[amount_col_monthly], valid_data[col])
                        corr_details.append({
                            'Variable': col,
                            'Correlación': f"{corr_val:.3f}",
                            'p-valor': f"{p_val:.4f}",
                            'Significativo': "Sí" if p_val < 0.05 else "No"
                        })
                
                if corr_details:
                    detail_df = pd.DataFrame(corr_details)
                    st.dataframe(detail_df)
                else:
                    st.dataframe(corr_df)
            else:
                st.dataframe(corr_df)
        except:
            # Fallback to original display if significance testing fails
            st.dataframe(corr_df)
    else:
        st.warning("⚠️ No se pudieron calcular correlaciones")


def _show_analisis_por_ejecutiva(df_original: pd.DataFrame, monthly: pd.DataFrame):
    """Display analysis by ejecutiva."""
    st.write("---")
    st.subheader("👥 Análisis por Ejecutiva")
    
    if 'EJECUTIVA' not in df_original.columns:
        st.info("ℹ️ Para análisis por ejecutiva, asegúrese de que el archivo tenga una columna 'EJECUTIVA'")
        return
        
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        # Get current month from monthly data (last row is the current/incomplete month)
        if len(monthly) == 0:
            st.warning("⚠️ No hay datos mensuales disponibles")
            return
            
        current_period = monthly.iloc[-1]['AÑO_MES']
        
        # Extract year and month from the period
        try:
            current_year = int(str(current_period)[:4])
            current_month_num = int(str(current_period)[5:7])
        except (ValueError, IndexError):
            # Fallback: try to get from año and mes columns if they exist
            if 'año' in monthly.columns and 'mes' in monthly.columns:
                current_year = int(monthly.iloc[-1]['año'])
                current_month_num = int(monthly.iloc[-1]['mes'])
            else:
                st.error("⚠️ No se pudo extraer el año y mes de los datos")
                return
        
        # Get the current day of month from the original data for the current year and month
        current_day_data = df_original[
            (df_original['AÑO'] == current_year) & 
            (df_original['MES_NUM'] == current_month_num)
        ]
        if len(current_day_data) > 0:
            current_day_of_month = int(current_day_data['DIA'].max())
        else:
            # If no data for current year/month, fallback to using the month length
            current_day_of_month = monthly.iloc[-1]['dias_mes'] if 'dias_mes' in monthly.columns else 30
        
        st.write(f"**Comparando hasta el día {current_day_of_month} del mes {current_month_num}**")
        
        # Filter current month data (up to current day)
        current_month_data = df_original[
            (df_original['AÑO'] == current_year) & 
            (df_original['MES_NUM'] == current_month_num) &
            (df_original['DIA'] <= current_day_of_month)
        ]
        
        # Filter historical data (same month/day, previous years)
        historical_data = df_original[
            (df_original['MES_NUM'] == current_month_num) &
            (df_original['DIA'] <= current_day_of_month) &
            (df_original['AÑO'] < current_year)
        ]
        
        if len(current_month_data) == 0 and len(historical_data) == 0:
            st.warning("⚠️ No hay datos disponibles para el análisis.")
            return
            
        if len(current_month_data) > 0:
            # Aggregate by ejecutiva for current month
            ejecutivo_current = current_month_data.groupby('EJECUTIVA').agg(
                monto_total=('MONTO', 'sum'),
                num_pagos=('MONTO', 'count')
            ).reset_index()
            
            # Sort by monto_total descending
            ejecutivo_current = ejecutivo_current.sort_values('monto_total', ascending=False)
        else:
            ejecutivo_current = pd.DataFrame(columns=['EJECUTIVA', 'monto_total', 'num_pagos'])
            
        if len(historical_data) > 0:
            # Aggregate by ejecutiva for historical data
            ejecutivo_historical = historical_data.groupby(['AÑO', 'EJECUTIVA']).agg(
                monto_total=('MONTO', 'sum'),
                num_pagos=('MONTO', 'count')
            ).reset_index()
            
            # Calculate historical average by ejecutiva
            ejecutivo_historical_avg = ejecutivo_historical.groupby('EJECUTIVA').agg(
                monto_total=('monto_total', 'mean'),
                num_pagos=('num_pagos', 'mean')
            ).reset_index()
            
            # Sort by monto_total descending
            ejecutivo_historical_avg = ejecutivo_historical_avg.sort_values('monto_total', ascending=False)
        else:
            ejecutivo_historical_avg = pd.DataFrame(columns=['EJECUTIVA', 'monto_total', 'num_pagos'])
        
        # Display metrics for top ejecutivas
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 5 Ejecutivas - Mes Actual (Monto)**")
            if len(ejecutivo_current) > 0:
                top_5_current = ejecutivo_current.head(5)[['EJECUTIVA', 'monto_total', 'num_pagos']]
                top_5_current['monto_total'] = top_5_current['monto_total'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(top_5_current, hide_index=True)
            else:
                st.write("No hay datos para el mes actual")
        
        with col2:
            st.write("**Top 5 Ejecutivas - Promedio Histórico (Monto)**")
            if len(ejecutivo_historical_avg) > 0:
                top_5_historical = ejecutivo_historical_avg.head(5)[['EJECUTIVA', 'monto_total', 'num_pagos']]
                top_5_historical['monto_total'] = top_5_historical['monto_total'].apply(lambda x: f"${x:,.0f}")
                st.dataframe(top_5_historical, hide_index=True)
            else:
                st.write("No hay datos históricos")
        
        # Calculate growth for each ejecutiva present in both periods
        if len(ejecutivo_current) > 0 and len(ejecutivo_historical_avg) > 0:
            ejecutivo_merged = pd.merge(
                ejecutivo_current[['EJECUTIVA', 'monto_total']],
                ejecutivo_historical_avg[['EJECUTIVA', 'monto_total']],
                on='EJECUTIVA',
                suffixes=('_actual', '_historico')
            )
            
            if len(ejecutivo_merged) > 0:
                ejecutivo_merged['crecimiento_pct'] = (
                    (ejecutivo_merged['monto_total_actual'] - ejecutivo_merged['monto_total_historico']) / 
                    ejecutivo_merged['monto_total_historico'] * 100
                )
                
                # Show growth leaders
                st.write("**Crecimiento por Ejecutiva (vs Promedio Histórico)**")
                growth_data = ejecutivo_merged[['EJECUTIVA', 'monto_total_actual', 'monto_total_historico', 'crecimiento_pct']].copy()
                growth_data['monto_total_actual'] = growth_data['monto_total_actual'].apply(lambda x: f"${x:,.0f}")
                growth_data['monto_total_historico'] = growth_data['monto_total_historico'].apply(lambda x: f"${x:,.0f}")
                growth_data['crecimiento_pct'] = growth_data['crecimiento_pct'].apply(lambda x: f"{x:+.1f}%")
                st.dataframe(growth_data.sort_values('crecimiento_pct', ascending=False), hide_index=True)
            else:
                st.info("ℹ️ No hay ejecutivas comunes entre el período actual y histórico para comparar crecimiento.")
        else:
            if len(ejecutivo_current) == 0:
                st.warning("⚠️ No hay datos para el mes actual hasta el día de hoy.")
            if len(ejecutivo_historical_avg) == 0:
                st.warning("⚠️ No hay datos históricos para el mismo período en años anteriores.")
                
        # Create visualization for ejecutivo comparison
        if len(ejecutivo_current) > 0 and len(ejecutivo_historical_avg) > 0:
            st.write("---")
            st.subheader("📊 Comparación Visual por Ejecutiva")
            
            # Prepare data for bar chart - only include ejecutivas with data in current month
            ejecutivos_current = ejecutivo_current['EJECUTIVA'].tolist()
            ejecutivos_all = list(ejecutivos_current)  # Only current month ejecutivas
            
            # Prepare data for bar chart
            comparison_data = pd.DataFrame({
                'Ejecutivo': ejecutivos_all,
                'Mes Actual': [ejecutivo_current.set_index('EJECUTIVA').loc[ej, 'monto_total'] for ej in ejecutivos_all],
                'Promedio Histórico': [ejecutivo_historical_avg.set_index('EJECUTIVA').loc[ej, 'monto_total'] if ej in ejecutivo_historical_avg['EJECUTIVA'].values else 0 for ej in ejecutivos_all]
            })
            
            # Sort by current month values descending
            comparison_data = comparison_data.sort_values('Mes Actual', ascending=False)
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=(12, 6))
            
            x = np.arange(len(comparison_data))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, comparison_data['Mes Actual'], width, label='Mes Actual', color='skyblue', edgecolor='navy')
            bars2 = ax.bar(x + width/2, comparison_data['Promedio Histórico'], width, label='Promedio Histórico', color='lightcoral', edgecolor='darkred')
            
            ax.set_xlabel('Ejecutiva', fontsize=12)
            ax.set_ylabel('Monto Total ($)', fontsize=12)
            ax.set_title('Comparación de Pagos por Ejecutiva: Mes Actual vs Promedio Histórico', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(comparison_data['Ejecutivo'], rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            def add_value_labels(bars):
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'${height:,.0f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            add_value_labels(bars1)
            add_value_labels(bars2)
            
            plt.tight_layout()
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"Error en el análisis por ejecutiva: {e}")
        st.info("ℹ️ Verifique que los datos tengan el formato correcto")


def _show_analisis_mensual_comparativo(monthly: pd.DataFrame, df_original: pd.DataFrame):
    """Display comparative analysis of current month vs historical data for the same month up to the same day."""
    st.write("---")
    st.subheader("📊 Análisis Mensual Comparativo (Hasta el Día Actual)")
    st.info("""
    **Este análisis compara el mes actual hasta el día de hoy con el mismo período en años anteriores:**
    1. **Valores actuales**: Monto total y número de pagos del mes actual hasta hoy
    2. **Promedio histórico**: Promedio del mismo período (hasta el mismo día) en años anteriores
    3. **Variación porcentual**: Cambio porcentual respecto al promedio histórico
    4. **Tendencia**: Evolución del mismo período a lo largo de los años
    """)
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import numpy as np
        
        # Get current month from monthly data (last row is the current/incomplete month)
        if len(monthly) == 0:
            st.warning("⚠️ No hay datos mensuales disponibles")
            return
            
        current_month_data = monthly.iloc[-1]
        current_period = current_month_data['AÑO_MES']
        st.write(f"**Analizando datos para el período: {current_period}**")
        
        # Extract year and month from the period
        try:
            current_year = int(str(current_period)[:4])
            current_month_num = int(str(current_period)[5:7])
        except (ValueError, IndexError):
            # Fallback: try to get from año and mes columns if they exist
            if 'año' in monthly.columns and 'mes' in monthly.columns:
                current_year = int(monthly.iloc[-1]['año'])
                current_month_num = int(monthly.iloc[-1]['mes'])
            else:
                st.error("⚠️ No se pudo extraer el año y mes de los datos")
                return
        
        # Get the current day of month from the original data for the current year and month
        current_day_data = df_original[
            (df_original['AÑO'] == current_year) & 
            (df_original['MES_NUM'] == current_month_num)
        ]
        if len(current_day_data) > 0:
            current_day_of_month = int(current_day_data['DIA'].max())
        else:
            # If no data for current year/month (should not happen), fallback to using the month length
            current_day_of_month = monthly.iloc[-1]['dias_mes'] if 'dias_mes' in monthly.columns else 30
        
        st.write(f"**Comparando hasta el día {current_day_of_month} del mes {current_month_num}**")
        
        # Filter historical data (excluding current year) for same month and day <= current_day_of_month
        historical_mask = (
            (df_original['MES_NUM'] == current_month_num) &
            (df_original['DIA'] <= current_day_of_month) &
            (df_original['AÑO'] < current_year)
        )
        
        historical_data = df_original[historical_mask]
        
        if len(historical_data) == 0:
            st.warning("⚠️ No hay datos históricos disponibles para el mismo período (hasta el día actual) en años anteriores.")
            # Still show current month data
            _show_current_month_only(current_month_data)
            return
        
        # Aggregate by year to get yearly totals up to the current day
        historical_yearly = historical_data.groupby('AÑO').agg(
            monto_total=('MONTO', 'sum'),
            num_pagos=('MONTO', 'count')
        ).reset_index()
        
        if len(historical_yearly) == 0:
            st.warning("⚠️ No se pudieron calcular agregados anuales históricos.")
            _show_current_month_only(current_month_data)
            return
        
        # Calculate metrics
        current_amount = current_month_data['monto_total']
        current_payments = current_month_data['num_pagos']
        
        historical_avg_amount = historical_yearly['monto_total'].mean()
        historical_avg_payments = historical_yearly['num_pagos'].mean()
        
        # Calculate year-over-year change (if we have previous year data)
        previous_year_data = historical_yearly[historical_yearly['AÑO'] == current_year - 1]
        yoy_change_amount = None
        yoy_change_payments = None
        
        if len(previous_year_data) > 0:
            prev_year_amount = previous_year_data.iloc[0]['monto_total']
            prev_year_payments = previous_year_data.iloc[0]['num_pagos']
            
            if prev_year_amount != 0:
                yoy_change_amount = ((current_amount - prev_year_amount) / prev_year_amount) * 100
            if prev_year_payments != 0:
                yoy_change_payments = ((current_payments - prev_year_payments) / prev_year_payments) * 100
        
        # Calculate percentage change from historical average
        pct_change_amount = ((current_amount - historical_avg_amount) / historical_avg_amount * 100) if historical_avg_amount != 0 else 0
        pct_change_payments = ((current_payments - historical_avg_payments) / historical_avg_payments * 100) if historical_avg_payments != 0 else 0
        
        # Calculate statistical significance of the difference from historical average (using yearly values)
        try:
            from scipy import stats as scipy_stats
            # Perform t-test comparing current month to historical yearly same-month-up-to-day data
            if len(historical_yearly) > 1:
                # For amount
                t_stat_amount, p_val_amount = scipy_stats.ttest_1samp(
                    historical_yearly['monto_total'].values, 
                    current_amount
                )
                # For payments
                t_stat_payments, p_val_payments = scipy_stats.ttest_1samp(
                    historical_yearly['num_pagos'].values, 
                    current_payments
                )
                
                # Determine significance
                sig_amount = p_val_amount < 0.05
                sig_payments = p_val_payments < 0.05
            else:
                p_val_amount = p_val_payments = 1.0
                sig_amount = sig_payments = False
        except:
            p_val_amount = p_val_payments = 1.0
            sig_amount = sig_payments = False
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Monto Actual", 
                f"${current_amount:,.0f}",
                f"{pct_change_amount:+.1f}% vs promedio histórico"
            )
            # Add significance indicator
            if len(historical_yearly) > 1:
                sig_text = " (significativo)" if sig_amount else " (no significativo)"
                st.caption(f"p-valor: {p_val_amount:.4f}{sig_text}")
            
        with col2:
            st.metric(
                "Pagos Actuales", 
                f"{current_payments:,.0f}",
                f"{pct_change_payments:+.1f}% vs promedio histórico"
            )
            # Add significance indicator
            if len(historical_yearly) > 1:
                sig_text = " (significativo)" if sig_payments else " (no significativo)"
                st.caption(f"p-valor: {p_val_payments:.4f}{sig_text}")
            
        with col3:
            if yoy_change_amount is not None:
                st.metric(
                    "Cambio Anual Monto", 
                    f"{yoy_change_amount:+.1f}%",
                    f"vs {current_year-1}"
                )
            else:
                st.metric(
                    "Promedio Histórico Monto", 
                    f"${historical_avg_amount:,.0f}"
                )
                
        with col4:
            if yoy_change_payments is not None:
                st.metric(
                    "Cambio Anual Pagos", 
                    f"{yoy_change_payments:+.1f}%",
                    f"vs {current_year-1}"
                )
            else:
                st.metric(
                    "Promedio Histórico Pagos", 
                    f"{historical_avg_payments:,.0f}"
                )
        
        # Create visualization - Separate charts for amount and payments for clarity
        fig = plt.figure(figsize=(16, 12))
        
        # Create a grid: 3 rows, 2 columns
        # Row 0: Comparison charts (Actual vs Historical Average)
        # Row 1: Trend charts (Historical Evolution)
        # Row 2: Waterfall and Distribution charts
        
        # 1. Comparison: Actual vs Historical Average - Amount (top left)
        ax1 = plt.subplot(3, 2, 1)
        comparison_data = [current_amount, historical_avg_amount]
        comparison_labels = ['Actual', 'Promedio Histórico']
        bars1 = ax1.bar(comparison_labels, comparison_data, color=['skyblue', 'lightcoral'], alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Monto Total ($)', fontsize=12)
        ax1.set_title('Monto Total: Actual vs Promedio Histórico', fontweight='bold', fontsize=14)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars1, comparison_data):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(comparison_data),
                    f'${value:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Comparison: Actual vs Historical Average - Payments (top right)
        ax2 = plt.subplot(3, 2, 2)
        comparison_data_p = [current_payments, historical_avg_payments]
        bars2 = ax2.bar(comparison_labels, comparison_data_p, color=['lightgreen', 'orange'], alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Número de Pagos', fontsize=12)
        ax2.set_title('Número de Pagos: Actual vs Promedio Histórico', fontweight='bold', fontsize=14)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars2, comparison_data_p):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01*max(comparison_data_p),
                    f'{value:,.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Historical Trend - Amount (middle left)
        ax3 = plt.subplot(3, 2, 3)
        if len(historical_yearly) > 0:
            # Sort by year for proper line chart
            historical_yearly_sorted = historical_yearly.sort_values('AÑO')
            años = historical_yearly_sorted['AÑO'].tolist()
            montos = historical_yearly_sorted['monto_total'].tolist()
            
            ax3.plot(años, montos, 'b-o', linewidth=2.5, markersize=6, label='Monto Total')
            ax3.set_xlabel('Año', fontsize=12)
            ax3.set_ylabel('Monto Total ($)', fontsize=12, color='b')
            ax3.tick_params(axis='y', labelcolor='b')
            ax3.set_title(f'Tendencia Histórica del Monto - Mes {current_month_num}', fontweight='bold', fontsize=14)
            ax3.set_xticks(años)
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
            
            # Highlight current year point if it exists in historical data
            current_in_historical = historical_yearly[historical_yearly['AÑO'] == current_year]
            if len(current_in_historical) > 0:
                ax3.plot(current_year, current_in_historical.iloc[0]['monto_total'], 'b*', markersize=15, label='Actual')
            
            ax3.legend(loc='best')
        else:
            ax3.text(0.5, 0.5, 'No hay datos históricos suficientes\npara mostrar tendencia', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title(f'Tendencia Histórica del Monto - Mes {current_month_num}', fontweight='bold', fontsize=14)
        
        # 4. Historical Trend - Payments (middle right)
        ax4 = plt.subplot(3, 2, 4)
        if len(historical_yearly) > 0:
            # Sort by year for proper line chart
            historical_yearly_sorted = historical_yearly.sort_values('AÑO')
            años = historical_yearly_sorted['AÑO'].tolist()
            pagos = historical_yearly_sorted['num_pagos'].tolist()
            
            ax4.plot(años, pagos, 'r-s', linewidth=2.5, markersize=6, label='Número de Pagos')
            ax4.set_xlabel('Año', fontsize=12)
            ax4.set_ylabel('Número de Pagos', fontsize=12, color='r')
            ax4.tick_params(axis='y', labelcolor='r')
            ax4.set_title(f'Tendencia Histórica de Pagos - Mes {current_month_num}', fontweight='bold', fontsize=14)
            ax4.set_xticks(años)
            ax4.tick_params(axis='x', rotation=45)
            ax4.grid(True, alpha=0.3)
            
            # Highlight current year point if it exists in historical data
            current_in_historical = historical_yearly[historical_yearly['AÑO'] == current_year]
            if len(current_in_historical) > 0:
                ax4.plot(current_year, current_in_historical.iloc[0]['num_pagos'], 'r*', markersize=15, label='Actual')
            
            ax4.legend(loc='best')
        else:
            ax4.text(0.5, 0.5, 'No hay datos históricos suficientes\npara mostrar tendencia', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title(f'Tendencia Histórica de Pagos - Mes {current_month_num}', fontweight='bold', fontsize=14)
        
        # 5. Waterfall Chart - Amount Changes (bottom left)
        ax5 = plt.subplot(3, 2, 5)
        change_amount = current_amount - historical_avg_amount
        
        # Data for waterfall (Amount)
        categories_waterfall_a = ['Promedio Histórico', 'Cambio', 'Valor Actual']
        values_waterfall_a = [historical_avg_amount, change_amount, current_amount]
        
        # Calculate cumulative positions for waterfall
        cumulative_a = [0]
        for i in range(len(values_waterfall_a)-1):
            cumulative_a.append(cumulative_a[-1] + values_waterfall_a[i])
        
        # Colors: negative = red, positive = green
        colors_waterfall_a = ['gray']  # Starting point
        for i in range(1, len(values_waterfall_a)-1):
            colors_waterfall_a.append('green' if values_waterfall_a[i] >= 0 else 'red')
        colors_waterfall_a.append('blue')  # End point
        
        bars5 = ax5.bar(range(len(categories_waterfall_a)), values_waterfall_a, 
                       color=colors_waterfall_a, alpha=0.8, edgecolor='black')
        
        # Add connector lines for waterfall effect
        for i in range(1, len(bars5)-1):
            ax5.plot([i-1+0.4, i-1+0.6], [cumulative_a[i], cumulative_a[i]], 'k--', alpha=0.5)
            ax5.plot([i-0.4, i-0.4], [cumulative_a[i], cumulative_a[i+1]], 'k--', alpha=0.5)
        
        ax5.set_ylabel('Monto Total ($)', fontsize=12)
        ax5.set_title('Desglose de Cambios en Monto Total', fontweight='bold', fontsize=14)
        ax5.set_xticks(range(len(categories_waterfall_a)))
        ax5.set_xticklabels(categories_waterfall_a, rotation=15)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars5, values_waterfall_a)):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., 
                    height + (0.01 * abs(height) if height >= 0 else -0.01 * abs(height)),
                    f'${value:,.0f}', ha='center', 
                    va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')
        
        # 6. Waterfall Chart - Payment Changes (bottom right)
        ax6 = plt.subplot(3, 2, 6)
        change_payments = current_payments - historical_avg_payments
        
        # Data for waterfall (Payments)
        categories_waterfall_p = ['Promedio Histórico', 'Cambio', 'Valor Actual']
        values_waterfall_p = [historical_avg_payments, change_payments, current_payments]
        
        # Calculate cumulative positions for waterfall
        cumulative_p = [0]
        for i in range(len(values_waterfall_p)-1):
            cumulative_p.append(cumulative_p[-1] + values_waterfall_p[i])
        
        # Colors: negative = red, positive = green
        colors_waterfall_p = ['gray']  # Starting point
        for i in range(1, len(values_waterfall_p)-1):
            colors_waterfall_p.append('green' if values_waterfall_p[i] >= 0 else 'red')
        colors_waterfall_p.append('blue')  # End point
        
        bars6 = ax6.bar(range(len(categories_waterfall_p)), values_waterfall_p, 
                       color=colors_waterfall_p, alpha=0.8, edgecolor='black')
        
        # Add connector lines for waterfall effect
        for i in range(1, len(bars6)-1):
            ax6.plot([i-1+0.4, i-1+0.6], [cumulative_p[i], cumulative_p[i]], 'k--', alpha=0.5)
            ax6.plot([i-0.4, i-0.4], [cumulative_p[i], cumulative_p[i+1]], 'k--', alpha=0.5)
        
        ax6.set_ylabel('Número de Pagos', fontsize=12)
        ax6.set_title('Desglose de Cambios en Número de Pagos', fontweight='bold', fontsize=14)
        ax6.set_xticks(range(len(categories_waterfall_p)))
        ax6.set_xticklabels(categories_waterfall_p, rotation=15)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars6, values_waterfall_p)):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., 
                    height + (0.01 * abs(height) if height >= 0 else -0.01 * abs(height)),
                    f'{value:,.0f}', ha='center', 
                    va='bottom' if height >= 0 else 'top', fontsize=10, fontweight='bold')
        
        plt.tight_layout(pad=3.0)
        st.pyplot(fig)
        
        # Show historical data table for reference
        with st.expander("Ver datos históricos detallados"):
            if len(historical_yearly) > 0:
                display_data = historical_yearly[['AÑO', 'monto_total', 'num_pagos']].copy()
                if 'monto_total' in display_data.columns:
                    display_data['monto_total'] = display_data['monto_total'].apply(lambda x: f"${x:,.0f}" if pd.notnull(x) else "$0")
                display_data = display_data.sort_values('AÑO', ascending=False)
                st.dataframe(display_data)
            else:
                st.write("No hay datos históricos disponibles para este período.")
                
    except Exception as e:
        st.error(f"Error en el análisis mensual comparativo: {e}")
        st.info("ℹ️ Verifique que los datos tengan el formato correcto")


def _show_current_month_only(current_month_data):
    """Helper function to show only current month data when no historical data is available."""
    st.write("### Datos del Mes Actual (Sin datos históricos para comparación)")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Monto Actual", f"${current_month_data['monto_total']:,.0f}")
    with col2:
        st.metric("Pagos Actuales", f"${current_month_data['num_pagos']:,.0f}")
    
    st.info("ℹ️ Se necesitan datos de años anteriores para realizar comparativas históricas.")


def _show_basic_stats_fallback(df: pd.DataFrame, amount_col: str):
    """Show basic statistics when monthly aggregation fails."""
    st.info("ℹ️ Falling back to basic statistics")
    
    numeric_series = pd.to_numeric(df[amount_col], errors='coerce').dropna()
    
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


def crear_features(df_hist):
    """Create lagged features for prediction models (matching notebook approach)."""
    feat = pd.DataFrame(index=df_hist.index)
    
    # Features básicos - use uppercase column names from date processing
    feat['año'] = df_hist['AÑO'] if 'AÑO' in df_hist.columns else df_hist['año'] if 'año' in df_hist.columns else 0
    feat['mes'] = df_hist['MES_NUM'] if 'MES_NUM' in df_hist.columns else df_hist['mes'] if 'mes' in df_hist.columns else 1
    
    # Features de lags (valores anteriores)
    for lag in [1, 2, 3, 6, 12]:
        feat[f'monto_lag_{lag}'] = df_hist['monto_total'].shift(lag)
        feat[f'num_pagos_lag_{lag}'] = df_hist['num_pagos'].shift(lag)
    
    # Medias móviles
    for window in [3, 6, 12]:
        feat[f'monto_ma_{window}'] = df_hist['monto_total'].rolling(window).mean()
        feat[f'num_pagos_ma_{window}'] = df_hist['num_pagos'].rolling(window).mean()
    
    # Desviación estándar móvil
    feat['monto_std_6'] = df_hist['monto_total'].rolling(6).std()
    feat['num_pagos_std_6'] = df_hist['num_pagos'].rolling(6).std()
    
    # Tendencia (diferencia con mes anterior)
    feat['monto_diff_1'] = df_hist['monto_total'].diff(1)
    feat['monto_diff_12'] = df_hist['monto_total'].diff(12)
    
    # Porcentaje de judiciales (puede afectar el monto)
    feat['pct_judicial'] = df_hist['pct_judicial'] if 'pct_judicial' in df_hist.columns else 0
    feat['pct_castigo'] = df_hist['pct_castigo'] if 'pct_castigo' in df_hist.columns else 0
    
    # Días del mes (para estimar recuperación parcial)
    feat['dias_mes'] = df_hist['dias_mes'] if 'dias_mes' in df_hist.columns else 30
    
    return feat


def _show_prediction_analysis(monthly: pd.DataFrame, df_original: pd.DataFrame):
    """Display ML-based prediction for current month payments."""
    if not HAS_ML_LIBS:
        st.warning("⚠️ Librerías de Machine Learning no disponibles para predicciones")
        return
        
    st.write("---")
    st.subheader("🔮 Predicción de Pagos para el Mes Actual")
    st.info("""
    **Este análisis utiliza modelos de Machine Learning para predecir:**
    1. **Pagos esperados para el mes total**: Basado en patrones históricos y tendencias
    2. **Monto esperado para el mes total**: Predicción del valor total a recuperar
    3. **Pagos restantes estimados**: Diferencia entre lo esperado y lo ya registrado
    4. **Monto restante estimado**: Diferencia entre lo esperado y lo ya registrado
    """)
    
    try:
        # Get current month from monthly data (last row is the current/incomplete month)
        if len(monthly) == 0:
            st.warning("⚠️ No hay datos mensuales disponibles")
            return
            
        current_month_period = monthly.iloc[-1]['AÑO_MES']
        # Extract year and month directly from AÑO_MES to avoid dependency on column creation
        try:
            current_year = int(str(current_month_period)[:4])
            current_month_num = int(str(current_month_period)[5:7])
        except (ValueError, IndexError):
            # Fallback: try to get from año and mes columns if they exist
            if 'año' in monthly.columns and 'mes' in monthly.columns:
                current_year = int(monthly.iloc[-1]['año'])
                current_month_num = int(monthly.iloc[-1]['mes'])
            else:
                st.error("⚠️ No se pudo extraer el año y mes de los datos")
                return
        
        st.write(f"**Prediciendo para el mes: {current_month_period}**")
        
        # Prepare data for prediction - use historical monthly data (exclude current month for training)
        # We need at least 2 months of data to make a prediction (1 for training, 1 for the current month)
        if len(monthly) < 2:
            st.warning("⚠️ No hay suficientes datos históricos para generar predicciones confiables")
            return
            
        # Use all but the last row (current month) for training
        df_hist = monthly.iloc[:-1].copy()
        
        if len(df_hist) < 15:  # Need minimum data for features
            st.warning("⚠️ No hay suficientes datos históricos para generar predicciones confiables")
            return
            
        # Ensure required columns exist
        required_columns = ['monto_total', 'num_pagos']
        for col in required_columns:
            if col not in df_hist.columns:
                st.warning(f"⚠️ No se encontró la columna '{col}' para el análisis predictivo")
                return
                
        # Ensure columns are numeric
        df_hist['monto_total'] = pd.to_numeric(df_hist['monto_total'], errors='coerce')
        df_hist['num_pagos'] = pd.to_numeric(df_hist['num_pagos'], errors='coerce')
        
        # Remove rows with missing data
        df_hist = df_hist.dropna(subset=['monto_total', 'num_pagos'])
        
        if len(df_hist) < 15:
            st.warning("⚠️ No hay suficientes datos válidos después de limpiar para predicciones")
            return
            
        # Create features for historical data
        X_hist = crear_features(df_hist)
        
        # Targets
        y_pagos = df_hist['num_pagos'].values
        y_monto = df_hist['monto_total'].values
        
        # Remove rows with NaN (due to lag features)
        valid_idx = X_hist.dropna().index
        if len(valid_idx) < 10:
            st.warning("⚠️ No hay suficientes datos válidos después de crear características para predicciones")
            return
            
        X_clean = X_hist.loc[valid_idx].values
        y_pagos_clean = y_pagos[valid_idx]
        y_monto_clean = y_monto[valid_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        # Create feature vector for prediction (next month base)
        # We create features based on historical data up to the last known month
        pred_features_dict = {
            'año': int(df_hist['año'].iloc[-1]),
            'mes': int(df_hist['mes'].iloc[-1]),
        }
        
        # Add lagged values from historical data
        monto_series = df_hist['monto_total'].values
        num_pagos_series = df_hist['num_pagos'].values
        
        for lag in [1, 2, 3, 6, 12]:
            if len(monto_series) >= lag:
                pred_features_dict[f'monto_lag_{lag}'] = monto_series[-lag]
                pred_features_dict[f'num_pagos_lag_{lag}'] = num_pagos_series[-lag]
            else:
                pred_features_dict[f'monto_lag_{lag}'] = np.nan
                pred_features_dict[f'num_pagos_lag_{lag}'] = np.nan
        
        # Add moving averages
        for window in [3, 6, 12]:
            if len(monto_series) >= window:
                pred_features_dict[f'monto_ma_{window}'] = np.mean(monto_series[-window:])
                pred_features_dict[f'num_pagos_ma_{window}'] = np.mean(num_pagos_series[-window:])
            else:
                pred_features_dict[f'monto_ma_{window}'] = np.nan
                pred_features_dict[f'num_pagos_ma_{window}'] = np.nan
        
        # Add rolling std dev
        if len(monto_series) >= 6:
            pred_features_dict['monto_std_6'] = np.std(monto_series[-6:])
            pred_features_dict['num_pagos_std_6'] = np.std(num_pagos_series[-6:])
        else:
            pred_features_dict['monto_std_6'] = np.nan
            pred_features_dict['num_pagos_std_6'] = np.nan
        
        # Add differences
        if len(monto_series) >= 2:
            pred_features_dict['monto_diff_1'] = monto_series[-1] - monto_series[-2]
        else:
            pred_features_dict['monto_diff_1'] = 0
            
        if len(monto_series) >= 13:
            pred_features_dict['monto_diff_12'] = monto_series[-1] - monto_series[-13]
        else:
            pred_features_dict['monto_diff_12'] = 0
        
        # Add percentages (use last available or default - these would need to be calculated from original data)
        # For now, we'll use default values since we don't have these in the monthly aggregated data
        pred_features_dict['pct_judicial'] = 0.0
        pred_features_dict['pct_castigo'] = 0.0
        pred_features_dict['dias_mes'] = df_hist['dias_mes'].iloc[-1] if 'dias_mes' in df_hist.columns else 30
        
        # Convert to DataFrame and handle NaN
        X_pred = pd.DataFrame([pred_features_dict])
        
        # Fill any NaN values with column means from training data (simple approach)
        for col in X_pred.columns:
            if pd.isna(X_pred[col].iloc[0]):
                # Find corresponding column in X_hist for mean
                hist_col = col
                if hist_col in X_hist.columns:
                    X_pred[col] = X_hist[hist_col].mean()
                else:
                    X_pred[col] = 0
        
        # Ensure column order matches training
        # Get feature names from X_hist (after creating features)
        X_hist_features = crear_features(df_hist)
        feature_names = X_hist_features.columns.tolist()
        # Remove the target columns if they got included
        feature_names = [f for f in feature_names if f not in ['monto_total', 'num_pagos']]
        
        # Reorder X_pred to match
        X_pred = X_pred[feature_names]
        
        # Scale prediction features
        X_pred_scaled = scaler.transform(X_pred)
        
        # Define models (same as notebook)
        modelos_pagos = {
            'XGBoost': xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, random_state=42, verbosity=0),
            'LightGBM': lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, random_state=42, verbose=-1),
            'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=42),
        }
        
        modelos_monto = {
            'XGBoost': xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, random_state=42, verbosity=0),
            'LightGBM': lgb.LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, random_state=42, verbose=-1),
            'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=300, max_depth=4, learning_rate=0.03, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=42),
        }
        
        # Train models and make predictions
        pred_pagos = {}
        pred_monto = {}
        resultados_pagos = {}
        resultados_monto = {}
        
        # Evaluate models for num_pagos (to get weights)
        print("Evaluating models for num_pagos...")
        tscv = TimeSeriesSplit(n_splits=3)  # Use fewer splits for smaller data
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
                'MAE': np.mean(mae_scores),
                'MAPE': np.mean(mape_scores),
                'weights': 1 / (np.mean(mape_scores) + 1)  # Inverse MAPE as weight
            }
        
        # Evaluate models for monto_total (to get weights)
        print("Evaluating models for monto_total...")
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
                'MAE': np.mean(mae_scores),
                'MAPE': np.mean(mape_scores),
                'weights': 1 / (np.mean(mape_scores) + 1)  # Inverse MAPE as weight
            }
        
        # Calculate normalized weights
        total_weight_pagos = sum(r['weights'] for r in resultados_pagos.values())
        total_weight_monto = sum(r['weights'] for r in resultados_monto.values())
        
        for nombre in resultados_pagos:
            resultados_pagos[nombre]['normalized_weight'] = resultados_pagos[nombre]['weights'] / total_weight_pagos
        for nombre in resultados_monto:
            resultados_monto[nombre]['normalized_weight'] = resultados_monto[nombre]['weights'] / total_weight_monto
        
        # Train final models on all data and predict
        print("Training final models...")
        # Store feature importance for each model
        feature_importance_pagos = {}
        feature_importance_monto = {}
        
        for nombre, modelo in modelos_pagos.items():
            modelo.fit(X_scaled, y_pagos_clean)
            pred = modelo.predict(X_pred_scaled)[0]
            pred_pagos[nombre] = max(0, pred)  # Ensure non-negative
            # Extract feature importance if available
            if hasattr(modelo, 'feature_importances_'):
                feature_importance_pagos[nombre] = modelo.feature_importances_
        
        for nombre, modelo in modelos_monto.items():
            modelo.fit(X_scaled, y_monto_clean)
            pred = modelo.predict(X_pred_scaled)[0]
            pred_monto[nombre] = max(0, pred)  # Ensure non-negative
            # Extract feature importance if available
            if hasattr(modelo, 'feature_importances_'):
                feature_importance_monto[nombre] = modelo.feature_importances_
        
        # Calculate ensemble predictions
        pred_pagos_ensemble = sum(pred_pagos[n] * resultados_pagos[n]['normalized_weight'] for n in pred_pagos)
        pred_monto_ensemble = sum(pred_monto[n] * resultados_monto[n]['normalized_weight'] for n in pred_monto)
        
        # Get current month actuals (last row of monthly data)
        payments_so_far = monthly.iloc[-1]['num_pagos']
        amount_so_far = monthly.iloc[-1]['monto_total']
        
        # Calculate remaining
        remaining_pagos = pred_pagos_ensemble - payments_so_far
        remaining_monto = pred_monto_ensemble - amount_so_far
        
        # Display results
        st.write("### 📊 Resultados de la Predicción")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Pagos Registrados", 
                f"{payments_so_far:,.0f}",
                help="Pagos ya registrados en el mes actual"
            )
        
        with col2:
            st.metric(
                "Monto Registrado", 
                f"${amount_so_far:,.0f}",
                help="Monto ya registrado en el mes actual"
            )
        
        with col3:
            st.metric(
                "Pagos Esperados (Total)", 
                f"{pred_pagos_ensemble:,.0f}",
                f"{remaining_pagos:,.0f} restantes",
                help="Predicción total de pagos para el mes"
            )
        
        with col4:
            st.metric(
                "Monto Esperado (Total)", 
                f"${pred_monto_ensemble:,.0f}",
                f"${remaining_monto:,.0f} restante",
                help="Predicción total de monto para el mes"
            )
        
        # Model breakdown
        with st.expander("Ver detalles por modelo"):
            st.write("**Predicción de Número de Pagos:**")
            pagos_df = pd.DataFrame({
                'Modelo': list(pred_pagos.keys()),
                'Predicción': [f"{pred_pagos[n]:,.0f}" for n in pred_pagos.keys()],
                'Peso': [f"{resultados_pagos[n]['normalized_weight']:.1%}" for n in pred_pagos.keys()],
                'MAE': [f"{resultados_pagos[n]['MAE']:,.2f}" for n in pred_pagos.keys()],
                'MAPE': [f"{resultados_pagos[n]['MAPE']:.2f}%" for n in pred_pagos.keys()]
            })
            st.dataframe(pagos_df)
            
            # Explanation of metrics
            st.info("""
            **Entendiendo las métricas:**
            - **MAE (Error Absoluto Medio)**: Promedio de las diferencias absolutas entre las predicciones y los valores reales. Indica cuánto se equivoca el modelo en promedio, en las mismas unidades que la variable objetivo (por ejemplo, número de pagos o monto en pesos).
            - **MAPE (Error Porcentual Absoluto Medio)**: Promedio de los errores porcentuales absolutos. Muestra el error como porcentaje del valor real, lo que permite comparar la precisión entre diferentes escalas.
            """)
            
            # Feature importance for pagos
            if feature_importance_pagos:
                st.write("**Importancia de Características (Pagos):**")
                # Get feature names from the cleaned features
                feature_names = crear_features(df_hist).dropna(axis=1, how='all').columns.tolist()
                # Remove target columns if they got included
                feature_names = [f for f in feature_names if f not in ['monto_total', 'num_pagos']]
                
                # Calculate average importance across models
                avg_importance = np.mean([feature_importance_pagos[model] for model in feature_importance_pagos.keys()], axis=0)
                
                # Create dataframe for display and get top 5 features
                importance_df_pagos = pd.DataFrame({
                    'Característica': feature_names,
                    'Importancia': avg_importance
                }).sort_values('Importancia', ascending=False)
                
                # Get top 5 features
                top_features = importance_df_pagos.head(5)
                
                # Define descriptions for each feature
                feature_descriptions = {
                    'monto_lag_1': 'Monto del mes anterior - Indica la tendencia inmediata de recuperación',
                    'monto_lag_2': 'Monto de hace 2 meses - Patrón a corto plazo',
                    'monto_lag_3': 'Monto de hace 3 meses - Influencia trimestral',
                    'monto_lag_6': 'Monto de hace 6 meses - Patrón semestral',
                    'monto_lag_12': 'Monto de hace 12 meses - Estacionalidad anual',
                    'monto_ma_3': 'Media móvil de 3 meses - Tendencia a corto plazo suavizada',
                    'monto_ma_6': 'Media móvil de 6 meses - Tendencia intermedia',
                    'monto_ma_12': 'Media móvil de 12 meses - Tendencia anual',
                    'monto_std_6': 'Desviación estándar de 6 meses - Volatilidad reciente',
                    'monto_diff_1': 'Cambio mensual del monto - Momentum inmediato',
                    'monto_diff_12': 'Cambio anual del monto - Momentum estacional',
                    'num_pagos_lag_1': 'Número de pagos del mes anterior - Volumen inmediato',
                    'num_pagos_lag_2': 'Número de pagos de hace 2 meses - Volumen a corto plazo',
                    'num_pagos_lag_3': 'Número de pagos de hace 3 meses - Volumen trimestral',
                    'num_pagos_lag_6': 'Número de pagos de hace 6 meses - Volumen semestral',
                    'num_pagos_lag_12': 'Número de pagos de hace 12 meses - Volumen anual',
                    'num_pagos_ma_3': 'Media móvil de pagos (3 meses) - Tendencia de volumen',
                    'num_pagos_ma_6': 'Media móvil de pagos (6 meses) - Tendencia de volumen intermedia',
                    'num_pagos_ma_12': 'Media móvil de pagos (12 meses) - Tendencia de volumen anual',
                    'num_pagos_std_6': 'Desviación estándar de pagos (6 meses) - Volatilidad de volumen',
                    'num_pagos_diff_1': 'Cambio mensual en número de pagos - Momentum de volumen',
                    'num_pagos_diff_12': 'Cambio anual en número de pagos - Momentum estacional de volumen',
                    'año': 'Año actual - Factor de tendencia temporal',
                    'mes': 'Mes actual - Factor estacional',
                    'pct_judicial': 'Porcentaje de casos judiciales - Impacto de procesos legales',
                    'pct_castigo': 'Porcentaje de casos con castigo - Impacto de medidas punitivas',
                    'dias_mes': 'Número de días en el mes - Factor de oportunidad de recuperación'
                }
                
                # Create display with descriptions
                display_data = []
                for _, row in top_features.iterrows():
                    feature = row['Característica']
                    importance = row['Importancia']
                    description = feature_descriptions.get(feature, 'Característica derivada de datos históricos')
                    display_data.append({
                        'Característica': feature,
                        'Importancia': f"{importance:.3f}",
                        'Descripción': description
                    })
                
                display_df = pd.DataFrame(display_data)
                st.dataframe(display_df, hide_index=True)
                
                # Add explanation
                st.caption("""Las características con mayor importancia tienen más influencia en las predicciones.
                Los lags muestran valores históricos, las medias móviles muestran tendencias suavizadas,
                y las diferencias muestran cambios momentáneos.""")
            
            st.write("**Predicción de Monto Total:**")
            monto_df = pd.DataFrame({
                'Modelo': list(pred_monto.keys()),
                'Predicción': [f"${pred_monto[n]:,.0f}" for n in pred_monto.keys()],
                'Peso': [f"{resultados_monto[n]['normalized_weight']:.1%}" for n in pred_monto.keys()],
                'MAE': [f"{resultados_monto[n]['MAE']:,.2f}" for n in pred_monto.keys()],
                'MAPE': [f"{resultados_monto[n]['MAPE']:.2f}%" for n in pred_monto.keys()]
            })
            st.dataframe(monto_df)
            
            # Explanation of metrics
            st.info("""
            **Entendiendo las métricas:**
            - **MAE (Error Absoluto Medio)**: Promedio de las diferencias absolutas entre las predicciones y los valores reales. Indica cuánto se equivoca el modelo en promedio, en las mismas unidades que la variable objetivo (por ejemplo, número de pagos o monto en pesos).
            - **MAPE (Error Porcentual Absoluto Medio)**: Promedio de los errores porcentuales absolutos. Muestra el error como porcentaje del valor real, lo que permite comparar la precisión entre diferentes escalas.
            """)
            
            # Feature importance for monto
            if feature_importance_monto:
                st.write("**Importancia de Características (Monto):**")
                # Get feature names from the cleaned features
                feature_names = crear_features(df_hist).dropna(axis=1, how='all').columns.tolist()
                # Remove target columns if they got included
                feature_names = [f for f in feature_names if f not in ['monto_total', 'num_pagos']]
                
                # Calculate average importance across models
                avg_importance = np.mean([feature_importance_monto[model] for model in feature_importance_monto.keys()], axis=0)
                
                # Create dataframe for display and get top 5 features
                importance_df_monto = pd.DataFrame({
                    'Característica': feature_names,
                    'Importancia': avg_importance
                }).sort_values('Importancia', ascending=False)
                
                # Get top 5 features
                top_features = importance_df_monto.head(5)
                
                # Define descriptions for each feature
                feature_descriptions = {
                    'monto_lag_1': 'Monto del mes anterior - Indica la tendencia inmediata de recuperación',
                    'monto_lag_2': 'Monto de hace 2 meses - Patrón a corto plazo',
                    'monto_lag_3': 'Monto de hace 3 meses - Influencia trimestral',
                    'monto_lag_6': 'Monto de hace 6 meses - Patrón semestral',
                    'monto_lag_12': 'Monto de hace 12 meses - Estacionalidad anual',
                    'monto_ma_3': 'Media móvil de 3 meses - Tendencia a corto plazo suavizada',
                    'monto_ma_6': 'Media móvil de 6 meses - Tendencia intermedia',
                    'monto_ma_12': 'Media móvil de 12 meses - Tendencia anual',
                    'monto_std_6': 'Desviación estándar de 6 meses - Volatilidad reciente',
                    'monto_diff_1': 'Cambio mensual del monto - Momentum inmediato',
                    'monto_diff_12': 'Cambio anual del monto - Momentum estacional',
                    'num_pagos_lag_1': 'Número de pagos del mes anterior - Volumen inmediato',
                    'num_pagos_lag_2': 'Número de pagos de hace 2 meses - Volumen a corto plazo',
                    'num_pagos_lag_3': 'Número de pagos de hace 3 meses - Volumen trimestral',
                    'num_pagos_lag_6': 'Número de pagos de hace 6 meses - Volumen semestral',
                    'num_pagos_lag_12': 'Número de pagos de hace 12 meses - Volumen anual',
                    'num_pagos_ma_3': 'Media móvil de pagos (3 meses) - Tendencia de volumen',
                    'num_pagos_ma_6': 'Media móvil de pagos (6 meses) - Tendencia de volumen intermedia',
                    'num_pagos_ma_12': 'Media móvil de pagos (12 meses) - Tendencia de volumen anual',
                    'num_pagos_std_6': 'Desviación estándar de pagos (6 meses) - Volatilidad de volumen',
                    'num_pagos_diff_1': 'Cambio mensual en número de pagos - Momentum de volumen',
                    'num_pagos_diff_12': 'Cambio anual en número de pagos - Momentum estacional de volumen',
                    'año': 'Año actual - Factor de tendencia temporal',
                    'mes': 'Mes actual - Factor estacional',
                    'pct_judicial': 'Porcentaje de casos judiciales - Impacto de procesos legales',
                    'pct_castigo': 'Porcentaje de casos con castigo - Impacto de medidas punitivas',
                    'dias_mes': 'Número de días en el mes - Factor de oportunidad de recuperación'
                }
                
                # Create display with descriptions
                display_data = []
                for _, row in top_features.iterrows():
                    feature = row['Característica']
                    importance = row['Importancia']
                    description = feature_descriptions.get(feature, 'Característica derivada de datos históricos')
                    display_data.append({
                        'Característica': feature,
                        'Importancia': f"{importance:.3f}",
                        'Descripción': description
                    })
                
                display_df = pd.DataFrame(display_data)
                st.dataframe(display_df, hide_index=True)
                
                # Add explanation
                st.caption("""Las características con mayor importancia tienen más influencia en las predicciones.
                Los lags muestran valores históricos, las medias móviles muestran tendencias suavizadas,
                y las diferencias muestran cambios momentáneos.""")
        
        # Progress indicator
        if len(monthly) > 1:
            # Estimate days elapsed (simplified)
            days_in_month = monthly.iloc[-1]['dias_mes'] if 'dias_mes' in monthly.columns else 30
            # This is a simplification - ideally we'd calculate from actual dates
            estimated_progress = min(95.0, (payments_so_far / pred_pagos_ensemble) * 100) if pred_pagos_ensemble > 0 else 0
            st.progress(min(1.0, estimated_progress / 100))
            st.caption(f"Progreso estimado: {estimated_progress:.1f}% completado")
        
    except Exception as e:
        st.error(f"Error en el análisis predictivo: {e}")
        st.info("ℹ️ Verifique que los datos sean suficientes y tengan el formato correcto")