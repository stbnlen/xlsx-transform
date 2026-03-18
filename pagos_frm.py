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
            st.subheader("📊 Monthly Aggregation Analysis")
             
            amount_col = find_amount_column(df_clean)
             
            if 'AÑO' in df_clean.columns and 'MES_NUM' in df_clean.columns and 'MONTO' in df_clean.columns:
                try:
                    monthly = aggregate_monthly(df_clean, 'MONTO')
                       
                    if monthly is not None and len(monthly) > 0:
                        _show_monthly_metrics(monthly)
                        _show_descriptive_stats(monthly)
                        _show_eda_charts(monthly, df_clean, amount_col)
                        _show_seasonal_analysis(monthly)
                        _show_trend_analysis(monthly)
                        _show_patterns_analysis(monthly)
                        _show_correlation_analysis(monthly)
                        _show_analisis_mensual_comparativo(monthly, df_clean)
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
    """Display monthly metrics summary."""
    MES_ACTUAL = monthly.iloc[-1]
    historico = monthly.iloc[:-1].copy().reset_index(drop=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Meses históricos completos", len(historico))
    with col2:
        st.metric(
            "Mes parcial actual", 
            f"{MES_ACTUAL['AÑO_MES']}", 
            f"{MES_ACTUAL['num_pagos']:.0f} pagos, ${MES_ACTUAL['monto_total']:,.0f}"
        )


def _show_descriptive_stats(monthly: pd.DataFrame):
    """Display descriptive statistics."""
    st.write("---")
    st.subheader("📈 Estadísticas Descriptivas Detalladas")
    
    historico = monthly.iloc[:-1].copy().reset_index(drop=True)
    y = historico['monto_total'].astype(float).values
    
    stats_data = calculate_descriptive_stats(y)
    
    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
    
    with metrics_col1:
        st.write("**Medidas de Tendencia Central:**")
        st.write(f"• Media: ${stats_data['mean']:,.0f}")
        st.write(f"• Mediana: ${stats_data['median']:,.0f}")
    
    with metrics_col2:
        st.write("**Medidas de Dispersión:**")
        st.write(f"• Desv. Estándar: ${stats_data['std']:,.0f}")
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
    """Display seasonal decomposition analysis."""
    st.write("---")
    st.subheader("🔄 Descomposición de la Serie Temporal")
    st.info("""
    **Este análisis muestra la descomposición de la serie temporal en:**
    1. **Tendencia**: Dirección general a largo plazo de los datos
    2. **Estacionalidad**: Patrones que se repiten en períodos regulares (mensuales)
    3. **Residuo**: Variación restante después de eliminar tendencia y estacionalidad
    """)
    create_seasonal_decomposition_chart(monthly.iloc[:-1].copy())
    
    seasonal_df = calculate_seasonal_indices(monthly.iloc[:-1].copy())
    if seasonal_df is not None:
        st.write("**Índices Estacionales:**")
        st.dataframe(seasonal_df.round(4))
        st.write("**Interpretación:** Valores > 1 indican meses con recuperación superior al promedio")


def _show_trend_analysis(monthly: pd.DataFrame):
    """Display trend analysis."""
    st.write("---")
    st.subheader("📈 Análisis de Tendencia")
    st.info("""
    **Este análisis muestra la tendencia de los datos a través de:**
    1. **Línea de Tendencia**: Recta que mejor ajusta los datos para mostrar dirección general
    2. **Media Móvil**: Promedio de los últimos 3 meses para suavizar fluctuaciones
    3. **Distribución**: Histograma que muestra la frecuencia de diferentes valores de recuperación
    4. **Recuperación Acumulada**: Suma progresiva de los montos a lo largo del tiempo
    """)
    create_trend_analysis(monthly.iloc[:-1].copy())


def _show_patterns_analysis(monthly: pd.DataFrame):
    """Display yearly and monthly patterns."""
    st.write("---")
    st.subheader("📅 Patrones Anuales y Mensuales")
    st.info("""
    **Este análisis muestra los patrones recurrentes en los datos:**
    1. **Estadísticas por Año**: Promedio, desviación estándar, mínimo, máximo y crecimiento interanual
    2. **Crecimiento Interanual**: Porcentaje de cambio año a año en la recuperación promedio
    3. **Patrón Mensual**: Comportamiento típico de cada mes a lo largo de los años (estacionalidad)
    """)
    
    historico = monthly.iloc[:-1].copy().reset_index(drop=True)
    
    yearly = calculate_yearly_stats(historico)
    st.write("**Estadísticas por Año:**")
    st.dataframe(yearly.round(2))
    
    create_year_growth_chart(yearly)
    
    monthly_stats = calculate_monthly_stats(historico)
    st.write("**Estadísticas por Mes (Patrón Estacional):**")
    st.dataframe(monthly_stats.round(2))
    
    create_monthly_pattern_chart(historico)


def _show_correlation_analysis(monthly: pd.DataFrame):
    """Display correlation analysis."""
    st.write("---")
    st.subheader("🔗 Análisis de Correlaciones")
    st.info("""
    **Este análisis muestra las relaciones entre diferentes variables:**
    1. **Heatmap de Correlación**: Visualización de color que muestra la fuerza y dirección de las relaciones entre variables
    2. **Correlaciones con Monto Total**: Valores numéricos que indican cómo cada variable se relaciona con el monto total recuperado
    """)
    
    create_correlation_heatmap(monthly)
    
    corr_df = calculate_correlations(monthly)
    
    if corr_df is not None:
        st.write("**Correlaciones con Monto Total:**")
        st.dataframe(corr_df)
    elif monthly.shape[1] < 2:
        st.info("ℹ️ No hay suficientes columnas numéricas para análisis de correlaciones")
    else:
        st.warning("⚠️ La columna 'monto_total' no está disponible para correlación")






def _show_analisis_mensual_comparativo(monthly: pd.DataFrame, df_original: pd.DataFrame):
    """Display comparative analysis of current month vs historical data for the same month."""
    st.write("---")
    st.subheader("📊 Análisis Mensual Comparativo")
    st.info("""
    **Este análisis compara el mes actual con el mismo mes en años anteriores:**
    1. **Valores actuales**: Monto total y número de pagos del mes actual
    2. **Promedio histórico**: Promedio del mismo mes en años anteriores
    3. **Variación porcentual**: Cambio porcentual respecto al promedio histórico
    4. **Tendencia**: Evolución del mismo mes a lo largo de los años
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
        
        # Get historical data (excluding current month for comparison)
        historico = monthly.iloc[:-1].copy()
        
        if len(historico) == 0:
            st.warning("⚠️ No hay datos históricos disponibles para comparación")
            # Still show current month data
            _show_current_month_only(current_month_data)
            return
        
        # Filter historical data for the same month number
        mismo_mes_historico = historico[historico['mes'] == current_month_num]
        
        if len(mismo_mes_historico) == 0:
            st.warning(f"⚠️ No hay datos históricos para el mes {current_month_num} para comparar")
            # Still show current month data
            _show_current_month_only(current_month_data)
            return
        
        # Calculate metrics
        current_amount = current_month_data['monto_total']
        current_payments = current_month_data['num_pagos']
        
        historical_avg_amount = mismo_mes_historico['monto_total'].mean()
        historical_avg_payments = mismo_mes_historico['num_pagos'].mean()
        
        # Calculate year-over-year change (if we have previous year data)
        previous_year_data = mismo_mes_historico[mismo_mes_historico['año'] == current_year - 1]
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
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Monto Actual", 
                f"${current_amount:,.0f}",
                f"{pct_change_amount:+.1f}% vs promedio histórico"
            )
            
        with col2:
            st.metric(
                "Pagos Actuales", 
                f"{current_payments:,.0f}",
                f"{pct_change_payments:+.1f}% vs promedio histórico"
            )
            
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
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Bar chart: Current vs Historical Average
        ax1 = axes[0, 0]
        categories = ['Monto Total', 'Número de Pagos']
        current_values = [current_amount, current_payments]
        historical_values = [historical_avg_amount, historical_avg_payments]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, current_values, width, label='Actual', color='skyblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, historical_values, width, label='Promedio Histórico', color='lightcoral', alpha=0.8)
        
        ax1.set_title('Comparación: Actual vs Promedio Histórico', fontweight='bold')
        ax1.set_ylabel('Valor')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:,.0f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
        
        autolabel(bars1)
        autolabel(bars2)
        
        # 2. Line chart: Historical trend for same month
        ax2 = axes[0, 1]
        if len(mismo_mes_historico) > 0:
            # Sort by year for proper line chart
            mismo_mes_sorted = mismo_mes_historico.sort_values('año')
            años = mismo_mes_sorted['año'].tolist()
            montos = mismo_mes_sorted['monto_total'].tolist()
            pagos = mismo_mes_sorted['num_pagos'].tolist()
            
            ax2_twin = ax2.twinx()
            
            line1 = ax2.plot(años, montos, 'b-o', label='Monto Total', linewidth=2, markersize=4)
            line2 = ax2_twin.plot(años, pagos, 'r-s', label='Número de Pagos', linewidth=2, markersize=4)
            
            # Highlight current year if it exists in historical data (shouldn't, but just in case)
            current_in_historico = mismo_mes_historico[mismo_mes_historico['año'] == current_year]
            if len(current_in_historico) > 0:
                ax2.plot(current_year, current_in_historico.iloc[0]['monto_total'], 'b*', markersize=12, label='Actual')
                ax2_twin.plot(current_year, current_in_historico.iloc[0]['num_pagos'], 'r*', markersize=12, label='Actual')
            
            ax2.set_xlabel('Año')
            ax2.set_ylabel('Monto Total ($)', color='b')
            ax2_twin.set_ylabel('Número de Pagos', color='r')
            ax2.set_title(f'Tendencia Histórica: Mes {current_month_num}', fontweight='bold')
            
            # Combine legends
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax2_twin.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No hay datos históricos suficientes\npara mostrar tendencia', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'Tendencia Histórica: Mes {current_month_num}', fontweight='bold')
        
        # 3. Waterfall chart showing contributions to change from historical average
        ax3 = axes[1, 0]
        change_amount = current_amount - historical_avg_amount
        change_payments = current_payments - historical_avg_payments
        
        # Data for waterfall
        categories_waterfall = ['Promedio Histórico', 'Cambio Monto', 'Cambio Pagos', 'Valor Actual']
        values_waterfall = [historical_avg_amount, change_amount, change_payments, current_amount]
        
        # Calculate cumulative positions for waterfall
        cumulative = [0]
        for i in range(len(values_waterfall)-1):
            cumulative.append(cumulative[-1] + values_waterfall[i])
        
        # Colors: positive = green, negative = red
        colors_waterfall = ['gray']  # Starting point
        for i in range(1, len(values_waterfall)-1):
            colors_waterfall.append('green' if values_waterfall[i] >= 0 else 'red')
        colors_waterfall.append('blue')  # End point
        
        bars3 = ax3.bar(range(len(categories_waterfall)), values_waterfall, 
                       color=colors_waterfall, alpha=0.7, edgecolor='black')
        
        # Add connector lines for waterfall effect
        for i in range(1, len(bars3)-1):
            ax3.plot([i-1+0.4, i-1+0.6], [cumulative[i], cumulative[i]], 'k--', alpha=0.5)
            ax3.plot([i-0.4, i-0.4], [cumulative[i], cumulative[i+1]], 'k--', alpha=0.5)
        
        ax3.set_title('Desglose de Cambios respecto al Promedio Histórico', fontweight='bold')
        ax3.set_ylabel('Valor')
        ax3.set_xticks(range(len(categories_waterfall)))
        ax3.set_xticklabels(categories_waterfall, rotation=15)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars3, values_waterfall)):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2, 
                    height + (0.01 * abs(height) if height >= 0 else -0.01 * abs(height)),
                    f'{value:,.0f}', ha='center', 
                    va='bottom' if height >= 0 else 'top', fontsize=9)
        
        # 4. Box plot showing distribution of historical data for same month
        ax4 = axes[1, 1]
        if len(mismo_mes_historico) > 0:
            box_data = [mismo_mes_historico['monto_total'].values, mismo_mes_historico['num_pagos'].values]
            box_plot = ax4.boxplot(box_data, labels=['Monto Total', 'Número de Pagos'], patch_artist=True)
            
            # Color the boxes
            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(box_plot['boxes'], colors):
                patch.set_facecolor(color)
            
            # Add current value as a point
            ax4.scatter([1], [current_amount], color='red', s=100, zorder=5, label='Actual (Monto)')
            ax4.scatter([2], [current_payments], color='red', s=100, zorder=5, label='Actual (Pagos)')
            
            ax4.set_title('Distribución Histórica del Mes', fontweight='bold')
            ax4.set_ylabel('Valor')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No hay datos históricos suficientes\npara mostrar distribución', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Distribución Histórica del Mes', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show historical data table for reference
        with st.expander("Ver datos históricos detallados"):
            if len(mismo_mes_historico) > 0:
                display_data = mismo_mes_historico[['año', 'mes', 'monto_total', 'num_pagos']].copy()
                display_data['monto_total'] = display_data['monto_total'].apply(lambda x: f"${x:,.0f}")
                display_data = display_data.sort_values('año', ascending=False)
                st.dataframe(display_data)
            else:
                st.write("No hay datos históricos disponibles para este mes.")
                
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
        for nombre, modelo in modelos_pagos.items():
            modelo.fit(X_scaled, y_pagos_clean)
            pred = modelo.predict(X_pred_scaled)[0]
            pred_pagos[nombre] = max(0, pred)  # Ensure non-negative
        
        for nombre, modelo in modelos_monto.items():
            modelo.fit(X_scaled, y_monto_clean)
            pred = modelo.predict(X_pred_scaled)[0]
            pred_monto[nombre] = max(0, pred)  # Ensure non-negative
        
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
                'Peso': [f"{resultados_pagos[n]['normalized_weight']:.1%}" for n in pred_pagos.keys()]
            })
            st.dataframe(pagos_df)
            
            st.write("**Predicción de Monto Total:**")
            monto_df = pd.DataFrame({
                'Modelo': list(pred_monto.keys()),
                'Predicción': [f"${pred_monto[n]:,.0f}" for n in pred_monto.keys()],
                'Peso': [f"{resultados_monto[n]['normalized_weight']:.1%}" for n in pred_monto.keys()]
            })
            st.dataframe(monto_df)
        
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