import pandas as pd
import numpy as np
from utils import clean_dataframe, process_date_columns, find_amount_column, aggregate_monthly

# Cargar y preparar datos como lo hace la aplicación
df = pd.read_excel('frm_2023-2026.xlsx', sheet_name='recupero historico')
df.columns = df.columns.str.strip().str.rstrip('.')
df = df.rename(columns={
    'CLIENTE': 'CLIENTE',
    'CONTRATO': 'CONTRATO',
    'MANDANTE': 'MANDANTE',
    'ESTADO': 'ESTADO',
    'FECHA DE PAGO': 'FECHA_PAGO',
    'MONTO PAGADO': 'MONTO',
    'TIPO DE PAGO': 'TIPO_PAGO',
    'Saldo capital': 'SALDO_CAPITAL',
    'ESTADO 2': 'ESTADO2',
})
df = df.drop(columns=[c for c in df.columns if 'Unnamed' in c], errors='ignore')
df['FECHA_PAGO'] = pd.to_datetime(df['FECHA_PAGO'])
df['ESTADO'] = df['ESTADO'].str.strip().str.upper()

print("=== DATOS ORIGINALES ===")
print(f"Filas: {len(df)}")
print(f"Columnas: {list(df.columns)}")
print(f"Rango de fechas: {df['FECHA_PAGO'].min()} a {df['FECHA_PAGO'].max()}")
print(f"MONTO - Min: {df['MONTO'].min()}, Max: {df['MONTO'].max()}, Mean: {df['MONTO'].mean()}")

# Limpiar y procesar
df_clean = clean_dataframe(df)
df_processed, date_columns = process_date_columns(df_clean, known_date_col='FECHA_PAGO')

print("\n=== DESPUÉS DE PROCESAR FECHAS ===")
print(f"Columnas: {list(df_processed.columns)}")
if 'AÑO' in df_processed.columns:
    print(f"AÑO - Min: {df_processed['AÑO'].min()}, Max: {df_processed['AÑO'].max()}")
if 'MES_NUM' in df_processed.columns:
    print(f"MES_NUM - Min: {df_processed['MES_NUM'].min()}, Max: {df_processed['MES_NUM'].max()}")

# Encontrar columna de monto
amount_col = find_amount_column(df_processed)
print(f"\nColumna de monto encontrada: {amount_col}")
if amount_col:
    print(f"Valores en {amount_col} - Min: {df_processed[amount_col].min()}, Max: {df_processed[amount_col].max()}")

# Agrupar mensualmente
if amount_col and 'AÑO' in df_processed.columns and 'MES_NUM' in df_processed.columns:
    monthly = aggregate_monthly(df_processed, amount_col)
    
    if monthly is not None:
        print("\n=== DATOS AGREGADOS MENSUALMENTE ===")
        print(f"Forma: {monthly.shape}")
        print(f"Columnas: {list(monthly.columns)}")
        print("\nPrimeras 5 filas:")
        print(monthly.head())
        print("\nÚltimas 5 filas:")
        print(monthly.tail())
        
        # Verificar si hay valores negativos
        if 'monto_total' in monthly.columns:
            neg_count = (monthly['monto_total'] < 0).sum()
            print(f"\nValores negativos en monto_total: {neg_count}")
            if neg_count > 0:
                print("Filas con valores negativos:")
                print(monthly[monthly['monto_total'] < 0])
                
        # Separar histórico y mes actual
        historico = monthly.iloc[:-1].copy().reset_index(drop=True)
        mes_actual = monthly.iloc[-1]
        
        print(f"\n=== HISTÓRICO ({len(historico)} meses) ===")
        if 'monto_total' in historico.columns:
            print(f"monto_total - Min: {historico['monto_total'].min()}, Max: {historico['monto_total'].max()}")
            
        print(f"\n=== MES ACTUAL ===")
        print(f"AÑO_MES: {mes_actual['AÑO_MES']}")
        if 'monto_total' in mes_actual:
            print(f"monto_total: {mes_actual['monto_total']}")
    else:
        print("ERROR: aggregate_monthly devolvió None")
else:
    print("ERROR: Faltan columnas requeridas para la agregación")