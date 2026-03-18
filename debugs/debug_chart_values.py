import pandas as pd
import numpy as np
from utils import clean_dataframe, process_date_columns, find_amount_column, aggregate_monthly

# Replicate exactly what happens in the app
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

# Clean and process like in the app
df_clean = clean_dataframe(df)
df_processed, date_columns = process_date_columns(df_clean, known_date_col='FECHA_PAGO')
amount_col = find_amount_column(df_processed)

# Aggregate like in the app
if amount_col and 'AÑO' in df_processed.columns and 'MES_NUM' in df_processed.columns:
    monthly = aggregate_monthly(df_processed, amount_col)
    
    if monthly is not None and len(monthly) > 0:
        # This is what gets passed to _show_eda_charts
        historico = monthly.iloc[:-1].copy().reset_index(drop=True)
        
        print("=== VALORES QUE SE PASAN A LA FUNCIÓN DE GRÁFICOS ===")
        print(f"Tipo de datos de monto_total: {historico['monto_total'].dtype}")
        print(f"Valores de monto_total:")
        print(historico['monto_total'].head(10))
        print(f"\nEstadísticas:")
        print(f"Mínimo: {historico['monto_total'].min()}")
        print(f"Máximo: {historico['monto_total'].max()}")
        print(f"Media: {historico['monto_total'].mean()}")
        
        # Check if there are any negative values
        neg_count = (historico['monto_total'] < 0).sum()
        print(f"\nValores negativos: {neg_count}")
        
        if neg_count > 0:
            print("Valores negativos encontrados:")
            print(historico[historico['monto_total'] < 0][['AÑO_MES', 'monto_total']])
        
        # Check what happens when we divide by 1e6 (as done in the charts)
        print(f"\n=== VALORES DESPUÉS DE DIVIDIR POR 1e6 (como en los gráficos) ===")
        monto_en_millones = historico['monto_total'] / 1e6
        print(f"Valores en millones:")
        print(monto_en_millones.head(10))
        print(f"\nEstadísticas en millones:")
        print(f"Mínimo: {monto_en_millones.min()}")
        print(f"Máximo: {monto_en_millones.max()}")
        print(f"Media: {monto_en_millones.mean()}")
        
        # Check if there are any negative values after division
        neg_count_millones = (monto_en_millones < 0).sum()
        print(f"\nValores negativos en millones: {neg_count_millones}")
        
        if neg_count_millones > 0:
            print("Valores negativos en millones encontrados:")
            print(historico[monto_en_millones < 0][['AÑO_MES', 'monto_total']])
    else:
        print("ERROR: monthly is None or empty")
else:
    print("ERROR: Missing required columns for aggregation")