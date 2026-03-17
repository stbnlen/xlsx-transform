import pandas as pd
from utils import clean_dataframe, process_date_columns, find_amount_column, aggregate_monthly
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
df_clean = clean_dataframe(df)
df_processed, date_columns = process_date_columns(df_clean, known_date_col='FECHA_PAGO')
amount_col = find_amount_column(df_processed)
print('Amount column found:', amount_col)
if amount_col and 'AÑO' in df_processed.columns and 'MES_NUM' in df_processed.columns:
    monthly = aggregate_monthly(df_processed, amount_col)
    if monthly is not None:
        print('APP STYLE AGGREGATION:')
        print('Shape:', monthly.shape)
        print('Columns:', list(monthly.columns))
        print()
        print('First 5 rows:')
        print(monthly[['AÑO_MES', 'monto_total', 'num_pagos']].head())
        print()
        print('Last 5 rows:')
        print(monthly[['AÑO_MES', 'monto_total', 'num_pagos']].tail())
        
        MES_ACTUAL = monthly.iloc[-1]
        historico = monthly.iloc[:-1].copy().reset_index(drop=True)
        
        print()
        print('Meses históricos completos:', len(historico))
        print('Mes parcial: {} — {} pagos, ${:,.0f}'.format(MES_ACTUAL['AÑO_MES'], MES_ACTUAL['num_pagos'], MES_ACTUAL['monto_total']))
    else:
        print('Monthly aggregation returned None')
else:
    print('Missing required columns for aggregation')