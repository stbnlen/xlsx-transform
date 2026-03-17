import pandas as pd
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
df['AÑO'] = df['FECHA_PAGO'].dt.year
df['MES_NUM'] = df['FECHA_PAGO'].dt.month
df['DIA'] = df['FECHA_PAGO'].dt.day
df['DIA_SEMANA'] = df['FECHA_PAGO'].dt.dayofweek
df['AÑO_MES'] = df['FECHA_PAGO'].dt.to_period('M')

monthly = df.groupby('AÑO_MES').agg(
    monto_total   = ('MONTO', 'sum'),
    num_pagos     = ('MONTO', 'count'),
    monto_prom    = ('MONTO', 'mean'),
    monto_mediana = ('MONTO', 'median'),
    monto_std     = ('MONTO', 'std'),
).reset_index()

monthly['año'] = monthly['AÑO_MES'].dt.year
monthly['mes'] = monthly['AÑO_MES'].dt.month

MES_ACTUAL = monthly.iloc[-1]
historico = monthly.iloc[:-1].copy().reset_index(drop=True)

print('Meses historicos completos: {}'.format(len(historico)))
print('Mes parcial: {} — {} pagos, ${:,.0f}'.format(MES_ACTUAL['AÑO_MES'], MES_ACTUAL['num_pagos'], MES_ACTUAL['monto_total']))
print()
print('First few rows of historico:')
print(historico[['AÑO_MES', 'monto_total', 'num_pagos']].head())