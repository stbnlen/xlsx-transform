# Excel Transformador - Filtro de Columnas

Una aplicación de Streamlit que permite a los usuarios subir un archivo Excel, previsualizar su contenido, filtrar columnas específicas según el módulo seleccionado y descargar el resultado filtrado.

## Características

- Navegar entre módulos usando la navegación nativa de páginas de Streamlit
- **Módulo Asignaciones**: Contiene las opciones de filtrado Q_BANCO y Q_CMR
- **Módulo Pagos**: Contiene las opciones de PAGOS_FRM y PAGOS BCI para análisis de datos financieros
- Subir archivos Excel (.xlsx, .xls)
- Previsualizar tanto los datos originales como los procesados en tablas interactivas
- Procesamiento/filtrado automático basado en el modo seleccionado:
  - **Modo Q_BANCO**: Extrae y renombra columnas específicas para operaciones bancarias
  - **Modo Q_CMR**: Filtra a columnas específicas para gestión de cartera comercial
  - **Modo PAGOS_FRM**: Análisis comprehensivo de datos financieros que incluye:
    - Agregación mensual y análisis de tendencias
    - Estadísticas descriptivas y detección de valores atípicos
    - Gráficos de análisis exploratorio de datos
    - Descomposición estacional y análisis de tendencias
    - Análisis de patrones anuales y mensuales
    - Análisis de correlaciones
    - Análisis de desempeño por ejecutiva
    - Capacidades de modelado predictivo
  - **Modo PAGOS BCI**: Filtrado específico de columnas para procesamiento BCI (definidas en pagos_bci.py)
- Descargar archivos Excel procesados
- Construido con Streamlit, pandas, numpy, scipy, scikit-learn, XGBoost, LightGBM, matplotlib y seaborn

## Estructura del Proyecto

- `app.py` - Aplicación principal de Streamlit (página de inicio)
- `pages/asig.py` - Módulo Asignaciones con pestañas Q_BANCO y Q_CMR
- `pages/pagos.py` - Módulo Pagos con pestañas PAGOS_FRM y PAGOS BCI
- `q_banco.py` - Lógica de filtrado para Q_BANCO
- `q_cmr.py` - Lógica de filtrado para Q_CMR
- `pagos_frm.py` - Lógica de filtrado para PAGOS_FRM
- `pagos_bci.py` - Lógica de filtrado para PAGOS BCI
- `utils.py` - Funciones utilitarias para normalización y validación de columnas
- `tests/` - Directorio que contiene pruebas unitarias
- `requirements.txt` - Dependencias de Python
- `AGENTS.md` - Directrices para agentes de IA que trabajan en este proyecto
- `README.md` - Este archivo (versión en inglés)
- `README.es.md` - Este archivo (versión en español)

## Instalación

1. Clone o descargue este repositorio
2. Navegue al directorio del proyecto
3. Instale las dependencias requeridas:

```bash
pip install -r requirements.txt
```

## Uso

Para ejecutar la aplicación localmente:

```bash
streamlit run app.py
```

La aplicación se abrirá en su navegador web predeterminado (usualmente en http://localhost:8501).

### Navegación

Use la barra lateral para navegar entre:
- **Asignaciones**: Contiene los módulos Q_BANCO y Q_CMR
- **Pagos**: Contiene los módulos PAGOS_FRM y PAGOS BCI

### Pasos para usar cada módulo:

1. Seleccione el módulo deseado desde la navegación lateral
2. Dentro de cada módulo, use las pestañas para seleccionar la vista específica (por ejemplo, Q_BANCO o Q_CMR en el módulo Asignaciones)
3. Haga clic en "Examinar archivos" o arrastre y suelte un archivo Excel (.xlsx o .xls) en el área de carga
4. Una vez cargado, verá:
   - Una vista previa de los datos originales con sus dimensiones
   - Si todas las columnas requeridas están presentes para el modo seleccionado, una vista previa de los datos filtrados con sus dimensiones
   - Un botón "Descargar Excel Filtrado" para descargar el resultado
5. Si faltan columnas requeridas, la aplicación mostrará un error con la lista de columnas disponibles

## Explicación de los Módulos

### Módulo Asignaciones

#### Pestaña Q_BANCO
Filtra para mantener estas columnas específicas:
- rut, dv, n_operacion (de n_operacion_principal), origen_core, nombre_completo_cliente, 
- SUCURSAL, CARTERA, ESTADO CRM, ESTADO JUDICIAL, SALDO CAPITAL (de saldo_capital), 
- % DESCUENTO, comuna_particular

#### Pestaña Q_CMR  
Filtra para mantener estas columnas específicas:
- rut, n_operacion_principal, dv, nombre_completo_cliente, CARTERA, CATEGORIA, 
- SUCURSAL, EJECUTIVA ASIGNADA, ESTADO JUDICIAL, DESCUENTO CAMPAÑA, SALDO_DEUDA, TRAMO, estado_cuenta

### Módulo Pagos

#### Pestaña PAGOS_FRM
Filtrado específico de columnas para procesamiento financiero (ver pagos_frm.py para la lista exacta de columnas)

#### Pestaña PAGOS BCI
Filtrado específico de columnas para procesamiento BCI (ver pagos_bci.py para la lista exacta de columnas)

## Dependencias

- streamlit
- pandas
- openpyxl

## Cómo Funciona

La aplicación sigue estos pasos:

1. El usuario selecciona un módulo (Asignaciones o Pagos) mediante la barra de navegación lateral
2. Dentro del módulo seleccionado, el usuario elige una vista específica mediante pestañas
3. El usuario sube un archivo Excel a través de `st.file_uploader`
4. El archivo se lee en un DataFrame de pandas usando `pd.read_excel`
5. El DataFrame original se muestra usando `st.dataframe`
6. La aplicación verifica las columnas requeridas basándose en la vista seleccionada y muestra un error si faltan algunas
7. Si todas las columnas están presentes, filtra el DataFrame para mantener solo las columnas específicas de la vista
8. Para el modo Q_BANCO, se renombran columnas específicas:
   - 'n_operacion_principal' → 'n_operacion'
   - 'saldo_capital' → 'SALDO CAPITAL'
9. El DataFrame filtrado se muestra usando `st.dataframe`
10. El DataFrame filtrado se escribe en un búfer en memoria usando `pd.ExcelWriter` con el motor openpyxl
11. El contenido del búfer se pone a disposición para descarga mediante `st.download_button`

## Notas

- La aplicación valida que todas las columnas requeridas existan en el archivo subido antes de procesar para la vista seleccionada
- Las transformaciones de nombres de columna se aplican solo en el modo Q_BANCO para coincidir exactamente con el formato de salida solicitado
- La aplicación utiliza operaciones en memoria, por lo que no se guardan archivos temporales en disco

## Desarrollo

Para modificar la aplicación:

1. Edite los archivos de vista respectivos (`q_banco.py`, `q_cmr.py`, `pagos_frm.py`, `pagos_bci.py`) para cambiar la lógica de filtrado, la selección de columnas o las definiciones de modo
2. Para cambios estructurales, modifique los archivos de página en el directorio `pages/`
3. Pruebe los cambios localmente con `streamlit run app.py`
4. Asegúrese de que las dependencias estén actualizadas en `requirements.txt`

## Licencia

Este proyecto es de código abierto y está disponible para su modificación y distribución.