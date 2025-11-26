import pandas as pd
import numpy as np
import datetime as dt


def procesar_precios_bonos(df_precios, df_universo):
    """
    Realiza la limpieza integral de la serie de precios de bonos.
    Retorna: DataFrame limpio y un diccionario con métricas del proceso.
    """
    # --- 1. PREPARACIÓN DE TIPOS DE DATOS ---
    print("1. Iniciando conversión de tipos...")
    
    # Copias para no afectar los originales
    precios = df_precios.copy()
    universo = df_universo.copy()
    
    # A) Precios: Índice a Datetime y Valores a Numérico
    precios.index = pd.to_datetime(precios.index, dayfirst=True)
    precios = precios.sort_index()
    precios = precios.apply(pd.to_numeric, errors='coerce')
    
    # B) Universo: Maturity a Datetime
    universo['Maturity'] = pd.to_datetime(universo['Maturity'], errors='coerce')

    # --- 2. FILTRADO DE CALENDARIO (Fines de semana y Festivos) ---
    print("2. Filtrando días no laborables y festivos...")
    
    dims_inicial = precios.shape
    
    # A) Eliminar Sábados (5) y Domingos (6)
    precios = precios[precios.index.dayofweek < 5]
    
    # B) Eliminar filas que estén TOTALMENTE vacías (Festivos comunes)
    precios = precios.dropna(how='all')
    
    dims_trading = precios.shape
    dias_eliminados = dims_inicial[0] - dims_trading[0]

    # --- 3. IMPUTACIÓN ROBUSTA (FFILL + MÁSCARA DE VENCIMIENTO) ---
    print("3. Aplicando Forward Fill con control de Vencimientos...")
    
    # Paso A: Forward Fill "ciego"
    precios_ffill = precios.ffill()
    
    # Paso B: Crear Mapa de Vencimientos para búsqueda rápida
    # Intentamos alinear índices o usar columna Description
    if set(precios.columns).issubset(set(universo.index)):
        map_vencimientos = universo['Maturity'].to_dict()
    elif 'Description' in universo.columns:
        # Si el identificador está en la columna Description
        # Aseguramos que no haya duplicados que rompan el to_dict
        temp_uni = universo.drop_duplicates(subset='Description').set_index('Description')
        map_vencimientos = temp_uni['Maturity'].to_dict()
    else:
        print("ADVERTENCIA: No se pudo alinear columnas. Se omite el corte por vencimiento.")
        map_vencimientos = {}

    # Paso C: Aplicar la Máscara de Vencimiento
    precios_clean = precios_ffill.copy()
    datos_zombis_eliminados = 0
    
    # Iteramos columnas
    for bono in precios_clean.columns:
        fecha_vencimiento = map_vencimientos.get(bono)
        
        # Si tenemos fecha y es válida
        if pd.notna(fecha_vencimiento):
            mask_zombie = precios_clean.index > fecha_vencimiento
            
            # Contamos cuántos datos eran "zombis" (no nulos antes de borrar)
            datos_zombis_eliminados += precios_clean.loc[mask_zombie, bono].notna().sum()
            
            # Matamos al zombie (NaN)
            precios_clean.loc[mask_zombie, bono] = np.nan

    # --- 4. RESUMEN FINAL ---
    resumen = {
        'Filas Originales': dims_inicial[0],
        'Filas Finales (Trading Days)': dims_trading[0],
        'Días Eliminados': dias_eliminados,
        'Celdas Zombie Corregidas': datos_zombis_eliminados, # Clave corregida (sin comillas internas)
        'Nulos restantes': precios_clean.isnull().sum().sum(),
        'Forma Final': precios_clean.shape
    }
    
    return precios_clean, resumen



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# 1. HERRAMIENTAS COMUNES (HELPER FUNCTIONS)
# ==============================================================================

def limpieza_base_timeseries(df_input, nombre="DataFrame"):
    """
    Realiza la higiene básica de series temporales:
    1. Índice a Datetime.
    2. Orden cronológico.
    3. Conversión a numérico.
    4. Eliminación de fines de semana y festivos vacíos.
    """
    print(f"   -> Iniciando limpieza base para: {nombre}...")
    df = df_input.copy()
    
    # 1. Ajuste del Índice
    if 'Date' in df.columns:
        df = df.set_index('Date')
    df.index = pd.to_datetime(df.index, dayfirst=True)
    df = df.sort_index()
    
    # 2. Conversión a Numérico
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # 3. Filtrado de Calendario (Trading Days Only)
    df = df[df.index.dayofweek < 5] # Quitamos Sáb/Dom
    df = df.dropna(how='all')       # Quitamos festivos vacíos
    
    print(f"      Dimensión tras limpieza base: {df.shape}")
    return df

# ==============================================================================
# 2. PROCESAMIENTO ESPECÍFICO POR DATASET
# ==============================================================================

def procesar_bonos(df_precios, df_universo):
    """
    Procesa precios de bonos: Limpieza base + Ffill + Máscara de Vencimiento.
    """
    print("--- [1/3] Procesando Precios de Bonos ---")
    
    # A) Limpieza Base
    precios = limpieza_base_timeseries(df_precios, "Bonos Históricos")
    
    # B) Imputación de iliquidez (Forward Fill)
    precios = precios.ffill()
    
    # C) Máscara de Vencimiento (Anti-Zombies)
    print("      Aplicando máscara de vencimientos...")
    
    # Mapa de vencimientos
    df_universo = df_universo.copy()
    df_universo['Maturity'] = pd.to_datetime(df_universo['Maturity'], errors='coerce')
    
    if set(precios.columns).issubset(set(df_universo.index)):
        map_vencimientos = df_universo['Maturity'].to_dict()
    elif 'Description' in df_universo.columns:
        temp = df_universo.drop_duplicates(subset='Description').set_index('Description')
        map_vencimientos = temp['Maturity'].to_dict()
    else:
        map_vencimientos = {}

    # Aplicar corte
    zombis_count = 0
    for bono in precios.columns:
        fecha_vencimiento = map_vencimientos.get(bono)
        if pd.notna(fecha_vencimiento):
            mask_zombie = precios.index > fecha_vencimiento
            if mask_zombie.any():
                zombis_count += precios.loc[mask_zombie, bono].notna().sum()
                precios.loc[mask_zombie, bono] = np.nan
                
    print(f"      Datos 'Zombie' eliminados: {zombis_count}")
    return precios

def procesar_mercado(df_mercado, df_bonos_referencia):
    """
    Procesa precios de mercado: Limpieza base + Alineación estricta con Bonos.
    """
    print("--- [2/3] Procesando Variables de Mercado ---")
    
    # A) Limpieza Base
    mercado = limpieza_base_timeseries(df_mercado, "Variables de Mercado")
    
    # B) Alineación (Intersección de fechas)
    print("      Alineando fechas con el DataFrame de Bonos...")
    fechas_comunes = df_bonos_referencia.index.intersection(mercado.index)
    mercado_aligned = mercado.loc[fechas_comunes]
    
    return mercado_aligned

def procesar_curva_estr(df_input):
    """
    Procesa Curva ESTR: 
    1. Limpieza de fechas.
    2. Expansión a FRECUENCIA DIARIA DE CALENDARIO ('D').
       (Incluye Sábados y Domingos para facilitar la valoración posterior).
    3. Interpolación Log-Linear y Backfill.
    """
    print("--- [3/3] Procesando Curva ESTR (Calendario Completo) ---")
    
    # A) Limpieza inicial
    nodos_originales = df_input.copy()
    if 'Date' in nodos_originales.columns:
        nodos_originales = nodos_originales.set_index('Date')
        
    nodos_originales.index = pd.to_datetime(nodos_originales.index, dayfirst=True)
    nodos_originales = nodos_originales.sort_index()
    
    # B) Resampling a Días NATURALES ('D') en lugar de Business Days ('B')
    # Esto crea filas para sábados, domingos y festivos con NaN
    curva_diaria = nodos_originales.asfreq('D')
    
    # C) Interpolación (Linear + Backfill)
    # Al usar method='time', Python calcula el valor exacto para el sábado y domingo
    # basándose en la distancia entre el viernes y el lunes.
    cols_tasas = ['Market Rate', 'Zero Rate']
    for col in cols_tasas:
        if col in curva_diaria.columns:
            curva_diaria[col] = curva_diaria[col].interpolate(method='time').bfill().ffill()

    # D) Interpolación de Descuentos (Log-Linear)
    if 'Discount' in curva_diaria.columns:
        # 1. Logaritmo
        curva_diaria['Log_Discount'] = np.log(curva_diaria['Discount'])
        # 2. Interpolación Temporal (rellena fines de semana correctamente)
        curva_diaria['Log_Discount'] = curva_diaria['Log_Discount'].interpolate(method='time').bfill().ffill()
        # 3. Exponencial
        curva_diaria['Discount'] = np.exp(curva_diaria['Log_Discount'])
        curva_diaria = curva_diaria.drop(columns=['Log_Discount'])

    print(f"      Curva expandida de {len(nodos_originales)} nodos a {len(curva_diaria)} días naturales.")
    return curva_diaria


# ==============================================================================
# 2. HELPER FUNCTIONS EN VALORACIÓN DE BONOS
# ==============================================================================


# Mantenemos la función auxiliar igual, ya que es puramente matemática
def obtener_factor_descuento(fecha_objetivo, df_curva):
    # (Misma lógica que antes: busca en curva diaria o interpola si es fin de semana)
    if fecha_objetivo in df_curva.index:
        return df_curva.loc[fecha_objetivo, 'Discount']
    
    try:
        idx_pos = df_curva.index.get_indexer([fecha_objetivo], method='nearest')[0]
        fecha_cercana = df_curva.index[idx_pos]
        if fecha_cercana == fecha_objetivo: return df_curva.iloc[idx_pos]['Discount']
        
        # Interpolación Log-Linear robusta
        fechas_num = df_curva.index.to_series().apply(lambda x: x.toordinal()).values
        log_discounts = np.log(df_curva['Discount'].values)
        target_num = fecha_objetivo.toordinal()
        log_disc_interp = np.interp(target_num, fechas_num, log_discounts)
        return np.exp(log_disc_interp)
    except:
        return 1.0

# --- NUEVA VERSIÓN: Retorna pd.Series ---
def valorar_bono_pandas(row, df_curva, spread_bps=0, fecha_val=pd.Timestamp('2025-10-01')):
    """
    Calcula la valoración devolviendo una Serie.
    Versión robusta para dtype='object'.
    """
    salida_error = pd.Series({
        'Precio Sucio': np.nan, 'Cupón Corrido': np.nan, 
        'Precio Limpio': np.nan, 'Error': None
    })
    
    try:
        # --- A. Datos y Limpieza de Tipos ---
        nominal = 100.0
        spread = spread_bps / 10000.0
        
        # 1. CUPÓN: Forzamos conversión a float por si viene como string
        try:
            val_cupon = float(row['Coupon'])
        except (ValueError, TypeError):
            val_cupon = 0.0 # Fallback si no hay cupón
            
        coupon_rate = val_cupon / 100.0
        
        # 2. FRECUENCIA: Manejo robusto de strings '1', '2', '4' o enteros
        raw_freq = row['Coupon Frequency']
        # Mapeo seguro str/int -> int
        freq_map = {
            '1': 1, '2': 2, '4': 4, '12': 12,
            1: 1, 2: 2, 4: 4, 12: 12,
            1.0: 1, 2.0: 2, 4.0: 4
        }
        freq = freq_map.get(raw_freq, 1) # Por defecto anual si falla
        
        # --- B. Fechas (Maturity y Call) ---
        # Usamos errors='coerce' para que si la fecha es basura ('-') se convierta en NaT
        maturity = pd.to_datetime(row['Maturity'], dayfirst=True, errors='coerce')
        next_call = pd.to_datetime(row['Next Call Date'], dayfirst=True, errors='coerce')
        
        # Callable: A veces viene como 'Y'/'N', a veces True/False
        val_callable = row['Callable']
        es_callable = (val_callable == 'Y') or (val_callable == True)
        
        # Lógica de Vencimiento Efectivo
        fecha_fin = maturity
        
        # Regla: Si es callable y tiene fecha, cortamos ahí.
        # Si es perpetuo (maturity NaT) y tiene call, cortamos ahí.
        if es_callable and pd.notna(next_call): 
            fecha_fin = next_call
        elif pd.isna(maturity) and pd.notna(next_call): 
            fecha_fin = next_call
            
        # Check de seguridad: Si después de todo no hay fecha o ya venció
        if pd.isna(fecha_fin) or fecha_fin <= fecha_val:
            salida_error['Error'] = 'Vencido/Sin Fecha'
            return salida_error

        # --- C. Flujos (Backward) ---
        flujos = []
        fecha_cursor = fecha_fin
        
        # Generamos fechas hacia atrás
        while fecha_cursor > fecha_val:
            flujos.append(fecha_cursor)
            meses_restar = int(12 / freq)
            # Restar meses de forma segura
            fecha_cursor = fecha_cursor - pd.DateOffset(months=meses_restar)
            
        flujos = sorted(flujos)
        
        # Si no hay flujos futuros (ej. vence mañana), devolver sucio=nominal
        if not flujos:
             # Caso borde: Vence muy pronto
             return pd.Series({
                'Precio Sucio': nominal, 'Cupón Corrido': 0, 
                'Precio Limpio': nominal, 'Error': 'Vence Inmediato'
            })

        # --- D. Pricing ---
        # Fecha inicio del cupón actual
        fecha_inicio_cupon = fecha_cursor
        
        # Cupón Corrido (ACT/365)
        dias_devengados = (fecha_val - fecha_inicio_cupon).days
        cupon_anual_dinero = nominal * coupon_rate
        cup_corrido = cupon_anual_dinero * (dias_devengados / 365.0)
        
        vp_sucio = 0
        for fecha_pago in flujos:
            t_years = (fecha_pago - fecha_val).days / 365.0
            
            # Buscamos el descuento en la curva (usando .loc directo por eficiencia)
            # Usamos try/except por si la fecha se sale del rango de la curva (muy futuro)
            try:
                df_risk_free = df_curva.loc[fecha_pago, 'Discount']
            except KeyError:
                # Flat extrapolation con el último dato disponible
                df_risk_free = df_curva['Discount'].iloc[-1]
            
            # Factor Total
            df_total = df_risk_free * np.exp(-spread * t_years)
            
            # Flujo de caja
            pago = (cupon_anual_dinero / freq)
            if fecha_pago == fecha_fin: 
                pago += nominal
            
            vp_sucio += pago * df_total
            
        return pd.Series({
            'Precio Sucio': vp_sucio,
            'Cupón Corrido': cup_corrido,
            'Precio Limpio': vp_sucio - cup_corrido,
            'Error': None
        })

    except Exception as e:
        salida_error['Error'] = f"Crash: {str(e)}"
        return salida_error
    
    


from scipy import optimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. DEFINICIÓN DE FUNCIONES DEL SOLVER
# ==============================================================================

def objetivo_precio(spread_bps, row, df_curva, precio_target, fecha_val):
    """
    Función de error: Precio Teórico(spread) - Precio Mercado
    """
    # Importante: Asegúrate de que valorar_bono_pandas esté definida en tu entorno
    resultado = tools.valorar_bono_pandas(row, df_curva, spread_bps=spread_bps, fecha_val=fecha_val)
    
    if resultado['Error'] is not None or pd.isna(resultado['Precio Limpio']):
        return np.nan 
        
    return resultado['Precio Limpio'] - precio_target

def calcular_z_spread(row, df_curva, precio_mercado, fecha_val):
    """
    Encuentra el spread que hace que Precio Teórico == Precio Mercado.
    """
    # Filtros básicos de integridad
    if pd.isna(precio_mercado) or precio_mercado <= 1: # Precios <= 1 suelen ser errores o defaults
        return np.nan

    # Lambda para el optimizador
    func = lambda s: objetivo_precio(s, row, df_curva, precio_mercado, fecha_val)
    
    try:
        # Búsqueda de raíces. Ampliamos rango por seguridad (-5000 bps a +20000 bps)
        # Un spread negativo muy grande puede pasar si el bono cotiza muy sobre par.
        spread_impl = optimize.brentq(func, -5000, 20000) 
        return spread_impl
    except (ValueError, RuntimeError):
        # ValueError salta si f(a) y f(b) tienen el mismo signo (la raíz no está en el intervalo)
        return np.nan
    except Exception:
        return np.nan



# ==============================================================================
# 4. VISUALIZACIÓN FASE 1 (UNIVERSO COMPLETO)
# ==============================================================================

def plot_spreads_by_rating(df_data, titulo):
    # 1. Limpieza
    plot_data = df_data.dropna(subset=['Calculated Spread (bps)', 'Rating']).copy()
    
    if plot_data.empty:
        print(f"⚠️ No hay datos para graficar: {titulo}")
        return

    # 2. Orden de Ratings
    orden_ratings = [
        'AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 
        'BBB+', 'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'NR'
    ]
    orden_existente = [r for r in orden_ratings if r in plot_data['Rating'].unique()]
    
    # 3. Plot
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        data=plot_data, 
        x='Rating', 
        y='Calculated Spread (bps)', 
        order=orden_existente,
        palette='RdYlGn_r'
    )
    plt.title(titulo)
    plt.ylabel('Z-Spread (bps)')
    plt.grid(axis='y', alpha=0.3)
    
    # 4. Límite Y inteligente (Evita el error de NaN/Inf)
    q01 = plot_data['Calculated Spread (bps)'].quantile(0.01)
    q99 = plot_data['Calculated Spread (bps)'].quantile(0.99)
    
    # Fallback por si q99 es NaN (aunque con el check empty anterior no debería)
    if pd.isna(q99): q99 = 500
    if pd.isna(q01): q01 = -100
        
    plt.ylim(q01 - 50, q99 * 1.2) 
    plt.show()



