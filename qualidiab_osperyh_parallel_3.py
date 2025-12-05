"""
Imputación paralela de valores faltantes en un dataset usando IterativeImputer + XGBoost.
----------------------------------------------------------------------------------------

Este script:
    - Carga un dataset con valores nulos.
    - Identifica columnas con datos faltantes.
    - Divide esas columnas entre varios procesos (multiprocessing).
    - Cada proceso ejecuta IterativeImputer con XGBRegressor SOLO sobre su subset de columnas.
    - Reconstruye el DataFrame final imputado.
    - Mide tiempo y memoria utilizada.

Requiere:
    - Python 3.8+
    - pandas, numpy
    - scikit-learn
    - xgboost
    - multiprocessing (incluida en Python)

"""

import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor

import time
import tracemalloc
from multiprocessing import Pool


# =============================================================================
#                               CARGA DEL DATASET
# =============================================================================

df = pd.read_excel('fichas_dm2_0_con_nulos.xlsx')

# Identificar columnas con valores nulos
cols_with_nan = df.columns[df.isna().any()].tolist()
num_nan_cols = len(cols_with_nan)

print(f"Columnas totales: {df.shape[1]}")
print(f"Columnas con nulos: {num_nan_cols}")


# =============================================================================
#                        FUNCIÓN DE IMPUTACIÓN (PARALELO)
# =============================================================================
def impute_selected_columns(args):
    """
    Aplica IterativeImputer con un modelo XGBoost a un subset de columnas.

    Parameters
    ----------
    args : tuple
        (df_full, cols_to_impute)
        - df_full: DataFrame completo original.
        - cols_to_impute: lista de columnas que este proceso debe imputar.

    Returns
    -------
    pandas.DataFrame
        DataFrame parcial con SOLO las columnas imputadas por este proceso.
    """
    
    df_full, cols_to_impute = args

    # Copia para seguridad
    df_subset = df_full.copy()

    # Configurar imputador
    imputer = IterativeImputer(
        estimator=XGBRegressor(
            n_estimators=100,
            n_jobs=1,         # importante: cada proceso usa solo 1 core
            random_state=42
        ),
        random_state=42
    )

    # Ajustar únicamente sobre columnas asignadas
    imputed_values = imputer.fit_transform(df_subset[cols_to_impute])

    # Retornar DataFrame con las columnas imputadas
    return pd.DataFrame(imputed_values, columns=cols_to_impute)


# =============================================================================
#                     PARTICIÓN ÓPTIMA ENTRE PROCESOS
# =============================================================================

N_CORES = 2   # Cambiar según hardware disponible: 2, 4, 8...

# Dividir columnas con nulos entre los procesos
col_partitions = np.array_split(cols_with_nan, N_CORES)

print("\nAsignación de columnas por core:")
for i, part in enumerate(col_partitions):
    print(f"Core {i+1}: {list(part)}")


# =============================================================================
#                          EJECUCIÓN EN PARALELO
# =============================================================================
tracemalloc.start()
t0 = time.time()

with Pool(processes=N_CORES) as pool:
    results = pool.map(impute_selected_columns,
                       [(df, list(cols)) for cols in col_partitions])

# Reconstrucción del dataframe imputado
df_imputed = df.copy()
for part_df in results:
    for col in part_df.columns:
        df_imputed[col] = part_df[col]

t1 = time.time()
current_mem, peak_mem = tracemalloc.get_traced_memory()
tracemalloc.stop()


# =============================================================================
#                               RESULTADOS
# =============================================================================
print("\n============ RESULTADOS ============")
print(f"Tiempo de ejecución: {t1 - t0:.3f} segundos")
print(f"Memoria actual usada: {current_mem / 1024 / 1024:.2f} MB")
print(f"Memoria pico (peak): {peak_mem / 1024 / 1024:.2f} MB")

# Guardar resultado si se desea
# df_imputed.to_excel("fichas_xgb_4_cores_balanced_100_est.xlsx", index=False)
