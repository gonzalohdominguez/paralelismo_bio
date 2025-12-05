"""
Imputación secuencial de valores faltantes utilizando IterativeImputer con XGBRegressor.

Descripción:
    Este script realiza imputación multivariante utilizando XGBoost como estimador
    base dentro del IterativeImputer. A diferencia de otros métodos, aquí se imputan
    únicamente las columnas que contienen valores faltantes, lo que reduce
    significativamente el costo computacional y de memoria.

    Además, el script mide:
        - Tiempo total de ejecución
        - Memoria usada
        - Pico máximo de memoria alcanzada

Requisitos:
    Python 3.8+
    pandas
    numpy
    scikit-learn
    xgboost
    openpyxl (para leer .xlsx)

Uso:
    python3 imputacion_xgb_secuencial.py

"""

# =========================================
#               IMPORTS
# =========================================

import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor

import time
import tracemalloc


# =========================================
#              EJECUCIÓN PRINCIPAL
# =========================================

def main():
    """
    Flujo principal del script:
        1. Cargar dataset y detectar columnas con nulos
        2. Configurar imputador basado en XGBoost
        3. Imputar únicamente columnas con valores faltantes
        4. Medir tiempo y uso de memoria
        5. Mostrar resultados
    """

    # ========================
    #    CARGA DEL DATASET
    # ========================

    input_file = "fichas_dm2_0_con_nulos.xlsx"
    print(f"Cargando dataset desde: {input_file}")

    df = pd.read_excel(input_file)
    print(f"Dataset cargado. Dimensiones: {df.shape}")

    # Detectar columnas con valores faltantes
    cols_with_nan = df.columns[df.isna().any()].tolist()
    print(f"Columnas totales: {df.shape[1]}")
    print(f"Columnas con nulos: {len(cols_with_nan)}")

    # ========================
    #   MEDICIÓN DE RECURSOS
    # ========================

    tracemalloc.start()
    t0 = time.time()

    # ========================
    #     IMPUTACIÓN SECUENCIAL
    # ========================

    n_trees = 100  # Ajustable para experimentos

    print(f"Iniciando imputación con XGBoost ({n_trees} árboles)...")

    # Trabajamos solo con columnas que contienen nulos
    df_subset = df[cols_with_nan]

    # Configuración del imputador
    imputer = IterativeImputer(
        estimator=XGBRegressor(
            n_estimators=n_trees,
            random_state=42,
            n_jobs=1,  # Cambiar para estudiar paralelismo
        ),
        random_state=42
    )

    # Ejecución de la imputación
    imputed_array = imputer.fit_transform(df_subset)

    # Reconstruir dataset completo manteniendo columnas originales
    df_imputed = df.copy()
    df_imputed[cols_with_nan] = imputed_array

    # ========================
    #   FIN DE MEDICIÓN
    # ========================

    t1 = time.time()
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # ========================
    #     RESULTADOS
    # ========================

    print("\n===== RESULTADOS =====")
    print(f"Tiempo de ejecución:      {t1 - t0:.3f} segundos")
    print(f"Memoria actual usada:     {current_mem / 1024 / 1024:.2f} MB")
    print(f"Memoria pico (peak):      {peak_mem / 1024 / 1024:.2f} MB")

    # ========================
    #     EXPORTACIÓN
    # ========================

    # output_file = "fichas_xgb_100_est_solo_nulos.xlsx"
    # df_imputed.to_excel(output_file, index=False)
    # print(f"Archivo exportado: {output_file}")


# =========================================
#             PUNTO DE ENTRADA
# =========================================

if __name__ == "__main__":
    main()
