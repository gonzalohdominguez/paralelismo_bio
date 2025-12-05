"""
Imputación de valores faltantes utilizando IterativeImputer con XGBRegressor.

Descripción:
    Este script carga un dataset con valores faltantes y realiza imputación 
    multivariante usando un modelo XGBoost Regressor como estimador base del 
    IterativeImputer. Además, mide el tiempo de ejecución y el uso de memoria, 
    lo que permite estudiar el rendimiento y paralelismo del proceso.

Requisitos:
    Python 3.8+
    pandas
    numpy
    scikit-learn
    xgboost
    openpyxl (para lectura de archivos .xlsx)

Uso:
    python3 imputacion_xgboost.py

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
#           FUNCIÓN PRINCIPAL
# =========================================

def main():
    """
    Flujo principal del script:
        1. Cargar dataset
        2. Configurar imputación con XGBRegressor
        3. Ejecutar imputación
        4. Medir uso de memoria y tiempo
        5. Mostrar resultados
    """

    # ========================
    #    CARGA DEL DATASET
    # ========================

    input_file = "fichas_dm2_0_con_nulos.xlsx"
    print(f"Cargando dataset desde: {input_file}")

    fichas_dm2_0_con_nulos = pd.read_excel(input_file)
    print(f"Dataset cargado. Dimensiones: {fichas_dm2_0_con_nulos.shape}")

    # ========================
    #   MEDICIÓN DE RECURSOS
    # ========================

    tracemalloc.start()
    t0 = time.time()

    # ========================
    #       IMPUTACIÓN
    # ========================

    # Número de árboles del modelo XGBoost
    n_trees = 100

    print(f"Iniciando imputación utilizando XGBoost con {n_trees} árboles...")

    # Configuración del imputador
    imputer = IterativeImputer(
        estimator=XGBRegressor(
            n_estimators=n_trees,
            random_state=42,
            n_jobs=1,        # Ajustar para estudiar paralelismo
        ),
        random_state=42
    )

    # Columnas a imputar
    cols_to_impute = fichas_dm2_0_con_nulos.columns
    data_to_impute = fichas_dm2_0_con_nulos[cols_to_impute]

    # Ejecución de imputación
    imputed_matrix = imputer.fit_transform(data_to_impute)

    # Convertir salida a DataFrame
    fichas_dm2_0_xgb = pd.DataFrame(imputed_matrix, columns=cols_to_impute)

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
    print(f"Tiempo de ejecución total: {t1 - t0:.3f} segundos")
    print(f"Memoria actual usada:     {current_mem / 1024 / 1024:.2f} MB")
    print(f"Memoria pico (peak):      {peak_mem / 1024 / 1024:.2f} MB")

    # ========================
    #     EXPORTACIÓN
    # ========================

    # Guardado opcional (comentado por defecto)
    # output_file = "fichas_xgb_100_trees.xlsx"
    # fichas_dm2_0_xgb.to_excel(output_file, index=False)
    # print(f"Archivo Excel exportado: {output_file}")


# =========================================
#               EJECUCIÓN
# =========================================

if __name__ == "__main__":
    main()