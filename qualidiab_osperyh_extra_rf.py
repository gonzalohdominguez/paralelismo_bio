"""
Script de imputación de valores faltantes utilizando IterativeImputer +
ExtraTreesRegressor.

Descripción:
    Este script carga un dataset con valores faltantes y realiza una imputación
    multivariante empleando un modelo ExtraTreesRegressor dentro de un 
    IterativeImputer. Además, mide el tiempo de ejecución y el uso de memoria
    (actual y pico) para análisis de rendimiento y paralelismo.

Requisitos:
    Python 3.8+
    pandas
    numpy
    scikit-learn
    openpyxl (para lectura de Excel)

Uso:
    python3 imputacion_extratrees.py

"""

# =========================================
#               IMPORTS
# =========================================

import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import ExtraTreesRegressor

import time
import tracemalloc


# =========================================
#           FUNCIÓN PRINCIPAL
# =========================================

def main():
    """
    Ejecuta el flujo completo:
        1. Carga del dataset
        2. Configuración del imputador ExtraTreesRegressor
        3. Medición de tiempo y memoria
        4. Imputación de datos
        5. Impresión de métricas de rendimiento
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

    tracemalloc.start()   # seguimiento de memoria
    t0 = time.time()      # inicio de medición del tiempo

    # ========================
    #       IMPUTACIÓN
    # ========================

    # Número de árboles del ExtraTreesRegressor
    n_trees = 100

    # Parámetros adicionales disponibles para experimentar:
    # max_depth_value = None
    # max_samples_value = None
    # max_features_value = 1
    # min_samples_split_value = 2
    # min_samples_leaf_value = 1
    # n_jobs_value = 2        # Ajustar para paralelismo

    print(f"Iniciando imputación usando ExtraTrees con {n_trees} árboles...")

    imputer = IterativeImputer(
        estimator=ExtraTreesRegressor(
            n_estimators=n_trees,
            random_state=42,
            n_jobs=1   # Cambiar este parámetro para pruebas de paralelismo
        ),
        random_state=42
    )

    # Columnas a imputar
    cols_to_impute = fichas_dm2_0_con_nulos.columns
    data_to_impute = fichas_dm2_0_con_nulos[cols_to_impute]

    # Ejecución de imputación
    imputed_data_matrix = imputer.fit_transform(data_to_impute)

    # Convertir resultado a DataFrame
    fichas_dm2_0_extratrees = pd.DataFrame(
        imputed_data_matrix,
        columns=cols_to_impute
    )

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
    print(f"Tiempo de ejecución total: {t1 - t0:.3f} s")
    print(f"Memoria actual usada:     {current_mem / 1024 / 1024:.2f} MB")
    print(f"Memoria pico (peak):      {peak_mem / 1024 / 1024:.2f} MB")

    # Guardado opcional (comentado)
    # output_file = "fichas_extrarf_100_est.xlsx"
    # fichas_dm2_0_extratrees.to_excel(output_file, index=False)
    # print(f"Archivo guardado: {output_file}")


# =========================================
#               EJECUCIÓN
# =========================================

if __name__ == "__main__":
    main()