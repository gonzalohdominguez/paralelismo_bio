"""
Script de imputación de valores faltantes utilizando IterativeImputer + RandomForestRegressor.
Descripción:
    Este script carga un dataset con valores nulos, ejecuta un proceso de imputación 
    basado en Random Forest (configurable), mide el tiempo de ejecución y memoria utilizada, 
    y devuelve un DataFrame imputado.

Requisitos:
    - pandas
    - numpy
    - scikit-learn
    - openpyxl (para leer Excel)
    - Python 3.8+

Uso:
    python3 imputacion_rf.py
"""

# =========================================
#               IMPORTS
# =========================================

import pandas as pd
import numpy as np

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

import time
import tracemalloc   # Para medición de memoria


# =========================================
#           FUNCIÓN PRINCIPAL
# =========================================

def main():
    """
    Función principal que ejecuta todo el flujo:
    - Carga del dataset
    - Configuración del modelo de imputación
    - Medición de tiempo y memoria
    - Ejecución de imputación
    - Exportación opcional (comentada)
    """
    
    # ========================
    #    CARGA DEL DATASET
    # ========================
    input_file = "fichas_dm2_0_con_nulos.xlsx"
    print(f"Cargando dataset: {input_file}")

    fichas_dm2_0_con_nulos = pd.read_excel(input_file)
    print(f"Shape del dataset cargado: {fichas_dm2_0_con_nulos.shape}")

    # ========================
    #   MEDICIÓN DE RECURSOS
    # ========================
    tracemalloc.start()
    t0 = time.time()

    # ========================
    #       IMPUTACIÓN
    # ========================

    # Número de árboles (puede modificarse para experimentos paralelos)
    n_trees = 100

    # Ejemplo de parámetros adicionales (pueden habilitarse según la experimentación)
    # max_depth_value = None
    # max_samples_value = None
    # max_features_value = 1
    # min_samples_split_value = 2
    # min_samples_leaf_value = 1
    # n_jobs_value = 1  # Ajustar para paralelización (más CPUs → más rápido, más RAM usada)

    print(f"Iniciando imputación con {n_trees} árboles...")

    # Configurar IterativeImputer con RandomForestRegressor
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(
            n_estimators=n_trees,
            random_state=42,
            n_jobs=1  # IMPORTANTE: modificar para paralelizar
        ),
        random_state=42
    )

    # Selección de columnas a imputar
    cols_to_impute = fichas_dm2_0_con_nulos.columns
    data_to_impute = fichas_dm2_0_con_nulos[cols_to_impute]

    # Ejecutar imputación
    imputed_matrix = imputer.fit_transform(data_to_impute)

    # Convertir a DataFrame imputado
    fichas_dm2_0_rf_estimators = pd.DataFrame(imputed_matrix, columns=cols_to_impute)

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
    print(f"Tiempo de ejecución: {t1 - t0:.3f} segundos")
    print(f"Memoria actual usada: {current_mem / 1024 / 1024:.2f} MB")
    print(f"Memoria pico (peak): {peak_mem / 1024 / 1024:.2f} MB")

    # Guardado opcional
    # output_file = "fichas_rf_100_est.xlsx"
    # fichas_dm2_0_rf_estimators.to_excel(output_file, index=False)
    # print(f"Archivo exportado: {output_file}")


# =========================================
#           EJECUCIÓN
# =========================================

if __name__ == "__main__":
    main()
