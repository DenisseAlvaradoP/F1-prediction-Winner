# Script para preprocesamiento de datos

"""
data_preprocessing.py
----------------------
Funciones para cargar y preprocesar datos históricos de Fórmula 1.
"""

import pandas as pd

def load_data(filepath):
    """
    Carga los datos desde un archivo CSV.
    Args:
        filepath (str): Ruta al archivo CSV.
    Returns:
        pd.DataFrame: DataFrame con los datos cargados.
    """
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """
    Preprocesa el DataFrame para convertir la posición final en una categoría.
    Args:
        df (pd.DataFrame): Datos originales.
    Returns:
        X (pd.DataFrame): Características.
        y (pd.Series): Etiquetas de clasificación.
    """
    # Selección de columnas relevantes
    df = df[['Track', 'Position', 'No', 'Driver', 'Team', 'Starting Grid']].dropna()

    # Conversión de tipos de datos
    df['Position'] = df['Position'].astype(int)
    df['Starting Grid'] = df['Starting Grid'].astype(int)

    # Definición de la variable objetivo
    def categorize_position(pos):
        if pos <= 3:
            return "Top3"
        elif pos <= 10:
            return "Top10"
        else:
            return "Others"

    df['Category'] = df['Position'].apply(categorize_position)

    # Codificación de variables categóricas
    df_encoded = pd.get_dummies(df[['Track', 'Driver', 'Team']], drop_first=True)

    # Características y etiquetas
    X = pd.concat([df_encoded, df[['Starting Grid']]], axis=1)
    y = df['Category']

    return X, y
