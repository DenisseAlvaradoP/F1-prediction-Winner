# Script para preprocesamiento de datos

import pandas as pd

def preprocess_data(path):
    df = pd.read_csv(path)
    # Preprocesamiento básico
    return df