"""
main.py
-------
Script principal para ejecutar el flujo completo de predicción.
"""

from scripts.data_preprocessing import load_data, preprocess_data
from scripts.train_model import train_model, evaluate_model, predict_sample
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    # Ruta al archivo de datos
    data_path = 'data/Formula1_2024season_raceResults.csv'

    # Cargar y preprocesar datos
    print("Cargando y preprocesando datos...")
    df = load_data(data_path)
    X, y = preprocess_data(df)

    # Dividir en conjuntos de entrenamiento y prueba
    print("Dividiendo datos en entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar el modelo
    print("Entrenando el modelo...")
    model = train_model(X_train, y_train)

    # Evaluar el modelo
    print("Evaluando el modelo...")
    evaluate_model(model, X_test, y_test)

    # Simulación de predicción
    print("Simulando predicción con una muestra ficticia...")
    sample = X_test.iloc[[0]]
    result = predict_sample(model, sample)
    print("Predicción:", result)

if __name__ == "__main__":
    main()
