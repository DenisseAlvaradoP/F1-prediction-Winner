# main.py

from scripts.data_preprocessing import load_data, preprocess_data
from scripts.train_model import train_model, evaluate_model, predict_sample
from sklearn.model_selection import train_test_split
import pandas as pd

def main():
    # Paso 1: Cargar datos
    print("Cargando datos...")
    df = load_data()

    # Paso 2: Preprocesar
    print("Preprocesando datos...")
    X, y = preprocess_data(df)

    # Paso 3: Dividir en entrenamiento y prueba
    print("Dividiendo en conjuntos de entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Paso 4: Entrenar el modelo
    print("Entrenando modelo...")
    model = train_model(X_train, y_train)

    # Paso 5: Evaluar el modelo
    print("Evaluando modelo...")
    evaluate_model(model, X_test, y_test)

    # Paso 6: Simulación de predicción
    print("Simulando predicción con muestra ficticia:")
    sample = pd.DataFrame([[20, 1, 9, 3]], columns=['raceId', 'driverId', 'constructorId', 'grid'])
    result = predict_sample(model, sample)
    print("Predicción:", result)

if __name__ == "__main__":
    main()
