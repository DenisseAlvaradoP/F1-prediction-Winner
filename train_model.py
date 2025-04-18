# Script para entrenamiento del modelo
"""
train_model.py
--------------
Funciones para entrenar, evaluar y predecir utilizando XGBoost.
"""

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(X_train, y_train):
    """
    Entrena un modelo XGBoost con los datos de entrenamiento.
    Args:
        X_train (pd.DataFrame): Características de entrenamiento.
        y_train (pd.Series): Etiquetas de entrenamiento.
    Returns:
        XGBClassifier: Modelo entrenado.
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo entrenado con métricas de clasificación y matriz de confusión.
    Args:
        model (XGBClassifier): Modelo entrenado.
        X_test (pd.DataFrame): Características de prueba.
        y_test (pd.Series): Etiquetas de prueba.
    """
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

def predict_sample(model, sample):
    """
    Predice la categoría de posición para una muestra individual.
    Args:
        model (XGBClassifier): Modelo entrenado.
        sample (pd.DataFrame): Muestra a predecir.
    Returns:
        str: Categoría predicha.
    """
    prediction = model.predict(sample)
    return prediction[0]
