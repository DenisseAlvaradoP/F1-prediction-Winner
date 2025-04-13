# Script para entrenamiento del modelo

from xgboost import XGBClassifier

def train_model(X, y):
    model = XGBClassifier()
    model.fit(X, y)
    return model