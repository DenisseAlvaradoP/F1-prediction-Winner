# F1 Race Result Prediction
Project Description
Title: Formula 1 Race Results Prediction Using Machine Learning

This project is a complete pipeline designed to predict the performance of Formula 1 drivers in races using machine learning techniques, specifically using the XGBoost classifier.

🧠 Objective:
To classify the expected race outcome of a driver as:

Top 3 (podium finish),

Top 10 (points),

or Others (outside the points zone),

based on data from the 2024 F1 season, such as:

the driver’s name,

the constructor (team),

the track (circuit),

and the starting grid position.

🧩 Main Components:
Data Collection & Preparation (data_preprocessing.py)

Loads the dataset Formula1_2024season_raceResults.csv.

Cleans missing data and selects key features.

Encodes categorical variables (Driver, Team, Track) into machine-readable format.

Categorizes final race positions into classes (Top3, Top10, Others).

Model Training and Evaluation (train_model.py)

Trains an XGBClassifier on historical data.

Evaluates the model using accuracy, classification report, and confusion matrix.

Allows making predictions on custom inputs.

Main Pipeline Execution (main.py)

Ties together data loading, preprocessing, training, evaluation, and sample prediction.

Provides a full, ready-to-run script to demonstrate the model’s performance.

🔬 Technologies Used:
Python

XGBoost

Pandas

Scikit-learn

Matplotlib & Seaborn

🎯 Use Case:
This can be used as a foundation to:

Build predictive tools for F1 race outcomes.

Experiment with advanced features (e.g., weather, pit stops, lap times).

Create a real-time race forecast system.





Descripción del Proyecto
Título: Predicción de Resultados de Carreras de Fórmula 1 con Machine Learning

Este proyecto es una pipeline completa diseñada para predecir el rendimiento de los pilotos de Fórmula 1 en las carreras utilizando técnicas de aprendizaje automático, específicamente el clasificador XGBoost.

🧠 Objetivo:
Clasificar el resultado esperado de un piloto como:

Top 3 (podio),

Top 10 (zona de puntos),

o Otros (fuera de puntos),

basándose en los datos de la temporada 2024 como:

el nombre del piloto,

el equipo (constructor),

el circuito,

y la posición de salida (grid).

🧩 Componentes Principales:
Carga y Preprocesamiento de Datos (data_preprocessing.py)

Carga el archivo Formula1_2024season_raceResults.csv.

Limpia los datos faltantes y selecciona características clave.

Codifica variables categóricas (Piloto, Equipo, Circuito).

Clasifica la posición final en categorías (Top3, Top10, Otros).

Entrenamiento y Evaluación del Modelo (train_model.py)

Entrena un modelo XGBClassifier con datos históricos.

Evalúa el modelo con métricas de precisión, reporte de clasificación y matriz de confusión.

Permite realizar predicciones con datos nuevos.

Ejecución del Flujo Principal (main.py)

Conecta todos los pasos: carga de datos, procesamiento, entrenamiento, evaluación y predicción de ejemplo.

Script completo listo para ejecutar.

🔬 Tecnologías Utilizadas:
Python

XGBoost

Pandas

Scikit-learn

Matplotlib & Seaborn

🎯 Caso de Uso:
Se puede utilizar como base para:

Construir herramientas predictivas de resultados en F1.

Experimentar con más variables (clima, paradas en pits, tiempos por vuelta).

Desarrollar un sistema de pronóstico en tiempo real para carreras.

