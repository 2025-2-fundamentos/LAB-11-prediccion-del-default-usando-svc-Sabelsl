# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#


import json
import gzip
import pickle
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
precision_score,
balanced_accuracy_score,
recall_score,
f1_score,
confusion_matrix)

df_train = pd.read_csv("files/input/train_data.csv.zip")
df_test = pd.read_csv("files/input/test_data.csv.zip")

df_train = df_train.rename(columns={"default payment next month": "default"})
df_test = df_test.rename(columns={"default payment next month": "default"})

df_train = df_train.drop(columns=["ID"])
df_test = df_test.drop(columns=["ID"])

df_train = df_train.dropna()
df_test = df_test.dropna()

df_train.loc[df_train["EDUCATION"] > 4, "EDUCATION"] = 4
df_test.loc[df_test["EDUCATION"] > 4, "EDUCATION"] = 4

# Paso 2
X_tr = df_train.drop(columns="default")
y_tr = df_train["default"]

X_te = df_test.drop(columns="default")
y_te = df_test["default"]

# Paso 3
cols_cat = ["SEX", "EDUCATION", "MARRIAGE"]
cols_num = [col for col in X_tr.columns if col not in cols_cat]

transformador = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cols_cat),
        ("num", StandardScaler(), cols_num),
    ]
)

pipeline = Pipeline(
    steps=[
        ("prep", transformador),
        ("pca", PCA()),
        ("select", SelectKBest(score_func=f_classif)),
        ("svc", SVC()),
    ]
)

# Paso 4
param_grid = {
    "pca__n_components": [21],
    "select__k": [12],
    "svc__C": [0.8],
    "svc__kernel": ["rbf"],
    "svc__gamma": [0.1],
}

busqueda = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
)

busqueda.fit(X_tr, y_tr)

# Paso 5
os.makedirs("files/models", exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as f_modelo:
    pickle.dump(busqueda, f_modelo)

# Paso 6
def construir_metricas(nombre, y_real, y_predicho):
    return {
        "type": "metrics",
        "dataset": nombre,
        "precision": float(round(precision_score(y_real, y_predicho), 3)),
        "balanced_accuracy": float(
            round(balanced_accuracy_score(y_real, y_predicho), 3)
        ),
        "recall": float(round(recall_score(y_real, y_predicho), 3)),
        "f1_score": float(round(f1_score(y_real, y_predicho), 3)),
    }


met_train = construir_metricas("train", y_tr, busqueda.predict(X_tr))
met_test = construir_metricas("test", y_te, busqueda.predict(X_te))

# Paso 7
def construir_cm(nombre, y_real, y_predicho):
    tn, fp, fn, tp = confusion_matrix(y_real, y_predicho).ravel()
    return {
        "type": "cm_matrix",
        "dataset": nombre,
        "true_0": {
            "predicted_0": int(tn),
            "predicted_1": int(fp),
        },
        "true_1": {
            "predicted_0": int(fn),
            "predicted_1": int(tp),
        },
    }


cm_train = construir_cm("train", y_tr, busqueda.predict(X_tr))
cm_test = construir_cm("test", y_te, busqueda.predict(X_te))

salida = [met_train, met_test, cm_train, cm_test]

os.makedirs("files/output", exist_ok=True)

with open("files/output/metrics.json", "w") as f_salida:
    for registro in salida:
        f_salida.write(json.dumps(registro) + "\n")



