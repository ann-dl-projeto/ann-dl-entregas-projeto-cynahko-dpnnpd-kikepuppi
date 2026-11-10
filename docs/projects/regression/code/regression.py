"""Regressão do CarDekho (Car details v3): gera as figuras em ../figures e imprime os números do relatório.

Rodar a partir da raiz do repositório:
    python docs/projects/regression/code/regression.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eda" / "code"))
from preprocessing import SEED, build_pipeline, load_clean, split

# Pasta onde as figuras são salvas
FIGURAS = Path(__file__).resolve().parents[1] / "figures"


# Funções auxiliares
def titulo(texto):
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


def metricas(y_log, prev_log):
    """MAE, RMSE e R² em rúpias: os modelos preveem log(preço) e voltamos com exp."""
    real, prev = np.exp(y_log), np.exp(prev_log)
    return {"MAE": mean_absolute_error(real, prev),
            "RMSE": root_mean_squared_error(real, prev),
            "R²": r2_score(real, prev)}


# 0. Ponto de partida: mesmo split da EDA e validação tirada do treino
titulo("0. Ponto de partida")
dados = load_clean()
X_treino, X_te, y_treino, y_te = split(dados)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_treino, y_treino, test_size=0.2, random_state=SEED, stratify=pd.qcut(y_treino, 10, labels=False))

prep = build_pipeline().fit(X_tr)
Z_tr, Z_val, Z_te = prep.transform(X_tr), prep.transform(X_val), prep.transform(X_te)
print("linhas treino/validação/teste:", len(X_tr), len(X_val), len(X_te))
print("features depois do pipeline:", Z_tr.shape[1])

# 1. Baselines: mediana e linear, avaliados na validação
titulo("1. Baselines (validação)")
baselines = {"Dummy (mediana)": DummyRegressor(strategy="median"), "Ridge": Ridge()}
resultados = {}
for nome, modelo in baselines.items():
    modelo.fit(Z_tr, y_tr)
    resultados[nome] = metricas(y_val, modelo.predict(Z_val))
print(pd.DataFrame(resultados).T.round(3))
