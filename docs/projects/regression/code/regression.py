"""Regressão do CarDekho (Car details v3): gera as figuras em ../figures e imprime os números do relatório.

Rodar a partir da raiz do repositório:
    python docs/projects/regression/code/regression.py
"""
import copy
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eda" / "code"))
from preprocessing import SEED, build_pipeline, load_clean, split

# Pasta onde as figuras são salvas
FIGURAS = Path(__file__).resolve().parents[1] / "figures"
FIGURAS.mkdir(exist_ok=True)

# Mesma paleta da EDA
AZUL, LARANJA = "#2a78d6", "#eb6834"


# Funções auxiliares
def titulo(texto):
    print(f"\n{'=' * 78}\n{texto}\n{'=' * 78}")


def salvar(fig, nome):
    fig.tight_layout()
    fig.savefig(FIGURAS / nome, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] {nome}")


def tensor(x):
    return torch.tensor(np.asarray(x), dtype=torch.float32)


def metricas(y_log, prev_log):
    """MAE, RMSE e R² em rúpias: os modelos preveem log(preço) e voltamos com exp."""
    real, prev = np.exp(y_log), np.exp(prev_log)
    return {"MAE": mean_absolute_error(real, prev),
            "RMSE": root_mean_squared_error(real, prev),
            "R²": r2_score(real, prev)}


def criar_rede(entradas, camadas=(64, 32)):
    """MLP: cada camada escondida é Linear + ReLU; a saída é 1 neurônio linear (log do preço)."""
    blocos, anterior = [], entradas
    for largura in camadas:
        blocos += [nn.Linear(anterior, largura), nn.ReLU()]
        anterior = largura
    blocos.append(nn.Linear(anterior, 1))
    rede = nn.Sequential(*blocos)
    for camada in rede:
        if isinstance(camada, nn.Linear):
            nn.init.kaiming_normal_(camada.weight, nonlinearity="relu")  # inicialização He
            nn.init.zeros_(camada.bias)
    return rede


def treinar(rede, perda, lr=1e-3, batch=64, epocas=500, paciencia=30):
    """Adam em mini-batches com early stopping na validação; no fim, volta aos melhores pesos."""
    X, Y = tensor(Z_tr), tensor(y_tr).view(-1, 1)
    X_v, Y_v = tensor(Z_val), tensor(y_val).view(-1, 1)
    otimizador = torch.optim.Adam(rede.parameters(), lr=lr)
    historico = {"treino": [], "validação": []}
    melhor, melhores_pesos, sem_melhora = float("inf"), None, 0
    for epoca in range(epocas):
        rede.train()
        ordem = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            lote = ordem[i:i + batch]
            otimizador.zero_grad()
            perda(rede(X[lote]), Y[lote]).backward()
            otimizador.step()

        rede.eval()
        with torch.no_grad():
            historico["treino"].append(perda(rede(X), Y).item())
            historico["validação"].append(perda(rede(X_v), Y_v).item())
        if historico["validação"][-1] < melhor:
            melhor, melhores_pesos, sem_melhora = historico["validação"][-1], copy.deepcopy(rede.state_dict()), 0
        else:
            sem_melhora += 1
            if sem_melhora == paciencia:
                break
    rede.load_state_dict(melhores_pesos)
    return historico


def prever(rede, Z):
    rede.eval()
    with torch.no_grad():
        return rede(tensor(Z)).numpy().ravel()


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

# 2. A rede: arquitetura, número de parâmetros e escolha da loss
titulo("2. A rede")
torch.manual_seed(SEED)
rede = criar_rede(Z_tr.shape[1])
print(rede)
parametros = sum(p.numel() for p in rede.parameters())
print("parâmetros:", parametros, "| linhas de treino:", len(Z_tr),
      "| parâmetros por linha:", round(parametros / len(Z_tr), 2))

# Resíduos do Ridge no treino (em log): mostram a cauda que justifica a Huber
residuo = (y_tr - baselines["Ridge"].predict(Z_tr)).abs()
print("|resíduo| do Ridge, quantis 50/95/max:", residuo.quantile([.5, .95, 1]).round(2).tolist())
print("% de resíduos acima de 0,5:", round(100 * (residuo > 0.5).mean(), 1))
perda = nn.HuberLoss(delta=0.5)

# 3. Treino: Adam, batch 64, até 500 épocas, early stopping com paciência 30
titulo("3. Treino")
historico = treinar(rede, perda)
melhor_epoca = int(np.argmin(historico["validação"])) + 1
print("épocas rodadas:", len(historico["validação"]), "| melhor época:", melhor_epoca)
print("loss na melhor época, treino/validação:",
      round(historico["treino"][melhor_epoca - 1], 4), round(historico["validação"][melhor_epoca - 1], 4))
resultados["Rede"] = metricas(y_val, prever(rede, Z_val))
print(pd.DataFrame(resultados).T.round(3))

# Figura 1: curvas de loss de treino e validação
epocas = range(1, len(historico["treino"]) + 1)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epocas, historico["treino"], color=AZUL, label="treino")
ax.plot(epocas, historico["validação"], color=LARANJA, label="validação")
ax.axvline(melhor_epoca, color="gray", ls="--", label=f"melhor época ({melhor_epoca})")
ax.set(title="Figura 1. Loss Huber por época", xlabel="época", ylabel="loss Huber (log do preço)", yscale="log")
ax.legend(frameon=False)
salvar(fig, "fig01-curvas.png")
