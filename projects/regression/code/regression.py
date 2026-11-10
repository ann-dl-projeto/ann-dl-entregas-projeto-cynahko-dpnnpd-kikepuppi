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


def criar_rede(entradas, camadas=(64, 32), dropout=0.0):
    """MLP: cada camada escondida é Linear + ReLU (+ Dropout); a saída é 1 neurônio linear (log do preço)."""
    blocos, anterior = [], entradas
    for largura in camadas:
        blocos += [nn.Linear(anterior, largura), nn.ReLU()]
        if dropout > 0:
            blocos.append(nn.Dropout(dropout))
        anterior = largura
    blocos.append(nn.Linear(anterior, 1))
    rede = nn.Sequential(*blocos)
    for camada in rede:
        if isinstance(camada, nn.Linear):
            nn.init.kaiming_normal_(camada.weight, nonlinearity="relu")  # inicialização He
            nn.init.zeros_(camada.bias)
    return rede


def treinar(rede, perda, lr=1e-3, weight_decay=0.0, batch=64, epocas=500, paciencia=30):
    """Adam em mini-batches com early stopping na validação; no fim, volta aos melhores pesos."""
    X, Y = tensor(Z_tr), tensor(y_tr).view(-1, 1)
    X_v, Y_v = tensor(Z_val), tensor(y_val).view(-1, 1)
    otimizador = torch.optim.Adam(rede.parameters(), lr=lr, weight_decay=weight_decay)
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

# 4. Regularização e busca: cada configuração treina do zero com a mesma seed e é medida na validação
titulo("4. Regularização e hiperparâmetros (validação)")
configuracoes = [
    # camadas, dropout, weight decay, learning rate
    ((64, 32), 0.0, 0, 1e-3),  # sem regularização (a rede da seção 3)
    ((64, 32), 0.1, 0, 1e-3),
    ((64, 32), 0.2, 0, 1e-3),
    ((64, 32), 0.0, 1e-4, 1e-3),
    ((64, 32), 0.1, 1e-4, 1e-3),
    ((64, 32), 0.1, 1e-4, 3e-3),
    ((64, 32), 0.0, 0, 3e-3),
    ((32, 16), 0.0, 0, 1e-3),
    ((32, 16), 0.1, 1e-4, 1e-3),
    ((128, 64), 0.0, 0, 1e-3),
    ((128, 64), 0.1, 0, 1e-3),
    ((128, 64), 0.2, 1e-4, 1e-3),
]
busca = []
for camadas, dropout, wd, lr in configuracoes:
    torch.manual_seed(SEED)
    modelo = criar_rede(Z_tr.shape[1], camadas, dropout)
    hist = treinar(modelo, perda, lr=lr, weight_decay=wd)
    busca.append({"camadas": "-".join(map(str, camadas)), "dropout": dropout, "weight decay": wd, "lr": lr,
                  "parâmetros": sum(p.numel() for p in modelo.parameters()),
                  "melhor época": int(np.argmin(hist["validação"])) + 1,
                  **metricas(y_val, prever(modelo, Z_val))})
busca = pd.DataFrame(busca).sort_values("MAE").reset_index(drop=True)
print(busca.round({"MAE": 0, "RMSE": 0, "R²": 3}).to_string())

# 5. Teste, usado uma única vez: baselines e a configuração escolhida (#1 da busca) em 3 seeds
titulo("5. Teste")
teste = {nome: metricas(y_te, modelo.predict(Z_te)) for nome, modelo in baselines.items()}

redes_finais = {}
for seed in [42, 43, 44]:
    torch.manual_seed(seed)
    redes_finais[seed] = criar_rede(Z_tr.shape[1], (64, 32), dropout=0.1)
    hist = treinar(redes_finais[seed], perda, lr=3e-3, weight_decay=1e-4)
    print(f"seed {seed}: parou na época {len(hist['validação'])}, melhor época {int(np.argmin(hist['validação'])) + 1}")
por_seed = pd.DataFrame({seed: metricas(y_te, prever(r, Z_te)) for seed, r in redes_finais.items()}).T
print("rede, por seed:\n", por_seed.round(3))
teste["Rede (média de 3 seeds)"] = por_seed.mean().to_dict()
print(pd.DataFrame(teste).T.round(3))
print("desvio padrão da rede:", por_seed.std().round(3).to_dict())

reducao = 1 - por_seed["MAE"].mean() / teste["Ridge"]["MAE"]
print(f"MAE da rede vs. Ridge: {100 * reducao:.1f}% menor | critério (>= 15%):", "atingido" if reducao >= 0.15 else "não atingido")

# Figuras 2 e 3: predito vs. real e resíduo vs. predito (rede da seed 42, em ₹)
real, previsto = np.exp(y_te), np.exp(prever(redes_finais[42], Z_te))
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.scatter(real, previsto, s=8, alpha=0.4, color=AZUL, label="carros do teste")
limites = [real.min(), real.max()]
ax.plot(limites, limites, color=LARANJA, label="previsão perfeita")
ax.set(title="Figura 2. Preço previsto vs. real (teste)", xlabel="preço real (₹)", ylabel="preço previsto (₹)",
       xscale="log", yscale="log")
ax.legend(frameon=False)
salvar(fig, "fig02-previsto-real.png")

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(previsto, real - previsto, s=8, alpha=0.4, color=AZUL, label="carros do teste")
ax.axhline(0, color=LARANJA, label="resíduo zero")
ax.set(title="Figura 3. Resíduo vs. preço previsto (teste)", xlabel="preço previsto (₹)",
       ylabel="resíduo = real − previsto (₹)", xscale="log")
ax.legend(frameon=False)
salvar(fig, "fig03-residuos.png")

# 6. Análise de erro (rede da seed 42, no teste)
titulo("6. Análise de erro")
erros = X_te.assign(real=real, previsto=previsto, erro=(real - previsto).abs())
erros["erro %"] = 100 * erros["erro"] / erros["real"]
erros["faixa de preço"] = pd.qcut(erros["real"], 5, labels=["1 (mais baratos)", "2", "3", "4", "5 (mais caros)"])


def por_grupo(coluna):
    return erros.groupby(coluna, observed=True).agg(
        carros=("erro", "size"), MAE=("erro", "mean"), erro_pct_mediano=("erro %", "median")).round(1)


for coluna in ["faixa de preço", "fuel", "transmission", "seller_type"]:
    print(por_grupo(coluna), "\n")
marcas = erros["brand"].value_counts().head(8).index
print(por_grupo("brand").loc[marcas].sort_values("MAE"), "\n")
print("faixas de preço (₹):", pd.qcut(erros["real"], 5).cat.categories.tolist())

print("\n10 piores casos:")
print(erros.nlargest(10, "erro")[["brand", "year", "km_driven", "fuel", "transmission", "seller_type",
                                    "max_power", "real", "previsto", "erro %"]].round(0).to_string())
print("carros dessas marcas no treino:", X_tr["brand"].value_counts().reindex(
    erros.nlargest(10, "erro")["brand"].unique()).fillna(0).astype(int).to_dict())

# Figura 4: MAE por faixa de preço
faixas = por_grupo("faixa de preço")
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(faixas.index.astype(str), faixas["MAE"], color=AZUL, label="MAE (₹)")
ax.set(title="Figura 4. MAE da rede por faixa de preço real (teste)", xlabel="quintil do preço real",
       ylabel="MAE (₹)")
ax.legend(frameon=False)
salvar(fig, "fig04-erro-faixa.png")

# Permutation importance: embaralha uma coluna original por vez e mede quanto o MAE piora
rng = np.random.default_rng(SEED)
mae_base = metricas(y_te, prever(redes_finais[42], Z_te))["MAE"]
importancia = {}
for coluna in X_te.columns:
    pioras = []
    for _ in range(5):
        X_emb = X_te.copy()
        X_emb[coluna] = rng.permutation(X_emb[coluna].values)
        pioras.append(metricas(y_te, prever(redes_finais[42], prep.transform(X_emb)))["MAE"] - mae_base)
    importancia[coluna] = np.mean(pioras)
importancia = pd.Series(importancia).sort_values(ascending=False)
print("\npermutation importance (aumento do MAE em ₹, média de 5 embaralhamentos):")
print(importancia.round(0))
