---
project: regression
task: regression
dataset: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho
team:
  - Cynthia Naoko Yasutake
  - Davi Peter Bastian Nehls
  - Henrique Bromfman de Puppi e Silva
ai_use: "used for planning, code review and writing support"
---
# 2. Regressão

!!! abstract "Entrega 2 de 3 do [Projeto](../index.md)"

    [Projects](https://insper.github.io/ann-dl/){:target='_blank'}

!!! info "Equipe"

    | Nome completo | GitHub |
    |---------------|--------|
    | Cynthia Naoko Yasutake | CYNahko|
    | Davi Peter Bastian Nehls | dpnnpd |
    | Henrique Bromfman de Puppi e Silva | kikepuppi |

!!! tip "Como reproduzir"

    Todos os números e figuras desta página saem de `code/regression.py`, com `random_state=42` em tudo:

    ```bash
    python -m venv .venv
    pip install -r requirements.txt
    python docs/projects/regression/code/regression.py
    ```

Prevemos o preço de revenda de carros usados (`selling_price`, em ₹) a partir do
`Car details v3.csv`, o mesmo dataset e alvo da [EDA](../eda/index.md).

## 0. Ponto de partida

Reusamos o pipeline da EDA sem mudanças (`eda/code/preprocessing.py`): a mesma limpeza, o mesmo
split 80/20 estratificado por decis de `log(preço)` e o mesmo `random_state=42`. A validação sai do
treino, com outro split 80/20 estratificado da mesma forma. O pipeline é ajustado só nas linhas de
treino e depois aplicado à validação e ao teste. Não recebemos feedback da EDA para corrigir.

| Conjunto | Linhas |
|----------|--------|
| Treino | 4.420 |
| Validação | 1.105 |
| Teste | 1.382 |

Depois do pipeline, cada carro vira um vetor de 40 features.

**Métrica principal:** MAE em ₹, na escala original do preço. É o erro médio em rúpias, fácil de
ler para quem compra ou vende.

**Critério de sucesso**, fixado antes de treinar: a rede precisa ter MAE no teste pelo menos 15%
menor que o do melhor baseline.
