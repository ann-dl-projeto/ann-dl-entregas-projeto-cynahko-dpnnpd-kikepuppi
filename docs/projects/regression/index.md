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

## 1. Baselines

Treinamos dois modelos simples sobre o `log(preço)` e avaliamos na validação, depois de voltar à
escala em ₹ com `exp`:

- **Dummy (mediana):** prevê sempre a mediana do treino. É o piso: qualquer modelo útil tem que
  ficar abaixo dele.
- **Ridge:** regressão linear com regularização L2 (`alpha=1`).

| Modelo | MAE (₹) | RMSE (₹) | R² |
|--------|--------:|---------:|---:|
| Dummy (mediana) | 278.744 | 523.161 | −0,054 |
| Ridge | 92.824 | 192.640 | 0,857 |

O Ridge é o melhor baseline. Erra em média ₹92.824, um terço do erro da mediana, e explica 85,7%
da variância do preço. Pelo critério de sucesso, a rede precisa ficar pelo menos 15% abaixo do MAE
do Ridge no teste.

## 2. A rede

A rede é um MLP (*multilayer perceptron*) em PyTorch, montado pela função `criar_rede` em
`code/regression.py`:

```
entrada (40) → Linear(40, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 1) → log(preço)
```

| Escolha | Valor | Por quê |
|---------|-------|---------|
| Entrada | 40 neurônios | Uma por feature que sai do pipeline da EDA |
| Camadas escondidas | 2, com 64 e 32 neurônios | Cada camada combina as features da anterior. Duas já captam interações como combustível × ano × câmbio, que o Ridge não vê. A largura começa um pouco acima da entrada e afunila até a saída |
| Ativação | ReLU, `max(0, x)` | Sem ela, as camadas se juntariam numa única transformação linear e a rede seria só outra regressão linear. A ReLU é o padrão em MLP e não satura para valores positivos |
| Inicialização | He (`kaiming_normal_`), bias zero | Sorteia os pesos com variância 2/entradas, a escala certa para a ReLU: o sinal não some nem explode ao passar pelas camadas |
| Saída | 1 neurônio linear, sem ativação | A tarefa é regressão. A saída é o `log(preço)`, que pode ser qualquer número real |
| Loss | Huber com `delta=0.5` | Explicação abaixo |

**Parâmetros contra linhas de treino.** A rede tem 4.737 parâmetros (pesos e bias) e o treino tem
4.420 linhas, cerca de 1,07 parâmetro por linha. Com essa proporção a rede tem capacidade de sobra
para decorar o treino, então o controle de overfitting fica com o early stopping (seção 3) e com
a regularização (seção 4). Uma rede bem maior teria muito mais parâmetros que dados.

**Loss.** Treinamos no `log(preço)`, como na EDA, porque o preço bruto tem assimetria de 5,57 e um
erro de ₹100 mil pesa diferente num carro de ₹3 lakh e num de ₹50 lakh. No log, o erro passa a ser
relativo. Mesmo assim sobram outliers: nos resíduos do Ridge no treino, a mediana do erro absoluto
é 0,15, o percentil 95 é 0,49 e o máximo é 2,25, ou seja, um carro com o preço errado por um fator
de 9. Com MSE, esses poucos casos dominariam o gradiente. A Huber é quadrática para erros até
`delta` e linear acima dele. Com `delta=0.5`, só 4,7% das linhas caem na parte linear: o resto é
tratado como no MSE e os outliers perdem peso.
