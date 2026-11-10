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

## 3. Treino

O treino fica na função `treinar` em `code/regression.py`.

| Escolha | Valor | Por quê |
|---------|-------|---------|
| Otimizador | Adam | Ajusta um passo próprio para cada peso, a partir da média e da variância dos gradientes recentes. É o padrão para MLP e funciona bem sem ajuste fino |
| Learning rate | 0,001 | Padrão do Adam. Grande demais faz a loss oscilar; pequeno demais deixa o treino lento |
| Batch | 64 | 4.420 linhas viram cerca de 70 atualizações por época. O ruído de um lote pequeno ajuda a rede a não decorar o treino |
| Épocas | até 500 | É só um teto. Quem decide a parada é o early stopping |
| Early stopping | paciência 30 | A cada época medimos a loss na validação. Se ela não melhora por 30 épocas seguidas, o treino para e a rede volta aos pesos da melhor época |

O treino parou na época 114, e a melhor época foi a 84. Nela, a loss Huber foi de 0,0175 no treino
e 0,0254 na validação.

![Figura 1](figures/fig01-curvas.png)

A Figura 1 mostra três fases:

- **Épocas 1 a ~15:** as duas curvas caem juntas e rápido. A rede aprende o padrão geral do preço.
- **Épocas ~15 a 84:** a validação quase não melhora mais, enquanto o treino continua descendo devagar.
  A distância entre as curvas cresce: é o começo do overfitting.
- **Depois da 84:** a validação não melhora por 30 épocas e o treino para.

Os picos nas duas curvas, como os das épocas 65 e 100, são o ruído normal dos mini-batches. Como
eles aparecem juntos no treino e na validação, não indicam problema.

Na validação, a rede já fica à frente do Ridge:

| Modelo | MAE (₹) | RMSE (₹) | R² |
|--------|--------:|---------:|---:|
| Dummy (mediana) | 278.744 | 523.161 | −0,054 |
| Ridge | 92.824 | 192.640 | 0,857 |
| Rede, sem regularização | 84.318 | 178.861 | 0,877 |

O MAE é 9,2% menor que o do Ridge, ainda abaixo dos 15% do critério. A seção 4 testa regularização e
outras configurações para tentar fechar essa diferença.

## 4. Regularização e hiperparâmetros

Testamos duas técnicas de regularização, as duas para combater o overfitting visto na Figura 1:

- **Dropout:** a cada lote do treino, zera ao acaso uma fração dos neurônios escondidos (10% ou 20%).
  A rede não pode depender de um neurônio específico e precisa espalhar o que aprende. Na avaliação,
  o dropout fica desligado.
- **Weight decay:** soma à loss uma penalidade proporcional ao quadrado dos pesos (L2, `1e-4`). Pesos
  grandes custam caro, e a rede fica mais suave.

Todas as configurações treinam do zero, com a mesma seed, o mesmo treino da seção 3 (Adam, batch 64,
early stopping com paciência 30) e a escolha feita só na validação.

**Regularização contra a rede sem regularização** (rede 64-32, learning rate 0,001):

| Regularização | MAE (₹) | Diferença |
|---------------|--------:|----------:|
| Nenhuma | 84.318 | |
| Weight decay 1e-4 | 82.566 | −2,1% |
| Dropout 0,1 + weight decay 1e-4 | 85.717 | +1,7% |
| Dropout 0,1 | 88.102 | +4,5% |
| Dropout 0,2 | 94.630 | +12,2% |

O weight decay ajudou um pouco. O dropout piorou, e mais quanto maior a taxa. Com cerca de 1 parâmetro
por linha, a rede já é pequena, e desligar neurônios tira a capacidade que ela tem para aprender.

**Busca completa**, 12 configurações ordenadas pelo MAE na validação:

| # | Camadas | Dropout | Weight decay | LR | Parâmetros | Melhor época | MAE (₹) | RMSE (₹) | R² |
|--:|---------|--------:|-------------:|---:|-----------:|-------------:|--------:|---------:|---:|
| 1 | 64-32 | 0,1 | 1e-4 | 0,003 | 4.737 | 197 | 80.782 | 164.196 | 0,896 |
| 2 | 32-16 | 0 | 0 | 0,001 | 1.857 | 109 | 81.931 | 162.747 | 0,898 |
| 3 | 64-32 | 0 | 1e-4 | 0,001 | 4.737 | 84 | 82.566 | 171.631 | 0,887 |
| 4 | 64-32 | 0 | 0 | 0,001 | 4.737 | 84 | 84.318 | 178.861 | 0,877 |
| 5 | 128-64 | 0 | 0 | 0,001 | 13.569 | 50 | 85.086 | 172.636 | 0,885 |
| 6 | 64-32 | 0,1 | 1e-4 | 0,001 | 4.737 | 121 | 85.717 | 157.580 | 0,904 |
| 7 | 64-32 | 0 | 0 | 0,003 | 4.737 | 32 | 86.919 | 184.323 | 0,869 |
| 8 | 64-32 | 0,1 | 0 | 0,001 | 4.737 | 121 | 88.102 | 170.466 | 0,888 |
| 9 | 128-64 | 0,2 | 1e-4 | 0,001 | 13.569 | 134 | 92.223 | 186.966 | 0,865 |
| 10 | 128-64 | 0,1 | 0 | 0,001 | 13.569 | 53 | 93.832 | 177.093 | 0,879 |
| 11 | 64-32 | 0,2 | 0 | 0,001 | 4.737 | 116 | 94.630 | 190.553 | 0,860 |
| 12 | 32-16 | 0,1 | 1e-4 | 0,001 | 1.857 | 69 | 96.982 | 215.240 | 0,822 |

**Configuração escolhida:** a #1, rede 64-32 com dropout 0,1, weight decay 1e-4 e learning rate 0,003.
Ela teve MAE de ₹80.782 na validação, 13,0% abaixo do Ridge. O dropout que atrapalhava com LR 0,001
funcionou com LR 0,003: o passo maior acelera o treino, e o ruído do dropout segura o overfitting,
tanto que a melhor época foi a 197, contra 84 na rede sem regularização.

As redes maiores (128-64) foram piores: com 13.569 parâmetros, cerca de 3 por linha, sobram pesos
para decorar o treino. Entre as cinco primeiras, a diferença de MAE é de até 5%, e parte disso pode
ser sorte da seed. Por isso a seção 5 mede a escolhida com 3 seeds.
