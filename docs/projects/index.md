# Projeto

!!! abstract "Enunciados"

    [Projects](https://insper.github.io/ann-dl/){:target='_blank'}

O projeto é um só, feito em equipe sobre o mesmo dataset e entregue em três partes ao longo do
semestre, cada uma com data e peso próprios.

## Equipe

!!! danger "Preencha antes de qualquer entrega"

    Toda entrega do projeto é avaliada em equipe. Sem os nomes aqui, não há como atribuir a nota.
    O mesmo vale para o `mkdocs.yml`: o campo `site_author` deve listar o grupo.

| Nome completo | E-mail | GitHub |
|---------------|--------|--------|
| Cynthia Naoko Yasutake | cynthiany@al.insper.edu.br | CYNahko |
| Davi Peter Bastian Nehls | davipbn@al.insper.edu.br | dpnnpd |
| Henrique Bromfman de Puppi e Silva | henriquebps@al.insper.edu.br | kikepuppi |

Os times têm de 2 a 3 pessoas. Os nomes se repetem no cabeçalho de cada entrega, porque quem
corrige pode abrir uma página direto, sem passar por aqui.

## As três entregas

| # | Entrega | Página |
|---|---------|--------|
| 1 | EDA | [EDA](eda/index.md) |
| 2 | Regressão | [Regressão](regression/index.md) |
| 3 | Generativo | em breve |

As datas e os pesos de cada entrega estão no [overview](https://insper.github.io/ann-dl/){:target='_blank'}
da edição.

!!! danger "A nota do projeto costuma ser limitada por uma prova sobre o próprio projeto"

    Um relatório bem escrito não salva uma equipe que não consegue explicar o que entregou.
    Escreva os relatórios de modo que dê para defendê-los meses depois, e confira no overview
    da edição como a prova entra na nota.

## Dataset

O mesmo dataset vale para as três entregas, então a escolha feita no EDA define o que as outras
duas conseguem fazer.

| Campo | Valor |
|-------|-------|
| Nome | Vehicle dataset from CarDekho, arquivo `Car details v3.csv` |
| Fonte (URL) | <https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho> |
| Licença / termos de uso | DbCL v1.0 (Database Contents License), conforme a página do Kaggle |
| Amostras | 8.128 anúncios no arquivo bruto; 6.907 após remover duplicatas |
| Features | 12: 7 numéricas (`year`, `km_driven`, `mileage`, `engine`, `max_power`, `torque_nm`, `seats`) e 5 categóricas (`fuel`, `seller_type`, `transmission`, `owner`, `brand`) |
| Variável alvo | `selling_price`, preço de revenda em rúpias (₹), modelado como `log(selling_price)` |
| Tarefa escolhida | Regressão |

Escolhemos este dataset porque ele mistura numéricas e categóricas, traz o alvo no próprio arquivo
e tem bem mais que as 1.000 linhas mínimas. O preço tem assimetria de 5,57, quatro colunas numéricas
vêm como texto com unidades misturadas (Nm e kgm, kmpl e km/kg), 14,8% das linhas são duplicadas e
o preço depende de interações: o diesel entrega quase o dobro de torque por bhp. Mesmo assim, os
dados têm padrão de sobra para aprender, e um kNN simples sobre os dados pré-processados já chega a
R² = 0,875 no log do preço.

## Status

- [x] 1. EDA
- [x] 2. Classificação ou Regressão
- [ ] 3. Generativo

## Registro de decisões

Aqui ficam, com data, as decisões que afetam mais de uma entrega, como troca de dataset, mudança de
alvo ou corte de features. É com esse registro que dá para reconstruir o raciocínio na prova de projeto.

| Data | Decisão | Motivo |
|------|---------|--------|
| 07/10/2026 | Usar `Car details v3.csv` entre os quatro arquivos do pacote | `car data.csv` tem 301 linhas (mínimo é 1.000), `CAR DETAILS FROM CAR DEKHO.csv` não tem ficha técnica e `car details v4.csv` tem só 2.059 linhas |
| 07/10/2026 | Tarefa de regressão sobre `log(selling_price)` | A assimetria do preço cai de 5,57 para −0,16 e o erro passa a ser relativo |
| 07/10/2026 | Remover 1.221 duplicatas antes do split | Evitar a mesma linha no treino e no teste |
| 07/10/2026 | Descartar `name` e extrair `brand`; trocar `torque` (texto) por `torque_nm` | `name` tem 2.058 valores (quase um ID); o rpm do torque vem em formatos irreconciliáveis |
| 07/10/2026 | Split 80/20 estratificado por decis de preço, `random_state=42` | Garantir carros caros nos dois lados; o dataset não tem data para um split temporal |
| 07/10/2026 | Winsorizar (0,5% e 99,5%) em vez de remover outliers | A regra do IQR marcaria carros legítimos (todo carro que não tem 5 lugares, SUVs) |
