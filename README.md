# Predição de Ações com LSTM

## 📊 Visão Geral

Este projeto implementa um modelo LSTM (Long Short-Term Memory) para predição de preços de ações utilizando PyTorch. O sistema inclui coleta de dados históricos, pré-processamento, treinamento do modelo e visualização de resultados.

## 🚀 Funcionalidades

- Coleta automática de dados históricos de ações via Yahoo Finance
- Pré-processamento e normalização dos dados
- Modelo LSTM customizável para predição de séries temporais
- Sistema completo de logging e visualização do treinamento
- Geração de relatórios e métricas de performance
- Dashboard para análise de resultados (ainda não é interativo, são próximos passos)

## 📁 Estrutura do Projeto

```
├── checkpoints/         # Modelos salvos e checkpoints
├── data/               # Dados brutos e processados
├── logs/               # Logs de treinamento
├── plots/              # Gráficos e visualizações
├── scripts/            # Scripts principais
│   ├── collect_data.py # Coleta de dados
│   ├── train.py       # Treinamento do modelo
│   └── model.py       # Definição do modelo
└── utils/              # Utilitários
    ├── logger.py      # Sistema de logging
    ├── reporter.py    # Geração de relatórios
    ├── scaler.py     # Normalização de dados
    └── visualizer.py  # Visualizações
```

## 🔧 Requisitos

- Python 3.8+
- PyTorch
- pandas
- yfinance
- polars
- matplotlib
- tqdm
- colorama

## ⚙️ Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/stock-lstm-project.git

# Instale as dependências
pip install -r requirements.txt
```

## 📈 Uso

### 1. Coleta de Dados

```bash
python scripts/collect_data.py --symbol AAPL --start 2010-01-01 --end 2023-12-31
```

### 2. Pré-processamento

Aqui por enquanto apenas apresentamos os dados coletados, modificações futuras para melhorar o pré processamento, otimização de memória e feature engineering para melhora de desempenho

```bash
python scripts/preprocess_data.py --file data/raw/2023.parquet
```

### 3. Treinamento

```bash
python scripts/train.py --epochs 50 --batch-size 32 --lr 0.001
```

### 4. Visualização após estar treinado

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pth --data_dir data/raw --output_dir plots
```

## 📊 Visualização de Resultados

O projeto gera automaticamente:

- Gráficos de perdas de treino/validação
- Comparações entre predições e valores reais
- Distribuição de erros
- Métricas de performance (MAE, RMSE, MAPE)

## 📝 Logs e Relatórios

- Logs detalhados são salvos em `logs/`
- Relatórios de treinamento em formato JSON
- Métricas e parâmetros são registrados para cada experimento
