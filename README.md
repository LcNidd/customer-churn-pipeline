# 🛒 E-Commerce Customer Churn Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen.svg)

Pipeline de Engenharia de Machine Learning *end-to-end* projetado para prever a probabilidade de retenção e *churn* de clientes na plataforma de e-commerce Olist (93.358 clientes únicos).

---

## 📌 Contexto & Problema de Negócio

No e-commerce, adquirir novos clientes custa significativamente mais do que reter os atuais. O objetivo deste projeto é identificar antecipadamente comportamentos de abandono (*churn* — definidos por **>90 dias sem compras**) e extrair os fatores operacionais que mais impactam essa evasão.

### Key Insights
- **Top Predictor:** O **valor médio do frete (`avg_freight_value`)** responde por **62.4%** da importância na decisão de churn do cliente.
- **Ticket Médio:** O preço médio dos itens e o valor gasto acumulado completam os principais fatores de decisão.

---

## 📊 Performance dos Modelos

Comparação entre o modelo baseline e a solução final ensemble:

| Modelo | Acurácia | ROC-AUC | Recall (Ativo) | F1-Score (Macro) |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 43% | 0.5504 | 66% | 0.51 |
| **Random Forest Classifier (Final)** | **72%** | **0.7880** | **70%** | **0.59** |

---

## 🛠️ Arquitetura do Projeto

O projeto evoluiu de uma fase experimental em Jupyter Notebooks para uma estrutura de módulos limpos e reutilizáveis:

```text
customer-churn-pipeline/
├── data/                  # Datasets brutos da Olist (CSV)
├── notebooks/             # Exploratory Data Analysis (EDA) & Model Prototyping
├── models/                # Artefactos serializados (.joblib)
├── src/                   # Módulos Python em nível de produção
│   ├── data_loader.py     # Ingestão e joins dos dados
│   ├── features.py        # Engenharia de atributos (RFM + Logística)
│   └── train.py           # Treino, avaliação e exportação do modelo
├── main.py                # CLI Entrypoint para execução do pipeline
└── requirements.txt       # Dependências do ambiente