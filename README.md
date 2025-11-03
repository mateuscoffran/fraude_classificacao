# 🔐 Detecção de Transações Fraudulentas

****

## 🔍 Sobre o Projeto

O objetivo deste projeto é **construir Modelos de Machine Learning capazes de detectar transações fraudulentas** com a melhor precisão a fim de **minimizar as perdas financeiras** potenciais geradas por fraudes.

O projeto foi desenvolvido em Jupyter Notebook. Todo o trabalho foi elaborado a partir de uma base de dados em Excel com diversas informações de transações financeiras. 

A base de dados do case está disponível no seguinte link: Preparatório para Entrevistas em Dados (PED).

## 🛠️ Bibliotecas principais
- `pandas` - Manipulação de dados 
- `scikit-learn` - Modelos de ML e métricas 
- `numpy` - Operações numéricas 
- `matplotlib` / `seaborn` - Visualizações 
- `feature_engine` - Engenharia de features 
- `optuna` - Otimização de hiperparâmetros 
- `shap` - Interpretabilidade

## 📊 Metodologia 
### Abordagem de Desenvolvimento 
O projeto seguiu uma metodologia iterativa em ciclos, permitindo: 
- Avaliação incremental do impacto de cada transformação 
- Validação contínua da performance 
- Entregas de valor progressivas 

### Etapas Implementadas 

#### 1️⃣ ** Preparação dos Dados** 
- Análise de desbalanceamento das classes 
- Detecção de dados faltantes 

#### 2️⃣ **Engenharia de Features**
Foram experimentadas e avaliadas as seguintes técnicas 
- **Encoding de variáveis categóricas:** Target Encoding e CatBoostEncoder
- **Imputação robusta:** KNN Imputer
- **Features temporais:** Extração e transformação de ciclicidade (aplicação de seno e cosseno em dia, mês, etc)
 - **Discretização:** K-Means Discretizer, Decision Tree Discretizer 
- **Features polinomiais:** Interações entre features relevantes 
- **Padronização dos Dados:** StandardScaler e RobustScaler
- **Seleção de features:** Método Boruta 

#### 3️⃣ **Modelagem** 
Foram treinados e avaliados **8 modelos** de diferentes famílias de algoritmos: 
| Modelo | Tipo |
 |--------|------|
| Regressão Logística | Linear | 
| SVM (Support Vector Machine) | Linear | 
| Árvore de Decisão | Tree-based | 
| Random Forest | Ensemble (Bagging) |
 | CatBoost | Gradient Boosting | 
| LightGBM | Gradient Boosting | 
| XGBoost | Gradient Boosting | 
| Rede Neural Artificial | Deep Learning |

#### 4️⃣ **Otimização** 
- **Tunagem de hiperparâmetros:** Optuna (Bayesian Optimization) 
- **Ajuste de threshold:** TunedThresholdClassifierCV 
- **Tratamento de desbalanceamento:** `class_weight='balanced'`

--- 
## 📈 Métricas de Avaliação 
### Métricas Utilizadas 
- **Recall** (Sensibilidade) 
- **Precision** (Precisão) 
- **F1-Score** (Média harmônica entre Precision e Recall) 
- **F2-Score** (**Métrica principal** - maior peso para Recall do que para o Precision) 
- **ROC-AUC** (Área sob a curva ROC) 
- **Acurácia** 
- **Matriz de Confusão**

### 🎯 Métrica Principal: F2-Score 
Neste case de fraude, não temos explicitamente o custo da fraude e o custo de oportunidade de mover recursos para tratar uma transação normal como fraudulenta
A escolha do **F2-Score** como métrica principal se justifica pelo contexto do negócio: 
``` 
F2-Score = (1 + β²) × (Precision × Recall) / (β² × Precision + Recall) onde β = 2 (β² = 4) 
```
 **Justificativa:** 
- O **custo de uma fraude não detectada** (Falso Negativo) é significativamente **superior** ao custo de investigar uma transação legítima (Falso Positivo)
- O F2-Score dá **4x mais peso ao Recall**, priorizando a captura de fraudes 
- Mas ainda considera o Precision, evitando excesso de alarmes falsos que sobrecarregam a operação.

## 🏆 Resultados

Após Engenharia de Features, Seleção de variáveis (Boruta), Otimização de  E hiperparâmetros (Optuna) ajuste do Threshold de decisão, O modelo com **melhor desempenho preditivo** foi o **XGBoost**.

## 🧠 Interpretabilidade 
Utilizando **SHAP (SHapley Additive exPlanations)**, foram identificadas as features mais importantes para a detecção de fraudes: 
1. doc_2_vazio^2
2. categoria_produto
3. score_9
