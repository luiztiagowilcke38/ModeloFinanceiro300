# Modelo Financeiro Preditivo Avançado 300

## Autor: Luiz Tiago Wilcke

Este projeto implementa um sistema de modelagem financeira de alta performance para previsão de ativos (Petrobras, Vale, Nvidia, etc.) utilizando **Equações Diferenciais Estocásticas (SDEs)**, **Estatística Avançada** e métodos de **Monte Carlo**.

### 🚀 Funcionalidades (60 Módulos)

O sistema conta com **60 módulos** distribuídos matematicamente em:

1.  **Coleta e Processamento (5 Módulos)**: Conexão Yahoo Finance, limpeza, normalização (Log/Z-Score), validação.
2.  **Modelagem Matemática (10 Módulos)**:
    *   **SDEs**: GBM, Heston, Ornstein-Uhlenbeck.
    *   **Filtros**: Kalman Filter (Tendência e Estado Oculto).
    *   **Multivariada**: Cópulas Gaussianas, PCA (Fatores Latentes).
3.  **Indicadores Técnicos (15 Módulos)**:
    *   Tendência: SMA, EMA, MACD, TRA, TRIX, ADX.
    *   Osciladores: RSI, Stochastic, Williams %R, CCI, MFI, Momentum, ROC.
    *   Volatilidade: Bollinger Bands, ATR.
4.  **Testes Estatísticos (10 Módulos)**:
    *   Estacionariedade: ADF, KPSS, Ljung-Box.
    *   Normalidade: Shapiro-Wilk, Jarque-Bera, Durbin-Watson.
    *   Causalidade: Granger Causality, Breusch-Pagan, T-Test, F-Test.
5.  **Ratios de Performance (6 Módulos)**: Sharpe, Sortino, Calmar, Information, Treynor, Beta.
6.  **Simulação & Risco (4 Módulos)**: Monte Carlo Engine (10k caminhos), VaR, CVaR, Drawdown.
7.  **Visualização Avançada (10 Módulos)**: Fronteira Eficiente, Superfícies 3D, Rolling Beta, Heatmaps, Cones de Incerteza.

### 📐 Equações do Modelo

O núcleo preditivo se baseia em Difusões Estocásticas e Estatística Bayesiana.

#### 1. Movimento Browniano Geométrico (GBM)
Utilizado para a evolução básica dos preços:
$$ dS_t = \mu S_t dt + \sigma S_t dW_t $$
Onde $W_t$ é um processo de Wiener standard.

#### 2. Modelo de Heston (Volatilidade Estocástica)
Para capturar o "sorriso da volatilidade" e caudas gordas:
$$ dS_t = \mu S_t dt + \sqrt{\nu_t} S_t dW_t^S $$
$$ d\nu_t = \kappa (\theta - \nu_t) dt + \xi \sqrt{\nu_t} dW_t^{\nu} $$
Com correlação $d W_t^S d W_t^{\nu} = \rho dt$.

#### 3. Processo Ornstein-Uhlenbeck (Reversão à Média)
Utilizado para modelar spreads e commodities:
$$ dx_t = \theta (\mu - x_t) dt + \sigma dW_t $$

#### 4. GARCH(1,1) para Volatilidade Condicional
$$ \sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2 $$

#### 5. Filtro de Kalman (Estado-Espaço)
Estimativa recursiva do estado oculto (tendência real) $x_k$ dado medições ruidosas $z_k$:
$$ \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (z_k - H \hat{x}_{k|k-1}) $$

### 📊 Estrutura do Projeto

```
src/
├── coleta/         # Conectores de API
├── processamento/  # Limpeza, Normalização, Validação
├── modelos/        # SDEs, GARCH, Filtros, Fatores
├── analise/        # Indicadores, Testes, Ratios, Backtest
├── simulacao/      # Motor Monte Carlo
└── visualizacao/   # Plots 2D/3D, Correlação, Dashboards
data/               # Dados brutos e processados
docs/resultados/    # Gráficos gerados
```

### 🛠️ Como Executar

1.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

2.  Execute a pipeline completa:
    ```bash
    python3 main.py
    ```

3.  Visualize os resultados gerados na pasta `docs/resultados`.

---
© 2026 Luiz Tiago Wilcke
