import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Import modules
from src.coleta.yahoo_connector import obter_dados_historicos
from src.processamento.cleaner import preencher_valores_nulos, remover_outliers
from src.processamento.normalizer import calcular_log_retornos
from src.processamento.validator import validar_dados
from src.modelos.sde import ProcessosEstocasticos
from src.modelos.statistics import ModelosEstatisticos
from src.modelos.filters import FiltrosAvancados
from src.modelos.factors import AnaliseFatores
from src.simulacao.monte_carlo import SimuladorMonteCarlo
from src.analise.risk_metrics import AnaliseRisco
from src.visualizacao.plots import VisualizadorSeries
from src.visualizacao.advanced_plots import VisualizadorAvancado
from src.analise.indicators import IndicadoresTecnicos
from src.analise.tests import TestesEstatisticos
from src.analise.ratios import RatiosAvancados
from src.visualizacao.correlation import VisualizadorCorrelacao
from src.visualizacao.surfaces import VisualizadorSuperficie

# Configuração
TICKERS = ['PETR4.SA', 'VALE3.SA', 'NVDA', 'ITUB4.SA', 'BBAS3.SA', 'WEGE3.SA', 'PRIO3.SA', 'BTC-USD', 'AAPL', 'MSFT']
DATA_INICIO = '2020-01-01'
DATA_FIM = datetime.now().strftime('%Y-%m-%d')
CAMINHO_OUT = 'docs/resultados'
os.makedirs(CAMINHO_OUT, exist_ok=True)

def main():
    print("--- Iniciando Modelo Financeiro Avançado (2026-2028) ---")
    
    # 1. Coleta e Tratamento
    print(f"[1/6] Coletando dados para: {TICKERS}")
    precos = obter_dados_historicos(TICKERS, DATA_INICIO, DATA_FIM)
    precos = preencher_valores_nulos(precos)
    
    if not validar_dados(precos):
        return

    # 2. Processamento
    print("[2/6] Processando retornos e estatísticas...")
    retornos = calcular_log_retornos(precos).dropna()
    retornos_clean = remover_outliers(retornos)
    
    # 3. Análises Preliminares e Gráficos
    print("[3/6] Gerando análises visuais iniciais e calculando 60 Módulos...")
    
    # 3.1 Indicadores Técnicos (Exemplo PETR4)
    petr4 = precos['PETR4.SA']
    petr4_ret = retornos_clean['PETR4.SA']
    
    rsi = IndicadoresTecnicos.rsi(petr4)
    macd = IndicadoresTecnicos.macd(petr4)
    bb = IndicadoresTecnicos.bollinger_bands(petr4)
    
    # Salvar gráfico de indicadores
    plt.figure(figsize=(12,8))
    plt.subplot(3,1,1)
    plt.plot(petr4, label='Preço')
    plt.plot(bb['Upper'], 'g--', alpha=0.5)
    plt.plot(bb['Lower'], 'r--', alpha=0.5)
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(rsi, label='RSI')
    plt.axhline(70, color='r', linestyle='--')
    plt.axhline(30, color='g', linestyle='--')
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(macd['MACD'], label='MACD')
    plt.plot(macd['Signal'], label='Signal')
    plt.legend()
    plt.savefig(f'{CAMINHO_OUT}/03_indicadores_tecnicos.png')
    plt.close()

    # 3.2 Testes Estatísticos (Para PETR4 Returns)
    adf = TestesEstatisticos.teste_adf(petr4_ret)
    shapiro = TestesEstatisticos.teste_shapiro(petr4_ret)
    print(f"  -> Teste ADF (Estacionariedade) PETR4: p-val={adf['p-value']:.4f} ({'Estacionaria' if adf['Estacionaria'] else 'Nao Estacionaria'})")
    print(f"  -> Teste Shapiro (Normalidade) PETR4: p-val={shapiro['p-value']:.4f} ({'Normal' if shapiro['Normal'] else 'Nao Normal'})")

    # 3.3 Ratios (Sharpe Simples)
    sharpe = RatiosAvancados.sharpe_ratio(petr4_ret, risk_free_rate=0.1375) # Selic approx
    print(f"  -> Sharpe Ratio PETR4: {sharpe:.2f}")

    # Matriz de Correlação
    VisualizadorCorrelacao.plotar_heatmap_correlacao(retornos_clean, f'{CAMINHO_OUT}/01_heatmap_correlacao.png')
    
    # PCA
    pca_df = AnaliseFatores.extrair_componentes_principais(retornos_clean, n_componentes=3)
    plt.figure(figsize=(10,6))
    plt.plot(pca_df)
    plt.title('Componentes Principais (Fatores Latentes de Mercado)')
    plt.savefig(f'{CAMINHO_OUT}/02_pca_fatores.png')
    plt.close()
    
    # 4. Modelagem e Simulação (Monte Carlo)
    print(f"[4/6] Executando Simulações de Monte Carlo (Projeção 2026-2028) para {len(TICKERS)} ativos...")
    
    simulador = SimuladorMonteCarlo(n_caminhos=1000, horizonte_anos=2.0)
    
    for i, ticker in enumerate(TICKERS):
        print(f"  -> Processando {ticker}...")
        
        # Parâmetros para GBM e Heston
        S0 = precos[ticker].iloc[-1]
        mu = retornos_clean[ticker].mean() * 252
        sigma = retornos_clean[ticker].std() * np.sqrt(252)
        
        # 4.1 Simulação GBM
        params_gbm = {'S0': S0, 'mu': mu, 'sigma': sigma}
        df_sim_gbm = simulador.executar_simulacao('gbm', params_gbm)
        resumo_gbm = simulador.agregar_resultados(df_sim_gbm)
        
        # Ajuste de datas para projeção
        resumo_gbm.index = pd.date_range(start=pd.Timestamp.now(), periods=len(resumo_gbm))
        
        # Gráficos de Projeção
        VisualizadorSeries.plotar_precos_com_projecao(
            precos[ticker], resumo_gbm['Media'], resumo_gbm['Q5'], resumo_gbm['Q95'],
            f'Projeção GBM 2026-2028: {ticker}', f'{CAMINHO_OUT}/projecao_gbm_{ticker}.png'
        )
        
        # 4.2 Modelo Heston (Simulação de Volatilidade)
        # Parâmetros estimados "na mão" para exemplo, idealmente viriam de calibração
        params_heston = {
            'S0': S0, 'v0': sigma**2, 'mu': mu, 
            'kappa': 2.0, 'theta': sigma**2, 'xi': 0.3, 'rho': -0.7
        }
        # Simulamos apenas para pegar paths de volatilidade se quiséssemos, 
        # mas aqui vamos simplificar e usar GARCH para volatilidade
        
        # 4.3 GARCH Volatility
        garch_fit = ModelosEstatisticos.ajustar_garch(retornos_clean[ticker])
        vol_proj = ModelosEstatisticos.prever_volatilidade_garch(garch_fit, horizonte=252*2)
        
        plt.figure(figsize=(10,5))
        plt.plot(vol_proj)
        plt.title(f'Projeção de Volatilidade GARCH (2 Anos): {ticker}')
        plt.savefig(f'{CAMINHO_OUT}/volatilidade_garch_{ticker}.png')
        plt.close()

        # 4.4 Filtro de Kalman
        df_kalman = FiltrosAvancados.aplicar_filtro_kalman(precos[ticker])
        plt.figure(figsize=(10,5))
        plt.plot(precos[ticker][-200:], label='Real')
        plt.plot(df_kalman['Preco_Estimado'][-200:], label='Kalman', linestyle='--')
        plt.title(f'Filtro de Kalman - Tendência Recente: {ticker}')
        plt.legend()
        plt.savefig(f'{CAMINHO_OUT}/kalman_{ticker}.png')
        plt.close()
        
    # 5. Superfícies de Volatilidade (Exemplo Sintético)
    print("[5/6] Gerando Superfícies de Volatilidade...")
    X = np.linspace(80, 120, 20)
    Y = np.linspace(0.1, 2, 20)
    X, Y = np.meshgrid(X, Y)
    Z = 0.2 + 0.05 * np.exp(-0.01*(X-100)**2) + 0.02*Y # Smile de volatilidade fake
    
    VisualizadorSuperficie.plotar_superficie_volatilidade(X, Y, Z, 'Superfície de Volatilidade Implícita (Exemplo)', f'{CAMINHO_OUT}/superficie_vol.png')

    # 6. Visualizações Avançadas (Extras)
    print("[6/6] Gerando Gráficos Avançados (Risco/Retorno, Beta, Distribuições)...")
    
    # 6.1 Fronteira Eficiente
    VisualizadorAvancado.plotar_fronteira_eficiente(retornos_clean, f'{CAMINHO_OUT}/04_fronteira_eficiente.png')
    
    # 6.2 Distribuições (PETR4)
    VisualizadorAvancado.plotar_distribuicao_retornos(retornos_clean['PETR4.SA'], 'PETR4', f'{CAMINHO_OUT}/05_dist_petr4.png')
    VisualizadorAvancado.plotar_qq(retornos_clean['PETR4.SA'], 'PETR4', f'{CAMINHO_OUT}/06_qq_petr4.png')
    
    # 6.3 Drawdown Underwater (VALE3)
    VisualizadorAvancado.plotar_drawdown_underwater(precos['VALE3.SA'], 'VALE3', f'{CAMINHO_OUT}/07_drawdown_vale3.png')
    
    # 6.4 Rolling Beta (PETR4 vs IBOV Proxy - Média)
    # Criando proxy de mercado (equiponderado)
    mercado_proxy = retornos_clean.mean(axis=1)
    VisualizadorAvancado.plotar_rolling_beta(retornos_clean['PETR4.SA'], mercado_proxy, 60, 'Rolling Beta (60d): PETR4 vs Média Mercado', f'{CAMINHO_OUT}/08_rolling_beta_petr4.png')
    
    # 6.5 ACF/PACF (Bitcoin)
    VisualizadorAvancado.plotar_acf_pacf(retornos_clean['BTC-USD'], 'BTC-USD', f'{CAMINHO_OUT}/09_acf_pacf_btc.png')
    
    # 6.6 Boxplot Volatilidade
    VisualizadorAvancado.plotar_boxplot_volatilidade(retornos_clean, f'{CAMINHO_OUT}/10_boxplot_volatilidade.png')

    print(f"--- Processamento Concluído. Resultados salvos em {CAMINHO_OUT} ---")

if __name__ == '__main__':
    main()
