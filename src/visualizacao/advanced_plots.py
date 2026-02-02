import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

class VisualizadorAvancado:
    """
    Gráficos financeiros avançados para análise de portfólio e estatística.
    """

    @staticmethod
    def plotar_fronteira_eficiente(retornos: pd.DataFrame, arquivo: str):
        """
        Plota Risco (Volatilidade) x Retorno Anual dos ativos.
        """
        retornos_anual = retornos.mean() * 252
        volatilidade_anual = retornos.std() * np.sqrt(252)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(volatilidade_anual, retornos_anual, s=100, alpha=0.7)
        
        for i, txt in enumerate(retornos.columns):
            plt.annotate(txt, (volatilidade_anual.iloc[i], retornos_anual.iloc[i]), xytext=(10,10), textcoords='offset points')
            
        plt.title('Fronteira Eficiente (Risco x Retorno)', fontsize=16)
        plt.xlabel('Volatilidade Anualizada (Risco)')
        plt.ylabel('Retorno Anualizado Esperado')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(arquivo)
        plt.close()

    @staticmethod
    def plotar_distribuicao_retornos(series: pd.Series, titulo: str, arquivo: str):
        """
        Histograma com KDE e ajuste de distribuição Normal.
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(series, kde=True, stat="density", linewidth=0, label='Dados Reais')
        
        # Ajuste Normal
        mu, std = stats.norm.fit(series)
        x = np.linspace(series.min(), series.max(), 100)
        p = stats.norm.pdf(x, mu, std)
        plt.plot(x, p, 'k--', linewidth=2, label='Normal Teórica')
        
        plt.title(f'Distribuição de Retornos: {titulo}', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(arquivo)
        plt.close()

    @staticmethod
    def plotar_qq(series: pd.Series, titulo: str, arquivo: str):
        """
        QQ-Plot para verificar normalidade.
        """
        plt.figure(figsize=(8, 6))
        sm.qqplot(series, line='s')
        plt.title(f'QQ-Plot: {titulo}', fontsize=14)
        plt.tight_layout()
        plt.savefig(arquivo)
        plt.close()

    @staticmethod
    def plotar_drawdown_underwater(precos: pd.Series, titulo: str, arquivo: str):
        """
        Gráfico Underwater de Drawdown.
        """
        rolling_max = precos.cummax()
        drawdown = (precos - rolling_max) / rolling_max
        
        plt.figure(figsize=(12, 6))
        plt.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        plt.plot(drawdown.index, drawdown, color='red', linewidth=1)
        plt.title(f'Underwater Drawdown: {titulo}', fontsize=14)
        plt.ylabel('Drawdown %')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(arquivo)
        plt.close()

    @staticmethod
    def plotar_rolling_beta(ativo: pd.Series, benchmark: pd.Series, janela: int, titulo: str, arquivo: str):
        """
        Calcula e plota o Beta móvel (Rolling Beta).
        """
        # Calcular retornos se forem preços
        # Assumindo que já entram como Retornos
        cov = ativo.rolling(window=janela).cov(benchmark)
        var = benchmark.rolling(window=janela).var()
        beta = cov / var
        
        plt.figure(figsize=(12, 6))
        plt.plot(beta.index, beta, label=f'Rolling Beta ({janela} dias)')
        plt.axhline(1, color='k', linestyle='--', alpha=0.5)
        plt.title(titulo, fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(arquivo)
        plt.close()

    @staticmethod
    def plotar_acf_pacf(series: pd.Series, titulo: str, arquivo: str):
        """
        Plota Autocorrelação (ACF) e Autocorrelação Parcial (PACF).
        """
        # Utiliza statsmodels plot_acf e plot_pacf
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        plot_acf(series, ax=ax1, lags=40, title=f'Autocorrelação (ACF) - {titulo}')
        plot_pacf(series, ax=ax2, lags=40, title=f'Autocorrelação Parcial (PACF) - {titulo}')
        plt.tight_layout()
        plt.savefig(arquivo)
        plt.close()

    @staticmethod
    def plotar_boxplot_volatilidade(retornos: pd.DataFrame, arquivo: str):
        """
        Boxplot comparativo da volatilidade (retornos absolutos ou quadrados).
        """
        # Usando volatilidade realizada (retorno absoluto como proxy simples visual)
        vol_proxy = retornos.abs()
        
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=vol_proxy)
        plt.title('Dispersão de Volatilidade (Retornos Absolutos)', fontsize=16)
        plt.ylim(0, vol_proxy.quantile(0.99).max() * 1.5) # Limitar y para visualização
        plt.tight_layout()
        plt.savefig(arquivo)
        plt.close()
