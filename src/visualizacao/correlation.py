import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class VisualizadorCorrelacao:
    """
    Mapas de calor e análise de correlação.
    """
    
    @staticmethod
    def plotar_heatmap_correlacao(dados: pd.DataFrame, arquivo: str):
        """
        Gera heatmap de correlação de Pearson.
        """
        corr = dados.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Matriz de Correlação entre Ativos', fontsize=16)
        plt.tight_layout()
        plt.savefig(arquivo)
        plt.close()
