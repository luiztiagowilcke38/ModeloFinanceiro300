import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import List, Optional

class VisualizadorSeries:
    """
    Geração de gráficos para séries temporais financeiras.
    """
    
    @staticmethod
    def plotar_precos_com_projecao(historico: pd.Series, projecao_media: pd.Series, 
                                 q5: pd.Series, q95: pd.Series, titulo: str, arquivo: str):
        """
        Plota histórico e projeção com cone de incerteza (Intervalo de Confiança).
        """
        plt.figure(figsize=(14, 7))
        plt.plot(historico.index, historico.values, label='Histórico', color='blue')
        plt.plot(projecao_media.index, projecao_media.values, label='Projeção Média', color='orange', linestyle='--')
        
        plt.fill_between(projecao_media.index, q5, q95, color='orange', alpha=0.3, label='IC 95%')
        
        plt.title(titulo, fontsize=16)
        plt.xlabel('Data')
        plt.ylabel('Preço')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(arquivo)
        plt.close()

    @staticmethod
    def plotar_comparacao_ativos(dados_normalizados: pd.DataFrame, arquivo: str):
        """
        Plota performance relativa de múltiplos ativos.
        """
        plt.figure(figsize=(14, 7))
        for coluna in dados_normalizados.columns:
            plt.plot(dados_normalizados.index, dados_normalizados[coluna], label=coluna)
            
        plt.title('Performance Relativa (Normalizada)', fontsize=16)
        plt.xlabel('Data')
        plt.ylabel('Retorno Acumulado / Valor Normalizado')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(arquivo)
        plt.close()
