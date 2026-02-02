import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

class VisualizadorSuperficie:
    """
    Visualização 3D de superfícies (ex: Volatilidade Local).
    """
    
    @staticmethod
    def plotar_superficie_volatilidade(X: np.ndarray, Y: np.ndarray, Z: np.ndarray, titulo: str, arquivo: str):
        """
        Plota superfície 3D.
        X: Moneyness ou Strike
        Y: Tempo até vencimento
        Z: Volatilidade Implícita
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        ax.set_title(titulo, fontsize=16)
        ax.set_xlabel('Preço / Strike')
        ax.set_ylabel('Tempo (Anos)')
        ax.set_zlabel('Volatilidade')
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout()
        plt.savefig(arquivo)
        plt.close()
