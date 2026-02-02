import numpy as np
import pandas as pd

class AnaliseRisco:
    """
    Cálculo de métricas de risco financeiro.
    """
    
    @staticmethod
    def calcular_var(retornos: pd.Series, confianca: float = 0.95) -> float:
        """
        Calcula o Value at Risk (VaR) histórico.
        """
        return np.percentile(retornos, 100 * (1 - confianca))

    @staticmethod
    def calcular_cvar(retornos: pd.Series, confianca: float = 0.95) -> float:
        """
        Calcula o Conditional Value at Risk (CVaR) / Expected Shortfall.
        """
        var = AnaliseRisco.calcular_var(retornos, confianca)
        return retornos[retornos <= var].mean()

    @staticmethod
    def calcular_drawdown_maximo(precos: pd.Series) -> float:
        """
        Calcula o Maximum Drawdown de uma série de preços.
        """
        cummax = precos.cummax()
        drawdown = (precos - cummax) / cummax
        return drawdown.min()
