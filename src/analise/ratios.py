import pandas as pd
import numpy as np
from src.analise.risk_metrics import AnaliseRisco

class RatiosAvancados:
    """
    Biblioteca de 6 Ratios de Performance Financeira.
    """

    @staticmethod
    def sharpe_ratio(retornos: pd.Series, risk_free_rate: float = 0.0) -> float:
        """1. Sharpe Ratio
        (Rp - Rf) / sigma_p
        """
        excesso_retorno = retornos - risk_free_rate/252
        if excesso_retorno.std() == 0: return 0.0
        return (excesso_retorno.mean() / excesso_retorno.std()) * np.sqrt(252)

    @staticmethod
    def sortino_ratio(retornos: pd.Series, risk_free_rate: float = 0.0, target_return: float = 0.0) -> float:
        """2. Sortino Ratio
        (Rp - Rf) / downside_deviation
        """
        excesso_retorno = retornos - risk_free_rate/252
        downside = retornos[retornos < target_return]
        downside_std = downside.std() * np.sqrt(252)
        
        if downside_std == 0: return 0.0
        return (excesso_retorno.mean() * 252) / downside_std

    @staticmethod
    def calmar_ratio(retornos: pd.Series, risk_free_rate: float = 0.0) -> float:
        """3. Calmar Ratio
        (Rp - Rf) / Max Drawdown
        """
        # Necessita serie de precos para drawdown, vou assumir retornos cumulativos
        precos_ficticios = (1 + retornos).cumprod()
        max_dd = abs(AnaliseRisco.calcular_drawdown_maximo(precos_ficticios))
        retorno_anual = retornos.mean() * 252
        
        if max_dd == 0: return 0.0
        return retorno_anual / max_dd

    @staticmethod
    def information_ratio(retornos_ativo: pd.Series, retornos_benchmark: pd.Series) -> float:
        """4. Information Ratio
        (Rp - Rb) / Tracking Error
        """
        diferenca = retornos_ativo - retornos_benchmark
        tracking_error = diferenca.std() * np.sqrt(252)
        
        if tracking_error == 0: return 0.0
        return (diferenca.mean() * 252) / tracking_error

    @staticmethod
    def treynor_ratio(retornos_ativo: pd.Series, retornos_mercado: pd.Series, risk_free_rate: float = 0.0) -> float:
        """5. Treynor Ratio
        (Rp - Rf) / Beta
        """
        beta = RatiosAvancados.beta(retornos_ativo, retornos_mercado)
        retorno_ativo_anual = retornos_ativo.mean() * 252
        
        if beta == 0: return 0.0
        return (retorno_ativo_anual - risk_free_rate) / beta

    @staticmethod
    def beta(retornos_ativo: pd.Series, retornos_mercado: pd.Series) -> float:
        """6. Beta Coefficient"""
        # Covariancia / Variancia Mercado
        matrix = np.cov(retornos_ativo, retornos_mercado)
        beta_val = matrix[0, 1] / matrix[1, 1]
        return beta_val
