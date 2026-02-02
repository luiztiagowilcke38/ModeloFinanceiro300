from statsmodels.tsa.stattools import adfuller, kpss, grangercausalitytests
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro, jarque_bera, ttest_ind, f_oneway
import pandas as pd
import numpy as np
from typing import Dict, Any

class TestesEstatisticos:
    """
    Biblioteca de 10 Testes Estatísticos Rigorosos.
    """

    @staticmethod
    def teste_adf(series: pd.Series) -> Dict[str, Any]:
        """1. Augmented Dickey-Fuller Test (Estacionariedade)"""
        result = adfuller(series.dropna())
        return {'ADF Statistic': result[0], 'p-value': result[1], 'Estacionaria': result[1] < 0.05}

    @staticmethod
    def teste_kpss(series: pd.Series) -> Dict[str, Any]:
        """2. KPSS Test (Estacionariedade)"""
        result = kpss(series.dropna(), regression='c', nlags="auto")
        return {'KPSS Statistic': result[0], 'p-value': result[1], 'Estacionaria': result[1] > 0.05}

    @staticmethod
    def teste_shapiro(series: pd.Series) -> Dict[str, Any]:
        """3. Shapiro-Wilk Test (Normalidade)"""
        if len(series) > 5000:
            stat, p = shapiro(series.sample(5000))
        else:
            stat, p = shapiro(series.dropna())
        return {'Statistic': stat, 'p-value': p, 'Normal': p > 0.05}

    @staticmethod
    def teste_jarque_bera(series: pd.Series) -> Dict[str, Any]:
        """4. Jarque-Bera Test (Normalidade - Kurtosis/Skew)"""
        stat, p = jarque_bera(series.dropna())
        return {'Statistic': stat, 'p-value': p, 'Normal': p > 0.05}

    @staticmethod
    def teste_durbin_watson(residuos: np.ndarray) -> float:
        """5. Durbin-Watson Test (Autocorrelação dos resíduos)"""
        return durbin_watson(residuos)

    @staticmethod
    def teste_breusch_pagan(residuos: np.ndarray, exog: np.ndarray) -> Dict[str, Any]:
        """6. Breusch-Pagan Test (Heterocedasticidade)"""
        # Simplificação: exog precisa ser matriz de design
        # Aqui, vamos assumir que o usuário trata exog corretamente ou retornar placeholder se erro
        try:
            test = het_breuschpagan(residuos, exog)
            return {'LM Statistic': test[0], 'p-value': test[1], 'Homocedastico': test[1] > 0.05}
        except:
            return {'Error': 'Requer matriz exógena válida'}

    @staticmethod
    def teste_granger(data: pd.DataFrame, maxlag: int = 4) -> Dict[str, Any]:
        """7. Granger Causality Test"""
        # Data deve ter 2 colunas: [Target, Predictor]
        try:
            test = grangercausalitytests(data, maxlag=maxlag, verbose=False)
            # Retorna p-value do lag 1
            p_val = test[1][0]['ssr_ftest'][1]
            return {'Lag 1 p-value': p_val, 'Causa Granger': p_val < 0.05}
        except:
            return {'Error': 'Falha no teste de Granger'}

    @staticmethod
    def teste_ljung_box(series: pd.Series, lags: int = 10) -> Dict[str, Any]:
        """8. Ljung-Box Test (Autocorrelação)"""
        lb = acorr_ljungbox(series.dropna(), lags=[lags])
        p_val = lb.iloc[0, 1]
        return {'p-value': p_val, 'Ruido Branco': p_val > 0.05}

    @staticmethod
    def teste_t(amostra1: pd.Series, amostra2: pd.Series) -> Dict[str, Any]:
        """9. T-Test (Comparação de Médias)"""
        stat, p = ttest_ind(amostra1.dropna(), amostra2.dropna())
        return {'Statistic': stat, 'p-value': p, 'Medias Iguais': p > 0.05}

    @staticmethod
    def teste_f(amostras: list) -> Dict[str, Any]:
        """10. F-Test (ANOVA - Comparação de Variâncias/Médias Múltiplas)"""
        safe_amostras = [a.dropna() for a in amostras]
        stat, p = f_oneway(*safe_amostras)
        return {'Statistic': stat, 'p-value': p, 'Grupos Iguais': p > 0.05}
