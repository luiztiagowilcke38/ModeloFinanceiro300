from arch import arch_model
import numpy as np
import pandas as pd

class ModelosEstatisticos:
    """
    Modelos estatísticos avançados para séries temporais financeiras.
    """
    
    @staticmethod
    def ajustar_garch(retornos: pd.Series, p: int = 1, q: int = 1) -> object:
        """
        Ajusta um modelo GARCH(p,q) aos retornos.
        """
        if len(retornos) < 100:
             print("Aviso: Série temporal curta para GARCH.")
        
        # Reescala para evitar problemas de convergência se os valores forem muito pequenos
        escala = 100
        retornos_scaled = retornos * escala
        
        modelo = arch_model(retornos_scaled, vol='Garch', p=p, q=q)
        resultado = modelo.fit(disp='off')
        return resultado

    @staticmethod
    def prever_volatilidade_garch(resultado_garch: object, horizonte: int) -> np.ndarray:
        """
        Realiza projeção de volatilidade.
        Returns: Volatilidade desescalada.
        """
        forecasts = resultado_garch.forecast(horizon=horizonte)
        # Variância prevista -> sqrt -> desescalar
        vol_prevista = np.sqrt(forecasts.variance.values[-1, :]) / 100
        return vol_prevista
