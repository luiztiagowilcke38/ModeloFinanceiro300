import pandas as pd
import numpy as np
from typing import Callable

class Backtest:
    """
    Framework de backtesting para estratégias e modelos.
    """
    
    @staticmethod
    def executar_backtest(series_precos: pd.Series, funcao_sinal: Callable[[pd.Series], int], janela: int = 50) -> pd.DataFrame:
        """
        Executa um backtest simples baseado em sinais.
        """
        sinais = []
        retornos_estrategia = []
        
        for i in range(janela, len(series_precos)):
            janela_dados = series_precos.iloc[i-janela:i]
            sinal = funcao_sinal(janela_dados)  # 1 (Compra), -1 (Venda), 0 (Neutro)
            sinais.append(sinal)
            
            # Retorno do dia seguinte
            retorno_dia = (series_precos.iloc[i] / series_precos.iloc[i-1]) - 1
            retornos_estrategia.append(sinal * retorno_dia)
            
        df_res = pd.DataFrame({
            'Sinal': sinais,
            'Retorno_Estrategia': retornos_estrategia
        }, index=series_precos.index[janela:])
        
        df_res['Retorno_Acumulado'] = (1 + df_res['Retorno_Estrategia']).cumprod()
        return df_res
