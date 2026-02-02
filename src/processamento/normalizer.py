import pandas as pd
import numpy as np

def calcular_log_retornos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o log-retorno dos preços.
    """
    return np.log(df / df.shift(1))

def normalizar_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza os dados usando Z-Score (StandardScaler).
    """
    return (df - df.mean()) / df.std()

def calcular_volatilidade_movel(retornos: pd.DataFrame, janela: int = 21) -> pd.DataFrame:
    """
    Calcula a volatilidade histórica móvel anualizada.
    """
    return retornos.rolling(window=janela).std() * np.sqrt(252)
