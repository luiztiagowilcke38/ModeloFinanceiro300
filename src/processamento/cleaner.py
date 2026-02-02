import pandas as pd
import numpy as np

def preencher_valores_nulos(df: pd.DataFrame, metodo: str = 'ffill') -> pd.DataFrame:
    """
    Preenche valores nulos em um DataFrame.
    """
    if metodo == 'ffill':
        return df.ffill().bfill()
    elif metodo == 'interpolate':
        return df.interpolate()
    else:
        return df.dropna()

def remover_outliers(df: pd.DataFrame, z_thresh: float = 3.0) -> pd.DataFrame:
    """
    Remove ou substitui outliers baseados no Z-Score.
    Neste caso, substituiremos por NaN e depois interpolaremos.
    """
    from scipy.stats import zscore
    
    df_clean = df.copy()
    z_scores = np.abs(zscore(df_clean.select_dtypes(include=[np.number]).dropna()))
    
    # Simples remoção por enquanto, pode ser aprimorado para winsorização
    mask = (z_scores < z_thresh).all(axis=1)
    
    # Alinhamento de índices pode ser complicado aqui se tiver NaNs antes
    # Abordagem simplificada: clipar valores
    return df_clean.clip(lower=df_clean.quantile(0.01), upper=df_clean.quantile(0.99), axis=1)
