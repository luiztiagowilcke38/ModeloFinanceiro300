import pandas as pd

def validar_dados(df: pd.DataFrame) -> bool:
    """
    Verifica se os dados estão prontos para modelagem.
    """
    if df.isnull().any().any():
        print("Aviso: Existem valores nulos no DataFrame.")
        return False
    
    if len(df) < 30:
        print("Erro: Dados insuficientes para análise estatística robusta (< 30 amostras).")
        return False
        
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Erro: O índice do DataFrame deve ser DatetimeIndex.")
        return False
        
    return True
