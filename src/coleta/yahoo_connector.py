import yfinance as yf
import pandas as pd
from typing import List, Optional

def obter_dados_historicos(tickers: List[str], inicio: str, fim: str) -> pd.DataFrame:
    """
    Obtém dados históricos de ações do Yahoo Finance.
    
    Args:
        tickers: Lista de símbolos das ações (ex: ['PETR4.SA', 'VALE3.SA'])
        inicio: Data inicial no formato 'YYYY-MM-DD'
        fim: Data final no formato 'YYYY-MM-DD'
        
    Returns:
        DataFrame com os dados de fechamento ajustado.
    """
    try:
        dados = yf.download(tickers, start=inicio, end=fim)
        
        if dados.empty:
            raise ValueError("Nenhum dado retornado. Verifique os tickers e as datas.")
            
        # Tenta pegar Fechamento Ajustado, senão Fechamento
        if 'Adj Close' in dados.columns.get_level_values(0):
            return dados['Adj Close']
        elif 'Close' in dados.columns.get_level_values(0):
            return dados['Close']
        else:
            # Fallback: se não tiver MultiIndex ou estrutura diferente
            return dados
    except Exception as e:
        print(f"Erro ao baixar dados: {e}")
        return pd.DataFrame()

def obter_dados_completos(tickers: List[str], inicio: str, fim: str) -> pd.DataFrame:
    """
    Obtém dados completos (OHLCV) de ações do Yahoo Finance.
    """
    try:
        dados = yf.download(tickers, start=inicio, end=fim, group_by='ticker')
        return dados
    except Exception as e:
        print(f"Erro ao baixar dados completos: {e}")
        return pd.DataFrame()
