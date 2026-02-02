from src.modelos.sde import ProcessosEstocasticos
import numpy as np
import pandas as pd
from typing import Dict, Any

class SimuladorMonteCarlo:
    """
    Motor de simulação de Monte Carlo para projeção de preços.
    """
    
    def __init__(self, n_caminhos: int = 10000, horizonte_anos: float = 2.0):
        self.n_caminhos = n_caminhos
        self.horizonte = horizonte_anos
        self.dt = 1/252  # Passos diários
        
    def executar_simulacao(self, modelo: str, parametros: Dict[str, Any]) -> pd.DataFrame:
        """
        Executa a simulação baseada no modelo escolhido (GBM, Heston, etc).
        """
        if modelo == 'gbm':
            caminhos = ProcessosEstocasticos.gbm(
                S0=parametros['S0'],
                mu=parametros['mu'],
                sigma=parametros['sigma'],
                T=self.horizonte,
                dt=self.dt,
                n_caminhos=self.n_caminhos
            )
        elif modelo == 'heston':
            caminhos, _ = ProcessosEstocasticos.heston(
                S0=parametros['S0'],
                v0=parametros['v0'],
                mu=parametros['mu'],
                kappa=parametros['kappa'],
                theta=parametros['theta'],
                xi=parametros['xi'],
                rho=parametros['rho'],
                T=self.horizonte,
                dt=self.dt,
                n_caminhos=self.n_caminhos
            )
        else:
            raise ValueError(f"Modelo '{modelo}' não suportado.")
            
        # Criar DataFrame com timestamps futuros
        dias_uteis = int(self.horizonte * 252)
        dates = pd.date_range(start=pd.Timestamp.now(), periods=dias_uteis)
        
        # Ajustar para garantir que o numero de periodos bata com a simulacao
        # SDE retorna N pontos baseados em T/dt.
        # Se T=2, dt=1/252 => N=504.
        
        # Transpor para ter Caminhos nas colunas e Datas no índex? 
        # Ou Datas no index e colunas Caminho_0, Caminho_1...
        # Como são 10k caminhos, melhor manter numpy array interno até agregação 
        # ou retornar um DataFrame resumido (quantis).
        
        return pd.DataFrame(data=caminhos.T) # Linhas = Tempo, Colunas = Caminhos
    
    def agregar_resultados(self, df_simulacao: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula média, mediana e intervalos de confiança (5%, 95%) dos caminhos.
        """
        resumo = pd.DataFrame()
        resumo['Media'] = df_simulacao.mean(axis=1)
        resumo['Mediana'] = df_simulacao.median(axis=1)
        resumo['Q5'] = df_simulacao.quantile(0.05, axis=1)
        resumo['Q95'] = df_simulacao.quantile(0.95, axis=1)
        return resumo
