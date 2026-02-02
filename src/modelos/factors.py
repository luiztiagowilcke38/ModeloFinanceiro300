from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

class AnaliseFatores:
    """
    Análise de fatores latentes usando PCA.
    """
    
    @staticmethod
    def extrair_componentes_principais(df_retornos: pd.DataFrame, n_componentes: int = 3) -> pd.DataFrame:
        """
        Extrai os componentes principais dos retornos dos ativos.
        Útil para identificar 'fatores de mercado' não observáveis.
        """
        # Tratar NaNs
        df_clean = df_retornos.dropna()
        
        scaler = StandardScaler()
        dados_norm = scaler.fit_transform(df_clean)
        
        pca = PCA(n_components=n_componentes)
        componentes = pca.fit_transform(dados_norm)
        
        cols = [f'PC{i+1}' for i in range(n_componentes)]
        df_pca = pd.DataFrame(componentes, columns=cols, index=df_clean.index)
        
        explained_variance = pca.explained_variance_ratio_
        print(f"Variância explicada pelos {n_componentes} componentes: {np.sum(explained_variance):.2%}")
        
        return df_pca
