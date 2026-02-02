from filterpy.kalman import KalmanFilter
import numpy as np
import pandas as pd

class FiltrosAvancados:
    """
    Implementação de filtros para estimativa de estados latentes.
    """
    
    @staticmethod
    def aplicar_filtro_kalman(dados: pd.Series) -> pd.DataFrame:
        """
        Aplica Filtro de Kalman para suavização de preços e estimativa de tendência.
        Modelo Local Level (Random Walk + Ruído).
        """
        kf = KalmanFilter(dim_x=2, dim_z=1)
        
        # State Transition Matrix
        # x_t = x_{t-1} + v_{t-1} * dt
        # v_t = v_{t-1}
        # dt = 1
        kf.F = np.array([[1., 1.],
                         [0., 1.]])
        
        # Measurement Function
        # z_t = x_t
        kf.H = np.array([[1., 0.]])
        
        # Covariance Matrices
        kf.P *= 1000.                  # Initial uncertainty
        kf.R = 5                       # Measurement noise
        kf.Q = np.array([[0.1, 0.1],   # Process noise
                         [0.1, 0.1]])

        # Initial State
        kf.x = np.array([dados.iloc[0], 0.])
        
        estados_estimados = []
        for z in dados:
            kf.predict()
            kf.update(z)
            estados_estimados.append(kf.x.copy())
            
        df_res = pd.DataFrame(estados_estimados, columns=['Preco_Estimado', 'Tendencia'], index=dados.index)
        return df_res
