import numpy as np
import pandas as pd
from typing import Tuple

class ProcessosEstocasticos:
    """
    Implementação de Equações Diferenciais Estocásticas (SDEs) para modelagem financeira.
    """
    
    @staticmethod
    def gbm(S0: float, mu: float, sigma: float, T: float, dt: float, n_caminhos: int) -> np.ndarray:
        """
        Movimento Browniano Geométrico (GBM).
        dS_t = mu*S_t*dt + sigma*S_t*dW_t
        """
        N = int(T / dt)
        t = np.linspace(0, T, N)
        W = np.random.standard_normal(size=(n_caminhos, N)) 
        W = np.cumsum(W, axis=1) * np.sqrt(dt) ### Standard Brownian Motion
        
        X = (mu - 0.5 * sigma**2) * t + sigma * W
        S = S0 * np.exp(X)
        return S

    @staticmethod
    def heston(S0: float, v0: float, mu: float, kappa: float, theta: float, xi: float, 
               rho: float, T: float, dt: float, n_caminhos: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Modelo de Heston (Volatilidade Estocástica).
        dS_t = mu*S_t*dt + sqrt(v_t)*S_t*dW_t^S
        dv_t = kappa*(theta - v_t)*dt + xi*sqrt(v_t)*dW_t^v
        """
        N = int(T / dt)
        S = np.zeros((n_caminhos, N))
        v = np.zeros((n_caminhos, N))
        S[:, 0] = S0
        v[:, 0] = v0
        
        for t in range(1, N):
            # Correlated Brownian Motions
            Z1 = np.random.standard_normal(n_caminhos)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.standard_normal(n_caminhos)
            
            # Euler-Maruyama discretization for Heston
            # Ensure volatility is positive (Full Truncation or Reflection)
            v_prev = np.maximum(v[:, t-1], 0)
            
            v[:, t] = v[:, t-1] + kappa * (theta - v_prev) * dt + xi * np.sqrt(v_prev) * np.sqrt(dt) * Z2
            S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * v_prev) * dt + np.sqrt(v_prev) * np.sqrt(dt) * Z1)
            
        return S, v

    @staticmethod
    def ornstein_uhlenbeck(X0: float, theta: float, mu: float, sigma: float, T: float, dt: float, n_caminhos: int) -> np.ndarray:
        """
        Processo de Ornstein-Uhlenbeck (Mean Reversion).
        dX_t = theta*(mu - X_t)*dt + sigma*dW_t
        """
        N = int(T / dt)
        X = np.zeros((n_caminhos, N))
        X[:, 0] = X0
        
        for t in range(1, N):
            W = np.random.standard_normal(n_caminhos)
            X[:, t] = X[:, t-1] + theta * (mu - X[:, t-1]) * dt + sigma * np.sqrt(dt) * W
            
        return X
