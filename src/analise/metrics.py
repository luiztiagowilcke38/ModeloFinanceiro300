import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class MetricasErro:
    """
    Cálculo de métricas de erro para validação de modelos.
    """
    
    @staticmethod
    def calcular_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def calcular_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def calcular_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
