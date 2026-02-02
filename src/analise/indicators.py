import pandas as pd
import numpy as np

class IndicadoresTecnicos:
    """
    Biblioteca de 15 Indicadores Técnicos Avançados.
    """

    @staticmethod
    def sma(series: pd.Series, window: int = 14) -> pd.Series:
        """1. Simple Moving Average"""
        return series.rolling(window=window).mean()

    @staticmethod
    def ema(series: pd.Series, window: int = 14) -> pd.Series:
        """2. Exponential Moving Average"""
        return series.ewm(span=window, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        """3. Relative Strength Index"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """4. Moving Average Convergence Divergence"""
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return pd.DataFrame({'MACD': macd_line, 'Signal': signal_line})

    @staticmethod
    def bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2) -> pd.DataFrame:
        """5. Bollinger Bands"""
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        return pd.DataFrame({'Upper': upper, 'Lower': lower})

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """6. Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """7. Stochastic Oscillator"""
        lowest_low = low.rolling(window=window).min()
        highest_high = high.rolling(window=window).max()
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        return k

    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """8. On-Balance Volume"""
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """9. Average Directional Index (Simplificado)"""
        # Implementação completa requer +DI e -DI, simplificando para exemplo
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        tr = IndicadoresTecnicos.atr(high, low, close, window)
        plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / tr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/window).mean() / tr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        return dx.ewm(alpha=1/window).mean()

    @staticmethod
    def momentum(series: pd.Series, window: int = 10) -> pd.Series:
        """10. Momentum"""
        return series.diff(window)

    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """11. Williams %R"""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """12. Commodity Channel Index"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=window).mean()
        mean_dev = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
        return (tp - sma_tp) / (0.015 * mean_dev)

    @staticmethod
    def roc(series: pd.Series, window: int = 12) -> pd.Series:
        """13. Rate of Change"""
        return ((series - series.shift(window)) / series.shift(window)) * 100

    @staticmethod
    def mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
        """14. Money Flow Index"""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=window).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=window).sum()
        
        mfi_ratio = positive_flow / negative_flow
        return 100 - (100 / (1 + mfi_ratio))

    @staticmethod
    def trix(series: pd.Series, window: int = 15) -> pd.Series:
        """15. TRIX"""
        ema1 = series.ewm(span=window, adjust=False).mean()
        ema2 = ema1.ewm(span=window, adjust=False).mean()
        ema3 = ema2.ewm(span=window, adjust=False).mean()
        return ema3.pct_change() * 100
