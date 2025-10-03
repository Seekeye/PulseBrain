#!/usr/bin/env python3
"""
Indicadores de Tendencia - CryptoPulse Pro
SMA, EMA, MACD, ADX, Parabolic SAR
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from utils.logger import get_logger, log_function_call

class TrendIndicators:
    """
    Calculadora de indicadores de tendencia
    - Simple Moving Average (SMA)
    - Exponential Moving Average (EMA)
    - MACD (Moving Average Convergence Divergence)
    - ADX (Average Directional Index)
    - Parabolic SAR
    """
    
    def __init__(self):
        self.logger = get_logger("TrendIndicators")
        self.logger.info("📈 TrendIndicators initialized")
    
    @log_function_call
    async def calculate_all(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular todos los indicadores de tendencia"""
        try:
            indicators = {}
            
            # SMAs
            indicators.update(await self.calculate_sma(df, [20, 50, 200]))
            
            # EMAs
            indicators.update(await self.calculate_ema(df, [12, 26]))
            
            # MACD
            macd_data = await self.calculate_macd(df)
            indicators.update(macd_data)
            
            # ADX
            adx_data = await self.calculate_adx(df)
            indicators.update(adx_data)
            
            # Parabolic SAR
            sar_data = await self.calculate_parabolic_sar(df)
            indicators.update(sar_data)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating trend indicators: {e}")
            return {}
    
    @log_function_call
    async def calculate_sma(self, df: pd.DataFrame, periods: List[int]) -> Dict[str, float]:
        """Calcular Simple Moving Average"""
        try:
            sma_data = {}
            
            for period in periods:
                if len(df) >= period:
                    sma_values = df['close'].rolling(window=period).mean()
                    sma_data[f'sma_{period}'] = float(sma_values.iloc[-1])
                else:
                    sma_data[f'sma_{period}'] = float(df['close'].iloc[-1])
            
            return sma_data
            
        except Exception as e:
            self.logger.error(f"Error calculating SMA: {e}")
            return {}
    
    @log_function_call
    async def calculate_ema(self, df: pd.DataFrame, periods: List[int]) -> Dict[str, float]:
        """Calcular Exponential Moving Average"""
        try:
            ema_data = {}
            
            for period in periods:
                if len(df) >= period:
                    ema_values = df['close'].ewm(span=period).mean()
                    ema_data[f'ema_{period}'] = float(ema_values.iloc[-1])
                else:
                    ema_data[f'ema_{period}'] = float(df['close'].iloc[-1])
            
            return ema_data
            
        except Exception as e:
            self.logger.error(f"Error calculating EMA: {e}")
            return {}
    
    @log_function_call
    async def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, float]:
        """Calcular MACD"""
        try:
            if len(df) < slow:
                return {
                    'macd_line': 0.0,
                    'macd_signal': 0.0,
                    'macd_histogram': 0.0
                }
            
            # Calcular EMAs
            ema_fast = df['close'].ewm(span=fast).mean()
            ema_slow = df['close'].ewm(span=slow).mean()
            
            # MACD Line
            macd_line = ema_fast - ema_slow
            
            # Signal Line
            macd_signal = macd_line.ewm(span=signal).mean()
            
            # Histogram
            macd_histogram = macd_line - macd_signal
            
            return {
                'macd_line': float(macd_line.iloc[-1]),
                'macd_signal': float(macd_signal.iloc[-1]),
                'macd_histogram': float(macd_histogram.iloc[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return {
                'macd_line': 0.0,
                'macd_signal': 0.0,
                'macd_histogram': 0.0
            }
    
    @log_function_call
    async def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
        """Calcular ADX (Average Directional Index)"""
        try:
            if len(df) < period + 1:
                return {
                    'adx': 0.0,
                    'di_plus': 0.0,
                    'di_minus': 0.0,
                    'trend_direction_raw': 0.0
                }
            
            # Calcular True Range
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift(1))
            low_close = np.abs(df['low'] - df['close'].shift(1))
            
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            
            # Calcular Directional Movement
            high_diff = df['high'].diff()
            low_diff = -df['low'].diff()
            
            dm_plus = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
            dm_minus = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
            
            # Convertir a Series de pandas para usar ewm
            dm_plus_series = pd.Series(dm_plus, index=df.index)
            dm_minus_series = pd.Series(dm_minus, index=df.index)
            true_range_series = pd.Series(true_range, index=df.index)
            
            # Suavizar con EMA
            atr = true_range_series.ewm(span=period).mean()
            di_plus = 100 * (dm_plus_series.ewm(span=period).mean() / atr)
            di_minus = 100 * (dm_minus_series.ewm(span=period).mean() / atr)
            
            # Calcular ADX
            dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.ewm(span=period).mean()
            
            # Dirección de tendencia
            trend_direction = di_plus.iloc[-1] - di_minus.iloc[-1]
            
            return {
                'adx': float(adx.iloc[-1]),
                'di_plus': float(di_plus.iloc[-1]),
                'di_minus': float(di_minus.iloc[-1]),
                'trend_direction_raw': float(trend_direction)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ADX: {e}")
            return {
                'adx': 0.0,
                'di_plus': 0.0,
                'di_minus': 0.0,
                'trend_direction_raw': 0.0
            }
    
    @log_function_call
    async def calculate_parabolic_sar(self, df: pd.DataFrame, step: float = 0.02, maximum: float = 0.2) -> Dict[str, float]:
        """Calcular Parabolic SAR"""
        try:
            if len(df) < 2:
                return {
                    'parabolic_sar': float(df['close'].iloc[-1]),
                    'sar_direction': 0.0
                }
            
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Inicializar arrays
            sar = np.zeros(len(df))
            trend = np.zeros(len(df))
            af = np.zeros(len(df))
            ep = np.zeros(len(df))
            
            # Valores iniciales
            sar[0] = low[0]
            trend[0] = 1
            af[0] = step
            ep[0] = high[0]
            
            for i in range(1, len(df)):
                # Calcular SAR
                sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                
                # Verificar si hay reversión
                if trend[i-1] == 1:  # Tendencia alcista
                    if low[i] <= sar[i]:
                        # Reversión a bajista
                        trend[i] = -1
                        sar[i] = ep[i-1]
                        af[i] = step
                        ep[i] = low[i]
                    else:
                        # Continuar alcista
                        trend[i] = 1
                        if high[i] > ep[i-1]:
                            ep[i] = high[i]
                            af[i] = min(af[i-1] + step, maximum)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]
                else:  # Tendencia bajista
                    if high[i] >= sar[i]:
                        # Reversión a alcista
                        trend[i] = 1
                        sar[i] = ep[i-1]
                        af[i] = step
                        ep[i] = high[i]
                    else:
                        # Continuar bajista
                        trend[i] = -1
                        if low[i] < ep[i-1]:
                            ep[i] = low[i]
                            af[i] = min(af[i-1] + step, maximum)
                        else:
                            ep[i] = ep[i-1]
                            af[i] = af[i-1]
            
            return {
                'parabolic_sar': float(sar[-1]),
                'sar_direction': float(trend[-1])
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Parabolic SAR: {e}")
            return {
                'parabolic_sar': float(df['close'].iloc[-1]),
                'sar_direction': 0.0
            }
    
    @log_function_call
    async def calculate_ichimoku(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular Ichimoku Cloud"""
        try:
            if len(df) < 52:
                return {}
            
            # Parámetros Ichimoku
            tenkan_period = 9
            kijun_period = 26
            senkou_span_b_period = 52
            
            # Tenkan-sen (Línea de Conversión)
            tenkan_high = df['high'].rolling(window=tenkan_period).max()
            tenkan_low = df['low'].rolling(window=tenkan_period).min()
            tenkan_sen = (tenkan_high + tenkan_low) / 2
            
            # Kijun-sen (Línea Base)
            kijun_high = df['high'].rolling(window=kijun_period).max()
            kijun_low = df['low'].rolling(window=kijun_period).min()
            kijun_sen = (kijun_high + kijun_low) / 2
            
            # Senkou Span A (Línea de Avance A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
            
            # Senkou Span B (Línea de Avance B)
            senkou_span_b_high = df['high'].rolling(window=senkou_span_b_period).max()
            senkou_span_b_low = df['low'].rolling(window=senkou_span_b_period).min()
            senkou_span_b = ((senkou_span_b_high + senkou_span_b_low) / 2).shift(kijun_period)
            
            # Chikou Span (Línea de Retraso)
            chikou_span = df['close'].shift(-kijun_period)
            
            # Señal Ichimoku
            current_price = df['close'].iloc[-1]
            current_tenkan = tenkan_sen.iloc[-1]
            current_kijun = kijun_sen.iloc[-1]
            current_senkou_a = senkou_span_a.iloc[-1]
            current_senkou_b = senkou_span_b.iloc[-1]
            
            if current_price > current_tenkan and current_price > current_kijun:
                ichimoku_signal = 'BULLISH'
            elif current_price < current_tenkan and current_price < current_kijun:
                ichimoku_signal = 'BEARISH'
            else:
                ichimoku_signal = 'NEUTRAL'
            
            return {
                'ichimoku_tenkan': float(current_tenkan),
                'ichimoku_kijun': float(current_kijun),
                'ichimoku_senkou_a': float(current_senkou_a),
                'ichimoku_senkou_b': float(current_senkou_b),
                'ichimoku_chikou': float(chikou_span.iloc[-1]),
                'ichimoku_signal': ichimoku_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Ichimoku: {e}")
            return {
                'ichimoku_tenkan': 0.0,
                'ichimoku_kijun': 0.0,
                'ichimoku_senkou_a': 0.0,
                'ichimoku_senkou_b': 0.0,
                'ichimoku_chikou': 0.0,
                'ichimoku_signal': 'NEUTRAL'
            }