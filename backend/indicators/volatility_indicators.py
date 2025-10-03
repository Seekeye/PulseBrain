#!/usr/bin/env python3
"""
Indicadores de Volatilidad - CryptoPulse Pro
Bollinger Bands, ATR, Keltner Channels, Donchian Channels
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from utils.logger import get_logger, log_function_call

class VolatilityIndicators:
    """
    Calculadora de indicadores de volatilidad
    - Bollinger Bands
    - ATR (Average True Range)
    - Keltner Channels
    - Donchian Channels
    - Standard Deviation
    """
    
    def __init__(self):
        self.logger = get_logger("VolatilityIndicators")
        self.logger.info("📈 VolatilityIndicators initialized")
    
    @log_function_call
    async def calculate_all(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular todos los indicadores de volatilidad"""
        try:
            indicators = {}
            
            # Bollinger Bands
            try:
                bb_data = await self.calculate_bollinger_bands(df)
                if bb_data:
                    indicators.update(bb_data)
            except Exception as e:
                self.logger.error(f"Error in Bollinger Bands: {e}")
            
            # ATR
            try:
                atr_data = await self.calculate_atr(df)
                if atr_data:
                    indicators.update(atr_data)
            except Exception as e:
                self.logger.error(f"Error in ATR: {e}")
            
            # Keltner Channels
            try:
                keltner_data = await self.calculate_keltner_channels(df)
                if keltner_data:
                    indicators.update(keltner_data)
            except Exception as e:
                self.logger.error(f"Error in Keltner Channels: {e}")
            
            # Donchian Channels
            try:
                donchian_data = await self.calculate_donchian_channels(df)
                if donchian_data:
                    indicators.update(donchian_data)
            except Exception as e:
                self.logger.error(f"Error in Donchian Channels: {e}")
            
            # Standard Deviation
            try:
                std_data = await self.calculate_standard_deviation(df)
                if std_data:
                    indicators.update(std_data)
            except Exception as e:
                self.logger.error(f"Error in Standard Deviation: {e}")
            
            # Volatilidad personalizada
            try:
                custom_volatility = await self.calculate_custom_volatility(df)
                if custom_volatility:
                    indicators.update(custom_volatility)
            except Exception as e:
                self.logger.error(f"Error in custom volatility: {e}")
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility indicators: {e}")
            return {}
    
    @log_function_call
    async def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
        """Calcular Bollinger Bands"""
        try:
            if len(df) < period:
                current_price = df['close'].iloc[-1]
                return {
                    'bollinger_upper': current_price,
                    'bollinger_middle': current_price,
                    'bollinger_lower': current_price,
                    'bollinger_width': 0.0,
                    'bollinger_position': 0.5,
                    'bollinger_signal': 'NEUTRAL'
                }
            
            # Calcular SMA
            sma = df['close'].rolling(window=period).mean()
            
            # Calcular desviación estándar
            std = df['close'].rolling(window=period).std()
            
            # Calcular bandas
            upper_band = sma + (std * std_dev)
            lower_band = sma - (std * std_dev)
            
            # Ancho de las bandas
            bb_width = ((upper_band - lower_band) / sma) * 100
            
            # Posición del precio en las bandas
            current_price = df['close'].iloc[-1]
            current_upper = upper_band.iloc[-1]
            current_lower = lower_band.iloc[-1]
            current_middle = sma.iloc[-1]
            
            if current_upper != current_lower:
                bb_position = (current_price - current_lower) / (current_upper - current_lower)
            else:
                bb_position = 0.5
            
            # Señal Bollinger Bands
            if bb_position > 0.95:
                bb_signal = 'UPPER_BAND_TOUCH'
            elif bb_position < 0.05:
                bb_signal = 'LOWER_BAND_TOUCH'
            elif bb_position > 0.8:
                bb_signal = 'NEAR_UPPER_BAND'
            elif bb_position < 0.2:
                bb_signal = 'NEAR_LOWER_BAND'
            else:
                bb_signal = 'NEUTRAL'
            
            return {
                'bollinger_upper': float(current_upper),
                'bollinger_middle': float(current_middle),
                'bollinger_lower': float(current_lower),
                'bollinger_width': float(bb_width.iloc[-1]),
                'bollinger_position': float(bb_position),
                'bollinger_signal': bb_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            current_price = df['close'].iloc[-1]
            return {
                'bollinger_upper': current_price,
                'bollinger_middle': current_price,
                'bollinger_lower': current_price,
                'bollinger_width': 0.0,
                'bollinger_position': 0.5,
                'bollinger_signal': 'NEUTRAL'
            }
    
    @log_function_call
    async def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
        """Calcular ATR (Average True Range)"""
        try:
            if len(df) < 2:
                return {
                    'atr': 0.0,
                    'atr_percentage': 0.0,
                    'atr_signal': 'NEUTRAL'
                }
            
            # Calcular True Range
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            
            # Convertir a Serie de pandas para usar rolling
            true_range_series = pd.Series(true_range, index=df.index)
            
            # Calcular ATR
            atr = true_range_series.rolling(window=period).mean()
            
            # ATR como porcentaje del precio
            current_price = df['close'].iloc[-1]
            current_atr = atr.iloc[-1]
            atr_percentage = (current_atr / current_price) * 100
            
            # Señal ATR
            if atr_percentage > 5:
                atr_signal = 'HIGH_VOLATILITY'
            elif atr_percentage < 1:
                atr_signal = 'LOW_VOLATILITY'
            else:
                atr_signal = 'NORMAL_VOLATILITY'
            
            return {
                'atr': float(current_atr),
                'atr_percentage': float(atr_percentage),
                'atr_signal': atr_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return {
                'atr': 0.0,
                'atr_percentage': 0.0,
                'atr_signal': 'NEUTRAL'
            }
    
    @log_function_call
    async def calculate_keltner_channels(self, df: pd.DataFrame, period: int = 20, multiplier: float = 2.0) -> Dict[str, float]:
        """Calcular Keltner Channels"""
        try:
            if len(df) < period:
                current_price = df['close'].iloc[-1]
                return {
                    'keltner_upper': current_price,
                    'keltner_middle': current_price,
                    'keltner_lower': current_price,
                    'keltner_width': 0.0,
                    'keltner_position': 0.5,
                    'keltner_signal': 'NEUTRAL'
                }
            
            # Calcular EMA
            ema = df['close'].ewm(span=period).mean()
            
            # Calcular ATR directamente aquí
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr1 = high - low
            tr2 = np.abs(high - close.shift(1))
            tr3 = np.abs(low - close.shift(1))
            
            true_range = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = true_range.rolling(window=period).mean()
            
            # Calcular canales
            upper_channel = ema + (atr * multiplier)
            lower_channel = ema - (atr * multiplier)
            
            # Ancho de los canales
            keltner_width = ((upper_channel - lower_channel) / ema) * 100
            
            # Posición del precio en los canales
            current_price = df['close'].iloc[-1]
            current_upper = upper_channel.iloc[-1]
            current_lower = lower_channel.iloc[-1]
            current_middle = ema.iloc[-1]
            
            if current_upper != current_lower:
                keltner_position = (current_price - current_lower) / (current_upper - current_lower)
            else:
                keltner_position = 0.5
            
            # Señal Keltner
            if keltner_position > 0.95:
                keltner_signal = 'UPPER_CHANNEL_TOUCH'
            elif keltner_position < 0.05:
                keltner_signal = 'LOWER_CHANNEL_TOUCH'
            elif keltner_position > 0.8:
                keltner_signal = 'NEAR_UPPER_CHANNEL'
            elif keltner_position < 0.2:
                keltner_signal = 'NEAR_LOWER_CHANNEL'
            else:
                keltner_signal = 'NEUTRAL'
            
            return {
                'keltner_upper': float(current_upper),
                'keltner_middle': float(current_middle),
                'keltner_lower': float(current_lower),
                'keltner_width': float(keltner_width.iloc[-1]),
                'keltner_position': float(keltner_position),
                'keltner_signal': keltner_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Keltner Channels: {e}")
            current_price = df['close'].iloc[-1]
            return {
                'keltner_upper': current_price,
                'keltner_middle': current_price,
                'keltner_lower': current_price,
                'keltner_width': 0.0,
                'keltner_position': 0.5,
                'keltner_signal': 'NEUTRAL'
            }
    @log_function_call
    async def calculate_donchian_channels(self, df: pd.DataFrame, period: int = 20) -> Dict[str, float]:
        """Calcular Donchian Channels"""
        try:
            if len(df) < period:
                current_price = df['close'].iloc[-1]
                return {
                    'donchian_upper': current_price,
                    'donchian_lower': current_price,
                    'donchian_middle': current_price,
                    'donchian_width': 0.0,
                    'donchian_position': 0.5,
                    'donchian_signal': 'NEUTRAL'
                }
            
            # Calcular máximos y mínimos
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            
            # Calcular canal medio
            middle_channel = (highest_high + lowest_low) / 2
            
            # Ancho del canal
            donchian_width = ((highest_high - lowest_low) / middle_channel) * 100
            
            # Posición del precio en el canal
            current_price = df['close'].iloc[-1]
            current_upper = highest_high.iloc[-1]
            current_lower = lowest_low.iloc[-1]
            current_middle = middle_channel.iloc[-1]
            
            if current_upper != current_lower:
                donchian_position = (current_price - current_lower) / (current_upper - current_lower)
            else:
                donchian_position = 0.5
            
            # Señal Donchian
            if donchian_position > 0.95:
                donchian_signal = 'UPPER_BREAKOUT'
            elif donchian_position < 0.05:
                donchian_signal = 'LOWER_BREAKOUT'
            elif donchian_position > 0.8:
                donchian_signal = 'NEAR_UPPER'
            elif donchian_position < 0.2:
                donchian_signal = 'NEAR_LOWER'
            else:
                donchian_signal = 'NEUTRAL'
            
            return {
                'donchian_upper': float(current_upper),
                'donchian_lower': float(current_lower),
                'donchian_middle': float(current_middle),
                'donchian_width': float(donchian_width.iloc[-1]),
                'donchian_position': float(donchian_position),
                'donchian_signal': donchian_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Donchian Channels: {e}")
            current_price = df['close'].iloc[-1]
            return {
                'donchian_upper': current_price,
                'donchian_lower': current_price,
                'donchian_middle': current_price,
                'donchian_width': 0.0,
                'donchian_position': 0.5,
                'donchian_signal': 'NEUTRAL'
            }
    
    @log_function_call
    async def calculate_standard_deviation(self, df: pd.DataFrame, period: int = 20) -> Dict[str, float]:
        """Calcular Standard Deviation"""
        try:
            if len(df) < period:
                return {
                    'standard_deviation': 0.0,
                    'std_percentage': 0.0,
                    'std_signal': 'NEUTRAL'
                }
            
            # Calcular desviación estándar
            std = df['close'].rolling(window=period).std()
            
            # Desviación estándar como porcentaje del precio
            current_price = df['close'].iloc[-1]
            current_std = std.iloc[-1]
            std_percentage = (current_std / current_price) * 100
            
            # Señal Standard Deviation
            if std_percentage > 3:
                std_signal = 'HIGH_VOLATILITY'
            elif std_percentage < 1:
                std_signal = 'LOW_VOLATILITY'
            else:
                std_signal = 'NORMAL_VOLATILITY'
            
            return {
                'standard_deviation': float(current_std),
                'std_percentage': float(std_percentage),
                'std_signal': std_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Standard Deviation: {e}")
            return {
                'standard_deviation': 0.0,
                'std_percentage': 0.0,
                'std_signal': 'NEUTRAL'
            }
    
    @log_function_call
    async def calculate_custom_volatility(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular indicadores de volatilidad personalizados"""
        try:
            custom_indicators = {}
            
            # Volatilidad histórica
            if len(df) >= 20:
                returns = df['close'].pct_change().dropna()
                historical_volatility = returns.std() * np.sqrt(252)  # Anualizada
                custom_indicators['historical_volatility'] = float(historical_volatility * 100)
            
            # Volatilidad de Parkinson
            if len(df) >= 10:
                high_low_ratio = (df['high'] / df['low']) ** 2
                parkinson_vol = np.sqrt(np.mean(high_low_ratio) / (4 * np.log(2)))
                custom_indicators['parkinson_volatility'] = float(parkinson_vol * 100)
            
            # Volatilidad de Garman-Klass
            if len(df) >= 10:
                log_hl = (np.log(df['high'] / df['low'])) ** 2
                log_co = (np.log(df['close'] / df['open'])) ** 2
                gk_vol = np.sqrt(np.mean(0.5 * log_hl - (2 * np.log(2) - 1) * log_co))
                custom_indicators['garman_klass_volatility'] = float(gk_vol * 100)
            
            # Volatilidad implícita (simplificada)
            if len(df) >= 20:
                # Usar el ancho de Bollinger Bands como proxy
                bb_data = await self.calculate_bollinger_bands(df, 20, 2.0)
                custom_indicators['implied_volatility_proxy'] = bb_data['bollinger_width']
            
            return custom_indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating custom volatility indicators: {e}")
            return {}
    