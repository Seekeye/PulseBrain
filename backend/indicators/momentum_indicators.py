#!/usr/bin/env python3
"""
Indicadores de Momentum - CryptoPulse Pro
RSI, Stochastic, Williams %R, CCI, ROC
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from utils.logger import get_logger, log_function_call

class MomentumIndicators:
    """
    Calculadora de indicadores de momentum
    - RSI (Relative Strength Index)
    - Stochastic Oscillator
    - Williams %R
    - CCI (Commodity Channel Index)
    - ROC (Rate of Change)
    """
    
    def __init__(self):
        self.logger = get_logger("MomentumIndicators")
        self.logger.info("📊 MomentumIndicators initialized")
    
    @log_function_call
    async def calculate_all(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular todos los indicadores de momentum"""
        try:
            indicators = {}
            
            # RSI
            rsi_data = await self.calculate_rsi(df)
            indicators.update(rsi_data)
            
            # Stochastic
            stoch_data = await self.calculate_stochastic(df)
            indicators.update(stoch_data)
            
            # Williams %R
            williams_data = await self.calculate_williams_r(df)
            indicators.update(williams_data)
            
            # CCI
            cci_data = await self.calculate_cci(df)
            indicators.update(cci_data)
            
            # ROC
            roc_data = await self.calculate_roc(df)
            indicators.update(roc_data)
            
            # Momentum personalizado
            custom_momentum = await self.calculate_custom_momentum(df)
            indicators.update(custom_momentum)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum indicators: {e}")
            return {}
    
    @log_function_call
    async def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
        """Calcular RSI (Relative Strength Index)"""
        try:
            if len(df) < period + 1:
                return {
                    'rsi': 50.0,
                    'rsi_sma': 50.0,
                    'rsi_ema': 50.0,
                    'rsi_signal': 'NEUTRAL'
                }
            
            # Calcular cambios de precio
            delta = df['close'].diff()
            
            # Separar ganancias y pérdidas
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calcular medias móviles exponenciales
            avg_gains = gains.ewm(span=period).mean()
            avg_losses = losses.ewm(span=period).mean()
            
            # Calcular RS y RSI
            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))
            
            # RSI con SMA (más suave)
            rsi_sma = rsi.rolling(window=period).mean()
            
            # RSI con EMA (más reactivo)
            rsi_ema = rsi.ewm(span=period).mean()
            
            # Señal RSI
            current_rsi = rsi.iloc[-1]
            if current_rsi > 70:
                rsi_signal = 'OVERBOUGHT'
            elif current_rsi < 30:
                rsi_signal = 'OVERSOLD'
            else:
                rsi_signal = 'NEUTRAL'
            
            return {
                'rsi': float(current_rsi),
                'rsi_sma': float(rsi_sma.iloc[-1]),
                'rsi_ema': float(rsi_ema.iloc[-1]),
                'rsi_signal': rsi_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return {
                'rsi': 50.0,
                'rsi_sma': 50.0,
                'rsi_ema': 50.0,
                'rsi_signal': 'NEUTRAL'
            }
    
    @log_function_call
    async def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, float]:
        """Calcular Stochastic Oscillator"""
        try:
            if len(df) < k_period:
                return {
                    'stochastic_k': 50.0,
                    'stochastic_d': 50.0,
                    'stochastic_signal': 'NEUTRAL'
                }
            
            # Calcular %K
            lowest_low = df['low'].rolling(window=k_period).min()
            highest_high = df['high'].rolling(window=k_period).max()
            
            k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
            
            # Calcular %D (SMA de %K)
            d_percent = k_percent.rolling(window=d_period).mean()
            
            # Señal Stochastic
            current_k = k_percent.iloc[-1]
            current_d = d_percent.iloc[-1]
            
            if current_k > 80 and current_d > 80:
                stoch_signal = 'OVERBOUGHT'
            elif current_k < 20 and current_d < 20:
                stoch_signal = 'OVERSOLD'
            elif current_k > current_d:
                stoch_signal = 'BULLISH'
            elif current_k < current_d:
                stoch_signal = 'BEARISH'
            else:
                stoch_signal = 'NEUTRAL'
            
            return {
                'stochastic_k': float(current_k),
                'stochastic_d': float(current_d),
                'stochastic_signal': stoch_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Stochastic: {e}")
            return {
                'stochastic_k': 50.0,
                'stochastic_d': 50.0,
                'stochastic_signal': 'NEUTRAL'
            }
    
    @log_function_call
    async def calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
        """Calcular Williams %R"""
        try:
            if len(df) < period:
                return {
                    'williams_r': -50.0,
                    'williams_r_signal': 'NEUTRAL'
                }
            
            # Calcular Williams %R
            highest_high = df['high'].rolling(window=period).max()
            lowest_low = df['low'].rolling(window=period).min()
            
            williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
            
            # Señal Williams %R
            current_wr = williams_r.iloc[-1]
            if current_wr > -20:
                wr_signal = 'OVERBOUGHT'
            elif current_wr < -80:
                wr_signal = 'OVERSOLD'
            else:
                wr_signal = 'NEUTRAL'
            
            return {
                'williams_r': float(current_wr),
                'williams_r_signal': wr_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Williams %R: {e}")
            return {
                'williams_r': -50.0,
                'williams_r_signal': 'NEUTRAL'
            }
    
    @log_function_call
    async def calculate_cci(self, df: pd.DataFrame, period: int = 20) -> Dict[str, float]:
        """Calcular CCI (Commodity Channel Index)"""
        try:
            if len(df) < period:
                return {
                    'cci': 0.0,
                    'cci_signal': 'NEUTRAL'
                }
            
            # Calcular Typical Price
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Calcular SMA de Typical Price
            sma_tp = typical_price.rolling(window=period).mean()
            
            # Calcular Mean Deviation
            mean_deviation = typical_price.rolling(window=period).apply(
                lambda x: np.mean(np.abs(x - x.mean()))
            )
            
            # Calcular CCI
            cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
            
            # Señal CCI
            current_cci = cci.iloc[-1]
            if current_cci > 100:
                cci_signal = 'OVERBOUGHT'
            elif current_cci < -100:
                cci_signal = 'OVERSOLD'
            else:
                cci_signal = 'NEUTRAL'
            
            return {
                'cci': float(current_cci),
                'cci_signal': cci_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating CCI: {e}")
            return {
                'cci': 0.0,
                'cci_signal': 'NEUTRAL'
            }
    
    @log_function_call
    async def calculate_roc(self, df: pd.DataFrame, period: int = 12) -> Dict[str, float]:
        """Calcular ROC (Rate of Change)"""
        try:
            if len(df) < period + 1:
                return {
                    'roc': 0.0,
                    'roc_signal': 'NEUTRAL'
                }
            
            # Calcular ROC
            roc = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
            
            # Señal ROC
            current_roc = roc.iloc[-1]
            if current_roc > 5:
                roc_signal = 'BULLISH'
            elif current_roc < -5:
                roc_signal = 'BEARISH'
            else:
                roc_signal = 'NEUTRAL'
            
            return {
                'roc': float(current_roc),
                'roc_signal': roc_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ROC: {e}")
            return {
                'roc': 0.0,
                'roc_signal': 'NEUTRAL'
            }
    
    @log_function_call
    async def calculate_custom_momentum(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular indicadores de momentum personalizados"""
        try:
            custom_indicators = {}
            
            # Momentum Multi-timeframe
            if len(df) >= 50:
                # RSI de 5 períodos (más reactivo)
                rsi_5 = await self.calculate_rsi(df, 5)
                custom_indicators['rsi_5'] = rsi_5['rsi']
                
                # RSI de 21 períodos (más suave)
                rsi_21 = await self.calculate_rsi(df, 21)
                custom_indicators['rsi_21'] = rsi_21['rsi']
                
                # Momentum Divergence
                momentum_divergence = await self._calculate_momentum_divergence(df)
                custom_indicators.update(momentum_divergence)
            
            # Momentum Strength
            momentum_strength = await self._calculate_momentum_strength(df)
            custom_indicators.update(momentum_strength)
            
            return custom_indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating custom momentum: {e}")
            return {}
    
    @log_function_call
    async def _calculate_momentum_divergence(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular divergencias de momentum"""
        try:
            if len(df) < 20:
                return {
                    'momentum_divergence': 0.0,
                    'divergence_signal': 'NEUTRAL'
                }
            
            # Calcular RSI para divergencia
            rsi = await self.calculate_rsi(df, 14)
            rsi_values = rsi['rsi']
            
            # Encontrar máximos y mínimos en precio y RSI
            price_highs = df['high'].rolling(window=5, center=True).max()
            price_lows = df['low'].rolling(window=5, center=True).min()
            
            # Detectar divergencias (simplificado)
            divergence_score = 0.0
            
            # Buscar divergencias alcistas (precio baja, RSI sube)
            for i in range(10, len(df) - 5):
                if (price_lows.iloc[i] < price_lows.iloc[i-5] and 
                    rsi_values > 50):  # RSI mejorando
                    divergence_score += 0.1
            
            # Buscar divergencias bajistas (precio sube, RSI baja)
            for i in range(10, len(df) - 5):
                if (price_highs.iloc[i] > price_highs.iloc[i-5] and 
                    rsi_values < 50):  # RSI empeorando
                    divergence_score -= 0.1
            
            # Normalizar score
            divergence_score = max(-1, min(1, divergence_score))
            
            if divergence_score > 0.3:
                div_signal = 'BULLISH_DIVERGENCE'
            elif divergence_score < -0.3:
                div_signal = 'BEARISH_DIVERGENCE'
            else:
                div_signal = 'NO_DIVERGENCE'
            
            return {
                'momentum_divergence': divergence_score,
                'divergence_signal': div_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum divergence: {e}")
            return {
                'momentum_divergence': 0.0,
                'divergence_signal': 'NEUTRAL'
            }
    
    @log_function_call
    async def _calculate_momentum_strength(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular fuerza del momentum"""
        try:
            if len(df) < 20:
                return {
                    'momentum_strength': 0.0,
                    'momentum_trend': 'NEUTRAL'
                }
            
            # Calcular múltiples indicadores de momentum
            rsi = await self.calculate_rsi(df, 14)
            stoch = await self.calculate_stochastic(df, 14, 3)
            williams = await self.calculate_williams_r(df, 14)
            cci = await self.calculate_cci(df, 20)
            
            # Calcular score de momentum
            momentum_score = 0.0
            indicators_count = 0
            
            # RSI contribution
            rsi_val = rsi['rsi']
            if rsi_val > 70:
                momentum_score += 1
            elif rsi_val < 30:
                momentum_score -= 1
            indicators_count += 1
            
            # Stochastic contribution
            stoch_k = stoch['stochastic_k']
            if stoch_k > 80:
                momentum_score += 1
            elif stoch_k < 20:
                momentum_score -= 1
            indicators_count += 1
            
            # Williams %R contribution
            wr_val = williams['williams_r']
            if wr_val > -20:
                momentum_score += 1
            elif wr_val < -80:
                momentum_score -= 1
            indicators_count += 1
            
            # CCI contribution
            cci_val = cci['cci']
            if cci_val > 100:
                momentum_score += 1
            elif cci_val < -100:
                momentum_score -= 1
            indicators_count += 1
            
            # Normalizar score
            momentum_strength = momentum_score / indicators_count if indicators_count > 0 else 0
            
            # Determinar tendencia
            if momentum_strength > 0.5:
                momentum_trend = 'STRONG_BULLISH'
            elif momentum_strength > 0.2:
                momentum_trend = 'BULLISH'
            elif momentum_strength < -0.5:
                momentum_trend = 'STRONG_BEARISH'
            elif momentum_strength < -0.2:
                momentum_trend = 'BEARISH'
            else:
                momentum_trend = 'NEUTRAL'
            
            return {
                'momentum_strength': momentum_strength,
                'momentum_trend': momentum_trend
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum strength: {e}")
            return {
                'momentum_strength': 0.0,
                'momentum_trend': 'NEUTRAL'
            }
    
    @log_function_call
    async def calculate_momentum_consensus(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular consenso de momentum entre indicadores"""
        try:
            # Calcular todos los indicadores
            rsi = await self.calculate_rsi(df, 14)
            stoch = await self.calculate_stochastic(df, 14, 3)
            williams = await self.calculate_williams_r(df, 14)
            cci = await self.calculate_cci(df, 20)
            roc = await self.calculate_roc(df, 12)
            
            # Contar señales alcistas y bajistas
            bullish_signals = 0
            bearish_signals = 0
            total_signals = 0
            
            # RSI
            if rsi['rsi'] < 30:
                bullish_signals += 1
            elif rsi['rsi'] > 70:
                bearish_signals += 1
            total_signals += 1
            
            # Stochastic
            if stoch['stochastic_k'] < 20:
                bullish_signals += 1
            elif stoch['stochastic_k'] > 80:
                bearish_signals += 1
            total_signals += 1
            
            # Williams %R
            if williams['williams_r'] < -80:
                bullish_signals += 1
            elif williams['williams_r'] > -20:
                bearish_signals += 1
            total_signals += 1
            
            # CCI
            if cci['cci'] < -100:
                bullish_signals += 1
            elif cci['cci'] > 100:
                bearish_signals += 1
            total_signals += 1
            
            # ROC
            if roc['roc'] > 5:
                bullish_signals += 1
            elif roc['roc'] < -5:
                bearish_signals += 1
            total_signals += 1
            
            # Calcular consenso
            if total_signals > 0:
                bullish_consensus = bullish_signals / total_signals
                bearish_consensus = bearish_signals / total_signals
                net_consensus = bullish_consensus - bearish_consensus
            else:
                bullish_consensus = 0.5
                bearish_consensus = 0.5
                net_consensus = 0.0
            
            return {
                'bullish_consensus': bullish_consensus,
                'bearish_consensus': bearish_consensus,
                'net_consensus': net_consensus,
                'momentum_consensus': 'BULLISH' if net_consensus > 0.2 else 'BEARISH' if net_consensus < -0.2 else 'NEUTRAL'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum consensus: {e}")
            return {
                'bullish_consensus': 0.5,
                'bearish_consensus': 0.5,
                'net_consensus': 0.0,
                'momentum_consensus': 'NEUTRAL'
            }
