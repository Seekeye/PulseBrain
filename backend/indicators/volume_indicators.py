#!/usr/bin/env python3
"""
Indicadores de Volumen - CryptoPulse Pro
OBV, VWAP, Volume Profile, A/D, MFI
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from utils.logger import get_logger, log_function_call

class VolumeIndicators:
    """
    Calculadora de indicadores de volumen
    - OBV (On-Balance Volume)
    - VWAP (Volume Weighted Average Price)
    - Volume Profile
    - Accumulation/Distribution
    - Money Flow Index
    """
    
    def __init__(self):
        self.logger = get_logger("VolumeIndicators")
        self.logger.info("📊 VolumeIndicators initialized")
    
    @log_function_call
    async def calculate_all(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular todos los indicadores de volumen"""
        try:
            indicators = {}
            
            # OBV
            obv_data = await self.calculate_obv(df)
            indicators.update(obv_data)
            
            # VWAP
            vwap_data = await self.calculate_vwap(df)
            indicators.update(vwap_data)
            
            # Volume Profile
            volume_profile_data = await self.calculate_volume_profile(df)
            indicators.update(volume_profile_data)
            
            # Accumulation/Distribution
            ad_data = await self.calculate_accumulation_distribution(df)
            indicators.update(ad_data)
            
            # Money Flow Index
            mfi_data = await self.calculate_money_flow_index(df)
            indicators.update(mfi_data)
            
            # Volume personalizado
            custom_volume = await self.calculate_custom_volume(df)
            indicators.update(custom_volume)
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating volume indicators: {e}")
            return {}
    
    @log_function_call
    async def calculate_obv(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular OBV (On-Balance Volume)"""
        try:
            if len(df) < 2:
                return {
                    'obv': 0.0,
                    'obv_trend': 0.0,
                    'obv_signal': 'NEUTRAL'
                }
            
            # Calcular OBV
            price_change = df['close'].diff()
            obv = np.zeros(len(df))
            obv[0] = df['volume'].iloc[0]
            
            for i in range(1, len(df)):
                if price_change.iloc[i] > 0:
                    obv[i] = obv[i-1] + df['volume'].iloc[i]
                elif price_change.iloc[i] < 0:
                    obv[i] = obv[i-1] - df['volume'].iloc[i]
                else:
                    obv[i] = obv[i-1]
            
            # Calcular tendencia OBV
            obv_series = pd.Series(obv)
            obv_trend = obv_series.pct_change().iloc[-1] * 100
            
            # Señal OBV
            if obv_trend > 5:
                obv_signal = 'BULLISH'
            elif obv_trend < -5:
                obv_signal = 'BEARISH'
            else:
                obv_signal = 'NEUTRAL'
            
            return {
                'obv': float(obv[-1]),
                'obv_trend': float(obv_trend),
                'obv_signal': obv_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating OBV: {e}")
            return {
                'obv': 0.0,
                'obv_trend': 0.0,
                'obv_signal': 'NEUTRAL'
            }
    
    @log_function_call
    async def calculate_vwap(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular VWAP (Volume Weighted Average Price)"""
        try:
            if len(df) == 0:
                return {
                    'vwap': 0.0,
                    'vwap_deviation': 0.0,
                    'vwap_signal': 'NEUTRAL'
                }
            
            # Calcular Typical Price
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Calcular VWAP
            cumulative_volume = df['volume'].cumsum()
            cumulative_volume_price = (typical_price * df['volume']).cumsum()
            vwap = cumulative_volume_price / cumulative_volume
            
            # Desviación del VWAP
            current_price = df['close'].iloc[-1]
            current_vwap = vwap.iloc[-1]
            vwap_deviation = ((current_price - current_vwap) / current_vwap) * 100
            
            # Señal VWAP
            if vwap_deviation > 2:
                vwap_signal = 'ABOVE_VWAP'
            elif vwap_deviation < -2:
                vwap_signal = 'BELOW_VWAP'
            else:
                vwap_signal = 'NEAR_VWAP'
            
            return {
                'vwap': float(current_vwap),
                'vwap_deviation': float(vwap_deviation),
                'vwap_signal': vwap_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating VWAP: {e}")
            return {
                'vwap': 0.0,
                'vwap_deviation': 0.0,
                'vwap_signal': 'NEUTRAL'
            }
    
    @log_function_call
    async def calculate_volume_profile(self, df: pd.DataFrame, bins: int = 20) -> Dict[str, float]:
        """Calcular Volume Profile"""
        try:
            if len(df) < 10:
                return {
                    'volume_profile_poc': 0.0,
                    'volume_profile_vah': 0.0,
                    'volume_profile_val': 0.0,
                    'volume_profile_signal': 'NEUTRAL'
                }
            
            # Calcular rango de precios
            price_range = df['high'].max() - df['low'].min()
            if price_range == 0:
                return {
                    'volume_profile_poc': float(df['close'].iloc[-1]),
                    'volume_profile_vah': float(df['close'].iloc[-1]),
                    'volume_profile_val': float(df['close'].iloc[-1]),
                    'volume_profile_signal': 'NEUTRAL'
                }
            
            # Crear bins de precio
            price_bins = np.linspace(df['low'].min(), df['high'].max(), bins + 1)
            
            # Calcular volumen por bin
            volume_by_price = np.zeros(bins)
            for i in range(len(df)):
                # Distribuir volumen proporcionalmente entre high y low
                price_range_candle = df['high'].iloc[i] - df['low'].iloc[i]
                if price_range_candle > 0:
                    # Encontrar bins afectados
                    low_bin = np.digitize(df['low'].iloc[i], price_bins) - 1
                    high_bin = np.digitize(df['high'].iloc[i], price_bins) - 1
                    
                    # Distribuir volumen
                    for bin_idx in range(max(0, low_bin), min(bins, high_bin + 1)):
                        bin_start = price_bins[bin_idx]
                        bin_end = price_bins[bin_idx + 1]
                        
                        # Calcular intersección
                        intersection_start = max(bin_start, df['low'].iloc[i])
                        intersection_end = min(bin_end, df['high'].iloc[i])
                        
                        if intersection_end > intersection_start:
                            intersection_ratio = (intersection_end - intersection_start) / price_range_candle
                            volume_by_price[bin_idx] += df['volume'].iloc[i] * intersection_ratio
            
            # Encontrar POC (Point of Control)
            poc_bin = np.argmax(volume_by_price)
            poc_price = (price_bins[poc_bin] + price_bins[poc_bin + 1]) / 2
            
            # Calcular VAH y VAL (simplificado)
            total_volume = np.sum(volume_by_price)
            if total_volume > 0:
                # VAH: 70% del volumen por encima
                cumulative_volume = np.cumsum(volume_by_price[::-1])
                vah_bin = bins - 1 - np.argmax(cumulative_volume >= total_volume * 0.7)
                vah_price = (price_bins[vah_bin] + price_bins[vah_bin + 1]) / 2
                
                # VAL: 70% del volumen por debajo
                cumulative_volume = np.cumsum(volume_by_price)
                val_bin = np.argmax(cumulative_volume >= total_volume * 0.7)
                val_price = (price_bins[val_bin] + price_bins[val_bin + 1]) / 2
            else:
                vah_price = poc_price
                val_price = poc_price
            
            # Señal Volume Profile
            current_price = df['close'].iloc[-1]
            if current_price > vah_price:
                vp_signal = 'ABOVE_VAH'
            elif current_price < val_price:
                vp_signal = 'BELOW_VAL'
            else:
                vp_signal = 'BETWEEN_VAL_VAH'
            
            return {
                'volume_profile_poc': float(poc_price),
                'volume_profile_vah': float(vah_price),
                'volume_profile_val': float(val_price),
                'volume_profile_signal': vp_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating Volume Profile: {e}")
            return {
                'volume_profile_poc': float(df['close'].iloc[-1]),
                'volume_profile_vah': float(df['close'].iloc[-1]),
                'volume_profile_val': float(df['close'].iloc[-1]),
                'volume_profile_signal': 'NEUTRAL'
            }
    
    @log_function_call
    async def calculate_accumulation_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular Accumulation/Distribution Line"""
        try:
            if len(df) < 2:
                return {
                    'accumulation_distribution': 0.0,
                    'ad_trend': 0.0,
                    'ad_signal': 'NEUTRAL'
                }
            
            # Calcular Money Flow Multiplier
            high = df['high']
            low = df['low']
            close = df['close']
            volume = df['volume']
            
            mfm = ((close - low) - (high - close)) / (high - low)
            mfm = mfm.fillna(0)  # Evitar division por cero
            
            # Calcular Money Flow Volume
            mfv = mfm * volume
            
            # Calcular A/D Line
            ad_line = mfv.cumsum()
            
            # Calcular tendencia A/D
            ad_trend = ad_line.pct_change().iloc[-1] * 100
            
            # Señal A/D
            if ad_trend > 1:
                ad_signal = 'ACCUMULATION'
            elif ad_trend < -1:
                ad_signal = 'DISTRIBUTION'
            else:
                ad_signal = 'NEUTRAL'
            
            return {
                'accumulation_distribution': float(ad_line.iloc[-1]),
                'ad_trend': float(ad_trend),
                'ad_signal': ad_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating A/D: {e}")
            return {
                'accumulation_distribution': 0.0,
                'ad_trend': 0.0,
                'ad_signal': 'NEUTRAL'
            }
    
    @log_function_call
    async def calculate_money_flow_index(self, df: pd.DataFrame, period: int = 14) -> Dict[str, float]:
        """Calcular Money Flow Index"""
        try:
            if len(df) < period + 1:
                return {
                    'money_flow_index': 50.0,
                    'mfi_signal': 'NEUTRAL'
                }
            
            # Calcular Typical Price
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Calcular Raw Money Flow
            raw_money_flow = typical_price * df['volume']
            
            # Calcular Positive y Negative Money Flow
            price_change = typical_price.diff()
            
            positive_money_flow = raw_money_flow.where(price_change > 0, 0).rolling(window=period).sum()
            negative_money_flow = raw_money_flow.where(price_change < 0, 0).rolling(window=period).sum()
            
            # Calcular Money Flow Ratio
            money_flow_ratio = positive_money_flow / negative_money_flow
            
            # Calcular MFI
            mfi = 100 - (100 / (1 + money_flow_ratio))
            
            # Señal MFI
            current_mfi = mfi.iloc[-1]
            if current_mfi > 80:
                mfi_signal = 'OVERBOUGHT'
            elif current_mfi < 20:
                mfi_signal = 'OVERSOLD'
            else:
                mfi_signal = 'NEUTRAL'
            
            return {
                'money_flow_index': float(current_mfi),
                'mfi_signal': mfi_signal
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating MFI: {e}")
            return {
                'money_flow_index': 50.0,
                'mfi_signal': 'NEUTRAL'
            }
    
    @log_function_call
    async def calculate_custom_volume(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calcular indicadores de volumen personalizados"""
        try:
            custom_indicators = {}
            
            # Volume Trend
            if len(df) >= 10:
                volume_sma_short = df['volume'].rolling(window=5).mean()
                volume_sma_long = df['volume'].rolling(window=10).mean()
                volume_trend = ((volume_sma_short.iloc[-1] - volume_sma_long.iloc[-1]) / volume_sma_long.iloc[-1]) * 100
                custom_indicators['volume_trend'] = float(volume_trend)
            
            # Volume Price Trend (VPT)
            if len(df) >= 2:
                price_change_pct = df['close'].pct_change()
                vpt = (price_change_pct * df['volume']).cumsum()
                custom_indicators['volume_price_trend'] = float(vpt.iloc[-1])
            
            # Volume Oscillator
            if len(df) >= 20:
                volume_ma_short = df['volume'].rolling(window=5).mean()
                volume_ma_long = df['volume'].rolling(window=20).mean()
                volume_oscillator = ((volume_ma_short.iloc[-1] - volume_ma_long.iloc[-1]) / volume_ma_long.iloc[-1]) * 100
                custom_indicators['volume_oscillator'] = float(volume_oscillator)
            
            # Volume Rate of Change
            if len(df) >= 10:
                volume_roc = ((df['volume'].iloc[-1] - df['volume'].iloc[-10]) / df['volume'].iloc[-10]) * 100
                custom_indicators['volume_roc'] = float(volume_roc)
            
            return custom_indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating custom volume indicators: {e}")
            return {}