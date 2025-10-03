#!/usr/bin/env python3
"""
Analizador Técnico Principal - CryptoPulse Pro
Combina múltiples indicadores técnicos para análisis completo
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger, log_execution_time, log_function_call
from config.settings import TradingConfig, TimeFrame
from indicators.trend_indicators import TrendIndicators
from indicators.momentum_indicators import MomentumIndicators
from indicators.volume_indicators import VolumeIndicators
from indicators.volatility_indicators import VolatilityIndicators
from indicators.custom_indicators import CustomIndicators

class TrendDirection(str, Enum):
    """Direcciones de tendencia"""
    UP = "UP"
    DOWN = "DOWN"
    SIDEWAYS = "SIDEWAYS"

class MarketPhase(str, Enum):
    """Fases del mercado"""
    ACCUMULATION = "ACCUMULATION"
    MARKUP = "MARKUP"
    DISTRIBUTION = "DISTRIBUTION"
    MARKDOWN = "MARKDOWN"

@dataclass
class TechnicalAnalysis:
    """Resultado del análisis técnico"""
    symbol: str
    timeframe: TimeFrame
    timestamp: datetime
    
    # Precios
    current_price: float
    high_24h: float
    low_24h: float
    volume_24h: float
    
    # Tendencias
    trend_direction: TrendDirection
    trend_strength: float
    market_phase: MarketPhase
    
    # Indicadores de tendencia
    sma_20: float
    sma_50: float
    sma_200: float
    ema_12: float
    ema_26: float
    macd_line: float
    macd_signal: float
    macd_histogram: float
    adx: float
    parabolic_sar: float
    
    # Indicadores de momentum
    rsi: float
    stochastic_k: float
    stochastic_d: float
    williams_r: float
    cci: float
    roc: float
    
    # Indicadores de volumen
    obv: float
    vwap: float
    volume_profile: Dict[str, float]
    accumulation_distribution: float
    money_flow_index: float
    
    # Indicadores de volatilidad
    bollinger_upper: float
    bollinger_middle: float
    bollinger_lower: float
    atr: float
    keltner_upper: float
    keltner_lower: float
    donchian_upper: float
    donchian_lower: float
    
    # Indicadores personalizados
    multi_timeframe_rsi: Dict[str, float]
    volume_weighted_rsi: float
    sentiment_weighted_macd: float
    news_impact_score: float
    social_sentiment_score: float
    
    # Análisis general
    overall_score: float
    buy_signals: int
    sell_signals: int
    neutral_signals: int
    volatility: float
    support_levels: List[float]
    resistance_levels: List[float]
    
    # Patrones detectados
    patterns: List[str]
    breakout_probability: float
    reversal_probability: float

class TechnicalAnalyzer:
    """
    Analizador técnico principal que combina múltiples indicadores
    - Análisis multi-timeframe
    - Detección de patrones
    - Cálculo de soportes y resistencias
    - Scoring inteligente
    """
    
    def __init__(self, trading_config: TradingConfig, binance_client):
        self.trading_config = trading_config
        self.binance_client = binance_client
        
        # Inicializar indicadores
        self.trend_indicators = TrendIndicators()
        self.momentum_indicators = MomentumIndicators()
        self.volume_indicators = VolumeIndicators()
        self.volatility_indicators = VolatilityIndicators()
        self.custom_indicators = CustomIndicators()
        
        self.logger = get_logger("TechnicalAnalyzer")
        self.logger.info("📈 TechnicalAnalyzer initialized")
    
    async def initialize(self):
        """Inicializar el analizador técnico"""
        self.logger.info("🔧 Initializing technical analyzer...")
        
        # Inicializar indicadores personalizados
        await self.custom_indicators.initialize()
        
        self.logger.info("✅ Technical analyzer initialized")
    
    @log_execution_time
    async def analyze_symbol(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analizar un símbolo con datos de múltiples timeframes"""
        try:
            self.logger.info(f"🔍 Analyzing {symbol}")
            
            # Obtener análisis para cada timeframe
            timeframe_analyses = {}
            
            for timeframe, data in market_data.items():
                if data and len(data) > 0:
                    analysis = await self._analyze_timeframe(symbol, timeframe, data)
                    timeframe_analyses[timeframe] = analysis
            
            # Combinar análisis de múltiples timeframes
            combined_analysis = await self._combine_timeframe_analyses(symbol, timeframe_analyses)
            
            # Calcular score general
            overall_score = await self._calculate_overall_score(combined_analysis)
            combined_analysis['overall_score'] = overall_score
            
            # Detectar patrones
            patterns = await self._detect_patterns(combined_analysis)
            combined_analysis['patterns'] = patterns
            
            # Calcular probabilidades
            breakout_prob = await self._calculate_breakout_probability(combined_analysis)
            reversal_prob = await self._calculate_reversal_probability(combined_analysis)
            combined_analysis['breakout_probability'] = breakout_prob
            combined_analysis['reversal_probability'] = reversal_prob
            
            self.logger.analysis_result({
                'symbol': symbol,
                'overall_score': overall_score,
                'trend_direction': combined_analysis.get('trend_direction'),
                'patterns': patterns
            })
            
            return combined_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}", exception=e)
            return {}
    
    @log_function_call
    async def _analyze_timeframe(self, symbol: str, timeframe: str, data: pd.DataFrame) -> Dict[str, Any]:
        """Analizar un timeframe específico"""
        try:
            # Convertir datos a formato estándar
            df = self._prepare_data(data)
            
            # Calcular indicadores de tendencia
            trend_analysis = await self.trend_indicators.calculate_all(df)
            
            # Calcular indicadores de momentum
            momentum_analysis = await self.momentum_indicators.calculate_all(df)
            
            # Calcular indicadores de volumen
            volume_analysis = await self.volume_indicators.calculate_all(df)
            
            # Calcular indicadores de volatilidad
            volatility_analysis = await self.volatility_indicators.calculate_all(df)
            
            # Calcular indicadores personalizados
            custom_analysis = await self.custom_indicators.calculate_all(df, symbol)
            
            # Combinar todos los análisis
            analysis = {
                'timeframe': timeframe,
                'symbol': symbol,
                'timestamp': datetime.utcnow(),
                'current_price': float(df['close'].iloc[-1]),
                'high_24h': float(df['high'].max()),
                'low_24h': float(df['low'].min()),
                'volume_24h': float(df['volume'].sum()),
                **trend_analysis,
                **momentum_analysis,
                **volume_analysis,
                **volatility_analysis,
                **custom_analysis
            }
            
            # Determinar tendencia general
            analysis['trend_direction'] = self._determine_trend_direction(analysis)
            analysis['trend_strength'] = self._calculate_trend_strength(analysis)
            analysis['market_phase'] = self._determine_market_phase(analysis)
            
            # Calcular soportes y resistencias
            analysis['support_levels'] = await self._calculate_support_levels(df)
            analysis['resistance_levels'] = await self._calculate_resistance_levels(df)
            
            # Calcular volatilidad
            analysis['volatility'] = self._calculate_volatility(df)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing timeframe {timeframe} for {symbol}", exception=e)
            return {}
    
    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preparar datos para análisis técnico"""
        try:
            # Asegurar que tenemos las columnas necesarias
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Available: {data.columns.tolist()}")
            
            # Convertir a float
            for col in required_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Eliminar filas con NaN
            data = data.dropna()
            
            # Ordenar por índice (tiempo)
            data = data.sort_index()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            return pd.DataFrame()
    
    def _determine_trend_direction(self, analysis: Dict[str, Any]) -> TrendDirection:
        """Determinar dirección de la tendencia"""
        try:
            # Ponderar diferentes indicadores de tendencia
            trend_score = 0
            weight_sum = 0
            
            # MACD
            if 'macd_line' in analysis and 'macd_signal' in analysis:
                if analysis['macd_line'] > analysis['macd_signal']:
                    trend_score += 1
                else:
                    trend_score -= 1
                weight_sum += 1
            
            # SMA 20 vs 50
            if 'sma_20' in analysis and 'sma_50' in analysis:
                if analysis['sma_20'] > analysis['sma_50']:
                    trend_score += 1
                else:
                    trend_score -= 1
                weight_sum += 1
            
            # EMA 12 vs 26
            if 'ema_12' in analysis and 'ema_26' in analysis:
                if analysis['ema_12'] > analysis['ema_26']:
                    trend_score += 1
                else:
                    trend_score -= 1
                weight_sum += 1
            
            # ADX
            if 'adx' in analysis and analysis['adx'] > 25:
                if analysis.get('trend_direction_raw', 0) > 0:
                    trend_score += 1
                else:
                    trend_score -= 1
                weight_sum += 1
            
            # Determinar tendencia basada en score
            if weight_sum > 0:
                normalized_score = trend_score / weight_sum
                if normalized_score > 0.3:
                    return TrendDirection.UP
                elif normalized_score < -0.3:
                    return TrendDirection.DOWN
                else:
                    return TrendDirection.SIDEWAYS
            else:
                return TrendDirection.SIDEWAYS
                
        except Exception as e:
            self.logger.error(f"Error determining trend direction: {e}")
            return TrendDirection.SIDEWAYS
    
    def _calculate_trend_strength(self, analysis: Dict[str, Any]) -> float:
        """Calcular fuerza de la tendencia"""
        try:
            strength_indicators = []
            
            # ADX
            if 'adx' in analysis:
                adx_strength = min(analysis['adx'] / 100, 1.0)
                strength_indicators.append(adx_strength)
            
            # Separación entre medias móviles
            if 'sma_20' in analysis and 'sma_50' in analysis:
                sma_separation = abs(analysis['sma_20'] - analysis['sma_50']) / analysis['sma_50']
                strength_indicators.append(min(sma_separation * 10, 1.0))
            
            # MACD histograma
            if 'macd_histogram' in analysis:
                macd_strength = abs(analysis['macd_histogram']) / 1000  # Normalizar
                strength_indicators.append(min(macd_strength, 1.0))
            
            return np.mean(strength_indicators) if strength_indicators else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.0
    
    def _determine_market_phase(self, analysis: Dict[str, Any]) -> MarketPhase:
        """Determinar fase del mercado"""
        try:
            # Análisis de volumen y precio
            volume_trend = analysis.get('volume_trend', 0)
            price_trend = analysis.get('trend_strength', 0)
            volatility = analysis.get('volatility', 0)
            
            # Acumulación: Bajo volumen, precio lateral, baja volatilidad
            if volume_trend < 0.1 and price_trend < 0.3 and volatility < 0.02:
                return MarketPhase.ACCUMULATION
            
            # Markup: Alto volumen, tendencia alcista, volatilidad moderada
            elif volume_trend > 0.5 and price_trend > 0.6 and 0.02 < volatility < 0.05:
                return MarketPhase.MARKUP
            
            # Distribución: Alto volumen, precio lateral, alta volatilidad
            elif volume_trend > 0.5 and price_trend < 0.3 and volatility > 0.05:
                return MarketPhase.DISTRIBUTION
            
            # Markdown: Alto volumen, tendencia bajista, alta volatilidad
            elif volume_trend > 0.5 and price_trend < -0.6 and volatility > 0.05:
                return MarketPhase.MARKDOWN
            
            else:
                return MarketPhase.ACCUMULATION  # Default
                
        except Exception as e:
            self.logger.error(f"Error determining market phase: {e}")
            return MarketPhase.ACCUMULATION
    
    async def _calculate_support_levels(self, df: pd.DataFrame) -> List[float]:
        """Calcular niveles de soporte"""
        try:
            # Usar mínimos locales
            lows = df['low'].rolling(window=5, center=True).min()
            support_levels = []
            
            for i in range(2, len(lows) - 2):
                if (lows.iloc[i] < lows.iloc[i-1] and 
                    lows.iloc[i] < lows.iloc[i+1] and
                    lows.iloc[i] < lows.iloc[i-2] and
                    lows.iloc[i] < lows.iloc[i+2]):
                    support_levels.append(float(lows.iloc[i]))
            
            # Ordenar y devolver los más relevantes
            support_levels = sorted(set(support_levels), reverse=True)[:5]
            return support_levels
            
        except Exception as e:
            self.logger.error(f"Error calculating support levels: {e}")
            return []
    
    async def _calculate_resistance_levels(self, df: pd.DataFrame) -> List[float]:
        """Calcular niveles de resistencia"""
        try:
            # Usar máximos locales
            highs = df['high'].rolling(window=5, center=True).max()
            resistance_levels = []
            
            for i in range(2, len(highs) - 2):
                if (highs.iloc[i] > highs.iloc[i-1] and 
                    highs.iloc[i] > highs.iloc[i+1] and
                    highs.iloc[i] > highs.iloc[i-2] and
                    highs.iloc[i] > highs.iloc[i+2]):
                    resistance_levels.append(float(highs.iloc[i]))
            
            # Ordenar y devolver los más relevantes
            resistance_levels = sorted(set(resistance_levels))[:5]
            return resistance_levels
            
        except Exception as e:
            self.logger.error(f"Error calculating resistance levels: {e}")
            return []
    
    def _calculate_volatility(self, df: pd.DataFrame) -> float:
        """Calcular volatilidad del precio"""
        try:
            # Usar desviación estándar de los retornos
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(24)  # Anualizar para crypto
            return float(volatility)
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return 0.0
    
    async def _combine_timeframe_analyses(self, symbol: str, timeframe_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Combinar análisis de múltiples timeframes"""
        try:
            if not timeframe_analyses:
                return {}
            
            # Usar el timeframe más relevante como base
            primary_timeframe = self._select_primary_timeframe(timeframe_analyses)
            combined = timeframe_analyses[primary_timeframe].copy()
            
            # Agregar información de otros timeframes
            combined['multi_timeframe_analysis'] = {}
            
            for tf, analysis in timeframe_analyses.items():
                if tf != primary_timeframe:
                    combined['multi_timeframe_analysis'][tf] = {
                        'trend_direction': analysis.get('trend_direction'),
                        'trend_strength': analysis.get('trend_strength'),
                        'rsi': analysis.get('rsi'),
                        'macd_line': analysis.get('macd_line'),
                        'volume_trend': analysis.get('volume_trend', 0)
                    }
            
            # Calcular consenso entre timeframes
            consensus = self._calculate_timeframe_consensus(timeframe_analyses)
            combined['timeframe_consensus'] = consensus
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Error combining timeframe analyses: {e}")
            return {}
    
    def _select_primary_timeframe(self, timeframe_analyses: Dict[str, Any]) -> str:
        """Seleccionar timeframe principal para el análisis"""
        # Prioridad: 1h > 4h > 15m > 5m > 1d
        priority_order = ['1h', '4h', '15m', '5m', '1d']
        
        for tf in priority_order:
            if tf in timeframe_analyses:
                return tf
        
        # Si no hay ninguno de los prioritarios, usar el primero disponible
        return list(timeframe_analyses.keys())[0]
    
    def _calculate_timeframe_consensus(self, timeframe_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Calcular consenso entre timeframes"""
        try:
            consensus = {
                'trend_agreement': 0.0,
                'momentum_agreement': 0.0,
                'volume_agreement': 0.0,
                'overall_consensus': 0.0
            }
            
            if len(timeframe_analyses) < 2:
                return consensus
            
            # Analizar tendencias
            trend_directions = [analysis.get('trend_direction') for analysis in timeframe_analyses.values()]
            trend_agreement = len(set(trend_directions)) / len(trend_directions)
            consensus['trend_agreement'] = 1.0 - trend_agreement  # Menos direcciones = más acuerdo
            
            # Analizar momentum (RSI)
            rsi_values = [analysis.get('rsi', 50) for analysis in timeframe_analyses.values()]
            rsi_agreement = 1.0 - (np.std(rsi_values) / 50)  # Menor desviación = más acuerdo
            consensus['momentum_agreement'] = max(0, min(1, rsi_agreement))
            
            # Analizar volumen
            volume_trends = [analysis.get('volume_trend', 0) for analysis in timeframe_analyses.values()]
            volume_agreement = 1.0 - (np.std(volume_trends) / 1.0)  # Menor desviación = más acuerdo
            consensus['volume_agreement'] = max(0, min(1, volume_agreement))
            
            # Consenso general
            consensus['overall_consensus'] = np.mean([
                consensus['trend_agreement'],
                consensus['momentum_agreement'],
                consensus['volume_agreement']
            ])
            
            return consensus
            
        except Exception as e:
            self.logger.error(f"Error calculating timeframe consensus: {e}")
            return consensus
    
    async def _calculate_overall_score(self, analysis: Dict[str, Any]) -> float:
        """Calcular score general del análisis"""
        try:
            scores = []
            weights = []
            
            # Score de tendencia (40%)
            trend_score = self._calculate_trend_score(analysis)
            scores.append(trend_score)
            weights.append(0.4)
            
            # Score de momentum (25%)
            momentum_score = self._calculate_momentum_score(analysis)
            scores.append(momentum_score)
            weights.append(0.25)
            
            # Score de volumen (20%)
            volume_score = self._calculate_volume_score(analysis)
            scores.append(volume_score)
            weights.append(0.2)
            
            # Score de volatilidad (15%)
            volatility_score = self._calculate_volatility_score(analysis)
            scores.append(volatility_score)
            weights.append(0.15)
            
            # Calcular score ponderado
            overall_score = np.average(scores, weights=weights)
            return min(max(overall_score, 0), 100)  # Clamp entre 0 y 100
            
        except Exception as e:
            self.logger.error(f"Error calculating overall score: {e}")
            return 50.0  # Score neutral por defecto
    
    def _calculate_trend_score(self, analysis: Dict[str, Any]) -> float:
        """Calcular score de tendencia"""
        try:
            score = 50.0  # Base neutral
            
            # Dirección de tendencia
            trend_direction = analysis.get('trend_direction')
            if trend_direction == 'UP':
                score += 20
            elif trend_direction == 'DOWN':
                score -= 20
            
            # Fuerza de tendencia
            trend_strength = analysis.get('trend_strength', 0)
            score += trend_strength * 30
            
            # MACD
            macd_line = analysis.get('macd_line', 0)
            macd_signal = analysis.get('macd_signal', 0)
            if macd_line > macd_signal:
                score += 10
            else:
                score -= 10
            
            # ADX
            adx = analysis.get('adx', 0)
            if adx > 25:  # Tendencia fuerte
                score += 15
            elif adx < 20:  # Tendencia débil
                score -= 10
            
            return min(max(score, 0), 100)
            
        except Exception as e:
            self.logger.error(f"Error calculating trend score: {e}")
            return 50.0
    
    def _calculate_momentum_score(self, analysis: Dict[str, Any]) -> float:
        """Calcular score de momentum"""
        try:
            score = 50.0  # Base neutral
            
            # RSI
            rsi = analysis.get('rsi', 50)
            if 30 < rsi < 70:  # Rango normal
                score += 10
            elif rsi < 30:  # Sobreventa
                score += 20
            elif rsi > 70:  # Sobrecompra
                score -= 20
            
            # Stochastic
            stoch_k = analysis.get('stochastic_k', 50)
            if 20 < stoch_k < 80:  # Rango normal
                score += 10
            elif stoch_k < 20:  # Sobreventa
                score += 15
            elif stoch_k > 80:  # Sobrecompra
                score -= 15
            
            # Williams %R
            williams_r = analysis.get('williams_r', -50)
            if -80 < williams_r < -20:  # Rango normal
                score += 10
            elif williams_r > -20:  # Sobreventa
                score += 15
            elif williams_r < -80:  # Sobrecompra
                score -= 15
            
            return min(max(score, 0), 100)
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum score: {e}")
            return 50.0
    
    def _calculate_volume_score(self, analysis: Dict[str, Any]) -> float:
        """Calcular score de volumen"""
        try:
            score = 50.0  # Base neutral
            
            # OBV
            obv_trend = analysis.get('obv_trend', 0)
            if obv_trend > 0:
                score += 15
            else:
                score -= 15
            
            # Money Flow Index
            mfi = analysis.get('money_flow_index', 50)
            if 20 < mfi < 80:  # Rango normal
                score += 10
            elif mfi < 20:  # Sobreventa
                score += 15
            elif mfi > 80:  # Sobrecompra
                score -= 15
            
            # Volume trend
            volume_trend = analysis.get('volume_trend', 0)
            score += volume_trend * 20
            
            return min(max(score, 0), 100)
            
        except Exception as e:
            self.logger.error(f"Error calculating volume score: {e}")
            return 50.0
    
    def _calculate_volatility_score(self, analysis: Dict[str, Any]) -> float:
        """Calcular score de volatilidad"""
        try:
            score = 50.0  # Base neutral
            
            # ATR
            atr = analysis.get('atr', 0)
            current_price = analysis.get('current_price', 1)
            atr_pct = (atr / current_price) * 100
            
            if 1 < atr_pct < 5:  # Volatilidad óptima
                score += 20
            elif atr_pct < 1:  # Muy baja volatilidad
                score -= 10
            elif atr_pct > 10:  # Muy alta volatilidad
                score -= 20
            
            # Bollinger Bands
            bb_position = analysis.get('bollinger_position', 0.5)
            if 0.2 < bb_position < 0.8:  # Dentro de las bandas
                score += 10
            elif bb_position < 0.2 or bb_position > 0.8:  # Fuera de las bandas
                score -= 10
            
            return min(max(score, 0), 100)
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility score: {e}")
            return 50.0
    
    async def _detect_patterns(self, analysis: Dict[str, Any]) -> List[str]:
        """Detectar patrones en el análisis"""
        patterns = []
        
        try:
            # Patrones de tendencia
            if analysis.get('trend_direction') == 'UP' and analysis.get('trend_strength', 0) > 0.7:
                patterns.append('STRONG_UPTREND')
            
            if analysis.get('trend_direction') == 'DOWN' and analysis.get('trend_strength', 0) > 0.7:
                patterns.append('STRONG_DOWNTREND')
            
            # Patrones de momentum
            rsi = analysis.get('rsi', 50)
            if rsi < 30:
                patterns.append('OVERSOLD')
            elif rsi > 70:
                patterns.append('OVERBOUGHT')
            
            # Patrones de volatilidad
            bb_position = analysis.get('bollinger_position', 0.5)
            if bb_position < 0.1:
                patterns.append('BOLLINGER_OVERSOLD')
            elif bb_position > 0.9:
                patterns.append('BOLLINGER_OVERBOUGHT')
            
            # Patrones de volumen
            volume_trend = analysis.get('volume_trend', 0)
            if volume_trend > 0.5:
                patterns.append('HIGH_VOLUME')
            elif volume_trend < -0.5:
                patterns.append('LOW_VOLUME')
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error detecting patterns: {e}")
            return []
    
    async def _calculate_breakout_probability(self, analysis: Dict[str, Any]) -> float:
        """Calcular probabilidad de breakout"""
        try:
            probability = 0.5  # Base neutral
            
            # Volumen alto aumenta probabilidad
            volume_trend = analysis.get('volume_trend', 0)
            probability += volume_trend * 0.3
            
            # Volatilidad alta aumenta probabilidad
            volatility = analysis.get('volatility', 0)
            probability += min(volatility * 10, 0.2)
            
            # Consenso entre timeframes
            consensus = analysis.get('timeframe_consensus', {})
            overall_consensus = consensus.get('overall_consensus', 0.5)
            probability += (overall_consensus - 0.5) * 0.4
            
            return min(max(probability, 0), 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating breakout probability: {e}")
            return 0.5
    
    async def _calculate_reversal_probability(self, analysis: Dict[str, Any]) -> float:
        """Calcular probabilidad de reversión"""
        try:
            probability = 0.3  # Base baja
            
            # RSI extremos aumentan probabilidad
            rsi = analysis.get('rsi', 50)
            if rsi < 30 or rsi > 70:
                probability += 0.3
            
            # Bollinger Bands extremos
            bb_position = analysis.get('bollinger_position', 0.5)
            if bb_position < 0.1 or bb_position > 0.9:
                probability += 0.2
            
            # Divergencias (si están disponibles)
            if analysis.get('divergence_detected', False):
                probability += 0.2
            
            return min(max(probability, 0), 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating reversal probability: {e}")
            return 0.3
    
    async def get_historical_data(self, symbol: str, timeframe: TimeFrame, limit: int = 500) -> Optional[pd.DataFrame]:
        """Obtener datos históricos del símbolo"""
        try:
            return await self.binance_client.get_klines(symbol, timeframe.value, limit)
        except Exception as e:
            self.logger.error(f"Error getting historical data for {symbol}", exception=e)
            return None
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtener precio actual del símbolo"""
        try:
            return await self.binance_client.get_current_price(symbol)
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}", exception=e)
            return None
    
    async def get_atr(self, symbol: str, timeframe: TimeFrame, period: int = 14) -> Optional[List[float]]:
        """Obtener ATR del símbolo"""
        try:
            data = await self.get_historical_data(symbol, timeframe, period + 20)
            if data is not None and len(data) > period:
                atr_values = await self.volatility_indicators.calculate_atr(data, period)
                return atr_values
            return None
        except Exception as e:
            self.logger.error(f"Error getting ATR for {symbol}", exception=e)
            return None
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del analizador técnico"""
        try:
            # Verificar que todos los indicadores estén funcionando
            indicators_healthy = all([
                self.trend_indicators is not None,
                self.momentum_indicators is not None,
                self.volume_indicators is not None,
                self.volatility_indicators is not None,
                self.custom_indicators is not None
            ])
            
            return {
                'healthy': indicators_healthy,
                'message': 'Technical analyzer healthy' if indicators_healthy else 'Some indicators not initialized',
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f'Health check failed: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }