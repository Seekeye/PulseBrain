#!/usr/bin/env python3
"""
Superprocesos de IA Coherentes - CryptoPulse Pro
Procesos avanzados de inteligencia artificial integrados con el sistema existente
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
import json
from dataclasses import dataclass

from utils.logger import get_logger

@dataclass
class MarketRegime:
    """Regímenes de mercado identificados por IA"""
    regime_type: str  # BULL, BEAR, SIDEWAYS, VOLATILE
    confidence: float
    duration: int  # minutos
    volatility_level: float
    trend_strength: float
    volume_profile: str  # HIGH, NORMAL, LOW

@dataclass
class SmartSignal:
    """Señal inteligente generada por superprocesos"""
    symbol: str
    signal_type: str  # BUY, SELL, HOLD
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    reasoning: List[str]
    risk_level: str  # LOW, MEDIUM, HIGH
    expected_duration: int  # minutos
    market_regime: MarketRegime

class SuperAIProcesses:
    """Superprocesos de IA coherentes con el sistema existente"""
    
    def __init__(self):
        self.logger = get_logger("SuperAIProcesses")
        self.market_regimes = {}
        self.pattern_memory = {}
        self.correlation_matrix = {}
        self.volatility_clusters = {}
        
    async def analyze_market_regime(self, symbol: str, data: Dict[str, Any]) -> MarketRegime:
        """Analizar régimen de mercado usando IA avanzada"""
        try:
            # Obtener datos de precios
            prices = data.get('prices', [])
            volumes = data.get('volumes', [])
            
            if len(prices) < 20:
                return MarketRegime("SIDEWAYS", 0.5, 60, 0.3, 0.2, "NORMAL")
            
            # Convertir a numpy arrays
            price_array = np.array(prices[-50:])  # Últimos 50 períodos
            volume_array = np.array(volumes[-50:]) if volumes else np.ones(50)
            
            # 1. ANÁLISIS DE VOLATILIDAD INTELIGENTE
            volatility = self._calculate_smart_volatility(price_array)
            
            # 2. ANÁLISIS DE TENDENCIA MULTI-TIMEFRAME
            trend_strength = self._calculate_trend_strength(price_array)
            
            # 3. ANÁLISIS DE VOLUMEN INTELIGENTE
            volume_profile = self._analyze_volume_profile(volume_array)
            
            # 4. DETECCIÓN DE PATRONES DE MERCADO
            pattern_score = self._detect_market_patterns(price_array)
            
            # 5. CLASIFICACIÓN DEL RÉGIMEN
            regime = self._classify_market_regime(
                volatility, trend_strength, volume_profile, pattern_score
            )
            
            self.market_regimes[symbol] = regime
            return regime
            
        except Exception as e:
            self.logger.error(f"Error analyzing market regime for {symbol}: {e}")
            return MarketRegime("SIDEWAYS", 0.3, 60, 0.3, 0.2, "NORMAL")
    
    def _calculate_smart_volatility(self, prices: np.ndarray) -> float:
        """Calcular volatilidad inteligente usando múltiples métodos"""
        try:
            # 1. Volatilidad histórica estándar
            returns = np.diff(np.log(prices))
            hv = np.std(returns) * np.sqrt(24 * 60)  # Anualizada
            
            # 2. Volatilidad de Parkinson (usando high-low)
            # Aproximación usando rangos de precios
            price_ranges = np.abs(np.diff(prices))
            parkinson_vol = np.sqrt(np.mean(price_ranges**2) / (4 * np.log(2)))
            
            # 3. Volatilidad de Garman-Klass
            # Aproximación usando volatilidad de precios
            gk_vol = np.sqrt(np.mean(price_ranges**2))
            
            # 4. Combinar métodos con pesos inteligentes
            combined_vol = 0.4 * hv + 0.3 * parkinson_vol + 0.3 * gk_vol
            
            return min(combined_vol, 2.0)  # Cap en 200%
            
        except Exception as e:
            self.logger.error(f"Error calculating smart volatility: {e}")
            return 0.3
    
    def _calculate_trend_strength(self, prices: np.ndarray) -> float:
        """Calcular fuerza de tendencia usando múltiples indicadores"""
        try:
            # 1. Slope de regresión lineal
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]
            slope_strength = abs(slope) / np.mean(prices)
            
            # 2. ADX aproximado
            high_low = np.abs(np.diff(prices))
            adx_approx = np.mean(high_low) / np.mean(prices)
            
            # 3. Momentum de precios
            momentum = (prices[-1] - prices[0]) / prices[0]
            momentum_strength = abs(momentum)
            
            # 4. Consistencia direccional
            price_changes = np.diff(prices)
            positive_changes = np.sum(price_changes > 0)
            negative_changes = np.sum(price_changes < 0)
            consistency = abs(positive_changes - negative_changes) / len(price_changes)
            
            # Combinar métricas
            trend_strength = (
                0.3 * slope_strength + 
                0.3 * adx_approx + 
                0.2 * momentum_strength + 
                0.2 * consistency
            )
            
            return min(trend_strength, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating trend strength: {e}")
            return 0.2
    
    def _analyze_volume_profile(self, volumes: np.ndarray) -> str:
        """Analizar perfil de volumen inteligentemente"""
        try:
            if len(volumes) == 0:
                return "NORMAL"
            
            # Calcular métricas de volumen
            avg_volume = np.mean(volumes)
            volume_std = np.std(volumes)
            recent_volume = np.mean(volumes[-10:])  # Últimos 10 períodos
            
            # Análisis de tendencia de volumen
            volume_trend = np.polyfit(range(len(volumes)), volumes, 1)[0]
            
            # Clasificación inteligente
            if recent_volume > avg_volume + 2 * volume_std:
                return "HIGH"
            elif recent_volume < avg_volume - volume_std:
                return "LOW"
            else:
                return "NORMAL"
                
        except Exception as e:
            self.logger.error(f"Error analyzing volume profile: {e}")
            return "NORMAL"
    
    def _detect_market_patterns(self, prices: np.ndarray) -> float:
        """Detectar patrones de mercado usando IA"""
        try:
            if len(prices) < 10:
                return 0.5
            
            # 1. Detectar tendencias
            trend_score = self._detect_trend_patterns(prices)
            
            # 2. Detectar reversiones
            reversal_score = self._detect_reversal_patterns(prices)
            
            # 3. Detectar consolidación
            consolidation_score = self._detect_consolidation_patterns(prices)
            
            # 4. Detectar breakout
            breakout_score = self._detect_breakout_patterns(prices)
            
            # Combinar scores
            pattern_score = (
                0.3 * trend_score + 
                0.25 * reversal_score + 
                0.25 * consolidation_score + 
                0.2 * breakout_score
            )
            
            return min(pattern_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error detecting market patterns: {e}")
            return 0.5
    
    def _detect_trend_patterns(self, prices: np.ndarray) -> float:
        """Detectar patrones de tendencia"""
        try:
            # Calcular medias móviles
            sma_short = np.mean(prices[-5:])
            sma_long = np.mean(prices[-15:])
            
            # Análisis de pendiente
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]
            
            # Score de tendencia
            if sma_short > sma_long and slope > 0:
                return 0.8  # Tendencia alcista fuerte
            elif sma_short < sma_long and slope < 0:
                return 0.8  # Tendencia bajista fuerte
            else:
                return 0.3  # Sin tendencia clara
                
        except Exception:
            return 0.5
    
    def _detect_reversal_patterns(self, prices: np.ndarray) -> float:
        """Detectar patrones de reversión"""
        try:
            if len(prices) < 10:
                return 0.5
            
            # Calcular RSI aproximado
            gains = np.maximum(0, np.diff(prices))
            losses = np.maximum(0, -np.diff(prices))
            
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            # Detectar sobrecompra/sobreventa
            if rsi > 80:
                return 0.7  # Posible reversión bajista
            elif rsi < 20:
                return 0.7  # Posible reversión alcista
            else:
                return 0.3
                
        except Exception:
            return 0.5
    
    def _detect_consolidation_patterns(self, prices: np.ndarray) -> float:
        """Detectar patrones de consolidación"""
        try:
            if len(prices) < 10:
                return 0.5
            
            # Calcular rango de precios
            price_range = np.max(prices) - np.min(prices)
            avg_price = np.mean(prices)
            
            # Ratio de consolidación
            consolidation_ratio = price_range / avg_price
            
            if consolidation_ratio < 0.05:  # Menos del 5% de rango
                return 0.8  # Consolidación fuerte
            elif consolidation_ratio < 0.1:  # Menos del 10% de rango
                return 0.6  # Consolidación moderada
            else:
                return 0.2  # No hay consolidación
                
        except Exception:
            return 0.5
    
    def _detect_breakout_patterns(self, prices: np.ndarray) -> float:
        """Detectar patrones de breakout"""
        try:
            if len(prices) < 10:
                return 0.5
            
            # Calcular niveles de soporte y resistencia
            recent_high = np.max(prices[-10:])
            recent_low = np.min(prices[-10:])
            current_price = prices[-1]
            
            # Detectar breakout
            if current_price > recent_high * 1.01:  # 1% por encima del máximo
                return 0.8  # Breakout alcista
            elif current_price < recent_low * 0.99:  # 1% por debajo del mínimo
                return 0.8  # Breakout bajista
            else:
                return 0.3  # Sin breakout
                
        except Exception:
            return 0.5
    
    def _classify_market_regime(self, volatility: float, trend_strength: float, 
                              volume_profile: str, pattern_score: float) -> MarketRegime:
        """Clasificar régimen de mercado basado en análisis IA"""
        try:
            # Lógica de clasificación inteligente
            if volatility > 0.8 and trend_strength > 0.7:
                regime_type = "VOLATILE"
                confidence = 0.8
                duration = 30  # 30 minutos
            elif trend_strength > 0.6:
                if pattern_score > 0.6:
                    regime_type = "BULL" if trend_strength > 0 else "BEAR"
                    confidence = 0.7
                    duration = 60  # 1 hora
                else:
                    regime_type = "SIDEWAYS"
                    confidence = 0.5
                    duration = 45  # 45 minutos
            elif volatility > 0.6:
                regime_type = "VOLATILE"
                confidence = 0.6
                duration = 20  # 20 minutos
            else:
                regime_type = "SIDEWAYS"
                confidence = 0.4
                duration = 60  # 1 hora
            
            return MarketRegime(
                regime_type=regime_type,
                confidence=confidence,
                duration=duration,
                volatility_level=volatility,
                trend_strength=trend_strength,
                volume_profile=volume_profile
            )
            
        except Exception as e:
            self.logger.error(f"Error classifying market regime: {e}")
            return MarketRegime("SIDEWAYS", 0.3, 60, 0.3, 0.2, "NORMAL")
    
    async def generate_smart_signal(self, symbol: str, data: Dict[str, Any], 
                                  market_regime: MarketRegime) -> SmartSignal:
        """Generar señal inteligente usando superprocesos de IA"""
        try:
            # Obtener datos necesarios
            prices = data.get('prices', [])
            volumes = data.get('volumes', [])
            
            if len(prices) < 20:
                return self._create_hold_signal(symbol, "Insufficient data")
            
            current_price = prices[-1]
            
            # 1. ANÁLISIS DE ENTRADA INTELIGENTE
            entry_analysis = self._analyze_entry_conditions(prices, volumes, market_regime)
            
            # 2. CÁLCULO DE NIVELES DE RIESGO
            risk_levels = self._calculate_risk_levels(prices, current_price, market_regime)
            
            # 3. GENERACIÓN DE RAZONAMIENTO
            reasoning = self._generate_smart_reasoning(entry_analysis, market_regime, risk_levels)
            
            # 4. DETERMINAR TIPO DE SEÑAL
            signal_type = self._determine_signal_type(entry_analysis, market_regime)
            
            # 5. CALCULAR CONFIANZA INTELIGENTE
            confidence = self._calculate_smart_confidence(entry_analysis, market_regime, reasoning)
            
            # 6. CREAR SEÑAL INTELIGENTE
            if signal_type == "HOLD":
                return self._create_hold_signal(symbol, reasoning[0] if reasoning else "No clear signal")
            
            return SmartSignal(
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=risk_levels['stop_loss'],
                take_profit_1=risk_levels['tp1'],
                take_profit_2=risk_levels['tp2'],
                take_profit_3=risk_levels['tp3'],
                reasoning=reasoning,
                risk_level=risk_levels['risk_level'],
                expected_duration=market_regime.duration,
                market_regime=market_regime
            )
            
        except Exception as e:
            self.logger.error(f"Error generating smart signal for {symbol}: {e}")
            return self._create_hold_signal(symbol, f"Error: {str(e)}")
    
    def _analyze_entry_conditions(self, prices: np.ndarray, volumes: np.ndarray, 
                                market_regime: MarketRegime) -> Dict[str, Any]:
        """Analizar condiciones de entrada usando IA"""
        try:
            # Calcular indicadores técnicos
            sma_20 = np.mean(prices[-20:])
            sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else sma_20
            
            # RSI aproximado
            gains = np.maximum(0, np.diff(prices))
            losses = np.maximum(0, -np.diff(prices))
            avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
            avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
            rsi = 100 - (100 / (1 + (avg_gain / max(avg_loss, 0.001))))
            
            # Momentum
            momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
            
            # Volatilidad
            volatility = np.std(np.diff(prices)) / np.mean(prices)
            
            # Análisis de volumen
            volume_ratio = np.mean(volumes[-5:]) / np.mean(volumes[-20:]) if len(volumes) >= 20 else 1
            
            return {
                'price_vs_sma20': (prices[-1] - sma_20) / sma_20,
                'price_vs_sma50': (prices[-1] - sma_50) / sma_50,
                'rsi': rsi,
                'momentum': momentum,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'trend_alignment': 1 if (prices[-1] > sma_20 > sma_50) else -1 if (prices[-1] < sma_20 < sma_50) else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing entry conditions: {e}")
            return {}
    
    def _calculate_risk_levels(self, prices: np.ndarray, current_price: float, 
                             market_regime: MarketRegime) -> Dict[str, Any]:
        """Calcular niveles de riesgo inteligentemente"""
        try:
            # Calcular ATR aproximado
            price_ranges = np.abs(np.diff(prices))
            atr = np.mean(price_ranges[-14:]) if len(price_ranges) >= 14 else np.mean(price_ranges)
            
            # Ajustar según volatilidad del régimen
            volatility_multiplier = 1 + market_regime.volatility_level
            
            # Calcular niveles
            stop_loss_distance = atr * 2 * volatility_multiplier
            take_profit_distance = atr * 3 * volatility_multiplier
            
            # Determinar dirección basada en tendencia
            trend_direction = 1 if market_regime.trend_strength > 0.5 else -1
            
            if trend_direction > 0:  # Tendencia alcista
                stop_loss = current_price - stop_loss_distance
                tp1 = current_price + take_profit_distance
                tp2 = current_price + take_profit_distance * 1.5
                tp3 = current_price + take_profit_distance * 2
            else:  # Tendencia bajista
                stop_loss = current_price + stop_loss_distance
                tp1 = current_price - take_profit_distance
                tp2 = current_price - take_profit_distance * 1.5
                tp3 = current_price - take_profit_distance * 2
            
            # Determinar nivel de riesgo
            risk_level = "HIGH" if market_regime.volatility_level > 0.7 else "MEDIUM" if market_regime.volatility_level > 0.4 else "LOW"
            
            return {
                'stop_loss': stop_loss,
                'tp1': tp1,
                'tp2': tp2,
                'tp3': tp3,
                'risk_level': risk_level,
                'atr': atr
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk levels: {e}")
            return {
                'stop_loss': current_price * 0.98,
                'tp1': current_price * 1.02,
                'tp2': current_price * 1.04,
                'tp3': current_price * 1.06,
                'risk_level': "MEDIUM",
                'atr': current_price * 0.01
            }
    
    def _generate_smart_reasoning(self, entry_analysis: Dict[str, Any], 
                                market_regime: MarketRegime, risk_levels: Dict[str, Any]) -> List[str]:
        """Generar razonamiento inteligente para la señal"""
        try:
            reasoning = []
            
            # Análisis de régimen de mercado
            reasoning.append(f"Régimen: {market_regime.regime_type} (confianza: {market_regime.confidence:.1%})")
            
            # Análisis técnico
            if entry_analysis:
                if entry_analysis.get('trend_alignment', 0) > 0:
                    reasoning.append("Alineación alcista: Precio > SMA20 > SMA50")
                elif entry_analysis.get('trend_alignment', 0) < 0:
                    reasoning.append("Alineación bajista: Precio < SMA20 < SMA50")
                
                rsi = entry_analysis.get('rsi', 50)
                if rsi > 70:
                    reasoning.append(f"RSI sobrecompra: {rsi:.1f}")
                elif rsi < 30:
                    reasoning.append(f"RSI sobreventa: {rsi:.1f}")
                else:
                    reasoning.append(f"RSI neutral: {rsi:.1f}")
                
                momentum = entry_analysis.get('momentum', 0)
                if abs(momentum) > 0.02:
                    direction = "alcista" if momentum > 0 else "bajista"
                    reasoning.append(f"Momentum {direction}: {momentum:.2%}")
            
            # Análisis de volatilidad
            if market_regime.volatility_level > 0.7:
                reasoning.append("Alta volatilidad detectada - mayor riesgo/recompensa")
            elif market_regime.volatility_level < 0.3:
                reasoning.append("Baja volatilidad - mercado consolidado")
            
            # Análisis de volumen
            if market_regime.volume_profile == "HIGH":
                reasoning.append("Alto volumen - confirmación de movimiento")
            elif market_regime.volume_profile == "LOW":
                reasoning.append("Bajo volumen - precaución en la señal")
            
            # Análisis de riesgo
            risk_level = risk_levels.get('risk_level', 'MEDIUM')
            reasoning.append(f"Nivel de riesgo: {risk_level}")
            
            return reasoning
            
        except Exception as e:
            self.logger.error(f"Error generating smart reasoning: {e}")
            return ["Análisis técnico básico aplicado"]
    
    def _determine_signal_type(self, entry_analysis: Dict[str, Any], 
                             market_regime: MarketRegime) -> str:
        """Determinar tipo de señal basado en análisis IA"""
        try:
            if not entry_analysis:
                return "HOLD"
            
            # Factores para BUY
            buy_factors = 0
            if entry_analysis.get('trend_alignment', 0) > 0:
                buy_factors += 2
            if entry_analysis.get('rsi', 50) < 70:
                buy_factors += 1
            if entry_analysis.get('momentum', 0) > 0.01:
                buy_factors += 1
            if market_regime.regime_type in ["BULL", "VOLATILE"]:
                buy_factors += 1
            
            # Factores para SELL
            sell_factors = 0
            if entry_analysis.get('trend_alignment', 0) < 0:
                sell_factors += 2
            if entry_analysis.get('rsi', 50) > 30:
                sell_factors += 1
            if entry_analysis.get('momentum', 0) < -0.01:
                sell_factors += 1
            if market_regime.regime_type in ["BEAR", "VOLATILE"]:
                sell_factors += 1
            
            # Decisión
            if buy_factors >= 3:
                return "BUY"
            elif sell_factors >= 3:
                return "SELL"
            else:
                return "HOLD"
                
        except Exception as e:
            self.logger.error(f"Error determining signal type: {e}")
            return "HOLD"
    
    def _calculate_smart_confidence(self, entry_analysis: Dict[str, Any], 
                                  market_regime: MarketRegime, reasoning: List[str]) -> float:
        """Calcular confianza inteligente de la señal"""
        try:
            base_confidence = market_regime.confidence
            
            # Ajustar por análisis técnico
            if entry_analysis:
                # RSI en zona favorable
                rsi = entry_analysis.get('rsi', 50)
                if 30 <= rsi <= 70:
                    base_confidence += 0.1
                
                # Momentum fuerte
                momentum = abs(entry_analysis.get('momentum', 0))
                if momentum > 0.02:
                    base_confidence += 0.1
                
                # Alineación de tendencia
                if entry_analysis.get('trend_alignment', 0) != 0:
                    base_confidence += 0.1
            
            # Ajustar por volatilidad
            if market_regime.volatility_level > 0.6:
                base_confidence += 0.05  # Mayor volatilidad = mayor oportunidad
            
            # Ajustar por volumen
            if market_regime.volume_profile == "HIGH":
                base_confidence += 0.05
            
            return min(base_confidence, 0.95)  # Cap en 95%
            
        except Exception as e:
            self.logger.error(f"Error calculating smart confidence: {e}")
            return 0.5
    
    def _create_hold_signal(self, symbol: str, reason: str) -> SmartSignal:
        """Crear señal HOLD con razonamiento"""
        return SmartSignal(
            symbol=symbol,
            signal_type="HOLD",
            confidence=0.3,
            entry_price=0.0,
            stop_loss=0.0,
            take_profit_1=0.0,
            take_profit_2=0.0,
            take_profit_3=0.0,
            reasoning=[reason],
            risk_level="LOW",
            expected_duration=60,
            market_regime=MarketRegime("SIDEWAYS", 0.3, 60, 0.3, 0.2, "NORMAL")
        )
    
    async def get_correlation_analysis(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """Obtener análisis de correlación entre símbolos"""
        try:
            # Esta función se implementaría con datos reales de precios
            # Por ahora, devolvemos correlaciones simuladas
            correlations = {}
            
            for symbol in symbols:
                correlations[symbol] = {}
                for other_symbol in symbols:
                    if symbol != other_symbol:
                        # Correlación simulada basada en tipo de activo
                        if "BTC" in symbol and "BTC" in other_symbol:
                            correlations[symbol][other_symbol] = 0.8
                        elif "ETH" in symbol and "ETH" in other_symbol:
                            correlations[symbol][other_symbol] = 0.7
                        else:
                            correlations[symbol][other_symbol] = np.random.uniform(0.3, 0.6)
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error getting correlation analysis: {e}")
            return {}
    
    async def optimize_portfolio_allocation(self, signals: List[SmartSignal]) -> Dict[str, float]:
        """Optimizar asignación de portfolio basada en señales"""
        try:
            if not signals:
                return {}
            
            # Calcular pesos basados en confianza y riesgo
            weights = {}
            total_weight = 0
            
            for signal in signals:
                if signal.signal_type != "HOLD":
                    # Peso basado en confianza y riesgo inverso
                    risk_multiplier = {"LOW": 1.0, "MEDIUM": 0.7, "HIGH": 0.4}[signal.risk_level]
                    weight = signal.confidence * risk_multiplier
                    weights[signal.symbol] = weight
                    total_weight += weight
            
            # Normalizar pesos
            if total_weight > 0:
                for symbol in weights:
                    weights[symbol] = weights[symbol] / total_weight
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio allocation: {e}")
            return {}
