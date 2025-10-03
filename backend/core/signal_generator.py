#!/usr/bin/env python3
"""
Generador de Señales - CryptoPulse Pro
Genera señales de trading basadas en análisis técnico, sentimiento y ML
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger, log_execution_time, log_function_call
from config.settings import TradingConfig, TimeFrame

class SignalType(str, Enum):
    """Tipos de señales"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class SignalStatus(str, Enum):
    """Estados de las señales"""
    PENDING = "PENDING"
    ACTIVE = "ACTIVE"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"

@dataclass
class Signal:
    """Estructura de una señal de trading"""
    id: str
    symbol: str
    signal_type: SignalType
    entry_price: float
    confidence_score: float
    risk_reward_ratio: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    timeframe: TimeFrame
    strategy: str
    reasoning: str
    timestamp: datetime
    status: SignalStatus = SignalStatus.PENDING
    current_price: float = 0.0
    pnl_percentage: float = 0.0
    market_context: str = "NEUTRAL"
    expected_duration: str = "MEDIUM"

class SignalGenerator:
    """
    Generador de señales de trading
    - Combina análisis técnico, sentimiento y ML
    - Calcula niveles de entrada, stop loss y take profits
    - Filtra señales de baja calidad
    - Genera reasoning detallado
    """
    
    def __init__(self, 
                 trading_config: TradingConfig,
                 technical_analyzer,
                 sentiment_analyzer,
                 news_analyzer,
                 ml_predictor,
                 data_fusion_hub,
                 risk_manager,
                 signals_db):
        self.trading_config = trading_config
        self.technical_analyzer = technical_analyzer
        self.sentiment_analyzer = sentiment_analyzer
        self.news_analyzer = news_analyzer
        self.ml_predictor = ml_predictor
        self.data_fusion_hub = data_fusion_hub
        self.risk_manager = risk_manager
        self.signals_db = signals_db
        
        self.logger = get_logger("SignalGenerator")
        self.logger.info("🎯 SignalGenerator initialized")
    
    async def initialize(self):
        """Inicializar el generador de señales"""
        self.logger.info("🔧 Initializing signal generator...")
        # Aquí se pueden inicializar modelos ML, cargar datos históricos, etc.
        self.logger.info("✅ Signal generator initialized")
    
    @log_execution_time
    async def generate_signals(self, symbol: str) -> List[Signal]:
        """Generar señales para un símbolo específico"""
        try:
            self.logger.info(f"🔍 Generating signals for {symbol}")
            
            # Obtener datos de mercado
            market_data = await self._get_market_data(symbol)
            if not market_data:
                self.logger.warning(f"No market data available for {symbol}")
                return []
            
            # Realizar análisis
            analysis_results = await self._perform_analysis(symbol, market_data)
            
            # Generar señales basadas en el análisis
            signals = await self._create_signals(symbol, analysis_results, market_data)
            
            # Filtrar señales de baja calidad
            filtered_signals = await self._filter_signals(signals)
            
            # Guardar señales en la base de datos
            if filtered_signals:
                await self._save_signals(filtered_signals)
                self.logger.info(f"Generated {len(filtered_signals)} signals for {symbol}")
            
            return filtered_signals
            
        except Exception as e:
            self.logger.error(f"Error generating signals for {symbol}", exception=e)
            return []
    
    @log_function_call
    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Obtener datos de mercado para el símbolo"""
        try:
            # Obtener datos de diferentes timeframes
            market_data = {}
            
            for timeframe in self.trading_config.analysis_timeframes:
                data = await self.technical_analyzer.get_historical_data(symbol, timeframe)
                if data:
                    market_data[timeframe] = data
            
            return market_data if market_data else None
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}", exception=e)
            return None
    
    @log_function_call
    async def _perform_analysis(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Realizar análisis completo del símbolo usando Data Fusion Hub"""
        try:
            # Análisis técnico
            technical_analysis = await self.technical_analyzer.analyze_symbol(symbol, market_data)
            
            # Análisis de sentimiento
            sentiment_analysis = await self.sentiment_analyzer.analyze_sentiment(symbol)
            
            # Análisis de noticias
            news_analysis = await self.news_analyzer.analyze_news(symbol)
            
            # Análisis ML
            ml_analysis = {}
            if self.ml_predictor:
                ml_analysis = await self.ml_predictor.predict_price_movement(symbol, market_data)
            
            # FUSIONAR TODOS LOS DATOS CON DATA FUSION HUB
            fused_analysis = await self.data_fusion_hub.fuse_all_data(
                technical=technical_analysis,
                ml=ml_analysis,
                news=news_analysis,
                onchain={},  # Por ahora vacío, se puede agregar después
                sentiment=sentiment_analysis
            )
            
            return fused_analysis
            
        except Exception as e:
            self.logger.error(f"Error performing analysis for {symbol}: {e}")
            return {}
    
    @log_function_call
    async def _calculate_consensus(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calcular consenso entre diferentes análisis"""
        consensus = {
            'overall_sentiment': 'NEUTRAL',
            'confidence_score': 0.0,
            'signal_strength': 0.0,
            'risk_level': 'MEDIUM',
            'timeframe_preference': 'MEDIUM'
        }
        
        try:
            # Ponderar diferentes análisis
            technical_weight = 0.4
            sentiment_weight = 0.3
            news_weight = 0.2
            ml_weight = 0.1
            
            # Calcular score técnico
            technical_score = 0.0
            if analysis_results['technical']:
                technical_score = analysis_results['technical'].get('overall_score', 0.0)
            
            # Calcular score de sentimiento
            sentiment_score = 0.0
            if analysis_results['sentiment']:
                sentiment_score = analysis_results['sentiment'].get('overall_sentiment_score', 0.0)
            
            # Calcular score de noticias
            news_score = 0.0
            if analysis_results['news']:
                news_score = analysis_results['news'].get('impact_score', 0.0)
            
            # Calcular score ML
            ml_score = 0.0
            if analysis_results['ml_prediction']:
                ml_score = analysis_results['ml_prediction'].get('confidence', 0.0)
            
            # Calcular consenso ponderado
            consensus_score = (
                technical_score * technical_weight +
                sentiment_score * sentiment_weight +
                news_score * news_weight +
                ml_score * ml_weight
            )
            
            consensus['confidence_score'] = min(max(consensus_score, 0.0), 100.0)
            
            # Determinar sentimiento general
            if consensus_score >= 70:
                consensus['overall_sentiment'] = 'BULLISH'
            elif consensus_score <= 30:
                consensus['overall_sentiment'] = 'BEARISH'
            else:
                consensus['overall_sentiment'] = 'NEUTRAL'
            
            # Determinar fuerza de la señal
            if consensus_score >= 80:
                consensus['signal_strength'] = 'STRONG'
            elif consensus_score >= 60:
                consensus['signal_strength'] = 'MEDIUM'
            else:
                consensus['signal_strength'] = 'WEAK'
            
            # Determinar nivel de riesgo
            if consensus_score >= 85 or consensus_score <= 15:
                consensus['risk_level'] = 'LOW'
            elif consensus_score >= 70 or consensus_score <= 30:
                consensus['risk_level'] = 'MEDIUM'
            else:
                consensus['risk_level'] = 'HIGH'
            
            return consensus
            
        except Exception as e:
            self.logger.error("Error calculating consensus", exception=e)
            return consensus
    
    @log_function_call
    async def _create_signals(self, symbol: str, analysis_results: Dict[str, Any], 
                            market_data: Dict[str, Any]) -> List[Signal]:
        """Crear señales basadas en el análisis"""
        signals = []
        
        try:
            consensus = analysis_results['consensus']
            technical = analysis_results['technical']
            
            # Solo generar señales si hay consenso suficiente
            if consensus['confidence_score'] < self.trading_config.min_confidence_score:
                return signals
            
            # Obtener precio actual
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return signals
            
            # Determinar tipo de señal
            signal_type = self._determine_signal_type(consensus, technical)
            if signal_type == SignalType.HOLD:
                return signals
            
            # Calcular niveles de entrada, stop loss y take profits
            entry_price, stop_loss, take_profits = await self._calculate_levels(
                symbol, current_price, signal_type, analysis_results
            )
            
            # Calcular risk-reward ratio
            risk_reward_ratio = self._calculate_risk_reward_ratio(
                entry_price, stop_loss, take_profits[0]
            )
            
            # Verificar ratio mínimo
            if risk_reward_ratio < self.trading_config.min_risk_reward_ratio:
                self.logger.debug(f"Signal rejected: risk-reward ratio {risk_reward_ratio:.2f} too low")
                return signals
            
            # Crear señal
            signal = Signal(
                id=f"{symbol}_{signal_type.value}_{int(datetime.utcnow().timestamp())}",
                symbol=symbol,
                signal_type=signal_type,
                entry_price=entry_price,
                confidence_score=consensus['confidence_score'],
                risk_reward_ratio=risk_reward_ratio,
                stop_loss=stop_loss,
                take_profit_1=take_profits[0],
                take_profit_2=take_profits[1],
                take_profit_3=take_profits[2],
                timeframe=self._select_optimal_timeframe(analysis_results),
                strategy=self._select_strategy(consensus, technical),
                reasoning=self._generate_reasoning(analysis_results),
                timestamp=datetime.utcnow(),
                current_price=current_price,
                market_context=consensus['overall_sentiment']
            )
            
            signals.append(signal)
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error creating signals for {symbol}", exception=e)
            return signals
    
    def _determine_signal_type(self, consensus: Dict[str, Any], technical: Dict[str, Any]) -> SignalType:
        """Determinar el tipo de señal basado en el análisis"""
        if consensus['overall_sentiment'] == 'BULLISH' and consensus['confidence_score'] >= 70:
            return SignalType.BUY
        elif consensus['overall_sentiment'] == 'BEARISH' and consensus['confidence_score'] >= 70:
            return SignalType.SELL
        else:
            return SignalType.HOLD
    
    async def _calculate_levels(self, symbol: str, current_price: float, signal_type: SignalType,
                              analysis_results: Dict[str, Any]) -> Tuple[float, float, List[float]]:
        """Calcular niveles de entrada, stop loss y take profits"""
        try:
            # Obtener volatilidad del símbolo
            volatility = await self._get_symbol_volatility(symbol)
            
            # Calcular stop loss basado en volatilidad
            if signal_type == SignalType.BUY:
                stop_loss = current_price * (1 - volatility * 2)  # 2x volatilidad
                take_profit_1 = current_price * (1 + volatility * 2)
                take_profit_2 = current_price * (1 + volatility * 4)
                take_profit_3 = current_price * (1 + volatility * 8)
            else:  # SELL
                stop_loss = current_price * (1 + volatility * 2)
                take_profit_1 = current_price * (1 - volatility * 2)
                take_profit_2 = current_price * (1 - volatility * 4)
                take_profit_3 = current_price * (1 - volatility * 8)
            
            return current_price, stop_loss, [take_profit_1, take_profit_2, take_profit_3]
            
        except Exception as e:
            self.logger.error(f"Error calculating levels for {symbol}", exception=e)
            # Valores por defecto
            if signal_type == SignalType.BUY:
                return current_price, current_price * 0.98, [current_price * 1.02, current_price * 1.04, current_price * 1.08]
            else:
                return current_price, current_price * 1.02, [current_price * 0.98, current_price * 0.96, current_price * 0.92]
    
    async def _get_symbol_volatility(self, symbol: str) -> float:
        """Obtener volatilidad del símbolo"""
        try:
            # Usar ATR para calcular volatilidad
            atr_data = await self.technical_analyzer.get_atr(symbol, TimeFrame.H1, 14)
            if atr_data and len(atr_data) > 0:
                return atr_data[-1] / 100  # Convertir a porcentaje
            else:
                return 0.02  # 2% por defecto
        except Exception as e:
            self.logger.error(f"Error getting volatility for {symbol}", exception=e)
            return 0.02
    
    def _calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, take_profit: float) -> float:
        """Calcular ratio riesgo-recompensa"""
        if entry_price > stop_loss:  # BUY
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:  # SELL
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        
        return reward / risk if risk > 0 else 0.0
    
    def _select_optimal_timeframe(self, analysis_results: Dict[str, Any]) -> TimeFrame:
        """Seleccionar timeframe óptimo basado en el análisis"""
        # Lógica para seleccionar timeframe basado en volatilidad y tendencia
        technical = analysis_results.get('technical', {})
        
        if technical.get('trend_strength', 0) > 0.7:
            return TimeFrame.H4  # Tendencias fuertes en 4H
        elif technical.get('volatility', 0) > 0.05:
            return TimeFrame.M15  # Alta volatilidad en 15M
        else:
            return TimeFrame.H1  # Default 1H
    
    def _select_strategy(self, consensus: Dict[str, Any], technical: Dict[str, Any]) -> str:
        """Seleccionar estrategia basada en el análisis"""
        if consensus['signal_strength'] == 'STRONG' and technical.get('trend_strength', 0) > 0.8:
            return 'breakout'
        elif technical.get('volatility', 0) > 0.03:
            return 'scalping'
        elif consensus['overall_sentiment'] in ['BULLISH', 'BEARISH']:
            return 'swing'
        else:
            return 'mean_reversion'
    
    def _generate_reasoning(self, analysis_results: Dict[str, Any]) -> str:
        """Generar reasoning detallado para la señal"""
        reasoning_parts = []
        
        # Análisis técnico
        technical = analysis_results.get('technical', {})
        if technical:
            if technical.get('trend_direction') == 'UP':
                reasoning_parts.append("Tendencia alcista detectada")
            elif technical.get('trend_direction') == 'DOWN':
                reasoning_parts.append("Tendencia bajista detectada")
            
            if technical.get('momentum', {}).get('rsi', 0) > 70:
                reasoning_parts.append("RSI sobrecomprado")
            elif technical.get('momentum', {}).get('rsi', 0) < 30:
                reasoning_parts.append("RSI sobrevendido")
        
        # Análisis de sentimiento
        sentiment = analysis_results.get('sentiment', {})
        if sentiment:
            if sentiment.get('overall_sentiment_score', 0) > 0.7:
                reasoning_parts.append("Sentimiento muy positivo")
            elif sentiment.get('overall_sentiment_score', 0) < 0.3:
                reasoning_parts.append("Sentimiento muy negativo")
        
        # Análisis de noticias
        news = analysis_results.get('news', {})
        if news:
            if news.get('impact_score', 0) > 0.7:
                reasoning_parts.append("Noticias de alto impacto")
        
        # Consenso
        consensus = analysis_results.get('consensus', {})
        if consensus:
            reasoning_parts.append(f"Confianza: {consensus.get('confidence_score', 0):.1f}%")
            reasoning_parts.append(f"Fuerza: {consensus.get('signal_strength', 'MEDIUM')}")
        
        return " | ".join(reasoning_parts) if reasoning_parts else "Análisis técnico estándar"
    
    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Obtener precio actual del símbolo"""
        try:
            return await self.technical_analyzer.get_current_price(symbol)
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}", exception=e)
            return None
    
    async def _filter_signals(self, signals: List[Signal]) -> List[Signal]:
        """Filtrar señales de baja calidad"""
        filtered_signals = []
        
        for signal in signals:
            # Filtrar por confianza mínima
            if signal.confidence_score < self.trading_config.min_confidence_score:
                continue
            
            # Filtrar por risk-reward ratio
            if signal.risk_reward_ratio < self.trading_config.min_risk_reward_ratio:
                continue
            
            # Filtrar por volatilidad mínima
            if abs(signal.entry_price - signal.stop_loss) / signal.entry_price < 0.01:  # 1% mínimo
                continue
            
            filtered_signals.append(signal)
        
        return filtered_signals
    
    async def _save_signals(self, signals: List[Signal]):
        """Guardar señales en la base de datos"""
        try:
            for signal in signals:
                signal_data = {
                    'id': signal.id,
                    'symbol': signal.symbol,
                    'signal_type': signal.signal_type.value,
                    'entry_price': signal.entry_price,
                    'confidence_score': signal.confidence_score,
                    'risk_reward_ratio': signal.risk_reward_ratio,
                    'stop_loss': signal.stop_loss,
                    'take_profit_1': signal.take_profit_1,
                    'take_profit_2': signal.take_profit_2,
                    'take_profit_3': signal.take_profit_3,
                    'timeframe': signal.timeframe.value,
                    'strategy': signal.strategy,
                    'reasoning': signal.reasoning,
                    'timestamp': signal.timestamp.isoformat(),
                    'status': signal.status.value,
                    'current_price': signal.current_price,
                    'market_context': signal.market_context,
                    'expected_duration': signal.expected_duration
                }
                
                await self.signals_db.save_signal(signal_data)
                self.logger.trading_signal(signal_data)
                
        except Exception as e:
            self.logger.error("Error saving signals", exception=e)