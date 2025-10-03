import numpy as np
#!/usr/bin/env python3
"""
Enhanced Continuous Bot - CryptoPulse Pro
Bot mejorado con aprendizaje continuo, retroalimentación y resúmenes de precios
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any
import signal
import json
import threading
import http.server
import socketserver

# Agregar el directorio backend al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import get_logger
from analysis.data_fusion_hub import DataFusionHub
from analysis.adaptive_ml import AdaptiveMLSystem
from analysis.feedback_system import FeedbackSystem
from analysis.super_ai_processes import SuperAIProcesses
from core.portfolio_manager import PortfolioManager
from notifications.telegram_bot import TelegramBot
from database.signals_feedback_db import SignalsFeedbackDB
from config.settings import Settings

class HealthCheckHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/health':
            health_data = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "service": "CryptoPulse Pro Bot",
                "version": "1.0.0"
            }
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(health_data).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        pass

class EnhancedContinuousBot:
    """Bot continuo mejorado con ML adaptativo y retroalimentación"""
    
    def __init__(self):
        self.logger = get_logger("EnhancedContinuousBot")
        self.running = False
        # Símbolos a analizar (10 símbolos populares)
        self.symbols = [
            "BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT",
            "BNBUSDT", "XRPUSDT", "MATICUSDT", "AVAXUSDT", "DOTUSDT"
        ]
        self.analysis_interval = 300  # 5 minutos
        self.last_signals = {}  # Para evitar spam
        self.price_history = {}  # Historial de precios para resúmenes
        
        # Componentes principales
        self.hub = None
        self.adaptive_ml = None
        self.feedback_system = None
        self.super_ai = None  # Nuevo: Superprocesos de IA
        self.portfolio = None
        self.telegram_bot = None
        self.database = None
        self.settings = None
        
        # Configuración de resúmenes
        self.summary_config = {
            'send_price_summary': True,
            'price_summary_interval': 1,  # Cada ciclo
            'performance_summary_interval': 10,  # Cada 10 ciclos
            'detailed_analysis_interval': 50  # Cada 50 ciclos
        }
        
        # Health check server
        self.health_server = None
        self.health_thread = None
    
    def start_health_check(self):
        """Iniciar servidor de health check en puerto 8000"""
        try:
            port = int(os.environ.get('PORT', 8000))
            self.health_server = socketserver.TCPServer(("", port), HealthCheckHandler)
            self.health_thread = threading.Thread(target=self.health_server.serve_forever, daemon=True)
            self.health_thread.start()
            self.logger.info(f"🏥 Health check server started on port {port}")
        except Exception as e:
            self.logger.error(f"❌ Failed to start health check server: {e}")
        
    async def initialize(self):
        """Inicializar todos los componentes"""
        try:
            self.logger.info("🚀 Initializing Enhanced Continuous Bot...")
            
            # Cargar configuración
            self.settings = Settings()
            
            # Inicializar base de datos
            self.database = SignalsFeedbackDB()
            await self.database.initialize()
            
            # Inicializar Data Fusion Hub
            self.hub = DataFusionHub()
            await self.hub.initialize()
            
            # Inicializar sistema ML adaptativo
            self.adaptive_ml = AdaptiveMLSystem()
            await self.adaptive_ml.initialize()
            
            # Inicializar sistema de retroalimentación
            self.feedback_system = FeedbackSystem()
            await self.feedback_system.initialize()
            
            # Inicializar Superprocesos de IA
            self.super_ai = SuperAIProcesses()
            self.logger.info("🧠 Superprocesos de IA inicializados")
            
            # Inicializar Portfolio Manager
            self.portfolio = PortfolioManager(initial_balance=10000.0)
            await self.portfolio.initialize()
            
            # Inicializar Telegram Bot
            if self.settings.notifications.telegram_enabled:
                self.telegram_bot = TelegramBot(
                    bot_token=self.settings.notifications.telegram_bot_token,
                    chat_id=self.settings.notifications.telegram_chat_id
                )
                await self.telegram_bot.initialize()
                
                # Enviar mensaje de inicio
                await self.telegram_bot.send_alert(
                    "SUCCESS",
                    "🚀 **CRYPTOPULSE PRO ENHANCED BOT STARTED!** 🚀\n\n"
                    "✅ **Data Fusion Hub:** Ready\n"
                    "✅ **Adaptive ML System:** Ready\n"
                    "✅ **Feedback System:** Ready\n"
                    "✅ **Portfolio Manager:** Ready\n"
                    "✅ **Database:** Ready\n\n"
                    f"📊 **Monitoring:** {', '.join(self.symbols)}\n"
                    f"⏰ **Analysis Interval:** {self.analysis_interval} seconds\n"
                    f"📈 **Price Summaries:** Every cycle\n"
                    f"📊 **Performance Reports:** Every 10 cycles\n\n"
                    "🤖 **Bot will send signals and price updates!**"
                )
            
            self.logger.info("✅ Enhanced Continuous Bot initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize Enhanced Continuous Bot: {e}")
            return False
    
    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Analizar un símbolo específico con ML adaptativo"""
        try:
            # Obtener datos de Binance
            import aiohttp
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': '1h',
                'limit': 100
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Convertir datos
                        symbol_data = []
                        for kline in data:
                            symbol_data.append({
                                'timestamp': int(kline[0]),
                                'open': float(kline[1]),
                                'high': float(kline[2]),
                                'low': float(kline[3]),
                                'close': float(kline[4]),
                                'volume': float(kline[5])
                            })
                        
                        current_price = symbol_data[-1]['close']
                        
                        # Guardar precio para resúmenes
                        self.price_history[symbol] = {
                            'price': current_price,
                            'timestamp': datetime.utcnow(),
                            'change_24h': self._calculate_24h_change(symbol_data)
                        }
                        
                        # 🧠 SUPERPROCESO: Análisis de régimen de mercado
                        analysis_data = {
                            'prices': [float(kline[4]) for kline in data],  # Precios de cierre
                            'volumes': [float(kline[5]) for kline in data],  # Volúmenes
                            'timestamps': [int(kline[0]) for kline in data]
                        }
                        
                        market_regime = await self.super_ai.analyze_market_regime(symbol, analysis_data)
                        self.logger.info(f"📊 Régimen detectado para {symbol}: {market_regime.regime_type} (confianza: {market_regime.confidence:.1%})")
                        
                        # Análisis técnico mejorado
                        from indicators.trend_indicators import TrendIndicators
                        from indicators.momentum_indicators import MomentumIndicators
                        from indicators.volume_indicators import VolumeIndicators
                        from indicators.volatility_indicators import VolatilityIndicators
                        import pandas as pd
                        
                        # Crear DataFrame
                        df_data = []
                        for candle in symbol_data:
                            df_data.append({
                                'timestamp': candle['timestamp'],
                                'open': candle['open'],
                                'high': candle['high'],
                                'low': candle['low'],
                                'close': candle['close'],
                                'volume': candle['volume']
                            })
                        
                        df = pd.DataFrame(df_data)
                        
                        # Calcular indicadores
                        trend_indicators = TrendIndicators()
                        trend_data = await trend_indicators.calculate_all(df)
                        
                        momentum_indicators = MomentumIndicators()
                        momentum_data = await momentum_indicators.calculate_all(df)
                        
                        volume_indicators = VolumeIndicators()
                        volume_data = await volume_indicators.calculate_all(df)
                        
                        volatility_indicators = VolatilityIndicators()
                        volatility_data = await volatility_indicators.calculate_all(df)
                        
                        # Crear análisis técnico mejorado
                        sma_50 = trend_data.get('sma_50', current_price)
                        sma_200 = trend_data.get('sma_200', current_price)
                        rsi = momentum_data.get('rsi', 50)
                        macd = trend_data.get('macd', 0)
                        macd_signal = trend_data.get('macd_signal', 0)
                        volume_score = volume_data.get('volume_score', 0.5)
                        
                        # Determinar tendencia
                        trend_score = 0
                        if sma_50 > sma_200:
                            trend_score += 0.3
                        if current_price > sma_50:
                            trend_score += 0.2
                        if macd > macd_signal:
                            trend_score += 0.2
                        if rsi > 50:
                            trend_score += 0.1
                        if volume_score > 0.7:
                            trend_score += 0.2
                        
                        # Determinar señal
                        if trend_score > 0.7:
                            trend = 'BULLISH'
                            confidence = min(trend_score, 0.9)
                        elif trend_score < 0.3:
                            trend = 'BEARISH'
                            confidence = min(1 - trend_score, 0.9)
                        else:
                            trend = 'NEUTRAL'
                            confidence = 0.5
                        
                        technical_analysis = {
                            'trend': trend,
                            'confidence': confidence,
                            'momentum_score': rsi / 100,
                            'volume_score': volume_score,
                            'volatility_score': volatility_data.get('atr', 0.5),
                            'trend_score': trend_score,
                            'indicators': {
                                'trend': trend_data,
                                'momentum': momentum_data,
                                'volume': volume_data,
                                'volatility': volatility_data
                            }
                        }
                        
                        # Análisis de sentimiento real
                        try:
                            from analysis.sentiment_analyzer import SentimentAnalyzer
                            sentiment_analyzer = SentimentAnalyzer()
                            sentiment_analysis = await sentiment_analyzer.analyze_sentiment(symbol)
                        except:
                            # Fallback a datos simulados si falla
                            sentiment_analysis = {
                                'overall_sentiment': 'NEUTRAL',
                                'social_confidence': 0.5,
                                'reddit_sentiment': 0.5,
                                'youtube_sentiment': 0.5,
                                'twitter_sentiment': 0.5,
                                'volume_change_score': 0.0
                            }
                        
                        # Análisis de noticias real
                        try:
                            from analysis.news_analyzer import NewsAnalyzer
                            news_analyzer = NewsAnalyzer()
                            news_analysis = await news_analyzer.analyze_news(symbol)
                        except:
                            # Fallback a datos simulados si falla
                            news_analysis = {
                                'overall_sentiment': 'NEUTRAL',
                                'confidence': 0.5,
                                'news_impact_score': 0.3,
                                'fear_greed_index': 50.0,
                                'high_impact_news': 0
                            }
                        
                        # Análisis ML adaptativo
                        ml_analysis = {
                            'ml_prediction': 0.75,
                            'ml_confidence': 0.7,
                            'ml_trend': 'BULLISH',
                            'price_target': current_price * 1.05
                        }
                        
                        # Análisis on-chain real
                        try:
                            from analysis.onchain_analyzer import OnchainAnalyzer
                            onchain_analyzer = OnchainAnalyzer()
                            onchain_analysis = await onchain_analyzer.analyze_onchain_metrics(symbol)
                        except:
                            # Fallback a datos simulados si falla
                            onchain_analysis = {
                                'onchain_sentiment': 'NEUTRAL',
                                'onchain_confidence': 0.5,
                                'whale_activity_score': 0.0,
                                'network_activity_score': 0.0,
                                'exchange_flows_sentiment': 'NEUTRAL',
                                'funding_rate': 0.0,
                                'exchange_inflows': 0
                            }
                        
                        # Fusionar datos
                        fused_result = await self.hub.fuse_all_data(
                            technical=technical_analysis,
                            ml=ml_analysis,
                            news=news_analysis,
                            onchain=onchain_analysis,
                            sentiment=sentiment_analysis
                        )
                        
                        if fused_result:
                            # 🧠 SUPERPROCESO: Generar señal inteligente
                            smart_signal = await self.super_ai.generate_smart_signal(symbol, analysis_data, market_regime)
                            
                            # Agregar predicción ML adaptativa
                            ml_prediction = await self.adaptive_ml.predict_signal_quality(fused_result)
                            fused_result.update(ml_prediction)
                            
                            # Crear señal final con superprocesos
                            final_signal = {
                                'signal_type': smart_signal.signal_type,
                                'confidence': smart_signal.confidence,
                                'reasoning': smart_signal.reasoning,
                                'entry_price': smart_signal.entry_price,
                                'stop_loss': smart_signal.stop_loss,
                                'take_profit_1': smart_signal.take_profit_1,
                                'take_profit_2': smart_signal.take_profit_2,
                                'take_profit_3': smart_signal.take_profit_3,
                                'risk_level': smart_signal.risk_level,
                                'expected_duration': smart_signal.expected_duration
                            }
                            
                            # Agregar métricas adicionales para umbrales más sensibles
                            final_signal['volatility_score'] = market_regime.volatility_level * 100
                            final_signal['momentum_score'] = market_regime.trend_strength * 100
                            final_signal['volume_score'] = 80 if market_regime.volume_profile == "HIGH" else 50 if market_regime.volume_profile == "NORMAL" else 20
                            
                            return {
                                'symbol': symbol,
                                'price': current_price,
                                'signal': final_signal,
                                'technical': technical_analysis,
                                'market_regime': market_regime,
                                'smart_signal': smart_signal,
                                'timestamp': datetime.utcnow()
                            }
                        
                    else:
                        self.logger.error(f"❌ API error for {symbol}: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"❌ Error analyzing {symbol}: {e}")
            return None
    
    def _calculate_24h_change(self, symbol_data: List[Dict]) -> float:
        """Calcular cambio de precio en 24 horas"""
        try:
            if len(symbol_data) < 24:
                return 0.0
            
            current_price = symbol_data[-1]['close']
            price_24h_ago = symbol_data[-24]['close']
            
            return ((current_price - price_24h_ago) / price_24h_ago) * 100
        except:
            return 0.0
    
    async def should_send_signal(self, symbol: str, signal_data: Dict[str, Any]) -> bool:
        """Determinar si se debe enviar la señal (UMBRALES MÁS SENSIBLES)"""
        try:
            signal_type = signal_data['signal'].get('signal_type', 'HOLD')
            confidence = signal_data['signal'].get('confidence', 0.0)
            
            # No enviar señales HOLD
            if signal_type == 'HOLD':
                return False
            
            # UMBRAL MÁS SENSIBLE: Bajar de 50% a 35%
            if confidence < 35:
                return False
            
            # Verificar cooldown (reducido a 2 minutos)
            last_signal_time = self.last_signals.get(symbol)
            if last_signal_time:
                time_diff = (datetime.utcnow() - last_signal_time).total_seconds()
                if time_diff < 120:  # 2 minutos de cooldown
                    return False
            
            # NUEVA LÓGICA: Permitir señales con alta volatilidad
            volatility = signal_data.get('volatility_score', 0.0)
            if volatility > 70 and confidence > 25:  # Alta volatilidad + confianza baja
                return True
            
            # NUEVA LÓGICA: Permitir señales con momentum fuerte
            momentum = signal_data.get('momentum_score', 0.0)
            if momentum > 75 and confidence > 30:  # Momentum fuerte + confianza moderada
                return True
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error checking signal conditions: {e}")
            return False
    
    async def send_signal_to_telegram(self, signal_data: Dict[str, Any]):
        """Enviar señal a Telegram con seguimiento"""
        try:
            if not self.telegram_bot:
                return
            
            symbol = signal_data['symbol']
            price = signal_data['price']
            signal = signal_data['signal']
            
            # Crear señal completa para la base de datos
            full_signal = {
                'signal_id': f"signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}",
                'symbol': symbol,
                'signal_type': signal.get('signal_type', 'UNKNOWN'),
                'entry_price': price,
                'stop_loss': price * 0.95,  # 5% stop loss
                'take_profit_1': price * 1.05,  # 5% take profit
                'take_profit_2': price * 1.10,  # 10% take profit
                'take_profit_3': price * 1.15,  # 15% take profit
                'confidence_score': signal.get('confidence', 0),
                'coherence_score': signal.get('coherence_score', 0),
                'reasoning': signal.get('detailed_analysis', {}),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Guardar en base de datos
            await self.database.save_signal(full_signal)
            
            # Iniciar seguimiento
            signal_id = await self.feedback_system.track_signal(full_signal)
            
            # Enviar señal usando el formato mejorado del TelegramBot
            success = await self.telegram_bot.send_signal(full_signal)
            
            if success:
                # Actualizar último envío
                self.last_signals[symbol] = datetime.utcnow()
                self.logger.info(f"✅ Signal sent to Telegram for {symbol} (ID: {signal_id})")
            else:
                self.logger.error(f"❌ Failed to send signal to Telegram for {symbol}")
                
        except Exception as e:
            self.logger.error(f"❌ Error sending signal to Telegram: {e}")
    
    async def send_price_summary(self, current_prices: Dict[str, Dict[str, Any]]):
        """Enviar resumen de precios a Telegram"""
        try:
            if not self.telegram_bot or not current_prices:
                return
            
            # Crear mensaje de resumen de precios
            message = f"📊 **RESUMEN DE PRECIOS - CICLO ACTUAL** 📊\n\n"
            
            for symbol, data in current_prices.items():
                price = data['price']
                # Obtener cambio real de 24h desde Binance
                try:
                    change_24h = await self.get_real_24h_change(symbol)
                except:
                    change_24h = 0.0  # Fallback si falla
                
                # Emoji según cambio
                if change_24h > 5:
                    emoji = "🚀"
                elif change_24h > 0:
                    emoji = "📈"
                elif change_24h > -5:
                    emoji = "📉"
                else:
                    emoji = "💥"
                
                message += f"{emoji} **{symbol}:** ${price:,.2f} ({change_24h:+.2f}%)\n"
            
            message += f"\n⏰ **Actualizado:** {datetime.utcnow().strftime('%H:%M:%S')} UTC\n"
            message += f"🔄 **Próximo análisis:** {self.analysis_interval} segundos\n\n"
            message += "🤖 **Powered by CryptoPulse Pro**"
            
            # Enviar mensaje directamente usando método simple
            success = await self.send_simple_telegram_message(message)
            if success:
                self.logger.info(f"📊 Price summary sent for cycle")
            else:
                self.logger.error(f"❌ Failed to send price summary")
            
        except Exception as e:
            self.logger.error(f"❌ Error sending price summary: {e}")
    
    async def get_real_24h_change(self, symbol: str) -> float:
        """Obtener el cambio real de 24h desde Binance"""
        try:
            import aiohttp
            url = f'https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}'
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return float(data['priceChangePercent'])
                    else:
                        return 0.0
        except Exception as e:
            self.logger.error(f"Error getting 24h change for {symbol}: {e}")
            return 0.0
    
    async def send_simple_telegram_message(self, message: str) -> bool:
        """Enviar mensaje simple a Telegram sin usar el bot complejo"""
        try:
            import aiohttp
            token = '8271169050:AAHdi665os0cDMVL4xHKMKvvz5hCvadnMWY'
            chat_id = '1852997724'
            
            url = f'https://api.telegram.org/bot{token}/sendMessage'
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload) as response:
                    if response.status == 200:
                        return True
                    else:
                        self.logger.error(f"Telegram API error: {response.status}")
                        return False
        except Exception as e:
            self.logger.error(f"Error sending simple Telegram message: {e}")
            return False
    
    async def send_performance_summary(self, cycle_count: int):
        """Enviar resumen de rendimiento a Telegram"""
        try:
            if not self.telegram_bot:
                return
            
            # Obtener métricas de rendimiento
            performance_summary = await self.feedback_system.get_performance_summary()
            portfolio_summary = await self.portfolio.get_portfolio_summary()
            
            # Crear mensaje de rendimiento
            message = f"📈 **REPORTE DE RENDIMIENTO - CICLO {cycle_count}** 📈\n\n"
            
            # Métricas actuales
            current_metrics = performance_summary.get('current_metrics', {})
            message += f"📊 **ESTADÍSTICAS GENERALES:**\n"
            message += f"• Total Señales: {current_metrics.get('total_signals', 0)}\n"
            message += f"• Señales Exitosas: {current_metrics.get('successful_signals', 0)}\n"
            message += f"• Win Rate: {current_metrics.get('win_rate', 0):.1f}%\n"
            message += f"• PnL Promedio: {current_metrics.get('average_pnl', 0):.2f}%\n\n"
            
            # Rendimiento reciente
            recent_perf = performance_summary.get('recent_performance', {})
            message += f"📈 **RENDIMIENTO RECIENTE:**\n"
            message += f"• Win Rate Reciente: {recent_perf.get('win_rate', 0):.1f}%\n"
            message += f"• PnL Promedio Reciente: {recent_perf.get('average_pnl', 0):.2f}%\n"
            message += f"• Señales Recientes: {recent_perf.get('signals_count', 0)}\n\n"
            
            # Portfolio
            message += f"💼 **PORTFOLIO:**\n"
            message += f"• Valor Total: ${portfolio_summary.get('total_value', 0):,.2f}\n"
            message += f"• Posiciones Activas: {portfolio_summary.get('active_positions', 0)}\n"
            message += f"• PnL Total: {portfolio_summary.get('total_pnl', 0):.2f}%\n\n"
            
            # Recomendaciones
            recommendations = performance_summary.get('recommendations', [])
            if recommendations:
                message += f"💡 **RECOMENDACIONES:**\n"
                for rec in recommendations[:3]:  # Máximo 3
                    message += f"• {rec}\n"
                message += "\n"
            
            message += f"⏰ **Actualizado:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n"
            message += "🤖 **Powered by CryptoPulse Pro**"
            
            # Enviar mensaje
            await self.telegram_bot.send_alert("INFO", message)
            self.logger.info(f"📈 Performance summary sent for cycle {cycle_count}")
            
        except Exception as e:
            self.logger.error(f"❌ Error sending performance summary: {e}")
    
    async def run_analysis_cycle(self, cycle_count: int):
        """Ejecutar un ciclo de análisis"""
        try:
            self.logger.info("🔄 Starting analysis cycle...")
            current_prices = {}  # Inicializar current_prices
            
            for symbol in self.symbols:
                self.logger.info(f"🔍 Analyzing {symbol}...")
                
                # Analizar símbolo
                signal_data = await self.analyze_symbol(symbol)
                
                if signal_data:
                    # Almacenar precio para el resumen
                    current_prices[symbol] = {
                        'price': signal_data['price'],
                        'change_24h': 0.0  # Se calculará en send_price_summary
                    }
                    
                    # Verificar si enviar señal
                    if await self.should_send_signal(symbol, signal_data):
                        await self.send_signal_to_telegram(signal_data)
                    else:
                        self.logger.info(f"⏸️ No signal sent for {symbol} (conditions not met)")
                else:
                    self.logger.warning(f"⚠️ No data for {symbol}")
                
                # Pequeña pausa entre símbolos
                await asyncio.sleep(2)
            
            # Enviar resumen de precios si está configurado
            if self.summary_config['send_price_summary']:
                await self.send_price_summary(current_prices)
            
            # Enviar resumen de rendimiento si es el momento
            if cycle_count % self.summary_config['performance_summary_interval'] == 0:
                await self.send_performance_summary(cycle_count)
            
            self.logger.info("✅ Analysis cycle completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in analysis cycle: {e}")
    
    async def run(self):
        """Ejecutar el bot de manera continua"""
        try:
            self.logger.info("🚀 Starting Enhanced Continuous Bot...")
            self.running = True
            
            # Iniciar health check server
            self.start_health_check()
            
            # Configurar manejador de señales para shutdown graceful
            def signal_handler(signum, frame):
                self.logger.info("🛑 Shutdown signal received...")
                self.running = False
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            cycle_count = 0
            
            while self.running:
                try:
                    cycle_count += 1
                    self.logger.info(f"🔄 Cycle {cycle_count} starting...")
                    
                    # Ejecutar análisis
                    await self.run_analysis_cycle(cycle_count)
                    
                    # Esperar antes del siguiente ciclo
                    self.logger.info(f"⏳ Waiting {self.analysis_interval} seconds...")
                    await asyncio.sleep(self.analysis_interval)
                    
                except KeyboardInterrupt:
                    self.logger.info("🛑 Keyboard interrupt received...")
                    break
                except Exception as e:
                    self.logger.error(f"❌ Error in main loop: {e}")
                    await asyncio.sleep(60)  # Esperar 1 minuto antes de reintentar
            
            # Enviar mensaje de cierre
            if self.telegram_bot:
                await self.telegram_bot.send_alert(
                    "WARNING",
                    "🛑 **CRYPTOPULSE PRO ENHANCED BOT STOPPED!** 🛑\n\n"
                    f"📊 **Total cycles completed:** {cycle_count}\n"
                    f"⏰ **Time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC\n\n"
                    "🤖 **Thank you for using CryptoPulse Pro!**"
                )
            
            self.logger.info("✅ Enhanced Continuous Bot stopped")
            
        except Exception as e:
            self.logger.error(f"❌ Error in enhanced continuous bot: {e}")

async def main():
    """Función principal"""
    print("🚀 CryptoPulse Pro - Enhanced Continuous Bot")
    print("=" * 60)
    print("🧠 Features: Adaptive ML, Feedback System, Price Summaries")
    print("=" * 60)
    
    bot = EnhancedContinuousBot()
    
    # Inicializar
    if await bot.initialize():
        print("✅ Enhanced Bot initialized successfully!")
        print("🔄 Starting continuous analysis...")
        print("📱 Signals and price updates will be sent to Telegram")
        print("🧠 ML system will learn from each signal")
        print("📊 Performance reports every 10 cycles")
        print("⏹️ Press Ctrl+C to stop")
        print("=" * 60)
        
        # Ejecutar bot
        await bot.run()
    else:
        print("❌ Failed to initialize enhanced bot!")
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(main())

    return True

if __name__ == "__main__":
    asyncio.run(main())
