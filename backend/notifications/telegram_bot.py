#!/usr/bin/env python3
"""
Telegram Bot - CryptoPulse Pro
Bot de notificaciones para señales de trading
"""

import asyncio
import aiohttp
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from utils.logger import get_logger, log_execution_time, log_function_call

@dataclass
class TelegramMessage:
    """Estructura de mensaje de Telegram"""
    chat_id: str
    text: str
    parse_mode: str = "Markdown"
    reply_markup: Optional[Dict] = None

class TelegramBot:
    """
    Bot de Telegram para notificaciones
    - Envía señales de trading
    - Notificaciones de estado
    - Alertas de sistema
    """
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_url = f"https://api.telegram.org/bot{bot_token}"
        self.logger = get_logger("TelegramBot")
        
        self.logger.info("📱 TelegramBot initialized")
    
    async def initialize(self):
        """Inicializar el bot de Telegram"""
        try:
            self.logger.info("🔧 Initializing Telegram Bot...")
            
            # Verificar conexión
            await self._check_connection()
            
            self.logger.info("✅ Telegram Bot initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Telegram Bot: {e}")
            return False
    
    @log_execution_time
    async def send_signal(self, signal: Dict[str, Any]) -> bool:
        """Enviar señal de trading"""
        try:
            self.logger.info(f"📤 Sending signal: {signal.get('signal_type', 'UNKNOWN')}")
            
            # Formatear mensaje
            message = await self._format_signal_message(signal)
            
            # Enviar mensaje
            success = await self._send_message(message)
            
            if success:
                self.logger.info("✅ Signal sent successfully")
            else:
                self.logger.error("❌ Failed to send signal")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending signal: {e}")
            return False
    
    @log_execution_time
    async def send_alert(self, alert_type: str, message: str) -> bool:
        """Enviar alerta del sistema"""
        try:
            self.logger.info(f"🚨 Sending alert: {alert_type}")
            
            # Formatear alerta
            formatted_message = await self._format_alert_message(alert_type, message)
            
            # Enviar mensaje
            success = await self._send_message(formatted_message)
            
            if success:
                self.logger.info("✅ Alert sent successfully")
            else:
                self.logger.error("❌ Failed to send alert")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending alert: {e}")
            return False
    
    @log_execution_time
    async def send_status_update(self, status: Dict[str, Any]) -> bool:
        """Enviar actualización de estado"""
        try:
            self.logger.info("📊 Sending status update")
            
            # Formatear estado
            message = await self._format_status_message(status)
            
            # Enviar mensaje
            success = await self._send_message(message)
            
            if success:
                self.logger.info("✅ Status update sent successfully")
            else:
                self.logger.error("❌ Failed to send status update")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending status update: {e}")
            return False
    
    @log_function_call
    async def _format_signal_message(self, signal: Dict[str, Any]) -> TelegramMessage:
        """Formatear mensaje de señal"""
        try:
            signal_type = signal.get('signal_type', 'HOLD')
            confidence = signal.get('confidence_score', signal.get('confidence', 0.0))
            coherence_score = signal.get('coherence_score', 0.0)
            timestamp = signal.get('timestamp', datetime.utcnow().isoformat())
            
            # Emoji según tipo de señal
            emoji_map = {
                'STRONG_BUY': '🚀🚀🚀',
                'BUY': '🚀🚀',
                'WEAK_BUY': '🚀',
                'HOLD': '⏸️',
                'WEAK_SELL': '📉',
                'SELL': '📉📉',
                'STRONG_SELL': '📉📉📉'
            }
            
            emoji = emoji_map.get(signal_type, '❓')
            
            # Obtener datos adicionales
            symbol = signal.get('symbol', 'UNKNOWN')
            entry_price = signal.get('entry_price', 0.0)
            stop_loss = signal.get('stop_loss', 0.0)
            take_profit_1 = signal.get('take_profit_1', 0.0)
            take_profit_2 = signal.get('take_profit_2', 0.0)
            take_profit_3 = signal.get('take_profit_3', 0.0)
            reasoning = signal.get('reasoning', 'Análisis completo del mercado')
            
            # Calcular métricas adicionales
            risk_reward_ratio = ((take_profit_3 - entry_price) / (entry_price - stop_loss)) if entry_price > stop_loss else 0
            max_risk_percent = ((entry_price - stop_loss) / entry_price) * 100 if entry_price > stop_loss else 0
            potential_gain_percent = ((take_profit_3 - entry_price) / entry_price) * 100
            
            # Determinar nivel de riesgo
            if max_risk_percent <= 2:
                risk_level = "🟢 BAJO"
            elif max_risk_percent <= 5:
                risk_level = "🟡 MEDIO"
            else:
                risk_level = "🔴 ALTO"
            
            # Determinar calidad de la señal
            if confidence >= 85 and coherence_score >= 80:
                signal_quality = "🏆 EXCELENTE"
            elif confidence >= 70 and coherence_score >= 60:
                signal_quality = "⭐ BUENA"
            elif confidence >= 60:
                signal_quality = "✅ ACEPTABLE"
            else:
                signal_quality = "⚠️ CAUTELA"
            
            # Formatear mensaje mejorado
            message_text = f"""🚨 **NUEVA SEÑAL DE TRADING** 🚨

📈 **SÍMBOLO:** `{symbol}`
📊 **TIPO DE SEÑAL:** `{signal_type}`
💰 **PRECIO DE ENTRADA:** `${entry_price:,.2f}`

🎯 **NIVELES DE TRADING:**
🛡️ **Stop Loss:** `${stop_loss:,.2f}` (-{max_risk_percent:.1f}%)
🚀 **Take Profit 1:** `${take_profit_1:,.2f}` (+{((take_profit_1/entry_price-1)*100):.1f}%)
🚀 **Take Profit 2:** `${take_profit_2:,.2f}` (+{((take_profit_2/entry_price-1)*100):.1f}%)
🚀 **Take Profit 3:** `${take_profit_3:,.2f}` (+{potential_gain_percent:.1f}%)

📊 **MÉTRICAS DE CALIDAD:**
⭐ **Confianza:** `{confidence:.1f}%`
🧠 **Coherencia:** `{coherence_score:.1f}%`
🎯 **Calidad:** {signal_quality}
⚖️ **Riesgo:** {risk_level}
📈 **R/R Ratio:** `1:{risk_reward_ratio:.1f}`

💡 **ANÁLISIS DETALLADO:**
{self.format_reasoning(reasoning)}

📋 **RECOMENDACIONES:**
• Posición sugerida: 1-3% del portfolio
• Monitorear volumen durante las próximas 4 horas
• Considerar escalar entrada si rompe resistencia clave
• Usar gestión de riesgo estricta

⏰ **Timestamp:** `{datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')}`

🤖 **Powered by CryptoPulse Pro**"""
            
            # Agregar análisis detallado
            detailed_analysis = signal.get('detailed_analysis', {})
            if detailed_analysis:
                strengths = detailed_analysis.get('strengths', [])
                if strengths:
                    message_text += f"\n**Strengths:**\n"
                    for strength in strengths[:3]:  # Máximo 3
                        message_text += f"• {strength}\n"
                
                opportunities = detailed_analysis.get('opportunities', [])
                if opportunities:
                    message_text += f"\n**Opportunities:**\n"
                    for opportunity in opportunities[:3]:  # Máximo 3
                        message_text += f"• {opportunity}\n"
                
                threats = detailed_analysis.get('threats', [])
                if threats:
                    message_text += f"\n**Threats:**\n"
                    for threat in threats[:3]:  # Máximo 3
                        message_text += f"• {threat}\n"
            
            # Agregar recomendaciones
            recommendations = signal.get('recommendations', [])
            if recommendations:
                message_text += f"\n**Recommendations:**\n"
                for rec in recommendations[:3]:  # Máximo 3
                    message_text += f"• {rec}\n"
            
            # Agregar breakdown de fuentes
            source_breakdown = signal.get('source_breakdown', {})
            if source_breakdown:
                message_text += f"\n**Source Analysis:**\n"
                for source, data in source_breakdown.items():
                    confidence = data.get('confidence', 0.0)
                    reliability = data.get('reliability', 0.0)
                    message_text += f"• {source.title()}: {confidence:.1%} conf, {reliability:.1%} rel\n"
            
            return TelegramMessage(
                chat_id=self.chat_id,
                text=message_text,
                parse_mode="Markdown"
            )
            
        except Exception as e:
            self.logger.error(f"Error formatting signal message: {e}")
            return TelegramMessage(
                chat_id=self.chat_id,
                text=f"Error formatting signal: {str(e)}",
                parse_mode="Markdown"
            )
    
    @log_function_call
    async def _format_alert_message(self, alert_type: str, message: str) -> TelegramMessage:
        """Formatear mensaje de alerta"""
        try:
            # Emoji según tipo de alerta
            emoji_map = {
                'ERROR': '❌',
                'WARNING': '⚠️',
                'INFO': 'ℹ️',
                'SUCCESS': '✅',
                'CRITICAL': '🚨'
            }
            
            emoji = emoji_map.get(alert_type, '📢')
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            
            message_text = f"""
{emoji} **CRYPTOPULSE PRO ALERT** {emoji}

**Type:** {alert_type}
**Time:** {timestamp}

**Message:**
{message}
"""
            
            return TelegramMessage(
                chat_id=self.chat_id,
                text=message_text,
                parse_mode="Markdown"
            )
            
        except Exception as e:
            self.logger.error(f"Error formatting alert message: {e}")
            return TelegramMessage(
                chat_id=self.chat_id,
                text=f"Error formatting alert: {str(e)}",
                parse_mode="Markdown"
            )
    
    @log_function_call
    async def _format_status_message(self, status: Dict[str, Any]) -> TelegramMessage:
        """Formatear mensaje de estado"""
        try:
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            
            message_text = f"""
📊 **CRYPTOPULSE PRO STATUS** 📊

**Time:** {timestamp}
**Status:** {status.get('status', 'UNKNOWN')}

**Metrics:**
"""
            
            # Agregar métricas
            metrics = status.get('metrics', {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    message_text += f"• {key}: {value}\n"
                else:
                    message_text += f"• {key}: {value}\n"
            
            # Agregar salud del sistema
            health = status.get('health', {})
            if health:
                message_text += f"\n**System Health:**\n"
                for component, health_status in health.items():
                    status_emoji = "✅" if health_status.get('healthy', False) else "❌"
                    message_text += f"• {component}: {status_emoji} {health_status.get('message', 'Unknown')}\n"
            
            return TelegramMessage(
                chat_id=self.chat_id,
                text=message_text,
                parse_mode="Markdown"
            )
            
        except Exception as e:
            self.logger.error(f"Error formatting status message: {e}")
            return TelegramMessage(
                chat_id=self.chat_id,
                text=f"Error formatting status: {str(e)}",
                parse_mode="Markdown"
            )
    
    @log_function_call
    async def _send_message(self, message: TelegramMessage) -> bool:
        """Enviar mensaje a Telegram"""
        try:
            url = f"{self.api_url}/sendMessage"
            
            payload = {
                'chat_id': message.chat_id,
                'text': message.text,
                'parse_mode': message.parse_mode
            }
            
            if message.reply_markup:
                payload['reply_markup'] = json.dumps(message.reply_markup)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('ok'):
                            return True
                        else:
                            self.logger.error(f"Telegram API error: {data}")
                            return False
                    else:
                        error_text = await response.text()
                        self.logger.error(f"Telegram API error: {response.status} - {error_text}")
                        return False
            
        except Exception as e:
            self.logger.error(f"Error sending message to Telegram: {e}")
            return False
    
    @log_function_call
    async def _check_connection(self):
        """Verificar conexión con Telegram"""
        try:
            url = f"{self.api_url}/getMe"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('ok'):
                            bot_info = data.get('result', {})
                            self.logger.info(f"✅ Connected to Telegram Bot: {bot_info.get('first_name', 'Unknown')}")
                        else:
                            self.logger.error("❌ Telegram Bot connection failed")
                    else:
                        self.logger.error(f"❌ Telegram API error: {response.status}")
            
        except Exception as e:
            self.logger.error(f"Error checking Telegram connection: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del bot de Telegram"""
        try:
            # Verificar conexión
            url = f"{self.api_url}/getMe"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('ok'):
                            return {
                                'healthy': True,
                                'message': 'Telegram Bot healthy',
                                'bot_info': data.get('result', {}),
                                'timestamp': datetime.utcnow().isoformat()
                            }
                        else:
                            return {
                                'healthy': False,
                                'message': 'Telegram Bot API error',
                                'timestamp': datetime.utcnow().isoformat()
                            }
                    else:
                        return {
                            'healthy': False,
                            'message': f'Telegram API error: {response.status}',
                            'timestamp': datetime.utcnow().isoformat()
                        }
            
        except Exception as e:
            return {
                'healthy': False,
                'message': f'Telegram Bot unhealthy: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }
    
    def format_reasoning(self, reasoning: str) -> str:
        """Formatear el razonamiento de manera organizada"""
        try:
            # Dividir por puntos y organizar
            points = reasoning.split(' • ')
            
            # Categorizar los puntos
            technical_points = []
            ml_points = []
            sentiment_points = []
            onchain_points = []
            other_points = []
            
            for point in points:
                point = point.strip()
                if not point:
                    continue
                    
                if any(keyword in point.lower() for keyword in ['rsi', 'volumen', 'tendencia', 'bollinger', 'macd', 'sma']):
                    technical_points.append(f"• {point}")
                elif any(keyword in point.lower() for keyword in ['ml', 'predice', 'patrón', 'modelo']):
                    ml_points.append(f"• {point}")
                elif any(keyword in point.lower() for keyword in ['sentimiento', 'social', 'twitter', 'reddit', 'youtube']):
                    sentiment_points.append(f"• {point}")
                elif any(keyword in point.lower() for keyword in ['ballenas', 'funding', 'exchanges', 'on-chain', 'acumulación']):
                    onchain_points.append(f"• {point}")
                else:
                    other_points.append(f"• {point}")
            
            # Construir mensaje organizado
            formatted = ""
            
            if technical_points:
                formatted += "🔧 **Análisis Técnico:**\n" + "\n".join(technical_points[:3]) + "\n\n"
            
            if ml_points:
                formatted += "🤖 **Machine Learning:**\n" + "\n".join(ml_points[:2]) + "\n\n"
            
            if sentiment_points:
                formatted += "📱 **Sentimiento Social:**\n" + "\n".join(sentiment_points[:2]) + "\n\n"
            
            if onchain_points:
                formatted += "⛓️ **Datos On-Chain:**\n" + "\n".join(onchain_points[:2]) + "\n\n"
            
            if other_points:
                formatted += "📊 **Otros Factores:**\n" + "\n".join(other_points[:2]) + "\n\n"
            
            return formatted.strip()
            
        except Exception as e:
            self.logger.error(f"Error formatting reasoning: {e}")
            return f"• {reasoning}"
