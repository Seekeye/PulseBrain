#!/usr/bin/env python3
"""
Feedback System - CryptoPulse Pro
Sistema de retroalimentación para mejorar señales continuamente
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque
import os

from utils.logger import get_logger, log_execution_time, log_function_call

class FeedbackSystem:
    """
    Sistema que:
    - Rastrea el rendimiento de cada señal
    - Calcula métricas de éxito/fallo
    - Ajusta parámetros del sistema
    - Proporciona insights de mejora
    """
    
    def __init__(self):
        self.logger = get_logger("FeedbackSystem")
        
        # Almacenamiento de señales y resultados
        self.signal_tracking = {}  # {signal_id: signal_data}
        self.signal_outcomes = {}  # {signal_id: outcome_data}
        
        # Métricas de rendimiento
        self.performance_metrics = {
            'total_signals': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'win_rate': 0.0,
            'average_pnl': 0.0,
            'best_signal': None,
            'worst_signal': None
        }
        
        # Historial de rendimiento
        self.performance_history = deque(maxlen=1000)
        
        # Configuración de evaluación
        self.evaluation_config = {
            'evaluation_period_hours': 24,  # Evaluar señales después de 24h
            'success_threshold': 0.02,      # 2% de ganancia mínima para éxito
            'failure_threshold': -0.01,     # -1% de pérdida para fallo
            'min_confidence_for_tracking': 60  # Solo rastrear señales con >60% confianza
        }
        
        self.logger.info("📊 FeedbackSystem initialized")
    
    async def initialize(self):
        """Inicializar el sistema de retroalimentación"""
        try:
            self.logger.info("🔧 Initializing Feedback System...")
            
            # Cargar datos históricos si existen
            await self._load_historical_data()
            
            # Calcular métricas actuales
            await self._calculate_current_metrics()
            
            self.logger.info("✅ Feedback System initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing Feedback System: {e}")
    
    async def _load_historical_data(self):
        """Cargar datos históricos de rendimiento"""
        try:
            # En un sistema real, esto cargaría desde la base de datos
            self.logger.info("📊 Loading historical performance data...")
            
            # Simular algunos datos históricos
            for i in range(50):
                signal_id = f"signal_{i}_{datetime.now().strftime('%Y%m%d')}"
                self.performance_history.append({
                    'signal_id': signal_id,
                    'timestamp': (datetime.now() - timedelta(hours=i*2)).isoformat(),
                    'pnl_percentage': np.random.normal(0.01, 0.05),  # 1% promedio, 5% std
                    'success': np.random.random() > 0.3,  # 70% éxito
                    'confidence': np.random.uniform(60, 95)
                })
            
            self.logger.info(f"✅ Loaded {len(self.performance_history)} historical records")
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
    
    @log_execution_time
    async def track_signal(self, signal_data: Dict[str, Any]) -> str:
        """Iniciar seguimiento de una señal"""
        try:
            signal_id = f"signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{signal_data.get('symbol', 'UNKNOWN')}"
            
            # Solo rastrear señales con suficiente confianza
            confidence = signal_data.get('confidence_score', 0)
            if confidence < self.evaluation_config['min_confidence_for_tracking']:
                self.logger.info(f"Signal {signal_id} not tracked (low confidence: {confidence}%)")
                return signal_id
            
            # Almacenar datos de la señal
            self.signal_tracking[signal_id] = {
                'signal_data': signal_data,
                'tracking_start': datetime.now().isoformat(),
                'status': 'TRACKING',
                'entry_price': signal_data.get('entry_price', 0),
                'stop_loss': signal_data.get('stop_loss', 0),
                'take_profit_1': signal_data.get('take_profit_1', 0),
                'signal_type': signal_data.get('signal_type', 'HOLD')
            }
            
            self.logger.info(f"📈 Tracking signal {signal_id} for {signal_data.get('symbol', 'UNKNOWN')}")
            return signal_id
            
        except Exception as e:
            self.logger.error(f"Error tracking signal: {e}")
            return f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    @log_execution_time
    async def update_signal_outcome(self, signal_id: str, current_price: float, additional_data: Dict[str, Any] = None):
        """Actualizar el resultado de una señal"""
        try:
            if signal_id not in self.signal_tracking:
                self.logger.warning(f"Signal {signal_id} not found in tracking")
                return
            
            signal_info = self.signal_tracking[signal_id]
            entry_price = signal_info['entry_price']
            signal_type = signal_info['signal_type']
            
            if entry_price <= 0:
                self.logger.warning(f"Invalid entry price for signal {signal_id}")
                return
            
            # Calcular PnL
            if signal_type == 'BUY':
                pnl_percentage = ((current_price - entry_price) / entry_price) * 100
            elif signal_type == 'SELL':
                pnl_percentage = ((entry_price - current_price) / entry_price) * 100
            else:
                pnl_percentage = 0
            
            # Determinar si es éxito o fallo
            success = pnl_percentage >= (self.evaluation_config['success_threshold'] * 100)
            failure = pnl_percentage <= (self.evaluation_config['failure_threshold'] * 100)
            
            # Actualizar resultado
            self.signal_outcomes[signal_id] = {
                'signal_id': signal_id,
                'current_price': current_price,
                'pnl_percentage': pnl_percentage,
                'success': success,
                'failure': failure,
                'status': 'SUCCESS' if success else ('FAILURE' if failure else 'TRACKING'),
                'update_time': datetime.now().isoformat(),
                'additional_data': additional_data or {}
            }
            
            # Si es éxito o fallo definitivo, finalizar seguimiento
            if success or failure:
                await self._finalize_signal(signal_id, success, pnl_percentage)
            
            self.logger.info(f"📊 Updated signal {signal_id}: {pnl_percentage:.2f}% PnL")
            
        except Exception as e:
            self.logger.error(f"Error updating signal outcome: {e}")
    
    async def _finalize_signal(self, signal_id: str, success: bool, pnl_percentage: float):
        """Finalizar el seguimiento de una señal"""
        try:
            signal_info = self.signal_tracking[signal_id]
            signal_info['status'] = 'COMPLETED'
            signal_info['final_pnl'] = pnl_percentage
            signal_info['final_success'] = success
            signal_info['completion_time'] = datetime.now().isoformat()
            
            # Agregar al historial de rendimiento
            self.performance_history.append({
                'signal_id': signal_id,
                'timestamp': signal_info['tracking_start'],
                'pnl_percentage': pnl_percentage,
                'success': success,
                'confidence': signal_info['signal_data'].get('confidence_score', 0),
                'symbol': signal_info['signal_data'].get('symbol', 'UNKNOWN')
            })
            
            # Actualizar métricas
            await self._calculate_current_metrics()
            
            self.logger.info(f"✅ Finalized signal {signal_id}: {'SUCCESS' if success else 'FAILURE'}")
            
        except Exception as e:
            self.logger.error(f"Error finalizing signal: {e}")
    
    @log_execution_time
    async def _calculate_current_metrics(self):
        """Calcular métricas actuales de rendimiento"""
        try:
            if not self.performance_history:
                return
            
            # Filtrar señales completadas
            completed_signals = [s for s in self.performance_history if 'success' in s]
            
            if not completed_signals:
                return
            
            # Calcular métricas básicas
            total_signals = len(completed_signals)
            successful_signals = sum(1 for s in completed_signals if s['success'])
            failed_signals = total_signals - successful_signals
            
            win_rate = (successful_signals / total_signals) * 100 if total_signals > 0 else 0
            average_pnl = np.mean([s['pnl_percentage'] for s in completed_signals])
            
            # Encontrar mejor y peor señal
            best_signal = max(completed_signals, key=lambda x: x['pnl_percentage'])
            worst_signal = min(completed_signals, key=lambda x: x['pnl_percentage'])
            
            # Actualizar métricas
            self.performance_metrics.update({
                'total_signals': total_signals,
                'successful_signals': successful_signals,
                'failed_signals': failed_signals,
                'win_rate': win_rate,
                'average_pnl': average_pnl,
                'best_signal': best_signal,
                'worst_signal': worst_signal
            })
            
            self.logger.info(f"📊 Updated metrics: {win_rate:.1f}% win rate, {average_pnl:.2f}% avg PnL")
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
    
    @log_execution_time
    async def get_performance_summary(self) -> Dict[str, Any]:
        """Obtener resumen de rendimiento"""
        try:
            # Calcular tendencias
            recent_signals = list(self.performance_history)[-10:]  # Últimas 10 señales
            recent_win_rate = 0
            recent_avg_pnl = 0
            
            if recent_signals:
                recent_successful = sum(1 for s in recent_signals if s.get('success', False))
                recent_win_rate = (recent_successful / len(recent_signals)) * 100
                recent_avg_pnl = np.mean([s.get('pnl_percentage', 0) for s in recent_signals])
            
            # Calcular tendencia
            trend = 'IMPROVING'
            if len(self.performance_history) >= 20:
                old_signals = list(self.performance_history)[-20:-10]
                old_win_rate = 0
                if old_signals:
                    old_successful = sum(1 for s in old_signals if s.get('success', False))
                    old_win_rate = (old_successful / len(old_signals)) * 100
                
                if recent_win_rate < old_win_rate - 5:
                    trend = 'DECLINING'
                elif recent_win_rate > old_win_rate + 5:
                    trend = 'IMPROVING'
                else:
                    trend = 'STABLE'
            
            return {
                'current_metrics': self.performance_metrics,
                'recent_performance': {
                    'win_rate': recent_win_rate,
                    'average_pnl': recent_avg_pnl,
                    'signals_count': len(recent_signals)
                },
                'trend': trend,
                'recommendations': await self._generate_recommendations(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {
                'current_metrics': self.performance_metrics,
                'recent_performance': {'win_rate': 0, 'average_pnl': 0, 'signals_count': 0},
                'trend': 'UNKNOWN',
                'recommendations': [],
                'timestamp': datetime.now().isoformat()
            }
    
    async def _generate_recommendations(self) -> List[str]:
        """Generar recomendaciones basadas en el rendimiento"""
        try:
            recommendations = []
            
            # Recomendación basada en win rate
            if self.performance_metrics['win_rate'] < 50:
                recommendations.append("Win rate bajo - considerar ajustar umbrales de confianza")
            elif self.performance_metrics['win_rate'] > 80:
                recommendations.append("Excelente win rate - considerar aumentar tamaño de posiciones")
            
            # Recomendación basada en PnL promedio
            if self.performance_metrics['average_pnl'] < 0:
                recommendations.append("PnL promedio negativo - revisar estrategias de entrada")
            elif self.performance_metrics['average_pnl'] > 5:
                recommendations.append("Excelente PnL promedio - estrategia funcionando bien")
            
            # Recomendación basada en tendencia
            recent_signals = list(self.performance_history)[-5:]
            if len(recent_signals) >= 5:
                recent_success_rate = sum(1 for s in recent_signals if s.get('success', False)) / len(recent_signals)
                if recent_success_rate < 0.3:
                    recommendations.append("Rendimiento reciente muy bajo - considerar pausar señales")
                elif recent_success_rate > 0.8:
                    recommendations.append("Rendimiento reciente excelente - mantener estrategia actual")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return ["Error generando recomendaciones"]
    
    @log_execution_time
    async def get_signal_analysis(self, signal_id: str) -> Dict[str, Any]:
        """Obtener análisis detallado de una señal específica"""
        try:
            if signal_id not in self.signal_tracking:
                return {'error': 'Signal not found'}
            
            signal_info = self.signal_tracking[signal_id]
            outcome = self.signal_outcomes.get(signal_id, {})
            
            analysis = {
                'signal_info': signal_info,
                'outcome': outcome,
                'analysis': {
                    'tracking_duration': self._calculate_tracking_duration(signal_info),
                    'price_movement': self._calculate_price_movement(signal_info, outcome),
                    'risk_reward_ratio': self._calculate_risk_reward_ratio(signal_info),
                    'performance_rating': self._calculate_performance_rating(signal_info, outcome)
                }
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing signal: {e}")
            return {'error': f'Analysis failed: {str(e)}'}
    
    def _calculate_tracking_duration(self, signal_info: Dict[str, Any]) -> str:
        """Calcular duración del seguimiento"""
        try:
            start_time = datetime.fromisoformat(signal_info['tracking_start'])
            end_time = datetime.fromisoformat(signal_info.get('completion_time', datetime.now().isoformat()))
            duration = end_time - start_time
            return f"{duration.total_seconds() / 3600:.1f} hours"
        except:
            return "Unknown"
    
    def _calculate_price_movement(self, signal_info: Dict[str, Any], outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Calcular movimiento de precio"""
        try:
            entry_price = signal_info['entry_price']
            current_price = outcome.get('current_price', entry_price)
            
            movement = ((current_price - entry_price) / entry_price) * 100
            
            return {
                'entry_price': entry_price,
                'current_price': current_price,
                'movement_percentage': movement,
                'direction': 'UP' if movement > 0 else 'DOWN'
            }
        except:
            return {'error': 'Could not calculate price movement'}
    
    def _calculate_risk_reward_ratio(self, signal_info: Dict[str, Any]) -> float:
        """Calcular ratio riesgo/recompensa"""
        try:
            entry_price = signal_info['entry_price']
            stop_loss = signal_info['stop_loss']
            take_profit = signal_info['take_profit_1']
            
            if entry_price <= 0 or stop_loss <= 0 or take_profit <= 0:
                return 0
            
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            
            return reward / risk if risk > 0 else 0
        except:
            return 0
    
    def _calculate_performance_rating(self, signal_info: Dict[str, Any], outcome: Dict[str, Any]) -> str:
        """Calcular rating de rendimiento"""
        try:
            pnl = outcome.get('pnl_percentage', 0)
            
            if pnl >= 5:
                return 'EXCELLENT'
            elif pnl >= 2:
                return 'GOOD'
            elif pnl >= 0:
                return 'AVERAGE'
            elif pnl >= -2:
                return 'POOR'
            else:
                return 'TERRIBLE'
        except:
            return 'UNKNOWN'

if __name__ == "__main__":
    async def test_feedback_system():
        feedback = FeedbackSystem()
        await feedback.initialize()
        
        # Simular una señal
        signal_data = {
            'symbol': 'BTCUSDT',
            'signal_type': 'BUY',
            'entry_price': 50000,
            'stop_loss': 48000,
            'take_profit_1': 52000,
            'confidence_score': 85
        }
        
        # Rastrear señal
        signal_id = await feedback.track_signal(signal_data)
        print(f"Tracking signal: {signal_id}")
        
        # Simular actualizaciones de precio
        prices = [50500, 51000, 51500, 52000]
        for price in prices:
            await feedback.update_signal_outcome(signal_id, price)
            await asyncio.sleep(0.1)
        
        # Obtener resumen
        summary = await feedback.get_performance_summary()
        print(f"Performance Summary: {json.dumps(summary, indent=2)}")
    
    asyncio.run(test_feedback_system())
