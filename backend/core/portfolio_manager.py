#!/usr/bin/env python3
"""
Portfolio Manager - CryptoPulse Pro
Gestión inteligente de posiciones y portfolio
"""

import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger, log_execution_time, log_function_call

class PositionStatus(str, Enum):
    """Estados de posición"""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL = "PARTIAL"

class PositionSide(str, Enum):
    """Lado de la posición"""
    LONG = "LONG"
    SHORT = "SHORT"

@dataclass
class Position:
    """Estructura de una posición"""
    id: str
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: float
    current_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    status: PositionStatus
    pnl: float
    pnl_percentage: float
    timestamp: datetime
    strategy: str
    confidence: float

@dataclass
class PortfolioMetrics:
    """Métricas del portfolio"""
    total_value: float
    total_pnl: float
    total_pnl_percentage: float
    active_positions: int
    closed_positions: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    last_update: datetime

class PortfolioManager:
    """
    Gestor inteligente de portfolio
    - Gestión de posiciones
    - Cálculo de métricas
    - Gestión de riesgo
    - Optimización de portfolio
    """
    
    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.logger = get_logger("PortfolioManager")
        
        # Configuración de riesgo
        self.max_position_size = 0.1  # 10% del portfolio por posición
        self.max_total_exposure = 0.8  # 80% del portfolio máximo
        self.stop_loss_percentage = 0.02  # 2% stop loss por defecto
        
        self.logger.info("💼 PortfolioManager initialized")
    
    async def initialize(self):
        """Inicializar el gestor de portfolio"""
        try:
            self.logger.info("🔧 Initializing Portfolio Manager...")
            
            # Cargar posiciones existentes
            await self._load_positions()
            
            # Calcular métricas iniciales
            await self._calculate_metrics()
            
            self.logger.info("✅ Portfolio Manager initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing Portfolio Manager: {e}")
    
    @log_execution_time
    async def open_position(self, signal: Dict[str, Any], current_price: float) -> Optional[str]:
        """Abrir nueva posición"""
        try:
            self.logger.info(f"📈 Opening position for {signal.get('symbol', 'UNKNOWN')}")
            
            # Validar señal
            if not await self._validate_signal(signal):
                self.logger.warning("Signal validation failed")
                return None
            
            # Calcular tamaño de posición
            position_size = await self._calculate_position_size(signal, current_price)
            if position_size <= 0:
                self.logger.warning("Position size too small")
                return None
            
            # Crear posición
            position_id = f"{signal.get('symbol', 'UNKNOWN')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            position = Position(
                id=position_id,
                symbol=signal.get('symbol', 'UNKNOWN'),
                side=PositionSide.LONG if signal.get('signal_type', '').endswith('BUY') else PositionSide.SHORT,
                entry_price=current_price,
                quantity=position_size,
                current_price=current_price,
                stop_loss=signal.get('stop_loss', current_price * (1 - self.stop_loss_percentage)),
                take_profit_1=signal.get('take_profit_1', current_price * 1.02),
                take_profit_2=signal.get('take_profit_2', current_price * 1.04),
                take_profit_3=signal.get('take_profit_3', current_price * 1.06),
                status=PositionStatus.OPEN,
                pnl=0.0,
                pnl_percentage=0.0,
                timestamp=datetime.utcnow(),
                strategy=signal.get('strategy', 'UNKNOWN'),
                confidence=signal.get('confidence', 0.0)
            )
            
            # Agregar posición
            self.positions[position_id] = position
            
            # Actualizar balance
            self.current_balance -= position_size * current_price
            
            # Calcular métricas
            await self._calculate_metrics()
            
            self.logger.info(f"✅ Position opened: {position_id}")
            return position_id
            
        except Exception as e:
            self.logger.error(f"Error opening position: {e}")
            return None
    
    @log_execution_time
    async def close_position(self, position_id: str, current_price: float, reason: str = "MANUAL") -> bool:
        """Cerrar posición"""
        try:
            self.logger.info(f"📉 Closing position: {position_id}")
            
            if position_id not in self.positions:
                self.logger.warning(f"Position not found: {position_id}")
                return False
            
            position = self.positions[position_id]
            
            # Calcular PnL
            if position.side == PositionSide.LONG:
                pnl = (current_price - position.entry_price) * position.quantity
            else:
                pnl = (position.entry_price - current_price) * position.quantity
            
            pnl_percentage = (pnl / (position.entry_price * position.quantity)) * 100
            
            # Actualizar posición
            position.current_price = current_price
            position.pnl = pnl
            position.pnl_percentage = pnl_percentage
            position.status = PositionStatus.CLOSED
            
            # Mover a posiciones cerradas
            self.closed_positions.append(position)
            del self.positions[position_id]
            
            # Actualizar balance
            self.current_balance += pnl + (position.quantity * current_price)
            
            # Calcular métricas
            await self._calculate_metrics()
            
            self.logger.info(f"✅ Position closed: {position_id} - PnL: {pnl:.2f} ({pnl_percentage:.2f}%)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False
    
    @log_execution_time
    async def update_positions(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """Actualizar posiciones con precios actuales"""
        try:
            updates = []
            
            for position_id, position in self.positions.items():
                if position.symbol in current_prices:
                    old_price = position.current_price
                    new_price = current_prices[position.symbol]
                    
                    # Actualizar precio
                    position.current_price = new_price
                    
                    # Calcular PnL
                    if position.side == PositionSide.LONG:
                        pnl = (new_price - position.entry_price) * position.quantity
                    else:
                        pnl = (position.entry_price - new_price) * position.quantity
                    
                    pnl_percentage = (pnl / (position.entry_price * position.quantity)) * 100
                    
                    position.pnl = pnl
                    position.pnl_percentage = pnl_percentage
                    
                    # Verificar stop loss y take profit
                    should_close = await self._check_exit_conditions(position)
                    
                    if should_close:
                        await self.close_position(position_id, new_price, "AUTO")
                        updates.append({
                            'position_id': position_id,
                            'action': 'CLOSED',
                            'reason': 'EXIT_CONDITION',
                            'pnl': pnl,
                            'pnl_percentage': pnl_percentage
                        })
                    else:
                        updates.append({
                            'position_id': position_id,
                            'action': 'UPDATED',
                            'old_price': old_price,
                            'new_price': new_price,
                            'pnl': pnl,
                            'pnl_percentage': pnl_percentage
                        })
            
            # Calcular métricas
            await self._calculate_metrics()
            
            return updates
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
            return []
    
    @log_function_call
    async def _validate_signal(self, signal: Dict[str, Any]) -> bool:
        """Validar señal antes de abrir posición"""
        try:
            # Verificar confianza mínima
            confidence = signal.get('confidence', 0.0)
            if confidence < 0.6:
                self.logger.warning(f"Signal confidence too low: {confidence}")
                return False
            
            # Verificar coherencia
            coherence_score = signal.get('coherence_score', 0.0)
            if coherence_score < 0.5:
                self.logger.warning(f"Signal coherence too low: {coherence_score}")
                return False
            
            # Verificar tipo de señal
            signal_type = signal.get('signal_type', 'HOLD')
            if signal_type == 'HOLD':
                self.logger.warning("Signal type is HOLD")
                return False
            
            # Verificar exposición máxima
            current_exposure = await self._calculate_current_exposure()
            if current_exposure >= self.max_total_exposure:
                self.logger.warning(f"Maximum exposure reached: {current_exposure:.2%}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False
    
    @log_function_call
    async def _calculate_position_size(self, signal: Dict[str, Any], current_price: float) -> float:
        """Calcular tamaño de posición"""
        try:
            # Tamaño base basado en confianza
            confidence = signal.get('confidence', 0.0)
            base_size = self.current_balance * self.max_position_size
            
            # Ajustar por confianza
            confidence_multiplier = min(1.0, confidence)
            adjusted_size = base_size * confidence_multiplier
            
            # Ajustar por coherencia
            coherence_score = signal.get('coherence_score', 0.0)
            coherence_multiplier = min(1.0, coherence_score)
            final_size = adjusted_size * coherence_multiplier
            
            # Convertir a cantidad
            quantity = final_size / current_price
            
            return quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    @log_function_call
    async def _calculate_current_exposure(self) -> float:
        """Calcular exposición actual del portfolio"""
        try:
            total_exposure = 0.0
            
            for position in self.positions.values():
                position_value = position.quantity * position.current_price
                total_exposure += position_value
            
            return total_exposure / self.current_balance if self.current_balance > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating current exposure: {e}")
            return 0.0
    
    @log_function_call
    async def _check_exit_conditions(self, position: Position) -> bool:
        """Verificar condiciones de salida"""
        try:
            current_price = position.current_price
            
            # Verificar stop loss
            if position.side == PositionSide.LONG:
                if current_price <= position.stop_loss:
                    self.logger.info(f"Stop loss triggered for {position.id}")
                    return True
                
                # Verificar take profit
                if current_price >= position.take_profit_3:
                    self.logger.info(f"Take profit 3 triggered for {position.id}")
                    return True
            else:
                if current_price >= position.stop_loss:
                    self.logger.info(f"Stop loss triggered for {position.id}")
                    return True
                
                # Verificar take profit
                if current_price <= position.take_profit_3:
                    self.logger.info(f"Take profit 3 triggered for {position.id}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return False
    
    @log_function_call
    async def _calculate_metrics(self):
        """Calcular métricas del portfolio"""
        try:
            # Calcular PnL total
            total_pnl = 0.0
            for position in self.positions.values():
                total_pnl += position.pnl
            
            for position in self.closed_positions:
                total_pnl += position.pnl
            
            # Calcular valor total
            total_value = self.current_balance + total_pnl
            
            # Calcular win rate
            winning_positions = [p for p in self.closed_positions if p.pnl > 0]
            win_rate = len(winning_positions) / len(self.closed_positions) if self.closed_positions else 0.0
            
            # Calcular drawdown máximo
            max_drawdown = await self._calculate_max_drawdown()
            
            # Calcular Sharpe ratio
            sharpe_ratio = await self._calculate_sharpe_ratio()
            
            # Crear métricas
            self.metrics = PortfolioMetrics(
                total_value=total_value,
                total_pnl=total_pnl,
                total_pnl_percentage=(total_pnl / self.initial_balance) * 100,
                active_positions=len(self.positions),
                closed_positions=len(self.closed_positions),
                win_rate=win_rate,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                last_update=datetime.utcnow()
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
    
    @log_function_call
    async def _calculate_max_drawdown(self) -> float:
        """Calcular drawdown máximo"""
        try:
            if not self.closed_positions:
                return 0.0
            
            # Simular balance histórico
            balance_history = [self.initial_balance]
            current_balance = self.initial_balance
            
            for position in sorted(self.closed_positions, key=lambda x: x.timestamp):
                current_balance += position.pnl
                balance_history.append(current_balance)
            
            # Calcular drawdown
            peak = balance_history[0]
            max_dd = 0.0
            
            for balance in balance_history:
                if balance > peak:
                    peak = balance
                else:
                    dd = (peak - balance) / peak
                    max_dd = max(max_dd, dd)
            
            return max_dd
            
        except Exception as e:
            self.logger.error(f"Error calculating max drawdown: {e}")
            return 0.0
    
    @log_function_call
    async def _calculate_sharpe_ratio(self) -> float:
        """Calcular Sharpe ratio"""
        try:
            if len(self.closed_positions) < 2:
                return 0.0
            
            # Calcular returns
            returns = []
            for position in self.closed_positions:
                return_pct = position.pnl_percentage / 100
                returns.append(return_pct)
            
            if not returns:
                return 0.0
            
            # Calcular Sharpe ratio
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            if std_return == 0:
                return 0.0
            
            sharpe_ratio = mean_return / std_return
            return sharpe_ratio
            
        except Exception as e:
            self.logger.error(f"Error calculating Sharpe ratio: {e}")
            return 0.0
    
    @log_function_call
    async def _load_positions(self):
        """Cargar posiciones existentes"""
        try:
            # Por ahora, no hay persistencia
            # En el futuro, cargar desde base de datos
            self.logger.info("No existing positions to load")
            
        except Exception as e:
            self.logger.error(f"Error loading positions: {e}")
    
    async def get_portfolio_summary(self) -> Dict[str, Any]:
        """Obtener resumen del portfolio"""
        try:
            await self._calculate_metrics()
            
            return {
                'initial_balance': self.initial_balance,
                'current_balance': self.current_balance,
                'total_value': self.metrics.total_value,
                'total_pnl': self.metrics.total_pnl,
                'total_pnl_percentage': self.metrics.total_pnl_percentage,
                'active_positions': self.metrics.active_positions,
                'closed_positions': self.metrics.closed_positions,
                'win_rate': self.metrics.win_rate,
                'max_drawdown': self.metrics.max_drawdown,
                'sharpe_ratio': self.metrics.sharpe_ratio,
                'current_exposure': await self._calculate_current_exposure(),
                'last_update': self.metrics.last_update.isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    async def get_active_positions(self) -> List[Dict[str, Any]]:
        """Obtener posiciones activas"""
        try:
            positions = []
            
            for position in self.positions.values():
                positions.append({
                    'id': position.id,
                    'symbol': position.symbol,
                    'side': position.side.value,
                    'entry_price': position.entry_price,
                    'current_price': position.current_price,
                    'quantity': position.quantity,
                    'pnl': position.pnl,
                    'pnl_percentage': position.pnl_percentage,
                    'stop_loss': position.stop_loss,
                    'take_profit_1': position.take_profit_1,
                    'take_profit_2': position.take_profit_2,
                    'take_profit_3': position.take_profit_3,
                    'strategy': position.strategy,
                    'confidence': position.confidence,
                    'timestamp': position.timestamp.isoformat()
                })
            
            return positions
            
        except Exception as e:
            self.logger.error(f"Error getting active positions: {e}")
            return []
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del Portfolio Manager"""
        try:
            return {
                'healthy': True,
                'message': 'Portfolio Manager healthy',
                'active_positions': len(self.positions),
                'closed_positions': len(self.closed_positions),
                'current_balance': self.current_balance,
                'total_value': self.metrics.total_value if hasattr(self, 'metrics') else 0.0,
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f'Portfolio Manager unhealthy: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }
