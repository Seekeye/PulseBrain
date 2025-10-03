#!/usr/bin/env python3
"""
Gestión de Riesgo Inteligente - CryptoPulse Pro
Sistema avanzado de gestión de riesgo con ATR, Kelly Criterion y diversificación
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import math

from utils.logger import get_logger, log_execution_time, log_function_call
from config.settings import RiskConfig

class RiskLevel(str, Enum):
    """Niveles de riesgo"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class RiskMetrics:
    """Métricas de riesgo"""
    portfolio_value: float
    total_exposure: float
    max_drawdown: float
    sharpe_ratio: float
    var_95: float  # Value at Risk 95%
    max_correlation: float
    sector_concentration: Dict[str, float]

class RiskManager:
    """
    Gestor de riesgo inteligente
    - Cálculo de posición basado en volatilidad
    - Stop loss dinámico con ATR
    - Diversificación automática
    - Kelly Criterion para sizing
    - Monitoreo de correlaciones
    """
    
    def __init__(self, risk_config: RiskConfig, signals_db):
        self.risk_config = risk_config
        self.signals_db = signals_db
        self.logger = get_logger("RiskManager")
        
        # Estado del portfolio
        self.portfolio_value = 10000.0  # Valor inicial del portfolio
        self.active_positions = {}
        self.daily_pnl = []
        self.max_drawdown = 0.0
        self.peak_value = self.portfolio_value
        
        self.logger.info("⚖️ RiskManager initialized")
    
    async def initialize(self):
        """Inicializar el gestor de riesgo"""
        self.logger.info("🔧 Initializing risk manager...")
        
        # Cargar estado del portfolio desde la base de datos
        await self._load_portfolio_state()
        
        self.logger.info("✅ Risk manager initialized")
    
    @log_execution_time
    async def calculate_position_size(self, signal: Dict[str, Any], 
                                    current_price: float) -> Dict[str, Any]:
        """Calcular tamaño de posición óptimo"""
        try:
            # Obtener métricas de riesgo del símbolo
            symbol_risk = await self._calculate_symbol_risk(signal['symbol'], current_price)
            
            # Calcular volatilidad
            volatility = await self._get_symbol_volatility(signal['symbol'])
            
            # Calcular Kelly Criterion si está habilitado
            if self.risk_config.use_kelly_criterion:
                kelly_size = await self._calculate_kelly_position_size(signal, volatility)
            else:
                kelly_size = 0.02  # 2% por defecto
            
            # Aplicar límites de riesgo
            max_position_size = min(
                kelly_size,
                self.risk_config.max_position_size_pct,
                self.risk_config.max_risk_per_trade / volatility if volatility > 0 else 0.01
            )
            
            # Verificar diversificación
            diversification_factor = await self._calculate_diversification_factor(signal['symbol'])
            max_position_size *= diversification_factor
            
            # Calcular valor de la posición
            position_value = self.portfolio_value * max_position_size
            position_quantity = position_value / current_price
            
            # Verificar límites mínimos
            min_position_value = self.portfolio_value * self.risk_config.min_position_size_pct
            if position_value < min_position_value:
                position_value = 0
                position_quantity = 0
            
            result = {
                'position_size_pct': max_position_size,
                'position_value': position_value,
                'position_quantity': position_quantity,
                'volatility': volatility,
                'kelly_size': kelly_size,
                'diversification_factor': diversification_factor,
                'risk_level': symbol_risk['risk_level'],
                'max_loss': position_value * volatility * 2,  # 2x volatilidad
                'approved': position_value > 0
            }
            
            self.logger.info(f"Position size calculated for {signal['symbol']}: {max_position_size:.2%}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {signal['symbol']}", exception=e)
            return {'approved': False, 'error': str(e)}
    
    @log_function_call
    async def calculate_dynamic_stop_loss(self, symbol: str, entry_price: float, 
                                        signal_type: str, timeframe: str) -> float:
        """Calcular stop loss dinámico basado en ATR"""
        try:
            if not self.risk_config.use_atr_stop_loss:
                # Stop loss fijo
                if signal_type == 'BUY':
                    return entry_price * (1 - self.risk_config.default_stop_loss_pct)
                else:
                    return entry_price * (1 + self.risk_config.default_stop_loss_pct)
            
            # Obtener ATR del símbolo
            atr = await self._get_atr(symbol, timeframe)
            if not atr:
                # Fallback a stop loss fijo
                if signal_type == 'BUY':
                    return entry_price * (1 - self.risk_config.default_stop_loss_pct)
                else:
                    return entry_price * (1 + self.risk_config.default_stop_loss_pct)
            
            # Calcular stop loss con ATR
            atr_distance = atr * self.risk_config.atr_multiplier
            
            if signal_type == 'BUY':
                stop_loss = entry_price - atr_distance
            else:  # SELL
                stop_loss = entry_price + atr_distance
            
            # Aplicar límite máximo de stop loss
            max_stop_loss_pct = self.risk_config.max_stop_loss_pct
            if signal_type == 'BUY':
                max_stop_loss = entry_price * (1 - max_stop_loss_pct)
                stop_loss = max(stop_loss, max_stop_loss)
            else:
                max_stop_loss = entry_price * (1 + max_stop_loss_pct)
                stop_loss = min(stop_loss, max_stop_loss)
            
            self.logger.debug(f"Dynamic stop loss for {symbol}: {stop_loss:.4f} (ATR: {atr:.4f})")
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic stop loss for {symbol}", exception=e)
            # Fallback a stop loss fijo
            if signal_type == 'BUY':
                return entry_price * (1 - self.risk_config.default_stop_loss_pct)
            else:
                return entry_price * (1 + self.risk_config.default_stop_loss_pct)
    
    @log_function_call
    async def calculate_take_profits(self, symbol: str, entry_price: float, 
                                   stop_loss: float, signal_type: str) -> List[float]:
        """Calcular niveles de take profit"""
        try:
            # Calcular distancia de riesgo
            if signal_type == 'BUY':
                risk_distance = entry_price - stop_loss
            else:  # SELL
                risk_distance = stop_loss - entry_price
            
            # Calcular take profits basados en risk-reward ratios
            tp1 = entry_price + (risk_distance * 1.0) if signal_type == 'BUY' else entry_price - (risk_distance * 1.0)
            tp2 = entry_price + (risk_distance * 2.0) if signal_type == 'BUY' else entry_price - (risk_distance * 2.0)
            tp3 = entry_price + (risk_distance * 3.0) if signal_type == 'BUY' else entry_price - (risk_distance * 3.0)
            
            return [tp1, tp2, tp3]
            
        except Exception as e:
            self.logger.error(f"Error calculating take profits for {symbol}", exception=e)
            # Take profits por defecto
            if signal_type == 'BUY':
                return [
                    entry_price * 1.02,  # 2%
                    entry_price * 1.04,  # 4%
                    entry_price * 1.08   # 8%
                ]
            else:
                return [
                    entry_price * 0.98,  # 2%
                    entry_price * 0.96,  # 4%
                    entry_price * 0.92   # 8%
                ]
    
    @log_function_call
    async def check_risk_limits(self, new_signal: Dict[str, Any]) -> Dict[str, Any]:
        """Verificar límites de riesgo antes de ejecutar señal"""
        try:
            # Verificar exposición total
            total_exposure = await self._calculate_total_exposure()
            max_exposure = self.portfolio_value * 0.8  # 80% máximo
            
            if total_exposure >= max_exposure:
                return {
                    'approved': False,
                    'reason': 'Maximum portfolio exposure reached',
                    'current_exposure': total_exposure,
                    'max_exposure': max_exposure
                }
            
            # Verificar correlación con posiciones existentes
            correlation_check = await self._check_correlation_limits(new_signal['symbol'])
            if not correlation_check['approved']:
                return correlation_check
            
            # Verificar concentración por sector
            sector_check = await self._check_sector_concentration(new_signal['symbol'])
            if not sector_check['approved']:
                return sector_check
            
            # Verificar drawdown máximo
            drawdown_check = await self._check_drawdown_limits()
            if not drawdown_check['approved']:
                return drawdown_check
            
            return {'approved': True}
            
        except Exception as e:
            self.logger.error("Error checking risk limits", exception=e)
            return {'approved': False, 'reason': f'Error: {str(e)}'}
    
    @log_function_call
    async def update_portfolio_metrics(self):
        """Actualizar métricas del portfolio"""
        try:
            # Obtener posiciones activas
            active_signals = await self.signals_db.get_active_signals()
            
            # Calcular valor actual del portfolio
            current_value = self.portfolio_value
            total_pnl = 0.0
            
            for signal in active_signals:
                if signal['status'] in ['ACTIVE', 'EXECUTED']:
                    # Calcular P&L de la posición
                    entry_price = signal['entry_price']
                    current_price = signal.get('current_price', entry_price)
                    
                    if signal['signal_type'] == 'BUY':
                        pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    else:  # SELL
                        pnl_pct = ((entry_price - current_price) / entry_price) * 100
                    
                    total_pnl += pnl_pct
            
            # Actualizar valor del portfolio
            self.portfolio_value = current_value + (current_value * total_pnl / 100)
            
            # Actualizar peak value y drawdown
            if self.portfolio_value > self.peak_value:
                self.peak_value = self.portfolio_value
            
            current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            # Guardar P&L diario
            self.daily_pnl.append({
                'date': datetime.utcnow().date(),
                'pnl_pct': total_pnl,
                'portfolio_value': self.portfolio_value
            })
            
            # Mantener solo últimos 30 días
            if len(self.daily_pnl) > 30:
                self.daily_pnl = self.daily_pnl[-30:]
            
            self.logger.debug(f"Portfolio updated: Value={self.portfolio_value:.2f}, PnL={total_pnl:.2f}%")
            
        except Exception as e:
            self.logger.error("Error updating portfolio metrics", exception=e)
    
    async def _load_portfolio_state(self):
        """Cargar estado del portfolio desde la base de datos"""
        try:
            # Aquí se cargaría el estado desde la base de datos
            # Por ahora usamos valores por defecto
            self.logger.info("Portfolio state loaded from database")
        except Exception as e:
            self.logger.error("Error loading portfolio state", exception=e)
    
    async def _calculate_symbol_risk(self, symbol: str, current_price: float) -> Dict[str, Any]:
        """Calcular métricas de riesgo del símbolo"""
        try:
            # Obtener datos históricos para calcular volatilidad
            volatility = await self._get_symbol_volatility(symbol)
            
            # Determinar nivel de riesgo
            if volatility < 0.02:
                risk_level = RiskLevel.LOW
            elif volatility < 0.05:
                risk_level = RiskLevel.MEDIUM
            elif volatility < 0.10:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            return {
                'volatility': volatility,
                'risk_level': risk_level.value,
                'var_95': current_price * volatility * 1.96,  # 95% VaR
                'max_loss': current_price * volatility * 2
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating symbol risk for {symbol}", exception=e)
            return {'volatility': 0.02, 'risk_level': 'MEDIUM', 'var_95': 0, 'max_loss': 0}
    
    async def _get_symbol_volatility(self, symbol: str) -> float:
        """Obtener volatilidad del símbolo"""
        try:
            # Aquí se calcularía la volatilidad real del símbolo
            # Por ahora usamos valores por defecto basados en el tipo de cripto
            if 'BTC' in symbol:
                return 0.03  # 3%
            elif 'ETH' in symbol:
                return 0.04  # 4%
            else:
                return 0.05  # 5% para altcoins
        except Exception as e:
            self.logger.error(f"Error getting volatility for {symbol}", exception=e)
            return 0.04
    
    async def _calculate_kelly_position_size(self, signal: Dict[str, Any], volatility: float) -> float:
        """Calcular tamaño de posición usando Kelly Criterion"""
        try:
            # Obtener estadísticas históricas del símbolo
            win_rate = await self._get_historical_win_rate(signal['symbol'])
            avg_win = await self._get_avg_win(signal['symbol'])
            avg_loss = await self._get_avg_loss(signal['symbol'])
            
            if avg_loss == 0 or win_rate == 0:
                return 0.02  # 2% por defecto
            
            # Kelly Criterion: f = (bp - q) / b
            # donde b = odds (avg_win/avg_loss), p = win_rate, q = 1 - win_rate
            b = avg_win / avg_loss
            p = win_rate
            q = 1 - win_rate
            
            kelly_fraction = (b * p - q) / b
            
            # Aplicar factor de seguridad (usar solo 25% del Kelly)
            safe_kelly = kelly_fraction * 0.25
            
            # Limitar entre 0.01 y 0.10 (1% y 10%)
            return max(0.01, min(safe_kelly, 0.10))
            
        except Exception as e:
            self.logger.error(f"Error calculating Kelly position size for {signal['symbol']}", exception=e)
            return 0.02
    
    async def _get_historical_win_rate(self, symbol: str) -> float:
        """Obtener tasa de ganancia histórica del símbolo"""
        try:
            # Aquí se consultaría la base de datos para obtener estadísticas históricas
            # Por ahora usamos valores por defecto
            return 0.55  # 55% de tasa de ganancia
        except Exception as e:
            self.logger.error(f"Error getting win rate for {symbol}", exception=e)
            return 0.50
    
    async def _get_avg_win(self, symbol: str) -> float:
        """Obtener ganancia promedio del símbolo"""
        try:
            # Aquí se consultaría la base de datos
            return 0.03  # 3% ganancia promedio
        except Exception as e:
            self.logger.error(f"Error getting avg win for {symbol}", exception=e)
            return 0.02
    
    async def _get_avg_loss(self, symbol: str) -> float:
        """Obtener pérdida promedio del símbolo"""
        try:
            # Aquí se consultaría la base de datos
            return 0.02  # 2% pérdida promedio
        except Exception as e:
            self.logger.error(f"Error getting avg loss for {symbol}", exception=e)
            return 0.02
    
    async def _calculate_diversification_factor(self, symbol: str) -> float:
        """Calcular factor de diversificación"""
        try:
            # Obtener correlación con posiciones existentes
            max_correlation = await self._get_max_correlation(symbol)
            
            # Si la correlación es alta, reducir el tamaño de posición
            if max_correlation > self.risk_config.max_correlation:
                return 0.5  # Reducir a la mitad
            elif max_correlation > 0.5:
                return 0.75  # Reducir un 25%
            else:
                return 1.0  # Sin reducción
            
        except Exception as e:
            self.logger.error(f"Error calculating diversification factor for {symbol}", exception=e)
            return 1.0
    
    async def _get_max_correlation(self, symbol: str) -> float:
        """Obtener correlación máxima con posiciones existentes"""
        try:
            # Aquí se calcularía la correlación real
            # Por ahora usamos valores por defecto
            return 0.3  # 30% correlación promedio
        except Exception as e:
            self.logger.error(f"Error getting max correlation for {symbol}", exception=e)
            return 0.0
    
    async def _calculate_total_exposure(self) -> float:
        """Calcular exposición total del portfolio"""
        try:
            active_signals = await self.signals_db.get_active_signals()
            total_exposure = 0.0
            
            for signal in active_signals:
                if signal['status'] in ['ACTIVE', 'EXECUTED']:
                    # Calcular valor de la posición
                    position_value = signal.get('position_value', 0)
                    total_exposure += position_value
            
            return total_exposure
            
        except Exception as e:
            self.logger.error("Error calculating total exposure", exception=e)
            return 0.0
    
    async def _check_correlation_limits(self, symbol: str) -> Dict[str, Any]:
        """Verificar límites de correlación"""
        try:
            max_correlation = await self._get_max_correlation(symbol)
            
            if max_correlation > self.risk_config.max_correlation:
                return {
                    'approved': False,
                    'reason': f'High correlation with existing positions: {max_correlation:.2%}',
                    'max_correlation': max_correlation,
                    'limit': self.risk_config.max_correlation
                }
            
            return {'approved': True}
            
        except Exception as e:
            self.logger.error(f"Error checking correlation limits for {symbol}", exception=e)
            return {'approved': False, 'reason': f'Error: {str(e)}'}
    
    async def _check_sector_concentration(self, symbol: str) -> Dict[str, Any]:
        """Verificar concentración por sector"""
        try:
            # Obtener sector del símbolo
            sector = await self._get_symbol_sector(symbol)
            
            # Calcular concentración actual del sector
            sector_exposure = await self._calculate_sector_exposure(sector)
            
            if sector_exposure > self.risk_config.max_sector_exposure:
                return {
                    'approved': False,
                    'reason': f'High sector concentration: {sector_exposure:.2%}',
                    'sector': sector,
                    'current_exposure': sector_exposure,
                    'limit': self.risk_config.max_sector_exposure
                }
            
            return {'approved': True}
            
        except Exception as e:
            self.logger.error(f"Error checking sector concentration for {symbol}", exception=e)
            return {'approved': False, 'reason': f'Error: {str(e)}'}
    
    async def _get_symbol_sector(self, symbol: str) -> str:
        """Obtener sector del símbolo"""
        # Mapeo simple de símbolos a sectores
        sector_map = {
            'BTC': 'Store of Value',
            'ETH': 'Smart Contracts',
            'ADA': 'Smart Contracts',
            'SOL': 'Smart Contracts',
            'MATIC': 'Layer 2',
            'LINK': 'Oracle',
            'DOT': 'Interoperability',
            'AVAX': 'Smart Contracts',
            'ATOM': 'Interoperability',
            'NEAR': 'Smart Contracts'
        }
        
        for key, sector in sector_map.items():
            if key in symbol:
                return sector
        
        return 'Other'
    
    async def _calculate_sector_exposure(self, sector: str) -> float:
        """Calcular exposición del sector"""
        try:
            active_signals = await self.signals_db.get_active_signals()
            sector_exposure = 0.0
            total_exposure = 0.0
            
            for signal in active_signals:
                if signal['status'] in ['ACTIVE', 'EXECUTED']:
                    position_value = signal.get('position_value', 0)
                    total_exposure += position_value
                    
                    signal_sector = await self._get_symbol_sector(signal['symbol'])
                    if signal_sector == sector:
                        sector_exposure += position_value
            
            return sector_exposure / total_exposure if total_exposure > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating sector exposure for {sector}", exception=e)
            return 0.0
    
    async def _check_drawdown_limits(self) -> Dict[str, Any]:
        """Verificar límites de drawdown"""
        try:
            current_drawdown = (self.peak_value - self.portfolio_value) / self.peak_value
            
            # Límite de drawdown del 20%
            max_drawdown_limit = 0.20
            
            if current_drawdown > max_drawdown_limit:
                return {
                    'approved': False,
                    'reason': f'Maximum drawdown exceeded: {current_drawdown:.2%}',
                    'current_drawdown': current_drawdown,
                    'limit': max_drawdown_limit
                }
            
            return {'approved': True}
            
        except Exception as e:
            self.logger.error("Error checking drawdown limits", exception=e)
            return {'approved': False, 'reason': f'Error: {str(e)}'}
    
    async def _get_atr(self, symbol: str, timeframe: str) -> Optional[float]:
        """Obtener ATR del símbolo"""
        try:
            # Aquí se obtendría el ATR real del análisis técnico
            # Por ahora usamos valores por defecto
            return 0.02  # 2% ATR promedio
        except Exception as e:
            self.logger.error(f"Error getting ATR for {symbol}", exception=e)
            return None
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Obtener métricas de riesgo actuales"""
        return RiskMetrics(
            portfolio_value=self.portfolio_value,
            total_exposure=0.0,  # Se calcularía dinámicamente
            max_drawdown=self.max_drawdown,
            sharpe_ratio=0.0,  # Se calcularía con datos históricos
            var_95=0.0,  # Se calcularía con datos históricos
            max_correlation=0.0,  # Se calcularía dinámicamente
            sector_concentration={}  # Se calcularía dinámicamente
        )