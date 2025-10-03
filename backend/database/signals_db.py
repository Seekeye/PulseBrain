#!/usr/bin/env python3
"""
Base de Datos de Señales - CryptoPulse Pro
Maneja el almacenamiento y consulta de señales de trading
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy import create_engine, Column, String, Float, DateTime, Boolean, Integer, Text, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.sqlite import insert
from sqlalchemy import func, and_, or_, desc, asc
import json

from utils.logger import get_logger

Base = declarative_base()

class SignalModel(Base):
    """Modelo de señal en la base de datos"""
    __tablename__ = 'signals'
    
    # Identificadores
    id = Column(String, primary_key=True)
    symbol = Column(String, nullable=False, index=True)
    signal_type = Column(String, nullable=False)  # BUY, SELL
    status = Column(String, default='PENDING', index=True)  # PENDING, ACTIVE, EXECUTED, CANCELLED, EXPIRED
    
    # Precios
    entry_price = Column(Float, nullable=False)
    current_price = Column(Float, default=0.0)
    stop_loss = Column(Float, nullable=False)
    take_profit_1 = Column(Float, nullable=False)
    take_profit_2 = Column(Float, nullable=False)
    take_profit_3 = Column(Float, nullable=False)
    
    # Métricas
    confidence_score = Column(Float, nullable=False, index=True)
    risk_reward_ratio = Column(Float, nullable=False)
    pnl_percentage = Column(Float, default=0.0)
    
    # Estados de ejecución
    tp1_hit = Column(Boolean, default=False)
    tp2_hit = Column(Boolean, default=False)
    tp3_hit = Column(Boolean, default=False)
    stop_loss_hit = Column(Boolean, default=False)
    
    # Configuración
    timeframe = Column(String, nullable=False)
    strategy = Column(String, nullable=False, index=True)
    market_context = Column(String, default='NEUTRAL')
    expected_duration = Column(String, default='MEDIUM')
    
    # Análisis
    reasoning = Column(Text)
    technical_analysis = Column(Text)  # JSON string
    sentiment_analysis = Column(Text)  # JSON string
    news_analysis = Column(Text)  # JSON string
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    executed_at = Column(DateTime)
    closed_at = Column(DateTime)
    
    # Índices compuestos
    __table_args__ = (
        Index('idx_symbol_status', 'symbol', 'status'),
        Index('idx_created_at_status', 'created_at', 'status'),
        Index('idx_confidence_score', 'confidence_score'),
        Index('idx_strategy_status', 'strategy', 'status'),
    )

class TrackingEventModel(Base):
    """Modelo de eventos de seguimiento"""
    __tablename__ = 'tracking_events'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(String, nullable=False, index=True)
    event_type = Column(String, nullable=False)  # SIGNAL_CREATED, PRICE_UPDATE, TP_HIT, SL_HIT, SIGNAL_CLOSED
    event_data = Column(Text)  # JSON string
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    price = Column(Float)
    pnl_percentage = Column(Float)

class PerformanceStatsModel(Base):
    """Modelo de estadísticas de performance"""
    __tablename__ = 'performance_stats'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False, index=True)
    total_signals = Column(Integer, default=0)
    successful_signals = Column(Integer, default=0)
    failed_signals = Column(Integer, default=0)
    total_pnl = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    avg_confidence = Column(Float, default=0.0)
    best_signal = Column(String)
    worst_signal = Column(String)

class SignalsDatabase:
    """
    Base de datos de señales con funcionalidades avanzadas
    - Almacenamiento de señales
    - Seguimiento de eventos
    - Estadísticas de performance
    - Consultas optimizadas
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.SessionLocal = None
        self.logger = get_logger("SignalsDatabase")
        self.logger.info("🗄️ SignalsDatabase initialized")
    
    async def initialize(self):
        """Inicializar la base de datos"""
        try:
            self.logger.info("🔧 Initializing database...")
            
            # Crear engine
            self.engine = create_engine(
                self.database_url,
                echo=False,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            
            # Crear tablas
            Base.metadata.create_all(bind=self.engine)
            
            # Crear session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            self.logger.info("✅ Database initialized successfully")
            
        except Exception as e:
            self.logger.critical("Failed to initialize database", exception=e)
            raise
    
    def get_session(self):
        """Obtener sesión de base de datos"""
        return self.SessionLocal()
    
    async def save_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Guardar una señal en la base de datos"""
        try:
            with self.get_session() as session:
                # Crear modelo de señal
                signal = SignalModel(
                    id=signal_data['id'],
                    symbol=signal_data['symbol'],
                    signal_type=signal_data['signal_type'],
                    entry_price=signal_data['entry_price'],
                    current_price=signal_data.get('current_price', signal_data['entry_price']),
                    confidence_score=signal_data['confidence_score'],
                    risk_reward_ratio=signal_data['risk_reward_ratio'],
                    stop_loss=signal_data['stop_loss'],
                    take_profit_1=signal_data['take_profit_1'],
                    take_profit_2=signal_data['take_profit_2'],
                    take_profit_3=signal_data['take_profit_3'],
                    timeframe=signal_data['timeframe'],
                    strategy=signal_data['strategy'],
                    reasoning=signal_data.get('reasoning', ''),
                    status=signal_data.get('status', 'PENDING'),
                    market_context=signal_data.get('market_context', 'NEUTRAL'),
                    expected_duration=signal_data.get('expected_duration', 'MEDIUM'),
                    technical_analysis=json.dumps(signal_data.get('technical_analysis', {})),
                    sentiment_analysis=json.dumps(signal_data.get('sentiment_analysis', {})),
                    news_analysis=json.dumps(signal_data.get('news_analysis', {}))
                )
                
                # Usar upsert para evitar duplicados
                stmt = insert(SignalModel).values(**signal.__dict__)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['id'],
                    set_=dict(
                        current_price=stmt.excluded.current_price,
                        status=stmt.excluded.status,
                        updated_at=datetime.utcnow()
                    )
                )
                
                session.execute(stmt)
                session.commit()
                
                # Crear evento de seguimiento
                await self.create_tracking_event(
                    signal_data['id'],
                    'SIGNAL_CREATED',
                    signal_data
                )
                
                self.logger.info(f"Signal {signal_data['id']} saved successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving signal {signal_data.get('id', 'unknown')}", exception=e)
            return False
    
    async def update_signal(self, signal_id: str, updates: Dict[str, Any]) -> bool:
        """Actualizar una señal existente"""
        try:
            with self.get_session() as session:
                signal = session.query(SignalModel).filter(SignalModel.id == signal_id).first()
                
                if not signal:
                    self.logger.warning(f"Signal {signal_id} not found for update")
                    return False
                
                # Actualizar campos
                for key, value in updates.items():
                    if hasattr(signal, key):
                        setattr(signal, key, value)
                
                signal.updated_at = datetime.utcnow()
                session.commit()
                
                # Crear evento de seguimiento
                await self.create_tracking_event(
                    signal_id,
                    'SIGNAL_UPDATED',
                    updates
                )
                
                self.logger.info(f"Signal {signal_id} updated successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating signal {signal_id}", exception=e)
            return False
    
    async def get_signal(self, signal_id: str) -> Optional[Dict[str, Any]]:
        """Obtener una señal por ID"""
        try:
            with self.get_session() as session:
                signal = session.query(SignalModel).filter(SignalModel.id == signal_id).first()
                
                if not signal:
                    return None
                
                return self._signal_to_dict(signal)
                
        except Exception as e:
            self.logger.error(f"Error getting signal {signal_id}", exception=e)
            return None
    
    async def get_active_signals(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtener señales activas"""
        try:
            with self.get_session() as session:
                query = session.query(SignalModel).filter(
                    SignalModel.status.in_(['PENDING', 'ACTIVE'])
                )
                
                if symbol:
                    query = query.filter(SignalModel.symbol == symbol)
                
                signals = query.order_by(desc(SignalModel.created_at)).all()
                
                return [self._signal_to_dict(signal) for signal in signals]
                
        except Exception as e:
            self.logger.error(f"Error getting active signals", exception=e)
            return []
    
    async def get_signals_by_symbol(self, symbol: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtener señales por símbolo"""
        try:
            with self.get_session() as session:
                signals = session.query(SignalModel).filter(
                    SignalModel.symbol == symbol
                ).order_by(desc(SignalModel.created_at)).limit(limit).all()
                
                return [self._signal_to_dict(signal) for signal in signals]
                
        except Exception as e:
            self.logger.error(f"Error getting signals for {symbol}", exception=e)
            return []
    
    async def get_signals_by_strategy(self, strategy: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtener señales por estrategia"""
        try:
            with self.get_session() as session:
                signals = session.query(SignalModel).filter(
                    SignalModel.strategy == strategy
                ).order_by(desc(SignalModel.created_at)).limit(limit).all()
                
                return [self._signal_to_dict(signal) for signal in signals]
                
        except Exception as e:
            self.logger.error(f"Error getting signals for strategy {strategy}", exception=e)
            return []
    
    async def get_recent_signals(self, limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtener señales recientes"""
        try:
            with self.get_session() as session:
                query = session.query(SignalModel)
                
                if status:
                    query = query.filter(SignalModel.status == status)
                
                signals = query.order_by(desc(SignalModel.created_at)).limit(limit).all()
                
                return [self._signal_to_dict(signal) for signal in signals]
                
        except Exception as e:
            self.logger.error(f"Error getting recent signals", exception=e)
            return []
    
    async def get_signals_by_confidence(self, min_confidence: float, max_confidence: float = 100.0) -> List[Dict[str, Any]]:
        """Obtener señales por rango de confianza"""
        try:
            with self.get_session() as session:
                signals = session.query(SignalModel).filter(
                    and_(
                        SignalModel.confidence_score >= min_confidence,
                        SignalModel.confidence_score <= max_confidence
                    )
                ).order_by(desc(SignalModel.confidence_score)).all()
                
                return [self._signal_to_dict(signal) for signal in signals]
                
        except Exception as e:
            self.logger.error(f"Error getting signals by confidence", exception=e)
            return []
    
    async def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        """Obtener estadísticas de performance"""
        try:
            with self.get_session() as session:
                # Fecha de inicio
                start_date = datetime.utcnow() - timedelta(days=days)
                
                # Total de señales
                total_signals = session.query(SignalModel).filter(
                    SignalModel.created_at >= start_date
                ).count()
                
                # Señales exitosas (TP1, TP2, TP3 hit)
                successful_signals = session.query(SignalModel).filter(
                    and_(
                        SignalModel.created_at >= start_date,
                        or_(
                            SignalModel.tp1_hit == True,
                            SignalModel.tp2_hit == True,
                            SignalModel.tp3_hit == True
                        )
                    )
                ).count()
                
                # Señales fallidas (SL hit)
                failed_signals = session.query(SignalModel).filter(
                    and_(
                        SignalModel.created_at >= start_date,
                        SignalModel.stop_loss_hit == True
                    )
                ).count()
                
                # PnL total
                total_pnl = session.query(func.sum(SignalModel.pnl_percentage)).filter(
                    SignalModel.created_at >= start_date
                ).scalar() or 0.0
                
                # Tasa de éxito
                win_rate = (successful_signals / total_signals * 100) if total_signals > 0 else 0.0
                
                # Confianza promedio
                avg_confidence = session.query(func.avg(SignalModel.confidence_score)).filter(
                    SignalModel.created_at >= start_date
                ).scalar() or 0.0
                
                # Mejor señal
                best_signal = session.query(SignalModel).filter(
                    SignalModel.created_at >= start_date
                ).order_by(desc(SignalModel.pnl_percentage)).first()
                
                # Peor señal
                worst_signal = session.query(SignalModel).filter(
                    SignalModel.created_at >= start_date
                ).order_by(asc(SignalModel.pnl_percentage)).first()
                
                return {
                    'total_signals': total_signals,
                    'successful_signals': successful_signals,
                    'failed_signals': failed_signals,
                    'total_pnl': float(total_pnl),
                    'win_rate': float(win_rate),
                    'avg_confidence': float(avg_confidence),
                    'best_signal_id': best_signal.id if best_signal else None,
                    'worst_signal_id': worst_signal.id if worst_signal else None,
                    'period_days': days
                }
                
        except Exception as e:
            self.logger.error(f"Error getting performance stats", exception=e)
            return {}
    
    async def create_tracking_event(self, signal_id: str, event_type: str, event_data: Dict[str, Any]) -> bool:
        """Crear evento de seguimiento"""
        try:
            with self.get_session() as session:
                event = TrackingEventModel(
                    signal_id=signal_id,
                    event_type=event_type,
                    event_data=json.dumps(event_data),
                    price=event_data.get('current_price', 0.0),
                    pnl_percentage=event_data.get('pnl_percentage', 0.0)
                )
                
                session.add(event)
                session.commit()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error creating tracking event for {signal_id}", exception=e)
            return False
    
    async def get_tracking_events(self, signal_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtener eventos de seguimiento"""
        try:
            with self.get_session() as session:
                query = session.query(TrackingEventModel)
                
                if signal_id:
                    query = query.filter(TrackingEventModel.signal_id == signal_id)
                
                events = query.order_by(desc(TrackingEventModel.timestamp)).limit(limit).all()
                
                return [self._tracking_event_to_dict(event) for event in events]
                
        except Exception as e:
            self.logger.error(f"Error getting tracking events", exception=e)
            return []
    
    async def update_signal_price(self, signal_id: str, current_price: float) -> bool:
        """Actualizar precio actual de una señal"""
        try:
            with self.get_session() as session:
                signal = session.query(SignalModel).filter(SignalModel.id == signal_id).first()
                
                if not signal:
                    return False
                
                # Actualizar precio
                signal.current_price = current_price
                
                # Calcular PnL
                if signal.signal_type == 'BUY':
                    pnl = ((current_price - signal.entry_price) / signal.entry_price) * 100
                else:  # SELL
                    pnl = ((signal.entry_price - current_price) / signal.entry_price) * 100
                
                signal.pnl_percentage = pnl
                
                # Verificar si se alcanzaron niveles
                await self._check_price_levels(signal, current_price)
                
                signal.updated_at = datetime.utcnow()
                session.commit()
                
                # Crear evento de seguimiento
                await self.create_tracking_event(
                    signal_id,
                    'PRICE_UPDATE',
                    {
                        'current_price': current_price,
                        'pnl_percentage': pnl,
                        'timestamp': datetime.utcnow().isoformat()
                    }
                )
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating signal price for {signal_id}", exception=e)
            return False
    
    async def _check_price_levels(self, signal: SignalModel, current_price: float):
        """Verificar si se alcanzaron niveles de TP o SL"""
        try:
            # Verificar Take Profits
            if signal.signal_type == 'BUY':
                if current_price >= signal.take_profit_3 and not signal.tp3_hit:
                    signal.tp3_hit = True
                    signal.status = 'EXECUTED'
                    signal.executed_at = datetime.utcnow()
                    await self.create_tracking_event(
                        signal.id,
                        'TP3_HIT',
                        {'price': current_price, 'level': signal.take_profit_3}
                    )
                elif current_price >= signal.take_profit_2 and not signal.tp2_hit:
                    signal.tp2_hit = True
                    await self.create_tracking_event(
                        signal.id,
                        'TP2_HIT',
                        {'price': current_price, 'level': signal.take_profit_2}
                    )
                elif current_price >= signal.take_profit_1 and not signal.tp1_hit:
                    signal.tp1_hit = True
                    await self.create_tracking_event(
                        signal.id,
                        'TP1_HIT',
                        {'price': current_price, 'level': signal.take_profit_1}
                    )
                elif current_price <= signal.stop_loss and not signal.stop_loss_hit:
                    signal.stop_loss_hit = True
                    signal.status = 'EXECUTED'
                    signal.executed_at = datetime.utcnow()
                    await self.create_tracking_event(
                        signal.id,
                        'SL_HIT',
                        {'price': current_price, 'level': signal.stop_loss}
                    )
            
            else:  # SELL
                if current_price <= signal.take_profit_3 and not signal.tp3_hit:
                    signal.tp3_hit = True
                    signal.status = 'EXECUTED'
                    signal.executed_at = datetime.utcnow()
                    await self.create_tracking_event(
                        signal.id,
                        'TP3_HIT',
                        {'price': current_price, 'level': signal.take_profit_3}
                    )
                elif current_price <= signal.take_profit_2 and not signal.tp2_hit:
                    signal.tp2_hit = True
                    await self.create_tracking_event(
                        signal.id,
                        'TP2_HIT',
                        {'price': current_price, 'level': signal.take_profit_2}
                    )
                elif current_price <= signal.take_profit_1 and not signal.tp1_hit:
                    signal.tp1_hit = True
                    await self.create_tracking_event(
                        signal.id,
                        'TP1_HIT',
                        {'price': current_price, 'level': signal.take_profit_1}
                    )
                elif current_price >= signal.stop_loss and not signal.stop_loss_hit:
                    signal.stop_loss_hit = True
                    signal.status = 'EXECUTED'
                    signal.executed_at = datetime.utcnow()
                    await self.create_tracking_event(
                        signal.id,
                        'SL_HIT',
                        {'price': current_price, 'level': signal.stop_loss}
                    )
                    
        except Exception as e:
            self.logger.error(f"Error checking price levels for signal {signal.id}", exception=e)
    
    async def close_signal(self, signal_id: str, reason: str = "Manual close") -> bool:
        """Cerrar una señal manualmente"""
        try:
            with self.get_session() as session:
                signal = session.query(SignalModel).filter(SignalModel.id == signal_id).first()
                
                if not signal:
                    return False
                
                signal.status = 'EXECUTED'
                signal.closed_at = datetime.utcnow()
                signal.updated_at = datetime.utcnow()
                
                session.commit()
                
                # Crear evento de seguimiento
                await self.create_tracking_event(
                    signal_id,
                    'SIGNAL_CLOSED',
                    {'reason': reason, 'timestamp': datetime.utcnow().isoformat()}
                )
                
                self.logger.info(f"Signal {signal_id} closed: {reason}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error closing signal {signal_id}", exception=e)
            return False
    
    async def cancel_signal(self, signal_id: str, reason: str = "Manual cancel") -> bool:
        """Cancelar una señal"""
        try:
            with self.get_session() as session:
                signal = session.query(SignalModel).filter(SignalModel.id == signal_id).first()
                
                if not signal:
                    return False
                
                signal.status = 'CANCELLED'
                signal.updated_at = datetime.utcnow()
                
                session.commit()
                
                # Crear evento de seguimiento
                await self.create_tracking_event(
                    signal_id,
                    'SIGNAL_CANCELLED',
                    {'reason': reason, 'timestamp': datetime.utcnow().isoformat()}
                )
                
                self.logger.info(f"Signal {signal_id} cancelled: {reason}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error cancelling signal {signal_id}", exception=e)
            return False
    
    async def clear_old_signals(self, days: int = 30) -> int:
        """Limpiar señales antiguas"""
        try:
            with self.get_session() as session:
                cutoff_date = datetime.utcnow() - timedelta(days=days)
                
                # Obtener señales a eliminar
                old_signals = session.query(SignalModel).filter(
                    and_(
                        SignalModel.created_at < cutoff_date,
                        SignalModel.status.in_(['EXECUTED', 'CANCELLED', 'EXPIRED'])
                    )
                ).all()
                
                count = len(old_signals)
                
                # Eliminar señales
                for signal in old_signals:
                    session.delete(signal)
                
                session.commit()
                
                self.logger.info(f"Cleared {count} old signals")
                return count
                
        except Exception as e:
            self.logger.error(f"Error clearing old signals", exception=e)
            return 0
    
    def _signal_to_dict(self, signal: SignalModel) -> Dict[str, Any]:
        """Convertir modelo de señal a diccionario"""
        try:
            return {
                'id': signal.id,
                'symbol': signal.symbol,
                'signal_type': signal.signal_type,
                'entry_price': signal.entry_price,
                'current_price': signal.current_price,
                'confidence_score': signal.confidence_score,
                'risk_reward_ratio': signal.risk_reward_ratio,
                'stop_loss': signal.stop_loss,
                'take_profit_1': signal.take_profit_1,
                'take_profit_2': signal.take_profit_2,
                'take_profit_3': signal.take_profit_3,
                'tp1_hit': signal.tp1_hit,
                'tp2_hit': signal.tp2_hit,
                'tp3_hit': signal.tp3_hit,
                'stop_loss_hit': signal.stop_loss_hit,
                'timeframe': signal.timeframe,
                'strategy': signal.strategy,
                'reasoning': signal.reasoning,
                'status': signal.status,
                'market_context': signal.market_context,
                'expected_duration': signal.expected_duration,
                'pnl_percentage': signal.pnl_percentage,
                'created_at': signal.created_at.isoformat() if signal.created_at else None,
                'updated_at': signal.updated_at.isoformat() if signal.updated_at else None,
                'executed_at': signal.executed_at.isoformat() if signal.executed_at else None,
                'closed_at': signal.closed_at.isoformat() if signal.closed_at else None,
                'technical_analysis': json.loads(signal.technical_analysis) if signal.technical_analysis else {},
                'sentiment_analysis': json.loads(signal.sentiment_analysis) if signal.sentiment_analysis else {},
                'news_analysis': json.loads(signal.news_analysis) if signal.news_analysis else {}
            }
        except Exception as e:
            self.logger.error(f"Error converting signal to dict: {e}")
            return {}
    
    def _tracking_event_to_dict(self, event: TrackingEventModel) -> Dict[str, Any]:
        """Convertir evento de seguimiento a diccionario"""
        try:
            return {
                'id': event.id,
                'signal_id': event.signal_id,
                'event_type': event.event_type,
                'event_data': json.loads(event.event_data) if event.event_data else {},
                'timestamp': event.timestamp.isoformat() if event.timestamp else None,
                'price': event.price,
                'pnl_percentage': event.pnl_percentage
            }
        except Exception as e:
            self.logger.error(f"Error converting tracking event to dict: {e}")
            return {}
    
    async def close(self):
        """Cerrar conexión a la base de datos"""
        try:
            if self.engine:
                self.engine.dispose()
            self.logger.info("✅ Database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}")
