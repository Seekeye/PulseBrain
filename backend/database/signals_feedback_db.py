#!/usr/bin/env python3
"""
Signals Feedback Database - CryptoPulse Pro
Base de datos para almacenar señales y su retroalimentación
"""

import asyncio
import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import pandas as pd

from utils.logger import get_logger, log_execution_time, log_function_call

class SignalsFeedbackDB:
    """
    Base de datos SQLite para almacenar:
    - Señales generadas
    - Resultados de señales
    - Métricas de rendimiento
    - Datos de entrenamiento ML
    """
    
    def __init__(self, db_path: str = "data/signals_feedback.db"):
        self.db_path = db_path
        self.logger = get_logger("SignalsFeedbackDB")
        
        # Crear directorio si no existe
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.logger.info(f"📊 SignalsFeedbackDB initialized: {db_path}")
    
    async def initialize(self):
        """Inicializar la base de datos y crear tablas"""
        try:
            self.logger.info("🔧 Initializing Signals Feedback Database...")
            
            # Crear tablas
            await self._create_tables()
            
            # Crear índices para mejor rendimiento
            await self._create_indexes()
            
            self.logger.info("✅ Signals Feedback Database initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
    
    async def _create_tables(self):
        """Crear todas las tablas necesarias"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Tabla de señales
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit_1 REAL NOT NULL,
                    take_profit_2 REAL,
                    take_profit_3 REAL,
                    confidence_score REAL NOT NULL,
                    coherence_score REAL NOT NULL,
                    reasoning TEXT,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de resultados de señales
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signal_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    pnl_percentage REAL NOT NULL,
                    success BOOLEAN NOT NULL,
                    failure BOOLEAN NOT NULL,
                    status TEXT NOT NULL,
                    update_time TEXT NOT NULL,
                    additional_data TEXT,
                    FOREIGN KEY (signal_id) REFERENCES signals (signal_id)
                )
            """)
            
            # Tabla de métricas de rendimiento
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT NOT NULL,
                    total_signals INTEGER NOT NULL,
                    successful_signals INTEGER NOT NULL,
                    failed_signals INTEGER NOT NULL,
                    win_rate REAL NOT NULL,
                    average_pnl REAL NOT NULL,
                    best_signal_id TEXT,
                    worst_signal_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla de datos de entrenamiento ML
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ml_training_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL,
                    features TEXT NOT NULL,  -- JSON array
                    target REAL NOT NULL,
                    model_version INTEGER NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (signal_id) REFERENCES signals (signal_id)
                )
            """)
            
            # Tabla de configuraciones del sistema
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_config (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    config_key TEXT UNIQUE NOT NULL,
                    config_value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            
            self.logger.info("✅ Database tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating tables: {e}")
            raise
    
    async def _create_indexes(self):
        """Crear índices para mejorar rendimiento"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Índices para señales
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals (symbol)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_timestamp ON signals (timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_signals_type ON signals (signal_type)")
            
            # Índices para resultados
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_signal_id ON signal_outcomes (signal_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_outcomes_status ON signal_outcomes (status)")
            
            # Índices para métricas
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_date ON performance_metrics (date)")
            
            # Índices para ML
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_signal_id ON ml_training_data (signal_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_ml_version ON ml_training_data (model_version)")
            
            conn.commit()
            conn.close()
            
            self.logger.info("✅ Database indexes created successfully")
            
        except Exception as e:
            self.logger.error(f"Error creating indexes: {e}")
    
    @log_execution_time
    async def save_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Guardar una señal en la base de datos"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO signals (
                    signal_id, symbol, signal_type, entry_price, stop_loss,
                    take_profit_1, take_profit_2, take_profit_3,
                    confidence_score, coherence_score, reasoning, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_data.get('signal_id', ''),
                signal_data.get('symbol', ''),
                signal_data.get('signal_type', ''),
                signal_data.get('entry_price', 0),
                signal_data.get('stop_loss', 0),
                signal_data.get('take_profit_1', 0),
                signal_data.get('take_profit_2', 0),
                signal_data.get('take_profit_3', 0),
                signal_data.get('confidence_score', 0),
                signal_data.get('coherence_score', 0),
                json.dumps(signal_data.get('reasoning', {})),
                signal_data.get('timestamp', datetime.now().isoformat())
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Signal saved: {signal_data.get('signal_id', 'UNKNOWN')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving signal: {e}")
            return False
    
    @log_execution_time
    async def save_signal_outcome(self, signal_id: str, outcome_data: Dict[str, Any]) -> bool:
        """Guardar resultado de una señal"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO signal_outcomes (
                    signal_id, current_price, pnl_percentage, success, failure,
                    status, update_time, additional_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_id,
                outcome_data.get('current_price', 0),
                outcome_data.get('pnl_percentage', 0),
                outcome_data.get('success', False),
                outcome_data.get('failure', False),
                outcome_data.get('status', 'TRACKING'),
                outcome_data.get('update_time', datetime.now().isoformat()),
                json.dumps(outcome_data.get('additional_data', {}))
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Signal outcome saved: {signal_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving signal outcome: {e}")
            return False
    
    @log_execution_time
    async def save_performance_metrics(self, metrics_data: Dict[str, Any]) -> bool:
        """Guardar métricas de rendimiento"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO performance_metrics (
                    date, total_signals, successful_signals, failed_signals,
                    win_rate, average_pnl, best_signal_id, worst_signal_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics_data.get('date', datetime.now().strftime('%Y-%m-%d')),
                metrics_data.get('total_signals', 0),
                metrics_data.get('successful_signals', 0),
                metrics_data.get('failed_signals', 0),
                metrics_data.get('win_rate', 0),
                metrics_data.get('average_pnl', 0),
                metrics_data.get('best_signal_id', ''),
                metrics_data.get('worst_signal_id', '')
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"✅ Performance metrics saved for {metrics_data.get('date', 'UNKNOWN')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving performance metrics: {e}")
            return False
    
    @log_execution_time
    async def save_ml_training_data(self, signal_id: str, features: List[float], target: float, model_version: int) -> bool:
        """Guardar datos de entrenamiento ML"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO ml_training_data (
                    signal_id, features, target, model_version
                ) VALUES (?, ?, ?, ?)
            """, (
                signal_id,
                json.dumps(features),
                target,
                model_version
            ))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving ML training data: {e}")
            return False
    
    @log_execution_time
    async def get_signals_by_symbol(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener señales por símbolo"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM signals 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (symbol, limit))
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            signals = []
            for row in rows:
                signal = dict(zip(columns, row))
                # Parsear reasoning JSON
                if signal.get('reasoning'):
                    try:
                        signal['reasoning'] = json.loads(signal['reasoning'])
                    except:
                        signal['reasoning'] = {}
                signals.append(signal)
            
            conn.close()
            return signals
            
        except Exception as e:
            self.logger.error(f"Error getting signals by symbol: {e}")
            return []
    
    @log_execution_time
    async def get_signal_outcomes(self, signal_id: str) -> List[Dict[str, Any]]:
        """Obtener resultados de una señal"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM signal_outcomes 
                WHERE signal_id = ? 
                ORDER BY update_time ASC
            """, (signal_id,))
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            outcomes = []
            for row in rows:
                outcome = dict(zip(columns, row))
                # Parsear additional_data JSON
                if outcome.get('additional_data'):
                    try:
                        outcome['additional_data'] = json.loads(outcome['additional_data'])
                    except:
                        outcome['additional_data'] = {}
                outcomes.append(outcome)
            
            conn.close()
            return outcomes
            
        except Exception as e:
            self.logger.error(f"Error getting signal outcomes: {e}")
            return []
    
    @log_execution_time
    async def get_performance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Obtener resumen de rendimiento de los últimos N días"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Obtener métricas de los últimos días
            cursor.execute("""
                SELECT * FROM performance_metrics 
                WHERE date >= date('now', '-{} days')
                ORDER BY date DESC
            """.format(days))
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            metrics_history = []
            for row in rows:
                metrics = dict(zip(columns, row))
                metrics_history.append(metrics)
            
            # Calcular promedios
            if metrics_history:
                avg_win_rate = sum(m['win_rate'] for m in metrics_history) / len(metrics_history)
                avg_pnl = sum(m['average_pnl'] for m in metrics_history) / len(metrics_history)
                total_signals = sum(m['total_signals'] for m in metrics_history)
            else:
                avg_win_rate = 0
                avg_pnl = 0
                total_signals = 0
            
            conn.close()
            
            return {
                'period_days': days,
                'total_signals': total_signals,
                'average_win_rate': avg_win_rate,
                'average_pnl': avg_pnl,
                'metrics_history': metrics_history,
                'trend': 'IMPROVING' if len(metrics_history) > 1 and metrics_history[0]['win_rate'] > metrics_history[-1]['win_rate'] else 'STABLE'
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {
                'period_days': days,
                'total_signals': 0,
                'average_win_rate': 0,
                'average_pnl': 0,
                'metrics_history': [],
                'trend': 'UNKNOWN'
            }
    
    @log_execution_time
    async def get_ml_training_data(self, model_version: int = None, limit: int = 1000) -> List[Dict[str, Any]]:
        """Obtener datos de entrenamiento ML"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if model_version:
                cursor.execute("""
                    SELECT * FROM ml_training_data 
                    WHERE model_version = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (model_version, limit))
            else:
                cursor.execute("""
                    SELECT * FROM ml_training_data 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,))
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            training_data = []
            for row in rows:
                data = dict(zip(columns, row))
                # Parsear features JSON
                if data.get('features'):
                    try:
                        data['features'] = json.loads(data['features'])
                    except:
                        data['features'] = []
                training_data.append(data)
            
            conn.close()
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error getting ML training data: {e}")
            return []
    
    @log_execution_time
    async def get_database_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la base de datos"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            # Contar registros en cada tabla
            tables = ['signals', 'signal_outcomes', 'performance_metrics', 'ml_training_data']
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats[f'{table}_count'] = count
            
            # Obtener señales por tipo
            cursor.execute("""
                SELECT signal_type, COUNT(*) 
                FROM signals 
                GROUP BY signal_type
            """)
            signal_types = dict(cursor.fetchall())
            stats['signal_types'] = signal_types
            
            # Obtener señales por símbolo
            cursor.execute("""
                SELECT symbol, COUNT(*) 
                FROM signals 
                GROUP BY symbol 
                ORDER BY COUNT(*) DESC 
                LIMIT 10
            """)
            top_symbols = dict(cursor.fetchall())
            stats['top_symbols'] = top_symbols
            
            # Tamaño de la base de datos
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0]
            stats['database_size_bytes'] = db_size
            
            conn.close()
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting database stats: {e}")
            return {}

if __name__ == "__main__":
    async def test_database():
        db = SignalsFeedbackDB()
        await db.initialize()
        
        # Simular una señal
        signal_data = {
            'signal_id': 'test_signal_001',
            'symbol': 'BTCUSDT',
            'signal_type': 'BUY',
            'entry_price': 50000,
            'stop_loss': 48000,
            'take_profit_1': 52000,
            'confidence_score': 85,
            'coherence_score': 80,
            'reasoning': {'technical': 'RSI oversold', 'ml': 'Strong buy signal'},
            'timestamp': datetime.now().isoformat()
        }
        
        # Guardar señal
        await db.save_signal(signal_data)
        
        # Simular resultado
        outcome_data = {
            'current_price': 51000,
            'pnl_percentage': 2.0,
            'success': True,
            'failure': False,
            'status': 'SUCCESS',
            'update_time': datetime.now().isoformat()
        }
        
        await db.save_signal_outcome('test_signal_001', outcome_data)
        
        # Obtener estadísticas
        stats = await db.get_database_stats()
        print(f"Database Stats: {stats}")
    
    asyncio.run(test_database())
