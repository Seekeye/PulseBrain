#!/usr/bin/env python3
"""
Sistema de Configuración Avanzado para CryptoPulse Pro
Configuración centralizada con validación y tipos
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

class LogLevel(str, Enum):
    """Niveles de logging disponibles"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class TradingMode(str, Enum):
    """Modos de trading disponibles"""
    PAPER = "PAPER"          # Trading simulado
    LIVE = "LIVE"            # Trading real
    BACKTEST = "BACKTEST"    # Backtesting

class TimeFrame(str, Enum):
    """Timeframes disponibles"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

@dataclass
class DatabaseConfig:
    """Configuración de base de datos"""
    url: str = "sqlite:///data/cryptopulse.db"
    echo: bool = False
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600

@dataclass
class APIConfig:
    """Configuración de APIs externas"""
    # Binance
    binance_api_key: Optional[str] = None
    binance_secret_key: Optional[str] = None
    binance_testnet: bool = True
    
    # CoinGecko
    coingecko_api_key: Optional[str] = None
    coingecko_rate_limit: int = 50  # requests per minute
    
    # Reddit
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    reddit_user_agent: str = "CryptoPulsePro/1.0"
    
    # Twitter
    twitter_bearer_token: Optional[str] = None
    twitter_api_key: Optional[str] = None
    twitter_api_secret: Optional[str] = None
    twitter_access_token: Optional[str] = None
    twitter_access_token_secret: Optional[str] = None
    
    # Telegram
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    
    # Glassnode
    glassnode_api_key: Optional[str] = None

@dataclass
class TradingConfig:
    """Configuración de trading"""
    mode: TradingMode = TradingMode.PAPER
    max_positions: int = 10
    max_risk_per_trade: float = 0.02  # 2% del portfolio
    max_daily_risk: float = 0.10      # 10% del portfolio
    min_confidence_score: float = 70.0
    min_risk_reward_ratio: float = 2.0
    default_stop_loss_pct: float = 0.02  # 2%
    default_take_profit_pct: float = 0.04  # 4%
    
    # Timeframes para análisis
    analysis_timeframes: List[TimeFrame] = field(default_factory=lambda: [
        TimeFrame.M5, TimeFrame.M15, TimeFrame.H1, TimeFrame.H4, TimeFrame.D1
    ])
    
    # Símbolos a monitorear
    monitored_symbols: List[str] = field(default_factory=lambda: [
        "BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "MATICUSDT", 
        "LINKUSDT", "DOTUSDT", "AVAXUSDT", "ATOMUSDT", "NEARUSDT"
    ])
    
    # Estrategias activas
    active_strategies: List[str] = field(default_factory=lambda: [
        "scalping", "swing", "breakout", "mean_reversion"
    ])

@dataclass
class RiskConfig:
    """Configuración de gestión de riesgo"""
    # Stop Loss
    use_atr_stop_loss: bool = True
    atr_multiplier: float = 2.0
    max_stop_loss_pct: float = 0.05  # 5%
    
    # Take Profit
    use_trailing_stop: bool = True
    trailing_stop_pct: float = 0.01  # 1%
    tp1_pct: float = 0.02  # 2%
    tp2_pct: float = 0.04  # 4%
    tp3_pct: float = 0.08  # 8%
    
    # Posicionamiento
    use_kelly_criterion: bool = True
    max_position_size_pct: float = 0.10  # 10%
    min_position_size_pct: float = 0.01  # 1%
    
    # Diversificación
    max_correlation: float = 0.7
    max_sector_exposure: float = 0.3

@dataclass
class NotificationConfig:
    """Configuración de notificaciones"""
    # Telegram
    telegram_enabled: bool = True
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_send_signals: bool = True
    telegram_send_alerts: bool = True
    telegram_send_summary: bool = True
    telegram_summary_hour: int = 9  # 9 AM
    
    # Email
    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    email_to: List[str] = field(default_factory=list)
    
    # Webhook
    webhook_enabled: bool = False
    webhook_url: Optional[str] = None
    webhook_secret: Optional[str] = None

@dataclass
class MLConfig:
    """Configuración de Machine Learning"""
    enabled: bool = True
    model_retrain_hours: int = 24
    prediction_horizon_hours: int = 4
    min_training_samples: int = 1000
    
    # Modelos
    use_lstm: bool = True
    use_cnn: bool = True
    use_transformer: bool = False
    
    # Parámetros LSTM
    lstm_sequence_length: int = 60
    lstm_units: int = 50
    lstm_dropout: float = 0.2
    
    # Parámetros CNN
    cnn_filters: int = 32
    cnn_kernel_size: int = 3
    cnn_pool_size: int = 2

@dataclass
class SystemConfig:
    """Configuración del sistema"""
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_to_file: bool = True
    log_to_console: bool = True
    log_rotation_size_mb: int = 10
    log_retention_days: int = 30
    
    # Performance
    max_workers: int = 4
    cache_ttl_seconds: int = 300
    data_refresh_seconds: int = 60
    
    # Monitoreo
    health_check_interval: int = 30
    performance_metrics_interval: int = 300
    memory_usage_threshold: float = 0.8
    
    # Datos
    data_dir: str = "data"
    historical_data_days: int = 365
    backup_interval_hours: int = 6

class Settings:
    """
    Configuración principal de CryptoPulse Pro
    Centraliza toda la configuración del sistema
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/settings.yaml"
        self._load_config()
    
    def _load_config(self):
        """Cargar configuración desde archivo o variables de entorno"""
        
        # Configuración por defecto
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.trading = TradingConfig()
        self.risk = RiskConfig()
        self.notifications = NotificationConfig()
        self.ml = MLConfig()
        self.system = SystemConfig()
        
        # Cargar desde archivo si existe
        if os.path.exists(self.config_file):
            self._load_from_file()
        
        # Sobrescribir con variables de entorno
        self._load_from_env()
        
        # Validar configuración
        self._validate_config()
    
    def _load_from_file(self):
        """Cargar configuración desde archivo YAML"""
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Actualizar configuración
            if 'database' in config_data:
                self._update_dataclass(self.database, config_data['database'])
            if 'api' in config_data:
                self._update_dataclass(self.api, config_data['api'])
            if 'trading' in config_data:
                self._update_dataclass(self.trading, config_data['trading'])
            if 'risk' in config_data:
                self._update_dataclass(self.risk, config_data['risk'])
            if 'notifications' in config_data:
                self._update_dataclass(self.notifications, config_data['notifications'])
            if 'ml' in config_data:
                self._update_dataclass(self.ml, config_data['ml'])
            if 'system' in config_data:
                self._update_dataclass(self.system, config_data['system'])
                
        except Exception as e:
            print(f"Error loading config file: {e}")
    
    def _load_from_env(self):
        """Cargar configuración desde variables de entorno"""
        # Database
        if os.getenv('DATABASE_URL'):
            self.database.url = os.getenv('DATABASE_URL')
        
        # API Keys
        self.api.binance_api_key = os.getenv('BINANCE_API_KEY')
        self.api.binance_secret_key = os.getenv('BINANCE_SECRET_KEY')
        self.api.coingecko_api_key = os.getenv('COINGECKO_API_KEY')
        self.api.reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
        self.api.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
        self.api.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        self.api.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.api.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.api.glassnode_api_key = os.getenv('GLASSNODE_API_KEY')
        
        # Trading
        if os.getenv('TRADING_MODE'):
            self.trading.mode = TradingMode(os.getenv('TRADING_MODE'))
        
        # System
        if os.getenv('LOG_LEVEL'):
            self.system.log_level = LogLevel(os.getenv('LOG_LEVEL'))
    
    def _update_dataclass(self, obj, data: dict):
        """Actualizar dataclass con datos del diccionario"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def _validate_config(self):
        """Validar configuración"""
        # Validar trading mode
        if self.trading.mode == TradingMode.LIVE and not self.api.binance_api_key:
            raise ValueError("Binance API key required for live trading")
        
        # Validar notificaciones
        if self.notifications.telegram_enabled and not self.api.telegram_bot_token:
            raise ValueError("Telegram bot token required for notifications")
        
        # Validar risk config
        if hasattr(self.risk, 'max_risk_per_trade') and self.risk.max_risk_per_trade > 0.1:
            raise ValueError("Max risk per trade should not exceed 10%")
        
        # Validar timeframes
        if not self.trading.analysis_timeframes:
            raise ValueError("At least one timeframe must be specified")
    
    def save_config(self, file_path: Optional[str] = None):
        """Guardar configuración actual en archivo"""
        file_path = file_path or self.config_file
        
        config_data = {
            'database': self.database.__dict__,
            'api': self.api.__dict__,
            'trading': self.trading.__dict__,
            'risk': self.risk.__dict__,
            'notifications': self.notifications.__dict__,
            'ml': self.ml.__dict__,
            'system': self.system.__dict__
        }
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)
    
    def get_monitored_symbols(self) -> List[str]:
        """Obtener lista de símbolos monitoreados"""
        return self.trading.monitored_symbols
    
    def get_active_strategies(self) -> List[str]:
        """Obtener lista de estrategias activas"""
        return self.trading.active_strategies
    
    def is_live_trading(self) -> bool:
        """Verificar si está en modo live trading"""
        return self.trading.mode == TradingMode.LIVE
    
    def get_database_url(self) -> str:
        """Obtener URL de base de datos"""
        return self.database.url
    
    def get_log_level(self) -> str:
        """Obtener nivel de logging"""
        return self.system.log_level.value

# Instancia global de configuración
settings = Settings()
