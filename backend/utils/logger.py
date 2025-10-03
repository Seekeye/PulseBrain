#!/usr/bin/env python3
"""
Sistema de Logging Avanzado para CryptoPulse Pro
Logging estructurado con diferentes niveles y rotación automática
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import json
import traceback

class CryptoPulseLogger:
    """
    Sistema de logging avanzado para CryptoPulse Pro
    - Logging estructurado con JSON
    - Rotación automática de archivos
    - Diferentes niveles de logging
    - Colores en consola
    - Métricas de performance
    """
    
    def __init__(self, name: str = "CryptoPulse", log_level: str = "INFO"):
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path("data/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar logger principal
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Evitar duplicación de handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Configurar todos los handlers de logging"""
        
        # 1. Handler para consola con colores
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(name)-15s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 2. Handler para archivo general
        general_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "cryptopulse.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        general_handler.setLevel(logging.DEBUG)
        general_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-15s | %(funcName)-20s | %(lineno)-4d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        general_handler.setFormatter(general_formatter)
        self.logger.addHandler(general_handler)
        
        # 3. Handler para errores críticos
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "errors.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(general_formatter)
        self.logger.addHandler(error_handler)
        
        # 4. Handler para trading (señales y operaciones)
        trading_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "trading.log",
            maxBytes=20*1024*1024,  # 20MB
            backupCount=10
        )
        trading_handler.setLevel(logging.INFO)
        trading_formatter = logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        trading_handler.setFormatter(trading_formatter)
        
        # Logger específico para trading
        self.trading_logger = logging.getLogger(f"{self.name}.trading")
        self.trading_logger.setLevel(logging.INFO)
        self.trading_logger.addHandler(trading_handler)
        self.trading_logger.propagate = False
        
        # 5. Handler para análisis técnico
        analysis_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "analysis.log",
            maxBytes=15*1024*1024,  # 15MB
            backupCount=7
        )
        analysis_handler.setLevel(logging.INFO)
        analysis_handler.setFormatter(general_formatter)
        
        self.analysis_logger = logging.getLogger(f"{self.name}.analysis")
        self.analysis_logger.setLevel(logging.INFO)
        self.analysis_logger.addHandler(analysis_handler)
        self.analysis_logger.propagate = False
        
        # 6. Handler para notificaciones
        notification_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "notifications.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        notification_handler.setLevel(logging.INFO)
        notification_handler.setFormatter(general_formatter)
        
        self.notification_logger = logging.getLogger(f"{self.name}.notifications")
        self.notification_logger.setLevel(logging.INFO)
        self.notification_logger.addHandler(notification_handler)
        self.notification_logger.propagate = False
    
    def info(self, message: str, **kwargs):
        """Log de información general"""
        self.logger.info(message, extra=kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log de debug"""
        self.logger.debug(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log de advertencia"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log de error con stack trace"""
        if exception:
            self.logger.error(f"{message} | Exception: {str(exception)}", extra=kwargs)
            self.logger.error(f"Stack trace: {traceback.format_exc()}")
        else:
            self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log crítico"""
        if exception:
            self.logger.critical(f"{message} | Exception: {str(exception)}", extra=kwargs)
            self.logger.critical(f"Stack trace: {traceback.format_exc()}")
        else:
            self.logger.critical(message, extra=kwargs)
    
    def trading_signal(self, signal_data: dict):
        """Log específico para señales de trading"""
        signal_json = json.dumps(signal_data, indent=2, default=str)
        self.trading_logger.info(f"TRADING_SIGNAL | {signal_json}")
    
    def trading_execution(self, execution_data: dict):
        """Log específico para ejecuciones de trading"""
        execution_json = json.dumps(execution_data, indent=2, default=str)
        self.trading_logger.info(f"TRADING_EXECUTION | {execution_json}")
    
    def analysis_result(self, analysis_data: dict):
        """Log específico para análisis técnico"""
        analysis_json = json.dumps(analysis_data, indent=2, default=str)
        self.analysis_logger.info(f"ANALYSIS_RESULT | {analysis_json}")
    
    def notification_sent(self, notification_data: dict):
        """Log específico para notificaciones"""
        notification_json = json.dumps(notification_data, indent=2, default=str)
        self.notification_logger.info(f"NOTIFICATION_SENT | {notification_json}")
    
    def performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log de métricas de performance"""
        self.logger.info(f"PERFORMANCE | {metric_name}: {value} {unit}")
    
    def system_status(self, status: str, details: dict = None):
        """Log de estado del sistema"""
        if details:
            details_json = json.dumps(details, indent=2, default=str)
            self.logger.info(f"SYSTEM_STATUS | {status} | {details_json}")
        else:
            self.logger.info(f"SYSTEM_STATUS | {status}")


class ColoredFormatter(logging.Formatter):
    """Formatter con colores para la consola"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{log_color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


# Instancia global del logger
logger = CryptoPulseLogger("CryptoPulsePro", "INFO")

# Funciones de conveniencia
def get_logger(name: str = "CryptoPulsePro") -> CryptoPulseLogger:
    """Obtener instancia del logger"""
    return CryptoPulseLogger(name)

def log_function_call(func):
    """Decorator para loggear llamadas a funciones"""
    def wrapper(*args, **kwargs):
        logger.debug(f"Calling function: {func.__name__} with args: {args}, kwargs: {kwargs}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Function {func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"Function {func.__name__} failed", exception=e)
            raise
    return wrapper

def log_execution_time(func):
    """Decorator para medir tiempo de ejecución"""
    import time
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.performance_metric(f"{func.__name__}_execution_time", execution_time, "seconds")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {execution_time:.2f}s", exception=e)
            raise
    return wrapper
