#!/usr/bin/env python3
"""
Adaptive ML System - CryptoPulse Pro
Sistema de Machine Learning que aprende constantemente de las señales
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
import os
from collections import deque
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger, log_execution_time, log_function_call

class AdaptiveMLSystem:
    """
    Sistema de ML que aprende constantemente de:
    - Señales enviadas vs resultados reales
    - Patrones de mercado que funcionan
    - Ajuste dinámico de pesos y parámetros
    - Mejora continua de precisión
    """
    
    def __init__(self):
        self.logger = get_logger("AdaptiveML")
        
        # Modelos de ML adaptativos
        self.models = {
            'price_predictor': RandomForestRegressor(n_estimators=100, random_state=42),
            'signal_quality': MLPRegressor(hidden_layer_sizes=(50, 25), random_state=42),
            'market_regime': RandomForestRegressor(n_estimators=50, random_state=42)
        }
        
        # Escaladores
        self.scalers = {
            'price_predictor': StandardScaler(),
            'signal_quality': StandardScaler(),
            'market_regime': StandardScaler()
        }
        
        # Datos de entrenamiento
        self.training_data = {
            'features': deque(maxlen=10000),  # Últimos 10k registros
            'targets': deque(maxlen=10000),
            'signals': deque(maxlen=1000),    # Últimas 1k señales
            'outcomes': deque(maxlen=1000)    # Resultados de señales
        }
        
        # Métricas de rendimiento
        self.performance_metrics = {
            'accuracy_history': deque(maxlen=100),
            'precision_history': deque(maxlen=100),
            'recall_history': deque(maxlen=100),
            'f1_history': deque(maxlen=100)
        }
        
        # Configuración de aprendizaje
        self.learning_config = {
            'retrain_frequency': 24,  # Horas
            'min_samples_for_training': 100,
            'learning_rate': 0.01,
            'adaptation_threshold': 0.05  # 5% de mejora mínima
        }
        
        self.last_retrain = datetime.now()
        self.model_version = 1
        
        self.logger.info("🧠 AdaptiveMLSystem initialized")
    
    async def initialize(self):
        """Inicializar el sistema de ML adaptativo"""
        try:
            self.logger.info("🔧 Initializing Adaptive ML System...")
            
            # Cargar modelos existentes si existen
            await self._load_models()
            
            # Cargar datos históricos
            await self._load_historical_data()
            
            # Entrenar modelos iniciales si hay datos
            if len(self.training_data['features']) >= self.learning_config['min_samples_for_training']:
                await self._retrain_models()
            
            self.logger.info("✅ Adaptive ML System initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing Adaptive ML System: {e}")
    
    async def _load_models(self):
        """Cargar modelos pre-entrenados si existen"""
        try:
            models_dir = "models/adaptive_ml"
            os.makedirs(models_dir, exist_ok=True)
            
            for model_name in self.models.keys():
                model_path = f"{models_dir}/{model_name}_v{self.model_version}.joblib"
                scaler_path = f"{models_dir}/{model_name}_scaler_v{self.model_version}.joblib"
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    self.models[model_name] = joblib.load(model_path)
                    self.scalers[model_name] = joblib.load(scaler_path)
                    self.logger.info(f"✅ Loaded {model_name} model v{self.model_version}")
                    
        except Exception as e:
            self.logger.warning(f"Could not load existing models: {e}")
    
    async def _save_models(self):
        """Guardar modelos entrenados"""
        try:
            models_dir = "models/adaptive_ml"
            os.makedirs(models_dir, exist_ok=True)
            
            for model_name in self.models.keys():
                model_path = f"{models_dir}/{model_name}_v{self.model_version}.joblib"
                scaler_path = f"{models_dir}/{model_name}_scaler_v{self.model_version}.joblib"
                
                joblib.dump(self.models[model_name], model_path)
                joblib.dump(self.scalers[model_name], scaler_path)
                
            self.logger.info(f"✅ Saved models v{self.model_version}")
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    async def _load_historical_data(self):
        """Cargar datos históricos para entrenamiento"""
        try:
            # En un sistema real, esto cargaría desde la base de datos
            # Por ahora, simulamos algunos datos históricos
            self.logger.info("📊 Loading historical data...")
            
            # Simular datos históricos
            np.random.seed(42)
            for i in range(500):  # 500 muestras históricas
                features = np.random.randn(20)  # 20 características
                target = np.random.randn(1)[0]  # Precio objetivo
                
                self.training_data['features'].append(features)
                self.training_data['targets'].append(target)
            
            self.logger.info(f"✅ Loaded {len(self.training_data['features'])} historical samples")
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
    
    @log_execution_time
    async def add_signal_feedback(self, signal_data: Dict[str, Any], outcome: Dict[str, Any]):
        """Agregar retroalimentación de una señal"""
        try:
            # Extraer características de la señal
            features = self._extract_signal_features(signal_data)
            
            # Calcular resultado de la señal
            signal_outcome = self._calculate_signal_outcome(signal_data, outcome)
            
            # Agregar a datos de entrenamiento
            self.training_data['features'].append(features)
            self.training_data['targets'].append(signal_outcome)
            self.training_data['signals'].append(signal_data)
            self.training_data['outcomes'].append(outcome)
            
            self.logger.info(f"📝 Added signal feedback: {signal_data.get('symbol', 'UNKNOWN')}")
            
            # Verificar si es hora de re-entrenar
            if self._should_retrain():
                await self._retrain_models()
            
        except Exception as e:
            self.logger.error(f"Error adding signal feedback: {e}")
    
    def _extract_signal_features(self, signal_data: Dict[str, Any]) -> np.ndarray:
        """Extraer características de una señal para ML"""
        try:
            features = []
            
            # Características técnicas
            features.extend([
                signal_data.get('confidence_score', 0) / 100,
                signal_data.get('coherence_score', 0) / 100,
                1 if signal_data.get('signal_type') == 'BUY' else 0,
                1 if signal_data.get('signal_type') == 'SELL' else 0,
            ])
            
            # Características de precio
            entry_price = signal_data.get('entry_price', 0)
            stop_loss = signal_data.get('stop_loss', 0)
            take_profit_1 = signal_data.get('take_profit_1', 0)
            
            if entry_price > 0:
                features.extend([
                    (take_profit_1 - entry_price) / entry_price,  # Potencial ganancia
                    (entry_price - stop_loss) / entry_price,       # Riesgo
                    (take_profit_1 - stop_loss) / entry_price,     # R/R ratio
                ])
            else:
                features.extend([0, 0, 0])
            
            # Características temporales
            timestamp = signal_data.get('timestamp', datetime.now().isoformat())
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            features.extend([
                dt.hour / 24,  # Hora del día normalizada
                dt.weekday() / 7,  # Día de la semana normalizado
            ])
            
            # Rellenar con ceros si faltan características
            while len(features) < 20:
                features.append(0)
            
            return np.array(features[:20])  # Asegurar 20 características
            
        except Exception as e:
            self.logger.error(f"Error extracting signal features: {e}")
            return np.zeros(20)
    
    def _calculate_signal_outcome(self, signal_data: Dict[str, Any], outcome: Dict[str, Any]) -> float:
        """Calcular el resultado de una señal"""
        try:
            # Calcular PnL de la señal
            entry_price = signal_data.get('entry_price', 0)
            exit_price = outcome.get('exit_price', entry_price)
            signal_type = signal_data.get('signal_type', 'HOLD')
            
            if signal_type == 'BUY':
                pnl = (exit_price - entry_price) / entry_price
            elif signal_type == 'SELL':
                pnl = (entry_price - exit_price) / entry_price
            else:
                pnl = 0
            
            # Normalizar PnL entre -1 y 1
            return np.clip(pnl, -1, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating signal outcome: {e}")
            return 0
    
    def _should_retrain(self) -> bool:
        """Determinar si es hora de re-entrenar los modelos"""
        try:
            # Re-entrenar si han pasado las horas configuradas
            time_since_retrain = datetime.now() - self.last_retrain
            if time_since_retrain.total_seconds() / 3600 >= self.learning_config['retrain_frequency']:
                return True
            
            # Re-entrenar si hay suficientes datos nuevos
            if len(self.training_data['features']) >= self.learning_config['min_samples_for_training']:
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking retrain condition: {e}")
            return False
    
    @log_execution_time
    async def _retrain_models(self):
        """Re-entrenar todos los modelos con datos actualizados"""
        try:
            self.logger.info("🔄 Retraining ML models...")
            
            if len(self.training_data['features']) < self.learning_config['min_samples_for_training']:
                self.logger.warning("Not enough data for retraining")
                return
            
            # Convertir datos a arrays
            X = np.array(list(self.training_data['features']))
            y = np.array(list(self.training_data['targets']))
            
            # Dividir datos
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Entrenar cada modelo
            for model_name, model in self.models.items():
                try:
                    # Escalar características
                    X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                    X_test_scaled = self.scalers[model_name].transform(X_test)
                    
                    # Entrenar modelo
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluar rendimiento
                    y_pred = model.predict(X_test_scaled)
                    mse = np.mean((y_test - y_pred) ** 2)
                    
                    self.logger.info(f"✅ {model_name} retrained - MSE: {mse:.4f}")
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_name}: {e}")
            
            # Actualizar versión y guardar
            self.model_version += 1
            self.last_retrain = datetime.now()
            await self._save_models()
            
            self.logger.info("✅ All models retrained successfully")
            
        except Exception as e:
            self.logger.error(f"Error retraining models: {e}")
    
    @log_execution_time
    async def predict_signal_quality(self, signal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predecir la calidad de una señal usando ML"""
        try:
            # Extraer características
            features = self._extract_signal_features(signal_data)
            features_scaled = self.scalers['signal_quality'].transform([features])
            
            # Predecir calidad
            quality_score = self.models['signal_quality'].predict(features_scaled)[0]
            
            # Normalizar entre 0 y 1
            quality_score = np.clip(quality_score, 0, 1)
            
            return {
                'ml_quality_score': quality_score,
                'ml_confidence': min(quality_score * 100, 100),
                'ml_recommendation': 'STRONG_BUY' if quality_score > 0.8 else 'BUY' if quality_score > 0.6 else 'HOLD',
                'model_version': self.model_version,
                'training_samples': len(self.training_data['features'])
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting signal quality: {e}")
            return {
                'ml_quality_score': 0.5,
                'ml_confidence': 50,
                'ml_recommendation': 'HOLD',
                'model_version': 0,
                'training_samples': 0
            }
    
    @log_execution_time
    async def get_adaptive_insights(self) -> Dict[str, Any]:
        """Obtener insights del sistema adaptativo"""
        try:
            insights = {
                'model_version': self.model_version,
                'training_samples': len(self.training_data['features']),
                'last_retrain': self.last_retrain.isoformat(),
                'next_retrain': (self.last_retrain + timedelta(hours=self.learning_config['retrain_frequency'])).isoformat(),
                'learning_status': 'ACTIVE' if len(self.training_data['features']) > 0 else 'INACTIVE',
                'performance_trend': 'IMPROVING' if len(self.performance_metrics['accuracy_history']) > 0 else 'UNKNOWN'
            }
            
            # Calcular métricas de rendimiento si hay datos
            if len(self.performance_metrics['accuracy_history']) > 0:
                recent_accuracy = list(self.performance_metrics['accuracy_history'])[-5:]
                insights['recent_accuracy'] = np.mean(recent_accuracy)
                insights['accuracy_trend'] = 'IMPROVING' if len(recent_accuracy) > 1 and recent_accuracy[-1] > recent_accuracy[0] else 'STABLE'
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error getting adaptive insights: {e}")
            return {
                'model_version': 0,
                'training_samples': 0,
                'learning_status': 'ERROR',
                'performance_trend': 'UNKNOWN'
            }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de rendimiento del sistema"""
        try:
            metrics = {}
            
            for metric_name, history in self.performance_metrics.items():
                if len(history) > 0:
                    metrics[metric_name] = {
                        'current': history[-1],
                        'average': np.mean(list(history)),
                        'trend': 'IMPROVING' if len(history) > 1 and history[-1] > history[0] else 'STABLE',
                        'samples': len(history)
                    }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {}

if __name__ == "__main__":
    async def test_adaptive_ml():
        ml_system = AdaptiveMLSystem()
        await ml_system.initialize()
        
        # Simular una señal
        signal_data = {
            'symbol': 'BTCUSDT',
            'signal_type': 'BUY',
            'entry_price': 50000,
            'confidence_score': 85,
            'coherence_score': 80,
            'timestamp': datetime.now().isoformat()
        }
        
        # Predecir calidad
        prediction = await ml_system.predict_signal_quality(signal_data)
        print(f"ML Prediction: {prediction}")
        
        # Simular resultado
        outcome = {
            'exit_price': 52000,
            'pnl_percentage': 4.0
        }
        
        # Agregar retroalimentación
        await ml_system.add_signal_feedback(signal_data, outcome)
        
        # Obtener insights
        insights = await ml_system.get_adaptive_insights()
        print(f"Insights: {insights}")
    
    asyncio.run(test_adaptive_ml())
