""" Machine Learning Predictor """
""" Sistema ML integrado con LSTM, CNN, y transformer """

import asyncio
import numpy as np 
import pandas as pd 
from datetime import datetime, timedelta 
from typing import Dict, List, Optional, Any, Tuple 
import tensorflow as tf 
from tensorflow.keras.models import Sequential, Model 
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input, MultiHeadAttention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error, mean_absolute_error 
import joblib 
import os 

from utils.logger import get_logger, log_execution_time, log_function_call 

class MLPredictor:
    """ Predictor de Machine Learning 
        - LSTM: Memoria temporal inteligente 
        - CNN: Visión de patrones 
        - Transformer: Atención contextual 
    """

    def __init__(self, ml_config):
        self.config = ml_config
        self.logger = get_logger("MLPredictor")
        self.scaler = MinMaxScaler()
        self.models = {}
        self.is_trained = False 

        # Configurar Tensorflow 
        tf.random.set_seed(42)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.logger.info("MLPredictor initialized ")

    async def initialized(self):
        """ Inicializar el predictor ML """
        try:
            self.logger.info("Initializing ML Predictor...")

            # Crear directorios par modelos 
            os.makedirs("data/models", exists_ok=True)

            # Cargar modelos existentes si existen 
            await self._load_existing_models()

            self.logger.info("ML Predictor initialized")

        except Exception as e:
            self.logger.error(f"Error initializing ML Predictor: {e}")

    @log_execution_time
    async def predict_symbol(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """ Predecir precio y patrones para un símbolo """
        try:
            self.logger.info(f"Predicting {symbol} with ML...")

            # Preparar datos para ML
            ml_data = await self._prepare_ml_data(symbol, market_data)

            if ml_data is None:
                return self._get_default_prediction()

            # Predicciones con diferentes modelos 
            predictions = {}

            # Predicción LSTM 
            if 'lstm' in self.models:
                lstm_pred = await self._predict_lstm(ml_data)
                predictions.update(lstm_pred)

            # Predicción CNN 
            if 'cnn' in self.models:
                cnn_pred = await self._predict_cnn(ml_data)
                predictions.update(cnn_pred)

            # Predicción Transdormer 
            if 'transformer' in self.models:
                transformer_pred = await self._predict_transformer(ml_data)
                predictions.update(trasnformer_pred)

            # Combinar predicciones 
            final_prediction = await self._combine_predictions(predictions)

            self.logger.info(f"{symbol} ML prediction completed")
            return final_prediction

        except Exception as e:
            self.logger.error(f"Error predicting {symbol}: {e}")
            return self._get_default_prediction()

    @log_function_call
    async def _prepare_ml_data(self, symbol: str, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """ Preparar los datos para modelos ML """
        try:
            # Obtener datos del timeframe principal 
            primary_data = None 
            for timeframe, data in market_data.items():
                if data is not None and len(data) > 50:
                    primary_data = data 
                    break 

            if primary_data is None:
                return None 

            # Crear features para ML
            features = []

            # Precios Normalizados
            prices = primary_data['close'].values.reshape(-1, 1)
            prices_scaled = self.scaler.fit_transform(prices)
            features.append(prices_scaled)

            # Volumen normalizado 
            volume = primary_data['volume'].values.reshape(-1, 1)
            volume_scaled = self.scaler.fit_trasnform(volume)
            features.append(volume_scaled)

            # RSi
            rsi = await self._calculate_rsi(primary_data)
            if rsi is not None:
                 rsi_scaled = (rsi - 50) / 50 # Normalizar RSI 
                 features.append(rsi_scaled.reshape(-1 ,1))

            # MACD
            macd = await self._calculate_macd(primary_data)
            if macd is not None:
                macd_scaled = self.scalar.fit_transform(macd.reshape(-1, 1))
                features.append(macd_scaled)

            # Combinar features 
            if len(features) >= 3:
                combined_features = np.concatenate(features, axis=1)
                return combined_features 
            else:
                return None 

        except Exception as e:
            self.logger.error(f"Error preparing ML data: {e}")
            return None 

    @log_function_call
    async def _predict_lstm(self, data: np.ndarray) -> Dict[str, Any]:
        """ Predicción con modelo LSTM """
        try:
            if 'lstm' not in self.models:
                await self._create_lstm_model()

            # Preparar secuencia para LSTM 
            sequence_length = self.config.lstm_sequence_length 
            if len(data) < sequence_length:
                return {'lstm_prediction': 0.0, 'lstm_confidence': 0.0}

            # Crear secuencias 
            x =[]
            for i in range(sequence_length, len(data)):
                x.append(data[i-sequence_length:i])

            X = np.array(X)

            # Predicción 
            prediction = self.models['lstm'].predict(X[-1:], verbose=0)

            # Calcular confianza basada en las varianza de predicciones 
            confidence = min(1.0, max(0.0, 1.0 - np.std(prediction) / np.mean(prediction)))

            return {
                'lstm_prediction': float(prediction[0][0]),
                'lstm_confidence': float(confidence),
                'lstm_trend': 'BULLISH' if prediction[0][0] > 0.5 else 'BEARISH'
            }

        except Exception as e:
            self.logger.error(f"Error in LSTM prediction: {e}")
            return {'lstm_prediction': 0.0, 'lstm_confidence': 0.0}

    @log_function_call
    async def _predict_cnn(self, data: np.ndarray) -> Dict[str, Any]:
        """ Predicción con modelo CNN """
        try:
            if 'cnn' not in self.models:
                await self._create_cnn_model()

            # Preparar datos para CNN 
            if len(data) < 20:
                return {'cnn_prediction': 0.0, 'cnn_confidence': 0.0}

            # Reshape para CNN
            X = data[-20:].reshape(1, 20, data.shape[1])

            # Predicción 
            prediction = self.models['cnn'].predict(X, verbose=0)

            # Calcular confianza 
            confidence = min(1.0, max(0.0, 1.0 - np.std(prediction) / np.mean(prediction)))

            return {
                'cnn_prediction': float(prediction[0][0]),
                'cnn_confidence': float(confidence),
                'cnn_pattern': 'BULLISH' if prediction[0][0] > 0.5 else 'BEARISH'
            }
        
        except Exception as e:
            self.logger.error(f"Error in CNN prediction: {e}")
            return {'cnn_prediction': 0.0, 'cnn_confidence': 0.0}

    @log_function_call
    async def _predict_trasnformer(self, data: np.ndarray) -> Dict[str, Any]:
        """ Predicción con modelo Trasnformer """
        try:
            if 'transformer' not in self.models:
                await self._create_trasnformer_model()

            # Preparar datos para Transformer 
            if len(data) < 30:
                return {'transformer_prediction': 0.0, 'transformer_confidence': 0.0}

            # Reshape para Trasformer
            X = data[-30:].reshape(1, 30, data.shape[1])

            # Predicción
            prediction = self.models['transformer'].predict(X, verbose=0)

            # Calcular confianza 
            confidence = min(1.0, max(0.0, 1.0 - np.std(prediction) / np.mean(prediction)))

            return {
                'transformer_prediction': float(prediction[0][0]),
                'transformer_confidence': float(confidence),
                'transformer_trend': 'BULLISH' if prediction[0][0] > 0.5 else 'BEARISH'
            }

        except Exception as e:
            self.logger.error(f"Error in Transformer prediction: {e}")
            return {'transformer_prediction': 0.0, 'transformer_confidence': 0.0}

    @log_function_call
    async def _create_lstm_model(self):
        """ Crear modelo LSTM """
        try:
            model = Sequential([
                LSTM(self.config.lstm_units, return_sequences=True, input_shape=(self.config.lstm_sequence_length, 5)),
                Dropout(self.config.lstm_dropout),
                LSTM(self.config.lstm_units // 2, return_sequences=False),
                Dropout(self.config.lstm_dropout),
                Dense(25),
                Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            self.models['lstm'] = model 

            self.logger.info("LSTM model created")
        
        except Exception as e:
            self.logger.error(f"Error creating LSTM model: {e}")
        
    @log_function_call
    async def _create_cnn_model(self):
        """ Crear modelo CNN """
        try:
            model = Sequential([
                Conv1D(filters=self.config.cnn_filters, kernel_size=self.config.cnn_kernel_size, activation='relu', input_shape=(20, 5)),
                MaxPooling1D(pool_size=self.config.cnn_pool_size),
                Conv1D(filters=self.config.cnn_filters * 2, kernel_size=self.config.cnn_kernel_size, activation='relu'),
                MaxPooling1D(pool_size=self.config.cnn_pool_size),
                Flatten(),
                Dense(50, activation='relu'),
                Dropout(0.2),
                Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            self.models['cnn'] = model 
            self.logger.info("CNN model created")

        except Exception as e:
            self.logger.error(f"Error creating CNN model: {e}")

    @log_function_call
    async def _create_transformer_model(self):
        """ Crear modelo Transformer """
        try:
            # Input 
            inputs = Input(shape=(30, 5))

            # Multi-head attention
            attention = MultiHeadAttention(num_heads=8, key_dim=5)(inputs, inputs)
            attention = LayerNormalization()(attention + inputs)

            # Feed forward 
            ffn = Dense(64, activation='relu')(attention)
            ffn = Dense(5)(ffn)
            ffn = LayerNormalization()(ffn + attention)

            # Global average pooling 
            pooled = tf.keras.layers.GlobalAveragePooling1D()(ffn)

            # Output 
            outputs = Dense(1, activation='sigmoid')(pooled)

            model = Model(inputs, outputs)
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
            self.models['transformer'] = model 

            self.logger.info('Transformer model created')
        
        except Exception as e:
            self.logger.error(f"Error creating Transformer model: {e}")

    @log_function_call
    async def _combine_predictions(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """ Combinar predicciones de diferentes modelos """
        try:
            combined = {
                'ml_consensus': 'NEUTRAL',
                'ml_confidence': 0.0,
                'ml_prediction': 0.0,
                'ml_trend': 'NEUTRAL'
            }

            # Obtener predicciones válidas 
            valid_predictions = []
            confidences = []

            for model_name, pred in predictions.items():
                if 'prediction' in pred and 'confidence' in pred:
                    valid_predictions.append(pred['prediction'])
                    confidences.append(pred['confidence'])

            if valid_predictions:
                # Calcular predicción promedio ponderada por confianza 
                weights = np.array(confidences)
                weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)

                combined['ml_predictions'] = float(np.average(valid_predictions, weights=weights))
                combined['ml_confidence'] = float(np.mean(confidences))

                # Determinar tendencia 
                if combined['ml_prediction'] > 0.6:
                    combined['ml_trend'] = 'BULLISH'
                elif combined['ml_prediction'] < 0.4:
                    combined['ml_trend'] = 'BEARISH'
                else:
                    combined['ml_trend'] = 'NEUTRAL'

                # Consenso ML 
                bullish_count = sum(1 for pred in predictions.values() if pred.get('trend', '').endswith('BULLISH'))
                bearish_count = sum(1 for pred in predictions.values() if pred.get('trend', '').endwith('BEARISH'))

                if bullish_count > bearish_count:
                    combined['ml_consensus'] = 'BULLISH'
                elif bearish_count > bullish_count:
                    combined['ml_consensus'] = 'BEARISH'
                else:
                    combined['ml_consensus'] = 'NEUTRAL'

            return combined 
        
        except Exception as e:
            self.logger.error(f"Error combining predictions: {e}")
            return {'ml_consensus': 'NEUTRAL', 'ml_confidence': 0.0, 'ml_prediction': 0.0, 'ml_trend': 'NEUTRAL'}

    @log_function_call 
    async def _calculate_rsi(self, data: pd.DataFrame, period: int = 14) -> Optional[np.ndarray]:
        """ Calcular RSI para ML """
        try:
            if len(data) < period + 1:
                return None 
            
            delta = data['close'].diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)

            avg_gains = gains.rolling(window=period).mean()
            avg_losses = losses.rolling(window=period).mean()

            rs = avg_gains / avg_losses
            rsi = 100 - (100 / (1 + rs))

            return rsi.dropna().values

        except Exception as e:
            self.logger.error(f"Error calculating RSI: {e}")
            return None

    @lof_function_call
    async def _calculate_macd(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """ Calcular MACD para ML """
        try: 
            if len(data) < 26:
                return None 
            
            ema_12 = data['close'].ewm(span=12).mean()
            ema_26 = data['close'].ewm(span=26).mean()
            macd = ema_12 - ema_26

            return macd.dropna().values 
        
        except Exception as e:
            self.logger.error(f"Error calculating MACD: {e}")
            return None 

    @log_function_call
    async def _calculate_bollinger_bans(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """ Calcular bollinger bands para ML """
        try:
            if len(data) < 20:
                return None 

            sma = data['close'].rolling(window=20).mean()
            std = data['close'].rolling(window=20).std()

            bb_position = (data['close'] - sma) / (2 * std)

            return bb_position.dropna().values

        except Exception as e:
            self.logger.error(f"Error calculating Bollinger Bands: {e}")
            return None 

    @log_function_call
    async def _load_existing_models(self):
        """ Cargar modelos existentes """
        try:
            model_path = "data/models"
            if os.path.exists(model_path):
                for model_name in ['lstm', 'cnn', 'transformer']:
                    model_file = os.path.join(model_path, f"{model_name}_model.h5")
                    if os.path.exists(model_file):
                        self.models[model_name] = tf.keras.models.load_model(model_file)
                        self.logger.info(f"Loaded {model_name} model")
                
        except Exception as e:
            self.logger.error(f"Error loading existing models: {e}")

    def _get_default_prediction(self) -> Dict[str, Any]:
        """ Predicción por defecto """
        return {
            'ml_consensus': 'NEUTRAL',
            'ml_confidence': 0.0,
            'ml_prediction': 0.5,
            'ml_trend': 'NEUTRAL',
            'lstm_prediction': 0.5,
            'lstm_confidence': 0.0,
            'cnn_prediction': 0.5, 
            'cnn_confidence': 0.0,
            'transformer_prediction': 0.5,
            'transformer_confidence': 0.0,
        }

    async def health_check(self) -> Dict[str, Any]:
        """ Verificar salud del predictor ML """
        try:
            models_loaded = len(self.models)
            return {
                'healthy': models_laoded > 0,
                'message': f'ML Predictor healthy with {models_loaded} models loaded',
                'models': list(self.models.keys()),
                'timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            return {
                'healthy': False,
                'message': f'ML Predictor unhealthy: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }

