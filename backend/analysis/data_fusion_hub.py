#!/usr/bin/env python3
"""
Data Fusion Hub - CryptoPulse Pro
CEREBRO INTELIGENTE del sistema de trading
- Validación cruzada automática
- Detección de inconsistencias
- Aprendizaje continuo
- Coherencia total entre fuentes
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import pickle
from pathlib import Path

from utils.logger import get_logger, log_execution_time, log_function_call

class DataSource(Enum):
    """Fuentes de datos"""
    TECHNICAL = "technical"
    ML = "ml"
    NEWS = "news"
    ONCHAIN = "onchain"
    SENTIMENT = "sentiment"

class CoherenceLevel(Enum):
    """Niveles de coherencia"""
    VERY_HIGH = "very_high"  # 0.9+
    HIGH = "high"           # 0.8-0.9
    MEDIUM = "medium"       # 0.6-0.8
    LOW = "low"             # 0.4-0.6
    VERY_LOW = "very_low"   # <0.4

@dataclass
class DataSourceInfo:
    """Información de una fuente de datos"""
    source: DataSource
    data: Dict[str, Any]
    confidence: float
    timestamp: datetime
    reliability_score: float
    last_update: datetime

@dataclass
class CoherenceResult:
    """Resultado de validación de coherencia"""
    overall_coherence: float
    coherence_level: CoherenceLevel
    source_coherences: Dict[str, float]
    inconsistencies: List[str]
    warnings: List[str]
    recommendations: List[str]

class DataFusionHub:
    """
    CEREBRO INTELIGENTE del sistema de trading
    - Validación cruzada automática
    - Detección de inconsistencias inteligente
    - Aprendizaje continuo de patrones
    - Coherencia total entre fuentes
    """
    
    def __init__(self):
        self.logger = get_logger("DataFusionHub")
        
        # Configuración de coherencia
        self.coherence_thresholds = {
            'very_high': 0.9,
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
        
        # Pesos de fuentes de datos
        self.source_weights = {
            DataSource.TECHNICAL: 0.25,
            DataSource.ML: 0.20,
            DataSource.NEWS: 0.20,
            DataSource.ONCHAIN: 0.20,
            DataSource.SENTIMENT: 0.15
        }
        
        # Historial de coherencia para aprendizaje
        self.coherence_history = []
        
        # Cache de datos para eficiencia
        self.data_cache = {}
        self.cache_ttl = 300  # 5 minutos
        
        self.logger.info("🧠 DataFusionHub initialized (INTELLIGENT BRAIN)")
    
    async def initialize(self):
        """Inicializar el Data Fusion Hub"""
        try:
            self.logger.info("🔧 Initializing INTELLIGENT Data Fusion Hub...")
            
            # Cargar historial de coherencia
            await self._load_coherence_history()
            
            self.logger.info("✅ INTELLIGENT Data Fusion Hub initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing INTELLIGENT Data Fusion Hub: {e}")
    
    @log_execution_time
    async def fuse_all_data(self, technical: Dict[str, Any], ml: Dict[str, Any], 
                           news: Dict[str, Any], onchain: Dict[str, Any], 
                           sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Fusionar TODOS los datos de forma INTELIGENTE"""
        try:
            self.logger.info("🧠 INTELLIGENT data fusion starting...")
            
            # Crear fuentes de datos con metadatos
            sources = {
                DataSource.TECHNICAL: DataSourceInfo(
                    source=DataSource.TECHNICAL,
                    data=technical,
                    confidence=technical.get('confidence', 0.8),
                    timestamp=datetime.utcnow(),
                    reliability_score=0.9,  # Técnico es muy confiable
                    last_update=datetime.utcnow()
                ),
                DataSource.ML: DataSourceInfo(
                    source=DataSource.ML,
                    data=ml,
                    confidence=ml.get('ml_confidence', 0.7),
                    timestamp=datetime.utcnow(),
                    reliability_score=0.8,  # ML es confiable
                    last_update=datetime.utcnow()
                ),
                DataSource.NEWS: DataSourceInfo(
                    source=DataSource.NEWS,
                    data=news,
                    confidence=news.get('confidence', 0.6),
                    timestamp=datetime.utcnow(),
                    reliability_score=0.7,  # Noticias son moderadamente confiables
                    last_update=datetime.utcnow()
                ),
                DataSource.ONCHAIN: DataSourceInfo(
                    source=DataSource.ONCHAIN,
                    data=onchain,
                    confidence=onchain.get('onchain_confidence', 0.8),
                    timestamp=datetime.utcnow(),
                    reliability_score=0.9,  # On-chain es muy confiable
                    last_update=datetime.utcnow()
                ),
                DataSource.SENTIMENT: DataSourceInfo(
                    source=DataSource.SENTIMENT,
                    data=sentiment,
                    confidence=sentiment.get('social_confidence', 0.6),
                    timestamp=datetime.utcnow(),
                    reliability_score=0.6,  # Sentimiento es menos confiable
                    last_update=datetime.utcnow()
                )
            }
            
            # VALIDACIÓN CRUZADA INTELIGENTE
            coherence_result = await self._intelligent_cross_validation(sources)
            
            # DETECCIÓN DE INCONSISTENCIAS
            inconsistencies = await self._detect_intelligent_inconsistencies(sources)
            
            # ANÁLISIS DE COHERENCIA TEMPORAL
            temporal_analysis = await self._analyze_temporal_coherence(sources)
            
            # GENERACIÓN DE SEÑAL INTELIGENTE
            final_signal = await self._generate_intelligent_signal(
                sources, coherence_result, inconsistencies, temporal_analysis
            )
            
            # APRENDIZAJE CONTINUO
            await self._continuous_learning(sources, final_signal, coherence_result)
            
            # ACTUALIZAR CACHE
            await self._update_cache(sources, final_signal)
            
            self.logger.info("✅ INTELLIGENT data fusion completed")
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Error in INTELLIGENT data fusion: {e}")
            return self._get_default_signal()
    
    @log_function_call
    async def _intelligent_cross_validation(self, sources: Dict[DataSource, DataSourceInfo]) -> CoherenceResult:
        """Validación cruzada INTELIGENTE entre fuentes"""
        try:
            source_coherences = {}
            inconsistencies = []
            warnings = []
            recommendations = []
            
            # Coherencia Técnico + ML
            tech_ml_coherence = await self._validate_tech_ml_intelligent(
                sources[DataSource.TECHNICAL], sources[DataSource.ML]
            )
            source_coherences['tech_ml'] = tech_ml_coherence
            if tech_ml_coherence < 0.6:
                inconsistencies.append("TECH_ML_MISMATCH")
                warnings.append("Technical analysis and ML predictions don't align")
                recommendations.append("Check ML model accuracy and technical indicators")
            
            # Coherencia ML + On-chain
            ml_onchain_coherence = await self._validate_ml_onchain_intelligent(
                sources[DataSource.ML], sources[DataSource.ONCHAIN]
            )
            source_coherences['ml_onchain'] = ml_onchain_coherence
            if ml_onchain_coherence < 0.6:
                inconsistencies.append("ML_ONCHAIN_MISMATCH")
                warnings.append("ML predictions and on-chain data don't align")
                recommendations.append("Verify on-chain data quality and ML model")
            
            # Coherencia Noticias + Sentimiento
            news_sentiment_coherence = await self._validate_news_sentiment_intelligent(
                sources[DataSource.NEWS], sources[DataSource.SENTIMENT]
            )
            source_coherences['news_sentiment'] = news_sentiment_coherence
            if news_sentiment_coherence < 0.5:
                inconsistencies.append("NEWS_SENTIMENT_MISMATCH")
                warnings.append("News sentiment and social sentiment don't align")
                recommendations.append("Check news sources and social media data quality")
            
            # Coherencia On-chain + Técnico
            onchain_tech_coherence = await self._validate_onchain_tech_intelligent(
                sources[DataSource.ONCHAIN], sources[DataSource.TECHNICAL]
            )
            source_coherences['onchain_tech'] = onchain_tech_coherence
            if onchain_tech_coherence < 0.7:
                inconsistencies.append("ONCHAIN_TECH_MISMATCH")
                warnings.append("On-chain data and technical analysis don't align")
                recommendations.append("Verify on-chain data accuracy and technical indicators")
            
            # Coherencia general INTELIGENTE
            overall_coherence = await self._calculate_intelligent_coherence(
                source_coherences, sources
            )
            
            # Determinar nivel de coherencia
            coherence_level = self._get_coherence_level(overall_coherence)
            
            return CoherenceResult(
                overall_coherence=overall_coherence,
                coherence_level=coherence_level,
                source_coherences=source_coherences,
                inconsistencies=inconsistencies,
                warnings=warnings,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Error in intelligent cross validation: {e}")
            return CoherenceResult(0.0, CoherenceLevel.VERY_LOW, {}, [], [], [])
    
    @log_function_call
    async def _validate_tech_ml_intelligent(self, technical: DataSourceInfo, ml: DataSourceInfo) -> float:
        """Validación INTELIGENTE entre análisis técnico y ML"""
        try:
            tech_trend = technical.data.get('trend', 'NEUTRAL')
            ml_trend = ml.data.get('ml_trend', 'NEUTRAL')
            
            # Mapear tendencias a valores numéricos
            trend_values = {'BULLISH': 1, 'NEUTRAL': 0, 'BEARISH': -1}
            tech_value = trend_values.get(tech_trend, 0)
            ml_value = trend_values.get(ml_trend, 0)
            
            # Coherencia básica
            basic_coherence = 1.0 - abs(tech_value - ml_value) / 2.0
            
            # Coherencia de confianza
            confidence_coherence = 1.0 - abs(technical.confidence - ml.confidence)
            
            # Coherencia temporal (si los datos son recientes)
            time_diff = abs((technical.timestamp - ml.timestamp).total_seconds())
            temporal_coherence = 1.0 - min(1.0, time_diff / 3600)  # 1 hora = 0 coherencia
            
            # Coherencia ponderada
            weighted_coherence = (
                basic_coherence * 0.5 +
                confidence_coherence * 0.3 +
                temporal_coherence * 0.2
            )
            
            return weighted_coherence
            
        except Exception as e:
            self.logger.error(f"Error validating tech-ML: {e}")
            return 0.5
    
    @log_function_call
    async def _validate_ml_onchain_intelligent(self, ml: DataSourceInfo, onchain: DataSourceInfo) -> float:
        """Validación INTELIGENTE entre ML y on-chain"""
        try:
            ml_prediction = ml.data.get('ml_prediction', 0.5)
            onchain_sentiment_raw = onchain.data.get('onchain_sentiment', 0.5)
            
            # Convertir onchain_sentiment a número si es string
            if isinstance(onchain_sentiment_raw, str):
                sentiment_map = {'BULLISH': 0.8, 'BEARISH': 0.2, 'NEUTRAL': 0.5}
                onchain_sentiment = sentiment_map.get(onchain_sentiment_raw.upper(), 0.5)
            else:
                onchain_sentiment = float(onchain_sentiment_raw)
            
            # Coherencia de predicción
            prediction_coherence = 1.0 - abs(ml_prediction - onchain_sentiment)
            
            # Coherencia de confianza
            confidence_coherence = 1.0 - abs(ml.confidence - onchain.confidence)
            
            # Coherencia de whale activity vs ML
            whale_activity = onchain.data.get('whale_activity', 0.5)
            ml_confidence = ml.data.get('ml_confidence', 0.5)
            whale_ml_coherence = 1.0 - abs(whale_activity - ml_confidence)
            
            # Coherencia ponderada
            weighted_coherence = (
                prediction_coherence * 0.4 +
                confidence_coherence * 0.3 +
                whale_ml_coherence * 0.3
            )
            
            return weighted_coherence
            
        except Exception as e:
            self.logger.error(f"Error validating ML-onchain: {e}")
            return 0.5
    
    @log_function_call
    async def _validate_news_sentiment_intelligent(self, news: DataSourceInfo, sentiment: DataSourceInfo) -> float:
        """Validación INTELIGENTE entre noticias y sentimiento"""
        try:
            # Convertir sentimientos a números
            news_sentiment_str = news.data.get('overall_sentiment', 'NEUTRAL')
            social_sentiment_str = sentiment.data.get('overall_sentiment', 'NEUTRAL')
            
            sentiment_map = {'POSITIVE': 1.0, 'NEUTRAL': 0.0, 'NEGATIVE': -1.0}
            news_sentiment = sentiment_map.get(news_sentiment_str, 0.0)
            social_sentiment = sentiment_map.get(social_sentiment_str, 0.0)
            
            # Coherencia de sentimiento
            sentiment_coherence = 1.0 - abs(news_sentiment - social_sentiment) / 2.0
            
            # Coherencia de impacto
            news_impact = news.data.get('news_impact_score', 0.0)
            social_confidence = sentiment.data.get('social_confidence', 0.0)
            impact_coherence = 1.0 - abs(news_impact - social_confidence)
            
            # Coherencia de Fear & Greed
            fear_greed = news.data.get('fear_greed_index', 50.0)
            fear_greed_normalized = (fear_greed - 50) / 50
            social_sentiment_normalized = social_sentiment
            fear_greed_coherence = 1.0 - abs(fear_greed_normalized - social_sentiment_normalized) / 2.0
            
            # Coherencia ponderada
            weighted_coherence = (
                sentiment_coherence * 0.5 +
                impact_coherence * 0.3 +
                fear_greed_coherence * 0.2
            )
            
            return weighted_coherence
            
        except Exception as e:
            self.logger.error(f"Error validating news-sentiment: {e}")
            return 0.5
    
    @log_function_call
    async def _validate_onchain_tech_intelligent(self, onchain: DataSourceInfo, technical: DataSourceInfo) -> float:
        """Validación INTELIGENTE entre on-chain y técnico"""
        try:
            # Coherencia de tendencia
            onchain_trend = onchain.data.get('onchain_trend', 'NEUTRAL')
            tech_trend = technical.data.get('trend', 'NEUTRAL')
            
            trend_values = {'BULLISH': 1, 'NEUTRAL': 0, 'BEARISH': -1}
            onchain_value = trend_values.get(onchain_trend, 0)
            tech_value = trend_values.get(tech_trend, 0)
            
            trend_coherence = 1.0 - abs(onchain_value - tech_value) / 2.0
            
            # Coherencia de volumen
            onchain_activity = onchain.data.get('network_activity', 0.5)
            tech_volume = technical.data.get('volume_score', 0.5)
            volume_coherence = 1.0 - abs(onchain_activity - tech_volume)
            
            # Coherencia de whale activity vs momentum
            whale_activity = onchain.data.get('whale_activity', 0.5)
            tech_momentum = technical.data.get('momentum_score', 0.5)
            whale_momentum_coherence = 1.0 - abs(whale_activity - tech_momentum)
            
            # Coherencia ponderada
            weighted_coherence = (
                trend_coherence * 0.4 +
                volume_coherence * 0.3 +
                whale_momentum_coherence * 0.3
            )
            
            return weighted_coherence
            
        except Exception as e:
            self.logger.error(f"Error validating onchain-tech: {e}")
            return 0.5
    
    @log_function_call
    async def _calculate_intelligent_coherence(self, source_coherences: Dict[str, float], 
                                             sources: Dict[DataSource, DataSourceInfo]) -> float:
        """Calcular coherencia general INTELIGENTE"""
        try:
            # Coherencia promedio ponderada
            weighted_coherence = 0.0
            total_weight = 0.0
            
            for coherence_name, coherence_value in source_coherences.items():
                # Determinar peso basado en la importancia de la coherencia
                if 'tech_ml' in coherence_name:
                    weight = 0.3  # Muy importante
                elif 'ml_onchain' in coherence_name:
                    weight = 0.25  # Importante
                elif 'news_sentiment' in coherence_name:
                    weight = 0.2   # Moderadamente importante
                elif 'onchain_tech' in coherence_name:
                    weight = 0.25  # Importante
                else:
                    weight = 0.1   # Menos importante
                
                weighted_coherence += coherence_value * weight
                total_weight += weight
            
            # Normalizar
            if total_weight > 0:
                overall_coherence = weighted_coherence / total_weight
            else:
                overall_coherence = 0.5
            
            # Ajustar por confianza de fuentes
            confidence_adjustment = 0.0
            for source in sources.values():
                confidence_adjustment += source.confidence * source.reliability_score
            
            confidence_adjustment /= len(sources)
            
            # Coherencia final
            final_coherence = overall_coherence * 0.7 + confidence_adjustment * 0.3
            
            return min(1.0, max(0.0, final_coherence))
            
        except Exception as e:
            self.logger.error(f"Error calculating intelligent coherence: {e}")
            return 0.5
    
    @log_function_call
    async def _detect_intelligent_inconsistencies(self, sources: Dict[DataSource, DataSourceInfo]) -> List[str]:
        """Detección INTELIGENTE de inconsistencias"""
        try:
            inconsistencies = []
            
            # Inconsistencia 1: Técnico vs ML
            tech_trend = sources[DataSource.TECHNICAL].data.get('trend', 'NEUTRAL')
            ml_trend = sources[DataSource.ML].data.get('ml_trend', 'NEUTRAL')
            if tech_trend != ml_trend and tech_trend != 'NEUTRAL' and ml_trend != 'NEUTRAL':
                inconsistencies.append("TECH_ML_TREND_MISMATCH")
            
            # Inconsistencia 2: ML vs On-chain
            ml_prediction = sources[DataSource.ML].data.get('ml_prediction', 0.5)
            onchain_sentiment_raw = sources[DataSource.ONCHAIN].data.get('onchain_sentiment', 0.5)
            
            # Convertir onchain_sentiment a número si es string
            if isinstance(onchain_sentiment_raw, str):
                sentiment_map = {'BULLISH': 0.8, 'BEARISH': 0.2, 'NEUTRAL': 0.5}
                onchain_sentiment = sentiment_map.get(onchain_sentiment_raw.upper(), 0.5)
            else:
                onchain_sentiment = float(onchain_sentiment_raw)
            
            if abs(ml_prediction - onchain_sentiment) > 0.4:
                inconsistencies.append("ML_ONCHAIN_PREDICTION_MISMATCH")
            
            # Inconsistencia 3: Noticias vs Sentimiento
            news_sentiment_str = sources[DataSource.NEWS].data.get('overall_sentiment', 'NEUTRAL')
            social_sentiment_str = sources[DataSource.SENTIMENT].data.get('overall_sentiment', 'NEUTRAL')
            
            sentiment_map = {'POSITIVE': 1.0, 'NEUTRAL': 0.0, 'NEGATIVE': -1.0}
            news_sentiment = sentiment_map.get(news_sentiment_str, 0.0)
            social_sentiment = sentiment_map.get(social_sentiment_str, 0.0)
            
            if abs(news_sentiment - social_sentiment) > 0.5:
                inconsistencies.append("NEWS_SOCIAL_SENTIMENT_MISMATCH")
            
            # Inconsistencia 4: On-chain vs Técnico
            onchain_trend = sources[DataSource.ONCHAIN].data.get('onchain_trend', 'NEUTRAL')
            tech_trend = sources[DataSource.TECHNICAL].data.get('trend', 'NEUTRAL')
            if onchain_trend != tech_trend and onchain_trend != 'NEUTRAL' and tech_trend != 'NEUTRAL':
                inconsistencies.append("ONCHAIN_TECH_TREND_MISMATCH")
            
            # Inconsistencia 5: Confianza vs Datos
            for source_name, source_info in sources.items():
                if source_info.confidence > 0.8 and source_info.reliability_score < 0.6:
                    inconsistencies.append(f"HIGH_CONFIDENCE_LOW_RELIABILITY_{source_name.value.upper()}")
            
            return inconsistencies
            
        except Exception as e:
            self.logger.error(f"Error detecting intelligent inconsistencies: {e}")
            return []
    
    @log_function_call
    async def _analyze_temporal_coherence(self, sources: Dict[DataSource, DataSourceInfo]) -> Dict[str, Any]:
        """Análisis de coherencia temporal"""
        try:
            temporal_analysis = {
                'temporal_coherence': 0.0,
                'data_freshness': 0.0,
                'temporal_warnings': []
            }
            
            # Calcular frescura de datos
            now = datetime.utcnow()
            freshness_scores = []
            
            for source in sources.values():
                time_diff = (now - source.timestamp).total_seconds()
                freshness = max(0.0, 1.0 - time_diff / 3600)  # 1 hora = 0 frescura
                freshness_scores.append(freshness)
                
                if time_diff > 1800:  # 30 minutos
                    temporal_analysis['temporal_warnings'].append(f"Data from {source.source.value} is {time_diff/60:.1f} minutes old")
            
            # Coherencia temporal
            temporal_analysis['data_freshness'] = np.mean(freshness_scores)
            temporal_analysis['temporal_coherence'] = temporal_analysis['data_freshness']
            
            return temporal_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing temporal coherence: {e}")
            return {'temporal_coherence': 0.0, 'data_freshness': 0.0, 'temporal_warnings': []}
    
    @log_function_call
    async def _generate_intelligent_signal(self, sources: Dict[DataSource, DataSourceInfo], 
                                         coherence_result: CoherenceResult, 
                                         inconsistencies: List[str],
                                         temporal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generar señal INTELIGENTE final"""
        try:
            # Calcular puntuación general
            overall_score = await self._calculate_intelligent_score(sources, coherence_result)
            
            # Determinar tipo de señal
            signal_type = await self._determine_signal_type(overall_score, coherence_result)
            
            # Calcular confianza de la señal
            signal_confidence = await self._calculate_signal_confidence(
                overall_score, coherence_result, inconsistencies, temporal_analysis
            )
            
            # Generar análisis detallado
            detailed_analysis = await self._generate_detailed_analysis(
                sources, coherence_result, inconsistencies
            )
            
            # Generar recomendaciones
            recommendations = await self._generate_recommendations(
                coherence_result, inconsistencies, temporal_analysis
            )
            
            signal = {
                'timestamp': datetime.utcnow().isoformat(),
                'signal_type': signal_type,
                'confidence': signal_confidence,
                'overall_score': overall_score,
                'coherence_level': coherence_result.coherence_level.value,
                'coherence_score': coherence_result.overall_coherence,
                'inconsistencies': inconsistencies,
                'temporal_analysis': temporal_analysis,
                'detailed_analysis': detailed_analysis,
                'recommendations': recommendations,
                'source_breakdown': {
                    source.value: {
                        'confidence': source_info.confidence,
                        'reliability': source_info.reliability_score,
                        'data': source_info.data
                    }
                    for source, source_info in sources.items()
                }
            }
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Error generating intelligent signal: {e}")
            return self._get_default_signal()
    
    @log_function_call
    async def _calculate_intelligent_score(self, sources: Dict[DataSource, DataSourceInfo], 
                                         coherence_result: CoherenceResult) -> float:
        """Calcular puntuación INTELIGENTE general"""
        try:
            # Puntuación base de cada fuente
            source_scores = {}
            
            for source, source_info in sources.items():
                # Puntuación base
                base_score = 0.5
                
                # Ajustar por confianza
                confidence_adjustment = (source_info.confidence - 0.5) * 0.4
                
                # Ajustar por confiabilidad
                reliability_adjustment = (source_info.reliability_score - 0.5) * 0.3
                
                # Ajustar por coherencia
                coherence_adjustment = (coherence_result.overall_coherence - 0.5) * 0.3
                
                # Puntuación final
                source_scores[source] = base_score + confidence_adjustment + reliability_adjustment + coherence_adjustment
                source_scores[source] = min(1.0, max(0.0, source_scores[source]))
            
            # Puntuación ponderada
            weighted_score = 0.0
            total_weight = 0.0
            
            for source, score in source_scores.items():
                weight = self.source_weights[source]
                weighted_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                overall_score = weighted_score / total_weight
            else:
                overall_score = 0.5
            
            return overall_score
            
        except Exception as e:
            self.logger.error(f"Error calculating intelligent score: {e}")
            return 0.5
    
    @log_function_call
    async def _determine_signal_type(self, overall_score: float, coherence_result: CoherenceResult) -> str:
        """Determinar tipo de señal INTELIGENTE"""
        try:
            # Ajustar puntuación por coherencia
            coherence_adjustment = (coherence_result.overall_coherence - 0.5) * 0.2
            adjusted_score = overall_score + coherence_adjustment
            adjusted_score = min(1.0, max(0.0, adjusted_score))
            
            # Determinar tipo de señal
            if adjusted_score > 0.7 and coherence_result.coherence_level in [CoherenceLevel.VERY_HIGH, CoherenceLevel.HIGH]:
                return 'STRONG_BUY'
            elif adjusted_score > 0.6 and coherence_result.coherence_level in [CoherenceLevel.VERY_HIGH, CoherenceLevel.HIGH, CoherenceLevel.MEDIUM]:
                return 'BUY'
            elif adjusted_score > 0.4 and coherence_result.coherence_level in [CoherenceLevel.MEDIUM, CoherenceLevel.HIGH]:
                return 'WEAK_BUY'
            elif adjusted_score < 0.3 and coherence_result.coherence_level in [CoherenceLevel.VERY_HIGH, CoherenceLevel.HIGH]:
                return 'STRONG_SELL'
            elif adjusted_score < 0.4 and coherence_result.coherence_level in [CoherenceLevel.VERY_HIGH, CoherenceLevel.HIGH, CoherenceLevel.MEDIUM]:
                return 'SELL'
            elif adjusted_score < 0.6 and coherence_result.coherence_level in [CoherenceLevel.MEDIUM, CoherenceLevel.HIGH]:
                return 'WEAK_SELL'
            else:
                return 'HOLD'
                
        except Exception as e:
            self.logger.error(f"Error determining signal type: {e}")
            return 'HOLD'
    
    @log_function_call
    async def _calculate_signal_confidence(self, overall_score: float, coherence_result: CoherenceResult, 
                                         inconsistencies: List[str], temporal_analysis: Dict[str, Any]) -> float:
        """Calcular confianza de la señal INTELIGENTE"""
        try:
            # Confianza base
            base_confidence = overall_score
            
            # Ajustar por coherencia
            coherence_adjustment = (coherence_result.overall_coherence - 0.5) * 0.3
            
            # Ajustar por inconsistencias
            inconsistency_penalty = len(inconsistencies) * 0.1
            
            # Ajustar por frescura de datos
            freshness_adjustment = (temporal_analysis.get('data_freshness', 0.5) - 0.5) * 0.2
            
            # Confianza final
            final_confidence = base_confidence + coherence_adjustment - inconsistency_penalty + freshness_adjustment
            final_confidence = min(1.0, max(0.0, final_confidence))
            
            return final_confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating signal confidence: {e}")
            return 0.5
    
    @log_function_call
    async def _generate_detailed_analysis(self, sources: Dict[DataSource, DataSourceInfo], 
                                        coherence_result: CoherenceResult, 
                                        inconsistencies: List[str]) -> Dict[str, Any]:
        """Generar análisis detallado INTELIGENTE"""
        try:
            analysis = {
                'strengths': [],
                'weaknesses': [],
                'opportunities': [],
                'threats': [],
                'key_insights': [],
                'risk_factors': []
            }
            
            # Fortalezas
            if coherence_result.overall_coherence > 0.8:
                analysis['strengths'].append("High data coherence across all sources")
            if len(inconsistencies) == 0:
                analysis['strengths'].append("No data inconsistencies detected")
            
            # Debilidades
            if coherence_result.overall_coherence < 0.6:
                analysis['weaknesses'].append("Low data coherence between sources")
            if len(inconsistencies) > 2:
                analysis['weaknesses'].append("Multiple data inconsistencies detected")
            
            # Oportunidades
            if sources[DataSource.ONCHAIN].data.get('whale_activity', 0) > 0.7:
                analysis['opportunities'].append("High whale activity indicates potential price movement")
            if sources[DataSource.NEWS].data.get('high_impact_news', 0) > 3:
                analysis['opportunities'].append("Multiple high-impact news events")
            
            # Amenazas
            if sources[DataSource.ONCHAIN].data.get('exchange_flows', 0) > 0.5:
                analysis['threats'].append("High exchange flows indicate potential selling pressure")
            if sources[DataSource.SENTIMENT].data.get('social_sentiment', 0) < -0.3:
                analysis['threats'].append("Negative social sentiment")
            
            # Insights clave
            analysis['key_insights'].append(f"Overall coherence: {coherence_result.overall_coherence:.2f}")
            analysis['key_insights'].append(f"Data inconsistencies: {len(inconsistencies)}")
            
            # Factores de riesgo
            if coherence_result.coherence_level == CoherenceLevel.VERY_LOW:
                analysis['risk_factors'].append("Very low data coherence")
            if len(inconsistencies) > 3:
                analysis['risk_factors'].append("High number of data inconsistencies")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating detailed analysis: {e}")
            return {'strengths': [], 'weaknesses': [], 'opportunities': [], 'threats': [], 'key_insights': [], 'risk_factors': []}
    
    @log_function_call
    async def _generate_recommendations(self, coherence_result: CoherenceResult, 
                                      inconsistencies: List[str], 
                                      temporal_analysis: Dict[str, Any]) -> List[str]:
        """Generar recomendaciones INTELIGENTES"""
        try:
            recommendations = []
            
            # Recomendaciones de coherencia
            if coherence_result.coherence_level == CoherenceLevel.VERY_LOW:
                recommendations.append("Consider waiting for better data coherence before making trading decisions")
            elif coherence_result.coherence_level == CoherenceLevel.LOW:
                recommendations.append("Proceed with caution due to low data coherence")
            
            # Recomendaciones de inconsistencias
            if 'TECH_ML_MISMATCH' in inconsistencies:
                recommendations.append("Verify technical indicators and ML model accuracy")
            if 'NEWS_SOCIAL_SENTIMENT_MISMATCH' in inconsistencies:
                recommendations.append("Check news sources and social media data quality")
            
            # Recomendaciones temporales
            if temporal_analysis.get('data_freshness', 1.0) < 0.5:
                recommendations.append("Consider waiting for fresher data")
            
            # Recomendaciones generales
            if coherence_result.overall_coherence > 0.8 and len(inconsistencies) == 0:
                recommendations.append("High confidence signal - consider taking position")
            elif coherence_result.overall_coherence < 0.6 or len(inconsistencies) > 2:
                recommendations.append("Low confidence signal - consider waiting or reducing position size")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []
    
    @log_function_call
    async def _continuous_learning(self, sources: Dict[DataSource, DataSourceInfo], 
                                 final_signal: Dict[str, Any], 
                                 coherence_result: CoherenceResult):
        """Aprendizaje continuo INTELIGENTE"""
        try:
            # Guardar en historial de coherencia
            self.coherence_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'coherence': coherence_result.overall_coherence,
                'coherence_level': coherence_result.coherence_level.value,
                'inconsistencies': len(coherence_result.inconsistencies),
                'signal_type': final_signal.get('signal_type', 'HOLD'),
                'confidence': final_signal.get('confidence', 0.5)
            })
            
            # Limitar historial
            if len(self.coherence_history) > 1000:
                self.coherence_history = self.coherence_history[-1000:]
            
            # Guardar historial
            await self._save_coherence_history()
            
        except Exception as e:
            self.logger.error(f"Error in continuous learning: {e}")
    
    @log_function_call
    async def _update_cache(self, sources: Dict[DataSource, DataSourceInfo], final_signal: Dict[str, Any]):
        """Actualizar cache para eficiencia"""
        try:
            cache_key = f"fusion_{datetime.utcnow().strftime('%Y%m%d_%H%M')}"
            self.data_cache[cache_key] = {
                'sources': sources,
                'signal': final_signal,
                'timestamp': datetime.utcnow()
            }
            
            # Limpiar cache viejo
            now = datetime.utcnow()
            keys_to_remove = []
            for key, data in self.data_cache.items():
                if (now - data['timestamp']).total_seconds() > self.cache_ttl:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.data_cache[key]
                
        except Exception as e:
            self.logger.error(f"Error updating cache: {e}")
    
    @log_function_call
    async def _load_coherence_history(self):
        """Cargar historial de coherencia"""
        try:
            history_file = Path("data/coherence_history.json")
            if history_file.exists():
                with open(history_file, 'r') as f:
                    self.coherence_history = json.load(f)
                self.logger.info(f"✅ Loaded {len(self.coherence_history)} coherence records")
            else:
                self.coherence_history = []
                
        except Exception as e:
            self.logger.error(f"Error loading coherence history: {e}")
            self.coherence_history = []
    
    @log_function_call
    async def _save_coherence_history(self):
        """Guardar historial de coherencia"""
        try:
            history_file = Path("data/coherence_history.json")
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(history_file, 'w') as f:
                json.dump(self.coherence_history, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving coherence history: {e}")
    
    def _get_coherence_level(self, coherence: float) -> CoherenceLevel:
        """Obtener nivel de coherencia"""
        if coherence >= self.coherence_thresholds['very_high']:
            return CoherenceLevel.VERY_HIGH
        elif coherence >= self.coherence_thresholds['high']:
            return CoherenceLevel.HIGH
        elif coherence >= self.coherence_thresholds['medium']:
            return CoherenceLevel.MEDIUM
        elif coherence >= self.coherence_thresholds['low']:
            return CoherenceLevel.LOW
        else:
            return CoherenceLevel.VERY_LOW
    
    def _get_default_signal(self) -> Dict[str, Any]:
        """Señal por defecto"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'signal_type': 'HOLD',
            'confidence': 0.0,
            'overall_score': 0.5,
            'coherence_level': 'very_low',
            'coherence_score': 0.0,
            'inconsistencies': [],
            'temporal_analysis': {'temporal_coherence': 0.0, 'data_freshness': 0.0, 'temporal_warnings': []},
            'detailed_analysis': {'strengths': [], 'weaknesses': [], 'opportunities': [], 'threats': [], 'key_insights': [], 'risk_factors': []},
            'recommendations': ['Data fusion failed - using default signal'],
            'source_breakdown': {}
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del Data Fusion Hub"""
        try:
            return {
                'healthy': True,
                'message': 'INTELLIGENT Data Fusion Hub healthy',
                'coherence_history_size': len(self.coherence_history),
                'cache_size': len(self.data_cache),
                'coherence_thresholds': self.coherence_thresholds,
                'source_weights': {k.value: v for k, v in self.source_weights.items()},
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f'INTELLIGENT Data Fusion Hub unhealthy: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }
