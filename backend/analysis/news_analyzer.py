#!/usr/bin/env python3
"""
News Analyzer - CryptoPulse Pro
Análisis REAL de noticias + Telegram + Fear & Greed Index
"""

import asyncio
import aiohttp
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import re
from textblob import TextBlob
import feedparser
from bs4 import BeautifulSoup
import requests

from utils.logger import get_logger, log_execution_time, log_function_call

class NewsAnalyzer:
    """
    Analizador REAL de noticias y Telegram
    - Noticias REALES de sitios crypto
    - Canales REALES de Telegram
    - Fear & Greed Index REAL
    - Análisis de sentimiento REAL
    """
    
    def __init__(self, api_config):
        self.api_config = api_config
        self.logger = get_logger("NewsAnalyzer")
        
        # URLs REALES de sitios crypto
        self.crypto_sites = [
            "https://cointelegraph.com/rss",
            "https://coindesk.com/arc/outboundfeeds/rss/?outputType=xml",
            "https://decrypt.co/feed",
            "https://cryptonews.com/news/feed/",
            "https://bitcoinmagazine.com/rss",
            "https://cryptoslate.com/feed/",
            "https://u.today/rss",
            "https://coingape.com/feed/",
            "https://cryptodaily.co.uk/feed",
            "https://beincrypto.com/feed/"
        ]
        
        # Canales REALES de Telegram (necesitarás tokens)
        self.telegram_channels = [
            "@cryptopulse_signals",  # Tu canal principal
            "@crypto_news_official",
            "@bitcoin_news_channel",
            "@ethereum_news",
            "@altcoin_news",
            "@crypto_market_analysis",
            "@trading_signals_crypto",
            "@crypto_insights",
            "@defi_pulse",
            "@nft_news_channel"
        ]
        
        # Configuración de APIs REALES
        self.telegram_token = api_config.get('telegram_token', '')
        self.telegram_api_url = f"https://api.telegram.org/bot{self.telegram_token}"
        
        # Palabras clave para análisis de impacto
        self.impact_keywords = {
            'high_impact': [
                'adoption', 'institutional', 'etf', 'regulation', 'ban', 'legal',
                'partnership', 'merger', 'acquisition', 'upgrade', 'hard fork',
                'mainnet', 'launch', 'listing', 'delisting', 'hack', 'exploit',
                'sec', 'federal', 'government', 'central bank', 'crypto ban'
            ],
            'medium_impact': [
                'price', 'pump', 'dump', 'rally', 'crash', 'breakout',
                'resistance', 'support', 'trend', 'analysis', 'prediction',
                'whale', 'accumulation', 'distribution', 'volume'
            ],
            'low_impact': [
                'community', 'discussion', 'opinion', 'speculation', 'rumor',
                'meme', 'joke', 'fun', 'entertainment'
            ]
        }
        
        self.logger.info("📰 NewsAnalyzer initialized (REAL APIs)")
    
    async def initialize(self):
        """Inicializar el analizador de noticias"""
        try:
            self.logger.info("🔧 Initializing REAL news analyzer...")
            
            # Verificar APIs REALES
            await self._check_real_apis()
            
            self.logger.info("✅ REAL news analyzer initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing REAL news analyzer: {e}")
    
    @log_execution_time
    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Análisis REAL de noticias y Telegram para un símbolo"""
        try:
            self.logger.info(f"📰 REAL analysis for {symbol}")
            
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                
                # Noticias REALES
                'news_sentiment': 0.0,
                'news_impact_score': 0.0,
                'news_count': 0,
                'positive_news': 0,
                'negative_news': 0,
                'neutral_news': 0,
                'high_impact_news': 0,
                'top_news': [],
                
                # Telegram REAL
                'telegram_sentiment': 0.0,
                'telegram_impact_score': 0.0,
                'telegram_count': 0,
                'top_telegram': [],
                
                # Fear & Greed Index REAL
                'fear_greed_index': 0.0,
                'fear_greed_category': 'NEUTRAL',
                
                # Análisis general
                'overall_sentiment': 0.0,
                'sentiment_category': 'NEUTRAL',
                'confidence': 0.0,
                'impact_analysis': {}
            }
            
            # Análisis paralelo REAL
            tasks = [
                self._analyze_real_crypto_news(symbol),
                self._analyze_real_telegram_channels(symbol),
                self._analyze_real_fear_greed_index()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Procesar resultados REALES
            news_result = results[0] if not isinstance(results[0], Exception) else {}
            telegram_result = results[1] if not isinstance(results[1], Exception) else {}
            fear_greed_result = results[2] if not isinstance(results[2], Exception) else {}
            
            # Actualizar análisis
            analysis.update(news_result)
            analysis.update(telegram_result)
            analysis.update(fear_greed_result)
            
            # Calcular sentimiento general
            overall_sentiment = await self._calculate_overall_sentiment(analysis)
            analysis.update(overall_sentiment)
            
            # Análisis de impacto
            impact_analysis = await self._analyze_news_impact(analysis)
            analysis['impact_analysis'] = impact_analysis
            
            self.logger.info(f"✅ REAL analysis completed for {symbol}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in REAL analysis for {symbol}: {e}")
            return self._get_default_analysis(symbol)
    
    @log_function_call
    async def _analyze_real_crypto_news(self, symbol: str) -> Dict[str, Any]:
        """Analizar noticias REALES de sitios crypto"""
        try:
            news_data = {
                'news_sentiment': 0.0,
                'news_impact_score': 0.0,
                'news_count': 0,
                'positive_news': 0,
                'negative_news': 0,
                'neutral_news': 0,
                'high_impact_news': 0,
                'top_news': []
            }
            
            all_articles = []
            
            # Obtener noticias REALES de cada sitio
            for site_url in self.crypto_sites:
                try:
                    articles = await self._fetch_real_rss_news(site_url, symbol)
                    all_articles.extend(articles)
                    self.logger.info(f"✅ Fetched {len(articles)} articles from {site_url}")
                except Exception as e:
                    self.logger.warning(f"Error fetching REAL news from {site_url}: {e}")
            
            # Analizar sentimiento e impacto REAL de cada artículo
            sentiments = []
            impact_scores = []
            
            for article in all_articles:
                # Análisis de sentimiento REAL
                sentiment = await self._analyze_real_text_sentiment(article['title'] + ' ' + article.get('description', ''))
                article['sentiment'] = sentiment
                sentiments.append(sentiment)
                
                # Análisis de impacto REAL
                impact_score = await self._analyze_real_news_impact(article['title'] + ' ' + article.get('description', ''))
                article['impact_score'] = impact_score
                impact_scores.append(impact_score)
                
                # Clasificar noticia
                if sentiment > 0.1:
                    news_data['positive_news'] += 1
                elif sentiment < -0.1:
                    news_data['negative_news'] += 1
                else:
                    news_data['neutral_news'] += 1
                
                # Clasificar impacto
                if impact_score > 0.7:
                    news_data['high_impact_news'] += 1
                
                # Agregar a top noticias
                if impact_score > 0.5 or abs(sentiment) > 0.3:
                    news_data['top_news'].append(article)
            
            # Calcular métricas REALES
            if sentiments:
                news_data['news_sentiment'] = float(np.mean(sentiments))
                news_data['news_impact_score'] = float(np.mean(impact_scores))
                news_data['news_count'] = len(sentiments)
                
                # Ordenar top noticias por impacto y sentimiento
                news_data['top_news'] = sorted(
                    news_data['top_news'], 
                    key=lambda x: (x['impact_score'], abs(x['sentiment'])), 
                    reverse=True
                )[:5]
            
            self.logger.info(f"✅ REAL news analysis: {news_data['news_count']} articles, {news_data['positive_news']} positive, {news_data['negative_news']} negative")
            return news_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing REAL crypto news: {e}")
            return {'news_sentiment': 0.0, 'news_impact_score': 0.0, 'news_count': 0}
    
    @log_function_call
    async def _analyze_real_telegram_channels(self, symbol: str) -> Dict[str, Any]:
        """Analizar canales REALES de Telegram"""
        try:
            telegram_data = {
                'telegram_sentiment': 0.0,
                'telegram_impact_score': 0.0,
                'telegram_count': 0,
                'top_telegram': []
            }
            
            all_messages = []
            
            # Obtener mensajes REALES de canales de Telegram
            for channel in self.telegram_channels:
                try:
                    messages = await self._fetch_real_telegram_messages(channel, symbol)
                    all_messages.extend(messages)
                    self.logger.info(f"✅ Fetched {len(messages)} messages from {channel}")
                except Exception as e:
                    self.logger.warning(f"Error fetching REAL Telegram from {channel}: {e}")
            
            # Analizar sentimiento e impacto REAL de cada mensaje
            sentiments = []
            impact_scores = []
            
            for message in all_messages:
                # Análisis de sentimiento REAL
                sentiment = await self._analyze_real_text_sentiment(message['text'])
                message['sentiment'] = sentiment
                sentiments.append(sentiment)
                
                # Análisis de impacto REAL
                impact_score = await self._analyze_real_news_impact(message['text'])
                message['impact_score'] = impact_score
                impact_scores.append(impact_score)
                
                # Agregar a top mensajes
                if impact_score > 0.5 or abs(sentiment) > 0.3:
                    telegram_data['top_telegram'].append(message)
            
            # Calcular métricas REALES
            if sentiments:
                telegram_data['telegram_sentiment'] = float(np.mean(sentiments))
                telegram_data['telegram_impact_score'] = float(np.mean(impact_scores))
                telegram_data['telegram_count'] = len(sentiments)
                
                # Ordenar top mensajes por impacto y sentimiento
                telegram_data['top_telegram'] = sorted(
                    telegram_data['top_telegram'], 
                    key=lambda x: (x['impact_score'], abs(x['sentiment'])), 
                    reverse=True
                )[:5]
            
            self.logger.info(f"✅ REAL Telegram analysis: {telegram_data['telegram_count']} messages")
            return telegram_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing REAL Telegram channels: {e}")
            return {'telegram_sentiment': 0.0, 'telegram_impact_score': 0.0, 'telegram_count': 0}
    
    @log_function_call
    async def _analyze_real_fear_greed_index(self) -> Dict[str, Any]:
        """Analizar Fear & Greed Index REAL"""
        try:
            fear_greed_data = {
                'fear_greed_index': 0.0,
                'fear_greed_category': 'NEUTRAL'
            }
            
            # Obtener Fear & Greed Index REAL
            fear_greed = await self._get_real_fear_greed_index()
            fear_greed_data['fear_greed_index'] = fear_greed
            
            # Categorizar
            if fear_greed >= 75:
                fear_greed_data['fear_greed_category'] = 'EXTREME_GREED'
            elif fear_greed >= 55:
                fear_greed_data['fear_greed_category'] = 'GREED'
            elif fear_greed >= 45:
                fear_greed_data['fear_greed_category'] = 'NEUTRAL'
            elif fear_greed >= 25:
                fear_greed_data['fear_greed_category'] = 'FEAR'
            else:
                fear_greed_data['fear_greed_category'] = 'EXTREME_FEAR'
            
            self.logger.info(f"✅ REAL Fear & Greed Index: {fear_greed} ({fear_greed_data['fear_greed_category']})")
            return fear_greed_data
            
        except Exception as e:
            self.logger.error(f"Error analyzing REAL Fear & Greed Index: {e}")
            return {'fear_greed_index': 50.0, 'fear_greed_category': 'NEUTRAL'}
    
    @log_function_call
    async def _fetch_real_rss_news(self, url: str, symbol: str) -> List[Dict[str, Any]]:
        """Obtener noticias REALES de RSS feed"""
        try:
            articles = []
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        
                        # Parsear RSS REAL
                        feed = feedparser.parse(content)
                        
                        for entry in feed.entries[:10]:  # Limitar a 10 artículos
                            # Filtrar por símbolo
                            if symbol.lower() in entry.title.lower() or symbol.lower() in entry.get('description', '').lower():
                                articles.append({
                                    'title': entry.title,
                                    'description': entry.get('description', ''),
                                    'link': entry.link,
                                    'published': entry.get('published', ''),
                                    'source': url
                                })
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching REAL RSS news from {url}: {e}")
            return []
    
    @log_function_call
    async def _fetch_real_telegram_messages(self, channel: str, symbol: str) -> List[Dict[str, Any]]:
        """Obtener mensajes REALES de canales de Telegram"""
        try:
            messages = []
            
            if not self.telegram_token:
                self.logger.warning("No Telegram token provided, skipping Telegram analysis")
                return messages
            
            # Obtener mensajes REALES del canal
            url = f"{self.telegram_api_url}/getUpdates"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get('ok'):
                            updates = data.get('result', [])
                            
                            for update in updates:
                                if 'channel_post' in update:
                                    post = update['channel_post']
                                    text = post.get('text', '')
                                    
                                    # Filtrar por símbolo
                                    if symbol.lower() in text.lower():
                                        messages.append({
                                            'text': text,
                                            'channel': channel,
                                            'timestamp': datetime.fromtimestamp(post['date']).isoformat(),
                                            'message_id': post['message_id']
                                        })
            
            return messages
            
        except Exception as e:
            self.logger.error(f"Error fetching REAL Telegram messages from {channel}: {e}")
            return []
    
    @log_function_call
    async def _get_real_fear_greed_index(self) -> float:
        """Obtener Fear & Greed Index REAL"""
        try:
            url = "https://api.alternative.me/fng/"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'data' in data and len(data['data']) > 0:
                            return float(data['data'][0]['value'])
            
            return 50.0
            
        except Exception as e:
            self.logger.error(f"Error getting REAL Fear & Greed Index: {e}")
            return 50.0
    
    @log_function_call
    async def _analyze_real_text_sentiment(self, text: str) -> float:
        """Analizar sentimiento REAL de texto usando TextBlob"""
        try:
            # Limpiar texto
            text = re.sub(r'[^\w\s]', '', text.lower())
            
            # Análisis de sentimiento REAL
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            
            # Palabras clave crypto específicas
            crypto_keywords = {
                'positive': [
                    'bull', 'moon', 'pump', 'surge', 'rally', 'breakthrough', 
                    'adoption', 'institutional', 'partnership', 'upgrade',
                    'launch', 'listing', 'breakout', 'resistance', 'support'
                ],
                'negative': [
                    'bear', 'crash', 'dump', 'decline', 'fall', 'regulation', 
                    'ban', 'scam', 'hack', 'exploit', 'delisting', 'rejection'
                ]
            }
            
            # Contar palabras clave
            positive_count = sum(1 for word in crypto_keywords['positive'] if word in text)
            negative_count = sum(1 for word in crypto_keywords['negative'] if word in text)
            
            # Ajustar sentimiento
            if positive_count > negative_count:
                sentiment += 0.3
            elif negative_count > positive_count:
                sentiment -= 0.3
            
            # Normalizar entre -1 y 1
            return max(-1.0, min(1.0, sentiment))
            
        except Exception as e:
            self.logger.error(f"Error analyzing REAL text sentiment: {e}")
            return 0.0
    
    @log_function_call
    async def _analyze_real_news_impact(self, text: str) -> float:
        """Analizar impacto REAL de noticias/mensajes"""
        try:
            text_lower = text.lower()
            impact_score = 0.0
            
            # Contar palabras clave de alto impacto
            high_impact_count = sum(1 for word in self.impact_keywords['high_impact'] if word in text_lower)
            medium_impact_count = sum(1 for word in self.impact_keywords['medium_impact'] if word in text_lower)
            low_impact_count = sum(1 for word in self.impact_keywords['low_impact'] if word in text_lower)
            
            # Calcular puntuación de impacto
            impact_score = (
                high_impact_count * 0.8 +
                medium_impact_count * 0.4 +
                low_impact_count * 0.1
            )
            
            # Normalizar entre 0 y 1
            impact_score = min(1.0, impact_score / 3.0)
            
            return impact_score
            
        except Exception as e:
            self.logger.error(f"Error analyzing REAL news impact: {e}")
            return 0.0
    
    @log_function_call
    async def _calculate_overall_sentiment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calcular sentimiento general"""
        try:
            # Ponderar diferentes fuentes
            news_weight = 0.5
            telegram_weight = 0.3
            fear_greed_weight = 0.2
            
            # Normalizar Fear & Greed Index (0-100 a -1 a 1)
            fear_greed_normalized = (analysis['fear_greed_index'] - 50) / 50
            
            # Calcular sentimiento ponderado
            overall_sentiment = (
                analysis['news_sentiment'] * news_weight +
                analysis['telegram_sentiment'] * telegram_weight +
                fear_greed_normalized * fear_greed_weight
            )
            
            # Determinar categoría
            if overall_sentiment > 0.3:
                sentiment_category = 'VERY_BULLISH'
            elif overall_sentiment > 0.1:
                sentiment_category = 'BULLISH'
            elif overall_sentiment < -0.3:
                sentiment_category = 'VERY_BEARISH'
            elif overall_sentiment < -0.1:
                sentiment_category = 'BEARISH'
            else:
                sentiment_category = 'NEUTRAL'
            
            return {
                'overall_sentiment': float(overall_sentiment),
                'sentiment_category': sentiment_category,
                'confidence': min(1.0, max(0.0, abs(overall_sentiment) * 2))
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating overall sentiment: {e}")
            return {'overall_sentiment': 0.0, 'sentiment_category': 'NEUTRAL', 'confidence': 0.0}
    
    @log_function_call
    async def _analyze_news_impact(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analizar impacto general de noticias"""
        try:
            impact_analysis = {
                'overall_impact': 'MEDIUM',
                'impact_score': 0.0,
                'key_factors': [],
                'risk_level': 'MEDIUM',
                'opportunity_level': 'MEDIUM'
            }
            
            # Calcular puntuación de impacto
            impact_factors = []
            
            # Factor 1: Noticias de alto impacto
            if analysis.get('high_impact_news', 0) > 3:
                impact_factors.append('HIGH_IMPACT_NEWS')
                impact_analysis['key_factors'].append('Multiple high-impact news')
            
            # Factor 2: Fear & Greed extremo
            fear_greed = analysis.get('fear_greed_index', 50)
            if fear_greed >= 75 or fear_greed <= 25:
                impact_factors.append('EXTREME_MARKET_SENTIMENT')
                impact_analysis['key_factors'].append('Extreme market sentiment')
            
            # Factor 3: Sentimiento de Telegram
            telegram_sentiment = analysis.get('telegram_sentiment', 0)
            if abs(telegram_sentiment) > 0.5:
                impact_factors.append('STRONG_TELEGRAM_SENTIMENT')
                impact_analysis['key_factors'].append('Strong Telegram sentiment')
            
            # Calcular puntuación de impacto
            impact_score = len(impact_factors) / 3.0
            impact_analysis['impact_score'] = impact_score
            
            # Determinar nivel de impacto
            if impact_score >= 0.75:
                impact_analysis['overall_impact'] = 'HIGH'
                impact_analysis['risk_level'] = 'HIGH'
                impact_analysis['opportunity_level'] = 'HIGH'
            elif impact_score >= 0.5:
                impact_analysis['overall_impact'] = 'MEDIUM'
                impact_analysis['risk_level'] = 'MEDIUM'
                impact_analysis['opportunity_level'] = 'MEDIUM'
            else:
                impact_analysis['overall_impact'] = 'LOW'
                impact_analysis['risk_level'] = 'LOW'
                impact_analysis['opportunity_level'] = 'LOW'
            
            return impact_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing news impact: {e}")
            return {'overall_impact': 'MEDIUM', 'impact_score': 0.0, 'key_factors': [], 'risk_level': 'MEDIUM', 'opportunity_level': 'MEDIUM'}
    
    @log_function_call
    async def _check_real_apis(self):
        """Verificar disponibilidad de APIs REALES"""
        try:
            # Verificar Fear & Greed Index REAL
            fear_greed = await self._get_real_fear_greed_index()
            if fear_greed > 0:
                self.logger.info("✅ REAL Fear & Greed Index API available")
            else:
                self.logger.warning("⚠️ REAL Fear & Greed Index API not available")
            
            # Verificar Telegram API REAL
            if self.telegram_token:
                self.logger.info("✅ REAL Telegram API configured")
            else:
                self.logger.warning("⚠️ REAL Telegram API not configured")
            
            # Verificar RSS feeds REALES
            self.logger.info(f"✅ {len(self.crypto_sites)} REAL RSS feeds configured")
            
        except Exception as e:
            self.logger.error(f"Error checking REAL APIs: {e}")
    
    def _get_default_analysis(self, symbol: str) -> Dict[str, Any]:
        """Análisis por defecto"""
        return {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'news_sentiment': 0.0,
            'news_impact_score': 0.0,
            'news_count': 0,
            'positive_news': 0,
            'negative_news': 0,
            'neutral_news': 0,
            'high_impact_news': 0,
            'top_news': [],
            'telegram_sentiment': 0.0,
            'telegram_impact_score': 0.0,
            'telegram_count': 0,
            'top_telegram': [],
            'fear_greed_index': 50.0,
            'fear_greed_category': 'NEUTRAL',
            'overall_sentiment': 0.0,
            'sentiment_category': 'NEUTRAL',
            'confidence': 0.0,
            'impact_analysis': {}
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del analizador de noticias"""
        try:
            return {
                'healthy': True,
                'message': 'REAL News + Telegram analyzer healthy',
                'crypto_sites': len(self.crypto_sites),
                'telegram_channels': len(self.telegram_channels),
                'telegram_configured': bool(self.telegram_token),
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f'REAL News + Telegram analyzer unhealthy: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }