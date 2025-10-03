#!/usr/bin/env python3
"""
Enhanced News Analyzer - CryptoPulse Pro
Análisis mejorado de noticias con múltiples APIs
"""

import asyncio
import aiohttp
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import feedparser
from textblob import TextBlob

from utils.logger import get_logger, log_execution_time, log_function_call

class EnhancedNewsAnalyzer:
    """
    Analizador mejorado de noticias
    - NewsAPI
    - CryptoPanic
    - RSS Feeds
    - Fear & Greed Index
    """
    
    def __init__(self, api_config: Dict[str, str]):
        self.api_config = api_config
        self.logger = get_logger("EnhancedNewsAnalyzer")
        
        # URLs de APIs
        self.news_api_url = "https://newsapi.org/v2/everything"
        self.cryptopanic_url = "https://cryptopanic.com/api/v1/posts"
        self.fear_greed_url = "https://api.alternative.me/fng"
        
        # RSS feeds
        self.rss_feeds = [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss",
            "https://decrypt.co/feed",
            "https://bitcoinmagazine.com/feed",
            "https://cryptonews.com/news/feed"
        ]
        
        self.logger.info("📰 EnhancedNewsAnalyzer initialized")
    
    async def initialize(self):
        """Inicializar el analizador"""
        try:
            self.logger.info("🔧 Initializing Enhanced News Analyzer...")
            
            # Verificar APIs disponibles
            await self._check_api_availability()
            
            self.logger.info("✅ Enhanced News Analyzer initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing Enhanced News Analyzer: {e}")
    
    @log_execution_time
    async def analyze_news(self, symbol: str = "BTC") -> Dict[str, Any]:
        """Analizar noticias para un símbolo"""
        try:
            self.logger.info(f"📰 Analyzing news for {symbol}")
            
            # Obtener noticias de múltiples fuentes
            news_data = await self._get_all_news(symbol)
            
            # Analizar sentimiento
            sentiment_analysis = await self._analyze_sentiment(news_data)
            
            # Obtener Fear & Greed Index
            fear_greed = await self._get_fear_greed_index()
            
            # Calcular impacto de noticias
            impact_analysis = await self._calculate_news_impact(news_data)
            
            # Crear resultado final
            result = {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),
                'overall_sentiment': sentiment_analysis['overall_sentiment'],
                'confidence': sentiment_analysis['confidence'],
                'news_count': len(news_data),
                'fear_greed_index': fear_greed,
                'news_impact_score': impact_analysis['impact_score'],
                'high_impact_news': impact_analysis['high_impact_count'],
                'sentiment_breakdown': sentiment_analysis['breakdown'],
                'top_news': news_data[:5],  # Top 5 noticias
                'sources': list(set([news.get('source', 'Unknown') for news in news_data]))
            }
            
            self.logger.info(f"✅ News analysis completed for {symbol}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing news for {symbol}: {e}")
            return self._get_default_news_analysis()
    
    @log_function_call
    async def _get_all_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Obtener noticias de todas las fuentes"""
        all_news = []
        
        # NewsAPI
        if self.api_config.get('news_api_key'):
            newsapi_news = await self._get_newsapi_news(symbol)
            all_news.extend(newsapi_news)
        
        # CryptoPanic
        if self.api_config.get('cryptopanic_api_key'):
            cryptopanic_news = await self._get_cryptopanic_news(symbol)
            all_news.extend(cryptopanic_news)
        
        # RSS Feeds
        rss_news = await self._get_rss_news(symbol)
        all_news.extend(rss_news)
        
        # Eliminar duplicados
        unique_news = []
        seen_titles = set()
        
        for news in all_news:
            title = news.get('title', '').lower()
            if title not in seen_titles:
                seen_titles.add(title)
                unique_news.append(news)
        
        return unique_news
    
    @log_function_call
    async def _get_newsapi_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Obtener noticias de NewsAPI"""
        try:
            url = self.news_api_url
            params = {
                'q': f"{symbol} cryptocurrency",
                'apiKey': self.api_config.get('news_api_key'),
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': 20
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get('articles', [])
                        
                        news_list = []
                        for article in articles:
                            news_list.append({
                                'title': article.get('title', ''),
                                'description': article.get('description', ''),
                                'url': article.get('url', ''),
                                'publishedAt': article.get('publishedAt', ''),
                                'source': article.get('source', {}).get('name', 'NewsAPI'),
                                'content': article.get('content', '')
                            })
                        
                        return news_list
                    else:
                        self.logger.warning(f"NewsAPI error: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error getting NewsAPI news: {e}")
            return []
    
    @log_function_call
    async def _get_cryptopanic_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Obtener noticias de CryptoPanic"""
        try:
            url = self.cryptopanic_url
            params = {
                'auth_token': self.api_config.get('cryptopanic_api_key'),
                'currencies': symbol,
                'kind': 'news',
                'public': 'true'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        results = data.get('results', [])
                        
                        news_list = []
                        for result in results:
                            news_list.append({
                                'title': result.get('title', ''),
                                'description': result.get('metadata', {}).get('description', ''),
                                'url': result.get('url', ''),
                                'publishedAt': result.get('created_at', ''),
                                'source': 'CryptoPanic',
                                'content': result.get('title', ''),
                                'votes': result.get('votes', {}),
                                'kind': result.get('kind', 'news')
                            })
                        
                        return news_list
                    else:
                        self.logger.warning(f"CryptoPanic error: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error getting CryptoPanic news: {e}")
            return []
    
    @log_function_call
    async def _get_rss_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Obtener noticias de RSS feeds"""
        try:
            news_list = []
            
            for feed_url in self.rss_feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:10]:  # Top 10 de cada feed
                        # Filtrar por símbolo
                        title = entry.get('title', '').lower()
                        summary = entry.get('summary', '').lower()
                        
                        if symbol.lower() in title or symbol.lower() in summary:
                            news_list.append({
                                'title': entry.get('title', ''),
                                'description': entry.get('summary', ''),
                                'url': entry.get('link', ''),
                                'publishedAt': entry.get('published', ''),
                                'source': feed.feed.get('title', 'RSS'),
                                'content': entry.get('summary', '')
                            })
                            
                except Exception as e:
                    self.logger.warning(f"Error parsing RSS feed {feed_url}: {e}")
                    continue
            
            return news_list
            
        except Exception as e:
            self.logger.error(f"Error getting RSS news: {e}")
            return []
    
    @log_function_call
    async def _analyze_sentiment(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analizar sentimiento de las noticias"""
        try:
            if not news_data:
                return {
                    'overall_sentiment': 0.0,
                    'confidence': 0.0,
                    'breakdown': {}
                }
            
            sentiments = []
            source_sentiments = {}
            
            for news in news_data:
                # Combinar título y descripción para análisis
                text = f"{news.get('title', '')} {news.get('description', '')}"
                
                if text.strip():
                    # Análisis de sentimiento con TextBlob
                    blob = TextBlob(text)
                    sentiment = blob.sentiment.polarity  # -1 a 1
                    sentiments.append(sentiment)
                    
                    # Agrupar por fuente
                    source = news.get('source', 'Unknown')
                    if source not in source_sentiments:
                        source_sentiments[source] = []
                    source_sentiments[source].append(sentiment)
            
            if sentiments:
                overall_sentiment = np.mean(sentiments)
                confidence = min(1.0, len(sentiments) / 10.0)  # Más noticias = más confianza
                
                # Calcular sentimiento por fuente
                breakdown = {}
                for source, source_sents in source_sentiments.items():
                    if source_sents:
                        breakdown[source] = {
                            'sentiment': np.mean(source_sents),
                            'count': len(source_sents)
                        }
                
                return {
                    'overall_sentiment': overall_sentiment,
                    'confidence': confidence,
                    'breakdown': breakdown
                }
            else:
                return {
                    'overall_sentiment': 0.0,
                    'confidence': 0.0,
                    'breakdown': {}
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return {
                'overall_sentiment': 0.0,
                'confidence': 0.0,
                'breakdown': {}
            }
    
    @log_function_call
    async def _get_fear_greed_index(self) -> float:
        """Obtener Fear & Greed Index"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.fear_greed_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get('data'):
                            return float(data['data'][0]['value'])
                    return 50.0  # Neutral por defecto
                    
        except Exception as e:
            self.logger.error(f"Error getting Fear & Greed Index: {e}")
            return 50.0
    
    @log_function_call
    async def _calculate_news_impact(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcular impacto de las noticias"""
        try:
            if not news_data:
                return {
                    'impact_score': 0.0,
                    'high_impact_count': 0
                }
            
            impact_scores = []
            high_impact_count = 0
            
            for news in news_data:
                # Palabras clave de alto impacto
                high_impact_keywords = [
                    'crash', 'surge', 'breakthrough', 'hack', 'regulation',
                    'ban', 'adoption', 'partnership', 'upgrade', 'fork'
                ]
                
                title = news.get('title', '').lower()
                description = news.get('description', '').lower()
                text = f"{title} {description}"
                
                # Calcular score de impacto
                impact_score = 0.0
                
                # Contar palabras clave de alto impacto
                for keyword in high_impact_keywords:
                    if keyword in text:
                        impact_score += 0.2
                
                # Bonus por fuente confiable
                source = news.get('source', '').lower()
                if any(trusted in source for trusted in ['cointelegraph', 'coindesk', 'decrypt']):
                    impact_score += 0.1
                
                # Bonus por votos (CryptoPanic)
                if 'votes' in news:
                    votes = news['votes']
                    if isinstance(votes, dict):
                        positive = votes.get('positive', 0)
                        negative = votes.get('negative', 0)
                        if positive + negative > 0:
                            vote_ratio = positive / (positive + negative)
                            impact_score += vote_ratio * 0.3
                
                impact_scores.append(impact_score)
                
                if impact_score > 0.5:
                    high_impact_count += 1
            
            overall_impact = np.mean(impact_scores) if impact_scores else 0.0
            
            return {
                'impact_score': overall_impact,
                'high_impact_count': high_impact_count
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating news impact: {e}")
            return {
                'impact_score': 0.0,
                'high_impact_count': 0
            }
    
    @log_function_call
    async def _check_api_availability(self):
        """Verificar disponibilidad de APIs"""
        try:
            available_apis = []
            
            if self.api_config.get('news_api_key'):
                available_apis.append('NewsAPI')
            
            if self.api_config.get('cryptopanic_api_key'):
                available_apis.append('CryptoPanic')
            
            available_apis.append('RSS Feeds')
            available_apis.append('Fear & Greed Index')
            
            self.logger.info(f"📰 Available news sources: {', '.join(available_apis)}")
            
        except Exception as e:
            self.logger.error(f"Error checking API availability: {e}")
    
    def _get_default_news_analysis(self) -> Dict[str, Any]:
        """Análisis de noticias por defecto"""
        return {
            'symbol': 'BTC',
            'timestamp': datetime.utcnow().isoformat(),
            'overall_sentiment': 0.0,
            'confidence': 0.0,
            'news_count': 0,
            'fear_greed_index': 50.0,
            'news_impact_score': 0.0,
            'high_impact_news': 0,
            'sentiment_breakdown': {},
            'top_news': [],
            'sources': []
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del analizador"""
        try:
            return {
                'healthy': True,
                'message': 'Enhanced News Analyzer healthy',
                'available_apis': list(self.api_config.keys()),
                'rss_feeds_count': len(self.rss_feeds),
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f'Enhanced News Analyzer unhealthy: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }
