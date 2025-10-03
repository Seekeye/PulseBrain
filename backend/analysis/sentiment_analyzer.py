""" Sentiment Analyzer """
""" Analysis REAL de sentimiento social """

import asyncio 
import aiohttp 
import json 
import numpy as np 
from datetime import datetime, timedelta  
from typing import Dict, List, Optional, Any  
import re 
from textblob import TextBlob  
import praw 
from googleapiclient.discovery import build 

from utils.logger import get_logger, log_execution_time, log_function_call

class SentimentAnalyzer:
    """ Analizador REAL de sentimiento social 
        - Twitter sentiment 
        - Reddit sentiment 
        - Youtube sentiment 
        - Social media trends REALES 
    """

    def __init__(self, api_config):
        self.api_config = api_config 
        self.logger = get_logger("SentimentAnalyzer")

        # Configuración de API's GRATIS 
        self.reddit_client_id = api_config.get('reddit_client_id', '')
        self.reddit_client_secret = api_config.get('reddit_client_secret', '')
        self.reddit_user_agent = api_config.get('reddit_user_agent', 'CryptoPulseBot/1.0')
        self.youtube_api_key = api_config.get('youtube_api_key', '')
        self.twitter_bearer_token = api_config.get('twitter_bearer_token', '')

        # Inicializar clientes 
        self.reddit = None 
        self.youtube = None 

        # Subreddits crypto 
        self.crypto_subreddits = [
            'cryptocurrency',
            'bitcoin',
            'ethereum',
            'cardano',
            'cryptomarkets',
            'altcoin',
            'defi',
            'nft'
        ]

        self.logger.info("SentimentAnalyzer initialized (REAL APIs FREE)")
    
    async def initialize(self):
        """ Inicializar el analizador de sentimiento """
        try:
            self.logger.info("Initializing REAL sentiment analyzer...")

            # Inicializar Reddit API
            if self.reddit_client_id and self.reddit_client_secret:
                self.reddit = praw.reddit(
                    client_id=self.reddit_client_id,
                    client_secret=self.reddit_client_secret,
                    user_agent=self.reddit_user_agent

                    )
                self.logger.info("REAL Reddit API initialized")
            else:
                self.logger.warning("REAL Reddit API not configured")

            # Inicializar Yutube API 
            if self.youtube_api_key:
                self.youtube = build('youtube', 'v3', developerKey=self.youtube_api_key)
                self.logger.info("REAL Youtube API initialized")
            else:
                self.logger.warning("REAL Youtube API not configured")
            
            self.logger.info("✅ REAL sentiment analyzer initialized")
        
        except Exception as e:
            self.logger.error(f"Error initializing REAL sentiment analyzer: {e}")

    @log_execution_time
    async def analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """ Análisis REAL de sentimiento social para un símbolo """
        try:
            self.logger.info(f"REAL social sentiment analysis for {symbol}")

            analysis = {
                'symbol': symbol,
                'timestamp': datetime.utcnow().isoformat(),

                # Sentimiento social general 
                'social_sentiment': 0.0,
                'social_confidence': 0.0,
                'social_trend': 'NEUTRAL',
                # Twitter 
                'twitter_sentiment': 0.0,
                'twitter_mentions': 0,
                'twitter_engagment': 0.0,

                 # Reddit (GRATIS)
                'reddit_sentiment': 0.0,
                'reddit_mentions': 0,
                'reddit_upvotes': 0,
                
                # YouTube (GRATIS)
                'youtube_sentiment': 0.0,
                'youtube_mentions': 0,
                'youtube_views': 0,
                
                # Análisis de tendencias
                'trending_topics': [],
                'sentiment_breakdown': {}
            }

            # Análisis paralelo 
            tasks = [
                self._analyze_real_reddit_sentiment(symbol),
                self._analyze_real_youtube_sentiment(symbol),
                self._analyze_real_twitter_sentiment(symbol),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Procesar resultados REALES 
            reddit_result = results[0] if not isinstance(results[0], Exception) else {}
            youtube_result = results[1] if not isinstance(results[1], Exception) else {}
            twitter_result = results[2] if not isinstance(results[2], Exception) else {}

            # Actualizar análisis 
            analysis.update(reddit_result)
            analysis.update(youtube_result)
            analysis.update(twitter_result)

            # Calcular sentimiento general 
            social_sentiment = await self._calculate_social_sentiment(analysis)
            analysis.update(social_sentiment)

            self.logger.info("REAL social sentiment analysis completed for {symbol}")
            return analysis 

        except Exception as e:
            self.logger.error(f"Error in REAL social sentiment analysis for {symbol} : {e}")
            return self._get_default_analysis(symbol)

    @log_function_call 
    async def _analyze_real_reddit_sentiment(self, symbol: str) -> Dict[str, Any]:
        """ Analizar sentimiento REAL de Reddit """
        try:
            reddit_data = {
                'reddit_sentiment': 0.0,
                'reddit_mentions': 0,
                'reddit_upvotes': 0
            }

            if not self.reddit:
                self.logger.warning("Reddit API not available, skipping Reddit analysis")
                return reddit_data 

            all_posts = []
            all_comments = []

            # Obtener posts REALES de subreddits crypto 
            for subreddit_name in self.crypto_subreddits:
                try:
                    subreddit = self.reddit.subreddit(subreddit_name)

                    # Obtener posts calientes 
                    for post in subreddit.hot(limit=10):
                        if symbol.lower() in post.title.lower() or symbol.lower() in post.selftext.lower():
                            all_posts.append({
                                'title': post.title,
                                'text': post.selftext,
                                'score': post.score,
                                'upvote_ratio': post.upvote_ratio,
                                'subreddit': subreddit_name
                            })

                        # Obtener comentarios 
                        for comment in subreddit.comments(limit=50):
                            if symbol.lower() in comment.body.lower():
                                all_comments.append({
                                    'text': comment.body,
                                    'score': comment.score,
                                    'subreddit': subreddit_name
                                }) 
                            
                        except Exception as e:
                            self.logger.warning(f"Error fetching Reddit data from {subreddit_name}: {e}")

                    # Analizar senimiento REAL 
                    sentiments = []
                    total_upvotes = 0

                    for post in all_posts:
                        # Análisis de sentimiento del título y texto 
                        title_sentiment = await self._analyze_real_text_sentiment(post['title'])
                        text_sentiment = await self._analyze_real_text_sentiment(post['text'])

                        # Ponderar por upvotes 
                        weight = post['score'] / 1000.0
                        combined_sentiment = (title_sentiment + text_sentiment) / 2
                        weighted_sentiment = combined_sentiment * weight

                        sentiments.append(weighted_sentiment)
                        total_upvotes += post['score']

                    for comment in all_comments:
                        # Análisis de sentimiento del comentario 
                        comment_sentiment = await self._analyze_real_text_sentiment(comment['text'])

                        # Ponderar por score 
                        weight = comment['score'] / 100.0
                        weighted_sentiment = comment_sentiment * weight 

                        sentiments.append(weighted_sentiment)
                        total_upvotes += comment['score']

                    # Calcular métricas REALES 
                    if sentiments:
                        reddit_data['reddit_sentiment'] = float(np.mean(sentiments))
                        reddit_data['reddit_mentions'] = len(all_posts) + len(all_comments)
                        reddit_data['reddit_upvotes'] = total_upvotes 

                    self.logger.info(f"REAL Reddit analysis: {reddit_data['reddit_mentions']} mentions, {reddit_data['reddit_upvotes']} upvotes")
                    return reddit_data 

        except Exception as e:
            self.logger.error(f"Error analyzing REAL Reddit sentiment: {e}")
            return {'reddit_sentiment': 0.0, 'reddit_mentions': 0, 'reddit_upvotes':0}
        
    @log_function_call 
    async def _analyze_real_youtube_sentiment(self, symbol: str) -> Dict[str, Any]:
        """ Analizar sentimiento REAL de Youtube """
        try:
            youtube_data = {
                'youtube_sentiment': 0.0,
                'youtube_mentions': 0,
                'youtube_views': 0
            }

            if not self.youtube:
                self.logger.warning("Youtube API not available, skipping Youtube analysis")
                return youtube_data

            # Buscar videos REALES sobre el símbolo 
            search_query f"{symbol} cryptocurrency analysis"

            try:
                # Buscar vídeos 
                search_response = self.youtube.search().list(
                    q=search_query,
                    part='id,snippet',
                    maxResults=10,
                    type='video',
                    order='relevance'
                ).execute()

                sentiments = []
                total_views = 0

                for video in search_response.get('items', []):
                    video_id = video['id']['videoId']
                    title = video['snippet']['title']
                    description = video['snippet']['description']

                    # Obtener estadísticas del vídeo 
                    stats_response = self.youtube.videos().list(
                        part='statistics',
                        if=video_id
                    ).execute()

                    if stats_response.get('items'):
                        stats = stats_response['items'][0]['statistics']
                        view_count = int(stats.get('viewCount', 0))
                        total_views += view_count 

                        # Análisis de sentimiento del título y descripción 
                        title_sentiment = await self._analyze_real_text_sentiment(title)
                        desc_sentiment = await self._analyze_real_text_sentiment(description)

                        # Ponderar por views 
                        weight = min(1.0, view_count / 1000000.0)
                        combined_sentiment = (title_sentiment + desc_sentiment) / 2
                        weighted_sentiment = combined_sentiment * weight 

                        sentiments.append(weighted_sentiment)

                # Calcular métricas REALES 
                if sentiments:
                    youtube_data['youtube_sentiment'] = float(np.mean(sentiments))
                    youtube_data['youtube_mentions'] = len(sentiments)
                    youtube_data['youtube_views'] = total_views 

                self.logger.info(f"REAL Youtube analysis: {youtube_data['youtube_mentions']} mentions, {youtube_data['youtube_views']} views")
                return youtube_data

            except Exception as e:
                self.logger.warning(f"Error fetching Youtube data: {e}")
                return youtube_data  

        except Exception as e:
            self.logger.error(f"Error analyzing REAL Youtube sentiment: {e}")
            return {'youtube_sentiment': 0.0, 'youtube_mentions': 0, 'youtube_views':0}
        
    @log_function_call 
    async def _analyze_real_twitter_sentiment(self, symbol: str) -> Dict[str, Any]:
        """ Analizar sentimiento REAL de Twitter """
        try:
            twitter_data = {
                'twitter_sentiment': 0.0,
                'twitter_mentions': 0,
                'twitter_engagment': 0.0
            }

            if not self.twitter_bearer_token:
                self.logger.warning("Twitter API not available, skipping Twitter analysis")
                return twitter_data 

            # Twitter API v2 
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                'Authorization': f'Bearer {self.twitter_bearer_token}',
                'Content-Type': 'application/json'
            }
            params = {
                'query': f'{symbol} crypto',
                'max_results': 10,
                'tweet.fields': 'public_metrics,created_at'
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()

                        tweet = data.get('data', [])
                        sentiments = []
                        total_engagement = 0

                        for tweet in tweets:
                            text = tweet.get('text', '')
                            metrics = tweet.get('public_metrics', {})

                            # Análisis de sentimiento del tweet 
                            sentiment = await self._analyze_real_text_sentiment(text)

                            # Caluclar engagement 
                            engagement = (
                                metrics.get('like_count', 0) +
                                metrics.get('retweet_count', 0) + 
                                metrics.get('reply_count', 0)
                            )

                            # Ponderar por engagement 
                            weight = min(1.0, engagement / 1000.0)
                            weighted_sentiment = sentiment * weight 

                            sentiments.append(weighted_sentiment)
                            total_engagement += engagement 

                        # Calcular métricas REALES 
                        if sentiments:
                            twitter_data['twitter_sentiment'] = float(np.mean(sentiments))
                            twitter_data['twitter_mentions'] = len(sentiments)
                            twitter_data['twitter_engagment'] = total_engagement 

                        self.logger.info(f"REAL Twitter analysis: {twitter_data['twitter_mentions']} tweets, {twitter_data['twitter_engagement']} engagement")
                        return twitter_data
                    
                    else:
                        self.logger.warning(f"Twitter API error: {response.status}")
                        return twitter_data
        
        except Exception as e:
            self.logger.info(f"Error analyzing REAL Twitter sentiment: {e}")
            return {'twitter_sentiment': 0.0, 'twitter_mentions': 0, 'twitter_engagement': 0.0}


    @log_function_call 
    async def _analyze_real_text_sentiment(self, text: str) -> float:
        """ Analizar sentimiento REAL de texto usando TextBlob """
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
                    'launch', 'listing', 'breakout', 'resistance', 'support',
                    'hodl', 'diamond', 'hands', 'buy', 'long', 'bullish'
                ],
                'negative': [
                    'bear', 'crash', 'dump', 'decline', 'fall', 'regulation', 
                    'ban', 'scam', 'hack', 'exploit', 'delisting', 'rejection',
                    'sell', 'short', 'bearish', 'fud', 'panic', 'dump'
                ]
            }

            # Contar palabras clave 
            positive_count = sum(1 for word in crypto_keywords['positivw'] if word in text)
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
    async def _calculate_social_sentiment(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """ Calcular sentimiento social general """
        try:
            # Ponderar difernetes plataformas 
            weights = {
                'reddit': 0.35,
                'youtube': 0.3,
                'twitter': 0.35
            }

            # Calcular sentimiento ponderado 
            social_sentiment = (
                analysis.get('reddit_sentiment', 0.0) * weights['reddit'] +
                analysis.get('youtube_sentiment', 0.0) * weights['youtube'] + 
                analysis.get('twitter_sentiment', 0.0) * weights['twitter']
            )

            # Determine tendencia 
            if social_sentiment > 0.3:
                social_trend = 'BULLISH'
            elif social_sentiment > -0.3:
                social_trend = 'BEARISH'
            else:
                social_trend = 'NEUTRAL' 

            # Calcular confianza 
            confidence = min(1.0, max(0.0, abs(social_sentiment) * 2))

            return {
                'social_sentiment': float(social_sentiment),
                'social_trend': social_trend,
                'social_confidence': confidence
            }

        except Exception as e:
            self.logger.error(f"Error calculating social sentiment: {e}")
            return {'social_sentiment': 0.0, 'social_trend': 'NEUTRAL', 'social_confidence': 0.0}

    @log_function_call
    async def _check_real_apis(self):
        """ Verificar disponibilidad de APIs REALES """
        try:
            # Verificar Reddit API
            if self.reddit:
                try:
                    # Test simple 
                    subreddit = self.reddit.subreddit('cryptocurrency')
                    next(subreddit.hot(limit=1))
                    self.logger.info("REAL Reddit API working")
                except Exception as e:
                    self.logger.warning(f"REAL Reddit API error: {e}")
            else:
                self.logger.warning(f"REAL Reddit API not configured")

            
            # Verificar Youtube API 
            if self.youtube:
                try:
                    search_response = self.youtube.search().list(
                        q='bitcoin',
                        part='id',
                        maxResults=1
                    ).execute()
                    self.logger.info("REAL Youtube API working")
                except Exception as e:
                    self.logger.warning(f"REAL Youtube API error: {e}")
            else:
                self.logger.warning(f"REAL Youtube API not configured")

            # Verificar Twitter API 
            if self.twitter_bearer_token:
                try:
                    # Test simple
                    url = "https://api.twitter.com/2/tweets/search/recent"
                    headers = {'Authorization': f'Bearer {self.twitter_bearer_token}'}
                    params = {'query': 'bitcoin', 'max_results': 1}
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.get(url, headers=headers, params=params, timeout=5) as response:
                            if response.status == 200:
                                self.logger.info("✅ REAL Twitter API working (LIMITED)")
                            else:
                                self.logger.warning(f"⚠️ REAL Twitter API error: {response.status}")
                except Exception as e:
                    self.logger.warning(f"⚠️ REAL Twitter API error: {e}")
            else:
                self.logger.warning("⚠️ REAL Twitter API not configured (LIMITED)")
            
        except Exception as e:
            self.logger.error(f"Error checking REAL APIs: {e}")

    def _get_default_analysis(self, symbol: str) -> Dict[str, Any]:
        """Análisis por defecto"""
        return {
            'symbol': symbol,
            'timestamp': datetime.utcnow().isoformat(),
            'social_sentiment': 0.0,
            'social_confidence': 0.0,
            'social_trend': 'NEUTRAL',
            'twitter_sentiment': 0.0,
            'twitter_mentions': 0,
            'twitter_engagement': 0.0,
            'reddit_sentiment': 0.0,
            'reddit_mentions': 0,
            'reddit_upvotes': 0,
            'youtube_sentiment': 0.0,
            'youtube_mentions': 0,
            'youtube_views': 0,
            'trending_topics': [],
            'sentiment_breakdown': {}
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud del analizador de sentimiento"""
        try:
            return {
                'healthy': True,
                'message': 'REAL Social sentiment analyzer healthy (FREE APIs)',
                'reddit_configured': bool(self.reddit),
                'youtube_configured': bool(self.youtube),
                'twitter_configured': bool(self.twitter_bearer_token),
                'crypto_subreddits': len(self.crypto_subreddits),
                'timestamp': datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f'REAL Social sentiment analyzer unhealthy: {str(e)}',
                'timestamp': datetime.utcnow().isoformat()
            }
                
                