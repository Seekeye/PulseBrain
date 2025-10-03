#!/usr/bin/env python3
"""
Cliente Binance - CryptoPulse Pro
Interfaz para obtener datos de mercado de Binance
"""

import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import time

from utils.logger import get_logger, log_execution_time, log_function_call
from config.settings import APIConfig

class BinanceClient:
    """
    Cliente para la API de Binance
    - Obtener precios en tiempo real
    - Datos históricos (Klines)
    - Información de mercado
    - Gestión de rate limits
    """
    
    def __init__(self, api_config: APIConfig):
        self.api_config = api_config
        self.logger = get_logger("BinanceClient")
        
        # URLs de la API
        self.base_url = "https://api.binance.com"
        self.testnet_url = "https://testnet.binance.vision"
        
        # Rate limiting
        self.rate_limits = {
            'weight': 0,
            'last_reset': time.time()
        }
        
        # Configuración de timeout y reintentos
        self.timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.max_retries = 3
        
        # Cache de datos
        self.price_cache = {}
        self.cache_ttl = 60  # 1 minuto
        
        self.logger.info("🔗 BinanceClient initialized")
    
    async def _make_request_with_retry(self, url: str, params: Dict = None) -> Optional[Dict]:
        """Hacer request con reintentos y timeout"""
        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession(timeout=self.timeout) as session:
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            self.logger.warning(f"HTTP {response.status} on attempt {attempt + 1}")
                            if attempt == self.max_retries - 1:
                                return None
            except asyncio.TimeoutError:
                self.logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    return None
            except Exception as e:
                self.logger.warning(f"Error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    return None
            
            # Esperar antes del siguiente intento
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Backoff exponencial
        
        return None
    
    async def initialize(self):
        """Inicializar el cliente"""
        self.logger.info("🔧 Initializing Binance client...")
        
        # Verificar conectividad
        if await self.health_check():
            self.logger.info("✅ Binance client initialized")
        else:
            self.logger.warning("⚠️ Binance client initialization failed")
    
    @log_execution_time
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Obtener precio actual del símbolo"""
        try:
            # Verificar cache
            if symbol in self.price_cache:
                cached_data = self.price_cache[symbol]
                if time.time() - cached_data['timestamp'] < self.cache_ttl:
                    return cached_data['price']
            
            # Obtener precio de la API
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {'symbol': symbol}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        price = float(data['price'])
                        
                        # Actualizar cache
                        self.price_cache[symbol] = {
                            'price': price,
                            'timestamp': time.time()
                        }
                        
                        return price
                    else:
                        self.logger.error(f"Error getting price for {symbol}: {response.status}")
                        return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price for {symbol}", exception=e)
            return None
    
    @log_execution_time
    async def get_klines(self, symbol: str, interval: str, limit: int = 500, 
                        start_time: Optional[int] = None, end_time: Optional[int] = None) -> List[Dict[str, Any]]:
        """Obtener datos de velas (Klines)"""
        try:
            url = f"{self.base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': min(limit, 1000)  # Máximo 1000 por request
            }
            
            if start_time:
                params['startTime'] = start_time
            if end_time:
                params['endTime'] = end_time
            
            data = await self._make_request_with_retry(url, params)
            if data:
                # Convertir a formato estándar
                klines = []
                for kline in data:
                    klines.append({
                        'timestamp': int(kline[0]),
                        'open': float(kline[1]),
                        'high': float(kline[2]),
                        'low': float(kline[3]),
                        'close': float(kline[4]),
                        'volume': float(kline[5]),
                        'close_time': int(kline[6]),
                        'quote_volume': float(kline[7]),
                        'trades': int(kline[8]),
                        'taker_buy_base_volume': float(kline[9]),
                        'taker_buy_quote_volume': float(kline[10])
                    })
                
                return klines
            else:
                self.logger.error(f"Error getting klines for {symbol}: Request failed")
                return []
            
        except Exception as e:
            self.logger.error(f"Error getting klines for {symbol}", exception=e)
            return []
    
    @log_execution_time
    async def get_24hr_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Obtener estadísticas de 24 horas"""
        try:
            url = f"{self.base_url}/api/v3/ticker/24hr"
            params = {'symbol': symbol}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        return {
                            'symbol': data['symbol'],
                            'price_change': float(data['priceChange']),
                            'price_change_percent': float(data['priceChangePercent']),
                            'weighted_avg_price': float(data['weightedAvgPrice']),
                            'prev_close_price': float(data['prevClosePrice']),
                            'last_price': float(data['lastPrice']),
                            'last_qty': float(data['lastQty']),
                            'bid_price': float(data['bidPrice']),
                            'ask_price': float(data['askPrice']),
                            'open_price': float(data['openPrice']),
                            'high_price': float(data['highPrice']),
                            'low_price': float(data['lowPrice']),
                            'volume': float(data['volume']),
                            'quote_volume': float(data['quoteVolume']),
                            'open_time': int(data['openTime']),
                            'close_time': int(data['closeTime']),
                            'count': int(data['count'])
                        }
                    else:
                        self.logger.error(f"Error getting 24hr ticker for {symbol}: {response.status}")
                        return None
            
        except Exception as e:
            self.logger.error(f"Error getting 24hr ticker for {symbol}", exception=e)
            return None
    
    @log_execution_time
    async def get_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict[str, Any]]:
        """Obtener libro de órdenes"""
        try:
            url = f"{self.base_url}/api/v3/depth"
            params = {
                'symbol': symbol,
                'limit': min(limit, 5000)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        return {
                            'last_update_id': data['lastUpdateId'],
                            'bids': [[float(bid[0]), float(bid[1])] for bid in data['bids']],
                            'asks': [[float(ask[0]), float(ask[1])] for ask in data['asks']]
                        }
                    else:
                        self.logger.error(f"Error getting order book for {symbol}: {response.status}")
                        return None
            
        except Exception as e:
            self.logger.error(f"Error getting order book for {symbol}", exception=e)
            return None
    
    @log_execution_time
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtener trades recientes"""
        try:
            url = f"{self.base_url}/api/v3/trades"
            params = {
                'symbol': symbol,
                'limit': min(limit, 1000)
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        trades = []
                        for trade in data:
                            trades.append({
                                'id': int(trade['id']),
                                'price': float(trade['price']),
                                'qty': float(trade['qty']),
                                'quote_qty': float(trade['quoteQty']),
                                'time': int(trade['time']),
                                'is_buyer_maker': trade['isBuyerMaker'],
                                'is_best_match': trade['isBestMatch']
                            })
                        
                        return trades
                    else:
                        self.logger.error(f"Error getting recent trades for {symbol}: {response.status}")
                        return []
            
        except Exception as e:
            self.logger.error(f"Error getting recent trades for {symbol}", exception=e)
            return []
    
    @log_function_call
    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Obtener información del símbolo"""
        try:
            url = f"{self.base_url}/api/v3/exchangeInfo"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Buscar el símbolo específico
                        for symbol_info in data['symbols']:
                            if symbol_info['symbol'] == symbol:
                                return {
                                    'symbol': symbol_info['symbol'],
                                    'status': symbol_info['status'],
                                    'base_asset': symbol_info['baseAsset'],
                                    'quote_asset': symbol_info['quoteAsset'],
                                    'base_asset_precision': symbol_info['baseAssetPrecision'],
                                    'quote_precision': symbol_info['quotePrecision'],
                                    'filters': symbol_info['filters']
                                }
                        
                        return None
                    else:
                        self.logger.error(f"Error getting symbol info: {response.status}")
                        return None
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info for {symbol}", exception=e)
            return None
    
    @log_function_call
    async def get_server_time(self) -> Optional[int]:
        """Obtener tiempo del servidor"""
        try:
            url = f"{self.base_url}/api/v3/time"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data['serverTime']
                    else:
                        self.logger.error(f"Error getting server time: {response.status}")
                        return None
            
        except Exception as e:
            self.logger.error("Error getting server time", exception=e)
            return None
    
    @log_function_call
    async def health_check(self) -> Dict[str, Any]:
        """Verificar salud de la API"""
        try:
            start_time = time.time()
            
            # Obtener tiempo del servidor
            server_time = await self.get_server_time()
            
            if server_time:
                latency = time.time() - start_time
                
                return {
                    'healthy': True,
                    'latency': latency,
                    'server_time': server_time,
                    'message': 'Binance API is accessible'
                }
            else:
                return {
                    'healthy': False,
                    'latency': 0,
                    'server_time': 0,
                    'message': 'Failed to connect to Binance API'
                }
            
        except Exception as e:
            self.logger.error("Error in health check", exception=e)
            return {
                'healthy': False,
                'latency': 0,
                'server_time': 0,
                'message': f'Health check failed: {str(e)}'
            }
    
    @log_function_call
    async def get_latency(self) -> float:
        """Obtener latencia de la API"""
        try:
            start_time = time.time()
            await self.get_server_time()
            return time.time() - start_time
        except Exception as e:
            self.logger.error("Error measuring latency", exception=e)
            return 999.0  # Latencia muy alta en caso de error
    
    async def close(self):
        """Cerrar conexiones"""
        self.logger.info("✅ Binance client closed")
