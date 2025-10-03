""" Motor de trading principal """
""" Orquesta todo el sistema de trading con arquitectura escalable  """

import asyncio 
import time 
from datetime import datetime, timedelta 
from typing import Dict, List, Optional, Any 
from dataclasses import dataclass 
from enum import Enum 

from utils.logger import get_logger, log_execution_time, log_function_call 
from config.settings import Settings 
from core.signal_generator import SignalGenerator 
from core.portfolio_manager import PortfolioManager 
from core.risk_manager import RiskManager  
from analysis.technical_analyzer import TechnicalAnalyzer 
from analysis.sentiment_analyzer import SentimentAnalyzer 
from analysis.news_analyzer import NewsAnalyzer  
from analysis.ml_predictor import MLPredictor
from analysis.data_fusion_hub import DataFusionHub 
from data_sources.binance_client import BinanceClient  
from notifications.telegram_bot import TelegramBot  
from database.signals_db import SignalsDatabase 

class EngineState(str, Enum):
    """ Estados del motor de trading """
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"

@dataclass 
class EngineMetrics:
    """ Métricas del motor de trading """
    start_time: datetime 
    signals_generated: int = 0
    signals_executed: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    total_pnl: float = 0.0
    active_positions: int = 0
    last_update: datetime = None 

    def __post_init__(self):
        self.last_update = datetime.now()

class TradingEngine:
    """ Motor principal de trading que orquesta todo el sistema
        - Gestión de ciclos de análisis 
        - Ejecución de estrategias 
        - Coordinación entre módulos 
        - Monitoreo de performance 
        
    """

    def __init__(self, settings: Settings):
        self.settings = settings 
        self.logger = get_logger("TradingEngine")
        self.state = EngineState.INITIALIZING
        self.metrics = EngineMetrics(start_time=datetime.now())

        # Componentes prinicpales 
        self.signal_generator: Optional[SignalGenerator] = None 
        self.risk_manager: Optional[RiskManager] = None 
        self.portfolio_manager: Optional[PortfolioManager] = None 
        self.technical_analyzer: Optional[TechnicalAnalyzer] = None 
        self.sentiment_analyzer: Optional[SentimentAnalyzer] = None 
        self.news_analyzer: Optional[NewsAnalyzer] = None 
        self.ml_predictor: Optional[MLPredictor] = None 
        self.data_fusion_hub: Optional[DataFusionHub] = None
        self.binance_client: Optional[BinanceClient] = None 
        self.telegram_bot: Optional[TelegramBot] = None 
        self.signals_db: Optional[SignalsDatabase] = None 

        # Tareas asíncronas 
        self.tasks: List[asyncio.Task] = []
        self.running = False 

        self.logger.info("TradingEngine initialized successfully")

    @log_execution_time 
    async def initialize(self) -> bool:
        """ Inicializar todos los componnetes del sistema """
        try:
            self.logger.info("Initializing trading engine components...")
            self.state = EngineState.INITIALIZING 

            # Inicializar base de datos 
            await self._initialize_database()

            # Inicializar clientes de datos 
            await self._initialize_data_clients()

            # Inicializar analizadores 
            await self._initialize_analyzers() 

            # Inicializar gestores 
            await self._initialize_managers()

            # Inicializar notificaciones 
            await self._initialize_notifications()
            
            # Inicializar generador de señales 
            await self._initialize_signal_generator()

            self.state = EngineState.RUNNING 
            self.logger.system_status("INITIALIZED", {
                "components": len(self._get_initialized_components())
                "monitored_symbols": len(self.settings.trading.monitored_symbols),
                "active_strategies": len(self.settings.trading.active_strategies)
            })

            return True 

        except Exception as e:
            self.logger.critical("Failed to initialize trading engine", exception=e)
            self.state = EngineState.ERROR
            return False 

    async def _initialize_database(self):
        """ Inicializar base de datos """
        self.logger.info("Initializing database...")
        from database.signals_db import SignalDatabase 
        self.signals_db = SignalDatabase(self.settings.database.url)
        await self.signals_db.initialize()
        self.logger.info("Database initialized")

    async def _initialize_data_clients(self):
        """ Inicializar clientes de datos """
        self.logger.info("Initializing data clients...")
        from data_sources.binance_client import BinanceClient  
        self.binance_client = BinanceClient(self.settings.api)
        await self.binance_client.initialize()
        self.logger.info("Data clients initialized")

    async def _initialize_analyzers(self):
        """ Inicializar analizadores """
        self.logger.info("Initializing analyzers...")

        # Análisis técnico 
        from analysis.technical_analyzer import TechnicalAnalyzer 
        self.technical_analyzer = TechnicalAnalyzer(self.settings, self.binance_client)
        await self.technical_analyzer.initialize()

        # Análisis de sentimiento 
        from analysis.sentiment_analyzer import SentimentAnalyzer 
        self.sentiment_analyzer = SentimentAnalyzer(self.settings.api)
        await self.sentiment_analyzer.initialize()

        # Análisis de noticias 
        from analysis.news_analyzer import NewsAnalyzer 
        self.news_analyzer = NewsAnalyzer(self.settings.api)
        await self.news_analyzer.initialize()

        # Machine learning 
        if self.settings.ml.enabled:
            from analysis.ml_predictor import MLPredictor  
            self.ml_predictor = MLPredictor(self.settings.ml)
            await self.ml_predictor.initialize()
        
        # Data Fusion Hub
        from analysis.data_fusion_hub import DataFusionHub
        self.data_fusion_hub = DataFusionHub()
        await self.data_fusion_hub.initialize()

        self.logger.info("Analyzers initialized")
    
    async def _initialize_managers(self):
        """ Inicializar gestores """
        self.logger.info("Initializing managers...")

        # Gestón de riesgo 
        from core.risk_manager import RiskManager
        self.risk_manager = RiskManager(self.settings.risk, self.signals_db)
        await self.risk_manager.initialize()

        # Gestión de portfolio 
        from core.portfolio_manager import PortfolioManager 
        self.portfolio_manager = PortfolioManager(
            initial_balance=self.settings.trading.initial_balance
        )
        await self.portfolio_manager.initialize()

        self.logger.info("Managers initialized")

    async def _initialize_notifications(self):
        """ Inicializar notificaciones """
        if self.settings.notifications.telegram_enabled:
            self.logger.info("Initalizing Telegram bot...")
            from notifications.telegram_bot import TelegramBot
            self.telegram_bot = TelegramBot(
                bot_token=self.settings.notifications.telegram_bot_token,
                chat_id=self.settings.notifications.telegram_chat_id
            )
            await self.telegram_bot.initialize()
            self.logger.info("Telegram bot initialized")

    async def _initialize_signal_generator(self):
        """ Inicializar generador de señales """
        self.logger.info("Initializing signal generator...")
        self.signal_generator = SignalGenerator (
            self.settings.trading, 
            self.technical_analyzer, 
            self.sentiment_analyzer, 
            self.news_analyzer, 
            self.ml_predictor, 
            self.data_fusion_hub,
            self.risk_manager, 
            self.signals_db
        )
        await self.signal_generator.initialize()
        self.logger.info("Signal generator initialized")
        
        
    def _get_initialized_components(self) -> List[str]:
        """ Obtener lista de componentes inicializados """
        components = []
        if self.signals_db: components.append("Database")
        if self.binance_client: components.append("BinanceClient")
        if self.technical_analyzer: components.append("TechnicalAnalyzer")
        if self.sentiment_analyzer: components.append("SentimentAnalyzer")
        if self.news_analyzer: components.append("NewsAnalyzer")
        if self.ml_predictor: components.append("MLPredictor")
        if self.data_fusion_hub: components.append("DataFusionHub")
        if self.risk_manager: components.append("RiskManager")
        if self.portfolio_manager: components.append("PortfolioManager")
        if self.signal_generator: components.append("SignalGenerator")
        if self.telegram_bot: components.append("TelegramBot")
        return components 

    @log_execution_time
    async def run(self):
        """ Ejecutar el motor de trading """
        if self.state != EngineState.RUNNING:
            self.logger.error("Engine is not running")
            return 

        self.running = True 
        self.logger.info("Starting trading engine main loop...")

        try:
            # Crear tareas asincronas 
            self.tasks = [
                asyncio.create_task(self._main_trading_loop()),
                asyncio.create_task(self._monitoring_loop()),
                asyncio.create_task(self._performance_loop()),
                asyncio.create_task(self._health_check_loop())
            ]

            # Ejecutar todas las tareas 
            await asyncio.gather(*self.tasks, return_exceptions=True)

        except Exception as e:
            self.logger.critical("Error in main trading loop", exception=e)
            self.state = EngineState.ERROR
        finally:
            await self.shutdown()

    @log_function_call
    async def _main_trading_loop(self):
        """ Loop principal de trading """
        while self.running:
            try:
                start_time = time.time()

                # Generar señales para todos los símbolos 
                await self._generate_signals_for_all_symbols()

                # Procesar señales activas
                await self._process_active_signals()

                # Actualizar métricas 
                self._update_metrics()

                # Log de perfromance 
                execution_time = time.time() - start_time
                self.logger.performance_metric("main_trading_loop_time", execution_time, "seconds")

                # Esperar antes del sisguiente ciclo
                await asyncio.sleep(self.settings.system.data_refresh_seconds)

            except Exception as e:
                self.logger.error("Error in main trading loop", exception=e)
                await asyncio.sleep(10) # Esperar antes de reintentar 

    @log_function_call
    async def _generate_signals_for_all_symbols(self):
        """ Generar señales para todos los símbolos monitoreados """
        for symbol in self.settings.trading.monitored_symbols:
            try:
                # Generar señales para el símbolo 
                signals = await self.signal_generator.generate_signals(symbol)

                if signals:
                    self.logger.info(f"Generated {len(signals)} signals for {symbol}")
                    self.metrics.signals_generated += len(signals)

                    # Enviar notificaciones 
                    if self.telegram_bot:
                        await self.telegram_bot.send_signals(signals)
            except Exception as e:
                self.logger.error(f"Error generating signals for {symbol}", exceptions=e)

    @log_function_call
    async def _process_active_signals(self):
        """ Procesar señales acivas """
        try:
            # Obtener señales activas 
            active_signals = await self.signals_db.get_active_signals()

            for signal in active_signals:
                try:
                    # Verificar si la señal debe ser ejecutada 
                    if await self.portfolio_manager.should_exectute_signal(signal):
                        # Ejecutar señal 
                        execution_result = await self.portfolio_manager.execute_signal(signal)

                        if execution_result['success']:
                            self.metrics.signals_executed += 1
                            self.logger.trading_execution(execution_result)
                        else:
                            self.metrics.failed_signals += 1
                            self.logger.warning(f"Failed to execute signals: {execution_result['error']}")

                    # Verificarn si la señal debe ser cerrada
                    elif await self.porfolio_manager.should_close_signal(signal):
                        # Cerrar señal
                        close_reult = await self.portfolio_manager.close_signal(signal)

                        if close_result['success']:
                            self.logger.trading_execution(close_result)
                        else:
                            self.logger.warning(f"Failed to close signal: {close_result['error']}")
                except Exception as e:
                    self.logger.error(f"Error processing signal {signal.get('id', 'unknown')}", exception=e)
                
        except Exception as e:
            self.logger.error("Error processing active signals:", exception=e)

    @log_function_call
    async def _monitoring_loop(self):
        """ Loop de monitoreo del sistema """
        while self.running:
            try:
                # Verificar estado de componentes 
                await self._check_component_health()

                # Verificar métricas de performance 
                await self._check_performance_metrics()

                # Esperar antes del siguiente check 
                await asyncio.loop(60) # Check cada minuto 

            except Exception as e:
                self.logger.error("Error in monitoring loop", exception=e)
                await asyncio.sleep(30)

    @log_function_call
    async def _performance_loop(self):
        """ Loop de métricas de performance """
        while self.running:
            try:
                #Calcular métricas de performance 
                performance_data = await self._calculate_performance_metrics()

                # Log de métricas 
                self.logger.performance_metric("total_signals", performance_data['total_signals'])
                self.logger.performance_metric("success_rate", performance_data['success_rate'], "%")
                self.logger.performance_metric("total_pnl", performance_data['total_pnl'], "USD")

                # Esperar antes del siguiente calculo 
                await asyncio.sleep(self.settings.system.performance_metrics_interval)

            except Exception as e:
                self.logger.error("Error in performance loop", exception=e)
                await asyncio.sleep(60)

    @log_function_call
    async def _health_check_loop(self):
        """ Loop de health check """
        while self.running:
            try:
                # Verificar salud del sistema
                health_status = await self._check_system_health()

                if not health_status['healthy']:
                    self.logger.warning(f"System health check failed: {health_status['issues']}")

                # Esperar antes del siguiente check 
                await asyncio.sleep(self.settings.system.health_check_interval)

            except Exception as e:
                self.logger.error("Error in health check loop", exception=e)
                await asyncio.sleep(30)

    async def _check_component_health(self):
        """ Verificar salud de los componentes """
        components_status = {}
       
        # Verificar cada componente
        if self.binance_client:
            components_status['binance_client'] = await self.binance_client.health_check()
        
        if self.technical_analyzer:
            components_status['technical_analyzer'] = await self.technical_analyzer.health_check()
        
        if self.sentiment_analyzer:
            components_status['sentiment_analyzer'] = await self.sentiment_analyzer.health_check()
        
        if self.news_analyzer:
            components_status['news_analyzer'] = await self.news_analyzer.health_check()
        
        if self.ml_predictor:
            components_status['ml_predictor'] = await self.ml_predictor.health_check()
        
        # Log de estado de componentes
        for component, status in components_status.items():
            if not status['healthy']:
                self.logger.warning(f"Component {component} unhealthy: {status['message']}")

    async def _check_performance_metrics(self):
        """ Verificar métricas de performance """
        # Verificar uso de memoria 
        import psutil 
        memeory_usage = psutil.virtual_memory().percent / 100 

        if memory_usage > self.settings.system.memory_usage_threshold:
            self.logger.warning(f"High memory usage: {memory_usage:.1%}")

        # Verificar latencia de API's 
        if self.binance_client:
            latency = await self.binance_client.get_latency()
            if latency > 5.0:
                self.logger.warning(f"High API latency: {latency:.2f}s")

    async def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """ Calcular métricas de performance """
        try:
            # Obtener estadísticas de la base dedatos 
           stats = await self.signals_db.get_performance_stats()

           return {
            'total_signals': stats.get('total_signals', 0),
            'successful_signals': stats.get('successful_signals', 0),
            'success_rate': stats.get('success_rate', 0.0),
            'total_pnl': stats.get('total_pnl', 0.0),
            'active_positions': stats.get('active_positions', 0),
            'uptime_hours': (datetime.utcnow() - self.metrics.start_time).total_seconds() / 3600
           }
        except Exception as e: 
            self.logger.error("Error calculating performance metric", exception=e)
            return {}
    
    async def _check_system_health(self) -> Dict[str, Any]:
        """ Verificar salud general del sistema """
        issues = []

        # Verificar estado del motor 
        if self.state != EngineState.RUNNING:
            issues.append(f"Engine state: {self.state}")

        # Verificar componentes criticos 
        critical_components = [
            ('signal_generator', self.signal_generator),
            ('risk_manager', self.risk_manager),
            ('portfolio_manager', self.portfolio_manager),
            ('technical_analyzer', self.technical_analyzer),
        ]

        for name, component in critical_components:
            if not component:
                issues.append(f"Missing component: {name}")

        return {
            'healthy': len(issues) == 0,
            'issues': issues,
            'timestamp': datetime.utcnow().isoformat()
        }

    def _update_metrics(self):
        """ Actualizar métricas del motor """
        self.metrics.last_update = datetime.utcnow()

    async def pause(self):
        """ Pausar el motor de trading """
        self.logger.info("Pausing trading engine...")
        self.state = EngineState.PAUSED
        self.running = False 

    async def resume(self):
        """ Reanudar el motor de trading """
        self.logger.info("Resuming trading engine...")
        self.state = EngineState.RUNNING
        self.running = True 

    async def shutdown(self):
        """ Cerrar el motor de trading """
        self.logger.info("Shutting down trading engine...")
        self.state = EngineState.STOPPING
        self.running = False 

        # Cancelar tareas asincronas 
        for task in self.tasks:
            if not task.done():
                task.cancel()

        # Cerrar componentes 
        if self.binance_client:
            await self.binance_client.close()

        if telegram_bot:
            await self.telegram_bot.close()

        if signals_db:
            await self.signals_db.close()

        self.state = EngineState.STOPPED
        self.logger.system_status("SHUTDOWN", {
            "uptime_hours": (datetime.utcnow() - self.metrics.start_time).total_seconds() / 3600
            "total_signals": self.metrics.signals_generated,
            "executed_signals": self.metrics.signals_executed,
        })

    def get_status(self) -> Dict[str, Any]:
        """ Obtener estado actual del motor """
        return {
            'state': self.state.value,
            'running': self.running,
            'metrics': {
                'signals_generated': self.metrics.signals_generated,
                'signals_executed': self.metrics.signals_executed,
                'successful_signals': self.metrics.successful_signals,
                'failed_signals': self.metrics.failed_signals,
                'total_pnl': self.metrics.total_pnl,
                'active_positions': self.metrics.active_positions,
                'uptime_hours': (datetime.utcnow() - self.metrics.start_time).total_seconds() / 3600
            },
            'components': self._get_initialized_components()
            'timestamp': datetime.utcnow().isoformat()
        }
            

        
