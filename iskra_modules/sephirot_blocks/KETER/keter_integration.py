"""
KETHER INTEGRATION MODULE v2.0
Мост для интеграции Ketheric Block с системой ISKRA-4
Сефира: KETER (Венец)
"""

import asyncio
import time
import sys
import os
from typing import Dict, Any, Optional, List
import logging

# ============================================================
# 1. НАСТРОЙКА ПУТЕЙ И ИМПОРТОВ
# ============================================================

# Добавляем пути для импорта
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # iskra_modules
sys.path.insert(0, current_dir)  # sephirot_blocks/KETER

try:
    # Импортируем ядро KETER
    from keter_core import KetherCore, create_keter_core
    from keter_api import KetherAPI, create_keter_core_with_api
    
    # Импортируем архитектуру ISKRA-4
    from sephirot_base import ISephiraModule, SephiraConfig, EnergyLevel
    from sephirot_bus import SephirotBus, EventMessage
    import importlib
    sephirotic_engine_module = importlib.import_module("iskra_modules.sephirot_blocks.sephirotic_engine")
    SephiroticEngine = sephirotic_engine_module.SephiroticEngine
    SephirotIntegration = getattr(sephirotic_engine_module, "SephirotIntegration", None)
    
    KETER_MODULES_AVAILABLE = True
    ISKRA_ARCHITECTURE_AVAILABLE = True
    
except ImportError as e:
    logging.warning(f"Не удалось импортировать зависимости: {e}")
    KETER_MODULES_AVAILABLE = False
    ISKRA_ARCHITECTURE_AVAILABLE = False
    
    # Заглушки для разработки
    class ISephiraModule:
        async def initialize(self, config): pass
        async def activate(self): return True
        async def deactivate(self): pass
        async def process_energy(self, energy_type, amount): return True
        async def receive_event(self, event): pass
        async def get_state(self): return {}
    
    class SephiraConfig:
        pass
    
    class EnergyLevel:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    class SephirotBus:
        async def publish(self, event): pass
        async def subscribe(self, sephira_name, handler): pass
    
    class EventMessage:
        def __init__(self, event_type, data, source, target=None):
            self.event_type = event_type
            self.data = data
            self.source = source
            self.target = target
    
    class SephiroticEngine:
        pass
    
    class SephirotIntegration:
        pass

# ============================================================
# 2. КЛАСС ИНТЕГРАЦИИ KETHER В ISKRA-4
# ============================================================

class KeterIntegration(ISephiraModule):
    """
    Интеграционный модуль сефиры KETER для системы ISKRA-4
    Реализует интерфейс ISephiraModule и связывает KetherCore с SephiroticEngine
    """
    
    __sephira_name__ = "KETER"
    __sephira_number__ = 1
    __version__ = "2.0.0"
    
    def __init__(self, config: Optional[SephiraConfig] = None):
        self.logger = logging.getLogger(f"KeterIntegration")
        
        # Конфигурация сефиры
        self.config = config or SephiraConfig()
        
        # Ядро KETHER
        self.keter_core = None
        
        # API шлюз
        self.keter_api = None
        
        # Связь с ISKRA-4
        self.sephirot_bus = None
        self.sephirotic_engine = None
        
        # Состояние
        self.is_initialized = False
        self.is_active = False
        self.energy_level = EnergyLevel.LOW
        self.last_energy_update = 0
        self.integration_start_time = 0
        
        # Подписки на события ISKRA-4
        self.event_handlers = {}
        
        # Карта преобразования событий KETER -> ISKRA-4
        self.event_mapping = {
            # События модулей KETER
            "module.activated": "sephirot.module_activated",
            "module.deactivated": "sephirot.module_deactivated",
            "module.recovered": "sephirot.module_recovered",
            "module.failed": "sephirot.module_failed",
            
            # Энергетические события
            "energy.distributed": "sephirot.energy_flow",
            "energy.recharged": "sephirot.energy_recharged",
            "energy.critical": "sephirot.energy_critical",
            
            # Системные события
            "system.critical_warning": "sephirot.system_warning",
            "recovery.auto_completed": "sephirot.recovery_completed",
            "recovery.emergency_completed": "sephirot.emergency_recovery",
            "api.error": "sephirot.api_error",
            "system.shutdown": "sephirot.shutdown_initiated",
            
            # События от конкретных модулей
            "spiritual.synthesis": "keter.spiritual.synthesis",
            "willpower.boost": "keter.willpower.boost",
            "moral.alert": "keter.moral.alert",
            "policy.escalate": "keter.policy.escalate",
        }
        
        # Обратное преобразование событий ISKRA-4 -> KETER
        self.reverse_event_mapping = {
            "sephirot.energy_request": "energy.request",
            "sephirot.state_query": "system.status_request",
            "sephirot.command.activate": "system.activate",
            "sephirot.command.deactivate": "system.shutdown",
            "sephirot.config_update": "config.update",
            
            # События от других сефир
            "chokhmah.wisdom.update": "external.wisdom",
            "binah.understanding.update": "external.understanding",
            "chesed.mercy.update": "external.mercy",
            "gevurah.judgment.update": "external.judgment",
        }
        
        self.logger.info(f"KeterIntegration v{self.__version__} инициализирован")
    
    # ========================================================
    # 3. ИНТЕРФЕЙС ISephiraModule
    # ========================================================
    
    async def initialize(self, config: SephiraConfig) -> bool:
        """
        Инициализация сефиры KETER в системе ISKRA-4
        """
        if self.is_initialized:
            self.logger.warning("KeterIntegration уже инициализирован")
            return True
        
        self.logger.info("🚀 Инициализация KeterIntegration...")
        
        try:
            # Сохраняем конфигурацию
            self.config = config
            
            # Создаём ядро KETER
            keter_config = self._convert_to_keter_config(config)
            self.keter_core = create_keter_core(keter_config)
            
            # Создаём API шлюз
            self.keter_api = KetherAPI(self.keter_core)
            
            # Устанавливаем начальные значения
            self.is_initialized = True
            self.integration_start_time = time.time()
            
            # Регистрируем обработчики событий KETER
            await self._register_keter_event_handlers()
            
            self.logger.info("✅ KeterIntegration успешно инициализирован")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации KeterIntegration: {e}")
            self.is_initialized = False
            return False
    
    async def activate(self) -> bool:
        """
        Активация сефиры KETER
        """
        if not self.is_initialized:
            self.logger.error("KeterIntegration не инициализирован")
            return False
        
        if self.is_active:
            self.logger.warning("KeterIntegration уже активен")
            return True
        
        self.logger.info("⚡ Активация KeterIntegration...")
        
        try:
            # 1. Регистрируем модули KETER
            registration_result = await self.keter_core.register_all_modules()
            
            if not any("registered" in str(v) for v in registration_result.values()):
                self.logger.error("Не удалось зарегистрировать модули KETER")
                return False
            
            # 2. Запускаем каскадную активацию
            activation_result = await self.keter_core.activate_cascade()
            
            if activation_result["activated_modules"] == 0:
                self.logger.error("Не удалось активировать модули KETER")
                return False
            
            # 3. Обновляем состояние
            self.is_active = True
            self.energy_level = EnergyLevel.HIGH
            
            # 4. Публикуем событие активации в ISKRA-4
            if self.sephirot_bus:
                await self.sephirot_bus.publish(EventMessage(
                    event_type="sephirot.activated",
                    data={
                        "sephira": self.__sephira_name__,
                        "version": self.__version__,
                        "modules_activated": activation_result["activated_modules"],
                        "total_modules": activation_result["total_modules"],
                        "activation_time": activation_result["total_time"]
                    },
                    source=self.__sephira_name__
                ))
            
            self.logger.info(f"✅ KeterIntegration активирован. Модулей: {activation_result['activated_modules']}/{activation_result['total_modules']}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка активации KeterIntegration: {e}")
            self.is_active = False
            return False
    
    async def deactivate(self) -> bool:
        """
        Деактивация сефиры KETER
        """
        if not self.is_active:
            self.logger.warning("KeterIntegration уже неактивен")
            return True
        
        self.logger.info("🛑 Деактивация KeterIntegration...")
        
        try:
            # 1. Останавливаем ядро KETER
            if self.keter_core:
                shutdown_result = await self.keter_core.shutdown()
                
                if not shutdown_result.get("shutdown_completed", False):
                    self.logger.warning("Неполное выключение KeterCore")
            
            # 2. Обновляем состояние
            self.is_active = False
            self.energy_level = EnergyLevel.LOW
            
            # 3. Публикуем событие деактивации в ISKRA-4
            if self.sephirot_bus:
                await self.sephirot_bus.publish(EventMessage(
                    event_type="sephirot.deactivated",
                    data={
                        "sephira": self.__sephira_name__,
                        "reason": "graceful_shutdown",
                        "timestamp": time.time()
                    },
                    source=self.__sephira_name__
                ))
            
            self.logger.info("✅ KeterIntegration деактивирован")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка деактивации KeterIntegration: {e}")
            return False
    
    async def process_energy(self, energy_type: str, amount: float) -> bool:
        """
        Обработка энергии от системы ISKRA-4
        """
        if not self.is_active or not self.keter_core:
            self.logger.error("KeterIntegration не активен для обработки энергии")
            return False
        
        self.logger.info(f"⚡ Получена энергия: {energy_type} ({amount} units)")
        
        try:
            # Обновляем уровень энергии
            self.last_energy_update = time.time()
            
            # Пополняем резерв KETER
            success = await self.keter_core.recharge_energy(amount)
            
            if success:
                # Обновляем уровень энергии в соответствии с резервом
                reserve = self.keter_core.energy_reserve
                critical_threshold = self.keter_core.config["energy"]["critical_threshold"]
                
                if reserve < critical_threshold * 0.3:
                    self.energy_level = EnergyLevel.CRITICAL
                elif reserve < critical_threshold:
                    self.energy_level = EnergyLevel.LOW
                elif reserve < critical_threshold * 2:
                    self.energy_level = EnergyLevel.MEDIUM
                else:
                    self.energy_level = EnergyLevel.HIGH
                
                # Публикуем событие обновления энергии
                if self.sephirot_bus:
                    await self.sephirot_bus.publish(EventMessage(
                        event_type="sephirot.energy_processed",
                        data={
                            "sephira": self.__sephira_name__,
                            "energy_type": energy_type,
                            "amount": amount,
                            "new_reserve": reserve,
                            "energy_level": self.energy_level
                        },
                        source=self.__sephira_name__
                    ))
                
                return True
            else:
                self.logger.warning("Не удалось обработать энергию")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка обработки энергии: {e}")
            return False
    
    async def receive_event(self, event: EventMessage) -> bool:
        """
        Приём события от системы ISKRA-4
        """
        if not self.is_active:
            self.logger.warning(f"KeterIntegration не активен, игнорируем событие: {event.event_type}")
            return False
        
        self.logger.debug(f"📩 Получено событие ISKRA-4: {event.event_type} от {event.source}")
        
        try:
            # Преобразуем событие ISKRA-4 в событие KETER
            keter_event_type = self._map_to_keter_event(event.event_type)
            
            if not keter_event_type:
                self.logger.warning(f"Неизвестное событие ISKRA-4: {event.event_type}")
                return False
            
            # Маршрутизируем событие в KETER
            await self.keter_core.route_event(
                keter_event_type,
                event.data,
                f"iskra_{event.source}" if event.source != self.__sephira_name__ else "iskra_engine"
            )
            
            # Вызываем специфичные обработчики
            if event.event_type in self.event_handlers:
                for handler in self.event_handlers[event.event_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        self.logger.error(f"Ошибка обработчика события {event.event_type}: {e}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка обработки события ISKRA-4: {e}")
            return False
    
    async def get_state(self) -> Dict[str, Any]:
        """
        Получение текущего состояния сефиры KETER
        """
        state = {
            "sephira": self.__sephira_name__,
            "number": self.__sephira_number__,
            "version": self.__version__,
            "initialized": self.is_initialized,
            "active": self.is_active,
            "energy_level": self.energy_level,
            "last_energy_update": self.last_energy_update,
            "integration_uptime": time.time() - self.integration_start_time if self.integration_start_time > 0 else 0,
            "timestamp": time.time()
        }
        
        if self.keter_core:
            try:
                # Получаем состояние KETER
                keter_metrics = await self.keter_core.collect_metrics()
                
                state.update({
                    "keter_core": {
                        "modules_registered": len(self.keter_core.modules),
                        "modules_active": sum(1 for m in self.keter_core.modules.values() if m.is_active),
                        "energy_reserve": self.keter_core.energy_reserve,
                        "event_queue_size": self.keter_core.event_queue.qsize(),
                        "background_tasks": len(self.keter_core.background_tasks),
                        "uptime": keter_metrics["system"]["uptime"],
                        "health_percentage": (sum(1 for m in self.keter_core.modules.values() if m.is_active) / 
                                            len(self.keter_core.modules) * 100) if self.keter_core.modules else 0
                    }
                })
                
                # Информация о модулях KETER
                modules_info = []
                for name, module in self.keter_core.modules.items():
                    modules_info.append({
                        "name": name,
                        "active": module.is_active,
                        "order": module.activation_order,
                        "dependencies": module.dependencies
                    })
                
                state["modules"] = modules_info
                
            except Exception as e:
                state["keter_core_error"] = str(e)
        
        return state
    
    # ========================================================
    # 4. ИНТЕГРАЦИЯ С ISKRA-4 АРХИТЕКТУРОЙ
    # ========================================================
    
    async def connect_to_iskra(self, 
                              sephirot_bus: SephirotBus,
                              sephirotic_engine: SephiroticEngine) -> bool:
        """
        Подключение к архитектуре ISKRA-4
        """
        self.logger.info("🔗 Подключение к архитектуре ISKRA-4...")
        
        try:
            # Сохраняем ссылки
            self.sephirot_bus = sephirot_bus
            self.sephirotic_engine = sephirotic_engine
            
            # Подписываемся на события ISKRA-4
            await self._subscribe_to_iskra_events()
            
            # Регистрируем сефиру в движке
            if hasattr(sephirotic_engine, 'register_sephira'):
                registration_result = await sephirotic_engine.register_sephira(
                    self.__sephira_name__,
                    self,
                    {
                        "version": self.__version__,
                        "position": self.__sephira_number__,
                        "capabilities": ["spirit_synthesis", "willpower", "morality", "governance"],
                        "energy_requirements": {
                            "min": 100.0,
                            "optimal": 500.0,
                            "max": 1000.0
                        }
                    }
                )
                
                if not registration_result.get("success", False):
                    self.logger.error("Не удалось зарегистрировать сефиру в движке")
                    return False
            
            self.logger.info("✅ Успешно подключено к ISKRA-4")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка подключения к ISKRA-4: {e}")
            return False
    
    async def _subscribe_to_iskra_events(self):
        """Подписка на события ISKRA-4"""
        if not self.sephirot_bus:
            return
        
        # Подписываемся на системные события
        system_events = [
            "sephirot.energy_request",
            "sephirot.state_query",
            "sephirot.command.activate",
            "sephirot.command.deactivate",
            "sephirot.config_update",
            "sephirot.system_status",
        ]
        
        for event_type in system_events:
            await self.sephirot_bus.subscribe(
                self.__sephira_name__,
                self._handle_iskra_event
            )
        
        # Подписываемся на события других сефир
        other_sephirot_events = [
            "chokhmah.*",
            "binah.*",
            "chesed.*",
            "gevurah.*",
            "tiferet.*",
            "netzach.*",
            "hod.*",
            "yesod.*",
            "malkuth.*"
        ]
        
        for event_pattern in other_sephirot_events:
            await self.sephirot_bus.subscribe(
                self.__sephira_name__,
                self._handle_iskra_event
            )
        
        self.logger.info(f"Подписалось на события ISKRA-4")
    
    async def _handle_iskra_event(self, event: EventMessage):
        """Обработчик событий ISKRA-4"""
        await self.receive_event(event)
    
    # ========================================================
    # 5. ОБРАБОТКА СОБЫТИЙ KETER
    # ========================================================
    
    async def _register_keter_event_handlers(self):
        """Регистрация обработчиков событий KETER для преобразования в ISKRA-4"""
        if not self.keter_core:
            return
        
        # Регистрируем глобальный обработчик для всех событий KETER
        async def forward_keter_event(event_type: str, event_data: Dict):
            """Пересылка события KETER в ISKRA-4"""
            if not self.sephirot_bus:
                return
            
            # Преобразуем событие KETER в событие ISKRA-4
            iskra_event_type = self._map_to_iskra_event(event_type)
            
            if not iskra_event_type:
                # Если нет явного маппинга, создаём общее событие
                iskra_event_type = f"keter.{event_type}"
            
            # Публикуем событие в шину ISKRA-4
            await self.sephirot_bus.publish(EventMessage(
                event_type=iskra_event_type,
                data=event_data,
                source=self.__sephira_name__,
                target=event_data.get("target")
            ))
        
        # Подписываемся на основные события KETER
        for keter_event in self.event_mapping.keys():
            self.keter_core.subscribe(keter_event, forward_keter_event)
        
        # Также подписываемся на все события через wildcard
        self.keter_core.subscribe("*", forward_keter_event)
    
    def _map_to_iskra_event(self, keter_event_type: str) -> str:
        """Преобразование события KETER в событие ISKRA-4"""
        return self.event_mapping.get(keter_event_type, f"keter.{keter_event_type}")
    
    def _map_to_keter_event(self, iskra_event_type: str) -> str:
        """Преобразование события ISKRA-4 в событие KETER"""
        return self.reverse_event_mapping.get(iskra_event_type, f"iskra.{iskra_event_type}")
    
    # ========================================================
    # 6. УТИЛИТЫ И ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ========================================================
    
    def _convert_to_keter_config(self, sephira_config: SephiraConfig) -> Dict[str, Any]:
        """Преобразование конфигурации ISKRA-4 в конфигурацию KETER"""
        keter_config = {
            "activation": {
                "timeout": 30.0,
                "retry_attempts": 3,
                "retry_delay": 1.0
            },
            "energy": {
                "reserve": 1000.0,
                "recharge_rate": 10.0,
                "critical_threshold": 100.0
            },
            "events": {
                "enabled": True,
                "buffer_size": 1000,
                "processing_timeout": 5.0
            },
            "recovery": {
                "enabled": True,
                "auto_recover": True,
                "max_recovery_attempts": 3
            },
            "metrics": {
                "collection_interval": 5.0,
                "history_size": 1000,
                "export_enabled": True
            }
        }
        
        # Применяем настройки из ISKRA-4 если есть
        if hasattr(sephira_config, 'keter_settings'):
            keter_config.update(sephira_config.keter_settings)
        
        return keter_config
    
    async def get_detailed_state(self) -> Dict[str, Any]:
        """Получение детального состояния"""
        base_state = await self.get_state()
        
        # Добавляем информацию о событиях
        base_state["event_mappings"] = {
            "total_keter_events": len(self.event_mapping),
            "total_iskra_events": len(self.reverse_event_mapping),
            "keter_to_iskra": list(self.event_mapping.items())[:10],  # Первые 10
            "iskra_to_keter": list(self.reverse_event_mapping.items())[:10]
        }
        
        # Добавляем информацию о подключениях
        base_state["connections"] = {
            "sephirot_bus_connected": self.sephirot_bus is not None,
            "sephirotic_engine_connected": self.sephirotic_engine is not None,
            "event_handlers_registered": len(self.event_handlers),
            "iskra_subscriptions": len(self.reverse_event_mapping)
        }
        
        return base_state
    
    async def perform_health_check(self) -> Dict[str, Any]:
        """Выполнение проверки здоровья"""
        health_check = {
            "timestamp": time.time(),
            "sephira": self.__sephira_name__,
            "checks": [],
            "overall_status": "unknown"
        }
        
        # Проверка 1: Инициализация
        health_check["checks"].append({
            "name": "initialization",
            "status": "pass" if self.is_initialized else "fail",
            "message": "KeterIntegration инициализирован" if self.is_initialized else "KeterIntegration не инициализирован"
        })
        
        # Проверка 2: Активность
        health_check["checks"].append({
            "name": "activation",
            "status": "pass" if self.is_active else "warn",
            "message": "KeterIntegration активен" if self.is_active else "KeterIntegration не активен"
        })
        
        # Проверка 3: KeterCore
        if self.keter_core:
            try:
                keter_health = await self.keter_core.get_system_health_report()
                health_check["checks"].append({
                    "name": "keter_core",
                    "status": "pass" if keter_health.get("overall_health") == "healthy" else "warn",
                    "message": f"KeterCore: {keter_health.get('overall_health', 'unknown')}",
                    "details": {
                        "active_modules": keter_health["statistics"]["active_modules"],
                        "total_modules": keter_health["statistics"]["total_modules"],
                        "energy_reserve": keter_health["energy"]["reserve"]
                    }
                })
            except Exception as e:
                health_check["checks"].append({
                    "name": "keter_core",
                    "status": "fail",
                    "message": f"Ошибка проверки KeterCore: {str(e)}"
                })
        else:
            health_check["checks"].append({
                "name": "keter_core",
                "status": "fail",
                "message": "KeterCore не инициализирован"
            })
        
        # Проверка 4: Подключение к ISKRA-4
        health_check["checks"].append({
            "name": "iskra_connection",
            "status": "pass" if self.sephirot_bus else "warn",
            "message": "Подключено к ISKRA-4" if self.sephirot_bus else "Не подключено к ISKRA-4"
        })
        
        # Определение общего статуса
        failed_checks = [c for c in health_check["checks"] if c["status"] == "fail"]
        warning_checks = [c for c in health_check["checks"] if c["status"] == "warn"]
        
        if failed_checks:
            health_check["overall_status"] = "fail"
        elif warning_checks:
            health_check["overall_status"] = "warn"
        else:
            health_check["overall_status"] = "pass"
        
        return health_check
    
    # ========================================================
    # 7. API ДОСТУП К KETER ЧЕРЕЗ ИНТЕГРАЦИЮ
    # ========================================================
    
    def get_keter_core(self):
        """Получение доступа к ядру KETER"""
        return self.keter_core
    
    def get_keter_api(self):
        """Получение доступа к API KETER"""
        return self.keter_api
    
    async def call_keter_api(self, 
                            endpoint: str, 
                            method: str = "GET", 
                            data: Optional[Dict] = None) -> Dict[str, Any]:
        """Вызов API KETER через интеграцию"""
        if not self.keter_api:
            return {"error": "KeterAPI не инициализирован"}
        
        try:
            return await self.keter_api.api_call(endpoint, method, data)
        except Exception as e:
            return {"error": str(e), "endpoint": endpoint, "method": method}
    
    # ========================================================
    # 8. ФАБРИЧНЫЕ ФУНКЦИИ
    # ========================================================
    
    @classmethod
    def create(cls, config: Optional[SephiraConfig] = None) -> 'KeterIntegration':
        """Создание экземпляра интеграции"""
        return cls(config)
    
    @classmethod
    async def create_and_initialize(cls, 
                                  config: Optional[SephiraConfig] = None,
                                  sephirot_bus: Optional[SephirotBus] = None,
                                  sephirotic_engine: Optional[SephiroticEngine] = None) -> 'KeterIntegration':
        """Создание и инициализация интеграции"""
        instance = cls(config)
        
        # Инициализация
        if not await instance.initialize(config or SephiraConfig()):
            raise RuntimeError("Не удалось инициализировать KeterIntegration")
        
        # Подключение к ISKRA-4 если предоставлены компоненты
        if sephirot_bus and sephirotic_engine:
            if not await instance.connect_to_iskra(sephirot_bus, sephirotic_engine):
                raise RuntimeError("Не удалось подключиться к ISKRA-4")
        
        return instance

# ============================================================
# 9. ЭКСПОРТИРУЕМЫЕ ФУНКЦИИ
# ============================================================

def create_keter_integration(config=None):
    """Фабричная функция для создания интеграции"""
    return KeterIntegration(config)

async def initialize_keter_with_iskra(iskra_bus, iskra_engine, config=None):
    """Инициализация KETER с подключением к ISKRA-4"""
    integration = KeterIntegration(config)
    
    if not await integration.initialize(config or SephiraConfig()):
        return None
    
    if not await integration.connect_to_iskra(iskra_bus, iskra_engine):
        return None
    
    return integration

# ============================================================
# 10. ТЕСТОВАЯ ФУНКЦИЯ
# ============================================================

async def test_integration():
    """Тестирование интеграционного модуля"""
    print("🧪 Тестирование KeterIntegration...")
    
    try:
        # Создаём интеграцию
        integration = KeterIntegration()
        
        # Инициализируем
        success = await integration.initialize(SephiraConfig())
        
        if not success:
            print("❌ Не удалось инициализировать KeterIntegration")
            return False
        
        print("✅ KeterIntegration инициализирован")
        
        # Получаем состояние
        state = await integration.get_state()
        print(f"📊 Состояние: {state}")
        
        # Выполняем проверку здоровья
        health = await integration.perform_health_check()
        print(f"🩺 Проверка здоровья: {health['overall_status']}")
        
        print("🎯 KeterIntegration готов к интеграции с ISKRA-4!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        return False

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_integration())
