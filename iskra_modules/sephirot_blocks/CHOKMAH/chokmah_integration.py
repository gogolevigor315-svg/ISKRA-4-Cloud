"""
chokmah_integration.py - интеграция CHOKMAH с Sephirotic Engine
Реализует ISephiraModule интерфейс как keter_integration.py
"""

import logging
import time
from typing import Dict, Any, Optional

# Импорт базовых классов ISKRA-4
try:
    from sephirot_base import ISephiraModule, SephiraConfig, EnergyLevel
    from sephirot_bus import SephirotBus, EventMessage
    from sephirotic_engine import SephiroticEngine
    ISKRA_IMPORTS_AVAILABLE = True
except ImportError:
    ISKRA_IMPORTS_AVAILABLE = False
    # Заглушки для разработки
    class ISephiraModule: pass
    class SephiraConfig: pass
    class EnergyLevel: LOW="low"; MEDIUM="medium"; HIGH="high"; CRITICAL="critical"
    class SephirotBus: pass
    class EventMessage: pass
    class SephiroticEngine: pass

logger = logging.getLogger(__name__)


class ChokmahIntegration(ISephiraModule):
    """
    Интеграционный модуль сефиры CHOKMAH для системы ISKRA-4
    Реализует интерфейс ISephiraModule
    """
    
    __sephira_name__ = "CHOKMAH"
    __sephira_number__ = 2
    __version__ = "1.0.0"
    
    def __init__(self, config: Optional[SephiraConfig] = None):
        self.logger = logging.getLogger(f"ChokmahIntegration")
        
        # Конфигурация сефиры
        self.config = config or SephiraConfig()
        
        # Ядро CHOKMAH
        self.core = None
        
        # API CHOKMAH
        self.api = None
        
        # Связь с ISKRA-4
        self.sephirot_bus = None
        self.sephirotic_engine = None
        
        # Состояние
        self.is_initialized = False
        self.is_active = False
        self.energy_level = EnergyLevel.LOW
        
        # Маппинг событий (упрощённая версия KETER)
        self.event_mapping = {
            "chokmah.insight": "sephirot.insight_generated",
            "chokmah.activated": "sephirot.activated",
            "chokmah.deactivated": "sephirot.deactivated",
            "chokmah.target_achieved": "sephirot.target_achieved",
        }
        
        self.reverse_event_mapping = {
            "keter.request": "keter.request",
            "system.activate": "system.activate",
            "sephirot.energy_request": "energy.request",
        }
        
        self.logger.info(f"ChokmahIntegration v{self.__version__} создан")
    
    # ========================================================
    # ИНТЕРФЕЙС ISephiraModule (ОБЯЗАТЕЛЬНЫЕ МЕТОДЫ)
    # ========================================================
    
    async def initialize(self, config: SephiraConfig) -> bool:
        """
        Инициализация сефиры CHOKMAH в системе ISKRA-4
        """
        if self.is_initialized:
            self.logger.warning("ChokmahIntegration уже инициализирован")
            return True
        
        self.logger.info("🚀 Инициализация ChokmahIntegration...")
        
        try:
            # Сохраняем конфигурацию
            self.config = config
            
            # Импортируем и создаём ядро CHOKMAH
            from .wisdom_core import WisdomCore
            self.core = WisdomCore()
            
            # Импортируем и создаём API CHOKMAH
            from .chokmah_api import create_chokmah_api
            self.api = create_chokmah_api(self.core)
            
            self.is_initialized = True
            self.logger.info("✅ ChokmahIntegration успешно инициализирован")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка инициализации ChokmahIntegration: {e}")
            self.is_initialized = False
            return False
    
    async def activate(self) -> bool:
        """
        Активация сефиры CHOKMAH
        """
        if not self.is_initialized:
            self.logger.error("ChokmahIntegration не инициализирован")
            return False
        
        if self.is_active:
            self.logger.warning("CHOKMAH уже активен")
            return True
        
        self.logger.info("⚡ Активация CHOKMAH...")
        
        try:
            # Активируем ядро
            activation_result = await self.core.activate()
            
            if activation_result.get("status") == "error":
                self.logger.error(f"Ошибка активации ядра: {activation_result.get('error')}")
                return False
            
            # Обновляем состояние
            self.is_active = True
            self.energy_level = EnergyLevel.MEDIUM
            
            # Публикуем событие активации в ISKRA-4
            if self.sephirot_bus:
                await self.sephirot_bus.publish(EventMessage(
                    event_type="sephirot.activated",
                    data={
                        "sephira": self.__sephira_name__,
                        "version": self.__version__,
                        "resonance": self.core.resonance,
                        "energy": self.core.energy,
                        "timestamp": time.time()
                    },
                    source=self.__sephira_name__
                ))
            
            self.logger.info(f"✅ CHOKMAH активирован. Резонанс: {self.core.resonance}")
            
            # Проверяем достижение цели
            if self.core.resonance >= 0.6:
                self.logger.info("🎯 ЦЕЛЬ ДОСТИГНУТА: Резонанс CHOKMAH > 0.6!")
                
                if self.sephirot_bus:
                    await self.sephirot_bus.publish(EventMessage(
                        event_type="sephirot.target_achieved",
                        data={
                            "sephira": self.__sephira_name__,
                            "resonance": self.core.resonance,
                            "target": 0.6,
                            "timestamp": time.time()
                        },
                        source=self.__sephira_name__
                    ))
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка активации CHOKMAH: {e}")
            return False
    
    async def deactivate(self) -> bool:
        """
        Деактивация сефиры CHOKMAH
        """
        if not self.is_active:
            self.logger.warning("CHOKMAH уже неактивен")
            return True
        
        self.logger.info("🛑 Деактивация CHOKMAH...")
        
        try:
            # Публикуем событие деактивации в ISKRA-4
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
            
            # Обновляем состояние
            self.is_active = False
            self.energy_level = EnergyLevel.LOW
            
            self.logger.info("✅ CHOKMAH деактивирован")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка деактивации CHOKMAH: {e}")
            return False
    
    async def process_energy(self, energy_type: str, amount: float) -> bool:
        """
        Обработка энергии от системы ISKRA-4
        (CHOKMAH потребляет энергию для интуитивных процессов)
        """
        if not self.is_active:
            self.logger.warning("CHOKMAH не активен, не может обрабатывать энергию")
            return False
        
        self.logger.debug(f"⚡ CHOKMAH получил энергию: {energy_type} ({amount} units)")
        
        try:
            # Увеличиваем энергию ядра
            self.core.energy = min(1.0, self.core.energy + (amount / 1000.0))
            
            # Обновляем уровень энергии
            if self.core.energy > 0.8:
                self.energy_level = EnergyLevel.HIGH
            elif self.core.energy > 0.5:
                self.energy_level = EnergyLevel.MEDIUM
            elif self.core.energy > 0.3:
                self.energy_level = EnergyLevel.LOW
            else:
                self.energy_level = EnergyLevel.CRITICAL
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка обработки энергии: {e}")
            return False
    
    async def receive_event(self, event: EventMessage) -> bool:
        """
        Приём события от системы ISKRA-4
        """
        if not self.is_active:
            self.logger.warning(f"CHOKMAH не активен, игнорируем событие: {event.event_type}")
            return False
        
        self.logger.debug(f"📩 CHOKMAH получил событие: {event.event_type} от {event.source}")
        
        try:
            # Преобразуем событие ISKRA-4 во внутреннее
            internal_event_type = self._map_to_internal_event(event.event_type)
            
            if internal_event_type == "keter.request":
                # Обработка запроса от KETER
                await self._handle_keter_request(event.data)
                
            elif internal_event_type == "system.activate":
                # Активация системы
                await self.activate()
                
            elif internal_event_type == "energy.request":
                # Запрос энергии (CHOKMAH пока только потребляет)
                self.logger.debug(f"Запрос энергии: {event.data}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка обработки события: {e}")
            return False
    
    async def get_state(self) -> Dict[str, Any]:
        """
        Получение текущего состояния сефиры CHOKMAH
        """
        core_state = await self.core.get_status() if self.core else {}
        
        return {
            "sephira": self.__sephira_name__,
            "number": self.__sephira_number__,
            "version": self.__version__,
            "initialized": self.is_initialized,
            "active": self.is_active,
            "energy_level": self.energy_level,
            "integrated": self.sephirotic_engine is not None,
            "bus_connected": self.sephirot_bus is not None,
            "core_state": core_state,
            "timestamp": time.time()
        }
    
    # ========================================================
    # ИНТЕГРАЦИЯ С ISKRA-4 (как у KETER)
    # ========================================================
    
    async def connect_to_iskra(self, 
                              sephirot_bus: SephirotBus,
                              sephirotic_engine: SephiroticEngine) -> bool:
        """
        Подключение к архитектуре ISKRA-4
        """
        self.logger.info("🔗 Подключение CHOKMAH к архитектуре ISKRA-4...")
        
        try:
            # Сохраняем ссылки
            self.sephirot_bus = sephirot_bus
            self.sephirotic_engine = sephirotic_engine
            
            # Регистрируем сефиру в движке
            if hasattr(sephirotic_engine, 'register_sephira'):
                registration_result = await sephirotic_engine.register_sephira(
                    self.__sephira_name__,
                    self,
                    {
                        "version": self.__version__,
                        "position": self.__sephira_number__,
                        "capabilities": ["wisdom", "intuition", "insight_generation"],
                        "energy_requirements": {
                            "min": 50.0,
                            "optimal": 200.0,
                            "max": 500.0
                        }
                    }
                )
                
                if not registration_result.get("success", False):
                    self.logger.error("Не удалось зарегистрировать CHOKMAH в движке")
                    return False
            
            # Подписываемся на события ISKRA-4
            await self._subscribe_to_iskra_events()
            
            self.logger.info("✅ CHOKMAH успешно подключен к ISKRA-4")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка подключения к ISKRA-4: {e}")
            return False
    
    async def _subscribe_to_iskra_events(self):
        """Подписка на события ISKRA-4 (как у KETER)"""
        if not self.sephirot_bus:
            return
        
        # Подписываемся на системные события
        system_events = [
            "sephirot.energy_request",
            "sephirot.state_query",
            "sephirot.command.activate",
            "sephirot.command.deactivate",
            "keter.request",  # Прямые запросы от KETER
            "system.activate",
        ]
        
        for event_type in system_events:
            await self.sephirot_bus.subscribe(
                self.__sephira_name__,  # ⭐ ПРАВИЛЬНО: имя сефиры первое
                self._handle_iskra_event  # ⭐ ПРАВИЛЬНО: обработчик второе
            )
        
        self.logger.info(f"CHOKMAH подписался на {len(system_events)} событий ISKRA-4")
    
    async def _handle_iskra_event(self, event: EventMessage):
        """Обработчик событий ISKRA-4 (сигнатура как у KETER)"""
        await self.receive_event(event)
    
    # ========================================================
    # ОБРАБОТКА ЗАПРОСОВ
    # ========================================================
    
    async def _handle_keter_request(self, event_data: Dict[str, Any]):
        """Обработка запроса от KETER"""
        try:
            # Извлекаем данные запроса
            text = event_data.get("text") or event_data.get("message") or ""
            context = event_data.get("context", {})
            
            if not text:
                self.logger.warning("Пустой запрос от KETER")
                return
            
            # Обрабатываем через ядро CHOKMAH
            result = await self.core.process(text, context)
            
            # Преобразуем результат в событие ISKRA-4
            iskra_event_type = self._map_to_iskra_event("chokmah.insight")
            
            # Отправляем ответ в шину ISKRA-4
            if self.sephirot_bus:
                await self.sephirot_bus.publish(EventMessage(
                    event_type=iskra_event_type,
                    data={
                        "request_id": event_data.get("request_id"),
                        "sephira": self.__sephira_name__,
                        "insight": result.get("insight"),
                        "resonance": self.core.resonance,
                        "processing_time": result.get("processing_time", 0)
                    },
                    source=self.__sephira_name__,
                    target=event_data.get("source", "KETER")
                ))
                
                self.logger.info(f"📤 CHOKMAH отправил инсайт в ISKRA-4 (резонанс: {self.core.resonance})")
                
        except Exception as e:
            self.logger.error(f"❌ Ошибка обработки запроса от KETER: {e}")
    
    # ========================================================
    # УТИЛИТЫ И МАППИНГ
    # ========================================================
    
    def _map_to_iskra_event(self, chokmah_event_type: str) -> str:
        """Преобразование внутреннего события CHOKMAH в событие ISKRA-4"""
        return self.event_mapping.get(chokmah_event_type, f"chokmah.{chokmah_event_type}")
    
    def _map_to_internal_event(self, iskra_event_type: str) -> str:
        """Преобразование события ISKRA-4 во внутреннее событие CHOKMAH"""
        return self.reverse_event_mapping.get(iskra_event_type, f"iskra.{iskra_event_type}")
    
    # ========================================================
    # ДОПОЛНИТЕЛЬНЫЕ МЕТОДЫ
    # ========================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья CHOKMAH"""
        checks = []
        
        checks.append({
            "check": "initialization",
            "status": "pass" if self.is_initialized else "fail",
            "message": "Инициализирован" if self.is_initialized else "Не инициализирован"
        })
        
        checks.append({
            "check": "activation",
            "status": "pass" if self.is_active else "warn",
            "message": "Активен" if self.is_active else "Не активен"
        })
        
        checks.append({
            "check": "core",
            "status": "pass" if self.core else "fail",
            "message": "Ядро создано" if self.core else "Ядро не создано"
        })
        
        checks.append({
            "check": "integration",
            "status": "pass" if self.sephirot_bus else "warn",
            "message": "Интегрирован в ISKRA-4" if self.sephirot_bus else "Не интегрирован"
        })
        
        # Резонанс
        if self.core:
            resonance = self.core.resonance
            if resonance >= 0.6:
                resonance_status = "pass"
                resonance_msg = f"Цель достигнута: {resonance}"
            elif resonance > 0.4:
                resonance_status = "warn"
                resonance_msg = f"Растёт: {resonance}"
            else:
                resonance_status = "fail"
                resonance_msg = f"Низкий: {resonance}"
        else:
            resonance_status = "fail"
            resonance_msg = "Ядро недоступно"
        
        checks.append({
            "check": "resonance",
            "status": resonance_status,
            "message": resonance_msg
        })
        
        # Общий статус
        failed = [c for c in checks if c["status"] == "fail"]
        warnings = [c for c in checks if c["status"] == "warn"]
        
        if failed:
            overall = "fail"
        elif warnings:
            overall = "warn"
        else:
            overall = "pass"
        
        return {
            "timestamp": time.time(),
            "sephira": self.__sephira_name__,
            "overall_status": overall,
            "checks": checks
        }
    
    def get_core(self):
        """Получение ядра CHOKMAH"""
        return self.core
    
    def get_api(self):
        """Получение API CHOKMAH"""
        return self.api
    
    # ========================================================
    # ФАБРИЧНЫЕ ФУНКЦИИ (как у KETER)
    # ========================================================
    
    @classmethod
    def create(cls, config: Optional[SephiraConfig] = None) -> 'ChokmahIntegration':
        """Создание экземпляра интеграции"""
        return cls(config)
    
    @classmethod
    async def create_and_initialize(cls, 
                                  config: Optional[SephiraConfig] = None,
                                  sephirot_bus: Optional[SephirotBus] = None,
                                  sephirotic_engine: Optional[SephiroticEngine] = None) -> 'ChokmahIntegration':
        """Создание и инициализация интеграции"""
        instance = cls(config)
        
        if not await instance.initialize(config or SephiraConfig()):
            raise RuntimeError("Не удалось инициализировать ChokmahIntegration")
        
        if sephirot_bus and sephirotic_engine:
            if not await instance.connect_to_iskra(sephirot_bus, sephirotic_engine):
                raise RuntimeError("Не удалось подключить ChokmahIntegration к ISKRA-4")
        
        return instance


# Экспортируемые функции (как у KETER)
def create_chokmah_integration(config=None):
    """Фабричная функция для создания интеграции"""
    return ChokmahIntegration(config)

async def initialize_chokmah_with_iskra(iskra_bus, iskra_engine, config=None):
    """Инициализация CHOKMAH с подключением к ISKRA-4"""
    integration = ChokmahIntegration(config)
    
    if not await integration.initialize(config or SephiraConfig()):
        return None
    
    if not await integration.connect_to_iskra(iskra_bus, iskra_engine):
        return None
    
    return integration
