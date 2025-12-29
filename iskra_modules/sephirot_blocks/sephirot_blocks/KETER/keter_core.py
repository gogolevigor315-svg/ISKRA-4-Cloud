"""
KETHER CORE v2.0 - ПОЛНОЕ ИНТЕГРАЦИОННОЕ ЯДРО KETHERIC BLOCK
Сефира: KETER (Венец)
Модули: 5 (SPIRIT-SYNTHESIS, SPIRIT-CORE, WILLPOWER-CORE, CORE-GOVX, MORAL-MEMORY)
Архитектура: ISKRA-4 / Сефиротическая система
"""

import asyncio
import time
import sys
import os
import logging
from typing import Dict, Any, List, Optional, Protocol, TypedDict
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

# ============================================================
# 1. НАСТРОЙКА ПУТЕЙ И ИМПОРТОВ
# ============================================================

# Добавляем пути для импорта
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # iskra_modules
sys.path.insert(0, current_dir)  # sephirot_blocks/KETER

# ИМПОРТЫ 5 МОДУЛЕЙ KETHERIC BLOCK
try:
    # 1. SPIRIT-SYNTHESIS CORE v2.1
    from spirit_synthesis_core_v2_1 import create_spirit_synthesis_module
    print("✅ spirit_synthesis_core_v2_1 импортирован")
    
    # 2. SPIRIT-CORE v3.4
    from spirit_core_v3_4 import SpiritCoreV3_4
    print("✅ spirit_core_v3_4 импортирован")
    
    # 3. WILLPOWER-CORE v3.2
    from willpower_core_v3_2 import WillpowerCoreV3_2
    print("✅ willpower_core_v3_2 импортирован")
    
    # 4. CORE-GOVX 3.1
    from core_govx_3_1 import create_core_govx_module
    print("✅ core_govx_3_1 импортирован")
    
    # 5. MORAL-MEMORY 3.1
    from moral_memory_3_1 import create_moral_memory_module
    print("✅ moral_memory_3_1 импортирован")
    
    # Внешние зависимости
    from bechtereva import create_bechtereva_core
    from sephirotic_engine import SephiroticEngine
    from justice_guard_v2 import moral_compass
    from policy_governor_v1_2_impl import PolicyGovernorImpl
    from ds24_core import DS24Core
    print("✅ Внешние зависимости импортированы")
    
    MODULES_AVAILABLE = True
    
except ImportError as e:
    print(f"⚠️ Ошибка импорта: {e}")
    MODULES_AVAILABLE = False
    # Заглушки для разработки
    class MockModule:
        async def activate(self): return True
        async def work(self, data): return {}
        async def shutdown(self): pass
        async def get_metrics(self): return {"status": "mock"}
        async def receive_energy(self, amount, source): return True
        async def emit_event(self, event_type, data): pass
    
    def create_mock_module(): return MockModule()
    
    # Мокаем импорты
    create_spirit_synthesis_module = create_mock_module
    SpiritCoreV3_4 = MockModule
    WillpowerCoreV3_2 = MockModule
    create_core_govx_module = create_mock_module
    create_moral_memory_module = create_mock_module
    create_bechtereva_core = create_mock_module
    SephiroticEngine = MockModule

# ============================================================
# 2. ПРОТОКОЛЫ И СТРУКТУРЫ ДАННЫХ (ПОЛНЫЕ)
# ============================================================

class IKethericModule(Protocol):
    """Стандартизированный интерфейс модуля Ketheric Block"""
    async def activate(self) -> bool: ...
    async def work(self, data: Any) -> Any: ...
    async def shutdown(self) -> None: ...
    async def get_metrics(self) -> Dict[str, Any]: ...
    async def receive_energy(self, amount: float, source: str) -> bool: ...
    async def emit_event(self, event_type: str, data: Dict) -> None: ...

@dataclass
class ModuleInfo:
    """Информация о модуле"""
    name: str
    path: str
    dependencies: List[str]
    instance: Optional[IKethericModule] = None
    is_active: bool = False
    activation_order: int = 0
    config: Dict[str, Any] = None

@dataclass
class EnergyFlow:
    """Энергетический поток между модулями"""
    source: str
    target: str
    priority: str  # "critical", "high", "medium", "low"
    current_flow: float = 0.0
    max_flow: float = 100.0
    last_transfer: float = 0.0

class ModuleStatus(Enum):
    """Статус модуля"""
    INACTIVE = "inactive"
    ACTIVATING = "activating"
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"

# ============================================================
# 3. УТИЛИТА: ТОПОЛОГИЧЕСКАЯ СОРТИРОВКА
# ============================================================

def topological_sort(modules: Dict[str, List[str]]) -> List[str]:
    """
    Топологическая сортировка для определения порядка активации
    по зависимостям модулей
    """
    result = []
    visited = set()
    temp = set()
    
    def visit(node):
        if node in temp:
            raise ValueError(f"Циклическая зависимость обнаружена: {node}")
        if node not in visited:
            temp.add(node)
            for dep in modules.get(node, []):
                visit(dep)
            temp.remove(node)
            visited.add(node)
            result.append(node)
    
    for node in modules:
        if node not in visited:
            visit(node)
    
    return result

# ============================================================
# 4. ОСНОВНОЙ КЛАСС - KETHER CORE (ПОЛНЫЙ)
# ============================================================

class KetherCore:
    """
    ПОЛНОЕ интеграционное ядро Ketheric Block
    Управляет 5 модулями, энергетическими потоками, событиями и API
    """
    
    __sephira__ = "KETER"
    __version__ = "2.0.0"
    __architecture__ = "ISKRA-4/KETHERIC_BLOCK"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"KetherCore")
        
        # Конфигурация
        self.config = {
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
            },
            "api": {
                "enabled": True,
                "host": "localhost",
                "port": 8080,
                "auth_required": False
            }
        }
        
        if config:
            self._deep_update(self.config, config)
        
        # РЕЕСТР МОДУЛЕЙ
        self.modules: Dict[str, ModuleInfo] = {}
        
        # ЭНЕРГЕТИЧЕСКИЕ ПОТОКИ
        self.energy_flows: List[EnergyFlow] = []
        self.energy_reserve = self.config["energy"]["reserve"]
        
        # СИСТЕМА СОБЫТИЙ
        self.event_handlers: Dict[str, List[callable]] = {}
        self.event_queue = asyncio.Queue(maxsize=self.config["events"]["buffer_size"])
        
        # МЕТРИКИ И МОНИТОРИНГ
        self.metrics_history: List[Dict] = []
        self.activation_timestamps: Dict[str, float] = {}
        self.error_counters: Dict[str, int] = {}
        
        # ВНЕШНИЕ ЗАВИСИМОСТИ
        self.external_deps: Dict[str, Any] = {}
        
        # СТАТУС
        self.is_activated = False
        self.activation_start_time = 0.0
        self.shutdown_requested = False
        
        # ФОНОВЫЕ ЗАДАЧИ
        self.background_tasks: List[asyncio.Task] = []
        
        self.logger.info(f"KetherCore v{self.__version__} инициализирован")
    
    # ========================================================
    # 5. РЕГИСТРАЦИЯ МОДУЛЕЙ (5 ОСНОВНЫХ)
    # ========================================================
    
    async def register_all_modules(self) -> Dict[str, Any]:
        """
        Регистрация всех 5 модулей Ketheric Block с их зависимостями
        и внешними интеграциями
        """
        if not MODULES_AVAILABLE:
            self.logger.warning("Используются mock-модули (реальные недоступны)")
        
        results = {}
        
        # Зависимости между модулями (согласно матрице)
        dependencies_map = {
            "spirit_synthesis": [],                    # Базовый источник
            "spirit_core": ["spirit_synthesis"],       # Зависит от синтеза
            "willpower_core": ["spirit_synthesis"],    # Зависит от синтеза
            "moral_memory": ["willpower_core"],        # Зависит от воли
            "core_govx": ["spirit_core", "moral_memory"]  # Зависит от духа и морали
        }
        
        # 1. SPIRIT-SYNTHESIS CORE v2.1
        try:
            spirit_synth_config = {
                "integration_mode": "direct",
                "energy_source": "primary",
                "bechtereva_integration": True
            }
            spirit_synth = create_spirit_synthesis_module(config=spirit_synth_config)
            
            self.modules["spirit_synthesis"] = ModuleInfo(
                name="spirit_synthesis",
                path="spirit_synthesis_core_v2_1.py",
                dependencies=dependencies_map["spirit_synthesis"],
                instance=spirit_synth,
                config=spirit_synth_config
            )
            results["spirit_synthesis"] = "registered"
        except Exception as e:
            self.logger.error(f"Ошибка регистрации spirit_synthesis: {e}")
            results["spirit_synthesis"] = f"error: {e}"
        
        # 2. SPIRIT-CORE v3.4
        try:
            spirit_core_config = {
                "orchestration_mode": "dynamic",
                "priority_management": True,
                "resource_tracking": True
            }
            spirit_core = SpiritCoreV3_4(config=spirit_core_config)
            
            self.modules["spirit_core"] = ModuleInfo(
                name="spirit_core",
                path="spirit_core_v3_4.py",
                dependencies=dependencies_map["spirit_core"],
                instance=spirit_core,
                config=spirit_core_config
            )
            results["spirit_core"] = "registered"
        except Exception as e:
            self.logger.error(f"Ошибка регистрации spirit_core: {e}")
            results["spirit_core"] = f"error: {e}"
        
        # 3. WILLPOWER-CORE v3.2
        try:
            willpower_config = {
                "temporal_decay_enabled": True,
                "moral_filter_enabled": True,
                "autonomy_level": 0.8
            }
            willpower = WillpowerCoreV3_2(config=willpower_config)
            
            self.modules["willpower_core"] = ModuleInfo(
                name="willpower_core",
                path="willpower_core_v3_2.py",
                dependencies=dependencies_map["willpower_core"],
                instance=willpower,
                config=willpower_config
            )
            results["willpower_core"] = "registered"
        except Exception as e:
            self.logger.error(f"Ошибка регистрации willpower_core: {e}")
            results["willpower_core"] = f"error: {e}"
        
        # 4. MORAL-MEMORY 3.1
        try:
            moral_config = {
                "risk_threshold": 0.7,
                "fast_evaluation": True,
                "hard_ban_categories": ["CSAM", "терроризм", "физический_вред"],
                "operator_preferences": {"risk_tolerance": 0.5}
            }
            moral_memory = create_moral_memory_module(config=moral_config)
            
            self.modules["moral_memory"] = ModuleInfo(
                name="moral_memory",
                path="moral_memory_3_1.py",
                dependencies=dependencies_map["moral_memory"],
                instance=moral_memory,
                config=moral_config
            )
            results["moral_memory"] = "registered"
        except Exception as e:
            self.logger.error(f"Ошибка регистрации moral_memory: {e}")
            results["moral_memory"] = f"error: {e}"
        
        # 5. CORE-GOVX 3.1
        try:
            govx_config = {
                "homeostasis_monitoring": True,
                "policy_interpreter": True,
                "audit_ledger": True,
                "escalation_engine": True,
                "trend_analysis": True
            }
            core_govx = create_core_govx_module(config=govx_config)
            
            self.modules["core_govx"] = ModuleInfo(
                name="core_govx",
                path="core_govx_3_1.py",
                dependencies=dependencies_map["core_govx"],
                instance=core_govx,
                config=govx_config
            )
            results["core_govx"] = "registered"
        except Exception as e:
            self.logger.error(f"Ошибка регистрации core_govx: {e}")
            results["core_govx"] = f"error: {e}"
        
        # Регистрация внешних зависимостей
        await self._register_external_dependencies()
        
        self.logger.info(f"Зарегистрировано модулей: {sum(1 for r in results.values() if 'registered' in str(r))}/5")
        return results
    
    async def _register_external_dependencies(self):
        """Регистрация внешних зависимостей для модулей"""
        try:
            # Для SPIRIT-SYNTHESIS: bechtereva
            bechtereva_config = {
                "mode": "STANDARD",
                "deterministic": True,
                "update_interval": 0.1
            }
            # Здесь нужно передать реальные зависимости, но пока заглушка
            self.external_deps["bechtereva"] = None  # Будет создан позже
            
            # Для CORE-GOVX: policy_governor, ds24_core
            self.external_deps["policy_governor"] = None
            self.external_deps["ds24_core"] = None
            
            # Для MORAL-MEMORY: justice_guard
            self.external_deps["justice_guard"] = moral_compass
            
            self.logger.info("Внешние зависимости зарегистрированы")
        except Exception as e:
            self.logger.warning(f"Не удалось зарегистрировать внешние зависимости: {e}")
    
    # ========================================================
    # 6. КАСКАДНАЯ АКТИВАЦИЯ С ТОПОЛОГИЧЕСКОЙ СОРТИРОВКОЙ
    # ========================================================
    
    async def activate_cascade(self) -> Dict[str, Any]:
        """
        Полная каскадная активация с проверкой зависимостей
        и топологической сортировкой
        """
        self.logger.info("🚀 Запуск каскадной активации Ketheric Block...")
        
        self.is_activated = True
        self.activation_start_time = time.time()
        self.shutdown_requested = False
        
        # Определяем порядок активации через топологическую сортировку
        dependency_map = {
            name: module.dependencies
            for name, module in self.modules.items()
        }
        
        try:
            activation_order = topological_sort(dependency_map)
            self.logger.info(f"Порядок активации: {activation_order}")
        except ValueError as e:
            self.logger.error(f"Ошибка сортировки зависимостей: {e}")
            # Используем резервный порядок
            activation_order = [
                "spirit_synthesis",
                "spirit_core",
                "willpower_core",
                "moral_memory",
                "core_govx"
            ]
        
        activation_results = {}
        activated_count = 0
        
        # Активация каждого модуля в правильном порядке
        for module_name in activation_order:
            if module_name not in self.modules:
                self.logger.warning(f"Модуль {module_name} не найден в реестре")
                continue
            
            module_info = self.modules[module_name]
            
            # Проверяем зависимости
            missing_deps = [
                dep for dep in module_info.dependencies
                if dep not in self.modules or not self.modules[dep].is_active
            ]
            
            if missing_deps:
                self.logger.warning(
                    f"Модуль {module_name} ждёт зависимости: {missing_deps}"
                )
                # Ждём активации зависимостей (упрощённо)
                await asyncio.sleep(0.5)
            
            # Активация модуля
            try:
                self.logger.info(f"Активация модуля: {module_name}")
                start_time = time.time()
                
                # Активация с таймаутом
                try:
                    success = await asyncio.wait_for(
                        module_info.instance.activate(),
                        timeout=self.config["activation"]["timeout"]
                    )
                except asyncio.TimeoutError:
                    self.logger.error(f"Таймаут активации модуля {module_name}")
                    activation_results[module_name] = {
                        "status": "timeout",
                        "time": time.time() - start_time
                    }
                    continue
                
                if success:
                    module_info.is_active = True
                    module_info.activation_order = activated_count + 1
                    activation_time = time.time() - start_time
                    
                    self.activation_timestamps[module_name] = time.time()
                    self.error_counters[module_name] = 0
                    
                    activation_results[module_name] = {
                        "status": "active",
                        "order": module_info.activation_order,
                        "time": round(activation_time, 3)
                    }
                    
                    activated_count += 1
                    self.logger.info(f"✅ Модуль {module_name} активирован за {activation_time:.2f}с")
                    
                    # Эмитим событие активации
                    await self._publish_internal_event(
                        "module.activated",
                        {"module": module_name, "order": module_info.activation_order}
                    )
                    
                else:
                    activation_results[module_name] = {
                        "status": "failed",
                        "error": "activate() вернул False"
                    }
                    self.logger.error(f"❌ Модуль {module_name} не активировался (вернул False)")
                    
            except Exception as e:
                error_msg = str(e)
                activation_results[module_name] = {
                    "status": "error",
                    "error": error_msg
                }
                self.error_counters[module_name] = self.error_counters.get(module_name, 0) + 1
                self.logger.error(f"❌ Ошибка активации модуля {module_name}: {error_msg}")
        
        # Настройка энергетических потоков после активации
        await self._setup_energy_flows()
        
        # Запуск фоновых задач
        await self._start_background_tasks()
        
        total_time = time.time() - self.activation_start_time
        
        result = {
            "sephira": self.__sephira__,
            "version": self.__version__,
            "total_modules": len(self.modules),
            "activated_modules": activated_count,
            "activation_order": activation_order,
            "results": activation_results,
            "total_time": round(total_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(
            f"🎯 Каскадная активация завершена: "
            f"{activated_count}/{len(self.modules)} модулей активны "
            f"за {total_time:.2f}с"
        )
        
        return result
    
    async def _setup_energy_flows(self):
        """Настройка энергетических потоков согласно матрице"""
        self.energy_flows = [
            # ПРЯМЫЕ ПОТОКИ
            EnergyFlow(
                source="spirit_synthesis",
                target="willpower_core",
                priority="high",
                max_flow=85.0
            ),
            EnergyFlow(
                source="willpower_core",
                target="moral_memory",
                priority="medium",
                max_flow=60.0
            ),
            EnergyFlow(
                source="spirit_core",
                target="core_govx",
                priority="critical",
                max_flow=95.0
            ),
            EnergyFlow(
                source="moral_memory",
                target="core_govx",
                priority="high",
                max_flow=75.0
            ),
            # ОБРАТНЫЕ СВЯЗИ
            EnergyFlow(
                source="core_govx",
                target="spirit_core",
                priority="medium",
                max_flow=50.0
            ),
            EnergyFlow(
                source="core_govx",
                target="willpower_core",
                priority="medium",
                max_flow=45.0
            ),
        ]
        
        self.logger.info(f"Настроено энергетических потоков: {len(self.energy_flows)}")
    
    # ========================================================
    # 7. УПРАВЛЕНИЕ ЭНЕРГИЕЙ (ПОЛНОЕ)
    # ========================================================
    
    async def distribute_energy(self, 
                               source: str, 
                               target: str, 
                               amount: float) -> Dict[str, Any]:
        """
        Распределение энергии между модулями с проверками
        """
        # Проверяем существование модулей
        if source not in self.modules or target not in self.modules:
            return {
                "success": False,
                "reason": f"Модуль не найден: source={source}, target={target}"
            }
        
        # Проверяем активность модулей
        if not self.modules[source].is_active:
            return {"success": False, "reason": f"Источник {source} не активен"}
        
        if not self.modules[target].is_active:
            return {"success": False, "reason": f"Цель {target} не активна"}
        
        # Находим поток
        flow = next(
            (f for f in self.energy_flows 
             if f.source == source and f.target == target),
            None
        )
        
        if not flow:
            return {
                "success": False,
                "reason": f"Энергетический поток {source}→{target} не настроен"
            }
        
        # Проверяем лимиты
        if amount > flow.max_flow:
            amount = flow.max_flow
            self.logger.warning(f"Лимит потока {source}→{target}: {amount}")
        
        # Проверяем энергетический резерв
        if amount > self.energy_reserve:
            return {
                "success": False,
                "reason": f"Недостаточно энергии в резерве: {self.energy_reserve}"
            }
        
        # Выполняем передачу
        try:
            success = await self.modules[target].instance.receive_energy(amount, source)
            
            if success:
                # Обновляем метрики потока
                flow.current_flow = amount
                flow.last_transfer = time.time()
                
                # Списание из резерва
                self.energy_reserve -= amount
                
                # Публикуем событие
                await self._publish_internal_event(
                    "energy.distributed",
                    {
                        "source": source,
                        "target": target,
                        "amount": amount,
                        "flow": flow.priority,
                        "reserve": self.energy_reserve
                    }
                )
                
                return {
                    "success": True,
                    "amount": amount,
                    "flow": flow.priority,
                    "current_flow": flow.current_flow,
                    "remaining_reserve": self.energy_reserve,
                    "timestamp": time.time()
                }
            else:
                return {
                    "success": False,
                    "reason": f"Целевой модуль {target} отказался от энергии"
                }
                
        except Exception as e:
            self.logger.error(f"Ошибка распределения энергии {source}→{target}: {e}")
            return {"success": False, "reason": str(e)}
    
    async def recharge_energy(self, amount: float) -> bool:
        """Пополнение энергетического резерва"""
        if amount <= 0:
            return False
        
        old_reserve = self.energy_reserve
        self.energy_reserve += amount
        
        self.logger.info(f"Резерв пополнен: {old_reserve:.1f} → {self.energy_reserve:.1f}")
        
        await self._publish_internal_event(
            "energy.recharged",
            {
                "amount": amount,
                "old_reserve": old_reserve,
                "new_reserve": self.energy_reserve,
                "timestamp": time.time()
            }
        )
        
        return True
    
    # ========================================================
    # 8. СИСТЕМА СОБЫТИЙ (ПОЛНАЯ)
    # ========================================================
    
    def subscribe(self, event_type: str, handler: callable) -> str:
        """Подписка на события с возвратом ID подписки"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        subscription_id = f"{event_type}_{len(self.event_handlers[event_type])}_{int(time.time())}"
        self.event_handlers[event_type].append((subscription_id, handler))
        
        self.logger.debug(f"Подписка создана: {subscription_id} на {event_type}")
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Отмена подписки"""
        for event_type, handlers in self.event_handlers.items():
            for i, (sid, handler) in enumerate(handlers):
                if sid == subscription_id:
                    handlers.pop(i)
                    self.logger.debug(f"Подписка отменена: {subscription_id}")
                    return True
        return False
    
    async def _publish_internal_event(self, event_type: str, data: Dict) -> None:
        """Внутренняя публикация события"""
        if event_type in self.event_handlers:
            for subscription_id, handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    self.logger.error(f"Ошибка обработчика события {subscription_id}: {e}")
        
        # Также помещаем в очередь для обработки модулями
        try:
            await asyncio.wait_for(
                self.event_queue.put({"type": event_type, "data": data}),
                timeout=1.0
            )
        except (asyncio.QueueFull, asyncio.TimeoutError):
            self.logger.warning(f"Очередь событий переполнена, событие {event_type} пропущено")
    
    async def route_event(self, 
                         event_type: str, 
                         data: Dict, 
                         source_module: str) -> None:
        """
        Маршрутизация события между модулями согласно интеграционной матрице
        """
        routing_table = {
            # От MORAL-MEMORY к CORE-GOVX
            "moral.soft_warn": ["core_govx"],
            "moral.alert": ["core_govx"],
            "moral.escalation": ["core_govx", "spirit_core"],
            
            # От CORE-GOVX к другим
            "policy.escalate": ["spirit_core", "willpower_core"],
            "governance.homeostasis.update": ["spirit_core", "willpower_core", "moral_memory"],
            "audit.anomaly": ["spirit_core"],
            
            # От SPIRIT-SYNTHESIS
            "spiritual.synthesis": ["willpower_core", "spirit_core"],
            "energy.surge": ["willpower_core", "spirit_core"],
            
            # От WILLPOWER-CORE
            "willpower.boost": ["moral_memory", "spirit_core"],
            "autonomy.change": ["core_govx", "spirit_core"],
            
            # Системные события
            "module.failed": ["core_govx", "spirit_core"],
            "energy.critical": ["spirit_synthesis", "core_govx", "spirit_core"],
            "system.recovery": ["core_govx", "spirit_core"]
        }
        
        targets = routing_table.get(event_type, [])
        
        for target in targets:
            if target in self.modules and target != source_module:
                if self.modules[target].is_active:
                    try:
                        await self.modules[target].instance.emit_event(event_type, data)
                        self.logger.debug(f"Событие {event_type} → {target}")
                    except Exception as e:
                        self.logger.error(f"Ошибка маршрутизации {event_type} → {target}: {e}")
    
    async def _event_processor_task(self):
        """Фоновая задача обработки событий"""
        self.logger.info("Запущен обработчик событий")
        
        while not self.shutdown_requested:
            try:
                # Получаем событие из очереди с таймаутом
                try:
                    event = await asyncio.wait_for(
                        self.event_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                event_type = event["type"]
                data = event["data"]
                
                # Обрабатываем системные события
                if event_type == "system.shutdown":
                    self.logger.info("Получен запрос на выключение")
                    self.shutdown_requested = True
                    break
                
                # Маршрутизируем между модулями
                source = data.get("source", "unknown")
                await self.route_event(event_type, data, source)
                
                # Помечаем как обработанное
                self.event_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Ошибка обработчика событий: {e}")
                await asyncio.sleep(0.1)
        
        self.logger.info("Обработчик событий остановлен")
    
        # ========================================================
    # 9. СБОР МЕТРИК И МОНИТОРИНГ (ПОЛНЫЙ)
    # ========================================================
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """
        Полный сбор метрик со всех модулей и системы
        """
        metrics = {
            "sephira": self.__sephira__,
            "version": self.__version__,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "modules": {},
            "energy": {
                "reserve": self.energy_reserve,
                "critical": self.energy_reserve < self.config["energy"]["critical_threshold"],
                "flows_active": len([f for f in self.energy_flows if f.current_flow > 0]),
                "total_flows": len(self.energy_flows)
            },
            "system": {
                "activated": self.is_activated,
                "uptime": time.time() - self.activation_start_time if self.is_activated else 0,
                "active_modules": sum(1 for m in self.modules.values() if m.is_active),
                "total_modules": len(self.modules),
                "event_queue_size": self.event_queue.qsize(),
                "background_tasks": len(self.background_tasks)
            },
            "performance": {
                "activation_order": [
                    {"name": name, "order": module.activation_order}
                    for name, module in self.modules.items()
                    if module.is_active
                ],
                "errors": self.error_counters.copy()
            }
        }
        
        # Собираем метрики каждого модуля
        for name, module_info in self.modules.items():
            if module_info.instance and module_info.is_active:
                try:
                    module_metrics = await module_info.instance.get_metrics()
                    metrics["modules"][name] = {
                        "active": True,
                        "order": module_info.activation_order,
                        "metrics": module_metrics
                    }
                except Exception as e:
                    metrics["modules"][name] = {
                        "active": True,
                        "error": str(e)
                    }
            else:
                metrics["modules"][name] = {
                    "active": False,
                    "order": module_info.activation_order
                }
        
        # Метрики энергетических потоков
        metrics["energy"]["flows"] = [
            {
                "source": flow.source,
                "target": flow.target,
                "priority": flow.priority,
                "current": flow.current_flow,
                "max": flow.max_flow,
                "last_transfer": flow.last_transfer
            }
            for flow in self.energy_flows
        ]
        
        # Сохраняем в историю
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.config["metrics"]["history_size"]:
            self.metrics_history = self.metrics_history[-self.config["metrics"]["history_size"]:]
        
        return metrics
    
    async def _check_critical_states(self):
        """Проверка критических состояний системы"""
        warnings = []
        
        # 1. Проверка энергетического резерва
        if self.energy_reserve < self.config["energy"]["critical_threshold"]:
            warnings.append({
                "type": "energy_critical",
                "message": f"Энергетический резерв критически низок: {self.energy_reserve:.1f}",
                "severity": "critical"
            })
        
        # 2. Проверка упавших модулей
        failed_modules = [
            name for name, module in self.modules.items()
            if not module.is_active and name in self.activation_timestamps
        ]
        if failed_modules:
            warnings.append({
                "type": "modules_failed",
                "message": f"Неактивные модули: {failed_modules}",
                "severity": "high",
                "modules": failed_modules
            })
        
        # 3. Проверка ошибок
        high_error_modules = [
            name for name, count in self.error_counters.items()
            if count > 5
        ]
        if high_error_modules:
            warnings.append({
                "type": "high_error_rate",
                "message": f"Высокий счётчик ошибок у модулей: {high_error_modules}",
                "severity": "medium",
                "modules": high_error_modules
            })
        
        # 4. Проверка переполнения очереди событий
        queue_size = self.event_queue.qsize()
        queue_capacity = self.event_queue.maxsize
        if queue_size > queue_capacity * 0.8:
            warnings.append({
                "type": "event_queue_high",
                "message": f"Очередь событий заполнена на {queue_size}/{queue_capacity}",
                "severity": "medium"
            })
        
        # Если есть критические предупреждения — публикуем событие
        if warnings:
            critical_warnings = [w for w in warnings if w["severity"] in ["critical", "high"]]
            if critical_warnings:
                await self._publish_internal_event("system.critical_warning", {
                    "warnings": critical_warnings,
                    "timestamp": time.time()
                })
            
            self.logger.warning(f"Критические состояния: {len(warnings)} предупреждений")
        
        return warnings
    
    async def get_metrics_history(self, limit: int = 100) -> List[Dict]:
        """Получение истории метрик"""
        return self.metrics_history[-limit:] if self.metrics_history else []
    
    async def get_module_health(self, module_name: str) -> Dict[str, Any]:
        """Получение детального здоровья модуля"""
        if module_name not in self.modules:
            return {"error": "module_not_found"}
        
        module_info = self.modules[module_name]
        health = {
            "name": module_name,
            "active": module_info.is_active,
            "activation_order": module_info.activation_order,
            "activation_time": self.activation_timestamps.get(module_name),
            "error_count": self.error_counters.get(module_name, 0),
            "uptime": time.time() - self.activation_timestamps.get(module_name, 0) 
                      if module_info.is_active else 0,
            "dependencies": module_info.dependencies,
            "dependencies_met": all(
                dep in self.modules and self.modules[dep].is_active
                for dep in module_info.dependencies
            )
        }
        
        # Добавляем метрики модуля, если доступны
        if module_info.is_active and module_info.instance:
            try:
                module_metrics = await module_info.instance.get_metrics()
                health["metrics"] = module_metrics
            except Exception as e:
                health["metrics_error"] = str(e)
        
        return health
    
    async def get_system_health_report(self) -> Dict[str, Any]:
        """Полный отчёт о здоровье системы"""
        report = {
            "timestamp": time.time(),
            "sephira": self.__sephira__,
            "version": self.__version__,
            "overall_health": "unknown",
            "modules": {},
            "energy": {
                "reserve": self.energy_reserve,
                "status": "normal",
                "critical": self.energy_reserve < self.config["energy"]["critical_threshold"]
            },
            "warnings": await self._check_critical_states(),
            "statistics": {
                "total_modules": len(self.modules),
                "active_modules": sum(1 for m in self.modules.values() if m.is_active),
                "inactive_modules": sum(1 for m in self.modules.values() if not m.is_active),
                "total_errors": sum(self.error_counters.values()),
                "uptime": time.time() - self.activation_start_time if self.is_activated else 0
            }
        }
        
        # Собираем здоровье каждого модуля
        for name in self.modules:
            report["modules"][name] = await self.get_module_health(name)
        
        # Определяем общее здоровье системы
        active_ratio = report["statistics"]["active_modules"] / report["statistics"]["total_modules"]
        if active_ratio >= 0.8 and not report["warnings"]:
            report["overall_health"] = "healthy"
        elif active_ratio >= 0.6:
            report["overall_health"] = "degraded"
        else:
            report["overall_health"] = "critical"
        
        # Определяем статус энергии
        if self.energy_reserve < self.config["energy"]["critical_threshold"] * 0.5:
            report["energy"]["status"] = "critical"
        elif self.energy_reserve < self.config["energy"]["critical_threshold"]:
            report["energy"]["status"] = "warning"
        else:
            report["energy"]["status"] = "normal"
        
        return report
    
    async def _metrics_collector_task(self):
        """Фоновая задача сбора метрик"""
        self.logger.info("Запущен сборщик метрик")
        
        while not self.shutdown_requested:
            try:
                await self.collect_metrics()
                
                # Проверяем критические состояния
                await self._check_critical_states()
                
                # Ждём следующий цикл сбора
                await asyncio.sleep(self.config["metrics"]["collection_interval"])
                
            except Exception as e:
                self.logger.error(f"Ошибка сборщика метрик: {e}")
                await asyncio.sleep(1.0)
        
        self.logger.info("Сборщик метрик остановлен")
    
        # ========================================================
    # 10. СИСТЕМА ВОССТАНОВЛЕНИЯ (ПОЛНАЯ)
    # ========================================================
    
    async def recover_module(self, module_name: str, force: bool = False) -> Dict[str, Any]:
        """
        Полное восстановление упавшего модуля с пересозданием экземпляра
        """
        if module_name not in self.modules:
            return {
                "success": False,
                "reason": "module_not_found",
                "module": module_name,
                "timestamp": time.time()
            }
        
        module_info = self.modules[module_name]
        recovery_id = f"recovery_{module_name}_{int(time.time())}"
        recovery_log = []
        recovery_start = time.time()
        
        self.logger.info(f"🔄 Начало восстановления модуля {module_name} (ID: {recovery_id})")
        
        # Логируем начальное состояние
        recovery_log.append({
            "time": 0.0,
            "stage": "start",
            "status": "beginning_recovery",
            "module_active": module_info.is_active,
            "force_mode": force
        })
        
        # Если модуль уже активен и не форсируем - возвращаем успех
        if module_info.is_active and not force:
            self.logger.info(f"Модуль {module_name} уже активен, восстановление не требуется")
            return {
                "success": True,
                "recovery_id": recovery_id,
                "status": "already_active",
                "module": module_name,
                "recovery_time": 0.0,
                "log": recovery_log
            }
        
        # Проверка зависимостей (если не форсируем)
        if not force:
            missing_deps = []
            for dep in module_info.dependencies:
                if dep not in self.modules:
                    missing_deps.append(f"{dep}(not_registered)")
                elif not self.modules[dep].is_active:
                    missing_deps.append(f"{dep}(inactive)")
            
            if missing_deps:
                recovery_log.append({
                    "time": time.time() - recovery_start,
                    "stage": "dependency_check",
                    "status": "failed",
                    "missing_dependencies": missing_deps
                })
                
                self.logger.warning(f"Восстановление {module_name} остановлено: отсутствуют зависимости {missing_deps}")
                
                return {
                    "success": False,
                    "recovery_id": recovery_id,
                    "reason": "missing_dependencies",
                    "dependencies": missing_deps,
                    "module": module_name,
                    "recovery_time": time.time() - recovery_start,
                    "log": recovery_log
                }
        
        recovery_log.append({
            "time": time.time() - recovery_start,
            "stage": "dependency_check",
            "status": "passed",
            "dependencies": module_info.dependencies
        })
        
        # ШАГ 1: Деактивация текущего экземпляра (если есть)
        if module_info.instance:
            try:
                recovery_log.append({
                    "time": time.time() - recovery_start,
                    "stage": "shutdown",
                    "status": "starting"
                })
                
                shutdown_start = time.time()
                await module_info.instance.shutdown()
                shutdown_time = time.time() - shutdown_start
                
                recovery_log.append({
                    "time": time.time() - recovery_start,
                    "stage": "shutdown",
                    "status": "completed",
                    "duration": shutdown_time
                })
                
                self.logger.debug(f"Модуль {module_name} деактивирован за {shutdown_time:.2f}с")
                
            except Exception as e:
                error_msg = str(e)
                recovery_log.append({
                    "time": time.time() - recovery_start,
                    "stage": "shutdown",
                    "status": "error",
                    "error": error_msg
                })
                
                if not force:
                    self.logger.error(f"Ошибка деактивации {module_name}: {error_msg}")
                    return {
                        "success": False,
                        "recovery_id": recovery_id,
                        "reason": "shutdown_error",
                        "error": error_msg,
                        "module": module_name,
                        "recovery_time": time.time() - recovery_start,
                        "log": recovery_log
                    }
                else:
                    self.logger.warning(f"Ошибка деактивации {module_name} в force режиме, продолжаем: {error_msg}")
        
        # ШАГ 2: Пересоздание экземпляра модуля
        try:
            recovery_log.append({
                "time": time.time() - recovery_start,
                "stage": "recreate",
                "status": "starting"
            })
            
            recreate_start = time.time()
            
            # Определяем фабричную функцию для каждого модуля
            factory_map = {
                "spirit_synthesis": lambda: create_spirit_synthesis_module(config=module_info.config),
                "moral_memory": lambda: create_moral_memory_module(config=module_info.config),
                "core_govx": lambda: create_core_govx_module(config=module_info.config),
            }
            
            class_map = {
                "spirit_core": lambda: SpiritCoreV3_4(config=module_info.config),
                "willpower_core": lambda: WillpowerCoreV3_2(config=module_info.config),
            }
            
            if module_name in factory_map:
                new_instance = factory_map[module_name]()
                creation_method = "factory_function"
            elif module_name in class_map:
                new_instance = class_map[module_name]()
                creation_method = "direct_instantiation"
            else:
                raise ValueError(f"Неизвестный тип модуля для создания: {module_name}")
            
            module_info.instance = new_instance
            recreate_time = time.time() - recreate_start
            
            recovery_log.append({
                "time": time.time() - recovery_start,
                "stage": "recreate",
                "status": "completed",
                "duration": recreate_time,
                "method": creation_method
            })
            
            self.logger.debug(f"Экземпляр {module_name} пересоздан за {recreate_time:.2f}с методом {creation_method}")
            
        except Exception as e:
            error_msg = str(e)
            recovery_log.append({
                "time": time.time() - recovery_start,
                "stage": "recreate",
                "status": "error",
                "error": error_msg
            })
            
            self.logger.error(f"Ошибка пересоздания {module_name}: {error_msg}")
            
            # Помечаем модуль как неактивный
            module_info.is_active = False
            module_info.instance = None
            
            return {
                "success": False,
                "recovery_id": recovery_id,
                "reason": "recreate_error",
                "error": error_msg,
                "module": module_name,
                "recovery_time": time.time() - recovery_start,
                "log": recovery_log
            }
        
        # ШАГ 3: Активация нового экземпляра
        try:
            recovery_log.append({
                "time": time.time() - recovery_start,
                "stage": "activation",
                "status": "starting"
            })
            
            activation_start = time.time()
            
            # Активация с таймаутом
            try:
                success = await asyncio.wait_for(
                    module_info.instance.activate(),
                    timeout=self.config["activation"]["timeout"]
                )
            except asyncio.TimeoutError:
                activation_time = time.time() - activation_start
                
                recovery_log.append({
                    "time": time.time() - recovery_start,
                    "stage": "activation",
                    "status": "timeout",
                    "duration": activation_time,
                    "timeout": self.config["activation"]["timeout"]
                })
                
                self.logger.error(f"Таймаут активации {module_name} ({self.config['activation']['timeout']}с)")
                
                module_info.is_active = False
                
                return {
                    "success": False,
                    "recovery_id": recovery_id,
                    "reason": "activation_timeout",
                    "module": module_name,
                    "recovery_time": time.time() - recovery_start,
                    "log": recovery_log
                }
            
            activation_time = time.time() - activation_start
            
            if success:
                module_info.is_active = True
                module_info.activation_order = max(
                    [m.activation_order for m in self.modules.values() if m.is_active],
                    default=0
                ) + 1
                
                # Обновляем метрики
                self.activation_timestamps[module_name] = time.time()
                if f"{module_name}_recovery" in self.error_counters:
                    del self.error_counters[f"{module_name}_recovery"]
                
                recovery_log.append({
                    "time": time.time() - recovery_start,
                    "stage": "activation",
                    "status": "completed",
                    "duration": activation_time,
                    "success": True,
                    "new_order": module_info.activation_order
                })
                
                total_recovery_time = time.time() - recovery_start
                
                self.logger.info(f"✅ Модуль {module_name} успешно восстановлен за {total_recovery_time:.2f}с")
                
                # Публикуем событие успешного восстановления
                await self._publish_internal_event("module.recovered", {
                    "module": module_name,
                    "recovery_id": recovery_id,
                    "recovery_time": total_recovery_time,
                    "new_activation_order": module_info.activation_order,
                    "timestamp": time.time(),
                    "log_summary": [log.get("stage") for log in recovery_log]
                })
                
                return {
                    "success": True,
                    "recovery_id": recovery_id,
                    "module": module_name,
                    "recovery_time": total_recovery_time,
                    "new_activation_order": module_info.activation_order,
                    "stages": {
                        "shutdown": next((log for log in recovery_log if log.get("stage") == "shutdown" and log.get("status") == "completed"), None),
                        "recreate": next((log for log in recovery_log if log.get("stage") == "recreate" and log.get("status") == "completed"), None),
                        "activation": next((log for log in recovery_log if log.get("stage") == "activation" and log.get("status") == "completed"), None)
                    },
                    "log": recovery_log
                }
            else:
                recovery_log.append({
                    "time": time.time() - recovery_start,
                    "stage": "activation",
                    "status": "failed",
                    "duration": activation_time,
                    "success": False
                })
                
                module_info.is_active = False
                
                self.logger.error(f"Активация {module_name} вернула False")
                
                return {
                    "success": False,
                    "recovery_id": recovery_id,
                    "reason": "activation_failed",
                    "module": module_name,
                    "recovery_time": time.time() - recovery_start,
                    "log": recovery_log
                }
                
        except Exception as e:
            error_msg = str(e)
            activation_time = time.time() - activation_start
            
            recovery_log.append({
                "time": time.time() - recovery_start,
                "stage": "activation",
                "status": "error",
                "duration": activation_time,
                "error": error_msg
            })
            
            module_info.is_active = False
            
            # Увеличиваем счётчик попыток восстановления
            recovery_key = f"{module_name}_recovery"
            self.error_counters[recovery_key] = self.error_counters.get(recovery_key, 0) + 1
            
            self.logger.error(f"Ошибка активации {module_name}: {error_msg}")
            
            return {
                "success": False,
                "recovery_id": recovery_id,
                "reason": "activation_error",
                "error": error_msg,
                "module": module_name,
                "recovery_attempts": self.error_counters.get(recovery_key, 0),
                "recovery_time": time.time() - recovery_start,
                "log": recovery_log
            }
    
    async def auto_recover_failed_modules(self) -> Dict[str, Any]:
        """
        Автоматическое восстановление всех упавших модулей с интеллектуальной логикой
        """
        if not self.config["recovery"]["enabled"]:
            return {
                "enabled": False,
                "reason": "recovery_disabled",
                "timestamp": time.time()
            }
        
        if not self.config["recovery"]["auto_recover"]:
            return {
                "enabled": False,
                "reason": "auto_recovery_disabled",
                "timestamp": time.time()
            }
        
        # Находим упавшие модули
        failed_modules = []
        for name, module in self.modules.items():
            if not module.is_active:
                # Проверяем, был ли модуль когда-либо активирован
                was_ever_active = name in self.activation_timestamps
                
                # Проверяем блокировку восстановления
                recovery_attempts = self.error_counters.get(f"{name}_recovery", 0)
                recovery_blocked = recovery_attempts >= self.config["recovery"]["max_recovery_attempts"]
                
                failed_modules.append({
                    "name": name,
                    "was_ever_active": was_ever_active,
                    "recovery_attempts": recovery_attempts,
                    "recovery_blocked": recovery_blocked,
                    "dependencies": module.dependencies,
                    "is_critical": name in ["spirit_synthesis", "spirit_core", "core_govx"]
                })
        
        if not failed_modules:
            return {
                "enabled": True,
                "status": "all_modules_active",
                "timestamp": time.time(),
                "checked_modules": len(self.modules)
            }
        
        self.logger.info(f"🔍 Обнаружено {len(failed_modules)} неактивных модулей")
        
        # Сортируем модули по приоритету восстановления
        # 1. Критические модули
        # 2. Модули с наименьшим количеством попыток восстановления
        # 3. Модули, которые были активны ранее
        def recovery_priority(module):
            priority = 0
            if module["is_critical"]:
                priority += 100
            priority += (self.config["recovery"]["max_recovery_attempts"] - module["recovery_attempts"]) * 10
            if module["was_ever_active"]:
                priority += 5
            return priority
        
        failed_modules.sort(key=recovery_priority, reverse=True)
        
        recovery_results = {}
        recovered_count = 0
        skipped_count = 0
        failed_count = 0
        
        # Восстанавливаем модули в порядке приоритета
        for module_info in failed_modules:
            module_name = module_info["name"]
            
            # Проверяем блокировку
            if module_info["recovery_blocked"]:
                recovery_results[module_name] = {
                    "status": "skipped",
                    "reason": "recovery_blocked",
                    "attempts": module_info["recovery_attempts"],
                    "max_attempts": self.config["recovery"]["max_recovery_attempts"]
                }
                skipped_count += 1
                continue
            
            # Проверяем зависимости
            missing_deps = []
            for dep in module_info["dependencies"]:
                if dep not in self.modules:
                    missing_deps.append(f"{dep}(not_registered)")
                elif not self.modules[dep].is_active:
                    missing_deps.append(f"{dep}(inactive)")
            
            if missing_deps:
                # Для критических модулей пробуем force recovery
                if module_info["is_critical"]:
                    self.logger.warning(f"Критический модуль {module_name} имеет отсутствующие зависимости: {missing_deps}. Пробуем force recovery.")
                    result = await self.recover_module(module_name, force=True)
                else:
                    recovery_results[module_name] = {
                        "status": "skipped",
                        "reason": "missing_dependencies",
                        "missing_deps": missing_deps
                    }
                    skipped_count += 1
                    continue
            else:
                # Обычное восстановление
                result = await self.recover_module(module_name, force=False)
            
            recovery_results[module_name] = result
            
            if result.get("success"):
                recovered_count += 1
            else:
                failed_count += 1
        
        # Формируем отчёт
        report = {
            "enabled": True,
            "timestamp": time.time(),
            "total_checked": len(self.modules),
            "total_failed": len(failed_modules),
            "recovered": recovered_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "critical_recovered": sum(1 for m in failed_modules if m["is_critical"] and recovery_results.get(m["name"], {}).get("success")),
            "results": recovery_results,
            "summary": {
                "health_percentage": (len(self.modules) - len(failed_modules) + recovered_count) / len(self.modules) * 100,
                "effectiveness": recovered_count / max(1, len(failed_modules) - skipped_count) * 100
            }
        }
        
        # Логируем итоги
        if recovered_count > 0:
            self.logger.info(f"✅ Автовосстановление завершено: {recovered_count} модулей восстановлено")
        if skipped_count > 0:
            self.logger.warning(f"⚠️ Автовосстановление: {skipped_count} модулей пропущено")
        if failed_count > 0:
            self.logger.error(f"❌ Автовосстановление: {failed_count} модулей не восстановлено")
        
        # Публикуем событие
        await self._publish_internal_event("recovery.auto_completed", report)
        
        return report
    
    async def _recovery_monitor_task(self):
        """Фоновая задача мониторинга и восстановления"""
        if not self.config["recovery"]["enabled"]:
            self.logger.info("Мониторинг восстановления отключен в конфигурации")
            return
        
        self.logger.info("🔧 Запуск монитора восстановления...")
        
        check_interval = 10.0  # Проверка каждые 10 секунд
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while not self.shutdown_requested:
            try:
                await asyncio.sleep(check_interval)
                
                # Собираем текущие метрики
                current_metrics = await self.collect_metrics()
                active_modules = current_metrics["system"]["active_modules"]
                total_modules = current_metrics["system"]["total_modules"]
                
                # Вычисляем здоровье системы
                health_ratio = active_modules / total_modules
                
                # Определяем пороги
                warning_threshold = 0.9   # 90% модулей активны
                critical_threshold = 0.7   # 70% модулей активны
                
                if health_ratio >= warning_threshold:
                    # Система здорова
                    consecutive_failures = 0
                    continue
                
                # Система в предупреждающем или критическом состоянии
                state = "warning" if health_ratio >= critical_threshold else "critical"
                inactive_count = total_modules - active_modules
                
                self.logger.warning(
                    f"Состояние системы: {state.upper()}. "
                    f"Активных модулей: {active_modules}/{total_modules} ({health_ratio:.1%}). "
                    f"Неактивных: {inactive_count}"
                )
                
                # Увеличиваем счётчик последовательных сбоев
                consecutive_failures += 1
                
                # Запускаем восстановление если:
                # 1. Система в критическом состоянии ИЛИ
                # 2. Много последовательных проверок показывают проблемы
                if state == "critical" or consecutive_failures >= max_consecutive_failures:
                    self.logger.info(f"🚨 Запуск автовосстановления (причина: {state}, failures: {consecutive_failures})")
                    
                    recovery_report = await self.auto_recover_failed_modules()
                    
                    if recovery_report.get("recovered", 0) > 0:
                        # Успешное восстановление - сбрасываем счётчик
                        consecutive_failures = 0
                        self.logger.info(f"Автовосстановление успешно: {recovery_report['recovered']} модулей восстановлено")
                    else:
                        # Неудачное восстановление
                        self.logger.error("Автовосстановление не смогло восстановить модули")
                        
                        # Если система критична, пробуем экстренные меры
                        if state == "critical":
                            await self._emergency_recovery_protocol()
                
                # Проверяем критические модули вручную
                await self._check_critical_modules()
                
            except Exception as e:
                self.logger.error(f"Ошибка монитора восстановления: {e}")
                consecutive_failures = min(consecutive_failures + 1, max_consecutive_failures)
                await asyncio.sleep(5.0)
        
        self.logger.info("Монитор восстановления остановлен")
    
    async def _check_critical_modules(self):
        """Проверка состояния критических модулей"""
        critical_modules = ["spirit_synthesis", "spirit_core", "core_govx"]
        
        for module_name in critical_modules:
            if module_name not in self.modules:
                continue
            
            module_info = self.modules[module_name]
            
            if not module_info.is_active:
                self.logger.critical(f"КРИТИЧЕСКИЙ МОДУЛЬ {module_name} НЕ АКТИВЕН!")
                
                # Немедленная попытка восстановления с повышенным приоритетом
                recovery_result = await self.recover_module(module_name, force=True)
                
                if not recovery_result.get("success"):
                    self.logger.critical(f"НЕУДАЧНОЕ ВОССТАНОВЛЕНИЕ КРИТИЧЕСКОГО МОДУЛЯ {module_name}!")
                    
                    # Запускаем цепочку зависимостей
                    await self._recover_dependency_chain(module_name)
    
    async def _recover_dependency_chain(self, module_name: str):
        """Восстановление цепочки зависимостей для модуля"""
        if module_name not in self.modules:
            return
        
        module_info = self.modules[module_name]
        
        # Восстанавливаем зависимости сначала
        for dep in module_info.dependencies:
            if dep in self.modules and not self.modules[dep].is_active:
                self.logger.info(f"Восстановление зависимости {dep} для {module_name}")
                await self.recover_module(dep, force=True)
        
        # Затем пробуем восстановить основной модуль
        await asyncio.sleep(1.0)  # Даём время зависимостям активироваться
        await self.recover_module(module_name, force=True)
    
    async def _emergency_recovery_protocol(self):
        """Экстренный протокол восстановления при критическом состоянии"""
        self.logger.critical("🚨 АКТИВАЦИЯ ЭКСТРЕННОГО ПРОТОКОЛА ВОССТАНОВЛЕНИЯ")
        
        # 1. Останавливаем все фоновые задачи кроме критических
        await self._stop_non_critical_background_tasks()
        
        # 2. Деактивируем все модули
        deactivation_results = []
        for name, module in self.modules.items():
            if module.is_active and module.instance:
                try:
                    await module.instance.shutdown()
                    module.is_active = False
                    deactivation_results.append({"module": name, "status": "shutdown"})
                except Exception as e:
                    deactivation_results.append({"module": name, "status": "error", "error": str(e)})
        
        self.logger.info(f"Деактивировано модулей: {len([r for r in deactivation_results if r['status'] == 'shutdown'])}")
        
        # 3. Перезапускаем систему с чистого листа
        await asyncio.sleep(2.0)
        
        # 4. Активируем критические модули в правильном порядке
        critical_order = ["spirit_synthesis", "spirit_core", "core_govx"]
        activation_results = []
        
        for module_name in critical_order:
            if module_name in self.modules:
                result = await self.recover_module(module_name, force=True)
                activation_results.append({"module": module_name, "result": result})
                await asyncio.sleep(1.0)  # Пауза между активациями
        
        # 5. Активируем остальные модули
        other_modules = [name for name in self.modules if name not in critical_order]
        for module_name in other_modules:
            result = await self.recover_module(module_name, force=False)
            activation_results.append({"module": module_name, "result": result})
            await asyncio.sleep(0.5)
        
        # Формируем отчёт
        emergency_report = {
            "timestamp": time.time(),
            "deactivation_results": deactivation_results,
            "activation_results": activation_results,
            "final_active": sum(1 for m in self.modules.values() if m.is_active),
            "total_modules": len(self.modules)
        }
        
        self.logger.critical(f"Экстренный протокол завершён. Активных модулей: {emergency_report['final_active']}/{emergency_report['total_modules']}")
        
        # Публикуем событие
        await self._publish_internal_event("recovery.emergency_completed", emergency_report)
        
        return emergency_report
    
    async def _stop_non_critical_background_tasks(self):
        """Остановка некритических фоновых задач"""
        # Сохраняем только обработчик событий
        critical_tasks = []
        non_critical_tasks = []
        
        for task in self.background_tasks:
            task_name = task.get_name() if hasattr(task, 'get_name') else str(task)
            
            if "event_processor" in task_name or "recovery_monitor" in task_name:
                critical_tasks.append(task)
            else:
                non_critical_tasks.append(task)
        
        # Останавливаем некритические задачи
        for task in non_critical_tasks:
            try:
                task.cancel()
            except:
                pass
        
        self.background_tasks = critical_tasks
        self.logger.info(f"Остановлено некритических задач: {len(non_critical_tasks)}")
    
    async def get_recovery_status(self) -> Dict[str, Any]:
        """
        Полный отчёт о состоянии системы восстановления
        """
        module_statuses = {}
        
        for module_name, module_info in self.modules.items():
            recovery_key = f"{module_name}_recovery"
            attempts = self.error_counters.get(recovery_key, 0)
            blocked = attempts >= self.config["recovery"]["max_recovery_attempts"]
            
            # Проверяем зависимости
            deps_status = []
            for dep in module_info.dependencies:
                if dep in self.modules:
                    deps_status.append({
                        "name": dep,
                        "active": self.modules[dep].is_active,
                        "available": True
                    })
                else:
                    deps_status.append({
                        "name": dep,
                        "active": False,
                        "available": False
                    })
            
            module_statuses[module_name] = {
                "active": module_info.is_active,
                "recovery_attempts": attempts,
                "recovery_blocked": blocked,
                "max_attempts": self.config["recovery"]["max_recovery_attempts"],
                "dependencies": deps_status,
                "all_dependencies_active": all(dep["active"] for dep in deps_status if dep["available"]),
                "last_activation": self.activation_timestamps.get(module_name),
                "activation_order": module_info.activation_order,
                "is_critical": module_name in ["spirit_synthesis", "spirit_core", "core_govx"]
            }
        
        # Статистика
        total_modules = len(module_statuses)
        active_modules = sum(1 for s in module_statuses.values() if s["active"])
        blocked_modules = [name for name, s in module_statuses.items() if s["recovery_blocked"]]
        critical_inactive = [
            name for name, s in module_statuses.items()
            if s["is_critical"] and not s["active"]
        ]
        
        # Определяем общее здоровье системы восстановления
        if len(critical_inactive) > 0:
            recovery_health = "critical"
        elif len(blocked_modules) > 0:
            recovery_health = "degraded"
        elif active_modules == total_modules:
            recovery_health = "healthy"
        else:
            recovery_health = "warning"
        
        return {
            "timestamp": time.time(),
            "recovery_enabled": self.config["recovery"]["enabled"],
            "auto_recovery_enabled": self.config["recovery"]["auto_recover"],
            "health": recovery_health,
            "statistics": {
                "total_modules": total_modules,
                "active_modules": active_modules,
                "inactive_modules": total_modules - active_modules,
                "blocked_modules": len(blocked_modules),
                "critical_inactive": len(critical_inactive),
                "recovery_attempts_total": sum(self.error_counters.get(f"{name}_recovery", 0) for name in self.modules)
            },
            "critical_issues": {
                "blocked_modules": blocked_modules,
                "critical_inactive": critical_inactive,
                "modules_with_missing_deps": [
                    name for name, s in module_statuses.items()
                    if not s["all_dependencies_active"] and not s["active"]
                ]
            },
            "module_statuses": module_statuses,
            "config": {
                "max_recovery_attempts": self.config["recovery"]["max_recovery_attempts"],
                "auto_recover": self.config["recovery"]["auto_recover"],
                "monitor_interval": 10.0  # Хардкод, так как не в конфиге
            }
        }
    
    async def reset_recovery_attempts(self, module_name: str = None) -> Dict[str, Any]:
        """
        Сброс счётчиков попыток восстановления
        """
        reset_results = []
        
        if module_name:
            # Сброс для конкретного модуля
            if module_name not in self.modules:
                return {
                    "success": False,
                    "reason": "module_not_found",
                    "module": module_name
                }
            
            recovery_key = f"{module_name}_recovery"
            old_value = self.error_counters.get(recovery_key, 0)
            
            if recovery_key in self.error_counters:
                del self.error_counters[recovery_key]
            
            reset_results.append({
                "module": module_name,
                "old_attempts": old_value,
                "new_attempts": 0
            })
            
            self.logger.info(f"Сброс попыток восстановления для модуля {module_name}: {old_value} → 0")
            
        else:
            # Сброс для всех модулей
            for key in list(self.error_counters.keys()):
                if key.endswith("_recovery"):
                    module = key.replace("_recovery", "")
                    old_value = self.error_counters[key]
                    
                    reset_results.append({
                        "module": module,
                        "old_attempts": old_value,
                        "new_attempts": 0
                    })
                    
                    del self.error_counters[key]
            
            self.logger.info(f"Сброс попыток восстановления для {len(reset_results)} модулей")
        
        # Публикуем событие
        await self._publish_internal_event("recovery.attempts_reset", {
            "reset_results": reset_results,
            "timestamp": time.time()
        })
        
        return {
            "success": True,
            "reset_count": len(reset_results),
            "reset_modules": [r["module"] for r in reset_results],
            "details": reset_results,
            "timestamp": time.time()
        }
    
    async def get_recovery_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Получение истории восстановлений из событий
        """
        # В реальной реализации здесь был бы доступ к хранилищу событий
        # Сейчас возвращаем заглушку
        return [
            {
                "timestamp": time.time() - i * 3600,  # Имитация времени
                "type": "auto_recovery" if i % 3 == 0 else "manual_recovery",
                "modules_recovered": max(1, 5 - i % 5),
                "success_rate": 0.8 - i * 0.1
            }
            for i in range(min(limit, 20))
        ]
    
        # ========================================================
    # 11. API ШЛЮЗ И УПРАВЛЕНИЕ (ПОЛНОЕ)
    # ========================================================
    
    async def api_call(self, 
                      endpoint: str, 
                      method: str = "GET",
                      data: Optional[Dict] = None,
                      api_key: Optional[str] = None,
                      client_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        ПОЛНЫЙ API шлюз с маршрутизацией, валидацией, аутентификацией и лимитами
        """
        # Начало обработки запроса
        request_id = f"req_{int(time.time())}_{hash(endpoint) % 10000:04d}"
        start_time = time.time()
        
        self.logger.info(f"🌐 API запрос [{request_id}]: {method} {endpoint}")
        
        # Проверка аутентификации
        auth_result = await self._api_authenticate(api_key, client_info)
        if not auth_result["authenticated"]:
            self.logger.warning(f"API аутентификация провалена [{request_id}]: {auth_result.get('reason')}")
            return {
                "request_id": request_id,
                "error": "authentication_failed",
                "message": auth_result.get("message", "Invalid credentials"),
                "status_code": 401,
                "timestamp": time.time(),
                "processing_time": time.time() - start_time
            }
        
        # Проверка лимитов запросов
        if not await self._api_check_rate_limit(client_info):
            return {
                "request_id": request_id,
                "error": "rate_limit_exceeded",
                "message": "Too many requests",
                "status_code": 429,
                "timestamp": time.time(),
                "processing_time": time.time() - start_time
            }
        
        # Нормализация endpoint
        endpoint = endpoint.strip('/')
        if not endpoint.startswith('/'):
            endpoint = '/' + endpoint
        
        # Таблица маршрутизации API
        api_routes = {
            # === СИСТЕМНЫЕ ЭНДПОИНТЫ ===
            ("GET", "/"): self._api_root,
            ("GET", "/status"): self._api_system_status,
            ("GET", "/health"): self._api_system_health,
            ("GET", "/version"): self._api_version_info,
            ("GET", "/config"): self._api_get_config,
            
            # === МЕТРИКИ И МОНИТОРИНГ ===
            ("GET", "/metrics"): self._api_get_metrics,
            ("GET", "/metrics/latest"): self._api_get_latest_metrics,
            ("GET", "/metrics/history"): self._api_get_metrics_history,
            ("GET", "/metrics/module/{module}"): self._api_get_module_metrics,
            
            # === УПРАВЛЕНИЕ МОДУЛЯМИ ===
            ("GET", "/modules"): self._api_list_modules,
            ("GET", "/modules/all"): self._api_get_all_modules_info,
            ("GET", "/modules/{module}"): self._api_get_module_info,
            ("GET", "/modules/{module}/health"): self._api_get_module_health,
            ("GET", "/modules/{module}/status"): self._api_get_module_status,
            ("POST", "/modules/{module}/activate"): self._api_activate_module,
            ("POST", "/modules/{module}/deactivate"): self._api_deactivate_module,
            ("POST", "/modules/{module}/restart"): self._api_restart_module,
            
            # === ВОССТАНОВЛЕНИЕ ===
            ("GET", "/recovery"): self._api_get_recovery_status,
            ("GET", "/recovery/status"): self._api_get_recovery_status_full,
            ("POST", "/recovery/{module}"): self._api_recover_module,
            ("POST", "/recovery/auto"): self._api_auto_recover,
            ("POST", "/recovery/reset"): self._api_reset_recovery_attempts,
            ("GET", "/recovery/history"): self._api_get_recovery_history,
            
            # === ЭНЕРГЕТИЧЕСКОЕ УПРАВЛЕНИЕ ===
            ("GET", "/energy"): self._api_get_energy_status,
            ("GET", "/energy/flows"): self._api_get_energy_flows,
            ("POST", "/energy/distribute"): self._api_distribute_energy,
            ("POST", "/energy/recharge"): self._api_recharge_energy,
            ("POST", "/energy/set_reserve"): self._api_set_energy_reserve,
            
            # === СОБЫТИЯ ===
            ("GET", "/events"): self._api_get_event_capabilities,
            ("POST", "/events/subscribe"): self._api_subscribe_to_event,
            ("POST", "/events/publish"): self._api_publish_event,
            ("GET", "/events/subscriptions"): self._api_get_subscriptions,
            
            # === УПРАВЛЕНИЕ СИСТЕМОЙ ===
            ("POST", "/system/activate"): self._api_activate_system,
            ("POST", "/system/shutdown"): self._api_shutdown_system,
            ("POST", "/system/restart"): self._api_restart_system,
            ("GET", "/system/diagnostics"): self._api_get_diagnostics,
            
            # === АДМИНИСТРИРОВАНИЕ ===
            ("POST", "/admin/reload_config"): self._api_reload_config,
            ("POST", "/admin/clear_cache"): self._api_clear_cache,
            ("GET", "/admin/performance"): self._api_get_performance_stats,
        }
        
        # Поиск подходящего маршрута
        handler = None
        route_params = {}
        
        for (route_method, route_pattern), route_handler in api_routes.items():
            if method != route_method:
                continue
            
            # Проверяем точное совпадение
            if route_pattern == endpoint:
                handler = route_handler
                break
            
            # Проверяем паттерн с параметрами
            if '{' in route_pattern and '}' in route_pattern:
                # Создаём regex из паттерна
                import re
                pattern_parts = route_pattern.split('/')
                endpoint_parts = endpoint.split('/')
                
                if len(pattern_parts) != len(endpoint_parts):
                    continue
                
                match = True
                params = {}
                
                for i in range(len(pattern_parts)):
                    if pattern_parts[i].startswith('{') and pattern_parts[i].endswith('}'):
                        param_name = pattern_parts[i][1:-1]
                        params[param_name] = endpoint_parts[i]
                    elif pattern_parts[i] != endpoint_parts[i]:
                        match = False
                        break
                
                if match:
                    handler = route_handler
                    route_params = params
                    break
        
        # Если маршрут не найден
        if not handler:
            processing_time = time.time() - start_time
            self.logger.warning(f"API маршрут не найден [{request_id}]: {method} {endpoint}")
            
            # Возвращаем список доступных эндпоинтов
            available_endpoints = []
            for (route_method, route_pattern), _ in api_routes.items():
                if route_method in ["GET", "POST"]:  # Фильтруем по методам
                    available_endpoints.append(f"{route_method} {route_pattern}")
            
            return {
                "request_id": request_id,
                "error": "endpoint_not_found",
                "message": f"No handler for {method} {endpoint}",
                "status_code": 404,
                "available_endpoints": sorted(available_endpoints),
                "processing_time": processing_time,
                "timestamp": time.time()
            }
        
        # Выполнение обработчика
        try:
            # Подготавливаем контекст запроса
            request_context = {
                "request_id": request_id,
                "endpoint": endpoint,
                "method": method,
                "data": data or {},
                "params": route_params,
                "client_info": client_info or {},
                "auth_info": auth_result,
                "start_time": start_time
            }
            
            # Вызываем обработчик
            result = await handler(request_context)
            
            # Добавляем метаданные
            processing_time = time.time() - start_time
            result.update({
                "request_id": request_id,
                "processing_time": round(processing_time, 4),
                "timestamp": time.time(),
                "success": result.get("error") is None
            })
            
            # Логируем успешный запрос
            self.logger.info(f"✅ API запрос завершён [{request_id}]: {method} {endpoint} ({processing_time:.3f}s)")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = str(e)
            
            self.logger.error(f"❌ Ошибка обработки API [{request_id}]: {error_msg}")
            
            # Публикуем событие об ошибке API
            await self._publish_internal_event("api.error", {
                "request_id": request_id,
                "endpoint": endpoint,
                "method": method,
                "error": error_msg,
                "processing_time": processing_time,
                "timestamp": time.time()
            })
            
            return {
                "request_id": request_id,
                "error": "internal_server_error",
                "message": error_msg,
                "status_code": 500,
                "processing_time": processing_time,
                "timestamp": time.time()
            }
    
    # ========================================================
    # 11.1 АУТЕНТИФИКАЦИЯ И ЛИМИТЫ
    # ========================================================
    
    async def _api_authenticate(self, api_key: Optional[str], client_info: Optional[Dict]) -> Dict[str, Any]:
        """Аутентификация API запроса"""
        # Если аутентификация отключена - пропускаем
        if not self.config["api"]["auth_required"]:
            return {
                "authenticated": True,
                "auth_method": "none",
                "access_level": "full"
            }
        
        # Проверка API ключа
        valid_keys = {
            "ISKRA4_KETER_MASTER_KEY": {"level": "admin", "rate_limit": 1000},
            "KETHERIC_BLOCK_ADMIN": {"level": "admin", "rate_limit": 500},
            "SEPHIROTIC_ENGINE": {"level": "system", "rate_limit": 100},
            "METRICS_COLLECTOR": {"level": "monitor", "rate_limit": 50},
            "MODULE_INTEGRATION": {"level": "module", "rate_limit": 200},
        }
        
        if api_key and api_key in valid_keys:
            key_info = valid_keys[api_key]
            return {
                "authenticated": True,
                "auth_method": "api_key",
                "access_level": key_info["level"],
                "rate_limit": key_info["rate_limit"],
                "key_type": "valid"
            }
        
        # Проверка по client_info (например, для внутренних вызовов)
        if client_info and client_info.get("internal_call") == True:
            return {
                "authenticated": True,
                "auth_method": "internal",
                "access_level": "system",
                "rate_limit": 1000
            }
        
        return {
            "authenticated": False,
            "auth_method": "none",
            "access_level": "none",
            "message": "Invalid API key or credentials",
            "reason": "invalid_key"
        }
    
    async def _api_check_rate_limit(self, client_info: Optional[Dict]) -> bool:
        """Проверка лимитов запросов"""
        # TODO: Реализовать полноценную систему rate limiting
        # Сейчас просто возвращаем True
        return True
    
    # ========================================================
    # 11.2 ОСНОВНЫЕ API ОБРАБОТЧИКИ
    # ========================================================
    
    async def _api_root(self, context: Dict) -> Dict[str, Any]:
        """Корневой эндпоинт API"""
        return {
            "system": "ISKRA-4 Ketheric Block",
            "sephira": "KETER",
            "version": self.__version__,
            "status": "operational" if self.is_activated else "inactive",
            "endpoints": {
                "system": "/status, /health, /version, /config",
                "modules": "/modules, /modules/{module}, /modules/{module}/health",
                "metrics": "/metrics, /metrics/latest, /metrics/history",
                "energy": "/energy, /energy/flows, /energy/distribute",
                "recovery": "/recovery, /recovery/{module}, /recovery/auto",
                "events": "/events, /events/subscribe, /events/publish",
                "system_control": "/system/activate, /system/shutdown, /system/restart",
                "admin": "/admin/reload_config, /admin/clear_cache"
            },
            "active_modules": f"{sum(1 for m in self.modules.values() if m.is_active)}/{len(self.modules)}",
            "uptime": round(time.time() - self.activation_start_time, 1) if self.is_activated else 0
        }
    
    async def _api_system_status(self, context: Dict) -> Dict[str, Any]:
        """Статус системы"""
        active_modules = sum(1 for m in self.modules.values() if m.is_active)
        total_modules = len(self.modules)
        
        return {
            "sephira": self.__sephira__,
            "version": self.__version__,
            "status": "active" if self.is_activated else "inactive",
            "activation_time": self.activation_start_time if self.is_activated else None,
            "uptime": round(time.time() - self.activation_start_time, 1) if self.is_activated else 0,
            "modules": {
                "total": total_modules,
                "active": active_modules,
                "inactive": total_modules - active_modules,
                "health_percentage": round((active_modules / total_modules) * 100, 1) if total_modules > 0 else 0
            },
            "energy": {
                "reserve": self.energy_reserve,
                "status": "critical" if self.energy_reserve < self.config["energy"]["critical_threshold"] else "normal"
            },
            "events": {
                "queue_size": self.event_queue.qsize(),
                "max_queue": self.event_queue.maxsize
            },
            "background_tasks": len(self.background_tasks),
            "performance": {
                "request_id": context["request_id"],
                "api_version": "1.0"
            }
        }
    
    async def _api_system_health(self, context: Dict) -> Dict[str, Any]:
        """Полная проверка здоровья системы"""
        health_report = await self.get_system_health_report()
        return health_report
    
    async def _api_version_info(self, context: Dict) -> Dict[str, Any]:
        """Информация о версии"""
        return {
            "system": "ISKRA-4 Ketheric Block",
            "sephira": self.__sephira__,
            "core_version": self.__version__,
            "architecture": self.__architecture__,
            "python_version": sys.version,
            "modules": {
                name: {
                    "active": module.is_active,
                    "path": module.path,
                    "order": module.activation_order
                }
                for name, module in self.modules.items()
            },
            "capabilities": [
                "module_registry",
                "cascade_activation", 
                "energy_management",
                "event_routing",
                "metrics_collection",
                "auto_recovery",
                "api_gateway"
            ],
            "timestamp": time.time()
        }
    
    async def _api_get_config(self, context: Dict) -> Dict[str, Any]:
        """Получение конфигурации (без чувствительных данных)"""
        # Фильтруем чувствительные данные
        safe_config = {
            "activation": self.config["activation"],
            "energy": self.config["energy"],
            "events": self.config["events"],
            "recovery": self.config["recovery"],
            "metrics": self.config["metrics"],
            "api": {
                "enabled": self.config["api"]["enabled"],
                "host": self.config["api"]["host"],
                "port": self.config["api"]["port"]
            }
        }
        
        return {
            "config": safe_config,
            "sephira": self.__sephira__,
            "timestamp": time.time()
        }
    
    # ========================================================
    # 11.3 МЕТРИКИ И МОНИТОРИНГ API
    # ========================================================
    
    async def _api_get_metrics(self, context: Dict) -> Dict[str, Any]:
        """Получение текущих метрик"""
        return await self.collect_metrics()
    
    async def _api_get_latest_metrics(self, context: Dict) -> Dict[str, Any]:
        """Получение последних метрик с фильтрацией"""
        metrics = await self.collect_metrics()
        
        # Фильтруем по параметрам запроса
        params = context.get("params", {})
        data = context.get("data", {})
        
        filter_module = params.get("module") or data.get("module")
        if filter_module and filter_module in metrics["modules"]:
            return {
                "module": filter_module,
                "metrics": metrics["modules"][filter_module],
                "timestamp": metrics["timestamp"]
            }
        
        # Возвращаем сводные метрики
        summary = {
            "system": metrics["system"],
            "energy": metrics["energy"],
            "performance": metrics["performance"],
            "modules_summary": {
                "total": len(metrics["modules"]),
                "active": sum(1 for m in metrics["modules"].values() if m.get("active")),
                "with_errors": sum(1 for m in metrics["modules"].values() if "error" in m)
            }
        }
        
        return {
            "summary": summary,
            "timestamp": metrics["timestamp"]
        }
    
    async def _api_get_metrics_history(self, context: Dict) -> Dict[str, Any]:
        """Получение истории метрик"""
        data = context.get("data", {})
        limit = data.get("limit", 100)
        
        history = await self.get_metrics_history(limit)
        
        return {
            "history": history,
            "total_records": len(history),
            "limit": limit,
            "timestamp": time.time()
        }
    
    async def _api_get_module_metrics(self, context: Dict) -> Dict[str, Any]:
        """Метрики конкретного модуля"""
        module_name = context["params"].get("module")
        
        if not module_name or module_name not in self.modules:
            return {
                "error": "module_not_found",
                "message": f"Module {module_name} not found",
                "available_modules": list(self.modules.keys())
            }
        
        module_info = self.modules[module_name]
        
        if not module_info.is_active or not module_info.instance:
            return {
                "module": module_name,
                "active": False,
                "message": "Module is not active"
            }
        
        try:
            metrics = await module_info.instance.get_metrics()
            return {
                "module": module_name,
                "active": True,
                "metrics": metrics,
                "activation_order": module_info.activation_order,
                "timestamp": time.time()
            }
        except Exception as e:
            return {
                "module": module_name,
                "active": True,
                "error": str(e),
                "timestamp": time.time()
            }
    
    # ========================================================
    # 11.4 УПРАВЛЕНИЕ МОДУЛЯМИ API
    # ========================================================
    
    async def _api_list_modules(self, context: Dict) -> Dict[str, Any]:
        """Список всех модулей"""
        modules_list = []
        
        for name, module in self.modules.items():
            modules_list.append({
                "name": name,
                "active": module.is_active,
                "activation_order": module.activation_order,
                "dependencies": module.dependencies,
                "path": module.path,
                "has_instance": module.instance is not None
            })
        
        # Сортируем по порядку активации
        modules_list.sort(key=lambda x: x["activation_order"] or 999)
        
        return {
            "modules": modules_list,
            "total": len(modules_list),
            "active": sum(1 for m in modules_list if m["active"]),
            "timestamp": time.time()
        }
    
    async def _api_get_all_modules_info(self, context: Dict) -> Dict[str, Any]:
        """Полная информация о всех модулях"""
        modules_info = {}
        
        for name, module in self.modules.items():
            health = await self.get_module_health(name)
            modules_info[name] = health
        
        return {
            "modules": modules_info,
            "summary": {
                "total": len(modules_info),
                "active": sum(1 for m in modules_info.values() if m.get("active")),
                "healthy": sum(1 for m in modules_info.values() if m.get("active") and "error" not in m),
                "with_dependencies": sum(1 for m in modules_info.values() if m.get("dependencies"))
            },
            "timestamp": time.time()
        }
    
    async def _api_get_module_info(self, context: Dict) -> Dict[str, Any]:
        """Информация о конкретном модуле"""
        module_name = context["params"].get("module")
        
        if not module_name or module_name not in self.modules:
            return {
                "error": "module_not_found",
                "message": f"Module {module_name} not found",
                "available_modules": list(self.modules.keys())
            }
        
        module = self.modules[module_name]
        
        info = {
            "name": module_name,
            "active": module.is_active,
            "activation_order": module.activation_order,
            "dependencies": module.dependencies,
            "path": module.path,
            "config": module.config,
            "instance_present": module.instance is not None,
            "activation_time": self.activation_timestamps.get(module_name),
            "error_count": self.error_counters.get(module_name, 0),
            "recovery_attempts": self.error_counters.get(f"{module_name}_recovery", 0)
        }
        
        # Добавляем информацию о зависимостях
        deps_status = []
        for dep in module.dependencies:
            if dep in self.modules:
                dep_module = self.modules[dep]
                deps_status.append({
                    "name": dep,
                    "active": dep_module.is_active,
                    "order": dep_module.activation_order
                })
            else:
                deps_status.append({
                    "name": dep,
                    "active": False,
                    "error": "not_registered"
                })
        
        info["dependencies_status"] = deps_status
        info["all_dependencies_active"] = all(dep["active"] for dep in deps_status)
        
        return info
    
    async def _api_get_module_health(self, context: Dict) -> Dict[str, Any]:
        """Здоровье конкретного модуля"""
        module_name = context["params"].get("module")
        
        if not module_name or module_name not in self.modules:
            return {
                "error": "module_not_found",
                "message": f"Module {module_name} not found"
            }
        
        return await self.get_module_health(module_name)
    
    async def _api_get_module_status(self, context: Dict) -> Dict[str, Any]:
        """Статус конкретного модуля"""
        module_name = context["params"].get("module")
        
        if not module_name or module_name not in self.modules:
            return {
                "error": "module_not_found",
                "message": f"Module {module_name} not found"
            }
        
        module = self.modules[module_name]
        
        status = "active" if module.is_active else "inactive"
        
        if not module.is_active:
            if module_name in self.activation_timestamps:
                status = "failed"
            else:
                status = "never_activated"
        
        return {
            "module": module_name,
            "status": status,
            "active": module.is_active,
            "order": module.activation_order,
            "uptime": time.time() - self.activation_timestamps.get(module_name, 0) if module.is_active else 0,
            "timestamp": time.time()
        }
    
    async def _api_activate_module(self, context: Dict) -> Dict[str, Any]:
        """Активация модуля через API"""
        module_name = context["params"].get("module")
        
        if not module_name or module_name not in self.modules:
            return {
                "error": "module_not_found",
                "message": f"Module {module_name} not found"
            }
        
        module = self.modules[module_name]
        
        if module.is_active:
            return {
                "module": module_name,
                "status": "already_active",
                "message": "Module is already active",
                "order": module.activation_order
            }
        
        try:
            success = await module.instance.activate()
            
            if success:
                module.is_active = True
                module.activation_order = max(
                    [m.activation_order for m in self.modules.values() if m.is_active],
                    default=0
                ) + 1
                
                self.activation_timestamps[module_name] = time.time()
                
                return {
                    "module": module_name,
                    "status": "activated",
                    "success": True,
                    "new_order": module.activation_order,
                    "timestamp": time.time()
                }
            else:
                return {
                    "module": module_name,
                    "status": "activation_failed",
                    "success": False,
                    "message": "Module.activate() returned False",
                    "timestamp": time.time()
                }
                
        except Exception as e:
            return {
                "module": module_name,
                "status": "activation_error",
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _api_deactivate_module(self, context: Dict) -> Dict[str, Any]:
        """Деактивация модуля через API"""
        module_name = context["params"].get("module")
        
        if not module_name or module_name not in self.modules:
            return {
                "error": "module_not_found",
                "message": f"Module {module_name} not found"
            }
        
        module = self.modules[module_name]
        
        if not module.is_active or not module.instance:
            return {
                "module": module_name,
                "status": "already_inactive",
                "message": "Module is already inactive",
                "timestamp": time.time()
            }
        
        try:
            await module.instance.shutdown()
            module.is_active = False
            
            return {
                "module": module_name,
                "status": "deactivated",
                "success": True,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "module": module_name,
                "status": "deactivation_error",
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _api_restart_module(self, context: Dict) -> Dict[str, Any]:
        """Перезапуск модуля через API"""
        module_name = context["params"].get("module")
        
        if not module_name or module_name not in self.modules:
            return {
                "error": "module_not_found",
                "message": f"Module {module_name} not found"
            }
        
        # Деактивация
        deactivate_result = await self._api_deactivate_module(context)
        if not deactivate_result.get("success"):
            return deactivate_result
        
        # Пауза
        await asyncio.sleep(0.5)
        
        # Активация
        activate_result = await self._api_activate_module(context)
        
        return {
            "module": module_name,
            "operation": "restart",
            "deactivation": deactivate_result,
            "activation": activate_result,
            "overall_success": activate_result.get("success", False),
            "timestamp": time.time()
        }
    
    # ========================================================
    # 11.5 ВОССТАНОВЛЕНИЕ API
    # ========================================================
    
    async def _api_get_recovery_status(self, context: Dict) -> Dict[str, Any]:
        """Статус системы восстановления"""
        return await self.get_recovery_status()
    
    async def _api_get_recovery_status_full(self, context: Dict) -> Dict[str, Any]:
        """Полный статус восстановления"""
        status = await self.get_recovery_status()
        
        # Добавляем дополнительную информацию
        failed_modules = [
            name for name, module in self.modules.items()
            if not module.is_active
        ]
        
        recovery_blocked = [
            name for name in failed_modules
            if self.error_counters.get(f"{name}_recovery", 0) >= self.config["recovery"]["max_recovery_attempts"]
        ]
        
        status["detailed"] = {
            "failed_modules": failed_modules,
            "recovery_blocked": recovery_blocked,
            "can_auto_recover": self.config["recovery"]["auto_recover"],
            "auto_recovery_enabled": self.config["recovery"]["enabled"]
        }
        
        return status
    
    async def _api_recover_module(self, context: Dict) -> Dict[str, Any]:
        """Восстановление модуля через API"""
        module_name = context["params"].get("module")
        data = context.get("data", {})
        force = data.get("force", False)
        
        if not module_name or module_name not in self.modules:
            return {
                "error": "module_not_found",
                "message": f"Module {module_name} not found"
            }
        
        return await self.recover_module(module_name, force)
    
    async def _api_auto_recover(self, context: Dict) -> Dict[str, Any]:
        """Автовосстановление через API"""
        return await self.auto_recover_failed_modules()
    
    async def _api_reset_recovery_attempts(self, context: Dict) -> Dict[str, Any]:
        """Сброс попыток восстановления через API"""
        data = context.get("data", {})
        module_name = data.get("module")
        
        return await self.reset_recovery_attempts(module_name)
    
    async def _api_get_recovery_history(self, context: Dict) -> Dict[str, Any]:
        """История восстановлений"""
        data = context.get("data", {})
        limit = data.get("limit", 50)
        
        history = await self.get_recovery_history(limit)
        
        return {
            "history": history,
            "limit": limit,
            "total": len(history),
            "timestamp": time.time()
        }
    
    # ========================================================
    # 11.6 ЭНЕРГЕТИЧЕСКОЕ УПРАВЛЕНИЕ API
    # ========================================================
    
    async def _api_get_energy_status(self, context: Dict) -> Dict[str, Any]:
        """Статус энергии"""
        return {
            "energy": {
                "reserve": self.energy_reserve,
                "critical_threshold": self.config["energy"]["critical_threshold"],
                "status": "critical" if self.energy_reserve < self.config["energy"]["critical_threshold"] else "normal",
                "recharge_rate": self.config["energy"]["recharge_rate"]
            },
            "flows": {
                "total": len(self.energy_flows),
                "active": sum(1 for f in self.energy_flows if f.current_flow > 0),
                "by_priority": {
                    "critical": sum(1 for f in self.energy_flows if f.priority == "critical"),
                    "high": sum(1 for f in self.energy_flows if f.priority == "high"),
                    "medium": sum(1 for f in self.energy_flows if f.priority == "medium"),
                    "low": sum(1 for f in self.energy_flows if f.priority == "low")
                }
            },
            "timestamp": time.time()
        }
    
    async def _api_get_energy_flows(self, context: Dict) -> Dict[str, Any]:
        """Получение информации об энергетических потоках"""
        flows_info = []
        
        for flow in self.energy_flows:
            flows_info.append({
                "source": flow.source,
                "target": flow.target,
                "priority": flow.priority,
                "current_flow": flow.current_flow,
                "max_flow": flow.max_flow,
                "last_transfer": flow.last_transfer,
                "active": flow.current_flow > 0,
                "utilization": round((flow.current_flow / flow.max_flow) * 100, 1) if flow.max_flow > 0 else 0
            })
        
        # Сортируем по приоритету
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        flows_info.sort(key=lambda x: priority_order.get(x["priority"], 4))
        
        return {
            "flows": flows_info,
            "total": len(flows_info),
            "active": sum(1 for f in flows_info if f["active"]),
            "total_capacity": sum(f["max_flow"] for f in flows_info),
            "current_utilization": sum(f["current_flow"] for f in flows_info),
            "timestamp": time.time()
        }
    
    async def _api_distribute_energy(self, context: Dict) -> Dict[str, Any]:
        """Распределение энергии через API"""
        data = context.get("data", {})
        
        required = ["source", "target", "amount"]
        missing = [field for field in required if field not in data]
        if missing:
            return {
                "error": "missing_parameters",
                "message": f"Missing required parameters: {missing}",
                "required": required
            }
        
        source = data["source"]
        target = data["target"]
        amount = float(data["amount"])
        
        return await self.distribute_energy(source, target, amount)
    
    async def _api_recharge_energy(self, context: Dict) -> Dict[str, Any]:
        """Пополнение энергии через API"""
        data = context.get("data", {})
        amount = float(data.get("amount", 100.0))
        
        success = await self.recharge_energy(amount)
        
        return {
            "success": success,
            "amount": amount,
            "new_reserve": self.energy_reserve,
            "timestamp": time.time()
        }
    
    async def _api_set_energy_reserve(self, context: Dict) -> Dict[str, Any]:
        """Установка уровня энергетического резерва"""
        data = context.get("data", {})
        
        if "reserve" not in data:
            return {
                "error": "missing_parameter",
                "message": "Parameter 'reserve' is required",
                "timestamp": time.time()
            }
        
        new_reserve = float(data["reserve"])
        old_reserve = self.energy_reserve
        self.energy_reserve = new_reserve
        
        self.logger.info(f"Энергетический резерв изменён через API: {old_reserve:.1f} → {new_reserve:.1f}")
        
        return {
            "success": True,
            "old_reserve": old_reserve,
            "new_reserve": new_reserve,
            "difference": new_reserve - old_reserve,
            "timestamp": time.time()
        }
    
        # ========================================================
    # 11.7 СОБЫТИЯ API
    # ========================================================
    
    async def _api_get_event_capabilities(self, context: Dict) -> Dict[str, Any]:
        """Возможности системы событий"""
        event_types = list(self.event_handlers.keys())
        
        # Определяем системные события
        system_events = [
            "module.activated",
            "module.deactivated", 
            "module.recovered",
            "module.recovery_failed",
            "energy.distributed",
            "energy.recharged",
            "energy.critical",
            "system.critical_warning",
            "recovery.auto_completed",
            "recovery.emergency_completed",
            "recovery.attempts_reset",
            "api.error",
            "system.shutdown"
        ]
        
        # Определяем модульные события
        module_events = []
        for module_name in self.modules:
            module_events.extend([
                f"{module_name}.started",
                f"{module_name}.stopped",
                f"{module_name}.error",
                f"{module_name}.warning"
            ])
        
        return {
            "capabilities": {
                "total_event_types": len(event_types) + len(system_events) + len(module_events),
                "system_events": system_events,
                "module_events": module_events[:20],  # Ограничиваем вывод
                "custom_events": event_types,
                "queue_capacity": self.event_queue.maxsize,
                "current_queue_size": self.event_queue.qsize(),
                "subscriptions_count": sum(len(handlers) for handlers in self.event_handlers.values())
            },
            "subscription_methods": {
                "internal": "Через self.subscribe()",
                "api": "POST /events/subscribe",
                "webhook": "Поддержка webhooks (в разработке)"
            },
            "timestamp": time.time()
        }
    
    async def _api_subscribe_to_event(self, context: Dict) -> Dict[str, Any]:
        """Подписка на событие через API"""
        data = context.get("data", {})
        
        required = ["event_type", "callback_url"]
        missing = [field for field in required if field not in data]
        if missing:
            return {
                "error": "missing_parameters",
                "message": f"Missing required parameters: {missing}",
                "required": required,
                "timestamp": time.time()
            }
        
        event_type = data["event_type"]
        callback_url = data["callback_url"]
        filter_conditions = data.get("filters", {})
        
        # Создаём обработчик для webhook
        async def webhook_handler(event_data):
            import aiohttp
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(callback_url, json=event_data, timeout=5) as response:
                        if response.status != 200:
                            self.logger.warning(f"Webhook callback failed: {response.status}")
            except Exception as e:
                self.logger.error(f"Webhook error: {e}")
        
        # Подписываемся
        subscription_id = self.subscribe(event_type, webhook_handler)
        
        # Сохраняем информацию о подписке
        if not hasattr(self, '_webhook_subscriptions'):
            self._webhook_subscriptions = {}
        
        self._webhook_subscriptions[subscription_id] = {
            "event_type": event_type,
            "callback_url": callback_url,
            "filters": filter_conditions,
            "created": time.time(),
            "last_called": None
        }
        
        return {
            "success": True,
            "subscription_id": subscription_id,
            "event_type": event_type,
            "callback_url": callback_url,
            "message": f"Subscribed to {event_type}. Events will be sent to {callback_url}",
            "timestamp": time.time()
        }
    
    async def _api_publish_event(self, context: Dict) -> Dict[str, Any]:
        """Публикация события через API"""
        data = context.get("data", {})
        
        required = ["event_type", "data"]
        missing = [field for field in required if field not in data]
        if missing:
            return {
                "error": "missing_parameters",
                "message": f"Missing required parameters: {missing}",
                "required": required,
                "timestamp": time.time()
            }
        
        event_type = data["event_type"]
        event_data = data["data"]
        source = data.get("source", "api")
        
        # Публикуем событие
        await self._publish_internal_event(event_type, event_data)
        
        # Также маршрутизируем между модулями
        await self.route_event(event_type, event_data, source)
        
        return {
            "success": True,
            "event_type": event_type,
            "published": True,
            "source": source,
            "timestamp": time.time(),
            "queue_size": self.event_queue.qsize()
        }
    
    async def _api_get_subscriptions(self, context: Dict) -> Dict[str, Any]:
        """Получение списка подписок"""
        subscriptions = []
        
        # Внутренние подписки
        for event_type, handlers in self.event_handlers.items():
            for subscription_id, handler in handlers:
                subscriptions.append({
                    "id": subscription_id,
                    "event_type": event_type,
                    "handler_type": handler.__class__.__name__,
                    "source": "internal"
                })
        
        # Webhook подписки
        if hasattr(self, '_webhook_subscriptions'):
            for sub_id, sub_info in self._webhook_subscriptions.items():
                subscriptions.append({
                    "id": sub_id,
                    "event_type": sub_info["event_type"],
                    "callback_url": sub_info["callback_url"],
                    "filters": sub_info["filters"],
                    "created": sub_info["created"],
                    "last_called": sub_info["last_called"],
                    "source": "webhook"
                })
        
        return {
            "subscriptions": subscriptions,
            "total": len(subscriptions),
            "by_source": {
                "internal": sum(1 for s in subscriptions if s["source"] == "internal"),
                "webhook": sum(1 for s in subscriptions if s["source"] == "webhook")
            },
            "by_event_type": {
                event_type: sum(1 for s in subscriptions if s["event_type"] == event_type)
                for event_type in set(s["event_type"] for s in subscriptions)
            },
            "timestamp": time.time()
        }
    
    # ========================================================
    # 11.8 УПРАВЛЕНИЕ СИСТЕМОЙ API
    # ========================================================
    
    async def _api_activate_system(self, context: Dict) -> Dict[str, Any]:
        """Активация всей системы через API"""
        if self.is_activated:
            return {
                "status": "already_active",
                "message": "System is already activated",
                "active_modules": sum(1 for m in self.modules.values() if m.is_active),
                "total_modules": len(self.modules),
                "timestamp": time.time()
            }
        
        try:
            result = await self.activate_cascade()
            
            return {
                "status": "activation_started",
                "success": True,
                "result": result,
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "activation_failed",
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def _api_shutdown_system(self, context: Dict) -> Dict[str, Any]:
        """Выключение системы через API"""
        if not self.is_activated:
            return {
                "status": "already_inactive",
                "message": "System is already inactive",
                "timestamp": time.time()
            }
        
        # Публикуем событие выключения
        await self._publish_internal_event("system.shutdown", {
            "source": "api",
            "request_id": context.get("request_id"),
            "timestamp": time.time()
        })
        
        # Запускаем graceful shutdown
        shutdown_result = await self.shutdown()
        
        return {
            "status": "shutdown_initiated",
            "success": True,
            "result": shutdown_result,
            "message": "System shutdown initiated",
            "timestamp": time.time()
        }
    
    async def _api_restart_system(self, context: Dict) -> Dict[str, Any]:
        """Перезапуск системы через API"""
        # Сначала выключаем
        shutdown_result = await self._api_shutdown_system(context)
        
        if not shutdown_result.get("success"):
            return {
                "operation": "restart",
                "shutdown_phase": "failed",
                "error": shutdown_result.get("error"),
                "timestamp": time.time()
            }
        
        # Ждём завершения выключения
        await asyncio.sleep(2.0)
        
        # Затем включаем
        activation_result = await self._api_activate_system(context)
        
        return {
            "operation": "restart",
            "shutdown_phase": shutdown_result,
            "activation_phase": activation_result,
            "overall_success": activation_result.get("success", False),
            "timestamp": time.time()
        }
    
    async def _api_get_diagnostics(self, context: Dict) -> Dict[str, Any]:
        """Полная диагностика системы"""
        # Собираем информацию со всех модулей
        modules_diagnostics = {}
        
        for name, module in self.modules.items():
            if module.instance and module.is_active:
                try:
                    # Пробуем вызвать метод диагностики если есть
                    if hasattr(module.instance, 'get_diagnostics'):
                        modules_diagnostics[name] = await module.instance.get_diagnostics()
                    elif hasattr(module.instance, 'get_metrics'):
                        modules_diagnostics[name] = await module.instance.get_metrics()
                    else:
                        modules_diagnostics[name] = {"status": "no_diagnostics_method"}
                except Exception as e:
                    modules_diagnostics[name] = {"error": str(e)}
            else:
                modules_diagnostics[name] = {"status": "inactive"}
        
        # Собираем системную диагностику
        system_diagnostics = {
            "python": {
                "version": sys.version,
                "platform": sys.platform,
                "executable": sys.executable
            },
            "asyncio": {
                "loop_running": asyncio.get_event_loop().is_running(),
                "tasks": len(asyncio.all_tasks())
            },
            "memory": {
                # TODO: Добавить использование памяти
            },
            "timing": {
                "uptime": time.time() - self.activation_start_time if self.is_activated else 0,
                "current_time": time.time(),
                "timezone": time.tzname
            }
        }
        
        return {
            "system": system_diagnostics,
            "modules": modules_diagnostics,
            "keter_core": {
                "version": self.__version__,
                "modules_registered": len(self.modules),
                "modules_active": sum(1 for m in self.modules.values() if m.is_active),
                "energy_reserve": self.energy_reserve,
                "event_queue": self.event_queue.qsize(),
                "background_tasks": len(self.background_tasks),
                "error_counters": self.error_counters,
                "activation_timestamps": self.activation_timestamps
            },
            "timestamp": time.time()
        }
    
    # ========================================================
    # 11.9 АДМИНИСТРАТИВНЫЕ API
    # ========================================================
    
    async def _api_reload_config(self, context: Dict) -> Dict[str, Any]:
        """Перезагрузка конфигурации"""
        # TODO: Реализовать загрузку конфигурации из файла
        # Сейчас просто возвращаем текущую конфигурацию
        
        return {
            "operation": "reload_config",
            "status": "not_implemented",
            "message": "Config reload from file not implemented yet",
            "current_config": self.config,
            "timestamp": time.time()
        }
    
    async def _api_clear_cache(self, context: Dict) -> Dict[str, Any]:
        """Очистка кэшей"""
        data = context.get("data", {})
        cache_type = data.get("type", "all")
        
        cleared = []
        
        if cache_type in ["all", "metrics"]:
            old_size = len(self.metrics_history)
            self.metrics_history.clear()
            cleared.append({"type": "metrics", "entries_cleared": old_size})
        
        if cache_type in ["all", "events"]:
            # Очищаем очередь событий
            old_size = self.event_queue.qsize()
            while not self.event_queue.empty():
                try:
                    self.event_queue.get_nowait()
                    self.event_queue.task_done()
                except:
                    break
            cleared.append({"type": "events", "entries_cleared": old_size})
        
        return {
            "operation": "clear_cache",
            "success": True,
            "cache_type": cache_type,
            "cleared": cleared,
            "timestamp": time.time()
        }
    
    async def _api_get_performance_stats(self, context: Dict) -> Dict[str, Any]:
        """Статистика производительности"""
        # Собираем информацию о производительности
        api_requests = getattr(self, '_api_request_stats', [])
        
        # Информация о задачах
        tasks_info = []
        for task in self.background_tasks:
            try:
                tasks_info.append({
                    "name": task.get_name() if hasattr(task, 'get_name') else "unnamed",
                    "done": task.done(),
                    "cancelled": task.cancelled(),
                    "exception": str(task.exception()) if task.exception() else None
                })
            except:
                pass
        
        return {
            "performance": {
                "api_requests": {
                    "total": len(api_requests),
                    "last_hour": len([r for r in api_requests if r.get("timestamp", 0) > time.time() - 3600]),
                    "average_time": sum(r.get("processing_time", 0) for r in api_requests) / max(1, len(api_requests))
                },
                "background_tasks": {
                    "total": len(self.background_tasks),
                    "active": len([t for t in self.background_tasks if not t.done()]),
                    "tasks": tasks_info[:10]  # Ограничиваем вывод
                },
                "event_system": {
                    "queue_size": self.event_queue.qsize(),
                    "max_queue": self.event_queue.maxsize,
                    "subscriptions": sum(len(h) for h in self.event_handlers.values())
                },
                "modules": {
                    "total": len(self.modules),
                    "active": sum(1 for m in self.modules.values() if m.is_active),
                    "with_errors": sum(1 for name in self.modules if self.error_counters.get(name, 0) > 0)
                }
            },
            "timestamp": time.time()
        }
    
    # ========================================================
    # 12. ЗАПУСК И УПРАВЛЕНИЕ ФОНОВЫМИ ЗАДАЧАМИ
    # ========================================================
    
    async def _start_background_tasks(self):
        """Запуск всех фоновых задач"""
        self.logger.info("🚀 Запуск фоновых задач...")
        
        # Задачи для запуска
        background_tasks = [
            ("event_processor", self._event_processor_task),
            ("metrics_collector", self._metrics_collector_task),
            ("recovery_monitor", self._recovery_monitor_task),
            ("energy_manager", self._energy_manager_task),
        ]
        
        for task_name, task_func in background_tasks:
            try:
                task = asyncio.create_task(task_func(), name=task_name)
                self.background_tasks.append(task)
                self.logger.info(f"✅ Фоновая задача запущена: {task_name}")
                await asyncio.sleep(0.1)  # Небольшая пауза между запусками
            except Exception as e:
                self.logger.error(f"❌ Ошибка запуска задачи {task_name}: {e}")
    
    async def _energy_manager_task(self):
        """Фоновая задача управления энергией"""
        self.logger.info("⚡ Запуск менеджера энергии...")
        
        while not self.shutdown_requested:
            try:
                # Автоматическое пополнение энергии
                if self.energy_reserve < self.config["energy"]["reserve"] * 0.8:
                    recharge_amount = self.config["energy"]["recharge_rate"]
                    await self.recharge_energy(recharge_amount)
                
                # Балансировка энергетических потоков
                await self._balance_energy_flows()
                
                # Проверка критического уровня энергии
                if self.energy_reserve < self.config["energy"]["critical_threshold"]:
                    await self._publish_internal_event("energy.critical", {
                        "reserve": self.energy_reserve,
                        "threshold": self.config["energy"]["critical_threshold"],
                        "timestamp": time.time()
                    })
                
                await asyncio.sleep(5.0)  # Проверка каждые 5 секунд
                
            except Exception as e:
                self.logger.error(f"Ошибка менеджера энергии: {e}")
                await asyncio.sleep(10.0)
        
        self.logger.info("Менеджер энергии остановлен")
    
    async def _balance_energy_flows(self):
        """Балансировка энергетических потоков"""
        # Простая логика балансировки
        for flow in self.energy_flows:
            # Уменьшаем поток если давно не использовался
            if flow.current_flow > 0 and time.time() - flow.last_transfer > 30:
                flow.current_flow *= 0.9  # Постепенно уменьшаем
    
    async def _stop_all_background_tasks(self):
        """Остановка всех фоновых задач"""
        self.logger.info("🛑 Остановка фоновых задач...")
        
        self.shutdown_requested = True
        
        # Отменяем все задачи
        for task in self.background_tasks:
            try:
                task.cancel()
            except:
                pass
        
        # Ждём завершения задач
        if self.background_tasks:
            try:
                await asyncio.wait(self.background_tasks, timeout=5.0)
            except:
                pass
        
        self.background_tasks.clear()
        self.logger.info("Фоновые задачи остановлены")
    
    # ========================================================
    # 13. ГРАЦИОЗНОЕ ВЫКЛЮЧЕНИЕ
    # ========================================================
    
    async def shutdown(self) -> Dict[str, Any]:
        """
        Полное грациозное выключение системы
        """
        self.logger.info("🛑 Начало грациозного выключения KetherCore...")
        
        shutdown_start = time.time()
        shutdown_results = {}
        
        # 1. Останавливаем все фоновые задачи
        await self._stop_all_background_tasks()
        
        # 2. Деактивация модулей в обратном порядке
        reverse_order = sorted(
            [(name, module.activation_order) for name, module in self.modules.items() if module.is_active],
            key=lambda x: x[1],
            reverse=True
        )
        
        for module_name, _ in reverse_order:
            module_info = self.modules[module_name]
            
            if module_info.is_active and module_info.instance:
                try:
                    await module_info.instance.shutdown()
                    module_info.is_active = False
                    shutdown_results[module_name] = "success"
                    self.logger.info(f"✅ Модуль {module_name} выключен")
                except Exception as e:
                    shutdown_results[module_name] = f"error: {e}"
                    self.logger.error(f"❌ Ошибка выключения модуля {module_name}: {e}")
            else:
                shutdown_results[module_name] = "already_inactive"
        
        # 3. Очистка ресурсов
        self.is_activated = False
        self.event_handlers.clear()
        
        # 4. Публикация финальных метрик
        final_metrics = await self.collect_metrics()
        await self._publish_internal_event("system.shutdown_complete", {
            "shutdown_results": shutdown_results,
            "final_metrics": final_metrics,
            "shutdown_time": time.time() - shutdown_start,
            "timestamp": time.time()
        })
        
        total_time = time.time() - shutdown_start
        
        result = {
            "sephira": self.__sephira__,
            "version": self.__version__,
            "shutdown_completed": True,
            "total_time": round(total_time, 2),
            "results": shutdown_results,
            "successful_shutdowns": sum(1 for r in shutdown_results.values() if "success" in str(r)),
            "total_modules": len(shutdown_results),
            "timestamp": time.time()
        }
        
        self.logger.info(f"🎯 KetherCore выключен за {total_time:.2f}с. "
                        f"Успешно выключено: {result['successful_shutdowns']}/{result['total_modules']} модулей")
        
        return result
    
    # ========================================================
    # 14. УТИЛИТЫ
    # ========================================================
    
    def _deep_update(self, target: Dict, source: Dict) -> Dict:
        """Рекурсивное обновление словаря"""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
        return target
    
    async def _publish_internal_event(self, event_type: str, data: Dict) -> None:
        """Внутренняя публикация события с обработкой ошибок"""
        try:
            # Вызываем подписчиков
            if event_type in self.event_handlers:
                for subscription_id, handler in self.event_handlers[event_type]:
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(data)
                        else:
                            handler(data)
                    except Exception as e:
                        self.logger.error(f"Ошибка обработчика события {subscription_id}: {e}")
            
            # Помещаем в очередь если есть место
            if not self.event_queue.full():
                await self.event_queue.put({"type": event_type, "data": data})
                
        except Exception as e:
            self.logger.error(f"Ошибка публикации события {event_type}: {e}")
    
    def get_module(self, module_name: str) -> Optional[IKethericModule]:
        """Получение экземпляра модуля по имени"""
        if module_name in self.modules:
            return self.modules[module_name].instance
        return None
    
    def get_module_status(self, module_name: str) -> Optional[Dict]:
        """Получение статуса модуля"""
        if module_name in self.modules:
            module = self.modules[module_name]
            return {
                "active": module.is_active,
                "order": module.activation_order,
                "dependencies": module.dependencies,
                "has_instance": module.instance is not None
            }
        return None
    
    # ========================================================
    # 15. ТЕСТОВАЯ ФУНКЦИЯ И ЗАПУСК
    # ========================================================
    
    async def run_test_scenario(self) -> Dict[str, Any]:
        """Запуск тестового сценария интеграции"""
        self.logger.info("🧪 Запуск тестового сценария Ketheric Block...")
        
        test_results = {}
        
        # 1. Регистрация модулей
        test_results["registration"] = await self.register_all_modules()
        
        # 2. Активация
        test_results["activation"] = await self.activate_cascade()
        
        # 3. Сбор метрик
        test_results["metrics"] = await self.collect_metrics()
        
        # 4. Проверка API
        try:
            api_status = await self.api_call("/status", "GET", api_key="TEST_KEY")
            test_results["api_test"] = {"success": True, "response": api_status}
        except Exception as e:
            test_results["api_test"] = {"success": False, "error": str(e)}
        
        # 5. Проверка энергетических потоков
        energy_tests = []
        test_flows = [
            ("spirit_synthesis", "willpower_core", 10.0),
            ("spirit_core", "core_govx", 5.0),
        ]
        
        for source, target, amount in test_flows:
            result = await self.distribute_energy(source, target, amount)
            energy_tests.append({
                "flow": f"{source}→{target}",
                "amount": amount,
                "result": result
            })
        
        test_results["energy_tests"] = energy_tests
        
        # 6. Проверка восстановления
        test_results["recovery_status"] = await self.get_recovery_status()
        
        # 7. Грациозное выключение
        test_results["shutdown"] = await self.shutdown()
        
        # Итог
        active_modules = test_results["activation"]["activated_modules"]
        total_modules = test_results["activation"]["total_modules"]
        
        test_results["summary"] = {
            "success": active_modules == total_modules,
            "active_modules": f"{active_modules}/{total_modules}",
            "success_rate": (active_modules / total_modules) * 100 if total_modules > 0 else 0,
            "total_tests": 7,
            "passed_tests": sum(1 for key in ["registration", "activation", "metrics", "api_test", "energy_tests", "recovery_status", "shutdown"] 
                               if test_results.get(key, {}).get("success", False))
        }
        
        return test_results


# ============================================================
# 16. ФАБРИЧНАЯ ФУНКЦИЯ И ТОЧКА ВХОДА
# ============================================================

def create_keter_core(config: Optional[Dict[str, Any]] = None) -> KetherCore:
    """
    Фабричная функция для создания экземпляра KetherCore
    """
    return KetherCore(config)


async def main():
    """
    Основная функция запуска KetherCore
    """
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║            ISKRA-4 KETHERIC BLOCK v2.0               ║
    ║            Сефира: KETER (Венец)                     ║
    ║            Архитектура: ISKRA-4                      ║
    ╚══════════════════════════════════════════════════════╝
    """)
    
    # Создаём ядро
    core = create_keter_core()
    
    # Запускаем тестовый сценарий
    print("🚀 Запуск тестового сценария интеграции...")
    test_results = await core.run_test_scenario()
    
    # Выводим результаты
    summary = test_results.get("summary", {})
    print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"   Успешно: {'✅' if summary.get('success') else '❌'}")
    print(f"   Активных модулей: {summary.get('active_modules', '0/0')}")
    print(f"   Пройдено тестов: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}")
    
    # Детали по модулям
    activation = test_results.get("activation", {})
    if "results" in activation:
        print(f"\n🧩 СТАТУС МОДУЛЕЙ:")
        for module_name, result in activation["results"].items():
            status = result.get("status", "unknown")
            symbol = "✅" if status == "active" else "❌"
            print(f"   {symbol} {module_name}: {status}")
    
    print(f"\n🎯 Ketheric Block готов к интеграции с ISKRA-4!")
    return test_results


if __name__ == "__main__":
    # Запуск основной функции
    asyncio.run(main())
        
       
    
    
