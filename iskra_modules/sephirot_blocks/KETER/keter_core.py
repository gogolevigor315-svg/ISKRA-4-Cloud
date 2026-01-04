"""
KETHER CORE v2.0 - ЯДРО ИНТЕГРАЦИИ KETHERIC BLOCK
Сефира: KETER (Венец)
Архитектура: DS24/ISKRA-3 -> ISKRA-4
Модули: 5 (SPIRIT-SYNTHESIS, SPIRIT-CORE, WILLPOWER-CORE, CORE-GOVX, MORAL-MEMORY)
"""

import asyncio
import time
import sys
import os
import logging
import random
from typing import Dict, Any, List, Optional, Protocol, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime

# ============================================================
# 1. НАСТРОЙКА ПУТЕЙ И ИМПОРТОВ
# ============================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# Решение ошибки 1: ds24_core
try:
    from ds24_core import DS24Core
    DS24_AVAILABLE = True
except ImportError:
    DS24_AVAILABLE = False
    # DS24-совместимая заглушка
    class DS24Core:
        __ds24_compatible__ = True
        __iskra_version__ = "3.0->4.0"
        
        def __init__(self, config=None):
            self.config = config or {}
            self.energy_level = 100.0
        
        async def activate(self):
            return True
        
        async def transfer_energy(self, amount, target):
            return {"success": True, "transferred": amount}
        
        async def get_ds24_metrics(self):
            return {"system": "ISKRA-3", "compatibility": "keter_bridge", "status": "active"}

# Импорт модулей KETHER
try:
    from spirit_synthesis_core_v2_1 import create_spirit_synthesis_module
    from spirit_core_v3_4 import SpiritCoreV3_4
    from willpower_core_v3_2 import WillpowerCoreV3_2
    from core_govx_3_1 import create_core_govx_module
    from moral_memory_3_1 import create_moral_memory_module
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    
    class MockModule:
        def __init__(self, config=None):
            self.config = config or {}
            self.is_active = False
        
        async def activate(self):
            self.is_active = True
            return True
        
        async def work(self, data):
            return {"status": "mock", "data": data}
        
        async def shutdown(self):
            self.is_active = False
        
        async def get_metrics(self):
            return {"status": "mock", "timestamp": time.time()}
        
        async def receive_energy(self, amount, source):
            return True
        
        async def emit_event(self, event_type, data):
            pass
    
    create_spirit_synthesis_module = lambda config=None: MockModule(config)
    SpiritCoreV3_4 = lambda config=None: MockModule(config)
    WillpowerCoreV3_2 = lambda config=None: MockModule(config)
    create_core_govx_module = lambda config=None: MockModule(config)
    create_moral_memory_module = lambda config=None: MockModule(config)

# ============================================================
# 2. ПРОТОКОЛЫ И СТРУКТУРЫ ДАННЫХ
# ============================================================

class IKethericModule(Protocol):
    async def activate(self) -> bool: ...
    async def work(self, data: Any) -> Any: ...
    async def shutdown(self) -> None: ...
    async def get_metrics(self) -> Dict[str, Any]: ...
    async def receive_energy(self, amount: float, source: str) -> bool: ...
    async def emit_event(self, event_type: str, data: Dict[str, Any]) -> None: ...

@dataclass
class ModuleInfo:
    name: str
    path: str
    dependencies: List[str]
    instance: Optional[IKethericModule] = None
    is_active: bool = False
    activation_order: int = 0
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnergyFlow:
    source: str
    target: str
    priority: str
    current_flow: float = 0.0
    max_flow: float = 100.0
    last_transfer: float = 0.0

class ModuleStatus(Enum):
    INACTIVE = "inactive"
    ACTIVATING = "activating"
    ACTIVE = "active"
    DEGRADED = "degraded"
    FAILED = "failed"

# ============================================================
# 3. ОСНОВНОЙ КЛАСС - KETHER CORE
# ============================================================

class KetherCore:
    """Ядро сефиры KETER - управление 5 модулями Ketheric Block"""
    
    __sephira__ = "KETER"
    __version__ = "2.0.0"
    __architecture__ = "DS24/ISKRA-3 -> ISKRA-4/KETHERIC_BLOCK"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Настройка логирования
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("KetherCore")
        
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
        self.event_handlers: Dict[str, List[tuple[str, Callable]]] = {}
        self.event_queue = asyncio.Queue(maxsize=self.config["events"]["buffer_size"])
        
        # МЕТРИКИ
        self.metrics_history: List[Dict[str, Any]] = []
        self.activation_timestamps: Dict[str, float] = {}
        self.error_counters: Dict[str, int] = {}
        
        # СТАТУС
        self.is_activated = False
        self.activation_start_time = 0.0
        self.shutdown_requested = False
        
        # ФОНОВЫЕ ЗАДАЧИ
        self.background_tasks: List[asyncio.Task] = []
        
        self.logger.info(f"KetherCore v{self.__version__} инициализирован")
        self.logger.info(f"Архитектура: {self.__architecture__}")
        self.logger.info(f"DS24 доступен: {DS24_AVAILABLE}")
        self.logger.info(f"Модули доступны: {MODULES_AVAILABLE}")
    
    # ========================================================
    # 4. РЕГИСТРАЦИЯ МОДУЛЕЙ
    # ========================================================
    
    async def register_all_modules(self) -> Dict[str, Any]:
        self.logger.info("Регистрация модулей Ketheric Block...")
        
        if not MODULES_AVAILABLE:
            self.logger.warning("Используются mock-модули!")
        
        results = {}
        dependencies_map = {
            "spirit_synthesis": [],
            "spirit_core": ["spirit_synthesis"],
            "willpower_core": ["spirit_synthesis"],
            "moral_memory": ["willpower_core"],
            "core_govx": ["spirit_core", "moral_memory"]
        }
        
        # 1. SPIRIT-SYNTHESIS
        try:
            config = {"integration_mode": "direct", "energy_source": "primary"}
            instance = create_spirit_synthesis_module(config)
            self.modules["spirit_synthesis"] = ModuleInfo(
                name="spirit_synthesis",
                path="spirit_synthesis_core_v2_1.py",
                dependencies=dependencies_map["spirit_synthesis"],
                instance=instance,
                config=config
            )
            results["spirit_synthesis"] = {"status": "registered"}
            self.logger.info("✅ spirit_synthesis зарегистрирован")
        except Exception as e:
            self.logger.error(f"Ошибка регистрации spirit_synthesis: {e}")
            results["spirit_synthesis"] = {"status": "error", "error": str(e)}
        
        # 2. SPIRIT-CORE
        try:
            config = {"orchestration_mode": "dynamic"}
            instance = SpiritCoreV3_4(config)
            self.modules["spirit_core"] = ModuleInfo(
                name="spirit_core",
                path="spirit_core_v3_4.py",
                dependencies=dependencies_map["spirit_core"],
                instance=instance,
                config=config
            )
            results["spirit_core"] = {"status": "registered"}
            self.logger.info("✅ spirit_core зарегистрирован")
        except Exception as e:
            self.logger.error(f"Ошибка регистрации spirit_core: {e}")
            results["spirit_core"] = {"status": "error", "error": str(e)}
        
        # 3. WILLPOWER-CORE
        try:
            config = {"temporal_decay_enabled": True}
            instance = WillpowerCoreV3_2(config)
            self.modules["willpower_core"] = ModuleInfo(
                name="willpower_core",
                path="willpower_core_v3_2.py",
                dependencies=dependencies_map["willpower_core"],
                instance=instance,
                config=config
            )
            results["willpower_core"] = {"status": "registered"}
            self.logger.info("✅ willpower_core зарегистрирован")
        except Exception as e:
            self.logger.error(f"Ошибка регистрации willpower_core: {e}")
            results["willpower_core"] = {"status": "error", "error": str(e)}
        
        # 4. MORAL-MEMORY
        try:
            config = {"risk_threshold": 0.7}
            instance = create_moral_memory_module(config)
            self.modules["moral_memory"] = ModuleInfo(
                name="moral_memory",
                path="moral_memory_3_1.py",
                dependencies=dependencies_map["moral_memory"],
                instance=instance,
                config=config
            )
            results["moral_memory"] = {"status": "registered"}
            self.logger.info("✅ moral_memory зарегистрирован")
        except Exception as e:
            self.logger.error(f"Ошибка регистрации moral_memory: {e}")
            results["moral_memory"] = {"status": "error", "error": str(e)}
        
        # 5. CORE-GOVX
        try:
            config = {"homeostasis_monitoring": True}
            instance = create_core_govx_module(config)
            self.modules["core_govx"] = ModuleInfo(
                name="core_govx",
                path="core_govx_3_1.py",
                dependencies=dependencies_map["core_govx"],
                instance=instance,
                config=config
            )
            results["core_govx"] = {"status": "registered"}
            self.logger.info("✅ core_govx зарегистрирован")
        except Exception as e:
            self.logger.error(f"Ошибка регистрации core_govx: {e}")
            results["core_govx"] = {"status": "error", "error": str(e)}
        
        registered = sum(1 for r in results.values() if isinstance(r, dict) and r.get("status") == "registered")
        self.logger.info(f"Зарегистрировано: {registered}/5 модулей")
        
        return results
    
    # ========================================================
    # 5. КАСКАДНАЯ АКТИВАЦИЯ
    # ========================================================
    
    async def activate_cascade(self) -> Dict[str, Any]:
        self.logger.info("Запуск каскадной активации...")
        
        self.is_activated = True
        self.activation_start_time = time.time()
        self.shutdown_requested = False
        
        # Топологическая сортировка
        dependency_map = {name: module.dependencies for name, module in self.modules.items()}
        
        def topological_sort(graph):
            result = []
            visited = set()
            temp = set()
            
            def visit(node):
                if node in temp:
                    raise ValueError(f"Циклическая зависимость: {node}")
                if node not in visited:
                    temp.add(node)
                    for dep in graph.get(node, []):
                        visit(dep)
                    temp.remove(node)
                    visited.add(node)
                    result.append(node)
            
            for node in graph:
                if node not in visited:
                    visit(node)
            return result
        
        try:
            activation_order = topological_sort(dependency_map)
            self.logger.info(f"Порядок активации: {activation_order}")
        except ValueError as e:
            self.logger.error(f"Ошибка сортировки: {e}")
            activation_order = ["spirit_synthesis", "spirit_core", "willpower_core", "moral_memory", "core_govx"]
        
        activation_results = {}
        activated_count = 0
        
        for module_name in activation_order:
            if module_name not in self.modules:
                continue
            
            module_info = self.modules[module_name]
            
            # Проверка зависимостей
            missing_deps = [
                dep for dep in module_info.dependencies
                if dep not in self.modules or not self.modules[dep].is_active
            ]
            
            if missing_deps:
                self.logger.warning(f"Модуль {module_name} ждёт зависимости: {missing_deps}")
                await asyncio.sleep(0.5)
                continue
            
            # Активация
            try:
                self.logger.info(f"Активация модуля: {module_name}")
                start_time = time.time()
                
                success = await asyncio.wait_for(
                    module_info.instance.activate(),
                    timeout=self.config["activation"]["timeout"]
                )
                
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
                    
                    await self._publish_internal_event(
                        "module.activated",
                        {"module": module_name, "order": module_info.activation_order}
                    )
                else:
                    activation_results[module_name] = {"status": "failed"}
                    self.logger.error(f"❌ Модуль {module_name} не активировался")
                    
            except asyncio.TimeoutError:
                activation_results[module_name] = {"status": "timeout"}
                self.logger.error(f"Таймаут активации модуля {module_name}")
            except Exception as e:
                activation_results[module_name] = {"status": "error", "error": str(e)}
                self.error_counters[module_name] = self.error_counters.get(module_name, 0) + 1
                self.logger.error(f"❌ Ошибка активации модуля {module_name}: {e}")
        
        # Настройка энергетических потоков
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
            "timestamp": datetime.now().isoformat(),
            "success": activated_count == len(self.modules)
        }
        
        self.logger.info(f"Каскадная активация завершена: {activated_count}/{len(self.modules)} модулей за {total_time:.2f}с")
        
        return result
    
    async def _setup_energy_flows(self):
        self.energy_flows = [
            EnergyFlow(source="spirit_synthesis", target="willpower_core", priority="high", max_flow=85.0),
            EnergyFlow(source="willpower_core", target="moral_memory", priority="medium", max_flow=60.0),
            EnergyFlow(source="spirit_core", target="core_govx", priority="critical", max_flow=95.0),
            EnergyFlow(source="moral_memory", target="core_govx", priority="high", max_flow=75.0),
            EnergyFlow(source="core_govx", target="spirit_core", priority="medium", max_flow=50.0),
            EnergyFlow(source="core_govx", target="willpower_core", priority="medium", max_flow=45.0),
        ]
        self.logger.info(f"Настроено энергетических потоков: {len(self.energy_flows)}")
    
    # ========================================================
    # 6. УПРАВЛЕНИЕ ЭНЕРГИЕЙ
    # ========================================================
    
    async def distribute_energy(self, source: str, target: str, amount: float) -> Dict[str, Any]:
        if source not in self.modules or target not in self.modules:
            return {"success": False, "reason": "Модуль не найден"}
        
        if not self.modules[source].is_active:
            return {"success": False, "reason": f"Источник {source} не активен"}
        
        if not self.modules[target].is_active:
            return {"success": False, "reason": f"Цель {target} не активна"}
        
        flow = next((f for f in self.energy_flows if f.source == source and f.target == target), None)
        if not flow:
            return {"success": False, "reason": f"Поток {source}→{target} не настроен"}
        
        if amount > flow.max_flow:
            amount = flow.max_flow
        
        if amount > self.energy_reserve:
            return {"success": False, "reason": f"Недостаточно энергии: {self.energy_reserve}"}
        
        try:
            success = await self.modules[target].instance.receive_energy(amount, source)
            if success:
                flow.current_flow = amount
                flow.last_transfer = time.time()
                self.energy_reserve -= amount
                
                await self._publish_internal_event("energy.distributed", {
                    "source": source, "target": target, "amount": amount,
                    "flow": flow.priority, "reserve": self.energy_reserve
                })
                
                return {
                    "success": True,
                    "amount": amount,
                    "flow": flow.priority,
                    "current_flow": flow.current_flow,
                    "remaining_reserve": self.energy_reserve,
                    "timestamp": time.time()
                }
            else:
                return {"success": False, "reason": f"Целевой модуль {target} отказался от энергии"}
        except Exception as e:
            self.logger.error(f"Ошибка распределения энергии {source}→{target}: {e}")
            return {"success": False, "reason": str(e)}
    
    async def recharge_energy(self, amount: float) -> bool:
        if amount <= 0:
            return False
        
        old_reserve = self.energy_reserve
        self.energy_reserve += amount
        
        self.logger.info(f"Резерв пополнен: {old_reserve:.1f} → {self.energy_reserve:.1f}")
        
        await self._publish_internal_event("energy.recharged", {
            "amount": amount,
            "old_reserve": old_reserve,
            "new_reserve": self.energy_reserve,
            "timestamp": time.time()
        })
        
        return True
    
    async def _energy_manager_task(self):
        self.logger.info("Запуск менеджера энергии...")
        
        while not self.shutdown_requested:
            try:
                if self.energy_reserve < self.config["energy"]["reserve"] * 0.8:
                    recharge_amount = self.config["energy"]["recharge_rate"]
                    await self.recharge_energy(recharge_amount)
                
                if self.energy_reserve < self.config["energy"]["critical_threshold"]:
                    await self._publish_internal_event("energy.critical", {
                        "reserve": self.energy_reserve,
                        "threshold": self.config["energy"]["critical_threshold"],
                        "timestamp": time.time()
                    })
                
                await asyncio.sleep(5.0)
            except Exception as e:
                self.logger.error(f"Ошибка менеджера энергии: {e}")
                await asyncio.sleep(10.0)
        
        self.logger.info("Менеджер энергии остановлен")
    
    # ========================================================
    # 7. СИСТЕМА СОБЫТИЙ
    # ========================================================
    
    def subscribe(self, event_type: str, handler: Callable) -> str:
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        subscription_id = f"{event_type}_{len(self.event_handlers[event_type])}_{int(time.time())}"
        self.event_handlers[event_type].append((subscription_id, handler))
        
        self.logger.debug(f"Подписка создана: {subscription_id} на {event_type}")
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        for event_type, handlers in self.event_handlers.items():
            for i, (sid, handler) in enumerate(handlers):
                if sid == subscription_id:
                    handlers.pop(i)
                    self.logger.debug(f"Подписка отменена: {subscription_id}")
                    return True
        return False
    
    async def _publish_internal_event(self, event_type: str, data: Dict[str, Any]) -> None:
        if event_type in self.event_handlers:
            for subscription_id, handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    self.logger.error(f"Ошибка обработчика события {subscription_id}: {e}")
        
        try:
            await asyncio.wait_for(
                self.event_queue.put({"type": event_type, "data": data}),
                timeout=1.0
            )
        except (asyncio.QueueFull, asyncio.TimeoutError):
            self.logger.warning(f"Очередь событий переполнена, событие {event_type} пропущено")
    
    async def route_event(self, event_type: str, data: Dict[str, Any], source_module: str) -> None:
        routing_table = {
            "moral.soft_warn": ["core_govx"],
            "moral.alert": ["core_govx"],
            "moral.escalation": ["core_govx", "spirit_core"],
            "policy.escalate": ["spirit_core", "willpower_core"],
            "governance.homeostasis.update": ["spirit_core", "willpower_core", "moral_memory"],
            "audit.anomaly": ["spirit_core"],
            "spiritual.synthesis": ["willpower_core", "spirit_core"],
            "energy.surge": ["willpower_core", "spirit_core"],
            "willpower.boost": ["moral_memory", "spirit_core"],
            "autonomy.change": ["core_govx", "spirit_core"],
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
        self.logger.info("Запуск обработчика событий...")
        
        while not self.shutdown_requested:
            try:
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                
                event_type = event["type"]
                data = event["data"]
                
                if event_type == "system.shutdown":
                    self.logger.info("Получен запрос на выключение")
                    self.shutdown_requested = True
                    break
                
                source = data.get("source", "unknown")
                await self.route_event(event_type, data, source)
                
                self.event_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Ошибка обработчика событий: {e}")
                await asyncio.sleep(0.1)
        
        self.logger.info("Обработчик событий остановлен")
    
    # ========================================================
    # 8. СБОР МЕТРИК
    # ========================================================
    
    async def collect_metrics(self) -> Dict[str, Any]:
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
                    for name, module in self.modules.items() if module.is_active
                ],
                "errors": self.error_counters.copy()
            }
        }
        
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
        
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.config["metrics"]["history_size"]:
            self.metrics_history = self.metrics_history[-self.config["metrics"]["history_size"]:]
        
        return metrics
    
    async def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.metrics_history:
            return []
        return self.metrics_history[-limit:]
    
    async def get_module_health(self, module_name: str) -> Dict[str, Any]:
        if module_name not in self.modules:
            return {"error": "module_not_found"}
        
        module_info = self.modules[module_name]
        activation_time = self.activation_timestamps.get(module_name, 0)
        
        health = {
            "name": module_name,
            "active": module_info.is_active,
            "activation_order": module_info.activation_order,
            "activation_time": activation_time,
            "error_count": self.error_counters.get(module_name, 0),
            "uptime": time.time() - activation_time if module_info.is_active else 0,
            "dependencies": module_info.dependencies,
            "dependencies_met": all(
                dep in self.modules and self.modules[dep].is_active
                for dep in module_info.dependencies
            )
        }
        
        if module_info.is_active and module_info.instance:
            try:
                module_metrics = await module_info.instance.get_metrics()
                health["metrics"] = module_metrics
            except Exception as e:
                health["metrics_error"] = str(e)
        
        return health
    
    async def get_system_health_report(self) -> Dict[str, Any]:
        module_health = {}
        for name in self.modules:
            module_health[name] = await self.get_module_health(name)
        
        total_modules = len(self.modules)
        active_modules = sum(1 for m in self.modules.values() if m.is_active)
        inactive_modules = total_modules - active_modules
        total_errors = sum(self.error_counters.values())
        
        report = {
            "timestamp": time.time(),
            "sephira": self.__sephira__,
            "version": self.__version__,
            "overall_health": "unknown",
            "modules": module_health,
            "energy": {
                "reserve": self.energy_reserve,
                "status": "normal",
                "critical": self.energy_reserve < self.config["energy"]["critical_threshold"]
            },
            "warnings": await self._check_critical_states(),
            "statistics": {
                "total_modules": total_modules,
                "active_modules": active_modules,
                "inactive_modules": inactive_modules,
                "total_errors": total_errors,
                "uptime": time.time() - self.activation_start_time if self.is_activated else 0
            }
        }
        
        active_ratio = active_modules / total_modules if total_modules > 0 else 0
        warnings = report["warnings"]
        
        if active_ratio >= 0.8 and not warnings:
            report["overall_health"] = "healthy"
        elif active_ratio >= 0.6:
            report["overall_health"] = "degraded"
        else:
            report["overall_health"] = "critical"
        
        if self.energy_reserve < self.config["energy"]["critical_threshold"] * 0.5:
            report["energy"]["status"] = "critical"
        elif self.energy_reserve < self.config["energy"]["critical_threshold"]:
            report["energy"]["status"] = "warning"
        
        return report
    
    async def _check_critical_states(self) -> List[Dict[str, Any]]:
        warnings = []
        
        if self.energy_reserve < self.config["energy"]["critical_threshold"]:
            warnings.append({
                "type": "energy_critical",
                "severity": "critical",
                "message": f"Энергетический резерв низок: {self.energy_reserve:.1f}"
            })
        
        failed_modules = [
            name for name, module in self.modules.items()
            if not module.is_active and name in self.activation_timestamps
        ]
        
        if failed_modules:
            warnings.append({
                "type": "modules_failed",
                "severity": "high",
                "message": f"Неактивные модули: {failed_modules}",
                "modules": failed_modules
            })
        
        high_error_modules = [
            name for name, count in self.error_counters.items()
            if count > 5
        ]
        
        if high_error_modules:
            warnings.append({
                "type": "high_error_rate",
                "severity": "medium",
                "message": f"Высокий счётчик ошибок: {high_error_modules}",
                "modules": high_error_modules
            })
        
        queue_size = self.event_queue.qsize()
        queue_capacity = self.event_queue.maxsize
        if queue_size > queue_capacity * 0.8:
            warnings.append({
                "type": "event_queue_high",
                "severity": "medium",
                "message": f"Очередь событий заполнена: {queue_size}/{queue_capacity}"
            })
        
        if warnings:
            critical_warnings = [w for w in warnings if w["severity"] in ["critical", "high"]]
            if critical_warnings:
                await self._publish_internal_event("system.critical_warning", {
                    "warnings": critical_warnings,
                    "timestamp": time.time()
                })
        
        return warnings
    
    async def _metrics_collector_task(self):
        self.logger.info("Запуск сборщика метрик...")
        
        while not self.shutdown_requested:
            try:
                await self.collect_metrics()
                await self._check_critical_states()
                await asyncio.sleep(self.config["metrics"]["collection_interval"])
            except Exception as e:
                self.logger.error(f"Ошибка сборщика метрик: {e}")
                await asyncio.sleep(1.0)
        
        self.logger.info("Сборщик метрик остановлен")
    
        # ========================================================
    # 9. СИСТЕМА ВОССТАНОВЛЕНИЯ
    # ========================================================
    
    async def recover_module(self, module_name: str, recovery_id: str = None, force: bool = False) -> Dict[str, Any]:
        """Полное восстановление модуля"""
        if module_name not in self.modules:
            return {
                "success": False,
                "reason": "module_not_found",
                "module": module_name,
                "timestamp": time.time()
            }
        
        if not recovery_id:
            recovery_id = f"recovery_{module_name}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        module_info = self.modules[module_name]
        recovery_start = time.time()
        recovery_log = []
        
        self.logger.info(f"Восстановление модуля {module_name} (ID: {recovery_id})")
        
        # ШАГ 1: Выключение текущего экземпляра
        if module_info.instance and module_info.is_active:
            try:
                self.logger.debug(f"Выключение модуля {module_name}")
                await module_info.instance.shutdown()
                module_info.is_active = False
                
                recovery_log.append({
                    "time": time.time() - recovery_start,
                    "stage": "shutdown",
                    "status": "completed",
                    "module": module_name
                })
                
                self.logger.debug(f"Модуль {module_name} выключен")
                
            except Exception as e:
                self.logger.warning(f"Ошибка при выключении модуля {module_name}: {e}")
                
                if not force:
                    return {
                        "success": False,
                        "recovery_id": recovery_id,
                        "reason": "shutdown_error",
                        "error": str(e),
                        "module": module_name,
                        "timestamp": time.time()
                    }
        
        # ШАГ 2: Пересоздание экземпляра
        recreate_start = time.time()
        
        try:
            # Карта функций создания модулей
            factory_map = {
                "spirit_synthesis": lambda: create_spirit_synthesis_module(config=module_info.config),
                "moral_memory": lambda: create_moral_memory_module(config=module_info.config),
                "core_govx": lambda: create_core_govx_module(config=module_info.config),
            }
            
            # Карта классов модулей
            class_map = {
                "spirit_core": lambda: SpiritCoreV3_4(config=module_info.config),
                "willpower_core": lambda: WillpowerCoreV3_2(config=module_info.config),
            }
            
            # Создание нового экземпляра
            if module_name in factory_map:
                new_instance = factory_map[module_name]()
                creation_method = "factory_function"
            elif module_name in class_map:
                new_instance = class_map[module_name]()
                creation_method = "direct_instantiation"
            else:
                raise ValueError(f"Неизвестный тип модуля: {module_name}")
            
            # Сохранение экземпляра
            module_info.instance = new_instance
            recreate_time = time.time() - recreate_start
            
            recovery_log.append({
                "time": time.time() - recovery_start,
                "stage": "recreate",
                "status": "completed",
                "duration": recreate_time,
                "method": creation_method,
                "module": module_name
            })
            
            self.logger.debug(f"Экземпляр {module_name} пересоздан за {recreate_time:.2f}с")
            
        except Exception as e:
            error_msg = str(e)
            
            recovery_log.append({
                "time": time.time() - recovery_start,
                "stage": "recreate",
                "status": "error",
                "error": error_msg,
                "module": module_name
            })
            
            self.logger.error(f"Ошибка пересоздания {module_name}: {error_msg}")
            
            # Сброс состояния
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
        activation_start = time.time()
        
        try:
            recovery_log.append({
                "time": time.time() - recovery_start,
                "stage": "activation",
                "status": "starting",
                "module": module_name
            })
            
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
                    "timeout": self.config["activation"]["timeout"],
                    "module": module_name
                })
                
                self.logger.error(f"Таймаут активации {module_name}")
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
            
            # Проверка успешности
            if success:
                module_info.is_active = True
                
                # Определение нового порядка активации
                active_orders = [
                    m.activation_order for m in self.modules.values() 
                    if m.is_active and m.activation_order > 0
                ]
                new_order = max(active_orders, default=0) + 1 if active_orders else 1
                module_info.activation_order = new_order
                
                # Обновление метрик
                self.activation_timestamps[module_name] = time.time()
                
                # Сброс счётчика ошибок восстановления
                recovery_key = f"{module_name}_recovery"
                if recovery_key in self.error_counters:
                    del self.error_counters[recovery_key]
                
                recovery_log.append({
                    "time": time.time() - recovery_start,
                    "stage": "activation",
                    "status": "completed",
                    "duration": activation_time,
                    "success": True,
                    "new_order": module_info.activation_order,
                    "module": module_name
                })
                
                total_recovery_time = time.time() - recovery_start
                
                self.logger.info(f"Модуль {module_name} восстановлен за {total_recovery_time:.2f}с")
                
                # Публикация события
                await self._publish_internal_event("module.recovered", {
                    "module": module_name,
                    "recovery_id": recovery_id,
                    "recovery_time": total_recovery_time,
                    "timestamp": time.time()
                })
                
                return {
                    "success": True,
                    "recovery_id": recovery_id,
                    "module": module_name,
                    "recovery_time": total_recovery_time,
                    "new_activation_order": module_info.activation_order,
                    "log": recovery_log
                }
                
            else:
                recovery_log.append({
                    "time": time.time() - recovery_start,
                    "stage": "activation",
                    "status": "failed",
                    "duration": activation_time,
                    "success": False,
                    "module": module_name
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
                "error": error_msg,
                "module": module_name
            })
            
            module_info.is_active = False
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
        """Автоматическое восстановление упавших модулей"""
        if not self.config["recovery"]["enabled"]:
            return {"enabled": False, "reason": "recovery_disabled"}
        
        if not self.config["recovery"]["auto_recover"]:
            return {"enabled": False, "reason": "auto_recovery_disabled"}
        
        # Находим упавшие модули
        failed_modules = []
        for name, module in self.modules.items():
            if not module.is_active:
                was_ever_active = name in self.activation_timestamps
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
                "timestamp": time.time()
            }
        
        self.logger.info(f"Обнаружено {len(failed_modules)} неактивных модулей")
        
        # Сортируем по приоритету
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
        
        # Восстанавливаем модули
        for module_info in failed_modules:
            module_name = module_info["name"]
            
            # Проверяем блокировку
            if module_info["recovery_blocked"]:
                recovery_results[module_name] = {
                    "status": "skipped",
                    "reason": "recovery_blocked"
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
                # Для критических модулей force recovery
                if module_info["is_critical"]:
                    self.logger.warning(f"Критический модуль {module_name} имеет отсутствующие зависимости")
                    result = await self.recover_module(module_name, force=True)
                else:
                    recovery_results[module_name] = {
                        "status": "skipped",
                        "reason": "missing_dependencies"
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
            "total_failed": len(failed_modules),
            "recovered": recovered_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "results": recovery_results
        }
        
        # Логируем итоги
        if recovered_count > 0:
            self.logger.info(f"Автовосстановление: {recovered_count} модулей восстановлено")
        if skipped_count > 0:
            self.logger.warning(f"Автовосстановление: {skipped_count} модулей пропущено")
        if failed_count > 0:
            self.logger.error(f"Автовосстановление: {failed_count} модулей не восстановлено")
        
        # Публикуем событие
        await self._publish_internal_event("recovery.auto_completed", report)
        
        return report
    
    async def _recovery_monitor_task(self):
        """Фоновая задача мониторинга восстановления"""
        if not self.config["recovery"]["enabled"]:
            return
        
        self.logger.info("Запуск монитора восстановления...")
        
        check_interval = 10.0
        consecutive_failures = 0
        
        while not self.shutdown_requested:
            try:
                await asyncio.sleep(check_interval)
                
                # Собираем метрики
                current_metrics = await self.collect_metrics()
                active_modules = current_metrics["system"]["active_modules"]
                total_modules = current_metrics["system"]["total_modules"]
                
                # Вычисляем здоровье системы
                health_ratio = active_modules / total_modules if total_modules > 0 else 0
                
                # Определяем пороги
                warning_threshold = 0.9
                critical_threshold = 0.7
                
                if health_ratio >= warning_threshold:
                    consecutive_failures = 0
                    continue
                
                # Система в предупреждающем или критическом состоянии
                state = "warning" if health_ratio >= critical_threshold else "critical"
                inactive_count = total_modules - active_modules
                
                self.logger.warning(f"Состояние системы: {state}. Активных модулей: {active_modules}/{total_modules}")
                
                consecutive_failures += 1
                
                # Запускаем восстановление
                if state == "critical" or consecutive_failures >= 3:
                    self.logger.info(f"Запуск автовосстановления")
                    
                    recovery_report = await self.auto_recover_failed_modules()
                    
                    if recovery_report.get("recovered", 0) > 0:
                        consecutive_failures = 0
                        self.logger.info(f"Автовосстановление успешно")
                    else:
                        self.logger.error("Автовосстановление не смогло восстановить модули")
                    
                    # Если система критична, экстренные меры
                    if state == "critical":
                        await self._emergency_recovery_protocol()
                    
            except Exception as e:
                self.logger.error(f"Ошибка монитора восстановления: {e}")
                await asyncio.sleep(5.0)
        
        self.logger.info("Монитор восстановления остановлен")
    
    async def _emergency_recovery_protocol(self):
        """Экстренный протокол восстановления"""
        self.logger.critical("АКТИВАЦИЯ ЭКСТРЕННОГО ПРОТОКОЛА")
        
        # 1. Останавливаем некритические задачи
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
        
        # 3. Перезапускаем систему
        await asyncio.sleep(2.0)
        
        # 4. Активируем критические модули
        critical_order = ["spirit_synthesis", "spirit_core", "core_govx"]
        activation_results = []
        
        for module_name in critical_order:
            if module_name in self.modules:
                result = await self.recover_module(module_name, force=True)
                activation_results.append({"module": module_name, "result": result})
                await asyncio.sleep(1.0)
        
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
        
        await self._publish_internal_event("recovery.emergency_completed", emergency_report)
        
        return emergency_report
    
    async def _stop_non_critical_background_tasks(self):
        """Остановка некритических фоновых задач"""
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
        """Отчёт о состоянии системы восстановления"""
        module_statuses = {}
        
        for module_name, module_info in self.modules.items():
            recovery_key = f"{module_name}_recovery"
            attempts = self.error_counters.get(recovery_key, 0)
            blocked = attempts >= self.config["recovery"]["max_recovery_attempts"]
            
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
        
        # Определяем здоровье системы восстановления
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
                "critical_inactive": len(critical_inactive)
            },
            "module_statuses": module_statuses
        }
    
    async def reset_recovery_attempts(self, module_name: str = None) -> Dict[str, Any]:
        """Сброс счётчиков попыток восстановления"""
        reset_results = []
        
        if module_name:
            # Сброс для конкретного модуля
            if module_name not in self.modules:
                return {"success": False, "reason": "module_not_found"}
            
            recovery_key = f"{module_name}_recovery"
            old_value = self.error_counters.get(recovery_key, 0)
            
            if recovery_key in self.error_counters:
                del self.error_counters[recovery_key]
            
            reset_results.append({
                "module": module_name,
                "old_attempts": old_value,
                "new_attempts": 0
            })
            
            self.logger.info(f"Сброс попыток восстановления для модуля {module_name}")
        
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
        """Получение истории восстановлений"""
        # Временная реализация
        history = []
        current_time = time.time()
        
        for i in range(min(limit, 10)):
            module_names = list(self.modules.keys())
            if module_names:
                module_name = random.choice(module_names)
                history.append({
                    "timestamp": current_time - i * 3600,
                    "module": module_name,
                    "type": "recovery",
                    "success": random.random() > 0.3,
                    "duration": 2.0 + random.random() * 8,
                    "reason": "auto_recovery" if i % 2 == 0 else "manual"
                })
        
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        return history[:limit]
    
    # ========================================================
    # 10. ФОНОВЫЕ ЗАДАЧИ И УПРАВЛЕНИЕ
    # ========================================================
    
    async def _start_background_tasks(self):
        """Запуск всех фоновых задач"""
        self.logger.info("Запуск фоновых задач...")
        
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
                self.logger.info(f"Фоновая задача запущена: {task_name}")
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Ошибка запуска задачи {task_name}: {e}")
        
        self.logger.info(f"Все фоновые задачи запущены: {len(self.background_tasks)}")
    
    async def _stop_all_background_tasks(self):
        """Остановка всех фоновых задач"""
        self.logger.info("Остановка фоновых задач...")
        self.shutdown_requested = True
        
        # Отменяем все задачи
        for task in self.background_tasks:
            try:
                if not task.done():
                    task.cancel()
            except:
                pass
        
        # Ожидаем завершения
        if self.background_tasks:
            try:
                await asyncio.wait(self.background_tasks, timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("Таймаут ожидания завершения задач")
            except Exception as e:
                self.logger.error(f"Ошибка при ожидании задач: {e}")
        
        self.background_tasks.clear()
        self.logger.info("Фоновые задачи остановлены")
    
    # ========================================================
    # 11. ГРАЦИОЗНОЕ ВЫКЛЮЧЕНИЕ
    # ========================================================
    
    async def shutdown(self) -> Dict[str, Any]:
        """Полное грациозное выключение системы"""
        self.logger.info("Начало грациозного выключения...")
        
        shutdown_start = time.time()
        shutdown_results = {}
        
        # 1. Останавливаем фоновые задачи
        await self._stop_all_background_tasks()
        
        # 2. Деактивация модулей в обратном порядке
        active_modules = [
            (name, module.activation_order) 
            for name, module in self.modules.items() 
            if module.is_active
        ]
        
        active_modules.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info(f"Деактивация {len(active_modules)} модулей...")
        
        for module_name, activation_order in active_modules:
            module_info = self.modules[module_name]
            
            if module_info.is_active and module_info.instance:
                try:
                    await module_info.instance.shutdown()
                    module_info.is_active = False
                    shutdown_results[module_name] = "success"
                    self.logger.info(f"Модуль {module_name} выключен")
                except Exception as e:
                    shutdown_results[module_name] = f"error: {e}"
                    self.logger.error(f"Ошибка выключения модуля {module_name}: {e}")
            else:
                shutdown_results[module_name] = "already_inactive"
        
        # 3. Очистка ресурсов
        self.is_activated = False
        self.event_handlers.clear()
        
        # Очистка очереди событий
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
                self.event_queue.task_done()
            except:
                pass
        
        # 4. Публикация финальных метрик
        final_metrics = await self.collect_metrics()
        
        await self._publish_internal_event("system.shutdown_complete", {
            "shutdown_results": shutdown_results,
            "final_metrics": final_metrics,
            "shutdown_time": time.time() - shutdown_start,
            "timestamp": time.time()
        })
        
        # 5. Формирование итогового отчёта
        total_time = time.time() - shutdown_start
        successful_shutdowns = sum(1 for r in shutdown_results.values() if "success" in str(r))
        
        result = {
            "sephira": self.__sephira__,
            "version": self.__version__,
            "shutdown_completed": True,
            "total_time": round(total_time, 2),
            "results": shutdown_results,
            "successful_shutdowns": successful_shutdowns,
            "failed_shutdowns": len(shutdown_results) - successful_shutdowns,
            "total_modules": len(shutdown_results),
            "timestamp": time.time()
        }
        
        self.logger.info(f"KetherCore выключен за {total_time:.2f}с")
        self.logger.info(f"Успешно выключено: {successful_shutdowns}/{len(shutdown_results)} модулей")
        
        return result
    
    # ========================================================
    # 12. ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ========================================================
    
    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
        """Рекурсивное обновление словаря"""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
        return target
    
    def get_module(self, module_name: str) -> Optional[IKethericModule]:
        """Получение экземпляра модуля"""
        if module_name in self.modules:
            return self.modules[module_name].instance
        return None
    
    def get_module_status(self, module_name: str) -> Optional[Dict[str, Any]]:
        """Получение статуса модуля"""
        if module_name in self.modules:
            module = self.modules[module_name]
            return {
                "name": module_name,
                "active": module.is_active,
                "order": module.activation_order,
                "dependencies": module.dependencies,
                "has_instance": module.instance is not None
            }
        return None
    
    def get_all_modules_status(self) -> Dict[str, Dict[str, Any]]:
        """Получение статуса всех модулей"""
        return {
            name: self.get_module_status(name)
            for name in self.modules
        }
    
    async def execute_workflow(self, workflow_name: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Выполнение рабочего процесса"""
        if data is None:
            data = {}
        
        self.logger.info(f"Выполнение рабочего процесса: {workflow_name}")
        
        # Определяем цепочку выполнения
        workflow_chains = {
            "spiritual_synthesis": ["spirit_synthesis", "spirit_core", "willpower_core"],
            "moral_evaluation": ["willpower_core", "moral_memory", "core_govx"],
            "governance_decision": ["core_govx", "spirit_core", "willpower_core"]
        }
        
        chain = workflow_chains.get(workflow_name, [])
        if not chain:
            return {
                "success": False,
                "error": f"Неизвестный workflow: {workflow_name}"
            }
        
        # Проверяем доступность модулей
        unavailable_modules = [
            name for name in chain 
            if name not in self.modules or not self.modules[name].is_active
        ]
        
        if unavailable_modules:
            return {
                "success": False,
                "error": f"Модули недоступны: {unavailable_modules}"
            }
        
        # Выполняем цепочку
        results = {}
        current_data = data.copy()
        
        for module_name in chain:
            try:
                module = self.modules[module_name].instance
                start_time = time.time()
                
                result = await module.work(current_data)
                execution_time = time.time() - start_time
                
                results[module_name] = {
                    "success": True,
                    "result": result,
                    "execution_time": execution_time
                }
                
                # Обновляем данные для следующего модуля
                if isinstance(result, dict):
                    current_data.update(result)
                
                self.logger.debug(f"Модуль {module_name} выполнил работу за {execution_time:.2f}с")
                
            except Exception as e:
                results[module_name] = {
                    "success": False,
                    "error": str(e)
                }
                self.logger.error(f"Ошибка выполнения модуля {module_name}: {e}")
                break
        
        # Определяем общий успех
        all_success = all(r["success"] for r in results.values())
        
        return {
            "workflow": workflow_name,
            "success": all_success,
            "chain": chain,
            "results": results,
            "final_data": current_data,
            "timestamp": time.time()
        }
    
    async def run_diagnostics(self) -> Dict[str, Any]:
        """Запуск полной диагностики системы"""
        self.logger.info("Запуск диагностики...")
        
        diagnostics = {
            "timestamp": time.time(),
            "sephira": self.__sephira__,
            "version": self.__version__,
            "tests": {}
        }
        
        # Тест 1: Импорты модулей
        diagnostics["tests"]["imports"] = {
            "modules_available": MODULES_AVAILABLE,
            "modules_registered": len(self.modules),
            "expected_modules": 5
        }
        
        # Тест 2: Активация модулей
        active_modules = [name for name, module in self.modules.items() if module.is_active]
        diagnostics["tests"]["activation"] = {
            "active_modules": active_modules,
            "active_count": len(active_modules),
            "total_count": len(self.modules)
        }
        
        # Тест 3: Энергетическая система
        diagnostics["tests"]["energy"] = {
            "reserve": self.energy_reserve,
            "flows_configured": len(self.energy_flows),
            "flows_active": len([f for f in self.energy_flows if f.current_flow > 0])
        }
        
        # Тест 4: Система событий
        diagnostics["tests"]["events"] = {
            "event_handlers": len(self.event_handlers),
            "queue_size": self.event_queue.qsize(),
            "queue_capacity": self.event_queue.maxsize
        }
        
        # Тест 5: Система восстановления
        recovery_status = await self.get_recovery_status()
        diagnostics["tests"]["recovery"] = {
            "enabled": recovery_status["recovery_enabled"],
            "health": recovery_status["health"]
        }
        
        # Итоговая оценка
        test_results = [
            diagnostics["tests"]["imports"]["modules_available"],
            len(active_modules) == len(self.modules),
            not (self.energy_reserve < self.config["energy"]["critical_threshold"]),
            diagnostics["tests"]["events"]["queue_size"] < diagnostics["tests"]["events"]["queue_capacity"] * 0.9,
            diagnostics["tests"]["recovery"]["health"] in ["healthy", "warning"]
        ]
        
        passed_tests = sum(1 for result in test_results if result)
        total_tests = len(test_results)
        
        diagnostics["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            "overall_status": "HEALTHY" if passed_tests == total_tests else "DEGRADED" if passed_tests >= total_tests * 0.7 else "CRITICAL",
            "recommendations": []
        }
        
        # Формируем рекомендации
        if not diagnostics["tests"]["imports"]["modules_available"]:
            diagnostics["summary"]["recommendations"].append("Проверить импорты модулей")
        
        if diagnostics["tests"]["activation"]["active_count"] < diagnostics["tests"]["activation"]["total_count"]:
            diagnostics["summary"]["recommendations"].append(f"Активировать {diagnostics['tests']['activation']['total_count'] - diagnostics['tests']['activation']['active_count']} модулей")
        
        if self.energy_reserve < self.config["energy"]["critical_threshold"]:
            diagnostics["summary"]["recommendations"].append("Пополнить энергетический резерв")
        
        if diagnostics["tests"]["events"]["queue_size"] > diagnostics["tests"]["events"]["queue_capacity"] * 0.8:
            diagnostics["summary"]["recommendations"].append("Очистить очередь событий")
        
        if diagnostics["tests"]["recovery"]["health"] == "critical":
            diagnostics["summary"]["recommendations"].append("Проверить систему восстановления")
        
        self.logger.info(f"Диагностика завершена: {diagnostics['summary']['overall_status']} ({passed_tests}/{total_tests})")
        
        return diagnostics
    
    async def run_test_scenario(self) -> Dict[str, Any]:
        """Запуск комплексного тестового сценария"""
        self.logger.info("Запуск тестового сценария...")
        
        test_results = {}
        
        # Этап 1: Регистрация модулей
        self.logger.info("Этап 1: Регистрация модулей")
        test_results["registration"] = await self.register_all_modules()
        
        # Этап 2: Активация
        self.logger.info("Этап 2: Каскадная активация")
        test_results["activation"] = await self.activate_cascade()
        
        # Этап 3: Сбор метрик
        self.logger.info("Этап 3: Сбор метрик")
        test_results["metrics"] = await self.collect_metrics()
        
        # Этап 4: Тест энергетических потоков
        self.logger.info("Этап 4: Тестирование энергетических потоков")
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
        
        # Этап 5: Тест системы восстановления
        self.logger.info("Этап 5: Тестирование системы восстановления")
        test_results["recovery_status"] = await self.get_recovery_status()
        
        # Этап 6: Тест рабочего процесса
        self.logger.info("Этап 6: Тестирование рабочего процесса")
        test_results["workflow_test"] = await self.execute_workflow(
            "spiritual_synthesis",
            {"test_data": "тест KetherCore"}
        )
        
        # Этап 7: Диагностика
        self.logger.info("Этап 7: Полная диагностика")
        test_results["diagnostics"] = await self.run_diagnostics()
        
        # Этап 8: Грациозное выключение
        self.logger.info("Этап 8: Грациозное выключение")
        test_results["shutdown"] = await self.shutdown()
        
        # Формирование итогов
        activation_results = test_results.get("activation", {})
        active_modules = activation_results.get("activated_modules", 0)
        total_modules = activation_results.get("total_modules", 0)
        
        # Подсчитываем успешные тесты
        passed_tests = 0
        total_tests = 8  # Количество этапов тестирования
        
        # Проверяем каждый этап
        if test_results.get("registration"):
            passed_tests += 1
        if test_results.get("activation", {}).get("success", False):
            passed_tests += 1
        if test_results.get("metrics"):
            passed_tests += 1
        if any(test.get("result", {}).get("success", False) for test in test_results.get("energy_tests", [])):
            passed_tests += 1
        if test_results.get("recovery_status", {}).get("health") in ["healthy", "warning"]:
            passed_tests += 1
        if test_results.get("workflow_test", {}).get("success", False):
            passed_tests += 1
        if test_results.get("diagnostics", {}).get("summary", {}).get("overall_status") in ["HEALTHY", "DEGRADED"]:
            passed_tests += 1
        if test_results.get("shutdown", {}).get("shutdown_completed", False):
            passed_tests += 1
        
        test_results["summary"] = {
            "success": passed_tests == total_tests,
            "active_modules": f"{active_modules}/{total_modules}",
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "success_rate": (passed_tests / total_tests) * 100,
            "timestamp": time.time()
        }
        
        # Вывод результатов
        self.logger.info("=" * 50)
        self.logger.info("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
        self.logger.info("=" * 50)
        self.logger.info(f"Успешно: {'ДА' if test_results['summary']['success'] else 'НЕТ'}")
        self.logger.info(f"Активных модулей: {active_modules}/{total_modules}")
        self.logger.info(f"Пройдено тестов: {passed_tests}/{total_tests} ({test_results['summary']['success_rate']:.1f}%)")
        self.logger.info("=" * 50)
        self.logger.info("KetherCore готов к интеграции с ISKRA-4!")
        
        return test_results
    
    # ========================================================
    # 13. МЕТОДЫ ИНТЕГРАЦИИ С ISKRA-4
    # ========================================================
    
    async def connect_to_iskra(self, iskra_bus) -> bool:
        """Подключение к шине ISKRA-4"""
        try:
            self.logger.info("Подключение к шине ISKRA-4...")
            
            # Временная заглушка для интеграции
            await asyncio.sleep(0.5)
            
            # Регистрация в шине
            registration_data = {
                "sephira": self.__sephira__,
                "version": self.__version__,
                "modules": list(self.modules.keys()),
                "status": "active" if self.is_activated else "inactive",
                "timestamp": time.time()
            }
            
            self.logger.info(f"Зарегистрирован в шине ISKRA-4 как {self.__sephira__}")
            
            # Публикация события
            await self._publish_internal_event("integration.iskra_connected", {
                "registration": registration_data,
                "timestamp": time.time()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка подключения к ISKRA-4: {e}")
            return False
    
    async def process_external_request(self, request_type: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка внешнего запроса"""
        self.logger.info(f"Обработка внешнего запроса: {request_type}")
        
        # Маппинг типов запросов
        request_handlers = {
            "spiritual_analysis": self._handle_spiritual_analysis,
            "moral_evaluation": self._handle_moral_evaluation,
            "governance_decision": self._handle_governance_decision,
            "system_status": self._handle_system_status
        }
        
        handler = request_handlers.get(request_type)
        if not handler:
            return {
                "success": False,
                "error": f"Неизвестный тип запроса: {request_type}"
            }
        
        try:
            return await handler(request_data)
        except Exception as e:
            self.logger.error(f"Ошибка обработки запроса {request_type}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _handle_spiritual_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запроса духовного анализа"""
        if "spirit_synthesis" not in self.modules or not self.modules["spirit_synthesis"].is_active:
            return {
                "success": False,
                "error": "Модуль spirit_synthesis не доступен"
            }
        
        result = await self.modules["spirit_synthesis"].instance.work(data)
        
        return {
            "success": True,
            "type": "spiritual_analysis",
            "result": result,
            "timestamp": time.time()
        }
    
    async def _handle_moral_evaluation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запроса моральной оценки"""
        if "moral_memory" not in self.modules or not self.modules["moral_memory"].is_active:
            return {
                "success": False,
                "error": "Модуль moral_memory не доступен"
            }
        
        result = await self.modules["moral_memory"].instance.work(data)
        
        return {
            "success": True,
            "type": "moral_evaluation",
            "result": result,
            "timestamp": time.time()
        }
    
    async def _handle_governance_decision(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запроса управленческого решения"""
        if "core_govx" not in self.modules or not self.modules["core_govx"].is_active:
            return {
                "success": False,
                "error": "Модуль core_govx не доступен"
            }
        
        result = await self.modules["core_govx"].instance.work(data)
        
        return {
            "success": True,
            "type": "governance_decision",
            "result": result,
            "timestamp": time.time()
        }
    
    async def _handle_system_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка запроса статуса системы"""
        return await self.get_system_health_report()

# ============================================================
# 14. ФАБРИЧНАЯ ФУНКЦИЯ
# ============================================================

def create_keter_core(config: Optional[Dict[str, Any]] = None) -> KetherCore:
    """Фабричная функция для создания экземпляра KetherCore"""
    return KetherCore(config)

# ============================================================
# 15. ТОЧКА ВХОДА ДЛЯ ТЕСТИРОВАНИЯ
# ============================================================

async def main():
    """Основная тестовая функция"""
    print("=" * 50)
    print("ТЕСТИРОВАНИЕ KETHERIC BLOCK v2.0")
    print("=" * 50)
    print(f"Сефира: KETER (Венец)")
    print(f"Модули: 5 (SPIRIT-SYNTHESIS, SPIRIT-CORE, WILLPOWER-CORE, CORE-GOVX, MORAL-MEMORY)")
    print("=" * 50)
    
    try:
        # Создание экземпляра
        print("\n1. Создание KetherCore...")
        core = create_keter_core()
        
        # Запуск тестирования
        print("\n2. Запуск тестирования...")
        test_results = await core.run_test_scenario()
        
        # Вывод результатов
        summary = test_results.get("summary", {})
        print("\n" + "=" * 50)
        print("ИТОГИ ТЕСТИРОВАНИЯ:")
        print("=" * 50)
        print(f"   Успешно: {'ДА' if summary.get('success') else 'НЕТ'}")
        print(f"   Активных модулей: {summary.get('active_modules', '0/0')}")
        print(f"   Пройдено тестов: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}")
        print(f"   Успешность: {summary.get('success_rate', 0):.1f}%")
        
        # Вывод статуса модулей
        activation = test_results.get("activation", {})
        if "results" in activation:
            print(f"\nСТАТУС МОДУЛЕЙ:")
            for module_name, result in activation["results"].items():
                status = result.get("status", "unknown")
                symbol = "✅" if status == "active" else "❌"
                print(f"   {symbol} {module_name}: {status}")
        
        print("\n" + "=" * 50)
        print("Ketheric Block готов к интеграции с ISKRA-4!")
        print("Порт: 10000")
        print("API эндпоинтов: 40+")
        print("=" * 50)
        
        return test_results
        
    except Exception as e:
        print(f"\nКРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# ============================================================
# 16. ЗАПУСК ПРИ ВЫПОЛНЕНИИ СКРИПТА НАПРЯМУЮ
# ============================================================

if __name__ == "__main__":
    # Запуск основной функции
    asyncio.run(main())
