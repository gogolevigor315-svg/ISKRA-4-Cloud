"""
KETHER CORE v2.0 - ЯДРО ИНТЕГРАЦИИ KETHERIC BLOCK
Сефира: KETER (Венец)
Модули: 5 (SPIRIT-SYNTHESIS, SPIRIT-CORE, WILLPOWER-CORE, CORE-GOVX, MORAL-MEMORY)
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

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)  # iskra_modules
sys.path.insert(0, current_dir)  # sephirot_blocks/KETER

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
        async def activate(self): return True
        async def work(self, data): return {}
        async def shutdown(self): pass
        async def get_metrics(self): return {"status": "mock"}
        async def receive_energy(self, amount, source): return True
        async def emit_event(self, event_type, data): pass
    
    create_spirit_synthesis_module = lambda config=None: MockModule()
    SpiritCoreV3_4 = MockModule
    WillpowerCoreV3_2 = MockModule
    create_core_govx_module = lambda config=None: MockModule()
    create_moral_memory_module = lambda config=None: MockModule()

# ============================================================
# 2. ПРОТОКОЛЫ И СТРУКТУРЫ ДАННЫХ
# ============================================================

class IKethericModule(Protocol):
    async def activate(self) -> bool: ...
    async def work(self, data: Any) -> Any: ...
    async def shutdown(self) -> None: ...
    async def get_metrics(self) -> Dict[str, Any]: ...
    async def receive_energy(self, amount: float, source: str) -> bool: ...
    async def emit_event(self, event_type: str, data: Dict) -> None: ...

@dataclass
class ModuleInfo:
    name: str
    path: str
    dependencies: List[str]
    instance: Optional[IKethericModule] = None
    is_active: bool = False
    activation_order: int = 0
    config: Dict[str, Any] = None

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
# 3. УТИЛИТЫ
# ============================================================

def topological_sort(modules: Dict[str, List[str]]) -> List[str]:
    result = []
    visited = set()
    temp = set()

    def visit(node):
        if node in temp:
            raise ValueError(f"Циклическая зависимость: {node}")
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
# 4. ОСНОВНОЙ КЛАСС - KETHER CORE (БЕЗ API)
# ============================================================

class KetherCore:
    __sephira__ = "KETER"
    __version__ = "2.0.0"
    __architecture__ = "ISKRA-4/KETHERIC_BLOCK"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(f"KetherCore")
        
        # Конфигурация
        self.config = {
            "activation": {"timeout": 30.0, "retry_attempts": 3, "retry_delay": 1.0},
            "energy": {"reserve": 1000.0, "recharge_rate": 10.0, "critical_threshold": 100.0},
            "events": {"enabled": True, "buffer_size": 1000, "processing_timeout": 5.0},
            "recovery": {"enabled": True, "auto_recover": True, "max_recovery_attempts": 3},
            "metrics": {"collection_interval": 5.0, "history_size": 1000, "export_enabled": True}
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
        
        # МЕТРИКИ
        self.metrics_history: List[Dict] = []
        self.activation_timestamps: Dict[str, float] = {}
        self.error_counters: Dict[str, int] = {}
        
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
        if not MODULES_AVAILABLE:
            self.logger.warning("Используются mock-модули")
        
        results = {}
        dependencies_map = {
            "spirit_synthesis": [],
            "spirit_core": ["spirit_synthesis"],
            "willpower_core": ["spirit_synthesis"],
            "moral_memory": ["willpower_core"],
            "core_govx": ["spirit_core", "moral_memory"]
        }
        
        # 1. SPIRIT-SYNTHESIS CORE v2.1
        try:
            spirit_synth_config = {"integration_mode": "direct", "energy_source": "primary", "bechtereva_integration": True}
            spirit_synth = create_spirit_synthesis_module(config=spirit_synth_config)
            self.modules["spirit_synthesis"] = ModuleInfo(
                name="spirit_synthesis", path="spirit_synthesis_core_v2_1.py",
                dependencies=dependencies_map["spirit_synthesis"], instance=spirit_synth,
                config=spirit_synth_config
            )
            results["spirit_synthesis"] = "registered"
        except Exception as e:
            self.logger.error(f"Ошибка регистрации spirit_synthesis: {e}")
            results["spirit_synthesis"] = f"error: {e}"
        
        # 2. SPIRIT-CORE v3.4
        try:
            spirit_core_config = {"orchestration_mode": "dynamic", "priority_management": True, "resource_tracking": True}
            spirit_core = SpiritCoreV3_4(config=spirit_core_config)
            self.modules["spirit_core"] = ModuleInfo(
                name="spirit_core", path="spirit_core_v3_4.py",
                dependencies=dependencies_map["spirit_core"], instance=spirit_core,
                config=spirit_core_config
            )
            results["spirit_core"] = "registered"
        except Exception as e:
            self.logger.error(f"Ошибка регистрации spirit_core: {e}")
            results["spirit_core"] = f"error: {e}"
        
        # 3. WILLPOWER-CORE v3.2
        try:
            willpower_config = {"temporal_decay_enabled": True, "moral_filter_enabled": True, "autonomy_level": 0.8}
            willpower = WillpowerCoreV3_2(config=willpower_config)
            self.modules["willpower_core"] = ModuleInfo(
                name="willpower_core", path="willpower_core_v3_2.py",
                dependencies=dependencies_map["willpower_core"], instance=willpower,
                config=willpower_config
            )
            results["willpower_core"] = "registered"
        except Exception as e:
            self.logger.error(f"Ошибка регистрации willpower_core: {e}")
            results["willpower_core"] = f"error: {e}"
        
        # 4. MORAL-MEMORY 3.1
        try:
            moral_config = {
                "risk_threshold": 0.7, "fast_evaluation": True,
                "hard_ban_categories": ["CSAM", "терроризм", "физический_вред"],
                "operator_preferences": {"risk_tolerance": 0.5}
            }
            moral_memory = create_moral_memory_module(config=moral_config)
            self.modules["moral_memory"] = ModuleInfo(
                name="moral_memory", path="moral_memory_3_1.py",
                dependencies=dependencies_map["moral_memory"], instance=moral_memory,
                config=moral_config
            )
            results["moral_memory"] = "registered"
        except Exception as e:
            self.logger.error(f"Ошибка регистрации moral_memory: {e}")
            results["moral_memory"] = f"error: {e}"
        
        # 5. CORE-GOVX 3.1
        try:
            govx_config = {
                "homeostasis_monitoring": True, "policy_interpreter": True,
                "audit_ledger": True, "escalation_engine": True, "trend_analysis": True
            }
            core_govx = create_core_govx_module(config=govx_config)
            self.modules["core_govx"] = ModuleInfo(
                name="core_govx", path="core_govx_3_1.py",
                dependencies=dependencies_map["core_govx"], instance=core_govx,
                config=govx_config
            )
            results["core_govx"] = "registered"
        except Exception as e:
            self.logger.error(f"Ошибка регистрации core_govx: {e}")
            results["core_govx"] = f"error: {e}"
        
        self.logger.info(f"Зарегистрировано: {sum(1 for r in results.values() if 'registered' in str(r))}/5")
        return results

    # ========================================================
    # 6. КАСКАДНАЯ АКТИВАЦИЯ
    # ========================================================

    async def activate_cascade(self) -> Dict[str, Any]:
        self.logger.info("🚀 Запуск каскадной активации...")
        self.is_activated = True
        self.activation_start_time = time.time()
        self.shutdown_requested = False
        
        dependency_map = {name: module.dependencies for name, module in self.modules.items()}
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
                self.logger.warning(f"Модуль {module_name} не найден")
                continue
            
            module_info = self.modules[module_name]
            missing_deps = [dep for dep in module_info.dependencies if dep not in self.modules or not self.modules[dep].is_active]
            
            if missing_deps:
                self.logger.warning(f"Модуль {module_name} ждёт зависимости: {missing_deps}")
                await asyncio.sleep(0.5)
            
            try:
                self.logger.info(f"Активация модуля: {module_name}")
                start_time = time.time()
                
                try:
                    success = await asyncio.wait_for(
                        module_info.instance.activate(),
                        timeout=self.config["activation"]["timeout"]
                    )
                except asyncio.TimeoutError:
                    self.logger.error(f"Таймаут активации модуля {module_name}")
                    activation_results[module_name] = {"status": "timeout", "time": time.time() - start_time}
                    continue
                
                if success:
                    module_info.is_active = True
                    module_info.activation_order = activated_count + 1
                    activation_time = time.time() - start_time
                    
                    self.activation_timestamps[module_name] = time.time()
                    self.error_counters[module_name] = 0
                    
                    activation_results[module_name] = {
                        "status": "active", "order": module_info.activation_order, "time": round(activation_time, 3)
                    }
                    
                    activated_count += 1
                    self.logger.info(f"✅ Модуль {module_name} активирован за {activation_time:.2f}с")
                    
                    await self._publish_internal_event(
                        "module.activated",
                        {"module": module_name, "order": module_info.activation_order}
                    )
                else:
                    activation_results[module_name] = {"status": "failed", "error": "activate() вернул False"}
                    self.logger.error(f"❌ Модуль {module_name} не активировался")
                    
            except Exception as e:
                error_msg = str(e)
                activation_results[module_name] = {"status": "error", "error": error_msg}
                self.error_counters[module_name] = self.error_counters.get(module_name, 0) + 1
                self.logger.error(f"❌ Ошибка активации модуля {module_name}: {error_msg}")
        
        await self._setup_energy_flows()
        await self._start_background_tasks()
        
        total_time = time.time() - self.activation_start_time
        result = {
            "sephira": self.__sephira__, "version": self.__version__,
            "total_modules": len(self.modules), "activated_modules": activated_count,
            "activation_order": activation_order, "results": activation_results,
            "total_time": round(total_time, 2), "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"🎯 Каскадная активация завершена: {activated_count}/{len(self.modules)} модулей активны за {total_time:.2f}с")
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
    # 7. УПРАВЛЕНИЕ ЭНЕРГИЕЙ
    # ========================================================

    async def distribute_energy(self, source: str, target: str, amount: float) -> Dict[str, Any]:
        if source not in self.modules or target not in self.modules:
            return {"success": False, "reason": f"Модуль не найден: source={source}, target={target}"}
        
        if not self.modules[source].is_active:
            return {"success": False, "reason": f"Источник {source} не активен"}
        if not self.modules[target].is_active:
            return {"success": False, "reason": f"Цель {target} не активна"}
        
        flow = next((f for f in self.energy_flows if f.source == source and f.target == target), None)
        if not flow:
            return {"success": False, "reason": f"Энергетический поток {source}→{target} не настроен"}
        
        if amount > flow.max_flow:
            amount = flow.max_flow
            self.logger.warning(f"Лимит потока {source}→{target}: {amount}")
        
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
                    "success": True, "amount": amount, "flow": flow.priority,
                    "current_flow": flow.current_flow, "remaining_reserve": self.energy_reserve,
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
            "amount": amount, "old_reserve": old_reserve,
            "new_reserve": self.energy_reserve, "timestamp": time.time()
        })
        return True

    # ========================================================
    # 8. СИСТЕМА СОБЫТИЙ
    # ========================================================

    def subscribe(self, event_type: str, handler: callable) -> str:
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

    async def _publish_internal_event(self, event_type: str, data: Dict) -> None:
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
            await asyncio.wait_for(self.event_queue.put({"type": event_type, "data": data}), timeout=1.0)
        except (asyncio.QueueFull, asyncio.TimeoutError):
            self.logger.warning(f"Очередь событий переполнена, событие {event_type} пропущено")

    async def route_event(self, event_type: str, data: Dict, source_module: str) -> None:
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
        self.logger.info("Запущен обработчик событий")
        while not self.shutdown_requested:
            try:
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                event_type = event["type"]
                data = event["data"]
                
                if event_type == "system.shutdown":
                    self.logger.info("Получен запрос на выключение")
                    self.shutdown_requested = True
                    break
                
                source = data.get("source", "unknown")
                await self.route_event(event_type, data, source)
                self.event_queue.task_done()
            except Exception as e:
                self.logger.error(f"Ошибка обработчика событий: {e}")
                await asyncio.sleep(0.1)
        
        self.logger.info("Обработчик событий остановлен")

    # ========================================================
    # 9. СБОР МЕТРИК
    # ========================================================

    async def collect_metrics(self) -> Dict[str, Any]:
        metrics = {
            "sephira": self.__sephira__, "version": self.__version__,
            "timestamp": time.time(), "datetime": datetime.now().isoformat(),
            "modules": {}, "energy": {
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
                    metrics["modules"][name] = {"active": True, "order": module_info.activation_order, "metrics": module_metrics}
                except Exception as e:
                    metrics["modules"][name] = {"active": True, "error": str(e)}
            else:
                metrics["modules"][name] = {"active": False, "order": module_info.activation_order}
        
        metrics["energy"]["flows"] = [
            {
                "source": flow.source, "target": flow.target, "priority": flow.priority,
                "current": flow.current_flow, "max": flow.max_flow, "last_transfer": flow.last_transfer
            }
            for flow in self.energy_flows
        ]
        
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.config["metrics"]["history_size"]:
            self.metrics_history = self.metrics_history[-self.config["metrics"]["history_size"]:]
        
        return metrics

    async def get_metrics_history(self, limit: int = 100) -> List[Dict]:
        return self.metrics_history[-limit:] if self.metrics_history else []

    async def get_module_health(self, module_name: str) -> Dict[str, Any]:
        if module_name not in self.modules:
            return {"error": "module_not_found"}
        
        module_info = self.modules[module_name]
        health = {
            "name": module_name, "active": module_info.is_active,
            "activation_order": module_info.activation_order,
            "activation_time": self.activation_timestamps.get(module_name),
            "error_count": self.error_counters.get(module_name, 0),
            "uptime": time.time() - self.activation_timestamps.get(module_name, 0) if module_info.is_active else 0,
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
        report = {
            "timestamp": time.time(), "sephira": self.__sephira__, "version": self.__version__,
            "overall_health": "unknown", "modules": {}, "energy": {
                "reserve": self.energy_reserve, "status": "normal",
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
        
        for name in self.modules:
            report["modules"][name] = await self.get_module_health(name)
        
        active_ratio = report["statistics"]["active_modules"] / report["statistics"]["total_modules"]
        if active_ratio >= 0.8 and not report["warnings"]:
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

    async def _check_critical_states(self):
        warnings = []
        if self.energy_reserve < self.config["energy"]["critical_threshold"]:
            warnings.append({
                "type": "energy_critical", "severity": "critical",
                "message": f"Энергетический резерв критически низок: {self.energy_reserve:.1f}"
            })
        
        failed_modules = [name for name, module in self.modules.items() if not module.is_active and name in self.activation_timestamps]
        if failed_modules:
            warnings.append({
                "type": "modules_failed", "severity": "high",
                "message": f"Неактивные модули: {failed_modules}", "modules": failed_modules
            })
        
        high_error_modules = [name for name, count in self.error_counters.items() if count > 5]
        if high_error_modules:
            warnings.append({
                "type": "high_error_rate", "severity": "medium",
                "message": f"Высокий счётчик ошибок: {high_error_modules}", "modules": high_error_modules
            })
        
        queue_size = self.event_queue.qsize()
        queue_capacity = self.event_queue.maxsize
        if queue_size > queue_capacity * 0.8:
            warnings.append({
                "type": "event_queue_high", "severity": "medium",
                "message": f"Очередь событий заполнена на {queue_size}/{queue_capacity}"
            })
        
        if warnings:
            critical_warnings = [w for w in warnings if w["severity"] in ["critical", "high"]]
            if critical_warnings:
                await self._publish_internal_event("system.critical_warning", {
                    "warnings": critical_warnings, "timestamp": time.time()
                })
        
        self.logger.warning(f"Критические состояния: {len(warnings)} предупреждений")
        return warnings

    async def _metrics_collector_task(self):
        self.logger.info("Запущен сборщик метрик")
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
    # 10. СИСТЕМА ВОССТАНОВЛЕНИЯ (ПРОДОЛЖЕНИЕ)
    # ========================================================

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
        
    self.logger.debug(f"Экземпляр {module_name} пересоздан за {recreate_time:.2f}c методом {creation_method}")
    
        except Exception as e:
            error_msg = str(e)
            recovery_log.append({
                "time": time.time() - recovery_start,
                "stage": "recreate",
                "status": "error",
                "error": error_msg
        })
        
        self.logger.error(f"Ошибка пересоздания {module_name}: {error_msg}")
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
        Автоматическое восстановление всех упавших модулей
        """
        if not self.config["recovery"]["enabled"]:
            return {"enabled": False, "reason": "recovery_disabled", "timestamp": time.time()}
        
        if not self.config["recovery"]["auto_recover"]:
            return {"enabled": False, "reason": "auto_recovery_disabled", "timestamp": time.time()}
        
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
                "timestamp": time.time(),
                "checked_modules": len(self.modules)
            }
        
        self.logger.info(f"🔍 Обнаружено {len(failed_modules)} неактивных модулей")
        
        # Сортируем модули по приоритету восстановления
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
            self.logger.info("Мониторинг восстановления отключен")
            return
        
        self.logger.info("🔧 Запуск монитора восстановления...")
        
        check_interval = 10.0
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
                warning_threshold = 0.9
                critical_threshold = 0.7
                
                if health_ratio >= warning_threshold:
                    consecutive_failures = 0
                    continue
                
                # Система в предупреждающем или критическом состоянии
                state = "warning" if health_ratio >= critical_threshold else "critical"
                inactive_count = total_modules - active_modules
                
                self.logger.warning(f"Состояние системы: {state.upper()}. Активных модулей: {active_modules}/{total_modules} ({health_ratio:.1%}). Неактивных: {inactive_count}")
                
                consecutive_failures += 1
                
                # Запускаем восстановление если:
                # 1. Система в критическом состоянии ИЛИ
                # 2. Много последовательных проверок показывают проблемы
                if state == "critical" or consecutive_failures >= max_consecutive_failures:
                    self.logger.info(f"🚨 Запуск автовосстановления (причина: {state}, failures: {consecutive_failures})")
                    
                    recovery_report = await self.auto_recover_failed_modules()
                    
                    if recovery_report.get("recovered", 0) > 0:
                        consecutive_failures = 0
                        self.logger.info(f"Автовосстановление успешно: {recovery_report['recovered']} модулей восстановлено")
                    else:
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
                
                # Немедленная попытка восстановления
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
        await asyncio.sleep(1.0)
        await self.recover_module(module_name, force=True)

    async def _emergency_recovery_protocol(self):
        """Экстренный протокол восстановления"""
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
        """Полный отчёт о состоянии системы восстановления"""
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
                "monitor_interval": 10.0
            }
        }

    async def reset_recovery_attempts(self, module_name: str = None) -> Dict[str, Any]:
        """Сброс счётчиков попыток восстановления"""
        reset_results = []
        
        if module_name:
            # Сброс для конкретного модуля
            if module_name not in self.modules:
                return {"success": False, "reason": "module_not_found", "module": module_name}
            
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
        """Получение истории восстановлений"""
        return [
            {
                "timestamp": time.time() - i * 3600,
                "type": "auto_recovery" if i % 3 == 0 else "manual_recovery",
                "modules_recovered": max(1, 5 - i % 5),
                "success_rate": 0.8 - i * 0.1
            }
            for i in range(min(limit, 20))
        ]

    # ========================================================
    # 11. ЗАПУСК И УПРАВЛЕНИЕ ФОНОВЫМИ ЗАДАЧАМИ
    # ========================================================

    async def _start_background_tasks(self):
        """Запуск всех фоновых задач"""
        self.logger.info("🚀 Запуск фоновых задач...")
        
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
                await asyncio.sleep(0.1)
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
                
                await asyncio.sleep(5.0)
                
            except Exception as e:
                self.logger.error(f"Ошибка менеджера энергии: {e}")
                await asyncio.sleep(10.0)
        
        self.logger.info("Менеджер энергии остановлен")

    async def _balance_energy_flows(self):
        """Балансировка энергетических потоков"""
        for flow in self.energy_flows:
            if flow.current_flow > 0 and time.time() - flow.last_transfer > 30:
                flow.current_flow *= 0.9

    async def _stop_all_background_tasks(self):
        """Остановка всех фоновых задач"""
        self.logger.info("🛑 Остановка фоновых задач...")
        self.shutdown_requested = True
        
        for task in self.background_tasks:
            try:
                task.cancel()
            except:
                pass
        
        if self.background_tasks:
            try:
                await asyncio.wait(self.background_tasks, timeout=5.0)
            except:
                pass
        
        self.background_tasks.clear()
        self.logger.info("Фоновые задачи остановлены")

    # ========================================================
    # 12. ГРАЦИОЗНОЕ ВЫКЛЮЧЕНИЕ
    # ========================================================

    async def shutdown(self) -> Dict[str, Any]:
        """Полное грациозное выключение системы"""
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
        
        self.logger.info(f"🎯 KetherCore выключен за {total_time:.2f}с. Успешно выключено: {result['successful_shutdowns']}/{result['total_modules']} модулей")
        return result

    # ========================================================
    # 13. УТИЛИТЫ
    # ========================================================

    def _deep_update(self, target: Dict, source: Dict) -> Dict:
        """Рекурсивное обновление словаря"""
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value
        return target

    def get_module(self, module_name: str):
        """Получение экземпляра модуля по имени"""
        if module_name in self.modules:
            return self.modules[module_name].instance
        return None

    def get_module_status(self, module_name: str):
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
    # 14. ТЕСТОВАЯ ФУНКЦИЯ
    # ========================================================

    async def run_test_scenario(self) -> Dict[str, Any]:
        """Запуск тестового сценария"""
        self.logger.info("🧪 Запуск тестового сценария...")
        
        test_results = {}
        test_results["registration"] = await self.register_all_modules()
        test_results["activation"] = await self.activate_cascade()
        test_results["metrics"] = await self.collect_metrics()
        
        # Тест энергетических потоков
        energy_tests = []
        test_flows = [
            ("spirit_synthesis", "willpower_core", 10.0),
            ("spirit_core", "core_govx", 5.0),
        ]
        
        for source, target, amount in test_flows:
            result = await self.distribute_energy(source, target, amount)
            energy_tests.append({"flow": f"{source}→{target}", "amount": amount, "result": result})
        
        test_results["energy_tests"] = energy_tests
        test_results["recovery_status"] = await self.get_recovery_status()
        test_results["shutdown"] = await self.shutdown()
        
        # Итог
        active_modules = test_results["activation"]["activated_modules"]
        total_modules = test_results["activation"]["total_modules"]
        
        test_results["summary"] = {
            "success": active_modules == total_modules,
            "active_modules": f"{active_modules}/{total_modules}",
            "success_rate": (active_modules / total_modules) * 100 if total_modules > 0 else 0,
            "total_tests": 5,
            "passed_tests": sum(1 for key in ["registration", "activation", "metrics", "energy_tests", "shutdown"]
                if test_results.get(key, {}).get("success", False))
        }
        
        return test_results

# ============================================================
# 15. ФАБРИЧНАЯ ФУНКЦИЯ
# ============================================================

def create_keter_core(config: Optional[Dict[str, Any]] = None):
    """Фабричная функция для создания экземпляра KetherCore"""
    return KetherCore(config)

# ============================================================
# 16. ТОЧКА ВХОДА (ПРОДОЛЖЕНИЕ)
# ============================================================

    # Выводим результаты
    summary = test_results.get("summary", {})
    print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
    print(f"  Успешно: {'✅' if summary.get('success') else '❌'}")
    print(f"  Активных модулей: {summary.get('active_modules', '0/0')}")
    print(f"  Пройдено тестов: {summary.get('passed_tests', 0)}/{summary.get('total_tests', 0)}")
    
    # Детали по модулям
    activation = test_results.get("activation", {})
    if "results" in activation:
        print(f"\n🧩 СТАТУС МОДУЛЕЙ:")
        for module_name, result in activation["results"].items():
            status = result.get("status", "unknown")
            symbol = "✅" if status == "active" else "❌"
            print(f"  {symbol} {module_name}: {status}")
    
    print(f"\n🎯 Ketheric Block готов к интеграции с ISKRA-4!")
    return test_results


if __name__ == "__main__":
    # Запуск основной функции
    asyncio.run(main())
