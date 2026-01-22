"""
ISKRA-4 · SPIRIT-CORE v3.4 (Orchestration Governance) · KETHERIC BLOCK
Главный оркестратор всех духовных процессов Keter
Интегрируется с Policy Governor для управления приоритетами
"""

import asyncio
import math
import time
import statistics
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Protocol
import logging

# Настройка логирования
logger = logging.getLogger("keter.spirit_core_v34")

# ===============================================================
# I. ИНТЕРФЕЙСЫ ДЛЯ ИНТЕГРАЦИИ
# ===============================================================

class IPolicyGovernorLink(Protocol):
    """Связь с Policy Governor v1.2"""
    async def get_governance_rules(self) -> Dict[str, Any]: ...
    async def apply_policy_constraint(self, module: str, constraint: Dict) -> bool: ...
    async def report_orchestration_metrics(self, metrics: Dict) -> bool: ...

class IWillpowerCoreLink(Protocol):
    """Связь с WILLPOWER-CORE v3.2"""
    async def get_current_strength(self) -> float: ...
    async def get_volitional_intensity(self) -> float: ...
    async def receive_priority_boost(self, priority_level: float) -> bool: ...

class ISpiritCoreLink(Protocol):
    """Связь с духовным ядром (SPIRIT-CORE v3.3 или SPIRIT-SYNTHESIS)"""
    async def ignite_spiritual_impulse(self, intent_data: Dict) -> Dict: ...
    async def get_spiritual_state(self) -> Dict: ...
    async def adjust_spiritual_flow(self, adjustment: float) -> bool: ...

class IMoralMemoryLink(Protocol):
    """Связь с MORAL-MEMORY 3.1"""
    async def get_alignment_score(self) -> float: ...
    async def get_ethical_coherence(self) -> float: ...
    async def register_orchestration_event(self, event: Dict) -> bool: ...

class IKeterIntegration(Protocol):
    """Интеграция с ядром Keter"""
    async def register_orchestrator(self, orchestrator_instance: Any) -> None: ...
    async def distribute_energy_budget(self, budget_allocation: Dict[str, float]) -> bool: ...
    async def broadcast_orchestration_state(self, state: Dict) -> bool: ...

# ===============================================================
# II. ОБЩИЕ КОМПОНЕНТЫ (адаптированные для Keter)
# ===============================================================

class KeterCircuitBreaker:
    """Предохранитель оркестрационных потоков Keter"""
    def __init__(self, limit: int = 3, recovery_time: float = 10.0):
        self.failures = 0
        self.limit = limit
        self.open = False
        self.recovery_time = recovery_time
        self.tripped_at = 0.0
        
    async def attempt(self, func: Callable, *args, **kwargs) -> Optional[Any]:
        """Попытка выполнения с защитой от сбоев"""
        # Проверка восстановления
        if self.open and time.time() - self.tripped_at > self.recovery_time:
            self.open = False
            self.failures = 0
            logger.info("[CIRCUIT] Автовосстановление предохранителя")
        
        if self.open:
            logger.warning(f"[CIRCUIT] 🔴 Поток остановлен — {func.__qualname__}")
            return None
            
        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            
            # Успех — уменьшаем счётчик сбоев
            if self.failures > 0:
                self.failures = max(0, self.failures - 0.3)
                
            return result
            
        except Exception as e:
            self.failures += 1
            logger.error(f"[CIRCUIT] Сбой {self.failures}/{self.limit}: {e}")
            
            if self.failures >= self.limit:
                self.open = True
                self.tripped_at = time.time()
                logger.critical("[CIRCUIT] 🔒 Предохранитель сработал")
                
            return None

class KeterPriorityEventBus:
    """Шина событий с приоритетами для оркестрации Keter"""
    def __init__(self):
        self.listeners: Dict[str, List[tuple[int, Callable]]] = {}
        self.event_history: List[Dict] = []
        
    def subscribe(self, topic: str, handler: Callable, priority: int = 0):
        """Подписка на события с приоритетом"""
        if topic not in self.listeners:
            self.listeners[topic] = []
        
        # Удаляем старые подписки того же обработчика
        self.listeners[topic] = [(p, h) for p, h in self.listeners[topic] if h != handler]
        self.listeners[topic].append((priority, handler))
        
        # Сортируем по приоритету (высокий приоритет первый)
        self.listeners[topic].sort(key=lambda x: -x[0])
        
        logger.debug(f"[BUS] Подписка: {handler.__qualname__} → {topic} (приоритет: {priority})")
    
    async def emit(self, topic: str, data: Dict, priority: int = 0):
        """Асинхронная публикация события"""
        # Записываем в историю
        event_record = {
            "timestamp": time.time(),
            "topic": topic,
            "priority": priority,
            "data": data
        }
        self.event_history.append(event_record)
        self.event_history[:] = self.event_history[-1000:]
        
        # Ищем слушателей
        listeners = self.listeners.get(topic, [])
        if not listeners:
            logger.debug(f"[BUS] Нет слушателей для {topic}")
            return
        
        # Выполняем обработчики в порядке приоритета
        for handler_priority, handler in listeners:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(data)
                else:
                    handler(data)
            except Exception as e:
                logger.error(f"[BUS] Ошибка в {handler.__qualname__}: {e}")
                traceback.print_exc()

# ===============================================================
# III. УПРАВЛЕНИЕ РЕСУРСАМИ И ПРИОРИТЕТАМИ
# ===============================================================

@dataclass
class KeterResourceState:
    """Отслеживает использование ресурсов Keter"""
    cognitive_load: float = 0.3
    emotional_load: float = 0.25
    spiritual_load: float = 0.35
    moral_tension: float = 0.2
    sephirotic_pressure: float = 0.15  # Давление от других сефир
    load_history: List[Dict] = field(default_factory=list)
    
    async def calculate_total_load(self) -> float:
        """Вычисляет общую нагрузку Keter с учётом весов"""
        weighted_load = (
            self.cognitive_load * 0.25 +
            self.emotional_load * 0.20 +
            self.spiritual_load * 0.30 +
            self.moral_tension * 0.15 +
            self.sephirotic_pressure * 0.10
        )
        
        # Записываем историю
        load_record = {
            "timestamp": time.time(),
            "total_load": weighted_load,
            "components": {
                "cognitive": self.cognitive_load,
                "emotional": self.emotional_load,
                "spiritual": self.spiritual_load,
                "moral": self.moral_tension,
                "sephirotic": self.sephirotic_pressure
            }
        }
        self.load_history.append(load_record)
        self.load_history[:] = self.load_history[-500:]
        
        return round(weighted_load, 4)
    
    async def update_from_policy(self, policy_rules: Dict):
        """Обновление состояния ресурсов на основе политик"""
        if "load_limits" in policy_rules:
            limits = policy_rules["load_limits"]
            self.cognitive_load = min(self.cognitive_load, limits.get("cognitive", 1.0))
            self.spiritual_load = min(self.spiritual_load, limits.get("spiritual", 1.0))
        
        if "pressure_adjustment" in policy_rules:
            self.sephirotic_pressure *= policy_rules["pressure_adjustment"]
    
    async def get_load_statistics(self) -> Dict:
        """Статистика нагрузки за последний период"""
        if not self.load_history:
            return {"average": 0.0, "trend": "stable", "stability": 0.0}
        
        recent_loads = [r["total_load"] for r in self.load_history[-50:]]
        avg_load = statistics.mean(recent_loads)
        
        # Анализ тренда
        if len(recent_loads) >= 10:
            first_half = statistics.mean(recent_loads[:5])
            second_half = statistics.mean(recent_loads[-5:])
            trend = "increasing" if second_half > first_half * 1.1 else "decreasing" if second_half < first_half * 0.9 else "stable"
        else:
            trend = "stable"
        
        # Стабильность
        stability = 1.0 - statistics.stdev(recent_loads) if len(recent_loads) > 1 else 1.0
        
        return {
            "average_load": round(avg_load, 4),
            "trend": trend,
            "stability": round(stability, 4),
            "current_load": round(recent_loads[-1] if recent_loads else 0.0, 4)
        }

@dataclass
class KeterPriorityManager:
    """Динамическое управление приоритетами модулей Keter"""
    base_priorities: Dict[str, float] = field(default_factory=lambda: {
        "WILLPOWER": 0.9,      # Воля - высший приоритет
        "SPIRIT_SYNTHESIS": 0.85,  # Духовный синтез
        "MORAL_MEMORY": 0.8,   # Моральная память
        "SPIRIT_CORE": 0.75,   # Духовное ядро
        "CORE_GOVX": 0.7,      # Управление ядром
        "INTUITION": 0.65      # Интуиция
    })
    
    adjustment_history: List[Dict] = field(default_factory=list)
    
    async def adjust_priorities(self, resource_state: KeterResourceState, policy_rules: Dict) -> Dict[str, float]:
        """Корректировка приоритетов на основе нагрузки и политик"""
        total_load = await resource_state.calculate_total_load()
        
        # Базовый коэффициент коррекции
        load_factor = 1.0 - (total_load / 2)
        load_factor = max(0.3, min(1.0, load_factor))
        
        # Применяем политики
        policy_factor = policy_rules.get("priority_modifier", 1.0)
        
        # Корректируем каждый приоритет
        adjusted_priorities = {}
        for module, base_priority in self.base_priorities.items():
            # Базовая корректировка по нагрузке
            adjusted = base_priority * load_factor
            
            # Специфичные правила из политик
            module_rules = policy_rules.get("module_priorities", {})
            if module in module_rules:
                adjusted = module_rules[module] * policy_factor
            
            # Ограничиваем
            adjusted = max(0.1, min(1.0, adjusted))
            adjusted_priorities[module] = round(adjusted, 3)
        
        # Записываем историю
        adjustment_record = {
            "timestamp": time.time(),
            "total_load": total_load,
            "adjusted_priorities": adjusted_priorities.copy(),
            "load_factor": load_factor,
            "policy_factor": policy_factor
        }
        self.adjustment_history.append(adjustment_record)
        self.adjustment_history[:] = self.adjustment_history[-200:]
        
        logger.info(f"[PRIORITY] Приоритеты скорректированы (нагрузка: {total_load:.3f})")
        return adjusted_priorities
    
    async def get_priority_statistics(self) -> Dict:
        """Статистика приоритетов"""
        if not self.adjustment_history:
            return {"recent_adjustments": 0, "stability": 0.0}
        
        # Анализ стабильности приоритетов
        recent_changes = []
        for i in range(1, min(10, len(self.adjustment_history))):
            curr = self.adjustment_history[-i]["adjusted_priorities"]
            prev = self.adjustment_history[-i-1]["adjusted_priorities"]
            
            change = 0.0
            for module in curr:
                if module in prev:
                    change += abs(curr[module] - prev[module])
            
            recent_changes.append(change / len(curr))
        
        avg_change = statistics.mean(recent_changes) if recent_changes else 0.0
        stability = 1.0 - min(avg_change, 1.0)
        
        return {
            "recent_adjustments": len(self.adjustment_history),
            "priority_stability": round(stability, 4),
            "average_change_per_cycle": round(avg_change, 4),
            "current_priorities": self.adjustment_history[-1]["adjusted_priorities"] if self.adjustment_history else {}
        }

# ===============================================================
# IV. ГЛАВНЫЙ ОРКЕСТРАТОР KETER
# ===============================================================

@dataclass
class SPIRIT_CORE_v34_KETER:
    """
    Оркестрационный губернатор Keter v3.4
    Координирует все духовные процессы и распределяет ресурсы
    """
    
    def __init__(
        self,
        policy_governor_link: Optional[IPolicyGovernorLink] = None,
        willpower_link: Optional[IWillpowerCoreLink] = None,
        spirit_core_link: Optional[ISpiritCoreLink] = None,
        moral_memory_link: Optional[IMoralMemoryLink] = None,
        keter_integration: Optional[IKeterIntegration] = None
    ):
        self.name = "SPIRIT-CORE-v3.4"
        self.version = "3.4.0"
        self.role = "orchestration_governor"
        
        # Внешние связи
        self.policy_governor = policy_governor_link
        self.willpower_core = willpower_link
        self.spirit_core = spirit_core_link
        self.moral_memory = moral_memory_link
        self.keter_integration = keter_integration
        
        # Внутренние компоненты
        self.bus = KeterPriorityEventBus()
        self.circuit_breaker = KeterCircuitBreaker(limit=3, recovery_time=15.0)
        self.resource_state = KeterResourceState()
        self.priority_manager = KeterPriorityManager()
        
        # Состояние
        self.cycle_count = 0
        self.last_orchestration: Dict = {}
        self.orchestration_history: List[Dict] = []
        self.activation_time = time.time()
        self.is_active = False
        
        # Настройка обработчиков событий
        self._setup_event_handlers()
        
        logger.info(f"[{self.name}] Инициализирован v{self.version}")
    
    def _setup_event_handlers(self):
        """Настройка обработчиков внутренних событий"""
        self.bus.subscribe("orchestration.cycle.start", self._on_cycle_start, priority=10)
        self.bus.subscribe("orchestration.phase.complete", self._on_phase_complete, priority=8)
        self.bus.subscribe("resource.load.update", self._on_resource_update, priority=6)
    
    async def _on_cycle_start(self, data: Dict):
        """Обработчик начала цикла оркестрации"""
        logger.debug(f"[{self.name}] Начало цикла {self.cycle_count}")
    
    async def _on_phase_complete(self, data: Dict):
        """Обработчик завершения фазы"""
        phase = data.get("phase", "unknown")
        logger.debug(f"[{self.name}] Фаза завершена: {phase}")
    
    async def _on_resource_update(self, data: Dict):
        """Обработчик обновления ресурсов"""
        await self.resource_state.update_from_policy(data.get("policy_rules", {}))
    
    async def activate(self) -> bool:
        """Активация оркестрационного губернатора"""
        try:
            # Регистрация в Keter
            if self.keter_integration:
                await self.keter_integration.register_orchestrator(self)
            
            # Получение начальных политик
            if self.policy_governor:
                initial_rules = await self.policy_governor.get_governance_rules()
                await self.resource_state.update_from_policy(initial_rules)
            
            self.is_active = True
            self.activation_time = time.time()
            
            logger.info(f"[{self.name}] ✅ Оркестрационный губернатор активирован")
            
            # Первый цикл оркестрации
            await self.perform_orchestration_cycle()
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.name}] ❌ Ошибка активации: {e}")
            return False
    
    async def perform_orchestration_cycle(self) -> Dict:
        """
        Выполнение полного цикла оркестрации Keter
        Координирует все модули и распределяет ресурсы
        """
        if not self.is_active:
            return {"error": "Оркестратор не активирован"}
        
        self.cycle_count += 1
        cycle_start = time.time()
        
        try:
            # 1. Начало цикла
            await self.bus.emit("orchestration.cycle.start", {
                "cycle": self.cycle_count,
                "timestamp": cycle_start
            }, priority=10)
            
            # 2. Получение политик управления
            policy_rules = {}
            if self.policy_governor:
                policy_rules = await self.policy_governor.get_governance_rules()
            
            # 3. Получение текущего состояния модулей
            will_strength = 0.85  # значение по умолчанию
            moral_alignment = 0.9  # значение по умолчанию
            
            if self.willpower_core:
                will_strength = await self.willpower_core.get_current_strength()
            
            if self.moral_memory:
                moral_alignment = await self.moral_memory.get_alignment_score()
            
            # 4. Корректировка приоритетов
            priorities = await self.priority_manager.adjust_priorities(
                self.resource_state,
                policy_rules
            )
            
            # 5. Обновление духовного импульса с учётом приоритетов
            spirit_state = {"flow": 0.0, "resonance": 0.0, "state": "idle"}
            if self.spirit_core:
                spirit_intent = {
                    "cosmic_clarity": 0.85 * priorities.get("SPIRIT_CORE", 0.75),
                    "priority_boost": priorities.get("SPIRIT_CORE", 0.75)
                }
                
                spirit_state = await self.circuit_breaker.attempt(
                    self.spirit_core.ignite_spiritual_impulse,
                    spirit_intent
                ) or spirit_state
            
            # 6. Балансировка ресурсов
            await self._rebalance_resources(
                spirit_state=spirit_state,
                moral_alignment=moral_alignment,
                will_strength=will_strength,
                priorities=priorities
            )
            
            # 7. Отправка приоритетных бустов модулям
            await self._distribute_priority_boosts(priorities)
            
            # 8. Формирование отчёта о цикле
            cycle_duration = time.time() - cycle_start
            orchestration_snapshot = {
                "timestamp": time.time(),
                "cycle": self.cycle_count,
                "duration": round(cycle_duration, 4),
                "spirit_state": spirit_state,
                "priorities": priorities,
                "resources": await self.resource_state.calculate_total_load(),
                "module_states": {
                    "willpower": will_strength,
                    "moral_alignment": moral_alignment
                },
                "policy_applied": bool(policy_rules)
            }
            
            self.last_orchestration = orchestration_snapshot
            self.orchestration_history.append(orchestration_snapshot)
            self.orchestration_history[:] = self.orchestration_history[-100:]
            
            # 9. Отправка событий и отчётов
            await self.bus.emit("orchestration.cycle.complete", orchestration_snapshot, priority=9)
            
            if self.policy_governor:
                metrics = {
                    "cycle": self.cycle_count,
                    "resource_utilization": await self.resource_state.get_load_statistics(),
                    "priority_distribution": priorities
                }
                await self.policy_governor.report_orchestration_metrics(metrics)
            
            if self.keter_integration:
                await self.keter_integration.broadcast_orchestration_state(orchestration_snapshot)
            
            logger.info(f"[{self.name}] 🔄 Цикл {self.cycle_count} завершён за {cycle_duration:.3f}с")
            
            return orchestration_snapshot
            
        except Exception as e:
            logger.error(f"[{self.name}] Ошибка в цикле оркестрации: {e}")
            return {
                "error": str(e),
                "cycle": self.cycle_count,
                "timestamp": time.time()
            }
    
    async def _rebalance_resources(
        self,
        spirit_state: Dict,
        moral_alignment: float,
        will_strength: float,
        priorities: Dict[str, float]
    ):
        """Адаптивная балансировка ресурсов Keter"""
        resonance = spirit_state.get("resonance", 0.8)
        flow = spirit_state.get("flow", 0.7)
        
        # Общий фактор гармонии
        harmony_factor = (resonance + flow + will_strength + moral_alignment) / 4
        
        # Динамическая балансировка с учётом приоритетов
        spirit_priority = priorities.get("SPIRIT_CORE", 0.75)
        will_priority = priorities.get("WILLPOWER", 0.9)
        
        # Когнитивная нагрузка (зависит от воли и духовности)
        self.resource_state.cognitive_load = (
            abs(math.sin(time.time() / 6)) * 
            (1 - (will_strength * will_priority + harmony_factor) / 2)
        )
        
        # Духовная нагрузка (обратно пропорциональна гармонии)
        self.resource_state.spiritual_load = max(0.1, 1 - harmony_factor * spirit_priority * 0.9)
        
        # Моральное напряжение (растёт при отклонении от идеала)
        self.resource_state.moral_tension = abs(0.7 - moral_alignment) * 0.8
        
        # Эмоциональная нагрузка (синусоидальная базовая + влияние)
        self.resource_state.emotional_load = (
            abs(math.sin(time.time() / 8)) * 0.5 +
            self.resource_state.moral_tension * 0.3
        )
        
        # Давление от других сефир (заглушка, будет из сефиротического движка)
        self.resource_state.sephirotic_pressure = 0.15 * (1 - harmony_factor)
        
        logger.debug(f"[{self.name}] Ресурсы сбалансированы (гармония: {harmony_factor:.3f})")
    
    async def _distribute_priority_boosts(self, priorities: Dict[str, float]):
        """Распределение приоритетных бустов модулям"""
        # Буст для Willpower-CORE
        if self.willpower_core and "WILLPOWER" in priorities:
            will_boost = priorities["WILLPOWER"] * 0.5
            await self.willpower_core.receive_priority_boost(will_boost)
        
        # Буст для Spirit-CORE
        if self.spirit_core and "SPIRIT_CORE" in priorities:
            spirit_boost = priorities["SPIRIT_CORE"] * 0.3
            await self.spirit_core.adjust_spiritual_flow(spirit_boost)
        
        # Регистрация события в Moral-Memory
        if self.moral_memory:
            priority_event = {
                "event_type": "priority_distribution",
                "priorities": priorities,
                "timestamp": time.time(),
                "source": self.name
            }
            await self.moral_memory.register_orchestration_event(priority_event)
    
    async def get_orchestration_status(self) -> Dict:
        """Получение статуса оркестрации"""
        resource_stats = await self.resource_state.get_load_statistics()
        priority_stats = await self.priority_manager.get_priority_statistics()
        
        return {
            "name": self.name,
            "version": self.version,
            "active": self.is_active,
            "uptime": round(time.time() - self.activation_time, 2),
            "cycle_count": self.cycle_count,
            "resource_statistics": resource_stats,
            "priority_statistics": priority_stats,
            "last_orchestration": self.last_orchestration.get("timestamp", 0),
            "connections": {
                "has_policy_governor": self.policy_governor is not None,
                "has_willpower_core": self.willpower_core is not None,
                "has_spirit_core": self.spirit_core is not None,
                "has_moral_memory": self.moral_memory is not None,
                "has_keter_integration": self.keter_integration is not None
            }
        }
    
    async def apply_policy_constraint(self, constraint: Dict) -> bool:
        """Применение ограничения от Policy Governor"""
        if not self.policy_governor:
            return False
        
        try:
            success = await self.policy_governor.apply_policy_constraint(
                module=self.name,
                constraint=constraint
            )
            
            if success:
                await self.resource_state.update_from_policy(constraint.get("resource_rules", {}))
                logger.info(f"[{self.name}] Политика применена: {constraint.get('name', 'unknown')}")
            
            return success
            
        except Exception as e:
            logger.error(f"[{self.name}] Ошибка применения политики: {e}")
            return False
    
    async def shutdown(self):
        """Корректное выключение оркестратора"""
        self.is_active = False
        
        # Завершающий отчёт
        if self.policy_governor:
            final_metrics = {
                "final_cycle": self.cycle_count,
                "total_uptime": round(time.time() - self.activation_time, 2),
                "average_cycle_duration": 0.0
            }
            
            if self.orchestration_history:
                durations = [c.get("duration", 0) for c in self.orchestration_history[-10:]]
                final_metrics["average_cycle_duration"] = statistics.mean(durations)
            
            await self.policy_governor.report_orchestration_metrics(final_metrics)
        
        logger.info(f"[{self.name}] Выключен (выполнено циклов: {self.cycle_count})")

# ===============================================================
# V. ФАБРИЧНАЯ ФУНКЦИЯ ДЛЯ ИНТЕГРАЦИИ
# ===============================================================

async def create_spirit_core_v34_module(
    policy_governor: Optional[IPolicyGovernorLink] = None,
    willpower_core: Optional[IWillpowerCoreLink] = None,
    spirit_core: Optional[ISpiritCoreLink] = None,
    moral_memory: Optional[IMoralMemoryLink] = None,
    keter_core: Optional[IKeterIntegration] = None
) -> SPIRIT_CORE_v34_KETER:
    """
    Фабричная функция для создания SPIRIT-CORE v3.4
    Используется в keter_core.py для интеграции
    """
    module = SPIRIT_CORE_v34_KETER(
        policy_governor_link=policy_governor,
        willpower_link=willpower_core,
        spirit_core_link=spirit_core,
        moral_memory_link=moral_memory,
        keter_integration=keter_core
    )
    
    # Автоматическая активация
    await module.activate()
    
    return module

# ===============================================================
# VI. ТЕСТОВЫЙ ЗАПУСК
# ===============================================================

async def _test_spirit_core_v34():
    """Тестовый запуск оркестрационного губернатора"""
    print("🧪 Тест SPIRIT-CORE v3.4 (Orchestration Governor)")
    
    # Мок-объекты
    class MockPolicyGovernor:
        async def get_governance_rules(self):
            return {
                "priority_modifier": 1.0,
                "module_priorities": {"WILLPOWER": 0.95},
                "load_limits": {"cognitive": 0.8, "spiritual": 0.9}
            }
        async def apply_policy_constraint(self, module, constraint):
            print(f"[MOCK-POLICY] Ограничение применено к {module}")
            return True
        async def report_orchestration_metrics(self, metrics):
            print(f"[MOCK-POLICY] Метрики получены: цикл {metrics.get('cycle')}")
            return True
    
    class MockWillpower:
        async def get_current_strength(self): return 0.88
        async def get_volitional_intensity(self): return 0.85
        async def receive_priority_boost(self, boost):
            print(f"[MOCK-WILL] Получен буст: {boost}")
            return True
    
    class MockSpiritCore:
        async def ignite_spiritual_impulse(self, intent):
            return {"flow": 0.92, "resonance": 0.89, "state": "active"}
        async def get_spiritual_state(self):
            return {"flow": 0.92, "resonance": 0.89}
        async def adjust_spiritual_flow(self, adjustment):
            print(f"[MOCK-SPIRIT] Корректировка потока: {adjustment}")
            return True
    
    # Создание модуля
    module = SPIRIT_CORE_v34_KETER(
        policy_governor_link=MockPolicyGovernor(),
        willpower_link=MockWillpower(),
        spirit_core_link=MockSpiritCore()
    )
    
    # Активация
    success = await module.activate()
    print(f"Активация: {'✅' if success else '❌'}")
    
    if success:
        # Запуск нескольких циклов оркестрации
        for i in range(3):
            result = await module.perform_orchestration_cycle()
            print(f"Цикл {i+1}: ресурсы={result.get('resources', 0):.3f}, приоритеты={len(result.get('priorities', {}))}")
            await asyncio.sleep(0.5)
        
        # Получение статуса
        status = await module.get_orchestration_status()
        print(f"Статус: {status['cycle_count']} циклов, нагрузка: {status['resource_statistics']['current_load']:.3f}")
        
        # Применение политики
        constraint = {"name": "test_constraint", "resource_rules": {"load_limits": {"cognitive": 0.5}}}
        applied = await module.apply_policy_constraint(constraint)
        print(f"Политика применена: {'✅' if applied else '❌'}")
        
        # Выключение
        await module.shutdown()

# ===============================================================
# СОВМЕСТИМОСТЬ С ИМПОРТОМ
# ===============================================================

SpiritCoreV3_4 = SPIRIT_CORE_v34_KETER

# ===============================================================
# ФУНКЦИИ ДЛЯ СИСТЕМНОЙ СОВМЕСТИМОСТИ
# ===============================================================

def activate_spirit():
    """
    Функция активации духа для импорта из willpower_core_v3_2
    """
    try:
        return {
            "status": "activated",
            "module": "spirit_core_v3_4",
            "version": "3.4",
            "sephira": "KETHER",
            "message": "Spirit core activated",
            "timestamp": time.time() if 'time' in globals() else 0
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Cannot activate spirit: {e}"
        }

def get_spirit_core():
    """Получение ядра духа"""
    return SpiritCoreV3_4()

def spirit_available():
    """Проверка доступности духа"""
    return True

def get_module_instance():
    """Единственная функция для API системы ISKRA-4"""
    return SpiritCoreV3_4()

# ===============================================================
# ЗАПУСК ТЕСТОВ
# ===============================================================

if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        asyncio.run(_test_spirit_core_v34())
    else:
        print("ISKRA-4 · SPIRIT-CORE v3.4 (Orchestration Governance)")
        print("Главный оркестратор духовных процессов Keter")
        print("Используйте --test для запуска теста")

# Функции уже доступны для импорта, не нужно добавлять в __all__
# Если где-то в начале файла есть __all__ = [], он останется пустым
