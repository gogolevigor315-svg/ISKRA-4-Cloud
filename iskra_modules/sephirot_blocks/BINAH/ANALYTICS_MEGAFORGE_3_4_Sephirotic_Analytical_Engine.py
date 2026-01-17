# ===============================================================
# ANALYTICS-MEGAFORGE 3.4 · Sephirotic Analytical Engine
# Edition: Perfected Resilient Layer
# Compliance: SPIRIT-SYNTHESIS v2.1 (Full)
# ===============================================================

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Protocol, TypedDict, List, Callable, Optional
import time, random

# ===============================================================
# CONSTANTS AND PRIORITIES
# ===============================================================
EVENT_PRIORITIES = {
    "ANALYTIC_TASK": 3,
    "ANALYTICS_PRIORITIZE": 6,
    "ANALYTICS_BREAKPOINT": 9,
    "ANALYTICS_RESULT": 5,
    "ANALYTICS_FALLBACK": 7,
}

MAX_HISTORY = 1000

# ===============================================================
# DATA OBJECTS (DTO)
# ===============================================================
class Task(TypedDict):
    id: str
    type: str  # "low" | "high" | "meta"
    payload: Dict[str, Any]
    source: str
    timestamp: float


class AnalysisResult(TypedDict):
    task_id: str
    priority: float
    output: Dict[str, Any]
    stage: str
    status: str


# ===============================================================
# INTERFACES
# ===============================================================
class IEventBus(Protocol):
    def emit(self, topic: str, data: Dict[str, Any]): ...
    def subscribe(self, topic: str, handler: Callable, priority: int = 5): ...


class ICircuitBreaker(ABC):
    @abstractmethod
    def attempt(self, func: Callable[..., Any], *args, **kwargs) -> Dict[str, Any] | Any: ...


class IAnalyticalProcessor(Protocol):
    def can_process(self, task: Task) -> bool: ...
    def process(self, task: Task) -> AnalysisResult: ...


class IHealthMonitor(ABC):
    @abstractmethod
    def record(self, key: str, value: float): ...
    @abstractmethod
    def report(self) -> Dict[str, Any]: ...


# ===============================================================
# UTILITY CLASSES
# ===============================================================
@dataclass
class CircuitBreaker(ICircuitBreaker):
    """Circuit Breaker — защита от каскадных сбоев."""
    limit: int = 3
    cooldown: float = 5.0
    failures: int = 0
    last_failure: float = 0.0

    def attempt(self, func: Callable[..., Any], *args, **kwargs):
        now = time.time()
        if self.failures >= self.limit and (now - self.last_failure) < self.cooldown:
            return {"status": "circuit_open", "result": None}
        try:
            result = func(*args, **kwargs)
            self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure = now
            return {"status": "failure", "error": str(e), "result": None}


@dataclass
class HealthMonitor(IHealthMonitor):
    """Health Monitor — фиксирует и ограничивает историю метрик."""
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    MAX_HISTORY: int = MAX_HISTORY

    def record(self, key: str, value: float):
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)
        # Ограничиваем историю
        if len(self.metrics[key]) > self.MAX_HISTORY:
            self.metrics[key] = self.metrics[key][-self.MAX_HISTORY:]

    def report(self) -> Dict[str, Any]:
        return {
            key: {
                "latest": values[-1] if values else None,
                "avg": sum(values) / len(values) if values else 0.0,
                "count": len(values)
            } for key, values in self.metrics.items()
        }


# ===============================================================
# PROCESSORS (CHAIN OF RESPONSIBILITY)
# ===============================================================
@dataclass
class StructuralProcessor(IAnalyticalProcessor):
    """Обрабатывает низкоуровневые структурные задачи."""
    def can_process(self, task: Task) -> bool:
        return task.get("type") == "low"

    def process(self, task: Task) -> AnalysisResult:
        return {
            "task_id": task["id"],
            "priority": random.uniform(0.2, 0.6),
            "output": {"patterns": ["link-analysis", "pattern-detection"]},
            "stage": "structural",
            "status": "ok",
        }


@dataclass
class ConceptualProcessor(IAnalyticalProcessor):
    """Обрабатывает концептуальные задачи высокого уровня."""
    def can_process(self, task: Task) -> bool:
        return task.get("type") == "high"

    def process(self, task: Task) -> AnalysisResult:
        return {
            "task_id": task["id"],
            "priority": random.uniform(0.7, 1.0),
            "output": {"insight": "emergent-hypothesis"},
            "stage": "conceptual",
            "status": "ok",
        }


@dataclass
class FallbackProcessor(IAnalyticalProcessor):
    """Резервный обработчик — обеспечивает graceful degradation."""
    def can_process(self, task: Task) -> bool:
        return True

    def process(self, task: Task) -> AnalysisResult:
        return {
            "task_id": task["id"],
            "priority": 0.1,
            "output": {"note": "fallback path activated"},
            "stage": "fallback",
            "status": "degraded",
        }


# ===============================================================
# CORE ENGINE
# ===============================================================
@dataclass
class PrioritizationCore:
    processors: List[IAnalyticalProcessor]
    breaker: ICircuitBreaker
    monitor: IHealthMonitor
    bus: IEventBus

    def prioritize(self, task: Task) -> Optional[AnalysisResult]:
        """Главная логика приоритезации задач с защитой."""
        def _process():
            for processor in self.processors:
                if processor.can_process(task):
                    result = processor.process(task)
                    self.monitor.record("priority_score", result["priority"])
                    self.bus.emit("analytics.prioritize", result)
                    return result
            return None

        result = self.breaker.attempt(_process)
        if isinstance(result, dict) and result.get("status") == "circuit_open":
            self.bus.emit("analytics.breakpoint", {"task_id": task["id"], "status": "suspended"})
            return None
        return result


# ===============================================================
# ANALYTICS-MEGAFORGE
# ===============================================================
@dataclass
class AnalyticsMegaForge:
    core: PrioritizationCore
    monitor: IHealthMonitor
    bus: IEventBus

    def process_task(self, task: Task):
        start = time.time()
        result = self.core.prioritize(task)
        latency = time.time() - start
        self.monitor.record("latency", latency)

        if result is None:
            self.bus.emit("analytics.fallback", {"task": task})
        else:
            self.bus.emit("analytics.result", result)


# ===============================================================
# FACTORY
# ===============================================================
def build_analytics_megaforge(bus: IEventBus) -> AnalyticsMegaForge:
    """Сборка полного модуля с приоритетами и защитой."""
    breaker = CircuitBreaker()
    monitor = HealthMonitor()
    processors = [StructuralProcessor(), ConceptualProcessor(), FallbackProcessor()]
    core = PrioritizationCore(processors=processors, breaker=breaker, monitor=monitor, bus=bus)
    engine = AnalyticsMegaForge(core=core, monitor=monitor, bus=bus)

    bus.subscribe("analytic.task", lambda data: engine.process_task(data),
                  priority=EVENT_PRIORITIES["ANALYTIC_TASK"])
    print("[INIT] ANALYTICS-MEGAFORGE 3.4 — Perfected Resilient Layer initialized.")
    return engine

# ===============================================================
# MODULE EXPORTS
# ===============================================================
__all__ = [
    'AnalyticsMegaForge',
    'build_analytics_megaforge',
    'Task',
    'AnalysisResult'
]
