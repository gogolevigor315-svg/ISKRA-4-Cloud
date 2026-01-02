"""
INTUITION-MATRIX 3.4 · Sephirotic Chokhmah (Executor)
Чистый исполнительный код для интеграции в систему ISKRA-4
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, List, Protocol, TypeVar, TypedDict
from dataclasses import dataclass, field
import time
import threading

T = TypeVar("T")

# === TYPE DEFINITIONS =========================================================

class IntuitionSignals(TypedDict, total=False):
    clarity: float
    resonance: float
    confidence: float | None


class HealthThresholds:
    GREEN_MAX = 2
    YELLOW_MAX = 4


# === INTERFACES ===============================================================

class IEventBus(Protocol):
    def subscribe(self, topic: str, handler: Callable, priority: int = 5): ...
    def emit(self, topic: str, data: Dict[str, Any]): ...


class ICircuitBreaker(ABC):
    @abstractmethod
    def attempt(self, func: Callable[..., T], *args, **kwargs) -> Dict[str, Any] | T: ...


class IEventBuffer(ABC):
    @abstractmethod
    def add(self, hypothesis: Dict[str, Any]): ...
    @abstractmethod
    def flush(self, force: bool = False): ...
    @abstractmethod
    def get_queue_size(self) -> int: ...


class IHypothesisWeaver(ABC):
    @abstractmethod
    def generate(self, signals: IntuitionSignals) -> Dict[str, Any]: ...


class ITimingService(ABC):
    @abstractmethod
    def now(self) -> float: ...
    @abstractmethod
    def sleep(self, seconds: float): ...


class IIntuitionMonitor(ABC):
    @abstractmethod
    def update(self, queue_size: int, failures: int): ...
    @abstractmethod
    def report(self) -> Dict[str, Any]: ...


# === CORE SERVICES ============================================================

class TimingService(ITimingService):
    """Абстрактный сервис времени — изолирует зависимости для тестирования."""
    def now(self) -> float:
        return time.time()
    def sleep(self, seconds: float):
        time.sleep(seconds)


@dataclass
class CircuitBreaker(ICircuitBreaker):
    """Circuit Breaker — защита от каскадных сбоев."""
    limit: int = 3
    cooldown: float = 5.0
    failures: int = 0
    last_failure: float = 0.0
    timing: ITimingService = field(default_factory=TimingService)

    def attempt(self, func: Callable[..., T], *args, **kwargs) -> Dict[str, Any] | T:
        now = self.timing.now()
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


# === EVENT BUS ================================================================

class PriorityEventBus(IEventBus):
    """Шина событий с приоритетами и защитой от каскадных ошибок."""
    def __init__(self):
        self.listeners: Dict[str, List[tuple[int, Callable]]] = {}
        self.lock = threading.Lock()

    def subscribe(self, topic: str, handler: Callable, priority: int = 5):
        with self.lock:
            self.listeners.setdefault(topic, []).append((priority, handler))
            self.listeners[topic].sort(key=lambda x: x[0])

    def emit(self, topic: str, data: Dict[str, Any]):
        with self.lock:
            for _, handler in sorted(self.listeners.get(topic, []), key=lambda x: x[0]):
                try:
                    handler(data)
                except Exception as e:
                    print(f"[WARN][EventBus] {topic} → Handler error: {e}")


# === EVENT BUFFER LAYER =======================================================

@dataclass
class BufferManager:
    """Хранит гипотезы с ограничением размера."""
    max_size: int = 100
    queue: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, item: Dict[str, Any]):
        if len(self.queue) >= self.max_size:
            self.queue.pop(0)
        self.queue.append(item)

    def drain(self) -> List[Dict[str, Any]]:
        batch = self.queue.copy()
        self.queue.clear()
        return batch

    def size(self) -> int:
        return len(self.queue)


@dataclass
class EventDispatcher:
    """Отвечает за безопасную публикацию событий."""
    bus: IEventBus
    cb: ICircuitBreaker

    def dispatch(self, topic: str, data: Dict[str, Any]):
        result = self.cb.attempt(lambda: self.bus.emit(topic, data))
        if isinstance(result, dict) and result.get("status") == "circuit_open":
            print(f"[WARN] Circuit open for topic: {topic}")


@dataclass
class EventBufferLayer(IEventBuffer):
    buffer: BufferManager
    dispatcher: EventDispatcher
    timing: ITimingService
    flush_interval: float = 3.0
    last_flush: float = field(default_factory=lambda: time.time())

    def add(self, hypothesis: Dict[str, Any]):
        self.buffer.add(hypothesis)

    def flush(self, force: bool = False):
        now = self.timing.now()
        if force or (now - self.last_flush) >= self.flush_interval:
            batch = self.buffer.drain()
            if batch:
                self.dispatcher.dispatch("intuition.queue.sync", {"batch": batch, "timestamp": now})
            self.last_flush = now

    def get_queue_size(self) -> int:
        return self.buffer.size()


# === HYPOTHESIS WEAVER ========================================================

@dataclass
class HypothesisWeaver(IHypothesisWeaver):
    cb: ICircuitBreaker
    moral_weight: float = 0.8
    foresight_factor: float = 0.9
    timing: ITimingService = field(default_factory=TimingService)

    def generate(self, signals: IntuitionSignals) -> Dict[str, Any]:
        return self.cb.attempt(self._generate_impl, signals)

    def _generate_impl(self, signals: IntuitionSignals) -> Dict[str, Any]:
        clarity = signals.get("clarity", 0.5)
        resonance = signals.get("resonance", 0.6)
        confidence = signals.get("confidence", 0.7)
        probability = (clarity * self.foresight_factor + resonance * self.moral_weight + confidence) / 3
        return {
            "timestamp": self.timing.now(),
            "probability": round(probability, 3),
            "context": signals,
            "source": "HypothesisWeaver"
        }


# === MONITOR ================================================================

@dataclass
class IntuitionMonitor(IIntuitionMonitor):
    metrics: Dict[str, Any] = field(default_factory=lambda: {"history": []})

    def update(self, queue_size: int, failures: int):
        self.metrics["history"].append({"queue": queue_size, "failures": failures})
        self.metrics["history"] = self.metrics["history"][-100:]
        self.metrics["queue_size"] = queue_size
        self.metrics["failures"] = failures
        self.metrics["health"] = (
            "green"
            if failures <= HealthThresholds.GREEN_MAX
            else "yellow"
            if failures <= HealthThresholds.YELLOW_MAX
            else "red"
        )

    def report(self) -> Dict[str, Any]:
        return self.metrics


# === INTUITION MATRIX CORE ====================================================

@dataclass
class IntuitionMatrix:
    buffer: IEventBuffer
    weaver: IHypothesisWeaver
    monitor: IIntuitionMonitor
    bus: IEventBus

    def process(self, signals: IntuitionSignals):
        hypothesis = self.weaver.generate(signals)
        if isinstance(hypothesis, dict):
            self.buffer.add(hypothesis)
            self.monitor.update(queue_size=self.buffer.get_queue_size(), failures=0)

    def synchronize(self):
        self.buffer.flush(force=True)

    def attach(self):
        self.bus.subscribe("intuition.hypothesis.update", lambda d: self.process(d), priority=4)


# === FACTORY ================================================================

def build_intuition_matrix(bus: IEventBus) -> IntuitionMatrix:
    timing = TimingService()
    cb = CircuitBreaker(timing=timing)
    buffer_mgr = BufferManager()
    dispatcher = EventDispatcher(bus=bus, cb=cb)
    event_buffer = EventBufferLayer(buffer=buffer_mgr, dispatcher=dispatcher, timing=timing)
    weaver = HypothesisWeaver(cb=cb, timing=timing)
    monitor = IntuitionMonitor()

    matrix = IntuitionMatrix(buffer=event_buffer, weaver=weaver, monitor=monitor, bus=bus)
    matrix.attach()
    print("[INIT] Intuition-Matrix 3.4 initialized — full compliance with SPIRIT-SYNTHESIS v2.1")
    return matrix


# === EXECUTOR CLASS ===========================================================

@dataclass
class IntuitionMatrixExecutor:
    """
    Исполнительный класс для работы с INTUITION-MATRIX 3.4
    """
    bus: PriorityEventBus = field(default_factory=PriorityEventBus)
    matrix: IntuitionMatrix = None
    is_initialized: bool = False
    
    def initialize(self) -> bool:
        """Инициализация матрицы"""
        if self.is_initialized:
            return True
            
        try:
            self.matrix = build_intuition_matrix(self.bus)
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"[ERROR] Failed to initialize Intuition-Matrix: {e}")
            return False
    
    def process_signals(self, clarity: float = 0.5, resonance: float = 0.6, confidence: float = 0.7) -> Dict[str, Any]:
        """Обработка интуитивных сигналов"""
        if not self.is_initialized:
            self.initialize()
            
        signals: IntuitionSignals = {
            "clarity": clarity,
            "resonance": resonance,
            "confidence": confidence
        }
        
        self.matrix.process(signals)
        return {"status": "processed", "signals": signals}
    
    def flush_queue(self) -> Dict[str, Any]:
        """Принудительная синхронизация очереди"""
        if not self.is_initialized:
            return {"status": "not_initialized"}
            
        self.matrix.synchronize()
        return {"status": "flushed"}
    
    def get_status(self) -> Dict[str, Any]:
        """Получение статуса матрицы"""
        if not self.is_initialized:
            return {"status": "not_initialized", "health": "unknown"}
            
        report = self.matrix.monitor.report()
        return {
            "status": "active",
            "health": report.get("health", "unknown"),
            "queue_size": report.get("queue_size", 0),
            "failures": report.get("failures", 0)
        }
    
    def subscribe_handler(self, topic: str, handler: Callable, priority: int = 5):
        """Подписка на события шины"""
        self.bus.subscribe(topic, handler, priority)
    
    def emit_event(self, topic: str, data: Dict[str, Any]):
        """Публикация события в шину"""
        self.bus.emit(topic, data)


# === UTILITY FUNCTIONS =======================================================

def create_intuition_executor() -> IntuitionMatrixExecutor:
    """Фабрика для создания исполнителя"""
    executor = IntuitionMatrixExecutor()
    executor.initialize()
    return executor


def test_intuition_matrix():
    """Тестирование работы INTUITION-MATRIX 3.4"""
    print("=== Тестирование INTUITION-MATRIX 3.4 ===")
    
    executor = create_intuition_executor()
    
    # Тест 1: Обработка сигналов
    print("\n1. Обработка интуитивных сигналов...")
    result = executor.process_signals(clarity=0.8, resonance=0.7, confidence=0.9)
    print(f"   Результат: {result}")
    
    # Тест 2: Получение статуса
    print("\n2. Получение статуса...")
    status = executor.get_status()
    print(f"   Статус: {status}")
    
    # Тест 3: Синхронизация
    print("\n3. Синхронизация очереди...")
    flush_result = executor.flush_queue()
    print(f"   Результат: {flush_result}")
    
    # Тест 4: Подписка на событие
    print("\n4. Тест подписки на события...")
    
    def test_handler(data: Dict[str, Any]):
        print(f"   Обработчик получил данные: {data}")
    
    executor.subscribe_handler("test.topic", test_handler)
    executor.emit_event("test.topic", {"message": "Hello, Chokmah!"})
    
    print("\n=== Тестирование завершено ===")
    
    return executor


# === MAIN ENTRY POINT ========================================================

if __name__ == "__main__":
    # Автономный запуск для тестирования
    executor = test_intuition_matrix()
    
    print(f"\nИсполнитель готов. Статус: {executor.get_status()}")
    print("INTUITION-MATRIX 3.4 ожидает интеграции с Sephirotic Engine.")
