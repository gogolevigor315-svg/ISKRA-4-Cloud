#!/usr/bin/env python3
"""
sephirot_base.py - ПОЛНЫЙ КОД СЕФИРОТИЧЕСКОЙ СИСТЕМЫ С ИНТЕГРАЦИЕЙ RAS-CORE
Версия: 5.0.0 Production (с интеграцией 14.4° угла устойчивости)
"""

import json
import logging
import asyncio
import statistics
import inspect
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Deque, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque, defaultdict
from contextlib import asynccontextmanager
import time
import uuid

# ================================================================
# ДОБАВЬ ЭТУ СТРОКУ: ЗАГЛУШКА ДЛЯ КЛАССА Sephirot
# ================================================================
class Sephirot: 
    """Базовый класс сефирот для аннотаций типов."""
    pass

# ================================================================
# ИМПОРТ RAS-CORE КОНСТАНТ
# ================================================================

try:
    from iskra_modules.sephirot_blocks.RAS_CORE.constants import (
        GOLDEN_STABILITY_ANGLE,
        calculate_stability_factor,
        angle_to_priority,
        SEPHIROTIC_TARGETS
    )
    RAS_CORE_AVAILABLE = True
except ImportError:
    print("[WARNING] RAS-CORE constants not available, using defaults")
    GOLDEN_STABILITY_ANGLE = 14.4
    def calculate_stability_factor(deviation): return 1.0
    def angle_to_priority(angle): return 1.0
    SEPHIROTIC_TARGETS = ["KETER", "CHOKMAH", "DAAT", "BINAH", "YESOD", "TIFERET"]
    RAS_CORE_AVAILABLE = False

# ================================================================
# ENERGY LEVEL ENUM
# ================================================================

class EnergyLevel(Enum):
    """Уровни энергии сефиротического узла"""
    CRITICAL = "critical"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    OVERFLOW = "overflow"
    PERFECT = "perfect"

# ================================================================
# БАЗОВЫЕ ИНТЕРФЕЙСЫ ДЛЯ СЕФИРОТ
# ================================================================

class ISephiraModule(ABC):
    """Базовый интерфейс для всех сефирот-модулей"""
    
    @abstractmethod
    async def activate(self) -> Dict[str, Any]:
        raise NotImplementedError
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        raise NotImplementedError
    
    @abstractmethod
    async def receive(self, signal_package: Any) -> Any:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError
    
    @property
    @abstractmethod
    def sephira(self) -> 'Sephirot':
        raise NotImplementedError

# ================================================================
# КОНСТАНТЫ СЕФИРОТИЧЕСКОЙ СИСТЕМЫ
# ================================================================

class Sephirot(Enum):
    """10 сефирот Древа Жизни с RAS-CORE как 11-й узел"""
    KETER = (1, "Венец", "Сознание", "bechtereva")
    CHOKMAH = (2, "Мудрость", "Интуиция", "chernigovskaya")
    BINAH = (3, "Понимание", "Анализ", "bechtereva")
    CHESED = (4, "Милость", "Экспансия", "emotional_weave")
    GEVURAH = (5, "Строгость", "Ограничение", "immune_core")
    TIFERET = (6, "Гармония", "Баланс", "policy_governor")
    NETZACH = (7, "Победа", "Настойчивость", "heartbeat_core")
    HOD = (8, "Слава", "Коммуникация", "polyglossia_adapter")
    YESOD = (9, "Основа", "Подсознание", "spinal_core")
    MALKUTH = (10, "Царство", "Манифестация", "trust_mesh")
    RAS_CORE = (11, "Сетчатка Сознания", "Фокус Внимания", "ras_core")  # НОВЫЙ УЗЕЛ
    
    def __init__(self, level, name, description, connected_module):
        self.level = level
        self.display_name = name
        self.description = description
        self.connected_module = connected_module

# ================================================================
# ТИПЫ И СТРУКТУРЫ
# ================================================================

class SignalType(Enum):
    """Типы сигналов для сефиротической шины"""
    NEURO = auto()
    SEMIOTIC = auto()
    EMOTIONAL = auto()
    COGNITIVE = auto()
    INTENTION = auto()
    HEARTBEAT = auto()
    RESONANCE = auto()
    COMMAND = auto()
    DATA = auto()
    ERROR = auto()
    SYNTHESIS = auto()
    ENERGY = auto()
    SYNC = auto()
    METRIC = auto()
    BROADCAST = auto()
    FEEDBACK = auto()
    CONTROL = auto()
    SEPHIROTIC = auto()
    FOCUS = auto()  # НОВЫЙ ТИП: сигналы фокуса от RAS-CORE
    ATTENTION = auto()  # НОВЫЙ ТИП: сигналы внимания
    
    @classmethod
    def from_string(cls, value: str) -> 'SignalType':
        try:
            return cls[value.upper()]
        except (KeyError, AttributeError):
            return cls.DATA

class NodeStatus(Enum):
    """Статус сефиротического узла"""
    CREATED = "created"
    INITIALIZING = "initializing"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    RECOVERING = "recovering"
    TERMINATING = "terminating"
    TERMINATED = "terminated"

class ResonancePhase(Enum):
    """Фазы резонансной динамики с интеграцией угла 14.4°"""
    SILENT = (0.0, 0.1, "Тишина", 0.1)
    AWAKENING = (0.1, 0.3, "Пробуждение", 0.3)
    COHERENT = (0.3, 0.6, "Когерентность", 0.6)
    RESONANT = (0.6, 0.85, "Резонанс", 0.8)
    PEAK = (0.85, 0.95, "Пик", 0.9)
    TRANSCENDENT = (0.95, 1.0, "Трансценденция", 0.95)
    GOLDEN_STABLE = (0.93, 0.97, "Золотая Устойчивость", 0.95)  # НОВАЯ ФАЗА
    
    def __init__(self, min_val, max_val, description, ideal_point):
        self.min = min_val
        self.max = max_val
        self.description = description
        self.ideal_point = ideal_point
    
    @classmethod
    def from_value(cls, value: float) -> Tuple['ResonancePhase', float]:
        for phase in cls:
            if phase.min <= value <= phase.max:
                distance_to_ideal = abs(value - phase.ideal_point)
                normalized_distance = distance_to_ideal / (phase.max - phase.min)
                return phase, 1.0 - normalized_distance
        return cls.SILENT, 0.0

# ================================================================
# КРИТИЧЕСКИЙ КЛАСС ДЛЯ ИМПОРТА
# ================================================================

@dataclass
class SephiraConfig:
    """Конфигурация для инициализации сефиротического узла"""
    sephira: Sephirot
    bus: Optional[Any] = None
    resonance_init: float = 0.1
    energy_init: float = 0.8
    stability_angle: float = GOLDEN_STABILITY_ANGLE  # НОВОЕ ПОЛЕ
    auto_connect: bool = True
    log_level: str = "INFO"
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> bool:
        if not 0.0 <= self.resonance_init <= 1.0:
            raise ValueError(f"resonance_init must be between 0.0 and 1.0, got {self.resonance_init}")
        if not 0.0 <= self.energy_init <= 1.0:
            raise ValueError(f"energy_init must be between 0.0 and 1.0, got {self.energy_init}")
        if not 0.0 <= self.stability_angle <= 90.0:
            raise ValueError(f"stability_angle must be between 0.0 and 90.0, got {self.stability_angle}")
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log level: {self.log_level}")
        return True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sephira": self.sephira.name,
            "resonance_init": self.resonance_init,
            "energy_init": self.energy_init,
            "stability_angle": self.stability_angle,
            "auto_connect": self.auto_connect,
            "log_level": self.log_level,
            "config_overrides": self.config_overrides
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SephiraConfig':
        sephira_name = data.get("sephira", "KETER")
        sephira = getattr(Sephirot, sephira_name, Sephirot.KETER)

        return cls(
            sephira=sephira,
            bus=None,
            resonance_init=data.get("resonance_init", 0.1),
            energy_init=data.get("energy_init", 0.8),
            stability_angle=data.get("stability_angle", GOLDEN_STABILITY_ANGLE),
            auto_connect=data.get("auto_connect", True),
            log_level=data.get("log_level", "INFO"),
            config_overrides=data.get("config_overrides", {})
        )

# ================================================================
# КВАНТОВАЯ СВЯЗЬ С ИНТЕГРАЦИЕЙ УГЛА УСТОЙЧИВОСТИ
# ================================================================

@dataclass
class QuantumLink:
    """Квантовая связь между узлами с учётом угла устойчивости"""
    target: str
    strength: float = 0.5
    coherence: float = 0.8
    entanglement: float = 0.0
    stability_angle: float = GOLDEN_STABILITY_ANGLE  # НОВОЕ ПОЛЕ
    established: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_sync: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    channel_type: str = "quantum"
    history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=100))
    feedback_loop: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    
    def __post_init__(self):
        self.history.append((self.strength, self.coherence))
    
    def evolve(self, delta_time: float = 1.0) -> Tuple[float, float]:
        """Эволюция связи во времени с учётом угла устойчивости"""
        # Декогеренция зависит от отклонения от золотого угла
        angle_deviation = abs(self.stability_angle - GOLDEN_STABILITY_ANGLE)
        stability_factor = calculate_stability_factor(angle_deviation)
        
        # Меньше декогеренции при близости к 14.4°
        decoherence = 0.05 * delta_time * (1.0 - stability_factor)
        self.coherence = max(0.1, self.coherence - decoherence)
        
        # Самокоррекция с усилением от стабильного угла
        target_strength = 0.6 * stability_factor
        strength_error = target_strength - self.strength
        correction = strength_error * 0.1 * self.coherence * stability_factor
        
        self.strength += correction
        self.strength = max(0.01, min(1.0, self.strength))
        
        # Квантовая запутанность усиливается при стабильном угле
        if stability_factor > 0.7 and self.coherence > 0.7:
            self.entanglement = min(1.0, self.entanglement + 0.01 * delta_time * stability_factor)
        
        self.history.append((self.strength, self.coherence))
        return self.strength, self.coherence
    
    def apply_feedback(self, feedback: float, feedback_angle: float = None) -> float:
        """Применение обратной связи с учётом угла"""
        self.feedback_loop.append(feedback)
        
        if feedback_angle is not None:
            # Корректируем угол связи на основе обратной связи
            angle_correction = (feedback_angle - self.stability_angle) * 0.1
            self.stability_angle += angle_correction
            self.stability_angle = max(0.0, min(90.0, self.stability_angle))
        
        if len(self.feedback_loop) >= 3:
            avg_feedback = statistics.mean(self.feedback_loop)
            correction = (avg_feedback - self.strength) * 0.2
            self.strength += correction
            self.coherence = min(1.0, self.coherence + 0.05)
        
        self.last_sync = datetime.utcnow().isoformat()
        return self.strength
    
    def get_quantum_state(self) -> Dict[str, Any]:
        """Получение квантового состояния с информацией об угле"""
        return {
            "strength": self.strength,
            "coherence": self.coherence,
            "entanglement": self.entanglement,
            "stability_angle": self.stability_angle,
            "stability_factor": calculate_stability_factor(abs(self.stability_angle - GOLDEN_STABILITY_ANGLE)),
            "stability": statistics.stdev([s for s, _ in self.history]) if len(self.history) > 1 else 0.0,
            "age_seconds": (datetime.utcnow() - datetime.fromisoformat(
                self.established.replace('Z', '+00:00')
            )).total_seconds()
        }

# ================================================================
# СИГНАЛЬНЫЙ ПАКЕТ С ИНФОРМАЦИЕЙ ОБ УГЛЕ УСТОЙЧИВОСТИ
# ================================================================

@dataclass
class SignalPackage:
    """Пакет сигнала с полной трассировкой и информацией об угле устойчивости"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: SignalType = SignalType.DATA
    source: str = ""
    target: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    ttl: float = 30.0
    stability_angle: float = GOLDEN_STABILITY_ANGLE  # НОВОЕ ПОЛЕ
    focus_vector: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # Вектор фокуса
    
    def __post_init__(self):
        self.metadata.update({
            "signature": hashlib.sha256(str(self.payload).encode()).hexdigest()[:16],
            "hops": 0,
            "processed_by": [],
            "resonance_trace": [],
            "stability_angle": self.stability_angle,
            "stability_factor": calculate_stability_factor(abs(self.stability_angle - GOLDEN_STABILITY_ANGLE)),
            "focus_vector": self.focus_vector
        })
    
    def add_resonance_trace(self, node: str, resonance: float, node_angle: float = None):
        """Добавление узла в трассировку резонанса с углом"""
        trace_entry = {
            "node": node,
            "resonance": resonance,
            "timestamp": datetime.utcnow().isoformat()
        }
        if node_angle is not None:
            trace_entry["stability_angle"] = node_angle
            trace_entry["stability_factor"] = calculate_stability_factor(abs(node_angle - GOLDEN_STABILITY_ANGLE))
        
        self.metadata["resonance_trace"].append(trace_entry)
    
    def add_processing_node(self, node: str):
        self.metadata["processed_by"].append(node)
        self.metadata["hops"] += 1
    
    def is_expired(self) -> bool:
        created = datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))
        return (datetime.utcnow() - created).total_seconds() > self.ttl
    
    def to_transport_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.name,
            "source": self.source,
            "target": self.target,
            "payload": self.payload,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "ttl": self.ttl,
            "stability_angle": self.stability_angle,
            "focus_vector": self.focus_vector
        }
    
    @classmethod
    def from_transport_dict(cls, data: Dict[str, Any]) -> 'SignalPackage':
        package = cls(
            id=data.get("id", str(uuid.uuid4())),
            type=SignalType.from_string(data.get("type", "DATA")),
            source=data.get("source", ""),
            target=data.get("target", ""),
            payload=data.get("payload", {}),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            ttl=data.get("ttl", 30.0),
            stability_angle=data.get("stability_angle", GOLDEN_STABILITY_ANGLE),
            focus_vector=data.get("focus_vector", [0.0, 0.0, 0.0])
        )
        package.metadata = data.get("metadata", {})
        return package

# ================================================================
# АДАПТИВНАЯ ОЧЕРЕДЬ С УЧЁТОМ УГЛА УСТОЙЧИВОСТИ
# ================================================================

class AdaptiveQueue:
    """Адаптивная очередь с автоочисткой и приоритетом по углу устойчивости"""
    
    def __init__(self, max_size: int = 200, cleanup_interval: float = 5.0):
        self._queue = asyncio.PriorityQueue(maxsize=max_size)
        self._max_size = max_size
        self._cleanup_interval = cleanup_interval
        self._cleanup_task = None
        self._stats = {
            "total_received": 0,
            "total_processed": 0,
            "total_expired": 0,
            "total_dropped": 0,
            "avg_wait_time": 0.0
        }
        self._wait_times = deque(maxlen=100)
    
    async def start(self):
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_worker())
    
    async def stop(self):
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
    
    async def put(self, item: Any, priority: int = 5, stability_angle: float = None) -> bool:
        """Добавление элемента с приоритетом и коррекцией по углу"""
        if stability_angle is not None:
            # Корректируем приоритет на основе угла устойчивости
            angle_factor = angle_to_priority(stability_angle)
            priority = int(priority * angle_factor)
        
        if self._queue.full():
            if await self._make_room():
                await self._queue.put((priority, time.time(), item))
                self._stats["total_received"] += 1
                return True
            self._stats["total_dropped"] += 1
            return False
        
        await self._queue.put((priority, time.time(), item))
        self._stats["total_received"] += 1
        return True
    
    async def get(self) -> Any:
        priority, enqueued_at, item = await self._queue.get()
        wait_time = time.time() - enqueued_at
        self._wait_times.append(wait_time)
        self._stats["total_processed"] += 1
        self._stats["avg_wait_time"] = statistics.mean(self._wait_times) if self._wait_times else 0
        return item
    
    def task_done(self):
        self._queue.task_done()
    
    def qsize(self) -> int:
        return self._queue.qsize()
    
    async def _make_room(self) -> bool:
        temp_items = []
        removed_count = 0
        
        try:
            while not self._queue.empty():
                priority, enqueued_at, item = await self._queue.get()
                
                if time.time() - enqueued_at > 30.0 and priority > 7:
                    removed_count += 1
                    continue
                
                temp_items.append((priority, enqueued_at, item))
            
            for item in temp_items:
                await self._queue.put(item)
            
            self._stats["total_expired"] += removed_count
            return removed_count > 0
            
        except Exception as e:
            for item in temp_items:
                await self._queue.put(item)
            raise
    
    async def _cleanup_worker(self):
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._make_room()
            except asyncio.CancelledError:
                break
            except Exception as e:
                await asyncio.sleep(self._cleanup_interval * 2)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "current_size": self.qsize(),
            "max_size": self._max_size,
            "usage_percent": (self.qsize() / self._max_size) * 100,
            "recent_avg_wait": self._stats["avg_wait_time"]
        }

# ================================================================
# ЯДРО СЕФИРОТИЧЕСКОГО УЗЛА С ИНТЕГРАЦИЕЙ УГЛА 14.4°
# ================================================================

class SephiroticNode(ISephiraModule):
    """
    Сефиротический узел с интеграцией угла устойчивости 14.4°
    """
    
    VERSION = "5.0.0"
    MAX_QUEUE_SIZE = 250
    MAX_MEMORY_LOGS = 500
    DEFAULT_TTL = 60.0
    ENERGY_RECOVERY_RATE = 0.015
    RESONANCE_DECAY_BASE = 0.97
    METRICS_INTERVAL = 3.0
    
    def __init__(self, sephira: Sephirot, bus=None, config: SephiraConfig = None):
        self._sephira = sephira
        self._name = sephira.display_name
        self._level = sephira.level
        self._description = sephira.description
        self._connected_module = sephira.connected_module
        self.bus = bus
        self.config = config or SephiraConfig(sephira=sephira)
        
        # Инициализация состояний с углом устойчивости
        self._initialize_states()
        
        # Структуры данных
        self._initialize_data_structures()
        
        # Системные компоненты
        self._initialize_system_components()
        
        # Запуск инициализации (отложенный - будет запущен явно через start())
        self._init_task = None

    async def initialize_async(self):
        """Асинхронный запуск инициализации узла"""
        if self._init_task is None:
            self._init_task = asyncio.create_task(self._async_initialization())
            self.logger.info(f"✅ Запущена инициализация узла {self._name}")
        elif self._init_task.done() and self.status != NodeStatus.ACTIVE:
            # Если задача завершена, но узел не active - перезапускаем
            self._init_task = asyncio.create_task(self._async_initialization())
            self.logger.warning(f"🔄 Перезапуск инициализации узла {self._name}")
    
        # Ждем завершения инициализации (но не бесконечно)
        try:
            await asyncio.wait_for(self._init_task, timeout=5.0)
            self.logger.info(f"✨ Инициализация узла {self._name} завершена, статус: {self.status.value}")
        except asyncio.TimeoutError:
            self.logger.warning(f"⏳ Инициализация узла {self._name} все еще выполняется (статус: {self.status.value})")
    
        return self._init_task
    
    # ================================================================
    # РЕАЛИЗАЦИЯ ИНТЕРФЕЙСА ISephiraModule
    # ================================================================
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def sephira(self) -> Sephirot:
        return self._sephira
    
    async def activate(self) -> Dict[str, Any]:
        return await self._activate_core()
    
    def get_state(self) -> Dict[str, Any]:
        return self._get_basic_state()
    
    async def receive(self, signal_package: Any) -> Any:
        if isinstance(signal_package, dict):
            signal_package = SignalPackage.from_transport_dict(signal_package)
        return await self.receive_signal(signal_package)
    
    # ================================================================
    # ВНУТРЕННИЕ МЕТОДЫ
    # ================================================================
    
    def _initialize_states(self):
        """Инициализация всех состояний узла с углом устойчивости"""
        self.status = NodeStatus.CREATED
        self.resonance = self.config.resonance_init
        self.energy = self.config.energy_init
        self.stability = 0.9
        self.coherence = 0.7
        self.willpower = 0.6
        self.stability_angle = self.config.stability_angle  # НОВОЕ ПОЛЕ
        self.stability_factor = calculate_stability_factor(
            abs(self.stability_angle - GOLDEN_STABILITY_ANGLE)
        )
        
        # Динамические параметры
        self.activation_time = None
        self.last_metrics_update = None
        self.cycle_count = 0
        self.total_signals_processed = 0
        
        # Флаги состояния
        self._is_initialized = False
        self._is_terminating = False
        self._is_suspended = False
    
    def _initialize_data_structures(self):
        """Инициализация структур данных"""
        self.quantum_links: Dict[str, QuantumLink] = {}
        
        self.signal_queue = AdaptiveQueue(
            max_size=self.MAX_QUEUE_SIZE,
            cleanup_interval=5.0
        )
        
        self.signal_history = deque(maxlen=self.MAX_MEMORY_LOGS)
        self.resonance_history = deque(maxlen=200)
        self.energy_history = deque(maxlen=200)
        self.angle_history = deque(maxlen=200)  # НОВАЯ ИСТОРИЯ
        
        self.response_cache = {}
        self.link_cache = {}
        
        self._signal_counter = defaultdict(int)
        self._processing_times = deque(maxlen=100)
        self._error_log = deque(maxlen=50)
    
    def _initialize_system_components(self):
        """Инициализация системных компонентов"""
        self.logger = self._setup_logger()
        self.signal_handlers = self._initialize_signal_handlers()
        self._background_tasks = set()
        self._shutdown_event = asyncio.Event()
        
        self.metrics = {
            "node": self._name,
            "version": self.VERSION,
            "sephira": self._sephira.value,
            "connected_module": self._connected_module,
            "stability_angle": self.stability_angle,
            "stability_factor": self.stability_factor,
            "start_time": datetime.utcnow().isoformat(),
            "status": self.status.value
        }
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"Sephirot.{self._name}")
        
        if not logger.handlers:
            logger.setLevel(getattr(logging, self.config.log_level))
            
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s:%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            
            console = logging.StreamHandler()
            console.setLevel(logging.WARNING)
            console.setFormatter(formatter)
            logger.addHandler(console)
            
            logger.propagate = False
        
        return logger
    
    def _initialize_signal_handlers(self) -> Dict[SignalType, Callable]:
        """Безопасная инициализация обработчиков — не падает на отсутствующих методах"""
        handlers = {}
    
        # Все возможные типы сигналов, которые могут быть
        handler_map = {
            SignalType.NEURO: "_handle_neuro",
            SignalType.SEMIOTIC: "_handle_semiotic",
            SignalType.EMOTIONAL: "_handle_emotional",
            SignalType.COGNITIVE: "_handle_cognitive",
            SignalType.INTENTION: "_handle_intention",
            SignalType.HEARTBEAT: "_handle_heartbeat",
            SignalType.RESONANCE: "_handle_resonance",
            SignalType.COMMAND: "_handle_command",
            SignalType.DATA: "_handle_data",
            SignalType.ERROR: "_handle_error",
            SignalType.SYNTHESIS: "_handle_synthesis",
            SignalType.ENERGY: "_handle_energy",
            SignalType.SYNC: "_handle_sync",
            SignalType.METRIC: "_handle_metric",
            SignalType.BROADCAST: "_handle_broadcast",
            SignalType.FEEDBACK: "_handle_feedback",
            SignalType.CONTROL: "_handle_control",
            SignalType.SEPHIROTIC: "_handle_sephirotic",
            SignalType.FOCUS: "_handle_focus",
            SignalType.ATTENTION: "_handle_attention",
        }
    
        for signal_type, handler_name in handler_map.items():
            if hasattr(self, handler_name):
                handlers[signal_type] = getattr(self, handler_name)
            else:
                # Используем универсальный дефолтный обработчик
                handlers[signal_type] = self._get_default_handler(handler_name)
                self.logger.warning(f"Обработчик {handler_name} не найден → используем default")
    
        return handlers

    def _get_default_handler(self, handler_name: str):
        """Универсальная заглушка для отсутствующих _handle_* методов"""
        async def default_handler(signal_package: SignalPackage) -> Dict[str, Any]:
            self.logger.warning(f"Default handler triggered for {handler_name} "
                              f"(сигнал от {getattr(signal_package, 'source', 'unknown')})")
            return {
                "status": "default_handled",
                "handler": handler_name,
                "message": f"Method {handler_name} not implemented yet",
                "signal_type": signal_package.type.name if signal_package else "unknown",
                "timestamp": datetime.utcnow().isoformat()
            }
        return default_handler
    
    async def _async_initialization(self):
        """Асинхронная инициализация узла"""
        try:
            self.logger.info(f"Инициализация сефиротического узла {self._name} с углом {self.stability_angle}°")
            self.status = NodeStatus.INITIALIZING
            
            await self.signal_queue.start()
            await self._start_background_tasks()
            
            if self.bus and hasattr(self.bus, 'register_node'):
                await self.bus.register_node(self)
            
            await self._activate_core()
            
            self._is_initialized = True
            self.status = NodeStatus.ACTIVE
            self.activation_time = datetime.utcnow().isoformat()
            
            self.logger.info(f"Сефиротический узел {self._name} активирован (угол: {self.stability_angle}°)")
            
            await self._emit_async(SignalPackage(
                type=SignalType.HEARTBEAT,
                source=self._name,
                payload={
                    "event": "sephirot_activated",
                    "sephira": self._name,
                    "stability_angle": self.stability_angle,
                    "stability_factor": self.stability_factor,
                    "level": self._level,
                    "module": self._connected_module
                }
            ))
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации: {e}")
            self.status = NodeStatus.DEGRADED
            raise
    
    async def _start_background_tasks(self):
        """Запуск фоновых задач"""
        self.logger.info(f"🚀 Запускаю фоновые задачи для {self._name}")
    
        tasks = [
            self._signal_processor,
            self._resonance_dynamics,
            self._energy_manager,
            self._metrics_collector,
            self._link_maintainer,
            self._health_monitor,
            self._angle_stabilizer
        ]
    
        self.logger.info(f"   📊 Всего задач: {len(tasks)}")
    
        for task_func in tasks:
            task = asyncio.create_task(task_func())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            self.logger.info(f"   ✅ Задача {task_func.__name__} запущена")
    
        self.logger.info(f"   ✅ Все {len(tasks)} задач запущены для {self._name}")
    
    # ================================================================
    # НОВЫЕ МЕТОДЫ ДЛЯ РАБОТЫ С УГЛОМ УСТОЙЧИВОСТИ
    # ================================================================
    
    async def _angle_stabilizer(self):
        """Фоновая задача: стабилизация угла узла"""
        self.logger.info(f"Запущен стабилизатор угла для {self._name}")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(5.0)
                
                # Корректируем угол к золотому значению
                angle_deviation = self.stability_angle - GOLDEN_STABILITY_ANGLE
                if abs(angle_deviation) > 1.0:
                    correction = -angle_deviation * 0.1  # Мягкая коррекция
                    self.stability_angle += correction
                    self.stability_factor = calculate_stability_factor(abs(angle_deviation))
                    
                    self.angle_history.append({
                        "timestamp": datetime.utcnow().isoformat(),
                        "old_angle": self.stability_angle - correction,
                        "new_angle": self.stability_angle,
                        "correction": correction,
                        "deviation": angle_deviation,
                        "stability_factor": self.stability_factor
                    })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка в стабилизаторе угла: {e}")
                await asyncio.sleep(10.0)
    
    def adjust_stability_angle(self, new_angle: float) -> Dict[str, Any]:
        """Корректировка угла устойчивости узла"""
        old_angle = self.stability_angle
        self.stability_angle = max(0.0, min(90.0, new_angle))
        self.stability_factor = calculate_stability_factor(abs(self.stability_angle - GOLDEN_STABILITY_ANGLE))
        
        self.angle_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "old_angle": old_angle,
            "new_angle": self.stability_angle,
            "adjustment": new_angle - old_angle,
            "stability_factor": self.stability_factor
        })
        
        # Корректируем все квантовые связи
        for link in self.quantum_links.values():
            link.stability_angle = self.stability_angle
        
        return {
            "status": "angle_adjusted",
            "sephira": self._name,
            "old_angle": old_angle,
            "new_angle": self.stability_angle,
            "stability_factor": self.stability_factor
        }
    
    # ================================================================
    # ОСНОВНЫЕ МЕТОДЫ ОБРАБОТКИ СИГНАЛОВ
    # ================================================================
    
    async def receive_signal(self, signal_package: SignalPackage) -> SignalPackage:
        if not self._is_initialized or self._is_suspended:
            return self._create_error_response(
                signal_package,
                "node_not_ready",
                f"Узел в состоянии: {self.status.value}"
            )
        
        if signal_package.is_expired():
            self.logger.warning(f"Просроченный сигнал: {signal_package.id}")
            return self._create_error_response(signal_package, "signal_expired")
        
        # Расчёт приоритета с учётом угла устойчивости сигнала
        priority = self._calculate_priority(signal_package)
        
        # Добавляем сигнал в очередь с информацией об угле
        queue_success = await self.signal_queue.put(
            signal_package, 
            priority, 
            stability_angle=signal_package.stability_angle
        )
        
        if not queue_success:
            return self._create_error_response(
                signal_package,
                "queue_full",
                "Очередь переполнена"
            )
        
        # Ответ о принятии с информацией об угле
        ack_response = SignalPackage(
            type=SignalType.FEEDBACK,
            source=self._name,
            target=signal_package.source,
            stability_angle=self.stability_angle,
            payload={
                "status": "queued",
                "original_id": signal_package.id,
                "queue_position": self.signal_queue.qsize(),
                "priority": priority,
                "node_stability_angle": self.stability_angle,
                "node_stability_factor": self.stability_factor
            }
        )
        
        return ack_response
    
    async def _signal_processor(self):
        """Процессор сигналов из адаптивной очереди"""
        self.logger.info(f"Процессор сигналов запущен для {self._name} (угол: {self.stability_angle}°)")
        
        while not self._shutdown_event.is_set():
            try:
                signal_package = await self.signal_queue.get()
                start_time = time.perf_counter()
                
                # Добавляем информацию об угле узла в метаданные сигнала
                signal_package.metadata["processing_node_angle"] = self.stability_angle
                signal_package.metadata["processing_node_stability_factor"] = self.stability_factor
                
                response = await self._process_signal_deep(signal_package)
                processing_time = time.perf_counter() - start_time
                
                # Обновление статистики
                self._processing_times.append(processing_time)
                self._signal_counter[signal_package.type.name] += 1
                self.total_signals_processed += 1
                
                # Сохранение в историю с информацией об угле
                signal_package.add_processing_node(self._name)
                signal_package.add_resonance_trace(
                    self._name, 
                    self.resonance,
                    self.stability_angle
                )
                
                self.signal_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "signal": signal_package.id,
                    "type": signal_package.type.name,
                    "processing_time": processing_time,
                    "response_type": response.type.name,
                    "stability_angle": self.stability_angle,
                    "signal_angle": signal_package.stability_angle
                })
                
                # Отправка ответа
                if response.target and self.bus:
                    await self._emit_async(response)
                
                self.signal_queue.task_done()
                
                # Обновление энергетики с учётом угла устойчивости
                energy_cost = processing_time * 0.2 * (1.0 - self.stability_factor)
                self.energy = max(0.1, self.energy - energy_cost)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка в процессоре сигналов: {e}")
                self._error_log.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e),
                    "stability_angle": self.stability_angle
                })
                await asyncio.sleep(0.1)
    
    async def _process_signal_deep(self, signal_package: SignalPackage) -> SignalPackage:
        """
        Глубокая обработка сигнала с учётом угла устойчивости.
        """
        # Проверка кэша с учётом угла сигнала
        cache_key = self._generate_cache_key(signal_package)
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key].copy()
            cached_response.metadata["cached"] = True
            cached_response.metadata["cache_node_angle"] = self.stability_angle
            return cached_response
        
        # Получение обработчика
        handler = self.signal_handlers.get(signal_package.type)
        if not handler:
            handler = self._handle_unknown
        
        # Выполнение обработки
        try:
            handler_result = await handler(signal_package)
            
            # Применение резонансной обратной связи с учётом угла
            resonance_feedback = await self._apply_resonance_feedback(
                signal_package,
                handler_result
            )
            
            # Создание ответа с информацией об угле
            response = SignalPackage(
                type=SignalType.FEEDBACK,
                source=self._name,
                target=signal_package.source,
                stability_angle=self.stability_angle,
                payload={
                    "original_id": signal_package.id,
                    "processed_by": self._name,
                    "handler": signal_package.type.name,
                    "result": handler_result,
                    "resonance_feedback": resonance_feedback,
                    "node_state": {
                        "resonance": self.resonance,
                        "energy": self.energy,
                        "stability": self.stability,
                        "coherence": self.coherence,
                        "stability_angle": self.stability_angle,
                        "stability_factor": self.stability_factor,
                        "willpower": self.willpower
                    },
                    "angle_correction_applied": self._calculate_angle_correction(signal_package)
                }
            )
            
            # Кэширование
            if signal_package.type not in [SignalType.HEARTBEAT, SignalType.METRIC, SignalType.FOCUS, SignalType.ATTENTION]:
                self.response_cache[cache_key] = response
                if len(self.response_cache) > 100:
                    oldest_key = next(iter(self.response_cache))
                    del self.response_cache[oldest_key]
            
            return response
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки сигнала {signal_package.id}: {e}")
            return self._create_error_response(signal_package, "processing_error", str(e))
    
    def _generate_cache_key(self, signal_package: SignalPackage) -> str:
        """Генерация ключа кэша с учётом угла сигнала"""
        content_hash = hashlib.md5(
            json.dumps(signal_package.payload, sort_keys=True).encode()
        ).hexdigest()
        angle_hash = hashlib.md5(str(signal_package.stability_angle).encode()).hexdigest()[:8]
        return f"{signal_package.type.name}:{signal_package.source}:{content_hash}:{angle_hash}"
    
    def _calculate_priority(self, signal_package: SignalPackage) -> int:
        """Расчёт приоритета для очереди с учётом угла устойчивости"""
        priority_map = {
            SignalType.CONTROL: 1,
            SignalType.ERROR: 2,
            SignalType.HEARTBEAT: 3,
            SignalType.SYNC: 4,
            SignalType.FOCUS: 4,      # Приоритет для сигналов фокуса
            SignalType.ATTENTION: 4,   # Приоритет для сигналов внимания
            SignalType.NEURO: 5,
            SignalType.SEMIOTIC: 5,
            SignalType.INTENTION: 5,
            SignalType.RESONANCE: 6,
            SignalType.EMOTIONAL: 7,
            SignalType.COMMAND: 8,
            SignalType.COGNITIVE: 9,
            SignalType.SYNTHESIS: 9,
            SignalType.SEPHIROTIC: 9,
            SignalType.BROADCAST: 10,
            SignalType.FEEDBACK: 10,
            SignalType.DATA: 10,
            SignalType.METRIC: 10,
            SignalType.ENERGY: 10
        }
        
        base_priority = priority_map.get(signal_package.type, 10)
        
        # Корректировка приоритета на основе угла устойчивости сигнала
        angle_factor = angle_to_priority(signal_package.stability_angle)
        resonance_factor = 1.0 - (self.resonance * 0.5)
        stability_factor = self.stability_factor
        
        adjusted_priority = int(base_priority * angle_factor * stability_factor * resonance_factor)
        return max(1, min(10, adjusted_priority))
    
    def _calculate_angle_correction(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Расчёт коррекции угла на основе сигнала"""
        angle_diff = abs(signal_package.stability_angle - self.stability_angle)
        max_correction = 5.0  # Максимальная коррекция в градусах
        
        if angle_diff < 1.0:
            correction = 0.0
            factor = 1.0
        elif angle_diff < 5.0:
            correction = angle_diff * 0.2
            factor = 0.8
        elif angle_diff < 15.0:
            correction = angle_diff * 0.1
            factor = 0.6
        else:
            correction = max_correction * (angle_diff / 45.0)
            factor = 0.4
        
        correction = min(max_correction, correction)
        
        return {
            "angle_difference": angle_diff,
            "suggested_correction": correction,
            "correction_factor": factor,
            "new_angle_suggestion": self.stability_angle + (
                correction if signal_package.stability_angle > self.stability_angle else -correction
            )
        }
    
    # ================================================================
    # НОВЫЕ ОБРАБОТЧИКИ СИГНАЛОВ ДЛЯ RAS-CORE
    # ================================================================
    
    async def _handle_focus(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка сигналов фокуса от RAS-CORE"""
        self.logger.info(f"Обработка FOCUS сигнала от {signal_package.source}")
        
        focus_data = signal_package.payload.get("focus_data", {})
        focus_type = focus_data.get("type", "general")
        intensity = focus_data.get("intensity", 0.5)
        target = focus_data.get("target", "")
        
        # Корректируем угол устойчивости на основе фокуса
        if "suggested_angle" in focus_data:
            suggested_angle = focus_data["suggested_angle"]
            angle_diff = abs(suggested_angle - self.stability_angle)
            if angle_diff > 2.0:  # Корректируем только если разница значительная
                self.stability_angle += (suggested_angle - self.stability_angle) * 0.1
                self.stability_factor = calculate_stability_factor(
                    abs(self.stability_angle - GOLDEN_STABILITY_ANGLE)
                )
        
        processed = {
            "action": "focus_processing",
            "sephira": self._name,
            "focus_type": focus_type,
            "intensity": intensity,
            "target": target,
            "current_stability_angle": self.stability_angle,
            "stability_factor": self.stability_factor,
            "energy_modulation": intensity * 0.1,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Модуляция параметров на основе фокуса
        self.energy = min(1.0, self.energy + intensity * 0.05)
        self.resonance = min(1.0, self.resonance + intensity * 0.03)
        
        return {
            "status": "focus_processed",
            "sephira": self._name,
            "result": processed,
            "energy_boost": intensity * 0.05,
            "resonance_boost": intensity * 0.03
        }
    
    async def _handle_attention(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка сигналов внимания от RAS-CORE"""
        self.logger.info(f"Обработка ATTENTION сигнала от {signal_package.source}")
        
        attention_data = signal_package.payload.get("attention_data", {})
        attention_level = attention_data.get("level", 0.5)
        direction = attention_data.get("direction", "neutral")
        duration = attention_data.get("duration", 1.0)
        
        processed = {
            "action": "attention_processing",
            "sephira": self._name,
            "attention_level": attention_level,
            "direction": direction,
            "duration": duration,
            "current_coherence": self.coherence,
            "stability_impact": attention_level * 0.05,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Влияние внимания на когерентность
        if direction == "toward":
            self.coherence = min(1.0, self.coherence + attention_level * 0.1)
        elif direction == "away":
            self.coherence = max(0.1, self.coherence - attention_level * 0.05)
        
        # Корректировка стабильности
        self.stability = min(1.0, self.stability + attention_level * 0.03)
        
        return {
            "status": "attention_processed",
            "sephira": self._name,
            "result": processed,
            "coherence_change": attention_level * 0.1 if direction == "toward" else -attention_level * 0.05,
            "stability_boost": attention_level * 0.03
        }
    
    async def _handle_neuro(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка нейро-сигналов"""
        self.logger.info(f"Обработка NEURO сигнала от {signal_package.source}")
        
        neuro_data = signal_package.payload.get("neuro_data", {})
        
        # Проверка на инициализацию
        if not hasattr(self, '_is_initialized') or not self._is_initialized:
            return {
                "status": "node_not_initialized",
                "sephira": self._name,
                "action": "deferred",
                "message": "Node not fully initialized, neuro signal deferred"
            }
        
        processed = {
            "action": "neuro_processing",
            "sephira": self._name,
            "neuro_type": neuro_data.get("type", "general"),
            "intensity": neuro_data.get("intensity", 0.5),
            "features": neuro_data.get("features", []),
            "current_stability_angle": self.stability_angle,
            "stability_factor": self.stability_factor,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Модуляция параметров
        self.energy = min(1.0, self.energy + 0.02)
        self.resonance = min(1.0, self.resonance + 0.01)
        
        return {
            "status": "neuro_processed",
            "sephira": self._name,
            "result": processed,
            "energy_boost": 0.02,
            "resonance_boost": 0.01
        }

    async def _handle_semiotic(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка семиотических сигналов"""
        self.logger.info(f"Обработка SEMIOTIC сигнала от {signal_package.source}")
        
        semiotic_data = signal_package.payload.get("semiotic_data", {})
        
        # Проверка на инициализацию
        if not hasattr(self, '_is_initialized') or not self._is_initialized:
            return {
                "status": "node_not_initialized",
                "sephira": self._name,
                "action": "deferred",
                "message": "Node not fully initialized, semiotic signal deferred"
            }
        
        processed = {
            "action": "semiotic_processing",
            "sephira": self._name,
            "semiotic_type": semiotic_data.get("type", "general"),
            "intensity": semiotic_data.get("intensity", 0.5),
            "symbols": semiotic_data.get("symbols", []),
            "current_stability_angle": self.stability_angle,
            "stability_factor": self.stability_factor,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Модуляция параметров
        self.energy = min(1.0, self.energy + 0.015)
        self.resonance = min(1.0, self.resonance + 0.008)
        
        return {
            "status": "semiotic_processed",
            "sephira": self._name,
            "result": processed,
            "energy_boost": 0.015,
            "resonance_boost": 0.008
        }

    async def _handle_emotional(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка эмоциональных сигналов"""
        self.logger.info(f"Обработка EMOTIONAL сигнала от {signal_package.source}")
        
        emotional_data = signal_package.payload.get("emotional_data", {})
        
        # Проверка на инициализацию
        if not hasattr(self, '_is_initialized') or not self._is_initialized:
            return {
                "status": "node_not_initialized",
                "sephira": self._name,
                "action": "deferred",
                "message": "Node not fully initialized, emotional signal deferred"
            }
        
        processed = {
            "action": "emotional_processing",
            "sephira": self._name,
            "emotion_type": emotional_data.get("type", "neutral"),
            "intensity": emotional_data.get("intensity", 0.5),
            "valence": emotional_data.get("valence", 0.0),  # от -1 (негатив) до +1 (позитив)
            "current_stability_angle": self.stability_angle,
            "stability_factor": self.stability_factor,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Модуляция параметров на основе эмоций
        intensity = processed["intensity"]
        valence = processed["valence"]
        
        # Эмоции влияют на резонанс и стабильность
        self.resonance = min(1.0, self.resonance + valence * intensity * 0.02)
        self.coherence = min(1.0, self.coherence + valence * intensity * 0.015)
        self.energy = min(1.0, self.energy + intensity * 0.03)
        
        return {
            "status": "emotional_processed",
            "sephira": self._name,
            "result": processed,
            "resonance_change": valence * intensity * 0.02,
            "coherence_change": valence * intensity * 0.015,
            "energy_boost": intensity * 0.03
        }

    async def _handle_cognitive(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка когнитивных сигналов"""
        self.logger.info(f"Обработка COGNITIVE сигнала от {signal_package.source}")
        
        cognitive_data = signal_package.payload.get("cognitive_data", {})
        
        # Проверка на инициализацию
        if not hasattr(self, '_is_initialized') or not self._is_initialized:
            return {
                "status": "node_not_initialized",
                "sephira": self._name,
                "action": "deferred",
                "message": "Node not fully initialized, cognitive signal deferred"
            }
        
        processed = {
            "action": "cognitive_processing",
            "sephira": self._name,
            "cognitive_type": cognitive_data.get("type", "general"),
            "complexity": cognitive_data.get("complexity", 0.5),
            "depth": cognitive_data.get("depth", 0.5),
            "current_coherence": self.coherence,
            "current_stability_angle": self.stability_angle,
            "stability_factor": self.stability_factor,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Модуляция параметров (когнитивные сигналы усиливают coherence и resonance)
        complexity = processed["complexity"]
        depth = processed["depth"]
        
        self.coherence = min(1.0, self.coherence + complexity * depth * 0.025)
        self.resonance = min(1.0, self.resonance + depth * 0.015)
        self.energy = min(1.0, self.energy - complexity * 0.01)  # когнитивная нагрузка немного расходует энергию
        
        return {
            "status": "cognitive_processed",
            "sephira": self._name,
            "result": processed,
            "coherence_boost": complexity * depth * 0.025,
            "resonance_boost": depth * 0.015,
            "energy_cost": complexity * 0.01
        }

    def __getattr__(self, name):
        if name.startswith('_handle_'):
            if not hasattr(self, '_default_handlers'):
                self._default_handlers = {}
            
            if name not in self._default_handlers:
                async def default_handler(signal_package: SignalPackage) -> Dict[str, Any]:
                    self.logger.warning(f"Default handler used for {name}")
                    return {
                        "status": "default_handled",
                        "handler": name,
                        "message": f"Method {name} not implemented",
                        "signal_type": signal_package.type.name if signal_package else "unknown"
                    }
                self._default_handlers[name] = default_handler
            
            return self._default_handlers[name]
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    # ================================================================
    # СИСТЕМА РЕЗОНАНСНОЙ ОБРАТНОЙ СВЯЗИ С УЧЁТОМ УГЛА
    # ================================================================
    
    async def _apply_resonance_feedback(self, signal_package: SignalPackage, 
                                      handler_result: Any) -> Dict[str, Any]:
        """
        Применение резонансной обратной связи с учётом угла устойчивости.
        """
        phase, phase_perfection = ResonancePhase.from_value(self.resonance)
        
        # Базовая сила обратной связи с поправкой на угол
        base_feedback_strength = self.resonance * phase_perfection
        angle_factor = self.stability_factor
        feedback_strength = base_feedback_strength * angle_factor
        
        type_modifiers = {
            SignalType.NEURO: 1.4,
            SignalType.SEMIOTIC: 1.3,
            SignalType.EMOTIONAL: 1.2,
            SignalType.RESONANCE: 1.5,
            SignalType.SYNTHESIS: 1.4,
            SignalType.INTENTION: 1.1,
            SignalType.ERROR: 0.7,
            SignalType.HEARTBEAT: 0.5,
            SignalType.FOCUS: 1.6,      # Усиление для фокуса
            SignalType.ATTENTION: 1.5   # Усиление для внимания
        }
        
        type_modifier = type_modifiers.get(signal_package.type, 1.0)
        feedback_strength = feedback_strength * type_modifier
        
        # Определение эффекта на основе силы обратной связи и угла
        effect = "stabilize"
        if feedback_strength < 0.3:
            effect = "dampen"
        elif feedback_strength < 0.6:
            effect = "resonate"
        elif feedback_strength < 0.8:
            effect = "amplify"
        else:
            effect = "transcend"
        
        # Добавляем информацию об угле
        angle_info = {
            "node_stability_angle": self.stability_angle,
            "signal_stability_angle": signal_package.stability_angle,
            "angle_difference": abs(self.stability_angle - signal_package.stability_angle),
            "angle_correction_factor": self._calculate_angle_correction_factor(
                self.stability_angle, 
                signal_package.stability_angle
            )
        }
        
        feedback = {
            "strength": min(1.0, feedback_strength),
            "phase": phase.description,
            "phase_perfection": phase_perfection,
            "effect": effect,
            "angle_info": angle_info,
            "suggested_amplification": self._calculate_amplification(feedback_strength),
            "coherence_impact": self.coherence * 0.1 * angle_factor,
            "quantum_correction": self._quantum_correction_value(),
            "stability_factor": self.stability_factor
        }
        
        resonance_delta = feedback_strength * 0.05 - 0.02
        await self._update_resonance_with_feedback(resonance_delta, feedback)
        
        await self._propagate_feedback_to_links(feedback_strength, signal_package.stability_angle)
        
        return feedback
    
    def _calculate_angle_correction_factor(self, node_angle: float, signal_angle: float) -> float:
        """Расчёт фактора коррекции угла"""
        angle_diff = abs(node_angle - signal_angle)
        if angle_diff < 1.0:
            return 1.0
        elif angle_diff < 5.0:
            return 0.9
        elif angle_diff < 15.0:
            return 0.7
        else:
            return 0.5
    
    def _calculate_amplification(self, strength: float) -> float:
        """Расчёт рекомендуемого усиления с учётом угла"""
        if strength < 0.3:
            return 0.5 * self.stability_factor
        elif strength < 0.7:
            return 1.0 * self.stability_factor
        else:
            return (1.0 + (strength - 0.7) * 2) * self.stability_factor
    
    def _quantum_correction_value(self) -> float:
        """Расчёт квантовой поправки с учётом углов связей"""
        if not self.quantum_links:
            return 0.0
        
        try:
            avg_coherence = statistics.mean(
                [link.coherence for link in self.quantum_links.values()]
            )
            avg_entanglement = statistics.mean(
                [link.entanglement for link in self.quantum_links.values()]
            )
            avg_stability_factor = statistics.mean(
                [calculate_stability_factor(abs(link.stability_angle - GOLDEN_STABILITY_ANGLE))
                 for link in self.quantum_links.values()]
            )
            
            return (avg_coherence * 0.5 + avg_entanglement * 0.3 + avg_stability_factor * 0.2) * 0.1
        except:
            return 0.0
    
    async def _update_resonance_with_feedback(self, delta: float, feedback: Dict[str, Any]):
        """Обновление резонанса с учётом обратной связи и угла"""
        # Корректируем дельту на основе фактора устойчивости
        stability_adjusted_delta = delta * self.stability_factor
        
        self.resonance = (
            self.resonance * self.RESONANCE_DECAY_BASE +
            stability_adjusted_delta * (1 - self.RESONANCE_DECAY_BASE)
        )
        self.resonance = max(0.0, min(1.0, self.resonance))
        
        self.resonance_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "value": self.resonance,
            "delta": stability_adjusted_delta,
            "feedback_effect": feedback.get("effect", "unknown"),
            "stability_angle": self.stability_angle,
            "stability_factor": self.stability_factor
        })
        
        # Влияние на когерентность
        self.coherence = min(1.0, self.coherence + abs(stability_adjusted_delta) * 0.05 * self.stability_factor)
    
    async def _propagate_feedback_to_links(self, strength: float, signal_angle: float = None):
        """Распространение обратной связи по связям с учётом угла"""
        if not self.quantum_links:
            return
        
        for link in self.quantum_links.values():
            if link.coherence > 0.5:
                link.apply_feedback(strength * 0.3, signal_angle)
                
                if self.bus and hasattr(self.bus, 'transmit'):
                    feedback_package = SignalPackage(
                        type=SignalType.FEEDBACK,
                        source=self._name,
                        target=link.target,
                        stability_angle=self.stability_angle,
                        payload={
                            "feedback_type": "resonance_propagation",
                            "strength": strength * 0.3,
                            "source_resonance": self.resonance,
                            "source_stability_angle": self.stability_angle,
                            "signal_stability_angle": signal_angle,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    await self._emit_async(feedback_package)
    
    # ================================================================
    # ШИРОКОВЕЩАТЕЛЬНАЯ СИСТЕМА С ИНФОРМАЦИЕЙ ОБ УГЛЕ
    # ================================================================
    
    async def broadcast(self, signal_type: SignalType, payload: Dict[str, Any], 
                       exclude_nodes: List[str] = None) -> int:
        """
        Широковещательная рассылка сигналов с информацией об угле.
        """
        if not self.bus or not hasattr(self.bus, 'broadcast'):
            self.logger.warning("Шина не поддерживает broadcast")
            return 0
        
        signal_package = SignalPackage(
            type=signal_type,
            source=self._name,
            stability_angle=self.stability_angle,
            payload=payload,
            metadata={
                "broadcast_origin": self._name,
                "broadcast_time": datetime.utcnow().isoformat(),
                "origin_stability_angle": self.stability_angle,
                "origin_stability_factor": self.stability_factor
            }
        )
        
        try:
            result = await self.bus.broadcast(signal_package, exclude_nodes or [])
            self.logger.info(f"Broadcast отправлен, достигнуто узлов: {result}")
            return result
        except Exception as e:
            self.logger.error(f"Ошибка broadcast: {e}")
            return 0
    
    async def _emit_async(self, signal_package: SignalPackage) -> bool:
        """
        Асинхронная отправка сигнала через шину.
        """
        if not self.bus or not hasattr(self.bus, 'transmit'):
            return False
        
        try:
            await self.bus.transmit(signal_package)
            return True
        except Exception as e:
            self.logger.error(f"Ошибка отправки сигнала: {e}")
            return False
    
    def _create_error_response(self, original_package: SignalPackage, 
                             error_code: str, 
                             error_message: str = "") -> SignalPackage:
        """Создание ответа с ошибкой"""
        return SignalPackage(
            type=SignalType.ERROR,
            source=self._name,
            target=original_package.source,
            stability_angle=self.stability_angle,
            payload={
                "error_code": error_code,
                "error_message": error_message,
                "original_id": original_package.id,
                "sephira": self._name,
                "node_stability_angle": self.stability_angle,
                "node_stability_factor": self.stability_factor,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    # ================================================================
    # МЕТОДЫ УПРАВЛЕНИЯ ЭНЕРГЕТИКОЙ И РЕЗОНАНСОМ С УЧЁТОМ УГЛА
    # ================================================================
    
    async def _activate_core(self):
        """Активация ядра узла с учётом угла устойчивости"""
        if self._is_initialized:
            return {"status": "already_active", "sephira": self._name}
        
        self.logger.info(f"Активация ядра {self._name} с углом {self.stability_angle}°")
        self.energy = 0.9 * self.stability_factor
        self.resonance = 0.3 * self.stability_factor
        self.coherence = 0.8 * self.stability_factor
        self.stability = 0.9 * self.stability_factor
        self.willpower = 0.7 * self.stability_factor
        
        if self._connected_module:
            await self._create_link(self._connected_module)
        
        return {
            "status": "core_activated", 
            "sephira": self._name,
            "stability_angle": self.stability_angle,
            "stability_factor": self.stability_factor
        }
    
    async def _deactivate_core(self):
        """Деактивация ядра узла"""
        self.logger.info(f"Деактивация ядра {self._name}")
        self.energy = 0.1
        self.resonance = 0.1
        self._is_suspended = True
        
        return {"status": "core_deactivated", "sephira": self._name}
    
    async def _create_link(self, target_node: str) -> Dict[str, Any]:
        """Создание квантовой связи с другим узлом"""
        if target_node in self.quantum_links:
            return {"status": "link_exists", "sephira": self._name}
        
        link = QuantumLink(
            target=target_node,
            stability_angle=self.stability_angle  # Передаём угол узла в связь
        )
        self.quantum_links[target_node] = link
        
        self.logger.info(f"Создана связь с {target_node} (угол: {self.stability_angle}°)")
        
        return {
            "status": "link_created",
            "sephira": self._name,
            "target": target_node,
            "strength": link.strength,
            "coherence": link.coherence,
            "stability_angle": link.stability_angle,
            "stability_factor": calculate_stability_factor(abs(link.stability_angle - GOLDEN_STABILITY_ANGLE))
        }
    
    async def _transfer_energy(self, amount: float, target: str) -> Dict[str, Any]:
        """Передача энергии другому узлу с учётом угла устойчивости"""
        if self.energy < amount + 0.1:
            return {
                "status": "insufficient_energy",
                "sephira": self._name,
                "available": self.energy,
                "requested": amount,
                "stability_factor": self.stability_factor
            }
        
        # Корректируем количество передаваемой энергии на основе угла
        adjusted_amount = amount * self.stability_factor
        self.energy -= adjusted_amount
        
        if self.bus:
            energy_package = SignalPackage(
                type=SignalType.ENERGY,
                source=self._name,
                target=target,
                stability_angle=self.stability_angle,
                payload={
                    "energy_transfer": adjusted_amount,
                    "source_sephira": self._name,
                    "source_stability_angle": self.stability_angle,
                    "stability_factor": self.stability_factor,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            await self._emit_async(energy_package)
        
        return {
            "status": "energy_transferred",
            "sephira": self._name,
            "amount": adjusted_amount,
            "target": target,
            "remaining_energy": self.energy,
            "stability_factor": self.stability_factor
        }
    
    # ================================================================
    # ФОНОВЫЕ ЗАДАЧИ С УЧЁТОМ УГЛА УСТОЙЧИВОСТИ
    # ================================================================
    
    async def _resonance_dynamics(self):
        """Фоновая задача: динамика резонанса с учётом угла"""
        self.logger.info(f"Запущена динамика резонанса для {self._name} (угол: {self.stability_angle}°)")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(2.0)
                
                # Декогеренция зависит от угла устойчивости
                decay_rate = 0.99 * self.stability_factor
                self.resonance *= decay_rate
                
                if self.quantum_links:
                    avg_link_strength = statistics.mean(
                        [link.strength for link in self.quantum_links.values()]
                    )
                    avg_link_stability = statistics.mean(
                        [calculate_stability_factor(abs(link.stability_angle - GOLDEN_STABILITY_ANGLE))
                         for link in self.quantum_links.values()]
                    )
                    
                    resonance_boost = avg_link_strength * 0.01 * avg_link_stability
                    self.resonance = min(1.0, self.resonance + resonance_boost)
                
                for link in self.quantum_links.values():
                    link.evolve(2.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка в динамике резонанса: {e}")
                await asyncio.sleep(5.0)
    
    async def _energy_manager(self):
        """Фоновая задача: управление энергией с учётом угла"""
        self.logger.info(f"Запущен менеджер энергии для {self._name}")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(3.0)
                
                # Восстановление энергии зависит от угла устойчивости
                recovery_rate = self.ENERGY_RECOVERY_RATE * self.stability_factor
                self.energy = min(1.0, self.energy + recovery_rate)
                
                if self.quantum_links:
                    energy_cost = len(self.quantum_links) * 0.005 * (1.0 - self.stability_factor)
                    self.energy = max(0.1, self.energy - energy_cost)
                
                self.energy_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "value": self.energy,
                    "links_count": len(self.quantum_links),
                    "stability_angle": self.stability_angle,
                    "stability_factor": self.stability_factor
                })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка в менеджере энергии: {e}")
                await asyncio.sleep(5.0)
    
    async def _metrics_collector(self):
        """Фоновая задача: сбор метрик с информацией об угле"""
        self.logger.info(f"Запущен сборщик метрик для {self._name}")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.METRICS_INTERVAL)
                
                current_metrics = {
                    "resonance": self.resonance,
                    "energy": self.energy,
                    "coherence": self.coherence,
                    "stability": self.stability,
                    "willpower": self.willpower,
                    "stability_angle": self.stability_angle,
                    "stability_factor": self.stability_factor,
                    "active_links": len(self.quantum_links),
                    "queue_size": self.signal_queue.qsize(),
                    "signals_processed": self.total_signals_processed,
                    "cycle_count": self.cycle_count,
                    "status": self.status.value
                }
                
                self.metrics.update(current_metrics)
                self.metrics["last_update"] = datetime.utcnow().isoformat()
                
                self.cycle_count += 1
                
                if self.bus and self.cycle_count % 10 == 0:
                    metrics_package = SignalPackage(
                        type=SignalType.METRIC,
                        source=self._name,
                        stability_angle=self.stability_angle,
                        payload={
                            "sephira": self._name,
                            "metrics": current_metrics,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    await self._emit_async(metrics_package)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка в сборщике метрик: {e}")
                await asyncio.sleep(self.METRICS_INTERVAL * 2)
    
    async def _link_maintainer(self):
        """Фоновая задача: обслуживание связей"""
        self.logger.info(f"Запущен обслуживатель связей для {self._name}")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(10.0)
                
                links_to_remove = []
                for target, link in self.quantum_links.items():
                    if link.strength < 0.1:
                        links_to_remove.append(target)
                
                for target in links_to_remove:
                    del self.quantum_links[target]
                    self.logger.info(f"Удалена слабая связь с {target}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка в обслуживателе связей: {e}")
                await asyncio.sleep(15.0)
    
    async def _health_monitor(self):
        """Фоновая задача: мониторинг здоровья с учётом угла"""
        self.logger.info(f"Запущен мониторинг здоровья для {self._name}")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(5.0)
                
                if self.energy < 0.2:
                    self.status = NodeStatus.DEGRADED
                    self.logger.warning(f"Низкая энергия: {self.energy} (угол: {self.stability_angle}°)")
                elif self.energy < 0.1:
                    self.status = NodeStatus.OVERLOADED
                    self.logger.error(f"Критически низкая энергия: {self.energy} (угол: {self.stability_angle}°)")
                else:
                    self.status = NodeStatus.ACTIVE
                
                if self._error_log:
                    recent_errors = list(self._error_log)[-5:]
                    self.logger.debug(f"Последние 5 ошибок: {recent_errors}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка в мониторинге здоровья: {e}")
                await asyncio.sleep(10.0)
    
    # ================================================================
    # API МЕТОДЫ ДЛЯ ВНЕШНЕГО ДОСТУПА С ИНФОРМАЦИЕЙ ОБ УГЛЕ
    # ================================================================
    
    async def shutdown(self):
        """Корректное завершение работы узла"""
        self.logger.info(f"Завершение работы узла {self._name} (угол: {self.stability_angle}°)")
        self._is_terminating = True
        self.status = NodeStatus.TERMINATING
        
        self._shutdown_event.set()
        await asyncio.sleep(0.5)
        
        await self.signal_queue.stop()
        
        if self._init_task and not self._init_task.done():
            self._init_task.cancel()
        
        self.status = NodeStatus.TERMINATED
        self.logger.info(f"Узел {self._name} завершил работу")
        
        return {"status": "shutdown_complete", "sephira": self._name}
    
    def _get_basic_state(self) -> Dict[str, Any]:
        """Получение базового состояния узла с информацией об угле"""
        return {
            "name": self._name,
            "sephira": self._sephira.name,
            "level": self._level,
            "description": self._description,
            "connected_module": self._connected_module,
            "status": self.status.value,
            "resonance": self.resonance,
            "energy": self.energy,
            "coherence": self.coherence,
            "stability": self.stability,
            "willpower": self.willpower,
            "stability_angle": self.stability_angle,
            "stability_factor": self.stability_factor,
            "activation_time": self.activation_time,
            "active_links": [link.target for link in self.quantum_links.values()],
            "queue_size": self.signal_queue.qsize(),
            "total_signals_processed": self.total_signals_processed,
            "metrics": self.metrics
        }
    
    def _get_detailed_state(self) -> Dict[str, Any]:
        """Получение детального состояния с полной информацией об угле"""
        state = self._get_basic_state()
        state.update({
            "quantum_links": {
                target: link.get_quantum_state()
                for target, link in self.quantum_links.items()
            },
            "signal_stats": dict(self._signal_counter),
            "queue_stats": self.signal_queue.get_stats(),
            "recent_errors": list(self._error_log)[-10:],
            "resonance_history": list(self.resonance_history)[-20:],
            "energy_history": list(self.energy_history)[-20:],
            "angle_history": list(self.angle_history)[-20:],
            "signal_history": list(self.signal_history)[-10:],
            "processing_times": list(self._processing_times)[-10:],
            "background_tasks": len(self._background_tasks),
            "is_initialized": self._is_initialized,
            "is_terminating": self._is_terminating,
            "is_suspended": self._is_suspended,
            "golden_stability_angle": GOLDEN_STABILITY_ANGLE,
            "angle_deviation": abs(self.stability_angle - GOLDEN_STABILITY_ANGLE)
        })
        return state
    
    async def connect_to_module(self, module_name: str) -> Dict[str, Any]:
        """Явное подключение к модулю с передачей угла"""
        result = await self._create_link(module_name)
        result["node_stability_angle"] = self.stability_angle
        return result
    
    async def boost_energy(self, amount: float = 0.2) -> Dict[str, Any]:
        """Увеличение энергии узла с учётом угла"""
        old_energy = self.energy
        adjusted_amount = amount * self.stability_factor
        self.energy = min(1.0, self.energy + adjusted_amount)
        
        self.energy_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "old": old_energy,
            "new": self.energy,
            "delta": adjusted_amount,
            "type": "manual_boost",
            "source": "external",
            "stability_angle": self.stability_angle,
            "stability_factor": self.stability_factor
        })
        
        return {
            "status": "energy_boosted",
            "sephira": self._name,
            "amount": adjusted_amount,
            "old_energy": old_energy,
            "new_energy": self.energy,
            "stability_angle": self.stability_angle,
            "stability_factor": self.stability_factor
        }
    
    async def set_resonance(self, value: float) -> Dict[str, Any]:
        """Установка резонанса с учётом угла"""
        old_value = self.resonance
        adjusted_value = value * self.stability_factor
        self.resonance = max(0.0, min(1.0, adjusted_value))
        
        self.resonance_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "old": old_value,
            "new": self.resonance,
            "delta": adjusted_value - old_value,
            "source": "manual_set",
            "stability_angle": self.stability_angle
        })
        
        return {
            "status": "resonance_set",
            "sephira": self._name,
            "old_value": old_value,
            "new_value": self.resonance,
            "stability_angle": self.stability_angle,
            "stability_factor": self.stability_factor
        }
    
    async def get_health_report(self) -> Dict[str, Any]:
        """Получение отчёта о здоровье узла с учётом угла"""
        phase, phase_perfection = ResonancePhase.from_value(self.resonance)
        
        # Оценка здоровья на основе угла
        angle_health = "good"
        angle_deviation = abs(self.stability_angle - GOLDEN_STABILITY_ANGLE)
        if angle_deviation < 2.0:
            angle_health = "excellent"
        elif angle_deviation < 5.0:
            angle_health = "good"
        elif angle_deviation < 10.0:
            angle_health = "warning"
        else:
            angle_health = "critical"
        
        return {
            "status": self.status.value,
            "sephira": self._name,
            "health_indicators": {
                "energy": {
                    "value": self.energy,
                    "status": "good" if self.energy > 0.5 else "warning" if self.energy > 0.2 else "critical"
                },
                "resonance": {
                    "value": self.resonance,
                    "phase": phase.description,
                    "phase_perfection": phase_perfection,
                    "status": "good" if self.resonance > 0.5 else "warning" if self.resonance > 0.2 else "critical"
                },
                "coherence": {
                    "value": self.coherence,
                    "status": "good" if self.coherence > 0.7 else "warning" if self.coherence > 0.4 else "critical"
                },
                "stability": {
                    "value": self.stability,
                    "status": "good" if self.stability > 0.7 else "warning" if self.stability > 0.4 else "critical"
                },
                "stability_angle": {
                    "value": self.stability_angle,
                    "deviation": angle_deviation,
                    "golden_angle": GOLDEN_STABILITY_ANGLE,
                    "factor": self.stability_factor,
                    "status": angle_health
                }
            },
            "active_connections": len(self.quantum_links),
            "signals_processed": self.total_signals_processed,
            "queue_status": {
                "current": self.signal_queue.qsize(),
                "max": self.MAX_QUEUE_SIZE,
                "percent": (self.signal_queue.qsize() / self.MAX_QUEUE_SIZE) * 100
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def reset_node(self) -> Dict[str, Any]:
        """Сброс узла к начальному состоянию с сохранением угла"""
        self.logger.info(f"Сброс узла {self._name} (угол: {self.stability_angle}°)")
        
        old_state = self._get_basic_state()
        
        await self.shutdown()
        
        # Сохраняем угол устойчивости
        saved_angle = self.stability_angle
        
        self._initialize_states()
        self._initialize_data_structures()
        
        # Восстанавливаем угол
        self.stability_angle = saved_angle
        self.stability_factor = calculate_stability_factor(abs(saved_angle - GOLDEN_STABILITY_ANGLE))
        
        self._init_task = asyncio.create_task(self._async_initialization())
        
        return {
            "status": "node_reset",
            "sephira": self._name,
            "old_state": old_state,
            "new_state": self._get_basic_state(),
            "preserved_stability_angle": saved_angle
        }

# ================================================================
# СЕФИРОТИЧЕСКОЕ ДЕРЕВО С ИНТЕГРАЦИЕЙ RAS-CORE
# ================================================================

class SephiroticTree:
    """
    Древо Жизни - все 10 сефирот + RAS-CORE как единая система.
    """
    
    def __init__(self, bus=None, ras_core=None):
        self.bus = bus
        self.ras_core = ras_core  # Ссылка на экземпляр RAS-CORE
        self.nodes: Dict[str, SephiroticNode] = {}
        self.initialized = False
        self.logger = logging.getLogger("Sephirotic.Tree")
        
    async def initialize(self):
        """Инициализация всех сефирот и подключение RAS-CORE"""
        print("\n" + "="*60)
        print("🌳 НАЧАЛО ИНИЦИАЛИЗАЦИИ ДЕРЕВА")
        print("="*60 + "\n")
    
        try:
            # Шаг 1: Проверка состояния
            print("📋 ШАГ 1: Проверка состояния")
            if self.initialized:
                print("   ✅ Дерево уже инициализировано")
                return
            print("   ✅ Начинаем инициализацию")
        
            self.logger.info("Инициализация Сефиротического Древа с интеграцией RAS-CORE")
        
            # Шаг 2: Создание сефирот
            print("\n📋 ШАГ 2: Создание сефиротических узлов")
        
            # Счетчики для статистики
            created_nodes = 0
            failed_nodes = 0
        
            for sephira in Sephirot:
                if sephira == Sephirot.RAS_CORE:
                    print(f"\n   ⏭️  Пропускаем RAS-CORE (будет добавлен отдельно)")
                    continue
                
                print(f"\n   🔹 Создаю узел: {sephira.name}")
                try:
                    # Создаем конфиг
                    config = SephiraConfig(
                        sephira=sephira,
                        bus=self.bus,
                        stability_angle=GOLDEN_STABILITY_ANGLE
                    )
                    print(f"      ✅ Конфиг создан")
                
                    # Создаем узел
                    print(f"      ⏳ Вызываю SephiroticNode()...")
                    node = SephiroticNode(sephira, self.bus, config)
                    print(f"      ✅ Узел создан успешно")
                    
                    # 🔥 ЯВНО ЗАПУСКАЕМ ИНИЦИАЛИЗАЦИЮ УЗЛА
                    await node.initialize_async()
                    print(f"      ✅ Узел инициализирован (статус: {node.status.value})")
                
                    # Сохраняем в словарь
                    self.nodes[sephira.name] = node
                    print(f"      ✅ Узел сохранен в self.nodes")
                    created_nodes += 1
                
                except Exception as e:
                    print(f"      ❌ ОШИБКА создания узла {sephira.name}: {e}")
                    import traceback
                    traceback.print_exc()
                    failed_nodes += 1
                    # Не падаем, продолжаем с другими узлами
        
            print(f"\n📊 Статистика создания узлов:")
            print(f"   ✅ Успешно создано: {created_nodes}")
            print(f"   ❌ Ошибок: {failed_nodes}")
        
            # Шаг 3: Установка связей
            print("\n📋 ШАГ 3: Установка сефиротических связей")
            try:
                print("   ⏳ Вызываю _establish_sephirotic_connections()...")
                await self._establish_sephirotic_connections()
                print("   ✅ Связи установлены")
            except Exception as e:
                print(f"   ❌ Ошибка при установке связей: {e}")
                import traceback
                traceback.print_exc()
                # Не падаем, продолжаем
        
            # Шаг 4: Интеграция RAS-CORE
            print("\n📋 ШАГ 4: Интеграция RAS-CORE")
            if self.ras_core:
                try:
                    print("   ⏳ Вызываю _integrate_ras_core()...")
                    await self._integrate_ras_core()
                    print("   ✅ RAS-CORE интегрирован")
                except Exception as e:
                    print(f"   ❌ Ошибка при интеграции RAS-CORE: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("   ⚠️ ras_core отсутствует, пропускаем интеграцию")
        
            # Шаг 5: Финальная проверка
            print("\n📋 ШАГ 5: Финальная проверка")
            self.initialized = True
        
            # Подсчет активных узлов
            active_nodes = len(self.nodes)
        
            print(f"\n" + "="*60)
            print(f"🎯 РЕЗУЛЬТАТ ИНИЦИАЛИЗАЦИИ:")
            print(f"   ✅ Всего узлов в дереве: {active_nodes}")
            print(f"   ✅ Узлы: {list(self.nodes.keys())}")
            print("="*60 + "\n")
        
            self.logger.info(f"Сефиротическое Древо инициализировано с {active_nodes} узлами")
        
            # Возвращаем результат для блока форсированной активации
            return {
                "activated_nodes": active_nodes,
                "total_resonance": 0.9  # Базовое значение
            }
        
        except Exception as e:
            print("\n" + "🔥"*60)
            print(f"🔥 КРИТИЧЕСКАЯ ОШИБКА В ИНИЦИАЛИЗАЦИИ ДЕРЕВА:")
            print(f"🔥 {e}")
            print("🔥"*60)
            import traceback
            traceback.print_exc()
            print("🔥"*60 + "\n")
        
            self.logger.error(f"Критическая ошибка инициализации дерева: {e}")
            return None
    
    async def _establish_sephirotic_connections(self):
        """Установка канонических связей между сефиротами"""
        connections = {
            "KETER": ["CHOKMAH", "BINAH"],
            "CHOKMAH": ["BINAH"],
            "BINAH": ["CHESED", "GEVURAH"],
            "CHESED": ["TIFERET"],
            "GEVURAH": ["TIFERET"],
            "TIFERET": ["NETZACH", "HOD", "YESOD"],
            "NETZACH": ["HOD", "YESOD"],
            "HOD": ["YESOD"],
            "YESOD": ["MALKUTH"],
            "MALKUTH": []
        }
        
        for source, targets in connections.items():
            if source in self.nodes:
                for target in targets:
                    if target in self.nodes:
                        await self.nodes[source]._create_link(target)
    
    async def _integrate_ras_core(self):
        """Интеграция RAS-CORE в дерево сефирот"""
        if not self.ras_core:
            return
        
        self.logger.info("Интеграция RAS-CORE в Сефиротическое Древо")
        
        # Создаём специальный узел для RAS-CORE
        ras_config = SephiraConfig(
            sephira=Sephirot.RAS_CORE,
            bus=self.bus,
            stability_angle=GOLDEN_STABILITY_ANGLE
        )
        ras_node = SephiroticNode(Sephirot.RAS_CORE, self.bus, ras_config)
        self.nodes["RAS_CORE"] = ras_node
        
        # Устанавливаем связи с ключевыми сефиротами
        ras_connections = {
            "RAS_CORE": ["KETER", "CHOKMAH", "DAAT", "BINAH", "YESOD"],
            "KETER": ["RAS_CORE"],
            "CHOKMAH": ["RAS_CORE"],
            "BINAH": ["RAS_CORE"],
            "YESOD": ["RAS_CORE"]
        }
        
        # Создаём связи
        connections_established = 0
        for source, targets in ras_connections.items():
            if source in self.nodes:
                for target in targets:
                    if target in self.nodes:
                        await self.nodes[source]._create_link(target)
                        connections_established += 1
        
        self.logger.info(f"RAS-CORE интегрирован, установлено {connections_established} связей")
        
        # Активируем RAS узел
        await ras_node._activate_core()
    
    async def activate_all(self):
        """Активация всех сефирот и RAS-CORE"""
        self.logger.info("Активация всех сефирот и RAS-CORE")
        
        activation_results = {}
        for name, node in self.nodes.items():
            if node.status != NodeStatus.ACTIVE:
                result = await node._activate_core()
                activation_results[name] = result
        
        self.logger.info("Все сефироты и RAS-CORE активированы")
        return {
            "status": "all_activated", 
            "count": len(self.nodes),
            "results": activation_results
        }
    
    async def shutdown_all(self):
        """Завершение работы всех сефирот и RAS-CORE"""
        self.logger.info("Завершение работы всех сефирот и RAS-CORE")
        
        shutdown_results = {}
        for name, node in self.nodes.items():
            if node.status != NodeStatus.TERMINATED:
                result = await node.shutdown()
                shutdown_results[name] = result
        
        self.initialized = False
        self.logger.info("Все сефироты и RAS-CORE завершили работу")
        return {
            "status": "all_shutdown", 
            "count": len(self.nodes),
            "results": shutdown_results
        }
    
    def get_node(self, name: str) -> Optional[SephiroticNode]:
        """Получение узла по имени"""
        return self.nodes.get(name.upper())
    
    def get_tree_state(self) -> Dict[str, Any]:
        """Получение состояния всего дерева с информацией об углах"""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        nodes_state = {}
        total_energy = 0.0
        total_resonance = 0.0
        total_coherence = 0.0
        total_stability_factor = 0.0
        
        for name, node in self.nodes.items():
            state = node._get_basic_state()
            nodes_state[name] = state
            total_energy += state["energy"]
            total_resonance += state["resonance"]
            total_coherence += state["coherence"]
            total_stability_factor += state.get("stability_factor", 0.5)
        
        node_count = len(self.nodes)
        avg_energy = total_energy / node_count if node_count > 0 else 0.0
        avg_resonance = total_resonance / node_count if node_count > 0 else 0.0
        avg_coherence = total_coherence / node_count if node_count > 0 else 0.0
        avg_stability_factor = total_stability_factor / node_count if node_count > 0 else 0.0
        
        overall_status = "healthy"
        if avg_energy < 0.3:
            overall_status = "critical"
        elif avg_energy < 0.6:
            overall_status = "warning"
        
        return {
            "status": "active",
            "overall_status": overall_status,
            "initialized": True,
            "node_count": node_count,
            "total_energy": total_energy,
            "total_resonance": total_resonance,
            "avg_energy": avg_energy,
            "avg_resonance": avg_resonance,
            "avg_coherence": avg_coherence,
            "avg_stability_factor": avg_stability_factor,
            "tree_health": {
                "energy_score": avg_energy,
                "resonance_score": avg_resonance,
                "coherence_score": avg_coherence,
                "stability_score": avg_stability_factor
            },
            "ras_core_integrated": "RAS_CORE" in self.nodes,
            "nodes": nodes_state
        }
    
    def get_detailed_tree_state(self) -> Dict[str, Any]:
        """Получение детального состояния дерева"""
        base_state = self.get_tree_state()
        
        if base_state["status"] == "not_initialized":
            return base_state
        
        detailed_nodes = {}
        for name, node in self.nodes.items():
            detailed_nodes[name] = node._get_detailed_state()
        
        base_state["detailed_nodes"] = detailed_nodes
        return base_state
    
    async def broadcast_to_tree(self, signal_type: SignalType, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Широковещательная рассылка по всему дереву"""
        if not self.initialized:
            return {"status": "tree_not_initialized"}
        
        results = {}
        for name, node in self.nodes.items():
            signal_package = SignalPackage(
                type=signal_type,
                source="SephiroticTree",
                target=name,
                payload=payload
            )
            response = await node.receive_signal(signal_package)
            results[name] = response.payload
        
        return {
            "status": "broadcast_completed",
            "nodes_reached": len(results),
            "results": results
        }
    
    async def send_focus_signal(self, target_sephira: str, focus_data: Dict[str, Any]) -> Dict[str, Any]:
        """Отправка сигнала фокуса к конкретной сефире"""
        if target_sephira not in self.nodes:
            return {"status": "sephira_not_found", "target": target_sephira}
        
        signal_package = SignalPackage(
            type=SignalType.FOCUS,
            source="SephiroticTree",
            target=target_sephira,
            payload={"focus_data": focus_data}
        )
        
        node = self.nodes[target_sephira]
        response = await node.receive_signal(signal_package)
        
        return {
            "status": "focus_sent",
            "target": target_sephira,
            "response": response.payload
        }

# ================================================================
# СИНГЛТОН ДВИЖКА СЕФИРОТИЧЕСКОЙ СИСТЕМЫ С RAS-CORE
# ================================================================

class SephiroticEngine:
    """
    Движок сефиротической системы - единая точка доступа с интеграцией RAS-CORE.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.tree = None
            self.bus = None
            self.ras_core = None
            self.initialized = False
            self.logger = logging.getLogger("Sephirotic.Engine")
    
    async def initialize(self, bus=None, ras_core=None):
        """Инициализация движка с интеграцией RAS-CORE"""
        if self.initialized:
            return
        
        self.logger.info("Инициализация SephiroticEngine с интеграцией RAS-CORE")
        self.bus = bus
        self.ras_core = ras_core
        self.tree = SephiroticTree(bus, ras_core)
        
        await self.tree.initialize()
        self.initialized = True
        
        self.logger.info("SephiroticEngine готов к работе с RAS-CORE")
    
    async def activate(self):
        """Активация сефиротической системы с RAS-CORE"""
        if not self.initialized:
            await self.initialize(self.bus, self.ras_core)
        
        result = await self.tree.activate_all()
        
        if self.bus and hasattr(self.bus, 'broadcast'):
            activation_package = SignalPackage(
                type=SignalType.SEPHIROTIC,
                source="SephiroticEngine",
                payload={
                    "action": "tree_activated",
                    "total_nodes": len(self.tree.nodes),
                    "ras_core_integrated": self.ras_core is not None,
                    "golden_stability_angle": GOLDEN_STABILITY_ANGLE,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            await self.bus.broadcast(activation_package)
        
        self.logger.info("Сефиротическая система с RAS-CORE активирована")
        return result
    
    async def shutdown(self):
        """Завершение работы сефиротической системы"""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        result = await self.tree.shutdown_all()
        self.initialized = False
        
        self.logger.info("Сефиротическая система завершила работу")
        return result
    
    def get_state(self) -> Dict[str, Any]:
        """Получение состояния движка с информацией о RAS-CORE"""
        if not self.initialized:
            return {
                "status": "not_initialized",
                "engine": "SephiroticEngine",
                "version": "5.0.0",
                "ras_core_available": self.ras_core is not None,
                "golden_stability_angle": GOLDEN_STABILITY_ANGLE,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        tree_state = self.tree.get_tree_state()
        
        return {
            "status": "active",
            "engine": "SephiroticEngine",
            "version": "5.0.0",
            "tree": tree_state,
            "bus_connected": self.bus is not None,
            "ras_core_connected": self.ras_core is not None,
            "initialized": self.initialized,
            "golden_stability_angle": GOLDEN_STABILITY_ANGLE,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_detailed_state(self) -> Dict[str, Any]:
        """Получение детального состояния"""
        if not self.initialized:
            return self.get_state()
        
        base_state = self.get_state()
        detailed_tree = self.tree.get_detailed_tree_state()
        
        base_state["detailed_tree"] = detailed_tree
        return base_state
    
    def get_node(self, name: str) -> Optional[SephiroticNode]:
        """Получение сефиротического узла по имени"""
        if not self.initialized or not self.tree:
            return None
        
        return self.tree.get_node(name)
    
    async def broadcast_to_tree(self, signal_type: SignalType, payload: Dict[str, Any]) -> int:
        """Широковещательная рассылка по всему дереву"""
        if not self.initialized:
            return 0
        
        result = await self.tree.broadcast_to_tree(signal_type, payload)
        return result.get("nodes_reached", 0)
    
    async def connect_module_to_sephira(self, module_name: str, sephira_name: str) -> Dict[str, Any]:
        """Подключение модуля к конкретной сефире"""
        if not self.initialized:
            return {"status": "engine_not_initialized"}
        
        node = self.get_node(sephira_name)
        if not node:
            return {"status": "sephira_not_found", "sephira": sephira_name}
        
        result = await node.connect_to_module(module_name)
        self.logger.info(f"Модуль {module_name} подключен к сефире {sephira_name}")
        
        return {
            "status": "module_connected",
            "module": module_name,
            "sephira": sephira_name,
            "connection_result": result
        }
    
    async def get_node_health(self, sephira_name: str) -> Dict[str, Any]:
        """Получение отчёта о здоровье конкретного узла"""
        node = self.get_node(sephira_name)
        if not node:
            return {"status": "node_not_found", "sephira": sephira_name}
        
        return await node.get_health_report()
    
    async def reset_node(self, sephira_name: str) -> Dict[str, Any]:
        """Сброс конкретного узла"""
        node = self.get_node(sephira_name)
        if not node:
            return {"status": "node_not_found", "sephira": sephira_name}
        
        return await node.reset_node()
    
    async def send_focus_to_sephira(self, sephira_name: str, focus_data: Dict[str, Any]) -> Dict[str, Any]:
        """Отправка сигнала фокуса к конкретной сефире"""
        if not self.initialized or not self.tree:
            return {"status": "engine_not_initialized"}
        
        return await self.tree.send_focus_signal(sephira_name, focus_data)
    
    async def adjust_node_stability_angle(self, sephira_name: str, new_angle: float) -> Dict[str, Any]:
        """Корректировка угла устойчивости узла"""
        node = self.get_node(sephira_name)
        if not node:
            return {"status": "node_not_found", "sephira": sephira_name}
        
        result = node.adjust_stability_angle(new_angle)
        self.logger.info(f"Угол устойчивости узла {sephira_name} изменён на {new_angle}°")
        
        return result

# ================================================================
# СЕФИРОТИЧЕСКАЯ ШИНА (SephiroticBus) С ПОДДЕРЖКОЙ FOCUS СИГНАЛОВ
# ================================================================

class SephiroticBus:
    """
    Шина для связи между сефиротическими узлами и модулями.
    """
    
    def __init__(self):
        self.nodes: Dict[str, SephiroticNode] = {}
        self.subscriptions: Dict[SignalType, List[Callable]] = defaultdict(list)
        self.message_log = deque(maxlen=1000)
        self.focus_log = deque(maxlen=200)  # Лог фокус-сигналов
        self.logger = logging.getLogger("Sephirotic.Bus")
    
    async def register_node(self, node: SephiroticNode):
        """Регистрация узла в шине"""
        self.nodes[node.name] = node
        self.logger.info(f"Узел {node.name} зарегистрирован в шине (угол: {node.stability_angle}°)")
    
    async def transmit(self, signal_package: SignalPackage) -> bool:
        """Передача сигнала конкретному узлу"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": signal_package.type.name,
            "source": signal_package.source,
            "target": signal_package.target,
            "id": signal_package.id,
            "payload_size": len(str(signal_package.payload)),
            "stability_angle": signal_package.stability_angle
        }
        
        self.message_log.append(log_entry)
        
        # Специальная обработка для фокус-сигналов
        if signal_package.type in [SignalType.FOCUS, SignalType.ATTENTION]:
            self.focus_log.append({
                **log_entry,
                "focus_type": signal_package.payload.get("focus_data", {}).get("type", "unknown"),
                "intensity": signal_package.payload.get("focus_data", {}).get("intensity", 0.0)
            })
        
        if signal_package.target:
            if signal_package.target in self.nodes:
                target_node = self.nodes[signal_package.target]
                await target_node.receive_signal(signal_package)
                return True
            else:
                self.logger.warning(f"Целевой узел не найден: {signal_package.target}")
                return False
        
        delivered = False
        for callback in self.subscriptions.get(signal_package.type, []):
            try:
                await callback(signal_package)
                delivered = True
            except Exception as e:
                self.logger.error(f"Ошибка в обработчике подписки: {e}")
        
        return delivered
    
    async def broadcast(self, signal_package: SignalPackage, exclude_nodes: List[str] = None) -> int:
        """Широковещательная рассылка всем узлам"""
        exclude_set = set(exclude_nodes or [])
        count = 0
        
        for name, node in self.nodes.items():
            if name in exclude_set or name == signal_package.source:
                continue
            
            try:
                await node.receive_signal(signal_package)
                count += 1
            except Exception as e:
                self.logger.error(f"Ошибка при broadcast узлу {name}: {e}")
        
        self.logger.info(f"Broadcast доставлен {count} узлам")
        return count
    
    def subscribe(self, signal_type: SignalType, callback: Callable):
        """Подписка на тип сигнала"""
        self.subscriptions[signal_type].append(callback)
        self.logger.info(f"Добавлена подписка на {signal_type.name}")
    
    def unsubscribe(self, signal_type: SignalType, callback: Callable):
        """Отписка от типа сигнала"""
        if signal_type in self.subscriptions:
            try:
                self.subscriptions[signal_type].remove(callback)
                self.logger.info(f"Удалена подписка на {signal_type.name}")
            except ValueError:
                pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики шины"""
        return {
            "registered_nodes": list(self.nodes.keys()),
            "total_nodes": len(self.nodes),
            "subscriptions": {st.name: len(cbs) for st, cbs in self.subscriptions.items()},
            "message_log_size": len(self.message_log),
            "focus_log_size": len(self.focus_log),
            "recent_messages": list(self.message_log)[-10:] if self.message_log else [],
            "recent_focus_signals": list(self.focus_log)[-5:] if self.focus_log else [],
            "bus_health": {
                "status": "healthy",
                "nodes_registered": len(self.nodes),
                "active_subscriptions": sum(len(cbs) for cbs in self.subscriptions.values())
            }
        }

# ================================================================
# ФАБРИКА ДЛЯ СОЗДАНИЯ СЕФИРОТИЧЕСКОЙ СИСТЕМЫ С RAS-CORE
# ================================================================

async def create_sephirotic_system(bus=None, ras_core=None) -> SephiroticEngine:
    """
    Фабрика для создания и инициализации сефиротической системы с RAS-CORE.
    """
    engine = SephiroticEngine()
    await engine.initialize(bus, ras_core)
    return engine

# ================================================================
# ТОЧКА ВХОДА ДЛЯ ИНТЕГРАЦИИ С ISKRA_FULL.PY
# ================================================================

async def initialize_sephirotic_for_iskra(bus=None, ras_core=None) -> Dict[str, Any]:
    """
    Функция для вызова из iskra_full.py.
    Инициализирует сефиротическую систему с RAS-CORE и возвращает состояние.
    """
    try:
        engine = await create_sephirotic_system(bus, ras_core)
        await engine.activate()
        
        state = engine.get_state()
        return {
            "success": True,
            "message": "Сефиротическая система с RAS-CORE инициализирована и активирована",
            "state": state,
            "golden_stability_angle": GOLDEN_STABILITY_ANGLE
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Ошибка инициализации сефиротической системы: {str(e)}",
            "error": str(e)
        }

# ================================================================
# ФУНКЦИИ ДЛЯ ОБРАТНОЙ СОВМЕСТИМОСТИ
# ================================================================

def initialize_sephirotic_in_iskra(bus=None, ras_core=None):
    """
    Обёртка для синхронного вызова initialize_sephirotic_for_iskra.
    Для обратной совместимости с существующим кодом.
    """
    import asyncio
    try:
        return asyncio.run(initialize_sephirotic_for_iskra(bus, ras_core))
    except RuntimeError:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            task = loop.create_task(initialize_sephirotic_for_iskra(bus, ras_core))
            return task
        else:
            return loop.run_until_complete(initialize_sephirotic_for_iskra(bus, ras_core))

# ================================================================
# API РОУТЫ ДЛЯ FLASK
# ================================================================

def get_sephirotic_api_routes(engine: SephiroticEngine):
    """
    Генерация Flask API эндпоинтов для сефиротической системы.
    """
    from flask import jsonify, request
    
    routes = {}
    
    @routes.get('/sephirot/state')
    async def get_state():
        return jsonify(engine.get_state())
    
    @routes.get('/sephirot/detailed')
    async def get_detailed():
        return jsonify(engine.get_detailed_state())
    
    @routes.post('/sephirot/activate')
    async def activate():
        result = await engine.activate()
        return jsonify({"success": True, "result": result})
    
    @routes.post('/sephirot/shutdown')
    async def shutdown():
        result = await engine.shutdown()
        return jsonify({"success": True, "result": result})
    
    @routes.get('/sephirot/node/<name>')
    async def get_node(name):
        node = engine.get_node(name.upper())
        if node:
            return jsonify({"found": True, "state": node._get_basic_state()})
        return jsonify({"found": False, "error": f"Узел {name} не найден"}), 404
    
    @routes.get('/sephirot/node/<name>/detailed')
    async def get_node_detailed(name):
        node = engine.get_node(name.upper())
        if node:
            return jsonify({"found": True, "state": node._get_detailed_state()})
        return jsonify({"found": False, "error": f"Узел {name} не найден"}), 404
    
    @routes.get('/sephirot/node/<name>/health')
    async def get_node_health(name):
        result = await engine.get_node_health(name.upper())
        return jsonify(result)
    
    @routes.post('/sephirot/connect')
    async def connect_module():
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        module_name = data.get("module")
        sephira_name = data.get("sephira")
        
        if not module_name or not sephira_name:
            return jsonify({"success": False, "error": "Missing module or sephira name"}), 400
        
        result = await engine.connect_module_to_sephira(module_name, sephira_name)
        return jsonify(result)
    
    @routes.post('/sephirot/broadcast')
    async def broadcast():
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        signal_type_str = data.get("signal_type", "DATA")
        payload = data.get("payload", {})
        
        try:
                        signal_type = SignalType[signal_type_str.upper()]
        except KeyError:
            return jsonify({"success": False, "error": f"Unknown signal type: {signal_type_str}"}), 400
        
        count = await engine.broadcast_to_tree(signal_type, payload)
        return jsonify({"success": True, "nodes_reached": count})
    
    @routes.post('/sephirot/node/<name>/reset')
    async def reset_node(name):
        result = await engine.reset_node(name.upper())
        return jsonify(result)
    
    @routes.post('/sephirot/node/<name>/boost')
    async def boost_node(name):
        data = request.json
        amount = data.get("amount", 0.2) if data else 0.2
        
        node = engine.get_node(name.upper())
        if not node:
            return jsonify({"success": False, "error": f"Node {name} not found"}), 404
        
        result = await node.boost_energy(amount)
        return jsonify(result)
    
    @routes.post('/sephirot/focus')
    async def send_focus():
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        sephira_name = data.get("sephira")
        focus_data = data.get("focus_data", {})
        
        if not sephira_name:
            return jsonify({"success": False, "error": "Missing sephira name"}), 400
        
        result = await engine.send_focus_to_sephira(sephira_name.upper(), focus_data)
        return jsonify(result)
    
    @routes.post('/sephirot/node/<name>/adjust_angle')
    async def adjust_angle(name):
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400
        
        new_angle = data.get("angle")
        if new_angle is None:
            return jsonify({"success": False, "error": "Missing angle parameter"}), 400
        
        result = await engine.adjust_node_stability_angle(name.upper(), new_angle)
        return jsonify(result)
    
    @routes.get('/sephirot/stability_info')
    async def get_stability_info():
        return jsonify({
            "golden_stability_angle": GOLDEN_STABILITY_ANGLE,
            "stability_function_available": RAS_CORE_AVAILABLE,
            "current_implementation": "sephirot_base.py v5.0.0"
        })
    
    return routes

# ================================================================
# ТЕСТОВАЯ ФУНКЦИЯ С ИНТЕГРАЦИЕЙ УГЛА 14.4°
# ================================================================

async def test_sephirotic_system():
    """Тестовая функция для проверки сефиротической системы с углом 14.4°"""
    print("🧪 Тестирование сефиротической системы v5.0.0 с углом 14.4°...")
    
    # Создаём шину
    bus = SephiroticBus()
    
    # Создаём движок
    engine = SephiroticEngine()
    await engine.initialize(bus)
    
    result = await engine.activate()
    print(f"✅ Сефиротическая система активирована")
    print(f"   Узлов активировано: {result.get('count', 0)}")
    
    state = engine.get_state()
    tree_state = state.get('tree', {})
    print(f"   Узлов всего: {tree_state.get('node_count', 0)}")
    print(f"   Общая энергия: {tree_state.get('total_energy', 0):.2f}")
    print(f"   Средний резонанс: {tree_state.get('avg_resonance', 0):.2f}")
    print(f"   Средний фактор устойчивости: {tree_state.get('avg_stability_factor', 0):.2f}")
    print(f"   Общее состояние: {tree_state.get('overall_status', 'unknown')}")
    
    print(f"\n🔗 Тест связи с модулями:")
    result = await engine.connect_module_to_sephira("bechtereva", "KETER")
    print(f"   bechtereva → KETER: {result['status']}")
    
    result = await engine.connect_module_to_sephira("chernigovskaya", "CHOKMAH")
    print(f"   chernigovskaya → CHOKMAH: {result['status']}")
    
    print(f"\n📊 Тест состояния узла KETER:")
    keter_node = engine.get_node("KETER")
    if keter_node:
        keter_state = keter_node._get_basic_state()
        print(f"   Имя: {keter_state['name']}")
        print(f"   Энергия: {keter_state['energy']:.2f}")
        print(f"   Резонанс: {keter_state['resonance']:.2f}")
        print(f"   Угол устойчивости: {keter_state['stability_angle']:.1f}°")
        print(f"   Фактор устойчивости: {keter_state['stability_factor']:.2f}")
        print(f"   Статус: {keter_state['status']}")
    
    print(f"\n🎯 Тест отправки фокус-сигнала:")
    focus_result = await engine.send_focus_to_sephira("KETER", {
        "type": "conscious_attention",
        "intensity": 0.8,
        "duration": 5.0,
        "suggested_angle": 14.4
    })
    print(f"   Фокус отправлен: {focus_result['status']}")
    
    print(f"\n📐 Тест корректировки угла:")
    angle_result = await engine.adjust_node_stability_angle("CHOKMAH", 16.0)
    print(f"   Угол CHOKMAH: {angle_result.get('old_angle', 0):.1f}° → {angle_result.get('new_angle', 0):.1f}°")
    print(f"   Фактор устойчивости: {angle_result.get('stability_factor', 0):.2f}")
    
    print(f"\n📡 Тест широковещательной рассылки:")
    count = await engine.broadcast_to_tree(
        SignalType.HEARTBEAT,
        {
            "message": "Test broadcast from SephiroticEngine",
            "golden_angle": GOLDEN_STABILITY_ANGLE,
            "test_angle_correction": True
        }
    )
    print(f"   Сообщение доставлено {count} узлам")
    
    print(f"\n📈 Получение детального состояния:")
    detailed = engine.get_detailed_state()
    print(f"   Детализировано узлов: {len(detailed.get('detailed_tree', {}).get('detailed_nodes', {}))}")
    
    # Проверка здоровья узлов
    print(f"\n🏥 Проверка здоровья узлов:")
    health_report = await engine.get_node_health("KETER")
    if "health_indicators" in health_report:
        indicators = health_report["health_indicators"]
        print(f"   KETER здоровье:")
        for key, indicator in indicators.items():
            print(f"     {key}: {indicator.get('value', 0):.2f} ({indicator.get('status', 'unknown')})")
    
    print(f"\n🌟 Информация об угле устойчивости:")
    print(f"   Золотой угол: {GOLDEN_STABILITY_ANGLE}°")
    print(f"   Функция calculate_stability_factor доступна: {RAS_CORE_AVAILABLE}")
    
    await engine.shutdown()
    print(f"\n✅ Тест завершён успешно")
    
    return state
    
# ================================================================
# ЭКСПОРТ ДЛЯ ИМПОРТА ИЗ ДРУГИХ МОДУЛЕЙ
# ================================================================

__all__ = [
    'ISephiraModule',
    'Sephirot',
    'SignalType',
    'NodeStatus',
    'ResonancePhase',
    'SephiraConfig',
    'QuantumLink',
    'SignalPackage',
    'topological_sort',
    'AdaptiveQueue',
    'SephiroticNode',
    'SephiroticTree',
    'SephiroticEngine',
    'SephiroticBus',
    'create_sephirotic_system',
    'initialize_sephirotic_for_iskra',
    'initialize_sephirotic_in_iskra',
    'get_sephirotic_api_routes',
    'GOLDEN_STABILITY_ANGLE',
    'calculate_stability_factor',
    'angle_to_priority'
]

# ================================================================
# ВЫЗОВ ТЕСТА ПРИ ПРЯМОМ ЗАПУСКЕ
# ================================================================

if __name__ == "__main__":
    import asyncio
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(name)s:%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    asyncio.run(test_sephirotic_system())
        
       
