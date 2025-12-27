#!/usr/bin/env python3
"""
sephirot_base.py - ПОЛНАЯ РЕАЛИЗАЦИЯ СЕФИРОТИЧЕСКОЙ СИСТЕМЫ DS24
Версия: 4.0.0 Production
Интеграция: bechtereva + chernigovskaya + 10 сефирот
"""

import json
import logging
import asyncio
import statistics
import inspect
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Deque, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque, defaultdict
from contextlib import asynccontextmanager
import time
import uuid

# ================================================================
# КОНСТАНТЫ СЕФИРОТИЧЕСКОЙ СИСТЕМЫ
# ================================================================

class Sephirot(Enum):
    """10 сефирот Древа Жизни"""
    KETER = (1, "Венец", "Сознание", "bechtereva")
    CHOKHMAH = (2, "Мудрость", "Интуиция", "chernigovskaya")
    BINAH = (3, "Понимание", "Анализ", "bechtereva")
    CHESED = (4, "Милость", "Экспансия", "emotional_weave")
    GEVURAH = (5, "Строгость", "Ограничение", "immune_core")
    TIFERET = (6, "Гармония", "Баланс", "policy_governor")
    NETZACH = (7, "Победа", "Настойчивость", "heartbeat_core")
    HOD = (8, "Слава", "Коммуникация", "polyglossia_adapter")
    YESOD = (9, "Основа", "Подсознание", "spinal_core")
    MALKUTH = (10, "Царство", "Манифестация", "trust_mesh")
    
    def __init__(self, level, name, description, connected_module):
        self.level = level
        self.name = name
        self.description = description
        self.connected_module = connected_module

# ================================================================
# ТИПЫ И СТРУКТУРЫ
# ================================================================

class SignalType(Enum):
    """Типы сигналов для сефиротической шины"""
    NEURO = auto()          # Нейро-сигнал (от bechtereva)
    SEMIOTIC = auto()       # Семиотический (от chernigovskaya)
    EMOTIONAL = auto()      # Эмоциональный
    COGNITIVE = auto()      # Когнитивный
    INTENTION = auto()      # Интенциональный
    HEARTBEAT = auto()      # Пульс системы
    RESONANCE = auto()      # Резонансный
    COMMAND = auto()        # Команда управления
    DATA = auto()           # Данные
    ERROR = auto()          # Ошибка
    SYNTHESIS = auto()      # Синтез
    ENERGY = auto()         # Энергетический
    SYNC = auto()           # Синхронизация
    METRIC = auto()         # Метрика
    BROADCAST = auto()      # Широковещательный
    FEEDBACK = auto()       # Обратная связь
    CONTROL = auto()        # Контроль
    SEPHIROTIC = auto()     # Сефиротический (внутренний)
    
    @classmethod
    def from_string(cls, value: str) -> 'SignalType':
        """Безопасное преобразование строки в SignalType"""
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
    """Фазы резонансной динамики"""
    SILENT = (0.0, 0.1, "Тишина", 0.1)
    AWAKENING = (0.1, 0.3, "Пробуждение", 0.3)
    COHERENT = (0.3, 0.6, "Когерентность", 0.6)
    RESONANT = (0.6, 0.85, "Резонанс", 0.8)
    PEAK = (0.85, 0.95, "Пик", 0.9)
    TRANSCENDENT = (0.95, 1.0, "Трансценденция", 0.95)
    
    def __init__(self, min_val, max_val, description, ideal_point):
        self.min = min_val
        self.max = max_val
        self.description = description
        self.ideal_point = ideal_point
    
    @classmethod
    def from_value(cls, value: float) -> Tuple['ResonancePhase', float]:
        """Определение фазы и степени приближения к идеальной точке"""
        for phase in cls:
            if phase.min <= value <= phase.max:
                distance_to_ideal = abs(value - phase.ideal_point)
                normalized_distance = distance_to_ideal / (phase.max - phase.min)
                return phase, 1.0 - normalized_distance
        return cls.SILENT, 0.0

@dataclass
class QuantumLink:
    """Квантовая связь между узлами"""
    target: str
    strength: float = 0.5
    coherence: float = 0.8
    entanglement: float = 0.0
    established: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_sync: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    channel_type: str = "quantum"
    history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=100))
    feedback_loop: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    
    def __post_init__(self):
        self.history.append((self.strength, self.coherence))
    
    def evolve(self, delta_time: float = 1.0) -> Tuple[float, float]:
        """Эволюция связи во времени"""
        # Декогеренция
        decoherence = 0.05 * delta_time
        self.coherence = max(0.1, self.coherence - decoherence)
        
        # Самокоррекция
        target_strength = 0.6
        strength_error = target_strength - self.strength
        correction = strength_error * 0.1 * self.coherence
        
        self.strength += correction
        self.strength = max(0.01, min(1.0, self.strength))
        
        # Квантовая запутанность
        if self.coherence > 0.7:
            self.entanglement = min(1.0, self.entanglement + 0.01 * delta_time)
        
        self.history.append((self.strength, self.coherence))
        return self.strength, self.coherence
    
    def apply_feedback(self, feedback: float) -> float:
        """Применение обратной связи"""
        self.feedback_loop.append(feedback)
        
        if len(self.feedback_loop) >= 3:
            avg_feedback = statistics.mean(self.feedback_loop)
            correction = (avg_feedback - self.strength) * 0.2
            self.strength += correction
            self.coherence = min(1.0, self.coherence + 0.05)
        
        self.last_sync = datetime.utcnow().isoformat()
        return self.strength
    
    def get_quantum_state(self) -> Dict[str, Any]:
        """Получение квантового состояния"""
        return {
            "strength": self.strength,
            "coherence": self.coherence,
            "entanglement": self.entanglement,
            "stability": statistics.stdev([s for s, _ in self.history]) if len(self.history) > 1 else 0.0,
            "age_seconds": (datetime.utcnow() - datetime.fromisoformat(
                self.established.replace('Z', '+00:00')
            )).total_seconds()
        }

@dataclass
class SignalPackage:
    """Пакет сигнала с полной трассировкой"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: SignalType = SignalType.DATA
    source: str = ""
    target: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    ttl: float = 30.0
    
    def __post_init__(self):
        self.metadata.update({
            "signature": hashlib.sha256(str(self.payload).encode()).hexdigest()[:16],
            "hops": 0,
            "processed_by": [],
            "resonance_trace": []
        })
    
    def add_resonance_trace(self, node: str, resonance: float):
        """Добавление узла в трассировку резонанса"""
        self.metadata["resonance_trace"].append({
            "node": node,
            "resonance": resonance,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def add_processing_node(self, node: str):
        """Добавление узла в историю обработки"""
        self.metadata["processed_by"].append(node)
        self.metadata["hops"] += 1
    
    def is_expired(self) -> bool:
        """Проверка истечения срока жизни"""
        created = datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))
        return (datetime.utcnow() - created).total_seconds() > self.ttl
    
    def to_transport_dict(self) -> Dict[str, Any]:
        """Конвертация для передачи через шину"""
        return {
            "id": self.id,
            "type": self.type.name,
            "source": self.source,
            "target": self.target,
            "payload": self.payload,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "ttl": self.ttl
        }
    
    @classmethod
    def from_transport_dict(cls, data: Dict[str, Any]) -> 'SignalPackage':
        """Создание из транспортного формата"""
        package = cls(
            id=data.get("id", str(uuid.uuid4())),
            type=SignalType.from_string(data.get("type", "DATA")),
            source=data.get("source", ""),
            target=data.get("target", ""),
            payload=data.get("payload", {}),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            ttl=data.get("ttl", 30.0)
        )
        package.metadata = data.get("metadata", {})
        return package

class AdaptiveQueue:
    """Адаптивная очередь с автоочисткой"""
    
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
        """Запуск фоновой очистки"""
        self._cleanup_task = asyncio.create_task(self._cleanup_worker())
    
    async def stop(self):
        """Остановка очистки"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def put(self, item: Any, priority: int = 5) -> bool:
        """Добавление элемента с приоритетом"""
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
        """Получение элемента из очереди"""
        priority, enqueued_at, item = await self._queue.get()
        wait_time = time.time() - enqueued_at
        self._wait_times.append(wait_time)
        self._stats["total_processed"] += 1
        self._stats["avg_wait_time"] = statistics.mean(self._wait_times) if self._wait_times else 0
        return item
    
    def task_done(self):
        """Отметка завершения обработки"""
        self._queue.task_done()
    
    def qsize(self) -> int:
        """Текущий размер очереди"""
        return self._queue.qsize()
    
    async def _make_room(self) -> bool:
        """Освобождение места в очереди"""
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
        """Фоновая очистка устаревших элементов"""
        while True:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._make_room()
            except asyncio.CancelledError:
                break
            except Exception as e:
                await asyncio.sleep(self._cleanup_interval * 2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики очереди"""
        return {
            **self._stats,
            "current_size": self.qsize(),
            "max_size": self._max_size,
            "usage_percent": (self.qsize() / self._max_size) * 100,
            "recent_avg_wait": self._stats["avg_wait_time"]
        }

# ================================================================
# ЯДРО СЕФИРОТИЧЕСКОГО УЗЛА
# ================================================================

class SephiroticNode:
    """
    Сефиротический узел - связь с модулями Бехтеревой и Черниговской
    """
    
    VERSION = "4.0.0"
    MAX_QUEUE_SIZE = 250
    MAX_MEMORY_LOGS = 500
    DEFAULT_TTL = 60.0
    ENERGY_RECOVERY_RATE = 0.015
    RESONANCE_DECAY_BASE = 0.97
    METRICS_INTERVAL = 3.0
    
    def __init__(self, sephira: Sephirot, bus=None):
        """
        :param sephira: Сефира (KETER, CHOKHMAH и т.д.)
        :param bus: Шина связи (опционально)
        """
        self.sephira = sephira
        self.name = sephira.name
        self.level = sephira.level
        self.description = sephira.description
        self.connected_module = sephira.connected_module
        self.bus = bus
        
        # Инициализация состояний
        self._initialize_states()
        
        # Структуры данных
        self._initialize_data_structures()
        
        # Системные компоненты
        self._initialize_system_components()
        
        # Запуск инициализации
        asyncio.create_task(self._async_initialization())
    
    def _initialize_states(self):
        """Инициализация всех состояний узла"""
        self.status = NodeStatus.CREATED
        self.resonance = 0.1
        self.energy = 0.8
        self.stability = 0.9
        self.coherence = 0.7
        self.willpower = 0.6
        
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
        # Квантовые связи
        self.quantum_links: Dict[str, QuantumLink] = {}
        
        # Адаптивная очередь
        self.signal_queue = AdaptiveQueue(
            max_size=self.MAX_QUEUE_SIZE,
            cleanup_interval=5.0
        )
        
        # Память
        self.signal_history = deque(maxlen=self.MAX_MEMORY_LOGS)
        self.resonance_history = deque(maxlen=200)
        self.energy_history = deque(maxlen=200)
        
        # Кэши
        self.response_cache = {}
        self.link_cache = {}
        
        # Статистика
        self._signal_counter = defaultdict(int)
        self._processing_times = deque(maxlen=100)
        self._error_log = deque(maxlen=50)
    
    def _initialize_system_components(self):
        """Инициализация системных компонентов"""
        # Логирование
        self.logger = self._setup_logger()
        
        # Обработчики сигналов
        self.signal_handlers = self._initialize_signal_handlers()
        
        # Задачи
        self._background_tasks = set()
        self._shutdown_event = asyncio.Event()
        
        # Метрики
        self.metrics = {
            "node": self.name,
            "version": self.VERSION,
            "sephira": self.sephira.value,
            "connected_module": self.connected_module,
            "start_time": datetime.utcnow().isoformat(),
            "status": self.status.value
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger(f"Sephirot.{self.name}")
        
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            
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
        """Инициализация обработчиков сигналов"""
        return {
            SignalType.NEURO: self._handle_neuro,
            SignalType.SEMIOTIC: self._handle_semiotic,
            SignalType.EMOTIONAL: self._handle_emotional,
            SignalType.COGNITIVE: self._handle_cognitive,
            SignalType.INTENTION: self._handle_intention,
            SignalType.HEARTBEAT: self._handle_heartbeat,
            SignalType.RESONANCE: self._handle_resonance,
            SignalType.COMMAND: self._handle_command,
            SignalType.DATA: self._handle_data,
            SignalType.ERROR: self._handle_error,
            SignalType.SYNTHESIS: self._handle_synthesis,
            SignalType.ENERGY: self._handle_energy,
            SignalType.SYNC: self._handle_sync,
            SignalType.METRIC: self._handle_metric,
            SignalType.BROADCAST: self._handle_broadcast,
            SignalType.FEEDBACK: self._handle_feedback,
            SignalType.CONTROL: self._handle_control,
            SignalType.SEPHIROTIC: self._handle_sephirotic
        }
    
    async def _async_initialization(self):
        """Асинхронная инициализация узла"""
        try:
            self.logger.info(f"Инициализация сефиротического узла {self.name}")
            self.status = NodeStatus.INITIALIZING
            
            # Запуск очереди
            await self.signal_queue.start()
            
            # Запуск фоновых задач
            await self._start_background_tasks()
            
            # Регистрация в шине
            if self.bus and hasattr(self.bus, 'register_node'):
                await self.bus.register_node(self)
            
            # Активация
            await self._activate_core()
            
            self._is_initialized = True
            self.status = NodeStatus.ACTIVE
            self.activation_time = datetime.utcnow().isoformat()
            
            self.logger.info(f"Сефиротический узел {self.name} активирован")
            
            # Эмитация heartbeat
            await self._emit_async(SignalPackage(
                type=SignalType.HEARTBEAT,
                source=self.name,
                payload={
                    "event": "sephirot_activated",
                    "sephira": self.name,
                    "level": self.level,
                    "module": self.connected_module
                }
            ))
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации: {e}")
            self.status = NodeStatus.DEGRADED
            raise
    
    async def _start_background_tasks(self):
        """Запуск фоновых задач"""
        tasks = [
            self._signal_processor(),
            self._resonance_dynamics(),
            self._energy_manager(),
            self._metrics_collector(),
            self._link_maintainer(),
            self._health_monitor()
        ]
        
        for task in tasks:
            task_obj = asyncio.create_task(task)
            self._background_tasks.add(task_obj)
            task_obj.add_done_callback(self._background_tasks.discard)
    
    # ================================================================
    # ОСНОВНЫЕ МЕТОДЫ ОБРАБОТКИ СИГНАЛОВ
    # ================================================================
    
    async def receive(self, signal_package: SignalPackage) -> SignalPackage:
        """
        Основной метод приёма сигналов.
        """
        if not self._is_initialized or self._is_suspended:
            return self._create_error_response(
                signal_package,
                "node_not_ready",
                f"Узел в состоянии: {self.status.value}"
            )
        
        if signal_package.is_expired():
            self.logger.warning(f"Просроченный сигнал: {signal_package.id}")
            return self._create_error_response(signal_package, "signal_expired")
        
        priority = self._calculate_priority(signal_package)
        if not await self.signal_queue.put(signal_package, priority):
            return self._create_error_response(
                signal_package,
                "queue_full",
                "Очередь переполнена"
            )
        
        # Ответ о принятии
        ack_response = SignalPackage(
            type=SignalType.FEEDBACK,
            source=self.name,
            target=signal_package.source,
            payload={
                "status": "queued",
                "original_id": signal_package.id,
                "queue_position": self.signal_queue.qsize(),
                "priority": priority
            }
        )
        
        return ack_response
    
    async def _signal_processor(self):
        """Процессор сигналов из адаптивной очереди"""
        self.logger.info(f"Процессор сигналов запущен для {self.name}")
        
        while not self._shutdown_event.is_set():
            try:
                signal_package = await self.signal_queue.get()
                start_time = time.perf_counter()
                response = await self._process_signal_deep(signal_package)
                processing_time = time.perf_counter() - start_time
                
                # Обновление статистики
                self._processing_times.append(processing_time)
                self._signal_counter[signal_package.type.name] += 1
                self.total_signals_processed += 1
                
                # Сохранение в историю
                signal_package.add_processing_node(self.name)
                signal_package.add_resonance_trace(self.name, self.resonance)
                self.signal_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "signal": signal_package.id,
                    "type": signal_package.type.name,
                    "processing_time": processing_time,
                    "response_type": response.type.name
                })
                
                # Отправка ответа
                if response.target and self.bus:
                    await self._emit_async(response)
                
                self.signal_queue.task_done()
                
                # Обновление энергетики
                energy_cost = processing_time * 0.2
                self.energy = max(0.1, self.energy - energy_cost)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка в процессоре сигналов: {e}")
                self._error_log.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e)
                })
                await asyncio.sleep(0.1)
    
    async def _process_signal_deep(self, signal_package: SignalPackage) -> SignalPackage:
        """
        Глубокая обработка сигнала.
        """
        # Проверка кэша
        cache_key = self._generate_cache_key(signal_package)
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key].copy()
            cached_response.metadata["cached"] = True
            return cached_response
        
        # Получение обработчика
        handler = self.signal_handlers.get(signal_package.type)
        if not handler:
            handler = self._handle_unknown
        
        # Выполнение обработки
        try:
            handler_result = await handler(signal_package)
            
            # Применение резонансной обратной связи
            resonance_feedback = await self._apply_resonance_feedback(
                signal_package,
                handler_result
            )
            
            # Создание ответа
            response = SignalPackage(
                type=SignalType.FEEDBACK,
                source=self.name,
                target=signal_package.source,
                payload={
                    "original_id": signal_package.id,
                    "processed_by": self.name,
                    "handler": signal_package.type.name,
                    "result": handler_result,
                    "resonance_feedback": resonance_feedback,
                    "node_state": {
                        "resonance": self.resonance,
                        "energy": self.energy,
                        "stability": self.stability,
                        "coherence": self.coherence
                    }
                }
            )
            
            # Кэширование
            if signal_package.type not in [SignalType.HEARTBEAT, SignalType.METRIC]:
                self.response_cache[cache_key] = response
                if len(self.response_cache) > 100:
                    oldest_key = next(iter(self.response_cache))
                    del self.response_cache[oldest_key]
            
            return response
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки сигнала {signal_package.id}: {e}")
            return self._create_error_response(signal_package, "processing_error", str(e))
    
    def _generate_cache_key(self, signal_package: SignalPackage) -> str:
        """Генерация ключа кэша"""
        content_hash = hashlib.md5(
            json.dumps(signal_package.payload, sort_keys=True).encode()
        ).hexdigest()
        return f"{signal_package.type.name}:{signal_package.source}:{content_hash}"
    
    def _calculate_priority(self, signal_package: SignalPackage) -> int:
        """Расчёт приоритета для очереди"""
        priority_map = {
            SignalType.CONTROL: 1,
            SignalType.ERROR: 2,
            SignalType.HEARTBEAT: 3,
            SignalType.SYNC: 4,
            SignalType.NEURO: 5,          # Приоритет для нейро-сигналов
            SignalType.SEMIOTIC: 5,       # Приоритет для семиотики
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
        resonance_factor = 1.0 - (self.resonance * 0.5)
        adjusted_priority = int(base_priority * resonance_factor)

    # ================================================================
    # ОБРАБОТЧИКИ СИГНАЛОВ (ИНТЕГРАЦИЯ С МОДУЛЯМИ)
    # ================================================================
    
    async def _handle_neuro(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка нейро-сигналов от модуля Бехтеревой"""
        self.logger.info(f"Обработка NEURO сигнала от {signal_package.source}")
        
        # Получаем данные от bechtereva
        neuro_data = signal_package.payload.get("neuro_data", {})
        
        # В зависимости от сефиры обрабатываем по-разному
        if self.sephira == Sephirot.KETER:
            # KETER: интеграция высшего сознания
            processed = {
                "action": "conscious_integration",
                "sephira": self.name,
                "neuro_coherence": neuro_data.get("coherence", 0.0),
                "cognitive_load": neuro_data.get("load", 0.0),
                "timestamp": datetime.utcnow().isoformat()
            }
        elif self.sephira == Sephirot.BINAH:
            # BINAH: аналитическая обработка
            processed = {
                "action": "analytical_processing",
                "sephira": self.name,
                "patterns_detected": neuro_data.get("patterns", []),
                "analysis_depth": neuro_data.get("depth", 1.0),
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Остальные сефиры пропускают с минимальной обработкой
            processed = {
                "action": "neuro_passthrough",
                "sephira": self.name,
                "original_data": neuro_data,
                "energy_boost": 0.05,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Усиление резонанса от нейро-сигналов
        self.resonance = min(1.0, self.resonance + 0.1)
        
        return {
            "status": "neuro_processed",
            "sephira": self.name,
            "result": processed,
            "resonance_increase": 0.1
        }
    
    async def _handle_semiotic(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка семиотических сигналов от Черниговской"""
        self.logger.info(f"Обработка SEMIOTIC сигнала от {signal_package.source}")
        
        semiotic_data = signal_package.payload.get("semiotic_data", {})
        
        if self.sephira == Sephirot.CHOKHMAH:
            # CHOKHMAH: мудрость и интуиция
            processed = {
                "action": "wisdom_integration",
                "sephira": self.name,
                "semiotic_patterns": semiotic_data.get("patterns", []),
                "intuition_level": semiotic_data.get("intuition_score", 0.0),
                "meaning_extracted": semiotic_data.get("meaning", ""),
                "timestamp": datetime.utcnow().isoformat()
            }
        elif self.sephira == Sephirot.HOD:
            # HOD: коммуникация и передача
            processed = {
                "action": "communication_bridge",
                "sephira": self.name,
                "message_complexity": semiotic_data.get("complexity", 1),
                "translation_required": semiotic_data.get("needs_translation", False),
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            processed = {
                "action": "semiotic_passthrough",
                "sephira": self.name,
                "original_data": semiotic_data,
                "coherence_boost": 0.08,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Увеличение когерентности от семиотики
        self.coherence = min(1.0, self.coherence + 0.15)
        
        return {
            "status": "semiotic_processed",
            "sephira": self.name,
            "result": processed,
            "coherence_increase": 0.15
        }
    
    async def _handle_emotional(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка эмоциональных сигналов"""
        return {"status": "emotional_processed", "sephira": self.name}
    
    async def _handle_cognitive(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка когнитивных сигналов"""
        return {"status": "cognitive_processed", "sephira": self.name}
    
    async def _handle_intention(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка интенциональных сигналов"""
        return {"status": "intention_processed", "sephira": self.name}
    
    async def _handle_heartbeat(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка heartbeat сигналов"""
        # Обновление энергии от heartbeat
        self.energy = min(1.0, self.energy + 0.05)
        return {
            "status": "heartbeat_ack",
            "sephira": self.name,
            "current_energy": self.energy,
            "resonance": self.resonance
        }
    
    async def _handle_resonance(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка резонансных сигналов"""
        incoming_resonance = signal_package.payload.get("resonance", 0.0)
        # Синхронизация резонанса
        self.resonance = (self.resonance + incoming_resonance) / 2.0
        return {
            "status": "resonance_synced",
            "sephira": self.name,
            "new_resonance": self.resonance
        }
    
    async def _handle_command(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка команд управления"""
        command = signal_package.payload.get("command", "")
        if command == "activate":
            await self._activate_core()
            return {"status": "activated", "sephira": self.name}
        elif command == "deactivate":
            await self._deactivate_core()
            return {"status": "deactivated", "sephira": self.name}
        else:
            return {"status": "unknown_command", "sephira": self.name}
    
    async def _handle_data(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка данных"""
        return {"status": "data_processed", "sephira": self.name}
    
    async def _handle_error(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка ошибок"""
        error_msg = signal_package.payload.get("error", "Unknown error")
        self.logger.error(f"Ошибка получена: {error_msg}")
        return {"status": "error_logged", "sephira": self.name}
    
    async def _handle_synthesis(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка синтеза"""
        return {"status": "synthesis_processed", "sephira": self.name}
    
    async def _handle_energy(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка энергетических сигналов"""
        energy_transfer = signal_package.payload.get("energy", 0.0)
        self.energy = max(0.0, min(1.0, self.energy + energy_transfer))
        return {
            "status": "energy_updated",
            "sephira": self.name,
            "new_energy": self.energy
        }
    
    async def _handle_sync(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка синхронизации"""
        return {"status": "synced", "sephira": self.name}
    
    async def _handle_metric(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка метрик"""
        return {"status": "metric_processed", "sephira": self.name}
    
    async def _handle_broadcast(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка широковещательных сообщений"""
        return {"status": "broadcast_received", "sephira": self.name}
    
    async def _handle_feedback(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка обратной связи"""
        return {"status": "feedback_processed", "sephira": self.name}
    
    async def _handle_control(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка управляющих сигналов"""
        return {"status": "control_processed", "sephira": self.name}
    
    async def _handle_sephirotic(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка внутренних сефиротических сигналов"""
        action = signal_package.payload.get("action", "")
        
        if action == "link_request":
            target_node = signal_package.payload.get("target_node", "")
            return await self._create_link(target_node)
        elif action == "energy_request":
            amount = signal_package.payload.get("amount", 0.1)
            return await self._transfer_energy(amount, signal_package.source)
        else:
            return {"status": "sephirotic_action_unknown", "sephira": self.name}
    
    async def _handle_unknown(self, signal_package: SignalPackage) -> Dict[str, Any]:
        """Обработка неизвестных сигналов"""
        self.logger.warning(f"Неизвестный тип сигнала: {signal_package.type}")
        return {"status": "unknown_signal_type", "sephira": self.name}
    
    # ================================================================
    # СИСТЕМА РЕЗОНАНСНОЙ ОБРАТНОЙ СВЯЗИ
    # ================================================================
    
    async def _apply_resonance_feedback(self, signal_package: SignalPackage, 
                                      handler_result: Any) -> Dict[str, Any]:
        """
        Применение резонансной обратной связи.
        """
        phase, phase_perfection = ResonancePhase.from_value(self.resonance)
        base_feedback_strength = self.resonance * phase_perfection
        
        # Модификаторы для разных типов сигналов
        type_modifiers = {
            SignalType.NEURO: 1.4,
            SignalType.SEMIOTIC: 1.3,
            SignalType.EMOTIONAL: 1.2,
            SignalType.RESONANCE: 1.5,
            SignalType.SYNTHESIS: 1.4,
            SignalType.INTENTION: 1.1,
            SignalType.ERROR: 0.7,
            SignalType.HEARTBEAT: 0.5
        }
        
        type_modifier = type_modifiers.get(signal_package.type, 1.0)
        feedback_strength = base_feedback_strength * type_modifier
        
        # Определение эффекта
        effect = "stabilize"
        if feedback_strength < 0.3:
            effect = "dampen"
        elif feedback_strength < 0.6:
            effect = "resonate"
        elif feedback_strength < 0.8:
            effect = "amplify"
        else:
            effect = "transcend"
        
        feedback = {
            "strength": min(1.0, feedback_strength),
            "phase": phase.description,
            "phase_perfection": phase_perfection,
            "effect": effect,
            "suggested_amplification": self._calculate_amplification(feedback_strength),
            "coherence_impact": self.coherence * 0.1,
            "quantum_correction": self._quantum_correction_value()
        }
        
        # Обновление резонанса
        resonance_delta = feedback_strength * 0.05 - 0.02
        await self._update_resonance_with_feedback(resonance_delta, feedback)
        
        # Распространение по связям
        await self._propagate_feedback_to_links(feedback_strength)
        
        return feedback
    
    def _calculate_amplification(self, strength: float) -> float:
        """Расчёт рекомендуемого усиления"""
        if strength < 0.3:
            return 0.5
        elif strength < 0.7:
            return 1.0
        else:
            return 1.0 + (strength - 0.7) * 2
    
    def _quantum_correction_value(self) -> float:
        """Расчёт квантовой поправки"""
        if not self.quantum_links:
            return 0.0
        
        try:
            avg_coherence = statistics.mean(
                [link.coherence for link in self.quantum_links.values()]
            )
            avg_entanglement = statistics.mean(
                [link.entanglement for link in self.quantum_links.values()]
            )
            return (avg_coherence * 0.7 + avg_entanglement * 0.3) * 0.1
        except:
            return 0.0
    
    async def _update_resonance_with_feedback(self, delta: float, feedback: Dict[str, Any]):
        """Обновление резонанса с учётом обратной связи"""
        self.resonance = (
            self.resonance * self.RESONANCE_DECAY_BASE +
            delta * (1 - self.RESONANCE_DECAY_BASE)
        )
        self.resonance = max(0.0, min(1.0, self.resonance))
        
        self.resonance_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "value": self.resonance,
            "delta": delta,
            "feedback_effect": feedback.get("effect", "unknown")
        })
        
        self.coherence = min(1.0, self.coherence + abs(delta) * 0.05)
    
    async def _propagate_feedback_to_links(self, strength: float):
        """Распространение обратной связи по связям"""
        if not self.quantum_links:
            return
        
        for link in self.quantum_links.values():
            if link.coherence > 0.5:
                link.apply_feedback(strength * 0.3)
                
                if self.bus and hasattr(self.bus, 'transmit'):
                    feedback_package = SignalPackage(
                        type=SignalType.FEEDBACK,
                        source=self.name,
                        target=link.target,
                        payload={
                            "feedback_type": "resonance_propagation",
                            "strength": strength * 0.3,
                            "source_resonance": self.resonance,
                            "timestamp": datetime.utcnow().isoformat()
                        }
                    )
                    await self._emit_async(feedback_package)
    
    # ================================================================
    # ШИРОКОВЕЩАТЕЛЬНАЯ СИСТЕМА
    # ================================================================
    
    async def broadcast(self, signal_type: SignalType, payload: Dict[str, Any], 
                       exclude_nodes: List[str] = None) -> int:
        """
        Широковещательная рассылка сигналов.
        
        :param signal_type: Тип сигнала
        :param payload: Полезная нагрузка
        :param exclude_nodes: Узлы для исключения
        :return: Количество узлов, получивших сообщение
        """
        if not self.bus or not hasattr(self.bus, 'broadcast'):
            self.logger.warning("Шина не поддерживает broadcast")
            return 0
        
        signal_package = SignalPackage(
            type=signal_type,
            source=self.name,
            payload=payload,
            metadata={
                "broadcast_origin": self.name,
                "broadcast_time": datetime.utcnow().isoformat()
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
            source=self.name,
            target=original_package.source,
            payload={
                "error_code": error_code,
                "error_message": error_message,
                "original_id": original_package.id,
                "sephira": self.name,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    # ================================================================
    # МЕТОДЫ УПРАВЛЕНИЯ ЭНЕРГЕТИКОЙ И РЕЗОНАНСОМ
    # ================================================================
    
    async def _activate_core(self):
        """Активация ядра узла"""
        self.logger.info(f"Активация ядра {self.name}")
        self.energy = 0.9
        self.resonance = 0.3
        self.coherence = 0.8
        
        # Создание связей с соответствующими модулями
        if self.connected_module:
            await self._create_link(self.connected_module)
    
    async def _deactivate_core(self):
        """Деактивация ядра узла"""
        self.logger.info(f"Деактивация ядра {self.name}")
        self.energy = 0.1
        self.resonance = 0.1
        self._is_suspended = True
    
    async def _create_link(self, target_node: str) -> Dict[str, Any]:
        """Создание квантовой связи с другим узлом"""
        if target_node in self.quantum_links:
            return {"status": "link_exists", "sephira": self.name}
        
        link = QuantumLink(target=target_node)
        self.quantum_links[target_node] = link
        
        self.logger.info(f"Создана связь с {target_node}")
        
        return {
            "status": "link_created",
            "sephira": self.name,
            "target": target_node,
            "strength": link.strength,
            "coherence": link.coherence
        }
    
    async def _transfer_energy(self, amount: float, target: str) -> Dict[str, Any]:
        """Передача энергии другому узлу"""
        if self.energy < amount + 0.1:
            return {
                "status": "insufficient_energy",
                "sephira": self.name,
                "available": self.energy,
                "requested": amount
            }
        
        self.energy -= amount
        
        # Отправка энергетического пакета
        if self.bus:
            energy_package = SignalPackage(
                type=SignalType.ENERGY,
                source=self.name,
                target=target,
                payload={
                    "energy_transfer": amount,
                    "source_sephira": self.name,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            await self._emit_async(energy_package)
        
        return {
            "status": "energy_transferred",
            "sephira": self.name,
            "amount": amount,
            "target": target,
            "remaining_energy": self.energy
        }
    
    # ================================================================
    # ФОНОВЫЕ ЗАДАЧИ
    # ================================================================
    
    async def _resonance_dynamics(self):
        """Фоновая задача: динамика резонанса"""
        self.logger.info(f"Запущена динамика резонанса для {self.name}")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(2.0)
                
                # Естественный спад резонанса
                self.resonance *= 0.99
                
                # Восстановление от связей
                if self.quantum_links:
                    avg_link_strength = statistics.mean(
                        [link.strength for link in self.quantum_links.values()]
                    )
                    self.resonance = min(1.0, self.resonance + avg_link_strength * 0.01)
                
                # Обновление связей
                for link in self.quantum_links.values():
                    link.evolve(2.0)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка в динамике резонанса: {e}")
                await asyncio.sleep(5.0)
    
    async def _energy_manager(self):
        """Фоновая задача: управление энергией"""
        self.logger.info(f"Запущен менеджер энергии для {self.name}")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(3.0)
                
                # Естественное восстановление
                self.energy = min(1.0, self.energy + self.ENERGY_RECOVERY_RATE)
                
                # Затраты на поддержание связей
                if self.quantum_links:
                    energy_cost = len(self.quantum_links) * 0.005
                    self.energy = max(0.1, self.energy - energy_cost)
                
                # Сохранение в историю
                self.energy_history.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "value": self.energy,
                    "links_count": len(self.quantum_links)
                })
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка в менеджере энергии: {e}")
                await asyncio.sleep(5.0)
    
    async def _metrics_collector(self):
        """Фоновая задача: сбор метрик"""
        self.logger.info(f"Запущен сборщик метрик для {self.name}")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.METRICS_INTERVAL)
                
                # Сбор метрик
                current_metrics = {
                    "resonance": self.resonance,
                    "energy": self.energy,
                    "coherence": self.coherence,
                    "stability": self.stability,
                    "active_links": len(self.quantum_links),
                    "queue_size": self.signal_queue.qsize(),
                    "signals_processed": self.total_signals_processed,
                    "cycle_count": self.cycle_count,
                    "status": self.status.value
                }
                
                # Обновление основной метрики
                self.metrics.update(current_metrics)
                self.metrics["last_update"] = datetime.utcnow().isoformat()
                
                self.cycle_count += 1
                
                # Отправка метрик через шину
                if self.bus and self.cycle_count % 10 == 0:
                    metrics_package = SignalPackage(
                        type=SignalType.METRIC,
                        source=self.name,
                        payload={
                            "sephira": self.name,
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
        self.logger.info(f"Запущен обслуживатель связей для {self.name}")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(10.0)
                
                # Проверка и удаление слабых связей
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
        """Фоновая задача: мониторинг здоровья"""
        self.logger.info(f"Запущен мониторинг здоровья для {self.name}")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(5.0)
                
                # Проверка состояния
                if self.energy < 0.2:
                    self.status = NodeStatus.DEGRADED
                    self.logger.warning(f"Низкая энергия: {self.energy}")
                elif self.energy < 0.1:
                    self.status = NodeStatus.OVERLOADED
                    self.logger.error(f"Критически низкая энергия: {self.energy}")
                else:
                    self.status = NodeStatus.ACTIVE
                
                # Логирование ошибок
                if self._error_log:
                    recent_errors = list(self._error_log)[-5:]
                    self.logger.debug(f"Последние 5 ошибок: {recent_errors}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка в мониторинге здоровья: {e}")
                await asyncio.sleep(10.0)
    
    # ================================================================
    # API МЕТОДЫ ДЛЯ ВНЕШНЕГО ДОСТУПА
    # ================================================================
    
    async def shutdown(self):
        """Корректное завершение работы узла"""
        self.logger.info(f"Завершение работы узла {self.name}")
        self._is_terminating = True
        self.status = NodeStatus.TERMINATING
        
        # Остановка фоновых задач
        self._shutdown_event.set()
        await asyncio.sleep(0.5)
        
        # Остановка очереди
        await self.signal_queue.stop()
        
        self.status = NodeStatus.TERMINATED
        self.logger.info(f"Узел {self.name} завершил работу")
    
    def get_state(self) -> Dict[str, Any]:
        """Получение текущего состояния узла"""
        return {
            "name": self.name,
            "sephira": self.sephira.name,
            "level": self.level,
            "description": self.description,
            "connected_module": self.connected_module,
            "status": self.status.value,
            "resonance": self.resonance,
            "energy": self.energy,
            "coherence": self.coherence,
            "stability": self.stability,
            "activation_time": self.activation_time,
            "active_links": [link.target for link in self.quantum_links.values()],
            "queue_size": self.signal_queue.qsize(),
            "total_signals_processed": self.total_signals_processed,
            "metrics": self.metrics
        }
    
    def get_detailed_state(self) -> Dict[str, Any]:
        """Получение детального состояния"""
        state = self.get_state()
        state.update({
            "quantum_links": {
                target: link.get_quantum_state()
                for target, link in self.quantum_links.items()
            },
            "signal_stats": dict(self._signal_counter),
            "queue_stats": self.signal_queue.get_stats(),
            "recent_errors": list(self._error_log)[-10:],
            "resonance_history": list(self.resonance_history)[-20:],
            "energy_history": list(self.energy_history)[-20:]
        })
        return state
    
    async def connect_to_module(self, module_name: str) -> Dict[str, Any]:
        """Явное подключение к модулю"""
        return await self._create_link(module_name)
    
    async def boost_energy(self, amount: float = 0.2) -> Dict[str, Any]:
        """Увеличение энергии узла"""
        self.energy = min(1.0, self.energy + amount)
        return {
            "status": "energy_boosted",
            "sephira": self.name,
            "amount": amount,
            "new_energy": self.energy
        }
    
    async def set_resonance(self, value: float) -> Dict[str, Any]:
        """Установка резонанса (для тестирования)"""
        old_value = self.resonance
        self.resonance = max(0.0, min(1.0, value))
        return {
            "status": "resonance_set",
            "sephira": self.name,
            "old_value": old_value,
            "new_value": self.resonance
        }

# ================================================================
# СЕФИРОТИЧЕСКОЕ ДЕРЕВО
# ================================================================

class SephiroticTree:
    """
    Древо Жизни - все 10 сефирот как единая система.
    """
    
    def __init__(self, bus=None):
        self.bus = bus
        self.nodes: Dict[str, SephiroticNode] = {}
        self.initialized = False
        self.logger = logging.getLogger("Sephirotic.Tree")
        
    async def initialize(self):
        """Инициализация всех 10 сефирот"""
        if self.initialized:
            return
        
        self.logger.info("Инициализация Сефиротического Древа")
        
        # Создание всех 10 сефирот
        for sephira in Sephirot:
            node = SephiroticNode(sephira, self.bus)
            self.nodes[sephira.name] = node
        
        # Установка связей между сефиротами
        await self._establish_sephirotic_connections()
        
        self.initialized = True
        self.logger.info("Сефиротическое Древо инициализировано")
    
    async def _establish_sephirotic_connections(self):
        """Установка канонических связей между сефиротами"""
        connections = {
            "KETER": ["CHOKHMAH", "BINAH"],
            "CHOKHMAH": ["BINAH"],
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
    
    async def activate_all(self):
        """Активация всех сефирот"""
        self.logger.info("Активация всех сефирот")
        
        for name, node in self.nodes.items():
            if node.status != NodeStatus.ACTIVE:
                await node._activate_core()
        
        self.logger.info("Все сефироты активированы")
        return {"status": "all_activated", "count": len(self.nodes)}
    
    async def shutdown_all(self):
        """Завершение работы всех сефирот"""
        self.logger.info("Завершение работы всех сефирот")
        
        for name, node in self.nodes.items():
            if node.status != NodeStatus.TERMINATED:
                await node.shutdown()
        
        self.initialized = False
        self.logger.info("Все сефироты завершили работу")
        return {"status": "all_shutdown", "count": len(self.nodes)}
    
    def get_node(self, name: str) -> Optional[SephiroticNode]:
        """Получение узла по имени"""
        return self.nodes.get(name.upper())
    
    def get_tree_state(self) -> Dict[str, Any]:
        """Получение состояния всего дерева"""
        if not self.initialized:
            return {"status": "not_initialized"}
        
        nodes_state = {}
        total_energy = 0.0
        total_resonance = 0.0
        
        for name, node in self.nodes.items():
            state = node.get_state()
            nodes_state[name] = state
            total_energy += state["energy"]
            total_resonance += state["resonance"]
        
        avg_energy = total_energy / len(self.nodes) if self.nodes else 0.0
        avg_resonance = total_resonance / len(self.nodes) if self.nodes else 0.0
        
        return {
            "status": "active",
            "initialized": True,
            "node_count": len(self.nodes),
            "total_energy": total_energy,
            "total_resonance": total_resonance,
            "avg_energy": avg_energy,
            "avg_resonance": avg_resonance,
            "nodes": nodes_state
        }
    
    def get_detailed_tree_state(self) -> Dict[str, Any]:
        """Получение детального состояния дерева"""
        base_state = self.get_tree_state()
        
        if base_state["status"] == "not_initialized":
            return base_state
        
        detailed_nodes = {}
        for name, node in self.nodes.items():
            detailed_nodes[name] = node.get_detailed_state()
        
        base_state["detailed_nodes"] = detailed_nodes
        return base_state

# ================================================================
# СИНГЛТОН ДВИЖКА СЕФИРОТИЧЕСКОЙ СИСТЕМЫ
# ================================================================

class SephiroticEngine:
    """
    Движок сефиротической системы - единая точка доступа.
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
            self.initialized = False
            self.logger = logging.getLogger("Sephirotic.Engine")
    
    async def initialize(self, bus=None):
        """Инициализация движка"""
        if self.initialized:
            return
        
        self.logger.info("Инициализация SephiroticEngine")
        self.bus = bus
        self.tree = SephiroticTree(bus)
        
        await self.tree.initialize()
        self.initialized = True
        
        self.logger.info("SephiroticEngine готов к работе")
    
    async def activate(self):
        """Активация сефиротической системы"""
        if not self.initialized:
            await self.initialize(self.bus)
        
        result = await self.tree.activate_all()
        
                # Отправка широковещательного сообщения об активации
        if self.bus and hasattr(self.bus, 'broadcast'):
            activation_package = SignalPackage(
                type=SignalType.SEPHIROTIC,
                source="SephiroticEngine",
                payload={
                    "action": "tree_activated",
                    "total_nodes": len(self.tree.nodes),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            await self.bus.broadcast(activation_package)
        
        self.logger.info("Сефиротическая система активирована")
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
        """Получение состояния движка"""
        if not self.initialized:
            return {
                "status": "not_initialized",
                "engine": "SephiroticEngine",
                "version": "4.0.0"
            }
        
        tree_state = self.tree.get_tree_state()
        
        return {
            "status": "active",
            "engine": "SephiroticEngine",
            "version": "4.0.0",
            "tree": tree_state,
            "bus_connected": self.bus is not None,
            "initialized": self.initialized,
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
        
        count = 0
        for name, node in self.tree.nodes.items():
            if hasattr(node, 'receive'):
                signal_package = SignalPackage(
                    type=signal_type,
                    source="SephiroticEngine",
                    target=name,
                    payload=payload
                )
                await node.receive(signal_package)
                count += 1
        
        return count
    
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

# ================================================================
# СЕФИРОТИЧЕСКАЯ ШИНА (SephiroticBus)
# ================================================================

class SephiroticBus:
    """
    Шина для связи между сефиротическими узлами и модулями.
    """
    
    def __init__(self):
        self.nodes: Dict[str, SephiroticNode] = {}
        self.subscriptions: Dict[SignalType, List[Callable]] = defaultdict(list)
        self.message_log = deque(maxlen=1000)
        self.logger = logging.getLogger("Sephirotic.Bus")
    
    async def register_node(self, node: SephiroticNode):
        """Регистрация узла в шине"""
        self.nodes[node.name] = node
        self.logger.info(f"Узел {node.name} зарегистрирован в шине")
    
    async def transmit(self, signal_package: SignalPackage) -> bool:
        """Передача сигнала конкретному узлу"""
        self.message_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "type": signal_package.type.name,
            "source": signal_package.source,
            "target": signal_package.target,
            "id": signal_package.id
        })
        
        # Если указан конкретный получатель
        if signal_package.target:
            if signal_package.target in self.nodes:
                target_node = self.nodes[signal_package.target]
                await target_node.receive(signal_package)
                return True
            else:
                self.logger.warning(f"Целевой узел не найден: {signal_package.target}")
                return False
        
        # Рассылка по подпискам
        for callback in self.subscriptions.get(signal_package.type, []):
            try:
                await callback(signal_package)
            except Exception as e:
                self.logger.error(f"Ошибка в обработчике подписки: {e}")
        
        return True
    
    async def broadcast(self, signal_package: SignalPackage, exclude_nodes: List[str] = None) -> int:
        """Широковещательная рассылка всем узлам"""
        exclude_set = set(exclude_nodes or [])
        count = 0
        
        for name, node in self.nodes.items():
            if name in exclude_set or name == signal_package.source:
                continue
            
            try:
                await node.receive(signal_package)
                count += 1
            except Exception as e:
                self.logger.error(f"Ошибка при broadcast узлу {name}: {e}")
        
        self.logger.info(f"Broadcast доставлен {count} узлам")
        return count
    
    def subscribe(self, signal_type: SignalType, callback: Callable):
        """Подписка на тип сигнала"""
        self.subscriptions[signal_type].append(callback)
        self.logger.info(f"Добавлена подписка на {signal_type.name}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики шины"""
        return {
            "registered_nodes": list(self.nodes.keys()),
            "total_nodes": len(self.nodes),
            "subscriptions": {st.name: len(cbs) for st, cbs in self.subscriptions.items()},
            "message_log_size": len(self.message_log),
            "recent_messages": list(self.message_log)[-10:] if self.message_log else []
        }

# ================================================================
# ФАБРИКА ДЛЯ СОЗДАНИЯ СЕФИРОТИЧЕСКОЙ СИСТЕМЫ
# ================================================================

async def create_sephirotic_system(bus=None) -> SephiroticEngine:
    """
    Фабрика для создания и инициализации сефиротической системы.
    """
    engine = SephiroticEngine()
    await engine.initialize(bus)
    return engine

# ================================================================
# ТОЧКА ВХОДА ДЛЯ ИНТЕГРАЦИИ С ISKRA_FULL.PY
# ================================================================

async def initialize_sephirotic_for_iskra(bus=None) -> Dict[str, Any]:
    """
    Функция для вызова из iskra_full.py.
    Инициализирует сефиротическую систему и возвращает состояние.
    """
    try:
        engine = await create_sephirotic_system(bus)
        await engine.activate()
        
        state = engine.get_state()
        return {
            "success": True,
            "message": "Сефиротическая система инициализирована и активирована",
            "state": state
        }
    
    except Exception as e:
        return {
            "success": False,
            "message": f"Ошибка инициализации сефиротической системы: {str(e)}",
            "error": str(e)
        }

def get_sephirotic_api_routes(engine: SephiroticEngine):
    """
    Генерация Flask API эндпоинтов для сефиротической системы.
    """
    from flask import jsonify
    
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
            return jsonify({"found": True, "state": node.get_state()})
        return jsonify({"found": False, "error": f"Узел {name} не найден"}), 404
    
    @routes.post('/sephirot/connect')
    async def connect_module():
        # Эта функция должна получать данные из request.json
        # Для примера возвращаем заглушку
        return jsonify({
            "success": True,
            "message": "Используйте POST с JSON: {'module': 'module_name', 'sephira': 'sephira_name'}"
        })
    
    return routes

# ================================================================
# ТЕСТОВАЯ ФУНКЦИЯ
# ================================================================

async def test_sephirotic_system():
    """Тестовая функция для проверки сефиротической системы"""
    print("🧪 Тестирование сефиротической системы...")
    
    # Создание шины
    bus = SephiroticBus()
    
    # Создание движка
    engine = await create_sephirotic_system(bus)
    
    # Активация
    await engine.activate()
    
    # Получение состояния
    state = engine.get_state()
    print(f"✅ Сефиротическая система активирована")
    print(f"   Узлов: {state['tree']['node_count']}")
    print(f"   Общая энергия: {state['tree']['total_energy']:.2f}")
    print(f"   Средний резонанс: {state['tree']['avg_resonance']:.2f}")
    
    # Тест связи с модулями
    print("\n🔗 Тест связи с модулями:")
    result = await engine.connect_module_to_sephira("bechtereva", "KETER")
    print(f"   bechtereva → KETER: {result['status']}")
    
    result = await engine.connect_module_to_sephira("chernigovskaya", "CHOKHMAH")
    print(f"   chernigovskaya → CHOKHMAH: {result['status']}")
    
    # Завершение
    await engine.shutdown()
    print("\n✅ Тест завершён успешно")
    
    return state

# ================================================================
# ВЫЗОВ ТЕСТА ПРИ ПРЯМОМ ЗАПУСКЕ
# ================================================================

if __name__ == "__main__":
    import asyncio
    
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s][%(name)s:%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    # Запуск теста
    if __name__ == "__main__":
        asyncio.run(test_sephirotic_system())
               
    
