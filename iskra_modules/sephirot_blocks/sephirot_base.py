#!/usr/bin/env python3
"""
sephirot_base.py - ИДЕАЛЬНЫЙ БАЗОВЫЙ КЛАСС СЕФИРОТИЧЕСКОГО УЗЛА
Архитектура: DS24 Sephirotic Protocol v3.0 (Complete Resonance Ecosystem)
"""

import json
import logging
import asyncio
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Set, Deque
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from contextlib import asynccontextmanager
import time

# ================================================================
# ADVANCED TYPES AND STRUCTURES
# ================================================================

class SignalType(Enum):
    """Типы сигналов для интеллектуальной маршрутизации"""
    EMOTIONAL = "emotional"
    COGNITIVE = "cognitive"
    INTENTION = "intention"
    HEARTBEAT = "heartbeat"
    RESONANCE = "resonance"
    COMMAND = "command"
    DATA = "data"
    ERROR = "error"
    SYNTHESIS = "synthesis"
    ENERGY = "energy"
    SYNC = "synchronization"
    METRIC = "metric"

class ResonanceState(Enum):
    """Состояния резонансной динамики с порогами перехода"""
    QUIESCENT = ("quiescent", 0.0, 0.2)      # Покой
    AWAKENING = ("awakening", 0.2, 0.4)      # Пробуждение
    RESONANT = ("resonant", 0.4, 0.7)        # Резонанс
    PEAK = ("peak", 0.7, 0.9)                # Пик
    OVERLOAD = ("overload", 0.9, 1.0)        # Перегрузка
    
    def __init__(self, label, min_val, max_val):
        self.label = label
        self.min = min_val
        self.max = max_val
    
    @classmethod
    def from_value(cls, value: float) -> 'ResonanceState':
        """Определение состояния по значению резонанса"""
        for state in cls:
            if state.min <= value < state.max:
                return state
        return cls.OVERLOAD

@dataclass
class ResonanceLink:
    """Умная двусторонняя связь с историей взаимодействий"""
    target: str
    strength: float = 0.5
    established: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_activity: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    channel_type: str = "bidirectional"
    history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    sync_coefficient: float = 0.8
    
    def __post_init__(self):
        """Инициализация после создания dataclass"""
        self.history.append(self.strength)
    
    def decay(self, decay_rate: float = 0.97, min_strength: float = 0.05) -> float:
        """Адаптивное затухание с учётом истории"""
        decay_factor = decay_rate * (1.0 - statistics.stdev(self.history) if len(self.history) > 1 else 1.0)
        self.strength *= decay_factor
        self.strength = max(min_strength, self.strength)
        self.history.append(self.strength)
        return self.strength
    
    def reinforce(self, amount: float = 0.1, max_strength: float = 1.0) -> float:
        """Усиление с насыщением и учетом синхронизации"""
        effective_amount = amount * self.sync_coefficient
        self.strength = min(max_strength, self.strength + effective_amount)
        self.last_activity = datetime.utcnow().isoformat()
        self.history.append(self.strength)
        return self.strength
    
    def sync_with(self, other_link: 'ResonanceLink') -> float:
        """Двусторонняя синхронизация силы связи"""
        avg_strength = (self.strength + other_link.strength) / 2
        self.strength = avg_strength * self.sync_coefficient
        other_link.strength = avg_strength * other_link.sync_coefficient
        self.history.append(self.strength)
        other_link.history.append(other_link.strength)
        return avg_strength
    
    def get_trend(self) -> str:
        """Определение тренда силы связи"""
        if len(self.history) < 2:
            return "stable"
        
        recent = list(self.history)[-5:]  # Последние 5 значений
        if len(recent) < 2:
            return "stable"
        
        diff = recent[-1] - recent[0]
        if diff > 0.05:
            return "growing"
        elif diff < -0.05:
            return "decaying"
        return "stable"

@dataclass
class SignalLog:
    """Структурированное логирование с метаданными"""
    timestamp: str
    signal_type: SignalType
    source: str
    channel: str
    payload: Dict[str, Any]
    processing_time: float = 0.0
    processed: bool = False
    response: Optional[Dict[str, Any]] = None
    energy_cost: float = 0.0

@dataclass
class NodeMetrics:
    """Живые метрики узла"""
    timestamp: str
    resonance_value: float
    resonance_state: str
    energy_level: float
    signal_count_1m: int = 0
    signal_count_5m: int = 0
    avg_processing_time: float = 0.0
    queue_size: int = 0
    active_links: int = 0
    memory_usage: int = 0
    stability_index: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для JSON сериализации"""
        return {
            "timestamp": self.timestamp,
            "resonance": round(self.resonance_value, 4),
            "resonance_state": self.resonance_state,
            "energy": round(self.energy_level, 4),
            "signals_1m": self.signal_count_1m,
            "signals_5m": self.signal_count_5m,
            "avg_process_time": round(self.avg_processing_time, 6),
            "queue": self.queue_size,
            "links": self.active_links,
            "memory": self.memory_usage,
            "stability": round(self.stability_index, 4)
        }

# ================================================================
# PERFECT SEPHIROTIC NODE CLASS
# ================================================================

class SephiroticNode:
    """
    ИДЕАЛЬНЫЙ базовый класс для сефиротических узлов ISKRA-4.
    Поддерживает адаптивную энергетику, буферизацию сигналов,
    двустороннюю синхронизацию и живые метрики.
    """
    
    # Оптимизированные константы
    MAX_MEMORY = 150
    SIGNAL_QUEUE_SIZE = 100
    RESONANCE_DECAY_BASE = 0.96
    ENERGY_DECAY_RATE = 0.98
    ENERGY_RECOVERY_RATE = 0.02
    METRICS_UPDATE_INTERVAL = 5.0  # секунд
    
    def __init__(self, name: str, level: int, bus=None):
        """
        Инициализация с расширенной конфигурацией.
        
        :param name: Имя узла (Kether, Chokhmah, Binah, ...)
        :param level: Уровень на Древе (1-10)
        :param bus: Ссылка на SephiroticBus
        """
        self.name = name
        self.level = level
        self.bus = bus
        
        # Ядро состояния
        self._state = {
            "activated": False,
            "resonance": 0.0,
            "energy": 0.7,
            "stability": 0.85,
            "cycle": 0,
            "last_state_change": None
        }
        
        # Расширенные структуры
        self.links: Dict[str, ResonanceLink] = {}  # target_name -> link
        self.signal_queue: asyncio.Queue = asyncio.Queue(maxsize=self.SIGNAL_QUEUE_SIZE)
        self.signal_log: Deque[SignalLog] = deque(maxlen=self.MAX_MEMORY)
        self.metrics_history: Deque[NodeMetrics] = deque(maxlen=60)  # 5 минут истории
        self.active_channels: Set[str] = set()
        
        # Обработчики сигналов
        self.signal_handlers = self._init_signal_handlers()
        
        # Асинхронные задачи
        self._processing_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._energy_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        
        # Статистика
        self._signal_timestamps: Deque[float] = deque(maxlen=300)  # 5 минут
        self._processing_times: Deque[float] = deque(maxlen=100)
        self._energy_history: Deque[float] = deque(maxlen=100)
        
        # Логирование
        self.logger = self._setup_logger()
        
        # Инициализация
        self._initialize_node()
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка продвинутого логгера"""
        logger = logging.getLogger(f"Sephirot.{self.name}")
        
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            
            # Console handler
            console = logging.StreamHandler()
            console.setFormatter(logging.Formatter(
                f'[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            ))
            logger.addHandler(console)
            
            # File handler для метрик
            file_handler = logging.FileHandler(f"logs/{self.name}.log", mode='a')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s,%(name)s,%(levelname)s,%(message)s'
            ))
            logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_node(self):
        """Начальная инициализация узла"""
        self.logger.info(f"Инициализация узла {self.name} (уровень {self.level})")
        
        # Регистрация в шине
        if self.bus and hasattr(self.bus, 'register_node'):
            self.bus.register_node(self)
        
        # Установка начального состояния
        self._state["last_state_change"] = datetime.utcnow().isoformat()
    
    def _init_signal_handlers(self) -> Dict[SignalType, callable]:
        """Инициализация обработчиков сигналов"""
        return {
            SignalType.EMOTIONAL: self._handle_emotion,
            SignalType.COGNITIVE: self._handle_cognition,
            SignalType.INTENTION: self._handle_intention,
            SignalType.HEARTBEAT: self._handle_heartbeat,
            SignalType.RESONANCE: self._handle_resonance,
            SignalType.COMMAND: self._handle_command,
            SignalType.DATA: self._handle_data,
            SignalType.ERROR: self._handle_error,
            SignalType.SYNTHESIS: self._handle_synthesis,
            SignalType.ENERGY: self._handle_energy,
            SignalType.SYNC: self._handle_sync,
            SignalType.METRIC: self._handle_metric
        }
    
    # ================================================================
    # CORE LIFECYCLE METHODS
    # ================================================================
    
    async def activate(self) -> Dict[str, Any]:
        """
        Полная активация узла с запуском всех фоновых задач.
        
        :return: Детальный отчет об активации
        """
        if self._state["activated"]:
            return {"status": "already_active", "node": self.name}
        
        self._state["activated"] = True
        self._state["last_state_change"] = datetime.utcnow().isoformat()
        
        # Запуск фоновых задач
        self._processing_task = asyncio.create_task(self._signal_processor())
        self._metrics_task = asyncio.create_task(self._metrics_collector())
        self._energy_task = asyncio.create_task(self._energy_manager())
        
        activation_report = {
            "status": "activated",
            "node": self.name,
            "level": self.level,
            "timestamp": self._state["last_state_change"],
            "initial_resonance": self._state["resonance"],
            "initial_energy": self._state["energy"],
            "tasks_started": [
                "signal_processor",
                "metrics_collector",
                "energy_manager"
            ]
        }
        
        self.logger.info(f"Активирован: {activation_report}")
        
        # Сигнализация об активации
        await self._emit_async({
            "type": "node_activated",
            "node": self.name,
            "level": self.level,
            "timestamp": self._state["last_state_change"]
        })
        
        return activation_report
    
    async def enqueue_signal(self, signal: Dict[str, Any], channel: str) -> bool:
        """
        Буферизованное добавление сигнала в очередь обработки.
        
        :param signal: Данные сигнала
        :param channel: Канал получения
        :return: True если успешно добавлен, False если очередь переполнена
        """
        try:
            # Обогащение метаданными
            enriched_signal = {
                **signal,
                "_enqueued_at": time.time(),
                "_channel": channel,
                "_node": self.name
            }
            
            # Немедленное добавление в очередь
            await asyncio.wait_for(
                self.signal_queue.put((enriched_signal, channel)),
                timeout=0.1
            )
            
            # Статистика
            self._signal_timestamps.append(time.time())
            
            self.logger.debug(f"Сигнал добавлен в очередь: {signal.get('type', 'unknown')}")
            return True
            
        except asyncio.TimeoutError:
            self.logger.warning(f"Очередь переполнена, сигнал отклонён: {signal.get('type', 'unknown')}")
            return False
        except Exception as e:
            self.logger.error(f"Ошибка добавления в очередь: {e}")
            return False
    
    async def _signal_processor(self):
        """
        Асинхронный процессор сигналов из очереди.
        Обрабатывает сигналы в порядке поступления.
        """
        self.logger.info("Запущен процессор сигналов")
        
        while not self._stop_event.is_set():
            try:
                # Получение сигнала из очереди с таймаутом
                try:
                    signal_data, channel = await asyncio.wait_for(
                        self.signal_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Обработка сигнала
                start_time = time.time()
                response = await self._process_signal_optimized(signal_data, channel)
                processing_time = time.time() - start_time
                
                # Сохранение статистики
                self._processing_times.append(processing_time)
                
                # Расчет энергозатрат
                energy_cost = processing_time * 0.1 + len(str(signal_data)) * 0.0001
                self._state["energy"] = max(0.1, self._state["energy"] - energy_cost)
                
                # Логирование
                signal_type = SignalType(signal_data.get("type", "data"))
                signal_log = SignalLog(
                    timestamp=datetime.utcnow().isoformat(),
                    signal_type=signal_type,
                    source=signal_data.get("_from", "unknown"),
                    channel=channel,
                    payload=signal_data,
                    processing_time=processing_time,
                    processed=True,
                    response=response,
                    energy_cost=energy_cost
                )
                
                self.signal_log.append(signal_log)
                
                # Подтверждение обработки
                self.signal_queue.task_done()
                
                # Обновление резонанса
                resonance_delta = response.get("resonance_delta", 0.05)
                await self._update_resonance_adaptive(resonance_delta)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка в процессоре сигналов: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_signal_optimized(self, signal: Dict[str, Any], channel: str) -> Dict[str, Any]:
        """
        Оптимизированная обработка сигнала с кэшированием и предсказанием.
        
        :param signal: Данные сигнала
        :param channel: Канал получения
        :return: Структурированный ответ
        """
        # Определение типа сигнала
        signal_type = SignalType(signal.get("type", "data"))
        
        # Получение обработчика
        handler = self.signal_handlers.get(signal_type)
        if not handler:
            handler = self._handle_unknown
        
        # Выполнение обработки
        try:
            result = handler(signal)
            
            # Генерация резонансного отклика
            resonance_feedback = await self._generate_resonance_feedback(
                signal_type=signal_type,
                signal_data=signal,
                channel=channel
            )
            
            return {
                "node": self.name,
                "signal_id": signal.get("_id", "unknown"),
                "processed_at": datetime.utcnow().isoformat(),
                "handler": signal_type.value,
                "result": result,
                "resonance_feedback": resonance_feedback,
                "resonance_delta": self._calculate_adaptive_delta(signal),
                "energy_level": self._state["energy"],
                "queue_size": self.signal_queue.qsize()
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки сигнала {signal_type.value}: {e}")
            return {
                "error": str(e),
                "node": self.name,
                "signal_type": signal_type.value
            }
    
    async def _energy_manager(self):
        """
        Управление адаптивной энергетикой узла.
        Узлы 'устают' при активной работе и 'восстанавливаются' в покое.
        """
        self.logger.info("Запущен менеджер энергии")
        
        while not self._stop_event.is_set():
            try:
                # Текущая активность
                recent_signals = self._count_recent_signals(60)  # Последняя минута
                activity_level = min(1.0, recent_signals / 10.0)  # Нормализация
                
                # Расчет изменения энергии
                if activity_level > 0.3:
                    # Активная фаза - расход энергии
                    energy_decay = self.ENERGY_DECAY_RATE * (1.0 - activity_level * 0.5)
                    self._state["energy"] *= energy_decay
                else:
                    # Фаза восстановления
                    recovery = self.ENERGY_RECOVERY_RATE * (1.0 - activity_level)
                    self._state["energy"] = min(1.0, self._state["energy"] + recovery)
                
                # Ограничение диапазона
                self._state["energy"] = max(0.1, min(1.0, self._state["energy"]))
                self._energy_history.append(self._state["energy"])
                
                # Эмитация уровня энергии при значительных изменениях
                if len(self._energy_history) >= 2:
                    if abs(self._energy_history[-1] - self._energy_history[-2]) > 0.1:
                        await self._emit_async({
                            "type": "energy_update",
                            "node": self.name,
                            "energy": self._state["energy"],
                            "activity": activity_level,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                
                await asyncio.sleep(2.0)  # Обновление каждые 2 секунды
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка в менеджере энергии: {e}")
                await asyncio.sleep(5.0)
    
    def _count_recent_signals(self, seconds: int) -> int:
        """Подсчет сигналов за последние N секунд"""
        cutoff = time.time() - seconds
        return sum(1 for ts in self._signal_timestamps if ts > cutoff)
    
    # ================================================================
    # RESONANCE DYNAMICS (ADAPTIVE)
    # ================================================================
    
    async def _update_resonance_adaptive(self, delta: float):
        """
        Адаптивное обновление резонанса с учетом текущего состояния.
        
        :param delta: Изменение резонанса
        """
        # Учет энергетического состояния
        energy_factor = self._state["energy"] * 0.5 + 0.5
        
        # Учет стабильности
        stability_factor = self._state["stability"] * 0.8 + 0.2
        
        # Комбинированный коэффициент
        adaptive_factor = energy_factor * stability_factor
        
        # Экспоненциальное обновление
        new_resonance = (
            self._state["resonance"] * self.RESONANCE_DECAY_BASE +
            delta * adaptive_factor * (1 - self.RESONANCE_DECAY_BASE)
        )
        
        # Ограничение
        new_resonance = max(0.0, min(1.0, new_resonance))
        
        # Определение изменения состояния
        old_state = ResonanceState.from_value(self._state["resonance"])
        new_state = ResonanceState.from_value(new_resonance)
        
        # Обновление
        self._state["resonance"] = new_resonance
        
        # Логирование изменения состояния
        if old_state != new_state:
            self._state["last_state_change"] = datetime.utcnow().isoformat()
            
            self.logger.info(
                f"Изменение резонанса: {old_state.label} ({self._state['resonance']:.3f}) → "
                f"{new_state.label} ({new_resonance:.3f})"
            )
            
            # Эмитация события
            await self._emit_async({
                "type": "resonance_state_transition",
                "node": self.name,
                "old_state": old_state.label,
                "new_state": new_state.label,
                "value": new_resonance,
                "timestamp": self._state["last_state_change"]
            })
    
    def _calculate_adaptive_delta(self, signal: Dict[str, Any]) -> float:
        """Адаптивный расчет изменения резонанса"""
        base_delta = 0.05
        
        # Факторы влияния
        intensity = signal.get("intensity", 0.5)
        complexity = signal.get("complexity", 0.3)
        emotional_content = signal.get("emotional_weight", 0.0)
        
        # Влияние типа сигнала
        signal_type = SignalType(signal.get("type", "data"))
        type_multipliers = {
            SignalType.EMOTIONAL: 1.2,
            SignalType.INTENTION: 1.5,
            SignalType.RESONANCE: 2.0,
            SignalType.SYNTHESIS: 1.8,
            SignalType.HEARTBEAT: 0.3,
            SignalType.DATA: 0.5
        }
        
        type_factor = type_multipliers.get(signal_type, 1.0)
        
        # Комбинированный расчет
        delta = (
            base_delta * intensity * type_factor +
            complexity * 0.1 +
            emotional_content * 0.05 -
            0.03  # Базовая утечка
        )
        
        # Ограничение с учетом энергии
        energy_limit = self._state["energy"] * 0.3
        return max(-0.15, min(energy_limit, delta))
    
    async def _generate_resonance_feedback(self, signal_type: SignalType, 
                                         signal_data: Dict[str, Any], 
                                         channel: str) -> Dict[str, Any]:
        """
        Генерация интеллектуального резонансного отклика.
        
        :param signal_type: Тип сигнала
        :param signal_data: Данные сигнала
        :param channel: Канал получения
        :return: Структурированный отклик
        """
        # Базовая сила отклика
        base_strength = min(self._state["resonance"] * 0.8, 0.7)
        
        # Усиление для определенных типов
        if signal_type in [SignalType.EMOTIONAL, SignalType.RESONANCE, SignalType.SYNTHESIS]:
            base_strength *= 1.3
        
        # Формирование контекста
        context = {
            "signal_type": signal_type.value,
            "channel": channel,
            "node_state": self.get_state(),
            "processing_capacity": self.signal_queue.maxsize - self.signal_queue.qsize(),
            "energy_level": self._state["energy"]
        }
        
        # Структурированный отклик
        feedback = {
            "type": "resonance_feedback",
            "value": round(base_strength, 4),
            "source": self.name,
            "resonance_state": ResonanceState.from_value(self._state["resonance"]).label,
            "context": context,
            "suggested_action": self._suggest_action(signal_type, base_strength),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.logger.debug(f"Сгенерирован отклик: сила={base_strength:.3f}")
        
        return feedback
    
    def _suggest_action(self, signal_type: SignalType, strength: float) -> str:
        """Предложение действия на основе типа сигнала и силы"""
        if strength < 0.3:
            return "observe"
        elif signal_type == SignalType.EMOTIONAL:
            return "resonate" if strength > 0.5 else "balance"
        elif signal_type == SignalType.INTENTION:
            return "amplify" if strength > 0.6 else "redirect"
        elif signal_type == SignalType.RESONANCE:
            return "synchronize"
        elif signal_type == SignalType.ERROR:
            return "stabilize"
        else:
            return "process"
    
    # ================================================================
    # LINK MANAGEMENT (BIDIRECTIONAL)
    # ================================================================
    
    async def create_link(self, target_node: 'SephiroticNode', 
                         initial_strength: float = 0.5,
                         channel_type: str = "bidirectional") -> Dict[str, Any]:
        """
        Создание умной двусторонней связи с синхронизацией.
        
        :param target_node: Целевой узел
        :param initial_strength: Начальная сила связи
        :param channel_type: Тип канала
        :return: Отчет о создании связи
        """
        # Создание связи на этом узле
        link = ResonanceLink(
            target=target_node.name,
            strength=initial_strength,
            channel_type=channel_type
        )
        
        self.links[target_node.name] = link
        
        # Попытка создания взаимной связи
        mutual_link = None
        if hasattr(target_node, 'create_link'):
            try:
                mutual_result = await target_node.create_link(
                    self, initial_strength, channel_type
                )
                if "link" in mutual_result:
                    mutual_link = mutual_result["link"]
                    
                    # Синхронизация сил связи
                    if isinstance(mutual_link, ResonanceLink):
                        link.sync_with(mutual_link)
            except Exception as e:
                self.logger.warning(f"Не удалось создать взаимную связь: {e}")
        
        # Эмитация события
        await self._emit_async({
            "type": "link_established",
            "from": self.name,
            "to": target_node.name,
            "strength": link.strength,
            "channel_type": channel_type,
            "mutual": mutual_link is not None,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        report = {
            "status": "link_created",
            "from": self.name,
            "to": target_node.name,
            "strength": link.strength,
            "channel_type": channel_type,
            "link": link,
            "mutual_established": mutual_link is not None
        }
        
        self.logger.info(
            f"Создана связь {self.name}↔{target_node.name}, "
            f"сила: {link.strength:.3f}, тип: {channel_type}"
        )
        
        return report
    
    async def synchronize_link(self, target_name: str) -> Dict[str, Any]:
        """
        Синхронизация связи с целевым узлом.
        
        :param target_name: Имя целевого узла
        :return: Результат синхронизации
        """
        if target_name not in self.links:
            return {"error": "link_not_found", "target": target_name}
        
        link = self.links[target_name]
        
        # Поиск целевого узла через шину
        if self.bus and hasattr(self.bus, 'get_node'):
            target_node = self.bus.get_node(target_name)
            if target_node and target_name in target_node.links:
                target_link = target_node.links[self.name]
                avg_strength = link.sync_with(target_link)
                
                return {
                    "status": "synchronized",
                    "from": self.name,
                    "to": target_name,
                    "new_strength": link.strength,
                    "average_strength": avg_strength,
                    "trend": link.get_trend(),
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        # Если взаимная связь не найдена, просто обновляем текущую
        link.strength = link.strength * 0.9 + 0.05  # Стабилизация
        return {
            "status": "stabilized",
            "link": target_name,
            "new_strength": link.strength,
            "note": "mutual_link_not_found"
        }
    
    async def update_links(self):
        """Обновление всех связей узла"""
        results = []
        for target_name in list(self.links.keys()):
            try:
                result = await self.synchronize_link(target_name)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Ошибка обновления связи {target_name}: {e}")
                results.append({"error": str(e), "target": target_name})
        
        # Очистка слабых связей
        weak_links = [name for name, link in self.links.items() if link.strength < 0.05]
        for name in weak_links:
            del self.links[name]
            self.logger.info(f"Удалена слабая связь: {name}")
        
        return {
            "links_updated": len(results),
            "weak_links_removed": len(weak_links),
            "results": results
        }
    
    # ================================================================
    # METRICS COLLECTION
    # ================================================================
    
    async def _metrics_collector(self):
        """Сбор живых метрик узла"""
        self.logger.info("Запущен сборщик метрик")
        
        while not self._stop_event.is_set():
            try:
                metrics = await self._collect_current_metrics()
                self.metrics_history.append(metrics)
                
                # Публикация метрик каждые 2 цикла
                if len(self.metrics_history) % 2 == 0:
                    await self._emit_async({
                        "type": "metrics_update",
                        "node": self.name,
                        "metrics": metrics.to_dict(),
                        "timestamp": metrics.timestamp
                    })
                
                await asyncio.sleep(self.METRICS_UPDATE_INTERVAL)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка в сборщике метрик: {e}")
                await asyncio.sleep(10.0)
    
    async def _collect_current_metrics(self) -> NodeMetrics:
        """Сбор текущих метрик"""
        current_time = datetime.utcnow().isoformat()
        
        # Расчет статистики сигналов
        signal_count_1m = self._count_recent_signals(60)
        signal_count_5m = self._count_recent_signals(300)
        
        # Расчет среднего времени обработки
        avg_processing_time = 0.0
        if self._processing_times:
            avg_processing_time = statistics.mean(self._processing_times)
        
        # Определение состояния резонанса
        resonance_state = ResonanceState.from_value(self._state["resonance"])
        
        # Создание метрик
        metrics = NodeMetrics(
            timestamp=current_time,
            resonance_value=self._state["resonance"],
            resonance_state=resonance_state.label,
            energy_level=self._state["energy"],
            signal_count_1m=signal_count_1m,
            signal_count_5m=signal_count_5m,
            avg_processing_time=avg_processing_time,
            queue_size=self.signal_queue.qsize(),
            active_links=len(self.links),
            memory_usage=len(self.signal_log),
            stability_index=self._state["stability"]
        )
        
        return metrics
    
    def get_metrics(self, historical: bool = False) -> Dict[str, Any]:
        """
        Получение метрик узла.
        
        :param historical: Включать ли историю метрик
        :return: Структурированные метрики
        """
        current_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        result = {
            "node": self.name,
            "level": self.level,
            "activated": self._state["activated"],
            "current_timestamp": datetime.utcnow().isoformat()
        }
        
        if current_metrics:
            result["current_metrics"] = current_metrics.to_dict()
        
        if historical and self.metrics_history:
            result["historical_metrics"] = [
                metrics.to_dict() for metrics in list(self.metrics_history)[-20:]
            ]
        
        # Добавление статистики очереди
        result["queue_stats"] = {
            "current_size": self.signal_queue.qsize(),
            "max_size": self.signal_queue.maxsize,
            "usage_percent": (self.signal_queue.qsize() / self.signal_queue.maxsize) * 100
        }
        
        # Статистика связей
        link_stats = []
        for target, link in self.links.items():
            link_stats.append({
                "target": target,
                "strength": round(link.strength, 3),
                "trend": link.get_trend(),
                "channel": link.channel_type,
                "history_size": len(link.history)
            })
        
        result["links"] = {
            "total": len(self.links),
            "details": link_stats
        }
        
        return result
    
    # ===============================================================
