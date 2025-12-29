# ===============================================================
# DS24_CORE v10.0 — КОМПЛЕКСНАЯ ПРОДУКЦИОННАЯ РЕАЛИЗАЦИЯ
# ===============================================================

import os
import sys
import yaml
import json
import threading
import hashlib
import numpy as np
import logging
import asyncio
import aiohttp
import redis
import pickle
import jwt
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Callable, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import wraps
import secrets
import socket
from pathlib import Path
from pydantic import BaseModel, Field, validator, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Flask и Prometheus
from flask import Flask, jsonify, Response, render_template_string, request, abort, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from prometheus_client import Counter, Gauge, Histogram, generate_latest, REGISTRY
from werkzeug.middleware.dispatcher import DispatcherMiddleware

# ===============================================================
# PYDANTIC CONFIG MODEL (СХЕМА ВАЛИДАЦИИ)
# ===============================================================

class DS24Config(BaseModel):
    """Строго типизированная конфигурация DS24."""
    
    # Обязательные поля
    seed: str = Field(default="DS24_PRODUCTION_SEED", min_length=10)
    log_level: str = Field(default="INFO", regex="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    flask_host: str = Field(default="0.0.0.0")
    flask_port: int = Field(default=5000, ge=1024, le=65535)
    
    # Настройки производительности
    max_history: int = Field(default=1000, gt=0)
    cycle_interval: float = Field(default=2.0, gt=0)
    event_ttl: int = Field(default=3600, gt=0)  # TTL событий в секундах
    max_retries: int = Field(default=3, ge=0)
    
    # Безопасность
    api_key: Optional[str] = None
    jwt_secret: Optional[str] = None
    rate_limit_default: str = Field(default="100 per minute")
    enable_cors: bool = Field(default=True)
    
    # Интеграции
    iskra_base_url: Optional[str] = None
    redis_url: Optional[str] = None
    prometheus_enabled: bool = Field(default=True)
    
    # Настройки сефиротики
    node_initial_energy: float = Field(default=0.5, ge=0, le=1)
    path_weight_range: Tuple[float, float] = Field(default=(0.1, 1.0))
    energy_flow_factor: float = Field(default=0.02, gt=0)
    
    @validator('seed')
    def validate_seed_strength(cls, v):
        """Проверка энтропии seed."""
        if len(set(v)) < 8:
            raise ValueError('Seed must have at least 8 unique characters')
        return v
    
    @validator('iskra_base_url')
    def validate_url(cls, v):
        if v and not (v.startswith('http://') or v.startswith('https://')):
            raise ValueError('URL must start with http:// or https://')
        return v
    
    @validator('redis_url')
    def validate_redis_url(cls, v):
        if v and not v.startswith('redis://'):
            raise ValueError('Redis URL must start with redis://')
        return v
    
    class Config:
        env_file = '.env'
        env_prefix = 'DS24_'
        case_sensitive = False

# ===============================================================
# ADVANCED LOGGING WITH STRUCTURED LOGS
# ===============================================================

class StructuredLogger:
    """Структурированный логгер с поддержкой JSON и контекста."""
    
    def __init__(self, config: DS24Config):
        self.config = config
        self.logger = logging.getLogger("DS24_CORE")
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # Форматтер для структурированных логов
        self.json_formatter = StructuredFormatter()
        self.text_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s | ctx=%(ctx)s'
        )
        
        # File handler с ротацией
        file_handler = logging.FileHandler('logs/ds24_core.json')
        file_handler.setFormatter(self.json_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.text_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Контекст сессии
        self.context = {
            "session_id": hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8],
            "instance_id": socket.gethostname(),
            "version": "10.0"
        }
    
    def _enhance_context(self, extra: Dict = None) -> Dict:
        """Улучшение контекста лога."""
        ctx = self.context.copy()
        ctx.update({
            "timestamp": datetime.utcnow().isoformat(),
            "thread": threading.current_thread().name,
            "cycle_id": getattr(self, 'cycle_id', 0)
        })
        if extra:
            ctx.update(extra)
        return ctx
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, extra={"ctx": json.dumps(self._enhance_context(kwargs))})
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra={"ctx": json.dumps(self._enhance_context(kwargs))})
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, extra={"ctx": json.dumps(self._enhance_context(kwargs))}, exc_info=True)
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra={"ctx": json.dumps(self._enhance_context(kwargs))})
    
    def metric(self, name: str, value: float, tags: Dict = None):
        """Логирование метрик."""
        self.info(f"METRIC {name}: {value}", metric_name=name, metric_value=value, tags=tags or {})

class StructuredFormatter(logging.Formatter):
    """Форматтер для структурированных JSON логов."""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "context": json.loads(getattr(record, 'ctx', '{}'))
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)

# ===============================================================
# PERSISTENT EVENT QUEUE WITH REDIS
# ===============================================================

class PersistentEventQueue:
    """
    Постоянная очередь событий на Redis с поддержкой TTL и восстановлением.
    """
    
    def __init__(self, redis_url: str, namespace: str = "ds24"):
        self.namespace = namespace
        self.redis = redis.Redis.from_url(redis_url, decode_responses=False)
        self.logger = logging.getLogger(f"{namespace}.queue")
        
        # Проверка соединения
        try:
            self.redis.ping()
            self.logger.info(f"Connected to Redis at {redis_url}")
        except redis.ConnectionError as e:
            self.logger.error(f"Redis connection failed: {e}")
            raise
    
    def push(self, topic: str, message: Dict[str, Any], ttl: int = 3600) -> str:
        """Добавление события в очередь с TTL."""
        message_id = hashlib.md5(f"{topic}:{datetime.now().timestamp()}".encode()).hexdigest()[:12]
        key = f"{self.namespace}:queue:{topic}:{message_id}"
        
        message_data = {
            "id": message_id,
            "topic": topic,
            "data": message,
            "created_at": datetime.utcnow().isoformat(),
            "ttl": ttl
        }
        
        # Сериализация
        serialized = pickle.dumps(message_data)
        
        # Сохранение в Redis с TTL
        self.redis.setex(key, ttl, serialized)
        
        # Добавление в список необработанных
        list_key = f"{self.namespace}:pending:{topic}"
        self.redis.lpush(list_key, message_id)
        
        self.logger.debug(f"Event queued: {topic}/{message_id}")
        return message_id
    
    def pop(self, topic: str) -> Optional[Dict[str, Any]]:
        """Извлечение события из очереди."""
        list_key = f"{self.namespace}:pending:{topic}"
        
        # Получение ID события
        message_id = self.redis.rpop(list_key)
        if not message_id:
            return None
        
        message_id = message_id.decode()
        key = f"{self.namespace}:queue:{topic}:{message_id}"
        
        # Получение данных
        serialized = self.redis.get(key)
        if not serialized:
            self.logger.warning(f"Event {message_id} expired or missing")
            return None
        
        # Десериализация
        message_data = pickle.loads(serialized)
        
        # Удаление из хранилища
        self.redis.delete(key)
        
        # Перемещение в обработанные
        processed_key = f"{self.namespace}:processed:{topic}"
        self.redis.lpush(processed_key, message_id)
        self.redis.ltrim(processed_key, 0, 999)  # Храним только 1000 последних
        
        self.logger.debug(f"Event popped: {topic}/{message_id}")
        return message_data
    
    def retry_failed(self, topic: str, message_id: str, max_retries: int = 3):
        """Повторная обработка неудачного события."""
        retry_key = f"{self.namespace}:retries:{topic}:{message_id}"
        
        # Подсчёт попыток
        retry_count = self.redis.incr(retry_key)
        self.redis.expire(retry_key, 3600)  # TTL для счётчика
        
        if retry_count <= max_retries:
            # Возврат в очередь
            list_key = f"{self.namespace}:pending:{topic}"
            self.redis.lpush(list_key, message_id)
            self.logger.warning(f"Event {message_id} requeued (attempt {retry_count})")
        else:
            # Перемещение в failed
            failed_key = f"{self.namespace}:failed:{topic}"
            self.redis.lpush(failed_key, message_id)
            self.logger.error(f"Event {message_id} moved to failed queue after {max_retries} attempts")
    
    def get_stats(self, topic: Optional[str] = None) -> Dict[str, Any]:
        """Статистика очереди."""
        if topic:
            pending = self.redis.llen(f"{self.namespace}:pending:{topic}")
            processed = self.redis.llen(f"{self.namespace}:processed:{topic}")
            failed = self.redis.llen(f"{self.namespace}:failed:{topic}")
        else:
            # Агрегированная статистика
            pending = processed = failed = 0
            for key in self.redis.scan_iter(f"{self.namespace}:pending:*"):
                pending += self.redis.llen(key)
            for key in self.redis.scan_iter(f"{self.namespace}:processed:*"):
                processed += self.redis.llen(key)
            for key in self.redis.scan_iter(f"{self.namespace}:failed:*"):
                failed += self.redis.llen(key)
        
        return {
            "pending": pending,
            "processed": processed,
            "failed": failed,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def cleanup_expired(self):
        """Очистка просроченных событий."""
        expired_count = 0
        
        for key in self.redis.scan_iter(f"{self.namespace}:queue:*"):
            if self.redis.ttl(key) == -2:  # Ключ истёк
                self.redis.delete(key)
                expired_count += 1
        
        if expired_count > 0:
            self.logger.info(f"Cleaned up {expired_count} expired events")

# ===============================================================
# ADVANCED EVENT BUS WITH TTL AND PRIORITY QUEUES
# ===============================================================

@dataclass
class TimedMessage(Message):
    """Сообщение с временными метками TTL."""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None
    ttl: Optional[int] = None
    
    def __post_init__(self):
        if self.ttl and not self.expires_at:
            expires_dt = datetime.fromisoformat(self.created_at) + timedelta(seconds=self.ttl)
            self.expires_at = expires_dt.isoformat()
    
    def is_expired(self) -> bool:
        """Проверка истечения срока жизни."""
        if not self.expires_at:
            return False
        return datetime.utcnow() > datetime.fromisoformat(self.expires_at)

class PriorityEventBus:
    """
    Усовершенствованный EventBus с приоритетными очередями, TTL и транзакционностью.
    """
    
    def __init__(self, config: DS24Config, persistent_queue: Optional[PersistentEventQueue] = None):
        self.config = config
        self.persistent_queue = persistent_queue
        
        # Приоритетные очереди (0 - highest, 4 - lowest)
        self.queues: List[List[TimedMessage]] = [[] for _ in range(5)]
        self.queue_locks = [threading.RLock() for _ in range(5)]
        
        # Подписчики
        self.subscribers: Dict[str, List[Tuple[Callable, int]]] = {}  # topic -> [(callback, priority)]
        self.subscriber_lock = threading.RLock()
        
        # История с TTL
        self.history: List[TimedMessage] = []
        self.history_lock = threading.RLock()
        
        # Транзакции
        self.transactions: Dict[str, List[TimedMessage]] = {}
        
        # Worker thread для обработки очереди
        self.worker_running = False
        self.worker_thread: Optional[threading.Thread] = None
        
        # Метрики
        self.metrics = {
            "messages_published": 0,
            "messages_expired": 0,
            "messages_processed": 0,
            "queue_sizes": [0, 0, 0, 0, 0]
        }
        
        self.logger = logging.getLogger("DS24.EventBus")
        self.logger.info("PriorityEventBus initialized")
    
    def start_worker(self):
        """Запуск worker thread для обработки очереди."""
        if self.worker_running:
            return
        
        self.worker_running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True, name="EventBus-Worker")
        self.worker_thread.start()
        self.logger.info("EventBus worker started")
    
    def stop_worker(self):
        """Остановка worker thread."""
        self.worker_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
            self.logger.info("EventBus worker stopped")
    
    def publish(self, sender: str, topic: str, data: Dict[str, Any], 
                priority: int = 2, ttl: Optional[int] = None, 
                persistent: bool = False) -> TimedMessage:
        """
        Публикация сообщения с приоритетом и TTL.
        
        Args:
            priority: 0 (highest) to 4 (lowest)
            ttl: Time to live in seconds
            persistent: Сохранять в персистентной очереди
        """
        if priority < 0 or priority > 4:
            raise ValueError("Priority must be between 0 and 4")
        
        # Создание сообщения
        msg = TimedMessage(
            sender=sender,
            topic=topic,
            data=data,
            priority=priority,
            ttl=ttl or self.config.event_ttl
        )
        
        # Добавление в приоритетную очередь
        with self.queue_locks[priority]:
            self.queues[priority].append(msg)
            self.metrics["queue_sizes"][priority] += 1
        
        # Сохранение в персистентную очередь
        if persistent and self.persistent_queue:
            try:
                self.persistent_queue.push(topic, {
                    "sender": sender,
                    "data": data,
                    "priority": priority
                }, ttl=ttl or self.config.event_ttl)
            except Exception as e:
                self.logger.error(f"Failed to persist event: {e}")
        
        # Добавление в историю
        with self.history_lock:
            self.history.append(msg)
            # Очистка устаревших сообщений
            self.history = [m for m in self.history if not m.is_expired()]
        
        self.metrics["messages_published"] += 1
        self.logger.debug(f"Message published: {topic} (priority: {priority}, ttl: {ttl})")
        
        return msg
    
    def _process_queue(self):
        """Обработка очереди сообщений в worker thread."""
        while self.worker_running:
            try:
                # Обработка по приоритетам (от высокого к низкому)
                for priority in range(5):
                    with self.queue_locks[priority]:
                        if not self.queues[priority]:
                            continue
                        
                        # Извлечение сообщения
                        msg = self.queues[priority].pop(0)
                        self.metrics["queue_sizes"][priority] -= 1
                        
                        # Проверка TTL
                        if msg.is_expired():
                            self.metrics["messages_expired"] += 1
                            self.logger.debug(f"Message expired: {msg.topic}")
                            continue
                        
                        # Вызов подписчиков
                        self._deliver_message(msg)
                        self.metrics["messages_processed"] += 1
                
                # Пауза между циклами
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in queue processing: {e}")
                time.sleep(1)
    
    def _deliver_message(self, msg: TimedMessage):
        """Доставка сообщения подписчикам."""
        with self.subscriber_lock:
            if msg.topic not in self.subscribers:
                return
            
            # Сортировка подписчиков по приоритету
            subscribers = sorted(self.subscribers[msg.topic], key=lambda x: x[1])
            
            for callback, _ in subscribers:
                try:
                    callback(msg)
                except Exception as e:
                    self.logger.error(f"Error in subscriber for {msg.topic}: {e}")
    
    def begin_transaction(self, tx_id: str):
        """Начало транзакции событий."""
        self.transactions[tx_id] = []
    
    def commit_transaction(self, tx_id: str):
        """Фиксация транзакции - публикация всех накопленных событий."""
        if tx_id not in self.transactions:
            return
        
        for msg in self.transactions[tx_id]:
            self.publish(msg.sender, msg.topic, msg.data, msg.priority, msg.ttl)
        
        del self.transactions[tx_id]
    
    def rollback_transaction(self, tx_id: str):
        """Откат транзакции."""
        if tx_id in self.transactions:
            del self.transactions[tx_id]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Получение метрик EventBus."""
        return {
            **self.metrics,
            "history_size": len(self.history),
            "active_subscribers": sum(len(subs) for subs in self.subscribers.values()),
            "active_transactions": len(self.transactions)
        }

# ===============================================================
# TRANSACTIONAL KERNEL WITH ROLLBACK SUPPORT
# ===============================================================

class TransactionalDS24Kernel:
    """
    Транзакционное ядро DS24 с поддержкой rollback и контрольными точками.
    """
    
    def __init__(self, config: DS24Config):
        self.config = config
        self.logger = logging.getLogger("DS24.Kernel")
        
        # Инициализация компонентов
        self.state_checkpoints: List[Dict[str, Any]] = []
        self.current_transaction: Optional[str] = None
        self.rollback_depth = 0
        self.max_rollback_depth = 10
        
        # Базовые компоненты
        self.dgen = DeterministicGenerator(config.seed)
        self.bus = PriorityEventBus(config)
        self.tree = SephiroticTree(self.dgen)
        
        # Инициализация с транзакцией
        self.begin_transaction("init")
        self._initialize_components()
        self.commit_transaction("init")
        
        self.logger.info("Transactional DS24 Kernel initialized")
    
    def _initialize_components(self):
        """Инициализация компонентов в транзакции."""
        # Создание контрольной точки
        checkpoint = self._create_checkpoint("pre_init")
        
        try:
            # Инициализация агентов
            self.agents = [
                EntropyAgent("EntropyAgent", self.bus, self.dgen),
                DeterminismAgent("DeterminismAgent", self.bus),
                ReflexAgent("ReflexAgent", self.bus),
                TelemetryAgent("TelemetryAgent", self.bus)
            ]
            
            # Conscious State
            self.conscious = ConsciousState(self.bus)
            
            # Контекст
            self.context = {}
            self.pipeline = [a.name for a in self.agents]
            
            # История
            self.cycle_history = []
            
            self.logger.debug("Components initialized successfully")
            
        except Exception as e:
            # Rollback при ошибке
            self.logger.error(f"Initialization failed, rolling back: {e}")
            self._restore_checkpoint(checkpoint)
            raise
    
    def _create_checkpoint(self, name: str) -> Dict[str, Any]:
        """Создание контрольной точки состояния."""
        checkpoint = {
            "id": f"{name}_{datetime.now().timestamp()}",
            "name": name,
            "timestamp": datetime.utcnow().isoformat(),
            "context": self.context.copy() if hasattr(self, 'context') else {},
            "tree_state": self.tree.snapshot() if hasattr(self, 'tree') else {},
            "cycle_history": self.cycle_history.copy() if hasattr(self, 'cycle_history') else [],
            "agent_states": {a.name: a.get_status() for a in self.agents} if hasattr(self, 'agents') else {}
        }
        
        # Сериализация через pickle для сложных объектов
        checkpoint["pickled"] = pickle.dumps({
            "tree": self.tree if hasattr(self, 'tree') else None,
            "agents": self.agents if hasattr(self, 'agents') else [],
            "conscious": self.conscious if hasattr(self, 'conscious') else None
        })
        
        # Сохранение
        self.state_checkpoints.append(checkpoint)
        if len(self.state_checkpoints) > self.max_rollback_depth:
            self.state_checkpoints.pop(0)
        
        self.logger.debug(f"Checkpoint created: {name}")
        return checkpoint
    
    def _restore_checkpoint(self, checkpoint: Dict[str, Any]):
        """Восстановление состояния из контрольной точки."""
        try:
            # Десериализация объектов
            pickled_data = pickle.loads(checkpoint["pickled"])
            
            if pickled_data["tree"]:
                self.tree = pickled_data["tree"]
            if pickled_data["agents"]:
                self.agents = pickled_data["agents"]
            if pickled_data["conscious"]:
                self.conscious = pickled_data["conscious"]
            
            # Восстановление простых данных
            self.context = checkpoint["context"].copy()
            self.cycle_history = checkpoint["cycle_history"].copy()
            
            self.logger.info(f"State restored from checkpoint: {checkpoint['name']}")
            self.rollback_depth += 1
            
        except Exception as e:
            self.logger.error(f"Failed to restore checkpoint: {e}")
            raise
    
    def begin_transaction(self, tx_name: str):
        """Начало транзакции."""
        if self.current_transaction:
            raise RuntimeError(f"Transaction {self.current_transaction} already in progress")
        
        self.current_transaction = tx_name
        self.bus.begin_transaction(tx_name)
        
        # Создание контрольной точки
        self._create_checkpoint(f"tx_{tx_name}_start")
        
        self.logger.debug(f"Transaction started: {tx_name}")
    
    def commit_transaction(self, tx_name: str):
        """Фиксация транзакции."""
        if self.current_transaction != tx_name:
            raise RuntimeError(f"Transaction mismatch: expected {tx_name}, got {self.current_transaction}")
        
        self.bus.commit_transaction(tx_name)
        self.current_transaction = None
        
        self.logger.debug(f"Transaction committed: {tx_name}")
    
    def rollback_transaction(self, tx_name: str):
        """Откат транзакции."""
        if self.current_transaction != tx_name:
            raise RuntimeError(f"Transaction mismatch: expected {tx_name}, got {self.current_transaction}")
        
        # Откат EventBus
        self.bus.rollback_transaction(tx_name)
        
        # Восстановление последней контрольной точки
        if self.state_checkpoints:
            last_checkpoint = self.state_checkpoints[-1]
            if last_checkpoint["name"].startswith(f"tx_{tx_name}_start"):
                self._restore_checkpoint(last_checkpoint)
                # Удаление использованной контрольной точки
                self.state_checkpoints.pop()
        
        self.current_transaction = None
        self.logger.warning(f"Transaction rolled back: {tx_name}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=0.1, max=2),
        retry=retry_if_exception_type((RuntimeError, ValueError))
    )
    def execute_cycle_with_rollback(self) -> Dict[str, Any]:
        """
        Выполнение цикла с поддержкой rollback при ошибках.
        """
        tx_id = f"cycle_{len(self.cycle_history) + 1}"
        
        try:
            # Начало транзакции
            self.begin_transaction(tx_id)
            
            # Выполнение цикла
            result = self._execute_cycle_core()
            
            # Фиксация транзакции
            self.commit_transaction(tx_id)
            
            return result
            
        except Exception as e:
            # Откат при ошибке
            self.rollback_transaction(tx_id)
            
            # Логирование и повторная попытка через retry декоратор
            self.logger.error(f"Cycle failed, rolled back: {e}")
            raise
    
    def _execute_cycle_core(self) -> Dict[str, Any]:
        """Ядро выполнения цикла (без транзакционной логики)."""
        cycle_start = time.time()
        
        # Выполнение потока в дереве
        collapsed_node = self.tree.flow(destructive=True)
        
        # Адаптация пайплайна
        self._adapt_pipeline()
        
        # Выполнение агентов
        for agent_name in self.pipeline:
            agent = next((a for a in self.agents if a.name == agent_name), None)
            if agent:
                result = agent.pulse(self.context)
                self.context.update(result)
        
        # Обновление сознательного слоя
        coherence = self.tree.coherence_index()
        self.conscious.update(
            self.context.get("entropy", 0),
            self.context.get("determinism", 1),
            coherence
        )
        
        # Сохранение результата
        cycle_record = {
            "id": len(self.cycle_history) + 1,
            "timestamp": datetime.utcnow().isoformat(),
            "collapsed_node": collapsed_node,
            "context": self.context.copy(),
            "coherence": coherence,
            "duration": time.time() - cycle_start,
            "success": True
        }
        
        self.cycle_history.append(cycle_record)
        
        return cycle_record

# ===============================================================
# SECURE FLASK API WITH JWT AND RATE LIMITING
# ===============================================================

def create_secure_flask_app(kernel: TransactionalDS24Kernel, config: DS24Config) -> Flask:
    """Создание защищённого Flask приложения."""
    
    app = Flask(__name__)
    
    # Настройка JWT
    JWT_SECRET = config.jwt_secret or secrets.token_hex(32)
    JWT_ALGORITHM = "HS256"
    
    # Rate Limiter
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=[config.rate_limit_default]
    )
    
    # Middleware для CORS
    if config.enable_cors:
        @app.after_request
        def add_cors_headers(response):
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            return response
    
    # Декораторы безопасности
    def require_jwt(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            
            if not auth_header:
                abort(401, description="Missing Authorization header")
            
            try:
                scheme, token = auth_header.split()
                if scheme.lower() != 'bearer':
                    abort(401, description="Invalid authorization scheme")
                
                # Верификация JWT
                payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
                g.user = payload.get('sub')
                g.roles = payload.get('roles', [])
                
            except jwt.ExpiredSignatureError:
                abort(401, description="Token expired")
            except jwt.InvalidTokenError:
                abort(401, description="Invalid token")
            
            return f(*args, **kwargs)
        return decorated
    
    def require_role(role: str):
        def decorator(f):
            @wraps(f)
            @require_jwt
            def decorated(*args, **kwargs):
                if role not in g.roles:
                    abort(403, description=f"Role {role} required")
                return f(*args, **kwargs)
            return decorated
        return decorator
    
    def require_api_key(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if not config.api_key:
                return f(*args, **kwargs)
            
            provided_key = request.headers.get('X-API-Key') or request.args.get('api_key')
            
            if not provided_key or provided_key != config.api_key:
                abort(403, description="Invalid API key")
            
            return f(*args, **kwargs)
        return decorated
    
    # Роуты API
    @app.route('/api/auth/token', methods=['POST'])
    @limiter.limit("10 per minute")
    def get_token():
        """Получение JWT токена."""
        data = request.get_json()
        
        # Простая аутентификация (в production использовать базу данных)
        username = data.get('username')
        password = data.get('password')
        
        # Пример: проверка статических учётных данных
        if username == 'admin' and password == config.seed[:8]:
            token = jwt.encode({
                'sub': username,
                'roles': ['admin', 'user'],
                'exp': datetime.utcnow() + timedelta(hours=24),
                'iat': datetime.utcnow()
            }, JWT_SECRET, algorithm=JWT_ALGORITHM)
            
            return jsonify({'token': token})
        
        abort(401, description="Invalid credentials")
    
    @app.route('/api/state')
    @require_api_key
    @limiter.limit("60 per minute")
    def api_state():
        """Получение состояния системы."""
        return jsonify(kernel.snapshot())
    
    @app.route('/api/cycle', methods=['POST'])
    @require_jwt
    @require_role('admin')
    @limiter.limit("30 per minute")
    def api_cycle():
        """Запуск цикла ядра."""
        result = kernel.execute_cycle_with_rollback()
        return jsonify(result)
    
    @app.route('/api/health')
    @limiter.exempt
    def api_health():
        """Health check endpoint."""
        return jsonify(kernel.health())
    
    @app.route('/api/metrics')
    @require_jwt
    @require_role('admin')
    def api_metrics():
        """Prometheus метрики."""
        return Response(generate_latest(REGISTRY), mimetype='text/plain')
    
    @app.route('/api/queue/stats')
    @require_jwt
    @require_role('admin')
    def api_queue_stats():
        """Статистика очередей."""
        if hasattr(kernel, 'persistent_queue'):
            stats = kernel.persistent_queue.get_stats()
            return jsonify(stats)
        return jsonify({"error": "Persistent queue not enabled"})
    
    @app.route('/api/rollback/<int:cycles>', methods=['POST'])
    @require_jwt
    @require_role('admin')
    @limiter.limit("10 per minute")
    def api_rollback(cycles: int):
        """Откат на указанное количество циклов."""
        if cycles > kernel.max_rollback_depth:
            abort(400, description=f"Max rollback depth is {kernel.max_rollback_depth}")
        
        # Сохранение контрольной точки для отката
        checkpoint = kernel._create_checkpoint(f"pre_rollback_{cycles}")
        
        try:
            # Удаление последних cycles записей
            kernel.cycle_history = kernel.cycle_history[:-cycles]
            
            # Восстановление состояния из контрольной точки
            kernel._restore_checkpoint(checkpoint)
            
            return jsonify({
                "success": True,
                "message": f"Rolled back {cycles} cycles",
                "remaining_cycles": len(kernel.cycle_history)
            })
        except Exception as e:
            kernel._restore_checkpoint(checkpoint)
            abort(500, description=f"Rollback failed: {e}")
    
    @app.route('/dashboard')
    def dashboard():
        """Интерактивный дашборд."""
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>DS24 Quantum Dashboard</title>
        <style>
            body { font-family: 'Courier New', monospace; background: #0a0a0a; color: #00ff00; }
            .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .card { background: #1a1a1a; border: 1px solid #00ff00; padding: 15px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>DS24 Dashboard</h1>
            <div class="grid">
                <div class="card">Module Status</div>
                <div class="card">System Metrics</div>
            </div>
        </div>
    </body>
    </html>
    ''')
