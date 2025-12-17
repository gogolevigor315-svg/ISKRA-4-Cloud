# neocortex_core.py - СОВЕРШЕННЫЙ НЕОКОРТЕКС ИСКРА (ФИНАЛЬНАЯ ВЕРСИЯ)
import asyncio
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import statistics
import json
import hashlib
import networkx as nx
from enum import Enum
import pickle
from scipy import stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging
import functools
import redis.asyncio as redis
from pathlib import Path
import lz4.frame
import msgpack
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import aiohttp
from aiohttp import web, WSMsgType
import websockets
import sqlite3
import zstandard as zstd

logger = logging.getLogger(__name__)


# ============================================================================
# ВЕБСОКЕТ ИНТЕРФЕЙС ДЛЯ ДАННЫХ
# ============================================================================

class DataStreamWebSocket:
    """WebSocket интерфейс для непрерывного стриминга данных из Data Bridge"""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.connections: Set[websockets.WebSocketServerProtocol] = set()
        self.data_bridge = None
        self.stream_tasks = {}
        self.message_queue = asyncio.Queue(maxsize=10000)
        self.subscriptions = defaultdict(set)  # topic -> connections
        self.server = None
        
        # Статистика
        self.stats = {
            'total_messages': 0,
            'active_connections': 0,
            'bytes_transferred': 0,
            'subscriptions': defaultdict(int)
        }
        
        # Схемы валидации сообщений
        self.schemas = {
            'sensor_data': {
                'required': ['timestamp', 'source', 'data'],
                'types': {
                    'timestamp': str,
                    'source': str,
                    'data': (dict, list)
                }
            },
            'cognitive_event': {
                'required': ['event_type', 'timestamp', 'payload'],
                'types': {
                    'event_type': str,
                    'timestamp': str,
                    'payload': dict
                }
            }
        }
        
        logger.info(f"WebSocket интерфейс инициализирован: {host}:{port}")
    
    async def start(self):
        """Запуск WebSocket сервера"""
        self.server = await websockets.serve(
            self.websocket_handler,
            self.host,
            self.port
        )
        
        # Запуск обработки очереди сообщений
        asyncio.create_task(self._process_message_queue())
        
        # Запуск мониторинга соединений
        asyncio.create_task(self._monitor_connections())
        
        logger.info(f"WebSocket сервер запущен на ws://{self.host}:{self.port}")
    
    async def websocket_handler(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Обработчик WebSocket соединений"""
        connection_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        
        self.connections.add(websocket)
        self.stats['active_connections'] += 1
        
        logger.info(f"Новое WebSocket соединение: {connection_id}")
        
        try:
            # Отправка приветственного сообщения
            await websocket.send(json.dumps({
                'type': 'welcome',
                'connection_id': connection_id,
                'timestamp': datetime.utcnow().isoformat(),
                'capabilities': ['subscribe', 'publish', 'query'],
                'schema_versions': list(self.schemas.keys())
            }))
            
            # Основной цикл обработки сообщений
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_client_message(websocket, connection_id, data)
                    
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'error': 'invalid_json',
                        'message': 'Invalid JSON format'
                    }))
                except Exception as e:
                    logger.error(f"Ошибка обработки сообщения: {e}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'error': 'processing_error',
                        'message': str(e)
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket соединение закрыто: {connection_id}")
        except Exception as e:
            logger.error(f"Ошибка WebSocket соединения {connection_id}: {e}")
        finally:
            # Очистка
            self.connections.remove(websocket)
            self.stats['active_connections'] -= 1
            
            # Удаление из подписок
            for topic, connections in self.subscriptions.items():
                if websocket in connections:
                    connections.remove(websocket)
                    self.stats['subscriptions'][topic] -= 1
    
    async def _handle_client_message(self, websocket: websockets.WebSocketServerProtocol, 
                                    connection_id: str, data: Dict[str, Any]):
        """Обработка сообщений от клиента"""
        msg_type = data.get('type')
        
        if msg_type == 'subscribe':
            # Подписка на топик
            topic = data.get('topic')
            if topic:
                self.subscriptions[topic].add(websocket)
                self.stats['subscriptions'][topic] += 1
                
                await websocket.send(json.dumps({
                    'type': 'subscription_confirmed',
                    'topic': topic,
                    'timestamp': datetime.utcnow().isoformat()
                }))
                
                logger.debug(f"Соединение {connection_id} подписано на топик: {topic}")
        
        elif msg_type == 'unsubscribe':
            # Отписка от топика
            topic = data.get('topic')
            if topic and websocket in self.subscriptions.get(topic, set()):
                self.subscriptions[topic].remove(websocket)
                self.stats['subscriptions'][topic] -= 1
        
        elif msg_type == 'publish':
            # Публикация данных
            topic = data.get('topic')
            payload = data.get('payload', {})
            
            if topic:
                # Валидация по схеме
                schema_name = data.get('schema', 'sensor_data')
                if schema_name in self.schemas:
                    if not self._validate_schema(payload, self.schemas[schema_name]):
                        await websocket.send(json.dumps({
                            'type': 'error',
                            'error': 'validation_error',
                            'message': f'Invalid schema: {schema_name}'
                        }))
                        return
                
                # Постановка в очередь для рассылки
                await self.message_queue.put({
                    'type': 'broadcast',
                    'topic': topic,
                    'payload': payload,
                    'source': connection_id,
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        elif msg_type == 'query':
            # Запрос данных
            query_type = data.get('query_type')
            if query_type == 'stats':
                await websocket.send(json.dumps({
                    'type': 'stats_response',
                    'stats': self.stats,
                    'timestamp': datetime.utcnow().isoformat()
                }))
        
        elif msg_type == 'ping':
            # Ping/Pong
            await websocket.send(json.dumps({
                'type': 'pong',
                'timestamp': datetime.utcnow().isoformat()
            }))
    
    def _validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
        """Валидация данных по схеме"""
        # Проверка обязательных полей
        for field in schema.get('required', []):
            if field not in data:
                return False
        
        # Проверка типов
        for field, expected_type in schema.get('types', {}).items():
            if field in data:
                if not isinstance(data[field], expected_type):
                    return False
        
        return True
    
    async def _process_message_queue(self):
        """Обработка очереди сообщений и рассылка подписчикам"""
        while True:
            try:
                message = await self.message_queue.get()
                
                if message['type'] == 'broadcast':
                    topic = message['topic']
                    payload = message['payload']
                    
                    # Рассылка всем подписчикам топика
                    if topic in self.subscriptions:
                        dead_connections = []
                        
                        for websocket in self.subscriptions[topic]:
                            try:
                                await websocket.send(json.dumps({
                                    'type': 'data',
                                    'topic': topic,
                                    'payload': payload,
                                    'timestamp': message['timestamp'],
                                    'source': message.get('source', 'system')
                                }))
                                
                                self.stats['total_messages'] += 1
                                self.stats['bytes_transferred'] += len(json.dumps(payload))
                                
                            except (websockets.exceptions.ConnectionClosed, 
                                   websockets.exceptions.InvalidState):
                                dead_connections.append(websocket)
                        
                        # Очистка мертвых соединений
                        for ws in dead_connections:
                            self.subscriptions[topic].remove(ws)
                            if ws in self.connections:
                                self.connections.remove(ws)
                
                self.message_queue.task_done()
                
            except Exception as e:
                logger.error(f"Ошибка обработки очереди сообщений: {e}")
                await asyncio.sleep(0.1)
    
    async def _monitor_connections(self):
        """Мониторинг и обслуживание соединений"""
        while True:
            try:
                # Проверка активности соединений
                dead_connections = []
                for websocket in self.connections:
                    try:
                        # Отправка ping
                        pong_waiter = await websocket.ping()
                        await asyncio.wait_for(pong_waiter, timeout=10)
                    except:
                        dead_connections.append(websocket)
                
                # Очистка мертвых соединений
                for ws in dead_connections:
                    self.connections.remove(ws)
                    for topic, connections in self.subscriptions.items():
                        if ws in connections:
                            connections.remove(ws)
                
                # Логирование статистики
                if len(dead_connections) > 0:
                    logger.debug(f"Очищено {len(dead_connections)} неактивных соединений")
                
                await asyncio.sleep(30)  # Проверка каждые 30 секунд
                
            except Exception as e:
                logger.error(f"Ошибка мониторинга соединений: {e}")
                await asyncio.sleep(10)
    
    async def stream_from_data_bridge(self, data_bridge, stream_name: str = "sensor_stream"):
        """Потоковая передача данных из Data Bridge через WebSocket"""
        self.data_bridge = data_bridge
        
        async def stream_generator():
            while True:
                try:
                    # Получение данных из Data Bridge
                    data = await data_bridge.pull_recent_stream(stream_name)
                    if data:
                        # Валидация и отправка
                        validated_data = self._validate_and_transform(data)
                        
                        await self.message_queue.put({
                            'type': 'broadcast',
                            'topic': f'data_bridge/{stream_name}',
                            'payload': validated_data,
                            'source': 'data_bridge',
                            'timestamp': datetime.utcnow().isoformat()
                        })
                    
                    await asyncio.sleep(0.1)  # Контроль частоты
                    
                except Exception as e:
                    logger.error(f"Ошибка стриминга из Data Bridge: {e}")
                    await asyncio.sleep(1)
        
        # Запуск генератора
        self.stream_tasks[stream_name] = asyncio.create_task(stream_generator())
        logger.info(f"Запущен стриминг из Data Bridge: {stream_name}")
    
    def _validate_and_transform(self, data: Any) -> Dict[str, Any]:
        """Валидация и трансформация данных из Data Bridge"""
        if isinstance(data, dict):
            # Добавление метаданных
            data['_metadata'] = {
                'processed_at': datetime.utcnow().isoformat(),
                'schema_version': '1.0',
                'source': 'data_bridge'
            }
            return data
        elif isinstance(data, list):
            return {
                'items': data,
                'count': len(data),
                '_metadata': {
                    'processed_at': datetime.utcnow().isoformat(),
                    'schema_version': '1.0',
                    'source': 'data_bridge'
                }
            }
        else:
            return {
                'value': data,
                '_metadata': {
                    'processed_at': datetime.utcnow().isoformat(),
                    'schema_version': '1.0',
                    'source': 'data_bridge'
                }
            }


# ============================================================================
# СОХРАНЕНИЕ СОСТОЯНИЯ В БАЗУ ДАННЫХ
# ============================================================================

class NeurocortexStateDB:
    """База данных для сохранения состояния неокортекса"""
    
    def __init__(self, db_path: str = "data/neurocortex_state.db", 
                 redis_url: str = None,
                 compression_level: int = 3):
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.compression_level = compression_level
        
        # SQLite для структурированных данных
        self.sqlite_conn = None
        self._init_sqlite()
        
        # Redis для кэширования и быстрого доступа
        self.redis_client = None
        if redis_url:
            self._init_redis(redis_url)
        
        # Пул для фонового сохранения
        self.save_queue = asyncio.Queue(maxsize=1000)
        self.save_tasks = []
        
        # Статистика
        self.stats = {
            'total_saves': 0,
            'last_save': None,
            'save_errors': 0,
            'compression_ratio': 1.0
        }
        
        # Запуск фонового сохранения
        self._start_background_saver()
        
        logger.info(f"База данных состояния неокортекса инициализирована: {db_path}")
    
    def _init_sqlite(self):
        """Инициализация SQLite базы данных"""
        self.sqlite_conn = sqlite3.connect(self.db_path, check_same_thread=False)
        cursor = self.sqlite_conn.cursor()
        
        # Таблица состояний неокортекса
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS neurocortex_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                snapshot_id TEXT UNIQUE NOT NULL,
                component TEXT NOT NULL,
                state_type TEXT NOT NULL,
                state_data BLOB NOT NULL,
                compressed_size INTEGER,
                original_size INTEGER,
                checksum TEXT,
                metadata TEXT
            )
        ''')
        
        # Таблица когнитивных событий
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cognitive_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                event_type TEXT NOT NULL,
                component TEXT NOT NULL,
                event_data TEXT NOT NULL,
                importance REAL DEFAULT 0.5,
                processed BOOLEAN DEFAULT 0,
                metadata TEXT
            )
        ''')
        
        # Таблица паттернов и схем
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cognitive_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                pattern_id TEXT NOT NULL,
                pattern_type TEXT NOT NULL,
                pattern_data BLOB NOT NULL,
                activation_history TEXT,
                confidence REAL DEFAULT 0.5,
                usage_count INTEGER DEFAULT 0,
                last_used DATETIME,
                metadata TEXT
            )
        ''')
        
        # Индексы
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_states_timestamp ON neurocortex_states(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_states_component ON neurocortex_states(component)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_timestamp ON cognitive_events(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_type ON cognitive_events(event_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_patterns_type ON cognitive_patterns(pattern_type)')
        
        self.sqlite_conn.commit()
    
    def _init_redis(self, redis_url: str):
        """Инициализация Redis"""
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            
            # Настройка Redis для эффективного хранения
            redis_config = {
                'maxmemory': '500mb',
                'maxmemory-policy': 'allkeys-lru',
                'save': '300 10'  # Сохранять каждые 300 секунд если 10+ изменений
            }
            
            logger.info(f"Redis подключен для кэширования состояний: {redis_url}")
            
        except Exception as e:
            logger.error(f"Ошибка подключения к Redis: {e}")
            self.redis_client = None
    
    def _start_background_saver(self):
        """Запуск фоновой задачи сохранения"""
        async def background_saver():
            while True:
                try:
                    save_item = await self.save_queue.get()
                    await self._process_save_item(save_item)
                    self.save_queue.task_done()
                    
                    # Небольшая пауза для предотвращения перегрузки
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    logger.error(f"Ошибка в фоновом сохранении: {e}")
                    self.stats['save_errors'] += 1
                    await asyncio.sleep(1)
        
        self.save_tasks.append(asyncio.create_task(background_saver()))
    
    async def save_state(self, component: str, state_type: str, 
                        state_data: Any, metadata: Dict[str, Any] = None,
                        immediate: bool = False) -> str:
        """Сохранение состояния компонента неокортекса"""
        
        snapshot_id = f"{component}_{state_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
        
        save_item = {
            'snapshot_id': snapshot_id,
            'timestamp': datetime.utcnow(),
            'component': component,
            'state_type': state_type,
            'state_data': state_data,
            'metadata': metadata or {},
            'immediate': immediate
        }
        
        if immediate:
            # Немедленное сохранение
            await self._process_save_item(save_item)
        else:
            # Постановка в очередь для фонового сохранения
            try:
                await self.save_queue.put(save_item)
            except asyncio.QueueFull:
                logger.warning("Очередь сохранения переполнена, выполняем немедленное сохранение")
                await self._process_save_item(save_item)
        
        return snapshot_id
    
    async def _process_save_item(self, save_item: Dict[str, Any]):
        """Обработка элемента сохранения"""
        try:
            # Сериализация данных
            original_data = pickle.dumps(save_item['state_data'])
            original_size = len(original_data)
            
            # Сжатие
            compressor = zstd.ZstdCompressor(level=self.compression_level)
            compressed_data = compressor.compress(original_data)
            compressed_size = len(compressed_data)
            
            # Расчет коэффициента сжатия
            if original_size > 0:
                compression_ratio = original_size / compressed_size
                self.stats['compression_ratio'] = (
                    self.stats['compression_ratio'] * 0.9 + compression_ratio * 0.1
                )
            
            # Checksum для проверки целостности
            checksum = hashlib.sha256(compressed_data).hexdigest()
            
            # Сохранение в SQLite
            cursor = self.sqlite_conn.cursor()
            cursor.execute('''
                INSERT INTO neurocortex_states 
                (timestamp, snapshot_id, component, state_type, state_data, 
                 compressed_size, original_size, checksum, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                save_item['timestamp'].isoformat(),
                save_item['snapshot_id'],
                save_item['component'],
                save_item['state_type'],
                compressed_data,
                compressed_size,
                original_size,
                checksum,
                json.dumps(save_item['metadata'], default=str)
            ))
            
            self.sqlite_conn.commit()
            
            # Кэширование в Redis
            if self.redis_client:
                cache_key = f"neurocortex:state:{save_item['component']}:{save_item['state_type']}:latest"
                await self.redis_client.setex(
                    cache_key,
                    timedelta(hours=24),
                    compressed_data
                )
                
                # Также сохраняем метаданные
                metadata_key = f"neurocortex:metadata:{save_item['snapshot_id']}"
                await self.redis_client.setex(
                    metadata_key,
                    timedelta(hours=24),
                    pickle.dumps({
                        'timestamp': save_item['timestamp'].isoformat(),
                        'component': save_item['component'],
                        'state_type': save_item['state_type'],
                        'original_size': original_size,
                        'compressed_size': compressed_size,
                        'checksum': checksum,
                        'metadata': save_item['metadata']
                    })
                )
            
            self.stats['total_saves'] += 1
            self.stats['last_save'] = datetime.utcnow().isoformat()
            
            logger.debug(f"Сохранено состояние: {save_item['snapshot_id']} "
                        f"({original_size} → {compressed_size} bytes, ratio: {compression_ratio:.2f})")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния {save_item.get('snapshot_id', 'unknown')}: {e}")
            raise
    
    async def load_state(self, component: str, state_type: str, 
                        snapshot_id: str = None) -> Optional[Any]:
        """Загрузка состояния компонента"""
        
        try:
            # Сначала проверяем Redis кэш
            if self.redis_client and not snapshot_id:
                cache_key = f"neurocortex:state:{component}:{state_type}:latest"
                cached_data = await self.redis_client.get(cache_key)
                
                if cached_data:
                    # Распаковка из Redis
                    decompressor = zstd.ZstdDecompressor()
                    decompressed = decompressor.decompress(cached_data)
                    state = pickle.loads(decompressed)
                    
                    logger.debug(f"Загружено из Redis кэша: {component}:{state_type}")
                    return state
            
            # Загрузка из SQLite
            cursor = self.sqlite_conn.cursor()
            
            if snapshot_id:
                cursor.execute('''
                    SELECT state_data, checksum FROM neurocortex_states 
                    WHERE snapshot_id = ? AND component = ? AND state_type = ?
                ''', (snapshot_id, component, state_type))
            else:
                cursor.execute('''
                    SELECT state_data, checksum FROM neurocortex_states 
                    WHERE component = ? AND state_type = ?
                    ORDER BY timestamp DESC LIMIT 1
                ''', (component, state_type))
            
            result = cursor.fetchone()
            
            if result:
                compressed_data, expected_checksum = result
                
                # Проверка целостности
                actual_checksum = hashlib.sha256(compressed_data).hexdigest()
                if actual_checksum != expected_checksum:
                    logger.error(f"Checksum mismatch для состояния {component}:{state_type}")
                    return None
                
                # Распаковка
                decompressor = zstd.ZstdDecompressor()
                decompressed = decompressor.decompress(compressed_data)
                state = pickle.loads(decompressed)
                
                # Обновление Redis кэша
                if self.redis_client:
                    cache_key = f"neurocortex:state:{component}:{state_type}:latest"
                    await self.redis_client.setex(
                        cache_key,
                        timedelta(hours=24),
                        compressed_data
                    )
                
                logger.debug(f"Загружено из SQLite: {component}:{state_type}")
                return state
            
            return None
            
        except Exception as e:
            logger.error(f"Ошибка загрузки состояния {component}:{state_type}: {e}")
            return None
    
    async def log_cognitive_event(self, event_type: str, component: str, 
                                 event_data: Dict[str, Any], 
                                 importance: float = 0.5,
                                 metadata: Dict[str, Any] = None):
        """Логирование когнитивного события"""
        
        event_item = {
            'timestamp': datetime.utcnow(),
            'event_type': event_type,
            'component': component,
            'event_data': event_data,
            'importance': importance,
            'metadata': metadata or {}
        }
        
        try:
            cursor = self.sqlite_conn.cursor()
            cursor.execute('''
                INSERT INTO cognitive_events 
                (timestamp, event_type, component, event_data, importance, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                event_item['timestamp'].isoformat(),
                event_type,
                component,
                json.dumps(event_data, default=str),
                importance,
                json.dumps(metadata or {}, default=str)
            ))
            
            self.sqlite_conn.commit()
            
            # Также отправляем в очередь для обработки
            await self.save_queue.put({
                'type': 'cognitive_event',
                'data': event_item,
                'immediate': True
            })
            
            logger.debug(f"Записано когнитивное событие: {event_type} в {component}")
            
        except Exception as e:
            logger.error(f"Ошибка логирования когнитивного события: {e}")
    
    async def save_cognitive_pattern(self, pattern_id: str, pattern_type: str,
                                    pattern_data: Any, activation_history: List[float] = None,
                                    confidence: float = 0.5, usage_count: int = 0,
                                    last_used: datetime = None,
                                    metadata: Dict[str, Any] = None):
        """Сохранение когнитивного паттерна"""
        
        # Сериализация и сжатие данных паттерна
        original_data = pickle.dumps(pattern_data)
        compressor = zstd.ZstdCompressor(level=self.compression_level)
        compressed_data = compressor.compress(original_data)
        
        cursor = self.sqlite_conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO cognitive_patterns 
            (timestamp, pattern_id, pattern_type, pattern_data, 
             activation_history, confidence, usage_count, last_used, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.utcnow().isoformat(),
            pattern_id,
            pattern_type,
            compressed_data,
            json.dumps(activation_history or []),
            confidence,
            usage_count,
            last_used.isoformat() if last_used else None,
            json.dumps(metadata or {}, default=str)
        ))
        
        self.sqlite_conn.commit()
        
        logger.debug(f"Сохранен когнитивный паттерн: {pattern_id} ({pattern_type})")
    
    async def restore_full_state(self, timestamp: datetime = None) -> Dict[str, Any]:
        """Полное восстановление состояния неокортекса на указанный момент времени"""
        
        restored_state = {}
        
        try:
            cursor = self.sqlite_conn.cursor()
            
            # Определение временной метки
            if not timestamp:
                cursor.execute('SELECT MAX(timestamp) FROM neurocortex_states')
                result = cursor.fetchone()
                if result and result[0]:
                    timestamp = datetime.fromisoformat(result[0])
                else:
                    logger.warning("Нет сохраненных состояний для восстановления")
                    return {}
            
            # Получение всех компонентов на указанный момент
            cursor.execute('''
                SELECT DISTINCT component, state_type 
                FROM neurocortex_states 
                WHERE timestamp <= ?
                ORDER BY timestamp DESC
            ''', (timestamp.isoformat(),))
            
            components = cursor.fetchall()
            
            # Восстановление каждого компонента
            for component, state_type in components:
                cursor.execute('''
                    SELECT state_data, checksum 
                    FROM neurocortex_states 
                    WHERE component = ? AND state_type = ? AND timestamp <= ?
                    ORDER BY timestamp DESC LIMIT 1
                ''', (component, state_type, timestamp.isoformat()))
                
                result = cursor.fetchone()
                if result:
                    compressed_data, expected_checksum = result
                    
                    # Проверка целостности
                    actual_checksum = hashlib.sha256(compressed_data).hexdigest()
                    if actual_checksum != expected_checksum:
                        logger.error(f"Checksum mismatch для {component}:{state_type}")
                        continue
                    
                    # Распаковка
                    decompressor = zstd.ZstdDecompressor()
                    decompressed = decompressor.decompress(compressed_data)
                    state = pickle.loads(decompressed)
                    
                    if component not in restored_state:
                        restored_state[component] = {}
                    restored_state[component][state_type] = state
            
            logger.info(f"Восстановлено состояние неокортекса на {timestamp}: "
                       f"{len(restored_state)} компонентов")
            
            return restored_state
            
        except Exception as e:
            logger.error(f"Ошибка полного восстановления состояния: {e}")
            return {}
    
    async def cleanup_old_states(self, keep_days: int = 7, 
                                keep_snapshots_per_component: int = 100):
        """Очистка старых состояний"""
        
        try:
            cursor = self.sqlite_conn.cursor()
            
            # Удаление старых по времени
            cutoff_date = datetime.utcnow() - timedelta(days=keep_days)
            
            cursor.execute('''
                DELETE FROM neurocortex_states 
                WHERE timestamp < ?
            ''', (cutoff_date.isoformat(),))
            
            time_deleted = cursor.rowcount
            
            # Удаление лишних снапшотов для каждого компонента
            cursor.execute('''
                SELECT DISTINCT component, state_type FROM neurocortex_states
            ''')
            
            components = cursor.fetchall()
            
            type_deleted = 0
            for component, state_type in components:
                cursor.execute('''
                    SELECT id FROM neurocortex_states 
                    WHERE component = ? AND state_type = ?
                    ORDER BY timestamp DESC
                    LIMIT -1 OFFSET ?
                ''', (component, state_type, keep_snapshots_per_component))
                
                ids_to_delete = [row[0] for row in cursor.fetchall()]
                
                if ids_to_delete:
                    placeholders = ','.join('?' * len(ids_to_delete))
                    cursor.execute(f'''
                        DELETE FROM neurocortex_states 
                        WHERE id IN ({placeholders})
                    ''', ids_to_delete)
                    
                    type_deleted += cursor.rowcount
            
            self.sqlite_conn.commit()
            
            logger.info(f"Очистка старых состояний: "
                       f"{time_deleted} по времени, {type_deleted} по количеству")
            
            return {'time_deleted': time_deleted, 'type_deleted': type_deleted}
            
        except Exception as e:
            logger.error(f"Ошибка очистки старых состояний: {e}")
            return {'error': str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики базы данных"""
        
        cursor = self.sqlite_conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM neurocortex_states')
        total_states = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM cognitive_events')
        total_events = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM cognitive_patterns')
        total_patterns = cursor.fetchone()[0]
        
        cursor.execute('SELECT SUM(original_size) FROM neurocortex_states')
        total_size = cursor.fetchone()[0] or 0
        
        cursor.execute('SELECT SUM(compressed_size) FROM neurocortex_states')
        compressed_size = cursor.fetchone()[0] or 0
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'total_states': total_states,
            'total_events': total_events,
            'total_patterns': total_patterns,
            'total_size_mb': total_size / (1024 * 1024),
            'compressed_size_mb': compressed_size / (1024 * 1024),
            'compression_ratio': total_size / compressed_size if compressed_size > 0 else 1.0,
            'save_queue_size': self.save_queue.qsize(),
            'redis_available': self.redis_client is not None,
            'background_saver_tasks': len(self.save_tasks)
        }


# ============================================================================
# ИНТРОСПЕКТИВНОЕ ЛОГИРОВАНИЕ
# ============================================================================

class IntrospectionLogger:
    """Расширенное логирование когнитивных актов с интроспекцией"""
    
    def __init__(self, log_dir: str = "logs/neurocortex", 
                 level: str = "INFO",
                 enable_telemetry: bool = True):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Настройка логирования
        self.logger = logging.getLogger('neurocortex_introspection')
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Обработчики
        self._setup_handlers()
        
        # Телеметрия
        self.telemetry_enabled = enable_telemetry
        self.telemetry_data = {
            'cognitive_acts': defaultdict(int),
            'processing_times': deque(maxlen=1000),
            'error_counts': defaultdict(int),
            'insights_generated': 0,
            'decisions_made': 0
        }
        
        # Контекстные трейсы
        self.context_traces = deque(maxlen=500)
        
        # Анализ паттернов логирования
        self.pattern_analyzer = LogPatternAnalyzer()
        
        logger.info(f"Интроспективное логирование инициализировано: {log_dir}")
    
    def _setup_handlers(self):
        """Настройка обработчиков логирования"""
        
        # Файловый обработчик с ротацией
        file_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "neurocortex.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding='utf-8'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s | %(context)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # JSON обработчик для структурированных логов
        json_handler = logging.FileHandler(
            self.log_dir / "neurocortex_structured.jsonl",
            encoding='utf-8'
        )
        json_handler.setFormatter(JsonFormatter())
        
        # Добавление обработчиков
        self.logger.addHandler(file_handler)
        self.logger.addHandler(json_handler)
        
        # Отключение распространения в корневой логгер
        self.logger.propagate = False
    
    def log_cognitive_act(self, act_type: str, component: str, 
                         data: Dict[str, Any], context: Dict[str, Any] = None,
                         level: str = "INFO"):
        """Логирование когнитивного акта с интроспекцией"""
        
        timestamp = datetime.utcnow
