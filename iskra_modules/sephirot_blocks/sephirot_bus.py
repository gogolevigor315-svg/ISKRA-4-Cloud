# sephirot_bus.py - АБСОЛЮТНОЕ СОВЕРШЕНСТВО С МИНИМАЛЬНЫМИ ШТРИХАМИ

import asyncio
import json
import hashlib
import pickle
import secrets
import jwt
import hmac
import redis.asyncio as redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
import statistics
import yaml
import numpy as np
from enum import Enum
import aiohttp
from aiohttp import web, WSMsgType
import graphviz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import prometheus_client
from prometheus_client import Gauge, Counter, Histogram, Summary, Info
from tensorflow import keras
from tensorflow.keras import layers
import threading
import os
import zipfile
import io
from pathlib import Path
import sqlite3
import pandas as pd
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import asyncio
import concurrent.futures
import logging
from logging.handlers import RotatingFileHandler


# ============================================================================
# УЛУЧШЕННЫЙ SECURITYAUDITDB С АВТОАРХИВАЦИЕЙ
# ============================================================================

class EnhancedSecurityAuditDB:
    """Улучшенная база данных аудита с автоархивацией и Redis-кэшем"""
    
    def __init__(self, db_path: str = "data/security_audit.db", 
                 redis_url: str = None,
                 archive_after_days: int = 30,
                 max_log_size_mb: int = 100):
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.archive_after_days = archive_after_days
        self.max_log_size_mb = max_log_size_mb
        self.connection = None
        self.lock = threading.Lock()
        self.redis_client = None
        self.archive_queue = asyncio.Queue(maxsize=1000)
        
        # Инициализация Redis если указан URL
        if redis_url:
            self._init_redis(redis_url)
        
        self._init_database()
        self._start_auto_archive_task()
        self._start_size_monitor_task()
        
        print(f"[AUDIT+] База данных аудита инициализирована с автоархивацией")
    
    def _init_redis(self, redis_url: str):
        """Инициализация Redis клиента"""
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            print(f"[AUDIT+] Redis клиент подключен: {redis_url}")
        except Exception as e:
            print(f"[AUDIT+] Ошибка подключения к Redis: {e}")
            self.redis_client = None
    
    def _init_database(self):
        """Расширенная инициализация структуры базы данных"""
        with self.lock:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.connection.cursor()
            
            # Основные таблицы (как раньше)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS auth_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    ip_address TEXT NOT NULL,
                    user_id TEXT,
                    token_jti TEXT,
                    success BOOLEAN NOT NULL,
                    failure_reason TEXT,
                    user_agent TEXT,
                    request_path TEXT,
                    metadata TEXT,
                    archived BOOLEAN DEFAULT 0,
                    archive_id TEXT
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attack_signatures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    signature_type TEXT NOT NULL,
                    source_ip TEXT,
                    pattern TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    countermeasures TEXT,
                    resolved BOOLEAN DEFAULT 0,
                    archived BOOLEAN DEFAULT 0,
                    archive_id TEXT
                )
            ''')
            
            # Таблица архивации
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS archive_log (
                    archive_id TEXT PRIMARY KEY,
                    timestamp DATETIME NOT NULL,
                    table_name TEXT NOT NULL,
                    rows_archived INTEGER NOT NULL,
                    oldest_record DATETIME,
                    newest_record DATETIME,
                    archive_file TEXT,
                    compression_ratio REAL,
                    metadata TEXT
                )
            ''')
            
            # Таблица статистики
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS audit_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    total_attempts INTEGER DEFAULT 0,
                    successful_auths INTEGER DEFAULT 0,
                    failed_auths INTEGER DEFAULT 0,
                    unique_ips INTEGER DEFAULT 0,
                    attacks_detected INTEGER DEFAULT 0,
                    avg_response_time REAL,
                    peak_requests_per_minute INTEGER,
                    UNIQUE(date)
                )
            ''')
            
            # Индексы для производительности
            cursor.execute(''CREATE INDEX IF NOT EXISTS idx_auth_timestamp_archived ON auth_attempts(timestamp, archived)'')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_auth_ip_archived ON auth_attempts(ip_address, archived)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_archive_timestamp ON archive_log(timestamp)')
            
            self.connection.commit()
    
    def _start_auto_archive_task(self):
        """Запуск задачи автоархивации"""
        async def auto_archive_worker():
            while True:
                try:
                    await asyncio.sleep(3600)  # Проверка каждый час
                    await self._check_and_archive_old_data()
                except Exception as e:
                    print(f"[AUDIT+] Ошибка автоархивации: {e}")
                    await asyncio.sleep(300)  # Пауза при ошибке
        
        asyncio.create_task(auto_archive_worker())
    
    def _start_size_monitor_task(self):
        """Запуск мониторинга размера БД"""
        async def size_monitor():
            while True:
                try:
                    await asyncio.sleep(1800)  # Проверка каждые 30 минут
                    db_size_mb = self.db_path.stat().st_size / (1024 * 1024)
                    
                    if db_size_mb > self.max_log_size_mb:
                        print(f"[AUDIT+] Размер БД превышен: {db_size_mb:.1f}MB > {self.max_log_size_mb}MB")
                        await self._emergency_archive()
                        
                except Exception as e:
                    print(f"[AUDIT+] Ошибка мониторинга размера: {e}")
                    await asyncio.sleep(300)
        
        asyncio.create_task(size_monitor())
    
    async def _check_and_archive_old_data(self):
        """Проверка и архивация старых данных"""
        cutoff_date = datetime.utcnow() - timedelta(days=self.archive_after_days)
        
        tables_to_archive = ['auth_attempts', 'attack_signatures']
        
        for table_name in tables_to_archive:
            with self.lock:
                cursor = self.connection.cursor()
                
                # Проверка количества старых записей
                cursor.execute(f'''
                    SELECT COUNT(*) FROM {table_name} 
                    WHERE timestamp < ? AND archived = 0
                ''', (cutoff_date.isoformat(),))
                
                old_records_count = cursor.fetchone()[0]
                
                if old_records_count > 1000:  # Архивируем если больше 1000 записей
                    await self._archive_table_data(table_name, cutoff_date)
    
    async def _archive_table_data(self, table_name: str, cutoff_date: datetime):
        """Архивация данных таблицы"""
        try:
            archive_id = f"archive_{table_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            archive_file = self.db_path.parent / "archives" / f"{archive_id}.zip"
            archive_file.parent.mkdir(exist_ok=True)
            
            print(f"[AUDIT+] Начало архивации {table_name} ({cutoff_date.date()})")
            
            with self.lock:
                cursor = self.connection.cursor()
                
                # Получение данных для архивации
                cursor.execute(f'''
                    SELECT * FROM {table_name} 
                    WHERE timestamp < ? AND archived = 0
                    ORDER BY timestamp
                ''', (cutoff_date.isoformat(),))
                
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                if not rows:
                    return
                
                # Сохранение в ZIP
                with zipfile.ZipFile(archive_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # Сохранение как CSV
                    csv_buffer = io.StringIO()
                    df = pd.DataFrame(rows, columns=columns)
                    df.to_csv(csv_buffer, index=False)
                    
                    zipf.writestr(f"{table_name}.csv", csv_buffer.getvalue())
                    
                    # Сохранение метаданных
                    metadata = {
                        'archive_id': archive_id,
                        'table_name': table_name,
                        'cutoff_date': cutoff_date.isoformat(),
                        'rows_archived': len(rows),
                        'columns': columns,
                        'compression_method': 'DEFLATE',
                        'created_at': datetime.utcnow().isoformat()
                    }
                    
                    zipf.writestr('metadata.json', json.dumps(metadata, indent=2))
                
                # Обновление статуса в БД
                cursor.execute(f'''
                    UPDATE {table_name} 
                    SET archived = 1, archive_id = ?
                    WHERE timestamp < ? AND archived = 0
                ''', (archive_id, cutoff_date.isoformat()))
                
                # Запись в лог архивации
                cursor.execute('''
                    INSERT INTO archive_log 
                    (archive_id, timestamp, table_name, rows_archived, 
                     oldest_record, newest_record, archive_file, compression_ratio, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    archive_id,
                    datetime.utcnow().isoformat(),
                    table_name,
                    len(rows),
                    rows[0][1] if rows else None,  # timestamp первой записи
                    rows[-1][1] if rows else None,  # timestamp последней записи
                    str(archive_file),
                    os.path.getsize(archive_file) / (len(csv_buffer.getvalue()) + 1),
                    json.dumps(metadata)
                ))
                
                self.connection.commit()
                
                # Кэширование статистики в Redis
                if self.redis_client:
                    await self._cache_archive_stats(archive_id, table_name, len(rows))
                
                print(f"[AUDIT+] Архивация завершена: {table_name} -> {len(rows)} записей")
                
        except Exception as e:
            print(f"[AUDIT+] Ошибка архивации {table_name}: {e}")
    
    async def _emergency_archive(self):
        """Аварийная архивация при превышении размера"""
        print(f"[AUDIT+] Запуск аварийной архивации...")
        
        # Архивируем самые старые данные до достижения целевого размера
        target_size_mb = self.max_log_size_mb * 0.7  # Цель - 70% от максимального
        
        while self.db_path.stat().st_size / (1024 * 1024) > target_size_mb:
            await self._archive_oldest_data()
            await asyncio.sleep(1)
    
    async def _archive_oldest_data(self):
        """Архивация самых старых данных"""
        with self.lock:
            cursor = self.connection.cursor()
            
            # Находим самую старую дату с непроархивированными данными
            cursor.execute('''
                SELECT MIN(timestamp) FROM auth_attempts 
                WHERE archived = 0
                UNION
                SELECT MIN(timestamp) FROM attack_signatures 
                WHERE archived = 0
            ''')
            
            result = cursor.fetchone()
            if not result or not result[0]:
                return
            
            oldest_date = datetime.fromisoformat(result[0])
            cutoff_date = oldest_date + timedelta(days=7)  # Архивируем неделю данных
            
            await self._archive_table_data('auth_attempts', cutoff_date)
            await self._archive_table_data('attack_signatures', cutoff_date)
    
    async def _cache_archive_stats(self, archive_id: str, table_name: str, rows_count: int):
        """Кэширование статистики архивации в Redis"""
        try:
            if not self.redis_client:
                return
            
            # Сохранение базовой статистики
            await self.redis_client.hset(
                f"audit:archive:{archive_id}",
                mapping={
                    'table': table_name,
                    'rows': str(rows_count),
                    'timestamp': datetime.utcnow().isoformat()
                }
            )
            
            # Обновление общей статистики
            total_archived = await self.redis_client.incrby(
                f"audit:stats:{table_name}:total_archived",
                rows_count
            )
            
            # Установка TTL (30 дней)
            await self.redis_client.expire(f"audit:archive:{archive_id}", 2592000)
            
        except Exception as e:
            print(f"[AUDIT+] Ошибка кэширования в Redis: {e}")
    
    def log_auth_attempt(self, ip_address: str, user_id: Optional[str], 
                        token_jti: Optional[str], success: bool, 
                        failure_reason: Optional[str] = None,
                        user_agent: str = "", request_path: str = "",
                        metadata: Dict[str, Any] = None):
        """Улучшенное логирование с кэшированием в Redis"""
        super().log_auth_attempt(ip_address, user_id, token_jti, success, 
                               failure_reason, user_agent, request_path, metadata)
        
        # Кэширование в Redis для быстрого доступа
        if self.redis_client:
            asyncio.create_task(self._cache_auth_attempt(
                ip_address, success, failure_reason
            ))
    
    async def _cache_auth_attempt(self, ip_address: str, success: bool, failure_reason: str = None):
        """Кэширование попытки аутентификации в Redis"""
        try:
            # Счетчик попыток по IP
            key = f"audit:ip:{ip_address}:attempts"
            await self.redis_client.incr(key)
            await self.redis_client.expire(key, 3600)  # TTL 1 час
            
            if not success:
                # Счетчик неудачных попыток
                fail_key = f"audit:ip:{ip_address}:fails"
                await self.redis_client.incr(fail_key)
                await self.redis_client.expire(fail_key, 3600)
            
            # Обновление глобальной статистики
            await self.redis_client.incr("audit:global:total_attempts")
            if success:
                await self.redis_client.incr("audit:global:successful")
            else:
                await self.redis_client.incr("audit:global:failed")
            
        except Exception as e:
            print(f"[AUDIT+] Ошибка кэширования в Redis: {e}")
    
    async def get_realtime_stats(self) -> Dict[str, Any]:
        """Получение статистики в реальном времени из Redis"""
        stats = {
            'timestamp': datetime.utcnow().isoformat(),
            'redis_available': self.redis_client is not None,
            'realtime_counts': {},
            'top_offenders': []
        }
        
        if not self.redis_client:
            return stats
        
        try:
            # Получение глобальных счетчиков
            counters = ['total_attempts', 'successful', 'failed']
            for counter in counters:
                value = await self.redis_client.get(f"audit:global:{counter}")
                stats['realtime_counts'][counter] = int(value) if value else 0
            
            # Поиск IP с наибольшим количеством неудачных попыток
            # (требует сканирования ключей, лучше использовать отсортированный набор)
            ip_pattern = "audit:ip:*:fails"
            cursor = '0'
            offender_scores = []
            
            while True:
                cursor, keys = await self.redis_client.scan(cursor, match=ip_pattern, count=100)
                
                for key in keys:
                    ip = key.split(':')[2]  # Извлекаем IP из ключа
                    fail_count = await self.redis_client.get(key)
                    if fail_count:
                        offender_scores.append((ip, int(fail_count)))
                
                if cursor == '0':
                    break
            
            # Сортировка по количеству неудач
            offender_scores.sort(key=lambda x: x[1], reverse=True)
            stats['top_offenders'] = offender_scores[:10]
            
        except Exception as e:
            print(f"[AUDIT+] Ошибка получения статистики Redis: {e}")
        
        return stats


# ============================================================================
# УЛУЧШЕННЫЙ ENCRYPTIONMANAGER С HMAC-КОНТРОЛЕМ
# ============================================================================

class HMACEncryptionManager:
    """Менеджер шифрования с HMAC контролем целостности ключей"""
    
    def __init__(self, master_key: str = None, 
                 key_file: str = "data/encryption_key.key",
                 hmac_key_file: str = "data/hmac_key.key"):
        
        self.key_file = Path(key_file)
        self.hmac_key_file = Path(hmac_key_file)
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Генерация или загрузка ключей
        self.master_key = self._load_or_generate_master_key(master_key)
        self.hmac_key = self._load_or_generate_hmac_key()
        
        # Инициализация Fernet и HMAC
        self.fernet = Fernet(self.master_key)
        
        # Регистрация ключей с HMAC защитой
        self._register_key_with_hmac()
        
        # Дополнительные ключи
        self.data_keys = self._derive_data_keys()
        
        print(f"[ENCRYPTION+] Менеджер шифрования с HMAC инициализирован")
    
    def _load_or_generate_master_key(self, master_key: str = None) -> bytes:
        """Загрузка или генерация мастер-ключа с HMAC защитой"""
        if master_key:
            # Использование предоставленного ключа
            key = base64.urlsafe_b64encode(hashlib.sha256(master_key.encode()).digest())
            self._verify_key_integrity(key)
            return key
        
        elif self.key_file.exists():
            # Загрузка существующего ключа с проверкой целостности
            with open(self.key_file, 'rb') as f:
                key = f.read()
            
            if not self._verify_key_integrity(key):
                print("[ENCRYPTION+] Обнаружена проблема целостности ключа!")
                return self._generate_new_key_pair()
            
            return key
        
        else:
            # Генерация новой пары ключей
            return self._generate_new_key_pair()
    
    def _load_or_generate_hmac_key(self) -> bytes:
        """Загрузка или генерация HMAC ключа"""
        if self.hmac_key_file.exists():
            with open(self.hmac_key_file, 'rb') as f:
                return f.read()
        else:
            # Генерация нового HMAC ключа
            hmac_key = secrets.token_bytes(32)
            with open(self.hmac_key_file, 'wb') as f:
                f.write(hmac_key)
            
            # Установка строгих прав доступа
            os.chmod(self.hmac_key_file, 0o600)
            
            return hmac_key
    
    def _generate_new_key_pair(self) -> bytes:
        """Генерация новой пары ключей (мастер + HMAC)"""
        # Генерация мастер-ключа
        master_key = Fernet.generate_key()
        
        # Генерация HMAC ключа если его нет
        if not self.hmac_key_file.exists():
            hmac_key = secrets.token_bytes(32)
            with open(self.hmac_key_file, 'wb') as f:
                f.write(hmac_key)
            os.chmod(self.hmac_key_file, 0o600)
            self.hmac_key = hmac_key
        
        # Сохранение мастер-ключа
        with open(self.key_file, 'wb') as f:
            f.write(master_key)
        os.chmod(self.key_file, 0o600)
        
        # Регистрация с HMAC
        self._register_key_with_hmac()
        
        print(f"[ENCRYPTION+] Сгенерирована новая пара ключей")
        return master_key
    
    def _register_key_with_hmac(self):
        """Регистрация ключа с HMAC защитой"""
        # Создание HMAC подписи для ключа
        key_signature = hmac.new(
            self.hmac_key,
            self.master_key,
            hashlib.sha256
        ).digest()
        
        # Сохранение подписи
        signature_file = self.key_file.with_suffix('.key.sig')
        with open(signature_file, 'wb') as f:
            f.write(key_signature)
        
        os.chmod(signature_file, 0o600)
    
    def _verify_key_integrity(self, key: bytes) -> bool:
        """Проверка целостности ключа с помощью HMAC"""
        signature_file = self.key_file.with_suffix('.key.sig')
        
        if not signature_file.exists():
            print(f"[ENCRYPTION+] Файл подписи не найден: {signature_file}")
            return False
        
        try:
            # Загрузка сохраненной подписи
            with open(signature_file, 'rb') as f:
                stored_signature = f.read()
            
            # Вычисление текущей подписи
            current_signature = hmac.new(
                self.hmac_key,
                key,
                hashlib.sha256
            ).digest()
            
            # Сравнение с постоянным временем
            return hmac.compare_digest(stored_signature, current_signature)
            
        except Exception as e:
            print(f"[ENCRYPTION+] Ошибка проверки целостности: {e}")
            return False
    
    def _derive_data_keys(self) -> Dict[str, Tuple[bytes, bytes]]:
        """Производные ключи с отдельными HMAC для каждого типа данных"""
        data_keys = {}
        
        # Для каждого типа данных создаем пару: ключ шифрования + HMAC ключ
        data_types = ['snapshots', 'config', 'metrics', 'logs']
        
        for data_type in data_types:
            # Производный ключ шифрования
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=f"sephirot_{data_type}_salt".encode(),
                iterations=100000,
            )
            encryption_key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
            
            # Производный HMAC ключ
            hmac_kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=f"sephirot_{data_type}_hmac_salt".encode(),
                iterations=100000,
            )
            hmac_key = hmac_kdf.derive(self.hmac_key)
            
            data_keys[data_type] = (encryption_key, hmac_key)
        
        return data_keys
    
    def encrypt_data(self, data: Any, data_type: str = 'snapshots') -> bytes:
        """Шифрование данных с HMAC аутентификацией"""
        try:
            # Получение ключей для типа данных
            if data_type not in self.data_keys:
                raise ValueError(f"Unknown data type: {data_type}")
            
            encryption_key, hmac_key = self.data_keys[data_type]
            
            # Сериализация данных
            serialized = pickle.dumps(data)
            
            # Создание HMAC подписи данных перед шифрованием
            data_hmac = hmac.new(
                hmac_key,
                serialized,
                hashlib.sha256
            ).digest()
            
            # Шифрование данных
            fernet = Fernet(encryption_key)
            encrypted = fernet.encrypt(serialized)
            
            # Создание HMAC подписи зашифрованных данных
            encrypted_hmac = hmac.new(
                hmac_key,
                encrypted,
                hashlib.sha256
            ).digest()
            
            # Упаковка с метаданными
            metadata = {
                'data_type': data_type,
                'timestamp': datetime.utcnow().isoformat(),
                'version': '2.0',
                'data_hmac': base64.b64encode(data_hmac).decode(),
                'encrypted_hmac': base64.b64encode(encrypted_hmac).decode(),
                'integrity_check': 'hmac_double_layer'
            }
            
            encrypted_package = {
                'metadata': metadata,
                'encrypted_data': encrypted,
                'data_hmac': data_hmac,
                'encrypted_hmac': encrypted_hmac
            }
            
            return pickle.dumps(encrypted_package)
            
        except Exception as e:
            print(f"[ENCRYPTION+] Ошибка шифрования: {e}")
            raise
    
    def decrypt_data(self, encrypted_package_bytes: bytes) -> Any:
        """Дешифрование данных с двойной HMAC проверкой"""
        try:
            # Распаковка пакета
            encrypted_package = pickle.loads(encrypted_package_bytes)
            metadata = encrypted_package['metadata']
            encrypted = encrypted_package['encrypted_data']
            stored_data_hmac = encrypted_package['data_hmac']
            stored_encrypted_hmac = encrypted_package['encrypted_hmac']
            
            data_type = metadata.get('data_type', 'snapshots')
            
            if data_type not in self.data_keys:
                raise ValueError(f"Unknown data type: {data_type}")
            
            encryption_key, hmac_key = self.data_keys[data_type]
            
            # Первая проверка: HMAC зашифрованных данных
            calculated_encrypted_hmac = hmac.new(
                hmac_key,
                encrypted,
                hashlib.sha256
            ).digest()
            
            if not hmac.compare_digest(stored_encrypted_hmac, calculated_encrypted_hmac):
                raise SecurityError("HMAC verification failed for encrypted data")
            
            # Дешифрование
            fernet = Fernet(encryption_key)
            decrypted = fernet.decrypt(encrypted)
            
            # Вторая проверка: HMAC исходных данных
            calculated_data_hmac = hmac.new(
                hmac_key,
                decrypted,
                hashlib.sha256
            ).digest()
            
            if not hmac.compare_digest(stored_data_hmac, calculated_data_hmac):
                raise SecurityError("HMAC verification failed for decrypted data")
            
            # Десериализация
            data = pickle.loads(decrypted)
            
            return data
            
        except SecurityError as e:
            print(f"[ENCRYPTION+] Ошибка безопасности при дешифровании: {e}")
            raise
        except Exception as e:
            print(f"[ENCRYPTION+] Ошибка дешифрования: {e}")
            raise
    
    def rotate_keys_with_verification(self):
        """Ротация ключей с верификацией"""
        try:
            print(f"[ENCRYPTION+] Начало ротации ключей с верификацией...")
            
            # 1. Верификация текущих ключей
            if not self._verify_key_integrity(self.master_key):
                raise SecurityError("Current key integrity check failed")
            
            # 2. Генерация новых ключей
            new_master_key = Fernet.generate_key()
            new_hmac_key = secrets.token_bytes(32)
            
            # 3. Шифрование новых ключей старыми
            fernet_old = Fernet(self.master_key)
            encrypted_new_master = fernet_old.encrypt(new_master_key)
            encrypted_new_hmac = fernet_old.encrypt(new_hmac_key)
            
            # 4. Сохранение зашифрованных новых ключей
            backup_file = self.key_file.with_suffix('.key.rotation_backup')
            with open(backup_file, 'wb') as f:
                pickle.dump({
                    'encrypted_new_master': encrypted_new_master,
                    'encrypted_new_hmac': encrypted_new_hmac,
                    'rotation_timestamp': datetime.utcnow().isoformat(),
                    'old_key_fingerprint': hashlib.sha256(self.master_key).hexdigest()[:16]
                }, f)
            
            # 5. Активация новых ключей
            self.master_key = new_master_key
            self.hmac_key = new_hmac_key
            self.fernet = Fernet(self.master_key)
            
            # 6. Сохранение новых ключей
            with open(self.key_file, 'wb') as f:
                f.write(self.master_key)
            
            with open(self.hmac_key_file, 'wb') as f:
                f.write(self.hmac_key)
            
            # 7. Регистрация новых ключей
            self._register_key_with_hmac()
            
            # 8. Обновление производных ключей
            self.data_keys = self._derive_data_keys()
            
            print(f"[ENCRYPTION+] Ротация ключей завершена успешно")
            
            # Логирование события
            self._log_key_rotation()
            
        except Exception as e:
            print(f"[ENCRYPTION+] Ошибка ротации ключей: {e}")
            raise
    
    def _log_key_rotation(self):
        """Логирование ротации ключей"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'event': 'key_rotation',
            'old_key_fingerprint': hashlib.sha256(self.master_key).hexdigest()[:16],
            'new_key_fingerprint': hashlib.sha256(self.master_key).hexdigest()[:16],
            'rotation_id': secrets.token_hex(8)
        }
        
        log_file = self.key_file.parent / "key_rotation_log.json"
        
        try:
            if log_file.exists():
                with open(log_file, 'r') as f:
                    logs = json.load(f)
            else:
                logs = []
            
            logs.append(log_entry)
            
            with open(log_file, 'w') as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            print(f"[ENCRYPTION+] Ошибка логирования ротации: {e}")
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Расширенный статус шифрования"""
        status = super().get_encryption_status()
        
        status.update({
            'hmac_protection': True,
            'double_layer_hmac': True,
            'key_integrity_verified': self._verify_key_integrity(self.master_key),
            'hmac_key_size': len(self.hmac_key) * 8,
            'derived_keys_count': len(self.data_keys),
            'key_rotation_log_count': self._get_rotation_log_count(),
            'security_level': 'military_grade'
        })
        
        return status
    
    def _get_rotation_log_count(self) -> int:
        """Получение количества записей в логе ротации"""
        log_file = self.key_file.parent / "key_rotation_log.json"
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    logs = json.load(f)
                return len(logs)
            except:
                return 0
        return 0


class SecurityError(Exception):
    """Исключение безопасности"""
    pass


# ============================================================================
# УЛУЧШЕННЫЙ ASYNCRECOVERYMANAGER С REDIS ОЧЕРЕДЬЮ
# ============================================================================

class RedisEnhancedRecoveryManager:
    """Менеджер восстановления с Redis очередью и расширенной валидацией"""
    
    def __init__(self, bus: 'SephiroticBus', redis_url: str = None):
        self.bus = bus
        self.redis_client = None
        self.recovery_lock = asyncio.Lock()
        self.validation_lock = asyncio.Lock()
        self.node_state_comparator = NodeStateComparator()
        
        # Инициализация Redis
        if redis_url:
            self._init_redis(redis_url)
        else:
            # Локальная очередь как fallback
            self.local_queue = asyncio.PriorityQueue(maxsize=100)
        
        print(f"[RECOVERY+] Менеджер восстановления с Redis инициализирован")
    
    def _init_redis(self, redis_url: str):
        """Инициализация Redis клиента"""
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=False)
            print(f"[RECOVERY+] Redis клиент подключен для очереди восстановления")
            
            # Создание отказоустойчивых очередей
            self.recovery_queue_key = "sephirot:recovery:queue"
            self.recovery_history_key = "sephirot:recovery:history"
            self.recovery_backup_key = "sephirot:recovery:backup"
            
        except Exception as e:
            print(f"[RECOVERY+] Ошибка подключения к Redis: {e}")
            self.redis_client = None
            self.local_queue = asyncio.PriorityQueue(maxsize=100)
    
    async def schedule_recovery(self, snapshot_id: str, priority: int = 5,
                               metadata: Dict[str, Any] = None) -> str:
        """Планирование восстановления с записью в Redis"""
        recovery_id = hashlib.md5(
            f"{snapshot_id}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:16]
        
        recovery_task = {
            'recovery_id': recovery_id,
            'snapshot_id': snapshot_id,
            'priority': priority,
            'metadata': metadata or {},
            'status': 'pending',
            'created_at': datetime.utcnow().isoformat(),
            'scheduled_at': None,
            'completed_at': None,
            'error': None,
            'validation_results': None
        }
        
        # Запись в Redis если доступен
        if self.redis_client:
            try:
                # Сохранение задачи
                task_key = f"sephirot:recovery:tasks:{recovery_id}"
                await self.redis_client.set(
                    task_key,
                    pickle.dumps(recovery_task),
                    ex=86400  # TTL 24 часа
                )
                
                # Добавление в приоритетную очередь Redis
                queue_item = {
                    'recovery_id': recovery_id,
                    'priority': priority,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
               
