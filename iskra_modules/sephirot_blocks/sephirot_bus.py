# sephirot_bus.py - БОЖЕСТВЕННЫЙ УРОВЕНЬ СОВЕРШЕНСТВА

import asyncio
import json
import hashlib
import pickle
import secrets
import jwt
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


# ============================================================================
# МОДУЛЬ АУДИТА БЕЗОПАСНОСТИ С БАЗОЙ ДАННЫХ
# ============================================================================

class SecurityAuditDB:
    """База данных аудита безопасности с SQLite"""
    
    def __init__(self, db_path: str = "data/security_audit.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        self.lock = threading.Lock()
        
        self._init_database()
        print(f"[AUDIT] База данных аудита инициализирована: {db_path}")
    
    def _init_database(self):
        """Инициализация структуры базы данных"""
        with self.lock:
            self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.connection.cursor()
            
            # Таблица аутентификаций
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
                    metadata TEXT
                )
            ''')
            
            # Таблица токенов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS token_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    user_id TEXT NOT NULL,
                    token_jti TEXT NOT NULL UNIQUE,
                    action TEXT NOT NULL,
                    expires_at DATETIME,
                    permissions TEXT,
                    metadata TEXT
                )
            ''')
            
            # Таблица сигнатур атак
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS attack_signatures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    signature_type TEXT NOT NULL,
                    source_ip TEXT,
                    pattern TEXT NOT NULL,
                    severity INTEGER NOT NULL,
                    countermeasures TEXT,
                    resolved BOOLEAN DEFAULT 0
                )
            ''')
            
            # Таблица системных событий
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    event_type TEXT NOT NULL,
                    component TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT,
                    metadata TEXT
                )
            ''')
            
            # Индексы для быстрого поиска
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_auth_timestamp ON auth_attempts(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_auth_ip ON auth_attempts(ip_address)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_attacks_timestamp ON attack_signatures(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_events_component ON system_events(component, timestamp)')
            
            self.connection.commit()
    
    def log_auth_attempt(self, ip_address: str, user_id: Optional[str], 
                        token_jti: Optional[str], success: bool, 
                        failure_reason: Optional[str] = None,
                        user_agent: str = "", request_path: str = "",
                        metadata: Dict[str, Any] = None):
        """Логирование попытки аутентификации"""
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO auth_attempts 
                (timestamp, ip_address, user_id, token_jti, success, failure_reason, user_agent, request_path, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.utcnow().isoformat(),
                ip_address,
                user_id,
                token_jti,
                1 if success else 0,
                failure_reason,
                user_agent,
                request_path,
                json.dumps(metadata or {})
            ))
            self.connection.commit()
            
            # Проверка на атаку
            if not success:
                self._check_attack_pattern(ip_address, failure_reason)
    
    def _check_attack_pattern(self, ip_address: str, failure_reason: str):
        """Проверка паттернов атак"""
        # Проверка brute force
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute('''
                SELECT COUNT(*) FROM auth_attempts 
                WHERE ip_address = ? AND success = 0 
                AND timestamp > datetime('now', '-5 minutes')
            ''', (ip_address,))
            
            recent_failures = cursor.fetchone()[0]
            
            if recent_failures > 10:
                self.log_attack_signature(
                    signature_type="brute_force",
                    source_ip=ip_address,
                    pattern=f"{recent_failures} failed attempts in 5 minutes",
                    severity=8,
                    countermeasures="temporary_ip_block"
                )
            
            # Проверка инъекций в токен
            if "Invalid token" in failure_reason and "malformed" not in failure_reason.lower():
                self.log_attack_signature(
                    signature_type="token_injection",
                    source_ip=ip_address,
                    pattern=f"Invalid token attempt: {failure_reason[:100]}",
                    severity=6,
                    countermeasures="token_validation_enhancement"
                )
    
    def log_token_action(self, user_id: str, token_jti: str, action: str,
                        expires_at: Optional[datetime] = None,
                        permissions: List[str] = None,
                        metadata: Dict[str, Any] = None):
        """Логирование действий с токенами"""
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO token_history 
                (timestamp, user_id, token_jti, action, expires_at, permissions, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.utcnow().isoformat(),
                user_id,
                token_jti,
                action,
                expires_at.isoformat() if expires_at else None,
                json.dumps(permissions or []),
                json.dumps(metadata or {})
            ))
            self.connection.commit()
    
    def log_attack_signature(self, signature_type: str, source_ip: str, 
                            pattern: str, severity: int, countermeasures: str):
        """Логирование сигнатуры атаки"""
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO attack_signatures 
                (timestamp, signature_type, source_ip, pattern, severity, countermeasures)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.utcnow().isoformat(),
                signature_type,
                source_ip,
                pattern,
                severity,
                countermeasures
            ))
            self.connection.commit()
            
            # Логирование системного события
            self.log_system_event(
                event_type="security_alert",
                component="auth",
                severity="high" if severity >= 7 else "medium",
                description=f"Attack detected: {signature_type}",
                metadata={
                    "source_ip": source_ip,
                    "pattern": pattern,
                    "countermeasures": countermeasures
                }
            )
    
    def log_system_event(self, event_type: str, component: str, 
                        severity: str, description: str,
                        metadata: Dict[str, Any] = None):
        """Логирование системного события"""
        with self.lock:
            cursor = self.connection.cursor()
            cursor.execute('''
                INSERT INTO system_events 
                (timestamp, event_type, component, severity, description, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.utcnow().isoformat(),
                event_type,
                component,
                severity,
                description,
                json.dumps(metadata or {})
            ))
            self.connection.commit()
    
    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Получение отчета по безопасности"""
        with self.lock:
            cursor = self.connection.cursor()
            
            # Статистика аутентификаций
            cursor.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(success) as successful,
                    COUNT(*) - SUM(success) as failed,
                    COUNT(DISTINCT ip_address) as unique_ips
                FROM auth_attempts 
                WHERE timestamp > datetime('now', ?)
            ''', (f'-{hours} hours',))
            
            auth_stats = cursor.fetchone()
            
            # Активные атаки
            cursor.execute('''
                SELECT COUNT(*) FROM attack_signatures 
                WHERE resolved = 0 AND timestamp > datetime('now', ?)
            ''', (f'-{hours} hours',))
            
            active_attacks = cursor.fetchone()[0]
            
            # Распределение по типам атак
            cursor.execute('''
                SELECT signature_type, COUNT(*) as count 
                FROM attack_signatures 
                WHERE timestamp > datetime('now', ?)
                GROUP BY signature_type 
                ORDER BY count DESC
            ''', (f'-{hours} hours',))
            
            attack_types = cursor.fetchall()
            
            # Топ IP с ошибками
            cursor.execute('''
                SELECT ip_address, COUNT(*) as failures 
                FROM auth_attempts 
                WHERE success = 0 AND timestamp > datetime('now', ?)
                GROUP BY ip_address 
                ORDER BY failures DESC 
                LIMIT 10
            ''', (f'-{hours} hours',))
            
            top_offenders = cursor.fetchall()
            
            # Токены
            cursor.execute('''
                SELECT action, COUNT(*) as count 
                FROM token_history 
                WHERE timestamp > datetime('now', ?)
                GROUP BY action
            ''', (f'-{hours} hours',))
            
            token_actions = cursor.fetchall()
            
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'period_hours': hours,
                'authentication': {
                    'total_attempts': auth_stats[0],
                    'successful': auth_stats[1],
                    'failed': auth_stats[2],
                    'success_rate': auth_stats[1] / auth_stats[0] if auth_stats[0] > 0 else 0,
                    'unique_ips': auth_stats[3]
                },
                'security_threats': {
                    'active_attacks': active_attacks,
                    'attack_types': [{'type': t[0], 'count': t[1]} for t in attack_types],
                    'top_offending_ips': [{'ip': ip[0], 'failures': ip[1]} for ip in top_offenders]
                },
                'token_activity': dict(token_actions),
                'recommendations': self._generate_security_recommendations(auth_stats, active_attacks)
            }
    
    def _generate_security_recommendations(self, auth_stats: Tuple, active_attacks: int) -> List[str]:
        """Генерация рекомендаций по безопасности"""
        recommendations = []
        
        total_attempts, successful, failed, unique_ips = auth_stats
        
        if failed > total_attempts * 0.3:  # >30% ошибок
            recommendations.append("high_failure_rate_detected_increase_monitoring")
        
        if active_attacks > 5:
            recommendations.append("multiple_active_attacks_consider_ip_blacklisting")
        
        if unique_ips > 1000 and total_attempts / unique_ips < 2:
            recommendations.append("suspicious_ip_diversity_possible_scanning")
        
        return recommendations
    
    def export_audit_data(self, start_date: datetime, end_date: datetime, 
                         format: str = "json") -> bytes:
        """Экспорт данных аудита"""
        with self.lock:
            cursor = self.connection.cursor()
            
            # Сбор данных из всех таблиц
            tables = ['auth_attempts', 'token_history', 'attack_signatures', 'system_events']
            data = {}
            
            for table in tables:
                cursor.execute(f'''
                    SELECT * FROM {table} 
                    WHERE timestamp BETWEEN ? AND ?
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                data[table] = {
                    'columns': columns,
                    'rows': [dict(zip(columns, row)) for row in rows]
                }
            
            if format == "json":
                return json.dumps(data, indent=2, default=str).encode('utf-8')
            elif format == "csv":
                # Создание ZIP с CSV файлами
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for table, table_data in data.items():
                        if table_data['rows']:
                            df = pd.DataFrame(table_data['rows'])
                            csv_buffer = io.StringIO()
                            df.to_csv(csv_buffer, index=False)
                            zip_file.writestr(f"{table}.csv", csv_buffer.getvalue())
                
                return buffer.getvalue()
            
            return b""


# ============================================================================
# МОДУЛЬ ШИФРОВАНИЯ FERNET/AES
# ============================================================================

class SecureEncryptionManager:
    """Менеджер шифрования для снапшотов и конфиденциальных данных"""
    
    def __init__(self, master_key: str = None, key_file: str = "data/encryption_key.key"):
        self.key_file = Path(key_file)
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Генерация или загрузка ключа
        self.master_key = self._load_or_generate_key(master_key)
        self.fernet = Fernet(self.master_key)
        
        # Дополнительные ключи для разных типов данных
        self.data_keys = self._derive_data_keys()
        
        print(f"[ENCRYPTION] Менеджер шифрования инициализирован")
    
    def _load_or_generate_key(self, master_key: str = None) -> bytes:
        """Загрузка или генерация мастер-ключа"""
        if master_key:
            # Использование предоставленного ключа
            return base64.urlsafe_b64encode(hashlib.sha256(master_key.encode()).digest())
        
        elif self.key_file.exists():
            # Загрузка существующего ключа
            with open(self.key_file, 'rb') as f:
                return f.read()
        else:
            # Генерация нового ключа
            new_key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(new_key)
            
            # Резервная копия ключа
            backup_key = self.key_file.with_suffix('.key.backup')
            with open(backup_key, 'wb') as f:
                f.write(new_key)
            
            print(f"[ENCRYPTION] Сгенерирован новый ключ шифрования")
            return new_key
    
    def _derive_data_keys(self) -> Dict[str, bytes]:
        """Производные ключи для разных типов данных"""
        data_keys = {}
        
        # Ключ для снапшотов
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"sephirot_snapshots_salt",
            iterations=100000,
        )
        data_keys['snapshots'] = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        
        # Ключ для конфигурации
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"sephirot_config_salt",
            iterations=100000,
        )
        data_keys['config'] = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        
        # Ключ для метрик
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"sephirot_metrics_salt",
            iterations=100000,
        )
        data_keys['metrics'] = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        
        return data_keys
    
    def encrypt_data(self, data: Any, data_type: str = 'snapshots') -> bytes:
        """Шифрование данных"""
        try:
            # Сериализация данных
            if isinstance(data, (dict, list)):
                serialized = pickle.dumps(data)
            elif isinstance(data, bytes):
                serialized = data
            else:
                serialized = pickle.dumps(data)
            
            # Шифрование
            fernet = Fernet(self.data_keys.get(data_type, self.master_key))
            encrypted = fernet.encrypt(serialized)
            
            # Добавление метаданных
            metadata = {
                'data_type': data_type,
                'timestamp': datetime.utcnow().isoformat(),
                'version': '1.0',
                'checksum': hashlib.sha256(serialized).hexdigest()
            }
            
            encrypted_with_metadata = pickle.dumps({
                'metadata': metadata,
                'encrypted_data': encrypted
            })
            
            return encrypted_with_metadata
            
        except Exception as e:
            print(f"[ENCRYPTION] Ошибка шифрования: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: bytes) -> Any:
        """Дешифрование данных"""
        try:
            # Извлечение метаданных
            wrapper = pickle.loads(encrypted_data)
            metadata = wrapper['metadata']
            encrypted = wrapper['encrypted_data']
            
            # Определение ключа
            data_type = metadata.get('data_type', 'snapshots')
            fernet = Fernet(self.data_keys.get(data_type, self.master_key))
            
            # Дешифрование
            decrypted = fernet.decrypt(encrypted)
            
            # Проверка целостности
            expected_checksum = metadata.get('checksum')
            actual_checksum = hashlib.sha256(decrypted).hexdigest()
            
            if expected_checksum and expected_checksum != actual_checksum:
                raise ValueError("Checksum mismatch - data may be corrupted")
            
            # Десериализация
            data = pickle.loads(decrypted)
            
            return data
            
        except Exception as e:
            print(f"[ENCRYPTION] Ошибка дешифрования: {e}")
            raise
    
    def encrypt_file(self, input_path: Path, output_path: Path = None, 
                    data_type: str = 'snapshots'):
        """Шифрование файла"""
        try:
            with open(input_path, 'rb') as f:
                data = f.read()
            
            encrypted = self.encrypt_data(data, data_type)
            
            if output_path is None:
                output_path = input_path.with_suffix(input_path.suffix + '.encrypted')
            
            with open(output_path, 'wb') as f:
                f.write(encrypted)
            
            print(f"[ENCRYPTION] Файл зашифрован: {input_path} → {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[ENCRYPTION] Ошибка шифрования файла: {e}")
            raise
    
    def decrypt_file(self, input_path: Path, output_path: Path = None):
        """Дешифрование файла"""
        try:
            with open(input_path, 'rb') as f:
                encrypted_data = f.read()
            
            decrypted = self.decrypt_data(encrypted_data)
            
            if output_path is None:
                output_path = input_path.with_suffix('').with_suffix('.decrypted')
            
            with open(output_path, 'wb') as f:
                if isinstance(decrypted, bytes):
                    f.write(decrypted)
                else:
                    pickle.dump(decrypted, f)
            
            print(f"[ENCRYPTION] Файл дешифрован: {input_path} → {output_path}")
            return output_path
            
        except Exception as e:
            print(f"[ENCRYPTION] Ошибка дешифрования файла: {e}")
            raise
    
    def rotate_keys(self):
        """Ротация ключей шифрования"""
        try:
            # Генерация нового мастер-ключа
            new_master_key = Fernet.generate_key()
            
            # Шифрование нового ключа старым ключом
            encrypted_new_key = self.fernet.encrypt(new_master_key)
            
            # Сохранение нового ключа
            with open(self.key_file, 'wb') as f:
                f.write(new_master_key)
            
            # Сохранение зашифрованной копии старого ключа
            old_key_backup = self.key_file.with_suffix('.key.rotated')
            with open(old_key_backup, 'wb') as f:
                f.write(encrypted_new_key)
            
            # Обновление производных ключей
            self.master_key = new_master_key
            self.fernet = Fernet(self.master_key)
            self.data_keys = self._derive_data_keys()
            
            print(f"[ENCRYPTION] Ключи успешно ротированы")
            
        except Exception as e:
            print(f"[ENCRYPTION] Ошибка ротации ключей: {e}")
            raise
    
    def get_encryption_status(self) -> Dict[str, Any]:
        """Получение статуса шифрования"""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'key_file_exists': self.key_file.exists(),
            'key_size_bits': len(self.master_key) * 8 if self.master_key else 0,
            'derived_keys': list(self.data_keys.keys()),
            'encryption_algorithm': 'Fernet (AES-128-CBC)',
            'key_rotation_recommended': self._should_rotate_keys()
        }
    
    def _should_rotate_keys(self) -> bool:
        """Проверка необходимости ротации ключей"""
        if not self.key_file.exists():
            return False
        
        key_age_days = (datetime.now() - datetime.fromtimestamp(self.key_file.stat().st_mtime)).days
        return key_age_days > 90  # Ротация каждые 90 дней


# ============================================================================
# МОДУЛЬ АСИНХРОННОГО ВОССТАНОВЛЕНИЯ
# ============================================================================

class AsyncRecoveryManager:
    """Менеджер асинхронного восстановления состояния"""
    
    def __init__(self, bus: 'SephiroticBus'):
        self.bus = bus
        self.recovery_queue = asyncio.Queue(maxsize=10)
        self.recovery_history: deque = deque(maxlen=100)
        self.is_recovering = False
        self.recovery_lock = asyncio.Lock()
        
        print(f"[RECOVERY] Менеджер восстановления инициализирован")
    
    async def schedule_recovery(self, snapshot_id: str, priority: int = 5, 
                               metadata: Dict[str, Any] = None) -> str:
        """Планирование восстановления"""
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
            'error': None
        }
        
        await self.recovery_queue.put((priority, recovery_task))
        
        # Запуск обработки если еще не запущена
        if not self.is_recovering:
            asyncio.create_task(self._process_recovery_queue())
        
        print(f"[RECOVERY] Восстановление запланировано: {recovery_id}")
        return recovery_id
    
    async def _process_recovery_queue(self):
        """Обработка очереди восстановления"""
        self.is_recovering = True
        
        try:
            while not self.recovery_queue.empty():
                try:
                    # Получение задачи с наивысшим приоритетом
                    priority, recovery_task = await asyncio.wait_for(
                        self.recovery_queue.get(), timeout=1.0
                    )
                    
                    recovery_task['scheduled_at'] = datetime.utcnow().isoformat()
                    recovery_task['status'] = 'processing'
                    
                    # Выполнение восстановления
                    success = await self._execute_recovery(recovery_task)
                    
                    if success:
                        recovery_task['status'] = 'completed'
                        recovery_task['completed_at'] = datetime.utcnow().isoformat()
                    else:
                        recovery_task['status'] = 'failed'
                    
                    self.recovery_history.append(recovery_task)
                    self.recovery_queue.task_done()
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    print(f"[RECOVERY] Ошибка обработки очереди: {e}")
                    
        finally:
            self.is_recovering = False
    
    async def _execute_recovery(self, recovery_task: Dict[str, Any]) -> bool:
        """Выполнение восстановления из снапшота"""
        recovery_id = recovery_task['recovery_id']
        snapshot_id = recovery_task['snapshot_id']
        
        print(f"[RECOVERY] Начало восстановления {recovery_id} из снапшота {snapshot_id}")
        
        try:
            async with self.recovery_lock:
                # 1. Приостановка обычных операций шины
                await self.bus._pause_operations()
                
                # 2. Загрузка снапшота
                snapshot_data = await self._load_snapshot(snapshot_id)
                if not snapshot_data:
                    recovery_task['error'] = "Snapshot not found"
                    return False
                
                # 3. Восстановление узлов
                await self._restore_nodes(snapshot_data.get('nodes', {}))
                
                # 4. Восстановление каналов
                await self._restore_channels(snapshot_data.get('channels', {}))
                
                # 5. Восстановление состояния шины
                await self._restore_bus_state(snapshot_data.get('bus_state', {}))
                
                # 6. Восстановление метрик
                await self._restore_metrics(snapshot_id)
                
                # 7. Возобновление операций
                await self.bus._resume_operations()
                
                # 8. Валидация восстановления
                validation_result = await self._validate_recovery()
                
                print(f"[RECOVERY] Восстановление {recovery_id} завершено успешно")
                
                # Логирование события
                if hasattr(self.bus, 'security_audit'):
                    self.bus.security_audit.log_system_event(
                        event_type="system_recovery",
                        component="bus",
                        severity="high",
                        description=f"System recovered from snapshot {snapshot_id}",
                        metadata={
                            'recovery_id': recovery_id,
                            'snapshot_id': snapshot_id,
                            'validation_result': validation_result
                        }
                    )
                
                return True
                
        except Exception as e:
            print(f"[RECOVERY] Ошибка восстановления {recovery_id}: {e}")
            recovery_task['error'] = str(e)
            
            # Аварийное восстановление базового состояния
            try:
                await self._emergency_recovery()
            except Exception as emergency_error:
                print(f"[RECOVERY] Аварийное восстановление также не удалось: {emergency_error}")
            
            return False
    
    async def _load_snapshot(self, snapshot_id: str) -> Optional[Dict[str, Any]]:
        """Загрузка снапшота"""
        try:
            # Используем HistoryAutoSaver если есть
            if hasattr(self.bus, 'history_saver'):
                snapshot_path = self.bus.history_saver.save_dir / "snapshots" / f"{snapshot_id}_full.pkl"
                
                if snapshot_path.exists():
                    with open(snapshot_path, 'rb') as f:
                        encrypted_data = f.read()
                    
                    # Дешифрование если используется шифрование
                    if hasattr(self.bus, 'encryption_manager'):
                        snapshot_data = self.bus.encryption_manager.decrypt_data(encrypted_data)
                    else:
                        snapshot_data = pickle.loads(encrypted_data)
                    
                    return snapshot_data
            
            return None
            
        except Exception as e:
            print(f"[RECOVERY] Ошибка загрузки снапшота: {e}")
            return None
    
    async def _restore_nodes(self, nodes_data: Dict[str, Any]):
        """Восстановление узлов"""
        # Очистка текущих узлов
        self.bus.nodes.clear()
        
        # Восстановление из данных
        for node_name, node_data in nodes_data.items():
            try:
                # Создание узла
                node = SephiroticNode.deserialize(node_data)
                await self.bus.register_node(node)
                
                # Восстановление состояния
                if hasattr(node, 'resonance') and 'resonance' in node_data:
                    node.resonance = node_data['resonance']
                
                if hasattr(node, 'energy') and 'energy' in node_data:
                    node.energy = node_data['energy']
                
                print(f"[RECOVERY] Узел восстановлен: {node_name}")
                
            except Exception as e:
                print(f"[RECOVERY] Ошибка восстановления узла {node_name}: {e}")
    
    async def _restore_channels(self, channels_data: Dict[str, Any]):
        """Восстановление каналов"""
        self.bus.channels.clear()
        
        for channel_id, channel_data in channels_data.items():
            try:
                channel = QuantumChannel(**channel_data)
                self.bus.channels[channel_id] = channel
                
                print(f"[RECOVERY] Канал восстановлен: {channel_id}")
                
            except Exception as e:
                print(f"[RECOVERY] Ошибка восстановления канала {channel_id}: {e}")
    
    async def _restore_bus_state(self, bus_state: Dict[str, Any]):
        """Восстановление состояния шины"""
        # Восстановление очередей
        if 'queues' in bus_state:
            # Очистка очередей
            while not self.bus.signal_queue.empty():
                try:
                    self.bus.signal_queue.get_nowait()
                    self.bus.signal_queue.task_done()
                except:
                    break
            
            while not self.bus.feedback_queue.empty():
                try:
                    self.bus.feedback_queue.get_nowait()
                    self.bus.feedback_queue.task_done()
                except:
                    break
        
        # Восстановление конфигурации
        if 'config' in bus_state:
            self.bus.config = bus_state['config']
    
    async def _restore_metrics(self, snapshot_id: str):
        """Восстановление метрик"""
        if hasattr(self.bus, 'history_saver'):
            metrics_path = self.bus.history_saver.save_dir / "metrics" / f"{snapshot_id}_metrics.json"
            
            if metrics_path.exists():
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    metrics_data = json.load(f)
                
                # Здесь можно восстановить метрики Prometheus
                # или другие метрики системы
    
    async def _validate_recovery(self) -> Dict[str, Any]:
        """Валидация восстановления"""
        validation_results = {
            'timestamp': datetime.utcnow().isoformat(),
            'nodes_restored': len(self.bus.nodes),
            'channels_restored': len(self.bus.channels),
            'system_coherence': await self.bus._calculate_enhanced_coherence(),
            'checks_passed': [],
            'checks_failed': []
        }
        
        # Проверка узлов
        for node_name, node in self.bus.nodes.items():
            if node.status == NodeStatus.ACTIVE:
                validation_results['checks_passed'].append(f"node_{node_name}_active")
            else:
                validation_results['checks_failed'].append(f"node_{node_name}_inactive")
        
        # Проверка каналов
        for channel_id, channel in self.bus.channels.items():
            if channel.is_active:
                validation_results['checks_passed'].append(f"channel_{channel_id}_active")
            else:
                validation_results['checks_failed'].append(f"channel_{channel_id}_inactive")
        
        # Проверка связности
        connectivity_score = await self._check_connectivity()
        validation_results['connectivity_score'] = connectivity_score
        
        if connectivity_score > 0.7:
            validation_results['checks_passed'].append("good_connectivity")
        else:
            validation_results['checks_failed'].append("poor_connectivity")
        
        validation_results['overall_success'] = (
            len(validation_results['checks_failed']) == 0 and 
            validation_results['system_coherence'] > 0.6
        )
        
        return validation_results
    
    async def _check_connectivity(self) -> float:
        """Проверка связности сети"""
        if not self.bus.nodes or not self.bus.channels:
            return 0.0
        
        # Простая проверка: процент узлов имеющих исходящие соединения
        connected_nodes = 0
        
        for node_name in self.bus.nodes:
            outgoing_channels = [
                channel for channel in self.bus.channels.values()
                if channel.from_sephira == node_name
            ]
            
            if outgoing_channels:
                connected_nodes += 1
        
        return connected_nodes / len(self.bus.nodes) if self.bus.nodes else 0.0
    
    async def _emergency_recovery(self):
        """Аварийное восстановление базового состояния"""
        print(f"[RECOVERY] Запуск аварийного восстановления...")
        
        # Создание минимального рабочего состояния
        self.bus.nodes.clear()
        self.bus.channels.clear()
        
        # Создание базовых узлов
        core_nodes = ['Kether', 'Tiferet', 'Yesod']
        
        for node_name in core_nodes:
            try:
                node = SephiroticNode(node_name, 1 if node_name == 'Kether' else 6 if node_name == 'Tiferet' else 9)
                await self.bus.register_node(node)
            except:
                pass
        
        # Создание базовых каналов
        basic_channels = [
            QuantumChannel(
                id="emergency_kether_tiferet",
                hebrew_letter="EMG",
                from_sephira="Kether",
                to_sephira="Tiferet",
                strength=0.5,
                is_active=True
            ),
            QuantumChannel(
                id="emergency_tiferet_yesod",
                hebrew_letter="EMG2",
                from_seph
