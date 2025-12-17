# sephirot_bus.py - АБСОЛЮТНО СОВЕРШЕННАЯ СЕФИРОТИЧЕСКАЯ ШИНА
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
import prometheus_client
from prometheus_client import Gauge, Counter, Histogram, Summary, Info
from tensorflow import keras
from tensorflow.keras import layers
import threading
import os
import zipfile
import io
from pathlib import Path

from .sephirot_base import (
    SephiroticNode, QuantumLink, SignalPackage, 
    SignalType, NodeStatus, ResonancePhase, NodeMetrics
)


# ============================================================================
# МОДУЛЬ БЕЗОПАСНОСТИ JWT
# ============================================================================

class JWTSecurityManager:
    """Менеджер безопасности с JWT аутентификацией"""
    
    def __init__(self, secret_key: str = None, token_expiry_hours: int = 24):
        self.secret_key = secret_key or secrets.token_hex(32)
        self.token_expiry_hours = token_expiry_hours
        self.revoked_tokens: Set[str] = set()
        self.token_history: deque = deque(maxlen=1000)
        self.access_log: deque = deque(maxlen=5000)
        
        print(f"[SECURITY] JWT менеджер инициализирован. Токены действительны {token_expiry_hours}ч")
    
    def create_token(self, user_id: str, permissions: List[str], 
                    metadata: Dict[str, Any] = None) -> str:
        """Создание JWT токена"""
        payload = {
            'user_id': user_id,
            'permissions': permissions,
            'exp': datetime.utcnow() + timedelta(hours=self.token_expiry_hours),
            'iat': datetime.utcnow(),
            'jti': secrets.token_hex(16),  # Уникальный ID токена
            'metadata': metadata or {}
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        # Логирование создания токена
        self.token_history.append({
            'timestamp': datetime.utcnow().isoformat(),
            'user_id': user_id,
            'token_jti': payload['jti'],
            'permissions': permissions,
            'action': 'token_created'
        })
        
        return token
    
    def validate_token(self, token: str, required_permissions: List[str] = None) -> Dict[str, Any]:
        """Валидация JWT токена"""
        try:
            # Проверка отозванных токенов
            if token in self.revoked_tokens:
                raise jwt.InvalidTokenError("Token revoked")
            
            # Декодирование токена
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Проверка срока действия
            if datetime.utcnow() > datetime.fromtimestamp(payload['exp']):
                raise jwt.ExpiredSignatureError("Token expired")
            
            # Проверка разрешений
            if required_permissions:
                user_permissions = set(payload.get('permissions', []))
                required_set = set(required_permissions)
                
                if not required_set.issubset(user_permissions):
                    raise jwt.InvalidTokenError("Insufficient permissions")
            
            # Логирование успешного доступа
            self.access_log.append({
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': payload['user_id'],
                'token_jti': payload['jti'],
                'permissions': payload['permissions'],
                'action': 'access_granted',
                'ip_address': '127.0.0.1'  # Будет заполнено из контекста запроса
            })
            
            return {
                'valid': True,
                'payload': payload,
                'user_id': payload['user_id'],
                'permissions': payload['permissions'],
                'metadata': payload.get('metadata', {})
            }
            
        except jwt.ExpiredSignatureError as e:
            self.access_log.append({
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'action': 'access_denied_expired'
            })
            return {'valid': False, 'error': 'Token expired'}
            
        except jwt.InvalidTokenError as e:
            self.access_log.append({
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'action': 'access_denied_invalid'
            })
            return {'valid': False, 'error': str(e)}
            
        except Exception as e:
            self.access_log.append({
                'timestamp': datetime.utcnow().isoformat(),
                'error': str(e),
                'action': 'access_denied_error'
            })
            return {'valid': False, 'error': 'Invalid token'}
    
    def revoke_token(self, token: str) -> bool:
        """Отзыв токена"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            self.revoked_tokens.add(token)
            
            self.token_history.append({
                'timestamp': datetime.utcnow().isoformat(),
                'user_id': payload.get('user_id'),
                'token_jti': payload.get('jti'),
                'action': 'token_revoked'
            })
            
            return True
        except:
            return False
    
    def create_auth_middleware(self, required_permissions: List[str] = None):
        """Создание middleware для аутентификации"""
        @web.middleware
        async def auth_middleware(request: web.Request, handler: Callable):
            # Исключение для публичных эндпоинтов
            public_paths = ['/metrics', '/health', '/login', '/docs']
            if any(request.path.startswith(path) for path in public_paths):
                return await handler(request)
            
            # Проверка заголовка Authorization
            auth_header = request.headers.get('Authorization', '')
            
            if not auth_header.startswith('Bearer '):
                return web.json_response(
                    {'error': 'Missing or invalid Authorization header'},
                    status=401
                )
            
            token = auth_header[7:]  # Убираем "Bearer "
            validation_result = self.validate_token(token, required_permissions)
            
            if not validation_result['valid']:
                return web.json_response(
                    {'error': validation_result.get('error', 'Authentication failed')},
                    status=403
                )
            
            # Добавление информации о пользователе в запрос
            request['user'] = validation_result
            
            # Логирование IP
            peer = request.transport.get_extra_info('peername')
            if peer:
                ip_address = peer[0]
                if self.access_log:
                    self.access_log[-1]['ip_address'] = ip_address
            
            return await handler(request)
        
        return auth_middleware
    
    def get_security_report(self) -> Dict[str, Any]:
        """Отчет о безопасности"""
        now = datetime.utcnow()
        
        # Статистика по доступам
        recent_accesses = list(self.access_log)[-100:] if self.access_log else []
        
        access_stats = {
            'total_attempts': len(self.access_log),
            'successful_accesses': len([a for a in recent_accesses if a.get('action') == 'access_granted']),
            'failed_accesses': len([a for a in recent_accesses if a.get('action', '').startswith('access_denied')]),
            'common_errors': defaultdict(int)
        }
        
        for access in recent_accesses:
            if 'error' in access:
                access_stats['common_errors'][access['error']] += 1
        
        return {
            'timestamp': now.isoformat(),
            'tokens_issued': len(self.token_history),
            'tokens_revoked': len(self.revoked_tokens),
            'active_sessions': len(self.token_history) - len(self.revoked_tokens),
            'access_statistics': access_stats,
            'security_level': 'high',
            'recommendations': self._generate_security_recommendations()
        }
    
    def _generate_security_recommendations(self) -> List[str]:
        """Генерация рекомендаций по безопасности"""
        recommendations = []
        
        if len(self.access_log) > 1000:
            # Частый доступ - проверка на брутфорс
            recent_failures = len([
                a for a in list(self.access_log)[-100:]
                if a.get('action', '').startswith('access_denied')
            ])
            
            if recent_failures > 20:
                recommendations.append("high_failure_rate_detected")
        
        if len(self.revoked_tokens) > 50:
            recommendations.append("consider_token_rotation")
        
        if self.token_expiry_hours > 24:
            recommendations.append("reduce_token_lifetime_for_security")
        
        return recommendations


# ============================================================================
# МОДУЛЬ АВТОСОХРАНЕНИЯ ИСТОРИИ
# ============================================================================

class HistoryAutoSaver:
    """Автоматическое сохранение истории и состояния системы"""
    
    def __init__(self, save_dir: str = "data/bus_history", 
                 save_interval_minutes: int = 5,
                 max_snapshots: int = 100):
        
        self.save_dir = Path(save_dir)
        self.save_interval_minutes = save_interval_minutes
        self.max_snapshots = max_snapshots
        self.last_save: Optional[datetime] = None
        self.snapshot_counter = 0
        self.compression_enabled = True
        
        # Создание директории если не существует
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Создание поддиректорий
        (self.save_dir / "metrics").mkdir(exist_ok=True)
        (self.save_dir / "models").mkdir(exist_ok=True)
        (self.save_dir / "snapshots").mkdir(exist_ok=True)
        (self.save_dir / "logs").mkdir(exist_ok=True)
        
        print(f"[HISTORY] Автосохранение инициализировано. Интервал: {save_interval_minutes} минут")
    
    async def save_system_state(self, bus: 'SephiroticBus', 
                               force: bool = False) -> bool:
        """Сохранение состояния системы"""
        
        now = datetime.utcnow()
        
        # Проверка необходимости сохранения
        if (not force and self.last_save and 
            (now - self.last_save).total_seconds() < self.save_interval_minutes * 60):
            return False
        
        try:
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            snapshot_id = f"snapshot_{timestamp}_{self.snapshot_counter:04d}"
            
            # Сохранение различных компонентов
            await self._save_metrics(bus, snapshot_id)
            await self._save_models(bus, snapshot_id)
            await self._save_snapshot(bus, snapshot_id)
            await self._save_logs(bus, snapshot_id)
            
            # Создание метаданных снапшота
            metadata = {
                'snapshot_id': snapshot_id,
                'timestamp': now.isoformat(),
                'components_saved': ['metrics', 'models', 'snapshot', 'logs'],
                'bus_state': {
                    'nodes_count': len(bus.nodes),
                    'channels_count': len(bus.channels),
                    'system_coherence': await bus._calculate_enhanced_coherence(),
                    'queue_sizes': {
                        'signal_queue': bus.signal_queue.qsize(),
                        'feedback_queue': bus.feedback_queue.qsize()
                    }
                }
            }
            
            await self._save_metadata(metadata, snapshot_id)
            
            # Очистка старых снапшотов
            await self._cleanup_old_snapshots()
            
            self.last_save = now
            self.snapshot_counter += 1
            
            print(f"[HISTORY] Снапшот сохранен: {snapshot_id}")
            return True
            
        except Exception as e:
            print(f"[HISTORY] Ошибка сохранения: {e}")
            return False
    
    async def _save_metrics(self, bus: 'SephiroticBus', snapshot_id: str):
        """Сохранение метрик"""
        metrics_file = self.save_dir / "metrics" / f"{snapshot_id}_metrics.json"
        
        metrics_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'channel_metrics': {},
            'system_metrics': await bus.get_network_state(),
            'tracer_analytics': bus.tracer.analyze_trace_patterns() if hasattr(bus, 'tracer') else {},
            'predictor_history': bus.predictor.training_history if hasattr(bus, 'predictor') else []
        }
        
        # Сохранение метрик каналов
        if hasattr(bus, 'channels'):
            for channel_id, channel in bus.channels.items():
                metrics_data['channel_metrics'][channel_id] = channel.get_health_report()
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False, default=str)
    
    async def _save_models(self, bus: 'SephiroticBus', snapshot_id: str):
        """Сохранение моделей машинного обучения"""
        if hasattr(bus, 'predictor') and bus.predictor.model:
            model = bus.predictor.model
            
            # Сохранение архитектуры модели
            arch_file = self.save_dir / "models" / f"{snapshot_id}_architecture.json"
            model_json = model.to_json()
            
            with open(arch_file, 'w', encoding='utf-8') as f:
                f.write(model_json)
            
            # Сохранение весов
            weights_file = self.save_dir / "models" / f"{snapshot_id}_weights.h5"
            model.save_weights(str(weights_file))
            
            # Сохранение scaler
            if bus.predictor.scaler:
                scaler_file = self.save_dir / "models" / f"{snapshot_id}_scaler.pkl"
                with open(scaler_file, 'wb') as f:
                    pickle.dump(bus.predictor.scaler, f)
    
    async def _save_snapshot(self, bus: 'SephiroticBus', snapshot_id: str):
        """Сохранение полного снапшота состояния"""
        snapshot_file = self.save_dir / "snapshots" / f"{snapshot_id}_full.pkl"
        
        # Подготовка данных для сериализации
        snapshot_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'nodes': {name: node.serialize() for name, node in bus.nodes.items() 
                     if hasattr(node, 'serialize')},
            'channels': {cid: asdict(channel) for cid, channel in bus.channels.items()},
            'queues': {
                'signal_queue_size': bus.signal_queue.qsize(),
                'feedback_queue_size': bus.feedback_queue.qsize()
            },
            'config': bus.config if hasattr(bus, 'config') else {}
        }
        
        # Сжатие данных если включено
        if self.compression_enabled:
            compressed_data = self._compress_data(snapshot_data)
            with open(snapshot_file, 'wb') as f:
                f.write(compressed_data)
        else:
            with open(snapshot_file, 'wb') as f:
                pickle.dump(snapshot_data, f)
    
    async def _save_logs(self, bus: 'SephiroticBus', snapshot_id: str):
        """Сохранение логов"""
        logs_file = self.save_dir / "logs" / f"{snapshot_id}_logs.json"
        
        logs_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'access_logs': list(bus.security_manager.access_log)[-1000:] if hasattr(bus, 'security_manager') else [],
            'token_history': list(bus.security_manager.token_history)[-500:] if hasattr(bus, 'security_manager') else [],
            'bus_events': list(bus.metrics.bus_events)[-1000:] if hasattr(bus, 'metrics') else []
        }
        
        with open(logs_file, 'w', encoding='utf-8') as f:
            json.dump(logs_data, f, indent=2, ensure_ascii=False, default=str)
    
    async def _save_metadata(self, metadata: Dict[str, Any], snapshot_id: str):
        """Сохранение метаданных снапшота"""
        meta_file = self.save_dir / "snapshots" / f"{snapshot_id}_metadata.json"
        
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
    
    def _compress_data(self, data: Any) -> bytes:
        """Сжатие данных"""
        buffer = io.BytesIO()
        
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Сериализация и сжатие
            pickled_data = pickle.dumps(data)
            
            # Добавление в zip
            zip_file.writestr('data.pkl', pickled_data)
        
        return buffer.getvalue()
    
    async def _cleanup_old_snapshots(self):
        """Очистка старых снапшотов"""
        try:
            # Получение всех снапшотов
            snapshot_files = list(self.save_dir.glob("snapshots/*_metadata.json"))
            
            if len(snapshot_files) <= self.max_snapshots:
                return
            
            # Сортировка по времени создания
            snapshot_files.sort(key=lambda x: x.stat().st_mtime)
            
            # Удаление старых файлов
            files_to_remove = snapshot_files[:-self.max_snapshots]
            
            for meta_file in files_to_remove:
                # Удаление всех связанных файлов
                snapshot_prefix = meta_file.stem.replace('_metadata', '')
                
                for ext in ['_full.pkl', '_metrics.json', '_architecture.json', 
                           '_weights.h5', '_scaler.pkl', '_logs.json']:
                    related_file = self.save_dir / "snapshots" / f"{snapshot_prefix}{ext}"
                    if related_file.exists():
                        related_file.unlink()
                
                # Удаление метаданных
                meta_file.unlink()
            
            print(f"[HISTORY] Удалено {len(files_to_remove)} старых снапшотов")
            
        except Exception as e:
            print(f"[HISTORY] Ошибка очистки снапшотов: {e}")
    
    async def restore_from_snapshot(self, snapshot_id: str, bus: 'SephiroticBus') -> bool:
        """Восстановление состояния из снапшота"""
        try:
            meta_file = self.save_dir / "snapshots" / f"{snapshot_id}_metadata.json"
            
            if not meta_file.exists():
                print(f"[HISTORY] Снапшот {snapshot_id} не найден")
                return False
            
            print(f"[HISTORY] Восстановление из снапшота: {snapshot_id}")
            
            # Здесь будет логика восстановления состояния
            # (зависит от структуры данных в шине)
            
            return True
            
        except Exception as e:
            print(f"[HISTORY] Ошибка восстановления: {e}")
            return False
    
    def get_storage_report(self) -> Dict[str, Any]:
        """Отчет о хранилище"""
        total_size = 0
        file_counts = defaultdict(int)
        
        # Расчет размера и количества файлов
        for root, dirs, files in os.walk(self.save_dir):
            for file in files:
                file_path = Path(root) / file
                total_size += file_path.stat().st_size
                
                # Подсчет по типам файлов
                ext = file_path.suffix
                file_counts[ext] += 1
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'storage_path': str(self.save_dir),
            'total_size_mb': total_size / (1024 * 1024),
            'file_counts': dict(file_counts),
            'snapshots_stored': len(list(self.save_dir.glob("snapshots/*_metadata.json"))),
            'last_save': self.last_save.isoformat() if self.last_save else None,
            'next_scheduled_save': (
                (self.last_save + timedelta(minutes=self.save_interval_minutes)).isoformat()
                if self.last_save else None
            ),
            'compression_enabled': self.compression_enabled
        }


# ============================================================================
# МОДУЛЬ ОБЪЯСНИМОСТИ ИИ
# ============================================================================

class AIExplainabilityLogger:
    """Логирование объяснимости решений ИИ"""
    
    def __init__(self, log_dir: str = "data/ai_explanations"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.decision_logs: deque = deque(maxlen=5000)
        self.feature_importance_history: Dict[str, List[Tuple[datetime, List[float]]]] = defaultdict(list)
        self.prediction_insights: deque = deque(maxlen=1000)
        
        print(f"[EXPLAINABILITY] Логгер объяснимости ИИ инициализирован")
    
    def log_lstm_decision(self, channel_id: str, inputs: np.ndarray, 
                         predictions: np.ndarray, actual_values: Optional[np.ndarray] = None,
                         feature_names: List[str] = None, confidence: float = 1.0):
        """Логирование решения LSTM модели"""
        
        timestamp = datetime.utcnow()
        
        # Анализ важности признаков (упрощенный)
        if len(inputs.shape) == 3 and inputs.shape[0] == 1:
            input_sequence = inputs[0].flatten()
            
            # Простой анализ: какие значения изменились больше всего
            if len(input_sequence) > 1:
                diffs = np.abs(np.diff(input_sequence))
                important_indices = np.argsort(diffs)[-3:]  # Топ-3 изменения
                
                if feature_names and len(feature_names) >= len(input_sequence):
                    important_features = [
                        feature_names[i] for i in important_indices 
                        if i < len(feature_names)
                    ]
                else:
                    important_features = [f"feature_{i}" for i in important_indices]
                
                # Сохранение важности признаков
                self.feature_importance_history[channel_id].append(
                    (timestamp, diffs.tolist())
                )
            else:
                important_features = ["single_feature"]
        
        # Формирование insight
        insight = self._generate_prediction_insight(
            predictions, actual_values, confidence
        )
        
        # Запись лога
        log_entry = {
            'timestamp': timestamp.isoformat(),
            'channel_id': channel_id,
            'model_type': 'LSTM',
            'input_shape': inputs.shape,
            'input_summary': {
                'mean': float(np.mean(inputs)),
                'std': float(np.std(inputs)),
                'min': float(np.min(inputs)),
                'max': float(np.max(inputs))
            },
            'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
            'actual_values': actual_values.tolist() if actual_values is not None and hasattr(actual_values, 'tolist') else actual_values,
            'important_features': important_features,
            'confidence': confidence,
            'insight': insight,
            'explanation': self._explain_prediction(predictions, insight)
        }
        
        self.decision_logs.append(log_entry)
        self.prediction_insights.append(insight)
        
        # Автосохранение каждые 100 записей
        if len(self.decision_logs) % 100 == 0:
            self._auto_save_logs()
        
        return log_entry
    
    def _generate_prediction_insight(self, predictions: np.ndarray, 
                                    actual_values: Optional[np.ndarray],
                                    confidence: float) -> Dict[str, Any]:
        """Генерация инсайта из предсказания"""
        
        if len(predictions) == 0:
            return {"type": "empty_prediction", "confidence": 0}
        
        # Анализ тренда
        if len(predictions) >= 2:
            trend = "stable"
            first_val = predictions[0]
            last_val = predictions[-1]
            
            if last_val > first_val * 1.1:
                trend = "improving"
            elif last_val < first_val * 0.9:
                trend = "degrading"
            
            # Анализ волатильности
            volatility = np.std(predictions) / (np.mean(predictions) + 1e-10)
            
            # Проверка на аномалии
            anomalies = []
            mean_val = np.mean(predictions)
            std_val = np.std(predictions)
            
            for i, val in enumerate(predictions):
                if abs(val - mean_val) > 2 * std_val:
                    anomalies.append({
                        'position': i,
                        'value': float(val),
                        'deviation': float(abs(val - mean_val) / std_val)
                    })
        else:
            trend = "unknown"
            volatility = 0
            anomalies = []
        
        # Сравнение с фактическими значениями если есть
        accuracy = None
        if actual_values is not None and len(actual_values) == len(predictions):
            mae = np.mean(np.abs(predictions - actual_values))
            accuracy = 1.0 / (1.0 + mae)
        
        return {
            'trend': trend,
            'volatility': float(volatility),
            'anomalies': anomalies,
            'prediction_range': {
                'min': float(np.min(predictions)),
                'max': float(np.max(predictions)),
                'mean': float(np.mean(predictions))
            },
            'accuracy': accuracy,
            'confidence': confidence,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _explain_prediction(self, predictions: np.ndarray, insight: Dict[str, Any]) -> str:
        """Генерация текстового объяснения предсказания"""
        
        trend = insight.get('trend', 'unknown')
        volatility = insight.get('volatility', 0)
        anomalies = insight.get('anomalies', [])
        
        explanation_parts = []
        
        # Объяснение тренда
        if trend == "improving":
            explanation_parts.append("Система предсказывает улучшение состояния канала.")
        elif trend == "degrading":
            explanation_parts.append("Обнаружена тенденция к деградации канала.")
        else:
            explanation_parts.append("Состояние канала остается стабильным.")
        
        # Объяснение волатильности
        if volatility > 0.5:
            explanation_parts.append("Высокая волатильность указывает на нестабильность канала.")
        elif volatility > 0.2:
            explanation_parts.append("Умеренная волатильность наблюдается в предсказаниях.")
        else:
            explanation_parts.append("Предсказания демонстрируют низкую волатильность.")
        
        # Объяснение аномалий
        if anomalies:
            explanation_parts.append(f"Обнаружено {len(anomalies)} аномальных точек в предсказании.")
        
        # Добавление рекомендации
        if trend == "degrading" and volatility > 0.3:
            explanation_parts.append("Рекомендуется провести диагностику канала.")
        
        return " ".join(explanation_parts)
    
    def _auto_save_logs(self):
        """Автосохранение логов"""
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M")
            log_file = self.log_dir / f"explanations_{timestamp}.json"
            
            logs_to_save = {
                'timestamp': datetime.utcnow().isoformat(),
                'total_entries': len(self.decision_logs),
                'recent_decisions': list(self.decision_logs)[-100:],
                'feature_importance_summary': self._summarize_feature_importance(),
                'insight_statistics': self._analyze_insights()
            }
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs_to_save, f, indent=2, ensure_ascii=False, default=str)
                
        except Exception as e:
            print(f"[EXPLAINABILITY] Ошибка автосохранения: {e}")
    
    def _summarize_feature_importance(self) -> Dict[str, Any]:
        """Суммаризация важности признаков"""
        summary = {}
        
        for channel_id, importance_history in self.feature_importance_history.items():
            if importance_history:
                # Берем последние записи
                recent_history = importance_history[-10:]
                all_importances = []
                
                for timestamp, importances in recent_history:
                    all_importances.extend(importances)
                
                if all_importances:
                    summary[channel_id] = {
                        'avg_importance': statistics.mean(all_importances),
                        'max_importance': max(all_importances),
                        'stability': 1.0 - (statistics.stdev(all_importances) / (statistics.mean(all_importances) + 1e-10))
                    }
        
        return summary
    
    def _analyze_insights(self) -> Dict[str, Any]:
        """Анализ инсайтов"""
        if not self.prediction_insights:
            return {}
        
        insights = list(self.prediction_insights)
        
        trend_counts = defaultdict(int)
        anomaly_counts = []
        confidence_values = []
        
        for insight in insights:
            trend_counts[insight.get('trend', 'unknown')] += 1
            
            if 'anomalies' in insight:
                anomaly_counts.append(len(insight['anomalies']))
            
            if 'confidence' in insight:
                confidence_values.append(insight['confidence'])
        
        return {
            'trend_distribution': dict(trend_counts),
            'avg_anomalies_per_prediction': statistics.mean(anomaly_counts) if anomaly_counts else 0,
            'avg_confidence': statistics.mean(confidence_values) if confidence_values else 0,
            'total_insights_analyzed': len(insights)
        }
    
    def get_explainability_report(self, channel_id: str = None) -> Dict[str, Any]:
        """Отчет по объяснимости"""
        
        if channel_id:
            # Отчет для конкретного канала
            channel_logs = [
                log for log in self.decision_logs 
                if log['channel_id'] == channel_id
            ]
            
            if not channel_logs:
                return {"error": f"No logs for channel {channel_id}"}
            
            recent_logs = channel_logs[-10:]
            
            # Анализ важности признаков для этого канала
            feature_history = self.feature_importance_history.get(channel_id, [])
            
            return {
                'channel_id': channel_id,
                'total_decisions_logged': len(channel_logs),
                'recent_decisions': recent_logs,
                'feature_importance_trend': self._analyze_feature_trend(feature_history),
                'prediction_accuracy_trend': self._analyze_accuracy_trend(channel_logs),
                'recommended_actions': self._generate_channel_recommendations(channel_logs)
            }
        
        else:
            # Общий отчет
            return {
                'timestamp': datetime.utcnow().isoformat(),
                'total_decisions_logged': len(self.decision_logs),
                'channels_monitored': list(set(
                    log['channel_id'] for log in self.decision_logs
                )),
                'insight_statistics': self._analyze_insights(),
                'feature_importance_summary': self._summarize_feature_importance(),
                'model_confidence_distribution': self._analyze_confidence_distribution(),
                'explainability_score': self._calculate_explainability_score()
            }
    
    def _analyze_feature_trend(self, feature_history: List[Tuple[datetime, List[float]]]) -> Dict[str, Any]:
        """Анализ тренда важности признаков"""
        if not feature_history:
            return {}
        
        # Берем последние 5 записей
        recent = feature_history[-5:]
        
        # Усреднение по времени
        all_importances = []
        for timestamp, importances in recent:
            all_importances.extend(importances)
        
        if not all_importances:
            return {}
        
        return {
            'average_importance': statistics.mean(all_importances),
            'importance_volatility': statistics.stdev(all_importances) / (statistics.mean(all_importances) + 1e-10),
            'trend': 'increasing' if len(all_importances) > 1 and all_importances[-1] > all_importances[0] * 1.1 else 'stable'
        }
    
    def _analyze_accuracy_trend(self, channel_logs: List[Dict]) -> Dict[str, Any]:
        """Анализ тренда точности предсказаний"""
        if not channel_logs:
            return {}
        
        accuracies = []
        for log in channel_logs:
            if 'actual_values' in log and log['actual_values'] is not None:
                # Упрощенный расчет точности
                predictions = np.array(log['predictions'])
                actuals = np.array(log['actual_values'])
                
                if len(predictions) == len(actuals):
                    mae = np.mean(np.abs(predictions - actuals))
                    accuracy = 1.0 / (1.0 + mae)
                    accuracies.append(accuracy)
        
        if not accuracies:
            return {}
        
        return {
            'average_accuracy': statistics.mean(accuracies),
            'accuracy_trend': 'improving' if len(accuracies) > 1 and accuracies[-1] > accuracies[0] else 'stable',
            'accuracy_stability': 1.0 - (statistics.stdev(accuracies) / (statistics.mean(accuracies) + 1e-10))
        }
    
    def _analyze_confidence_distribution(self) -> Dict[str, Any]:
        """Анализ распределения уверенности модели"""
        confidences = [log.get('confidence', 0) for log in self.decision_logs if 'confidence' in log]
        
        if not confidences:
            return {}
        
        return {
            'average_confidence': statistics.mean(confidences),
            'confidence_std': statistics.stdev(confidences) if len(confidences) > 1 else 0,
            'high_confidence_percentage': len([c for c in confidences if c > 0.8]) / len(confidences) * 100,
            'low_confidence_percentage': len([c for c in confidences if c < 0.3]) / len(confidences) * 100
        }
    
    def _calculate_explainability_score(self) -> float:
        """Расчет оценки объяснимости"""
        score = 0.5  # Базовая оценка
        
        # Факторы увеличивающие оценку
       
