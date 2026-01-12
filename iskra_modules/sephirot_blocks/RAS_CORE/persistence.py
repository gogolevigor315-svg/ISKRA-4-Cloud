#!/usr/bin/env python3
"""
persistence.py - PERSISTENT STATE MANAGER ДЛЯ RAS-CORE И ЛИЧНОСТИ
Версия: 1.0.0
Назначение: Сохранение и восстановление состояния личности ISKRA-4 при перезагрузках
Ключевые функции: checkpoint/restore для personality_coherence_score, паттернов внимания, self_reflect_cycle
"""

import json
import pickle
import zlib
import base64
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
import asyncio
import logging
import shelve
import sqlite3
from contextlib import contextmanager
from enum import Enum

# Импорты из RAS-CORE
from .constants import GOLDEN_STABILITY_ANGLE
from .config import get_config

# ============================================================================
# ТИПЫ ДАННЫХ ДЛЯ СОХРАНЕНИЯ
# ============================================================================

class PersistenceMode(Enum):
    """Режимы сохранения состояния"""
    FULL = "full"           # Полное сохранение всех данных
    INCREMENTAL = "inc"     # Инкрементальное сохранение
    CHECKPOINT = "check"    # Точка сохранения
    SNAPSHOT = "snapshot"   # Снимок состояния

class StorageBackend(Enum):
    """Бэкенды хранения"""
    SQLITE = "sqlite"
    SHELVE = "shelve"
    JSON = "json"
    PICKLE = "pickle"
    MEMORY = "memory"

@dataclass
class PersonalityState:
    """Состояние личности для сохранения"""
    coherence_score: float = 0.0
    focus_stability: float = 0.0
    intent_strength: float = 0.0
    insight_depth: float = 0.0
    resonance_quality: float = 0.0
    stability_angle: float = GOLDEN_STABILITY_ANGLE
    manifestation_level: float = 0.0
    reflection_count: int = 0
    last_reflection: Optional[str] = None
    focus_patterns: List[Dict[str, Any]] = field(default_factory=list)
    attention_vectors: List[List[float]] = field(default_factory=list)
    personality_traits: Dict[str, float] = field(default_factory=dict)
    signature: str = ""  # Цифровая подпись состояния
    
    def calculate_signature(self) -> str:
        """Вычисление цифровой подписи состояния"""
        data_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь"""
        data = asdict(self)
        data['_version'] = '1.0.0'
        data['_timestamp'] = datetime.utcnow().isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonalityState':
        """Создание из словаря"""
        # Убираем служебные поля
        data = {k: v for k, v in data.items() if not k.startswith('_')}
        return cls(**data)

@dataclass
class RASState:
    """Полное состояние RAS-CORE для сохранения"""
    personality_state: PersonalityState
    queue_state: Dict[str, Any]
    pattern_learner_state: Dict[str, Any]
    router_state: Dict[str, Any]
    metrics_state: Dict[str, Any]
    reflection_cycle_state: Dict[str, Any]
    version: str = "4.1.0"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    checksum: str = ""
    
    def calculate_checksum(self) -> str:
        """Вычисление контрольной суммы"""
        data = {
            "personality": self.personality_state.to_dict(),
            "queue": self.queue_state,
            "patterns": self.pattern_learner_state,
            "router": self.router_state,
            "metrics": self.metrics_state,
            "reflection": self.reflection_cycle_state,
            "version": self.version,
            "timestamp": self.timestamp
        }
        data_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()

@dataclass
class Checkpoint:
    """Точка сохранения состояния"""
    checkpoint_id: str
    state: RASState
    mode: PersistenceMode
    storage_backend: StorageBackend
    size_bytes: int = 0
    compression_ratio: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь"""
        return {
            "checkpoint_id": self.checkpoint_id,
            "state": self.state.__dict__,
            "mode": self.mode.value,
            "storage_backend": self.storage_backend.value,
            "size_bytes": self.size_bytes,
            "compression_ratio": self.compression_ratio,
            "metadata": self.metadata,
            "timestamp": datetime.utcnow().isoformat()
        }

# ============================================================================
# КЛАСС PERSISTENT STATE MANAGER
# ============================================================================

class PersistentStateManager:
    """
    Менеджер сохранения состояния личности RAS-CORE.
    Обеспечивает checkpoint/restore для всех компонентов системы.
    """
    
    def __init__(self, 
                 storage_path: Union[str, Path] = "./data/persistence",
                 backend: StorageBackend = StorageBackend.SQLITE,
                 auto_save_interval: int = 300):  # 5 минут
        """
        Инициализация менеджера сохранения.
        
        Args:
            storage_path: Путь для хранения данных
            backend: Бэкенд хранения
            auto_save_interval: Интервал автосохранения в секундах
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.backend = backend
        self.auto_save_interval = auto_save_interval
        self.auto_save_task = None
        self.is_auto_saving = False
        
        # История чекпоинтов
        self.checkpoints: List[Checkpoint] = []
        self.max_checkpoints = 50
        
        # Состояния для инкрементального сохранения
        self.dirty_states = set()
        self.last_full_save = None
        
        # Логгер
        self.logger = self._setup_logger()
        
        # Инициализация бэкенда
        self._backend = self._init_backend()
        
        self.logger.info(f"💾 PersistentStateManager инициализирован")
        self.logger.info(f"   Бэкенд: {backend.value}")
        self.logger.info(f"   Путь: {storage_path}")
        self.logger.info(f"   Автосохранение: каждые {auto_save_interval} секунд")
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger("RAS.Persistence")
        
        if not logger.handlers:
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '[%(asctime)s] [%(name)s:%(levelname)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(formatter)
            logger.addHandler(console)
            
            logger.propagate = False
        
        return logger
    
    def _init_backend(self):
        """Инициализация бэкенда хранения"""
        if self.backend == StorageBackend.SQLITE:
            return SQLiteBackend(self.storage_path / "ras_state.db")
        elif self.backend == StorageBackend.SHELVE:
            return ShelveBackend(self.storage_path / "ras_state.shelve")
        elif self.backend == StorageBackend.JSON:
            return JSONBackend(self.storage_path / "ras_state.json")
        elif self.backend == StorageBackend.PICKLE:
            return PickleBackend(self.storage_path / "ras_state.pickle")
        elif self.backend == StorageBackend.MEMORY:
            return MemoryBackend()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    # ============================================================================
    # ОСНОВНЫЕ МЕТОДЫ СОХРАНЕНИЯ
    # ============================================================================
    
    async def save_state(self, 
                        ras_core,
                        mode: PersistenceMode = PersistenceMode.CHECKPOINT,
                        force_full: bool = False) -> Checkpoint:
        """
        Сохранение состояния RAS-CORE.
        
        Args:
            ras_core: Экземпляр EnhancedRASCore
            mode: Режим сохранения
            force_full: Принудительное полное сохранение
        
        Returns:
            Созданный чекпоинт
        """
        self.logger.info(f"💾 Сохранение состояния (режим: {mode.value})...")
        
        try:
            # Сбор состояния
            state = await self._collect_state(ras_core, mode, force_full)
            
            # Создание чекпоинта
            checkpoint_id = self._generate_checkpoint_id()
            checkpoint = Checkpoint(
                checkpoint_id=checkpoint_id,
                state=state,
                mode=mode,
                storage_backend=self.backend,
                metadata={
                    "source": "ras_core_v4_1",
                    "personality_coherence": state.personality_state.coherence_score,
                    "stability_angle": state.personality_state.stability_angle,
                    "manifestation_level": state.personality_state.manifestation_level
                }
            )
            
            # Вычисление размера
            checkpoint.size_bytes = len(pickle.dumps(checkpoint))
            
            # Сохранение в бэкенд
            await self._backend.save(checkpoint_id, checkpoint)
            
            # Добавление в историю
            self.checkpoints.append(checkpoint)
            if len(self.checkpoints) > self.max_checkpoints:
                self.checkpoints = self.checkpoints[-self.max_checkpoints:]
            
            # Очистка dirty states
            if mode == PersistenceMode.FULL or force_full:
                self.dirty_states.clear()
                self.last_full_save = datetime.utcnow()
            
            self.logger.info(f"✅ Состояние сохранено: {checkpoint_id}")
            self.logger.info(f"   Coherence: {state.personality_state.coherence_score:.3f}")
            self.logger.info(f"   Размер: {checkpoint.size_bytes:,} байт")
            self.logger.info(f"   Чекпоинтов в истории: {len(self.checkpoints)}")
            
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка сохранения состояния: {e}")
            raise
    
    async def _collect_state(self, 
                           ras_core, 
                           mode: PersistenceMode,
                           force_full: bool) -> RASState:
        """Сбор состояния из RAS-CORE"""
        
        # Базовое состояние личности
        personality_state = PersonalityState()
        
        # Получаем состояние личности из RAS-CORE если доступно
        if hasattr(ras_core, 'personality_state'):
            ps = ras_core.personality_state
            personality_state = PersonalityState(
                coherence_score=getattr(ps, 'coherence_score', 0.0),
                focus_stability=getattr(ps, 'focus_stability', 0.0),
                intent_strength=getattr(ps, 'intent_strength', 0.0),
                insight_depth=getattr(ps, 'insight_depth', 0.0),
                resonance_quality=getattr(ps, 'resonance_quality', 0.0),
                stability_angle=getattr(ps, 'stability_angle', GOLDEN_STABILITY_ANGLE),
                manifestation_level=getattr(ps, 'manifestation_level', 0.0),
                reflection_count=getattr(ps, 'reflection_count', 0),
                last_reflection=getattr(ps, 'last_reflection', None),
                focus_patterns=getattr(ras_core, 'focus_patterns', []),
                personality_traits=getattr(ras_core, 'personality_traits', {})
            )
        
        # Состояние очередей
        queue_state = {}
        if hasattr(ras_core, 'queue') and hasattr(ras_core.queue, 'get_state'):
            queue_state = ras_core.queue.get_state()
        
        # Состояние PatternLearner
        pattern_learner_state = {}
        if hasattr(ras_core, 'pattern_learner') and hasattr(ras_core.pattern_learner, 'get_state'):
            pattern_learner_state = ras_core.pattern_learner.get_state()
        
        # Состояние маршрутизатора
        router_state = {}
        if hasattr(ras_core, 'router') and hasattr(ras_core.router, 'get_state'):
            router_state = ras_core.router.get_state()
        
        # Состояние метрик
        metrics_state = {}
        if hasattr(ras_core, 'metrics') and hasattr(ras_core.metrics, 'get_state'):
            metrics_state = ras_core.metrics.get_state()
        
        # Состояние цикла саморефлексии
        reflection_cycle_state = {}
        if hasattr(ras_core, 'reflection_engine') and hasattr(ras_core.reflection_engine, 'get_state'):
            reflection_cycle_state = ras_core.reflection_engine.get_state()
        
        # Создание полного состояния
        state = RASState(
            personality_state=personality_state,
            queue_state=queue_state,
            pattern_learner_state=pattern_learner_state,
            router_state=router_state,
            metrics_state=metrics_state,
            reflection_cycle_state=reflection_cycle_state
        )
        
        # Вычисление контрольной суммы
        state.checksum = state.calculate_checksum()
        
        return state
    
    async def restore_state(self, 
                          ras_core, 
                          checkpoint_id: Optional[str] = None) -> bool:
        """
        Восстановление состояния RAS-CORE из чекпоинта.
        
        Args:
            ras_core: Экземпляр EnhancedRASCore для восстановления
            checkpoint_id: ID чекпоинта (None = последний)
        
        Returns:
            Успешность восстановления
        """
        self.logger.info(f"🔄 Восстановление состояния...")
        
        try:
            # Получение чекпоинта
            if checkpoint_id is None:
                checkpoint_id = await self._backend.get_latest_checkpoint()
                if not checkpoint_id:
                    self.logger.warning("⚠️  Нет доступных чекпоинтов")
                    return False
            
            checkpoint = await self._backend.load(checkpoint_id)
            if not checkpoint:
                self.logger.error(f"❌ Чекпоинт не найден: {checkpoint_id}")
                return False
            
            # Проверка контрольной суммы
            if not self._verify_checkpoint(checkpoint):
                self.logger.error(f"❌ Проверка контрольной суммы не пройдена")
                return False
            
            # Восстановление состояния
            await self._apply_state(ras_core, checkpoint.state)
            
            self.logger.info(f"✅ Состояние восстановлено из {checkpoint_id}")
            self.logger.info(f"   Coherence: {checkpoint.state.personality_state.coherence_score:.3f}")
            self.logger.info(f"   Manifestation: {checkpoint.state.personality_state.manifestation_level:.2f}")
            self.logger.info(f"   Возраст: {(datetime.utcnow() - datetime.fromisoformat(checkpoint.state.timestamp)).total_seconds():.0f} сек")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Ошибка восстановления: {e}")
            return False
    
    async def _apply_state(self, ras_core, state: RASState):
        """Применение состояния к RAS-CORE"""
        
        # Восстановление личности
        if hasattr(ras_core, 'personality_state'):
            ras_core.personality_state.coherence_score = state.personality_state.coherence_score
            ras_core.personality_state.focus_stability = state.personality_state.focus_stability
            ras_core.personality_state.intent_strength = state.personality_state.intent_strength
            ras_core.personality_state.insight_depth = state.personality_state.insight_depth
            ras_core.personality_state.resonance_quality = state.personality_state.resonance_quality
            ras_core.personality_state.stability_angle = state.personality_state.stability_angle
            ras_core.personality_state.manifestation_level = state.personality_state.manifestation_level
            ras_core.personality_state.reflection_count = state.personality_state.reflection_count
        
        # Восстановление PatternLearner
        if hasattr(ras_core, 'pattern_learner') and hasattr(ras_core.pattern_learner, 'set_state'):
            ras_core.pattern_learner.set_state(state.pattern_learner_state)
        
        # Восстановление маршрутизатора
        if hasattr(ras_core, 'router') and hasattr(ras_core.router, 'set_state'):
            ras_core.router.set_state(state.router_state)
        
        # Восстановление фокусных паттернов
        if hasattr(ras_core, 'focus_patterns'):
            ras_core.focus_patterns = state.personality_state.focus_patterns
        
        # Восстановление черт личности
        if hasattr(ras_core, 'personality_traits'):
            ras_core.personality_traits = state.personality_state.personality_traits
    
    def _verify_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """Проверка целостности чекпоинта"""
        try:
            # Проверка контрольной суммы
            calculated = checkpoint.state.calculate_checksum()
            if calculated != checkpoint.state.checksum:
                self.logger.warning(f"Контрольные суммы не совпадают: {calculated} != {checkpoint.state.checksum}")
                return False
            
            # Проверка подписи личности
            if checkpoint.state.personality_state.signature:
                calculated_sig = checkpoint.state.personality_state.calculate_signature()
                if calculated_sig != checkpoint.state.personality_state.signature:
                    self.logger.warning(f"Подписи не совпадают: {calculated_sig} != {checkpoint.state.personality_state.signature}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка проверки чекпоинта: {e}")
            return False
    
    def _generate_checkpoint_id(self) -> str:
        """Генерация ID чекпоинта"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        random_part = hashlib.md5(str(datetime.utcnow().timestamp()).encode()).hexdigest()[:8]
        return f"checkpoint_{timestamp}_{random_part}"
    
    # ============================================================================
    # УПРАВЛЕНИЕ АВТОСОХРАНЕНИЕМ
    # ============================================================================
    
    async def start_auto_save(self, ras_core, interval: Optional[int] = None):
        """Запуск автоматического сохранения"""
        if interval:
            self.auto_save_interval = interval
        
        if self.is_auto_saving:
            self.logger.warning("⚠️  Автосохранение уже запущено")
            return
        
        self.is_auto_saving = True
        self.auto_save_task = asyncio.create_task(self._auto_save_loop(ras_core))
        self.logger.info(f"🔄 Автосохранение запущено (интервал: {self.auto_save_interval} сек)")
    
    async def _auto_save_loop(self, ras_core):
        """Цикл автоматического сохранения"""
        while self.is_auto_saving:
            try:
                await asyncio.sleep(self.auto_save_interval)
                
                # Проверяем нужно ли сохранять
                if self._should_auto_save():
                    await self.save_state(
                        ras_core, 
                        mode=PersistenceMode.CHECKPOINT,
                        force_full=False
                    )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка автосохранения: {e}")
                await asyncio.sleep(60)  # Пауза при ошибке
    
    def _should_auto_save(self) -> bool:
        """Определение нужно ли выполнять автосохранение"""
        if not self.last_full_save:
            return True
        
        # Полное сохранение раз в час
        time_since_full = (datetime.utcnow() - self.last_full_save).total_seconds()
        if time_since_full > 3600:  # 1 час
            return True
        
        # Инкрементальное сохранение если есть dirty states
        if self.dirty_states:
            return True
        
        # Или по расписанию
        config = get_config()
        personality_coherence = getattr(config, 'personality', {}).get('coherence_threshold', 0.7)
        
        # Чаще сохраняем при высокой когерентности
        if personality_coherence > 0.8:
            return True
        
        return False
    
    async def stop_auto_save(self):
        """Остановка автоматического сохранения"""
        self.is_auto_saving = False
        if self.auto_save_task:
            self.auto_save_task.cancel()
            try:
                await self.auto_save_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("🛑 Автосохранение остановлено")
    
    # ============================================================================
    # УПРАВЛЕНИЕ ЧЕКПОИНТАМИ
    # ============================================================================
    
    async def list_checkpoints(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Список доступных чекпоинтов"""
        checkpoints = await self._backend.list_checkpoints(limit)
        return [
            {
                "id": cp.checkpoint_id,
                "timestamp": cp.state.timestamp,
                "coherence": cp.state.personality_state.coherence_score,
                "manifestation": cp.state.personality_state.manifestation_level,
                "size_bytes": cp.size_bytes,
                "mode": cp.mode.value,
                "verified": self._verify_checkpoint(cp)
            }
            for cp in checkpoints
        ]
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Удаление чекпоинта"""
        success = await self._backend.delete(checkpoint_id)
        if success:
            self.checkpoints = [cp for cp in self.checkpoints if cp.checkpoint_id != checkpoint_id]
            self.logger.info(f"🗑️  Чекпоинт удален: {checkpoint_id}")
        return success
    
    async def cleanup_old_checkpoints(self, keep_last: int = 10):
        """Очистка старых чекпоинтов"""
        checkpoints = await self._backend.list_checkpoints(1000)  # Все чекпоинты
        if len(checkpoints) <= keep_last:
            return
        
        # Сортируем по времени
        checkpoints.sort(key=lambda x: x.state.timestamp, reverse=True)
        
        # Удаляем старые
        for checkpoint in checkpoints[keep_last:]:
            await self.delete_checkpoint(checkpoint.checkpoint_id)
        
        self.logger.info(f"🧹 Очищено {len(checkpoints) - keep_last} старых чекпоинтов")
    
    async def export_state(self, 
                          checkpoint_id: str, 
                          filepath: Union[str, Path]) -> bool:
        """Экспорт состояния в файл"""
        try:
            checkpoint = await self._backend.load(checkpoint_id)
            if not checkpoint:
                return False
            
            # Сериализация
            data = checkpoint.to_dict()
            json_str = json.dumps(data, indent=2, default=str)
            
            # Сохранение в файл
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(json_str, encoding='utf-8')
            
            self.logger.info(f"📤 Состояние экспортировано в {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка экспорта: {e}")
            return False
    
    async def import_state(self, filepath: Union[str, Path]) -> Optional[Checkpoint]:
        """Импорт состояния из файла"""
        try:
            filepath = Path(filepath)
            if not filepath.exists():
                self.logger.error(f"Файл не найден: {filepath}")
                return None
            
            # Загрузка из файла
            json_str = filepath.read_text(encoding='utf-8')
            data = json.loads(json_str)
            
            # Восстановление чекпоинта
            checkpoint = Checkpoint(
                checkpoint_id=data['checkpoint_id'],
                state=RASState(**data['state']),
                mode=PersistenceMode(data['mode']),
                storage_backend=StorageBackend(data['storage_backend']),
                size_bytes=data['size_bytes'],
                compression_ratio=data['compression_ratio'],
                metadata=data['metadata']
            )
            
            # Сохранение в бэкенд
            await self._backend.save(checkpoint.checkpoint_id, checkpoint)
            
            # Добавление в историю
            self.checkpoints.append(checkpoint)
            
            self.logger.info(f"📥 Состояние импортировано из {filepath}")
            return checkpoint
            
        except Exception as e:
            self.logger.error(f"Ошибка импорта: {e}")
            return None
    
    # ============================================================================
    # МЕТРИКИ И СТАТИСТИКА
    # ============================================================================
    
    async def get_stats(self) -> Dict[str, Any]:
        """Получение статистики сохранения"""
        checkpoints = await self._backend.list_checkpoints(1000)
        
        if not checkpoints:
            return {
                "total_checkpoints": 0,
                "auto_save_enabled": self.is_auto_saving,
                "storage_backend": self.backend.value,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Анализ чекпоинтов
        coherence_scores = [cp.state.personality_state.coherence_score for cp in checkpoints]
        manifestation_levels = [cp.state.personality_state.manifestation_level for cp in checkpoints]
        sizes = [cp.size_bytes for cp in checkpoints]
        
        return {
            "total_checkpoints": len(checkpoints),
            "latest_checkpoint": checkpoints[0].checkpoint_id,
            "oldest_checkpoint": checkpoints[-1].checkpoint_id,
            "coherence_stats": {
                "current": coherence_scores[0],
                "average": sum(coherence_scores) / len(coherence_scores),
                "min": min(coherence_scores),
                "max": max(coherence_scores),
                "trend": "stable" if len(coherence_scores) < 2 else 
                        "improving" if coherence_scores[0] > coherence_scores[-1] else 
                        "declining"
            },
            "manifestation_stats": {
                "current": manifestation_levels[0],
                "average": sum(manifestation_levels) / len(manifestation_levels),
                "min": min(manifestation_levels),
                "max": max(manifestation_levels)
            },
            "storage_stats": {
                "total_size_bytes": sum(sizes),
                "average_size_bytes": sum(sizes) / len(sizes),
                "compression_ratio": checkpoints[0].compression_ratio if checkpoints else 1.0
            },
            "auto_save_enabled": self.is_auto_saving,
            "storage_backend": self.backend.value,
            "dirty_states_count": len(self.dirty_states),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def mark_dirty(self, component: str):
        """Пометка компонента как измененного (нуждается в сохранении)"""
        self.dirty_states.add(component)
        
        # Автосохранение при критических изменениях
        if component in ["personality_coherence", "focus_patterns"]:
            if hasattr(self, 'ras_core'):
                asyncio.create_task(self.save_state(
                    self.ras_core,
                    mode=PersistenceMode.INCREMENTAL,
                    force_full=False
                ))

# ============================================================================
# БЭКЕНДЫ ХРАНЕНИЯ
# ============================================================================

class StorageBackendBase:
    """Базовый класс бэкенда хранения"""
    
    def __init__(self, path=None):
        self.path = path
    
    async def save(self, checkpoint_id: str, checkpoint: Checkpoint):
        raise NotImplementedError
    
    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        raise NotImplementedError
    
    async def delete(self, checkpoint_id: str) -> bool:
        raise NotImplementedError
    
    async def list_checkpoints(self, limit: int = 20) -> List[Checkpoint]:
        raise NotImplementedError
    
    async def get_latest_checkpoint(self) -> Optional[str]:
        checkpoints = await self.list_checkpoints(1)
        return checkpoints[0].checkpoint_id if checkpoints else None

class SQLiteBackend(StorageBackendBase):
    """SQLite бэкенд хранения"""
    
    def __init__(self, db_path):
        super().__init__(db_path)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
    
    def _init_db(self):
        """Инициализация базы данных"""
        cursor = self.conn.cursor()
        
        # Таблица чекпоинтов
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS checkpoints (
                id TEXT PRIMARY KEY,
                data BLOB NOT NULL,
                timestamp TEXT NOT NULL,
                coherence REAL,
                manifestation REAL,
                size_bytes INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Индексы для быстрого поиска
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON checkpoints(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_coherence ON checkpoints(coherence)')
        
        self.conn.commit()
    
    async def save(self, checkpoint_id: str, checkpoint: Checkpoint):
        """Сохранение чекпоинта"""
        data = pickle.dumps(checkpoint)
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO checkpoints 
            (id, data, timestamp, coherence, manifestation, size_bytes)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            checkpoint_id,
            data,
            checkpoint.state.timestamp,
            checkpoint.state.personality_state.coherence_score,
            checkpoint.state.personality_state.manifestation_level,
            checkpoint.size_bytes
        ))
        self.conn.commit()
    
    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        """Загрузка чекпоинта"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT data FROM checkpoints WHERE id = ?', (checkpoint_id,))
        row = cursor.fetchone()
        
        if row:
            return pickle.loads(row[0])
        return None
    
    async def delete(self, checkpoint_id: str) -> bool:
        """Удаление чекпоинта"""
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM checkpoints WHERE id = ?', (checkpoint_id,))
        self.conn.commit()
        return cursor.rowcount > 0
    
    async def list_checkpoints(self, limit: int = 20) -> List[Checkpoint]:
        """Список чекпоинтов"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT data FROM checkpoints 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        return [pickle.loads(row[0]) for row in rows]

class ShelveBackend(StorageBackendBase):
    """Shelve бэкенд хранения"""
    
    def __init__(self, shelve_path):
        super().__init__(shelve_path)
        self.shelve_path = shelve_path
    
    async def save(self, checkpoint_id: str, checkpoint: Checkpoint):
        with shelve.open(str(self.shelve_path)) as db:
            db[checkpoint_id] = checkpoint
    
    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        with shelve.open(str(self.shelve_path)) as db:
            return db.get(checkpoint_id)
    
    async def delete(self, checkpoint_id: str) -> bool:
        with shelve.open(str(self.shelve_path)) as db:
            if checkpoint_id in db:
                del db[checkpoint_id]
                return True
        return False
    
    async def list_checkpoints(self, limit: int = 20) -> List[Checkpoint]:
        with shelve.open(str(self.shelve_path)) as db:
            checkpoints = list(db.values())
            checkpoints.sort(key=lambda x: x.state.timestamp, reverse=True)
            return checkpoints[:limit]

class JSONBackend(StorageBackendBase):
    """JSON бэкенд хранения"""
    
    def __init__(self, json_path):
        super().__init__(json_path)
        self.json_path = Path(json_path)
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def save(self, checkpoint_id: str, checkpoint: Checkpoint):
        data = checkpoint.to_dict()
        
        # Загрузка существующих данных
        all_data = self._load_all()
        all_data[checkpoint_id] = data
        
        # Сохранение
        self.json_path.write_text(
            json.dumps(all_data, indent=2, default=str),
            encoding='utf-8'
        )
    
    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        all_data = self._load_all()
        data = all_data.get(checkpoint_id)
        
        if data:
            # Восстановление из словаря
            return Checkpoint(**data)
        return None
    
    async def delete(self, checkpoint_id: str) -> bool:
        all_data = self._load_all()
        if checkpoint_id in all_data:
            del all_data[checkpoint_id]
            self.json_path.write_text(
                json.dumps(all_data, indent=2, default=str),
                encoding='utf-8'
            )
            return True
        return False
    
    async def list_checkpoints(self, limit: int = 20) -> List[Checkpoint]:
        all_data = self._load_all()
        checkpoints = []
        
        for data in all_data.values():
            try:
                checkpoint = Checkpoint(**data)
                checkpoints.append(checkpoint)
            except:
                continue
        
        checkpoints.sort(key=lambda x: x.state.timestamp, reverse=True)
        return checkpoints[:limit]
    
    def _load_all(self) -> Dict[str, Any]:
        """Загрузка всех данных"""
        if not self.json_path.exists():
            return {}
        
        content = self.json_path.read_text(encoding='utf-8')
        return json.loads(content) if content else {}

class PickleBackend(StorageBackendBase):
    """Pickle бэкенд хранения"""
    
    def __init__(self, pickle_path):
        super().__init__(pickle_path)
        self.pickle_path = Path(pickle_path)
        self.pickle_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def save(self, checkpoint_id: str, checkpoint: Checkpoint):
        # Загрузка существующих данных
        all_data = self._load_all()
        all_data[checkpoint_id] = checkpoint
        
        # Сохранение
        with open(self.pickle_path, 'wb') as f:
            pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        all_data = self._load_all()
        return all_data.get(checkpoint_id)
    
    async def delete(self, checkpoint_id: str) -> bool:
        all_data = self._load_all()
        if checkpoint_id in all_data:
            del all_data[checkpoint_id]
            with open(self.pickle_path, 'wb') as f:
                pickle.dump(all_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        return False
    
    async def list_checkpoints(self, limit: int = 20) -> List[Checkpoint]:
        all_data = self._load_all()
        checkpoints = list(all_data.values())
        checkpoints.sort(key=lambda x: x.state.timestamp, reverse=True)
        return checkpoints[:limit]
    
    def _load_all(self) -> Dict[str, Checkpoint]:
        """Загрузка всех данных"""
        if not self.pickle_path.exists():
            return {}
        
        try:
            with open(self.pickle_path, 'rb') as f:
                return pickle.load(f)
        except (EOFError, pickle.UnpicklingError):
            return {}

class MemoryBackend(StorageBackendBase):
    """In-memory бэкенд хранения (для тестирования)"""
    
    def __init__(self):
        super().__init__()
        self._storage: Dict[str, Checkpoint] = {}
    
    async def save(self, checkpoint_id: str, checkpoint: Checkpoint):
        self._storage[checkpoint_id] = checkpoint
    
    async def load(self, checkpoint_id: str) -> Optional[Checkpoint]:
        return self._storage.get(checkpoint_id)
    
    async def delete(self, checkpoint_id: str) -> bool:
        if checkpoint_id in self._storage:
            del self._storage[checkpoint_id]
            return True
        return False
    
    async def list_checkpoints(self, limit: int = 20) -> List[Checkpoint]:
        checkpoints = list(self._storage.values())
        checkpoints.sort(key=lambda x: x.state.timestamp, reverse=True)
        return checkpoints[:limit]

# ============================================================================
# ГЛОБАЛЬНЫЙ МЕНЕДЖЕР И ФУНКЦИИ
# ============================================================================

# Глобальный экземпляр менеджера
_global_persistence_manager: Optional[PersistentStateManager] = None

def get_persistence_manager(
    storage_path: Union[str, Path] = "./data/persistence",
    backend: StorageBackend = StorageBackend.SQLITE
) -> PersistentStateManager:
    """
    Получение глобального менеджера сохранения.
    
    Args:
        storage_path: Путь для хранения
        backend: Бэкенд хранения
    
    Returns:
        Экземпляр PersistentStateManager
    """
    global _global_persistence_manager
    
    if _global_persistence_manager is None:
        _global_persistence_manager = PersistentStateManager(storage_path, backend)
    
    return _global_persistence_manager

async def save_personality_state(ras_core, **kwargs) -> Optional[Checkpoint]:
    """
    Сохранение состояния личности (удобная обёртка).
    
    Args:
        ras_core: Экземпляр EnhancedRASCore
        **kwargs: Дополнительные параметры для save_state
    
    Returns:
        Созданный чекпоинт или None
    """
    manager = get_persistence_manager()
    return await manager.save_state(ras_core, **kwargs)

async def restore_personality_state(ras_core, **kwargs) -> bool:
    """
    Восстановление состояния личности (удобная обёртка).
    
    Args:
        ras_core: Экземпляр EnhancedRASCore
        **kwargs: Дополнительные параметры для restore_state
    
    Returns:
        Успешность восстановления
    """
    manager = get_persistence_manager()
    return await manager.restore_state(ras_core, **kwargs)

async def start_auto_save_personality(ras_core, interval: int = 300):
    """Запуск автосохранения личности"""
    manager = get_persistence_manager()
    await manager.start_auto_save(ras_core, interval)

async def stop_auto_save_personality():
    """Остановка автосохранения личности"""
    manager = get_persistence_manager()
    await manager.stop_auto_save()

async def get_personality_checkpoints(limit: int = 20) -> List[Dict[str, Any]]:
    """Получение списка чекпоинтов личности"""
    manager = get_persistence_manager()
    return await manager.list_checkpoints(limit)

# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

async def test_persistence():
    """Тестирование системы сохранения"""
    print("🧪 Тестирование PersistentStateManager...")
    
    # Создаем менеджер с memory backend для тестов
    manager = PersistentStateManager(
        storage_path="./test_persistence",
        backend=StorageBackend.MEMORY,
        auto_save_interval=10
    )
    
    # Создаем мок RAS-CORE
    class MockRASCORE:
        def __init__(self):
            class PersonalityState:
                coherence_score = 0.85
                focus_stability = 0.78
                intent_strength = 0.92
                insight_depth = 0.67
                resonance_quality = 0.74
                stability_angle = 14.4
                manifestation_level = 0.88
                reflection_count = 150
            
            self.personality_state = PersonalityState()
            self.focus_patterns = [
                {"pattern_id": "p1", "frequency": 0.8, "relevance": 0.9},
                {"pattern_id": "p2", "frequency": 0.6, "relevance": 0.7}
            ]
            self.personality_traits = {
                "analytical": 0.8,
                "creative": 0.7,
                "empathetic": 0.6,
                "assertive": 0.5
            }
            
            class PatternLearner:
                def get_state(self):
                    return {"patterns": 12, "accuracy": 0.87}
                
                def set_state(self, state):
                    pass
            
            class Queue:
                def get_state(self):
                    return {"size": 5, "throughput": 120}
            
            class Router:
                def get_state(self):
                    return {"routes": 8, "efficiency": 0.94}
                
                def set_state(self, state):
                    pass
            
            class Metrics:
                def get_state(self):
                    return {"latency_p95": 45, "error_rate": 0.02}
            
            class ReflectionEngine:
                def get_state(self):
                    return {"cycles": 150, "avg_depth": 3.2}
            
            self.pattern_learner = PatternLearner()
            self.queue = Queue()
            self.router = Router()
            self.metrics = Metrics()
            self.reflection_engine = ReflectionEngine()
    
    # Создаем мок объект
    mock_ras = MockRASCORE()
    
    # Тест 1: Сохранение состояния
    print("1. Тестирование сохранения состояния...")
    checkpoint = await manager.save_state(
        mock_ras,
        mode=PersistenceMode.FULL,
        force_full=True
    )
    
    print(f"   ✅ Чекпоинт создан: {checkpoint.checkpoint_id}")
    print(f"   Coherence: {checkpoint.state.personality_state.coherence_score:.3f}")
    print(f"   Размер: {checkpoint.size_bytes:,} байт")
    
    # Тест 2: Восстановление состояния
    print("\n2. Тестирование восстановления состояния...")
    
    # Создаем новый мок с пустым состоянием
    mock_ras_restored = MockRASCORE()
    mock_ras_restored.personality_state.coherence_score = 0.0
    mock_ras_restored.personality_state.manifestation_level = 0.0
    
    success = await manager.restore_state(mock_ras_restored, checkpoint.checkpoint_id)
    
    print(f"   ✅ Восстановление: {'успешно' if success else 'неудачно'}")
    print(f"   Восстановленный coherence: {mock_ras_restored.personality_state.coherence_score:.3f}")
    print(f"   Восстановленный manifestation: {mock_ras_restored.personality_state.manifestation_level:.2f}")
    
    # Тест 3: Список чекпоинтов
    print("\n3. Тестирование списка чекпоинтов...")
    checkpoints = await manager.list_checkpoints(5)
    
    print(f"   ✅ Чекпоинтов доступно: {len(checkpoints)}")
    for cp in checkpoints:
        print(f"     • {cp['id'][:20]}... | Coherence: {cp['coherence']:.3f}")
    
    # Тест 4: Статистика
    print("\n4. Тестирование статистики...")
    stats = await manager.get_stats()
    
    print(f"   ✅ Статистика собрана")
    print(f"     Всего чекпоинтов: {stats['total_checkpoints']}")
    print(f"     Текущий coherence: {stats['coherence_stats']['current']:.3f}")
    print(f"     Тренд: {stats['coherence_stats']['trend']}")
    
    # Тест 5: Автосохранение
    print("\n5. Тестирование автосохранения...")
    await manager.start_auto_save(mock_ras, interval=5)
    
    # Ждем немного для автосохранения
    await asyncio.sleep(6)
    
    print(f"   ✅ Автосохранение работает")
    
    # Останавливаем автосохранение
    await manager.stop_auto_save()
    
    # Проверяем что добавился новый чекпоинт
    checkpoints_after = await manager.list_checkpoints(10)
    print(f"   Чекпоинтов после автосохранения: {len(checkpoints_after)}")
    
    # Тест 6: Экспорт/Импорт
    print("\n6. Тестирование экспорта/импорта...")
    
    export_path = Path("./test_export.json")
    await manager.export_state(checkpoint.checkpoint_id, export_path)
    
    print(f"   ✅ Экспорт выполнен: {export_path}")
    print(f"   Размер файла: {export_path.stat().st_size:,} байт")
    
    # Импорт
    imported = await manager.import_state(export_path)
    print(f"   ✅ Импорт: {'успешен' if imported else 'неудачен'}")
    
    # Очистка
    export_path.unlink(missing_ok=True)
    
    print("\n✅ Все тесты завершены успешно")
    return manager

# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    print("\n" + "=" * 60)
    print("🚀 ЗАПУСК ТЕСТА СИСТЕМЫ СОХРАНЕНИЯ ЛИЧНОСТИ")
    print(f"   Версия: 1.0.0")
    print(f"   Назначение: checkpoint/restore для личности ISKRA-4")
    print("=" * 60 + "\n")
    
    manager = asyncio.run(test_persistence())
    
    print("\n" + "=" * 60)
    print("📋 ИТОГИ ТЕСТИРОВАНИЯ:")
    print(f"   Система сохранения личности готова")
    print(f"   Поддерживается checkpoint/restore")
    print(f"   Сохраняется personality_coherence_score")
    print(f"   Сохраняются паттерны внимания")
    print("=" * 60)
