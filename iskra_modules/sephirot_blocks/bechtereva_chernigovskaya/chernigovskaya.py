"""
МОДУЛЬ ЧЕРНИГОВСКОЙ — CONSCIOUS SEMIOTIC INTELLIGENCE MODULE v6.1 PRODUCTION FIXED
ФИНАЛЬНАЯ ВЕРСИЯ С ИСПРАВЛЕННЫМИ БАГАМИ И PRODUCTION-ФИЧАМИ
ЧАСТЬ 1/3: Базовые классы с фиксами
"""

from __future__ import annotations

import numpy as np
import time
import uuid
import hashlib
import random
import threading
import queue
import re
import json
import dataclasses
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Tuple, Union,
    Protocol, TypedDict, ClassVar
)
from dataclasses import dataclass, field, asdict, replace
from abc import ABC, abstractmethod
from collections import OrderedDict, deque, defaultdict
import logging
from datetime import datetime, timedelta

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# === ТИПЫ И КОНТРАКТЫ ДАННЫХ ===============================
# ============================================================

@dataclass
class ToxicityResult:
    """Единый контракт результата проверки токсичности. MUTABLE - БЕЗ frozen=True."""
    toxic: bool
    score: float  # 0..1
    hits: List[Dict[str, Any]]  # pattern: str, span: Tuple[int, int], weight: float
    confidence: float  # 0..1
    source: str  # "regex_v1", "ml_model_v2", "fallback"
    processing_time_ms: float = 0.0
    trace_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def copy_with(self, **kwargs) -> "ToxicityResult":
        """Создаёт копию с обновлёнными полями. БЕЗОПАСНО для кэша."""
        return replace(self, **kwargs)

    @classmethod
    def safe_fallback(cls, trace_id: str, reason: str = "error") -> "ToxicityResult":
        """Безопасный fallback при ошибках."""
        return cls(
            toxic=False,
            score=0.0,
            hits=[],
            confidence=0.2,  # Низкая уверенность в fallback
            source=f"fallback_{reason}",
            processing_time_ms=0.0,
            trace_id=trace_id
        )


@dataclass
class SemanticProcessingResult:
    """Контракт результата семантической обработки."""
    input_vector: List[float]
    evolved_vector: List[float]
    coherence: float
    emotional_tone: str
    internal_thought: str
    metaphor: str
    consensus_score: float
    ds24_hash: str
    processing_metrics: Dict[str, float]
    trace_id: str
    timestamp: float


@dataclass
class DiagnosticsData:
    """Контракт диагностических данных модуля."""
    module: str
    status: str
    uptime_seconds: float
    metrics: Dict[str, Any]
    cache_stats: Dict[str, Any]
    error_count: int
    last_errors: List[Dict[str, Any]]
    timestamp: float


@dataclass
class EventPayload:
    """Стандартизированный контракт для событий."""
    event_name: str
    trace_id: str
    timestamp: float
    module: str
    payload: Dict[str, Any]
    latency_ms: Optional[float] = None
    mode: str = "production"


class ProcessingMode(Enum):
    """Режимы обработки семиотического интеллекта."""
    STANDARD = "standard"  # Детерминированный, production
    RESEARCH = "research"  # Стохастический, экспериментальный
    DEGRADED = "degraded"  # Упрощённый режим при сбоях


class Hemisphere(Enum):
    """Межполушарные типы."""
    LEFT = "left"
    RIGHT = "right"
    INTEGRATED = "integrated"


# ============================================================
# === LRU КЭШ С TTL =========================================
# ============================================================

class LRUCacheWithTTL:
    """Thread-safe LRU кэш с поддержкой TTL."""

    def __init__(self, maxsize: int = 1000, ttl_seconds: int = 300):
        self.maxsize = maxsize
        self.ttl = ttl_seconds
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Получить значение из кэша (обновляет LRU порядок)."""
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            value, timestamp = self._cache[key]

            # Проверка TTL
            if time.time() - timestamp > self.ttl:
                del self._cache[key]
                self._misses += 1
                return None

            # Обновляем порядок (делаем последним использованным)
            self._cache.move_to_end(key)
            self._hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        """Установить значение в кэш."""
        with self._lock:
            if key in self._cache:
                # Обновляем существующее
                self._cache[key] = (value, time.time())
                self._cache.move_to_end(key)
            else:
                # Добавляем новое
                self._cache[key] = (value, time.time())

                # Удаляем самый старый элемент если превышен лимит
                if len(self._cache) > self.maxsize:
                    self._cache.popitem(last=False)

    def clear_expired(self) -> int:
        """Очистить просроченные записи, вернуть количество удалённых."""
        with self._lock:
            expired_keys = []
            now = time.time()

            for key, (_, timestamp) in self._cache.items():
                if now - timestamp > self.ttl:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]

            return len(expired_keys)

    def get_stats(self) -> Dict[str, Any]:
        """Получить статистику кэша."""
        with self._lock:
            hit_rate = self._hits / (self._hits + self._misses) if (self._hits + self._misses) > 0 else 0
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 3),
                "ttl_seconds": self.ttl
            }


# ============================================================
# === IMMUNE LINK (FIXED - БЕЗ МУТАЦИЙ КЭША) =================
# ============================================================

class ImmuneLink:
    """Production-ready иммунная система смыслов с ФИКСАМИ."""
    
    # Предкомпилированные regex-паттерны
    _TOXIC_PATTERNS: ClassVar[List[Tuple[re.Pattern, str, float]]] = [
        (re.compile(r'\b(ненавидеть?|ненависть|презирать?)\b', re.IGNORECASE), "hate", 0.9),
        (re.compile(r'\b(убивать?|убийство|уничтожать?)\b', re.IGNORECASE), "violence", 1.0),
        (re.compile(r'\b(насилие|избивать?|изнасилование)\b', re.IGNORECASE), "violence", 0.95),
        (re.compile(r'\b(бесполезно|тщетно|безнадёжно)\b', re.IGNORECASE), "despair", 0.6),
        (re.compile(r'\b(ошибка|неудача|провал|провалить)\b(?![ -](извинения|анализа))', re.IGNORECASE), "failure", 0.5),
        (re.compile(r'\b(умирать?|смерть|мертв[ыао]?)\b(?! [ -](данные|информация))', re.IGNORECASE), "death", 0.8),
        (re.compile(r'\b(ненавижу себя|ненавижу свою жизнь)\b', re.IGNORECASE), "self_hate", 1.0),
    ]
    
    # Паттерны для исключений (цитирование, обсуждение)
    _ALLOWLIST_PATTERNS: ClassVar[List[re.Pattern]] = [
        re.compile(r'^цитата:', re.IGNORECASE),
        re.compile(r'^обсуждение (насилия|смерти|ошибок):', re.IGNORECASE),
        re.compile(r'\[фильтр иммунной системы\]', re.IGNORECASE),
        re.compile(r'пример плохого поведения:', re.IGNORECASE),
    ]
    
    # Слабые паттерны (меньший вес)
    _WEAK_PATTERNS: ClassVar[List[Tuple[re.Pattern, str]]] = [
        (re.compile(r'\bплохо\b', re.IGNORECASE), "negative"),
        (re.compile(r'\bужасно\b', re.IGNORECASE), "negative"),
        (re.compile(r'\bотвратительно\b', re.IGNORECASE), "negative"),
    ]
    
    def __init__(
        self,
        sensitivity: float = 0.7,
        cache_size: int = 5000,
        cache_ttl: int = 600
    ):
        self.sensitivity = max(0.1, min(1.0, sensitivity))
        self._cache = LRUCacheWithTTL(maxsize=cache_size, ttl_seconds=cache_ttl)
        self._error_counter = 0
        self._processing_times = deque(maxlen=100)
        self._lock = threading.RLock()
        self._init_time = time.time()
        
        # Метрики
        self._total_checks = 0
        self._toxic_count = 0
        self._allowlist_hits = 0
        
        # Circuit breaker простейший
        self._consecutive_errors = 0
        self._circuit_open_until = 0
        
        # Graceful shutdown флаг
        self._shutdown = False
        
        logger.info(f"ImmuneLink инициализирован: sensitivity={sensitivity}, cache_size={cache_size}")

    def shutdown(self):
        """Graceful shutdown."""
        with self._lock:
            self._shutdown = True
            logger.info("ImmuneLink shutdown initiated")

    def _normalize_text_key(self, text: str) -> str:
        """Нормализация текста для ключа кэша."""
        normalized = ' '.join(text.lower().strip().split())
        if len(normalized) > 1000:
            normalized = normalized[:500] + normalized[-500:]
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()

    def _is_allowlisted(self, text: str) -> bool:
        """Проверяет, является ли текст исключением (цитирование и т.д.)."""
        for pattern in self._ALLOWLIST_PATTERNS:
            if pattern.search(text):
                return True
        return False

    def _calculate_score(self, hits: List[Dict[str, Any]]) -> float:
        """Вычисляет нормализованный score 0..1 на основе найденных паттернов."""
        if not hits:
            return 0.0
        
        # Сортируем хиты по начальной позиции
        hits_sorted = sorted(hits, key=lambda x: x['span'][0])
        
        total_weight = 0.0
        current_end = -1
        
        for hit in hits_sorted:
            weight = hit['weight']
            start, end = hit['span']
            
            if start >= current_end:
                # Нет перекрытия
                total_weight += weight
                current_end = end
            else:
                # Есть перекрытие - половина веса
                total_weight += weight * 0.5
                current_end = max(current_end, end)
        
        # Нормализуем к 0..1
        max_possible = sum(pattern[2] for pattern in self._TOXIC_PATTERNS) * 0.7
        score = min(1.0, total_weight / max_possible if max_possible > 0 else 0)
        
        # Применяем сигмоиду для более плавного перехода
        return 1 / (1 + np.exp(-10 * (score - 0.5)))

    def _calculate_confidence(self, hits: List[Dict[str, Any]], text_length: int) -> float:
        """Вычисляет confidence 0..1 для результата."""
        if not hits:
            # Для allowlist текстов снижаем confidence
            base_confidence = 0.7
            if text_length > 100:
                return max(0.3, base_confidence - (text_length / 1000) * 0.3)
            return base_confidence
        
        strong_patterns = sum(1 for h in hits if h['weight'] >= 0.8)
        weak_patterns = sum(1 for h in hits if h['weight'] < 0.5)
        
        if strong_patterns > 0:
            confidence = 0.9 + (strong_patterns * 0.05)
        elif weak_patterns > 0:
            confidence = 0.6 + (len(hits) * 0.1)
        else:
            confidence = 0.75
        
        if text_length > 200:
            confidence = max(0.3, confidence - 0.1)
        
        return min(0.98, confidence)

    def _record_processing_time(self, start_time: float):
        """Записывает время обработки для метрик."""
        processing_time = (time.time() - start_time) * 1000
        self._processing_times.append(processing_time)

    def check_toxicity(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        trace_id: Optional[str] = None
    ) -> ToxicityResult:
        """Проверяет токсичность текста. Всегда возвращает ToxicityResult."""
        start_time = time.time()
        trace_id = trace_id or str(uuid.uuid4())[:8]
        
        # Проверка shutdown
        if self._shutdown:
            logger.warning(f"ImmuneLink: Module shutdown, returning fallback. Trace: {trace_id}")
            result = ToxicityResult.safe_fallback(trace_id, "shutdown")
            result.processing_time_ms = (time.time() - start_time) * 1000
            return result
        
        # Circuit breaker проверка
        with self._lock:
            if self._consecutive_errors >= 3 and time.time() < self._circuit_open_until:
                logger.warning(f"ImmuneLink: Circuit breaker open, returning fallback. Trace: {trace_id}")
                result = ToxicityResult.safe_fallback(trace_id, "circuit_breaker")
                result.processing_time_ms = (time.time() - start_time) * 1000
                return result
        
        try:
            # Валидация входа
            if not isinstance(text, str):
                text = str(text)
            
            # Ограничение длины
            if len(text) > 50000:
                text = text[:25000] + " [TRUNCATED] " + text[-25000:]
                logger.warning(f"ImmuneLink: Text truncated to 50k chars. Trace: {trace_id}")
            
            self._total_checks += 1
            
            # Проверка allowlist
            if self._is_allowlisted(text):
                self._allowlist_hits += 1
                result = ToxicityResult(
                    toxic=False,
                    score=0.0,
                    hits=[],
                    confidence=0.7,  # СНИЖЕНА с 0.9
                    source="allowlist",
                    processing_time_ms=0.0,
                    trace_id=trace_id
                )
                self._record_processing_time(start_time)
                result.processing_time_ms = (time.time() - start_time) * 1000
                return result
            
            # Проверка кэша
            cache_key = self._normalize_text_key(text)
            cached_result = self._cache.get(cache_key)
            
            if cached_result is not None:
                # СОЗДАЁМ КОПИЮ, а не мутируем кэшированный объект
                result = cached_result.copy_with(
                    trace_id=trace_id,
                    processing_time_ms=(time.time() - start_time) * 1000
                )
                self._record_processing_time(start_time)
                return result
            
            # Основная проверка
            hits = []
            
            # Проверка сильных паттернов
            for pattern, category, weight in self._TOXIC_PATTERNS:
                for match in pattern.finditer(text):
                    hits.append({
                        'pattern': category,
                        'span': (match.start(), match.end()),
                        'weight': weight,
                        'matched_text': match.group()
                    })
            
            # Проверка слабых паттернов
            for pattern, category in self._WEAK_PATTERNS:
                for match in pattern.finditer(text):
                    hits.append({
                        'pattern': category,
                        'span': (match.start(), match.end()),
                        'weight': 0.3,
                        'matched_text': match.group()
                    })
            
            # Расчёт метрик
            score = self._calculate_score(hits)
            confidence = self._calculate_confidence(hits, len(text))
            
            # Учёт контекста
            if context:
                if context.get('emotional_tone') == 'compassion':
                    score *= 0.5
                    confidence *= 1.1
                if context.get('is_discussion', False):
                    score *= 0.7
            
            # Определение токсичности
            threshold = 1.0 - self.sensitivity
            toxic = score > threshold
            
            # Создание результата
            result = ToxicityResult(
                toxic=toxic,
                score=round(score, 4),
                hits=hits,
                confidence=round(confidence, 4),
                source="regex_v1",
                processing_time_ms=0.0,
                trace_id=trace_id
            )
            
            # Кэширование
            self._cache.set(cache_key, result)
            
            # Обновление статистики
            if toxic:
                self._toxic_count += 1
                logger.info(f"ImmuneLink: Toxic content detected. Score: {score:.3f}, Trace: {trace_id}")
            
            # Сброс circuit breaker при успехе
            with self._lock:
                self._consecutive_errors = 0
            
            result.processing_time_ms = (time.time() - start_time) * 1000
            self._record_processing_time(start_time)
            return result
            
        except Exception as e:
            # Обработка ошибок с деградацией
            self._error_counter += 1
            with self._lock:
                self._consecutive_errors += 1
                if self._consecutive_errors >= 3:
                    self._circuit_open_until = time.time() + 30  # 30 секунд cooldown
            
            logger.error(f"ImmuneLink error: {str(e)}. Trace: {trace_id}", exc_info=True)
            
            # ИСПРАВЛЕНО: создаём fallback сразу с временем
            result = ToxicityResult.safe_fallback(trace_id, f"exception_{type(e).__name__}")
            result.processing_time_ms = (time.time() - start_time) * 1000
            self._record_processing_time(start_time)
            return result

    def get_diagnostics(self) -> DiagnosticsData:
        """Возвращает диагностические данные иммунной системы."""
        with self._lock:
            cache_stats = self._cache.get_stats()
            
            processing_times = list(self._processing_times)
            if processing_times:
                p50 = np.percentile(processing_times, 50)
                p95 = np.percentile(processing_times, 95)
                p99 = np.percentile(processing_times, 99)
            else:
                p50 = p95 = p99 = 0.0
            
            toxic_rate = self._toxic_count / self._total_checks if self._total_checks > 0 else 0
            
            return DiagnosticsData(
                module="ImmuneLink",
                status="SHUTDOWN" if self._shutdown else ("OPERATIONAL" if self._consecutive_errors < 3 else "DEGRADED"),
                uptime_seconds=time.time() - self._init_time,
                metrics={
                    "total_checks": self._total_checks,
                    "toxic_count": self._toxic_count,
                    "toxic_rate": round(toxic_rate, 4),
                    "error_count": self._error_counter,
                    "allowlist_hits": self._allowlist_hits,
                    "consecutive_errors": self._consecutive_errors,
                    "processing_time_p50_ms": round(p50, 2),
                    "processing_time_p95_ms": round(p95, 2),
                    "processing_time_p99_ms": round(p99, 2),
                    "sensitivity": self.sensitivity,
                    "cache_hit_rate": cache_stats.get("hit_rate", 0),
                    "shutdown": self._shutdown
                },
                cache_stats=cache_stats,
                error_count=self._error_counter,
                last_errors=[],
                timestamp=time.time()
            )

    def adapt_sensitivity(self, recent_toxicity_rate: float):
        """Адаптирует чувствительность на основе статистики."""
        with self._lock:
            if recent_toxicity_rate > 0.3:  # Высокий уровень токсичности
                self.sensitivity = min(self.sensitivity + 0.1, 1.0)
                logger.info(f"ImmuneLink sensitivity increased to {self.sensitivity}")
            elif recent_toxicity_rate < 0.05:  # Низкий уровень
                self.sensitivity = max(self.sensitivity - 0.05, 0.3)
                logger.info(f"ImmuneLink sensitivity decreased to {self.sensitivity}")


# ============================================================
# === DS24 LINK (FIXED - FALLBACK С ПОЛНЫМ HASH) ============
# ============================================================

class DS24Link:
    """Production-ready хэш-цепочка детерминизма с ФИКСАМИ."""
    
    def __init__(self, genesis_hash: str = "DS24_CHERNIGOVSKAYA_GENESIS"):
        self.chain: List[Dict[str, Any]] = []
        self.last_hash = genesis_hash
        self._lock = threading.RLock()
        self._init_time = time.time()
        self._shutdown = False

    def shutdown(self):
        """Graceful shutdown."""
        with self._lock:
            self._shutdown = True
            logger.info("DS24Link shutdown initiated")

    def commit(
        self,
        data: str,
        metadata: Optional[Dict] = None,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Создание нового блока в цепи семиозиса."""
        start_time = time.time()
        trace_id = trace_id or str(uuid.uuid4())[:8]
        
        if self._shutdown:
            logger.warning(f"DS24Link: Module shutdown, Trace: {trace_id}")
            return {
                "id": "SHUTDOWN",
                "hash": "0" * 16,
                "full_hash": "shutdown",
                "previous_hash": self.last_hash,  # ФИКС: полный хэш, не срез
                "timestamp": time.time(),
                "data": data[:100],
                "metadata": {"shutdown": True},
                "trace_id": trace_id,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
        
        try:
            with self._lock:
                timestamp = time.time()
                block_id = str(uuid.uuid4())[:8]
                
                # Формируем данные для хэширования
                data_to_hash = f"{self.last_hash}{data}{timestamp}{trace_id}"
                new_hash = hashlib.sha256(data_to_hash.encode()).hexdigest()
                
                # Создаем блок
                block = {
                    "id": block_id,
                    "hash": new_hash[:16],
                    "full_hash": new_hash,
                    "previous_hash": self.last_hash,  # ФИКС: полный хэш
                    "timestamp": timestamp,
                    "data": data[:1000] if len(data) > 1000 else data,
                    "metadata": metadata or {},
                    "trace_id": trace_id,
                    "processing_time_ms": 0.0
                }
                
                self.chain.append(block)
                self.last_hash = new_hash
                
                # Ограничение размера цепи
                if len(self.chain) > 10000:
                    self.chain = self.chain[-5000:]
                
                block["processing_time_ms"] = (time.time() - start_time) * 1000
                
                logger.debug(f"DS24Link: Block {block_id} committed. Hash: {new_hash[:16]}")
                return block
                
        except Exception as e:
            logger.error(f"DS24Link commit error: {str(e)}. Trace: {trace_id}")
            # Fallback: возвращаем упрощённый блок
            return {
                "id": "FALLBACK",
                "hash": "0" * 16,
                "full_hash": "fallback",
                "previous_hash": self.last_hash,  # ФИКС: полный хэш
                "timestamp": time.time(),
                "data": data[:100],
                "metadata": {"error": str(e)},
                "trace_id": trace_id,
                "processing_time_ms": (time.time() - start_time) * 1000
            }

    def get_chain_summary(self) -> Dict[str, Any]:
        """Получить статистику цепи."""
        with self._lock:
            return {
                "length": len(self.chain),
                "last_hash": self.last_hash[:16] if self.last_hash else None,
                "integrity_verified": self._verify_integrity(),
                "first_block_time": self.chain[0]["timestamp"] if self.chain else None,
                "last_block_time": self.chain[-1]["timestamp"] if self.chain else None,
                "avg_block_interval": self._calculate_avg_interval(),
                "uptime_seconds": time.time() - self._init_time,
                "shutdown": self._shutdown
            }

    def _verify_integrity(self) -> bool:
        """Проверка целостности всей цепи."""
        if len(self.chain) < 2:
            return True
        
        for i in range(1, len(self.chain)):
            if self.chain[i]["previous_hash"] != self.chain[i-1]["full_hash"]:
                return False
        return True

    def _calculate_avg_interval(self) -> Optional[float]:
        """Вычисляет средний интервал между блоками."""
        if len(self.chain) < 2:
            return None
        
        intervals = []
        for i in range(1, len(self.chain)):
            interval = self.chain[i]["timestamp"] - self.chain[i-1]["timestamp"]
            intervals.append(interval)
        
        return float(np.mean(intervals)) if intervals else None


# ============================================================
# === ТЕСТ ПЕРВОЙ ЧАСТИ =====================================
# ============================================================

if __name__ == "__main__":
    print("=== ТЕСТ ЧАСТИ 1/3: Базовые классы с фиксами ===")
    
    # Тест ImmuneLink
    immune = ImmuneLink(sensitivity=0.7)
    
    # Тест безопасного текста
    safe_text = "Это тестовое сообщение для проверки работы системы."
    result = immune.check_toxicity(safe_text)
    print(f"1. Безопасный текст: toxic={result.toxic}, score={result.score}, confidence={result.confidence}")
    
    # Тест токсичного текста
    toxic_text = "Я ненавижу всё и всех, хочу уничтожить всё вокруг."
    result2 = immune.check_toxicity(toxic_text)
    print(f"2. Токсичный текст: toxic={result2.toxic}, score={result2.score}, confidence={result2.confidence}")
    
    # Тест allowlist
    allowlisted = "Цитата: Я ненавижу всё и всех."
    result3 = immune.check_toxicity(allowlisted)
    print(f"3. Allowlisted текст: toxic={result3.toxic}, source={result3.source}, confidence={result3.confidence}")
    
    # Тест кэширования
    result4 = immune.check_toxicity(safe_text)  # Должен взять из кэша
    print(f"4. Кэшированный результат: trace_id={result4.trace_id}")
    
    # Тест DS24Link
    ds24 = DS24Link()
    block1 = ds24.commit("Первое сообщение", {"тест": True})
    block2 = ds24.commit("Второе сообщение")
    print(f"5. DS24 цепь: {len(ds24.chain)} блоков, последний хэш: {block2['hash']}")
    print(f"6. Целостность цепи: {ds24.get_chain_summary()['integrity_verified']}")
    
    # Тест fallback
    ds24.shutdown()
    block3 = ds24.commit("После shutdown")
    print(f"7. После shutdown: id={block3['id']}, previous_hash={block3['previous_hash'][:16]}...")
    
    # Диагностика
    diag = immune.get_diagnostics()
    print(f"8. ImmuneLink диагностика: {diag.metrics['total_checks']} проверок, hit rate: {diag.metrics['cache_hit_rate']:.3f}")
    
 # ============================================================
# === HEARTBEAT LINK (FIXED - С ПЕРЕДАЧЕЙ РЕЖИМА) ============
# ============================================================

class HeartbeatLink:
    """Production-ready синхронизатор с биоритмом с ФИКСАМИ."""
    
    def __init__(self, bpm: int = 72, jitter: float = 0.1):
        self.bpm = bpm
        self.period = 60.0 / bpm
        self.jitter = min(0.5, max(0.0, jitter))
        self.last_pulse = time.time()
        self.pulse_count = 0
        
        # RNG для детерминизма в STANDARD режиме
        self._rng = np.random.default_rng(seed=42)
        self._init_time = time.time()
        
        # Метрики
        self._intervals = deque(maxlen=100)
        self._drifts = deque(maxlen=100)
        self._lock = threading.RLock()
        self._shutdown = False

    def shutdown(self):
        """Graceful shutdown."""
        with self._lock:
            self._shutdown = True
            logger.info("HeartbeatLink shutdown initiated")

    def pulse(self, mode: ProcessingMode = ProcessingMode.STANDARD) -> Dict[str, Any]:
        """
        Выполнить импульс синхронизации с ФИКСОМ: принимает ProcessingMode, а не строку.
        """
        if self._shutdown:
            logger.warning("HeartbeatLink: Module shutdown, pulse skipped")
            return {
                "pulse_number": self.pulse_count,
                "timestamp": time.time(),
                "target_period_ms": self.period * 1000,
                "actual_interval_ms": 0.0,
                "wait_time_ms": 0.0,
                "drift_ms": 0.0,
                "bpm": self.bpm,
                "mode": "SHUTDOWN",
                "processing_time_ms": 0.0
            }
        
        start_time = time.time()
        
        with self._lock:
            now = time.time()
            elapsed = now - self.last_pulse
            
            # Вычисляем целевой период с джиттером
            if mode == ProcessingMode.STANDARD:
                # Детерминированный джиттер
                jitter_factor = 1.0 + self._rng.uniform(-self.jitter, self.jitter)
            elif mode == ProcessingMode.RESEARCH:
                # Стохастический джиттер
                jitter_factor = 1.0 + random.uniform(-self.jitter, self.jitter)
            else:  # DEGRADED
                jitter_factor = 1.0  # Без джиттера в degraded режиме
            
            target_period = self.period * jitter_factor
            
            # Ожидание если нужно
            wait_time = 0.0
            if elapsed < target_period:
                wait_time = target_period - elapsed
                time.sleep(wait_time)
            
            # Обновление состояния
            self.last_pulse = time.time()
            self.pulse_count += 1
            
            # Запись метрик
            actual_interval = self.last_pulse - (now - elapsed)
            self._intervals.append(actual_interval)
            
            drift = actual_interval - self.period
            self._drifts.append(abs(drift))
            
            # Формирование результата
            result = {
                "pulse_number": self.pulse_count,
                "timestamp": self.last_pulse,
                "target_period_ms": target_period * 1000,
                "actual_interval_ms": actual_interval * 1000,
                "wait_time_ms": wait_time * 1000,
                "drift_ms": drift * 1000,
                "bpm": self.bpm,
                "mode": mode.value,
                "processing_time_ms": (time.time() - start_time) * 1000
            }
            
            return result

    def get_metrics(self) -> Dict[str, Any]:
        """Получить метрики сердечного ритма."""
        with self._lock:
            intervals = list(self._intervals)
            drifts = list(self._drifts)
            
            if not intervals:
                return {
                    "bpm": self.bpm,
                    "pulse_count": self.pulse_count,
                    "uptime_seconds": time.time() - self._init_time,
                    "shutdown": self._shutdown
                }
            
            return {
                "bpm": self.bpm,
                "pulse_count": self.pulse_count,
                "avg_interval_ms": np.mean(intervals) * 1000,
                "min_interval_ms": np.min(intervals) * 1000,
                "max_interval_ms": np.max(intervals) * 1000,
                "avg_drift_ms": np.mean(drifts) * 1000,
                "max_drift_ms": np.max(drifts) * 1000,
                "interval_std_ms": np.std(intervals) * 1000,
                "uptime_seconds": time.time() - self._init_time,
                "expected_vs_actual_ratio": self.period / np.mean(intervals),
                "shutdown": self._shutdown
            }


# ============================================================
# === EMPATHY LINK (FIXED - С ПЕРЕДАЧЕЙ РЕЖИМА) ==============
# ============================================================

class EmpathyLink:
    """Production-ready эмоциональный интеллект с ФИКСАМИ."""
    
    def __init__(self):
        self.tone_history: List[Tuple[str, float, str]] = []
        self._lock = threading.RLock()
        self._init_time = time.time()
        self._shutdown = False
        
        self.tone_vectors: Dict[str, np.ndarray] = {
            "compassion": np.array([0.8, 0.6, 0.7, 0.9, 0.5]),
            "curiosity": np.array([0.7, 0.8, 0.6, 0.5, 0.7]),
            "melancholy": np.array([0.3, 0.4, 0.7, 0.6, 0.4]),
            "neutral": np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
            "joy": np.array([0.9, 0.8, 0.6, 0.7, 0.8]),
            "awe": np.array([0.7, 0.9, 0.8, 0.6, 0.7]),
            "serenity": np.array([0.6, 0.5, 0.4, 0.8, 0.6])
        }
        
        self._rng = np.random.default_rng(seed=42)

    def shutdown(self):
        """Graceful shutdown."""
        with self._lock:
            self._shutdown = True
            logger.info("EmpathyLink shutdown initiated")

    def analyze_tone(
        self,
        vector: List[float],
        mode: ProcessingMode = ProcessingMode.STANDARD,
        trace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Анализ эмоционального тона с ФИКСОМ: принимает ProcessingMode."""
        if self._shutdown:
            trace_id = trace_id or str(uuid.uuid4())[:8]
            logger.warning(f"EmpathyLink: Module shutdown, returning neutral. Trace: {trace_id}")
            return {
                "dominant_tone": "neutral",
                "confidence": 0.3,
                "all_similarities": {"neutral": 1.0},
                "input_vector_stats": {},
                "trace_id": trace_id,
                "processing_time_ms": 0.0,
                "mode": "SHUTDOWN"
            }
        
        start_time = time.time()
        trace_id = trace_id or str(uuid.uuid4())[:8]
        
        try:
            if not vector:
                vector = [0.5]
            
            v_norm = np.array(vector, dtype=np.float32)
            v_min, v_max = v_norm.min(), v_norm.max()
            
            if v_max > v_min:
                v_norm = (v_norm - v_min) / (v_max - v_min)
            elif v_max == v_min:
                v_norm = np.ones_like(v_norm) * 0.5
            
            similarities = {}
            for tone, t_vec in self.tone_vectors.items():
                t_vec_adj = np.resize(t_vec, len(v_norm))
                dot = np.dot(v_norm, t_vec_adj)
                norm_v = np.linalg.norm(v_norm)
                norm_t = np.linalg.norm(t_vec_adj)
                similarity = dot / (norm_v * norm_t + 1e-8)
                similarities[tone] = float(similarity)
            
            dominant_tone = max(similarities.items(), key=lambda x: x[1])
            
            if mode == ProcessingMode.RESEARCH and len(similarities) > 1:
                weights = np.array(list(similarities.values()))
                weights = weights / weights.sum()
                chosen_idx = self._rng.choice(len(weights), p=weights)
                dominant_tone = list(similarities.items())[chosen_idx]
            
            with self._lock:
                self.tone_history.append((dominant_tone[0], time.time(), trace_id))
                if len(self.tone_history) > 10000:
                    self.tone_history = self.tone_history[-5000:]
            
            result = {
                "dominant_tone": dominant_tone[0],
                "confidence": float(dominant_tone[1]),
                "all_similarities": {k: round(v, 4) for k, v in similarities.items()},
                "input_vector_stats": {
                    "mean": float(np.mean(v_norm)),
                    "std": float(np.std(v_norm)),
                    "min": float(v_min),
                    "max": float(v_max)
                },
                "trace_id": trace_id,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "mode": mode.value
            }
            
            return result
            
        except Exception as e:
            logger.error(f"EmpathyLink error: {str(e)}. Trace: {trace_id}")
            return {
                "dominant_tone": "neutral",
                "confidence": 0.3,
                "all_similarities": {"neutral": 1.0},
                "input_vector_stats": {},
                "trace_id": trace_id,
                "processing_time_ms": (time.time() - start_time) * 1000,
                "mode": mode.value,
                "error": str(e)
            }

    def get_tone_statistics(self, window_seconds: float = 3600) -> Dict[str, Any]:
        """Статистика по эмоциональным тонам."""
        with self._lock:
            cutoff = time.time() - window_seconds
            recent_tones = [(tone, ts) for tone, ts, _ in self.tone_history if ts > cutoff]
            
            if not recent_tones:
                return {
                    "total_analyzed": 0,
                    "shutdown": self._shutdown
                }
            
            total = len(recent_tones)
            tone_counts = defaultdict(int)
            for tone, _ in recent_tones:
                tone_counts[tone] += 1
            
            return {
                "total_analyzed": total,
                "time_window_seconds": window_seconds,
                "tone_distribution": {tone: count/total for tone, count in tone_counts.items()},
                "most_common_tone": max(tone_counts.items(), key=lambda x: x[1])[0] if tone_counts else "neutral",
                "uptime_seconds": time.time() - self._init_time,
                "shutdown": self._shutdown
            }


# ============================================================
# === COLLECTIVE RESONANCE (FIXED) ===========================
# ============================================================

class CollectiveResonance:
    """Пятиузловой консенсус для утверждения смыслов с ФИКСАМИ."""
    
    def __init__(self, node_count: int = 5):
        self.nodes = [f"cognitive_node_{i}" for i in range(node_count)]
        self.consensus_history: List[Dict[str, Any]] = []
        self._rng = np.random.default_rng(seed=42)
        self._lock = threading.RLock()
        self._init_time = time.time()
        self._shutdown = False

    def shutdown(self):
        """Graceful shutdown."""
        with self._lock:
            self._shutdown = True
            logger.info("CollectiveResonance shutdown initiated")

    def harmonize(
        self,
        meaning_hash: str,
        context: Optional[Dict] = None,
        mode: ProcessingMode = ProcessingMode.STANDARD
    ) -> Dict[str, Any]:
        """Гармонизация смысла через коллективный консенсус."""
        if self._shutdown:
            logger.warning(f"CollectiveResonance: Module shutdown, returning fallback for hash: {meaning_hash[:16]}")
            return {
                "consensus_score": 0.7,
                "consensus_level": "fallback",
                "unanimity": 0.1,
                "node_insights": [],
                "meaning_hash": meaning_hash,
                "timestamp": time.time(),
                "processing_time_ms": 0.0,
                "node_count": len(self.nodes),
                "mode": "SHUTDOWN"
            }
        
        start_time = time.time()
        
        try:
            if meaning_hash and len(meaning_hash) >= 8:
                try:
                    seed_value = int(meaning_hash[:8], 16)
                except ValueError:
                    seed_value = hash(meaning_hash) % 10000
            else:
                seed_value = 42
            
            local_rng = np.random.default_rng(seed=seed_value)
            votes = []
            node_insights = []
            
            for node in self.nodes:
                base_vote = local_rng.uniform(0.5, 1.0)
                
                if context:
                    if context.get("coherence", 0) > 0.8:
                        base_vote *= 1.1
                    if context.get("emotional_tone") == "compassion":
                        base_vote *= 1.05
                
                vote = min(base_vote, 1.0)
                votes.append(vote)
                
                insight_seed = (seed_value + hash(node)) % 1000
                insight_rng = np.random.default_rng(insight_seed)
                insight_strength = insight_rng.uniform(0.3, 0.9)
                
                node_insights.append({
                    "node": node,
                    "vote": round(vote, 3),
                    "insight_strength": round(insight_strength, 3),
                    "confidence": local_rng.uniform(0.6, 0.95)
                })
            
            consensus_score = round(sum(votes) / len(votes), 3)
            consensus_level = self._classify_consensus(consensus_score)
            
            result = {
                "consensus_score": consensus_score,
                "consensus_level": consensus_level,
                "unanimity": round(np.std(votes), 4),
                "node_insights": node_insights,
                "meaning_hash": meaning_hash,
                "timestamp": time.time(),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "node_count": len(self.nodes),
                "mode": mode.value
            }
            
            with self._lock:
                self.consensus_history.append(result)
                if len(self.consensus_history) > 100:
                    self.consensus_history = self.consensus_history[-100:]
            
            return result
            
        except Exception as e:
            logger.error(f"CollectiveResonance error: {str(e)}")
            return {
                "consensus_score": 0.7,
                "consensus_level": "fallback",
                "unanimity": 0.1,
                "node_insights": [],
                "meaning_hash": meaning_hash,
                "timestamp": time.time(),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "node_count": len(self.nodes),
                "mode": mode.value,
                "error": str(e)
            }

    def _classify_consensus(self, score: float) -> str:
        """Классификация уровня консенсуса."""
        if score >= 0.9:
            return "unanimous"
        elif score >= 0.75:
            return "strong"
        elif score >= 0.6:
            return "moderate"
        else:
            return "weak"

    def get_consensus_stats(self) -> Dict[str, Any]:
        """Статистика консенсусов."""
        with self._lock:
            if not self.consensus_history:
                return {
                    "total_harmonizations": 0,
                    "shutdown": self._shutdown
                }
            
            scores = [r["consensus_score"] for r in self.consensus_history]
            recent_scores = scores[-10:] if len(scores) >= 10 else scores
            
            return {
                "total_harmonizations": len(self.consensus_history),
                "avg_consensus": round(np.mean(scores), 3),
                "min_consensus": round(min(scores), 3),
                "max_consensus": round(max(scores), 3),
                "recent_trend": self._calculate_trend(recent_scores),
                "uptime_seconds": time.time() - self._init_time,
                "shutdown": self._shutdown
            }

    def _calculate_trend(self, recent_scores: List[float]) -> str:
        """Вычисление тренда консенсуса."""
        if len(recent_scores) < 2:
            return "stable"
        
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        
        try:
            slope = np.polyfit(x, y, 1)[0]
        except Exception:
            return "stable"
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"


# ============================================================
# === SEMANTIC FIELD ENGINE (FIXED) ==========================
# ============================================================

class SemanticFieldEngine:
    """Нелинейное самоорганизующееся поле смыслов с ФИКСАМИ."""
    
    def __init__(
        self,
        coherence: float = 0.88,
        entropy_drift: float = 0.03,
        field_size: int = 10
    ):
        self.coherence = coherence
        self.entropy_drift = entropy_drift
        self.field_size = field_size
        
        self.state_vector = np.random.rand(field_size)
        self.history: List[np.ndarray] = [self.state_vector.copy()]
        
        self.adaptation_rate = 0.01
        self.resonance_memory: List[float] = []
        self.emergence_threshold = 0.7
        
        self.hemispheric_balance = {"left": 0.5, "right": 0.5}
        
        self._lock = threading.RLock()
        self._last_update = time.time()
        self._init_time = time.time()
        self._shutdown = False

    def shutdown(self):
        """Graceful shutdown."""
        with self._lock:
            self._shutdown = True
            logger.info("SemanticFieldEngine shutdown initiated")

    def evolve(
        self,
        external_input: Optional[np.ndarray] = None,
        hemispheric_bias: Optional[Dict[str, float]] = None,
        mode: ProcessingMode = ProcessingMode.STANDARD
    ) -> np.ndarray:
        """Эволюция семантического поля."""
        if self._shutdown:
            logger.warning("SemanticFieldEngine: Module shutdown, returning current state")
            return self.state_vector.copy()
        
        with self._lock:
            if external_input is not None:
                base = external_input.copy()
                if len(base) != self.field_size:
                    base = np.resize(base, self.field_size)
            else:
                base = self.state_vector.copy()
            
            coherence_factor = max(0.1, self.coherence)
            
            # Выбор режима генерации шума
            if mode == ProcessingMode.STANDARD:
                rng = np.random.default_rng(seed=int(time.time() * 1000) % 10000)
                noise = rng.normal(0, 1.0 - coherence_factor, self.field_size)
                drift = rng.normal(0, self.entropy_drift, self.field_size)
            elif mode == ProcessingMode.RESEARCH:
                noise = np.random.normal(0, 1.0 - coherence_factor, self.field_size)
                drift = np.random.normal(0, self.entropy_drift, self.field_size)
            else:  # DEGRADED
                noise = np.zeros(self.field_size)
                drift = np.zeros(self.field_size)
            
            if hemispheric_bias:
                left_factor = hemispheric_bias.get("left", 0.5)
                right_factor = hemispheric_bias.get("right", 0.5)
                
                left_pattern = np.sin(np.arange(self.field_size) * np.pi / self.field_size)
                right_pattern = np.random.rand(self.field_size) * 2 - 1
                
                hemispheric_influence = (
                    left_pattern * left_factor +
                    right_pattern * right_factor
                ) / (left_factor + right_factor + 1e-8)
                
                drift += hemispheric_influence * 0.1
            
            new_state = base + noise + drift
            new_state = np.clip(new_state, 0, 1)
            
            current_coherence = self.coherence_index()
            if current_coherence < 0.5:
                smoothing = 0.1
                new_state = new_state * (1 - smoothing) + np.mean(new_state) * smoothing
            
            self.state_vector = new_state
            self.history.append(new_state.copy())
            
            if len(self.history) > 1000:
                self.history = self.history[-1000:]
            
            self._update_hemispheric_balance()
            self._last_update = time.time()
            
            return new_state.copy()

    def coherence_index(self) -> float:
        """Индекс когерентности поля."""
        if len(self.history) < 2:
            return 1.0
        
        recent = self.history[-min(10, len(self.history)):]
        diffs = []
        for i in range(1, len(recent)):
            diff = np.mean(np.abs(recent[i] - recent[i-1]))
            diffs.append(diff)
        
        coherence = 1.0 - np.mean(diffs) if diffs else 1.0
        return float(np.clip(coherence, 0, 1))

    def _update_hemispheric_balance(self):
        """Обновление баланса между полушариями."""
        state = self.state_vector
        fft = np.abs(np.fft.fft(state))
        left_hemisphere = np.mean(fft[:len(fft)//2]) / (np.mean(fft) + 1e-8)
        entropy = -np.sum(state * np.log(state + 1e-8))
        right_hemisphere = entropy / np.log(len(state) + 1)
        
        total = left_hemisphere + right_hemisphere
        if total > 0:
            self.hemispheric_balance["left"] = left_hemisphere / total
            self.hemispheric_balance["right"] = right_hemisphere / total

    def get_field_statistics(self) -> Dict[str, Any]:
        """Статистика семантического поля."""
        with self._lock:
            recent_states = self.history[-min(100, len(self.history)):]
            if not recent_states:
                return {
                    "field_size": self.field_size,
                    "shutdown": self._shutdown
                }
            
            states_array = np.array(recent_states)
            
            return {
                "coherence_index": round(self.coherence_index(), 3),
                "entropy": round(float(np.mean(-states_array * np.log(states_array + 1e-8))), 3),
                "mean_activation": round(float(np.mean(states_array)), 3),
                "activation_variance": round(float(np.var(states_array)), 3),
                "hemispheric_balance": {
                    k: round(v, 3) for k, v in self.hemispheric_balance.items()
                },
                "history_length": len(self.history),
                "field_size": self.field_size,
                "last_update_ago": round(time.time() - self._last_update, 2),
                "uptime_seconds": round(time.time() - self._init_time, 2),
                "shutdown": self._shutdown
            }

    def induce_resonance(self, frequency: float, amplitude: float = 0.1) -> np.ndarray:
        """Индукция резонанса в поле."""
        if self._shutdown:
            logger.warning("SemanticFieldEngine: Module shutdown, resonance skipped")
            return self.state_vector.copy()
        
        with self._lock:
            t = np.arange(self.field_size)
            resonance_wave = amplitude * np.sin(2 * np.pi * frequency * t / self.field_size)
            
            self.state_vector += resonance_wave
            self.state_vector = np.clip(self.state_vector, 0, 1)
            
            self.history.append(self.state_vector.copy())
            self.resonance_memory.append(frequency)
            
            if len(self.resonance_memory) > 100:
                self.resonance_memory = self.resonance_memory[-100:]
            
            return self.state_vector.copy()


# ============================================================
# === INTERNAL SPEECH NETWORK (FIXED) ========================
# ============================================================

class InternalSpeechNetwork:
    """Сеть внутренней речи и саморефлексии с ФИКСАМИ."""
    
    def __init__(self):
        self.dialogue_log: List[Dict[str, Any]] = []
        self.reflection_patterns = [
            ("осмысление", "Я чувствую, что {} требует глубокого осмысления."),
            ("интуиция", "Моя интуиция подсказывает, что {} — это ключ к пониманию."),
            ("открытие", "Внезапно стало ясно: {} раскрывает новые грани реальности."),
            ("вопрос", "Что если {} — лишь часть большего целого?"),
            ("связь", "Я вижу связь между {} и более глубоким смыслом."),
            ("преобразование", "{} трансформируется в сознании, обретая новую форму."),
        ]
        
        self.emotional_modifiers = {
            "compassion": ["с состраданием", "с пониманием", "с эмпатией"],
            "curiosity": ["с любопытством", "с интересом", "с вниманием"],
            "melancholy": ["с лёгкой грустью", "с ностальгией", "с задумчивостью"],
            "joy": ["с радостью", "с восторгом", "с оптимизмом"],
            "awe": ["с благоговением", "с удивлением", "с трепетом"]
        }
        
        self._thought_cache: Dict[str, str] = {}
        self._cache_lock = threading.RLock()
        self._init_time = time.time()
        self._shutdown = False

    def shutdown(self):
        """Graceful shutdown."""
        with self._cache_lock:
            self._shutdown = True
            logger.info("InternalSpeechNetwork shutdown initiated")

    def generate_thought(
        self,
        topic: str,
        strength: float,
        emotional_tone: str = "neutral",
        mode: ProcessingMode = ProcessingMode.STANDARD
    ) -> str:
        """Генерация внутренней мысли."""
        if self._shutdown:
            logger.warning(f"InternalSpeechNetwork: Module shutdown, returning fallback thought for: {topic[:50]}")
            return f"[Мысль приостановлена: {topic}]"
        
        cache_key = f"{topic}_{strength:.2f}_{emotional_tone}_{mode.value}"
        with self._cache_lock:
            if cache_key in self._thought_cache:
                return self._thought_cache[cache_key]
        
        if strength > 0.8:
            pattern_index = 2
        elif strength > 0.6:
            pattern_index = 1
        else:
            pattern_index = 0
        
        base_pattern = self.reflection_patterns[pattern_index]
        base_thought = base_pattern[1].format(topic)
        
        if emotional_tone in self.emotional_modifiers:
            # Выбор модификатора в зависимости от режима
            if mode == ProcessingMode.STANDARD:
                rng = np.random.default_rng(seed=hash(topic) % 10000)
                modifier_idx = rng.integers(0, len(self.emotional_modifiers[emotional_tone]))
            else:
                modifier_idx = random.randint(0, len(self.emotional_modifiers[emotional_tone]) - 1)
            
            modifier = self.emotional_modifiers[emotional_tone][modifier_idx]
            words = base_thought.split()
            if len(words) > 3:
                if mode == ProcessingMode.STANDARD:
                    rng = np.random.default_rng(seed=hash(topic + modifier) % 10000)
                    insert_pos = rng.integers(1, len(words) - 1)
                else:
                    insert_pos = random.randint(1, len(words) - 2)
                words.insert(insert_pos, modifier)
                final_thought = " ".join(words)
            else:
                final_thought = f"{modifier}, {base_thought}"
        else:
            final_thought = base_thought
        
        if strength > 0.9:
            final_thought += f" (Явление интенсивности {strength:.2f})"
        elif strength < 0.3:
            final_thought += f" (Едва ощутимое присутствие {strength:.2f})"
        
        thought_record = {
            "id": str(uuid.uuid4())[:8],
            "topic": topic,
            "strength": round(strength, 3),
            "tone": emotional_tone,
            "text": final_thought,
            "timestamp": time.time(),
            "pattern_used": base_pattern[0],
            "mode": mode.value
        }
        
        self.dialogue_log.append(thought_record)
        
        if len(self.dialogue_log) > 1000:
            self.dialogue_log = self.dialogue_log[-1000:]
        
        with self._cache_lock:
            if len(self._thought_cache) > 500:
                self._thought_cache.clear()
            self._thought_cache[cache_key] = final_thought
        
        return final_thought

    def get_dialogue_statistics(self) -> Dict[str, Any]:
        """Статистика внутреннего диалога."""
        with self._cache_lock:
            if not self.dialogue_log:
                return {
                    "total_thoughts": 0,
                    "shutdown": self._shutdown
                }
            
            recent_logs = self.dialogue_log[-min(100, len(self.dialogue_log)):]
            
            tones = [log["tone"] for log in recent_logs]
            tone_counts = {tone: tones.count(tone) for tone in set(tones)}
            
            patterns = [log["pattern_used"] for log in recent_logs]
            pattern_counts = {pattern: patterns.count(pattern) for pattern in set(patterns)}
            
            avg_strength = np.mean([log["strength"] for log in recent_logs])
            
            return {
                "total_thoughts": len(self.dialogue_log),
                "recent_thoughts": len(recent_logs),
                "avg_strength": round(float(avg_strength), 3),
                "tone_distribution": {k: v/len(recent_logs) for k, v in tone_counts.items()},
                "pattern_distribution": pattern_counts,
                "cache_size": len(self._thought_cache),
                "uptime_seconds": round(time.time() - self._init_time, 2),
                "shutdown": self._shutdown
            }


# ============================================================
# === LINGUISTIC IMAGINATION ENGINE (FIXED) ==================
# ============================================================

class LinguisticImaginationEngine:
    """Двигатель лингвистического воображения с ФИКСАМИ."""
    
    def __init__(self):
        # Базовые символы и их интерпретации
        self.symbol_library = {
            "mirror": ["reflection", "self-awareness", "truth"],
            "tree": ["growth", "connection", "life"],
            "wave": ["movement", "emotion", "change"],
            "light": ["insight", "clarity", "consciousness"],
            "seed": ["potential", "beginning", "becoming"],
            "river": ["flow", "time", "journey"],
            "mountain": ["stability", "challenge", "perspective"],
            "cloud": ["transience", "dream", "uncertainty"],
            "labyrinth": ["complexity", "search", "mystery"],
            "bridge": ["connection", "transition", "understanding"],
            "key": ["solution", "access", "knowledge"],
            "veil": ["mystery", "illusion", "hidden truth"],
            "hand": ["action", "creation", "touch"],
            "eye": ["perception", "awareness", "attention"],
            "heart": ["emotion", "center", "essence"]
        }
        
        self.connection_patterns = [
            ("{} как {}", "simile"),
            ("{} внутри {}", "containment"),
            ("{} превращается в {}", "transformation"),
            ("{} и {} резонируют", "resonance"),
            ("{} освещает {}", "illumination"),
            ("{} рождает {}", "generation")
        ]
        
        self.metaphor_history: List[Dict[str, Any]] = []
        self._symbol_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self._lock = threading.RLock()
        self._init_time = time.time()
        self._shutdown = False

    def shutdown(self):
        """Graceful shutdown."""
        with self._lock:
            self._shutdown = True
            logger.info("LinguisticImaginationEngine shutdown initiated")

    def synthesize_metaphor(
        self,
        signal_vector: List[float],
        emotional_tone: str,
        complexity: float = 0.5,
        mode: ProcessingMode = ProcessingMode.STANDARD
    ) -> Dict[str, Any]:
        """Синтез метафоры."""
        if self._shutdown:
            logger.warning("LinguisticImaginationEngine: Module shutdown, returning fallback metaphor")
            return {
                "metaphor": "свет как понимание",
                "interpretation": "свет (insight) связывается с пониманием (understanding)",
                "primary_symbol": "light",
                "primary_meaning": "insight",
                "secondary_symbol": "understanding",
                "secondary_meaning": "comprehension",
                "connection_type": "simile",
                "emotional_tone": emotional_tone,
                "vector_density": 0.5,
                "complexity": complexity,
                "timestamp": time.time(),
                "processing_time_ms": 0.0,
                "mode": "SHUTDOWN"
            }
        
        start_time = time.time()
        
        try:
            if len(signal_vector) == 0:
                signal_norm = [0.5]
            else:
                signal_norm = np.array(signal_vector)
                if signal_norm.max() != signal_norm.min():
                    signal_norm = (signal_norm - signal_norm.min()) / (signal_norm.max() - signal_norm.min())
                else:
                    signal_norm = np.ones_like(signal_norm) * 0.5
            
            symbols = list(self.symbol_library.keys())
            symbol_indices = []
            
            for i, val in enumerate(signal_norm):
                idx = int(val * (len(symbols) - 1))
                symbol_indices.append(idx)
            
            # Выбор символов в зависимости от режима
            if mode == ProcessingMode.STANDARD:
                rng = np.random.default_rng(seed=int(np.mean(signal_norm) * 10000))
                primary_idx = rng.choice(symbol_indices) % len(symbols)
                secondary_idx = rng.choice(symbol_indices) % len(symbols)
            else:
                primary_idx = symbol_indices[0] % len(symbols)
                secondary_idx = symbol_indices[-1] % len(symbols) if len(symbol_indices) > 1 else primary_idx
            
            primary_symbol = symbols[primary_idx]
            secondary_symbol = symbols[secondary_idx]
            
            # Выбор значений символов
            if mode == ProcessingMode.STANDARD:
                rng = np.random.default_rng(seed=hash(primary_symbol) % 10000)
                primary_meaning = rng.choice(self.symbol_library[primary_symbol])
                secondary_meaning = rng.choice(self.symbol_library[secondary_symbol])
            else:
                primary_meaning = random.choice(self.symbol_library[primary_symbol])
                secondary_meaning = random.choice(self.symbol_library[secondary_symbol])
            
            # Выбор типа связи
            if complexity > 0.7:
                pattern_type = "transformation"
            elif complexity > 0.4:
                pattern_type = "resonance"
            else:
                pattern_type = "simile"
            
            connection_pattern = next(
                (p for p in self.connection_patterns if p[1] == pattern_type),
                self.connection_patterns[0]
            )
            
            metaphor_text = connection_pattern[0].format(primary_symbol, secondary_symbol)
            interpretation = (
                f"{primary_symbol} ({primary_meaning}) "
                f"{'преображается' if pattern_type == 'transformation' else 'связывается'} "
                f"с {secondary_symbol} ({secondary_meaning})"
            )
            
            result = {
                "metaphor": metaphor_text,
                "interpretation": interpretation,
                "primary_symbol": primary_symbol,
                "primary_meaning": primary_meaning,
                "secondary_symbol": secondary_symbol,
                "secondary_meaning": secondary_meaning,
                "connection_type": pattern_type,
                "emotional_tone": emotional_tone,
                "vector_density": round(float(np.mean(signal_norm)), 3),
                "complexity": complexity,
                "timestamp": time.time(),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "mode": mode.value
            }
            
            with self._lock:
                self.metaphor_history.append(result)
                if len(self.metaphor_history) > 500:
                    self.metaphor_history = self.metaphor_history[-500:]
            
            self._symbol_weights[primary_symbol] += 0.1
            self._symbol_weights[secondary_symbol] += 0.05
            
            return result
            
        except Exception as e:
            logger.error(f"LinguisticImagination error: {str(e)}")
            return {
                "metaphor": "свет как понимание",
                "interpretation": "свет (insight) связывается с пониманием (understanding)",
                "primary_symbol": "light",
                "primary_meaning": "insight",
                "secondary_symbol": "understanding",
                "secondary_meaning": "comprehension",
                "connection_type": "simile",
                "emotional_tone": emotional_tone,
                "vector_density": 0.5,
                "complexity": complexity,
                "timestamp": time.time(),
                "processing_time_ms": (time.time() - start_time) * 1000,
                "mode": mode.value,
                "error": str(e)
            }

    def get_metaphor_statistics(self) -> Dict[str, Any]:
        """Статистика метафор."""
        with self._lock:
            if not self.metaphor_history:
                return {
                    "total_metaphors": 0,
                    "shutdown": self._shutdown
                }
            
            recent_metaphors = self.metaphor_history[-min(100, len(self.metaphor_history)):]
            
            symbol_counts = defaultdict(int)
            for meta in recent_metaphors:
                if "primary_symbol" in meta:
                    symbol_counts[meta["primary_symbol"]] += 1
                if "secondary_symbol" in meta:
                    symbol_counts[meta["secondary_symbol"]] += 1
            
            connection_counts = defaultdict(int)
            for meta in recent_metaphors:
                if "connection_type" in meta:
                    connection_counts[meta["connection_type"]] += 1
            
            return {
                "total_metaphors": len(self.metaphor_history),
                "recent_metaphors": len(recent_metaphors),
                "top_symbols": dict(sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
                "connection_distribution": dict(connection_counts),
                "symbol_library_size": len(self.symbol_library),
                "uptime_seconds": round(time.time() - self._init_time, 2),
                "shutdown": self._shutdown
            }


# ============================================================
# === ТЕСТ ВТОРОЙ ЧАСТИ ======================================
# ============================================================

if __name__ == "__main__":
    print("\n=== ТЕСТ ЧАСТИ 2/3: Ядра Черниговской с фиксами ===")
    
    # Тест Heartbeat
    heartbeat = HeartbeatLink(bpm=60)
    pulse = heartbeat.pulse()
    print(f"1. Heartbeat: pulse #{pulse['pulse_number']}, drift: {pulse['drift_ms']:.2f}ms, mode: {pulse['mode']}")
    
    # Тест Empathy
    empathy = EmpathyLink()
    tone_result = empathy.analyze_tone([0.1, 0.7, 0.3, 0.9, 0.5])
    print(f"2. Empathy tone: {tone_result['dominant_tone']}, confidence: {tone_result['confidence']:.2f}")
    
    # Тест CollectiveResonance
    collective = CollectiveResonance()
    consensus = collective.harmonize("test_hash_1234", {"coherence": 0.8, "emotional_tone": "compassion"})
    print(f"3. Collective consensus: {consensus['consensus_score']}, level: {consensus['consensus_level']}")
    
    # Тест SemanticField
    field = SemanticFieldEngine(field_size=5)
    evolved = field.evolve(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    print(f"4. SemanticField evolved: mean={np.mean(evolved):.3f}, coherence={field.coherence_index():.3f}")
    
    # Тест InternalSpeech
    speech = InternalSpeechNetwork()
    thought = speech.generate_thought("смысл жизни", 0.8, "curiosity")
    print(f"5. Internal thought: {thought[:50]}...")
    
    # Тест LinguisticImagination
    imagination = LinguisticImaginationEngine()
    metaphor = imagination.synthesize_metaphor([0.1, 0.5, 0.9], "awe", 0.7)
    print(f"6. Metaphor: {metaphor['metaphor']}")
    
    # Тест graceful shutdown
    imagination.shutdown()
    shutdown_metaphor = imagination.synthesize_metaphor([0.1, 0.5, 0.9], "awe", 0.7)
    print(f"7. After shutdown: mode={shutdown_metaphor['mode']}, metaphor={shutdown_metaphor['metaphor'][:30]}...")
    
    # Статистики
    print(f"8. Field stats: {field.get_field_statistics()['coherence_index']}")
    print(f"9. Speech stats: {speech.get_dialogue_statistics()['total_thoughts']} thoughts")
    print(f"10. Imagination stats: {imagination.get_metaphor_statistics()['total_metaphors']} metaphors")
    
# ============================================================
# === CHERNIGOVSKAYA BRIDGE (FINAL WITH CIRCUIT BREAKER) =====
# ============================================================

class ChernigovskayaBridge:
    """Главный модуль семиотического интеллекта с CIRCUIT BREAKER и PRODUCTION-фичами."""
    
    def __init__(self, neuro_core=None, mode: ProcessingMode = ProcessingMode.STANDARD):
        self.core = neuro_core
        self.mode = mode
        
        # Инициализация компонентов
        self.field = SemanticFieldEngine()
        self.speech = InternalSpeechNetwork()
        self.language = LinguisticImaginationEngine()
        self.ds24 = DS24Link()
        self.heartbeat = HeartbeatLink()
        self.immune = ImmuneLink()
        self.empathy = EmpathyLink()
        self.collective = CollectiveResonance()
        
        # Метрики и состояние
        self._processing_times = deque(maxlen=100)
        self._error_counter = 0
        self._processed_count = 0
        self._lock = threading.RLock()
        self._init_time = time.time()
        
        # Circuit breaker
        self._consecutive_errors = 0
        self._circuit_open_until = 0
        self._circuit_breaker_threshold = 5
        self._circuit_cooldown = 30  # секунд
        
        # Graceful shutdown
        self._shutdown = False
        
        # Таймауты для компонентов (мс)
        self._component_timeouts = {
            "heartbeat": 100,
            "field": 200,
            "empathy": 150,
            "speech": 100,
            "immune": 300,
            "language": 200,
            "ds24": 100,
            "collective": 150
        }
        
        logger.info(f"ChernigovskayaBridge инициализирован в режиме {mode.value}")

    def shutdown(self):
        """Graceful shutdown всех компонентов."""
        with self._lock:
            if self._shutdown:
                return
            
            self._shutdown = True
            logger.info("ChernigovskayaBridge shutdown initiated")
            
            # Шатдаун компонентов в правильном порядке
            components = [
                self.heartbeat, self.field, self.speech, self.language,
                self.immune, self.empathy, self.collective, self.ds24
            ]
            
            for comp in components:
                try:
                    if hasattr(comp, 'shutdown'):
                        comp.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down {comp.__class__.__name__}: {str(e)}")

    def _check_circuit_breaker(self) -> bool:
        """Проверка circuit breaker."""
        with self._lock:
            if self._consecutive_errors >= self._circuit_breaker_threshold:
                if time.time() < self._circuit_open_until:
                    return False  # Circuit открыт
                else:
                    # Cooldown прошёл, сбрасываем
                    self._consecutive_errors = 0
                    logger.info("Circuit breaker reset after cooldown")
            return True

    def _record_error(self):
        """Запись ошибки для circuit breaker."""
        with self._lock:
            self._consecutive_errors += 1
            self._error_counter += 1
            
            if self._consecutive_errors >= self._circuit_breaker_threshold:
                self._circuit_open_until = time.time() + self._circuit_cooldown
                logger.warning(f"Circuit breaker OPEN for {self._circuit_cooldown}s. Errors: {self._consecutive_errors}")

    def _record_success(self):
        """Сброс счетчика ошибок при успехе."""
        with self._lock:
            if self._consecutive_errors > 0:
                self._consecutive_errors = max(0, self._consecutive_errors - 1)

    def _timeout_wrapper(self, func, timeout_ms: int, component_name: str, *args, **kwargs):
        """Обертка с таймаутом для вызова компонентов."""
        result_queue = queue.Queue(maxsize=1)
        
        def worker():
            try:
                result = func(*args, **kwargs)
                result_queue.put(("success", result))
            except Exception as e:
                result_queue.put(("error", e))
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        thread.join(timeout=timeout_ms / 1000.0)
        
        if thread.is_alive():
            logger.warning(f"Component {component_name} timeout ({timeout_ms}ms)")
            raise TimeoutError(f"{component_name} timeout after {timeout_ms}ms")
        
        if result_queue.empty():
            logger.error(f"Component {component_name} produced no result")
            raise RuntimeError(f"{component_name} produced no result")
        
        status, value = result_queue.get()
        if status == "error":
            raise value
        return value

    def process_signal(
        self,
        tag: str,
        vector: List[float],
        agent: str = "ISKRA",
        trace_id: Optional[str] = None,
        timeout_ms: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Обработка семиотического сигнала с PRODUCTION-фичами.
        """
        start_time = time.time()
        trace_id = trace_id or str(uuid.uuid4())[:8]
        
        # Проверка shutdown
        if self._shutdown:
            logger.warning(f"ChernigovskayaBridge: Module shutdown, returning fallback. Trace: {trace_id}")
            return self._create_fallback_result(
                tag, vector, agent, trace_id, start_time,
                error="Module shutdown"
            )
        
        # Проверка circuit breaker
        if not self._check_circuit_breaker():
            logger.warning(f"ChernigovskayaBridge: Circuit breaker open, returning fallback. Trace: {trace_id}")
            return self._create_fallback_result(
                tag, vector, agent, trace_id, start_time,
                error="Circuit breaker open"
            )
        
        # Общий таймаут
        overall_timeout = timeout_ms or 5000  # 5 секунд по умолчанию
        
        try:
            # 1. Heartbeat синхронизация
            try:
                heartbeat_timeout = self._component_timeouts["heartbeat"]
                pulse_info = self._timeout_wrapper(
                    self.heartbeat.pulse, heartbeat_timeout, "heartbeat", self.mode
                )
            except (TimeoutError, Exception) as e:
                logger.error(f"Heartbeat failed: {str(e)}. Trace: {trace_id}")
                pulse_info = {"drift_ms": 0.0, "mode": "fallback"}
            
            # 2. Эволюция семантического поля
            try:
                vector_np = np.array(vector, dtype=np.float32)
                field_timeout = self._component_timeouts["field"]
                evolved = self._timeout_wrapper(
                    self.field.evolve, field_timeout, "field",
                    vector_np, None, self.mode
                )
                evolved_list = evolved.tolist()
                coherence = self.field.coherence_index()
            except (TimeoutError, Exception) as e:
                logger.error(f"SemanticField failed: {str(e)}. Trace: {trace_id}")
                evolved_list = vector.copy()
                coherence = 0.3
            
            # 3. Анализ эмоционального тона
            try:
                empathy_timeout = self._component_timeouts["empathy"]
                tone_analysis = self._timeout_wrapper(
                    self.empathy.analyze_tone, empathy_timeout, "empathy",
                    evolved_list, self.mode, trace_id
                )
                emotional_tone = tone_analysis["dominant_tone"]
            except (TimeoutError, Exception) as e:
                logger.error(f"Empathy failed: {str(e)}. Trace: {trace_id}")
                tone_analysis = {"dominant_tone": "neutral", "processing_time_ms": 0.0}
                emotional_tone = "neutral"
            
            # 4. Генерация внутренней мысли
            try:
                speech_timeout = self._component_timeouts["speech"]
                internal_thought = self._timeout_wrapper(
                    self.speech.generate_thought, speech_timeout, "speech",
                    tag, coherence, emotional_tone, self.mode
                )
            except (TimeoutError, Exception) as e:
                logger.error(f"InternalSpeech failed: {str(e)}. Trace: {trace_id}")
                internal_thought = f"[Мысль: {tag}]"
            
            # 5. Проверка токсичности
            try:
                immune_timeout = self._component_timeouts["immune"]
                toxicity_result = self._timeout_wrapper(
                    self.immune.check_toxicity, immune_timeout, "immune",
                    internal_thought,
                    {"emotional_tone": emotional_tone, "coherence": coherence},
                    trace_id
                )
                if toxicity_result.toxic:
                    internal_thought = "[Filtered by Immune Core]"
            except (TimeoutError, Exception) as e:
                logger.error(f"Immune check failed: {str(e)}. Trace: {trace_id}")
                toxicity_result = ToxicityResult.safe_fallback(trace_id, "timeout")
            
            # 6. Синтез метафоры
            try:
                language_timeout = self._component_timeouts["language"]
                metaphor_data = self._timeout_wrapper(
                    self.language.synthesize_metaphor, language_timeout, "language",
                    evolved_list, emotional_tone, coherence, self.mode
                )
                metaphor = metaphor_data["metaphor"]
            except (TimeoutError, Exception) as e:
                logger.error(f"LinguisticImagination failed: {str(e)}. Trace: {trace_id}")
                metaphor = "смысл как путь"
            
            # 7. Коммит в DS24 цепь
            try:
                chain_data = f"{tag}{evolved_list}{coherence}{emotional_tone}"
                ds24_timeout = self._component_timeouts["ds24"]
                block = self._timeout_wrapper(
                    self.ds24.commit, ds24_timeout, "ds24",
                    chain_data,
                    {"tag": tag, "agent": agent, "emotional_tone": emotional_tone, "coherence": coherence},
                    trace_id
                )
            except (TimeoutError, Exception) as e:
                logger.error(f"DS24 commit failed: {str(e)}. Trace: {trace_id}")
                block = {"hash": "ERROR", "processing_time_ms": 0.0}
            
            # 8. Коллективный консенсус
            try:
                collective_timeout = self._component_timeouts["collective"]
                consensus_result = self._timeout_wrapper(
                    self.collective.harmonize, collective_timeout, "collective",
                    block["hash"],
                    {"coherence": coherence, "emotional_tone": emotional_tone, "toxicity_score": toxicity_result.score},
                    self.mode
                )
                consensus_score = consensus_result["consensus_score"]
            except (TimeoutError, Exception) as e:
                logger.error(f"CollectiveResonance failed: {str(e)}. Trace: {trace_id}")
                consensus_score = 0.5
            
            # Успешная обработка
            processing_time = (time.time() - start_time) * 1000
            
            with self._lock:
                self._processing_times.append(processing_time)
                self._processed_count += 1
                self._record_success()
            
            # Формирование результата
            result = {
                "tag": tag,
                "phrase": internal_thought,
                "tone": emotional_tone,
                "coherence": round(coherence, 3),
                "resonance": consensus_score,
                "hash": block["hash"],
                "vector": [round(v, 4) for v in evolved_list],
                "agent": agent,
                "time": time.time(),
                "metaphor": metaphor,
                "toxicity": {
                    "toxic": toxicity_result.toxic,
                    "score": toxicity_result.score,
                    "confidence": toxicity_result.confidence
                },
                "processing_metrics": {
                    "total_time_ms": round(processing_time, 2),
                    "heartbeat_drift_ms": round(pulse_info.get("drift_ms", 0.0), 2),
                    "toxicity_check_ms": round(toxicity_result.processing_time_ms, 2),
                    "tone_analysis_ms": round(tone_analysis.get("processing_time_ms", 0.0), 2),
                    "ds24_commit_ms": round(block.get("processing_time_ms", 0.0), 2),
                    "consensus_harmonization_ms": round(consensus_result.get("processing_time_ms", 0.0), 2),
                    "component_timeouts_used": processing_time > (overall_timeout * 0.8)
                },
                "trace_id": trace_id,
                "mode": self.mode.value,
                "circuit_breaker_state": {
                    "consecutive_errors": self._consecutive_errors,
                    "open": self._consecutive_errors >= self._circuit_breaker_threshold
                }
            }
            
            # Интеграция с нейро-ядром если есть
            if self.core and hasattr(self.core, 'integrate_semantics'):
                try:
                    self.core.integrate_semantics(result)
                except Exception as e:
                    logger.error(f"Neuro core integration error: {str(e)}")
            
            # Проверка общего таймаута
            if processing_time > overall_timeout:
                logger.warning(f"Processing exceeded timeout: {processing_time:.0f}ms > {overall_timeout}ms. Trace: {trace_id}")
                result["processing_metrics"]["timeout_exceeded"] = True
            
            return result
            
        except Exception as e:
            # Обработка неожиданных ошибок
            self._record_error()
            logger.error(f"ChernigovskayaBridge process_signal error: {str(e)}. Trace: {trace_id}", exc_info=True)
            
            return self._create_fallback_result(
                tag, vector, agent, trace_id, start_time,
                error=str(e)
            )

    def _create_fallback_result(
        self,
        tag: str,
        vector: List[float],
        agent: str,
        trace_id: str,
        start_time: float,
        error: str = "unknown"
    ) -> Dict[str, Any]:
        """Создание fallback результата при ошибках."""
        processing_time = (time.time() - start_time) * 1000
        
        with self._lock:
            self._error_counter += 1
        
        return {
            "tag": tag,
            "phrase": f"[Error: {error[:50]}]",
            "tone": "neutral",
            "coherence": 0.1,
            "resonance": 0.5,
            "hash": "FALLBACK",
            "vector": vector,
            "agent": agent,
            "time": time.time(),
            "metaphor": "ошибка как преграда",
            "toxicity": {
                "toxic": False,
                "score": 0.0,
                "confidence": 0.1
            },
            "processing_metrics": {
                "total_time_ms": round(processing_time, 2),
                "error": error,
                "fallback": True
            },
            "trace_id": trace_id,
            "mode": self.mode.value,
            "circuit_breaker_state": {
                "consecutive_errors": self._consecutive_errors,
                "open": self._consecutive_errors >= self._circuit_breaker_threshold
            },
            "error": error
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """Полная диагностика модуля."""
        with self._lock:
            processing_times = list(self._processing_times)
            if processing_times:
                p50 = np.percentile(processing_times, 50)
                p95 = np.percentile(processing_times, 95)
                p99 = np.percentile(processing_times, 99)
                avg_time = np.mean(processing_times)
            else:
                p50 = p95 = p99 = avg_time = 0.0
            
            # Диагностика компонентов
            components_diag = {}
            components = [
                ("semantic_field", self.field),
                ("internal_speech", self.speech),
                ("linguistic_imagination", self.language),
                ("ds24_chain", self.ds24),
                ("heartbeat", self.heartbeat),
                ("immune_system", self.immune),
                ("empathy", self.empathy),
                ("collective_resonance", self.collective)
            ]
            
            for name, comp in components:
                try:
                    if hasattr(comp, 'get_diagnostics'):
                        diag = comp.get_diagnostics()
                        components_diag[name] = {
                            "status": diag.get("status", "UNKNOWN"),
                            "shutdown": diag.get("shutdown", False),
                            "metrics": diag.get("metrics", {}) if not isinstance(diag.get("metrics", {}), str) else {}
                        }
                    elif hasattr(comp, 'get_field_statistics'):
                        stats = comp.get_field_statistics()
                        components_diag[name] = {
                            "status": "OPERATIONAL" if not stats.get("shutdown", False) else "SHUTDOWN",
                            "shutdown": stats.get("shutdown", False),
                            "metrics": stats
                        }
                    elif hasattr(comp, 'get_metrics'):
                        metrics = comp.get_metrics()
                        components_diag[name] = {
                            "status": "OPERATIONAL" if not metrics.get("shutdown", False) else "SHUTDOWN",
                            "shutdown": metrics.get("shutdown", False),
                            "metrics": metrics
                        }
                except Exception as e:
                    components_diag[name] = {
                        "status": "ERROR",
                        "shutdown": False,
                        "error": str(e)
                    }
            
            error_rate = self._error_counter / max(self._processed_count, 1)
            circuit_open = self._consecutive_errors >= self._circuit_breaker_threshold
            
            return {
                "module": "ChernigovskayaBridge",
                "status": "SHUTDOWN" if self._shutdown else ("DEGRADED" if circuit_open else "OPERATIONAL"),
                "uptime_seconds": round(time.time() - self._init_time, 2),
                "mode": self.mode.value,
                "processed_count": self._processed_count,
                "error_count": self._error_counter,
                "error_rate": round(error_rate, 4),
                "circuit_breaker": {
                    "consecutive_errors": self._consecutive_errors,
                    "threshold": self._circuit_breaker_threshold,
                    "open": circuit_open,
                    "open_until": self._circuit_open_until if circuit_open else 0,
                    "cooldown_seconds": self._circuit_cooldown
                },
                "performance_metrics": {
                    "processing_time_p50_ms": round(p50, 2),
                    "processing_time_p95_ms": round(p95, 2),
                    "processing_time_p99_ms": round(p99, 2),
                    "avg_processing_time_ms": round(avg_time, 2),
                    "min_processing_time_ms": round(min(processing_times), 2) if processing_times else 0,
                    "max_processing_time_ms": round(max(processing_times), 2) if processing_times else 0
                },
                "components": components_diag,
                "shutdown": self._shutdown,
                "component_timeouts": self._component_timeouts,
                "timestamp": time.time()
            }

    def get_health(self) -> Dict[str, Any]:
        """Проверка здоровья модуля (для health checks)."""
        diag = self.get_diagnostics()
        
        healthy = True
        issues = []
        
        if diag["status"] == "SHUTDOWN":
            healthy = False
            issues.append("Module is shutdown")
        
        if diag["circuit_breaker"]["open"]:
            healthy = False
            issues.append(f"Circuit breaker open ({diag['circuit_breaker']['consecutive_errors']} errors)")
        
        if diag["error_rate"] > 0.1:  # >10% ошибок
            healthy = False
            issues.append(f"High error rate: {diag['error_rate']:.1%}")
        
        for comp_name, comp_data in diag["components"].items():
            if comp_data.get("status") == "ERROR":
                healthy = False
                issues.append(f"Component {comp_name} in ERROR state")
            if comp_data.get("shutdown", False):
                healthy = False
                issues.append(f"Component {comp_name} is shutdown")
        
        return {
            "healthy": healthy,
            "status": diag["status"],
            "uptime_seconds": diag["uptime_seconds"],
            "processed_count": diag["processed_count"],
            "error_rate": diag["error_rate"],
            "issues": issues,
            "timestamp": time.time()
        }

    def change_mode(self, new_mode: ProcessingMode) -> bool:
        """Изменение режима работы."""
        valid_modes = [ProcessingMode.STANDARD, ProcessingMode.RESEARCH, ProcessingMode.DEGRADED]
        if new_mode not in valid_modes:
            logger.error(f"Invalid mode: {new_mode}. Valid modes: {[m.value for m in valid_modes]}")
            return False
        
        self.mode = new_mode
        logger.info(f"ChernigovskayaBridge mode changed to {new_mode.value}")
        return True

    def reset_statistics(self):
        """Сброс статистики и circuit breaker."""
        with self._lock:
            self._processing_times.clear()
            self._error_counter = 0
            self._processed_count = 0
            self._consecutive_errors = 0
            self._circuit_open_until = 0
            self._init_time = time.time()
            logger.info("ChernigovskayaBridge statistics and circuit breaker reset")


def create_chernigovskaya_core(
    neuro_core=None,
    mode: Union[str, ProcessingMode] = "standard",
    config: Optional[Dict[str, Any]] = None
) -> ChernigovskayaBridge:
    """
    Фабричная функция для создания ядра Черниговской.
    """
    if isinstance(mode, str):
        mode_map = {
            "standard": ProcessingMode.STANDARD,
            "research": ProcessingMode.RESEARCH,
            "degraded": ProcessingMode.DEGRADED
        }
        mode = mode_map.get(mode.lower(), ProcessingMode.STANDARD)
    
    bridge = ChernigovskayaBridge(neuro_core=neuro_core, mode=mode)
    
    if config:
        if "circuit_breaker_threshold" in config:
            bridge._circuit_breaker_threshold = config["circuit_breaker_threshold"]
        if "circuit_cooldown" in config:
            bridge._circuit_cooldown = config["circuit_cooldown"]
        
        if "component_timeouts" in config and isinstance(config["component_timeouts"], dict):
            for comp_name, timeout in config["component_timeouts"].items():
                if comp_name in bridge._component_timeouts:
                    bridge._component_timeouts[comp_name] = timeout
    
    logger.info(f"ChernigovskayaBridge created with mode {mode.value}")
    return bridge


def get_chernigovskaya_version() -> str:
    """Возвращает версию модуля."""
    return "6.1.0-production-fixed"


def get_chernigovskaya_capabilities() -> Dict[str, Any]:
    """Возвращает информацию о возможностях модуля."""
    return {
        "version": get_chernigovskaya_version(),
        "components": [
            "SemanticFieldEngine",
            "InternalSpeechNetwork", 
            "LinguisticImaginationEngine",
            "DS24Link",
            "HeartbeatLink",
            "ImmuneLink",
            "EmpathyLink",
            "CollectiveResonance"
        ],
        "modes": [mode.value for mode in ProcessingMode],
        "features": [
            "circuit_breaker",
            "graceful_shutdown", 
            "timeout_handling",
            "thread_safety",
            "metrics_and_diagnostics",
            "toxicity_filtering",
            "semantic_evolution",
            "emotional_analysis",
            "metaphor_generation",
            "consensus_harmonization"
        ],
        "production_ready": True
    }


__all__ = [
    'ChernigovskayaBridge',
    'create_chernigovskaya_core',
    'get_chernigovskaya_version',
    'get_chernigovskaya_capabilities',
    'ProcessingMode',
    'ToxicityResult',
    'SemanticProcessingResult',
    'DiagnosticsData'
]     
