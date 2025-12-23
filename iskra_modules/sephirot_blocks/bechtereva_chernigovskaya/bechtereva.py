"""
БЕХТЕРЕВА.PY v1.5.0 (PRODUCTION 10/10) - ЧАСТЬ 1/2
Нейро-интеграционный блок DS24/ISKRA-4
Безупречная реализация INeuroIntegration протокола
Принципы Н.П. Бехтеревой + промышленные стандарты качества
"""

import numpy as np
import time
import threading
import hashlib
import warnings
from typing import Dict, List, Optional, Any, Protocol, TypedDict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from collections import deque


# ============================================================
# АРХИТЕКТУРНЫЕ КОНСТАНТЫ
# ============================================================

ARCHITECTURAL_CONSTRAINTS = {
    "LIMITS": {
        "MAX_PREDICTION_DEPTH": 5,
        "MIN_SEMANTIC_RESONANCE": 0.85,
        "CLUSTER_MERGE_THRESHOLD": 0.8,
        "MAX_HISTORY": 1000,
    },
    "TIMING": {
        "ECO_RESONANCE_TIMEOUT_SEC": 180,  # СЕКУНДЫ
        "FORESIGHT_PROCESSING_MS": 100,
        "CLUSTER_MERGE_COOLDOWN_MS": 500,
    },
}

ARCHITECTURAL_VARIANTS = {
    "LIGHTWEIGHT_MODE": {"MAX_HISTORY": 100, "MAX_PREDICTION_DEPTH": 2},
    "RESEARCH_MODE": {"MAX_HISTORY": 10000, "MAX_PREDICTION_DEPTH": 10},
}


# ============================================================
# ТИПЫ ДАННЫХ (TypedDict наверху)
# ============================================================

class IntentVector(TypedDict):
    """Вектор интенций из системы DS24"""
    source_id: str
    values: List[float]
    timestamp: float


class EmotionState(TypedDict):
    """Эмоциональное состояние из emotional_weave"""
    resonance_level: float
    stability_index: float
    hsbi: float
    timestamp: float


class ForesightDelta(TypedDict):
    """Дельта предсказания от нейро-интеграции"""
    predicted_outcome: str
    confidence: float
    correction_vector: List[float]
    timestamp: float


class ResonanceSignal(TypedDict):
    """Резонансный сигнал активации"""
    frequency: float
    amplitude: float
    coherence: float
    timestamp: float


class MeaningSignal(TypedDict):
    """Смысловой сигнал для распространения"""
    context_id: str
    semantic_vector: List[float]
    resonance: float
    timestamp: float


# ============================================================
# ПРОТОКОЛЫ
# ============================================================

class ICircuitBreaker(Protocol):
    """Протокол автомата защиты от сбоев"""
    def check(self) -> bool: ...
    def record_failure(self) -> None: ...
    def reset(self) -> None: ...


class IEventBus(Protocol):
    """Протокол шины событий DS24"""
    def publish(self, topic: str, payload: dict) -> None: ...
    def subscribe(self, topic: str, callback: callable) -> None: ...


class ISpiritCore(Protocol):
    """Протокол ядра духа для анкеровки смыслов"""
    def meaning_anchor(self, signal: dict) -> None: ...


class IEmotionOptimizer(Protocol):
    """Протокол оптимизатора эмоциональных состояний"""
    def stabilize(self, state: dict) -> dict: ...


class IRadarEngine(Protocol):
    """Протокол сканера интенций"""
    def scan_intent(self) -> dict: ...


class ISymbiosisCore(Protocol):
    """Протокол симбиозного ядра для приёма смыслов"""
    def receive_meaning(self, signal: dict) -> None: ...


class INeuroIntegration(Protocol):
    """Главный протокол нейро-интеграционного блока v1.3"""
    def process_foresight(self, intent_vector: IntentVector) -> ForesightDelta: ...
    def activate_resonance(self, emotion_state: EmotionState) -> ResonanceSignal: ...
    def propagate_meaning(self, signal: MeaningSignal) -> None: ...


# ============================================================
# УТИЛИТЫ: ВАЛИДАЦИЯ, ХЭШИРОВАНИЕ, САНИТАЙЗИНГ
# ============================================================

def sanitize_vector(values: List[float], 
                   max_len: int = 1000,
                   clip_range: tuple = (-10.0, 10.0),
                   pad_to_len: bool = True) -> List[float]:
    """
    Очистка вектора от NaN, Inf, нормализация длины
    
    Args:
        values: Входной вектор
        max_len: Максимальная длина (обрезается)
        clip_range: Диапазон значений
        pad_to_len: Дополнять нулями до max_len
    
    Returns:
        Очищенный вектор
    """
    if not values:
        return [0.0] if pad_to_len else []
    
    # Преобразование в numpy
    arr = np.array(values, dtype=np.float64)
    
    # Замена NaN/Inf
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Обрезка экстремальных значений
    arr = np.clip(arr, clip_range[0], clip_range[1])
    
    # Обрезка длины
    if len(arr) > max_len:
        arr = arr[:max_len]
    elif pad_to_len and len(arr) < max_len:
        # Дополнение нулями
        arr = np.pad(arr, (0, max_len - len(arr)), 'constant')
    
    return arr.tolist()


def deterministic_hash(data: Any, seed: int = 42) -> int:
    """Детерминированный хэш для воспроизводимости"""
    data_str = str(data).encode('utf-8')
    return int(hashlib.md5(data_str + str(seed).encode()).hexdigest()[:8], 16)


# ============================================================
# PRODUCTION CIRCUIT BREAKER
# ============================================================

@dataclass
class ProductionCircuitBreaker(ICircuitBreaker):
    """Production-ready автомат защиты с метриками"""
    failure_threshold: int = 5
    reset_timeout_sec: int = 30
    
    def __post_init__(self) -> None:
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = 0.0
        self._lock = threading.RLock()
        self.metrics = {
            "total_failures": 0,
            "total_resets": 0,
            "last_reset": None
        }
    
    def check(self) -> bool:
        """Проверка состояния с авто-сбросом"""
        with self._lock:
            # Авто-сброс после таймаута
            if (self.state == "OPEN" and 
                time.time() - self.last_failure_time > self.reset_timeout_sec):
                self.reset()
            
            return self.state == "CLOSED"
    
    def record_failure(self) -> None:
        """Регистрация сбоя"""
        with self._lock:
            self.failure_count += 1
            self.metrics["total_failures"] += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                warnings.warn(
                    f"CircuitBreaker открыт после {self.failure_count} сбоев",
                    RuntimeWarning
                )
    
    def reset(self) -> None:
        """Сброс состояния"""
        with self._lock:
            self.state = "CLOSED"
            self.failure_count = 0
            self.metrics["total_resets"] += 1
            self.metrics["last_reset"] = datetime.now().isoformat()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Метрики автомата"""
        with self._lock:
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "last_failure": self.last_failure_time,
                **self.metrics
            }


# ============================================================
# КОЛЬЦЕВЫЕ АНСАМБЛИ (ОПТИМИЗИРОВАННЫЕ)
# ============================================================

class WaveType(Enum):
    """Типы нейронных волн по Бехтеревой"""
    ALPHA = "alpha"      # 8-12 Гц
    BETA = "beta"        # 12-30 Гц
    THETA = "theta"      # 4-8 Гц
    GAMMA = "gamma"      # 30-100 Гц
    DELTA = "delta"      # 0.5-4 Гц


@dataclass
class RingNeuralEnsemble:
    """Кольцевой нейронный ансамбль - основная единица по Бехтеревой"""
    ensemble_id: str
    neuron_count: int = 100
    plasticity_factor: float = 0.1
    
    def __post_init__(self) -> None:
        self.phases = np.random.rand(self.neuron_count) * 2 * np.pi
        self.amplitudes = np.ones(self.neuron_count) * 0.8
        self.frequencies = np.full(self.neuron_count, 10.0)
        
        self.connections = self._create_ring_connections()
        self.wave_type = WaveType.ALPHA
        self.sync_level = 0.0
        self.last_update = time.time()
        self.phase_history = deque(maxlen=100)
        self._update_count = 0
        
    def _create_ring_connections(self) -> np.ndarray:
        """Создание кольцевой матрицы связей"""
        n = self.neuron_count
        conn = np.zeros((n, n))
        i = np.arange(n)
        conn[i, (i + 1) % n] = 0.8
        conn[i, (i - 1) % n] = 0.8
        np.fill_diagonal(conn, 1.0)
        return conn
    
    def update_phase(self, dt: float = 0.01) -> np.ndarray:
        """Обновление фаз нейронов с метриками"""
        self.phases += self.frequencies * dt
        phase_diffs = np.roll(self.phases, 1) - self.phases
        coupling = 0.1 * np.sin(phase_diffs)
        self.phases += coupling
        self.phases %= (2 * np.pi)
        
        self.phase_history.append(self.phases.copy())
        self._update_count += 1
        self.last_update = time.time()
        
        return self.amplitudes * np.sin(self.phases)
    
    def synchronize_wave(self, target_wave: WaveType) -> float:
        """Синхронизация на целевую волну"""
        self.wave_type = target_wave
        base_freqs = {
            WaveType.GAMMA: 40.0,
            WaveType.BETA: 20.0,
            WaveType.ALPHA: 10.0,
            WaveType.THETA: 6.0,
            WaveType.DELTA: 2.0
        }
        target_freq = base_freqs.get(target_wave, 10.0)
        freq_diff = target_freq - self.frequencies
        self.frequencies += 0.1 * freq_diff
        
        phase_std = np.std(self.phases)
        self.sync_level = 1.0 / (1.0 + phase_std)
        
        return self.sync_level
    
    def calculate_phase_coherence(self) -> float:
        """Вычисление когерентности фаз (order parameter Kuramoto)"""
        if len(self.phase_history) < 2:
            return 0.0
        
        recent_phases = np.array(list(self.phase_history)[-10:])
        if recent_phases.size == 0:
            return 0.0
        
        complex_phases = np.exp(1j * recent_phases)
        mean_complex = np.mean(complex_phases, axis=1)
        order_param = np.abs(np.mean(mean_complex))
        
        return float(order_param)
    
    def calculate_neuro_stability_index(self) -> float:
        """Композитная метрика стабильности нейроансамбля"""
        coherence = self.calculate_phase_coherence()
        phase_std = np.std(self.phases)
        amp_var = np.var(self.amplitudes)
        
        norm_coherence = coherence
        norm_phase_std = 1.0 / (1.0 + phase_std)
        norm_amp_var = 1.0 / (1.0 + amp_var)
        
        w1, w2, w3 = 0.6, 0.3, 0.1
        stability = (w1 * norm_coherence + 
                    w2 * norm_phase_std + 
                    w3 * norm_amp_var)
        
        return float(stability)
    
    def adapt_topology(self, activation_values: np.ndarray, 
                      threshold: float = 0.5,
                      deterministic: bool = True) -> np.ndarray:
        """
        Адаптация топологии на основе значений активации
        
        Args:
            activation_values: Вектор значений активации
            threshold: Порог для активации
            deterministic: Детерминированный режим
        
        Returns:
            Обновлённая матрица связей
        """
        n = self.neuron_count
        
        # Явное преобразование к булевой маске
        if deterministic:
            activation_mask = activation_values > threshold
        else:
            # RESEARCH режим: стохастическая активация
            activation_mask = np.random.rand(n) > (1 - np.clip(activation_values, 0, 1))
        
        if np.any(activation_mask):
            self.connections[activation_mask, :] *= (1 + self.plasticity_factor)
            self.connections[:, activation_mask] *= (1 + self.plasticity_factor)
        
        # Ослабление неактивных
        inactive_mask = ~activation_mask
        if np.any(inactive_mask):
            self.connections[inactive_mask, :] *= (1 - self.plasticity_factor * 0.3)
            self.connections[:, inactive_mask] *= (1 - self.plasticity_factor * 0.3)
        
        # Нормализация
        np.fill_diagonal(self.connections, 1.0)
        self.connections = (self.connections + self.connections.T) / 2
        self.connections = np.clip(self.connections, 0.1, 1.0)
        
        return self.connections
    
    def to_dict(self) -> Dict[str, Any]:
        """Диагностическая информация ансамбля"""
        return {
            "id": self.ensemble_id,
            "neurons": self.neuron_count,
            "wave": self.wave_type.value,
            "sync": round(self.sync_level, 3),
            "coherence": round(self.calculate_phase_coherence(), 3),
            "stability": round(self.calculate_neuro_stability_index(), 3),
            "mean_freq": round(float(np.mean(self.frequencies)), 1),
            "updates": self._update_count
        }


# ============================================================
# ДИНАМИЧЕСКИЙ ВОЛНОВОЙ СИНХРОНИЗАТОР (PRODUCTION 10/10)
# ============================================================

@dataclass
class DynamicWaveSynchronizer:
    """Production-синхронизатор с управлением нагрузкой и метриками"""
    bus: IEventBus
    circuit_breaker: ICircuitBreaker
    update_interval: float = 0.1
    
    def __post_init__(self) -> None:
        self.ensembles: Dict[str, RingNeuralEnsemble] = {}
        self.global_wave = WaveType.ALPHA
        
        # Throttle для публикации метрик (каждые 5 секунд)
        self._last_metrics_bucket: Optional[int] = None
        
        # Потокобезопасный кэш когерентности
        self._coherence_cache: Dict[str, float] = {}
        self._cache_lock = threading.RLock()
        
        # Метрики производительности с окном 5 минут
        self._metrics = {
            "loop_lag_ms": deque(maxlen=100),
            "exception_timestamps": deque(maxlen=1000),
            "avg_cycle_time_ms": 0.0,
            "total_cycles": 0,
        }
        
        # Управление жизненным циклом
        self._stop_event = threading.Event()
        self._drain_on_stop = True
        self._thread: Optional[threading.Thread] = None
        
        self._start_sync_thread()
    
    def _start_sync_thread(self) -> None:
        """Запуск управляемого потока синхронизации"""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sync_loop,
            name="WaveSynchronizer",
            daemon=True
        )
        self._thread.start()
        print(f"🌀 [WaveSynchronizer] Started (interval={self.update_interval}s)")
    
    def _sync_loop(self) -> None:
        """Основной цикл синхронизации с метриками производительности"""
        while not self._stop_event.is_set():
            cycle_start = time.time()
            
            try:
                if not self.circuit_breaker.check():
                    time.sleep(self.update_interval)
                    continue
                
                # 1. Обновление фаз всех ансамблей
                for ensemble in self.ensembles.values():
                    ensemble.update_phase(self.update_interval)
                
                # 2. ОБНОВЛЕНИЕ КЭША КОГЕРЕНТНОСТИ ДЛЯ ВСЕХ АНСАМБЛЕЙ
                with self._cache_lock:
                    for eid, ensemble in self.ensembles.items():
                        coherence = ensemble.calculate_phase_coherence()
                        self._coherence_cache[eid] = coherence
                
                # 3. Throttle публикации метрик
                current_bucket = int(time.time()) // 5
                if current_bucket != self._last_metrics_bucket:
                    self._last_metrics_bucket = current_bucket
                    self._publish_metrics()
                
                # 4. Метрики производительности
                cycle_time = (time.time() - cycle_start) * 1000
                self._metrics["loop_lag_ms"].append(cycle_time)
                if self._metrics["loop_lag_ms"]:
                    self._metrics["avg_cycle_time_ms"] = np.mean(self._metrics["loop_lag_ms"])
                self._metrics["total_cycles"] += 1
                
                # 5. Контроль времени цикла
                elapsed = time.time() - cycle_start
                sleep_time = max(0, self.update_interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                # Регистрация исключения с таймстампом
                self.circuit_breaker.record_failure()
                self._metrics["exception_timestamps"].append(time.time())
                
                # Публикация ошибки
                self.bus.publish("synchronizer.error", {
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Экспоненциальный backoff
                error_count = self._get_exceptions_last_5min()
                backoff = min(5.0, 0.1 * (2 ** min(error_count, 5)))
                time.sleep(backoff)
        
        # Graceful shutdown: drain events
        if self._drain_on_stop:
            self._publish_final_metrics()
    
    def _get_exceptions_last_5min(self) -> int:
        """Количество исключений за последние 5 минут"""
        cutoff = time.time() - 300  # 5 минут
        count = sum(1 for ts in self._metrics["exception_timestamps"] if ts > cutoff)
        return count
    
    def _publish_metrics(self) -> None:
        """Публикация метрик с throttle"""
        if not self.ensembles:
            return
        
        global_coherence = self.get_coherence_index()
        neuro_stability = self.calculate_neuro_stability_index()
        
        with self._cache_lock:
            cache_snapshot = self._coherence_cache.copy()
        
        # Подготовка метрик производительности
        loop_lag_list = list(self._metrics["loop_lag_ms"])
        loop_lag_p95 = np.percentile(loop_lag_list, 95) if loop_lag_list else 0.0
        
        payload = {
            "global_coherence": round(global_coherence, 4),
            "neuro_stability_index": round(neuro_stability, 4),
            "ensemble_coherences": cache_snapshot,
            "active_ensembles": len(self.ensembles),
            "wave_type": self.global_wave.value,
            "performance": {
                "avg_cycle_time_ms": round(self._metrics["avg_cycle_time_ms"], 2),
                "loop_lag_p95": round(loop_lag_p95, 2),
                "total_cycles": self._metrics["total_cycles"],
                "exceptions_last_5min": self._get_exceptions_last_5min()
            },
            "timestamp": datetime.now().isoformat()
        }
        
        self.bus.publish("bechtereva.metrics", payload)
    
    def _publish_final_metrics(self) -> None:
        """Публикация финальных метрик при shutdown"""
        self.bus.publish("synchronizer.stopped", {
            "final_coherence": round(self.get_coherence_index(), 4),
            "ensembles_count": len(self.ensembles),
            "total_cycles": self._metrics["total_cycles"],
            "exceptions_last_5min": self._get_exceptions_last_5min(),
            "timestamp": datetime.now().isoformat()
        })
    
    def calculate_neuro_stability_index(self) -> float:
        """Композитный индекс нейростабильности (публичный метод)"""
        if not self.ensembles:
            return 0.0
        
        stability_values = []
        for ensemble in self.ensembles.values():
            stability_values.append(ensemble.calculate_neuro_stability_index())
        
        return float(np.mean(stability_values))
    
    def register_ensemble(self, ensemble: RingNeuralEnsemble) -> None:
        """Регистрация ансамбля в системе"""
        self.ensembles[ensemble.ensemble_id] = ensemble
        
        with self._cache_lock:
            self._coherence_cache[ensemble.ensemble_id] = 0.0
        
        self.bus.publish("ensemble.registered", {
            "id": ensemble.ensemble_id,
            "size": ensemble.neuron_count,
            "wave_type": self.global_wave.value,
            "timestamp": datetime.now().isoformat()
        })
    
    def synchronize_all(self, target_wave: WaveType) -> Dict[str, float]:
        """Синхронизация всех ансамблей на целевую волну"""
        if not self.circuit_breaker.check():
            return {}
        
        self.global_wave = target_wave
        results = {}
        
        for eid, ensemble in self.ensembles.items():
            sync_level = ensemble.synchronize_wave(target_wave)
            results[eid] = sync_level
        
        self.bus.publish("wave.synchronized", {
            "wave_type": target_wave.value,
            "results": results,
            "global_stability": round(self.calculate_neuro_stability_index(), 3),
            "timestamp": datetime.now().isoformat()
        })
        
        return results
    
    def get_coherence_index(self) -> float:
        """Получение глобальной когерентности (потокобезопасно)"""
        with self._cache_lock:
            if not self._coherence_cache:
                return 0.0
            coherences = list(self._coherence_cache.values())
        
        return float(np.mean(coherences))
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Метрики производительности синхронизатора"""
        return {
            "update_interval": self.update_interval,
            "active_ensembles": len(self.ensembles),
            "performance": {
                "avg_cycle_time_ms": round(self._metrics["avg_cycle_time_ms"], 2),
                "total_cycles": self._metrics["total_cycles"],
                "exceptions_last_5min": self._get_exceptions_last_5min(),
                "exception_timestamps_count": len(self._metrics["exception_timestamps"])
            },
            "circuit_breaker": self.circuit_breaker.get_metrics()
        }
    
    def stop(self, drain: bool = True) -> None:
        """Грациозная остановка с drain событий"""
        self._drain_on_stop = drain
        self._stop_event.set()
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
            if self._thread.is_alive():
                warnings.warn("WaveSynchronizer thread did not stop gracefully")
        
        print("🌀 [WaveSynchronizer] Stopped")    

# ============================================================
# ГЛАВНЫЙ КЛАСС: PRODUCTION 10/10
# ============================================================

class BechterevaNeuroCore(INeuroIntegration):
    """
    Production-реализация нейро-интеграционного блока v1.5
    Полное соответствие INeuroIntegration протоколу DS24
    """
    
    def __init__(self,
                 bus: IEventBus,
                 radar: IRadarEngine,
                 emotion_optimizer: IEmotionOptimizer,
                 spirit_core: ISpiritCore,
                 symbiosis_core: ISymbiosisCore,
                 circuit_breaker_factory: callable = None,
                 mode: str = "STANDARD",
                 update_interval: float = 0.1):
        """
        Инициализация нейро-интеграционного ядра
        
        Args:
            bus: Шина событий DS24
            radar: Сканер интенций
            emotion_optimizer: Оптимизатор эмоций
            spirit_core: Ядро духа для анкеровки смыслов
            symbiosis_core: Симбиозное ядро
            circuit_breaker_factory: Фабрика автоматов защиты
            mode: Режим работы (STANDARD, LIGHTWEIGHT, RESEARCH)
            update_interval: Интервал обновления синхронизатора
        """
        # Валидация зависимостей
        if not all([bus, radar, emotion_optimizer, spirit_core, symbiosis_core]):
            raise ValueError("Все зависимости должны быть предоставлены")
        
        self.bus = bus
        self.radar = radar
        self.emotion_optimizer = emotion_optimizer
        self.spirit_core = spirit_core
        self.symbiosis_core = symbiosis_core
        
        # Режим работы
        self.mode = mode.upper()
        if self.mode not in ["STANDARD", "LIGHTWEIGHT", "RESEARCH"]:
            raise ValueError(f"Недопустимый режим: {mode}. Допустимо: STANDARD, LIGHTWEIGHT, RESEARCH")
        
        # Конфигурация
        variant = ARCHITECTURAL_VARIANTS.get(self.mode + "_MODE", {})
        self.limits = {**ARCHITECTURAL_CONSTRAINTS["LIMITS"], **variant}
        self.timing = ARCHITECTURAL_CONSTRAINTS["TIMING"]
        
        # Детерминированность
        self.deterministic = (self.mode != "RESEARCH")
        self._random_seed = 42 if self.deterministic else int(time.time())
        
        # Circuit breaker
        self.circuit_breaker = (
            circuit_breaker_factory() 
            if circuit_breaker_factory 
            else ProductionCircuitBreaker()
        )
        
        # Cooldown для резонанса (неблокирующий)
        self._resonance_cooldown_until = 0.0
        
        # Создание синхронизатора
        self.wave_synchronizer = DynamicWaveSynchronizer(
            bus=bus,
            circuit_breaker=self.circuit_breaker,
            update_interval=update_interval
        )
        
        # Ансамбли
        self.ensembles = {
            "anticipation": RingNeuralEnsemble("anticipation", 150),
            "resonance": RingNeuralEnsemble("resonance", 120),
            "meaning": RingNeuralEnsemble("meaning", 100),
            "integration": RingNeuralEnsemble("integration", 200),
        }
        
        for eid, ensemble in self.ensembles.items():
            self.wave_synchronizer.register_ensemble(ensemble)
        
        # История операций
        self.foresight_history: List[ForesightDelta] = []
        self.resonance_history: List[ResonanceSignal] = []
        self.meaning_history: List[MeaningSignal] = []
        
        self._operation_count = 0
        self._last_foresight_time = 0.0
        self._init_time = time.time()
        
        print(f"🧠 [Bechtereva v1.5.0] Initialized | Mode={self.mode} | "
              f"Deterministic={self.deterministic} | "
              f"UpdateInterval={update_interval}s")
    
    # ========================================================
    # РЕАЛИЗАЦИЯ INEUROINTEGRATION ПРОТОКОЛА
    # ========================================================
    
    def process_foresight(self, intent_vector: IntentVector) -> ForesightDelta:
        """
        Обработка предвосхищения на основе интенций
        Соответствует методу из архитектуры v1.3
        
        Args:
            intent_vector: Вектор интенций из системы DS24
        
        Returns:
            Дельта предсказания с корректирующим вектором
        
        Raises:
            RuntimeError: При сбое CircuitBreaker или обработки
        """
        if not self.circuit_breaker.check():
            raise RuntimeError("CircuitBreaker: операции приостановлены")
        
        # Валидация входных данных
        validated_values = sanitize_vector(
            intent_vector.get("values", []),
            max_len=100,
            clip_range=(-1.0, 1.0),
            pad_to_len=False  # Сохраняем оригинальную длину
        )
        
        # Rate limiting
        current_time = time.time()
        processing_time = self.timing["FORESIGHT_PROCESSING_MS"] / 1000.0
        
        if current_time - self._last_foresight_time < processing_time:
            time.sleep(processing_time * 0.5)
        
        self._last_foresight_time = current_time
        
        try:
            # Синхронизация ансамбля предвосхищения
            self.ensembles["anticipation"].synchronize_wave(WaveType.BETA)
            
            # Сканирование интенций
            scanned_data = self.radar.scan_intent()
            
            # Детерминированная/стохастическая логика
            if self.deterministic:
                seed = deterministic_hash(scanned_data, self._random_seed)
                np.random.seed(seed)
                noise = np.random.randn(len(validated_values)) * 0.02
                confidence = 0.7 + (seed % 1000) / 5000.0
            else:
                noise = np.random.randn(len(validated_values)) * 0.05
                confidence = 0.7 + np.random.rand() * 0.25
            
            # Формирование корректирующего вектора
            correction_vector = (np.array(validated_values) * 0.95 + noise).tolist()
            correction_vector = sanitize_vector(correction_vector, 
                                              clip_range=(-1.0, 1.0))
            
            # Создание delta-предсказания
            delta: ForesightDelta = {
                "predicted_outcome": f"shift-{deterministic_hash(scanned_data) % 1000000:06d}",
                "confidence": min(0.99, max(0.1, confidence)),
                "correction_vector": correction_vector,
                "timestamp": current_time
            }
            
            # Сохранение в историю
            self.foresight_history.append(delta)
            self.foresight_history = self.foresight_history[-self.limits["MAX_HISTORY"]:]
            
            # Публикация события
            self.bus.publish("foresight.delta", {
                **delta,
                "mode": self.mode,
                "deterministic": self.deterministic,
                "source_id": intent_vector.get("source_id", "unknown")
            })
            
            # Адаптация топологии на основе уверенности
            if delta["confidence"] > 0.8:
                activation = np.array(correction_vector[:150])
                if len(activation) < 150:
                    activation = np.pad(activation, (0, 150 - len(activation)), 'constant')
                activation = activation.astype(float)
                
                self.ensembles["anticipation"].adapt_topology(
                    activation_values=activation,
                    threshold=0.5,
                    deterministic=self.deterministic
                )
            
            self._operation_count += 1
            return delta
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.bus.publish("foresight.error", {
                "error": str(e),
                "intent_vector": intent_vector.get("source_id", "unknown"),
                "timestamp": datetime.now().isoformat()
            })
            raise RuntimeError(f"Foresight processing failed: {e}")
    
    def activate_resonance(self, emotion_state: EmotionState) -> ResonanceSignal:
        """
        Активация резонанса на основе эмоционального состояния
        Соответствует методу из архитектуры v1.3
        
        Args:
            emotion_state: Эмоциональное состояние из системы
        
        Returns:
            Резонансный сигнал активации
        
        Raises:
            RuntimeError: При сбое CircuitBreaker или cooldown
        """
        if not self.circuit_breaker.check():
            raise RuntimeError("CircuitBreaker: резонанс невозможен")
        
        # Проверка cooldown
        current_time = time.time()
        if current_time < self._resonance_cooldown_until:
            wait_time = self._resonance_cooldown_until - current_time
            raise RuntimeError(
                f"Resonance cooldown active. Wait {wait_time:.1f}s"
            )
        
        try:
            # Валидация эмоционального состояния
            resonance_level = max(0.0, min(1.0, 
                emotion_state.get("resonance_level", 0.5)))
            stability_index = max(0.0, min(1.0,
                emotion_state.get("stability_index", 0.5)))
            
            # Синхронизация на гамма-волну для пикового резонанса
            self.wave_synchronizer.synchronize_all(WaveType.GAMMA)
            
            # Стабилизация эмоционального состояния
            stabilized = self.emotion_optimizer.stabilize({
                "resonance_level": resonance_level,
                "stability_index": stability_index,
                "hsbi": emotion_state.get("hsbi", 0.5),
                "timestamp": emotion_state.get("timestamp", time.time())
            })
            
            # Активация резонансного ансамбля
            resonance_ens = self.ensembles["resonance"]
            resonance_ens.synchronize_wave(WaveType.GAMMA)
            
            # Вычисление параметров резонансного сигнала
            frequency = stabilized.get("resonance_level", 0.5) * 20.0 + 10.0
            amplitude = stabilized.get("stability_index", 0.5)
            coherence = self.wave_synchronizer.get_coherence_index()
            
            signal: ResonanceSignal = {
                "frequency": round(frequency, 2),
                "amplitude": round(amplitude, 3),
                "coherence": round(coherence, 3),
                "timestamp": current_time
            }
            
            # Сохранение в историю
            self.resonance_history.append(signal)
            self.resonance_history = self.resonance_history[-self.limits["MAX_HISTORY"]:]
            
            # Публикация события
            self.bus.publish("eco.resonance.activate", {
                **signal,
                "emotion_state": {
                    "resonance_level": resonance_level,
                    "stability_index": stability_index,
                    "hsbi": emotion_state.get("hsbi", 0.5)
                },
                "mode": self.mode
            })
            
            # Установка cooldown (НЕ БЛОКИРУЕМ ТЕКУЩИЙ ПОТОК)
            self._resonance_cooldown_until = current_time + self.timing["ECO_RESONANCE_TIMEOUT_SEC"]
            
            self._operation_count += 1
            return signal
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.bus.publish("resonance.error", {
                "error": str(e),
                "emotion_state": emotion_state,
                "timestamp": datetime.now().isoformat()
            })
            raise RuntimeError(f"Resonance activation failed: {e}")
    
    def propagate_meaning(self, signal: MeaningSignal) -> None:
        """
        Распространение смыслового сигнала
        Соответствует методу из архитектуры v1.3
        
        Args:
            signal: Смысловой сигнал для распространения
        """
        if not self.circuit_breaker.check():
            return
        
        # Валидация резонанса
        if signal.get("resonance", 0) < self.limits["MIN_SEMANTIC_RESONANCE"]:
            self.bus.publish("meaning.rejected", {
                "reason": "low_resonance",
                "resonance": signal.get("resonance", 0),
                "threshold": self.limits["MIN_SEMANTIC_RESONANCE"],
                "context_id": signal.get("context_id", "unknown"),
                "timestamp": datetime.now().isoformat()
            })
            return
        
        # Валидация вектора (без padding для сохранения статистики)
        semantic_vector = sanitize_vector(
            signal.get("semantic_vector", []),
            max_len=200,
            clip_range=(-1.0, 1.0),
            pad_to_len=False
        )
        
        validated_signal = signal.copy()
        validated_signal["semantic_vector"] = semantic_vector
        
        try:
            # Синхронизация на альфа-волну для смысловой обработки
            self.ensembles["meaning"].synchronize_wave(WaveType.ALPHA)
            
            # Анкеровка смысла
            self.spirit_core.meaning_anchor(validated_signal)
            
            # Передача симбиозному ядру
            self.symbiosis_core.receive_meaning(validated_signal)
            
            # Сохранение в историю
            self.meaning_history.append(validated_signal)
            self.meaning_history = self.meaning_history[-self.limits["MAX_HISTORY"]:]
            
            # Публикация события
            self.bus.publish("meaning.signal", validated_signal)
            
            # Адаптация интеграционного ансамбля
            if semantic_vector:
                activation = np.array(semantic_vector[:200])
                if len(activation) < 200:
                    activation = np.pad(activation, (0, 200 - len(activation)), 'constant')
                activation = activation.astype(float)
                
                # Динамический порог на основе персентиля
                threshold = (np.percentile(activation, 50) 
                           if len(activation) > 0 else 0.5)
                
                self.ensembles["integration"].adapt_topology(
                    activation_values=activation,
                    threshold=threshold,
                    deterministic=self.deterministic
                )
            
            self._operation_count += 1
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.bus.publish("meaning.propagation.error", {
                "error": str(e),
                "context_id": signal.get("context_id", "unknown"),
                "timestamp": datetime.now().isoformat()
            })
    
    # ========================================================
    # ДОПОЛНИТЕЛЬНЫЕ МЕТОДЫ
    # ========================================================
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Полная диагностика модуля
        
        Returns:
            Словарь с полной диагностической информацией
        """
        coherence = self.wave_synchronizer.get_coherence_index()
        performance = self.wave_synchronizer.get_performance_metrics()
        neuro_stability = self.wave_synchronizer.calculate_neuro_stability_index()
        
        return {
            "module": "BechterevaNeuroCore",
            "version": "1.5.0",
            "mode": self.mode,
            "deterministic": self.deterministic,
            "operations": self._operation_count,
            "uptime_seconds": round(time.time() - self._init_time, 1),
            "circuit_breaker": self.circuit_breaker.get_metrics(),
            "coherence": round(coherence, 4),
            "neuro_stability": round(neuro_stability, 4),
            "ensembles": {
                eid: ens.to_dict() for eid, ens in self.ensembles.items()
            },
            "histories": {
                "foresight": len(self.foresight_history),
                "resonance": len(self.resonance_history),
                "meaning": len(self.meaning_history)
            },
            "resonance_cooldown": {
                "active": time.time() < self._resonance_cooldown_until,
                "until": self._resonance_cooldown_until,
                "seconds_remaining": max(0, self._resonance_cooldown_until - time.time())
            },
            "performance": performance,
            "limits": self.limits,
            "timestamp": datetime.now().isoformat()
        }
    
    def switch_wave_mode(self, wave_type: WaveType) -> Dict[str, float]:
        """
        Переключение волнового режима всех ансамблей
        
        Args:
            wave_type: Целевой тип волны
        
        Returns:
            Результаты синхронизации по ансамблям
        """
        return self.wave_synchronizer.synchronize_all(wave_type)
    
    def get_resonance_cooldown_status(self) -> Dict[str, Any]:
        """
        Статус cooldown для резонансной активации
        
        Returns:
            Информация о cooldown
        """
        current_time = time.time()
        is_active = current_time < self._resonance_cooldown_until
        remaining = max(0, self._resonance_cooldown_until - current_time)
        
        return {
            "active": is_active,
            "until": self._resonance_cooldown_until,
            "remaining_seconds": round(remaining, 1),
            "timeout_seconds": self.timing["ECO_RESONANCE_TIMEOUT_SEC"],
            "timestamp": current_time
        }
    
    def reset_resonance_cooldown(self) -> None:
        """Сброс cooldown для резонансной активации"""
        self._resonance_cooldown_until = 0.0
        self.bus.publish("resonance.cooldown.reset", {
            "timestamp": datetime.now().isoformat()
        })
    
    def shutdown(self) -> None:
        """
        Грациозное выключение модуля
        Останавливает все фоновые потоки и публикует финальные метрики
        """
        print("🧠 [Bechtereva] Starting graceful shutdown...")
        
        # Остановка синхронизатора
        self.wave_synchronizer.stop(drain=True)
        
        # Публикация финальной диагностики
        diagnostics = self.get_diagnostics()
        self.bus.publish("bechtereva.shutdown", diagnostics)
        
        print(f"🧠 [Bechtereva] Shutdown completed | "
              f"Operations={self._operation_count} | "
              f"Uptime={round(time.time() - self._init_time, 1)}s")
    
    def connect_to_sephira(self, sephira_name: str, module_name: str) -> bool:
        """
        Подключение модуля к сефиротическому узлу
        
        Args:
            sephira_name: Название сефиры
            module_name: Название модуля для подключения
        
        Returns:
            True если запрос отправлен успешно
        """
        self.bus.publish("sephira.connection.request", {
            "sephira": sephira_name,
            "module": module_name,
            "neuro_core": "Bechtereva",
            "mode": self.mode,
            "deterministic": self.deterministic,
            "timestamp": datetime.now().isoformat()
        })
        return True


# ============================================================
# ФАБРИКА ДЛЯ ИНТЕГРАЦИИ С DS24
# ============================================================

def create_bechtereva_core(
    bus: IEventBus,
    radar: IRadarEngine,
    emotion_optimizer: IEmotionOptimizer,
    spirit_core: ISpiritCore,
    symbiosis_core: ISymbiosisCore,
    circuit_breaker_factory: callable = None,
    mode: str = "STANDARD",
    update_interval: float = 0.1
) -> BechterevaNeuroCore:
    """
    Фабрика для создания production-версии нейро-интеграционного ядра
    
    Args:
        bus: Шина событий DS24
        radar: Сканер интенций
        emotion_optimizer: Оптимизатор эмоций
        spirit_core: Ядро духа для анкеровки смыслов
        symbiosis_core: Симбиозное ядро
        circuit_breaker_factory: Фабрика автоматов защиты
        mode: Режим работы (STANDARD, LIGHTWEIGHT, RESEARCH)
        update_interval: Интервал обновления синхронизатора
    
    Returns:
        Готовый к работе экземпляр BechterevaNeuroCore
    
    Raises:
        ValueError: При недопустимом режиме или отсутствии зависимостей
    """
    # Валидация режима
    valid_modes = ["STANDARD", "LIGHTWEIGHT", "RESEARCH"]
    if mode.upper() not in valid_modes:
        raise ValueError(
            f"Недопустимый режим: {mode}. Допустимо: {', '.join(valid_modes)}"
        )
    
    return BechterevaNeuroCore(
        bus=bus,
        radar=radar,
        emotion_optimizer=emotion_optimizer,
        spirit_core=spirit_core,
        symbiosis_core=symbiosis_core,
        circuit_breaker_factory=circuit_breaker_factory,
        mode=mode,
        update_interval=update_interval
    )


# ============================================================
# ЭКСПОРТ ДЛЯ ИСПОЛЬЗОВАНИЯ В ДРУГИХ МОДУЛЯХ
# ============================================================

__all__ = [
    # Главные классы
    'BechterevaNeuroCore',
    'RingNeuralEnsemble',
    'DynamicWaveSynchronizer',
    'ProductionCircuitBreaker',
    
    # Перечисления
    'WaveType',
    
    # Типы данных
    'IntentVector',
    'EmotionState',
    'ForesightDelta',
    'ResonanceSignal',
    'MeaningSignal',
    
    # Протоколы
    'INeuroIntegration',
    'ICircuitBreaker',
    'IEventBus',
    'ISpiritCore',
    'IEmotionOptimizer',
    'IRadarEngine',
    'ISymbiosisCore',
    
    # Утилиты
    'sanitize_vector',
    'deterministic_hash',
    
    # Фабрика
    'create_bechtereva_core',
    
    # Константы
    'ARCHITECTURAL_CONSTRAINTS',
    'ARCHITECTURAL_VARIANTS',
]


# ============================================================
# МИНИМАЛЬНЫЙ ТЕСТ ДЛЯ ПРОВЕРКИ ИМПОРТА
# ============================================================

if __name__ == "__main__":
    print("🧪 [Bechtereva Module Test]")
    print(f"✅ Module loaded successfully")
    print(f"✅ Version: 1.5.0")
    print(f"✅ Classes: {len(__all__)} available")
    print("✅ Ready for integration with DS24")
