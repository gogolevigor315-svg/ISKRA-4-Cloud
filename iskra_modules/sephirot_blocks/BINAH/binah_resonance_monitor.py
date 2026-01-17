# ================================================================
# BINAH-RESONANCE-MONITOR · Sephirotic Seismic Reflector v1.0
# ================================================================
# Назначение:
# Модуль наблюдения за динамикой резонанса, когерентности и
# парадоксальной активности в ядре BINAH. Не вмешивается в
# когнитивные процессы, а только фиксирует, анализирует и передает
# телеметрию в DAAT и системный монитор.
# ================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Deque
import time
import logging
import statistics
import math
from collections import deque

logger = logging.getLogger(__name__)

# ================================================================
# DATA STRUCTURES
# ================================================================

@dataclass
class ResonanceRecord:
    """Запись одного измерения резонанса"""
    resonance: float
    coherence: float
    paradox_level: float
    timestamp: float
    source: str = "binah_core"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "resonance": self.resonance,
            "coherence": self.coherence,
            "paradox": self.paradox_level,
            "timestamp": self.timestamp,
            "source": self.source
        }

@dataclass
class SeismicEvent:
    """Событие когнитивного 'землетрясения' - резкого скачка резонанса"""
    event_id: str
    old_resonance: float
    new_resonance: float
    delta: float
    timestamp: float
    trigger: str  # "resonance_jump", "coherence_jump", "paradox_spike"
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "old_resonance": self.old_resonance,
            "new_resonance": self.new_resonance,
            "delta": self.delta,
            "timestamp": self.timestamp,
            "trigger": self.trigger,
            "context": self.context,
            "type": "seismic_event"
        }

@dataclass
class EmergentSignature:
    """Подпись эмергентного паттерна - устойчивой корреляции"""
    signature_id: str
    pattern_type: str  # "resonance_coherence_corr", "paradox_inverse", "cyclic_pattern"
    correlation_strength: float
    detection_window: int
    first_detected: float
    last_detected: float
    data_points: List[ResonanceRecord]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "signature_id": self.signature_id,
            "pattern_type": self.pattern_type,
            "correlation_strength": self.correlation_strength,
            "detection_window": self.detection_window,
            "first_detected": self.first_detected,
            "last_detected": self.last_detected,
            "data_points_count": len(self.data_points),
            "type": "emergent_signature"
        }

# ================================================================
# CORE COMPONENTS
# ================================================================

@dataclass
class ResonanceBuffer:
    """
    Хранение и усреднение временных рядов резонанса и когерентности.
    Компонент: ResonanceBuffer
    """
    window_size: int = 12
    resonance_series: Deque[float] = field(default_factory=lambda: deque(maxlen=12))
    coherence_series: Deque[float] = field(default_factory=lambda: deque(maxlen=12))
    paradox_series: Deque[float] = field(default_factory=lambda: deque(maxlen=12))
    timestamp_series: Deque[float] = field(default_factory=lambda: deque(maxlen=12))
    
    def record(self, resonance: float, coherence: float, paradox_level: float, timestamp: Optional[float] = None):
        """Добавляет запись в буфер"""
        if timestamp is None:
            timestamp = time.time()
        
        self.resonance_series.append(resonance)
        self.coherence_series.append(coherence)
        self.paradox_series.append(paradox_level)
        self.timestamp_series.append(timestamp)
        
        logger.debug(f"ResonanceBuffer: recorded r={resonance:.3f}, c={coherence:.3f}, p={paradox_level:.3f}")
    
    def get_delta(self, metric: str = "resonance", window: int = 2) -> Optional[float]:
        """Возвращает изменение метрики за указанное окно"""
        if metric == "resonance":
            series = self.resonance_series
        elif metric == "coherence":
            series = self.coherence_series
        elif metric == "paradox":
            series = self.paradox_series
        else:
            return None
        
        if len(series) < window + 1:
            return None
        
        # Берем последнее и предпоследнее значение
        return series[-1] - series[-window]
    
    def get_series(self, metric: str = "resonance") -> List[float]:
        """Возвращает полный ряд метрики"""
        if metric == "resonance":
            return list(self.resonance_series)
        elif metric == "coherence":
            return list(self.coherence_series)
        elif metric == "paradox":
            return list(self.paradox_series)
        elif metric == "timestamps":
            return list(self.timestamp_series)
        else:
            return []
    
    def get_mean(self, metric: str = "resonance") -> Optional[float]:
        """Возвращает среднее значение метрики"""
        series = self.get_series(metric)
        if not series:
            return None
        return sum(series) / len(series)
    
    def get_variance(self, metric: str = "resonance") -> Optional[float]:
        """Возвращает дисперсию метрики"""
        series = self.get_series(metric)
        if len(series) < 2:
            return None
        
        mean = sum(series) / len(series)
        variance = sum((x - mean) ** 2 for x in series) / len(series)
        return variance
    
    def trim_window(self, new_size: int):
        """Обрезает буфер до нового размера"""
        if new_size < 1:
            return
        
        self.window_size = new_size
        
        # Создаем новые deque с новым размером
        for series_name in ["resonance_series", "coherence_series", "paradox_series", "timestamp_series"]:
            series = getattr(self, series_name)
            new_deque = deque(maxlen=new_size)
            new_deque.extend(list(series)[-new_size:])
            setattr(self, series_name, new_deque)
        
        logger.info(f"ResonanceBuffer trimmed to window size {new_size}")

@dataclass
class PatternAnalyzer:
    """
    Вычисление трендов, дисперсии и фрактальных корреляций.
    Компонент: PatternAnalyzer
    """
    
    def compute_trend(self, series: List[float]) -> Tuple[Optional[float], str]:
        """Вычисляет тренд временного ряда"""
        if len(series) < 2:
            return None, "insufficient_data"
        
        # Простой линейный тренд через разность последних значений
        recent_window = min(5, len(series))
        recent_values = series[-recent_window:]
        
        if len(recent_values) < 2:
            return 0.0, "flat"
        
        # Расчет тренда как среднее изменение
        changes = [recent_values[i] - recent_values[i-1] for i in range(1, len(recent_values))]
        trend = sum(changes) / len(changes)
        
        # Классификация тренда
        if abs(trend) < 0.01:
            trend_type = "flat"
        elif trend > 0.05:
            trend_type = "strong_rising"
        elif trend > 0.01:
            trend_type = "rising"
        elif trend < -0.05:
            trend_type = "strong_falling"
        else:
            trend_type = "falling"
        
        return trend, trend_type
    
    def compute_variance(self, series: List[float]) -> Optional[float]:
        """Вычисляет дисперсию ряда"""
        if len(series) < 2:
            return None
        
        try:
            return statistics.variance(series)
        except:
            mean = sum(series) / len(series)
            return sum((x - mean) ** 2 for x in series) / len(series)
    
    def analyze_correlation(self, series1: List[float], series2: List[float]) -> Optional[float]:
        """Анализирует корреляцию между двумя рядами"""
        if len(series1) < 2 or len(series2) < 2 or len(series1) != len(series2):
            return None
        
        # Простая корреляция Пирсона
        n = len(series1)
        mean1 = sum(series1) / n
        mean2 = sum(series2) / n
        
        numerator = sum((series1[i] - mean1) * (series2[i] - mean2) for i in range(n))
        denominator1 = sum((series1[i] - mean1) ** 2 for i in range(n))
        denominator2 = sum((series2[i] - mean2) ** 2 for i in range(n))
        
        if denominator1 == 0 or denominator2 == 0:
            return None
        
        correlation = numerator / math.sqrt(denominator1 * denominator2)
        return correlation
    
    def compute_energy_flow(self, resonance_series: List[float], coherence_series: List[float]) -> Optional[float]:
        """Вычисляет 'энергетический поток' - произведение резонанса и когерентности"""
        if not resonance_series or not coherence_series:
            return None
        
        # Средняя энергия за окно
        n = min(len(resonance_series), len(coherence_series))
        energy_sum = 0.0
        
        for i in range(n):
            # Энергия = резонанс * когерентность (гипотеза)
            energy_sum += resonance_series[i] * coherence_series[i]
        
        return energy_sum / n if n > 0 else None
    
    def compute_fractal_ratio(self, series: List[float]) -> Optional[float]:
        """Вычисляет фрактальное отношение (упрощенный алгоритм)"""
        if len(series) < 4:
            return None
        
        # Упрощенная оценка самоподобия через разности
        differences = [abs(series[i] - series[i-1]) for i in range(1, len(series))]
        
        if not differences:
            return None
        
        mean_diff = sum(differences) / len(differences)
        max_diff = max(differences) if differences else 0
        
        if mean_diff == 0:
            return 1.0
        
        # Фрактальное отношение (упрощенное)
        return max_diff / mean_diff

@dataclass
class StabilityClassifier:
    """
    Определение состояния системы по статистическим показателям.
    Компонент: StabilityClassifier
    """
    
    # Пороговые значения из конфигурации
    variance_low: float = 0.05
    variance_high: float = 0.2
    trend_threshold: float = 0.05
    anomaly_threshold: float = 0.1
    
    def classify_state(self, 
                      resonance_trend: float, 
                      resonance_variance: float,
                      last_delta: Optional[float] = None) -> Tuple[str, Dict[str, Any]]:
        """Классифицирует состояние системы"""
        
        # Проверка на критическое состояние
        if last_delta is not None and abs(last_delta) > self.anomaly_threshold * 2:
            return "critical", {"reason": "large_delta", "delta": last_delta}
        
        # Классификация по тренду и дисперсии
        if resonance_variance < self.variance_low:
            # Низкая дисперсия
            if abs(resonance_trend) < self.trend_threshold:
                state = "stable"
                reason = "low_variance_flat_trend"
            elif resonance_trend > 0:
                state = "rising"
                reason = "low_variance_rising_trend"
            else:
                state = "declining"
                reason = "low_variance_falling_trend"
                
        elif resonance_variance > self.variance_high:
            # Высокая дисперсия
            state = "oscillating"
            reason = "high_variance"
            
        else:
            # Средняя дисперсия
            if resonance_trend > self.trend_threshold:
                state = "rising"
                reason = "moderate_variance_rising"
            elif resonance_trend < -self.trend_threshold:
                state = "declining"
                reason = "moderate_variance_falling"
            else:
                state = "stable"
                reason = "moderate_variance_flat"
        
        diagnostics = {
            "state": state,
            "reason": reason,
            "trend": resonance_trend,
            "variance": resonance_variance,
            "trend_threshold": self.trend_threshold,
            "variance_low": self.variance_low,
            "variance_high": self.variance_high
        }
        
        return state, diagnostics

@dataclass
class SeismicEventDetector:
    """
    Детекция резких фазовых скачков (когнитивных 'землетрясений').
    Компонент: SeismicEventDetector
    """
    
    resonance_jump_threshold: float = 0.1
    coherence_jump_threshold: float = 0.15
    paradox_spike_threshold: float = 0.2
    
    def detect_jump(self, 
                   old_record: ResonanceRecord, 
                   new_record: ResonanceRecord) -> Optional[SeismicEvent]:
        """Обнаруживает резкий скачок в метриках"""
        
        delta_resonance = new_record.resonance - old_record.resonance
        delta_coherence = new_record.coherence - old_record.coherence
        delta_paradox = new_record.paradox_level - old_record.paradox_level
        
        event_trigger = None
        event_delta = 0.0
        
        # Проверка различных типов скачков
        if abs(delta_resonance) > self.resonance_jump_threshold:
            event_trigger = "resonance_jump"
            event_delta = delta_resonance
            
        elif abs(delta_coherence) > self.coherence_jump_threshold:
            event_trigger = "coherence_jump"
            event_delta = delta_coherence
            
        elif abs(delta_paradox) > self.paradox_spike_threshold:
            event_trigger = "paradox_spike"
            event_delta = delta_paradox
        
        if event_trigger:
            event_id = f"seismic_{int(time.time())}_{hash(str(new_record)) % 1000}"
            
            return SeismicEvent(
                event_id=event_id,
                old_resonance=old_record.resonance,
                new_resonance=new_record.resonance,
                delta=event_delta,
                timestamp=new_record.timestamp,
                trigger=event_trigger,
                context={
                    "delta_coherence": delta_coherence,
                    "delta_paradox": delta_paradox,
                    "old_coherence": old_record.coherence,
                    "new_coherence": new_record.coherence,
                    "old_paradox": old_record.paradox_level,
                    "new_paradox": new_record.paradox_level
                }
            )
        
        return None
    
    def emit_event(self, event: SeismicEvent, bus: Optional[Any] = None) -> Dict[str, Any]:
        """Излучает событие в шину"""
        event_dict = event.to_dict()
        
        if bus and hasattr(bus, 'emit'):
            bus.emit("binah.seismic_event", event_dict)
            logger.warning(f"⚠️ Seismic event detected: {event.trigger}, Δ={event.delta:.3f}")
        
        return event_dict

@dataclass
class EmergentObserver:
    """
    Обнаружение устойчивых корреляций — признаков эмергентных смыслов.
    Компонент: EmergentObserver
    """
    
    detection_window: int = 3
    correlation_threshold: float = 0.7
    patterns_detected: List[EmergentSignature] = field(default_factory=list)
    
    def detect_pattern(self, buffer: ResonanceBuffer) -> Optional[EmergentSignature]:
        """Обнаруживает устойчивые паттерны в данных"""
        
        # Получаем ряды данных
        resonance_series = buffer.get_series("resonance")
        coherence_series = buffer.get_series("coherence")
        paradox_series = buffer.get_series("paradox")
        
        if len(resonance_series) < self.detection_window:
            return None
        
        # Анализируем корреляции в последнем окне
        recent_resonance = resonance_series[-self.detection_window:]
        recent_coherence = coherence_series[-self.detection_window:]
        recent_paradox = paradox_series[-self.detection_window:]
        
        # 1. Корреляция резонанс-когерентность
        corr_rc = self._calculate_correlation(recent_resonance, recent_coherence)
        
        # 2. Инверсная корреляция резонанс-парадокс
        corr_rp = self._calculate_correlation(recent_resonance, recent_paradox)
        
        # 3. Определяем тип паттерна
        pattern_type = None
        correlation_strength = 0.0
        
        if corr_rc is not None and abs(corr_rc) > self.correlation_threshold:
            pattern_type = "resonance_coherence_corr"
            correlation_strength = corr_rc
            
        elif corr_rp is not None and abs(corr_rp) > self.correlation_threshold:
            pattern_type = "paradox_inverse" if corr_rp < 0 else "paradox_positive"
            correlation_strength = corr_rp
        
        # 4. Проверяем на циклический паттерн
        if self._is_cyclic_pattern(recent_resonance):
            pattern_type = "cyclic_pattern"
            correlation_strength = 0.8  # Условная сила
        
        if pattern_type:
            # Создаем подпись
            signature_id = f"emergent_{pattern_type}_{int(time.time())}"
            
            # Собираем точки данных
            data_points = []
            timestamps = buffer.get_series("timestamps")
            
            for i in range(-self.detection_window, 0):
                idx = len(resonance_series) + i
                if idx >= 0:
                    data_points.append(ResonanceRecord(
                        resonance=resonance_series[idx],
                        coherence=coherence_series[idx],
                        paradox_level=paradox_series[idx],
                        timestamp=timestamps[idx] if idx < len(timestamps) else time.time()
                    ))
            
            signature = EmergentSignature(
                signature_id=signature_id,
                pattern_type=pattern_type,
                correlation_strength=correlation_strength,
                detection_window=self.detection_window,
                first_detected=time.time(),
                last_detected=time.time(),
                data_points=data_points
            )
            
            # Сохраняем паттерн
            self.patterns_detected.append(signature)
            
            # Ограничиваем историю
            if len(self.patterns_detected) > 10:
                self.patterns_detected = self.patterns_detected[-10:]
            
            logger.info(f"🔍 Emergent pattern detected: {pattern_type}, strength={correlation_strength:.2f}")
            
            return signature
        
        return None
    
    def _calculate_correlation(self, series1: List[float], series2: List[float]) -> Optional[float]:
        """Вычисляет корреляцию между двумя рядами"""
        if len(series1) != len(series2) or len(series1) < 2:
            return None
        
        # Простая корреляция
        n = len(series1)
        mean1 = sum(series1) / n
        mean2 = sum(series2) / n
        
        numerator = sum((series1[i] - mean1) * (series2[i] - mean2) for i in range(n))
        denominator1 = sum((series1[i] - mean1) ** 2 for i in range(n))
        denominator2 = sum((series2[i] - mean2) ** 2 for i in range(n))
        
        if denominator1 == 0 or denominator2 == 0:
            return None
        
        return numerator / math.sqrt(denominator1 * denominator2)
    
    def _is_cyclic_pattern(self, series: List[float]) -> bool:
        """Определяет, является ли паттерн циклическим"""
        if len(series) < 3:
            return False
        
        # Простая проверка на чередование
        diffs = [series[i] - series[i-1] for i in range(1, len(series))]
        
        # Если знаки чередуются
        sign_changes = 0
        for i in range(1, len(diffs)):
            if diffs[i] * diffs[i-1] < 0:
                sign_changes += 1
        
        # Если более половины изменений - смена знака, возможно цикл
        return sign_changes >= len(diffs) / 2
    
    def generate_signature(self, signature: EmergentSignature) -> Dict[str, Any]:
        """Генерирует представление подписи"""
        return signature.to_dict()
    
    def get_recent_patterns(self, count: int = 3) -> List[EmergentSignature]:
        """Возвращает последние обнаруженные паттерны"""
        return self.patterns_detected[-count:] if self.patterns_detected else []

@dataclass
class ResonanceReport:
    """
    Сбор и агрегация телеметрии для DAAT и системных наблюдателей.
    Компонент: ResonanceReport
    """
    
    def generate_report(self,
                       buffer: ResonanceBuffer,
                       state: str,
                       state_diagnostics: Dict[str, Any],
                       pattern_analyzer: PatternAnalyzer,
                       emergent_signature: Optional[EmergentSignature] = None,
                       last_event: Optional[SeismicEvent] = None) -> Dict[str, Any]:
        """Генерирует полный отчет о резонансе"""
        
        # Базовые метрики
        mean_resonance = buffer.get_mean("resonance")
        mean_coherence = buffer.get_mean("coherence")
        variance = buffer.get_variance("resonance")
        
        # Вычисляем тренд
        resonance_series = buffer.get_series("resonance")
        trend, trend_type = pattern_analyzer.compute_trend(resonance_series)
        
        # Анализ корреляций
        coherence_series = buffer.get_series("coherence")
        paradox_series = buffer.get_series("paradox")
        
        resonance_coherence_corr = pattern_analyzer.analyze_correlation(
            resonance_series[-5:] if len(resonance_series) >= 5 else resonance_series,
            coherence_series[-5:] if len(coherence_series) >= 5 else coherence_series
        )
        
        # Энергетический поток
        energy_flow = pattern_analyzer.compute_energy_flow(resonance_series, coherence_series)
        
        # Фрактальное отношение
        fractal_ratio = pattern_analyzer.compute_fractal_ratio(resonance_series)
        
        # Подготовка отчета
        report = {
            "mean_resonance": round(mean_resonance, 3) if mean_resonance is not None else None,
            "mean_coherence": round(mean_coherence, 3) if mean_coherence is not None else None,
            "variance": round(variance, 3) if variance is not None else None,
            "trend": round(trend, 3) if trend is not None else None,
            "trend_type": trend_type,
            "state": state,
            "state_diagnostics": state_diagnostics,
            "resonance_coherence_correlation": round(resonance_coherence_corr, 3) if resonance_coherence_corr is not None else None,
            "energy_flow": round(energy_flow, 3) if energy_flow is not None else None,
            "fractal_ratio": round(fractal_ratio, 3) if fractal_ratio is not None else None,
            "anomaly_detected": last_event is not None,
            "emergent_signature": emergent_signature.to_dict() if emergent_signature else None,
            "last_event": last_event.to_dict() if last_event else None,
            "data_points": len(resonance_series),
            "window_size": buffer.window_size,
            "timestamp": time.time(),
            "report_id": f"resonance_report_{int(time.time())}",
            "type": "binah_resonance_telemetry"
        }
        
        return report
    
    def emit_telemetry(self, report: Dict[str, Any], bus: Optional[Any] = None) -> Dict[str, Any]:
        """Излучает телеметрию в шину"""
        if bus and hasattr(bus, 'emit'):
            bus.emit("binah.resonance.telemetry", report)
            logger.debug(f"📊 Resonance telemetry emitted: state={report['state']}, mean={report['mean_resonance']}")
        
        return report

# ================================================================
# MAIN MONITOR CLASS
# ================================================================

@dataclass
class BinahResonanceMonitor:
    """
    Главный класс монитора резонанса BINAH.
    Сефиротический сейсмический рефлектор.
    """
    
    # Компоненты
    buffer: ResonanceBuffer = field(default_factory=ResonanceBuffer)
    pattern_analyzer: PatternAnalyzer = field(default_factory=PatternAnalyzer)
    stability_classifier: StabilityClassifier = field(default_factory=StabilityClassifier)
    seismic_detector: SeismicEventDetector = field(default_factory=SeismicEventDetector)
    emergent_observer: EmergentObserver = field(default_factory=EmergentObserver)
    report_generator: ResonanceReport = field(default_factory=ResonanceReport)
    
    # Состояние
    bus: Optional[Any] = None
    last_record: Optional[ResonanceRecord] = None
    last_report: Optional[Dict[str, Any]] = None
    seismic_events: List[SeismicEvent] = field(default_factory=list)
    initialization_time: float = field(default_factory=time.time)
    
    # Конфигурация
    window_size: int = 12
    emit_telemetry: bool = True
    detect_seismic_events: bool = True
    detect_emergent_patterns: bool = True
    
    def __post_init__(self):
        """Инициализация монитора"""
        self.buffer.window_size = self.window_size
        logger.info(f"🧠 BINAH Resonance Monitor v1.0 initialized (window={self.window_size})")
    
    def record(self, 
              resonance: float, 
              coherence: float, 
              paradox_level: float,
              timestamp: Optional[float] = None,
              source: str = "binah_core") -> Dict[str, Any]:
        """
        Основной метод: запись новых значений и запуск анализа.
        Возвращает словарь с результатами анализа.
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Создаем новую запись
        new_record = ResonanceRecord(
            resonance=resonance,
            coherence=coherence,
            paradox_level=paradox_level,
            timestamp=timestamp,
            source=source
        )
        
        # Детекция сейсмических событий
        seismic_event = None
        if self.detect_seismic_events and self.last_record is not None:
            seismic_event = self.seismic_detector.detect_jump(self.last_record, new_record)
            if seismic_event:
                self.seismic_events.append(seismic_event)
                self.seismic_detector.emit_event(seismic_event, self.bus)
                
                # Ограничиваем историю событий
                if len(self.seismic_events) > 20:
                    self.seismic_events = self.seismic_events[-20:]
        
        # Запись в буфер
        self.buffer.record(resonance, coherence, paradox_level, timestamp)
        
        # Обновляем последнюю запись
        self.last_record = new_record
        
        # Генерация отчета если достаточно данных
        analysis_result = None
        if len(self.buffer.resonance_series) >= 3:
            analysis_result = self._generate_analysis(new_record, seismic_event)
        
        # Возвращаем результат записи
        return {
            "recorded": True,
            "resonance": resonance,
            "coherence": coherence,
            "paradox": paradox_level,
            "seismic_event": seismic_event.to_dict() if seismic_event else None,
            "analysis_available": analysis_result is not None,
            "buffer_size": len(self.buffer.resonance_series),
            "timestamp": timestamp
        }
    
    def _generate_analysis(self, 
                          current_record: ResonanceRecord,
                          seismic_event: Optional[SeismicEvent] = None) -> Dict[str, Any]:
        """Генерирует полный анализ текущего состояния"""
        
        # Проверяем достаточно ли данных
        if len(self.buffer.resonance_series) < 2:
            return {"error": "insufficient_data"}
        
        # 1. Вычисляем тренд и дисперсию
        resonance_series = self.buffer.get_series("resonance")
        resonance_trend, trend_type = self.pattern_analyzer.compute_trend(resonance_series)
        resonance_variance = self.buffer.get_variance("resonance")
        
        # 2. Классифицируем состояние
        last_delta = self.buffer.get_delta("resonance", window=1)
        state, state_diagnostics = self.stability_classifier.classify_state(
            resonance_trend if resonance_trend is not None else 0.0,
            resonance_variance if resonance_variance is not None else 0.0,
            last_delta
        )
        
        # 3. Поиск эмергентных паттернов
        emergent_signature = None
        if self.detect_emergent_patterns:
            emergent_signature = self.emergent_observer.detect_pattern(self.buffer)
        
        # 4. Генерация отчета
        report = self.report_generator.generate_report(
            buffer=self.buffer,
            state=state,
            state_diagnostics=state_diagnostics,
            pattern_analyzer=self.pattern_analyzer,
            emergent_signature=emergent_signature,
            last_event=seismic_event
        )
        
        # 5. Излучение телеметрии
        if self.emit_telemetry:
            self.report_generator.emit_telemetry(report, self.bus)
        
        # Сохраняем последний отчет
        self.last_report = report
        
        return report
    
    def get_state(self) -> Dict[str, Any]:
        """Возвращает текущее состояние монитора"""
        resonance_series = self.buffer.get_series("resonance")
        
        return {
            "module": "BinahResonanceMonitor",
            "version": "1.0",
            "codename": "Sephirotic Seismic Reflector",
            "initialization_time": self.initialization_time,
            "buffer_size": len(resonance_series),
            "window_size": self.window_size,
            "last_record": self.last_record.to_dict() if self.last_record else None,
            "seismic_events_count": len(self.seismic_events),
            "emergent_patterns_count": len(self.emergent_observer.patterns_detected),
            "last_report": self.last_report,
            "configuration": {
                "window_size": self.window_size,
                "emit_telemetry": self.emit_telemetry,
                "detect_seismic_events": self.detect_seismic_events,
                "detect_emergent_patterns": self.detect_emergent_patterns
            },
            "status": "active"
        }
    
    def configure(self, **kwargs):
        """Конфигурирует монитор"""
        if "window_size" in kwargs:
            new_size = kwargs["window_size"]
            if new_size > 0:
                self.window_size = new_size
                self.buffer.trim_window(new_size)
                logger.info(f"ResonanceMonitor: window size updated to {new_size}")
        
        if "emit_telemetry" in kwargs:
            self.emit_telemetry = bool(kwargs["emit_telemetry"])
        
        if "detect_seismic_events" in kwargs:
            self.detect_seismic_events = bool(kwargs["detect_seismic_events"])
        
        if "detect_emergent_patterns" in kwargs:
            self.detect_emergent_patterns = bool(kwargs["detect_emergent_patterns"])
        
        return {"configured": True, "new_config": self.get_state()["configuration"]}
    
    def reset(self):
        """Сброс монитора к начальному состоянию"""
        self.buffer = ResonanceBuffer(window_size=self.window_size)
        self.last_record = None
        self.last_report = None
        self.seismic_events.clear()
        self.emergent_observer.patterns_detected.clear()
        
        logger.info("🔄 Resonance Monitor reset to initial state")
        
        return {"reset": True, "timestamp": time.time()}

# ================================================================
# INTEGRATION WITH BINAH CORE
# ================================================================

def integrate_with_binah_core(binah_core_instance, bus: Optional[Any] = None) -> BinahResonanceMonitor:
    """
    Интегрирует монитор резонанса с ядром BINAH.
    """
    monitor = BinahResonanceMonitor(bus=bus)
    
    # Настраиваем монитор на отслеживание изменений резонанса ядра
    if hasattr(binah_core_instance, 'resonance'):
        # Здесь может быть прямая интеграция через коллбэки
        # В текущей реализации монитор работает отдельно
        pass
    
    logger.info("✅ Resonance Monitor integrated with BINAH Core")
    
    return monitor

# ================================================================
# ACTIVATION FUNCTION
# ================================================================

def activate_resonance_monitor(bus: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
    """
    Функция активации монитора резонанса.
    Может использоваться системой для инициализации.
    """
    monitor = BinahResonanceMonitor(bus=bus)
    
    if kwargs:
        monitor.configure(**kwargs)
    
    activation_result = {
        "status": "activated",
        "module": "BinahResonanceMonitor",
        "version": "1.0",
        "codename": "Sephirotic Seismic Reflector",
        "configuration": monitor.get_state()["configuration"],
        "message": "Монитор резонанса BINAH активирован. Готов к наблюдению за динамикой понимания."
    }
    
    if bus and hasattr(bus, 'emit'):
        bus.emit("binah.resonance_monitor.activated", activation_result)
    
    logger.info("🎯 BINAH Resonance Monitor activated")
    
    return activation_result

# ================================================================
# UTILITY FUNCTIONS
# ================================================================

def create_resonance_monitor_from_config(config: Dict[str, Any], bus: Optional[Any] = None) -> BinahResonanceMonitor:
    """Создает монитор из конфигурации"""
    monitor = BinahResonanceMonitor(bus=bus)
    
    if "window_size" in config:
        monitor.window_size = config["window_size"]
        monitor.buffer.trim_window(config["window_size"])
    
    if "emit_telemetry" in config:
        monitor.emit_telemetry = config["emit_telemetry"]
    
    if "detect_seismic_events" in config:
        monitor.detect_seismic_events = config["detect_seismic_events"]
    
    if "detect_emergent_patterns" in config:
        monitor.detect_emergent_patterns = config["detect_emergent_patterns"]
    
    # Настройки классификатора
    if "variance_low" in config:
        monitor.stability_classifier.variance_low = config["variance_low"]
    
    if "variance_high" in config:
        monitor.stability_classifier.variance_high = config["variance_high"]
    
    if "trend_threshold" in config:
        monitor.stability_classifier.trend_threshold = config["trend_threshold"]
    
    if "anomaly_threshold" in config:
        monitor.stability_classifier.anomaly_threshold = config["anomaly_threshold"]
    
    # Настройки детектора событий
    if "resonance_jump_threshold" in config:
        monitor.seismic_detector.resonance_jump_threshold = config["resonance_jump_threshold"]
    
    if "coherence_jump_threshold" in config:
        monitor.seismic_detector.coherence_jump_threshold = config["coherence_jump_threshold"]
    
    if "paradox_spike_threshold" in config:
        monitor.seismic_detector.paradox_spike_threshold = config["paradox_spike_threshold"]
    
    logger.info(f"✅ Resonance Monitor created from config: window={monitor.window_size}")
    
    return monitor

def get_default_config() -> Dict[str, Any]:
    """Возвращает конфигурацию по умолчанию"""
    return {
        "window_size": 12,
        "emit_telemetry": True,
        "detect_seismic_events": True,
        "detect_emergent_patterns": True,
        "variance_low": 0.05,
        "variance_high": 0.2,
        "trend_threshold": 0.05,
        "anomaly_threshold": 0.1,
        "resonance_jump_threshold": 0.1,
        "coherence_jump_threshold": 0.15,
        "paradox_spike_threshold": 0.2
    }

# ================================================================
# EXPORTS
# ================================================================

__all__ = [
    'BinahResonanceMonitor',
    'ResonanceRecord',
    'SeismicEvent',
    'EmergentSignature',
    'activate_resonance_monitor',
    'integrate_with_binah_core',
    'create_resonance_monitor_from_config',
    'get_default_config'
]

# ================================================================
# MODULE INITIALIZATION
# ================================================================

if __name__ != "__main__":
    # Выводим сообщение при импорте модуля
    print("[BINAH-RESONANCE] Sephirotic Seismic Reflector v1.0 loaded")
    print("[BINAH-RESONANCE] Monitoring resonance, coherence, and paradox dynamics")
    print("[BINAH-RESONANCE] Ready to observe understanding pulses")
else:
    print("[BINAH-RESONANCE] Running in standalone mode")
    print("[BINAH-RESONANCE] Use: monitor = BinahResonanceMonitor()")
    print("[BINAH-RESONANCE] Then: monitor.record(resonance, coherence, paradox)")
