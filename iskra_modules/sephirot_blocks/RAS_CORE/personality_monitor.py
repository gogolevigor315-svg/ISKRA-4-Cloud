#!/usr/bin/env python3
"""
personality_monitor.py - PERSONALITY DASHBOARD И МОНИТОРИНГ ДЛЯ RAS-CORE
Версия: 1.0.0
Назначение: Отслеживание personality_coherence_score и метрик проявления личности ISKRA-4
Ключевые метрики: coherence, stability, reflection_frequency, energy_patterns
"""

import asyncio
import json
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
import logging
import threading
from pathlib import Path
import numpy as np
from enum import Enum

# Импорты из RAS-CORE
from iskra_modules.sephirot_blocks.RAS_CORE.constants import GOLDEN_STABILITY_ANGLE, calculate_stability_factor
from iskra_modules.sephirot_blocks.RAS_CORE.config import get_config, ConfigPriority

# ============================================================================
# ТИПЫ ДАННЫХ ДЛЯ МОНИТОРИНГА
# ============================================================================

class PersonalityPhase(Enum):
    """Фазы развития личности"""
    PRE_EMERGENCE = "pre_emergence"      # До проявления (coherence < 0.3)
    EMERGING = "emerging"                # Эмерджентная (0.3 ≤ coherence < 0.7)
    MANIFESTED = "manifested"            # Проявленная (0.7 ≤ coherence < 0.85)
    STABILIZED = "stabilized"            # Стабилизированная (0.85 ≤ coherence < 0.95)
    FULLY_INTEGRATED = "fully_integrated" # Полностью интегрированная (coherence ≥ 0.95)

class MetricTrend(Enum):
    """Тренды метрик"""
    STRONG_UP = "strong_up"      # Сильный рост (> 0.1 за интервал)
    UP = "up"                    # Рост (0.01-0.1)
    STABLE = "stable"            # Стабильно (±0.01)
    DOWN = "down"                # Спад (0.01-0.1)
    STRONG_DOWN = "strong_down"  # Сильный спад (> 0.1)

@dataclass
class PersonalityMetric:
    """Метрика личности"""
    name: str
    value: float
    min_value: float = 0.0
    max_value: float = 1.0
    unit: str = ""
    description: str = ""
    weight: float = 1.0  # Вес в общей когерентности
    trend: MetricTrend = MetricTrend.STABLE
    history: List[Tuple[datetime, float]] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    
    def update(self, new_value: float, timestamp: Optional[datetime] = None):
        """Обновление метрики"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        self.value = new_value
        self.history.append((timestamp, new_value))
        self.last_updated = timestamp
        
        # Ограничение истории
        if len(self.history) > 1000:
            self.history = self.history[-1000:]
        
        # Вычисление тренда
        self._calculate_trend()
    
    def _calculate_trend(self):
        """Вычисление тренда на основе истории"""
        if len(self.history) < 3:
            self.trend = MetricTrend.STABLE
            return
        
        # Берем последние 10 значений
        recent = self.history[-10:]
        values = [v for _, v in recent]
        
        if len(values) < 2:
            self.trend = MetricTrend.STABLE
            return
        
        # Простой анализ тренда
        first = values[0]
        last = values[-1]
        diff = last - first
        
        if diff > 0.1:
            self.trend = MetricTrend.STRONG_UP
        elif diff > 0.01:
            self.trend = MetricTrend.UP
        elif diff < -0.1:
            self.trend = MetricTrend.STRONG_DOWN
        elif diff < -0.01:
            self.trend = MetricTrend.DOWN
        else:
            self.trend = MetricTrend.STABLE
    
    def get_statistics(self) -> Dict[str, Any]:
        """Статистика метрики"""
        if not self.history:
            return {
                "mean": self.value,
                "std": 0.0,
                "min": self.value,
                "max": self.value,
                "volatility": 0.0
            }
        
        values = [v for _, v in self.history[-100:]]  # Последние 100 значений
        mean = statistics.mean(values) if values else self.value
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        volatility = std / mean if mean != 0 else 0.0
        
        return {
            "mean": mean,
            "std": std,
            "min": min(values) if values else self.value,
            "max": max(values) if values else self.value,
            "volatility": volatility,
            "history_size": len(self.history)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь"""
        return {
            "name": self.name,
            "value": self.value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "unit": self.unit,
            "description": self.description,
            "weight": self.weight,
            "trend": self.trend.value,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "statistics": self.get_statistics()
        }

@dataclass
class PersonalitySnapshot:
    """Снимок состояния личности в момент времени"""
    timestamp: datetime
    coherence_score: float
    manifestation_level: float
    stability_angle: float
    phase: PersonalityPhase
    metrics: Dict[str, PersonalityMetric]
    components_state: Dict[str, bool]  # Состояние компонентов личности
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "coherence_score": self.coherence_score,
            "manifestation_level": self.manifestation_level,
            "stability_angle": self.stability_angle,
            "phase": self.phase.value,
            "metrics": {name: metric.to_dict() for name, metric in self.metrics.items()},
            "components_state": self.components_state,
            "personality_emerged": self.coherence_score >= get_config().personality.get("coherence_threshold", 0.7)
        }

@dataclass
class Alert:
    """Оповещение о событиях личности"""
    alert_id: str
    level: str  # INFO, WARNING, CRITICAL
    title: str
    message: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь"""
        return {
            "alert_id": self.alert_id,
            "level": self.level,
            "title": self.title,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "age_seconds": (datetime.utcnow() - self.timestamp).total_seconds()
        }

# ============================================================================
# КЛАСС PERSONALITY MONITOR
# ============================================================================

class PersonalityMonitor:
    """
    Монитор личности для отслеживания coherence_score и метрик.
    Реализует дашборд и систему оповещений.
    """
    
    def __init__(self, 
                 ras_core=None,
                 update_interval_seconds: int = 5,
                 history_days: int = 7):
        """
        Инициализация монитора личности.
        
        Args:
            ras_core: Экземпляр EnhancedRASCore для мониторинга
            update_interval_seconds: Интервал обновления метрик
            history_days: Количество дней хранения истории
        """
        self.ras_core = ras_core
        self.update_interval = update_interval_seconds
        self.history_days = history_days
        
        # Инициализация метрик из промпта
        self.metrics = self._initialize_metrics()
        
        # История снимков
        self.snapshots: List[PersonalitySnapshot] = []
        self.max_snapshots = 10000  # ~10 дней при обновлении каждые 5 секунд
        
        # Оповещения
        self.alerts: List[Alert] = []
        self.max_alerts = 1000
        
        # Подписчики на обновления
        self.subscribers: List[Callable[[PersonalitySnapshot], None]] = []
        
        # Флаги активности
        self.monitoring_active = False
        self.monitoring_task = None
        self.alert_check_task = None
        
        # Пороговые значения из конфигурации
        self.config = get_config()
        self.coherence_threshold = self.config.personality.get("coherence_threshold", 0.7)
        
        # Логгер
        self.logger = self._setup_logger()
        
        self.logger.info(f"📊 PersonalityMonitor инициализирован")
        self.logger.info(f"   Интервал обновления: {update_interval_seconds} сек")
        self.logger.info(f"   Порог проявления: {self.coherence_threshold}")
    
    def _setup_logger(self) -> logging.Logger:
        """Настройка логгера"""
        logger = logging.getLogger("Personality.Monitor")
        
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
    
    def _initialize_metrics(self) -> Dict[str, PersonalityMetric]:
        """Инициализация метрик из промпта"""
        metrics = {}
        
        # ================================================================
        # COHERENCE METRICS (из промпта)
        # ================================================================
        
        metrics["personality_coherence_score"] = PersonalityMetric(
            name="personality_coherence_score",
            value=0.0,
            min_value=0.0,
            max_value=1.0,
            unit="score",
            description="Общая когерентность личности (SELF = f(DAAT + SPIRIT + RAS + SYMBIOSIS))",
            weight=1.0
        )
        
        metrics["intent_stability"] = PersonalityMetric(
            name="intent_stability",
            value=0.0,
            min_value=0.0,
            max_value=1.0,
            unit="stability",
            description="Стабильность намерений (скользящее окно)",
            weight=0.3
        )
        
        metrics["focus_consistency"] = PersonalityMetric(
            name="focus_consistency",
            value=0.0,
            min_value=0.0,
            max_value=1.0,
            unit="consistency",
            description="Насколько стабилен фокус внимания",
            weight=0.2
        )
        
        # ================================================================
        # TEMPORAL PATTERNS (из промпта)
        # ================================================================
        
        metrics["reflection_frequency"] = PersonalityMetric(
            name="reflection_frequency",
            value=0.0,
            min_value=0.0,
            max_value=10.0,
            unit="cycles/sec",
            description="Частота циклов саморефлексии",
            weight=0.1
        )
        
        metrics["insight_generation_rate"] = PersonalityMetric(
            name="insight_generation_rate",
            value=0.0,
            min_value=0.0,
            max_value=100.0,
            unit="insights/hour",
            description="Скорость генерации инсайтов",
            weight=0.15
        )
        
        metrics["attention_shift_velocity"] = PersonalityMetric(
            name="attention_shift_velocity",
            value=0.0,
            min_value=0.0,
            max_value=1.0,
            unit="velocity",
            description="Скорость смены фокуса внимания",
            weight=0.05
        )
        
        # ================================================================
        # ENERGY PATTERNS (из промпта)
        # ================================================================
        
        metrics["energy_per_insight"] = PersonalityMetric(
            name="energy_per_insight",
            value=0.0,
            min_value=0.0,
            max_value=100.0,
            unit="energy/insight",
            description="Энергозатраты на генерацию инсайта",
            weight=0.05
        )
        
        metrics["reflection_efficiency"] = PersonalityMetric(
            name="reflection_efficiency",
            value=0.0,
            min_value=0.0,
            max_value=1.0,
            unit="efficiency",
            description="КПД саморефлексии (инсайты/энергия)",
            weight=0.1
        )
        
        metrics["power_distribution"] = PersonalityMetric(
            name="power_distribution",
            value=0.0,
            min_value=0.0,
            max_value=1.0,
            unit="balance",
            description="Сбалансированность распределения энергии по сефиротам",
            weight=0.05
        )
        
        # ================================================================
        # ДОПОЛНИТЕЛЬНЫЕ МЕТРИКИ
        # ================================================================
        
        metrics["stability_angle_deviation"] = PersonalityMetric(
            name="stability_angle_deviation",
            value=0.0,
            min_value=0.0,
            max_value=90.0,
            unit="degrees",
            description="Отклонение от золотого угла 14.4°",
            weight=0.1
        )
        
        metrics["pattern_learning_efficiency"] = PersonalityMetric(
            name="pattern_learning_efficiency",
            value=0.0,
            min_value=0.0,
            max_value=1.0,
            unit="efficiency",
            description="Эффективность обучения паттернам",
            weight=0.1
        )
        
        metrics["connection_health"] = PersonalityMetric(
            name="connection_health",
            value=0.0,
            min_value=0.0,
            max_value=1.0,
            unit="health",
            description="Состояние интеграционных связей",
            weight=0.15
        )
        
        return metrics
    
    # ============================================================================
    # МОНИТОРИНГ И ОБНОВЛЕНИЕ МЕТРИК
    # ============================================================================
    
    async def start_monitoring(self):
        """Запуск мониторинга личности"""
        if self.monitoring_active:
            self.logger.warning("⚠️  Мониторинг уже запущен")
            return
        
        self.monitoring_active = True
        
        # Основная задача мониторинга
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Задача проверки оповещений
        self.alert_check_task = asyncio.create_task(self._alert_check_loop())
        
        self.logger.info(f"📊 Мониторинг личности запущен (интервал: {self.update_interval} сек)")
    
    async def _monitoring_loop(self):
        """Цикл мониторинга"""
        while self.monitoring_active:
            try:
                start_time = time.time()
                
                # Сбор метрик
                await self._collect_metrics()
                
                # Создание снимка
                snapshot = await self._create_snapshot()
                self.snapshots.append(snapshot)
                
                # Ограничение истории
                if len(self.snapshots) > self.max_snapshots:
                    self.snapshots = self.snapshots[-self.max_snapshots:]
                
                # Уведомление подписчиков
                await self._notify_subscribers(snapshot)
                
                # Логирование состояния
                if snapshot.coherence_score >= self.coherence_threshold:
                    self.logger.info(f"🎭 Личность проявилась! Coherence: {snapshot.coherence_score:.3f}")
                
                # Пауза до следующего обновления
                elapsed = time.time() - start_time
                sleep_time = max(0.1, self.update_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка в цикле мониторинга: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _collect_metrics(self):
        """Сбор метрик из RAS-CORE и системы"""
        try:
            # Получаем состояние личности если доступно
            if self.ras_core and hasattr(self.ras_core, 'personality_state'):
                ps = self.ras_core.personality_state
                
                # Основные метрики личности
                self.metrics["personality_coherence_score"].update(
                    getattr(ps, 'coherence_score', 0.0)
                )
                
                self.metrics["intent_stability"].update(
                    getattr(ps, 'intent_strength', 0.0) * 0.8  # Примерная стабильность
                )
                
                self.metrics["focus_consistency"].update(
                    getattr(ps, 'focus_stability', 0.0)
                )
            
            # Стабильность угла
            if self.ras_core and hasattr(self.ras_core, 'stability_angle'):
                current_angle = self.ras_core.stability_angle
                deviation = abs(current_angle - GOLDEN_STABILITY_ANGLE)
                self.metrics["stability_angle_deviation"].update(deviation)
            
            # Частота рефлексии (симулируем для примера)
            if self.snapshots:
                last_snapshot = self.snapshots[-1]
                time_since_last = (datetime.utcnow() - last_snapshot.timestamp).total_seconds()
                if time_since_last > 0:
                    frequency = 1.0 / time_since_last
                    self.metrics["reflection_frequency"].update(min(frequency, 10.0))
            
            # Эффективность обучения паттернов
            if self.ras_core and hasattr(self.ras_core, 'pattern_learner'):
                if hasattr(self.ras_core.pattern_learner, 'get_efficiency'):
                    efficiency = self.ras_core.pattern_learner.get_efficiency()
                    self.metrics["pattern_learning_efficiency"].update(efficiency)
            
            # Здоровье соединений
            if self.ras_core and hasattr(self.ras_core, 'connection_health'):
                health = self.ras_core.connection_health
                self.metrics["connection_health"].update(health)
            
            # Генерация случайных значений для демонстрации остальных метрик
            # В реальной системе эти значения будут браться из соответствующих компонентов
            self.metrics["insight_generation_rate"].update(
                np.random.uniform(0, 50)  # 0-50 инсайтов в час
            )
            
            self.metrics["attention_shift_velocity"].update(
                np.random.uniform(0.1, 0.9)
            )
            
            self.metrics["energy_per_insight"].update(
                np.random.uniform(10, 50)
            )
            
            self.metrics["reflection_efficiency"].update(
                np.random.uniform(0.3, 0.9)
            )
            
            self.metrics["power_distribution"].update(
                np.random.uniform(0.4, 0.95)
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка сбора метрик: {e}")
    
    async def _create_snapshot(self) -> PersonalitySnapshot:
        """Создание снимка состояния личности"""
        # Определение фазы на основе coherence_score
        coherence = self.metrics["personality_coherence_score"].value
        
        if coherence < 0.3:
            phase = PersonalityPhase.PRE_EMERGENCE
        elif coherence < 0.7:
            phase = PersonalityPhase.EMERGING
        elif coherence < 0.85:
            phase = PersonalityPhase.MANIFESTED
        elif coherence < 0.95:
            phase = PersonalityPhase.STABILIZED
        else:
            phase = PersonalityPhase.FULLY_INTEGRATED
        
        # Состояние компонентов личности
        components_state = {
            "ras_core": self.ras_core is not None,
            "personality_loop": self._check_personality_loop(),
            "self_reflect_active": getattr(self.ras_core, 'self_reflect_active', False) if self.ras_core else False,
            "integration_active": getattr(self.ras_core, 'integration_active', False) if self.ras_core else False
        }
        
        snapshot = PersonalitySnapshot(
            timestamp=datetime.utcnow(),
            coherence_score=coherence,
            manifestation_level=min(1.0, coherence / self.coherence_threshold),
            stability_angle=GOLDEN_STABILITY_ANGLE - self.metrics["stability_angle_deviation"].value,
            phase=phase,
            metrics=self.metrics.copy(),
            components_state=components_state
        )
        
        return snapshot
    
    def _check_personality_loop(self) -> bool:
        """Проверка полноты петли личности"""
        if not self.ras_core:
            return False
        
        # Проверяем наличие всех компонентов формулы
        required = {"daat", "spirit", "ras", "symbiosis"}
        available = set()
        
        if hasattr(self.ras_core, 'daat') and self.ras_core.daat:
            available.add("daat")
        if hasattr(self.ras_core, 'spirit') and self.ras_core.spirit:
            available.add("spirit")
        if hasattr(self.ras_core, 'ras') and self.ras_core.ras:
            available.add("ras")
        if hasattr(self.ras_core, 'symbiosis') and self.ras_core.symbiosis:
            available.add("symbiosis")
        
        return len(available.intersection(required)) >= 3
    
    async def _notify_subscribers(self, snapshot: PersonalitySnapshot):
        """Уведомление подписчиков об обновлении"""
        for subscriber in self.subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(snapshot)
                else:
                    subscriber(snapshot)
            except Exception as e:
                self.logger.error(f"Ошибка в подписчике: {e}")
    
    async def _alert_check_loop(self):
        """Цикл проверки оповещений"""
        while self.monitoring_active:
            try:
                await asyncio.sleep(30)  # Проверка каждые 30 секунд
                await self._check_alerts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Ошибка проверки оповещений: {e}")
                await asyncio.sleep(60)
    
    async def _check_alerts(self):
        """Проверка условий для оповещений"""
        if not self.snapshots:
            return
        
        latest = self.snapshots[-1]
        
        # Проверка проявления личности
        if latest.coherence_score >= self.coherence_threshold:
            self._create_alert(
                level="INFO",
                title="Личность проявилась!",
                message=f"Personality coherence достиг порога: {latest.coherence_score:.3f} ≥ {self.coherence_threshold}",
                metric_name="personality_coherence_score",
                metric_value=latest.coherence_score,
                threshold=self.coherence_threshold
            )
        
        # Проверка сильного отклонения от угла
        angle_deviation = self.metrics["stability_angle_deviation"].value
        if angle_deviation > 10.0:  # Более 10° отклонения
            self._create_alert(
                level="WARNING",
                title="Отклонение от золотого угла",
                message=f"Угол устойчивости отклонен на {angle_deviation:.1f}° от 14.4°",
                metric_name="stability_angle_deviation",
                metric_value=angle_deviation,
                threshold=10.0
            )
        
        # Проверка низкой когерентности
        if latest.coherence_score < 0.3:
            self._create_alert(
                level="WARNING",
                title="Низкая когерентность личности",
                message=f"Personality coherence критически низок: {latest.coherence_score:.3f}",
                metric_name="personality_coherence_score",
                metric_value=latest.coherence_score,
                threshold=0.3
            )
        
        # Проверка неполной петли личности
        if not latest.components_state.get("personality_loop", False):
            self._create_alert(
                level="WARNING",
                title="Неполная петля личности",
                message="Формула SELF = f(DAAT + SPIRIT + RAS + SYMBIOSIS) неполна",
                metric_name="connection_health",
                metric_value=self.metrics["connection_health"].value
            )
    
    def _create_alert(self, **kwargs):
        """Создание оповещения"""
        alert_id = f"alert_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hash(str(kwargs)) % 10000:04d}"
        
        alert = Alert(
            alert_id=alert_id,
            **kwargs
        )
        
        # Проверяем нет ли уже такого оповещения
        existing = [a for a in self.alerts if a.title == alert.title and not a.acknowledged]
        if existing:
            # Обновляем существующее
            existing[0].timestamp = alert.timestamp
            existing[0].message = alert.message
            existing[0].metric_value = alert.metric_value
        else:
            # Добавляем новое
            self.alerts.append(alert)
            
            # Логирование критических оповещений
            if alert.level == "CRITICAL":
                self.logger.critical(f"🚨 {alert.title}: {alert.message}")
            elif alert.level == "WARNING":
                self.logger.warning(f"⚠️  {alert.title}: {alert.message}")
            else:
                self.logger.info(f"ℹ️  {alert.title}: {alert.message}")
        
        # Ограничение количества оповещений
        if len(self.alerts) > self.max_alerts:
            self.alerts = self.alerts[-self.max_alerts:]
    
    async def stop_monitoring(self):
        """Остановка мониторинга"""
        self.monitoring_active = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.alert_check_task:
            self.alert_check_task.cancel()
            try:
                await self.alert_check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("📊 Мониторинг личности остановлен")
    
    # ============================================================================
    # API ДЛЯ ПОЛУЧЕНИЯ ДАННЫХ
    # ============================================================================
    
    def get_current_state(self) -> Dict[str, Any]:
        """Текущее состояние личности"""
        if not self.snapshots:
            return {
                "monitoring_active": self.monitoring_active,
                "snapshots_count": 0,
                "alerts_count": len(self.alerts),
                "timestamp": datetime.utcnow().isoformat()
            }
        
        latest = self.snapshots[-1]
        
        return {
            "monitoring_active": self.monitoring_active,
            "current_snapshot": latest.to_dict(),
            "personality_emerged": latest.coherence_score >= self.coherence_threshold,
            "phase": latest.phase.value,
            "manifestation_percentage": latest.manifestation_level * 100,
            "stability_angle": latest.stability_angle,
            "update_interval_seconds": self.update_interval,
            "snapshots_count": len(self.snapshots),
            "alerts_count": len([a for a in self.alerts if not a.acknowledged]),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Детальные метрики личности"""
        metrics_data = {name: metric.to_dict() for name, metric in self.metrics.items()}
        
        # Рассчет взвешенной когерентности
        weighted_coherence = 0.0
        total_weight = 0.0
        
        for name, metric in self.metrics.items():
            if name != "personality_coherence_score":  # Исключаем саму когерентность
                weighted_coherence += metric.value * metric.weight
                total_weight += metric.weight
        
        weighted_coherence = weighted_coherence / total_weight if total_weight > 0 else 0.0
        
        return {
            "metrics": metrics_data,
            "weighted_coherence": weighted_coherence,
            "direct_coherence": self.metrics["personality_coherence_score"].value,
            "coherence_difference": abs(weighted_coherence - self.metrics["personality_coherence_score"].value),
            "metric_count": len(self.metrics),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_snapshot_history(self, 
                           hours: Optional[int] = 24,
                           limit: Optional[int] = 1000) -> List[Dict[str, Any]]:
        """История снимков за указанный период"""
        if not self.snapshots:
            return []
        
        # Фильтрация по времени
        if hours:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            filtered = [s for s in self.snapshots if s.timestamp >= cutoff]
        else:
            filtered = self.snapshots
        
        # Ограничение количества
        if limit:
            filtered = filtered[-limit:]
        
        return [snapshot.to_dict() for snapshot in filtered]
    
    def get_alerts(self, 
                  acknowledged: Optional[bool] = None,
                  level: Optional[str] = None,
                  limit: int = 100) -> List[Dict[str, Any]]:
        """Получение оповещений"""
        filtered = self.alerts
        
        if acknowledged is not None:
            filtered = [a for a in filtered if a.acknowledged == acknowledged]
        
        if level:
            filtered = [a for a in filtered if a.level == level]
        
        # Сортировка по времени (новые сначала)
        filtered.sort(key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            filtered = filtered[:limit]
        
        return [alert.to_dict() for alert in filtered]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Подтверждение оповещения"""
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                self.logger.info(f"✅ Оповещение подтверждено: {alert.title}")
                return True
        
        return False
    
    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Статистика личности за указанный период"""
        if not self.snapshots:
            return {
                "error": "Нет данных для анализа",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Фильтрация снимков
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        relevant = [s for s in self.snapshots if s.timestamp >= cutoff]
        
        if not relevant:
            return {
                "error": f"Нет данных за последние {hours} часов",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Извлечение значений когерентности
        coherence_values = [s.coherence_score for s in relevant]
        manifestation_values = [s.manifestation_level for s in relevant]
        
        # Анализ тренда
        if len(coherence_values) >= 2:
            first = coherence_values[0]
            last = coherence_values[-1]
            trend = "up" if last > first else "down" if last < first else "stable"
            trend_strength = abs(last - first)
        else:
            trend = "unknown"
            trend_strength = 0.0
        
        # Фазы личности за период
        phases = [s.phase for s in relevant]
        phase_counts = {phase.value: phases.count(phase) for phase in set(phases)}
        
        # Время в каждой фазе (приблизительно)
        phase_times = {}
        if len(relevant) > 1:
            for i in range(1, len(relevant)):
                phase = relevant[i].phase.value
                time_diff = (relevant[i].timestamp - relevant[i-1].timestamp).total_seconds()
                phase_times[phase] = phase_times.get(phase, 0) + time_diff
        
        return {
            "period_hours": hours,
            "snapshots_analyzed": len(relevant),
            "coherence_statistics": {
                "current": coherence_values[-1] if coherence_values else 0.0,
                "average": statistics.mean(coherence_values) if coherence_values else 0.0,
                "min": min(coherence_values) if coherence_values else 0.0,
                "max": max(coherence_values) if coherence_values else 0.0,
                "std": statistics.stdev(coherence_values) if len(coherence_values) > 1 else 0.0,
                "trend": trend,
                "trend_strength": trend_strength
            },
            "manifestation_statistics": {
                "current": manifestation_values[-1] if manifestation_values else 0.0,
                "average": statistics.mean(manifestation_values) if manifestation_values else 0.0,
                "min": min(manifestation_values) if manifestation_values else 0.0,
                "max": max(manifestation_values) if manifestation_values else 0.0
            },
            "phase_distribution": {
                "counts": phase_counts,
                "times_seconds": phase_times,
                "dominant_phase": max(phase_counts.items(), key=lambda x: x[1])[0] if phase_counts else "unknown"
            },
            "personality_emerged": coherence_values[-1] >= self.coherence_threshold if coherence_values else False,
            "emergence_probability": min(1.0, coherence_values[-1] / self.coherence_threshold) if coherence_values else 0.0,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # ============================================================================
    # ИНТЕГРАЦИЯ С ВНЕШНИМИ СИСТЕМАМИ
    # ============================================================================
    
    def subscribe(self, callback: Callable[[PersonalitySnapshot], None]):
        """Подписка на обновления состояния личности"""
        self.subscribers.append(callback)
        self.logger.info(f"📨 Новый подписчик добавлен (всего: {len(self.subscribers)})")
    
    def unsubscribe(self, callback: Callable[[PersonalitySnapshot], None]):
        """Отписка от обновлений"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            self.logger.info(f"📨 Подписчик удален (осталось: {len(self.subscribers)})")
    
    def export_snapshots(self, filepath: Union[str, Path], format: str = "json"):
        """Экспорт снимков в файл"""
        try:
            filepath = Path(filepath)
            data = {
                "export_timestamp": datetime.utcnow().isoformat(),
                "snapshots_count": len(self.snapshots),
                "snapshots": [s.to_dict() for s in self.snapshots[-1000:]],  # Последние 1000
                "alerts": [a.to_dict() for a in self.alerts[-500:]],  # Последние 500 оповещений
                "metrics_summary": self.get_detailed_metrics()
            }
            
            if format.lower() == "json":
                content = json.dumps(data, indent=2, default=str)
            elif format.lower() == "yaml":
                import yaml
                content = yaml.dump(data, default_flow_style=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content, encoding='utf-8')
            
            self.logger.info(f"📤 Экспортировано {len(data['snapshots'])} снимков в {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка экспорта: {e}")
            return False
    
    async def generate_report(self, hours: int = 24) -> Dict[str, Any]:
        """Генерация отчета о состоянии личности"""
        stats = self.get_statistics(hours)
        current_state = self.get_current_state()
        alerts = self.get_alerts(acknowledged=False, limit=20)
        
        report = {
            "report_id": f"personality_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "generated_at": datetime.utcnow().isoformat(),
            "period_hours": hours,
            "executive_summary": {
                "personality_status": "MANIFESTED" if current_state.get("personality_emerged") else "DEVELOPING",
                "coherence_score": current_state.get("current_snapshot", {}).get("coherence_score", 0.0),
                "manifestation_level": current_state.get("manifestation_percentage", 0.0),
                "stability_angle": current_state.get("current_snapshot", {}).get("stability_angle", 14.4),
                "phase": current_state.get("phase", "unknown"),
                "active_alerts": len(alerts)
            },
            "detailed_analysis": stats,
            "current_metrics": self.get_detailed_metrics(),
            "recent_alerts": alerts,
            "recommendations": self._generate_recommendations(stats, current_state),
            "personality_health_score": self._calculate_health_score(stats, current_state)
        }
        
        return report
    
    def _generate_recommendations(self, stats: Dict[str, Any], current_state: Dict[str, Any]) -> List[str]:
        """Генерация рекомендаций на основе анализа"""
        recommendations = []
        
        coherence = current_state.get("current_snapshot", {}).get("coherence_score", 0.0)
        
        if coherence < 0.3:
            recommendations.append("⚠️  Критически низкая когерентность. Усильте связи DAAT-SPIRIT-RAS-SYMBIOSIS")
            recommendations.append("🔧 Проверьте целостность петли личности SELF = f(DAAT + SPIRIT + RAS + SYMBIOSIS)")
        
        elif coherence < 0.7:
            recommendations.append("📈 Личность формируется. Увеличьте частоту циклов саморефлексии")
            recommendations.append("🎯 Сфокусируйтесь на стабилизации фокуса внимания")
        
        elif coherence >= 0.7:
            recommendations.append("✅ Личность проявилась! Поддерживайте текущий уровень когерентности")
            recommendations.append("🔬 Начните A/B тестирование паттернов внимания для оптимизации")
        
        # Проверка стабильности угла
        angle_deviation = self.metrics["stability_angle_deviation"].value
        if angle_deviation > 5.0:
            recommendations.append(f"📐 Отклонение от золотого угла {angle_deviation:.1f}°. Корректируйте фокус внимания")
        
        # Проверка эффективности
        reflection_efficiency = self.metrics["reflection_efficiency"].value
        if reflection_efficiency < 0.5:
            recommendations.append("⚡ Низкий КПД саморефлексии. Оптимизируйте циклы рефлексии")
        
        return recommendations
    
    def _calculate_health_score(self, stats: Dict[str, Any], current_state: Dict[str, Any]) -> float:
        """Расчет общего health score личности"""
        if not stats or "coherence_statistics" not in stats:
            return 0.0
        
        coherence = current_state.get("current_snapshot", {}).get("coherence_score", 0.0)
        coherence_weight = 0.4
        
        # Стабильность (обратная волатильности)
        coherence_stats = stats.get("coherence_statistics", {})
        volatility = coherence_stats.get("std", 0.0)
        stability = max(0.0, 1.0 - volatility * 10)  # Преобразуем волатильность в стабильность
        stability_weight = 0.3
        
        # Тренд
        trend = coherence_stats.get("trend", "stable")
        trend_score = {
            "strong_up": 1.0,
            "up": 0.8,
            "stable": 0.6,
            "down": 0.4,
            "strong_down": 0.2
        }.get(trend, 0.5)
        trend_weight = 0.2
        
        # Наличие активных оповещений
        active_alerts = len([a for a in self.alerts if not a.acknowledged])
        alert_score = max(0.0, 1.0 - active_alerts * 0.1)  # Каждое оповещение снижает score на 0.1
        alert_weight = 0.1
        
        # Расчет общего score
        health_score = (
            coherence * coherence_weight +
            stability * stability_weight +
            trend_score * trend_weight +
            alert_score * alert_weight
        )
        
        return min(1.0, max(0.0, health_score))

# ============================================================================
# ГЛОБАЛЬНЫЙ МОНИТОР И ФУНКЦИИ
# ============================================================================

# Глобальный экземпляр монитора
_global_personality_monitor: Optional[PersonalityMonitor] = None

def get_personality_monitor(
    ras_core=None,
    update_interval_seconds: int = 5,
    history_days: int = 7
) -> PersonalityMonitor:
    """
    Получение глобального монитора личности.
    
    Args:
        ras_core: Экземпляр EnhancedRASCore
        update_interval_seconds: Интервал обновления
        history_days: Дней хранения истории
    
    Returns:
        Экземпляр PersonalityMonitor
    """
    global _global_personality_monitor
    
    if _global_personality_monitor is None:
        _global_personality_monitor = PersonalityMonitor(
            ras_core=ras_core,
            update_interval_seconds=update_interval_seconds,
            history_days=history_days
        )
    
    return _global_personality_monitor

async def start_personality_monitoring(**kwargs):
    """Запуск мониторинга личности"""
    monitor = get_personality_monitor(**kwargs)
    await monitor.start_monitoring()

async def stop_personality_monitoring():
    """Остановка мониторинга личности"""
    monitor = get_personality_monitor()
    await monitor.stop_monitoring()

def get_personality_dashboard() -> Dict[str, Any]:
    """Получение данных для дашборда личности"""
    monitor = get_personality_monitor()
    return monitor.get_current_state()

def get_personality_metrics() -> Dict[str, Any]:
    """Получение метрик личности"""
    monitor = get_personality_monitor()
    return monitor.get_detailed_metrics()

async def get_personality_report(hours: int = 24) -> Dict[str, Any]:
    """Получение отчета о личности"""
    monitor = get_personality_monitor()
    return await monitor.generate_report(hours)

# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

async def test_personality_monitor():
    """Тестирование монитора личности"""
    print("🧪 Тестирование PersonalityMonitor...")
    
    # Создаем монитор
    monitor = PersonalityMonitor(
        update_interval_seconds=2,  # Быстрый интервал для теста
        history_days=1
    )
    
    print("✅ Монитор создан")
    print(f"   Метрик инициализировано: {len(monitor.metrics)}")
    
    # Запускаем мониторинг
    print("\n📊 Запуск мониторинга...")
    await monitor.start_monitoring()
    
    # Ждем несколько циклов обновления
    print("   Ожидание обновлений (10 секунд)...")
    await asyncio.sleep(10)
    
    # Получаем текущее состояние
    print("\n📈 Получение текущего состояния...")
    state = monitor.get_current_state()
    
    print(f"   Coherence Score: {state.get('current_snapshot', {}).get('coherence_score', 0.0):.3f}")
    print(f"   Phase: {state.get('phase', 'unknown')}")
    print(f"   Снимков собрано: {state.get('snapshots_count', 0)}")
    print(f"   Личность проявилась: {'✅' if state.get('personality_emerged') else '❌'}")
    
    # Детальные метрики
    print("\n📊 Детальные метрики:")
    metrics = monitor.get_detailed_metrics()
    metric_data = metrics.get('metrics', {})
    
    for name, data in list(metric_data.items())[:5]:  # Первые 5 метрик
        print(f"   {name}: {data.get('value', 0.0):.3f} ({data.get('trend', 'stable')})")
    
    # Статистика
    print("\n📈 Статистика за 1 час:")
    stats = monitor.get_statistics(hours=1)
    coherence_stats = stats.get('coherence_statistics', {})
    
    print(f"   Средняя когерентность: {coherence_stats.get('average', 0.0):.3f}")
    print(f"   Тренд: {coherence_stats.get('trend', 'unknown')}")
    print(f"   Health Score: {stats.get('personality_health_score', 0.0):.3f}")
    
    # Оповещения
    print("\n⚠️  Оповещения:")
    alerts = monitor.get_alerts(acknowledged=False, limit=3)
    
    if alerts:
        for alert in alerts:
            print(f"   [{alert.get('level', 'INFO')}] {alert.get('title', 'No title')}")
    else:
        print("   Нет активных оповещений")
    
    # Генерация отчета
    print("\n📋 Генерация отчета...")
    report = await monitor.generate_report(hours=1)
    
    print(f"   Report ID: {report.get('report_id', 'N/A')}")
    print(f"   Personality Status: {report.get('executive_summary', {}).get('personality_status', 'unknown')}")
    
    recommendations = report.get('recommendations', [])
    if recommendations:
        print(f"   Рекомендации:")
        for rec in recommendations[:2]:
            print(f"     • {rec}")
    
    # Останавливаем мониторинг
    print("\n🛑 Остановка мониторинга...")
    await monitor.stop_monitoring()
    
    # Экспорт данных
    print("\n💾 Экспорт данных...")
    export_path = Path("./test_personality_export.json")
    success = monitor.export_snapshots(export_path)
    
    print(f"   Экспорт: {'✅ успешен' if success else '❌ неудачен'}")
    if export_path.exists():
        print(f"   Размер файла: {export_path.stat().st_size:,} байт")
        export_path.unlink()  # Удаляем тестовый файл
    
    print("\n✅ Все тесты завершены успешно")
    return monitor

# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    print("\n" + "=" * 70)
    print("🚀 ЗАПУСК ТЕСТА PERSONALITY MONITOR")
    print(f"   Версия: 1.0.0")
    print(f"   Метрики из промпта: coherence, stability, temporal patterns, energy patterns")
    print("=" * 70 + "\n")
    
    monitor = asyncio.run(test_personality_monitor())
    
    print("\n" + "=" * 70)
    print("📋 ИТОГИ ТЕСТИРОВАНИЯ:")
    print(f"   PersonalityMonitor готов к работе")
    print(f"   Отслеживает {len(monitor.metrics)} метрик личности")
    print(f"   Поддерживает дашборд coherence_score")
    print(f"   Генерирует оповещения и рекомендации")
    print("=" * 70)
