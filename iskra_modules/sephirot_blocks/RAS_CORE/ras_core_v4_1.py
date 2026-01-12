# ================================================================
# DS24 · ISKRA5 MODULE · RAS-CORE v4.1 "PRIORITY CONSCIOUS ENGINE"
# UPDATED WITH 14.4° STABILITY ANGLE INTEGRATION
# ================================================================

import asyncio
import json
import random
import statistics
import time
import math
from typing import Any, Dict, List, Optional, Protocol
from datetime import datetime

# ================================================================
# CONSTANTS & UTILS
# ================================================================

GOLDEN_STABILITY_ANGLE = 14.4  # Угол устойчивости бойца/системы
MAX_REFLECTION_DEPTH = 10      # Максимальная глубина рефлексии

def calculate_stability_factor(deviation: float) -> float:
    """Коэффициент устойчивости на основе угла 14.4°."""
    if deviation <= 0:
        return 1.0
    angle_diff = abs(deviation - GOLDEN_STABILITY_ANGLE)
    return max(0.0, 1 - angle_diff / GOLDEN_STABILITY_ANGLE)

def angle_to_priority(angle: float) -> float:
    """Преобразует угол (0-90°) в приоритет (0-1)."""
    # 14.4° -> приоритет 1.0
    # Чем дальше от 14.4°, тем ниже приоритет
    diff = abs(angle - GOLDEN_STABILITY_ANGLE)
    return max(0.1, 1 - diff / 45.0)  # 45° - максимальное отклонение

# ================================================================
# INTERFACES
# ================================================================

class ISephiroticBusAdapter(Protocol):
    async def publish(self, topic: str, message: Dict[str, Any]) -> None: ...
    async def get_load(self, target: str) -> float: ...
    async def get_triad_balance(self) -> Dict[str, float]: ...  # Новый метод

# ================================================================
# DATA MODEL
# ================================================================

class RASSignal:
    """Unified data entity with stability angle integration."""
    
    def __init__(self, data: Dict[str, Any]):
        self.id = data.get("id", f"sig-{int(time.time() * 1000)}-{random.randint(1000,9999)}")
        self.payload = data.get("payload", "")
        self.neuro_weight = data.get("neuro_weight", 1.0)
        self.semiotic_tags = data.get("semiotic_tags", [])
        self.priority = data.get("priority", 0.5)
        self.correlation_id = data.get("correlation_id", None)
        self.created_at = time.time()
        self.metadata = data
        
        # Угловые метрики
        self.stability_angle = data.get("stability_angle", GOLDEN_STABILITY_ANGLE)
        self.angle_confidence = data.get("angle_confidence", 1.0)
        
        # Фокусные параметры
        self.focus_vector = data.get("focus_vector", [0.0, 0.0, 0.0])  # [x, y, z]
        self.resonance_index = data.get("resonance_index", 0.5)
    
    def duration(self) -> float:
        return time.time() - self.created_at
    
    def calculate_stability_score(self) -> float:
        """Оценка устойчивости сигнала на основе угла."""
        angle_score = calculate_stability_factor(self.stability_angle)
        resonance_score = self.resonance_index
        
        # Комбинированная оценка
        stability_score = (angle_score * 0.6 + resonance_score * 0.4)
        return min(1.0, stability_score * self.angle_confidence)
    
    def to_dict(self):
        return {
            "id": self.id,
            "payload": self.payload[:100] + "..." if len(self.payload) > 100 else self.payload,
            "neuro_weight": self.neuro_weight,
            "semiotic_tags": self.semiotic_tags,
            "priority": self.priority,
            "stability_angle": self.stability_angle,
            "stability_score": self.calculate_stability_score(),
            "correlation_id": self.correlation_id,
            "focus_vector": self.focus_vector,
            "resonance_index": self.resonance_index,
            "created_at": datetime.fromtimestamp(self.created_at).isoformat(),
            "metadata": {k: v for k, v in self.metadata.items() if k not in ['payload']}
        }

# ================================================================
# ENHANCED SUPPORT COMPONENTS WITH 14.4° INTEGRATION
# ================================================================

class StabilityAwarePriorityQueue(PrioritySignalQueue):
    """Очередь с учётом угла устойчивости 14.4°."""
    
    async def calculate_signal_priority(self, signal: RASSignal) -> float:
        """Вычисляет приоритет на основе угла устойчивости и других факторов."""
        base_priority = signal.priority
        stability_score = signal.calculate_stability_score()
        
        # Сигналы с углом близким к 14.4° получают буст
        angle_factor = 1.0
        if abs(signal.stability_angle - GOLDEN_STABILITY_ANGLE) < 2.0:  # ±2° от идеала
            angle_factor = 1.3  # +30% приоритет
        
        # Финальный приоритет
        final_priority = base_priority * stability_score * angle_factor
        
        # Ограничиваем и нормализуем
        return max(0.1, min(1.0, final_priority))
    
    async def push(self, signal: RASSignal):
        # Пересчитываем приоритет с учётом угла устойчивости
        enhanced_priority = await self.calculate_signal_priority(signal)
        signal.priority = enhanced_priority
        
        # Используем базовую логику приоритизации
        if enhanced_priority >= 0.9:
            await self.queues["critical"].put(signal)
        elif enhanced_priority >= 0.6:
            await self.queues["high"].put(signal)
        else:
            await self.queues["normal"].put(signal)

# ================================================================
# ENHANCED CORE COMPONENTS
# ================================================================

class RASConfigManager:
    def __init__(self):
        self.threshold = 0.7
        self.focus_patterns = ["смысл", "инсайт", "анализ", "паттерн", "устойчивость"]
        self.golden_angle = GOLDEN_STABILITY_ANGLE
        self.ideal_deviation = 2.0  # Допустимое отклонение от 14.4°
        
        # Настройки циклов рефлексии
        self.reflection_cycle_ms = 144  # 14.4 * 10
        self.max_reflection_depth = MAX_REFLECTION_DEPTH
    
    async def adjust_for_stability(self, current_angle: float) -> Dict[str, Any]:
        """Корректирует параметры на основе текущего угла устойчивости."""
        deviation = abs(current_angle - self.golden_angle)
        stability_factor = calculate_stability_factor(deviation)
        
        adjustments = {
            "threshold_multiplier": stability_factor,
            "processing_speed": 1.0 + (1.0 - stability_factor) * 0.5,  # Быстрее при нестабильности
            "pattern_recognition_boost": stability_factor * 1.2,
            "current_stability": stability_factor
        }
        
        # Динамически корректируем порог
        self.threshold = max(0.3, min(1.0, 0.7 * stability_factor))
        
        return adjustments

class AngleAwareSephiroticRouter(SephiroticRouter):
    """Маршрутизатор с учётом угла устойчивости."""
    
    async def calculate_sephira_stability(self, target: str, bus: ISephiroticBusAdapter) -> float:
        """Оценивает устойчивость сефиры для маршрутизации."""
        try:
            # Получаем баланс триады (если доступно)
            triad_balance = await bus.get_triad_balance()
            
            if triad_balance and target in ["KETER", "CHOKMAH", "BINAH"]:
                # Для узлов триады используем их текущие значения
                node_value = triad_balance.get(target, 0.5)
                
                # Рассчитываем отклонение от идеала для этой сефиры
                # В идеале все три сефиры должны быть сбалансированы
                avg_value = sum(triad_balance.values()) / 3
                deviation = abs(node_value - avg_value)
                
                # Преобразуем в коэффициент устойчивости
                return calculate_stability_factor(deviation * 90)  # Нормализуем к градусам
        except:
            pass
        
        # Fallback: используем нагрузку
        load = await bus.get_load(target)
        return 1.0 - load  # Меньше нагрузка → больше устойчивость
    
    async def route_to_sephira(self, signal: RASSignal) -> str:
        """Улучшенная маршрутизация с учётом устойчивости узлов."""
        # Базовые оценки
        score = signal.metadata.get("daat_insight", 0.5)
        stability_score = signal.calculate_stability_score()
        
        # Оцениваем устойчивость каждой потенциальной цели
        target_scores = {}
        for target in self.targets:
            try:
                node_stability = await self.calculate_sephira_stability(target, self.bus)
                
                # Комбинированная оценка
                if target == "KETER":
                    target_score = score * 0.7 + stability_score * 0.3
                elif target == "CHOKMAH":
                    target_score = (score * 0.6 + node_stability * 0.4) * 1.1
                elif target == "DAAT":
                    target_score = (score * 0.8 + node_stability * 0.2) * 1.2
                elif target == "BINAH":
                    target_score = (score * 0.5 + node_stability * 0.5) * 1.0
                else:
                    target_score = node_stability
                
                target_scores[target] = target_score * node_stability
            except:
                target_scores[target] = 0.1
        
        # Выбираем цель с максимальной комбинированной оценкой
        best_target = max(target_scores, key=target_scores.get)
        
        # Логируем решение
        signal.metadata["routing_decision"] = {
            "chosen": best_target,
            "scores": {k: round(v, 3) for k, v in target_scores.items()},
            "signal_stability": round(stability_score, 3),
            "signal_angle": signal.stability_angle
        }
        
        return best_target

class StabilityMetricsCollector(MetricsCollector):
    """Сбор метрик с акцентом на устойчивость."""
    
    def __init__(self):
        super().__init__()
        self.stability_scores = []
        self.angle_deviations = []
        self.reflection_depths = []
    
    def observe_stability(self, score: float):
        self.stability_scores.append(score)
    
    def observe_angle(self, angle: float):
        deviation = abs(angle - GOLDEN_STABILITY_ANGLE)
        self.angle_deviations.append(deviation)
    
    def observe_reflection(self, depth: int):
        self.reflection_depths.append(depth)
    
    def summary(self):
        base_summary = super().summary()
        
        stability_metrics = {}
        if self.stability_scores:
            stability_metrics["avg_stability"] = round(statistics.mean(self.stability_scores), 3)
            stability_metrics["min_stability"] = round(min(self.stability_scores), 3)
        
        if self.angle_deviations:
            stability_metrics["avg_angle_deviation"] = round(statistics.mean(self.angle_deviations), 2)
            stability_metrics["close_to_golden_ratio"] = round(
                sum(1 for d in self.angle_deviations if d < 5.0) / len(self.angle_deviations), 3
            )
        
        if self.reflection_depths:
            stability_metrics["avg_reflection_depth"] = round(statistics.mean(self.reflection_depths), 1)
            stability_metrics["max_reflection_depth"] = max(self.reflection_depths)
        
        base_summary.update(stability_metrics)
        return base_summary

# ================================================================
# SELF-REFLECTION CYCLE WITH 14.4° RHYTHM
# ================================================================

class SelfReflectionEngine:
    """Двигатель саморефлексии с ритмом 14.4°."""
    
    def __init__(self, ras_core: "EnhancedRASCore"):
        self.core = ras_core
        self.reflection_counter = 0
        self.last_reflection_time = 0
        self.current_depth = 0
    
    async def reflection_cycle(self):
        """Основной цикл саморефлексии с защитой от зацикливания."""
        while True:
            cycle_start = time.time()
            
            # Проверяем глубину рефлексии
            if self.current_depth >= MAX_REFLECTION_DEPTH:
                print(f"[REFLECTION] Max depth reached ({self.current_depth}), forcing external focus")
                await self.force_external_focus()
                self.current_depth = 0
                continue
            
            # Выполняем шаг рефлексии
            reflection_result = await self.execute_reflection_step()
            
            # Обновляем метрики
            self.reflection_counter += 1
            self.current_depth = reflection_result.get("depth", self.current_depth + 1)
            
            # Ритм цикла: 14.4% от базового интервала или 144 мс
            cycle_duration = time.time() - cycle_start
            target_interval = 0.144  # 144 мс
            
            # Корректируем интервал на основе устойчивости
            stability = reflection_result.get("stability_score", 0.5)
            adjusted_interval = target_interval * (1.0 + (1.0 - stability) * 0.5)
            
            sleep_time = max(0.01, adjusted_interval - cycle_duration)
            await asyncio.sleep(sleep_time)
            
            # Периодически сбрасываем глубину
            if self.reflection_counter % 10 == 0:
                self.current_depth = max(0, self.current_depth - 2)
    
    async def execute_reflection_step(self) -> Dict[str, Any]:
        """Выполняет один шаг саморефлексии."""
        try:
            # Получаем текущее состояние
            metrics = self.core.metrics_snapshot()
            
            # Анализируем устойчивость
            recent_signals = self.core.pattern_learner.successful[-5:] if self.core.pattern_learner.successful else []
            if recent_signals:
                avg_stability = sum(s.calculate_stability_score() for s in recent_signals) / len(recent_signals)
                avg_angle = sum(s.stability_angle for s in recent_signals) / len(recent_signals)
            else:
                avg_stability = 0.5
                avg_angle = GOLDEN_STABILITY_ANGLE
            
            # Оцениваем отклонение от идеального угла
            angle_deviation = abs(avg_angle - GOLDEN_STABILITY_ANGLE)
            stability_factor = calculate_stability_factor(angle_deviation)
            
            # Формируем инсайт
            insight = {
                "timestamp": time.time(),
                "stability_score": avg_stability,
                "current_angle": avg_angle,
                "angle_deviation": angle_deviation,
                "stability_factor": stability_factor,
                "processing_health": metrics.get("error_rate", 0.0),
                "reflection_depth": self.current_depth,
                "cycle_number": self.reflection_counter
            }
            
            # Публикуем инсайт в DAAT для мета-осознания
            if self.core.bus:
                await self.core.bus.publish("DAAT", {
                    "type": "self_reflection_insight",
                    "insight": insight,
                    "source": "RAS_CORE",
                    "priority": stability_factor
                })
            
            # Логируем
            if self.reflection_counter % 5 == 0:
                print(f"[REFLECTION] Cycle {self.reflection_counter}: "
                      f"angle={avg_angle:.1f}°, stability={stability_factor:.3f}, "
                      f"depth={self.current_depth}")
            
            return insight
            
        except Exception as e:
            print(f"[REFLECTION] Error in reflection step: {e}")
            return {"error": str(e), "depth": self.current_depth}
    
    async def force_external_focus(self):
        """Принудительный выход из глубокой рефлексии."""
        print("[REFLECTION] Forcing external focus to break reflection loop")
        
        # Генерируем внешне-ориентированный сигнал
        external_signal = RASSignal({
            "payload": "EXTERNAL_FOCUS_INJECTION",
            "priority": 0.9,
            "semiotic_tags": ["external", "focus", "reset"],
            "stability_angle": GOLDEN_STABILITY_ANGLE,
            "focus_vector": [1.0, 0.0, 0.0],  # Направлен вовне
            "resonance_index": 0.3
        })
        
        # Отправляем в CHOKMAH для интуитивной обработки
        if self.core.bus:
            await self.core.bus.publish("CHOKMAH", external_signal.to_dict())

# ================================================================
# ENHANCED RAS CORE
# ================================================================

class EnhancedRASCore(RASCore):
    """Улучшенный RAS-CORE с полной интеграцией угла 14.4°."""
    
    def __init__(self, bus: ISephiroticBusAdapter):
        super().__init__(bus)
        
        # Заменяем компоненты на улучшенные версии
        self.queue = StabilityAwarePriorityQueue()
        self.router = AngleAwareSephiroticRouter(bus)
        self.metrics = StabilityMetricsCollector()
        self.config = RASConfigManager()
        
        # Двигатель саморефлексии
        self.reflection_engine = SelfReflectionEngine(self)
        
        # Триадный монитор
        self.triad_monitor = TriadStabilityMonitor(self)
        
        # Запускаем фоновые задачи
        self.reflection_task = None
        self.monitoring_task = None
    
    async def start_background_tasks(self):
        """Запускает фоновые процессы."""
        self.reflection_task = asyncio.create_task(self.reflection_engine.reflection_cycle())
        self.monitoring_task = asyncio.create_task(self.triad_monitor.monitor_loop())
        print("[RAS-CORE] Background tasks started: reflection_cycle, triad_monitor")
    
    async def process(self, data: Dict[str, Any]):
        """Улучшенная обработка с учётом устойчивости."""
        start = time.time()
        
        # Добавляем угловые параметры, если их нет
        if "stability_angle" not in data:
            data["stability_angle"] = GOLDEN_STABILITY_ANGLE
        
        signal = RASSignal(data)
        
        # Собираем метрики устойчивости
        self.metrics.observe_stability(signal.calculate_stability_score())
        self.metrics.observe_angle(signal.stability_angle)
        
        try:
            # Обработка через приоритетную очередь
            await self.queue.push(signal)
            processed_signal = await self.queue.pop()
            
            if not processed_signal:
                return {"status": "queue_empty"}
            
            # Маршрутизация с учётом устойчивости
            target = await self.router.route_to_sephira(processed_signal)
            processed_signal.metadata["target"] = target
            
            # Отправка в выбранную сефиру
            await self.bus.publish(target, processed_signal.to_dict())
            
            # Архивация и обучение
            await self.archiver.archive(processed_signal, {"status": "success", "target": target})
            self.pattern_learner.successful.append(processed_signal)
            await self.pattern_learner.evolve_patterns()
            
            # Обновляем статистику
            self.metrics.observe_latency(start)
            self.stats["processed"] += 1
            
            # Проверяем и корректируем устойчивость
            await self.triad_monitor.check_and_adjust(
                processed_signal.stability_angle,
                processed_signal.calculate_stability_score()
            )
            
            return {
                "status": "success",
                "target": target,
                "signal_id": processed_signal.id,
                "stability_score": round(processed_signal.calculate_stability_score(), 3)
            }
            
        except Exception as e:
            self.breaker_open = True
            self.metrics.record_error()
            self.stats["failures"] += 1
            print(f"[RAS-CORE] Processing error: {e}")
            
            # Пробуем восстановить устойчивость
            recovery_signal = RASSignal({
                "payload": f"RECOVERY_FROM_ERROR: {str(e)[:50]}",
                "priority": 0.8,
                "stability_angle": GOLDEN_STABILITY_ANGLE,
                "semiotic_tags": ["recovery", "error", "resilience"]
            })
            await self.queue.push(recovery_signal)
            
            return {"status": "error", "error": str(e)}
    
    def enhanced_metrics_snapshot(self):
        """Расширенный снимок метрик с акцентом на устойчивость."""
        base_metrics = self.metrics_snapshot()
        stability_metrics = self.metrics.summary()
        
        # Добавляем информацию об угле
        base_metrics.update({
            "golden_stability_angle": GOLDEN_STABILITY_ANGLE,
            "stability_metrics": stability_metrics,
            "reflection_status": {
                "counter": self.reflection_engine.reflection_counter,
                "current_depth": self.reflection_engine.current_depth,
                "max_depth": MAX_REFLECTION_DEPTH
            },
            "config": {
                "threshold": self.config.threshold,
                "focus_patterns_count": len(self.config.focus_patterns),
                "reflection_cycle_ms": self.config.reflection_cycle_ms
            }
        })
        
        return base_metrics

# ================================================================
# TRIAD STABILITY MONITOR
# ================================================================

class TriadStabilityMonitor:
    """Мониторинг и корректировка устойчивости триады KETER-CHOKMAH-BINAH."""
    
    def __init__(self, ras_core: EnhancedRASCore):
        self.core = ras_core
        self.triad_history = []
        self.last_correction = 0
        self.correction_cooldown = 5.0  # секунд
    
    async def monitor_loop(self):
        """Фоновый мониторинг триады."""
        while True:
            try:
                if hasattr(self.core.bus, 'get_triad_balance'):
                    triad_balance = await self.core.bus.get_triad_balance()
                    if triad_balance:
                        await self.analyze_triad_balance(triad_balance)
            except:
                pass
            
            await asyncio.sleep(2.0)  # Проверка каждые 2 секунды
    
    async def analyze_triad_balance(self, balance: Dict[str, float]):
        """Анализирует баланс триады и вычисляет отклонения."""
        if not all(k in balance for k in ["KETER", "CHOKMAH", "BINAH"]):
            return
        
        # Вычисляем среднее значение
        values = list(balance.values())
        avg_value = sum(values) / 3
        
        # Вычисляем отклонения
        deviations = {k: abs(v - avg_value) for k, v in balance.items()}
        max_deviation = max(deviations.values())
        
        # Сохраняем в историю
        self.triad_history.append({
            "timestamp": time.time(),
            "balance": balance,
            "avg_value": avg_value,
            "max_deviation": max_deviation,
            "deviations": deviations
        })
        
        # Ограничиваем размер истории
        if len(self.triad_history) > 100:
            self.triad_history.pop(0)
        
        # Проверяем необходимость корректировки
        if max_deviation > 0.3:  # Пороговое значение
            await self.suggest_correction(balance, max_deviation, deviations)
    
    async def suggest_correction(self, balance: Dict[str, float], 
                                max_deviation: float, deviations: Dict[str, float]):
        """Предлагает корректировки для восстановления баланса."""
        current_time = time.time()
        if current_time - self.last_correction < self.correction_cooldown:
            return
        
        # Определяем самую отклонённую сефиру
        unbalanced_sephira = max(deviations, key=deviations.get)
        deviation_value = deviations[unbalanced_sephira]
        
        # Создаём корректирующий сигнал
        correction_signal = {
            "payload": f"TRIAD_CORRECTION: {unbalanced_sephira} deviation={deviation_value:.3f}",
            "priority": 0.7,
            "semiotic_tags": ["triad", "correction", "balance", unbalanced_sephira.lower()],
            "stability_angle": GOLDEN_STABILITY_ANGLE,
            "focus_vector": self.get_correction_vector(unbalanced_sephira),
            "correction_metadata": {
                "unbalanced_node": unbalanced_sephira,
                "deviation": deviation_value,
                "current_balance": balance,
                "suggested_action": "load_redistribution"
            }
        }
        
        # Отправляем в DAAT для анализа
        await self.core.bus.publish("DAAT", correction_signal)
        
        print(f"[TRIAD-MONITOR] Correction suggested for {unbalanced_sephira} "
              f"(deviation: {deviation_value:.3f})")
        
        self.last_correction = current_time
    
    def get_correction_vector(self, sephira: str) -> List[float]:
        """Возвращает вектор фокуса для корректировки указанной сефиры."""
        vectors = {
            "KETER": [0.0, 1.0, 0.0],    # Вверх (сознание)
            "CHOKMAH": [1.0, 0.0, 0.0],  # Вправо (интуиция)
            "BINAH": [0.0, 0.0, 1.0]     # Вглубь (анализ)
        }
        return vectors.get(sephira, [0.5, 0.5, 0.5])
    
    async def check_and_adjust(self, current_angle: float, stability_score: float):
        """Проверяет и корректирует параметры на основе текущей устойчивости."""
        if stability_score < 0.6:
            # Увеличиваем порог для более строгой фильтрации
            await self.core.config.evolve_threshold(0.05)
            
            # Генерируем стабилизирующий сигнал
            stabilization_signal = RASSignal({
                "payload": "STABILIZATION_INJECTION",
                "priority": 0.8,
                "stability_angle": GOLDEN_STABILITY_ANGLE,
                "semiotic_tags": ["stabilization", "recovery"],
                "focus_vector": [0.0, 0.0, 1.0]  # Вглубь для стабилизации
            })
            await self.core.queue.push(stabilization_signal)

# ================================================================
# ENHANCED MOCK BUS FOR TESTING
# ================================================================

class EnhancedMockBus(MockBus):
    """Mock bus с поддержкой триадного баланса."""
    
    async def get_triad_balance(self) -> Dict[str, float]:
        """Возвращает текущий баланс триады (для тестирования)."""
        return {
            "KETER": random.uniform(0.4, 0.9),
            "CHOKMAH": random.uniform(0.3, 0.8),
            "BINAH": random.uniform(0.5, 0.95)
        }
    
    async def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """Улучшенная публикация с логированием угла устойчивости."""
        stability_info = ""
        if "stability_angle" in message:
            angle = message["stability_angle"]
            deviation = abs(angle - GOLDEN_STABILITY_ANGLE)
            stability_info = f" ∠{angle:.1f}° (Δ{deviation:.1f}°)"
        
        payload_preview = message.get("payload", "")[:40]
        if len(payload_preview) >= 40:
            payload_preview = payload_preview[:37] + "..."
        
        print(f"[BUS→{topic:8}] {payload_preview:40} {stability_info}")

# ================================================================
# TEST HARNESS WITH 14.4° FOCUS
# ================================================================

async def initialize_enhanced_ras_core() -> EnhancedRASCore:
    """Инициализация улучшенного RAS-CORE."""
    bus = EnhancedMockBus()
    core = EnhancedRASCore(bus)
    
    # Запускаем фоновые задачи
    await core.start_background_tasks()
    
    print("=" * 60)
    print(f"ENHANCED RAS-CORE v4.1 INITIALIZED")
    print(f"Golden Stability Angle: {GOLDEN_STABILITY_ANGLE}°")
    print(f"Focus Patterns: {', '.join(core.config.focus_patterns)}")
    print(f"Reflection Cycle: {core.config.reflection_cycle_ms}ms")
    print("=" * 60)
    
    return core

async def test_stability_focused_processing():
    """Тестирование обработки с фокусом на устойчивость."""
    core = await initialize_enhanced_ras_core()
    
    print("\n[TEST] Generating test signals with varying stability angles...")
    
    # Генерируем тестовые сигналы с разными углами
    test_angles = [
        10.0, 12.0, 14.0, 14.4, 14.8, 16.0, 18.0,  # Вокруг идеального
        5.0, 25.0, 45.0, 90.0                      # Экстремальные
    ]
    
    for i, angle in enumerate(test_angles):
        # Создаём сигнал с указанным углом
        signal_data = {
            "payload": f"Тестовый сигнал {i+1} с углом {angle}°",
            "priority": random.uniform(0.4, 0.9),
            "semiotic_tags": random.choice([
                ["инсайт", "паттерн"],
                ["анализ", "смысл"],
                ["интуиция", "озарение"]
            ]),
            "stability_angle": angle,
            "angle_confidence": random.uniform(0.7, 1.0),
            "focus_vector": [random.uniform(-1, 1) for _ in range(3)],
            "resonance_index": random.uniform(0.3, 0.9)
        }
        
        # Обрабатываем сигнал
        result = await core.process(signal_data)
        
        # Небольшая задержка между сигналами
        await asyncio.sleep(0.05)
    
    # Даём время на обработку
    await asyncio.sleep(0.5)
    
    # Выводим метрики
    print("\n" + "=" * 60)
    print("FINAL METRICS WITH STABILITY FOCUS:")
    print("=" * 60)
    
    enhanced_metrics = core.enhanced_metrics_snapshot()
    print(json.dumps(enhanced_metrics, ensure_ascii=False, indent=2, default=str))
    
    # Выводим статистику по углам
    if core.metrics.angle_deviations:
        close_to_golden = sum(1 for d in core.metrics.angle_deviations if d < 5.0)
        total_angles = len(core.metrics.angle_deviations)
        golden_ratio = close_to_golden / total_angles if total_angles > 0 else 0
        
        print(f"\n[STABILITY ANALYSIS]")
        print(f"Signals close to 14.4° (±5°): {close_to_golden}/{total_angles} ({golden_ratio:.1%})")
        print(f"Average deviation: {statistics.mean(core.metrics.angle_deviations):.2f}°")
        print(f"Minimum deviation: {min(core.metrics.angle_deviations):.2f}°")
    
    return core

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TESTING RAS-CORE v4.1 WITH 14.4° STABILITY ANGLE INTEGRATION")
    print("=" * 60)
    
    asyncio.run(test_stability_focused_processing())
