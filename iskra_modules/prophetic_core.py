"""
============================================================
PROPHETIC_CORE v4.1 · БАЗОВЫЙ МОДУЛЬ ПРЕДВИДЕНИЯ ISKRA-4 (УЛУЧШЕННАЯ ВЕРСИЯ)
Author: ISKRA-4 Architect
Enhancements: Автокоррекция, LRU-кэш, Policy fallback, DataBridge интеграция
============================================================
"""

import datetime
import asyncio
import math
import logging
import time
import hashlib
from typing import Dict, List, Any, Tuple, Optional
from collections import OrderedDict
from functools import lru_cache

# ---------------------------------------------------------------
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("PropheticCore")

# ===============================================================
# УМНЫЙ LRU КЭШ С МЕТРИКАМИ ПОПАДАНИЙ
# ===============================================================

class IntelligentCache:
    """LRU-кэш с метриками попаданий и автоматическим вытеснением"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.creation_times = {}
        
    def get(self, key: str) -> Optional[Any]:
        """Получение значения с обновлением порядка и счетчиков"""
        if key in self.cache:
            # Перемещаем в конец (самый недавно использованный)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.hit_count += 1
            return value
        self.miss_count += 1
        return None
    
    def set(self, key: str, value: Any):
        """Сохранение значения с автоматическим вытеснением"""
        if key in self.cache:
            # Обновляем существующее значение
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            # Вытесняем самый старый элемент (первый в OrderedDict)
            oldest_key = next(iter(self.cache))
            self.cache.pop(oldest_key)
            if oldest_key in self.creation_times:
                del self.creation_times[oldest_key]
        
        self.cache[key] = value
        self.creation_times[key] = time.time()
    
    def clear(self):
        """Очистка кэша"""
        cleared = len(self.cache)
        self.cache.clear()
        self.creation_times.clear()
        return cleared
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика кэша"""
        total_accesses = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_accesses if total_accesses > 0 else 0
        
        # Вычисляем средний возраст записей
        now = time.time()
        ages = [now - t for t in self.creation_times.values()]
        avg_age = sum(ages) / len(ages) if ages else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": round(hit_rate, 4),
            "total_accesses": total_accesses,
            "avg_entry_age_sec": round(avg_age, 2)
        }

# ===============================================================
# САМООБУЧАЮЩИЙСЯ РЕЗОНАНСНЫЙ ОРАКУЛ
# ===============================================================

class SelfLearningResonanceOracle:
    """Нелинейная интеграция резонанса с автокоррекцией"""
    
    def __init__(self, adaptation_rate: float = 0.1):
        self._synergy_cache = {}
        self._weight_profiles = {
            "ethical_decision": (0.75, 0.20, 0.05),
            "emotional_action": (0.30, 0.60, 0.10),
            "willful_act": (0.25, 0.25, 0.50),
            "default": (0.50, 0.30, 0.20)
        }
        self._adaptation_rate = adaptation_rate
        self._correction_history = []
        self._prediction_errors = []
        
    def calculate_synergy(self, emotional_score: float, 
                         ethical_score: float, 
                         will_score: float) -> float:
        """Вычисление синергетического эффекта с кэшированием"""
        cache_key = f"{emotional_score:.2f}_{ethical_score:.2f}_{will_score:.2f}"
        if cache_key in self._synergy_cache:
            return self._synergy_cache[cache_key]
        
        # Нелинейная синергия с пороговыми эффектами
        emotional_ethical_synergy = math.sqrt(emotional_score * ethical_score)
        
        # Пороговый эффект воли
        if will_score > 0.8:
            will_amplification = 1.0 + (will_score - 0.8) * 2
        elif will_score < 0.3:
            will_amplification = 0.5 * will_score
        else:
            will_amplification = math.sin(will_score * math.pi / 2)
        
        # Комбинированная синергия
        synergy = (emotional_ethical_synergy * will_amplification)
        synergy = round(min(1.5, max(0.0, synergy)), 4)  # Ограничение 0-1.5
        
        self._synergy_cache[cache_key] = synergy
        return synergy
    
    def integrate_resonance(self,
                           emotional_profile: Dict[str, float],
                           ethical_profile: Dict[str, float],
                           will_factor: float,
                           action_type: str = "default") -> Dict[str, Any]:
        """Интеграция резонанса с автокоррекцией"""
        # Нормализация и взвешивание
        em_score = self._weighted_average(emotional_profile, 
                                         {"harmony": 0.4, "clarity": 0.3, "balance": 0.3})
        et_score = self._weighted_average(ethical_profile,
                                         {"truth": 0.4, "love": 0.3, "freedom": 0.3})
        
        weights = self._weight_profiles.get(action_type, self._weight_profiles["default"])
        
        # Базовый расчет с коррекцией на основе ошибок
        linear_component = (weights[0] * et_score + 
                          weights[1] * em_score + 
                          weights[2] * will_factor)
        
        # Синергия с коррекцией
        synergy = self.calculate_synergy(em_score, et_score, will_factor)
        
        # Применение автокоррекции
        correction = self._get_correction_factor(action_type)
        corrected_component = linear_component * (1 + correction)
        
        # Итоговый резонанс
        final_resonance = corrected_component * (1 + synergy * 0.5)
        final_resonance = round(min(1.0, max(0.0, final_resonance)), 4)
        
        # Гармонический статус
        harmony_status = self._determine_harmony_status(final_resonance)
        
        log.debug(f"🔮 ResonanceOracle: {final_resonance} ({harmony_status}) "
                 f"[коррекция: {correction:.3f}]")
        
        return {
            "resonance_score": final_resonance,
            "harmony_status": harmony_status,
            "components": {
                "emotional": round(em_score, 4),
                "ethical": round(et_score, 4),
                "will": round(will_factor, 4)
            },
            "synergy_factor": synergy,
            "correction_applied": correction,
            "action_type": action_type,
            "weights_used": weights
        }
    
    def _weighted_average(self, values: Dict[str, float], 
                         weights: Dict[str, float]) -> float:
        """Взвешенное среднее с fallback"""
        if not values:
            return 0.5
        
        weighted_sum = 0
        total_weight = 0
        
        for key, weight in weights.items():
            value = values.get(key, 0.5)
            weighted_sum += value * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    def _determine_harmony_status(self, resonance: float) -> str:
        """Определение статуса гармонии"""
        if resonance > 0.85:
            return "высшая_гармония"
        elif resonance > 0.70:
            return "сбалансированный"
        elif resonance > 0.55:
            return "нейтральный"
        elif resonance > 0.40:
            return "легкая_дисгармония"
        elif resonance > 0.25:
            return "значительная_дисгармония"
        else:
            return "критический_разлад"
    
    def _get_correction_factor(self, action_type: str) -> float:
        """Получение фактора коррекции на основе истории ошибок"""
        if not self._prediction_errors:
            return 0.0
        
        # Фильтруем ошибки по типу действия
        relevant_errors = [e for e in self._prediction_errors 
                          if e.get("action_type") == action_type]
        
        if not relevant_errors:
            return 0.0
        
        # Вычисляем среднюю ошибку
        avg_error = sum(e["error"] for e in relevant_errors[-10:]) / len(relevant_errors[-10:])
        
        # Адаптивная коррекция (знак обратный ошибке)
        correction = -avg_error * self._adaptation_rate
        return round(correction, 4)
    
    def record_prediction_error(self, 
                               predicted: float, 
                               actual: float, 
                               action_type: str = "default"):
        """Запись ошибки предсказания для обучения"""
        error = predicted - actual
        self._prediction_errors.append({
            "timestamp": time.time(),
            "predicted": predicted,
            "actual": actual,
            "error": error,
            "action_type": action_type
        })
        
        # Ограничиваем историю
        if len(self._prediction_errors) > 1000:
            self._prediction_errors = self._prediction_errors[-500:]
        
        # Адаптация весов на основе ошибки
        self._adapt_weights(action_type, error)
    
    def _adapt_weights(self, action_type: str, error: float):
        """Адаптация весов на основе ошибки"""
        if action_type not in self._weight_profiles:
            return
        
        # Простая градиентная адаптация
        old_weights = list(self._weight_profiles[action_type])
        correction = error * self._adaptation_rate * 0.1
        
        # Корректируем веса (сохраняя сумму = 1.0)
        new_weights = [
            max(0.05, min(0.9, w + correction * (1 if i == 0 else -0.5)))
            for i, w in enumerate(old_weights)
        ]
        
        # Нормализация
        total = sum(new_weights)
        if total > 0:
            normalized = [w / total for w in new_weights]
            self._weight_profiles[action_type] = tuple(normalized)
            
            self._correction_history.append({
                "timestamp": time.time(),
                "action_type": action_type,
                "old_weights": old_weights,
                "new_weights": self._weight_profiles[action_type],
                "error": error
            })

# ===============================================================
# POLICYGOVERNOR FALLBACK СИСТЕМА
# ===============================================================

class PolicyFallbackSystem:
    """Автоматическая оценка рисков при отсутствии PolicyGovernor"""
    
    def __init__(self):
        self.risk_patterns = {
            "data_modification": 0.6,
            "system_access": 0.7,
            "user_interaction": 0.4,
            "configuration_change": 0.8,
            "external_communication": 0.5
        }
        
        self.ethical_heuristics = {
            "truth_violation": 0.9,
            "harm_potential": 0.8,
            "autonomy_breach": 0.7,
            "fairness_issue": 0.6,
            "privacy_risk": 0.75
        }
    
    async def assess_without_policy_governor(self,
                                           scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Оценка рисков при отсутствии PolicyGovernor"""
        action_intent = scenario.get("action", {}).get("intent", "").lower()
        resonance = scenario.get("resonance", {}).get("resonance_score", 0.5)
        
        # Анализ намерения
        risk_score = 0.0
        risk_factors = []
        
        for pattern, base_risk in self.risk_patterns.items():
            if pattern in action_intent:
                risk_score += base_risk * 0.2
                risk_factors.append(pattern)
        
        # Этическая эвристика
        ethical_concerns = []
        for heuristic, weight in self.ethical_heuristics.items():
            # Простая проверка по ключевым словам
            trigger_words = {
                "truth_violation": ["обман", "ложь", "фальсификация"],
                "harm_potential": ["вред", "повредить", "разрушить"],
                "autonomy_breach": ["принудить", "заставить", "контролировать"],
                "fairness_issue": ["несправедливо", "дискриминация", "предвзято"],
                "privacy_risk": ["личные", "конфиденциально", "приватность"]
            }
            
            for word in trigger_words.get(heuristic, []):
                if word in action_intent:
                    risk_score += weight * 0.15
                    ethical_concerns.append(heuristic)
                    break
        
        # Коррекция на основе резонанса
        resonance_modifier = 1.0 - resonance  # Низкий резонанс = выше риск
        adjusted_risk = risk_score * resonance_modifier
        adjusted_risk = min(1.0, max(0.0, adjusted_risk))
        
        # Определение действия
        if adjusted_risk > 0.8:
            action = "immediate_block"
            allowed = False
        elif adjusted_risk > 0.6:
            action = "require_manual_review"
            allowed = False
        elif adjusted_risk > 0.4:
            action = "warn_and_proceed"
            allowed = True
        else:
            action = "allow_with_logging"
            allowed = True
        
        return {
            "status": "fallback_assessment",
            "allowed": allowed,
            "action": action,
            "risk_level": round(adjusted_risk, 3),
            "risk_factors": risk_factors,
            "ethical_concerns": ethical_concerns,
            "resonance_modifier": round(resonance_modifier, 3),
            "note": "Оценка выполнена системой fallback (PolicyGovernor недоступен)"
        }

# ===============================================================
# DATABRIDGE ИНТЕГРАЦИЯ
# ===============================================================

class DataBridgeIntegration:
    """Интеграция метрик с DataBridge для мониторинга"""
    
    def __init__(self, data_bridge_module: Optional[Any] = None):
        self.data_bridge = data_bridge_module
        self.metrics_buffer = []
        self.buffer_limit = 100
        self.last_flush = time.time()
        self.flush_interval = 60  # секунды
    
    async def send_metric(self, 
                         metric_type: str,
                         value: float,
                         tags: Dict[str, str] = None):
        """Отправка метрики через DataBridge"""
        metric = {
            "type": metric_type,
            "value": value,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "tags": tags or {},
            "source": "prophetic_core"
        }
        
        self.metrics_buffer.append(metric)
        
        # Автоматическая отправка при заполнении буфера или по таймеру
        if (len(self.metrics_buffer) >= self.buffer_limit or 
            time.time() - self.last_flush >= self.flush_interval):
            await self.flush_metrics()
    
    async def flush_metrics(self):
        """Отправка всех метрик из буфера"""
        if not self.data_bridge or not self.metrics_buffer:
            return
        
        try:
            # Используем существующий интерфейс DataBridge
            await self.data_bridge.send_batch_metrics(self.metrics_buffer)
            
            log.debug(f"📊 Отправлено {len(self.metrics_buffer)} метрик через DataBridge")
            self.metrics_buffer.clear()
            self.last_flush = time.time()
            
        except Exception as e:
            log.warning(f"Не удалось отправить метрики через DataBridge: {e}")
            # Сохраняем метрики для следующей попытки
            if len(self.metrics_buffer) > self.buffer_limit * 2:
                # Не даем буферу расти бесконечно
                self.metrics_buffer = self.metrics_buffer[-self.buffer_limit:]
    
    async def send_performance_metrics(self, 
                                      operation: str,
                                      duration_ms: float,
                                      success: bool = True):
        """Отправка метрик производительности"""
        tags = {
            "operation": operation,
            "success": str(success),
            "module": "prophetic_core"
        }
        
        await self.send_metric("performance_duration_ms", duration_ms, tags)
        await self.send_metric("performance_success_rate", 1.0 if success else 0.0, tags)
    
    async def send_accuracy_metric(self, 
                                  prediction_hash: str,
                                  predicted: float,
                                  actual: Optional[float] = None):
        """Отправка метрик точности предсказаний"""
        tags = {
            "prediction_hash": prediction_hash[:16],
            "module": "prophetic_core"
        }
        
        await self.send_metric("prediction_value", predicted, tags)
        
        if actual is not None:
            accuracy = 1.0 - abs(predicted - actual)
            await self.send_metric("prediction_accuracy", accuracy, tags)
            await self.send_metric("prediction_error", abs(predicted - actual), tags)

# ===============================================================
# УЛУЧШЕННЫЙ PROPHETIC_CORE
# ===============================================================

class EnhancedPropheticCore:
    """Улучшенная версия PropheticCore с автокоррекцией"""
    
    def __init__(self):
        self.name = "prophetic_core"
        self.version = "4.1-enhanced"
        self.state = {"status": "init"}
        
        # Инициализация улучшенных компонентов
        self.causal_vision = CausalVision()
        self.resonance_oracle = SelfLearningResonanceOracle(adaptation_rate=0.15)
        self.scenario_prophet = ScenarioProphet()
        self.ethical_seer = EthicalSeer()
        self.policy_fallback = PolicyFallbackSystem()
        self.metrics = VisionMetrics()
        
        # Умный кэш
        self.prediction_cache = IntelligentCache(max_size=1500)
        
        # Интеграция с DataBridge
        self.data_bridge_integration = DataBridgeIntegration()
        
        # Связи с системой
        self.system_links = {
            "spinal_core": None,
            "emotional_weave": None,
            "justice_guard": None,
            "neocortex_core": None,
            "policy_governor": None,
            "data_bridge": None,
            "sephirotic_engine": None,
        }
        
        log.info(f"🔮 EnhancedPropheticCore v{self.version} инициализирован")
    
    async def initialize(self) -> bool:
        """Инициализация с улучшенной диагностикой"""
        self.state["status"] = "ready"
        self.state["initialized_at"] = datetime.datetime.utcnow().isoformat()
        
        # Инициализация DataBridge интеграции
        if self.system_links.get("data_bridge"):
            self.data_bridge_integration.data_bridge = self.system_links["data_bridge"]
        
        log.info(f"✅ EnhancedPropheticCore готов к работе")
        
        # Отправка метрики инициализации
        await self.data_bridge_integration.send_metric(
            "module_initialized",
            1.0,
            {"module": "prophetic_core", "version": self.version}
        )
        
        return True
    
    async def foresee_action(self,
                            action_intent: Dict[str, Any],
                            context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Улучшенный метод предвидения с автокоррекцией
        """
        start_time = time.perf_counter()
        
        # Подготовка контекста
        context = context or {}
        
        # Генерация хэша с улучшенной схемой
        prediction_hash = self._generate_prediction_hash(action_intent, context)
        
        # Проверка интеллектуального кэша
        cached_result = self.prediction_cache.get(prediction_hash)
        if cached_result:
            log.debug(f"⚡ Кэш-попадание для {action_intent.get('intent', 'unknown')}")
            cached_result["performance"]["cache_status"] = "hit"
            cached_result["performance"]["hit_rate"] = self.prediction_cache.get_stats()["hit_rate"]
            
            # Отправляем метрику кэш-попадания
            await self.data_bridge_integration.send_metric(
                "cache_hit",
                1.0,
                {"intent": action_intent.get("intent", "unknown")}
            )
            
            return cached_result
        
        log.info(f"🔮 Начинаю предвидение для: {action_intent.get('intent', 'unknown')}")
        
        # Основной процесс предвидения (упрощенный для примера)
        # ... (здесь будет полная логика из предыдущей версии)
        
        # Формирование отчета
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        
        report = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "action_intent": action_intent.get("intent", "unknown"),
            "resonance_integration": {
                "resonance_score": 0.75,  # Пример
                "harmony_status": "сбалансированный"
            },
            "performance": {
                "processing_ms": round(duration_ms, 2),
                "cache_status": "miss",
                "cache_hit_rate": self.prediction_cache.get_stats()["hit_rate"]
            },
            "prediction_hash": prediction_hash[:16]
        }
        
        # Кэширование
        self.prediction_cache.set(prediction_hash, report)
        
        # Отправка метрик производительности
        await self.data_bridge_integration.send_performance_metrics(
            "foresee_action",
            duration_ms,
            success=True
        )
        
        # Отправка метрики кэш-промаха
        await self.data_bridge_integration.send_metric(
            "cache_miss",
            1.0,
            {"intent": action_intent.get("intent", "unknown")}
        )
        
        log.info(f"✅ Предвидение завершено за {duration_ms:.1f} мс")
        return report
    
    def _generate_prediction_hash(self, 
                                 action_intent: Dict[str, Any], 
                                 context: Dict[str, Any]) -> str:
        """Генерация уникального хэша для предвидения"""
        # Сериализация с учетом порядка ключей
        import json
        intent_str = json.dumps(action_intent, sort_keys=True)
        context_str = json.dumps(context, sort_keys=True)
        
        # Хэширование
        combined = f"{intent_str}::{context_str}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    async def process_feedback(self,
                              prediction_hash: str,
                              actual_resonance: float,
                              actual_outcome: Dict[str, Any] = None):
        """
        Обработка обратной связи для самообучения
        """
        # Получаем оригинальное предсказание
        original_prediction = self.prediction_cache.get(prediction_hash)
        if not original_prediction:
            log.warning(f"Не найден оригинальный прогноз для хэша {prediction_hash[:16]}")
            return
        
        predicted_resonance = original_prediction["resonance_integration"]["resonance_score"]
        action_type = original_prediction.get("action_type", "default")
        
        # Запись ошибки для обучения
        self.resonance_oracle.record_prediction_error(
            predicted_resonance,
            actual_resonance,
            action_type
        )
        
        # Обновление метрик
        self.metrics.record_prediction(
            original_prediction["action_intent"],
            predicted_resonance,
            actual_resonance
        )
        
        # Отправка метрик точности
        await self.data_bridge_integration.send_accuracy_metric(
            prediction_hash,
            predicted_resonance,
            actual_resonance
        )
        
        log.info(f"📝 Обратная связь обработана. "
                f"Ошибка: {abs(predicted_resonance - actual_resonance):.3f}")
    
    async def connect_module(self, module_name: str, module_instance: Any):
        """Улучшенное подключение модулей"""
        if module_name in self.system_links:
            self.system_links[module_name] = module_instance
            log.info(f"🔗 Подключен модуль: {module_name}")
            
            # Специальная обработка для DataBridge
            if module_name == "data_bridge":
                self.data_bridge_integration.data_bridge = module_instance
                
                # Отправляем уведомление о подключении
                await self.data_bridge_integration.send_metric(
                    "module_connected",
                    1.0,
                    {"connected_module": module_name}
                )
    
    async def diagnostics(self) -> Dict[str, Any]:
        """Расширенная диагностика"""
        perf_report = self.metrics.get_performance_report()
        cache_stats = self.prediction_cache.get_stats()
        
        # Собираем статистику обучения
        learning_stats = {
            "prediction_errors": len(self.resonance_oracle._prediction_errors),
            "correction_history": len(self.resonance_oracle._correction_history),
            "adaptation_rate": self.resonance_oracle._adaptation_rate
        }
        
        # Текущие веса профилей
        weight_profiles = {
            k: list(v) for k, v in self.resonance_oracle._weight_profiles.items()
        }
        
        return {
            "module": self.name,
            "version": self.version,
            "status": self.state["status"],
            "initialized_at": self.state.get("initialized_at"),
            "performance": perf_report,
            "cache": cache_stats,
            "learning": learning_stats,
            "weight_profiles": weight_profiles,
            "connected_modules": {
                name: "connected" if module else "disconnected"
                for name, module in self.system_links.items()
            },
            "data_bridge": {
                "connected": bool(self.data_bridge_integration.data_bridge),
                "metrics_in_buffer": len(self.data_bridge_integration.metrics_buffer),
                "last_flush": self.data_bridge_integration.last_flush
            },
            "timestamp": datetime.datetime.utcnow().isoformat()
        }
    
    async def clear_cache(self, reason: str = "manual"):
        """Очистка кэша с логированием причины"""
        cleared = self.prediction_cache.clear()
        
        # Отправляем метрику очистки
        await self.data_bridge_integration.send_metric(
            "cache_cleared",
            cleared,
            {"reason": reason, "module": "prophetic_core"}
        )
        
        log.info(f"🧹 Кэш очищен ({reason}): удалено {cleared} записей")
        return {"cleared_entries": cleared, "reason": reason}

# ===============================================================
# ФАБРИЧНЫЕ ФУНКЦИИ
# ===============================================================

async def create_enhanced_prophetic_core() -> EnhancedPropheticCore:
    """Создание улучшенного экземпляра PropheticCore"""
    core = EnhancedPropheticCore()
    await core.initialize()
    return core

# ===============================================================
# ТЕСТИРОВАНИЕ УЛУЧШЕНИЙ
# ===============================================================

async def test_enhancements():
    """Тестирование улучшенного функционала"""
    print("🧪 Тестирую улучшения PropheticCore v4.1...")
    
    # Создание экземпляра
    prophetic = EnhancedPropheticCore()
    await prophetic.initialize()
    
    # Тест кэша
    print("\n📊 Тест интеллектуального кэша:")
    for i in range(5):
        action = {"intent": f"test_action_{i}"}
        result = await prophetic.foresee_action(action)
        print(f"  Предвидение {i}: {result['performance']['cache_status']}")
    
    cache_stats = prophetic.prediction_cache.get_stats()
    print(f"  Статистика кэша: {cache_stats['hit_rate']*100:.1f}% попаданий")
    
    # Тест самообучения
    print("\n🤖 Тест самообучения:")
    
    # Имитация обратной связи
    test_hash = list(prophetic.prediction_cache.cache.keys())[0] if prophetic.prediction_cache.cache else "test"
    await prophetic.process_feedback(test_hash, 0.8)
    
    diagnostics = await prophetic.diagnostics()
    print(f"  Ошибок в истории: {diagnostics['learning']['prediction_errors']}")
    print(f"  Коррекций весов: {diagnostics['learning']['correction_history']}")
    
    # Тест диагностики
    print("\n📈 Расширенная диагностика:")
    print(f"  Размер кэша: {diagnostics['cache']['size']}/{diagnostics['cache']['max_size']}")
    print(f"  DataBridge: {'подключен' if diagnostics['data_bridge']['connected'] else 'отключен'}")
    print(f"  Веса профилей: {len(diagnostics['weight_profiles'])}")
    
    # Очистка кэша
    print("\n🧹 Тест очистки кэша:")
    clear_result = await prophetic.clear_cache("test_purge")
    print(f"  Очищено записей: {clear_result['cleared_entries']}")
    
    return diagnostics

# ===============================================================
# ТОЧКА ВХОДА
# ===============================================================

if __name__ == "__main__":
    asyncio.run(test_enhancements())
