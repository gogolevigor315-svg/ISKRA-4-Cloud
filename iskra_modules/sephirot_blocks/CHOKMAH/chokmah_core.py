# ============================================================
# INTUITION-MATRIX 3.4 · Sephirotic Chokhmah (УСОВЕРШЕНСТВОВАННЫЙ КОД)
# Интеграция с ISKRA-4 Cloud и Sephirotic Engine
# Версия: 3.4.1
# ============================================================

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)

# === КОНФИГУРАЦИЯ ===========================================================

@dataclass
class ChokmahConfig:
    """Конфигурация Chokmah для тонкой настройки"""
    # Резонанс
    INITIAL_RESONANCE: float = 0.3
    ACTIVATION_THRESHOLD: float = 0.6
    MAX_RESONANCE: float = 0.85
    RESONANCE_GROWTH_PER_SIGNAL: float = 0.008  # Рост за сигнал
    MIN_SIGNALS_FOR_GROWTH: int = 5  # Минимум сигналов для роста
    TIME_WINDOW_FOR_GROWTH: float = 30.0  # Секунд для учёта активности
    
    # Энергия
    ENERGY_PER_INSIGHT: float = 0.03
    ENERGY_RECHARGE_RATE: float = 0.01  # В секунду
    
    # Логирование
    LOG_LEVEL: str = "INFO"
    DEBUG_METRICS: bool = True
    
    # Производительность
    PARALLEL_PROCESSING: bool = True
    MAX_CONCURRENT_TASKS: int = 3
    
    def __post_init__(self):
        """Валидация конфигурации"""
        assert 0 < self.INITIAL_RESONANCE <= 1.0, "Некорректный начальный резонанс"
        assert 0 < self.ACTIVATION_THRESHOLD <= 1.0, "Некорректный порог активации"
        assert self.RESONANCE_GROWTH_PER_SIGNAL > 0, "Рост резонанса должен быть > 0"

# === CHOKMAH NODE INTEGRATION (УЛУЧШЕННАЯ) =================================

@dataclass
class ResonanceController:
    """Управление резонансом с защитой от скачков"""
    config: ChokmahConfig
    current_resonance: float = field(default_factory=lambda: ChokmahConfig.INITIAL_RESONANCE)
    signal_counter: int = 0
    last_growth_time: float = field(default_factory=time.time)
    resonance_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def signal_received(self) -> float:
        """Обработка нового сигнала с умным ростом резонанса"""
        self.signal_counter += 1
        now = time.time()
        
        # Проверяем, пора ли увеличивать резонанс
        should_grow = (
            self.signal_counter >= self.config.MIN_SIGNALS_FOR_GROWTH and
            (now - self.last_growth_time) >= self.config.TIME_WINDOW_FOR_GROWTH
        )
        
        if should_grow:
            old_resonance = self.current_resonance
            growth = self.config.RESONANCE_GROWTH_PER_SIGNAL
            
            # Медленный рост при высоком резонансе
            if self.current_resonance > 0.7:
                growth *= 0.5
            
            self.current_resonance = min(
                self.current_resonance + growth,
                self.config.MAX_RESONANCE
            )
            
            self.last_growth_time = now
            self.signal_counter = 0
            
            # Логируем изменение
            self.resonance_history.append({
                "timestamp": now,
                "old": old_resonance,
                "new": self.current_resonance,
                "growth": growth
            })
            
            # Держим историю разумного размера
            if len(self.resonance_history) > 100:
                self.resonance_history = self.resonance_history[-50:]
            
            logger.debug(
                f"Резонанс вырос: {old_resonance:.3f} → {self.current_resonance:.3f} "
                f"(рост: {growth:.4f})"
            )
        
        return self.current_resonance
    
    def get_resonance_report(self) -> Dict[str, Any]:
        """Отчёт по динамике резонанса"""
        if not self.resonance_history:
            avg_growth = 0.0
        else:
            growths = [h["growth"] for h in self.resonance_history]
            avg_growth = sum(growths) / len(growths)
        
        return {
            "current": round(self.current_resonance, 4),
            "signal_counter": self.signal_counter,
            "time_since_last_growth": time.time() - self.last_growth_time,
            "avg_growth_rate": round(avg_growth, 5),
            "history_size": len(self.resonance_history),
            "status": (
                "sleeping" if self.current_resonance < 0.4 else
                "awakening" if self.current_resonance < 0.6 else
                "active" if self.current_resonance < 0.8 else
                "peak"
            )
        }

class ChokmahNodeIntegration:
    """Улучшенная интеграция с узлом Chokmah"""
    
    def __init__(self, sephirotic_engine=None, config: Optional[ChokmahConfig] = None):
        self.engine = sephirotic_engine
        self.config = config or ChokmahConfig()
        self.node = None
        self.resonance_ctrl = ResonanceController(self.config)
        self.energy = 0.9
        self.signals_processed = 0
        self.energy_last_update = time.time()
        
    async def connect(self) -> Dict[str, Any]:
        """Подключение к существующему узлу Chokmah"""
        try:
            # Получаем узел CHOKHMAH из Sephirotic Engine
            node_found = False
            
            if self.engine:
                # Пробуем разные способы получения узла
                if hasattr(self.engine, 'nodes') and isinstance(self.engine.nodes, dict):
                    self.node = self.engine.nodes.get('CHOKHMAH')
                    node_found = self.node is not None
                elif hasattr(self.engine, 'get_node'):
                    self.node = await self.engine.get_node('CHOKHMAH')
                    node_found = self.node is not None
            
            if not node_found:
                logger.warning("Узел CHOKHMAH не найден в движке, создаю локальную репрезентацию")
                self.node = {
                    'name': 'Мудрость',
                    'sephira': 'CHOKHMAH',
                    'resonance': self.resonance_ctrl.current_resonance,
                    'energy': self.energy,
                    'total_signals_processed': 0,
                    'description': 'Интуиция',
                    'connected_module': 'chernigovskaya'
                }
            
            logger.info(
                f"Узел CHOKHMAH подключен: {self.node.get('name', 'Unknown')} "
                f"(резонанс: {self.resonance_ctrl.current_resonance:.2f})"
            )
            
            return {
                "status": "connected",
                "node": "CHOKHMAH",
                "resonance": self.resonance_ctrl.current_resonance,
                "engine_integrated": node_found
            }
            
        except Exception as e:
            logger.error(f"Ошибка подключения к CHOKHMAH: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    def update_energy(self):
        """Обновление энергии с регенерацией"""
        now = time.time()
        time_passed = now - self.energy_last_update
        
        # Регенерация энергии
        recharge = time_passed * self.config.ENERGY_RECHARGE_RATE
        self.energy = min(1.0, self.energy + recharge)
        
        self.energy_last_update = now
        return self.energy

    def consume_energy(self, amount: float) -> bool:
        """Потребление энергии с проверкой"""
        self.update_energy()
        
        if self.energy >= amount:
            self.energy -= amount
            logger.debug(f"Энергия потреблена: {amount:.3f}, осталось: {self.energy:.3f}")
            return True
        else:
            logger.warning(f"Недостаточно энергии: требуется {amount:.3f}, доступно {self.energy:.3f}")
            return False

    def increment_signals(self):
        """Обработка нового сигнала"""
        self.signals_processed += 1
        self.update_energy()
        
        # Обновляем резонанс
        new_resonance = self.resonance_ctrl.signal_received()
        
        # Синхронизируем с узлом
        if self.node:
            if isinstance(self.node, dict):
                self.node['resonance'] = new_resonance
                self.node['total_signals_processed'] = self.signals_processed
                self.node['energy'] = self.energy
            elif hasattr(self.node, 'resonance'):
                self.node.resonance = new_resonance
                if hasattr(self.node, 'total_signals_processed'):
                    self.node.total_signals_processed = self.signals_processed
                if hasattr(self.node, 'energy'):
                    self.node.energy = self.energy

    def get_status(self) -> Dict[str, Any]:
        """Детальный статус интеграции"""
        resonance_report = self.resonance_ctrl.get_resonance_report()
        
        return {
            "node": "CHOKHMAH",
            "resonance": resonance_report["current"],
            "resonance_status": resonance_report["status"],
            "energy": round(self.energy, 3),
            "signals_processed": self.signals_processed,
            "connected": self.node is not None,
            "node_type": type(self.node).__name__ if self.node else None,
            "resonance_metrics": {
                "signal_counter": self.resonance_ctrl.signal_counter,
                "time_to_next_growth": max(0, self.config.TIME_WINDOW_FOR_GROWTH - 
                                          (time.time() - self.resonance_ctrl.last_growth_time)),
                "growth_threshold": self.config.MIN_SIGNALS_FOR_GROWTH
            },
            "timestamp": datetime.now().isoformat()
        }

# === УЛУЧШЕННАЯ ОБРАБОТКА ЗАПРОСОВ ===========================================

class RequestValidator:
    """Валидация входящих запросов"""
    
    @staticmethod
    def validate_intuition_request(data: Any) -> Dict[str, Any]:
        """Валидация запроса на интуитивный анализ"""
        if not data:
            raise ValueError("Пустой запрос")
        
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                raise ValueError("Некорректный JSON")
        
        if not isinstance(data, dict):
            raise ValueError("Запрос должен быть словарём")
        
        # Обязательные поля или значения по умолчанию
        validated = {
            "text": data.get("text", ""),
            "clarity": float(data.get("clarity", 0.7)),
            "context": data.get("context", {}),
            "urgency": int(data.get("urgency", 1)),
            "confidence": data.get("confidence"),
            "metadata": data.get("metadata", {})
        }
        
        # Валидация значений
        if not 0 <= validated["clarity"] <= 1.0:
            raise ValueError(f"Некорректная ясность: {validated['clarity']}")
        
        if not 1 <= validated["urgency"] <= 5:
            raise ValueError(f"Некорректная срочность: {validated['urgency']}")
        
        if validated["confidence"] is not None:
            try:
                validated["confidence"] = float(validated["confidence"])
                if not 0 <= validated["confidence"] <= 1.0:
                    raise ValueError
            except (ValueError, TypeError):
                raise ValueError(f"Некорректная уверенность: {validated['confidence']}")
        
        return validated

# === PARALLEL PROCESSING =====================================================

class ParallelProcessor:
    """Обработка параллельных задач"""
    
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
    async def process_parallel(self, tasks: List[Dict]) -> List[Any]:
        """Параллельное выполнение задач"""
        if not tasks:
            return []
        
        async def process_with_semaphore(task_func, *args, **kwargs):
            async with self.semaphore:
                return await task_func(*args, **kwargs)
        
        # Запускаем задачи параллельно
        results = await asyncio.gather(
            *[process_with_semaphore(task["func"], *task.get("args", []), 
                                    **task.get("kwargs", {})) 
              for task in tasks],
            return_exceptions=True
        )
        
        # Обрабатываем исключения
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Ошибка в параллельной задаче {i}: {result}")
                final_results.append({"error": str(result), "success": False})
            else:
                final_results.append({"result": result, "success": True})
        
        return final_results

# === INTUITION MATRIX WITH INTEGRATION (УЛУЧШЕННАЯ) =========================

class EnhancedIntuitionMatrix:
    """
    Усиленная версия IntuitionMatrix с улучшениями
    """
    
    def __init__(self, bus: IEventBus, sephirotic_engine=None, config: Optional[ChokmahConfig] = None):
        self.config = config or ChokmahConfig()
        
        # Настройка логирования
        if self.config.DEBUG_METRICS:
            logging.getLogger(__name__).setLevel(logging.DEBUG)
        
        # Основной матрикс
        self.matrix = build_intuition_matrix(bus)
        
        # Интеграции
        self.node_integration = ChokmahNodeIntegration(sephirotic_engine, self.config)
        self.chernigovskaya = ChernigovskayaIntegration()
        
        # Валидатор и параллельный процессор
        self.validator = RequestValidator()
        self.parallel_processor = ParallelProcessor(self.config.MAX_CONCURRENT_TASKS)
        
        # Состояние
        self.activated = False
        self.activation_level = 0.0
        self.start_time = time.time()
        self.total_insights = 0
        
        logger.info(f"EnhancedIntuitionMatrix инициализирован (конфиг: {self.config})")
    
    async def activate(self) -> Dict[str, Any]:
        """Полная активация Chokmah с улучшенной обработкой"""
        try:
            logger.info("Начинаю активацию Chokmah...")
            
            # 1. Подключаемся к узлу Chokmah
            node_status = await self.node_integration.connect()
            if node_status["status"] == "error":
                raise Exception(f"Ошибка подключения к узлу: {node_status.get('error')}")
            
            # 2. Подключаемся к модулю Черниговской
            chern_status = await self.chernigovskaya.connect_to_existing_module()
            logger.info(f"Статус Черниговской: {chern_status['status']}")
            
            # 3. Повышаем резонанс до порога активации
            target_resonance = self.config.ACTIVATION_THRESHOLD
            self.node_integration.resonance_ctrl.current_resonance = target_resonance
            
            # 4. Активируем матрицу
            self.activated = True
            self.activation_level = 0.9
            
            # Потребляем энергию для активации
            energy_used = 0.15
            if not self.node_integration.consume_energy(energy_used):
                logger.warning("Мало энергии для полной активации")
                self.activation_level = 0.7
            
            activation_time = time.time() - self.start_time
            
            logger.info(
                f"Chokmah активирован за {activation_time:.2f} сек! "
                f"Резонанс: {target_resonance:.2f}, "
                f"Энергия: {self.node_integration.energy:.2f}"
            )
            
            return {
                "status": "activated",
                "activation_level": self.activation_level,
                "resonance": target_resonance,
                "energy_used": energy_used,
                "remaining_energy": self.node_integration.energy,
                "activation_time": activation_time,
                "node_integration": node_status,
                "chernigovskaya": chern_status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Критическая ошибка активации Chokmah: {e}", exc_info=True)
            return {
                "status": "activation_failed",
                "error": str(e),
                "activated": False,
                "activation_level": 0.0
            }
    
    async def process_intuition_request(self, request_data: Any) -> Dict[str, Any]:
        """Улучшенная обработка запроса на интуитивный анализ"""
        start_time = time.time()
        
        try:
            # 1. Валидация запроса
            validated_request = self.validator.validate_intuition_request(request_data)
            logger.debug(f"Запрос валидирован: {len(validated_request['text'])} chars")
            
            # 2. Проверка активации и энергии
            if not self.activated:
                return {
                    "status": "not_activated",
                    "error": "Chokmah не активирован",
                    "processing_time": time.time() - start_time
                }
            
            energy_needed = self.config.ENERGY_PER_INSIGHT
            if not self.node_integration.consume_energy(energy_needed):
                return {
                    "status": "insufficient_energy",
                    "required_energy": energy_needed,
                    "available_energy": self.node_integration.energy,
                    "processing_time": time.time() - start_time
                }
            
            # 3. Увеличиваем счётчик сигналов
            self.node_integration.increment_signals()
            
            # 4. Подготавливаем сигналы для матрицы
            signals: IntuitionSignals = {
                "clarity": validated_request["clarity"],
                "resonance": self.node_integration.resonance_ctrl.current_resonance,
                "confidence": validated_request["confidence"]
            }
            
            # 5. Параллельная обработка (если включена)
            if self.config.PARALLEL_PROCESSING and validated_request["text"]:
                tasks = [
                    {
                        "func": self._process_matrix,
                        "args": [signals],
                        "kwargs": {}
                    },
                    {
                        "func": self.chernigovskaya.analyze_text,
                        "args": [validated_request["text"]],
                        "kwargs": {}
                    }
                ]
                
                parallel_results = await self.parallel_processor.process_parallel(tasks)
                
                hypothesis_result = parallel_results[0]
                chern_insight = parallel_results[1]
                
                # Извлекаем результаты
                hypothesis = hypothesis_result["result"] if hypothesis_result["success"] else None
                if chern_insight["success"]:
                    chern_result = chern_insight["result"]
                else:
                    chern_result = {"error": "chernigovskaya_failed"}
            else:
                # Последовательная обработка
                hypothesis = await self._process_matrix(signals)
                chern_insight = {}
                if validated_request["text"]:
                    chern_insight = await self.chernigovskaya.analyze_text(validated_request["text"])
            
            # 6. Проверяем результат матрицы
            if hypothesis is None:
                logger.warning("Матрица вернула None, используем fallback")
                hypothesis = {"status": "fallback", "probability": 0.5}
            elif not isinstance(hypothesis, dict):
                hypothesis = {"raw_result": str(hypothesis), "probability": 0.5}
            
            # 7. Генерация финального инсайта
            final_insight = self._generate_final_insight(
                hypothesis, 
                chern_insight if isinstance(chern_insight, dict) else {},
                validated_request
            )
            
            self.total_insights += 1
            
            # 8. Отправка события в шину
            self.matrix.bus.emit("chokmah.insight.generated", {
                "insight": final_insight,
                "request": validated_request,
                "resonance": self.node_integration.resonance_ctrl.current_resonance,
                "processing_time": time.time() - start_time
            })
            
            processing_time = time.time() - start_time
            
            logger.info(
                f"Инсайт сгенерирован за {processing_time:.3f} сек. "
                f"Уверенность: {final_insight.get('confidence', 0):.2f}, "
                f"Резонанс: {self.node_integration.resonance_ctrl.current_resonance:.3f}"
            )
            
            return {
                "status": "success",
                "insight": final_insight,
                "processing_time": processing_time,
                "energy_used": energy_needed,
                "signals_processed": self.node_integration.signals_processed,
                "total_insights": self.total_insights,
                "current_resonance": self.node_integration.resonance_ctrl.current_resonance,
                "current_energy": self.node_integration.energy,
                "matrix_hypothesis": hypothesis,
                "chernigovskaya_analysis": chern_insight if validated_request["text"] else None
            }
            
        except ValueError as e:
            # Ошибка валидации
            logger.warning(f"Ошибка валидации запроса: {e}")
            return {
                "status": "validation_error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
        except Exception as e:
            # Общая ошибка
            logger.error(f"Ошибка обработки запроса: {e}", exc_info=True)
            return {
                "status": "processing_error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _process_matrix(self, signals: IntuitionSignals) -> Optional[Dict[str, Any]]:
        """Безопасная обработка через матрицу"""
        try:
            # Используем Circuit Breaker из матрицы
            if hasattr(self.matrix.weaver, 'generate'):
                result = self.matrix.weaver.generate(signals)
                
                # Обрабатываем результат Circuit Breaker
                if isinstance(result, dict) and result.get("status") in ["failure", "circuit_open"]:
                    logger.warning(f"Circuit Breaker состояние: {result.get('status')}")
                    return None
                
                # Проверяем, что это валидный результат
                if isinstance(result, dict) and "probability" in result:
                    self.matrix.buffer.add(result)
                    if hasattr(self.matrix.monitor, 'update'):
                        self.matrix.monitor.update(
                            queue_size=self.matrix.buffer.get_queue_size(), 
                            failures=0
                        )
                    return result
                
            return None
        except Exception as e:
            logger.error(f"Ошибка в _process_matrix: {e}")
            return None
    
    def _generate_final_insight(self, hypothesis: Dict[str, Any], 
                               chern_insight: Dict[str, Any], 
                               request: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация финального интуитивного прозрения"""
        
        # Базовые параметры
        base_probability = hypothesis.get("probability", 0.5) if hypothesis else 0.5
        resonance = self.node_integration.resonance_ctrl.current_resonance
        
        # Корректировка на основе резонанса
        resonance_boost = resonance * 0.3  # 0.0-0.3
        final_confidence = min(0.95, base_probability + resonance_boost)
        
        # Определение типа инсайта
        insight_types = [
            ("semantic_leap", 0.3, "Chokmah обнаружил семантический скачок"),
            ("pattern_connection", 0.3, "Chokmah соединил паттерны в целостную картину"),
            ("hidden_structure", 0.2, "Chokmah распознал скрытую структуру"),
            ("metaphorical_link", 0.1, "Chokmah установил метафорическую связь"),
            ("temporal_insight", 0.1, "Chokmah получил временное прозрение")
        ]
        
        # Выбор на основе вероятности
        import random
        r = random.random()
        cumulative = 0
        for insight_type, prob, message in insight_types:
            cumulative += prob
            if r <= cumulative:
                selected_type = insight_type
                selected_message = message
                break
        else:
            selected_type, selected_message = insight_types[0][0], insight_types[0][2]
        
        # Формируем компоненты
        components = ["intuition_matrix_3.4"]
        if chern_insight and not chern_insight.get("error"):
            components.append("chernigovskaya_analysis")
        if resonance > 0.6:
            components.append("resonance_boost")
        
        return {
            "type": selected_type,
            "message": selected_message,
            "confidence": round(final_confidence, 3),
            "resonance_level": round(resonance, 3),
            "resonance_boost": round(resonance_boost, 3),
            "timestamp": datetime.now().isoformat(),
            "components_used": components,
            "actionable": final_confidence > 0.65,
            "energy_cost": self.config.ENERGY_PER_INSIGHT,
            "urgency": request.get("urgency", 1),
            "context_hint": request.get("context", {}).get("hint", "")
        }
    
    def get_status_report(self) -> Dict[str, Any]:
        """Детальный отчёт о состоянии"""
        node_status = self.node_integration.get_status()
        matrix_status = self.matrix.monitor.report() if hasattr(self.matrix.monitor, 'report') else {}
        
        uptime = time.time() - self.start_time
        
        return {
            "sephira": "CHOKHMAH",
            "activated": self.activated,
            "activation_level": self.activation_level,
            "uptime_seconds": round(uptime, 1),
            "uptime_human": str(timedelta(seconds=int(uptime))),
            "total_insights": self.total_insights,
            "matrix": {
                "health": matrix_status.get("health", "unknown"),
                "queue_size": matrix_status.get("queue_size", 0),
                "failures": matrix_status.get("failures", 0)
            },
            "node": node_status,
            "chernigovskaya": {
                "connected": self.chernigovskaya.connected,
                "mock_mode": isinstance(getattr(self.chernigovskaya.module, '__class__', None), MockChernigovskaya)
            },
            "config": {
                "parallel_processing": self.config.PARALLEL_PROCESSING,
                "max_concurrent_tasks": self.config.MAX_CONCURRENT_TASKS,
                "debug_metrics": self.config.DEBUG_METRICS
            },
            "system_time": datetime.now().isoformat()
        }

# === API ENDPOINT INTEGRATION (УЛУЧШЕННЫЕ) ===================================

def create_chokmah_api_endpoints(app):
    """Создание API эндпоинтов для Chokmah с улучшенной обработкой ошибок"""
    
    chokmah_instance = None
    config = ChokmahConfig()
    
    @app.route('/chokmah/activate', methods=['POST'])
    async def activate_chokmah():
        """Активация сефиры Chokmah"""
        nonlocal chokmah_instance
        
        try:
            # Проверяем, не активирован ли уже
            if chokmah_instance and chokmah_instance.activated:
                status = chokmah_instance.get_status_report()
                return {
                    "status": "already_activated",
                    "since": status["uptime_human"],
                    "resonance": status["node"]["resonance"],
                    "energy": status["node"]["energy"]
                }, 200
            
            # Получаем шину событий
            try:
                from sephirot_bus import get_global_bus
                bus = get_global_bus()
            except ImportError:
                logger.warning("Глобальная шина не найдена, создаю локальную")
                bus = PriorityEventBus()
            
            # Создаём и активируем
            chokmah_instance = EnhancedIntuitionMatrix(bus, config=config)
            result = await chokmah_instance.activate()
            
            if result["status"] == "activated":
                return result, 200
            else:
                return result, 500
                
        except Exception as e:
            logger.error(f"Ошибка в /chokmah/activate: {e}", exc_info=True)
            return {
                "status": "endpoint_error",
                "error": str(e)
            }, 500
    
    @app.route('/chokmah/insight', methods=['POST'])
    async def get_insight():
        """Получение интуитивного прозрения"""
        nonlocal chokmah_instance
        
        try:
            # Проверяем экземпляр
            if not chokmah_instance:
                return {
                    "status": "not_initialized",
                    "error": "Chokmah не инициализирован"
                }, 400
            
            # Получаем JSON
            from flask import request
            if not request.is_json:
                return {
                    "status": "invalid_content_type",
                    "error": "Требуется application/json"
                }, 400
            
            request_data = request.get_json(silent=True)
            if request_data is None:
                return {
                    "status": "invalid_json",
                    "error": "Некорректный JSON"
                }, 400
            
            # Обрабатываем запрос
            result = await chokmah_instance.process_intuition_request(request_data)
            
            # Определяем HTTP статус
            status_code = 200
            if result["status"] in ["not_activated", "insufficient_energy"]:
                status_code = 400
            elif result["status"] in ["validation_error", "processing_error"]:
                status_code = 422
            
            return result, status_code
            
        except Exception as e:
            logger.error(f"Ошибка в /chokmah/insight: {e}", exc_info=True)
            return {
                "status": "endpoint_error",
                "error": str(e)
            }, 500
    
    @app.route('/chokmah/status', methods=['GET'])
    async def chokmah_status():
        """Статус сефиры Chokmah"""
        nonlocal chokmah_instance
        
        try:
            if not chokmah_instance:
                return {
                    "status": "not_initialized",
                    "sephira": "CHOKHMAH",
                    "resonance": config.INITIAL_RESONANCE,
                    "message": "Используйте POST /chokmah/activate для активации"
                }, 200
            
            status_report = chokmah_instance.get_status_report()
            return status_report, 200
            
        except Exception as e:
            logger.error(f"Ошибка в /chokmah/status: {e}", exc_info=True)
            return {
                "status": "endpoint_error",
                "error": str(e)
            }, 500
    
    @app.route('/chokmah/debug', methods=['GET'])
    async def chokmah_debug():
        """Отладочная информация"""
        nonlocal chokmah_instance
        
        try:
            from flask import request
            debug_level = request.args.get('level', 'basic')
            
            response = {
                "sephira": "CHOKHMAH",
                "instance_exists": chokmah_instance is not None,
                "config": {
                    "INITIAL_RESONANCE": config.INITIAL_RESONANCE,
                    "ACTIVATION_THRESHOLD": config.ACTIVATION_THRESHOLD,
                    "MAX_RESONANCE": config.MAX_RESONANCE,
                    "PARALLEL_PROCESSING": config.PARALLEL_PROCESSING
                },
                "system": {
                    "time": datetime.now().isoformat(),
                    "python_version": sys.version
                }
            }
            
            if chokmah_instance and debug_level == "detailed":
                node_status = chokmah_instance.node_integration.get_status()
                response["detailed_status"] = node_status
            
            return response, 200
            
        except Exception as e:
            return {
                "status": "debug_error",
                "error": str(e)
            }, 500
    
    @app.route('/chokmah/resonance', methods=['GET'])
    async def get_resonance():
        """Текущий резонанс Chokmah"""
        nonlocal chokmah_instance
        
        try:
            if not chokmah_instance:
                return {
                    "sephira": "CHOKHMAH",
                    "resonance": config.INITIAL_RESONANCE,
                    "status": "sleeping"
                }, 200
            
            node_status = chokmah_instance.node_integration.get_status()
            
            return {
                "sephira": "CHOKHMAH",
                "resonance": node_status["resonance"],
                "resonance_status": node_status["resonance_status"],
                "signals_processed": node_status["signals_processed"],
                "energy": node_status["energy"],
                "thresholds": {
                    "sleeping": "< 0.4",
                    "awakening": "0.4 - 0.6",
                    "active": "0.6 - 0.8",
                    "peak": "> 0.8"
                }
            }, 200
            
        except Exception as e:
            logger.error(f"Ошибка в /chokmah/resonance: {e}")
            return {"error": str(e)}, 500
    
    logger.info("Chokmah API endpoints registered (enhanced)")
    return app

# === ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ =================================================

class ChernigovskayaIntegration:
    """Интеграция с существующим модулем Черниговской"""
    
    def __init__(self):
        self.module = None
        self.connected = False
        self.last_connection_attempt = 0
        self.connection_cooldown = 10.0  # секунд между попытками
        
    async def connect_to_existing_module(self) -> Dict[str, Any]:
        """Подключение к уже существующему модулю chernigovskaya"""
        now = time.time()
        
        # Проверяем кд на подключение
        if now - self.last_connection_attempt < self.connection_cooldown:
            logger.debug(f"КД на подключение к Черниговской: {self.connection_cooldown} сек")
            return {
                "status": "cooldown",
                "module": "chernigovskaya",
                "cooldown_remaining": self.connection_cooldown - (now - self.last_connection_attempt)
            }
        
        self.last_connection_attempt = now
        
        try:
            # Пытаемся импортировать существующий модуль
            import sys
            import os
            
            # Путь к модулю Черниговской
            chernigovskaya_path = "bechtereva_chernigovskaya.chernigovskaya"
            
            try:
                # Попытка импорта
                import importlib
                module = importlib.import_module("bechtereva_chernigovskaya.chernigovskaya")
                self.module = module
                self.connected = True
                
                logger.info("Успешно подключено к модулю Черниговской")
                return {
                    "status": "connected",
                    "module": "chernigovskaya",
                    "path": chernigovskaya_path,
                    "functions_available": [f for f in dir(module) if not f.startswith('_')]
                }
            except ImportError as e:
                logger.warning(f"Модуль Черниговской не найден: {e}")
                # Создаём заглушку для разработки
                self.module = MockChernigovskaya()
                self.connected = True
                return {
                    "status": "mock_mode",
                    "module": "chernigovskaya_mock",
                    "note": "Режим заглушки для разработки",
                    "warning": "Настоящий модуль не найден"
                }
                
        except Exception as e:
            logger.error(f"Ошибка подключения к Черниговской: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}
    
    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Анализ текста через модуль Черниговской"""
        if not self.connected or not self.module:
            logger.warning("Попытка анализа без подключенного модуля Черниговской")
            return {
                "error": "Модуль Черниговской не подключен",
                "text_preview": text[:100] + "..." if len(text) > 100 else text
            }
        
        start_time = time.time()
        
        try:
            # Пробуем разные методы
            if hasattr(self.module, 'analyze'):
                # Асинхронный или синхронный вызов
                if asyncio.iscoroutinefunction(self.module.analyze):
                    result = await self.module.analyze(text)
                else:
                    result = self.module.analyze(text)
            elif hasattr(self.module, 'process'):
                if asyncio.iscoroutinefunction(self.module.process):
                    result = await self.module.process(text)
                else:
                    result = self.module.process(text)
            elif isinstance(self.module, MockChernigovskaya):
                result = self.module.analyze(text)
            else:
                # Фолбэк
                result = {
                    "linguistic_patterns": ["basic_analysis"],
                    "source": "chernigovskaya_fallback",
                    "text_length": len(text),
                    "warning": "Метод analyze не найден"
                }
            
            processing_time = time.time() - start_time
            logger.debug(f"Анализ Черниговской занял {processing_time:.3f} сек")
            
            # Добавляем метаданные
            if isinstance(result, dict):
                result["_metadata"] = {
                    "processing_time": processing_time,
                    "text_length": len(text),
                    "module_type": "chernigovskaya_real" if not isinstance(self.module, MockChernigovskaya) else "chernigovskaya_mock"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка анализа текста через Черниговскую: {e}", exc_info=True)
            return {
                "error": str(e),
                "source": "chernigovskaya",
                "processing_time": time.time() - start_time
            }


class MockChernigovskaya:
    """Заглушка модуля Черниговской для разработки"""
    
    def __init__(self):
        self.analysis_count = 0
        self.patterns_db = {
            "semantic": ["subject_object", "cause_effect", "comparison", "contrast"],
            "syntactic": ["simple_sentence", "complex_sentence", "question", "exclamation"],
            "pragmatic": ["request", "statement", "command", "question"],
            "emotional": ["positive", "negative", "neutral", "mixed"]
        }
    
    def analyze(self, text: str) -> Dict[str, Any]:
        """Мок-анализ текста"""
        self.analysis_count += 1
        
        import random
        import hashlib
        
        # Генерация детерминированных, но разнообразных результатов
        text_hash = hashlib.md5(text.encode()).hexdigest()
        hash_int = int(text_hash[:8], 16)
        
        # Определяем паттерны на основе хэша
        selected_patterns = []
        for pattern_type, patterns in self.patterns_db.items():
            if hash_int % (self.analysis_count + 1) % 2 == 0:
                selected_patterns.append(random.choice(patterns))
        
        # Сложность текста
        word_count = len(text.split())
        complexity = min(word_count / 50, 1.0)
        
        # Эмоциональный тон
        emotional_words = ["хорош", "плох", "отлич", "ужас", "рад", "груст"]
        emotional_score = 0.0
        for word in emotional_words:
            if word in text.lower():
                emotional_score += 0.1
        
        return {
            "linguistic_patterns": selected_patterns,
            "semantic_network": {
                "nodes": word_count // 10,
                "connections": word_count // 5,
                "density": complexity
            },
            "neurolinguistic_score": 0.5 + (complexity * 0.3) + (emotional_score * 0.2),
            "processing_time": random.uniform(0.01, 0.1),
            "processed_by": "chernigovskaya_mock",
            "text_preview": text[:50] + "..." if len(text) > 50 else text,
            "analysis_type": "mock_analysis",
            "analysis_id": self.analysis_count,
            "text_hash": text_hash[:8],
            "metrics": {
                "word_count": word_count,
                "complexity": complexity,
                "emotional_score": emotional_score
            }
        }


# === MAIN И ТЕСТИРОВАНИЕ =====================================================

if __name__ == "__main__":
    # Настройка расширенного логирования для тестов
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"chokmah_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    import sys
    
    async def run_comprehensive_test():
        """Всестороннее тестирование Chokmah"""
        print("=" * 60)
        print("🧠 ВСЕСТОРОННЕЕ ТЕСТИРОВАНИЕ CHOKMAH")
        print("=" * 60)
        
        # 1. Создание экземпляра
        print("\n1. Создание EnhancedIntuitionMatrix...")
        bus = PriorityEventBus()
        chokmah = EnhancedIntuitionMatrix(bus)
        
        # 2. Активация
        print("\n2. Активация Chokmah...")
        activation_result = await chokmah.activate()
        
        print(f"   Статус: {activation_result['status']}")
        print(f"   Уровень активации: {activation_result.get('activation_level', 0)}")
        print(f"   Резонанс: {activation_result.get('resonance', 0)}")
        print(f"   Энергия: {activation_result.get('remaining_energy', 0)}")
        
        if activation_result["status"] != "activated":
            print("   ❌ Активация не удалась!")
            return
        
        # 3. Проверка статуса
        print("\n3. Проверка статуса...")
        status = chokmah.get_status_report()
        print(f"   Активирован: {status['activated']}")
        print(f"   Uptime: {status['uptime_human']}")
        print(f"   Инсайтов: {status['total_insights']}")
        print(f"   Резонанс: {status['node']['resonance']}")
        
        # 4. Тестовые запросы
        print("\n4. Тестовые запросы...")
        
        test_cases = [
            {
                "text": "Что скрыто за этим паттерном поведения системы?",
                "clarity": 0.8,
                "context": {"source": "system_diagnostics"},
                "urgency": 2
            },
            {
                "text": "Какие скрытые связи существуют между модулями Keter и Chokmah?",
                "clarity": 0.6,
                "context": {"domain": "sephirotic_architecture"},
                "confidence": 0.7,
                "urgency": 3
            },
            {
                "text": "",  # Пустой текст
                "clarity": 0.5,
                "context": {"test": "empty_text"}
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n   Запрос {i}: '{test_case.get('text', 'NO_TEXT')[:30]}...'")
            
            result = await chokmah.process_intuition_request(test_case)
            
            print(f"     Статус: {result['status']}")
            if result['status'] == 'success':
                insight = result.get('insight', {})
                print(f"     Тип инсайта: {insight.get('type', 'N/A')}")
                print(f"     Уверенность: {insight.get('confidence', 0)}")
                print(f"     Действенный: {insight.get('actionable', False)}")
                print(f"     Время обработки: {result.get('processing_time', 0):.3f} сек")
                print(f"     Энергия потрачена: {result.get('energy_used', 0)}")
            else:
                print(f"     Ошибка: {result.get('error', 'Unknown error')}")
        
        # 5. Серия запросов для проверки роста резонанса
        print("\n5. Проверка роста резонанса при нагрузке...")
        
        initial_resonance = chokmah.node_integration.resonance_ctrl.current_resonance
        print(f"   Начальный резонанс: {initial_resonance:.3f}")
        
        # Делаем несколько быстрых запросов
        quick_requests = 15
        print(f"   Отправка {quick_requests} быстрых запросов...")
        
        tasks = []
        for j in range(quick_requests):
            task_data = {
                "text": f"Быстрый запрос #{j+1} для тестирования",
                "clarity": 0.5,
                "urgency": 1
            }
            task = chokmah.process_intuition_request(task_data)
            tasks.append(task)
        
        # Запускаем параллельно
        import asyncio
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = sum(1 for r in results if isinstance(r, dict) and r.get('status') == 'success')
        print(f"   Успешных запросов: {successful}/{quick_requests}")
        
        # 6. Финальный статус
        print("\n6. Финальный статус...")
        final_status = chokmah.get_status_report()
        final_resonance = final_status['node']['resonance']
        
        print(f"   Финальный резонанс: {final_resonance:.3f}")
        print(f"   Изменение резонанса: {final_resonance - initial_resonance:+.3f}")
        print(f"   Всего сигналов: {final_status['node']['signals_processed']}")
        print(f"   Энергия: {final_status['node']['energy']:.3f}")
        
        resonance_status = final_status['node']['resonance_status']
        status_symbol = {
            'sleeping': '💤',
            'awakening': '🌅',
            'active': '🌟',
            'peak': '⚡'
        }.get(resonance_status, '❓')
        
        print(f"\n   Статус резонанса: {resonance_status} {status_symbol}")
        
        # 7. Генерация отчёта
        print("\n7. Генерация детального отчёта...")
        
        resonance_report = chokmah.node_integration.resonance_ctrl.get_resonance_report()
        
        report = {
            "test_timestamp": datetime.now().isoformat(),
            "chokmah_status": final_status,
            "resonance_dynamics": resonance_report,
            "test_summary": {
                "total_requests": len(test_cases) + quick_requests,
                "successful_requests": successful + sum(1 for r in results[:len(test_cases)] 
                                                      if isinstance(r, dict) and r.get('status') == 'success'),
                "resonance_growth": final_resonance - initial_resonance,
                "activation_successful": activation_result['status'] == 'activated',
                "total_processing_time": sum(r.get('processing_time', 0) for r in results 
                                            if isinstance(r, dict))
            }
        }
        
        # Сохраняем отчёт в файл
        report_filename = f"chokmah_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        import json
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"   Отчёт сохранён в: {report_filename}")
        
        # 8. Визуализация
        print("\n8. Визуализация состояния...")
        
        resonance_bar = "█" * int(final_resonance * 20) + "░" * (20 - int(final_resonance * 20))
        energy_bar = "█" * int(final_status['node']['energy'] * 20) + "░" * (20 - int(final_status['node']['energy'] * 20))
        
        print(f"   Резонанс: [{resonance_bar}] {final_resonance:.2f}")
        print(f"   Энергия:  [{energy_bar}] {final_status['node']['energy']:.2f}")
        
        # Определяем рекомендации
        recommendations = []
        if final_resonance < 0.5:
            recommendations.append("Увеличьте количество запросов для пробуждения Chokmah")
        if final_status['node']['energy'] < 0.3:
            recommendations.append("Дайте системе время на восстановление энергии")
        if successful / (len(test_cases) + quick_requests) < 0.7:
            recommendations.append("Проверьте конфигурацию системы или качество запросов")
        
        if recommendations:
            print("\n   📋 Рекомендации:")
            for rec in recommendations:
                print(f"     • {rec}")
        
        print("\n" + "=" * 60)
        print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
        print("=" * 60)
    
    # Запуск тестов
    try:
        asyncio.run(run_comprehensive_test())
    except KeyboardInterrupt:
        print("\n\nТестирование прервано пользователем")
    except Exception as e:
        print(f"\n\n❌ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()


# === УТИЛИТЫ ДЛЯ РАБОТЫ С СИСТЕМОЙ =========================================

def integrate_chokmah_into_system(sephirotic_engine=None, flask_app=None):
    """
    Основная функция интеграции Chokmah в систему ISKRA-4
    
    Использование:
        chokmah_core = integrate_chokmah_into_system(sephirotic_engine, app)
    """
    logger.info("Начинаю интеграцию Chokmah в систему ISKRA-4...")
    
    try:
        # 1. Получаем или создаём шину событий
        bus = None
        try:
            from sephirot_bus import get_global_bus
            bus = get_global_bus()
            logger.info("Использую глобальную шину событий ISKRA-4")
        except ImportError:
            logger.warning("Глобальная шина не найдена, создаю локальную")
            bus = PriorityEventBus()
        
        # 2. Создаём ядро Chokmah
        chokmah_core = EnhancedIntuitionMatrix(
            bus=bus,
            sephirotic_engine=sephirotic_engine,
            config=ChokmahConfig(DEBUG_METRICS=True, PARALLEL_PROCESSING=True)
        )
        
        logger.info("Ядро Chokmah создано")
        
        # 3. Регистрируем API эндпоинты, если передан Flask app
        if flask_app:
            create_chokmah_api_endpoints(flask_app)
            logger.info("API эндпоинты Chokmah зарегистрированы")
        
        # 4. Автоматическая активация (опционально)
        async def auto_activate():
            try:
                result = await chokmah_core.activate()
                if result["status"] == "activated":
                    logger.info(f"Chokmah автоматически активирован (резонанс: {result['resonance']})")
                else:
                    logger.warning(f"Автоактивация не удалась: {result.get('error', 'unknown')}")
            except Exception as e:
                logger.error(f"Ошибка автоактивации: {e}")
        
        # Запускаем активацию в фоне
        import threading
        activation_thread = threading.Thread(
            target=lambda: asyncio.run(auto_activate()),
            daemon=True,
            name="ChokmahAutoActivation"
        )
        activation_thread.start()
        
        # 5. Создаём мониторинговую задачу
        async def monitoring_task():
            """Фоновая задача мониторинга состояния Chokmah"""
            while True:
                try:
                    await asyncio.sleep(60)  # Каждую минуту
                    if chokmah_core.activated:
                        status = chokmah_core.get_status_report()
                        logger.debug(
                            f"Chokmah мониторинг: "
                            f"резонанс={status['node']['resonance']:.2f}, "
                            f"энергия={status['node']['energy']:.2f}, "
                            f"инсайтов={status['total_insights']}"
                        )
                except Exception as e:
                    logger.error(f"Ошибка в задаче мониторинга: {e}")
                    await asyncio.sleep(10)
        
        monitoring_thread = threading.Thread(
            target=lambda: asyncio.run(monitoring_task()),
            daemon=True,
            name="ChokmahMonitoring"
        )
        monitoring_thread.start()
        
        logger.info("Chokmah успешно интегрирован в систему ISKRA-4")
        
        return chokmah_core
        
    except Exception as e:
        logger.error(f"Критическая ошибка интеграции Chokmah: {e}", exc_info=True)
        raise


# === ЭКСПОРТ ОСНОВНЫХ ФУНКЦИЙ ===============================================

__all__ = [
    'EnhancedIntuitionMatrix',
    'ChokmahConfig',
    'ChokmahNodeIntegration',
    'ChernigovskayaIntegration',
    'RequestValidator',
    'ResonanceController',
    'ParallelProcessor',
    'create_chokmah_api_endpoints',
    'integrate_chokmah_into_system',
    'create_chokmah_core',
    'build_intuition_matrix',
    'IntuitionMatrix',
    'PriorityEventBus',
    'IntuitionSignals',
    'TimingService',
    'CircuitBreaker'
]

# === ПРОСТАЯ ФУНКЦИЯ ДЛЯ БЫСТРОГО СТАРТА ==================================

def create_chokmah_core(sephirotic_engine=None, bus=None, config=None):
    """
    Быстрое создание ядра Chokmah
    
    Args:
        sephirotic_engine: Движок сефирот (опционально)
        bus: Шина событий (опционально)
        config: Конфигурация (опционально)
    
    Returns:
        EnhancedIntuitionMatrix instance
    """
    if bus is None:
        bus = PriorityEventBus()
    
    if config is None:
        config = ChokmahConfig()
    
    return EnhancedIntuitionMatrix(bus, sephirotic_engine, config)

# === ПРОСТАЯ ДОКУМЕНТАЦИЯ ==================================================

CHOKMAH_DOCS = """
Chokmah Core v3.4.1 - Ядро интуитивной сефиры

Основные классы:
- EnhancedIntuitionMatrix: Основной класс с интеграцией в ISKRA-4
- ChokmahConfig: Конфигурация параметров резонанса и энергии
- ChokmahNodeIntegration: Интеграция с узлом Chokmah в Sephirotic Engine
- ChernigovskayaIntegration: Подключение к модулю Черниговской

Быстрый старт:
1. from sephirot_blocks.CHOKMAH.chokmah_core import create_chokmah_core
2. chokmah = create_chokmah_core(sephirotic_engine)
3. await chokmah.activate()
4. result = await chokmah.process_intuition_request({"text": "Ваш запрос"})

API эндпоинты:
- POST /chokmah/activate - активация
- POST /chokmah/insight - получение инсайта
- GET /chokmah/status - статус системы
- GET /chokmah/resonance - текущий резонанс

Резонансные уровни:
- 0.3-0.4: sleeping (спящий)
- 0.4-0.6: awakening (пробуждающийся)
- 0.6-0.8: active (активный)
- >0.8: peak (пиковый)
"""

# === ПРОВЕРКА ЗАВИСИМОСТЕЙ ================================================

def check_dependencies():
    """Проверка необходимых зависимостей"""
    import sys
    
    dependencies = {
        'asyncio': True,
        'dataclasses': sys.version_info >= (3, 7),
        'typing': True,
        'json': True,
        'time': True,
        'threading': True,
        'logging': True
    }
    
    missing = []
    for dep, available in dependencies.items():
        if not available:
            missing.append(dep)
    
    if missing:
        logger.warning(f"Отсутствуют зависимости: {missing}")
        return False
    
    return True

# === ИНИЦИАЛИЗАЦИЯ МОДУЛЯ =================================================

# Автоматическая проверка зависимостей при импорте
if check_dependencies():
    logger.debug("Все зависимости Chokmah доступны")
else:
    logger.warning("Некоторые зависимости Chokmah недоступны")



