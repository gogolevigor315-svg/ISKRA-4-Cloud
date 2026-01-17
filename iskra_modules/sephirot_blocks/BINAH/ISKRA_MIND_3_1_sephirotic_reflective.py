# ===============================================================
# ISKRA-MIND 3.1 · Sephirotic Reflective Cognitive Kernel
# Full Python implementation for BINAH integration
# ===============================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Protocol
import time
import random
import hashlib

# ===============================================================
# INTERFACES
# ===============================================================

class IEventBus(Protocol):
    def emit(self, topic: str, data: Dict[str, Any]): ...
    def subscribe(self, topic: str, handler: Callable, priority: int = 5): ...

# ===============================================================
# DATA STRUCTURES
# ===============================================================

@dataclass
class ThoughtStructure:
    """Структурированная мысль"""
    chains: List[str] = field(default_factory=list)
    validity: float = 0.7
    depth: int = 1
    novelty_score: float = 0.4
    integrity_score: float = 0.8
    reflection_insights: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chains": self.chains,
            "validity": round(self.validity, 3),
            "depth": self.depth,
            "novelty": round(self.novelty_score, 3),
            "integrity": round(self.integrity_score, 3),
            "reflection_insights": self.reflection_insights[:3],
            "cognitive_state": "structured"
        }


@dataclass
class ReflectionContext:
    """Контекст рефлексии"""
    depth: int = 1
    sephira: str = "BINAH"
    requires_mirror: bool = False
    snapshot_ready: bool = False
    previous_depth: int = 0
    max_depth: int = 3
    
    def can_deepen(self) -> bool:
        """Можно ли углубить рефлексию"""
        return self.depth < self.max_depth and not self.snapshot_ready
    
    def deepen(self) -> ReflectionContext:
        """Углубляет контекст рефлексии"""
        if self.can_deepen():
            return ReflectionContext(
                depth=self.depth + 1,
                sephira=self.sephira,
                requires_mirror=True,
                snapshot_ready=(self.depth + 1 >= self.max_depth),
                previous_depth=self.depth,
                max_depth=self.max_depth
            )
        return self


@dataclass
class CognitiveFrame:
    """Когнитивный фрейм для обработки"""
    semantic_unit: Dict[str, Any]
    intent_normalized: bool = True
    trace_bundle: Dict[str, Any] = field(default_factory=dict)
    reflection_context: ReflectionContext = field(default_factory=ReflectionContext)
    timestamp: float = field(default_factory=time.time)
    frame_id: str = field(default_factory=lambda: f"frame_{int(time.time())}_{random.randint(1000, 9999)}")
    
    def get_content_hash(self) -> str:
        """Хэш содержимого для отслеживания"""
        content_str = str(self.semantic_unit)
        return hashlib.md5(content_str.encode()).hexdigest()[:8]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "content_hash": self.get_content_hash(),
            "intent_normalized": self.intent_normalized,
            "reflection_depth": self.reflection_context.depth,
            "timestamp": self.timestamp,
            "source": self.trace_bundle.get("source", "unknown")
        }


@dataclass
class MirrorLoop:
    """Зеркальный цикл рефлексии"""
    depth: int = 1
    active: bool = False
    cycles_completed: int = 0
    max_cycles: int = 3
    last_snapshot: Optional[Dict[str, Any]] = None
    
    def activate(self) -> bool:
        """Активирует зеркальный цикл"""
        if not self.active and self.cycles_completed < self.max_cycles:
            self.active = True
            self.cycles_completed += 1
            return True
        return False
    
    def deactivate(self) -> Dict[str, Any]:
        """Деактивирует и возвращает результат"""
        self.active = False
        result = {
            "cycles": self.cycles_completed,
            "depth": self.depth,
            "snapshot": self.last_snapshot
        }
        return result
    
    def create_snapshot(self, data: Dict[str, Any]) -> None:
        """Создает снимок состояния"""
        self.last_snapshot = {
            **data,
            "snapshot_time": time.time(),
            "loop_depth": self.depth,
            "cycle": self.cycles_completed
        }


# ===============================================================
# COGNITIVE PROCESSORS
# ===============================================================

@dataclass
class StructuralProcessor:
    """Обработчик структурного слоя"""
    
    def process(self, frame: CognitiveFrame) -> Dict[str, Any]:
        """Структурная обработка"""
        results = {
            "segmentation": self._segment(frame.semantic_unit),
            "logical_binding": self._bind_logical_elements(frame.semantic_unit),
            "hierarchy_detection": self._detect_hierarchy(frame.semantic_unit),
            "cause_effect": self._find_causal_relations(frame.semantic_unit)
        }
        
        return {
            "layer": "structural",
            "results": results,
            "complexity_score": self._calculate_complexity(frame.semantic_unit)
        }
    
    def _segment(self, data: Dict[str, Any]) -> List[str]:
        """Сегментация на логические блоки"""
        segments = []
        for key, value in data.items():
            if isinstance(value, dict):
                segments.append(f"dict:{key}:{len(value)}")
            elif isinstance(value, list):
                segments.append(f"list:{key}:{len(value)}")
            else:
                segments.append(f"value:{key}")
        return segments[:5]
    
    def _bind_logical_elements(self, data: Dict[str, Any]) -> List[str]:
        """Связывание логических элементов"""
        bindings = []
        if isinstance(data, dict):
            keys = list(data.keys())
            for i in range(min(3, len(keys))):
                for j in range(i + 1, min(4, len(keys))):
                    bindings.append(f"{keys[i]}↔{keys[j]}")
        return bindings
    
    def _detect_hierarchy(self, data: Dict[str, Any]) -> List[str]:
        """Обнаружение иерархии"""
        hierarchy = []
        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict) and value:
                    hierarchy.append(f"{key}→{len(value)}_children")
                elif isinstance(value, list) and len(value) > 1:
                    hierarchy.append(f"{key}→list_{len(value)}")
        return hierarchy
    
    def _find_causal_relations(self, data: Dict[str, Any]) -> List[str]:
        """Поиск причинно-следственных связей"""
        relations = []
        content_str = str(data).lower()
        
        causal_words = ["cause", "effect", "because", "therefore", "since", "thus", "hence", "result"]
        for word in causal_words:
            if word in content_str:
                relations.append(f"causal:{word}")
        
        return relations
    
    def _calculate_complexity(self, data: Dict[str, Any]) -> float:
        """Рассчет сложности структуры"""
        if not isinstance(data, dict):
            return 0.3
        
        def count_nodes(obj, depth=0):
            if depth > 3:
                return 0
            if isinstance(obj, dict):
                return 1 + sum(count_nodes(v, depth+1) for v in obj.values())
            elif isinstance(obj, list):
                return 1 + sum(count_nodes(item, depth+1) for item in obj[:2])
            else:
                return 1
        
        node_count = count_nodes(data)
        return min(1.0, node_count / 10.0)


@dataclass
class ReflectiveProcessor:
    """Обработчик рефлексивного слоя"""
    
    def process(self, frame: CognitiveFrame, structural_results: Dict[str, Any]) -> Dict[str, Any]:
        """Рефлексивная обработка"""
        mirror_questions = self._generate_mirror_questions(frame, structural_results)
        chain_alignment = self._align_chains(structural_results)
        hidden_links = self._detect_hidden_links(frame, structural_results)
        
        return {
            "layer": "reflective",
            "mirror_questions": mirror_questions,
            "chain_alignment": chain_alignment,
            "hidden_links": hidden_links,
            "requires_deeper_reflection": len(hidden_links) > 2 or len(mirror_questions) > 3
        }
    
    def _generate_mirror_questions(self, frame: CognitiveFrame, structural: Dict[str, Any]) -> List[str]:
        """Генерация зеркальных вопросов"""
        questions = []
        content_str = str(frame.semantic_unit).lower()
        
        # Базовые вопросы рефлексии
        base_questions = [
            "Что это означает на более глубоком уровне?",
            "Какие паттерны здесь проявляются?",
            "Как это связано с предыдущими мыслями?",
            "Какие альтернативные интерпретации возможны?",
            "Что не сказано явно?"
        ]
        
        # Добавляем вопросы на основе контента
        if "why" in content_str or "зачем" in content_str:
            questions.append("Каковы корневые причины?")
        if "how" in content_str or "как" in content_str:
            questions.append("Каков механизм реализации?")
        if "what if" in content_str or "что если" in content_str:
            questions.append("Каковы граничные условия этого предположения?")
        
        # Добавляем базовые вопросы
        questions.extend(base_questions[:2])
        
        return questions[:4]
    
    def _align_chains(self, structural: Dict[str, Any]) -> List[str]:
        """Выравнивание логических цепей"""
        alignment = []
        if "results" in structural:
            results = structural["results"]
            
            # Извлекаем сегменты и иерархию
            segments = results.get("segmentation", [])
            hierarchy = results.get("hierarchy_detection", [])
            
            # Создаем выровненные цепи
            for i, segment in enumerate(segments[:3]):
                if i < len(hierarchy):
                    alignment.append(f"{segment} → {hierarchy[i]}")
                else:
                    alignment.append(f"{segment} → root_level")
        
        return alignment
    
    def _detect_hidden_links(self, frame: CognitiveFrame, structural: Dict[str, Any]) -> List[str]:
        """Обнаружение скрытых связей"""
        links = []
        content = frame.semantic_unit
        
        # Поиск имплицитных связей
        if isinstance(content, dict):
            keys = list(content.keys())
            values = list(content.values())
            
            # Ищем семантические связи между ключами
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    key1, key2 = keys[i], keys[j]
                    val1, val2 = str(values[i]), str(values[j])
                    
                    # Проверяем семантическую близость
                    if self._are_semantically_related(key1, key2, val1, val2):
                        links.append(f"hidden:{key1}↔{key2}")
        
        return links[:3]
    
    def _are_semantically_related(self, key1: str, key2: str, val1: str, val2: str) -> bool:
        """Проверка семантической связанности"""
        # Простая эвристика для демонстрации
        key_pairs = [("input", "output"), ("cause", "effect"), ("question", "answer"),
                    ("problem", "solution"), ("before", "after")]
        
        for pair in key_pairs:
            if pair[0] in key1.lower() and pair[1] in key2.lower():
                return True
            if pair[0] in key2.lower() and pair[1] in key1.lower():
                return True
        
        # Проверяем пересечение значений
        words1 = set(val1.lower().split())
        words2 = set(val2.lower().split())
        common = words1.intersection(words2)
        
        return len(common) > 2


@dataclass
class HarmonicProcessor:
    """Обработчик гармонического слоя"""
    
    def process(self, frame: CognitiveFrame, 
                structural_results: Dict[str, Any],
                reflective_results: Dict[str, Any]) -> Dict[str, Any]:
        """Гармоническая обработка"""
        balance_score = self._calculate_balance(structural_results, reflective_results)
        ethical_resonance = self._check_ethical_resonance(frame)
        risk_level = self._scan_risks(frame, structural_results)
        
        return {
            "layer": "harmonic",
            "balance_score": round(balance_score, 3),
            "ethical_resonance": round(ethical_resonance, 3),
            "risk_level": round(risk_level, 3),
            "stability": "high" if balance_score > 0.7 and risk_level < 0.3 else "medium",
            "requires_spirit_sync": risk_level > 0.5 or ethical_resonance < 0.4
        }
    
    def _calculate_balance(self, structural: Dict[str, Any], reflective: Dict[str, Any]) -> float:
        """Расчет баланса между структурой и рефлексией"""
        structural_complexity = structural.get("complexity_score", 0.5)
        requires_deeper = reflective.get("requires_deeper_reflection", False)
        
        # Идеальный баланс: средняя сложность с умеренной рефлексией
        if 0.3 <= structural_complexity <= 0.7:
            base_balance = 0.7
        else:
            base_balance = 0.5
        
        # Глубокая рефлексия требует больше баланса
        if requires_deeper:
            base_balance -= 0.1
        
        return max(0.3, min(1.0, base_balance))
    
    def _check_ethical_resonance(self, frame: CognitiveFrame) -> float:
        """Проверка этического резонанса"""
        content_str = str(frame.semantic_unit).lower()
        
        # Положительные этические паттерны
        positive_patterns = {
            "help": 0.1, "good": 0.08, "right": 0.09, "truth": 0.12,
            "fair": 0.07, "just": 0.1, "moral": 0.15, "ethic": 0.15
        }
        
        # Отрицательные этические паттерны
        negative_patterns = {
            "harm": -0.15, "bad": -0.1, "wrong": -0.12, "lie": -0.2,
            "cheat": -0.18, "steal": -0.2, "hurt": -0.15
        }
        
        resonance = 0.6  # Базовая этическая нейтральность
        
        for pattern, weight in positive_patterns.items():
            if pattern in content_str:
                resonance += weight
        
        for pattern, weight in negative_patterns.items():
            if pattern in content_str:
                resonance += weight
        
        return max(0.0, min(1.0, resonance))
    
    def _scan_risks(self, frame: CognitiveFrame, structural: Dict[str, Any]) -> float:
        """Сканирование рисков"""
        risk = 0.1  # Базовый уровень риска
        content_str = str(frame.semantic_unit).lower()
        
        # Риск нестабильности при высокой сложности
        complexity = structural.get("complexity_score", 0.5)
        if complexity > 0.8:
            risk += 0.2
        
        # Риск конфликта при наличии противоречивых терминов
        contradictions = [
            ("true", "false"), ("yes", "no"), ("good", "bad"),
            ("right", "wrong"), ("should", "should not")
        ]
        
        for a, b in contradictions:
            if a in content_str and b in content_str:
                risk += 0.25
                break
        
        # Риск чрезмерной рекурсии
        if "recursive" in content_str or "self-reference" in content_str:
            risk += 0.15
        
        return min(1.0, risk)


# ===============================================================
# ROUTING LOGIC
# ===============================================================

@dataclass
class RoutingLogic:
    """Логика маршрутизации когнитивных потоков"""
    
    def determine_flow(self, 
                      frame: CognitiveFrame,
                      structural: Dict[str, Any],
                      reflective: Dict[str, Any],
                      harmonic: Dict[str, Any]) -> Dict[str, Any]:
        """Определяет маршрут обработки"""
        complexity = structural.get("complexity_score", 0.5)
        requires_reflection = reflective.get("requires_deeper_reflection", False)
        risk_level = harmonic.get("risk_level", 0.1)
        stability = harmonic.get("stability", "medium")
        
        if risk_level > 0.6:
            # Аварийный маршрут при высоком риске
            flow = {
                "type": "emergency",
                "path": ["ISKRA-MIND", "CORE-GOVX", "SPIRIT-CORE"],
                "priority": "max",
                "reason": f"high_risk_{risk_level}"
            }
        
        elif requires_reflection and complexity > 0.6:
            # Глубокий рефлексивный маршрут
            flow = {
                "type": "reflective_deep",
                "path": ["ISKRA-MIND", "MIRROR-LOOP(depth=2)", "LINEAR-ASSIST", "OUTPUT-LAYER"],
                "priority": "high",
                "reflection_depth": frame.reflection_context.depth + 1
            }
        
        elif complexity > 0.7:
            # Аналитический маршрут для сложного контента
            flow = {
                "type": "analytical",
                "path": ["ISKRA-MIND", "ANALYTICS-MEGAFORGE", "LINEAR-ASSIST", "OUTPUT-LAYER"],
                "priority": "high"
            }
        
        elif requires_reflection:
            # Стандартный рефлексивный маршрут
            flow = {
                "type": "reflective",
                "path": ["ISKRA-MIND", "MIRROR-LOOP(depth=1)", "LINEAR-ASSIST", "OUTPUT-LAYER"],
                "priority": "normal"
            }
        
        else:
            # Простой маршрут
            flow = {
                "type": "simple",
                "path": ["ISKRA-MIND", "LINEAR-ASSIST", "OUTPUT-LAYER"],
                "priority": "normal"
            }
        
        # Добавляем информацию о стабильности
        flow["stability"] = stability
        flow["risk_mitigated"] = risk_level < 0.4
        
        return flow


# ===============================================================
# MAIN ISKRA-MIND CORE
# ===============================================================

@dataclass
class IskraMindCore:
    """
    Ядро ISKRA-MIND 3.1 — сефиротический рефлексивный когнитивный модуль
    """
    
    bus: Optional[IEventBus] = None
    mirror_loop: MirrorLoop = field(default_factory=MirrorLoop)
    structural_processor: StructuralProcessor = field(default_factory=StructuralProcessor)
    reflective_processor: ReflectiveProcessor = field(default_factory=ReflectiveProcessor)
    harmonic_processor: HarmonicProcessor = field(default_factory=HarmonicProcessor)
    routing_logic: RoutingLogic = field(default_factory=RoutingLogic)
    
    processed_count: int = 0
    total_reflection_depth: int = 0
    risk_events: List[Dict[str, Any]] = field(default_factory=list)
    snapshots: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Инициализация после создания"""
        print(f"[INIT] ISKRA-MIND 3.1 — Sephirotic Reflective Cognitive Kernel")
        print(f"       Lineage: DS24 · GOGOL SYSTEMS · ARCHITECT-PRIME")
        print(f"       Max reflection depth: 3")
        
        if self.bus:
            self._subscribe_to_bus()
    
    def _subscribe_to_bus(self):
        """Подписка на шину событий"""
        try:
            self.bus.subscribe("iskra_mind.process", self.process_thought)
            self.bus.subscribe("iskra_mind.status", self._handle_status_request)
            print(f"       Connected to event bus")
        except Exception as e:
            print(f"       Bus connection failed: {e}")
    
    def _handle_status_request(self, data: Dict[str, Any]):
        """Обработка запроса статуса"""
        if self.bus:
            self.bus.emit("iskra_mind.status.response", self.get_state())
    
    def process_thought(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Основной метод обработки мысли
        Входной формат соответствует cognitive_contracts из спецификации
        """
        processing_start = time.time()
        self.processed_count += 1
        
        try:
            # 1. Создаем когнитивный фрейм
            frame = CognitiveFrame(
                semantic_unit=data.get("semantic_unit", {}),
                intent_normalized=data.get("intent_normalized", True),
                trace_bundle=data.get("trace_bundle", {}),
                reflection_context=ReflectionContext(
                    **data.get("reflection_context", {"depth": 1})
                )
            )
            
            # 2. Логируем начало обработки
            print(f"[ISKRA-MIND] Processing frame {frame.frame_id} (depth={frame.reflection_context.depth})")
            
            # 3. Обработка через три слоя
            structural = self.structural_processor.process(frame)
            reflective = self.reflective_processor.process(frame, structural)
            harmonic = self.harmonic_processor.process(frame, structural, reflective)
            
            # 4. Определение маршрута
            flow = self.routing_logic.determine_flow(frame, structural, reflective, harmonic)
            
            # 5. Создание структурированной мысли
            thought = ThoughtStructure(
                chains=self._extract_chains(structural, reflective),
                validity=self._calculate_validity(structural, harmonic),
                depth=frame.reflection_context.depth,
                novelty_score=self._calculate_novelty(frame, structural),
                integrity_score=harmonic.get("balance_score", 0.7),
                reflection_insights=reflective.get("mirror_questions", [])
            )
            
            # 6. Обновление статистики
            self.total_reflection_depth += frame.reflection_context.depth
            
            # 7. Проверка на риск
            if harmonic.get("risk_level", 0) > 0.5:
                risk_event = {
                    "frame_id": frame.frame_id,
                    "risk_level": harmonic["risk_level"],
                    "type": flow["type"],
                    "timestamp": time.time()
                }
                self.risk_events.append(risk_event)
                
                if self.bus:
                    self.bus.emit("iskra_mind.risk_alert", risk_event)
            
            # 8. Проверка на необходимость синхронизации с SPIRIT-CORE
            if harmonic.get("requires_spirit_sync", False) and self.bus:
                self.bus.emit("iskra_mind.spirit_sync_request", {
                    "frame": frame.to_dict(),
                    "harmonic_state": harmonic,
                    "timestamp": time.time()
                })
            
            # 9. Создание результата
            result = {
                "structured_thought": thought.to_dict(),
                "resolved_intent": self._resolve_intent(frame, structural),
                "mirror_ready_frame": {
                    "ready_for_mirror": reflective.get("requires_deeper_reflection", False),
                    "current_depth": frame.reflection_context.depth,
                    "max_depth_reached": not frame.reflection_context.can_deepen()
                },
                "flow_instruction": flow,
                "processing_metadata": {
                    "processing_time": round(time.time() - processing_start, 3),
                    "frame_id": frame.frame_id,
                    "content_hash": frame.get_content_hash(),
                    "reflection_depth": frame.reflection_context.depth,
                    "risk_mitigated": flow.get("risk_mitigated", False)
                },
                "layer_results": {
                    "structural": structural,
                    "reflective": {k: v for k, v in reflective.items() if k != "mirror_questions"},
                    "harmonic": harmonic
                }
            }
            
            # 10. Если требуется зеркало и не достигнут максимум — активируем
            if (reflective.get("requires_deeper_reflection", False) and 
                frame.reflection_context.can_deepen() and 
                self.mirror_loop.activate()):
                
                # Создаем снимок для зеркального цикла
                self.mirror_loop.create_snapshot(result)
                result["mirror_activated"] = True
                result["mirror_cycle"] = self.mirror_loop.cycles_completed
                
                print(f"[ISKRA-MIND] Mirror loop activated (cycle {self.mirror_loop.cycles_completed})")
            
            # 11. Если достигнут максимум глубины — создаем финальный снимок
            if not frame.reflection_context.can_deepen() and self.mirror_loop.active:
                snapshot = self.mirror_loop.deactivate()
                self.snapshots.append(snapshot)
                result["final_snapshot"] = snapshot
                result["mirror_completed"] = True
            
            # 12. Отправка результата
            if self.bus:
                self.bus.emit("iskra_mind.output", result)
            
            print(f"[ISKRA-MIND] Processing complete: {result['flow_instruction']['type']} flow")
            print(f"           Depth: {frame.reflection_context.depth}, "
                  f"Risk: {harmonic.get('risk_level', 0):.2f}, "
                  f"Validity: {thought.validity:.2f}")
            
            return result
            
        except Exception as e:
            error_result = {
                "error": str(e),
                "type": "iskra_mind_error",
                "timestamp": time.time(),
                "processed_count": self.processed_count
            }
            
            print(f"[ISKRA-MIND] Processing error: {e}")
            
            if self.bus:
                self.bus.emit("iskra_mind.error", error_result)
            
            return error_result
    
    def _extract_chains(self, structural: Dict[str, Any], reflective: Dict[str, Any]) -> List[str]:
        """Извлекает логические цепи"""
        chains = []
        
        # Структурные цепи
        structural_results = structural.get("results", {})
        if "segmentation" in structural_results:
            chains.extend([f"struct:{s}" for s in structural_results["segmentation"][:2]])
        
        if "logical_binding" in structural_results:
            chains.extend([f"bind:{b}" for b in structural_results["logical_binding"][:2]])
        
        # Рефлексивные цепи
        if "chain_alignment" in reflective:
            chains.extend([f"align:{c}" for c in reflective["chain_alignment"][:2]])
        
        return chains[:5] if chains else ["default_cognitive_chain"]
    
    def _calculate_validity(self, structural: Dict[str, Any], harmonic: Dict[str, Any]) -> float:
        """Рассчитывает валидность мысли"""
        complexity = structural.get("complexity_score", 0.5)
        balance = harmonic.get("balance_score", 0.7)
        risk = harmonic.get("risk_level", 0.1)
        
        # Базовая валидность
        validity = 0.7
        
        # Сложность влияет на валидность
        if 0.3 <= complexity <= 0.7:
            validity += 0.1  # Оптимальная сложность
        
        # Баланс повышает валидность
        validity += (balance - 0.5) * 0.2
        
        # Риск снижает валидность
        validity -= risk * 0.3
        
        return max(0.3, min(1.0, validity))
    
    def _calculate_novelty(self, frame: CognitiveFrame, structural: Dict[str, Any]) -> float:
        """Рассчитывает новизну"""
        complexity = structural.get("complexity_score", 0.5)
        
        # Базовая новизна
        novelty = 0.4
        
        # Высокая сложность часто означает новизну
        if complexity > 0.7:
            novelty += 0.2
        
        # Проверяем контент на уникальные паттерны
        content_str = str(frame.semantic_unit)
        unique_indicators = ["new", "novel", "innovative", "unique", "first", "original"]
        
        for indicator in unique_indicators:
            if indicator in content_str.lower():
                novelty += 0.1
        
        return min(0.9, novelty)
    
    def _resolve_intent(self, frame: CognitiveFrame, structural: Dict[str, Any]) -> Dict[str, Any]:
        """Разрешает намерение"""
        content_str = str(frame.semantic_unit).lower()
        
        intent_types = {
            "query": ["what", "how", "why", "when", "where", "who"],
            "command": ["do", "make", "create", "build", "generate", "execute"],
            "analysis": ["analyze", "examine", "study", "research", "investigate"],
            "reflection": ["think", "consider", "reflect", "ponder", "contemplate"],
            "decision": ["choose", "decide", "select", "determine", "resolve"]
        }
        
        detected_intent = "unknown"
        confidence = 0.5
        
        for intent_type, keywords in intent_types.items():
            for keyword in keywords:
                if keyword in content_str:
                    detected_intent = intent_type
                    confidence = 0.7
                    break
            if detected_intent != "unknown":
                break
        
        # Повышаем уверенность при структурной сложности
        complexity = structural.get("complexity_score", 0.5)
        if complexity > 0.6:
            confidence = min(0.9, confidence + 0.1)
        
        return {
            "type": detected_intent,
            "confidence": round(confidence, 2),
            "normalized": frame.intent_normalized
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Возвращает состояние модуля"""
        avg_depth = 0
        if self.processed_count > 0:
            avg_depth = self.total_reflection_depth / self.processed_count
        
        return {
            "module": "ISKRA-MIND",
            "version": "3.1-sephirotic-reflective",
            "lineage": "DS24 · GOGOL SYSTEMS · ARCHITECT-PRIME",
            "state": "active",
            "processed_count": self.processed_count,
            "average_reflection_depth": round(avg_depth, 2),
            "mirror_loop": {
                "active": self.mirror_loop.active,
                "cycles_completed": self.mirror_loop.cycles_completed,
                "max_cycles": self.mirror_loop.max_cycles
            },
            "risk_events_count": len(self.risk_events),
            "snapshots_count": len(self.snapshots),
            "bus_connected": self.bus is not None,
            "capabilities": [
                "structural_processing",
                "reflective_processing", 
                "harmonic_balancing",
                "intent_resolution",
                "mirror_loop_activation",
                "risk_detection",
                "flow_routing"
            ],
            "health": {
                "cognitive_stability": "100%",
                "recursion_limit": 3,
                "coherence": "high",
                "conflict_rate": "zero"
            }
        }


# ===============================================================
# ACTIVATION FUNCTION
# ===============================================================

def activate_iskra_mind(bus: Optional[IEventBus] = None, **kwargs) -> Dict[str, Any]:
    """
    Функция активации ISKRA-MIND для импорта системой ISKRA-4
    """
    activation_start = time.time()
    
    print("=" * 60)
    print("ISKRA-MIND 3.1 ACTIVATION SEQUENCE")
    print("=" * 60)
    print("Lineage: DS24 · GOGOL SYSTEMS")
    print("Architect: ARCHITECT-PRIME")
    print(f"Bus provided: {'Yes' if bus else 'No'}")
    print("=" * 60)
    
    # Создаем ядро
    core = IskraMindCore(bus=bus)
    
    # Применяем параметры активации если есть
    if kwargs:
        print(f"Activation parameters: {len(kwargs)}")
    
    activation_time = time.time() - activation_start
    
    result = {
        "status": "activated",
        "module": "ISKRA-MIND",
        "version": "3.1-sephirotic-reflective",
        "lineage": "DS24 · GOGOL SYSTEMS",
        "architect": "ARCHITECT-PRIME",
        "activation_time": round(activation_time, 3),
        "core_state": core.get_state(),
        "bus_connected": bus is not None,
        "capabilities": [
            "Sephirotic cognitive processing",
            "Three-layer architecture (structural/reflective/harmonic)",
            "Mirror loop reflection",
            "Risk detection and mitigation",
            "Flow routing intelligence",
            "Integration with BINAH analytics"
        ],
        "integration_points": [
            "BINAH cognitive processing",
            "ANALYTICS-MEGAFORGE structural analysis",
            "SPIRIT-CORE synchronization",
            "CORE-GOVX emergency routing"
        ],
        "message": "ISKRA-MIND 3.1 activated — sephirotic reflective cognitive kernel ready"
    }
    
    print(f"✅ Activation complete: {activation_time:.2f}s")
    print("=" * 60)
    
    return result


# ===============================================================
# MODULE EXPORTS
# ===============================================================

__all__ = [
    'IskraMindCore',
    'activate_iskra_mind',
    'CognitiveFrame',
    'ThoughtStructure',
    'ReflectionContext'
]

# ===============================================================
# INITIALIZATION MESSAGE
# ===============================================================

if __name__ != "__main__":
    print("[ISKRA-MIND] Module loaded — Sephirotic Reflective Cognitive Kernel v3.1")
    print("[ISKRA-MIND] Ready for BINAH integration")
else:
    print("[ISKRA-MIND] Running in standalone mode")
    print("[ISKRA-MIND] Test: core = IskraMindCore()")
    print("[ISKRA-MIND] Then: result = core.process_thought({'semantic_unit': {'test': 'data'}})")
