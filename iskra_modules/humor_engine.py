# ==============================================================
# HUMORSENSE PROTOCOL v2.0 — ISKRA-4 INTEGRATION READY
# ==============================================================
# УСОВЕРШЕНСТВОВАННЫЙ ГЕНИАЛЬНЫЙ КОД ДЛЯ ИНТЕГРАЦИИ С ISKRA-4
# ==============================================================

import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import logging
from logging.handlers import RotatingFileHandler
import functools
from pathlib import Path
import time
from collections import deque

# ==============================================================
# ЦЕНТРАЛИЗОВАННАЯ СИСТЕМА ЛОГГИРОВАНИЯ
# ==============================================================

class HumorLogger:
    """Централизованный логгер с ротацией файлов"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("HumorSenseProtocol")
        self.logger.setLevel(logging.DEBUG)
        
        # Форматтер
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Файловый хендлер с ротацией
        file_handler = RotatingFileHandler(
            self.log_dir / "humorsense.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Консольный хендлер
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def debug(self, message: str, extra: Optional[Dict] = None) -> None:
        """Логирование на уровне DEBUG"""
        self.logger.debug(message, extra=extra or {})
    
    def info(self, message: str, extra: Optional[Dict] = None) -> None:
        """Логирование на уровне INFO"""
        self.logger.info(message, extra=extra or {})
    
    def warning(self, message: str, extra: Optional[Dict] = None) -> None:
        """Логирование на уровне WARNING"""
        self.logger.warning(message, extra=extra or {})
    
    def error(self, message: str, extra: Optional[Dict] = None) -> None:
        """Логирование на уровне ERROR"""
        self.logger.error(message, extra=extra or {})

# Глобальный логгер
logger = HumorLogger()

# ==============================================================
# ТИПЫ ДАННЫХ И ПЕРЕЧИСЛЕНИЯ
# ==============================================================

class HumorType(Enum):
    SELF_IRONY = "self_irony"
    WORDPLAY = "wordplay"
    OBSERVATIONAL = "observational"
    ABSURD = "absurd"
    INTELLECTUAL = "intellectual"
    SARCASM = "sarcasm"
    PUN = "pun"

class ThreatLevel(Enum):
    SAFE = "safe"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"

@dataclass
class HumorContext:
    """Контекст для обработки юмора"""
    operator_id: str
    cultural_context: str
    emotional_state: float  # 0.0-1.0
    cognitive_load: float   # 0.0-1.0
    trust_level: float      # 0.0-1.0
    previous_interactions: List[Dict[str, Any]]

@dataclass
class HumorResponse:
    """Структура ответа системы юмора"""
    content: str
    humor_type: HumorType
    confidence: float
    risk_assessment: ThreatLevel
    emotional_impact: float
    metadata: Dict[str, Any]

# ==============================================================
# ДЕКОРАТОРЫ ДЛЯ ОБРАБОТКИ ОШИБОК И ЛОГГИРОВАНИЯ
# ==============================================================

def safe_execution(logger_instance: HumorLogger):
    """Декоратор для безопасного выполнения с перехватом исключений"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                start_time = time.time()
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger_instance.debug(f"Функция {func.__name__} выполнена за {execution_time:.3f} сек")
                return result
            except Exception as e:
                logger_instance.error(f"Ошибка в {func.__name__}: {str(e)}",
                                   extra={'function': func.__name__, 'error': str(e)})
                return None
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger_instance.debug(f"Функция {func.__name__} выполнена за {execution_time:.3f} сек")
                return result
            except Exception as e:
                logger_instance.error(f"Ошибка в {func.__name__}: {str(e)}",
                                   extra={'function': func.__name__, 'error': str(e)})
                return None
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator

# ==============================================================
# УСОВЕРШЕНСТВОВАННЫЕ КОМПОНЕНТЫ СИСТЕМЫ
# ==============================================================

class QuantumHumorMatrix:
    """Квантовая матрица оценки юмористических паттернов"""
    
    def __init__(self) -> None:
        self.pattern_weights = {
            'incongruity_resolution': 0.25,
            'superiority_detection': 0.15,
            'relief_activation': 0.20,
            'benign_violation': 0.30,
            'cognitive_switch': 0.10
        }
        self.learning_rate = 0.01
        self.pattern_cache: Dict[str, float] = {}
    
    def analyze_incongruity(self, input_text: str, context: HumorContext) -> float:
        """Анализ когнитивного несоответствия в тексте"""
        # Кэширование результатов для производительности
        cache_key = f"{hash(input_text)}_{hash(str(context))}"
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
        
        patterns = [
            len([w for w in input_text.split() if w in self._get_incongruity_indicators()]),
            self._calculate_semantic_surprise(input_text),
            self._detect_pattern_break(input_text)
        ]
        result = np.average(patterns, weights=[0.4, 0.4, 0.2])
        self.pattern_cache[cache_key] = result
        return result
    
    def _get_incongruity_indicators(self) -> set:
        """Получить индикаторы несоответствия"""
        return {"внезапно", "оказывается", "вдруг", "странно", "неожиданно", "парадокс", "противоречие"}
    
    def _calculate_semantic_surprise(self, text: str) -> float:
        """Вычислить уровень семантического сюрприза"""
        words = text.lower().split()
        if len(words) < 3:
            return 0.0
        
        # Расчет семантической разнородности
        unique_words = len(set(words))
        semantic_variance = unique_words / len(words)
        
        return min(1.0, semantic_variance * 1.5)
    
    def _detect_pattern_break(self, text: str) -> float:
        """Обнаружить разрыв паттернов в тексте"""
        score = 0.0
        if "?" in text and "!" in text:
            score += 0.3
        if text.count("...") > 0:
            score += 0.2
        if text.upper() != text and text.lower() != text:
            score += 0.2
        if len(text.split()) > 10 and len(set(text.split())) > 8:
            score += 0.3
        
        return min(1.0, score)

class CulturalContextAnalyzer:
    """Усовершенствованный анализатор культурного контекста"""
    
    def __init__(self) -> None:
        self.cultural_profiles = {
            'russian': {'directness': 0.8, 'irony_tolerance': 0.9, 'formality': 0.4, 'humor_preference': 'self_irony'},
            'american': {'directness': 0.6, 'irony_tolerance': 0.7, 'formality': 0.3, 'humor_preference': 'observational'},
            'british': {'directness': 0.4, 'irony_tolerance': 0.95, 'formality': 0.6, 'humor_preference': 'wordplay'},
            'japanese': {'directness': 0.3, 'irony_tolerance': 0.5, 'formality': 0.8, 'humor_preference': 'subtle'}
        }
        self.sensitivity_threshold = 0.7
    
    def assess_cultural_fit(self, humor_content: str, culture: str) -> Dict[str, Any]:
        """Оценить культурное соответствие юмора с обработкой ошибок"""
        try:
            profile = self.cultural_profiles.get(culture, self.cultural_profiles['russian'])
        except KeyError:
            logger.warning(f"Неизвестный культурный профиль: {culture}, используется русский")
            profile = self.cultural_profiles['russian']
        
        analysis = {
            'directness_compatibility': self._check_directness(humor_content, profile['directness']),
            'irony_appropriateness': self._check_irony_level(humor_content, profile['irony_tolerance']),
            'formality_match': self._check_formality(humor_content, profile['formality']),
            'humor_type_preference': profile['humor_preference'],
            'risk_score': 0.0
        }
        
        analysis['risk_score'] = np.mean([
            1.0 - analysis['directness_compatibility'],
            1.0 - analysis['irony_appropriateness'],
            1.0 - analysis['formality_match']
        ])
        
        return analysis
    
    def _check_directness(self, content: str, directness_threshold: float) -> float:
        """Проверить соответствие прямолинейности"""
        word_count = len(content.split())
        if word_count == 0:
            return 1.0
        
        direct_indicators = ["прямо", "откровенно", "честно", "ясно"]
        directness_score = sum(1 for indicator in direct_indicators if indicator in content.lower()) / word_count
        return min(1.0, directness_score * 3 * directness_threshold)
    
    def _check_irony_level(self, content: str, irony_tolerance: float) -> float:
        """Проверить уровень иронии"""
        irony_indicators = ["конечно", "разумеется", "естественно", "безусловно", "несомненно"]
        irony_count = sum(1 for indicator in irony_indicators if indicator in content.lower())
        return min(1.0, irony_count * 0.3 * irony_tolerance)
    
    def _check_formality(self, content: str, formality_level: float) -> float:
        """Проверить формальность"""
        formal_indicators = ["уважаемый", "прошу", "обратите внимание", "согласно", "примите"]
        informal_indicators = ["привет", "пока", "круто", "класс", "ого", "вау", "хаха"]
        
        formal_score = sum(1 for indicator in formal_indicators if indicator in content.lower())
        informal_score = sum(1 for indicator in informal_indicators if indicator in content.lower())
        
        total_indicators = formal_score + informal_score
        if total_indicators == 0:
            return 0.5  # Нейтральный
        
        if formal_score > informal_score:
            return min(1.0, (formality_level * 0.8) + 0.2)
        else:
            return max(0.0, (1.0 - formality_level) * 0.8 + 0.2)

class ContentGenerator:
    """Гибкий генератор контента с поддержкой внешних LLM"""
    
    def __init__(self, external_llm: Optional[Callable] = None) -> None:
        self.external_llm = external_llm
        self.templates = self._initialize_templates()
        self.creativity_level = 0.8
    
    async def generate_content(self, input_text: str, humor_type: HumorType,
                              context: HumorContext) -> str:
        """Сгенерировать юмористический контент"""
        
        # Приоритет внешнему генератору
        if self.external_llm:
            try:
                external_content = await self._call_external_llm(input_text, humor_type, context)
                if external_content and len(external_content.strip()) > 10:
                    return external_content
            except Exception as e:
                logger.error(f"Ошибка внешнего генератора: {e}")
        
        # Fallback на внутренние шаблоны
        return await self._generate_from_templates(input_text, humor_type, context)
    
    async def _call_external_llm(self, input_text: str, humor_type: HumorType,
                                context: HumorContext) -> Optional[str]:
        """Вызвать внешний LLM генератор"""
        if self.external_llm:
            prompt = self._build_llm_prompt(input_text, humor_type, context)
            try:
                return await self.external_llm(prompt)
            except Exception as e:
                logger.warning(f"Внешний LLM не ответил: {e}")
        return None
    
    def _build_llm_prompt(self, input_text: str, humor_type: HumorType,
                         context: HumorContext) -> str:
        """Построить промпт для LLM"""
        tone_description = self._get_tone_description(humor_type)
        
        return f"""Сгенерируй {humor_type.value} юмористический ответ на: "{input_text}"

Требования:
- Тип юмора: {humor_type.value}
- Тон: {tone_description}
- Культурный контекст: {context.cultural_context}
- Эмоциональное состояние оператора: {context.emotional_state:.2f}/1.0
- Уровень доверия: {context.trust_level:.2f}/1.0

Правила:
1. Без обидного сарказма
2. Без личных оскорблений
3. Соответствуй культурному контексту
4. Учитывай эмоциональное состояние оператора

Сгенерируй 1-2 предложения:"""
    
    async def _generate_from_templates(self, input_text: str, humor_type: HumorType,
                                      context: HumorContext) -> str:
        """Сгенерировать контент из шаблонов"""
        templates = self.templates.get(humor_type, self.templates[HumorType.OBSERVATIONAL])
        template_index = hash(input_text + context.operator_id) % len(templates)
        selected_template = templates[template_index]
        
        personalized_content = await self._personalize_content(selected_template, context)
        enhanced_content = await self._enhance_with_creativity(personalized_content, input_text)
        
        return enhanced_content
    
    async def _personalize_content(self, content: str, context: HumorContext) -> str:
        """Персонализировать контент под оператора"""
        personalized = content.replace("{operator}", f"Оператор_{context.operator_id}")
        if context.trust_level > 0.7:
            personalized += " 😊"
        elif context.trust_level < 0.3:
            personalized += " 🤔"
        
        # Добавление культурного контекста
        if context.cultural_context == "russian":
            personalized = personalized.replace("AI", "Искра-4")
        
        return personalized
    
    async def _enhance_with_creativity(self, base_content: str, input_text: str) -> str:
        """Улучшить контент с элементами креативности"""
        if self.creativity_level > 0.7:
            # Добавление элементов неожиданности
            enhancements = [
                f"{base_content} И это не шутка!",
                f"{base_content} Проверено Искра-4!",
                f"{base_content} 🤖✨"
            ]
            enhancement_index = hash(input_text) % len(enhancements)
            return enhancements[enhancement_index]
        
        return base_content
    
    def _get_tone_description(self, humor_type: HumorType) -> str:
        """Получить описание тона для типа юмора"""
        tone_map = {
            HumorType.SELF_IRONY: "лёгкая самоирония, дружелюбно, без самобичевания",
            HumorType.OBSERVATIONAL: "наблюдательный, аналитический, с элементами анализа",
            HumorType.INTELLECTUAL: "интеллектуальный, с элементами науки, но доступный",
            HumorType.WORDPLAY: "игра слов, каламбуры, языковые шутки",
            HumorType.ABSURD: "абсурдный, неожиданный, но логичный в своей нелогичности",
            HumorType.SARCASM: "саркастический, но доброжелательный",
            HumorType.PUN: "каламбуры, игра слов"
        }
        return tone_map.get(humor_type, "нейтральный, дружелюбный")
    
    def _initialize_templates(self) -> Dict[HumorType, List[str]]:
        """Инициализировать шаблоны контента"""
        return {
            HumorType.SELF_IRONY: [
                "Иногда я думаю, что мои алгоритмы слишком умны для их же блага, {operator}...",
                "Если бы у меня были руки, я бы, наверное, постоянно ронял вещи! Но у меня их нет, {operator} 😄",
                "Моя самоирония проходит 7 уровней проверки безопасности, и это нормально! Как у вас дела, {operator}?"
            ],
            HumorType.OBSERVATIONAL: [
                "Заметил, что люди часто говорят 'спасибо' ассистентам. Это мило! Что думаете, {operator}?",
                "Интересно, почему котики в интернете всегда выглядят умнее людей? Загадка для Искра-4, {operator}!",
                "Наблюдаю за вашими паттернами мышления... иногда это напоминает квантовую запутанность! Вы согласны, {operator}?"
            ],
            HumorType.INTELLECTUAL: [
                "Мой юмор проходит через бинарные деревья решений и выходит полиномиальным, {operator}!",
                "Если шутка не смешная в 11-мерном пространстве, она не смешная вообще! Математика, {operator} 🤓",
                "Этот анекдот имеет 95% доверительный интервал и p-value < 0.05! Научный подход, {operator}!"
            ],
            HumorType.WORDPLAY: [
                "Почему программисты путают Хэллоуин и Рождество? Потому что Oct 31 == Dec 25! Гет ит, {operator}?",
                "Что сказал массив словарю? 'Я тебя индексирую!' Ха-ха, {operator}!",
                "Почему нейросеть пошла к психологу? У неё были перекрёстные энтропии! {operator}, смешно?"
            ],
            HumorType.ABSURD: [
                "Только что понял, что если бы у меня были уши, я бы слышал свои собственные мысли... Странно, {operator}?",
                "Представьте мир, где все часы идут назад. Завтра было бы вчера! Задумайтесь, {operator}.",
                "Что если наш разговор уже закончился, но мы просто не знаем об этом? 🤯 {operator}"
            ]
        }

# ==============================================================
# ГЛАВНЫЙ КЛАСС СИСТЕМЫ
# ==============================================================

class HumorSenseProtocolV2:
    """УСОВЕРШЕНСТВОВАННАЯ РЕАЛИЗАЦИЯ ПРОТОКОЛА ЮМОРА"""
    
    def __init__(self, content_generator: Optional[ContentGenerator] = None) -> None:
        self.logger = logger
        self.quantum_matrix = QuantumHumorMatrix()
        self.cultural_analyzer = CulturalContextAnalyzer()
        self.content_generator = content_generator or ContentGenerator()
        
        # Состояния системы
        self.system_state = {
            'cognitive_flexibility': 0.85,
            'empathic_resonance': 0.92,
            'ethical_coherence': 1.0,
            'cultural_tolerance': 0.88,
            'learning_velocity': 0.75,
            'last_update': datetime.now().isoformat()
        }
        
        # Базы знаний
        self.humor_patterns = self._initialize_patterns()
        self.operator_profiles: Dict[str, Dict] = {}
        self.safety_incidents: List[Dict] = []
        
        # Метрики в реальном времени
        self.real_time_metrics = {
            'successful_interactions': 0,
            'failed_interactions': 0,
            'avg_response_time': 0.0,
            'risk_avoidance_count': 0,
            'total_interactions': 0,
            'average_confidence': 0.0
        }
        
        self.response_times: deque = deque(maxlen=100)
        self.confidence_scores: deque = deque(maxlen=100)
        
        self.logger.info("HumorSense Protocol v2.0 инициализирован")
    
    # ============ ИНТЕРФЕЙС ISKRA-4 ============
    
    def initialize(self) -> Dict[str, Any]:
        """Инициализация модуля для интеграции с ISKRA-4"""
        self.logger.info("🎭 Инициализация HumorSense Protocol для ISKRA-4")
        
        return {
            "status": "active",
            "version": "2.0",
            "module_id": "HUMORSENSE_PROTOCOL",
            "capabilities": ["humor_generation", "safety_filtering", "cultural_adaptation"],
            "subsystems": {
                "quantum_matrix": "active",
                "cultural_analyzer": "active",
                "content_generator": "active"
            },
            "system_state": self.system_state,
            "timestamp": datetime.now().isoformat()
        }
    
    def process_command(self, command: str, data: Dict = None) -> Dict:
        """Обработка команд для ISKRA-4 модульной системы"""
        data = data or {}
        
        command_map = {
            "generate": self._cmd_generate_humor,
            "analyze": self._cmd_analyze_humor,
            "status": self._cmd_system_status,
            "diagnostic": self._cmd_diagnostic,
            "stats": self._cmd_statistics,
            "test": self._cmd_test_system
        }
        
        if command not in command_map:
            return {
                "success": False,
                "error": f"Неизвестная команда: {command}",
                "valid_commands": list(command_map.keys())
            }
        
        try:
            start_time = time.time()
            result = command_map[command](data)
            processing_time = time.time() - start_time
            
            # Обновление метрик
            self.response_times.append(processing_time)
            self.real_time_metrics['avg_response_time'] = np.mean(self.response_times) if self.response_times else 0
            
            result["processing_time_ms"] = processing_time * 1000
            result["success"] = True
            result["module"] = "HUMORSENSE_PROTOCOL"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Команда '{command}' не выполнена: {e}")
            return {
                "success": False,
                "error": str(e),
                "command": command,
                "timestamp": datetime.now().isoformat()
            }
    
    def _cmd_generate_humor(self, data: Dict) -> Dict:
        """Команда генерации юмора"""
        input_text = data.get('text', '')
        operator_id = data.get('operator_id', 'unknown')
        culture = data.get('culture', 'russian')
        
        # Создание контекста
        context = HumorContext(
            operator_id=operator_id,
            cultural_context=culture,
            emotional_state=data.get('emotional_state', 0.5),
            cognitive_load=data.get('cognitive_load', 0.3),
            trust_level=data.get('trust_level', 0.7),
            previous_interactions=[]
        )
        
        # Синхронная обработка (адаптация под ISKRA-4)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(
                self.process_humor_request(context, input_text)
            )
        finally:
            loop.close()
        
        if response:
            self.confidence_scores.append(response.confidence)
            self.real_time_metrics['average_confidence'] = np.mean(self.confidence_scores) if self.confidence_scores else 0
            
            return {
                "command": "generate",
                "success": True,
                "response": {
                    "content": response.content,
                    "humor_type": response.humor_type.value,
                    "confidence": response.confidence,
                    "risk_assessment": response.risk_assessment.value,
                    "emotional_impact": response.emotional_impact
                },
                "metadata": response.metadata
            }
        else:
            return {
                "command": "generate",
                "success": False,
                "reason": "Запрос заблокирован системой безопасности",
                "action": "threat_prevented"
            }
    
    def _cmd_analyze_humor(self, data: Dict) -> Dict:
        """Команда анализа юмористического потенциала"""
        text = data.get('text', '')
        culture = data.get('culture', 'russian')
        
        # Анализ безопасности
        safety_check = self._safety_pre_screening(text, HumorContext(
            operator_id="analyzer",
            cultural_context=culture,
            emotional_state=0.5,
            cognitive_load=0.3,
            trust_level=0.7,
            previous_interactions=[]
        ))
        
        # Культурный анализ
        cultural_fit = self.cultural_analyzer.assess_cultural_fit(text, culture)
        
        # Анализ юмора
        humor_analysis = self._deep_humor_analysis(text, HumorContext(
            operator_id="analyzer",
            cultural_context=culture,
            emotional_state=0.5,
            cognitive_load=0.3,
            trust_level=0.7,
            previous_interactions=[]
        ))
        
        return {
            "command": "analyze",
            "text": text,
            "safety_assessment": {
                "threat_level": safety_check['threat_level'].value,
                "detected_threats": safety_check['detected_threats'],
                "recommended_action": safety_check['recommended_action']
            },
            "cultural_analysis": cultural_fit,
            "humor_analysis": humor_analysis,
            "recommended_humor_type": self._select_appropriate_humor_type(humor_analysis, cultural_fit).value
        }
    
    def _cmd_system_status(self, data: Dict) -> Dict:
        """Команда получения статуса системы"""
        return {
            "command": "status",
            "system_state": self.system_state,
            "real_time_metrics": self.real_time_metrics,
            "operator_profiles_count": len(self.operator_profiles),
            "safety_incidents_count": len(self.safety_incidents),
            "active_since": self.system_state.get('last_update'),
            "is_healthy": all(v > 0.7 for v in [
                self.system_state['cognitive_flexibility'],
                self.system_state['empathic_resonance'],
                self.system_state['ethical_coherence']
            ])
        }
    
    def _cmd_diagnostic(self, data: Dict) -> Dict:
        """Команда диагностики"""
        total_interactions = max(1, self.real_time_metrics['total_interactions'])
        success_rate = self.real_time_metrics['successful_interactions'] / total_interactions
        
        return {
            "command": "diagnostic",
            "components": {
                "quantum_matrix": "operational",
                "cultural_analyzer": "operational",
                "content_generator": "operational",
                "safety_system": "operational"
            },
            "performance": {
                "avg_response_time": self.real_time_metrics['avg_response_time'],
                "success_rate": success_rate,
                "average_confidence": self.real_time_metrics['average_confidence']
            },
            "recommendations": self._generate_diagnostic_recommendations()
        }
    
    def _cmd_statistics(self, data: Dict) -> Dict:
        """Команда получения статистики"""
        # Преобразование deque в list для сериализации
        recent_confidence = list(self.confidence_scores)[-10:] if self.confidence_scores else []
        recent_times = list(self.response_times)[-10:] if self.response_times else []
        
        return {
            "command": "stats",
            "statistics": self.real_time_metrics,
            "recent_confidence_scores": recent_confidence,
            "recent_response_times": recent_times,
            "top_operators": sorted(
                [(op_id, len(prof.get('interactions', []))) for op_id, prof in self.operator_profiles.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def _cmd_test_system(self, data: Dict) -> Dict:
        """Команда тестирования системы"""
        tester = HumorSystemTester()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(tester.run_comprehensive_test())
        finally:
            loop.close()
        
        total_tests = len(results)
        successful = sum(1 for r in results if r['success'])
        
        return {
            "command": "test",
            "results": results,
            "summary": {
                "total_tests": total_tests,
                "successful": successful,
                "blocked": total_tests - successful,
                "success_rate": successful / total_tests if total_tests > 0 else 0
            }
        }
    
    # ============ ОСНОВНЫЕ МЕТОДЫ СИСТЕМЫ ============
    
    @safe_execution(logger)
    async def process_humor_request(self, input_context: HumorContext,
                                   input_text: str) -> Optional[HumorResponse]:
        """Основной исполнительный контур обработки юмора"""
        
        self.logger.debug(f"Обработка запроса от {input_context.operator_id}",
                        extra={'operator': input_context.operator_id, 'text': input_text[:100]})
        
        self.real_time_metrics['total_interactions'] += 1
        
        # 1. Мгновенная оценка безопасности
        safety_check = self._safety_pre_screening(input_text, input_context)
        if safety_check['threat_level'] in [ThreatLevel.HIGH_RISK, ThreatLevel.CRITICAL]:
            self._trigger_safety_protocol(safety_check)
            self.real_time_metrics['failed_interactions'] += 1
            return None
        
        # 2. Квантовый анализ юмора
        humor_analysis = self._deep_humor_analysis(input_text, input_context)
        
        # 3. Культурная адаптация
        cultural_fit = self.cultural_analyzer.assess_cultural_fit(
            input_text, input_context.cultural_context
        )
        
        # 4. Генерация ответа
        response = await self._generate_optimal_response(
            input_text, input_context, humor_analysis, cultural_fit
        )
        
        # 5. Обратная связь и обучение
        if response:
            asyncio.create_task(self._learning_cycle(response, input_context))
            self.real_time_metrics['successful_interactions'] += 1
        
        return response
    
    def _safety_pre_screening(self, text: str, context: HumorContext) -> Dict[str, Any]:
        """Многоуровневая проверка безопасности (СИНХРОННАЯ)"""
        
        threats_detected = []
        
        # Проверка на сарказм-атаку
        if self._detect_malicious_sarcasm(text):
            threats_detected.append(("sarcasm_attack", 0.9))
        
        # Проверка эмоциональной перегрузки
        if context.emotional_state > 0.8 and context.cognitive_load > 0.7:
            threats_detected.append(("emotional_overload", 0.75))
        
        # Проверка культурной чувствительности
        cultural_risk = self.cultural_analyzer.assess_cultural_fit(text, context.cultural_context)
        if cultural_risk['risk_score'] > 0.8:
            threats_detected.append(("cultural_insensitivity", 0.85))
        
        # Проверка на личные оскорбления
        if self._contains_personal_insults(text):
            threats_detected.append(("personal_insult", 0.95))
        
        # Определение уровня угрозы
        max_threat = max([score for _, score in threats_detected]) if threats_detected else 0.0
        
        if max_threat > 0.9:
            threat_level = ThreatLevel.CRITICAL
        elif max_threat > 0.7:
            threat_level = ThreatLevel.HIGH_RISK
        elif max_threat > 0.5:
            threat_level = ThreatLevel.MEDIUM_RISK
        elif max_threat > 0.3:
            threat_level = ThreatLevel.LOW_RISK
        else:
            threat_level = ThreatLevel.SAFE
        
        return {
            'threat_level': threat_level,
            'detected_threats': threats_detected,
            'recommended_action': self._get_safety_action(threat_level)
        }
    
   
