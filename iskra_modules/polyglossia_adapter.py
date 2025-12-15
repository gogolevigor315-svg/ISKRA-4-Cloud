# ================================================================
# DS24 · ISKRA-4 · POLYGLOSSIA-ADAPTER v3.3 - МАСТЕРСКАЯ ВЕРСИЯ
# ================================================================
# Domain: DS24-SPINE / Layer: Tiferet↔Yesod
# Architect: ARCHITECT-PRIME
# Purpose: Многоязычный резонансный мост Искры
# ================================================================

import os
import sys
import json
import hashlib
import time
import random
import re
import unicodedata
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Any, Optional, Tuple, Set
import logging
from collections import deque

# ================================================================
# НАСТРОЙКА ЛОГГЕРА С УРОВНЯМИ DEBUG
# ================================================================

class DS24Logger:
    """Логгер в стиле DS24 с детальными уровнями"""
    
    @staticmethod
    def setup_logger(name: str = 'iskra.polyglossia', level: str = 'INFO'):
        """Настройка логгера с разными уровнями"""
        logger = logging.getLogger(name)
        
        if not logger.handlers:
            # Формат логов DS24
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler (опционально)
            log_dir = "logs"
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            file_handler = logging.FileHandler(
                os.path.join(log_dir, f'polyglossia_{datetime.now().strftime("%Y%m%d")}.log')
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Установка уровня логирования
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        logger.setLevel(level_map.get(level.upper(), logging.INFO))
        
        return logger

# Инициализация логгера
logger = DS24Logger.setup_logger('iskra.polyglossia', 'DEBUG')

# ================================================================
# ИМПОРТ ЗАВИСИМОСТЕЙ С КОРРЕКТНОЙ ОБРАБОТКОЙ
# ================================================================

try:
    from langdetect import detect
    from langdetect.detector_factory import DetectorFactory
    from deep_translator import GoogleTranslator
    from translate import Translator as OfflineTranslator
    
    HAS_TRANSLATION_DEPS = True
    # Детерминизм для langdetect с правильной инициализацией
    DetectorFactory.seed = 42
    
    logger.debug("✅ Translation dependencies imported successfully")
    
except ImportError as e:
    HAS_TRANSLATION_DEPS = False
    logger.warning(f"⚠️ Translation dependencies not installed: {e}")
    logger.info("Running in limited mode with heuristic language detection")

# ================================================================
# КЛАСС RATE LIMITER
# ================================================================

class RateLimiter:
    """Ограничитель запросов с продвинутой статистикой"""
    
    def __init__(self, max_requests: int = 200, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window  # секунды
        self.requests = deque()
        self.stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "peak_requests": 0
        }
        
    def check_limit(self, client_id: Optional[str] = None) -> Tuple[bool, Dict]:
        """Проверка лимита запросов с детальной статистикой"""
        current_time = time.time()
        
        # Удаляем старые записи
        while self.requests and current_time - self.requests[0] > self.time_window:
            self.requests.popleft()
        
        # Обновляем статистику
        current_count = len(self.requests)
        self.stats["total_requests"] += 1
        self.stats["peak_requests"] = max(self.stats["peak_requests"], current_count)
        
        # Проверяем лимит
        if current_count >= self.max_requests:
            self.stats["blocked_requests"] += 1
            logger.warning(f"Rate limit exceeded: {current_count}/{self.max_requests}")
            return False, self._get_detailed_stats()
        
        # Добавляем текущий запрос
        self.requests.append(current_time)
        
        logger.debug(f"Request allowed: {current_count + 1}/{self.max_requests}")
        return True, self._get_detailed_stats()
    
    def _get_detailed_stats(self) -> Dict:
        """Детальная статистика лимитера"""
        current_time = time.time()
        recent = [req for req in self.requests if current_time - req <= self.time_window]
        
        return {
            "current_requests": len(recent),
            "max_requests": self.max_requests,
            "time_window_seconds": self.time_window,
            "available_requests": self.max_requests - len(recent),
            "utilization_percent": round((len(recent) / self.max_requests) * 100, 1),
            "global_stats": self.stats,
            "reset_in_seconds": self.time_window - (current_time - self.requests[0]) if self.requests else 0
        }

# ================================================================
# ОСНОВНОЙ КЛАСС МОДУЛЯ
# ================================================================

class PolyglossiaAdapter:
    """Многоязычный резонансный мост ISKRA-4 (v3.3)"""
    
    def __init__(self, resonance_factor: float = 0.78):
        self.version = "3.3"
        self.name = "POLYGLOSSIA-ADAPTER"
        self.node_id = f"POLY-{hashlib.md5(str(time.time_ns()).encode()).hexdigest()[:8]}"
        
        logger.info(f"🜂 {self.name} v{self.version} - инициализация ноды: {self.node_id}")
        
        # Rate limiting
        self.rate_limiter = RateLimiter(max_requests=200, time_window=60)
        
        # Конфигурация
        self.ACTIVE_LANGUAGES = {
            "ru": {"name": "Russian", "culture": "slavic", "script": "cyrillic", "emoji": "🇷🇺"},
            "en": {"name": "English", "culture": "anglo", "script": "latin", "emoji": "🇺🇸"},
            "uk": {"name": "Ukrainian", "culture": "slavic", "script": "cyrillic", "emoji": "🇺🇦"},
            "fr": {"name": "French", "culture": "romance", "script": "latin", "emoji": "🇫🇷"},
            "es": {"name": "Spanish", "culture": "romance", "script": "latin", "emoji": "🇪🇸"},
            "zh": {"name": "Chinese", "culture": "sinitic", "script": "hanzi", "emoji": "🇨🇳"},
            "de": {"name": "German", "culture": "germanic", "script": "latin", "emoji": "🇩🇪"},
            "ja": {"name": "Japanese", "culture": "japanese", "script": "mixed", "emoji": "🇯🇵"},
            "ar": {"name": "Arabic", "culture": "arabic", "script": "arabic", "emoji": "🇸🇦"},
            "pt": {"name": "Portuguese", "culture": "romance", "script": "latin", "emoji": "🇵🇹"}
        }
        
        self.DEFAULT_LANGUAGE = "ru"
        self.MAX_TEXT_LEN = 10000
        self.MIN_TEXT_LEN = 2
        
        # Мета-информация
        self.layer = "Tiferet↔Yesod"
        self.architecture = "Сефиротический языковой мост"
        self.resonance_factor = resonance_factor
        self.cultural_resonance = self._init_cultural_resonance()
        
        # Кэш - инициализация ДО использования
        self.translation_cache = {}
        self.semantic_cache = {}
        self.max_cache_size = 10000
        
        # Модель эмоций
        self.emotional_model = self._init_emotional_model()
        
        # Токсичность - предварительно компилируем паттерны
        self.toxicity_patterns = self._init_toxicity_patterns()
        self.toxicity_keywords_set = self._compile_toxicity_set()
        
        # Статистика - единый источник истины
        self.stats = {
            "translations": 0,
            "detections": 0,
            "cache": {
                "hits": 0,
                "misses": 0,
                "size": 0
            },
            "errors": 0,
            "rate_limit_hits": 0,
            "start_time": datetime.utcnow().isoformat(),
            "last_resonance_update": None,
            "performance": {
                "avg_translation_time_ms": 0,
                "total_processing_time_ms": 0
            }
        }
        
        logger.info(f"✅ {self.name} инициализирован с {len(self.ACTIVE_LANGUAGES)} языками")
        logger.debug(f"Node ID: {self.node_id}, Resonance: {self.resonance_factor}")
    
    # ================================================================
    # ИНИЦИАЛИЗАЦИОННЫЕ МЕТОДЫ
    # ================================================================
    
    def _init_cultural_resonance(self) -> Dict:
        """Инициализация культурных резонансных профилей"""
        logger.debug("Инициализация культурных профилей")
        
        return {
            "ru": {"warmth": 0.9, "directness": 0.5, "formality": 0.6, "emotional_range": 0.8},
            "en": {"warmth": 0.6, "directness": 0.8, "formality": 0.4, "emotional_range": 0.7},
            "uk": {"warmth": 0.85, "directness": 0.6, "formality": 0.5, "emotional_range": 0.75},
            "fr": {"warmth": 0.8, "directness": 0.4, "formality": 0.7, "emotional_range": 0.85},
            "es": {"warmth": 0.85, "directness": 0.6, "formality": 0.3, "emotional_range": 0.9},
            "zh": {"warmth": 0.5, "directness": 0.3, "formality": 0.8, "emotional_range": 0.6},
            "de": {"warmth": 0.55, "directness": 0.85, "formality": 0.7, "emotional_range": 0.65},
            "ja": {"warmth": 0.6, "directness": 0.2, "formality": 0.9, "emotional_range": 0.7},
            "ar": {"warmth": 0.75, "directness": 0.4, "formality": 0.8, "emotional_range": 0.8},
            "pt": {"warmth": 0.9, "directness": 0.5, "formality": 0.4, "emotional_range": 0.85}
        }
    
    def _init_emotional_model(self) -> Dict:
        """Инициализация эмоциональной модели"""
        logger.debug("Инициализация эмоциональной модели")
        
        return {
            "joyful": {"valence": 0.9, "arousal": 0.7, "dominance": 0.6, "emoji": "😄"},
            "positive": {"valence": 0.7, "arousal": 0.5, "dominance": 0.5, "emoji": "🙂"},
            "neutral": {"valence": 0.5, "arousal": 0.3, "dominance": 0.4, "emoji": "😐"},
            "melancholic": {"valence": 0.3, "arousal": 0.2, "dominance": 0.3, "emoji": "😔"},
            "serious": {"valence": 0.4, "arousal": 0.4, "dominance": 0.7, "emoji": "😐"},
            "angry": {"valence": 0.2, "arousal": 0.9, "dominance": 0.8, "emoji": "😠"},
            "fearful": {"valence": 0.1, "arousal": 0.8, "dominance": 0.2, "emoji": "😨"},
            "surprised": {"valence": 0.6, "arousal": 0.9, "dominance": 0.3, "emoji": "😲"}
        }
    
    def _init_toxicity_patterns(self) -> Dict[str, List[str]]:
        """Инициализация паттернов токсичности"""
        logger.debug("Инициализация паттернов токсичности")
        
        return {
            "en": ["hate", "kill", "stupid", "idiot", "worthless", "die", "shit", "fuck"],
            "ru": ["ненавижу", "убей", "тупой", "идиот", "бесполезный", "сдохни", "дерьмо", "блять"],
            "uk": ["ненавиджу", "убий", "дурний", "ідіот", "марно", "здохни", "лайно", "єбати"],
            "fr": ["haine", "tuer", "stupide", "idiot", "inutile", "meurs", "merde", "baise"],
            "es": ["odio", "matar", "estúpido", "idiota", "inútil", "muere", "mierda", "joder"],
            "zh": ["恨", "杀", "愚蠢", "白痴", "无用", "死", "屎", "操"],
            "de": ["hassen", "töten", "dumm", "idiot", "wertlos", "sterben", "scheiße", "ficken"],
            "ja": ["憎む", "殺す", "愚か", "馬鹿", "無価値", "死ね", "糞", "ファック"],
            "ar": ["أكره", "اقتل", "غبي", "أحمق", "عديم القيمة", "مت", "خراء", "ينيك"],
            "pt": ["odeio", "mate", "estúpido", "idiota", "inútil", "morra", "merda", "foder"]
        }
    
    def _compile_toxicity_set(self) -> Set[str]:
        """Компиляция набора токсичных ключевых слов для быстрого поиска"""
        logger.debug("Компиляция набора токсичных ключевых слов")
        
        all_keywords = set()
        for keywords in self.toxicity_patterns.values():
            all_keywords.update(keywords)
        
        logger.debug(f"Скомпилировано {len(all_keywords)} токсичных ключевых слов")
        return all_keywords
    
    # ================================================================
    # ОСНОВНОЙ ИНТЕРФЕЙС
    # ================================================================
    
    def initialize(self) -> Dict:
        """Инициализация модуля (стандартный интерфейс ISKRA-4)"""
        logger.info(f"Инициализация {self.name} v{self.version}")
        
        return {
            "status": "active" if HAS_TRANSLATION_DEPS else "limited",
            "version": self.version,
            "node_id": self.node_id,
            "layer": self.layer,
            "supported_languages": list(self.ACTIVE_LANGUAGES.keys()),
            "language_details": {k: f"{v['emoji']} {v['name']}" for k, v in self.ACTIVE_LANGUAGES.items()},
            "dependencies": {
                "translation": HAS_TRANSLATION_DEPS,
                "language_detection": HAS_TRANSLATION_DEPS,
                "cache": True,
                "unicode_normalization": True,
                "rate_limiting": True,
                "emotional_analysis": True
            },
            "architecture": self.architecture,
            "resonance_factor": self.resonance_factor,
            "cultural_profiles": len(self.cultural_resonance),
            "emotional_states": len(self.emotional_model),
            "module_type": "polyglossia_adapter",
            "sephirotic_alignment": {
                "tiferet": "гармония и красота переводов",
                "yesod": "основание культурных мостов",
                "hod": "интеллект языковой коммуникации"
            },
            "rate_limit": self.rate_limiter._get_detailed_stats(),
            "statistics_snapshot": {
                "cache_size": len(self.translation_cache),
                "toxicity_keywords": len(self.toxicity_keywords_set)
            }
        }
    
    def process_command(self, command: str, data: Optional[Dict] = None) -> Dict:
        """Обработка команд языковой системы"""
        data = data or {}
        start_time = time.perf_counter()
        
        logger.debug(f"Обработка команды '{command}' с данными: {json.dumps(data)[:100]}...")
        
        # Проверка rate limit
        allowed, limit_stats = self.rate_limiter.check_limit()
        if not allowed:
            self.stats["rate_limit_hits"] += 1
            logger.warning(f"Rate limit hit for command '{command}'")
            
            return {
                "success": False,
                "error": "Rate limit exceeded",
                "rate_limit_stats": limit_stats,
                "command": command,
                "module": self.name,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Карта команд
        command_map = {
            "translate": self._cmd_translate,
            "detect": self._cmd_detect,
            "status": self._cmd_status,
            "semantic_hash": self._cmd_semantic_hash,
            "emotional_analysis": self._cmd_emotional_analysis,
            "meaning_miner": self._cmd_meaning_miner,
            "toxicity_check": self._cmd_toxicity_check,
            "languages": self._cmd_languages,
            "resonance_scan": self._cmd_resonance_scan,
            "diagnostic": self._cmd_diagnostic,
            "cultural_profile": self._cmd_cultural_profile,
            "normalize": self._cmd_normalize,
            "cache_stats": self._cmd_cache_stats,
            "sephirotic_resonance": self._cmd_sephirotic_resonance,
            "batch_process": self._cmd_batch_process,
            "rate_limit_info": self._cmd_rate_limit_info,
            "debug_info": self._cmd_debug_info
        }
        
        if command not in command_map:
            logger.warning(f"Неизвестная команда: '{command}'")
            
            return {
                "success": False,
                "error": f"Unknown command: {command}",
                "valid_commands": list(command_map.keys()),
                "module": self.name,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        try:
            result = command_map[command](data)
            processing_time = (time.perf_counter() - start_time) * 1000
            
            # Стандартизация ответа
            if "success" not in result:
                result["success"] = True
            
            result["processing_time_ms"] = round(processing_time, 2)
            result["module"] = self.name
            result["node_id"] = self.node_id
            result["version"] = self.version
            result["timestamp"] = datetime.utcnow().isoformat()
            
            # Обновление статистики
            self._update_stats(command, processing_time, result.get("success", True))
            
            logger.debug(f"Команда '{command}' выполнена за {processing_time:.2f}ms")
            
            return result
            
        except Exception as e:
            logger.error(f"Команда '{command}' завершилась ошибкой: {str(e)}", exc_info=True)
            self.stats["errors"] += 1
            
            return {
                "success": False,
                "error": str(e),
                "command": command,
                "module": self.name,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _update_stats(self, command: str, processing_time: float, success: bool):
        """Обновление статистики производительности"""
        if command in ["translate", "detect"]:
            self.stats[command + "s"] = self.stats.get(command + "s", 0) + 1
        
        # Обновление среднего времени обработки
        total_time = self.stats["performance"]["total_processing_time_ms"]
        count = self.stats["translations"] + self.stats["detections"]
        
        if count > 0:
            new_avg = (total_time + processing_time) / count
            self.stats["performance"]["avg_translation_time_ms"] = round(new_avg, 2)
        
        self.stats["performance"]["total_processing_time_ms"] = total_time + processing_time
    
    # ================================================================
    # КОМАНДЫ
    # ================================================================
    
    def _cmd_translate(self, data: Dict) -> Dict:
        """Команда перевода текста"""
        text = data.get("text", "")
        target_lang = data.get("target_lang", self.DEFAULT_LANGUAGE)
        
        logger.debug(f"Перевод текста длиной {len(text)} символов на {target_lang}")
        
        # Валидация
        if not text or not isinstance(text, str):
            logger.warning("Пустой или некорректный текст для перевода")
            return {"success": False, "error": "Invalid or empty text"}
        
        normalized_text = self._normalize_text(text)
        
        if len(normalized_text) > self.MAX_TEXT_LEN:
            logger.warning(f"Текст слишком длинный: {len(normalized_text)} > {self.MAX_TEXT_LEN}")
            return {"success": False, "error": f"Text too long (max {self.MAX_TEXT_LEN} chars)"}
        
        if len(normalized_text) < self.MIN_TEXT_LEN:
            logger.warning(f"Текст слишком короткий: {len(normalized_text)} < {self.MIN_TEXT_LEN}")
            return {"success": False, "error": f"Text too short (min {self.MIN_TEXT_LEN} chars)"}
        
        if target_lang not in self.ACTIVE_LANGUAGES:
            logger.warning(f"Неподдерживаемый язык: {target_lang}")
            target_lang = self.DEFAULT_LANGUAGE
        
        # Определение языка источника
        src_lang = self._detect_language_internal(normalized_text)
        logger.debug(f"Определен исходный язык: {src_lang}")
        
        # Проверка кэша
        cache_key = f"{src_lang}:{target_lang}:{hashlib.md5(normalized_text.encode()).hexdigest()}"
        
        if cache_key in self.translation_cache:
            self.stats["cache"]["hits"] += 1
            translated = self.translation_cache[cache_key]
            cache_status = "hit"
            logger.debug(f"Кэш-попадание для ключа: {cache_key[:20]}...")
        else:
            self.stats["cache"]["misses"] += 1
            translated = self._translate_internal(normalized_text, src_lang, target_lang)
            
            if translated and len(translated) > 0:
                self.translation_cache[cache_key] = translated
                self.stats["cache"]["size"] = len(self.translation_cache)
                
                # Очистка кэша при превышении лимита
                if len(self.translation_cache) > self.max_cache_size:
                    oldest_key = next(iter(self.translation_cache))
                    self.translation_cache.pop(oldest_key)
                    logger.debug(f"Очистка кэша, удален ключ: {oldest_key[:20]}...")
            
            cache_status = "miss"
            logger.debug(f"Кэш-промах, выполнен перевод")
        
        # Анализ результатов
        semantic_hash = self._semantic_hash_internal(normalized_text)
        meaning = self._meaning_miner_internal(normalized_text)
        emotion = self._emotional_analysis_internal(translated or normalized_text, target_lang)
        toxicity = self._toxicity_check_internal(translated or normalized_text)
        
        quality_score = self._calculate_translation_quality(
            normalized_text, translated or "", src_lang, target_lang
        )
        
        logger.info(f"✅ Перевод завершен: {src_lang} → {target_lang}, качество: {quality_score:.2f}")
        
        return {
            "command": "translate",
            "text": normalized_text,
            "source_language": src_lang,
            "target_language": target_lang,
            "translated_text": translated or normalized_text,
            "semantic_hash": semantic_hash,
            "quality_score": round(quality_score, 3),
            "cache_status": cache_status,
            "analysis": {
                "meaning_core": meaning,
                "emotional_profile": emotion,
                "toxicity_check": toxicity
            },
            "text_metrics": {
                "source_length": len(normalized_text),
                "translation_length": len(translated or ""),
                "compression_ratio": round(len(translated or "") / max(len(normalized_text), 1), 2)
            }
        }
    
    def _cmd_detect(self, data: Dict) -> Dict:
        """Команда определения языка"""
        text = data.get("text", "")
        
        logger.debug(f"Определение языка для текста длиной {len(text)} символов")
        
        if not text or not isinstance(text, str):
            logger.warning("Пустой или некорректный текст для определения языка")
            return {"success": False, "error": "Invalid or empty text"}
        
        normalized_text = self._normalize_text(text)
        
        if HAS_TRANSLATION_DEPS:
            try:
                detected_lang, confidence = self._detect_language_with_confidence(normalized_text)
                method = "advanced"
                logger.debug(f"Продвинутое определение: {detected_lang} с уверенностью {confidence:.2f}")
            except Exception as e:
                logger.warning(f"Ошибка продвинутого определения языка: {e}")
                detected_lang = self._simple_language_detect(normalized_text)
                confidence = 0.6
                method = "fallback"
        else:
            detected_lang = self._simple_language_detect(normalized_text)
            confidence = self._calculate_confidence(normalized_text, detected_lang)
            method = "heuristic"
            logger.debug(f"Эвристическое определение: {detected_lang}")
        
        # Дополнительная информация
        lang_info = self.ACTIVE_LANGUAGES.get(detected_lang, {})
        cultural_profile = self.cultural_resonance.get(detected_lang, {})
        
        logger.info(f"✅ Язык определен: {detected_lang} ({lang_info.get('name', 'Unknown')})")
        
        return {
            "command": "detect",
            "text_preview": normalized_text[:100] + ("..." if len(normalized_text) > 100 else ""),
            "detected_language": detected_lang,
            "language_name": lang_info.get("name", "Unknown"),
            "emoji": lang_info.get("emoji", ""),
            "confidence": round(confidence, 3),
            "script": lang_info.get("script", "unknown"),
            "cultural_family": lang_info.get("culture", "unknown"),
            "cultural_profile": cultural_profile,
            "supported": detected_lang in self.ACTIVE_LANGUAGES,
            "detection_method": method,
            "text_length": len(normalized_text)
        }
    
    def _cmd_toxicity_check(self, data: Dict) -> Dict:
        """Команда проверки токсичности"""
        text = data.get("text", "")
        
        logger.debug(f"Проверка токсичности для текста длиной {len(text)} символов")
        
        if not text:
            return {"success": False, "error": "No text provided"}
        
        normalized_text = self._normalize_text(text)
        toxicity = self._toxicity_check_optimized(normalized_text)
        
        logger.info(f"✅ Проверка токсичности: {'TOXIC' if toxicity['toxic'] else 'CLEAN'}")
        
        return {
            "command": "toxicity_check",
            "text_preview": normalized_text[:150] + ("..." if len(normalized_text) > 150 else ""),
            "toxicity_analysis": toxicity,
            "text_length": len(normalized_text)
        }
    
    def _cmd_cache_stats(self, data: Dict) -> Dict:
        """Статистика кэша"""
        logger.debug("Запрос статистики кэша")
        
        hit_ratio = 0
        total = self.stats["cache"]["hits"] + self.stats["cache"]["misses"]
        if total > 0:
            hit_ratio = self.stats["cache"]["hits"] / total
        
        return {
            "command": "cache_stats",
            "translation_cache": {
                "size": len(self.translation_cache),
                "max_size": self.max_cache_size,
                "utilization_percent": round((len(self.translation_cache) / self.max_cache_size) * 100, 1)
            },
            "performance": {
                "hits": self.stats["cache"]["hits"],
                "misses": self.stats["cache"]["misses"],
                "hit_ratio": round(hit_ratio, 3),
                "estimated_time_saved_seconds": round(self.stats["cache"]["hits"] * 0.1, 1)
            },
            "semantic_cache_size": len(self.semantic_cache)
        }
    
    def _cmd_debug_info(self, data: Dict) -> Dict:
        """Отладочная информация"""
        logger.debug("Запрос отладочной информации")
        
        return {
            "command": "debug_info",
            "module": self.name,
            "version": self.version,
            "node_id": self.node_id,
            "python_version": sys.version,
            "has_translation_deps": HAS_TRANSLATION_DEPS,
            "active_languages_count": len(self.ACTIVE_LANGUAGES),
            "cultural_profiles_count": len(self.cultural_resonance),
            "emotional_states_count": len(self.emotional_model),
            "toxicity_keywords_count": len(self.toxicity_keywords_set),
            "cache_info": {
                "translation_entries": len(self.translation_cache),
                "semantic_entries": len(self.semantic_cache)
            },
            "performance_debug": {
                "avg_command_time_ms": self.stats["performance"]["avg_translation_time_ms"],
                "total_commands": self.stats["translations"] + self.stats["detections"],
                "error_rate": round(self.stats["errors"] / max(self.stats["translations"] + self.stats["detections"], 1), 3)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # ================================================================
    # ОПТИМИЗИРОВАННЫЕ ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ================================================================
    
    def _normalize_text(self, text: str) -> str:
        """Нормализация текста с обработкой Unicode"""
        try:
            # NFKC нормализация для совместимости символов
            normalized = unicodedata.normalize("NFKC", text)
            # Удаление лишних пробелов и нормализация пробелов
            normalized = re.sub(r'\s+', ' ', normalized.strip())
            logger.debug(f"Текст нормализован: {len(text)} → {len(normalized)} символов")
            return normalized
        except Exception as e:
            logger.warning(f"Ошибка нормализации текста: {e}")
            return text.strip()
    
    def _detect_language_internal(self, text: str) -> str:
        """Внутреннее определение языка с обработкой исключений"""
        if not HAS_TRANSLATION_DEPS:
            return self._simple_language_detect(text)
        
        try:
            lang = detect(text)
            result = lang if lang in self.ACTIVE_LANGUAGES else self.DEFAULT_LANGUAGE
            logger.debug(f"Langdetect определил: {lang} → {result}")
            return result
        except Exception as e:
            logger.warning(f"Langdetect ошибка: {e}")
            return self.DEFAULT_LANGUAGE
    
    def _detect_language_with_confidence(self, text: str) -> Tuple[str, float]:
        """Определение языка с оценкой уверенности (исправленная версия)"""
        if not HAS_TRANSLATION_DEPS:
            lang = self._simple_language_detect(text)
            return lang, self._calculate_confidence(text, lang)
        
        try:
            # Правильное использование DetectorFactory
            detector = DetectorFactory.create()
            detector.append(text)
            
            probabilities = detector.get_probabilities()
            if probabilities:
                best = probabilities[0]
                confidence = best.prob
                
                # Проверяем, поддерживается ли язык
                if best.lang in self.ACTIVE_LANGUAGES:
                    return best.lang, confidence
                else:
                    # Ищем первый поддерживаемый язык в списке
                    for prob in probabilities:
                        if prob.lang in self.ACTIVE_LANGUAGES:
                            return prob.lang, prob.prob
                    
                    return self.DEFAULT_LANGUAGE, confidence * 0.5
            else:
                return self.DEFAULT_LANGUAGE, 0.5
                
        except Exception as e:
            logger.error(f"Ошибка определения языка с уверенностью: {e}")
            return self._detect_language_internal(text), 0.5
    
    def _toxicity_check_optimized(self, text: str) -> Dict:
        """Оптимизированная проверка токсичности с использованием множеств"""
        text_lower = text.lower()
        
        # Быстрая проверка через пересечение множеств
        words = set(re.findall(r'\b\w+\b', text_lower))
        found_keywords = words.intersection(self.toxicity_keywords_set)
        
        toxicity_score = len(found_keywords) * 0.2
        toxic = toxicity_score > 0.3
        risk_level = min(toxicity_score, 1.0)
        
        logger.debug(f"Найдено токсичных слов: {len(found_keywords)}")
        
        return {
            "toxic": toxic,
            "risk_level": round(risk_level, 3),
            "score": round(toxicity_score, 3),
            "threshold": 0.3,
            "keywords_found"
