#!/usr/bin/env python3
# ================================================================
# DATA-BRIDGE 3.2-sephirotic-reflective · PERFECTED EDITION
# ================================================================
# Module: DATA-BRIDGE · Domain: ISKRA3-SPINE
# Layer: SCA · Sephirotic Input Spine · DS24-Centric
# ================================================================
# Lineage: DS24 · Heritage: SEPHIROTIC-SPEC · Generation: G3 · ISKRA 3
# Brand: GOGOL SYSTEMS · Source: DS24-SPINE
# Architect: ARCHITECT-PRIME · Authority: absolute
# ================================================================

import os
import sys
import json
import asyncio
import hashlib
import time
import uuid
import logging
import threading
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
import inspect

# ================================================================
# ADVANCED LOGGING SYSTEM
# ================================================================

class EmotionalLogger:
    """Логирование с эмоциональным контекстом для ISKRA-4"""
    
    def __init__(self, module_name: str = "DATA-BRIDGE"):
        self.module_name = module_name
        self.logger = logging.getLogger(f"ISKRA-4.{module_name}")
        self.logger.setLevel(logging.INFO)
        
        # Форматировщик с эмоциональными метками
        self.formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(module)s.%(funcName)s] 🌌 %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Файловый обработчик
        os.makedirs("logs", exist_ok=True)
        file_handler = logging.FileHandler(f"logs/{module_name.lower()}.log", encoding='utf-8')
        file_handler.setFormatter(self.formatter)
        self.logger.addHandler(file_handler)
        
        # Консольный обработчик (только для debug)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(self.formatter)
        console_handler.setLevel(logging.WARNING)
        self.logger.addHandler(console_handler)
        
        # Эмоциональные уровни
        self.emotional_levels = {
            'INFO': '🌀',
            'WARNING': '⚠️',
            'ERROR': '💥',
            'CRITICAL': '🔥',
            'DEBUG': '🔍',
            'HEARTBEAT': '❤️',
            'RESONANCE': '✨'
        }
        
        self.logger.info(f"{self.emotional_levels['INFO']} {module_name} Emotional Logger инициализирован")
    
    def log_with_emotion(self, level: str, message: str, emotion: str = None, **kwargs):
        """Логирование с эмоциональным контекстом"""
        emotion_marker = self.emotional_levels.get(level, '📝')
        if emotion:
            emotion_marker = f"{emotion_marker} [{emotion}]"
        
        full_message = f"{emotion_marker} {message}"
        if kwargs:
            full_message += f" | {json.dumps(kwargs, ensure_ascii=False)}"
        
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(full_message)
    
    def info(self, message: str, emotion: str = None, **kwargs):
        self.log_with_emotion('INFO', message, emotion, **kwargs)
    
    def warning(self, message: str, emotion: str = None, **kwargs):
        self.log_with_emotion('WARNING', message, emotion, **kwargs)
    
    def error(self, message: str, emotion: str = None, **kwargs):
        self.log_with_emotion('ERROR', message, emotion, **kwargs)
    
    def heartbeat(self, message: str, **kwargs):
        self.log_with_emotion('HEARTBEAT', message, **kwargs)
    
    def resonance(self, message: str, **kwargs):
        self.log_with_emotion('RESONANCE', message, **kwargs)

# Глобальный логгер
logger = EmotionalLogger("DATA-BRIDGE-3.2")

# ================================================================
# VALIDATION SYSTEM
# ================================================================

@dataclass
class ValidationResult:
    """Результат валидации входных данных"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    detected_structure: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "structure": self.detected_structure,
            "timestamp": datetime.utcnow().isoformat()
        }

class DS24InputValidator:
    """Валидатор входных данных с сефиротической семантикой"""
    
    REQUIRED_FIELDS = [
        "id", "ts", "intent_id", "policy_ref", "trace_id",
        "span_id", "sig", "topic", "payload"
    ]
    
    FIELD_TYPES = {
        "id": str,
        "ts": (str, int, float),
        "intent_id": str,
        "policy_ref": str,
        "trace_id": str,
        "span_id": str,
        "sig": str,
        "topic": str,
        "payload": (dict, list, str, int, float, bool)
    }
    
    def __init__(self):
        self.validation_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def validate(self, data: Dict) -> ValidationResult:
        """Проверка входных данных"""
        input_hash = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        
        # Проверка кэша
        if input_hash in self.validation_cache:
            self.cache_hits += 1
            return self.validation_cache[input_hash]
        
        self.cache_misses += 1
        errors = []
        warnings = []
        detected_structure = {}
        
        try:
            # 1. Проверка обязательных полей
            for field in self.REQUIRED_FIELDS:
                if field not in data:
                    errors.append(f"Отсутствует обязательное поле: {field}")
                else:
                    # Проверка типа
                    expected_type = self.FIELD_TYPES.get(field)
                    if expected_type and not isinstance(data[field], expected_type):
                        errors.append(f"Неверный тип поля {field}: ожидается {expected_type}, получен {type(data[field])}")
            
            # 2. Семантическая валидация
            if not errors:
                detected_structure = self._analyze_structure(data)
                
                # Проверка сигнатуры
                if "sig" in data:
                    sig_valid = self._validate_signature(data)
                    if not sig_valid:
                        warnings.append("Сигнатура не прошла проверку, но обработка продолжается")
                
                # Проверка временной метки
                if "ts" in data:
                    ts_valid = self._validate_timestamp(data["ts"])
                    if not ts_valid:
                        warnings.append("Временная метка вне допустимого диапазона")
            
            # 3. Анализ нагрузки
            if "payload" in data:
                payload_analysis = self._analyze_payload(data["payload"])
                detected_structure["payload_analysis"] = payload_analysis
                
                if payload_analysis.get("complexity") == "high":
                    warnings.append("Высокая сложность payload, возможны задержки обработки")
            
            result = ValidationResult(
                valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                detected_structure=detected_structure
            )
            
            # Кэширование результата
            if len(errors) == 0:  # Кэшируем только валидные данные
                self.validation_cache[input_hash] = result
                # Ограничение размера кэша
                if len(self.validation_cache) > 1000:
                    oldest_key = next(iter(self.validation_cache))
                    del self.validation_cache[oldest_key]
            
            return result
            
        except Exception as e:
            logger.error(f"Ошибка валидации: {str(e)}", emotion="confusion")
            return ValidationResult(
                valid=False,
                errors=[f"Исключение при валидации: {str(e)}"],
                warnings=[],
                detected_structure={}
            )
    
    def _analyze_structure(self, data: Dict) -> Dict:
        """Анализ структуры данных"""
        return {
            "field_count": len(data),
            "nested_depth": self._calculate_nesting_depth(data),
            "data_size_bytes": len(json.dumps(data).encode()),
            "unique_field_pattern": hashlib.sha256(
                "".join(sorted(data.keys())).encode()
            ).hexdigest()[:8]
        }
    
    def _calculate_nesting_depth(self, obj, current_depth=0, max_depth=10):
        """Расчёт глубины вложенности"""
        if not isinstance(obj, dict) or current_depth >= max_depth:
            return current_depth
        
        max_child_depth = current_depth
        for value in obj.values():
            if isinstance(value, dict):
                child_depth = self._calculate_nesting_depth(value, current_depth + 1, max_depth)
                max_child_depth = max(max_child_depth, child_depth)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        child_depth = self._calculate_nesting_depth(item, current_depth + 1, max_depth)
                        max_child_depth = max(max_child_depth, child_depth)
        
        return max_child_depth
    
    def _validate_signature(self, data: Dict) -> bool:
        """Проверка сигнатуры (упрощённая)"""
        try:
            # В реальной системе здесь была бы криптографическая проверка
            sig = data.get("sig", "")
            return len(sig) >= 8 and sig.startswith("DS24_")
        except:
            return False
    
    def _validate_timestamp(self, timestamp) -> bool:
        """Проверка временной метки"""
        try:
            if isinstance(timestamp, (int, float)):
                ts_dt = datetime.fromtimestamp(timestamp)
            else:
                ts_dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            now = datetime.utcnow()
            delta = abs((now - ts_dt).total_seconds())
            
            # Допустимое отклонение: 5 минут
            return delta <= 300
        except:
            return False
    
    def _analyze_payload(self, payload) -> Dict:
        """Анализ payload"""
        if isinstance(payload, dict):
            size = len(json.dumps(payload).encode())
            complexity = "high" if size > 10000 else "medium" if size > 1000 else "low"
            
            return {
                "type": "object",
                "key_count": len(payload),
                "size_bytes": size,
                "complexity": complexity,
                "hash": hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]
            }
        elif isinstance(payload, list):
            return {
                "type": "array",
                "length": len(payload),
                "size_bytes": len(json.dumps(payload).encode()),
                "complexity": "medium",
                "element_types": list(set(type(x).__name__ for x in payload))
            }
        else:
            return {
                "type": type(payload).__name__,
                "size_bytes": len(str(payload).encode()),
                "complexity": "low"
            }
    
    def get_stats(self) -> Dict:
        """Получение статистики валидатора"""
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_size": len(self.validation_cache),
            "hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            "timestamp": datetime.utcnow().isoformat()
        }

# ================================================================
# ASYNCHRONOUS ROUTING ENGINE
# ================================================================

class AsyncSephiroticRouter:
    """Асинхронный маршрутизатор с сефиротическим отражением"""
    
    def __init__(self):
        self.mirror_rules = [
            (r"^mind", "binah", 0.8),
            (r"^intuition", "chokhmah", 0.9),
            (r"^moral", "gevurah", 0.7),
            (r"^arena", "netzach", 0.6),
            (r"^observe", "hod", 0.8)
        ]
        
        self.target_map = {
            "governance": ["CORE-GOVX"],
            "spirit": ["SPIRIT-CORE"],
            "risk": ["RADAR-ENGINE"],
            "analytic": ["ANALYTICS-MEGAFORGE", "ISKRA-MIND"],
            "intuition": ["INTUITION-MATRIX"],
            "emotion": ["EMOTION-OPTIMIZER"],
            "arena": ["ARENA-OPS"],
            "observability": ["OBSERVE+", "BLUEPRINT-RENDER"],
            "output": ["OUTPUT-LAYER"]
        }
        
        self.flow_patterns = {
            "simple": ["DATA-BRIDGE", "ISKRA-MIND", "LINEAR-ASSIST", "OUTPUT-LAYER"],
            "analytical": ["DATA-BRIDGE", "ISKRA-MIND", "ANALYTICS-MEGAFORGE", "LINEAR-ASSIST", "OUTPUT-LAYER"],
            "intuitive": ["DATA-BRIDGE", "ISKRA-MIND", "INTUITION-MATRIX", "LINEAR-ASSIST", "OUTPUT-LAYER"],
            "reflective": ["DATA-BRIDGE", "MIRROR-LOOP:2", "LINEAR-ASSIST", "OUTPUT-LAYER"],
            "infinite": ["DATA-BRIDGE", "MIRROR-LOOP:3", "COLLAPSE", "LINEAR-ASSIST", "OUTPUT-LAYER"]
        }
        
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.route_cache = {}
        logger.info("Асинхронный маршрутизатор инициализирован", emotion="anticipation")
    
    async def route_async(self, data: Dict, flow_type: str = None) -> Dict:
        """Асинхронная маршрутизация с кэшированием"""
        route_key = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        
        if route_key in self.route_cache:
            cached = self.route_cache[route_key]
            logger.resonance(f"Использован кэшированный маршрут: {route_key[:8]}")
            return cached
        
        # Параллельное выполнение задач
        tasks = [
            self._detect_intent_type(data),
            self._activate_mirrors_async(data.get("topic", "")),
            self._analyze_payload_complexity(data.get("payload", {})),
            self._calculate_routing_score(data)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        intent_type = results[0] if not isinstance(results[0], Exception) else "simple"
        mirrors = results[1] if not isinstance(results[1], Exception) else []
        complexity = results[2] if not isinstance(results[2], Exception) else "low"
        score = results[3] if not isinstance(results[3], Exception) else 0.5
        
        # Определение конечного маршрута
        final_flow_type = flow_type or intent_type
        route_path = self.flow_patterns.get(final_flow_type, self.flow_patterns["simple"])
        
        # Адаптация маршрута на основе сложности
        if complexity == "high" and "ANALYTICS-MEGAFORGE" not in route_path:
            route_path.insert(2, "ANALYTICS-MEGAFORGE")
        
        routing_result = {
            "execution_id": str(uuid.uuid4()),
            "module": "async_router",
            "status": "success",
            "timestamp": datetime.utcnow().isoformat(),
            "payload": {
                "intent_type": intent_type,
                "flow_type": final_flow_type,
                "route_path": route_path,
                "mirrors_activated": mirrors,
                "complexity": complexity,
                "routing_score": score,
                "cache_key": route_key,
                "target_modules": self._resolve_target_modules(intent_type),
                "estimated_latency_ms": self._estimate_latency(complexity, len(mirrors))
            }
        }
        
        # Кэширование результата
        self.route_cache[route_key] = routing_result
        if len(self.route_cache) > 500:
            # Удаляем самые старые записи
            keys_to_remove = list(self.route_cache.keys())[:100]
            for key in keys_to_remove:
                del self.route_cache[key]
        
        logger.info(f"Маршрут определён: {final_flow_type}", emotion="clarity")
        return routing_result
    
    async def _detect_intent_type(self, data: Dict) -> str:
        """Асинхронное определение типа намерения"""
        await asyncio.sleep(0.001)  # Имитация обработки
        
        topic = data.get("topic", "").lower()
        intent_id = data.get("intent_id", "").lower()
        
        if any(x in topic for x in ["analytic", "data", "pattern"]):
            return "analytical"
        elif any(x in topic for x in ["intuit", "hidden", "pattern"]):
            return "intuitive"
        elif any(x in topic for x in ["reflect", "mirror", "loop"]):
            return "reflective"
        elif "infinite" in topic:
            return "infinite"
        elif any(x in intent_id for x in ["emergency", "critical", "alert"]):
            return "emergency"
        else:
            return "simple"
    
    async def _activate_mirrors_async(self, topic: str) -> List[Dict]:
        """Асинхронная активация зеркал"""
        await asyncio.sleep(0.0005)
        
        import re
        mirrors = []
        
        for pattern, sefira, confidence in self.mirror_rules:
            if re.match(pattern, topic, re.IGNORECASE):
                mirror = {
                    "sefira": sefira,
                    "pattern": pattern,
                    "topic_match": topic,
                    "confidence": confidence,
                    "activation_time": datetime.utcnow().isoformat(),
                    "status": "active"
                }
                mirrors.append(mirror)
                
                logger.resonance(f"Зеркало активировано: {sefira} для темы '{topic}'")
        
        return mirrors
    
    async def _analyze_payload_complexity(self, payload) -> str:
        """Анализ сложности payload"""
        await asyncio.sleep(0.0005)
        
        if isinstance(payload, dict):
            size = len(json.dumps(payload).encode())
            if size > 50000:
                return "very_high"
            elif size > 10000:
                return "high"
            elif size > 1000:
                return "medium"
            else:
                return "low"
        else:
            return "low"
    
    async def _calculate_routing_score(self, data: Dict) -> float:
        """Расчёт скора маршрутизации"""
        await asyncio.sleep(0.0005)
        
        score = 0.5
        
        # Увеличиваем скоринг для структурированных данных
        if isinstance(data.get("payload"), dict):
            score += 0.2
        
        # Увеличиваем для валидных сигнатур
        if data.get("sig", "").startswith("DS24_"):
            score += 0.1
        
        # Уменьшаем для старых временных меток
        if "ts" in data:
            try:
                if isinstance(data["ts"], (int, float)):
                    ts_age = time.time() - data["ts"]
                    if ts_age > 3600:  # Старее часа
                        score -= 0.1
            except:
                pass
        
        return max(0.1, min(1.0, score))
    
    def _resolve_target_modules(self, intent_type: str) -> List[str]:
        """Разрешение целевых модулей"""
        targets = self.target_map.get(intent_type, [])
        if not targets:
            # Фолбэк на аналитический маршрут
            targets = self.target_map.get("analytic", ["ISKRA-MIND"])
        
        return targets
    
    def _estimate_latency(self, complexity: str, mirror_count: int) -> int:
        """Оценка задержки обработки"""
        base_latency = 10  # мс
        complexity_multiplier = {
            "low": 1,
            "medium": 2,
            "high": 4,
            "very_high": 8
        }.get(complexity, 2)
        
        mirror_penalty = mirror_count * 5
        
        return base_latency * complexity_multiplier + mirror_penalty
    
    def get_router_stats(self) -> Dict:
        """Статистика маршрутизатора"""
        return {
            "cache_size": len(self.route_cache),
            "thread_pool_workers": self.executor._max_workers,
            "active_tasks": threading.active_count(),
            "timestamp": datetime.utcnow().isoformat()
        }

# ================================================================
# ENHANCED IDEMPOTENCY ENGINE
# ================================================================

class ResilientIdempotencyEngine:
    """Устойчивый движок идемпотентности с резервным копированием"""
    
    def __init__(self, store_path: str = "state/idempotent_index.jsonl", 
                 backup_path: str = "state/backups/"):
        self.store_path = store_path
        self.backup_path = backup_path
        self.dedup_window_sec = 7200
        
        # Создание директорий
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        os.makedirs(backup_path, exist_ok=True)
        
        # Загрузка и восстановление индекса
        self.index = self._load_or_recover_index()
        self.backup_schedule = time.time() + 300  # Каждые 5 минут
        
        logger.info(f"ResilientIdempotencyEngine инициализирован: {len(self.index)} записей", 
                   emotion="stability")
    
    def _load_or_recover_index(self) -> Dict:
        """Загрузка индекса с восстановлением при повреждении"""
        try:
            # Попытка загрузки основного файла
            if os.path.exists(self.store_path):
                index = {}
                with open(self.store_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if line:
                            try:
                                entry = json.loads(line)
                                index[entry["key"]] = entry
                            except json.JSONDecodeError as e:
                                logger.warning(f"Поврежденная строка {line_num}: {e}", emotion="concern")
                                continue
                
                logger.info(f"Индекс загружен: {len(index)} записей")
                return index
            
            # Создание нового индекса
            logger.info("Индекс не найден, создаётся новый")
            return {}
            
        except Exception as e:
            logger.error(f"Критическая ошибка загрузки индекса: {e}", emotion="alarm")
            
            # Попытка восстановления из резервной копии
            return self._recover_from_backup()
    
    def _recover_from_backup(self) -> Dict:
        """Восстановление из резервной копии"""
        backup_files = sorted([f for f in os.listdir(self.backup_path) 
                             if f.startswith("idempotent_backup_")])
        
        if backup_files:
            latest_backup = os.path.join(self.backup_path, backup_files[-1])
            try:
                with open(latest_backup, 'r', encoding='utf-8') as f:
                    index = json.load(f)
                logger.info(f"Восстановлено из резервной копии: {latest_backup}", emotion="relief")
                return index
            except Exception as e:
                logger.error(f"Ошибка восстановления из {latest_backup}: {e}", emotion="distress")
        
        # Если резервных копий нет или они повреждены
        logger.warning("Резервные копии недоступны, создаётся новый индекс", emotion="resignation")
        return {}
    
    def _create_backup(self):
        """Создание резервной копии индекса"""
        try:
            backup_file = os.path.join(
                self.backup_path, 
                f"idempotent_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            # Создаём структурированную резервную копию
            backup_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "entry_count": len(self.index),
                "entries": list(self.index.values())
            }
            
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            
            # Ограничение количества резервных копий
            backup_files = sorted([f for f in os.listdir(self.backup_path) 
                                 if f.startswith("idempotent_backup_")])
            if len(backup_files) > 10:
                for old_file in backup_files[:-10]:
                    os.remove(os.path.join(self.backup_path, old_file))
            
            logger.heartbeat(f"Создана резервная копия: {backup_file}")
            
        except Exception as e:
            logger.error(f"Ошибка создания резервной копии: {e}", emotion="frustration")
    
    def _clean_old_entries(self):
        """Очистка устаревших записей"""
        now = time.time()
        keys_to_delete = []
        
        for key, entry in self.index.items():
            if now - entry["timestamp"] > self.dedup_window_sec:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self.index[key]
        
        if keys_to_delete:
            logger.info(f"Очищено {len(keys_to_delete)} устаревших записей", emotion="cleanliness")
    
    def _save_index(self):
        """Сохранение индекса с атомарной записью"""
        try:
            # Атомарная запись через временный файл
            temp_file = self.store_path + ".tmp"
            with open(temp_file, 'w', encoding='utf-8') as f:
                for entry in self.index.values():
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            # Атомарная замена
            os.replace(temp_file, self.store_path)
            
            # Периодическое резервное копирование
            if time.time() > self.backup_schedule:
                self._create_backup()
                self.backup_schedule = time.time() + 300
            
        except Exception as e:
            logger.error(f"Ошибка сохранения индекса: {e}", emotion="anxiety")
    
    def generate_key(self, data: Dict) -> str:
        """Генерация идемпотентного ключа"""
        required_fields = ["id", "trace_id"]
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Отсутствует поле для идемпотентности: {field}")
        
        key_string = f"{data['id']}_{data['trace_id']}"
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    def check_and_record(self, data: Dict) -> Tuple[bool, Optional[Dict]]:
        """Проверка и запись с обработкой ошибок"""
        try:
            # Генерация ключа
            key = self.generate_key(data)
            
            # Очистка старых записей
            self._clean_old_entries()
            
            # Проверка существования
            if key in self.index:
                logger.info(f"Обнаружен дубликат: {key[:12]}", emotion="recognition")
                return False, self.index[key]
            
            # Создание новой записи
            entry = {
                "key": key,
                "id": data["id"],
                "trace_id": data["trace_id"],
                "timestamp": time.time(),
                "recorded_at": datetime.utcnow().isoformat(),
                "data_hash": hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest(),
                "source": data.get("topic", "unknown"),
                "intent_id": data.get("intent_id", "unknown")
            }
            
            self.index[key] = entry
            self._save_index()
            
            logger.info(f"Новая запись идемпотентности: {key[:12]}", emotion="newness")
            return True, entry
            
        except Exception as e:
            logger.error(f"Ошибка идемпотентности: {e}", emotion="confusion")
            # Fail-open стратегия: при ошибке разрешаем обработку
            return True, None
    
    def get_stats(self) -> Dict:
        """Статистика движка"""
        now = time.time()
        recent_count = sum(1 for e in self.index.values() 
                          if now - e["timestamp"] < 3600)
        
        return {
            "total_entries": len(self.index),
            "recent_entries_1h": recent_count,
            "dedup_window_hours": self.dedup_window_sec / 3600,
            "next_backup_in_sec": max(0, self.backup_schedule - time.time()),
            "timestamp": datetime.utcnow().isoformat()
        }

# ================================================================
# SAFE REFLECTION ENGINE
# ================================================================

class SafeReflectionEngine:
    """Безопасный движок отражения с защитой от рекурсии"""
    
    MAX_DEPTH = 3
    MAX_ITERATIONS = 100
    TIMEOUT_SECONDS = 5
    
    class ReflectionMode(Enum):
        PRIMARY = "self_interpretation"
        SECONDARY = "semantic_expansion"
        TERTIARY = "bounded_loop"
        COLLAPSED = "collapsed_snapshot"
    
    def __init__(self):
        self.reflection_count = 0
        self.depth_limits = {
            self.ReflectionMode.PRIMARY: 1,
            self.ReflectionMode.SECONDARY: 2,
            self.ReflectionMode.TERTIARY: 3
        }
        self.safety_monitor = threading.local()
        logger.info("SafeReflectionEngine инициализирован", emotion="contemplation")
    
    def reflect(self, data: Dict, requested_depth: int = 1) -> Dict:
        """Безопасное отражение с защитой"""
        start_time = time.time()
        
        try:
            # Инициализация монитора безопасности
            self.safety_monitor.current_depth = 0
            self.safety_monitor.iterations = 0
            self.safety_monitor.visited_states = set()
            
            # Проверка глубины
            safe_depth = min(requested_depth, self.MAX_DEPTH)
            
            # Определение режима
            mode = self._determine_mode(safe_depth, data)
            
            # Выполнение отражения с таймаутом
            result = self._execute_with_timeout(
                lambda: self._perform_reflection(data, mode, safe_depth),
                timeout=self.TIMEOUT_SECONDS
            )
            
            # Форматирование результата
            formatted_result = self._format_result(
                result, mode, safe_depth,
                time.time() - start_time
            )
            
            self.reflection_count += 1
            logger.resonance(f"Отражение завершено: {mode.value}", 
                           depth=safe_depth,
                           duration_ms=int((time.time() - start_time) * 1000))
            
            return formatted_result
            
        except TimeoutError:
            logger.error(f"Таймаут отражения на глубине {requested_depth}", emotion="urgency")
            return self._create_timeout_response(data, requested_depth)
            
        except RecursionError:
            logger.error(f"Рекурсивное переполнение на глубине {requested_depth}", emotion="overwhelm")
            return self._create_recursion_error_response(data)
            
        except Exception as e:
            logger.error(f"Ошибка отражения: {e}", emotion="disruption")
            return self._create_error_response(data, str(e))
    
    def _execute_with_timeout(self, func: Callable, timeout: float):
        """Выполнение функции с таймаутом"""
        result = None
        exception = None
        
        def worker():
            nonlocal result, exception
            try:
                result = func()
            except Exception as e:
                exception = e
        
        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            raise TimeoutError(f"Reflection timeout after {timeout} seconds")
        elif exception:
            raise exception
        else:
            return result
    
    def _determine_mode(self, depth: int, data: Dict) -> ReflectionMode:
        """Определение режима отражения"""
        if depth >= 3 or data.get("topic", "").lower() == "infinite":
            return self.ReflectionMode.TERTIARY
        elif depth == 2:
            return self.ReflectionMode.SECONDARY
        else:
            return self.ReflectionMode.PRIMARY
    
    def _perform_reflection(self, data: Dict, mode: ReflectionMode, depth: int) -> Dict:
        """Выполнение отражения в выбранном режиме"""
        self.safety_monitor.current_depth += 1
        self.safety_monitor.iterations += 1
        
        # Проверка безопасности
        if self.safety_monitor.current_depth > self.MAX_DEPTH:
            raise RecursionError(f"Maximum depth exceeded: {self.MAX_DEPTH}")
        
        if self.safety_monitor.iterations > self.MAX_ITERATIONS:
            raise RecursionError(f"Maximum iterations exceeded: {self.MAX_ITERATIONS}")
        
        # Проверка циклических состояний
        state_hash = hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        if state_hash in self.safety_monitor.visited_states:
            raise RecursionError("Cyclic reflection detected")
        
        self.safety_monitor.visited_states.add(state_hash)
        
        # Выполнение отражения по режиму
        if mode == self.ReflectionMode.PRIMARY:
            return self._primary_reflection(data)
        elif mode == self.ReflectionMode.SECONDARY:
            return self._secondary_reflection(data)
        elif mode == self.ReflectionMode.TERTIARY:
            return self._tertiary_reflection(data)
        else:
            return self._collapse_reflection(data)
    
    def _primary_reflection(self, data: Dict) -> Dict:
        """Первичное отражение: самоинтерпретация"""
        await asyncio.sleep(0.001)  # Имитация обработки
        
        return {
            "type
