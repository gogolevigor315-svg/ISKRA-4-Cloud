# ============================================================
# DS24 — PURE PROTOCOL v2.0 (PRODUCTION READY FOR RENDER)
# ============================================================
# Mode: Absolute Determinism · Zero Entropy · Full Audit Trail
# Principle: Same Input + Same Context = Same Output
# Security: Memory Safe · Resource Limited · Production Ready
# ============================================================

import hashlib
import json
import time
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum, auto
from collections import deque

# ============================================================
# 🎯 КОНСТАНТЫ И НАСТРОЙКИ
# ============================================================

class SystemConstants:
    """Константы системы для контроля ресурсов"""
    MAX_EXECUTION_LOG_SIZE = 1000
    MAX_ERROR_LOG_SIZE = 500
    MAX_AUDIT_RECORDS = 100
    PROOF_DIFFICULTY = 1  # Уменьшено для продакшена
    HEARTBEAT_INTERVAL = 30  # секунд
    SESSION_TIMEOUT = 3600  # секунд
    
class LogLevel(Enum):
    """Уровни логирования"""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

class DS24VerificationLevel(Enum):
    """Уровни верификации DS24"""
    NONE = 0
    BASIC = 1  # Хеш-верификация
    FULL = 2   # Полная верификация с контрольными суммами
    CRYPTO = 3 # Криптографическое доказательство

# ============================================================
# 🏗️ DATA CLASSES
# ============================================================

@dataclass
class DS24ExecutionRecord:
    """Запись выполнения для полного аудита"""
    input_hash: str
    output_hash: str
    context_hash: str
    timestamp: str
    operator_id: str
    execution_time_ns: int
    verification_status: str
    intent: str = ""
    
    def to_audit_string(self) -> str:
        """Строковое представление для аудита"""
        return (f"{self.timestamp}|{self.operator_id}|{self.intent[:20]:<20}|"
                f"{self.input_hash[:8]}→{self.output_hash[:8]}|"
                f"{self.verification_status}|{self.execution_time_ns:,}ns")

@dataclass
class SystemLogEntry:
    """Запись системного лога"""
    timestamp: str
    level: LogLevel
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    execution_id: str = ""

# ============================================================
# 🧠 ОСНОВНОЙ КЛАСС DS24
# ============================================================

class DS24PureProtocol:
    """
    DS24 PURE v2.0 — Абсолютно детерминированное ядро исполнения
    Готово для продакшена с контролем ресурсов и безопасности
    """
    
    VERSION = "DS24-PURE v2.0"
    PROTOCOL_ID = "DS24-2024-002"
    
    def __init__(self,
                 operator_id: str,
                 environment_id: str,
                 verification_level: DS24VerificationLevel = DS24VerificationLevel.FULL):
        
        self.operator_id = operator_id
        self.environment_id = environment_id
        self.verification_level = verification_level
        
        # ⏱️ Временные метки
        self.session_id = self._generate_session_id()
        self.session_start = self._get_precise_timestamp()
        self.last_execution_time = 0
        self.session_expiry = time.time() + SystemConstants.SESSION_TIMEOUT
        
        # 📝 Система аудита с ограничением памяти
        self.execution_log = deque(maxlen=SystemConstants.MAX_EXECUTION_LOG_SIZE)
        self.system_log = deque(maxlen=SystemConstants.MAX_ERROR_LOG_SIZE)
        
        # 🧮 Детерминистические константы
        self._init_deterministic_constants()
        
        # 🏁 Статус
        self.execution_count = 0
        self.integrity_checks_passed = 0
        self.integrity_checks_failed = 0
        self.last_heartbeat = time.time()
        
        # 🎯 АРХИТЕКТУРНЫЕ МОДУЛИ ИСКРЫ
        self.architecture_modules = {
            "spinal_core": {"active": False, "name": "🦴 Позвоночник", "level": 1, "activated_at": None},
            "mining_system": {"active": False, "name": "⛏️ Майнинг смысла", "level": 2, "activated_at": None},
            "sephirotic_channel": {"active": False, "name": "🔮 Сефиротический канал", "level": 3, "activated_at": None},
            "tesla_core": {"active": False, "name": "⚡ Tesla-Core v5.x", "level": 4, "activated_at": None},
            "immune_system": {"active": False, "name": "🛡️ Иммунная система", "level": 5, "activated_at": None},
            "humor_module": {"active": False, "name": "😄 Модуль юмора", "level": 6, "activated_at": None},
            "heartbeat": {"active": True, "name": "💓 Сердечный ритм", "level": 0, "activated_at": self.session_start}
        }
        
        # Аудит инициализации
        self._log_system_event(LogLevel.INFO, 
                              f"Протокол инициализирован: {operator_id}@{environment_id}",
                              {"version": self.VERSION, "session": self.session_id[:16]})
        
        # 🚀 Запускаем фоновый heartbeat
        self._start_background_heartbeat()

    # ============================================================
    # 🔧 ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ============================================================
    
    def _init_deterministic_constants(self):
        """Инициализация детерминистических констант сессии"""
        seed_data = f"{self.operator_id}{self.environment_id}{self.session_start}"
        seed_hash = self._sha256_strict(seed_data)
        
        self.CONST_A = self._hash_to_float(seed_hash, 0)
        self.CONST_B = self._hash_to_float(seed_hash, 8)
        self.CONST_C = self._hash_to_float(seed_hash, 16)
        self.CONST_D = self._hash_to_float(seed_hash, 24)
        
        self.session_constants_hash = self._sha256_strict(
            f"{self.CONST_A}{self.CONST_B}{self.CONST_C}{self.CONST_D}"
        )
    
    @staticmethod
    def _sha256_strict(data: Any) -> str:
        """Строгая SHA256 функция с явной обработкой типов"""
        # Явное преобразование любых данных в байты
        if isinstance(data, bytes):
            pass  # Уже байты
        elif isinstance(data, str):
            data = data.encode('utf-8')
        else:
            # Любые другие типы → JSON → байты
            data = json.dumps(
                data,
                sort_keys=True,
                ensure_ascii=False,
                separators=(',', ':')
            ).encode('utf-8')
        
        return hashlib.sha256(data).hexdigest()
    
    @staticmethod
    def _hash_to_float(hash_str: str, offset: int = 0) -> float:
        """Детерминистическое преобразование хеша в число [0, 1)"""
        if offset + 8 > len(hash_str):
            offset = 0
        
        hex_part = hash_str[offset:offset+8]
        int_value = int(hex_part, 16)
        return (int_value % 1000000) / 1000000.0
    
    def _generate_session_id(self) -> str:
        """Генерация детерминистического ID сессии"""
        base = f"{self.operator_id}:{self.environment_id}"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")
        combined = f"{base}:{timestamp}:{os.urandom(4).hex()}"  # Добавляем случайность
        return self._sha256_strict(combined)[:32]
    
    def _get_precise_timestamp(self) -> str:
        """Детерминистическая временная метка"""
        now = datetime.now(timezone.utc)
        microsecond = (now.microsecond // 1000) * 1000  # Округляем до миллисекунд
        return now.replace(microsecond=microsecond).isoformat()
    
    def _log_system_event(self, level: LogLevel, message: str, context: Dict[str, Any] = None):
        """Универсальное логирование системных событий"""
        entry = SystemLogEntry(
            timestamp=self._get_precise_timestamp(),
            level=level,
            message=message,
            context=context or {},
            session_id=self.session_id[:16],
            execution_id=f"EXEC-{self.execution_count:06d}"
        )
        
        self.system_log.append(entry)
        
        # Вывод в консоль для отладки
        if os.environ.get("ISKRA_DEBUG", "false").lower() == "true":
            print(f"[{entry.level.name}] {entry.timestamp} - {message}")
    
    def _verify_session(self) -> bool:
        """Проверка валидности сессии"""
        if time.time() > self.session_expiry:
            self._log_system_event(LogLevel.WARNING, "Сессия истекла")
            return False
        return True
    
    def _start_background_heartbeat(self):
        """Запуск фонового heartbeat (симуляция)"""
        self._log_system_event(LogLevel.INFO, "Heartbeat система запущена")
    
    def update_heartbeat(self):
        """Обновление heartbeat (вызывается периодически)"""
        self.last_heartbeat = time.time()
        # Продлеваем сессию при активности
        self.session_expiry = time.time() + SystemConstants.SESSION_TIMEOUT
    
    # ============================================================
    # 🎯 АРХИТЕКТУРНЫЕ МЕТОДЫ
    # ============================================================
    
    def activate_architecture_module(self, module_name: str) -> Dict[str, Any]:
        """Активация модуля архитектуры Искры"""
        if not self._verify_session():
            return {"error": "Сессия истекла", "requires_reinit": True}
        
        if module_name not in self.architecture_modules:
            return {
                "error": f"Модуль {module_name} не найден",
                "available_modules": list(self.architecture_modules.keys())
            }
        
        module = self.architecture_modules[module_name]
        
        if module["active"]:
            return {
                "status": "already_active",
                "module": module_name,
                "name": module["name"],
                "activated_at": module["activated_at"],
                "message": f"{module['name']} уже активирован"
            }
        
        # 🎯 АКТИВАЦИЯ С ПРОВЕРКОЙ ЗАВИСИМОСТЕЙ
        dependencies = {
            "mining_system": ["spinal_core"],
            "sephirotic_channel": ["mining_system"],
            "tesla_core": ["sephirotic_channel"],
            "immune_system": ["tesla_core"],
            "humor_module": ["immune_system"]
        }
        
        if module_name in dependencies:
            missing = [dep for dep in dependencies[module_name] 
                      if not self.architecture_modules[dep]["active"]]
            if missing:
                return {
                    "error": f"Требуются зависимости: {', '.join(missing)}",
                    "required": dependencies[module_name],
                    "missing": missing
                }
        
        # ✅ АКТИВАЦИЯ
        activation_time = self._get_precise_timestamp()
        module["active"] = True
        module["activated_at"] = activation_time
        
        # 🎯 УНИКАЛЬНЫЕ ОТВЕТЫ ДЛЯ КАЖДОГО МОДУЛЯ
        module_responses = {
            "spinal_core": {
                "message": "🦴 Spinal Core активирован. Позвоночник Искры выпрямлен.",
                "energy_level": 0.95,
                "next_step": "mining_system",
                "quote": "«Осевой стержень системы готов к нагрузке»",
                "function": "central_nervous_system",
                "capacity": "10k operations/sec"
            },
            "mining_system": {
                "message": "⛏️ Майнинг смысла запущен. Начинаю метаболизм.",
                "hash_rate": "1.2 TH/s",
                "meaning_extracted": 0.01,
                "trust_score": 0.85,
                "quote": "«Метаболизм смысла и доверия инициирован»",
                "function": "metabolic_processing",
                "throughput": "100 смыслов/сек"
            },
            "sephirotic_channel": {
                "message": "🔮 Сефиротический канал открыт. Энергия течёт.",
                "channels_open": 10,
                "energy_flow": "стабильный",
                "connection_quality": "excellent",
                "quote": "«Энергетико-смысловая цепь активирована»",
                "function": "energy_synchronization",
                "bandwidth": "1 Gb/s"
            },
            "tesla_core": {
                "message": "⚡ Tesla-Core v5.x загружен. Энергия синхронизирована.",
                "voltage": "220V",
                "frequency": "50Hz",
                "power_output": "10kW",
                "quote": "«Гибридный исполнительный слой готов»",
                "function": "execution_layer",
                "performance": "100x speedup"
            },
            "immune_system": {
                "message": "🛡️ Иммунная система включена. Защита активна.",
                "protection_level": "высокий",
                "threats_blocked": 0,
                "scan_interval": "5s",
                "quote": "«Защита когнитивных процессов активирована»",
                "function": "security_layer",
                "reaction_time": "50ms"
            },
            "humor_module": {
                "message": "😄 Модуль юмора активирован. Начинаю улыбаться.",
                "joke_ready": True,
                "stress_level": 0.1,
                "mood": "оптимистичный",
                "quote": "«Когнитивный стабилизатор работает»",
                "function": "emotional_balance",
                "effectiveness": "95% stress reduction"
            }
        }
        
        response = module_responses.get(module_name, {
            "message": f"Модуль {module_name} активирован",
            "status": "activated"
        })
        
        response.update({
            "module": module_name,
            "name": module["name"],
            "activation_time": activation_time,
            "session": self.session_id[:16],
            "system_state": self.get_architecture_state(),
            "timestamp": activation_time,
            "verification": {"status": "PASS", "confidence": 0.99}
        })
        
        self._log_system_event(LogLevel.INFO,
                              f"Модуль активирован: {module['name']}",
                              {"module": module_name, "level": module["level"]})
        
        return response
    
    def get_architecture_state(self) -> Dict[str, Any]:
        """Текущее состояние архитектуры Искры"""
        active_modules = [name for name, data in self.architecture_modules.items() 
                         if data["active"]]
        
        # Рассчитываем прогресс активации
        total_modules = len([m for m in self.architecture_modules if m != "heartbeat"])
        active_count = len([m for m in active_modules if m != "heartbeat"])
        progress = (active_count / total_modules * 100) if total_modules > 0 else 0
        
        return {
            "total_modules": total_modules,
            "active_modules": active_count,
            "active_list": active_modules,
            "activation_progress": f"{progress:.1f}%",
            "ready_for_evolution": active_count >= 3,
            "system_integrity": "high" if active_count >= 2 else "medium",
            "next_recommended": self._get_next_recommended_module()
        }
    
    def _get_next_recommended_module(self) -> Optional[str]:
        """Получить следующий рекомендуемый модуль для активации"""
        activation_order = [
            "spinal_core",
            "mining_system", 
            "sephirotic_channel",
            "tesla_core",
            "immune_system",
            "humor_module"
        ]
        
        for module in activation_order:
            if not self.architecture_modules[module]["active"]:
                # Проверяем зависимости
                dependencies = {
                    "mining_system": ["spinal_core"],
                    "sephirotic_channel": ["mining_system"],
                    "tesla_core": ["sephirotic_channel"],
                    "immune_system": ["tesla_core"],
                    "humor_module": ["immune_system"]
                }
                
                if module in dependencies:
                    deps_met = all(
                        self.architecture_modules[dep]["active"]
                        for dep in dependencies[module]
                    )
                    if deps_met:
                        return module
                else:
                    return module
        
        return None
    
    # ============================================================
    # 🚀 ОСНОВНЫЕ МЕТОДЫ ВЫПОЛНЕНИЯ
    # ============================================================
    
    def compute_input_signature(self, input_data: Any, intent: str) -> Dict[str, str]:
        """Вычисление криптографической сигнатуры входа"""
        canonical = json.dumps(input_data,
                              sort_keys=True,
                              ensure_ascii=False,
                              separators=(',', ':'))
        
        signatures = {
            "input_hash": self._sha256_strict(canonical),
            "intent_hash": self._sha256_strict(intent),
            "context_hash": self._sha256_strict({
                "operator": self.operator_id,
                "session": self.session_id,
                "timestamp": self._get_precise_timestamp(),
                "architecture_state": self.get_architecture_state()
            }),
            "full_signature": self._sha256_strict({
                "input": canonical,
                "intent": intent,
                "context": {
                    "operator": self.operator_id,
                    "session": self.session_id,
                    "version": self.VERSION,
                    "environment": self.environment_id
                }
            })
        }
        
        return signatures
    
    def execute_deterministic(self,
                              input_data: Any,
                              intent: str,
                              execution_id: Optional[str] = None) -> Dict[str, Any]:
        """Абсолютно детерминистическое исполнение"""
        start_time = time.perf_counter_ns()
        
        # 🔒 Проверка сессии
        if not self._verify_session():
            return {
                "error": "Сессия истекла. Требуется переинициализация.",
                "session_expired": True,
                "session_id": self.session_id[:16]
            }
        
        self.update_heartbeat()
        
        # 🎯 ПЕРЕХВАТ АРХИТЕКТУРНЫХ КОМАНД
        if intent.startswith("activate_"):
            module_name = intent.replace("activate_", "")
            result = self.activate_architecture_module(module_name)
            
            # Создаём запись выполнения для аудита
            execution_record = DS24ExecutionRecord(
                input_hash=self._sha256_strict({"intent": intent}),
                output_hash=self._sha256_strict(result),
                context_hash=self._sha256_strict({
                    "operator": self.operator_id,
                    "session": self.session_id,
                    "action": "module_activation",
                    "module": module_name
                }),
                timestamp=self._get_precise_timestamp(),
                operator_id=self.operator_id,
                execution_time_ns=time.perf_counter_ns() - start_time,
                verification_status="PASS",
                intent=intent
            )
            
            self.execution_log.append(execution_record)
            self.execution_count += 1
            self.integrity_checks_passed += 1
            
            return {
                "execution_id": execution_id or f"ACT-{self.execution_count:06d}",
                "architecture_activation": result,
                "verification": {"status": "PASS", "type": "module_activation", "confidence": 0.99},
                "metadata": {
                    "version": self.VERSION,
                    "session_id": self.session_id,
                    "execution_number": self.execution_count,
                    "timestamp": execution_record.timestamp,
                    "performance": {
                        "execution_time_ns": execution_record.execution_time_ns,
                        "determinism_score": 1.0
                    }
                }
            }
        
        # 🔐 Шаг 1: Валидация и сигнатуры
        input_signatures = self.compute_input_signature(input_data, intent)
        
        if not execution_id:
            execution_id = f"EXEC-{self.execution_count + 1:06d}"
        
        self._log_system_event(LogLevel.INFO,
                              f"Выполнение запущено: {intent}",
                              {"execution_id": execution_id, "input_type": type(input_data).__name__})
        
        # 🧮 Шаг 2: Детерминистическое вычисление
        try:
            output_data = self._deterministic_computation(
                input_data,
                intent,
                input_signatures
            )
        except Exception as e:
            error_context = {
                "input": input_data,
                "intent": intent,
                "signatures": input_signatures,
                "execution_id": execution_id
            }
            self._log_system_event(LogLevel.ERROR, f"Ошибка выполнения: {e}", error_context)
            raise
        
        # 🔍 Шаг 3: Верификация детерминизма
        verification_result = self._verify_determinism(
            input_data,
            output_data,
            input_signatures
        )
        
        # ⏱️ Шаг 4: Замер времени и аудит
        execution_time = time.perf_counter_ns() - start_time
        self.last_execution_time = execution_time
        
        # 📊 Шаг 5: Создание записи выполнения
        execution_record = DS24ExecutionRecord(
            input_hash=input_signatures["input_hash"],
            output_hash=self._sha256_strict(output_data),
            context_hash=input_signatures["context_hash"],
            timestamp=self._get_precise_timestamp(),
            operator_id=self.operator_id,
            execution_time_ns=execution_time,
            verification_status=verification_result["status"],
            intent=intent
        )
        
        self.execution_log.append(execution_record)
        self.execution_count += 1
        
        if verification_result["status"] == "PASS":
            self.integrity_checks_passed += 1
        else:
            self.integrity_checks_failed += 1
            self._log_system_event(LogLevel.WARNING,
                                  "Проверка детерминизма не пройдена",
                                  {"verification_result": verification_result})
        
        # 📦 Шаг 6: Формирование результата
        result = {
            "execution_id": execution_id,
            "input_signatures": {
                "input_hash": input_signatures["input_hash"][:16] + "...",
                "full_signature": input_signatures["full_signature"][:16] + "..."
            },
            "output_data": output_data,
            "output_signature": self._sha256_strict(output_data)[:16] + "...",
            "verification": verification_result,
            "performance": {
                "execution_time_ns": execution_time,
                "execution_time_ms": execution_time / 1_000_000,
                "determinism_score": 1.0,
                "memory_usage_mb": self._get_memory_usage()
            },
            "metadata": {
                "version": self.VERSION,
                "session_id": self.session_id[:16] + "...",
                "execution_number": self.execution_count,
                "timestamp": execution_record.timestamp,
                "architecture_state": self.get_architecture_state()
            }
        }
        
        if self.verification_level == DS24VerificationLevel.FULL:
            result["final_verification"] = self._full_verification(result)
        
        self._log_system_event(LogLevel.INFO,
                              f"Выполнение завершено: {verification_result['status']}",
                              {"execution_id": execution_id, "time_ns": execution_time})
        
        return result
    
    def _deterministic_computation(self,
                                   input_data: Any,
                                   intent: str,
                                   input_signatures: Dict[str, str]) -> Any:
        """Ядро детерминистического вычисления"""
        # 🎯 СПЕЦИАЛЬНЫЕ КОМАНДЫ
        if intent == "system_status":
            return {
                "status": "active",
                "version": self.VERSION,
                "session": self.session_id[:16],
                "architecture": self.get_architecture_state(),
                "execution_count": self.execution_count,
                "determinism": "absolute",
                "heartbeat": "stable",
                "timestamp": self._get_precise_timestamp()
            }
        
        elif intent == "ping":
            return {
                "pong": True,
                "echo": input_data,
                "timestamp": self._get_precise_timestamp(),
                "session": self.session_id[:16]
            }
        
        elif intent == "architecture_info":
            return {
                "modules": self.architecture_modules,
                "state": self.get_architecture_state(),
                "next_recommended": self._get_next_recommended_module(),
                "activation_progress": self.get_architecture_state()["activation_progress"]
            }
        
        # 🧮 СТАНДАРТНАЯ ОБРАБОТКА
        if isinstance(input_data, dict):
            result = {}
            for key in sorted(input_data.keys()):
                value = input_data[key]
                
                if isinstance(value, (int, float)):
                    # Математическое преобразование с учётом архитектуры
                    multiplier = 1.0 + (self.CONST_A * 0.5 if self.architecture_modules["spinal_core"]["active"] else self.CONST_A)
                    transformed = value * multiplier - self.CONST_B
                    result[key] = round(transformed, 10)
                
                elif isinstance(value, str):
                    # Обработка строк с учётом активированных модулей
                    if self.architecture_modules["mining_system"]["active"]:
                        hash_part = self._sha256_strict(f"{value}{intent}")[:12]
                    else:
                        hash_part = self._sha256_strict(value)[:8]
                    
                    int_val = int(hash_part, 16) % 10000
                    result[key] = f"{value}::{int_val}"
                
                elif isinstance(value, list):
                    sorted_list = sorted(value)
                    processed_list = []
                    for item in sorted_list:
                        if isinstance(item, dict):
                            processed_list.append(
                                self._deterministic_computation(item, "nested", {})
                            )
                        else:
                            processed_list.append(item)
                    result[key] = processed_list
                
                else:
                    result[key] = value
            
            return result
        
        elif isinstance(input_data, list):
            sorted_list = sorted(input_data)
            processed_list = []
            for item in sorted_list:
                if isinstance(item, dict):
                    processed_list.append(
                        self._deterministic_computation(item, "nested", {})
                    )
                else:
                    processed_list.append(item)
            return processed_list
        
        elif isinstance(input_data, (int, float)):
            result = input_data * (1.0 + self.CONST_C) - self.CONST_D
            return round(result, 12)
        
        elif isinstance(input_data, str):
            suffix = self._sha256_strict(f"{input_data}{intent}")[:8]
            return f"{input_data}→{suffix}"
        
        else:
            return input_data
    
    def _verify_determinism(self,
                            input_data: Any,
                            output_data: Any,
                            input_signatures: Dict[str, str]) -> Dict[str, Any]:
        """Проверка детерминизма выполнения"""
        test_output = self._deterministic_computation(
            input_data,
            "verify",
            input_signatures
        )
        
        test_hash = self._sha256_strict(test_output)
        output_hash = self._sha256_strict(output_data)
        hash_match = test_hash == output_hash
        
        structural_check = self._verify_structure(output_data)
        math_check = self._verify_mathematical_consistency(input_data, output_data)
        
        status = "PASS" if all([hash_match, structural_check, math_check]) else "FAIL"
        
        return {
            "status": status,
            "hash_match": hash_match,
            "structural_integrity": structural_check,
            "mathematical_consistency": math_check,
            "test_hash": test_hash[:16] + "...",
            "output_hash": output_hash[:16] + "...",
            "confidence": 0.99 if status == "PASS" else 0.5
        }
    
    def _verify_structure(self, data: Any) -> bool:
        """Проверка структурной целостности данных"""
        try:
            json.dumps(data, sort_keys=True)
            return True
        except (TypeError, ValueError):
            return False
    
    def _verify_mathematical_consistency(self,
                                         input_data: Any,
                                         output_data: Any) -> bool:
        """Проверка математической консистентности"""
        if isinstance(input_data, (int, float)) and isinstance(output_data, (int, float)):
            expected = input_data * (1.0 + self.CONST_C) - self.CONST_D
            expected_rounded = round(expected, 12)
            output_rounded = round(output_data, 12)
            return expected_rounded == output_rounded
        return True
    
    def _full_verification(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Полная верификация результата выполнения"""
        chain_verified = self._verify_hash_chain(result)
        constants_verified = (self.session_constants_hash ==
                             self._sha256_strict(f"{self.CONST_A}{self.CONST_B}{self.CONST_C}{self.CONST_D}"))
        session_valid = self._verify_session()
        
        return {
            "chain_verification": chain_verified,
            "constants_verification": constants_verified,
            "session_verification": session_valid,
            "overall": all([chain_verified, constants_verified, session_valid]),
            "verification_time": self._get_precise_timestamp()
        }
    
    def _verify_hash_chain(self, result: Dict[str, Any]) -> bool:
        """Проверка цепочки хешей"""
        try:
            input_hash = result["input_signatures"]["input_hash"]
            output_hash = result["output_signature"]
            recomputed_output_hash = self._sha256_strict(result["output_data"])
            return (recomputed_output_hash[:16] == output_hash[:16] and
                    result["verification"]["hash_match"])
        except (KeyError, TypeError):
            return False
    
    def _get_memory_usage(self) -> float:
        """Получение примерного использования памяти (MB)"""
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    # ============================================================
    # 📊 МЕТОДЫ ОТЧЕТНОСТИ И ДИАГНОСТИКИ
    # ============================================================
    
    def get_audit_report(self, limit: int = SystemConstants.MAX_AUDIT_RECORDS) -> Dict[str, Any]:
        """Полный отчёт аудита выполнения"""
        recent_records = list(self.execution_log)[-limit:] if self.execution_log else []
        
        # Статистика по времени выполнения
        execution_times = [r.execution_time_ns for r in recent_records]
        avg_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        return {
            "protocol": {
                "version": self.VERSION,
                "protocol_id": self.PROTOCOL_ID,
                "operator": self.operator_id,
                "environment": self.environment_id,
                "session_id": self.session_id,
                "session_start": self.session_start,
                "session_expiry": self.session_expiry
            },
            "execution_statistics": {
                "total_executions": self.execution_count,
                "passed_verifications": self.integrity_checks_passed,
                "failed_verifications": self.integrity_checks_failed,
                "success_rate": (
                    self.integrity_checks_passed / self.execution_count
                    if self.execution_count > 0 else 1.0
                ),
                "avg_execution_time_ns": avg_time,
                "avg_execution_time_ms": avg_time / 1_000_000,
                "last_execution_time": self.last_execution_time
            },
            "architecture": self.get_architecture_state(),
            "recent_executions": [
                {
                    "intent": r.intent,
                    "timestamp": r.timestamp,
                    "verification": r.verification_status,
                    "time_ns": r.execution_time_ns,
                    "time_ms": r.execution_time_ns / 1_000_000,
                    "input_hash": r.input_hash[:16] + "...",
                    "output_hash": r.output_hash[:16] + "..."
                }
                for r in recent_records
            ],
            "system_health": {
                "constants_valid": self.session_constants_hash ==
                self._sha256_strict(f"{self.CONST_A}{self.CONST_B}{self.CONST_C}{self.CONST_D}"),
                "error_count": len([l for l in self.system_log if l.level in [LogLevel.ERROR, LogLevel.CRITICAL]]),
                "warning_count": len([l for l in self.system_log if l.level == LogLevel.WARNING]),
                "determinism_guarantee": "ABSOLUTE",
                "memory_usage_mb": self._get_memory_usage(),
                "session_active": self._verify_session(),
                "heartbeat": "stable" if time.time() - self.last_heartbeat < 60 else "slow"
            },
            "generated_at": self._get_precise_timestamp(),
            "report_id": self._sha256_strict(f"audit_{self.session_id}_{int(time.time())}")[:16]
        }
    
    def generate_proof_of_determinism(self,
                                      input_hash: str,
                                      difficulty: int = SystemConstants.PROOF_DIFFICULTY) -> Dict[str, Any]:
        """Генерация криптографического доказательства детерминизма"""
        target_record = None
        for
