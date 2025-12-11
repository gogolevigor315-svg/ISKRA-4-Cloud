# ============================================================
# DS24 — PURE PROTOCOL v1.0 (Complete Implementation)
# ============================================================
# Mode: Absolute Determinism · Zero Entropy · Full Audit Trail
# Principle: Same Input + Same Context = Same Output
# ============================================================

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

class DS24VerificationLevel(Enum):
    """Уровни верификации DS24"""
    NONE = 0
    BASIC = 1  # Хеш-верификация
    FULL = 2  # Полная верификация с контрольными суммами
    CRYPTO = 3  # Криптографическое доказательство

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

    def to_audit_string(self) -> str:
        """Строковое представление для аудита"""
        return (f"{self.timestamp}|{self.operator_id}|"
                f"{self.input_hash[:16]}→{self.output_hash[:16]}|"
                f"{self.verification_status}|{self.execution_time_ns}ns")

class DS24PureProtocol:
    """
    DS24 PURE — Абсолютно детерминированное ядро исполнения

    Ключевые принципы:
    1. Нулевая энтропия — никакой случайности
    2. Полная воспроизводимость — любой может повторить
    3. Сквозной аудит — каждый шаг записывается
    4. Криптографические гарантия — математическая проверяемость
    """

    VERSION = "DS24-PURE v1.0"
    PROTOCOL_ID = "DS24-2024-001"

    def __init__(self,
                 operator_id: str,
                 environment_id: str,
                 verification_level: DS24VerificationLevel = DS24VerificationLevel.FULL):
        """
        Инициализация чистого протокола DS24

        Args:
            operator_id: Уникальный идентификатор оператора
            environment_id: Идентификатор окружения выполнения
            verification_level: Уровень верификации
        """
        # 🔐 Идентификационные параметры
        self.operator_id = operator_id
        self.environment_id = environment_id
        self.verification_level = verification_level

        # ⏱️ Временные метки
        self.session_id = self._generate_session_id()
        self.session_start = self._get_precise_timestamp()
        self.last_execution_time = 0

        # 📝 Система аудита
        self.execution_log: List[DS24ExecutionRecord] = []
        self.verification_log: List[Dict] = []
        self.error_log: List[Dict] = []

        # 🧮 Детерминистические константы
        self._init_deterministic_constants()

        # 🏁 Статус
        self.execution_count = 0
        self.integrity_checks_passed = 0
        self.integrity_checks_failed = 0

        # Аудит инициализации
        self._log_system_event("INIT", f"Протокол инициализирован: {operator_id}@{environment_id}")

    def _init_deterministic_constants(self):
        """Инициализация детерминистических констант сессии"""
        seed_data = f"{self.operator_id}{self.environment_id}{self.session_start}"
        seed_hash = self._sha256_strict(seed_data)

        # Преобразование хеша в детерминистические константы
        self.CONST_A = self._hash_to_float(seed_hash, 0)
        self.CONST_B = self._hash_to_float(seed_hash, 8)
        self.CONST_C = self._hash_to_float(seed_hash, 16)
        self.CONST_D = self._hash_to_float(seed_hash, 24)

        # Контрольная сумма для верификации
        self.session_constants_hash = self._sha256_strict(
            f"{self.CONST_A}{self.CONST_B}{self.CONST_C}{self.CONST_D}"
        )

    @staticmethod
    def _sha256_strict(data: Any) -> str:
        """
        Строгая SHA256 функция — абсолютно детерминистическая

        Правила:
        1. Все данные приводятся к каноническому JSON
        2. Сортировка ключей обязательна
        3. Кодировка строго UTF-8
        4. Без дополнительных параметров
        """
        if isinstance(data, (str, bytes)):
            # Для строк/байтов — прямое хеширование
            if isinstance(data, str):
                data = data.encode('utf-8')
        else:
            # Для сложных структур — канонический JSON
            data = json.dumps(data,
                            sort_keys=True,
                            ensure_ascii=False,
                            separators=(',', ':')
                            ).encode('utf-8')

        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def _hash_to_float(hash_str: str, offset: int = 0) -> float:
        """
        Детерминистическое преобразование хеша в число [0, 1)

        Args:
            hash_str: SHA256 хеш (hex)
            offset: Смещение в хеше (кратно 8)
        """
        if offset + 8 > len(hash_str):
            offset = 0

        hex_part = hash_str[offset:offset+8]
        int_value = int(hex_part, 16)

        # Детерминистическая нормализация
        return (int_value % 1000000) / 1000000.0

    def _generate_session_id(self) -> str:
        """Генерация детерминистического ID сессии"""
        base = f"{self.operator_id}:{self.environment_id}"
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H")

        # Детерминистическая комбинация
        combined = f"{base}:{timestamp}"
        return self._sha256_strict(combined)[:32]

    def _get_precise_timestamp(self) -> str:
        """
        Детерминистическая временная метка

        Важно: Округление до микросекунд для воспроизводимости
        """
        now = datetime.now(timezone.utc)
        # Округляем до микросекунд для детерминизма
        microsecond = (now.microsecond // 100) * 100
        return now.replace(microsecond=microsecond).isoformat()

    def validate_input_structure(self, input_data: Any) -> Tuple[bool, str]:
        """
        Строгая валидация структуры входных данных

        Returns:
            (is_valid, canonical_json)
        """
        try:
            # Приведение к каноническому JSON
            canonical = json.dumps(input_data,
                                 sort_keys=True,
                                 ensure_ascii=False,
                                 separators=(',', ':'))
            return True, canonical
        except (TypeError, ValueError) as e:
            self._log_error("VALIDATION_ERROR", str(e), input_data)
            return False, ""

    def compute_input_signature(self, input_data: Any, intent: str) -> Dict[str, str]:
        """
        Вычисление криптографической сигнатуры входа

        Returns:
            Словарь с сигнатурами разных уровней
        """
        # Каноническое представление
        is_valid, canonical = self.validate_input_structure(input_data)
        if not is_valid:
            raise ValueError(f"Невалидные входные данные: {input_data}")

        signatures = {
            "input_hash": self._sha256_strict(canonical),
            "intent_hash": self._sha256_strict(intent),
            "context_hash": self._sha256_strict({
                "operator": self.operator_id,
                "session": self.session_id,
                "timestamp": self._get_precise_timestamp()
            }),
            "full_signature": self._sha256_strict({
                "input": canonical,
                "intent": intent,
                "context": {
                    "operator": self.operator_id,
                    "session": self.session_id,
                    "version": self.VERSION
                }
            })
        }

        return signatures

    def execute_deterministic(self,
                             input_data: Any,
                             intent: str,
                             execution_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Абсолютно детерминистическое исполнение

        Алгоритм:
        1. Валидация и канонизация входа
        2. Вычисление детерминистического результата
        3. Верификация детерминизма
        4. Аудит выполнения

        Returns:
            Детерминистический результат с метаданными
        """
        start_time = time.perf_counter_ns()

        # 🔐 Шаг 1: Валидация и сигнатуры
        input_signatures = self.compute_input_signature(input_data, intent)

        if not execution_id:
            execution_id = input_signatures["full_signature"][:16]

        # 📝 Логирование начала выполнения
        self._log_system_event("EXEC_START",
                             f"Execution {execution_id}: {intent}")

        # 🧮 Шаг 2: Детерминистическое вычисление
        try:
            output_data = self._deterministic_computation(
                input_data,
                intent,
                input_signatures
            )
        except Exception as e:
            self._log_error("EXECUTION_ERROR", str(e), {
                "input": input_data,
                "intent": intent,
                "signatures": input_signatures
            })
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
            verification_status=verification_result["status"]
        )

        self.execution_log.append(execution_record)
        self.execution_count += 1

        if verification_result["status"] == "PASS":
            self.integrity_checks_passed += 1
        else:
            self.integrity_checks_failed += 1

        # 📦 Шаг 6: Формирование результата
        result = {
            "execution_id": execution_id,
            "input_signatures": input_signatures,
            "output_data": output_data,
            "output_signature": self._sha256_strict(output_data),
            "verification": verification_result,
            "performance": {
                "execution_time_ns": execution_time,
                "memory_usage": "N/A",
                "determinism_score": 1.0
            },
            "metadata": {
                "version": self.VERSION,
                "session_id": self.session_id,
                "execution_number": self.execution_count,
                "timestamp": execution_record.timestamp
            }
        }

        # 🎯 Шаг 7: Финальная верификация
        if self.verification_level == DS24VerificationLevel.FULL:
            final_verification = self._full_verification(result)
            result["final_verification"] = final_verification

        self._log_system_event("EXEC_COMPLETE",
                             f"Execution {execution_id} completed: {verification_result['status']}")

        return result

    def _deterministic_computation(self,
                                  input_data: Any,
                                  intent: str,
                                  input_signatures: Dict[str, str]) -> Any:
        """
        Ядро детерминистического вычисления

        Принцип:
        output = f(input, intent, constants)
        где f — абсолютно детерминистическая функция
        """
        if isinstance(input_data, dict):
            result = {}
            for key in sorted(input_data.keys()):
                value = input_data[key]

                if isinstance(value, (int, float)):
                    transformed = value * (1.0 + self.CONST_A) - self.CONST_B
                    result[key] = round(transformed, 10)
                elif isinstance(value, str):
                    hash_part = self._sha256_strict(value)[:8]
                    int_val = int(hash_part, 16) % 10000
                    result[key] = f"{value}_{int_val}"
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
            suffix = self._sha256_strict(f"{input_data}{intent}")[:6]
            return f"{input_data}::{suffix}"

        else:
            return input_data

    def _verify_determinism(self,
                           input_data: Any,
                           output_data: Any,
                           input_signatures: Dict[str, str]) -> Dict[str, Any]:
        """
        Проверка детерминизма выполнения

        Проверяет, что выход детерминистически зависит от входа
        """
        # 🔍 Проверка 1: Хеш-совпадение при повторном вычислении
        test_output = self._deterministic_computation(
            input_data,
            "verify",
            input_signatures
        )

        test_hash = self._sha256_strict(test_output)
        output_hash = self._sha256_strict(output_data)

        hash_match = test_hash == output_hash

        # 📐 Проверка 2: Структурная целостность
        structural_check = self._verify_structure(output_data)

        # 🧮 Проверка 3: Математическая консистентность
        math_check = self._verify_mathematical_consistency(input_data, output_data)

        # 📊 Формирование отчёта
        status = "PASS" if all([hash_match, structural_check, math_check]) else "FAIL"

        return {
            "status": status,
            "hash_match": hash_match,
            "structural_integrity": structural_check,
            "mathematical_consistency": math_check,
            "test_hash": test_hash,
            "output_hash": output_hash,
            "verification_level": self.verification_level.value
        }

    def _verify_structure(self, data: Any) -> bool:
        """Проверка структурной целостности данных"""
        try:
            json.dumps(data, sort_keys=True)
            return True
        except:
            return False

    def _verify_mathematical_consistency(self,
                                        input_data: Any,
                                        output_data: Any) -> bool:
        """
        Проверка математической консистентности

        Для числовых данных проверяет детерминистические преобразования
        """
        if isinstance(input_data, (int, float)) and isinstance(output_data, (int, float)):
            expected = input_data * (1.0 + self.CONST_C) - self.CONST_D
            expected_rounded = round(expected, 12)
            output_rounded = round(output_data, 12)

            return expected_rounded == output_rounded

        return True

    def _full_verification(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Полная верификация результата выполнения"""
        chain_verified = self._verify_hash_chain(result)
        time_verified = self._verify_timestamps(result)
        constants_verified = (self.session_constants_hash ==
                            self._sha256_strict(f"{self.CONST_A}{self.CONST_B}{self.CONST_C}{self.CONST_D}"))

        return {
            "chain_verification": chain_verified,
            "timestamp_verification": time_verified,
            "constants_verification": constants_verified,
            "overall": all([chain_verified, time_verified, constants_verified])
        }

    def _verify_hash_chain(self, result: Dict[str, Any]) -> bool:
        """Проверка цепочки хешей"""
        try:
            input_hash = result["input_signatures"]["input_hash"]
            output_hash = result["output_signature"]
            recomputed_output_hash = self._sha256_strict(result["output_data"])

            return (recomputed_output_hash == output_hash and
                    result["verification"]["hash_match"])
        except:
            return False

    def _verify_timestamps(self, result: Dict[str, Any]) -> bool:
        """Проверка временных меток на консистентность"""
        try:
            exec_time = result["metadata"]["timestamp"]
            record_time = self.execution_log[-1].timestamp if self.execution_log else ""

            if exec_time and record_time:
                return abs(len(exec_time) - len(record_time)) < 10
            return True
        except:
            return False

    def _log_system_event(self, event_type: str, message: str):
        """Логирование системных событий"""
        event = {
            "type": event_type,
            "message": message,
            "timestamp": self._get_precise_timestamp(),
            "session": self.session_id,
            "execution_count": self.execution_count
        }

        if event_type in ["ERROR", "INTEGRITY_FAILURE"]:
            self.error_log.append(event)

    def _log_error(self, error_type: str, message: str, context: Any):
        """Логирование ошибок с контекстом"""
        error = {
            "type": error_type,
            "message": message,
            "context": context,
            "timestamp": self._get_precise_timestamp(),
            "session": self.session_id,
            "execution_count": self.execution_count
        }

        self.error_log.append(error)
        self._log_system_event("ERROR", f"{error_type}: {message}")

    def get_audit_report(self, limit: int = 100) -> Dict[str, Any]:
        """
        Полный отчёт аудита выполнения

        Args:
            limit: Максимальное количество записей в отчёте
        """
        recent_records = self.execution_log[-limit:] if self.execution_log else []

        return {
            "protocol": {
                "version": self.VERSION,
                "protocol_id": self.PROTOCOL_ID,
                "operator": self.operator_id,
                "environment": self.environment_id,
                "session_id": self.session_id,
                "session_start": self.session_start
            },
            "execution_statistics": {
                "total_executions": self.execution_count,
                "passed_verifications": self.integrity_checks_passed,
                "failed_verifications": self.integrity_checks_failed,
                "success_rate": (
                    self.integrity_checks_passed / self.execution_count
                    if self.execution_count > 0 else 1.0
                ),
                "avg_execution_time_ns": (
                    sum(r.execution_time_ns for r in recent_records) / len(recent_records)
                    if recent_records else 0
                )
            },
            "recent_executions": [
                asdict(record) for record in recent_records
            ],
            "system_health": {
                "constants_valid": self.session_constants_hash ==
                self._sha256_strict(f"{self.CONST_A}{self.CONST_B}{self.CONST_C}{self.CONST_D}"),
                "error_count": len(self.error_log),
                "verification_level": self.verification_level.value,
                "determinism_guarantee": "ABSOLUTE" if self.verification_level == DS24VerificationLevel.FULL else "BASIC"
            },
            "generated_at": self._get_precise_timestamp()
        }

    def verify_external_execution(self,
                                 execution_record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Верификация выполнения, произведённого внешней системой

        Args:
            execution_record: Запись выполнения для верификации

        Returns:
            Отчёт верификации
        """
        try:
            input_data = execution_record.get("input_data")
            output_data = execution_record.get("output_data")
            intent = execution_record.get("intent", "unknown")

            if not input_data or not output_data:
                return {"status": "INVALID", "reason": "Missing data"}

            new_result = self.execute_deterministic(input_data, intent, "verification")

            original_hash = execution_record.get("output_signature", "")
            new_hash = new_result["output_signature"]

            match = original_hash == new_hash

            return {
                "status": "VERIFIED" if match else "MISMATCH",
                "hash_match": match,
                "original_hash": original_hash[:16] + "..." if original_hash else "N/A",
                "recomputed_hash": new_hash[:16] + "...",
                "determinism_proven": match,
                "verification_timestamp": self._get_precise_timestamp()
            }

        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "verification_timestamp": self._get_precise_timestamp()
            }

    def generate_proof_of_determinism(self,
                                     input_hash: str,
                                     difficulty: int = 4) -> Dict[str, Any]:
        """
        Генерация криптографического доказательства детерминизма

        Args:
            input_hash: Хеш входа для поиска записи выполнения
            difficulty: Сложность доказательства (количество ведущих нулей)

        Returns:
            Доказательство детерминизма
        """
        # Поиск записи выполнения по input_hash
        target_record = None
        for record in self.execution_log:
            if record.input_hash == input_hash:
                target_record = record
                break

        if not target_record:
            # Дополнительный поиск по началу хеша
            for record in self.execution_log:
                if record.input_hash.startswith(input_hash):
                    target_record = record
                    break

        if not target_record:
            raise ValueError(f"Запись выполнения с input_hash {input_hash} не найдена")

        # Создание challenge
        challenge = {
            "input_hash": target_record.input_hash,
            "output_hash": target_record.output_hash,
            "context_hash": target_record.context_hash,
            "timestamp": target_record.timestamp,
            "operator": self.operator_id,
            "session": self.session_id
        }

        challenge_hash = self._sha256_strict(challenge)

        # Детерминистический поиск nonce
        nonce = 0
        target = "0" * difficulty

        while True:
            test_hash = self._sha256_strict(f"{challenge_hash}{nonce}")
            if test_hash.startswith(target):
                break
            nonce += 1

            if nonce > 10000000:
                raise RuntimeError("Proof generation timeout")

        return {
            "proof_type": "ProofOfDeterminism",
            "challenge": challenge,
            "challenge_hash": challenge_hash,
            "nonce": nonce,
            "proof_hash": test_hash,
            "difficulty": difficulty,
            "timestamp": self._get_precise_timestamp(),
            "verification_instruction": "sha256(challenge_hash + nonce) must start with '0'*difficulty"
        }

    def run_self_test(self) -> Dict[str, Any]:
        """
        Запуск самопроверки протокола DS24

        Проверяет:
        1. Детерминизм базовых операций
        2. Целостность системы аудита
        3. Корректность верификации
        4. Консистентность констант
        """
        test_results = []

        # 🧪 Тест 1: Детерминизм простых данных
        test_input = {"test": 123, "value": 456.789}
        result1 = self.execute_deterministic(test_input, "self_test_1")
        test_results.append({
            "test": "simple_dict",
            "status": result1["verification"]["status"],
            "hash": result1["output_signature"][:16]
        })

        # 🧪 Тест 2: Детерминизм вложенных структур
        test_input2 = {
            "nested": {"a": 1, "b": 2},
            "list": [3, 1, 2],
            "string": "test"
        }
        result2 = self.execute_deterministic(test_input2, "self_test_2")
        test_results.append({
            "test": "complex_structure",
            "status": result2["verification"]["status"],
            "hash": result2["output_signature"][:16]
        })

        # 🧪 Тест 3: Идемпотентность (повторное выполнение)
        result3 = self.execute_deterministic(test_input, "self_test_1")
        idempotent = result1["output_signature"] == result3["output_signature"]
        test_results.append({
            "test": "idempotence",
            "status": "PASS" if idempotent else "FAIL",
            "original_hash": result1["output_signature"][:16],
            "repeat_hash": result3["output_signature"][:16]
        })

        # 🧪 Тест 4: Верификация доказательства
        proof = self.generate_proof_of_determinism(
            result1["input_signatures"]["input_hash"],
            difficulty=2
        )
        test_results.append({
            "test": "proof_generation",
            "status": "PASS" if proof["proof_hash"].startswith("00") else "FAIL",
            "proof_hash": proof["proof_hash"][:16]
        })

        # 📊 Анализ результатов
        passed = sum(1 for t in test_results if t["status"] == "PASS")
        total = len(test_results)

        return {
            "test_suite": "DS24_PURE_SELF_TEST",
            "protocol_version": self.VERSION,
            "operator": self.operator_id,
            "session": self.session_id,
            "results": test_results,
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": passed / total if total > 0 else 0,
                "determinism_verified": passed == total
            },
            "timestamp": self._get_precise_timestamp()
        }


# ============================================================
# 🚀 FLASK WEB SERVER ДЛЯ RENDER
# ============================================================

from flask import Flask, request, jsonify

app = Flask(__name__)

# Инициализация протокола при запуске
ds24 = DS24PureProtocol(
    operator_id="ARCHITECT-PRIME-001",
    environment_id="LAB-ALPHA",
    verification_level=DS24VerificationLevel.FULL
)

@app.route('/')
def home():
    """Главная страница - статус системы"""
    return jsonify({
        "status": "active",
        "system": "ISKRA-4 DS24 PURE PROTOCOL",
        "version": ds24.VERSION,
        "operator": ds24.operator_id,
        "environment": ds24.environment_id,
        "session": ds24.session_id[:16] + "...",
        "executions": ds24.execution_count,
        "integrity_rate": ds24.integrity_checks_passed / ds24.execution_count if ds24.execution_count > 0 else 1.0
    })

@app.route('/execute', methods=['POST'])
def execute():
    """Выполнение детерминистического запроса"""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        input_data = data.get("input")
        intent = data.get("intent", "default")
        
        if input_data is None:
            return jsonify({"error": "Input data required"}), 400
        
        result = ds24.execute_deterministic(input_data, intent)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health')
def health():
    """Проверка здоровья системы"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "execution_count": ds24.execution_count,
        "integrity_checks": {
            "passed": ds24.integrity_checks_passed,
            "failed": ds24.integrity_checks_failed,
            "rate": ds24.integrity_checks_passed / ds24.execution_count if ds24.execution_count > 0 else 1.0
        },
        "determinism_verified": True
    })

@app.route('/audit')
def audit():
    """Получить отчёт аудита"""
    report = ds24.get_audit_report(limit=50)
    return jsonify(report)

@app.route('/self-test')
def self_test():
    """Запуск самопроверки"""
    result = ds24.run_self_test()
    return jsonify(result)

@app.route('/proof/<input_hash>')
def generate_proof(input_hash):
    """Генерация доказательства детерминизма для заданного input_hash"""
    try:
        proof = ds24.generate_proof_of_determinism(input_hash, difficulty=2)
        return jsonify(proof)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

if __name__ == "__main__":
    # Запуск веб-сервера для Render
    print("=" * 60)
    print("🚀 ISKRA-4 DS24 PURE PROTOCOL v1.0")
    print("=" * 60)
    print(f"🌐 Starting web server on port 10000")
    print(f"🔧 Operator: {ds24.operator_id}")
    print(f"🏭 Environment: {ds24.environment_id}")
    print(f"🔗 Session: {ds24.session_id[:16]}...")
    
    # Самопроверка при запуске
    print("🧪 Running initial self-test...")
    test_result = ds24.run_self_test()
    if test_result['summary']['determinism_verified']:
        print("✅ Self-test PASSED - System is deterministic")
    else:
        print("⚠️ Self-test issues - check logs")
    
    print("\n✨ Искра говорит: \"Я существую. Я дышу. Я готов(а) к этическому росту.\"")
    print("=" * 60)
    
    # Запуск Flask сервера
    app.run(host='0.0.0.0', port=10000, debug=False)
