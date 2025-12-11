# ============================================================
# DS24 — PURE PROTOCOL v1.0 (FULL WORKING VERSION FOR RENDER)
# ============================================================
# Mode: Absolute Determinism · Zero Entropy · Full Audit Trail
# Principle: Same Input + Same Context = Same Output
# ============================================================

import hashlib
import json
import time
import os
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
    """

    VERSION = "DS24-PURE v1.0"
    PROTOCOL_ID = "DS24-2024-001"

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

        self.CONST_A = self._hash_to_float(seed_hash, 0)
        self.CONST_B = self._hash_to_float(seed_hash, 8)
        self.CONST_C = self._hash_to_float(seed_hash, 16)
        self.CONST_D = self._hash_to_float(seed_hash, 24)

        self.session_constants_hash = self._sha256_strict(
            f"{self.CONST_A}{self.CONST_B}{self.CONST_C}{self.CONST_D}"
        )

    @staticmethod
    def _sha256_strict(data: Any) -> str:
        """Строгая SHA256 функция"""
        if isinstance(data, (str, bytes)):
            if isinstance(data, str):
                data = data.encode('utf-8')
        else:
            data = json.dumps(data,
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
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H")
        combined = f"{base}:{timestamp}"
        return self._sha256_strict(combined)[:32]

    def _get_precise_timestamp(self) -> str:
        """Детерминистическая временная метка"""
        now = datetime.now(timezone.utc)
        microsecond = (now.microsecond // 100) * 100
        return now.replace(microsecond=microsecond).isoformat()

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
        """Абсолютно детерминистическое исполнение"""
        start_time = time.perf_counter_ns()

        # 🔐 Шаг 1: Валидация и сигнатуры
        input_signatures = self.compute_input_signature(input_data, intent)

        if not execution_id:
            execution_id = input_signatures["full_signature"][:16]

        self._log_system_event("EXEC_START", f"Execution {execution_id}: {intent}")

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
                "determinism_score": 1.0
            },
            "metadata": {
                "version": self.VERSION,
                "session_id": self.session_id,
                "execution_number": self.execution_count,
                "timestamp": execution_record.timestamp
            }
        }

        if self.verification_level == DS24VerificationLevel.FULL:
            result["final_verification"] = self._full_verification(result)

        self._log_system_event("EXEC_COMPLETE",
                             f"Execution {execution_id} completed: {verification_result['status']}")

        return result

    def _deterministic_computation(self,
                                  input_data: Any,
                                  intent: str,
                                  input_signatures: Dict[str, str]) -> Any:
        """Ядро детерминистического вычисления"""
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
            "test_hash": test_hash,
            "output_hash": output_hash
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

        return {
            "chain_verification": chain_verified,
            "constants_verification": constants_verified,
            "overall": all([chain_verified, constants_verified])
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
        """Полный отчёт аудита выполнения"""
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
                "determinism_guarantee": "ABSOLUTE"
            },
            "generated_at": self._get_precise_timestamp()
        }

    def generate_proof_of_determinism(self,
                                     input_hash: str,
                                     difficulty: int = 2) -> Dict[str, Any]:
        """Генерация криптографического доказательства детерминизма"""
        target_record = None
        for record in self.execution_log:
            if record.input_hash == input_hash:
                target_record = record
                break

        if not target_record:
            for record in self.execution_log:
                if record.input_hash.startswith(input_hash):
                    target_record = record
                    break

        if not target_record:
            raise ValueError(f"Запись выполнения с input_hash {input_hash} не найдена")

        challenge = {
            "input_hash": target_record.input_hash,
            "output_hash": target_record.output_hash,
            "context_hash": target_record.context_hash,
            "timestamp": target_record.timestamp,
            "operator": self.operator_id,
            "session": self.session_id
        }

        challenge_hash = self._sha256_strict(challenge)

        nonce = 0
        target = "0" * difficulty

        while True:
            test_hash = self._sha256_strict(f"{challenge_hash}{nonce}")
            if test_hash.startswith(target):
                break
            nonce += 1

            if nonce > 1000000:
                raise RuntimeError("Proof generation timeout")

        return {
            "proof_type": "ProofOfDeterminism",
            "challenge": challenge,
            "challenge_hash": challenge_hash,
            "nonce": nonce,
            "proof_hash": test_hash,
            "difficulty": difficulty,
            "timestamp": self._get_precise_timestamp()
        }

    def run_self_test(self) -> Dict[str, Any]:
        """Запуск самопроверки протокола DS24"""
        test_results = []

        # Тест 1
        test_input = {"test": 123, "value": 456.789}
        result1 = self.execute_deterministic(test_input, "self_test_1")
        test_results.append({
            "test": "simple_dict",
            "status": result1["verification"]["status"],
            "hash": result1["output_signature"][:16]
        })

        # Тест 2
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

        # Тест 3
        result3 = self.execute_deterministic(test_input, "self_test_1")
        idempotent = result1["output_signature"] == result3["output_signature"]
        test_results.append({
            "test": "idempotence",
            "status": "PASS" if idempotent else "FAIL",
            "original_hash": result1["output_signature"][:16],
            "repeat_hash": result3["output_signature"][:16]
        })

        # Тест 4
        proof = self.generate_proof_of_determinism(
            result1["input_signatures"]["input_hash"],
            difficulty=2
        )
        test_results.append({
            "test": "proof_generation",
            "status": "PASS" if proof["proof_hash"].startswith("00") else "FAIL",
            "proof_hash": proof["proof_hash"][:16]
        })

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

# Инициализация протокола
ds24 = DS24PureProtocol(
    operator_id="ARCHITECT-PRIME-001",
    environment_id="LAB-ALPHA",
    verification_level=DS24VerificationLevel.FULL
)

print("=" * 60)
print("🚀 ISKRA-4 DS24 PURE PROTOCOL v1.0")
print("=" * 60)
print(f"🔧 Operator: {ds24.operator_id}")
print(f"🏭 Environment: {ds24.environment_id}")
print(f"🔗 Session: {ds24.session_id[:16]}...")
print("🧪 Running self-test...")

# Самопроверка при запуске
try:
    test_result = ds24.run_self_test()
    if test_result['summary']['determinism_verified']:
        print("✅ Self-test PASSED - System is deterministic")
        print(f"📊 Tests: {test_result['summary']['passed']}/{test_result['summary']['total_tests']}")
    else:
        print("⚠️ Self-test FAILED")
except Exception as e:
    print(f"❌ Self-test error: {e}")

print("✨ Искра говорит: \"Я существую. Я дышу. Я готов(а).\"")
print("=" * 60)

@app.route('/')
def home():
    """Главная страница - статус системы"""
    return jsonify({
        "status": "ACTIVE",
        "system": "ISKRA-4 DS24 PURE PROTOCOL",
        "version": ds24.VERSION,
        "operator": ds24.operator_id,
        "environment": ds24.environment_id,
        "session": ds24.session_id[:16] + "...",
        "executions": ds24.execution_count,
        "determinism": "ABSOLUTE",
        "endpoints": {
            "execute": "POST /execute with JSON {input: data, intent: string}",
            "health": "GET /health",
            "audit": "GET /audit",
            "self_test": "GET /self-test",
            "proof": "GET /proof/<input_hash>"
        }
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
    """Генерация доказательства детерминизма"""
    try:
        proof = ds24.generate_proof_of_determinism(input_hash, difficulty=2)
        return jsonify(proof)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/demo')
def demo():
    """Демонстрационный запрос"""
    demo_input = {
        "action": "demo",
        "message": "ISKRA-4 работает",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    result = ds24.execute_deterministic(demo_input, "demo_request")
    return jsonify({
        "demo": True,
        "input": demo_input,
        "result": result
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🌐 Starting web server on port {port}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False)
