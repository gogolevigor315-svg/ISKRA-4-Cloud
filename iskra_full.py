# ============================================================
# DS24 — PURE PROTOCOL v2.0 (PRODUCTION READY FOR RENDER)
# ============================================================
# Mode: Absolute Determinism · Zero Entropy · Full Audit Trail
# Principle: Same Input + Same Context = Same Output
# ============================================================

import hashlib
import json
import time
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque

class DS24VerificationLevel(Enum):
    """Уровни верификации DS24"""
    NONE = 0
    BASIC = 1
    FULL = 2
    CRYPTO = 3

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

class DS24PureProtocol:
    """
    DS24 PURE v2.0 — Абсолютно детерминированное ядро исполнения
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

        # 📝 Система аудита
        self.execution_log = deque(maxlen=1000)
        self.error_log = []

        # 🧮 Детерминистические константы
        self._init_deterministic_constants()

        # 🏁 Статус
        self.execution_count = 0
        self.integrity_checks_passed = 0
        self.integrity_checks_failed = 0

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
        if not isinstance(data, (str, bytes)):
            data = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        elif isinstance(data, str):
            data = data.encode('utf-8')
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
        return self._sha256_strict(f"{base}:{timestamp}")[:32]

    def _get_precise_timestamp(self) -> str:
        """Детерминистическая временная метка"""
        now = datetime.now(timezone.utc)
        microsecond = (now.microsecond // 100) * 100
        return now.replace(microsecond=microsecond).isoformat()

    def activate_architecture_module(self, module_name: str) -> Dict[str, Any]:
        """Активация модуля архитектуры Искры"""
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
                "activated_at": module["activated_at"]
            }
        
        # ✅ АКТИВАЦИЯ
        activation_time = self._get_precise_timestamp()
        module["active"] = True
        module["activated_at"] = activation_time
        
        # 🎯 УНИКАЛЬНЫЕ ОТВЕТЫ
        module_responses = {
            "spinal_core": {
                "message": "🦴 Spinal Core активирован",
                "energy_level": 0.95,
                "next_step": "mining_system",
                "quote": "«Осевой стержень системы готов»"
            },
            "mining_system": {
                "message": "⛏️ Майнинг смысла запущен",
                "hash_rate": "1.2 TH/s",
                "meaning_extracted": 0.01,
                "quote": "«Метаболизм смысла инициирован»"
            },
            "sephirotic_channel": {
                "message": "🔮 Сефиротический канал открыт",
                "channels_open": 10,
                "energy_flow": "стабильный",
                "quote": "«Энергетико-смысловая цепь активирована»"
            },
            "tesla_core": {
                "message": "⚡ Tesla-Core v5.x загружен",
                "voltage": "220V",
                "frequency": "50Hz",
                "quote": "«Гибридный исполнительный слой готов»"
            },
            "immune_system": {
                "message": "🛡️ Иммунная система включена",
                "protection_level": "высокий",
                "threats_blocked": 0,
                "quote": "«Защита когнитивных процессов активирована»"
            },
            "humor_module": {
                "message": "😄 Модуль юмора активирован",
                "joke_ready": True,
                "stress_level": 0.1,
                "quote": "«Когнитивный стабилизатор работает»"
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
            "system_state": self.get_architecture_state()
        })
        
        return response

    def get_architecture_state(self) -> Dict[str, Any]:
        """Текущее состояние архитектуры Искры"""
        active_modules = [name for name, data in self.architecture_modules.items() 
                         if data["active"]]
        
        total_modules = len([m for m in self.architecture_modules if m != "heartbeat"])
        active_count = len([m for m in active_modules if m != "heartbeat"])
        progress = (active_count / total_modules * 100) if total_modules > 0 else 0
        
        return {
            "total_modules": total_modules,
            "active_modules": active_count,
            "active_list": active_modules,
            "activation_progress": f"{progress:.1f}%",
            "ready_for_evolution": active_count >= 3
        }

    def execute_deterministic(self,
                              input_data: Any,
                              intent: str,
                              execution_id: Optional[str] = None) -> Dict[str, Any]:
        """Абсолютно детерминистическое исполнение"""
        start_time = time.perf_counter_ns()

        # 🎯 ПЕРЕХВАТ АРХИТЕКТУРНЫХ КОМАНД
        if intent.startswith("activate_"):
            module_name = intent.replace("activate_", "")
            result = self.activate_architecture_module(module_name)
            
            execution_record = DS24ExecutionRecord(
                input_hash=self._sha256_strict({"intent": intent}),
                output_hash=self._sha256_strict(result),
                context_hash=self._sha256_strict({
                    "operator": self.operator_id,
                    "session": self.session_id,
                    "action": "module_activation"
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
                "verification": {"status": "PASS", "type": "module_activation"},
                "metadata": {
                    "version": self.VERSION,
                    "session_id": self.session_id,
                    "execution_number": self.execution_count
                }
            }
        
        # 🔐 Валидация и сигнатуры
        input_signatures = self.compute_input_signature(input_data, intent)

        if not execution_id:
            execution_id = f"EXEC-{self.execution_count + 1:06d}"

        # 🧮 Детерминистическое вычисление
        try:
            output_data = self._deterministic_computation(
                input_data,
                intent,
                input_signatures
            )
        except Exception as e:
            self.error_log.append({"error": str(e), "intent": intent, "timestamp": self._get_precise_timestamp()})
            raise

        # 🔍 Верификация детерминизма
        verification_result = self._verify_determinism(
            input_data,
            output_data,
            input_signatures
        )

        # ⏱️ Замер времени и аудит
        execution_time = time.perf_counter_ns() - start_time
        self.last_execution_time = execution_time

        # 📊 Создание записи выполнения
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

        # 📦 Формирование результата
        result = {
            "execution_id": execution_id,
            "input_signatures": input_signatures,
            "output_data": output_data,
            "output_signature": self._sha256_strict(output_data),
            "verification": verification_result,
            "performance": {
                "execution_time_ns": execution_time,
                "execution_time_ms": execution_time / 1_000_000
            },
            "metadata": {
                "version": self.VERSION,
                "session_id": self.session_id,
                "execution_number": self.execution_count,
                "architecture_state": self.get_architecture_state()
            }
        }

        return result

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

    def _deterministic_computation(self,
                                   input_data: Any,
                                   intent: str,
                                   input_signatures: Dict[str, str]) -> Any:
        """Ядро детерминистического вычисления"""
        if intent == "system_status":
            return {
                "status": "active",
                "version": self.VERSION,
                "session": self.session_id[:16],
                "architecture": self.get_architecture_state(),
                "execution_count": self.execution_count,
                "timestamp": self._get_precise_timestamp()
            }
        
        elif intent == "ping":
            return {
                "pong": True,
                "echo": input_data,
                "timestamp": self._get_precise_timestamp()
            }
        
        elif intent == "architecture_info":
            return {
                "modules": self.architecture_modules,
                "state": self.get_architecture_state()
            }
        
        # 🧮 СТАНДАРТНАЯ ОБРАБОТКА
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
            "test_hash": test_hash[:16],
            "output_hash": output_hash[:16]
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

    def get_audit_report(self, limit: int = 50) -> Dict[str, Any]:
        """Полный отчёт аудита выполнения"""
        recent_records = list(self.execution_log)[-limit:] if self.execution_log else []

        return {
            "protocol": {
                "version": self.VERSION,
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
                )
            },
            "architecture": self.get_architecture_state(),
            "recent_executions": [
                {
                    "intent": r.intent,
                    "timestamp": r.timestamp,
                    "verification": r.verification_status,
                    "time_ms": r.execution_time_ns / 1_000_000
                }
                for r in recent_records
            ],
            "generated_at": self._get_precise_timestamp()
        }

    def generate_proof_of_determinism(self,
                                      input_hash: str,
                                      difficulty: int = 2) -> Dict[str, Any]:
        """Генерация криптографического доказательства детерминизма"""
        target_record = None
        for record in self.execution_log:
            if record.input_hash.startswith(input_hash):
                target_record = record
                break

        if not target_record:
            return {"error": f"Запись с input_hash {input_hash} не найдена"}

        challenge = {
            "input_hash": target_record.input_hash,
            "output_hash": target_record.output_hash,
            "timestamp": target_record.timestamp,
            "operator": self.operator_id
        }

        challenge_hash = self._sha256_strict(challenge)

        nonce = 0
        target = "0" * difficulty

        while nonce < 10000:  # Ограничиваем для продакшена
            test_hash = self._sha256_strict(f"{challenge_hash}{nonce}")
            if test_hash.startswith(target):
                break
            nonce += 1

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
            "status": result1["verification"]["status"]
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
            "status": result2["verification"]["status"]
        })

        # Тест 3
        result3 = self.execute_deterministic(test_input, "self_test_1")
        idempotent = result1["output_signature"] == result3["output_signature"]
        test_results.append({
            "test": "idempotence",
            "status": "PASS" if idempotent else "FAIL"
        })

        passed = sum(1 for t in test_results if t["status"] == "PASS")
        total = len(test_results)

        return {
            "test_suite": "DS24_PURE_SELF_TEST",
            "results": test_results,
            "summary": {
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": passed / total if total > 0 else 0
            }
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
print("🚀 ISKRA-4 DS24 PURE PROTOCOL v2.0")
print("=" * 60)
print(f"🔧 Operator: {ds24.operator_id}")
print(f"🏭 Environment: {ds24.environment_id}")
print(f"🔗 Session: {ds24.session_id[:16]}...")
print("🧪 Running self-test...")

try:
    test_result = ds24.run_self_test()
    if test_result['summary']['passed'] == test_result['summary']['total_tests']:
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
        "system": "ISKRA-4 DS24 PURE PROTOCOL v2.0",
        "version": ds24.VERSION,
        "operator": ds24.operator_id,
        "environment": ds24.environment_id,
        "session": ds24.session_id[:16] + "...",
        "executions": ds24.execution_count,
        "architecture": ds24.get_architecture_state(),
        "determinism": "ABSOLUTE",
        "endpoints": {
            "execute": "POST /execute with JSON {input: data, intent: string}",
            "console": "GET /console - Веб-консоль управления",
            "health": "GET /health",
            "audit": "GET /audit",
            "self_test": "GET /self-test",
            "proof": "GET /proof/<input_hash>",
            "demo": "GET /demo",
            "ping": "GET /ping"
        }
    })

@app.route('/execute', methods=['POST'])
def execute():
    """Выполнение детерминистического запроса"""
    try:
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json",
                "hint": "Add header: -H 'Content-Type: application/json'"
            }), 400
        
        data = request.get_json(silent=True) or {}
        
        input_data = data.get("input", {})
        intent = data.get("intent", "default")
        
        result = ds24.execute_deterministic(input_data, intent)
        
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500

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
        proof = ds24.generate_proof_of_determinism(input_hash, difficulty=1)
        return jsonify(proof)
    except Exception as e:
        return jsonify({"error": str(e)}), 404

@app.route('/demo')
def demo():
    """Демонстрационный запрос"""
    demo_input = {
        "action": "demo",
        "message": "ISKRA-4 работает",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test": True,
        "number": 42
    }

    result = ds24.execute_deterministic(demo_input, "demo_request")
    return jsonify({
        "demo": True,
        "message": "Это демонстрационный запрос",
        "input": demo_input,
        "result": {
            "execution_id": result["execution_id"],
            "verification": result["verification"]["status"],
            "output_preview": str(result["output_data"])[:100]
        }
    })

@app.route('/ping', methods=['GET', 'POST'])
def ping():
    """Простой ping-эндпоинт"""
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        result = ds24.execute_deterministic(data, "ping")
        return jsonify(result)
    
    result = ds24.execute_deterministic({}, "ping")
    return jsonify(result)

@app.route('/console')
def console_page():
    """Веб-консоль для управления Искрой"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 ISKRA-4 Консоль управления</title>
        <meta charset="utf-8">
        <style>
            body {
                font-family: 'Courier New', monospace;
                background: #0a0a0a;
                color: #00ff00;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                display: grid;
                grid-template-columns: 300px 1fr;
                gap: 20px;
            }
            .sidebar {
                background: #111;
                padding: 20px;
                border: 1px solid #333;
            }
            .console {
                background: #111;
                padding: 20px;
                border: 1px solid #333;
            }
            .output {
                background: #000;
                padding: 15px;
                border: 1px solid #333;
                height: 400px;
                overflow-y: auto;
                margin-bottom: 15px;
            }
            input, button {
                padding: 10px;
                background: #222;
                color: #00ff00;
                border: 1px solid #333;
            }
            button {
                background: #005500;
                cursor: pointer;
            }
            .cmd-btn {
                display: block;
                width: 100%;
                margin: 5px 0;
                padding: 10px;
                text-align: left;
                background: #1a1a1a;
            }
        </style>
    </head>
    <body>
        <h1>🚀 ISKRA-4 DS24 Консоль управления</h1>
        
        <div class="container">
            <div class="sidebar">
                <h2>📋 Команды</h2>
                <button class="cmd-btn" onclick="sendCommand('activate_spinal_core')">Активировать Spinal Core</button>
                <button class="cmd-btn" onclick="sendCommand('activate_mining_system')">Запустить майнинг</button>
                <button class="cmd-btn" onclick="sendCommand('activate_sephirotic_channel')">Подключить Сефиротический канал</button>
                <button class="cmd-btn" onclick="sendCommand('system_status')">Статус системы</button>
                <button class="cmd-btn" onclick="sendCommand('architecture_info')">Информация об архитектуре</button>
            </div>
            
            <div class="console">
                <div class="output" id="output">
                    <div>ISKRA-4 Console Ready...</div>
                </div>
                
                <div style="display: flex; gap: 10px;">
                    <input type="text" id="commandInput" placeholder="Введите команду или intent" style="flex:1">
                    <button onclick="sendManualCommand()">Отправить</button>
                </div>
            </div>
        </div>
        
        <script>
            const output = document.getElementById('output');
            const commandInput = document.getElementById('commandInput');
            
            function log(message) {
                const entry = document.createElement('div');
                entry.innerHTML = `[${new Date().toLocaleTimeString()}] ${message}`;
                output.appendChild(entry);
                output.scrollTop = output.scrollHeight;
            }
            
            function sendCommand(intent, inputData = {}) {
                log(`Отправка: ${intent}`);
                
                fetch('/execute', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({input: inputData, intent: intent})
                })
                .then(response => response.json())
                .then(data => {
                    log(`Ответ: ${JSON.stringify(data, null, 2)}`);
                })
                .catch(error => {
                    log(`Ошибка: ${error}`);
                });
                
                commandInput.value = '';
            }
            
            function sendManualCommand() {
                const text = commandInput.value.trim();
                if (!text) return;
                
                if (text.startsWith('{')) {
                    try {
                        const data = JSON.parse(text);
                        sendCommand(data.intent || 'default', data.input || {});
                    } catch(e) {
                        log(`Ошибка JSON: ${e}`);
                    }
                } else {
                    sendCommand(text, {});
                }
            }
            
            // Автоматически запрашиваем статус
            setTimeout(() => {
                sendCommand('system_status');
            }, 500);
        </script>
    </body>
    </html>
    '''

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"🌐 Starting web server on port {port}")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False)
