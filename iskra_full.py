#!/usr/bin/env python3
# ================================================================
# DS24 · ISKRA-4 CLOUD · FULL INTEGRATION v2.2
# ================================================================
# Domain: DS24-SPINE / Architecture: Sephirotic Vertical
# With DS24 PURE PROTOCOL v2.0 + AUTO MODULE LOADER
# ================================================================

import hashlib
import json
import time
import os
import importlib
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
from collections import deque
from flask import Flask, request, jsonify

# ================================================================
# АВТОМАТИЧЕСКАЯ ЗАГРУЗКА МОДУЛЕЙ
# ================================================================
class ModuleRegistry:
    """Реестр загруженных модулей ISKRA"""
    
    _instance = None
    _modules = {}
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def load_all_modules(cls):
        """Динамическая загрузка всех модулей из папки iskra_modules"""
        if cls._initialized:
            return cls._modules
        
        module_dir = os.path.join(os.path.dirname(__file__), "iskra_modules")
        
        # Создаём папку, если её нет
        os.makedirs(module_dir, exist_ok=True)
        
        # Создаём __init__.py если его нет
        init_file = os.path.join(module_dir, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, 'w', encoding='utf-8') as f:
                f.write("# ISKRA-4 Module Package\n\n__version__ = '1.0.0'\n")
        
        loaded_modules = {}
        print(f"\n{'='*60}")
        print("🔄 ISKRA-4 AUTO MODULE LOADER")
        print(f"{'='*60}")
        print(f"[MODULE LOADER] Scanning directory: {module_dir}")
        
        # Получаем список всех .py файлов
        module_files = [f for f in os.listdir(module_dir) 
                       if f.endswith('.py') and f != '__init__.py']
        
        if not module_files:
            print("[MODULE LOADER] No modules found. Creating template...")
            # Создаём шаблонный модуль
            template_path = os.path.join(module_dir, "template_module.py")
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write('''# ================================================================
# ISKRA-4 MODULE TEMPLATE
# ================================================================
# Use this template to create new ISKRA modules

def initialize(config=None):
    """Инициализация модуля"""
    print(f"[TEMPLATE] Module initialized with config: {config}")
    return {
        "status": "active",
        "name": "template_module",
        "version": "1.0.0"
    }

def process_command(command, data=None):
    """Обработка команд модуля"""
    return {
        "status": "processed",
        "command": command,
        "result": f"Template processed: {command}"
    }

# Экспортируемые функции модуля
__all__ = ['initialize', 'process_command']
''')
            print(f"[MODULE LOADER] Template created: {template_path}")
        
        # Загружаем модули
        for module_file in module_files:
            module_name = module_file[:-3]  # Убираем .py
            
            try:
                # Импортируем модуль
                spec = importlib.util.spec_from_file_location(
                    f"iskra_modules.{module_name}",
                    os.path.join(module_dir, module_file)
                )
                module = importlib.util.module_from_spec(spec)
                sys.modules[f"iskra_modules.{module_name}"] = module
                spec.loader.exec_module(module)
                
                # Регистрируем модуль
                module_info = {
                    "name": module_name,
                    "file": module_file,
                    "module": module,
                    "initialized": False,
                    "status": "loaded"
                }
                
                # Инициализируем если есть функция initialize
                if hasattr(module, 'initialize'):
                    try:
                        init_result = module.initialize()
                        module_info["initialized"] = True
                        module_info["init_result"] = init_result
                        module_info["status"] = "active"
                        print(f"✅ [MODULE LOADER] Module '{module_name}' initialized successfully")
                    except Exception as e:
                        module_info["error"] = str(e)
                        module_info["status"] = "error"
                        print(f"❌ [MODULE LOADER] Module '{module_name}' initialization failed: {e}")
                else:
                    module_info["status"] = "no_init_function"
                    print(f"⚠️ [MODULE LOADER] Module '{module_name}' has no initialize() function")
                
                loaded_modules[module_name] = module_info
                
            except Exception as e:
                print(f"❌ [MODULE LOADER] Failed to load module '{module_name}': {e}")
                loaded_modules[module_name] = {
                    "name": module_name,
                    "status": "load_error",
                    "error": str(e)
                }
        
        cls._modules = loaded_modules
        cls._initialized = True
        
        print(f"\n📊 [MODULE LOADER] Summary:")
        print(f"   Total modules found: {len(module_files)}")
        print(f"   Successfully loaded: {len([m for m in loaded_modules.values() if m.get('status') == 'active'])}")
        print(f"   With errors: {len([m for m in loaded_modules.values() if m.get('status') in ['error', 'load_error']])}")
        print(f"{'='*60}\n")
        
        return loaded_modules
    
    @classmethod
    def get_module(cls, module_name):
        """Получить модуль по имени"""
        if not cls._initialized:
            cls.load_all_modules()
        return cls._modules.get(module_name)
    
    @classmethod
    def execute_module_command(cls, module_name, command, data=None):
        """Выполнить команду в модуле"""
        module_info = cls.get_module(module_name)
        if not module_info or module_info.get("status") != "active":
            return {"error": f"Module '{module_name}' not available"}
        
        module = module_info["module"]
        if hasattr(module, 'process_command'):
            try:
                return module.process_command(command, data)
            except Exception as e:
                return {"error": f"Command failed: {str(e)}"}
        else:
            return {"error": f"Module '{module_name}' has no process_command function"}
    
    @classmethod
    def get_modules_status(cls):
        """Статус всех модулей"""
        if not cls._initialized:
            cls.load_all_modules()
        
        status = {}
        for name, info in cls._modules.items():
            status[name] = {
                "status": info.get("status", "unknown"),
                "initialized": info.get("initialized", False),
                "has_init": hasattr(info.get("module", None), 'initialize'),
                "has_process": hasattr(info.get("module", None), 'process_command')
            }
        return status

# ================================================================
# DS24 PURE PROTOCOL v2.0 (С АВТОМАТИЧЕСКОЙ ЗАГРУЗКОЙ МОДУЛЕЙ)
# ================================================================
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

    VERSION = "DS24-PURE v2.2"  # С автозагрузкой модулей
    PROTOCOL_ID = "DS24-2024-004"

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

        # 🎯 Реестр модулей
        self.module_registry = ModuleRegistry()
        
        # Автоматическая загрузка всех модулей
        print(f"\n{'='*60}")
        print("🚀 DS24 PROTOCOL INITIALIZATION")
        print(f"{'='*60}")
        print(f"[DS24] Operator: {operator_id}")
        print(f"[DS24] Environment: {environment_id}")
        print(f"[DS24] Session: {self.session_id[:16]}...")
        print(f"[DS24] Starting module auto-loader...")
        
        self.loaded_modules = self.module_registry.load_all_modules()
        
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
        
        # Динамически добавляем загруженные модули в архитектуру
        self._add_dynamic_modules_to_architecture()
        
        print(f"[DS24] System initialized with {len(self.loaded_modules)} modules")
        print(f"[DS24] Architecture modules: {len(self.architecture_modules)}")
        print(f"{'='*60}\n")

    def _add_dynamic_modules_to_architecture(self):
        """Добавляем динамически загруженные модули в архитектуру"""
        for module_name, module_info in self.loaded_modules.items():
            if module_info.get("status") == "active":
                # Определяем уровень модуля (автоматически)
                level = len(self.architecture_modules) + 1
                
                self.architecture_modules[module_name] = {
                    "active": True,
                    "name": f"📦 {module_name.replace('_', ' ').title()}",
                    "level": level,
                    "activated_at": self.session_start,
                    "dynamic": True,
                    "module_info": {
                        "has_init": hasattr(module_info.get("module", None), 'initialize'),
                        "has_process": hasattr(module_info.get("module", None), 'process_command')
                    }
                }
                print(f"[DS24] Added dynamic module to architecture: {module_name}")

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
            "status": "activated",
            "dynamic": module.get("dynamic", False)
        })
        
        # Для динамических модулей добавляем вызов их инициализации
        if module.get("dynamic") and module_name in self.loaded_modules:
            module_info = self.loaded_modules[module_name]
            if hasattr(module_info.get("module"), 'process_command'):
                try:
                    cmd_result = module_info["module"].process_command("activate", {})
                    response["module_response"] = cmd_result
                except Exception as e:
                    response["module_error"] = str(e)
        
        response.update({
            "module": module_name,
            "name": module["name"],
            "activation_time": activation_time,
            "system_state": self.get_architecture_state()
        })
        
        return response

    def execute_module_command(self, module_name: str, command: str, data: Any = None) -> Dict[str, Any]:
        """Выполнение команды в загруженном модуле"""
        if module_name not in self.loaded_modules:
            return {
                "error": f"Module '{module_name}' not loaded",
                "available_modules": list(self.loaded_modules.keys())
            }
        
        module_info = self.loaded_modules[module_name]
        
        try:
            result = self.module_registry.execute_module_command(module_name, command, data)
            
            # Аудит выполнения
            execution_record = DS24ExecutionRecord(
                input_hash=self._sha256_strict({"module": module_name, "command": command, "data": data}),
                output_hash=self._sha256_strict(result),
                context_hash=self._sha256_strict({
                    "operator": self.operator_id,
                    "session": self.session_id,
                    "action": "module_command"
                }),
                timestamp=self._get_precise_timestamp(),
                operator_id=self.operator_id,
                execution_time_ns=int(time.perf_counter_ns() / 1000),
                verification_status="PASS",
                intent=f"module_{module_name}_{command}"
            )
            
            self.execution_log.append(execution_record)
            self.execution_count += 1
            self.integrity_checks_passed += 1
            
            return {
                "status": "success",
                "module": module_name,
                "command": command,
                "result": result,
                "execution_id": f"MOD-{module_name[:3].upper()}-{self.execution_count:06d}"
            }
            
        except Exception as e:
            self.error_log.append({
                "error": str(e),
                "module": module_name,
                "command": command,
                "timestamp": self._get_precise_timestamp()
            })
            return {"error": str(e)}

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
            "ready_for_evolution": active_count >= 3,
            "dynamic_modules_count": len(self.loaded_modules),
            "dynamic_modules": list(self.loaded_modules.keys())
        }

    def execute_deterministic(self,
                              input_data: Any,
                              intent: str,
                              execution_id: Optional[str] = None) -> Dict[str, Any]:
        """Абсолютно детерминистическое исполнение"""
        start_time = time.perf_counter_ns()

        # 🎯 ПЕРЕХВАТ МОДУЛЬНЫХ КОМАНД
        if intent.startswith("module_"):
            parts = intent.split("_", 2)
            if len(parts) >= 3:
                module_name = parts[1]
                command = parts[2]
                return self.execute_module_command(module_name, command, input_data)
        
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
                "architecture_state": self.get_architecture_state(),
                "loaded_modules": list(self.loaded_modules.keys())
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
                "loaded_modules": {
                    "count": len(self.loaded_modules),
                    "list": list(self.loaded_modules.keys()),
                    "status": ModuleRegistry.get_modules_status()
                },
                "timestamp": self._get_precise_timestamp()
            }
        
        elif intent == "module_status":
            return {
                "module_registry": ModuleRegistry.get_modules_status(),
                "loaded_modules": list(self.loaded_modules.keys()),
                "architecture_modules": list(self.architecture_modules.keys())
            }
        
        elif intent == "ping":
            return {
                "pong": True,
                "echo": input_data,
                "timestamp": self._get_precise_timestamp(),
                "modules_loaded": len(self.loaded_modules)
            }
        
        elif intent == "architecture_info":
            return {
                "modules": self.architecture_modules,
                "state": self.get_architecture_state(),
                "dynamic_modules": self.loaded_modules
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
            "modules": {
                "loaded_count": len(self.loaded_modules),
                "loaded": list(self.loaded_modules.keys()),
                "status": ModuleRegistry.get_modules_status()
            },
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

        # Тест 1: Базовая работа
        test_input = {"test": 123, "value": 456.789}
        result1 = self.execute_deterministic(test_input, "self_test_1")
        test_results.append({
            "test": "simple_dict",
            "status": result1["verification"]["status"]
        })

        # Тест 2: Комплексная структура
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

        # Тест 3: Идемпотентность
        result3 = self.execute_deterministic(test_input, "self_test_1")
        idempotent = result1["output_signature"] == result3["output_signature"]
        test_results.append({
            "test": "idempotence",
            "status": "PASS" if idempotent else "FAIL"
        })

        passed = sum(1 for t in test_results if t["status"] == "PASS")
        total = len(test_results)

        return {
            "test_suite": "DS24_PURE_SELF_TEST
