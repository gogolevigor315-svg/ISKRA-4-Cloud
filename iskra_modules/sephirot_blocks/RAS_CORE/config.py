#!/usr/bin/env python3
"""
config.py - ДИНАМИЧЕСКАЯ КОНФИГУРАЦИЯ RAS-CORE v4.1
Версия: 1.0.0
Назначение: Управление runtime-параметрами RAS-CORE с углом 14.4°
"""

import json
import os
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import asyncio
from pathlib import Path

# Импорты из RAS-CORE
from .constants import GOLDEN_STABILITY_ANGLE, calculate_stability_factor

# ============================================================================
# ТИПЫ КОНФИГУРАЦИИ
# ============================================================================

class ConfigSource(Enum):
    """Источник конфигурации"""
    DEFAULT = "default"
    RUNTIME = "runtime"
    FILE = "file"
    API = "api"
    ENV = "environment"

class ConfigPriority(Enum):
    """Приоритет конфигурации (чем выше, тем важнее)"""
    CRITICAL = 100
    HIGH = 75
    NORMAL = 50
    LOW = 25
    DEFAULT = 0

@dataclass
class ConfigChange:
    """Запись изменения конфигурации"""
    timestamp: datetime
    key: str
    old_value: Any
    new_value: Any
    source: ConfigSource
    priority: ConfigPriority
    reason: str = ""
    applied: bool = False
    rollback_possible: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "key": self.key,
            "old_value": str(self.old_value) if not isinstance(self.old_value, (int, float, bool, str)) else self.old_value,
            "new_value": str(self.new_value) if not isinstance(self.new_value, (int, float, bool, str)) else self.new_value,
            "source": self.source.value,
            "priority": self.priority.value,
            "reason": self.reason,
            "applied": self.applied,
            "rollback_possible": self.rollback_possible
        }

# ============================================================================
# ОСНОВНОЙ КЛАСС КОНФИГУРАЦИИ RAS-CORE
# ============================================================================

@dataclass
class RASConfig:
    """
    Динамическая конфигурация RAS-CORE с поддержкой угла 14.4°.
    Все параметры могут изменяться в runtime.
    """
    
    # ================================================================
    # БАЗОВЫЕ НАСТРОЙКИ
    # ================================================================
    
    # Золотой угол устойчивости
    golden_stability_angle: float = GOLDEN_STABILITY_ANGLE
    
    # Циклы и тайминги
    reflection_cycle_ms: int = 144  # 14.4 × 10
    health_check_interval_ms: int = 5000
    metrics_collection_interval_ms: int = 30000
    
    # Очереди
    max_queue_size: int = 1000
    signal_ttl_seconds: float = 30.0
    cleanup_interval_seconds: int = 60
    
    # ================================================================
    # ПОРОГИ И ЛИМИТЫ
    # ================================================================
    
    # Пороги приоритетов
    priority_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "critical": 0.9,
        "high": 0.6,
        "normal": 0.3,
        "low": 0.1
    })
    
    # Пороги стабильности
    stability_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "excellent": 0.85,
        "good": 0.70,
        "warning": 0.50,
        "critical": 0.30
    })
    
    # Лимиты глубины
    max_reflection_depth: int = 10
    max_learning_iterations: int = 1000
    max_pattern_history: int = 100
    
    # ================================================================
    # МАРШРУТИЗАЦИЯ
    # ================================================================
    
    # Цели маршрутизации
    sephirotic_routing: Dict[str, Any] = field(default_factory=lambda: {
        "targets": ["KETER", "CHOKMAH", "DAAT", "BINAH", "YESOD", "TIFERET"],
        "default_target": "DAAT",
        "fallback_target": "YESOD",
        "angle_based_routing": True,
        "min_angle_for_routing": 5.0,
        "max_angle_for_routing": 45.0
    })
    
    # Веса маршрутизации
    routing_weights: Dict[str, float] = field(default_factory=lambda: {
        "neuro_weight": 0.4,
        "semiotic_weight": 0.3,
        "priority_weight": 0.2,
        "stability_weight": 0.1
    })
    
    # ================================================================
    # ОБУЧЕНИЕ И АДАПТАЦИЯ
    # ================================================================
    
    # PatternLearner
    pattern_learning: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "learning_rate": 0.01,
        "exploration_rate": 0.1,
        "forgetting_factor": 0.99,
        "min_samples_for_pattern": 10,
        "pattern_validation_samples": 50
    })
    
    # A/B тестирование
    ab_testing: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": False,
        "test_duration_minutes": 60,
        "min_samples_per_variant": 100,
        "confidence_level": 0.95
    })
    
    # ================================================================
    # ЭНЕРГЕТИЧЕСКИЕ НАСТРОЙКИ
    # ================================================================
    
    energy_management: Dict[str, Any] = field(default_factory=lambda: {
        "energy_aware_routing": False,
        "max_energy_per_signal": 0.5,
        "energy_saving_mode": False,
        "power_scaling_enabled": True,
        "min_power_level": 0.3,
        "max_power_level": 1.0
    })
    
    # ================================================================
    # САМОРЕФЛЕКСИЯ И ЛИЧНОСТЬ
    # ================================================================
    
    self_reflection: Dict[str, Any] = field(default_factory=lambda: {
        "enabled": True,
        "min_coherence_for_reflection": 0.3,
        "max_reflection_time_ms": 1000,
        "external_reality_check_interval": 10,
        "forced_external_focus_threshold": 20
    })
    
    personality: Dict[str, Any] = field(default_factory=lambda: {
        "coherence_threshold": 0.7,
        "stability_window_size": 10,
        "manifestation_check_interval": 30,
        "history_persistence_interval": 60
    })
    
    # ================================================================
    # БЕЗОПАСНОСТЬ И УСТОЙЧИВОСТЬ
    # ================================================================
    
    safety: Dict[str, Any] = field(default_factory=lambda: {
        "circuit_breaker_enabled": True,
        "max_failures_before_break": 5,
        "circuit_breaker_timeout_ms": 5000,
        "rate_limiting_enabled": True,
        "max_requests_per_second": 100,
        "validation_strictness": "medium"  # low, medium, high
    })
    
    # ================================================================
    # МОНИТОРИНГ И ЛОГИРОВАНИЕ
    # ================================================================
    
    monitoring: Dict[str, Any] = field(default_factory=lambda: {
        "metrics_enabled": True,
        "log_level": "INFO",  # DEBUG, INFO, WARNING, ERROR
        "enable_tracing": False,
        "trace_sample_rate": 0.1,
        "dashboard_update_interval_ms": 1000
    })
    
    # ================================================================
    # RUNTIME-ПАРАМЕТРЫ (меняются динамически)
    # ================================================================
    
    runtime: Dict[str, Any] = field(default_factory=lambda: {
        "current_stability_factor": 1.0,
        "angle_adjustment_speed": 0.1,
        "focus_intensity": 0.7,
        "learning_enabled": True,
        "adaptive_mode": True,
        "personality_manifestation_level": 0.0,
        "last_config_update": datetime.utcnow().isoformat()
    })
    
    # ================================================================
    # МЕТОДЫ КОНФИГУРАЦИИ
    # ================================================================
    
    def __post_init__(self):
        """Инициализация после создания dataclass"""
        self._change_history: List[ConfigChange] = []
        self._validation_rules = self._create_validation_rules()
        self._listeners: Dict[str, List[callable]] = {}
        
        # Запись начальной конфигурации
        self._record_change(
            key="__init__",
            old_value=None,
            new_value="initialized",
            source=ConfigSource.DEFAULT,
            priority=ConfigPriority.DEFAULT,
            reason="Initial configuration"
        )
    
    def _create_validation_rules(self) -> Dict[str, callable]:
        """Создание правил валидации для параметров"""
        return {
            "golden_stability_angle": lambda x: 0 <= x <= 90,
            "reflection_cycle_ms": lambda x: 10 <= x <= 10000,
            "max_queue_size": lambda x: x > 0,
            "priority_thresholds.critical": lambda x: 0 <= x <= 1,
            "priority_thresholds.high": lambda x: 0 <= x <= 1,
            "priority_thresholds.normal": lambda x: 0 <= x <= 1,
            "stability_thresholds.excellent": lambda x: 0 <= x <= 1,
            "max_reflection_depth": lambda x: x > 0,
            "routing_weights.neuro_weight": lambda x: 0 <= x <= 1,
            "pattern_learning.learning_rate": lambda x: 0 <= x <= 1,
            "personality.coherence_threshold": lambda x: 0 <= x <= 1,
            "runtime.current_stability_factor": lambda x: 0 <= x <= 1,
        }
    
    def _record_change(self, **kwargs) -> ConfigChange:
        """Запись изменения конфигурации"""
        change = ConfigChange(
            timestamp=datetime.utcnow(),
            **kwargs
        )
        self._change_history.append(change)
        
        # Ограничение истории
        if len(self._change_history) > 1000:
            self._change_history = self._change_history[-1000:]
        
        return change
    
    def validate_value(self, key: str, value: Any) -> bool:
        """Валидация значения для ключа"""
        if key in self._validation_rules:
            return self._validation_rules[key](value)
        
        # Проверка вложенных словарей
        for rule_key, validator in self._validation_rules.items():
            if '.' in rule_key:
                prefix, subkey = rule_key.split('.', 1)
                if key == prefix and isinstance(value, dict) and subkey in value:
                    if not validator(value[subkey]):
                        return False
        
        return True
    
    def update(self, 
               updates: Dict[str, Any], 
               source: ConfigSource = ConfigSource.RUNTIME,
               priority: ConfigPriority = ConfigPriority.NORMAL,
               reason: str = "") -> Dict[str, Any]:
        """
        Обновление конфигурации с валидацией.
        
        Args:
            updates: Словарь с обновлениями
            source: Источник обновления
            priority: Приоритет обновления
            reason: Причина обновления
        
        Returns:
            Dict с результатами обновления
        """
        results = {
            "successful": [],
            "failed": [],
            "skipped": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        for key_path, new_value in updates.items():
            try:
                # Получаем текущее значение
                old_value = self.get_nested(key_path)
                
                # Проверяем валидность
                if not self.validate_value(key_path, new_value):
                    results["failed"].append({
                        "key": key_path,
                        "error": "Validation failed",
                        "old_value": old_value,
                        "new_value": new_value
                    })
                    continue
                
                # Если значение не изменилось - пропускаем
                if old_value == new_value:
                    results["skipped"].append({
                        "key": key_path,
                        "reason": "Value unchanged",
                        "value": old_value
                    })
                    continue
                
                # Применяем изменение
                self.set_nested(key_path, new_value)
                
                # Записываем изменение
                change = self._record_change(
                    key=key_path,
                    old_value=old_value,
                    new_value=new_value,
                    source=source,
                    priority=priority,
                    reason=reason,
                    applied=True
                )
                
                # Обновляем время последнего изменения
                if key_path == "runtime.last_config_update":
                    self.runtime["last_config_update"] = datetime.utcnow().isoformat()
                elif key_path.startswith("runtime."):
                    self.runtime["last_config_update"] = datetime.utcnow().isoformat()
                
                # Уведомляем слушателей
                self._notify_listeners(key_path, old_value, new_value)
                
                results["successful"].append({
                    "key": key_path,
                    "old_value": old_value,
                    "new_value": new_value,
                    "change_id": len(self._change_history) - 1
                })
                
            except Exception as e:
                results["failed"].append({
                    "key": key_path,
                    "error": str(e),
                    "new_value": new_value
                })
        
        return results
    
    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """Получение вложенного значения по пути"""
        keys = key_path.split('.')
        value = self
        
        for key in keys:
            if hasattr(value, key):
                value = getattr(value, key)
            elif isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def set_nested(self, key_path: str, value: Any):
        """Установка вложенного значения по пути"""
        keys = key_path.split('.')
        target = self
        
        # Идем по всем ключам кроме последнего
        for key in keys[:-1]:
            if hasattr(target, key):
                target = getattr(target, key)
            elif isinstance(target, dict):
                if key not in target:
                    target[key] = {}
                target = target[key]
            else:
                raise AttributeError(f"Cannot navigate to {key} in {key_path}")
        
        # Устанавливаем значение
        last_key = keys[-1]
        if hasattr(target, last_key):
            setattr(target, last_key, value)
        elif isinstance(target, dict):
            target[last_key] = value
        else:
            raise AttributeError(f"Cannot set {last_key} in {key_path}")
    
    def adjust_for_stability(self, stability_factor: float):
        """
        Корректировка параметров на основе текущей устойчивости.
        
        Args:
            stability_factor: Фактор устойчивости (0.0-1.0)
        """
        updates = {}
        
        if stability_factor < 0.5:
            # Низкая устойчивость - консервативные настройки
            updates["reflection_cycle_ms"] = 200
            updates["max_reflection_depth"] = 5
            updates["runtime.focus_intensity"] = 0.5
            updates["pattern_learning.learning_rate"] = 0.005
            updates["energy_management.energy_saving_mode"] = True
            
            reason = f"Low stability adjustment (factor: {stability_factor:.2f})"
        elif stability_factor < 0.7:
            # Средняя устойчивость - баланс
            updates["reflection_cycle_ms"] = 144
            updates["max_reflection_depth"] = 8
            updates["runtime.focus_intensity"] = 0.7
            updates["pattern_learning.learning_rate"] = 0.01
            
            reason = f"Medium stability adjustment (factor: {stability_factor:.2f})"
        else:
            # Высокая устойчивость - агрессивные настройки
            updates["reflection_cycle_ms"] = 100
            updates["max_reflection_depth"] = 12
            updates["runtime.focus_intensity"] = 0.9
            updates["pattern_learning.learning_rate"] = 0.02
            updates["ab_testing.enabled"] = True
            
            reason = f"High stability adjustment (factor: {stability_factor:.2f})"
        
        # Обновляем stability factor
        updates["runtime.current_stability_factor"] = stability_factor
        
        # Применяем изменения
        return self.update(
            updates=updates,
            source=ConfigSource.RUNTIME,
            priority=ConfigPriority.HIGH,
            reason=reason
        )
    
    def register_listener(self, key_path: str, callback: callable):
        """Регистрация слушателя изменений"""
        if key_path not in self._listeners:
            self._listeners[key_path] = []
        
        self._listeners[key_path].append(callback)
    
    def _notify_listeners(self, key_path: str, old_value: Any, new_value: Any):
        """Уведомление слушателей об изменении"""
        if key_path in self._listeners:
            for callback in self._listeners[key_path]:
                try:
                    callback(key_path, old_value, new_value)
                except Exception as e:
                    print(f"Error in config listener for {key_path}: {e}")
    
    # ================================================================
    # СЕРИАЛИЗАЦИЯ И СОХРАНЕНИЕ
    # ================================================================
    
    def to_dict(self, include_runtime: bool = True, include_history: bool = False) -> Dict[str, Any]:
        """Экспорт конфигурации в словарь"""
        config_dict = asdict(self)
        
        # Убираем служебные поля
        if '_change_history' in config_dict:
            del config_dict['_change_history']
        if '_validation_rules' in config_dict:
            del config_dict['_validation_rules']
        if '_listeners' in config_dict:
            del config_dict['_listeners']
        
        # Добавляем историю если нужно
        if include_history:
            config_dict['change_history'] = [change.to_dict() for change in self._change_history[-100:]]
        
        # Убираем runtime если не нужно
        if not include_runtime and 'runtime' in config_dict:
            del config_dict['runtime']
        
        return config_dict
    
    def to_json(self, **kwargs) -> str:
        """Экспорт конфигурации в JSON"""
        return json.dumps(self.to_dict(**kwargs), indent=2, default=str)
    
    def to_yaml(self, **kwargs) -> str:
        """Экспорт конфигурации в YAML"""
        return yaml.dump(self.to_dict(**kwargs), default_flow_style=False)
    
    def save_to_file(self, filepath: Union[str, Path], format: str = "json"):
        """Сохранение конфигурации в файл"""
        filepath = Path(filepath)
        
        if format.lower() == "json":
            content = self.to_json(include_runtime=True, include_history=True)
        elif format.lower() == "yaml":
            content = self.to_yaml(include_runtime=True, include_history=True)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        filepath.write_text(content, encoding='utf-8')
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RASConfig':
        """Создание конфигурации из словаря"""
        config = cls()
        
        # Рекурсивное обновление из словаря
        def update_from_dict(target, source):
            for key, value in source.items():
                if isinstance(value, dict) and hasattr(target, key) and isinstance(getattr(target, key), dict):
                    update_from_dict(getattr(target, key), value)
                elif hasattr(target, key):
                    setattr(target, key, value)
                elif isinstance(target, dict):
                    target[key] = value
        
        update_from_dict(config, data)
        
        # Обновляем время последнего изменения
        config.runtime["last_config_update"] = datetime.utcnow().isoformat()
        
        return config
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> 'RASConfig':
        """Загрузка конфигурации из файла"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        content = filepath.read_text(encoding='utf-8')
        
        if filepath.suffix.lower() in ['.json', '.jsonc']:
            data = json.loads(content)
        elif filepath.suffix.lower() in ['.yaml', '.yml']:
            data = yaml.safe_load(content)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return cls.from_dict(data)
    
    # ================================================================
    # СТАТИСТИКА И МОНИТОРИНГ
    # ================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики конфигурации"""
        return {
            "total_changes": len(self._change_history),
            "last_change": self._change_history[-1].to_dict() if self._change_history else None,
            "active_listeners": sum(len(listeners) for listeners in self._listeners.values()),
            "validation_rules_count": len(self._validation_rules),
            "stability_factor": self.runtime.get("current_stability_factor", 0.0),
            "golden_angle": self.golden_stability_angle,
            "personality_coherence_threshold": self.personality.get("coherence_threshold", 0.7),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def get_change_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Получение истории изменений"""
        history = self._change_history[-limit:] if self._change_history else []
        return [change.to_dict() for change in history]

# ============================================================================
# ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ И МЕНЕДЖЕР
# ============================================================================

class ConfigManager:
    """Менеджер глобальной конфигурации RAS-CORE"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Инициализация менеджера"""
        self._config = RASConfig()
        self._config_file = None
        self._auto_save = False
        self._save_interval = 300  # 5 минут
        self._save_task = None
    
    @property
    def config(self) -> RASConfig:
        """Получение текущей конфигурации"""
        return self._config
    
    def setup_auto_save(self, filepath: Union[str, Path], interval_seconds: int = 300):
        """Настройка автоматического сохранения"""
        self._config_file = Path(filepath)
        self._auto_save = True
        self._save_interval = interval_seconds
        
        # Запуск задачи автосохранения
        if self._save_task is None or self._save_task.done():
            self._save_task = asyncio.create_task(self._auto_save_task())
    
    async def _auto_save_task(self):
        """Задача автоматического сохранения"""
        while self._auto_save:
            try:
                await asyncio.sleep(self._save_interval)
                if self._config_file:
                    self._config.save_to_file(self._config_file, format="json")
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Auto-save error: {e}")
    
    def stop_auto_save(self):
        """Остановка автоматического сохранения"""
        self._auto_save = False
        if self._save_task and not self._save_task.done():
            self._save_task.cancel()
    
    def save_config(self, filepath: Union[str, Path] = None, format: str = "json"):
        """Сохранение конфигурации"""
        if filepath is None and self._config_file is None:
            raise ValueError("No filepath specified for saving config")
        
        save_path = Path(filepath) if filepath else self._config_file
        self._config.save_to_file(save_path, format)
    
    def load_config(self, filepath: Union[str, Path]):
        """Загрузка конфигурации из файла"""
        self._config = RASConfig.from_file(filepath)
        self._config_file = Path(filepath)

# ============================================================================
# ГЛОБАЛЬНЫЕ ФУНКЦИИ ДЛЯ ИМПОРТА
# ============================================================================

# Глобальный менеджер конфигурации
_config_manager = ConfigManager()

def get_config() -> RASConfig:
    """Получение глобальной конфигурации"""
    return _config_manager.config

def update_config(updates: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Обновление глобальной конфигурации"""
    return _config_manager.config.update(updates, **kwargs)

def save_config(filepath: Union[str, Path] = None, **kwargs):
    """Сохранение конфигурации"""
    _config_manager.save_config(filepath, **kwargs)

def load_config(filepath: Union[str, Path]):
    """Загрузка конфигурации из файла"""
    _config_manager.load_config(filepath)

def setup_auto_save(filepath: Union[str, Path], **kwargs):
    """Настройка автоматического сохранения"""
    _config_manager.setup_auto_save(filepath, **kwargs)

# ============================================================================
# ТЕСТИРОВАНИЕ
# ============================================================================

async def test_config():
    """Тестирование конфигурации"""
    print("🧪 Тестирование RASConfig...")
    
    # Создаем конфигурацию
    config = RASConfig()
    
    print(f"✅ Конфигурация создана")
    print(f"   Золотой угол: {config.golden_stability_angle}°")
    print(f"   Цикл рефлексии: {config.reflection_cycle_ms} мс")
    print(f"   Порог личности: {config.personality.get('coherence_threshold', 0.7)}")
    
    # Тестируем обновление
    print("\n🔄 Тестирование обновления...")
    updates = {
        "reflection_cycle_ms": 200,
        "runtime.focus_intensity": 0.8,
        "personality.coherence_threshold": 0.75
    }
    
    result = config.update(updates, reason="Тестовое обновление")
    print(f"   Успешно: {len(result['successful'])}")
    print(f"   Неудачно: {len(result['failed'])}")
    
    # Проверяем изменения
    print(f"   Новый цикл: {config.reflection_cycle_ms} мс")
    print(f"   Новый порог личности: {config.personality.get('coherence_threshold', 0.7)}")
    
    # Тестируем адаптацию к стабильности
    print("\n📐 Тестирование адаптации к стабильности...")
    
    # Низкая стабильность
    print("   Низкая стабильность (0.3):")
    result_low = config.adjust_for_stability(0.3)
    print(f"     Цикл: {config.reflection_cycle_ms} мс")
    print(f"     Интенсивность фокуса: {config.runtime.get('focus_intensity', 0.0)}")
    
    # Высокая стабильность
    print("   Высокая стабильность (0.9):")
    result_high = config.adjust_for_stability(0.9)
    print(f"     Цикл: {config.reflection_cycle_ms} мс")
    print(f"     Интенсивность фокуса: {config.runtime.get('focus_intensity', 0.0)}")
    
    # Тестируем сериализацию
    print("\n💾 Тестирование сериализации...")
    config_dict = config.to_dict()
    print(f"   Размер конфигурации: {len(str(config_dict))} символов")
    print(f"   Ключей: {len(config_dict)}")
    
    # История изменений
    print("\n📜 История изменений:")
    history = config.get_change_history(3)
    for change in history:
        print(f"   {change['key']}: {change['old_value']} → {change['new_value']}")
    
    print("\n✅ Тестирование завершено")
    return config

# ============================================================================
# ТОЧКА ВХОДА
# ============================================================================

if __name__ == "__main__":
    import asyncio
    
    print("\n" + "=" * 60)
    print("🚀 ЗАПУСК ТЕСТА КОНФИГУРАЦИИ RAS-CORE")
    print(f"   Версия: 1.0.0")
    print(f"   Золотой угол: {GOLDEN_STABILITY_ANGLE}°")
    print("=" * 60 + "\n")
    
    config = asyncio.run(test_config())
    
    print("\n" + "=" * 60)
    print("📋 ИТОГИ ТЕСТИРОВАНИЯ:")
    print(f"   Конфигурация готова к использованию")
    print(f"   Поддерживает динамические изменения")
    print(f"   Интегрирован угол {GOLDEN_STABILITY_ANGLE}°")
    print("=" * 60)
