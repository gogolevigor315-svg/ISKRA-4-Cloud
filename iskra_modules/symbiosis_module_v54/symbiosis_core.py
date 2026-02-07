# iskra_modules/symbiosis_core/symbiosis_core.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISKRA-4 · SYMBIOSIS-CORE v5.4 (АДАПТИРОВАННАЯ ВЕРСИЯ)
Полная интеграция с ISKRA-4 архитектурой
"""

import json
import time
import threading
import pathlib
import traceback
from typing import Any, Dict, List, Optional, Tuple

# ============================ КЛАССЫ БЕЗОПАСНОСТИ ============================ #

class EmergencyProtocol:
    def __init__(self) -> None:
        self.rollback_count = 0
        self.consecutive_errors = 0
        self.life_cvar = 0

    def handle_event(self, event: str, context: Dict[str, Any]) -> List[str]:
        recs: List[str] = []
        
        if event == "resonance_low":
            recs.append("immediate_rollback")
            recs.append("pause_all_operations")
            self.rollback_count += 1
            
        elif event == "energy_low":
            recs.append("suspend_operations_1h")
            
        elif event == "shadow_too_high":
            recs.append("isolate_module")
            
        elif event == "errors_high":
            recs.append("safe_mode")
            
        else:
            recs.append(f"review_required:{event}")
            
        return recs

    def check_emergency(self, resonance: float, energy: float, shadow_level: int, error_count: int) -> List[str]:
        actions = []
        
        if resonance < 0.95:
            actions.extend(self.handle_event("resonance_low", {"resonance": resonance}))
            
        if energy < 800:
            actions.extend(self.handle_event("energy_low", {"energy": energy}))
            
        if shadow_level > 8:
            actions.extend(self.handle_event("shadow_too_high", {"shadow_level": shadow_level}))
            
        if error_count >= 3:
            actions.extend(self.handle_event("errors_high", {"errors": error_count}))
            
        return actions


class CrisisProtocol:
    def __init__(self) -> None:
        self.protocols: Dict[str, Dict[str, Any]] = {
            "system_collapse": {
                "steps": [
                    "dump_full_state",
                    "isolate_core",
                    "activate_minimal_mode",
                ]
            },
            "daat_breach": {
                "steps": [
                    "block_daat_access",
                    "rollback_to_safe_state",
                ]
            },
        }

    def evaluate(self, resonance: float, energy: float, shadow_level: int) -> List[str]:
        actions: List[str] = []
        
        if resonance < 0.90:
            actions.extend(self.protocols["system_collapse"]["steps"])
            
        if shadow_level == 10:
            actions.extend(self.protocols["daat_breach"]["steps"])
            
        return actions


class RollbackProtocol:
    def __init__(self, threshold: float = 0.15) -> None:
        self.threshold = threshold
        self.rollback_map: Dict[str, List[str]] = {
            "apply_resonance_adjustment": ["revert_resonance_change"],
            "apply_energy_adjustment": ["revert_energy_change"],
            "shadow_operation": ["cancel_shadow_session"],
        }

    def evaluate(self, prev_sym: float, new_sym: float, prev_actions: List[str]) -> Dict[str, Any]:
        if prev_sym is None or not prev_actions:
            return {"rollback_needed": False, "delta": 0.0, "plan": []}
            
        delta = new_sym - prev_sym
        
        if delta < -self.threshold:
            plan: List[str] = []
            for action in prev_actions:
                if action in self.rollback_map:
                    plan.extend(self.rollback_map[action])
                    
            return {
                "rollback_needed": True,
                "delta": delta,
                "actions": prev_actions,
                "plan": plan
            }
            
        return {"rollback_needed": False, "delta": delta, "plan": []}


class ShadowConsentManager:
    def __init__(self, ttl_sec: float = 30 * 60.0) -> None:
        self.ttl_sec = ttl_sec
        self._cache: Dict[str, Any] = {
            "granted": False,
            "expires_at": 0.0,
        }

    def _cache_valid(self) -> bool:
        return bool(self._cache.get("granted") and 
                   self._cache.get("expires_at", 0.0) > time.time())

    def _set_cache_granted(self) -> None:
        self._cache["granted"] = True
        self._cache["expires_at"] = time.time() + self.ttl_sec

    def _clear_cache(self) -> None:
        self._cache["granted"] = False
        self._cache["expires_at"] = 0.0

    def check_consent(self, shadow_level: int, session_mode: str) -> Tuple[bool, List[str]]:
        recs: List[str] = []
        
        if self._cache_valid():
            return True, recs

        # Автоматическое согласие для низких уровней
        if shadow_level <= 3:
            self._set_cache_granted()
            return True, recs
            
        # Для уровней 4-6 требуется проверка
        if 4 <= shadow_level <= 6:
            recs.append("shadow_consent_required")
            recs.append("operator_confirmation_needed")
            return False, recs
            
        # Уровни 7+ запрещены без ручного подтверждения
        if shadow_level >= 7:
            recs.append("critical_shadow_level")
            recs.append("manual_operator_approval_required")
            return False, recs
            
        return False, ["unknown_shadow_level"]


# ============================ ОСНОВНОЙ КЛАСС ============================ #

class SymbiosisCore:
    def __init__(self):
        self.version = "5.4-iskra"
        
        # ЛИМИТЫ В КОДЕ (ISKRA-4 СТИЛЬ)
        self.limits = {
            "max_resonance_delta": 0.05,
            "max_energy_delta": 50,
            "min_resonance": 0.9,
            "min_energy": 700,
            "shadow_consent_threshold": 4,
            "emergency_resonance": 0.95,
            "emergency_energy": 800,
            "max_shadow_level": 8
        }
        
        # Режимы работы (как в ISKRA-4)
        self.session_mode = "readonly"  # readonly, balanced, advanced, experimental
        self.consecutive_errors = 0
        self.last_backup = None
        self.symbiosis_score = 0.0
        self.shadow_level = 0
        self.life_cvar = 0
        self.rollback_count = 0
        
        # ИНТЕГРАЦИЯ С ISKRA-4 АРХИТЕКТУРОЙ
        # Используем iskra_integration.py вместо прямых HTTP запросов
        from .iskra_integration import ISKRAAdapter
        self.iskra_adapter = ISKRAAdapter(use_bus=True)
        
        # История состояний
        self.state_history: List[Dict[str, Any]] = []
        self.recommendation_history: List[Dict[str, Any]] = []
        
        # Инициализация компонентов безопасности (без файловых флагов)
        self.emergency = EmergencyProtocol()
        self.crisis = CrisisProtocol()
        self.rollback = RollbackProtocol()
        self.shadow_consent = ShadowConsentManager()
        
        # Блокировка для потокобезопасности
        self.lock = threading.Lock()
        
        print(f"[SYMBIOSIS-CORE v{self.version}] Инициализирован")
        print(f"  Режим: {self.session_mode}")
        print(f"  Лимиты: резонанс ±{self.limits['max_resonance_delta']}, энергия ±{self.limits['max_energy_delta']}")

    # ======================= ИНТЕГРАЦИЯ С ISKRA-4 ======================= #

    def get_iskra_state(self) -> Dict[str, Any]:
        """Получение состояния ISKRA-4 через адаптер"""
        return self.iskra_adapter.get_sephirot_state()

    def apply_to_iskra(self, recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Применение рекомендаций к ISKRA-4 через адаптер"""
        if self.session_mode == "readonly":
            return {"status": "readonly_mode", "applied": False}
            
        # Проверка shadow consent
        shadow_level = recommendations.get("shadow_level", 0)
        if shadow_level >= self.limits["shadow_consent_threshold"]:
            consent_ok, _ = self.shadow_consent.check_consent(shadow_level, self.session_mode)
            if not consent_ok:
                return {"status": "shadow_consent_required", "applied": False}
        
        # Получаем дельты с ограничениями
        resonance_delta = recommendations.get("resonance_delta", 0.0)
        energy_delta = recommendations.get("energy_delta", 0.0)
        
        # Применяем лимиты
        resonance_delta = max(-self.limits["max_resonance_delta"],
                            min(self.limits["max_resonance_delta"], resonance_delta))
        energy_delta = max(-self.limits["max_energy_delta"],
                         min(self.limits["max_energy_delta"], energy_delta))
        
        # Применяем через адаптер ISKRA-4
        result = self.iskra_adapter.apply_symbiosis_delta(resonance_delta, energy_delta)
        
        # Добавляем информацию о режиме
        result["session_mode"] = self.session_mode
        result["limits_applied"] = True
        
        return result

    # ======================= ВЫЧИСЛЕНИЕ СИМБИОЗА ======================= #

    def compute_symbiosis(self) -> Dict[str, Any]:
        """Основной метод вычисления симбиоза для ISKRA-4"""
        with self.lock:
            try:
                # 1. Получаем состояние ISKRA-4 через адаптер
                iskra_state = self.get_iskra_state()
                
                # 2. Извлекаем ключевые метрики
                resonance = iskra_state.get("average_resonance", 1.0)
                energy = iskra_state.get("total_energy", 1000.0)
                activated = iskra_state.get("activated", True)
                
                # 3. Анализ теневых паттернов
                shadow_analysis = self._analyze_shadow_patterns(resonance, energy)
                self.shadow_level = shadow_analysis["level"]
                self.life_cvar = shadow_analysis["risk"]
                
                # 4. Вычисление симбиоз-скора
                self.symbiosis_score = self._calculate_symbiosis_score(
                    resonance, energy, shadow_analysis
                )
                
                # 5. Проверка аварийных условий
                emergency_actions = self.emergency.check_emergency(
                    resonance, energy, self.shadow_level, self.consecutive_errors
                )
                
                crisis_actions = self.crisis.evaluate(resonance, energy, self.shadow_level)
                
                # 6. Генерация рекомендаций с ограничениями
                recommendations = self._generate_recommendations(
                    resonance, energy, self.symbiosis_score, shadow_analysis
                )
                
                # 7. Проверка необходимости отката
                prev_state = self.state_history[-1] if self.state_history else None
                prev_sym = prev_state.get("symbiosis_score") if prev_state else None
                prev_actions = prev_state.get("actions") if prev_state else []
                
                rollback_info = self.rollback.evaluate(
                    prev_sym, self.symbiosis_score, prev_actions
                )
                
                if rollback_info["rollback_needed"]:
                    recommendations["actions"].extend(rollback_info["plan"])
                    self.rollback_count += 1
                
                # 8. Формирование полного состояния
                state = {
                    "timestamp": time.time(),
                    "version": self.version,
                    "session_mode": self.session_mode,
                    
                    "iskra_state": {
                        "resonance": resonance,
                        "energy": energy,
                        "activated": activated,
                        "shadow_level": self.shadow_level
                    },
                    
                    "symbiosis_metrics": {
                        "score": self.symbiosis_score,
                        "shadow_level": self.shadow_level,
                        "life_cvar": self.life_cvar,
                        "rollback_count": self.rollback_count
                    },
                    
                    "recommendations": recommendations,
                    "emergency_actions": emergency_actions,
                    "crisis_actions": crisis_actions,
                    "rollback_info": rollback_info,
                    
                    "limits": self.limits,
                    "consecutive_errors": self.consecutive_errors
                }
                
                # 9. Сохранение в историю
                self.state_history.append(state)
                if len(self.state_history) > 1000:
                    self.state_history = self.state_history[-1000:]
                
                # 10. Сброс счетчика ошибок при успехе
                self.consecutive_errors = 0
                
                return state
                
            except Exception as e:
                self.consecutive_errors += 1
                self._log_error(e)
                
                return {
                    "timestamp": time.time(),
                    "error": str(e),
                    "consecutive_errors": self.consecutive_errors,
                    "status": "error"
                }

    # ======================= ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ======================= #

    def _analyze_shadow_patterns(self, resonance: float, energy: float) -> Dict[str, Any]:
        """Анализ теневых паттернов"""
        # Базовый анализ стабильности
        if resonance < 0.95:
            level = min(10, int((0.95 - resonance) * 100))
            risk = min(100, int((0.95 - resonance) * 200))
        elif energy < 800:
            level = min(10, int((800 - energy) / 20))
            risk = min(100, int((800 - energy) / 2))
        else:
            level = 0
            risk = 0
            
        return {
            "level": level,
            "risk": risk,
            "stable": resonance >= 0.95 and energy >= 800,
            "resonance_status": "low" if resonance < 0.95 else "optimal",
            "energy_status": "low" if energy < 800 else "optimal"
        }

    def _calculate_symbiosis_score(self, resonance: float, energy: float, 
                                  shadow_analysis: Dict[str, Any]) -> float:
        """Вычисление оценки симбиоза (0.0-1.0)"""
        resonance_score = max(0.0, min(1.0, resonance))
        energy_score = max(0.0, min(1.0, energy / 1000.0))
        shadow_score = max(0.0, min(1.0, 1.0 - (shadow_analysis.get("level", 0) / 10)))
        
        # Взвешенная сумма с учетом режима
        if self.session_mode == "readonly":
            weights = (0.4, 0.3, 0.3)  # Больший вес стабильности
        else:
            weights = (0.3, 0.3, 0.4)  # Больший вес shadow-анализа
            
        score = (resonance_score * weights[0] + 
                energy_score * weights[1] + 
                shadow_score * weights[2])
        
        return round(score, 3)

    def _generate_recommendations(self, resonance: float, energy: float, 
                                 symbiosis_score: float, shadow_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Генерация безопасных рекомендаций"""
        actions = []
        warnings = []
        
        # Анализ состояния
        if resonance < 0.95:
            warnings.append("resonance_below_optimal")
            resonance_delta = min(self.limits["max_resonance_delta"], 1.0 - resonance)
        elif resonance > 1.05:
            warnings.append("resonance_above_optimal")
            resonance_delta = max(-self.limits["max_resonance_delta"], 1.0 - resonance)
        else:
            resonance_delta = 0.0
            
        if energy < 800:
            warnings.append("energy_below_optimal")
            energy_delta = min(self.limits["max_energy_delta"], 1000 - energy)
        elif energy > 1000:
            warnings.append("energy_above_max")
            energy_delta = max(-self.limits["max_energy_delta"], 1000 - energy)
        else:
            energy_delta = 0.0
        
        # Генерация действий по режимам
        if self.session_mode == "readonly":
            actions = ["monitor_only", "log_metrics"]
            resonance_delta = 0.0
            energy_delta = 0.0
            
        elif self.session_mode == "balanced":
            if resonance_delta != 0 or energy_delta != 0:
                actions.append("apply_micro_adjustments")
                
        elif self.session_mode == "advanced":
            if resonance_delta != 0 or energy_delta != 0:
                actions.append("apply_balanced_adjustments")
                if shadow_analysis.get("level", 0) >= self.limits["shadow_consent_threshold"]:
                    actions.append("require_shadow_consent")
                    
        elif self.session_mode == "experimental":
            actions.append("apply_full_adjustments")
            if shadow_analysis.get("level", 0) >= self.limits["shadow_consent_threshold"]:
                actions.append("shadow_operations_allowed")
        
        # Проверка симбиоз-скора
        if symbiosis_score < 0.4:
            actions.append("symbiosis_review_needed")
            warnings.append("low_symbiosis_score")
        elif symbiosis_score > 0.8:
            actions.append("symbiosis_pattern_good")
        
        return {
            "resonance_delta": resonance_delta,
            "energy_delta": energy_delta,
            "actions": actions,
            "warnings": warnings,
            "shadow_level": shadow_analysis.get("level", 0)
        }

    def _log_error(self, error: Exception) -> None:
        """Логирование ошибок в память (ISKRA-4 стиль)"""
        error_record = {
            "timestamp": time.time(),
            "error": str(error),
            "consecutive_errors": self.consecutive_errors,
            "session_mode": self.session_mode
        }
        
        # Сохраняем в историю ошибок
        if not hasattr(self, 'error_history'):
            self.error_history = []
        
        self.error_history.append(error_record)
        if len(self.error_history) > 100:
            self.error_history = self.error_history[-100:]

    # ======================= ИНТЕГРАЦИОННЫЕ МЕТОДЫ ======================= #

    def integrate_to_iskra(self) -> Dict[str, Any]:
        """
        Основной метод интеграции с ISKRA-4.
        Вызывается из основного цикла ISKRA-4.
        """
        try:
            # 1. Вычисляем симбиоз
            state = self.compute_symbiosis()
            
            if "error" in state:
                return state
            
            # 2. Получаем рекомендации
            recommendations = state["recommendations"]
            
            # 3. В режиме readonly только мониторинг
            if self.session_mode == "readonly":
                return {
                    "status": "monitoring",
                    "symbiosis_score": state["symbiosis_metrics"]["score"],
                    "shadow_level": state["symbiosis_metrics"]["shadow_level"],
                    "resonance_delta": 0.0,
                    "energy_delta": 0.0,
                    "actions": ["monitor_only"],
                    "session_mode": self.session_mode
                }
            
            # 4. Проверка shadow consent для высоких уровней
            shadow_level = recommendations["shadow_level"]
            if shadow_level >= 7:
                return {
                    "status": "shadow_critical",
                    "symbiosis_score": state["symbiosis_metrics"]["score"],
                    "shadow_level": shadow_level,
                    "resonance_delta": 0.0,
                    "energy_delta": 0.0,
                    "actions": ["critical_shadow_level"],
                    "requires_operator_approval": True
                }
            
            # 5. Возвращаем безопасные рекомендации
            return {
                "status": "recommendations_ready",
                "symbiosis_score": state["symbiosis_metrics"]["score"],
                "shadow_level": shadow_level,
                "resonance_delta": recommendations["resonance_delta"],
                "energy_delta": recommendations["energy_delta"],
                "actions": recommendations["actions"],
                "warnings": recommendations["warnings"],
                "session_mode": self.session_mode
            }
            
        except Exception as e:
            self.consecutive_errors += 1
            return {
                "status": "integration_error",
                "error": str(e),
                "consecutive_errors": self.consecutive_errors
            }

    def get_status(self) -> Dict[str, Any]:
        """Получение текущего статуса модуля"""
        return {
            "version": self.version,
            "session_mode": self.session_mode,
            "symbiosis_score": self.symbiosis_score,
            "shadow_level": self.shadow_level,
            "life_cvar": self.life_cvar,
            "rollback_count": self.rollback_count,
            "consecutive_errors": self.consecutive_errors,
            "last_backup": self.last_backup,
            "limits": self.limits,
            "history_size": len(self.state_history),
            "iskra_adapter_connected": self.iskra_adapter.bus_available,
            "error_history_size": len(getattr(self, 'error_history', []))
        }

    def set_session_mode(self, mode: str) -> bool:
        """Установка режима работы"""
        allowed_modes = ["readonly", "balanced", "advanced", "experimental"]
        
        if mode in allowed_modes:
            self.session_mode = mode
            return True
            
        return False

    def update_limits(self, new_limits: Dict[str, Any]) -> Dict[str, Any]:
        """Обновление лимитов через API (ISKRA-4 стиль)"""
        updated = []
        
        for key, value in new_limits.items():
            if key in self.limits:
                old_value = self.limits[key]
                self.limits[key] = value
                updated.append({
                    "parameter": key,
                    "old_value": old_value,
                    "new_value": value
                })
        
        return {
            "status": "limits_updated",
            "updated_parameters": updated,
            "total_limits": self.limits
        }

    def backup_state(self) -> Dict[str, Any]:
        """Создание резервной копии состояния"""
        backup = {
            "timestamp": time.time(),
            "state_history": self.state_history[-100:] if self.state_history else [],
            "symbiosis_score": self.symbiosis_score,
            "shadow_level": self.shadow_level,
            "session_mode": self.session_mode,
            "rollback_count": self.rollback_count
        }
        
        self.last_backup = time.time()
        return backup

# ============================ ТОЧКА ВХОДА ============================ #

if __name__ == "__main__":
    # Тестирование модуля
    symbiosis = SymbiosisCore()
    print(f"[SYMBIOSIS-CORE v{symbiosis.version}] Инициализирован")
    print(f"  Режим: {symbiosis.session_mode}")
    print(f"  Лимиты: резонанс ±{symbiosis.limits['max_resonance_delta']}, энергия ±{symbiosis.limits['max_energy_delta']}")
    
    # Тестовый запуск
    result = symbiosis.integrate_to_iskra()
    print(f"Результат интеграции: {result.get('status', 'unknown')}")
