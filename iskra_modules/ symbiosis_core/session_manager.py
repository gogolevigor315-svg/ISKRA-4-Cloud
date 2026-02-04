import time
import json
import threading
from typing import Dict, Any, Optional, List

class SessionManager:
    def __init__(self):
        self.current_mode = "readonly"
        self.session_start = None
        self.session_id = None
        self.session_data = {}
        self.last_operation = None
        self.shadow_consent = {
            "granted": False,
            "level": 0,
            "expires_at": 0,
            "granted_by": None
        }
        self.operation_log = []
        self.lock = threading.Lock()
        
    def start_session(self, mode: str = "readonly") -> str:
        with self.lock:
            self.current_mode = mode
            self.session_start = time.time()
            self.session_id = f"symbiosis_{int(time.time())}_{hash(mode)}"
            self.session_data = {
                "mode": mode,
                "start_time": self.session_start,
                "operations_count": 0,
                "last_resonance": None,
                "last_energy": None
            }
            return self.session_id
    
    def end_session(self) -> Dict[str, Any]:
        with self.lock:
            duration = time.time() - self.session_start if self.session_start else 0
            session_summary = {
                "session_id": self.session_id,
                "mode": self.current_mode,
                "duration_seconds": duration,
                "operations_count": len(self.operation_log),
                "session_data": self.session_data
            }
            
            # Сброс
            self.session_start = None
            self.session_id = None
            self.session_data = {}
            
            return session_summary
    
    def is_active(self) -> bool:
        return self.session_start is not None
    
    def get_session_duration(self) -> float:
        if self.session_start:
            return time.time() - self.session_start
        return 0
    
    def grant_shadow_consent(self, level: int, duration_seconds: int = 1800) -> Dict[str, Any]:
        with self.lock:
            self.shadow_consent = {
                "granted": True,
                "level": level,
                "expires_at": time.time() + duration_seconds,
                "granted_at": time.time(),
                "granted_by": "operator"
            }
            
            return {
                "status": "consent_granted",
                "level": level,
                "expires_at": self.shadow_consent["expires_at"],
                "duration_seconds": duration_seconds
            }
    
    def revoke_shadow_consent(self) -> Dict[str, Any]:
        with self.lock:
            was_granted = self.shadow_consent["granted"]
            self.shadow_consent = {
                "granted": False,
                "level": 0,
                "expires_at": 0,
                "granted_by": None
            }
            
            return {
                "status": "consent_revoked",
                "was_granted": was_granted,
                "timestamp": time.time()
            }
    
    def has_shadow_consent(self, required_level: int = 0) -> bool:
        with self.lock:
            if not self.shadow_consent["granted"]:
                return False
            
            if time.time() > self.shadow_consent["expires_at"]:
                self.shadow_consent["granted"] = False
                return False
            
            if required_level > self.shadow_consent["level"]:
                return False
            
            return True
    
    def update_session_mode(self, mode: str):
        with self.lock:
            self.current_mode = mode
            if self.session_data:
                self.session_data["mode"] = mode
    
    def log_operation(self, operation_type: str, data: Dict[str, Any]):
        with self.lock:
            operation = {
                "timestamp": time.time(),
                "type": operation_type,
                "data": data,
                "session_id": self.session_id,
                "mode": self.current_mode
            }
            
            self.operation_log.append(operation)
            
            # Ограничение размера лога
            if len(self.operation_log) > 1000:
                self.operation_log = self.operation_log[-1000:]
            
            # Обновление статистики сессии
            if self.session_data:
                self.session_data["operations_count"] = len(self.operation_log)
                self.last_operation = operation
    
    def should_log_operation(self) -> bool:
        # Логируем всегда в advanced и experimental режимах
        return self.current_mode in ["advanced", "experimental"]
    
    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "session_active": self.is_active(),
                "session_id": self.session_id,
                "current_mode": self.current_mode,
                "session_duration": self.get_session_duration(),
                "shadow_consent": self.shadow_consent,
                "operations_log_size": len(self.operation_log),
                "last_operation_time": self.last_operation["timestamp"] if self.last_operation else None,
                "session_data": self.session_data
            }
    
    def get_operation_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.lock:
            return self.operation_log[-limit:] if self.operation_log else []


class ShadowConsentManager:
    """Управление согласием для shadow операций (расширенная версия)"""
    
    def __init__(self):
        self.consent_cache = {}
        self.audit_log = []
        
    def request_consent(self, operation_id: str, shadow_level: int, 
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Запрос согласия на shadow операцию"""
        
        # Автоматическое согласие для низких уровней
        if shadow_level <= 3:
            consent = {
                "granted": True,
                "level": shadow_level,
                "auto_granted": True,
                "reason": "low_shadow_level",
                "expires_at": time.time() + 3600  # 1 час
            }
            
        # Для средних уровней требуется проверка
        elif 4 <= shadow_level <= 6:
            consent = {
                "granted": False,
                "level": shadow_level,
                "requires_review": True,
                "reason": "medium_shadow_level",
                "review_deadline": time.time() + 300  # 5 минут на проверку
            }
            
        # Для высоких уровней требуется явное подтверждение
        else:  # 7-10
            consent = {
                "granted": False,
                "level": shadow_level,
                "requires_explicit_approval": True,
                "reason": "high_shadow_level",
                "operator_notification_required": True
            }
        
        # Сохранение в кеш
        self.consent_cache[operation_id] = {
            **consent,
            "requested_at": time.time(),
            "context": context
        }
        
        # Аудит
        self.audit_log.append({
            "timestamp": time.time(),
            "operation_id": operation_id,
            "action": "consent_requested",
            "shadow_level": shadow_level,
            "consent_granted": consent.get("granted", False)
        })
        
        return consent
    
    def check_consent(self, operation_id: str) -> Dict[str, Any]:
        """Проверка существующего согласия"""
        if operation_id not in self.consent_cache:
            return {
                "granted": False,
                "reason": "consent_not_requested",
                "valid": False
            }
        
        consent = self.consent_cache[operation_id]
        
        # Проверка срока действия
        if consent.get("expires_at", 0) < time.time():
            return {
                "granted": False,
                "reason": "consent_expired",
                "valid": False
            }
        
        return {
            "granted": consent.get("granted", False),
            "level": consent.get("level", 0),
            "reason": consent.get("reason", "unknown"),
            "valid": True,
            "consent_data": consent
        }
