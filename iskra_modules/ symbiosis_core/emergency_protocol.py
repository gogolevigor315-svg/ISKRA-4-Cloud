cat > iskra_modules/symbiosis_core/emergency_protocol.py << 'EOF'
"""
EmergencyProtocol для SYMBIOSIS-CORE
Аварийные процедуры и защита от сбоев
"""

import time
import logging
from typing import Dict, Any

class EmergencyProtocol:
    def __init__(self):
        self.active = False
        self.emergency_level = 0  # 0-5: 0=нет, 5=полная остановка
        self.last_triggered = None
        self.error_log = []
        
        # Настройка логирования
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - EMERGENCY - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.WARNING)
    
    def get_status(self) -> Dict[str, Any]:
        """Возвращает статус аварийного протокола"""
        return {
            "active": self.active,
            "emergency_level": self.emergency_level,
            "last_triggered": self.last_triggered,
            "error_count": len(self.error_log),
            "timestamp": time.time()
        }
    
    def handle_error(self, error_type: str, error_msg: str) -> None:
        """Обработка ошибки с классификацией по типу"""
        error_entry = {
            "type": error_type,
            "message": error_msg,
            "timestamp": time.time(),
            "level": self._classify_error(error_type)
        }
        
        self.error_log.append(error_entry)
        self.logger.warning(f"EmergencyProtocol: {error_type} - {error_msg}")
        
        # Повышение уровня аварии при критических ошибках
        if error_type in ["critical", "fatal", "integration_error"]:
            self.emergency_level = min(5, self.emergency_level + 1)
            self.active = True
            self.last_triggered = time.time()
    
    def trigger_emergency_stop(self) -> Dict[str, Any]:
        """Активация полной аварийной остановки"""
        self.active = True
        self.emergency_level = 5
        self.last_triggered = time.time()
        
        self.logger.critical("EMERGENCY STOP ACTIVATED - ALL SYSTEMS IN READONLY MODE")
        
        return {
            "status": "emergency_stop_activated",
            "level": 5,
            "timestamp": self.last_triggered,
            "message": "Все системы переведены в режим только для чтения"
        }
    
    def reset_emergency(self) -> Dict[str, Any]:
        """Сброс аварийного состояния"""
        was_active = self.active
        self.active = False
        self.emergency_level = 0
        
        if was_active:
            self.logger.info("EmergencyProtocol reset to normal mode")
        
        return {
            "status": "reset",
            "was_active": was_active,
            "timestamp": time.time()
        }
    
    def _classify_error(self, error_type: str) -> int:
        """Классификация ошибок по уровням серьёзности"""
        error_levels = {
            "info": 1,
            "warning": 2,
            "error": 3,
            "critical": 4,
            "fatal": 5,
            "integration_error": 4,
            "shadow_error": 3,
            "session_error": 2
        }
        return error_levels.get(error_type.lower(), 3)
EOF
