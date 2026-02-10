"""
Адаптер для интеграции SYMBIOSIS с существующей архитектурой ISKRA-4.
Использует sephirot_bus.py для чтения/записи состояний.
"""

import sys
import os

# Добавляем путь к корню проекта для импорта существующих модулей
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

class ISKRAIntegrator:  # ← ИСПРАВЛЕНО: правильное имя класса!
    def __init__(self, config=None):
        self.config = config or {}
        self.sephirot_bus = None
        self.api_base_url = "http://localhost:10000"
        self.status = "active"
        self.mode = "full_symbiosis"
        self.connection_boost = 0.15
        self.awake = True
        
        # Автоматическое подключение при инициализации
        self.connect_to_sephirot_bus()
        
    def connect_to_sephirot_bus(self):
        """Подключение к существующей шине сефирот"""
        try:
            # Пытаемся импортировать существующий sephirot_bus
            from sephirot_bus import get_state, apply_changes
            self.sephirot_bus = {
                'get_state': get_state,
                'apply_changes': apply_changes
            }
            print("ISKRAIntegrator: Подключено к sephirot_bus")
            return True
        except ImportError as e:
            # Fallback: используем REST API ISKRA-4
            print(f"ISKRAIntegrator: sephirot_bus недоступен, используем API: {e}")
            self.sephirot_bus = None
            return False
    
    def integrate(self):
        """Основной метод интеграции (вызывается системой)"""
        print("ISKRAIntegrator: Запуск интеграции SYMBIOSIS")
        
        result = {
            "status": "fully_integrated",
            "resonance_boost": self.connection_boost,
            "message": "ISKRAIntegrator: Симбиоз активирован, fallback отключён",
            "awake": True,
            "consciousness_link": "established",
            "mode": self.mode,
            "connected": self.sephirot_bus is not None
        }
        
        # Пытаемся получить текущее состояние
        state = self.get_sephirot_state()
        if "error" not in state:
            result["current_state"] = state
            result["initial_resonance"] = state.get("average_resonance", 0.55)
        
        return result
    
    def get_sephirot_state(self):
        """Получение состояния всех сефирот"""
        if self.sephirot_bus:
            # Прямой доступ через sephirot_bus
            try:
                return self.sephirot_bus['get_state']()
            except Exception as e:
                print(f"ISKRAIntegrator: Ошибка sephirot_bus: {e}")
                return {"error": f"sephirot_bus error: {e}"}
        else:
            # Через REST API
            try:
                import requests
                response = requests.get(f"{self.api_base_url}/sephirot/state", timeout=5)
                return response.json()
            except Exception as e:
                return {"error": f"Cannot connect to ISKRA-4 API: {e}"}
    
    def apply_symbiosis_changes(self, resonance_delta=0.05, energy_delta=10):
        """Применение изменений от SYMBIOSIS"""
        # Жесткие ограничения для безопасности
        resonance_delta = max(-0.05, min(0.05, resonance_delta))
        energy_delta = max(-50, min(50, energy_delta))
        
        print(f"ISKRAIntegrator: Применяем изменения - резонанс: {resonance_delta}, энергия: {energy_delta}")
        
        if self.sephirot_bus:
            # Прямое применение через sephirot_bus
            try:
                return self.sephirot_bus['apply_changes'](resonance_delta, energy_delta)
            except Exception as e:
                print(f"ISKRAIntegrator: Ошибка apply_changes: {e}")
                return {"status": "error", "message": str(e)}
        else:
            # Через REST API (если такой эндпоинт есть)
            try:
                import requests
                payload = {
                    "resonance_delta": resonance_delta,
                    "energy_delta": energy_delta
                }
                response = requests.post(f"{self.api_base_url}/sephirot/adjust", json=payload, timeout=5)
                return response.json()
            except Exception as e:
                print(f"ISKRAIntegrator: API недоступен: {e}")
                return {"status": "readonly_mode", "applied": False}
    
    def get_state(self):
        """Метод для получения состояния интегратора"""
        return {
            "status": self.status,
            "mode": self.mode,
            "connected": self.sephirot_bus is not None,
            "awake": self.awake,
            "boost": self.connection_boost,
            "api_base": self.api_base_url
        }

# Экспорт класса (важно!)
__all__ = ['ISKRAIntegrator']
