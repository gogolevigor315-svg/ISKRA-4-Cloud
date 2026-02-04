"""
Адаптер для интеграции SYMBIOSIS с существующей архитектурой ISKRA-4.
Использует sephirot_bus.py для чтения/записи состояний.
"""

import sys
import os

# Добавляем путь к корню проекта для импорта существующих модулей
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

class ISKRAIntegrator:
    def __init__(self):
        self.sephirot_bus = None
        self.api_base_url = "http://localhost:10000"
        
    def connect_to_sephirot_bus(self):
        """Подключение к существующей шине сефирот"""
        try:
            # Пытаемся импортировать существующий sephirot_bus
            from sephirot_bus import get_state, apply_changes
            self.sephirot_bus = {
                'get_state': get_state,
                'apply_changes': apply_changes
            }
            return True
        except ImportError:
            # Fallback: используем REST API ISKRA-4
            import requests
            self.sephirot_bus = None
            return False
    
    def get_sephirot_state(self):
        """Получение состояния всех сефирот"""
        if self.sephirot_bus:
            # Прямой доступ через sephirot_bus
            return self.sephirot_bus['get_state']()
        else:
            # Через REST API
            try:
                import requests
                response = requests.get(f"{self.api_base_url}/sephirot/state", timeout=5)
                return response.json()
            except:
                return {"error": "Cannot connect to ISKRA-4"}
    
    def apply_symbiosis_changes(self, resonance_delta, energy_delta):
        """Применение изменений от SYMBIOSIS"""
        # Жесткие ограничения
        resonance_delta = max(-0.05, min(0.05, resonance_delta))
        energy_delta = max(-50, min(50, energy_delta))
        
        if self.sephirot_bus:
            # Прямое применение через sephirot_bus
            return self.sephirot_bus['apply_changes'](resonance_delta, energy_delta)
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
            except:
                return {"status": "readonly_mode", "applied": False}
