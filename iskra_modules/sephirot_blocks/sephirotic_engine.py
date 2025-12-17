# helpers/quantum_link_validator.py
from datetime import datetime, timedelta
from typing import Any


class QuantumLinkValidator:
    """Валидатор квантовых связей"""
    
    def __init__(self, inactive_threshold: int = 50):
        self.inactive_threshold = inactive_threshold
        self.validation_history = {}
    
    async def validate_link(self, link: Any) -> bool:
        """Валидация квантовой связи"""
        if not hasattr(link, 'last_activity'):
            return False
        
        # Проверка времени последней активности
        if hasattr(link.last_activity, 'timestamp'):
            last_active = link.last_activity.timestamp
        else:
            last_active = getattr(link, 'last_active_cycle', 0)
        
        # Расчет неактивности
        current_time = datetime.utcnow().timestamp()
        time_since_active = current_time - last_active
        
        # Если есть счетчик циклов
        if hasattr(link, 'cycles_since_activity'):
            cycles_inactive = link.cycles_since_activity
        else:
            cycles_inactive = int(time_since_active / 2)  # Примерная оценка
        
        # Связь считается активной, если была активна недавно
        is_active = cycles_inactive < self.inactive_threshold
        
        # Запись истории валидации
        link_id = id(link)
        self.validation_history[link_id] = {
            'last_validated': datetime.utcnow().isoformat(),
            'is_active': is_active,
            'cycles_inactive': cycles_inactive
        }
        
        return is_active
    
    def get_inactive_links(self, links: list) -> list:
        """Получение списка неактивных связей"""
        inactive = []
        
        for link in links:
            if
