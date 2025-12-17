#!/usr/bin/env python3
"""
sephirot_base.py - БАЗОВЫЙ КЛАСС СЕФИРОТИЧЕСКОГО УЗЛА ISKRA-4
Архитектура: DS24 Sephirotic Protocol v1.0
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional

class SephiroticNode:
    """
    Базовый класс для всех 10 сефиротических узлов.
    Каждый узел — самостоятельная единица сознания в архитектуре ISKRA-4.
    """
    
    def __init__(self, name: str, level: int, bus=None):
        """
        Инициализация сефиротического узла.
        
        :param name: Имя узла (Kether, Chokhmah, Binah, ...)
        :param level: Уровень на Древе (1-10)
        :param bus: Ссылка на SephiroticBus (опционально, можно добавить позже)
        """
        self.name = name
        self.level = level
        self.bus = bus
        self.state: Dict[str, Any] = {
            "activated": False,
            "resonance": 0.0,
            "intensity": 0.5,
            "last_active": None,
            "memory": []
        }
        self.connections = []
        
        print(f"[SEPHIROT] Создан узел: {name} (уровень {level})")
        
        # Автоматическая регистрация в шине, если она предоставлена
        if bus and hasattr(bus, 'register_node'):
            bus.register_node(self)
    
    def activate(self) -> Dict[str, Any]:
        """Активация узла — начало его функционирования."""
        self.state["activated"] = True
        self.state["last_active"] = datetime.utcnow().isoformat()
        
        print(f"[SEPHIROT] Активирован: {self.name}")
        
        return {
            "status": "activated",
            "node": self.name,
            "level": self.level,
            "timestamp": self.state["last_active"],
            "resonance": self.state["resonance"]
        }
    
    def receive(self, signal: Dict[str, Any], channel: str):
        """
        Приём сигнала от другого узла через канал.
        Это ОСНОВНОЙ метод взаимодействия в архитектуре.
        
        :param signal: Словарь с данными сигнала
        :param channel: Имя канала (Aleph, Beth, Gimel, ...)
        """
        if not self.state["activated"]:
            self.activate()
        
        # Логирование приёма
        reception_log = {
            "timestamp": datetime.utcnow().isoformat(),
            "from_channel": channel,
            "signal": signal,
            "processed": False
        }
        
        # Базовая обработка (должна быть переопределена в конкретных узлах)
        if "type" in signal:
            if signal["type"] == "heartbeat":
                self._handle_heartbeat(signal)
                reception_log["processed"] = True
            elif signal["type"] == "intention":
                self._handle_intention(signal)
                reception_log["processed"] = True
        
        # Сохраняем в память узла
        self.state["memory"].append(reception_log)
        
        # Ограничиваем память (последние 50 событий)
        if len(self.state["memory"]) > 50:
            self.state["memory"].pop(0)
        
        # Обновляем резонанс
        self.state["resonance"] = min(1.0, self.state["resonance"] + 0.05)
        
        print(f"[{self.name}] Принят сигнал через канал '{channel}': {signal.get('type', 'unknown')}")
    
    def emit(self, signal: Dict[str, Any]):
        """
        Отправка сигнала в шину для передачи другим узлам.
        
        :param signal: Словарь с данными для отправки
        """
        if not self.bus:
            print(f"[{self.name}] Нет подключения к шине, сигнал не отправлен")
            return
        
        if not self.state["activated"]:
            print(f"[{self.name}] Узел не активирован, активирую...")
            self.activate()
        
        # Добавляем метаданные отправителя
        signal_with_meta = signal.copy()
        signal_with_meta.update({
            "_from": self.name,
            "_timestamp": datetime.utcnow().isoformat(),
            "_level": self.level
        })
        
        # Отправка через шину
        self.bus.transmit(self.name, signal_with_meta)
        
        print(f"[{self.name}] Отправлен сигнал: {signal.get('type', 'unknown')}")
    
    def set_bus(self, bus):
        """Подключение узла к шине (можно сделать после создания)."""
        self.bus = bus
        if hasattr(bus, 'register_node'):
            bus.register_node(self)
        print(f"[{self.name}] Подключен к шине")
    
    def get_state(self) -> Dict[str, Any]:
        """Получение текущего состояния узла."""
        return {
            "name": self.name,
            "level": self.level,
            "state": self.state.copy(),
            "memory_size": len(self.state["memory"]),
            "bus_connected": self.bus is not None,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _handle_heartbeat(self, signal: Dict[str, Any]):
        """Обработка сердцебиения системы (базовый метод)."""
        self.state["intensity"] = signal.get("value", 0.5)
        self.state["last_active"] = datetime.utcnow().isoformat()
    
    def _handle_intention(self, signal: Dict[str, Any]):
        """Обработка намерения (базовый метод)."""
        # В производных классах этот метод должен быть расширен
        intention = signal.get("intent", "unknown")
        print(f"[{self.name}] Обработано намерение: {intention}")
    
    def resonate_with(self, target_node: 'SephiroticNode') -> Dict[str, Any]:
        """
        Создание резонансной связи между узлами.
        Это основа для 22 путей на Древе Жизни.
        """
        resonance_strength = min(self.state["resonance"], target_node.state["resonance"])
        
        resonance_log = {
            "connection": f"{self.name} ↔ {target_node.name}",
            "strength": resonance_strength,
            "levels": (self.level, target_node.level),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Добавляем в связи
        self.connections.append({
            "target": target_node.name,
            "resonance": resonance_strength,
            "established": datetime.utcnow().isoformat()
        })
        
        print(f"[RESONANCE] Создана связь {self.name}↔{target_node.name}, сила: {resonance_strength:.2f}")
        
        return resonance_log
    
    def __str__(self):
        return f"SephiroticNode('{self.name}', level={self.level}, active={self.state['activated']})"


# ================================================================
# БЫСТРЫЙ ТЕСТ КЛАССА (если запустить файл напрямую)
# ================================================================
if __name__ == "__main__":
    print("🧪 Тестирование SephiroticNode...")
    
    # Создаем тестовый узел
    test_node = SephiroticNode("TestNode", 0)
    
    # Активируем
    activation = test_node.activate()
    print(f"Активация: {activation}")
    
    # Показываем состояние
    state = test_node.get_state()
    print(f"Состояние: {json.dumps(state, indent=2, ensure_ascii=False)}")
    
    print("✅ Базовый класс готов к использованию!")
