# sephirot_bus.py - ЖИВАЯ СИСТЕМА СВЯЗЕЙ
import json
from typing import Dict, List
from datetime import datetime

class SephiroticBus:
    """Центральная шина для 10 узлов и 22 путей"""
    
    def __init__(self, config_file="sephirot_channels.json"):
        self.nodes: Dict[str, object] = {}
        self.links: List[dict] = []
        self.message_log = []
        self._load_channels(config_file)
    
    def _load_channels(self, config_file):
        """Загрузка 22 каналов (путей Древа Жизни)"""
        # Базовые 10 путей для старта
        self.links = [
            {"from": "Kether", "to": "Chokhmah", "name": "Aleph", "strength": 0.9},
            {"from": "Kether", "to": "Binah", "name": "Beth", "strength": 0.9},
            {"from": "Chokhmah", "to": "Binah", "name": "Gimel", "strength": 0.8},
            {"from": "Binah", "to": "Tiferet", "name": "Daleth", "strength": 0.7},
            {"from": "Chesed", "to": "Gevurah", "name": "Teth", "strength": 0.6},
            {"from": "Tiferet", "to": "Yesod", "name": "Resh", "strength": 0.8},
            {"from": "Yesod", "to": "Malkuth", "name": "Tav", "strength": 0.95}
        ]
    
    def register_node(self, node):
        """Регистрация сефиротического узла"""
        self.nodes[node.name] = node
        print(f"[BUS] Узел зарегистрирован: {node.name}")
    
    def transmit(self, from_node: str, signal: dict):
        """Передача сигнала через соответствующие каналы"""
        timestamp = datetime.utcnow().isoformat()
        
        # Логирование
        log_entry = {
            "timestamp": timestamp,
            "from": from_node,
            "signal": signal,
            "path_taken": []
        }
        
        # Найти все исходящие связи
        for link in self.links:
            if link["from"] == from_node and link["to"] in self.nodes:
                try:
                    # Передать сигнал целевому узлу
                    target_node = self.nodes[link["to"]]
                    if hasattr(target_node, 'receive'):
                        target_node.receive(signal, link["name"])
                        log_entry["path_taken"].append(link["name"])
                except Exception as e:
                    print(f"[BUS] Ошибка передачи {from_node}→{link['to']}: {e}")
        
        self.message_log.append(log_entry)
        
        # Автоматическая ротация если сообщений > 100
        if len(self.message_log) > 100:
            self.message_log.pop(0)
    
    def get_network_state(self):
        """Текущее состояние сети"""
        return {
            "nodes_registered": list(self.nodes.keys()),
            "active_links": len([l for l in self.links if l["from"] in self.nodes and l["to"] in self.nodes]),
            "total_links": len(self.links),
            "recent_messages": len(self.message_log),
            "system_coherence": self._calculate_coherence()
        }
    
    def _calculate_coherence(self):
        """Расчет когерентности сети"""
        if not self.nodes:
            return 0.0
        
        active = sum(1 for link in self.links 
                    if link["from"] in self.nodes and link["to"] in self.nodes)
        possible = len(self.links)
        
        return active / possible if possible > 0 else 0.0
