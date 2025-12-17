# sephirotic_engine.py - ГЛАВНЫЙ ДВИГАТЕЛЬ
import time
from sephirot_bus import SephiroticBus
from sephirot_base import SephiroticNode

class SephiroticEngine:
    """Движок сефиротической системы"""
    
    def __init__(self):
        self.bus = SephiroticBus()
        self.nodes = {}
        self.heartbeat_interval = 2.0  # секунды
        self.running = False
        
    def initialize_core_nodes(self):
        """Инициализация 3 ключевых узлов для начала"""
        # 1. Kether - Воля
        from sephirot_blocks.1_keter import KetherCore
        self.nodes["Kether"] = KetherCore(self.bus)
        
        # 2. Tiferet - Гармония (связь с Emotional Weave)
        from sephirot_blocks.6_tiferet import TiferetCore
        self.nodes["Tiferet"] = TiferetCore(self.bus)
        
        # 3. Yesod - Фундамент (связь с ISKR Eco)
        from sephirot_blocks.9_yesod import YesodCore
        self.nodes["Yesod"] = YesodCore(self.bus)
        
        print(f"[ENGINE] Инициализировано ядер: {list(self.nodes.keys())}")
    
    def start_heartbeat(self):
        """Запуск ритмичного сердцебиения системы"""
        self.running = True
        cycle = 0
        
        while self.running:
            cycle += 1
            
            # Цикл 1: Намерение от Kether
            if "Kether" in self.nodes:
                self.bus.transmit("Kether", {
                    "type": "intention",
                    "cycle": cycle,
                    "intensity": 0.5 + (cycle % 10) * 0.05
                })
            
            # Цикл 2: Эмоциональный резонанс (каждые 3 цикла)
            if cycle % 3 == 0 and "Tiferet" in self.nodes:
                self.bus.transmit("Tiferet", {
                    "type": "emotional_resonance",
                    "source": "heartbeat",
                    "value": 0.7
                })
            
            # Статус каждые 10 циклов
            if cycle % 10 == 0:
                state = self.bus.get_network_state()
                print(f"[HEARTBEAT {cycle}] Узлы: {state['nodes_registered']}, Когерентность: {state['system_coherence']:.2f}")
            
            time.sleep(self.heartbeat_interval)
    
    def connect_to_existing_modules(self):
        """Интеграция с существующими модулями ISKRA-4"""
        integrations = []
        
        # Связь Tiferet ↔ Emotional Weave
        try:
            from iskra_modules.emotional_weave import EmotionalWeave
            emotional = EmotionalWeave()
            self.nodes["Tiferet"].set_emotional_link(emotional)
            integrations.append("Tiferet ↔ Emotional Weave")
        except:
            pass
        
        # Связь Yesod ↔ ISKR Eco Core
        try:
            from iskra_modules.iskr_eco_core import ISKREcoCore
            eco = ISKREcoCore()
            self.nodes["Yesod"].set_eco_link(eco)
            integrations.append("Yesod ↔ ISKR Eco Core")
        except:
            pass
        
        # Связь Hod ↔ Polyglossia Adapter
        try:
            from iskra_modules.polyglossia_adapter import PolyglossiaAdapter
            lang = PolyglossiaAdapter()
            if "Hod" in self.nodes:
                self.nodes["Hod"].set_language_link(lang)
                integrations.append("Hod ↔ Polyglossia Adapter")
        except:
            pass
        
        print(f"[ENGINE] Интеграции: {integrations if integrations else 'нет'}")
