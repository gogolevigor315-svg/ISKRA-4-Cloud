# sephirotic_engine.py - ИДЕАЛЬНЫЙ ДВИГАТЕЛЬ С РАСПРЕДЕЛЕННОЙ АРХИТЕКТУРОЙ
import asyncio
import importlib
import inspect
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
import statistics
from dataclasses import dataclass, field
from collections import deque, defaultdict
import random
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import aiohttp
from aiohttp import web, WSMsgType

from .sephirot_bus import SephiroticBus
from .sephirot_base import SephiroticNode, NodeStatus, SignalPackage, SignalType


@dataclass
class IntegrationLink:
    """Структура для связи сефирота с модулем ISKRA"""
    sephirot_name: str
    module_name: str
    link_type: str
    active: bool = True
    last_sync: Optional[datetime] = None
    sync_frequency: float = 5.0  # секунды между синхронизациями
    performance_score: float = 1.0  # оценка эффективности связи (0.0-1.0)


@dataclass
class NetworkFatigue:
    """Модель усталости сети"""
    fatigue_level: float = 0.0  # текущий уровень усталости (0.0-1.0)
    recovery_rate: float = 0.1  # скорость восстановления в секунду
    fatigue_threshold: float = 0.8  # порог перехода в режим отдыха
    rest_mode: bool = False
    fatigue_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    signal_volume_window: deque = field(default_factory=lambda: deque(maxlen=60))
    
    def update(self, signal_count: int, time_delta: float = 1.0) -> None:
        """Обновление уровня усталости на основе нагрузки"""
        self.signal_volume_window.append(signal_count)
        
        # Вычисление нагрузки (нормализованная)
        avg_signals = statistics.mean(self.signal_volume_window) if self.signal_volume_window else 0
        load_factor = min(avg_signals / 100.0, 1.0)  # макс 100 сигналов/сек
        
        if self.rest_mode:
            # Восстановление в режиме отдыха
            self.fatigue_level = max(0.0, self.fatigue_level - self.recovery_rate * 2 * time_delta)
            if self.fatigue_level < 0.3:
                self.rest_mode = False
        else:
            # Накопление усталости
            fatigue_increase = load_factor * 0.05 * time_delta
            self.fatigue_level = min(1.0, self.fatigue_level + fatigue_increase)
            
            if self.fatigue_level > self.fatigue_threshold:
                self.rest_mode = True
        
        self.fatigue_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "fatigue": self.fatigue_level,
            "load": load_factor,
            "rest_mode": self.rest_mode
        })


class DistributedHeartbeatManager:
    """Менеджер распределенного сердцебиения через WebSocket"""
    
    def __init__(self, engine: 'SephiroticEngine'):
        self.engine = engine
        self.connections: Set[aiohttp.WebSocketResponse] = set()
        self.node_registry: Dict[str, Dict[str, Any]] = {}  # внешние узлы
        self.session: Optional[aiohttp.ClientSession] = None
        self.heartbeat_web_task: Optional[asyncio.Task] = None
        
    async def initialize(self, host: str = "0.0.0.0", port: int = 8081) -> None:
        """Инициализация WebSocket сервера для распределенного сердцебиения"""
        app = web.Application()
        app.router.add_get('/heartbeat', self.websocket_handler)
        app.router.add_post('/register_node', self.register_node_handler)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        self.session = aiohttp.ClientSession()
        
        print(f"[DISTRIBUTED] WebSocket сервер запущен на {host}:{port}")
    
    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Обработчик WebSocket соединений"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.connections.add(ws)
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self.handle_heartbeat_message(data, ws)
                elif msg.type == WSMsgType.ERROR:
                    print(f"[DISTRIBUTED] WebSocket ошибка: {ws.exception()}")
        finally:
            self.connections.remove(ws)
        
        return ws
    
    async def register_node_handler(self, request: web.Request) -> web.Response:
        """Регистрация внешнего узла"""
        data = await request.json()
        node_id = data.get('node_id')
        node_url = data.get('url')
        
        if node_id and node_url:
            self.node_registry[node_id] = {
                'url': node_url,
                'last_seen': datetime.utcnow().isoformat(),
                'status': 'active',
                'capabilities': data.get('capabilities', [])
            }
            
            # Синхронизация состояния с новым узлом
            await self.sync_with_node(node_id, node_url)
            
            return web.json_response({
                'status': 'registered',
                'node_id': node_id,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return web.json_response({'error': 'invalid_data'}, status=400)
    
    async def handle_heartbeat_message(self, data: Dict[str, Any], ws: aiohttp.WebSocketResponse) -> None:
        """Обработка сообщений сердцебиения"""
        msg_type = data.get('type')
        
        if msg_type == 'heartbeat':
            # Рассылка сердцебиения по всем соединениям
            heartbeat_data = {
                'type': 'heartbeat',
                'source': 'main_engine',
                'timestamp': datetime.utcnow().isoformat(),
                'cycle': self.engine.cycle_counter,
                'resonance': self.engine._calculate_system_coherence(),
                'fatigue': self.engine.network_fatigue.fatigue_level
            }
            
            await self.broadcast_heartbeat(heartbeat_data)
            
        elif msg_type == 'sync_request':
            # Запрос синхронизации состояния
            await self.send_system_state(ws)
    
    async def broadcast_heartbeat(self, data: Dict[str, Any]) -> None:
        """Широковещательная рассылка сердцебиения"""
        message = json.dumps(data)
        
        for ws in list(self.connections):
            try:
                await ws.send_str(message)
            except Exception as e:
                print(f"[DISTRIBUTED] Ошибка отправки сердцебиения: {e}")
                self.connections.remove(ws)
    
    async def send_system_state(self, ws: aiohttp.WebSocketResponse) -> None:
        """Отправка состояния системы"""
        state = {
            'type': 'system_state',
            'nodes': list(self.engine.nodes.keys()),
            'active_connections': sum(len(n.quantum_links) for n in self.engine.nodes.values()),
            'resonance_map': {name: node.resonance for name, node in self.engine.nodes.items()},
            'fatigue': self.engine.network_fatigue.fatigue_level,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        await ws.send_str(json.dumps(state))
    
    async def sync_with_node(self, node_id: str, node_url: str) -> None:
        """Синхронизация с внешним узлом"""
        try:
            async with self.session.post(f"{node_url}/sync", json={
                'nodes': list(self.engine.nodes.keys()),
                'resonance_levels': {name: node.resonance for name, node in self.engine.nodes.items()}
            }) as response:
                if response.status == 200:
                    data = await response.json()
                    print(f"[DISTRIBUTED] Синхронизирован с узлом {node_id}")
        except Exception as e:
            print(f"[DISTRIBUTED] Ошибка синхронизации с {node_id}: {e}")
    
    async def periodic_sync(self) -> None:
        """Периодическая синхронизация со всеми зарегистрированными узлами"""
        while True:
            try:
                for node_id, node_info in list(self.node_registry.items()):
                    # Проверка активности (если не видели более 30 секунд)
                    last_seen = datetime.fromisoformat(node_info['last_seen'])
                    if datetime.utcnow() - last_seen > timedelta(seconds=30):
                        node_info['status'] = 'inactive'
                    else:
                        await self.sync_with_node(node_id, node_info['url'])
                
                await asyncio.sleep(10.0)  # Синхронизация каждые 10 секунд
            except Exception as e:
                print(f"[DISTRIBUTED] Ошибка периодической синхронизации: {e}")
                await asyncio.sleep(5.0)


class SephiroticVisualizer:
    """Визуализатор состояния Древа Жизни"""
    
    def __init__(self, engine: 'SephiroticEngine'):
        self.engine = engine
        self.graph = nx.Graph()
        self.layout_positions = {}
        self._initialize_graph()
    
    def _initialize_graph(self) -> None:
        """Инициализация графа сефирот с позициями для визуализации"""
        # Классические позиции сефирот на Древе Жизни
        sephirot_positions = {
            'Kether': (0, 2),
            'Chokmah': (-1, 1),
            'Binah': (1, 1),
            'Chesed': (-2, 0),
            'Gevurah': (2, 0),
            'Tiferet': (0, 0),
            'Netzach': (-1, -1),
            'Hod': (1, -1),
            'Yesod': (0, -2),
            'Malkuth': (0, -3)
        }
        
        # Добавляем узлы
        for sephirot_name, node in self.engine.nodes.items():
            self.graph.add_node(sephirot_name, 
                               resonance=node.resonance,
                               energy=node.energy,
                               status=node.status.value)
            
            # Устанавливаем позицию если известна
            if sephirot_name in sephirot_positions:
                self.layout_positions[sephirot_name] = sephirot_positions[sephirot_name]
        
        # Добавляем связи (22 пути)
        connections = [
            ('Kether', 'Chokmah'), ('Kether', 'Binah'), ('Kether', 'Tiferet'),
            ('Chokmah', 'Binah'), ('Chokmah', 'Tiferet'), ('Chokmah', 'Chesed'),
            ('Binah', 'Tiferet'), ('Binah', 'Gevurah'),
            ('Chesed', 'Tiferet'), ('Chesed', 'Gevurah'), ('Chesed', 'Netzach'),
            ('Gevurah', 'Tiferet'), ('Gevurah', 'Chesed'), ('Gevurah', 'Hod'),
            ('Tiferet', 'Netzach'), ('Tiferet', 'Hod'), ('Tiferet', 'Yesod'),
            ('Netzach', 'Hod'), ('Netzach', 'Yesod'),
            ('Hod', 'Yesod'),
            ('Yesod', 'Malkuth')
        ]
        
        for source, target in connections:
            if source in self.graph.nodes and target in self.graph.nodes:
                # Вычисляем силу связи на основе резонанса
                source_node = self.engine.nodes.get(source)
                target_node = self.engine.nodes.get(target)
                
                if source_node and target_node:
                    resonance_strength = (source_node.resonance + target_node.resonance) / 2.0
                    has_quantum_link = any(
                        link.target_node == target for link in source_node.quantum_links
                    ) if hasattr(source_node, 'quantum_links') else False
                    
                    self.graph.add_edge(source, target, 
                                       strength=resonance_strength,
                                       quantum=has_quantum_link)
    
    def generate_plotly_figure(self) -> go.Figure:
        """Генерация интерактивного графа Plotly"""
        if not self.graph.nodes:
            return go.Figure()
        
        # Обновляем данные графа
        self._initialize_graph()
        
        # Извлекаем данные узлов
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node_name in self.graph.nodes():
            node = self.engine.nodes.get(node_name)
            if node_name in self.layout_positions:
                x, y = self.layout_positions[node_name]
                node_x.append(x)
                node_y.append(y)
                
                # Информация для tooltip
                resonance = node.resonance if node else 0.0
                energy = node.energy if node else 0.0
                status = node.status.value if node else 'unknown'
                
                node_text.append(
                    f"<b>{node_name}</b><br>"
                    f"Резонанс: {resonance:.2f}<br>"
                    f"Энергия: {energy:.2f}<br>"
                    f"Статус: {status}<br>"
                    f"Циклов: {node.total_signals_processed if node else 0}"
                )
                
                # Цвет по резонансу (зеленый=высокий, красный=низкий)
                node_color.append(f'rgba({int((1-resonance)*255)}, {int(resonance*255)}, 0, 0.8)')
                
                # Размер по энергии
                node_size.append(15 + (node.energy if node else 0.0) * 25)
        
        # Извлекаем данные ребер
        edge_x = []
        edge_y = []
        edge_color = []
        edge_width = []
        
        for edge in self.graph.edges(data=True):
            source, target, data = edge
            
            if source in self.layout_positions and target in self.layout_positions:
                x0, y0 = self.layout_positions[source]
                x1, y1 = self.layout_positions[target]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                # Цвет по типу связи
                if data.get('quantum', False):
                    edge_color.append('rgba(0, 200, 255, 0.7)')  # Голубой для квантовых
                else:
                    edge_color.append('rgba(150, 150, 150, 0.3)')  # Серый для обычных
                
                # Толщина по силе связи
                edge_width.append(1 + data.get('strength', 0.5) * 3)
        
        # Создаем фигуру
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=['Древо Жизни - Сефиротическая Система']
        )
        
        # Добавляем ребра
        for i in range(0, len(edge_x)-1, 3):
            if i+2 < len(edge_x):
                fig.add_trace(go.Scatter(
                    x=[edge_x[i], edge_x[i+1]],
                    y=[edge_y[i], edge_y[i+1]],
                    mode='lines',
                    line=dict(
                        color=edge_color[i//3],
                        width=edge_width[i//3]
                    ),
                    hoverinfo='none',
                    showlegend=False
                ), row=1, col=1)
        
        # Добавляем узлы
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=[name for name in self.graph.nodes()],
            textposition="top center",
            hovertext=node_text,
            hoverinfo='text',
            name='Сефироты'
        ), row=1, col=1)
        
        # Настройка макета
        fig.update_layout(
            title="Сефиротическая Сеть ISKRA-4",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(10, 10, 20, 0.9)',
            paper_bgcolor='rgba(10, 10, 20, 0.9)',
            font=dict(color='white')
        )
        
        return fig
    
    def generate_network_html(self) -> str:
        """Генерация HTML с интерактивным графом"""
        fig = self.generate_plotly_figure()
        return fig.to_html(include_plotlyjs='cdn', full_html=False)


class SephiroticEngine:
    """Движок сефиротической системы - распределенный и адаптивный"""
    
    def __init__(self, config_path: str = "config/sephirot_config.yaml"):
        self.bus = SephiroticBus()
        self.nodes: Dict[str, SephiroticNode] = {}
        self.integrations: List[IntegrationLink] = []
        self.heartbeat_interval = 2.0
        self.running = False
        self.metrics_history = deque(maxlen=5000)
        self.cycle_counter = 0
        self.config_path = config_path
        self.monitor_task: Optional[asyncio.Task] = None
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.sync_task: Optional[asyncio.Task] = None
        
        # Усталость сети
        self.network_fatigue = NetworkFatigue()
        
        # Распределенная архитектура
        self.distributed_manager = DistributedHeartbeatManager(self)
        
        # Визуализация
        self.visualizer = SephiroticVisualizer(self)
        
        # Адаптивные параметры
        self.adaptive_params = {
            'resonance_threshold': 0.65,
            'energy_recovery_rate': 0.05,
            'quantum_link_strength': 0.85,
            'signal_processing_delay': 0.01
        }
        
        # Эволюционная память
        self.evolutionary_memory = {
            'successful_patterns': deque(maxlen=100),
            'failed_patterns': deque(maxlen=100),
            'resonance_peaks': deque(maxlen=50),
            'performance_trend': 1.0
        }
        
        # Загрузка конфигурации
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации из YAML"""
        import yaml
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Конфигурация по умолчанию"""
        return {
            "sephirot": {
                "active_nodes": ["Kether", "Tiferet", "Yesod", "Malkuth"],
                "resonance_threshold": 0.65,
                "auto_connect": True,
                "heartbeat_interval": 2.0,
                "fatigue_recovery_rate": 0.1,
                "fatigue_threshold": 0.8
            },
            "distribution": {
                "enabled": True,
                "websocket_port": 8081,
                "sync_interval": 10.0,
                "max_external_nodes": 10
            },
            "visualization": {
                "enabled": True,
                "update_interval": 5.0,
                "html_output": "templates/sephirot_network.html"
            },
            "evolution": {
                "learning_enabled": True,
                "pattern_memory_size": 100,
                "adaptation_rate": 0.01
            }
        }
    
    async def initialize(self) -> Dict[str, Any]:
        """Полная инициализация движка"""
        print(f"[ENGINE] Инициализация распределенной сефиротической системы...")
        
        # 1. Автоматическое обнаружение и создание узлов
        await self._auto_discover_nodes()
        
        # 2. Создание квантовых связей между узлами
        await self._create_quantum_links()
        
        # 3. Автоматическая интеграция с модулями ISKRA
        await self._auto_integrate_modules()
        
        # 4. Инициализация распределенной архитектуры
        if self.config.get('distribution', {}).get('enabled', True):
            await self.distributed_manager.initialize(
                port=self.config['distribution'].get('websocket_port', 8081)
            )
            self.sync_task = asyncio.create_task(self.distributed_manager.periodic_sync())
        
        # 5. Запуск фоновых задач
        self._start_background_tasks()
        
        # 6. Инициализация визуализации
        if self.config.get('visualization', {}).get('enabled', True):
            await self._initialize_visualization()
        
        return {
            "status": "initialized",
            "nodes_active": len(self.nodes),
            "integrations": len([i for i in self.integrations if i.active]),
            "distribution_enabled": self.config.get('distribution', {}).get('enabled', False),
            "visualization_enabled": self.config.get('visualization', {}).get('enabled', False),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _initialize_visualization(self) -> None:
        """Инициализация системы визуализации"""
        # Создаем начальный граф
        self.visualizer._initialize_graph()
        
        # Сохраняем HTML визуализацию
        html_content = self.visualizer.generate_network_html()
        output_path = self.config['visualization'].get('html_output', 'templates/sephirot_network.html')
        
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"[VISUALIZATION] Граф сохранен в {output_path}")
    
    async def _heartbeat_cycle(self) -> None:
        """Асинхронный цикл сердцебиения системы с учетом усталости"""
        self.running = True
        print(f"[HEARTBEAT] Распределенный цикл сердцебиения запущен")
        
        while self.running:
            try:
                self.cycle_counter += 1
                
                # Учет усталости сети
                if self.network_fatigue.rest_mode:
                    # Режим отдыха - снижаем активность
                    self.heartbeat_interval = 5.0  # Увеличиваем интервал
                    processing_intensity = 0.3  # Снижаем интенсивность обработки
                else:
                    # Нормальный режим
                    self.heartbeat_interval = self.config['sephirot'].get('heartbeat_interval', 2.0)
                    processing_intensity = 1.0 - self.network_fatigue.fatigue_level
                
                # Подсчет сигналов для учета усталости
                signal_count = 0
                
                # 1. Генерация волевого импульса от Kether (с учетом усталости)
                if "Kether" in self.nodes and processing_intensity > 0.5:
                    intention_strength = 0.5 * processing_intensity
                    intention_signal = SignalPackage(
                        source="ENGINE",
                        target="Kether",
                        type=SignalType.INTENTION,
                        payload={
                            "intention": {
                                "strength": intention_strength,
                                "purpose": "system_sustainment",
                                "cycle": self.cycle_counter,
                                "fatigue_adjusted": processing_intensity
                            }
                        }
                    )
                    await self.nodes["Kether"].receive_signal(intention_signal)
                    signal_count += 1
                
                # 2. Адаптивная эмоциональная гармония
                if self.cycle_counter % max(1, int(3 / processing_intensity)) == 0 and "Tiferet" in self.nodes:
                    emotion_signal = SignalPackage(
                        source="ENGINE",
                        target="Tiferet",
                        type=SignalType.EMOTIONAL,
                        payload={
                            "emotion_type": "adaptive_harmony",
                            "intensity": 0.7 * processing_intensity,
                            "fatigue_level": self.network_fatigue.fatigue_level,
                            "adaptation_rate": self.adaptive_params.get('adaptation_rate', 0.01)
                        }
                    )
                    await self.nodes["Tiferet"].receive_signal(emotion_signal)
                    signal_count += 1
                
                # 3. Эволюционное обновление фундамента
                if self.cycle_counter % 5 == 0 and "Yesod" in self.nodes:
                    # Адаптируем параметры на основе истории
                    self._adapt_parameters_from_history()
                    
                    foundation_signal = SignalPackage(
                        source="ENGINE",
                        target="Yesod",
                        type=SignalType.DATA,
                        payload={
                            "key": f"evolutionary_cycle_{self.cycle_counter}",
                            "value": {
                                "cycle": self.cycle_counter,
                                "adaptive_params": self.adaptive_params,
                                "fatigue": self.network_fatigue.fatigue_level,
                                "performance_trend": self.evolutionary_memory['performance_trend']
                            }
                        }
                    )
                    await self.nodes["Yesod"].receive_signal(foundation_signal)
                    signal_count += 1
                
                # 4. Распределенное сердцебиение
                if self.config.get('distribution', {}).get('enabled', False):
                    await self.distributed_manager.broadcast_heartbeat({
                        'type': 'heartbeat',
                        'source': f"engine_{hashlib.md5(str(id(self)).encode()).hexdigest()[:8]}",
                        'timestamp': datetime.utcnow().isoformat(),
                        'cycle': self.cycle_counter,
                        'resonance': await self._calculate_system_coherence(),
                        'fatigue': self.network_fatigue.fatigue_level,
                        'node_count': len(self.nodes)
                    })
                
                # 5. Обновление усталости
                self.network_fatigue.update(signal_count)
                
                # 6. Статус и метрики
                if self.cycle_counter % 10 == 0:
                    await self._log_system_status()
                
                # 7. Сбор метрик с адаптацией
                await self._collect_adaptive_metrics()
                
                # 8. Обновление визуализации
                if self.cycle_counter % 20 == 0 and self.config.get('visualization', {}).get('enabled', True):
                    await self._update_visualization()
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                print("[HEARTBEAT] Цикл сердцебиения остановлен")
                break
            except Exception as e:
                print(f"[HEARTBEAT] Ошибка в цикле: {e}")
                await asyncio.sleep(1)
    
    def _adapt_parameters_from_history(self) -> None:
        """Адаптация параметров на основе исторических данных"""
        if len(self.metrics_history) < 10:
            return
        
        # Анализ последних метрик
        recent_metrics = list(self.metrics_history)[-10:]
        resonances = [m.get("average_resonance", 0) for m in recent_metrics]
        
        if len(resonances) > 1:
            avg_resonance = statistics.mean(resonances)
            resonance_std = statistics.stdev(resonances)
            
            # Адаптация порога резонанса
            if avg_resonance > 0.8 and resonance_std < 0.1:
                # Система стабильна, можно повысить требования
                self.adaptive_params['resonance_threshold'] = min(
                    0.9, self.adaptive_params['resonance_threshold'] + 0.01
                )
            elif avg_resonance < 0.5 or resonance_std > 0.2:
                # Система нестабильна, снижаем требования
                self.adaptive_params['resonance_threshold'] = max(
                    0.4, self.adaptive_params['resonance_threshold'] - 0.02
                )
            
            # Запоминаем пики резонанса
            if avg_resonance > 0.85:
                self.evolutionary_memory['resonance_peaks'].append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'resonance': avg_resonance,
                    'params': self.adaptive_params.copy()
                })
            
            # Обновляем тренд производительности
            if len(self.metrics_history) > 20:
                old_avg = statistics.mean([m.get("average_resonance", 0) 
                                          for m in list(self.metrics_history)[-20:-10]])
                self.evolutionary_memory['performance_trend'] = avg_resonance / max(old_avg, 0.01)
    
    async def _collect_adaptive_metrics(self) -> None:
        """Сбор метрик с адаптивным анализом"""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "cycle": self.cycle_counter,
            "active_nodes": len([n for n in self.nodes.values() if n.status == NodeStatus.ACTIVE]),
            "average_resonance": statistics.mean(
                [n.resonance for n in self.nodes.values()]
            ) if self.nodes else 0,
            "average_energy": statistics.mean(
                [n.energy for n in self.nodes.values()]
            ) if self.nodes else 0,
            "fatigue_level": self.network_fatigue.fatigue_level,
            "rest_mode": self.network_fatigue.rest_mode,
            "adaptive_params": self.adaptive_params.copy(),
            "evolutionary_trend": self.evolutionary_memory['performance_trend'],
            "quantum_connections": sum(
                len(n.quantum_links) for n in self.nodes.values()
            ),
            "distribution_active": len(self.distributed_manager.connections) > 0,
            "visualization_ready": hasattr(self, 'visualizer') and len(self.visualizer.graph.nodes) > 0
        }
        
        self.metrics_history.append(metrics)
        
        # Сохраняем успешные паттерны
        if metrics['average_resonance'] > 0.8 and metrics['fatigue_level'] < 0.3:
            pattern_hash = hashlib.md5(
                json.dumps(metrics, sort_keys=True).encode()
            ).hexdigest()[:16]
            
            self.evolutionary_memory['successful_patterns'].append({
                'hash': pattern_hash,
                'metrics': metrics,
                'timestamp': datetime.utcnow().isoformat()
            })
    
    async def _update_visualization(self) -> None:
        """Обновление визуализации"""
        try:
            html_content = self.visualizer.generate_network_html()
            output_path = self.config['visualization'].get('html_output', 
                                                         'templates/sephirot_network.html')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # Также обновляем в памяти для быстрого доступа
            self.current_visualization_html = html_content
            
        except Exception as e:
            print(f"[VISUALIZATION] Ошибка обновления: {e}")
    
    async def get_visualization_html(self) -> str:
        """Получение HTML визуализации"""
        if hasattr(self, 'current_visualization_html'):
            return self.current_visualization_html
        return self.visualizer.generate_network_html()
    
    async def get_evolutionary_report(self) -> Dict[str, Any]:
        """Отчет об эволюционном развитии системы"""
        successful_patterns = list(self.evolutionary_memory['successful_patterns'])
        resonance_peaks = list(self.evolutionary_memory['resonance_peaks'])
        
        # Анализ лучших паттернов
        best_patterns = []
        if successful_patterns:
            sorted_patterns = sorted(successful_patterns, 
                                   key=lambda x: x['metrics']['average_resonance'], 
                                   reverse=True)[:5]
            
            for pattern in sorted_patterns:
                best_patterns.append({
                    'resonance': pattern['metrics']['average_resonance'],
                    'fatigue': pattern['metrics']['fatigue_level'],
                    'timestamp': pattern['timestamp'],
                    'hash': pattern['hash'][:8]
                })
        
        # Анализ трендов
        trends = {
            'resonance_trend': 'stable',
            'fatigue_trend': 'stable',
            'adaptation_rate': self.adaptive_params.get('adaptation_rate', 0.01)
        }
        
        if len(self.metrics_history) > 10:
            recent = list(self.metrics_history)[-10:]
            old = list(self.metrics_history)[-20:-10] if len(self.metrics_history) > 20 else recent
            
            recent_res = statistics.mean([m['average_resonance'] for m in recent])
            old_res = statistics.mean([m['average_resonance'] for m in old])
            
            if recent_res > old_res * 1.1:
                trends['resonance_trend'] = 'improving'
            elif recent_res < old_res * 0.9:
                trends['resonance_trend'] = 'declining'
        
        return {
            'evolutionary_summary': {
                'successful_patterns_count': len(successful_patterns),
                'resonance_peaks_count': len(resonance_peaks),
                'performance_trend': self.evolutionary_memory['performance_trend'],
                'system_age_cycles': self.cycle_counter
            },
            'best_patterns': best_patterns,
            'adaptive_parameters': self.adaptive_params,
            'trend_analysis': trends,
            'recommendations': self._generate_evolutionary_recommendations(),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _generate_evolutionary_recommendations(self) -> List[str]:
        """Генерация рекомендаций на основе эволюционного анализа"""
        recommendations = []
        
        
