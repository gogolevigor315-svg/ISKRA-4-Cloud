# sephirotic_engine.py - ПОЛНЫЙ ИДЕАЛЬНЫЙ ДВИГАТЕЛЬ (ВСЕ В ОДНОМ ФАЙЛЕ)
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


# ============================================================================
# ВСПОМОГАТЕЛЬНЫЕ КЛАССЫ (ВСТРОЕННЫЕ)
# ============================================================================

@dataclass
class IntegrationLink:
    """Структура для связи сефирота с модулем ISKRA"""
    sephirot_name: str
    module_name: str
    link_type: str
    active: bool = True
    last_sync: Optional[datetime] = None
    sync_frequency: float = 5.0
    performance_score: float = 1.0


@dataclass
class NetworkFatigue:
    """Модель усталости сети"""
    fatigue_level: float = 0.0
    recovery_rate: float = 0.1
    fatigue_threshold: float = 0.8
    rest_mode: bool = False
    fatigue_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    signal_volume_window: deque = field(default_factory=lambda: deque(maxlen=60))
    
    def update(self, signal_count: int, time_delta: float = 1.0) -> None:
        """Обновление уровня усталости"""
        self.signal_volume_window.append(signal_count)
        avg_signals = statistics.mean(self.signal_volume_window) if self.signal_volume_window else 0
        load_factor = min(avg_signals / 100.0, 1.0)
        
        if self.rest_mode:
            self.fatigue_level = max(0.0, self.fatigue_level - self.recovery_rate * 2 * time_delta)
            if self.fatigue_level < 0.3:
                self.rest_mode = False
        else:
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


class QuantumLinkValidator:
    """Валидатор квантовых связей с автоочисткой"""
    
    def __init__(self, inactive_threshold: int = 50):
        self.inactive_threshold = inactive_threshold
        self.validation_history: Dict[int, Dict[str, Any]] = {}
        self.cleaned_links: Set[Tuple[str, str]] = set()
    
    async def validate_link(self, link: Any, source_node: str = "unknown") -> Tuple[bool, str]:
        """Валидация квантовой связи с определением причины"""
        if not hasattr(link, 'target_node'):
            return False, "no_target_node_attribute"
        
        target_node = link.target_node
        link_key = (source_node, target_node)
        
        # Проверка на дубликат (уже очищенная связь)
        if link_key in self.cleaned_links:
            return False, "already_cleaned"
        
        # Проверка времени последней активности
        if hasattr(link, 'last_activity_timestamp'):
            last_active = datetime.fromisoformat(link.last_activity_timestamp)
            inactive_seconds = (datetime.utcnow() - last_active).total_seconds()
            cycles_inactive = int(inactive_seconds / 2)  # Пример: 1 цикл = 2 секунды
        elif hasattr(link, 'cycles_since_activity'):
            cycles_inactive = link.cycles_since_activity
        else:
            cycles_inactive = self.inactive_threshold + 1  # Помечаем как неактивную
        
        # Проверка резонанса связи
        if hasattr(link, 'resonance_strength'):
            resonance_ok = link.resonance_strength > 0.1
        else:
            resonance_ok = True
        
        is_active = (cycles_inactive < self.inactive_threshold) and resonance_ok
        
        # Запись в историю
        link_id = id(link)
        self.validation_history[link_id] = {
            'source': source_node,
            'target': target_node,
            'last_validated': datetime.utcnow().isoformat(),
            'is_active': is_active,
            'cycles_inactive': cycles_inactive,
            'reason': "active" if is_active else f"inactive_for_{cycles_inactive}_cycles"
        }
        
        if not is_active:
            self.cleaned_links.add(link_key)
        
        return is_active, self.validation_history[link_id]['reason']
    
    def get_inactive_links_report(self) -> List[Dict[str, Any]]:
        """Отчет о неактивных связях"""
        return [
            {
                'link_id': link_id,
                **info
            }
            for link_id, info in self.validation_history.items()
            if not info['is_active']
        ]


class SephiroticVisualizer:
    """Визуализатор Древа Жизни с автообновлением"""
    
    def __init__(self, engine: 'SephiroticEngine'):
        self.engine = engine
        self.graph = nx.Graph()
        self.layout_positions = {}
        self.last_update = None
        self.html_cache = None
    
    async def initialize(self) -> bool:
        """Инициализация визуализации"""
        try:
            self._create_base_graph()
            self.last_update = datetime.utcnow()
            self.html_cache = self.generate_html()
            return True
        except Exception as e:
            print(f"[VISUALIZER] Ошибка инициализации: {e}")
            return False
    
    def _create_base_graph(self) -> None:
        """Создание базового графа с позициями сефирот"""
        # Классические позиции Древа Жизни
        positions = {
            'Kether': (0, 2),      # Корона
            'Chokmah': (-1, 1),    # Мудрость
            'Binah': (1, 1),       # Понимание
            'Chesed': (-2, 0),     # Милосердие
            'Gevurah': (2, 0),     # Строгость
            'Tiferet': (0, 0),     # Красота/Гармония
            'Netzach': (-1, -1),   # Победа
            'Hod': (1, -1),        # Слава
            'Yesod': (0, -2),      # Основание
            'Malkuth': (0, -3)     # Царство
        }
        
        # Добавление существующих узлов
        for node_name in self.engine.nodes:
            self.graph.add_node(node_name, type='sephirot', resonance=0.5, energy=0.5)
            if node_name in positions:
                self.layout_positions[node_name] = positions[node_name]
            else:
                # Автоматическое позиционирование для новых узлов
                x = random.uniform(-2, 2)
                y = random.uniform(-2, 2)
                self.layout_positions[node_name] = (x, y)
        
        # Добавление связей (22 пути)
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
                self.graph.add_edge(source, target, type='sephirotic_path', strength=0.5)
    
    async def update(self) -> bool:
        """Обновление визуализации"""
        try:
            # Обновление данных узлов
            for node_name, node in self.engine.nodes.items():
                if node_name in self.graph.nodes:
                    self.graph.nodes[node_name]['resonance'] = node.resonance
                    self.graph.nodes[node_name]['energy'] = node.energy
                    self.graph.nodes[node_name]['status'] = node.status.value
            
            # Обновление связей
            for source, target in list(self.graph.edges()):
                if source in self.engine.nodes and target in self.engine.nodes:
                    source_node = self.engine.nodes[source]
                    if hasattr(source_node, 'quantum_links'):
                        has_link = any(link.target_node == target for link in source_node.quantum_links)
                        self.graph.edges[source, target]['quantum'] = has_link
            
            # Обновление HTML
            self.html_cache = self.generate_html()
            self.last_update = datetime.utcnow()
            return True
        except Exception as e:
            print(f"[VISUALIZER] Ошибка обновления: {e}")
            return False
    
    def generate_html(self) -> str:
        """Генерация HTML с интерактивным графом"""
        if not self.graph.nodes:
            return "<div style='padding: 20px; color: #ccc;'>Граф не инициализирован</div>"
        
        # Подготовка данных узлов
        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        
        for node_name, (x, y) in self.layout_positions.items():
            if node_name in self.graph.nodes:
                node_x.append(x)
                node_y.append(y)
                
                node_data = self.graph.nodes[node_name]
                resonance = node_data.get('resonance', 0)
                energy = node_data.get('energy', 0)
                status = node_data.get('status', 'unknown')
                
                # Tooltip с детальной информацией
                node_text.append(
                    f"<b>{node_name}</b><br>"
                    f"Резонанс: {resonance:.2f}<br>"
                    f"Энергия: {energy:.2f}<br>"
                    f"Статус: {status}<br>"
                    f"Связей: {self.graph.degree(node_name)}"
                )
                
                # Цвет по резонансу (градиент от красного к зеленому)
                r = int((1 - resonance) * 255)
                g = int(resonance * 255)
                node_color.append(f'rgba({r}, {g}, 50, 0.8)')
                
                # Размер по энергии
                node_size.append(15 + energy * 25)
        
        # Подготовка данных связей
        edge_x, edge_y, edge_colors, edge_widths = [], [], [], []
        
        for edge in self.graph.edges(data=True):
            source, target, data = edge
            if source in self.layout_positions and target in self.layout_positions:
                x0, y0 = self.layout_positions[source]
                x1, y1 = self.layout_positions[target]
                
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
                # Цвет и толщина по типу связи
                if data.get('quantum', False):
                    edge_colors.append('rgba(0, 200, 255, 0.7)')  # Голубой для квантовых
                    edge_widths.append(3)
                else:
                    edge_colors.append('rgba(150, 150, 150, 0.3)')  # Серый для обычных
                    edge_widths.append(1 + data.get('strength', 0.5) * 2)
        
        # Создание фигуры Plotly
        fig = go.Figure()
        
        # Добавление связей
        for i in range(0, len(edge_x)-1, 3):
            if i+2 < len(edge_x):
                fig.add_trace(go.Scatter(
                    x=[edge_x[i], edge_x[i+1]],
                    y=[edge_y[i], edge_y[i+1]],
                    mode='lines',
                    line=dict(
                        color=edge_colors[i//3],
                        width=edge_widths[i//3]
                    ),
                    hoverinfo='none',
                    showlegend=False
                ))
        
        # Добавление узлов
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            text=[name for name in self.layout_positions.keys()],
            textposition="top center",
            hovertext=node_text,
            hoverinfo='text',
            name='Сефироты'
        ))
        
        # Настройка макета
        fig.update_layout(
            title=dict(
                text="🌳 Древо Жизни - Сефиротическая Система ISKRA-4",
                font=dict(size=20, color='white')
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=60),
            xaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-3, 3]
            ),
            yaxis=dict(
                showgrid=False,
                zeroline=False,
                showticklabels=False,
                range=[-4, 3]
            ),
            plot_bgcolor='rgba(10, 10, 30, 0.95)',
            paper_bgcolor='rgba(10, 10, 30, 0.95)',
            font=dict(color='white', size=12)
        )
        
        return fig.to_html(
            include_plotlyjs='cdn',
            full_html=False,
            config={'responsive': True}
        )
    
    def save_to_file(self, filename: str = "templates/sephirot_network.html") -> bool:
        """Сохранение визуализации в файл"""
        try:
            import os
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(self.html_cache or self.generate_html())
            
            return True
        except Exception as e:
            print(f"[VISUALIZER] Ошибка сохранения: {e}")
            return False


class DistributedHeartbeatManager:
    """Менеджер распределенного сердцебиения с защитой от дубликатов"""
    
    def __init__(self, engine: 'SephiroticEngine'):
        self.engine = engine
        self.connections: Set[aiohttp.WebSocketResponse] = set()
        self.node_registry: Dict[str, Dict[str, Any]] = {}  # внешние узлы
        self.duplicate_check: Dict[str, List[datetime]] = defaultdict(list)
        self.session: Optional[aiohttp.ClientSession] = None
        self.runner: Optional[web.AppRunner] = None
        self.register_handler = None
    
    async def initialize(self, host: str = "0.0.0.0", port: int = 8081) -> bool:
        """Инициализация WebSocket сервера"""
        try:
            app = web.Application()
            app.router.add_get('/heartbeat', self.websocket_handler)
            app.router.add_post('/register_node', self.handle_register_node)
            
            self.runner = web.AppRunner(app)
            await self.runner.setup()
            
            site = web.TCPSite(self.runner, host, port)
            await site.start()
            
            self.session = aiohttp.ClientSession()
            
            print(f"[DISTRIBUTED] WebSocket сервер запущен на {host}:{port}")
            return True
            
        except Exception as e:
            print(f"[DISTRIBUTED] Ошибка инициализации: {e}")
            return False
    
    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Обработчик WebSocket соединений с проверкой активности"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        client_ip = request.remote or "unknown"
        
        # Проверка на флуд (максимум 10 подключений в минуту с одного IP)
        now = datetime.utcnow()
        recent_connections = [
            dt for dt in self.duplicate_check.get(client_ip, [])
            if (now - dt).total_seconds() < 60
        ]
        
        if len(recent_connections) >= 10:
            print(f"[DISTRIBUTED] Блокировка флуда с IP: {client_ip}")
            await ws.close()
            return ws
        
        self.duplicate_check[client_ip].append(now)
        self.connections.add(ws)
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self.handle_heartbeat_message(data, ws)
                    except json.JSONDecodeError:
                        print(f"[DISTRIBUTED] Невалидный JSON от {client_ip}")
                        
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                    break
                    
        except Exception as e:
            print(f"[DISTRIBUTED] Ошибка в соединении: {e}")
            
        finally:
            self.connections.remove(ws)
            # Очистка старых записей
            if client_ip in self.duplicate_check:
                self.duplicate_check[client_ip] = [
                    dt for dt in self.duplicate_check[client_ip]
                    if (now - dt).total_seconds() < 300  # Храним 5 минут
                ]
        
        return ws
    
    async def handle_register_node(self, request: web.Request) -> web.Response:
        """Обработчик регистрации узлов с защитой от дубликатов"""
        try:
            data = await request.json()
            node_id = data.get('node_id')
            node_url = data.get('url')
            
            if not node_id or not node_url:
                return web.json_response(
                    {"error": "missing_required_fields", "fields": ["node_id", "url"]},
                    status=400
                )
            
            # Если есть кастомный обработчик
            if self.register_handler:
                return web.json_response(await self.register_handler(data))
            
            # Стандартная логика регистрации
            return await self._register_node_standard(node_id, node_url, data)
            
        except json.JSONDecodeError:
            return web.json_response({"error": "invalid_json"}, status=400)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def _register_node_standard(self, node_id: str, node_url: str, data: Dict[str, Any]) -> web.Response:
        """Стандартная логика регистрации узла"""
        now = datetime.utcnow()
        
        # Проверка на дубликат (по ID и URL)
        if node_id in self.node_registry:
            existing = self.node_registry[node_id]
            
            # Если тот же URL - обновляем время
            if existing.get('url') == node_url:
                existing['last_seen'] = now.isoformat()
                existing['registration_count'] = existing.get('registration_count', 0) + 1
                
                return web.json_response({
                    "status": "updated",
                    "node_id": node_id,
                    "message": "Узел уже зарегистрирован, время обновлено",
                    "timestamp": now.isoformat()
                })
            else:
                # Разные URL с одинаковым ID - конфликт
                return web.json_response({
                    "error": "node_id_conflict",
                    "message": f"Узел с ID '{node_id}' уже зарегистрирован с другого URL",
                    "existing_url": existing.get('url'),
                    "new_url": node_url
                }, status=409)
        
        # Регистрация нового узла
        self.node_registry[node_id] = {
            "url": node_url,
            "registered": now.isoformat(),
            "last_seen": now.isoformat(),
            "capabilities": data.get('capabilities', []),
            "metadata": data.get('metadata', {}),
            "registration_count": 1
        }
        
        print(f"[DISTRIBUTED] Зарегистрирован новый узел: {node_id} ({node_url})")
        
        return web.json_response({
            "status": "registered",
            "node_id": node_id,
            "message": "Узел успешно зарегистрирован",
            "timestamp": now.isoformat(),
            "assigned_id": node_id
        })
    
    async def handle_heartbeat_message(self, data: Dict[str, Any], ws: aiohttp.WebSocketResponse) -> None:
        """Обработка сообщений сердцебиения"""
        msg_type = data.get('type')
        
        if msg_type == 'heartbeat':
            # Рассылка сердцебиения всем подключенным клиентам
            heartbeat_data = {
                'type': 'heartbeat',
                'source': 'main_engine',
                'timestamp': datetime.utcnow().isoformat(),
                'cycle': self.engine.cycle_counter,
                'resonance': await self.engine._calculate_system_coherence(),
                'fatigue': self.engine.network_fatigue.fatigue_level,
                'active_nodes': len(self.engine.nodes)
            }
            
            await self.broadcast_heartbeat(heartbeat_data)
            
        elif msg_type == 'sync_request':
            # Отправка состояния системы
            await self.send_system_state(ws)
    
    async def broadcast_heartbeat(self, data: Dict[str, Any]) -> None:
        """Широковещательная рассылка сердцебиения"""
        message = json.dumps(data)
        closed_connections = []
        
        for ws in list(self.connections):
            try:
                if not ws.closed:
                    await ws.send_str(message)
                else:
                    closed_connections.append(ws)
            except Exception as e:
                print(f"[DISTRIBUTED] Ошибка отправки: {e}")
                closed_connections.append(ws)
        
        # Очистка закрытых соединений
        for ws in closed_connections:
            if ws in self.connections:
                self.connections.remove(ws)
    
    async def send_system_state(self, ws: aiohttp.WebSocketResponse) -> None:
        """Отправка состояния системы"""
        try:
            state = {
                'type': 'system_state',
                'nodes': list(self.engine.nodes.keys()),
                'active_count': len([n for n in self.engine.nodes.values() 
                                   if n.status == NodeStatus.ACTIVE]),
                'resonance_map': {name: node.resonance 
                                 for name, node in self.engine.nodes.items()},
                'fatigue': self.engine.network_fatigue.fatigue_level,
                'external_nodes': len(self.node_registry),
                'timestamp': datetime.utcnow().isoformat()
            }
            
            await ws.send_str(json.dumps(state))
        except Exception as e:
            print(f"[DISTRIBUTED] Ошибка отправки состояния: {e}")
    
    async def cleanup_inactive_nodes(self, max_inactive_minutes: int = 30) -> List[str]:
        """Очистка неактивных внешних узлов"""
        now = datetime.utcnow()
        inactive_nodes = []
        
        for node_id, node_info in list(self.node_registry.items()):
            last_seen = datetime.fromisoformat(node_info['last_seen'])
            inactive_minutes = (now - last_seen).total_seconds() / 60
            
            if inactive_minutes > max_inactive_minutes:
                inactive_nodes.append(node_id)
                del self.node_registry[node_id]
        
        if inactive_nodes:
            print(f"[DISTRIBUTED] Очищены неактивные узлы: {inactive_nodes}")
        
        return inactive_nodes
    
    async def shutdown(self) -> None:
        """Корректное завершение работы"""
        # Закрытие соединений
        for ws in list(self.connections):
            try:
                await ws.close()
            except:
                pass
        
        self.connections.clear()
        
        # Закрытие сессии
        if self.session:
            await self.session.close()
        
        # Остановка сервера
        if self.runner:
            await self.runner.cleanup()


class IntegrationManager:
    """Менеджер интеграций с модулями ISKRA"""
    
    def __init__(self):
        self.integrations: List[IntegrationLink] = []
        self.integration_cache: Dict[str, Any] = {}
    
    async def initialize(self, engine: 'SephiroticEngine') -> List[str]:
        """Инициализация всех интеграций"""
        initialized = []
        
        # Карта интеграций: сефирот -> (модуль, класс, тип_связи)
        integration_map = {
            "Tiferet": ("emotional_weave", "EmotionalWeave", "emotional"),
            "Hod": ("polyglossia_adapter", "PolyglossiaAdapter", "language"),
            "Yesod": ("iskr_eco_core", "ISKREcoCore", "eco")
        }
        
        for sephirot_name, (module_name, class_name, link_type) in integration_map.items():
            if sephirot_name in engine.nodes:
                try:
                    # Динамический импорт
                    module = importlib.import_module(f"iskra_modules.{module_name}")
                    module_class = getattr(module, class_name)
                    module_instance = module_class()
                    
                    # Установка связи
                    if hasattr(engine.nodes[sephirot_name], f"set_{link_type}_link"):
                        getattr(engine.nodes[sephirot_name], f"set_{link_type}_link")(module_instance)
                        
                        # Создание записи об интеграции
                        link = IntegrationLink(
                            sephirot_name=sephirot_name,
                            module_name=class_name,
                            link_type=link_type
                        )
                        self.integrations.append(link)
                        
                        # Кэширование
                        cache_key = f"{sephirot_name}_{class_name}"
                        self.integration_cache[cache_key] = {
                            'instance': module_instance,
                            'last_used': datetime.utcnow()
                        }
                        
                        initialized.append(f"{sephirot_name}↔{class_name}")
                        print(f"[INTEGRATION] Установлена связь: {sephirot_name} ↔ {class_name}")
                        
                except ImportError as e:
                    print(f"[INTEGRATION] Модуль не найден: {module_name} ({e})")
                except AttributeError as e:
                    print(f"[INTEGRATION] Ошибка атрибута: {e}")
                except Exception as e:
                    print(f"[INTEGRATION] Общая ошибка: {e}")
        
        return initialized


# ============================================================================
# ГЛАВНЫЙ ДВИГАТЕЛЬ СЕФИРОТИЧЕСКОЙ СИСТЕМЫ
# ============================================================================

class SephiroticEngine:
    """Движок сефиротической системы - полная оптимизированная версия"""
    
    def __init__(self, config_path: str = "config/sephirot_config.yaml"):
        # Ядро системы
        self.bus = SephiroticBus()
        self.nodes: Dict[str, SephiroticNode] = {}
        self.node_registry: Dict[str, Dict[str, Any]] = {}  # Защита от дубликатов
        self.running = False
        self.cycle_counter = 0
        self.config_path = config_path
        
        # Вспомогательные системы
        self.network_fatigue = NetworkFatigue()
        self.visualizer = SephiroticVisualizer(self)
        self.quantum_validator = QuantumLinkValidator(inactive_threshold=50)
        self.integration_manager = IntegrationManager()
        self.distributed_manager: Optional[DistributedHeartbeatManager] = None
        
        # Задачи и фоновые процессы
        self.tasks: List[asyncio.Task] = []
        self.background_operations = {
            'link_validation': None,
            'node_cleanup': None,
            'visualization_update': None,
            'fatigue_monitoring': None
        }
        
        # Адаптивные параметры
        self.adaptive_params = {
            'heartbeat_interval': 2.0,
            'resonance_threshold': 0.65,
            'max_inactive_cycles': 100,
            'auto_connect': True,
            'quantum_link_timeout': 50,
            'visualization_update_interval': 5.0
        }
        
        # Конфигурация
        self.config = self._load_config()
        self._merge_config_with_adaptive_params()
        
        # Эволюционная память
        self.evolutionary_memory = {
            'successful_connections': deque(maxlen=100),
            'failed_connections': deque(maxlen=100),
            'resonance_peaks': deque(maxlen=50),
            'performance_trend': 1.0,
            'adaptation_history': deque(maxlen=200)
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Загрузка конфигурации из YAML"""
        import yaml
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"[ENGINE] Конфиг не найден: {self.config_path}, использую значения по умолчанию")
            return {}
        except Exception as e:
            print(f"[ENGINE] Ошибка загрузки конфига: {e}")
            return {}
    
    def _merge_config_with_adaptive_params(self) -> None:
        """Слияние конфигурации с адаптивными параметрами"""
        if 'sephirot' in self.config:
            for key, value in self.config['sephirot'].items():
                if key in self.adaptive_params:
                    self.adaptive_params[key] = value
        
        # Установка порога для валидатора
        if 'quantum_link_timeout' in self.adaptive_params:
            self.quantum_validator.inactive_threshold = self.adaptive_params['quantum_link_timeout']
    
    async def initialize(self) -> Dict[str, Any]:
        """Полная инициализация движка"""
        print("=" * 60)
        print("🚀 ИНИЦИАЛИЗАЦИЯ СЕФИРОТИЧЕСКОЙ СИСТЕМЫ ISKRA-4")
        print("=" * 60)
        
        results = {
            "nodes": [],
            "integrations": [],
            "distribution": False,
            "visualization": False,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # 1. Обнаружение и создание узлов с защитой от дубликатов
            nodes_created = await self._discover_and_create_nodes()
            results["nodes"] = list(nodes_created.keys())
            print(f"✅ Узлы созданы: {len(nodes_created)}")
            
            # 2. Создание квантовых связей
            links_created = await self._create_quantum_links()
            print(f"✅ Связи созданы: {links_created}")
            
            # 3. Инициализация интеграций
            integrations = await self.integration_manager.initialize(self)
            results["integrations"] = integrations
            if integrations:
                print(f"✅ Интеграции: {', '.join(integrations)}")
            
            # 4. Распределенная система (если включена в конфиге)
            if self.config.get('distribution', {}).get('enabled', False):
                await self._initialize_distributed_system()
                results["distribution"] = True
                print("✅ Распределенная система активирована")
            
            # 5. Визуализация
            if self.config.get('visualization', {}).get('enabled', True):
                viz_ok = await self.visualizer.initialize()
                results["visualization"] = viz_ok
                if viz_ok:
                    print("✅ Визуализация инициализирована")
            
            # 6. Запуск фоновых задач
            await self._start_background_tasks()
            print(f"✅ Фоновые задачи запущены: {len(self.tasks)}")
            
            results["status"] = "initialized"
            results["success"] = True
            results["cycle_counter"] = self.cycle_counter
            
            print("=" * 60)
            print("🎯 СЕФИРОТИЧЕСКАЯ СИСТЕМА ГОТОВА К РАБОТЕ")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Ошибка инициализации: {e}")
            results["status"] = "failed"
            results["success"] = False
            results["error"] = str(e)
        
        return results
    
    async def _discover_and_create_nodes(self) -> Dict[str, SephiroticNode]:
        """Обнаружение и создание узлов с защитой от дубликатов"""
        created_nodes = {}
        
        # Сначала пытаемся загрузить существующие узлы из модулей
        loaded_nodes = await self._load_existing_nodes()
        created_nodes.update(loaded_nodes)
        
        # Если не загрузили достаточно узлов, создаем базовые
        if len(created_nodes) < 3:
            core_nodes = await self._create_core_nodes()
            created_nodes.update(core_nodes)
        
        self.nodes = created_nodes
        return created_nodes
    
    async def _load_existing_nodes(self) -> Dict[str, SephiroticNode]:
        """Загрузка существующих узлов из модулей"""
        loaded = {}
        
        # Базовые модули для проверки
        modules_to_check = [
            ("sephirot_blocks._1_keter.keter_core", "KetherCore", "Kether"),
            ("sephirot_blocks._6_tiferet.tiferet_core", "TiferetCore", "Tiferet"),
            ("sephirot_blocks._9_yesod.yesod_core", "YesodCore", "Yesod")
        ]
        
        for module_path, class_name, node_name in modules_to_check:
            try:
                # Проверка на дубликат
                if node_name in self.node_registry:
                    print(f"[ENGINE] Узел {node_name
