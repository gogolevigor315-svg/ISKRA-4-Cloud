 # sephirot_bus.py - АБСОЛЮТНО ИДЕАЛЬНАЯ СЕФИРОТИЧЕСКАЯ ШИНА
import asyncio
import json
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
import statistics
import yaml
import numpy as np
from enum import Enum
import aiohttp
from aiohttp import web, WSMsgType
import graphviz
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import prometheus_client
from prometheus_client import Gauge, Counter, Histogram, Summary, Info
from tensorflow import keras
from tensorflow.keras import layers
import threading

from .sephirot_base import (
    SephiroticNode, QuantumLink, SignalPackage, 
    SignalType, NodeStatus, ResonancePhase, NodeMetrics
)


# ============================================================================
# МОДУЛЬ МЕТРИК PROMETHEUS
# ============================================================================

class PrometheusMetricsExporter:
    """Экспортер метрик в формате Prometheus с поддержкой многопоточности"""
    
    def __init__(self, namespace: str = "sephirot_bus"):
        self.namespace = namespace
        self.metrics = {}
        self.lock = threading.Lock()
        
        # Инициализация Prometheus метрик
        self._init_prometheus_metrics()
    
    def _init_prometheus_metrics(self):
        """Инициализация всех Prometheus метрик"""
        with self.lock:
            # Гаужи (текущие значения)
            self.metrics["channels_total"] = Gauge(
                f"{self.namespace}_channels_total",
                "Total number of quantum channels",
                ["direction", "status"]
            )
            
            self.metrics["channels_active"] = Gauge(
                f"{self.namespace}_channels_active",
                "Number of active quantum channels"
            )
            
            self.metrics["channel_strength"] = Gauge(
                f"{self.namespace}_channel_strength",
                "Current channel strength",
                ["channel_id", "hebrew_letter", "from_sephira", "to_sephira"]
            )
            
            self.metrics["channel_resonance"] = Gauge(
                f"{self.namespace}_channel_resonance",
                "Current channel resonance factor",
                ["channel_id", "hebrew_letter"]
            )
            
            self.metrics["channel_load_percentage"] = Gauge(
                f"{self.namespace}_channel_load_percentage",
                "Current channel load percentage",
                ["channel_id"]
            )
            
            self.metrics["nodes_registered"] = Gauge(
                f"{self.namespace}_nodes_registered",
                "Number of registered sephirotic nodes"
            )
            
            self.metrics["nodes_active"] = Gauge(
                f"{self.namespace}_nodes_active",
                "Number of active sephirotic nodes"
            )
            
            self.metrics["system_coherence"] = Gauge(
                f"{self.namespace}_system_coherence",
                "Current system coherence level (0-1)"
            )
            
            self.metrics["queue_sizes"] = Gauge(
                f"{self.namespace}_queue_size",
                "Current queue sizes",
                ["queue_type"]
            )
            
            # Каунтеры (накопительные)
            self.metrics["signals_transmitted"] = Counter(
                f"{self.namespace}_signals_transmitted_total",
                "Total number of signals transmitted",
                ["signal_type", "status"]
            )
            
            self.metrics["feedback_messages"] = Counter(
                f"{self.namespace}_feedback_messages_total",
                "Total number of feedback messages processed"
            )
            
            self.metrics["channel_transmissions"] = Counter(
                f"{self.namespace}_channel_transmissions_total",
                "Total transmissions per channel",
                ["channel_id", "result"]
            )
            
            # Гистограммы (распределения)
            self.metrics["signal_processing_time"] = Histogram(
                f"{self.namespace}_signal_processing_seconds",
                "Signal processing time distribution",
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
            )
            
            self.metrics["channel_latency"] = Histogram(
                f"{self.namespace}_channel_latency_seconds",
                "Channel latency distribution",
                ["channel_id"],
                buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
            )
            
            # Саммари (сводки)
            self.metrics["signal_strength_summary"] = Summary(
                f"{self.namespace}_signal_strength_summary",
                "Signal strength summary statistics",
                ["signal_type"]
            )
            
            # Инфо (статическая информация)
            self.metrics["bus_info"] = Info(
                f"{self.namespace}_info",
                "Information about the Sephirotic Bus"
            )
            
            print(f"[METRICS] Prometheus экспортер инициализирован ({self.namespace})")
    
    def update_channel_metrics(self, channel: 'QuantumChannel'):
        """Обновление метрик канала"""
        with self.lock:
            # Гаужи
            self.metrics["channel_strength"].labels(
                channel_id=channel.id,
                hebrew_letter=channel.hebrew_letter,
                from_sephira=channel.from_sephira,
                to_sephira=channel.to_sephira
            ).set(channel.current_strength)
            
            self.metrics["channel_resonance"].labels(
                channel_id=channel.id,
                hebrew_letter=channel.hebrew_letter
            ).set(channel.resonance_factor)
            
            self.metrics["channel_load_percentage"].labels(
                channel_id=channel.id
            ).set((channel.current_load / channel.max_bandwidth) * 100 if channel.max_bandwidth > 0 else 0)
    
    def update_system_metrics(self, nodes_total: int, nodes_active: int, 
                             coherence: float, queue_sizes: Dict[str, int]):
        """Обновление системных метрик"""
        with self.lock:
            self.metrics["nodes_registered"].set(nodes_total)
            self.metrics["nodes_active"].set(nodes_active)
            self.metrics["system_coherence"].set(coherence)
            
            for queue_type, size in queue_sizes.items():
                self.metrics["queue_sizes"].labels(queue_type=queue_type).set(size)
    
    def record_signal_transmission(self, signal_type: str, success: bool, 
                                  processing_time: float = None, 
                                  strength: float = None):
        """Запись метрик передачи сигнала"""
        with self.lock:
            status = "success" if success else "failure"
            self.metrics["signals_transmitted"].labels(
                signal_type=signal_type,
                status=status
            ).inc()
            
            if processing_time is not None:
                self.metrics["signal_processing_time"].observe(processing_time)
            
            if strength is not None:
                self.metrics["signal_strength_summary"].labels(
                    signal_type=signal_type
                ).observe(strength)
    
    def record_channel_transmission(self, channel_id: str, success: bool, 
                                   latency: float = None):
        """Запись метрик передачи по каналу"""
        with self.lock:
            result = "success" if success else "failure"
            self.metrics["channel_transmissions"].labels(
                channel_id=channel_id,
                result=result
            ).inc()
            
            if latency is not None:
                self.metrics["channel_latency"].labels(
                    channel_id=channel_id
                ).observe(latency)
    
    def record_feedback_message(self):
        """Запись метрик обратной связи"""
        with self.lock:
            self.metrics["feedback_messages"].inc()
    
    def update_bus_info(self, info: Dict[str, str]):
        """Обновление информации о шине"""
        with self.lock:
            self.metrics["bus_info"].info(info)
    
    def get_metrics_http_handler(self):
        """Получение HTTP обработчика для метрик Prometheus"""
        return prometheus_client.make_wsgi_app()
    
    def generate_metrics_report(self) -> Dict[str, Any]:
        """Генерация отчета по метрикам"""
        with self.lock:
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "namespace": self.namespace,
                "metrics": {}
            }
            
            # Сбор данных по метрикам
            for name, metric in self.metrics.items():
                if hasattr(metric, '_metrics'):
                    # Для метрик с лейблами
                    metric_data = {}
                    for label_values, metric_instance in metric._metrics.items():
                        if hasattr(metric_instance, '_value'):
                            metric_data[str(label_values)] = metric_instance._value.get()
                    
                    if metric_data:
                        report["metrics"][name] = metric_data
                elif hasattr(metric, '_value'):
                    # Для простых метрик
                    report["metrics"][name] = metric._value.get()
            
            return report


# ============================================================================
# МОДУЛЬ ВИЗУАЛИЗАЦИИ ГРАФА
# ============================================================================

class GraphVisualizer:
    """Продвинутый визуализатор графа сефиротической сети"""
    
    def __init__(self):
        self.graphviz_graph = None
        self.plotly_figure = None
        self.last_update = None
        self.layout_cache = {}
        
    def create_graphviz_graph(self, channels: List['QuantumChannel'], 
                             nodes: Dict[str, SephiroticNode], 
                             title: str = "Сефиротическая Сеть") -> graphviz.Digraph:
        """Создание графа Graphviz"""
        
        # Создание направленного графа
        graph = graphviz.Digraph(
            comment=title,
            format='svg',
            engine='neato',  # Для позиционирования
            graph_attr={
                'label': title,
                'labelloc': 't',
                'fontsize': '20',
                'fontname': 'Helvetica',
                'bgcolor': '#0f0f1f',
                'rankdir': 'TB',  # Top to Bottom
                'splines': 'curved',
                'overlap': 'false'
            },
            node_attr={
                'shape': 'circle',
                'style': 'filled',
                'fontname': 'Helvetica',
                'fontsize': '12',
                'width': '0.8',
                'height': '0.8'
            },
            edge_attr={
                'fontname': 'Helvetica',
                'fontsize': '10',
                'arrowsize': '0.7'
            }
        )
        
        # Цветовая схема для сефирот
        sephira_colors = {
            "Kether": "#ffd700",    # Золотой
            "Chokhmah": "#4169e1",  # Королевский синий
            "Binah": "#8a2be2",     # Сине-фиолетовый
            "Chesed": "#32cd32",    # Лаймовый
            "Gevurah": "#dc143c",   # Малиновый
            "Tiferet": "#ff69b4",   # Ярко-розовый
            "Netzach": "#00ced1",   # Темный бирюзовый
            "Hod": "#ff8c00",       # Темно-оранжевый
            "Yesod": "#9370db",     # Средне-фиолетовый
            "Malkuth": "#2e8b57"    # Морская зелень
        }
        
        # Добавление узлов (сефирот)
        for node_name, node in nodes.items():
            color = sephira_colors.get(node_name, "#808080")
            
            # Определение активности
            is_active = node.status == NodeStatus.ACTIVE if hasattr(node, 'status') else True
            
            node_attrs = {
                'fillcolor': f"{color}{'ff' if is_active else '80'}",  # Полная или полупрозрачная
                'color': color,
                'penwidth': '3' if is_active else '1',
                'label': f"{node_name}\n{node.resonance:.2f}" if hasattr(node, 'resonance') else node_name
            }
            
            if not is_active:
                node_attrs['style'] = 'filled,dashed'
            
            graph.node(node_name, **node_attrs)
        
        # Добавление ребер (каналов)
        for channel in channels:
            if channel.from_sephira in nodes and channel.to_sephira in nodes:
                # Цвет ребра на основе силы канала
                strength_color = self._strength_to_color(channel.current_strength)
                resonance_alpha = hex(int(channel.resonance_factor * 255))[2:].zfill(2)
                
                edge_attrs = {
                    'color': f"{strength_color}{resonance_alpha}",
                    'penwidth': str(max(1, channel.current_strength * 5)),
                    'label': channel.hebrew_letter,
                    'fontcolor': strength_color,
                    'dir': 'both' if channel.direction == ChannelDirection.BIDIRECTIONAL else 'forward',
                    'style': 'solid' if channel.is_active else 'dashed'
                }
                
                # Для перегруженных каналов
                load_percentage = (channel.current_load / channel.max_bandwidth) if channel.max_bandwidth > 0 else 0
                if load_percentage > 0.8:
                    edge_attrs['color'] = '#ff0000'  # Красный для перегруженных
                    edge_attrs['penwidth'] = '3'
                    edge_attrs['style'] = 'bold'
                
                graph.edge(channel.from_sephira, channel.to_sephira, **edge_attrs)
        
        self.graphviz_graph = graph
        self.last_update = datetime.utcnow()
        
        return graph
    
    def _strength_to_color(self, strength: float) -> str:
        """Конвертация силы канала в цвет"""
        if strength > 0.8:
            return "#00ff00"  # Зеленый
        elif strength > 0.6:
            return "#aaff00"  # Лаймовый
        elif strength > 0.4:
            return "#ffff00"  # Желтый
        elif strength > 0.2:
            return "#ffaa00"  # Оранжевый
        else:
            return "#ff0000"  # Красный
    
    def create_plotly_visualization(self, channels: List['QuantumChannel'], 
                                   nodes: Dict[str, SephiroticNode]) -> go.Figure:
        """Создание интерактивной визуализации Plotly"""
        
        # Позиции сефирот в 3D пространстве
        positions = self._calculate_3d_positions(nodes)
        
        # Создание фигуры
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{'type': 'scatter3d'}]],
            subplot_titles=['🌳 Сефиротическая Сеть в 3D']
        )
        
        # Добавление узлов
        node_x, node_y, node_z = [], [], []
        node_text, node_color, node_size = [], [], []
        
        for node_name, (x, y, z) in positions.items():
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)
            
            node = nodes.get(node_name)
            resonance = node.resonance if node and hasattr(node, 'resonance') else 0.5
            energy = node.energy if node and hasattr(node, 'energy') else 0.5
            
            # Текст для tooltip
            node_text.append(
                f"<b>{node_name}</b><br>"
                f"Резонанс: {resonance:.2f}<br>"
                f"Энергия: {energy:.2f}<br>"
                f"Статус: {node.status.value if node else 'unknown'}"
            )
            
            # Цвет на основе резонанса
            r = int((1 - resonance) * 255)
            g = int(resonance * 255)
            node_color.append(f'rgb({r}, {g}, 100)')
            
            # Размер на основе энергии
            node_size.append(10 + energy * 15)
        
        # Добавление узлов в граф
        fig.add_trace(go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers+text',
            marker=dict(
                size=node_size,
                color=node_color,
                line=dict(width=2, color='white'),
                opacity=0.9
            ),
            text=[name for name in positions.keys()],
            textposition="top center",
            hovertext=node_text,
            hoverinfo='text',
            name='Сефироты'
        ), row=1, col=1)
        
        # Добавление ребер (каналов)
        for channel in channels:
            if (channel.from_sephira in positions and 
                channel.to_sephira in positions):
                
                x0, y0, z0 = positions[channel.from_sephira]
                x1, y1, z1 = positions[channel.to_sephira]
                
                # Цвет ребра на основе силы
                strength_color = self._strength_to_plotly_color(channel.current_strength)
                
                fig.add_trace(go.Scatter3d(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    z=[z0, z1, None],
                    mode='lines',
                    line=dict(
                        color=strength_color,
                        width=max(1, channel.current_strength * 3),
                        dash='solid' if channel.is_active else 'dash'
                    ),
                    opacity=0.7,
                    hoverinfo='none',
                    showlegend=False,
                    name=f"{channel.hebrew_letter}: {channel.current_strength:.2f}"
                ), row=1, col=1)
        
        # Настройка макета
        fig.update_layout(
            title=dict(
                text="Сефиротическая Сеть ISKRA-4",
                font=dict(size=24, color='white')
            ),
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, title=''),
                yaxis=dict(showbackground=False, showticklabels=False, title=''),
                zaxis=dict(showbackground=False, showticklabels=False, title=''),
                bgcolor='rgba(10, 10, 30, 1)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            paper_bgcolor='rgba(10, 10, 30, 1)',
            font=dict(color='white', size=12),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='white',
                borderwidth=1
            )
        )
        
        self.plotly_figure = fig
        return fig
    
    def _calculate_3d_positions(self, nodes: Dict[str, SephiroticNode]) -> Dict[str, Tuple[float, float, float]]:
        """Расчет 3D позиций для сефирот"""
        
        # Классическая схема Древа Жизни в 3D
        positions = {
            "Kether": (0, 0, 2),      # Вверху
            "Chokhmah": (-1, 0, 1),   # Слева-сверху
            "Binah": (1, 0, 1),       # Справа-сверху
            "Chesed": (-1.5, 0, 0),   # Слева-середина
            "Gevurah": (1.5, 0, 0),   # Справа-середина
            "Tiferet": (0, 0, 0),     # Центр
            "Netzach": (-1, 0, -1),   # Слева-снизу
            "Hod": (1, 0, -1),        # Справа-снизу
            "Yesod": (0, 0, -1.5),    # Снизу-центр
            "Malkuth": (0, 0, -2.5)   # В самом низу
        }
        
        # Адаптация под существующие узлы
        actual_positions = {}
        for node_name in nodes.keys():
            if node_name in positions:
                actual_positions[node_name] = positions[node_name]
            else:
                # Случайное позиционирование для новых узлов
                actual_positions[node_name] = (
                    np.random.uniform(-2, 2),
                    np.random.uniform(-2, 2),
                    np.random.uniform(-2, 2)
                )
        
        return actual_positions
    
    def _strength_to_plotly_color(self, strength: float) -> str:
        """Конвертация силы канала в цвет для Plotly"""
        if strength > 0.8:
            return "rgba(0, 255, 0, 0.8)"
        elif strength > 0.6:
            return "rgba(170, 255, 0, 0.7)"
        elif strength > 0.4:
            return "rgba(255, 255, 0, 0.6)"
        elif strength > 0.2:
            return "rgba(255, 170, 0, 0.5)"
        else:
            return "rgba(255, 0, 0, 0.4)"
    
    def save_graphviz_to_file(self, filename: str = "sephirot_network.svg"):
        """Сохранение графа Graphviz в файл"""
        if self.graphviz_graph:
            self.graphviz_graph.render(
                filename=filename.replace('.svg', ''),
                format='svg',
                cleanup=True
            )
            return True
        return False
    
    def get_plotly_html(self, include_plotlyjs: str = 'cdn') -> str:
        """Получение HTML с Plotly визуализацией"""
        if self.plotly_figure:
            return self.plotly_figure.to_html(
                include_plotlyjs=include_plotlyjs,
                full_html=True,
                config={'responsive': True}
            )
        return "<div>Визуализация не готова</div>"
    
    def generate_live_dashboard(self, bus_state: Dict[str, Any]) -> str:
        """Генерация живого дашборда"""
        
        html = f"""
        <!DOCTYPE html>
        <html lang="ru">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Сефиротическая Сеть - Live Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #0f0f1f 0%, #1a1a2e 100%);
                    color: white;
                    font-family: 'Arial', sans-serif;
                }}
                .dashboard {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    max-width: 1800px;
                    margin: 0 auto;
                }}
                .card {{
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 10px;
                    padding: 20px;
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                }}
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 10px;
                    margin-top: 15px;
                }}
                .metric {{
                    background: rgba(0, 0, 0, 0.3);
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #4dabf7;
                }}
                .metric-label {{
                    font-size: 12px;
                    color: #adb5bd;
                }}
                h1, h2, h3 {{
                    margin-top: 0;
                    color: #ffd700;
                }}
                #graph3d {{
                    height: 600px;
                }}
                .health-indicator {{
                    display: inline-block;
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    margin-right: 5px;
                }}
                .healthy {{ background: #40c057; }}
                .warning {{ background: #fab005; }}
                .critical {{ background: #fa5252; }}
            </style>
        </head>
        <body>
            <h1>🌳 Древо Жизни - Live Dashboard</h1>
            
            <div class="dashboard">
                <div class="card" style="grid-column: span 2;">
                    <h2>3D Визуализация Сети</h2>
                    <div id="graph3d"></div>
                </div>
                
                <div class="card">
                    <h2>Системные Метрики</h2>
                    <div class="metrics-grid">
                        <div class="metric">
                            <div class="metric-value">{bus_state.get('nodes_active', 0)}/{bus_state.get('total_node_count', 0)}</div>
                            <div class="metric-label">Активных узлов</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{bus_state.get('system_coherence', 0):.2%}</div>
                            <div class="metric-label">Когерентность</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{bus_state.get('channel_statistics', {{}}).get('active', 0)}/{bus_state.get('channel_statistics', {{}}).get('total', 0)}</div>
                            <div class="metric-label">Активных каналов</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{bus_state.get('recent_signals', 0)}</div>
                            <div class="metric-label">Сигналов (24ч)</div>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Состояние Каналов</h2>
                    <div id="channels-health"></div>
                </div>
                
                <div class="card" style="grid-column: span 2;">
                    <h2>Текущая Активность</h2>
                    <div id="recent-activity"></div>
                </div>
            </div>
            
            <script>
                // JavaScript для живого обновления
                function updateDashboard() {{
                    fetch('/bus/state')
                        .then(response => response.json())
                        .then(data => {{
                            // Обновление метрик
                            document.querySelector('.metric-value:nth-child(1)').textContent = 
                                `${{data.nodes_active}}/${{data.total_node_count}}`;
                            
                            document.querySelector('.metric-value:nth-child(2)').textContent = 
                                `${{(data.system_coherence * 100).toFixed(2)}}%`;
                            
                            // Обновление 3D графа каждые 30 секунд
                            if (window.lastGraphUpdate && (Date.now() - window.lastGraphUpdate) > 30000) {{
                                update3DGraph(data);
                                window.lastGraphUpdate = Date.now();
                            }}
                        }});
                }}
                
                // Автообновление каждые 5 секунд
                setInterval(updateDashboard, 5000);
                updateDashboard();
            </script>
        </body>
        </html>
        """
        
        return html


# ============================================================================
# НЕЙРОННЫЙ ПРЕДИКТОР КАНАЛОВ LSTM
# ============================================================================

class ChannelDegradationPredictor:
    """LSTM нейронная сеть для предсказания деградации каналов"""
    
    def __init__(self, sequence_length: int = 10, prediction_horizon: int = 5):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.scaler = None
        self.training_history = []
        self.is_trained = False
        
        # История данных для каждого канала
        self.channel_histories: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=sequence_length * 2)
        )
        
    def build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Построение LSTM модели"""
        
        model = keras.Sequential([
            layers.LSTM(
                64,
                input_shape=input_shape,
                return_sequences=True,
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            layers.LSTM(
                32,
                dropout=0.2,
                recurrent_dropout=0.2
            ),
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.prediction_horizon, activation='linear')  # Прогноз на N шагов вперед
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        self.model = model
        return model
    
    def prepare_training_data(self, channel_histories: Dict[str, List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        """Подготовка данных для обучения"""
        
        sequences = []
        targets = []
        
        for channel_id, history in channel_histories.items():
            if len(history) >= self.sequence_length + self.prediction_horizon:
                history_array = np.array(history)
                
                # Нормализация
                if self.scaler is None:
                    from sklearn.preprocessing import MinMaxScaler
                    self.scaler = MinMaxScaler()
                    history_array = history_array.reshape(-1, 1)
                    history_array = self.scaler.fit_transform(history_array).flatten()
                else:
                    history_array = history_array.reshape(-1, 1)
                    history_array = self.scaler.transform(history_array).flatten()
                
                # Создание последовательностей
                for i in range(len(history_array) - self.sequence_length - self.prediction_horizon + 1):
                    seq = history_array[i:i + self.sequence_length]
                    target = history_array[i + self.sequence_length:i + self.sequence_length + self.prediction_horizon]
                    
                    sequences.append(seq)
                    targets.append(target)
        
        if not sequences:
            return np.array([]), np.array([])
        
        X = np.array(sequences).reshape(-1, self.sequence_length, 1)
        y = np.array(targets)
        
        return X, y
    
    async def train(self, channel_histories: Dict[str, List[float]], 
                   epochs: int = 50, validation_split: float = 0.2):
        """Обучение модели"""
        
        X, y = self.prepare_training_data(channel_histories)
        
        if len(X) == 0:
            print("[PREDICTOR] Недостаточно данных для обучения")
            return
        
        print(f"[PREDICTOR] Обучение на {len(X)} последовательностях...")
        
        # Разделение на тренировочную и валидационную выборки
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Построение модели если еще не построена
        if self.model is None:
            self.build_model((self.sequence_length, 1))
        
        # Обучение
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5
                )
            ]
        )
        
        self.training_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "epochs": epochs,
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "final_loss": history.history['loss'][-1],
            "final_val_loss": history.history['val_loss'][-1]
        })
        
        self.is_trained = True
        print(f"[PREDICTOR] Обучение завершено. Final loss: {history.history['val_loss'][-1]:.4f}")
    
    async def predict_degradation(self, channel_id: str, 
                                 current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Предсказание деградации канала"""
        
        if not self.is_trained or self.model is None:
            return {"error": "Модель не обучена", "confidence": 0}
        
        # Добавление текущих метрик в историю
        if channel_id not in self.channel_histories:
            self.channel_histories[channel_id] = deque(maxlen=self.sequence_length * 2)
        
        # Используем силу канала как основной показатель
        if 'current_strength' in current_metrics:
            self.channel_histories[channel_id].append(current_metrics['current_strength'])
        
        # Проверка наличия достаточной истории
        if len(self.channel_histories[channel_id]) < self.sequence_length:
            return {"error": "Недостаточно данных", "confidence": 0}
        
        # Подготовка последовательности для предсказания
        recent_history = list(self.channel_histories[channel_id])[-self.sequence_length:]
        
        # Нормализация
        if self.scaler:
            history_array = np.array(recent_history).reshape(-1, 1)
            history_array = self.scaler.transform(history_array).flatten()
        else:
            history_array = np.array(recent_history)
        
        # Предсказание
        X_pred = history_array.reshape(1, self.sequence_length, 1)
        predictions = self.model.predict(X_pred, verbose=0)[0]
        
        # Денормализация если есть scaler
        if self.scaler:
            predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        # Анализ предсказаний
        current_value = recent_history[-1]
        predicted_values = predictions.tolist()
        
        # Расчет тренда
        trend = "stable"
        if len(predicted_values) >= 2:
            if predicted_values[-1] < current_value * 0.8:
                trend = "degrading"
            elif predicted_values[-           
