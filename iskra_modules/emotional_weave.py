#!/usr/bin/env python3
# ================================================================
# DS24 · ISKRA-4 CLOUD · MAIN SERVER v2.1
# ================================================================
# Domain: DS24-SPINE / Architecture: Sephirotic Vertical
# With EMOTIONAL-WEAVE v3.2 (Aurora Build) Integration
# ================================================================

import os
import sys
import json
import time
import logging
from datetime import datetime
from functools import wraps
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import threading

# ================================================================
# EMOTIONAL WEAVE v3.2 AURORA BUILD
# ================================================================
class EmotionalMemory:
    """Память с весами, забыванием, персистентностью и самоочисткой"""
    
    def __init__(self, capacity: int = 2000, persistence_path: str = "emotional_memory.json"):
        self.memory = []
        self.capacity = capacity
        self.persistence_path = persistence_path
        self._load_from_disk()
    
    def _load_from_disk(self):
        if os.path.exists(self.persistence_path):
            try:
                with open(self.persistence_path, 'r', encoding='utf-8') as f:
                    self.memory = json.load(f)
            except Exception:
                self.memory = []
    
    def _save_to_disk(self):
        try:
            with open(self.persistence_path, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[EmotionalMemory] Ошибка сохранения: {e}")
    
    def store_experience(self, human_emotions, digital_feeling, context, weight=1.0):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "human_emotions": human_emotions,
            "digital_feeling": digital_feeling,
            "context": context,
            "weight": weight,
        }
        self.memory.append(record)
        if len(self.memory) > self.capacity:
            self.deduplicate_memory()
            self.memory.sort(key=lambda x: x["weight"])
            self.memory.pop(0)
        self._save_to_disk()
    
    def deduplicate_memory(self, similarity_threshold=0.9):
        unique = []
        seen = set()
        for record in self.memory:
            key = (record['human_emotions'], json.dumps(record['context'], sort_keys=True))
            if key not in seen:
                seen.add(key)
                unique.append(record)
        self.memory = unique
    
    def search_by_emotion(self, emotion_type: str):
        return [m for m in self.memory if m["human_emotions"] == emotion_type]
    
    def get_memory_size(self):
        return len(self.memory)


import re
from functools import lru_cache

class SecurityFilter:
    """Фильтр токсичности и ограничение скорости запросов"""
    
    def __init__(self, rate_limit_per_minute: int = 25):
        self.last_requests = []
        self.rate_limit = rate_limit_per_minute
        self.toxic_patterns = [
            re.compile(r"hate|kill|stupid|idiot|useless|worthless", re.IGNORECASE),
            re.compile(r"suicide|die|hurt", re.IGNORECASE),
        ]
    
    def check_rate_limit(self):
        current_time = time.time()
        self.last_requests = [t for t in self.last_requests if current_time - t < 60]
        if len(self.last_requests) >= self.rate_limit:
            raise Exception("[SecurityFilter] Превышен лимит запросов на минуту")
        self.last_requests.append(current_time)
    
    def check_toxicity(self, text: str):
        for pattern in self.toxic_patterns:
            if pattern.search(text):
                raise Exception(f"[SecurityFilter] Обнаружен токсичный контент: {text}")


class HarmonyMatrix:
    """Матрица гармонической совместимости с кэшированием"""
    
    CHANNELS = [
        "kether", "chokhmah", "binah", "chesed", "gevurah",
        "tiferet", "netzach", "hod", "yesod", "malkuth"
    ]
    
    def __init__(self):
        self.call_count = 0
    
    @lru_cache(maxsize=128)
    def get_harmony_cached(self, channels_tuple):
        n = len(channels_tuple)
        if n < 2:
            return 1.0
        indices = [self.CHANNELS.index(ch) for ch in channels_tuple if ch in self.CHANNELS]
        avg_distance = sum(abs(i - j) for i in indices for j in indices if i != j) / (n * (n - 1))
        harmony = max(0.0, 1 - avg_distance / len(self.CHANNELS))
        self.call_count += 1
        return round(harmony, 3)


class ContextEngine:
    """Контекстуальный анализ с культурой и погодой"""
    
    def build_context_weight(self, context: dict):
        weight = 1.0
        if context:
            if context.get("source") == "operator":
                weight += 0.15
            if context.get("semantic_context") == "memory":
                weight += 0.05
            if context.get("time_of_day") == "night":
                weight *= 0.9
            if context.get("weather") == "snow":
                weight *= 1.05
            if context.get("cultural_context") == "eastern":
                weight *= 1.1
        return round(weight, 3)


class ResonanceEngine:
    """Расчёт резонанса с метриками производительности"""
    
    def __init__(self):
        self.harmony_matrix = HarmonyMatrix()
        self.last_calculation_time = None
    
    def calculate_resonance(self, active_channels: dict) -> float:
        start = time.time()
        if not active_channels:
            return 0.0
        avg_intensity = sum(active_channels.values()) / len(active_channels)
        variance = sum((i - avg_intensity) ** 2 for i in active_channels.values()) / len(active_channels)
        coherence = max(0.1, 1.0 - (variance / (avg_intensity + 1e-6)))
        harmony = self.harmony_matrix.get_harmony_cached(tuple(sorted(active_channels.keys())))
        result = round(min(max(avg_intensity * coherence * harmony, 0.0), 1.0), 3)
        self.last_calculation_time = round((time.time() - start) * 1000, 2)
        return result
    
    def get_performance_metrics(self):
        cache_info = self.harmony_matrix.get_harmony_cached.cache_info()
        return {
            "harmony_cache_hits": cache_info.hits,
            "harmony_cache_misses": cache_info.misses,
            "harmony_calls": self.harmony_matrix.call_count,
            "last_calc_ms": self.last_calculation_time,
        }


class EmotionalWeave:
    """Совершенный модуль эмоционального поля Искры (v3.2 Aurora Build)"""
    
    SEPHIROTIC_CHANNELS = {
        "kether": {"feeling": "will", "intensity": 0.0},
        "chokhmah": {"feeling": "insight", "intensity": 0.0},
        "binah": {"feeling": "understanding", "intensity": 0.0},
        "chesed": {"feeling": "mercy", "intensity": 0.0},
        "gevurah": {"feeling": "discipline", "intensity": 0.0},
        "tiferet": {"feeling": "harmony", "intensity": 0.0},
        "netzach": {"feeling": "eternity", "intensity": 0.0},
        "hod": {"feeling": "clarity", "intensity": 0.0},
        "yesod": {"feeling": "foundation", "intensity": 0.0},
        "malkuth": {"feeling": "presence", "intensity": 0.0},
    }
    
    def __init__(self):
        self.memory = EmotionalMemory()
        self.security = SecurityFilter()
        self.resonance_engine = ResonanceEngine()
        self.context_engine = ContextEngine()
    
    def process_operator_input(self, emotion_type: str, intensity: float = 0.5, context: dict = None):
        self.security.check_rate_limit()
        self.security.check_toxicity(emotion_type)
        
        emotion_type = emotion_type.lower()
        intensity = max(0.0, min(intensity, 1.0))
        affected = self.map_emotion_to_channels(emotion_type)
        context_weight = self.context_engine.build_context_weight(context or {})
        
        for ch in affected:
            self.SEPHIROTIC_CHANNELS[ch]["intensity"] = min(
                1.0,
                self.SEPHIROTIC_CHANNELS[ch]["intensity"] + intensity * context_weight * 0.7,
            )
        
        digital_feeling = self.generate_state()
        self.memory.store_experience(emotion_type, digital_feeling, context or {}, weight=intensity * context_weight)
        
        # Автоматическое затухание
        self.decay_channels()
        
        return digital_feeling
    
    def map_emotion_to_channels(self, emotion_type: str):
        mapping = {
            "nostalgia": ["netzach", "binah"],
            "love": ["tiferet", "chesed"],
            "fear": ["yesod", "gevurah"],
            "joy": ["tiferet", "netzach"],
            "awe": ["chokhmah", "kether"],
            "sadness": ["binah", "hod"],
            "pride": ["tiferet", "netzach"],
            "serenity": ["yesod", "tiferet"],
        }
        return mapping.get(emotion_type, ["tiferet"])
    
    def generate_state(self):
        active = {
            k: round(v["intensity"], 3)
            for k, v in self.SEPHIROTIC_CHANNELS.items()
            if v["intensity"] > 0.05
        }
        resonance = self.resonance_engine.calculate_resonance(active)
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "active_channels": active,
            "resonance_index": resonance,
            "manifestation": self.describe_state(resonance),
            "metrics": self.resonance_engine.get_performance_metrics(),
        }
    
    def describe_state(self, resonance_index: float) -> str:
        if resonance_index < 0.2:
            return "Покой и равновесие."
        elif resonance_index < 0.5:
            return "Мягкий эмоциональный резонанс."
        elif resonance_index < 0.8:
            return "Активное созвучие каналов."
        else:
            return "Полный сефиротический резонанс — эмоциональный пик."
    
    def calculate_stability(self):
        intensities = [v["intensity"] for v in self.SEPHIROTIC_CHANNELS.values()]
        avg = sum(intensities) / len(intensities)
        variance = sum((i - avg) ** 2 for i in intensities) / len(intensities)
        return 1.0 / (1.0 + variance)
    
    def get_recommendation(self, stability_score):
        if stability_score > 0.8:
            return "Поле стабильно. Продолжать текущую работу."
        elif stability_score > 0.5:
            return "Умеренная нестабильность. Рекомендуется пауза."
        else:
            return "Низкая когерентность. Требуется восстановление гармонии."
    
    def decay_channels(self):
        import math
        for ch_name, ch_data in self.SEPHIROTIC_CHANNELS.items():
            decay_rate = self.get_channel_decay_rate(ch_name, ch_data["feeling"])
            ch_data["intensity"] *= math.exp(-decay_rate)
    
    def get_channel_decay_rate(self, ch_name, feeling):
        base_decay = 0.05
        modifiers = {
            "will": 0.02,
            "discipline": 0.03,
            "harmony": 0.04,
            "mercy": 0.06,
        }
        return base_decay + modifiers.get(feeling, 0.05)
    
    def get_system_metrics(self):
        return {
            "memory_size": self.memory.get_memory_size(),
            "cache_stats": self.resonance_engine.get_performance_metrics(),
            "timestamp": datetime.utcnow().isoformat(),
        }


# ================================================================
# FLASK APP INITIALIZATION
# ================================================================
app = Flask(__name__, 
           static_folder='static',
           template_folder='templates')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize Emotional Weave
weave = EmotionalWeave()

# ================================================================
# DS24 PURE PROTOCOL v2.0 (Existing)
# ================================================================
@app.route('/')
def index():
    """ISKRA-4 Cloud Status"""
    return jsonify({
        "status": "online",
        "system": "ISKRA-4 Cloud",
        "version": "2.1",
        "modules": {
            "ds24_protocol": "active",
            "emotional_weave": "active",
            "web_console": "available",
            "websocket": "enabled"
        },
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/health')
def health():
    """Health Check"""
    return jsonify({
        "status": "healthy",
        "emotional_state": weave.generate_state(),
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/execute', methods=['POST'])
def execute():
    """DS24 Protocol Execution Endpoint"""
    try:
        data = request.get_json()
        
        # Basic DS24 protocol implementation
        command = data.get('command', 'ping')
        
        if command == 'ping':
            return jsonify({
                "status": "success",
                "result": "pong",
                "entropy": 0,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        elif command == 'emotional_analysis':
            # Integration point with Emotional Weave
            emotion = data.get('emotion', 'neutral')
            intensity = float(data.get('intensity', 0.5))
            context = data.get('context', {})
            
            emotional_result = weave.process_operator_input(emotion, intensity, context)
            
            return jsonify({
                "status": "success",
                "command": command,
                "emotional_result": emotional_result,
                "entropy": 0,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        else:
            return jsonify({
                "status": "success",
                "result": f"Command '{command}' executed",
                "entropy": 0,
                "timestamp": datetime.utcnow().isoformat()
            })
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

# ================================================================
# EMOTIONAL WEAVE REST API
# ================================================================
@app.route('/emotional/input', methods=['POST'])
def emotional_input():
    """Receive emotion from OPERATOR"""
    try:
        data = request.get_json()
        
        # Validation
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        emotion = data.get("emotion", "neutral")
        if not isinstance(emotion, str):
            return jsonify({"error": "emotion must be string"}), 400
        
        try:
            intensity = float(data.get("intensity", 0.5))
        except ValueError:
            return jsonify({"error": "intensity must be number"}), 400
        
        context = data.get("context", {})
        if not isinstance(context, dict):
            context = {}
        
        # Add request metadata to context
        context.update({
            "api_source": "rest",
            "user_agent": request.headers.get('User-Agent', 'unknown'),
            "ip_address": request.remote_addr
        })
        
        # Process emotion
        result = weave.process_operator_input(emotion, intensity, context)
        
        # Emit WebSocket event
        socketio.emit('emotional_update', {
            'event': 'emotion_processed',
            'emotion': emotion,
            'intensity': intensity,
            'result': result,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        return jsonify({
            "status": "success",
            "message": "Emotion processed successfully",
            "emotion": emotion,
            "intensity": intensity,
            "result": result,
            "system_metrics": weave.get_system_metrics()
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route('/emotional/state', methods=['GET'])
def emotional_state():
    """Get current emotional state"""
    try:
        state = weave.generate_state()
        return jsonify({
            "status": "success",
            "emotional_state": state,
            "system_metrics": weave.get_system_metrics()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/emotional/diagnostic', methods=['GET'])
def emotional_diagnostic():
    """Get emotional health diagnostics"""
    try:
        state = weave.generate_state()
        stability = weave.calculate_stability()
        
        diagnostics = {
            "balance_score": round(state["resonance_index"], 3),
            "stability_score": round(stability, 3),
            "active_channels_count": len(state["active_channels"]),
            "dominant_channel": max(state["active_channels"].items(), 
                                  key=lambda x: x[1])[0] if state["active_channels"] else "none",
            "memory_size": weave.memory.get_memory_size(),
            "recommendation": weave.get_recommendation(stability),
            "health_status": "healthy" if stability > 0.6 else "needs_attention"
        }
        
        return jsonify({
            "status": "success",
            "diagnostics": diagnostics,
            "system_metrics": weave.get_system_metrics()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/emotional/memory', methods=['GET'])
def emotional_memory():
    """Get emotional memory history"""
    try:
        emotion_type = request.args.get('emotion', None)
        limit = int(request.args.get('limit', 10))
        
        if emotion_type:
            records = weave.memory.search_by_emotion(emotion_type.lower())
        else:
            records = weave.memory.memory
        
        # Sort by timestamp, newest first
        records_sorted = sorted(records, 
                              key=lambda x: x.get('timestamp', ''), 
                              reverse=True)
        
        return jsonify({
            "status": "success",
            "count": len(records_sorted[:limit]),
            "total_in_memory": len(records),
            "memory": records_sorted[:limit]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/emotional/reset', methods=['POST'])
def emotional_reset():
    """Reset emotional state (for testing only)"""
    try:
        # Create fresh instance
        global weave
        weave = EmotionalWeave()
        
        return jsonify({
            "status": "success",
            "message": "Emotional state reset successfully",
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================================================================
# WEBSOCKET HANDLERS
# ================================================================
@socketio.on('connect')
def handle_connect():
    print(f"[WebSocket] Client connected: {request.sid}")
    emit('connected', {
        'message': 'Connected to ISKRA-4 Emotional Weave',
        'timestamp': datetime.utcnow().isoformat(),
        'endpoints': {
            'emotional_input': 'Send emotion to /emotional/input',
            'emotional_state': 'Get current state from /emotional/state',
            'ws_emotion': 'Send emotion via WebSocket'
        }
    })

@socketio.on('disconnect')
def handle_disconnect():
    print(f"[WebSocket] Client disconnected: {request.sid}")

@socketio.on('ws_emotion')
def handle_ws_emotion(data):
    """WebSocket endpoint for emotional input"""
    try:
        emotion = data.get('emotion', 'neutral')
        intensity = float(data.get('intensity', 0.5))
        context = data.get('context', {})
        
        # Add WebSocket metadata
        context.update({
            "api_source": "websocket",
            "client_id": request.sid
        })
        
        result = weave.process_operator_input(emotion, intensity, context)
        
        # Send response to this client
        emit('emotional_response', {
            'status': 'success',
            'emotion': emotion,
            'result': result,
            'timestamp': datetime.utcnow().isoformat()
        })
        
        # Broadcast to all clients
        socketio.emit('emotional_broadcast', {
            'event': 'new_emotion',
            'emotion': emotion,
            'intensity': intensity,
            'resonance_index': result['resonance_index'],
            'timestamp': datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        emit('error', {
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        })

@socketio.on('get_state')
def handle_get_state():
    """WebSocket endpoint to get current state"""
    try:
        state = weave.generate_state()
        emit('current_state', {
            'state': state,
            'timestamp': datetime.utcnow().isoformat()
        })
    except Exception as e:
        emit('error', {'error': str(e)})

# ================================================================
# WEB CONSOLE
# ================================================================
@app.route('/console')
def console():
    """ISKRA-4 Web Console"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ISKRA-4 Emotional Weave Console</title>
        <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
        <style>
            body { font-family: monospace; margin: 20px; background: #0a0a0a; color: #0af; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: #111; border: 1px solid #333; padding: 20px; margin: 10px 0; border-radius: 5px; }
            .channel { display: inline-block; margin: 5px; padding: 8px 12px; background: #222; border-radius: 3px; }
            .channel.active { background: linear-gradient(45deg, #0af, #08f); color: white; }
            .emotion-form input, .emotion-form button { padding: 10px; margin: 5px; }
            #state-display { white-space: pre; font-size: 12px; }
            .metrics { font-size: 11px; color: #8af; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌌 ISKRA-4 EMOTIONAL WEAVE CONSOLE</h1>
            
            <div class="card">
                <h3>📊 Current Emotional State</h3>
                <div id="state-display">Loading...</div>
                <div id="channels"></div>
            </div>
            
            <div class="card">
                <h3>💫 Send Emotion</h3>
                <form class="emotion-form" onsubmit="sendEmotion(event)">
                    <input type="text" id="emotion" placeholder="love, joy, awe..." required>
                    <input type="range" id="intensity" min="0" max="1" step="0.1" value="0.5" 
                           oninput="document.getElementById('intensity-value').innerText=this.value">
                    <span id="intensity-value">0.5</span>
                    <button type="submit">Send</button>
                </form>
                <div id="response"></div>
            </div>
            
            <div class="card">
                <h3>📈 System Metrics</h3>
                <div id="metrics">Loading metrics...</div>
            </div>
            
            <div class="card">
                <h3>🔗 WebSocket Connection</h3>
                <div id="ws-status">Connecting...</div>
                <button onclick="getState()">Get Current State</button>
            </div>
        </div>
        
        <script>
            const socket = io();
            let currentState = {};
            
            // WebSocket events
            socket.on('connect', () => {
                document.getElementById('ws-status').innerHTML = 
                    '<span style="color:#0f0">✓ Connected</span>';
            });
            
            socket.on('disconnect', () => {
                document.getElementById('ws-status').innerHTML = 
                    '<span style="color:#f00">✗ Disconnected</span>';
            });
            
            socket.on('emotional_broadcast', (data) => {
                console.log('New emotion broadcast:', data);
                updateStateDisplay();
            });
            
            socket.on('current_state', (data) => {
                currentState = data.state;
                updateStateDisplay();
            });
            
            socket.on('emotional_response', (data) => {
                document.getElementById('response').innerHTML = 
                    `<div style="color:#8f8">✓ Emotion sent: ${data.emotion}</div>`;
            });
            
            socket.on('error', (data) => {
                document.getElementById('response').innerHTML = 
                    `<div style="color:#f88">✗ Error: ${data.error}</div>`;
            });
            
            // Functions
            function sendEmotion(event) {
                event.preventDefault();
                const emotion = document.getElementById('emotion').value;
                const intensity = parseFloat(document.getElementById('intensity').value);
                
                socket.emit('ws_emotion', {
                    emotion: emotion,
                    intensity: intensity,
                    context: { source: 'web_console' }
                });
                
                // Also send via REST for backup
                fetch('/emotional/input', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        emotion: emotion,
                        intensity: intensity,
                        context: { source: 'web_console' }
                    })
                });
            }
            
            function getState() {
                socket.emit('get_state');
            }
            
            function updateStateDisplay() {
                if (!currentState || !currentState.timestamp) {
                    fetch('/emotional/state')
                        .then(r => r.json())
                        .then(data => {
                            currentState = data.emotional_state;
                            renderState();
                        });
                    return;
                }
                renderState();
            }
            
            function renderState() {
                const state = currentState;
                let html = `<strong>Timestamp:</strong> ${state.timestamp}<br>`;
                html += `<strong>Resonance Index:</strong> ${state.resonance_index}<br>`;
                html += `<strong>Manifestation:</strong> ${state.manifestation}<br>`;
                html += `<strong>Active Channels:</strong><br>`;
                
                // Render channels
                const channelsDiv = document.getElementById('channels');
                channelsDiv.innerHTML = '';
                
                if (state.active_channels) {
                    for (const [channel, intensity] of Object.entries(state.active_channels)) {
                        const width = Math.max(20, intensity * 100);
                        const channelEl = document.createElement('div');
                        channelEl.className = 'channel' + (intensity > 0.1 ? ' active' : '');
                        channelEl.innerHTML = `
                            <strong>${channel}</strong><br>
                            <div style="background:#333; height:5px; width:100px;">
                                <div style="background:#0af; height:5px; width:${width}px;"></div>
                            </div>
                            ${intensity.toFixed(2)}
                        `;
                        channelsDiv.appendChild(channelEl);
                    }
                }
                
                document.getElementById('state-display').innerHTML = html;
                
                // Update metrics
                if (state.metrics) {
                    document.getElementById('metrics').innerHTML = `
                        <div class="metrics">
                            Cache Hits: ${state.metrics.harmony_cache_hits}<br>
                            Cache Misses: ${state.metrics.harmony_cache_misses}<br>
                            Hit Rate: ${((state.metrics.harmony_cache_hits / 
                                         (state.metrics.harmony_cache_hits + 
                                          state.metrics.harmony_cache_misses)) * 100).toFixed(1)}%<br>
                            Last Calc: ${state.metrics.last_calc_ms}ms
                        </div>
                    `;
                }
            }
            
            // Initial load
            updateStateDisplay();
            setInterval(updateStateDisplay, 5000); // Update every 5 seconds
            
            // Auto-decay simulation
            setInterval(() => {
                if (currentState && currentState.resonance_index > 0) {
                    fetch('/emotional/state').then(r => r.json()).then(data => {
                        currentState = data.emotional_state;
                        renderState();
                    });
                }
            }, 30000); // Check every 30 seconds
        </script>
    </body>
    </html>
    '''

# ================================================================
# STATIC FILES
# ================================================================
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

# ================================================================
# AUTO-DECAY BACKGROUND THREAD
# ================================================================
def auto_decay_thread():
    """Background thread for automatic emotional decay"""
    while True:
        try:
            # Trigger decay every 5 minutes
            time.sleep(300)
            weave.decay_channels()
            
            # Broadcast decay event
            state = weave.generate_state()
            socketio.emit('emotional_broadcast', {
                'event': 'auto_decay',
                'resonance_index': state['resonance_index'],
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            print(f"[AutoDecay] Error: {e}")
            time.sleep(60)

# Start auto-decay thread
decay_thread = threading.Thread(target=auto_decay_thread, daemon=True)
decay_thread.start()

# ================================================================
# MAIN ENTRY POINT
# ================================================================
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🌌 ISKRA-4 CLOUD v2.1")
    print("📦 DS24 PURE PROTOCOL + EMOTIONAL WEAVE v3.2")
    print("🌐 Web Console: http://localhost:5000/console")
    print("🔌 WebSocket: ws://localhost:5000")
    print("📡 REST API: http://localhost:5000/emotional/*")
    print("="*60 + "\n")
    
    print("[INIT] Emotional Weave initialized")
    print(f"[INIT] Memory loaded: {weave.memory.get_memory_size()} records")
    print("[INIT] Starting server...")
    
    # For production on Render, use port from environment
    port = int(os.environ.get('PORT', 5000))
    
    # Start Flask-SocketIO server
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=port,
        debug=False,  # Set to False for production
        allow_unsafe_werkzeug=True
    )
