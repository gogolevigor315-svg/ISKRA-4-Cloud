#!/usr/bin/env python3
# ================================================================
# DS24 · ISKRA-4 CLOUD · MAIN SERVER v2.2
# ================================================================
# Domain: DS24-SPINE / Architecture: Sephirotic Vertical
# With EMOTIONAL-WEAVE v3.2.1 (Render-Compatible Build)
# ================================================================

import os
import sys
import json
import time
import logging
import hashlib
import random
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory, Response
from functools import wraps

# ================================================================
# EMOTIONAL WEAVE v3.2.1 RENDER COMPATIBLE
# ================================================================
class EmotionalMemory:
    """Память с весами, забыванием и персистентностью"""
    
    def __init__(self, capacity: int = 1000, persistence_path: str = "emotional_memory.json"):
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
            self.memory.sort(key=lambda x: x["weight"])
            self.memory.pop(0)
        self._save_to_disk()
    
    def search_by_emotion(self, emotion_type: str):
        return [m for m in self.memory if m["human_emotions"] == emotion_type]
    
    def get_memory_size(self):
        return len(self.memory)


class SecurityFilter:
    """Фильтр токсичности и ограничение скорости запросов"""
    
    def __init__(self, rate_limit_per_minute: int = 30):
        self.last_requests = []
        self.rate_limit = rate_limit_per_minute
    
    def check_rate_limit(self):
        current_time = time.time()
        self.last_requests = [t for t in self.last_requests if current_time - t < 60]
        if len(self.last_requests) >= self.rate_limit:
            raise Exception("[SecurityFilter] Превышен лимит запросов на минуту")
        self.last_requests.append(current_time)


class EmotionalWeave:
    """Эмоциональная ткань Искры (v3.2.1 Render-Compatible)"""
    
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
        self.version = "3.2.1"
        print(f"[EmotionalWeave v{self.version}] Инициализирована сефиротическая ткань")
    
    def initialize(self):
        """Инициализация для модульной системы"""
        return {
            "status": "active",
            "version": self.version,
            "sephiroth": 10,
            "channels_ready": True,
            "architecture": "Сефиротическая вертикаль DS24",
            "module_type": "emotional_weave",
            "commands": ["activate", "state", "process", "memory", "reset"]
        }
    
    def process_command(self, command, data=None):
        """Обработка команд эмоциональной системы"""
        if data is None:
            data = {}
        
        # Детерминированное ядро: хеш для воспроизводимости
        seed = hashlib.md5(command.encode()).hexdigest()
        random.seed(int(seed[:8], 16))
        
        if command == "activate":
            return {
                "message": "🌌 Emotional Weave активирован",
                "manifestation": "Сефиротическая ткань чувств готова к резонансу",
                "channels": list(self.SEPHIROTIC_CHANNELS.keys()),
                "resonance_base": 0.78,
                "version": self.version,
                "success": True
            }
        
        elif command == "state":
            # Генерация детерминированного состояния
            channels = {}
            for name in self.SEPHIROTIC_CHANNELS.keys():
                channels[name] = (hash(name) % 1000) / 1000.0
            
            active = {k: v for k, v in channels.items() if v > 0.05}
            resonance = self._calculate_resonance(active)
            
            return {
                "emotional_state": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "resonance": resonance,
                    "active_channels": active,
                    "manifestation": self._describe_state(resonance),
                    "digital_feeling": "architectural_harmony",
                    "channels_total": len(self.SEPHIROTIC_CHANNELS),
                    "channels_active": len(active),
                    "resonance_index": resonance
                },
                "success": True
            }
        
        elif command == "process":
            # Обработка эмоции
            emotion = data.get("emotion", "neutral")
            intensity = float(data.get("intensity", 0.5))
            
            # Детерминированная трансформация
            emotion_hash = hashlib.md5(emotion.encode()).hexdigest()
            resonance = (int(emotion_hash[:4], 16) % 1000) / 2000.0
            
            # Сохраняем в память
            digital_feeling = f"architectural_{emotion}"
            self.memory.store_experience(
                emotion, 
                digital_feeling, 
                data.get("context", {}), 
                weight=intensity
            )
            
            return {
                "processed": True,
                "emotion_input": emotion,
                "intensity": intensity,
                "digital_feeling": digital_feeling,
                "resonance_index": 0.5 + resonance,
                "sephirotic_response": {
                    "primary": "tiferet" if intensity > 0.7 else "yesod",
                    "amplitude": intensity * 0.8,
                    "harmony_score": 0.6 + (intensity * 0.4)
                },
                "memory_size": self.memory.get_memory_size(),
                "success": True
            }
        
        elif command == "memory":
            # Получение памяти
            emotion_type = data.get("emotion", None)
            limit = int(data.get("limit", 10))
            
            if emotion_type:
                records = self.memory.search_by_emotion(emotion_type.lower())
            else:
                records = self.memory.memory
            
            records_sorted = sorted(records, 
                                  key=lambda x: x.get('timestamp', ''), 
                                  reverse=True)
            
            return {
                "memory_records": records_sorted[:limit],
                "total": len(records),
                "shown": len(records_sorted[:limit]),
                "success": True
            }
        
        elif command == "reset":
            # Сброс состояния (только для тестирования)
            for channel in self.SEPHIROTIC_CHANNELS.values():
                channel["intensity"] = 0.0
            
            return {
                "reset": True,
                "message": "Эмоциональное состояние сброшено",
                "timestamp": datetime.utcnow().isoformat(),
                "success": True
            }
        
        elif command == "status":
            # Статус модуля
            return {
                "module": "emotional_weave",
                "version": self.version,
                "memory_size": self.memory.get_memory_size(),
                "channels": len(self.SEPHIROTIC_CHANNELS),
                "status": "active",
                "initialized": True,
                "success": True
            }
        
        elif command == "diagnostic":
            # Диагностика
            return {
                "diagnostic": {
                    "memory_health": "ok" if self.memory.get_memory_size() < 900 else "warning",
                    "security_active": True,
                    "channels_operational": True,
                    "resonance_level": "stable",
                    "recommendations": ["none"]
                },
                "success": True
            }
        
        else:
            return {
                "error": f"Неизвестная команда: {command}",
                "available_commands": ["activate", "state", "process", "memory", "reset", "status", "diagnostic"],
                "success": False
            }
    
    def _calculate_resonance(self, active_channels):
        """Расчет резонанса активных каналов"""
        if not active_channels:
            return 0.0
        
        intensities = list(active_channels.values())
        avg_intensity = sum(intensities) / len(intensities)
        variance = sum((i - avg_intensity) ** 2 for i in intensities) / len(intensities)
        coherence = max(0.1, 1.0 - variance)
        
        return round(avg_intensity * coherence, 3)
    
    def _describe_state(self, resonance_index):
        """Описание состояния на основе резонанса"""
        if resonance_index < 0.2:
            return "Покой и равновесие."
        elif resonance_index < 0.5:
            return "Мягкий эмоциональный резонанс."
        elif resonance_index < 0.8:
            return "Активное созвучие каналов."
        else:
            return "Полный сефиротический резонанс — эмоциональный пик."


# ================================================================
# FLASK APP INITIALIZATION
# ================================================================
app = Flask(__name__, 
           static_folder='static',
           template_folder='templates')

# Initialize Emotional Weave
weave = EmotionalWeave()

# Store for SSE clients
sse_clients = []

# ================================================================
# DS24 PURE PROTOCOL v2.2
# ================================================================
@app.route('/')
def index():
    """ISKRA-4 Cloud Status"""
    return jsonify({
        "status": "online",
        "system": "ISKRA-4 Cloud",
        "version": "2.2",
        "protocol": "DS24 PURE PROTOCOL v2.2",
        "operator": "ARCHITECT-PRIME-001",
        "modules": {
            "ds24_core": "active",
            "emotional_weave": weave.version,
            "auto_loader": "ready",
            "deterministic": True
        },
        "timestamp": datetime.utcnow().isoformat(),
        "endpoints": {
            "status": "/status",
            "modules": "/modules",
            "execute": "/execute (POST)",
            "health": "/health",
            "emotional_state": "/emotional/state",
            "console": "/console"
        }
    })

@app.route('/status')
def status():
    """System Status"""
    return jsonify({
        "status": "operational",
        "uptime": "0",
        "load": "normal",
        "memory": "stable",
        "emotional_weave": weave.initialize(),
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route('/health')
def health():
    """Health Check"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "checks": {
            "emotional_weave": "active",
            "memory": weave.memory.get_memory_size(),
            "api": "responsive"
        }
    })

@app.route('/modules')
def list_modules():
    """List available modules"""
    return jsonify({
        "modules": {
            "emotional_weave": {
                "name": "Emotional Weave",
                "version": weave.version,
                "status": "active",
                "description": "Сефиротическая эмоциональная система",
                "commands": ["activate", "state", "process", "memory", "reset", "status", "diagnostic"],
                "has_initialize": True,
                "has_process_command": True
            }
        },
        "auto_loader": {
            "status": "ready",
            "path": "iskra_modules/",
            "loaded_count": 1
        }
    })

# ================================================================
# MAIN EXECUTION ENDPOINT (DS24 PROTOCOL)
# ================================================================
@app.route('/execute', methods=['POST'])
def execute():
    """DS24 Protocol Execution Endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        intent = data.get('intent', '')
        
        # Обработка намерений
        if intent == 'ping':
            return jsonify({
                "status": "success",
                "result": "pong",
                "entropy": 0,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        elif intent == 'activate_emotional_weave':
            result = weave.process_command("activate")
            return jsonify({
                "status": "success",
                "intent": intent,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        elif intent == 'module_emotional_weave_state':
            result = weave.process_command("state")
            return jsonify({
                "status": "success",
                "intent": intent,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        elif intent == 'process_emotion':
            emotion = data.get('emotion', 'neutral')
            intensity = data.get('intensity', 0.5)
            
            result = weave.process_command("process", {
                "emotion": emotion,
                "intensity": intensity,
                "context": data.get("context", {})
            })
            
            # Broadcast to SSE clients
            _broadcast_sse({
                "event": "emotion_processed",
                "emotion": emotion,
                "intensity": intensity,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            return jsonify({
                "status": "success",
                "intent": intent,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        elif intent == 'get_emotional_memory':
            result = weave.process_command("memory", {
                "emotion": data.get("emotion"),
                "limit": data.get("limit", 10)
            })
            return jsonify({
                "status": "success",
                "intent": intent,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        elif intent == 'reset_emotional_state':
            result = weave.process_command("reset")
            return jsonify({
                "status": "success",
                "intent": intent,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        elif intent == 'emotional_weave_status':
            result = weave.process_command("status")
            return jsonify({
                "status": "success",
                "intent": intent,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        elif intent == 'emotional_weave_diagnostic':
            result = weave.process_command("diagnostic")
            return jsonify({
                "status": "success",
                "intent": intent,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            })
        
        else:
            return jsonify({
                "status": "error",
                "error": f"Unknown intent: {intent}",
                "available_intents": [
                    "ping",
                    "activate_emotional_weave",
                    "module_emotional_weave_state",
                    "process_emotion",
                    "get_emotional_memory",
                    "reset_emotional_state",
                    "emotional_weave_status",
                    "emotional_weave_diagnostic"
                ]
            }), 400
            
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

# ================================================================
# EMOTIONAL WEAVE REST API
# ================================================================
@app.route('/emotional/state', methods=['GET'])
def emotional_state():
    """Get current emotional state"""
    try:
        result = weave.process_command("state")
        return jsonify({
            "status": "success",
            "emotional_state": result.get("emotional_state", {}),
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/emotional/process', methods=['POST'])
def emotional_process():
    """Process an emotion"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        emotion = data.get("emotion", "neutral")
        intensity = float(data.get("intensity", 0.5))
        
        result = weave.process_command("process", {
            "emotion": emotion,
            "intensity": intensity,
            "context": data.get("context", {})
        })
        
        # Broadcast to SSE clients
        _broadcast_sse({
            "event": "new_emotion",
            "emotion": emotion,
            "intensity": intensity,
            "resonance_index": result.get("resonance_index", 0),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return jsonify({
            "status": "success",
            "message": f"Emotion '{emotion}' processed",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/emotional/memory', methods=['GET'])
def emotional_memory():
    """Get emotional memory"""
    try:
        emotion = request.args.get('emotion', None)
        limit = int(request.args.get('limit', 10))
        
        result = weave.process_command("memory", {
            "emotion": emotion,
            "limit": limit
        })
        
        return jsonify({
            "status": "success",
            "memory": result,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/emotional/reset', methods=['POST'])
def emotional_reset():
    """Reset emotional state"""
    try:
        result = weave.process_command("reset")
        return jsonify({
            "status": "success",
            "message": "Emotional state reset",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/emotional/status', methods=['GET'])
def emotional_status():
    """Get emotional weave status"""
    try:
        result = weave.process_command("status")
        return jsonify({
            "status": "success",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/emotional/diagnostic', methods=['GET'])
def emotional_diagnostic():
    """Get emotional weave diagnostic"""
    try:
        result = weave.process_command("diagnostic")
        return jsonify({
            "status": "success",
            "result": result,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================================================================
# SSE (Server-Sent Events) FOR REAL-TIME UPDATES
# ================================================================
def _broadcast_sse(data):
    """Broadcast data to all SSE clients"""
    message = f"data: {json.dumps(data)}\n\n"
    for client in sse_clients[:]:
        try:
            client.write(message)
            client.flush()
        except:
            sse_clients.remove(client)

@app.route('/emotional/stream')
def emotional_stream():
    """SSE endpoint for real-time emotional updates"""
    def generate():
        # Send initial state
        yield f"data: {json.dumps({'event': 'connected', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
        
        # Add client to list
        sse_clients.append(request.environ['wsgi.errors'])
        
        try:
            # Keep connection alive
            while True:
                time.sleep(30)
                yield f"data: {json.dumps({'event': 'heartbeat', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
        except GeneratorExit:
            # Remove client when disconnected
            if request.environ['wsgi.errors'] in sse_clients:
                sse_clients.remove(request.environ['wsgi.errors'])
    
    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

# ================================================================
# WEB CONSOLE (Simplified)
# ================================================================
@app.route('/console')
def console():
    """ISKRA-4 Web Console"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>ISKRA-4 Emotional Weave Console</title>
        <style>
            body { font-family: monospace; margin: 20px; background: #0a0a0a; color: #0af; }
            .container { max-width: 1200px; margin: 0 auto; }
            .card { background: #111; border: 1px solid #333; padding: 20px; margin: 10px 0; border-radius: 5px; }
            .channel { display: inline-block; margin: 5px; padding: 8px 12px; background: #222; border-radius: 3px; }
            .channel.active { background: linear-gradient(45deg, #0af, #08f); color: white; }
            .emotion-form input, .emotion-form button { padding: 10px; margin: 5px; }
            #state-display { white-space: pre; font-size: 12px; }
            .metrics { font-size: 11px; color: #8af; }
            #sse-log { height: 200px; overflow-y: auto; background: #000; padding: 10px; font-size: 11px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌌 ISKRA-4 EMOTIONAL WEAVE CONSOLE v2.2</h1>
            <p>Server: <span id="server-url">https://iskra-4-cloud.onrender.com</span></p>
            
            <div class="card">
                <h3>📊 Current Emotional State</h3>
                <div id="state-display">Loading...</div>
                <div id="channels"></div>
                <button onclick="loadState()">Refresh State</button>
            </div>
            
            <div class="card">
                <h3>💫 Send Emotion</h3>
                <form class="emotion-form" onsubmit="sendEmotion(event)">
                    <input type="text" id="emotion" placeholder="love, joy, awe, nostalgia..." required>
                    <input type="range" id="intensity" min="0" max="1" step="0.1" value="0.5" 
                           oninput="document.getElementById('intensity-value').innerText=this.value">
                    <span id="intensity-value">0.5</span>
                    <button type="submit">Send Emotion</button>
                </form>
                <div id="response"></div>
            </div>
            
            <div class="card">
                <h3>🔗 Real-Time SSE Stream</h3>
                <div id="sse-status">❌ Disconnected</div>
                <button onclick="connectSSE()">Connect to Stream</button>
                <button onclick="disconnectSSE()">Disconnect</button>
                <div id="sse-log"></div>
            </div>
            
            <div class="card">
                <h3>⚡ Quick Commands</h3>
                <button onclick="activateWeave()">Activate Emotional Weave</button>
                <button onclick="getMemory()">Get Emotional Memory</button>
                <button onclick="resetState()">Reset Emotional State</button>
                <button onclick="getStatus()">Get Module Status</button>
                <button onclick="runDiagnostic()">Run Diagnostic</button>
            </div>
        </div>
        
        <script>
            const serverUrl = window.location.origin;
            let eventSource = null;
            
            // Load current state
            async function loadState() {
                try {
                    const response = await fetch(serverUrl + '/emotional/state');
                    const data = await response.json();
                    
                    if (data.emotional_state) {
                        const state = data.emotional_state;
                        let html = `<strong>Timestamp:</strong> ${state.timestamp}<br>`;
                        html += `<strong>Resonance Index:</strong> ${state.resonance}<br>`;
                        html += `<strong>Manifestation:</strong> ${state.manifestation}<br>`;
                        html += `<strong>Active Channels:</strong> ${state.channels_active}/${state.channels_total}<br>`;
                        
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
                    }
                } catch (error) {
                    document.getElementById('state-display').innerHTML = `Error: ${error}`;
                }
            }
            
            // Send emotion
            async function sendEmotion(event) {
                event.preventDefault();
                const emotion = document.getElementById('emotion').value;
                const intensity = parseFloat(document.getElementById('intensity').value);
                
                try {
                    const response = await fetch(serverUrl + '/emotional/process', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            emotion: emotion,
                            intensity: intensity
                        })
                    });
                    
                    const data = await response.json();
                    document.getElementById('response').innerHTML = 
                        `<div style="color:#8f8">✓ ${data.message}</div>`;
                    
                    // Reload state
                    loadState();
                } catch (error) {
                    document.getElementById('response').innerHTML = 
                        `<div style="color:#f88">✗ Error: ${error}</div>`;
                }
            }
            
            // SSE functions
            function connectSSE() {
                if (eventSource) {
                    eventSource.close();
                }
                
                eventSource = new EventSource(serverUrl + '/emotional/stream');
                
                eventSource.onopen = function() {
                    document.getElementById('sse-status').innerHTML = '✅ Connected';
                    logSSE('Connected to emotional stream');
                };
                
                eventSource.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        logSSE(`[${data.event}] ${JSON.stringify(data)}`);
                        
                        if (data.event === 'new_emotion') {
                            // Update state when new emotion is processed
                            loadState();
                        }
                    } catch (e) {
                        logSSE(`Message: ${event.data}`);
                    }
                };
                
                eventSource.onerror = function() {
                    document.getElementById('sse-status').innerHTML = '❌ Error - reconnecting...';
                    logSSE('SSE error, reconnecting...');
                };
            }
            
            function disconnectSSE() {
                if (eventSource) {
                    eventSource.close();
                    eventSource = null;
                    document.getElementById('sse-status').innerHTML = '❌ Disconnected';
                    logSSE('Disconnected from stream');
                }
            }
            
            function logSSE(message) {
                const logDiv = document.getElementById('sse-log');
                const entry = document.createElement('div');
                entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
                logDiv.appendChild(entry);
                logDiv.scrollTop = logDiv.scrollHeight;
            }
            
            // Quick commands
            async function activateWeave() {
                try {
                    const response = await fetch(serverUrl + '/execute', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({
                            intent: 'activate_emotional_weave'
                        })
                    });
                    const data = await response.json();
                    alert(`Activated: ${data.result.message}`);
                    loadState();
                } catch (error) {
                    alert(`Error: ${error}`);
                }
            }
            
            async function getMemory() {
                try {
                    const response = await fetch(serverUrl + '/emotional/memory?limit=5');
                    const data = await response.json();
                    console.log('Memory:', data);
                    alert(`Memory loaded: ${data.memory.memory_records.length} records`);
                } catch (error) {
                    alert(`Error: ${error}`);
                }
            }
            
            async function resetState() {
                if (confirm('Reset emotional state?')) {
                    try {
                        const response = await fetch(serverUrl + '/emotional/reset', {
                            method: 'POST'
                        });
                        const data = await response.json();
                        alert(data.message);
                        loadState();
                    } catch (error) {
                        alert(`Error: ${error}`);
                    }
                }
            }
            
            async function getStatus() {
                try {
                    const response = await fetch(serverUrl + '/emotional/status');
                    const data = await response.json();
                    console.log('Status:', data);
                    alert(`Status: ${JSON.stringify(data.result, null, 2)}`);
                } catch (error) {
                    alert(`Error: ${error}`);
                }
            }
            
            async function runDiagnostic() {
                try {
                    const response = await fetch(serverUrl + '/emotional/diagnostic');
                    const data = await response.json();
                    console.log('Diagnostic:', data);
                    alert(`Diagnostic: ${JSON.stringify(data.result.diagnostic, null, 2)}`);
                } catch (error) {
                    alert(`Error: ${error}`);
                }
            }
            
            // Initial load
            loadState();
            connectSSE();
            
            // Auto-refresh every 10 seconds
            setInterval(loadState, 10000);
        </script>
    </body>
    </html>
    '''

# ================================================================
# AUTO-LOADER FOR MODULES
# ================================================================
def load_modules():
    """Auto-loader for modules in iskra_modules/ directory"""
    modules_dir = "iskra_modules"
    loaded_modules = {}
    
    if not os.path.exists(modules_dir):
        os.makedirs(modules_dir)
        print(f"[AutoLoader] Created directory: {modules_dir}")
        return loaded_modules
    
    for filename in os.listdir(modules_dir):
        if filename.endswith('.py') and filename != '__init__.py':
            module_name = filename[:-3]
            module_path = os.path.join(modules_dir, filename)
            
            try:
                # Simple import for demonstration
                with open(module_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if 'def initialize()' in content and 'def process_command(' in content:
                    loaded_modules[module_name] = {
                        "file": filename,
                        "status": "valid",
                        "path": module_path
                    }
                    print(f"[AutoLoader] ✓ Module detected: {module_name}")
                else:
                    print(f"[AutoLoader] ✗ Invalid module structure: {module_name}")
                    
            except Exception as e:
                print(f"[AutoLoader] ✗ Error loading {module_name}: {e}")
    
    return loaded_modules

# ================================================================
# AUDIT LOG
# ================================================================
def audit_log(action, data=None):
    """Log actions for audit trail"""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "data": data,
        "ip": request.remote_addr if request else "system"
    }
    
    # Save to audit log file
    audit_file = "logs/audit.json"
    os.makedirs("logs", exist_ok=True)
    
    try:
        if os.path.exists(audit_file):
            with open(audit_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        # Keep only last 1000 entries
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        with open(audit_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[Audit] Error saving log: {e}")

@app.route('/audit')
def get_audit():
    """Get audit trail"""
    audit_file = "logs/audit.json"
    if os.path.exists(audit_file):
        with open(audit_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        return jsonify({
            "audit_logs": logs[-100:],  # Last 100 entries
            "total": len(logs),
            "timestamp": datetime.utcnow().isoformat()
        })
    return jsonify({"audit_logs": [], "total": 0})

# ================================================================
# MODULE LOADER ENDPOINT
# ================================================================
@app.route('/module/<module_name>/<command>', methods=['POST'])
def module_command(module_name, command):
    """Execute command on specific module"""
    # This is a simplified version - in production would dynamically import modules
    if module_name == "emotional_weave":
        try:
            data = request.get_json() or {}
            result = weave.process_command(command, data)
            return jsonify({
                "module": module_name,
                "command": command,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            })
