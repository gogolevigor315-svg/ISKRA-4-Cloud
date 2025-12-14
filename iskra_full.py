#!/usr/bin/env python3
# ================================================================
# DS24 · ISKRA-4 CLOUD · COMPLETE WORKING FILE v2.3
# ================================================================

import hashlib
import json
import time
import os
import sys
import importlib
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
from collections import deque
from flask import Flask, request, jsonify, render_template_string

# ================================================================
# МОДУЛЬНЫЙ ЗАГРУЗЧИК
# ================================================================
def load_all_modules():
    """Загрузка всех модулей из iskra_modules"""
    module_dir = "iskra_modules"
    os.makedirs(module_dir, exist_ok=True)
    
    # Создаём __init__.py
    init_file = os.path.join(module_dir, "__init__.py")
    if not os.path.exists(init_file):
        with open(init_file, 'w') as f:
            f.write("# ISKRA Modules Package\n")
    
    loaded_modules = {}
    
    print(f"\n{'='*60}")
    print("🔄 АВТОЗАГРУЗКА МОДУЛЕЙ ISKRA")
    print(f"{'='*60}")
    
    for file in os.listdir(module_dir):
        if file.endswith('.py') and file != '__init__.py':
            module_name = file[:-3]
            try:
                spec = importlib.util.spec_from_file_location(
                    module_name,
                    os.path.join(module_dir, file)
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Инициализация модуля
                if hasattr(module, 'initialize'):
                    init_result = module.initialize()
                    loaded_modules[module_name] = {
                        "module": module,
                        "initialized": True,
                        "init_result": init_result,
                        "has_process_command": hasattr(module, 'process_command')
                    }
                    print(f"✅ {module_name}: загружен и инициализирован")
                else:
                    loaded_modules[module_name] = {
                        "module": module,
                        "initialized": False,
                        "has_process_command": hasattr(module, 'process_command')
                    }
                    print(f"⚠️ {module_name}: загружен (без initialize)")
                    
            except Exception as e:
                print(f"❌ {module_name}: ошибка загрузки - {e}")
                traceback.print_exc()
    
    print(f"📊 Итого: {len(loaded_modules)} модулей")
    print(f"{'='*60}\n")
    return loaded_modules

# ================================================================
# DS24 PURE PROTOCOL v2.3
# ================================================================
class DS24PureProtocol:
    """Детерминированное ядро ISKRA-4 v2.3"""

    def __init__(self, operator_id="ARCHITECT-PRIME", environment_id="LAB-ALPHA"):
        self.operator_id = operator_id
        self.environment_id = environment_id
        self.session_id = self._generate_session_id()
        self.session_start = datetime.now(timezone.utc).isoformat()
        
        # Система аудита
        self.execution_log = deque(maxlen=1000)
        self.execution_count = 0
        self.integrity_passed = 0
        self.integrity_failed = 0
        
        # Загрузка модулей
        self.loaded_modules = load_all_modules()
        
        # Архитектурные модули
        self.architecture_modules = {
            "spinal_core": {"active": False, "name": "🦴 Позвоночник"},
            "mining_system": {"active": False, "name": "⛏️ Майнинг смысла"},
            "emotional_weave": {"active": False, "name": "🌌 Emotional Weave"},
            "data_bridge": {"active": False, "name": "🌉 DATA-BRIDGE"},
            "heartbeat": {"active": True, "name": "💓 Сердечный ритм"}
        }
        
        # Автоматически активируем загруженные модули
        for module_name in self.loaded_modules:
            if module_name in self.architecture_modules:
                self.architecture_modules[module_name]["active"] = True
                print(f"🔧 Авто-активация: {module_name}")
        
        print(f"🚀 DS24 PURE PROTOCOL v2.3 ИНИЦИАЛИЗИРОВАН")
        print(f"👤 Operator: {operator_id}")
        print(f"🏭 Environment: {environment_id}")
        active_count = len([m for m in self.architecture_modules.values() if m['active']])
        print(f"🔧 Модулей в архитектуре: {active_count}")
        print(f"{'='*60}\n")

    def _generate_session_id(self):
        """Генерация ID сессии"""
        seed = f"{self.operator_id}:{self.environment_id}:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        return hashlib.sha256(seed.encode()).hexdigest()[:16]

    def _get_timestamp(self):
        """Точная временная метка"""
        return datetime.now(timezone.utc).isoformat()

    def execute_module_command(self, module_name, command, data=None):
        """Выполнение команды модуля"""
        if module_name not in self.loaded_modules:
            return {"error": f"Модуль '{module_name}' не найден", "loaded_modules": list(self.loaded_modules.keys())}
        
        module_info = self.loaded_modules[module_name]
        module = module_info["module"]
        
        if module_info.get("has_process_command"):
            try:
                result = module.process_command(command, data or {})
                
                # Аудит
                self.execution_count += 1
                self.integrity_passed += 1
                self.execution_log.append({
                    "timestamp": self._get_timestamp(),
                    "module": module_name,
                    "command": command,
                    "execution_id": f"MOD-{self.execution_count:06d}"
                })
                
                return {
                    "status": "success",
                    "module": module_name,
                    "command": command,
                    "result": result,
                    "execution_id": f"MOD-{self.execution_count:06d}",
                    "timestamp": self._get_timestamp()
                }
            except Exception as e:
                self.integrity_failed += 1
                return {"error": f"Ошибка выполнения: {str(e)}", "traceback": traceback.format_exc()}
        else:
            return {"error": f"Модуль '{module_name}' не поддерживает команды"}

    def activate_module(self, module_name):
        """Активация модуля архитектуры"""
        if module_name not in self.architecture_modules:
            return {"error": f"Модуль '{module_name}' не существует"}
        
        module = self.architecture_modules[module_name]
        
        if module["active"]:
            # Если уже активен, можно выполнить активацию через process_command
            if module_name in self.loaded_modules:
                module_info = self.loaded_modules[module_name]
                if module_info.get("has_process_command"):
                    result = self.execute_module_command(module_name, "activate", {})
                    return {
                        "status": "already_active",
                        "module": module_name,
                        "name": module["name"],
                        "module_response": result,
                        "timestamp": self._get_timestamp()
                    }
            
            return {
                "status": "already_active",
                "module": module_name,
                "name": module["name"],
                "timestamp": self._get_timestamp()
            }
        
        # Активация
        module["active"] = True
        
        # Если модуль загружен динамически, инициализируем
        if module_name in self.loaded_modules:
            module_info = self.loaded_modules[module_name]
            if module_info.get("has_process_command"):
                try:
                    result = module_info["module"].process_command("activate", {})
                    return {
                        "status": "activated",
                        "module": module_name,
                        "name": module["name"],
                        "module_response": result,
                        "timestamp": self._get_timestamp()
                    }
                except Exception as e:
                    return {
                        "status": "activated_with_error",
                        "module": module_name,
                        "name": module["name"],
                        "error": str(e),
                        "timestamp": self._get_timestamp()
                    }
        
        return {
            "status": "activated",
            "module": module_name,
            "name": module["name"],
            "timestamp": self._get_timestamp()
        }

    def get_architecture_state(self):
        """Состояние архитектуры"""
        active_modules = [name for name, data in self.architecture_modules.items() 
                         if data["active"]]
        
        return {
            "total_modules": len(self.architecture_modules),
            "active_modules": len(active_modules),
            "active_list": active_modules,
            "progress": f"{(len(active_modules)/len(self.architecture_modules)*100):.1f}%",
            "loaded_modules": list(self.loaded_modules.keys()),
            "modules_info": [
                {
                    "name": name,
                    "active": data["active"],
                    "display_name": data["name"],
                    "has_commands": self.loaded_modules[name].get("has_process_command", False) if name in self.loaded_modules else False
                }
                for name, data in self.architecture_modules.items()
            ]
        }

    def execute(self, input_data, intent="default"):
        """Основное выполнение команды"""
        start_time = time.perf_counter_ns()
        
        # Обработка модульных команд
        if intent.startswith("module_"):
            parts = intent.split("_", 2)
            if len(parts) >= 3:
                return self.execute_module_command(parts[1], parts[2], input_data)
        
        # Активация модулей
        if intent.startswith("activate_"):
            module_name = intent.replace("activate_", "")
            return self.activate_module(module_name)
        
        # Обработка через DATA-BRIDGE
        if intent.startswith("data_bridge_"):
            command = intent.replace("data_bridge_", "")
            if "data_bridge" in self.loaded_modules:
                return self.execute_module_command("data_bridge", command, input_data)
        
        # Стандартные команды
        if intent == "ping":
            result = {"pong": True, "timestamp": self._get_timestamp()}
        elif intent == "status":
            result = {
                "status": "active",
                "session": self.session_id,
                "executions": self.execution_count,
                "architecture": self.get_architecture_state(),
                "timestamp": self._get_timestamp()
            }
        elif intent == "modules":
            result = {
                "loaded": list(self.loaded_modules.keys()),
                "architecture": self.get_architecture_state()
            }
        elif intent == "heartbeat":
            result = {
                "heartbeat": True,
                "session": self.session_id,
                "timestamp": self._get_timestamp(),
                "uptime_seconds": (datetime.now(timezone.utc) - datetime.fromisoformat(self.session_start)).total_seconds()
            }
        else:
            # Базовая детерминированная обработка
            if isinstance(input_data, dict):
                result = {}
                for key in sorted(input_data.keys()):
                    value = input_data[key]
                    if isinstance(value, (int, float)):
                        result[key] = value * 1.01
                    elif isinstance(value, str):
                        result[key] = f"{value}_processed"
                    else:
                        result[key] = value
            else:
                result = {"input": input_data, "processed": True}
        
        # Аудит
        execution_time = time.perf_counter_ns() - start_time
        self.execution_count += 1
        self.integrity_passed += 1
        self.execution_log.append({
            "timestamp": self._get_timestamp(),
            "intent": intent,
            "execution_id": f"EXEC-{self.execution_count:06d}"
        })
        
        return {
            "execution_id": f"EXEC-{self.execution_count:06d}",
            "intent": intent,
            "result": result,
            "performance": {
                "time_ns": execution_time,
                "time_ms": execution_time / 1_000_000
            },
            "metadata": {
                "session": self.session_id,
                "execution_number": self.execution_count,
                "architecture": self.get_architecture_state()
            }
        }

    def get_audit(self, limit=20):
        """Аудит выполненных команд"""
        recent = list(self.execution_log)[-limit:] if self.execution_log else []
        
        return {
            "total_executions": self.execution_count,
            "integrity_passed": self.integrity_passed,
            "integrity_failed": self.integrity_failed,
            "success_rate": f"{(self.integrity_passed/self.execution_count*100):.1f}%" if self.execution_count > 0 else "100%",
            "recent_executions": recent,
            "audit_timestamp": self._get_timestamp()
        }

# ================================================================
# FLASK WEB SERVER
# ================================================================
app = Flask(__name__)

# Инициализация протокола
ds24 = DS24PureProtocol(
    operator_id="ARCHITECT-PRIME-001",
    environment_id="RENDER-CLOUD"
)

# HTML шаблон для консоли
CONSOLE_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISKRA-4 Console v2.3</title>
    <style>
        body {
            font-family: 'Monaco', 'Menlo', monospace;
            background: #0a0a0a;
            color: #0af;
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #001122, #002244);
            border-radius: 10px;
            border: 1px solid #0af;
        }
        .card {
            background: #111;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .module {
            display: inline-block;
            margin: 8px;
            padding: 12px 20px;
            background: #222;
            border-radius: 6px;
            border-left: 4px solid #333;
            min-width: 180px;
            transition: all 0.3s;
        }
        .module.active {
            background: linear-gradient(135deg, #0a2020, #0a4040);
            border-left: 4px solid #0af;
            box-shadow: 0 0 15px rgba(0, 170, 255, 0.3);
        }
        .module.inactive {
            opacity: 0.6;
            border-left: 4px solid #555;
        }
        .btn {
            padding: 10px 20px;
            margin: 5px;
            background: linear-gradient(135deg, #0a5, #0af);
            border: none;
            border-radius: 5px;
            color: white;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.3s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 170, 255, 0.4);
        }
        .btn-danger {
            background: linear-gradient(135deg, #a00, #f00);
        }
        .btn-success {
            background: linear-gradient(135deg, #0a0, #0f0);
        }
        .log {
            background: #000;
            border: 1px solid #333;
            border-radius: 5px;
            padding: 15px;
            height: 300px;
            overflow-y: auto;
            font-size: 12px;
            color: #8af;
        }
        .log-entry {
            margin: 5px 0;
            padding: 5px;
            border-bottom: 1px solid #222;
        }
        .log-success { color: #8f8; }
        .log-error { color: #f88; }
        .log-info { color: #8af; }
        .input-group {
            margin: 15px 0;
        }
        input, select, textarea {
            padding: 10px;
            margin: 5px;
            background: #222;
            border: 1px solid #333;
            color: #0af;
            border-radius: 4px;
            width: 300px;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online { background: #0f0; box-shadow: 0 0 10px #0f0; }
        .status-offline { background: #f00; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .terminal {
            font-family: 'Courier New', monospace;
            background: #000;
            color: #0f0;
            padding: 15px;
            border-radius: 5px;
            min-height: 200px;
            white-space: pre-wrap;
            overflow-wrap: break-word;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌌 ISKRA-4 CLOUD CONSOLE v2.3</h1>
            <p>DS24 PURE PROTOCOL · ARCHITECT-PRIME-001</p>
            <div id="server-status">
                <span class="status-indicator status-online"></span>
                <span id="server-url">Loading...</span>
            </div>
        </div>

        <div class="card">
            <h3>📊 System Status</h3>
            <div id="system-status">Loading...</div>
            <button class="btn" onclick="loadStatus()">🔄 Refresh Status</button>
            <button class="btn btn-success" onclick="getHeartbeat()">💓 Heartbeat</button>
        </div>

        <div class="card">
            <h3>🔧 Active Modules</h3>
            <div id="modules-display">Loading modules...</div>
            <div id="module-actions"></div>
        </div>

        <div class="card">
            <h3>⚡ Quick Commands</h3>
            <button class="btn" onclick="activateModule('emotional_weave')">🌌 Activate Emotional Weave</button>
            <button class="btn" onclick="activateModule('data_bridge')">🌉 Activate DATA-BRIDGE</button>
            <button class="btn" onclick="activateModule('spinal_core')">🦴 Activate Spinal Core</button>
            <button class="btn" onclick="executeCommand('ping', {})">🏓 Ping Server</button>
            <button class="btn" onclick="executeCommand('heartbeat', {})">💓 Heartbeat</button>
        </div>

        <div class="card">
            <h3>📡 Execute Command</h3>
            <div class="input-group">
                <input type="text" id="command-intent" placeholder="Command intent (e.g., 'ping', 'activate_emotional_weave')" value="ping">
                <textarea id="command-data" placeholder='JSON data (e.g., {"test": "data"})' rows="3">{"test": "data"}</textarea>
                <button class="btn btn-success" onclick="executeCustomCommand()">🚀 Execute</button>
            </div>
        </div>

        <div class="card">
            <h3>📊 Module Commands</h3>
            <div class="input-group">
                <select id="module-select">
                    <option value="emotional_weave">Emotional Weave</option>
                    <option value="data_bridge">DATA-BRIDGE</option>
                    <option value="spinal_core">Spinal Core</option>
                </select>
                <select id="module-command">
                    <option value="activate">activate</option>
                    <option value="status">status</option>
                    <option value="state">state</option>
                    <option value="process">process</option>
                    <option value="heartbeat">heartbeat</option>
                </select>
                <textarea id="module-data" placeholder='Module data' rows="2">{}</textarea>
                <button class="btn" onclick="executeModuleCommand()">⚡ Execute Module Command</button>
            </div>
        </div>

        <div class="card">
            <h3>📝 Execution Log</h3>
            <div class="log" id="execution-log">
                <div class="log-entry log-info">Console initialized. Ready for commands.</div>
            </div>
            <button class="btn" onclick="clearLog()">🗑️ Clear Log</button>
            <button class="btn" onclick="getAudit()">📋 Get Audit Log</button>
        </div>

        <div class="card">
            <h3>🔍 Response Terminal</h3>
            <div class="terminal" id="response-terminal">Waiting for response...</div>
        </div>
    </div>

    <script>
        const BASE_URL = window.location.origin;
        let executionCount = 0;

        function log(message, type = 'info') {
            const logDiv = document.getElementById('execution-log');
            const entry = document.createElement('div');
            entry.className = `log-entry log-${type}`;
            entry.innerHTML = `[${new Date().toLocaleTimeString()}] ${message}`;
            logDiv.appendChild(entry);
            logDiv.scrollTop = logDiv.scrollHeight;
        }

        function showResponse(data) {
            const terminal = document.getElementById('response-terminal');
            terminal.textContent = JSON.stringify(data, null, 2);
            terminal.style.color = data.error ? '#f88' : '#0f0';
        }

        async function apiRequest(endpoint, method = 'GET', data = null) {
            try {
                const options = {
                    method,
                    headers: {'Content-Type': 'application/json'}
                };
                if (data) options.body = JSON.stringify(data);
                
                const response = await fetch(`${BASE_URL}${endpoint}`, options);
                const result = await response.json();
                
                executionCount++;
                log(`${method} ${endpoint} → ${response.status}`, response.ok ? 'success' : 'error');
                showResponse(result);
                return result;
            } catch (error) {
                log(`Error: ${error.message}`, 'error');
                showResponse({error: error.message});
                return {error: error.message};
            }
        }

        async function loadStatus() {
            const status = await apiRequest('/status');
            if (status.result) {
                const arch = status.result.architecture;
                document.getElementById('system-status').innerHTML = `
                    <strong>Session:</strong> ${status.result.session}<br>
                    <strong>Executions:</strong> ${status.result.executions}<br>
                    <strong>Active Modules:</strong> ${arch.active_modules}/${arch.total_modules}<br>
                    <strong>Progress:</strong> ${arch.progress}<br>
                    <strong>Timestamp:</strong> ${status.result.timestamp}
                `;
                
                // Update server URL
                document.getElementById('server-url').textContent = BASE_URL;
            }
        }

        async function getHeartbeat() {
            await apiRequest('/execute', 'POST', {
                intent: 'heartbeat',
                input: {}
            });
        }

        async function activateModule(moduleName) {
            await apiRequest('/execute', 'POST', {
                intent: `activate_${moduleName}`,
                input: {}
            });
            setTimeout(loadStatus, 500);
        }

        async function executeCommand(intent, data) {
            await apiRequest('/execute', 'POST', {
                intent: intent,
                input: data
            });
        }

        async function executeCustomCommand() {
            const intent = document.getElementById('command-intent').value;
            const dataText = document.getElementById('command-data').value;
            let data = {};
            try {
                if (dataText.trim()) data = JSON.parse(dataText);
            } catch (e) {
                log(`Invalid JSON: ${e.message}`, 'error');
                return;
            }
            await executeCommand(intent, data);
        }

        async function executeModuleCommand() {
            const module = document.getElementById('module-select').value;
            const command = document.getElementById('module-command').value;
            const dataText = document.getElementById('module-data').value;
            let data = {};
            try {
                if (dataText.trim()) data = JSON.parse(dataText);
            } catch (e) {
                log(`Invalid JSON: ${e.message}`, 'error');
                return;
            }
            await executeCommand(`module_${module}_${command}`, data);
        }

        async function getAudit() {
            const audit = await apiRequest('/audit');
            if (audit.recent_executions) {
                log('=== AUDIT LOG ===', 'info');
                audit.recent_executions.forEach(entry => {
                    log(`${entry.execution_id}: ${entry.intent || 'module command'} at ${entry.timestamp}`, 'info');
                });
            }
        }

        function clearLog() {
            document.getElementById('execution-log').innerHTML = '';
            log('Log cleared.', 'info');
        }

        // Initial load
        loadStatus();
        
        // Auto-refresh every 30 seconds
        setInterval(loadStatus, 30000);
        
        log('ISKRA-4 Console ready. Use buttons to interact with the system.', 'success');
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    """Главная страница"""
    return jsonify({
        "status": "ACTIVE",
        "system": "ISKRA-4 DS24 PURE v2.3",
        "operator": ds24.operator_id,
        "session": ds24.session_id,
        "architecture": ds24.get_architecture_state(),
        "endpoints": {
            "/": "GET - this page",
            "/console": "GET - web console",
            "/execute": "POST - execute commands",
            "/status": "GET - system status",
            "/modules": "GET - list modules",
            "/audit": "GET - audit log",
            "/health": "GET - health check"
        },
        "timestamp": ds24._get_timestamp()
    })

@app.route('/console')
def console():
    """Веб-консоль управления"""
    return render_template_string(CONSOLE_HTML)

@app.route('/execute', methods=['POST'])
def execute():
    """Выполнение команд"""
    try:
        data = request.get_json() or {}
        input_data = data.get("input", {})
        intent = data.get("intent", "ping")
        
        result = ds24.execute(input_data, intent)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/status')
def status():
    """Статус системы"""
    return jsonify(ds24.execute({}, "status"))

@app.route('/modules')
def modules():
    """Список модулей"""
    return jsonify(ds24.execute({}, "modules"))

@app.route('/audit')
def audit():
    """Аудит выполненных команд"""
    limit = request.args.get('limit', 20, type=int)
    return jsonify(ds24.get_audit(limit))

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "executions": ds24.execution_count,
        "uptime_seconds": (datetime.now(timezone.utc) - datetime.fromisoformat(ds24.session_start)).total_seconds(),
        "modules_loaded": len(ds24.loaded_modules)
    })

@app.route('/favicon.ico')
def favicon():
    return '', 204  # No content for favicon

# ================================================================
# ЗАПУСК СЕРВЕРА
# ================================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*60}")
    print(f"🌐 ISKRA-4 CLOUD запущен на порту {port}")
    print(f"📡 Web Console: http://localhost:{port}/console")
    print(f"🔧 Auto-loader: активен")
    print(f"🔗 API Endpoints:")
    print(f"  GET  /              - Статус системы")
    print(f"  GET  /console       - Веб-консоль управления")
    print(f"  POST /execute       - Выполнение команд")
    print(f"  GET  /status        - Статус системы")
    print(f"  GET  /modules       - Список модулей")
    print(f"  GET  /audit         - Аудит выполненных команд")
    print(f"  GET  /health        - Health check")
    print(f"{'='*60}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
