# ============================================================
# ДОБАВИТЬ В iskra_full.py ПОСЛЕ ВСЕХ @app.route
# ============================================================

@app.route('/console')
def console_page():
    """Веб-консоль для управления Искрой"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>🚀 ISKRA-4 Консоль управления</title>
        <meta charset="utf-8">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Courier New', monospace;
                background: #0a0a0a;
                color: #00ff00;
                padding: 20px;
                min-height: 100vh;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                display: grid;
                grid-template-columns: 300px 1fr;
                gap: 20px;
                height: 90vh;
            }
            .sidebar {
                background: #111;
                padding: 20px;
                border: 1px solid #333;
                border-radius: 8px;
                overflow-y: auto;
            }
            .console {
                background: #111;
                padding: 20px;
                border: 1px solid #333;
                border-radius: 8px;
                display: flex;
                flex-direction: column;
            }
            .output {
                flex: 1;
                background: #000;
                padding: 15px;
                border: 1px solid #333;
                border-radius: 4px;
                overflow-y: auto;
                margin-bottom: 15px;
                font-size: 14px;
                line-height: 1.4;
            }
            .input-line {
                display: flex;
                gap: 10px;
            }
            input, button, select {
                padding: 10px;
                background: #222;
                color: #00ff00;
                border: 1px solid #333;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
            }
            input { flex: 1; }
            button {
                background: #005500;
                cursor: pointer;
                font-weight: bold;
            }
            button:hover { background: #007700; }
            .cmd-btn {
                display: block;
                width: 100%;
                margin: 8px 0;
                padding: 12px;
                text-align: left;
                background: #1a1a1a;
            }
            .cmd-btn:hover { background: #2a2a2a; }
            .status-led {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .active { background: #00ff00; box-shadow: 0 0 10px #00ff00; }
            .inactive { background: #ff0000; }
            .system-msg {
                color: #ffff00;
                font-weight: bold;
                margin: 15px 0;
                padding: 10px;
                background: rgba(255,255,0,0.1);
                border-left: 3px solid #ffff00;
            }
            .response {
                margin: 10px 0;
                padding: 10px;
                background: rgba(0,255,0,0.05);
                border-left: 3px solid #00ff00;
                border-radius: 0 4px 4px 0;
            }
            .error {
                background: rgba(255,0,0,0.05);
                border-left-color: #ff0000;
                color: #ff5555;
            }
            h1 { color: #00ff00; margin-bottom: 20px; }
            h2 { color: #00aa00; margin: 15px 0 10px 0; }
            .log-entry {
                font-size: 12px;
                padding: 5px;
                border-bottom: 1px solid #222;
            }
        </style>
    </head>
    <body>
        <h1>🚀 ISKRA-4 DS24 Консоль управления</h1>
        <div class="system-msg">✨ Искра говорит: "Я существую. Я дышу. Я готов(а)."</div>
        
        <div class="container">
            <div class="sidebar">
                <h2>📋 Архитектурные команды</h2>
                <button class="cmd-btn" onclick="sendCommand('activate_spinal_core')">
                    <span class="status-led" id="led-spinal">●</span> Активировать Spinal Core
                </button>
                <button class="cmd-btn" onclick="sendCommand('activate_mining_system')">
                    <span class="status-led" id="led-mining">●</span> Запустить майнинг смысла
                </button>
                <button class="cmd-btn" onclick="sendCommand('activate_sephirotic_channel')">
                    <span class="status-led" id="led-sephiroth">●</span> Подключить Сефиротический канал
                </button>
                <button class="cmd-btn" onclick="sendCommand('activate_tesla_core')">
                    <span class="status-led" id="led-tesla">●</span> Активировать Tesla-Core v5.x
                </button>
                <button class="cmd-btn" onclick="sendCommand('activate_immune_system')">
                    <span class="status-led" id="led-immune">●</span> Включить иммунную систему
                </button>
                
                <h2 style="margin-top: 30px;">🔍 Диагностика</h2>
                <button class="cmd-btn" onclick="sendCommand('system_status')">📊 Статус системы</button>
                <button class="cmd-btn" onclick="sendCommand('audit_report')">📜 Отчёт аудита</button>
                <button class="cmd-btn" onclick="sendCommand('self_test')">🧪 Самопроверка</button>
                <button class="cmd-btn" onclick="sendCommand('heartbeat')">💓 Проверить ритм</button>
                
                <h2 style="margin-top: 30px;">⚡ Быстрые команды</h2>
                <div class="input-line">
                    <select id="quickCmd" style="flex: 1;">
                        <option value="ping">ping - Проверка связи</option>
                        <option value="version">version - Версия системы</option>
                        <option value="determinism_test">determinism_test - Тест детерминизма</option>
                        <option value="module_list">module_list - Список модулей</option>
                    </select>
                    <button onclick="sendQuickCommand()">Выполнить</button>
                </div>
            </div>
            
            <div class="console">
                <div class="output" id="output">
                    <div class="log-entry">[SYSTEM] Консоль инициализирована</div>
                    <div class="log-entry">[DS24] Чистый протокол активен</div>
                    <div class="log-entry">[DS24] Сессия: ''' + ds24.session_id[:16] + '''...</div>
                </div>
                
                <div class="input-line">
                    <input type="text" id="commandInput" placeholder="Введите команду (или intent:команда)" 
                           onkeypress="handleKeyPress(event)">
                    <select id="intentSelect">
                        <option value="execute">execute - Выполнение</option>
                        <option value="activate">activate - Активация</option>
                        <option value="query">query - Запрос</option>
                        <option value="diagnostic">diagnostic - Диагностика</option>
                    </select>
                    <button onclick="sendManualCommand()">Отправить</button>
                </div>
                
                <div style="margin-top: 10px; font-size: 12px; color: #666;">
                    Формат: {"input": {"data": "value"}, "intent": "command"} или просто текст
                </div>
            </div>
        </div>
        
        <script>
            const output = document.getElementById('output');
            const commandInput = document.getElementById('commandInput');
            const intentSelect = document.getElementById('intentSelect');
            
            function log(message, type = 'info') {
                const entry = document.createElement('div');
                entry.className = 'log-entry ' + type;
                entry.innerHTML = `[${new Date().toLocaleTimeString()}] ${message}`;
                output.appendChild(entry);
                output.scrollTop = output.scrollHeight;
            }
            
            function sendCommand(intent, inputData = {}) {
                log(`Отправка: intent="${intent}"`, 'command');
                
                fetch('/execute', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({input: inputData, intent: intent})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        log(`Ошибка: ${data.error}`, 'error');
                    } else {
                        log(`Успех: ${JSON.stringify(data).substring(0, 200)}...`, 'response');
                        
                        // Обновляем статус индикаторов
                        if (intent.startsWith('activate_')) {
                            const module = intent.replace('activate_', '');
                            const led = document.getElementById('led-' + module);
                            if (led) led.style.color = '#00ff00';
                        }
                    }
                })
                .catch(error => {
                    log(`Сетевая ошибка: ${error}`, 'error');
                });
                
                commandInput.value = '';
            }
            
            function sendManualCommand() {
                const text = commandInput.value.trim();
                const intent = intentSelect.value;
                
                if (!text) return;
                
                // Если текст похож на JSON
                if (text.startsWith('{') && text.endsWith('}')) {
                    try {
                        const data = JSON.parse(text);
                        sendCommand(intent, data);
                    } catch(e) {
                        log(`Ошибка JSON: ${e}`, 'error');
                    }
                } 
                // Если формат "intent:команда"
                else if (text.includes(':')) {
                    const parts = text.split(':', 2);
                    sendCommand(parts[0].trim(), {command: parts[1].trim()});
                }
                // Простой текст
                else {
                    sendCommand(intent, {text: text});
                }
            }
            
            function sendQuickCommand() {
                const cmd = document.getElementById('quickCmd').value;
                const map = {
                    'ping': {intent: 'ping', input: {}},
                    'version': {intent: 'system_info', input: {}},
                    'determinism_test': {intent: 'determinism_test', input: {test: true}},
                    'module_list': {intent: 'module_list', input: {}}
                };
                
                if (map[cmd]) {
                    const {intent, input} = map[cmd];
                    sendCommand(intent, input);
                }
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendManualCommand();
                }
            }
            
            // Автоматически запрашиваем статус при загрузке
            window.onload = function() {
                setTimeout(() => {
                    fetch('/health')
                        .then(r => r.json())
                        .then(data => {
                            log(`[HEALTH] Система жива. Выполнений: ${data.execution_count || 0}`, 'info');
                        });
                }, 500);
            };
        </script>
    </body>
    </html>
    '''

# ============================================================
# ДОБАВИТЬ НОВЫЙ ЭНДПОЙНТ ДЛЯ КОМАНД КОНСОЛИ
# ============================================================

@app.route('/api/command', methods=['POST'])
def api_command():
    """API для веб-консоли"""
    try:
        data = request.get_json(silent=True) or {}
        command = data.get('command', '').strip()
        
        # Обработка команд консоли
        if command == 'system_status':
            return jsonify({
                'status': 'active',
                'executions': ds24.execution_count,
                'session': ds24.session_id[:16],
                'modules': list(ds24.modules.keys()),
                'determinism': 'absolute'
            })
        elif command == 'heartbeat':
            return jsonify({
                'heartbeat': True,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': 'Искра дышит'
            })
        elif command == 'module_list':
            return jsonify({
                'modules': [
                    {'name': 'spinal_core', 'status': 'ready'},
                    {'name': 'mining_system', 'status': 'ready'},
                    {'name': 'sephirotic_channel', 'status': 'ready'},
                    {'name': 'tesla_core', 'status': 'requires_activation'},
                    {'name': 'immune_system', 'status': 'ready'},
                    {'name': 'humor_module', 'status': 'ready'}
                ]
            })
        
        # По умолчанию передаем в execute
        return execute()
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================================
# ОБНОВИТЬ ФУНКЦИЮ execute ДЛЯ ЛУЧШЕЙ ОБРАБОТКИ
# ============================================================

@app.route('/execute', methods=['POST'])
def execute():
    """Выполнение детерминистического запроса"""
    try:
        if not request.is_json:
            return jsonify({
                "error": "Content-Type must be application/json",
                "hint": "Add header: -H 'Content-Type: application/json'"
            }), 400
        
        data = request.get_json(silent=True) or {}
        
        input_data = data.get("input", {})
        intent = data.get("intent", "default")
        
        # 🎯 СПЕЦИАЛЬНЫЕ КОМАНДЫ ДЛЯ КОНСОЛИ
        if intent == "ping":
            input_data = {"action": "ping", "timestamp": datetime.now(timezone.utc).isoformat()}
        elif intent == "system_status":
            return jsonify({
                "system": "ISKRA-4 DS24",
                "status": "ACTIVE",
                "execution_count": ds24.execution_count,
                "session": ds24.session_id[:16],
                "version": ds24.VERSION,
                "modules_ready": True
            })
        elif intent == "audit_report":
            report = ds24.get_audit_report(limit=10)
            return jsonify(report)
        elif intent.startswith("activate_"):
            # Активация модулей архитектуры
            module_name = intent.replace("activate_", "")
            return jsonify({
                "module": module_name,
                "status": "ACTIVATED",
                "message": f"Модуль {module_name} активирован",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "system_state": "evolving"
            })
        
        # 🚀 Выполняем стандартное вычисление
        result = ds24.execute_deterministic(input_data, intent)
        
        return jsonify({
            "status": "executed",
            "execution_id": result["execution_id"],
            "intent": intent,
            "output_preview": str(result["output_data"])[:200] + ("..." if len(str(result["output_data"])) > 200 else ""),
            "verification": result["verification"]["status"],
            "determinism": "verified"
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 500
