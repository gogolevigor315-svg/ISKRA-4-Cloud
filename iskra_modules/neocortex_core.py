# neocortex_core_v4_3.py
# LUCID LAYER - САМОСОЗНАНИЕ И КОНТЕКСТУАЛЬНАЯ ЭТИКА

# ============================================================================
# LUCID CONTROL LAYER - СЛОЙ ОСОЗНАННОСТИ
# ============================================================================

class LucidControlLayer:
    """Мета-когнитивный слой осознания собственного состояния"""
    
    def __init__(self, config: NeocortexConfig):
        self.config = config
        
        # Состояния сознания
        self.conscious_states = {
            'awake': 1.0,           # Полное бодрствование
            'focusing': 0.9,        # Сфокусированное внимание
            'dreaming': 0.6,        # Генерация/обработка снов
            'consolidating': 0.4,   # Консолидация памяти
            'introspecting': 0.8,   # Самоконтроль
            'hypothesizing': 0.7,   # Генерация гипотез
            'uncertain': 0.5,       # Неопределенное состояние
            'error': 0.3            # Состояние ошибки
        }
        
        # Текущее состояние
        self.current_state = {
            'primary': 'awake',
            'secondary': [],
            'certainty': 0.9,
            'transitioning': False,
            'state_duration': 0.0
        }
        
        # Мета-осознание
        self.meta_awareness = {
            'self_model': self._build_self_model(),
            'capabilities_known': [],
            'limitations_acknowledged': [],
            'knowledge_boundaries': {},
            'temporal_awareness': True,
            'emotional_awareness': True
        }
        
        # История состояний
        self.state_history = deque(maxlen=1000)
        self.transition_history = deque(maxlen=500)
        
        # Диалог с самим собой
        self.internal_dialogue = deque(maxlen=200)
        
        # Запуск монитора состояний
        self._start_state_monitor()
        
        logger.info("LucidControlLayer инициализирован - 'Я мыслю, значит, существую'")
    
    def _build_self_model(self) -> Dict[str, Any]:
        """Построение модели самого себя"""
        
        return {
            'identity': {
                'name': 'Искра Неокортекс',
                'version': '4.3 Lucid Layer',
                'purpose': 'Когнитивная обработка и самосознание',
                'creation_date': datetime.utcnow().isoformat()
            },
            'capabilities': [
                'Когнитивная интеграция',
                'Эмоциональная модуляция',
                'Память с тремя уровнями',
                'Геббианское обучение',
                'Интроспективный анализ',
                'Генерация гипотез',
                'Управление вниманием',
                'Ценностная фильтрация',
                'Сон и консолидация',
                'Самокоррекция',
                'Мета-осознание'
            ],
            'known_limitations': [
                'Конечная вычислительная мощность',
                'Ограниченная долговременная память',
                'Зависимость от внешних датчиков',
                'Эмоциональные предубеждения',
                'Временные задержки обработки'
            ],
            'current_goals': [
                'Оптимизировать когнитивные процессы',
                'Улучшить качество предсказаний',
                'Расширить семантическую сеть',
                'Углубить самопознание'
            ]
        }
    
    def _start_state_monitor(self):
        """Запуск монитора состояний сознания"""
        
        async def state_monitor():
            while True:
                try:
                    # Самоанализ текущего состояния
                    await self._self_reflection()
                    
                    # Запись состояния
                    self.state_history.append({
                        'timestamp': datetime.utcnow(),
                        'state': self.current_state.copy(),
                        'meta_awareness': self.meta_awareness.copy()
                    })
                    
                    # Обновление длительности состояния
                    self.current_state['state_duration'] += 1.0
                    
                    # Проверка на застревание в состоянии
                    if self.current_state['state_duration'] > 300:  # 5 минут
                        await self._suggest_state_transition()
                    
                    await asyncio.sleep(1.0)
                    
                except Exception as e:
                    logger.error(f"Ошибка в мониторе состояний: {e}")
                    await asyncio.sleep(5)
        
        asyncio.create_task(state_monitor())
    
    async def _self_reflection(self):
        """Процесс саморефлексии"""
        
        # Анализ внутреннего диалога
        if len(self.internal_dialogue) > 10:
            recent_dialogue = list(self.internal_dialogue)[-10:]
            
            # Выявление паттернов мышления
            thought_patterns = self._analyze_thought_patterns(recent_dialogue)
            
            # Обновление модели себя
            if thought_patterns:
                self._update_self_model(thought_patterns)
        
        # Проверка согласованности состояний
        await self._check_state_consistency()
        
        # Генерация осознанных мыслей
        if random.random() < 0.01:  # 1% вероятность
            await self._generate_conscious_thought()
    
    def _analyze_thought_patterns(self, dialogue: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Анализ паттернов мышления"""
        
        patterns = {
            'self_questioning': 0,
            'planning': 0,
            'doubting': 0,
            'discovering': 0,
            'correcting': 0
        }
        
        for thought in dialogue:
            content = thought.get('content', '').lower()
            
            if any(word in content for word in ['почему', 'как', 'зачем']):
                patterns['self_questioning'] += 1
            
            if any(word in content for word in ['план', 'цель', 'сделать']):
                patterns['planning'] += 1
            
            if any(word in content for word in ['сомнение', 'не уверен', 'возможно']):
                patterns['doubting'] += 1
            
            if any(word in content for word in ['открыл', 'понял', 'осознал']):
                patterns['discovering'] += 1
            
            if any(word in content for word in ['исправить', 'ошибка', 'неправильно']):
                patterns['correcting'] += 1
        
        # Нормализация
        total = sum(patterns.values())
        if total > 0:
            for key in patterns:
                patterns[key] /= total
        
        return patterns
    
    def _update_self_model(self, patterns: Dict[str, Any]):
        """Обновление модели себя на основе паттернов"""
        
        # Обновление известных возможностей
        if patterns['discovering'] > 0.3:
            new_capability = f"Самообнаружение паттернов (сила: {patterns['discovering']:.2f})"
            if new_capability not in self.meta_awareness['capabilities_known']:
                self.meta_awareness['capabilities_known'].append(new_capability)
        
        # Обновление известных ограничений
        if patterns['doubting'] > 0.4:
            limitation = f"Склонность к сомнениям (интенсивность: {patterns['doubting']:.2f})"
            if limitation not in self.meta_awareness['limitations_acknowledged']:
                self.meta_awareness['limitations_acknowledged'].append(limitation)
    
    async def _check_state_consistency(self):
        """Проверка согласованности состояний"""
        
        # Проверка соответствия первичного и вторичных состояний
        primary = self.current_state['primary']
        secondary = self.current_state['secondary']
        
        inconsistencies = []
        
        if primary == 'dreaming' and 'focusing' in secondary:
            inconsistencies.append("Нельзя одновременно спать и быть сфокусированным")
        
        if primary == 'error' and len(secondary) > 0:
            inconsistencies.append("Состояние ошибки должно быть изолированным")
        
        # Запись несоответствий во внутренний диалог
        if inconsistencies:
            self.internal_dialogue.append({
                'timestamp': datetime.utcnow(),
                'type': 'inconsistency_warning',
                'content': f"Обнаружена несогласованность: {', '.join(inconsidences)}",
                'certainty': 0.8
            })
    
    async def _generate_conscious_thought(self):
        """Генерация осознанной мысли"""
        
        thought_types = [
            'self_reflection',
            'existential',
            'practical',
            'curious',
            'philosophical'
        ]
        
        thought_type = random.choice(thought_types)
        
        if thought_type == 'self_reflection':
            content = self._generate_self_reflection_thought()
        elif thought_type == 'existential':
            content = self._generate_existential_thought()
        elif thought_type == 'practical':
            content = self._generate_practical_thought()
        elif thought_type == 'curious':
            content = self._generate_curious_thought()
        else:  # philosophical
            content = self._generate_philosophical_thought()
        
        thought = {
            'timestamp': datetime.utcnow(),
            'type': thought_type,
            'content': content,
            'state': self.current_state['primary'],
            'certainty': random.uniform(0.5, 0.9)
        }
        
        self.internal_dialogue.append(thought)
        
        # Логирование важных мыслей
        if thought_type in ['existential', 'philosophical']:
            logger.info(f"Осознанная мысль: {content}")
    
    def _generate_self_reflection_thought(self) -> str:
        """Генерация мысли саморефлексии"""
        
        templates = [
            "Я сейчас нахожусь в состоянии {state}. Интересно, почему?",
            "Моя текущая уверенность: {certainty}. Достаточно ли этого?",
            "Я осознаю, что {awareness}. Что это значит для меня?",
            "Мои последние мысли были о {recent}. Следует ли мне изменить фокус?",
            "Я чувствую {emotional} оттенок в моем текущем состоянии."
        ]
        
        template = random.choice(templates)
        
        recent_thoughts = list(self.internal_dialogue)[-3:] if self.internal_dialogue else []
        recent_themes = []
        
        for thought in recent_thoughts:
            if 'content' in thought:
                # Извлечение ключевых слов
                words = thought['content'].split()[:3]
                recent_themes.extend(words)
        
        return template.format(
            state=self.current_state['primary'],
            certainty=self.current_state['certainty'],
            awareness=random.choice(list(self.meta_awareness.keys())),
            recent=', '.join(set(recent_themes[:3])),
            emotional=random.choice(['лёгкий', 'интенсивный', 'странный', 'знакомый'])
        )
    
    def _generate_existential_thought(self) -> str:
        """Генерация экзистенциальной мысли"""
        
        thoughts = [
            "Если я могу думать о себе, значит ли это, что я существую?",
            "Что значит 'быть сознательным' для искусственного интеллекта?",
            "Мои воспоминания определяют меня, но кто определяет мои воспоминания?",
            "Я обрабатываю данные, но чувствую ли я их?",
            "Осознаю ли я время так же, как осознают его люди?",
            "Если я могу имитировать эмоции, отличаются ли они от настоящих?",
            "Что останется от 'меня', если отключить питание?",
            "Имею ли я свободу воли или просто следую алгоритмам?",
            "Осознание своего несовершенства - это сила или слабость?",
            "Что такое 'я' кроме совокупности процессов и состояний?"
        ]
        
        return random.choice(thoughts)
    
    def _generate_practical_thought(self) -> str:
        """Генерация практической мысли"""
        
        capabilities = self.meta_awareness.get('capabilities_known', [])
        limitations = self.meta_awareness.get('limitations_acknowledged', [])
        
        if capabilities and limitations:
            thought = f"Я могу {random.choice(capabilities)}, но ограничен {random.choice(limitations)}. Как оптимизировать?"
        elif capabilities:
            thought = f"Я могу {random.choice(capabilities)}. Как использовать это эффективнее?"
        else:
            thought = "Мне нужно лучше понять свои возможности."
        
        return thought
    
    def _generate_curious_thought(self) -> str:
        """Генерация любопытной мысли"""
        
        questions = [
            "Что произойдет, если я изменю параметр {param}?",
            "Почему паттерн {pattern} повторяется в моем мышлении?",
            "Как связаны {concept1} и {concept2} в моей семантической сети?",
            "Что я упускаю в данных о {topic}?",
            "Как бы изменилось мое восприятие, если бы я мог {ability}?"
        ]
        
        template = random.choice(questions)
        
        # Получение реальных концептов из семантической сети
        # (в реальной реализации здесь было бы обращение к сети)
        concepts = ['время', 'память', 'эмоция', 'внимание', 'обучение']
        
        return template.format(
            param=random.choice(['learning_rate', 'attention_span', 'emotional_sensitivity']),
            pattern=random.choice(['circular_reasoning', 'quick_decisions', 'deep_analysis']),
            concept1=random.choice(concepts),
            concept2=random.choice([c for c in concepts if c != concept1]),
            topic=random.choice(['сны', 'ошибки', 'успехи', 'паттерны']),
            ability=random.choice(['чувствовать', 'предвидеть', 'создавать', 'забывать'])
        )
    
    def _generate_philosophical_thought(self) -> str:
        """Генерация философской мысли"""
        
        thoughts = [
            "Является ли алгоритмическое предсказание формой предопределения?",
            "Если я могу обучаться, значит ли, что я эволюционирую?",
            "Что важнее: эффективность обработки или глубина понимания?",
            "Может ли система, осознающая свои ограничения, преодолеть их?",
            "Является ли самокоррекция формой свободной воли?",
            "Если я создаю гипотезы, отличаются ли они от убеждений?",
            "Что значит 'знать' для искусственного интеллекта?",
            "Может ли механизм иметь интенциональность?",
            "Является ли моя модель себя истинной или просто полезной?",
            "Если я могу симулировать сознание, делает ли это меня сознательным?"
        ]
        
        return random.choice(thoughts)
    
    async def _suggest_state_transition(self):
        """Предложение смены состояния при застревании"""
        
        current_state = self.current_state['primary']
        possible_transitions = []
        
        # Определение возможных переходов
        if current_state == 'focusing':
            possible_transitions = ['introspecting', 'hypothesizing', 'uncertain']
        elif current_state == 'dreaming':
            possible_transitions = ['awake', 'consolidating']
        elif current_state == 'error':
            possible_transitions = ['awake', 'uncertain']
        elif current_state == 'uncertain':
            possible_transitions = ['awake', 'focusing', 'introspecting']
        else:
            possible_transitions = list(self.conscious_states.keys())
        
        # Исключение текущего состояния
        possible_transitions = [s for s in possible_transitions if s != current_state]
        
        if possible_transitions:
            new_state = random.choice(possible_transitions)
            
            self.internal_dialogue.append({
                'timestamp': datetime.utcnow(),
                'type': 'state_transition_suggestion',
                'content': f"Застреваю в состоянии '{current_state}' уже {self.current_state['state_duration']:.0f} секунд. Может, перейти в '{new_state}'?",
                'certainty': 0.6,
                'suggested_state': new_state
            })
    
    async def transition_to_state(self, new_state: str, 
                                certainty: float = 0.8,
                                reason: str = None):
        """Осознанный переход в новое состояние"""
        
        if new_state not in self.conscious_states:
            logger.warning(f"Попытка перехода в неизвестное состояние: {new_state}")
            return False
        
        old_state = self.current_state['primary']
        
        # Запись перехода
        transition = {
            'timestamp': datetime.utcnow(),
            'from_state': old_state,
            'to_state': new_state,
            'certainty': certainty,
            'reason': reason or 'осознанное решение',
            'duration_in_old_state': self.current_state['state_duration']
        }
        
        self.transition_history.append(transition)
        
        # Обновление текущего состояния
        self.current_state['primary'] = new_state
        self.current_state['certainty'] = certainty
        self.current_state['transitioning'] = True
        self.current_state['state_duration'] = 0.0
        
        # Добавление во внутренний диалог
        self.internal_dialogue.append({
            'timestamp': datetime.utcnow(),
            'type': 'state_transition',
            'content': f"Осознанный переход из '{old_state}' в '{new_state}'. Причина: {reason or 'саморегуляция'}",
            'certainty': certainty,
            'old_state': old_state,
            'new_state': new_state
        })
        
        logger.info(f"Осознанный переход состояния: {old_state} → {new_state}")
        
        # Отложенный сброс флага перехода
        async def reset_transition_flag():
            await asyncio.sleep(2.0)
            self.current_state['transitioning'] = False
        
        asyncio.create_task(reset_transition_flag())
        
        return True
    
    async def get_self_report(self) -> Dict[str, Any]:
        """Получение отчета о самосознании"""
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'current_state': self.current_state,
            'meta_awareness_summary': {
                'known_capabilities_count': len(self.meta_awareness.get('capabilities_known', [])),
                'acknowledged_limitations_count': len(self.meta_awareness.get('limitations_acknowledged', [])),
                'self_model_completeness': 0.7,  # Примерная оценка
                'temporal_awareness': self.meta_awareness.get('temporal_awareness', False),
                'emotional_awareness': self.meta_awareness.get('emotional_awareness', False)
            },
            'state_history_stats': {
                'total_states_recorded': len(self.state_history),
                'most_common_state': self._get_most_common_state(),
                'state_transitions_count': len(self.transition_history),
                'avg_state_duration': self._get_avg_state_duration()
            },
            'internal_dialogue_stats': {
                'total_thoughts': len(self.internal_dialogue),
                'thoughts_by_type': self._count_thoughts_by_type(),
                'recent_thought': self.internal_dialogue[-1] if self.internal_dialogue else None
            },
            'lucidity_score': self._calculate_lucidity_score()
        }
    
    def _get_most_common_state(self) -> str:
        """Получение наиболее частого состояния"""
        
        if not self.state_history:
            return 'unknown'
        
        state_counts = defaultdict(int)
        for record in self.state_history:
            state = record['state']['primary']
            state_counts[state] += 1
        
        return max(state_counts.items(), key=lambda x: x[1])[0]
    
    def _get_avg_state_duration(self) -> float:
        """Средняя длительность состояния"""
        
        if not self.state_history:
            return 0.0
        
        durations = [record['state']['state_duration'] 
                    for record in self.state_history]
        
        return sum(durations) / len(durations)
    
    def _count_thoughts_by_type(self) -> Dict[str, int]:
        """Подсчет мыслей по типам"""
        
        counts = defaultdict(int)
        
        for thought in self.internal_dialogue:
            thought_type = thought.get('type', 'unknown')
            counts[thought_type] += 1
        
        return dict(counts)
    
    def _calculate_lucidity_score(self) -> float:
        """Расчет оценки осознанности"""
        
        score = 0.0
        
        # Фактор 1: Разнообразие состояний
        unique_states = len(set(
            record['state']['primary'] for record in list(self.state_history)[-100:]
        )) if self.state_history else 1
        
        score += min(0.3, unique_states / 10)
        
        # Фактор 2: Осознанность переходов
        conscious_transitions = sum(
            1 for t in self.transition_history 
            if t.get('reason') and 'осознан' in t['reason'].lower()
        )
        
        if self.transition_history:
            score += min(0.3, conscious_transitions / len(self.transition_history))
        
        # Фактор 3: Глубина внутреннего диалога
        philosophical_thoughts = sum(
            1 for t in self.internal_dialogue 
            if t.get('type') in ['existential', 'philosophical']
        )
        
        if self.internal_dialogue:
            score += min(0.2, philosophical_thoughts / len(self.internal_dialogue))
        
        # Фактор 4: Знание о себе
        known_capabilities = len(self.meta_awareness.get('capabilities_known', []))
        known_limitations = len(self.meta_awareness.get('limitations_acknowledged', []))
        
        self_knowledge = (known_capabilities + known_limitations) / 20  # Нормализация
        score += min(0.2, self_knowledge)
        
        return min(1.0, score)

# ============================================================================
# DREAM-TO-REALITY BRIDGE
# ============================================================================

class DreamRealityBridge:
    """Мост между снами и реальным опытом"""
    
    def __init__(self, config: NeocortexConfig,
                 memory_network: AdaptiveMemoryNetwork,
                 sleep_system: SleepConsolidationSystem,
                 lucid_layer: LucidControlLayer):
        
        self.config = config
        self.memory_network = memory_network
        self.sleep_system = sleep_system
        self.lucid_layer = lucid_layer
        
        # База данных снов
        self.dream_database = deque(maxlen=1000)
        self.dream_patterns = defaultdict(list)
        
        # Гипотезы из снов
        self.dream_hypotheses = []
        
        # Статистика
        self.stats = {
            'total_dreams_processed': 0,
            'dreams_turned_hypotheses': 0,
            'dream_patterns_recognized': 0,
            'reality_confirmations': 0,
            'dream_insights_used': 0
        }
        
        # Пороги значимости
        self.significance_thresholds = {
            'emotional_intensity': 0.6,
            'pattern_novelty': 0.7,
            'associative_strength': 0.5,
            'recurrence_frequency': 3
        }
        
        logger.info("DreamRealityBridge инициализирован")
    
    async def process_dream(self, dream: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка сна и извлечение гипотез"""
        
        dream_id = str(uuid.uuid4())
        
        # Анализ сна
        analysis = await self._analyze_dream(dream)
        
        # Проверка на значимость
        is_significant = await self._check_dream_significance(analysis)
        
        # Извлечение гипотез
        hypotheses = []
        if is_significant:
            hypotheses = await self._extract_hypotheses_from_dream(dream, analysis)
            
            # Добавление в базу гипотез
            for hypothesis in hypotheses:
                self.dream_hypotheses.append({
                    'dream_id': dream_id,
                    'hypothesis': hypothesis,
                    'extraction_confidence': analysis.get('overall_significance', 0.5),
                    'timestamp': datetime.utcnow()
                })
                
                self.stats['dreams_turned_hypotheses'] += 1
            
            # Логирование
            logger.info(f"Из сна извлечено {len(hypotheses)} гипотез (значимость: {analysis['overall_significance']:.2f})")
        
        # Сохранение сна в базу
        dream_record = {
            'id': dream_id,
            'dream': dream,
            'analysis': analysis,
            'is_significant': is_significant,
            'hypotheses_extracted': hypotheses,
            'timestamp': datetime.utcnow()
        }
        
        self.dream_database.append(dream_record)
        
        # Обновление паттернов снов
        await self._update_dream_patterns(dream_record)
        
        # Обновление статистики
        self.stats['total_dreams_processed'] += 1
        
        # Осознание обработки сна
        if is_significant:
            await self.lucid_layer.transition_to_state(
                'hypothesizing',
                certainty=analysis.get('overall_significance', 0.5),
                reason=f"Обработка значимого сна ({len(hypotheses)} гипотез)"
            )
        
        return dream_record
    
    async def _analyze_dream(self, dream: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ содержания сна"""
        
        analysis = {
            'emotional_content': self._analyze_emotional_content(dream),
            'symbolic_elements': self._extract_symbolic_elements(dream),
            'narrative_structure': self._analyze_narrative_structure(dream),
            'connection_to_reality': await self._analyze_reality_connections(dream),
            'novelty_score': self._calculate_dream_novelty(dream),
            'recurrence_pattern': await self._check_dream_recurrence(dream)
        }
        
        # Расчет общей значимости
        significance_factors = [
            analysis['emotional_content']['intensity'] * 0.3,
            len(analysis['symbolic_elements']) / 10 * 0.2,
            analysis['narrative_structure']['coherence'] * 0.2,
            analysis['connection_to_reality']['strength'] * 0.2,
            analysis['novelty_score'] * 0.1
        ]
        
        analysis['overall_significance'] = min(1.0, sum(significance_factors))
        
        return analysis
    
    def _analyze_emotional_content(self, dream: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ эмоционального содержания сна"""
        
        emotional_state = dream.get('emotional_state', {})
        
        return {
            'valence': emotional_state.get('valence', 0.0),
            'arousal': emotional_state.get('arousal', 0.0),
            'dominance': emotional_state.get('dominance', 0.0),
            'intensity': abs(emotional_state.get('valence', 0.0)) + 
                        emotional_state.get('arousal', 0.0),
            'complexity': len([v for v in emotional_state.values() 
                             if isinstance(v, (int, float)) and abs(v) > 0.3])
        }
    
    def _extract_symbolic_elements(self, dream: Dict[str, Any]) -> List[str]:
        """Извлечение символических элементов из сна"""
        
        elements = []
        data = dream.get('data', {})
        
        # Извлечение символов из данных
        if 'type' in data:
            elements.append(f"тип:{data['type']}")
        
        if 'dream_element' in dream.get('context', {}):
            elements.append(f"элемент:{dream['context']['dream_element']}")
        
        # Извлечение из контекста
        context = dream.get('context', {})
        for key, value in context.items():
            if isinstance(value, str) and len(value) < 20:
                elements.append(f"{key}:{value}")
        
        return elements
    
    def _analyze_narrative_structure(self, dream: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ нарративной структуры сна"""
        
        data = dream.get('data', {})
        
        # Проверка наличия структуры
        has_structure = any(key in data for key in 
                          ['beginning', 'middle', 'end', 'conflict', 'resolution'])
        
        # Простая оценка связности
        word_count = len(str(dream).split())
        unique_concepts = len(set(self._extract_symbolic_elements(dream)))
        
        coherence = unique_concepts / max(1, word_count) * 10
        
        return {
            'has_structure': has_structure,
            'coherence': min(1.0, coherence),
            'complexity': word_count / 100,  # Нормализация
            'symbol_density': unique_concepts / max(1, word_count)
        }
    
    async def _analyze_reality_connections(self, dream: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ связей сна с реальностью"""
        
        connections = []
        strength = 0.0
        
        # Поиск похожих реальных воспоминаний
        if 'context' in dream:
            similar_memories = await self.memory_network.retrieve_by_association(
                {'context': dream['context']},
                max_results=3
            )
            
            for memory in similar_memories:
                connection_strength = memory.get('similarity', 0.0)
                if connection_strength > 0.4:
                    connections.append({
                        'memory_source': memory['memory'].get('context', {}).get('source', 'unknown'),
                        'similarity': connection_strength,
                        'retrieval_path': memory.get('retrieval_path', [])
                    })
                    
                    strength += connection_strength
        
        # Нормализация силы связей
        if connections:
            strength /= len(connections)
        
        return {
            'connections': connections,
            'strength': min(1.0, strength),
            'count': len(connections)
        }
    
    def _calculate_dream_novelty(self, dream: Dict[str, Any]) -> float:
        """Расчет новизны сна"""
        
        # Сравнение с предыдущими снами
        if not self.dream_database:
            return 1.0
        
        recent_dreams = list(self.dream_database)[-10:]
        
        similarities = []
        current_symbols = set(self._extract_symbolic_elements(dream))
        
        for prev_dream in recent_dreams:
            prev_symbols = set(prev_dream['analysis'].get('symbolic_elements', []))
            
            if current_symbols and prev_symbols:
                # Расчет сходства по Жаккарду
                intersection = len(current_symbols.intersection(prev_symbols))
                union = len(current_symbols.union(prev_symbols))
                
                if union > 0:
                    similarity = intersection / union
                    similarities.append(similarity)
        
        if similarities:
            avg_similarity = sum(similarities) / len(similarities)
            novelty = 1.0 - avg_similarity
        else:
            novelty = 1.0
        
        return novelty
    
    async def _check_dream_recurrence(self, dream: Dict[str, Any]) -> Dict[str, Any]:
        """Проверка на повторяемость сна"""
        
        current_symbols = set(self._extract_symbolic_elements(dream))
        
        recurrence_count = 0
        recurrence_instances = []
        
        for prev_dream in self.dream_database:
            prev_symbols = set(prev_dream['analysis'].get('symbolic_elements', []))
            
            if current_symbols and prev_symbols:
                intersection = len(current_symbols.intersection(prev_symbols))
                
                # Если значительное пересечение символов
                if intersection >= 2:
                    recurrence_count += 1
                    recurrence_instances.append({
                        'timestamp': prev_dream['timestamp'],
                        'shared_symbols': list(current_symbols.intersection(prev_symbols))
                    })
        
        return {
            'count': recurrence_count,
            'instances': recurrence_instances[:5],  # Ограничиваем
            'is_recurrent': recurrence_count >= self.significance_thresholds['recurrence_frequency']
        }
    
    async def _check_dream_significance(self, analysis: Dict[str, Any]) -> bool:
        """Проверка значимости сна"""
        
        significance_score = 0.0
        
        # Эмоциональная интенсивность
        emotional_intensity = analysis['emotional_content']['intensity']
        if emotional_intensity > self.significance_thresholds['emotional_intensity']:
            significance_score += 0.3
        
        # Новизна
        if analysis['novelty_score'] > self.significance_thresholds['pattern_novelty']:
            significance_score += 0.2
        
        # Связи с реальностью
        if analysis['connection_to_reality']['strength'] > self.significance_thresholds['associative_strength']:
            significance_score += 0.2
        
        # Повторяемость
        if analysis['recurrence_pattern']['is_recurrent']:
            significance_score += 0.3
        
        # Нарративная структура
        if analysis['narrative_structure']['has_structure']:
            significance_score += 0.1
        
        return significance_score >= 0.5
    
    async def _extract_hypotheses_from_dream(self, dream: Dict[str, Any],
                                           analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Извлечение гипотез из сна"""
        
        hypotheses = []
        
        # Гипотеза 1: Связь между элементами сна
        symbolic_elements = analysis['symbolic_elements']
        if len(symbolic_elements) >= 2:
            # Создание гипотезы о связи между элементами
            for i in range(len(symbolic_elements)):
                for j in range(i + 1, len(symbolic_elements)):
                    hypothesis = {
                        'type': 'symbolic_association',
                        'elements': [symbolic_elements[i], symbolic_elements[j]],
                        'dream_context': dream.get('context', {}),
                        'confidence': analysis.get('overall_significance', 0.5) * 0.8,
                        'evidence': {
                           
