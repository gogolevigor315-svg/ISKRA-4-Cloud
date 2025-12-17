# neocortex_core_v4_2.py
# РАСШИРЕНИЕ ВЕРСИИ 4.1

# ============================================================================
# МЕХАНИЗМ СНА И КОНСОЛИДАЦИИ ПАМЯТИ
# ============================================================================

class SleepConsolidationSystem:
    """Система консолидации памяти во время "сна" (оффлайн обработки)"""
    
    def __init__(self, config: NeocortexConfig, 
                 memory_network: AdaptiveMemoryNetwork,
                 hierarchical_memory: HierarchicalMemory):
        
        self.config = config
        self.memory_network = memory_network
        self.hierarchical_memory = hierarchical_memory
        
        # Состояние сна
        self.sleep_state = {
            'is_sleeping': False,
            'sleep_cycle': 0,
            'last_sleep': None,
            'dreams_generated': 0,
            'memory_consolidated': 0
        }
        
        # Параметры сна
        self.sleep_params = {
            'consolidation_interval': 3600,  # Каждый час
            'dream_probability': 0.3,
            'memory_replay_ratio': 0.1,  # 10% воспоминаний для перепроигрывания
            'synaptic_pruning_threshold': 0.1,
            'min_sleep_duration': 300,  # 5 минут
            'max_sleep_duration': 1800  # 30 минут
        }
        
        # Очередь консолидации
        self.consolidation_queue = asyncio.Queue(maxsize=1000)
        self.consolidation_tasks = set()
        
        # Статистика
        self.stats = {
            'total_sleep_cycles': 0,
            'total_consolidation_time': 0,
            'memories_processed': 0,
            'dreams_created': 0,
            'pruned_connections': 0
        }
        
        # Запуск фоновой консолидации
        self._start_sleep_monitor()
        
        logger.info("SleepConsolidationSystem инициализирована")
    
    def _start_sleep_monitor(self):
        """Запуск мониторинга необходимости сна"""
        
        async def sleep_monitor():
            while True:
                try:
                    # Проверка времени с последнего сна
                    if self.sleep_state['last_sleep']:
                        time_since_sleep = (datetime.utcnow() - 
                                          self.sleep_state['last_sleep']).total_seconds()
                        
                        if time_since_sleep > self.sleep_params['consolidation_interval']:
                            await self.initiate_sleep_cycle()
                    
                    await asyncio.sleep(60)  # Проверка каждую минуту
                    
                except Exception as e:
                    logger.error(f"Ошибка в мониторе сна: {e}")
                    await asyncio.sleep(10)
        
        asyncio.create_task(sleep_monitor())
    
    async def initiate_sleep_cycle(self, forced: bool = False) -> bool:
        """Инициация цикла сна и консолидации"""
        
        if self.sleep_state['is_sleeping'] and not forced:
            logger.debug("Уже в состоянии сна")
            return False
        
        logger.info(f"Инициация цикла сна (цикл {self.sleep_state['sleep_cycle']})")
        
        # Начало сна
        self.sleep_state['is_sleeping'] = True
        self.sleep_state['sleep_cycle'] += 1
        
        sleep_start = datetime.utcnow()
        
        try:
            # Фаза 1: Сбор памяти для консолидации
            memories_to_consolidate = await self._gather_memories_for_consolidation()
            
            # Фаза 2: Перепроигрывание и усиление важных воспоминаний
            consolidated_count = await self._replay_and_consolidate(memories_to_consolidate)
            
            # Фаза 3: Генерация "снов" (случайные ассоциации)
            if random.random() < self.sleep_params['dream_probability']:
                dreams_generated = await self._generate_dreams(memories_to_consolidate)
                self.sleep_state['dreams_generated'] += dreams_generated
            
            # Фаза 4: Синаптическое прунирование
            pruned_count = await self._synaptic_pruning()
            
            # Фаза 5: Оптимизация индексов
            await self._optimize_memory_indices()
            
            # Обновление статистики
            sleep_duration = (datetime.utcnow() - sleep_start).total_seconds()
            self.stats['total_consolidation_time'] += sleep_duration
            self.stats['memories_processed'] += consolidated_count
            self.stats['pruned_connections'] += pruned_count
            self.stats['total_sleep_cycles'] += 1
            
            self.sleep_state['last_sleep'] = datetime.utcnow()
            self.sleep_state['memory_consolidated'] += consolidated_count
            
            logger.info(f"Цикл сна завершен: "
                       f"{consolidated_count} воспоминаний консолидировано, "
                       f"{pruned_count} связей обрезано, "
                       f"длительность: {sleep_duration:.1f}с")
            
            return True
            
        except Exception as e:
            logger.error(f"Ошибка во время цикла сна: {e}")
            return False
            
        finally:
            self.sleep_state['is_sleeping'] = False
    
    async def _gather_memories_for_consolidation(self) -> List[Dict[str, Any]]:
        """Сбор воспоминаний для консолидации"""
        
        memories = []
        
        try:
            # Получение недавних воспоминаний из сети
            episodic_memories = list(self.memory_network.episodic_memory)
            
            # Фильтрация по важности и новизне
            for memory in episodic_memories:
                # Расчет оценки для консолидации
                consolidation_score = self._calculate_consolidation_score(memory)
                
                if consolidation_score > 0.3:  # Порог консолидации
                    memories.append({
                        'memory': memory,
                        'score': consolidation_score,
                        'age': (datetime.utcnow() - memory['timestamp']).total_seconds()
                    })
            
            # Сортировка по оценке
            memories.sort(key=lambda x: x['score'], reverse=True)
            
            # Ограничение количества
            max_memories = int(len(episodic_memories) * 
                             self.sleep_params['memory_replay_ratio'])
            memories = memories[:max_memories]
            
            logger.debug(f"Собрано {len(memories)} воспоминаний для консолидации")
            
        except Exception as e:
            logger.error(f"Ошибка сбора воспоминаний: {e}")
        
        return memories
    
    def _calculate_consolidation_score(self, memory: Dict[str, Any]) -> float:
        """Расчет оценки важности для консолидации"""
        
        score = 0.0
        
        # 1. Эмоциональная значимость
        if 'emotional_intensity' in memory.get('experience', {}):
            emotional_score = memory['experience']['emotional_intensity']
            score += emotional_score * 0.4
        
        # 2. Временная свежесть (новые воспоминания важнее)
        age = (datetime.utcnow() - memory['timestamp']).total_seconds()
        recency_score = max(0, 1 - age / 86400)  # Затухание за сутки
        score += recency_score * 0.3
        
        # 3. Частота активации (из семантической сети)
        concepts = self.memory_network._extract_concepts(
            memory.get('context', {}),
            memory.get('features', {})
        )
        
        activation_score = 0.0
        for concept in concepts[:5]:
            if concept in self.memory_network.semantic_network:
                weight = self.memory_network.semantic_network.nodes[concept].get('weight', 0)
                activation_score += weight
        
        activation_score = min(1.0, activation_score / 10)
        score += activation_score * 0.3
        
        return min(1.0, score)
    
    async def _replay_and_consolidate(self, memories: List[Dict[str, Any]]) -> int:
        """Перепроигрывание и консолидация воспоминаний"""
        
        consolidated_count = 0
        
        for memory_info in memories:
            try:
                memory = memory_info['memory']
                
                # Перепроигрывание опыта через сеть
                experience = memory['experience']
                features = await self.memory_network.process_experience(experience)
                
                # Усиление ассоциаций
                await self._strengthen_associations(memory, features)
                
                # Продвижение в иерархической памяти
                if memory_info['score'] > 0.7:
                    # Важные воспоминания переводим в долговременную память
                    key = hashlib.md5(
                        json.dumps(memory['experience'], sort_keys=True).encode()
                    ).hexdigest()
                    
                    await self.hierarchical_memory.promote_memory(
                        key, "long_term"
                    )
                
                consolidated_count += 1
                
                # Пауза для предотвращения перегрузки
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Ошибка консолидации воспоминания: {e}")
                continue
        
        return consolidated_count
    
    async def _strengthen_associations(self, memory: Dict[str, Any], 
                                     features: Dict[str, Any]):
        """Усиление ассоциативных связей"""
        
        # Усиление связей в семантической сети
        concepts = self.memory_network._extract_concepts(
            memory.get('context', {}),
            features
        )
        
        for concept in concepts:
            if concept in self.memory_network.semantic_network:
                # Усиление узла
                current_weight = self.memory_network.semantic_network.nodes[concept].get('weight', 1.0)
                self.memory_network.semantic_network.nodes[concept]['weight'] = min(
                    10.0, current_weight * 1.1
                )
                
                # Усиление связей
                for neighbor in self.memory_network.semantic_network.neighbors(concept):
                    edge_data = self.memory_network.semantic_network[concept][neighbor]
                    current_edge_weight = edge_data.get('weight', 0.1)
                    self.memory_network.semantic_network[concept][neighbor]['weight'] = min(
                        5.0, current_edge_weight * 1.05
                    )
    
    async def _generate_dreams(self, memories: List[Dict[str, Any]]) -> int:
        """Генерация "снов" - случайных ассоциаций"""
        
        dreams_generated = 0
        
        if not memories:
            return 0
        
        try:
            # Выбор случайных воспоминаний для комбинации
            num_dreams = random.randint(1, min(5, len(memories) // 2))
            
            for _ in range(num_dreams):
                if len(memories) < 2:
                    break
                
                # Выбор двух случайных воспоминаний
                mem1, mem2 = random.sample(memories, 2)
                
                # Создание гибридного "сна"
                dream = self._combine_memories_into_dream(mem1['memory'], mem2['memory'])
                
                # Обработка "сна" через сеть (но без сохранения)
                await self.memory_network.process_experience(dream)
                
                dreams_generated += 1
                
                logger.debug(f"Сгенерирован сон из {mem1['memory'].get('context', {}).get('source', '?')} "
                           f"и {mem2['memory'].get('context', {}).get('source', '?')}")
        
        except Exception as e:
            logger.error(f"Ошибка генерации снов: {e}")
        
        return dreams_generated
    
    def _combine_memories_into_dream(self, memory1: Dict[str, Any], 
                                   memory2: Dict[str, Any]) -> Dict[str, Any]:
        """Комбинация двух воспоминаний в "сон" """
        
        # Случайное смешивание контекстов
        context1 = memory1.get('context', {})
        context2 = memory2.get('context', {})
        
        dream_context = {}
        for key in set(context1.keys()) | set(context2.keys()):
            if random.random() < 0.5:
                dream_context[key] = context1.get(key)
            else:
                dream_context[key] = context2.get(key)
        
        # Случайные искажения
        if random.random() < 0.3:
            # Добавление случайного элемента
            dream_context['dream_element'] = random.choice([
                'flying', 'floating', 'transformation', 'unfamiliar_place'
            ])
        
        return {
            'data': {
                'type': 'dream',
                'source_memory_1': memory1.get('context', {}).get('source', 'unknown'),
                'source_memory_2': memory2.get('context', {}).get('source', 'unknown'),
                'dream_timestamp': datetime.utcnow().isoformat()
            },
            'context': dream_context,
            'emotional_state': {
                'valence': random.uniform(-0.5, 0.5),
                'arousal': random.uniform(0.3, 0.8)
            },
            'timestamp': datetime.utcnow()
        }
    
    async def _synaptic_pruning(self) -> int:
        """Синаптическое прунирование - обрезка слабых связей"""
        
        pruned_count = 0
        
        try:
            # Обрезка в геббианских сетях
            layers = [
                self.memory_network.sensory_encoder,
                self.memory_network.associative_layer,
                self.memory_network.pattern_layer
            ]
            
            for layer in layers:
                layer.prune_weak_connections(self.sleep_params['synaptic_pruning_threshold'])
                
                # Подсчет обрезанных связей
                stats = layer.get_learning_stats()
                pruned_count += stats.get('pruned_connections', 0)
            
            # Обрезка в семантической сети
            edges_to_remove = []
            for u, v, data in self.memory_network.semantic_network.edges(data=True):
                if data.get('weight', 0) < self.sleep_params['synaptic_pruning_threshold']:
                    edges_to_remove.append((u, v))
            
            self.memory_network.semantic_network.remove_edges_from(edges_to_remove)
            pruned_count += len(edges_to_remove)
            
            # Удаление изолированных узлов
            isolated_nodes = [node for node, degree in 
                            self.memory_network.semantic_network.degree() 
                            if degree == 0]
            self.memory_network.semantic_network.remove_nodes_from(isolated_nodes)
            
            logger.debug(f"Синаптическое прунирование: {pruned_count} связей обрезано")
            
        except Exception as e:
            logger.error(f"Ошибка синаптического прунирования: {e}")
        
        return pruned_count
    
    async def _optimize_memory_indices(self):
        """Оптимизация индексов памяти"""
        
        try:
            # Оптимизация индексов в иерархической памяти
            # (в реальной реализации здесь был бы вызов методов оптимизации БД)
            
            # Перестроение кэша
            self.hierarchical_memory.short_term = OrderedDict(
                list(self.hierarchical_memory.short_term.items())[-500:]
            )
            
            logger.debug("Индексы памяти оптимизированы")
            
        except Exception as e:
            logger.error(f"Ошибка оптимизации индексов: {e}")
    
    def get_sleep_stats(self) -> Dict[str, Any]:
        """Получение статистики сна"""
        
        return {
            'sleep_state': self.sleep_state,
            'sleep_params': self.sleep_params,
            'stats': self.stats,
            'next_sleep_in': self._time_until_next_sleep()
        }
    
    def _time_until_next_sleep(self) -> Optional[float]:
        """Время до следующего сна"""
        
        if not self.sleep_state['last_sleep']:
            return 0
        
        time_since_sleep = (datetime.utcnow() - 
                          self.sleep_state['last_sleep']).total_seconds()
        
        return max(0, self.sleep_params['consolidation_interval'] - time_since_sleep)

# ============================================================================
# ЦЕННОСТНЫЙ ФИЛЬТР
# ============================================================================

class ValueFilter:
    """Фильтр значимости на основе эмоциональных сигналов и системных ценностей"""
    
    def __init__(self, config: NeocortexConfig):
        self.config = config
        
        # Системные ценности (веса)
        self.values = {
            'curiosity': 0.8,       # Любопытство и исследование
            'efficiency': 0.7,      # Эффективность обработки
            'stability': 0.6,       # Стабильность системы
            'learning': 0.9,        # Обучение и адаптация
            'self_preservation': 0.5, # Самосохранение
            'goal_achievement': 0.75 # Достижение целей
        }
        
        # Эмоциональные преобразователи
        self.emotional_transformers = {
            'valence': self._transform_valence,
            'arousal': self._transform_arousal,
            'dominance': self._transform_dominance
        }
        
        # История оценок
        self.value_history = deque(maxlen=1000)
        
        # Адаптивные веса
        self.adaptive_weights = self.values.copy()
        
        logger.info("ValueFilter инициализирован")
    
    async def evaluate_cognitive_act(self, act: CognitiveAct) -> Dict[str, Any]:
        """Оценка когнитивного акта через призму ценностей"""
        
        evaluation = {
            'act_id': act.id,
            'timestamp': datetime.utcnow(),
            'value_scores': {},
            'total_value': 0.0,
            'recommendation': 'process',  # process, prioritize, defer, ignore
            'reasoning': []
        }
        
        # Оценка по каждой ценности
        for value_name, base_weight in self.adaptive_weights.items():
            score = await self._evaluate_by_value(act, value_name, base_weight)
            evaluation['value_scores'][value_name] = score
            
            # Накопление общего значения
            evaluation['total_value'] += score * base_weight
        
        # Нормализация общего значения
        max_possible = sum(self.adaptive_weights.values())
        if max_possible > 0:
            evaluation['total_value'] /= max_possible
        
        # Эмоциональная модуляция
        if act.emotional_state:
            emotional_modulation = self._apply_emotional_modulation(
                act.emotional_state,
                evaluation['total_value']
            )
            
            evaluation['total_value'] = emotional_modulation['final_value']
            evaluation['emotional_impact'] = emotional_modulation['impact']
            evaluation['reasoning'].extend(emotional_modulation['reasoning'])
        
        # Генерация рекомендации
        evaluation['recommendation'] = self._generate_recommendation(
            evaluation['total_value'],
            act.importance
        )
        
        # Обновление адаптивных весов
        await self._update_adaptive_weights(act, evaluation)
        
        # Запись в историю
        self.value_history.append({
            'act': act.to_dict(),
            'evaluation': evaluation,
            'timestamp': datetime.utcnow()
        })
        
        return evaluation
    
    async def _evaluate_by_value(self, act: CognitiveAct, 
                               value_name: str, 
                               base_weight: float) -> float:
        """Оценка акта по конкретной ценности"""
        
        score = 0.0
        
        if value_name == 'curiosity':
            # Любопытство ценит новизну и сложность
            if act.act_type in ['exploration', 'discovery', 'hypothesis_generation']:
                score += 0.7
            
            # Новые данные ценнее старых
            if act.data.get('novelty', 0) > 0.5:
                score += 0.3
        
        elif value_name == 'efficiency':
            # Эффективность ценит быстрые и точные решения
            if act.act_type in ['decision_making', 'optimization']:
                score += 0.6
            
            if act.confidence > 0.8:
                score += 0.4
        
        elif value_name == 'stability':
            # Стабильность ценит предсказуемость и контроль
            if act.act_type in ['monitoring', 'error_handling', 'stabilization']:
                score += 0.8
            
            if act.data.get('risk_level', 0) < 0.3:
                score += 0.2
        
        elif value_name == 'learning':
            # Обучение ценит новые знания и адаптацию
            if act.act_type in ['learning', 'adaptation', 'pattern_recognition']:
                score += 0.9
            
            if act.data.get('learning_potential', 0) > 0.5:
                score += 0.1
        
        elif value_name == 'self_preservation':
            # Самосохранение ценит безопасность и ресурсы
            if act.act_type in ['threat_detection', 'resource_management']:
                score += 0.7
            
            if act.data.get('threat_level', 0) > 0:
                score += 0.3
        
        elif value_name == 'goal_achievement':
            # Достижение целей ценит прогресс и завершение
            if act.act_type in ['goal_progress', 'task_completion']:
                score += 0.8
            
            if act.data.get('goal_relevance', 0) > 0.5:
                score += 0.2
        
        # Модуляция на основе контекста
        context_modulation = self._contextual_modulation(act.context, value_name)
        score *= context_modulation
        
        return min(1.0, score)
    
    def _contextual_modulation(self, context: Dict[str, Any], 
                             value_name: str) -> float:
        """Модуляция оценки на основе контекста"""
        
        modulation = 1.0
        
        if 'system_load' in context:
            load = context['system_load']
            
            if value_name == 'efficiency' and load > 0.8:
                modulation *= 1.5  # При высокой нагрузке эффективность важнее
            
            if value_name == 'stability' and load > 0.9:
                modulation *= 2.0  # При перегрузке стабильность критична
        
        if 'time_constraint' in context:
            time_left = context['time_constraint']
            
            if value_name == 'goal_achievement' and time_left < 10:
                modulation *= 1.8  # Срочные цели важнее
        
        return modulation
    
    def _apply_emotional_modulation(self, emotional_state: EmotionalState,
                                  base_value: float) -> Dict[str, Any]:
        """Применение эмоциональной модуляции к оценке"""
        
        impact = {
            'valence_impact': 0.0,
            'arousal_impact': 0.0,
            'dominance_impact': 0.0
        }
        
        reasoning = []
        final_value = base_value
        
        # Валентность (положительная/отрицательная)
        valence_impact = self.emotional_transformers['valence'](emotional_state.valence)
        impact['valence_impact'] = valence_impact
        
        if valence_impact != 0:
            final_value += valence_impact * 0.2
            reasoning.append(f"Валентность {emotional_state.valence:.2f} → "
                           f"изменение: {valence_impact:.2f}")
        
        # Возбуждение (интенсивность)
        arousal_impact = self.emotional_transformers['arousal'](emotional_state.arousal)
        impact['arousal_impact'] = arousal_impact
        
        if arousal_impact != 0:
            final_value *= (1 + arousal_impact * 0.3)
            reasoning.append(f"Возбуждение {emotional_state.arousal:.2f} → "
                           f"множитель: {arousal_impact:.2f}")
        
        # Доминантность (контроль)
        dominance_impact = self.emotional_transformers['dominance'](emotional_state.dominance)
        impact['dominance_impact'] = dominance_impact
        
        if dominance_impact != 0:
            # Доминантность стабилизирует оценку
            adjustment = dominance_impact * (0.5 - abs(base_value - 0.5))
            final_value += adjustment
            reasoning.append(f"Доминантность {emotional_state.dominance:.2f} → "
                           f"стабилизация: {adjustment:.2f}")
        
        return {
            'final_value': max(0.0, min(1.0, final_value)),
            'impact': impact,
            'reasoning': reasoning
        }
    
    def _transform_valence(self, valence: float) -> float:
        """Преобразование валентности в воздействие на оценку"""
        
        if valence > 0.3:
            return 0.1  # Положительные эмоции слегка повышают оценку
        elif valence < -0.3:
            return -0.1  # Отрицательные эмоции слегка понижают оценку
        else:
            return 0.0
    
    def _transform_arousal(self, arousal: float) -> float:
        """Преобразование возбуждения в воздействие на оценку"""
        
        if arousal > 0.7:
            return 0.2  # Высокое возбуждение усиливает оценку
        elif arousal < 0.3:
            return -0.1  # Низкое возбуждение ослабляет оценку
        else:
            return 0.0
    
    def _transform_dominance(self, dominance: float) -> float:
        """Преобразование доминантности в воздействие на оценку"""
        
        if dominance > 0.6:
            return 0.15  # Высокая доминантность стабилизирует
        elif dominance < 0.4:
            return -0.1  # Низкая доминантность дестабилизирует
        else:
            return 0.0
    
    def _generate_recommendation(self, total_value: float, 
                               importance: float) -> str:
        """Генерация рекомендации на основе оценки"""
        
        combined_score = (total_value * 0.7 + importance * 0.3)
        
        if combined_score > 0.8:
            return 'prioritize'  # Высокий приоритет
        elif combined_score > 0.5:
            return 'process'     # Обычная обработка
        elif combined_score > 0.3:
            return 'defer'       # Отложить
        else:
            return 'ignore'      # Игнорировать
    
    async def _update_adaptive_weights(self, act: CognitiveAct,
                                     evaluation: Dict[str, Any]):
        """Адаптивное обновление весов ценностей"""
        
        # Обновление на основе успешности акта
        if act.act_type == 'decision_making':
            # Если решение было успешным, усиливаем соответствующие ценности
            outcome = act.data.get('outcome', 'unknown')
            
            if outcome == 'success':
                # Усиливаем ценности, способствовавшие успеху
                for value_name, score in evaluation['value_scores'].items():
                    if score > 0.5:
                        self.adaptive_weights[value_name] = min(
                            1.0, 
                            self.adaptive_weights[value_name] * 1.05
                        )
            
            elif outcome == 'failure':
                # Ослабляем ценности, не предотвратившие неудачу
                for value_name, score in evaluation['value_scores'].items():
                    if score < 0.3:
                        self.adaptive_weights[value_name] = max(
                            0.1, 
                            self.adaptive_weights[value_name] * 0.95
                        )
        
        # Нормализация весов (чтобы сумма оставалась постоянной)
        total = sum(self.adaptive_weights.values())
        if total > 0:
            normalization_factor = sum(self.values.values()) / total
            for key in self.adaptive_weights:
                self.adaptive_weights[key] *= normalization_factor
    
    def get_value_stats(self) -> Dict[str, Any]:
        """Получение статистики ценностного фильтра"""
        
        recent_evaluations = list(self.value_history)[-100:] if self.value_history else []
        
        avg_values = {}
        if recent_evaluations:
            for eval_data in recent_evaluations:
                for value_name, score in eval_data['evaluation']['value_scores'].items():
                    if value_name not in avg_values:
                        avg_values[value_name] = []
                    avg_values[value_name].append(score)
            
            avg_values = {k: np.mean(v) for k, v in avg_values.items()}
        
        return {
            'current_weights': self.adaptive_weights,
            'base_weights': self.values,
            'total_evaluations': len(self.value_history),
            'recent_avg_scores': avg_values,
            'recommendation_distribution': self._get_recommendation_distribution()
        }
    
    def _get_recommendation_distribution(self) -> Dict[str, int]:
        """Распределение рекомендаций"""
        
        distribution = defaultdict(int)
        
        for eval_data in self.value_history:
            recommendation = eval_data['evaluation']['recommendation']
            distribution[recommendation] += 1
        
        return dict(distribution)

# ============================================================================
# ОБРАТНАЯ СВЯЗЬ И САМОКОРРЕКЦИЯ
# ============================================================================

class FeedbackLoopSystem:
    """Система обратной связи и самокоррекции"""
    
    def __init__(self, config: NeocortexConfig,
                 cognitive_integrator: CognitiveIntegrator,
                 focus_manager: FocusManager,
                 value_filter: ValueFilter):
        
        self.config = config
        self.integrator = cognitive_integrator
        self.focus = focus_manager
        self.value_filter = value_filter
        
        # Обратная связь
        self.feedback_history = deque(maxlen=500)
        self.correction_history = deque(maxlen=200)
        
        # Параметры адаптации
        self.adaptation_params = {
            'focus_adjustment_rate': 0.1,
            'learning_rate_adjustment': 0.05,
            'success_threshold': 0.7,
            'failure_threshold': 0.3,
            'feedback_window': 100  # Количество актов для анализа
        }
        
        # Модель успешности
        self.success_model = defaultdict(lambda: {'successes': 0, 'total': 0})
        
        logger.info("FeedbackLoopSystem инициализирована")
    
    async def process_feedback(self, act: CognitiveAct, 
                             integration_result: Dict[str, Any]):
        """Обработка обратной связи после интеграции"""
        
        feedback = {
            'act_id': act.id,
            'timestamp': datetime.utcnow(),
            'integration_success': True,  # Предполагаем успех
            'performance_metrics': {},
            'corrections_applied': [],
            'focus_adjustments': []
        }
        
        # Анализ успешности
        success_analysis = await self._analyze_success(act, integration_result)
        feedback['performance_metrics'] = success_analysis
        
        # Обновление модели успешности
        await self._update_success_model(act, success_analysis)
        
        # Применение коррекций при необходимости
        if success_analysis['overall_success'] < self.adaptation_params['failure_threshold']:
            corrections = await self._apply_corrections(act, success_analysis)
            feedback['corrections_applied'] = corrections
        
        # Корректировка фокуса внимания
        focus_adjustments = await self._adjust_focus_based_on_feedback(act, success_analysis)
        feedback['focus_adjustments'] = focus_adjustments
        
        # Запись обратной связи
        self.feedback_history.append(feedback)
        
        # Логирование
        if feedback['corrections_applied']:
            logger.info(f"Применены коррекции для акта {act.id}: "
                       f"{feedback['corrections_applied']}")
        
        return feedback
    
    async def _analyze_success(self, act: CognitiveAct,
                             integration_result: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ успешности когнитивного акта"""
        
        analysis = {
            'processing_success': True,
            'memory_success': True,
            'learning_success': False,
            'focus_success': True,
            'overall_success': 0.5
        }
        
        # Проверка успешности компонентов
        components = integration_result.get('components', {})
        
        # Успешность обработки памяти
        if 'memory' in components:
            analysis['memory_success'] = components['memory'].get('success', False)
        
        # Успешность обучения сети
        if 'network' in components:
            network_result = components['network']
            if 'novelty_score' in network_result:
                analysis['learning_success'] = network_result['novelty_score'] > 0.3
        
        # Успешность фокуса внимания
        if 'focus' in components:
            focus_result = components['focus']
            analysis['focus_success'] = focus_result.get('focus_updated', False)
        
        # Расчет общего успеха
        success_factors = [
            analysis['processing_success'] * 0.3,
            analysis['memory_success'] * 0.2,
            analysis['learning_success'] * 0.3,
            analysis['focus_success'] * 0.2
        ]
        
        analysis['overall_success'] = sum(success_factors)
        
        # Учет важности акта
        analysis['overall_success'] *= (0.5 + act.importance * 0.5)
        
        return analysis
    
    async def _update_success_model(self, act: CognitiveAct,
                                  success_analysis: Dict[str, Any]):
        """Обновление модели успешности"""
        
        key = f"{act.component}:{act.act_type}"
        
        self.success_model[key]['total'] += 1
        
        if success_analysis['overall_success'] > self.adaptation_params['success_threshold']:
            self.success_model[key]['successes'] += 1
        
        # Адаптация параметров на основе успешности
        success_rate = (self.success_model[key]['successes'] / 
                       max(1, self.success_model[key]['total']))
        
        if success_rate
