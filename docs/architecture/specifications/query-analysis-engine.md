# Query Analysis Engine Specification

## Overview

The Query Analysis Engine is the Master Agent's intelligent front-end that processes natural language queries, extracts intent and requirements, and provides structured analysis for optimal agent routing and knowledge retrieval.

## Architecture Components

### 1. NLP Processing Pipeline

```python
class NLPProcessor:
    def __init__(self, model_config):
        self.intent_classifier = IntentClassifier(model_config.intent_model)
        self.entity_extractor = EntityExtractor(model_config.entity_model)
        self.complexity_analyzer = ComplexityAnalyzer(model_config.complexity_model)
        self.domain_detector = DomainDetector(model_config.domain_model)
        self.capability_mapper = CapabilityMapper()
    
    async def process_query(self, query_text, context=None):
        """
        Process natural language query through complete NLP pipeline
        """
        
        # Step 1: Basic preprocessing
        cleaned_query = self.preprocess_text(query_text)
        
        # Step 2: Intent classification
        intent_analysis = await self.intent_classifier.classify(cleaned_query, context)
        
        # Step 3: Entity extraction
        entities = await self.entity_extractor.extract(cleaned_query, intent_analysis)
        
        # Step 4: Complexity analysis
        complexity_metrics = await self.complexity_analyzer.analyze(
            cleaned_query, intent_analysis, entities
        )
        
        # Step 5: Domain detection
        technical_domains = await self.domain_detector.detect(
            cleaned_query, entities, context
        )
        
        # Step 6: Capability mapping
        required_capabilities = await self.capability_mapper.map_requirements(
            intent_analysis, entities, technical_domains
        )
        
        return QueryAnalysis(
            original_query=query_text,
            processed_query=cleaned_query,
            intent=intent_analysis,
            entities=entities,
            complexity=complexity_metrics,
            technical_domains=technical_domains,
            required_capabilities=required_capabilities,
            confidence_score=self.calculate_overall_confidence(
                intent_analysis, entities, complexity_metrics
            )
        )
```

### 2. Intent Classification System

```python
class IntentClassifier:
    def __init__(self, model_path):
        self.model = load_classification_model(model_path)
        self.intent_categories = {
            # Development intents
            "code_generation": {
                "keywords": ["create", "generate", "build", "implement", "write", "develop"],
                "patterns": [r"create.*(?:function|class|component)", r"implement.*(?:feature|algorithm)"],
                "confidence_threshold": 0.8
            },
            "code_review": {
                "keywords": ["review", "check", "analyze", "audit", "validate", "inspect"],
                "patterns": [r"review.*code", r"check.*(?:quality|security|performance)"],
                "confidence_threshold": 0.75
            },
            "debugging": {
                "keywords": ["debug", "fix", "error", "issue", "problem", "bug", "troubleshoot"],
                "patterns": [r"fix.*(?:bug|error|issue)", r"debug.*(?:code|application)"],
                "confidence_threshold": 0.85
            },
            "refactoring": {
                "keywords": ["refactor", "optimize", "improve", "restructure", "reorganize"],
                "patterns": [r"refactor.*code", r"optimize.*(?:performance|structure)"],
                "confidence_threshold": 0.8
            },
            "testing": {
                "keywords": ["test", "verify", "validate", "check", "assert"],
                "patterns": [r"write.*test", r"create.*(?:unit|integration).*test"],
                "confidence_threshold": 0.9
            },
            
            # Architecture intents
            "system_design": {
                "keywords": ["design", "architecture", "system", "structure", "plan"],
                "patterns": [r"design.*(?:system|architecture)", r"create.*(?:architecture|design)"],
                "confidence_threshold": 0.7
            },
            "performance_analysis": {
                "keywords": ["performance", "optimization", "benchmark", "analyze", "profile"],
                "patterns": [r"analyze.*performance", r"optimize.*(?:speed|memory|cpu)"],
                "confidence_threshold": 0.8
            },
            
            # Project management intents  
            "task_management": {
                "keywords": ["task", "project", "manage", "organize", "plan", "schedule"],
                "patterns": [r"create.*task", r"manage.*project"],
                "confidence_threshold": 0.75
            },
            "documentation": {
                "keywords": ["document", "docs", "readme", "guide", "manual", "specification"],
                "patterns": [r"create.*(?:documentation|docs)", r"write.*(?:guide|manual)"],
                "confidence_threshold": 0.85
            },
            
            # Knowledge queries
            "knowledge_search": {
                "keywords": ["search", "find", "lookup", "query", "information", "how"],
                "patterns": [r"how.*(?:to|do)", r"what.*(?:is|are)", r"find.*(?:information|example)"],
                "confidence_threshold": 0.7
            },
            "explanation": {
                "keywords": ["explain", "describe", "clarify", "understand", "meaning"],
                "patterns": [r"explain.*(?:how|what|why)", r"what.*(?:does|means)"],
                "confidence_threshold": 0.8
            }
        }
    
    async def classify(self, query_text, context=None):
        """Classify query intent with confidence scoring"""
        
        # Rule-based classification
        rule_based_scores = self.rule_based_classification(query_text)
        
        # ML-based classification (if model available)
        ml_scores = await self.ml_classification(query_text, context)
        
        # Combine scores with weighting
        combined_scores = self.combine_classification_scores(
            rule_based_scores, ml_scores, weights=[0.6, 0.4]
        )
        
        # Get top intent
        primary_intent = max(combined_scores, key=combined_scores.get)
        confidence = combined_scores[primary_intent]
        
        # Get secondary intents (above threshold)
        secondary_intents = {
            intent: score for intent, score in combined_scores.items()
            if score > 0.3 and intent != primary_intent
        }
        
        return IntentAnalysis(
            primary_intent=primary_intent,
            confidence=confidence,
            secondary_intents=secondary_intents,
            all_scores=combined_scores
        )
    
    def rule_based_classification(self, query_text):
        """Rule-based intent classification using keywords and patterns"""
        scores = {}
        query_lower = query_text.lower()
        
        for intent, config in self.intent_categories.items():
            score = 0.0
            
            # Keyword matching
            keyword_matches = sum(1 for keyword in config["keywords"] 
                                if keyword in query_lower)
            keyword_score = min(keyword_matches / len(config["keywords"]), 1.0) * 0.6
            
            # Pattern matching
            pattern_matches = sum(1 for pattern in config.get("patterns", [])
                                if re.search(pattern, query_lower))
            pattern_score = min(pattern_matches / max(len(config.get("patterns", [])), 1), 1.0) * 0.4
            
            scores[intent] = keyword_score + pattern_score
        
        return scores
```

### 3. Entity Extraction System

```python
class EntityExtractor:
    def __init__(self, model_config):
        self.technical_entities = {
            "programming_languages": [
                "python", "javascript", "typescript", "java", "cpp", "c++", "rust", 
                "go", "php", "ruby", "swift", "kotlin", "scala", "haskell", "r"
            ],
            "frameworks": [
                "react", "vue", "angular", "svelte", "django", "flask", "fastapi",
                "spring", "express", "nest", "next", "nuxt", "laravel", "rails"
            ],
            "technologies": [
                "docker", "kubernetes", "aws", "azure", "gcp", "terraform",
                "jenkins", "github", "gitlab", "redis", "mongodb", "postgresql"
            ],
            "concepts": [
                "api", "rest", "graphql", "microservices", "authentication", "oauth",
                "jwt", "database", "cache", "queue", "websocket", "ci/cd", "devops"
            ],
            "file_types": [
                ".py", ".js", ".ts", ".java", ".cpp", ".rs", ".go", ".php",
                ".html", ".css", ".json", ".xml", ".yaml", ".yml", ".md"
            ]
        }
    
    async def extract(self, query_text, intent_analysis=None):
        """Extract technical entities from query"""
        
        entities = {
            "programming_languages": [],
            "frameworks": [],
            "technologies": [],
            "concepts": [],
            "file_types": [],
            "custom_entities": []
        }
        
        query_lower = query_text.lower()
        
        # Extract predefined technical entities
        for category, entity_list in self.technical_entities.items():
            for entity in entity_list:
                if entity.lower() in query_lower:
                    entities[category].append({
                        "entity": entity,
                        "position": query_lower.find(entity.lower()),
                        "confidence": 0.9
                    })
        
        # Extract custom entities using NER (if available)
        custom_entities = await self.extract_custom_entities(query_text)
        entities["custom_entities"] = custom_entities
        
        # Extract numerical entities (file sizes, quantities, etc.)
        numerical_entities = self.extract_numerical_entities(query_text)
        entities["numerical"] = numerical_entities
        
        return entities
    
    async def extract_custom_entities(self, query_text):
        """Extract domain-specific entities using NER"""
        # This would use a trained NER model for software development
        # For now, using pattern-based extraction
        
        patterns = {
            "function_names": r"\b[a-zA-Z_][a-zA-Z0-9_]*\(\)",
            "class_names": r"\bclass\s+([A-Z][a-zA-Z0-9_]*)",
            "file_paths": r"[a-zA-Z0-9_\-./]+\.[a-zA-Z]{2,4}",
            "urls": r"https?://[^\s]+",
            "version_numbers": r"\b\d+\.\d+(?:\.\d+)?\b"
        }
        
        custom_entities = []
        for entity_type, pattern in patterns.items():
            matches = re.finditer(pattern, query_text)
            for match in matches:
                custom_entities.append({
                    "type": entity_type,
                    "value": match.group(),
                    "position": match.start(),
                    "confidence": 0.8
                })
        
        return custom_entities
```

### 4. Complexity Analysis System

```python
class ComplexityAnalyzer:
    def __init__(self, config):
        self.complexity_indicators = {
            "simple": {
                "keywords": ["simple", "basic", "quick", "small", "easy"],
                "max_entities": 3,
                "max_domains": 2,
                "typical_duration": "< 1 hour"
            },
            "medium": {
                "keywords": ["medium", "moderate", "standard", "typical"],
                "max_entities": 8,
                "max_domains": 4,
                "typical_duration": "1-4 hours"
            },
            "complex": {
                "keywords": ["complex", "advanced", "comprehensive", "enterprise", "large", "full"],
                "max_entities": 15,
                "max_domains": 8,
                "typical_duration": "4+ hours"
            }
        }
    
    async def analyze(self, query_text, intent_analysis, entities):
        """Analyze query complexity using multiple factors"""
        
        complexity_scores = {}
        
        # Factor 1: Keyword-based complexity
        keyword_scores = self.analyze_complexity_keywords(query_text)
        
        # Factor 2: Entity count complexity
        entity_count = sum(len(entity_list) for entity_list in entities.values())
        entity_complexity = self.classify_entity_complexity(entity_count)
        
        # Factor 3: Intent complexity
        intent_complexity = self.classify_intent_complexity(intent_analysis)
        
        # Factor 4: Query length complexity
        length_complexity = self.classify_length_complexity(query_text)
        
        # Combine factors with weights
        for complexity_level in ["simple", "medium", "complex"]:
            complexity_scores[complexity_level] = (
                keyword_scores.get(complexity_level, 0) * 0.3 +
                entity_complexity.get(complexity_level, 0) * 0.25 +
                intent_complexity.get(complexity_level, 0) * 0.3 +
                length_complexity.get(complexity_level, 0) * 0.15
            )
        
        # Determine primary complexity level
        primary_complexity = max(complexity_scores, key=complexity_scores.get)
        confidence = complexity_scores[primary_complexity]
        
        # Calculate resource requirements
        resource_requirements = self.calculate_resource_requirements(
            primary_complexity, entity_count, intent_analysis
        )
        
        return ComplexityMetrics(
            level=primary_complexity,
            confidence=confidence,
            all_scores=complexity_scores,
            entity_count=entity_count,
            estimated_duration=self.complexity_indicators[primary_complexity]["typical_duration"],
            resource_requirements=resource_requirements,
            parallel_processing_recommended=entity_count > 5 or primary_complexity == "complex"
        )
    
    def classify_intent_complexity(self, intent_analysis):
        """Classify complexity based on intent type"""
        
        intent_complexity_map = {
            # Simple intents
            "knowledge_search": "simple",
            "explanation": "simple",
            "debugging": "simple",
            
            # Medium intents
            "code_generation": "medium", 
            "code_review": "medium",
            "testing": "medium",
            "documentation": "medium",
            
            # Complex intents
            "system_design": "complex",
            "refactoring": "complex",
            "performance_analysis": "complex",
            "task_management": "complex"
        }
        
        primary_intent = intent_analysis.primary_intent
        mapped_complexity = intent_complexity_map.get(primary_intent, "medium")
        
        scores = {"simple": 0, "medium": 0, "complex": 0}
        scores[mapped_complexity] = intent_analysis.confidence
        
        # Adjust for secondary intents
        for secondary_intent, confidence in intent_analysis.secondary_intents.items():
            secondary_complexity = intent_complexity_map.get(secondary_intent, "medium")
            scores[secondary_complexity] += confidence * 0.3
        
        return scores
    
    def calculate_resource_requirements(self, complexity_level, entity_count, intent_analysis):
        """Calculate estimated resource requirements"""
        
        base_requirements = {
            "simple": {"agents": 1, "memory_mb": 256, "cpu_cores": 1, "duration_minutes": 15},
            "medium": {"agents": 2, "memory_mb": 512, "cpu_cores": 2, "duration_minutes": 60},
            "complex": {"agents": 4, "memory_mb": 1024, "cpu_cores": 4, "duration_minutes": 240}
        }
        
        requirements = base_requirements[complexity_level].copy()
        
        # Adjust based on entity count
        entity_multiplier = min(entity_count / 5.0, 2.0)
        requirements["agents"] = int(requirements["agents"] * entity_multiplier)
        requirements["memory_mb"] = int(requirements["memory_mb"] * entity_multiplier)
        
        # Adjust based on intent
        if intent_analysis.primary_intent in ["system_design", "performance_analysis"]:
            requirements["agents"] += 1
            requirements["duration_minutes"] = int(requirements["duration_minutes"] * 1.5)
        
        return requirements
```

### 5. Domain Detection System

```python
class DomainDetector:
    def __init__(self, config):
        self.domain_mappings = {
            "web_development": {
                "technologies": ["html", "css", "javascript", "react", "vue", "angular"],
                "concepts": ["frontend", "backend", "api", "rest", "graphql", "spa"],
                "frameworks": ["express", "django", "flask", "next", "nuxt"]
            },
            "mobile_development": {
                "technologies": ["swift", "kotlin", "react-native", "flutter", "xamarin"],
                "concepts": ["ios", "android", "mobile", "app", "cross-platform"],
                "frameworks": ["ionic", "cordova", "flutter"]
            },
            "data_science": {
                "technologies": ["python", "r", "sql", "pandas", "numpy", "tensorflow"],
                "concepts": ["machine learning", "ai", "analysis", "visualization", "model"],
                "frameworks": ["scikit-learn", "pytorch", "keras", "matplotlib"]
            },
            "devops": {
                "technologies": ["docker", "kubernetes", "aws", "azure", "terraform"],
                "concepts": ["ci/cd", "deployment", "infrastructure", "monitoring", "scaling"],
                "frameworks": ["jenkins", "gitlab-ci", "github-actions"]
            },
            "backend_development": {
                "technologies": ["node", "python", "java", "go", "rust"],
                "concepts": ["api", "database", "microservices", "authentication", "scaling"],
                "frameworks": ["express", "fastapi", "spring", "gin"]
            },
            "database": {
                "technologies": ["sql", "nosql", "postgresql", "mongodb", "redis"],
                "concepts": ["query", "schema", "optimization", "indexing", "migration"],
                "frameworks": ["prisma", "sequelize", "mongoose"]
            }
        }
    
    async def detect(self, query_text, entities, context=None):
        """Detect technical domains from query and entities"""
        
        domain_scores = {}
        query_lower = query_text.lower()
        
        # Score domains based on entity matches
        for domain, domain_config in self.domain_mappings.items():
            score = 0.0
            
            # Check technology matches
            for tech_category, tech_list in entities.items():
                if tech_category in domain_config:
                    matches = len([e for e in tech_list if e["entity"] in domain_config[tech_category]])
                    score += matches * 0.4
            
            # Check keyword matches
            for category, items in domain_config.items():
                keyword_matches = sum(1 for item in items if item in query_lower)
                score += keyword_matches * 0.2
            
            domain_scores[domain] = min(score, 1.0)
        
        # Filter domains above threshold
        relevant_domains = [domain for domain, score in domain_scores.items() if score > 0.3]
        
        # Add context-based domains if available
        if context and hasattr(context, 'project_domains'):
            for domain in context.project_domains:
                if domain not in relevant_domains:
                    relevant_domains.append(domain)
                    domain_scores[domain] = 0.6  # Context-based score
        
        return sorted(relevant_domains, key=lambda d: domain_scores[d], reverse=True)
```

### 6. Context Analysis and Memory Integration

```python
class ContextAnalyzer:
    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.conversation_context = {}
        self.project_context = {}
    
    async def analyze_context(self, query, user_id, session_id=None):
        """Analyze query context from conversation history and project state"""
        
        # Get conversation context
        conversation_history = await self.memory_manager.get_conversation_history(
            user_id, session_id, limit=10
        )
        
        # Get project context
        current_project = await self.memory_manager.get_current_project(user_id)
        
        # Analyze conversation flow
        conversation_context = self.analyze_conversation_flow(
            query, conversation_history
        )
        
        # Analyze project relevance
        project_context = self.analyze_project_relevance(
            query, current_project
        )
        
        return ContextAnalysis(
            conversation_context=conversation_context,
            project_context=project_context,
            session_continuity=self.calculate_session_continuity(conversation_history),
            preferred_agents=self.get_preferred_agents(user_id),
            working_directory=current_project.get("directory") if current_project else None,
            recent_technologies=self.extract_recent_technologies(conversation_history)
        )
    
    def analyze_conversation_flow(self, current_query, history):
        """Analyze conversation flow for context awareness"""
        
        if not history:
            return {"type": "new_conversation", "continuity": 0.0}
        
        last_query = history[-1].get("query", "")
        last_intent = history[-1].get("intent", "")
        
        # Check for continuation patterns
        continuation_patterns = [
            r"^(also|additionally|furthermore|moreover)",
            r"^(and|but|however|though)",
            r"^(now|next|then|after)"
        ]
        
        is_continuation = any(
            re.match(pattern, current_query.lower()) 
            for pattern in continuation_patterns
        )
        
        # Check for reference patterns
        reference_patterns = [
            r"(that|this|it|the above|previous)",
            r"(same|similar|like before)"
        ]
        
        has_references = any(
            re.search(pattern, current_query.lower())
            for pattern in reference_patterns
        )
        
        # Calculate context relevance
        relevance_score = self.calculate_context_relevance(current_query, history)
        
        return {
            "type": "continuation" if is_continuation else "new_topic",
            "has_references": has_references,
            "relevance_score": relevance_score,
            "last_intent": last_intent,
            "conversation_length": len(history)
        }
```

## Integration with Master Agent

### Query Processing Pipeline

```python
class QueryProcessor:
    def __init__(self, nlp_processor, context_analyzer, rag_client):
        self.nlp_processor = nlp_processor
        self.context_analyzer = context_analyzer
        self.rag_client = rag_client
    
    async def process_user_query(self, query, user_context):
        """Complete query processing pipeline"""
        
        # Step 1: Context analysis
        context_analysis = await self.context_analyzer.analyze_context(
            query, user_context.user_id, user_context.session_id
        )
        
        # Step 2: NLP processing
        query_analysis = await self.nlp_processor.process_query(
            query, context_analysis
        )
        
        # Step 3: Knowledge retrieval planning
        knowledge_requirements = self.plan_knowledge_retrieval(
            query_analysis, context_analysis
        )
        
        # Step 4: Agent requirements analysis
        agent_requirements = self.analyze_agent_requirements(
            query_analysis, context_analysis, knowledge_requirements
        )
        
        return ProcessedQuery(
            original_query=query,
            query_analysis=query_analysis,
            context_analysis=context_analysis,
            knowledge_requirements=knowledge_requirements,
            agent_requirements=agent_requirements,
            processing_timestamp=datetime.utcnow(),
            confidence_score=self.calculate_overall_confidence(query_analysis, context_analysis)
        )
    
    def plan_knowledge_retrieval(self, query_analysis, context_analysis):
        """Plan RAG queries based on analysis results"""
        
        rag_queries = []
        
        # Primary RAG query based on intent and entities
        primary_query = self.construct_primary_rag_query(query_analysis)
        rag_queries.append({
            "query": primary_query,
            "type": "primary",
            "source_domains": self.determine_source_domains(query_analysis),
            "priority": 1
        })
        
        # Secondary queries for context and examples
        if query_analysis.complexity.level in ["medium", "complex"]:
            example_query = self.construct_example_query(query_analysis)
            rag_queries.append({
                "query": example_query,
                "type": "examples",
                "source_domains": ["code_examples"],
                "priority": 2
            })
        
        # Context-specific queries
        if context_analysis.project_context:
            context_query = self.construct_context_query(
                query_analysis, context_analysis.project_context
            )
            rag_queries.append({
                "query": context_query,
                "type": "contextual",
                "source_domains": ["project_docs"],
                "priority": 3
            })
        
        return rag_queries
```

This Query Analysis Engine provides sophisticated natural language understanding capabilities, enabling the Master Agent to make intelligent routing decisions based on comprehensive query analysis, context awareness, and knowledge requirements planning.