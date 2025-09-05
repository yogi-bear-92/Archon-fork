# RAG Integration Layer Specification

## Overview

The RAG Integration Layer serves as the intelligent knowledge retrieval system that seamlessly integrates Archon's knowledge base with the Master Agent's decision-making process, providing contextual information to enhance agent performance and user experience.

## Architecture Components

### 1. Knowledge Query Router

```python
class KnowledgeQueryRouter:
    def __init__(self, archon_client, claude_flow_client, config):
        self.archon_client = archon_client
        self.claude_flow_client = claude_flow_client
        self.query_strategies = self.initialize_query_strategies()
        self.fallback_sources = config.fallback_sources
        self.cache = KnowledgeCache(config.cache_config)
    
    def initialize_query_strategies(self):
        """Initialize different RAG query strategies"""
        return {
            "contextual_embeddings": ContextualEmbeddingStrategy(),
            "hybrid_search": HybridSearchStrategy(),
            "code_example_search": CodeExampleStrategy(),
            "documentation_search": DocumentationStrategy(),
            "project_context_search": ProjectContextStrategy()
        }
    
    async def route_knowledge_query(self, query_requirements, context):
        """Route knowledge queries to optimal sources and strategies"""
        
        results = []
        
        for requirement in query_requirements:
            # Check cache first
            cache_key = self.generate_cache_key(requirement, context)
            cached_result = await self.cache.get(cache_key)
            
            if cached_result and not cached_result.is_expired():
                results.append(cached_result)
                continue
            
            # Route to appropriate strategy
            strategy = self.select_strategy(requirement, context)
            result = await self.execute_strategy(strategy, requirement, context)
            
            # Cache successful results
            if result.success and result.confidence > 0.6:
                await self.cache.set(cache_key, result, ttl=3600)
            
            results.append(result)
        
        return self.consolidate_results(results)
    
    def select_strategy(self, requirement, context):
        """Select optimal RAG strategy based on requirement type"""
        
        if requirement.type == "code_examples":
            return self.query_strategies["code_example_search"]
        elif requirement.type == "contextual":
            return self.query_strategies["contextual_embeddings"]
        elif requirement.type == "documentation":
            return self.query_strategies["documentation_search"]
        elif requirement.type == "project_specific":
            return self.query_strategies["project_context_search"]
        else:
            # Default to hybrid search for complex queries
            return self.query_strategies["hybrid_search"]
    
    async def execute_strategy(self, strategy, requirement, context):
        """Execute RAG strategy with fallback handling"""
        
        try:
            # Primary strategy execution
            result = await strategy.execute(
                requirement, context, self.archon_client
            )
            
            if result.confidence >= 0.7:
                return result
            
            # Fallback to hybrid search if primary strategy confidence is low
            if strategy != self.query_strategies["hybrid_search"]:
                fallback_result = await self.query_strategies["hybrid_search"].execute(
                    requirement, context, self.archon_client
                )
                
                # Merge results if both have moderate confidence
                if result.confidence > 0.4 and fallback_result.confidence > 0.4:
                    return self.merge_results([result, fallback_result])
                
                # Return better result
                return result if result.confidence > fallback_result.confidence else fallback_result
            
            return result
            
        except Exception as e:
            # Fallback to basic search on errors
            return await self.execute_fallback_search(requirement, context, str(e))
```

### 2. Contextual Embedding Strategy

```python
class ContextualEmbeddingStrategy:
    def __init__(self):
        self.embedding_model = "text-embedding-3-small"
        self.context_window_size = 512
        self.semantic_similarity_threshold = 0.75
    
    async def execute(self, requirement, context, archon_client):
        """Execute contextual embedding-based RAG query"""
        
        # Construct contextual query
        contextual_query = self.build_contextual_query(requirement, context)
        
        # Execute RAG query with contextual embeddings
        rag_result = await archon_client.perform_rag_query(
            query=contextual_query,
            source_domain=self.determine_source_domain(requirement, context),
            match_count=5
        )
        
        if not rag_result.get("success"):
            return RAGResult(
                success=False,
                results=[],
                confidence=0.0,
                error=rag_result.get("error")
            )
        
        # Post-process results for contextual relevance
        processed_results = await self.post_process_results(
            rag_result["results"], requirement, context
        )
        
        # Calculate overall confidence
        confidence = self.calculate_confidence(processed_results, requirement)
        
        return RAGResult(
            success=True,
            results=processed_results,
            confidence=confidence,
            strategy="contextual_embeddings",
            query_used=contextual_query
        )
    
    def build_contextual_query(self, requirement, context):
        """Build context-aware query string"""
        
        base_query = requirement.query
        
        # Add project context
        if context.project_context:
            project_keywords = self.extract_project_keywords(context.project_context)
            base_query = f"{base_query} in context of {' '.join(project_keywords[:3])}"
        
        # Add technology context
        if context.recent_technologies:
            tech_context = ', '.join(context.recent_technologies[:2])
            base_query = f"{base_query} with {tech_context}"
        
        # Add intent context
        if hasattr(requirement, 'intent_context'):
            intent_keywords = self.get_intent_keywords(requirement.intent_context)
            base_query = f"{intent_keywords} {base_query}"
        
        return base_query
    
    async def post_process_results(self, raw_results, requirement, context):
        """Post-process RAG results for contextual relevance"""
        
        processed = []
        
        for result in raw_results:
            # Calculate contextual relevance score
            relevance_score = await self.calculate_contextual_relevance(
                result, requirement, context
            )
            
            # Enhance result with contextual metadata
            enhanced_result = {
                **result,
                "contextual_relevance": relevance_score,
                "context_factors": self.analyze_context_factors(result, context),
                "applicability_score": self.calculate_applicability(result, requirement)
            }
            
            processed.append(enhanced_result)
        
        # Sort by combined relevance score
        processed.sort(
            key=lambda x: (x["contextual_relevance"] + x["applicability_score"]) / 2,
            reverse=True
        )
        
        return processed[:3]  # Return top 3 most contextually relevant results
    
    async def calculate_contextual_relevance(self, result, requirement, context):
        """Calculate contextual relevance using multiple factors"""
        
        relevance_factors = {}
        
        # Project alignment factor
        if context.project_context:
            project_alignment = self.calculate_project_alignment(
                result["content"], context.project_context
            )
            relevance_factors["project_alignment"] = project_alignment * 0.3
        
        # Technology stack alignment
        if context.recent_technologies:
            tech_alignment = self.calculate_tech_alignment(
                result["content"], context.recent_technologies
            )
            relevance_factors["tech_alignment"] = tech_alignment * 0.25
        
        # Intent alignment
        intent_alignment = self.calculate_intent_alignment(
            result["content"], requirement
        )
        relevance_factors["intent_alignment"] = intent_alignment * 0.25
        
        # Recency factor (prefer recent documentation)
        recency_factor = self.calculate_recency_factor(result.get("metadata", {}))
        relevance_factors["recency"] = recency_factor * 0.2
        
        return sum(relevance_factors.values())
```

### 3. Code Example Search Strategy

```python
class CodeExampleStrategy:
    def __init__(self):
        self.code_patterns = {
            "function_definition": r"def\s+\w+\s*\([^)]*\):",
            "class_definition": r"class\s+\w+\s*(\([^)]*\))?:",
            "api_usage": r"\w+\.\w+\([^)]*\)",
            "configuration": r"\w+\s*=\s*{[^}]*}",
            "import_statement": r"(?:from\s+\w+\s+)?import\s+\w+"
        }
    
    async def execute(self, requirement, context, archon_client):
        """Execute code example search with pattern matching"""
        
        # Search for code examples
        code_results = await archon_client.search_code_examples(
            query=requirement.query,
            source_domain=self.determine_code_source_domain(requirement, context),
            match_count=8
        )
        
        if not code_results.get("success"):
            return RAGResult(
                success=False,
                results=[],
                confidence=0.0,
                error=code_results.get("error")
            )
        
        # Analyze and rank code examples
        analyzed_examples = await self.analyze_code_examples(
            code_results["results"], requirement, context
        )
        
        # Filter by relevance and quality
        filtered_examples = self.filter_code_examples(analyzed_examples, requirement)
        
        confidence = self.calculate_code_confidence(filtered_examples, requirement)
        
        return RAGResult(
            success=True,
            results=filtered_examples,
            confidence=confidence,
            strategy="code_example_search",
            metadata={"code_patterns_found": self.detected_patterns}
        )
    
    async def analyze_code_examples(self, raw_examples, requirement, context):
        """Analyze code examples for quality and relevance"""
        
        analyzed = []
        self.detected_patterns = {}
        
        for example in raw_examples:
            content = example.get("content", "")
            
            # Detect code patterns
            patterns_found = self.detect_code_patterns(content)
            self.detected_patterns[example.get("id", "")] = patterns_found
            
            # Calculate code quality score
            quality_score = self.assess_code_quality(content)
            
            # Calculate relevance to requirement
            relevance_score = self.calculate_code_relevance(
                content, patterns_found, requirement, context
            )
            
            # Extract key information
            key_concepts = self.extract_code_concepts(content, patterns_found)
            
            analyzed_example = {
                **example,
                "patterns_found": patterns_found,
                "quality_score": quality_score,
                "relevance_score": relevance_score,
                "key_concepts": key_concepts,
                "complexity_level": self.assess_code_complexity(content),
                "framework_detected": self.detect_framework(content),
                "best_practices_score": self.assess_best_practices(content)
            }
            
            analyzed.append(analyzed_example)
        
        return analyzed
    
    def detect_code_patterns(self, code_content):
        """Detect common code patterns in content"""
        
        patterns_found = {}
        
        for pattern_name, pattern_regex in self.code_patterns.items():
            matches = re.findall(pattern_regex, code_content)
            if matches:
                patterns_found[pattern_name] = len(matches)
        
        # Detect additional patterns
        if "async" in code_content and "await" in code_content:
            patterns_found["async_pattern"] = code_content.count("async")
        
        if "try:" in code_content and "except" in code_content:
            patterns_found["error_handling"] = code_content.count("try:")
        
        if any(test_indicator in code_content.lower() 
               for test_indicator in ["test_", "assert", "unittest", "pytest"]):
            patterns_found["test_code"] = 1
        
        return patterns_found
    
    def assess_code_quality(self, code_content):
        """Assess code quality based on multiple factors"""
        
        quality_factors = {}
        
        # Documentation factor (docstrings, comments)
        doc_lines = len([line for line in code_content.split('\n') 
                        if line.strip().startswith(('"""', "'''", '#'))])
        total_lines = len([line for line in code_content.split('\n') if line.strip()])
        quality_factors["documentation"] = min(doc_lines / max(total_lines, 1) * 4, 1.0)
        
        # Error handling factor
        has_error_handling = any(keyword in code_content 
                               for keyword in ["try:", "except", "raise", "assert"])
        quality_factors["error_handling"] = 1.0 if has_error_handling else 0.3
        
        # Type hints factor (for Python)
        type_hints = code_content.count(":") - code_content.count(":")  # Simplified check
        quality_factors["type_hints"] = min(type_hints / max(total_lines * 0.3, 1), 1.0)
        
        # Structure factor (classes, functions)
        has_functions = "def " in code_content
        has_classes = "class " in code_content
        quality_factors["structure"] = 1.0 if has_functions or has_classes else 0.5
        
        return sum(quality_factors.values()) / len(quality_factors)
    
    def filter_code_examples(self, analyzed_examples, requirement):
        """Filter code examples by relevance and quality thresholds"""
        
        filtered = []
        
        for example in analyzed_examples:
            # Quality threshold
            if example["quality_score"] < 0.4:
                continue
            
            # Relevance threshold
            if example["relevance_score"] < 0.5:
                continue
            
            # Complexity alignment
            if requirement.complexity_level == "simple" and example["complexity_level"] == "complex":
                continue
            
            filtered.append(example)
        
        # Sort by combined score
        filtered.sort(
            key=lambda x: (x["quality_score"] + x["relevance_score"]) / 2,
            reverse=True
        )
        
        return filtered[:5]  # Return top 5 examples
```

### 4. Hybrid Search Strategy

```python
class HybridSearchStrategy:
    def __init__(self):
        self.semantic_weight = 0.6
        self.keyword_weight = 0.4
        self.min_results_threshold = 3
    
    async def execute(self, requirement, context, archon_client):
        """Execute hybrid search combining semantic and keyword search"""
        
        # Execute semantic search
        semantic_results = await self.execute_semantic_search(
            requirement, context, archon_client
        )
        
        # Execute keyword search
        keyword_results = await self.execute_keyword_search(
            requirement, context, archon_client
        )
        
        # Merge and rank results
        merged_results = self.merge_search_results(
            semantic_results, keyword_results, requirement
        )
        
        # Calculate hybrid confidence
        confidence = self.calculate_hybrid_confidence(
            semantic_results, keyword_results, merged_results
        )
        
        return RAGResult(
            success=True,
            results=merged_results,
            confidence=confidence,
            strategy="hybrid_search",
            metadata={
                "semantic_results_count": len(semantic_results),
                "keyword_results_count": len(keyword_results),
                "merged_results_count": len(merged_results)
            }
        )
    
    async def execute_semantic_search(self, requirement, context, archon_client):
        """Execute semantic search using embeddings"""
        
        enhanced_query = self.enhance_query_for_semantic_search(requirement, context)
        
        result = await archon_client.perform_rag_query(
            query=enhanced_query,
            source_domain=None,  # Search all domains
            match_count=8
        )
        
        return result.get("results", []) if result.get("success") else []
    
    async def execute_keyword_search(self, requirement, context, archon_client):
        """Execute keyword-based search"""
        
        keywords = self.extract_keywords(requirement, context)
        keyword_query = " ".join(keywords)
        
        # Use a more targeted search for keywords
        result = await archon_client.perform_rag_query(
            query=keyword_query,
            source_domain=self.determine_keyword_domain(requirement, context),
            match_count=5
        )
        
        return result.get("results", []) if result.get("success") else []
    
    def merge_search_results(self, semantic_results, keyword_results, requirement):
        """Merge semantic and keyword search results with intelligent deduplication"""
        
        # Create result map for deduplication
        result_map = {}
        
        # Add semantic results with semantic weight
        for result in semantic_results:
            result_id = self.generate_result_id(result)
            semantic_score = result.get("similarity_score", 0) * self.semantic_weight
            
            result_map[result_id] = {
                **result,
                "semantic_score": semantic_score,
                "keyword_score": 0.0,
                "hybrid_score": semantic_score,
                "sources": ["semantic"]
            }
        
        # Add keyword results with keyword weight
        for result in keyword_results:
            result_id = self.generate_result_id(result)
            keyword_score = result.get("similarity_score", 0) * self.keyword_weight
            
            if result_id in result_map:
                # Merge with existing semantic result
                result_map[result_id]["keyword_score"] = keyword_score
                result_map[result_id]["hybrid_score"] += keyword_score
                result_map[result_id]["sources"].append("keyword")
            else:
                # Add as new keyword-only result
                result_map[result_id] = {
                    **result,
                    "semantic_score": 0.0,
                    "keyword_score": keyword_score,
                    "hybrid_score": keyword_score,
                    "sources": ["keyword"]
                }
        
        # Sort by hybrid score
        merged_results = list(result_map.values())
        merged_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        return merged_results[:6]  # Return top 6 results
```

### 5. Context Enrichment Engine

```python
class ContextEnrichmentEngine:
    def __init__(self, archon_client):
        self.archon_client = archon_client
        self.enrichment_strategies = {
            "project_context": self.enrich_with_project_context,
            "code_examples": self.enrich_with_code_examples,
            "related_concepts": self.enrich_with_related_concepts,
            "best_practices": self.enrich_with_best_practices
        }
    
    async def enrich_knowledge_results(self, rag_results, query_analysis, context):
        """Enrich RAG results with additional contextual information"""
        
        enriched_results = []
        
        for result in rag_results:
            enriched_result = result.copy()
            
            # Apply enrichment strategies
            for strategy_name, strategy_func in self.enrichment_strategies.items():
                try:
                    enrichment = await strategy_func(result, query_analysis, context)
                    if enrichment:
                        enriched_result[f"enrichment_{strategy_name}"] = enrichment
                except Exception as e:
                    # Log error but continue with other enrichments
                    print(f"Enrichment strategy {strategy_name} failed: {e}")
            
            # Calculate enrichment score
            enrichment_score = self.calculate_enrichment_score(enriched_result)
            enriched_result["enrichment_score"] = enrichment_score
            
            enriched_results.append(enriched_result)
        
        return enriched_results
    
    async def enrich_with_project_context(self, result, query_analysis, context):
        """Enrich result with relevant project context"""
        
        if not context.project_context:
            return None
        
        # Search for related project documentation
        project_query = f"project {context.project_context.get('name', '')} {result.get('content', '')[:100]}"
        
        project_docs = await self.archon_client.perform_rag_query(
            query=project_query,
            source_domain="project_docs",
            match_count=2
        )
        
        if project_docs.get("success") and project_docs.get("results"):
            return {
                "related_project_docs": project_docs["results"],
                "project_relevance_score": self.calculate_project_relevance(
                    result, project_docs["results"]
                )
            }
        
        return None
    
    async def enrich_with_code_examples(self, result, query_analysis, context):
        """Enrich result with relevant code examples"""
        
        # Extract technical terms from result content
        technical_terms = self.extract_technical_terms(result.get("content", ""))
        
        if not technical_terms:
            return None
        
        # Search for code examples
        code_query = " ".join(technical_terms[:3])  # Use top 3 technical terms
        
        code_examples = await self.archon_client.search_code_examples(
            query=code_query,
            match_count=3
        )
        
        if code_examples.get("success") and code_examples.get("results"):
            return {
                "related_code_examples": code_examples["results"],
                "code_relevance_score": self.calculate_code_relevance(
                    result, code_examples["results"]
                )
            }
        
        return None
    
    async def enrich_with_best_practices(self, result, query_analysis, context):
        """Enrich result with relevant best practices"""
        
        # Identify domains from result content
        domains = self.identify_domains_from_content(result.get("content", ""))
        
        if not domains:
            return None
        
        best_practices_results = []
        
        for domain in domains[:2]:  # Limit to top 2 domains
            practices_query = f"best practices {domain} guidelines standards"
            
            practices = await self.archon_client.perform_rag_query(
                query=practices_query,
                source_domain="documentation",
                match_count=2
            )
            
            if practices.get("success") and practices.get("results"):
                best_practices_results.extend(practices["results"])
        
        if best_practices_results:
            return {
                "best_practices": best_practices_results,
                "practices_relevance_score": self.calculate_practices_relevance(
                    result, best_practices_results
                )
            }
        
        return None
```

### 6. Knowledge Cache Manager

```python
class KnowledgeCache:
    def __init__(self, config):
        self.cache_backend = self.initialize_cache_backend(config)
        self.default_ttl = config.get("default_ttl", 3600)  # 1 hour
        self.max_cache_size = config.get("max_size", 1000)
        self.cache_hit_stats = {"hits": 0, "misses": 0}
    
    def generate_cache_key(self, requirement, context):
        """Generate cache key from requirement and context"""
        
        key_components = [
            requirement.get("query", ""),
            requirement.get("type", ""),
            str(sorted(requirement.get("domains", []))),
            context.get("user_id", ""),
            context.get("project_id", "")
        ]
        
        # Create hash from components
        key_string = "|".join(str(component) for component in key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get(self, cache_key):
        """Retrieve cached result"""
        
        try:
            cached_data = await self.cache_backend.get(cache_key)
            
            if cached_data:
                self.cache_hit_stats["hits"] += 1
                return CachedResult.from_json(cached_data)
            else:
                self.cache_hit_stats["misses"] += 1
                return None
                
        except Exception as e:
            print(f"Cache retrieval error: {e}")
            self.cache_hit_stats["misses"] += 1
            return None
    
    async def set(self, cache_key, result, ttl=None):
        """Cache result with TTL"""
        
        try:
            cache_ttl = ttl or self.default_ttl
            cached_result = CachedResult(
                result=result,
                cached_at=datetime.utcnow(),
                ttl=cache_ttl
            )
            
            await self.cache_backend.set(
                cache_key, 
                cached_result.to_json(), 
                expire=cache_ttl
            )
            
        except Exception as e:
            print(f"Cache storage error: {e}")
    
    def get_cache_stats(self):
        """Get cache performance statistics"""
        
        total_requests = self.cache_hit_stats["hits"] + self.cache_hit_stats["misses"]
        hit_rate = self.cache_hit_stats["hits"] / max(total_requests, 1)
        
        return {
            "hit_rate": hit_rate,
            "total_hits": self.cache_hit_stats["hits"],
            "total_misses": self.cache_hit_stats["misses"],
            "total_requests": total_requests
        }

class CachedResult:
    def __init__(self, result, cached_at, ttl):
        self.result = result
        self.cached_at = cached_at
        self.ttl = ttl
    
    def is_expired(self):
        """Check if cached result has expired"""
        expiry_time = self.cached_at + timedelta(seconds=self.ttl)
        return datetime.utcnow() > expiry_time
    
    def to_json(self):
        """Serialize to JSON"""
        return json.dumps({
            "result": self.result,
            "cached_at": self.cached_at.isoformat(),
            "ttl": self.ttl
        })
    
    @classmethod
    def from_json(cls, json_data):
        """Deserialize from JSON"""
        data = json.loads(json_data)
        return cls(
            result=data["result"],
            cached_at=datetime.fromisoformat(data["cached_at"]),
            ttl=data["ttl"]
        )
```

## Integration with Master Agent

### RAG Workflow Integration

```python
class RAGWorkflowOrchestrator:
    def __init__(self, knowledge_router, enrichment_engine, cache_manager):
        self.knowledge_router = knowledge_router
        self.enrichment_engine = enrichment_engine
        self.cache_manager = cache_manager
    
    async def execute_rag_workflow(self, query_analysis, context, agent_requirements):
        """Execute complete RAG workflow for agent enhancement"""
        
        # Step 1: Plan knowledge retrieval
        knowledge_requirements = self.plan_knowledge_retrieval(
            query_analysis, context, agent_requirements
        )
        
        # Step 2: Execute knowledge queries
        rag_results = await self.knowledge_router.route_knowledge_query(
            knowledge_requirements, context
        )
        
        # Step 3: Enrich results with additional context
        enriched_results = await self.enrichment_engine.enrich_knowledge_results(
            rag_results, query_analysis, context
        )
        
        # Step 4: Format for agent consumption
        agent_knowledge_context = self.format_for_agents(
            enriched_results, agent_requirements
        )
        
        # Step 5: Track performance metrics
        self.track_rag_performance(query_analysis, rag_results, enriched_results)
        
        return agent_knowledge_context
    
    def format_for_agents(self, enriched_results, agent_requirements):
        """Format enriched results for agent consumption"""
        
        formatted_context = {
            "primary_knowledge": [],
            "code_examples": [],
            "best_practices": [],
            "project_context": [],
            "confidence_scores": {}
        }
        
        for result in enriched_results:
            # Categorize result based on content type and agent needs
            if "code" in result.get("content", "").lower():
                formatted_context["code_examples"].append(
                    self.extract_code_context(result)
                )
            elif "best practice" in result.get("content", "").lower():
                formatted_context["best_practices"].append(
                    self.extract_practices_context(result)
                )
            elif result.get("source_type") == "project":
                formatted_context["project_context"].append(
                    self.extract_project_context(result)
                )
            else:
                formatted_context["primary_knowledge"].append(
                    self.extract_primary_context(result)
                )
            
            # Track confidence scores
            formatted_context["confidence_scores"][result.get("id", "")] = {
                "relevance": result.get("similarity_score", 0),
                "enrichment": result.get("enrichment_score", 0),
                "overall": (result.get("similarity_score", 0) + 
                          result.get("enrichment_score", 0)) / 2
            }
        
        return formatted_context
```

This RAG Integration Layer provides sophisticated knowledge retrieval and enrichment capabilities, enabling the Master Agent to enhance agent performance with contextually relevant information from Archon's knowledge base and multiple fallback sources.