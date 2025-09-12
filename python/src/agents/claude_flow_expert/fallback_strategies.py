"""
Fallback Strategies for resilient information retrieval and task execution.

This module implements various fallback mechanisms when primary services 
(RAG, agent coordination, etc.) fail or are unavailable.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class FallbackType:
    """Types of fallback strategies available."""
    
    WIKI_SEARCH = "wiki_search"
    LOCAL_KNOWLEDGE = "local_knowledge"
    SINGLE_AGENT = "single_agent"
    CACHED_RESPONSE = "cached_response"
    BASIC_LLM = "basic_llm"
    TEMPLATE_RESPONSE = "template_response"


class FallbackManager:
    """
    Manager for fallback strategies when primary services fail.
    
    Provides multiple layers of fallback mechanisms to ensure the system
    remains functional even when specific services are unavailable.
    """
    
    def __init__(self):
        """Initialize the fallback manager."""
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.fallback_stats = {
            "wiki_searches": 0,
            "single_agent_fallbacks": 0,
            "cache_hits": 0,
            "template_responses": 0
        }
        
        # Local knowledge base for common queries
        self.local_knowledge = self._initialize_local_knowledge()
        
        logger.info("Fallback manager initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()
    
    async def wiki_search_fallback(
        self, 
        query: str, 
        max_results: int = 3
    ) -> Dict[str, Any]:
        """
        Wikipedia search fallback for general information retrieval.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Search results from Wikipedia
        """
        try:
            logger.info(f"Executing Wikipedia search fallback for: {query}")
            self.fallback_stats["wiki_searches"] += 1
            
            # Check cache first
            cache_key = f"wiki_{query}_{max_results}"
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                if self._is_cache_valid(cached_result):
                    self.fallback_stats["cache_hits"] += 1
                    return cached_result["data"]
            
            # Clean query for Wikipedia search
            cleaned_query = self._clean_query_for_search(query)
            
            # Search Wikipedia using their API
            search_url = "https://en.wikipedia.org/api/rest_v1/page/search"
            search_params = {
                "q": cleaned_query,
                "limit": max_results
            }
            
            search_response = await self.http_client.get(search_url, params=search_params)
            search_response.raise_for_status()
            search_data = search_response.json()
            
            results = []
            
            # Get details for each search result
            for page in search_data.get("pages", [])[:max_results]:
                page_title = page.get("title", "")
                page_key = page.get("key", "")
                
                if page_key:
                    # Get page summary
                    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{page_key}"
                    
                    try:
                        summary_response = await self.http_client.get(summary_url)
                        summary_response.raise_for_status()
                        summary_data = summary_response.json()
                        
                        result = {
                            "title": page_title,
                            "summary": summary_data.get("extract", ""),
                            "url": f"https://en.wikipedia.org/wiki/{page_key}",
                            "score": page.get("score", 0)
                        }
                        results.append(result)
                        
                    except Exception as e:
                        logger.warning(f"Failed to get summary for {page_title}: {e}")
                        continue
                    
                    # Add delay to be respectful to Wikipedia API
                    await asyncio.sleep(0.1)
            
            fallback_result = {
                "status": "success",
                "fallback_type": FallbackType.WIKI_SEARCH,
                "query": query,
                "results": results,
                "total_results": len(results),
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            self.cache[cache_key] = {
                "data": fallback_result,
                "timestamp": datetime.now().timestamp(),
                "ttl": 3600  # 1 hour
            }
            
            return fallback_result
            
        except Exception as e:
            logger.error(f"Wikipedia search fallback failed: {e}")
            return await self._local_knowledge_fallback(query)
    
    async def single_agent_fallback(
        self,
        objective: str,
        preferred_agent: str = "coder",
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Single agent fallback when multi-agent coordination fails.
        
        Args:
            objective: Task objective
            preferred_agent: Preferred agent type
            context: Optional context information
            
        Returns:
            Single agent execution result
        """
        try:
            logger.info(f"Executing single agent fallback with {preferred_agent}")
            self.fallback_stats["single_agent_fallbacks"] += 1
            
            # Create a simplified prompt for single agent execution
            agent_prompt = self._create_single_agent_prompt(
                objective, preferred_agent, context
            )
            
            result = {
                "status": "success",
                "fallback_type": FallbackType.SINGLE_AGENT,
                "agent_used": preferred_agent,
                "objective": objective,
                "response": agent_prompt,
                "context": context,
                "timestamp": datetime.now().isoformat(),
                "note": "Executed as single agent fallback due to coordination failure"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Single agent fallback failed: {e}")
            return await self._template_response_fallback(objective)
    
    async def cached_response_fallback(
        self,
        query: str,
        fallback_type: str = "general"
    ) -> Optional[Dict[str, Any]]:
        """
        Cached response fallback for previously processed queries.
        
        Args:
            query: The query to look up
            fallback_type: Type of fallback to cache under
            
        Returns:
            Cached response if available
        """
        try:
            cache_key = f"{fallback_type}_{query}"
            
            if cache_key in self.cache:
                cached_result = self.cache[cache_key]
                
                if self._is_cache_valid(cached_result):
                    self.fallback_stats["cache_hits"] += 1
                    logger.info(f"Retrieved cached response for: {query}")
                    
                    # Update timestamp to show it's from cache
                    result = cached_result["data"].copy()
                    result["from_cache"] = True
                    result["original_timestamp"] = result.get("timestamp")
                    result["timestamp"] = datetime.now().isoformat()
                    
                    return result
            
            return None
            
        except Exception as e:
            logger.error(f"Cache lookup failed: {e}")
            return None
    
    async def basic_llm_fallback(
        self,
        query: str,
        model: str = "gpt-3.5-turbo"
    ) -> Dict[str, Any]:
        """
        Basic LLM fallback without specialized tools or coordination.
        
        Args:
            query: User query
            model: LLM model to use
            
        Returns:
            Basic LLM response
        """
        try:
            logger.info(f"Executing basic LLM fallback for: {query}")
            
            # This would integrate with a basic LLM client
            # For now, return a structured response indicating the fallback
            
            result = {
                "status": "success",
                "fallback_type": FallbackType.BASIC_LLM,
                "query": query,
                "response": f"Basic response for: {query} (using {model} fallback)",
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "note": "This is a basic LLM fallback response"
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Basic LLM fallback failed: {e}")
            return await self._template_response_fallback(query)
    
    async def _local_knowledge_fallback(self, query: str) -> Dict[str, Any]:
        """
        Local knowledge base fallback for common programming questions.
        
        Args:
            query: User query
            
        Returns:
            Local knowledge response
        """
        try:
            logger.info(f"Executing local knowledge fallback for: {query}")
            
            query_lower = query.lower()
            matched_topics = []
            
            # Search through local knowledge base
            for topic, info in self.local_knowledge.items():
                if any(keyword in query_lower for keyword in info["keywords"]):
                    matched_topics.append({
                        "topic": topic,
                        "content": info["content"],
                        "relevance": self._calculate_relevance(query_lower, info["keywords"])
                    })
            
            # Sort by relevance
            matched_topics.sort(key=lambda x: x["relevance"], reverse=True)
            
            result = {
                "status": "success",
                "fallback_type": FallbackType.LOCAL_KNOWLEDGE,
                "query": query,
                "matches": matched_topics[:3],  # Top 3 matches
                "total_matches": len(matched_topics),
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Local knowledge fallback failed: {e}")
            return await self._template_response_fallback(query)
    
    async def _template_response_fallback(self, query: str) -> Dict[str, Any]:
        """
        Template response fallback as last resort.
        
        Args:
            query: User query
            
        Returns:
            Template response
        """
        self.fallback_stats["template_responses"] += 1
        
        templates = {
            "coding": "I can help with coding tasks. Please provide more specific details about what you'd like to implement.",
            "research": "I can help with research tasks. Let me know what topic you'd like to explore.",
            "analysis": "I can help with analysis tasks. Please share the data or topic you'd like analyzed.",
            "general": "I'm here to help. Could you please provide more details about what you need assistance with?"
        }
        
        # Determine template based on query
        query_lower = query.lower()
        template_key = "general"
        
        if any(word in query_lower for word in ["code", "program", "function", "class", "bug"]):
            template_key = "coding"
        elif any(word in query_lower for word in ["research", "find", "search", "learn"]):
            template_key = "research"
        elif any(word in query_lower for word in ["analyze", "review", "check", "examine"]):
            template_key = "analysis"
        
        return {
            "status": "success",
            "fallback_type": FallbackType.TEMPLATE_RESPONSE,
            "query": query,
            "response": templates[template_key],
            "template_used": template_key,
            "timestamp": datetime.now().isoformat(),
            "note": "This is a template response fallback - please provide more specific details for better assistance"
        }
    
    def _initialize_local_knowledge(self) -> Dict[str, Dict[str, Any]]:
        """Initialize local knowledge base for common topics."""
        return {
            "python_basics": {
                "keywords": ["python", "basics", "syntax", "variables", "functions"],
                "content": {
                    "overview": "Python is a high-level programming language with simple syntax",
                    "key_concepts": ["Variables", "Functions", "Classes", "Modules"],
                    "example": "def hello_world():\\n    print('Hello, World!')"
                }
            },
            "web_development": {
                "keywords": ["web", "html", "css", "javascript", "frontend", "backend"],
                "content": {
                    "overview": "Web development involves creating websites and web applications",
                    "technologies": ["HTML", "CSS", "JavaScript", "React", "Node.js"],
                    "patterns": ["MVC", "SPA", "REST API", "GraphQL"]
                }
            },
            "database_design": {
                "keywords": ["database", "sql", "schema", "tables", "relationships"],
                "content": {
                    "overview": "Database design involves structuring data efficiently",
                    "concepts": ["Normalization", "Relationships", "Indexing", "Transactions"],
                    "best_practices": ["Use appropriate data types", "Create proper indexes", "Normalize when needed"]
                }
            },
            "api_development": {
                "keywords": ["api", "rest", "graphql", "endpoints", "authentication"],
                "content": {
                    "overview": "API development involves creating interfaces for applications to communicate",
                    "types": ["REST", "GraphQL", "RPC", "WebSocket"],
                    "best_practices": ["Use proper HTTP methods", "Implement authentication", "Version your APIs"]
                }
            },
            "testing": {
                "keywords": ["testing", "unit", "integration", "test", "coverage"],
                "content": {
                    "overview": "Testing ensures code quality and reliability",
                    "types": ["Unit Testing", "Integration Testing", "E2E Testing", "Performance Testing"],
                    "frameworks": ["pytest", "Jest", "Selenium", "JUnit"]
                }
            },
            "deployment": {
                "keywords": ["deployment", "docker", "kubernetes", "aws", "cloud"],
                "content": {
                    "overview": "Deployment involves making applications available to users",
                    "strategies": ["Blue-Green", "Rolling", "Canary", "A/B Testing"],
                    "tools": ["Docker", "Kubernetes", "AWS", "Azure", "Google Cloud"]
                }
            }
        }
    
    def _clean_query_for_search(self, query: str) -> str:
        """Clean query for external search APIs."""
        # Remove special characters and extra whitespace
        cleaned = re.sub(r'[^\w\s]', ' ', query)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Limit length for API compatibility
        if len(cleaned) > 100:
            cleaned = cleaned[:100].rsplit(' ', 1)[0]
        
        return cleaned
    
    def _calculate_relevance(self, query: str, keywords: List[str]) -> float:
        """Calculate relevance score between query and keywords."""
        query_words = set(query.lower().split())
        keyword_words = set(word.lower() for word in keywords)
        
        # Calculate Jaccard similarity
        intersection = len(query_words & keyword_words)
        union = len(query_words | keyword_words)
        
        return intersection / union if union > 0 else 0.0
    
    def _create_single_agent_prompt(
        self,
        objective: str,
        agent_type: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a prompt for single agent execution."""
        agent_descriptions = {
            "coder": "You are a skilled software developer who writes clean, efficient code.",
            "researcher": "You are a thorough researcher who finds and analyzes information.",
            "analyst": "You are a data analyst who examines and interprets information.",
            "reviewer": "You are a code reviewer who ensures quality and best practices.",
            "tester": "You are a testing specialist who creates comprehensive test cases.",
            "planner": "You are a project planner who organizes tasks and resources.",
            "architect": "You are a system architect who designs scalable solutions."
        }
        
        description = agent_descriptions.get(agent_type, "You are a helpful AI assistant.")
        
        prompt = f"{description}\n\nObjective: {objective}\n"
        
        if context:
            prompt += f"\nContext: {json.dumps(context, indent=2)}\n"
        
        prompt += "\nPlease provide a detailed response addressing the objective."
        
        return prompt
    
    def _is_cache_valid(self, cached_item: Dict[str, Any]) -> bool:
        """Check if cached item is still valid."""
        try:
            cache_time = cached_item.get("timestamp", 0)
            ttl = cached_item.get("ttl", 3600)  # Default 1 hour
            current_time = datetime.now().timestamp()
            
            return (current_time - cache_time) < ttl
            
        except Exception:
            return False
    
    def cache_response(
        self,
        key: str,
        response: Dict[str, Any],
        ttl: int = 3600
    ):
        """Cache a response for future fallback use."""
        try:
            self.cache[key] = {
                "data": response,
                "timestamp": datetime.now().timestamp(),
                "ttl": ttl
            }
            
            # Cleanup old cache entries periodically
            if len(self.cache) > 100:
                self._cleanup_cache()
                
        except Exception as e:
            logger.error(f"Failed to cache response: {e}")
    
    def _cleanup_cache(self):
        """Remove expired cache entries."""
        try:
            current_time = datetime.now().timestamp()
            expired_keys = []
            
            for key, cached_item in self.cache.items():
                if not self._is_cache_valid(cached_item):
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def get_fallback_stats(self) -> Dict[str, Any]:
        """Get fallback usage statistics."""
        return {
            **self.fallback_stats,
            "cache_size": len(self.cache),
            "local_knowledge_topics": len(self.local_knowledge),
            "timestamp": datetime.now().isoformat()
        }
    
    def reset_stats(self):
        """Reset fallback statistics."""
        for key in self.fallback_stats:
            self.fallback_stats[key] = 0
        logger.info("Fallback statistics reset")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of fallback mechanisms."""
        try:
            # Test Wikipedia API connectivity
            wiki_health = await self._test_wiki_connectivity()
            
            # Check local knowledge availability
            local_knowledge_health = len(self.local_knowledge) > 0
            
            # Check cache functionality
            cache_health = True
            try:
                test_key = "health_check_test"
                self.cache_response(test_key, {"test": True}, ttl=1)
                cache_health = test_key in self.cache
                if cache_health:
                    del self.cache[test_key]
            except Exception:
                cache_health = False
            
            return {
                "status": "healthy" if all([wiki_health, local_knowledge_health, cache_health]) else "degraded",
                "components": {
                    "wikipedia_api": "healthy" if wiki_health else "unhealthy",
                    "local_knowledge": "healthy" if local_knowledge_health else "unhealthy", 
                    "cache": "healthy" if cache_health else "unhealthy"
                },
                "fallback_stats": self.get_fallback_stats(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _test_wiki_connectivity(self) -> bool:
        """Test Wikipedia API connectivity."""
        try:
            test_url = "https://en.wikipedia.org/api/rest_v1/page/search"
            test_params = {"q": "test", "limit": 1}
            
            response = await self.http_client.get(test_url, params=test_params, timeout=10)
            return response.status_code == 200
            
        except Exception as e:
            logger.warning(f"Wikipedia connectivity test failed: {e}")
            return False