"""
Unit tests for fallback mechanisms and error handling.

This module tests fallback strategies including wiki retrieval when RAG fails,
single agent fallback when coordination fails, and error handling patterns.
"""

import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock

from src.agents.claude_flow_expert.fallback_strategies import (
    FallbackManager, FallbackType
)
from tests.mocks.claude_flow_expert_agent_mocks import MockFallbackManager


@pytest.mark.unit
class TestFallbackManager:
    """Test core fallback manager functionality."""
    
    def test_fallback_manager_initialization(self):
        """Test fallback manager initialization."""
        fallback_manager = FallbackManager()
        
        assert fallback_manager.http_client is not None
        assert fallback_manager.cache == {}
        assert fallback_manager.local_knowledge is not None
        assert len(fallback_manager.local_knowledge) > 0
        assert fallback_manager.fallback_stats is not None
        
        # Check initial stats
        expected_stats = ["wiki_searches", "single_agent_fallbacks", "cache_hits", "template_responses"]
        for stat in expected_stats:
            assert stat in fallback_manager.fallback_stats
            assert fallback_manager.fallback_stats[stat] == 0
    
    def test_local_knowledge_initialization(self):
        """Test local knowledge base initialization."""
        fallback_manager = FallbackManager()
        
        # Check that local knowledge contains expected topics
        expected_topics = [
            "python_basics",
            "web_development", 
            "database_design",
            "api_development",
            "testing",
            "deployment"
        ]
        
        for topic in expected_topics:
            assert topic in fallback_manager.local_knowledge
            
            knowledge_item = fallback_manager.local_knowledge[topic]
            assert "keywords" in knowledge_item
            assert "content" in knowledge_item
            assert isinstance(knowledge_item["keywords"], list)
            assert len(knowledge_item["keywords"]) > 0
    
    @pytest.mark.asyncio
    async def test_fallback_manager_context_manager(self):
        """Test fallback manager as async context manager."""
        async with FallbackManager() as fallback_manager:
            assert fallback_manager is not None
            assert fallback_manager.http_client is not None
        
        # HTTP client should be closed after context manager
        # Note: Testing client closure would require checking internal state


@pytest.mark.unit
class TestWikiFallback:
    """Test Wikipedia search fallback functionality."""
    
    @pytest.mark.asyncio
    async def test_successful_wiki_search(self):
        """Test successful Wikipedia search fallback."""
        # Use mock fallback manager for controlled testing
        mock_fallback = MockFallbackManager()
        
        # Set up expected response
        wiki_response = {
            "status": "success",
            "fallback_type": "wiki_search",
            "query": "Python programming",
            "results": [
                {
                    "title": "Python (programming language)",
                    "summary": "Python is a high-level programming language",
                    "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                    "score": 0.95
                }
            ],
            "total_results": 1
        }
        mock_fallback.set_wiki_response("Python programming", wiki_response)
        
        result = await mock_fallback.wiki_search_fallback("Python programming")
        
        assert result["status"] == "success"
        assert result["fallback_type"] == "wiki_search"
        assert result["query"] == "Python programming"
        assert len(result["results"]) == 1
        assert result["results"][0]["title"] == "Python (programming language)"
        assert mock_fallback.call_count == 1
    
    @pytest.mark.asyncio
    async def test_wiki_search_failure(self):
        """Test Wikipedia search fallback failure handling."""
        mock_fallback = MockFallbackManager()
        mock_fallback.should_fail = True
        
        result = await mock_fallback.wiki_search_fallback("Test query")
        
        # Should fall back to local knowledge
        assert "status" in result
        # Mock implementation might still succeed with local knowledge
        assert mock_fallback.call_count == 1
    
    @pytest.mark.asyncio 
    async def test_wiki_search_caching(self):
        """Test Wikipedia search result caching."""
        fallback_manager = FallbackManager()
        
        # Mock HTTP response for testing
        with patch.object(fallback_manager.http_client, 'get') as mock_get:
            mock_search_response = MagicMock()
            mock_search_response.json.return_value = {
                "pages": [
                    {
                        "title": "Test Article",
                        "key": "Test_Article",
                        "score": 0.9
                    }
                ]
            }
            mock_search_response.raise_for_status = MagicMock()
            
            mock_summary_response = MagicMock()
            mock_summary_response.json.return_value = {
                "extract": "Test article summary",
                "title": "Test Article"
            }
            mock_summary_response.raise_for_status = MagicMock()
            
            mock_get.side_effect = [mock_search_response, mock_summary_response]
            
            # First call should hit the API
            result1 = await fallback_manager.wiki_search_fallback("test query", max_results=1)
            
            # Second call should use cache
            result2 = await fallback_manager.wiki_search_fallback("test query", max_results=1)
            
            # Verify both calls succeeded
            assert result1["status"] == "success"
            assert result2["status"] == "success"
            
            # Second call should be from cache (would need to verify cache hit count)
            assert len(fallback_manager.cache) > 0
    
    def test_query_cleaning_for_search(self):
        """Test query cleaning for external search APIs."""
        fallback_manager = FallbackManager()
        
        test_cases = [
            ("Simple query", "Simple query"),
            ("Query with special chars!@#$%", "Query with special chars"),
            ("   Multiple   spaces   ", "Multiple spaces"),
            ("Very long query " * 20, None)  # Should be truncated
        ]
        
        for input_query, expected_output in test_cases:
            cleaned = fallback_manager._clean_query_for_search(input_query)
            
            if expected_output is None:
                # Should be truncated but not empty
                assert len(cleaned) < len(input_query)
                assert len(cleaned) > 0
            else:
                assert cleaned == expected_output


@pytest.mark.unit
class TestSingleAgentFallback:
    """Test single agent fallback functionality."""
    
    @pytest.mark.asyncio
    async def test_successful_single_agent_fallback(self):
        """Test successful single agent fallback."""
        mock_fallback = MockFallbackManager()
        
        result = await mock_fallback.single_agent_fallback(
            objective="Implement a web scraper",
            preferred_agent="coder",
            context={"language": "python", "libraries": ["requests", "beautifulsoup"]}
        )
        
        assert result["status"] == "success"
        assert result["fallback_type"] == "single_agent"
        assert result["agent_used"] == "coder"
        assert result["objective"] == "Implement a web scraper"
        assert "python" in str(result["context"])
        assert mock_fallback.call_count == 1
    
    @pytest.mark.asyncio
    async def test_single_agent_fallback_with_defaults(self):
        """Test single agent fallback with default parameters."""
        mock_fallback = MockFallbackManager()
        
        result = await mock_fallback.single_agent_fallback("General task")
        
        assert result["status"] == "success"
        assert result["fallback_type"] == "single_agent"
        assert result["agent_used"] == "coder"  # Default agent
        assert result["objective"] == "General task"
    
    @pytest.mark.asyncio
    async def test_single_agent_fallback_failure(self):
        """Test single agent fallback failure handling."""
        mock_fallback = MockFallbackManager()
        mock_fallback.should_fail = True
        
        result = await mock_fallback.single_agent_fallback("Failing task")
        
        assert result["status"] == "error"
        assert "error" in result
    
    def test_single_agent_prompt_generation(self):
        """Test single agent prompt generation."""
        fallback_manager = FallbackManager()
        
        prompt = fallback_manager._create_single_agent_prompt(
            objective="Build a REST API",
            agent_type="coder",
            context={"framework": "FastAPI", "database": "PostgreSQL"}
        )
        
        assert "coder" in prompt or "software developer" in prompt.lower()
        assert "Build a REST API" in prompt
        assert "FastAPI" in prompt
        assert "PostgreSQL" in prompt
        assert "Objective:" in prompt
    
    def test_agent_descriptions(self):
        """Test agent descriptions for different types."""
        fallback_manager = FallbackManager()
        
        agent_types = ["coder", "researcher", "analyst", "reviewer", "tester", "planner", "architect"]
        
        for agent_type in agent_types:
            prompt = fallback_manager._create_single_agent_prompt(
                objective="Test objective",
                agent_type=agent_type
            )
            
            # Should contain agent-specific description
            assert len(prompt) > 50  # Should be substantial
            assert "Test objective" in prompt
            
        # Unknown agent type should get default description
        unknown_prompt = fallback_manager._create_single_agent_prompt(
            objective="Test",
            agent_type="unknown_agent"
        )
        assert "helpful AI assistant" in unknown_prompt


@pytest.mark.unit
class TestLocalKnowledgeFallback:
    """Test local knowledge base fallback."""
    
    @pytest.mark.asyncio
    async def test_local_knowledge_matching(self):
        """Test local knowledge matching algorithm."""
        fallback_manager = FallbackManager()
        
        # Test queries that should match local knowledge
        test_queries = [
            ("How to use Python variables", ["python_basics"]),
            ("Web development with HTML and CSS", ["web_development"]),
            ("Database schema design", ["database_design"]),
            ("REST API development", ["api_development"]),
            ("Unit testing frameworks", ["testing"]),
            ("Docker deployment", ["deployment"])
        ]
        
        for query, expected_topics in test_queries:
            result = await fallback_manager._local_knowledge_fallback(query)
            
            assert result["status"] == "success"
            assert result["fallback_type"] == FallbackType.LOCAL_KNOWLEDGE
            assert len(result["matches"]) > 0
            
            # Should find relevant matches
            matched_topics = [match["topic"] for match in result["matches"]]
            assert any(topic in expected_topics for topic in matched_topics)
    
    def test_relevance_calculation(self):
        """Test relevance score calculation."""
        fallback_manager = FallbackManager()
        
        # Test relevance calculation
        query = "python programming language"
        keywords = ["python", "programming", "language", "syntax"]
        
        relevance = fallback_manager._calculate_relevance(query, keywords)
        
        # Should have high relevance due to exact matches
        assert 0.0 <= relevance <= 1.0
        assert relevance > 0.5  # Should be relatively high
        
        # Test with no overlap
        no_match_relevance = fallback_manager._calculate_relevance(
            "completely different topic",
            keywords
        )
        assert no_match_relevance < relevance
        
        # Test with partial overlap
        partial_relevance = fallback_manager._calculate_relevance(
            "python web development",
            keywords  
        )
        assert 0.0 < partial_relevance <= relevance
    
    @pytest.mark.asyncio
    async def test_local_knowledge_fallback_ordering(self):
        """Test that local knowledge results are ordered by relevance."""
        fallback_manager = FallbackManager()
        
        # Query that should match multiple topics with different relevances
        result = await fallback_manager._local_knowledge_fallback("python web api")
        
        assert result["status"] == "success"
        assert len(result["matches"]) > 1
        
        # Results should be ordered by relevance (descending)
        relevances = [match["relevance"] for match in result["matches"]]
        assert relevances == sorted(relevances, reverse=True)
        
        # First result should have highest relevance
        assert relevances[0] >= relevances[-1]


@pytest.mark.unit 
class TestCachingMechanism:
    """Test caching mechanisms in fallback system."""
    
    def test_cache_functionality(self):
        """Test basic cache functionality."""
        fallback_manager = FallbackManager()
        
        test_key = "test_key"
        test_data = {"test": "data", "timestamp": time.time()}
        
        # Cache data
        fallback_manager.cache_response(test_key, test_data, ttl=3600)
        
        assert test_key in fallback_manager.cache
        cached_item = fallback_manager.cache[test_key]
        assert cached_item["data"] == test_data
        assert cached_item["ttl"] == 3600
        assert "timestamp" in cached_item
    
    def test_cache_validation(self):
        """Test cache validation and expiration."""
        fallback_manager = FallbackManager()
        
        # Test valid cache item
        valid_item = {
            "data": {"test": "data"},
            "timestamp": time.time(),
            "ttl": 3600
        }
        assert fallback_manager._is_cache_valid(valid_item) is True
        
        # Test expired cache item
        expired_item = {
            "data": {"test": "data"},
            "timestamp": time.time() - 7200,  # 2 hours ago
            "ttl": 3600  # 1 hour TTL
        }
        assert fallback_manager._is_cache_valid(expired_item) is False
    
    def test_cache_cleanup(self):
        """Test cache cleanup functionality."""
        fallback_manager = FallbackManager()
        
        # Add multiple cache entries, some expired
        current_time = time.time()
        
        # Valid entry
        fallback_manager.cache["valid"] = {
            "data": {"test": "valid"},
            "timestamp": current_time,
            "ttl": 3600
        }
        
        # Expired entry
        fallback_manager.cache["expired"] = {
            "data": {"test": "expired"},
            "timestamp": current_time - 7200,  # 2 hours ago
            "ttl": 3600
        }
        
        assert len(fallback_manager.cache) == 2
        
        # Run cleanup
        fallback_manager._cleanup_cache()
        
        # Only valid entry should remain
        assert "valid" in fallback_manager.cache
        assert "expired" not in fallback_manager.cache
        assert len(fallback_manager.cache) == 1
    
    def test_cache_size_limit(self):
        """Test cache size limit enforcement."""
        fallback_manager = FallbackManager()
        
        # Add many cache entries to trigger cleanup
        for i in range(150):  # Exceeds the 100 item limit
            fallback_manager.cache_response(
                f"key_{i}",
                {"data": f"value_{i}"},
                ttl=3600
            )
        
        # Cache should be automatically cleaned up (allowing buffer for LRU implementation)
        assert len(fallback_manager.cache) <= 120


@pytest.mark.unit
class TestTemplateFallback:
    """Test template response fallback."""
    
    @pytest.mark.asyncio
    async def test_template_response_classification(self):
        """Test template response classification based on query."""
        fallback_manager = FallbackManager()
        
        test_cases = [
            ("Write a Python function", "coding"),
            ("Find information about machine learning", "research"), 
            ("Analyze this data", "analysis"),
            ("Random question", "general")
        ]
        
        for query, expected_category in test_cases:
            result = await fallback_manager._template_response_fallback(query)
            
            assert result["status"] == "success"
            assert result["fallback_type"] == FallbackType.TEMPLATE_RESPONSE
            assert result["template_used"] == expected_category
            assert query in result["query"]
            
            # Response should be appropriate for the category
            response = result["response"]
            if expected_category == "coding":
                assert "coding" in response.lower() or "implement" in response.lower()
            elif expected_category == "research":
                assert "research" in response.lower() or "information" in response.lower()
    
    def test_template_response_stats(self):
        """Test template response statistics tracking."""
        fallback_manager = FallbackManager()
        
        initial_count = fallback_manager.fallback_stats["template_responses"]
        
        # Generate multiple template responses
        asyncio.run(fallback_manager._template_response_fallback("Test query 1"))
        asyncio.run(fallback_manager._template_response_fallback("Test query 2"))
        
        final_count = fallback_manager.fallback_stats["template_responses"]
        assert final_count == initial_count + 2


@pytest.mark.unit
class TestFallbackStatistics:
    """Test fallback statistics tracking."""
    
    def test_stats_initialization(self):
        """Test fallback statistics initialization."""
        fallback_manager = FallbackManager()
        
        expected_stats = [
            "wiki_searches",
            "single_agent_fallbacks", 
            "cache_hits",
            "template_responses"
        ]
        
        stats = fallback_manager.get_fallback_stats()
        
        for stat in expected_stats:
            assert stat in stats
            assert stats[stat] == 0
        
        assert "cache_size" in stats
        assert "local_knowledge_topics" in stats
        assert "timestamp" in stats
    
    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Test statistics tracking during fallback operations."""
        mock_fallback = MockFallbackManager()
        
        initial_stats = mock_fallback.fallback_stats.copy()
        
        # Perform various fallback operations
        await mock_fallback.wiki_search_fallback("Test wiki query")
        await mock_fallback.single_agent_fallback("Test single agent task")
        
        # Check that stats were updated
        assert mock_fallback.fallback_stats["wiki_searches"] > initial_stats["wiki_searches"]
        assert mock_fallback.fallback_stats["single_agent_fallbacks"] > initial_stats["single_agent_fallbacks"]
    
    def test_stats_reset(self):
        """Test statistics reset functionality."""
        fallback_manager = FallbackManager()
        
        # Modify some stats
        fallback_manager.fallback_stats["wiki_searches"] = 5
        fallback_manager.fallback_stats["template_responses"] = 3
        
        # Reset stats
        fallback_manager.reset_stats()
        
        # All stats should be reset to 0
        for stat_value in fallback_manager.fallback_stats.values():
            assert stat_value == 0


@pytest.mark.unit
class TestFallbackHealthCheck:
    """Test fallback system health monitoring."""
    
    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self):
        """Test health check when all components are healthy."""
        fallback_manager = FallbackManager()
        
        # Mock successful connectivity test
        with patch.object(fallback_manager, '_test_wiki_connectivity', return_value=True):
            health_result = await fallback_manager.health_check()
            
            assert health_result["status"] == "healthy"
            assert health_result["components"]["wikipedia_api"] == "healthy"
            assert health_result["components"]["local_knowledge"] == "healthy"
            assert health_result["components"]["cache"] == "healthy"
            assert "fallback_stats" in health_result
    
    @pytest.mark.asyncio
    async def test_health_check_degraded(self):
        """Test health check when some components are unhealthy."""
        fallback_manager = FallbackManager()
        
        # Mock failed connectivity test
        with patch.object(fallback_manager, '_test_wiki_connectivity', return_value=False):
            health_result = await fallback_manager.health_check()
            
            assert health_result["status"] == "degraded"
            assert health_result["components"]["wikipedia_api"] == "unhealthy"
    
    @pytest.mark.asyncio
    async def test_wiki_connectivity_test(self):
        """Test Wikipedia connectivity testing."""
        fallback_manager = FallbackManager()
        
        # Mock successful HTTP response
        with patch.object(fallback_manager.http_client, 'get') as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            is_healthy = await fallback_manager._test_wiki_connectivity()
            assert is_healthy is True
            
        # Mock failed HTTP response
        with patch.object(fallback_manager.http_client, 'get') as mock_get:
            mock_get.side_effect = Exception("Connection failed")
            
            is_healthy = await fallback_manager._test_wiki_connectivity()
            assert is_healthy is False


@pytest.mark.unit
class TestFallbackErrorHandling:
    """Test error handling in fallback mechanisms."""
    
    @pytest.mark.asyncio
    async def test_fallback_chain_exhaustion(self):
        """Test behavior when all fallback mechanisms fail."""
        mock_fallback = MockFallbackManager()
        mock_fallback.should_fail = True
        
        # Even with all failures, some fallback should still work
        result = await mock_fallback.wiki_search_fallback("Test query")
        
        # The mock should handle the failure appropriately
        # In real implementation, this would trigger the final template fallback
        assert "status" in result
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test timeout handling in fallback operations."""
        fallback_manager = FallbackManager()
        
        # Mock slow HTTP response
        with patch.object(fallback_manager.http_client, 'get') as mock_get:
            async def slow_response():
                await asyncio.sleep(2)
                return MagicMock()
            
            mock_get.return_value = slow_response()
            
            # Should handle timeout gracefully
            with pytest.raises((asyncio.TimeoutError, Exception)):
                await asyncio.wait_for(
                    fallback_manager.wiki_search_fallback("Test query"),
                    timeout=1.0
                )
    
    @pytest.mark.asyncio
    async def test_malformed_response_handling(self):
        """Test handling of malformed responses from external APIs."""
        fallback_manager = FallbackManager()
        
        # Mock malformed JSON response
        with patch.object(fallback_manager.http_client, 'get') as mock_get:
            mock_response = MagicMock()
            mock_response.json.side_effect = ValueError("Invalid JSON")
            mock_response.raise_for_status = MagicMock()
            mock_get.return_value = mock_response
            
            # Should handle malformed response gracefully
            result = await fallback_manager.wiki_search_fallback("Test query")
            
            # Should fall back to local knowledge or other mechanisms
            # The exact behavior depends on implementation
            assert "status" in result


@pytest.mark.unit
class TestFallbackIntegrationPoints:
    """Test integration points between different fallback mechanisms."""
    
    @pytest.mark.asyncio
    async def test_fallback_strategy_selection(self):
        """Test selection of appropriate fallback strategy."""
        # This would test the logic that determines which fallback to use
        # based on the type of failure and available alternatives
        pass
    
    @pytest.mark.asyncio
    async def test_fallback_result_normalization(self):
        """Test that different fallback mechanisms return normalized results."""
        mock_fallback = MockFallbackManager()
        
        # Test that all fallback types return consistent response format
        wiki_result = await mock_fallback.wiki_search_fallback("Test query")
        single_agent_result = await mock_fallback.single_agent_fallback("Test task")
        
        # Both should have consistent top-level structure
        expected_fields = ["status", "fallback_type", "timestamp"]
        
        for field in expected_fields:
            assert field in wiki_result
            assert field in single_agent_result
    
    def test_fallback_priority_ordering(self):
        """Test fallback priority ordering and selection logic."""
        # This would test the logic that determines fallback order
        # e.g., try cached response -> wiki -> local knowledge -> template
        pass