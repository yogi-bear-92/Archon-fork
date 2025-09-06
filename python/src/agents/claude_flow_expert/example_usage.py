"""
Example usage of the Claude Flow Expert Agent with PydanticAI framework.

This demonstrates how to use the Claude Flow Expert Agent for various types of queries
with RAG enhancement, agent coordination, and fallback strategies.
"""

import asyncio
import json
import logging
from typing import Any, Dict

from .claude_flow_expert_agent import (
    ClaudeFlowExpertAgent,
    ClaudeFlowExpertConfig,
    ClaudeFlowExpertDependencies,
    QueryRequest,
    QueryType,
    ProcessingStrategy
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClaudeFlowExpertExample:
    """Example usage class for the Claude Flow Expert Agent."""
    
    def __init__(self):
        """Initialize example with custom configuration."""
        # Configure the claude flow expert agent
        config = ClaudeFlowExpertConfig(
            model="openai:gpt-4o",
            max_retries=3,
            timeout=120,
            rag_enabled=True,
            max_coordinated_agents=5,
            enable_metrics=True,
            circuit_breaker_enabled=True
        )
        
        self.claude_flow_expert = ClaudeFlowExpertAgent(config=config)
        logger.info("Claude Flow Expert Agent example initialized")
    
    async def example_coding_query(self) -> Dict[str, Any]:
        """Example: Processing a coding-related query with RAG enhancement."""
        print("\n=== Coding Query Example ===")
        
        # Create a coding request
        request = QueryRequest(
            query="How do I implement a RESTful API with FastAPI and PostgreSQL? Include authentication.",
            query_type=QueryType.CODING,
            context={"framework": "FastAPI", "database": "PostgreSQL", "features": ["authentication"]},
            require_rag=True,
            max_agents=2
        )
        
        # Set up dependencies
        deps = ClaudeFlowExpertDependencies(
            user_id="example_user",
            request_id="coding_001",
            query_type=QueryType.CODING,
            processing_strategy=ProcessingStrategy.RAG_ENHANCED,
            coordinate_agents=True
        )
        
        try:
            # Process the query
            result = await self.claude_flow_expert.process_query(request, deps)
            
            print(f"Query Type: {result.query_type}")
            print(f"Processing Strategy: {result.processing_strategy}")
            print(f"Success: {result.success}")
            print(f"Processing Time: {result.processing_time:.2f}s")
            print(f"Agents Used: {result.agents_used}")
            print(f"RAG Sources: {result.rag_sources_used}")
            print(f"Fallback Used: {result.fallback_used}")
            print(f"Message: {result.message}")
            
            return result.data or {}
            
        except Exception as e:
            logger.error(f"Coding query failed: {e}")
            return {"error": str(e)}
    
    async def example_research_query(self) -> Dict[str, Any]:
        """Example: Research query with multi-agent coordination."""
        print("\n=== Research Query Example ===")
        
        request = QueryRequest(
            query="What are the latest trends in microservices architecture and container orchestration?",
            query_type=QueryType.RESEARCH,
            context={"topics": ["microservices", "kubernetes", "docker", "architecture"]},
            preferred_agents=["researcher", "analyst", "system-architect"],
            max_agents=3
        )
        
        deps = ClaudeFlowExpertDependencies(
            user_id="example_user",
            request_id="research_001",
            query_type=QueryType.RESEARCH,
            processing_strategy=ProcessingStrategy.MULTI_AGENT,
            coordinate_agents=True
        )
        
        try:
            result = await self.claude_flow_expert.process_query(request, deps)
            
            print(f"Query processed with {len(result.agents_used)} agents")
            print(f"Coordination metrics: {result.coordination_metrics}")
            print(f"Success: {result.success}")
            
            return result.data or {}
            
        except Exception as e:
            logger.error(f"Research query failed: {e}")
            return {"error": str(e)}
    
    async def example_agent_routing(self) -> Dict[str, Any]:
        """Example: Intelligent agent routing based on query type."""
        print("\n=== Agent Routing Example ===")
        
        queries = [
            ("Fix this Python bug in my authentication module", QueryType.CODING),
            ("Analyze the performance metrics of our API", QueryType.ANALYSIS),
            ("Research best practices for database indexing", QueryType.RESEARCH),
            ("Plan the migration from monolith to microservices", QueryType.COORDINATION)
        ]
        
        routing_results = []
        
        for query, query_type in queries:
            try:
                # Route to appropriate agents
                recommended_agents = await self.claude_flow_expert.route_to_agent(
                    query=query,
                    query_type=query_type,
                    preferred_agents=None
                )
                
                routing_results.append({
                    "query": query,
                    "query_type": query_type.value,
                    "recommended_agents": recommended_agents
                })
                
                print(f"Query: {query}")
                print(f"Type: {query_type.value}")
                print(f"Recommended Agents: {recommended_agents}")
                print("---")
                
            except Exception as e:
                logger.error(f"Routing failed for query '{query}': {e}")
        
        return {"routing_results": routing_results}
    
    async def example_multi_agent_coordination(self) -> Dict[str, Any]:
        """Example: Multi-agent coordination for complex task."""
        print("\n=== Multi-Agent Coordination Example ===")
        
        objective = """
        Build a complete user authentication system with the following requirements:
        1. JWT-based authentication with refresh tokens
        2. Password hashing with bcrypt
        3. Rate limiting for login attempts
        4. User registration with email verification
        5. Comprehensive test suite
        6. API documentation
        """
        
        agent_types = ["backend-dev", "tester", "reviewer", "api-docs"]
        
        try:
            result = await self.claude_flow_expert.coordinate_multi_agent(
                objective=objective,
                agent_types=agent_types,
                max_agents=4
            )
            
            print(f"Coordination Status: {result.get('status')}")
            print(f"Agents Coordinated: {result.get('agents_coordinated')}")
            print(f"Coordination Time: {result.get('coordination_time', 0):.2f}s")
            
            if result.get("metrics"):
                print(f"Metrics: {json.dumps(result['metrics'], indent=2)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-agent coordination failed: {e}")
            return {"error": str(e)}
    
    async def example_fallback_strategies(self) -> Dict[str, Any]:
        """Example: Fallback strategies when primary services fail."""
        print("\n=== Fallback Strategies Example ===")
        
        # Test different fallback scenarios
        fallback_results = []
        
        # 1. Wiki search fallback
        try:
            wiki_result = await self.claude_flow_expert.fallback_to_wiki(
                "What is Docker containerization?"
            )
            fallback_results.append({
                "type": "wiki_search",
                "result": wiki_result
            })
            print("Wiki fallback completed")
            
        except Exception as e:
            logger.error(f"Wiki fallback failed: {e}")
        
        # 2. Single agent fallback
        try:
            single_agent_result = await self.claude_flow_expert.fallback_manager.single_agent_fallback(
                objective="Create a simple Hello World API",
                preferred_agent="coder"
            )
            fallback_results.append({
                "type": "single_agent",
                "result": single_agent_result
            })
            print("Single agent fallback completed")
            
        except Exception as e:
            logger.error(f"Single agent fallback failed: {e}")
        
        return {"fallback_results": fallback_results}
    
    async def example_performance_metrics(self) -> Dict[str, Any]:
        """Example: Getting performance metrics and system status."""
        print("\n=== Performance Metrics Example ===")
        
        try:
            # Get current performance metrics
            metrics = await self.claude_flow_expert.get_performance_metrics()
            
            print("=== Claude Flow Expert Agent Metrics ===")
            print(f"Queries Processed: {metrics.get('queries_processed', 0)}")
            print(f"Success Rate: {metrics.get('successful_queries', 0)}/{metrics.get('queries_processed', 0)}")
            print(f"RAG Queries: {metrics.get('rag_queries', 0)}")
            print(f"Multi-Agent Coordinations: {metrics.get('multi_agent_coordinations', 0)}")
            print(f"Fallbacks Used: {metrics.get('fallbacks_used', 0)}")
            print(f"Average Processing Time: {metrics.get('average_processing_time', 0):.2f}s")
            
            # Get capability matrix stats
            capability_stats = self.claude_flow_expert.capability_matrix.export_capabilities()
            print(f"\nTotal Agent Capabilities: {capability_stats.get('total_agents', 0)}")
            
            # Get fallback stats
            fallback_stats = self.claude_flow_expert.fallback_manager.get_fallback_stats()
            print(f"Fallback Statistics: {json.dumps(fallback_stats, indent=2)}")
            
            return {
                "claude_flow_expert_metrics": metrics,
                "capability_stats": capability_stats,
                "fallback_stats": fallback_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            return {"error": str(e)}
    
    async def example_capability_management(self) -> Dict[str, Any]:
        """Example: Dynamic capability management and updates."""
        print("\n=== Capability Management Example ===")
        
        try:
            # Get current capabilities
            capabilities = self.claude_flow_expert.capability_matrix.get_capabilities()
            print(f"Total capabilities loaded: {len(capabilities)}")
            
            # Get capabilities for specific query type
            coding_capabilities = self.claude_flow_expert.capability_matrix.get_capabilities_for_query_type(
                QueryType.CODING, max_results=5
            )
            
            print("Top 5 coding capabilities:")
            for cap in coding_capabilities:
                print(f"- {cap['agent_type']}: {cap['relevance_score']:.3f}")
            
            # Update capability performance
            update_success = self.claude_flow_expert.capability_matrix.update_performance_metrics(
                agent_type="coder",
                success=True,
                response_time=25.5
            )
            
            print(f"Capability update success: {update_success}")
            
            # Get coordination-compatible agents
            coord_agents = self.claude_flow_expert.capability_matrix.get_coordination_compatible_agents()
            print(f"Coordination-compatible agents: {len(coord_agents)}")
            
            return {
                "total_capabilities": len(capabilities),
                "coding_capabilities": coding_capabilities,
                "coordination_agents": len(coord_agents)
            }
            
        except Exception as e:
            logger.error(f"Capability management failed: {e}")
            return {"error": str(e)}
    
    async def run_all_examples(self) -> Dict[str, Any]:
        """Run all examples in sequence."""
        print("🚀 Running Claude Flow Expert Agent Examples")
        print("=" * 50)
        
        results = {}
        
        examples = [
            ("coding_query", self.example_coding_query),
            ("research_query", self.example_research_query),
            ("agent_routing", self.example_agent_routing),
            ("multi_agent_coordination", self.example_multi_agent_coordination),
            ("fallback_strategies", self.example_fallback_strategies),
            ("performance_metrics", self.example_performance_metrics),
            ("capability_management", self.example_capability_management)
        ]
        
        for example_name, example_func in examples:
            try:
                print(f"\n🔄 Running {example_name}...")
                result = await example_func()
                results[example_name] = result
                print(f"✅ {example_name} completed")
                
            except Exception as e:
                logger.error(f"Example {example_name} failed: {e}")
                results[example_name] = {"error": str(e)}
                print(f"❌ {example_name} failed")
        
        print("\n🏁 All examples completed!")
        return results


async def main():
    """Main function to run examples."""
    try:
        example = ClaudeFlowExpertExample()
        results = await example.run_all_examples()
        
        print("\n" + "=" * 50)
        print("📊 SUMMARY")
        print("=" * 50)
        
        for example_name, result in results.items():
            status = "✅ SUCCESS" if "error" not in result else "❌ ERROR"
            print(f"{example_name}: {status}")
        
        # Save results to file for inspection
        with open("claude_flow_expert_example_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n📁 Results saved to claude_flow_expert_example_results.json")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")


if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())