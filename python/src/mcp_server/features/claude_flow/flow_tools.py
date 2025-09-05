"""
Claude Flow MCP Tools for Archon Integration

Provides MCP tools for Claude Flow orchestration, SPARC workflows,
and swarm coordination directly through Archon's MCP server.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource

from ...utils.http_client import get_http_client
from ...utils.error_handling import MCPErrorFormatter

logger = logging.getLogger(__name__)


def _determine_query_type(query: str) -> str:
    """Determine the type of Claude Flow query for specialized handling."""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["swarm", "topology", "mesh", "hierarchical"]):
        return "swarm_coordination"
    elif any(word in query_lower for word in ["sparc", "specification", "pseudocode", "architecture", "refinement"]):
        return "sparc_methodology"
    elif any(word in query_lower for word in ["agent", "spawn", "type", "capabilities"]):
        return "agent_management"
    elif any(word in query_lower for word in ["setup", "install", "configure", "init"]):
        return "setup_configuration"
    elif any(word in query_lower for word in ["performance", "metrics", "optimization", "speed"]):
        return "performance_optimization"
    elif any(word in query_lower for word in ["memory", "neural", "training", "pattern"]):
        return "neural_features"
    elif any(word in query_lower for word in ["github", "integration", "mcp", "api"]):
        return "integration_tools"
    else:
        return "general_guidance"


def _generate_expert_response(query: str, response_type: str, context_info: List[Dict]) -> str:
    """Generate specialized expert response based on query type and context."""
    
    # Extract key information from context
    relevant_content = "\n".join([ctx["content"] for ctx in context_info[:2]])  # Top 2 results
    
    base_responses = {
        "swarm_coordination": f"""Based on the Claude Flow documentation, here's how to work with swarms:

{relevant_content}

For your specific question: "{query}"

Key recommendations:
- Use mesh topology for collaborative work between equal agents
- Use hierarchical topology for structured coordination with clear roles
- Use adaptive topology to let Claude Flow choose based on your task complexity
- Start with 3-5 agents and scale up based on performance

Example command:
```bash
npx claude-flow swarm init --topology mesh --max-agents 5
npx claude-flow swarm "your objective" --agents "coder,reviewer,tester"
```""",

        "sparc_methodology": f"""The SPARC methodology in Claude Flow provides systematic development workflows:

{relevant_content}

For your question: "{query}"

SPARC phases:
- **S**pecification: Requirements analysis and planning
- **P**seudocode: Algorithm design and logic flow  
- **A**rchitecture: System design and structure
- **R**efinement: Test-driven implementation
- **C**ompletion: Integration and finalization

Example usage:
```bash
npx claude-flow sparc tdd "your feature description"
```""",

        "agent_management": f"""Claude Flow offers 54+ specialized agent types:

{relevant_content}

For your question about: "{query}"

Popular agent combinations:
- **Web Development**: backend-dev, coder, system-architect, tester, reviewer
- **ML Projects**: ml-developer, researcher, tester, system-architect
- **DevOps**: cicd-engineer, system-architect, reviewer, tester

Example spawning:
```bash
npx claude-flow agent spawn ml-developer "Build recommendation system"
```""",

        "setup_configuration": f"""Setting up Claude Flow with Archon integration:

{relevant_content}

For your setup question: "{query}"

Quick setup steps:
1. Install: `npm install -g claude-flow@alpha`
2. Configure: Create `.claude-flow/config/topology.json`
3. Initialize: `npx claude-flow swarm init --topology adaptive`
4. Enable Archon integration in your configuration

The system will handle MCP protocol and coordination automatically.""",

        "performance_optimization": f"""Claude Flow performance insights:

{relevant_content}

Regarding your question: "{query}"

Key performance features:
- **84.8% SWE-Bench solve rate**: Industry-leading task completion
- **32.3% token reduction**: Optimized resource usage
- **2.8-4.4x speed improvement**: Parallel execution benefits

Optimization tips:
- Use parallel execution for independent tasks
- Monitor metrics with `npx claude-flow swarm monitor`
- Enable neural training for repetitive patterns""",

        "neural_features": f"""Claude Flow's neural capabilities:

{relevant_content}

For your neural question: "{query}"

Available cognitive patterns:
- Convergent: Focused problem-solving
- Divergent: Creative exploration
- Lateral: Alternative approaches
- Systems: Holistic thinking
- Critical: Analytical evaluation
- Adaptive: Dynamic adjustment

Neural training automatically improves agent performance over time.""",

        "integration_tools": f"""Claude Flow integration capabilities:

{relevant_content}

For your integration question: "{query}"

Available integrations:
- **MCP Protocol**: Model Context Protocol support
- **GitHub**: Repository operations, PR management, code review
- **Archon**: Native task management integration
- **Memory Systems**: Cross-session persistence
- **API Endpoints**: RESTful API for custom integrations

Example GitHub integration:
```bash
npx claude-flow github "Create PR for feature branch" --repo "owner/repo"
```""",

        "general_guidance": f"""General Claude Flow guidance:

{relevant_content}

For your question: "{query}"

Claude Flow is a multi-agent orchestration framework that excels at:
- Collaborative AI workflows
- SPARC methodology implementation  
- Performance optimization through parallel execution
- Integration with knowledge management systems like Archon

Start with simple agent spawning and gradually explore swarm coordination as your needs grow."""
    }
    
    return base_responses.get(response_type, base_responses["general_guidance"])


def _get_recommendations(response_type: str, query: str) -> List[str]:
    """Get contextual recommendations based on query type."""
    
    recommendations_map = {
        "swarm_coordination": [
            "Start with adaptive topology to let Claude Flow choose optimal coordination",
            "Monitor swarm performance with 'npx claude-flow swarm monitor'",
            "Use memory operations to share context between agents",
            "Scale gradually - begin with 3-5 agents before expanding"
        ],
        "sparc_methodology": [
            "Use 'npx claude-flow sparc tdd' for test-driven development",
            "Try 'npx claude-flow sparc batch' for multiple similar tasks",
            "Enable Archon integration for project management features",
            "Review generated specifications before proceeding to implementation"
        ],
        "agent_management": [
            "Check available agent types with the MCP tool archon:claude_flow_get_agent_types",
            "Combine complementary agents (e.g., coder + reviewer + tester)",
            "Use specialized agents for specific domains (ml-developer, backend-dev)",
            "Monitor agent metrics to optimize performance"
        ],
        "setup_configuration": [
            "Ensure you have Node.js 18+ installed",
            "Configure your API keys in environment variables",
            "Start with basic agent spawning before trying swarms",
            "Test MCP integration in your IDE after setup"
        ],
        "performance_optimization": [
            "Enable parallel execution for independent tasks",
            "Use neural pattern training for repetitive workflows",
            "Monitor token usage to optimize costs",
            "Benchmark performance against your specific use cases"
        ],
        "neural_features": [
            "Start with adaptive cognitive patterns for versatility",
            "Enable learning from successful patterns",
            "Use convergent patterns for focused problem-solving",
            "Monitor neural training progress with metrics"
        ],
        "integration_tools": [
            "Test MCP tools individually before complex workflows",
            "Use Archon's task management for project tracking",
            "Enable GitHub integration for repository operations",
            "Configure memory persistence for session continuity"
        ],
        "general_guidance": [
            "Start with single agents before exploring swarms",
            "Read the integration guide in Archon's knowledge base",
            "Join the Claude Flow community for support",
            "Experiment with different topologies for your use case"
        ]
    }
    
    return recommendations_map.get(response_type, recommendations_map["general_guidance"])


def register_claude_flow_tools(mcp_server: FastMCP):
    """Register Claude Flow orchestration tools with the MCP server."""
    
    @mcp_server.tool("archon:claude_flow_swarm_init")
    async def claude_flow_swarm_init(
        topology: str = "adaptive",
        max_agents: int = 10,
        archon_integration: bool = True
    ) -> str:
        """
        Initialize Claude Flow swarm with Archon integration.
        
        Args:
            topology: Swarm topology (adaptive, mesh, hierarchical)
            max_agents: Maximum number of agents
            archon_integration: Enable Archon task integration
        """
        try:
            logger.info(f"Initializing Claude Flow swarm: topology={topology}")
            
            payload = {
                "topology": topology,
                "max_agents": max_agents,
                "archon_integration": archon_integration
            }
            
            async with get_http_client() as client:
                response_data = await client.post("http://localhost:8181/api/claude-flow/swarm/init", json=payload)
                response = response_data.json()
            
            if response["status"] == "initialized":
                return json.dumps({
                    "status": "success",
                    "message": f"Claude Flow swarm initialized with {topology} topology",
                    "session_id": response["session_id"],
                    "max_agents": max_agents,
                    "archon_integration": archon_integration
                }, indent=2)
            else:
                return json.dumps({"status": "error", "error": response.get("error", "Unknown error")})
                
        except Exception as e:
            return MCPErrorFormatter.from_exception(e, "initialize Claude Flow swarm")
    
    @mcp_server.tool("archon:claude_flow_spawn_agents")
    async def claude_flow_spawn_agents(
        objective: str,
        agents: str,
        strategy: str = "development",
        archon_task_id: Optional[str] = None
    ) -> str:
        """
        Spawn Claude Flow agents for a specific objective.
        
        Args:
            objective: The objective for the agents to accomplish
            agents: Comma-separated list of agent types
            strategy: Execution strategy (development, research, analysis)
            archon_task_id: Optional Archon task ID for integration
        """
        try:
            agent_list = [agent.strip() for agent in agents.split(",")]
            logger.info(f"Spawning agents: {agent_list} for objective: {objective}")
            
            payload = {
                "objective": objective,
                "agents": agent_list,
                "strategy": strategy,
                "archon_task_id": archon_task_id
            }
            
            async with get_http_client() as client:
                response_data = await client.post("http://localhost:8181/api/claude-flow/agents/spawn", json=payload)
                response = response_data.json()
            
            if response["status"] == "spawned":
                return json.dumps({
                    "status": "success",
                    "message": f"Spawned {len(agent_list)} agents for objective",
                    "objective": objective,
                    "agents": agent_list,
                    "strategy": strategy,
                    "archon_task_id": archon_task_id
                }, indent=2)
            else:
                return json.dumps({"status": "error", "error": response.get("error", "Unknown error")})
                
        except Exception as e:
            return MCPErrorFormatter.from_exception(e, "spawn Claude Flow agents")
    
    @mcp_server.tool("archon:claude_flow_sparc_execute")
    async def claude_flow_sparc_execute(
        task: str,
        mode: str = "tdd",
        archon_project_id: Optional[str] = None
    ) -> str:
        """
        Execute SPARC methodology workflow with Archon integration.
        
        Args:
            task: Task description for SPARC workflow
            mode: SPARC mode (tdd, batch, pipeline, concurrent)
            archon_project_id: Optional Archon project ID for integration
        """
        try:
            logger.info(f"Executing SPARC workflow: mode={mode}, task={task}")
            
            payload = {
                "task": task,
                "mode": mode,
                "archon_project_id": archon_project_id
            }
            
            async with get_http_client() as client:
                response_data = await client.post("http://localhost:8181/api/claude-flow/sparc/execute", json=payload)
                response = response_data.json()
            
            if response["status"] == "executed":
                return json.dumps({
                    "status": "success",
                    "message": f"SPARC {mode} workflow executed successfully",
                    "task": task,
                    "mode": mode,
                    "archon_project_id": archon_project_id,
                    "phases": ["specification", "pseudocode", "architecture", "refinement", "completion"]
                }, indent=2)
            else:
                return json.dumps({"status": "error", "error": response.get("error", "Unknown error")})
                
        except Exception as e:
            return MCPErrorFormatter.from_exception(e, "execute SPARC workflow")
    
    @mcp_server.tool("archon:claude_flow_get_status")
    async def claude_flow_get_status() -> str:
        """
        Get current Claude Flow swarm status and metrics.
        """
        try:
            logger.info("Getting Claude Flow swarm status")
            
            async with get_http_client() as client:
                response_data = await client.get("http://localhost:8181/api/claude-flow/status")
                response = response_data.json()
            
            if response["status"] == "success":
                info = response.get("info", {})
                return json.dumps({
                    "status": "success",
                    "timestamp": info.get("timestamp"),
                    "memory_available": info.get("memory_available", False),
                    "config_present": info.get("config_present", False),
                    "raw_status": info.get("raw_status", "No status available")
                }, indent=2)
            else:
                return json.dumps({"status": "error", "error": response.get("error", "Unknown error")})
                
        except Exception as e:
            return MCPErrorFormatter.from_exception(e, "get Claude Flow status")
    
    @mcp_server.tool("archon:claude_flow_get_metrics")
    async def claude_flow_get_metrics() -> str:
        """
        Get Claude Flow agent performance metrics.
        """
        try:
            logger.info("Getting Claude Flow agent metrics")
            
            async with get_http_client() as client:
                response_data = await client.get("http://localhost:8181/api/claude-flow/metrics")
                response = response_data.json()
            
            if response["status"] == "success":
                metrics = response.get("metrics", {})
                
                # Process and format metrics
                formatted_metrics = {}
                for category, data in metrics.items():
                    formatted_metrics[category] = data
                
                return json.dumps({
                    "status": "success",
                    "metrics": formatted_metrics,
                    "categories": list(metrics.keys()),
                    "total_categories": len(metrics)
                }, indent=2)
            else:
                return json.dumps({"status": "error", "error": response.get("error", "Unknown error")})
                
        except Exception as e:
            return MCPErrorFormatter.from_exception(e, "get agent metrics")
    
    @mcp_server.tool("archon:claude_flow_execute_hook")
    async def claude_flow_execute_hook(
        hook_name: str,
        context: str = "{}"
    ) -> str:
        """
        Execute Claude Flow hooks with context.
        
        Args:
            hook_name: Name of the hook to execute (pre-task, post-task, etc.)
            context: JSON context for hook execution
        """
        try:
            logger.info(f"Executing Claude Flow hook: {hook_name}")
            
            # Parse context JSON
            try:
                context_dict = json.loads(context)
            except json.JSONDecodeError:
                context_dict = {"raw_context": context}
            
            payload = {
                "hook_name": hook_name,
                "context": context_dict
            }
            
            async with get_http_client() as client:
                response_data = await client.post("http://localhost:8181/api/claude-flow/hooks/execute", json=payload)
                response = response_data.json()
            
            if response["status"] == "executed":
                return json.dumps({
                    "status": "success",
                    "message": f"Hook '{hook_name}' executed successfully",
                    "hook_name": hook_name,
                    "context": context_dict,
                    "result": response.get("result", "No output")
                }, indent=2)
            else:
                return json.dumps({"status": "error", "error": response.get("error", "Unknown error")})
                
        except Exception as e:
            return MCPErrorFormatter.from_exception(e, "execute Claude Flow hook")
    
    @mcp_server.tool("archon:claude_flow_memory_operation")
    async def claude_flow_memory_operation(
        operation: str,
        key: Optional[str] = None,
        value: Optional[str] = None
    ) -> str:
        """
        Perform Claude Flow memory operations.
        
        Args:
            operation: Memory operation (store, retrieve, search)
            key: Memory key for store/retrieve operations
            value: Value to store (JSON string)
        """
        try:
            logger.info(f"Claude Flow memory operation: {operation}")
            
            # Parse value if provided
            parsed_value = None
            if value:
                try:
                    parsed_value = json.loads(value)
                except json.JSONDecodeError:
                    parsed_value = value
            
            payload = {
                "operation": operation,
                "key": key,
                "value": parsed_value
            }
            
            async with get_http_client() as client:
                response_data = await client.post("http://localhost:8181/api/claude-flow/memory", json=payload)
                response = response_data.json()
            
            if response["status"] == "success":
                return json.dumps({
                    "status": "success",
                    "operation": operation,
                    "key": key,
                    "result": response.get("result", "Operation completed")
                }, indent=2)
            else:
                return json.dumps({"status": "error", "error": response.get("error", "Unknown error")})
                
        except Exception as e:
            return MCPErrorFormatter.from_exception(e, "perform memory operation")
    
    @mcp_server.tool("archon:claude_flow_neural_train")
    async def claude_flow_neural_train(
        patterns: str,
        model_type: str = "performance"
    ) -> str:
        """
        Execute Claude Flow neural pattern training.
        
        Args:
            patterns: JSON string containing training patterns
            model_type: Type of model to train (performance, coordination, etc.)
        """
        try:
            logger.info(f"Claude Flow neural training: model_type={model_type}")
            
            # Parse patterns JSON
            try:
                patterns_list = json.loads(patterns)
                if not isinstance(patterns_list, list):
                    patterns_list = [patterns_list]
            except json.JSONDecodeError:
                return json.dumps({"status": "error", "error": "Invalid patterns JSON"})
            
            payload = {
                "patterns": patterns_list,
                "model_type": model_type
            }
            
            async with get_http_client() as client:
                response_data = await client.post("http://localhost:8181/api/claude-flow/neural/train", json=payload)
                response = response_data.json()
            
            if response["status"] == "success":
                return json.dumps({
                    "status": "success",
                    "message": f"Neural training completed for {model_type} model",
                    "model_type": model_type,
                    "patterns_count": len(patterns_list),
                    "result": response.get("result", "Training completed")
                }, indent=2)
            else:
                return json.dumps({"status": "error", "error": response.get("error", "Unknown error")})
                
        except Exception as e:
            return MCPErrorFormatter.from_exception(e, "train neural patterns")
    
    @mcp_server.tool("archon:claude_flow_get_agent_types")
    async def claude_flow_get_agent_types() -> str:
        """
        Get available Claude Flow agent types organized by category.
        """
        try:
            logger.info("Getting available Claude Flow agent types")
            
            async with get_http_client() as client:
                response_data = await client.get("http://localhost:8181/api/claude-flow/agents/types")
                response = response_data.json()
            
            return json.dumps({
                "status": "success",
                "agent_types": response,
                "total_categories": len(response),
                "usage_examples": {
                    "core": "Basic development agents for coding, testing, and reviewing",
                    "sparc": "SPARC methodology agents for systematic development",
                    "archon": "Archon-specific agents with native integration",
                    "swarm": "Coordination agents for managing multiple agents",
                    "github": "GitHub integration and workflow management agents",
                    "testing": "Specialized testing and validation agents",
                    "specialized": "Domain-specific agents for mobile, DevOps, etc."
                }
            }, indent=2)
                
        except Exception as e:
            return MCPErrorFormatter.from_exception(e, "get agent types")
    
    @mcp_server.tool("archon:claude_flow_get_sparc_modes")
    async def claude_flow_get_sparc_modes() -> str:
        """
        Get available SPARC workflow modes.
        """
        try:
            logger.info("Getting available SPARC modes")
            
            async with get_http_client() as client:
                response_data = await client.get("http://localhost:8181/api/claude-flow/sparc/modes")
                response = response_data.json()
            
            return json.dumps({
                "status": "success",
                "sparc_modes": response,
                "recommended_usage": {
                    "tdd": "Use for new feature development with test-driven approach",
                    "batch": "Use for processing multiple similar tasks",
                    "pipeline": "Use for complex multi-phase projects",
                    "concurrent": "Use when multiple tasks can be parallelized",
                    "spec-pseudocode": "Use for requirements and algorithm design",
                    "architect": "Use for system design and architecture planning"
                }
            }, indent=2)
                
        except Exception as e:
            return MCPErrorFormatter.from_exception(e, "get SPARC modes")
    
    @mcp_server.tool("archon:claude_flow_expert_query")
    async def claude_flow_expert_query(
        query: str,
        use_rag: bool = True,
        context_sources: str = "claude-flow"
    ) -> str:
        """
        Query the Claude Flow expert agent with RAG-enhanced knowledge.
        
        Args:
            query: Question or request about Claude Flow
            use_rag: Whether to use RAG for enhanced responses
            context_sources: Sources to search (claude-flow, integration, sparc)
        """
        try:
            logger.info(f"Claude Flow expert query: {query}")
            
            if use_rag:
                # First, search the knowledge base for relevant information
                rag_payload = {
                    "query": query,
                    "match_count": 5,
                    "source": None  # Search all Claude Flow sources
                }
                
                async with get_http_client() as client:
                    # Get RAG results
                    rag_response_data = await client.post("http://localhost:8181/api/rag/query", json=rag_payload)
                    rag_response = rag_response_data.json()
                    
                    # Process results into context
                    context_info = []
                    if rag_response.get("success") and rag_response.get("results"):
                        for result in rag_response["results"]:
                            context_info.append({
                                "content": result["content"][:500] + "...",  # Truncate for readability
                                "source": result["metadata"].get("filename", result["metadata"].get("url", "unknown")),
                                "tags": result["metadata"].get("tags", []),
                                "similarity_score": result.get("similarity_score", 0)
                            })
                    
                    # Generate expert response based on query type
                    response_type = _determine_query_type(query)
                    expert_response = _generate_expert_response(query, response_type, context_info)
                    
                    return json.dumps({
                        "status": "success",
                        "query": query,
                        "response_type": response_type,
                        "expert_response": expert_response,
                        "knowledge_sources": len(context_info),
                        "context_used": context_info[:3] if context_info else [],  # Show top 3 sources
                        "recommendations": _get_recommendations(response_type, query)
                    }, indent=2)
                    
        except Exception as e:
            return MCPErrorFormatter.from_exception(e, "process expert query")


    # Register resources for Claude Flow documentation
    @mcp_server.resource("claude-flow://config")
    async def get_claude_flow_config() -> str:
        """Get Claude Flow configuration and setup information."""
        return """# Claude Flow Integration with Archon

## Available Tools:
- archon:claude_flow_swarm_init - Initialize swarm with Archon integration
- archon:claude_flow_spawn_agents - Spawn agents for specific objectives  
- archon:claude_flow_sparc_execute - Execute SPARC methodology workflows
- archon:claude_flow_get_status - Get swarm status and metrics
- archon:claude_flow_get_metrics - Get agent performance metrics
- archon:claude_flow_execute_hook - Execute lifecycle hooks
- archon:claude_flow_memory_operation - Perform memory operations
- archon:claude_flow_neural_train - Train neural patterns
- archon:claude_flow_get_agent_types - Get available agent types
- archon:claude_flow_get_sparc_modes - Get SPARC workflow modes
- archon:claude_flow_expert_query - Query Claude Flow expert with RAG-enhanced knowledge

## Integration Features:
- Native Archon task management integration
- SPARC methodology with progressive refinement
- Swarm coordination with adaptive topologies
- Neural pattern learning and optimization
- Memory persistence across sessions
- Real-time performance monitoring
- RAG-enhanced expert consultation system

## Expert Query Categories:
- Swarm Coordination: Topology setup, multi-agent orchestration
- SPARC Methodology: Specification, pseudocode, architecture, refinement
- Agent Management: Agent types, spawning, capabilities
- Setup & Configuration: Installation, initialization, integration
- Performance Optimization: Metrics, speed improvements, benchmarking
- Neural Features: Cognitive patterns, training, autonomous learning
- Integration Tools: MCP, GitHub, API endpoints, memory systems
- General Guidance: Best practices, troubleshooting, recommendations
"""