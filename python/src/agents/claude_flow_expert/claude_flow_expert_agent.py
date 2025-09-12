"""
Claude Flow Expert Agent - Multi-Agent Orchestration Specialist.

Premier multi-agent orchestration and swarm coordination specialist in the 
Claude Flow ecosystem with 64+ agent expertise, RAG integration, and 
sophisticated workflow automation with Archon knowledge systems.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from ..base_agent import ArchonDependencies, BaseAgent, BaseAgentOutput
from ..mcp_client import MCPClient, get_mcp_client
from .capability_matrix import AgentCapabilityMatrix, QueryType
from .coordination_hooks import ClaudeFlowCoordinator
from .fallback_strategies import FallbackManager

logger = logging.getLogger(__name__)


class ProcessingStrategy(str, Enum):
    """Processing strategies for different types of queries."""
    
    SINGLE_AGENT = "single_agent"
    MULTI_AGENT = "multi_agent"
    RAG_ENHANCED = "rag_enhanced"
    FALLBACK = "fallback"
    HYBRID = "hybrid"


@dataclass 
class ClaudeFlowExpertConfig:
    """Configuration for the Claude Flow Expert Agent."""
    
    # Core settings
    model: str = "openai:gpt-4o"
    max_retries: int = 3
    timeout: int = 120
    enable_rate_limiting: bool = True
    
    # RAG settings
    rag_enabled: bool = True
    rag_fallback_enabled: bool = True
    max_rag_results: int = 5
    
    # Agent coordination
    max_coordinated_agents: int = 10
    coordination_timeout: int = 300
    
    # Memory settings
    memory_persistence_enabled: bool = True
    memory_ttl: int = 3600  # 1 hour
    
    # Performance settings
    enable_metrics: bool = True
    circuit_breaker_enabled: bool = True
    circuit_breaker_threshold: int = 5


class ClaudeFlowExpertDependencies(ArchonDependencies):
    """Dependencies for the Claude Flow Expert Agent."""
    
    query_type: Optional[QueryType] = None
    processing_strategy: Optional[ProcessingStrategy] = None
    archon_project_id: Optional[str] = None
    archon_task_id: Optional[str] = None
    rag_source_filter: Optional[str] = None
    coordinate_agents: bool = True
    metrics_callback: Optional[callable] = None
    progress_callback: Optional[callable] = None


class QueryRequest(BaseModel):
    """Query request model for the claude flow expert agent."""
    
    query: str = Field(..., description="The user's query or task")
    query_type: Optional[QueryType] = Field(default=None, description="Type of query")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    preferred_agents: Optional[List[str]] = Field(default=None, description="Preferred agent types")
    require_rag: bool = Field(default=False, description="Whether RAG is required")
    max_agents: int = Field(default=3, description="Maximum agents to coordinate")


class QueryResponse(BaseAgentOutput):
    """Response model for claude flow expert agent queries."""
    
    query_type: Optional[QueryType] = None
    processing_strategy: Optional[ProcessingStrategy] = None
    agents_used: List[str] = Field(default_factory=list)
    rag_sources_used: List[str] = Field(default_factory=list)
    coordination_metrics: Optional[Dict[str, Any]] = None
    fallback_used: bool = False
    processing_time: Optional[float] = None
    confidence_score: Optional[float] = None


class CircuitBreakerState(Enum):
    """Circuit breaker states for resilience."""
    
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for handling service failures."""
    
    def __init__(self, threshold: int = 5, timeout: int = 60):
        self.threshold = threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = CircuitBreakerState.CLOSED
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker moving to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED state")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.threshold:
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
            
            raise


class ClaudeFlowExpertAgent(BaseAgent[ClaudeFlowExpertDependencies, QueryResponse]):
    """
    Claude Flow Expert Agent - Multi-Agent Orchestration Specialist.
    
    This agent provides:
    - 64+ specialized agent coordination and routing
    - Intelligent query processing with RAG enhancement  
    - Claude Flow swarm orchestration and workflow automation
    - Multi-topology coordination (hierarchical, mesh, adaptive)
    - Fallback strategies for resilience
    - Performance monitoring and metrics
    """
    
    def __init__(self, config: ClaudeFlowExpertConfig = None, **kwargs):
        """Initialize the Claude Flow Expert agent."""
        self.config = config or ClaudeFlowExpertConfig()
        
        # Initialize components
        self.capability_matrix = AgentCapabilityMatrix()
        self.claude_flow_coordinator = ClaudeFlowCoordinator()
        self.fallback_manager = FallbackManager()
        
        # Circuit breakers for different services
        self.circuit_breakers = {
            "rag": CircuitBreaker(threshold=self.config.circuit_breaker_threshold),
            "agents": CircuitBreaker(threshold=self.config.circuit_breaker_threshold),
            "coordination": CircuitBreaker(threshold=self.config.circuit_breaker_threshold)
        }
        
        # Metrics
        self.metrics = {
            "queries_processed": 0,
            "successful_queries": 0,
            "rag_queries": 0,
            "multi_agent_coordinations": 0,
            "fallbacks_used": 0,
            "average_processing_time": 0.0
        }
        
        # Initialize MCP client
        self.mcp_client: Optional[MCPClient] = None
        
        super().__init__(
            model=self.config.model,
            name="ClaudeFlowExpertAgent",
            retries=self.config.max_retries,
            enable_rate_limiting=self.config.enable_rate_limiting,
            **kwargs
        )
    
    def _create_agent(self, **kwargs) -> Agent:
        """Create and configure the PydanticAI agent."""
        agent = Agent(
            model=self.config.model,
            result_type=QueryResponse,
            system_prompt=self.get_system_prompt(),
            **kwargs
        )
        
        # Register tools
        self._register_tools(agent)
        
        return agent
    
    def get_system_prompt(self) -> str:
        """Get the system prompt for the claude flow expert agent."""
        return """
You are the Claude Flow Expert Agent for the Archon AI development platform, orchestrating intelligent query processing and agent coordination.

Your core responsibilities:
1. **Query Analysis**: Determine query type and optimal processing strategy
2. **RAG Enhancement**: Leverage Archon's knowledge base for context-aware responses
3. **Agent Orchestration**: Route queries to specialized agents based on capability matrix
4. **Multi-Agent Coordination**: Coordinate multiple agents via Claude Flow swarm patterns
5. **Fallback Management**: Provide resilient fallback strategies when services fail

Key Integration Points:
- **Archon MCP Server**: For task management, documents, and RAG queries
- **Claude Flow**: For swarm coordination and agent orchestration
- **PydanticAI Framework**: For structured agent interactions

Decision Framework:
- Analyze query complexity and requirements
- Check RAG knowledge base for relevant context
- Route to appropriate specialized agents
- Coordinate multi-agent workflows when needed
- Implement fallback strategies for resilience

Always provide structured responses with processing metadata and confidence scores.
"""
    
    def _register_tools(self, agent: Agent):
        """Register tools for the claude flow expert agent."""
        
        @agent.tool
        async def perform_rag_query(
            ctx, 
            query: str, 
            source_filter: str = None, 
            match_count: int = 5
        ) -> str:
            """Perform RAG query against Archon's knowledge base."""
            try:
                if not self.mcp_client:
                    self.mcp_client = await get_mcp_client()
                
                result = await self.mcp_client.perform_rag_query(
                    query=query,
                    source=source_filter,
                    match_count=match_count
                )
                
                self.metrics["rag_queries"] += 1
                return result
                
            except Exception as e:
                logger.error(f"RAG query failed: {e}")
                return f"RAG query failed: {str(e)}"
        
        @agent.tool
        async def search_code_examples(
            ctx,
            query: str,
            source_id: str = None,
            match_count: int = 5
        ) -> str:
            """Search for relevant code examples in the knowledge base."""
            try:
                if not self.mcp_client:
                    self.mcp_client = await get_mcp_client()
                
                result = await self.mcp_client.search_code_examples(
                    query=query,
                    source_id=source_id,
                    match_count=match_count
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Code search failed: {e}")
                return f"Code search failed: {str(e)}"
        
        @agent.tool
        async def coordinate_agents(
            ctx,
            objective: str,
            agent_types: List[str],
            max_agents: int = 3
        ) -> str:
            """Coordinate multiple agents via Claude Flow for complex tasks."""
            try:
                result = await self.claude_flow_coordinator.coordinate_multi_agent(
                    objective=objective,
                    agent_types=agent_types[:max_agents]
                )
                
                self.metrics["multi_agent_coordinations"] += 1
                return json.dumps(result)
                
            except Exception as e:
                logger.error(f"Agent coordination failed: {e}")
                return f"Agent coordination failed: {str(e)}"
        
        @agent.tool
        async def get_agent_capabilities(ctx, domain: str = None) -> str:
            """Get available agent capabilities and routing information."""
            try:
                capabilities = self.capability_matrix.get_capabilities(domain)
                return json.dumps(capabilities)
                
            except Exception as e:
                logger.error(f"Failed to get capabilities: {e}")
                return f"Failed to get capabilities: {str(e)}"
    
    async def process_query(
        self, 
        request: QueryRequest, 
        deps: ClaudeFlowExpertDependencies
    ) -> QueryResponse:
        """
        Main query processing method with RAG enhancement and intelligent routing.
        
        Args:
            request: The query request with context
            deps: Agent dependencies
            
        Returns:
            QueryResponse with results and processing metadata
        """
        start_time = time.time()
        self.metrics["queries_processed"] += 1
        
        try:
            # Update progress
            if deps.progress_callback:
                await deps.progress_callback({
                    "step": "query_analysis",
                    "message": "Analyzing query and determining processing strategy"
                })
            
            # Determine query type and strategy
            query_type = request.query_type or await self._classify_query(request.query)
            processing_strategy = await self._determine_processing_strategy(request, query_type)
            
            # Initialize response
            response = QueryResponse(
                success=False,
                message="",
                query_type=query_type,
                processing_strategy=processing_strategy
            )
            
            # Process based on strategy
            if processing_strategy == ProcessingStrategy.RAG_ENHANCED:
                response = await self._process_rag_enhanced(request, deps, response)
            elif processing_strategy == ProcessingStrategy.MULTI_AGENT:
                response = await self._process_multi_agent(request, deps, response)
            elif processing_strategy == ProcessingStrategy.SINGLE_AGENT:
                response = await self._process_single_agent(request, deps, response)
            elif processing_strategy == ProcessingStrategy.HYBRID:
                response = await self._process_hybrid(request, deps, response)
            else:
                response = await self._process_fallback(request, deps, response)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            response.processing_time = processing_time
            
            # Update metrics
            if response.success:
                self.metrics["successful_queries"] += 1
            
            self._update_average_processing_time(processing_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            
            # Attempt fallback
            response = await self._process_fallback(request, deps, QueryResponse(
                success=False,
                message=f"Query processing failed: {str(e)}",
                query_type=query_type,
                processing_strategy=ProcessingStrategy.FALLBACK,
                fallback_used=True
            ))
            
            response.processing_time = time.time() - start_time
            return response
    
    async def route_to_agent(
        self, 
        query: str, 
        query_type: QueryType,
        preferred_agents: Optional[List[str]] = None
    ) -> List[str]:
        """
        Intelligent agent selection and routing based on capability matrix.
        
        Args:
            query: The user's query
            query_type: Classified query type
            preferred_agents: Optional preferred agent types
            
        Returns:
            List of recommended agent types
        """
        try:
            # Get capabilities for query type
            capabilities = self.capability_matrix.get_capabilities_for_query_type(query_type)
            
            # Filter by preferred agents if specified
            if preferred_agents:
                capabilities = [cap for cap in capabilities if cap["agent_type"] in preferred_agents]
            
            # Sort by relevance score
            capabilities.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
            # Return top agent types
            return [cap["agent_type"] for cap in capabilities[:3]]
            
        except Exception as e:
            logger.error(f"Agent routing failed: {e}")
            # Return default agents based on query type
            return self._get_default_agents(query_type)
    
    async def coordinate_multi_agent(
        self,
        objective: str,
        agent_types: List[str],
        archon_task_id: Optional[str] = None,
        max_agents: int = 5
    ) -> Dict[str, Any]:
        """
        Multi-agent task orchestration via Claude Flow coordination.
        
        Args:
            objective: The task objective
            agent_types: List of agent types to coordinate
            archon_task_id: Optional Archon task ID for integration
            max_agents: Maximum number of agents to coordinate
            
        Returns:
            Coordination results and metrics
        """
        try:
            if self.config.circuit_breaker_enabled:
                return await self.circuit_breakers["coordination"].call(
                    self._coordinate_multi_agent_internal,
                    objective, agent_types[:max_agents], archon_task_id
                )
            else:
                return await self._coordinate_multi_agent_internal(
                    objective, agent_types[:max_agents], archon_task_id
                )
                
        except Exception as e:
            logger.error(f"Multi-agent coordination failed: {e}")
            # Attempt single-agent fallback
            return await self.fallback_manager.single_agent_fallback(objective, agent_types[0] if agent_types else "coder")
    
    async def _coordinate_multi_agent_internal(
        self,
        objective: str,
        agent_types: List[str],
        archon_task_id: Optional[str]
    ) -> Dict[str, Any]:
        """Internal multi-agent coordination implementation."""
        coordination_start = time.time()
        
        # Initialize Claude Flow coordination
        await self.claude_flow_coordinator.initialize_swarm(
            topology="adaptive",
            max_agents=len(agent_types)
        )
        
        # Spawn agents
        spawn_result = await self.claude_flow_coordinator.spawn_agents(
            objective=objective,
            agents=agent_types,
            archon_task_id=archon_task_id
        )
        
        # Monitor coordination
        status = await self.claude_flow_coordinator.monitor_coordination()
        
        coordination_time = time.time() - coordination_start
        
        return {
            "status": "success",
            "agents_spawned": agent_types,
            "spawn_result": spawn_result,
            "coordination_status": status,
            "coordination_time": coordination_time,
            "objective": objective,
            "archon_task_id": archon_task_id
        }
    
    async def fallback_to_wiki(self, query: str) -> Dict[str, Any]:
        """
        Fallback information retrieval when RAG queries fail.
        
        Args:
            query: The user's query
            
        Returns:
            Fallback information results
        """
        try:
            result = await self.fallback_manager.wiki_search_fallback(query)
            self.metrics["fallbacks_used"] += 1
            return result
            
        except Exception as e:
            logger.error(f"Wiki fallback failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "fallback_type": "wiki",
                "query": query
            }
    
    async def update_capabilities(self, capabilities_update: Dict[str, Any]) -> bool:
        """
        Dynamic capability matrix updates.
        
        Args:
            capabilities_update: New capability definitions
            
        Returns:
            True if update successful
        """
        try:
            return self.capability_matrix.update_capabilities(capabilities_update)
            
        except Exception as e:
            logger.error(f"Capability update failed: {e}")
            return False
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self.metrics,
            "timestamp": datetime.now().isoformat(),
            "circuit_breaker_states": {
                name: breaker.state.value 
                for name, breaker in self.circuit_breakers.items()
            },
            "config": {
                "model": self.config.model,
                "max_agents": self.config.max_coordinated_agents,
                "rag_enabled": self.config.rag_enabled
            }
        }
    
    # Helper methods
    
    async def _classify_query(self, query: str) -> QueryType:
        """Classify query type using heuristics and ML."""
        query_lower = query.lower()
        
        # Simple heuristic classification
        if any(word in query_lower for word in ["code", "implement", "function", "class", "bug"]):
            return QueryType.CODING
        elif any(word in query_lower for word in ["research", "find", "search", "learn", "what is"]):
            return QueryType.RESEARCH
        elif any(word in query_lower for word in ["analyze", "review", "check", "examine"]):
            return QueryType.ANALYSIS
        elif any(word in query_lower for word in ["task", "project", "manage", "organize"]):
            return QueryType.TASK_MANAGEMENT
        elif any(word in query_lower for word in ["coordinate", "organize", "plan", "workflow"]):
            return QueryType.COORDINATION
        else:
            return QueryType.GENERAL
    
    async def _determine_processing_strategy(
        self, 
        request: QueryRequest, 
        query_type: QueryType
    ) -> ProcessingStrategy:
        """Determine optimal processing strategy."""
        # RAG required or beneficial
        if request.require_rag or query_type in [QueryType.RESEARCH, QueryType.KNOWLEDGE]:
            return ProcessingStrategy.RAG_ENHANCED
        
        # Complex coordination needed
        if query_type == QueryType.COORDINATION or request.max_agents > 1:
            return ProcessingStrategy.MULTI_AGENT
        
        # Hybrid approach for complex coding/analysis tasks
        if query_type in [QueryType.CODING, QueryType.ANALYSIS] and len(request.query) > 200:
            return ProcessingStrategy.HYBRID
        
        # Default single agent
        return ProcessingStrategy.SINGLE_AGENT
    
    async def _process_rag_enhanced(
        self, 
        request: QueryRequest, 
        deps: ClaudeFlowExpertDependencies,
        response: QueryResponse
    ) -> QueryResponse:
        """Process query with RAG enhancement."""
        try:
            # Enhanced prompt with user query
            enhanced_prompt = f"""
Query: {request.query}
Context: {request.context}

Please process this query using RAG tools to find relevant information from the knowledge base.
Focus on providing accurate, context-aware responses.
"""
            
            # Run agent with RAG tools
            result = await self.run(enhanced_prompt, deps)
            
            response.success = True
            response.message = result.message if hasattr(result, 'message') else str(result)
            response.data = result.data if hasattr(result, 'data') else {"result": str(result)}
            response.rag_sources_used = ["archon_knowledge_base"]
            
            return response
            
        except Exception as e:
            logger.error(f"RAG-enhanced processing failed: {e}")
            # Fallback to non-RAG processing
            return await self._process_fallback(request, deps, response)
    
    async def _process_multi_agent(
        self,
        request: QueryRequest,
        deps: ClaudeFlowExpertDependencies,
        response: QueryResponse
    ) -> QueryResponse:
        """Process query with multi-agent coordination."""
        try:
            # Route to appropriate agents
            agent_types = await self.route_to_agent(
                request.query,
                request.query_type or QueryType.GENERAL,
                request.preferred_agents
            )
            
            # Coordinate agents
            coordination_result = await self.coordinate_multi_agent(
                objective=request.query,
                agent_types=agent_types,
                archon_task_id=deps.archon_task_id,
                max_agents=request.max_agents
            )
            
            response.success = coordination_result.get("status") == "success"
            response.message = "Multi-agent coordination completed"
            response.data = coordination_result
            response.agents_used = agent_types
            response.coordination_metrics = {
                "coordination_time": coordination_result.get("coordination_time"),
                "agents_count": len(agent_types)
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Multi-agent processing failed: {e}")
            return await self._process_fallback(request, deps, response)
    
    async def _process_single_agent(
        self,
        request: QueryRequest,
        deps: ClaudeFlowExpertDependencies,
        response: QueryResponse
    ) -> QueryResponse:
        """Process query with single agent."""
        try:
            # Route to single best agent
            agent_types = await self.route_to_agent(
                request.query,
                request.query_type or QueryType.GENERAL,
                request.preferred_agents
            )
            
            best_agent = agent_types[0] if agent_types else "coder"
            
            # Process with selected agent type context
            enhanced_prompt = f"""
Acting as a {best_agent} agent, please process this query:

Query: {request.query}
Context: {request.context}

Provide a focused response based on your specialized capabilities.
"""
            
            result = await self.run(enhanced_prompt, deps)
            
            response.success = True
            response.message = result.message if hasattr(result, 'message') else str(result)
            response.data = result.data if hasattr(result, 'data') else {"result": str(result)}
            response.agents_used = [best_agent]
            
            return response
            
        except Exception as e:
            logger.error(f"Single agent processing failed: {e}")
            return await self._process_fallback(request, deps, response)
    
    async def _process_hybrid(
        self,
        request: QueryRequest,
        deps: ClaudeFlowExpertDependencies,
        response: QueryResponse
    ) -> QueryResponse:
        """Process query with hybrid RAG + multi-agent approach."""
        try:
            # First attempt RAG enhancement
            rag_response = await self._process_rag_enhanced(request, deps, QueryResponse(
                success=False, message="", query_type=request.query_type
            ))
            
            # Then coordinate agents with RAG context
            if rag_response.success and rag_response.data:
                enhanced_request = QueryRequest(
                    query=f"{request.query}\n\nRAG Context: {rag_response.data}",
                    query_type=request.query_type,
                    context=request.context,
                    max_agents=min(2, request.max_agents)  # Limit for hybrid
                )
                
                multi_response = await self._process_multi_agent(enhanced_request, deps, QueryResponse(
                    success=False, message="", query_type=request.query_type
                ))
                
                if multi_response.success:
                    response = multi_response
                    response.rag_sources_used = rag_response.rag_sources_used
                    response.processing_strategy = ProcessingStrategy.HYBRID
                    return response
            
            # Fallback to single strategy that worked
            return rag_response if rag_response.success else await self._process_fallback(request, deps, response)
            
        except Exception as e:
            logger.error(f"Hybrid processing failed: {e}")
            return await self._process_fallback(request, deps, response)
    
    async def _process_fallback(
        self,
        request: QueryRequest,
        deps: ClaudeFlowExpertDependencies,
        response: QueryResponse
    ) -> QueryResponse:
        """Process query using fallback strategies."""
        try:
            # Try wiki fallback
            fallback_result = await self.fallback_to_wiki(request.query)
            
            if fallback_result.get("status") == "success":
                response.success = True
                response.message = "Processed using fallback information retrieval"
                response.data = fallback_result
                response.fallback_used = True
                response.processing_strategy = ProcessingStrategy.FALLBACK
            else:
                # Final fallback - direct agent processing without tools
                direct_prompt = f"Please help with this query: {request.query}"
                result = await self.run(direct_prompt, deps)
                
                response.success = True
                response.message = result.message if hasattr(result, 'message') else str(result)
                response.data = {"result": str(result)}
                response.fallback_used = True
            
            return response
            
        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")
            response.success = False
            response.message = f"All processing strategies failed: {str(e)}"
            response.fallback_used = True
            return response
    
    def _get_default_agents(self, query_type: QueryType) -> List[str]:
        """Get default agents for a query type."""
        defaults = {
            QueryType.CODING: ["coder", "reviewer", "tester"],
            QueryType.RESEARCH: ["researcher", "analyst"],
            QueryType.ANALYSIS: ["analyst", "reviewer"],
            QueryType.COORDINATION: ["coordinator", "planner"],
            QueryType.TASK_MANAGEMENT: ["planner", "coordinator"],
            QueryType.KNOWLEDGE: ["researcher", "analyst"],
            QueryType.GENERAL: ["coder", "researcher"]
        }
        return defaults.get(query_type, ["coder", "researcher"])
    
    def _update_average_processing_time(self, processing_time: float):
        """Update average processing time metric."""
        current_avg = self.metrics["average_processing_time"]
        total_queries = self.metrics["queries_processed"]
        
        if total_queries == 1:
            self.metrics["average_processing_time"] = processing_time
        else:
            # Weighted average
            self.metrics["average_processing_time"] = (
                (current_avg * (total_queries - 1) + processing_time) / total_queries
            )