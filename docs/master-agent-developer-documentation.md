# Master Agent System - Developer Documentation

## Table of Contents

- [API Reference](#api-reference)
- [Integration Patterns](#integration-patterns)
- [Extension Points](#extension-points)
- [Performance Tuning Guidelines](#performance-tuning-guidelines)
- [Testing Framework](#testing-framework)
- [Custom Agent Development](#custom-agent-development)
- [Advanced Configuration](#advanced-configuration)

---

## API Reference

### Master Agent Core API

The Master Agent System provides a comprehensive API for intelligent task orchestration, RAG-enhanced processing, and multi-agent coordination.

#### Core Agent Classes

##### MasterAgent

The primary orchestrator for all AI operations within Archon.

```python
from src.agents.master.master_agent import MasterAgent, MasterAgentConfig

class MasterAgent(BaseAgent):
    """
    Core Master Agent for Archon with PydanticAI framework integration.
    
    Provides intelligent routing, RAG-enhanced query processing, and seamless 
    integration with Claude Flow's swarm coordination and Archon's knowledge systems.
    """
    
    def __init__(self, config: MasterAgentConfig):
        """
        Initialize Master Agent with configuration.
        
        Args:
            config: MasterAgentConfig instance with system settings
        """
        
    async def process_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None,
        strategy: ProcessingStrategy = ProcessingStrategy.HYBRID
    ) -> MasterAgentResponse:
        """
        Process a query using optimal strategy selection.
        
        Args:
            query: Natural language query or task description
            context: Optional context dictionary with project/session data
            strategy: Processing strategy (SINGLE_AGENT, MULTI_AGENT, RAG_ENHANCED, etc.)
            
        Returns:
            MasterAgentResponse with results, metadata, and performance metrics
            
        Example:
            >>> agent = MasterAgent(config)
            >>> response = await agent.process_query(
            ...     "Optimize FastAPI performance for high-traffic API",
            ...     context={"project_id": "123", "domain": "backend"},
            ...     strategy=ProcessingStrategy.RAG_ENHANCED
            ... )
            >>> print(response.content)  # Optimized recommendations
            >>> print(response.sources)  # Referenced knowledge sources
        """
```

##### Configuration Classes

```python
@dataclass
class MasterAgentConfig:
    """Configuration for the Master Agent."""
    
    # Core settings
    model: str = "openai:gpt-4o"           # Primary LLM model
    max_retries: int = 3                   # Retry attempts for failed operations
    timeout: int = 120                     # Operation timeout in seconds
    enable_rate_limiting: bool = True      # Rate limiting for API calls
    
    # RAG settings
    rag_strategies: List[str] = field(default_factory=lambda: [
        "contextual_embeddings",
        "hybrid_search", 
        "agentic_rag",
        "reranking"
    ])
    
    # Agent coordination settings
    swarm_coordination: bool = True        # Enable Claude Flow coordination
    fallback_strategies: List[str] = field(default_factory=lambda: [
        "single_agent",
        "rag_enhanced"
    ])
    
    # Performance targets
    performance_targets: Dict[str, str] = field(default_factory=lambda: {
        "simple_query": "200ms",
        "complex_query": "500ms", 
        "multi_step_task": "2s"
    })
```

#### Response Models

```python
class MasterAgentResponse(BaseModel):
    """Response model for Master Agent operations."""
    
    content: str                           # Primary response content
    confidence: float                      # Confidence score (0.0-1.0)
    processing_time: float                 # Response time in seconds
    strategy_used: ProcessingStrategy      # Strategy that was applied
    sources: List[SourceReference]         # Knowledge sources referenced
    metadata: Dict[str, Any]               # Additional response metadata
    recommendations: List[str]             # Follow-up recommendations
    
class SourceReference(BaseModel):
    """Reference to a knowledge source used in processing."""
    
    source_id: str                         # Unique source identifier
    source_type: str                       # Type: document, code, api_reference
    relevance_score: float                 # Relevance to query (0.0-1.0)
    chunk_text: str                        # Referenced text snippet
    metadata: Dict[str, Any]               # Source-specific metadata
```

#### Agent Capability Matrix API

```python
from src.agents.master.capability_matrix import AgentCapabilityMatrix, QueryType

class AgentCapabilityMatrix:
    """
    Intelligent agent selection and capability routing system.
    """
    
    def get_optimal_agent(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> AgentRecommendation:
        """
        Determine optimal agent for query processing.
        
        Args:
            query: Input query to analyze
            context: Optional context for routing decisions
            
        Returns:
            AgentRecommendation with agent selection and reasoning
            
        Example:
            >>> matrix = AgentCapabilityMatrix()
            >>> recommendation = matrix.get_optimal_agent(
            ...     "Optimize vector search performance",
            ...     context={"domain": "rag", "complexity": "high"}
            ... )
            >>> print(recommendation.agent_type)  # "archon-rag-specialist"
            >>> print(recommendation.confidence)  # 0.95
        """
        
    def get_agent_capabilities(self, agent_type: str) -> AgentCapabilities:
        """
        Get detailed capabilities for a specific agent type.
        
        Args:
            agent_type: Agent identifier (e.g., "archon-master")
            
        Returns:
            AgentCapabilities with skills, domains, and performance metrics
        """

class AgentRecommendation(BaseModel):
    """Agent selection recommendation."""
    
    agent_type: str                        # Recommended agent identifier
    confidence: float                      # Selection confidence (0.0-1.0) 
    reasoning: str                         # Human-readable selection rationale
    alternative_agents: List[str]          # Alternative agent options
    expected_performance: Dict[str, Any]   # Performance expectations
```

### MCP Integration API

#### MCP Client Interface

```python
from src.agents.mcp_client import MCPClient

class MCPClient:
    """
    Model Context Protocol client for integration with Archon services.
    """
    
    def __init__(self, base_url: str = "http://localhost:8051"):
        """Initialize MCP client with server URL."""
        
    async def perform_rag_query(
        self,
        query: str,
        sources: Optional[List[str]] = None,
        strategy: str = "hybrid",
        max_results: int = 5
    ) -> RAGQueryResponse:
        """
        Execute RAG query against knowledge base.
        
        Args:
            query: Search query string
            sources: Optional list of source IDs to search within
            strategy: RAG strategy ("contextual", "hybrid", "agentic", "rerank")
            max_results: Maximum number of results to return
            
        Returns:
            RAGQueryResponse with search results and metadata
            
        Example:
            >>> client = MCPClient()
            >>> response = await client.perform_rag_query(
            ...     "FastAPI authentication patterns",
            ...     sources=["fastapi_docs", "security_guides"],
            ...     strategy="hybrid",
            ...     max_results=3
            ... )
            >>> for result in response.results:
            ...     print(f"{result.title}: {result.relevance_score}")
        """
        
    async def create_task(
        self,
        project_id: str,
        title: str,
        description: str,
        priority: str = "medium",
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskResponse:
        """
        Create a new task in project management system.
        
        Args:
            project_id: Target project identifier
            title: Task title
            description: Detailed task description
            priority: Task priority (low, medium, high, urgent)
            metadata: Optional task metadata
            
        Returns:
            TaskResponse with created task information
        """
        
    async def search_code_examples(
        self,
        query: str,
        language: Optional[str] = None,
        framework: Optional[str] = None
    ) -> CodeExampleResponse:
        """
        Search for relevant code examples in knowledge base.
        
        Args:
            query: Description of desired functionality
            language: Programming language filter
            framework: Framework/library filter
            
        Returns:
            CodeExampleResponse with matching code snippets
        """
```

#### MCP Tool Definitions

The system exposes 14 specialized MCP tools for AI client integration:

**RAG Operations (7 tools):**
- `perform_rag_query` - Execute knowledge base searches
- `get_available_sources` - List available knowledge sources  
- `search_code_examples` - Find code implementation patterns
- `get_document_summary` - Summarize document contents
- `analyze_document_relevance` - Assess document relevance to query
- `get_search_suggestions` - Get query improvement suggestions
- `update_knowledge_index` - Refresh knowledge base index

**Project Management (7 tools):**
- `create_project` - Initialize new project
- `get_project` - Retrieve project details
- `list_projects` - List available projects  
- `create_task` - Add task to project
- `update_task` - Modify existing task
- `list_tasks` - Retrieve project tasks
- `create_document` - Add project documentation

### Claude Flow Integration API

#### Swarm Coordination

```python
from src.agents.master.coordination_hooks import ClaudeFlowCoordinator

class ClaudeFlowCoordinator:
    """
    Integration layer for Claude Flow swarm coordination.
    """
    
    async def init_swarm_session(
        self,
        topology: str = "mesh",
        max_agents: int = 6,
        session_id: Optional[str] = None
    ) -> SwarmSession:
        """
        Initialize swarm coordination session.
        
        Args:
            topology: Swarm topology (mesh, hierarchical, adaptive)
            max_agents: Maximum number of coordinated agents
            session_id: Optional session identifier
            
        Returns:
            SwarmSession with coordination details
        """
        
    async def spawn_coordinated_agent(
        self,
        agent_type: str,
        task_description: str,
        coordination_rules: Optional[Dict[str, Any]] = None
    ) -> CoordinatedAgent:
        """
        Spawn agent with swarm coordination capabilities.
        
        Args:
            agent_type: Type of agent to spawn
            task_description: Task for the agent to perform
            coordination_rules: Optional coordination constraints
            
        Returns:
            CoordinatedAgent instance with swarm integration
        """
        
    async def coordinate_multi_agent_task(
        self,
        task: ComplexTask,
        agents: List[str],
        strategy: str = "collaborative"
    ) -> MultiAgentResponse:
        """
        Coordinate complex task across multiple agents.
        
        Args:
            task: Complex task requiring multiple agents
            agents: List of agent types to coordinate
            strategy: Coordination strategy
            
        Returns:
            MultiAgentResponse with consolidated results
        """
```

---

## Integration Patterns

### Archon PRP Framework Integration

The Master Agent System integrates deeply with Archon's Progressive Refinement Protocol (PRP) framework, providing intelligent orchestration across all SPARC phases.

#### PRP Phase Integration

```python
class PRPIntegration:
    """
    Progressive Refinement Protocol integration for SPARC methodology.
    """
    
    async def specification_phase(
        self,
        requirements: str,
        project_context: Dict[str, Any]
    ) -> SpecificationResponse:
        """
        Handle Specification phase with RAG-enhanced analysis.
        
        Process:
        1. Analyze requirements against existing knowledge base
        2. Identify gaps and ambiguities in specifications
        3. Provide technical feasibility assessment
        4. Suggest specification refinements
        
        Returns:
            SpecificationResponse with analysis and recommendations
        """
        # RAG query for similar projects and patterns
        similar_projects = await self.rag_client.perform_rag_query(
            f"Similar projects: {requirements}",
            sources=["architectural_decisions", "project_templates"]
        )
        
        # Analyze feasibility with master agent
        feasibility = await self.master_agent.process_query(
            f"Technical feasibility analysis: {requirements}",
            context={**project_context, "phase": "specification"}
        )
        
        return SpecificationResponse(
            refined_requirements=feasibility.content,
            gap_analysis=self._extract_gaps(feasibility),
            recommendations=feasibility.recommendations,
            similar_implementations=similar_projects.results
        )
    
    async def architecture_phase(
        self,
        specifications: str,
        constraints: Dict[str, Any]
    ) -> ArchitectureResponse:
        """
        Handle Architecture phase with pattern-based design.
        
        Process:
        1. Search for applicable architectural patterns
        2. Validate against system constraints
        3. Generate architecture recommendations
        4. Identify integration points and risks
        """
        # Search architectural patterns
        patterns = await self.rag_client.search_code_examples(
            f"Architecture patterns for: {specifications}",
            framework="system_design"
        )
        
        # Master agent architectural analysis
        architecture = await self.master_agent.process_query(
            f"Design system architecture: {specifications}",
            context={"constraints": constraints, "phase": "architecture"},
            strategy=ProcessingStrategy.RAG_ENHANCED
        )
        
        return ArchitectureResponse(
            architectural_design=architecture.content,
            applicable_patterns=patterns.examples,
            integration_points=self._identify_integrations(architecture),
            risk_assessment=self._analyze_risks(architecture)
        )
```

#### Multi-Agent SPARC Coordination

```python
async def coordinate_sparc_workflow(
    self,
    project_requirements: str,
    team_preferences: Dict[str, Any]
) -> SPARCWorkflowResponse:
    """
    Coordinate complete SPARC workflow across specialized agents.
    
    Agents involved:
    - specification: Requirements analysis and refinement
    - pseudocode: Algorithm design and logic planning  
    - architecture: System design and integration planning
    - refinement: Implementation guidance and optimization
    - completion: Integration testing and deployment
    """
    
    # Initialize swarm coordination
    session = await self.coordinator.init_swarm_session(
        topology="hierarchical",
        max_agents=5
    )
    
    # Spawn specialized agents for each phase
    agents = {}
    for phase in ["specification", "pseudocode", "architecture", "refinement", "completion"]:
        agents[phase] = await self.coordinator.spawn_coordinated_agent(
            agent_type=f"sparc-{phase}",
            task_description=f"Handle {phase} phase for: {project_requirements}",
            coordination_rules={"session_id": session.id, "phase": phase}
        )
    
    # Execute phases with coordination
    results = {}
    for phase in agents.keys():
        # Pass previous phase results as context
        context = {"previous_phases": results, "requirements": project_requirements}
        results[phase] = await agents[phase].execute_phase(context)
        
        # Update shared memory for coordination
        await self.coordinator.update_shared_memory(
            session.id, 
            f"phase_{phase}_results", 
            results[phase]
        )
    
    return SPARCWorkflowResponse(
        session_id=session.id,
        phase_results=results,
        integration_report=self._generate_integration_report(results),
        quality_metrics=self._calculate_quality_metrics(results)
    )
```

### Claude Flow Swarm Integration

#### Advanced Swarm Patterns

```python
class SwarmPatterns:
    """
    Advanced patterns for Claude Flow swarm coordination.
    """
    
    async def hierarchical_problem_solving(
        self,
        complex_problem: str,
        domain_context: Dict[str, Any]
    ) -> HierarchicalResponse:
        """
        Use hierarchical coordination for complex problem decomposition.
        
        Structure:
        - Master Agent: Problem analysis and coordination
        - Specialist Agents: Domain-specific analysis
        - Worker Agents: Implementation and execution
        """
        
        # Initialize hierarchical topology
        await self.swarm.init_topology("hierarchical", levels=3)
        
        # Level 1: Master coordination
        master = await self.swarm.spawn_agent(
            "archon-master",
            "Analyze and decompose complex problem",
            level=1
        )
        
        # Level 2: Domain specialists  
        specialists = []
        domains = self._identify_domains(complex_problem)
        for domain in domains:
            specialist = await self.swarm.spawn_agent(
                f"archon-{domain}-specialist", 
                f"Provide {domain} expertise",
                level=2,
                reports_to=master.id
            )
            specialists.append(specialist)
        
        # Level 3: Worker agents
        workers = []
        for specialist in specialists:
            tasks = await specialist.decompose_tasks()
            for task in tasks:
                worker = await self.swarm.spawn_agent(
                    self._select_worker_type(task),
                    task.description,
                    level=3,
                    reports_to=specialist.id
                )
                workers.append(worker)
        
        # Execute coordinated workflow
        results = await self.swarm.execute_hierarchical_workflow()
        
        return HierarchicalResponse(
            problem_decomposition=master.analysis,
            specialist_insights=[s.results for s in specialists],
            implementation_results=[w.results for w in workers],
            synthesis=master.synthesize_results(results)
        )
    
    async def mesh_collaboration(
        self,
        collaborative_task: str,
        required_expertise: List[str]
    ) -> MeshCollaborationResponse:
        """
        Use mesh topology for peer-to-peer agent collaboration.
        
        Pattern:
        - All agents can communicate directly
        - Consensus building through peer review
        - Distributed decision making
        - Failure tolerance through redundancy
        """
        
        await self.swarm.init_topology("mesh")
        
        # Spawn peer agents with equivalent standing
        peers = []
        for expertise in required_expertise:
            peer = await self.swarm.spawn_agent(
                f"archon-{expertise}-expert",
                f"Contribute {expertise} perspective to: {collaborative_task}",
                peer_level=True
            )
            peers.append(peer)
        
        # Enable direct peer communication
        await self.swarm.enable_peer_communication(peers)
        
        # Execute collaborative workflow
        collaboration_results = []
        
        # Phase 1: Independent analysis
        for peer in peers:
            analysis = await peer.independent_analysis(collaborative_task)
            await self.swarm.broadcast_to_peers(peer.id, analysis)
        
        # Phase 2: Peer review and refinement
        for peer in peers:
            peer_inputs = await self.swarm.get_peer_inputs(peer.id)
            refined_analysis = await peer.refine_with_peer_input(peer_inputs)
            collaboration_results.append(refined_analysis)
        
        # Phase 3: Consensus building
        consensus = await self.swarm.build_consensus(collaboration_results)
        
        return MeshCollaborationResponse(
            individual_contributions=collaboration_results,
            peer_interactions=self.swarm.get_interaction_history(),
            consensus_result=consensus,
            quality_metrics=self._assess_collaboration_quality(consensus)
        )
```

### Real-time Communication Patterns

#### WebSocket Integration

```python
class RealtimeCoordination:
    """
    Real-time coordination using WebSocket communication.
    """
    
    def __init__(self, socketio_client):
        self.sio = socketio_client
        
    async def setup_realtime_agent_coordination(
        self,
        session_id: str,
        participants: List[str]
    ):
        """
        Setup real-time coordination channels for agent communication.
        """
        
        # Create coordination room
        await self.sio.emit('create_coordination_room', {
            'session_id': session_id,
            'participants': participants,
            'room_type': 'agent_coordination'
        })
        
        # Setup event handlers for agent communication
        @self.sio.on('agent_message')
        async def handle_agent_message(data):
            """Handle messages between coordinated agents."""
            await self.route_agent_message(
                from_agent=data['from_agent'],
                to_agent=data['to_agent'],
                message=data['message'],
                session_id=data['session_id']
            )
        
        @self.sio.on('coordination_update') 
        async def handle_coordination_update(data):
            """Handle coordination state updates."""
            await self.update_coordination_state(
                session_id=data['session_id'],
                state_update=data['state_update']
            )
    
    async def broadcast_agent_progress(
        self,
        agent_id: str,
        progress_update: Dict[str, Any],
        session_id: str
    ):
        """
        Broadcast agent progress to coordination participants.
        """
        await self.sio.emit('agent_progress', {
            'agent_id': agent_id,
            'progress': progress_update,
            'session_id': session_id,
            'timestamp': datetime.utcnow().isoformat()
        }, room=f"coordination_{session_id}")
```

---

## Extension Points

### Custom Agent Development

#### Agent Base Class Extension

```python
from src.agents.base_agent import BaseAgent, BaseAgentOutput

class CustomSpecialistAgent(BaseAgent):
    """
    Base class for developing custom specialist agents.
    """
    
    def __init__(self, specialization: str, capabilities: List[str]):
        super().__init__()
        self.specialization = specialization
        self.capabilities = capabilities
        
    async def can_handle(self, query: str, context: Dict[str, Any]) -> bool:
        """
        Determine if this agent can handle the given query.
        
        Args:
            query: Input query to evaluate
            context: Query context and metadata
            
        Returns:
            Boolean indicating capability to handle query
            
        Implementation should analyze query against agent capabilities.
        """
        raise NotImplementedError
    
    async def process(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> BaseAgentOutput:
        """
        Process query using agent specialization.
        
        Args:
            query: Query to process
            context: Processing context
            
        Returns:
            BaseAgentOutput with results and metadata
        """
        raise NotImplementedError
        
    async def get_capabilities(self) -> Dict[str, Any]:
        """
        Return detailed capability information for routing decisions.
        
        Returns:
            Dictionary with capability metadata
        """
        return {
            "specialization": self.specialization,
            "capabilities": self.capabilities,
            "performance_characteristics": self._get_performance_metrics(),
            "supported_contexts": self._get_supported_contexts()
        }
```

#### Example: Security Specialist Agent

```python
class SecuritySpecialistAgent(CustomSpecialistAgent):
    """
    Security-focused specialist agent for code analysis and recommendations.
    """
    
    def __init__(self):
        super().__init__(
            specialization="security",
            capabilities=[
                "vulnerability_detection",
                "secure_coding_patterns", 
                "authentication_design",
                "authorization_implementation",
                "security_best_practices"
            ]
        )
        
    async def can_handle(self, query: str, context: Dict[str, Any]) -> bool:
        """Check if query relates to security concerns."""
        security_keywords = [
            "security", "authentication", "authorization", "vulnerability",
            "encryption", "oauth", "jwt", "ssl", "tls", "xss", "sql injection"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in security_keywords)
    
    async def process(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> BaseAgentOutput:
        """Process security-related query."""
        
        # Analyze security requirements
        security_analysis = await self._analyze_security_requirements(query)
        
        # Search for security patterns in knowledge base
        security_patterns = await self.mcp_client.search_code_examples(
            query + " security implementation patterns",
            framework="security"
        )
        
        # Generate security recommendations
        recommendations = await self._generate_security_recommendations(
            security_analysis, 
            security_patterns
        )
        
        return BaseAgentOutput(
            content=recommendations["implementation_guide"],
            metadata={
                "security_level": recommendations["security_level"],
                "vulnerabilities_addressed": recommendations["vulnerabilities"],
                "compliance_standards": recommendations["standards"]
            },
            sources=security_patterns.sources,
            confidence=recommendations["confidence"]
        )
    
    async def _analyze_security_requirements(
        self, 
        query: str
    ) -> Dict[str, Any]:
        """Analyze query for security requirements and threats."""
        # Implementation specific to security analysis
        pass
    
    async def _generate_security_recommendations(
        self,
        analysis: Dict[str, Any],
        patterns: Any
    ) -> Dict[str, Any]:
        """Generate security implementation recommendations."""
        # Implementation specific to security recommendations
        pass
```

### Custom RAG Strategy Development

#### RAG Strategy Interface

```python
from abc import ABC, abstractmethod

class RAGStrategy(ABC):
    """
    Base class for developing custom RAG strategies.
    """
    
    @abstractmethod
    async def execute_search(
        self,
        query: str,
        knowledge_base: Any,
        context: Dict[str, Any],
        max_results: int = 5
    ) -> RAGSearchResults:
        """
        Execute search using this RAG strategy.
        
        Args:
            query: Search query
            knowledge_base: Knowledge base interface
            context: Search context and parameters
            max_results: Maximum number of results
            
        Returns:
            RAGSearchResults with ranked results
        """
        pass
    
    @abstractmethod
    def get_strategy_metadata(self) -> Dict[str, Any]:
        """
        Return metadata about this RAG strategy.
        
        Returns:
            Dictionary with strategy information
        """
        pass
    
    def estimate_performance_improvement(
        self,
        baseline_accuracy: float
    ) -> float:
        """
        Estimate performance improvement over baseline.
        
        Args:
            baseline_accuracy: Current accuracy baseline
            
        Returns:
            Estimated accuracy improvement (0.0-1.0)
        """
        return 0.0  # Override in implementations
```

#### Example: Domain-Aware RAG Strategy

```python
class DomainAwareRAGStrategy(RAGStrategy):
    """
    RAG strategy that adapts search based on query domain classification.
    """
    
    def __init__(self, domain_classifier, domain_configs):
        self.domain_classifier = domain_classifier
        self.domain_configs = domain_configs
        
    async def execute_search(
        self,
        query: str,
        knowledge_base: Any,
        context: Dict[str, Any],
        max_results: int = 5
    ) -> RAGSearchResults:
        """Execute domain-aware search."""
        
        # Classify query domain
        domain = await self.domain_classifier.classify(query, context)
        
        # Get domain-specific configuration
        domain_config = self.domain_configs.get(
            domain, 
            self.domain_configs["default"]
        )
        
        # Execute search with domain-specific parameters
        search_params = {
            "embedding_model": domain_config["embedding_model"],
            "chunk_strategy": domain_config["chunk_strategy"],
            "similarity_threshold": domain_config["similarity_threshold"],
            "reranking_model": domain_config["reranking_model"]
        }
        
        # Perform domain-adapted search
        results = await knowledge_base.search(
            query=query,
            max_results=max_results,
            **search_params
        )
        
        # Apply domain-specific post-processing
        processed_results = await self._post_process_results(
            results, 
            domain, 
            domain_config
        )
        
        return RAGSearchResults(
            results=processed_results,
            strategy_used="domain_aware",
            domain_detected=domain,
            metadata={
                "domain_confidence": await self.domain_classifier.get_confidence(),
                "search_params": search_params,
                "post_processing_applied": True
            }
        )
    
    def get_strategy_metadata(self) -> Dict[str, Any]:
        """Return domain-aware strategy metadata."""
        return {
            "name": "domain_aware_rag",
            "description": "Adapts search parameters based on query domain classification",
            "supported_domains": list(self.domain_configs.keys()),
            "performance_characteristics": {
                "accuracy_improvement": "15-25% over baseline",
                "latency_overhead": "50-100ms for domain classification",
                "memory_overhead": "Minimal"
            }
        }
    
    def estimate_performance_improvement(
        self,
        baseline_accuracy: float
    ) -> float:
        """Estimate 20% average improvement with domain adaptation."""
        return min(0.20, 1.0 - baseline_accuracy)  # Cap at perfect accuracy
```

### Custom MCP Tool Development

#### MCP Tool Interface

```python
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

class CustomMCPTool:
    """
    Base class for developing custom MCP tools.
    """
    
    def __init__(self, tool_name: str, description: str):
        self.tool_name = tool_name
        self.description = description
        
    def get_tool_schema(self) -> Dict[str, Any]:
        """
        Return JSON schema for this MCP tool.
        
        Returns:
            Tool schema for MCP protocol registration
        """
        raise NotImplementedError
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute the MCP tool with given parameters.
        
        Args:
            parameters: Tool execution parameters
            context: Optional execution context
            
        Returns:
            Tool execution results
        """
        raise NotImplementedError
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """
        Validate parameters against tool schema.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Boolean indicating parameter validity
        """
        # Default validation using schema
        # Override for custom validation logic
        return True
```

#### Example: Code Analysis Tool

```python
class CodeAnalysisTool(CustomMCPTool):
    """
    MCP tool for analyzing code quality and patterns.
    """
    
    def __init__(self):
        super().__init__(
            tool_name="analyze_code_quality",
            description="Analyze code quality, patterns, and potential improvements"
        )
        
    def get_tool_schema(self) -> Dict[str, Any]:
        """Return schema for code analysis tool."""
        return {
            "type": "function",
            "function": {
                "name": self.tool_name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Code to analyze"
                        },
                        "language": {
                            "type": "string", 
                            "description": "Programming language (python, javascript, etc.)",
                            "enum": ["python", "javascript", "typescript", "java", "cpp"]
                        },
                        "analysis_types": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Types of analysis to perform",
                            "enum": [
                                "complexity", "style", "security", 
                                "performance", "maintainability"
                            ]
                        },
                        "project_context": {
                            "type": "object",
                            "description": "Optional project context for analysis"
                        }
                    },
                    "required": ["code", "language"]
                }
            }
        }
    
    async def execute(
        self,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute code analysis."""
        
        code = parameters["code"]
        language = parameters["language"]
        analysis_types = parameters.get("analysis_types", ["complexity", "style"])
        project_context = parameters.get("project_context", {})
        
        results = {}
        
        # Perform requested analyses
        for analysis_type in analysis_types:
            if analysis_type == "complexity":
                results["complexity"] = await self._analyze_complexity(code, language)
            elif analysis_type == "style":
                results["style"] = await self._analyze_style(code, language)
            elif analysis_type == "security":
                results["security"] = await self._analyze_security(code, language)
            elif analysis_type == "performance":
                results["performance"] = await self._analyze_performance(code, language)
            elif analysis_type == "maintainability":
                results["maintainability"] = await self._analyze_maintainability(
                    code, language, project_context
                )
        
        # Generate overall assessment
        results["overall_assessment"] = await self._generate_assessment(results)
        
        # Provide improvement recommendations
        results["recommendations"] = await self._generate_recommendations(
            results, project_context
        )
        
        return {
            "analysis_results": results,
            "metadata": {
                "code_lines": len(code.split('\n')),
                "language": language,
                "analysis_types": analysis_types,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    async def _analyze_complexity(
        self, 
        code: str, 
        language: str
    ) -> Dict[str, Any]:
        """Analyze code complexity metrics."""
        # Implementation for complexity analysis
        # - Cyclomatic complexity
        # - Lines of code
        # - Function/method complexity
        pass
    
    async def _analyze_style(
        self, 
        code: str, 
        language: str
    ) -> Dict[str, Any]:
        """Analyze code style and conventions."""
        # Implementation for style analysis
        # - Naming conventions
        # - Code formatting
        # - Best practices adherence
        pass
```

---

## Performance Tuning Guidelines

### System-Level Optimization

#### Database Performance Tuning

```python
# Database connection optimization
DATABASE_CONFIG = {
    "pool_size": 20,                    # Connection pool size
    "max_overflow": 30,                 # Maximum overflow connections
    "pool_pre_ping": True,              # Validate connections
    "pool_recycle": 3600,               # Recycle connections every hour
    
    # Vector search optimization
    "vector_index_type": "ivfflat",     # Index type for pgvector
    "vector_index_lists": 1000,         # Number of lists in IVF index
    "vector_probes": 10,                # Number of probes for search
    
    # Query optimization
    "statement_timeout": "30s",         # Query timeout
    "lock_timeout": "10s",              # Lock timeout
    "work_mem": "256MB",                # Memory for query operations
    "shared_buffers": "1GB",            # Shared buffer cache
}

# Vector index optimization
async def optimize_vector_indexes():
    """Optimize vector indexes for better search performance."""
    
    # Create optimized vector index
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_embedding_ivfflat 
        ON documents 
        USING ivfflat (embedding vector_cosine_ops) 
        WITH (lists = 1000);
    """)
    
    # Analyze table statistics
    await db.execute("ANALYZE documents;")
    
    # Set optimal search parameters
    await db.execute("SET ivfflat.probes = 10;")
```

#### Memory Management

```python
class PerformanceOptimizer:
    """
    System performance optimization and monitoring.
    """
    
    def __init__(self):
        self.memory_cache = LRUCache(maxsize=10000)
        self.query_cache = TTLCache(maxsize=5000, ttl=300)
        self.embedding_cache = TTLCache(maxsize=50000, ttl=3600)
        
    async def optimize_memory_usage(self):
        """Optimize system memory usage patterns."""
        
        # Configure garbage collection
        import gc
        gc.set_threshold(700, 10, 10)  # More aggressive GC
        
        # Setup memory monitoring
        import psutil
        process = psutil.Process()
        
        memory_info = process.memory_info()
        if memory_info.rss > 2 * 1024 * 1024 * 1024:  # > 2GB
            # Trigger cache cleanup
            await self._cleanup_caches()
            
            # Force garbage collection
            gc.collect()
            
        # Monitor memory growth rate
        await self._monitor_memory_growth()
    
    async def _cleanup_caches(self):
        """Clean up various system caches."""
        
        # Clear least recently used items
        if len(self.memory_cache) > self.memory_cache.maxsize * 0.8:
            # Clear 20% of cache
            items_to_clear = int(len(self.memory_cache) * 0.2)
            for _ in range(items_to_clear):
                self.memory_cache.popitem(last=False)
        
        # Clear expired query cache items
        self.query_cache.expire()
        
        # Clear old embedding cache items
        self.embedding_cache.expire()
    
    async def optimize_query_performance(self, query: str) -> Dict[str, Any]:
        """Optimize individual query performance."""
        
        # Check query cache first
        cache_key = self._generate_cache_key(query)
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Optimize query execution
        start_time = time.time()
        
        # Use connection pooling for database queries
        async with self.db_pool.acquire() as conn:
            # Execute with optimized parameters
            result = await self._execute_optimized_query(conn, query)
        
        execution_time = time.time() - start_time
        
        # Cache result if execution was successful and fast
        if execution_time < 1.0:  # Cache queries under 1 second
            self.query_cache[cache_key] = result
        
        return result
```

#### Concurrent Processing Optimization

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Callable, Any

class ConcurrentProcessor:
    """
    Optimized concurrent processing for high-throughput operations.
    """
    
    def __init__(self, max_workers: int = 10):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)
        
    async def process_batch_concurrent(
        self,
        items: List[Any],
        processor: Callable,
        batch_size: int = 50
    ) -> List[Any]:
        """
        Process items concurrently in batches for optimal performance.
        """
        
        results = []
        
        # Process in batches to manage memory and connections
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Create concurrent tasks for batch
            tasks = []
            for item in batch:
                task = self._process_with_semaphore(processor, item)
                tasks.append(task)
            
            # Execute batch concurrently
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions and collect results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error(f"Processing error: {result}")
                    continue
                results.append(result)
                
            # Brief pause between batches to prevent overwhelming
            if i + batch_size < len(items):
                await asyncio.sleep(0.1)
        
        return results
    
    async def _process_with_semaphore(
        self,
        processor: Callable,
        item: Any
    ) -> Any:
        """Process item with semaphore-controlled concurrency."""
        
        async with self.semaphore:
            # Use thread pool for CPU-intensive operations
            if self._is_cpu_intensive(processor):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.executor, 
                    processor, 
                    item
                )
            else:
                # Direct async execution for I/O operations
                return await processor(item)
```

### Agent-Specific Optimizations

#### Master Agent Performance Tuning

```python
class MasterAgentOptimizations:
    """
    Performance optimizations specific to Master Agent operations.
    """
    
    def __init__(self):
        self.strategy_cache = {}
        self.capability_cache = TTLCache(maxsize=1000, ttl=600)
        self.agent_performance_metrics = defaultdict(list)
        
    async def optimize_strategy_selection(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> ProcessingStrategy:
        """
        Optimize strategy selection based on performance history.
        """
        
        # Generate strategy cache key
        strategy_key = self._generate_strategy_key(query, context)
        
        if strategy_key in self.strategy_cache:
            cached_strategy = self.strategy_cache[strategy_key]
            
            # Validate cached strategy is still optimal
            if await self._is_strategy_still_optimal(cached_strategy, context):
                return cached_strategy
        
        # Analyze query characteristics
        query_analysis = await self._analyze_query_characteristics(query)
        
        # Select optimal strategy based on analysis and performance history
        optimal_strategy = await self._select_optimal_strategy(
            query_analysis,
            context,
            self.agent_performance_metrics
        )
        
        # Cache strategy decision
        self.strategy_cache[strategy_key] = optimal_strategy
        
        return optimal_strategy
    
    async def optimize_agent_routing(
        self,
        query: str,
        available_agents: List[str]
    ) -> str:
        """
        Optimize agent routing based on performance characteristics.
        """
        
        # Check capability cache
        cache_key = f"routing_{hash(query)}_{hash(tuple(available_agents))}"
        if cache_key in self.capability_cache:
            return self.capability_cache[cache_key]
        
        # Analyze query requirements
        requirements = await self._extract_query_requirements(query)
        
        # Score agents based on capability match and performance
        agent_scores = {}
        for agent in available_agents:
            capability_score = await self._score_agent_capability(
                agent, 
                requirements
            )
            performance_score = self._get_agent_performance_score(agent)
            
            # Combined score with capability weighted higher
            combined_score = (capability_score * 0.7) + (performance_score * 0.3)
            agent_scores[agent] = combined_score
        
        # Select highest scoring agent
        optimal_agent = max(agent_scores, key=agent_scores.get)
        
        # Cache routing decision
        self.capability_cache[cache_key] = optimal_agent
        
        return optimal_agent
    
    def update_performance_metrics(
        self,
        agent_type: str,
        processing_time: float,
        success: bool,
        query_complexity: str
    ):
        """
        Update performance metrics for continuous optimization.
        """
        
        metric = {
            "timestamp": datetime.utcnow(),
            "processing_time": processing_time,
            "success": success,
            "query_complexity": query_complexity
        }
        
        self.agent_performance_metrics[agent_type].append(metric)
        
        # Keep only recent metrics (last 1000 entries)
        if len(self.agent_performance_metrics[agent_type]) > 1000:
            self.agent_performance_metrics[agent_type] = \
                self.agent_performance_metrics[agent_type][-1000:]
    
    def _get_agent_performance_score(self, agent_type: str) -> float:
        """
        Calculate performance score for agent based on historical data.
        """
        
        metrics = self.agent_performance_metrics.get(agent_type, [])
        if not metrics:
            return 0.5  # Neutral score for unknown agents
        
        # Recent metrics are weighted more heavily
        recent_metrics = [m for m in metrics 
                         if (datetime.utcnow() - m["timestamp"]).seconds < 3600]
        
        if not recent_metrics:
            recent_metrics = metrics[-50:]  # Last 50 if no recent data
        
        # Calculate success rate
        success_rate = sum(1 for m in recent_metrics if m["success"]) / len(recent_metrics)
        
        # Calculate average response time (normalized)
        avg_time = sum(m["processing_time"] for m in recent_metrics) / len(recent_metrics)
        time_score = max(0, 1 - (avg_time / 10.0))  # Normalize to 0-1, 10s = 0 score
        
        # Combined performance score
        return (success_rate * 0.6) + (time_score * 0.4)
```

### RAG Performance Optimization

#### Embedding and Vector Search Tuning

```python
class RAGPerformanceOptimizer:
    """
    Specialized performance optimization for RAG operations.
    """
    
    def __init__(self):
        self.embedding_batch_size = 100
        self.search_cache = TTLCache(maxsize=10000, ttl=300)
        self.embedding_model_cache = {}
        
    async def optimize_embedding_generation(
        self,
        texts: List[str],
        model: str = "text-embedding-3-small"
    ) -> List[List[float]]:
        """
        Optimize embedding generation with batching and caching.
        """
        
        # Check cache for existing embeddings
        cached_embeddings = {}
        texts_to_embed = []
        
        for i, text in enumerate(texts):
            cache_key = f"emb_{model}_{hash(text)}"
            if cache_key in self.embedding_model_cache:
                cached_embeddings[i] = self.embedding_model_cache[cache_key]
            else:
                texts_to_embed.append((i, text))
        
        # Generate embeddings for uncached texts in batches
        new_embeddings = {}
        if texts_to_embed:
            for i in range(0, len(texts_to_embed), self.embedding_batch_size):
                batch = texts_to_embed[i:i + self.embedding_batch_size]
                batch_texts = [text for _, text in batch]
                
                # Generate batch embeddings
                batch_embeddings = await self._generate_embeddings_batch(
                    batch_texts, 
                    model
                )
                
                # Cache and store results
                for j, (original_index, text) in enumerate(batch):
                    embedding = batch_embeddings[j]
                    cache_key = f"emb_{model}_{hash(text)}"
                    
                    self.embedding_model_cache[cache_key] = embedding
                    new_embeddings[original_index] = embedding
        
        # Combine cached and new embeddings in original order
        result_embeddings = []
        for i in range(len(texts)):
            if i in cached_embeddings:
                result_embeddings.append(cached_embeddings[i])
            else:
                result_embeddings.append(new_embeddings[i])
        
        return result_embeddings
    
    async def optimize_vector_search(
        self,
        query_embedding: List[float],
        collection_name: str,
        max_results: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Optimize vector search with caching and index optimization.
        """
        
        # Generate search cache key
        cache_key = f"search_{collection_name}_{hash(str(query_embedding))}_{max_results}_{similarity_threshold}"
        
        if cache_key in self.search_cache:
            return self.search_cache[cache_key]
        
        # Optimize search query based on collection characteristics
        optimized_params = await self._get_optimized_search_params(
            collection_name,
            max_results
        )
        
        # Execute optimized vector search
        search_results = await self._execute_optimized_search(
            query_embedding=query_embedding,
            collection_name=collection_name,
            max_results=max_results,
            similarity_threshold=similarity_threshold,
            **optimized_params
        )
        
        # Cache results
        self.search_cache[cache_key] = search_results
        
        return search_results
    
    async def _get_optimized_search_params(
        self,
        collection_name: str,
        max_results: int
    ) -> Dict[str, Any]:
        """
        Get optimized search parameters based on collection characteristics.
        """
        
        # Get collection statistics
        stats = await self._get_collection_stats(collection_name)
        
        params = {
            "ef_search": min(max_results * 4, 200),  # Adaptive ef_search
            "probes": min(stats.get("size", 1000) // 1000 + 1, 20)  # Adaptive probes
        }
        
        # Adjust based on collection size
        if stats.get("size", 0) > 100000:
            params["ef_search"] = max(params["ef_search"], 100)
            params["probes"] = max(params["probes"], 10)
        
        return params
```

This developer documentation provides comprehensive guidance for extending and optimizing the Master Agent System. The next sections will cover testing frameworks and operational guidelines.