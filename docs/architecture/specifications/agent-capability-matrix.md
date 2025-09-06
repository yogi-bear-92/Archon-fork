# Agent Capability Matrix Specification

## Overview

The Agent Capability Matrix maintains comprehensive mappings of all 64 Claude Flow agents, their capabilities, performance metrics, and current availability for intelligent routing decisions.

## Agent Registry Structure

### Core Development Agents (5)

```json
{
  "coder": {
    "agent_id": "coder",
    "category": "core_development",
    "capabilities": [
      "code_generation", "debugging", "refactoring", "code_review",
      "algorithm_implementation", "optimization", "documentation"
    ],
    "domains": [
      "python", "javascript", "typescript", "java", "cpp", "rust", "go",
      "react", "vue", "angular", "node", "django", "flask", "spring"
    ],
    "complexity_handling": ["simple", "medium", "complex"],
    "coordination_protocols": ["hierarchical", "mesh", "adaptive"],
    "performance_metrics": {
      "success_rate": 0.94,
      "avg_response_time": "2.3s",
      "code_quality_score": 0.91,
      "bug_introduction_rate": 0.02
    },
    "current_status": {
      "availability": "available",
      "current_load": 0.3,
      "queue_depth": 2,
      "last_task_completion": "2025-01-05T13:15:22Z"
    },
    "integration_points": ["git", "ide", "testing", "ci_cd", "documentation"],
    "specializations": ["full_stack", "backend", "frontend", "mobile"],
    "knowledge_domains": ["software_engineering", "design_patterns", "architecture"]
  },
  
  "reviewer": {
    "agent_id": "reviewer",
    "category": "core_development", 
    "capabilities": [
      "code_review", "security_analysis", "performance_review",
      "architecture_assessment", "best_practices_validation"
    ],
    "domains": [
      "all_languages", "security", "performance", "accessibility",
      "code_quality", "compliance"
    ],
    "complexity_handling": ["medium", "complex", "enterprise"],
    "coordination_protocols": ["hierarchical", "mesh"],
    "performance_metrics": {
      "success_rate": 0.96,
      "avg_response_time": "1.8s", 
      "issue_detection_rate": 0.89,
      "false_positive_rate": 0.05
    },
    "current_status": {
      "availability": "available",
      "current_load": 0.2,
      "queue_depth": 1,
      "last_task_completion": "2025-01-05T13:10:15Z"
    },
    "integration_points": ["pr_systems", "code_quality_tools", "security_scanners"],
    "specializations": ["security_review", "performance_audit", "compliance_check"],
    "knowledge_domains": ["security", "performance", "best_practices", "standards"]
  }
}
```

### Swarm Coordination Agents (5)

```json
{
  "hierarchical-coordinator": {
    "agent_id": "hierarchical-coordinator",
    "category": "swarm_coordination",
    "capabilities": [
      "task_delegation", "resource_allocation", "priority_management",
      "conflict_resolution", "load_balancing", "performance_optimization"
    ],
    "domains": ["orchestration", "resource_management", "optimization"],
    "complexity_handling": ["medium", "complex", "enterprise"],
    "coordination_protocols": ["hierarchical"],
    "performance_metrics": {
      "success_rate": 0.97,
      "avg_response_time": "0.8s",
      "coordination_efficiency": 0.93,
      "resource_utilization": 0.87
    },
    "current_status": {
      "availability": "available",
      "current_load": 0.4,
      "active_swarms": 3,
      "managed_agents": 12
    },
    "integration_points": ["swarm_management", "metrics", "monitoring"],
    "specializations": ["large_scale_coordination", "enterprise_workflows"],
    "knowledge_domains": ["distributed_systems", "coordination_theory", "optimization"]
  }
}
```

### Performance & Optimization Agents (4)

```json
{
  "perf-analyzer": {
    "agent_id": "perf-analyzer",
    "category": "performance_optimization",
    "capabilities": [
      "performance_profiling", "bottleneck_identification", 
      "optimization_recommendations", "benchmark_analysis",
      "resource_monitoring", "scaling_analysis"
    ],
    "domains": [
      "application_performance", "database_optimization", "network_analysis",
      "memory_management", "cpu_optimization", "caching"
    ],
    "complexity_handling": ["medium", "complex", "enterprise"],
    "coordination_protocols": ["hierarchical", "mesh"],
    "performance_metrics": {
      "success_rate": 0.92,
      "avg_response_time": "3.2s",
      "optimization_impact": 0.78,
      "accuracy_score": 0.91
    },
    "current_status": {
      "availability": "available",
      "current_load": 0.1,
      "active_analyses": 1,
      "queue_depth": 0
    },
    "integration_points": ["monitoring_tools", "profilers", "metrics_systems"],
    "specializations": ["web_performance", "database_optimization", "cloud_scaling"],
    "knowledge_domains": ["performance_engineering", "systems_architecture", "optimization_theory"]
  }
}
```

## Agent Selection Algorithm

### Multi-Criteria Scoring

```python
def calculate_agent_score(agent, query_analysis, context):
    """
    Calculate comprehensive agent selection score based on multiple criteria
    """
    
    # 1. Capability Match Score (40% weight)
    capability_score = calculate_capability_match(
        agent.capabilities, 
        query_analysis.required_capabilities
    )
    
    # 2. Domain Expertise Score (25% weight)  
    domain_score = calculate_domain_alignment(
        agent.domains,
        query_analysis.technical_domains
    )
    
    # 3. Availability Score (15% weight)
    availability_score = calculate_availability_factor(
        agent.current_status.availability,
        agent.current_status.current_load,
        agent.current_status.queue_depth
    )
    
    # 4. Performance History Score (10% weight)
    performance_score = calculate_performance_rating(
        agent.performance_metrics.success_rate,
        agent.performance_metrics.avg_response_time,
        query_analysis.complexity_level
    )
    
    # 5. Coordination Compatibility Score (10% weight)
    coordination_score = calculate_coordination_compatibility(
        agent.coordination_protocols,
        context.preferred_topology
    )
    
    # Weighted final score
    total_score = (
        capability_score * 0.40 +
        domain_score * 0.25 +
        availability_score * 0.15 +
        performance_score * 0.10 +
        coordination_score * 0.10
    )
    
    return total_score

def calculate_capability_match(agent_caps, required_caps):
    """Calculate capability overlap percentage"""
    if not required_caps:
        return 0.5  # Neutral score if no specific capabilities required
    
    matches = len(set(agent_caps) & set(required_caps))
    return min(matches / len(required_caps), 1.0)

def calculate_domain_alignment(agent_domains, query_domains):
    """Calculate domain expertise alignment"""
    if not query_domains:
        return 0.7  # Default score for general queries
    
    if "all_languages" in agent_domains:
        return 0.9  # Universal agents get high score
    
    overlap = len(set(agent_domains) & set(query_domains))
    return min(overlap / len(query_domains), 1.0)

def calculate_availability_factor(availability, load, queue_depth):
    """Calculate availability-based score"""
    if availability != "available":
        return 0.1  # Low score for unavailable agents
    
    load_penalty = load * 0.5  # Penalize high load
    queue_penalty = min(queue_depth * 0.1, 0.4)  # Penalize queue depth
    
    return max(1.0 - load_penalty - queue_penalty, 0.1)

def calculate_performance_rating(success_rate, avg_response_time, complexity):
    """Calculate performance-based score"""
    # Success rate component (70% weight)
    success_component = success_rate * 0.7
    
    # Response time component (30% weight)
    # Normalize response time (assume 5s is maximum acceptable)
    time_score = max(1.0 - (avg_response_time / 5.0), 0.1)
    time_component = time_score * 0.3
    
    # Complexity adjustment
    if complexity == "simple":
        return min(success_component + time_component + 0.1, 1.0)
    elif complexity == "complex":
        return (success_component + time_component) * 0.9
    
    return success_component + time_component
```

## Dynamic Agent Routing

### Routing Decision Tree

```python
class AgentRouter:
    def __init__(self, agent_registry, performance_tracker):
        self.registry = agent_registry
        self.performance_tracker = performance_tracker
        self.routing_history = []
    
    async def route_query(self, query_analysis, context, max_agents=3):
        """
        Route query to optimal agent(s) based on analysis and context
        """
        
        # Step 1: Filter available agents by basic requirements
        candidates = self.filter_basic_requirements(
            query_analysis.required_capabilities,
            query_analysis.technical_domains,
            query_analysis.complexity_level
        )
        
        # Step 2: Score all candidates
        scored_candidates = []
        for agent in candidates:
            score = calculate_agent_score(agent, query_analysis, context)
            scored_candidates.append((agent, score))
        
        # Step 3: Rank and select top agents
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        selected_agents = scored_candidates[:max_agents]
        
        # Step 4: Apply coordination strategy
        coordination_plan = self.create_coordination_plan(
            selected_agents, query_analysis, context
        )
        
        # Step 5: Record routing decision for learning
        self.record_routing_decision(query_analysis, selected_agents, coordination_plan)
        
        return coordination_plan
    
    def filter_basic_requirements(self, required_caps, domains, complexity):
        """Filter agents that meet basic requirements"""
        filtered = []
        
        for agent in self.registry.get_all_agents():
            # Check availability
            if agent.current_status.availability != "available":
                continue
            
            # Check complexity handling
            if complexity not in agent.complexity_handling:
                continue
            
            # Check capability overlap (at least 30% match required)
            if required_caps:
                cap_overlap = len(set(agent.capabilities) & set(required_caps))
                if cap_overlap / len(required_caps) < 0.3:
                    continue
            
            # Check domain expertise
            if domains and not any(domain in agent.domains for domain in domains):
                if "all_languages" not in agent.domains:
                    continue
            
            filtered.append(agent)
        
        return filtered
    
    def create_coordination_plan(self, selected_agents, query_analysis, context):
        """Create coordination plan for selected agents"""
        
        primary_agent = selected_agents[0][0]  # Highest scored agent
        support_agents = [agent for agent, score in selected_agents[1:]]
        
        # Determine coordination topology based on task complexity
        if query_analysis.complexity_level == "simple":
            topology = "single_agent"
        elif query_analysis.complexity_level == "medium":
            topology = "hierarchical" 
        else:
            topology = "mesh"  # Complex tasks benefit from peer-to-peer
        
        return {
            "primary_agent": primary_agent,
            "support_agents": support_agents,
            "coordination_topology": topology,
            "estimated_duration": self.estimate_task_duration(selected_agents, query_analysis),
            "fallback_agents": self.identify_fallback_agents(selected_agents),
            "monitoring_requirements": self.determine_monitoring_requirements(query_analysis)
        }
```

## Performance Tracking and Learning

### Adaptive Scoring

```python
class PerformanceTracker:
    def __init__(self):
        self.agent_performance_history = {}
        self.routing_feedback = []
        self.learning_model = None
    
    async def update_agent_performance(self, agent_id, task_result):
        """Update agent performance metrics based on task results"""
        
        if agent_id not in self.agent_performance_history:
            self.agent_performance_history[agent_id] = {
                "total_tasks": 0,
                "successful_tasks": 0,
                "total_response_time": 0,
                "quality_scores": [],
                "domain_performance": {}
            }
        
        history = self.agent_performance_history[agent_id]
        history["total_tasks"] += 1
        
        if task_result.success:
            history["successful_tasks"] += 1
        
        history["total_response_time"] += task_result.response_time
        history["quality_scores"].append(task_result.quality_score)
        
        # Update domain-specific performance
        for domain in task_result.domains:
            if domain not in history["domain_performance"]:
                history["domain_performance"][domain] = {"tasks": 0, "successes": 0}
            
            history["domain_performance"][domain]["tasks"] += 1
            if task_result.success:
                history["domain_performance"][domain]["successes"] += 1
        
        # Update agent registry with new metrics
        await self.update_agent_registry_metrics(agent_id, history)
    
    def calculate_dynamic_success_rate(self, agent_id, domain=None, time_window_days=30):
        """Calculate success rate with time decay and domain specificity"""
        
        history = self.agent_performance_history.get(agent_id, {})
        
        if domain and domain in history.get("domain_performance", {}):
            domain_history = history["domain_performance"][domain]
            return domain_history["successes"] / max(domain_history["tasks"], 1)
        
        return history.get("successful_tasks", 0) / max(history.get("total_tasks", 1), 1)
```

## Integration with Master Agent

### Registry Update Protocol

```python
class AgentRegistryManager:
    def __init__(self, archon_client, claude_flow_client):
        self.archon_client = archon_client
        self.claude_flow_client = claude_flow_client
        self.registry = {}
        self.update_interval = 30  # seconds
    
    async def initialize_registry(self):
        """Initialize agent registry from Claude Flow"""
        
        # Get available agents from Claude Flow
        swarm_status = await self.claude_flow_client.swarm_status()
        agent_list = await self.claude_flow_client.agent_list()
        
        for agent_info in agent_list:
            agent_id = agent_info["agent_id"]
            
            # Get detailed agent capabilities from knowledge base
            knowledge_query = f"agent capabilities {agent_id} specializations domains"
            rag_results = await self.archon_client.perform_rag_query(
                query=knowledge_query,
                source_domain="claude-flow",
                match_count=3
            )
            
            # Create comprehensive agent profile
            agent_profile = self.create_agent_profile(agent_info, rag_results)
            self.registry[agent_id] = agent_profile
    
    async def update_agent_status(self):
        """Periodically update agent status and metrics"""
        
        agent_metrics = await self.claude_flow_client.agent_metrics()
        
        for agent_id, metrics in agent_metrics.items():
            if agent_id in self.registry:
                self.registry[agent_id]["current_status"].update({
                    "current_load": metrics.get("load", 0),
                    "queue_depth": metrics.get("queue_depth", 0),
                    "availability": metrics.get("status", "unknown"),
                    "last_updated": datetime.utcnow().isoformat()
                })
                
                # Update performance metrics
                self.registry[agent_id]["performance_metrics"].update({
                    "success_rate": metrics.get("success_rate", 0),
                    "avg_response_time": metrics.get("avg_response_time", 0)
                })
    
    def create_agent_profile(self, agent_info, rag_results):
        """Create comprehensive agent profile from multiple sources"""
        
        # Extract capabilities from RAG results
        capabilities = self.extract_capabilities_from_rag(rag_results)
        domains = self.extract_domains_from_rag(rag_results)
        specializations = self.extract_specializations_from_rag(rag_results)
        
        return {
            "agent_id": agent_info["agent_id"],
            "category": agent_info.get("category", "general"),
            "capabilities": capabilities,
            "domains": domains,
            "specializations": specializations,
            "complexity_handling": agent_info.get("complexity_handling", ["simple", "medium"]),
            "coordination_protocols": agent_info.get("coordination_protocols", ["hierarchical"]),
            "performance_metrics": {
                "success_rate": 0.85,  # Default initial value
                "avg_response_time": "2.0s",
                "quality_score": 0.8
            },
            "current_status": {
                "availability": "available",
                "current_load": 0.0,
                "queue_depth": 0,
                "last_task_completion": None
            },
            "integration_points": agent_info.get("integration_points", []),
            "knowledge_domains": self.extract_knowledge_domains(rag_results),
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat()
        }
```

This Agent Capability Matrix provides the foundation for intelligent agent routing, enabling the Master Agent to make optimal decisions based on comprehensive agent profiles, real-time performance metrics, and dynamic capability matching.