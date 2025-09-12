"""
Agent Capability Matrix for intelligent routing and orchestration.

This module manages the capability matrix that maps agent types to their strengths,
domains of expertise, and performance characteristics for intelligent query routing.
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from enum import Enum


class QueryType(str, Enum):
    """Types of queries that can be processed by the claude flow expert agent."""
    
    RESEARCH = "research"
    CODING = "coding"
    ANALYSIS = "analysis"
    COORDINATION = "coordination"
    TASK_MANAGEMENT = "task_management"
    KNOWLEDGE = "knowledge"
    GENERAL = "general"

logger = logging.getLogger(__name__)


class AgentCapability:
    """Represents a single agent's capabilities and characteristics."""
    
    def __init__(
        self,
        agent_type: str,
        domains: List[str],
        strengths: List[str],
        performance_score: float = 0.8,
        specializations: Optional[List[str]] = None,
        coordination_compatible: bool = True,
        resource_requirements: Optional[Dict[str, Any]] = None
    ):
        self.agent_type = agent_type
        self.domains = domains
        self.strengths = strengths
        self.performance_score = performance_score
        self.specializations = specializations or []
        self.coordination_compatible = coordination_compatible
        self.resource_requirements = resource_requirements or {}
        self.success_rate = 0.85  # Default success rate
        self.average_response_time = 30.0  # Default 30 seconds
        self.last_updated = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert capability to dictionary representation."""
        return {
            "agent_type": self.agent_type,
            "domains": self.domains,
            "strengths": self.strengths,
            "performance_score": self.performance_score,
            "specializations": self.specializations,
            "coordination_compatible": self.coordination_compatible,
            "resource_requirements": self.resource_requirements,
            "success_rate": self.success_rate,
            "average_response_time": self.average_response_time,
            "last_updated": self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentCapability':
        """Create capability from dictionary representation."""
        capability = cls(
            agent_type=data["agent_type"],
            domains=data["domains"],
            strengths=data["strengths"],
            performance_score=data.get("performance_score", 0.8),
            specializations=data.get("specializations", []),
            coordination_compatible=data.get("coordination_compatible", True),
            resource_requirements=data.get("resource_requirements", {})
        )
        capability.success_rate = data.get("success_rate", 0.85)
        capability.average_response_time = data.get("average_response_time", 30.0)
        capability.last_updated = data.get("last_updated", datetime.now().isoformat())
        return capability
    
    def calculate_relevance_score(
        self, 
        query_type: QueryType, 
        query_keywords: Set[str],
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate relevance score for a specific query."""
        score = 0.0
        
        # Base performance score (30% weight)
        score += self.performance_score * 0.3
        
        # Domain matching (40% weight)
        domain_match = 0.0
        query_type_domains = self._get_query_type_domains(query_type)
        matching_domains = set(self.domains) & set(query_type_domains)
        if self.domains:
            domain_match = len(matching_domains) / len(self.domains)
        score += domain_match * 0.4
        
        # Keyword matching (20% weight)
        keyword_match = 0.0
        all_keywords = set(self.strengths + self.specializations)
        matching_keywords = query_keywords & all_keywords
        if all_keywords:
            keyword_match = len(matching_keywords) / len(all_keywords)
        score += keyword_match * 0.2
        
        # Success rate bonus (10% weight)
        score += self.success_rate * 0.1
        
        return min(score, 1.0)
    
    def _get_query_type_domains(self, query_type: QueryType) -> List[str]:
        """Map query type to relevant domains."""
        domain_mapping = {
            QueryType.CODING: ["programming", "software_development", "debugging", "testing"],
            QueryType.RESEARCH: ["research", "analysis", "information_retrieval", "documentation"],
            QueryType.ANALYSIS: ["analysis", "review", "evaluation", "optimization"],
            QueryType.COORDINATION: ["project_management", "coordination", "workflow", "planning"],
            QueryType.TASK_MANAGEMENT: ["project_management", "task_management", "organization"],
            QueryType.KNOWLEDGE: ["knowledge_management", "documentation", "research"],
            QueryType.GENERAL: ["general_purpose", "problem_solving", "assistance"]
        }
        return domain_mapping.get(query_type, [])


class AgentCapabilityMatrix:
    """
    Manages the capability matrix for all agent types in the system.
    
    Provides intelligent routing, capability matching, and dynamic updates
    based on agent performance and specialization.
    """
    
    def __init__(self):
        """Initialize the capability matrix with default agent definitions."""
        self.capabilities: Dict[str, AgentCapability] = {}
        self._initialize_default_capabilities()
        logger.info("Agent capability matrix initialized")
    
    def _initialize_default_capabilities(self):
        """Initialize default capabilities for Claude Flow's 64 agent types."""
        
        # Core Development Agents
        core_agents = [
            AgentCapability(
                "coder",
                domains=["programming", "software_development", "implementation"],
                strengths=["code_generation", "debugging", "refactoring", "algorithm_implementation"],
                performance_score=0.9,
                specializations=["python", "javascript", "typescript", "api_development"],
                resource_requirements={"memory": "medium", "cpu": "medium"}
            ),
            AgentCapability(
                "reviewer",
                domains=["code_review", "quality_assurance", "security"],
                strengths=["code_analysis", "security_review", "best_practices", "optimization"],
                performance_score=0.88,
                specializations=["static_analysis", "security_audit", "performance_review"]
            ),
            AgentCapability(
                "tester",
                domains=["testing", "quality_assurance", "validation"],
                strengths=["test_generation", "test_automation", "coverage_analysis", "bug_detection"],
                performance_score=0.85,
                specializations=["unit_testing", "integration_testing", "e2e_testing"]
            ),
            AgentCapability(
                "planner",
                domains=["project_management", "planning", "coordination"],
                strengths=["task_breakdown", "timeline_planning", "resource_allocation", "risk_assessment"],
                performance_score=0.82,
                specializations=["agile_planning", "sprint_planning", "roadmap_creation"]
            ),
            AgentCapability(
                "researcher",
                domains=["research", "analysis", "information_retrieval"],
                strengths=["information_gathering", "literature_review", "trend_analysis", "documentation"],
                performance_score=0.87,
                specializations=["technical_research", "market_research", "competitive_analysis"]
            )
        ]
        
        # Specialized Development Agents
        specialized_agents = [
            AgentCapability(
                "backend-dev",
                domains=["backend_development", "api_development", "database"],
                strengths=["api_design", "database_schema", "microservices", "server_architecture"],
                performance_score=0.89,
                specializations=["rest_api", "graphql", "database_optimization", "cloud_architecture"]
            ),
            AgentCapability(
                "mobile-dev",
                domains=["mobile_development", "ui_ux", "native_development"],
                strengths=["mobile_ui", "responsive_design", "native_apis", "cross_platform"],
                performance_score=0.85,
                specializations=["react_native", "flutter", "ios", "android"]
            ),
            AgentCapability(
                "ml-developer",
                domains=["machine_learning", "ai", "data_science"],
                strengths=["model_development", "data_preprocessing", "feature_engineering", "evaluation"],
                performance_score=0.88,
                specializations=["deep_learning", "nlp", "computer_vision", "recommendation_systems"]
            ),
            AgentCapability(
                "cicd-engineer",
                domains=["devops", "automation", "deployment"],
                strengths=["pipeline_setup", "automation", "containerization", "monitoring"],
                performance_score=0.86,
                specializations=["docker", "kubernetes", "github_actions", "aws"]
            )
        ]
        
        # Coordination and Management Agents
        coordination_agents = [
            AgentCapability(
                "system-architect",
                domains=["system_design", "architecture", "scalability"],
                strengths=["system_design", "scalability_planning", "technology_selection", "integration"],
                performance_score=0.91,
                specializations=["microservices", "distributed_systems", "cloud_architecture"]
            ),
            AgentCapability(
                "api-docs",
                domains=["documentation", "api_design", "technical_writing"],
                strengths=["api_documentation", "technical_writing", "specification", "examples"],
                performance_score=0.84,
                specializations=["openapi", "swagger", "postman", "readme"]
            ),
            AgentCapability(
                "code-analyzer",
                domains=["analysis", "metrics", "optimization"],
                strengths=["code_analysis", "metrics_collection", "performance_analysis", "refactoring"],
                performance_score=0.86,
                specializations=["static_analysis", "complexity_analysis", "dependency_analysis"]
            )
        ]
        
        # Performance and Optimization Agents
        performance_agents = [
            AgentCapability(
                "perf-analyzer",
                domains=["performance", "optimization", "profiling"],
                strengths=["performance_testing", "bottleneck_identification", "optimization", "monitoring"],
                performance_score=0.87,
                specializations=["load_testing", "memory_profiling", "database_optimization"]
            ),
            AgentCapability(
                "performance-benchmarker",
                domains=["benchmarking", "testing", "evaluation"],
                strengths=["benchmark_creation", "performance_comparison", "metrics_collection"],
                performance_score=0.85,
                specializations=["api_benchmarking", "database_benchmarking", "ui_performance"]
            )
        ]
        
        # SPARC Methodology Agents
        sparc_agents = [
            AgentCapability(
                "specification",
                domains=["requirements", "specification", "analysis"],
                strengths=["requirement_analysis", "specification_writing", "user_stories", "acceptance_criteria"],
                performance_score=0.88,
                specializations=["functional_requirements", "non_functional_requirements", "use_cases"]
            ),
            AgentCapability(
                "pseudocode",
                domains=["algorithm_design", "logic", "planning"],
                strengths=["algorithm_design", "pseudocode_generation", "logic_flow", "problem_decomposition"],
                performance_score=0.86,
                specializations=["data_structures", "algorithms", "complexity_analysis"]
            ),
            AgentCapability(
                "architecture",
                domains=["system_architecture", "design_patterns", "structure"],
                strengths=["architectural_design", "design_patterns", "component_design", "integration"],
                performance_score=0.89,
                specializations=["microservices", "event_driven", "layered_architecture"]
            ),
            AgentCapability(
                "refinement",
                domains=["optimization", "improvement", "iteration"],
                strengths=["code_refinement", "optimization", "iteration", "enhancement"],
                performance_score=0.87,
                specializations=["performance_optimization", "code_quality", "maintainability"]
            )
        ]
        
        # GitHub and Repository Management Agents
        github_agents = [
            AgentCapability(
                "pr-manager",
                domains=["version_control", "collaboration", "review"],
                strengths=["pull_request_management", "code_review", "merge_conflict_resolution"],
                performance_score=0.84,
                specializations=["github", "git_workflow", "branch_management"]
            ),
            AgentCapability(
                "issue-tracker",
                domains=["project_management", "bug_tracking", "task_management"],
                strengths=["issue_management", "bug_triage", "task_tracking", "prioritization"],
                performance_score=0.83,
                specializations=["github_issues", "jira", "bug_reporting"]
            ),
            AgentCapability(
                "release-manager",
                domains=["deployment", "versioning", "release_management"],
                strengths=["release_planning", "version_management", "deployment_coordination"],
                performance_score=0.85,
                specializations=["semantic_versioning", "changelog", "deployment_automation"]
            )
        ]
        
        # Swarm Coordination Agents
        swarm_agents = [
            AgentCapability(
                "hierarchical-coordinator",
                domains=["coordination", "management", "hierarchy"],
                strengths=["hierarchical_coordination", "task_delegation", "progress_monitoring"],
                performance_score=0.88,
                coordination_compatible=True,
                specializations=["team_management", "task_distribution", "reporting"]
            ),
            AgentCapability(
                "mesh-coordinator",
                domains=["coordination", "peer_to_peer", "distributed"],
                strengths=["peer_coordination", "consensus_building", "distributed_decision_making"],
                performance_score=0.86,
                coordination_compatible=True,
                specializations=["p2p_coordination", "consensus_algorithms", "load_balancing"]
            ),
            AgentCapability(
                "adaptive-coordinator",
                domains=["coordination", "adaptation", "optimization"],
                strengths=["adaptive_coordination", "dynamic_optimization", "learning"],
                performance_score=0.89,
                coordination_compatible=True,
                specializations=["machine_learning", "optimization", "pattern_recognition"]
            )
        ]
        
        # Combine all agent types
        all_agents = (
            core_agents + specialized_agents + coordination_agents + 
            performance_agents + sparc_agents + github_agents + swarm_agents
        )
        
        # Add to capabilities dictionary
        for agent in all_agents:
            self.capabilities[agent.agent_type] = agent
        
        logger.info(f"Initialized {len(all_agents)} agent capabilities")
    
    def get_capabilities(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get agent capabilities, optionally filtered by domain.
        
        Args:
            domain: Optional domain filter
            
        Returns:
            List of agent capability dictionaries
        """
        capabilities = []
        
        for agent_capability in self.capabilities.values():
            if domain is None or domain in agent_capability.domains:
                capabilities.append(agent_capability.to_dict())
        
        return capabilities
    
    def get_capabilities_for_query_type(
        self, 
        query_type: QueryType,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get agent capabilities ranked by relevance for a query type.
        
        Args:
            query_type: The type of query being processed
            max_results: Maximum number of capabilities to return
            
        Returns:
            List of ranked agent capability dictionaries with relevance scores
        """
        ranked_capabilities = []
        
        for agent_capability in self.capabilities.values():
            relevance_score = agent_capability.calculate_relevance_score(
                query_type=query_type,
                query_keywords=set()  # Could be enhanced with actual keyword extraction
            )
            
            capability_dict = agent_capability.to_dict()
            capability_dict["relevance_score"] = relevance_score
            
            ranked_capabilities.append(capability_dict)
        
        # Sort by relevance score (descending)
        ranked_capabilities.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return ranked_capabilities[:max_results]
    
    def get_agent_capability(self, agent_type: str) -> Optional[AgentCapability]:
        """
        Get capability information for a specific agent type.
        
        Args:
            agent_type: The agent type to look up
            
        Returns:
            AgentCapability object or None if not found
        """
        return self.capabilities.get(agent_type)
    
    def update_capabilities(self, capabilities_update: Dict[str, Any]) -> bool:
        """
        Update agent capabilities dynamically.
        
        Args:
            capabilities_update: Dictionary containing capability updates
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            for agent_type, update_data in capabilities_update.items():
                if agent_type in self.capabilities:
                    # Update existing capability
                    capability = self.capabilities[agent_type]
                    
                    # Update fields that are provided
                    if "performance_score" in update_data:
                        capability.performance_score = update_data["performance_score"]
                    if "success_rate" in update_data:
                        capability.success_rate = update_data["success_rate"]
                    if "average_response_time" in update_data:
                        capability.average_response_time = update_data["average_response_time"]
                    if "specializations" in update_data:
                        capability.specializations = update_data["specializations"]
                    
                    capability.last_updated = datetime.now().isoformat()
                    
                else:
                    # Create new capability
                    if all(key in update_data for key in ["domains", "strengths"]):
                        new_capability = AgentCapability.from_dict({
                            "agent_type": agent_type,
                            **update_data
                        })
                        self.capabilities[agent_type] = new_capability
            
            logger.info(f"Updated capabilities for {len(capabilities_update)} agents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update capabilities: {e}")
            return False
    
    def update_performance_metrics(
        self, 
        agent_type: str, 
        success: bool, 
        response_time: float
    ) -> bool:
        """
        Update performance metrics for an agent based on execution results.
        
        Args:
            agent_type: The agent type that was executed
            success: Whether the execution was successful
            response_time: Time taken for the execution
            
        Returns:
            True if update successful
        """
        try:
            if agent_type not in self.capabilities:
                logger.warning(f"Agent type {agent_type} not found in capabilities")
                return False
            
            capability = self.capabilities[agent_type]
            
            # Update success rate with exponential smoothing
            alpha = 0.1  # Smoothing factor
            new_success = 1.0 if success else 0.0
            capability.success_rate = (
                alpha * new_success + (1 - alpha) * capability.success_rate
            )
            
            # Update average response time with exponential smoothing
            capability.average_response_time = (
                alpha * response_time + (1 - alpha) * capability.average_response_time
            )
            
            # Update performance score based on success rate and response time
            # Normalize response time (assuming 60s is baseline)
            normalized_time = min(response_time / 60.0, 2.0)  # Cap at 2x baseline
            time_penalty = max(0, normalized_time - 1.0) * 0.2  # Penalty for slow response
            
            capability.performance_score = min(
                capability.success_rate - time_penalty,
                1.0
            )
            
            capability.last_updated = datetime.now().isoformat()
            
            logger.debug(f"Updated metrics for {agent_type}: success_rate={capability.success_rate:.3f}, "
                        f"response_time={capability.average_response_time:.1f}s")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update performance metrics for {agent_type}: {e}")
            return False
    
    def get_coordination_compatible_agents(self) -> List[str]:
        """
        Get list of agents that support coordination.
        
        Returns:
            List of coordination-compatible agent types
        """
        compatible_agents = []
        
        for agent_type, capability in self.capabilities.items():
            if capability.coordination_compatible:
                compatible_agents.append(agent_type)
        
        return compatible_agents
    
    def get_agents_by_domain(self, domain: str) -> List[str]:
        """
        Get agents that specialize in a specific domain.
        
        Args:
            domain: The domain to search for
            
        Returns:
            List of agent types specializing in the domain
        """
        domain_agents = []
        
        for agent_type, capability in self.capabilities.items():
            if domain in capability.domains:
                domain_agents.append(agent_type)
        
        return domain_agents
    
    def export_capabilities(self) -> Dict[str, Any]:
        """
        Export all capabilities to a dictionary for serialization.
        
        Returns:
            Dictionary containing all capability data
        """
        return {
            "capabilities": {
                agent_type: capability.to_dict()
                for agent_type, capability in self.capabilities.items()
            },
            "export_timestamp": datetime.now().isoformat(),
            "total_agents": len(self.capabilities)
        }
    
    def import_capabilities(self, capabilities_data: Dict[str, Any]) -> bool:
        """
        Import capabilities from a dictionary.
        
        Args:
            capabilities_data: Dictionary containing capability data
            
        Returns:
            True if import successful
        """
        try:
            if "capabilities" in capabilities_data:
                for agent_type, capability_dict in capabilities_data["capabilities"].items():
                    capability = AgentCapability.from_dict(capability_dict)
                    self.capabilities[agent_type] = capability
                
                logger.info(f"Imported {len(capabilities_data['capabilities'])} capabilities")
                return True
            else:
                logger.error("Invalid capabilities data format")
                return False
                
        except Exception as e:
            logger.error(f"Failed to import capabilities: {e}")
            return False