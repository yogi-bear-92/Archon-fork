"""
End-to-end tests for Claude Flow Expert Agent system.

This module tests complete user workflows, real-world scenarios,
and system behavior under production-like conditions.
"""

import pytest
import asyncio
import time
from typing import List, Dict, Any
from unittest.mock import patch

from src.agents.claude_flow_expert.claude_flow_expert_agent import (
    ClaudeFlowExpertAgent, ClaudeFlowExpertConfig, ClaudeFlowExpertDependencies, QueryRequest
)
from src.agents.claude_flow_expert.capability_matrix import QueryType
from tests.mocks.claude_flow_expert_agent_mocks import TestDataGenerator
from tests.fixtures.claude_flow_expert_agent_fixtures import (
    performance_test_scenarios, integration_test_scenarios
)


@pytest.mark.e2e
class TestCompleteUserWorkflows:
    """Test complete user workflows from start to finish."""
    
    @pytest.mark.asyncio
    async def test_software_development_project_workflow(self, patched_claude_flow_expert_agent_dependencies):
        """Test complete software development project workflow."""
        # Set up comprehensive mock responses
        mcp_client = patched_claude_flow_expert_agent_dependencies["mcp_client"]
        coordinator = patched_claude_flow_expert_agent_dependencies["coordinator"]
        
        # Configure realistic responses for each phase
        research_responses = {
            "FastAPI best practices": {
                "success": True,
                "results": [
                    {
                        "content": "FastAPI best practices include using Pydantic models, dependency injection, and async/await",
                        "source": "fastapi-guide",
                        "relevance": 0.95
                    },
                    {
                        "content": "Structure FastAPI applications with routers, models, and services",
                        "source": "architecture-guide",
                        "relevance": 0.88
                    }
                ]
            },
            "PostgreSQL schema design": {
                "success": True,
                "results": [
                    {
                        "content": "Use appropriate data types, indexes, and foreign key constraints",
                        "source": "db-design-guide",
                        "relevance": 0.92
                    }
                ]
            },
            "JWT authentication implementation": {
                "success": True,
                "results": [
                    {
                        "content": "Implement JWT with refresh tokens for secure authentication",
                        "source": "security-guide", 
                        "relevance": 0.94
                    }
                ]
            }
        }
        
        for query, response in research_responses.items():
            mcp_client.set_rag_response(query, response)
        
        # Initialize claude flow expert agent
        config = ClaudeFlowExpertConfig(
            rag_enabled=True,
            max_coordinated_agents=8,
            memory_persistence_enabled=True
        )
        agent = ClaudeFlowExpertAgent(config)
        deps = ClaudeFlowExpertDependencies()
        
        # Define complete project workflow
        project_workflow = [
            {
                "phase": "requirements_gathering",
                "query": "Help me understand the requirements for building a modern web application with user authentication and data management",
                "expected_agents": ["researcher", "analyst"],
                "context": {"project_type": "web_application"}
            },
            {
                "phase": "technology_research", 
                "query": "Research FastAPI best practices for building scalable web APIs",
                "expected_rag": True,
                "context": {"framework": "fastapi", "building_on": "requirements"}
            },
            {
                "phase": "database_design",
                "query": "Research PostgreSQL schema design best practices for user management and data storage",
                "expected_rag": True,
                "context": {"database": "postgresql", "building_on": "technology_research"}
            },
            {
                "phase": "authentication_research",
                "query": "Research JWT authentication implementation with refresh tokens",
                "expected_rag": True,
                "context": {"security": "jwt", "building_on": "database_design"}
            },
            {
                "phase": "architecture_design",
                "query": "Design the overall system architecture based on the research findings",
                "expected_coordination": True,
                "max_agents": 4,
                "context": {"phase": "architecture", "building_on": "authentication_research"}
            },
            {
                "phase": "api_implementation",
                "query": "Implement the FastAPI application with user authentication endpoints",
                "expected_coordination": True,
                "max_agents": 3,
                "context": {"phase": "implementation", "component": "api"}
            },
            {
                "phase": "database_implementation", 
                "query": "Implement the PostgreSQL database schema and connection handling",
                "expected_coordination": True,
                "max_agents": 2,
                "context": {"phase": "implementation", "component": "database"}
            },
            {
                "phase": "testing_implementation",
                "query": "Create comprehensive test suite for the API endpoints and database operations",
                "expected_coordination": True,
                "max_agents": 3,
                "context": {"phase": "testing", "building_on": "implementation"}
            },
            {
                "phase": "deployment_planning",
                "query": "Plan deployment strategy with Docker and CI/CD pipeline",
                "expected_coordination": True,
                "max_agents": 2,
                "context": {"phase": "deployment", "building_on": "testing"}
            },
            {
                "phase": "documentation",
                "query": "Create API documentation and deployment instructions",
                "expected_agents": ["api-docs", "technical-writer"],
                "context": {"phase": "documentation", "building_on": "deployment"}
            }
        ]
        
        # Execute workflow
        workflow_results = []
        total_start_time = time.time()
        
        for step in project_workflow:
            step_start_time = time.time()
            
            # Build request
            request = QueryRequest(
                query=step["query"],
                context=step.get("context", {}),
                require_rag=step.get("expected_rag", False),
                max_agents=step.get("max_agents", 1)
            )
            
            # Process step
            result = await agent.process_query(request, deps)
            step_end_time = time.time()
            
            # Verify step success
            assert result.success is True, f"Step {step['phase']} failed: {result.message}"
            
            # Store results
            workflow_results.append({
                "phase": step["phase"],
                "result": result,
                "processing_time": step_end_time - step_start_time,
                "expected_rag": step.get("expected_rag", False),
                "expected_coordination": step.get("expected_coordination", False)
            })
        
        total_workflow_time = time.time() - total_start_time
        
        # Verify complete workflow
        assert len(workflow_results) == len(project_workflow)
        assert all(step_result["result"].success for step_result in workflow_results)
        
        # Verify workflow characteristics
        rag_steps = [step for step in workflow_results if step["expected_rag"]]
        coordination_steps = [step for step in workflow_results if step["expected_coordination"]]
        
        assert len(rag_steps) >= 3, "Should have multiple RAG-enhanced research steps"
        assert len(coordination_steps) >= 4, "Should have multiple coordination steps"
        
        # Verify component usage
        assert mcp_client.call_count >= 3, "Should have made multiple RAG queries"
        assert coordinator.call_count >= 4, "Should have coordinated multiple agent teams"
        
        # Performance verification
        assert total_workflow_time < 60.0, "Complete workflow should finish under 1 minute"
        avg_step_time = sum(step["processing_time"] for step in workflow_results) / len(workflow_results)
        assert avg_step_time < 5.0, "Average step processing time should be under 5 seconds"
        
        # Verify system metrics
        final_metrics = await agent.get_performance_metrics()
        assert final_metrics["queries_processed"] == len(project_workflow)
        assert final_metrics["successful_queries"] == len(project_workflow)
        assert final_metrics["rag_queries"] >= len(rag_steps)
        assert final_metrics["multi_agent_coordinations"] >= len(coordination_steps)
    
    @pytest.mark.asyncio
    async def test_research_and_analysis_workflow(self, patched_claude_flow_expert_agent_dependencies):
        """Test research and analysis workflow."""
        mcp_client = patched_claude_flow_expert_agent_dependencies["mcp_client"]
        
        # Set up research responses
        research_topics = {
            "microservices architecture patterns": {
                "success": True,
                "results": [
                    {
                        "content": "Common microservices patterns include API Gateway, Service Discovery, and Circuit Breaker",
                        "source": "microservices-guide",
                        "relevance": 0.95
                    }
                ]
            },
            "container orchestration with Kubernetes": {
                "success": True, 
                "results": [
                    {
                        "content": "Kubernetes provides container orchestration with automated deployment, scaling, and management",
                        "source": "k8s-guide",
                        "relevance": 0.90
                    }
                ]
            }
        }
        
        for topic, response in research_topics.items():
            mcp_client.set_rag_response(topic, response)
        
        config = ClaudeFlowExpertConfig(rag_enabled=True)
        agent = ClaudeFlowExpertAgent(config)
        deps = ClaudeFlowExpertDependencies()
        
        # Research workflow
        research_steps = [
            {
                "query": "Research microservices architecture patterns and best practices",
                "expected_rag": True,
                "topic": "architecture"
            },
            {
                "query": "Research container orchestration with Kubernetes for microservices deployment",
                "expected_rag": True,
                "topic": "deployment"
            },
            {
                "query": "Analyze the trade-offs between microservices and monolithic architectures",
                "expected_analysis": True,
                "context": {"building_on": "previous_research"}
            },
            {
                "query": "Create a comprehensive comparison report of deployment strategies",
                "expected_synthesis": True,
                "context": {"synthesize": "all_research"}
            }
        ]
        
        research_results = []
        
        for step in research_steps:
            request = QueryRequest(
                query=step["query"],
                require_rag=step.get("expected_rag", False),
                context=step.get("context", {})
            )
            
            result = await agent.process_query(request, deps)
            
            assert result.success is True, f"Research step failed: {result.message}"
            research_results.append(result)
        
        # Verify research workflow
        assert len(research_results) == len(research_steps)
        rag_results = [r for r in research_results if len(r.rag_sources_used) > 0]
        assert len(rag_results) >= 2, "Should have performed RAG queries"
        
        # Verify knowledge accumulation
        assert mcp_client.call_count >= 2
    
    @pytest.mark.asyncio
    async def test_troubleshooting_and_debugging_workflow(self, patched_claude_flow_expert_agent_dependencies):
        """Test troubleshooting and debugging workflow."""
        coordinator = patched_claude_flow_expert_agent_dependencies["coordinator"]
        
        config = ClaudeFlowExpertConfig()
        agent = ClaudeFlowExpertAgent(config)
        deps = ClaudeFlowExpertDependencies()
        
        # Debugging workflow
        debug_scenarios = [
            {
                "query": "My FastAPI application is returning 500 errors intermittently",
                "context": {"error_type": "server_error", "frequency": "intermittent"}
            },
            {
                "query": "Analyze the error patterns and identify potential root causes",
                "max_agents": 2,
                "context": {"phase": "analysis", "building_on": "error_report"}
            },
            {
                "query": "Implement monitoring and logging to capture more diagnostic information",
                "max_agents": 2,
                "context": {"phase": "monitoring", "building_on": "analysis"}
            },
            {
                "query": "Create a systematic debugging checklist for similar issues",
                "context": {"phase": "documentation", "building_on": "monitoring"}
            }
        ]
        
        debug_results = []
        
        for scenario in debug_scenarios:
            request = QueryRequest(
                query=scenario["query"],
                context=scenario.get("context", {}),
                max_agents=scenario.get("max_agents", 1)
            )
            
            result = await agent.process_query(request, deps)
            assert result.success is True
            debug_results.append(result)
        
        # Verify debugging workflow progression
        assert len(debug_results) == len(debug_scenarios)
        coordination_results = [r for r in debug_results if len(r.agents_used) > 1]
        assert len(coordination_results) >= 2, "Should have coordinated multiple agents"


@pytest.mark.e2e
class TestRealWorldScenarios:
    """Test realistic production scenarios and edge cases."""
    
    @pytest.mark.asyncio
    async def test_high_availability_system_design(self, patched_claude_flow_expert_agent_dependencies):
        """Test designing a high-availability system."""
        mcp_client = patched_claude_flow_expert_agent_dependencies["mcp_client"]
        coordinator = patched_claude_flow_expert_agent_dependencies["coordinator"]
        
        # Set up HA-related research responses
        ha_responses = {
            "high availability architecture patterns": {
                "success": True,
                "results": [
                    {
                        "content": "HA patterns include load balancing, redundancy, failover, and geographic distribution",
                        "source": "ha-guide",
                        "relevance": 0.96
                    }
                ]
            },
            "database replication strategies": {
                "success": True,
                "results": [
                    {
                        "content": "Database replication strategies include master-slave, master-master, and sharding",
                        "source": "db-replication-guide",
                        "relevance": 0.94
                    }
                ]
            }
        }
        
        for query, response in ha_responses.items():
            mcp_client.set_rag_response(query, response)
        
        config = ClaudeFlowExpertConfig(rag_enabled=True, max_coordinated_agents=6)
        agent = ClaudeFlowExpertAgent(config)
        deps = ClaudeFlowExpertDependencies()
        
        # High availability design workflow
        ha_workflow = [
            {
                "query": "Research high availability architecture patterns for web applications",
                "require_rag": True
            },
            {
                "query": "Research database replication strategies for high availability",
                "require_rag": True
            },
            {
                "query": "Design a multi-region architecture with automatic failover",
                "max_agents": 4,
                "context": {"requirements": ["99.9% uptime", "multi_region", "auto_failover"]}
            },
            {
                "query": "Plan monitoring and alerting strategy for the HA system",
                "max_agents": 3,
                "context": {"monitoring_targets": ["application", "database", "infrastructure"]}
            },
            {
                "query": "Create disaster recovery procedures and runbooks",
                "max_agents": 2,
                "context": {"scenarios": ["region_failure", "database_corruption", "traffic_spike"]}
            }
        ]
        
        ha_results = []
        
        for step in ha_workflow:
            request = QueryRequest(
                query=step["query"],
                require_rag=step.get("require_rag", False),
                max_agents=step.get("max_agents", 1),
                context=step.get("context", {})
            )
            
            result = await agent.process_query(request, deps)
            assert result.success is True, f"HA workflow step failed: {result.message}"
            ha_results.append(result)
        
        # Verify comprehensive HA planning
        assert len(ha_results) == len(ha_workflow)
        assert mcp_client.call_count >= 2  # Research queries
        assert coordinator.call_count >= 3  # Multi-agent coordination
        
        # Check that complex system design was addressed
        multi_agent_results = [r for r in ha_results if len(r.agents_used) > 1]
        assert len(multi_agent_results) >= 3, "Should have multiple complex design phases"
    
    @pytest.mark.asyncio
    async def test_legacy_system_migration(self, patched_claude_flow_expert_agent_dependencies):
        """Test legacy system migration planning and execution."""
        mcp_client = patched_claude_flow_expert_agent_dependencies["mcp_client"]
        coordinator = patched_claude_flow_expert_agent_dependencies["coordinator"]
        
        # Migration-related responses
        migration_responses = {
            "legacy system migration strategies": {
                "success": True,
                "results": [
                    {
                        "content": "Migration strategies include Big Bang, Parallel Run, and Strangler Fig patterns",
                        "source": "migration-guide",
                        "relevance": 0.93
                    }
                ]
            },
            "data migration best practices": {
                "success": True,
                "results": [
                    {
                        "content": "Data migration requires careful planning, validation, and rollback procedures",
                        "source": "data-migration-guide",
                        "relevance": 0.91
                    }
                ]
            }
        }
        
        for query, response in migration_responses.items():
            mcp_client.set_rag_response(query, response)
        
        config = ClaudeFlowExpertConfig(rag_enabled=True, max_coordinated_agents=8)
        agent = ClaudeFlowExpertAgent(config)
        deps = ClaudeFlowExpertDependencies()
        
        migration_workflow = [
            {
                "query": "Analyze the current legacy system architecture and identify migration challenges",
                "max_agents": 3,
                "context": {
                    "legacy_system": "monolithic_java_app",
                    "target": "microservices_architecture",
                    "constraints": ["zero_downtime", "data_integrity"]
                }
            },
            {
                "query": "Research legacy system migration strategies and best practices",
                "require_rag": True
            },
            {
                "query": "Research data migration best practices for large datasets",
                "require_rag": True
            },
            {
                "query": "Design a phased migration strategy with risk mitigation",
                "max_agents": 4,
                "context": {"approach": "strangler_fig", "phases": 4}
            },
            {
                "query": "Plan data migration strategy with validation and rollback procedures",
                "max_agents": 3,
                "context": {"data_volume": "100GB", "downtime_tolerance": "4_hours"}
            },
            {
                "query": "Create comprehensive testing strategy for migration validation",
                "max_agents": 3,
                "context": {"testing_types": ["functional", "performance", "integration"]}
            },
            {
                "query": "Develop monitoring and rollback procedures for migration phases",
                "max_agents": 2,
                "context": {"rollback_time": "30_minutes", "monitoring": "real_time"}
            }
        ]
        
        migration_results = []
        
        for step in migration_workflow:
            request = QueryRequest(
                query=step["query"],
                require_rag=step.get("require_rag", False),
                max_agents=step.get("max_agents", 1),
                context=step.get("context", {})
            )
            
            result = await agent.process_query(request, deps)
            assert result.success is True, f"Migration step failed: {result.message}"
            migration_results.append(result)
        
        # Verify comprehensive migration planning
        assert len(migration_results) == len(migration_workflow)
        
        # Check research was performed
        rag_results = [r for r in migration_results if len(r.rag_sources_used) > 0]
        assert len(rag_results) >= 2, "Should have performed migration research"
        
        # Check complex planning was coordinated
        coordination_results = [r for r in migration_results if len(r.agents_used) > 1]
        assert len(coordination_results) >= 5, "Should have coordinated complex planning steps"
    
    @pytest.mark.asyncio
    async def test_security_incident_response(self, patched_claude_flow_expert_agent_dependencies):
        """Test security incident response workflow."""
        coordinator = patched_claude_flow_expert_agent_dependencies["coordinator"]
        fallback_manager = patched_claude_flow_expert_agent_dependencies["fallback"]
        
        config = ClaudeFlowExpertConfig()
        agent = ClaudeFlowExpertAgent(config)
        deps = ClaudeFlowExpertDependencies()
        
        # Security incident response workflow
        incident_workflow = [
            {
                "query": "Security alert: Unusual API access patterns detected with potential data exfiltration",
                "max_agents": 4,
                "context": {
                    "incident_type": "potential_data_breach",
                    "severity": "high",
                    "systems_affected": ["api_gateway", "user_database", "logging"]
                }
            },
            {
                "query": "Immediate containment: Identify and isolate affected systems",
                "max_agents": 3,
                "context": {
                    "phase": "containment",
                    "priority": "critical",
                    "actions_needed": ["access_control", "traffic_filtering", "system_isolation"]
                }
            },
            {
                "query": "Forensic analysis: Analyze logs and identify attack vectors and scope",
                "max_agents": 3,
                "context": {
                    "phase": "investigation",
                    "data_sources": ["api_logs", "database_logs", "system_logs"],
                    "timeline": "last_24_hours"
                }
            },
            {
                "query": "Recovery planning: Develop system recovery and security hardening plan",
                "max_agents": 3,
                "context": {
                    "phase": "recovery",
                    "requirements": ["data_integrity", "security_improvements", "service_restoration"]
                }
            },
            {
                "query": "Post-incident review: Document lessons learned and improve security procedures",
                "max_agents": 2,
                "context": {
                    "phase": "post_incident",
                    "deliverables": ["incident_report", "security_improvements", "procedure_updates"]
                }
            }
        ]
        
        incident_results = []
        
        for step in incident_workflow:
            request = QueryRequest(
                query=step["query"],
                max_agents=step.get("max_agents", 1),
                context=step.get("context", {})
            )
            
            result = await agent.process_query(request, deps)
            assert result.success is True, f"Incident response step failed: {result.message}"
            incident_results.append(result)
        
        # Verify incident response coordination
        assert len(incident_results) == len(incident_workflow)
        
        # All steps should have used coordination due to urgency and complexity
        coordination_results = [r for r in incident_results if len(r.agents_used) > 1]
        assert len(coordination_results) >= 4, "Should have coordinated response teams"
        
        # Verify rapid response (all steps should complete quickly)
        processing_times = [r.processing_time for r in incident_results if r.processing_time]
        if processing_times:
            avg_response_time = sum(processing_times) / len(processing_times)
            assert avg_response_time < 3.0, "Incident response should be fast"


@pytest.mark.e2e
class TestSystemReliabilityScenarios:
    """Test system reliability under various failure conditions."""
    
    @pytest.mark.asyncio
    async def test_partial_system_degradation(self, patched_claude_flow_expert_agent_dependencies):
        """Test system behavior under partial degradation."""
        mcp_client = patched_claude_flow_expert_agent_dependencies["mcp_client"]
        coordinator = patched_claude_flow_expert_agent_dependencies["coordinator"]
        fallback_manager = patched_claude_flow_expert_agent_dependencies["fallback"]
        
        # Configure partial failures
        mcp_client.failure_rate = 0.5  # 50% failure rate for RAG
        coordinator.should_fail = False  # Coordination working
        
        config = ClaudeFlowExpertConfig(
            rag_enabled=True,
            rag_fallback_enabled=True,
            circuit_breaker_enabled=True
        )
        agent = ClaudeFlowExpertAgent(config)
        deps = ClaudeFlowExpertDependencies()
        
        # Test various query types under partial degradation
        degraded_queries = [
            QueryRequest(query="Research query (may fail)", require_rag=True),
            QueryRequest(query="Coordination task", max_agents=3),
            QueryRequest(query="Simple query", query_type=QueryType.GENERAL),
            QueryRequest(query="Another research query", require_rag=True),
            QueryRequest(query="Complex task", max_agents=2, require_rag=False)
        ]
        
        results = []
        
        for query in degraded_queries:
            result = await agent.process_query(query, deps)
            results.append(result)
        
        # System should remain functional despite partial failures
        successful_results = [r for r in results if r.success]
        assert len(successful_results) >= 3, "Most queries should succeed despite degradation"
        
        # Fallback mechanisms should be engaged
        fallback_used = [r for r in results if r.fallback_used]
        assert len(fallback_used) > 0, "Should have used fallback mechanisms"
    
    @pytest.mark.asyncio
    async def test_recovery_after_system_restoration(self, patched_claude_flow_expert_agent_dependencies):
        """Test system recovery after components are restored."""
        mcp_client = patched_claude_flow_expert_agent_dependencies["mcp_client"]
        coordinator = patched_claude_flow_expert_agent_dependencies["coordinator"]
        
        config = ClaudeFlowExpertConfig(circuit_breaker_enabled=True)
        agent = ClaudeFlowExpertAgent(config)
        deps = ClaudeFlowExpertDependencies()
        
        # Phase 1: System degradation
        mcp_client.should_fail = True
        coordinator.should_fail = True
        
        degraded_queries = [
            QueryRequest(query="Query during outage 1", require_rag=True),
            QueryRequest(query="Query during outage 2", max_agents=2)
        ]
        
        degraded_results = []
        for query in degraded_queries:
            result = await agent.process_query(query, deps)
            degraded_results.append(result)
        
        # Phase 2: System recovery
        mcp_client.should_fail = False
        coordinator.should_fail = False
        
        # Allow some time for circuit breakers to recover
        await asyncio.sleep(0.1)
        
        recovery_queries = [
            QueryRequest(query="Query after recovery 1", require_rag=True),
            QueryRequest(query="Query after recovery 2", max_agents=2),
            QueryRequest(query="Query after recovery 3", require_rag=True)
        ]
        
        recovery_results = []
        for query in recovery_queries:
            result = await agent.process_query(query, deps)
            recovery_results.append(result)
        
        # System should recover and work normally
        successful_recovery = [r for r in recovery_results if r.success]
        assert len(successful_recovery) == len(recovery_queries), "All queries should succeed after recovery"
        
        # Should use primary systems again, not just fallbacks
        non_fallback_results = [r for r in recovery_results if not r.fallback_used]
        assert len(non_fallback_results) > 0, "Should use primary systems after recovery"


@pytest.mark.e2e
@pytest.mark.slow
class TestLongRunningWorkflows:
    """Test extended workflows that simulate real-world usage patterns."""
    
    @pytest.mark.asyncio
    async def test_day_long_development_session(self, patched_claude_flow_expert_agent_dependencies):
        """Simulate a full day development session with various tasks."""
        mcp_client = patched_claude_flow_expert_agent_dependencies["mcp_client"]
        coordinator = patched_claude_flow_expert_agent_dependencies["coordinator"]
        
        # Set up comprehensive responses for a day's work
        daily_responses = {
            "morning standup planning": {"success": True, "results": [{"content": "Daily planning guidance"}]},
            "code review best practices": {"success": True, "results": [{"content": "Code review guidelines"}]},
            "performance optimization": {"success": True, "results": [{"content": "Performance tuning tips"}]},
            "deployment strategies": {"success": True, "results": [{"content": "Deployment best practices"}]},
            "incident postmortem": {"success": True, "results": [{"content": "Postmortem procedures"}]}
        }
        
        for query, response in daily_responses.items():
            mcp_client.set_rag_response(query, response)
        
        config = ClaudeFlowExpertConfig(
            rag_enabled=True,
            max_coordinated_agents=6,
            memory_persistence_enabled=True
        )
        agent = ClaudeFlowExpertAgent(config)
        deps = ClaudeFlowExpertDependencies()
        
        # Simulate a full day of development tasks
        daily_workflow = [
            # Morning (9 AM - 12 PM): Planning and Research
            {"time": "09:00", "query": "Plan today's development tasks based on sprint goals", "context": {"phase": "morning_planning"}},
            {"time": "09:30", "query": "Research morning standup planning best practices", "require_rag": True},
            {"time": "10:00", "query": "Review and analyze yesterday's code commits for issues", "max_agents": 2},
            {"time": "11:00", "query": "Research code review best practices for team improvement", "require_rag": True},
            
            # Midday (12 PM - 2 PM): Implementation
            {"time": "12:00", "query": "Implement user authentication feature with proper error handling", "max_agents": 3},
            {"time": "13:00", "query": "Create comprehensive tests for the authentication feature", "max_agents": 2},
            
            # Afternoon (2 PM - 5 PM): Optimization and Integration
            {"time": "14:00", "query": "Analyze application performance bottlenecks", "max_agents": 2},
            {"time": "14:30", "query": "Research performance optimization techniques", "require_rag": True},
            {"time": "15:00", "query": "Implement performance improvements based on analysis", "max_agents": 3},
            {"time": "16:00", "query": "Integrate new features with existing system", "max_agents": 2},
            
            # Late afternoon (5 PM - 6 PM): Deployment and Review
            {"time": "17:00", "query": "Research deployment strategies for the updated application", "require_rag": True},
            {"time": "17:30", "query": "Plan deployment rollout with monitoring and rollback procedures", "max_agents": 3},
            
            # End of day (6 PM): Incident Response
            {"time": "18:00", "query": "Conduct postmortem analysis of yesterday's deployment issue", "max_agents": 2},
            {"time": "18:30", "query": "Research incident postmortem best practices", "require_rag": True},
        ]
        
        daily_results = []
        session_start_time = time.time()
        
        for task in daily_workflow:
            task_start_time = time.time()
            
            request = QueryRequest(
                query=task["query"],
                require_rag=task.get("require_rag", False),
                max_agents=task.get("max_agents", 1),
                context={**task.get("context", {}), "session_time": task["time"]}
            )
            
            result = await agent.process_query(request, deps)
            task_end_time = time.time()
            
            assert result.success is True, f"Daily task at {task['time']} failed: {result.message}"
            
            daily_results.append({
                "time": task["time"],
                "result": result,
                "processing_time": task_end_time - task_start_time
            })
            
            # Brief pause between tasks (simulate realistic work patterns)
            await asyncio.sleep(0.05)
        
        session_end_time = time.time()
        total_session_time = session_end_time - session_start_time
        
        # Verify full day session
        assert len(daily_results) == len(daily_workflow)
        assert all(task["result"].success for task in daily_results)
        
        # Analyze session patterns
        rag_tasks = [task for task in daily_results if len(task["result"].rag_sources_used) > 0]
        coordination_tasks = [task for task in daily_results if len(task["result"].agents_used) > 1]
        
        assert len(rag_tasks) >= 5, "Should have performed research throughout the day"
        assert len(coordination_tasks) >= 8, "Should have coordinated complex tasks"
        
        # Performance over time should remain stable
        processing_times = [task["processing_time"] for task in daily_results]
        avg_processing_time = sum(processing_times) / len(processing_times)
        assert avg_processing_time < 2.0, "Should maintain performance throughout session"
        
        # Session should complete in reasonable total time
        assert total_session_time < 120.0, "Full day simulation should complete in under 2 minutes"
        
        # Verify sustained system health
        final_metrics = await agent.get_performance_metrics()
        assert final_metrics["queries_processed"] == len(daily_workflow)
        assert final_metrics["successful_queries"] == len(daily_workflow)
        assert final_metrics["average_processing_time"] > 0
        
        # System components should have been heavily utilized
        assert mcp_client.call_count >= 5
        assert coordinator.call_count >= 8
    
    @pytest.mark.asyncio 
    async def test_multi_project_context_switching(self, patched_claude_flow_expert_agent_dependencies):
        """Test handling multiple projects with context switching."""
        mcp_client = patched_claude_flow_expert_agent_dependencies["mcp_client"]
        coordinator = patched_claude_flow_expert_agent_dependencies["coordinator"]
        
        # Set up responses for different projects
        project_responses = {
            "web application development": {"success": True, "results": [{"content": "Web dev guidance"}]},
            "mobile app development": {"success": True, "results": [{"content": "Mobile dev guidance"}]},
            "data pipeline architecture": {"success": True, "results": [{"content": "Data pipeline guidance"}]},
            "machine learning model": {"success": True, "results": [{"content": "ML model guidance"}]}
        }
        
        for query, response in project_responses.items():
            mcp_client.set_rag_response(query, response)
        
        config = ClaudeFlowExpertConfig(
            rag_enabled=True,
            max_coordinated_agents=8,
            memory_persistence_enabled=True
        )
        agent = ClaudeFlowExpertAgent(config)
        deps = ClaudeFlowExpertDependencies()
        
        # Define multiple projects with interleaved tasks
        multi_project_tasks = [
            # Project A: Web Application
            {"project": "web_app", "query": "Research web application development best practices", "require_rag": True},
            {"project": "web_app", "query": "Design REST API architecture for the web application", "max_agents": 3},
            
            # Project B: Mobile App
            {"project": "mobile_app", "query": "Research mobile app development frameworks", "require_rag": True},
            {"project": "mobile_app", "query": "Plan mobile app UI/UX design strategy", "max_agents": 2},
            
            # Project C: Data Pipeline
            {"project": "data_pipeline", "query": "Research data pipeline architecture patterns", "require_rag": True},
            {"project": "data_pipeline", "query": "Design scalable data processing pipeline", "max_agents": 4},
            
            # Context switching back to Project A
            {"project": "web_app", "query": "Implement authentication system for web application", "max_agents": 2},
            {"project": "web_app", "query": "Create testing strategy for web API endpoints", "max_agents": 3},
            
            # Project D: ML Model
            {"project": "ml_model", "query": "Research machine learning model deployment strategies", "require_rag": True},
            {"project": "ml_model", "query": "Plan ML model training and evaluation pipeline", "max_agents": 3},
            
            # More context switching
            {"project": "mobile_app", "query": "Implement mobile app backend integration", "max_agents": 2},
            {"project": "data_pipeline", "query": "Implement data validation and monitoring", "max_agents": 2},
            
            # Final tasks
            {"project": "web_app", "query": "Plan deployment strategy for web application", "max_agents": 2},
            {"project": "ml_model", "query": "Create ML model monitoring and alerting system", "max_agents": 3}
        ]
        
        project_results = []
        project_contexts = {}
        
        for task in multi_project_tasks:
            project_id = task["project"]
            
            # Maintain project context
            if project_id not in project_contexts:
                project_contexts[project_id] = {"task_count": 0, "previous_tasks": []}
            
            project_contexts[project_id]["task_count"] += 1
            project_contexts[project_id]["previous_tasks"].append(task["query"])
            
            request = QueryRequest(
                query=task["query"],
                require_rag=task.get("require_rag", False),
                max_agents=task.get("max_agents", 1),
                context={
                    "project_id": project_id,
                    "project_context": project_contexts[project_id],
                    "task_number": project_contexts[project_id]["task_count"]
                }
            )
            
            result = await agent.process_query(request, deps)
            assert result.success is True, f"Multi-project task failed: {result.message}"
            
            project_results.append({
                "project": project_id,
                "result": result,
                "task_number": project_contexts[project_id]["task_count"]
            })
        
        # Verify multi-project handling
        assert len(project_results) == len(multi_project_tasks)
        assert all(task["result"].success for task in project_results)
        
        # Verify context switching worked
        projects_handled = set(task["project"] for task in project_results)
        assert len(projects_handled) == 4, "Should have handled 4 different projects"
        
        # Each project should have multiple tasks
        for project in projects_handled:
            project_tasks = [task for task in project_results if task["project"] == project]
            assert len(project_tasks) >= 2, f"Project {project} should have multiple tasks"
        
        # System should maintain performance across context switches
        final_metrics = await agent.get_performance_metrics()
        assert final_metrics["queries_processed"] == len(multi_project_tasks)
        assert final_metrics["successful_queries"] == len(multi_project_tasks)


@pytest.mark.e2e
class TestUserExperienceScenarios:
    """Test user experience and interaction patterns."""
    
    @pytest.mark.asyncio
    async def test_progressive_complexity_learning(self, patched_claude_flow_expert_agent_dependencies):
        """Test system adaptation to increasing query complexity."""
        config = ClaudeFlowExpertConfig(memory_persistence_enabled=True)
        agent = ClaudeFlowExpertAgent(config)
        deps = ClaudeFlowExpertDependencies()
        
        # Progressive complexity queries
        complexity_progression = [
            # Level 1: Simple queries
            {"query": "What is Python?", "complexity": "simple"},
            {"query": "How to create a Python function?", "complexity": "simple"},
            
            # Level 2: Intermediate queries
            {"query": "How to implement error handling in Python applications?", "complexity": "intermediate"},
            {"query": "Design a simple REST API with proper structure", "complexity": "intermediate", "max_agents": 2},
            
            # Level 3: Complex queries
            {"query": "Architect a scalable microservices system with proper monitoring", "complexity": "complex", "max_agents": 4},
            {"query": "Design a comprehensive CI/CD pipeline with testing and deployment automation", "complexity": "complex", "max_agents": 3},
            
            # Level 4: Expert queries
            {"query": "Design a distributed system architecture with multi-region deployment, auto-scaling, and disaster recovery", "complexity": "expert", "max_agents": 5},
            {"query": "Create a comprehensive security framework for a multi-tenant SaaS platform with compliance requirements", "complexity": "expert", "max_agents": 4}
        ]
        
        progression_results = []
        
        for i, query_info in enumerate(complexity_progression):
            request = QueryRequest(
                query=query_info["query"],
                max_agents=query_info.get("max_agents", 1),
                context={
                    "complexity_level": query_info["complexity"],
                    "progression_step": i + 1,
                    "user_expertise": "growing"
                }
            )
            
            result = await agent.process_query(request, deps)
            assert result.success is True, f"Complexity progression step {i+1} failed"
            
            progression_results.append({
                "step": i + 1,
                "complexity": query_info["complexity"],
                "result": result
            })
        
        # Verify progressive complexity handling
        assert len(progression_results) == len(complexity_progression)
        
        # More complex queries should use more agents
        simple_queries = [r for r in progression_results if r["complexity"] == "simple"]
        expert_queries = [r for r in progression_results if r["complexity"] == "expert"]
        
        avg_agents_simple = sum(len(r["result"].agents_used) for r in simple_queries) / len(simple_queries)
        avg_agents_expert = sum(len(r["result"].agents_used) for r in expert_queries) / len(expert_queries)
        
        assert avg_agents_expert > avg_agents_simple, "Expert queries should use more agents"
        
        # Processing time may increase with complexity but should remain reasonable
        processing_times = [r["result"].processing_time for r in progression_results if r["result"].processing_time]
        if processing_times:
            max_processing_time = max(processing_times)
            assert max_processing_time < 10.0, "Even complex queries should complete in reasonable time"


@pytest.mark.e2e
class TestSystemIntegrationValidation:
    """Validate complete system integration and behavior."""
    
    @pytest.mark.asyncio
    async def test_comprehensive_system_validation(self, patched_claude_flow_expert_agent_dependencies):
        """Comprehensive validation of all system components working together."""
        mcp_client = patched_claude_flow_expert_agent_dependencies["mcp_client"]
        coordinator = patched_claude_flow_expert_agent_dependencies["coordinator"]
        fallback_manager = patched_claude_flow_expert_agent_dependencies["fallback"]
        capabilities = patched_claude_flow_expert_agent_dependencies["capabilities"]
        
        # Configure comprehensive test responses
        validation_responses = {
            "system integration testing": {"success": True, "results": [{"content": "Integration testing guidance"}]},
            "end to end testing": {"success": True, "results": [{"content": "E2E testing strategies"}]},
            "performance monitoring": {"success": True, "results": [{"content": "Monitoring best practices"}]}
        }
        
        for query, response in validation_responses.items():
            mcp_client.set_rag_response(query, response)
        
        config = ClaudeFlowExpertConfig(
            rag_enabled=True,
            max_coordinated_agents=10,
            memory_persistence_enabled=True,
            enable_metrics=True,
            circuit_breaker_enabled=True
        )
        agent = ClaudeFlowExpertAgent(config)
        deps = ClaudeFlowExpertDependencies()
        
        # Comprehensive validation workflow
        validation_tests = [
            # Test all processing strategies
            {"query": "Simple validation query", "expected_strategy": "single_agent"},
            {"query": "Research system integration testing best practices", "require_rag": True, "expected_strategy": "rag_enhanced"},
            {"query": "Coordinate comprehensive system testing across multiple teams", "max_agents": 5, "expected_strategy": "multi_agent"},
            {"query": "Design and implement a complete end-to-end testing framework with comprehensive documentation and team coordination", "max_agents": 4, "expected_strategy": "hybrid"},
            
            # Test error handling and resilience
            {"query": "Handle potential system failures gracefully", "test_resilience": True},
            
            # Test performance under load
            {"query": "Performance test query 1", "load_test": True},
            {"query": "Performance test query 2", "load_test": True},
            {"query": "Performance test query 3", "load_test": True},
            
            # Test memory and context management
            {"query": "Query with extensive context", "context": {"large_context": "x" * 1000}, "test_memory": True},
            
            # Test fallback mechanisms
            {"query": "Query that may require fallback", "require_rag": True, "test_fallback": True}
        ]
        
        validation_results = []
        total_validation_start = time.time()
        
        for test in validation_tests:
            test_start = time.time()
            
            # Configure test conditions
            if test.get("test_resilience"):
                # Temporarily introduce some instability
                mcp_client.failure_rate = 0.3
                coordinator.coordination_delay = 0.2
            elif test.get("test_fallback"):
                # Force fallback activation
                mcp_client.should_fail = True
            else:
                # Normal operation
                mcp_client.failure_rate = 0.0
                mcp_client.should_fail = False
                coordinator.should_fail = False
            
            request = QueryRequest(
                query=test["query"],
                require_rag=test.get("require_rag", False),
                max_agents=test.get("max_agents", 1),
                context=test.get("context", {})
            )
            
            result = await agent.process_query(request, deps)
            test_end = time.time()
            
            # Reset conditions
            mcp_client.failure_rate = 0.0
            mcp_client.should_fail = False
            coordinator.should_fail = False
            
            validation_results.append({
                "test": test,
                "result": result,
                "processing_time": test_end - test_start,
                "success": result.success
            })
        
        total_validation_time = time.time() - total_validation_start
        
        # Comprehensive validation assertions
        assert len(validation_results) == len(validation_tests)
        
        # All tests should succeed (system should handle all scenarios)
        successful_tests = [r for r in validation_results if r["success"]]
        success_rate = len(successful_tests) / len(validation_results)
        assert success_rate >= 0.9, f"System validation success rate: {success_rate:.2f}"
        
        # Test processing strategy selection
        strategy_results = {}
        for result in validation_results:
            if "expected_strategy" in result["test"]:
                expected = result["test"]["expected_strategy"]
                actual = result["result"].processing_strategy.value if result["result"].processing_strategy else "unknown"
                strategy_results[expected] = actual
        
        # At least some strategy matching should occur
        assert len(strategy_results) > 0, "Should have tested strategy selection"
        
        # Performance validation
        processing_times = [r["processing_time"] for r in validation_results]
        avg_processing_time = sum(processing_times) / len(processing_times)
        max_processing_time = max(processing_times)
        
        assert avg_processing_time < 3.0, f"Average processing time too high: {avg_processing_time:.2f}s"
        assert max_processing_time < 10.0, f"Maximum processing time too high: {max_processing_time:.2f}s"
        assert total_validation_time < 60.0, f"Total validation time too high: {total_validation_time:.2f}s"
        
        # Component utilization validation
        assert mcp_client.call_count > 0, "Should have used RAG functionality"
        assert coordinator.call_count > 0, "Should have used coordination functionality"
        
        # System metrics validation
        final_metrics = await agent.get_performance_metrics()
        assert final_metrics["queries_processed"] == len(validation_tests)
        assert final_metrics["successful_queries"] >= len(validation_tests) * 0.9
        assert final_metrics["average_processing_time"] > 0
        
        # Memory and resource validation
        assert "timestamp" in final_metrics
        assert "circuit_breaker_states" in final_metrics
        assert "config" in final_metrics