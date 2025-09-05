"""
Coordination protocol tests for Serena Master Agent communication.

Tests agent communication protocols, message routing, error handling,
hooks execution, and coordination state management.
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Dict, Any, List, Optional, Callable

from tests.fixtures.serena.test_fixtures import (
    SerenaTestData,
    serena_coordination_messages,
    mock_claude_flow_coordination
)
from tests.mocks.serena.mock_serena_tools import (
    MockSerenaTools,
    MockClaudeFlowCoordination,
    create_performance_monitor
)


class TestCoordinationProtocols:
    """Test basic coordination protocol implementations."""
    
    @pytest.mark.asyncio
    async def test_message_routing_and_delivery(self, mock_claude_flow_coordination):
        """Test message routing between agents and delivery confirmation."""
        mock_coordination = MockClaudeFlowCoordination()
        
        # Create agent network
        agents = {}
        agent_types = [
            ("serena_master", ["coordination", "semantic_analysis"]),
            ("coder", ["code_generation", "refactoring"]),
            ("reviewer", ["code_review", "quality_analysis"]), 
            ("tester", ["test_generation", "validation"]),
            ("performance_analyst", ["performance_monitoring", "optimization"])
        ]
        
        for agent_type, capabilities in agent_types:
            agent = await mock_coordination.agent_spawn(agent_type, capabilities)
            agents[agent_type] = agent["agent_id"]
        
        # Test direct message routing
        semantic_context = {
            "type": "semantic_context",
            "action": "provide_analysis",
            "context": {
                "target_function": "calculate_fibonacci",
                "complexity": "exponential",
                "optimization_hints": ["memoization", "iterative_approach"],
                "dependencies": [],
                "performance_impact": "high"
            },
            "priority": "high",
            "requires_response": True,
            "timeout_ms": 30000
        }
        
        # Serena sends context to coder
        message_1 = await mock_coordination.send_coordination_message(
            agents["serena_master"],
            agents["coder"],
            semantic_context
        )
        
        assert message_1["status"] == "sent"
        message_1_id = message_1["message_id"]
        
        # Coder responds with implementation plan
        implementation_plan = {
            "type": "implementation_response",
            "action": "plan_created",
            "in_response_to": message_1_id,
            "plan": {
                "approach": "iterative_with_memoization",
                "estimated_time": "20_minutes",
                "dependencies": ["math_utils", "performance_decorators"],
                "test_requirements": ["unit_tests", "performance_benchmarks"]
            }
        }
        
        message_2 = await mock_coordination.send_coordination_message(
            agents["coder"],
            agents["serena_master"],
            implementation_plan
        )
        
        assert message_2["status"] == "sent"
        
        # Test broadcast message (serena to multiple agents)
        project_update = {
            "type": "project_broadcast",
            "action": "milestone_update",
            "milestone": "optimization_phase_complete",
            "results": {
                "functions_optimized": 5,
                "performance_improvement": "3.2x",
                "test_coverage": 94
            },
            "next_phase": "documentation_and_review"
        }
        
        # Send to reviewer and tester
        broadcast_targets = [agents["reviewer"], agents["tester"], agents["performance_analyst"]]
        broadcast_messages = []
        
        for target in broadcast_targets:
            msg = await mock_coordination.send_coordination_message(
                agents["serena_master"],
                target,
                project_update
            )
            broadcast_messages.append(msg)
        
        # Verify all broadcast messages were sent
        assert len(broadcast_messages) == 3
        assert all(msg["status"] == "sent" for msg in broadcast_messages)
        
        # Verify message queue contains all messages
        all_messages = mock_coordination.message_queue
        assert len(all_messages) == 5  # 2 direct + 3 broadcast
        
        # Verify message content and routing
        serena_to_coder = [msg for msg in all_messages 
                          if msg["from_agent"] == agents["serena_master"] 
                          and msg["to_agent"] == agents["coder"]][0]
        assert serena_to_coder["message"]["type"] == "semantic_context"
        assert serena_to_coder["message"]["context"]["target_function"] == "calculate_fibonacci"
        
        coder_to_serena = [msg for msg in all_messages
                          if msg["from_agent"] == agents["coder"]
                          and msg["to_agent"] == agents["serena_master"]][0]
        assert coder_to_serena["message"]["type"] == "implementation_response"
        assert coder_to_serena["message"]["in_response_to"] == message_1_id
        
        # Verify broadcast messages
        broadcast_msgs = [msg for msg in all_messages 
                         if msg["message"]["type"] == "project_broadcast"]
        assert len(broadcast_msgs) == 3
        assert all(msg["from_agent"] == agents["serena_master"] for msg in broadcast_msgs)
    
    @pytest.mark.asyncio
    async def test_message_acknowledgment_and_timeout(self, mock_claude_flow_coordination):
        """Test message acknowledgment handling and timeout scenarios."""
        mock_coordination = MockClaudeFlowCoordination()
        
        # Create agents
        serena = await mock_coordination.agent_spawn("serena_master", ["coordination"])
        slow_agent = await mock_coordination.agent_spawn("slow_processor", ["heavy_analysis"])
        
        serena_id = serena["agent_id"]
        slow_id = slow_agent["agent_id"]
        
        # Send message requiring acknowledgment
        complex_task = {
            "type": "complex_analysis_request",
            "action": "deep_code_analysis",
            "requires_ack": True,
            "timeout_ms": 5000,  # 5 second timeout
            "task_details": {
                "files_to_analyze": ["src/complex_algorithm.py", "src/data_structures.py"],
                "analysis_depth": "comprehensive",
                "include_performance_profiling": True,
                "generate_optimization_report": True
            },
            "expected_completion": "15_minutes"
        }
        
        # Send message
        message_result = await mock_coordination.send_coordination_message(
            serena_id,
            slow_id,
            complex_task
        )
        
        assert message_result["status"] == "sent"
        message_id = message_result["message_id"]
        
        # Simulate acknowledgment from slow agent
        ack_message = {
            "type": "acknowledgment",
            "action": "task_received",
            "ack_for": message_id,
            "estimated_completion": "12_minutes",
            "agent_status": "processing",
            "resource_requirements": {
                "cpu_intensive": True,
                "memory_requirements": "high",
                "estimated_duration": "12_minutes"
            }
        }
        
        ack_result = await mock_coordination.send_coordination_message(
            slow_id,
            serena_id,
            ack_message
        )
        
        assert ack_result["status"] == "sent"
        
        # Verify acknowledgment message in queue
        messages = mock_coordination.message_queue
        ack_msg = [msg for msg in messages if msg["message"]["type"] == "acknowledgment"][0]
        assert ack_msg["message"]["ack_for"] == message_id
        assert ack_msg["message"]["estimated_completion"] == "12_minutes"
        
        # Test timeout scenario (simulate no response)
        timeout_task = {
            "type": "urgent_request",
            "action": "immediate_response_needed",
            "requires_ack": True,
            "timeout_ms": 1000,  # Very short timeout
            "urgent": True
        }
        
        timeout_result = await mock_coordination.send_coordination_message(
            serena_id,
            slow_id,
            timeout_task
        )
        
        assert timeout_result["status"] == "sent"
        
        # In a real implementation, timeout handling would be done by the coordination system
        # For testing, we simulate the timeout handling
        
        # Simulate timeout detection and handling
        timeout_handler = {
            "type": "timeout_notification", 
            "action": "message_timeout",
            "original_message_id": timeout_result["message_id"],
            "timeout_duration_ms": 1000,
            "suggested_action": "retry_or_escalate"
        }
        
        # This would normally be sent by the coordination system itself
        timeout_notification = await mock_coordination.send_coordination_message(
            "system",  # System-generated message
            serena_id,
            timeout_handler
        )
        
        assert timeout_notification["status"] == "sent"
    
    @pytest.mark.asyncio
    async def test_priority_based_message_handling(self, mock_claude_flow_coordination):
        """Test priority-based message routing and processing."""
        mock_coordination = MockClaudeFlowCoordination()
        
        # Create agents
        serena = await mock_coordination.agent_spawn("serena_master", ["priority_coordination"])
        worker = await mock_coordination.agent_spawn("worker_agent", ["task_processing"])
        
        serena_id = serena["agent_id"]
        worker_id = worker["agent_id"]
        
        # Send messages with different priorities
        messages_to_send = [
            {
                "priority": "low",
                "type": "routine_check",
                "action": "status_update",
                "content": "Regular status check - no urgency"
            },
            {
                "priority": "critical",
                "type": "urgent_fix",
                "action": "fix_critical_bug",
                "content": "Critical production issue detected",
                "deadline": "immediate"
            },
            {
                "priority": "high",
                "type": "optimization_task",
                "action": "performance_optimization",
                "content": "High-priority performance issue",
                "deadline": "2_hours"
            },
            {
                "priority": "medium",
                "type": "enhancement",
                "action": "feature_improvement",
                "content": "Medium priority enhancement request"
            },
            {
                "priority": "critical",
                "type": "security_alert",
                "action": "security_vulnerability",
                "content": "Security vulnerability discovered",
                "severity": "high"
            }
        ]
        
        # Send all messages
        sent_messages = []
        for msg_data in messages_to_send:
            result = await mock_coordination.send_coordination_message(
                serena_id,
                worker_id,
                msg_data
            )
            sent_messages.append((result["message_id"], msg_data["priority"]))
        
        # Verify all messages were sent
        assert len(sent_messages) == 5
        
        # Verify message queue contains all messages
        queue_messages = mock_coordination.message_queue
        assert len(queue_messages) == 5
        
        # Test priority-based retrieval (would be handled by coordination system)
        # Group messages by priority
        priority_groups = {"critical": [], "high": [], "medium": [], "low": []}
        
        for msg in queue_messages:
            priority = msg["message"]["priority"]
            priority_groups[priority].append(msg)
        
        # Verify priority distribution
        assert len(priority_groups["critical"]) == 2  # urgent_fix + security_alert
        assert len(priority_groups["high"]) == 1     # optimization_task
        assert len(priority_groups["medium"]) == 1   # enhancement
        assert len(priority_groups["low"]) == 1      # routine_check
        
        # Verify critical messages have correct content
        critical_messages = priority_groups["critical"]
        critical_types = [msg["message"]["type"] for msg in critical_messages]
        assert "urgent_fix" in critical_types
        assert "security_alert" in critical_types
        
        security_msg = [msg for msg in critical_messages 
                       if msg["message"]["type"] == "security_alert"][0]
        assert security_msg["message"]["severity"] == "high"
    
    @pytest.mark.asyncio
    async def test_message_filtering_and_routing_rules(self, mock_claude_flow_coordination):
        """Test message filtering and intelligent routing rules."""
        mock_coordination = MockClaudeFlowCoordination()
        
        # Create specialized agents
        agents = {
            "serena": await mock_coordination.agent_spawn("serena_master", ["coordination", "routing"]),
            "security_expert": await mock_coordination.agent_spawn("security_expert", ["security_analysis"]),
            "performance_expert": await mock_coordination.agent_spawn("performance_expert", ["performance_optimization"]),
            "code_quality_expert": await mock_coordination.agent_spawn("code_quality_expert", ["quality_analysis"]),
            "general_coder": await mock_coordination.agent_spawn("general_coder", ["general_coding"])
        }
        
        agent_ids = {name: agent["agent_id"] for name, agent in agents.items()}
        
        # Test routing rules based on message content
        test_messages = [
            {
                "content_type": "security_analysis",
                "keywords": ["vulnerability", "security", "authentication"],
                "message": {
                    "type": "analysis_request",
                    "action": "security_review",
                    "content": "Please review authentication vulnerability in login system",
                    "security_keywords": ["sql_injection", "xss", "authentication"]
                },
                "expected_target": "security_expert"
            },
            {
                "content_type": "performance_optimization",
                "keywords": ["performance", "optimization", "bottleneck"],
                "message": {
                    "type": "analysis_request", 
                    "action": "performance_analysis",
                    "content": "Analyze performance bottleneck in data processing pipeline",
                    "performance_metrics": {"current_latency": "2.5s", "target_latency": "500ms"}
                },
                "expected_target": "performance_expert"
            },
            {
                "content_type": "code_quality",
                "keywords": ["quality", "maintainability", "refactor"],
                "message": {
                    "type": "analysis_request",
                    "action": "quality_review", 
                    "content": "Review code quality and suggest refactoring for maintainability",
                    "quality_focus": ["complexity", "duplication", "naming"]
                },
                "expected_target": "code_quality_expert"
            },
            {
                "content_type": "general_coding",
                "keywords": ["implement", "code", "function"],
                "message": {
                    "type": "coding_request",
                    "action": "implement_feature",
                    "content": "Implement user profile management feature",
                    "requirements": ["CRUD_operations", "validation", "tests"]
                },
                "expected_target": "general_coder"
            }
        ]
        
        # Simulate intelligent routing based on content
        routed_messages = []
        
        for test_msg in test_messages:
            # Serena (as coordination master) analyzes message and routes appropriately
            target_agent = agent_ids[test_msg["expected_target"]]
            
            # Add routing metadata
            enhanced_message = test_msg["message"].copy()
            enhanced_message["routing_info"] = {
                "routed_by": agent_ids["serena"],
                "routing_reason": f"content_analysis_matched_{test_msg['content_type']}",
                "keywords_matched": test_msg["keywords"],
                "routing_confidence": 0.95
            }
            
            result = await mock_coordination.send_coordination_message(
                agent_ids["serena"],
                target_agent,
                enhanced_message
            )
            
            routed_messages.append({
                "message_id": result["message_id"],
                "target_agent": test_msg["expected_target"],
                "content_type": test_msg["content_type"]
            })
        
        # Verify routing was successful
        assert len(routed_messages) == 4
        
        # Verify messages were routed to correct agents
        queue_messages = mock_coordination.message_queue
        assert len(queue_messages) == 4
        
        # Check each routing
        for routed_msg in routed_messages:
            matching_queue_msg = [msg for msg in queue_messages 
                                 if msg["id"] == routed_msg["message_id"]][0]
            
            # Verify routing metadata was added
            assert "routing_info" in matching_queue_msg["message"]
            routing_info = matching_queue_msg["message"]["routing_info"]
            assert routing_info["routed_by"] == agent_ids["serena"]
            assert routing_info["routing_confidence"] == 0.95
            
            # Verify content-based routing
            if routed_msg["content_type"] == "security_analysis":
                assert matching_queue_msg["to_agent"] == agent_ids["security_expert"]
            elif routed_msg["content_type"] == "performance_optimization":
                assert matching_queue_msg["to_agent"] == agent_ids["performance_expert"]
            elif routed_msg["content_type"] == "code_quality":
                assert matching_queue_msg["to_agent"] == agent_ids["code_quality_expert"]
            elif routed_msg["content_type"] == "general_coding":
                assert matching_queue_msg["to_agent"] == agent_ids["general_coder"]


class TestCoordinationStateManagement:
    """Test coordination state management and persistence."""
    
    @pytest.mark.asyncio
    async def test_coordination_state_tracking(self, mock_claude_flow_coordination):
        """Test tracking and updating coordination state."""
        mock_coordination = MockClaudeFlowCoordination()
        
        # Initialize project coordination state
        initial_state = {
            "project_id": "optimization_project",
            "phase": "analysis",
            "active_agents": [],
            "completed_tasks": [],
            "pending_tasks": [
                "semantic_analysis",
                "performance_profiling", 
                "optimization_planning",
                "implementation",
                "testing",
                "documentation"
            ],
            "coordination_history": [],
            "state_version": 1
        }
        
        await mock_coordination.memory_store("coordination/project_state", initial_state)
        
        # Spawn agents and update state
        serena = await mock_coordination.agent_spawn("serena_master", ["coordination"])
        analyzer = await mock_coordination.agent_spawn("performance_analyzer", ["analysis"])
        coder = await mock_coordination.agent_spawn("coder", ["implementation"])
        
        # Update state with active agents
        state_update_1 = initial_state.copy()
        state_update_1["active_agents"] = [serena["agent_id"], analyzer["agent_id"], coder["agent_id"]]
        state_update_1["phase"] = "active_analysis"
        state_update_1["coordination_history"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "agents_spawned",
            "agents": [serena["agent_id"], analyzer["agent_id"], coder["agent_id"]]
        })
        state_update_1["state_version"] = 2
        
        await mock_coordination.memory_store("coordination/project_state", state_update_1)
        
        # Simulate task completion and state updates
        await mock_coordination.send_coordination_message(
            analyzer["agent_id"],
            serena["agent_id"],
            {
                "type": "task_completion",
                "action": "task_completed",
                "task": "semantic_analysis",
                "results": {
                    "functions_analyzed": 15,
                    "complexity_score": 7.2,
                    "optimization_opportunities": 8
                },
                "next_recommended_task": "performance_profiling"
            }
        )
        
        # Update coordination state
        state_update_2 = state_update_1.copy()
        state_update_2["completed_tasks"].append("semantic_analysis")
        state_update_2["pending_tasks"].remove("semantic_analysis")
        state_update_2["current_task"] = "performance_profiling"
        state_update_2["coordination_history"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "task_completed",
            "task": "semantic_analysis",
            "completed_by": analyzer["agent_id"],
            "results_summary": "15_functions_analyzed"
        })
        state_update_2["state_version"] = 3
        
        await mock_coordination.memory_store("coordination/project_state", state_update_2)
        
        # Verify state tracking
        current_state = await mock_coordination.memory_retrieve("coordination/project_state")
        assert current_state["status"] == "retrieved"
        
        state_data = current_state["value"]
        assert state_data["state_version"] == 3
        assert len(state_data["active_agents"]) == 3
        assert "semantic_analysis" in state_data["completed_tasks"]
        assert "semantic_analysis" not in state_data["pending_tasks"]
        assert state_data["current_task"] == "performance_profiling"
        assert len(state_data["coordination_history"]) == 2
        
        # Test state rollback capability
        previous_state_data = state_update_1.copy()
        await mock_coordination.memory_store("coordination/project_state_v2", previous_state_data)
        
        rollback_state = await mock_coordination.memory_retrieve("coordination/project_state_v2")
        assert rollback_state["status"] == "retrieved"
        assert rollback_state["value"]["state_version"] == 2
        assert "semantic_analysis" not in rollback_state["value"]["completed_tasks"]
    
    @pytest.mark.asyncio
    async def test_coordination_conflict_resolution(self, mock_claude_flow_coordination):
        """Test conflict resolution in coordination scenarios."""
        mock_coordination = MockClaudeFlowCoordination()
        
        # Create agents that might have conflicting views
        serena = await mock_coordination.agent_spawn("serena_master", ["coordination"])
        performance_expert = await mock_coordination.agent_spawn("performance_expert", ["performance"])
        security_expert = await mock_coordination.agent_spawn("security_expert", ["security"])
        
        serena_id = serena["agent_id"]
        perf_id = performance_expert["agent_id"]
        sec_id = security_expert["agent_id"]
        
        # Scenario: Conflicting optimization recommendations
        # Performance expert recommends aggressive caching
        perf_recommendation = {
            "type": "optimization_recommendation",
            "action": "implement_optimization",
            "recommendation": {
                "strategy": "aggressive_caching",
                "expected_improvement": "80%_performance_gain",
                "implementation_complexity": "medium",
                "memory_impact": "high",
                "security_implications": "minimal"
            },
            "priority": "high",
            "confidence": 0.9
        }
        
        await mock_coordination.send_coordination_message(perf_id, serena_id, perf_recommendation)
        
        # Security expert raises concerns about caching strategy
        security_concern = {
            "type": "security_concern",
            "action": "flag_security_issue",
            "regarding": "aggressive_caching_proposal",
            "concerns": {
                "data_exposure": "cached_sensitive_data_risk",
                "access_control": "cache_bypasses_authorization",
                "compliance": "GDPR_data_retention_issues"
            },
            "recommendation": {
                "strategy": "selective_caching_with_encryption",
                "security_measures": ["data_classification", "encrypted_cache", "TTL_management"],
                "performance_impact": "moderate_reduction"
            },
            "priority": "critical",
            "confidence": 0.85
        }
        
        await mock_coordination.send_coordination_message(sec_id, serena_id, security_concern)
        
        # Serena coordinates conflict resolution
        conflict_analysis = {
            "conflict_id": str(uuid.uuid4()),
            "conflicting_recommendations": [
                {"agent": perf_id, "recommendation": "aggressive_caching", "priority": "high"},
                {"agent": sec_id, "recommendation": "selective_caching", "priority": "critical"}
            ],
            "conflict_type": "priority_vs_security",
            "resolution_strategy": "compromise_solution",
            "proposed_resolution": {
                "strategy": "encrypted_selective_caching",
                "performance_gain": "60%",  # Reduced from 80%
                "security_compliance": "full",
                "implementation_complexity": "high",  # Increased from medium
                "trade_offs_accepted": ["reduced_performance_gain", "increased_complexity"]
            }
        }
        
        await mock_coordination.memory_store("conflicts/caching_strategy", conflict_analysis)
        
        # Send compromise solution to both experts for validation
        compromise_proposal = {
            "type": "compromise_proposal",
            "action": "validate_compromise",
            "conflict_id": conflict_analysis["conflict_id"],
            "proposed_solution": conflict_analysis["proposed_resolution"],
            "requires_approval": True,
            "approval_timeout": "24_hours"
        }
        
        # Send to both experts
        perf_response = await mock_coordination.send_coordination_message(
            serena_id, perf_id, compromise_proposal
        )
        sec_response = await mock_coordination.send_coordination_message(
            serena_id, sec_id, compromise_proposal
        )
        
        # Simulate expert approvals
        perf_approval = {
            "type": "compromise_approval",
            "action": "approve_compromise",
            "conflict_id": conflict_analysis["conflict_id"],
            "approval_status": "approved",
            "conditions": ["performance_monitoring", "benchmarking_required"],
            "expert_domain": "performance"
        }
        
        sec_approval = {
            "type": "compromise_approval",
            "action": "approve_compromise", 
            "conflict_id": conflict_analysis["conflict_id"],
            "approval_status": "approved",
            "conditions": ["security_audit", "compliance_verification"],
            "expert_domain": "security"
        }
        
        await mock_coordination.send_coordination_message(perf_id, serena_id, perf_approval)
        await mock_coordination.send_coordination_message(sec_id, serena_id, sec_approval)
        
        # Verify conflict resolution workflow
        all_messages = mock_coordination.message_queue
        
        # Count message types
        recommendations = [msg for msg in all_messages if msg["message"]["type"] == "optimization_recommendation"]
        concerns = [msg for msg in all_messages if msg["message"]["type"] == "security_concern"]
        proposals = [msg for msg in all_messages if msg["message"]["type"] == "compromise_proposal"]
        approvals = [msg for msg in all_messages if msg["message"]["type"] == "compromise_approval"]
        
        assert len(recommendations) == 1
        assert len(concerns) == 1
        assert len(proposals) == 2  # Sent to both experts
        assert len(approvals) == 2   # Both experts approved
        
        # Verify conflict was stored and managed
        conflict_data = await mock_coordination.memory_retrieve("conflicts/caching_strategy")
        assert conflict_data["status"] == "retrieved"
        assert conflict_data["value"]["conflict_type"] == "priority_vs_security"
        assert conflict_data["value"]["resolution_strategy"] == "compromise_solution"
    
    @pytest.mark.asyncio
    async def test_coordination_escalation_procedures(self, mock_claude_flow_coordination):
        """Test escalation procedures when coordination fails."""
        mock_coordination = MockClaudeFlowCoordination()
        
        # Create agent hierarchy
        serena = await mock_coordination.agent_spawn("serena_master", ["coordination", "escalation"])
        supervisor = await mock_coordination.agent_spawn("supervisor_agent", ["supervision", "decision_making"])
        worker1 = await mock_coordination.agent_spawn("worker_1", ["task_execution"])
        worker2 = await mock_coordination.agent_spawn("worker_2", ["task_execution"]) 
        
        serena_id = serena["agent_id"]
        supervisor_id = supervisor["agent_id"]
        worker1_id = worker1["agent_id"]
        worker2_id = worker2["agent_id"]
        
        # Establish escalation hierarchy
        escalation_hierarchy = {
            "levels": [
                {"level": 1, "agents": [worker1_id, worker2_id], "escalate_to": serena_id},
                {"level": 2, "agents": [serena_id], "escalate_to": supervisor_id},
                {"level": 3, "agents": [supervisor_id], "escalate_to": "external_intervention"}
            ],
            "escalation_triggers": [
                "task_timeout",
                "resource_unavailable", 
                "coordination_deadlock",
                "conflicting_priorities",
                "agent_unresponsive"
            ],
            "escalation_thresholds": {
                "timeout_ms": 10000,
                "retry_attempts": 3,
                "silence_period_ms": 30000
            }
        }
        
        await mock_coordination.memory_store("coordination/escalation_hierarchy", escalation_hierarchy)
        
        # Scenario 1: Task timeout escalation
        complex_task = {
            "type": "complex_analysis",
            "action": "perform_analysis",
            "task_id": str(uuid.uuid4()),
            "timeout_ms": 5000,
            "max_retries": 2,
            "escalate_on_timeout": True
        }
        
        # Send task to worker1
        task_message = await mock_coordination.send_coordination_message(
            serena_id, worker1_id, complex_task
        )
        
        # Simulate timeout (no response from worker1)
        # In real system, this would be detected by timeout monitoring
        timeout_event = {
            "type": "escalation_event",
            "action": "escalate_timeout",
            "original_task_id": complex_task["task_id"],
            "failed_agent": worker1_id,
            "escalation_reason": "task_timeout",
            "escalation_level": 2,  # Escalate to serena (level 2)
            "attempted_retries": 2,
            "escalation_timestamp": datetime.now().isoformat()
        }
        
        escalation_message = await mock_coordination.send_coordination_message(
            "system",  # System-generated escalation
            serena_id,
            timeout_event
        )
        
        # Serena handles escalation by reassigning task
        task_reassignment = {
            "type": "task_reassignment",
            "action": "reassign_failed_task",
            "original_task_id": complex_task["task_id"],
            "failed_agent": worker1_id,
            "new_agent": worker2_id,
            "escalation_response": True,
            "modified_requirements": {
                "timeout_ms": 10000,  # Extended timeout
                "priority": "high",    # Increased priority
                "additional_resources": True
            }
        }
        
        reassignment_message = await mock_coordination.send_coordination_message(
            serena_id, worker2_id, task_reassignment
        )
        
        # Scenario 2: Further escalation to supervisor
        # Suppose worker2 also fails
        second_escalation = {
            "type": "escalation_event",
            "action": "escalate_to_supervisor",
            "original_task_id": complex_task["task_id"],
            "failed_agents": [worker1_id, worker2_id],
            "escalation_reason": "multiple_agent_failures",
            "escalation_level": 3,  # Escalate to supervisor
            "escalation_history": [
                {"level": 2, "agent": worker1_id, "reason": "timeout"},
                {"level": 2, "agent": worker2_id, "reason": "resource_unavailable"}
            ]
        }
        
        supervisor_escalation = await mock_coordination.send_coordination_message(
            serena_id, supervisor_id, second_escalation
        )
        
        # Supervisor makes executive decision
        supervisor_decision = {
            "type": "executive_decision",
            "action": "override_and_resolve", 
            "decision_id": str(uuid.uuid4()),
            "original_task_id": complex_task["task_id"],
            "resolution": {
                "approach": "simplified_task_breakdown",
                "resource_allocation": "additional_compute_resources",
                "timeline_adjustment": "extend_by_4_hours",
                "success_criteria": "reduced_scope_acceptable"
            },
            "authority_level": "supervisor",
            "final_decision": True
        }
        
        final_decision = await mock_coordination.send_coordination_message(
            supervisor_id, serena_id, supervisor_decision
        )
        
        # Verify escalation chain
        all_messages = mock_coordination.message_queue
        
        escalation_messages = [msg for msg in all_messages 
                             if msg["message"]["type"] == "escalation_event"]
        assert len(escalation_messages) == 2
        
        reassignment_messages = [msg for msg in all_messages
                               if msg["message"]["type"] == "task_reassignment"]
        assert len(reassignment_messages) == 1
        
        executive_decisions = [msg for msg in all_messages
                             if msg["message"]["type"] == "executive_decision"]
        assert len(executive_decisions) == 1
        
        # Verify escalation levels were followed
        first_escalation = escalation_messages[0]
        second_escalation_msg = escalation_messages[1]
        
        assert first_escalation["message"]["escalation_level"] == 2
        assert second_escalation_msg["message"]["escalation_level"] == 3
        assert executive_decisions[0]["from_agent"] == supervisor_id