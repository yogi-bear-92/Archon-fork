#!/usr/bin/env python3
"""
Phase 3 Multi-Swarm Coordination System
Enterprise-scale hierarchical orchestration with intelligent load balancing

Features:
- Hierarchical Swarm Management (Central → Regional → Local)
- Cross-Swarm Communication Protocol
- Distributed Task Management with Intelligent Routing
- Dynamic Load Balancing and Auto-Scaling
- Global State Synchronization
- Fault Tolerance and Recovery
- Performance Optimization Across Swarms

Architecture:
┌─────────────────────────────────────────────────────────┐
│                GLOBAL ORCHESTRATOR                      │
│                (Master Controller)                      │
├─────────────────────────────────────────────────────────┤
│    REGIONAL COORDINATORS (Domain-Specific)             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      │
│  │   AI/ML     │ │  Backend    │ │  Frontend   │      │
│  │ Coordinator │ │ Coordinator │ │ Coordinator │      │
│  └─────────────┘ └─────────────┘ └─────────────┘      │
├─────────────────────────────────────────────────────────┤
│              LOCAL SWARMS (Task Execution)             │
│ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌──────────┐│
│ │ Swarm A   │ │ Swarm B   │ │ Swarm C   │ │ Swarm D  ││
│ │(Research) │ │(Coding)   │ │(Testing)  │ │(Deploy)  ││
│ └───────────┘ └───────────┘ └───────────┘ └──────────┘│
└─────────────────────────────────────────────────────────┘

Author: Claude Code Multi-Swarm Team
Target: Enterprise-grade coordination with 99.2% reliability
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import weakref
import sys
from pathlib import Path

# Add project paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

class SwarmPriority(Enum):
    """Swarm priority levels for task routing."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5

class TaskType(Enum):
    """Task types for intelligent routing."""
    RESEARCH = "research"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    OPTIMIZATION = "optimization"
    COORDINATION = "coordination"
    ANALYSIS = "analysis"

class SwarmState(Enum):
    """Swarm operational states."""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class SwarmMetrics:
    """Swarm performance and health metrics."""
    swarm_id: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    task_queue_length: int = 0
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_response_time: float = 0.0
    success_rate: float = 1.0
    last_activity: datetime = field(default_factory=datetime.now)
    load_score: float = 0.0  # Composite load metric

@dataclass
class TaskRequest:
    """Distributed task request with routing information."""
    task_id: str
    task_type: TaskType
    priority: SwarmPriority
    requirements: Dict[str, Any]
    constraints: Dict[str, Any]
    estimated_duration: int  # seconds
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    preferred_swarms: List[str] = field(default_factory=list)
    excluded_swarms: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SwarmCapability:
    """Swarm capability definition."""
    capability_name: str
    proficiency_score: float  # 0.0 to 1.0
    max_concurrent_tasks: int
    resource_requirements: Dict[str, Any]
    specializations: List[str] = field(default_factory=list)

class SwarmNode:
    """Individual swarm node in the multi-swarm hierarchy."""
    
    def __init__(self, swarm_id: str, swarm_type: str, coordinator_id: Optional[str] = None):
        self.swarm_id = swarm_id
        self.swarm_type = swarm_type
        self.coordinator_id = coordinator_id
        self.state = SwarmState.INITIALIZING
        self.metrics = SwarmMetrics(swarm_id)
        self.capabilities: Dict[str, SwarmCapability] = {}
        self.active_tasks: Dict[str, TaskRequest] = {}
        self.task_queue: deque = deque()
        self.children: List[str] = []  # Child swarm IDs
        self.parent: Optional[str] = None  # Parent swarm ID
        self.last_heartbeat = datetime.now()
        self.configuration = {}
        
    async def process_task(self, task: TaskRequest) -> Dict[str, Any]:
        """Process a task within this swarm."""
        try:
            self.active_tasks[task.task_id] = task
            self.metrics.active_tasks += 1
            
            logger.info(f"🎯 Swarm {self.swarm_id} processing task {task.task_id}")
            
            # Simulate task processing (replace with actual task execution)
            start_time = datetime.now()
            
            # Task-specific processing logic would go here
            await asyncio.sleep(min(task.estimated_duration, 5))  # Simulate work
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            self.metrics.completed_tasks += 1
            self.metrics.active_tasks -= 1
            self.metrics.average_response_time = (
                (self.metrics.average_response_time * (self.metrics.completed_tasks - 1) + processing_time) 
                / self.metrics.completed_tasks
            )
            self.metrics.success_rate = (
                self.metrics.completed_tasks / 
                (self.metrics.completed_tasks + self.metrics.failed_tasks)
            )
            self.metrics.last_activity = datetime.now()
            
            # Remove from active tasks
            del self.active_tasks[task.task_id]
            
            return {
                'success': True,
                'task_id': task.task_id,
                'processing_time': processing_time,
                'swarm_id': self.swarm_id,
                'result': f"Task {task.task_id} completed successfully"
            }
            
        except Exception as e:
            self.metrics.failed_tasks += 1
            self.metrics.active_tasks -= 1
            logger.error(f"❌ Swarm {self.swarm_id} task {task.task_id} failed: {e}")
            
            return {
                'success': False,
                'task_id': task.task_id,
                'error': str(e),
                'swarm_id': self.swarm_id
            }
    
    def calculate_load_score(self) -> float:
        """Calculate composite load score for load balancing."""
        # Weighted load calculation
        cpu_weight = 0.3
        memory_weight = 0.3
        queue_weight = 0.25
        task_weight = 0.15
        
        # Normalize values
        cpu_score = min(self.metrics.cpu_usage / 100.0, 1.0)
        memory_score = min(self.metrics.memory_usage / 100.0, 1.0)
        queue_score = min(self.metrics.task_queue_length / 50.0, 1.0)  # Assume 50 is high
        task_score = min(self.metrics.active_tasks / 20.0, 1.0)  # Assume 20 is high
        
        load_score = (
            cpu_weight * cpu_score +
            memory_weight * memory_score +
            queue_weight * queue_score +
            task_weight * task_score
        )
        
        self.metrics.load_score = load_score
        return load_score
    
    def can_handle_task(self, task: TaskRequest) -> bool:
        """Check if swarm can handle the given task."""
        if self.state not in [SwarmState.ACTIVE]:
            return False
        
        # Check if swarm is overloaded
        if self.metrics.load_score > 0.9:
            return False
        
        # Check task type compatibility
        task_type_str = task.task_type.value
        if task_type_str in self.capabilities:
            capability = self.capabilities[task_type_str]
            if self.metrics.active_tasks >= capability.max_concurrent_tasks:
                return False
        
        # Check constraints
        for constraint_key, constraint_value in task.constraints.items():
            if constraint_key not in self.configuration:
                continue
            if self.configuration[constraint_key] != constraint_value:
                return False
        
        return True
    
    def get_priority_score(self, task: TaskRequest) -> float:
        """Calculate priority score for task assignment."""
        base_score = 1.0 - self.metrics.load_score  # Lower load = higher score
        
        # Task type proficiency bonus
        task_type_str = task.task_type.value
        if task_type_str in self.capabilities:
            proficiency_bonus = self.capabilities[task_type_str].proficiency_score * 0.3
            base_score += proficiency_bonus
        
        # Priority adjustment
        priority_multiplier = {
            SwarmPriority.CRITICAL: 2.0,
            SwarmPriority.HIGH: 1.5,
            SwarmPriority.MEDIUM: 1.0,
            SwarmPriority.LOW: 0.8,
            SwarmPriority.BACKGROUND: 0.5
        }
        
        base_score *= priority_multiplier.get(task.priority, 1.0)
        
        # Success rate bonus
        base_score *= self.metrics.success_rate
        
        return base_score

class RegionalCoordinator:
    """Regional coordinator for domain-specific swarm management."""
    
    def __init__(self, coordinator_id: str, domain: str):
        self.coordinator_id = coordinator_id
        self.domain = domain
        self.managed_swarms: Dict[str, SwarmNode] = {}
        self.task_routing_table: Dict[TaskType, List[str]] = defaultdict(list)
        self.load_balancer_config = {
            'algorithm': 'weighted_round_robin',
            'health_check_interval': 30,
            'failover_threshold': 0.95,
            'auto_scale_threshold': 0.8
        }
        self.metrics_history: deque = deque(maxlen=1000)
        
    async def register_swarm(self, swarm: SwarmNode):
        """Register a swarm with this coordinator."""
        swarm.coordinator_id = self.coordinator_id
        self.managed_swarms[swarm.swarm_id] = swarm
        
        # Update routing table based on swarm capabilities
        for capability_name in swarm.capabilities:
            try:
                task_type = TaskType(capability_name)
                if swarm.swarm_id not in self.task_routing_table[task_type]:
                    self.task_routing_table[task_type].append(swarm.swarm_id)
            except ValueError:
                continue  # Skip unknown task types
        
        logger.info(f"✅ Registered swarm {swarm.swarm_id} with coordinator {self.coordinator_id}")
    
    async def route_task(self, task: TaskRequest) -> Optional[str]:
        """Intelligently route task to best available swarm."""
        # Get candidate swarms for task type
        candidate_swarms = self.task_routing_table.get(task.task_type, [])
        
        if not candidate_swarms:
            logger.warning(f"⚠️ No swarms available for task type {task.task_type}")
            return None
        
        # Filter by preferred swarms if specified
        if task.preferred_swarms:
            candidate_swarms = [s for s in candidate_swarms if s in task.preferred_swarms]
        
        # Filter out excluded swarms
        if task.excluded_swarms:
            candidate_swarms = [s for s in candidate_swarms if s not in task.excluded_swarms]
        
        # Filter by availability and capability
        available_swarms = []
        for swarm_id in candidate_swarms:
            swarm = self.managed_swarms.get(swarm_id)
            if swarm and swarm.can_handle_task(task):
                available_swarms.append(swarm)
        
        if not available_swarms:
            logger.warning(f"⚠️ No available swarms for task {task.task_id}")
            return None
        
        # Select best swarm based on priority score
        best_swarm = max(available_swarms, key=lambda s: s.get_priority_score(task))
        
        logger.info(f"🎯 Routed task {task.task_id} to swarm {best_swarm.swarm_id}")
        return best_swarm.swarm_id
    
    async def balance_load(self):
        """Perform load balancing across managed swarms."""
        overloaded_swarms = [
            swarm for swarm in self.managed_swarms.values()
            if swarm.metrics.load_score > self.load_balancer_config['failover_threshold']
        ]
        
        if overloaded_swarms:
            logger.info(f"⚖️ Load balancing for {len(overloaded_swarms)} overloaded swarms")
            
            for swarm in overloaded_swarms:
                # Move tasks from queue to other swarms
                tasks_to_redistribute = []
                while swarm.task_queue and len(tasks_to_redistribute) < 5:
                    tasks_to_redistribute.append(swarm.task_queue.popleft())
                
                for task in tasks_to_redistribute:
                    new_swarm_id = await self.route_task(task)
                    if new_swarm_id and new_swarm_id != swarm.swarm_id:
                        target_swarm = self.managed_swarms[new_swarm_id]
                        target_swarm.task_queue.append(task)
                        logger.info(f"📋 Redistributed task {task.task_id} from {swarm.swarm_id} to {new_swarm_id}")
                    else:
                        # Put task back if can't redistribute
                        swarm.task_queue.appendleft(task)
    
    def get_coordinator_metrics(self) -> Dict[str, Any]:
        """Get comprehensive coordinator metrics."""
        total_swarms = len(self.managed_swarms)
        active_swarms = sum(1 for s in self.managed_swarms.values() if s.state == SwarmState.ACTIVE)
        total_active_tasks = sum(s.metrics.active_tasks for s in self.managed_swarms.values())
        total_completed_tasks = sum(s.metrics.completed_tasks for s in self.managed_swarms.values())
        average_load = sum(s.metrics.load_score for s in self.managed_swarms.values()) / max(total_swarms, 1)
        
        return {
            'coordinator_id': self.coordinator_id,
            'domain': self.domain,
            'total_swarms': total_swarms,
            'active_swarms': active_swarms,
            'total_active_tasks': total_active_tasks,
            'total_completed_tasks': total_completed_tasks,
            'average_load': average_load,
            'task_routing_table': {k.value: v for k, v in self.task_routing_table.items()}
        }

class GlobalOrchestrator:
    """Global orchestrator for enterprise-wide multi-swarm coordination."""
    
    def __init__(self):
        self.orchestrator_id = str(uuid.uuid4())
        self.regional_coordinators: Dict[str, RegionalCoordinator] = {}
        self.global_task_queue: deque = deque()
        self.active_global_tasks: Dict[str, TaskRequest] = {}
        self.task_dependencies: Dict[str, List[str]] = defaultdict(list)
        self.global_metrics = {
            'total_tasks_processed': 0,
            'total_swarms': 0,
            'total_coordinators': 0,
            'system_efficiency': 0.0,
            'uptime': datetime.now()
        }
        self.configuration = {
            'max_task_retries': 3,
            'task_timeout_seconds': 300,
            'health_check_interval': 60,
            'metrics_collection_interval': 30,
            'auto_scaling_enabled': True,
            'fault_tolerance_level': 'high'
        }
        self._running = False
        self._monitoring_tasks: List[asyncio.Task] = []
        
    async def register_coordinator(self, coordinator: RegionalCoordinator):
        """Register a regional coordinator."""
        self.regional_coordinators[coordinator.coordinator_id] = coordinator
        self.global_metrics['total_coordinators'] = len(self.regional_coordinators)
        
        logger.info(f"✅ Registered coordinator {coordinator.coordinator_id} for domain {coordinator.domain}")
    
    async def submit_global_task(self, task: TaskRequest) -> str:
        """Submit a task to the global orchestrator."""
        self.active_global_tasks[task.task_id] = task
        self.global_task_queue.append(task)
        
        logger.info(f"📋 Global task submitted: {task.task_id} ({task.task_type.value})")
        return task.task_id
    
    async def process_global_tasks(self):
        """Process tasks from the global queue."""
        while self._running:
            if not self.global_task_queue:
                await asyncio.sleep(1)
                continue
            
            task = self.global_task_queue.popleft()
            
            # Check dependencies
            if await self._check_task_dependencies(task):
                await self._execute_distributed_task(task)
            else:
                # Put task back at end of queue
                self.global_task_queue.append(task)
                await asyncio.sleep(2)  # Wait before retry
    
    async def _check_task_dependencies(self, task: TaskRequest) -> bool:
        """Check if task dependencies are satisfied."""
        if not task.dependencies:
            return True
        
        # Check if all dependency tasks are completed
        for dep_task_id in task.dependencies:
            if dep_task_id in self.active_global_tasks:
                return False  # Dependency still active
        
        return True
    
    async def _execute_distributed_task(self, task: TaskRequest):
        """Execute a task across the distributed swarm network."""
        try:
            # Find best coordinator for task
            best_coordinator = await self._select_coordinator(task)
            
            if not best_coordinator:
                logger.error(f"❌ No suitable coordinator found for task {task.task_id}")
                await self._handle_task_failure(task, "No suitable coordinator")
                return
            
            # Route task to coordinator
            swarm_id = await best_coordinator.route_task(task)
            
            if not swarm_id:
                logger.error(f"❌ No suitable swarm found for task {task.task_id}")
                await self._handle_task_failure(task, "No suitable swarm")
                return
            
            # Execute task
            swarm = best_coordinator.managed_swarms[swarm_id]
            result = await swarm.process_task(task)
            
            if result['success']:
                logger.info(f"✅ Global task {task.task_id} completed successfully")
                self.global_metrics['total_tasks_processed'] += 1
                await self._handle_task_completion(task, result)
            else:
                logger.error(f"❌ Global task {task.task_id} failed")
                await self._handle_task_failure(task, result.get('error', 'Unknown error'))
                
        except Exception as e:
            logger.error(f"❌ Error executing global task {task.task_id}: {e}")
            await self._handle_task_failure(task, str(e))
    
    async def _select_coordinator(self, task: TaskRequest) -> Optional[RegionalCoordinator]:
        """Select the best coordinator for a task."""
        if not self.regional_coordinators:
            return None
        
        # Score coordinators based on their capability to handle the task
        coordinator_scores = []
        
        for coordinator in self.regional_coordinators.values():
            score = 0.0
            
            # Check if coordinator has swarms for this task type
            if task.task_type in coordinator.task_routing_table:
                suitable_swarms = coordinator.task_routing_table[task.task_type]
                available_swarms = [
                    coordinator.managed_swarms[swarm_id] 
                    for swarm_id in suitable_swarms 
                    if coordinator.managed_swarms[swarm_id].can_handle_task(task)
                ]
                
                if available_swarms:
                    # Base score from number of available swarms
                    score += len(available_swarms) * 10
                    
                    # Average load score (lower is better)
                    avg_load = sum(s.metrics.load_score for s in available_swarms) / len(available_swarms)
                    score += (1.0 - avg_load) * 50
                    
                    # Success rate bonus
                    avg_success = sum(s.metrics.success_rate for s in available_swarms) / len(available_swarms)
                    score += avg_success * 30
            
            coordinator_scores.append((coordinator, score))
        
        # Select coordinator with highest score
        if coordinator_scores:
            best_coordinator = max(coordinator_scores, key=lambda x: x[1])
            if best_coordinator[1] > 0:
                return best_coordinator[0]
        
        return None
    
    async def _handle_task_completion(self, task: TaskRequest, result: Dict[str, Any]):
        """Handle successful task completion."""
        if task.task_id in self.active_global_tasks:
            del self.active_global_tasks[task.task_id]
        
        # Update global metrics
        self._update_system_efficiency()
        
        logger.info(f"📊 Task {task.task_id} completed - Total processed: {self.global_metrics['total_tasks_processed']}")
    
    async def _handle_task_failure(self, task: TaskRequest, error: str):
        """Handle task failure with retry logic."""
        retry_count = task.context.get('retry_count', 0)
        
        if retry_count < self.configuration['max_task_retries']:
            # Retry task
            task.context['retry_count'] = retry_count + 1
            self.global_task_queue.append(task)
            logger.info(f"🔄 Retrying task {task.task_id} (attempt {retry_count + 1})")
        else:
            # Max retries reached
            if task.task_id in self.active_global_tasks:
                del self.active_global_tasks[task.task_id]
            logger.error(f"❌ Task {task.task_id} failed permanently after {retry_count} retries")
    
    def _update_system_efficiency(self):
        """Update global system efficiency metrics."""
        total_swarms = sum(len(coord.managed_swarms) for coord in self.regional_coordinators.values())
        active_swarms = sum(
            sum(1 for s in coord.managed_swarms.values() if s.state == SwarmState.ACTIVE)
            for coord in self.regional_coordinators.values()
        )
        
        self.global_metrics['total_swarms'] = total_swarms
        self.global_metrics['system_efficiency'] = (active_swarms / max(total_swarms, 1)) * 100
    
    async def start_orchestration(self):
        """Start the global orchestration system."""
        logger.info("🚀 Starting Global Multi-Swarm Orchestrator")
        self._running = True
        
        # Start background monitoring tasks
        self._monitoring_tasks = [
            asyncio.create_task(self.process_global_tasks()),
            asyncio.create_task(self._global_health_monitor()),
            asyncio.create_task(self._metrics_collector()),
            asyncio.create_task(self._load_balancer())
        ]
        
        logger.info("✅ Global orchestrator started successfully")
    
    async def stop_orchestration(self):
        """Stop the global orchestration system."""
        logger.info("🛑 Stopping Global Multi-Swarm Orchestrator")
        self._running = False
        
        # Cancel monitoring tasks
        for task in self._monitoring_tasks:
            task.cancel()
        
        # Wait for tasks to finish
        await asyncio.gather(*self._monitoring_tasks, return_exceptions=True)
        
        logger.info("✅ Global orchestrator stopped successfully")
    
    async def _global_health_monitor(self):
        """Monitor health of all coordinators and swarms."""
        while self._running:
            try:
                for coordinator in self.regional_coordinators.values():
                    for swarm in coordinator.managed_swarms.values():
                        # Check swarm health
                        time_since_activity = (datetime.now() - swarm.last_heartbeat).total_seconds()
                        
                        if time_since_activity > 120:  # 2 minutes timeout
                            if swarm.state != SwarmState.ERROR:
                                swarm.state = SwarmState.ERROR
                                logger.warning(f"⚠️ Swarm {swarm.swarm_id} marked as ERROR - no activity for {time_since_activity:.1f}s")
                        
                        # Update load score
                        swarm.calculate_load_score()
                        
                        # Auto-scale if needed
                        if (self.configuration['auto_scaling_enabled'] and 
                            swarm.metrics.load_score > 0.8 and 
                            swarm.state == SwarmState.ACTIVE):
                            logger.info(f"📈 Auto-scaling recommended for swarm {swarm.swarm_id}")
                
                await asyncio.sleep(self.configuration['health_check_interval'])
                
            except Exception as e:
                logger.error(f"❌ Error in health monitor: {e}")
                await asyncio.sleep(10)
    
    async def _metrics_collector(self):
        """Collect and aggregate metrics from all components."""
        while self._running:
            try:
                # Update system efficiency
                self._update_system_efficiency()
                
                # Log system status
                logger.info(f"📊 System Status - Coordinators: {self.global_metrics['total_coordinators']}, "
                           f"Swarms: {self.global_metrics['total_swarms']}, "
                           f"Efficiency: {self.global_metrics['system_efficiency']:.1f}%")
                
                await asyncio.sleep(self.configuration['metrics_collection_interval'])
                
            except Exception as e:
                logger.error(f"❌ Error in metrics collector: {e}")
                await asyncio.sleep(10)
    
    async def _load_balancer(self):
        """Global load balancing across all coordinators."""
        while self._running:
            try:
                for coordinator in self.regional_coordinators.values():
                    await coordinator.balance_load()
                
                await asyncio.sleep(30)  # Load balance every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error in load balancer: {e}")
                await asyncio.sleep(10)
    
    def get_global_metrics(self) -> Dict[str, Any]:
        """Get comprehensive global metrics."""
        coordinator_metrics = {
            coord_id: coord.get_coordinator_metrics()
            for coord_id, coord in self.regional_coordinators.items()
        }
        
        return {
            'orchestrator_id': self.orchestrator_id,
            'global_metrics': self.global_metrics,
            'active_tasks': len(self.active_global_tasks),
            'queued_tasks': len(self.global_task_queue),
            'coordinators': coordinator_metrics,
            'configuration': self.configuration,
            'uptime_seconds': (datetime.now() - self.global_metrics['uptime']).total_seconds()
        }

# Factory functions for easy swarm creation

def create_research_swarm(swarm_id: str = None) -> SwarmNode:
    """Create a research-specialized swarm."""
    if swarm_id is None:
        swarm_id = f"research_swarm_{uuid.uuid4().hex[:8]}"
    
    swarm = SwarmNode(swarm_id, "research")
    swarm.capabilities["research"] = SwarmCapability("research", 0.9, 5, {"cpu": 2, "memory": "4GB"})
    swarm.capabilities["analysis"] = SwarmCapability("analysis", 0.85, 3, {"cpu": 1, "memory": "2GB"})
    swarm.state = SwarmState.ACTIVE
    
    return swarm

def create_development_swarm(swarm_id: str = None) -> SwarmNode:
    """Create a development-specialized swarm."""
    if swarm_id is None:
        swarm_id = f"dev_swarm_{uuid.uuid4().hex[:8]}"
    
    swarm = SwarmNode(swarm_id, "development")
    swarm.capabilities["development"] = SwarmCapability("development", 0.95, 8, {"cpu": 4, "memory": "8GB"})
    swarm.capabilities["testing"] = SwarmCapability("testing", 0.8, 4, {"cpu": 2, "memory": "4GB"})
    swarm.state = SwarmState.ACTIVE
    
    return swarm

def create_testing_swarm(swarm_id: str = None) -> SwarmNode:
    """Create a testing-specialized swarm."""
    if swarm_id is None:
        swarm_id = f"test_swarm_{uuid.uuid4().hex[:8]}"
    
    swarm = SwarmNode(swarm_id, "testing")
    swarm.capabilities["testing"] = SwarmCapability("testing", 0.95, 6, {"cpu": 3, "memory": "6GB"})
    swarm.capabilities["monitoring"] = SwarmCapability("monitoring", 0.7, 2, {"cpu": 1, "memory": "2GB"})
    swarm.state = SwarmState.ACTIVE
    
    return swarm

def create_deployment_swarm(swarm_id: str = None) -> SwarmNode:
    """Create a deployment-specialized swarm."""
    if swarm_id is None:
        swarm_id = f"deploy_swarm_{uuid.uuid4().hex[:8]}"
    
    swarm = SwarmNode(swarm_id, "deployment")
    swarm.capabilities["deployment"] = SwarmCapability("deployment", 0.9, 4, {"cpu": 2, "memory": "4GB"})
    swarm.capabilities["monitoring"] = SwarmCapability("monitoring", 0.85, 3, {"cpu": 1, "memory": "2GB"})
    swarm.state = SwarmState.ACTIVE
    
    return swarm

# Example usage and testing functions

async def create_example_multi_swarm_system() -> GlobalOrchestrator:
    """Create an example multi-swarm system for demonstration."""
    logger.info("🏗️ Creating example multi-swarm system")
    
    # Create global orchestrator
    orchestrator = GlobalOrchestrator()
    
    # Create regional coordinators
    ai_coordinator = RegionalCoordinator("ai_coord_001", "AI_ML")
    backend_coordinator = RegionalCoordinator("backend_coord_001", "Backend")
    
    # Register coordinators
    await orchestrator.register_coordinator(ai_coordinator)
    await orchestrator.register_coordinator(backend_coordinator)
    
    # Create and register swarms
    research_swarm = create_research_swarm()
    dev_swarm1 = create_development_swarm("dev_swarm_primary")
    dev_swarm2 = create_development_swarm("dev_swarm_secondary")
    test_swarm = create_testing_swarm()
    deploy_swarm = create_deployment_swarm()
    
    # Register swarms with appropriate coordinators
    await ai_coordinator.register_swarm(research_swarm)
    await backend_coordinator.register_swarm(dev_swarm1)
    await backend_coordinator.register_swarm(dev_swarm2)
    await backend_coordinator.register_swarm(test_swarm)
    await backend_coordinator.register_swarm(deploy_swarm)
    
    logger.info("✅ Example multi-swarm system created successfully")
    return orchestrator

if __name__ == "__main__":
    async def main():
        # Create and start multi-swarm system
        orchestrator = await create_example_multi_swarm_system()
        await orchestrator.start_orchestration()
        
        # Create some example tasks
        tasks = [
            TaskRequest(
                task_id="task_001",
                task_type=TaskType.RESEARCH,
                priority=SwarmPriority.HIGH,
                requirements={"accuracy": 0.9},
                constraints={},
                estimated_duration=30
            ),
            TaskRequest(
                task_id="task_002", 
                task_type=TaskType.DEVELOPMENT,
                priority=SwarmPriority.CRITICAL,
                requirements={"language": "python"},
                constraints={},
                estimated_duration=60
            ),
            TaskRequest(
                task_id="task_003",
                task_type=TaskType.TESTING,
                priority=SwarmPriority.MEDIUM,
                requirements={"coverage": 0.8},
                constraints={},
                estimated_duration=45
            )
        ]
        
        # Submit tasks
        for task in tasks:
            await orchestrator.submit_global_task(task)
        
        # Run for demonstration
        print("🚀 Multi-swarm system running... Press Ctrl+C to stop")
        try:
            while True:
                await asyncio.sleep(10)
                metrics = orchestrator.get_global_metrics()
                print(f"📊 System Status: {metrics['global_metrics']['total_tasks_processed']} tasks processed")
        except KeyboardInterrupt:
            print("\n🛑 Stopping multi-swarm system...")
            await orchestrator.stop_orchestration()

    asyncio.run(main())