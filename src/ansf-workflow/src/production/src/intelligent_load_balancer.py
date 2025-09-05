#!/usr/bin/env python3
"""
Intelligent Load Balancer
Mock implementation for testing load balancing across swarms
"""

import asyncio
import random
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

class LoadBalancingAlgorithm(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"  
    WEIGHTED = "weighted"
    HYBRID = "hybrid"

class SwarmHealthState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"

@dataclass
class SwarmEndpoint:
    swarm_id: str
    endpoint: str
    region: str
    weight: float = 1.0
    max_capacity: int = 100
    health_state: SwarmHealthState = SwarmHealthState.HEALTHY
    active_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_health_check: datetime = field(default_factory=datetime.now)

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def record_success(self):
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
    
    def can_attempt_request(self) -> bool:
        if self.state == "closed":
            return True
        elif self.state == "open":
            if self.last_failure_time:
                time_since_failure = (datetime.now() - self.last_failure_time).seconds
                if time_since_failure >= self.recovery_timeout:
                    self.state = "half_open"
                    return True
            return False
        elif self.state == "half_open":
            return True
        
        return False

class IntelligentLoadBalancer:
    def __init__(self, algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN):
        self.algorithm = algorithm
        self.swarm_endpoints: Dict[str, SwarmEndpoint] = {}
        self.regional_endpoints: Dict[str, List[str]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.load_balancing_weights: Dict[str, float] = {}
        self.round_robin_index = 0
        self.performance_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0
        }
    
    def register_swarm(self, endpoint: SwarmEndpoint):
        """Register a swarm endpoint"""
        self.swarm_endpoints[endpoint.swarm_id] = endpoint
        
        # Add to regional index
        if endpoint.region not in self.regional_endpoints:
            self.regional_endpoints[endpoint.region] = []
        self.regional_endpoints[endpoint.region].append(endpoint.swarm_id)
        
        # Initialize circuit breaker
        self.circuit_breakers[endpoint.swarm_id] = CircuitBreaker()
        
        # Set load balancing weight
        self.load_balancing_weights[endpoint.swarm_id] = endpoint.weight
        
        logger.info(f"Registered swarm {endpoint.swarm_id} in region {endpoint.region}")
    
    def unregister_swarm(self, swarm_id: str):
        """Unregister a swarm endpoint"""
        if swarm_id in self.swarm_endpoints:
            endpoint = self.swarm_endpoints[swarm_id]
            
            # Remove from regional index
            if endpoint.region in self.regional_endpoints:
                if swarm_id in self.regional_endpoints[endpoint.region]:
                    self.regional_endpoints[endpoint.region].remove(swarm_id)
            
            # Clean up
            del self.swarm_endpoints[swarm_id]
            del self.circuit_breakers[swarm_id]
            del self.load_balancing_weights[swarm_id]
            
            logger.info(f"Unregistered swarm {swarm_id}")
    
    async def route_request(self, region_preference: Optional[str] = None) -> Optional[str]:
        """Route request to optimal swarm"""
        available_swarms = self._get_available_swarms(region_preference)
        
        if not available_swarms:
            logger.warning("No available swarms for request routing")
            return None
        
        selected_swarm = None
        
        if self.algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            selected_swarm = self._route_round_robin(available_swarms)
        elif self.algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            selected_swarm = self._route_least_connections(available_swarms)
        elif self.algorithm == LoadBalancingAlgorithm.WEIGHTED:
            selected_swarm = self._route_weighted(available_swarms)
        elif self.algorithm == LoadBalancingAlgorithm.HYBRID:
            selected_swarm = self._route_hybrid(available_swarms)
        
        if selected_swarm:
            # Update connection count
            self.swarm_endpoints[selected_swarm].active_connections += 1
            self.swarm_endpoints[selected_swarm].total_requests += 1
            self.performance_stats['total_requests'] += 1
            
            logger.debug(f"Routed request to swarm {selected_swarm}")
        
        return selected_swarm
    
    def _get_available_swarms(self, region_preference: Optional[str] = None) -> List[str]:
        """Get list of available swarms"""
        available = []
        
        swarms_to_check = []
        if region_preference and region_preference in self.regional_endpoints:
            swarms_to_check = self.regional_endpoints[region_preference]
        else:
            swarms_to_check = list(self.swarm_endpoints.keys())
        
        for swarm_id in swarms_to_check:
            endpoint = self.swarm_endpoints[swarm_id]
            circuit_breaker = self.circuit_breakers[swarm_id]
            
            # Check health and circuit breaker
            if (endpoint.health_state == SwarmHealthState.HEALTHY and 
                circuit_breaker.can_attempt_request() and
                endpoint.active_connections < endpoint.max_capacity):
                available.append(swarm_id)
        
        return available
    
    def _route_round_robin(self, available_swarms: List[str]) -> str:
        """Round robin routing"""
        if not available_swarms:
            return None
        
        selected = available_swarms[self.round_robin_index % len(available_swarms)]
        self.round_robin_index += 1
        return selected
    
    def _route_least_connections(self, available_swarms: List[str]) -> str:
        """Least connections routing"""
        if not available_swarms:
            return None
        
        min_connections = float('inf')
        selected_swarm = None
        
        for swarm_id in available_swarms:
            endpoint = self.swarm_endpoints[swarm_id]
            if endpoint.active_connections < min_connections:
                min_connections = endpoint.active_connections
                selected_swarm = swarm_id
        
        return selected_swarm
    
    def _route_weighted(self, available_swarms: List[str]) -> str:
        """Weighted random routing"""
        if not available_swarms:
            return None
        
        weights = [self.load_balancing_weights.get(swarm_id, 1.0) for swarm_id in available_swarms]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return random.choice(available_swarms)
        
        rand_val = random.uniform(0, total_weight)
        cumulative_weight = 0
        
        for i, swarm_id in enumerate(available_swarms):
            cumulative_weight += weights[i]
            if rand_val <= cumulative_weight:
                return swarm_id
        
        return available_swarms[-1]
    
    def _route_hybrid(self, available_swarms: List[str]) -> str:
        """Hybrid routing combining multiple strategies"""
        if not available_swarms:
            return None
        
        # Score each swarm based on multiple factors
        swarm_scores = {}
        
        for swarm_id in available_swarms:
            endpoint = self.swarm_endpoints[swarm_id]
            
            # Factors: weight, connection load, response time
            weight_score = self.load_balancing_weights.get(swarm_id, 1.0)
            load_score = 1.0 - (endpoint.active_connections / max(endpoint.max_capacity, 1))
            response_time_score = 1.0 / (endpoint.average_response_time + 1.0)
            
            combined_score = (weight_score * 0.3 + load_score * 0.5 + response_time_score * 0.2)
            swarm_scores[swarm_id] = combined_score
        
        # Select swarm with highest score
        best_swarm = max(swarm_scores.items(), key=lambda x: x[1])
        return best_swarm[0]
    
    async def report_request_completion(self, swarm_id: str, success: bool, response_time_ms: float):
        """Report request completion"""
        if swarm_id not in self.swarm_endpoints:
            return
        
        endpoint = self.swarm_endpoints[swarm_id]
        circuit_breaker = self.circuit_breakers[swarm_id]
        
        # Update connection count
        endpoint.active_connections = max(0, endpoint.active_connections - 1)
        
        # Update stats
        if success:
            endpoint.successful_requests += 1
            self.performance_stats['successful_requests'] += 1
            circuit_breaker.record_success()
        else:
            endpoint.failed_requests += 1
            self.performance_stats['failed_requests'] += 1
            circuit_breaker.record_failure()
        
        # Update response time (simple moving average)
        if endpoint.total_requests > 0:
            endpoint.average_response_time = (
                (endpoint.average_response_time * (endpoint.total_requests - 1) + response_time_ms) /
                endpoint.total_requests
            )
        
        # Update global average response time
        if self.performance_stats['total_requests'] > 0:
            self.performance_stats['average_response_time'] = (
                (self.performance_stats['average_response_time'] * (self.performance_stats['total_requests'] - 1) + response_time_ms) /
                self.performance_stats['total_requests']
            )
    
    async def perform_health_check(self, swarm_id: str) -> bool:
        """Perform health check on a swarm"""
        if swarm_id not in self.swarm_endpoints:
            return False
        
        endpoint = self.swarm_endpoints[swarm_id]
        
        try:
            # Simulate health check
            await asyncio.sleep(0.01)
            
            # Simple health determination based on success rate
            if endpoint.total_requests > 0:
                success_rate = endpoint.successful_requests / endpoint.total_requests
                if success_rate < 0.5:
                    endpoint.health_state = SwarmHealthState.DEGRADED
                elif success_rate < 0.1:
                    endpoint.health_state = SwarmHealthState.FAILED
                else:
                    endpoint.health_state = SwarmHealthState.HEALTHY
            
            endpoint.last_health_check = datetime.now()
            return endpoint.health_state == SwarmHealthState.HEALTHY
            
        except Exception as e:
            logger.error(f"Health check failed for swarm {swarm_id}: {e}")
            endpoint.health_state = SwarmHealthState.FAILED
            return False
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics"""
        return {
            'algorithm': self.algorithm.value,
            'registered_swarms': len(self.swarm_endpoints),
            'performance_stats': self.performance_stats.copy(),
            'swarm_details': {
                swarm_id: {
                    'health_state': endpoint.health_state.value,
                    'active_connections': endpoint.active_connections,
                    'total_requests': endpoint.total_requests,
                    'success_rate': endpoint.successful_requests / max(endpoint.total_requests, 1),
                    'average_response_time': endpoint.average_response_time,
                    'circuit_breaker_state': self.circuit_breakers[swarm_id].state
                }
                for swarm_id, endpoint in self.swarm_endpoints.items()
            }
        }