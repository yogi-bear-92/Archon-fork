"""
Mock objects for Serena Master Agent MCP tools testing.

Provides comprehensive mocks for all Serena MCP tool integrations
and coordination protocols used in testing scenarios.
"""

import json
import asyncio
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from unittest.mock import MagicMock, AsyncMock, patch
from contextlib import asynccontextmanager


class MockSerenaTools:
    """Mock implementation of Serena MCP tools for testing."""
    
    def __init__(self):
        self.call_history = []
        self.responses = {}
        self.delays = {}
        self.error_conditions = {}
        self.metrics = {
            'call_count': 0,
            'total_time': 0,
            'errors': 0
        }
    
    def set_response(self, tool_name: str, response: Any):
        """Set mock response for a specific tool."""
        self.responses[tool_name] = response
    
    def set_delay(self, tool_name: str, delay_ms: int):
        """Set artificial delay for a tool to test performance."""
        self.delays[tool_name] = delay_ms / 1000.0
    
    def set_error_condition(self, tool_name: str, error: Exception):
        """Set error condition for a tool to test error handling."""
        self.error_conditions[tool_name] = error
    
    async def _simulate_call(self, tool_name: str, *args, **kwargs):
        """Simulate tool call with tracking and delays."""
        start_time = time.time()
        self.metrics['call_count'] += 1
        
        call_record = {
            'tool': tool_name,
            'args': args,
            'kwargs': kwargs,
            'timestamp': datetime.now().isoformat()
        }
        self.call_history.append(call_record)
        
        # Simulate delay if configured
        if tool_name in self.delays:
            await asyncio.sleep(self.delays[tool_name])
        
        # Simulate error if configured
        if tool_name in self.error_conditions:
            self.metrics['errors'] += 1
            raise self.error_conditions[tool_name]
        
        # Return configured response or default
        response = self.responses.get(tool_name, {"status": "success", "data": {}})
        
        end_time = time.time()
        self.metrics['total_time'] += (end_time - start_time)
        
        return response
    
    # Serena MCP Tool Mocks
    async def list_dir(self, relative_path: str, recursive: bool = False, max_answer_chars: int = -1):
        """Mock list_dir tool."""
        return await self._simulate_call('list_dir', relative_path, recursive, max_answer_chars)
    
    async def find_file(self, file_mask: str, relative_path: str):
        """Mock find_file tool."""
        return await self._simulate_call('find_file', file_mask, relative_path)
    
    async def search_for_pattern(
        self, 
        substring_pattern: str,
        relative_path: str = "",
        context_lines_before: int = 0,
        context_lines_after: int = 0,
        paths_include_glob: str = "",
        paths_exclude_glob: str = "",
        restrict_search_to_code_files: bool = False,
        max_answer_chars: int = -1
    ):
        """Mock search_for_pattern tool."""
        return await self._simulate_call(
            'search_for_pattern',
            substring_pattern,
            relative_path=relative_path,
            context_lines_before=context_lines_before,
            context_lines_after=context_lines_after,
            paths_include_glob=paths_include_glob,
            paths_exclude_glob=paths_exclude_glob,
            restrict_search_to_code_files=restrict_search_to_code_files,
            max_answer_chars=max_answer_chars
        )
    
    async def get_symbols_overview(self, relative_path: str, max_answer_chars: int = -1):
        """Mock get_symbols_overview tool."""
        return await self._simulate_call('get_symbols_overview', relative_path, max_answer_chars)
    
    async def find_symbol(
        self,
        name_path: str,
        relative_path: str = "",
        depth: int = 0,
        include_body: bool = False,
        include_kinds: List[int] = None,
        exclude_kinds: List[int] = None,
        substring_matching: bool = False,
        max_answer_chars: int = -1
    ):
        """Mock find_symbol tool."""
        return await self._simulate_call(
            'find_symbol',
            name_path,
            relative_path=relative_path,
            depth=depth,
            include_body=include_body,
            include_kinds=include_kinds or [],
            exclude_kinds=exclude_kinds or [],
            substring_matching=substring_matching,
            max_answer_chars=max_answer_chars
        )
    
    async def find_referencing_symbols(
        self,
        name_path: str,
        relative_path: str,
        include_kinds: List[int] = None,
        exclude_kinds: List[int] = None,
        max_answer_chars: int = -1
    ):
        """Mock find_referencing_symbols tool."""
        return await self._simulate_call(
            'find_referencing_symbols',
            name_path,
            relative_path,
            include_kinds=include_kinds or [],
            exclude_kinds=exclude_kinds or [],
            max_answer_chars=max_answer_chars
        )
    
    async def replace_symbol_body(self, name_path: str, relative_path: str, body: str):
        """Mock replace_symbol_body tool."""
        return await self._simulate_call('replace_symbol_body', name_path, relative_path, body)
    
    async def insert_after_symbol(self, name_path: str, relative_path: str, body: str):
        """Mock insert_after_symbol tool."""
        return await self._simulate_call('insert_after_symbol', name_path, relative_path, body)
    
    async def insert_before_symbol(self, name_path: str, relative_path: str, body: str):
        """Mock insert_before_symbol tool."""
        return await self._simulate_call('insert_before_symbol', name_path, relative_path, body)
    
    async def write_memory(self, memory_name: str, content: str, max_answer_chars: int = -1):
        """Mock write_memory tool."""
        return await self._simulate_call('write_memory', memory_name, content, max_answer_chars)
    
    async def read_memory(self, memory_file_name: str, max_answer_chars: int = -1):
        """Mock read_memory tool."""
        return await self._simulate_call('read_memory', memory_file_name, max_answer_chars)
    
    async def list_memories(self):
        """Mock list_memories tool."""
        return await self._simulate_call('list_memories')
    
    async def delete_memory(self, memory_file_name: str):
        """Mock delete_memory tool."""
        return await self._simulate_call('delete_memory', memory_file_name)
    
    async def check_onboarding_performed(self):
        """Mock check_onboarding_performed tool."""
        return await self._simulate_call('check_onboarding_performed')
    
    async def onboarding(self):
        """Mock onboarding tool."""
        return await self._simulate_call('onboarding')
    
    async def think_about_collected_information(self):
        """Mock think_about_collected_information tool."""
        return await self._simulate_call('think_about_collected_information')
    
    async def think_about_task_adherence(self):
        """Mock think_about_task_adherence tool."""
        return await self._simulate_call('think_about_task_adherence')
    
    async def think_about_whether_you_are_done(self):
        """Mock think_about_whether_you_are_done tool."""
        return await self._simulate_call('think_about_whether_you_are_done')
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from mock calls."""
        avg_time = self.metrics['total_time'] / max(1, self.metrics['call_count'])
        return {
            'total_calls': self.metrics['call_count'],
            'total_time': self.metrics['total_time'],
            'average_time': avg_time,
            'error_count': self.metrics['errors'],
            'success_rate': 1 - (self.metrics['errors'] / max(1, self.metrics['call_count']))
        }
    
    def get_call_history(self) -> List[Dict[str, Any]]:
        """Get history of all tool calls."""
        return self.call_history.copy()
    
    def reset_metrics(self):
        """Reset all metrics and call history."""
        self.call_history.clear()
        self.metrics = {'call_count': 0, 'total_time': 0, 'errors': 0}


class MockClaudeFlowCoordination:
    """Mock Claude Flow coordination system for testing."""
    
    def __init__(self):
        self.swarms = {}
        self.agents = {}
        self.memory_store = {}
        self.message_queue = []
        self.hooks_executed = []
        self.neural_patterns = {}
        self.performance_metrics = {
            'swarm_operations': 0,
            'agent_spawns': 0,
            'memory_operations': 0,
            'coordination_messages': 0
        }
    
    async def swarm_init(self, topology: str = "mesh", max_agents: int = 5) -> Dict[str, Any]:
        """Mock swarm initialization."""
        swarm_id = f"swarm_{len(self.swarms) + 1}"
        self.swarms[swarm_id] = {
            'id': swarm_id,
            'topology': topology,
            'max_agents': max_agents,
            'active_agents': [],
            'status': 'initialized',
            'created_at': datetime.now().isoformat()
        }
        self.performance_metrics['swarm_operations'] += 1
        
        return {
            'status': 'initialized',
            'swarm_id': swarm_id,
            'topology': topology,
            'max_agents': max_agents
        }
    
    async def agent_spawn(self, agent_type: str, capabilities: List[str] = None) -> Dict[str, Any]:
        """Mock agent spawning."""
        agent_id = f"agent_{len(self.agents) + 1}"
        self.agents[agent_id] = {
            'id': agent_id,
            'type': agent_type,
            'capabilities': capabilities or [],
            'status': 'spawned',
            'spawned_at': datetime.now().isoformat(),
            'performance': {
                'tasks_completed': 0,
                'avg_response_time': 0,
                'success_rate': 1.0
            }
        }
        self.performance_metrics['agent_spawns'] += 1
        
        return {
            'status': 'spawned',
            'agent_id': agent_id,
            'type': agent_type,
            'capabilities': capabilities or []
        }
    
    async def memory_store(self, key: str, value: Any, ttl: int = None) -> Dict[str, Any]:
        """Mock memory store operation."""
        self.memory_store[key] = {
            'value': value,
            'stored_at': datetime.now().isoformat(),
            'ttl': ttl
        }
        self.performance_metrics['memory_operations'] += 1
        
        return {'status': 'stored', 'key': key}
    
    async def memory_retrieve(self, key: str) -> Dict[str, Any]:
        """Mock memory retrieve operation."""
        self.performance_metrics['memory_operations'] += 1
        
        if key in self.memory_store:
            return {
                'status': 'retrieved',
                'key': key,
                'value': self.memory_store[key]['value']
            }
        else:
            return {'status': 'not_found', 'key': key}
    
    async def send_coordination_message(
        self, 
        from_agent: str, 
        to_agent: str, 
        message: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock coordination message sending."""
        message_id = f"msg_{len(self.message_queue) + 1}"
        message_data = {
            'id': message_id,
            'from_agent': from_agent,
            'to_agent': to_agent,
            'message': message,
            'sent_at': datetime.now().isoformat(),
            'status': 'delivered'
        }
        self.message_queue.append(message_data)
        self.performance_metrics['coordination_messages'] += 1
        
        return {'status': 'sent', 'message_id': message_id}
    
    async def execute_hook(self, hook_name: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Mock hook execution."""
        hook_execution = {
            'hook_name': hook_name,
            'context': context or {},
            'executed_at': datetime.now().isoformat(),
            'result': f"Hook {hook_name} executed successfully"
        }
        self.hooks_executed.append(hook_execution)
        
        return {'status': 'executed', 'hook_name': hook_name, 'result': hook_execution['result']}
    
    async def neural_train(self, patterns: List[Dict], model_type: str = "coordination") -> Dict[str, Any]:
        """Mock neural pattern training."""
        model_id = f"model_{model_type}_{len(self.neural_patterns) + 1}"
        self.neural_patterns[model_id] = {
            'model_type': model_type,
            'patterns_count': len(patterns),
            'trained_at': datetime.now().isoformat(),
            'accuracy': 0.85 + (len(patterns) * 0.01)  # Simulated improvement
        }
        
        return {
            'status': 'trained',
            'model_id': model_id,
            'patterns_processed': len(patterns),
            'accuracy': self.neural_patterns[model_id]['accuracy']
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get coordination system performance metrics."""
        return {
            'operations': self.performance_metrics.copy(),
            'swarms_count': len(self.swarms),
            'agents_count': len(self.agents),
            'memory_items': len(self.memory_store),
            'queued_messages': len(self.message_queue),
            'hooks_executed': len(self.hooks_executed),
            'neural_models': len(self.neural_patterns)
        }
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of specific agent."""
        if agent_id in self.agents:
            return self.agents[agent_id]
        else:
            return {'status': 'not_found', 'agent_id': agent_id}
    
    def get_swarm_status(self, swarm_id: str) -> Dict[str, Any]:
        """Get status of specific swarm."""
        if swarm_id in self.swarms:
            swarm = self.swarms[swarm_id].copy()
            swarm['agents'] = [aid for aid, agent in self.agents.items() 
                             if agent_id in swarm.get('active_agents', [])]
            return swarm
        else:
            return {'status': 'not_found', 'swarm_id': swarm_id}
    
    def reset(self):
        """Reset all mock data."""
        self.swarms.clear()
        self.agents.clear()
        self.memory_store.clear()
        self.message_queue.clear()
        self.hooks_executed.clear()
        self.neural_patterns.clear()
        self.performance_metrics = {
            'swarm_operations': 0,
            'agent_spawns': 0,
            'memory_operations': 0,
            'coordination_messages': 0
        }


class MockHttpClient:
    """Mock HTTP client for testing MCP communications."""
    
    def __init__(self):
        self.requests = []
        self.responses = {}
        self.delays = {}
        self.error_conditions = {}
    
    def set_response(self, url_pattern: str, response: Dict[str, Any]):
        """Set mock response for URL pattern."""
        self.responses[url_pattern] = response
    
    def set_delay(self, url_pattern: str, delay_ms: int):
        """Set delay for URL pattern."""
        self.delays[url_pattern] = delay_ms / 1000.0
    
    def set_error(self, url_pattern: str, error: Exception):
        """Set error condition for URL pattern."""
        self.error_conditions[url_pattern] = error
    
    @asynccontextmanager
    async def request(self, method: str, url: str, **kwargs):
        """Mock HTTP request context manager."""
        request_record = {
            'method': method,
            'url': url,
            'kwargs': kwargs,
            'timestamp': datetime.now().isoformat()
        }
        self.requests.append(request_record)
        
        # Find matching response
        response_data = None
        delay = 0
        error = None
        
        for pattern, resp in self.responses.items():
            if pattern in url:
                response_data = resp
                break
        
        for pattern, d in self.delays.items():
            if pattern in url:
                delay = d
                break
        
        for pattern, err in self.error_conditions.items():
            if pattern in url:
                error = err
                break
        
        # Simulate delay
        if delay > 0:
            await asyncio.sleep(delay)
        
        # Simulate error
        if error:
            raise error
        
        # Return mock response
        class MockResponse:
            def __init__(self, data):
                self.data = data
            
            def json(self):
                return self.data
            
            @property
            def status(self):
                return 200
        
        yield MockResponse(response_data or {'status': 'success'})
    
    async def get(self, url: str, **kwargs):
        """Mock GET request."""
        async with self.request('GET', url, **kwargs) as response:
            return response
    
    async def post(self, url: str, **kwargs):
        """Mock POST request."""
        async with self.request('POST', url, **kwargs) as response:
            return response
    
    def get_request_history(self) -> List[Dict[str, Any]]:
        """Get history of all requests."""
        return self.requests.copy()
    
    def reset(self):
        """Reset request history."""
        self.requests.clear()


def create_performance_monitor():
    """Create a performance monitoring context for tests."""
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.memory_snapshots = []
            self.cpu_samples = []
        
        async def __aenter__(self):
            import psutil
            self.start_time = time.time()
            self.memory_snapshots.append(psutil.virtual_memory().used)
            self.cpu_samples.append(psutil.cpu_percent())
            return self
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            import psutil
            self.end_time = time.time()
            self.memory_snapshots.append(psutil.virtual_memory().used)
            self.cpu_samples.append(psutil.cpu_percent())
        
        def get_metrics(self) -> Dict[str, Any]:
            if self.start_time is None or self.end_time is None:
                return {}
            
            return {
                'execution_time_ms': (self.end_time - self.start_time) * 1000,
                'memory_delta_mb': (self.memory_snapshots[-1] - self.memory_snapshots[0]) / 1024 / 1024,
                'avg_cpu_percent': sum(self.cpu_samples) / len(self.cpu_samples),
                'peak_memory_mb': max(self.memory_snapshots) / 1024 / 1024
            }
    
    return PerformanceMonitor()


# Decorator for timing test functions
def time_test(func):
    """Decorator to time test execution."""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000
            
            # Add timing info to result if it's a dict
            if isinstance(result, dict):
                result['_test_timing'] = {
                    'execution_time_ms': execution_time,
                    'start_time': start_time,
                    'end_time': end_time
                }
            
            return result
        except Exception as e:
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000
            
            # Add timing info to exception
            e._test_timing = {
                'execution_time_ms': execution_time,
                'start_time': start_time,
                'end_time': end_time
            }
            raise
    
    return wrapper