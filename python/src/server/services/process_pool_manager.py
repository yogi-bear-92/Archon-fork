"""
Process Pool Manager - Phase 3 Implementation  
On-demand tool spawning with intelligent process pooling
Memory optimization through lazy loading and resource management
"""

import asyncio
import logging
import time
import weakref
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from collections import defaultdict
from enum import Enum

logger = logging.getLogger(__name__)

class ProcessState(Enum):
    IDLE = "idle"
    BUSY = "busy" 
    STARTING = "starting"
    STOPPING = "stopping"
    ERROR = "error"

@dataclass
class ProcessInfo:
    """Information about a managed process"""
    process_id: str
    command: List[str]
    state: ProcessState = ProcessState.IDLE
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0
    memory_usage: Optional[int] = None
    process: Optional[asyncio.subprocess.Process] = None
    
class ProcessPoolManager:
    """Intelligent process pool with lazy loading and memory management"""
    
    def __init__(self, 
                 max_processes: int = 5,
                 max_idle_time: int = 300,  # 5 minutes
                 memory_limit_mb: int = 100,
                 cleanup_interval: int = 60):  # 1 minute
        
        self.max_processes = max_processes
        self.max_idle_time = max_idle_time
        self.memory_limit_mb = memory_limit_mb
        self.cleanup_interval = cleanup_interval
        
        # Process management
        self.processes: Dict[str, ProcessInfo] = {}
        self.process_pools: Dict[str, List[str]] = defaultdict(list)  # tool -> process_ids
        self.process_queue: Dict[str, List[Callable]] = defaultdict(list)  # tool -> waiting tasks
        
        # Statistics
        self.stats = {
            "processes_created": 0,
            "processes_reused": 0,
            "processes_cleaned": 0,
            "memory_optimizations": 0,
            "total_executions": 0
        }
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self):
        """Start the process pool manager"""
        logger.info("🚀 Starting Process Pool Manager")
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
    async def stop(self):
        """Stop the process pool manager and cleanup all processes"""
        logger.info("⏹️ Stopping Process Pool Manager")
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        # Cleanup all processes
        await self._cleanup_all_processes()
        
    async def execute_tool(self, 
                          tool_name: str, 
                          command: List[str], 
                          timeout: int = 30,
                          prefer_new_process: bool = False) -> Dict[str, Any]:
        """Execute tool command using process pool"""
        
        self.stats["total_executions"] += 1
        
        try:
            # Try to get existing process if not requesting new one
            process_info = None
            if not prefer_new_process:
                process_info = await self._get_available_process(tool_name, command)
                
            # Create new process if none available
            if not process_info:
                process_info = await self._create_process(tool_name, command)
                
            if not process_info:
                raise RuntimeError(f"Failed to create process for {tool_name}")
                
            # Execute command
            result = await self._execute_with_process(process_info, command, timeout)
            
            # Update usage statistics
            process_info.last_used = time.time()
            process_info.usage_count += 1
            
            # Check if process should be kept alive
            await self._manage_process_lifecycle(process_info)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to execute {tool_name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": tool_name
            }
    
    async def _get_available_process(self, tool_name: str, command: List[str]) -> Optional[ProcessInfo]:
        """Get available process for tool or None"""
        
        # Check if we have idle processes for this tool
        tool_processes = self.process_pools.get(tool_name, [])
        
        for process_id in tool_processes:
            process_info = self.processes.get(process_id)
            if (process_info and 
                process_info.state == ProcessState.IDLE and
                process_info.command[:len(command)] == command):  # Command prefix match
                
                process_info.state = ProcessState.BUSY
                self.stats["processes_reused"] += 1
                logger.debug(f"♻️ Reusing process {process_id} for {tool_name}")
                return process_info
                
        return None
    
    async def _create_process(self, tool_name: str, command: List[str]) -> Optional[ProcessInfo]:
        """Create new process for tool"""
        
        # Check if we're at max capacity
        if len(self.processes) >= self.max_processes:
            # Try to cleanup idle processes first
            await self._cleanup_idle_processes()
            
            if len(self.processes) >= self.max_processes:
                logger.warning(f"⚠️ Max processes reached ({self.max_processes}), queuing request")
                return None
        
        # Generate process ID
        process_id = f"{tool_name}_{int(time.time() * 1000)}"
        
        try:
            logger.info(f"🚀 Creating process {process_id} for {tool_name}")
            
            # Create process info
            process_info = ProcessInfo(
                process_id=process_id,
                command=command,
                state=ProcessState.STARTING
            )
            
            # Store process
            self.processes[process_id] = process_info
            self.process_pools[tool_name].append(process_id)
            
            # Update stats
            self.stats["processes_created"] += 1
            process_info.state = ProcessState.BUSY
            
            return process_info
            
        except Exception as e:
            logger.error(f"❌ Failed to create process {process_id}: {e}")
            # Cleanup failed process
            if process_id in self.processes:
                del self.processes[process_id]
            if process_id in self.process_pools[tool_name]:
                self.process_pools[tool_name].remove(process_id)
            return None
    
    async def _execute_with_process(self, 
                                   process_info: ProcessInfo, 
                                   command: List[str], 
                                   timeout: int) -> Dict[str, Any]:
        """Execute command with process"""
        
        try:
            # Create subprocess
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=1024*1024  # 1MB buffer limit
            )
            
            process_info.process = process
            
            # Execute with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                return {
                    "success": process.returncode == 0,
                    "returncode": process.returncode,
                    "stdout": stdout.decode('utf-8', errors='ignore'),
                    "stderr": stderr.decode('utf-8', errors='ignore'),
                    "process_id": process_info.process_id,
                    "reused": process_info.usage_count > 0
                }
                
            except asyncio.TimeoutError:
                # Kill timed out process
                process.kill()
                await process.wait()
                
                return {
                    "success": False,
                    "error": f"Command timeout after {timeout}s",
                    "process_id": process_info.process_id
                }
                
        except Exception as e:
            return {
                "success": False, 
                "error": str(e),
                "process_id": process_info.process_id
            }
        finally:
            process_info.process = None
            process_info.state = ProcessState.IDLE
    
    async def _manage_process_lifecycle(self, process_info: ProcessInfo):
        """Decide whether to keep process alive or terminate it"""
        
        current_time = time.time()
        
        # Keep frequently used processes alive
        if process_info.usage_count >= 3:
            process_info.state = ProcessState.IDLE
            return
            
        # Keep recently used processes alive
        if current_time - process_info.last_used < 60:  # 1 minute
            process_info.state = ProcessState.IDLE
            return
            
        # Terminate infrequently used processes
        await self._terminate_process(process_info)
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                await self._cleanup_idle_processes()
                await self._memory_optimization_check()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"❌ Cleanup loop error: {e}")
    
    async def _cleanup_idle_processes(self):
        """Clean up idle processes that exceed idle time"""
        current_time = time.time()
        to_cleanup = []
        
        for process_id, process_info in self.processes.items():
            if (process_info.state == ProcessState.IDLE and 
                current_time - process_info.last_used > self.max_idle_time):
                to_cleanup.append(process_info)
        
        for process_info in to_cleanup:
            await self._terminate_process(process_info)
            
        if to_cleanup:
            logger.info(f"🧹 Cleaned up {len(to_cleanup)} idle processes")
    
    async def _memory_optimization_check(self):
        """Check memory usage and optimize if needed"""
        
        # Simple heuristic: if too many processes, cleanup least used
        if len(self.processes) > self.max_processes * 0.8:
            
            # Sort by usage and recency
            processes_by_usage = sorted(
                self.processes.values(),
                key=lambda p: (p.usage_count, p.last_used)
            )
            
            # Cleanup bottom 25% if idle
            cleanup_count = max(1, len(processes_by_usage) // 4)
            
            for process_info in processes_by_usage[:cleanup_count]:
                if process_info.state == ProcessState.IDLE:
                    await self._terminate_process(process_info)
                    self.stats["memory_optimizations"] += 1
    
    async def _terminate_process(self, process_info: ProcessInfo):
        """Terminate a specific process"""
        try:
            logger.debug(f"🗑️ Terminating process {process_info.process_id}")
            
            if process_info.process:
                process_info.process.terminate()
                try:
                    await asyncio.wait_for(process_info.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    process_info.process.kill()
                    await process_info.process.wait()
            
            # Remove from tracking
            process_id = process_info.process_id
            if process_id in self.processes:
                del self.processes[process_id]
                
            # Remove from pools
            for tool_processes in self.process_pools.values():
                if process_id in tool_processes:
                    tool_processes.remove(process_id)
                    
            self.stats["processes_cleaned"] += 1
            
        except Exception as e:
            logger.error(f"❌ Error terminating process {process_info.process_id}: {e}")
    
    async def _cleanup_all_processes(self):
        """Cleanup all managed processes"""
        for process_info in list(self.processes.values()):
            await self._terminate_process(process_info)
            
        self.processes.clear()
        self.process_pools.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get process pool statistics"""
        current_time = time.time()
        
        active_processes = sum(1 for p in self.processes.values() if p.state != ProcessState.ERROR)
        idle_processes = sum(1 for p in self.processes.values() if p.state == ProcessState.IDLE)
        busy_processes = sum(1 for p in self.processes.values() if p.state == ProcessState.BUSY)
        
        return {
            "total_processes": len(self.processes),
            "active_processes": active_processes,
            "idle_processes": idle_processes,
            "busy_processes": busy_processes,
            "max_processes": self.max_processes,
            "memory_limit_mb": self.memory_limit_mb,
            "stats": self.stats.copy(),
            "tools_with_processes": len(self.process_pools),
            "average_process_age": sum(current_time - p.created_at for p in self.processes.values()) / max(len(self.processes), 1),
            "memory_optimization": {
                "enabled": True,
                "cleanup_interval": self.cleanup_interval,
                "max_idle_time": self.max_idle_time
            }
        }

# Global process pool manager
process_pool_manager = ProcessPoolManager()