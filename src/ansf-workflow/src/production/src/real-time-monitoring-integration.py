#!/usr/bin/env python3
"""
Real-Time Monitoring Integration for Claude Flow Hooks
Memory-efficient monitoring system for ANSF production environment

Features:
- Real-time hook performance tracking
- Memory usage alerts and optimization
- Cross-system coordination monitoring
- Neural prediction accuracy tracking
- Automated performance degradation detection
- Integration with Claude Flow, Serena, Archon, and ANSF

Author: Claude Code Monitoring Team
Target: <1ms monitoring overhead, 99.99% availability
"""

import asyncio
import json
import logging
import time
import uuid
import websockets
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import deque, defaultdict
from threading import Lock
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MonitoringMetrics:
    """Real-time metrics for hook performance."""
    timestamp: datetime = field(default_factory=datetime.now)
    hook_name: str = ""
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    success: bool = True
    error_message: str = ""
    coordination_accuracy: float = 0.0
    neural_prediction_confidence: float = 0.0
    system_component: str = "claude-flow"  # claude-flow, serena, archon, ansf

@dataclass
class SystemHealthStatus:
    """Overall system health status."""
    overall_health: str = "healthy"  # healthy, degraded, critical
    memory_available_mb: float = 0.0
    memory_usage_percent: float = 0.0
    active_hooks: int = 0
    average_response_time_ms: float = 0.0
    coordination_accuracy: float = 0.0
    neural_accuracy: float = 0.0
    alerts: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

class MemoryEfficientRingBuffer:
    """Memory-efficient ring buffer for metrics storage."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = Lock()
    
    def add(self, metric: MonitoringMetrics):
        """Add metric to buffer."""
        with self.lock:
            self.buffer.append(metric)
    
    def get_recent(self, minutes: int = 5) -> List[MonitoringMetrics]:
        """Get metrics from the last N minutes."""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        with self.lock:
            return [
                metric for metric in self.buffer 
                if metric.timestamp >= cutoff_time
            ]
    
    def get_statistics(self, minutes: int = 5) -> Dict[str, Any]:
        """Get statistical summary of recent metrics."""
        recent_metrics = self.get_recent(minutes)
        if not recent_metrics:
            return {}
        
        execution_times = [m.execution_time_ms for m in recent_metrics if m.success]
        memory_usage = [m.memory_usage_mb for m in recent_metrics]
        accuracies = [m.coordination_accuracy for m in recent_metrics if m.coordination_accuracy > 0]
        
        return {
            "total_hooks": len(recent_metrics),
            "successful_hooks": sum(1 for m in recent_metrics if m.success),
            "success_rate": sum(1 for m in recent_metrics if m.success) / len(recent_metrics),
            "avg_execution_time_ms": statistics.mean(execution_times) if execution_times else 0,
            "p95_execution_time_ms": statistics.quantiles(execution_times, n=20)[18] if len(execution_times) > 10 else 0,
            "avg_memory_usage_mb": statistics.mean(memory_usage) if memory_usage else 0,
            "peak_memory_usage_mb": max(memory_usage) if memory_usage else 0,
            "avg_coordination_accuracy": statistics.mean(accuracies) if accuracies else 0,
            "component_breakdown": defaultdict(int)
        }

class RealTimeMonitoringSystem:
    """Main monitoring system for Claude Flow hooks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_buffer = MemoryEfficientRingBuffer(
            max_size=config.get("buffer_size", 1000)
        )
        self.system_health = SystemHealthStatus()
        self.alert_thresholds = {
            "memory_critical_percent": 95,
            "memory_high_percent": 85,
            "response_time_critical_ms": 1000,
            "response_time_warning_ms": 500,
            "accuracy_critical": 0.85,
            "accuracy_warning": 0.90
        }
        self.websocket_clients = set()
        self.monitoring_active = False
        self.alert_callbacks: List[Callable] = []
        
    async def start_monitoring(self):
        """Start the real-time monitoring system."""
        logger.info("🔍 Starting real-time monitoring system")
        self.monitoring_active = True
        
        # Start monitoring tasks
        monitoring_tasks = [
            asyncio.create_task(self._system_health_monitor()),
            asyncio.create_task(self._performance_analyzer()),
            asyncio.create_task(self._memory_monitor()),
            asyncio.create_task(self._websocket_server()),
        ]
        
        await asyncio.gather(*monitoring_tasks, return_exceptions=True)
    
    async def record_hook_execution(self, 
                                  hook_name: str, 
                                  execution_time_ms: float,
                                  success: bool = True,
                                  error_message: str = "",
                                  additional_data: Dict[str, Any] = None):
        """Record hook execution metrics."""
        additional_data = additional_data or {}
        
        # Get current system metrics
        memory_mb = await self._get_current_memory_usage()
        cpu_percent = await self._get_current_cpu_usage()
        
        metric = MonitoringMetrics(
            hook_name=hook_name,
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            success=success,
            error_message=error_message,
            coordination_accuracy=additional_data.get("accuracy", 0.0),
            neural_prediction_confidence=additional_data.get("confidence", 0.0),
            system_component=additional_data.get("component", "claude-flow")
        )
        
        # Add to buffer
        self.metrics_buffer.add(metric)
        
        # Check for alerts
        await self._check_alerts(metric)
        
        # Broadcast to WebSocket clients
        await self._broadcast_metric(metric)
        
        logger.debug(f"Recorded hook execution: {hook_name} ({execution_time_ms:.2f}ms)")
    
    async def _system_health_monitor(self):
        """Monitor overall system health."""
        while self.monitoring_active:
            try:
                # Update system health
                self.system_health.memory_available_mb = await self._get_available_memory()
                self.system_health.memory_usage_percent = await self._get_memory_usage_percent()
                
                # Get recent statistics
                stats = self.metrics_buffer.get_statistics(minutes=5)
                if stats:
                    self.system_health.active_hooks = stats["total_hooks"]
                    self.system_health.average_response_time_ms = stats["avg_execution_time_ms"]
                    self.system_health.coordination_accuracy = stats["avg_coordination_accuracy"]
                
                # Determine overall health
                self.system_health.overall_health = self._calculate_overall_health()
                self.system_health.last_updated = datetime.now()
                
                # Broadcast health update
                await self._broadcast_health_status()
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in system health monitoring: {e}")
                await asyncio.sleep(10)
    
    async def _performance_analyzer(self):
        """Analyze performance trends and patterns."""
        while self.monitoring_active:
            try:
                # Analyze trends every minute
                await asyncio.sleep(60)
                
                stats = self.metrics_buffer.get_statistics(minutes=10)
                if not stats:
                    continue
                
                # Performance degradation detection
                if stats["p95_execution_time_ms"] > self.alert_thresholds["response_time_critical_ms"]:
                    alert = f"Performance degradation detected: P95 response time {stats['p95_execution_time_ms']:.0f}ms"
                    await self._trigger_alert("performance_critical", alert)
                
                # Accuracy degradation detection
                if stats["avg_coordination_accuracy"] > 0 and stats["avg_coordination_accuracy"] < self.alert_thresholds["accuracy_critical"]:
                    alert = f"Coordination accuracy degraded: {stats['avg_coordination_accuracy']:.1%}"
                    await self._trigger_alert("accuracy_critical", alert)
                
                # Memory trend analysis
                if stats["peak_memory_usage_mb"] > 0:
                    # Predict memory exhaustion
                    recent_5min = self.metrics_buffer.get_statistics(minutes=5)
                    recent_1min = self.metrics_buffer.get_statistics(minutes=1)
                    
                    if (recent_1min.get("peak_memory_usage_mb", 0) > 
                        recent_5min.get("avg_memory_usage_mb", 0) * 1.5):
                        alert = "Memory usage spike detected - potential leak"
                        await self._trigger_alert("memory_spike", alert)
                
            except Exception as e:
                logger.error(f"Error in performance analysis: {e}")
    
    async def _memory_monitor(self):
        """Monitor memory usage and trigger cleanups."""
        while self.monitoring_active:
            try:
                memory_percent = await self._get_memory_usage_percent()
                
                if memory_percent > self.alert_thresholds["memory_critical_percent"]:
                    # Critical memory - trigger emergency cleanup
                    alert = f"Critical memory usage: {memory_percent:.1f}%"
                    await self._trigger_alert("memory_critical", alert)
                    await self._trigger_emergency_cleanup()
                    
                elif memory_percent > self.alert_thresholds["memory_high_percent"]:
                    # High memory - trigger preventive cleanup
                    alert = f"High memory usage: {memory_percent:.1f}%"
                    await self._trigger_alert("memory_high", alert)
                    await self._trigger_preventive_cleanup()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _websocket_server(self):
        """WebSocket server for real-time monitoring dashboard."""
        async def handle_client(websocket, path):
            self.websocket_clients.add(websocket)
            logger.info(f"New monitoring client connected: {len(self.websocket_clients)} total")
            
            try:
                # Send initial system status
                await websocket.send(json.dumps({
                    "type": "system_health",
                    "data": asdict(self.system_health)
                }))
                
                # Keep connection alive
                async for message in websocket:
                    # Handle client requests
                    try:
                        request = json.loads(message)
                        await self._handle_client_request(websocket, request)
                    except json.JSONDecodeError:
                        await websocket.send(json.dumps({
                            "type": "error",
                            "message": "Invalid JSON"
                        }))
                        
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.websocket_clients.discard(websocket)
                logger.info(f"Client disconnected: {len(self.websocket_clients)} remaining")
        
        # Start WebSocket server
        port = self.config.get("websocket_port", 8765)
        try:
            await websockets.serve(handle_client, "localhost", port)
            logger.info(f"WebSocket monitoring server started on port {port}")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
    
    async def _handle_client_request(self, websocket, request: Dict[str, Any]):
        """Handle client requests for monitoring data."""
        request_type = request.get("type")
        
        if request_type == "get_statistics":
            minutes = request.get("minutes", 5)
            stats = self.metrics_buffer.get_statistics(minutes)
            await websocket.send(json.dumps({
                "type": "statistics",
                "data": stats
            }))
            
        elif request_type == "get_recent_metrics":
            minutes = request.get("minutes", 5)
            metrics = self.metrics_buffer.get_recent(minutes)
            await websocket.send(json.dumps({
                "type": "recent_metrics",
                "data": [asdict(m) for m in metrics[-50:]]  # Limit to 50 most recent
            }))
            
        elif request_type == "trigger_cleanup":
            await self._trigger_preventive_cleanup()
            await websocket.send(json.dumps({
                "type": "cleanup_triggered",
                "message": "Cleanup initiated"
            }))
    
    async def _broadcast_metric(self, metric: MonitoringMetrics):
        """Broadcast metric to all connected WebSocket clients."""
        if not self.websocket_clients:
            return
        
        message = json.dumps({
            "type": "metric_update",
            "data": asdict(metric)
        })
        
        disconnected = set()
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.debug(f"Error broadcasting to client: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected
    
    async def _broadcast_health_status(self):
        """Broadcast system health status."""
        if not self.websocket_clients:
            return
        
        message = json.dumps({
            "type": "health_update",
            "data": asdict(self.system_health)
        })
        
        disconnected = set()
        for client in self.websocket_clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception:
                disconnected.add(client)
        
        self.websocket_clients -= disconnected
    
    def _calculate_overall_health(self) -> str:
        """Calculate overall system health status."""
        # Critical conditions
        if (self.system_health.memory_usage_percent > self.alert_thresholds["memory_critical_percent"] or
            self.system_health.average_response_time_ms > self.alert_thresholds["response_time_critical_ms"]):
            return "critical"
        
        # Degraded conditions
        if (self.system_health.memory_usage_percent > self.alert_thresholds["memory_high_percent"] or
            self.system_health.average_response_time_ms > self.alert_thresholds["response_time_warning_ms"] or
            (self.system_health.coordination_accuracy > 0 and 
             self.system_health.coordination_accuracy < self.alert_thresholds["accuracy_warning"])):
            return "degraded"
        
        return "healthy"
    
    async def _check_alerts(self, metric: MonitoringMetrics):
        """Check if metric triggers any alerts."""
        alerts = []
        
        # Memory alerts
        if metric.memory_usage_mb > 0:
            memory_percent = await self._get_memory_usage_percent()
            if memory_percent > self.alert_thresholds["memory_critical_percent"]:
                alerts.append(f"Critical memory usage: {memory_percent:.1f}%")
        
        # Performance alerts
        if metric.execution_time_ms > self.alert_thresholds["response_time_critical_ms"]:
            alerts.append(f"Hook {metric.hook_name} exceeded critical response time: {metric.execution_time_ms:.0f}ms")
        
        # Accuracy alerts
        if (metric.coordination_accuracy > 0 and 
            metric.coordination_accuracy < self.alert_thresholds["accuracy_critical"]):
            alerts.append(f"Coordination accuracy below critical threshold: {metric.coordination_accuracy:.1%}")
        
        # Trigger alerts
        for alert in alerts:
            await self._trigger_alert("metric_alert", alert)
    
    async def _trigger_alert(self, alert_type: str, message: str):
        """Trigger an alert."""
        alert_data = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "system_health": asdict(self.system_health)
        }
        
        # Add to system alerts
        self.system_health.alerts.append(message)
        
        # Keep only recent alerts (last 10)
        self.system_health.alerts = self.system_health.alerts[-10:]
        
        # Log alert
        logger.warning(f"ALERT [{alert_type}]: {message}")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                await callback(alert_data)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        # Broadcast to clients
        if self.websocket_clients:
            alert_message = json.dumps({
                "type": "alert",
                "data": alert_data
            })
            
            for client in list(self.websocket_clients):
                try:
                    await client.send(alert_message)
                except Exception:
                    self.websocket_clients.discard(client)
    
    async def _trigger_emergency_cleanup(self):
        """Trigger emergency memory cleanup."""
        logger.warning("🚨 Triggering emergency memory cleanup")
        
        try:
            # Execute Claude Flow emergency cleanup
            import subprocess
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "emergency-cleanup",
                "--aggressive", "--force-gc", "--cache-clear"
            ], timeout=10)
            
            logger.info("✅ Emergency cleanup completed")
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
    
    async def _trigger_preventive_cleanup(self):
        """Trigger preventive memory cleanup."""
        logger.info("🧹 Triggering preventive cleanup")
        
        try:
            # Execute Claude Flow preventive cleanup
            import subprocess
            subprocess.run([
                "npx", "claude-flow@alpha", "hooks", "cleanup",
                "--cache-optimize", "--gc-collect"
            ], timeout=5)
            
        except Exception as e:
            logger.debug(f"Preventive cleanup error: {e}")
    
    async def _get_available_memory(self) -> float:
        """Get available system memory in MB."""
        # Implementation similar to production deployment script
        try:
            import subprocess
            
            # Try macOS first
            result = subprocess.run(
                ["vm_stat"], 
                capture_output=True, 
                text=True, 
                timeout=2
            )
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                page_size = 16384
                free_pages = 0
                
                for line in lines:
                    if 'page size of' in line:
                        page_size = int(line.split()[-2])
                    elif 'Pages free:' in line:
                        free_pages = int(line.split()[-1].rstrip('.'))
                
                return (free_pages * page_size) / (1024 * 1024)
            
            # Fallback
            return 100.0
            
        except Exception:
            return 100.0
    
    async def _get_current_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
    
    async def _get_memory_usage_percent(self) -> float:
        """Get system memory usage percentage."""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            # Fallback calculation
            available = await self._get_available_memory()
            if available < 100:
                return 95.0  # Assume critical if very low
            elif available < 200:
                return 85.0  # Assume high
            else:
                return 60.0  # Assume normal
        except Exception:
            return 60.0
    
    async def _get_current_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=None)
        except ImportError:
            return 0.0
        except Exception:
            return 0.0
    
    def add_alert_callback(self, callback: Callable):
        """Add callback function for alerts."""
        self.alert_callbacks.append(callback)
    
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        logger.info("Stopping real-time monitoring system")
        self.monitoring_active = False

# Factory function for easy integration
def create_monitoring_system(config: Optional[Dict[str, Any]] = None) -> RealTimeMonitoringSystem:
    """Create and configure monitoring system."""
    if config is None:
        config = {
            "buffer_size": 1000,
            "websocket_port": 8765,
            "alert_thresholds": {
                "memory_critical_percent": 95,
                "memory_high_percent": 85,
                "response_time_critical_ms": 1000,
                "response_time_warning_ms": 500,
                "accuracy_critical": 0.85,
                "accuracy_warning": 0.90
            }
        }
    
    return RealTimeMonitoringSystem(config)

# Example usage and integration
async def example_claude_flow_integration():
    """Example of how to integrate monitoring with Claude Flow hooks."""
    
    # Create monitoring system
    monitor = create_monitoring_system()
    
    # Add custom alert handler
    async def custom_alert_handler(alert_data):
        print(f"CUSTOM ALERT: {alert_data['message']}")
    
    monitor.add_alert_callback(custom_alert_handler)
    
    # Start monitoring in background
    monitoring_task = asyncio.create_task(monitor.start_monitoring())
    
    # Example: Record hook executions
    await monitor.record_hook_execution(
        hook_name="pre-task",
        execution_time_ms=150.5,
        success=True,
        additional_data={
            "accuracy": 0.95,
            "confidence": 0.88,
            "component": "claude-flow"
        }
    )
    
    await monitor.record_hook_execution(
        hook_name="neural-predict", 
        execution_time_ms=450.2,
        success=True,
        additional_data={
            "accuracy": 0.92,
            "confidence": 0.85,
            "component": "ansf"
        }
    )
    
    # Let monitoring run for a while
    await asyncio.sleep(10)
    
    # Stop monitoring
    await monitor.stop_monitoring()

if __name__ == "__main__":
    # Run example integration
    asyncio.run(example_claude_flow_integration())