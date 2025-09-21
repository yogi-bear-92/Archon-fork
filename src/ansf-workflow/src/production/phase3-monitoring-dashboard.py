#!/usr/bin/env python3
"""
ANSF Phase 3: Optimal Mode - Enterprise Monitoring Dashboard
Real-time multi-swarm coordination monitoring with advanced analytics

Target: 97%+ coordination accuracy with enterprise-grade observability
Features: Multi-swarm metrics, predictive analytics, neural performance tracking
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import aiohttp
import websockets
from pathlib import Path

# Configure enterprise-grade logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/ansf_phase3_monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SwarmMetrics:
    """Individual swarm performance metrics"""
    swarm_id: str
    swarm_name: str
    agent_count: int
    accuracy: float
    efficiency: float
    response_time_ms: float
    success_rate: float
    task_count: int
    status: str
    last_updated: str

@dataclass
class SystemMetrics:
    """Overall system performance metrics"""
    coordination_accuracy: float
    neural_model_accuracy: float
    avg_response_time: float
    system_efficiency: float
    total_swarms: int
    active_swarms: int
    total_agents: int
    tasks_processed: int
    uptime_minutes: int
    target_achieved: bool

class Phase3MonitoringDashboard:
    """Enterprise monitoring dashboard for ANSF Phase 3 Optimal Mode"""
    
    def __init__(self):
        self.swarm_metrics = {}
        self.system_metrics = SystemMetrics(
            coordination_accuracy=0.947,  # Starting from Phase 2
            neural_model_accuracy=0.887,
            avg_response_time=245.0,
            system_efficiency=0.94,
            total_swarms=0,
            active_swarms=0,
            total_agents=0,
            tasks_processed=0,
            uptime_minutes=0,
            target_achieved=False
        )
        
        self.performance_history = []
        self.alert_thresholds = {
            'coordination_accuracy_min': 0.95,     # 95% minimum
            'neural_accuracy_min': 0.85,          # 85% minimum
            'response_time_max': 500,             # 500ms maximum
            'system_efficiency_min': 0.90,        # 90% minimum
            'swarm_health_min': 0.80              # 80% minimum
        }
        
        self.start_time = datetime.now()
        self.monitoring_active = False
        self.websocket_clients = set()
        
        logger.info("🚀 ANSF Phase 3 Monitoring Dashboard Initialized")
        logger.info("📊 Enterprise-grade multi-swarm observability active")

    async def start_monitoring(self, port: int = 8053):
        """Start the monitoring dashboard with WebSocket server"""
        logger.info(f"🔧 Starting Phase 3 monitoring dashboard on port {port}...")
        
        self.monitoring_active = True
        
        # Start background monitoring tasks
        monitoring_tasks = [
            asyncio.create_task(self.collect_system_metrics()),
            asyncio.create_task(self.collect_swarm_metrics()),
            asyncio.create_task(self.analyze_performance_trends()),
            asyncio.create_task(self.check_system_health()),
            asyncio.create_task(self.start_websocket_server(port))
        ]
        
        logger.info("✅ Phase 3 monitoring dashboard started successfully")
        logger.info("📊 Real-time metrics collection active")
        logger.info("🌐 WebSocket server ready for clients")
        
        try:
            await asyncio.gather(*monitoring_tasks)
        except KeyboardInterrupt:
            logger.info("🛑 Shutting down monitoring dashboard...")
            self.monitoring_active = False
            
    async def start_websocket_server(self, port: int):
        """WebSocket server for real-time dashboard updates"""
        async def handle_client(websocket, path):
            self.websocket_clients.add(websocket)
            logger.info(f"📡 Dashboard client connected from {websocket.remote_address}")
            
            try:
                # Send initial system state
                await websocket.send(json.dumps({
                    'type': 'system_status',
                    'data': self.get_dashboard_data()
                }))
                
                # Keep connection alive
                async for message in websocket:
                    # Handle client requests
                    try:
                        request = json.loads(message)
                        response = await self.handle_client_request(request)
                        await websocket.send(json.dumps(response))
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON from client: {message}")
                        
            except websockets.exceptions.ConnectionClosed:
                logger.info("📡 Dashboard client disconnected")
            finally:
                self.websocket_clients.discard(websocket)
        
        server = await websockets.serve(handle_client, "localhost", port)
        logger.info(f"🌐 WebSocket server listening on ws://localhost:{port}")
        
        await server.wait_closed()
    
    async def handle_client_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle client requests for specific data"""
        request_type = request.get('type', 'unknown')
        
        if request_type == 'system_status':
            return {
                'type': 'system_status_response',
                'data': self.get_dashboard_data()
            }
        elif request_type == 'performance_history':
            return {
                'type': 'performance_history_response', 
                'data': self.performance_history[-50:]  # Last 50 data points
            }
        elif request_type == 'swarm_details':
            swarm_id = request.get('swarm_id')
            return {
                'type': 'swarm_details_response',
                'data': self.get_swarm_details(swarm_id)
            }
        else:
            return {
                'type': 'error',
                'message': f'Unknown request type: {request_type}'
            }
    
    async def collect_system_metrics(self):
        """Collect overall system performance metrics"""
        while self.monitoring_active:
            try:
                # Update system uptime
                uptime = datetime.now() - self.start_time
                self.system_metrics.uptime_minutes = int(uptime.total_seconds() / 60)
                
                # Simulate gradual system improvement (in production, collect real metrics)
                self.system_metrics.coordination_accuracy = min(0.98, 
                    self.system_metrics.coordination_accuracy + 0.0001)
                
                self.system_metrics.neural_model_accuracy = min(0.94,
                    self.system_metrics.neural_model_accuracy + 0.00005)
                
                # Update target achievement status
                self.system_metrics.target_achieved = (
                    self.system_metrics.coordination_accuracy >= 0.97
                )
                
                # Update swarm counts
                active_swarms = sum(1 for metrics in self.swarm_metrics.values() 
                                  if metrics.status == 'active')
                self.system_metrics.active_swarms = active_swarms
                self.system_metrics.total_swarms = len(self.swarm_metrics)
                
                # Update total agents
                self.system_metrics.total_agents = sum(
                    metrics.agent_count for metrics in self.swarm_metrics.values()
                )
                
                # Log system metrics every 5 minutes
                if self.system_metrics.uptime_minutes % 5 == 0 and self.system_metrics.uptime_minutes > 0:
                    logger.info(f"📊 System Metrics Update:")
                    logger.info(f"  🎯 Coordination Accuracy: {self.system_metrics.coordination_accuracy:.1%}")
                    logger.info(f"  🧠 Neural Model Accuracy: {self.system_metrics.neural_model_accuracy:.1%}")
                    logger.info(f"  ⚡ Avg Response Time: {self.system_metrics.avg_response_time:.0f}ms")
                    logger.info(f"  🤖 Active Swarms: {self.system_metrics.active_swarms}/{self.system_metrics.total_swarms}")
                    logger.info(f"  👥 Total Agents: {self.system_metrics.total_agents}")
                
                await self.broadcast_metrics_update()
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                
            await asyncio.sleep(30)  # Collect every 30 seconds
    
    async def collect_swarm_metrics(self):
        """Collect individual swarm performance metrics"""
        while self.monitoring_active:
            try:
                # Simulate swarm metrics collection (in production, query actual swarms)
                await self.simulate_swarm_metrics_collection()
                
                # Update system averages based on swarm metrics
                if self.swarm_metrics:
                    avg_accuracy = sum(m.accuracy for m in self.swarm_metrics.values()) / len(self.swarm_metrics)
                    avg_response = sum(m.response_time_ms for m in self.swarm_metrics.values()) / len(self.swarm_metrics)
                    avg_efficiency = sum(m.efficiency for m in self.swarm_metrics.values()) / len(self.swarm_metrics)
                    
                    self.system_metrics.coordination_accuracy = avg_accuracy
                    self.system_metrics.avg_response_time = avg_response
                    self.system_metrics.system_efficiency = avg_efficiency
                
            except Exception as e:
                logger.error(f"Error collecting swarm metrics: {e}")
                
            await asyncio.sleep(60)  # Collect every minute
    
    async def simulate_swarm_metrics_collection(self):
        """Simulate swarm metrics collection (replace with real data in production)"""
        swarms_config = {
            'ai-research': {'agents': 8, 'base_accuracy': 0.94},
            'backend-dev': {'agents': 6, 'base_accuracy': 0.92},
            'frontend-ui': {'agents': 4, 'base_accuracy': 0.90},
            'testing-qa': {'agents': 6, 'base_accuracy': 0.93},
            'devops-deploy': {'agents': 4, 'base_accuracy': 0.91},
            'security-compliance': {'agents': 4, 'base_accuracy': 0.95}
        }
        
        for swarm_name, config in swarms_config.items():
            swarm_id = f"swarm-{swarm_name}-{int(time.time())}"
            
            # Simulate performance metrics with gradual improvement
            base_accuracy = config['base_accuracy']
            accuracy_variance = 0.02 * (0.5 - abs(0.5 - (time.time() % 60) / 60))
            
            metrics = SwarmMetrics(
                swarm_id=swarm_id,
                swarm_name=swarm_name,
                agent_count=config['agents'],
                accuracy=min(0.98, base_accuracy + accuracy_variance),
                efficiency=min(0.96, 0.88 + abs(accuracy_variance)),
                response_time_ms=180 + (200 * abs(accuracy_variance)),
                success_rate=min(0.99, base_accuracy + 0.03),
                task_count=int(20 + (time.time() % 100)),
                status='active',
                last_updated=datetime.now().isoformat()
            )
            
            self.swarm_metrics[swarm_name] = metrics
    
    async def analyze_performance_trends(self):
        """Analyze performance trends and store historical data"""
        while self.monitoring_active:
            try:
                # Create performance snapshot
                snapshot = {
                    'timestamp': datetime.now().isoformat(),
                    'coordination_accuracy': self.system_metrics.coordination_accuracy,
                    'neural_accuracy': self.system_metrics.neural_model_accuracy,
                    'response_time': self.system_metrics.avg_response_time,
                    'system_efficiency': self.system_metrics.system_efficiency,
                    'active_swarms': self.system_metrics.active_swarms,
                    'total_agents': self.system_metrics.total_agents
                }
                
                self.performance_history.append(snapshot)
                
                # Keep only last 1000 data points
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-1000:]
                
                # Analyze trends (every 15 minutes)
                if len(self.performance_history) >= 30:  # 30 data points = 15 minutes
                    await self.detect_performance_trends()
                
            except Exception as e:
                logger.error(f"Error analyzing performance trends: {e}")
                
            await asyncio.sleep(30)  # Analyze every 30 seconds
    
    async def detect_performance_trends(self):
        """Detect performance trends and anomalies"""
        if len(self.performance_history) < 10:
            return
            
        recent_data = self.performance_history[-10:]
        
        # Calculate trends
        accuracy_trend = self.calculate_trend([d['coordination_accuracy'] for d in recent_data])
        response_trend = self.calculate_trend([d['response_time'] for d in recent_data])
        
        # Detect significant changes
        if accuracy_trend < -0.01:  # Accuracy decreasing
            logger.warning(f"⚠️ Coordination accuracy trending down: {accuracy_trend:.3f}")
            await self.trigger_alert('accuracy_decline', {'trend': accuracy_trend})
            
        if response_trend > 10:  # Response time increasing
            logger.warning(f"⚠️ Response time trending up: +{response_trend:.0f}ms")
            await self.trigger_alert('response_degradation', {'trend': response_trend})
            
        # Detect positive trends
        if accuracy_trend > 0.005:
            logger.info(f"📈 Coordination accuracy improving: +{accuracy_trend:.3f}")
            
        if response_trend < -5:
            logger.info(f"📈 Response time improving: {response_trend:.0f}ms")
    
    def calculate_trend(self, values: List[float]) -> float:
        """Calculate simple linear trend"""
        if len(values) < 2:
            return 0.0
            
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        try:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
        except ZeroDivisionError:
            return 0.0
    
    async def check_system_health(self):
        """Monitor system health and trigger alerts"""
        while self.monitoring_active:
            try:
                alerts = []
                
                # Check coordination accuracy
                if self.system_metrics.coordination_accuracy < self.alert_thresholds['coordination_accuracy_min']:
                    alerts.append({
                        'type': 'accuracy_low',
                        'severity': 'critical',
                        'message': f"Coordination accuracy below threshold: {self.system_metrics.coordination_accuracy:.1%}",
                        'threshold': self.alert_thresholds['coordination_accuracy_min']
                    })
                
                # Check neural model accuracy
                if self.system_metrics.neural_model_accuracy < self.alert_thresholds['neural_accuracy_min']:
                    alerts.append({
                        'type': 'neural_accuracy_low',
                        'severity': 'warning',
                        'message': f"Neural model accuracy below threshold: {self.system_metrics.neural_model_accuracy:.1%}",
                        'threshold': self.alert_thresholds['neural_accuracy_min']
                    })
                
                # Check response time
                if self.system_metrics.avg_response_time > self.alert_thresholds['response_time_max']:
                    alerts.append({
                        'type': 'response_time_high',
                        'severity': 'warning',
                        'message': f"Average response time above threshold: {self.system_metrics.avg_response_time:.0f}ms",
                        'threshold': self.alert_thresholds['response_time_max']
                    })
                
                # Check system efficiency
                if self.system_metrics.system_efficiency < self.alert_thresholds['system_efficiency_min']:
                    alerts.append({
                        'type': 'efficiency_low',
                        'severity': 'warning',
                        'message': f"System efficiency below threshold: {self.system_metrics.system_efficiency:.1%}",
                        'threshold': self.alert_thresholds['system_efficiency_min']
                    })
                
                # Check swarm health
                unhealthy_swarms = [
                    name for name, metrics in self.swarm_metrics.items()
                    if metrics.accuracy < self.alert_thresholds['swarm_health_min']
                ]
                
                if unhealthy_swarms:
                    alerts.append({
                        'type': 'swarm_health_low',
                        'severity': 'warning',
                        'message': f"Unhealthy swarms detected: {', '.join(unhealthy_swarms)}",
                        'swarms': unhealthy_swarms
                    })
                
                # Process alerts
                for alert in alerts:
                    await self.trigger_alert(alert['type'], alert)
                
                # Log health status periodically
                if self.system_metrics.uptime_minutes % 10 == 0 and self.system_metrics.uptime_minutes > 0:
                    health_status = "🟢 HEALTHY" if not alerts else "🟡 WARNINGS" if all(a['severity'] == 'warning' for a in alerts) else "🔴 CRITICAL"
                    logger.info(f"🏥 System Health Check: {health_status}")
                    
                    if alerts:
                        for alert in alerts:
                            logger.warning(f"  ⚠️ {alert['message']}")
                
            except Exception as e:
                logger.error(f"Error checking system health: {e}")
                
            await asyncio.sleep(120)  # Check every 2 minutes
    
    async def trigger_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Trigger system alert and notify clients"""
        alert = {
            'type': 'alert',
            'alert_type': alert_type,
            'timestamp': datetime.now().isoformat(),
            'data': alert_data
        }
        
        # Broadcast alert to connected clients
        await self.broadcast_to_clients(alert)
        
        # Log alert
        severity = alert_data.get('severity', 'info')
        message = alert_data.get('message', f'Alert triggered: {alert_type}')
        
        if severity == 'critical':
            logger.error(f"🚨 CRITICAL ALERT: {message}")
        elif severity == 'warning':
            logger.warning(f"⚠️ WARNING: {message}")
        else:
            logger.info(f"ℹ️ INFO: {message}")
    
    async def broadcast_metrics_update(self):
        """Broadcast metrics update to all connected clients"""
        if self.websocket_clients:
            update = {
                'type': 'metrics_update',
                'timestamp': datetime.now().isoformat(),
                'data': self.get_dashboard_data()
            }
            
            await self.broadcast_to_clients(update)
    
    async def broadcast_to_clients(self, message: Dict[str, Any]):
        """Broadcast message to all connected WebSocket clients"""
        if not self.websocket_clients:
            return
            
        message_json = json.dumps(message)
        disconnected_clients = set()
        
        for client in self.websocket_clients:
            try:
                await client.send(message_json)
            except websockets.exceptions.ConnectionClosed:
                disconnected_clients.add(client)
            except Exception as e:
                logger.warning(f"Error sending to client: {e}")
                disconnected_clients.add(client)
        
        # Remove disconnected clients
        self.websocket_clients -= disconnected_clients
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        return {
            'system_metrics': asdict(self.system_metrics),
            'swarm_metrics': {name: asdict(metrics) for name, metrics in self.swarm_metrics.items()},
            'alert_thresholds': self.alert_thresholds,
            'performance_summary': self.get_performance_summary(),
            'last_updated': datetime.now().isoformat()
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.performance_history:
            return {}
        
        recent_data = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        
        return {
            'accuracy_trend': self.calculate_trend([d['coordination_accuracy'] for d in recent_data]),
            'response_trend': self.calculate_trend([d['response_time'] for d in recent_data]),
            'efficiency_trend': self.calculate_trend([d['system_efficiency'] for d in recent_data]),
            'peak_accuracy': max(d['coordination_accuracy'] for d in self.performance_history),
            'min_response_time': min(d['response_time'] for d in self.performance_history),
            'avg_uptime': self.system_metrics.uptime_minutes
        }
    
    def get_swarm_details(self, swarm_id: str) -> Dict[str, Any]:
        """Get detailed information for a specific swarm"""
        for name, metrics in self.swarm_metrics.items():
            if metrics.swarm_id == swarm_id or name == swarm_id:
                return {
                    'swarm_metrics': asdict(metrics),
                    'performance_history': [
                        h for h in self.performance_history[-50:]
                        if h.get('swarm_id') == swarm_id
                    ],
                    'health_status': self.get_swarm_health_status(metrics)
                }
        
        return {'error': f'Swarm not found: {swarm_id}'}
    
    def get_swarm_health_status(self, metrics: SwarmMetrics) -> str:
        """Determine swarm health status"""
        if metrics.accuracy >= 0.95 and metrics.efficiency >= 0.90:
            return 'excellent'
        elif metrics.accuracy >= 0.90 and metrics.efficiency >= 0.85:
            return 'good'
        elif metrics.accuracy >= 0.85 and metrics.efficiency >= 0.80:
            return 'fair'
        else:
            return 'poor'

async def main():
    """Main function to run the monitoring dashboard"""
    print("🚀 ANSF Phase 3: Optimal Mode - Monitoring Dashboard")
    print("=" * 60)
    
    dashboard = Phase3MonitoringDashboard()
    
    try:
        await dashboard.start_monitoring(port=8053)
    except KeyboardInterrupt:
        print("\n🛑 Monitoring dashboard stopped by user")
    except Exception as e:
        print(f"\n❌ Dashboard error: {e}")

if __name__ == "__main__":
    asyncio.run(main())