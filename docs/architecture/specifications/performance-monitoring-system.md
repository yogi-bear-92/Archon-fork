# Performance Monitoring System Specification

## Overview

The Performance Monitoring System provides comprehensive real-time monitoring, metrics collection, and performance analysis across all Master Agent components, enabling data-driven optimization and proactive issue detection.

## Architecture Components

### 1. Metrics Collection Engine

```python
class MetricsCollectionEngine:
    def __init__(self, config):
        self.config = config
        self.collectors = self.initialize_collectors()
        self.metrics_store = MetricsStore(config.storage)
        self.real_time_processor = RealTimeProcessor(config.real_time)
        self.aggregation_engine = AggregationEngine(config.aggregation)
        self.alert_manager = AlertManager(config.alerts)
    
    def initialize_collectors(self):
        """Initialize all metric collectors"""
        return {
            "system": SystemMetricsCollector(),
            "agent": AgentPerformanceCollector(),
            "rag": RAGMetricsCollector(),
            "coordination": CoordinationMetricsCollector(),
            "query": QueryAnalysisMetricsCollector(),
            "network": NetworkMetricsCollector(),
            "user": UserExperienceMetricsCollector()
        }
    
    async def start_collection(self):
        """Start comprehensive metrics collection"""
        
        collection_tasks = []
        
        # Start all collectors
        for collector_name, collector in self.collectors.items():
            collection_task = asyncio.create_task(
                self.run_collector(collector_name, collector)
            )
            collection_tasks.append(collection_task)
        
        # Start real-time processing
        processing_task = asyncio.create_task(
            self.real_time_processor.start_processing()
        )
        collection_tasks.append(processing_task)
        
        # Start aggregation engine
        aggregation_task = asyncio.create_task(
            self.aggregation_engine.start_aggregation()
        )
        collection_tasks.append(aggregation_task)
        
        return collection_tasks
    
    async def run_collector(self, collector_name, collector):
        """Run individual metrics collector"""
        
        while True:
            try:
                # Collect metrics
                metrics = await collector.collect()
                
                # Add metadata
                enriched_metrics = self.enrich_metrics(collector_name, metrics)
                
                # Store metrics
                await self.metrics_store.store_metrics(enriched_metrics)
                
                # Process for real-time alerts
                await self.real_time_processor.process_metrics(enriched_metrics)
                
                # Wait for next collection interval
                await asyncio.sleep(collector.collection_interval)
                
            except Exception as e:
                print(f"Error in {collector_name} collector: {e}")
                await asyncio.sleep(30)  # Backoff on errors
    
    def enrich_metrics(self, collector_name, metrics):
        """Enrich metrics with metadata and context"""
        
        current_time = datetime.utcnow()
        
        enriched = []
        for metric in metrics:
            enriched_metric = {
                **metric,
                "collector": collector_name,
                "timestamp": current_time.isoformat(),
                "hostname": self.config.hostname,
                "environment": self.config.environment,
                "version": self.config.version
            }
            enriched.append(enriched_metric)
        
        return enriched
```

### 2. System Metrics Collector

```python
class SystemMetricsCollector:
    def __init__(self):
        self.collection_interval = 10  # seconds
        self.previous_cpu_times = None
        self.previous_network_stats = None
    
    async def collect(self):
        """Collect comprehensive system metrics"""
        
        current_time = datetime.utcnow()
        metrics = []
        
        # CPU Metrics
        cpu_metrics = await self.collect_cpu_metrics()
        metrics.extend(cpu_metrics)
        
        # Memory Metrics
        memory_metrics = await self.collect_memory_metrics()
        metrics.extend(memory_metrics)
        
        # Disk Metrics
        disk_metrics = await self.collect_disk_metrics()
        metrics.extend(disk_metrics)
        
        # Network Metrics
        network_metrics = await self.collect_network_metrics()
        metrics.extend(network_metrics)
        
        # Process Metrics
        process_metrics = await self.collect_process_metrics()
        metrics.extend(process_metrics)
        
        return metrics
    
    async def collect_cpu_metrics(self):
        """Collect CPU performance metrics"""
        
        import psutil
        
        # CPU utilization
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_count_logical = psutil.cpu_count(logical=True)
        
        # CPU times
        cpu_times = psutil.cpu_times()
        
        # Load average (Unix/Linux)
        try:
            load_avg = psutil.getloadavg()
        except (AttributeError, OSError):
            load_avg = [0, 0, 0]  # Fallback for Windows
        
        return [
            {
                "metric_name": "cpu_utilization_percent",
                "value": cpu_percent,
                "unit": "percent",
                "tags": {"component": "system"}
            },
            {
                "metric_name": "cpu_count_physical",
                "value": cpu_count,
                "unit": "count",
                "tags": {"component": "system"}
            },
            {
                "metric_name": "cpu_count_logical",
                "value": cpu_count_logical,
                "unit": "count",
                "tags": {"component": "system"}
            },
            {
                "metric_name": "load_average_1m",
                "value": load_avg[0],
                "unit": "load",
                "tags": {"component": "system", "period": "1m"}
            },
            {
                "metric_name": "load_average_5m",
                "value": load_avg[1],
                "unit": "load",
                "tags": {"component": "system", "period": "5m"}
            },
            {
                "metric_name": "load_average_15m",
                "value": load_avg[2],
                "unit": "load",
                "tags": {"component": "system", "period": "15m"}
            }
        ]
    
    async def collect_memory_metrics(self):
        """Collect memory usage metrics"""
        
        import psutil
        
        # Virtual memory
        virtual_memory = psutil.virtual_memory()
        
        # Swap memory
        swap_memory = psutil.swap_memory()
        
        return [
            {
                "metric_name": "memory_total_bytes",
                "value": virtual_memory.total,
                "unit": "bytes",
                "tags": {"component": "system", "type": "virtual"}
            },
            {
                "metric_name": "memory_available_bytes",
                "value": virtual_memory.available,
                "unit": "bytes",
                "tags": {"component": "system", "type": "virtual"}
            },
            {
                "metric_name": "memory_used_bytes",
                "value": virtual_memory.used,
                "unit": "bytes",
                "tags": {"component": "system", "type": "virtual"}
            },
            {
                "metric_name": "memory_utilization_percent",
                "value": virtual_memory.percent,
                "unit": "percent",
                "tags": {"component": "system", "type": "virtual"}
            },
            {
                "metric_name": "swap_total_bytes",
                "value": swap_memory.total,
                "unit": "bytes",
                "tags": {"component": "system", "type": "swap"}
            },
            {
                "metric_name": "swap_used_bytes",
                "value": swap_memory.used,
                "unit": "bytes",
                "tags": {"component": "system", "type": "swap"}
            },
            {
                "metric_name": "swap_utilization_percent",
                "value": swap_memory.percent,
                "unit": "percent",
                "tags": {"component": "system", "type": "swap"}
            }
        ]
```

### 3. Agent Performance Collector

```python
class AgentPerformanceCollector:
    def __init__(self):
        self.collection_interval = 15  # seconds
        self.agent_registry = None  # Injected during initialization
        self.performance_history = {}
    
    async def collect(self):
        """Collect agent performance metrics"""
        
        if not self.agent_registry:
            return []
        
        metrics = []
        current_time = datetime.utcnow()
        
        # Get all active agents
        active_agents = await self.agent_registry.get_active_agents()
        
        for agent in active_agents:
            agent_metrics = await self.collect_agent_metrics(agent, current_time)
            metrics.extend(agent_metrics)
        
        # Collect swarm-level metrics
        swarm_metrics = await self.collect_swarm_metrics(active_agents, current_time)
        metrics.extend(swarm_metrics)
        
        return metrics
    
    async def collect_agent_metrics(self, agent, timestamp):
        """Collect metrics for individual agent"""
        
        agent_id = agent.agent_id
        agent_metrics = []
        
        # Basic performance metrics
        performance = agent.performance_metrics
        
        agent_metrics.extend([
            {
                "metric_name": "agent_success_rate",
                "value": performance.get("success_rate", 0),
                "unit": "ratio",
                "tags": {
                    "agent_id": agent_id,
                    "agent_type": agent.category,
                    "component": "agent"
                }
            },
            {
                "metric_name": "agent_avg_response_time",
                "value": self.parse_response_time(performance.get("avg_response_time", "0s")),
                "unit": "seconds",
                "tags": {
                    "agent_id": agent_id,
                    "agent_type": agent.category,
                    "component": "agent"
                }
            },
            {
                "metric_name": "agent_current_load",
                "value": agent.current_status.get("current_load", 0),
                "unit": "ratio",
                "tags": {
                    "agent_id": agent_id,
                    "agent_type": agent.category,
                    "component": "agent"
                }
            },
            {
                "metric_name": "agent_queue_depth",
                "value": agent.current_status.get("queue_depth", 0),
                "unit": "count",
                "tags": {
                    "agent_id": agent_id,
                    "agent_type": agent.category,
                    "component": "agent"
                }
            }
        ])
        
        # Availability status
        availability = 1.0 if agent.current_status.get("availability") == "available" else 0.0
        agent_metrics.append({
            "metric_name": "agent_availability",
            "value": availability,
            "unit": "binary",
            "tags": {
                "agent_id": agent_id,
                "agent_type": agent.category,
                "component": "agent"
            }
        })
        
        # Task completion metrics
        if agent_id in self.performance_history:
            history = self.performance_history[agent_id]
            
            # Tasks per minute
            recent_tasks = self.count_recent_tasks(history, minutes=5)
            tasks_per_minute = recent_tasks / 5.0
            
            agent_metrics.append({
                "metric_name": "agent_tasks_per_minute",
                "value": tasks_per_minute,
                "unit": "per_minute",
                "tags": {
                    "agent_id": agent_id,
                    "agent_type": agent.category,
                    "component": "agent"
                }
            })
            
            # Error rate
            recent_errors = self.count_recent_errors(history, minutes=15)
            total_recent_tasks = self.count_recent_tasks(history, minutes=15)
            error_rate = recent_errors / max(total_recent_tasks, 1)
            
            agent_metrics.append({
                "metric_name": "agent_error_rate",
                "value": error_rate,
                "unit": "ratio",
                "tags": {
                    "agent_id": agent_id,
                    "agent_type": agent.category,
                    "component": "agent"
                }
            })
        
        return agent_metrics
    
    async def collect_swarm_metrics(self, active_agents, timestamp):
        """Collect swarm-level performance metrics"""
        
        swarm_metrics = []
        
        # Total agents
        total_agents = len(active_agents)
        available_agents = len([a for a in active_agents 
                              if a.current_status.get("availability") == "available"])
        
        swarm_metrics.extend([
            {
                "metric_name": "swarm_total_agents",
                "value": total_agents,
                "unit": "count",
                "tags": {"component": "swarm"}
            },
            {
                "metric_name": "swarm_available_agents",
                "value": available_agents,
                "unit": "count",
                "tags": {"component": "swarm"}
            },
            {
                "metric_name": "swarm_availability_ratio",
                "value": available_agents / max(total_agents, 1),
                "unit": "ratio",
                "tags": {"component": "swarm"}
            }
        ])
        
        # Average metrics across all agents
        if active_agents:
            avg_success_rate = sum(
                a.performance_metrics.get("success_rate", 0) 
                for a in active_agents
            ) / len(active_agents)
            
            avg_load = sum(
                a.current_status.get("current_load", 0)
                for a in active_agents
            ) / len(active_agents)
            
            total_queue_depth = sum(
                a.current_status.get("queue_depth", 0)
                for a in active_agents
            )
            
            swarm_metrics.extend([
                {
                    "metric_name": "swarm_avg_success_rate",
                    "value": avg_success_rate,
                    "unit": "ratio",
                    "tags": {"component": "swarm"}
                },
                {
                    "metric_name": "swarm_avg_load",
                    "value": avg_load,
                    "unit": "ratio",
                    "tags": {"component": "swarm"}
                },
                {
                    "metric_name": "swarm_total_queue_depth",
                    "value": total_queue_depth,
                    "unit": "count",
                    "tags": {"component": "swarm"}
                }
            ])
        
        return swarm_metrics
```

### 4. RAG Performance Collector

```python
class RAGMetricsCollector:
    def __init__(self):
        self.collection_interval = 20  # seconds
        self.rag_performance_tracker = None  # Injected
        self.query_cache_stats = None  # Injected
    
    async def collect(self):
        """Collect RAG system performance metrics"""
        
        metrics = []
        current_time = datetime.utcnow()
        
        # Query performance metrics
        if self.rag_performance_tracker:
            query_metrics = await self.collect_query_performance_metrics()
            metrics.extend(query_metrics)
        
        # Cache performance metrics
        if self.query_cache_stats:
            cache_metrics = await self.collect_cache_metrics()
            metrics.extend(cache_metrics)
        
        # Knowledge base metrics
        kb_metrics = await self.collect_knowledge_base_metrics()
        metrics.extend(kb_metrics)
        
        # Retrieval quality metrics
        quality_metrics = await self.collect_retrieval_quality_metrics()
        metrics.extend(quality_metrics)
        
        return metrics
    
    async def collect_query_performance_metrics(self):
        """Collect RAG query performance metrics"""
        
        performance_stats = await self.rag_performance_tracker.get_recent_stats(
            time_window_minutes=15
        )
        
        return [
            {
                "metric_name": "rag_query_count",
                "value": performance_stats.get("total_queries", 0),
                "unit": "count",
                "tags": {"component": "rag", "window": "15m"}
            },
            {
                "metric_name": "rag_avg_query_time",
                "value": performance_stats.get("avg_query_time", 0),
                "unit": "seconds",
                "tags": {"component": "rag", "window": "15m"}
            },
            {
                "metric_name": "rag_median_query_time",
                "value": performance_stats.get("median_query_time", 0),
                "unit": "seconds",
                "tags": {"component": "rag", "window": "15m"}
            },
            {
                "metric_name": "rag_p95_query_time",
                "value": performance_stats.get("p95_query_time", 0),
                "unit": "seconds",
                "tags": {"component": "rag", "window": "15m"}
            },
            {
                "metric_name": "rag_success_rate",
                "value": performance_stats.get("success_rate", 0),
                "unit": "ratio",
                "tags": {"component": "rag", "window": "15m"}
            }
        ]
    
    async def collect_cache_metrics(self):
        """Collect RAG cache performance metrics"""
        
        cache_stats = await self.query_cache_stats.get_cache_stats()
        
        return [
            {
                "metric_name": "rag_cache_hit_rate",
                "value": cache_stats.get("hit_rate", 0),
                "unit": "ratio",
                "tags": {"component": "rag", "subsystem": "cache"}
            },
            {
                "metric_name": "rag_cache_total_hits",
                "value": cache_stats.get("total_hits", 0),
                "unit": "count",
                "tags": {"component": "rag", "subsystem": "cache"}
            },
            {
                "metric_name": "rag_cache_total_misses",
                "value": cache_stats.get("total_misses", 0),
                "unit": "count",
                "tags": {"component": "rag", "subsystem": "cache"}
            },
            {
                "metric_name": "rag_cache_size",
                "value": cache_stats.get("cache_size", 0),
                "unit": "count",
                "tags": {"component": "rag", "subsystem": "cache"}
            }
        ]
    
    async def collect_retrieval_quality_metrics(self):
        """Collect RAG retrieval quality metrics"""
        
        quality_stats = await self.rag_performance_tracker.get_quality_stats(
            time_window_minutes=30
        )
        
        return [
            {
                "metric_name": "rag_avg_relevance_score",
                "value": quality_stats.get("avg_relevance_score", 0),
                "unit": "score",
                "tags": {"component": "rag", "aspect": "quality"}
            },
            {
                "metric_name": "rag_avg_confidence_score",
                "value": quality_stats.get("avg_confidence_score", 0),
                "unit": "score",
                "tags": {"component": "rag", "aspect": "quality"}
            },
            {
                "metric_name": "rag_low_confidence_queries_ratio",
                "value": quality_stats.get("low_confidence_ratio", 0),
                "unit": "ratio",
                "tags": {"component": "rag", "aspect": "quality"}
            },
            {
                "metric_name": "rag_fallback_queries_ratio",
                "value": quality_stats.get("fallback_ratio", 0),
                "unit": "ratio",
                "tags": {"component": "rag", "aspect": "quality"}
            }
        ]
```

### 5. Real-time Processing Engine

```python
class RealTimeProcessor:
    def __init__(self, config):
        self.config = config
        self.processing_queue = asyncio.Queue(maxsize=10000)
        self.anomaly_detector = AnomalyDetector(config.anomaly_detection)
        self.threshold_manager = ThresholdManager(config.thresholds)
        self.alert_manager = AlertManager(config.alerts)
        self.trend_analyzer = TrendAnalyzer(config.trends)
    
    async def start_processing(self):
        """Start real-time metrics processing"""
        
        processing_tasks = []
        
        # Start main processing loop
        main_processor = asyncio.create_task(
            self.main_processing_loop()
        )
        processing_tasks.append(main_processor)
        
        # Start anomaly detection
        anomaly_detector = asyncio.create_task(
            self.anomaly_detection_loop()
        )
        processing_tasks.append(anomaly_detector)
        
        # Start trend analysis
        trend_analyzer = asyncio.create_task(
            self.trend_analysis_loop()
        )
        processing_tasks.append(trend_analyzer)
        
        return processing_tasks
    
    async def process_metrics(self, metrics):
        """Queue metrics for real-time processing"""
        
        try:
            await self.processing_queue.put(metrics)
        except asyncio.QueueFull:
            print("Real-time processing queue full, dropping metrics")
    
    async def main_processing_loop(self):
        """Main real-time processing loop"""
        
        while True:
            try:
                # Get metrics from queue (with timeout)
                metrics = await asyncio.wait_for(
                    self.processing_queue.get(),
                    timeout=10.0
                )
                
                # Process metrics
                await self.process_metrics_batch(metrics)
                
            except asyncio.TimeoutError:
                # No metrics received, continue loop
                continue
            except Exception as e:
                print(f"Error in real-time processing: {e}")
                await asyncio.sleep(5)
    
    async def process_metrics_batch(self, metrics):
        """Process batch of metrics in real-time"""
        
        processing_tasks = []
        
        for metric in metrics:
            # Check thresholds
            threshold_task = asyncio.create_task(
                self.check_thresholds(metric)
            )
            processing_tasks.append(threshold_task)
            
            # Detect anomalies
            anomaly_task = asyncio.create_task(
                self.detect_anomalies(metric)
            )
            processing_tasks.append(anomaly_task)
            
            # Update trends
            trend_task = asyncio.create_task(
                self.update_trends(metric)
            )
            processing_tasks.append(trend_task)
        
        # Wait for all processing to complete
        await asyncio.gather(*processing_tasks, return_exceptions=True)
    
    async def check_thresholds(self, metric):
        """Check metric against configured thresholds"""
        
        metric_name = metric.get("metric_name")
        value = metric.get("value", 0)
        tags = metric.get("tags", {})
        
        # Get applicable thresholds
        thresholds = await self.threshold_manager.get_thresholds(
            metric_name, tags
        )
        
        for threshold in thresholds:
            if self.threshold_exceeded(value, threshold):
                # Generate alert
                alert = {
                    "type": "threshold_exceeded",
                    "metric_name": metric_name,
                    "current_value": value,
                    "threshold_value": threshold["value"],
                    "threshold_type": threshold["type"],
                    "severity": threshold["severity"],
                    "tags": tags,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await self.alert_manager.send_alert(alert)
    
    def threshold_exceeded(self, value, threshold):
        """Check if value exceeds threshold"""
        
        threshold_type = threshold["type"]
        threshold_value = threshold["value"]
        
        if threshold_type == "greater_than":
            return value > threshold_value
        elif threshold_type == "less_than":
            return value < threshold_value
        elif threshold_type == "greater_than_or_equal":
            return value >= threshold_value
        elif threshold_type == "less_than_or_equal":
            return value <= threshold_value
        else:
            return False
```

### 6. Performance Dashboard and Visualization

```python
class PerformanceDashboard:
    def __init__(self, metrics_store, config):
        self.metrics_store = metrics_store
        self.config = config
        self.dashboard_config = self.initialize_dashboard_config()
        self.widget_generators = self.initialize_widget_generators()
    
    def initialize_dashboard_config(self):
        """Initialize dashboard configuration"""
        return {
            "refresh_interval": 30,  # seconds
            "time_ranges": ["5m", "15m", "1h", "4h", "24h"],
            "default_time_range": "1h",
            "widgets": [
                {
                    "id": "system_overview",
                    "title": "System Overview",
                    "type": "multi_metric",
                    "metrics": ["cpu_utilization_percent", "memory_utilization_percent", "load_average_1m"]
                },
                {
                    "id": "agent_performance",
                    "title": "Agent Performance",
                    "type": "agent_grid",
                    "metrics": ["agent_success_rate", "agent_avg_response_time", "agent_availability"]
                },
                {
                    "id": "rag_performance",
                    "title": "RAG Performance",
                    "type": "time_series",
                    "metrics": ["rag_avg_query_time", "rag_success_rate", "rag_cache_hit_rate"]
                },
                {
                    "id": "swarm_coordination",
                    "title": "Swarm Coordination",
                    "type": "network_graph",
                    "metrics": ["swarm_total_agents", "swarm_availability_ratio"]
                },
                {
                    "id": "alerts_summary",
                    "title": "Recent Alerts",
                    "type": "alert_list",
                    "metrics": []
                }
            ]
        }
    
    async def generate_dashboard_data(self, time_range="1h"):
        """Generate complete dashboard data"""
        
        dashboard_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "time_range": time_range,
            "widgets": {}
        }
        
        # Generate data for each widget
        for widget_config in self.dashboard_config["widgets"]:
            widget_id = widget_config["id"]
            widget_type = widget_config["type"]
            
            generator = self.widget_generators.get(widget_type)
            if generator:
                try:
                    widget_data = await generator.generate_data(widget_config, time_range)
                    dashboard_data["widgets"][widget_id] = widget_data
                except Exception as e:
                    print(f"Error generating widget {widget_id}: {e}")
                    dashboard_data["widgets"][widget_id] = {"error": str(e)}
        
        return dashboard_data
    
    async def get_performance_summary(self, time_range="1h"):
        """Get high-level performance summary"""
        
        end_time = datetime.utcnow()
        start_time = end_time - self.parse_time_range(time_range)
        
        # Get key metrics
        summary_metrics = await self.metrics_store.get_summary_metrics(
            start_time=start_time,
            end_time=end_time,
            metrics=[
                "cpu_utilization_percent",
                "memory_utilization_percent", 
                "swarm_avg_success_rate",
                "swarm_availability_ratio",
                "rag_avg_query_time",
                "rag_success_rate"
            ]
        )
        
        # Calculate performance score
        performance_score = self.calculate_performance_score(summary_metrics)
        
        # Get recent alerts
        recent_alerts = await self.get_recent_alerts(limit=10)
        
        return {
            "performance_score": performance_score,
            "system_health": self.assess_system_health(summary_metrics),
            "key_metrics": summary_metrics,
            "recent_alerts": recent_alerts,
            "recommendations": await self.generate_recommendations(summary_metrics)
        }
    
    def calculate_performance_score(self, metrics):
        """Calculate overall performance score (0-100)"""
        
        scores = []
        
        # System resource efficiency (weight: 25%)
        cpu_util = metrics.get("cpu_utilization_percent", {}).get("avg", 50)
        memory_util = metrics.get("memory_utilization_percent", {}).get("avg", 50)
        
        # Optimal utilization is around 70% - penalize both under and over utilization
        cpu_score = max(0, 100 - abs(cpu_util - 70) * 2)
        memory_score = max(0, 100 - abs(memory_util - 70) * 2)
        system_score = (cpu_score + memory_score) / 2
        scores.append(("system", system_score, 0.25))
        
        # Agent performance (weight: 35%)
        agent_success = metrics.get("swarm_avg_success_rate", {}).get("avg", 0.8) * 100
        agent_availability = metrics.get("swarm_availability_ratio", {}).get("avg", 0.9) * 100
        agent_score = (agent_success + agent_availability) / 2
        scores.append(("agents", agent_score, 0.35))
        
        # RAG performance (weight: 25%)
        rag_success = metrics.get("rag_success_rate", {}).get("avg", 0.8) * 100
        rag_speed = min(100, max(0, 100 - metrics.get("rag_avg_query_time", {}).get("avg", 1) * 20))
        rag_score = (rag_success + rag_speed) / 2
        scores.append(("rag", rag_score, 0.25))
        
        # Stability (weight: 15%) - based on absence of critical alerts
        stability_score = 90  # Default, would be adjusted based on alert frequency
        scores.append(("stability", stability_score, 0.15))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in scores)
        
        return {
            "overall": round(total_score, 1),
            "components": {name: round(score, 1) for name, score, _ in scores}
        }
```

### 7. Alert Management System

```python
class AlertManager:
    def __init__(self, config):
        self.config = config
        self.alert_channels = self.initialize_alert_channels()
        self.alert_rules = self.load_alert_rules()
        self.alert_history = AlertHistory(config.history)
        self.rate_limiter = AlertRateLimiter(config.rate_limiting)
    
    def initialize_alert_channels(self):
        """Initialize alert delivery channels"""
        
        channels = {}
        
        if self.config.get("email", {}).get("enabled", False):
            channels["email"] = EmailAlertChannel(self.config["email"])
        
        if self.config.get("slack", {}).get("enabled", False):
            channels["slack"] = SlackAlertChannel(self.config["slack"])
        
        if self.config.get("webhook", {}).get("enabled", False):
            channels["webhook"] = WebhookAlertChannel(self.config["webhook"])
        
        # Always include console for debugging
        channels["console"] = ConsoleAlertChannel()
        
        return channels
    
    async def send_alert(self, alert):
        """Send alert through configured channels"""
        
        # Check rate limiting
        if await self.rate_limiter.is_rate_limited(alert):
            return
        
        # Enrich alert with context
        enriched_alert = await self.enrich_alert(alert)
        
        # Determine channels based on severity
        channels_to_use = self.select_channels_for_severity(
            enriched_alert.get("severity", "medium")
        )
        
        # Send through selected channels
        send_tasks = []
        for channel_name in channels_to_use:
            if channel_name in self.alert_channels:
                channel = self.alert_channels[channel_name]
                send_task = asyncio.create_task(
                    channel.send_alert(enriched_alert)
                )
                send_tasks.append(send_task)
        
        # Wait for all sends to complete
        await asyncio.gather(*send_tasks, return_exceptions=True)
        
        # Record alert in history
        await self.alert_history.record_alert(enriched_alert)
    
    async def enrich_alert(self, alert):
        """Enrich alert with additional context"""
        
        enriched = alert.copy()
        
        # Add system context
        enriched["hostname"] = self.config.get("hostname", "unknown")
        enriched["environment"] = self.config.get("environment", "unknown")
        
        # Add related metrics if available
        if "metric_name" in alert:
            related_metrics = await self.get_related_metrics(
                alert["metric_name"], 
                alert.get("tags", {})
            )
            enriched["related_metrics"] = related_metrics
        
        # Add historical context
        similar_alerts = await self.alert_history.get_similar_alerts(
            alert, limit=5, time_window_hours=24
        )
        enriched["recent_similar_alerts"] = len(similar_alerts)
        
        return enriched
    
    def select_channels_for_severity(self, severity):
        """Select appropriate channels based on alert severity"""
        
        channel_config = {
            "low": ["console"],
            "medium": ["console", "webhook"],
            "high": ["console", "webhook", "slack"],
            "critical": ["console", "webhook", "slack", "email"]
        }
        
        return channel_config.get(severity, ["console"])

class ConsoleAlertChannel:
    """Console alert channel for development and debugging"""
    
    async def send_alert(self, alert):
        """Send alert to console"""
        
        severity = alert.get("severity", "unknown").upper()
        metric_name = alert.get("metric_name", "unknown")
        current_value = alert.get("current_value", "N/A")
        threshold_value = alert.get("threshold_value", "N/A")
        timestamp = alert.get("timestamp", datetime.utcnow().isoformat())
        
        print(f"""
[{severity} ALERT] {timestamp}
Metric: {metric_name}
Current Value: {current_value}
Threshold: {threshold_value}
Tags: {alert.get('tags', {})}
""")
```

This Performance Monitoring System provides comprehensive real-time monitoring capabilities, enabling the Master Agent to maintain optimal performance through data-driven insights, proactive alerting, and detailed performance analytics across all system components.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Analyze system requirements and create architecture overview", "status": "completed", "activeForm": "Analyzed system requirements and created architecture overview"}, {"content": "Design Query Analysis Engine with NLP capabilities", "status": "completed", "activeForm": "Designed Query Analysis Engine with NLP capabilities"}, {"content": "Create Agent Capability Matrix and routing algorithms", "status": "completed", "activeForm": "Created Agent Capability Matrix and routing algorithms"}, {"content": "Design RAG Integration Layer with contextual embeddings", "status": "completed", "activeForm": "Designed RAG Integration Layer with contextual embeddings"}, {"content": "Implement Coordination Protocol Handler with fault tolerance", "status": "completed", "activeForm": "Implemented Coordination Protocol Handler with fault tolerance"}, {"content": "Build Performance Monitoring System with real-time metrics", "status": "completed", "activeForm": "Built Performance Monitoring System with real-time metrics"}, {"content": "Create C4 architecture diagrams and component specifications", "status": "completed", "activeForm": "Created C4 architecture diagrams and component specifications"}, {"content": "Design integration points with Archon MCP and Claude Flow", "status": "completed", "activeForm": "Designed integration points with Archon MCP and Claude Flow"}, {"content": "Implement SPARC methodology compliance framework", "status": "pending", "activeForm": "Implementing SPARC methodology compliance framework"}, {"content": "Create implementation guidelines and deployment strategy", "status": "in_progress", "activeForm": "Creating implementation guidelines and deployment strategy"}]