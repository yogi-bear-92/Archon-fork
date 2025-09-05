# Master Agent System - Operational Guide

## Table of Contents

- [Deployment Procedures](#deployment-procedures)
- [Monitoring and Alerting](#monitoring-and-alerting)
- [Maintenance Procedures](#maintenance-procedures)
- [Performance Optimization](#performance-optimization)
- [Health Checks](#health-checks)
- [Incident Response](#incident-response)
- [Backup and Recovery](#backup-and-recovery)

---

## Deployment Procedures

### Production Deployment Architecture

The Master Agent System is designed for robust production deployment with multiple environment support and zero-downtime deployment capabilities.

#### Infrastructure Requirements

```yaml
Production Infrastructure:
  
  Minimum Requirements:
    CPU: 8 cores (16 cores recommended)
    RAM: 16GB (32GB recommended)
    Storage: 100GB SSD (500GB recommended)
    Network: 1Gbps connection
    
  Recommended Production Setup:
    Load Balancer: NGINX or HAProxy
    Application Servers: 3+ instances for high availability
    Database: PostgreSQL with replication (Primary + 2 replicas)
    Cache Layer: Redis cluster (3 nodes minimum)
    Monitoring: Prometheus + Grafana + AlertManager
    Logging: ELK Stack or similar centralized logging
    
  Kubernetes Resources (per service):
    archon-server:
      requests: { cpu: "1000m", memory: "2Gi" }
      limits: { cpu: "2000m", memory: "4Gi" }
      replicas: 3
      
    archon-agents:
      requests: { cpu: "2000m", memory: "4Gi" }
      limits: { cpu: "4000m", memory: "8Gi" }
      replicas: 2
      
    archon-mcp:
      requests: { cpu: "500m", memory: "1Gi" }
      limits: { cpu: "1000m", memory: "2Gi" }
      replicas: 2
```

#### Deployment Configurations

##### Docker Compose Production

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  archon-server:
    image: archon/server:${VERSION}
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
      restart_policy:
        condition: on-failure
        max_attempts: 3
    environment:
      - ENVIRONMENT=production
      - LOG_LEVEL=INFO
      - WORKERS=4
      - MAX_CONNECTIONS=500
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_SERVICE_KEY=${SUPABASE_SERVICE_KEY}
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8181/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - archon-network
      
  archon-agents:
    image: archon/agents:${VERSION}
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
    environment:
      - ENVIRONMENT=production
      - GPU_ENABLED=${GPU_ENABLED:-false}
      - MODEL_CACHE_SIZE=10GB
    volumes:
      - model-cache:/app/models
    networks:
      - archon-network

  nginx-proxy:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    depends_on:
      - archon-server
    networks:
      - archon-network

networks:
  archon-network:
    driver: overlay
    attachable: true

volumes:
  model-cache:
    driver: local
```

##### Kubernetes Deployment

```yaml
# k8s/archon-server-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: archon-server
  labels:
    app: archon-server
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: archon-server
  template:
    metadata:
      labels:
        app: archon-server
    spec:
      containers:
      - name: archon-server
        image: archon/server:${VERSION}
        ports:
        - containerPort: 8181
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: SUPABASE_URL
          valueFrom:
            secretKeyRef:
              name: archon-secrets
              key: supabase-url
        - name: SUPABASE_SERVICE_KEY
          valueFrom:
            secretKeyRef:
              name: archon-secrets
              key: supabase-service-key
        resources:
          requests:
            cpu: "1000m"
            memory: "2Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8181
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8181
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2

---
apiVersion: v1
kind: Service
metadata:
  name: archon-server-service
spec:
  selector:
    app: archon-server
  ports:
  - name: http
    port: 80
    targetPort: 8181
    protocol: TCP
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: archon-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - archon.yourdomain.com
    secretName: archon-tls
  rules:
  - host: archon.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: archon-server-service
            port:
              number: 80
```

#### Deployment Scripts

```bash
#!/bin/bash
# deploy.sh - Production deployment script

set -e

# Configuration
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
NAMESPACE=${3:-archon}

echo "Deploying Archon Master Agent System"
echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"
echo "Namespace: $NAMESPACE"

# Pre-deployment checks
echo "Running pre-deployment checks..."
./scripts/pre-deployment-checks.sh

# Database migrations
echo "Running database migrations..."
kubectl exec -n $NAMESPACE deployment/archon-server -- python -m alembic upgrade head

# Update secrets
echo "Updating secrets..."
kubectl apply -f k8s/secrets/${ENVIRONMENT}-secrets.yaml -n $NAMESPACE

# Deploy services with rolling update
echo "Deploying services..."

# Deploy in dependency order
kubectl apply -f k8s/archon-server-deployment.yaml -n $NAMESPACE
kubectl rollout status deployment/archon-server -n $NAMESPACE --timeout=600s

kubectl apply -f k8s/archon-agents-deployment.yaml -n $NAMESPACE  
kubectl rollout status deployment/archon-agents -n $NAMESPACE --timeout=600s

kubectl apply -f k8s/archon-mcp-deployment.yaml -n $NAMESPACE
kubectl rollout status deployment/archon-mcp -n $NAMESPACE --timeout=600s

# Update ingress and services
kubectl apply -f k8s/ingress.yaml -n $NAMESPACE
kubectl apply -f k8s/services.yaml -n $NAMESPACE

# Post-deployment verification
echo "Running post-deployment verification..."
./scripts/post-deployment-checks.sh $ENVIRONMENT

echo "Deployment completed successfully!"
```

#### Blue-Green Deployment

```bash
#!/bin/bash
# blue-green-deploy.sh - Zero downtime deployment

CURRENT_ENV=$(kubectl get service archon-server-active -o jsonpath='{.spec.selector.version}')
NEW_ENV=$([ "$CURRENT_ENV" = "blue" ] && echo "green" || echo "blue")

echo "Current environment: $CURRENT_ENV"
echo "Deploying to: $NEW_ENV"

# Deploy new version to inactive environment
sed "s/VERSION_PLACEHOLDER/$NEW_ENV/g" k8s/archon-deployment-template.yaml | kubectl apply -f -

# Wait for new environment to be ready
kubectl wait --for=condition=available --timeout=600s deployment/archon-server-$NEW_ENV

# Run health checks on new environment
./scripts/health-check.sh $NEW_ENV

if [ $? -eq 0 ]; then
    echo "Health checks passed. Switching traffic..."
    
    # Update active service to point to new environment
    kubectl patch service archon-server-active -p '{"spec":{"selector":{"version":"'$NEW_ENV'"}}}'
    
    echo "Traffic switched to $NEW_ENV environment"
    
    # Wait and verify
    sleep 30
    ./scripts/health-check.sh production
    
    if [ $? -eq 0 ]; then
        echo "Deployment successful. Cleaning up old environment..."
        kubectl delete deployment archon-server-$CURRENT_ENV
        kubectl delete deployment archon-agents-$CURRENT_ENV
        kubectl delete deployment archon-mcp-$CURRENT_ENV
    else
        echo "Post-switch health check failed. Rolling back..."
        kubectl patch service archon-server-active -p '{"spec":{"selector":{"version":"'$CURRENT_ENV'"}}}'
        exit 1
    fi
else
    echo "Health checks failed. Aborting deployment."
    kubectl delete deployment archon-server-$NEW_ENV
    kubectl delete deployment archon-agents-$NEW_ENV
    kubectl delete deployment archon-mcp-$NEW_ENV
    exit 1
fi
```

---

## Monitoring and Alerting

### Comprehensive Monitoring Stack

#### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "archon-rules.yml"

scrape_configs:
  - job_name: 'archon-server'
    static_configs:
      - targets: ['archon-server:8181']
    metrics_path: /metrics
    scrape_interval: 10s
    
  - job_name: 'archon-agents'
    static_configs:
      - targets: ['archon-agents:8052']
    metrics_path: /metrics
    scrape_interval: 15s
    
  - job_name: 'archon-mcp'
    static_configs:
      - targets: ['archon-mcp:8051']
    metrics_path: /metrics
    scrape_interval: 10s

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

#### Key Metrics and Alerts

```yaml
# archon-rules.yml
groups:
- name: archon.rules
  rules:
  
  # Response Time Alerts
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is {{ $value }}s for {{ $labels.instance }}"
      
  - alert: VeryHighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 3
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Very high response time detected"
      description: "95th percentile response time is {{ $value }}s for {{ $labels.instance }}"

  # Error Rate Alerts
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value | humanizePercentage }} for {{ $labels.instance }}"
      
  - alert: CriticalErrorRate  
    expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.20
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Critical error rate detected"
      description: "Error rate is {{ $value | humanizePercentage }} for {{ $labels.instance }}"

  # Resource Usage Alerts
  - alert: HighCPUUsage
    expr: rate(process_cpu_seconds_total[5m]) * 100 > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is {{ $value }}% for {{ $labels.instance }}"
      
  - alert: HighMemoryUsage
    expr: process_resident_memory_bytes / 1024 / 1024 / 1024 > 6
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is {{ $value }}GB for {{ $labels.instance }}"

  # Agent-Specific Alerts
  - alert: AgentQueueBacklog
    expr: archon_agent_queue_size > 100
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "Agent queue backlog detected"
      description: "Agent queue size is {{ $value }} for {{ $labels.agent_type }}"
      
  - alert: RAGQueryLatency
    expr: histogram_quantile(0.95, rate(archon_rag_query_duration_seconds_bucket[5m])) > 2
    for: 3m
    labels:
      severity: warning
    annotations:
      summary: "RAG query latency high"
      description: "RAG query 95th percentile latency is {{ $value }}s"

  # Database Alerts
  - alert: DatabaseConnectionsHigh
    expr: pg_stat_database_numbackends / pg_settings_max_connections * 100 > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Database connections high"
      description: "Database connections at {{ $value }}% of maximum"
      
  - alert: VectorSearchLatency
    expr: histogram_quantile(0.95, rate(archon_vector_search_duration_seconds_bucket[5m])) > 1
    for: 3m
    labels:
      severity: warning
    annotations:
      summary: "Vector search latency high" 
      description: "Vector search 95th percentile latency is {{ $value }}s"
```

#### Application Metrics Implementation

```python
# monitoring.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import functools

# Define metrics
REQUEST_COUNT = Counter(
    'archon_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'archon_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint']
)

AGENT_QUEUE_SIZE = Gauge(
    'archon_agent_queue_size',
    'Current agent queue size',
    ['agent_type']
)

RAG_QUERY_DURATION = Histogram(
    'archon_rag_query_duration_seconds',
    'RAG query duration in seconds',
    ['strategy']
)

VECTOR_SEARCH_DURATION = Histogram(
    'archon_vector_search_duration_seconds',
    'Vector search duration in seconds',
    ['collection']
)

ACTIVE_AGENT_SESSIONS = Gauge(
    'archon_active_agent_sessions',
    'Number of active agent sessions',
    ['agent_type']
)

def monitor_requests(func):
    """Decorator to monitor HTTP requests."""
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        status = "200"
        
        try:
            result = await func(*args, **kwargs)
            return result
        except Exception as e:
            status = "500"
            raise
        finally:
            duration = time.time() - start_time
            
            # Extract method and endpoint from request context
            method = getattr(args[0], 'method', 'unknown')
            path = getattr(args[0], 'path', 'unknown')
            
            REQUEST_COUNT.labels(
                method=method,
                endpoint=path,
                status=status
            ).inc()
            
            REQUEST_DURATION.labels(
                method=method,
                endpoint=path
            ).observe(duration)
    
    return wrapper

def monitor_rag_query(strategy):
    """Decorator to monitor RAG query performance."""
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                RAG_QUERY_DURATION.labels(strategy=strategy).observe(duration)
        return wrapper
    return decorator

class MetricsCollector:
    """Centralized metrics collection and reporting."""
    
    def __init__(self):
        self.start_metrics_server()
        
    def start_metrics_server(self, port: int = 9090):
        """Start Prometheus metrics server."""
        start_http_server(port)
        
    def update_agent_queue_size(self, agent_type: str, size: int):
        """Update agent queue size metric."""
        AGENT_QUEUE_SIZE.labels(agent_type=agent_type).set(size)
        
    def update_active_sessions(self, agent_type: str, count: int):
        """Update active agent sessions metric."""
        ACTIVE_AGENT_SESSIONS.labels(agent_type=agent_type).set(count)
        
    @monitor_rag_query("vector_search")
    async def track_vector_search(self, collection: str, search_func):
        """Track vector search performance."""
        start_time = time.time()
        try:
            result = await search_func()
            return result
        finally:
            duration = time.time() - start_time
            VECTOR_SEARCH_DURATION.labels(collection=collection).observe(duration)
```

#### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "Archon Master Agent System",
    "tags": ["archon", "ai", "agents"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(archon_requests_total[5m])",
            "legendFormat": "{{ method }} {{ endpoint }}"
          }
        ],
        "yAxes": [
          {
            "label": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time (95th percentile)",
        "type": "graph", 
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(archon_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ],
        "yAxes": [
          {
            "label": "Seconds"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(archon_requests_total{status=~\"5..\"}[5m]) / rate(archon_requests_total[5m])",
            "legendFormat": "Error Rate"
          }
        ],
        "yAxes": [
          {
            "label": "Percentage",
            "max": 1
          }
        ]
      },
      {
        "title": "Agent Queue Sizes",
        "type": "graph",
        "targets": [
          {
            "expr": "archon_agent_queue_size",
            "legendFormat": "{{ agent_type }}"
          }
        ]
      },
      {
        "title": "RAG Query Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(archon_rag_query_duration_seconds_bucket[5m]))",
            "legendFormat": "{{ strategy }} (95th percentile)"
          }
        ]
      },
      {
        "title": "Vector Search Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(archon_vector_search_duration_seconds_bucket[5m]))",
            "legendFormat": "{{ collection }} (95th percentile)"
          }
        ]
      }
    ]
  }
}
```

### Log Management

#### Centralized Logging Configuration

```yaml
# filebeat.yml
filebeat.inputs:
- type: container
  paths:
    - '/var/lib/docker/containers/*/*.log'
  processors:
    - add_kubernetes_metadata:
        host: ${NODE_NAME}
        matchers:
        - logs_path:
            logs_path: "/var/lib/docker/containers/"

output.elasticsearch:
  hosts: ['${ELASTICSEARCH_HOST:elasticsearch}:${ELASTICSEARCH_PORT:9200}']
  index: "archon-logs-%{+yyyy.MM.dd}"

logging.level: info
logging.to_files: true
logging.files:
  path: /var/log/filebeat
  name: filebeat
  keepfiles: 7
  permissions: 0644
```

#### Structured Logging Implementation

```python
# logging_config.py
import logging
import json
import sys
from datetime import datetime
from typing import Dict, Any

class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
            
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)

def setup_logging(level: str = "INFO"):
    """Setup structured logging configuration."""
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add structured handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    logger.addHandler(handler)
    
    return logger

class AgentLogger:
    """Specialized logger for agent operations."""
    
    def __init__(self, agent_type: str):
        self.logger = logging.getLogger(f"archon.agent.{agent_type}")
        self.agent_type = agent_type
        
    def log_query_processing(
        self,
        query: str,
        processing_time: float,
        strategy: str,
        success: bool,
        extra_fields: Dict[str, Any] = None
    ):
        """Log query processing event."""
        
        fields = {
            "event_type": "query_processing",
            "agent_type": self.agent_type,
            "query_hash": hash(query),
            "processing_time": processing_time,
            "strategy": strategy,
            "success": success
        }
        
        if extra_fields:
            fields.update(extra_fields)
        
        level = logging.INFO if success else logging.ERROR
        self.logger.log(
            level,
            f"Query processed: success={success}, time={processing_time}s",
            extra={'extra_fields': fields}
        )
    
    def log_coordination_event(
        self,
        event_type: str,
        session_id: str,
        participants: list,
        metadata: Dict[str, Any] = None
    ):
        """Log agent coordination event."""
        
        fields = {
            "event_type": f"coordination_{event_type}",
            "agent_type": self.agent_type,
            "session_id": session_id,
            "participant_count": len(participants),
            "participants": participants
        }
        
        if metadata:
            fields.update(metadata)
        
        self.logger.info(
            f"Coordination event: {event_type} (session: {session_id})",
            extra={'extra_fields': fields}
        )
```

---

## Maintenance Procedures

### Routine Maintenance Tasks

#### Daily Maintenance

```bash
#!/bin/bash
# daily-maintenance.sh

echo "Starting daily maintenance tasks..."

# Check system health
./scripts/health-check.sh production

# Monitor resource usage
./scripts/resource-monitor.sh

# Check log files for errors
./scripts/error-log-analysis.sh

# Verify backup integrity
./scripts/verify-backups.sh

# Clean up temporary files
./scripts/cleanup-temp-files.sh

# Generate daily health report
./scripts/generate-health-report.sh

echo "Daily maintenance completed."
```

#### Weekly Maintenance

```bash
#!/bin/bash
# weekly-maintenance.sh

echo "Starting weekly maintenance tasks..."

# Update knowledge base indexes
./scripts/rebuild-vector-indexes.sh

# Analyze query performance trends
./scripts/performance-trend-analysis.sh

# Clean up old log files
./scripts/log-rotation.sh

# Update agent performance metrics
./scripts/update-agent-metrics.sh

# Security vulnerability scanning
./scripts/security-scan.sh

# Backup cleanup (keep last 4 weeks)
./scripts/backup-cleanup.sh --keep-weeks=4

# Generate weekly performance report
./scripts/generate-weekly-report.sh

echo "Weekly maintenance completed."
```

#### Monthly Maintenance

```bash
#!/bin/bash
# monthly-maintenance.sh

echo "Starting monthly maintenance tasks..."

# Full system performance analysis
./scripts/full-performance-analysis.sh

# Capacity planning assessment
./scripts/capacity-planning.sh

# Security audit
./scripts/security-audit.sh

# Disaster recovery testing
./scripts/dr-test.sh

# Update documentation
./scripts/update-documentation.sh

# Generate monthly executive report
./scripts/generate-executive-report.sh

echo "Monthly maintenance completed."
```

### Database Maintenance

#### Index Optimization

```sql
-- index-maintenance.sql
-- Run weekly to optimize database performance

-- Analyze table statistics
ANALYZE documents;
ANALYZE projects;
ANALYZE tasks;
ANALYZE mcp_client_sessions;

-- Reindex vector indexes if fragmented
REINDEX INDEX CONCURRENTLY idx_documents_embedding_ivfflat;

-- Update index statistics
SELECT pg_stat_reset();

-- Vacuum analyze for better performance
VACUUM ANALYZE documents;
VACUUM ANALYZE projects; 
VACUUM ANALYZE tasks;

-- Check index usage
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats 
WHERE tablename IN ('documents', 'projects', 'tasks')
ORDER BY tablename, attname;

-- Identify unused indexes
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY schemaname, tablename;
```

#### Vector Index Optimization

```python
# vector_maintenance.py
import asyncio
import logging
from typing import Dict, Any
import asyncpg

logger = logging.getLogger(__name__)

class VectorIndexMaintenance:
    """Automated vector index maintenance and optimization."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        
    async def optimize_vector_indexes(self):
        """Optimize vector indexes for better performance."""
        
        conn = await asyncpg.connect(self.db_url)
        
        try:
            # Get current index statistics
            stats = await self.get_index_statistics(conn)
            
            # Analyze index performance
            performance_analysis = await self.analyze_index_performance(conn, stats)
            
            # Rebuild indexes if needed
            if performance_analysis['needs_rebuild']:
                await self.rebuild_indexes(conn)
                
            # Update index parameters
            await self.optimize_index_parameters(conn, performance_analysis)
            
            logger.info("Vector index optimization completed")
            
        finally:
            await conn.close()
    
    async def get_index_statistics(self, conn) -> Dict[str, Any]:
        """Get current vector index statistics."""
        
        query = """
        SELECT 
            indexname,
            indexdef,
            pg_size_pretty(pg_relation_size(indexname::regclass)) as index_size,
            pg_stat_get_numscans(indexrelid) as num_scans,
            pg_stat_get_tuples_returned(indexrelid) as tuples_returned
        FROM pg_indexes 
        JOIN pg_stat_user_indexes USING (indexname)
        WHERE indexname LIKE '%vector%' OR indexname LIKE '%embedding%';
        """
        
        result = await conn.fetch(query)
        return {row['indexname']: dict(row) for row in result}
    
    async def analyze_index_performance(
        self, 
        conn, 
        stats: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze index performance and determine optimization needs."""
        
        analysis = {
            'needs_rebuild': False,
            'recommended_parameters': {},
            'performance_issues': []
        }
        
        for index_name, index_stats in stats.items():
            # Check scan frequency
            if index_stats['num_scans'] < 10:
                analysis['performance_issues'].append(
                    f"Low usage for index {index_name}: {index_stats['num_scans']} scans"
                )
            
            # Check index size efficiency
            size_mb = await self._parse_size_to_mb(index_stats['index_size'])
            if size_mb > 1000:  # > 1GB
                analysis['needs_rebuild'] = True
                analysis['performance_issues'].append(
                    f"Large index size for {index_name}: {index_stats['index_size']}"
                )
        
        return analysis
    
    async def rebuild_indexes(self, conn):
        """Rebuild vector indexes concurrently."""
        
        indexes_to_rebuild = [
            "idx_documents_embedding_ivfflat",
            "idx_documents_metadata_gin"
        ]
        
        for index_name in indexes_to_rebuild:
            logger.info(f"Rebuilding index: {index_name}")
            
            try:
                await conn.execute(f"REINDEX INDEX CONCURRENTLY {index_name}")
                logger.info(f"Successfully rebuilt index: {index_name}")
                
            except Exception as e:
                logger.error(f"Failed to rebuild index {index_name}: {e}")
    
    async def optimize_index_parameters(self, conn, analysis: Dict[str, Any]):
        """Optimize index parameters based on analysis."""
        
        # Optimize IVF parameters based on data size
        data_size_query = "SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL"
        data_count = await conn.fetchval(data_size_query)
        
        # Calculate optimal IVF lists
        optimal_lists = min(max(data_count // 1000, 100), 65536)
        
        # Update IVF search parameters
        await conn.execute(f"ALTER DATABASE archon SET ivfflat.probes = {optimal_lists // 100}")
        
        logger.info(f"Optimized IVF parameters: lists={optimal_lists}")
```

---

## Performance Optimization

### System Performance Tuning

#### Database Performance Optimization

```python
# db_performance_tuning.py
class DatabasePerformanceOptimizer:
    """Advanced database performance optimization."""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        
    async def optimize_query_performance(self):
        """Optimize database query performance."""
        
        conn = await asyncpg.connect(self.db_url)
        
        try:
            # Enable query optimization features
            await conn.execute("SET enable_seqscan = off")
            await conn.execute("SET random_page_cost = 1.1")
            await conn.execute("SET effective_cache_size = '8GB'")
            
            # Optimize work memory for large queries
            await conn.execute("SET work_mem = '256MB'")
            await conn.execute("SET maintenance_work_mem = '2GB'")
            
            # Enable parallel query execution
            await conn.execute("SET max_parallel_workers_per_gather = 4")
            
            # Optimize vector search parameters
            await self.optimize_vector_search_parameters(conn)
            
        finally:
            await conn.close()
    
    async def optimize_vector_search_parameters(self, conn):
        """Optimize vector search specific parameters."""
        
        # Get collection statistics
        stats_query = """
        SELECT 
            COUNT(*) as total_vectors,
            AVG(array_length(embedding, 1)) as avg_dimensions,
            pg_size_pretty(pg_total_relation_size('documents')) as table_size
        FROM documents 
        WHERE embedding IS NOT NULL;
        """
        
        stats = await conn.fetchrow(stats_query)
        
        # Calculate optimal parameters based on data
        total_vectors = stats['total_vectors']
        
        if total_vectors < 10000:
            # Small dataset: exact search
            lists = max(total_vectors // 100, 10)
            probes = lists
        elif total_vectors < 100000:
            # Medium dataset: balanced approach
            lists = max(total_vectors // 500, 100)
            probes = max(lists // 10, 10)
        else:
            # Large dataset: performance-optimized
            lists = max(total_vectors // 1000, 1000)
            probes = max(lists // 50, 20)
        
        # Apply optimized parameters
        await conn.execute(f"ALTER DATABASE archon SET ivfflat.probes = {probes}")
        
        logger.info(f"Optimized vector search: lists={lists}, probes={probes}")
```

#### Application Performance Optimization

```python
# app_performance_optimization.py
import asyncio
import aioredis
from typing import Dict, Any, Optional
import time

class ApplicationPerformanceOptimizer:
    """Application-level performance optimization."""
    
    def __init__(self):
        self.redis_client = None
        self.performance_cache = {}
        self.optimization_rules = self.load_optimization_rules()
        
    async def initialize(self):
        """Initialize performance optimization components."""
        
        self.redis_client = await aioredis.from_url("redis://localhost:6379")
        
    def load_optimization_rules(self) -> Dict[str, Any]:
        """Load performance optimization rules."""
        
        return {
            "query_caching": {
                "enabled": True,
                "ttl": 300,  # 5 minutes
                "max_size": 10000
            },
            "response_compression": {
                "enabled": True,
                "min_size": 1024,  # bytes
                "algorithms": ["gzip", "deflate"]
            },
            "connection_pooling": {
                "min_size": 10,
                "max_size": 100,
                "max_queries": 50000,
                "max_inactive_time": 300
            },
            "batch_processing": {
                "enabled": True,
                "batch_size": 50,
                "max_wait_time": 100  # ms
            }
        }
    
    async def optimize_query_processing(
        self,
        query_func,
        query: str,
        context: Dict[str, Any]
    ) -> Any:
        """Optimize query processing with caching and batching."""
        
        # Check cache first
        cache_key = self.generate_cache_key(query, context)
        cached_result = await self.get_cached_result(cache_key)
        
        if cached_result:
            return cached_result
            
        # Execute query with performance monitoring
        start_time = time.time()
        result = await query_func(query, context)
        processing_time = time.time() - start_time
        
        # Cache result if beneficial
        if processing_time > 0.1:  # Cache slow queries
            await self.cache_result(cache_key, result)
        
        # Update performance metrics
        await self.update_performance_metrics(query, processing_time)
        
        return result
    
    async def optimize_batch_operations(
        self,
        operation_func,
        items: list,
        batch_size: Optional[int] = None
    ) -> list:
        """Optimize batch operations for better throughput."""
        
        if batch_size is None:
            batch_size = self.optimization_rules["batch_processing"]["batch_size"]
        
        results = []
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Process batches concurrently with controlled parallelism
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent batches
        
        async def process_batch(batch):
            async with semaphore:
                return await operation_func(batch)
        
        batch_tasks = [process_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks)
        
        # Flatten results
        for batch_result in batch_results:
            results.extend(batch_result)
            
        return results
    
    async def optimize_memory_usage(self):
        """Optimize application memory usage."""
        
        import gc
        import psutil
        
        # Get current memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        if memory_percent > 80:
            # Aggressive optimization for high memory usage
            logger.warning(f"High memory usage detected: {memory_percent}%")
            
            # Clear caches
            self.performance_cache.clear()
            await self.redis_client.flushdb()
            
            # Force garbage collection
            gc.collect()
            
            logger.info("Aggressive memory optimization applied")
            
        elif memory_percent > 60:
            # Moderate optimization
            logger.info(f"Moderate memory usage: {memory_percent}%")
            
            # Clear old cache entries
            await self.cleanup_old_cache_entries()
            
            # Trigger garbage collection
            gc.collect()
```

### Agent Performance Optimization

```python
# agent_performance_optimization.py
class AgentPerformanceOptimizer:
    """Specialized performance optimization for AI agents."""
    
    def __init__(self):
        self.agent_performance_history = {}
        self.optimization_strategies = self.load_optimization_strategies()
        
    def load_optimization_strategies(self) -> Dict[str, Any]:
        """Load agent-specific optimization strategies."""
        
        return {
            "model_caching": {
                "enabled": True,
                "cache_size": "10GB",
                "preload_models": ["gpt-4o", "text-embedding-3-small"]
            },
            "request_batching": {
                "enabled": True,
                "max_batch_size": 10,
                "max_wait_time": 50  # ms
            },
            "response_streaming": {
                "enabled": True,
                "chunk_size": 1024,
                "buffer_size": 4096
            },
            "load_balancing": {
                "strategy": "least_connections",
                "health_check_interval": 30,
                "failover_timeout": 5
            }
        }
    
    async def optimize_agent_selection(
        self,
        query: str,
        available_agents: list,
        performance_history: Dict[str, list]
    ) -> str:
        """Optimize agent selection based on performance history."""
        
        # Calculate performance scores for available agents
        agent_scores = {}
        
        for agent in available_agents:
            history = performance_history.get(agent, [])
            
            if not history:
                # New agent, give neutral score
                agent_scores[agent] = 0.5
                continue
            
            # Calculate recent performance metrics
            recent_history = history[-50:]  # Last 50 operations
            
            avg_response_time = sum(h['response_time'] for h in recent_history) / len(recent_history)
            success_rate = sum(1 for h in recent_history if h['success']) / len(recent_history)
            
            # Current load factor
            current_load = await self.get_agent_current_load(agent)
            load_factor = max(0, 1 - (current_load / 100))  # Convert to 0-1 scale
            
            # Combined score (lower response time is better)
            time_score = max(0, 1 - (avg_response_time / 10))  # Normalize to 0-1
            combined_score = (success_rate * 0.4) + (time_score * 0.4) + (load_factor * 0.2)
            
            agent_scores[agent] = combined_score
        
        # Select agent with highest score
        optimal_agent = max(agent_scores, key=agent_scores.get)
        
        logger.info(f"Selected agent {optimal_agent} with score {agent_scores[optimal_agent]:.3f}")
        return optimal_agent
    
    async def optimize_model_inference(
        self,
        model_name: str,
        input_data: Any,
        optimization_params: Dict[str, Any] = None
    ) -> Any:
        """Optimize model inference with caching and batching."""
        
        if optimization_params is None:
            optimization_params = self.optimization_strategies
        
        # Check model cache
        if optimization_params["model_caching"]["enabled"]:
            cache_key = self.generate_inference_cache_key(model_name, input_data)
            cached_result = await self.get_cached_inference(cache_key)
            
            if cached_result:
                return cached_result
        
        # Execute inference with optimization
        result = await self.execute_optimized_inference(
            model_name,
            input_data,
            optimization_params
        )
        
        # Cache result if caching is enabled
        if optimization_params["model_caching"]["enabled"]:
            await self.cache_inference_result(cache_key, result)
        
        return result
    
    async def execute_optimized_inference(
        self,
        model_name: str,
        input_data: Any,
        optimization_params: Dict[str, Any]
    ) -> Any:
        """Execute model inference with performance optimizations."""
        
        # Apply request batching if enabled
        if optimization_params["request_batching"]["enabled"]:
            return await self.execute_batched_inference(
                model_name,
                input_data,
                optimization_params["request_batching"]
            )
        else:
            return await self.execute_single_inference(model_name, input_data)
    
    async def monitor_agent_performance(self, agent_type: str):
        """Monitor and log agent performance metrics."""
        
        metrics = {
            "timestamp": time.time(),
            "agent_type": agent_type,
            "active_sessions": await self.get_active_sessions(agent_type),
            "queue_size": await self.get_queue_size(agent_type),
            "average_response_time": await self.get_average_response_time(agent_type),
            "success_rate": await self.get_success_rate(agent_type),
            "resource_usage": await self.get_resource_usage(agent_type)
        }
        
        # Store metrics for performance analysis
        if agent_type not in self.agent_performance_history:
            self.agent_performance_history[agent_type] = []
        
        self.agent_performance_history[agent_type].append(metrics)
        
        # Keep only recent metrics (last 1000 entries)
        if len(self.agent_performance_history[agent_type]) > 1000:
            self.agent_performance_history[agent_type] = \
                self.agent_performance_history[agent_type][-1000:]
        
        return metrics
```

This operational guide provides comprehensive procedures for deploying, monitoring, and maintaining the Master Agent System in production environments. The next section will cover examples and tutorials for common use cases.