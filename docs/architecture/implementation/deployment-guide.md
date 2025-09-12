# Master Agent Implementation and Deployment Guide

## Overview

This guide provides comprehensive instructions for implementing and deploying the Master Agent architecture that integrates Claude Flow's specialized agents with Archon's RAG knowledge system.

## SPARC Methodology Implementation

Following SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) principles:

### 1. Specification Phase ✅

**Completed Components:**
- Master Agent Architecture Design
- Query Analysis Engine Specification  
- Agent Capability Matrix Specification
- RAG Integration Layer Specification
- Coordination Protocol Handler Specification
- Performance Monitoring System Specification

### 2. Pseudocode Phase

**Core Master Agent Workflow:**
```pseudocode
FUNCTION master_agent_process_request(user_query, context):
    // Phase 1: Query Analysis
    query_analysis = query_analysis_engine.analyze(user_query, context)
    
    // Phase 2: Knowledge Retrieval
    knowledge_context = rag_integration_layer.retrieve_knowledge(query_analysis, context)
    
    // Phase 3: Agent Selection
    optimal_agents = agent_capability_matcher.select_agents(query_analysis, knowledge_context)
    
    // Phase 4: Coordination Setup
    coordination_plan = coordination_protocol_handler.create_plan(optimal_agents, query_analysis)
    
    // Phase 5: Task Execution
    execution_results = coordination_protocol_handler.execute(coordination_plan)
    
    // Phase 6: Performance Monitoring
    performance_monitoring_system.record_execution(execution_results)
    
    RETURN execution_results
END FUNCTION
```

## 3. Architecture Implementation

### Project Structure

```
master-agent/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── master_agent.py
│   │   └── config.py
│   ├── query_analysis/
│   │   ├── __init__.py
│   │   ├── nlp_processor.py
│   │   ├── intent_classifier.py
│   │   ├── entity_extractor.py
│   │   ├── complexity_analyzer.py
│   │   └── domain_detector.py
│   ├── agent_management/
│   │   ├── __init__.py
│   │   ├── capability_matrix.py
│   │   ├── agent_registry.py
│   │   └── routing_engine.py
│   ├── rag_integration/
│   │   ├── __init__.py
│   │   ├── knowledge_router.py
│   │   ├── contextual_embeddings.py
│   │   ├── hybrid_search.py
│   │   └── enrichment_engine.py
│   ├── coordination/
│   │   ├── __init__.py
│   │   ├── protocol_manager.py
│   │   ├── hierarchical_protocol.py
│   │   ├── mesh_protocol.py
│   │   ├── adaptive_protocol.py
│   │   └── fault_tolerance.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics_collector.py
│   │   ├── performance_tracker.py
│   │   ├── alert_manager.py
│   │   └── dashboard.py
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       └── helpers.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── performance/
├── config/
│   ├── development.yaml
│   ├── production.yaml
│   └── docker.yaml
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.prod.yml
├── deployment/
│   ├── kubernetes/
│   ├── terraform/
│   └── scripts/
├── docs/
│   ├── api/
│   ├── deployment/
│   └── troubleshooting/
├── requirements.txt
├── setup.py
└── README.md
```

### Core Implementation

**1. Master Agent Core (`src/core/master_agent.py`):**

```python
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..query_analysis import QueryAnalysisEngine
from ..agent_management import AgentCapabilityMatcher
from ..rag_integration import RAGIntegrationLayer
from ..coordination import CoordinationProtocolHandler
from ..monitoring import PerformanceMonitoringSystem

class MasterAgent:
    """
    Main orchestrator that integrates Claude Flow agents with Archon RAG system
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.query_analyzer = QueryAnalysisEngine(config.query_analysis)
        self.agent_matcher = AgentCapabilityMatcher(config.agent_management)
        self.rag_integrator = RAGIntegrationLayer(config.rag_integration)
        self.coordinator = CoordinationProtocolHandler(config.coordination)
        self.monitor = PerformanceMonitoringSystem(config.monitoring)
        self.active_sessions = {}
        self.system_status = "initializing"
    
    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize components in order
            await self.query_analyzer.initialize()
            await self.agent_matcher.initialize()
            await self.rag_integrator.initialize()
            await self.coordinator.initialize()
            await self.monitor.initialize()
            
            self.system_status = "ready"
            await self.monitor.log_system_event("master_agent_initialized")
            
        except Exception as e:
            self.system_status = "failed"
            await self.monitor.log_error("master_agent_initialization_failed", str(e))
            raise
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Main request processing workflow"""
        
        session_id = request.get("session_id", f"session_{datetime.utcnow().timestamp()}")
        request_id = f"req_{datetime.utcnow().timestamp()}"
        
        try:
            # Start request monitoring
            await self.monitor.start_request_tracking(request_id, request)
            
            # Step 1: Query Analysis
            query_analysis = await self.query_analyzer.analyze(
                query=request["query"],
                context=request.get("context", {}),
                user_id=request.get("user_id"),
                session_id=session_id
            )
            
            # Step 2: Knowledge Retrieval
            knowledge_context = await self.rag_integrator.retrieve_knowledge(
                query_analysis=query_analysis,
                context=request.get("context", {})
            )
            
            # Step 3: Agent Selection
            optimal_agents = await self.agent_matcher.select_agents(
                query_analysis=query_analysis,
                knowledge_context=knowledge_context,
                constraints=request.get("agent_constraints", {})
            )
            
            # Step 4: Coordination Setup
            coordination_plan = await self.coordinator.create_coordination_plan(
                agents=optimal_agents,
                query_analysis=query_analysis,
                knowledge_context=knowledge_context
            )
            
            # Step 5: Task Execution
            execution_results = await self.coordinator.execute_coordinated_task(
                coordination_plan=coordination_plan,
                monitor_callback=self.monitor.track_execution_progress
            )
            
            # Step 6: Response Preparation
            response = await self.prepare_response(
                execution_results=execution_results,
                query_analysis=query_analysis,
                knowledge_context=knowledge_context
            )
            
            # Complete monitoring
            await self.monitor.complete_request_tracking(request_id, response)
            
            return {
                "success": True,
                "request_id": request_id,
                "session_id": session_id,
                "response": response,
                "metadata": {
                    "agents_used": [agent.agent_id for agent in optimal_agents],
                    "coordination_protocol": coordination_plan.protocol_type,
                    "knowledge_sources": len(knowledge_context.get("sources", [])),
                    "execution_time_ms": execution_results.get("execution_time_ms"),
                    "confidence_score": response.get("confidence_score")
                }
            }
            
        except Exception as e:
            await self.monitor.log_error("request_processing_failed", str(e))
            return {
                "success": False,
                "request_id": request_id,
                "session_id": session_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def prepare_response(self, execution_results, query_analysis, knowledge_context):
        """Prepare final response with enrichment"""
        
        # Base response from execution results
        response = {
            "content": execution_results.get("response", ""),
            "confidence_score": execution_results.get("confidence", 0.8)
        }
        
        # Add knowledge context references
        if knowledge_context.get("sources"):
            response["knowledge_sources"] = [
                {
                    "source": source.get("source_type", "unknown"),
                    "relevance": source.get("relevance_score", 0),
                    "url": source.get("url", "")
                }
                for source in knowledge_context["sources"][:3]  # Top 3 sources
            ]
        
        # Add agent insights
        if execution_results.get("agent_insights"):
            response["insights"] = execution_results["agent_insights"]
        
        # Add suggestions for follow-up
        if query_analysis.complexity.level in ["medium", "complex"]:
            response["follow_up_suggestions"] = await self.generate_follow_up_suggestions(
                query_analysis, execution_results
            )
        
        return response
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        
        health_status = {
            "status": self.system_status,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Check each component
        components = [
            ("query_analyzer", self.query_analyzer),
            ("agent_matcher", self.agent_matcher), 
            ("rag_integrator", self.rag_integrator),
            ("coordinator", self.coordinator),
            ("monitor", self.monitor)
        ]
        
        for name, component in components:
            try:
                if hasattr(component, 'health_check'):
                    component_health = await component.health_check()
                    health_status["components"][name] = component_health
                else:
                    health_status["components"][name] = {"status": "unknown"}
            except Exception as e:
                health_status["components"][name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Overall system status
        component_statuses = [
            comp.get("status", "error") 
            for comp in health_status["components"].values()
        ]
        
        if all(status == "healthy" for status in component_statuses):
            health_status["status"] = "healthy"
        elif any(status == "error" for status in component_statuses):
            health_status["status"] = "degraded"
        else:
            health_status["status"] = "warning"
        
        return health_status
```

## 4. Deployment Configuration

### Environment Configuration (`config/production.yaml`)

```yaml
# Master Agent Production Configuration

master_agent:
  name: "master-agent-production"
  version: "1.0.0"
  environment: "production"
  hostname: "${HOSTNAME}"
  
  # Logging configuration
  logging:
    level: "INFO"
    format: "json"
    handlers:
      - type: "file"
        file: "/var/log/master-agent/master-agent.log"
        max_size: "100MB"
        backup_count: 10
      - type: "console"
        stream: "stdout"

# Query Analysis Engine Configuration
query_analysis:
  nlp_processor:
    model_cache_dir: "/opt/master-agent/models"
    max_query_length: 2000
    timeout_seconds: 30
  
  intent_classifier:
    confidence_threshold: 0.7
    fallback_intent: "general_assistance"
  
  entity_extractor:
    max_entities_per_query: 50
    confidence_threshold: 0.6
  
  complexity_analyzer:
    simple_threshold: 0.3
    complex_threshold: 0.7

# Agent Management Configuration  
agent_management:
  capability_matrix:
    refresh_interval_seconds: 300  # 5 minutes
    performance_window_minutes: 60
  
  agent_registry:
    claude_flow_url: "${CLAUDE_FLOW_URL}"
    archon_mcp_url: "${ARCHON_MCP_URL}"
    connection_timeout: 30
    max_retries: 3
  
  routing_engine:
    max_agents_per_task: 5
    load_balancing: "weighted_round_robin"
    failover_enabled: true

# RAG Integration Configuration
rag_integration:
  knowledge_router:
    archon_client:
      base_url: "${ARCHON_BASE_URL}"
      api_key: "${ARCHON_API_KEY}"
      timeout: 45
    
    query_strategies:
      contextual_embeddings:
        enabled: true
        model: "text-embedding-3-small"
      hybrid_search:
        enabled: true
        semantic_weight: 0.6
        keyword_weight: 0.4
      code_example_search:
        enabled: true
        quality_threshold: 0.4
  
  cache:
    backend: "redis"
    redis_url: "${REDIS_URL}"
    default_ttl: 3600
    max_size: 10000

# Coordination Configuration
coordination:
  protocol_manager:
    supported_protocols: ["hierarchical", "mesh", "adaptive", "ring", "star"]
    default_protocol: "adaptive"
    switch_threshold: 0.3
  
  fault_tolerance:
    detection_interval: 10
    failure_timeout: 60
    max_healing_attempts: 3
  
  self_healing:
    enabled: true
    strategies: ["restart_agent", "reassign_tasks", "switch_protocol"]

# Monitoring Configuration
monitoring:
  metrics_collection:
    enabled: true
    collection_interval: 15
    storage_backend: "prometheus"
    prometheus_url: "${PROMETHEUS_URL}"
  
  performance_tracking:
    enabled: true
    track_query_performance: true
    track_agent_performance: true
    track_coordination_performance: true
  
  alerting:
    enabled: true
    alert_channels:
      slack:
        enabled: true
        webhook_url: "${SLACK_WEBHOOK_URL}"
      email:
        enabled: true
        smtp_host: "${SMTP_HOST}"
        smtp_port: 587
        username: "${SMTP_USERNAME}"
        password: "${SMTP_PASSWORD}"
    
    thresholds:
      cpu_utilization_percent: 85
      memory_utilization_percent: 90
      agent_success_rate: 0.8
      rag_avg_query_time: 5.0

# Integration URLs and Credentials
integrations:
  claude_flow:
    base_url: "${CLAUDE_FLOW_URL}"
    api_version: "v1"
    timeout: 60
  
  archon:
    base_url: "${ARCHON_BASE_URL}" 
    mcp_port: "${ARCHON_MCP_PORT:-8051}"
    api_key: "${ARCHON_API_KEY}"
  
  supabase:
    url: "${SUPABASE_URL}"
    service_key: "${SUPABASE_SERVICE_KEY}"
  
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4"
    embedding_model: "text-embedding-3-small"

# Security Configuration
security:
  api_keys:
    master_agent_key: "${MASTER_AGENT_API_KEY}"
    jwt_secret: "${JWT_SECRET}"
  
  rate_limiting:
    enabled: true
    requests_per_minute: 60
    burst_limit: 10
  
  cors:
    enabled: true
    allowed_origins: ["http://localhost:3000", "https://archon.example.com"]
```

### Docker Configuration (`docker/docker-compose.prod.yml`)

```yaml
version: '3.8'

services:
  master-agent:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    image: master-agent:latest
    container_name: master-agent-prod
    restart: unless-stopped
    ports:
      - "${MASTER_AGENT_PORT:-8090}:8090"
    environment:
      - ENVIRONMENT=production
      - CLAUDE_FLOW_URL=${CLAUDE_FLOW_URL}
      - ARCHON_BASE_URL=${ARCHON_BASE_URL}
      - ARCHON_API_KEY=${ARCHON_API_KEY}
      - SUPABASE_URL=${SUPABASE_URL}
      - SUPABASE_SERVICE_KEY=${SUPABASE_SERVICE_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - REDIS_URL=redis://redis:6379
      - PROMETHEUS_URL=http://prometheus:9090
    volumes:
      - master-agent-data:/opt/master-agent/data
      - master-agent-logs:/var/log/master-agent
      - master-agent-models:/opt/master-agent/models
    depends_on:
      - redis
      - prometheus
    networks:
      - master-agent-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8090/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  redis:
    image: redis:7-alpine
    container_name: master-agent-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    networks:
      - master-agent-network
    command: redis-server --appendonly yes

  prometheus:
    image: prom/prometheus:latest
    container_name: master-agent-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    networks:
      - master-agent-network
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'

  grafana:
    image: grafana/grafana:latest
    container_name: master-agent-grafana
    restart: unless-stopped
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD:-admin}
      - GF_USERS_ALLOW_SIGN_UP=false
    volumes:
      - grafana-data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./grafana/datasources:/etc/grafana/provisioning/datasources
    networks:
      - master-agent-network
    depends_on:
      - prometheus

volumes:
  master-agent-data:
  master-agent-logs:
  master-agent-models:
  redis-data:
  prometheus-data:
  grafana-data:

networks:
  master-agent-network:
    driver: bridge
```

### Kubernetes Deployment (`deployment/kubernetes/master-agent.yaml`)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: master-agent
  namespace: master-agent
  labels:
    app: master-agent
    version: v1.0.0
spec:
  replicas: 3
  selector:
    matchLabels:
      app: master-agent
  template:
    metadata:
      labels:
        app: master-agent
        version: v1.0.0
    spec:
      containers:
      - name: master-agent
        image: master-agent:v1.0.0
        ports:
        - containerPort: 8090
          name: http
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: CLAUDE_FLOW_URL
          valueFrom:
            configMapKeyRef:
              name: master-agent-config
              key: claude-flow-url
        - name: ARCHON_BASE_URL
          valueFrom:
            configMapKeyRef:
              name: master-agent-config  
              key: archon-base-url
        - name: ARCHON_API_KEY
          valueFrom:
            secretKeyRef:
              name: master-agent-secrets
              key: archon-api-key
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: master-agent-secrets
              key: openai-api-key
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8090
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8090
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /opt/master-agent/config
        - name: models
          mountPath: /opt/master-agent/models
      volumes:
      - name: config
        configMap:
          name: master-agent-config
      - name: models
        persistentVolumeClaim:
          claimName: master-agent-models-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: master-agent-service
  namespace: master-agent
spec:
  selector:
    app: master-agent
  ports:
  - name: http
    port: 80
    targetPort: 8090
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: master-agent-ingress
  namespace: master-agent
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "60"
spec:
  tls:
  - hosts:
    - master-agent.example.com
    secretName: master-agent-tls
  rules:
  - host: master-agent.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: master-agent-service
            port:
              number: 80
```

## 5. Deployment and Operations

### Deployment Steps

**1. Environment Setup:**
```bash
# Create environment file
cp .env.example .env
# Edit .env with your configuration

# Create Docker network
docker network create master-agent-network

# Create volumes
docker volume create master-agent-data
docker volume create master-agent-logs
```

**2. Local Development Deployment:**
```bash
# Start development environment
docker-compose up -d

# Check service health
curl http://localhost:8090/health

# View logs
docker-compose logs -f master-agent
```

**3. Production Deployment:**
```bash
# Build production image
docker build -f docker/Dockerfile -t master-agent:v1.0.0 .

# Deploy with production configuration
docker-compose -f docker/docker-compose.prod.yml up -d

# Verify deployment
curl http://localhost:8090/health
curl http://localhost:8090/metrics
```

**4. Kubernetes Deployment:**
```bash
# Create namespace
kubectl create namespace master-agent

# Apply configuration
kubectl apply -f deployment/kubernetes/

# Check deployment status
kubectl get pods -n master-agent
kubectl logs -f deployment/master-agent -n master-agent
```

### Monitoring and Maintenance

**Health Check Endpoints:**
- `GET /health` - Overall system health
- `GET /ready` - Readiness probe
- `GET /metrics` - Prometheus metrics
- `GET /status` - Detailed component status

**Operational Commands:**
```bash
# View system status
curl http://localhost:8090/status

# Get performance metrics
curl http://localhost:8090/metrics | grep master_agent

# Test query processing
curl -X POST http://localhost:8090/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I implement authentication?", "user_id": "test_user"}'
```

**Performance Tuning:**
- Monitor CPU and memory usage
- Adjust agent pool sizes based on workload
- Tune cache settings for optimal hit rates
- Configure coordination protocols based on task types
- Set appropriate timeout values

This comprehensive implementation guide provides the foundation for deploying the Master Agent architecture following SPARC methodology principles, ensuring robust, scalable, and maintainable integration between Claude Flow and Archon systems.

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "Analyze system requirements and create architecture overview", "status": "completed", "activeForm": "Analyzed system requirements and created architecture overview"}, {"content": "Design Query Analysis Engine with NLP capabilities", "status": "completed", "activeForm": "Designed Query Analysis Engine with NLP capabilities"}, {"content": "Create Agent Capability Matrix and routing algorithms", "status": "completed", "activeForm": "Created Agent Capability Matrix and routing algorithms"}, {"content": "Design RAG Integration Layer with contextual embeddings", "status": "completed", "activeForm": "Designed RAG Integration Layer with contextual embeddings"}, {"content": "Implement Coordination Protocol Handler with fault tolerance", "status": "completed", "activeForm": "Implemented Coordination Protocol Handler with fault tolerance"}, {"content": "Build Performance Monitoring System with real-time metrics", "status": "completed", "activeForm": "Built Performance Monitoring System with real-time metrics"}, {"content": "Create C4 architecture diagrams and component specifications", "status": "completed", "activeForm": "Created C4 architecture diagrams and component specifications"}, {"content": "Design integration points with Archon MCP and Claude Flow", "status": "completed", "activeForm": "Designed integration points with Archon MCP and Claude Flow"}, {"content": "Implement SPARC methodology compliance framework", "status": "completed", "activeForm": "Implemented SPARC methodology compliance framework"}, {"content": "Create implementation guidelines and deployment strategy", "status": "completed", "activeForm": "Created implementation guidelines and deployment strategy"}]