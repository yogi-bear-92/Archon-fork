# Master Agent System - User Guide

## Table of Contents

- [Quick Start](#quick-start)
- [Setup Instructions](#setup-instructions)
- [Usage Examples](#usage-examples)
- [Configuration Options](#configuration-options)
- [Troubleshooting](#troubleshooting)
- [Common Workflows](#common-workflows)
- [Best Practices](#best-practices)

---

## Quick Start

The Archon Master Agent System is an intelligent AI orchestration platform that combines expert-level agents with progressive refinement capabilities. Get started in minutes with our streamlined setup process.

### Prerequisites

Before you begin, ensure you have:

- **Docker Desktop** installed and running
- **Node.js 18+** for development mode
- **Supabase account** (free tier works)
- **OpenAI API key** (optional - configurable via UI)
- **10GB+ disk space** for containers and data

### 30-Second Quickstart

```bash
# 1. Clone and navigate
git clone https://github.com/coleam00/archon.git
cd archon

# 2. Configure environment
cp .env.example .env
# Edit .env with your Supabase credentials

# 3. Setup database
# Execute migration/complete_setup.sql in Supabase SQL Editor

# 4. Start all services
docker compose up --build -d

# 5. Access the system
open http://localhost:3737
```

**✅ Success Indicators:**
- Web UI loads at `http://localhost:3737`
- All services show "healthy" status
- You can create a test project and add tasks

---

## Setup Instructions

### Detailed Environment Configuration

1. **Supabase Setup**:
   ```bash
   # Required: Core database connection
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_SERVICE_KEY=your-service-key-here
   
   # Note: Use the legacy (longer) service key, not the new type
   ```

2. **AI Provider Configuration**:
   ```bash
   # Optional: Can be set via UI
   OPENAI_API_KEY=sk-your-openai-key
   GEMINI_API_KEY=your-gemini-key
   OLLAMA_BASE_URL=http://localhost:11434
   ```

3. **Service Ports** (Optional):
   ```bash
   ARCHON_UI_PORT=3737           # Web interface
   ARCHON_SERVER_PORT=8181       # Core API
   ARCHON_MCP_PORT=8051         # MCP protocol
   ARCHON_AGENTS_PORT=8052      # AI agents
   ```

4. **Advanced Settings**:
   ```bash
   # Performance tuning
   LOG_LEVEL=INFO               # DEBUG for troubleshooting
   AGENTS_ENABLED=true          # Enable AI agent processing
   CLAUDE_FLOW_ENABLED=true     # Enable swarm coordination
   
   # Custom hostname (for remote access)
   HOST=localhost               # or your-server.com
   ```

### Database Migration

Execute these SQL scripts in your Supabase SQL Editor:

```sql
-- 1. Core setup (required)
-- Copy contents from migration/complete_setup.sql

-- 2. If upgrading, check for new migrations:
-- migration/2_archon_projects.sql
-- migration/3_mcp_client_management.sql
```

### Service Verification

After startup, verify all services are running:

```bash
# Check service health
curl http://localhost:8181/health    # Server API
curl http://localhost:8051/health    # MCP Server
curl http://localhost:8052/health    # Agents Service

# Check web interface
curl http://localhost:3737           # Frontend
```

---

## Usage Examples

### Common Use Cases

#### 1. Document Knowledge Management

```bash
# Via Web UI (Recommended)
1. Navigate to http://localhost:3737
2. Go to "Knowledge Base" → "Crawl Website"
3. Enter documentation URL: https://docs.python.org/3/
4. Wait for crawling completion
5. Test search: "How to use asyncio"

# Via MCP Client (Advanced)
# Connect your AI coding assistant to http://localhost:8051
# Use tools like: perform_rag_query, search_code_examples
```

#### 2. Project Task Management

```bash
# Creating a new project
1. Projects → "Create New Project"
2. Name: "API Modernization"
3. Description: "Upgrade REST API to FastAPI"
4. Add GitHub repo (optional): https://github.com/user/project

# AI-assisted task generation
1. In project view → "Generate Tasks"
2. Describe goal: "Convert Flask app to FastAPI"
3. AI generates structured task breakdown
4. Review and customize generated tasks
```

#### 3. MCP Integration with AI Clients

**Claude Desktop Configuration:**
```json
{
  "mcpServers": {
    "archon": {
      "command": "npx",
      "args": ["-y", "@anthropic-ai/mcp-server-http", "http://localhost:8051"]
    }
  }
}
```

**Cursor/Windsurf Integration:**
1. Install MCP extension
2. Add server: `http://localhost:8051`
3. Available tools: 14 specialized MCP tools for RAG and project management

#### 4. Advanced RAG Queries

```python
# Via MCP tools in your AI client
perform_rag_query(
    query="How to implement OAuth2 in FastAPI?",
    sources=["fastapi_docs", "security_guides"],
    strategy="hybrid_search"
)

# Results include:
# - Relevant documentation snippets
# - Code examples with context
# - Best practices and warnings
# - Related concepts and patterns
```

### Specialized Agent Usage

#### Archon Master Agent

```bash
# System architecture analysis
npx claude-flow agent spawn archon-master "Analyze current system architecture and suggest improvements"

# Complex integration guidance
npx claude-flow agent spawn archon-master "Design enterprise-grade Archon deployment with monitoring"

# Performance optimization
npx claude-flow agent spawn archon-master "Optimize RAG performance for 10M+ document corpus"
```

#### RAG Specialist Agent

```bash
# Search optimization
npx claude-flow agent spawn archon-rag-specialist "Implement hybrid search for technical documentation"

# Knowledge extraction
npx claude-flow agent spawn archon-rag-specialist "Extract and index code patterns from GitHub repositories"

# Query strategy tuning
npx claude-flow agent spawn archon-rag-specialist "Tune embedding strategy for domain-specific queries"
```

#### Project Manager Agent

```bash
# Project initialization
npx claude-flow agent spawn archon-project-manager "Set up ML project with automated task generation"

# Workflow optimization
npx claude-flow agent spawn archon-project-manager "Design agile workflow for distributed team"

# Progress tracking
npx claude-flow agent spawn archon-project-manager "Generate comprehensive project status report"
```

---

## Configuration Options

### Core System Configuration

#### Service Configuration

| Setting | Default | Description | Impact |
|---------|---------|-------------|--------|
| `ARCHON_SERVER_PORT` | 8181 | Main API server port | Client connections |
| `ARCHON_MCP_PORT` | 8051 | MCP protocol port | AI client integration |
| `ARCHON_AGENTS_PORT` | 8052 | AI agents service port | Agent processing |
| `LOG_LEVEL` | INFO | Logging verbosity | Troubleshooting |

#### Performance Tuning

```bash
# High-performance configuration
AGENTS_ENABLED=true
CLAUDE_FLOW_ENABLED=true
WORKERS=4                    # Uvicorn worker processes
MAX_CONNECTIONS=1000         # Database connection pool
VECTOR_DIMENSION=1536        # OpenAI embedding dimension
CHUNK_SIZE=1000             # Document chunk size
```

#### RAG Strategy Configuration

```json
{
  "rag_strategies": {
    "contextual_embeddings": {
      "enabled": true,
      "improvement_target": 0.30,
      "context_window": 2000
    },
    "hybrid_search": {
      "enabled": true,
      "bm25_weight": 0.3,
      "semantic_weight": 0.7
    },
    "agentic_rag": {
      "enabled": true,
      "max_iterations": 3,
      "confidence_threshold": 0.85
    },
    "reranking": {
      "enabled": true,
      "model": "cross-encoder/ms-marco-MiniLM-L-2-v2"
    }
  }
}
```

### Master Agent Configuration

#### Agent Specialization Settings

```python
# Master agent configuration
{
    "model": "openai:gpt-4o",
    "max_retries": 3,
    "timeout": 120,
    "enable_rate_limiting": True,
    "rag_integration": True,
    "swarm_coordination": True,
    "fallback_strategies": ["single_agent", "rag_enhanced"],
    "performance_targets": {
        "simple_query": "200ms",
        "complex_query": "500ms",
        "multi_step_task": "2s"
    }
}
```

#### Capability Matrix Settings

```yaml
Agent Roles:
  archon-master:
    expertise_level: "expert"
    domains: ["architecture", "integration", "optimization"]
    capabilities: ["system_design", "performance_tuning", "troubleshooting"]
    
  archon-rag-specialist:
    expertise_level: "advanced"
    domains: ["knowledge_retrieval", "search_optimization"]
    capabilities: ["vector_operations", "query_strategies", "accuracy_tuning"]
    
  archon-project-manager:
    expertise_level: "advanced"
    domains: ["task_management", "workflow_design"]
    capabilities: ["project_lifecycle", "progress_tracking", "collaboration"]
```

### Environment Variables Reference

#### Required Configuration

```bash
# Database (Required)
SUPABASE_URL=                    # Your Supabase project URL
SUPABASE_SERVICE_KEY=            # Service role key (use legacy format)

# AI Providers (At least one required)
OPENAI_API_KEY=                  # OpenAI API key
GEMINI_API_KEY=                  # Google Gemini API key
OLLAMA_BASE_URL=                 # Local Ollama URL
```

#### Optional Configuration

```bash
# Service Ports
ARCHON_UI_PORT=3737
ARCHON_SERVER_PORT=8181
ARCHON_MCP_PORT=8051
ARCHON_AGENTS_PORT=8052

# Feature Flags
AGENTS_ENABLED=true
CLAUDE_FLOW_ENABLED=true
LOGFIRE_ENABLED=false

# Performance
LOG_LEVEL=INFO
WORKERS=1
MAX_CONNECTIONS=100

# Network
HOST=localhost
CORS_ORIGINS=["http://localhost:3737"]
```

---

## Troubleshooting

### Common Issues and Solutions

#### Service Startup Issues

**Problem**: Services fail to start or show unhealthy status

```bash
# Diagnosis steps:
1. Check Docker Desktop is running
2. Verify ports are available:
   lsof -i :3737 -i :8181 -i :8051 -i :8052
3. Check environment variables:
   docker compose config
4. Review service logs:
   docker compose logs -f archon-server

# Common solutions:
- Update port configuration in .env
- Restart Docker Desktop
- Check firewall settings
- Verify Supabase credentials
```

**Problem**: Database connection errors

```bash
# Check database configuration:
1. Verify SUPABASE_URL format: https://xyz.supabase.co
2. Confirm service key is correct (use legacy format)
3. Test connection:
   curl -H "apikey: YOUR_SERVICE_KEY" "YOUR_SUPABASE_URL/rest/v1/"

# Migration issues:
1. Check if tables exist in Supabase dashboard
2. Re-run migration scripts if needed
3. Clear browser cache and restart services
```

#### MCP Integration Issues

**Problem**: AI clients can't connect to MCP server

```bash
# Verification steps:
1. Test MCP server health:
   curl http://localhost:8051/health
2. Check MCP server logs:
   docker compose logs -f archon-mcp
3. Verify client configuration matches server port

# Configuration fixes:
- Update client config with correct port
- Check network connectivity
- Verify MCP protocol version compatibility
```

**Problem**: MCP tools not available in AI client

```bash
# Debug tools availability:
1. List available tools:
   curl http://localhost:8051/tools
2. Check tool registration:
   docker compose logs archon-mcp | grep "tool"
3. Verify client permissions and capabilities

# Common solutions:
- Restart MCP client
- Update client to latest version
- Check tool-specific requirements
```

#### Performance Issues

**Problem**: Slow query responses or high resource usage

```bash
# Performance diagnosis:
1. Check system resources:
   docker stats
2. Monitor query response times in logs
3. Analyze database query performance
4. Review vector index efficiency

# Optimization strategies:
- Increase worker processes: WORKERS=4
- Optimize chunk size: CHUNK_SIZE=800
- Enable connection pooling
- Add vector index optimization
- Scale database resources
```

**Problem**: Memory consumption too high

```bash
# Memory optimization:
1. Monitor container memory usage:
   docker stats --format "table {{.Name}}\t{{.MemUsage}}"
2. Adjust memory limits in docker-compose.yml
3. Optimize embedding cache size
4. Reduce concurrent operations

# Configuration adjustments:
deploy:
  resources:
    limits:
      memory: 2G    # Adjust per service
    reservations:
      memory: 512M
```

### Diagnostic Commands

#### System Health Check

```bash
# Comprehensive system check
make check                    # Environment verification
docker compose ps             # Service status
curl http://localhost:8181/health  # API health
curl http://localhost:8051/tools   # MCP tools list

# Log analysis
docker compose logs --tail=50 archon-server
docker compose logs --tail=50 archon-mcp
docker compose logs --tail=50 archon-agents
```

#### Performance Monitoring

```bash
# Resource monitoring
docker stats                  # Real-time resource usage
docker system df             # Disk usage
docker system events         # System events

# Application monitoring
curl http://localhost:8181/metrics    # Application metrics
tail -f logs/archon.log              # Application logs
```

#### Database Diagnostics

```sql
-- In Supabase SQL Editor
-- Check table status
SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del 
FROM pg_stat_user_tables 
WHERE schemaname = 'public';

-- Check vector index performance
SELECT * FROM pg_stat_user_indexes 
WHERE indexrelname LIKE '%vector%';

-- Analyze query performance
SELECT query, calls, mean_exec_time, rows 
FROM pg_stat_statements 
WHERE query LIKE '%archon%' 
ORDER BY mean_exec_time DESC;
```

### Recovery Procedures

#### Complete System Reset

```bash
# ⚠️ WARNING: This will delete ALL Archon data

# 1. Stop all services
docker compose down

# 2. Remove containers and volumes
docker compose down --volumes --remove-orphans

# 3. Reset database (run in Supabase SQL Editor)
-- Execute contents of migration/RESET_DB.sql

# 4. Rebuild from scratch
docker compose up --build -d

# 5. Reconfigure system
# - Set API keys via UI
# - Re-upload documents
# - Recreate projects
```

#### Partial Recovery (Preserve Data)

```bash
# Service restart without data loss
docker compose restart

# Individual service recovery
docker compose restart archon-server
docker compose restart archon-mcp

# Clear cache but preserve data
docker volume prune
docker compose up -d
```

---

## Common Workflows

### Development Workflow

#### Setting Up a New Project

1. **Project Initialization**:
   ```bash
   1. Navigate to Projects → "Create New Project"
   2. Configure project details:
      - Name: "E-commerce API"
      - Description: "Modernize legacy e-commerce system"
      - Repository: github.com/company/ecommerce-api
      - Type: "backend-modernization"
   ```

2. **Knowledge Base Preparation**:
   ```bash
   1. Knowledge Base → "Crawl Website"
      - Target: https://fastapi.tiangolo.com/
      - Depth: 3 levels
      - Include code examples: Yes
   
   2. Knowledge Base → "Upload Documents"
      - Upload: existing_api_documentation.pdf
      - Upload: architecture_diagrams.pdf
      - Upload: migration_requirements.md
   ```

3. **AI-Assisted Task Generation**:
   ```bash
   1. In project view → "Generate Tasks"
   2. Prompt: "Create comprehensive task breakdown for migrating Django REST API to FastAPI while maintaining backwards compatibility"
   3. Review generated tasks
   4. Customize priorities and dependencies
   5. Assign tasks to team members
   ```

#### Daily Development Cycle

1. **Morning Standup Preparation**:
   ```bash
   # Generate status report
   npx claude-flow agent spawn archon-project-manager "Generate daily standup report for all active projects"
   
   # Results include:
   # - Completed tasks from yesterday
   # - Today's priorities
   # - Blocked items requiring attention
   # - Risk assessment and mitigation
   ```

2. **Development with AI Assistance**:
   ```bash
   # Connect your AI coding assistant to Archon MCP
   # Available throughout your coding session:
   - perform_rag_query: Search knowledge base
   - search_code_examples: Find implementation patterns  
   - create_task: Add new requirements as tasks
   - update_task: Track progress on current work
   - get_project_context: Understand full project scope
   ```

3. **End-of-Day Progress Update**:
   ```bash
   # Automatic progress tracking via MCP tools
   # Manual updates via web UI:
   1. Review task completion status
   2. Update task descriptions with findings
   3. Create new tasks for tomorrow
   4. Document decisions and rationale
   ```

### Knowledge Management Workflow

#### Comprehensive Documentation Indexing

1. **Multi-Source Knowledge Gathering**:
   ```bash
   # Technical documentation
   Knowledge Base → Crawl:
   - https://docs.python.org/3/
   - https://fastapi.tiangolo.com/
   - https://docs.pydantic.dev/
   
   # Framework-specific guides
   Upload Documents:
   - django_to_fastapi_migration_guide.pdf
   - api_security_best_practices.pdf
   - performance_optimization_handbook.pdf
   
   # Team knowledge
   - meeting_notes_architecture_decisions.md
   - code_review_guidelines.md
   - deployment_procedures.md
   ```

2. **Quality Assurance and Optimization**:
   ```bash
   # Test search effectiveness
   Search Test Queries:
   - "How to handle database migrations?"
   - "FastAPI authentication patterns"
   - "Performance monitoring setup"
   - "Docker deployment strategies"
   
   # Optimize based on results:
   - Adjust chunk size for better granularity
   - Add missing documentation sources
   - Improve search query patterns
   - Tag content for better filtering
   ```

3. **Maintenance and Updates**:
   ```bash
   # Regular knowledge base maintenance
   Weekly Tasks:
   - Review and update outdated content
   - Add new documentation sources
   - Analyze search patterns and gaps
   - Optimize embedding strategies
   
   # Version control integration:
   - Auto-update from repository changes
   - Track documentation version alignment
   - Validate code example currency
   ```

### Multi-Agent Coordination Workflow

#### Complex Problem Solving

1. **Problem Analysis Phase**:
   ```bash
   # Deploy analysis team
   npx claude-flow swarm init --topology mesh
   npx claude-flow agent spawn archon-master "Analyze system architecture bottlenecks in e-commerce API"
   npx claude-flow agent spawn archon-rag-specialist "Research performance optimization strategies for high-traffic APIs"
   npx claude-flow agent spawn performance-analyzer "Profile current system performance characteristics"
   ```

2. **Solution Design Phase**:
   ```bash
   # Coordinate solution development
   npx claude-flow agent spawn system-architect "Design scalable microservices architecture based on analysis findings"
   npx claude-flow agent spawn security-specialist "Assess security implications of proposed architecture changes"
   npx claude-flow agent spawn database-expert "Design data layer optimization strategy"
   ```

3. **Implementation Planning Phase**:
   ```bash
   # Create execution roadmap
   npx claude-flow agent spawn archon-project-manager "Create detailed implementation roadmap with risk assessment"
   npx claude-flow agent spawn technical-writer "Generate comprehensive technical specification"
   npx claude-flow agent spawn qa-specialist "Design testing strategy for architecture migration"
   ```

---

## Best Practices

### System Configuration

#### Production Deployment

```bash
# Recommended production environment variables
ENVIRONMENT=production
LOG_LEVEL=INFO
AGENTS_ENABLED=true
CLAUDE_FLOW_ENABLED=true

# Resource allocation
WORKERS=4                    # Scale based on CPU cores
MAX_CONNECTIONS=500          # Database connection pool
MEMORY_LIMIT=4G             # Per service memory limit

# Security settings
CORS_ORIGINS=["https://your-domain.com"]
ALLOWED_HOSTS=["your-domain.com"]
SSL_REDIRECT=true

# Monitoring
LOGFIRE_ENABLED=true
HEALTH_CHECK_INTERVAL=30
METRICS_COLLECTION=true
```

#### High-Performance Configuration

```yaml
# Docker Compose overrides for production
services:
  archon-server:
    deploy:
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
      - WORKERS=4
      - MAX_CONNECTIONS=500
```

### Knowledge Management Best Practices

#### Document Organization Strategy

1. **Source Categorization**:
   ```bash
   # Organize by domain and freshness
   Categories:
   - technical_docs_current     # Latest official documentation
   - technical_docs_legacy      # Historical reference material
   - internal_knowledge         # Team-specific information
   - code_examples_verified     # Tested code snippets
   - architectural_decisions    # ADRs and design decisions
   ```

2. **Tagging and Metadata Strategy**:
   ```json
   {
     "document_metadata": {
       "domain": ["backend", "frontend", "devops"],
       "technology": ["python", "fastapi", "docker"],
       "complexity": ["beginner", "intermediate", "expert"],
       "last_verified": "2024-09-05",
       "source_authority": "official"
     }
   }
   ```

3. **Quality Control Process**:
   ```bash
   # Weekly knowledge base maintenance
   1. Review search analytics for gap identification
   2. Update outdated technical documentation
   3. Verify code example functionality
   4. Optimize chunk boundaries for better retrieval
   5. Add missing cross-references and tags
   ```

#### RAG Optimization Guidelines

```python
# Optimal RAG strategy configuration
rag_config = {
    # Document processing
    "chunk_size": 800,           # Balance context and precision
    "chunk_overlap": 200,        # Maintain context continuity
    "embedding_model": "text-embedding-3-small",  # Cost-effective accuracy
    
    # Search strategy
    "hybrid_search_weight": {
        "semantic": 0.7,         # Favor meaning over keywords
        "keyword": 0.3           # Include exact matches
    },
    
    # Quality thresholds
    "relevance_threshold": 0.75, # Filter low-quality matches
    "max_results": 5,           # Optimal for LLM context
    "reranking_enabled": True    # Improve result ordering
}
```

### Development Workflow Best Practices

#### Agent Utilization Strategy

1. **Task-Appropriate Agent Selection**:
   ```bash
   # Simple queries → Specialized agents
   archon-rag-specialist: "Find FastAPI authentication examples"
   archon-project-manager: "Update task status for user authentication"
   
   # Complex queries → Master agent
   archon-master: "Design complete authentication system with OAuth2, JWT, and role-based access control"
   
   # Multi-domain queries → Agent coordination
   claude-flow swarm: "Implement secure API with monitoring, testing, and deployment pipeline"
   ```

2. **Progressive Problem Solving**:
   ```bash
   # Start simple, escalate complexity
   Level 1: Direct tool usage (search, create task)
   Level 2: Specialized agent consultation
   Level 3: Master agent orchestration
   Level 4: Multi-agent swarm coordination
   ```

3. **Context Management**:
   ```bash
   # Maintain conversation continuity
   - Use project context consistently
   - Reference previous decisions in new tasks
   - Document rationale for future reference
   - Update knowledge base with new insights
   ```

#### Performance Optimization

1. **Query Optimization**:
   ```bash
   # Efficient search patterns
   Good: "FastAPI dependency injection patterns for database connections"
   Better: "FastAPI database dependency injection with connection pooling and transaction management"
   
   # Use specific technical terms
   Good: "authentication setup"
   Better: "OAuth2 JWT authentication with refresh tokens"
   ```

2. **Resource Management**:
   ```bash
   # Monitor and optimize
   - Track query response times
   - Analyze embedding cache hit rates
   - Monitor database query performance
   - Optimize vector index configuration
   - Scale services based on usage patterns
   ```

3. **Caching Strategy**:
   ```json
   {
     "cache_layers": {
       "embedding_cache": "24h TTL for stable documents",
       "search_cache": "1h TTL for repeated queries", 
       "session_cache": "30min TTL for conversation context",
       "tool_cache": "5min TTL for dynamic data"
     }
   }
   ```

### Security and Maintenance

#### Security Best Practices

```bash
# Environment security
- Store API keys in environment variables only
- Use service role keys with minimal required permissions
- Enable CORS restrictions for production deployments
- Implement rate limiting for public-facing services
- Regular security updates for dependencies

# Data protection
- Encrypt sensitive data at rest
- Use secure communication channels (HTTPS)
- Implement audit logging for sensitive operations
- Regular backup and recovery testing
- Monitor for unusual access patterns
```

#### Maintenance Schedule

```bash
# Daily
- Monitor service health and performance
- Review error logs and alerts
- Check disk space and resource usage

# Weekly  
- Update knowledge base with new content
- Review and optimize search patterns
- Analyze usage metrics and patterns
- Update documentation and procedures

# Monthly
- Security updates and dependency upgrades
- Performance optimization and tuning
- Backup validation and recovery testing
- Capacity planning and scaling assessment
```

This comprehensive user guide provides everything needed to successfully deploy, configure, and operate the Archon Master Agent System. For additional support, consult the developer documentation and architecture guides.