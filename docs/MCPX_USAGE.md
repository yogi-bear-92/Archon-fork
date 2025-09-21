# 🌙 Lunar MCPX Gateway Usage Guide

## Overview
Your Lunar MCPX Gateway is now running and aggregating multiple MCP servers into a single interface, following official Lunar.dev architecture patterns.

## Current Status
- ✅ **MCPX Gateway**: Running in Docker (archon-lunar-mcpx)
- ✅ **Archon PRP**: HTTP transport (localhost:8181)
- ✅ **Claude Flow**: Command transport (claude-flow@alpha)  
- ⏸️ **Serena**: Disabled (Docker mode compatibility)
- ❌ **Flow-Nexus**: Not running (port 8051 unavailable)

**Overall Health: DEGRADED (66.7%) - 2/4 servers active**

## Integration Methods

### 1. Claude Desktop Integration

Add to your `~/.claude_desktop/mcp_settings.json`:

```json
{
  "mcpServers": {
    "archon-lunar-mcpx": {
      "command": "docker",
      "args": ["exec", "-i", "archon-lunar-mcpx", "node", "lunar-mcpx-gateway.js"],
      "env": {
        "SERVICE_DISCOVERY_MODE": "docker_compose"
      }
    }
  }
}
```

### 2. Direct MCP Protocol Usage

```bash
# Test MCPX gateway directly
echo '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}' | \
  docker exec -i archon-lunar-mcpx node lunar-mcpx-gateway.js
```

### 3. API Integration

Access underlying services that MCPX manages:

```bash
# Archon API (managed by MCPX)
curl http://localhost:8181/api/projects

# MCP Bridge (managed by MCPX)  
curl http://localhost:8051/mcp

# Frontend UI
open http://localhost:3737
```

## Available Tools Through MCPX

The MCPX Gateway provides unified access to tools from:

### Archon PRP Tools:
- `archon_project_create` - Create new projects
- `archon_project_list` - List all projects
- `archon_task_create` - Create tasks
- `archon_task_update` - Update task status
- `archon_rag_query` - Query knowledge base
- `archon_knowledge_store` - Store knowledge

### Claude Flow Tools:
- `claude_flow_sparc_execute` - Execute SPARC workflows
- `claude_flow_swarm_init` - Initialize agent swarms
- `claude_flow_agent_spawn` - Spawn individual agents
- `claude_flow_task_orchestrate` - Orchestrate complex tasks

### MCPX Management Tools:
- `mcpx_server_status` - Get server health status
- `mcpx_server_restart` - Restart specific servers
- `mcpx_route_tool` - Route tools to best available server

## Key Features

### 🔄 **Automatic Failover**
MCPX automatically routes requests to healthy servers and handles failures gracefully.

### ⚡ **Priority-Based Routing**  
Tools are routed to servers based on priority (1=highest):
1. Archon PRP (Priority 1)
2. Claude Flow (Priority 2)  
3. Serena (Priority 3, disabled)
4. Flow-Nexus (Priority 4, unavailable)

### 🛡️ **Circuit Breaker Protection**
Failed servers are temporarily disabled to prevent cascading failures.

### 📊 **Health Monitoring**
Real-time health checks every 30 seconds with automatic recovery.

## System Architecture

```
┌─────────────────────────────────────────────────┐
│  🌙 Lunar MCPX Gateway (Docker)                 │
│  ├─ stdio transport (MCP protocol)              │  
│  ├─ HTTP Control Plane (port 8090) *planned    │
│  └─ Server aggregation & routing logic          │
├─────────────────────────────────────────────────┤
│  📊 Managed MCP Servers                        │
│  ├─ ✅ Archon PRP (HTTP)                        │
│  ├─ ✅ Claude Flow (Command)                     │
│  ├─ ⏸️ Serena (Disabled)                        │
│  └─ ❌ Flow-Nexus (Unavailable)                 │
├─────────────────────────────────────────────────┤
│  🎯 Backend Services                           │
│  ├─ Archon Server (localhost:8181)             │
│  ├─ Archon MCP Bridge (localhost:8051)         │  
│  └─ Frontend UI (localhost:3737)               │
└─────────────────────────────────────────────────┘
```

## Troubleshooting

### Check MCPX Status
```bash
docker logs archon-lunar-mcpx
docker ps | grep archon
```

### Test Individual Services
```bash
# Test Archon API
curl http://localhost:8181/health

# Test MCP Bridge
curl http://localhost:8051/health

# Test Frontend
curl http://localhost:3737
```

### Restart MCPX Gateway
```bash
docker restart archon-lunar-mcpx
```

## Performance Metrics

- **Server Health**: 2/4 servers active (66.7%)
- **Response Time**: Optimized with circuit breakers
- **Reliability**: Automatic failover and recovery
- **Scalability**: Docker-native horizontal scaling

## Next Steps

1. **Enable Flow-Nexus**: Start Flow-Nexus server for workflow tools
2. **Add More Servers**: Extend MCPX with additional MCP servers
3. **Monitor Performance**: Use built-in health monitoring
4. **Scale Horizontally**: Deploy multiple MCPX instances