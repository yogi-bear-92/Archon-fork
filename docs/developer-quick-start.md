# Developer Quick Start: Unified Serena + Archon + Claude Flow

## 🚨 CRITICAL MEMORY SITUATION

Your system is currently at **99.6% memory usage** with only ~66MB free out of 16GB total. This requires immediate action before development can proceed safely.

## IMMEDIATE ACTION REQUIRED

### Step 1: Emergency Memory Optimization
```bash
# Run immediate memory fix (stops competing services, clears caches)
cd /Users/yogi/Projects/Archon-fork
./scripts/immediate-memory-fix.sh --emergency

# This will:
# - Kill memory-intensive processes
# - Clear all caches aggressively 
# - Start only essential services
# - Enable memory monitoring
```

### Step 2: Verify Memory Recovery
```bash
# Check memory status
./scripts/immediate-memory-fix.sh --status

# Goal: Reduce usage to <85% before proceeding
```

### Step 3: Start Unified Development Environment
```bash
# Once memory is stable, start optimized services
./scripts/unified-startup.sh optimized

# This starts Serena + Archon with memory limits
# Claude Flow starts on-demand only
```

## UNIFIED WORKFLOW ARCHITECTURE

### System Hierarchy (Memory Optimized)

```
┌─────────────────────────────────────────┐
│           DEVELOPER INTERFACE           │ ← Your primary workspace
│        Claude Code + VS Code            │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│           EXECUTION LAYER               │ ← All actual work happens here
│           (Claude Code)                 │   Memory: 2-3GB max
│  • Task spawning                       │
│  • File operations                     │
│  • Git management                      │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│        CODE INTELLIGENCE               │ ← Semantic understanding
│          (Serena MCP)                   │   Memory: 512MB-1GB
│  • LSP services                        │
│  • Code analysis                       │
│  • Symbol navigation                   │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│      KNOWLEDGE ORCHESTRATION           │ ← RAG and project management
│         (Archon PRP)                    │   Memory: 1-1.5GB
│  • Knowledge base                      │
│  • Task management                     │
│  • Multi-agent coordination            │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│       SWARM COORDINATION               │ ← Advanced coordination
│        (Claude Flow)                    │   Memory: 0.5-1GB
│  • Neural training                     │   (On-demand only)
│  • Performance optimization            │
│  • Cross-session memory               │
└─────────────────────────────────────────┘
```

## DEVELOPMENT PATTERNS

### Memory-Aware Agent Spawning

```javascript
// ✅ CORRECT: Memory-conscious concurrent execution
[Single Message with Memory Checks]:
  
  // Check memory before spawning
  Bash("node scripts/memory-monitor.js --check-spawn 512MB")
  
  // Spawn maximum 2-3 agents based on available memory
  Task("Code Analyzer", `
    Analyze project structure with Serena integration.
    Memory limit: 256MB. Use semantic cache.
    Hooks: npx claude-flow hooks pre-task --description "Analysis"
  `, "code-analyzer")
  
  Task("Backend Developer", `
    Implement features using Archon PRP patterns.
    Memory limit: 512MB. Query knowledge base efficiently.
    Hooks: npx claude-flow hooks post-edit --file {file}
  `, "backend-dev")
  
  // Batch all todos together (required pattern)
  TodoWrite([
    {content: "Analyze codebase structure", status: "in_progress"},
    {content: "Design API endpoints", status: "pending"},
    {content: "Implement business logic", status: "pending"},
    {content: "Write comprehensive tests", status: "pending"},
    {content: "Document integration points", status: "pending"}
  ])
  
  // Batch all file operations
  MultiEdit("src/main.py", [...edits])
  Read("python/requirements.txt")
  Write("docs/architecture.md", content)
```

### Progressive Development Workflow

```javascript
// Phase 1: Analysis (Lightweight, Serena-heavy)
[Memory Budget: 1GB]:
  Task("Researcher", "Analyze requirements using Serena semantic search", "researcher")
  Task("Architect", "Design system using Archon knowledge base", "system-architect")

// Phase 2: Implementation (Moderate memory, All systems)  
[Memory Budget: 2GB]:
  Task("Coder", "Implement features with PRP guidance", "coder")
  Task("Reviewer", "Review code with semantic awareness", "reviewer")

// Phase 3: Validation (Minimal memory, focused testing)
[Memory Budget: 1GB]:
  Task("Tester", "Run focused test suites", "tester")
  Task("Validator", "Validate against requirements", "production-validator")
```

## KEY INTEGRATION POINTS

### 1. Serena → Claude Code
```bash
# Serena provides semantic intelligence to Claude Code tasks
Task("analyzer", "Use Serena LSP for deep code analysis", "code-analyzer")
# Agent automatically uses: mcp__serena__find_symbol, get_symbols_overview
```

### 2. Archon → Task Management
```bash
# Archon provides knowledge and project context
Task("researcher", "Query Archon knowledge base for patterns", "researcher") 
# Agent automatically uses: mcp__archon__perform_rag_query, create_task
```

### 3. Claude Flow → Coordination
```bash
# Claude Flow provides swarm coordination and optimization
Task("coordinator", "Optimize multi-agent workflow", "task-orchestrator")
# Agent automatically uses: mcp__claude-flow__task_orchestrate, neural_train
```

## MEMORY MANAGEMENT COMMANDS

### Monitor Memory
```bash
# Check current status
node scripts/memory-monitor.js --status

# Continuous monitoring
node scripts/memory-monitor.js --monitor

# Check if agent can spawn
node scripts/memory-monitor.js --check-spawn 256MB
```

### Emergency Actions
```bash
# Emergency cleanup (stops all services)
./scripts/immediate-memory-fix.sh --emergency

# Cleanup only (keeps essential services)
./scripts/immediate-memory-fix.sh --cleanup-only

# Stop all services gracefully
./scripts/stop-services.sh
```

### Service Management
```bash
# Start in different modes based on available memory
./scripts/unified-startup.sh minimal    # <4GB free (Serena only)
./scripts/unified-startup.sh optimized # 4-8GB free (Serena + Archon)
./scripts/unified-startup.sh normal    # >8GB free (All services)
./scripts/unified-startup.sh auto      # Auto-detect best mode
```

## COORDINATION HOOKS

Every agent spawned via Task tool automatically runs:

### Before Starting
```bash
npx claude-flow hooks pre-task --description "[task description]"
npx claude-flow hooks session-restore --session-id "unified-[timestamp]"
```

### During Execution
```bash
npx claude-flow hooks post-edit --file "[edited-file]" --memory-key "agent/progress"
npx claude-flow hooks notify --message "[status update]"
```

### After Completion
```bash
npx claude-flow hooks post-task --task-id "[task-id]"
npx claude-flow hooks session-end --export-metrics true
```

## PERFORMANCE TARGETS

### Memory Usage Goals
- **Phase 1** (Emergency): Reduce to <90% (15.3GB)
- **Phase 2** (Stabilize): Achieve <75% (12.8GB)
- **Phase 3** (Optimize): Target <50% (8.5GB)

### Response Times
- **Code Analysis**: <200ms (Serena cached)
- **Knowledge Queries**: <300ms (Archon + pgvector)
- **Agent Coordination**: <500ms (Claude Flow)
- **File Operations**: <100ms (Claude Code direct)

## TROUBLESHOOTING

### Memory Still Critical After Cleanup
```bash
# Force restart of system (macOS)
sudo purge

# Check for memory leaks
lsof +L1  # Check for deleted files still held open

# Restart Claude Code completely
pkill -f claude-code && sleep 5 && code .
```

### Services Won't Start
```bash
# Check port conflicts
lsof -i :8051 -i :8080 -i :8052

# Check logs
tail -f logs/serena.log
tail -f logs/archon.log
tail -f logs/memory-monitor.log
```

### Poor Performance
```bash
# Check memory pressure
./scripts/memory-monitor.js --status

# Reduce concurrent agents
# Use sequential rather than parallel task execution
# Clear caches more frequently
```

## SUCCESS METRICS

Your unified workflow is working when you see:

1. **Memory Usage** <75% consistently
2. **Agent Response Times** <1 second
3. **No Service Crashes** during development
4. **Seamless Integration** between Serena analysis, Archon knowledge, and Claude Code execution
5. **Effective Coordination** via Claude Flow hooks

## IMMEDIATE NEXT STEPS

1. **Run emergency memory fix** (critical)
2. **Verify memory recovery** (<90% usage)
3. **Start optimized services** (Serena + Archon only)
4. **Test integration** with simple Task spawning
5. **Monitor performance** and adjust memory limits

Remember: This unified workflow prioritizes **stability over features** until memory pressure is resolved. Once optimized, you'll have the full power of all three systems working together efficiently.