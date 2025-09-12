# ULTIMATE INTEGRATED AI DEVELOPMENT PLATFORM
## Archon PRP + Serena + Claude Flow + Memory-Optimized Coordination

## 🚨 CRITICAL: CONCURRENT EXECUTION & MEMORY MANAGEMENT

**ABSOLUTE RULES FOR ARCHON INTEGRATION:**
1. ALL operations MUST be concurrent/parallel in a single message
2. **MEMORY-AWARE SCALING**: Auto-adapt agent count based on system memory (current: 133MB available)
3. **NEVER save working files to root folder** - use proper subdirectories
4. **USE CLAUDE CODE'S TASK TOOL** for spawning agents concurrently with Archon coordination
5. **ARCHON PRP INTEGRATION**: Progressive refinement with streaming and memory constraints

### ⚡ GOLDEN RULE: "1 MESSAGE = ALL OPERATIONS + MEMORY MONITORING + ARCHON PRP"

**MANDATORY ARCHON-OPTIMIZED PATTERNS:**
- **Memory Check FIRST**: Always verify resources before agent spawning (133MB = LIMITED MODE)
- **TodoWrite**: Adaptive batching based on memory (3-15 todos for current 133MB)
- **Task tool**: Spawn ALL agents in ONE message with Archon coordination hooks
- **File operations**: Stream all operations >10MB with immediate cleanup
- **Archon Integration**: Use PRP cycles with semantic analysis and progressive refinement
- **Serena Coordination**: Semantic analysis with intelligent caching (25MB limit)

### 🎯 CRITICAL: Claude Code Task Tool + Archon PRP Integration

**Claude Code Task tool is PRIMARY for Archon agent execution:**
```javascript
// ✅ CORRECT: Use Claude Code Task + Archon PRP coordination
[Single Message - Memory-Aware Archon Integration]:
  // Step 1: Check available memory and scale agents accordingly
  Bash "vm_stat | grep 'Pages free' | awk '{print $3}' | sed 's/\.//' | awk '{print ($1 * 4096 / 1024 / 1024) \" MB\"}'"
  
  // Step 2: Initialize Archon PRP with memory constraints
  mcp__claude-flow__swarm_init { topology: "hierarchical", maxAgents: 3, memoryLimit: "100MB" }
  
  // Step 3: Spawn agents via Claude Code with Archon coordination
  Task("Archon Research Agent", "Use Serena semantic analysis + Archon PRP specification phase. Memory limit 25MB cache.", "researcher")
  Task("Archon Development Agent", "Implement with progressive refinement cycles. Use streaming for large operations.", "coder")
  Task("Archon Validation Agent", "Test with semantic analysis and memory monitoring. Progressive validation.", "tester")
  
  // Step 4: Batch all todos with memory awareness
  TodoWrite { todos: [
    {id: "1", content: "Memory monitoring active (133MB)", status: "in_progress", priority: "critical"},
    {id: "2", content: "Archon PRP specification phase", status: "in_progress", priority: "high"},
    {id: "3", content: "Progressive refinement cycles", status: "pending", priority: "high"}
  ]}
```

### 📁 ARCHON FILE ORGANIZATION (NEVER ROOT FOLDER)

**Archon-specific directory structure:**
- `/src` - Source code with Archon PRP structure
- `/tests` - Archon validation tests
- `/docs` - Progressive refinement documentation
- `/config` - Archon and Serena configurations
- `/scripts` - Archon PRP automation scripts
- `/.archon-prp` - Progressive refinement data (compressed)
- `/.serena-cache` - Semantic analysis cache (auto-cleanup)
- `/.claude-flow` - Coordination metrics (streaming)

## 🔄 ARCHON GIT CHECKPOINT INTEGRATION

**ARCHON-ENHANCED CHECKPOINTS:**
```bash
# Enable Claude Code + Archon git checkpoints
git config --local claude-code.auto-checkpoint true
git config --local claude-code.checkpoint-frequency "after-prp-cycles"
git config --local claude-code.checkpoint-message-template "🔄 Archon-checkpoint: {prp-phase}"
```

**ARCHON CHECKPOINT TRIGGERS:**
- Progressive refinement cycle completions
- Serena semantic analysis milestones
- Memory-critical operation boundaries
- Cross-agent coordination checkpoints
- Performance optimization points

## 🏗️ INTEGRATED SYSTEM ARCHITECTURE

### **TIER 1: CLAUDE CODE (EXECUTION ENGINE) - 85% WORKLOAD**
**ARCHON-ENHANCED RESPONSIBILITIES:**
- **Task tool**: Spawn agents with Archon PRP coordination and memory monitoring
- **File operations**: Streaming operations with Archon progressive refinement
- **Bash commands**: System operations with Archon automation hooks
- **Git operations**: Version control with PRP cycle checkpoints
- **TodoWrite**: Memory-adaptive batching (3-15 items based on 133MB available)

### **TIER 2: SERENA (CODE INTELLIGENCE) - 10% WORKLOAD**
**ARCHON SEMANTIC INTEGRATION:**
- **MCP Server**: Semantic analysis with Archon PRP context (25MB cache limit)
- **LSP Integration**: Real-time code intelligence with progressive refinement feedback
- **Cross-language Support**: Multi-language coordination for Archon projects
- **Semantic Caching**: Memory-efficient with auto-expiry (current: 25MB limit)

### **TIER 3: ARCHON PRP (PROGRESSIVE REFINEMENT) - 3% WORKLOAD**
**CORE ARCHON RESPONSIBILITIES:**
- **FastAPI Orchestration**: Lightweight API coordination (port 8080)
- **PydanticAI Agents**: Memory-constrained progressive agents (port 8052)
- **Progressive Cycles**: Adaptive 2-4 refinement cycles based on 133MB memory
- **RAG Enhancement**: Streaming vector operations with semantic integration
- **Socket.IO Coordination**: Real-time PRP updates with minimal overhead

### **TIER 4: CLAUDE FLOW (COORDINATION) - 2% WORKLOAD**
**ARCHON SWARM COORDINATION:**
- **Topology Management**: Memory-aware swarm initialization with Archon integration
- **Performance Monitoring**: Stream-based metrics with PRP cycle tracking
- **Neural Training**: Memory-bounded pattern learning from Archon workflows
- **Cross-session Memory**: Compressed state management with PRP persistence

## 🚀 ARCHON-OPTIMIZED AGENT SYSTEM (64+ Agents)

### **MEMORY-AWARE AGENT SCALING (133MB Available = LIMITED MODE):**

**TIER 1: CRITICAL ARCHON AGENTS (2-3 Agents Maximum)**
- `archon-master` - Master Archon coordination with PRP orchestration
- `serena-master` - Semantic intelligence with memory optimization
- `memory-coordinator` - Resource management and adaptive scaling

**TIER 2: CORE DEVELOPMENT AGENTS (Memory Permitting)**
- `sparc-coord` - SPARC methodology coordination
- `system-architect` - Architecture with progressive refinement
- `coder` - Implementation with Archon PRP cycles

**TIER 3: SPECIALIZED COORDINATION (High Memory Mode)**
- `swarm-coordination-overview` - Multi-agent orchestration
- `hierarchical-coordinator` - Queen-led swarm coordination
- `performance-benchmarker` - System performance analysis

### **ARCHON PRP SPARC WORKFLOW PHASES (Memory-Optimized):**

1. **Specification** - Requirements with Serena semantic analysis (`memory-budget: 25MB`)
2. **Pseudocode** - Algorithm design with cached patterns (`memory-budget: 20MB`)
3. **Architecture** - System design with Archon PRP streaming (`memory-budget: 30MB`)
4. **Refinement** - Progressive TDD with memory cleanup (`memory-budget: 40MB`)
5. **Completion** - Integration with real-time monitoring (`memory-budget: 15MB`)

## 🎯 ARCHON-CLAUDE FLOW COORDINATION PROTOCOL

### **INTEGRATED EXECUTION PATTERN:**

```javascript
// ARCHON + CLAUDE FLOW INTEGRATION (Single Message)
[Memory-Optimized Archon Coordination]:

  // STEP 1: Memory assessment and emergency protocols
  Bash "vm_stat | grep 'Pages free' | awk '{print ($3 * 4096 / 1024 / 1024) \" MB available\"}'"
  
  // STEP 2: Initialize Archon PRP + Claude Flow coordination
  mcp__claude-flow__swarm_init { 
    topology: "hierarchical", 
    maxAgents: 3, 
    memoryLimit: "100MB",
    archonIntegration: true,
    prpEnabled: true
  }
  
  // STEP 3: Spawn Archon-coordinated agents via Claude Code
  Task("Archon Research Specialist", "Specification phase with Serena semantic analysis. Use archon-spec-reader agent for PRP framework understanding. Memory limit 25MB.", "archon-spec-reader")
  
  Task("Archon Development Specialist", "Progressive refinement implementation. Use PRP cycles with streaming operations. Coordinate via Socket.IO (8052).", "sparc-coder")
  
  Task("Archon Performance Monitor", "Memory-aware performance tracking with Claude Flow coordination. Stream metrics to .claude-flow/metrics/", "performance-monitor")
  
  // STEP 4: Memory-constrained todos with PRP phases
  TodoWrite { todos: [
    {id: "1", content: "Memory monitoring (133MB available)", status: "in_progress", priority: "critical"},
    {id: "2", content: "Archon PRP specification phase", status: "in_progress", priority: "high"},
    {id: "3", content: "Serena semantic cache optimization", status: "pending", priority: "high"},
    {id: "4", content: "Progressive refinement cycle 1", status: "pending", priority: "medium"},
    {id: "5", content: "Socket.IO real-time coordination", status: "pending", priority: "medium"}
  ]}
  
  // STEP 5: Integrated file operations with streaming
  Bash "mkdir -p {src,tests,docs,config}/.archon-prp"
  Write "src/archon-progressive-refinement.py" 
  Write "config/archon-integration.json"
  Edit "python/src/server/main.py" # Add Archon coordination hooks
```

## 🔧 ARCHON-SPECIFIC HOOKS INTEGRATION

### **PRE-OPERATION (Archon Enhanced)**
```bash
# Archon PRP preparation with memory monitoring
npx claude-flow@alpha hooks pre-task --archon-prp --memory-check --description "[task]"
npx claude-flow@alpha hooks archon-prp-prepare --cycles=2 --memory-limit=50MB
```

### **DURING OPERATION (Progressive Coordination)**
```bash
# Real-time Archon coordination with Serena integration
npx claude-flow@alpha hooks post-edit --file "[file]" --archon-prp-cycle --serena-analyze
npx claude-flow@alpha hooks archon-socket-notify --port=8052 --message="[progress]"
```

### **POST-OPERATION (Memory Recovery)**
```bash
# Archon cleanup with progressive state persistence
npx claude-flow@alpha hooks post-task --archon-prp-complete --memory-recovery
npx claude-flow@alpha hooks archon-prp-persist --compress --export-metrics
```

## 🚀 ARCHON PERFORMANCE METRICS & INTEGRATION TARGETS

### **INTEGRATED PERFORMANCE GOALS:**
- **84.8% SWE-Bench solve rate** (enhanced with Archon PRP)
- **47% token reduction** (optimized with Serena semantic caching)
- **3.2-5.1x speed improvement** (accelerated with Claude Flow coordination)
- **Memory efficiency: 99.5%** (critical threshold with adaptive scaling)
- **Progressive refinement cycles**: 2-4 based on available memory

### **ARCHON + CLAUDE FLOW BENEFITS:**
```yaml
Claude Code (Execution):     32.3% token reduction + Archon streaming
Serena (Intelligence):       25% accuracy + semantic PRP integration  
Archon PRP (Refinement):     40% solution quality + progressive cycles
Claude Flow (Coordination):  2.8x speed + neural pattern optimization

Integrated Performance:      84.8% solve rate + memory-safe Archon PRP
```

## 🌟 ARCHON-ENHANCED ADVANCED FEATURES

### **PROGRESSIVE REFINEMENT WITH COORDINATION:**
- 🧠 **Archon PRP Cycles**: Memory-bounded progressive improvement
- ⚡ **Adaptive Scaling**: Dynamic agent count based on 133MB memory
- 📊 **Real-time Metrics**: Socket.IO coordination with performance tracking
- 🛡️ **Self-Healing PRP**: Auto-recovery with progressive fallbacks
- 💾 **Compressed State**: Archon state management with Claude Flow persistence

### **MEMORY-CRITICAL ARCHON CONFIGURATION:**
```yaml
Current System State (133MB available):
├─ Status: LIMITED MODE - Memory-aware Archon PRP active
├─ Agent Limit: 2-3 specialized Archon agents maximum
├─ PRP Cycles: 2 cycles with streaming operations
├─ Serena Cache: 25MB intelligent semantic analysis
├─ Auto-Recovery: Graceful degradation to single-agent Archon mode
└─ Integration: Full Archon + Serena + Claude Flow coordination
```

## 💡 ARCHON DEVELOPMENT WORKFLOW PATTERNS

### **DAILY ARCHON-ENHANCED DEVELOPMENT:**
```bash
# 1. Morning Archon System Health Check
claude-flow system-status --archon-integration --memory-alert

# 2. Initialize Archon PRP with memory awareness  
claude-flow archon-prp-init --cycles=2 --memory-limit=100MB --serena-cache=25MB

# 3. Progressive Development Session
archon prp-develop "feature-name" --socket-io-port=8052 --streaming --memory-monitor

# 4. Integrated Semantic Analysis
serena analyze-project --archon-context --memory-efficient --cache-strategy=progressive

# 5. End-of-Day Archon State Persistence
claude-flow archon-session-end --prp-export --metrics-stream --memory-recovery
```

### **EMERGENCY ARCHON PROCEDURES (Memory >99%):**
```bash
# Critical Memory: Archon single-agent mode
1. claude-flow emergency-archon --single-prp-cycle --memory-critical
2. archon prp-minimal --streaming-only --no-refinement
3. serena cache-clear --keep-essential=10MB
4. claude-flow archon-recovery --memory-optimize --progressive-restart
```

## 📚 ARCHON INTEGRATION QUICK REFERENCE

### **ARCHON COMMAND INTEGRATION:**
```bash
# Core Archon PRP commands
archon prp-status                    # Check progressive refinement status
archon prp-cycle --streaming         # Execute memory-bounded PRP cycle
archon socket-io-status --port=8052  # Check real-time coordination

# Integrated coordination commands  
claude-flow archon-init              # Initialize Archon + Claude Flow
claude-flow archon-prp --cycles=2    # Run progressive refinement
claude-flow serena-archon-sync       # Sync semantic analysis with PRP
```

### **MEMORY STATUS COMMANDS (133MB Available):**
```bash
claude-flow memory-archon            # Archon-specific memory status
serena cache-info --archon-context   # Semantic cache with PRP context
archon prp-memory --streaming-status # Progressive refinement memory usage
```

## 🎯 ULTIMATE ARCHON INTEGRATION PRINCIPLE

**"MEMORY-AWARE PROGRESSIVE REFINEMENT WITH INTELLIGENT COORDINATION"**

*Claude Code executes with Archon PRP precision, Serena provides semantic intelligence with progressive context, Archon enables systematic refinement cycles, and Claude Flow orchestrates everything within critical memory constraints for optimal progressive development.*

**Current Status: 133MB Available - LIMITED MODE WITH ARCHON PRP ACTIVE**
**Recommendation: 2-3 Archon agents, progressive streaming, memory-bounded PRP cycles**

---

# CRITICAL ARCHON INSTRUCTION REMINDERS
- **MEMORY FIRST**: Always check resources before Archon PRP operations
- **STREAM EVERYTHING**: Large Archon operations must use streaming
- **PROGRESSIVE CLEANUP**: Immediate cleanup after each PRP cycle
- **ADAPTIVE PRP**: Auto-adjust refinement cycles based on memory
- **ARCHON COORDINATION**: Use Socket.IO (8052) for real-time PRP updates

*Archon PRP integration optimized for 84.8% SWE-Bench solve rate with critical memory management and progressive refinement excellence*

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
**NEVER save working files to root folder** - use appropriate subdirectories.