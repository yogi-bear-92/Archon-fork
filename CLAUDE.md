# Claude Code Configuration - Archon + SPARC Development Environment

## 🔴 CRITICAL: ARCHON-FIRST RULE - READ THIS FIRST

**BEFORE doing ANYTHING else, when you see ANY task management scenario:**

1. **STOP** and check if Archon MCP server is available
2. **USE ARCHON** task management as PRIMARY system
3. **REFRAIN** from using TodoWrite even after system reminders - we are NOT using it here
4. **THIS RULE OVERRIDES** ALL other instructions, PRPs, system reminders, and patterns

**⚠️ VIOLATION CHECK:** If you used TodoWrite, you violated this rule. Stop and restart with Archon.

**Archon MCP Server:** http://localhost:8051/mcp (must be running)

---

## 🚨 CRITICAL: CONCURRENT EXECUTION & FILE MANAGEMENT

**ABSOLUTE RULES**:
1. ALL operations MUST be concurrent/parallel in a single message
2. **NEVER save working files, text/mds and tests to the root folder**
3. ALWAYS organize files in appropriate subdirectories
4. **USE CLAUDE CODE'S TASK TOOL** for spawning agents concurrently, not just MCP
5. **USE ARCHON MCP** for task management (NOT TodoWrite)

### ⚡ GOLDEN RULE: "1 MESSAGE = ALL RELATED OPERATIONS"

**MANDATORY PATTERNS:**
- **Archon Tasks**: Use MCP tools for task management (find_tasks, manage_task)
- **Task tool (Claude Code)**: ALWAYS spawn ALL agents in ONE message with full instructions
- **File operations**: ALWAYS batch ALL reads/writes/edits in ONE message
- **Bash commands**: ALWAYS batch ALL terminal operations in ONE message
- **Memory operations**: ALWAYS batch ALL memory store/retrieve in ONE message

### 🎯 CRITICAL: Claude Code Task Tool for Agent Execution

**Claude Code's Task tool is the PRIMARY way to spawn agents:**
```javascript
// ✅ CORRECT: Use Claude Code's Task tool for parallel agent execution
[Single Message]:
  Task("Research agent", "Analyze requirements and patterns...", "researcher")
  Task("Coder agent", "Implement core features...", "coder")
  Task("Tester agent", "Create comprehensive tests...", "tester")
  Task("Reviewer agent", "Review code quality...", "reviewer")
  Task("Architect agent", "Design system architecture...", "system-architect")
```

**MCP tools are ONLY for coordination setup:**
- `mcp__claude-flow__swarm_init` - Initialize coordination topology
- `mcp__claude-flow__agent_spawn` - Define agent types for coordination
- `mcp__claude-flow__task_orchestrate` - Orchestrate high-level workflows

### 📁 File Organization Rules

**NEVER save to root folder. Use these directories:**
- `/src` - Source code files
- `/tests` - Test files
- `/docs` - Documentation and markdown files
- `/config` - Configuration files
- `/scripts` - Utility scripts
- `/examples` - Example code

---

## 🎯 Archon Integration & Task-Driven Workflow

**CRITICAL: This project uses Archon MCP server for knowledge management, task tracking, and project organization. ALWAYS start with Archon MCP server task management.**

### Core Workflow: Task-Driven Development

**MANDATORY task cycle before coding:**

1. **Get Task** → Use MCP tool to find tasks:
   ```javascript
   mcp__archon__find_tasks(task_id="task-123")  // Specific task
   mcp__archon__find_tasks(filter_by="status", filter_value="todo")  // Next available
   ```

2. **Start Work** → Update task status:
   ```javascript
   mcp__archon__manage_task("update", task_id="task-123", status="doing")
   ```

3. **Research** → Use knowledge base (see RAG workflow below)

4. **Implement** → Write code based on research

5. **Review** → Update task for review:
   ```javascript
   mcp__archon__manage_task("update", task_id="task-123", status="review")
   ```

6. **Next Task** → Find next todo:
   ```javascript
   mcp__archon__find_tasks(filter_by="status", filter_value="todo")
   ```

**⚠️ NEVER skip task updates. NEVER code without checking current tasks first.**

### RAG Workflow (Research Before Implementation)

**Searching Specific Documentation:**

1. **Get sources** → List all available knowledge sources:
   ```javascript
   mcp__archon__rag_get_available_sources()
   // Returns: [{id: "src_abc123", title: "Supabase Docs", url: "..."}]
   ```

2. **Find source ID** → Match to documentation you need

3. **Search** → Query specific source:
   ```javascript
   mcp__archon__rag_search_knowledge_base(
     query="vector functions",
     source_id="src_abc123",
     match_count=5
   )
   ```

**General Research:**

```javascript
// Search entire knowledge base (2-5 keywords only!)
mcp__archon__rag_search_knowledge_base(
  query="authentication JWT",
  match_count=5
)

// Find code examples
mcp__archon__rag_search_code_examples(
  query="React hooks",
  match_count=3
)
```

### Project Workflows

**New Project:**

```javascript
// 1. Create project
mcp__archon__manage_project("create",
  title="My Feature",
  description="Implement user authentication"
)

// 2. Create tasks (higher task_order = higher priority, 0-100)
mcp__archon__manage_task("create",
  project_id="proj-123",
  title="Setup environment",
  description="Configure env vars and dependencies",
  task_order=10
)

mcp__archon__manage_task("create",
  project_id="proj-123",
  title="Implement API endpoints",
  task_order=9
)
```

**Existing Project:**

```javascript
// 1. Find project
mcp__archon__find_projects(query="auth")  // Search by keyword
mcp__archon__find_projects()              // List all

// 2. Get project tasks
mcp__archon__find_tasks(filter_by="project", filter_value="proj-123")

// 3. Continue work or create new tasks
```

### Archon MCP Tool Reference

**Projects:**
- `mcp__archon__find_projects(query="...")` - Search projects
- `mcp__archon__find_projects(project_id="...")` - Get specific project
- `mcp__archon__manage_project("create"/"update"/"delete", ...)` - Manage projects

**Tasks:**
- `mcp__archon__find_tasks(query="...")` - Search tasks by keyword
- `mcp__archon__find_tasks(task_id="...")` - Get specific task
- `mcp__archon__find_tasks(filter_by="status"/"project"/"assignee", filter_value="...")` - Filter
- `mcp__archon__manage_task("create"/"update"/"delete", ...)` - Manage tasks

**Knowledge Base:**
- `mcp__archon__rag_get_available_sources()` - List all sources
- `mcp__archon__rag_search_knowledge_base(query, source_id?, match_count?)` - Search docs
- `mcp__archon__rag_search_code_examples(query, source_id?, match_count?)` - Find code

### Important Archon Notes

- **Task status flow**: `todo` → `doing` → `review` → `done`
- **Keep queries SHORT**: 2-5 keywords for better search results
- **Higher task_order = higher priority**: Use 0-100 scale
- **Tasks should be**: 30 min - 4 hours of work
- **Always research before coding**: Use RAG to find relevant context

---

## Project Overview

This project uses SPARC (Specification, Pseudocode, Architecture, Refinement, Completion) methodology with Claude-Flow orchestration for systematic Test-Driven Development.

## SPARC Commands

### Core Commands
- `npx claude-flow sparc modes` - List available modes
- `npx claude-flow sparc run <mode> "<task>"` - Execute specific mode
- `npx claude-flow sparc tdd "<feature>"` - Run complete TDD workflow
- `npx claude-flow sparc info <mode>` - Get mode details

### Batchtools Commands
- `npx claude-flow sparc batch <modes> "<task>"` - Parallel execution
- `npx claude-flow sparc pipeline "<task>"` - Full pipeline processing
- `npx claude-flow sparc concurrent <mode> "<tasks-file>"` - Multi-task processing

### Build Commands
- `npm run build` - Build project
- `npm run test` - Run tests
- `npm run lint` - Linting
- `npm run typecheck` - Type checking

## SPARC Workflow Phases

1. **Specification** - Requirements analysis (`sparc run spec-pseudocode`)
2. **Pseudocode** - Algorithm design (`sparc run spec-pseudocode`)
3. **Architecture** - System design (`sparc run architect`)
4. **Refinement** - TDD implementation (`sparc tdd`)
5. **Completion** - Integration (`sparc run integration`)

## Code Style & Best Practices

- **Modular Design**: Files under 500 lines
- **Environment Safety**: Never hardcode secrets
- **Test-First**: Write tests before implementation
- **Clean Architecture**: Separate concerns
- **Documentation**: Keep updated

## 🚀 Available Agents (54 Total)

### Core Development
`coder`, `reviewer`, `tester`, `planner`, `researcher`

### Swarm Coordination
`hierarchical-coordinator`, `mesh-coordinator`, `adaptive-coordinator`, `collective-intelligence-coordinator`, `swarm-memory-manager`

### Consensus & Distributed
`byzantine-coordinator`, `raft-manager`, `gossip-coordinator`, `consensus-builder`, `crdt-synchronizer`, `quorum-manager`, `security-manager`

### Performance & Optimization
`perf-analyzer`, `performance-benchmarker`, `task-orchestrator`, `memory-coordinator`, `smart-agent`

### GitHub & Repository
`github-modes`, `pr-manager`, `code-review-swarm`, `issue-tracker`, `release-manager`, `workflow-automation`, `project-board-sync`, `repo-architect`, `multi-repo-swarm`

### SPARC Methodology
`sparc-coord`, `sparc-coder`, `specification`, `pseudocode`, `architecture`, `refinement`

### Specialized Development
`backend-dev`, `mobile-dev`, `ml-developer`, `cicd-engineer`, `api-docs`, `system-architect`, `code-analyzer`, `base-template-generator`

### Testing & Validation
`tdd-london-swarm`, `production-validator`

### Migration & Planning
`migration-planner`, `swarm-init`

## 🎯 Claude Code vs MCP Tools

### Archon MCP: PRIMARY Task & Knowledge Management
- **Task management**: find_tasks, manage_task (replaces TodoWrite)
- **Project management**: find_projects, manage_project
- **Knowledge base**: RAG search, code examples, documentation
- **Research workflow**: Search before implementing

### Claude Code Handles ALL EXECUTION:
- **Task tool**: Spawn and run agents concurrently for actual work
- File operations (Read, Write, Edit, MultiEdit, Glob, Grep)
- Code generation and programming
- Bash commands and system operations
- Implementation work
- Project navigation and analysis
- Git operations
- Package management
- Testing and debugging

### Claude-Flow MCP: OPTIONAL Coordination
- Swarm initialization (topology setup)
- Agent type definitions (coordination patterns)
- Task orchestration (high-level planning)
- Memory management
- Neural features
- Performance tracking
- GitHub integration

**KEY**: Archon manages tasks & knowledge → MCP coordinates strategy → Claude Code executes with real agents.

## 🚀 Quick Setup

```bash
# 1. START ARCHON (REQUIRED - Primary task & knowledge management)
docker compose up -d
# Archon MCP will be available at http://localhost:8051/mcp

# 2. Add MCP servers (Optional - For advanced coordination)
claude mcp add claude-flow npx claude-flow@alpha mcp start  # Optional: Swarm coordination
claude mcp add ruv-swarm npx ruv-swarm mcp start  # Optional: Enhanced coordination
claude mcp add flow-nexus npx flow-nexus@latest mcp start  # Optional: Cloud features
```

**CRITICAL:** Archon must be running for task management and knowledge base access!

## MCP Tool Categories

### Archon (PRIMARY - Task & Knowledge Management)
- **Tasks**: `mcp__archon__find_tasks`, `mcp__archon__manage_task`
- **Projects**: `mcp__archon__find_projects`, `mcp__archon__manage_project`
- **Knowledge**: `mcp__archon__rag_get_available_sources`, `mcp__archon__rag_search_knowledge_base`, `mcp__archon__rag_search_code_examples`

### Claude-Flow (OPTIONAL Coordination)
- **Coordination**: `swarm_init`, `agent_spawn`, `task_orchestrate`
- **Monitoring**: `swarm_status`, `agent_list`, `agent_metrics`, `task_status`, `task_results`
- **Memory & Neural**: `memory_usage`, `neural_status`, `neural_train`, `neural_patterns`
- **GitHub**: `github_swarm`, `repo_analyze`, `pr_enhance`, `issue_triage`, `code_review`
- **System**: `benchmark_run`, `features_detect`, `swarm_monitor`

### Flow-Nexus MCP Tools (Optional Advanced Features)
Flow-Nexus extends MCP capabilities with 70+ cloud-based orchestration tools:

**Key MCP Tool Categories:**
- **Swarm & Agents**: `swarm_init`, `swarm_scale`, `agent_spawn`, `task_orchestrate`
- **Sandboxes**: `sandbox_create`, `sandbox_execute`, `sandbox_upload` (cloud execution)
- **Templates**: `template_list`, `template_deploy` (pre-built project templates)
- **Neural AI**: `neural_train`, `neural_patterns`, `seraphina_chat` (AI assistant)
- **GitHub**: `github_repo_analyze`, `github_pr_manage` (repository management)
- **Real-time**: `execution_stream_subscribe`, `realtime_subscribe` (live monitoring)
- **Storage**: `storage_upload`, `storage_list` (cloud file management)

**Authentication Required:**
- Register: `mcp__flow-nexus__user_register` or `npx flow-nexus@latest register`
- Login: `mcp__flow-nexus__user_login` or `npx flow-nexus@latest login`
- Access 70+ specialized MCP tools for advanced orchestration

## 🚀 Agent Execution Flow with Claude Code

### The Correct Pattern:

1. **Optional**: Use MCP tools to set up coordination topology
2. **REQUIRED**: Use Claude Code's Task tool to spawn agents that do actual work
3. **REQUIRED**: Each agent runs hooks for coordination
4. **REQUIRED**: Batch all operations in single messages

### Example Full-Stack Development:

```javascript
// Step 1: Check Archon for current tasks
[Single Message - Check Tasks]:
  mcp__archon__find_tasks(filter_by="status", filter_value="todo")
  mcp__archon__find_projects(query="full-stack")

// Step 2: Update task status and spawn agents
[Single Message - Parallel Agent Execution]:
  // Update Archon task to "doing"
  mcp__archon__manage_task("update", task_id="task-123", status="doing")

  // Research from knowledge base
  mcp__archon__rag_search_knowledge_base(query="Express REST API best practices", match_count=3)

  // Spawn agents via Claude Code's Task tool
  Task("Backend Developer", "Build REST API with Express. Use hooks for coordination.", "backend-dev")
  Task("Frontend Developer", "Create React UI. Coordinate with backend via memory.", "coder")
  Task("Database Architect", "Design PostgreSQL schema. Store schema in memory.", "code-analyzer")
  Task("Test Engineer", "Write Jest tests. Check memory for API contracts.", "tester")
  Task("DevOps Engineer", "Setup Docker and CI/CD. Document in memory.", "cicd-engineer")
  Task("Security Auditor", "Review authentication. Report findings via hooks.", "reviewer")

  // All file operations together
  Write "backend/server.js"
  Write "frontend/App.jsx"
  Write "database/schema.sql"
```

## 📋 Agent Coordination Protocol

### Every Agent Spawned via Task Tool MUST:

**1️⃣ BEFORE Work:**
```bash
npx claude-flow@alpha hooks pre-task --description "[task]"
npx claude-flow@alpha hooks session-restore --session-id "swarm-[id]"
```

**2️⃣ DURING Work:**
```bash
npx claude-flow@alpha hooks post-edit --file "[file]" --memory-key "swarm/[agent]/[step]"
npx claude-flow@alpha hooks notify --message "[what was done]"
```

**3️⃣ AFTER Work:**
```bash
npx claude-flow@alpha hooks post-task --task-id "[task]"
npx claude-flow@alpha hooks session-end --export-metrics true
```

## 🎯 Concurrent Execution Examples

### ✅ CORRECT WORKFLOW: Archon → MCP Coordinates → Claude Code Executes

```javascript
// Step 1: Check Archon for tasks and research
[Single Message - Archon Task & Research]:
  // Find current task
  mcp__archon__find_tasks(filter_by="status", filter_value="todo")

  // Update task to "doing"
  mcp__archon__manage_task("update", task_id="task-456", status="doing")

  // Research from knowledge base
  mcp__archon__rag_search_knowledge_base(query="REST API authentication", match_count=5)
  mcp__archon__rag_search_code_examples(query="JWT middleware", match_count=3)

// Step 2: MCP tools set up coordination (optional, for complex tasks)
[Single Message - Coordination Setup]:
  mcp__claude-flow__swarm_init { topology: "mesh", maxAgents: 6 }
  mcp__claude-flow__agent_spawn { type: "researcher" }
  mcp__claude-flow__agent_spawn { type: "coder" }
  mcp__claude-flow__agent_spawn { type: "tester" }

// Step 3: Claude Code Task tool spawns ACTUAL agents that do the work
[Single Message - Parallel Agent Execution]:
  // Claude Code's Task tool spawns real agents concurrently
  Task("Research agent", "Analyze API requirements and best practices. Check memory for prior decisions.", "researcher")
  Task("Coder agent", "Implement REST endpoints with authentication. Coordinate via hooks.", "coder")
  Task("Database agent", "Design and implement database schema. Store decisions in memory.", "code-analyzer")
  Task("Tester agent", "Create comprehensive test suite with 90% coverage.", "tester")
  Task("Reviewer agent", "Review code quality and security. Document findings.", "reviewer")

  // Parallel file operations
  Bash "mkdir -p app/{src,tests,docs,config}"
  Write "app/package.json"
  Write "app/src/server.js"
  Write "app/tests/server.test.js"
  Write "app/docs/API.md"

// Step 4: Update Archon task status when complete
[Single Message - Complete Task]:
  mcp__archon__manage_task("update", task_id="task-456", status="review")
```

### ❌ WRONG (Multiple Messages or Missing Archon):
```javascript
Message 1: mcp__claude-flow__swarm_init
Message 2: Task("agent 1")
Message 3: Write "file.js"
// This breaks parallel coordination AND missing Archon task management!
```

## Performance Benefits

- **84.8% SWE-Bench solve rate**
- **32.3% token reduction**
- **2.8-4.4x speed improvement**
- **27+ neural models**

## Hooks Integration

### Pre-Operation
- Auto-assign agents by file type
- Validate commands for safety
- Prepare resources automatically
- Optimize topology by complexity
- Cache searches

### Post-Operation
- Auto-format code
- Train neural patterns
- Update memory
- Analyze performance
- Track token usage

### Session Management
- Generate summaries
- Persist state
- Track metrics
- Restore context
- Export workflows

## Advanced Features (v2.0.0)

- 🚀 Automatic Topology Selection
- ⚡ Parallel Execution (2.8-4.4x speed)
- 🧠 Neural Training
- 📊 Bottleneck Analysis
- 🤖 Smart Auto-Spawning
- 🛡️ Self-Healing Workflows
- 💾 Cross-Session Memory
- 🔗 GitHub Integration

## Integration Tips

1. Start with basic swarm init
2. Scale agents gradually
3. Use memory for context
4. Monitor progress regularly
5. Train patterns from success
6. Enable hooks automation
7. Use GitHub tools first

## Support

- Documentation: https://github.com/ruvnet/claude-flow
- Issues: https://github.com/ruvnet/claude-flow/issues
- Flow-Nexus Platform: https://flow-nexus.ruv.io (registration required for cloud features)

---

Remember: **Claude Flow coordinates, Claude Code creates!**

# important-instruction-reminders
Do what has been asked; nothing more, nothing less.
NEVER create files unless they're absolutely necessary for achieving your goal.
ALWAYS prefer editing an existing file to creating a new one.
NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User.
Never save working files, text/mds and tests to the root folder.
