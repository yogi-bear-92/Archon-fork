# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Archon Development Guidelines

**Local-only deployment** - each user runs their own instance with Archon PRP integration.

### Core Principles

- **Archon PRP Integration**: Progressive refinement with automated optimization
- **Claude Flow Auto-Scaling**: Let Claude Flow handle all memory and resource management automatically
- **Concurrent execution**: ALL operations MUST be concurrent/parallel in a single message
- **No backwards compatibility; we follow a fix‑forward approach** — remove deprecated code immediately
- **Detailed errors over graceful failures** - we want to identify and fix issues fast
- **Break things to improve them** - beta is for rapid iteration
- **Continuous improvement** - embrace change and learn from mistakes
- **KISS** - keep it simple
- **DRY** when appropriate
- **YAGNI** — don't implement features that are not needed
- **NEVER save working files to root folder** - use proper subdirectories
- **USE CLAUDE CODE'S TASK TOOL** for spawning agents concurrently with Archon coordination

### Error Handling

**Core Principle**: In beta, we need to intelligently decide when to fail hard and fast to quickly address issues, and when to allow processes to complete in critical services despite failures. Read below carefully and make intelligent decisions on a case-by-case basis.

#### When to Fail Fast and Loud (Let it Crash!)

These errors should stop execution and bubble up immediately: (except for crawling flows)

- **Service startup failures** - If credentials, database, or any service can't initialize, the system should crash with a clear error
- **Missing configuration** - Missing environment variables or invalid settings should stop the system
- **Database connection failures** - Don't hide connection issues, expose them
- **Authentication/authorization failures** - Security errors must be visible and halt the operation
- **Data corruption or validation errors** - Never silently accept bad data, Pydantic should raise
- **Critical dependencies unavailable** - If a required service is down, fail immediately
- **Invalid data that would corrupt state** - Never store zero embeddings, null foreign keys, or malformed JSON

#### When to Complete but Log Detailed Errors

These operations should continue but track and report failures clearly:

- **Batch processing** - When crawling websites or processing documents, complete what you can and report detailed failures for each item
- **Background tasks** - Embedding generation, async jobs should finish the queue but log failures
- **WebSocket events** - Don't crash on a single event failure, log it and continue serving other clients
- **Optional features** - If projects/tasks are disabled, log and skip rather than crash
- **External API calls** - Retry with exponential backoff, then fail with a clear message about what service failed and why

#### Critical Nuance: Never Accept Corrupted Data

When a process should continue despite failures, it must **skip the failed item entirely** rather than storing corrupted data

#### Error Message Guidelines

- **Include relevant context**: User ID, project ID, file path, database table, external service name
- **Describe the failure clearly**: "Failed to generate embeddings for document XYZ" not just "OpenAI API error"
- **Include the underlying error**: The full exception chain, not just a generic message
- **Suggest next steps**: What the user or system should do to resolve this

### Archon Integration Patterns

**MANDATORY ARCHON-OPTIMIZED PATTERNS:**
- **TodoWrite**: Batch all todos efficiently (Claude Flow handles sizing)
- **Task tool**: Spawn ALL agents in ONE message with Archon coordination hooks
- **File operations**: Standard operations with Claude Flow optimization
- **Archon Integration**: Use PRP cycles with semantic analysis and progressive refinement
- **Serena Coordination**: Semantic analysis with intelligent caching

### File Organization

**Archon-specific directory structure:**
- `/src` - Source code with Archon PRP structure
- `/tests` - Archon validation tests
- `/docs` - Progressive refinement documentation
- `/config` - Archon and Serena configurations
- `/scripts` - Archon PRP automation scripts
- `/.archon-prp` - Progressive refinement data
- `/.serena-cache` - Semantic analysis cache
- `/.claude-flow` - Coordination metrics

### Git Integration

Enable Claude Code + Archon git checkpoints:
```bash
git config --local claude-code.auto-checkpoint true
git config --local claude-code.checkpoint-frequency "after-prp-cycles"
git config --local claude-code.checkpoint-message-template "🔄 Archon-checkpoint: {prp-phase}"
```

### Development Workflow

1. **Initialize Archon PRP with Claude Flow auto-scaling**
2. **Spawn agents via Claude Code with Archon coordination**
3. **Progressive refinement cycles with Claude Flow optimization**
4. **Real-time coordination with Socket.IO (port 8052)**
5. **Auto-scaling resource management**

### Important Reminders

- Do what has been asked; nothing more, nothing less
- NEVER create files unless they're absolutely necessary for achieving your goal
- ALWAYS prefer editing an existing file to creating a new one
- NEVER proactively create documentation files (*.md) or README files. Only create documentation files if explicitly requested by the User
- **NEVER save working files to root folder** - use appropriate subdirectories