# Daily Workflow Patterns: Optimized Serena + Archon + Claude Flow Integration

## Overview

This document provides detailed daily workflow patterns designed to maximize productivity while maintaining optimal system performance and memory efficiency. Based on the memory optimization improvements (showing 134-251MB free vs previous 69-90MB), these patterns ensure sustainable development practices.

## 🌅 Morning Startup Workflow

### System Health Check and Initialization

#### Pre-Development System Assessment
```bash
#!/bin/bash
# Morning system health check script
echo "=== MORNING SYSTEM HEALTH CHECK ==="

# Check overnight memory patterns
npx claude-flow metrics --memory-trend --hours=12

# Analyze system resource availability
npx archon system-check --memory --cpu --disk --detailed

# Clean up overnight artifacts
rm -rf /tmp/archon-* /tmp/serena-* /tmp/claude-flow-*
find . -name "*.log" -mtime +1 -delete

# Memory optimization check
MEMORY_USAGE=$(node -e "
  const usage = process.memoryUsage();
  console.log((usage.heapUsed / 1024 / 1024).toFixed(2));
")

if (( $(echo "$MEMORY_USAGE > 1024" | bc -l) )); then
  echo "⚠️  High memory usage detected: ${MEMORY_USAGE}MB"
  echo "Applying memory optimization..."
  npx archon optimize --aggressive-cleanup
fi

echo "✅ System health check complete"
```

#### Optimized Service Startup Sequence
```javascript
// Morning initialization with memory awareness
[Morning Setup - Single Message]:
  // Check system resources first
  Bash("bash scripts/morning-health-check.sh")
  
  // Start services in optimal order with memory limits
  Bash("export NODE_OPTIONS='--max-old-space-size=1536' && npx archon start --profile=morning")
  
  // Initialize Serena with cached semantic analysis
  mcp__serena__initialize({
    project_path: process.cwd(),
    cache_mode: "warm_start",
    memory_limit: "512MB",
    semantic_depth: "incremental"
  })
  
  // Load Archon knowledge base with yesterday's context
  mcp__archon__session_restore({
    session_type: "daily_continuation",
    context_window: "24h",
    memory_mode: "streaming"
  })
  
  // Configure Claude Flow with morning topology
  mcp__claude-flow__swarm_init({
    topology: "morning_hierarchical",
    maxAgents: 3,
    memoryBudget: "1GB",
    optimization: "memory_first"
  })
  
  // Setup daily tracking
  TodoWrite([
    {id: "morning-1", content: "System health verified", status: "completed", priority: "high"},
    {id: "morning-2", content: "Services started with memory optimization", status: "completed", priority: "high"},
    {id: "morning-3", content: "Daily context loaded", status: "completed", priority: "medium"},
    {id: "morning-4", content: "Ready for development tasks", status: "completed", priority: "low"}
  ])
```

## 🏗️ Core Development Workflows

### Pattern 1: Feature Development (Memory-Optimized)

#### Phase 1: Analysis and Planning
```javascript
// Lightweight analysis phase to minimize memory usage
[Analysis Phase - Memory Conscious]:
  // Use cached semantic analysis when possible
  Task("Requirements Analyst", `
    Analyze feature requirements using Archon knowledge base.
    Priority: Use cached analysis from similar features.
    Query patterns: Focus on architectural implications.
    Memory budget: 128MB maximum.
    
    Coordination hooks:
    - npx claude-flow hooks pre-task --description "requirements-analysis"
    - npx serena cache-warm --feature-context
    
    Output: Structured requirements with memory-efficient format.
  `, "researcher")
  
  // Minimal memory footprint for system design
  Task("System Designer", `
    Design system components using Serena code intelligence.
    Priority: Leverage existing architectural patterns.
    Focus: Integration points with current codebase.
    Memory budget: 256MB maximum.
    
    Use Serena for:
    - Existing code pattern analysis
    - API compatibility checking
    - Integration point identification
    
    Memory optimization:
    - Stream large code analysis results
    - Use compressed cache format
    - Lazy-load detailed semantic data
  `, "system-architect")
```

#### Phase 2: Progressive Implementation
```javascript
// Implementation with progressive refinement
[Implementation Phase - Progressive]:
  // Core implementation with memory pooling
  Task("Core Developer", `
    Implement feature using Archon PRP methodology.
    
    Progressive Refinement Cycles (4 iterations):
    1. Basic structure implementation (256MB)
    2. Core functionality (512MB)
    3. Integration and error handling (384MB) 
    4. Optimization and polish (256MB)
    
    Memory management:
    - Release memory between refinement cycles
    - Use streaming for large data processing
    - Cache only essential context
    
    Coordination:
    - Store progress in Claude Flow memory after each cycle
    - Update Archon project status
    - Share intermediate results via memory-efficient formats
    
    Hooks sequence:
    npx claude-flow hooks pre-task --description "core-implementation"
    # Between each cycle:
    npx claude-flow hooks post-edit --memory-release
    npx archon update-progress --cycle=N --memory-optimize
  `, "backend-dev")
  
  // Parallel testing with resource constraints
  Task("Test Engineer", `
    Create comprehensive test suite with memory awareness.
    
    Testing strategy:
    - Unit tests: 128MB budget (isolated testing)
    - Integration tests: 256MB budget (controlled environment)
    - End-to-end tests: 512MB budget (full system)
    
    Memory optimization:
    - Test data streaming instead of full loading
    - Parallel test execution with memory pooling
    - Clean test artifacts after each suite
    
    Coordination:
    - Share test results via compressed format
    - Update coverage metrics in real-time
    - Alert on memory usage spikes during testing
  `, "tester")
```

#### Phase 3: Review and Integration
```javascript
// Memory-efficient review process
[Review Phase - Coordinated]:
  Task("Code Quality Reviewer", `
    Comprehensive code review using integrated tool intelligence.
    
    Review priorities (memory-optimized):
    1. Memory efficiency patterns (Serena semantic analysis)
    2. Archon pattern compliance (knowledge base queries)
    3. Performance implications (Claude Flow metrics)
    4. Integration compatibility (cached analysis)
    
    Memory management:
    - Review in focused chunks (256MB per file group)
    - Use cached semantic analysis where possible
    - Stream large codebases instead of loading entirely
    
    Coordination protocol:
    - Real-time findings sharing via memory-efficient protocol
    - Progressive review with checkpoint saves
    - Memory pressure alerts during deep analysis
  `, "reviewer")
  
  // Integration management
  Task("Integration Coordinator", `
    Coordinate feature integration with system-wide awareness.
    
    Integration checklist:
    - Memory impact analysis of new feature
    - Performance regression testing
    - Cross-system compatibility verification
    - Resource usage optimization
    
    Memory budget: 384MB for coordination tasks
    
    Use all three systems:
    - Serena: Code compatibility analysis
    - Archon: Integration pattern matching
    - Claude Flow: Multi-system coordination
  `, "integration-manager")
```

### Pattern 2: Code Maintenance and Optimization

#### System Health Monitoring
```javascript
// Daily system health and optimization
[Maintenance Workflow - System Health]:
  // Automated health analysis
  Task("System Health Monitor", `
    Perform comprehensive system health analysis.
    
    Health check priorities:
    1. Memory usage patterns and trends
    2. Performance bottlenecks identification
    3. Resource allocation efficiency
    4. Inter-system coordination health
    
    Analysis tools:
    - Memory trend analysis from system metrics
    - Performance profiling with minimal overhead
    - Cache hit ratio optimization
    - Garbage collection efficiency review
    
    Memory budget: 256MB (use streaming analysis)
    
    Output: System health report with optimization recommendations
  `, "system-monitor")
  
  // Performance optimization
  Task("Performance Optimizer", `
    Identify and resolve performance bottlenecks.
    
    Optimization targets:
    - Memory allocation patterns
    - Cache efficiency improvement
    - Agent coordination latency reduction
    - Resource pool optimization
    
    Use integrated metrics from:
    - Claude Flow: Agent coordination performance
    - Archon: Knowledge base query efficiency  
    - Serena: Semantic analysis cache performance
    
    Memory management:
    - Profile memory usage during optimization
    - Test changes with controlled memory budgets
    - Verify optimization doesn't increase memory usage
  `, "performance-engineer")
```

#### Code Refactoring Workflow
```javascript
// Memory-aware refactoring process
[Refactoring Workflow - Progressive]:
  Task("Refactoring Analyst", `
    Analyze codebase for refactoring opportunities.
    
    Analysis scope (memory-optimized):
    - Technical debt identification (use cached analysis)
    - Memory efficiency improvement opportunities
    - Code complexity reduction potential
    - Integration pattern optimization
    
    Serena integration:
    - Semantic analysis of code relationships
    - Symbol usage tracking for safe refactoring
    - Impact analysis with memory-efficient scanning
    
    Memory budget: 384MB for large codebase analysis
    
    Strategy: Process codebase in memory-managed chunks
  `, "code-analyzer")
  
  Task("Refactoring Implementer", `
    Implement refactoring changes using progressive approach.
    
    Refactoring methodology:
    1. Small, isolated changes first (128MB budget)
    2. Module-level refactoring (256MB budget)
    3. Cross-module integration (512MB budget)
    4. System-wide optimization (384MB budget)
    
    Safety measures:
    - Comprehensive test coverage before changes
    - Incremental commits with rollback capability
    - Memory usage monitoring during refactoring
    - Performance impact validation
    
    Archon integration:
    - Use knowledge base for refactoring patterns
    - Progressive refinement of changes
    - Documentation updates in parallel
  `, "refactoring-expert")
```

## 📊 Memory-Aware Task Patterns

### Dynamic Memory Allocation

#### Memory Budget Calculation
```javascript
// Dynamic memory budget based on system state
const calculateOptimalMemoryBudget = async (taskType, systemState) => {
  const baseMemoryBudgets = {
    'analysis': 256,      // MB
    'implementation': 512,
    'testing': 384,
    'review': 256,
    'optimization': 384,
    'coordination': 128
  };
  
  const availableMemory = systemState.memoryFree;
  const basebudget = baseMemoryBudgets[taskType] || 256;
  
  // Adaptive scaling based on available memory
  if (availableMemory < 500 * 1024 * 1024) { // Less than 500MB
    return Math.min(basebudget, 128); // Emergency mode
  } else if (availableMemory < 1024 * 1024 * 1024) { // Less than 1GB
    return Math.min(basebudget, 256); // Conservative mode
  } else if (availableMemory > 2048 * 1024 * 1024) { // More than 2GB
    return Math.min(basebudget * 1.5, 768); // Optimal mode
  }
  
  return basebudget; // Normal mode
};
```

#### Task Queue Management
```javascript
// Memory-aware task queueing
class MemoryAwareTaskQueue {
  constructor() {
    this.queue = [];
    this.runningTasks = new Map();
    this.maxConcurrentMemory = 2048; // MB
  }
  
  async addTask(task, memoryRequirement) {
    const queueItem = {
      task,
      memoryRequirement,
      priority: task.priority || 'medium',
      timestamp: Date.now()
    };
    
    // Check if we can run immediately
    const currentMemoryUsage = this.getCurrentMemoryUsage();
    
    if (currentMemoryUsage + memoryRequirement <= this.maxConcurrentMemory) {
      return await this.executeTask(queueItem);
    }
    
    // Queue for later execution
    this.queue.push(queueItem);
    this.queue.sort((a, b) => {
      // Sort by priority, then by memory efficiency
      const priorityWeight = { 'high': 3, 'medium': 2, 'low': 1 };
      const aPriority = priorityWeight[a.priority] || 1;
      const bPriority = priorityWeight[b.priority] || 1;
      
      if (aPriority !== bPriority) return bPriority - aPriority;
      
      // If same priority, prefer lower memory requirements
      return a.memoryRequirement - b.memoryRequirement;
    });
  }
  
  async processQueue() {
    while (this.queue.length > 0) {
      const availableMemory = this.getAvailableMemory();
      const nextTask = this.queue.find(task => 
        task.memoryRequirement <= availableMemory
      );
      
      if (!nextTask) {
        // Wait for memory to free up
        await this.waitForMemoryAvailable();
        continue;
      }
      
      // Remove from queue and execute
      this.queue = this.queue.filter(task => task !== nextTask);
      await this.executeTask(nextTask);
    }
  }
}
```

### Coordination Patterns by Memory Pressure

#### High Memory Pressure (>90% usage)
```javascript
// Emergency coordination pattern
[Emergency Memory Mode]:
  // Only one agent at a time
  // Minimal coordination overhead
  // Direct communication only
  
  const emergencyCoordination = {
    maxConcurrentAgents: 1,
    coordinationProtocol: 'direct',
    cacheSize: '64MB',
    semanticDepth: 'minimal',
    memoryMonitoring: 'aggressive'
  };
  
  // Example emergency task
  Task("Emergency Implementer", `
    Critical bug fix with minimal memory usage.
    
    Constraints:
    - Single-threaded execution
    - No semantic caching
    - Direct file operations only
    - Minimal coordination overhead
    
    Memory limit: 128MB hard limit
    
    Focus: Essential functionality only
    Emergency protocols: Fail fast, no retry logic
  `, "emergency-coder")
```

#### Medium Memory Pressure (75-90% usage)
```javascript
// Conservative coordination pattern
[Conservative Memory Mode]:
  // Limited concurrent agents
  // Efficient coordination
  // Selective caching
  
  const conservativeCoordination = {
    maxConcurrentAgents: 2,
    coordinationProtocol: 'hierarchical',
    cacheSize: '256MB',
    semanticDepth: 'balanced',
    memoryMonitoring: 'active'
  };
```

#### Low Memory Pressure (<75% usage)
```javascript
// Optimal coordination pattern
[Optimal Memory Mode]:
  // Full agent coordination
  // Rich semantic analysis
  // Comprehensive caching
  
  const optimalCoordination = {
    maxConcurrentAgents: 5,
    coordinationProtocol: 'intelligent_mesh',
    cacheSize: '1GB',
    semanticDepth: 'comprehensive',
    memoryMonitoring: 'baseline'
  };
```

## 🕐 Time-Based Workflow Optimization

### Hourly Optimization Cycles

#### Memory Cleanup Cycle (Every Hour)
```bash
#!/bin/bash
# Hourly memory optimization
npx claude-flow hooks memory-cleanup --gentle
npx serena cache-optimize --incremental
npx archon gc-trigger --background
```

#### Performance Review Cycle (Every 2 Hours)
```javascript
// Performance check and optimization
[Bi-Hourly Performance Review]:
  // Lightweight performance analysis
  mcp__claude-flow__swarm_status({
    include_memory_metrics: true,
    performance_analysis: true,
    optimization_suggestions: true
  })
  
  // Memory trend analysis
  mcp__archon__performance_metrics({
    timeframe: "2h",
    focus: "memory_efficiency",
    recommendations: true
  })
```

### End-of-Day Cleanup

#### Comprehensive System Optimization
```javascript
// End-of-day optimization workflow
[End-of-Day Cleanup]:
  // Export daily metrics
  mcp__claude-flow__session_end({
    export_metrics: true,
    compress_logs: true,
    cleanup_temp_files: true
  })
  
  // Optimize knowledge base
  mcp__archon__daily_optimization({
    compress_embeddings: true,
    cleanup_cache: true,
    export_session_summary: true
  })
  
  // Serena cache optimization
  mcp__serena__end_of_day({
    persist_semantic_cache: true,
    cleanup_temp_analysis: true,
    export_code_metrics: true
  })
  
  // System resource cleanup
  Bash(`
    # Clean temporary files
    find /tmp -name "*archon*" -o -name "*serena*" -o -name "*claude-flow*" -mtime +0 -delete
    
    # Force garbage collection
    pkill -USR2 node 2>/dev/null || true
    
    # Compress logs
    gzip ~/.cache/archon/logs/*.log 2>/dev/null || true
    
    # Update system status
    echo "$(date): Daily cleanup completed" >> ~/.cache/archon/daily-status.log
  `)
  
  // Final status update
  TodoWrite([
    {id: "eod-1", content: "Daily metrics exported", status: "completed", priority: "medium"},
    {id: "eod-2", content: "Memory optimization completed", status: "completed", priority: "high"},
    {id: "eod-3", content: "Cache cleanup finished", status: "completed", priority: "medium"},
    {id: "eod-4", content: "System ready for next day", status: "completed", priority: "low"}
  ])
```

## 📈 Productivity Optimization Patterns

### Context Switching Optimization

#### Project Switching with Memory Management
```javascript
// Efficient project context switching
const switchProject = async (fromProject, toProject) => {
  // Save current context with compression
  await mcp__archon__save_project_context({
    project: fromProject,
    compression: true,
    persistent: true
  });
  
  // Clear memory-intensive caches
  await mcp__serena__clear_project_cache({
    project: fromProject,
    keep_essential: true
  });
  
  // Load new project context gradually
  await mcp__archon__load_project_context({
    project: toProject,
    lazy_loading: true,
    memory_limit: "512MB"
  });
  
  // Warm up essential caches only
  await mcp__serena__warm_cache({
    project: toProject,
    priority_files: await getRecentlyModifiedFiles(toProject),
    memory_budget: "256MB"
  });
};
```

### Focus Mode Patterns

#### Deep Work Session
```javascript
// Optimized deep work session
[Deep Work Mode - 2 Hour Session]:
  // Minimize distractions and optimize for sustained performance
  mcp__claude-flow__configure_session({
    mode: "deep_work",
    duration: "120m",
    interruption_blocking: true,
    memory_optimization: "sustained_performance"
  })
  
  // Single focused task with progressive development
  Task("Deep Work Developer", `
    Extended development session with sustained focus.
    
    Session structure:
    - 25min: Core implementation
    - 5min: Memory cleanup break  
    - 25min: Testing and refinement
    - 5min: Status update break
    - 25min: Integration work
    - 5min: Documentation break
    - 25min: Review and polish
    
    Memory management:
    - Progressive memory release between segments
    - Cache optimization during breaks
    - Memory usage monitoring throughout session
    
    Productivity tracking:
    - Code complexity metrics
    - Feature completion percentage
    - Memory efficiency maintained
  `, "focused-developer")
```

#### Collaborative Session
```javascript
// Optimized team collaboration session
[Collaborative Mode - Multi-Developer]:
  // Configure for team coordination with shared resources
  mcp__claude-flow__team_session({
    participants: team_members,
    shared_memory_pool: "2GB",
    coordination_topology: "team_mesh",
    resource_sharing: "intelligent"
  })
  
  // Spawn coordinated team agents
  team_members.forEach((member, index) => {
    Task(`Team Member ${index + 1}`, `
      Collaborative development with shared context.
      
      Coordination requirements:
      - Share progress via team memory pool
      - Coordinate resource usage with other team members
      - Use compressed communication protocols
      - Memory budget: ${sharedMemoryBudget / team_members.length}MB
      
      Collaboration tools:
      - Shared Archon knowledge base
      - Team Serena semantic cache
      - Claude Flow team coordination
      
      Real-time updates:
      - Progress sharing every 15 minutes
      - Memory usage monitoring
      - Conflict detection and resolution
    `, "team-developer")
  })
```

This comprehensive daily workflow guide ensures optimal productivity while maintaining system stability through intelligent memory management and resource optimization across all three integrated tools.