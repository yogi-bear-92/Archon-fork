# Integration Troubleshooting Guide: Serena + Archon + Claude Flow

## Overview

This comprehensive troubleshooting guide addresses the most common issues encountered when using the integrated Serena + Archon + Claude Flow development platform. Based on analysis of system metrics showing memory optimization improvements (from 69MB to 134-251MB free), this guide provides practical solutions for maintaining system stability.

## 🚨 Critical Issues and Emergency Procedures

### Issue 1: Memory Exhaustion Crisis

#### Symptoms
- Memory usage consistently >95% (16.3GB+ of 17GB)
- System becomes unresponsive
- Agent spawn failures with OOM errors
- Frequent process crashes
- IDE/editor freezing

#### Immediate Diagnosis
```bash
# Quick memory assessment
echo "=== EMERGENCY MEMORY DIAGNOSTIC ==="

# Current memory usage
free -h | head -2

# Top memory consumers
ps aux --sort=-%mem | head -15

# Node.js process memory details
pgrep -f node | xargs -I {} sh -c 'echo "PID: {}"; cat /proc/{}/status | grep -E "(VmRSS|VmSize)"'

# Check for memory leaks
lsof | grep node | wc -l  # File descriptor count (should be <1000)
```

#### Emergency Recovery Procedure
```bash
#!/bin/bash
# Emergency memory recovery - execute immediately

echo "🚨 EXECUTING EMERGENCY MEMORY RECOVERY"

# Step 1: Stop all non-essential processes
echo "Stopping non-essential services..."
pkill -f "typescript.*language.*server" 2>/dev/null || true
pkill -f "claude-flow.*worker" 2>/dev/null || true
pkill -f "serena.*background" 2>/dev/null || true
pkill -f "ruv-swarm" 2>/dev/null || true

# Step 2: Clear all caches aggressively
echo "Clearing caches..."
rm -rf ~/.cache/typescript-language-server/
rm -rf ~/.cache/serena/semantic/
rm -rf ~/.cache/archon/embeddings/
rm -rf /tmp/archon-* /tmp/serena-* /tmp/claude-flow-*
npm cache clean --force 2>/dev/null || true

# Step 3: Force garbage collection
echo "Forcing garbage collection..."
pgrep -f node | xargs -I {} kill -USR2 {} 2>/dev/null || true
sleep 5

# Step 4: Restart essential services only
echo "Restarting essential services with memory limits..."
export NODE_OPTIONS="--max-old-space-size=1024 --gc-interval=50"
npx archon start --profile=emergency --memory-limit=512MB &

# Wait for memory to stabilize
sleep 10
MEMORY_CHECK=$(free | awk 'NR==2{printf "%.1f", $3*100/$2}')
echo "Memory usage after recovery: ${MEMORY_CHECK}%"

if (( $(echo "$MEMORY_CHECK > 85.0" | bc -l) )); then
    echo "⚠️  Memory still high. Manual intervention required."
    echo "Consider restarting the system or killing more processes."
else
    echo "✅ Emergency recovery successful."
fi
```

#### Long-term Prevention
```javascript
// Implement memory circuit breaker
class MemoryCircuitBreaker {
  constructor() {
    this.thresholds = {
      warning: 0.80,    // 80% memory usage
      critical: 0.90,   // 90% memory usage
      emergency: 0.95   // 95% memory usage
    };
    
    this.monitoringInterval = 5000; // Check every 5 seconds
    this.recoveryTimeout = 300000; // 5 minutes to recover
    
    this.startMonitoring();
  }
  
  startMonitoring() {
    setInterval(async () => {
      const memoryUsage = await this.getMemoryUsage();
      
      if (memoryUsage > this.thresholds.emergency) {
        await this.emergencyShutdown();
      } else if (memoryUsage > this.thresholds.critical) {
        await this.criticalModeActivation();
      } else if (memoryUsage > this.thresholds.warning) {
        await this.warningModeActivation();
      }
    }, this.monitoringInterval);
  }
  
  async emergencyShutdown() {
    console.log('🚨 Emergency memory shutdown initiated');
    
    // Stop all non-essential agents
    await this.stopNonEssentialAgents();
    
    // Clear all caches
    await this.clearAllCaches();
    
    // Force garbage collection
    if (global.gc) global.gc();
    
    // Restart with minimal configuration
    await this.restartMinimal();
  }
}
```

### Issue 2: Service Coordination Failures

#### Symptoms
- MCP server connection errors
- Tool commands hanging without response
- Inconsistent state between Serena, Archon, and Claude Flow
- "Connection refused" errors
- Duplicate work being performed by different tools

#### Diagnosis Commands
```bash
# Service status check
echo "=== SERVICE COORDINATION DIAGNOSTIC ==="

# Check MCP server status
npx serena status --verbose
curl -f http://localhost:8080/health || echo "Archon API not responding"
npx claude-flow connection-test --all

# Check port bindings
netstat -tulpn | grep -E "(8080|8051|8052)"

# Process tree analysis
pstree -p | grep -E "(node|python|archon|serena|claude-flow)"

# Check service logs for errors
tail -50 ~/.cache/archon/logs/error.log 2>/dev/null || echo "No Archon logs"
tail -50 ~/.cache/serena/logs/mcp.log 2>/dev/null || echo "No Serena logs"
tail -50 ~/.cache/claude-flow/logs/coordination.log 2>/dev/null || echo "No Claude Flow logs"
```

#### Step-by-Step Resolution
```bash
#!/bin/bash
# Service coordination recovery procedure

echo "🔧 SERVICE COORDINATION RECOVERY"

# Step 1: Stop all services cleanly
echo "Stopping all services..."
npx claude-flow stop --graceful
npx serena stop --save-state
curl -X POST http://localhost:8080/shutdown 2>/dev/null || true

# Wait for clean shutdown
sleep 10

# Step 2: Clear coordination state
echo "Clearing coordination state..."
rm -rf ~/.cache/claude-flow/coordination/*
rm -rf ~/.cache/serena/mcp-state/*
rm -rf /tmp/archon-mcp-*

# Step 3: Restart in correct order with coordination
echo "Restarting services with coordination..."

# Start Archon first (knowledge base)
cd python && python -m uvicorn main:app --port 8080 --reload &
ARCHON_PID=$!
echo "Archon started with PID: $ARCHON_PID"

# Wait for Archon to be ready
until curl -f http://localhost:8080/health >/dev/null 2>&1; do
  echo "Waiting for Archon..."
  sleep 2
done

# Start Serena MCP server
npx serena start --mcp-port=8051 --archon-integration &
SERENA_PID=$!
echo "Serena started with PID: $SERENA_PID"

# Start Claude Flow coordination
npx claude-flow start --topology=unified --archon-endpoint=http://localhost:8080 &
CLAUDE_FLOW_PID=$!
echo "Claude Flow started with PID: $CLAUDE_FLOW_PID"

# Step 4: Test coordination
sleep 15
echo "Testing coordination..."

# Test Archon
ARCHON_STATUS=$(curl -s http://localhost:8080/health | jq -r .status 2>/dev/null || echo "error")
echo "Archon status: $ARCHON_STATUS"

# Test Serena
SERENA_STATUS=$(npx serena ping 2>&1 | grep -q "pong" && echo "ok" || echo "error")
echo "Serena status: $SERENA_STATUS"

# Test Claude Flow
CLAUDE_FLOW_STATUS=$(npx claude-flow status --json | jq -r .status 2>/dev/null || echo "error")
echo "Claude Flow status: $CLAUDE_FLOW_STATUS"

if [[ "$ARCHON_STATUS" == "ok" && "$SERENA_STATUS" == "ok" && "$CLAUDE_FLOW_STATUS" == "running" ]]; then
    echo "✅ All services coordinated successfully"
else
    echo "⚠️  Coordination issues remain. Check individual service logs."
fi
```

### Issue 3: Performance Degradation Over Time

#### Symptoms
- Increasing response times for all operations
- Memory usage gradually climbing
- Agent coordination taking longer
- Semantic analysis becoming slower
- System becomes sluggish after extended use

#### Performance Analysis
```javascript
// Comprehensive performance analysis
[Performance Diagnosis - Single Message]:
  // Memory trend analysis
  mcp__claude-flow__memory_usage({
    action: "trend_analysis",
    timeframe: "6h",
    export_detailed: true
  })
  
  // Archon performance metrics
  mcp__archon__performance_analysis({
    include_query_times: true,
    cache_hit_ratios: true,
    memory_allocation_patterns: true
  })
  
  // Serena semantic cache analysis
  mcp__serena__cache_analysis({
    performance_metrics: true,
    cache_efficiency: true,
    memory_usage: true
  })
  
  // System resource analysis
  Bash(`
    # CPU usage patterns
    sar -u 1 10 > /tmp/cpu-analysis.log
    
    # Memory usage patterns
    vmstat 1 10 > /tmp/memory-analysis.log
    
    # Disk I/O patterns
    iostat -x 1 10 > /tmp/io-analysis.log
    
    # Process analysis
    ps auxf > /tmp/process-tree.log
    
    echo "Performance analysis exported to /tmp/"
  `)
```

#### Performance Recovery Actions
```bash
#!/bin/bash
# Performance optimization procedure

echo "🚀 PERFORMANCE RECOVERY PROCEDURE"

# Step 1: Identify performance bottlenecks
echo "Analyzing performance bottlenecks..."

# Memory fragmentation check
cat /proc/buddyinfo | grep -v "Node 0"
FRAGMENTATION_HIGH=$(cat /proc/buddyinfo | awk '{sum += $4} END {print (sum > 1000) ? "yes" : "no"}')

# Cache efficiency check
CACHE_SIZE=$(find ~/.cache -type f -name "*.cache" -exec du -ch {} + | grep total | cut -f1)
echo "Total cache size: $CACHE_SIZE"

# Process memory analysis
MEMORY_LEAKS=$(ps aux --sort=-%mem | head -10 | awk '$6 > 1000000 {print $2, $6}' | wc -l)
echo "Potential memory leaks detected: $MEMORY_LEAKS processes"

# Step 2: Optimize system performance
echo "Applying performance optimizations..."

# Cache cleanup with intelligence
npx archon cache-optimize --intelligent --keep-hot-data
npx serena cache-cleanup --preserve-recent --compress-old
npx claude-flow cache-optimize --memory-pressure-adaptive

# Memory optimization
if [[ "$FRAGMENTATION_HIGH" == "yes" ]]; then
    echo "High memory fragmentation detected. Forcing compaction..."
    echo 1 > /proc/sys/vm/compact_memory 2>/dev/null || echo "Compaction requires root"
fi

# Process optimization
echo "Optimizing process configurations..."
pgrep -f node | xargs -I {} renice +5 {} 2>/dev/null || true  # Lower priority for background tasks

# Step 3: Restart services with optimized configuration
echo "Restarting with optimized configuration..."

# Export current state
npx archon export-state --compress
npx serena export-cache --selective
npx claude-flow export-session --minimal

# Restart services
./scripts/restart-optimized.sh

echo "✅ Performance recovery completed"
```

## 🔧 Common Issues and Solutions

### Issue 4: Agent Spawn Failures

#### Problem: Agents fail to spawn or hang during initialization
```javascript
// Robust agent spawning with fallbacks
const spawnAgentWithFallbacks = async (agentType, task, options = {}) => {
  const maxRetries = 3;
  const fallbackAgents = {
    'code-analyzer': 'coder',
    'system-architect': 'planner',
    'performance-engineer': 'reviewer'
  };
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      // Check memory before spawning
      const memoryCheck = await checkMemoryAvailability();
      if (!memoryCheck.sufficient) {
        throw new Error(`Insufficient memory: ${memoryCheck.available}MB available`);
      }
      
      // Spawn with timeout
      const timeoutMs = options.timeout || 30000;
      const agent = await Promise.race([
        Task(agentType, task, options),
        new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Agent spawn timeout')), timeoutMs)
        )
      ]);
      
      return agent;
      
    } catch (error) {
      console.warn(`Agent spawn attempt ${attempt} failed:`, error.message);
      
      if (attempt === maxRetries) {
        // Try fallback agent
        const fallbackType = fallbackAgents[agentType];
        if (fallbackType) {
          console.log(`Using fallback agent: ${fallbackType}`);
          return await Task(fallbackType, `${task} (fallback from ${agentType})`, {
            ...options,
            fallback: true
          });
        }
        
        throw new Error(`Failed to spawn ${agentType} after ${maxRetries} attempts: ${error.message}`);
      }
      
      // Wait before retry
      await new Promise(resolve => setTimeout(resolve, attempt * 1000));
    }
  }
};
```

### Issue 5: Knowledge Base Query Timeouts

#### Problem: Archon RAG queries hanging or timing out
```python
# Robust knowledge base querying with fallbacks
class RobustKnowledgeQuery:
    def __init__(self, timeout=30):
        self.timeout = timeout
        self.fallback_sources = [
            'cached_embeddings',
            'simplified_search',
            'keyword_fallback'
        ]
    
    async def query_with_fallback(self, query, context=None):
        try:
            # Primary query with timeout
            result = await asyncio.wait_for(
                self.primary_rag_query(query, context),
                timeout=self.timeout
            )
            return result
            
        except asyncio.TimeoutError:
            print(f"Primary query timed out after {self.timeout}s, trying fallbacks...")
            
            for fallback_method in self.fallback_sources:
                try:
                    result = await self.execute_fallback(fallback_method, query, context)
                    if result:
                        return result
                except Exception as e:
                    print(f"Fallback {fallback_method} failed: {e}")
                    
            # Final fallback - return structured empty response
            return {
                'status': 'fallback',
                'message': 'Knowledge query failed, proceeding with limited context',
                'context': context or {},
                'suggestions': self.generate_basic_suggestions(query)
            }
    
    async def execute_fallback(self, method, query, context):
        if method == 'cached_embeddings':
            return await self.query_cached_embeddings(query)
        elif method == 'simplified_search':
            return await self.keyword_search(query, max_results=5)
        elif method == 'keyword_fallback':
            return await self.basic_keyword_match(query)
```

### Issue 6: Semantic Analysis Failures

#### Problem: Serena semantic analysis crashing or producing incorrect results
```bash
#!/bin/bash
# Serena semantic analysis recovery

echo "🔍 SERENA SEMANTIC ANALYSIS RECOVERY"

# Check Serena health
SERENA_HEALTH=$(npx serena health-check --json 2>/dev/null || echo '{"status":"error"}')
SERENA_STATUS=$(echo $SERENA_HEALTH | jq -r .status 2>/dev/null || echo "error")

if [[ "$SERENA_STATUS" != "ok" ]]; then
    echo "Serena unhealthy. Attempting recovery..."
    
    # Clear corrupted semantic cache
    rm -rf ~/.cache/serena/semantic/*.corrupted
    
    # Reset language servers
    pkill -f "typescript.*language.*server"
    pkill -f "python.*language.*server"
    
    # Restart Serena with conservative settings
    npx serena restart --safe-mode --rebuild-cache
    
    # Wait for recovery
    sleep 15
    
    # Test semantic analysis
    TEST_RESULT=$(npx serena test-semantic --file="package.json" 2>&1)
    if echo "$TEST_RESULT" | grep -q "success"; then
        echo "✅ Serena semantic analysis recovered"
    else
        echo "⚠️  Serena still having issues. Manual intervention required."
        echo "Error: $TEST_RESULT"
    fi
else
    echo "✅ Serena is healthy"
fi
```

## 🚨 Emergency Procedures

### Complete System Reset

#### When to use: System completely unresponsive or corrupted state
```bash
#!/bin/bash
# Complete system reset procedure

echo "🔄 COMPLETE SYSTEM RESET - LAST RESORT"

# Backup current state if possible
echo "Attempting state backup..."
mkdir -p ~/.backup/$(date +%Y%m%d_%H%M%S)
cp -r ~/.cache/archon/projects ~/.backup/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true
cp -r ~/.cache/serena/semantic ~/.backup/$(date +%Y%m%d_%H%M%S)/ 2>/dev/null || true

# Stop all processes forcefully
echo "Stopping all processes..."
pkill -9 -f "archon" 2>/dev/null || true
pkill -9 -f "serena" 2>/dev/null || true  
pkill -9 -f "claude-flow" 2>/dev/null || true
pkill -9 -f "typescript.*language.*server" 2>/dev/null || true

# Clean all caches and temporary files
echo "Cleaning all system state..."
rm -rf ~/.cache/archon/
rm -rf ~/.cache/serena/
rm -rf ~/.cache/claude-flow/
rm -rf /tmp/archon-*
rm -rf /tmp/serena-*
rm -rf /tmp/claude-flow-*

# Reset configuration to defaults
echo "Resetting configurations..."
cp config/default.archon.json ~/.config/archon/config.json
cp config/default.serena.json ~/.config/serena/config.json
cp config/default.claude-flow.json ~/.config/claude-flow/config.json

# Reinstall if necessary
echo "Checking installations..."
npm install --no-save 2>/dev/null || echo "NPM install failed"
pip install -r python/requirements.txt --quiet 2>/dev/null || echo "Pip install failed"

# Restart system with minimal configuration
echo "Starting fresh system..."
export NODE_OPTIONS="--max-old-space-size=1024"
npx archon start --fresh --minimal &
sleep 5
npx serena start --fresh --basic-cache &
sleep 5
npx claude-flow start --fresh --simple-topology &

# Verify system health
sleep 30
echo "Verifying system health..."
HEALTH_CHECK=$(npx archon health && npx serena ping && npx claude-flow status)
if [[ $? -eq 0 ]]; then
    echo "✅ System reset completed successfully"
else
    echo "❌ System reset failed. Manual investigation required."
fi
```

## 📊 Monitoring and Prevention

### Proactive Monitoring Setup

```javascript
// Comprehensive monitoring system
class ProactiveMonitoring {
  constructor() {
    this.monitors = {
      memory: new MemoryMonitor(),
      performance: new PerformanceMonitor(), 
      coordination: new CoordinationMonitor(),
      health: new HealthMonitor()
    };
    
    this.alertThresholds = {
      memory: { warning: 0.80, critical: 0.90 },
      responseTime: { warning: 1000, critical: 5000 },
      errorRate: { warning: 0.05, critical: 0.15 },
      coordinationLatency: { warning: 500, critical: 2000 }
    };
  }
  
  startMonitoring() {
    // Memory monitoring every 5 seconds
    setInterval(() => this.monitors.memory.check(), 5000);
    
    // Performance monitoring every 30 seconds
    setInterval(() => this.monitors.performance.check(), 30000);
    
    // Coordination health every 60 seconds  
    setInterval(() => this.monitors.coordination.check(), 60000);
    
    // Overall health check every 5 minutes
    setInterval(() => this.monitors.health.comprehensiveCheck(), 300000);
  }
  
  async handleAlert(alertType, severity, details) {
    console.log(`🚨 Alert: ${alertType} (${severity})`, details);
    
    // Automatic remediation for known issues
    if (alertType === 'memory' && severity === 'warning') {
      await this.performMemoryCleanup();
    } else if (alertType === 'coordination' && severity === 'critical') {
      await this.restartCoordinationServices();
    } else if (alertType === 'performance' && severity === 'critical') {
      await this.performEmergencyOptimization();
    }
    
    // Log for analysis
    await this.logAlert(alertType, severity, details);
  }
}
```

### Automated Recovery Scripts

```bash
#!/bin/bash
# Automated recovery system

# Create monitoring cron job
(crontab -l 2>/dev/null; echo "*/5 * * * * /path/to/health-check.sh") | crontab -

# Create logrotate configuration
cat > /etc/logrotate.d/archon-integration << EOF
~/.cache/archon/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    postrotate
        pkill -USR1 node 2>/dev/null || true
    endscript
}
EOF

echo "✅ Automated monitoring and recovery system installed"
```

This troubleshooting guide provides comprehensive coverage of the most common issues and their solutions, ensuring system stability and optimal performance of the integrated development platform.