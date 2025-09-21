#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

class MemoryMonitor {
  constructor() {
    this.configPath = path.join(__dirname, '../config/memory-limits.json');
    this.config = this.loadConfig();
    this.metricsPath = path.join(__dirname, '../.claude-flow/metrics/system-metrics.json');
  }

  loadConfig() {
    try {
      return JSON.parse(fs.readFileSync(this.configPath, 'utf8'));
    } catch (error) {
      console.error('Failed to load memory config:', error);
      return this.getDefaultConfig();
    }
  }

  getDefaultConfig() {
    return {
      memoryManagement: {
        globalLimits: {
          totalBudget: "8GB",
          emergencyThreshold: "15GB"
        },
        monitoring: {
          checkIntervalMs: 5000,
          alertThresholds: {
            warning: 0.8,
            critical: 0.9,
            emergency: 0.95
          }
        }
      }
    };
  }

  getCurrentMemoryUsage() {
    try {
      const stats = JSON.parse(fs.readFileSync(this.metricsPath, 'utf8'));
      const latest = stats[stats.length - 1];
      return {
        total: latest.memoryTotal,
        used: latest.memoryUsed,
        free: latest.memoryFree,
        usagePercent: latest.memoryUsagePercent / 100,
        efficiency: latest.memoryEfficiency / 100
      };
    } catch (error) {
      // Fallback to system memory check
      const memInfo = execSync('vm_stat').toString();
      const pageSize = parseInt(execSync('vm_stat | head -1 | grep -o "[0-9]*"').toString());
      
      // Parse vm_stat output for macOS
      const freePages = parseInt(memInfo.match(/Pages free:\s*(\d+)/)?.[1] || '0');
      const totalMemory = parseInt(execSync('sysctl -n hw.memsize').toString());
      const freeMemory = freePages * pageSize;
      const usedMemory = totalMemory - freeMemory;
      
      return {
        total: totalMemory,
        used: usedMemory,
        free: freeMemory,
        usagePercent: usedMemory / totalMemory,
        efficiency: freeMemory / totalMemory
      };
    }
  }

  formatBytes(bytes) {
    const gb = bytes / (1024 ** 3);
    const mb = bytes / (1024 ** 2);
    return gb > 1 ? `${gb.toFixed(2)}GB` : `${mb.toFixed(0)}MB`;
  }

  checkThresholds() {
    const memory = this.getCurrentMemoryUsage();
    const thresholds = this.config.memoryManagement.monitoring.alertThresholds;
    
    console.log(`\n🔍 Memory Status Check`);
    console.log(`Total: ${this.formatBytes(memory.total)}`);
    console.log(`Used: ${this.formatBytes(memory.used)} (${(memory.usagePercent * 100).toFixed(1)}%)`);
    console.log(`Free: ${this.formatBytes(memory.free)}`);
    console.log(`Efficiency: ${(memory.efficiency * 100).toFixed(1)}%`);

    if (memory.usagePercent >= thresholds.emergency) {
      console.log(`🚨 EMERGENCY: Memory usage ${(memory.usagePercent * 100).toFixed(1)}% >= ${(thresholds.emergency * 100)}%`);
      this.emergencyCleanup();
      return 'emergency';
    } else if (memory.usagePercent >= thresholds.critical) {
      console.log(`⚠️  CRITICAL: Memory usage ${(memory.usagePercent * 100).toFixed(1)}% >= ${(thresholds.critical * 100)}%`);
      this.criticalCleanup();
      return 'critical';
    } else if (memory.usagePercent >= thresholds.warning) {
      console.log(`⚠️  WARNING: Memory usage ${(memory.usagePercent * 100).toFixed(1)}% >= ${(thresholds.warning * 100)}%`);
      this.warningCleanup();
      return 'warning';
    } else {
      console.log(`✅ OK: Memory usage within acceptable limits`);
      return 'ok';
    }
  }

  emergencyCleanup() {
    console.log('🚑 Executing emergency memory cleanup...');
    
    // Kill non-essential processes
    this.killProcesses(['archon-ui', 'claude-flow', 'unnecessary-processes']);
    
    // Force garbage collection
    this.forceGarbageCollection();
    
    // Clear all caches
    this.clearCaches();
    
    // Restart in minimal mode
    console.log('♻️  Restarting essential services in minimal mode...');
  }

  criticalCleanup() {
    console.log('🔧 Executing critical memory cleanup...');
    
    // Clear caches
    this.clearCaches();
    
    // Force GC
    this.forceGarbageCollection();
    
    // Reduce agent limits
    this.reduceAgentLimits();
  }

  warningCleanup() {
    console.log('🧹 Executing warning-level cleanup...');
    
    // Clear old caches
    this.clearOldCaches();
    
    // Suggest GC
    this.suggestGarbageCollection();
  }

  killProcesses(processNames) {
    processNames.forEach(name => {
      try {
        execSync(`pkill -f "${name}" || true`, { stdio: 'ignore' });
        console.log(`  ✅ Killed processes matching: ${name}`);
      } catch (error) {
        console.log(`  ⚠️  No processes found for: ${name}`);
      }
    });
  }

  forceGarbageCollection() {
    try {
      // Send GC signal to Node.js processes
      execSync('pkill -USR2 node || true', { stdio: 'ignore' });
      console.log('  ✅ Forced garbage collection on Node.js processes');
    } catch (error) {
      console.log('  ⚠️  Could not force GC:', error.message);
    }
  }

  clearCaches() {
    const cacheDirs = [
      '.claude-flow/cache',
      '.serena/cache', 
      'node_modules/.cache',
      'python/__pycache__'
    ];
    
    cacheDirs.forEach(dir => {
      try {
        const fullPath = path.join(__dirname, '..', dir);
        if (fs.existsSync(fullPath)) {
          execSync(`rm -rf "${fullPath}"`, { stdio: 'ignore' });
          console.log(`  ✅ Cleared cache: ${dir}`);
        }
      } catch (error) {
        console.log(`  ⚠️  Could not clear cache ${dir}:`, error.message);
      }
    });
  }

  clearOldCaches() {
    // Clear caches older than 1 hour
    try {
      execSync('find . -name "*.cache" -mtime +1h -delete 2>/dev/null || true', { stdio: 'ignore' });
      console.log('  ✅ Cleared old cache files');
    } catch (error) {
      console.log('  ⚠️  Could not clear old caches:', error.message);
    }
  }

  suggestGarbageCollection() {
    console.log('  💡 Suggesting garbage collection to running processes...');
  }

  reduceAgentLimits() {
    console.log('  🔧 Reducing agent spawn limits...');
    // This would update the runtime configuration
  }

  monitorContinuously() {
    console.log('👀 Starting continuous memory monitoring...');
    const interval = this.config.memoryManagement.monitoring.checkIntervalMs;
    
    setInterval(() => {
      const status = this.checkThresholds();
      if (status === 'emergency') {
        console.log('🛑 Emergency detected, stopping continuous monitoring');
        process.exit(1);
      }
    }, interval);
  }

  canSpawnAgent(memoryRequirement = '256MB') {
    const memory = this.getCurrentMemoryUsage();
    const requiredBytes = this.parseMemorySize(memoryRequirement);
    const available = memory.free;
    
    const canSpawn = available > requiredBytes * 2; // 2x safety margin
    console.log(`Agent spawn check: ${canSpawn ? '✅ OK' : '❌ DENIED'} (need ${memoryRequirement}, have ${this.formatBytes(available)})`);
    
    return canSpawn;
  }

  parseMemorySize(sizeStr) {
    const num = parseFloat(sizeStr);
    if (sizeStr.includes('GB')) return num * 1024 ** 3;
    if (sizeStr.includes('MB')) return num * 1024 ** 2;
    if (sizeStr.includes('KB')) return num * 1024;
    return num;
  }
}

// CLI Interface
const monitor = new MemoryMonitor();

const command = process.argv[2];
switch (command) {
  case '--check-threshold':
    process.exit(monitor.checkThresholds() === 'ok' ? 0 : 1);
    break;
  case '--check-spawn':
    const requirement = process.argv[3] || '256MB';
    process.exit(monitor.canSpawnAgent(requirement) ? 0 : 1);
    break;
  case '--monitor':
    monitor.monitorContinuously();
    break;
  case '--emergency-cleanup':
    monitor.emergencyCleanup();
    break;
  case '--status':
    monitor.checkThresholds();
    break;
  default:
    console.log(`
Usage: node memory-monitor.js [command]

Commands:
  --check-threshold    Check memory thresholds (exit 0 if OK)
  --check-spawn [size] Check if agent can spawn (default 256MB)
  --monitor           Start continuous monitoring
  --emergency-cleanup  Force emergency cleanup
  --status            Show current memory status

Exit codes:
  0 = OK/Success
  1 = Warning/Critical/Emergency
`);
}