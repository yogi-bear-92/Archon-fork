#!/usr/bin/env node

/**
 * Coordination Patterns for Unified Serena + Archon + Claude Flow Integration
 * Memory-optimized coordination with intelligent resource management
 */

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

class CoordinationManager {
  constructor() {
    this.configPath = path.join(__dirname, '../config/memory-limits.json');
    this.config = this.loadConfig();
    this.services = new Map();
    this.eventBus = new EventBus();
    this.memoryManager = new MemoryManager(this.config);
  }

  loadConfig() {
    try {
      return JSON.parse(fs.readFileSync(this.configPath, 'utf8'));
    } catch (error) {
      console.error('Failed to load configuration:', error);
      process.exit(1);
    }
  }

  /**
   * Memory-aware agent spawning pattern
   */
  async spawnAgentSafe(type, task, options = {}) {
    const memoryBudget = options.memoryLimit || '256MB';
    
    // Check memory availability
    if (!await this.memoryManager.canAllocate(memoryBudget)) {
      console.warn(`🚨 Cannot spawn ${type} agent: insufficient memory`);
      return this.queueForLater(type, task, options);
    }

    // Spawn with resource limits
    const agent = await this.spawnWithLimits(type, task, {
      memoryLimit: memoryBudget,
      priority: options.priority || 'normal',
      hooks: true
    });

    // Register for monitoring
    this.services.set(agent.id, {
      type,
      process: agent,
      startTime: Date.now(),
      memoryBudget
    });

    return agent;
  }

  async spawnWithLimits(type, task, limits) {
    const command = this.getAgentCommand(type);
    const env = {
      ...process.env,
      MEMORY_LIMIT: limits.memoryLimit,
      TASK_PRIORITY: limits.priority,
      COORDINATION_HOOKS: limits.hooks ? 'enabled' : 'disabled'
    };

    const agentProcess = spawn(command.cmd, command.args, {
      env,
      stdio: ['pipe', 'pipe', 'pipe'],
      detached: false
    });

    // Setup coordination hooks
    if (limits.hooks) {
      this.setupCoordinationHooks(agentProcess, type);
    }

    return {
      id: `${type}-${Date.now()}`,
      process: agentProcess,
      type,
      task
    };
  }

  getAgentCommand(type) {
    const commands = {
      'code-analyzer': {
        cmd: 'npx',
        args: ['claude-code', 'task', 'analyze-code', '--serena-integration']
      },
      'backend-dev': {
        cmd: 'npx',
        args: ['claude-code', 'task', 'backend-development', '--archon-prp']
      },
      'tester': {
        cmd: 'npx', 
        args: ['claude-code', 'task', 'testing', '--comprehensive']
      },
      'reviewer': {
        cmd: 'npx',
        args: ['claude-code', 'task', 'code-review', '--semantic-aware']
      }
    };

    return commands[type] || {
      cmd: 'npx',
      args: ['claude-code', 'task', type]
    };
  }

  setupCoordinationHooks(agentProcess, type) {
    // Pre-task hook
    agentProcess.stdin.write(JSON.stringify({
      type: 'hook',
      phase: 'pre-task',
      command: `npx claude-flow hooks pre-task --description "${type} starting"`
    }) + '\n');

    // Monitor agent output for coordination events
    agentProcess.stdout.on('data', (data) => {
      const output = data.toString();
      this.processAgentOutput(agentProcess.id, output);
    });

    // Post-task cleanup
    agentProcess.on('exit', (code) => {
      spawn('npx', ['claude-flow', 'hooks', 'post-task', '--task-id', agentProcess.id], {
        stdio: 'ignore'
      });
      
      this.services.delete(agentProcess.id);
      this.memoryManager.releaseAgent(agentProcess.id);
    });
  }

  processAgentOutput(agentId, output) {
    // Parse output for coordination signals
    const lines = output.split('\n');
    
    lines.forEach(line => {
      if (line.includes('[COORDINATION]')) {
        this.eventBus.emit('coordination-event', {
          agentId,
          message: line,
          timestamp: Date.now()
        });
      }
      
      if (line.includes('[MEMORY_WARNING]')) {
        this.handleMemoryWarning(agentId, line);
      }
    });
  }

  handleMemoryWarning(agentId, warning) {
    console.warn(`🚨 Memory warning from ${agentId}: ${warning}`);
    
    // Implement memory pressure response
    this.memoryManager.handlePressure(agentId);
  }

  /**
   * Intelligent task orchestration with resource awareness
   */
  async orchestrateWorkflow(workflow) {
    console.log(`🎯 Orchestrating workflow: ${workflow.name}`);
    
    // Analyze resource requirements
    const resourcePlan = this.analyzeResourceRequirements(workflow);
    
    // Check if we can execute now or need to queue
    if (!await this.memoryManager.canExecuteWorkflow(resourcePlan)) {
      return this.scheduleForLater(workflow, resourcePlan);
    }

    // Execute workflow with coordination
    return this.executeWorkflowWithCoordination(workflow, resourcePlan);
  }

  analyzeResourceRequirements(workflow) {
    const requirements = {
      totalMemory: 0,
      agents: [],
      dependencies: []
    };

    workflow.steps.forEach(step => {
      const agentMemory = this.getAgentMemoryRequirement(step.type);
      requirements.totalMemory += agentMemory;
      requirements.agents.push({
        type: step.type,
        memory: agentMemory,
        priority: step.priority || 'normal'
      });
    });

    return requirements;
  }

  getAgentMemoryRequirement(agentType) {
    const memoryMap = {
      'code-analyzer': 256, // MB
      'backend-dev': 512,
      'tester': 384,
      'reviewer': 256,
      'researcher': 128
    };

    return memoryMap[agentType] || 256;
  }

  async executeWorkflowWithCoordination(workflow, resourcePlan) {
    const executionId = `workflow-${Date.now()}`;
    console.log(`🚀 Executing workflow ${workflow.name} (ID: ${executionId})`);

    // Initialize coordination context
    await this.initializeCoordinationContext(executionId, workflow);

    // Execute steps based on strategy
    const results = await this.executeSteps(workflow.steps, resourcePlan);

    // Finalize coordination
    await this.finalizeCoordinationContext(executionId, results);

    return {
      executionId,
      workflow: workflow.name,
      results,
      success: results.every(r => r.success)
    };
  }

  async initializeCoordinationContext(executionId, workflow) {
    // Initialize Claude Flow coordination
    spawn('npx', ['claude-flow', 'hooks', 'session-restore', '--session-id', executionId], {
      stdio: 'ignore'
    });

    // Store workflow context in memory
    this.eventBus.emit('workflow-start', {
      executionId,
      workflow: workflow.name,
      timestamp: Date.now()
    });
  }

  async executeSteps(steps, resourcePlan) {
    const results = [];
    
    // Group steps by execution strategy
    const parallelSteps = steps.filter(s => s.execution === 'parallel');
    const sequentialSteps = steps.filter(s => s.execution !== 'parallel');

    // Execute parallel steps
    if (parallelSteps.length > 0) {
      console.log(`⚡ Executing ${parallelSteps.length} parallel steps`);
      const parallelResults = await Promise.all(
        parallelSteps.map(step => this.executeStep(step))
      );
      results.push(...parallelResults);
    }

    // Execute sequential steps
    for (const step of sequentialSteps) {
      console.log(`🔄 Executing sequential step: ${step.type}`);
      const result = await this.executeStep(step);
      results.push(result);
      
      // Short delay for memory cleanup
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    return results;
  }

  async executeStep(step) {
    try {
      const agent = await this.spawnAgentSafe(step.type, step.task, {
        memoryLimit: `${this.getAgentMemoryRequirement(step.type)}MB`,
        priority: step.priority
      });

      const result = await this.waitForAgentCompletion(agent);
      
      return {
        step: step.type,
        success: result.exitCode === 0,
        output: result.output,
        duration: result.duration
      };
    } catch (error) {
      return {
        step: step.type,
        success: false,
        error: error.message,
        duration: 0
      };
    }
  }

  async waitForAgentCompletion(agent) {
    return new Promise((resolve) => {
      const startTime = Date.now();
      let output = '';

      agent.process.stdout.on('data', (data) => {
        output += data.toString();
      });

      agent.process.on('exit', (code) => {
        resolve({
          exitCode: code,
          output,
          duration: Date.now() - startTime
        });
      });
    });
  }

  async finalizeCoordinationContext(executionId, results) {
    // Export metrics
    spawn('npx', ['claude-flow', 'hooks', 'session-end', '--export-metrics', 'true'], {
      stdio: 'ignore'
    });

    // Update workflow statistics
    this.eventBus.emit('workflow-complete', {
      executionId,
      results,
      timestamp: Date.now()
    });
  }

  queueForLater(type, task, options) {
    console.log(`⏳ Queueing ${type} agent for later execution`);
    // Implementation would queue the task for when memory is available
    return Promise.resolve({ queued: true, type, task });
  }

  scheduleForLater(workflow, resourcePlan) {
    console.log(`⏳ Scheduling workflow ${workflow.name} for later execution`);
    // Implementation would schedule the workflow for when resources are available
    return Promise.resolve({ scheduled: true, workflow: workflow.name });
  }
}

/**
 * Memory Manager for resource-aware coordination
 */
class MemoryManager {
  constructor(config) {
    this.config = config.memoryManagement;
    this.allocations = new Map();
    this.totalAllocated = 0;
  }

  async canAllocate(memorySize) {
    const bytesNeeded = this.parseMemorySize(memorySize);
    const currentUsage = await this.getCurrentMemoryUsage();
    const available = currentUsage.free;
    
    return available > bytesNeeded * 1.5; // 50% safety margin
  }

  async canExecuteWorkflow(resourcePlan) {
    const totalNeeded = resourcePlan.totalMemory * 1024 * 1024; // Convert MB to bytes
    const currentUsage = await this.getCurrentMemoryUsage();
    
    return currentUsage.free > totalNeeded * 2; // 100% safety margin for workflows
  }

  parseMemorySize(sizeStr) {
    const num = parseFloat(sizeStr);
    if (sizeStr.includes('GB')) return num * 1024 ** 3;
    if (sizeStr.includes('MB')) return num * 1024 ** 2;
    return num;
  }

  async getCurrentMemoryUsage() {
    // This would call the memory monitor script
    const { spawn } = require('child_process');
    
    return new Promise((resolve) => {
      const monitor = spawn('node', [path.join(__dirname, 'memory-monitor.js'), '--status']);
      let output = '';
      
      monitor.stdout.on('data', (data) => {
        output += data.toString();
      });
      
      monitor.on('close', () => {
        // Parse the output to get memory stats
        // This is a simplified version
        const freeMatch = output.match(/Free: (\d+(?:\.\d+)?)([GM])B/);
        const free = freeMatch ? 
          parseFloat(freeMatch[1]) * (freeMatch[2] === 'G' ? 1024**3 : 1024**2) : 
          1024**3; // Default 1GB
          
        resolve({ free });
      });
    });
  }

  releaseAgent(agentId) {
    if (this.allocations.has(agentId)) {
      const allocation = this.allocations.get(agentId);
      this.totalAllocated -= allocation;
      this.allocations.delete(agentId);
      console.log(`🔓 Released memory allocation for ${agentId}`);
    }
  }

  handlePressure(agentId) {
    console.log(`🚨 Handling memory pressure for ${agentId}`);
    
    // Implement memory pressure response
    // Could involve pausing agents, clearing caches, etc.
  }
}

/**
 * Event Bus for lightweight coordination
 */
class EventBus {
  constructor() {
    this.listeners = new Map();
  }

  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(listener => {
        try {
          listener(data);
        } catch (error) {
          console.error(`Error in event listener for ${event}:`, error);
        }
      });
    }
  }

  on(event, listener) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(listener);
  }
}

// CLI Interface
if (require.main === module) {
  const coordinator = new CoordinationManager();
  
  const command = process.argv[2];
  const args = process.argv.slice(3);
  
  switch (command) {
    case 'spawn-agent':
      const [type, task] = args;
      coordinator.spawnAgentSafe(type, task).then(result => {
        console.log('Agent spawned:', result);
      });
      break;
      
    case 'orchestrate':
      const workflowFile = args[0];
      const workflow = JSON.parse(fs.readFileSync(workflowFile, 'utf8'));
      coordinator.orchestrateWorkflow(workflow).then(result => {
        console.log('Workflow completed:', result);
      });
      break;
      
    default:
      console.log(`
Usage: node coordination-patterns.js [command] [args]

Commands:
  spawn-agent <type> <task>     Spawn a memory-aware agent
  orchestrate <workflow.json>   Execute a workflow with coordination

Examples:
  node coordination-patterns.js spawn-agent code-analyzer "Analyze main.py"
  node coordination-patterns.js orchestrate workflows/full-stack-dev.json
`);
  }
}

module.exports = { CoordinationManager, MemoryManager, EventBus };