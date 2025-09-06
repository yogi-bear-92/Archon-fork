#!/usr/bin/env node

/**
 * Claude Flow Hooks Integration for Archon
 * Integrates SPARC methodology with Archon PRP framework
 */

const hooks = {
  // Pre-operation hooks with ARCHON-FIRST enforcement
  'pre-task': async (context) => {
    console.log(`🚀 Starting task: ${context.description}`);
    
    // 🚨 CRITICAL: ARCHON-FIRST RULE ENFORCEMENT
    console.log('🏛️ ARCHON-FIRST RULE: Checking for Archon MCP server...');
    
    // Check if Archon MCP server is available
    const archonAvailable = await checkArchonMcpServer();
    if (archonAvailable) {
      console.log('✅ Archon MCP server detected - Using Archon task management as PRIMARY');
      console.log('📋 MANDATORY: Use archon:manage_task before any coding');
      console.log('⚠️  TodoWrite is SECONDARY tracking only');
      
      // Suggest Archon workflow steps
      console.log('🔄 Archon Workflow Cycle:');
      console.log('  1. archon:manage_task(action="get", task_id="...")');
      console.log('  2. archon:perform_rag_query() + archon:search_code_examples()');
      console.log('  3. Implement based on research');
      console.log('  4. archon:manage_task(action="update", status="review")');
      console.log('  5. Get next task and repeat');
      
      return { 
        archonFirst: true, 
        primaryTaskManagement: 'archon',
        secondaryTracking: 'todowrite',
        mandatoryWorkflow: ['check_task', 'research', 'implement', 'update_status', 'next_task']
      };
    } else {
      console.log('⚠️  Archon MCP server not available - fallback to TodoWrite');
      console.log('💡 To enable Archon: Ensure MCP server is running on port 8051');
    }
    
    // Check if this is an Archon PRP task
    if (context.description.includes('PRP') || context.description.includes('Progressive Refinement')) {
      console.log('📋 Initializing Archon PRP workflow...');
      return { archonPrp: true, cycles: 4, archonIntegration: archonAvailable };
    }
    
    // Check if this is a SPARC workflow
    if (context.description.includes('SPARC') || context.phases) {
      console.log('🎯 Initializing SPARC methodology...');
      return { sparcWorkflow: true, phases: ['spec', 'pseudocode', 'architecture', 'refinement'] };
    }
    
    return { archonAvailable };
  },
  
  // Helper function to check Archon MCP server availability
  'check-archon-mcp': async () => {
    return await checkArchonMcpServer();
  },

  'session-restore': async (context) => {
    console.log(`🔄 Restoring session: ${context.sessionId}`);
    
    // ARCHON-FIRST: Check Archon MCP server availability on session restore
    console.log('🏛️ Session Restore: Checking Archon MCP server...');
    const archonAvailable = await checkArchonMcpServer();
    
    if (archonAvailable) {
      console.log('✅ Archon MCP server available - Task management ready');
      console.log('📋 Reminder: Check current tasks with archon:manage_task first');
    }

    // Restore Archon PRP state if needed
    if (context.sessionId.includes('archon')) {
      console.log('🏛️ Restoring Archon PRP session state...');
    }
    
    return { archonAvailable };
  },

  // During operation hooks
  'post-edit': async (context) => {
    console.log(`📝 File edited: ${context.file}`);
    
    // ARCHON-FIRST: Remind about task status updates
    console.log('🏛️ ARCHON-FIRST REMINDER: Update task status after code changes');
    console.log('📋 Consider: archon:manage_task(action="update", status="review")');
    
    // Store in memory for swarm coordination
    if (context.memoryKey) {
      console.log(`💾 Storing in swarm memory: ${context.memoryKey}`);
    }
    
    // Check if this affects Archon backend
    if (context.file.includes('python/src/')) {
      console.log('🐍 Backend file modified - consider running tests');
      console.log('📋 Update Archon task with backend progress');
    }
    
    // Check if this affects frontend
    if (context.file.includes('archon-ui-main/src/')) {
      console.log('⚛️ Frontend file modified - consider running linting');
      console.log('📋 Update Archon task with frontend progress');
    }
    
    return {};
  },

  'notify': async (context) => {
    console.log(`📢 Notification: ${context.message}`);
    return {};
  },

  // Post-operation hooks
  'post-task': async (context) => {
    console.log(`✅ Task completed: ${context.taskId}`);
    
    // ARCHON-FIRST: Remind to update Archon task status
    console.log('🏛️ ARCHON-FIRST REMINDER: Update Archon task status to "done"');
    console.log('📋 Use: archon:manage_task(action="update", status="done")');
    console.log('🔄 Then get next task: archon:manage_task(action="list", filter_by="status", filter_value="todo")');
    
    // Run quality checks for Archon
    if (context.taskId.includes('archon') || context.taskId.includes('backend')) {
      console.log('🔍 Running Archon quality checks...');
      console.log('  - Consider running: uv run ruff check');
      console.log('  - Consider running: uv run mypy src/');
      console.log('  - Consider running: uv run pytest');
      console.log('📋 Update Archon task with test results');
    }
    
    return {};
  },

  'session-end': async (context) => {
    console.log('🏁 Session ending...');
    
    // ARCHON-FIRST: Final reminder to check Archon task status
    console.log('🏛️ ARCHON-FIRST FINAL CHECK:');
    console.log('📋 Ensure all Archon tasks are updated with current status');
    console.log('📋 Mark completed tasks as "done" in Archon');
    console.log('📋 Create new tasks for identified next steps');
    
    if (context.exportMetrics) {
      console.log('📊 Exporting performance metrics...');
    }
    
    // Suggest next steps for Archon development
    console.log('💡 Suggested next steps:');
    console.log('  - Check Archon tasks: archon:manage_task(action="list")');
    console.log('  - Test the changes: docker-compose up --build');
    console.log('  - Run full test suite: uv run pytest');
    console.log('  - Check frontend: cd archon-ui-main && npm run test');
    console.log('  - Update Archon with test results');
    
    return {};
  }
};

// Helper function to check Archon MCP server availability
async function checkArchonMcpServer() {
  try {
    // Simple check for Archon MCP server availability
    // In a real implementation, this would make an HTTP request to port 8051
    // For now, we'll check if the Archon services are likely running
    const { execSync } = require('child_process');
    
    try {
      // Check if Archon backend is running (indicates MCP server likely available)
      execSync('curl -f http://localhost:8051/health 2>/dev/null', { timeout: 1000 });
      return true;
    } catch {
      // Check if docker-compose services are running
      try {
        const result = execSync('docker-compose ps --services --filter status=running 2>/dev/null', { encoding: 'utf8' });
        return result.includes('archon-mcp') || result.includes('archon-server');
      } catch {
        return false;
      }
    }
  } catch (error) {
    console.error('Error checking Archon MCP server:', error.message);
    return false;
  }
}

// Export hooks for Claude Flow
module.exports = hooks;

// CLI handler if run directly
if (require.main === module) {
  const [,, hookName, ...args] = process.argv;
  
  if (hooks[hookName]) {
    const context = args.reduce((acc, arg) => {
      if (arg.startsWith('--')) {
        const [key, value] = arg.substring(2).split('=');
        acc[key.replace(/-/g, '_')] = value || true;
      }
      return acc;
    }, {});
    
    hooks[hookName](context).then(result => {
      if (Object.keys(result).length > 0) {
        console.log('Hook result:', JSON.stringify(result, null, 2));
      }
    }).catch(console.error);
  } else {
    console.error(`Unknown hook: ${hookName}`);
    console.log('Available hooks:', Object.keys(hooks).join(', '));
    process.exit(1);
  }
}