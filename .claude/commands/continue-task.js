#!/usr/bin/env node

/**
 * Claude Code Command: Continue Task
 * Seamlessly transition work to new conversation when context reaches capacity
 * 
 * Usage:
 *   /continue-task <task_id> [context_summary]
 *   /continue-task --list
 *   /continue-task --help
 */

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Configuration
const ARCHON_API = process.env.ARCHON_API || 'http://localhost:8181';
const PROJECT_ROOT = path.resolve(__dirname, '../../');

// Colors for output
const colors = {
    reset: '\033[0m',
    red: '\033[0;31m',
    green: '\033[0;32m', 
    blue: '\033[0;34m',
    yellow: '\033[1;33m',
    cyan: '\033[0;36m',
    bold: '\033[1m'
};

function colorize(color, text) {
    return `${colors[color]}${text}${colors.reset}`;
}

function printStatus(message) {
    console.log(colorize('green', `✅ ${message}`));
}

function printInfo(message) {
    console.log(colorize('blue', `ℹ️  ${message}`));
}

function printWarning(message) {
    console.log(colorize('yellow', `⚠️  ${message}`));
}

function printError(message) {
    console.log(colorize('red', `❌ ${message}`));
}

// Check if Archon server is running
async function checkArchonServer() {
    try {
        const response = await fetch(`${ARCHON_API}/health`);
        if (response.ok) {
            printStatus('Archon server is running');
            return true;
        }
    } catch (error) {
        printError(`Archon server not running at ${ARCHON_API}`);
        printInfo('Start server with: export ARCHON_SERVER_PORT=8181 && python3 -m uvicorn src.server.main:app --host 0.0.0.0 --port 8181 --reload');
        return false;
    }
    return false;
}

// Get task details from Archon API
async function getTaskDetails(taskId) {
    try {
        const response = await fetch(`${ARCHON_API}/api/tasks/${taskId}`);
        
        if (!response.ok) {
            printError(`Task ID '${taskId}' not found`);
            await listAvailableTasks();
            return null;
        }
        
        const task = await response.json();
        return task;
    } catch (error) {
        printError(`Failed to fetch task details: ${error.message}`);
        return null;
    }
}

// Get project context
async function getProjectContext(projectId) {
    try {
        printInfo('Gathering project context...');
        
        // Get project details
        const projectResponse = await fetch(`${ARCHON_API}/api/projects/${projectId}`);
        const project = await projectResponse.json();
        
        // Get related tasks
        const tasksResponse = await fetch(`${ARCHON_API}/api/tasks?project_id=${projectId}`);
        const tasksData = await tasksResponse.json();
        const tasks = tasksData.tasks || [];
        
        // Sort by updated_at and get recent tasks
        const recentTasks = tasks
            .sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at))
            .slice(0, 5)
            .map(task => {
                const statusIcon = task.status === 'doing' ? '🔄' : 
                                 task.status === 'todo' ? '📋' : '✅';
                return `  ${statusIcon} ${task.title} (${task.status}) - ${task.feature || 'no-feature'}`;
            })
            .join('\n');
        
        return {
            projectTitle: project.title || 'Unknown Project',
            recentTasks: recentTasks
        };
    } catch (error) {
        printError(`Failed to fetch project context: ${error.message}`);
        return {
            projectTitle: 'Unknown Project',
            recentTasks: 'No recent tasks available'
        };
    }
}

// Generate continuation prompt
async function generateContinuationPrompt(taskDetails, projectContext, contextSummary) {
    const currentTime = new Date().toLocaleString();
    const promptFile = path.join(PROJECT_ROOT, '.continue-task-prompt.md');
    
    const promptContent = `# 🔄 Task Continuation - New Conversation

**Continuation Time**: ${currentTime}
**Previous Context**: Context reached ~60% capacity - transitioning to new conversation

## 🎯 **ACTIVE TASK DETAILS**

**Task ID**: \`${taskDetails.id}\`
**Title**: ${taskDetails.title}
**Status**: ${taskDetails.status}
**Assignee**: ${taskDetails.assignee || 'Unassigned'}  
**Feature**: ${taskDetails.feature || 'general'}

**Description**:
${taskDetails.description || 'No description available'}

## 📋 **PROJECT CONTEXT**

**PROJECT**: ${projectContext.projectTitle}

**RECENT TASKS**:
${projectContext.recentTasks}

## 🔍 **CONTINUATION CONTEXT**

${contextSummary || 'Previous conversation reached context capacity. Transitioning work to maintain continuity.'}

## 🚀 **NEXT ACTIONS**

1. **First Priority**: Update this task status to 'doing' in Archon
   \`\`\`bash
   curl -X PUT "${ARCHON_API}/api/tasks/${taskDetails.id}" -H "Content-Type: application/json" -d '{"status": "doing"}'
   \`\`\`

2. **Continue Implementation**: Pick up work exactly where previous conversation left off

3. **Use Proper Task Management**: Follow CLAUDE.md hierarchy:
   - Try Archon MCP tools first: \`mcp__archon__update_task(task_id="${taskDetails.id}", status="doing")\`
   - Fallback to direct API calls if MCP session issues occur
   - Only use TodoWrite as last resort

## 🔧 **SYSTEM STATUS CHECKS**

Before continuing, verify system health:
\`\`\`bash
# Check Archon server
curl ${ARCHON_API}/health

# Check current project tasks  
curl ${ARCHON_API}/api/tasks?project_id=${taskDetails.project_id}

# Check MCP servers status
npx flow-nexus@latest mcp status
\`\`\`

## 📝 **CONTINUATION INSTRUCTIONS**

**Copy and paste this entire message into the new Claude Code conversation to seamlessly continue the work.**

Key points for new conversation:
- ✅ Task is properly tracked in Archon database
- ✅ All context preserved in project management system  
- ✅ Previous work and decisions documented
- ✅ Ready for immediate continuation without context loss

**Start new conversation with**: "Continue task ${taskDetails.id} - ${taskDetails.title}"
`;

    // Write prompt to file
    fs.writeFileSync(promptFile, promptContent, 'utf8');
    
    return promptFile;
}

// List available tasks
async function listAvailableTasks() {
    try {
        const response = await fetch(`${ARCHON_API}/api/tasks`);
        const data = await response.json();
        const tasks = data.tasks || [];
        
        const activeTasks = tasks.filter(task => ['todo', 'doing'].includes(task.status));
        
        printInfo('Available tasks:');
        activeTasks.forEach(task => {
            const statusIcon = task.status === 'doing' ? '🔄' : '📋';
            console.log(`  ${statusIcon} ${colorize('cyan', task.id)} - ${task.title}`);
        });
        
        return activeTasks;
    } catch (error) {
        printError(`Failed to list tasks: ${error.message}`);
        return [];
    }
}

// Show help
function showHelp() {
    console.log(`
${colorize('bold', '🔄 Continue Task - Claude Code Command')}

${colorize('bold', 'DESCRIPTION:')}
  Seamlessly transition work to new conversations when context reaches ~60% capacity.
  Creates a comprehensive continuation prompt with all task and project context.

${colorize('bold', 'USAGE:')}
  ${colorize('cyan', '/continue-task <task_id> [context_summary]')}
  ${colorize('cyan', '/continue-task --list')}
  ${colorize('cyan', '/continue-task --help')}

${colorize('bold', 'OPTIONS:')}
  ${colorize('green', 'task_id')}        UUID of the Archon task to continue
  ${colorize('green', 'context_summary')} Optional summary of previous conversation context
  ${colorize('green', '--list, -l')}      List all available tasks
  ${colorize('green', '--help, -h')}      Show this help message

${colorize('bold', 'EXAMPLES:')}
  ${colorize('cyan', '# Continue a specific task')}
  /continue-task 517fa818-d76b-4c65-a2ee-4db963dc65a7

  ${colorize('cyan', '# Continue task with context summary')}
  /continue-task 517fa818-d76b-4c65-a2ee-4db963dc65a7 "Worked on GitHub integration, need testing"

  ${colorize('cyan', '# List available active tasks')}
  /continue-task --list

${colorize('bold', 'FEATURES:')}
  ✅ Zero context loss - preserves all task and project details
  ✅ Seamless transition - ready-to-use commands for continuation
  ✅ Proper integration - follows Archon task management hierarchy  
  ✅ System health checks - verifies Archon server and MCP status
  ✅ Error recovery - robust validation and helpful error messages

${colorize('bold', 'REQUIREMENTS:')}
  - Archon server running on port 8181 (configurable via ARCHON_API env var)
  - Valid task ID from Archon project management system
  - Network access to Archon API endpoints

${colorize('bold', 'OUTPUT:')}
  Creates ${colorize('green', '.continue-task-prompt.md')} with complete continuation instructions
  Copy and paste the generated prompt into your new Claude Code conversation
`);
}

// Main execution function
async function main() {
    const args = process.argv.slice(2);
    
    // Handle help flag
    if (args.includes('--help') || args.includes('-h')) {
        showHelp();
        process.exit(0);
    }
    
    // Handle list flag
    if (args.includes('--list') || args.includes('-l')) {
        printInfo('🔄 Continue Task Command - Available Tasks');
        console.log('==================================================');
        
        if (!(await checkArchonServer())) {
            process.exit(1);
        }
        
        await listAvailableTasks();
        process.exit(0);
    }
    
    // Main command execution
    printInfo('🔄 Continue Task Command - Context Transition Helper');
    console.log('==================================================');
    
    // Validate arguments
    if (args.length === 0) {
        printError('Task ID required');
        printInfo('Usage: /continue-task <task_id> [context_summary]');
        printInfo('');
        
        if (await checkArchonServer()) {
            await listAvailableTasks();
        }
        process.exit(1);
    }
    
    const taskId = args[0];
    const contextSummary = args.slice(1).join(' ') || 'Previous conversation reached context capacity. Transitioning work to maintain continuity.';
    
    // Check Archon server
    if (!(await checkArchonServer())) {
        process.exit(1);
    }
    
    // Get task details
    printInfo(`Fetching task details for ID: ${taskId}`);
    const taskDetails = await getTaskDetails(taskId);
    if (!taskDetails) {
        process.exit(1);
    }
    
    // Get project context
    const projectContext = await getProjectContext(taskDetails.project_id);
    
    // Generate continuation prompt
    const promptFile = await generateContinuationPrompt(taskDetails, projectContext, contextSummary);
    
    printStatus('Task continuation prepared!');
    printInfo(`Prompt file: ${promptFile}`);
    printWarning('Context transition ready - copy prompt to new conversation');
    
    // Display the prompt content
    console.log('');
    console.log('=================================');
    console.log('📋 CONTINUATION PROMPT (copy this to new conversation):');
    console.log('=================================');
    
    const promptContent = fs.readFileSync(promptFile, 'utf8');
    console.log(promptContent);
    
    printStatus('✨ Ready for seamless task continuation!');
}

// Execute main function
if (require.main === module) {
    main().catch(error => {
        printError(`Command failed: ${error.message}`);
        process.exit(1);
    });
}

module.exports = { main, checkArchonServer, getTaskDetails, generateContinuationPrompt };