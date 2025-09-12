#!/usr/bin/env node

/**
 * Claude Code Command: Continue Task (Auto-Session)
 * Automatically starts new Claude session with continuation context
 * 
 * Usage:
 *   /continue-task-auto <task_id> [context_summary]
 *   /continue-task-auto --list
 *   /continue-task-auto --help
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');

// Import base functionality from continue-task.js
const { checkArchonServer, getTaskDetails, generateContinuationPrompt } = require('./continue-task.js');

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
    bold: '\033[1m',
    magenta: '\033[0;35m'
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

function printMagic(message) {
    console.log(colorize('magenta', `🪄 ${message}`));
}

// Get project context (reuse from original)
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

// Generate enhanced continuation prompt for auto-session
async function generateAutoSessionPrompt(taskDetails, projectContext, contextSummary) {
    const currentTime = new Date().toLocaleString();
    
    const promptContent = `Continue task ${taskDetails.id} - ${taskDetails.title}

# 🔄 Auto-Session Task Continuation

**Continuation Time**: ${currentTime}
**Previous Context**: ${contextSummary}
**Auto-Session**: New Claude Code session automatically initiated

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

## 🚀 **IMMEDIATE NEXT ACTIONS**

1. **Update task status to 'doing'** using Archon MCP:
   Use: \`mcp__archon__update_task(task_id="${taskDetails.id}", status="doing")\`

2. **Continue implementation** from where previous session ended

3. **Follow proper task hierarchy**: Archon MCP → Direct API → TodoWrite

## 💡 **AUTO-SESSION FEATURES ACTIVE**

- ✅ Zero context loss preservation
- ✅ Automatic session management
- ✅ Proper task integration
- ✅ System health verified
- 🆕 **Enhanced**: Auto-started new session
- 🆕 **Enhanced**: Previous session context preserved

Ready to continue work seamlessly!`;

    return promptContent;
}

// Check Claude CLI availability
function checkClaudeCLI() {
    try {
        execSync('which claude', { stdio: 'ignore' });
        return true;
    } catch (error) {
        return false;
    }
}

// Start new Claude session with continuation prompt
async function startAutoSession(taskDetails, projectContext, contextSummary) {
    try {
        printMagic('Starting new Claude Code session...');
        
        // Generate the continuation prompt
        const prompt = await generateAutoSessionPrompt(taskDetails, projectContext, contextSummary);
        
        // Write prompt to temporary file
        const tempPromptFile = path.join('/tmp', `continue-task-${taskDetails.id}.txt`);
        fs.writeFileSync(tempPromptFile, prompt, 'utf8');
        
        printStatus(`Continuation prompt saved: ${tempPromptFile}`);
        printMagic('Launching new Claude session with task context...');
        
        // Start Claude with the continuation prompt
        const claudeProcess = spawn('claude', [prompt], {
            stdio: 'inherit',
            detached: false
        });
        
        // Handle process events
        claudeProcess.on('close', (code) => {
            if (code === 0) {
                printStatus('Claude session completed successfully');
            } else {
                printWarning(`Claude session exited with code ${code}`);
            }
            
            // Clean up temp file
            if (fs.existsSync(tempPromptFile)) {
                fs.unlinkSync(tempPromptFile);
            }
        });
        
        claudeProcess.on('error', (error) => {
            printError(`Failed to start Claude session: ${error.message}`);
        });
        
        return claudeProcess;
        
    } catch (error) {
        printError(`Auto-session failed: ${error.message}`);
        return null;
    }
}

// Show help
function showHelp() {
    console.log(`
${colorize('bold', '🪄 Continue Task Auto-Session - Claude Code Command')}

${colorize('bold', 'DESCRIPTION:')}
  ${colorize('magenta', '🆕 ENHANCED:')} Automatically starts new Claude Code session with task continuation context.
  Eliminates manual copy/paste - seamlessly transitions between conversations!

${colorize('bold', 'USAGE:')}
  ${colorize('cyan', '/continue-task-auto <task_id> [context_summary]')}
  ${colorize('cyan', '/continue-task-auto --list')}
  ${colorize('cyan', '/continue-task-auto --help')}

${colorize('bold', 'AUTO-SESSION FEATURES:')}
  ${colorize('magenta', '🪄 Automatic Session Start')} - Launches new Claude Code session
  ${colorize('magenta', '🔄 Context Preservation')} - Transfers all task and project context  
  ${colorize('magenta', '⚡ Zero Manual Steps')} - No copy/paste required
  ${colorize('magenta', '🎯 Task Integration')} - Proper Archon MCP integration ready
  ${colorize('magenta', '🛡️ Error Recovery')} - Fallback to manual mode if needed

${colorize('bold', 'EXAMPLES:')}
  ${colorize('cyan', '# Auto-start session for specific task')}
  /continue-task-auto 517fa818-d76b-4c65-a2ee-4db963dc65a7

  ${colorize('cyan', '# Auto-start with context summary')}
  /continue-task-auto 517fa818-d76b-4c65-a2ee-4db963dc65a7 "GitHub integration complete, need tests"

  ${colorize('cyan', '# List available tasks')}
  /continue-task-auto --list

${colorize('bold', 'WORKFLOW:')}
  ${colorize('green', '1. Context Warning')} - Current session approaching capacity
  ${colorize('green', '2. Auto-Command')} - Run /continue-task-auto <task-id>
  ${colorize('green', '3. Magic Happens')} - New session starts automatically
  ${colorize('green', '4. Keep Working')} - Continue seamlessly with full context

${colorize('bold', 'REQUIREMENTS:')}
  - Claude CLI available (${checkClaudeCLI() ? colorize('green', '✅ Available') : colorize('red', '❌ Missing')})
  - Archon server running on port 8181
  - Valid task ID from Archon project management system

${colorize('bold', 'FALLBACK:')}
  If auto-session fails, automatically falls back to manual mode (/continue-task)
`);
}

// List available tasks (reuse from original)
async function listAvailableTasks() {
    try {
        const response = await fetch(`${ARCHON_API}/api/tasks`);
        const data = await response.json();
        const tasks = data.tasks || [];
        
        const activeTasks = tasks.filter(task => ['todo', 'doing'].includes(task.status));
        
        printInfo('Available tasks for auto-session:');
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
        printMagic('🪄 Continue Task Auto-Session - Available Tasks');
        console.log('====================================================');
        
        if (!(await checkArchonServer())) {
            process.exit(1);
        }
        
        await listAvailableTasks();
        process.exit(0);
    }
    
    // Main auto-session execution
    printMagic('🪄 Continue Task Auto-Session - Magic Session Transition');
    console.log('=======================================================');
    
    // Check Claude CLI availability
    if (!checkClaudeCLI()) {
        printError('Claude CLI not available! Falling back to manual mode...');
        printInfo('Install Claude CLI or use /continue-task for manual mode');
        
        // Import and run original continue-task as fallback
        const originalCommand = require('./continue-task.js');
        return await originalCommand.main();
    }
    
    printStatus('Claude CLI available - auto-session mode ready!');
    
    // Validate arguments
    if (args.length === 0) {
        printError('Task ID required');
        printInfo('Usage: /continue-task-auto <task_id> [context_summary]');
        printInfo('');
        
        if (await checkArchonServer()) {
            await listAvailableTasks();
        }
        process.exit(1);
    }
    
    const taskId = args[0];
    const contextSummary = args.slice(1).join(' ') || 'Previous conversation reached context capacity. Auto-transitioning to new session.';
    
    // Check Archon server
    if (!(await checkArchonServer())) {
        process.exit(1);
    }
    
    // Get task details
    printInfo(`Fetching task details for auto-session: ${taskId}`);
    const taskDetails = await getTaskDetails(taskId);
    if (!taskDetails) {
        process.exit(1);
    }
    
    // Get project context
    const projectContext = await getProjectContext(taskDetails.project_id);
    
    printStatus('All context gathered - ready for auto-session magic!');
    
    // Start automatic session
    const session = await startAutoSession(taskDetails, projectContext, contextSummary);
    
    if (session) {
        printMagic('🎉 Auto-session started successfully!');
        printInfo('New Claude Code conversation active with full task context');
        
        // Note: The current process will continue running until the Claude session ends
        // This allows the user to see the transition happening
    } else {
        printWarning('Auto-session failed, falling back to manual mode...');
        
        // Fallback to original continue-task
        const originalCommand = require('./continue-task.js');
        return await originalCommand.main();
    }
}

// Execute main function
if (require.main === module) {
    main().catch(error => {
        printError(`Auto-session command failed: ${error.message}`);
        
        // Final fallback
        console.log(colorize('yellow', 'Attempting fallback to manual continue-task...'));
        try {
            const originalCommand = require('./continue-task.js');
            originalCommand.main();
        } catch (fallbackError) {
            printError(`Fallback also failed: ${fallbackError.message}`);
            process.exit(1);
        }
    });
}

module.exports = { main, startAutoSession, checkClaudeCLI };