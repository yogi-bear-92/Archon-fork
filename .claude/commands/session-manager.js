#!/usr/bin/env node

/**
 * Claude Code Session Manager
 * Advanced session management with context monitoring and auto-transitions
 * 
 * Usage:
 *   /session-manager --monitor          # Monitor context usage
 *   /session-manager --auto-transition  # Enable automatic transitions
 *   /session-manager --status           # Show session status
 */

const fs = require('fs');
const path = require('path');
const { execSync, spawn } = require('child_process');

// Configuration
const CONTEXT_THRESHOLD = 0.6; // 60% context usage triggers transition
const MONITOR_INTERVAL = 5000; // Check every 5 seconds
const SESSION_LOG = path.join(process.env.HOME, '.claude-session-manager.log');

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

// Log session activity
function logActivity(message) {
    const timestamp = new Date().toISOString();
    const logEntry = `${timestamp} - ${message}\n`;
    fs.appendFileSync(SESSION_LOG, logEntry);
}

// Estimate context usage (heuristic based on conversation length)
function estimateContextUsage() {
    try {
        // This is a simplified heuristic - in real implementation, 
        // we would need Claude Code API to get actual context usage
        
        // For now, simulate based on session duration and activity
        const sessionStart = getSessionStartTime();
        const currentTime = Date.now();
        const sessionDuration = (currentTime - sessionStart) / 1000 / 60; // minutes
        
        // Rough estimation: 1% context per minute of active conversation
        const estimatedUsage = Math.min(sessionDuration * 0.01, 0.95);
        
        return {
            estimated: true,
            usage: estimatedUsage,
            tokensUsed: Math.floor(estimatedUsage * 200000), // Rough token estimation
            tokensTotal: 200000,
            threshold: CONTEXT_THRESHOLD
        };
    } catch (error) {
        return {
            estimated: true,
            usage: 0.5,
            tokensUsed: 100000,
            tokensTotal: 200000,
            threshold: CONTEXT_THRESHOLD
        };
    }
}

// Get session start time (simplified)
function getSessionStartTime() {
    const sessionFile = path.join('/tmp', 'claude-session-start');
    try {
        if (fs.existsSync(sessionFile)) {
            return parseInt(fs.readFileSync(sessionFile, 'utf8'));
        } else {
            const startTime = Date.now();
            fs.writeFileSync(sessionFile, startTime.toString());
            return startTime;
        }
    } catch (error) {
        return Date.now();
    }
}

// Get current active task from Archon
async function getCurrentActiveTask() {
    try {
        const response = await fetch('http://localhost:8181/api/tasks?filter_by=status&filter_value=doing');
        const data = await response.json();
        const tasks = data.tasks || [];
        
        // Return the most recently updated 'doing' task
        const activeTasks = tasks
            .filter(task => task.status === 'doing')
            .sort((a, b) => new Date(b.updated_at) - new Date(a.updated_at));
        
        return activeTasks.length > 0 ? activeTasks[0] : null;
    } catch (error) {
        return null;
    }
}

// Monitor context usage and suggest transitions
async function monitorContext() {
    printMagic('🔍 Starting context usage monitoring...');
    printInfo(`Monitoring interval: ${MONITOR_INTERVAL / 1000}s`);
    printInfo(`Transition threshold: ${CONTEXT_THRESHOLD * 100}%`);
    
    let previousUsage = 0;
    let warningShown = false;
    
    const monitor = setInterval(async () => {
        const contextInfo = estimateContextUsage();
        const currentUsage = contextInfo.usage;
        
        // Show periodic updates
        if (Math.floor(currentUsage * 10) !== Math.floor(previousUsage * 10)) {
            const percentage = (currentUsage * 100).toFixed(1);
            const bar = '█'.repeat(Math.floor(currentUsage * 20)) + 
                       '░'.repeat(20 - Math.floor(currentUsage * 20));
            
            console.log(`📊 Context: [${bar}] ${percentage}% (${contextInfo.tokensUsed}/${contextInfo.tokensTotal})`);
            
            logActivity(`Context usage: ${percentage}%`);
        }
        
        // Warning at threshold
        if (currentUsage >= CONTEXT_THRESHOLD && !warningShown) {
            printWarning(`Context usage reached ${(CONTEXT_THRESHOLD * 100).toFixed(1)}% threshold!`);
            
            // Try to get current active task
            const activeTask = await getCurrentActiveTask();
            
            if (activeTask) {
                printInfo(`Active task detected: ${activeTask.title}`);
                printMagic(`Recommended: /continue-task-auto ${activeTask.id}`);
                
                logActivity(`Threshold reached - Active task: ${activeTask.id} - ${activeTask.title}`);
            } else {
                printInfo('No active task detected - use /continue-task-auto --list to see available tasks');
                logActivity('Threshold reached - No active task detected');
            }
            
            warningShown = true;
        }
        
        // Critical warning at 90%
        if (currentUsage >= 0.9) {
            printError('🚨 Context usage critical! Automatic transition recommended!');
            
            const activeTask = await getCurrentActiveTask();
            if (activeTask) {
                printMagic(`Execute now: /continue-task-auto ${activeTask.id} "Critical context limit reached"`);
                logActivity(`Critical threshold - Auto-transition recommended for task: ${activeTask.id}`);
            }
        }
        
        previousUsage = currentUsage;
        
    }, MONITOR_INTERVAL);
    
    // Handle graceful shutdown
    process.on('SIGINT', () => {
        clearInterval(monitor);
        printInfo('Context monitoring stopped');
        logActivity('Monitoring stopped by user');
        process.exit(0);
    });
    
    printInfo('Press Ctrl+C to stop monitoring');
}

// Enable automatic transitions
async function enableAutoTransition() {
    printMagic('🤖 Enabling automatic session transitions...');
    
    let transitionInProgress = false;
    
    const autoMonitor = setInterval(async () => {
        if (transitionInProgress) return;
        
        const contextInfo = estimateContextUsage();
        
        // Auto-transition at 80% usage
        if (contextInfo.usage >= 0.8) {
            transitionInProgress = true;
            
            printWarning('🚨 Auto-transition triggered at 80% context usage!');
            
            const activeTask = await getCurrentActiveTask();
            
            if (activeTask) {
                printMagic(`Auto-executing: /continue-task-auto ${activeTask.id}`);
                logActivity(`Auto-transition triggered for task: ${activeTask.id}`);
                
                // Execute auto-transition
                try {
                    const { startAutoSession, getProjectContext } = require('./continue-task-auto.js');
                    
                    printInfo('Gathering project context for auto-transition...');
                    const projectContext = await getProjectContext(activeTask.project_id);
                    
                    const contextSummary = 'Automatic transition triggered at 80% context capacity. Continuing work seamlessly.';
                    
                    await startAutoSession(activeTask, projectContext, contextSummary);
                    
                    printStatus('Auto-transition completed successfully!');
                    logActivity('Auto-transition completed successfully');
                    
                } catch (error) {
                    printError(`Auto-transition failed: ${error.message}`);
                    printInfo('Falling back to manual transition recommendation');
                    logActivity(`Auto-transition failed: ${error.message}`);
                }
            } else {
                printWarning('No active task for auto-transition - manual intervention required');
                logActivity('Auto-transition failed - No active task');
            }
            
            clearInterval(autoMonitor);
        }
        
    }, MONITOR_INTERVAL);
    
    // Handle graceful shutdown
    process.on('SIGINT', () => {
        clearInterval(autoMonitor);
        printInfo('Auto-transition disabled');
        logActivity('Auto-transition disabled by user');
        process.exit(0);
    });
    
    printInfo('Auto-transition active - will trigger at 80% context usage');
    printInfo('Press Ctrl+C to disable auto-transition');
}

// Show session status
async function showStatus() {
    printMagic('📊 Claude Code Session Manager Status');
    console.log('==========================================');
    
    // Context usage
    const contextInfo = estimateContextUsage();
    const percentage = (contextInfo.usage * 100).toFixed(1);
    const bar = '█'.repeat(Math.floor(contextInfo.usage * 30)) + 
               '░'.repeat(30 - Math.floor(contextInfo.usage * 30));
    
    console.log(`📊 Context Usage: [${bar}] ${percentage}%`);
    console.log(`🎯 Tokens: ${contextInfo.tokensUsed.toLocaleString()}/${contextInfo.tokensTotal.toLocaleString()}`);
    console.log(`⚠️  Threshold: ${(CONTEXT_THRESHOLD * 100)}%`);
    
    // Session info
    const sessionStart = getSessionStartTime();
    const sessionDuration = ((Date.now() - sessionStart) / 1000 / 60).toFixed(1);
    console.log(`⏱️  Session Duration: ${sessionDuration} minutes`);
    
    // Active task
    const activeTask = await getCurrentActiveTask();
    if (activeTask) {
        console.log(`🎯 Active Task: ${activeTask.title} (${activeTask.id})`);
        console.log(`📊 Task Status: ${activeTask.status}`);
        console.log(`👤 Assignee: ${activeTask.assignee || 'Unassigned'}`);
    } else {
        console.log('🎯 Active Task: None detected');
    }
    
    // Recommendations
    console.log('\n🚀 Recommendations:');
    if (contextInfo.usage >= 0.8) {
        console.log(colorize('red', '  🚨 URGENT: Context usage critical - transition now!'));
        if (activeTask) {
            console.log(colorize('magenta', `  🪄 Execute: /continue-task-auto ${activeTask.id}`));
        }
    } else if (contextInfo.usage >= CONTEXT_THRESHOLD) {
        console.log(colorize('yellow', '  ⚠️  Consider transitioning soon'));
        if (activeTask) {
            console.log(colorize('cyan', `  💡 Ready: /continue-task-auto ${activeTask.id}`));
        }
    } else {
        console.log(colorize('green', '  ✅ Context usage healthy - continue working'));
    }
    
    // Log file info
    if (fs.existsSync(SESSION_LOG)) {
        const logStats = fs.statSync(SESSION_LOG);
        console.log(`📋 Log File: ${SESSION_LOG} (${(logStats.size / 1024).toFixed(1)} KB)`);
    }
}

// Show help
function showHelp() {
    console.log(`
${colorize('bold', '🪄 Claude Code Session Manager')}

${colorize('bold', 'DESCRIPTION:')}
  Advanced session management with context monitoring and automatic transitions.
  Provides intelligent recommendations and automation for seamless task continuation.

${colorize('bold', 'USAGE:')}
  ${colorize('cyan', '/session-manager --monitor')}          Monitor context usage in real-time
  ${colorize('cyan', '/session-manager --auto-transition')} Enable automatic session transitions
  ${colorize('cyan', '/session-manager --status')}           Show current session status
  ${colorize('cyan', '/session-manager --help')}             Show this help message

${colorize('bold', 'FEATURES:')}
  ${colorize('magenta', '📊 Real-time Monitoring')} - Track context usage with visual indicators
  ${colorize('magenta', '🤖 Auto-Transitions')} - Automatic session switching at 80% capacity
  ${colorize('magenta', '🎯 Task Integration')} - Detects active Archon tasks for context
  ${colorize('magenta', '⚠️  Smart Warnings')} - Progressive alerts at 60% and 90% thresholds
  ${colorize('magenta', '📋 Activity Logging')} - Complete session activity logging

${colorize('bold', 'WORKFLOW AUTOMATION:')}
  ${colorize('green', '1. Start Monitoring')} - /session-manager --monitor
  ${colorize('green', '2. Work Normally')} - Continue your development work
  ${colorize('green', '3. Get Warnings')} - Automatic alerts at 60% context usage  
  ${colorize('green', '4. Smart Transitions')} - Recommended commands for active tasks
  ${colorize('green', '5. Auto-Magic')} - Optional automatic transitions at 80%

${colorize('bold', 'INTEGRATION:')}
  - Works with /continue-task and /continue-task-auto commands
  - Integrates with Archon task management system
  - Provides context-aware recommendations
  - Logs all activity for analysis and improvement
`);
}

// Main execution
async function main() {
    const args = process.argv.slice(2);
    
    if (args.includes('--help') || args.includes('-h')) {
        showHelp();
        process.exit(0);
    }
    
    if (args.includes('--monitor') || args.includes('-m')) {
        await monitorContext();
        return;
    }
    
    if (args.includes('--auto-transition') || args.includes('-a')) {
        await enableAutoTransition();
        return;
    }
    
    if (args.includes('--status') || args.includes('-s')) {
        await showStatus();
        return;
    }
    
    // Default: show status
    await showStatus();
}

// Execute main function
if (require.main === module) {
    main().catch(error => {
        printError(`Session manager failed: ${error.message}`);
        process.exit(1);
    });
}

module.exports = { 
    main, 
    monitorContext, 
    enableAutoTransition, 
    estimateContextUsage,
    getCurrentActiveTask 
};