#!/usr/bin/env node

/**
 * Test script for Claude Code commands
 */

const { exec } = require('child_process');
const path = require('path');

const COMMANDS_DIR = path.join(__dirname, 'commands');

// Colors for output
const colors = {
    reset: '\033[0m',
    red: '\033[0;31m',
    green: '\033[0;32m',
    blue: '\033[0;34m',
    yellow: '\033[1;33m'
};

function colorize(color, text) {
    return `${colors[color]}${text}${colors.reset}`;
}

function runTest(description, command, expectedPattern = null) {
    return new Promise((resolve) => {
        console.log(colorize('blue', `\n🧪 Testing: ${description}`));
        console.log(colorize('yellow', `   Command: ${command}`));
        
        exec(command, { cwd: __dirname }, (error, stdout, stderr) => {
            if (error && !expectedPattern) {
                console.log(colorize('red', `   ❌ FAILED: ${error.message}`));
                resolve(false);
                return;
            }
            
            if (expectedPattern) {
                const output = stdout + stderr;
                if (output.includes(expectedPattern)) {
                    console.log(colorize('green', `   ✅ PASSED: Found expected pattern "${expectedPattern}"`));
                    resolve(true);
                } else {
                    console.log(colorize('red', `   ❌ FAILED: Expected pattern "${expectedPattern}" not found`));
                    console.log(colorize('yellow', `   Output: ${output.substring(0, 200)}...`));
                    resolve(false);
                }
            } else {
                console.log(colorize('green', `   ✅ PASSED: Command executed successfully`));
                resolve(true);
            }
        });
    });
}

async function main() {
    console.log(colorize('blue', '🔄 Testing Claude Code Commands'));
    console.log('================================================');
    
    const tests = [
        {
            description: 'continue-task help command',
            command: `node ${path.join(COMMANDS_DIR, 'continue-task.js')} --help`,
            expectedPattern: 'Continue Task - Claude Code Command'
        },
        {
            description: 'continue-task list command (requires Archon server)',
            command: `node ${path.join(COMMANDS_DIR, 'continue-task.js')} --list`,
            expectedPattern: 'Available tasks:'
        },
        {
            description: 'ct alias help command',
            command: `node ${path.join(COMMANDS_DIR, 'ct.js')} --help`,
            expectedPattern: 'Continue Task - Claude Code Command'
        },
        {
            description: 'continue alias help command',
            command: `node ${path.join(COMMANDS_DIR, 'continue.js')} --help`,
            expectedPattern: 'Continue Task - Claude Code Command'
        }
    ];
    
    let passed = 0;
    let total = tests.length;
    
    for (const test of tests) {
        const result = await runTest(test.description, test.command, test.expectedPattern);
        if (result) passed++;
        
        // Add small delay between tests
        await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    console.log('\n================================================');
    console.log(colorize('blue', `📊 Test Results: ${passed}/${total} passed`));
    
    if (passed === total) {
        console.log(colorize('green', '🎉 All tests passed!'));
        process.exit(0);
    } else {
        console.log(colorize('red', `❌ ${total - passed} tests failed`));
        process.exit(1);
    }
}

if (require.main === module) {
    main().catch(error => {
        console.error(colorize('red', `Test runner failed: ${error.message}`));
        process.exit(1);
    });
}