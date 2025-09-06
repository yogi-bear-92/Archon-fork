# Continue Task Command - Usage Guide

The `continue-task` command enables seamless transition of work between Claude Code conversations when context reaches capacity (~60%). It preserves all task and project context for zero-loss continuity.

## Available Interfaces

### 1. Claude Code Command (Recommended)
```bash
# Primary usage
/continue-task <task_id> [context_summary]

# List available tasks
/continue-task --list

# Show help
/continue-task --help
```

### 2. Command Aliases
```bash
/ct <task_id> [context_summary]      # Short alias
/continue <task_id> [context_summary] # Descriptive alias
```

### 3. Bash Script (Alternative)
```bash
./scripts/continue-task.sh <task_id> [context_summary]
```

## Usage Examples

### Basic Task Continuation
```bash
# Continue specific task
/continue-task 517fa818-d76b-4c65-a2ee-4db963dc65a7

# With custom context summary
/continue-task 517fa818-d76b-4c65-a2ee-4db963dc65a7 "Completed GitHub integration implementation, ready for testing"
```

### Discovery and Management
```bash
# List all available active tasks
/continue-task --list

# Get comprehensive help
/continue-task --help

# Using short alias
/ct --list
```

### Error Handling
```bash
# Invalid task ID - shows available tasks
/continue-task invalid-task-id

# No arguments - shows usage and available tasks
/continue-task
```

## Generated Output

The command creates a comprehensive continuation prompt that includes:

### 🎯 Task Details
- **Task ID**: Exact UUID for API calls
- **Title**: Full task title for context
- **Status**: Current status (todo/doing/review/done)
- **Assignee**: Who is responsible for the task
- **Feature**: Feature label for grouping
- **Description**: Complete task description with acceptance criteria

### 📋 Project Context
- **Project Name**: Full project title
- **Recent Tasks**: 5 most recent related tasks with status icons
- **Task Relationships**: Understanding of task dependencies

### 🚀 Next Actions
1. **Priority Commands**: Exact API calls to update task status
2. **Tool Hierarchy**: Proper MCP → API → TodoWrite fallback sequence
3. **System Checks**: Health verification commands

### 🔧 System Integration
- **API Endpoints**: Direct curl commands for task management
- **MCP Integration**: Archon MCP tool usage examples  
- **Status Verification**: Commands to check system health

## Workflow Integration

### Typical Usage Pattern

1. **Context Warning** (~60% context used)
   ```
   User notices conversation getting lengthy
   ```

2. **Task Continuation**
   ```bash
   /continue-task <current-task-id> "Brief summary of current state"
   ```

3. **Copy Generated Prompt**
   ```
   Command outputs complete continuation instructions
   ```

4. **New Conversation**
   ```
   Paste prompt into fresh Claude Code conversation
   ```

5. **Seamless Resume**
   ```
   Work continues with full context preserved
   ```

### Best Practices

#### Context Summary Guidelines
- **Be Specific**: "Implemented GitHub webhook handling, need to add tests"
- **Mention Blockers**: "Waiting for API key, can work on documentation"
- **Note Decisions**: "Chose FastAPI over Flask, database schema finalized"
- **Highlight Progress**: "Core features complete, working on error handling"

#### When to Use
- Context approaching 60-70% capacity
- Before switching between major implementation phases
- When conversation becomes difficult to follow
- Before taking breaks in long development sessions

## System Requirements

### Dependencies
- **Archon Server**: Running on port 8181 (configurable via `ARCHON_API`)
- **Node.js**: Version 18+ for Claude commands
- **Network Access**: To Archon API endpoints
- **Valid Tasks**: Active tasks in Archon project management system

### Configuration
```bash
# Optional: Custom Archon server URL
export ARCHON_API="http://localhost:8181"

# Check server health
curl http://localhost:8181/health
```

## Troubleshooting

### Common Issues

#### "Archon server not running"
```bash
# Start Archon server
export ARCHON_SERVER_PORT=8181
python3 -m uvicorn src.server.main:app --host 0.0.0.0 --port 8181 --reload
```

#### "Task ID not found"
```bash
# List available tasks
/continue-task --list

# Use exact UUID from list
```

#### "No available tasks"
```bash
# Check project has active tasks
curl http://localhost:8181/api/tasks

# Create tasks via Archon MCP or API
```

### Debug Commands
```bash
# Test command functionality
node .claude/test-commands.js

# Verify API access
curl http://localhost:8181/health

# Check command files
ls -la .claude/commands/
```

## Advanced Usage

### Environment Customization
```bash
# Use different Archon server
ARCHON_API="http://remote-server:8181" /continue-task <task_id>

# Custom output file location  
ARCHON_API="http://localhost:8181" /continue-task <task_id> > custom-prompt.md
```

### Integration with CI/CD
```bash
# Automated task continuation in scripts
if [ "$CONTEXT_USAGE" -gt 60 ]; then
    /continue-task "$CURRENT_TASK_ID" "Automated context transition"
fi
```

### Team Workflows
- **Handoffs**: Use continue-task for developer transitions
- **Code Reviews**: Preserve context when switching reviewers  
- **Documentation**: Maintain context across documentation sessions
- **Testing**: Continue work across test development phases

## Output Files

### `.continue-task-prompt.md`
- **Location**: Project root directory
- **Content**: Complete continuation instructions
- **Usage**: Copy/paste into new Claude Code conversation
- **Format**: Markdown with proper formatting for readability

### Example Output Structure
```markdown
# 🔄 Task Continuation - New Conversation

## 🎯 ACTIVE TASK DETAILS
[Complete task information]

## 📋 PROJECT CONTEXT  
[Project and related tasks]

## 🔍 CONTINUATION CONTEXT
[User-provided summary]

## 🚀 NEXT ACTIONS
[Exact commands to continue]

## 🔧 SYSTEM STATUS CHECKS
[Health verification commands]

## 📝 CONTINUATION INSTRUCTIONS
[Step-by-step guidance]
```

This command ensures zero context loss and seamless task continuation across Claude Code conversations! 🚀