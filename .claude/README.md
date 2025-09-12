# Claude Code Commands for Archon PRP

This directory contains custom Claude Code commands for the Archon PRP framework, enabling seamless task management and workflow continuation.

## Available Commands

### `/continue-task` (aliases: `/ct`, `/continue`)

Seamlessly transition work to new conversations when context reaches ~60% capacity. This command creates a comprehensive continuation prompt preserving all task and project context for zero-loss transitions.

#### Usage
```bash
/continue-task <task_id> [context_summary]
/continue-task --list
/continue-task --help
```

#### Examples
```bash
# Continue a specific task
/continue-task 517fa818-d76b-4c65-a2ee-4db963dc65a7

# Continue task with context summary  
/continue-task 517fa818-d76b-4c65-a2ee-4db963dc65a7 "Worked on GitHub integration, need testing"

# List available active tasks
/continue-task --list

# Show help
/continue-task --help
```

#### Features
- ✅ **Zero Context Loss**: Preserves all task and project details
- ✅ **Seamless Transition**: Ready-to-use commands for continuation
- ✅ **Proper Integration**: Follows Archon task management hierarchy
- ✅ **System Health Checks**: Verifies Archon server and MCP status
- ✅ **Error Recovery**: Robust validation and helpful error messages
- ✅ **Multiple Interfaces**: Available as both bash script and Claude command

## Command Structure

The commands are organized as follows:

```
.claude/
├── commands.json           # Command registry and configuration
├── package.json           # Node.js dependencies and metadata
├── commands/              # Individual command implementations
│   └── continue-task.js   # Continue task command
└── README.md             # This documentation
```

## Configuration

### Environment Variables

- `ARCHON_API`: Archon server URL (default: `http://localhost:8181`)

### Requirements

- Node.js 18+ for JavaScript commands
- Archon server running and accessible
- Valid Archon project with tasks

## Integration with Archon PRP

These commands integrate directly with the Archon PRP framework:

1. **Task Management**: Reads from Archon's task management system
2. **Project Context**: Gathers related project information
3. **API Integration**: Uses Archon REST API endpoints
4. **MCP Compatibility**: Follows MCP tool hierarchy guidelines

## Development

To add new commands:

1. Create a new `.js` file in the `commands/` directory
2. Follow the existing pattern for argument parsing and API integration
3. Add the command to `commands.json` with proper metadata
4. Test the command using Node.js directly
5. Update this README with documentation

## Bash Script Compatibility

The `/continue-task` functionality is also available as a bash script:

```bash
# Using bash script directly
./scripts/continue-task.sh <task_id> [context_summary]

# Using Claude command 
/continue-task <task_id> [context_summary]
```

Both interfaces provide identical functionality but the Claude command offers:
- Better error handling and colored output
- Integrated help system
- Consistent interface with other Claude commands
- Better integration with Claude Code workflow

## Support

For issues or questions:
- Check Archon server status: `curl http://localhost:8181/health`
- Verify task exists: `/continue-task --list`
- Review generated prompts: `.continue-task-prompt.md`
- Ensure proper CLAUDE.md configuration for task management hierarchy