# Claude Hook Integration System

## Overview

The Claude Hook Integration System enables automatic task detection and creation from user messages in Claude Code conversations. When a user sends a message that contains actionable work items, the system automatically:

1. **Analyzes the message** using AI to detect potential tasks
2. **Evaluates task confidence** to determine auto-creation vs suggestion
3. **Creates tasks in Archon projects** with proper metadata and context
4. **Provides feedback** on what was created or suggested

## Architecture

```
Claude Code User Message
        ↓
  Post-User-Prompt Hook
        ↓
  Task Detection Service (AI Analysis)
        ↓
  Archon MCP Coordinator (Task Creation)
        ↓
  Project Management System
```

## Components

### 1. Post-User-Prompt Hook (`src/hooks/post_user_prompt_hook.py`)

**Purpose**: Entry point that receives user messages from Claude Code and triggers the analysis pipeline.

**Key Features**:
- Command-line script executed by Claude Code after each user prompt
- Accepts user message, conversation context, and project context as arguments
- Outputs JSON results for Claude Code to parse
- Handles errors gracefully with proper logging

**Usage**:
```bash
python3 src/hooks/post_user_prompt_hook.py "User message" "context" "project"
```

### 2. Task Detection Service (`src/server/services/task_detection_service.py`)

**Purpose**: AI-powered analysis of user messages to identify actionable tasks.

**Key Features**:
- Uses GPT-4o-mini for fast, cost-effective task detection
- Sophisticated prompt engineering with confidence scoring
- Caching to avoid re-analyzing identical messages
- Fallback heuristic analysis when AI is unavailable
- Structured output with task metadata (title, description, confidence, urgency, etc.)

**Detection Criteria**:
- **High Confidence (0.8-1.0)**: Explicit requests with clear deliverables
- **Medium Confidence (0.5-0.7)**: Implied tasks or problems needing solutions
- **Low Confidence (0.2-0.4)**: Discussions that might lead to tasks later
- **No Task (0.0-0.1)**: Pure questions, explanations, or casual conversation

### 3. Hook Integration Service (`src/server/services/hook_integration_service.py`)

**Purpose**: Coordination layer between Claude Code hooks and the Archon system.

**Key Features**:
- Programmatic hook execution for testing and API access
- Hook configuration management
- Performance tracking and execution history
- Integration testing and validation
- Background execution support

### 4. Hook Management API (`src/server/api_routes/hook_management_api.py`)

**Purpose**: REST API endpoints for managing and testing the hook system.

**Key Endpoints**:
- `GET /api/hooks/` - Get hook configuration
- `POST /api/hooks/test` - Test hook execution with sample messages
- `POST /api/hooks/enable/{hook_name}` - Enable a specific hook
- `POST /api/hooks/disable/{hook_name}` - Disable a specific hook
- `GET /api/hooks/history` - Get execution history
- `GET /api/hooks/test-integration` - Run integration tests

### 5. Hook Configuration (`.claude-hooks.json`)

**Purpose**: Configuration file that tells Claude Code when and how to execute hooks.

**Key Settings**:
- **Conditions**: Message length requirements, pattern matching for inclusion/exclusion
- **Execution**: Timeout, background execution, logging preferences
- **Environment**: Working directory, environment variables, Python path

## Configuration

### Hook Configuration File (`.claude-hooks.json`)

```json
{
  "hooks": {
    "post-user-prompt": {
      "description": "Analyze user messages for potential tasks and auto-create them in Archon projects",
      "command": "python3 src/hooks/post_user_prompt_hook.py",
      "timeout": 30000,
      "enabled": true,
      "pass_user_message": true,
      "pass_conversation_context": true,
      "background": true,
      "log_output": true,
      "conditions": {
        "min_message_length": 10,
        "exclude_patterns": [
          "^(yes|no|ok|thanks?)$",
          "^\\s*$",
          "^[.!?]+$"
        ],
        "include_patterns": [
          "\\b(implement|create|build|add|fix|develop|code|task|feature|bug|issue)\\b",
          "\\b(can you|could you|please|need to|should|must|want to)\\b",
          "\\b(help me|I need|let's)\\b"
        ]
      }
    }
  },
  "global_settings": {
    "working_directory": "/path/to/archon/python",
    "environment_variables": {
      "PYTHONPATH": "/path/to/archon/python/src",
      "TASK_AUTO_CREATION_THRESHOLD": "0.7",
      "HOOK_LOGGING_ENABLED": "true"
    }
  }
}
```

### Environment Variables

- `TASK_AUTO_CREATION_THRESHOLD`: Confidence threshold for automatic task creation (default: 0.7)
- `HOOK_LOGGING_ENABLED`: Enable detailed logging for hook execution
- `URL_DETECTION_ENABLED`: Enable URL detection middleware

## Usage Examples

### 1. Automatic Task Creation

**User Message**: "Can you implement user authentication with JWT tokens and password hashing?"

**System Response**:
- Detects high-confidence task (0.9)
- Auto-creates task: "Implement user authentication system"
- Assigns to appropriate agent (e.g., "Backend Developer")
- Categorizes as "feature" with "high" urgency

### 2. Task Suggestions

**User Message**: "The login page seems slow, we might want to optimize it"

**System Response**:
- Detects medium-confidence task (0.6)
- Stores as suggestion: "Optimize login page performance" 
- Provides feedback but doesn't auto-create
- Available for user review via API

### 3. No Task Detection

**User Message**: "What's the weather like today?"

**System Response**:
- Correctly identifies as non-actionable (0.1 confidence)
- No tasks created or suggested
- Minimal processing overhead

## API Usage

### Test Hook Execution

```bash
curl -X POST "http://localhost:8080/api/hooks/test" \
     -H "Content-Type: application/json" \
     -d '{
       "user_message": "Please create a REST API for user management",
       "conversation_context": "Planning new features",
       "project_context": "Web application development"
     }'
```

### Get Hook Status

```bash
curl "http://localhost:8080/api/hooks/"
```

### Run Integration Tests

```bash
curl "http://localhost:8080/api/hooks/test-integration"
```

## Testing

### Manual Testing

Run the comprehensive test script:

```bash
cd /path/to/archon/python
python3 test_hook_integration.py
```

The test script validates:
1. AI task detection with various message types
2. Archon MCP integration and project management
3. Hook execution system and configuration
4. End-to-end workflow from message to task creation

### API Testing

Use the Hook Management API to test individual components:

```bash
# Test specific message
curl -X POST "http://localhost:8080/api/hooks/test" \
     -H "Content-Type: application/json" \
     -d '{"user_message": "Build a dashboard with charts"}'

# Check execution history
curl "http://localhost:8080/api/hooks/history"

# Run system integration tests
curl "http://localhost:8080/api/hooks/test-integration"
```

## Performance and Optimization

### Caching Strategy

- **Task Detection**: Caches analysis results based on message hash
- **Semantic Analysis**: Leverages Serena's intelligent caching
- **Execution History**: In-memory cache with cleanup mechanisms

### Resource Management

- **Lightweight Execution**: Hook runs in background, doesn't block user interaction
- **Fast AI Model**: Uses GPT-4o-mini for quick analysis (typically <2 seconds)
- **Graceful Degradation**: Falls back to heuristic analysis if AI unavailable
- **Memory Efficiency**: Automatic cleanup of old cache entries

### Confidence Thresholds

- **Auto-Creation**: Tasks with confidence ≥ 0.7 are automatically created
- **Suggestions**: Tasks with confidence 0.4-0.7 are stored as suggestions
- **Filtering**: Tasks below 0.4 confidence are discarded to reduce noise

## Integration with Archon Components

### Project Management

- **Automatic Project Detection**: Finds existing projects or creates default ones
- **Task Categorization**: Uses feature labels and assignee suggestions
- **Priority Mapping**: Converts urgency levels to task order priorities
- **Metadata Enrichment**: Adds context, reasoning, and source information

### Real-time Updates

- **Socket.IO Integration**: Broadcasts task creation events to connected clients
- **Progress Tracking**: Updates project progress when tasks are auto-created
- **Notification System**: Alerts users about automatic task creation

### Memory and Context

- **Conversation Awareness**: Uses conversation context to improve task detection
- **Project Context**: Leverages current project information for better categorization
- **Cross-session Learning**: Learns from user feedback to improve detection accuracy

## Troubleshooting

### Common Issues

1. **Hook Not Executing**
   - Check `.claude-hooks.json` configuration
   - Verify Python path and dependencies
   - Check Claude Code hook system status

2. **Task Detection Failures**
   - Verify OpenAI API key configuration
   - Check network connectivity
   - Review task detection service logs

3. **Archon Integration Issues**
   - Verify MCP server is running
   - Check project permissions
   - Validate database connectivity

### Debugging

Enable detailed logging:

```bash
export HOOK_LOGGING_ENABLED=true
export LOGFIRE_ENABLED=true
```

Check logs in:
- Hook execution output (stdout/stderr)
- FastAPI server logs
- Archon MCP coordinator logs

### Support and Maintenance

- Monitor execution history via API
- Regular cache cleanup (automatically handled)
- Update confidence thresholds based on user feedback
- Review and adjust hook conditions for optimal detection

## Future Enhancements

### Planned Features

1. **User Feedback Loop**: Learn from user task acceptance/rejection
2. **Smart Prioritization**: AI-based priority assignment based on project context
3. **Multi-language Support**: Extend beyond English for international teams
4. **Integration Templates**: Pre-configured hooks for specific development workflows
5. **Advanced Context**: Git history and code analysis integration

### Extension Points

- **Custom Detection Models**: Plugin architecture for specialized task detection
- **Workflow Integration**: Connect with external project management tools
- **Team Coordination**: Multi-user task assignment and workload balancing
- **Analytics Dashboard**: Comprehensive metrics and insights on task creation patterns

## Conclusion

The Claude Hook Integration System provides seamless, intelligent task management that bridges the gap between conversational AI interactions and structured project management. By automatically detecting and creating actionable tasks from natural language conversations, it enables teams to maintain productivity without manual overhead while ensuring no important work items are lost in conversation.

The system is designed for reliability, performance, and extensibility, making it suitable for both individual developers and large development teams using Claude Code for their daily workflows.