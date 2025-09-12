# Interactive Task Creation System

## Overview

The Interactive Task Creation System transforms the automated approach into a user-guided, conversational workflow. Instead of automatically creating tasks, the system:

1. **🔍 Detects potential tasks** in user messages using AI analysis
2. **❓ Asks for user confirmation** before proceeding with task creation
3. **💬 Gathers additional details** through intelligent questions
4. **✅ Creates tasks only when users explicitly confirm** with complete information

This approach ensures users have full control over what tasks are created while still benefiting from AI assistance in task analysis and requirement gathering.

## Architecture

```
User Message in Claude Code
          ↓
    Task Detection (AI)
          ↓
   User Confirmation? ────→ No ────→ End
          ↓ Yes
   Gather Requirements
          ↓
  Ask Clarifying Questions
          ↓
   Final User Confirmation ────→ No ────→ End
          ↓ Yes
   Create Task in Archon
```

## Key Components

### 1. Interactive Task Service (`interactive_task_service.py`)

**Purpose**: Core orchestration service managing the conversational task creation workflow.

**Key Features**:
- **Session Management**: Tracks user conversations through state machine
- **AI-Powered Question Generation**: Creates relevant questions based on task context
- **Progressive Detail Gathering**: Builds comprehensive task specifications iteratively
- **State Persistence**: Maintains conversation state between user interactions

**Workflow States**:
```python
class TaskCreationState(str, Enum):
    INITIAL_DETECTION = "initial_detection"      # AI detects potential task
    USER_CONFIRMATION = "user_confirmation"     # Ask user if they want task
    GATHERING_DETAILS = "gathering_details"     # Ask clarifying questions
    FINAL_CONFIRMATION = "final_confirmation"   # Show final task for approval
    TASK_CREATED = "task_created"               # Task successfully created
    USER_DECLINED = "user_declined"             # User declined task creation
```

### 2. Interactive Task API (`interactive_task_api.py`)

**Purpose**: REST API endpoints for managing interactive task creation sessions.

**Key Endpoints**:
- `POST /api/interactive-tasks/detect` - Detect tasks and start session
- `POST /api/interactive-tasks/respond` - Process user responses
- `GET /api/interactive-tasks/sessions` - List active sessions
- `DELETE /api/interactive-tasks/sessions/{id}` - Cancel session

### 3. Enhanced Hook Scripts

#### Post-User-Prompt Hook (`post_user_prompt_hook.py`)
- **Purpose**: Entry point that analyzes user messages for tasks
- **New Behavior**: Creates interactive sessions instead of auto-creating tasks
- **Output**: Provides Claude Code with confirmation questions for users

#### Interactive Response Hook (`interactive_task_response_hook.py`)
- **Purpose**: Processes user responses during task creation conversations
- **Usage**: Called manually when users respond to task creation prompts
- **Workflow**: Advances sessions through states until completion or cancellation

## Interactive Workflow Examples

### Example 1: Simple Task Confirmation

**User**: "Can you implement user authentication with JWT tokens?"

**System Response**:
```
I detected a potential task: Implement JWT Authentication System

Description: Create JWT-based authentication with login, logout, and token management functionality.

Would you like me to help you create this as a task in your project? (yes/no)
```

**User**: "yes"

**System**: ✅ Task created successfully! The task has been added to your project.

### Example 2: Complex Task with Questions

**User**: "We need to build a dashboard for our app"

**System Response**:
```
I detected a potential task: Build Application Dashboard

Would you like me to help you create this as a task? (yes/no)
```

**User**: "yes"

**System Questions**:
```
Great! I'd like to gather some more details:

1. What specific data should the dashboard display (analytics, user metrics, system status, etc.)?
2. Who will be the primary users of this dashboard (admins, managers, end users)?
3. Do you need real-time updates or is periodic refresh sufficient?
4. Are there any specific technologies or frameworks you'd prefer to use?
```

**User**: "The dashboard should show user analytics, system performance metrics, and recent activity logs. It's for administrators and should have real-time updates."

**System Final Confirmation**:
```
Perfect! Here's the task I'll create:

**Title**: Build Admin Dashboard with Real-time Analytics

**Description**: Create a comprehensive admin dashboard displaying user analytics, system performance metrics, and activity logs. Must support real-time updates and be optimized for administrative use.

*Note: The coordinator agent will automatically assign this task to the most appropriate team member based on the requirements.*

Should I create this task? (yes/no)
```

**User**: "yes"

**System**: ✅ Task created successfully!

### Example 3: User Declines

**User**: "Maybe we should add some caching"

**System**:
```
I detected a potential task: Implement Caching System

Would you like me to create this as a task? (yes/no)
```

**User**: "no, not right now"

**System**: No problem! I won't create a task. Let me know if you change your mind.

## Configuration

### Hook Configuration (`.claude-hooks.json`)

```json
{
  "hooks": {
    "post-user-prompt": {
      "description": "Analyze user messages for potential tasks and initiate interactive creation process",
      "command": "python3 src/hooks/post_user_prompt_hook.py",
      "timeout": 30000,
      "enabled": true,
      "background": false,
      "conditions": {
        "min_message_length": 10,
        "include_patterns": [
          "\\b(implement|create|build|add|fix|develop)\\b",
          "\\b(can you|could you|please|need to)\\b"
        ]
      }
    },
    "interactive-task-response": {
      "description": "Process user responses in interactive task creation sessions",
      "command": "python3 src/hooks/interactive_task_response_hook.py",
      "manual_trigger": true,
      "parameters": {
        "session_id": "required",
        "user_response": "required"
      }
    }
  }
}
```

### Environment Variables

```bash
# Task detection settings
TASK_CONFIDENCE_THRESHOLD=0.6          # Minimum confidence to suggest tasks
INTERACTIVE_SESSION_TIMEOUT=1800       # Session timeout in seconds (30 min)
MAX_CLARIFYING_QUESTIONS=5             # Maximum questions per session

# AI model settings
TASK_DETECTION_MODEL="gpt-4o-mini"     # Fast model for detection
QUESTION_GENERATION_MODEL="gpt-4o"     # Better model for questions
TASK_CREATION_MODEL="gpt-4o"           # Best model for final task creation
```

## API Usage Examples

### Detecting Tasks via API

```bash
curl -X POST "http://localhost:8080/api/interactive-tasks/detect" \
     -H "Content-Type: application/json" \
     -d '{
       "user_message": "I need to create a REST API for managing user profiles",
       "conversation_context": "Planning new features",
       "project_context": "User management system"
     }'
```

**Response**:
```json
{
  "success": true,
  "session_active": true,
  "session_id": "task_session_20241206_143022",
  "state": "user_confirmation",
  "message": "I detected a potential task: **Create User Profile Management API**\n\nWould you like me to help you create this as a task in your project? (yes/no)",
  "action_required": true,
  "task_preview": {
    "title": "Create User Profile Management API",
    "confidence": 0.85,
    "category": "feature",
    "urgency": "medium"
  }
}
```

### Processing User Response

```bash
curl -X POST "http://localhost:8080/api/interactive-tasks/respond" \
     -H "Content-Type: application/json" \
     -d '{
       "session_id": "task_session_20241206_143022",
       "user_response": "yes, please create this task"
     }'
```

## Session Management

### Session States

1. **USER_CONFIRMATION**: Waiting for user to confirm task creation
2. **GATHERING_DETAILS**: Asking clarifying questions
3. **FINAL_CONFIRMATION**: Showing complete task for final approval
4. **TASK_CREATED**: Task successfully created
5. **USER_DECLINED**: User declined task creation

### Session Cleanup

- **Automatic**: Sessions expire after 30 minutes of inactivity
- **Manual**: Users can cancel sessions via API
- **On Completion**: Sessions are automatically cleaned up when tasks are created or declined

## Intelligence Features

### Smart Question Generation

The system uses GPT-4o to generate contextually relevant questions:

**Question Categories**:
- **Scope & Requirements**: "What specific features should be included?"
- **Technical Approach**: "Which technologies or frameworks should be used?"
- **User Experience**: "Who will use this and how?"
- **Priority & Timeline**: "How urgent is this task?"
- **Dependencies**: "What other systems does this depend on?"

### Adaptive Detail Gathering

- **Progressive Questions**: Asks follow-up questions based on previous answers
- **Context Awareness**: Considers project context and conversation history
- **Completeness Check**: Stops asking when sufficient information is gathered
- **Fallback Logic**: Proceeds with available information if AI fails

### Final Task Enhancement

The system creates comprehensive task specifications by combining:
- Original user message
- Conversation context
- Gathered requirements
- Best practices for task documentation

## Integration with Archon

### Project Management Integration

- **Auto Project Detection**: Finds appropriate existing projects
- **Smart Categorization**: Assigns tasks to relevant features/categories
- **Assignee Suggestions**: Recommends appropriate team members
- **Priority Mapping**: Converts user urgency to task priorities

### Real-time Updates

- **Socket.IO Events**: Broadcasts task creation to connected clients
- **Progress Tracking**: Updates project metrics when tasks are created
- **History Logging**: Maintains audit trail of interactive task creation

## Testing and Validation

### Comprehensive Test Suite

Run the complete test suite:

```bash
cd /path/to/archon/python
python3 test_interactive_tasks.py
```

**Test Coverage**:
- ✅ Task detection accuracy with various message types
- ✅ Interactive workflow from detection to creation
- ✅ User decline scenarios and cleanup
- ✅ Session management and expiration
- ✅ API integration and error handling
- ✅ Question generation and detail gathering
- ✅ Final task creation and Archon integration

### Manual Testing Scenarios

1. **Happy Path**: User confirms task, answers questions, approves final task
2. **Decline Scenarios**: User says no at confirmation or final approval
3. **Ambiguous Messages**: System asks clarifying questions
4. **Edge Cases**: Very short messages, unclear requests, multiple tasks
5. **Session Management**: Multiple concurrent sessions, timeouts, cancellations

## Performance Optimization

### Caching Strategy

- **Detection Cache**: Avoids re-analyzing identical messages
- **Question Templates**: Reuses question patterns for similar contexts
- **Session Persistence**: Maintains state efficiently in memory

### Resource Management

- **Lightweight Sessions**: Minimal memory footprint per session
- **Fast AI Models**: Uses GPT-4o-mini for quick detection
- **Graceful Degradation**: Fallback behavior when AI is unavailable
- **Automatic Cleanup**: Regular session maintenance and cleanup

## User Experience Benefits

### Improved Control

- **No Surprise Tasks**: Users explicitly confirm what gets created
- **Better Requirements**: Interactive questions lead to clearer tasks
- **Flexible Workflow**: Users can decline or modify at any stage

### Enhanced Task Quality

- **Comprehensive Descriptions**: Gathered details create better specifications
- **Proper Categorization**: AI assigns appropriate categories and priorities
- **Context Preservation**: Full conversation context is maintained in tasks

### Reduced Noise

- **Confidence Filtering**: Only high-quality task candidates are suggested
- **User Validation**: Reduces false positives from automated detection
- **Intentional Creation**: Tasks are created only when users truly want them

## Best Practices

### For Users

1. **Be Specific**: Provide clear task descriptions for better detection
2. **Engage with Questions**: Answer clarifying questions for better tasks
3. **Review Final Tasks**: Check the final specification before confirming
4. **Use Decline Option**: Say no if the task isn't what you want

### For Administrators

1. **Monitor Sessions**: Use API to track active task creation sessions
2. **Adjust Thresholds**: Tune confidence levels based on user feedback
3. **Review Patterns**: Analyze what types of messages trigger task detection
4. **Optimize Questions**: Update question generation based on usage patterns

## Troubleshooting

### Common Issues

1. **Tasks Not Detected**
   - Check if message meets minimum confidence threshold
   - Verify hook configuration and patterns
   - Review message length and content requirements

2. **Questions Not Generated**
   - Check AI model availability and API keys
   - Verify network connectivity to OpenAI
   - Review question generation prompt templates

3. **Sessions Not Persisting**
   - Check session timeout settings
   - Verify service initialization
   - Review memory and resource usage

4. **Task Creation Failures**
   - Verify Archon MCP coordinator connection
   - Check project permissions and availability
   - Review task specification format

### Debugging Commands

```bash
# Check active sessions
curl "http://localhost:8080/api/interactive-tasks/sessions"

# Test task detection
curl -X POST "http://localhost:8080/api/interactive-tasks/detect" \
     -H "Content-Type: application/json" \
     -d '{"user_message": "test task creation"}'

# Run health check
curl "http://localhost:8080/api/interactive-tasks/health"

# View hook configuration
cat .claude-hooks.json
```

## Migration from Automatic System

### Key Changes

1. **Hook Behavior**: Post-user-prompt hook now creates sessions instead of tasks
2. **User Interaction**: Claude Code will ask users for confirmation
3. **Response Handling**: New hook processes user responses in sessions
4. **API Changes**: New endpoints for interactive task management

### Backward Compatibility

- **Hook Configuration**: Updated to support both automatic and interactive modes
- **API Endpoints**: Original hook management API still available
- **Service Integration**: Existing Archon integration unchanged

## Future Enhancements

### Planned Features

1. **Multi-Task Sessions**: Handle multiple tasks in one conversation
2. **Template-Based Creation**: Pre-configured task templates for common scenarios
3. **Team Collaboration**: Multi-user task creation and approval workflows
4. **Smart Defaults**: Learn user preferences for faster task creation
5. **Voice Integration**: Support for voice-based task creation conversations

### Integration Opportunities

1. **External Tools**: Connect with Jira, Trello, GitHub Issues
2. **Calendar Integration**: Schedule tasks and set deadlines
3. **Notification Systems**: Alert team members about new tasks
4. **Analytics Dashboard**: Track task creation patterns and success rates

## Conclusion

The Interactive Task Creation System provides a balanced approach between AI assistance and user control. By requiring explicit user confirmation and gathering detailed requirements through conversation, the system ensures that:

- **Users remain in control** of what tasks are created
- **Task quality is high** with comprehensive specifications
- **False positives are eliminated** through user validation
- **Requirements are clear** through interactive questioning

This conversational approach transforms task management from a passive automated system into an active collaborative tool that enhances productivity while maintaining user agency.