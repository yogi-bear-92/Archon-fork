#!/bin/bash
# Continue Task Command - Seamlessly transition work to new conversation
# Usage: ./continue-task.sh [task_id] [optional: context_summary]

set -e

# Configuration
ARCHON_API="http://localhost:8181"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if Archon server is running
check_archon_server() {
    if ! curl -s "$ARCHON_API/health" >/dev/null 2>&1; then
        print_error "Archon server not running at $ARCHON_API"
        print_info "Start server with: export ARCHON_SERVER_PORT=8181 && python3 -m uvicorn src.server.main:app --host 0.0.0.0 --port 8181 --reload"
        exit 1
    fi
    print_status "Archon server is running"
}

# Function to get task details
get_task_details() {
    local task_id="$1"
    
    print_info "Fetching task details for ID: $task_id" >&2
    
    # Get specific task information using the correct API endpoint
    local task_response=$(curl -s "$ARCHON_API/api/tasks/$task_id")
    
    # Check if the response contains an error
    if echo "$task_response" | grep -q '"error"'; then
        print_error "Task ID '$task_id' not found" >&2
        print_info "Available tasks:" >&2
        curl -s "$ARCHON_API/api/tasks" | python3 -c "
import sys, json
data = json.load(sys.stdin)
tasks = data.get('tasks', [])
for task in tasks:
    status_icon = '🔄' if task['status'] == 'doing' else '📋' if task['status'] == 'todo' else '✅'
    print(f\"  {status_icon} {task['id']} - {task['title']} ({task['status']})\")" >&2
        exit 1
    fi
    
    echo "$task_response"
}

# Function to get project context
get_project_context() {
    local project_id="$1"
    
    print_info "Gathering project context..." >&2
    
    # Get project details
    local project=$(curl -s "$ARCHON_API/api/projects/$project_id")
    
    # Get related tasks
    local related_tasks=$(curl -s "$ARCHON_API/api/tasks?project_id=$project_id" | python3 -c "
import sys, json
data = json.load(sys.stdin)
tasks = data.get('tasks', [])
recent_tasks = sorted(tasks, key=lambda x: x['updated_at'], reverse=True)[:5]
for task in recent_tasks:
    status_icon = '🔄' if task['status'] == 'doing' else '📋' if task['status'] == 'todo' else '✅'
    print(f\"  {status_icon} {task['title']} ({task['status']}) - {task.get('feature', 'no-feature')}\")
")
    
    # Extract project title safely
    local temp_project_file=$(mktemp)
    echo "$project" > "$temp_project_file"
    local project_title=$(python3 -c "
import sys, json
try:
    with open('$temp_project_file', 'r') as f:
        data = json.load(f)
    print(data.get('title', 'Unknown Project'))
except:
    print('Unknown Project')
")
    rm -f "$temp_project_file"
    
    echo "PROJECT: $project_title"
    echo "RECENT TASKS:"
    echo "$related_tasks"
}

# Function to generate continuation prompt
generate_continuation_prompt() {
    local task_details="$1"
    local project_context="$2"
    local context_summary="$3"
    local current_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    
    # Create temp file for task data to avoid shell piping issues
    local temp_task_file=$(mktemp)
    echo "$task_details" > "$temp_task_file"
    
    # Extract key information from task details
    local task_title=$(python3 -c "
import sys, json
try:
    with open('$temp_task_file', 'r') as f:
        data = json.load(f)
    print(data.get('title', 'Unknown Task'))
except:
    print('Unknown Task')
")
    local task_description=$(python3 -c "
import sys, json
try:
    with open('$temp_task_file', 'r') as f:
        data = json.load(f)
    print(data.get('description', 'No description available'))
except:
    print('No description available')
")
    local task_id=$(python3 -c "
import sys, json
try:
    with open('$temp_task_file', 'r') as f:
        data = json.load(f)
    print(data.get('id', 'unknown'))
except:
    print('unknown')
")
    local task_status=$(python3 -c "
import sys, json
try:
    with open('$temp_task_file', 'r') as f:
        data = json.load(f)
    print(data.get('status', 'unknown'))
except:
    print('unknown')
")
    local assignee=$(python3 -c "
import sys, json
try:
    with open('$temp_task_file', 'r') as f:
        data = json.load(f)
    print(data.get('assignee', 'Unassigned'))
except:
    print('Unassigned')
")
    local feature=$(python3 -c "
import sys, json
try:
    with open('$temp_task_file', 'r') as f:
        data = json.load(f)
    print(data.get('feature', 'general'))
except:
    print('general')
")
    
    # Clean up temp file
    rm -f "$temp_task_file"
    
    # Create continuation prompt file
    local prompt_file="$PROJECT_ROOT/.continue-task-prompt.md"
    
    cat > "$prompt_file" << EOF
# 🔄 Task Continuation - New Conversation

**Continuation Time**: $current_time
**Previous Context**: Context reached ~60% capacity - transitioning to new conversation

## 🎯 **ACTIVE TASK DETAILS**

**Task ID**: \`$task_id\`
**Title**: $task_title
**Status**: $task_status
**Assignee**: $assignee  
**Feature**: $feature

**Description**:
$task_description

## 📋 **PROJECT CONTEXT**

$project_context

## 🔍 **CONTINUATION CONTEXT**

$context_summary

## 🚀 **NEXT ACTIONS**

1. **First Priority**: Update this task status to 'doing' in Archon
   \`\`\`bash
   curl -X PUT "http://localhost:8181/api/tasks/$task_id" -H "Content-Type: application/json" -d '{"status": "doing"}'
   \`\`\`

2. **Continue Implementation**: Pick up work exactly where previous conversation left off

3. **Use Proper Task Management**: Follow CLAUDE.md hierarchy:
   - Try Archon MCP tools first: \`mcp__archon__update_task(task_id="$task_id", status="doing")\`
   - Fallback to direct API calls if MCP session issues occur
   - Only use TodoWrite as last resort

## 🔧 **SYSTEM STATUS CHECKS**

Before continuing, verify system health:
\`\`\`bash
# Check Archon server
curl http://localhost:8181/health

# Check current project tasks  
curl http://localhost:8181/api/tasks?project_id=\$(curl -s http://localhost:8181/api/tasks | python3 -c "import sys,json; tasks=json.load(sys.stdin)['tasks']; task=next(t for t in tasks if t['id']=='$task_id'); print(task['project_id'])")

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

**Start new conversation with**: "Continue task $task_id - $task_title"
EOF

    print_status "Continuation prompt generated: $prompt_file"
}

# Main function
main() {
    print_info "🔄 Continue Task Command - Context Transition Helper"
    echo "=================================================="
    
    # Check if task ID provided
    if [ -z "$1" ]; then
        print_error "Task ID required"
        print_info "Usage: $0 <task_id> [context_summary]"
        print_info ""
        print_info "Available tasks:"
        check_archon_server
        curl -s "$ARCHON_API/api/tasks" | python3 -c "
import sys, json
data = json.load(sys.stdin)
tasks = data.get('tasks', [])
active_tasks = [t for t in tasks if t['status'] in ['todo', 'doing']]
for task in active_tasks:
    status_icon = '🔄' if task['status'] == 'doing' else '📋'
    print(f\"  {status_icon} {task['id']} - {task['title']}\")"
        exit 1
    fi
    
    local task_id="$1"
    local context_summary="${2:-Previous conversation reached context capacity. Transitioning work to maintain continuity.}"
    
    # Verify Archon server
    check_archon_server
    
    # Get task details
    local task_details=$(get_task_details "$task_id")
    
    # Extract project_id using temp file
    local temp_task_file=$(mktemp)
    echo "$task_details" > "$temp_task_file"
    local project_id=$(python3 -c "
import sys, json
try:
    with open('$temp_task_file', 'r') as f:
        data = json.load(f)
    print(data.get('project_id', ''))
except:
    print('')
" && rm -f "$temp_task_file")
    
    # Get project context
    local project_context=$(get_project_context "$project_id")
    
    # Generate continuation prompt
    local prompt_file="$PROJECT_ROOT/.continue-task-prompt.md"
    generate_continuation_prompt "$task_details" "$project_context" "$context_summary"
    
    print_status "Task continuation prepared!"
    print_info "Prompt file: $prompt_file"
    print_warning "Context transition ready - copy prompt to new conversation"
    
    # Display the prompt file content
    echo ""
    echo "================================="
    echo "📋 CONTINUATION PROMPT (copy this to new conversation):"
    echo "================================="
    cat "$prompt_file"
    
    return 0
}

# Run main function with all arguments
main "$@"