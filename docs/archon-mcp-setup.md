# Unified Archon MCP Setup

## 🎯 Single MCP Server for All Archon Capabilities

Instead of managing multiple MCP servers (Claude Flow, Serena, etc.), you now only need to install **one unified Archon MCP server** that provides access to all functionality.

## 🚀 Quick Setup

### 1. Add to Claude Desktop Configuration

Add this to your Claude Desktop MCP configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "archon": {
      "command": "python3",
      "args": ["/path/to/your/Archon-fork/python/scripts/start_unified_mcp.py"],
      "env": {
        "ARCHON_BASE_URL": "http://localhost:8080"
      }
    }
  }
}
```

### 2. Alternative: Using uv (Recommended)

```json
{
  "mcpServers": {
    "archon": {
      "command": "uv",
      "args": ["run", "/path/to/your/Archon-fork/python/scripts/start_unified_mcp.py"],
      "env": {
        "ARCHON_BASE_URL": "http://localhost:8080"
      }
    }
  }
}
```

## 🛠️ Available Tools

The unified Archon MCP provides these powerful tools:

### **Project & Task Management**
- `archon_project_create` - Create intelligent projects with auto-analysis
- `archon_task_create_semantic` - Create tasks with semantic code context

### **Code Intelligence (Powered by Serena)**
- `archon_code_intelligence` - Deep code analysis, patterns, completions, refactoring

### **Orchestration (Powered by Claude Flow)**  
- `archon_orchestrate` - Execute SPARC workflows, manage swarms, coordinate agents

### **Integrated Workflows**
- `archon_analyze_and_implement` - Complete workflow: Analysis → Tasks → Implementation

### **System Management**
- `archon_status` - Comprehensive status across all services

## 📋 Usage Examples

### Create an Intelligent Project
```javascript
// Automatically analyzes code and creates structured project
archon_project_create({
  "title": "My Web App",
  "description": "React frontend with FastAPI backend",
  "github_repo": "https://github.com/user/repo",
  "auto_analyze": true
})
```

### Deep Code Analysis  
```javascript
// Comprehensive code intelligence
archon_code_intelligence({
  "operation": "analyze_structure",
  "file_path": "src/main.py",
  "project_path": "."
})
```

### Complete Implementation Workflow
```javascript
// Analysis → Task Creation → Implementation
archon_analyze_and_implement({
  "task_description": "Add user authentication system",
  "project_path": ".",
  "create_tasks": true
})
```

### SPARC Methodology Execution
```javascript
// Execute with Claude Flow orchestration
archon_orchestrate({
  "operation": "sparc_workflow", 
  "task": "Implement OAuth2 authentication",
  "mode": "tdd"
})
```

## 🔧 Configuration

### Environment Variables
- `ARCHON_BASE_URL` - Backend API URL (default: http://localhost:8080)
- `ARCHON_SERVER_PORT` - Server port (default: 8080)

### Prerequisites
1. **Archon Backend Running** - Start with `python -m src.server.main`
2. **Dependencies Installed** - Run `uv sync` in the python directory
3. **Database Configured** - Ensure Supabase connection is set up

## 🎯 Benefits of Unified Architecture

### **For Users:**
- ✅ **Single Installation** - One MCP server instead of multiple
- ✅ **Unified Interface** - Consistent commands across all functionality
- ✅ **Intelligent Routing** - Automatically routes to appropriate services
- ✅ **Cross-Service Workflows** - Integrated operations across multiple systems

### **For Developers:**
- ✅ **Simplified Maintenance** - One codebase to maintain
- ✅ **Better Integration** - Shared context and coordination
- ✅ **Enhanced Capabilities** - Combined power of all services
- ✅ **Consistent Error Handling** - Unified logging and debugging

## 🚨 Migration from Individual MCP Servers

If you previously had separate MCP servers configured:

### Remove Old Configuration:
```json
// Remove these from claude_desktop_config.json
{
  "mcpServers": {
    "claude-flow": { ... },    // Remove
    "serena": { ... },         // Remove  
    "archon-tasks": { ... }    // Remove
  }
}
```

### Add New Unified Configuration:
```json
{
  "mcpServers": {
    "archon": {
      "command": "python3",
      "args": ["/path/to/Archon-fork/python/scripts/start_unified_mcp.py"],
      "env": {
        "ARCHON_BASE_URL": "http://localhost:8080"
      }
    }
  }
}
```

## 🔍 Troubleshooting

### MCP Server Won't Start
1. Check Python path in configuration
2. Ensure backend is running on port 8080
3. Verify dependencies are installed (`uv sync`)

### Tools Not Working  
1. Check `archon_status` tool to see service health
2. Verify `ARCHON_BASE_URL` points to running backend
3. Check logs in Claude Desktop for error details

### Performance Issues
1. Ensure backend is running locally (not remote)
2. Check service status with `archon_status`
3. Restart backend if needed

## 📚 Documentation

- **Full API Documentation**: Visit http://localhost:8080/docs when backend is running
- **Code Intelligence**: Powered by Serena MCP with semantic analysis
- **Orchestration**: Powered by Claude Flow with SPARC methodology
- **Project Management**: Native Archon backend with RAG capabilities

---

**🎉 You now have the complete power of Archon through a single, unified MCP interface!**