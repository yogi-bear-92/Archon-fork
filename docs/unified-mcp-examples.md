# Unified Archon MCP - Usage Examples

## 🎯 Real-World Usage Examples

### 1. Complete Project Analysis and Setup

```javascript
// Create an intelligent project with automatic analysis
archon_project_create({
  "title": "E-commerce API",
  "description": "RESTful API for e-commerce platform with authentication and payments",
  "github_repo": "https://github.com/myorg/ecommerce-api",
  "auto_analyze": true
})
```

**What happens:**
1. 📝 Project created in Archon database
2. 🧠 Serena analyzes code structure and patterns
3. 🔍 Architectural patterns detected (MVC, Repository, etc.)
4. 📊 Integration with Archon RAG system
5. 📋 Structured project ready with intelligence context

### 2. Semantic Task Creation

```javascript
// Create task with intelligent code context
archon_task_create_semantic({
  "project_id": "proj-123",
  "title": "Optimize user authentication flow",
  "description": "Improve performance and security of login system",
  "code_context": "src/auth/login.py",
  "auto_suggestions": true
})
```

**What happens:**
1. 📁 Analyzes `src/auth/login.py` with Serena
2. 🔧 Generates refactoring suggestions
3. 📝 Enhanced task description with context
4. 💡 AI-powered improvement recommendations
5. 🔗 Links relevant code files and patterns

### 3. Deep Code Intelligence

```javascript
// Comprehensive code analysis
archon_code_intelligence({
  "operation": "analyze_structure",
  "file_path": "src/main.py",
  "project_path": "."
})

// Pattern detection across entire project
archon_code_intelligence({
  "operation": "detect_patterns",
  "project_path": ".",
  "pattern_types": ["architectural", "design", "anti-pattern"]
})

// Semantic code search
archon_code_intelligence({
  "operation": "semantic_search", 
  "query": "user authentication with JWT tokens",
  "project_path": "."
})
```

**Capabilities:**
- 🏗️ **Structure Analysis**: Functions, classes, dependencies
- 🎨 **Pattern Detection**: MVC, Repository, Singleton, etc.
- 🔍 **Semantic Search**: Natural language code search
- 💻 **Intelligent Completion**: Context-aware suggestions
- 🔧 **Refactoring**: Automated improvement suggestions

### 4. Complete Implementation Workflow

```javascript
// End-to-end workflow: Analysis → Tasks → Implementation
archon_analyze_and_implement({
  "task_description": "Add Redis caching layer to user API",
  "file_path": "src/api/users.py",
  "project_path": ".",
  "create_tasks": true
})
```

**Workflow Steps:**
1. 📊 **Analysis Phase**: Code structure + patterns with Serena
2. 📝 **Task Phase**: Auto-generated tasks based on analysis
3. 🚀 **Implementation Phase**: SPARC methodology with Claude Flow
4. ✅ **Integration**: Results combined across all services

### 5. SPARC Methodology Execution

```javascript
// Test-driven development workflow
archon_orchestrate({
  "operation": "sparc_workflow",
  "task": "Implement OAuth2 Google authentication",
  "mode": "tdd"
})

// Batch processing workflow  
archon_orchestrate({
  "operation": "sparc_workflow",
  "task": "Refactor authentication system",
  "mode": "batch"
})

// Initialize coordinated swarm
archon_orchestrate({
  "operation": "swarm_init",
  "topology": "adaptive", 
  "max_agents": 8
})
```

**SPARC Modes:**
- **TDD**: Test-driven development
- **Batch**: Parallel task processing
- **Pipeline**: Sequential workflow processing
- **Concurrent**: Multi-task coordination

### 6. System Status and Monitoring

```javascript
// Comprehensive system health check
archon_status()
```

**Returns:**
```json
{
  "status": "success",
  "services": {
    "unified_mcp": { "status": "healthy", "services_ready": true },
    "serena": { "status": "success", "mcp_available": false, "simulation_mode": true },
    "claude_flow": { "status": "success", "info": { "timestamp": "..." }},
    "backend": { "status": "healthy", "ready": true, "credentials_loaded": true }
  }
}
```

## 🔄 Advanced Workflows

### Multi-Service Code Refactoring

```javascript
// Step 1: Analyze current code
const analysis = archon_code_intelligence({
  "operation": "analyze_structure",
  "file_path": "src/legacy_module.py"
})

// Step 2: Detect patterns and issues  
const patterns = archon_code_intelligence({
  "operation": "detect_patterns",
  "project_path": "src/"
})

// Step 3: Get refactoring suggestions
const suggestions = archon_code_intelligence({
  "operation": "refactor_suggestions",
  "file_path": "src/legacy_module.py",
  "line_start": 1,
  "line_end": 100
})

// Step 4: Implement with SPARC methodology
const implementation = archon_orchestrate({
  "operation": "sparc_workflow",
  "task": "Refactor legacy module based on analysis",
  "mode": "tdd"
})
```

### Project Intelligence Dashboard

```javascript
// Create comprehensive project overview
const project = archon_project_create({
  "title": "Project Intelligence Dashboard",
  "auto_analyze": true
})

// Get detailed code intelligence
const intelligence = archon_analyze_and_implement({
  "task_description": "Generate comprehensive project documentation",
  "create_tasks": false  // Just analysis, no task creation
})

// System status for monitoring
const status = archon_status()
```

## 📊 Integration Benefits

### **Before (Multiple MCP Servers):**
```json
{
  "mcpServers": {
    "claude-flow": { "command": "...", "args": [...] },
    "serena": { "command": "...", "args": [...] },  
    "archon-tasks": { "command": "...", "args": [...] }
  }
}
```
- ❌ Complex configuration
- ❌ No cross-service coordination  
- ❌ Inconsistent interfaces
- ❌ Manual workflow orchestration

### **After (Unified Archon MCP):**
```json
{
  "mcpServers": {
    "archon": { 
      "command": "python3",
      "args": ["/path/to/start_unified_mcp.py"]
    }
  }
}
```
- ✅ Single configuration
- ✅ Intelligent cross-service workflows
- ✅ Unified command interface  
- ✅ Automated orchestration

## 🚀 Power User Tips

### 1. Chaining Operations
```javascript
// Chain multiple operations in sequence
const project = archon_project_create({ "title": "My App", "auto_analyze": true })
// → Use project_id from response
const task = archon_task_create_semantic({ 
  "project_id": project.project_id,
  "title": "Implement feature",
  "code_context": "src/main.py"
})
// → Use task context for implementation  
const workflow = archon_analyze_and_implement({
  "task_description": task.title + " with semantic context"
})
```

### 2. Combining Intelligence with Orchestration
```javascript
// Get code intelligence first
const analysis = archon_code_intelligence({
  "operation": "analyze_structure", 
  "project_path": "."
})

// Use insights for targeted orchestration
const execution = archon_orchestrate({
  "operation": "sparc_workflow",
  "task": `Optimize based on analysis: ${analysis.summary}`,
  "mode": "tdd"
})
```

### 3. Progressive Development Workflow
```javascript
// 1. Initial project setup with intelligence
archon_project_create({ "auto_analyze": true })

// 2. Create semantic tasks based on analysis  
archon_task_create_semantic({ "auto_suggestions": true })

// 3. Complete implementation workflow
archon_analyze_and_implement({ "create_tasks": true })

// 4. Monitor system health
archon_status()
```

---

**🎉 The unified Archon MCP provides the complete power of Archon, Serena, and Claude Flow through a single, intelligent interface!**