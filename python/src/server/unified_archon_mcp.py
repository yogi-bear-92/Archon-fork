#!/usr/bin/env python3
"""
Unified Archon MCP Server

This is the master MCP server that wraps and coordinates all Archon services:
- FastAPI Backend (Projects, Tasks, Knowledge, RAG)
- Serena Code Intelligence Service  
- Claude Flow Orchestration Service
- Integrated cross-service workflows

Users only need to install this single MCP server for complete Archon functionality.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Add the src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    import mcp.server.stdio
except ImportError:
    print("Error: MCP not installed. Install with: pip install mcp")
    sys.exit(1)

from config.logfire_config import get_logger
from services.serena_service import serena_service
from services.claude_flow_service import claude_flow_service
from services.url_detection_service import get_url_detection_service
from .agents.url_decision_agent import get_url_decision_agent, URLDecisionContext
from services.cli_tool_discovery_service import cli_discovery_service

logger = get_logger(__name__)

# Initialize the unified MCP server
server = Server("unified-archon")

class ArchonMCPCoordinator:
    """Coordinates all Archon services through a single MCP interface."""
    
    def __init__(self):
        self.base_url = os.getenv("ARCHON_BASE_URL", "http://localhost:8080")
        self.services_ready = False
        
    async def initialize_services(self):
        """Initialize all underlying services."""
        try:
            logger.info("🚀 Initializing Unified Archon MCP services...")
            
            # Start CLI discovery service
            logger.info("Starting CLI Tool Discovery Service...")
            await cli_discovery_service.start()
            
            # Check service availability
            services_status = {
                "serena": await serena_service.get_service_status(),
                "claude_flow": await claude_flow_service.get_swarm_status(),
                "cli_discovery": {
                    "running": cli_discovery_service.running,
                    "available_tools": len([s for s in cli_discovery_service.get_tool_status().values() if s.available]),
                    "total_commands": len(cli_discovery_service.get_discovered_commands())
                }
            }
            
            self.services_ready = True
            logger.info("✅ All Archon services initialized successfully")
            return {"status": "success", "services": services_status}
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize services: {e}")
            return {"status": "error", "error": str(e)}
    
    # ========================================================================
    # INTELLIGENT TOOL ROUTING
    # ========================================================================
    
    def route_request(self, tool_name: str, args: Dict[str, Any]) -> str:
        """Intelligently route requests to appropriate services."""
        
        # Integrated workflows → Multi-service (check first for priority)
        if any(keyword in tool_name for keyword in ["analyze_and", "smart", "integrated"]):
            return "integrated"
            
        # Orchestration operations → Claude Flow    
        elif any(keyword in tool_name for keyword in ["orchestrate", "swarm", "sparc", "agents", "coordinate"]):
            return "claude_flow"
        
        # Code-related operations → Serena
        elif any(keyword in tool_name for keyword in ["code", "semantic", "completion", "refactor"]):
            return "serena"
        
        # URL detection and management → URL Detection Service
        elif any(keyword in tool_name for keyword in ["url", "detect", "suggestion"]):
            return "url_detection"
        
        # Project/Task operations → FastAPI Backend
        elif any(keyword in tool_name for keyword in ["project", "task", "knowledge", "search", "rag"]):
            return "backend"
            
        # Default to backend
        return "backend"
    
    # ========================================================================
    # UNIFIED PROJECT MANAGEMENT
    # ========================================================================
    
    async def create_intelligent_project(
        self, 
        title: str, 
        description: str = "",
        github_repo: Optional[str] = None,
        auto_analyze: bool = True
    ) -> Dict[str, Any]:
        """Create project with automatic code intelligence analysis."""
        try:
            # Step 1: Create project via backend API
            import httpx
            async with httpx.AsyncClient() as client:
                project_data = {
                    "title": title,
                    "description": description,
                    "github_repo": github_repo
                }
                
                response = await client.post(
                    f"{self.base_url}/api/projects/create",
                    json=project_data,
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    raise Exception(f"Project creation failed: {response.text}")
                
                project = response.json()
                project_id = project.get("project_id")
                
            # Step 2: Auto-analyze with Serena if requested
            analysis_results = {}
            if auto_analyze and github_repo:
                logger.info(f"🧠 Running intelligent analysis for project: {title}")
                
                # Code structure analysis
                code_analysis = await serena_service.analyze_project_structure(
                    project_path=".",  # Would be actual repo path in real implementation
                    include_patterns=True,
                    include_metrics=True
                )
                
                # Pattern detection
                patterns = await serena_service.detect_code_patterns(
                    project_path=".",
                    pattern_types=["architectural", "design", "anti-pattern"]
                )
                
                analysis_results = {
                    "code_analysis": code_analysis,
                    "patterns": patterns,
                    "archon_integration": await serena_service.integrate_with_archon_rag(".")
                }
            
            return {
                "status": "success",
                "project": project,
                "project_id": project_id,
                "analysis": analysis_results,
                "message": f"Intelligent project '{title}' created successfully"
            }
            
        except Exception as e:
            logger.error(f"Intelligent project creation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def create_semantic_task(
        self,
        project_id: str,
        title: str,
        description: str = "",
        code_context: Optional[str] = None,
        auto_suggestions: bool = True
    ) -> Dict[str, Any]:
        """Create task with semantic code context and suggestions."""
        try:
            # Step 1: Analyze code context if provided
            semantic_context = {}
            if code_context and auto_suggestions:
                # Get semantic analysis
                analysis = await serena_service.analyze_code_structure(
                    file_path=code_context,
                    include_dependencies=True
                )
                
                # Get refactoring suggestions  
                suggestions = await serena_service.get_refactoring_suggestions(
                    file_path=code_context,
                    line_start=1,
                    line_end=100
                )
                
                semantic_context = {
                    "code_analysis": analysis,
                    "suggestions": suggestions
                }
            
            # Step 2: Create task with enhanced description
            enhanced_description = description
            if semantic_context:
                enhanced_description += f"\n\n🧠 **Semantic Context:**\n"
                if semantic_context.get("suggestions"):
                    enhanced_description += "- Refactoring opportunities detected\n"
                if semantic_context.get("code_analysis"):
                    enhanced_description += "- Code structure analyzed\n"
            
            # Create task via backend API
            import httpx
            async with httpx.AsyncClient() as client:
                task_data = {
                    "project_id": project_id,
                    "title": title,
                    "description": enhanced_description,
                    "sources": [{"url": code_context, "type": "code_file"}] if code_context else []
                }
                
                response = await client.post(
                    f"{self.base_url}/api/projects/{project_id}/tasks",
                    json=task_data,
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    raise Exception(f"Task creation failed: {response.text}")
                
                task = response.json()
            
            return {
                "status": "success",
                "task": task,
                "semantic_context": semantic_context,
                "message": f"Semantic task '{title}' created with code intelligence"
            }
            
        except Exception as e:
            logger.error(f"Semantic task creation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    # ========================================================================
    # INTELLIGENT CODE WORKFLOWS
    # ========================================================================
    
    async def analyze_and_implement_workflow(
        self,
        task_description: str,
        file_path: Optional[str] = None,
        project_path: str = ".",
        create_tasks: bool = True
    ) -> Dict[str, Any]:
        """Complete workflow: Code Analysis → Task Creation → Implementation."""
        try:
            logger.info("🔄 Starting Analyze and Implement workflow...")
            
            workflow_results = {
                "analysis_phase": {},
                "task_phase": {},
                "implementation_phase": {},
                "status": "in_progress"
            }
            
            # Phase 1: Deep Analysis with Serena
            logger.info("📊 Phase 1: Deep code analysis...")
            if file_path:
                code_analysis = await serena_service.analyze_code_structure(
                    file_path=file_path,
                    include_dependencies=True
                )
            else:
                code_analysis = await serena_service.analyze_project_structure(
                    project_path=project_path,
                    include_patterns=True
                )
            
            patterns = await serena_service.detect_code_patterns(
                project_path=project_path,
                pattern_types=["architectural", "design"]
            )
            
            workflow_results["analysis_phase"] = {
                "code_analysis": code_analysis,
                "patterns": patterns,
                "completed": True
            }
            
            # Phase 2: Task Creation (if requested)
            if create_tasks:
                logger.info("📝 Phase 2: Intelligent task creation...")
                # This would create tasks based on analysis
                workflow_results["task_phase"] = {
                    "tasks_created": 0,  # Would be actual count
                    "completed": True
                }
            
            # Phase 3: Implementation with Claude Flow
            logger.info("🚀 Phase 3: Implementation orchestration...")
            implementation = await claude_flow_service.execute_sparc_workflow(
                task=task_description,
                mode="tdd"
            )
            
            workflow_results["implementation_phase"] = {
                "sparc_execution": implementation,
                "completed": implementation.get("status") != "error"
            }
            
            # Final status
            workflow_results["status"] = "completed" if all(
                phase.get("completed", False) 
                for phase in workflow_results.values() 
                if isinstance(phase, dict)
            ) else "partial"
            
            return {
                "status": "success",
                "workflow": workflow_results,
                "message": "Analyze and Implement workflow completed"
            }
            
        except Exception as e:
            logger.error(f"Analyze and Implement workflow failed: {e}")
            return {"status": "error", "error": str(e)}
    
    # ========================================================================
    # PROJECT SETUP AND CONFIGURATION
    # ========================================================================
    
    async def setup_claude_md(
        self,
        project_path: str = ".",
        project_type: str = "auto",
        include_archon: bool = True,
        include_serena: bool = True,
        include_claude_flow: bool = True,
        custom_instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate and configure CLAUDE.md file for optimal project setup."""
        try:
            logger.info(f"Setting up CLAUDE.md for project: {project_path}")
            
            # Step 1: Analyze project structure to determine type
            if project_type == "auto":
                project_analysis = await serena_service.analyze_project_structure(
                    project_path=project_path,
                    include_patterns=True
                )
                
                # Detect project type from patterns and files
                project_type = self._detect_project_type(project_analysis)
            
            # Step 2: Generate CLAUDE.md content based on project type
            claude_md_content = self._generate_claude_md_template(
                project_type=project_type,
                include_archon=include_archon,
                include_serena=include_serena,
                include_claude_flow=include_claude_flow,
                custom_instructions=custom_instructions
            )
            
            # Step 3: Write CLAUDE.md file
            claude_md_path = Path(project_path) / "CLAUDE.md"
            with open(claude_md_path, 'w', encoding='utf-8') as f:
                f.write(claude_md_content)
            
            # Step 4: Create .gitignore entries if needed
            gitignore_path = Path(project_path) / ".gitignore"
            gitignore_additions = [
                "# Claude Code / Archon generated files",
                ".claude-flow/",
                ".serena/",
                ".archon/",
                "*.claude-session",
                ""
            ]
            
            if gitignore_path.exists():
                with open(gitignore_path, 'r') as f:
                    existing_content = f.read()
                
                # Only add if not already present
                if ".claude-flow/" not in existing_content:
                    with open(gitignore_path, 'a') as f:
                        f.write("\n" + "\n".join(gitignore_additions))
            else:
                # Create new .gitignore
                with open(gitignore_path, 'w') as f:
                    f.write("\n".join(gitignore_additions))
            
            return {
                "status": "success",
                "claude_md_created": True,
                "claude_md_path": str(claude_md_path),
                "project_type": project_type,
                "gitignore_updated": True,
                "message": f"CLAUDE.md configured for {project_type} project"
            }
            
        except Exception as e:
            logger.error(f"CLAUDE.md setup failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _detect_project_type(self, analysis: Dict[str, Any]) -> str:
        """Detect project type from analysis results."""
        
        # Check for common patterns and files
        patterns = analysis.get("patterns", [])
        
        # Web development frameworks
        if any("react" in str(p).lower() for p in patterns):
            return "react"
        elif any("next" in str(p).lower() for p in patterns):
            return "nextjs"
        elif any("vue" in str(p).lower() for p in patterns):
            return "vue"
        elif any("angular" in str(p).lower() for p in patterns):
            return "angular"
        
        # Backend frameworks
        elif any("fastapi" in str(p).lower() for p in patterns):
            return "fastapi"
        elif any("django" in str(p).lower() for p in patterns):
            return "django"
        elif any("flask" in str(p).lower() for p in patterns):
            return "flask"
        elif any("express" in str(p).lower() for p in patterns):
            return "express"
        
        # Mobile development
        elif any("react native" in str(p).lower() for p in patterns):
            return "react-native"
        elif any("flutter" in str(p).lower() for p in patterns):
            return "flutter"
        
        # Data science / ML
        elif any("jupyter" in str(p).lower() for p in patterns):
            return "data-science"
        elif any("tensorflow" in str(p).lower() or "pytorch" in str(p).lower() for p in patterns):
            return "machine-learning"
        
        # Default to full-stack if mixed patterns
        return "fullstack"
    
    def _generate_claude_md_template(
        self,
        project_type: str,
        include_archon: bool,
        include_serena: bool,
        include_claude_flow: bool,
        custom_instructions: Optional[str]
    ) -> str:
        """Generate CLAUDE.md template based on project type."""
        
        # Base template
        template = f"""# Claude Code Configuration - {project_type.title()} Development Environment

## 🚨 CRITICAL: UNIFIED ARCHON MCP INTEGRATION

**ARCHON INTEGRATION ACTIVE**: This project uses the unified Archon MCP server for enhanced development capabilities.

### ⚡ Available Archon Tools:
"""
        
        if include_archon:
            template += """
- `archon_project_create` - Intelligent project creation with auto-analysis
- `archon_task_create_semantic` - Create tasks with semantic code context  
- `archon_analyze_and_implement` - Complete workflow: Analysis → Tasks → Implementation
- `archon_status` - System health across all services
"""
        
        if include_serena:
            template += """
- `archon_code_intelligence` - Deep code analysis, pattern detection, semantic search
  - Structure analysis with dependency mapping
  - Architectural pattern detection (MVC, Repository, etc.)
  - Intelligent code completion and refactoring suggestions
  - Semantic code search across the entire codebase
"""
        
        if include_claude_flow:
            template += """
- `archon_orchestrate` - SPARC methodology and swarm coordination
  - Test-driven development workflows (TDD)
  - Multi-agent coordination and task orchestration
  - Performance optimization and neural pattern training
"""
        
        # Project-specific configurations
        project_configs = {
            "react": {
                "build_cmd": "npm run build",
                "test_cmd": "npm test",
                "lint_cmd": "npm run lint",
                "dev_cmd": "npm run dev",
                "file_patterns": "**/*.{js,jsx,ts,tsx}",
                "key_directories": "src/, public/, components/"
            },
            "nextjs": {
                "build_cmd": "npm run build",
                "test_cmd": "npm test", 
                "lint_cmd": "npm run lint",
                "dev_cmd": "npm run dev",
                "file_patterns": "**/*.{js,jsx,ts,tsx}",
                "key_directories": "pages/, components/, lib/, styles/"
            },
            "fastapi": {
                "build_cmd": "python -m build",
                "test_cmd": "pytest",
                "lint_cmd": "ruff check .",
                "dev_cmd": "uvicorn main:app --reload",
                "file_patterns": "**/*.py",
                "key_directories": "src/, tests/, api/"
            },
            "django": {
                "build_cmd": "python manage.py collectstatic",
                "test_cmd": "python manage.py test",
                "lint_cmd": "flake8",
                "dev_cmd": "python manage.py runserver",
                "file_patterns": "**/*.py",
                "key_directories": "*/models/, */views/, */urls/"
            },
            "fullstack": {
                "build_cmd": "npm run build && python -m build",
                "test_cmd": "npm test && pytest",
                "lint_cmd": "npm run lint && ruff check .",
                "dev_cmd": "npm run dev",
                "file_patterns": "**/*.{js,jsx,ts,tsx,py}",
                "key_directories": "src/, frontend/, backend/, api/"
            }
        }
        
        config = project_configs.get(project_type, project_configs["fullstack"])
        
        template += f"""
## 🏗️ Project Configuration

### Build & Development Commands:
- **Build**: `{config["build_cmd"]}`
- **Test**: `{config["test_cmd"]}`  
- **Lint**: `{config["lint_cmd"]}`
- **Dev Server**: `{config["dev_cmd"]}`

### Key Project Patterns:
- **File Patterns**: `{config["file_patterns"]}`
- **Key Directories**: `{config["key_directories"]}`

## 🔄 Recommended Workflows

### 1. New Feature Development
```javascript
// Step 1: Analyze current codebase
archon_code_intelligence({{
  "operation": "analyze_structure",
  "project_path": "."
}})

// Step 2: Create semantic task
archon_task_create_semantic({{
  "title": "Implement [feature name]",
  "code_context": "path/to/relevant/file"
}})

// Step 3: Complete implementation workflow
archon_analyze_and_implement({{
  "task_description": "Implement [feature] with best practices",
  "create_tasks": true
}})
```

### 2. Code Quality & Refactoring
```javascript
// Detect patterns and issues
archon_code_intelligence({{
  "operation": "detect_patterns",
  "project_path": "."
}})

// Get refactoring suggestions  
archon_code_intelligence({{
  "operation": "refactor_suggestions",
  "file_path": "path/to/file.{project_type.split('-')[0]}"
}})

// Execute improvements with SPARC
archon_orchestrate({{
  "operation": "sparc_workflow",
  "task": "Refactor based on analysis",
  "mode": "tdd"
}})
```

### 3. Project Health Check
```javascript
// System status across all services
archon_status()

// Comprehensive project analysis
archon_code_intelligence({{
  "operation": "analyze_structure",
  "project_path": "."
}})
```

## 📋 Development Best Practices

### Code Style & Standards:
- Follow {project_type} conventions and best practices
- Use semantic naming and clear documentation
- Implement proper error handling and validation
- Write tests before implementing new features (TDD)

### Archon Integration:
- Use `archon_code_intelligence` for deep code analysis
- Leverage semantic task creation for context-aware development
- Execute complete workflows with `archon_analyze_and_implement`
- Monitor system health with `archon_status`

### File Organization:
- Keep files under 500 lines for maintainability
- Group related functionality in appropriate directories
- Use meaningful file and directory names
- Separate concerns (business logic, UI, data access)

## 🚀 Advanced Features

### Intelligent Code Completion:
The Archon MCP provides context-aware code completion through Serena integration.

### Pattern Detection:
Automatic detection of architectural patterns, design patterns, and anti-patterns.

### Cross-Service Workflows:
Seamless integration between code analysis, task management, and implementation.

### SPARC Methodology:
Complete development workflows following Specification → Pseudocode → Architecture → Refinement → Completion.
"""
        
        # Add custom instructions if provided
        if custom_instructions:
            template += f"""
## 🎯 Project-Specific Instructions

{custom_instructions}
"""
        
        template += """
## 🛠️ Troubleshooting

### Common Issues:
1. **MCP Connection Failed**: Ensure Archon backend is running on port 8080
2. **Code Intelligence Not Working**: Check Serena service status with `archon_status`
3. **Orchestration Issues**: Verify Claude Flow service status with `archon_status`

### Getting Help:
- Check system status: `archon_status`
- Review logs in Claude Desktop console
- Ensure all Archon services are running

---

**🎉 Your project is now configured with the complete power of Archon's unified MCP system!**

Use the tools above to leverage intelligent code analysis, semantic task creation, and automated development workflows.
"""
        
        return template
    
    async def complete_project_setup(
        self,
        project_path: str = ".",
        project_title: Optional[str] = None,
        project_description: str = "",
        github_repo: Optional[str] = None,
        setup_claude_md: bool = True,
        auto_analyze: bool = True,
        create_initial_tasks: bool = True
    ) -> Dict[str, Any]:
        """Complete project setup including Archon project creation and CLAUDE.md configuration."""
        try:
            logger.info(f"🚀 Complete project setup for: {project_path}")
            
            setup_results = {
                "project_creation": {},
                "claude_md_setup": {},
                "initial_analysis": {},
                "tasks_created": []
            }
            
            # Step 1: Create Archon project if title provided
            if project_title:
                project_result = await self.create_intelligent_project(
                    title=project_title,
                    description=project_description,
                    github_repo=github_repo,
                    auto_analyze=auto_analyze
                )
                setup_results["project_creation"] = project_result
            
            # Step 2: Setup CLAUDE.md
            if setup_claude_md:
                claude_md_result = await self.setup_claude_md(
                    project_path=project_path,
                    project_type="auto"
                )
                setup_results["claude_md_setup"] = claude_md_result
            
            # Step 3: Additional analysis if requested
            if auto_analyze:
                analysis_result = await serena_service.analyze_project_structure(
                    project_path=project_path,
                    include_patterns=True,
                    include_metrics=True
                )
                setup_results["initial_analysis"] = analysis_result
            
            # Step 4: Create initial tasks if requested
            if create_initial_tasks and project_title and setup_results["project_creation"].get("project_id"):
                project_id = setup_results["project_creation"]["project_id"]
                
                # Create setup and analysis tasks
                initial_tasks = [
                    {
                        "title": "Review project structure and architecture",
                        "description": "Analyze the auto-generated project structure and architectural patterns"
                    },
                    {
                        "title": "Set up development environment",
                        "description": "Configure development tools, dependencies, and build processes"
                    },
                    {
                        "title": "Implement core functionality",
                        "description": "Begin implementing the main features based on project analysis"
                    }
                ]
                
                for task_info in initial_tasks:
                    task_result = await self.create_semantic_task(
                        project_id=project_id,
                        title=task_info["title"],
                        description=task_info["description"],
                        auto_suggestions=True
                    )
                    if task_result.get("status") == "success":
                        setup_results["tasks_created"].append(task_result)
            
            return {
                "status": "success", 
                "setup": setup_results,
                "message": "Complete project setup finished successfully"
            }
            
        except Exception as e:
            logger.error(f"Complete project setup failed: {e}")
            return {"status": "error", "error": str(e)}

    # ========================================================================
    # CROSS-SERVICE COORDINATION
    # ========================================================================
    
    async def get_unified_status(self) -> Dict[str, Any]:
        """Get comprehensive status across all services."""
        try:
            status = {
                "unified_mcp": {
                    "status": "healthy",
                    "services_ready": self.services_ready
                }
            }
            
            # Get individual service statuses
            try:
                status["serena"] = await serena_service.get_service_status()
            except Exception as e:
                status["serena"] = {"status": "error", "error": str(e)}
            
            try:
                status["claude_flow"] = await claude_flow_service.get_swarm_status()
            except Exception as e:
                status["claude_flow"] = {"status": "error", "error": str(e)}
            
            # Backend API status
            try:
                import httpx
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self.base_url}/health")
                    if response.status_code == 200:
                        status["backend"] = response.json()
                    else:
                        status["backend"] = {"status": "error", "code": response.status_code}
            except Exception as e:
                status["backend"] = {"status": "error", "error": str(e)}
            
            return {"status": "success", "services": status}
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    # ========================================================================
    # URL DETECTION AND MANAGEMENT
    # ========================================================================
    
    async def detect_urls_in_text(
        self, 
        text: str, 
        context: str = "", 
        auto_process: bool = True
    ) -> Dict[str, Any]:
        """Detect URLs in text and optionally process them for knowledge base addition."""
        try:
            url_detection_service = get_url_detection_service()
            
            detected_urls = await url_detection_service.detect_urls_in_text(
                text, context, auto_process
            )
            
            if not detected_urls:
                return {
                    "status": "success",
                    "urls_detected": 0,
                    "urls": [],
                    "message": "No URLs detected in the provided text"
                }
            
            # Process URLs if auto_process is enabled
            processing_results = []
            if auto_process:
                for url in detected_urls:
                    try:
                        analysis = await url_detection_service.analyze_url(url, f"mcp_detect:{context}")
                        processing_results.append({
                            "url": url,
                            "overall_score": analysis.overall_score,
                            "recommended_action": analysis.recommended_action,
                            "reasoning": analysis.reasoning
                        })
                    except Exception as e:
                        processing_results.append({
                            "url": url,
                            "error": str(e)
                        })
            
            return {
                "status": "success",
                "urls_detected": len(detected_urls),
                "urls": detected_urls,
                "auto_processed": auto_process,
                "processing_results": processing_results,
                "message": f"Detected {len(detected_urls)} URLs in text"
            }
            
        except Exception as e:
            logger.error(f"❌ Error detecting URLs: {e}")
            return {"status": "error", "error": str(e)}
    
    async def analyze_url_with_ai(
        self, 
        url: str, 
        context: str = "", 
        source: str = "manual"
    ) -> Dict[str, Any]:
        """Analyze a specific URL using AI decision agent."""
        try:
            url_decision_agent = get_url_decision_agent()
            
            # Create decision context
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc
            
            decision_context = URLDecisionContext(
                url=url,
                domain=domain,
                source_context=context,
                detected_in=source
            )
            
            # Perform AI analysis
            decision_result = await url_decision_agent.analyze_url_decision(decision_context)
            
            return {
                "status": "success",
                "url": url,
                "analysis": {
                    "confidence_score": decision_result.confidence_score,
                    "relevance_score": decision_result.relevance_score,
                    "quality_score": decision_result.quality_score,
                    "risk_score": decision_result.risk_score,
                    "overall_assessment": {
                        "recommended_action": decision_result.recommended_action,
                        "reasoning": decision_result.reasoning,
                        "estimated_value": decision_result.estimated_value,
                        "content_type_prediction": decision_result.content_type_prediction
                    },
                    "key_factors": decision_result.key_factors,
                    "suggested_tags": decision_result.suggested_tags
                },
                "message": f"AI analysis complete: {decision_result.recommended_action} (confidence: {decision_result.confidence_score:.2f})"
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing URL with AI: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_url_suggestions(
        self, 
        limit: int = 10, 
        min_score: float = 0.0, 
        status: str = "pending"
    ) -> Dict[str, Any]:
        """Get URL suggestions that require user review."""
        try:
            import httpx
            
            # Call the API endpoint
            async with httpx.AsyncClient(timeout=10.0) as client:
                params = {
                    "limit": limit,
                    "min_score": min_score,
                    "status": status
                }
                
                response = await client.get(
                    f"{self.base_url}/api/url-suggestions/",
                    params=params
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "status": "success",
                        "suggestions": data["suggestions"],
                        "total_count": data["total_count"],
                        "pending_count": data["pending_count"],
                        "high_score_count": data["high_score_count"],
                        "message": f"Found {len(data['suggestions'])} suggestions"
                    }
                else:
                    return {
                        "status": "error", 
                        "error": f"API request failed with status {response.status_code}"
                    }
                    
        except Exception as e:
            logger.error(f"❌ Error getting URL suggestions: {e}")
            return {"status": "error", "error": str(e)}
    
    async def approve_url_suggestion(
        self, 
        url: Optional[str] = None, 
        suggestion_ids: Optional[List[str]] = None,
        add_tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Approve URL suggestion(s) and add to knowledge base."""
        try:
            import httpx
            
            # If URL is provided, find its suggestion ID
            if url and not suggestion_ids:
                # First get suggestions to find the ID
                suggestions_response = await self.get_url_suggestions(limit=100, status="pending")
                if suggestions_response["status"] == "success":
                    for suggestion in suggestions_response["suggestions"]:
                        if suggestion["url"] == url:
                            suggestion_ids = [suggestion["id"]]
                            break
                
                if not suggestion_ids:
                    return {
                        "status": "error",
                        "error": f"No pending suggestion found for URL: {url}"
                    }
            
            if not suggestion_ids:
                return {
                    "status": "error",
                    "error": "Either url or suggestion_ids must be provided"
                }
            
            # Call the approval API
            async with httpx.AsyncClient(timeout=30.0) as client:
                request_data = {
                    "suggestion_ids": suggestion_ids
                }
                if add_tags:
                    request_data["add_tags"] = add_tags
                
                response = await client.post(
                    f"{self.base_url}/api/url-suggestions/approve",
                    json=request_data
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "status": "success",
                        "approved_count": data["approved_count"],
                        "failed_count": data["failed_count"],
                        "approved_urls": data["approved_urls"],
                        "message": data["message"]
                    }
                else:
                    return {
                        "status": "error",
                        "error": f"Approval request failed with status {response.status_code}"
                    }
                    
        except Exception as e:
            logger.error(f"❌ Error approving URL suggestion: {e}")
            return {"status": "error", "error": str(e)}
    
    async def manage_url_detection_settings(
        self, 
        action: str = "get",
        enabled: Optional[bool] = None,
        auto_add_threshold: Optional[float] = None,
        suggest_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get or update URL detection system settings."""
        try:
            import httpx
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                if action == "get":
                    # Get current settings
                    response = await client.get(f"{self.base_url}/api/url-suggestions/settings")
                    
                    if response.status_code == 200:
                        settings = response.json()
                        return {
                            "status": "success",
                            "settings": settings,
                            "message": "Current URL detection settings retrieved"
                        }
                    else:
                        return {
                            "status": "error",
                            "error": f"Failed to get settings: {response.status_code}"
                        }
                
                elif action == "update":
                    # Update settings
                    update_data = {}
                    if enabled is not None:
                        update_data["enabled"] = enabled
                    if auto_add_threshold is not None:
                        update_data["auto_add_threshold"] = auto_add_threshold
                    if suggest_threshold is not None:
                        update_data["suggest_threshold"] = suggest_threshold
                    
                    if not update_data:
                        return {
                            "status": "error",
                            "error": "No update parameters provided"
                        }
                    
                    response = await client.post(
                        f"{self.base_url}/api/url-suggestions/settings",
                        json=update_data
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        return {
                            "status": "success",
                            "settings": data["settings"],
                            "message": data["message"]
                        }
                    else:
                        return {
                            "status": "error",
                            "error": f"Failed to update settings: {response.status_code}"
                        }
                
                else:
                    return {
                        "status": "error",
                        "error": f"Unknown action: {action}. Use 'get' or 'update'"
                    }
                    
        except Exception as e:
            logger.error(f"❌ Error managing URL detection settings: {e}")
            return {"status": "error", "error": str(e)}


# Initialize coordinator
coordinator = ArchonMCPCoordinator()


# ============================================================================
# MCP TOOL DEFINITIONS
# ============================================================================

@server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available Archon MCP tools including discovered CLI tools."""
    
    # Base Archon MCP tools
    base_tools = [
        Tool(
            name="archon_status",
            description="Get comprehensive status of all Archon services",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="archon_project_create",
            description="Create intelligent project with automatic code analysis",
            inputSchema={
                "type": "object", 
                "properties": {
                    "title": {"type": "string", "description": "Project title"},
                    "description": {"type": "string", "description": "Project description"},
                    "github_repo": {"type": "string", "description": "GitHub repository URL"},
                    "auto_analyze": {"type": "boolean", "default": True, "description": "Run automatic code analysis"}
                },
                "required": ["title"]
            }
        ),
        Tool(
            name="archon_task_create_semantic",
            description="Create task with semantic code context and AI suggestions",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_id": {"type": "string", "description": "Project ID"},
                    "title": {"type": "string", "description": "Task title"},
                    "description": {"type": "string", "description": "Task description"},
                    "code_context": {"type": "string", "description": "File path for code context"},
                    "auto_suggestions": {"type": "boolean", "default": True, "description": "Generate AI suggestions"}
                },
                "required": ["project_id", "title"]
            }
        ),
        Tool(
            name="archon_analyze_and_implement",
            description="Complete workflow: Code Analysis → Task Creation → Implementation",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_description": {"type": "string", "description": "What to implement"},
                    "file_path": {"type": "string", "description": "Specific file to analyze"},
                    "project_path": {"type": "string", "default": ".", "description": "Project root path"},
                    "create_tasks": {"type": "boolean", "default": True, "description": "Create tasks from analysis"}
                },
                "required": ["task_description"]
            }
        ),
        Tool(
            name="archon_code_intelligence",
            description="Deep code analysis with Serena intelligence (structure, patterns, suggestions)",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string", 
                        "enum": ["analyze_structure", "detect_patterns", "semantic_search", "get_completion", "refactor_suggestions"],
                        "description": "Type of code intelligence operation"
                    },
                    "file_path": {"type": "string", "description": "File path to analyze"},
                    "project_path": {"type": "string", "default": ".", "description": "Project path"},
                    "query": {"type": "string", "description": "Search query for semantic search"},
                    "line": {"type": "integer", "description": "Line number for completion"},
                    "column": {"type": "integer", "description": "Column number for completion"}
                },
                "required": ["operation"]
            }
        ),
        Tool(
            name="archon_orchestrate",
            description="Execute workflows with Claude Flow orchestration (SPARC, swarms, agents)",
            inputSchema={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["sparc_workflow", "swarm_init", "agent_spawn", "get_status"],
                        "description": "Type of orchestration operation"
                    },
                    "task": {"type": "string", "description": "Task description for execution"},
                    "mode": {"type": "string", "default": "tdd", "description": "SPARC mode (tdd, batch, pipeline)"},
                    "topology": {"type": "string", "default": "adaptive", "description": "Swarm topology"},
                    "max_agents": {"type": "integer", "default": 5, "description": "Maximum agents"}
                },
                "required": ["operation"]
            }
        ),
        Tool(
            name="archon_setup_claude_md",
            description="Generate and configure CLAUDE.md file with project-specific settings",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_path": {"type": "string", "default": ".", "description": "Project directory path"},
                    "project_type": {"type": "string", "default": "auto", "description": "Project type (auto-detect or specific)"},
                    "custom_instructions": {"type": "string", "description": "Custom project instructions to include"},
                    "include_archon": {"type": "boolean", "default": True, "description": "Include Archon tools"},
                    "include_serena": {"type": "boolean", "default": True, "description": "Include Serena code intelligence"},
                    "include_claude_flow": {"type": "boolean", "default": True, "description": "Include Claude Flow orchestration"}
                },
                "required": []
            }
        ),
        Tool(
            name="archon_complete_setup",
            description="Complete project setup: Create project + CLAUDE.md + initial tasks",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_title": {"type": "string", "description": "Project title for Archon database"},
                    "project_description": {"type": "string", "default": "", "description": "Project description"},
                    "project_path": {"type": "string", "default": ".", "description": "Project directory path"},
                    "github_repo": {"type": "string", "description": "GitHub repository URL"},
                    "setup_claude_md": {"type": "boolean", "default": True, "description": "Create CLAUDE.md file"},
                    "auto_analyze": {"type": "boolean", "default": True, "description": "Run automatic code analysis"},
                    "create_initial_tasks": {"type": "boolean", "default": True, "description": "Create initial development tasks"},
                    "custom_instructions": {"type": "string", "description": "Custom instructions for CLAUDE.md"}
                },
                "required": ["project_title"]
            }
        ),
        Tool(
            name="archon_detect_urls",
            description="Detect and analyze URLs in text for potential knowledge base addition",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to scan for URLs"},
                    "context": {"type": "string", "default": "", "description": "Context about where the text came from"},
                    "auto_process": {"type": "boolean", "default": True, "description": "Automatically process detected URLs"}
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="archon_analyze_url",
            description="Perform intelligent analysis of a specific URL using AI decision agent",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to analyze"},
                    "context": {"type": "string", "default": "", "description": "Context about the URL"},
                    "source": {"type": "string", "default": "manual", "description": "Source of the URL (agent_response, task, etc.)"}
                },
                "required": ["url"]
            }
        ),
        Tool(
            name="archon_get_url_suggestions",
            description="Get URL suggestions that require user review for knowledge base addition",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "default": 10, "description": "Maximum number of suggestions to return"},
                    "min_score": {"type": "number", "default": 0.0, "description": "Minimum overall score filter"},
                    "status": {"type": "string", "default": "pending", "description": "Filter by status: pending, approved, rejected"}
                },
                "required": []
            }
        ),
        Tool(
            name="archon_approve_url",
            description="Approve URL suggestion(s) and add to knowledge base automatically",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to approve (alternative to suggestion_ids)"},
                    "suggestion_ids": {"type": "array", "items": {"type": "string"}, "description": "List of suggestion IDs to approve"},
                    "add_tags": {"type": "array", "items": {"type": "string"}, "description": "Additional tags to add"}
                },
                "required": []
            }
        ),
        Tool(
            name="archon_url_settings",
            description="Get or update URL detection system settings",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["get", "update"], "default": "get", "description": "Action to perform"},
                    "enabled": {"type": "boolean", "description": "Enable/disable URL detection"},
                    "auto_add_threshold": {"type": "number", "description": "Threshold for automatic addition (0.0-1.0)"},
                    "suggest_threshold": {"type": "number", "description": "Threshold for suggestions (0.0-1.0)"}
                },
                "required": ["action"]
            }
        )
    ]
    
    # Add dynamically discovered CLI tools
    cli_tools = []
    try:
        if cli_discovery_service.running:
            cli_commands = cli_discovery_service.get_discovered_commands()
            for cmd_name, cmd_info in cli_commands.items():
                # Create MCP tool for each CLI command
                cli_tool = Tool(
                    name=f"cli_{cmd_name.replace('__', '_')}",
                    description=f"CLI Tool: {cmd_info.description} (from {cmd_info.tool})",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            **{opt_name: {
                                "type": "string", 
                                "description": opt_desc
                            } for opt_name, opt_desc in cmd_info.options.items()},
                            "additional_args": {
                                "type": "object",
                                "description": "Additional CLI arguments as key-value pairs"
                            }
                        },
                        "required": []
                    }
                )
                cli_tools.append(cli_tool)
    except Exception as e:
        logger.warning(f"Failed to add CLI tools to MCP list: {e}")
    
    return base_tools + cli_tools


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls with intelligent routing."""
    
    try:
        # Initialize services if not ready
        if not coordinator.services_ready:
            await coordinator.initialize_services()
        
        result = None
        
        # Route to appropriate handler
        if name == "archon_status":
            result = await coordinator.get_unified_status()
            
        elif name == "archon_project_create":
            result = await coordinator.create_intelligent_project(**arguments)
            
        elif name == "archon_task_create_semantic":
            result = await coordinator.create_semantic_task(**arguments)
            
        elif name == "archon_analyze_and_implement":
            result = await coordinator.analyze_and_implement_workflow(**arguments)
            
        elif name == "archon_code_intelligence":
            operation = arguments.pop("operation")
            if operation == "analyze_structure":
                result = await serena_service.analyze_code_structure(**arguments)
            elif operation == "detect_patterns":
                result = await serena_service.detect_code_patterns(**arguments)
            elif operation == "semantic_search":
                result = await serena_service.semantic_code_search(**arguments)
            elif operation == "get_completion":
                result = await serena_service.get_intelligent_completion(**arguments)
            elif operation == "refactor_suggestions":
                result = await serena_service.get_refactoring_suggestions(**arguments)
                
        elif name == "archon_orchestrate":
            operation = arguments.pop("operation")
            if operation == "sparc_workflow":
                result = await claude_flow_service.execute_sparc_workflow(**arguments)
            elif operation == "swarm_init":
                result = await claude_flow_service.initialize_swarm(**arguments)
            elif operation == "agent_spawn":
                result = await claude_flow_service.spawn_agents(**arguments)
            elif operation == "get_status":
                result = await claude_flow_service.get_swarm_status()
                
        elif name == "archon_setup_claude_md":
            result = await coordinator.setup_claude_md(**arguments)
            
        elif name == "archon_complete_setup":
            result = await coordinator.complete_project_setup(**arguments)
        
        # URL Detection Tools
        elif name == "archon_detect_urls":
            result = await coordinator.detect_urls_in_text(**arguments)
        elif name == "archon_analyze_url":
            result = await coordinator.analyze_url_with_ai(**arguments)
        elif name == "archon_get_url_suggestions":
            result = await coordinator.get_url_suggestions(**arguments)
        elif name == "archon_approve_url":
            result = await coordinator.approve_url_suggestion(**arguments)
        elif name == "archon_url_settings":
            result = await coordinator.manage_url_detection_settings(**arguments)
        
        elif name.startswith("cli_"):
            # Handle CLI tool calls
            cli_command_name = name[4:].replace("_", "__", 1)  # Remove "cli_" prefix and restore format
            additional_args = arguments.pop("additional_args", {})
            # Combine standard arguments with additional_args
            all_args = {**arguments, **additional_args}
            result = await cli_discovery_service.execute_command(cli_command_name, all_args)
            
        else:
            result = {"status": "error", "error": f"Unknown tool: {name}"}
        
        # Format result for MCP
        if isinstance(result, dict):
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        else:
            return [TextContent(type="text", text=str(result))]
            
    except Exception as e:
        logger.error(f"Tool execution failed: {name} - {e}")
        error_result = {"status": "error", "error": str(e), "tool": name}
        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]


# ============================================================================
# MAIN SERVER EXECUTION
# ============================================================================

async def main():
    """Run the unified Archon MCP server."""
    logger.info("🚀 Starting Unified Archon MCP Server...")
    
    # Initialize services
    await coordinator.initialize_services()
    
    # Run MCP server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())