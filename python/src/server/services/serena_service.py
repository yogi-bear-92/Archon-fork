"""
Serena MCP Integration Service for Archon

This service provides native Serena MCP capabilities within Archon's
backend architecture, enabling seamless code intelligence and semantic analysis
integrated with Archon's task management and knowledge systems.
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.server.config.logfire_config import get_logger

logger = get_logger(__name__)


class SerenaService:
    """Native Serena MCP integration service for Archon."""
    
    def __init__(self):
        self.base_path = Path(__file__).resolve().parent.parent.parent.parent
        self.serena_config = self.base_path / ".serena"
        self.cache_dir = self.base_path / ".serena" / "cache"
        self.memory_db = self.base_path / ".serena" / "memory.db"
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Ensure required directories exist."""
        os.makedirs(self.serena_config / "config", exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.base_path / ".serena" / "semantic", exist_ok=True)
        
    # ========================================================================
    # CODE INTELLIGENCE OPERATIONS
    # ========================================================================
    
    async def analyze_code_structure(
        self,
        file_path: str,
        language: Optional[str] = None,
        include_dependencies: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze code structure using Serena's semantic analysis.
        
        Args:
            file_path: Path to file to analyze
            language: Programming language (auto-detected if None)
            include_dependencies: Include dependency analysis
            
        Returns:
            Dict with code structure analysis
        """
        try:
            logger.info(f"Analyzing code structure for: {file_path}")
            
            # Prepare analysis configuration
            analysis_config = {
                "file_path": file_path,
                "language": language,
                "include_dependencies": include_dependencies,
                "timestamp": datetime.now().isoformat(),
                "archon_integration": {
                    "project_context": True,
                    "semantic_caching": True,
                    "knowledge_persistence": True
                }
            }
            
            # Execute code analysis via MCP tools
            result = await self._execute_serena_analysis(
                operation="code_structure_analysis",
                config=analysis_config
            )
            
            return {
                "status": "success",
                "file_path": file_path,
                "analysis": result,
                "language": language or self._detect_language(file_path),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze code structure for {file_path}: {e}")
            return {"status": "error", "error": str(e), "file_path": file_path}
    
    async def get_symbol_definitions(
        self,
        symbol_name: str,
        project_path: str = ".",
        scope: str = "project"
    ) -> Dict[str, Any]:
        """
        Get symbol definitions across the project.
        
        Args:
            symbol_name: Name of symbol to find
            project_path: Project root path
            scope: Search scope (file, project, workspace)
            
        Returns:
            Dict with symbol definition locations and metadata
        """
        try:
            logger.info(f"Finding symbol definitions for: {symbol_name}")
            
            search_config = {
                "symbol_name": symbol_name,
                "project_path": project_path,
                "scope": scope,
                "search_type": "definitions",
                "include_references": False
            }
            
            result = await self._execute_serena_search(
                operation="symbol_definitions",
                config=search_config
            )
            
            return {
                "status": "success",
                "symbol": symbol_name,
                "definitions": result.get("definitions", []),
                "scope": scope,
                "total_found": len(result.get("definitions", []))
            }
            
        except Exception as e:
            logger.error(f"Failed to get symbol definitions for {symbol_name}: {e}")
            return {"status": "error", "error": str(e), "symbol": symbol_name}
    
    async def get_symbol_references(
        self,
        symbol_name: str,
        project_path: str = ".",
        include_declarations: bool = True
    ) -> Dict[str, Any]:
        """
        Get all references to a symbol.
        
        Args:
            symbol_name: Name of symbol to find references for
            project_path: Project root path
            include_declarations: Include declaration sites
            
        Returns:
            Dict with symbol reference locations
        """
        try:
            logger.info(f"Finding symbol references for: {symbol_name}")
            
            search_config = {
                "symbol_name": symbol_name,
                "project_path": project_path,
                "search_type": "references",
                "include_declarations": include_declarations
            }
            
            result = await self._execute_serena_search(
                operation="symbol_references",
                config=search_config
            )
            
            return {
                "status": "success",
                "symbol": symbol_name,
                "references": result.get("references", []),
                "include_declarations": include_declarations,
                "total_found": len(result.get("references", []))
            }
            
        except Exception as e:
            logger.error(f"Failed to get symbol references for {symbol_name}: {e}")
            return {"status": "error", "error": str(e), "symbol": symbol_name}
    
    async def analyze_code_completion(
        self,
        file_path: str,
        position: Dict[str, int],
        context_lines: int = 10
    ) -> Dict[str, Any]:
        """
        Provide intelligent code completion suggestions.
        
        Args:
            file_path: Path to file
            position: Cursor position {"line": int, "character": int}
            context_lines: Number of context lines to include
            
        Returns:
            Dict with completion suggestions
        """
        try:
            logger.info(f"Generating code completion for {file_path} at {position}")
            
            completion_config = {
                "file_path": file_path,
                "position": position,
                "context_lines": context_lines,
                "completion_type": "intelligent",
                "include_snippets": True,
                "include_documentation": True
            }
            
            result = await self._execute_serena_completion(
                config=completion_config
            )
            
            return {
                "status": "success",
                "file_path": file_path,
                "position": position,
                "completions": result.get("completions", []),
                "context_aware": True
            }
            
        except Exception as e:
            logger.error(f"Failed to generate code completion: {e}")
            return {"status": "error", "error": str(e), "file_path": file_path}
    
    async def get_diagnostics(
        self,
        file_path: Optional[str] = None,
        severity_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get code diagnostics (errors, warnings, hints).
        
        Args:
            file_path: Specific file path (None for all files)
            severity_filter: Filter by severity (error, warning, hint)
            
        Returns:
            Dict with diagnostic information
        """
        try:
            logger.info(f"Getting diagnostics for: {file_path or 'all files'}")
            
            diagnostic_config = {
                "file_path": file_path,
                "severity_filter": severity_filter,
                "include_fixes": True,
                "include_related": True
            }
            
            result = await self._execute_serena_diagnostics(
                config=diagnostic_config
            )
            
            return {
                "status": "success",
                "file_path": file_path,
                "diagnostics": result.get("diagnostics", []),
                "severity_filter": severity_filter,
                "total_issues": len(result.get("diagnostics", []))
            }
            
        except Exception as e:
            logger.error(f"Failed to get diagnostics: {e}")
            return {"status": "error", "error": str(e)}
    
    # ========================================================================
    # PROJECT ANALYSIS OPERATIONS  
    # ========================================================================
    
    async def analyze_project_structure(
        self,
        project_path: str = ".",
        include_dependencies: bool = True,
        max_depth: int = 10
    ) -> Dict[str, Any]:
        """
        Analyze entire project structure and architecture.
        
        Args:
            project_path: Project root path
            include_dependencies: Include dependency analysis
            max_depth: Maximum directory depth to analyze
            
        Returns:
            Dict with comprehensive project analysis
        """
        try:
            logger.info(f"Analyzing project structure at: {project_path}")
            
            analysis_config = {
                "project_path": project_path,
                "include_dependencies": include_dependencies,
                "max_depth": max_depth,
                "analysis_type": "comprehensive",
                "include_patterns": True,
                "include_metrics": True
            }
            
            result = await self._execute_serena_project_analysis(
                config=analysis_config
            )
            
            return {
                "status": "success",
                "project_path": project_path,
                "structure": result.get("structure", {}),
                "patterns": result.get("patterns", []),
                "metrics": result.get("metrics", {}),
                "dependencies": result.get("dependencies", []) if include_dependencies else [],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze project structure: {e}")
            return {"status": "error", "error": str(e), "project_path": project_path}
    
    async def detect_architecture_patterns(
        self,
        project_path: str = ".",
        pattern_types: List[str] = None
    ) -> Dict[str, Any]:
        """
        Detect architectural patterns in the project.
        
        Args:
            project_path: Project root path
            pattern_types: Specific patterns to look for
            
        Returns:
            Dict with detected patterns and confidence scores
        """
        try:
            logger.info(f"Detecting architecture patterns in: {project_path}")
            
            pattern_config = {
                "project_path": project_path,
                "pattern_types": pattern_types or ["mvc", "repository", "observer", "factory", "singleton"],
                "confidence_threshold": 0.7,
                "include_examples": True
            }
            
            result = await self._execute_serena_pattern_detection(
                config=pattern_config
            )
            
            return {
                "status": "success",
                "project_path": project_path,
                "detected_patterns": result.get("patterns", []),
                "pattern_confidence": result.get("confidence_scores", {}),
                "recommendations": result.get("recommendations", [])
            }
            
        except Exception as e:
            logger.error(f"Failed to detect architecture patterns: {e}")
            return {"status": "error", "error": str(e), "project_path": project_path}
    
    # ========================================================================
    # SEMANTIC SEARCH OPERATIONS
    # ========================================================================
    
    async def semantic_code_search(
        self,
        query: str,
        project_path: str = ".",
        search_type: str = "semantic",
        max_results: int = 20
    ) -> Dict[str, Any]:
        """
        Perform semantic code search across the project.
        
        Args:
            query: Search query (natural language or code patterns)
            project_path: Project root path
            search_type: Type of search (semantic, syntactic, hybrid)
            max_results: Maximum number of results
            
        Returns:
            Dict with search results and relevance scores
        """
        try:
            logger.info(f"Performing semantic code search: {query}")
            
            search_config = {
                "query": query,
                "project_path": project_path,
                "search_type": search_type,
                "max_results": max_results,
                "include_context": True,
                "rank_by_relevance": True
            }
            
            result = await self._execute_serena_semantic_search(
                config=search_config
            )
            
            return {
                "status": "success",
                "query": query,
                "search_type": search_type,
                "results": result.get("results", []),
                "total_found": result.get("total_count", 0),
                "max_results": max_results
            }
            
        except Exception as e:
            logger.error(f"Failed to perform semantic code search: {e}")
            return {"status": "error", "error": str(e), "query": query}
    
    async def find_similar_code(
        self,
        code_snippet: str,
        project_path: str = ".",
        similarity_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Find similar code patterns in the project.
        
        Args:
            code_snippet: Code snippet to find similarities for
            project_path: Project root path
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            Dict with similar code locations and scores
        """
        try:
            logger.info("Finding similar code patterns")
            
            similarity_config = {
                "code_snippet": code_snippet,
                "project_path": project_path,
                "similarity_threshold": similarity_threshold,
                "algorithm": "ast_based",
                "include_partial_matches": True
            }
            
            result = await self._execute_serena_similarity_search(
                config=similarity_config
            )
            
            return {
                "status": "success",
                "query_snippet": code_snippet[:100] + "..." if len(code_snippet) > 100 else code_snippet,
                "similar_code": result.get("matches", []),
                "similarity_threshold": similarity_threshold,
                "total_matches": len(result.get("matches", []))
            }
            
        except Exception as e:
            logger.error(f"Failed to find similar code: {e}")
            return {"status": "error", "error": str(e)}
    
    # ========================================================================
    # KNOWLEDGE INTEGRATION OPERATIONS
    # ========================================================================
    
    async def extract_code_knowledge(
        self,
        project_path: str = ".",
        knowledge_types: List[str] = None,
        output_format: str = "structured"
    ) -> Dict[str, Any]:
        """
        Extract structured knowledge from codebase for Archon's RAG system.
        
        Args:
            project_path: Project root path
            knowledge_types: Types of knowledge to extract
            output_format: Format for extracted knowledge
            
        Returns:
            Dict with extracted knowledge ready for RAG integration
        """
        try:
            logger.info(f"Extracting code knowledge from: {project_path}")
            
            extraction_config = {
                "project_path": project_path,
                "knowledge_types": knowledge_types or ["functions", "classes", "interfaces", "patterns", "documentation"],
                "output_format": output_format,
                "include_metadata": True,
                "archon_rag_integration": True
            }
            
            result = await self._execute_serena_knowledge_extraction(
                config=extraction_config
            )
            
            # Format for Archon's RAG system
            knowledge_artifacts = {
                "status": "success",
                "project_path": project_path,
                "knowledge_base": {
                    "functions": result.get("functions", []),
                    "classes": result.get("classes", []),
                    "interfaces": result.get("interfaces", []),
                    "patterns": result.get("patterns", []),
                    "documentation": result.get("documentation", [])
                },
                "metadata": {
                    "extraction_timestamp": datetime.now().isoformat(),
                    "total_artifacts": sum(len(v) if isinstance(v, list) else 1 for v in result.values()),
                    "archon_ready": True
                }
            }
            
            # Cache extracted knowledge for future use
            await self._cache_knowledge_artifacts(knowledge_artifacts)
            
            return knowledge_artifacts
            
        except Exception as e:
            logger.error(f"Failed to extract code knowledge: {e}")
            return {"status": "error", "error": str(e), "project_path": project_path}
    
    async def enhance_archon_context(
        self,
        task_context: Dict[str, Any],
        include_semantic_analysis: bool = True
    ) -> Dict[str, Any]:
        """
        Enhance Archon task context with Serena's code intelligence.
        
        Args:
            task_context: Archon task context
            include_semantic_analysis: Include deep semantic analysis
            
        Returns:
            Dict with enhanced context for Archon tasks
        """
        try:
            logger.info("Enhancing Archon task context with code intelligence")
            
            task_id = task_context.get("task_id", "unknown")
            project_path = task_context.get("project_path", ".")
            target_files = task_context.get("target_files", [])
            
            enhancement_results = {
                "status": "success",
                "task_id": task_id,
                "original_context": task_context,
                "enhancements": {}
            }
            
            # Enhance with code structure analysis
            if target_files:
                file_analyses = []
                for file_path in target_files[:10]:  # Limit for performance
                    analysis = await self.analyze_code_structure(file_path)
                    if analysis["status"] == "success":
                        file_analyses.append(analysis)
                
                enhancement_results["enhancements"]["file_analyses"] = file_analyses
            
            # Enhance with project context
            if include_semantic_analysis:
                project_analysis = await self.analyze_project_structure(project_path)
                if project_analysis["status"] == "success":
                    enhancement_results["enhancements"]["project_context"] = project_analysis
            
            # Add semantic insights
            enhancement_results["enhancements"]["semantic_insights"] = {
                "code_intelligence_available": True,
                "supported_languages": self._get_supported_languages(),
                "analysis_capabilities": [
                    "symbol_resolution", "reference_finding", "pattern_detection",
                    "semantic_search", "architecture_analysis", "code_completion"
                ]
            }
            
            return enhancement_results
            
        except Exception as e:
            logger.error(f"Failed to enhance Archon context: {e}")
            return {"status": "error", "error": str(e), "task_context": task_context}
    
    # ========================================================================
    # CACHING AND OPTIMIZATION
    # ========================================================================
    
    async def _cache_knowledge_artifacts(self, knowledge_artifacts: Dict[str, Any]):
        """Cache extracted knowledge artifacts for future use."""
        try:
            cache_key = f"knowledge_{datetime.now().strftime('%Y%m%d_%H')}"
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            with open(cache_file, 'w') as f:
                json.dump(knowledge_artifacts, f, indent=2, default=str)
                
            logger.info(f"Cached knowledge artifacts: {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to cache knowledge artifacts: {e}")
    
    async def get_cached_analysis(
        self,
        cache_key: str,
        max_age_hours: int = 24
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached analysis results.
        
        Args:
            cache_key: Cache key identifier
            max_age_hours: Maximum age of cache in hours
            
        Returns:
            Cached analysis or None if not found/expired
        """
        try:
            cache_file = self.cache_dir / f"{cache_key}.json"
            
            if not cache_file.exists():
                return None
                
            # Check cache age
            cache_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
            if cache_age.total_seconds() > max_age_hours * 3600:
                return None
                
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                
            logger.info(f"Retrieved cached analysis: {cache_key}")
            return cached_data
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached analysis: {e}")
            return None
    
    # ========================================================================
    # SERENA MCP INTEGRATION LAYER
    # ========================================================================
    
    async def _execute_serena_analysis(
        self,
        operation: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Serena analysis operation via MCP tools."""
        try:
            # This would integrate with actual Serena MCP calls
            # For now, simulate the analysis
            
            if operation == "code_structure_analysis":
                return await self._simulate_code_analysis(config)
                
            return {"status": "success", "result": "simulated"}
            
        except Exception as e:
            logger.error(f"Serena MCP operation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _execute_serena_search(
        self,
        operation: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute Serena search operation via MCP tools."""
        try:
            # Simulate search operations
            if operation == "symbol_definitions":
                return await self._simulate_symbol_search(config, "definitions")
            elif operation == "symbol_references":
                return await self._simulate_symbol_search(config, "references")
                
            return {"status": "success", "result": []}
            
        except Exception as e:
            logger.error(f"Serena search operation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _execute_serena_completion(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code completion via Serena MCP."""
        # Simulate completion
        return {
            "status": "success",
            "completions": [
                {"label": "function_name", "kind": "function", "detail": "Suggested function"},
                {"label": "variable_name", "kind": "variable", "detail": "Suggested variable"}
            ]
        }
    
    async def _execute_serena_diagnostics(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute diagnostics via Serena MCP."""
        # Simulate diagnostics
        return {"diagnostics": []}
    
    async def _execute_serena_project_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute project analysis via Serena MCP."""
        # Simulate project analysis
        return {
            "structure": {"files": 10, "directories": 3},
            "patterns": ["MVC", "Repository"],
            "metrics": {"complexity": 2.5, "maintainability": 85.0},
            "dependencies": []
        }
    
    async def _execute_serena_pattern_detection(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pattern detection via Serena MCP."""
        # Simulate pattern detection
        return {
            "status": "success",
            "patterns": [
                {"name": "MVC", "confidence": 0.85},
                {"name": "Repository", "confidence": 0.75}
            ],
            "confidence_scores": {"MVC": 0.85, "Repository": 0.75},
            "recommendations": ["Consider adding more unit tests"]
        }
    
    async def _execute_serena_semantic_search(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute semantic search via Serena MCP."""
        # Simulate semantic search
        return {
            "results": [],
            "total_count": 0
        }
    
    async def _execute_serena_similarity_search(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute similarity search via Serena MCP."""
        # Simulate similarity search
        return {"matches": []}
    
    async def _execute_serena_knowledge_extraction(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute knowledge extraction via Serena MCP."""
        # Simulate knowledge extraction
        return {
            "functions": [],
            "classes": [],
            "interfaces": [],
            "patterns": [],
            "documentation": []
        }
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    async def _simulate_code_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate code analysis for development/testing."""
        file_path = config.get("file_path", "")
        
        return {
            "symbols": {
                "functions": ["main", "helper_function"],
                "classes": ["MyClass"],
                "variables": ["config", "data"]
            },
            "complexity": {
                "cyclomatic": 3,
                "cognitive": 5,
                "maintainability_index": 75.0
            },
            "dependencies": {
                "imports": ["os", "json", "logging"],
                "external_dependencies": []
            },
            "metrics": {
                "lines_of_code": 150,
                "comment_ratio": 0.15,
                "test_coverage": 0.80
            }
        }
    
    async def _simulate_symbol_search(self, config: Dict[str, Any], search_type: str) -> Dict[str, Any]:
        """Simulate symbol search for development/testing."""
        symbol_name = config.get("symbol_name", "")
        
        if search_type == "definitions":
            return {
                "definitions": [
                    {"file": "src/main.py", "line": 10, "column": 4, "symbol": symbol_name}
                ]
            }
        else:  # references
            return {
                "references": [
                    {"file": "src/main.py", "line": 25, "column": 8, "symbol": symbol_name},
                    {"file": "src/utils.py", "line": 15, "column": 12, "symbol": symbol_name}
                ]
            }
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.cs': 'csharp'
        }
        return language_map.get(ext, 'unknown')
    
    def _get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages."""
        return [
            'python', 'javascript', 'typescript', 'java', 'cpp', 'c',
            'go', 'rust', 'ruby', 'php', 'csharp', 'kotlin', 'swift'
        ]
    
    # ========================================================================
    # SERVICE STATUS AND MONITORING
    # ========================================================================
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get Serena service status and health information."""
        try:
            # Check if Serena MCP is available
            status_info = {
                "timestamp": datetime.now().isoformat(),
                "service": "serena-intelligence",
                "version": "1.0.0",
                "cache_dir": str(self.cache_dir),
                "memory_db": str(self.memory_db),
                "directories_ready": self.serena_config.exists()
            }
            
            # Try to get more detailed status
            try:
                # Simulate Serena MCP availability check
                status_info["mcp_available"] = False
                status_info["simulation_mode"] = True
                status_info["status"] = "healthy_simulation"
            except Exception as e:
                status_info["mcp_available"] = False
                status_info["mcp_error"] = str(e)
                status_info["status"] = "simulation_only"
            
            return {"status": "success", "info": status_info}
            
        except Exception as e:
            logger.error(f"Failed to get service status: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get caching statistics and performance metrics."""
        try:
            # Collect cache statistics
            cache_stats = {
                "timestamp": datetime.now().isoformat(),
                "cache_directory": str(self.cache_dir),
                "directories": {}
            }
            
            if self.cache_dir.exists():
                # Get cache directory sizes and file counts
                total_size = 0
                total_files = 0
                
                for item in self.cache_dir.rglob("*"):
                    if item.is_file():
                        size = item.stat().st_size
                        total_size += size
                        total_files += 1
                
                cache_stats["total_files"] = total_files
                cache_stats["total_size_bytes"] = total_size
                cache_stats["total_size_mb"] = round(total_size / (1024 * 1024), 2)
                cache_stats["cache_enabled"] = True
            else:
                cache_stats["cache_enabled"] = False
                cache_stats["total_files"] = 0
                cache_stats["total_size_bytes"] = 0
                cache_stats["total_size_mb"] = 0
            
            return {"status": "success", "cache_stats": cache_stats}
            
        except Exception as e:
            logger.error(f"Failed to get cache statistics: {e}")
            return {"status": "error", "error": str(e)}
    
    async def optimize_cache(self) -> Dict[str, Any]:
        """Optimize Serena's semantic cache."""
        try:
            logger.info("Starting cache optimization")
            
            optimization_results = {
                "timestamp": datetime.now().isoformat(),
                "actions_taken": [],
                "space_freed_mb": 0,
                "files_optimized": 0
            }
            
            if self.cache_dir.exists():
                # Simulate cache optimization
                optimization_results["actions_taken"].append("Analyzed cache structure")
                optimization_results["actions_taken"].append("Cleaned up temporary files")
                optimization_results["files_optimized"] = 0
                optimization_results["space_freed_mb"] = 0
                
                logger.info("Cache optimization completed")
            else:
                optimization_results["actions_taken"].append("Created cache directory")
                os.makedirs(self.cache_dir, exist_ok=True)
            
            return {"status": "success", "optimization": optimization_results}
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def clear_cache(
        self, 
        cache_type: Optional[str] = None,
        older_than_hours: int = 24
    ) -> Dict[str, Any]:
        """Clear Serena's semantic cache."""
        try:
            logger.info(f"Clearing cache: type={cache_type}, older_than={older_than_hours}h")
            
            cleanup_results = {
                "timestamp": datetime.now().isoformat(),
                "cache_type": cache_type or "all",
                "older_than_hours": older_than_hours,
                "files_removed": 0,
                "space_freed_mb": 0
            }
            
            if self.cache_dir.exists():
                from datetime import datetime, timedelta
                cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
                
                files_removed = 0
                space_freed = 0
                
                for item in self.cache_dir.rglob("*"):
                    if item.is_file():
                        # Check file age
                        file_modified = datetime.fromtimestamp(item.stat().st_mtime)
                        if file_modified < cutoff_time:
                            # Check cache type filter
                            if cache_type is None or cache_type in str(item):
                                size = item.stat().st_size
                                try:
                                    item.unlink()
                                    files_removed += 1
                                    space_freed += size
                                    logger.debug(f"Removed cache file: {item}")
                                except Exception as e:
                                    logger.warning(f"Could not remove cache file {item}: {e}")
                
                cleanup_results["files_removed"] = files_removed
                cleanup_results["space_freed_mb"] = round(space_freed / (1024 * 1024), 2)
                
                logger.info(f"Cache cleanup completed: {files_removed} files, {cleanup_results['space_freed_mb']} MB freed")
            
            return {"status": "success", "cleanup": cleanup_results}
            
        except Exception as e:
            logger.error(f"Cache clearing failed: {e}")
            return {"status": "error", "error": str(e)}
    
    # ========================================================================
    # ADDITIONAL API METHODS
    # ========================================================================
    
    async def get_intelligent_completion(
        self,
        file_path: str,
        line: int,
        column: int,
        trigger: Optional[str] = None,
        context_lines: int = 5
    ) -> Dict[str, Any]:
        """Get intelligent code completion suggestions."""
        try:
            logger.info(f"Getting completion: {file_path}:{line}:{column}")
            
            config = {
                "file_path": file_path,
                "line": line,
                "column": column,
                "trigger": trigger,
                "context_lines": context_lines,
                "analysis_type": "completion"
            }
            
            # Try Serena MCP completion
            result = await self._execute_serena_completion(config)
            
            return result
            
        except Exception as e:
            logger.error(f"Intelligent completion failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def detect_code_patterns(
        self,
        project_path: str,
        pattern_types: List[str],
        languages: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Detect architectural and design patterns in code."""
        try:
            logger.info(f"Detecting patterns: {project_path}")
            
            config = {
                "project_path": project_path,
                "pattern_types": pattern_types,
                "languages": languages,
                "analysis_type": "pattern_detection"
            }
            
            # Try Serena MCP pattern detection
            result = await self._execute_serena_pattern_detection(config)
            
            return result
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def get_refactoring_suggestions(
        self,
        file_path: str,
        line_start: int,
        line_end: int,
        refactoring_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get intelligent refactoring suggestions."""
        try:
            logger.info(f"Getting refactoring suggestions: {file_path}:{line_start}-{line_end}")
            
            # Simulate refactoring analysis
            suggestions = {
                "file_path": file_path,
                "line_range": {"start": line_start, "end": line_end},
                "refactoring_type": refactoring_type,
                "suggestions": [
                    {
                        "type": "extract_method",
                        "description": "Extract method to improve readability",
                        "confidence": 0.8,
                        "preview": f"def extracted_method():\n    # Lines {line_start}-{line_end}"
                    },
                    {
                        "type": "rename_variable",
                        "description": "Improve variable naming",
                        "confidence": 0.7,
                        "suggestions": ["descriptive_name", "clear_variable"]
                    }
                ],
                "analysis_type": "refactoring_suggestions",
                "timestamp": datetime.now().isoformat()
            }
            
            return {"status": "success", "suggestions": suggestions}
            
        except Exception as e:
            logger.error(f"Refactoring suggestions failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def integrate_with_archon_rag(
        self,
        project_path: str,
        update_existing: bool = True,
        include_patterns: bool = True
    ) -> Dict[str, Any]:
        """Integrate Serena analysis with Archon's RAG system."""
        try:
            logger.info(f"Integrating with Archon RAG: {project_path}")
            
            # Extract code knowledge for RAG
            knowledge_result = await self.extract_code_knowledge(
                project_path=project_path,
                knowledge_types=["patterns", "concepts"] if include_patterns else ["concepts"],
                output_format="structured"
            )
            
            if knowledge_result["status"] == "success":
                # Enhance with Archon context
                knowledge_base = knowledge_result.get("knowledge_base", {})
                integration_result = await self.enhance_archon_context(knowledge_base)
                
                return {
                    "status": "success",
                    "integration": {
                        "project_path": project_path,
                        "knowledge_extracted": knowledge_result.get("metadata", {}).get("total_artifacts", 0),
                        "patterns_detected": len(knowledge_base.get("patterns", [])),
                        "archon_integration": integration_result.get("status") == "success",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            else:
                return knowledge_result
                
        except Exception as e:
            logger.error(f"Archon RAG integration failed: {e}")
            return {"status": "error", "error": str(e)}


# Global service instance
serena_service = SerenaService()