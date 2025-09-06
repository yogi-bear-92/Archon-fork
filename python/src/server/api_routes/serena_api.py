"""
Serena Integration API Routes for Archon

This module provides REST API endpoints for Serena MCP integration
with Archon's code intelligence and task management systems.
"""

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
import logging

from ..config.logfire_config import get_logger
from ..services.serena_service import serena_service

logger = get_logger(__name__)

router = APIRouter(prefix="/api/serena", tags=["Serena Code Intelligence"])


# Pydantic models
class CodeAnalysisRequest(BaseModel):
    file_path: str = Field(..., description="Path to the file to analyze")
    language: Optional[str] = Field(None, description="Programming language (auto-detected if not provided)")
    include_dependencies: bool = Field(True, description="Include dependency analysis")
    analysis_depth: str = Field("comprehensive", description="Analysis depth: basic, comprehensive, deep")


class SymbolSearchRequest(BaseModel):
    symbol_name: str = Field(..., description="Symbol name to search for")
    symbol_type: Optional[str] = Field(None, description="Symbol type filter (function, class, variable)")
    language: Optional[str] = Field(None, description="Language filter")
    include_references: bool = Field(True, description="Include symbol references")


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., description="Semantic search query")
    project_path: str = Field(".", description="Project path to search within")
    max_results: int = Field(10, description="Maximum number of results")
    similarity_threshold: float = Field(0.7, description="Similarity threshold (0.0-1.0)")
    language_filter: Optional[List[str]] = Field(None, description="Filter by programming languages")


class CodeCompletionRequest(BaseModel):
    file_path: str = Field(..., description="File path for completion")
    line: int = Field(..., description="Line number (0-based)")
    column: int = Field(..., description="Column number (0-based)")
    trigger: Optional[str] = Field(None, description="Completion trigger character")
    context_lines: int = Field(5, description="Number of context lines to include")


class ProjectAnalysisRequest(BaseModel):
    project_path: str = Field(".", description="Project root path")
    include_patterns: bool = Field(True, description="Include architecture patterns")
    include_metrics: bool = Field(True, description="Include complexity metrics")
    languages: Optional[List[str]] = Field(None, description="Languages to analyze")


class KnowledgeExtractionRequest(BaseModel):
    content: str = Field(..., description="Content to extract knowledge from")
    content_type: str = Field("code", description="Content type: code, documentation, comments")
    language: Optional[str] = Field(None, description="Programming language")
    extract_patterns: bool = Field(True, description="Extract design patterns")
    extract_concepts: bool = Field(True, description="Extract key concepts")


# API Endpoints
@router.post("/analyze/code-structure")
async def analyze_code_structure(request: CodeAnalysisRequest) -> Dict[str, Any]:
    """Analyze code structure using Serena's semantic analysis."""
    try:
        logger.info(f"Analyzing code structure: {request.file_path}")
        
        result = await serena_service.analyze_code_structure(
            file_path=request.file_path,
            language=request.language,
            include_dependencies=request.include_dependencies,
            analysis_depth=request.analysis_depth
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Code structure analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/symbols/definitions")
async def get_symbol_definitions(request: SymbolSearchRequest) -> Dict[str, Any]:
    """Get symbol definitions and references."""
    try:
        logger.info(f"Getting symbol definitions: {request.symbol_name}")
        
        result = await serena_service.get_symbol_definitions(
            symbol_name=request.symbol_name,
            symbol_type=request.symbol_type,
            language=request.language,
            include_references=request.include_references
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Symbol definitions lookup failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/completion/intelligent")
async def get_intelligent_completion(request: CodeCompletionRequest) -> Dict[str, Any]:
    """Get intelligent code completion suggestions."""
    try:
        logger.info(f"Getting code completion: {request.file_path}:{request.line}:{request.column}")
        
        result = await serena_service.get_intelligent_completion(
            file_path=request.file_path,
            line=request.line,
            column=request.column,
            trigger=request.trigger,
            context_lines=request.context_lines
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Intelligent completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/semantic")
async def semantic_code_search(request: SemanticSearchRequest) -> Dict[str, Any]:
    """Perform semantic code search."""
    try:
        logger.info(f"Semantic search: {request.query}")
        
        result = await serena_service.semantic_code_search(
            query=request.query,
            project_path=request.project_path,
            max_results=request.max_results,
            similarity_threshold=request.similarity_threshold,
            language_filter=request.language_filter
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Semantic code search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/project")
async def analyze_project_structure(request: ProjectAnalysisRequest) -> Dict[str, Any]:
    """Analyze project structure and detect patterns."""
    try:
        logger.info(f"Analyzing project structure: {request.project_path}")
        
        result = await serena_service.analyze_project_structure(
            project_path=request.project_path,
            include_patterns=request.include_patterns,
            include_metrics=request.include_metrics,
            languages=request.languages
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Project structure analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/patterns/detect")
async def detect_code_patterns(request: ProjectAnalysisRequest) -> Dict[str, Any]:
    """Detect architectural and design patterns in code."""
    try:
        logger.info(f"Detecting code patterns: {request.project_path}")
        
        result = await serena_service.detect_code_patterns(
            project_path=request.project_path,
            pattern_types=["architectural", "design", "anti-pattern"],
            languages=request.languages
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Pattern detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/similarity/compare")
async def find_similar_code(
    file_path: str,
    target_function: Optional[str] = None,
    similarity_threshold: float = 0.8,
    max_results: int = 10
) -> Dict[str, Any]:
    """Find similar code patterns across the codebase."""
    try:
        logger.info(f"Finding similar code: {file_path}")
        
        result = await serena_service.find_similar_code(
            file_path=file_path,
            target_function=target_function,
            similarity_threshold=similarity_threshold,
            max_results=max_results
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Similar code search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/knowledge/extract")
async def extract_code_knowledge(request: KnowledgeExtractionRequest) -> Dict[str, Any]:
    """Extract knowledge from code for Archon's RAG system."""
    try:
        logger.info(f"Extracting knowledge from {request.content_type}")
        
        result = await serena_service.extract_code_knowledge(
            content=request.content,
            content_type=request.content_type,
            language=request.language,
            extract_patterns=request.extract_patterns,
            extract_concepts=request.extract_concepts
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Knowledge extraction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/refactoring/suggestions")
async def get_refactoring_suggestions(
    file_path: str,
    line_start: int,
    line_end: int,
    refactoring_type: Optional[str] = None
) -> Dict[str, Any]:
    """Get intelligent refactoring suggestions."""
    try:
        logger.info(f"Getting refactoring suggestions: {file_path}:{line_start}-{line_end}")
        
        result = await serena_service.get_refactoring_suggestions(
            file_path=file_path,
            line_start=line_start,
            line_end=line_end,
            refactoring_type=refactoring_type
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Refactoring suggestions failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_serena_status() -> Dict[str, Any]:
    """Get Serena service status and metrics."""
    try:
        result = await serena_service.get_service_status()
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Failed to get Serena status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cache/stats")
async def get_cache_statistics() -> Dict[str, Any]:
    """Get Serena caching statistics."""
    try:
        result = await serena_service.get_cache_statistics()
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Failed to get cache statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cache/optimize")
async def optimize_cache() -> Dict[str, Any]:
    """Optimize Serena's semantic cache."""
    try:
        logger.info("Optimizing Serena cache")
        
        result = await serena_service.optimize_cache()
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Cache optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cache/clear")
async def clear_cache(
    cache_type: Optional[str] = None,
    older_than_hours: int = 24
) -> Dict[str, Any]:
    """Clear Serena's semantic cache."""
    try:
        logger.info(f"Clearing Serena cache: type={cache_type}, older_than={older_than_hours}h")
        
        result = await serena_service.clear_cache(
            cache_type=cache_type,
            older_than_hours=older_than_hours
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Cache clearing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/languages/supported")
async def get_supported_languages() -> Dict[str, List[str]]:
    """Get supported programming languages and their features."""
    return {
        "primary_support": [
            "python", "javascript", "typescript", "java", "go", 
            "rust", "cpp", "c", "csharp", "ruby"
        ],
        "secondary_support": [
            "php", "kotlin", "scala", "swift", "dart", "lua",
            "r", "julia", "haskell", "elixir", "clojure"
        ],
        "markup_languages": [
            "html", "css", "scss", "sass", "xml", "yaml", "json",
            "markdown", "toml", "ini"
        ],
        "query_languages": [
            "sql", "graphql", "sparql"
        ],
        "shell_scripts": [
            "bash", "zsh", "fish", "powershell", "cmd"
        ]
    }


@router.get("/patterns/catalog")
async def get_pattern_catalog() -> Dict[str, Any]:
    """Get catalog of detectable code patterns."""
    return {
        "architectural_patterns": [
            "mvc", "mvp", "mvvm", "layered", "microservices",
            "event_driven", "pipe_filter", "client_server"
        ],
        "design_patterns": [
            "singleton", "factory", "observer", "strategy", "decorator",
            "adapter", "facade", "proxy", "command", "chain_of_responsibility"
        ],
        "anti_patterns": [
            "god_object", "spaghetti_code", "magic_numbers", "dead_code",
            "copy_paste_programming", "golden_hammer", "lava_flow"
        ],
        "code_smells": [
            "long_method", "large_class", "duplicate_code", "complex_conditional",
            "primitive_obsession", "data_clumps", "inappropriate_intimacy"
        ]
    }


@router.post("/integration/archon-rag")
async def integrate_with_archon_rag(
    project_path: str = ".",
    update_existing: bool = True,
    include_patterns: bool = True
) -> Dict[str, Any]:
    """Integrate Serena analysis with Archon's RAG system."""
    try:
        logger.info(f"Integrating with Archon RAG: {project_path}")
        
        result = await serena_service.integrate_with_archon_rag(
            project_path=project_path,
            update_existing=update_existing,
            include_patterns=include_patterns
        )
        
        if result["status"] == "error":
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
        
    except Exception as e:
        logger.error(f"Archon RAG integration failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for Serena service."""
    try:
        # Basic health check
        status = await serena_service.get_service_status()
        return {
            "status": "healthy",
            "service": "serena-intelligence",
            "timestamp": status.get("timestamp", "unknown"),
            "cache_status": status.get("cache", {}).get("status", "unknown")
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")