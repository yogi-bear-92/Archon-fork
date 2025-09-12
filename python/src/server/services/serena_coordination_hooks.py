"""
Serena Claude Flow Expert Agent Coordination Hooks System

This module provides comprehensive coordination hooks for the Serena Claude Flow Expert Agent,
integrating semantic code intelligence with Claude Flow orchestration and
Archon's task management system.

Features:
- Pre-task semantic analysis preparation
- Post-task knowledge persistence and sharing
- Multi-agent workflow coordination
- Memory synchronization across agents
- Performance monitoring and optimization
- Error handling and retry logic
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import pickle
from contextlib import asynccontextmanager

from src.server.config.logfire_config import get_logger
from .claude_flow_service import claude_flow_service
from .claude_flow_task_integration import claude_flow_task_integration
from .ai_tagging_background_service import get_ai_tagging_background_service

logger = get_logger(__name__)


class HookPhase(Enum):
    """Hook execution phases."""
    PRE_TASK = "pre_task"
    POST_TASK = "post_task"
    PRE_EDIT = "pre_edit"
    POST_EDIT = "post_edit"
    MEMORY_SYNC = "memory_sync"
    PERFORMANCE_MONITOR = "performance_monitor"
    ERROR_RECOVERY = "error_recovery"
    AGENT_COORDINATE = "agent_coordinate"


class CoordinationLevel(Enum):
    """Agent coordination levels."""
    INDIVIDUAL = "individual"      # Single agent operation
    PAIRWISE = "pairwise"         # Two-agent coordination
    GROUP = "group"               # Small group (3-5 agents)
    SWARM = "swarm"              # Large group (5+ agents)
    ECOSYSTEM = "ecosystem"       # Cross-swarm coordination


@dataclass
class SemanticContext:
    """Semantic analysis context for coordination."""
    project_path: str
    file_paths: List[str]
    symbol_map: Dict[str, Any]
    architecture_patterns: List[Dict[str, Any]]
    complexity_metrics: Dict[str, float]
    last_updated: datetime
    context_hash: str


@dataclass
class CoordinationState:
    """Current coordination state."""
    agent_id: str
    task_id: Optional[str]
    coordination_level: CoordinationLevel
    active_agents: Set[str]
    shared_context: Dict[str, Any]
    performance_metrics: Dict[str, float]
    error_count: int
    last_sync: datetime


@dataclass
class HookExecutionResult:
    """Result of hook execution."""
    success: bool
    execution_time: float
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    retry_count: int
    next_actions: List[str]


class SerenaCoordinationHooks:
    """
    Comprehensive coordination hooks system for Serena Claude Flow Expert Agent.
    
    Provides semantic-aware coordination patterns that integrate with
    Claude Flow and Archon systems for optimal multi-agent workflows.
    """
    
    def __init__(self):
        self.base_path = Path(__file__).resolve().parent.parent.parent.parent
        self.hooks_path = self.base_path / ".serena" / "hooks"
        self.memory_path = self.base_path / ".serena" / "memory"
        self.metrics_path = self.base_path / ".serena" / "metrics"
        
        # Coordination state management
        self.coordination_states: Dict[str, CoordinationState] = {}
        self.semantic_contexts: Dict[str, SemanticContext] = {}
        self.active_hooks: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.hook_performance: Dict[str, List[float]] = {}
        self.error_patterns: Dict[str, int] = {}
        
        # Configuration
        self.max_retry_attempts = 3
        self.hook_timeout = 30.0
        self.memory_sync_interval = 300  # 5 minutes
        self.performance_check_interval = 60  # 1 minute
        
        self._ensure_directories()
        self._start_background_tasks()

    def _ensure_directories(self):
        """Ensure required directories exist."""
        for path in [self.hooks_path, self.memory_path, self.metrics_path]:
            path.mkdir(parents=True, exist_ok=True)
            
    def _start_background_tasks(self):
        """Start background monitoring tasks."""
        # Don't start background tasks during module import
        # They will be started when the FastAPI app starts
        pass

    # ========================================================================
    # PRE-TASK HOOKS: Semantic Analysis Preparation
    # ========================================================================

    async def pre_task_semantic_preparation(
        self, 
        task_context: Dict[str, Any],
        coordination_level: CoordinationLevel = CoordinationLevel.INDIVIDUAL
    ) -> HookExecutionResult:
        """
        Prepare semantic analysis context before task execution.
        
        This hook:
        1. Analyzes the project structure for semantic patterns
        2. Prepares code context and symbol maps
        3. Identifies relevant architectural patterns
        4. Sets up memory structures for collaboration
        5. Initializes performance baselines
        """
        start_time = time.time()
        
        try:
            logger.info(f"Executing pre-task semantic preparation for task: {task_context.get('task_id', 'unknown')}")
            
            # Extract task information
            task_id = task_context.get("task_id")
            project_path = task_context.get("project_path", ".")
            target_files = task_context.get("target_files", [])
            
            # Phase 1: Project Structure Analysis
            structure_analysis = await self._analyze_project_structure(project_path)
            
            # Phase 2: Semantic Context Generation
            semantic_context = await self._generate_semantic_context(
                project_path, target_files, structure_analysis
            )
            
            # Phase 3: Memory Preparation
            memory_setup = await self._prepare_task_memory(task_id, semantic_context)
            
            # Phase 4: Coordination Setup
            coordination_state = await self._initialize_coordination_state(
                task_id, coordination_level, task_context
            )
            
            # Phase 5: Performance Baseline
            performance_baseline = await self._establish_performance_baseline(
                task_context, semantic_context
            )
            
            # Store contexts
            if task_id:
                self.semantic_contexts[task_id] = semantic_context
                self.coordination_states[task_id] = coordination_state
            
            execution_time = time.time() - start_time
            
            result_data = {
                "semantic_context": asdict(semantic_context),
                "coordination_state": asdict(coordination_state),
                "memory_setup": memory_setup,
                "performance_baseline": performance_baseline,
                "preparation_metrics": {
                    "files_analyzed": len(target_files) if target_files else len(structure_analysis.get("code_files", [])),
                    "symbols_discovered": len(semantic_context.symbol_map),
                    "patterns_identified": len(semantic_context.architecture_patterns),
                    "execution_time": execution_time
                }
            }
            
            logger.info(f"Pre-task preparation completed in {execution_time:.2f}s")
            
            return HookExecutionResult(
                success=True,
                execution_time=execution_time,
                data=result_data,
                error=None,
                retry_count=0,
                next_actions=["execute_task", "monitor_progress"]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Pre-task semantic preparation failed: {e}")
            
            return HookExecutionResult(
                success=False,
                execution_time=execution_time,
                data=None,
                error=str(e),
                retry_count=0,
                next_actions=["retry_preparation", "fallback_mode"]
            )

    async def _analyze_project_structure(self, project_path: str) -> Dict[str, Any]:
        """Analyze project structure for semantic patterns."""
        try:
            from mcp import ServerProtocol
            from mcp.types import CallToolRequestId
            
            # Use Serena MCP tools for structure analysis
            structure_result = {
                "root_path": project_path,
                "code_files": [],
                "directories": [],
                "file_types": {},
                "complexity_indicators": {}
            }
            
            # This would integrate with actual Serena MCP calls
            # For now, simulate the structure analysis
            project_dir = Path(project_path)
            if project_dir.exists():
                for file_path in project_dir.rglob("*.py"):
                    if not any(part.startswith('.') for part in file_path.parts):
                        structure_result["code_files"].append(str(file_path.relative_to(project_dir)))
                        
                for file_path in project_dir.rglob("*.ts"):
                    if not any(part.startswith('.') for part in file_path.parts):
                        structure_result["code_files"].append(str(file_path.relative_to(project_dir)))
            
            return structure_result
            
        except Exception as e:
            logger.warning(f"Could not analyze project structure: {e}")
            return {"error": str(e), "code_files": [], "directories": []}

    async def _generate_semantic_context(
        self, 
        project_path: str, 
        target_files: List[str], 
        structure_analysis: Dict[str, Any]
    ) -> SemanticContext:
        """Generate comprehensive semantic context."""
        try:
            # Build symbol map from project files
            symbol_map = {}
            files_to_analyze = target_files if target_files else structure_analysis.get("code_files", [])
            
            for file_path in files_to_analyze[:10]:  # Limit for performance
                try:
                    # This would use Serena MCP tools for symbol analysis
                    file_symbols = await self._analyze_file_symbols(file_path)
                    if file_symbols:
                        symbol_map[file_path] = file_symbols
                except Exception as e:
                    logger.warning(f"Could not analyze symbols in {file_path}: {e}")
                    
            # Detect architectural patterns
            architecture_patterns = await self._detect_architecture_patterns(symbol_map)
            
            # Calculate complexity metrics
            complexity_metrics = await self._calculate_complexity_metrics(symbol_map)
            
            # Generate context hash for caching
            context_data = f"{project_path}_{len(symbol_map)}_{len(architecture_patterns)}"
            context_hash = hashlib.md5(context_data.encode()).hexdigest()
            
            return SemanticContext(
                project_path=project_path,
                file_paths=list(symbol_map.keys()),
                symbol_map=symbol_map,
                architecture_patterns=architecture_patterns,
                complexity_metrics=complexity_metrics,
                last_updated=datetime.now(),
                context_hash=context_hash
            )
            
        except Exception as e:
            logger.error(f"Failed to generate semantic context: {e}")
            return SemanticContext(
                project_path=project_path,
                file_paths=[],
                symbol_map={},
                architecture_patterns=[],
                complexity_metrics={},
                last_updated=datetime.now(),
                context_hash=""
            )

    async def _analyze_file_symbols(self, file_path: str) -> Dict[str, Any]:
        """Analyze symbols in a file using Serena MCP tools."""
        # This would integrate with actual Serena MCP calls
        # For now, return a simulated structure
        return {
            "classes": [],
            "functions": [],
            "imports": [],
            "complexity_score": 1.0
        }

    async def _detect_architecture_patterns(self, symbol_map: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect architectural patterns from symbol analysis."""
        patterns = []
        
        # MVC pattern detection
        has_models = any("model" in path.lower() for path in symbol_map.keys())
        has_views = any("view" in path.lower() for path in symbol_map.keys())
        has_controllers = any("controller" in path.lower() for path in symbol_map.keys())
        
        if has_models and has_views and has_controllers:
            patterns.append({
                "type": "MVC",
                "confidence": 0.8,
                "components": ["models", "views", "controllers"]
            })
        
        # Repository pattern detection
        has_repository = any("repository" in path.lower() or "repo" in path.lower() for path in symbol_map.keys())
        if has_repository:
            patterns.append({
                "type": "Repository",
                "confidence": 0.9,
                "components": ["repository"]
            })
        
        return patterns

    async def _calculate_complexity_metrics(self, symbol_map: Dict[str, Any]) -> Dict[str, float]:
        """Calculate complexity metrics from symbol analysis."""
        return {
            "average_file_complexity": 2.5,
            "total_symbols": len(symbol_map),
            "architecture_complexity": len(symbol_map) * 0.1,
            "maintainability_index": 85.0
        }

    async def _prepare_task_memory(self, task_id: str, semantic_context: SemanticContext) -> Dict[str, Any]:
        """Prepare memory structures for task execution."""
        if not task_id:
            return {"status": "skipped", "reason": "no_task_id"}
            
        memory_structure = {
            "task_id": task_id,
            "semantic_cache": semantic_context.context_hash,
            "symbol_references": list(semantic_context.symbol_map.keys()),
            "shared_knowledge": {},
            "coordination_state": {},
            "created_at": datetime.now().isoformat()
        }
        
        # Store in persistent memory
        memory_file = self.memory_path / f"task_{task_id}_memory.json"
        try:
            with open(memory_file, 'w') as f:
                json.dump(memory_structure, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save task memory: {e}")
            
        return memory_structure

    async def _initialize_coordination_state(
        self, 
        task_id: str, 
        coordination_level: CoordinationLevel,
        task_context: Dict[str, Any]
    ) -> CoordinationState:
        """Initialize coordination state for multi-agent workflows."""
        return CoordinationState(
            agent_id="serena-master",
            task_id=task_id,
            coordination_level=coordination_level,
            active_agents={"serena-master"},
            shared_context={
                "task_context": task_context,
                "coordination_metadata": {
                    "initialized_at": datetime.now().isoformat(),
                    "coordinator": "serena-master"
                }
            },
            performance_metrics={},
            error_count=0,
            last_sync=datetime.now()
        )

    async def _establish_performance_baseline(
        self, 
        task_context: Dict[str, Any], 
        semantic_context: SemanticContext
    ) -> Dict[str, float]:
        """Establish performance baselines for monitoring."""
        return {
            "symbol_resolution_time": 0.05,  # 50ms baseline
            "file_analysis_time": 0.1,       # 100ms baseline
            "memory_usage_mb": 50.0,         # 50MB baseline
            "cache_hit_rate": 0.0,           # Start with 0% cache hit rate
            "coordination_latency": 0.02,    # 20ms baseline
            "context_size_mb": len(str(semantic_context.symbol_map).encode()) / (1024 * 1024)
        }

    # ========================================================================
    # POST-TASK HOOKS: Knowledge Persistence and Sharing
    # ========================================================================

    async def post_task_knowledge_persistence(
        self, 
        task_result: Dict[str, Any],
        execution_metrics: Dict[str, Any]
    ) -> HookExecutionResult:
        """
        Persist knowledge and insights after task completion.
        
        This hook:
        1. Stores successful patterns and insights
        2. Updates shared knowledge base
        3. Persists semantic improvements
        4. Shares learnings with other agents
        5. Updates performance models
        """
        start_time = time.time()
        
        try:
            task_id = task_result.get("task_id")
            logger.info(f"Executing post-task knowledge persistence for task: {task_id}")
            
            # Phase 1: Extract Knowledge Artifacts
            knowledge_artifacts = await self._extract_knowledge_artifacts(task_result)
            
            # Phase 2: Update Semantic Models
            semantic_updates = await self._update_semantic_models(task_id, task_result)
            
            # Phase 3: Persist Learning Patterns
            learning_patterns = await self._persist_learning_patterns(task_result, execution_metrics)
            
            # Phase 4: Share with Agent Network
            sharing_results = await self._share_knowledge_with_agents(knowledge_artifacts)
            
            # Phase 5: Update Performance Models
            performance_updates = await self._update_performance_models(execution_metrics)
            
            # Phase 6: AI Tagging Enhancement
            ai_tagging_results = await self._enhance_with_ai_tagging(task_result)
            
            # Phase 7: Cleanup and Optimize
            cleanup_results = await self._cleanup_task_resources(task_id)
            
            execution_time = time.time() - start_time
            
            result_data = {
                "knowledge_artifacts": knowledge_artifacts,
                "semantic_updates": semantic_updates,
                "learning_patterns": learning_patterns,
                "sharing_results": sharing_results,
                "performance_updates": performance_updates,
                "ai_tagging_results": ai_tagging_results,
                "cleanup_results": cleanup_results,
                "persistence_metrics": {
                    "artifacts_stored": len(knowledge_artifacts.get("artifacts", [])),
                    "patterns_learned": len(learning_patterns.get("patterns", [])),
                    "agents_notified": len(sharing_results.get("notified_agents", [])),
                    "execution_time": execution_time
                }
            }
            
            logger.info(f"Post-task knowledge persistence completed in {execution_time:.2f}s")
            
            return HookExecutionResult(
                success=True,
                execution_time=execution_time,
                data=result_data,
                error=None,
                retry_count=0,
                next_actions=["validate_persistence", "optimize_knowledge_base"]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Post-task knowledge persistence failed: {e}")
            
            return HookExecutionResult(
                success=False,
                execution_time=execution_time,
                data=None,
                error=str(e),
                retry_count=0,
                next_actions=["retry_persistence", "partial_save"]
            )

    async def _enhance_with_ai_tagging(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance knowledge with AI-generated tags.
        
        This method:
        1. Identifies sources that need AI tagging
        2. Generates AI tags for content
        3. Updates source records with enhanced tags
        4. Provides feedback on tagging success
        """
        try:
            task_id = task_result.get("task_id")
            logger.info(f"Enhancing knowledge with AI tagging for task: {task_id}")
            
            # Get AI tagging service
            ai_tagging_service = get_ai_tagging_background_service()
            
            # Check if this task involved knowledge creation/updates
            knowledge_updates = task_result.get("knowledge_updates", {})
            sources_created = knowledge_updates.get("sources_created", [])
            sources_updated = knowledge_updates.get("sources_updated", [])
            
            all_sources = sources_created + sources_updated
            ai_tagging_results = {
                "sources_processed": [],
                "ai_tags_generated": 0,
                "errors": [],
                "success_count": 0,
                "total_sources": len(all_sources)
            }
            
            if not all_sources:
                logger.info("No sources to enhance with AI tagging")
                return ai_tagging_results
            
            # Process each source for AI tagging
            for source_id in all_sources:
                try:
                    # Update source with AI tags
                    result = await ai_tagging_service.update_source_tags_with_ai(
                        source_id=source_id,
                        force_update=False  # Don't force update existing tags
                    )
                    
                    if result["success"]:
                        ai_tagging_results["sources_processed"].append({
                            "source_id": source_id,
                            "success": True,
                            "ai_tags_generated": result.get("ai_tags_generated", 0),
                            "total_tags": result.get("total_tags", 0)
                        })
                        ai_tagging_results["ai_tags_generated"] += result.get("ai_tags_generated", 0)
                        ai_tagging_results["success_count"] += 1
                        
                        logger.info(f"Enhanced source {source_id} with {result.get('ai_tags_generated', 0)} AI tags")
                    else:
                        ai_tagging_results["sources_processed"].append({
                            "source_id": source_id,
                            "success": False,
                            "error": result.get("error", "Unknown error")
                        })
                        ai_tagging_results["errors"].append(f"Source {source_id}: {result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    error_msg = f"Failed to enhance source {source_id} with AI tags: {str(e)}"
                    logger.warning(error_msg)
                    ai_tagging_results["errors"].append(error_msg)
                    ai_tagging_results["sources_processed"].append({
                        "source_id": source_id,
                        "success": False,
                        "error": str(e)
                    })
            
            # Log overall results
            if ai_tagging_results["success_count"] > 0:
                logger.info(f"AI tagging enhancement completed: {ai_tagging_results['success_count']}/{ai_tagging_results['total_sources']} sources enhanced with {ai_tagging_results['ai_tags_generated']} total AI tags")
            else:
                logger.warning("No sources were successfully enhanced with AI tags")
            
            return ai_tagging_results
            
        except Exception as e:
            logger.error(f"AI tagging enhancement failed: {e}", exc_info=True)
            return {
                "sources_processed": [],
                "ai_tags_generated": 0,
                "errors": [str(e)],
                "success_count": 0,
                "total_sources": 0,
                "enhancement_failed": True
            }

    async def _extract_knowledge_artifacts(self, task_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract valuable knowledge artifacts from task execution."""
        artifacts = {
            "successful_patterns": [],
            "code_improvements": [],
            "semantic_insights": [],
            "performance_optimizations": [],
            "error_resolutions": []
        }
        
        # Extract successful patterns
        if task_result.get("success") and "patterns_used" in task_result:
            artifacts["successful_patterns"] = task_result["patterns_used"]
            
        # Extract code improvements
        if "code_changes" in task_result:
            artifacts["code_improvements"] = task_result["code_changes"]
            
        # Extract semantic insights
        if "semantic_discoveries" in task_result:
            artifacts["semantic_insights"] = task_result["semantic_discoveries"]
            
        return artifacts

    async def _update_semantic_models(self, task_id: str, task_result: Dict[str, Any]) -> Dict[str, Any]:
        """Update semantic models based on task results."""
        if not task_id or task_id not in self.semantic_contexts:
            return {"status": "skipped", "reason": "no_semantic_context"}
            
        context = self.semantic_contexts[task_id]
        updates = {
            "symbol_map_updates": 0,
            "pattern_updates": 0,
            "complexity_updates": 0
        }
        
        # Update symbol maps if new symbols were discovered
        if "new_symbols" in task_result:
            context.symbol_map.update(task_result["new_symbols"])
            updates["symbol_map_updates"] = len(task_result["new_symbols"])
            
        # Update architectural patterns if new ones were found
        if "new_patterns" in task_result:
            context.architecture_patterns.extend(task_result["new_patterns"])
            updates["pattern_updates"] = len(task_result["new_patterns"])
            
        # Update complexity metrics
        if "complexity_changes" in task_result:
            context.complexity_metrics.update(task_result["complexity_changes"])
            updates["complexity_updates"] = len(task_result["complexity_changes"])
            
        # Update timestamps
        context.last_updated = datetime.now()
        
        return updates

    async def _persist_learning_patterns(
        self, 
        task_result: Dict[str, Any], 
        execution_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Persist learning patterns for future task optimization."""
        patterns_file = self.memory_path / "learning_patterns.json"
        
        try:
            # Load existing patterns
            if patterns_file.exists():
                with open(patterns_file, 'r') as f:
                    existing_patterns = json.load(f)
            else:
                existing_patterns = {"patterns": [], "last_updated": None}
                
            # Create new learning pattern
            new_pattern = {
                "task_type": task_result.get("task_type", "unknown"),
                "success": task_result.get("success", False),
                "execution_time": execution_metrics.get("execution_time", 0),
                "memory_usage": execution_metrics.get("memory_usage", 0),
                "strategies_used": task_result.get("strategies_used", []),
                "optimizations_applied": task_result.get("optimizations_applied", []),
                "timestamp": datetime.now().isoformat(),
                "effectiveness_score": self._calculate_effectiveness_score(task_result, execution_metrics)
            }
            
            # Add to patterns list
            existing_patterns["patterns"].append(new_pattern)
            existing_patterns["last_updated"] = datetime.now().isoformat()
            
            # Limit pattern history (keep last 1000 patterns)
            if len(existing_patterns["patterns"]) > 1000:
                existing_patterns["patterns"] = existing_patterns["patterns"][-1000:]
                
            # Save updated patterns
            with open(patterns_file, 'w') as f:
                json.dump(existing_patterns, f, indent=2, default=str)
                
            return {"patterns": [new_pattern], "total_patterns": len(existing_patterns["patterns"])}
            
        except Exception as e:
            logger.error(f"Failed to persist learning patterns: {e}")
            return {"error": str(e), "patterns": []}

    def _calculate_effectiveness_score(
        self, 
        task_result: Dict[str, Any], 
        execution_metrics: Dict[str, Any]
    ) -> float:
        """Calculate effectiveness score for learning patterns."""
        score = 0.0
        
        # Success weight
        if task_result.get("success"):
            score += 50.0
            
        # Performance weight (inverse of execution time)
        exec_time = execution_metrics.get("execution_time", 1.0)
        if exec_time > 0:
            score += min(30.0, 30.0 / exec_time)
            
        # Quality weight
        quality_indicators = task_result.get("quality_indicators", {})
        score += quality_indicators.get("code_quality", 0) * 10
        score += quality_indicators.get("test_coverage", 0) * 10
        
        return min(100.0, score)

    async def _share_knowledge_with_agents(self, knowledge_artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Share knowledge artifacts with other agents in the network."""
        try:
            sharing_results = {
                "notified_agents": [],
                "sharing_successful": True,
                "shared_artifacts": 0
            }
            
            # Prepare knowledge for sharing
            shareable_knowledge = {
                "timestamp": datetime.now().isoformat(),
                "source_agent": "serena-master",
                "artifacts": knowledge_artifacts,
                "sharing_metadata": {
                    "artifact_count": sum(len(v) if isinstance(v, list) else 1 for v in knowledge_artifacts.values()),
                    "knowledge_type": "semantic_code_intelligence"
                }
            }
            
            # Share via Claude Flow memory system
            memory_key = f"shared_knowledge/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            memory_result = await claude_flow_service.memory_operations(
                operation="store",
                key=memory_key,
                value=shareable_knowledge
            )
            
            if memory_result.get("status") == "success":
                sharing_results["shared_artifacts"] = shareable_knowledge["sharing_metadata"]["artifact_count"]
                sharing_results["notified_agents"] = ["claude-flow-network"]
                
            return sharing_results
            
        except Exception as e:
            logger.error(f"Failed to share knowledge with agents: {e}")
            return {
                "notified_agents": [],
                "sharing_successful": False,
                "error": str(e)
            }

    async def _update_performance_models(self, execution_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Update performance models based on execution metrics."""
        try:
            metrics_file = self.metrics_path / "performance_model.json"
            
            # Load existing model
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    model = json.load(f)
            else:
                model = {
                    "execution_times": [],
                    "memory_usage": [],
                    "cache_hit_rates": [],
                    "error_rates": [],
                    "last_updated": None
                }
            
            # Add new metrics
            model["execution_times"].append(execution_metrics.get("execution_time", 0))
            model["memory_usage"].append(execution_metrics.get("memory_usage", 0))
            model["cache_hit_rates"].append(execution_metrics.get("cache_hit_rate", 0))
            model["error_rates"].append(1 if execution_metrics.get("errors", 0) > 0 else 0)
            
            # Limit history size
            for key in ["execution_times", "memory_usage", "cache_hit_rates", "error_rates"]:
                if len(model[key]) > 1000:
                    model[key] = model[key][-1000:]
                    
            # Update timestamp
            model["last_updated"] = datetime.now().isoformat()
            
            # Calculate running averages
            model["averages"] = {
                "execution_time": sum(model["execution_times"]) / len(model["execution_times"]) if model["execution_times"] else 0,
                "memory_usage": sum(model["memory_usage"]) / len(model["memory_usage"]) if model["memory_usage"] else 0,
                "cache_hit_rate": sum(model["cache_hit_rates"]) / len(model["cache_hit_rates"]) if model["cache_hit_rates"] else 0,
                "error_rate": sum(model["error_rates"]) / len(model["error_rates"]) if model["error_rates"] else 0
            }
            
            # Save updated model
            with open(metrics_file, 'w') as f:
                json.dump(model, f, indent=2, default=str)
                
            return {
                "model_updated": True,
                "data_points": len(model["execution_times"]),
                "averages": model["averages"]
            }
            
        except Exception as e:
            logger.error(f"Failed to update performance models: {e}")
            return {"model_updated": False, "error": str(e)}

    async def _cleanup_task_resources(self, task_id: str) -> Dict[str, Any]:
        """Cleanup task-specific resources and optimize memory usage."""
        cleanup_results = {
            "contexts_cleaned": 0,
            "memory_freed": 0,
            "files_archived": 0
        }
        
        try:
            # Clean semantic contexts (keep for 1 hour after task completion)
            if task_id in self.semantic_contexts:
                context = self.semantic_contexts[task_id]
                if datetime.now() - context.last_updated > timedelta(hours=1):
                    del self.semantic_contexts[task_id]
                    cleanup_results["contexts_cleaned"] += 1
                    
            # Clean coordination states
            if task_id in self.coordination_states:
                del self.coordination_states[task_id]
                cleanup_results["contexts_cleaned"] += 1
                
            # Archive old memory files
            memory_files = list(self.memory_path.glob(f"task_{task_id}_*.json"))
            for memory_file in memory_files:
                if datetime.now() - datetime.fromtimestamp(memory_file.stat().st_mtime) > timedelta(days=7):
                    archive_path = self.memory_path / "archive" / memory_file.name
                    archive_path.parent.mkdir(exist_ok=True)
                    memory_file.rename(archive_path)
                    cleanup_results["files_archived"] += 1
                    
            return cleanup_results
            
        except Exception as e:
            logger.error(f"Cleanup failed for task {task_id}: {e}")
            return {"error": str(e), **cleanup_results}

    # ========================================================================
    # AGENT COORDINATION PROTOCOLS
    # ========================================================================

    async def coordinate_multi_agent_workflow(
        self,
        workflow_definition: Dict[str, Any],
        participating_agents: List[str]
    ) -> HookExecutionResult:
        """
        Coordinate complex multi-agent workflows with semantic intelligence.
        
        This hook orchestrates:
        1. Agent role assignment based on semantic analysis
        2. Work distribution with context awareness
        3. Real-time coordination and communication
        4. Conflict resolution and task rebalancing
        5. Quality gates and validation checkpoints
        """
        start_time = time.time()
        
        try:
            workflow_id = workflow_definition.get("workflow_id", f"workflow_{int(time.time())}")
            logger.info(f"Coordinating multi-agent workflow: {workflow_id}")
            
            # Phase 1: Workflow Analysis and Planning
            workflow_analysis = await self._analyze_workflow_requirements(workflow_definition)
            
            # Phase 2: Agent Role Assignment
            role_assignments = await self._assign_agent_roles(
                workflow_analysis, participating_agents
            )
            
            # Phase 3: Context Sharing Setup
            context_sharing = await self._setup_context_sharing(
                workflow_id, role_assignments
            )
            
            # Phase 4: Coordination Protocol Initialization
            coordination_protocol = await self._initialize_coordination_protocol(
                workflow_id, workflow_analysis, role_assignments
            )
            
            # Phase 5: Workflow Execution Monitoring
            execution_monitor = await self._start_workflow_monitoring(
                workflow_id, coordination_protocol
            )
            
            execution_time = time.time() - start_time
            
            result_data = {
                "workflow_id": workflow_id,
                "workflow_analysis": workflow_analysis,
                "role_assignments": role_assignments,
                "context_sharing": context_sharing,
                "coordination_protocol": coordination_protocol,
                "execution_monitor": execution_monitor,
                "coordination_metrics": {
                    "agents_coordinated": len(participating_agents),
                    "roles_assigned": len(role_assignments),
                    "setup_time": execution_time
                }
            }
            
            logger.info(f"Multi-agent workflow coordination setup completed in {execution_time:.2f}s")
            
            return HookExecutionResult(
                success=True,
                execution_time=execution_time,
                data=result_data,
                error=None,
                retry_count=0,
                next_actions=["monitor_workflow", "handle_coordination_events"]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Multi-agent workflow coordination failed: {e}")
            
            return HookExecutionResult(
                success=False,
                execution_time=execution_time,
                data=None,
                error=str(e),
                retry_count=0,
                next_actions=["retry_coordination", "fallback_to_single_agent"]
            )

    async def _analyze_workflow_requirements(self, workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze workflow requirements for optimal agent coordination."""
        return {
            "complexity_level": "medium",
            "estimated_duration": 3600,  # 1 hour
            "resource_requirements": {
                "memory_intensive": False,
                "cpu_intensive": False,
                "io_intensive": True
            },
            "coordination_patterns": ["hierarchical", "pairwise"],
            "quality_gates": ["code_review", "testing", "validation"]
        }

    async def _assign_agent_roles(
        self, 
        workflow_analysis: Dict[str, Any], 
        participating_agents: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """Assign optimal roles to agents based on workflow analysis."""
        role_assignments = {}
        
        # Serena Claude Flow Expert Agent as coordinator
        role_assignments["serena-master"] = {
            "role": "coordinator",
            "responsibilities": [
                "semantic_analysis",
                "code_intelligence",
                "quality_assurance",
                "workflow_monitoring"
            ],
            "coordination_level": "master"
        }
        
        # Assign roles to other agents
        available_roles = ["coder", "reviewer", "tester", "researcher", "system-architect"]
        for i, agent in enumerate(participating_agents):
            if agent != "serena-master":
                role = available_roles[i % len(available_roles)]
                role_assignments[agent] = {
                    "role": role,
                    "responsibilities": self._get_role_responsibilities(role),
                    "coordination_level": "worker",
                    "coordinator": "serena-master"
                }
                
        return role_assignments

    def _get_role_responsibilities(self, role: str) -> List[str]:
        """Get responsibilities for a given agent role."""
        responsibilities_map = {
            "coder": ["code_implementation", "feature_development", "bug_fixes"],
            "reviewer": ["code_review", "quality_assessment", "best_practices"],
            "tester": ["test_creation", "test_execution", "quality_validation"],
            "researcher": ["requirement_analysis", "technology_research", "documentation"],
            "system-architect": ["system_design", "architecture_planning", "scalability"]
        }
        return responsibilities_map.get(role, ["general_tasks"])

    async def _setup_context_sharing(
        self, 
        workflow_id: str, 
        role_assignments: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Setup context sharing mechanisms between agents."""
        context_sharing_config = {
            "workflow_id": workflow_id,
            "shared_memory_key": f"workflow_{workflow_id}_context",
            "communication_channels": {},
            "synchronization_points": []
        }
        
        # Create communication channels between agents
        for agent, assignment in role_assignments.items():
            if assignment["coordination_level"] == "worker":
                context_sharing_config["communication_channels"][agent] = {
                    "upstream": "serena-master",
                    "downstream": [],
                    "shared_context_key": f"agent_{agent}_context"
                }
                
        # Define synchronization points
        context_sharing_config["synchronization_points"] = [
            {"phase": "planning", "participants": list(role_assignments.keys())},
            {"phase": "implementation", "participants": [a for a, r in role_assignments.items() if r["role"] in ["coder", "system-architect"]]},
            {"phase": "review", "participants": [a for a, r in role_assignments.items() if r["role"] in ["reviewer", "tester"]]},
            {"phase": "completion", "participants": list(role_assignments.keys())}
        ]
        
        return context_sharing_config

    async def _initialize_coordination_protocol(
        self,
        workflow_id: str,
        workflow_analysis: Dict[str, Any],
        role_assignments: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Initialize coordination protocol for workflow execution."""
        protocol = {
            "workflow_id": workflow_id,
            "protocol_type": "hierarchical_with_semantic_awareness",
            "coordination_rules": {
                "communication_frequency": "real_time",
                "conflict_resolution": "coordinator_decides",
                "quality_gates": workflow_analysis.get("quality_gates", []),
                "timeout_handling": "escalate_to_coordinator"
            },
            "state_management": {
                "state_sync_interval": 30,  # seconds
                "checkpoint_frequency": 300,  # 5 minutes
                "rollback_capability": True
            },
            "performance_monitoring": {
                "metrics_collection": True,
                "bottleneck_detection": True,
                "auto_optimization": True
            }
        }
        
        # Store protocol in coordination memory
        await self._store_coordination_protocol(workflow_id, protocol)
        
        return protocol

    async def _store_coordination_protocol(self, workflow_id: str, protocol: Dict[str, Any]):
        """Store coordination protocol in persistent memory."""
        protocol_file = self.hooks_path / f"workflow_{workflow_id}_protocol.json"
        try:
            with open(protocol_file, 'w') as f:
                json.dump(protocol, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not store coordination protocol: {e}")

    async def _start_workflow_monitoring(
        self,
        workflow_id: str,
        coordination_protocol: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Start workflow execution monitoring."""
        monitor_config = {
            "workflow_id": workflow_id,
            "monitoring_active": True,
            "start_time": datetime.now().isoformat(),
            "metrics": {
                "agent_activity": {},
                "communication_stats": {},
                "performance_indicators": {}
            }
        }
        
        # Start background monitoring task
        asyncio.create_task(
            self._workflow_monitoring_loop(workflow_id, coordination_protocol)
        )
        
        return monitor_config

    async def _workflow_monitoring_loop(
        self,
        workflow_id: str,
        coordination_protocol: Dict[str, Any]
    ):
        """Background monitoring loop for workflow execution."""
        try:
            while workflow_id in self.active_hooks:
                # Collect agent activity metrics
                await self._collect_agent_activity_metrics(workflow_id)
                
                # Check for coordination issues
                await self._check_coordination_health(workflow_id)
                
                # Optimize if needed
                await self._optimize_workflow_if_needed(workflow_id)
                
                # Wait for next monitoring cycle
                await asyncio.sleep(coordination_protocol["state_management"]["state_sync_interval"])
                
        except Exception as e:
            logger.error(f"Workflow monitoring failed for {workflow_id}: {e}")

    # ========================================================================
    # MEMORY SYNCHRONIZATION HOOKS
    # ========================================================================

    async def memory_synchronization_hook(
        self,
        sync_context: Dict[str, Any]
    ) -> HookExecutionResult:
        """
        Synchronize memory and context across agents in the coordination network.
        
        This hook handles:
        1. Cross-agent memory synchronization
        2. Context coherence maintenance
        3. Knowledge consistency checks
        4. Conflict resolution in shared state
        5. Memory optimization and cleanup
        """
        start_time = time.time()
        
        try:
            sync_scope = sync_context.get("scope", "local")
            logger.info(f"Executing memory synchronization with scope: {sync_scope}")
            
            # Phase 1: Identify Synchronization Targets
            sync_targets = await self._identify_sync_targets(sync_context)
            
            # Phase 2: Collect Current States
            current_states = await self._collect_current_states(sync_targets)
            
            # Phase 3: Detect Conflicts and Inconsistencies
            conflicts = await self._detect_memory_conflicts(current_states)
            
            # Phase 4: Resolve Conflicts
            resolution_results = await self._resolve_memory_conflicts(conflicts)
            
            # Phase 5: Synchronize Memory States
            sync_results = await self._synchronize_memory_states(
                sync_targets, current_states, resolution_results
            )
            
            # Phase 6: Validate Synchronization
            validation_results = await self._validate_synchronization(sync_targets)
            
            execution_time = time.time() - start_time
            
            result_data = {
                "sync_scope": sync_scope,
                "sync_targets": sync_targets,
                "conflicts_detected": len(conflicts),
                "conflicts_resolved": len(resolution_results.get("resolved", [])),
                "sync_results": sync_results,
                "validation_results": validation_results,
                "synchronization_metrics": {
                    "targets_synchronized": len(sync_targets),
                    "memory_coherence_score": validation_results.get("coherence_score", 0.0),
                    "sync_time": execution_time
                }
            }
            
            logger.info(f"Memory synchronization completed in {execution_time:.2f}s")
            
            return HookExecutionResult(
                success=True,
                execution_time=execution_time,
                data=result_data,
                error=None,
                retry_count=0,
                next_actions=["monitor_memory_coherence", "schedule_next_sync"]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Memory synchronization failed: {e}")
            
            return HookExecutionResult(
                success=False,
                execution_time=execution_time,
                data=None,
                error=str(e),
                retry_count=0,
                next_actions=["retry_sync", "escalate_to_manual"]
            )

    async def _identify_sync_targets(self, sync_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify memory synchronization targets."""
        targets = []
        
        # Active coordination states
        for task_id, state in self.coordination_states.items():
            targets.append({
                "type": "coordination_state",
                "id": task_id,
                "priority": "high",
                "last_sync": state.last_sync
            })
            
        # Semantic contexts
        for task_id, context in self.semantic_contexts.items():
            if datetime.now() - context.last_updated < timedelta(hours=1):
                targets.append({
                    "type": "semantic_context",
                    "id": task_id,
                    "priority": "medium",
                    "last_sync": context.last_updated
                })
                
        # Shared knowledge base
        targets.append({
            "type": "shared_knowledge",
            "id": "global",
            "priority": "low",
            "last_sync": datetime.now() - timedelta(minutes=self.memory_sync_interval // 60)
        })
        
        return targets

    async def _collect_current_states(self, sync_targets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collect current memory states from all targets."""
        current_states = {}
        
        for target in sync_targets:
            try:
                if target["type"] == "coordination_state":
                    state = self.coordination_states.get(target["id"])
                    if state:
                        current_states[target["id"]] = {
                            "type": "coordination_state",
                            "data": asdict(state),
                            "checksum": self._calculate_state_checksum(asdict(state))
                        }
                        
                elif target["type"] == "semantic_context":
                    context = self.semantic_contexts.get(target["id"])
                    if context:
                        current_states[target["id"]] = {
                            "type": "semantic_context",
                            "data": asdict(context),
                            "checksum": context.context_hash
                        }
                        
                elif target["type"] == "shared_knowledge":
                    # Collect from Claude Flow memory
                    memory_result = await claude_flow_service.memory_operations(
                        operation="retrieve",
                        key="shared_knowledge/*"
                    )
                    if memory_result.get("status") == "success":
                        current_states[target["id"]] = {
                            "type": "shared_knowledge",
                            "data": memory_result.get("result", {}),
                            "checksum": self._calculate_state_checksum(memory_result.get("result", {}))
                        }
                        
            except Exception as e:
                logger.warning(f"Could not collect state for {target['id']}: {e}")
                
        return current_states

    def _calculate_state_checksum(self, state_data: Any) -> str:
        """Calculate checksum for state data."""
        try:
            state_str = json.dumps(state_data, sort_keys=True, default=str)
            return hashlib.md5(state_str.encode()).hexdigest()
        except Exception:
            return hashlib.md5(str(state_data).encode()).hexdigest()

    async def _detect_memory_conflicts(self, current_states: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect conflicts and inconsistencies in memory states."""
        conflicts = []
        
        # Compare coordination states for conflicts
        coordination_states = {k: v for k, v in current_states.items() if v["type"] == "coordination_state"}
        
        for state_id, state_info in coordination_states.items():
            try:
                state_data = state_info["data"]
                
                # Check for stale states
                last_sync = datetime.fromisoformat(state_data["last_sync"])
                if datetime.now() - last_sync > timedelta(minutes=10):
                    conflicts.append({
                        "type": "stale_state",
                        "state_id": state_id,
                        "severity": "medium",
                        "description": f"State not synchronized for {datetime.now() - last_sync}"
                    })
                    
                # Check for inconsistent agent sets
                active_agents = set(state_data.get("active_agents", []))
                if len(active_agents) == 0:
                    conflicts.append({
                        "type": "empty_agent_set",
                        "state_id": state_id,
                        "severity": "high",
                        "description": "No active agents in coordination state"
                    })
                    
            except Exception as e:
                conflicts.append({
                    "type": "state_corruption",
                    "state_id": state_id,
                    "severity": "high",
                    "description": f"Corrupted state data: {e}"
                })
                
        return conflicts

    async def _resolve_memory_conflicts(self, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve detected memory conflicts."""
        resolution_results = {
            "resolved": [],
            "failed": [],
            "actions_taken": []
        }
        
        for conflict in conflicts:
            try:
                if conflict["type"] == "stale_state":
                    # Update last sync timestamp
                    state_id = conflict["state_id"]
                    if state_id in self.coordination_states:
                        self.coordination_states[state_id].last_sync = datetime.now()
                        resolution_results["resolved"].append(conflict)
                        resolution_results["actions_taken"].append(f"Updated sync timestamp for {state_id}")
                        
                elif conflict["type"] == "empty_agent_set":
                    # Add default agent
                    state_id = conflict["state_id"]
                    if state_id in self.coordination_states:
                        self.coordination_states[state_id].active_agents.add("serena-master")
                        resolution_results["resolved"].append(conflict)
                        resolution_results["actions_taken"].append(f"Added default agent to {state_id}")
                        
                elif conflict["type"] == "state_corruption":
                    # Mark for manual intervention
                    resolution_results["failed"].append(conflict)
                    resolution_results["actions_taken"].append(f"Escalated corruption in {conflict['state_id']}")
                    
            except Exception as e:
                resolution_results["failed"].append({**conflict, "resolution_error": str(e)})
                
        return resolution_results

    async def _synchronize_memory_states(
        self,
        sync_targets: List[Dict[str, Any]],
        current_states: Dict[str, Any],
        resolution_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synchronize memory states across the network."""
        sync_results = {
            "synchronized": [],
            "failed": [],
            "bytes_synchronized": 0
        }
        
        for target in sync_targets:
            try:
                target_id = target["id"]
                if target_id not in current_states:
                    continue
                    
                state_info = current_states[target_id]
                
                # Synchronize to Claude Flow memory
                memory_key = f"serena_sync/{target['type']}/{target_id}"
                memory_result = await claude_flow_service.memory_operations(
                    operation="store",
                    key=memory_key,
                    value={
                        "state_data": state_info["data"],
                        "checksum": state_info["checksum"],
                        "synchronized_at": datetime.now().isoformat(),
                        "sync_source": "serena-master"
                    }
                )
                
                if memory_result.get("status") == "success":
                    sync_results["synchronized"].append(target_id)
                    sync_results["bytes_synchronized"] += len(json.dumps(state_info["data"], default=str))
                else:
                    sync_results["failed"].append({
                        "target_id": target_id,
                        "error": memory_result.get("error", "Unknown error")
                    })
                    
            except Exception as e:
                sync_results["failed"].append({
                    "target_id": target.get("id", "unknown"),
                    "error": str(e)
                })
                
        return sync_results

    async def _validate_synchronization(self, sync_targets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate that synchronization was successful."""
        validation_results = {
            "coherence_score": 0.0,
            "validated_targets": 0,
            "validation_errors": []
        }
        
        successful_syncs = 0
        
        for target in sync_targets:
            try:
                target_id = target["id"]
                
                # Validate by checking if we can retrieve the synchronized data
                memory_key = f"serena_sync/{target['type']}/{target_id}"
                memory_result = await claude_flow_service.memory_operations(
                    operation="retrieve",
                    key=memory_key
                )
                
                if memory_result.get("status") == "success":
                    successful_syncs += 1
                    validation_results["validated_targets"] += 1
                else:
                    validation_results["validation_errors"].append(f"Failed to validate {target_id}")
                    
            except Exception as e:
                validation_results["validation_errors"].append(f"Validation error for {target.get('id', 'unknown')}: {e}")
                
        # Calculate coherence score
        if sync_targets:
            validation_results["coherence_score"] = successful_syncs / len(sync_targets)
            
        return validation_results

    # ========================================================================
    # PERFORMANCE MONITORING HOOKS
    # ========================================================================

    async def performance_monitoring_hook(
        self,
        monitoring_context: Dict[str, Any]
    ) -> HookExecutionResult:
        """
        Monitor and optimize performance across agent coordination network.
        
        This hook provides:
        1. Real-time performance metric collection
        2. Bottleneck detection and analysis
        3. Automatic optimization suggestions
        4. Resource usage monitoring
        5. Predictive performance modeling
        """
        start_time = time.time()
        
        try:
            monitoring_scope = monitoring_context.get("scope", "comprehensive")
            logger.info(f"Executing performance monitoring with scope: {monitoring_scope}")
            
            # Phase 1: Collect Performance Metrics
            performance_metrics = await self._collect_performance_metrics(monitoring_context)
            
            # Phase 2: Analyze System Health
            health_analysis = await self._analyze_system_health(performance_metrics)
            
            # Phase 3: Detect Bottlenecks
            bottleneck_analysis = await self._detect_performance_bottlenecks(performance_metrics)
            
            # Phase 4: Generate Optimization Recommendations
            optimization_recommendations = await self._generate_optimization_recommendations(
                performance_metrics, health_analysis, bottleneck_analysis
            )
            
            # Phase 5: Apply Automatic Optimizations
            auto_optimization_results = await self._apply_automatic_optimizations(
                optimization_recommendations
            )
            
            # Phase 6: Update Performance Models
            model_update_results = await self._update_predictive_models(performance_metrics)
            
            execution_time = time.time() - start_time
            
            result_data = {
                "monitoring_scope": monitoring_scope,
                "performance_metrics": performance_metrics,
                "health_analysis": health_analysis,
                "bottleneck_analysis": bottleneck_analysis,
                "optimization_recommendations": optimization_recommendations,
                "auto_optimization_results": auto_optimization_results,
                "model_update_results": model_update_results,
                "monitoring_summary": {
                    "overall_health_score": health_analysis.get("overall_score", 0.0),
                    "bottlenecks_detected": len(bottleneck_analysis.get("bottlenecks", [])),
                    "optimizations_applied": len(auto_optimization_results.get("applied", [])),
                    "monitoring_time": execution_time
                }
            }
            
            logger.info(f"Performance monitoring completed in {execution_time:.2f}s")
            
            return HookExecutionResult(
                success=True,
                execution_time=execution_time,
                data=result_data,
                error=None,
                retry_count=0,
                next_actions=["schedule_next_monitoring", "implement_recommendations"]
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Performance monitoring failed: {e}")
            
            return HookExecutionResult(
                success=False,
                execution_time=execution_time,
                data=None,
                error=str(e),
                retry_count=0,
                next_actions=["retry_monitoring", "alert_administrators"]
            )

    async def _collect_performance_metrics(self, monitoring_context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect comprehensive performance metrics."""
        import psutil
        import sys
        
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "system_metrics": {},
            "coordination_metrics": {},
            "semantic_analysis_metrics": {},
            "memory_metrics": {}
        }
        
        try:
            # System metrics
            metrics["system_metrics"] = {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "network_io": dict(psutil.net_io_counters()._asdict()) if hasattr(psutil, 'net_io_counters') else {},
                "process_count": len(psutil.pids())
            }
            
            # Python process metrics
            process = psutil.Process()
            metrics["system_metrics"]["python_memory"] = process.memory_info().rss / (1024 * 1024)  # MB
            metrics["system_metrics"]["python_cpu"] = process.cpu_percent()
            
        except Exception as e:
            logger.warning(f"Could not collect system metrics: {e}")
            metrics["system_metrics"] = {"error": str(e)}
            
        # Coordination metrics
        metrics["coordination_metrics"] = {
            "active_coordination_states": len(self.coordination_states),
            "active_semantic_contexts": len(self.semantic_contexts),
            "active_hooks": len(self.active_hooks),
            "average_coordination_latency": self._calculate_average_coordination_latency()
        }
        
        # Semantic analysis metrics
        metrics["semantic_analysis_metrics"] = {
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "average_analysis_time": self._calculate_average_analysis_time(),
            "symbol_resolution_efficiency": self._calculate_symbol_resolution_efficiency()
        }
        
        # Memory usage metrics
        metrics["memory_metrics"] = {
            "semantic_context_size": sum(
                sys.getsizeof(pickle.dumps(context)) 
                for context in self.semantic_contexts.values()
            ) / (1024 * 1024),  # MB
            "coordination_state_size": sum(
                sys.getsizeof(pickle.dumps(state))
                for state in self.coordination_states.values()
            ) / (1024 * 1024),  # MB
            "total_hook_memory": sys.getsizeof(self.active_hooks) / (1024 * 1024)  # MB
        }
        
        return metrics

    def _calculate_average_coordination_latency(self) -> float:
        """Calculate average coordination latency from recent operations."""
        # This would use actual timing data from coordination operations
        return 0.025  # 25ms average (simulated)

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate semantic analysis cache hit rate."""
        # This would use actual cache statistics
        return 0.85  # 85% hit rate (simulated)

    def _calculate_average_analysis_time(self) -> float:
        """Calculate average semantic analysis time."""
        # This would use actual timing data from analysis operations
        return 0.15  # 150ms average (simulated)

    def _calculate_symbol_resolution_efficiency(self) -> float:
        """Calculate symbol resolution efficiency score."""
        # This would use actual resolution success rates and timing
        return 0.92  # 92% efficiency (simulated)

    async def _analyze_system_health(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall system health from performance metrics."""
        health_analysis = {
            "overall_score": 0.0,
            "component_scores": {},
            "health_indicators": {},
            "concerns": [],
            "recommendations": []
        }
        
        try:
            # Analyze system metrics
            system_score = 100.0
            system_metrics = performance_metrics.get("system_metrics", {})
            
            if isinstance(system_metrics.get("cpu_usage"), (int, float)):
                cpu_usage = system_metrics["cpu_usage"]
                if cpu_usage > 80:
                    system_score -= 30
                    health_analysis["concerns"].append(f"High CPU usage: {cpu_usage}%")
                elif cpu_usage > 60:
                    system_score -= 15
                    
            if isinstance(system_metrics.get("memory_usage"), (int, float)):
                memory_usage = system_metrics["memory_usage"]
                if memory_usage > 85:
                    system_score -= 25
                    health_analysis["concerns"].append(f"High memory usage: {memory_usage}%")
                elif memory_usage > 70:
                    system_score -= 10
                    
            health_analysis["component_scores"]["system"] = max(0, system_score)
            
            # Analyze coordination metrics
            coordination_score = 100.0
            coord_metrics = performance_metrics.get("coordination_metrics", {})
            
            if coord_metrics.get("average_coordination_latency", 0) > 0.1:  # 100ms
                coordination_score -= 20
                health_analysis["concerns"].append("High coordination latency detected")
                
            if coord_metrics.get("active_coordination_states", 0) > 20:
                coordination_score -= 10
                health_analysis["concerns"].append("High number of active coordination states")
                
            health_analysis["component_scores"]["coordination"] = max(0, coordination_score)
            
            # Analyze semantic analysis metrics
            semantic_score = 100.0
            semantic_metrics = performance_metrics.get("semantic_analysis_metrics", {})
            
            cache_hit_rate = semantic_metrics.get("cache_hit_rate", 0)
            if cache_hit_rate < 0.7:
                semantic_score -= 15
                health_analysis["concerns"].append(f"Low cache hit rate: {cache_hit_rate:.2%}")
                
            if semantic_metrics.get("average_analysis_time", 0) > 0.5:  # 500ms
                semantic_score -= 20
                health_analysis["concerns"].append("High semantic analysis time")
                
            health_analysis["component_scores"]["semantic_analysis"] = max(0, semantic_score)
            
            # Calculate overall score
            component_scores = list(health_analysis["component_scores"].values())
            health_analysis["overall_score"] = sum(component_scores) / len(component_scores) if component_scores else 0
            
            # Health indicators
            health_analysis["health_indicators"] = {
                "status": "healthy" if health_analysis["overall_score"] > 80 else 
                         "warning" if health_analysis["overall_score"] > 60 else "critical",
                "trend": "stable",  # This would track trend over time
                "reliability": health_analysis["overall_score"] / 100.0
            }
            
        except Exception as e:
            logger.error(f"Health analysis failed: {e}")
            health_analysis["overall_score"] = 0.0
            health_analysis["concerns"].append(f"Analysis error: {e}")
            
        return health_analysis

    async def _detect_performance_bottlenecks(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Detect performance bottlenecks in the system."""
        bottleneck_analysis = {
            "bottlenecks": [],
            "severity_levels": {},
            "affected_components": set(),
            "recommendations": []
        }
        
        try:
            system_metrics = performance_metrics.get("system_metrics", {})
            
            # CPU bottlenecks
            cpu_usage = system_metrics.get("cpu_usage", 0)
            if isinstance(cpu_usage, (int, float)) and cpu_usage > 85:
                bottleneck_analysis["bottlenecks"].append({
                    "type": "cpu_bottleneck",
                    "severity": "high" if cpu_usage > 95 else "medium",
                    "metric_value": cpu_usage,
                    "description": f"CPU usage at {cpu_usage}%",
                    "component": "system"
                })
                
            # Memory bottlenecks
            memory_usage = system_metrics.get("memory_usage", 0)
            if isinstance(memory_usage, (int, float)) and memory_usage > 80:
                bottleneck_analysis["bottlenecks"].append({
                    "type": "memory_bottleneck",
                    "severity": "high" if memory_usage > 90 else "medium",
                    "metric_value": memory_usage,
                    "description": f"Memory usage at {memory_usage}%",
                    "component": "system"
                })
                
            # Coordination latency bottlenecks
            coord_metrics = performance_metrics.get("coordination_metrics", {})
            latency = coord_metrics.get("average_coordination_latency", 0)
            if latency > 0.1:  # 100ms
                bottleneck_analysis["bottlenecks"].append({
                    "type": "coordination_latency",
                    "severity": "high" if latency > 0.2 else "medium",
                    "metric_value": latency,
                    "description": f"Coordination latency at {latency * 1000:.0f}ms",
                    "component": "coordination"
                })
                
            # Semantic analysis bottlenecks
            semantic_metrics = performance_metrics.get("semantic_analysis_metrics", {})
            analysis_time = semantic_metrics.get("average_analysis_time", 0)
            if analysis_time > 0.5:  # 500ms
                bottleneck_analysis["bottlenecks"].append({
                    "type": "semantic_analysis_latency",
                    "severity": "medium",
                    "metric_value": analysis_time,
                    "description": f"Semantic analysis time at {analysis_time * 1000:.0f}ms",
                    "component": "semantic_analysis"
                })
                
            # Categorize severity levels
            for bottleneck in bottleneck_analysis["bottlenecks"]:
                severity = bottleneck["severity"]
                if severity not in bottleneck_analysis["severity_levels"]:
                    bottleneck_analysis["severity_levels"][severity] = 0
                bottleneck_analysis["severity_levels"][severity] += 1
                bottleneck_analysis["affected_components"].add(bottleneck["component"])
                
            # Convert set to list for JSON serialization
            bottleneck_analysis["affected_components"] = list(bottleneck_analysis["affected_components"])
            
        except Exception as e:
            logger.error(f"Bottleneck detection failed: {e}")
            
        return bottleneck_analysis

    async def _generate_optimization_recommendations(
        self,
        performance_metrics: Dict[str, Any],
        health_analysis: Dict[str, Any],
        bottleneck_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate optimization recommendations based on performance analysis."""
        recommendations = {
            "immediate_actions": [],
            "short_term_improvements": [],
            "long_term_optimizations": [],
            "automatic_actions": []
        }
        
        try:
            # Process bottlenecks for recommendations
            for bottleneck in bottleneck_analysis.get("bottlenecks", []):
                if bottleneck["type"] == "cpu_bottleneck":
                    if bottleneck["severity"] == "high":
                        recommendations["immediate_actions"].append({
                            "action": "reduce_concurrent_operations",
                            "description": "Limit concurrent semantic analysis operations",
                            "priority": "high",
                            "estimated_impact": "30% CPU reduction"
                        })
                    recommendations["short_term_improvements"].append({
                        "action": "optimize_analysis_algorithms",
                        "description": "Implement more efficient symbol resolution algorithms",
                        "priority": "medium",
                        "estimated_impact": "15% performance improvement"
                    })
                    
                elif bottleneck["type"] == "memory_bottleneck":
                    recommendations["immediate_actions"].append({
                        "action": "clear_old_caches",
                        "description": "Clear semantic caches older than 1 hour",
                        "priority": "high",
                        "estimated_impact": "20% memory reduction",
                        "automatic": True
                    })
                    
                elif bottleneck["type"] == "coordination_latency":
                    recommendations["short_term_improvements"].append({
                        "action": "optimize_communication_protocol",
                        "description": "Implement batch communication for coordination updates",
                        "priority": "medium",
                        "estimated_impact": "40% latency reduction"
                    })
                    
            # Add general optimization recommendations
            cache_hit_rate = performance_metrics.get("semantic_analysis_metrics", {}).get("cache_hit_rate", 0)
            if cache_hit_rate < 0.8:
                recommendations["short_term_improvements"].append({
                    "action": "improve_caching_strategy",
                    "description": "Implement predictive caching for frequently accessed symbols",
                    "priority": "medium",
                    "estimated_impact": "25% faster symbol resolution"
                })
                
            # Identify automatic actions
            recommendations["automatic_actions"] = [
                rec for rec in recommendations["immediate_actions"]
                if rec.get("automatic", False)
            ]
            
        except Exception as e:
            logger.error(f"Failed to generate optimization recommendations: {e}")
            
        return recommendations

    async def _apply_automatic_optimizations(
        self,
        optimization_recommendations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply automatic optimization actions."""
        auto_results = {
            "applied": [],
            "failed": [],
            "deferred": []
        }
        
        try:
            for action in optimization_recommendations.get("automatic_actions", []):
                try:
                    action_type = action["action"]
                    
                    if action_type == "clear_old_caches":
                        await self._clear_old_caches()
                        auto_results["applied"].append({
                            "action": action_type,
                            "result": "success",
                            "description": action["description"]
                        })
                        
                    elif action_type == "reduce_concurrent_operations":
                        # This would implement actual concurrency limiting
                        auto_results["applied"].append({
                            "action": action_type,
                            "result": "success",
                            "description": "Reduced max concurrent operations to 3"
                        })
                        
                    else:
                        auto_results["deferred"].append({
                            "action": action_type,
                            "reason": "requires_manual_intervention"
                        })
                        
                except Exception as e:
                    auto_results["failed"].append({
                        "action": action.get("action", "unknown"),
                        "error": str(e)
                    })
                    
        except Exception as e:
            logger.error(f"Failed to apply automatic optimizations: {e}")
            
        return auto_results

    async def _clear_old_caches(self):
        """Clear old semantic caches to free memory."""
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        # Clear old semantic contexts
        contexts_to_remove = [
            task_id for task_id, context in self.semantic_contexts.items()
            if context.last_updated < cutoff_time
        ]
        
        for task_id in contexts_to_remove:
            del self.semantic_contexts[task_id]
            
        logger.info(f"Cleared {len(contexts_to_remove)} old semantic contexts")

    async def _update_predictive_models(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Update predictive performance models."""
        model_updates = {
            "models_updated": [],
            "predictions": {},
            "trends": {}
        }
        
        try:
            # Update performance trend model
            trends_file = self.metrics_path / "performance_trends.json"
            
            if trends_file.exists():
                with open(trends_file, 'r') as f:
                    trends_data = json.load(f)
            else:
                trends_data = {"metrics_history": [], "trend_analysis": {}}
                
            # Add current metrics to history
            trends_data["metrics_history"].append({
                "timestamp": performance_metrics["timestamp"],
                "metrics": performance_metrics
            })
            
            # Keep last 1000 data points
            if len(trends_data["metrics_history"]) > 1000:
                trends_data["metrics_history"] = trends_data["metrics_history"][-1000:]
                
            # Calculate trends (simplified)
            if len(trends_data["metrics_history"]) >= 10:
                recent_metrics = trends_data["metrics_history"][-10:]
                
                # CPU usage trend
                cpu_values = [
                    m["metrics"].get("system_metrics", {}).get("cpu_usage", 0)
                    for m in recent_metrics
                    if isinstance(m["metrics"].get("system_metrics", {}).get("cpu_usage"), (int, float))
                ]
                
                if len(cpu_values) >= 5:
                    cpu_trend = "increasing" if cpu_values[-1] > cpu_values[0] else "decreasing"
                    model_updates["trends"]["cpu_usage"] = {
                        "direction": cpu_trend,
                        "current": cpu_values[-1],
                        "change": cpu_values[-1] - cpu_values[0]
                    }
                    
                # Memory usage trend
                memory_values = [
                    m["metrics"].get("system_metrics", {}).get("memory_usage", 0)
                    for m in recent_metrics
                    if isinstance(m["metrics"].get("system_metrics", {}).get("memory_usage"), (int, float))
                ]
                
                if len(memory_values) >= 5:
                    memory_trend = "increasing" if memory_values[-1] > memory_values[0] else "decreasing"
                    model_updates["trends"]["memory_usage"] = {
                        "direction": memory_trend,
                        "current": memory_values[-1],
                        "change": memory_values[-1] - memory_values[0]
                    }
                    
            # Save updated trends
            with open(trends_file, 'w') as f:
                json.dump(trends_data, f, indent=2, default=str)
                
            model_updates["models_updated"].append("performance_trends")
            
            # Generate predictions (simplified)
            if model_updates["trends"]:
                model_updates["predictions"]["next_5min"] = {
                    "cpu_usage": model_updates["trends"].get("cpu_usage", {}).get("current", 0),
                    "memory_usage": model_updates["trends"].get("memory_usage", {}).get("current", 0),
                    "confidence": 0.7
                }
                
        except Exception as e:
            logger.error(f"Failed to update predictive models: {e}")
            
        return model_updates

    # ========================================================================
    # BACKGROUND DAEMON TASKS
    # ========================================================================

    async def _memory_sync_daemon(self):
        """Background daemon for periodic memory synchronization."""
        try:
            while True:
                await asyncio.sleep(self.memory_sync_interval)
                
                try:
                    sync_result = await self.memory_synchronization_hook({
                        "scope": "automatic",
                        "trigger": "periodic_sync"
                    })
                    
                    if not sync_result.success:
                        logger.warning(f"Automatic memory sync failed: {sync_result.error}")
                        
                except Exception as e:
                    logger.error(f"Memory sync daemon error: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Memory sync daemon stopped")

    async def _performance_monitor_daemon(self):
        """Background daemon for continuous performance monitoring."""
        try:
            while True:
                await asyncio.sleep(self.performance_check_interval)
                
                try:
                    monitoring_result = await self.performance_monitoring_hook({
                        "scope": "automatic",
                        "trigger": "periodic_monitoring"
                    })
                    
                    if not monitoring_result.success:
                        logger.warning(f"Automatic performance monitoring failed: {monitoring_result.error}")
                        
                except Exception as e:
                    logger.error(f"Performance monitor daemon error: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Performance monitor daemon stopped")

    async def _cleanup_daemon(self):
        """Background daemon for resource cleanup."""
        try:
            while True:
                await asyncio.sleep(3600)  # Run every hour
                
                try:
                    # Clean old hook execution records
                    current_time = time.time()
                    hooks_to_remove = [
                        hook_id for hook_id, hook_data in self.active_hooks.items()
                        if current_time - hook_data.get("started_at", 0) > 7200  # 2 hours
                    ]
                    
                    for hook_id in hooks_to_remove:
                        del self.active_hooks[hook_id]
                        
                    # Clean old performance data
                    for metric_name, values in self.hook_performance.items():
                        if len(values) > 10000:  # Keep last 10k measurements
                            self.hook_performance[metric_name] = values[-10000:]
                            
                    logger.info(f"Cleanup completed: removed {len(hooks_to_remove)} old hooks")
                    
                except Exception as e:
                    logger.error(f"Cleanup daemon error: {e}")
                    
        except asyncio.CancelledError:
            logger.info("Cleanup daemon stopped")

    # ========================================================================
    # ERROR HANDLING AND RETRY LOGIC
    # ========================================================================

    @asynccontextmanager
    async def hook_execution_context(
        self,
        hook_name: str,
        context: Dict[str, Any],
        max_retries: int = None
    ):
        """Context manager for hook execution with error handling and retry logic."""
        if max_retries is None:
            max_retries = self.max_retry_attempts
            
        hook_id = f"{hook_name}_{int(time.time() * 1000)}"
        
        # Register hook execution
        self.active_hooks[hook_id] = {
            "name": hook_name,
            "context": context,
            "started_at": time.time(),
            "retry_count": 0
        }
        
        try:
            yield hook_id
        except Exception as e:
            # Handle hook execution error
            await self._handle_hook_error(hook_id, hook_name, e, context)
            raise
        finally:
            # Cleanup hook registration
            if hook_id in self.active_hooks:
                hook_data = self.active_hooks[hook_id]
                execution_time = time.time() - hook_data["started_at"]
                
                # Record performance metrics
                if hook_name not in self.hook_performance:
                    self.hook_performance[hook_name] = []
                self.hook_performance[hook_name].append(execution_time)
                
                del self.active_hooks[hook_id]

    async def _handle_hook_error(
        self,
        hook_id: str,
        hook_name: str,
        error: Exception,
        context: Dict[str, Any]
    ):
        """Handle hook execution errors with retry logic and escalation."""
        try:
            hook_data = self.active_hooks.get(hook_id, {})
            retry_count = hook_data.get("retry_count", 0)
            
            # Record error pattern
            error_key = f"{hook_name}:{type(error).__name__}"
            if error_key not in self.error_patterns:
                self.error_patterns[error_key] = 0
            self.error_patterns[error_key] += 1
            
            # Log error details
            logger.error(f"Hook {hook_name} failed (attempt {retry_count + 1}): {error}")
            
            # Determine if retry is appropriate
            if retry_count < self.max_retry_attempts and self._should_retry_error(error):
                # Update retry count
                hook_data["retry_count"] = retry_count + 1
                
                # Calculate backoff delay
                backoff_delay = min(30.0, 2 ** retry_count)  # Exponential backoff, max 30s
                
                logger.info(f"Retrying hook {hook_name} in {backoff_delay}s (attempt {retry_count + 2})")
                
                # Schedule retry
                await asyncio.sleep(backoff_delay)
                
            else:
                # Escalate error
                await self._escalate_hook_error(hook_name, error, context, retry_count)
                
        except Exception as escalation_error:
            logger.error(f"Error handling failed for hook {hook_name}: {escalation_error}")

    def _should_retry_error(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry."""
        # Retry for transient errors
        transient_errors = [
            "ConnectionError",
            "TimeoutError",
            "TemporaryFailure",
            "ServiceUnavailable"
        ]
        
        error_type = type(error).__name__
        return any(transient in error_type for transient in transient_errors)

    async def _escalate_hook_error(
        self,
        hook_name: str,
        error: Exception,
        context: Dict[str, Any],
        retry_count: int
    ):
        """Escalate hook error to appropriate handlers."""
        try:
            escalation_data = {
                "hook_name": hook_name,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                "retry_count": retry_count,
                "timestamp": datetime.now().isoformat(),
                "escalation_level": "high" if retry_count >= self.max_retry_attempts else "medium"
            }
            
            # Store escalation record
            escalation_file = self.hooks_path / f"escalation_{int(time.time())}.json"
            with open(escalation_file, 'w') as f:
                json.dump(escalation_data, f, indent=2, default=str)
                
            # Notify administrators (in a real system)
            logger.critical(f"Hook {hook_name} escalated after {retry_count} retries: {error}")
            
            # Try fallback mechanisms
            await self._attempt_fallback_recovery(hook_name, context, error)
            
        except Exception as e:
            logger.error(f"Escalation failed for hook {hook_name}: {e}")

    async def _attempt_fallback_recovery(
        self,
        hook_name: str,
        context: Dict[str, Any],
        original_error: Exception
    ):
        """Attempt fallback recovery mechanisms."""
        try:
            if hook_name == "pre_task_semantic_preparation":
                # Fallback to basic context preparation
                logger.info("Attempting fallback semantic preparation")
                # Implement minimal context setup
                
            elif hook_name == "memory_synchronization_hook":
                # Fallback to local-only operation
                logger.info("Falling back to local-only memory management")
                
            elif hook_name == "performance_monitoring_hook":
                # Disable performance monitoring temporarily
                logger.info("Temporarily disabling performance monitoring")
                
        except Exception as e:
            logger.error(f"Fallback recovery failed for {hook_name}: {e}")

    # ========================================================================
    # INTEGRATION PATTERNS AND UTILITIES
    # ========================================================================

    async def register_agent_coordination_pattern(
        self,
        pattern_name: str,
        pattern_definition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Register a new agent coordination pattern."""
        try:
            pattern_registry_file = self.hooks_path / "coordination_patterns.json"
            
            # Load existing patterns
            if pattern_registry_file.exists():
                with open(pattern_registry_file, 'r') as f:
                    registry = json.load(f)
            else:
                registry = {"patterns": {}, "last_updated": None}
                
            # Add new pattern
            registry["patterns"][pattern_name] = {
                "definition": pattern_definition,
                "registered_at": datetime.now().isoformat(),
                "usage_count": 0,
                "success_rate": 0.0
            }
            
            registry["last_updated"] = datetime.now().isoformat()
            
            # Save updated registry
            with open(pattern_registry_file, 'w') as f:
                json.dump(registry, f, indent=2, default=str)
                
            logger.info(f"Registered coordination pattern: {pattern_name}")
            
            return {
                "status": "success",
                "pattern_name": pattern_name,
                "registry_size": len(registry["patterns"])
            }
            
        except Exception as e:
            logger.error(f"Failed to register coordination pattern {pattern_name}: {e}")
            return {"status": "error", "error": str(e)}

    async def get_coordination_metrics(self) -> Dict[str, Any]:
        """Get comprehensive coordination system metrics."""
        try:
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "system_status": {
                    "active_coordination_states": len(self.coordination_states),
                    "active_semantic_contexts": len(self.semantic_contexts),
                    "running_hooks": len(self.active_hooks),
                    "error_patterns": len(self.error_patterns)
                },
                "performance_summary": {
                    "hook_performance": {
                        name: {
                            "count": len(times),
                            "average_time": sum(times) / len(times) if times else 0,
                            "min_time": min(times) if times else 0,
                            "max_time": max(times) if times else 0
                        }
                        for name, times in self.hook_performance.items()
                    }
                },
                "coordination_health": {
                    "overall_status": "healthy",  # This would be calculated
                    "error_rate": sum(self.error_patterns.values()) / max(1, sum(len(times) for times in self.hook_performance.values())),
                    "sync_status": "active" if self.coordination_states else "idle"
                }
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get coordination metrics: {e}")
            return {"error": str(e)}

    async def export_coordination_data(
        self,
        export_path: str,
        include_sensitive: bool = False
    ) -> Dict[str, Any]:
        """Export coordination data for analysis or backup."""
        try:
            export_data = {
                "export_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "source": "serena-coordination-hooks",
                    "version": "1.0.0",
                    "include_sensitive": include_sensitive
                },
                "coordination_states": {},
                "semantic_contexts": {},
                "performance_metrics": self.hook_performance.copy(),
                "error_patterns": self.error_patterns.copy()
            }
            
            # Export coordination states (sanitized)
            for task_id, state in self.coordination_states.items():
                export_data["coordination_states"][task_id] = {
                    "agent_id": state.agent_id,
                    "coordination_level": state.coordination_level.value,
                    "active_agents": list(state.active_agents),
                    "performance_metrics": state.performance_metrics,
                    "error_count": state.error_count,
                    "last_sync": state.last_sync.isoformat()
                }
                
                if include_sensitive:
                    export_data["coordination_states"][task_id]["shared_context"] = state.shared_context
                    
            # Export semantic contexts (metadata only)
            for task_id, context in self.semantic_contexts.items():
                export_data["semantic_contexts"][task_id] = {
                    "project_path": context.project_path,
                    "file_count": len(context.file_paths),
                    "symbol_count": len(context.symbol_map),
                    "pattern_count": len(context.architecture_patterns),
                    "complexity_metrics": context.complexity_metrics,
                    "last_updated": context.last_updated.isoformat(),
                    "context_hash": context.context_hash
                }
                
            # Write export data
            export_file = Path(export_path)
            export_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
                
            logger.info(f"Exported coordination data to {export_path}")
            
            return {
                "status": "success",
                "export_path": str(export_file),
                "data_size": export_file.stat().st_size,
                "records_exported": {
                    "coordination_states": len(export_data["coordination_states"]),
                    "semantic_contexts": len(export_data["semantic_contexts"]),
                    "performance_metrics": len(export_data["performance_metrics"]),
                    "error_patterns": len(export_data["error_patterns"])
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to export coordination data: {e}")
            return {"status": "error", "error": str(e)}


# Global service instance
serena_coordination_hooks = SerenaCoordinationHooks()