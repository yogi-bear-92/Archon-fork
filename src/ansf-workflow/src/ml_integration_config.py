"""
ML Integration Configuration for ANSF System
Configures ML-enhanced coordination hooks with existing ANSF components

Author: Claude Code ML Developer
Integration Target: 94.7% coordination accuracy, 100MB semantic cache optimization
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class MLModelConfig:
    """Configuration for neural model integration."""
    model_path: Optional[str] = None
    model_type: str = "classification"  # classification, regression, neural_network
    target_accuracy: float = 0.887  # Baseline from model_1757102214409_0rv1o7t24
    classes: int = 5  # 5-class coordination optimization
    feature_count: int = 10
    prediction_confidence_threshold: float = 0.7
    learning_rate: float = 0.001
    batch_size: int = 32
    enable_retraining: bool = True
    retraining_threshold: int = 100  # Retrain after 100 new samples


@dataclass
class ANSFIntegrationConfig:
    """Configuration for ANSF system integration."""
    enable_phase2_integration: bool = True
    semantic_cache_budget_mb: int = 100  # 100MB target
    target_coordination_accuracy: float = 0.947  # 94.7% ANSF Phase 2 target
    lsp_integration_enabled: bool = True
    neural_learning_enabled: bool = True
    cross_language_analysis: bool = True
    progressive_loading: bool = True
    memory_optimization_threshold: float = 0.95  # 95% memory usage threshold
    
    # ANSF Phase 2 component paths
    phase2_orchestrator_path: str = "ansf-phase2-orchestrator.js"
    semantic_cache_path: str = "semantic/phase2-semantic-cache.js"
    lsp_integration_path: str = "semantic/lsp-integration.js"
    integration_hooks_path: str = "semantic/integration-hooks.js"


@dataclass
class SerenaIntegrationConfig:
    """Configuration for Serena coordination hooks integration."""
    enable_serena_integration: bool = True
    coordination_api_endpoint: str = "http://localhost:8080/serena/coordination"
    hook_phases: list = field(default_factory=lambda: [
        "pre_task", "post_task", "pre_edit", "post_edit", 
        "memory_sync", "performance_monitor", "error_recovery"
    ])
    coordination_levels: list = field(default_factory=lambda: [
        "individual", "pairwise", "group", "swarm", "ecosystem"
    ])
    semantic_analysis_timeout: int = 30  # seconds
    cache_strategy: str = "memory_first"


@dataclass
class ClaudeFlowIntegrationConfig:
    """Configuration for Claude Flow coordination integration."""
    enable_claude_flow: bool = True
    swarm_topologies: list = field(default_factory=lambda: [
        "mesh", "hierarchical", "ring", "star", "adaptive"
    ])
    coordination_strategies: list = field(default_factory=lambda: [
        "parallel", "sequential", "adaptive", "balanced"
    ])
    max_agents: int = 8
    performance_monitoring: bool = True
    neural_pattern_training: bool = True
    cross_session_memory: bool = True


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring and optimization."""
    target_metrics: Dict[str, float] = field(default_factory=lambda: {
        "coordination_efficiency": 0.947,  # 94.7% target
        "task_completion_rate": 0.95,
        "error_rate": 0.02,
        "memory_efficiency": 0.85,
        "cpu_utilization": 0.70,
        "cache_hit_ratio": 0.80,
        "semantic_accuracy": 0.90,
        "neural_prediction_accuracy": 0.887
    })
    
    monitoring_intervals: Dict[str, int] = field(default_factory=lambda: {
        "real_time_metrics": 5,    # seconds
        "performance_analysis": 30,  # seconds
        "learning_updates": 300,     # 5 minutes
        "model_retraining": 3600     # 1 hour
    })
    
    optimization_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "memory_critical": 0.95,
        "memory_warning": 0.85,
        "cpu_high": 0.90,
        "error_rate_high": 0.10,
        "coordination_efficiency_low": 0.70
    })


@dataclass
class MLEnhancedCoordinationConfig:
    """Master configuration for ML-enhanced coordination system."""
    
    # Core configurations
    ml_model: MLModelConfig = field(default_factory=MLModelConfig)
    ansf_integration: ANSFIntegrationConfig = field(default_factory=ANSFIntegrationConfig)
    serena_integration: SerenaIntegrationConfig = field(default_factory=SerenaIntegrationConfig)
    claude_flow_integration: ClaudeFlowIntegrationConfig = field(default_factory=ClaudeFlowIntegrationConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # System settings
    log_level: str = "INFO"
    enable_debug_mode: bool = False
    data_persistence_path: str = "data/ml_coordination"
    model_cache_path: str = "models/cache"
    metrics_export_path: str = "metrics/coordination"
    
    # Integration flags
    enable_fallback_mode: bool = True
    enable_heuristic_backup: bool = True
    enable_performance_profiling: bool = True
    enable_ml_explainability: bool = False
    
    # Resource management
    max_memory_usage_mb: int = 512
    max_concurrent_predictions: int = 10
    prediction_timeout_seconds: int = 30
    cache_cleanup_interval: int = 3600  # 1 hour
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        self._validate_config()
        self._setup_paths()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate target accuracy
        if not 0.5 <= self.ml_model.target_accuracy <= 1.0:
            raise ValueError(f"Invalid target accuracy: {self.ml_model.target_accuracy}")
        
        # Validate coordination accuracy target
        if not 0.5 <= self.ansf_integration.target_coordination_accuracy <= 1.0:
            raise ValueError(f"Invalid coordination accuracy target: {self.ansf_integration.target_coordination_accuracy}")
        
        # Validate memory settings
        if self.ansf_integration.semantic_cache_budget_mb > self.max_memory_usage_mb:
            logger.warning(f"Semantic cache budget ({self.ansf_integration.semantic_cache_budget_mb}MB) "
                          f"exceeds max memory ({self.max_memory_usage_mb}MB)")
        
        # Validate performance thresholds
        for metric, threshold in self.performance.optimization_thresholds.items():
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"Invalid threshold for {metric}: {threshold}")
    
    def _setup_paths(self):
        """Setup and create necessary directory paths."""
        paths_to_create = [
            self.data_persistence_path,
            self.model_cache_path,
            self.metrics_export_path
        ]
        
        for path_str in paths_to_create:
            path = Path(path_str)
            path.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "ml_model": {
                "model_path": self.ml_model.model_path,
                "model_type": self.ml_model.model_type,
                "target_accuracy": self.ml_model.target_accuracy,
                "classes": self.ml_model.classes,
                "feature_count": self.ml_model.feature_count,
                "prediction_confidence_threshold": self.ml_model.prediction_confidence_threshold,
                "enable_retraining": self.ml_model.enable_retraining
            },
            "ansf_integration": {
                "enable_phase2_integration": self.ansf_integration.enable_phase2_integration,
                "semantic_cache_budget_mb": self.ansf_integration.semantic_cache_budget_mb,
                "target_coordination_accuracy": self.ansf_integration.target_coordination_accuracy,
                "lsp_integration_enabled": self.ansf_integration.lsp_integration_enabled,
                "neural_learning_enabled": self.ansf_integration.neural_learning_enabled
            },
            "serena_integration": {
                "enable_serena_integration": self.serena_integration.enable_serena_integration,
                "coordination_api_endpoint": self.serena_integration.coordination_api_endpoint,
                "hook_phases": self.serena_integration.hook_phases,
                "coordination_levels": self.serena_integration.coordination_levels
            },
            "claude_flow_integration": {
                "enable_claude_flow": self.claude_flow_integration.enable_claude_flow,
                "swarm_topologies": self.claude_flow_integration.swarm_topologies,
                "max_agents": self.claude_flow_integration.max_agents,
                "performance_monitoring": self.claude_flow_integration.performance_monitoring
            },
            "performance": {
                "target_metrics": self.performance.target_metrics,
                "monitoring_intervals": self.performance.monitoring_intervals,
                "optimization_thresholds": self.performance.optimization_thresholds
            },
            "system": {
                "log_level": self.log_level,
                "enable_debug_mode": self.enable_debug_mode,
                "enable_fallback_mode": self.enable_fallback_mode,
                "max_memory_usage_mb": self.max_memory_usage_mb
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MLEnhancedCoordinationConfig':
        """Create configuration from dictionary."""
        config = cls()
        
        # Update ML model config
        if "ml_model" in config_dict:
            ml_config = config_dict["ml_model"]
            config.ml_model.model_path = ml_config.get("model_path", config.ml_model.model_path)
            config.ml_model.target_accuracy = ml_config.get("target_accuracy", config.ml_model.target_accuracy)
            config.ml_model.classes = ml_config.get("classes", config.ml_model.classes)
            config.ml_model.enable_retraining = ml_config.get("enable_retraining", config.ml_model.enable_retraining)
        
        # Update ANSF integration config
        if "ansf_integration" in config_dict:
            ansf_config = config_dict["ansf_integration"]
            config.ansf_integration.semantic_cache_budget_mb = ansf_config.get(
                "semantic_cache_budget_mb", config.ansf_integration.semantic_cache_budget_mb)
            config.ansf_integration.target_coordination_accuracy = ansf_config.get(
                "target_coordination_accuracy", config.ansf_integration.target_coordination_accuracy)
        
        # Update other configs similarly...
        
        return config
    
    def save_to_file(self, file_path: str):
        """Save configuration to JSON file."""
        config_dict = self.to_dict()
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {file_path}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'MLEnhancedCoordinationConfig':
        """Load configuration from JSON file."""
        if not Path(file_path).exists():
            logger.warning(f"Configuration file {file_path} not found, using defaults")
            return cls()
        
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        
        logger.info(f"Configuration loaded from {file_path}")
        return cls.from_dict(config_dict)


class ConfigurationManager:
    """Manages configuration for ML-enhanced coordination system."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/ml_coordination_config.json"
        self.config = self._load_configuration()
        
    def _load_configuration(self) -> MLEnhancedCoordinationConfig:
        """Load configuration with environment variable overrides."""
        # Load base configuration
        if Path(self.config_file).exists():
            config = MLEnhancedCoordinationConfig.load_from_file(self.config_file)
        else:
            config = MLEnhancedCoordinationConfig()
            # Save default configuration
            Path(self.config_file).parent.mkdir(parents=True, exist_ok=True)
            config.save_to_file(self.config_file)
        
        # Apply environment variable overrides
        config = self._apply_environment_overrides(config)
        
        return config
    
    def _apply_environment_overrides(self, config: MLEnhancedCoordinationConfig) -> MLEnhancedCoordinationConfig:
        """Apply configuration overrides from environment variables."""
        
        # Neural model overrides
        if os.getenv('ML_MODEL_PATH'):
            config.ml_model.model_path = os.getenv('ML_MODEL_PATH')
        
        if os.getenv('ML_TARGET_ACCURACY'):
            config.ml_model.target_accuracy = float(os.getenv('ML_TARGET_ACCURACY'))
        
        # ANSF integration overrides
        if os.getenv('ANSF_SEMANTIC_CACHE_MB'):
            config.ansf_integration.semantic_cache_budget_mb = int(os.getenv('ANSF_SEMANTIC_CACHE_MB'))
        
        if os.getenv('ANSF_TARGET_ACCURACY'):
            config.ansf_integration.target_coordination_accuracy = float(os.getenv('ANSF_TARGET_ACCURACY'))
        
        # Serena integration overrides
        if os.getenv('SERENA_API_ENDPOINT'):
            config.serena_integration.coordination_api_endpoint = os.getenv('SERENA_API_ENDPOINT')
        
        # Claude Flow overrides
        if os.getenv('CLAUDE_FLOW_MAX_AGENTS'):
            config.claude_flow_integration.max_agents = int(os.getenv('CLAUDE_FLOW_MAX_AGENTS'))
        
        # Performance overrides
        if os.getenv('MAX_MEMORY_MB'):
            config.max_memory_usage_mb = int(os.getenv('MAX_MEMORY_MB'))
        
        # Debug mode
        if os.getenv('DEBUG_MODE'):
            config.enable_debug_mode = os.getenv('DEBUG_MODE').lower() == 'true'
        
        return config
    
    def get_config(self) -> MLEnhancedCoordinationConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        config_dict = self.config.to_dict()
        
        # Deep update
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(config_dict, updates)
        
        # Create new config and save
        self.config = MLEnhancedCoordinationConfig.from_dict(config_dict)
        self.config.save_to_file(self.config_file)
        
        logger.info("Configuration updated and saved")
    
    def get_integration_settings(self) -> Dict[str, Dict[str, Any]]:
        """Get integration-specific settings for each system."""
        return {
            "ansf": {
                "enabled": self.config.ansf_integration.enable_phase2_integration,
                "cache_budget": self.config.ansf_integration.semantic_cache_budget_mb,
                "target_accuracy": self.config.ansf_integration.target_coordination_accuracy,
                "lsp_enabled": self.config.ansf_integration.lsp_integration_enabled,
                "neural_learning": self.config.ansf_integration.neural_learning_enabled
            },
            "serena": {
                "enabled": self.config.serena_integration.enable_serena_integration,
                "api_endpoint": self.config.serena_integration.coordination_api_endpoint,
                "hook_phases": self.config.serena_integration.hook_phases,
                "coordination_levels": self.config.serena_integration.coordination_levels,
                "cache_strategy": self.config.serena_integration.cache_strategy
            },
            "claude_flow": {
                "enabled": self.config.claude_flow_integration.enable_claude_flow,
                "topologies": self.config.claude_flow_integration.swarm_topologies,
                "strategies": self.config.claude_flow_integration.coordination_strategies,
                "max_agents": self.config.claude_flow_integration.max_agents,
                "performance_monitoring": self.config.claude_flow_integration.performance_monitoring
            },
            "ml_model": {
                "model_path": self.config.ml_model.model_path,
                "target_accuracy": self.config.ml_model.target_accuracy,
                "classes": self.config.ml_model.classes,
                "feature_count": self.config.ml_model.feature_count,
                "enable_retraining": self.config.ml_model.enable_retraining
            }
        }
    
    def validate_integration_requirements(self) -> Dict[str, bool]:
        """Validate that all integration requirements are met."""
        requirements = {
            "ml_model_available": True,
            "ansf_components_available": True,
            "serena_api_accessible": True,
            "claude_flow_accessible": True,
            "sufficient_memory": True,
            "required_dependencies": True
        }
        
        # Check ML model
        if self.config.ml_model.model_path:
            requirements["ml_model_available"] = Path(self.config.ml_model.model_path).exists()
        
        # Check ANSF components
        ansf_paths = [
            self.config.ansf_integration.phase2_orchestrator_path,
            self.config.ansf_integration.semantic_cache_path,
            self.config.ansf_integration.lsp_integration_path
        ]
        requirements["ansf_components_available"] = all(
            Path(path).exists() for path in ansf_paths if path
        )
        
        # Check memory requirements
        available_memory_mb = self._get_available_memory_mb()
        requirements["sufficient_memory"] = (
            available_memory_mb >= self.config.ansf_integration.semantic_cache_budget_mb + 200
        )
        
        return requirements
    
    def _get_available_memory_mb(self) -> int:
        """Get available system memory in MB."""
        try:
            import psutil
            return psutil.virtual_memory().available // (1024 * 1024)
        except ImportError:
            # Fallback estimate
            return 1024  # Assume 1GB available
    
    def get_optimization_profile(self, system_state: Dict[str, Any]) -> str:
        """Get recommended optimization profile based on current system state."""
        memory_usage = system_state.get('memory_usage_percent', 50)
        cpu_usage = system_state.get('cpu_usage_percent', 50)
        agent_count = system_state.get('agent_count', 3)
        
        if memory_usage > 95 or cpu_usage > 90:
            return "critical"
        elif memory_usage > 85 or cpu_usage > 80 or agent_count > 8:
            return "suboptimal"
        elif memory_usage < 50 and cpu_usage < 60 and agent_count <= 5:
            return "optimal"
        elif memory_usage < 70 and cpu_usage < 70:
            return "efficient"
        else:
            return "moderate"


# Global configuration manager instance
config_manager = ConfigurationManager()


def get_ml_coordination_config() -> MLEnhancedCoordinationConfig:
    """Get the global ML coordination configuration."""
    return config_manager.get_config()


def get_integration_settings() -> Dict[str, Dict[str, Any]]:
    """Get integration settings for all systems."""
    return config_manager.get_integration_settings()


def update_coordination_config(updates: Dict[str, Any]):
    """Update the global coordination configuration."""
    config_manager.update_config(updates)


if __name__ == "__main__":
    # Example usage and testing
    print("ML-Enhanced Coordination Configuration Manager")
    print("=" * 50)
    
    # Load configuration
    config = get_ml_coordination_config()
    
    print(f"Target Coordination Accuracy: {config.ansf_integration.target_coordination_accuracy:.1%}")
    print(f"Semantic Cache Budget: {config.ansf_integration.semantic_cache_budget_mb}MB")
    print(f"Neural Model Target Accuracy: {config.ml_model.target_accuracy:.1%}")
    print(f"Max Agents: {config.claude_flow_integration.max_agents}")
    
    # Test integration settings
    settings = get_integration_settings()
    print(f"\nIntegration Status:")
    print(f"ANSF Integration: {'✅' if settings['ansf']['enabled'] else '❌'}")
    print(f"Serena Integration: {'✅' if settings['serena']['enabled'] else '❌'}")
    print(f"Claude Flow Integration: {'✅' if settings['claude_flow']['enabled'] else '❌'}")
    
    # Test validation
    requirements = config_manager.validate_integration_requirements()
    print(f"\nRequirement Validation:")
    for req, met in requirements.items():
        print(f"{req}: {'✅' if met else '❌'}")
    
    # Test optimization profile
    test_system_state = {
        'memory_usage_percent': 75,
        'cpu_usage_percent': 60,
        'agent_count': 4
    }
    profile = config_manager.get_optimization_profile(test_system_state)
    print(f"\nRecommended Optimization Profile: {profile.upper()}")