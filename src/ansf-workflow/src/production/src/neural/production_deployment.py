#!/usr/bin/env python3
"""
Neural Models Production Deployment System
Handles model serving, GPU optimization, monitoring, and distributed inference
"""

import asyncio
import logging
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import threading

import torch
import torch.nn as nn
from torch.jit import ScriptModule
import numpy as np
import psutil
import GPUtil
from prometheus_client import Counter, Histogram, Gauge, start_http_server

import structlog
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import redis
from celery import Celery

# Import neural components
from .model_registry import NeuralModelRegistry, ModelMetadata
from .coordination.transformer_coordinator import AdvancedNeuralCoordinator, AgentState
from .ensemble.neural_ensemble_coordinator import AdvancedNeuralEnsemble
from .models.predictive_scaling_network import AdvancedPredictiveScaler, WorkloadMetrics

logger = structlog.get_logger(__name__)

# Prometheus metrics
INFERENCE_REQUESTS = Counter('neural_inference_requests_total', 'Total inference requests', ['model_type', 'status'])
INFERENCE_LATENCY = Histogram('neural_inference_duration_seconds', 'Inference latency', ['model_type'])
GPU_MEMORY_USAGE = Gauge('gpu_memory_usage_bytes', 'GPU memory usage', ['device_id'])
MODEL_ACCURACY = Gauge('model_accuracy_score', 'Model accuracy score', ['model_id', 'model_type'])
QUEUE_SIZE = Gauge('inference_queue_size', 'Size of inference queue', ['queue_type'])

@dataclass
class DeploymentConfig:
    """Configuration for production deployment"""
    
    # Server configuration
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    max_requests: int = 1000
    timeout: int = 300
    
    # Model serving
    model_batch_size: int = 16
    max_sequence_length: int = 512
    enable_quantization: bool = True
    enable_tensorrt: bool = False
    
    # GPU optimization
    gpu_devices: List[int] = None
    memory_fraction: float = 0.8
    enable_mixed_precision: bool = True
    
    # Caching
    cache_enabled: bool = True
    cache_ttl: int = 3600
    redis_url: str = "redis://localhost:6379"
    
    # Monitoring
    metrics_port: int = 8001
    health_check_interval: int = 30
    
    # Load balancing
    enable_load_balancing: bool = True
    max_queue_size: int = 1000
    
    # Model management
    model_warming: bool = True
    auto_scaling: bool = True
    fallback_models: Dict[str, str] = None

class InferenceRequest(BaseModel):
    """Request model for inference"""
    model_type: str
    input_data: Dict[str, Any]
    model_version: Optional[str] = "latest"
    priority: int = 1
    timeout: Optional[float] = 30.0
    return_confidence: bool = False
    return_explanations: bool = False

class InferenceResponse(BaseModel):
    """Response model for inference"""
    request_id: str
    model_id: str
    predictions: Dict[str, Any]
    confidence: Optional[float] = None
    explanations: Optional[Dict[str, Any]] = None
    inference_time: float
    queue_time: float
    timestamp: str
    status: str = "success"

class GPUManager:
    """GPU resource management and optimization"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.device_pools = {}
        self.memory_usage = {}
        
        # Initialize GPU devices
        if torch.cuda.is_available():
            self.setup_gpu_devices()
        else:
            logger.warning("CUDA not available, using CPU only")
    
    def setup_gpu_devices(self):
        """Setup and optimize GPU devices"""
        
        available_gpus = list(range(torch.cuda.device_count()))
        
        if self.config.gpu_devices:
            gpu_devices = [gpu for gpu in self.config.gpu_devices if gpu in available_gpus]
        else:
            gpu_devices = available_gpus
        
        logger.info(f"Initializing GPUs: {gpu_devices}")
        
        for gpu_id in gpu_devices:
            torch.cuda.set_device(gpu_id)
            
            # Set memory fraction
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(self.config.memory_fraction, gpu_id)
            
            # Initialize memory pool
            torch.cuda.empty_cache()
            
            self.device_pools[gpu_id] = {
                'device': torch.device(f'cuda:{gpu_id}'),
                'models': {},
                'load': 0,
                'memory_used': 0
            }
            
            logger.info(f"GPU {gpu_id} initialized with {self.config.memory_fraction:.1%} memory allocation")
    
    def get_optimal_device(self, model_type: str) -> torch.device:
        """Get optimal GPU device for model inference"""
        
        if not self.device_pools:
            return torch.device('cpu')
        
        # Find device with lowest load
        best_device = min(
            self.device_pools.keys(),
            key=lambda gpu_id: self.device_pools[gpu_id]['load']
        )
        
        return self.device_pools[best_device]['device']
    
    def update_device_metrics(self):
        """Update GPU device metrics"""
        
        for gpu_id in self.device_pools:
            try:
                # Get GPU stats
                gpu_stats = GPUtil.getGPUs()[gpu_id]
                
                memory_used = gpu_stats.memoryUsed * 1024 * 1024  # Convert to bytes
                GPU_MEMORY_USAGE.labels(device_id=gpu_id).set(memory_used)
                
                self.device_pools[gpu_id]['memory_used'] = memory_used
                
            except Exception as e:
                logger.error(f"Error updating GPU {gpu_id} metrics: {e}")
    
    def optimize_model_for_gpu(self, model: nn.Module, device: torch.device) -> nn.Module:
        """Optimize model for GPU inference"""
        
        model = model.to(device)
        model.eval()
        
        # Enable mixed precision if configured
        if self.config.enable_mixed_precision and device.type == 'cuda':
            # Convert model to half precision for compatible layers
            model = model.half()
        
        # Quantization
        if self.config.enable_quantization:
            try:
                # Dynamic quantization
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
                logger.info(f"Model quantized for device {device}")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")
        
        # TensorRT optimization (if available)
        if self.config.enable_tensorrt and device.type == 'cuda':
            try:
                import torch_tensorrt
                
                # Compile with TensorRT
                example_input = torch.randn(1, 512).to(device)
                if self.config.enable_mixed_precision:
                    example_input = example_input.half()
                
                model = torch_tensorrt.compile(
                    model,
                    inputs=[torch_tensorrt.Input(example_input.shape, dtype=example_input.dtype)],
                    enabled_precisions={torch.float16} if self.config.enable_mixed_precision else {torch.float32}
                )
                logger.info(f"TensorRT optimization applied to device {device}")
                
            except ImportError:
                logger.warning("TensorRT not available")
            except Exception as e:
                logger.warning(f"TensorRT optimization failed: {e}")
        
        return model

class ModelCache:
    """Model caching and preloading system"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.cache = {}
        self.access_times = {}
        self.cache_lock = threading.RLock()
        
        # Setup Redis if configured
        if config.cache_enabled and config.redis_url:
            try:
                self.redis_client = redis.from_url(config.redis_url, decode_responses=True)
                logger.info("Redis cache connected")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        else:
            self.redis_client = None
    
    def get_model(self, model_id: str) -> Optional[Tuple[nn.Module, Dict[str, Any]]]:
        """Get model from cache"""
        
        with self.cache_lock:
            if model_id in self.cache:
                self.access_times[model_id] = time.time()
                return self.cache[model_id]
        
        return None
    
    def cache_model(self, model_id: str, model: nn.Module, metadata: Dict[str, Any]):
        """Cache model with metadata"""
        
        with self.cache_lock:
            self.cache[model_id] = (model, metadata)
            self.access_times[model_id] = time.time()
            
            # Cleanup old entries if cache is full
            self.cleanup_cache()
    
    def cleanup_cache(self, max_entries: int = 10):
        """Remove least recently used models from cache"""
        
        if len(self.cache) <= max_entries:
            return
        
        # Sort by access time and remove oldest
        sorted_models = sorted(self.access_times.items(), key=lambda x: x[1])
        
        models_to_remove = sorted_models[:-max_entries]
        
        for model_id, _ in models_to_remove:
            del self.cache[model_id]
            del self.access_times[model_id]
            logger.info(f"Removed {model_id} from cache")
    
    def preload_models(self, model_ids: List[str], registry: NeuralModelRegistry):
        """Preload models into cache"""
        
        for model_id in model_ids:
            try:
                model, metadata = registry.load_model(model_id)
                self.cache_model(model_id, model, asdict(metadata))
                logger.info(f"Preloaded model: {model_id}")
            except Exception as e:
                logger.error(f"Failed to preload model {model_id}: {e}")

class InferenceQueue:
    """Async inference queue with priority and load balancing"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.queues = {
            'high': asyncio.Queue(maxsize=config.max_queue_size // 3),
            'medium': asyncio.Queue(maxsize=config.max_queue_size // 3),
            'low': asyncio.Queue(maxsize=config.max_queue_size // 3)
        }
        self.processing_lock = asyncio.Lock()
        
    async def enqueue(self, request: InferenceRequest) -> str:
        """Add request to appropriate priority queue"""
        
        request_id = f"req_{int(time.time() * 1000)}_{id(request)}"
        
        # Determine priority queue
        priority_map = {1: 'low', 2: 'medium', 3: 'high'}
        queue_name = priority_map.get(request.priority, 'medium')
        
        try:
            await self.queues[queue_name].put((request_id, request, time.time()))
            QUEUE_SIZE.labels(queue_type=queue_name).set(self.queues[queue_name].qsize())
            return request_id
        except asyncio.QueueFull:
            raise HTTPException(status_code=503, detail="Inference queue full")
    
    async def dequeue(self) -> Optional[Tuple[str, InferenceRequest, float]]:
        """Get next request from queues (high priority first)"""
        
        async with self.processing_lock:
            # Try high priority first, then medium, then low
            for queue_name in ['high', 'medium', 'low']:
                try:
                    item = self.queues[queue_name].get_nowait()
                    QUEUE_SIZE.labels(queue_type=queue_name).set(self.queues[queue_name].qsize())
                    return item
                except asyncio.QueueEmpty:
                    continue
        
        return None

class ModelInferenceEngine:
    """High-performance model inference engine"""
    
    def __init__(self, config: DeploymentConfig, registry: NeuralModelRegistry):
        self.config = config
        self.registry = registry
        
        # Initialize components
        self.gpu_manager = GPUManager(config)
        self.model_cache = ModelCache(config)
        self.inference_queue = InferenceQueue(config)
        
        # Model instances
        self.loaded_models = {}
        
        # Worker pool
        self.executor = ThreadPoolExecutor(max_workers=config.workers)
        
        # Background tasks
        self.background_tasks = []
        
    async def initialize(self):
        """Initialize the inference engine"""
        
        logger.info("Initializing inference engine...")
        
        # Load production models
        await self.load_production_models()
        
        # Start background tasks
        self.start_background_tasks()
        
        # Warm up models if configured
        if self.config.model_warming:
            await self.warm_up_models()
        
        logger.info("Inference engine initialized successfully")
    
    async def load_production_models(self):
        """Load all production models into memory"""
        
        production_models = self.registry.list_models(is_production=True)
        
        for metadata in production_models:
            try:
                model, _ = self.registry.load_model(metadata.model_id)
                
                # Optimize model for GPU
                device = self.gpu_manager.get_optimal_device(metadata.model_type)
                optimized_model = self.gpu_manager.optimize_model_for_gpu(model, device)
                
                # Cache model
                self.model_cache.cache_model(
                    metadata.model_id,
                    optimized_model,
                    asdict(metadata)
                )
                
                # Track in loaded models
                self.loaded_models[metadata.model_id] = {
                    'model': optimized_model,
                    'metadata': metadata,
                    'device': device,
                    'load_time': time.time()
                }
                
                MODEL_ACCURACY.labels(
                    model_id=metadata.model_id,
                    model_type=metadata.model_type
                ).set(metadata.accuracy_score)
                
                logger.info(f"Loaded production model: {metadata.model_id} on {device}")
                
            except Exception as e:
                logger.error(f"Failed to load model {metadata.model_id}: {e}")
    
    async def warm_up_models(self):
        """Warm up models with sample inputs"""
        
        logger.info("Warming up models...")
        
        for model_id, model_info in self.loaded_models.items():
            try:
                model = model_info['model']
                device = model_info['device']
                model_type = model_info['metadata'].model_type
                
                # Create sample input based on model type
                sample_input = self.create_sample_input(model_type, device)
                
                # Run inference to warm up
                with torch.no_grad():
                    _ = model(sample_input)
                
                logger.info(f"Warmed up model: {model_id}")
                
            except Exception as e:
                logger.warning(f"Failed to warm up model {model_id}: {e}")
    
    def create_sample_input(self, model_type: str, device: torch.device) -> torch.Tensor:
        """Create sample input for model warm-up"""
        
        if model_type == 'transformer':
            return torch.randn(1, 512, device=device)
        elif model_type == 'ensemble':
            return torch.randn(1, 256, device=device)
        elif model_type == 'predictive_scaler':
            return torch.randn(1, 256, device=device)
        else:
            return torch.randn(1, 512, device=device)
    
    async def process_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Process inference request"""
        
        start_time = time.time()
        request_id = f"req_{int(start_time * 1000)}_{hash(str(request))}"
        
        try:
            # Get model
            model_info = await self.get_model_for_inference(request.model_type, request.model_version)
            
            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model not found: {request.model_type}")
            
            model = model_info['model']
            metadata = model_info['metadata']
            device = model_info['device']
            
            # Prepare input
            input_tensor = self.prepare_input(request.input_data, request.model_type, device)
            
            # Run inference
            queue_time = time.time() - start_time
            inference_start = time.time()
            
            with torch.no_grad():
                if self.config.enable_mixed_precision and device.type == 'cuda':
                    with torch.cuda.amp.autocast():
                        predictions = model(input_tensor)
                else:
                    predictions = model(input_tensor)
            
            inference_time = time.time() - inference_start
            
            # Process outputs
            processed_predictions = self.process_predictions(predictions, request.model_type)
            
            # Calculate confidence if requested
            confidence = None
            if request.return_confidence:
                confidence = self.calculate_confidence(predictions, request.model_type)
            
            # Generate explanations if requested
            explanations = None
            if request.return_explanations:
                explanations = self.generate_explanations(
                    input_tensor, predictions, model, request.model_type
                )
            
            # Update metrics
            INFERENCE_REQUESTS.labels(model_type=request.model_type, status='success').inc()
            INFERENCE_LATENCY.labels(model_type=request.model_type).observe(inference_time)
            
            response = InferenceResponse(
                request_id=request_id,
                model_id=metadata.model_id,
                predictions=processed_predictions,
                confidence=confidence,
                explanations=explanations,
                inference_time=inference_time,
                queue_time=queue_time,
                timestamp=datetime.now().isoformat(),
                status="success"
            )
            
            return response
            
        except Exception as e:
            INFERENCE_REQUESTS.labels(model_type=request.model_type, status='error').inc()
            logger.error(f"Inference error for request {request_id}: {e}")
            
            return InferenceResponse(
                request_id=request_id,
                model_id="unknown",
                predictions={},
                inference_time=time.time() - start_time,
                queue_time=0.0,
                timestamp=datetime.now().isoformat(),
                status="error"
            )
    
    async def get_model_for_inference(self, model_type: str, version: str = "latest") -> Optional[Dict[str, Any]]:
        """Get model for inference with fallback support"""
        
        # Try to find exact model
        for model_id, model_info in self.loaded_models.items():
            metadata = model_info['metadata']
            if metadata.model_type == model_type and (version == "latest" or metadata.version == version):
                return model_info
        
        # Try fallback models
        if self.config.fallback_models and model_type in self.config.fallback_models:
            fallback_id = self.config.fallback_models[model_type]
            if fallback_id in self.loaded_models:
                return self.loaded_models[fallback_id]
        
        # Try to load on-demand
        try:
            production_models = self.registry.list_models(model_type=model_type, is_production=True)
            if production_models:
                metadata = production_models[0]  # Get first production model
                model, _ = self.registry.load_model(metadata.model_id)
                
                device = self.gpu_manager.get_optimal_device(model_type)
                optimized_model = self.gpu_manager.optimize_model_for_gpu(model, device)
                
                model_info = {
                    'model': optimized_model,
                    'metadata': metadata,
                    'device': device,
                    'load_time': time.time()
                }
                
                self.loaded_models[metadata.model_id] = model_info
                return model_info
        except Exception as e:
            logger.error(f"Failed to load model on-demand: {e}")
        
        return None
    
    def prepare_input(self, input_data: Dict[str, Any], model_type: str, device: torch.device) -> torch.Tensor:
        """Prepare input tensor from request data"""
        
        if model_type == 'transformer':
            # Handle coordination input
            if 'agent_states' in input_data:
                # Convert agent states to tensor
                features = []
                for agent_state in input_data['agent_states']:
                    state_features = [
                        agent_state.get('performance_score', 0.8),
                        agent_state.get('resource_utilization', 0.5),
                        agent_state.get('coordination_weight', 1.0),
                        agent_state.get('memory_usage', 500.0) / 1000.0
                    ]
                    features.extend(state_features)
                
                # Pad to expected size
                while len(features) < 512:
                    features.append(0.0)
                
                return torch.tensor(features[:512], dtype=torch.float32, device=device).unsqueeze(0)
            
            elif 'features' in input_data:
                features = input_data['features']
                return torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        
        elif model_type == 'predictive_scaler':
            # Handle workload metrics input
            if 'workload_metrics' in input_data:
                metrics = input_data['workload_metrics']
                features = self.extract_workload_features(metrics)
                return torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Default: convert input_data to tensor
        if 'input_tensor' in input_data:
            return torch.tensor(input_data['input_tensor'], dtype=torch.float32, device=device)
        elif 'features' in input_data:
            return torch.tensor(input_data['features'], dtype=torch.float32, device=device).unsqueeze(0)
        else:
            # Generic conversion
            values = list(input_data.values())
            if all(isinstance(v, (int, float)) for v in values):
                return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0)
        
        raise ValueError(f"Unable to prepare input for model type: {model_type}")
    
    def extract_workload_features(self, metrics: Dict[str, Any]) -> List[float]:
        """Extract features from workload metrics"""
        
        features = []
        
        # Basic metrics
        features.extend([
            metrics.get('task_complexity', 0.5),
            float(metrics.get('concurrent_tasks', 5)) / 50.0,
            metrics.get('error_rate', 0.01),
            metrics.get('throughput', 100.0) / 1000.0,
            float(metrics.get('queue_depth', 10)) / 100.0
        ])
        
        # Resource utilization
        resource_util = metrics.get('resource_utilization', {})
        features.extend([
            resource_util.get('cpu', 0.5),
            resource_util.get('memory', 0.6),
            resource_util.get('network', 0.3),
            resource_util.get('disk', 0.2)
        ])
        
        # Performance history
        perf_history = metrics.get('performance_history', [0.8] * 10)
        if len(perf_history) < 10:
            perf_history.extend([0.8] * (10 - len(perf_history)))
        features.extend(perf_history[:10])
        
        # Pad to 256 features
        while len(features) < 256:
            features.append(0.0)
        
        return features[:256]
    
    def process_predictions(self, predictions: torch.Tensor, model_type: str) -> Dict[str, Any]:
        """Process model predictions into response format"""
        
        predictions_cpu = predictions.cpu().numpy()
        
        if model_type == 'transformer':
            return {
                'coordination_success_probability': float(predictions_cpu[0, 0]) if predictions_cpu.size > 0 else 0.8,
                'estimated_accuracy_improvement': float(predictions_cpu[0, 1]) if predictions_cpu.size > 1 else 0.1,
                'coordination_quality_score': float(predictions_cpu[0, 2]) if predictions_cpu.size > 2 else 0.85
            }
        
        elif model_type == 'ensemble':
            return {
                'ensemble_prediction': predictions_cpu.tolist(),
                'consensus_score': float(np.std(predictions_cpu)) if predictions_cpu.size > 0 else 0.1,
                'prediction_confidence': float(np.mean(predictions_cpu)) if predictions_cpu.size > 0 else 0.8
            }
        
        elif model_type == 'predictive_scaler':
            if isinstance(predictions, dict):
                return {
                    key: value.cpu().numpy().tolist() if isinstance(value, torch.Tensor) else value
                    for key, value in predictions.items()
                }
            else:
                return {
                    'optimal_agent_count': int(predictions_cpu[0] * 32) if predictions_cpu.size > 0 else 4,
                    'scaling_confidence': float(predictions_cpu[1]) if predictions_cpu.size > 1 else 0.8,
                    'resource_efficiency': float(predictions_cpu[2]) if predictions_cpu.size > 2 else 0.7
                }
        
        else:
            return {'predictions': predictions_cpu.tolist()}
    
    def calculate_confidence(self, predictions: torch.Tensor, model_type: str) -> float:
        """Calculate prediction confidence"""
        
        predictions_cpu = predictions.cpu().numpy()
        
        if model_type in ['transformer', 'ensemble']:
            # Use standard deviation as confidence measure (lower = more confident)
            if predictions_cpu.size > 1:
                return max(0.0, 1.0 - float(np.std(predictions_cpu)))
            else:
                return 0.8
        
        elif model_type == 'predictive_scaler':
            # Use prediction magnitude as confidence
            return min(1.0, float(np.mean(np.abs(predictions_cpu))))
        
        else:
            return 0.8
    
    def generate_explanations(self, input_tensor: torch.Tensor, predictions: torch.Tensor,
                            model: nn.Module, model_type: str) -> Dict[str, Any]:
        """Generate model explanations (simplified implementation)"""
        
        try:
            # Basic feature importance using gradients
            input_tensor.requires_grad_(True)
            
            if isinstance(predictions, dict):
                # For multi-output models, use first output
                output = next(iter(predictions.values()))
            else:
                output = predictions
            
            # Calculate gradients
            if output.dim() > 1:
                output = output.sum()
            
            output.backward()
            
            gradients = input_tensor.grad.cpu().numpy()
            feature_importance = np.abs(gradients[0])  # First sample
            
            # Normalize
            if feature_importance.max() > 0:
                feature_importance = feature_importance / feature_importance.max()
            
            top_features = np.argsort(feature_importance)[-10:][::-1]  # Top 10 features
            
            return {
                'feature_importance': feature_importance.tolist(),
                'top_important_features': top_features.tolist(),
                'explanation_method': 'gradient_based',
                'model_type': model_type
            }
            
        except Exception as e:
            logger.warning(f"Failed to generate explanations: {e}")
            return {'error': 'explanation_generation_failed'}
    
    def start_background_tasks(self):
        """Start background monitoring and maintenance tasks"""
        
        # GPU monitoring task
        async def gpu_monitoring_task():
            while True:
                try:
                    self.gpu_manager.update_device_metrics()
                    await asyncio.sleep(self.config.health_check_interval)
                except Exception as e:
                    logger.error(f"GPU monitoring error: {e}")
                    await asyncio.sleep(60)
        
        # Model health check task
        async def model_health_check():
            while True:
                try:
                    for model_id, model_info in self.loaded_models.items():
                        # Simple health check with dummy input
                        try:
                            device = model_info['device']
                            model_type = model_info['metadata'].model_type
                            sample_input = self.create_sample_input(model_type, device)
                            
                            with torch.no_grad():
                                _ = model_info['model'](sample_input)
                            
                        except Exception as e:
                            logger.error(f"Model health check failed for {model_id}: {e}")
                    
                    await asyncio.sleep(self.config.health_check_interval * 2)
                except Exception as e:
                    logger.error(f"Model health check error: {e}")
                    await asyncio.sleep(60)
        
        # Start tasks
        self.background_tasks.extend([
            asyncio.create_task(gpu_monitoring_task()),
            asyncio.create_task(model_health_check())
        ])
    
    async def shutdown(self):
        """Graceful shutdown of the inference engine"""
        
        logger.info("Shutting down inference engine...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Inference engine shut down complete")

# FastAPI application
def create_inference_app(config: DeploymentConfig, registry: NeuralModelRegistry) -> FastAPI:
    """Create FastAPI application for neural inference"""
    
    app = FastAPI(
        title="Neural Models Inference API",
        description="Production inference API for neural coordination models",
        version="1.0.0"
    )
    
    # Initialize inference engine
    inference_engine = ModelInferenceEngine(config, registry)
    
    @app.on_event("startup")
    async def startup_event():
        await inference_engine.initialize()
        
        # Start Prometheus metrics server
        start_http_server(config.metrics_port)
        logger.info(f"Metrics server started on port {config.metrics_port}")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        await inference_engine.shutdown()
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "loaded_models": len(inference_engine.loaded_models),
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    @app.get("/models")
    async def list_models():
        """List available models"""
        models_info = []
        for model_id, model_info in inference_engine.loaded_models.items():
            metadata = model_info['metadata']
            models_info.append({
                "model_id": model_id,
                "model_type": metadata.model_type,
                "version": metadata.version,
                "accuracy_score": metadata.accuracy_score,
                "is_production": metadata.is_production,
                "device": str(model_info['device']),
                "load_time": model_info['load_time']
            })
        
        return {"models": models_info}
    
    @app.post("/predict", response_model=InferenceResponse)
    async def predict(request: InferenceRequest, background_tasks: BackgroundTasks):
        """Main inference endpoint"""
        
        # Add to queue
        request_id = await inference_engine.inference_queue.enqueue(request)
        
        # Process inference
        response = await inference_engine.process_inference(request)
        
        return response
    
    @app.post("/predict/batch")
    async def batch_predict(requests: List[InferenceRequest]):
        """Batch inference endpoint"""
        
        tasks = []
        for request in requests:
            task = asyncio.create_task(inference_engine.process_inference(request))
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {"responses": responses}
    
    @app.get("/metrics")
    async def get_metrics():
        """Get inference metrics"""
        
        return {
            "inference_requests": INFERENCE_REQUESTS._value._value,
            "average_latency": INFERENCE_LATENCY._sum._value / max(INFERENCE_LATENCY._count._value, 1),
            "queue_sizes": {
                queue_name: queue.qsize()
                for queue_name, queue in inference_engine.inference_queue.queues.items()
            },
            "gpu_memory": {
                f"gpu_{gpu_id}": info['memory_used']
                for gpu_id, info in inference_engine.gpu_manager.device_pools.items()
            }
        }
    
    return app

# Factory function
def create_deployment_config(**kwargs) -> DeploymentConfig:
    """Create deployment configuration with overrides"""
    return DeploymentConfig(**kwargs)

# Main deployment function
async def deploy_neural_models(registry_dir: Path, config: DeploymentConfig = None):
    """Deploy neural models for production inference"""
    
    if config is None:
        config = DeploymentConfig()
    
    # Initialize model registry
    registry = NeuralModelRegistry(registry_dir)
    
    # Create FastAPI app
    app = create_inference_app(config, registry)
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        workers=1,  # Use 1 worker to avoid model loading conflicts
        log_level="info"
    )

if __name__ == "__main__":
    import sys
    
    registry_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("model_registry")
    
    # Run deployment
    asyncio.run(deploy_neural_models(registry_dir))