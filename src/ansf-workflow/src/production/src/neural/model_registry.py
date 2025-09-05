#!/usr/bin/env python3
"""
Neural Model Registry - Production Model Management System
Handles model versioning, serialization, checkpoint management, and deployment
"""

import asyncio
import json
import logging
import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import hashlib
import pickle
import dill
import cloudpickle

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, Column, String, DateTime, Float, Text, Boolean, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import structlog

# Import model classes
from .coordination.transformer_coordinator import AdvancedNeuralCoordinator
from .ensemble.neural_ensemble_coordinator import AdvancedNeuralEnsemble
from .models.predictive_scaling_network import AdvancedPredictiveScaler, PredictiveScalingNetwork

logger = structlog.get_logger(__name__)
Base = declarative_base()

@dataclass
class ModelMetadata:
    """Complete model metadata"""
    model_id: str
    model_name: str
    model_type: str  # 'transformer', 'ensemble', 'predictive_scaler'
    version: str
    created_at: datetime
    updated_at: datetime
    model_size_mb: float
    parameters_count: int
    accuracy_score: float
    validation_loss: float
    training_samples: int
    config: Dict[str, Any]
    tags: List[str]
    description: str
    author: str
    is_production: bool
    checkpoint_path: str
    serialization_format: str  # 'pytorch', 'onnx', 'torchscript'

class ModelRecord(Base):
    """SQLAlchemy model for model registry database"""
    __tablename__ = 'model_registry'
    
    model_id = Column(String(255), primary_key=True)
    model_name = Column(String(255), nullable=False)
    model_type = Column(String(100), nullable=False)
    version = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    model_size_mb = Column(Float, default=0.0)
    parameters_count = Column(Integer, default=0)
    accuracy_score = Column(Float, default=0.0)
    validation_loss = Column(Float, default=0.0)
    training_samples = Column(Integer, default=0)
    config = Column(Text)  # JSON serialized
    tags = Column(Text)    # JSON serialized list
    description = Column(Text)
    author = Column(String(255))
    is_production = Column(Boolean, default=False)
    checkpoint_path = Column(String(500))
    serialization_format = Column(String(50), default='pytorch')

class ModelCheckpoint:
    """Model checkpoint management"""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer],
                       epoch: int, loss: float, metadata: Dict[str, Any]) -> Path:
        """Save model checkpoint with metadata"""
        
        checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{epoch}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pt"
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata,
            'model_class': model.__class__.__name__,
            'model_config': getattr(model, 'config', {}),
        }
        
        if optimizer:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        # Calculate model hash for integrity checking
        model_hash = self._calculate_model_hash(model.state_dict())
        checkpoint_data['model_hash'] = model_hash
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Save human-readable metadata
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump({
                'checkpoint_id': checkpoint_id,
                'epoch': epoch,
                'loss': loss,
                'timestamp': datetime.now().isoformat(),
                'model_class': model.__class__.__name__,
                'parameters_count': sum(p.numel() for p in model.parameters()),
                'model_size_mb': checkpoint_path.stat().st_size / 1024 / 1024,
                'metadata': metadata,
                'model_hash': model_hash
            }, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path: Path, model: nn.Module, 
                       optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
        """Load model checkpoint"""
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Verify model hash if available
        if 'model_hash' in checkpoint:
            expected_hash = checkpoint['model_hash']
            actual_hash = self._calculate_model_hash(checkpoint['model_state_dict'])
            if expected_hash != actual_hash:
                logger.warning(f"Model hash mismatch for {checkpoint_path}")
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state if provided
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List available checkpoints with metadata"""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.pt"):
            metadata_file = checkpoint_file.with_suffix('.json')
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    checkpoints.append({
                        'checkpoint_path': checkpoint_file,
                        'metadata_path': metadata_file,
                        **metadata
                    })
                except Exception as e:
                    logger.error(f"Error reading checkpoint metadata {metadata_file}: {e}")
        
        return sorted(checkpoints, key=lambda x: x.get('timestamp', ''), reverse=True)
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 10):
        """Remove old checkpoints, keeping the most recent ones"""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_last_n:
            return
        
        to_remove = checkpoints[keep_last_n:]
        for checkpoint in to_remove:
            try:
                checkpoint['checkpoint_path'].unlink()
                checkpoint['metadata_path'].unlink()
                logger.info(f"Removed old checkpoint: {checkpoint['checkpoint_path']}")
            except Exception as e:
                logger.error(f"Error removing checkpoint: {e}")
    
    def _calculate_model_hash(self, state_dict: Dict[str, torch.Tensor]) -> str:
        """Calculate hash of model state dict for integrity checking"""
        # Convert state dict to bytes for hashing
        model_bytes = b''
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key]
            model_bytes += key.encode() + tensor.cpu().numpy().tobytes()
        
        return hashlib.sha256(model_bytes).hexdigest()

class ModelSerializer:
    """Advanced model serialization with multiple formats"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model: nn.Module, model_id: str, 
                   format_type: str = 'pytorch') -> Dict[str, Path]:
        """Save model in specified format"""
        
        model_dir = self.models_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = {}
        
        if format_type in ['pytorch', 'all']:
            # Standard PyTorch save
            pytorch_path = model_dir / f"{model_id}.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_class': model.__class__.__name__,
                'config': getattr(model, 'config', {}),
                'timestamp': datetime.now().isoformat(),
                'pytorch_version': torch.__version__,
            }, pytorch_path)
            saved_paths['pytorch'] = pytorch_path
        
        if format_type in ['torchscript', 'all']:
            # TorchScript save
            try:
                model.eval()
                if hasattr(model, 'create_example_input'):
                    example_input = model.create_example_input()
                else:
                    # Create generic example input
                    example_input = self._create_example_input(model)
                
                traced_model = torch.jit.trace(model, example_input)
                torchscript_path = model_dir / f"{model_id}.torchscript"
                traced_model.save(str(torchscript_path))
                saved_paths['torchscript'] = torchscript_path
            except Exception as e:
                logger.warning(f"Failed to save TorchScript format: {e}")
        
        if format_type in ['onnx', 'all']:
            # ONNX export
            try:
                import onnx
                from torch.onnx import export
                
                model.eval()
                if hasattr(model, 'create_example_input'):
                    example_input = model.create_example_input()
                else:
                    example_input = self._create_example_input(model)
                
                onnx_path = model_dir / f"{model_id}.onnx"
                export(
                    model, example_input, str(onnx_path),
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                saved_paths['onnx'] = onnx_path
            except ImportError:
                logger.warning("ONNX not available for export")
            except Exception as e:
                logger.warning(f"Failed to save ONNX format: {e}")
        
        if format_type in ['pickle', 'all']:
            # Complete model pickling (for complex models)
            try:
                pickle_path = model_dir / f"{model_id}.pkl"
                with open(pickle_path, 'wb') as f:
                    cloudpickle.dump({
                        'model': model,
                        'timestamp': datetime.now().isoformat(),
                        'python_version': '.'.join(map(str, __import__('sys').version_info[:3])),
                    }, f)
                saved_paths['pickle'] = pickle_path
            except Exception as e:
                logger.warning(f"Failed to save pickle format: {e}")
        
        return saved_paths
    
    def load_model(self, model_path: Path, format_type: str = 'auto') -> Tuple[nn.Module, Dict[str, Any]]:
        """Load model from specified format"""
        
        if format_type == 'auto':
            format_type = self._detect_format(model_path)
        
        metadata = {}
        
        if format_type == 'pytorch':
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Need to reconstruct model - this requires knowing the model class
            model_class_name = checkpoint.get('model_class')
            config = checkpoint.get('config', {})
            
            # Create model instance (this is the tricky part)
            model = self._create_model_instance(model_class_name, config)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            metadata = {
                'timestamp': checkpoint.get('timestamp'),
                'pytorch_version': checkpoint.get('pytorch_version'),
                'config': config
            }
        
        elif format_type == 'torchscript':
            model = torch.jit.load(str(model_path))
            metadata = {'format': 'torchscript'}
        
        elif format_type == 'onnx':
            # ONNX runtime loading (requires onnxruntime)
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(str(model_path))
                # Return a wrapper that can be used like a PyTorch model
                model = ONNXModelWrapper(session)
                metadata = {'format': 'onnx', 'providers': ort.get_available_providers()}
            except ImportError:
                raise ImportError("onnxruntime required for ONNX model loading")
        
        elif format_type == 'pickle':
            with open(model_path, 'rb') as f:
                checkpoint = cloudpickle.load(f)
            model = checkpoint['model']
            metadata = {
                'timestamp': checkpoint.get('timestamp'),
                'python_version': checkpoint.get('python_version')
            }
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return model, metadata
    
    def _detect_format(self, model_path: Path) -> str:
        """Auto-detect model format from file extension"""
        suffix = model_path.suffix.lower()
        
        format_map = {
            '.pt': 'pytorch',
            '.pth': 'pytorch', 
            '.torchscript': 'torchscript',
            '.onnx': 'onnx',
            '.pkl': 'pickle',
            '.pickle': 'pickle'
        }
        
        return format_map.get(suffix, 'pytorch')
    
    def _create_example_input(self, model: nn.Module) -> torch.Tensor:
        """Create example input for model tracing"""
        # This is a heuristic - in practice, models should provide their own example inputs
        
        # Try to infer input shape from first layer
        first_layer = next(model.children())
        
        if isinstance(first_layer, nn.Linear):
            input_size = first_layer.in_features
            return torch.randn(1, input_size)
        elif isinstance(first_layer, nn.Conv2d):
            in_channels = first_layer.in_channels
            return torch.randn(1, in_channels, 224, 224)  # Default image size
        else:
            # Default generic input
            return torch.randn(1, 512)
    
    def _create_model_instance(self, model_class_name: str, config: Dict[str, Any]) -> nn.Module:
        """Create model instance from class name and config"""
        
        # Model creation mapping
        model_creators = {
            'AdvancedNeuralCoordinator': lambda cfg: AdvancedNeuralCoordinator(cfg),
            'AdvancedNeuralEnsemble': lambda cfg: AdvancedNeuralEnsemble(cfg.get('ensemble_size', 5)),
            'PredictiveScalingNetwork': lambda cfg: PredictiveScalingNetwork(cfg),
            'AdvancedPredictiveScaler': lambda cfg: AdvancedPredictiveScaler(cfg).network,
        }
        
        if model_class_name in model_creators:
            return model_creators[model_class_name](config)
        else:
            raise ValueError(f"Unknown model class: {model_class_name}")

class ONNXModelWrapper:
    """Wrapper to make ONNX models behave like PyTorch models"""
    
    def __init__(self, onnx_session):
        self.session = onnx_session
        self.input_names = [input.name for input in self.session.get_inputs()]
        self.output_names = [output.name for output in self.session.get_outputs()]
    
    def __call__(self, *args, **kwargs):
        # Convert PyTorch tensors to numpy
        if args:
            input_data = args[0]
            if hasattr(input_data, 'numpy'):
                input_data = input_data.numpy()
        elif 'input' in kwargs:
            input_data = kwargs['input']
            if hasattr(input_data, 'numpy'):
                input_data = input_data.numpy()
        else:
            raise ValueError("No input provided")
        
        # Run inference
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_data})
        
        # Convert back to PyTorch tensor
        return torch.from_numpy(outputs[0])
    
    def eval(self):
        return self  # ONNX models are always in eval mode

class NeuralModelRegistry:
    """Complete neural model registry system"""
    
    def __init__(self, registry_dir: Path, database_url: str = None):
        self.registry_dir = registry_dir
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.checkpoints = ModelCheckpoint(self.registry_dir / 'checkpoints')
        self.serializer = ModelSerializer(self.registry_dir / 'models')
        
        # Database setup
        if database_url is None:
            database_url = f"sqlite:///{self.registry_dir / 'models.db'}"
        
        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        logger.info(f"Model registry initialized at {self.registry_dir}")
    
    def register_model(self, model: nn.Module, metadata: ModelMetadata, 
                      save_formats: List[str] = ['pytorch']) -> str:
        """Register a new model in the registry"""
        
        model_id = metadata.model_id
        
        # Save model in requested formats
        saved_paths = {}
        for format_type in save_formats:
            paths = self.serializer.save_model(model, model_id, format_type)
            saved_paths.update(paths)
        
        # Calculate model statistics
        parameters_count = sum(p.numel() for p in model.parameters())
        model_size_mb = 0
        
        if 'pytorch' in saved_paths:
            model_size_mb = saved_paths['pytorch'].stat().st_size / 1024 / 1024
        
        # Update metadata
        metadata.parameters_count = parameters_count
        metadata.model_size_mb = model_size_mb
        metadata.checkpoint_path = str(saved_paths.get('pytorch', ''))
        metadata.serialization_format = ','.join(saved_paths.keys())
        
        # Save to database
        with self.SessionLocal() as session:
            # Check if model already exists
            existing = session.query(ModelRecord).filter_by(model_id=model_id).first()
            
            if existing:
                # Update existing record
                for key, value in asdict(metadata).items():
                    if key in ['config', 'tags']:
                        setattr(existing, key, json.dumps(value))
                    else:
                        setattr(existing, key, value)
                existing.updated_at = datetime.now()
            else:
                # Create new record
                record_data = asdict(metadata)
                record_data['config'] = json.dumps(record_data['config'])
                record_data['tags'] = json.dumps(record_data['tags'])
                
                record = ModelRecord(**record_data)
                session.add(record)
            
            session.commit()
        
        logger.info(f"Model registered: {model_id}")
        return model_id
    
    def load_model(self, model_id: str, version: str = 'latest') -> Tuple[nn.Module, ModelMetadata]:
        """Load model from registry"""
        
        with self.SessionLocal() as session:
            query = session.query(ModelRecord).filter_by(model_id=model_id)
            
            if version != 'latest':
                query = query.filter_by(version=version)
            
            record = query.order_by(ModelRecord.updated_at.desc()).first()
            
            if not record:
                raise ValueError(f"Model not found: {model_id} (version: {version})")
            
            # Load model
            checkpoint_path = Path(record.checkpoint_path)
            model, load_metadata = self.serializer.load_model(checkpoint_path)
            
            # Create metadata object
            metadata = ModelMetadata(
                model_id=record.model_id,
                model_name=record.model_name,
                model_type=record.model_type,
                version=record.version,
                created_at=record.created_at,
                updated_at=record.updated_at,
                model_size_mb=record.model_size_mb,
                parameters_count=record.parameters_count,
                accuracy_score=record.accuracy_score,
                validation_loss=record.validation_loss,
                training_samples=record.training_samples,
                config=json.loads(record.config) if record.config else {},
                tags=json.loads(record.tags) if record.tags else [],
                description=record.description or "",
                author=record.author or "",
                is_production=record.is_production,
                checkpoint_path=record.checkpoint_path,
                serialization_format=record.serialization_format
            )
            
            logger.info(f"Model loaded: {model_id} (version: {version})")
            return model, metadata
    
    def list_models(self, model_type: str = None, is_production: bool = None,
                   tags: List[str] = None) -> List[ModelMetadata]:
        """List models with filtering options"""
        
        with self.SessionLocal() as session:
            query = session.query(ModelRecord)
            
            if model_type:
                query = query.filter_by(model_type=model_type)
            
            if is_production is not None:
                query = query.filter_by(is_production=is_production)
            
            if tags:
                for tag in tags:
                    query = query.filter(ModelRecord.tags.contains(f'"{tag}"'))
            
            records = query.order_by(ModelRecord.updated_at.desc()).all()
            
            models = []
            for record in records:
                metadata = ModelMetadata(
                    model_id=record.model_id,
                    model_name=record.model_name,
                    model_type=record.model_type,
                    version=record.version,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    model_size_mb=record.model_size_mb,
                    parameters_count=record.parameters_count,
                    accuracy_score=record.accuracy_score,
                    validation_loss=record.validation_loss,
                    training_samples=record.training_samples,
                    config=json.loads(record.config) if record.config else {},
                    tags=json.loads(record.tags) if record.tags else [],
                    description=record.description or "",
                    author=record.author or "",
                    is_production=record.is_production,
                    checkpoint_path=record.checkpoint_path,
                    serialization_format=record.serialization_format
                )
                models.append(metadata)
            
            return models
    
    def promote_to_production(self, model_id: str, version: str = 'latest') -> bool:
        """Promote model to production status"""
        
        with self.SessionLocal() as session:
            query = session.query(ModelRecord).filter_by(model_id=model_id)
            
            if version != 'latest':
                query = query.filter_by(version=version)
            
            record = query.order_by(ModelRecord.updated_at.desc()).first()
            
            if not record:
                return False
            
            # Demote other production models of the same type
            session.query(ModelRecord).filter_by(
                model_type=record.model_type,
                is_production=True
            ).update({'is_production': False})
            
            # Promote this model
            record.is_production = True
            record.updated_at = datetime.now()
            
            session.commit()
            
            logger.info(f"Model promoted to production: {model_id} (version: {version})")
            return True
    
    def delete_model(self, model_id: str, version: str = None) -> bool:
        """Delete model from registry"""
        
        with self.SessionLocal() as session:
            query = session.query(ModelRecord).filter_by(model_id=model_id)
            
            if version:
                query = query.filter_by(version=version)
            
            records = query.all()
            
            if not records:
                return False
            
            # Delete files
            for record in records:
                try:
                    model_dir = self.registry_dir / 'models' / record.model_id
                    if model_dir.exists():
                        shutil.rmtree(model_dir)
                    
                    if record.checkpoint_path:
                        checkpoint_path = Path(record.checkpoint_path)
                        if checkpoint_path.exists():
                            checkpoint_path.unlink()
                except Exception as e:
                    logger.error(f"Error deleting model files: {e}")
                
                session.delete(record)
            
            session.commit()
            
            logger.info(f"Model deleted: {model_id}")
            return True
    
    def create_model_version(self, model_id: str, new_version: str, 
                           source_version: str = 'latest') -> bool:
        """Create new version of existing model"""
        
        with self.SessionLocal() as session:
            # Get source model
            query = session.query(ModelRecord).filter_by(model_id=model_id)
            
            if source_version != 'latest':
                query = query.filter_by(version=source_version)
            
            source_record = query.order_by(ModelRecord.updated_at.desc()).first()
            
            if not source_record:
                return False
            
            # Create new version
            new_record = ModelRecord(
                model_id=model_id,
                model_name=source_record.model_name,
                model_type=source_record.model_type,
                version=new_version,
                model_size_mb=source_record.model_size_mb,
                parameters_count=source_record.parameters_count,
                accuracy_score=0.0,  # Reset metrics for new version
                validation_loss=0.0,
                training_samples=0,
                config=source_record.config,
                tags=source_record.tags,
                description=source_record.description,
                author=source_record.author,
                is_production=False,
                checkpoint_path="",  # Will be set when model is saved
                serialization_format=source_record.serialization_format
            )
            
            session.add(new_record)
            session.commit()
            
            logger.info(f"New model version created: {model_id} v{new_version}")
            return True
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        
        with self.SessionLocal() as session:
            total_models = session.query(ModelRecord).count()
            production_models = session.query(ModelRecord).filter_by(is_production=True).count()
            
            # Count by type
            type_counts = {}
            for record in session.query(ModelRecord.model_type, ModelRecord.model_id).distinct():
                model_type = record.model_type
                type_counts[model_type] = type_counts.get(model_type, 0) + 1
            
            # Total storage used
            total_size = sum(record.model_size_mb for record in session.query(ModelRecord))
            
            return {
                'total_models': total_models,
                'production_models': production_models,
                'models_by_type': type_counts,
                'total_storage_mb': total_size,
                'registry_path': str(self.registry_dir)
            }
    
    def cleanup_old_versions(self, keep_versions_per_model: int = 5):
        """Clean up old model versions"""
        
        with self.SessionLocal() as session:
            # Group by model_id
            model_ids = session.query(ModelRecord.model_id).distinct()
            
            for (model_id,) in model_ids:
                records = session.query(ModelRecord).filter_by(
                    model_id=model_id
                ).order_by(ModelRecord.updated_at.desc()).all()
                
                if len(records) > keep_versions_per_model:
                    to_delete = records[keep_versions_per_model:]
                    
                    for record in to_delete:
                        if not record.is_production:  # Don't delete production models
                            try:
                                # Delete files
                                if record.checkpoint_path:
                                    Path(record.checkpoint_path).unlink()
                            except Exception as e:
                                logger.error(f"Error deleting old version files: {e}")
                            
                            session.delete(record)
                    
                    session.commit()
                    logger.info(f"Cleaned up old versions for model: {model_id}")

# Factory function
def create_model_registry(registry_dir: Union[str, Path], 
                         database_url: str = None) -> NeuralModelRegistry:
    """Create and initialize model registry"""
    return NeuralModelRegistry(Path(registry_dir), database_url)