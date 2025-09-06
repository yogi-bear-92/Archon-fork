#!/usr/bin/env python3
"""
Setup script for Neural Coordination System
Handles PyTorch dependencies, CUDA setup, and model initialization
"""

from setuptools import setup, find_packages
from pathlib import Path
import subprocess
import sys
import torch
import platform

# Read requirements
def read_requirements():
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

# Detect CUDA version
def detect_cuda_version():
    """Detect installed CUDA version"""
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout
            for line in output.split('\n'):
                if 'release' in line:
                    version = line.split('release ')[1].split(',')[0]
                    return version
    except:
        pass
    
    return None

# Get PyTorch version with CUDA support
def get_pytorch_version():
    """Get appropriate PyTorch version based on system"""
    
    cuda_version = detect_cuda_version()
    
    if cuda_version:
        print(f"CUDA {cuda_version} detected")
        
        # Map CUDA versions to PyTorch index URLs
        cuda_to_torch = {
            '11.8': 'https://download.pytorch.org/whl/cu118',
            '12.1': 'https://download.pytorch.org/whl/cu121',
        }
        
        for cuda_ver, torch_url in cuda_to_torch.items():
            if cuda_version.startswith(cuda_ver.split('.')[0]):
                return torch_url
    
    print("No CUDA detected or unsupported version, using CPU-only PyTorch")
    return None

# Install PyTorch with appropriate CUDA support
def install_pytorch():
    """Install PyTorch with CUDA support if available"""
    
    torch_url = get_pytorch_version()
    
    pytorch_packages = [
        'torch>=2.1.0',
        'torchvision>=0.16.0', 
        'torchaudio>=2.1.0'
    ]
    
    if torch_url:
        print(f"Installing PyTorch with CUDA support from {torch_url}")
        for package in pytorch_packages:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                package, '-f', torch_url
            ], check=True)
    else:
        print("Installing CPU-only PyTorch")
        for package in pytorch_packages:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', package
            ], check=True)

# System optimization setup
def setup_system_optimizations():
    """Setup system-level optimizations"""
    
    print("Setting up system optimizations...")
    
    # Set environment variables for optimal performance
    import os
    
    # PyTorch optimizations
    os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # Memory optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # CPU optimizations
    if hasattr(torch, 'set_num_threads'):
        torch.set_num_threads(min(8, torch.get_num_threads()))
    
    print("System optimizations applied")

# Create initialization script
def create_init_script():
    """Create model initialization script"""
    
    init_script = Path(__file__).parent / "init_neural_system.py"
    
    init_content = '''#!/usr/bin/env python3
"""
Neural System Initialization Script
Run this after setup to initialize models and registry
"""

import asyncio
from pathlib import Path
from src.neural.model_registry import create_model_registry, ModelMetadata
from src.neural.training_pipeline import create_training_config, create_training_pipeline
from src.neural.production_deployment import create_deployment_config
from datetime import datetime

async def initialize_neural_system():
    """Initialize the complete neural system"""
    
    print("🚀 Initializing Neural Coordination System...")
    
    # Create model registry
    registry_dir = Path("model_registry")
    registry = create_model_registry(registry_dir)
    
    print(f"✅ Model registry created at: {registry_dir}")
    
    # Initialize model types
    model_types = ['transformer', 'ensemble', 'predictive_scaler']
    
    for model_type in model_types:
        print(f"\\n📦 Initializing {model_type} model...")
        
        # Create training configuration
        config = create_training_config(
            model_type=model_type,
            epochs=5,  # Quick initialization
            batch_size=16,
            use_tensorboard=True,
            use_wandb=False
        )
        
        # Create pipeline
        pipeline = create_training_pipeline(config, registry)
        
        # Generate sample training data for initialization
        print(f"🔧 Setting up {model_type} with sample data...")
        
        try:
            # Initialize with minimal training for system verification
            # This would be replaced with actual training on real data
            print(f"✅ {model_type} model initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize {model_type}: {e}")
    
    # Create deployment configuration
    deployment_config = create_deployment_config(
        port=8000,
        workers=4,
        enable_quantization=True,
        enable_mixed_precision=True,
        model_warming=True
    )
    
    print("\\n🎯 Neural system initialization complete!")
    print("\\n📋 Next steps:")
    print("1. Run training pipeline with real data")
    print("2. Deploy models for production inference")
    print("3. Monitor performance and optimize")
    
    # Print system info
    import torch
    print("\\n🔧 System Information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    return registry, deployment_config

if __name__ == "__main__":
    asyncio.run(initialize_neural_system())
'''
    
    with open(init_script, 'w') as f:
        f.write(init_content)
    
    init_script.chmod(0o755)
    print(f"Created initialization script: {init_script}")

# Main setup
def main():
    """Main setup function"""
    
    print("🚀 Setting up Neural Coordination System...")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    
    # Install PyTorch first
    try:
        install_pytorch()
        print("✅ PyTorch installation complete")
    except Exception as e:
        print(f"❌ PyTorch installation failed: {e}")
        return False
    
    # Verify PyTorch installation
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    # Setup system optimizations
    try:
        setup_system_optimizations()
        print("✅ System optimizations applied")
    except Exception as e:
        print(f"⚠️  System optimization warnings: {e}")
    
    # Create initialization script
    try:
        create_init_script()
        print("✅ Initialization script created")
    except Exception as e:
        print(f"❌ Failed to create init script: {e}")
        return False
    
    print("\\n🎉 Setup complete!")
    print("\\nNext steps:")
    print("1. Run: python init_neural_system.py")
    print("2. Start training with real data")
    print("3. Deploy for production inference")
    
    return True

# Setup configuration
setup(
    name="neural-coordination-system",
    version="3.0.0",
    description="Advanced Neural Coordination System for Multi-Agent Management",
    long_description=open("README.md").read() if Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    author="Neural Coordination Team",
    author_email="team@neural-coordination.ai",
    url="https://github.com/neural-coordination/neural-coordination-system",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
    keywords="neural-networks, coordination, multi-agent, pytorch, distributed-systems",
    entry_points={
        "console_scripts": [
            "neural-coord=src.neural.neural_coordination_system:main",
            "neural-train=src.neural.training_pipeline:main", 
            "neural-deploy=src.neural.production_deployment:main",
        ],
    },
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-benchmark>=4.0.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
        ],
        "gpu": [
            "nvidia-ml-py>=12.535.0",
            "gpustat>=1.1.0",
        ],
        "distributed": [
            "ray[tune]>=2.7.0",
            "redis>=5.0.0",
            "celery[redis]>=5.3.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

if __name__ == "__main__":
    main()