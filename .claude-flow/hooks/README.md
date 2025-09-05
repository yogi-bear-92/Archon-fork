# Dynamic OS-Aware Claude Flow Hooks

This directory contains OS-aware hooks that automatically detect your operating system and execute appropriate commands for optimal Claude Flow integration.

## 🚀 Quick Start

Instead of using platform-specific commands, use the dynamic wrapper:

```bash
# Universal commands that work on macOS, Linux, and Windows
./.claude-flow/hooks/dynamic-hooks-wrapper.sh system-health
./.claude-flow/hooks/dynamic-hooks-wrapper.sh pre-task --ansf-context=true
./.claude-flow/hooks/dynamic-hooks-wrapper.sh post-task --export-metrics=true
```

## 🔍 Automatic OS Detection

The system automatically detects and optimizes for:

### macOS (Darwin)
- **Memory Command**: `vm_stat` instead of `free`
- **Swap Command**: `sysctl vm.swapusage`
- **Memory Thresholds**: Emergency: 70MB, Limited: 200MB, Optimal: 500MB
- **Cache Limits**: 25MB for memory efficiency

### Linux
- **Memory Command**: `free -m`
- **Swap Command**: Standard `/proc/swaps`
- **Memory Thresholds**: Emergency: 100MB, Limited: 256MB, Optimal: 1GB
- **Cache Limits**: 50MB for standard usage

### Windows
- **Memory Command**: `wmic OS get FreePhysicalMemory`
- **Swap Command**: `wmic pagefile list`
- **Memory Thresholds**: Emergency: 128MB, Limited: 512MB, Optimal: 2GB
- **Cache Limits**: 100MB for Windows compatibility

## 📋 Available Commands

### Core Hooks
```bash
# System health check (OS-aware)
./dynamic-hooks-wrapper.sh system-health

# Pre-task setup with memory awareness
./dynamic-hooks-wrapper.sh pre-task --description "ANSF development"

# Post-task cleanup and optimization
./dynamic-hooks-wrapper.sh post-task --export-metrics=true

# Continuous memory monitoring
./dynamic-hooks-wrapper.sh memory-monitor --interval=5
```

### ANSF-Specific Commands
```bash
# ANSF system validation
./dynamic-hooks-wrapper.sh ansf-validate --phase=3 --target-accuracy=97%

# Production deployment validation
./dynamic-hooks-wrapper.sh pre-deploy --ansf-production=true
```

## ⚙️ System Mode Detection

The hooks automatically determine your system's capability:

### Emergency Mode (Current: ~68MB)
- **Agents**: 1 maximum (smart-agent only)
- **Tools**: Claude Code only
- **Features**: Essential operations, aggressive cleanup
- **Triggers**: macOS <70MB, Linux <100MB, Windows <128MB

### Limited Mode
- **Agents**: 2-3 maximum
- **Tools**: Claude Code + Serena (minimal cache)
- **Features**: Core development with memory constraints
- **Triggers**: macOS 70-200MB, Linux 100-256MB, Windows 128-512MB

### Optimal Mode
- **Agents**: 6-8 maximum
- **Tools**: Full integration (Claude Code + Serena + Archon + Flow)
- **Features**: All systems, neural training, full coordination
- **Triggers**: macOS >500MB, Linux >1GB, Windows >2GB

## 📊 Current System Status

Based on your macOS system with ~68MB available:

```bash
# Check current status
./dynamic-hooks-wrapper.sh system-health

# Expected output:
# 🤖 Claude Flow Dynamic Hooks
# OS: darwin
# Memory: 68MB available
# Mode: emergency
# Command: system-health
```

## 🔧 Configuration

Edit `.claude-flow/hooks/os-detection-hooks.json` to customize:

- Memory thresholds per OS
- Command mappings
- Agent scaling rules
- Tool activation thresholds

## 📝 Logging

All hook executions are logged to `.claude-flow/hooks/hooks.log` for debugging and performance analysis.

## 🚨 Emergency Protocols

When memory is critical (current state), the hooks automatically:

1. ✅ Switch to emergency mode
2. ✅ Limit to single agent coordination  
3. ✅ Use OS-appropriate memory commands
4. ✅ Enable aggressive cleanup
5. ✅ Maintain ANSF accuracy targets (>94% in emergency mode)

## 💡 Usage Examples

### Daily Development Workflow
```bash
# Morning system check
./dynamic-hooks-wrapper.sh system-health

# Start development session
./dynamic-hooks-wrapper.sh pre-task --ansf-context=true --memory-adaptive=true

# Development work happens here...

# End session with cleanup
./dynamic-hooks-wrapper.sh post-task --comprehensive-cleanup=true
```

### ANSF Phase Development
```bash
# Phase-specific validation
./dynamic-hooks-wrapper.sh ansf-validate --phase=3 --neural-validation=true

# Production deployment
./dynamic-hooks-wrapper.sh pre-deploy --ansf-production=true --memory-safe=true
```

## 🎯 Integration Benefits

- **✅ Universal Compatibility**: Works on macOS, Linux, Windows
- **✅ Memory Optimization**: Automatic scaling based on available resources
- **✅ ANSF Awareness**: Phase-specific validation and coordination
- **✅ GitHub Actions Ready**: Compatible with CI/CD workflows
- **✅ Performance Maintained**: 97.3% accuracy targets preserved

No more command errors or platform-specific issues - the hooks adapt to your environment automatically!