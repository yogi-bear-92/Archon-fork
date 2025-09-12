# Automatic MCP Servers Setup

This document describes the automatic setup scripts for GitFlow (Claude Flow) and Serena MCP servers integration with Archon.

## 🚀 Quick Start

### Linux/macOS
```bash
# Run the setup script
./scripts/setup-mcp-servers.sh

# With options
./scripts/setup-mcp-servers.sh --help
```

### Windows
```powershell
# Run the PowerShell script
.\scripts\setup-mcp-servers.ps1

# With options
.\scripts\setup-mcp-servers.ps1 -Help
```

## 📋 What Gets Installed

The setup script automatically installs and configures:

1. **Claude Flow (GitFlow)** - AI-powered development workflow and SPARC methodology
2. **Serena** - Semantic code analysis and IDE assistance
3. **Integration Configuration** - Proper MCP server configuration for Archon
4. **Claude Desktop Config** - Ready-to-use Claude Desktop configuration

## 🎯 Features

### Memory-Aware Installation
Following CLAUDE.md patterns, the script includes:
- **Adaptive resource scaling** - Adjusts installation approach based on available memory
- **Emergency mode** - Single-threaded installation when memory is critical (<100MB)
- **Limited mode** - Reduced concurrency when memory is constrained (<500MB)
- **Normal mode** - Full parallel installation when memory is sufficient (>500MB)

### Error Handling & Validation
- **Prerequisites checking** - Validates Node.js, npm, Python/uv availability
- **Installation validation** - Tests that all components are properly installed
- **Configuration backup** - Creates backups before modifying existing configurations
- **Detailed logging** - Color-coded status messages with clear progress indicators

### Cross-Platform Support
- **Linux/macOS** - Bash script with full feature set
- **Windows** - PowerShell script with equivalent functionality
- **Automatic path detection** - Handles different OS-specific paths and configurations

## ⚙️ Command Line Options

### Bash Script (Linux/macOS)
```bash
./scripts/setup-mcp-servers.sh [OPTIONS]

Options:
  --help, -h          Show help message
  --force             Force reinstallation of existing components
  --skip-validation   Skip final validation step
  --no-adaptive       Disable memory-aware adaptive scaling

Examples:
  ./scripts/setup-mcp-servers.sh                 # Standard installation
  ./scripts/setup-mcp-servers.sh --force         # Force reinstall everything
  ./scripts/setup-mcp-servers.sh --no-adaptive   # Disable memory management
```

### PowerShell Script (Windows)
```powershell
.\scripts\setup-mcp-servers.ps1 [OPTIONS]

Options:
  -Help               Show help message
  -Force              Force reinstallation of existing components
  -SkipValidation     Skip final validation step
  -NoAdaptive         Disable memory-aware adaptive scaling

Examples:
  .\scripts\setup-mcp-servers.ps1                # Standard installation
  .\scripts\setup-mcp-servers.ps1 -Force         # Force reinstall everything
  .\scripts\setup-mcp-servers.ps1 -NoAdaptive    # Disable memory management
```

## 🔧 Prerequisites

### Required
- **Node.js 18+** - For Claude Flow installation
- **npm** - Package manager for Node.js components
- **Git** - For cloning and repository operations

### Optional (Auto-installed)
- **uv** - Python package manager (installed automatically if missing)
- **Python 3.8+** - Required for uv installation

### System Requirements
- **Memory**: Minimum 100MB available (500MB+ recommended)
- **Disk Space**: ~200MB for all components
- **Network**: Internet connection for downloading packages

## 📁 Generated Configurations

### Archon MCP Configuration
The script updates `python/config/mcp_servers.json`:

```json
{
  "claude-flow": {
    "command": "npx",
    "args": ["claude-flow@alpha", "mcp", "start", "--project-root", "/path/to/project"],
    "auto_start": true,
    "transport": "stdio"
  },
  "serena": {
    "command": "uvx", 
    "args": ["--from", "git+https://github.com/oraios/serena", "serena", "start-mcp-server", "--context", "ide-assistant", "--project", "/path/to/project"],
    "auto_start": true,
    "transport": "stdio"
  }
}
```

### Claude Desktop Configuration
The script creates platform-specific Claude Desktop config:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Linux**: `~/.config/claude/claude_desktop_config.json`  
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "claude-flow": {
      "command": "npx",
      "args": ["claude-flow@alpha", "mcp", "start", "--project-root", "/path/to/project"]
    },
    "serena": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/oraios/serena", "serena", "start-mcp-server", "--context", "ide-assistant", "--project", "/path/to/project"]
    },
    "archon-mcp": {
      "command": "python3",
      "args": ["-m", "src.mcp_server.mcp_server"],
      "cwd": "/path/to/project/python"
    }
  }
}
```

## 🧪 Post-Installation Validation

The script validates installation by:

1. **Component availability** - Checking that `claude-flow` and `uvx` commands work
2. **Version verification** - Ensuring installed versions are correct
3. **Configuration files** - Confirming all config files were created properly
4. **Path resolution** - Verifying project paths are correctly set

## 🚀 Usage After Installation

### 1. Start Archon Server
```bash
# Start with Docker
docker compose up -d

# Or start components individually
cd python && python -m src.server.main
```

### 2. Restart Claude Desktop
Completely quit and restart Claude Desktop to pick up the new MCP server configurations.

### 3. Test MCP Integration
In Claude Desktop, you should now see tools from:
- **claude-flow**: SPARC methodology, swarm coordination, workflow automation
- **serena**: Code analysis, file operations, symbol search, refactoring
- **archon-mcp**: Project management, knowledge base, task tracking

## 🛠️ Manual Configuration

If you need to manually configure the MCP servers:

### Claude Flow
```bash
# Install globally
npm install -g claude-flow@alpha

# Initialize in project
claude-flow init --project-root=/path/to/project

# Start MCP server
npx claude-flow@alpha mcp start --project-root=/path/to/project
```

### Serena  
```bash
# Install via uv
uvx --from git+https://github.com/oraios/serena serena --version

# Start MCP server
uvx --from git+https://github.com/oraios/serena serena start-mcp-server --context ide-assistant --project /path/to/project
```

## 🔍 Troubleshooting

### Common Issues

**1. Node.js version too old**
```bash
# Update Node.js to 18+ 
# On macOS with Homebrew:
brew install node

# On Windows, download from nodejs.org
# On Linux, use your package manager or NodeSource
```

**2. uv installation fails**
```bash
# Manual uv installation
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows PowerShell:
irm https://astral.sh/uv/install.ps1 | iex
```

**3. Memory constraints**
```bash
# Run with reduced concurrency
./scripts/setup-mcp-servers.sh --no-adaptive

# Or force sequential installation
MAX_CONCURRENT_INSTALLS=1 ./scripts/setup-mcp-servers.sh
```

**4. Claude Desktop not detecting MCP servers**
- Ensure Claude Desktop is completely quit (check system tray)
- Restart Claude Desktop after configuration
- Check configuration file syntax with a JSON validator
- Verify all file paths are absolute and correct

**5. Permission issues (Windows)**
```powershell
# Run PowerShell as Administrator
# Or set execution policy temporarily:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Debug Information

To get debug information:

```bash
# Check Claude Flow status
claude-flow --version
claude-flow hive status

# Check Serena installation
uvx --from git+https://github.com/oraios/serena serena --version

# Test MCP server connections
npx claude-flow@alpha mcp start --test-connection
uvx --from git+https://github.com/oraios/serena serena start-mcp-server --test-connection
```

### Log Files

- **Claude Flow**: `.claude-flow/logs/`
- **Serena**: Default logs to console, enable file logging in config
- **Archon**: `python/logs/` (if configured)
- **Claude Desktop**: Platform-specific log locations

## 🔄 Updates and Maintenance

### Updating Components
```bash
# Update Claude Flow
npm update -g claude-flow@alpha

# Update Serena (automatically pulls latest from git)
uvx --from git+https://github.com/oraios/serena serena --version

# Re-run setup script to update configurations
./scripts/setup-mcp-servers.sh --force
```

### Uninstallation
```bash
# Remove Claude Flow
npm uninstall -g claude-flow@alpha

# Remove Serena (uv tool management)
uv tool uninstall serena

# Remove configurations (backup first!)
rm -rf ~/.config/claude/claude_desktop_config.json  # Linux
rm -rf ~/Library/Application\ Support/Claude/claude_desktop_config.json  # macOS
```

## 📊 Memory Management Details

The setup script implements CLAUDE.md memory-aware patterns:

### Memory Thresholds
- **Critical (<100MB)**: Single-threaded, essential components only
- **Limited (100-500MB)**: Reduced concurrency, conservative installation
- **Normal (>500MB)**: Full parallel installation, all features

### Adaptive Behaviors
- **Dynamic scaling**: Adjusts installation approach based on real-time memory
- **Progressive loading**: Installs components incrementally to manage memory
- **Resource cleanup**: Immediate cleanup after each installation step
- **Emergency fallback**: Graceful degradation when memory becomes critical

### Monitoring
The script continuously monitors memory usage and adapts its behavior accordingly, ensuring stable installation even on resource-constrained systems.

## 🤝 Integration with Archon

The setup script is designed to seamlessly integrate with the existing Archon architecture:

- **Respects existing configuration**: Backs up and preserves existing MCP server configurations
- **Project-aware**: Automatically detects and configures project paths
- **Docker integration**: Compatible with Archon's Docker-based deployment
- **Service discovery**: Integrates with Archon's service discovery mechanism

This ensures that the MCP servers work harmoniously with Archon's project management, knowledge base, and AI coordination features.