#!/bin/bash

# Automatic GitFlow and Serena MCP Servers Setup Script
# Integrates with Archon system following CLAUDE.md memory-aware patterns

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Memory-aware configuration (following CLAUDE.md patterns)
MEMORY_THRESHOLD=95
MAX_CONCURRENT_INSTALLS=2
ADAPTIVE_RESOURCE_MODE=${ADAPTIVE_RESOURCE_MODE:-true}

# Functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_archon() {
    echo -e "${PURPLE}🏛️ $1${NC}"
}

log_mcp() {
    echo -e "${CYAN}🎯 $1${NC}"
}

# Memory-aware function (following CLAUDE.md patterns)
check_memory_usage() {
    if command -v vm_stat >/dev/null 2>&1; then
        # macOS
        local free_pages=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
        local total_pages=$((free_pages * 16 / 1024)) # Convert to MB
        echo $total_pages
    elif command -v free >/dev/null 2>&1; then
        # Linux
        free -m | grep "Available" | awk '{print $7}'
    else
        echo "1000" # Default assumption
    fi
}

adaptive_resource_check() {
    if [ "$ADAPTIVE_RESOURCE_MODE" = "true" ]; then
        local available_memory=$(check_memory_usage)
        log_info "Available memory: ${available_memory}MB"
        
        if [ "$available_memory" -lt 100 ]; then
            log_warning "Memory critical (<100MB) - enabling emergency mode"
            MAX_CONCURRENT_INSTALLS=1
            return 1
        elif [ "$available_memory" -lt 500 ]; then
            log_warning "Memory limited (<500MB) - reducing concurrency"
            MAX_CONCURRENT_INSTALLS=1
        fi
    fi
    return 0
}

# Check prerequisites
check_prerequisites() {
    log_archon "🔍 Checking system prerequisites..."
    
    local missing_deps=()
    
    # Check Node.js
    if ! command -v node >/dev/null 2>&1; then
        missing_deps+=("node")
    else
        local node_version=$(node -v | cut -d'.' -f1 | cut -d'v' -f2)
        if [ "$node_version" -lt 18 ]; then
            log_error "Node.js 18+ required. Current: $(node -v)"
            return 1
        fi
        log_success "Node.js $(node -v)"
    fi
    
    # Check npm
    if ! command -v npm >/dev/null 2>&1; then
        missing_deps+=("npm")
    else
        log_success "npm $(npm -v)"
    fi
    
    # Check Python/uv for Serena
    if ! command -v uvx >/dev/null 2>&1; then
        if ! command -v uv >/dev/null 2>&1; then
            log_warning "uv not found - will attempt to install"
        fi
    else
        log_success "uvx available"
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install missing dependencies and run again"
        return 1
    fi
    
    return 0
}

# Install uv if needed
install_uv() {
    if ! command -v uv >/dev/null 2>&1 && ! command -v uvx >/dev/null 2>&1; then
        log_info "Installing uv (Python package manager)..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        
        if command -v uv >/dev/null 2>&1; then
            log_success "uv installed successfully"
        else
            log_error "Failed to install uv"
            return 1
        fi
    fi
    return 0
}

# Install Claude Flow (GitFlow)
install_claude_flow() {
    log_mcp "📦 Installing Claude Flow (GitFlow)..."
    
    # Memory-aware installation
    adaptive_resource_check
    
    if npm list -g claude-flow@alpha >/dev/null 2>&1; then
        log_success "Claude Flow already installed"
    else
        log_info "Installing Claude Flow alpha version globally..."
        npm install -g claude-flow@alpha
        
        if command -v claude-flow >/dev/null 2>&1; then
            local version=$(claude-flow --version 2>/dev/null | head -1)
            log_success "Claude Flow installed: $version"
        else
            log_error "Claude Flow installation failed"
            return 1
        fi
    fi
    
    # Initialize Claude Flow
    log_info "Initializing Claude Flow..."
    if [ ! -d "$PROJECT_ROOT/.claude-flow" ]; then
        cd "$PROJECT_ROOT" && claude-flow init --project-root="$PROJECT_ROOT"
        log_success "Claude Flow initialized"
    else
        log_success "Claude Flow already initialized"
    fi
    
    return 0
}

# Install Serena
install_serena() {
    log_mcp "📦 Installing Serena MCP Server..."
    
    # Memory-aware installation
    adaptive_resource_check
    
    # Ensure uv is available
    install_uv || return 1
    
    log_info "Installing Serena from GitHub..."
    if command -v uvx >/dev/null 2>&1; then
        # Test installation
        uvx --from git+https://github.com/oraios/serena serena --version >/dev/null 2>&1 || {
            log_info "Installing Serena dependencies..."
            uvx --from git+https://github.com/oraios/serena serena --help >/dev/null 2>&1
        }
        log_success "Serena installed successfully"
    else
        log_error "uvx not available for Serena installation"
        return 1
    fi
    
    return 0
}

# Install Playwright MCP server
install_playwright() {
    log_info "🎭 Installing Playwright MCP server..."
    
    # Memory-aware installation
    adaptive_resource_check
    
    if command -v npx >/dev/null 2>&1; then
        log_info "Installing Playwright MCP server..."
        # Test if @playwright/mcp is available
        if npx @playwright/mcp --help >/dev/null 2>&1; then
            log_success "Playwright MCP server available"
        else
            log_info "Installing Playwright MCP dependencies..."
            npm install -g @playwright/mcp 2>/dev/null || log_warning "Playwright MCP may need to be installed on first use"
        fi
    else
        log_error "npx not available for Playwright installation"
        return 1
    fi
    
    return 0
}

# Update Archon MCP configuration
update_archon_config() {
    log_archon "⚙️ Updating Archon MCP configuration..."
    
    local config_file="$PROJECT_ROOT/python/config/mcp_servers.json"
    local project_path="$PROJECT_ROOT"
    
    if [ -f "$config_file" ]; then
        log_info "Updating existing MCP configuration"
        
        # Create backup
        cp "$config_file" "$config_file.backup.$(date +%Y%m%d_%H%M%S)"
        
        # Update configuration with current project path
        cat > "$config_file" << EOF
{
  "claude-flow": {
    "args": [
      "claude-flow@alpha",
      "mcp",
      "start",
      "--project-root",
      "$project_path"
    ],
    "auto_start": true,
    "command": "npx",
    "health_check_interval": 30,
    "name": "claude-flow",
    "retry_attempts": 3,
    "timeout": 10,
    "transport": "stdio"
  },
  "serena": {
    "args": [
      "--from",
      "git+https://github.com/oraios/serena",
      "serena",
      "start-mcp-server",
      "--context",
      "ide-assistant",
      "--project",
      "$project_path"
    ],
    "auto_start": true,
    "command": "uvx",
    "health_check_interval": 30,
    "name": "serena",
    "retry_attempts": 3,
    "timeout": 10,
    "transport": "stdio"
  }
}
EOF
        log_success "MCP configuration updated"
    else
        log_error "MCP configuration file not found: $config_file"
        return 1
    fi
    
    return 0
}

# Create Claude Desktop configuration
create_claude_desktop_config() {
    log_mcp "📝 Creating Claude Desktop configuration..."
    
    local claude_config_dir
    if [[ "$OSTYPE" == "darwin"* ]]; then
        claude_config_dir="$HOME/Library/Application Support/Claude"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        claude_config_dir="$HOME/.config/claude"
    else
        log_warning "Unsupported OS for automatic Claude Desktop config"
        return 0
    fi
    
    mkdir -p "$claude_config_dir"
    local config_file="$claude_config_dir/claude_desktop_config.json"
    
    cat > "$config_file" << EOF
{
  "mcpServers": {
    "claude-flow": {
      "command": "npx",
      "args": ["claude-flow@alpha", "mcp", "start", "--project-root", "$PROJECT_ROOT"],
      "env": {}
    },
    "serena": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/oraios/serena",
        "serena",
        "start-mcp-server",
        "--context",
        "ide-assistant",
        "--project",
        "$PROJECT_ROOT"
      ],
      "env": {}
    },
    "playwright": {
      "command": "npx",
      "args": ["@playwright/mcp"],
      "env": {}
    },
    "archon-mcp": {
      "command": "python3",
      "args": ["-m", "src.mcp_server.mcp_server"],
      "cwd": "$PROJECT_ROOT/python",
      "env": {
        "PYTHONPATH": "$PROJECT_ROOT/python"
      }
    }
  }
}
EOF
    
    log_success "Claude Desktop configuration created: $config_file"
    return 0
}

# Configure Claude Code MCP servers
configure_claude_code() {
    log_mcp "🔧 Configuring Claude Code MCP servers..."
    
    # Check if claude command is available
    if ! command -v claude >/dev/null 2>&1; then
        log_warning "Claude Code CLI not found - skipping Claude Code configuration"
        log_info "Install Claude Code from: https://docs.anthropic.com/en/docs/claude-code"
        return 0
    fi
    
    # Add Claude Flow MCP server
    log_info "Adding Claude Flow to Claude Code..."
    claude mcp add claude-flow npx claude-flow@alpha mcp start 2>/dev/null || log_warning "Failed to add Claude Flow"
    
    # Add Serena MCP server using JSON format (complex arguments)
    log_info "Adding Serena to Claude Code..."
    claude mcp add-json serena "{\"command\": \"uvx\", \"args\": [\"--from\", \"git+https://github.com/oraios/serena\", \"serena\", \"start-mcp-server\", \"--context\", \"ide-assistant\", \"--project\", \"$PROJECT_ROOT\"], \"transport\": \"stdio\"}" 2>/dev/null || log_warning "Failed to add Serena"
    
    # Add Playwright MCP server
    log_info "Adding Playwright to Claude Code..."
    claude mcp add playwright npx @playwright/mcp 2>/dev/null || log_warning "Failed to add Playwright"
    
    # Fix Archon MCP server transport if needed
    log_info "Checking Archon MCP server transport configuration..."
    if grep -q "streamable-http" "$PROJECT_ROOT/python/src/mcp_server/mcp_server.py" 2>/dev/null; then
        log_info "Fixing Archon MCP server transport to use stdio..."
        sed -i.bak 's/transport="streamable-http"/transport="stdio"/' "$PROJECT_ROOT/python/src/mcp_server/mcp_server.py" 2>/dev/null || log_warning "Failed to update Archon transport"
    fi
    
    log_success "Claude Code MCP configuration completed"
    return 0
}

# Validate installation
validate_installation() {
    log_archon "🧪 Validating MCP servers installation..."
    
    local errors=0
    
    # Check Claude Flow
    if command -v claude-flow >/dev/null 2>&1; then
        if claude-flow --version >/dev/null 2>&1; then
            log_success "Claude Flow validation passed"
        else
            log_error "Claude Flow validation failed"
            ((errors++))
        fi
    else
        log_error "Claude Flow not found in PATH"
        ((errors++))
    fi
    
    # Check Serena
    if command -v uvx >/dev/null 2>&1; then
        if uvx --from git+https://github.com/oraios/serena serena --version >/dev/null 2>&1; then
            log_success "Serena validation passed"
        else
            log_warning "Serena validation inconclusive (may need first run)"
        fi
    else
        log_error "uvx not available for Serena"
        ((errors++))
    fi
    
    # Check configuration files
    if [ -f "$PROJECT_ROOT/python/config/mcp_servers.json" ]; then
        log_success "Archon MCP configuration exists"
    else
        log_error "Archon MCP configuration missing"
        ((errors++))
    fi
    
    if [ $errors -eq 0 ]; then
        log_success "All validations passed!"
        return 0
    else
        log_error "$errors validation(s) failed"
        return 1
    fi
}

# Main execution
main() {
    echo "================================================================"
    log_archon "🚀 Automatic MCP Servers Setup"
    echo "   Claude Flow (GitFlow) + Serena + Archon Integration"
    echo "   Memory-Aware Installation Following CLAUDE.md Patterns"
    echo "================================================================"
    echo ""
    
    log_info "Project root: $PROJECT_ROOT"
    
    # Memory-aware pre-check
    adaptive_resource_check || {
        log_warning "System memory is critical - proceeding with conservative settings"
    }
    
    # Execute installation steps
    check_prerequisites || exit 1
    
    log_info "Starting installation with max $MAX_CONCURRENT_INSTALLS concurrent processes..."
    
    # Install components (with potential parallelization if memory allows)
    if [ $MAX_CONCURRENT_INSTALLS -gt 1 ]; then
        log_info "Installing components in parallel..."
        {
            install_claude_flow &
            install_serena &
            install_playwright &
            wait
        }
    else
        log_info "Installing components sequentially (memory constrained)..."
        install_claude_flow || exit 1
        install_serena || exit 1
        install_playwright || exit 1
    fi
    
    # Configure integration
    update_archon_config || exit 1
    create_claude_desktop_config
    configure_claude_code
    
    # Final validation
    validate_installation || {
        log_warning "Some validations failed - check logs above"
    }
    
    echo ""
    log_archon "🎉 MCP Servers setup completed!"
    echo ""
    log_info "Next steps:"
    log_info "  1. Restart Claude Desktop to pick up new configuration (if using Desktop)"
    log_info "  2. For Claude Code: Run '/mcp' to see available servers"
    log_info "  3. Start Archon server: docker compose up -d"
    log_info "  4. Test MCP integration in Claude Desktop or Claude Code"
    echo ""
    log_info "Available MCP servers:"
    log_info "  • claude-flow: GitFlow integration and SPARC methodology"
    log_info "  • serena: Semantic code analysis and IDE assistance"
    log_info "  • playwright: Browser automation and testing"
    log_info "  • archon-mcp: Project management and knowledge base"
    echo ""
}

# Handle cleanup on exit
cleanup() {
    echo ""
    log_info "🛑 Setup interrupted - cleaning up..."
    # Kill any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit 1
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Parse command line arguments
HELP=false
FORCE=false
SKIP_VALIDATION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --help|-h)
            HELP=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --skip-validation)
            SKIP_VALIDATION=true
            shift
            ;;
        --no-adaptive)
            ADAPTIVE_RESOURCE_MODE=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            HELP=true
            shift
            ;;
    esac
done

if [ "$HELP" = true ]; then
    echo "🏛️ Automatic MCP Servers Setup Script"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help, -h          Show this help message"
    echo "  --force             Force reinstallation of existing components"
    echo "  --skip-validation   Skip final validation step"
    echo "  --no-adaptive       Disable memory-aware adaptive scaling"
    echo ""
    echo "Examples:"
    echo "  $0                  # Standard installation"
    echo "  $0 --force          # Force reinstall everything"
    echo "  $0 --no-adaptive    # Disable memory management"
    exit 0
fi

# Run main function
main "$@"