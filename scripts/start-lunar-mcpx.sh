#!/bin/bash

# Start Lunar MCPX Gateway
# Official Lunar.dev integration for MCP server aggregation
# Integrates Claude Flow, Flow-Nexus, Serena, and Archon

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC_DIR="$PROJECT_ROOT/src"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

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

log_lunar() {
    echo -e "${PURPLE}🌙 $1${NC}"
}

log_mcpx() {
    echo -e "${CYAN}🎯 $1${NC}"
}

# Parse command line arguments
CONTROL_PLANE=true
DEBUG=false
CONFIG_FILE=""
HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-control-plane)
            CONTROL_PLANE=false
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --config|-c)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            HELP=true
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
    echo "🌙 Lunar MCPX Gateway Launcher"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --no-control-plane    Disable web-based control plane"
    echo "  --debug               Enable debug logging"
    echo "  --config, -c FILE     Use custom configuration file"
    echo "  --help, -h            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                           # Start with default config"
    echo "  $0 --debug                   # Start with debug logging"
    echo "  $0 --config custom.json      # Use custom configuration"
    echo "  $0 --no-control-plane        # Start without control plane"
    exit 0
fi

echo "================================================================"
log_lunar "Lunar MCPX Gateway"
echo "   Official Lunar.dev MCP Server Aggregation"
echo "   Claude Flow + Flow-Nexus + Serena + Archon PRP"
echo "================================================================"
echo ""

log_info "Project Root: $PROJECT_ROOT"
log_info "Source Directory: $SRC_DIR"

# Check if lunar-mcpx-gateway.js exists
if [ ! -f "$SRC_DIR/lunar-mcpx-gateway.js" ]; then
    log_error "lunar-mcpx-gateway.js not found in $SRC_DIR"
    exit 1
fi

# Check if package.json exists  
if [ ! -f "$SRC_DIR/package.json" ]; then
    log_error "package.json not found in $SRC_DIR"
    exit 1
fi

# Check configuration file
if [ -n "$CONFIG_FILE" ]; then
    if [ ! -f "$CONFIG_FILE" ]; then
        log_error "Configuration file not found: $CONFIG_FILE"
        exit 1
    fi
    log_success "Using configuration: $CONFIG_FILE"
elif [ -f "$SRC_DIR/mcpx-config.json" ]; then
    log_success "Using default configuration: mcpx-config.json"
else
    log_warning "No configuration file found, using embedded defaults"
fi

# Change to src directory
cd "$SRC_DIR"

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    log_info "Installing dependencies..."
    npm install
    log_success "Dependencies installed"
fi

# Check system prerequisites
log_info "Checking system prerequisites..."

# Check Node.js version
if ! command -v node > /dev/null 2>&1; then
    log_error "Node.js not found. Please install Node.js 18+"
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'.' -f1 | cut -d'v' -f2)
if [ "$NODE_VERSION" -lt 18 ]; then
    log_error "Node.js 18+ required. Current: $(node -v)"
    exit 1
fi

log_success "Node.js $(node -v)"

# Check npm
if ! command -v npm > /dev/null 2>&1; then
    log_error "npm not found"
    exit 1
fi

log_success "npm $(npm -v)"

# Check system health and service availability
log_mcpx "Checking MCP server availability..."

# Check if Archon server is running (port 8181)
if curl -s -f http://localhost:8181/health > /dev/null 2>&1; then
    log_success "Archon PRP server responding (localhost:8181)"
    ARCHON_HEALTH=$(curl -s http://localhost:8181/health | head -1)
    log_info "Archon status: $ARCHON_HEALTH"
else
    log_warning "Archon server not responding on port 8181"
    log_info "To start Archon: export ARCHON_SERVER_PORT=8181 && python3 -m uvicorn src.server.main:app --host 0.0.0.0 --port 8181"
fi

# Check if Flow-Nexus is running (port 8051)  
if curl -s -f http://localhost:8051 > /dev/null 2>&1; then
    log_success "Flow-Nexus server responding (localhost:8051)"
else
    log_warning "Flow-Nexus not responding on port 8051"
    log_info "To start Flow-Nexus: npx flow-nexus@latest mcp start"
fi

# Check Claude Flow availability
if command -v npx > /dev/null 2>&1; then
    if npx claude-flow@alpha --version > /dev/null 2>&1; then
        CLAUDE_FLOW_VERSION=$(npx claude-flow@alpha --version 2>/dev/null | head -1)
        log_success "Claude Flow available: $CLAUDE_FLOW_VERSION"
    else
        log_warning "Claude Flow not available - installing..."
        npm install -g claude-flow@alpha
    fi
else
    log_error "npx not found. Please install Node.js"
    exit 1
fi

# Check Serena availability
if npx serena@latest --version > /dev/null 2>&1; then
    SERENA_VERSION=$(npx serena@latest --version 2>/dev/null | head -1)
    log_success "Serena available: $SERENA_VERSION"
else
    log_warning "Serena not available - code intelligence features limited"
fi

# Check if control plane port is available (8090)
if [ "$CONTROL_PLANE" = true ]; then
    if netstat -an 2>/dev/null | grep -q ":8090 " || lsof -i :8090 > /dev/null 2>&1; then
        log_warning "Port 8090 already in use (MCPX Control Plane)"
        log_info "Control plane may not start properly"
    else
        log_success "Port 8090 available for control plane"
    fi
fi

log_success "System checks completed"

# Set environment variables
export NODE_ENV=${NODE_ENV:-production}
export MCPX_DEBUG=$DEBUG
export MCPX_CONTROL_PLANE=$CONTROL_PLANE

if [ -n "$CONFIG_FILE" ]; then
    export MCPX_CONFIG_FILE="$CONFIG_FILE"
fi

# Display startup information
echo ""
log_lunar "🚀 Starting Lunar MCPX Gateway..."
log_mcpx "MCPX Configuration:"
log_info "  • Gateway Mode: Lunar MCPX Aggregation"
log_info "  • Control Plane: $($CONTROL_PLANE && echo "Enabled (http://localhost:8090)" || echo "Disabled")"
log_info "  • Debug Mode: $($DEBUG && echo "Enabled" || echo "Disabled")"
log_info "  • Config File: $([ -n "$CONFIG_FILE" ] && echo "$CONFIG_FILE" || echo "Embedded/Default")"
echo ""
log_mcpx "Managed MCP Servers:"
log_info "  🔹 Archon PRP (HTTP) - localhost:8181"
log_info "  🔹 Claude Flow (Command) - npx claude-flow@alpha"
log_info "  🔹 Serena (Stdio) - npx serena@latest"
log_info "  🔹 Flow-Nexus (HTTP) - localhost:8051"
echo ""
log_mcpx "Features:"
log_info "  • Dynamic server health monitoring"
log_info "  • Priority-based tool routing"
log_info "  • Circuit breaker protection"
log_info "  • Live traffic inspection"
log_info "  • Automatic failover"
echo ""

if [ "$CONTROL_PLANE" = true ]; then
    log_lunar "🌐 Control Plane will be available at:"
    log_info "    http://localhost:8090"
    log_info "    Dashboard: http://localhost:8090/dashboard"
    log_info "    Metrics: http://localhost:8090/metrics"
    echo ""
fi

log_info "To connect with Claude Desktop, add to mcp_settings.json:"
echo '  {'
echo '    "mcpServers": {'
echo '      "archon-lunar-mcpx": {'
echo '        "command": "node",'
echo "        \"args\": [\"$SRC_DIR/lunar-mcpx-gateway.js\"]"
echo '      }'
echo '    }'
echo '  }'
echo ""

log_lunar "🎯 Starting gateway now..."
echo ""

# Handle cleanup on exit
cleanup() {
    echo ""
    log_info "🛑 Shutting down Lunar MCPX Gateway..."
    
    # Kill any child processes we might have started
    jobs -p | xargs -r kill 2>/dev/null || true
    
    log_success "Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start the Lunar MCPX gateway
exec node lunar-mcpx-gateway.js