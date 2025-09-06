#!/bin/bash

# Unified MCP Server Startup Script
# Integrates Claude Flow, Flow-Nexus, Serena, and Archon

set -e

echo "🚀 Starting Unified MCP Integration System..."

# Configuration
UNIFIED_MCP_DIR="/Users/yogi/Projects/Archon-fork/src"
UNIFIED_MCP_PORT=8050
ARCHON_SERVER_HOST="127.0.0.1"
ARCHON_SERVER_PORT=8181

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Check system prerequisites
check_prerequisites() {
    log_info "Checking system prerequisites..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        log_error "Node.js is not installed. Please install Node.js 18+ to continue."
        exit 1
    fi
    
    NODE_VERSION=$(node -v | cut -d'.' -f1 | cut -d'v' -f2)
    if [ "$NODE_VERSION" -lt 18 ]; then
        log_error "Node.js version 18 or higher is required. Current version: $(node -v)"
        exit 1
    fi
    
    log_success "Node.js $(node -v) detected"
    
    # Check npm
    if ! command -v npm &> /dev/null; then
        log_error "npm is not installed"
        exit 1
    fi
    
    log_success "npm $(npm -v) detected"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_warning "Python 3 not found - Archon server may not be available"
    else
        log_success "Python $(python3 --version 2>&1) detected"
    fi
}

# Check and install dependencies
install_dependencies() {
    log_info "Installing/checking dependencies..."
    
    cd "$UNIFIED_MCP_DIR"
    
    if [ ! -f "package.json" ]; then
        log_error "package.json not found in $UNIFIED_MCP_DIR"
        exit 1
    fi
    
    # Install Node.js dependencies
    if [ ! -d "node_modules" ] || [ "package.json" -nt "node_modules" ]; then
        log_info "Installing Node.js dependencies..."
        npm install
        log_success "Dependencies installed"
    else
        log_success "Dependencies up to date"
    fi
    
    # Check Claude Flow
    if command -v npx &> /dev/null; then
        if npx claude-flow@alpha --version &> /dev/null; then
            log_success "Claude Flow available"
        else
            log_warning "Claude Flow not available - installing..."
            npm install -g claude-flow@alpha
        fi
    fi
    
    # Check Flow-Nexus
    if npx flow-nexus --version &> /dev/null; then
        log_success "Flow-Nexus available"
    else
        log_warning "Flow-Nexus not available - some features may be limited"
    fi
    
    # Check Serena
    if command -v serena &> /dev/null || python3 -c "import serena" &> /dev/null; then
        log_success "Serena available"
    else
        log_warning "Serena not available - code intelligence features may be limited"
    fi
}

# Check service health
check_services() {
    log_info "Checking service availability..."
    
    # Check Archon server
    if curl -sf "http://${ARCHON_SERVER_HOST}:${ARCHON_SERVER_PORT}/health" > /dev/null 2>&1; then
        log_success "Archon server responding at ${ARCHON_SERVER_HOST}:${ARCHON_SERVER_PORT}"
    else
        log_warning "Archon server not responding - will attempt to start"
        
        # Try to start Archon server
        if [ -f "/Users/yogi/Projects/Archon-fork/python/src/server/main.py" ]; then
            log_info "Starting Archon server..."
            cd /Users/yogi/Projects/Archon-fork
            python3 -m uvicorn src.server.main:app --host ${ARCHON_SERVER_HOST} --port ${ARCHON_SERVER_PORT} --reload &
            ARCHON_PID=$!
            sleep 5
            
            if curl -sf "http://${ARCHON_SERVER_HOST}:${ARCHON_SERVER_PORT}/health" > /dev/null 2>&1; then
                log_success "Archon server started (PID: $ARCHON_PID)"
            else
                log_warning "Could not start Archon server - service will run in degraded mode"
            fi
        fi
    fi
    
    # Check other services
    if pgrep -f "flow-nexus" > /dev/null; then
        log_success "Flow-Nexus MCP server running"
    else
        log_warning "Flow-Nexus MCP server not detected"
    fi
    
    if pgrep -f "serena.*mcp" > /dev/null; then
        log_success "Serena MCP server running" 
    else
        log_warning "Serena MCP server not detected"
    fi
}

# Start unified MCP server
start_server() {
    log_info "Starting Unified MCP Server on port $UNIFIED_MCP_PORT..."
    
    cd "$UNIFIED_MCP_DIR"
    
    # Make sure the script is executable
    chmod +x unified-mcp-server.js
    
    # Set environment variables
    export UNIFIED_MCP_PORT=$UNIFIED_MCP_PORT
    export ARCHON_SERVER_HOST=$ARCHON_SERVER_HOST
    export ARCHON_SERVER_PORT=$ARCHON_SERVER_PORT
    export NODE_ENV=production
    
    log_success "🎯 Starting Unified MCP Server..."
    log_info "Configuration:"
    log_info "  • Master Controller: localhost:$UNIFIED_MCP_PORT"  
    log_info "  • Archon PRP: $ARCHON_SERVER_HOST:$ARCHON_SERVER_PORT"
    log_info "  • Claude Flow: Command-line integration"
    log_info "  • Serena: MCP server integration" 
    log_info "  • Flow-Nexus: MCP server integration"
    log_info ""
    log_info "To connect Claude Desktop:"
    log_info '  Add to mcp_settings.json:'
    echo '  {'
    echo '    "mcpServers": {'
    echo '      "unified-mcp": {'
    echo '        "command": "node",'
    echo '        "args": ["'$UNIFIED_MCP_DIR'/unified-mcp-server.js"]'
    echo '      }'
    echo '    }'
    echo '  }'
    log_info ""
    log_success "Starting server now..."
    
    # Start the server
    exec node unified-mcp-server.js
}

# Handle cleanup on exit
cleanup() {
    log_info "Shutting down services..."
    
    # Kill background processes if we started them
    if [ ! -z "$ARCHON_PID" ]; then
        kill $ARCHON_PID 2>/dev/null || true
        log_info "Stopped Archon server"
    fi
    
    log_success "Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
main() {
    echo "================================================================"
    echo "🎯 Unified MCP Integration System"
    echo "   Claude Flow + Flow-Nexus + Serena + Archon PRP"
    echo "================================================================"
    echo ""
    
    check_prerequisites
    install_dependencies  
    check_services
    start_server
}

# Run main function
main "$@"