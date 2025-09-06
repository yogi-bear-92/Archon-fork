#!/bin/bash

# Docker Lunar MCPX Gateway Startup Script
# Manages the MCPX gateway in Docker environment

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

log_docker() {
    echo -e "${CYAN}🐳 $1${NC}"
}

# Parse command line arguments
PROFILE="default"
BUILD=true
LOGS=false
HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --profile)
            PROFILE="$2"
            shift 2
            ;;
        --no-build)
            BUILD=false
            shift
            ;;
        --logs)
            LOGS=true
            shift
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
    echo "🌙🐳 Docker Lunar MCPX Gateway Launcher"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --profile PROFILE     Docker compose profile (default, agents)"
    echo "  --no-build            Skip building images"
    echo "  --logs                Follow logs after startup"
    echo "  --help, -h            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Start MCPX with default services"
    echo "  $0 --profile agents   # Start with agents profile"
    echo "  $0 --logs             # Start and follow logs"
    echo "  $0 --no-build         # Start without rebuilding"
    exit 0
fi

echo "================================================================"
log_lunar "🐳 Docker Lunar MCPX Gateway"
echo "   Container-based MCP Server Aggregation"
echo "   Archon + Claude Flow + Serena Integration"
echo "================================================================"
echo ""

cd "$PROJECT_ROOT"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

log_success "Docker daemon is running"

# Check if docker-compose exists
COMPOSE_CMD="docker-compose"
if ! command -v docker-compose > /dev/null 2>&1; then
    if command -v docker > /dev/null 2>&1 && docker compose version > /dev/null 2>&1; then
        COMPOSE_CMD="docker compose"
    else
        log_error "Neither docker-compose nor 'docker compose' found"
        exit 1
    fi
fi

log_success "Using: $COMPOSE_CMD"

# Set up environment
if [ -f ".env.mcpx" ]; then
    log_info "Loading MCPX environment configuration"
    export $(cat .env.mcpx | grep -v '^#' | xargs)
    log_success "Environment configured"
fi

# Build images if requested
if [ "$BUILD" = true ]; then
    log_docker "Building Docker images..."
    
    if [ "$PROFILE" = "agents" ]; then
        $COMPOSE_CMD -f deployment/docker/docker-compose.yml --profile agents build
    else
        $COMPOSE_CMD -f deployment/docker/docker-compose.yml build
    fi
    
    log_success "Docker images built"
fi

# Display startup information
log_lunar "🚀 Starting Docker services..."
log_info "Profile: $PROFILE"
log_info "Compose file: deployment/docker/docker-compose.yml"
echo ""

log_docker "Service Architecture:"
log_info "  🔹 archon-server (FastAPI) - :8181"
log_info "  🔹 archon-mcp (HTTP MCP) - :8051" 
log_info "  🔹 archon-claude-flow (Flow-Nexus) - :8053"
log_info "  🌙 archon-lunar-mcpx (MCPX Gateway) - :8090"
log_info "  🎨 archon-frontend (React UI) - :3737"

if [ "$PROFILE" = "agents" ]; then
    log_info "  🤖 archon-agents (AI Agents) - :8052"
fi

echo ""
log_lunar "🌐 MCPX Control Plane: http://localhost:8090"
log_docker "🎯 Frontend UI: http://localhost:3737"
echo ""

# Start services
log_docker "Starting containers..."

if [ "$PROFILE" = "agents" ]; then
    $COMPOSE_CMD -f deployment/docker/docker-compose.yml --profile agents up -d
else
    $COMPOSE_CMD -f deployment/docker/docker-compose.yml up -d
fi

log_success "Containers started"

# Wait for services to be healthy
log_info "Waiting for services to be ready..."
sleep 10

# Check service health
services=("archon-server" "archon-mcp" "archon-claude-flow" "archon-lunar-mcpx")
if [ "$PROFILE" = "agents" ]; then
    services+=("archon-agents")
fi

for service in "${services[@]}"; do
    if $COMPOSE_CMD -f deployment/docker/docker-compose.yml ps "$service" | grep -q "healthy\|Up"; then
        log_success "$service is running"
    else
        log_warning "$service status unknown - check logs"
    fi
done

echo ""
log_lunar "🎉 Lunar MCPX Gateway is now running!"
echo ""
log_info "Available endpoints:"
log_info "  • MCPX Control Plane: http://localhost:8090"
log_info "  • MCPX API Status: http://localhost:8090/api/status"
log_info "  • Archon Server: http://localhost:8181"
log_info "  • Frontend UI: http://localhost:3737"
echo ""

log_info "To connect with Claude Desktop, add to mcp_settings.json:"
echo '  {'
echo '    "mcpServers": {'
echo '      "archon-lunar-mcpx": {'
echo '        "command": "docker",'
echo '        "args": ["exec", "-i", "archon-lunar-mcpx", "node", "lunar-mcpx-gateway.js"]'
echo '      }'
echo '    }'
echo '  }'
echo ""

log_info "Useful commands:"
log_info "  • View logs: $COMPOSE_CMD -f deployment/docker/docker-compose.yml logs -f"
log_info "  • Stop services: $COMPOSE_CMD -f deployment/docker/docker-compose.yml down"
log_info "  • Restart MCPX: $COMPOSE_CMD -f deployment/docker/docker-compose.yml restart archon-lunar-mcpx"
echo ""

# Follow logs if requested
if [ "$LOGS" = true ]; then
    log_docker "Following container logs..."
    echo ""
    $COMPOSE_CMD -f deployment/docker/docker-compose.yml logs -f
fi