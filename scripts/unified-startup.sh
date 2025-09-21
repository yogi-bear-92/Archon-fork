#!/bin/bash

# Unified Development Environment Startup Script
# Optimized for memory-constrained systems

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$PROJECT_ROOT/config/memory-limits.json"

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR:${NC} $1"
}

success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] SUCCESS:${NC} $1"
}

# Check system requirements
check_system() {
    log "🔍 Checking system requirements..."
    
    # Check memory
    if ! node "$SCRIPT_DIR/memory-monitor.js" --check-threshold > /dev/null; then
        warn "High memory usage detected. Starting in optimized mode."
        MEMORY_MODE="optimized"
    else
        success "Memory usage within acceptable limits."
        MEMORY_MODE="normal"
    fi
    
    # Check for required commands
    for cmd in node npm npx python3; do
        if ! command -v "$cmd" &> /dev/null; then
            error "Required command not found: $cmd"
            exit 1
        fi
    done
    
    success "System requirements check passed"
}

# Kill existing processes
cleanup_processes() {
    log "🧹 Cleaning up existing processes..."
    
    # Kill existing instances
    pkill -f "archon" || true
    pkill -f "claude-flow" || true
    pkill -f "serena" || true
    pkill -f "uvicorn" || true
    
    # Wait for processes to terminate
    sleep 2
    
    success "Process cleanup completed"
}

# Set memory limits for Node.js
set_memory_limits() {
    local mode=$1
    
    case $mode in
        "optimized")
            export NODE_OPTIONS="--max-old-space-size=1024 --gc-interval=100"
            export UV_THREADPOOL_SIZE=4
            log "🔧 Set optimized memory limits (1GB Node.js heap)"
            ;;
        "minimal")
            export NODE_OPTIONS="--max-old-space-size=512 --gc-interval=50"
            export UV_THREADPOOL_SIZE=2
            log "🔧 Set minimal memory limits (512MB Node.js heap)"
            ;;
        *)
            export NODE_OPTIONS="--max-old-space-size=2048"
            export UV_THREADPOOL_SIZE=8
            log "🔧 Set normal memory limits (2GB Node.js heap)"
            ;;
    esac
}

# Start services based on memory mode
start_services() {
    local mode=$1
    
    cd "$PROJECT_ROOT"
    
    case $mode in
        "minimal")
            log "🚀 Starting minimal services (Serena + Claude Code only)..."
            
            # Start Serena MCP with memory limits
            log "Starting Serena MCP server..."
            nohup node -e "
                process.env.SERENA_MEMORY_LIMIT = '512MB';
                require('./node_modules/@serena-chat/serena/mcp-server.js');
            " > logs/serena.log 2>&1 &
            SERENA_PID=$!
            
            success "Minimal services started (PID: Serena=$SERENA_PID)"
            ;;
            
        "optimized")
            log "🚀 Starting optimized services (Serena + Archon)..."
            
            # Start Serena MCP
            log "Starting Serena MCP server..."
            nohup npx serena start --memory-limit=512MB > logs/serena.log 2>&1 &
            SERENA_PID=$!
            
            # Wait for Serena to be ready
            sleep 3
            
            # Start Archon API server
            log "Starting Archon API server..."
            cd python
            nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8080 --workers 1 > ../logs/archon.log 2>&1 &
            ARCHON_PID=$!
            cd ..
            
            success "Optimized services started (PIDs: Serena=$SERENA_PID, Archon=$ARCHON_PID)"
            ;;
            
        *)
            log "🚀 Starting full services (Serena + Archon + Claude Flow)..."
            
            # Start all services
            log "Starting Serena MCP server..."
            nohup npx serena start > logs/serena.log 2>&1 &
            SERENA_PID=$!
            
            sleep 3
            
            log "Starting Archon API server..."
            cd python
            nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8080 > ../logs/archon.log 2>&1 &
            ARCHON_PID=$!
            cd ..
            
            sleep 3
            
            log "Starting Claude Flow coordinator..."
            nohup npx claude-flow mcp start > logs/claude-flow.log 2>&1 &
            CLAUDE_FLOW_PID=$!
            
            success "Full services started (PIDs: Serena=$SERENA_PID, Archon=$ARCHON_PID, Claude-Flow=$CLAUDE_FLOW_PID)"
            ;;
    esac
}

# Start memory monitoring
start_monitoring() {
    log "📊 Starting memory monitoring..."
    
    nohup node "$SCRIPT_DIR/memory-monitor.js" --monitor > logs/memory-monitor.log 2>&1 &
    MONITOR_PID=$!
    
    success "Memory monitoring started (PID: $MONITOR_PID)"
}

# Save process PIDs for management
save_pids() {
    mkdir -p "$PROJECT_ROOT/.runtime"
    
    cat > "$PROJECT_ROOT/.runtime/services.pid" << EOF
SERENA_PID=${SERENA_PID:-}
ARCHON_PID=${ARCHON_PID:-}
CLAUDE_FLOW_PID=${CLAUDE_FLOW_PID:-}
MONITOR_PID=${MONITOR_PID:-}
MEMORY_MODE=$MEMORY_MODE
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
EOF
    
    log "Process PIDs saved to .runtime/services.pid"
}

# Wait for services to be ready
wait_for_services() {
    log "⏳ Waiting for services to be ready..."
    
    # Check Serena MCP
    for i in {1..30}; do
        if curl -s http://localhost:8051/health > /dev/null 2>&1; then
            success "Serena MCP is ready"
            break
        fi
        sleep 1
    done
    
    # Check Archon API if running
    if [[ -n "${ARCHON_PID:-}" ]]; then
        for i in {1..30}; do
            if curl -s http://localhost:8080/health > /dev/null 2>&1; then
                success "Archon API is ready"
                break
            fi
            sleep 1
        done
    fi
    
    success "All services are ready"
}

# Show status
show_status() {
    log "📋 Service Status:"
    echo
    printf "%-20s %-10s %-50s\n" "Service" "Status" "Endpoint"
    printf "%-20s %-10s %-50s\n" "-------" "------" "--------"
    
    # Check Serena
    if curl -s http://localhost:8051/health > /dev/null 2>&1; then
        printf "%-20s %-10s %-50s\n" "Serena MCP" "✅ UP" "http://localhost:8051"
    else
        printf "%-20s %-10s %-50s\n" "Serena MCP" "❌ DOWN" "http://localhost:8051"
    fi
    
    # Check Archon
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        printf "%-20s %-10s %-50s\n" "Archon API" "✅ UP" "http://localhost:8080"
    else
        printf "%-20s %-10s %-50s\n" "Archon API" "❌ DOWN" "http://localhost:8080"
    fi
    
    # Check Claude Flow
    if pgrep -f "claude-flow" > /dev/null 2>&1; then
        printf "%-20s %-10s %-50s\n" "Claude Flow" "✅ UP" "Background service"
    else
        printf "%-20s %-10s %-50s\n" "Claude Flow" "❌ DOWN" "Background service"
    fi
    
    echo
    
    # Show memory status
    node "$SCRIPT_DIR/memory-monitor.js" --status
}

# Main execution
main() {
    local mode="${1:-auto}"
    
    echo
    log "🎯 Starting Unified Development Environment"
    log "Mode: $mode"
    echo
    
    # Create logs directory
    mkdir -p "$PROJECT_ROOT/logs"
    
    # System checks
    check_system
    
    # Determine mode if auto
    if [[ "$mode" == "auto" ]]; then
        mode="$MEMORY_MODE"
    fi
    
    # Setup
    cleanup_processes
    set_memory_limits "$mode"
    
    # Start services
    start_services "$mode"
    
    # Start monitoring
    start_monitoring
    
    # Save runtime info
    save_pids
    
    # Wait and verify
    wait_for_services
    
    echo
    success "🎉 Unified Development Environment is ready!"
    echo
    
    show_status
    
    echo
    log "📖 Usage:"
    log "  - Archon PRP API: http://localhost:8080"
    log "  - Serena MCP: http://localhost:8051"
    log "  - Memory monitor: node scripts/memory-monitor.js --status"
    log "  - Stop services: ./scripts/stop-services.sh"
    echo
}

# Handle command line arguments
case "${1:-auto}" in
    "minimal"|"optimized"|"normal"|"auto")
        main "$1"
        ;;
    "--help"|"-h")
        echo "Usage: $0 [mode]"
        echo
        echo "Modes:"
        echo "  minimal    - Serena MCP only (lowest memory)"
        echo "  optimized  - Serena + Archon (balanced)"
        echo "  normal     - All services (full features)"
        echo "  auto       - Automatically detect best mode"
        echo
        echo "Environment will auto-adjust based on available memory."
        ;;
    *)
        error "Unknown mode: $1"
        error "Use --help for usage information"
        exit 1
        ;;
esac