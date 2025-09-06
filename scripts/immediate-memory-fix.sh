#!/bin/bash

# Immediate Memory Optimization Script
# Addresses critical 99%+ memory usage in Archon system

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

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

# Emergency memory cleanup
emergency_cleanup() {
    log "🚨 EMERGENCY: Executing immediate memory cleanup..."
    
    # Kill memory-intensive processes
    log "Stopping resource-intensive services..."
    pkill -f "archon-ui" || true
    pkill -f "claude-flow.*mcp" || true
    pkill -f "ruv-swarm.*mcp" || true
    pkill -f "uvicorn.*8080" || true
    pkill -f "uvicorn.*8052" || true
    pkill -f "node.*serena" || true
    
    # Wait for processes to terminate
    sleep 3
    
    # Force kill if still running
    pkill -9 -f "archon" || true
    pkill -9 -f "claude-flow" || true
    pkill -9 -f "serena" || true
    
    success "Services stopped"
    
    # Clear caches aggressively
    log "Clearing system caches..."
    
    # Node.js caches
    rm -rf "$PROJECT_ROOT/node_modules/.cache" || true
    rm -rf "$PROJECT_ROOT/.npm" || true
    rm -rf "$HOME/.npm/_cacache" || true
    
    # Python caches
    find "$PROJECT_ROOT" -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -name "*.pyc" -delete 2>/dev/null || true
    
    # Claude Flow caches
    rm -rf "$PROJECT_ROOT/.claude-flow/cache" || true
    rm -rf "$PROJECT_ROOT/.hive-mind/cache" || true
    rm -rf "$PROJECT_ROOT/.swarm/cache" || true
    
    # Serena caches
    rm -rf "$PROJECT_ROOT/.serena/cache" || true
    rm -rf "$PROJECT_ROOT/.cursor/cache" || true
    
    success "Caches cleared"
    
    # Force garbage collection
    log "Forcing garbage collection..."
    
    # macOS specific memory pressure relief
    sudo purge 2>/dev/null || {
        warn "Could not run 'sudo purge' - may need admin privileges"
    }
    
    # Force Node.js garbage collection on any remaining processes
    pkill -USR2 node 2>/dev/null || true
    
    success "Garbage collection attempted"
}

# Check current memory usage
check_memory_status() {
    log "📊 Checking current memory status..."
    
    if command -v vm_stat >/dev/null 2>&1; then
        # macOS memory check
        local vm_stat_output=$(vm_stat)
        local page_size=$(vm_stat | head -1 | grep -o '[0-9]*' | head -1)
        local free_pages=$(echo "$vm_stat_output" | grep "Pages free" | grep -o '[0-9]*')
        local total_memory=$(sysctl -n hw.memsize)
        local free_memory=$((free_pages * page_size))
        local used_memory=$((total_memory - free_memory))
        local usage_percent=$((used_memory * 100 / total_memory))
        
        log "Total Memory: $(numfmt --to=iec-i --suffix=B $total_memory)"
        log "Used Memory:  $(numfmt --to=iec-i --suffix=B $used_memory) (${usage_percent}%)"
        log "Free Memory:  $(numfmt --to=iec-i --suffix=B $free_memory)"
        
        if [ "$usage_percent" -gt 95 ]; then
            error "CRITICAL: Memory usage ${usage_percent}% - Emergency action required"
            return 1
        elif [ "$usage_percent" -gt 85 ]; then
            warn "WARNING: Memory usage ${usage_percent}% - Optimization recommended" 
            return 2
        else
            success "OK: Memory usage ${usage_percent}% - Within acceptable limits"
            return 0
        fi
    else
        warn "Cannot determine memory usage on this system"
        return 3
    fi
}

# Configure memory limits
configure_memory_limits() {
    log "🔧 Configuring memory limits..."
    
    # Set aggressive Node.js memory limits
    export NODE_OPTIONS="--max-old-space-size=1024 --gc-interval=50 --gc-global"
    export UV_THREADPOOL_SIZE=2
    
    # Create memory limits config if it doesn't exist
    if [ ! -f "$PROJECT_ROOT/config/memory-limits.json" ]; then
        warn "Memory limits config not found - using emergency defaults"
        mkdir -p "$PROJECT_ROOT/config"
        cat > "$PROJECT_ROOT/config/memory-limits.json" << 'EOF'
{
  "memoryManagement": {
    "globalLimits": {
      "totalBudget": "4GB",
      "emergencyThreshold": "15GB",
      "optimalTarget": "6GB"
    },
    "serviceLimits": {
      "claudeCode": { "maxMemory": "1GB" },
      "serenaMCP": { "maxMemory": "512MB" },
      "archonAPI": { "maxMemory": "1GB" }
    }
  }
}
EOF
    fi
    
    success "Memory limits configured"
}

# Start minimal services only
start_minimal_services() {
    log "🚀 Starting minimal services..."
    
    cd "$PROJECT_ROOT"
    
    # Create logs directory
    mkdir -p logs
    
    # Start only Serena MCP with memory constraints
    log "Starting Serena MCP with memory limits..."
    
    # Check if Serena is available
    if command -v npx >/dev/null 2>&1 && npx serena --help >/dev/null 2>&1; then
        nohup timeout 3600 npx serena start --memory-limit=512MB --port=8051 > logs/serena.log 2>&1 &
        SERENA_PID=$!
        log "Serena MCP started (PID: $SERENA_PID)"
        
        # Wait for service to be ready
        for i in {1..30}; do
            if curl -s http://localhost:8051/health >/dev/null 2>&1; then
                success "Serena MCP is ready"
                break
            fi
            sleep 1
        done
    else
        warn "Serena not available - skipping"
    fi
    
    # Save minimal runtime info
    mkdir -p "$PROJECT_ROOT/.runtime"
    cat > "$PROJECT_ROOT/.runtime/minimal.pid" << EOF
SERENA_PID=${SERENA_PID:-}
MODE=minimal
START_TIME=$(date '+%Y-%m-%d %H:%M:%S')
MEMORY_OPTIMIZED=true
EOF
    
    success "Minimal services started"
}

# Monitor memory continuously
start_memory_monitoring() {
    log "📊 Starting lightweight memory monitoring..."
    
    # Create a simple memory monitor
    cat > "$PROJECT_ROOT/.runtime/memory-watch.sh" << 'EOF'
#!/bin/bash
while true; do
    memory_percent=$(vm_stat | awk '/Pages free:/{free=$3} /Pages wired:/{wired=$3} /Pages active:/{active=$3} /Pages inactive:/{inactive=$3} END{total=free+wired+active+inactive; used=wired+active; printf "%.1f", used*100/total}')
    
    if (( $(echo "$memory_percent > 95" | bc -l) )); then
        echo "$(date): CRITICAL - Memory at ${memory_percent}%" >> logs/memory-alerts.log
        # Emergency cleanup
        pkill -f "claude-flow" || true
        pkill -f "archon-ui" || true
    elif (( $(echo "$memory_percent > 90" | bc -l) )); then
        echo "$(date): WARNING - Memory at ${memory_percent}%" >> logs/memory-alerts.log
    fi
    
    sleep 30
done
EOF
    
    chmod +x "$PROJECT_ROOT/.runtime/memory-watch.sh"
    nohup "$PROJECT_ROOT/.runtime/memory-watch.sh" > logs/memory-monitor.log 2>&1 &
    MONITOR_PID=$!
    
    echo "MONITOR_PID=$MONITOR_PID" >> "$PROJECT_ROOT/.runtime/minimal.pid"
    
    success "Memory monitoring started (PID: $MONITOR_PID)"
}

# Create stop script
create_stop_script() {
    cat > "$PROJECT_ROOT/scripts/stop-services.sh" << 'EOF'
#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🛑 Stopping all services..."

# Read PIDs if available
if [ -f "$PROJECT_ROOT/.runtime/minimal.pid" ]; then
    source "$PROJECT_ROOT/.runtime/minimal.pid"
    
    [ -n "${SERENA_PID:-}" ] && kill "$SERENA_PID" 2>/dev/null && echo "Stopped Serena (PID: $SERENA_PID)"
    [ -n "${MONITOR_PID:-}" ] && kill "$MONITOR_PID" 2>/dev/null && echo "Stopped Monitor (PID: $MONITOR_PID)"
fi

# Force kill any remaining processes
pkill -f "serena" || true
pkill -f "archon" || true
pkill -f "claude-flow" || true

# Clean up runtime files
rm -f "$PROJECT_ROOT/.runtime/minimal.pid"

echo "✅ All services stopped"
EOF
    
    chmod +x "$PROJECT_ROOT/scripts/stop-services.sh"
    success "Stop script created"
}

# Show final status
show_final_status() {
    echo
    log "📋 Final System Status:"
    echo
    
    # Check memory again
    check_memory_status || true
    
    echo
    printf "%-20s %-10s %-50s\n" "Service" "Status" "Endpoint"
    printf "%-20s %-10s %-50s\n" "-------" "------" "--------"
    
    # Check services
    if curl -s http://localhost:8051/health >/dev/null 2>&1; then
        printf "%-20s %-10s %-50s\n" "Serena MCP" "✅ UP" "http://localhost:8051"
    else
        printf "%-20s %-10s %-50s\n" "Serena MCP" "❌ DOWN" "Not running"
    fi
    
    printf "%-20s %-10s %-50s\n" "Archon API" "⏸️  PAUSED" "Memory optimization mode"
    printf "%-20s %-10s %-50s\n" "Claude Flow" "⏸️  PAUSED" "Memory optimization mode"
    
    echo
    log "💡 Next Steps:"
    log "  1. Monitor memory usage: tail -f logs/memory-alerts.log"
    log "  2. Start full services when memory permits: ./scripts/unified-startup.sh"
    log "  3. Stop all services: ./scripts/stop-services.sh"
    echo
}

# Main execution
main() {
    echo
    log "🚨 IMMEDIATE MEMORY OPTIMIZATION"
    log "Addressing critical 99%+ memory usage"
    echo
    
    # Check if running as emergency
    if [[ "${1:-}" == "--emergency" ]]; then
        emergency_cleanup
    fi
    
    # Current status
    if ! check_memory_status; then
        warn "Memory usage is critical - executing emergency cleanup"
        emergency_cleanup
        
        # Re-check after cleanup
        sleep 5
        if ! check_memory_status; then
            error "Memory still critical after cleanup - manual intervention required"
            exit 1
        fi
    fi
    
    # Configure limits
    configure_memory_limits
    
    # Start minimal services
    start_minimal_services
    
    # Start monitoring
    start_memory_monitoring
    
    # Create management scripts
    create_stop_script
    
    # Final status
    show_final_status
    
    success "🎉 Immediate memory optimization completed!"
    log "System now running in minimal mode with memory monitoring"
}

# Handle arguments
case "${1:-}" in
    "--emergency")
        main "$1"
        ;;
    "--cleanup-only")
        emergency_cleanup
        check_memory_status
        ;;
    "--status")
        check_memory_status
        ;;
    "--help"|"-h")
        echo "Usage: $0 [option]"
        echo
        echo "Options:"
        echo "  --emergency    Force emergency cleanup and restart"
        echo "  --cleanup-only Clean caches and stop services only"  
        echo "  --status       Check current memory status only"
        echo "  (no args)      Full optimization with minimal restart"
        ;;
    *)
        main
        ;;
esac