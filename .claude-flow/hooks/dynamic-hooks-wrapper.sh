#!/bin/bash

# Dynamic OS-Aware Claude Flow Hooks Wrapper
# Automatically detects OS and executes appropriate commands

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
HOOKS_DIR="$(dirname "$0")"
CONFIG_FILE="${HOOKS_DIR}/os-detection-hooks.json"
LOG_FILE="${HOOKS_DIR}/hooks.log"

# Logging function
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# OS Detection function
detect_os() {
    case "$(uname -s)" in
        Darwin*)
            echo "darwin"
            ;;
        Linux*)
            echo "linux"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            echo "win32"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

# Get memory available based on OS
get_memory_available() {
    local os_type="$1"
    
    case "$os_type" in
        darwin)
            vm_stat | awk '/Pages free/ {print int($3*16/1024)}'
            ;;
        linux)
            free -m | grep 'Mem:' | awk '{print $7}'
            ;;
        win32)
            # PowerShell command for Windows
            powershell -Command "(Get-WmiObject -Class Win32_OperatingSystem).FreePhysicalMemory / 1024" 2>/dev/null || echo "512"
            ;;
        *)
            echo "256" # Default fallback
            ;;
    esac
}

# Determine system mode based on available memory
get_system_mode() {
    local os_type="$1"
    local memory_mb="$2"
    
    # Load thresholds from config if jq is available
    if command -v jq &> /dev/null && [ -f "$CONFIG_FILE" ]; then
        local emergency_threshold=$(jq -r ".memory_thresholds.emergency.${os_type}" "$CONFIG_FILE" 2>/dev/null || echo "70")
        local limited_threshold=$(jq -r ".memory_thresholds.limited.${os_type}" "$CONFIG_FILE" 2>/dev/null || echo "200")
    else
        # Fallback thresholds
        case "$os_type" in
            darwin)
                emergency_threshold=70
                limited_threshold=200
                ;;
            linux)
                emergency_threshold=100
                limited_threshold=256
                ;;
            win32)
                emergency_threshold=128
                limited_threshold=512
                ;;
            *)
                emergency_threshold=100
                limited_threshold=300
                ;;
        esac
    fi
    
    if [ "$memory_mb" -lt "$emergency_threshold" ]; then
        echo "emergency"
    elif [ "$memory_mb" -lt "$limited_threshold" ]; then
        echo "limited"
    else
        echo "optimal"
    fi
}

# Execute OS-specific command
execute_os_command() {
    local command_type="$1"
    local os_type="$2"
    local memory_mb="$3"
    local system_mode="$4"
    shift 4
    local additional_args="$@"
    
    case "$command_type" in
        pre-task)
            case "$os_type" in
                darwin)
                    npx claude-flow@alpha hooks pre-task \
                        --memory-budget="${memory_mb}MB" \
                        --platform=darwin \
                        --cache-limit=25MB \
                        --mode="$system_mode" \
                        $additional_args
                    ;;
                linux)
                    npx claude-flow@alpha hooks pre-task \
                        --memory-budget="${memory_mb}MB" \
                        --platform=linux \
                        --cache-limit=50MB \
                        --mode="$system_mode" \
                        $additional_args
                    ;;
                win32)
                    npx claude-flow@alpha hooks pre-task \
                        --memory-budget="${memory_mb}MB" \
                        --platform=windows \
                        --cache-limit=100MB \
                        --mode="$system_mode" \
                        $additional_args
                    ;;
            esac
            ;;
        post-task)
            case "$os_type" in
                darwin)
                    npx claude-flow@alpha hooks post-task \
                        --cleanup-darwin \
                        --memory-recovery \
                        --gc-force \
                        --mode="$system_mode" \
                        $additional_args
                    ;;
                linux)
                    npx claude-flow@alpha hooks post-task \
                        --cleanup-linux \
                        --memory-recovery \
                        --gc-force \
                        --mode="$system_mode" \
                        $additional_args
                    ;;
                win32)
                    npx claude-flow@alpha hooks post-task \
                        --cleanup-windows \
                        --memory-recovery \
                        --gc-force \
                        --mode="$system_mode" \
                        $additional_args
                    ;;
            esac
            ;;
        memory-monitor)
            case "$os_type" in
                darwin)
                    npx claude-flow@alpha hooks memory-monitor \
                        --command='vm_stat' \
                        --interval=5 \
                        --threshold=99% \
                        --mode="$system_mode" \
                        $additional_args
                    ;;
                linux)
                    npx claude-flow@alpha hooks memory-monitor \
                        --command='free -m' \
                        --interval=5 \
                        --threshold=99% \
                        --mode="$system_mode" \
                        $additional_args
                    ;;
                win32)
                    npx claude-flow@alpha hooks memory-monitor \
                        --command='wmic OS get FreePhysicalMemory' \
                        --interval=5 \
                        --threshold=99% \
                        --mode="$system_mode" \
                        $additional_args
                    ;;
            esac
            ;;
        system-health)
            case "$os_type" in
                darwin)
                    echo -e "${BLUE}macOS System Health Check${NC}"
                    echo "Memory: $(vm_stat | awk '/Pages free/ {printf "%.1f MB free\n", $3*16/1024}')"
                    echo "Swap: $(sysctl vm.swapusage | awk '{print $7" "$8" "$9}')"
                    npx claude-flow@alpha hooks system-health \
                        --memory-cmd='vm_stat' \
                        --disk-cmd='df -h' \
                        --process-cmd='ps aux' \
                        --mode="$system_mode" \
                        $additional_args
                    ;;
                linux)
                    echo -e "${BLUE}Linux System Health Check${NC}"
                    free -m | head -2
                    npx claude-flow@alpha hooks system-health \
                        --memory-cmd='free -m' \
                        --disk-cmd='df -h' \
                        --process-cmd='ps aux' \
                        --mode="$system_mode" \
                        $additional_args
                    ;;
                win32)
                    echo -e "${BLUE}Windows System Health Check${NC}"
                    powershell -Command "Get-WmiObject -Class Win32_OperatingSystem | Select-Object FreePhysicalMemory,TotalVisibleMemorySize"
                    npx claude-flow@alpha hooks system-health \
                        --memory-cmd='wmic OS get FreePhysicalMemory' \
                        --disk-cmd='wmic logicaldisk' \
                        --process-cmd='tasklist' \
                        --mode="$system_mode" \
                        $additional_args
                    ;;
            esac
            ;;
        ansf-validate)
            npx claude-flow@alpha hooks validate-ansf \
                --phase3-target=97% \
                --neural-accuracy=88.7% \
                --memory-constrained="$system_mode" \
                --platform="$os_type" \
                $additional_args
            ;;
        *)
            echo -e "${RED}Unknown command type: $command_type${NC}"
            exit 1
            ;;
    esac
}

# Main function
main() {
    local command_type="$1"
    shift
    
    if [ -z "$command_type" ]; then
        echo -e "${RED}Usage: $0 <command_type> [additional_args]${NC}"
        echo -e "${YELLOW}Available commands: pre-task, post-task, memory-monitor, system-health, ansf-validate${NC}"
        exit 1
    fi
    
    # Detect OS
    local os_type=$(detect_os)
    log "Detected OS: $os_type"
    
    # Get available memory
    local memory_mb=$(get_memory_available "$os_type")
    log "Available memory: ${memory_mb}MB"
    
    # Determine system mode
    local system_mode=$(get_system_mode "$os_type" "$memory_mb")
    log "System mode: $system_mode"
    
    # Display status
    echo -e "${GREEN}🤖 Claude Flow Dynamic Hooks${NC}"
    echo -e "${BLUE}OS:${NC} $os_type"
    echo -e "${BLUE}Memory:${NC} ${memory_mb}MB available"
    echo -e "${BLUE}Mode:${NC} $system_mode"
    echo -e "${BLUE}Command:${NC} $command_type"
    echo ""
    
    # Execute command
    execute_os_command "$command_type" "$os_type" "$memory_mb" "$system_mode" "$@"
    
    log "Command completed: $command_type with args: $*"
}

# Run main function with all arguments
main "$@"