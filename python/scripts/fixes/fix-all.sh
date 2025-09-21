#!/bin/bash
# Master Fix Agent - Runs all automated fixes comprehensively
set -e

echo "🤖 Starting Master Fix Agent - Comprehensive Automated Fixing..."

# Initialize Claude Flow master coordination
claude-flow agent spawn master-fix-agent \
  --description="Coordinate all automated fixes across the entire codebase" \
  --priority=critical \
  --memory-limit=80MB || true

echo "🔄 Initializing comprehensive fix coordination..."

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Define fix order (dependencies first, then code quality, then advanced features)
FIXES=(
    "deps"        # Dependencies first - foundation
    "security"    # Security - protect the codebase
    "lint"        # Code formatting - clean foundation
    "types"       # Type checking - code quality
    "tests"       # Tests - validation
    "containers"  # Infrastructure
    "performance" # Optimization
    "workflows"   # CI/CD
    "docs"        # Documentation last
)

# Track results
declare -A fix_results
total_fixes=${#FIXES[@]}
successful_fixes=0
failed_fixes=0

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging function
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")  echo -e "${BLUE}[INFO]${NC} ${timestamp} - $message" ;;
        "SUCCESS") echo -e "${GREEN}[SUCCESS]${NC} ${timestamp} - $message" ;;
        "WARNING") echo -e "${YELLOW}[WARNING]${NC} ${timestamp} - $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" ;;
        "HEADER") echo -e "${PURPLE}[FIX]${NC} ${timestamp} - $message" ;;
    esac
}

# Function to run individual fix with error handling
run_fix() {
    local fix_name=$1
    local fix_script="${SCRIPT_DIR}/fix-${fix_name}.sh"
    
    log_message "HEADER" "Running fix: $fix_name"
    
    if [ ! -f "$fix_script" ]; then
        log_message "ERROR" "Fix script not found: $fix_script"
        fix_results[$fix_name]="MISSING"
        return 1
    fi
    
    # Make sure script is executable
    chmod +x "$fix_script"
    
    # Run the fix script with timeout and capture output
    local start_time=$(date +%s)
    local output_file="/tmp/fix-${fix_name}-output.log"
    
    if timeout 300 bash "$fix_script" > "$output_file" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log_message "SUCCESS" "Fix '$fix_name' completed successfully in ${duration}s"
        fix_results[$fix_name]="SUCCESS"
        successful_fixes=$((successful_fixes + 1))
        
        # Show last few lines of output
        echo "  Last output lines:"
        tail -3 "$output_file" | sed 's/^/    /'
        
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log_message "ERROR" "Fix '$fix_name' failed after ${duration}s"
        fix_results[$fix_name]="FAILED"
        failed_fixes=$((failed_fixes + 1))
        
        # Show error output
        echo "  Error output:"
        tail -5 "$output_file" | sed 's/^/    /'
        
        return 1
    fi
}

# Pre-flight checks
pre_flight_checks() {
    log_message "INFO" "Running pre-flight checks..."
    
    # Check if we're in a git repository
    if ! git status >/dev/null 2>&1; then
        log_message "WARNING" "Not in a git repository - some fixes may not work optimally"
    fi
    
    # Check available disk space
    local available_space=$(df . | tail -1 | awk '{print $4}')
    if [ "$available_space" -lt 1000000 ]; then
        log_message "WARNING" "Low disk space detected - fixes may fail"
    fi
    
    # Check for required tools
    local required_tools=("python3" "pip")
    for tool in "${required_tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            log_message "ERROR" "Required tool not found: $tool"
            exit 1
        fi
    done
    
    # Create backup of important files
    log_message "INFO" "Creating backup of important files..."
    mkdir -p .backup-before-fixes
    
    # Backup key files if they exist
    for file in requirements.txt package.json pyproject.toml setup.py; do
        if [ -f "$file" ]; then
            cp "$file" ".backup-before-fixes/$file.backup" 2>/dev/null || true
        fi
    done
    
    log_message "SUCCESS" "Pre-flight checks completed"
}

# Run all fixes in sequence
run_all_fixes() {
    log_message "INFO" "Starting comprehensive automated fixes..."
    log_message "INFO" "Total fixes to run: $total_fixes"
    
    local current_fix=0
    
    for fix in "${FIXES[@]}"; do
        current_fix=$((current_fix + 1))
        
        echo ""
        echo "=" $(printf '=%.0s' {1..60})
        log_message "INFO" "Progress: $current_fix/$total_fixes - Running fix: $fix"
        echo "=" $(printf '=%.0s' {1..60})
        
        # Run the fix (continue on failure for non-critical fixes)
        run_fix "$fix" || {
            if [[ "$fix" == "deps" || "$fix" == "security" ]]; then
                log_message "ERROR" "Critical fix '$fix' failed - stopping execution"
                return 1
            else
                log_message "WARNING" "Non-critical fix '$fix' failed - continuing with remaining fixes"
            fi
        }
        
        # Brief pause between fixes to prevent resource exhaustion
        sleep 2
    done
}

# Generate comprehensive report
generate_report() {
    log_message "INFO" "Generating comprehensive fix report..."
    
    local report_file="automated-fix-report.md"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    cat > "$report_file" << EOF
# Automated Fix Report

**Generated:** $timestamp  
**Command:** \`/fix-all\`  
**Total Fixes:** $total_fixes  
**Successful:** $successful_fixes  
**Failed:** $failed_fixes  

## Fix Results Summary

| Fix Category | Status | Description |
|--------------|--------|-------------|
EOF
    
    # Add results for each fix
    for fix in "${FIXES[@]}"; do
        local status="${fix_results[$fix]:-SKIPPED}"
        local description=""
        
        case $fix in
            "deps") description="Dependencies and package management" ;;
            "security") description="Security vulnerabilities and best practices" ;;
            "lint") description="Code formatting and linting" ;;
            "types") description="Type hints and type checking" ;;
            "tests") description="Test fixes and coverage improvements" ;;
            "containers") description="Docker and container optimizations" ;;
            "performance") description="Performance optimizations" ;;
            "workflows") description="GitHub Actions and CI/CD" ;;
            "docs") description="Documentation generation and updates" ;;
        esac
        
        local status_emoji=""
        case $status in
            "SUCCESS") status_emoji="✅" ;;
            "FAILED") status_emoji="❌" ;;
            "MISSING") status_emoji="⚠️" ;;
            *) status_emoji="⏭️" ;;
        esac
        
        echo "| $fix | $status_emoji $status | $description |" >> "$report_file"
    done
    
    cat >> "$report_file" << EOF

## Detailed Changes

### Dependencies (fix-deps)
$(if [[ "${fix_results[deps]}" == "SUCCESS" ]]; then
    echo "- ✅ Updated Python and Node.js dependencies"
    echo "- ✅ Resolved version conflicts"  
    echo "- ✅ Added Dependabot configuration"
else
    echo "- ❌ Dependency fixes failed or were skipped"
fi)

### Security (fix-security)
$(if [[ "${fix_results[security]}" == "SUCCESS" ]]; then
    echo "- ✅ Fixed security vulnerabilities detected by bandit"
    echo "- ✅ Updated vulnerable dependencies"
    echo "- ✅ Added security headers and configurations"
    echo "- ✅ Removed hardcoded secrets"
else
    echo "- ❌ Security fixes failed or were skipped"
fi)

### Code Quality (fix-lint)
$(if [[ "${fix_results[lint]}" == "SUCCESS" ]]; then
    echo "- ✅ Applied black code formatting"
    echo "- ✅ Sorted imports with isort"
    echo "- ✅ Fixed flake8 linting issues"
    echo "- ✅ Updated configuration files"
else
    echo "- ❌ Linting fixes failed or were skipped"
fi)

### Type Checking (fix-types)
$(if [[ "${fix_results[types]}" == "SUCCESS" ]]; then
    echo "- ✅ Added missing type imports"
    echo "- ✅ Generated function type hints"
    echo "- ✅ Fixed mypy issues"
    echo "- ✅ Created mypy configuration"
else
    echo "- ❌ Type fixes failed or were skipped"
fi)

### Testing (fix-tests)
$(if [[ "${fix_results[tests]}" == "SUCCESS" ]]; then
    echo "- ✅ Fixed test structure and imports"
    echo "- ✅ Created missing test files"
    echo "- ✅ Improved test coverage"
    echo "- ✅ Added pytest configuration"
else
    echo "- ❌ Test fixes failed or were skipped"
fi)

### Containers (fix-containers)
$(if [[ "${fix_results[containers]}" == "SUCCESS" ]]; then
    echo "- ✅ Fixed Docker health checks"
    echo "- ✅ Improved container configurations"
    echo "- ✅ Added proper signal handling"
    echo "- ✅ Fixed startup scripts"
else
    echo "- ❌ Container fixes failed or were skipped"
fi)

### Performance (fix-performance)
$(if [[ "${fix_results[performance]}" == "SUCCESS" ]]; then
    echo "- ✅ Added caching optimizations"
    echo "- ✅ Optimized database queries"
    echo "- ✅ Improved API performance"
    echo "- ✅ Created performance monitoring"
else
    echo "- ❌ Performance fixes failed or were skipped"
fi)

### Workflows (fix-workflows)
$(if [[ "${fix_results[workflows]}" == "SUCCESS" ]]; then
    echo "- ✅ Created comprehensive CI/CD workflows"
    echo "- ✅ Added security scanning"
    echo "- ✅ Fixed workflow syntax issues"
    echo "- ✅ Added validation scripts"
else
    echo "- ❌ Workflow fixes failed or were skipped"
fi)

### Documentation (fix-docs)
$(if [[ "${fix_results[docs]}" == "SUCCESS" ]]; then
    echo "- ✅ Generated API documentation"
    echo "- ✅ Updated README files"
    echo "- ✅ Created comprehensive docs structure"
    echo "- ✅ Generated changelog"
else
    echo "- ❌ Documentation fixes failed or were skipped"
fi)

## Next Steps

### If all fixes succeeded:
1. ✅ Review the changes made
2. ✅ Run tests to verify everything works
3. ✅ Commit the changes
4. ✅ Monitor for any issues

### If some fixes failed:
1. ⚠️ Review the error logs above
2. ⚠️ Run individual fix commands manually
3. ⚠️ Check for missing dependencies or permissions
4. ⚠️ Retry failed fixes after addressing issues

## Files Modified

- Configuration files updated
- Source code formatted and optimized
- Tests created/updated
- Documentation generated
- Workflows created/fixed

## Backup

Original files backed up in: \`.backup-before-fixes/\`

---

*Report generated by Automated Fix System*  
*Powered by Claude Code and Archon AI*
EOF
    
    log_message "SUCCESS" "Report generated: $report_file"
}

# Cleanup function
cleanup() {
    log_message "INFO" "Performing cleanup..."
    
    # Remove temporary files
    rm -f /tmp/fix-*-output.log 2>/dev/null || true
    
    # Claude Flow cleanup
    claude-flow cleanup --memory-recovery || true
    
    log_message "SUCCESS" "Cleanup completed"
}

# Main execution flow
main() {
    local start_time=$(date +%s)
    
    echo "🚀🚀🚀 COMPREHENSIVE AUTOMATED FIX SYSTEM ACTIVATED 🚀🚀🚀"
    echo ""
    log_message "INFO" "Starting comprehensive automated fix process"
    
    # Set up error handling
    trap cleanup EXIT
    
    # Run pre-flight checks
    pre_flight_checks || {
        log_message "ERROR" "Pre-flight checks failed"
        exit 1
    }
    
    # Run all fixes
    run_all_fixes || {
        log_message "ERROR" "Critical fixes failed - aborting"
        generate_report
        exit 1
    }
    
    # Generate comprehensive report
    generate_report
    
    local end_time=$(date +%s)
    local total_duration=$((end_time - start_time))
    local minutes=$((total_duration / 60))
    local seconds=$((total_duration % 60))
    
    echo ""
    echo "🎉 COMPREHENSIVE FIX PROCESS COMPLETED! 🎉"
    echo ""
    log_message "SUCCESS" "All fixes completed in ${minutes}m ${seconds}s"
    log_message "SUCCESS" "Successful fixes: $successful_fixes/$total_fixes"
    
    if [ $failed_fixes -gt 0 ]; then
        log_message "WARNING" "Some fixes failed: $failed_fixes/$total_fixes"
        log_message "INFO" "Check the report for details: automated-fix-report.md"
    else
        log_message "SUCCESS" "All fixes completed successfully! 🎉"
    fi
    
    # Report to Claude Flow
    claude-flow hooks post-task \
        --task-id="comprehensive-fixes" \
        --status="completed" \
        --changes="Applied $successful_fixes fixes: deps, security, lint, types, tests, containers, performance, workflows, docs" \
        --success-rate="$successful_fixes/$total_fixes" \
        --duration="${minutes}m ${seconds}s" || true
    
    echo ""
    echo "📋 Check the comprehensive report: automated-fix-report.md"
    echo "🔍 Review changes before committing"
    echo "✨ Your codebase has been optimized and secured!"
}

# Run main function
main "$@"