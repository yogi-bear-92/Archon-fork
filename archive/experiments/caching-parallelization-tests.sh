#!/bin/bash

# Caching and Parallelization Experiments
# Testing various strategies to optimize system performance

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

EXPERIMENT_DIR="$(dirname "$0")"
RESULTS_FILE="${EXPERIMENT_DIR}/experiment-results.json"
CACHE_DIR="${EXPERIMENT_DIR}/.cache"

mkdir -p "$CACHE_DIR"

echo -e "${GREEN}🧪 Caching and Parallelization Experiments${NC}"
echo -e "${BLUE}Starting comprehensive performance optimization tests...${NC}"
echo ""

# Initialize results
echo '{"experiments": []}' > "$RESULTS_FILE"

log_experiment() {
    local name="$1"
    local result="$2"
    local duration="$3"
    local memory_start="$4"
    local memory_end="$5"
    
    # Update results JSON
    python3 -c "
import json
import sys
with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)
data['experiments'].append({
    'name': '$name',
    'result': '$result',
    'duration_ms': $duration,
    'memory_start_mb': $memory_start,
    'memory_end_mb': $memory_end,
    'timestamp': '$(date -u +%Y-%m-%dT%H:%M:%SZ)'
})
with open('$RESULTS_FILE', 'w') as f:
    json.dump(data, f, indent=2)
"
}

get_memory_mb() {
    vm_stat | awk '/Pages free/ {print int($3*16/1024)}'
}

run_experiment() {
    local name="$1"
    local command="$2"
    
    echo -e "${YELLOW}🔬 Experiment: $name${NC}"
    
    local memory_start=$(get_memory_mb)
    local start_time=$(date +%s%3N)
    
    eval "$command"
    local exit_code=$?
    
    local end_time=$(date +%s%3N)
    local memory_end=$(get_memory_mb)
    local duration=$((end_time - start_time))
    
    local result="success"
    if [ $exit_code -ne 0 ]; then
        result="failed"
    fi
    
    echo -e "  Duration: ${duration}ms"
    echo -e "  Memory: ${memory_start}MB → ${memory_end}MB"
    echo -e "  Result: ${result}"
    echo ""
    
    log_experiment "$name" "$result" "$duration" "$memory_start" "$memory_end"
}

# Experiment 1: Memory Caching Strategies
echo -e "${BLUE}=== Memory Caching Strategy Tests ===${NC}"

run_experiment "Baseline Memory Check" "vm_stat | head -5 > /dev/null"

run_experiment "Cached OS Detection" "
if [ ! -f '$CACHE_DIR/os_type' ]; then
    uname -s > '$CACHE_DIR/os_type'
fi
cat '$CACHE_DIR/os_type' > /dev/null
"

run_experiment "Cached Memory Stats" "
cache_file='$CACHE_DIR/memory_$(date +%s)'
if [ ! -f \"\$cache_file\" ] || [ \$(find \"\$cache_file\" -mtime +5s 2>/dev/null | wc -l) -eq 0 ]; then
    vm_stat > \"\$cache_file\"
fi
cat \"\$cache_file\" | head -5 > /dev/null
"

run_experiment "In-Memory Variable Caching" "
if [ -z \"\$CACHED_MEMORY\" ]; then
    export CACHED_MEMORY=\$(vm_stat | awk '/Pages free/ {print int(\$3*16/1024)}')
fi
echo \"Cached memory: \$CACHED_MEMORY MB\" > /dev/null
"

# Experiment 2: Parallel Command Execution
echo -e "${BLUE}=== Parallel Command Execution Tests ===${NC}"

run_experiment "Sequential Commands" "
vm_stat > /dev/null
sysctl vm.swapusage > /dev/null  
uptime > /dev/null
uname -a > /dev/null
"

run_experiment "Parallel Commands" "
{
    vm_stat > /dev/null &
    sysctl vm.swapusage > /dev/null &
    uptime > /dev/null &
    uname -a > /dev/null &
    wait
}
"

run_experiment "Background with Immediate Return" "
{
    vm_stat > '$CACHE_DIR/vm_stat.tmp' &
    sysctl vm.swapusage > '$CACHE_DIR/swap.tmp' &
    uptime > '$CACHE_DIR/uptime.tmp' &
    uname -a > '$CACHE_DIR/uname.tmp' &
} 2>/dev/null
wait
"

# Experiment 3: File I/O Optimization
echo -e "${BLUE}=== File I/O Optimization Tests ===${NC}"

run_experiment "Standard File Operations" "
echo 'test data' > '$CACHE_DIR/test1.txt'
cat '$CACHE_DIR/test1.txt' > /dev/null
rm '$CACHE_DIR/test1.txt'
"

run_experiment "Buffered File Operations" "
{
    echo 'test data 1' > '$CACHE_DIR/test2a.txt'
    echo 'test data 2' > '$CACHE_DIR/test2b.txt'  
    echo 'test data 3' > '$CACHE_DIR/test2c.txt'
} &
wait
cat '$CACHE_DIR/test2*.txt' > /dev/null
rm '$CACHE_DIR/test2*.txt'
"

run_experiment "Memory-Based Operations" "
export TEST_DATA_1='test data in memory 1'
export TEST_DATA_2='test data in memory 2'
export TEST_DATA_3='test data in memory 3'
echo \"\$TEST_DATA_1 \$TEST_DATA_2 \$TEST_DATA_3\" > /dev/null
"

# Experiment 4: Process Optimization
echo -e "${BLUE}=== Process Optimization Tests ===${NC}"

run_experiment "Subprocess Heavy" "
for i in {1..5}; do
    echo 'test' | wc -c > /dev/null
done
"

run_experiment "Built-in Operations" "
for i in {1..5}; do
    test_string='test'
    echo \${#test_string} > /dev/null
done
"

run_experiment "Function Caching" "
get_cached_length() {
    if [ -z \"\$CACHED_LENGTH\" ]; then
        export CACHED_LENGTH=4
    fi
    echo \$CACHED_LENGTH
}

for i in {1..5}; do
    get_cached_length > /dev/null
done
"

# Experiment 5: Parallel Memory Monitoring
echo -e "${BLUE}=== Parallel Memory Monitoring Tests ===${NC}"

run_experiment "Sequential Memory Checks" "
for i in {1..3}; do
    vm_stat | awk '/Pages free/ {print int(\$3*16/1024)}' > /dev/null
    sleep 0.1
done
"

run_experiment "Parallel Memory Monitoring" "
{
    vm_stat | awk '/Pages free/ {print int(\$3*16/1024)}' > '$CACHE_DIR/mem1.tmp' &
    vm_stat | awk '/Pages active/ {print int(\$3*16/1024)}' > '$CACHE_DIR/mem2.tmp' &
    vm_stat | awk '/Pages inactive/ {print int(\$3*16/1024)}' > '$CACHE_DIR/mem3.tmp' &
    wait
}
cat '$CACHE_DIR/mem*.tmp' > /dev/null
rm '$CACHE_DIR/mem*.tmp'
"

# Experiment 6: Smart Caching with TTL
echo -e "${BLUE}=== Smart Caching with TTL Tests ===${NC}"

run_experiment "No TTL Caching" "
cache_file='$CACHE_DIR/no_ttl_cache'
if [ ! -f \"\$cache_file\" ]; then
    vm_stat > \"\$cache_file\"
fi
cat \"\$cache_file\" | head -3 > /dev/null
"

run_experiment "TTL-Based Caching" "
cache_file='$CACHE_DIR/ttl_cache'
ttl=5  # 5 seconds TTL

if [ ! -f \"\$cache_file\" ] || [ \$((\$(date +%s) - \$(stat -f %m \"\$cache_file\" 2>/dev/null || echo 0))) -gt \$ttl ]; then
    vm_stat > \"\$cache_file\"
fi
cat \"\$cache_file\" | head -3 > /dev/null
"

run_experiment "Adaptive TTL Caching" "
cache_file='$CACHE_DIR/adaptive_cache'
current_memory=\$(vm_stat | awk '/Pages free/ {print int(\$3*16/1024)}')

# Shorter TTL when memory is low
if [ \$current_memory -lt 100 ]; then
    ttl=2
else
    ttl=10
fi

if [ ! -f \"\$cache_file\" ] || [ \$((\$(date +%s) - \$(stat -f %m \"\$cache_file\" 2>/dev/null || echo 0))) -gt \$ttl ]; then
    vm_stat > \"\$cache_file\"
fi
cat \"\$cache_file\" | head -3 > /dev/null
"

# Performance Analysis
echo -e "${BLUE}=== Performance Analysis ===${NC}"

echo -e "${GREEN}📊 Generating Performance Analysis...${NC}"

python3 -c "
import json
import statistics

with open('$RESULTS_FILE', 'r') as f:
    data = json.load(f)

experiments = data['experiments']
print('\\n🎯 Experiment Results Summary:')
print('=' * 50)

# Group by experiment type
by_type = {}
for exp in experiments:
    exp_type = exp['name'].split(' ')[0]
    if exp_type not in by_type:
        by_type[exp_type] = []
    by_type[exp_type].append(exp)

for exp_type, exps in by_type.items():
    durations = [e['duration_ms'] for e in exps]
    avg_duration = statistics.mean(durations)
    min_duration = min(durations)
    max_duration = max(durations)
    
    print(f'\\n{exp_type} Operations:')
    print(f'  Average: {avg_duration:.1f}ms')
    print(f'  Range: {min_duration}ms - {max_duration}ms')
    
    # Best performer in category
    best = min(exps, key=lambda x: x['duration_ms'])
    print(f'  Best: {best[\"name\"]} ({best[\"duration_ms\"]}ms)')

# Overall recommendations
print('\\n🚀 Performance Recommendations:')
print('=' * 50)

# Find caching improvements
cache_exps = [e for e in experiments if 'Cache' in e['name'] or 'cache' in e['name']]
if cache_exps:
    best_cache = min(cache_exps, key=lambda x: x['duration_ms'])
    print(f'✅ Best Caching Strategy: {best_cache[\"name\"]} ({best_cache[\"duration_ms\"]}ms)')

# Find parallelization improvements  
parallel_exps = [e for e in experiments if 'Parallel' in e['name']]
if parallel_exps:
    best_parallel = min(parallel_exps, key=lambda x: x['duration_ms'])
    print(f'✅ Best Parallelization: {best_parallel[\"name\"]} ({best_parallel[\"duration_ms\"]}ms)')

print(f'\\n📁 Detailed results saved to: $RESULTS_FILE')
"

echo ""
echo -e "${GREEN}✅ Caching and Parallelization Experiments Complete!${NC}"
echo -e "${BLUE}Results and recommendations available in experiment results.${NC}"

# Cleanup
rm -rf "$CACHE_DIR"