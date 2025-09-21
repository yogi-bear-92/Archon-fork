#!/bin/bash
# Container Fix Agent - Handles Docker and container-related issues
set -e

echo "🐳 Starting Container Fix Agent..."

# Initialize Claude Flow agent for container fixes
claude-flow agent spawn container-fix-agent \
  --description="Fix Docker containers, health checks, and dependencies" \
  --priority=high \
  --memory-limit=50MB || true

echo "📋 Analyzing container configuration..."

# Fix common Docker issues
fix_docker_health_checks() {
    echo "🔍 Checking Docker health configurations..."
    
    if [ -f "Dockerfile" ]; then
        # Add health check if missing
        if ! grep -q "HEALTHCHECK" Dockerfile; then
            echo "Adding basic health check to Dockerfile..."
            sed -i '/EXPOSE/a HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\\n  CMD curl -f http://localhost:${PORT:-8080}/health || exit 1' Dockerfile
        fi
        
        # Fix common Python container issues
        if grep -q "python" Dockerfile; then
            # Ensure proper Python buffering
            if ! grep -q "PYTHONUNBUFFERED" Dockerfile; then
                sed -i '/FROM/a ENV PYTHONUNBUFFERED=1' Dockerfile
            fi
            
            # Add proper signal handling
            if ! grep -q "STOPSIGNAL" Dockerfile; then
                echo "STOPSIGNAL SIGTERM" >> Dockerfile
            fi
        fi
    fi
}

# Fix docker-compose issues
fix_docker_compose() {
    echo "🔧 Fixing docker-compose configurations..."
    
    for compose_file in docker-compose.yml docker-compose.yaml; do
        if [ -f "$compose_file" ]; then
            echo "Fixing $compose_file..."
            
            # Add restart policies
            python3 -c "
import yaml
import sys

with open('$compose_file', 'r') as f:
    data = yaml.safe_load(f)

if 'services' in data:
    for service_name, service in data['services'].items():
        # Add restart policy
        if 'restart' not in service:
            service['restart'] = 'unless-stopped'
        
        # Add health checks
        if 'healthcheck' not in service and 'image' in service:
            if 'python' in service['image'] or 'fastapi' in service['image']:
                service['healthcheck'] = {
                    'test': ['CMD', 'curl', '-f', 'http://localhost:8080/health'],
                    'interval': '30s',
                    'timeout': '10s',
                    'retries': 3,
                    'start_period': '40s'
                }
        
        # Fix environment variables
        if 'environment' in service:
            env = service['environment']
            if isinstance(env, list):
                # Convert to dict for easier handling
                env_dict = {}
                for item in env:
                    if '=' in item:
                        key, value = item.split('=', 1)
                        env_dict[key] = value
                service['environment'] = env_dict

with open('$compose_file', 'w') as f:
    yaml.dump(data, f, default_flow_style=False, indent=2)
"
        fi
    done
}

# Fix container startup scripts
fix_startup_scripts() {
    echo "📜 Fixing container startup scripts..."
    
    if [ -f "start.sh" ] || [ -f "entrypoint.sh" ]; then
        for script in start.sh entrypoint.sh; do
            if [ -f "$script" ]; then
                # Make sure script is executable
                chmod +x "$script"
                
                # Add proper error handling
                if ! grep -q "set -e" "$script"; then
                    sed -i '1a set -e' "$script"
                fi
                
                # Add signal handling for graceful shutdown
                if ! grep -q "trap" "$script"; then
                    cat >> "$script" << 'EOF'

# Graceful shutdown handling
trap 'echo "Received SIGTERM, shutting down gracefully..."; kill -TERM $PID; wait $PID' TERM

# Start main process in background
"$@" &
PID=$!
wait $PID
EOF
                fi
            fi
        done
    fi
}

# Fix dependency installation issues
fix_container_deps() {
    echo "📦 Fixing container dependency issues..."
    
    if [ -f "requirements.txt" ]; then
        # Pin Python dependencies to avoid conflicts
        echo "Checking Python dependencies..."
        
        # Create a backup
        cp requirements.txt requirements.txt.bak
        
        # Fix common dependency issues
        python3 -c "
import sys
import re

with open('requirements.txt', 'r') as f:
    lines = f.readlines()

fixed_lines = []
for line in lines:
    line = line.strip()
    if not line or line.startswith('#'):
        fixed_lines.append(line + '\n')
        continue
    
    # Fix common issues
    if '>=' in line and not ('==' in line or '<' in line):
        # Add upper bound for better dependency resolution
        pkg_name = line.split('>=')[0].strip()
        version = line.split('>=')[1].strip()
        fixed_lines.append(f'{pkg_name}>={version},<{version.split(\".\")[0]}.{int(version.split(\".\")[1]) + 5}\n')
    else:
        fixed_lines.append(line + '\n')

with open('requirements.txt', 'w') as f:
    f.writelines(fixed_lines)
"
    fi
    
    # Fix package.json if exists
    if [ -f "package.json" ]; then
        echo "Checking Node.js dependencies..."
        
        # Fix common Node.js issues in containers
        node -e "
const fs = require('fs');
const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));

// Add container-friendly scripts
if (!pkg.scripts) pkg.scripts = {};
if (!pkg.scripts.start) pkg.scripts.start = 'node server.js';
if (!pkg.scripts['container-start']) {
    pkg.scripts['container-start'] = 'npm ci --only=production && npm start';
}

fs.writeFileSync('package.json', JSON.stringify(pkg, null, 2));
" 2>/dev/null || echo "Node.js not available, skipping package.json fixes"
    fi
}

# Main execution
echo "🚀 Executing container fixes..."

fix_docker_health_checks
fix_docker_compose
fix_startup_scripts
fix_container_deps

echo "✅ Container fixes completed successfully!"

# Report to Claude Flow
claude-flow hooks post-task \
  --task-id="container-fixes" \
  --status="completed" \
  --changes="Docker health checks, compose configs, startup scripts, dependencies" || true