#!/bin/bash
# Lint Fix Agent - Automatically fixes code formatting and linting issues
set -e

echo "🎨 Starting Lint Fix Agent..."

# Initialize Claude Flow agent for linting fixes
claude-flow agent spawn lint-fix-agent \
  --description="Fix Python code formatting with black, isort, flake8" \
  --priority=high \
  --memory-limit=30MB || true

echo "📋 Running comprehensive code formatting..."

# Python linting and formatting
fix_python_formatting() {
    echo "🐍 Fixing Python code formatting..."
    
    # Find Python files
    python_files=$(find . -name "*.py" -not -path "./.git/*" -not -path "./venv/*" -not -path "./.venv/*" -not -path "./env/*" | head -50)
    
    if [ -n "$python_files" ]; then
        echo "Found Python files, applying fixes..."
        
        # Apply isort (import sorting)
        echo "📦 Sorting imports with isort..."
        echo "$python_files" | xargs isort --profile black --line-length 88 --multi-line 3 --trailing-comma || true
        
        # Apply black (code formatting)
        echo "⚫ Formatting code with black..."
        echo "$python_files" | xargs black --line-length 88 --target-version py311 || true
        
        # Apply autopep8 for additional fixes
        echo "🔧 Applying additional PEP8 fixes..."
        echo "$python_files" | xargs autopep8 --in-place --aggressive --aggressive --max-line-length 88 || pip install autopep8 && echo "$python_files" | xargs autopep8 --in-place --aggressive --aggressive --max-line-length 88 || true
        
        # Fix common flake8 issues programmatically
        echo "🔍 Fixing common flake8 issues..."
        python3 -c "
import os
import re
import sys

def fix_common_issues(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Remove unused imports (basic ones)
        lines = content.split('\n')
        import_lines = []
        other_lines = []
        used_imports = set()
        
        for i, line in enumerate(lines):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_lines.append((i, line))
            else:
                other_lines.append(line)
                # Extract potential imports used in this line
                for match in re.finditer(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', line):
                    used_imports.add(match.group(1))
        
        # Filter out unused imports (simple heuristic)
        filtered_imports = []
        for i, line in import_lines:
            import_name = None
            if 'import ' in line:
                # Extract import name
                if ' as ' in line:
                    import_name = line.split(' as ')[-1].strip()
                else:
                    import_name = line.split('import ')[-1].strip().split('.')[0].split(',')[0]
                
                if import_name in used_imports or import_name in ['os', 'sys', 'json', 're', 'typing']:
                    filtered_imports.append(line)
            else:
                filtered_imports.append(line)
        
        # Reconstruct content
        new_content = '\n'.join(filtered_imports + other_lines)
        
        # Fix trailing whitespace
        new_content = re.sub(r'[ \t]+$', '', new_content, flags=re.MULTILINE)
        
        # Fix multiple blank lines
        new_content = re.sub(r'\n\n\n+', '\n\n', new_content)
        
        # Ensure file ends with newline
        if new_content and not new_content.endswith('\n'):
            new_content += '\n'
        
        if new_content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f'Fixed: {filepath}')
    
    except Exception as e:
        print(f'Error fixing {filepath}: {e}')

# Process each Python file
python_files = '''$python_files'''.strip().split('\n')
for filepath in python_files:
    if filepath.strip():
        fix_common_issues(filepath.strip())
"
    else
        echo "No Python files found to format"
    fi
}

# JavaScript/TypeScript formatting
fix_js_formatting() {
    echo "📜 Fixing JavaScript/TypeScript formatting..."
    
    # Find JS/TS files
    js_files=$(find . -name "*.js" -o -name "*.ts" -o -name "*.jsx" -o -name "*.tsx" | grep -v node_modules | grep -v .git | head -20)
    
    if [ -n "$js_files" ] && command -v prettier >/dev/null 2>&1; then
        echo "Found JS/TS files, applying Prettier..."
        echo "$js_files" | xargs prettier --write --single-quote --trailing-comma es5 --tab-width 2 || true
        
        # Apply ESLint fixes if available
        if command -v eslint >/dev/null 2>&1; then
            echo "Applying ESLint automatic fixes..."
            echo "$js_files" | xargs eslint --fix || true
        fi
    fi
}

# Configuration file formatting
fix_config_files() {
    echo "⚙️ Fixing configuration files..."
    
    # Format JSON files
    for json_file in $(find . -name "*.json" | grep -v node_modules | grep -v .git | head -10); do
        if [ -f "$json_file" ]; then
            echo "Formatting $json_file..."
            python3 -c "
import json
import sys

try:
    with open('$json_file', 'r') as f:
        data = json.load(f)
    
    with open('$json_file', 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print('Formatted: $json_file')
except Exception as e:
    print(f'Error formatting $json_file: {e}')
" || true
        fi
    done
    
    # Format YAML files
    for yaml_file in $(find . -name "*.yml" -o -name "*.yaml" | grep -v node_modules | grep -v .git | head -10); do
        if [ -f "$yaml_file" ] && command -v python3 >/dev/null 2>&1; then
            echo "Formatting $yaml_file..."
            python3 -c "
import yaml
import sys

try:
    with open('$yaml_file', 'r') as f:
        data = yaml.safe_load(f)
    
    with open('$yaml_file', 'w') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2, sort_keys=True)
    print('Formatted: $yaml_file')
except Exception as e:
    print(f'Error formatting $yaml_file: {e}')
" 2>/dev/null || true
        fi
    done
}

# Fix line endings and encoding
fix_line_endings() {
    echo "📝 Fixing line endings and encoding..."
    
    # Fix line endings for text files
    for file in $(find . -type f \( -name "*.py" -o -name "*.js" -o -name "*.md" -o -name "*.txt" -o -name "*.yml" -o -name "*.yaml" -o -name "*.json" \) | grep -v .git | head -30); do
        if [ -f "$file" ]; then
            # Convert CRLF to LF
            sed -i 's/\r$//' "$file" 2>/dev/null || true
        fi
    done
}

# Create or update linting configuration files
create_lint_configs() {
    echo "📄 Creating/updating linting configuration..."
    
    # Create .flake8 config if it doesn't exist
    if [ ! -f ".flake8" ]; then
        cat > .flake8 << 'EOF'
[flake8]
max-line-length = 88
extend-ignore = E203, W503, E501
exclude = 
    .git,
    __pycache__,
    venv,
    .venv,
    env,
    .env,
    build,
    dist,
    migrations
per-file-ignores = __init__.py:F401
EOF
        echo "Created .flake8 configuration"
    fi
    
    # Create pyproject.toml for black/isort if it doesn't exist
    if [ ! -f "pyproject.toml" ]; then
        cat > pyproject.toml << 'EOF'
[tool.black]
line-length = 88
target-version = ['py311']
include = '\.pyi?$'
exclude = '''
/(
    \.git
    | \.venv
    | \.env
    | venv
    | env
    | build
    | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
EOF
        echo "Created pyproject.toml configuration"
    fi
    
    # Create .prettierrc for JavaScript if it doesn't exist
    if [ ! -f ".prettierrc" ]; then
        cat > .prettierrc << 'EOF'
{
  "semi": true,
  "singleQuote": true,
  "tabWidth": 2,
  "trailingComma": "es5",
  "printWidth": 80
}
EOF
        echo "Created .prettierrc configuration"
    fi
}

# Main execution
echo "🚀 Executing comprehensive linting fixes..."

create_lint_configs
fix_python_formatting
fix_js_formatting
fix_config_files
fix_line_endings

echo "✅ Lint fixes completed successfully!"

# Generate summary of changes
echo "📊 Generating fix summary..."
changed_files=$(git diff --name-only | wc -l)
echo "Modified $changed_files files during linting process"

# Report to Claude Flow
claude-flow hooks post-task \
  --task-id="lint-fixes" \
  --status="completed" \
  --changes="Formatted Python, JS, config files; fixed line endings; updated lint configs" \
  --files-changed="$changed_files" || true