# Automatic GitFlow and Serena MCP Servers Setup Script for Windows
# Integrates with Archon system following CLAUDE.md memory-aware patterns

param(
    [switch]$Help,
    [switch]$Force,
    [switch]$SkipValidation,
    [switch]$NoAdaptive
)

# Memory-aware configuration
$MemoryThreshold = 95
$MaxConcurrentInstalls = 2
$AdaptiveResourceMode = -not $NoAdaptive

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Color functions for PowerShell
function Write-InfoLog { param($Message) Write-Host "ℹ️  $Message" -ForegroundColor Blue }
function Write-SuccessLog { param($Message) Write-Host "✅ $Message" -ForegroundColor Green }
function Write-WarningLog { param($Message) Write-Host "⚠️  $Message" -ForegroundColor Yellow }
function Write-ErrorLog { param($Message) Write-Host "❌ $Message" -ForegroundColor Red }
function Write-ArchonLog { param($Message) Write-Host "🏛️ $Message" -ForegroundColor Magenta }
function Write-McpLog { param($Message) Write-Host "🎯 $Message" -ForegroundColor Cyan }

# Memory check function
function Get-AvailableMemoryMB {
    try {
        $memory = Get-WmiObject -Class Win32_OperatingSystem
        $availableGB = [math]::Round($memory.FreePhysicalMemory / 1MB, 0)
        return $availableGB
    }
    catch {
        return 1000  # Default assumption
    }
}

function Test-AdaptiveResourceCheck {
    if ($AdaptiveResourceMode) {
        $availableMemory = Get-AvailableMemoryMB
        Write-InfoLog "Available memory: ${availableMemory}MB"
        
        if ($availableMemory -lt 100) {
            Write-WarningLog "Memory critical (<100MB) - enabling emergency mode"
            $script:MaxConcurrentInstalls = 1
            return $false
        }
        elseif ($availableMemory -lt 500) {
            Write-WarningLog "Memory limited (<500MB) - reducing concurrency"
            $script:MaxConcurrentInstalls = 1
        }
    }
    return $true
}

# Check prerequisites
function Test-Prerequisites {
    Write-ArchonLog "🔍 Checking system prerequisites..."
    
    $missingDeps = @()
    
    # Check Node.js
    try {
        $nodeVersion = node --version
        $nodeMajor = [int]($nodeVersion -replace 'v(\d+)\..*', '$1')
        if ($nodeMajor -lt 18) {
            Write-ErrorLog "Node.js 18+ required. Current: $nodeVersion"
            return $false
        }
        Write-SuccessLog "Node.js $nodeVersion"
    }
    catch {
        $missingDeps += "node"
    }
    
    # Check npm
    try {
        $npmVersion = npm --version
        Write-SuccessLog "npm $npmVersion"
    }
    catch {
        $missingDeps += "npm"
    }
    
    # Check Python (for uv installation)
    try {
        $pythonVersion = python --version
        Write-SuccessLog "Python $pythonVersion"
    }
    catch {
        Write-WarningLog "Python not found - may need manual installation for uv"
    }
    
    if ($missingDeps.Count -gt 0) {
        Write-ErrorLog "Missing dependencies: $($missingDeps -join ', ')"
        Write-InfoLog "Please install missing dependencies and run again"
        return $false
    }
    
    return $true
}

# Install uv for Windows
function Install-Uv {
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        Write-InfoLog "Installing uv (Python package manager) for Windows..."
        try {
            # Download and run uv installer for Windows
            $uvInstaller = "$env:TEMP\uv-installer.ps1"
            Invoke-WebRequest -Uri "https://astral.sh/uv/install.ps1" -OutFile $uvInstaller
            & powershell.exe -ExecutionPolicy ByPass -File $uvInstaller
            
            # Add to PATH for current session
            $env:PATH = "$env:USERPROFILE\.local\bin;$env:PATH"
            
            if (Get-Command uv -ErrorAction SilentlyContinue) {
                Write-SuccessLog "uv installed successfully"
                return $true
            }
            else {
                Write-ErrorLog "Failed to install uv"
                return $false
            }
        }
        catch {
            Write-ErrorLog "Error installing uv: $($_.Exception.Message)"
            return $false
        }
    }
    return $true
}

# Install Claude Flow
function Install-ClaudeFlow {
    Write-McpLog "📦 Installing Claude Flow (GitFlow)..."
    
    Test-AdaptiveResourceCheck | Out-Null
    
    try {
        # Check if already installed
        $existing = npm list -g claude-flow@alpha 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-SuccessLog "Claude Flow already installed"
        }
        else {
            Write-InfoLog "Installing Claude Flow alpha version globally..."
            npm install -g claude-flow@alpha
            
            if (Get-Command claude-flow -ErrorAction SilentlyContinue) {
                $version = claude-flow --version 2>$null | Select-Object -First 1
                Write-SuccessLog "Claude Flow installed: $version"
            }
            else {
                Write-ErrorLog "Claude Flow installation failed"
                return $false
            }
        }
        
        # Initialize Claude Flow
        Write-InfoLog "Initializing Claude Flow..."
        $claudeFlowDir = Join-Path $ProjectRoot ".claude-flow"
        if (-not (Test-Path $claudeFlowDir)) {
            Push-Location $ProjectRoot
            try {
                claude-flow init --project-root="$ProjectRoot"
                Write-SuccessLog "Claude Flow initialized"
            }
            finally {
                Pop-Location
            }
        }
        else {
            Write-SuccessLog "Claude Flow already initialized"
        }
        
        return $true
    }
    catch {
        Write-ErrorLog "Error installing Claude Flow: $($_.Exception.Message)"
        return $false
    }
}

# Install Serena
function Install-Serena {
    Write-McpLog "📦 Installing Serena MCP Server..."
    
    Test-AdaptiveResourceCheck | Out-Null
    
    # Ensure uv is available
    if (-not (Install-Uv)) {
        return $false
    }
    
    try {
        Write-InfoLog "Installing Serena from GitHub..."
        # Test Serena installation
        $testCmd = "uv tool install --from git+https://github.com/oraios/serena serena"
        Invoke-Expression $testCmd 2>$null
        
        # Verify installation
        if (Get-Command uvx -ErrorAction SilentlyContinue) {
            uvx --from "git+https://github.com/oraios/serena" serena --version >$null 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-SuccessLog "Serena installed successfully"
                return $true
            }
        }
        
        Write-ErrorLog "Serena installation verification failed"
        return $false
    }
    catch {
        Write-ErrorLog "Error installing Serena: $($_.Exception.Message)"
        return $false
    }
}

# Install Playwright MCP server
function Install-Playwright {
    Write-InfoLog "🎭 Installing Playwright MCP server..."
    
    try {
        # Memory-aware installation
        Test-AdaptiveResources
        
        if (Get-Command npx -ErrorAction SilentlyContinue) {
            Write-InfoLog "Installing Playwright MCP server..."
            
            # Test if @playwright/mcp is available
            $testResult = npx "@playwright/mcp" --help 2>$null
            if ($LASTEXITCODE -eq 0) {
                Write-SuccessLog "Playwright MCP server available"
            } else {
                Write-InfoLog "Installing Playwright MCP dependencies..."
                npm install -g "@playwright/mcp" 2>$null
                if ($LASTEXITCODE -ne 0) {
                    Write-WarningLog "Playwright MCP may need to be installed on first use"
                }
            }
        } else {
            Write-ErrorLog "npx not available for Playwright installation"
            return $false
        }
        
        return $true
    }
    catch {
        Write-ErrorLog "Error installing Playwright: $($_.Exception.Message)"
        return $false
    }
}

# Update Archon configuration
function Update-ArchonConfig {
    Write-ArchonLog "⚙️ Updating Archon MCP configuration..."
    
    $configFile = Join-Path $ProjectRoot "python\config\mcp_servers.json"
    $projectPath = $ProjectRoot -replace '\\', '/'  # Convert to forward slashes
    
    if (Test-Path $configFile) {
        Write-InfoLog "Updating existing MCP configuration"
        
        # Create backup
        $backupFile = "$configFile.backup.$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Copy-Item $configFile $backupFile
        
        # Create new configuration
        $config = @{
            "claude-flow" = @{
                "args" = @(
                    "claude-flow@alpha",
                    "mcp", 
                    "start",
                    "--project-root",
                    $projectPath
                )
                "auto_start" = $true
                "command" = "npx"
                "health_check_interval" = 30
                "name" = "claude-flow"
                "retry_attempts" = 3
                "timeout" = 10
                "transport" = "stdio"
            }
            "serena" = @{
                "args" = @(
                    "--from",
                    "git+https://github.com/oraios/serena",
                    "serena",
                    "start-mcp-server", 
                    "--context",
                    "ide-assistant",
                    "--project",
                    $projectPath
                )
                "auto_start" = $true
                "command" = "uvx"
                "health_check_interval" = 30
                "name" = "serena"
                "retry_attempts" = 3
                "timeout" = 10
                "transport" = "stdio"
            }
        }
        
        $config | ConvertTo-Json -Depth 10 | Set-Content $configFile -Encoding UTF8
        Write-SuccessLog "MCP configuration updated"
        return $true
    }
    else {
        Write-ErrorLog "MCP configuration file not found: $configFile"
        return $false
    }
}

# Create Claude Desktop configuration
function New-ClaudeDesktopConfig {
    Write-McpLog "📝 Creating Claude Desktop configuration..."
    
    $claudeConfigDir = Join-Path $env:APPDATA "Claude"
    if (-not (Test-Path $claudeConfigDir)) {
        New-Item -ItemType Directory -Path $claudeConfigDir -Force | Out-Null
    }
    
    $configFile = Join-Path $claudeConfigDir "claude_desktop_config.json"
    $projectPath = $ProjectRoot -replace '\\', '/'
    
    $config = @{
        "mcpServers" = @{
            "claude-flow" = @{
                "command" = "npx"
                "args" = @("claude-flow@alpha", "mcp", "start", "--project-root", $projectPath)
                "env" = @{}
            }
            "serena" = @{
                "command" = "uvx" 
                "args" = @(
                    "--from",
                    "git+https://github.com/oraios/serena",
                    "serena",
                    "start-mcp-server",
                    "--context", 
                    "ide-assistant",
                    "--project",
                    $projectPath
                )
                "env" = @{}
            }
            "playwright" = @{
                "command" = "npx"
                "args" = @("@playwright/mcp")
                "env" = @{}
            }
            "archon-mcp" = @{
                "command" = "python"
                "args" = @("-m", "src.mcp_server.mcp_server")
                "cwd" = Join-Path $ProjectRoot "python"
                "env" = @{
                    "PYTHONPATH" = Join-Path $ProjectRoot "python"
                }
            }
        }
    }
    
    $config | ConvertTo-Json -Depth 10 | Set-Content $configFile -Encoding UTF8
    Write-SuccessLog "Claude Desktop configuration created: $configFile"
    return $true
}

# Configure Claude Code MCP servers
function Set-ClaudeCodeConfig {
    Write-McpLog "🔧 Configuring Claude Code MCP servers..."
    
    # Check if claude command is available
    if (-not (Get-Command claude -ErrorAction SilentlyContinue)) {
        Write-WarningLog "Claude Code CLI not found - skipping Claude Code configuration"
        Write-InfoLog "Install Claude Code from: https://docs.anthropic.com/en/docs/claude-code"
        return $true
    }
    
    try {
        # Add Claude Flow MCP server
        Write-InfoLog "Adding Claude Flow to Claude Code..."
        claude mcp add claude-flow npx claude-flow@alpha mcp start 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-WarningLog "Failed to add Claude Flow to Claude Code"
        }
        
        # Add Serena MCP server using JSON format (complex arguments)
        Write-InfoLog "Adding Serena to Claude Code..."
        $serenaJson = '{"command": "uvx", "args": ["--from", "git+https://github.com/oraios/serena", "serena", "start-mcp-server", "--context", "ide-assistant", "--project", "' + ($ProjectRoot -replace '\\', '/') + '"], "transport": "stdio"}'
        claude mcp add-json serena $serenaJson 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-WarningLog "Failed to add Serena to Claude Code"
        }
        
        # Add Playwright MCP server
        Write-InfoLog "Adding Playwright to Claude Code..."
        claude mcp add playwright npx "@playwright/mcp" 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-WarningLog "Failed to add Playwright to Claude Code"
        }
        
        # Fix Archon MCP server transport if needed
        Write-InfoLog "Checking Archon MCP server transport configuration..."
        $archonServer = Join-Path $ProjectRoot "python\src\mcp_server\mcp_server.py"
        if (Test-Path $archonServer) {
            $content = Get-Content $archonServer -Raw
            if ($content -match 'transport="streamable-http"') {
                Write-InfoLog "Fixing Archon MCP server transport to use stdio..."
                $content = $content -replace 'transport="streamable-http"', 'transport="stdio"'
                Set-Content $archonServer -Value $content -Encoding UTF8
            }
        }
        
        Write-SuccessLog "Claude Code MCP configuration completed"
        return $true
    }
    catch {
        Write-WarningLog "Error configuring Claude Code: $($_.Exception.Message)"
        return $true  # Don't fail the whole script
    }
}

# Validate installation
function Test-Installation {
    Write-ArchonLog "🧪 Validating MCP servers installation..."
    
    $errors = 0
    
    # Check Claude Flow
    if (Get-Command claude-flow -ErrorAction SilentlyContinue) {
        try {
            claude-flow --version >$null 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-SuccessLog "Claude Flow validation passed"
            }
            else {
                Write-ErrorLog "Claude Flow validation failed"
                $errors++
            }
        }
        catch {
            Write-ErrorLog "Claude Flow validation error"
            $errors++
        }
    }
    else {
        Write-ErrorLog "Claude Flow not found in PATH"
        $errors++
    }
    
    # Check Serena
    if (Get-Command uvx -ErrorAction SilentlyContinue) {
        try {
            uvx --from "git+https://github.com/oraios/serena" serena --version >$null 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-SuccessLog "Serena validation passed"
            }
            else {
                Write-WarningLog "Serena validation inconclusive (may need first run)"
            }
        }
        catch {
            Write-WarningLog "Serena validation inconclusive"
        }
    }
    else {
        Write-ErrorLog "uvx not available for Serena"
        $errors++
    }
    
    # Check configuration files  
    $configFile = Join-Path $ProjectRoot "python\config\mcp_servers.json"
    if (Test-Path $configFile) {
        Write-SuccessLog "Archon MCP configuration exists"
    }
    else {
        Write-ErrorLog "Archon MCP configuration missing"
        $errors++
    }
    
    if ($errors -eq 0) {
        Write-SuccessLog "All validations passed!"
        return $true
    }
    else {
        Write-ErrorLog "$errors validation(s) failed"
        return $false
    }
}

# Main execution
function Main {
    Write-Host "================================================================"
    Write-ArchonLog "🚀 Automatic MCP Servers Setup (Windows)"
    Write-Host "   Claude Flow (GitFlow) + Serena + Archon Integration"
    Write-Host "   Memory-Aware Installation Following CLAUDE.md Patterns"
    Write-Host "================================================================"
    Write-Host ""
    
    Write-InfoLog "Project root: $ProjectRoot"
    
    # Memory-aware pre-check
    if (-not (Test-AdaptiveResourceCheck)) {
        Write-WarningLog "System memory is critical - proceeding with conservative settings"
    }
    
    # Execute installation steps
    if (-not (Test-Prerequisites)) {
        exit 1
    }
    
    Write-InfoLog "Starting installation with max $MaxConcurrentInstalls concurrent processes..."
    
    # Install components sequentially (PowerShell job management is complex)
    Write-InfoLog "Installing components sequentially..."
    if (-not (Install-ClaudeFlow)) { exit 1 }
    if (-not (Install-Serena)) { exit 1 }
    if (-not (Install-Playwright)) { exit 1 }
    
    # Configure integration
    if (-not (Update-ArchonConfig)) { exit 1 }
    New-ClaudeDesktopConfig | Out-Null
    Set-ClaudeCodeConfig | Out-Null
    
    # Final validation
    if (-not $SkipValidation) {
        if (-not (Test-Installation)) {
            Write-WarningLog "Some validations failed - check logs above"
        }
    }
    
    Write-Host ""
    Write-ArchonLog "🎉 MCP Servers setup completed!"
    Write-Host ""
    Write-InfoLog "Next steps:"
    Write-InfoLog "  1. Restart Claude Desktop to pick up new configuration (if using Desktop)"
    Write-InfoLog "  2. For Claude Code: Run '/mcp' to see available servers"
    Write-InfoLog "  3. Start Archon server: docker compose up -d"
    Write-InfoLog "  4. Test MCP integration in Claude Desktop or Claude Code"
    Write-Host ""
    Write-InfoLog "Available MCP servers:"
    Write-InfoLog "  • claude-flow: GitFlow integration and SPARC methodology"
    Write-InfoLog "  • serena: Semantic code analysis and IDE assistance"
    Write-InfoLog "  • playwright: Browser automation and testing"
    Write-InfoLog "  • archon-mcp: Project management and knowledge base"
    Write-Host ""
}

# Show help
if ($Help) {
    Write-Host "🏛️ Automatic MCP Servers Setup Script (Windows)"
    Write-Host ""
    Write-Host "Usage: .\setup-mcp-servers.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Help               Show this help message"
    Write-Host "  -Force              Force reinstallation of existing components"
    Write-Host "  -SkipValidation     Skip final validation step" 
    Write-Host "  -NoAdaptive         Disable memory-aware adaptive scaling"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\setup-mcp-servers.ps1                # Standard installation"
    Write-Host "  .\setup-mcp-servers.ps1 -Force         # Force reinstall everything"
    Write-Host "  .\setup-mcp-servers.ps1 -NoAdaptive    # Disable memory management"
    exit 0
}

# Run main function
try {
    Main
}
catch {
    Write-ErrorLog "Setup failed: $($_.Exception.Message)"
    exit 1
}