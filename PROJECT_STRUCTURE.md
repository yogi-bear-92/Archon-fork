# Archon Project Structure

## 📁 **REORGANIZED DIRECTORY STRUCTURE**

```
archon-fork/
├── 📋 **CORE PROJECT FILES**
│   ├── CLAUDE.md                    # AI development configuration
│   ├── README.md                    # Main project documentation
│   ├── LICENSE                      # Project license
│   ├── .gitignore                   # Git ignore rules
│   ├── .env.example                 # Environment template
│   ├── package.json                 # Node.js dependencies
│   └── Makefile                     # Build automation
│
├── 🐍 **python/**                   # Python backend (FastAPI + Archon)
│   ├── src/server/                  # Main server code
│   ├── scripts/                     # Python utility scripts
│   ├── pyproject.toml               # Python dependencies
│   └── pyrightconfig.json           # Python type checking
│
├── 🖥️ **archon-ui-main/**           # React frontend
│   ├── src/                         # Frontend source code
│   ├── package.json                 # Frontend dependencies
│   └── README.md                    # Frontend documentation
│
├── 🔧 **tools/**                    # Organized utility tools
│   ├── database/                    # Database management
│   │   ├── backup_database_api.py   # Database backup utility
│   │   └── setup_database.py        # Database setup script
│   ├── project-management/          # Project management utilities
│   │   ├── cleanup_empty_projects.py
│   │   ├── merge_projects.py
│   │   └── merge_related_projects.py
│   ├── communication/               # Communication tools
│   │   ├── claude-legendary-cli.py
│   │   ├── claude-legendary-commands.py
│   │   ├── legendary-agent-communication.py
│   │   ├── legendary-comm-simple.py
│   │   └── legendary-communication-cli.py
│   └── analysis/                    # Analysis utilities
│
├── 🧪 **utilities/**                # Development utilities
│   ├── testing/                     # Test utilities
│   │   ├── test_auto_detection.py
│   │   └── test_duplicate_check.py
│   └── development/                 # Development tools
│
├── ⚙️ **config/**                   # Configuration files
│   ├── claude-flow/                 # Claude Flow configs
│   │   └── claude-flow.config.json
│   ├── system/                      # System configs
│   │   └── .mcp.json
│   ├── coordination-status-report.json
│   ├── memory-emergency-status.json
│   ├── memory-limits.json
│   └── memory-recovery-success-report.json
│
├── 🚀 **deployment/**               # Deployment files
│   ├── docker/                      # Docker configurations
│   │   ├── docker-compose.yml
│   │   └── docker-compose.docs.yml
│   └── scripts/                     # Deployment scripts
│       ├── claude-flow              # Unix script
│       ├── claude-flow.bat          # Windows batch
│       └── claude-flow.ps1          # PowerShell script
│
├── 📚 **docs/**                     # Documentation
│   ├── AI_TAGGING_IMPLEMENTATION_SUMMARY.md
│   ├── README-LEGENDARY-COMMANDS.md
│   ├── archon-mcp-setup.md          # MCP setup guide
│   ├── unified-mcp-examples.md      # Usage examples
│   └── [70+ other documentation files]
│
├── 🗃️ **archive/**                  # Archived content
│   ├── challenges/                  # Flow-Nexus challenges
│   │   └── flow-nexus-challenges/
│   ├── experiments/                 # Experimental code
│   │   └── experiments/
│   └── algorithm_duel_results.json
│
├── 📊 **tests/**                    # Test suites
│   ├── test_master_agent.py
│   └── test_ruv_optimizer.py
│
├── 🧠 **src/**                      # Algorithm implementations
│   ├── algorithm_duel_solution.py
│   ├── neural_deployment_analysis.py
│   └── [other algorithm files]
│
├── 📝 **scripts/**                  # Automation scripts
│   ├── performance analysis tools
│   ├── tagging utilities
│   └── optimization scripts
│
└── 🔧 **Hidden Config Directories**
    ├── .claude/                     # Claude IDE settings
    ├── .claude-flow/                # Claude Flow metrics
    ├── .serena/                     # Serena code intelligence
    ├── .hive-mind/                  # Hive mind coordination
    └── .git/                        # Git repository data
```

## 🎯 **BENEFITS OF REORGANIZATION**

### **✅ IMPROVED ORGANIZATION:**
- **Logical Grouping**: Related files organized by purpose
- **Clear Separation**: Tools, utilities, configs, and docs separated
- **Easy Navigation**: Developers can quickly find relevant files
- **Reduced Clutter**: Root directory contains only essential files

### **✅ ENHANCED MAINTAINABILITY:**
- **Tool Organization**: Database, project management, and communication tools grouped
- **Configuration Management**: All configs in dedicated `/config` directory
- **Documentation Structure**: All docs centralized with clear categories
- **Deployment Isolation**: Docker and deployment scripts separated

### **✅ BETTER DEVELOPMENT EXPERIENCE:**
- **IDE Integration**: Cleaner project structure in IDEs
- **Build Processes**: Clear separation of build artifacts and source
- **Team Collaboration**: Consistent file locations for team members
- **Onboarding**: New developers can understand structure quickly

## 🔄 **MIGRATION IMPACT**

### **FILES MOVED:**
- **15+ utility scripts** moved to `/tools/` subdirectories
- **Configuration files** moved to `/config/`
- **Docker files** moved to `/deployment/docker/`
- **Documentation** consolidated in `/docs/`
- **Archives** moved to `/archive/`

### **REFERENCES TO UPDATE:**
- Import statements in Python files
- Script paths in documentation
- Docker compose file paths
- CI/CD pipeline references
- README file links

## 📋 **QUICK REFERENCE**

### **COMMON TASKS:**
```bash
# Database operations
python tools/database/setup_database.py
python tools/database/backup_database_api.py

# Project management
python tools/project-management/cleanup_empty_projects.py
python tools/project-management/merge_projects.py

# Communication tools
python tools/communication/claude-legendary-cli.py
python tools/communication/legendary-agent-communication.py

# Testing utilities
python utilities/testing/test_auto_detection.py
python utilities/testing/test_duplicate_check.py

# Deployment
docker-compose -f deployment/docker/docker-compose.yml up
```

### **CONFIGURATION LOCATIONS:**
- **Claude Flow**: `config/claude-flow/claude-flow.config.json`
- **MCP Settings**: `config/system/.mcp.json`  
- **System Configs**: `config/` directory
- **Environment**: `.env` (root level)

---

**🎉 Project structure is now organized, maintainable, and developer-friendly!**