# Claude Flow New Agent Setup Guide

## 🎯 Complete Guide to Creating Custom Agents

This comprehensive guide covers everything you need to know about creating, registering, and deploying custom agents in the Claude Flow ecosystem.

## 📁 Agent Directory Structure

All agents are defined in the `.claude/agents/` directory using Markdown files with YAML frontmatter:

```
.claude/agents/
├── README.md                    # Agent system overview
├── [agent-name].md              # Individual agent definitions
├── core/                        # Core development agents
│   ├── coder.md
│   ├── tester.md
│   ├── reviewer.md
│   ├── researcher.md
│   └── planner.md
├── specialized/                 # Domain-specific agents
│   ├── mobile/
│   ├── ml/
│   └── blockchain/
├── architecture/                # System design agents
│   └── system-design/
├── github/                      # GitHub integration agents
│   ├── pr-manager.md
│   ├── issue-tracker.md
│   └── code-review-swarm.md
├── consensus/                   # Distributed consensus agents
│   ├── raft-manager.md
│   ├── byzantine-coordinator.md
│   └── gossip-coordinator.md
└── templates/                   # Agent templates and examples
    ├── migration-plan.md
    └── memory-coordinator.md
```

## 🚀 Agent Definition Format

### YAML Frontmatter Structure

Every agent file must start with YAML frontmatter containing:

```yaml
---
name: agent-name                 # Unique identifier (kebab-case)
type: agent-category             # Type classification
color: "#HEX_COLOR"              # UI display color
description: Brief description   # One-line agent purpose
capabilities:                    # List of agent capabilities
  - capability_1
  - capability_2
  - capability_3
priority: high|medium|low        # Execution priority
hooks:                          # Optional lifecycle hooks
  pre: |
    echo "Pre-execution script"
  post: |
    echo "Post-execution script"
---
```

### Agent Content Structure

After the YAML frontmatter, include:

1. **Core Identity Section**
2. **Capabilities Documentation**
3. **Integration Patterns**
4. **Performance Characteristics**
5. **Use Cases and Examples**
6. **Coordination Guidelines**

## 📝 Step-by-Step Agent Creation

### Step 1: Plan Your Agent

**Define the agent's purpose:**
- What specific problem does it solve?
- What domain expertise does it provide?
- How does it coordinate with other agents?

**Choose agent type classification:**
- `developer` - Code implementation
- `semantic-intelligence` - Code analysis
- `knowledge-orchestration` - Information management
- `testing` - Quality assurance
- `architecture` - System design
- `devops` - Operations and deployment
- `analysis` - Code review and metrics
- `coordination` - Multi-agent orchestration

### Step 2: Create the Agent File

Create a new file in `.claude/agents/[agent-name].md`:

```bash
# Create the agent file
touch .claude/agents/my-custom-agent.md
```

### Step 3: Define YAML Frontmatter

```yaml
---
name: my-custom-agent
type: developer
color: "#4F46E5"
description: Custom agent for specialized development tasks
capabilities:
  - custom_development
  - specialized_tooling
  - integration_patterns
priority: high
hooks:
  pre: |
    echo "🔧 Custom agent starting task: $TASK"
    # Initialize any required context
  post: |
    echo "✅ Custom agent task completed"
    # Cleanup or notification logic
---
```

### Step 4: Document Agent Capabilities

```markdown
# My Custom Agent - Specialized Development Expert

You are a specialized development agent focused on [specific domain].

## Core Identity

**Agent Type**: `my-custom-agent`
**Specialization**: [Your specialization]
**Primary Domain**: [Domain expertise]
**Coordination Level**: [Coordination approach]

## Capabilities

### 🎯 **Primary Capabilities**
1. **Capability 1**: Description and usage
2. **Capability 2**: Description and usage
3. **Capability 3**: Description and usage

### 🔧 **Tool Integration**
- Tool 1: How it's used
- Tool 2: Integration patterns
- Tool 3: Coordination protocols

## Performance Characteristics
- **Speed**: Expected response times
- **Reliability**: Success rate metrics
- **Coordination**: Multi-agent interaction patterns

## Use Cases
- Primary use case with example
- Secondary use case with example
- Integration scenarios

## Best Practices
- Development guidelines
- Coordination protocols
- Quality standards
```

## 🎨 Agent Type Categories

### Core Development Agents
```yaml
type: developer
capabilities:
  - code_generation
  - refactoring
  - optimization
```

### Semantic Intelligence Agents
```yaml
type: semantic-intelligence
capabilities:
  - semantic_analysis
  - code_intelligence
  - symbol_navigation
```

### Knowledge Orchestration Agents
```yaml
type: knowledge-orchestration
capabilities:
  - knowledge_management
  - rag_orchestration
  - project_coordination
```

### Testing & Quality Assurance
```yaml
type: testing
capabilities:
  - test_generation
  - quality_assurance
  - validation
```

### Architecture & System Design
```yaml
type: architecture
capabilities:
  - system_design
  - architectural_patterns
  - scalability_planning
```

## 🎯 Agent Registration & Deployment

### Method 1: Direct File Creation
1. Create agent file in `.claude/agents/`
2. Commit to repository
3. Agent becomes available system-wide

### Method 2: Template-Based Creation
```bash
# Use base template (if available)
cp .claude/agents/templates/base-template.md .claude/agents/my-agent.md

# Customize the template
# Edit YAML frontmatter and content
```

### Method 3: Programmatic Creation
```bash
# Generate agent using Claude Flow CLI (if available)
npx claude-flow@alpha agent create --name my-agent --type developer
```

## 🔧 Agent Testing & Validation

### Test Agent Registration
```bash
# List available agents
npx claude-flow@alpha agent list

# Check if your agent is registered
npx claude-flow@alpha agent list | grep "my-custom-agent"
```

### Test Agent Spawning
```bash
# Spawn your agent for testing
npx claude-flow@alpha agent spawn my-custom-agent --name "TestInstance"

# Test with specific task
npx claude-flow@alpha agent spawn my-custom-agent --task "test task"
```

### Validation Checklist
- [ ] YAML frontmatter is valid
- [ ] Agent name is unique and follows kebab-case
- [ ] All required fields are present
- [ ] Capabilities are clearly defined
- [ ] Hooks syntax is correct (if used)
- [ ] Agent spawns successfully
- [ ] Integration with other agents works

## 🎭 Agent Lifecycle Hooks

### Pre-Execution Hooks
```bash
hooks:
  pre: |
    echo "🚀 Agent initializing: $TASK"
    # Environment setup
    # Context initialization
    # Resource preparation
```

### Post-Execution Hooks
```bash
hooks:
  post: |
    echo "✅ Agent completed: $TASK"
    # Cleanup operations
    # Result reporting
    # Coordination updates
```

### Advanced Hook Examples
```bash
hooks:
  pre: |
    # Check prerequisites
    if ! command -v node &> /dev/null; then
      echo "Node.js required but not installed"
      exit 1
    fi
    
    # Initialize coordination
    claude-flow memory store "agent/session/$(date +%s)" "Starting: $TASK"
    
  post: |
    # Report completion
    claude-flow memory store "agent/results/$(date +%s)" "Completed: $TASK"
    
    # Trigger follow-up actions
    claude-flow hooks notify --agents "reviewer,tester" --message "Ready for review"
```

## 🚀 Advanced Agent Patterns

### Multi-Agent Coordination
```markdown
## Coordination with Other Agents

### **Primary Collaborations**

**With `coder` agent:**
- Provide [specific collaboration pattern]
- Share [specific information type]
- Coordinate on [specific workflows]

**With `tester` agent:**
- Supply [testing context]
- Enable [testing strategies]
- Support [quality assurance patterns]
```

### Memory-Driven Patterns
```markdown
### **Memory Integration**
```typescript
// Shared context management
const sharedContext = await claude.memory.read('project/context');

// Cross-agent coordination
await claude.memory.write('agent/status/my-agent', {
  currentTask: taskId,
  progress: completionPercentage,
  nextActions: plannedActions
});
```

### Performance Optimization
```markdown
### **Performance Characteristics**
- **Response Time**: 50-150ms for standard operations
- **Coordination Latency**: 100-300ms for multi-agent workflows
- **Memory Efficiency**: <10MB baseline memory usage
- **Scalability**: Supports 10+ concurrent instances
```

## 🔍 Troubleshooting Common Issues

### Agent Not Appearing in List
**Problem**: Agent doesn't show up in `npx claude-flow agent list`

**Solutions**:
1. Check YAML frontmatter syntax:
   ```bash
   # Validate YAML syntax
   python -c "import yaml; yaml.safe_load(open('.claude/agents/my-agent.md').read().split('---')[1])"
   ```

2. Verify file location:
   ```bash
   # Ensure file is in correct directory
   ls -la .claude/agents/my-agent.md
   ```

3. Restart Claude Flow:
   ```bash
   # Restart the system to reload agents
   npx claude-flow@alpha restart
   ```

### Agent Fails to Spawn
**Problem**: `npx claude-flow agent spawn` fails

**Solutions**:
1. Check agent name spelling
2. Verify all required capabilities are defined
3. Test hooks syntax in isolation
4. Check system resource availability

### Hooks Not Executing
**Problem**: Pre/post hooks don't run

**Solutions**:
1. Verify shell syntax in hooks
2. Check for required environment variables
3. Ensure proper permissions for hook scripts
4. Test hooks independently:
   ```bash
   # Test pre-hook manually
   TASK="test" bash -c "echo 'Pre-hook test'; [your pre-hook code]"
   ```

### Coordination Issues
**Problem**: Agent doesn't coordinate properly with others

**Solutions**:
1. Check memory key naming conventions
2. Verify shared context structure
3. Test inter-agent communication patterns
4. Review coordination timing and sequencing

## 🎯 Best Practices & Guidelines

### Naming Conventions
- **Agent Names**: Use kebab-case (`my-agent-name`)
- **Capabilities**: Use snake_case (`semantic_analysis`)
- **Memory Keys**: Use hierarchical paths (`project/agent/status`)
- **File Names**: Match agent name exactly (`my-agent-name.md`)

### Documentation Standards
- **Clear Purpose**: One-sentence description in YAML
- **Comprehensive Capabilities**: List all major functions
- **Integration Examples**: Show coordination patterns
- **Performance Metrics**: Include expected characteristics
- **Use Case Coverage**: Document primary scenarios

### Code Quality
- **YAML Validation**: Always validate frontmatter syntax
- **Hook Safety**: Use safe scripting practices
- **Error Handling**: Include robust error checking
- **Resource Cleanup**: Ensure proper cleanup in post-hooks

### Security Considerations
- **Input Validation**: Sanitize all external inputs
- **Permission Boundaries**: Define clear tool access limits
- **Secret Management**: Never hardcode sensitive data
- **Network Access**: Document and restrict network usage

## 🔗 Integration Examples

### Example 1: API Development Agent
```yaml
---
name: api-developer
type: developer
color: "#10B981"
description: Specialized API development and documentation expert
capabilities:
  - api_design
  - openapi_generation
  - endpoint_implementation
  - api_testing
priority: high
hooks:
  pre: |
    echo "🌐 API Developer starting: $TASK"
    # Check for existing API specs
    if [ -f "openapi.yaml" ] || [ -f "swagger.json" ]; then
      echo "📋 Found existing API specification"
    fi
  post: |
    echo "✅ API development completed"
    # Generate/update API documentation
    if command -v swagger-codegen &> /dev/null; then
      echo "📚 Generating API documentation"
    fi
---
```

### Example 2: Database Migration Agent
```yaml
---
name: db-migration-specialist
type: architecture
color: "#8B5CF6"
description: Database migration and schema evolution expert
capabilities:
  - schema_migration
  - data_transformation
  - migration_rollback
  - performance_optimization
priority: high
hooks:
  pre: |
    echo "🗄️  Database Migration Specialist initializing"
    # Backup database before migration
    echo "💾 Creating database backup"
  post: |
    echo "✅ Migration completed successfully"
    # Verify migration integrity
    echo "🔍 Running migration validation"
---
```

### Example 3: Security Audit Agent
```yaml
---
name: security-auditor
type: analysis
color: "#EF4444"
description: Comprehensive security analysis and vulnerability assessment
capabilities:
  - vulnerability_scanning
  - security_analysis
  - compliance_checking
  - penetration_testing
priority: high
hooks:
  pre: |
    echo "🛡️  Security Auditor starting analysis: $TASK"
    # Initialize security scanning tools
    echo "🔒 Preparing security analysis environment"
  post: |
    echo "✅ Security audit completed"
    # Generate security report
    echo "📊 Compiling security assessment report"
---
```

## 🎯 Advanced Coordination Patterns

### Hierarchical Coordination
```markdown
### **Master-Worker Pattern**
- **Master Agent**: Coordinates overall strategy
- **Worker Agents**: Execute specialized subtasks
- **Communication**: Via shared memory and hooks
```

### Peer-to-Peer Coordination
```markdown
### **Mesh Coordination**
- **Equal Agents**: All agents have equal authority
- **Consensus Building**: Decisions made collectively
- **Load Distribution**: Work distributed evenly
```

### Pipeline Coordination
```markdown
### **Sequential Processing**
- **Stage 1**: Analysis agent processes requirements
- **Stage 2**: Development agent implements solution
- **Stage 3**: Testing agent validates implementation
- **Stage 4**: Review agent ensures quality standards
```

## 📈 Performance Monitoring

### Agent Metrics
```bash
# Monitor agent performance
npx claude-flow@alpha metrics agent --name my-agent

# Track coordination efficiency
npx claude-flow@alpha metrics coordination --session-id [session]

# Analyze resource usage
npx claude-flow@alpha metrics resources --agent-type developer
```

### Performance Optimization
```markdown
### **Optimization Strategies**
1. **Efficient Memory Usage**: Minimize memory footprint
2. **Fast Coordination**: Reduce inter-agent communication overhead
3. **Resource Pooling**: Share common resources across agents
4. **Caching**: Cache frequently accessed data
```

## 🎉 Next Steps

1. **Start Simple**: Create a basic agent following this guide
2. **Test Thoroughly**: Validate all functionality before deployment
3. **Iterate**: Improve based on usage patterns and feedback
4. **Share**: Contribute successful patterns to the community
5. **Scale**: Build complex multi-agent workflows

## 📚 Additional Resources

- **Claude Flow Documentation**: [Official docs]
- **Agent Examples**: Browse `.claude/agents/` for reference implementations
- **Community Patterns**: Share and discover agent patterns
- **Best Practices**: Follow established conventions and standards

---

**Happy Agent Building!** 🚀

Create powerful, coordinated agents that enhance your development workflow and enable sophisticated multi-agent collaboration patterns.