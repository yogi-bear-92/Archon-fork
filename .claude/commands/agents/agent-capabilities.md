# agent-capabilities

Matrix of agent capabilities and their specializations.

## Capability Matrix

| Agent Type | Primary Skills | Best For |
|------------|---------------|----------|
| coder | Implementation, debugging | Feature development |
| researcher | Analysis, synthesis | Requirements gathering |
| tester | Testing, validation | Quality assurance |
| architect | Design, planning | System architecture |
| **serena-master** | **Semantic analysis, code intelligence** | **Advanced code understanding & refactoring** |
| **archon-master** | **Knowledge management, RAG orchestration** | **Project coordination & documentation** |

## Master Agent Capabilities

### Serena Master Agent
- **Semantic Code Intelligence**: Advanced LSP-powered analysis
- **Multi-Agent Coordination**: Orchestrate semantic workflows
- **Memory Management**: Cross-agent knowledge sharing
- **Pattern Recognition**: Intelligent code pattern discovery
- **Refactoring Expertise**: Safe large-scale transformations

### Archon Master Agent  
- **Knowledge Orchestration**: Enterprise-scale information management
- **RAG System Mastery**: 4 advanced retrieval strategies
- **Project Management**: Progressive refinement protocols
- **Multi-Agent Leadership**: Coordinate complex workflows
- **Real-time Collaboration**: Socket.IO powered team coordination

## Querying Capabilities
```bash
# List all capabilities
npx claude-flow agents capabilities

# For specific agent
npx claude-flow agents capabilities --type coder
npx claude-flow agents capabilities --type serena-master
npx claude-flow agents capabilities --type archon-master
```
