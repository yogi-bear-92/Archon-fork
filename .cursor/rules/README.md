# Cursor Rules for Archon Project

This directory contains Cursor rules that help AI assistants understand and work with the Archon project effectively. These rules are automatically applied based on file types and project context.

## Rule Files Overview

### 1. `project-architecture.mdc` (Always Applied)
- **Purpose**: Core project structure and architecture guidelines
- **Scope**: Always applied to provide context about the microservices architecture
- **Key Topics**: Directory structure, technology stack, service communication patterns

### 2. `typescript-react.mdc` (TypeScript/React Files)
- **Purpose**: Frontend development standards for React and TypeScript
- **Scope**: Applied to `*.ts`, `*.tsx`, `*.js`, `*.jsx` files
- **Key Topics**: TypeScript configuration, React patterns, TailwindCSS usage, testing

### 3. `python-backend.mdc` (Python Files)
- **Purpose**: Backend development standards for Python and FastAPI
- **Scope**: Applied to `*.py` files
- **Key Topics**: FastAPI patterns, Pydantic models, database integration, AI/ML integration

### 4. `development-workflow.mdc` (Always Applied)
- **Purpose**: Development environment, testing, and deployment guidelines
- **Scope**: Always applied for development context
- **Key Topics**: Docker setup, testing commands, linting, CI/CD, environment configuration

### 5. `archon-integration.mdc` (Manual Application)
- **Purpose**: Archon-specific integration patterns and MCP server guidelines
- **Scope**: Applied when working with Archon-specific features
- **Key Topics**: MCP protocol, knowledge management, AI agents, real-time collaboration

### 6. `coding-standards.mdc` (Always Applied)
- **Purpose**: General coding standards and best practices
- **Scope**: Always applied for code quality guidance
- **Key Topics**: Code organization, security, performance, testing, documentation

## How Cursor Rules Work

### Automatic Application
- **Always Applied**: Rules with `alwaysApply: true` are shown to AI for every request
- **File Type Based**: Rules with `globs` patterns are applied to matching file types
- **Manual Application**: Rules with `description` can be manually referenced by AI

### Rule Format
- **Frontmatter**: YAML metadata defining when rules apply
- **Markdown Content**: Human-readable guidelines and standards
- **File References**: Use `[filename](mdc:path)` to reference project files

### Best Practices
- **Specific Guidelines**: Each rule focuses on specific aspects of development
- **Project Context**: Rules are tailored to Archon's microservices architecture
- **Technology Stack**: Rules reflect the actual technologies used in the project
- **Consistent Standards**: All rules work together to maintain code quality

## Usage Examples

### For Frontend Development
When working on React/TypeScript files, Cursor will automatically apply:
- TypeScript and React rules
- Project architecture rules
- Development workflow rules
- General coding standards

### For Backend Development
When working on Python files, Cursor will automatically apply:
- Python backend rules
- Project architecture rules
- Development workflow rules
- General coding standards

### For Archon-Specific Features
When working with MCP servers, knowledge management, or AI agents:
- Archon integration rules (manually referenced)
- All other applicable rules

## Maintenance

### Updating Rules
- **Keep Current**: Update rules when project structure or standards change
- **Version Control**: Track rule changes in git for team consistency
- **Team Review**: Review rule updates with the development team
- **Documentation**: Update this README when adding new rules

### Rule Effectiveness
- **Monitor Usage**: Check if rules are being followed consistently
- **Gather Feedback**: Collect feedback on rule clarity and usefulness
- **Iterate**: Continuously improve rules based on team needs
- **Simplify**: Keep rules clear and actionable

## Integration with Development

These rules work alongside:
- **ESLint/Prettier**: Code formatting and linting
- **TypeScript**: Type checking and compilation
- **Pytest/Vitest**: Testing frameworks
- **Docker**: Containerization and deployment
- **CI/CD**: Automated quality checks

The rules provide context and guidance that complements these tools, helping AI assistants make better decisions about code structure, patterns, and best practices specific to the Archon project.
