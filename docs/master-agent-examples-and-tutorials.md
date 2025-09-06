# Master Agent System - Examples and Tutorials

## Table of Contents

- [Getting Started Tutorials](#getting-started-tutorials)
- [Common Use Cases](#common-use-cases)
- [Code Examples](#code-examples)
- [Configuration Samples](#configuration-samples)
- [Best Practices](#best-practices)
- [Advanced Usage Scenarios](#advanced-usage-scenarios)
- [Troubleshooting Examples](#troubleshooting-examples)

---

## Getting Started Tutorials

### Tutorial 1: Basic Setup and First Query

This tutorial walks you through setting up the Master Agent System and executing your first intelligent query.

#### Step 1: Environment Setup

```bash
# Clone the repository
git clone https://github.com/coleam00/archon.git
cd archon

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

**Required Environment Variables:**
```bash
# Core database connection
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...

# AI Provider (optional - can configure via UI)
OPENAI_API_KEY=sk-your-openai-key-here

# Service configuration (optional - uses defaults)
ARCHON_SERVER_PORT=8181
ARCHON_MCP_PORT=8051
ARCHON_AGENTS_PORT=8052
```

#### Step 2: Database Setup

1. **Login to Supabase Dashboard**: Go to https://supabase.com/dashboard
2. **Open SQL Editor**: Navigate to SQL Editor in your project
3. **Run Migration**: Copy and execute the contents of `migration/complete_setup.sql`

```sql
-- Example: Key tables that should be created
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('documents', 'projects', 'tasks', 'mcp_client_sessions');
```

#### Step 3: Start the System

```bash
# Start all services
docker compose up --build -d

# Verify services are running
docker compose ps

# Check service health
curl http://localhost:8181/health  # Server API
curl http://localhost:8051/health  # MCP Server
curl http://localhost:8052/health  # Agents Service
```

**Expected Output:**
```json
{
  "status": "healthy",
  "timestamp": "2024-09-05T10:00:00Z",
  "services": {
    "database": "connected",
    "agents": "ready",
    "mcp": "active"
  }
}
```

#### Step 4: Web Interface Configuration

1. **Open Web Interface**: Navigate to http://localhost:3737
2. **Complete Onboarding**: Follow the setup wizard
3. **Configure AI Provider**: Set your OpenAI API key (if not done via environment)
4. **Verify Configuration**: Test API connection

#### Step 5: Your First Query

**Via Web Interface:**
1. Go to "Knowledge Base" → "Quick Search"
2. Enter query: "How to implement FastAPI with async database operations?"
3. Observe the intelligent response with sources and examples

**Via MCP Client (Advanced):**
```python
# Example using the MCP client
import asyncio
from mcp_client import MCPClient

async def first_query_example():
    client = MCPClient("http://localhost:8051")
    
    response = await client.perform_rag_query(
        query="How to implement FastAPI with async database operations?",
        strategy="hybrid",
        max_results=3
    )
    
    print(f"Found {len(response.results)} relevant sources:")
    for result in response.results:
        print(f"- {result.title} (relevance: {result.relevance_score:.2f})")
        print(f"  {result.snippet}")

# Run the example
asyncio.run(first_query_example())
```

### Tutorial 2: Project Management with AI Assistance

Learn how to create and manage projects with intelligent task generation and tracking.

#### Step 1: Create Your First Project

**Via Web Interface:**
```bash
1. Navigate to "Projects" → "Create New Project"
2. Fill in project details:
   - Name: "E-commerce API Modernization"
   - Description: "Migrate legacy Django API to FastAPI with microservices architecture"
   - Repository: https://github.com/company/ecommerce-api (optional)
   - Type: "backend-modernization"
```

**Via API:**
```python
import httpx
import asyncio

async def create_project_example():
    async with httpx.AsyncClient() as client:
        project_data = {
            "name": "E-commerce API Modernization",
            "description": "Migrate legacy Django API to FastAPI with microservices architecture",
            "github_repo": "https://github.com/company/ecommerce-api",
            "project_type": "backend-modernization",
            "settings": {
                "ai_assistance": True,
                "auto_task_generation": True,
                "progress_tracking": True
            }
        }
        
        response = await client.post(
            "http://localhost:8181/api/projects",
            json=project_data
        )
        
        project = response.json()
        print(f"Created project: {project['name']} (ID: {project['id']})")
        return project

project = asyncio.run(create_project_example())
```

#### Step 2: AI-Powered Task Generation

**Via Web Interface:**
1. Open your project
2. Click "Generate Tasks" 
3. Enter detailed requirements:
   ```
   Create comprehensive task breakdown for migrating Django REST API to FastAPI:
   - Maintain backwards compatibility
   - Implement async database operations
   - Add comprehensive testing
   - Include deployment strategy
   - Focus on performance optimization
   ```
4. Review and customize generated tasks

**Generated Task Example:**
```yaml
Task: "Implement Database Migration Strategy"
Description: |
  Create migration strategy for moving from Django ORM to SQLAlchemy with async support
  - Analyze existing models and relationships
  - Design SQLAlchemy models with async support
  - Create migration scripts for data preservation
  - Implement rollback procedures
  - Test migration with sample data
Priority: High
Dependencies: ["Architecture Analysis", "Technology Stack Selection"]
Estimated Time: "2-3 days"
Tags: ["database", "migration", "async", "sqlalchemy"]
```

#### Step 3: Working with Tasks

**Add Context to Tasks:**
```python
async def enrich_task_with_context():
    """Example: Adding research context to a task"""
    
    client = MCPClient("http://localhost:8051")
    
    # Research FastAPI best practices
    research = await client.perform_rag_query(
        query="FastAPI async database patterns SQLAlchemy best practices",
        sources=["fastapi_docs", "sqlalchemy_docs"],
        strategy="contextual"
    )
    
    # Update task with research findings
    task_update = {
        "research_findings": [
            {"source": r.source, "insight": r.content}
            for r in research.results
        ],
        "implementation_notes": "Based on research, focus on async session management and connection pooling"
    }
    
    await client.update_task(
        task_id="task-123",
        updates=task_update
    )

asyncio.run(enrich_task_with_context())
```

### Tutorial 3: Knowledge Base Management

Build and manage a comprehensive knowledge base for your AI agents.

#### Step 1: Web Crawling Setup

**Crawl Technical Documentation:**
```bash
# Via Web Interface
1. Go to "Knowledge Base" → "Crawl Website"
2. Enter URL: https://fastapi.tiangolo.com/
3. Configuration:
   - Max Depth: 3 levels
   - Include Code Examples: Yes
   - Filter Patterns: /tutorial/, /advanced/, /reference/
   - Exclude Patterns: /blog/, /community/
4. Start crawling and monitor progress
```

**Programmatic Crawling:**
```python
async def setup_knowledge_crawling():
    """Example: Setting up automated knowledge crawling"""
    
    crawl_configs = [
        {
            "url": "https://fastapi.tiangolo.com/",
            "name": "FastAPI Documentation", 
            "max_depth": 3,
            "include_patterns": ["/tutorial/", "/advanced/", "/reference/"],
            "extract_code_examples": True
        },
        {
            "url": "https://docs.pydantic.dev/",
            "name": "Pydantic Documentation",
            "max_depth": 2,
            "extract_code_examples": True
        },
        {
            "url": "https://docs.sqlalchemy.org/",
            "name": "SQLAlchemy Documentation", 
            "max_depth": 2,
            "include_patterns": ["/orm/", "/core/", "/tutorial/"]
        }
    ]
    
    async with httpx.AsyncClient() as client:
        for config in crawl_configs:
            response = await client.post(
                "http://localhost:8181/api/crawl",
                json=config
            )
            
            crawl_job = response.json()
            print(f"Started crawling {config['name']}: {crawl_job['job_id']}")

asyncio.run(setup_knowledge_crawling())
```

#### Step 2: Document Upload and Processing

**Upload Documents:**
```python
async def upload_documents():
    """Example: Uploading and processing documents"""
    
    documents = [
        ("architecture_decisions.pdf", "application/pdf"),
        ("api_specification.yaml", "text/yaml"),
        ("team_guidelines.md", "text/markdown"),
        ("migration_notes.txt", "text/plain")
    ]
    
    async with httpx.AsyncClient() as client:
        for filename, content_type in documents:
            with open(f"docs/{filename}", "rb") as file:
                files = {"file": (filename, file, content_type)}
                data = {
                    "tags": ["architecture", "migration", "guidelines"],
                    "category": "internal_docs",
                    "auto_extract_code": True
                }
                
                response = await client.post(
                    "http://localhost:8181/api/documents/upload",
                    files=files,
                    data=data
                )
                
                upload_result = response.json()
                print(f"Uploaded {filename}: {upload_result['document_id']}")

asyncio.run(upload_documents())
```

#### Step 3: Knowledge Base Optimization

**Optimize Search Performance:**
```python
async def optimize_knowledge_base():
    """Example: Optimizing knowledge base for better performance"""
    
    client = MCPClient("http://localhost:8051")
    
    # Test search queries and analyze performance
    test_queries = [
        "FastAPI authentication with JWT tokens",
        "SQLAlchemy async session management", 
        "Docker deployment best practices",
        "API testing strategies with pytest",
        "Database migration rollback procedures"
    ]
    
    performance_results = []
    
    for query in test_queries:
        start_time = time.time()
        
        results = await client.perform_rag_query(
            query=query,
            strategy="hybrid",
            max_results=5
        )
        
        end_time = time.time()
        
        performance_results.append({
            "query": query,
            "response_time": end_time - start_time,
            "result_count": len(results.results),
            "avg_relevance": sum(r.relevance_score for r in results.results) / len(results.results)
        })
    
    # Analyze and optimize based on results
    avg_response_time = sum(r["response_time"] for r in performance_results) / len(performance_results)
    avg_relevance = sum(r["avg_relevance"] for r in performance_results) / len(performance_results)
    
    print(f"Knowledge base performance:")
    print(f"- Average response time: {avg_response_time:.2f}s")
    print(f"- Average relevance score: {avg_relevance:.2f}")
    
    if avg_response_time > 1.0:
        print("Recommendation: Consider optimizing vector indexes")
    if avg_relevance < 0.7:
        print("Recommendation: Improve document chunking strategy")

asyncio.run(optimize_knowledge_base())
```

---

## Common Use Cases

### Use Case 1: Legacy Code Modernization

**Scenario**: Modernizing a large legacy codebase with AI assistance and systematic planning.

#### Implementation Example

```python
class LegacyModerizationWorkflow:
    """Complete workflow for legacy code modernization with AI assistance."""
    
    def __init__(self):
        self.mcp_client = MCPClient("http://localhost:8051")
        self.master_agent = None
        
    async def analyze_legacy_codebase(self, repo_path: str):
        """Step 1: Comprehensive legacy codebase analysis"""
        
        # Use Serena for deep code analysis
        analysis_request = {
            "type": "legacy_analysis",
            "repository": repo_path,
            "analysis_depth": "comprehensive",
            "focus_areas": [
                "architectural_patterns",
                "technical_debt",
                "modernization_opportunities",
                "dependency_analysis",
                "security_vulnerabilities"
            ]
        }
        
        # Deploy specialized agents for analysis
        agents = await self.deploy_analysis_agents(analysis_request)
        
        # Coordinate comprehensive analysis
        analysis_results = {}
        for agent_type, agent in agents.items():
            analysis_results[agent_type] = await agent.analyze_codebase(repo_path)
        
        return analysis_results
    
    async def create_modernization_plan(self, analysis_results: dict):
        """Step 2: Create comprehensive modernization plan"""
        
        # Use Master Agent for high-level planning
        planning_query = f"""
        Based on the legacy codebase analysis, create a comprehensive modernization plan:
        
        Analysis Summary:
        - Architecture: {analysis_results['architecture']['summary']}
        - Technical Debt: {analysis_results['technical_debt']['score']}
        - Dependencies: {analysis_results['dependencies']['outdated_count']} outdated
        - Security Issues: {analysis_results['security']['high_risk_count']} high-risk
        
        Create a phased modernization plan with:
        1. Risk assessment and mitigation strategies
        2. Technology stack recommendations
        3. Migration timeline with milestones
        4. Testing and validation approach
        5. Rollback procedures
        """
        
        plan = await self.mcp_client.perform_rag_query(
            query=planning_query,
            sources=["modernization_guides", "migration_best_practices"],
            strategy="agentic"
        )
        
        return plan
    
    async def execute_modernization_phase(
        self, 
        phase_number: int,
        phase_plan: dict
    ):
        """Step 3: Execute specific modernization phase"""
        
        # Create project for this phase
        project = await self.mcp_client.create_project(
            name=f"Modernization Phase {phase_number}",
            description=phase_plan["description"],
            phase_details=phase_plan
        )
        
        # Generate detailed tasks for phase
        for milestone in phase_plan["milestones"]:
            # AI-assisted task generation
            task_generation_prompt = f"""
            Create detailed tasks for milestone: {milestone['name']}
            
            Context:
            - Objective: {milestone['objective']}
            - Success Criteria: {milestone['success_criteria']}
            - Resources: {milestone['resources']}
            - Constraints: {milestone['constraints']}
            
            Generate specific, actionable tasks with:
            - Clear acceptance criteria
            - Time estimates
            - Dependencies
            - Risk factors
            """
            
            tasks = await self.mcp_client.perform_rag_query(
                query=task_generation_prompt,
                strategy="hybrid"
            )
            
            # Create tasks in project
            for task_desc in tasks.generated_tasks:
                await self.mcp_client.create_task(
                    project_id=project["id"],
                    title=task_desc["title"],
                    description=task_desc["description"],
                    priority=task_desc["priority"],
                    metadata={
                        "phase": phase_number,
                        "milestone": milestone["name"],
                        "estimated_hours": task_desc["estimated_hours"],
                        "risk_level": task_desc["risk_level"]
                    }
                )
        
        return project

# Usage example
async def modernization_example():
    workflow = LegacyModerizationWorkflow()
    
    # Analyze existing codebase
    analysis = await workflow.analyze_legacy_codebase("/path/to/legacy/code")
    
    # Create modernization plan
    plan = await workflow.create_modernization_plan(analysis)
    
    # Execute phases
    for phase_num, phase_plan in enumerate(plan.phases, 1):
        project = await workflow.execute_modernization_phase(phase_num, phase_plan)
        print(f"Created modernization project: {project['name']}")

asyncio.run(modernization_example())
```

### Use Case 2: API Development with Documentation

**Scenario**: Building a comprehensive API with auto-generated documentation and testing.

```python
class APIDocumentationWorkflow:
    """Comprehensive API development with intelligent documentation generation."""
    
    async def design_api_architecture(self, requirements: str):
        """Design API architecture with Master Agent assistance"""
        
        architecture_query = f"""
        Design a comprehensive API architecture for:
        {requirements}
        
        Include:
        1. RESTful endpoint design with OpenAPI specification
        2. Authentication and authorization strategy
        3. Database schema design
        4. Error handling patterns
        5. Rate limiting and security considerations
        6. Testing strategy
        7. Deployment architecture
        """
        
        client = MCPClient("http://localhost:8051")
        architecture = await client.perform_rag_query(
            query=architecture_query,
            sources=["api_design_patterns", "openapi_specs", "security_guides"],
            strategy="agentic",
            max_results=10
        )
        
        return architecture
    
    async def generate_api_implementation(self, architecture: dict):
        """Generate API implementation with code examples"""
        
        implementation_tasks = [
            "FastAPI application structure with dependency injection",
            "Database models with SQLAlchemy and async support", 
            "Authentication middleware with JWT tokens",
            "API endpoint implementations with validation",
            "Error handling and logging setup",
            "Testing framework with pytest and fixtures",
            "Docker configuration for deployment",
            "CI/CD pipeline configuration"
        ]
        
        implementations = {}
        
        for task in implementation_tasks:
            code_examples = await self.mcp_client.search_code_examples(
                query=task,
                language="python",
                framework="fastapi"
            )
            
            implementations[task] = {
                "examples": code_examples.examples,
                "best_practices": code_examples.best_practices,
                "common_patterns": code_examples.patterns
            }
        
        return implementations
    
    async def create_comprehensive_documentation(
        self,
        architecture: dict,
        implementation: dict
    ):
        """Create comprehensive API documentation"""
        
        documentation_sections = {
            "getting_started": "Quick start guide with setup instructions",
            "authentication": "Authentication and authorization guide",
            "endpoints": "Complete endpoint reference with examples",
            "error_handling": "Error codes and handling patterns",
            "rate_limiting": "Rate limiting policies and headers",
            "testing": "Testing guide with example requests",
            "deployment": "Deployment guide and infrastructure requirements",
            "troubleshooting": "Common issues and solutions"
        }
        
        generated_docs = {}
        
        for section, description in documentation_sections.items():
            doc_content = await self.generate_documentation_section(
                section_name=section,
                description=description,
                architecture=architecture,
                implementation=implementation
            )
            generated_docs[section] = doc_content
        
        return generated_docs
    
    async def generate_documentation_section(
        self,
        section_name: str,
        description: str,
        architecture: dict,
        implementation: dict
    ):
        """Generate specific documentation section"""
        
        generation_prompt = f"""
        Create comprehensive {section_name} documentation:
        
        Section Description: {description}
        
        Architecture Context:
        - API Design: {architecture.get('design_summary', 'N/A')}
        - Authentication: {architecture.get('auth_strategy', 'N/A')}
        - Database: {architecture.get('database_design', 'N/A')}
        
        Implementation Context:
        - Available Examples: {len(implementation)} code examples
        - Frameworks: FastAPI, SQLAlchemy, Pydantic
        
        Generate documentation with:
        - Clear explanations for different skill levels
        - Working code examples with full context
        - Common use cases and edge cases
        - Troubleshooting tips
        - Links to relevant resources
        
        Format as Markdown with proper structure and code highlighting.
        """
        
        doc_content = await self.mcp_client.perform_rag_query(
            query=generation_prompt,
            sources=["documentation_templates", "api_examples"],
            strategy="contextual"
        )
        
        return doc_content.content

# Example usage
async def api_documentation_example():
    workflow = APIDocumentationWorkflow()
    
    # Define API requirements
    api_requirements = """
    E-commerce API with the following features:
    - User authentication and authorization
    - Product catalog management
    - Shopping cart functionality
    - Order processing and payment integration
    - Inventory management
    - Admin dashboard APIs
    - Real-time notifications
    - Analytics and reporting
    """
    
    # Design architecture
    architecture = await workflow.design_api_architecture(api_requirements)
    print(f"Generated architecture with {len(architecture.results)} components")
    
    # Generate implementation
    implementation = await workflow.generate_api_implementation(architecture)
    print(f"Generated implementation for {len(implementation)} components")
    
    # Create documentation
    documentation = await workflow.create_comprehensive_documentation(
        architecture,
        implementation
    )
    print(f"Generated {len(documentation)} documentation sections")
    
    return {
        "architecture": architecture,
        "implementation": implementation, 
        "documentation": documentation
    }

result = asyncio.run(api_documentation_example())
```

### Use Case 3: Multi-Agent Code Review

**Scenario**: Comprehensive code review using specialized agents with different expertise areas.

```python
class MultiAgentCodeReview:
    """Comprehensive code review with specialized agents."""
    
    def __init__(self):
        self.review_agents = {
            "security": "security-specialist",
            "performance": "performance-analyzer", 
            "maintainability": "code-quality-expert",
            "testing": "test-coverage-analyzer",
            "documentation": "docs-reviewer"
        }
    
    async def comprehensive_code_review(
        self,
        code_changes: dict,
        project_context: dict
    ):
        """Orchestrate comprehensive multi-agent code review"""
        
        # Initialize swarm coordination for code review
        swarm_session = await self.initialize_review_swarm(code_changes, project_context)
        
        # Deploy specialized review agents
        review_tasks = []
        for review_area, agent_type in self.review_agents.items():
            task = self.deploy_review_agent(
                agent_type=agent_type,
                review_area=review_area,
                code_changes=code_changes,
                project_context=project_context,
                session=swarm_session
            )
            review_tasks.append(task)
        
        # Execute reviews in parallel
        review_results = await asyncio.gather(*review_tasks)
        
        # Aggregate and synthesize reviews
        comprehensive_review = await self.synthesize_review_results(review_results)
        
        # Generate actionable recommendations
        recommendations = await self.generate_recommendations(comprehensive_review)
        
        return {
            "individual_reviews": review_results,
            "comprehensive_analysis": comprehensive_review,
            "recommendations": recommendations,
            "session_id": swarm_session.id
        }
    
    async def deploy_review_agent(
        self,
        agent_type: str,
        review_area: str, 
        code_changes: dict,
        project_context: dict,
        session: Any
    ):
        """Deploy specialized agent for specific review area"""
        
        review_prompt = self.generate_review_prompt(
            review_area,
            code_changes,
            project_context
        )
        
        # Use Claude Flow task spawning
        review_task = f"""
        Perform {review_area} code review:
        
        {review_prompt}
        
        Coordinate with other review agents via session {session.id}.
        Share findings that overlap with other review areas.
        """
        
        agent_result = await self.spawn_coordinated_agent(
            agent_type=agent_type,
            task_description=review_task,
            session=session
        )
        
        return {
            "review_area": review_area,
            "agent_type": agent_type,
            "findings": agent_result.findings,
            "severity_scores": agent_result.severity_scores,
            "recommendations": agent_result.recommendations
        }
    
    def generate_review_prompt(
        self,
        review_area: str,
        code_changes: dict,
        project_context: dict
    ) -> str:
        """Generate specialized review prompt for each area"""
        
        prompts = {
            "security": f"""
            Security Code Review:
            
            Analyze the following code changes for security vulnerabilities:
            {code_changes['diff']}
            
            Project Context:
            - Framework: {project_context.get('framework', 'Unknown')}
            - Authentication: {project_context.get('auth_type', 'Unknown')}
            - Database: {project_context.get('database', 'Unknown')}
            
            Focus Areas:
            - Input validation and sanitization
            - Authentication and authorization flaws
            - SQL injection and XSS vulnerabilities
            - Cryptographic implementations
            - Sensitive data handling
            - Dependency vulnerabilities
            
            Provide:
            1. Identified security issues with severity ratings
            2. Specific code locations and explanations
            3. Remediation recommendations
            4. Additional security measures to consider
            """,
            
            "performance": f"""
            Performance Code Review:
            
            Analyze code changes for performance implications:
            {code_changes['diff']}
            
            Context:
            - Expected Load: {project_context.get('expected_load', 'Unknown')}
            - Performance Requirements: {project_context.get('performance_targets', 'Unknown')}
            
            Focus Areas:
            - Algorithm complexity and efficiency
            - Database query optimization
            - Memory usage and potential leaks
            - Caching opportunities
            - Async/await patterns
            - Resource utilization
            
            Provide:
            1. Performance bottlenecks and inefficiencies
            2. Big O analysis for algorithms
            3. Database query optimization suggestions
            4. Memory usage analysis
            5. Scalability considerations
            """,
            
            "maintainability": f"""
            Code Quality and Maintainability Review:
            
            Analyze code changes for maintainability:
            {code_changes['diff']}
            
            Focus Areas:
            - Code readability and clarity
            - Design patterns and architectural consistency
            - Error handling and edge cases
            - Code duplication and reusability
            - Naming conventions and documentation
            - SOLID principles adherence
            
            Provide:
            1. Code quality assessment with metrics
            2. Design pattern recommendations
            3. Refactoring opportunities
            4. Documentation improvements needed
            5. Long-term maintainability considerations
            """,
            
            "testing": f"""
            Test Coverage and Quality Review:
            
            Analyze testing approach for code changes:
            {code_changes['diff']}
            
            Existing Tests:
            {code_changes.get('test_files', 'No test information provided')}
            
            Focus Areas:
            - Test coverage completeness
            - Test quality and effectiveness
            - Edge case coverage
            - Integration test needs
            - Mock usage and test isolation
            - Test maintainability
            
            Provide:
            1. Test coverage analysis
            2. Missing test scenarios
            3. Test quality improvements
            4. Integration testing recommendations
            5. Test automation suggestions
            """
        }
        
        return prompts.get(review_area, f"Perform {review_area} review of the code changes.")
    
    async def synthesize_review_results(self, review_results: list) -> dict:
        """Synthesize individual review results into comprehensive analysis"""
        
        synthesis_prompt = f"""
        Synthesize the following specialized code review results:
        
        {json.dumps(review_results, indent=2)}
        
        Create a comprehensive analysis that:
        1. Identifies cross-cutting concerns across review areas
        2. Prioritizes issues by business impact and severity
        3. Highlights conflicting recommendations and provides resolution
        4. Creates an overall quality score and risk assessment
        5. Provides executive summary for non-technical stakeholders
        
        Format as structured analysis with clear priorities and actions.
        """
        
        client = MCPClient("http://localhost:8051")
        synthesis = await client.perform_rag_query(
            query=synthesis_prompt,
            strategy="agentic"
        )
        
        return synthesis.content

# Example usage
async def code_review_example():
    reviewer = MultiAgentCodeReview()
    
    # Example code changes
    code_changes = {
        "diff": """
        + async def create_user(user_data: dict, db: Session):
        +     # Hash password
        +     hashed_password = hash_password(user_data['password'])
        +     
        +     # Create user record
        +     user = User(
        +         email=user_data['email'],
        +         password_hash=hashed_password,
        +         created_at=datetime.utcnow()
        +     )
        +     
        +     db.add(user)
        +     db.commit()
        +     db.refresh(user)
        +     
        +     return user
        """,
        "files_changed": ["api/users.py", "models/user.py"],
        "test_files": ["tests/test_users.py"]
    }
    
    project_context = {
        "framework": "FastAPI",
        "database": "PostgreSQL",
        "auth_type": "JWT",
        "expected_load": "10,000 users",
        "performance_targets": "< 200ms response time"
    }
    
    # Execute comprehensive review
    review_result = await reviewer.comprehensive_code_review(
        code_changes=code_changes,
        project_context=project_context
    )
    
    print("Code Review Results:")
    print(f"- {len(review_result['individual_reviews'])} specialized reviews completed")
    print(f"- Session ID: {review_result['session_id']}")
    print(f"- Total recommendations: {len(review_result['recommendations'])}")
    
    return review_result

review = asyncio.run(code_review_example())
```

---

## Code Examples

### Example 1: Custom Agent Implementation

```python
# custom_agent_example.py
from typing import Dict, Any, List
from src.agents.base_agent import BaseAgent, BaseAgentOutput
from src.agents.mcp_client import MCPClient

class CustomDatabaseExpert(BaseAgent):
    """
    Custom agent specialized in database optimization and design.
    """
    
    def __init__(self):
        super().__init__()
        self.specialization = "database_optimization"
        self.capabilities = [
            "query_optimization",
            "schema_design", 
            "performance_tuning",
            "migration_planning",
            "index_optimization"
        ]
        self.mcp_client = MCPClient()
        
    async def can_handle(self, query: str, context: Dict[str, Any]) -> bool:
        """Determine if this agent can handle the query."""
        
        database_keywords = [
            "database", "sql", "query", "index", "schema",
            "postgresql", "mysql", "mongodb", "performance",
            "optimization", "migration", "backup", "replication"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in database_keywords)
    
    async def process(
        self, 
        query: str, 
        context: Dict[str, Any]
    ) -> BaseAgentOutput:
        """Process database-related query with specialized expertise."""
        
        # Analyze query type
        query_type = await self.analyze_query_type(query)
        
        # Search for relevant database knowledge
        knowledge_results = await self.mcp_client.perform_rag_query(
            query=f"{query} database optimization best practices",
            sources=["database_docs", "performance_guides"],
            strategy="contextual"
        )
        
        # Generate specialized response based on query type
        if query_type == "optimization":
            response = await self.generate_optimization_response(query, knowledge_results)
        elif query_type == "schema_design":
            response = await self.generate_schema_design_response(query, knowledge_results)
        elif query_type == "migration":
            response = await self.generate_migration_response(query, knowledge_results)
        else:
            response = await self.generate_general_response(query, knowledge_results)
        
        return BaseAgentOutput(
            content=response["content"],
            confidence=response["confidence"],
            sources=knowledge_results.sources,
            metadata={
                "agent_type": "database_expert",
                "query_type": query_type,
                "specialization_match": response["specialization_match"],
                "optimization_opportunities": response.get("optimizations", [])
            }
        )
    
    async def analyze_query_type(self, query: str) -> str:
        """Analyze the type of database query to provide specialized handling."""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["slow", "optimize", "performance", "speed"]):
            return "optimization"
        elif any(word in query_lower for word in ["schema", "design", "table", "relationship"]):
            return "schema_design"
        elif any(word in query_lower for word in ["migration", "upgrade", "migrate"]):
            return "migration"
        elif any(word in query_lower for word in ["backup", "restore", "recovery"]):
            return "backup_recovery"
        else:
            return "general"
    
    async def generate_optimization_response(
        self, 
        query: str, 
        knowledge_results: Any
    ) -> Dict[str, Any]:
        """Generate specialized optimization response."""
        
        # Extract performance-related information
        performance_info = [
            result for result in knowledge_results.results
            if any(perf_term in result.content.lower() 
                  for perf_term in ["performance", "optimize", "index", "query plan"])
        ]
        
        optimization_steps = []
        
        # Analyze for common optimization opportunities
        if "slow query" in query.lower():
            optimization_steps.extend([
                "Analyze query execution plan with EXPLAIN ANALYZE",
                "Check for missing indexes on WHERE clause columns",
                "Consider query rewriting for better performance",
                "Evaluate table statistics and consider ANALYZE"
            ])
        
        if "index" in query.lower():
            optimization_steps.extend([
                "Review current index usage with pg_stat_user_indexes",
                "Identify unused indexes consuming space",
                "Consider composite indexes for multi-column queries",
                "Evaluate partial indexes for filtered queries"
            ])
        
        response_content = f"""
        Database Performance Optimization Analysis:
        
        Query Analysis: {query}
        
        Recommended Optimization Steps:
        {chr(10).join(f"- {step}" for step in optimization_steps)}
        
        Relevant Knowledge Sources:
        {chr(10).join(f"- {result.title}: {result.snippet}" for result in performance_info[:3])}
        
        Additional Recommendations:
        - Monitor query performance over time
        - Consider connection pooling optimization
        - Evaluate hardware resources (CPU, memory, I/O)
        - Implement query caching where appropriate
        """
        
        return {
            "content": response_content,
            "confidence": 0.9,
            "specialization_match": True,
            "optimizations": optimization_steps
        }

# Usage example
async def custom_agent_example():
    """Example of using custom database expert agent."""
    
    db_expert = CustomDatabaseExpert()
    
    # Test query
    query = "My PostgreSQL queries are running slowly. How can I optimize database performance?"
    context = {
        "database_type": "postgresql",
        "approximate_data_size": "100GB",
        "query_patterns": ["SELECT with JOINs", "INSERT batches", "UPDATE operations"]
    }
    
    # Check if agent can handle query
    can_handle = await db_expert.can_handle(query, context)
    print(f"Can handle query: {can_handle}")
    
    if can_handle:
        # Process query
        result = await db_expert.process(query, context)
        
        print(f"Response confidence: {result.confidence}")
        print(f"Optimization opportunities: {len(result.metadata.get('optimization_opportunities', []))}")
        print(f"Response: {result.content[:200]}...")

asyncio.run(custom_agent_example())
```

### Example 2: Advanced RAG Implementation

```python
# advanced_rag_example.py
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class AdvancedRAGConfig:
    """Configuration for advanced RAG implementation."""
    
    # Embedding settings
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 800
    chunk_overlap: int = 200
    
    # Search settings
    hybrid_search_enabled: bool = True
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    
    # Reranking settings
    reranking_enabled: bool = True
    reranking_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    max_rerank_candidates: int = 20
    
    # Quality settings
    relevance_threshold: float = 0.75
    max_results: int = 5

class AdvancedRAGEngine:
    """
    Advanced RAG implementation with multiple enhancement strategies.
    """
    
    def __init__(self, config: AdvancedRAGConfig):
        self.config = config
        self.embedding_cache = {}
        self.query_cache = {}
        
    async def process_query(
        self, 
        query: str,
        context: Optional[Dict[str, Any]] = None,
        sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Process query using advanced RAG strategies."""
        
        # Step 1: Query preprocessing and expansion
        processed_query = await self.preprocess_query(query, context)
        
        # Step 2: Multi-strategy retrieval
        retrieval_results = await self.execute_multi_strategy_retrieval(
            processed_query,
            sources or []
        )
        
        # Step 3: Result fusion and deduplication
        fused_results = await self.fuse_retrieval_results(retrieval_results)
        
        # Step 4: Reranking for relevance optimization
        if self.config.reranking_enabled:
            final_results = await self.rerank_results(processed_query, fused_results)
        else:
            final_results = fused_results[:self.config.max_results]
        
        # Step 5: Response generation with context
        response = await self.generate_contextual_response(
            query=processed_query,
            results=final_results,
            original_context=context
        )
        
        return response
    
    async def preprocess_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Advanced query preprocessing with context integration."""
        
        processed = {
            "original_query": query,
            "expanded_query": query,
            "context_terms": [],
            "query_type": await self.classify_query_type(query),
            "domain": await self.identify_domain(query, context)
        }
        
        # Context-based query expansion
        if context:
            context_terms = self.extract_context_terms(context)
            processed["context_terms"] = context_terms
            
            # Expand query with relevant context
            if context_terms:
                expanded = f"{query} {' '.join(context_terms[:3])}"
                processed["expanded_query"] = expanded
        
        # Domain-specific query enhancement
        domain_keywords = await self.get_domain_keywords(processed["domain"])
        if domain_keywords:
            processed["domain_keywords"] = domain_keywords
        
        return processed
    
    async def execute_multi_strategy_retrieval(
        self,
        processed_query: Dict[str, Any],
        sources: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Execute multiple retrieval strategies in parallel."""
        
        strategies = {}
        
        # Strategy 1: Semantic vector search
        strategies["semantic"] = await self.semantic_vector_search(
            processed_query["expanded_query"],
            sources
        )
        
        # Strategy 2: Keyword-based search (if hybrid enabled)
        if self.config.hybrid_search_enabled:
            strategies["keyword"] = await self.keyword_search(
                processed_query["original_query"],
                sources
            )
        
        # Strategy 3: Context-aware search
        if processed_query.get("context_terms"):
            strategies["contextual"] = await self.context_aware_search(
                processed_query,
                sources
            )
        
        # Strategy 4: Domain-specific search
        if processed_query.get("domain_keywords"):
            strategies["domain"] = await self.domain_specific_search(
                processed_query,
                sources
            )
        
        return strategies
    
    async def semantic_vector_search(
        self, 
        query: str, 
        sources: List[str]
    ) -> List[Dict[str, Any]]:
        """Perform semantic vector search."""
        
        # Generate query embedding
        query_embedding = await self.get_embedding(query)
        
        # Search vector database
        search_results = await self.vector_db_search(
            embedding=query_embedding,
            sources=sources,
            limit=self.config.max_rerank_candidates
        )
        
        return search_results
    
    async def keyword_search(
        self, 
        query: str, 
        sources: List[str]
    ) -> List[Dict[str, Any]]:
        """Perform keyword-based search using BM25."""
        
        # Extract keywords from query
        keywords = await self.extract_keywords(query)
        
        # Perform BM25 search
        bm25_results = await self.bm25_search(
            keywords=keywords,
            sources=sources,
            limit=self.config.max_rerank_candidates
        )
        
        return bm25_results
    
    async def context_aware_search(
        self,
        processed_query: Dict[str, Any],
        sources: List[str]
    ) -> List[Dict[str, Any]]:
        """Context-aware search using project/session context."""
        
        context_query = f"{processed_query['original_query']} {' '.join(processed_query['context_terms'])}"
        
        # Weight context terms higher in search
        context_embedding = await self.get_context_weighted_embedding(
            query=processed_query['original_query'],
            context_terms=processed_query['context_terms']
        )
        
        context_results = await self.vector_db_search(
            embedding=context_embedding,
            sources=sources,
            limit=self.config.max_rerank_candidates // 2
        )
        
        return context_results
    
    async def fuse_retrieval_results(
        self,
        strategy_results: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Fuse results from multiple retrieval strategies."""
        
        # Combine all results
        all_results = []
        for strategy, results in strategy_results.items():
            for result in results:
                result["retrieval_strategy"] = strategy
                all_results.append(result)
        
        # Deduplicate based on content similarity
        deduplicated = await self.deduplicate_results(all_results)
        
        # Score fusion using different weights
        fusion_weights = {
            "semantic": 0.4,
            "keyword": 0.3,
            "contextual": 0.2,
            "domain": 0.1
        }
        
        for result in deduplicated:
            strategy = result["retrieval_strategy"]
            base_score = result.get("score", 0.5)
            weight = fusion_weights.get(strategy, 0.1)
            result["fused_score"] = base_score * weight
        
        # Sort by fused score
        sorted_results = sorted(deduplicated, key=lambda x: x["fused_score"], reverse=True)
        
        return sorted_results
    
    async def rerank_results(
        self,
        processed_query: Dict[str, Any],
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Rerank results using cross-encoder model."""
        
        if len(results) <= self.config.max_results:
            return results
        
        query = processed_query["original_query"]
        
        # Prepare pairs for cross-encoder
        pairs = [(query, result["content"]) for result in results]
        
        # Get reranking scores
        rerank_scores = await self.cross_encoder_score(pairs)
        
        # Update results with rerank scores
        for result, score in zip(results, rerank_scores):
            result["rerank_score"] = score
            result["final_score"] = (result["fused_score"] + score) / 2
        
        # Filter by relevance threshold and limit
        relevant_results = [
            result for result in results
            if result["rerank_score"] > self.config.relevance_threshold
        ]
        
        # Sort by final score and limit
        final_results = sorted(
            relevant_results,
            key=lambda x: x["final_score"],
            reverse=True
        )[:self.config.max_results]
        
        return final_results
    
    async def generate_contextual_response(
        self,
        query: Dict[str, Any],
        results: List[Dict[str, Any]],
        original_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive response with context."""
        
        response = {
            "query": query["original_query"],
            "results": results,
            "total_sources": len(results),
            "average_relevance": sum(r["final_score"] for r in results) / len(results) if results else 0,
            "strategies_used": list(set(r["retrieval_strategy"] for r in results)),
            "context_integration": {
                "context_terms_used": query.get("context_terms", []),
                "domain_identified": query.get("domain"),
                "query_type": query.get("query_type")
            }
        }
        
        # Add performance metrics
        response["performance_metrics"] = {
            "reranking_applied": self.config.reranking_enabled,
            "hybrid_search_used": self.config.hybrid_search_enabled,
            "relevance_threshold": self.config.relevance_threshold
        }
        
        return response

# Usage example
async def advanced_rag_example():
    """Example of advanced RAG implementation."""
    
    config = AdvancedRAGConfig(
        hybrid_search_enabled=True,
        reranking_enabled=True,
        relevance_threshold=0.8,
        max_results=5
    )
    
    rag_engine = AdvancedRAGEngine(config)
    
    # Example query with context
    query = "How to implement async database operations in FastAPI?"
    context = {
        "project_type": "web_api",
        "framework": "fastapi",
        "database": "postgresql",
        "previous_queries": [
            "FastAPI dependency injection",
            "SQLAlchemy async sessions"
        ]
    }
    
    # Execute advanced RAG processing
    response = await rag_engine.process_query(
        query=query,
        context=context,
        sources=["fastapi_docs", "sqlalchemy_docs", "async_patterns"]
    )
    
    print(f"Query: {response['query']}")
    print(f"Found {response['total_sources']} relevant sources")
    print(f"Average relevance: {response['average_relevance']:.2f}")
    print(f"Strategies used: {response['strategies_used']}")
    print(f"Domain identified: {response['context_integration']['domain_identified']}")
    
    for i, result in enumerate(response['results'], 1):
        print(f"\n{i}. {result['title']} (score: {result['final_score']:.2f})")
        print(f"   Strategy: {result['retrieval_strategy']}")
        print(f"   Content: {result['content'][:100]}...")

asyncio.run(advanced_rag_example())
```

---

## Configuration Samples

### Production Configuration Templates

#### Complete Environment Configuration

```bash
# .env.production
# Core system configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
DEBUG=false

# Database configuration
SUPABASE_URL=https://your-production-project.supabase.co
SUPABASE_SERVICE_KEY=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...
DATABASE_SSL_MODE=require
CONNECTION_POOL_SIZE=20
MAX_OVERFLOW=30

# AI Provider configuration
OPENAI_API_KEY=sk-your-production-openai-key
OPENAI_ORG_ID=org-your-organization-id
GEMINI_API_KEY=AIzaSyD-your-gemini-key
OLLAMA_BASE_URL=https://ollama.yourdomain.com

# Service ports and networking
ARCHON_UI_PORT=3737
ARCHON_SERVER_PORT=8181
ARCHON_MCP_PORT=8051
ARCHON_AGENTS_PORT=8052
HOST=0.0.0.0
CORS_ORIGINS=["https://yourdomain.com", "https://app.yourdomain.com"]

# Performance configuration
WORKERS=4
MAX_CONNECTIONS=1000
REDIS_URL=redis://redis-cluster.yourdomain.com:6379
CACHE_TTL=300

# Agent configuration
AGENTS_ENABLED=true
CLAUDE_FLOW_ENABLED=true
MAX_CONCURRENT_AGENTS=10
AGENT_TIMEOUT=300
MODEL_CACHE_SIZE=10GB

# Security configuration
SECRET_KEY=your-super-secret-production-key
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440
RATE_LIMIT_ENABLED=true
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=3600

# Monitoring and logging
LOGFIRE_ENABLED=true
LOGFIRE_TOKEN=your-logfire-token
PROMETHEUS_METRICS_ENABLED=true
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id

# Storage configuration
UPLOAD_MAX_SIZE=100MB
TEMP_FILE_CLEANUP=true
BACKUP_ENABLED=true
BACKUP_SCHEDULE="0 2 * * *"
```

#### Docker Compose Production Override

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  archon-server:
    environment:
      - ENVIRONMENT=production
      - WORKERS=4
      - LOG_LEVEL=INFO
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
      restart_policy:
        condition: on-failure
        max_attempts: 3
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8181/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  archon-agents:
    environment:
      - CUDA_VISIBLE_DEVICES=0,1  # For GPU support
      - MODEL_CACHE_SIZE=10GB
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
    volumes:
      - model-cache:/app/models:cached
      - /dev/shm:/dev/shm  # Shared memory for performance

  archon-mcp:
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '500m'
          memory: 1G

  nginx-proxy:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/ssl:/etc/ssl/certs
      - ./nginx/logs:/var/log/nginx
    depends_on:
      - archon-server
    deploy:
      replicas: 2

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 1gb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
    deploy:
      resources:
        limits:
          memory: 1G

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'

volumes:
  model-cache:
    driver: local
  redis-data:
    driver: local
  prometheus-data:
    driver: local

networks:
  default:
    driver: overlay
    attachable: true
```

#### Kubernetes Production Configuration

```yaml
# k8s/production/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: archon-config
  namespace: archon-production
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  WORKERS: "4"
  MAX_CONNECTIONS: "1000"
  AGENTS_ENABLED: "true"
  CLAUDE_FLOW_ENABLED: "true"
  RATE_LIMIT_ENABLED: "true"
  PROMETHEUS_METRICS_ENABLED: "true"

---
# k8s/production/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: archon-secrets
  namespace: archon-production
type: Opaque
stringData:
  supabase-url: "https://your-production-project.supabase.co"
  supabase-service-key: "your-encrypted-service-key"
  openai-api-key: "your-encrypted-openai-key"
  secret-key: "your-super-secret-production-key"
  jwt-secret: "your-jwt-secret-key"

---
# k8s/production/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: archon-server
  namespace: archon-production
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: archon-server
  template:
    metadata:
      labels:
        app: archon-server
        version: production
    spec:
      containers:
      - name: archon-server
        image: archon/server:latest
        ports:
        - containerPort: 8181
        env:
        - name: ENVIRONMENT
          valueFrom:
            configMapKeyRef:
              name: archon-config
              key: ENVIRONMENT
        - name: SUPABASE_URL
          valueFrom:
            secretKeyRef:
              name: archon-secrets
              key: supabase-url
        resources:
          requests:
            cpu: "1000m"
            memory: "2Gi"
          limits:
            cpu: "2000m"
            memory: "4Gi"
        livenessProbe:
          httpGet:
            path: /health
            port: 8181
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8181
          initialDelaySeconds: 5
          periodSeconds: 5
        securityContext:
          allowPrivilegeEscalation: false
          runAsNonRoot: true
          runAsUser: 1000
```

### Agent Configuration Templates

#### Master Agent Configuration

```python
# config/master_agent_config.py
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class MasterAgentProductionConfig:
    """Production configuration for Master Agent."""
    
    # Core settings
    model: str = "openai:gpt-4o"
    fallback_models: List[str] = field(default_factory=lambda: [
        "openai:gpt-4o-mini",
        "gemini:gemini-pro"
    ])
    max_retries: int = 3
    timeout: int = 120
    enable_rate_limiting: bool = True
    
    # Performance settings
    concurrent_requests: int = 10
    request_batch_size: int = 5
    cache_enabled: bool = True
    cache_ttl: int = 300
    
    # RAG configuration
    rag_strategies: List[str] = field(default_factory=lambda: [
        "contextual_embeddings",
        "hybrid_search",
        "agentic_rag",
        "reranking"
    ])
    
    # Swarm coordination
    swarm_coordination: bool = True
    max_coordinated_agents: int = 8
    coordination_timeout: int = 180
    
    # Quality thresholds
    confidence_threshold: float = 0.7
    relevance_threshold: float = 0.75
    max_response_length: int = 4000
    
    # Monitoring
    metrics_enabled: bool = True
    detailed_logging: bool = True
    performance_tracking: bool = True
    
    def get_deployment_config(self) -> Dict[str, Any]:
        """Get configuration optimized for deployment environment."""
        
        return {
            "model_settings": {
                "primary_model": self.model,
                "fallback_models": self.fallback_models,
                "timeout": self.timeout,
                "max_retries": self.max_retries
            },
            "performance_settings": {
                "concurrent_requests": self.concurrent_requests,
                "batch_size": self.request_batch_size,
                "cache_enabled": self.cache_enabled,
                "cache_ttl": self.cache_ttl
            },
            "rag_settings": {
                "strategies": self.rag_strategies,
                "confidence_threshold": self.confidence_threshold,
                "relevance_threshold": self.relevance_threshold
            },
            "coordination_settings": {
                "swarm_enabled": self.swarm_coordination,
                "max_agents": self.max_coordinated_agents,
                "timeout": self.coordination_timeout
            }
        }
```

#### RAG Strategy Configuration

```yaml
# config/rag_strategies.yaml
rag_configurations:
  contextual_embeddings:
    enabled: true
    embedding_model: "text-embedding-3-small"
    context_window: 2000
    improvement_target: 0.30
    cache_embeddings: true
    batch_size: 100
    
  hybrid_search:
    enabled: true
    semantic_weight: 0.7
    keyword_weight: 0.3
    bm25_k1: 1.2
    bm25_b: 0.75
    min_keyword_length: 3
    
  agentic_rag:
    enabled: true
    max_iterations: 3
    confidence_threshold: 0.85
    query_expansion: true
    result_validation: true
    iterative_refinement: true
    
  reranking:
    enabled: true
    model: "cross-encoder/ms-marco-MiniLM-L-2-v2"
    max_candidates: 20
    score_threshold: 0.75
    batch_reranking: true

performance_optimization:
  vector_search:
    index_type: "ivfflat"
    lists: 1000
    probes: 10
    ef_search: 100
    
  caching:
    query_cache_size: 10000
    query_cache_ttl: 300
    embedding_cache_size: 50000
    embedding_cache_ttl: 3600
    
  batch_processing:
    enabled: true
    batch_size: 50
    max_wait_time: 100
    concurrent_batches: 5

quality_control:
  relevance_threshold: 0.75
  max_results_per_strategy: 10
  deduplication_threshold: 0.9
  content_quality_filter: true
  source_diversity_factor: 0.2
```

This comprehensive examples and tutorials document provides practical guidance for implementing, configuring, and operating the Master Agent System across various use cases and deployment scenarios.