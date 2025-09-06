#!/bin/bash

# Comprehensive Knowledge Base Tagging Implementation Script
# Applies optimal tag sets to improve discoverability and search accuracy

set -e

BASE_URL="http://localhost:8080"
API_URL="${BASE_URL}/api"

echo "🏷️ Starting Comprehensive Knowledge Base Tagging Implementation"
echo "=" * 70

# Function to update source tags
update_source_tags() {
    local source_id="$1"
    local tags_json="$2"
    local description="$3"
    
    echo "📝 Updating tags for $description ($source_id)..."
    
    # Create payload
    local payload=$(cat <<EOF
{
  "metadata": {
    "tags": $tags_json
  }
}
EOF
)
    
    # Make the API call
    response=$(curl -s -w "%{http_code}" -o /tmp/response.json \
        -X PUT \
        -H "Content-Type: application/json" \
        -d "$payload" \
        "$API_URL/knowledge-items/$source_id")
    
    if [[ "$response" == "200" ]]; then
        tag_count=$(echo "$tags_json" | jq 'length')
        echo "  ✅ Successfully applied $tag_count tags"
    else
        echo "  ❌ Failed with HTTP $response"
        if [[ -f /tmp/response.json ]]; then
            echo "  Response: $(cat /tmp/response.json)"
        fi
        return 1
    fi
}

# AWS Labs MCP (56cb969b4f4e75d5) - Infrastructure management platform
echo ""
echo "🏗️ Applying AWS Labs MCP tags..."
AWS_TAGS='[
    "aws-labs", "mcp-protocol", "model-context-protocol", "infrastructure-management", 
    "cloud-infrastructure", "aws-services", "infrastructure-automation", "cloud-management",
    "python-implementation", "typescript-support", "nodejs-integration", "sdk-development",
    "api-integration", "client-libraries", "server-implementation", "protocol-implementation",
    "devops-tools", "automation-tools", "infrastructure-as-code", "deployment-automation",
    "resource-provisioning", "configuration-management", "operational-efficiency",
    "security-validation", "token-based-auth", "access-control", "security-frameworks",
    "compliance-tools", "security-best-practices", "authentication-systems",
    "developer-tools", "documentation-tools", "code-examples", "best-practices",
    "integration-guides", "setup-documentation", "troubleshooting-guides",
    "test-coverage", "code-quality", "performance-optimization", "reliability",
    "scalability", "monitoring-tools", "observability",
    "open-source", "community-driven", "collaborative-development", "transparency",
    "version-control", "github-integration", "contribution-guidelines"
]'

update_source_tags "56cb969b4f4e75d5" "$AWS_TAGS" "AWS Labs MCP"

# Claude Flow Wiki (65516ba46d606b01) - Enterprise AI orchestration
echo ""
echo "🤖 Applying Claude Flow Wiki tags..."
CLAUDE_FLOW_TAGS='[
    "claude-flow", "ai-orchestration", "multi-agent-systems", "swarm-intelligence", 
    "agent-coordination", "distributed-ai", "neural-networks", "ai-workflows",
    "enterprise-ai", "enterprise-grade", "production-ready", "scalable-ai",
    "business-intelligence", "enterprise-workflows", "corporate-automation",
    "neural-pattern-recognition", "truth-verification", "pair-programming", 
    "collaborative-ai", "self-organizing-systems", "adaptive-learning", "cognitive-patterns",
    "87-mcp-tools", "real-time-processing", "output-chaining", "continuous-training",
    "automated-processes", "intelligent-routing", "performance-optimization",
    "ai-development", "workflow-automation", "integration-platform", "api-framework",
    "development-tools", "programming-assistance", "code-generation", "testing-automation",
    "high-performance", "fault-tolerance", "reliability", "system-resilience", 
    "performance-monitoring", "bottleneck-analysis", "resource-optimization",
    "comprehensive-documentation", "tutorials", "best-practices", "troubleshooting",
    "configuration-guides", "deployment-guides", "maintenance-procedures"
]'

update_source_tags "65516ba46d606b01" "$CLAUDE_FLOW_TAGS" "Claude Flow Wiki"

# Claude Code (92913be64b1ead25) - Terminal-integrated AI coding tool
echo ""
echo "💻 Applying Claude Code tags..."
CLAUDE_CODE_TAGS='[
    "claude-code", "ai-coding-assistant", "terminal-integration", "code-generation",
    "natural-language-programming", "ai-powered-development", "coding-automation",
    "multi-file-editing", "project-context", "architectural-decisions", "code-review",
    "version-control-integration", "git-integration", "workflow-enhancement",
    "anthropic-claude", "large-language-model", "natural-language-interface",
    "context-understanding", "intelligent-suggestions", "code-completion",
    "automated-testing", "test-generation", "code-quality", "bug-detection",
    "performance-analysis", "code-optimization", "refactoring-assistance",
    "developer-productivity", "streamlined-workflow", "efficiency-tools",
    "reliability-focused", "user-experience", "command-line-interface",
    "terminal-tools", "shell-integration", "cross-platform", "development-environment",
    "ide-alternative", "lightweight-tool", "fast-execution",
    "documentation-tools", "code-documentation", "learning-assistance", 
    "best-practices", "coding-standards", "development-guides"
]'

update_source_tags "92913be64b1ead25" "$CLAUDE_CODE_TAGS" "Claude Code"

# PydanticAI (a51526d65470cb31) - AI framework for multi-agent applications
echo ""
echo "🐍 Applying PydanticAI tags..."
PYDANTIC_AI_TAGS='[
    "pydantic-ai", "ai-framework", "python-framework", "type-safe-ai", 
    "structured-data", "data-validation", "schema-validation", "model-framework",
    "multi-agent-applications", "agent-orchestration", "distributed-agents",
    "agent-communication", "collaborative-ai", "swarm-coordination",
    "openai-integration", "google-genai", "multiple-llm-providers", "llm-abstraction",
    "ai-provider-management", "model-switching", "ai-model-integration",
    "python-development", "async-programming", "modern-python", "developer-tools",
    "api-development", "web-frameworks", "microservices", "backend-development",
    "data-modeling", "input-validation", "output-parsing", "type-checking",
    "runtime-validation", "schema-design", "data-structures", "serialization",
    "production-deployment", "scalable-applications", "containerization", 
    "cloud-deployment", "performance-optimization", "monitoring-integration",
    "security-focused", "input-sanitization", "safe-execution", "best-practices",
    "error-handling", "logging-integration", "debugging-tools", "testing-support"
]'

update_source_tags "a51526d65470cb31" "$PYDANTIC_AI_TAGS" "PydanticAI"

# Archon Repository (ccbb49fd5eb8b6a3) - AI coding assistant operating system
echo ""
echo "🏛️ Applying Archon Repository tags..."
ARCHON_TAGS='[
    "archon", "ai-operating-system", "coding-assistant-platform", "ai-development-framework",
    "integrated-platform", "comprehensive-solution", "development-environment",
    "ai-powered-workflows", "intelligent-automation", "development-acceleration",
    "ai-assisted-coding", "smart-suggestions", "context-aware-assistance",
    "docusaurus-2", "documentation-platform", "technical-documentation", "api-docs",
    "user-guides", "developer-documentation", "interactive-docs",
    "cross-industry", "versatile-platform", "scalable-solutions", "enterprise-ready",
    "production-applications", "business-automation", "workflow-optimization",
    "seamless-integration", "team-collaboration", "shared-workflows", "version-control",
    "project-management", "task-automation", "collaborative-development",
    "security-focused", "secure-deployment", "access-control", "privacy-protection",
    "enterprise-security", "compliance-ready", "audit-trails",
    "development-acceleration", "productivity-enhancement", "learning-resources",
    "troubleshooting-support", "community-support", "extensive-examples"
]'

update_source_tags "ccbb49fd5eb8b6a3" "$ARCHON_TAGS" "Archon Repository"

# Enhanced tags for existing tagged documents
echo ""
echo "🔧 Enhancing existing tagged documents..."

# Claude Flow Expert Agent Documentation (file_claude-flow-expert-agent_md_e532ca06)
EXPERT_AGENT_ENHANCED_TAGS='[
    "claude-flow", "expert-agent", "configuration", "ai-consultant", "sparc",
    "multi-agent-orchestration", "workflow-automation", "agent-coordination", 
    "intelligent-automation", "enterprise-ai", "swarm-intelligence", "neural-patterns",
    "performance-optimization", "distributed-systems", "ai-development", "best-practices",
    "integration-guides", "configuration-management", "troubleshooting", "monitoring-tools"
]'

update_source_tags "file_claude-flow-expert-agent_md_e532ca06" "$EXPERT_AGENT_ENHANCED_TAGS" "Claude Flow Expert Agent Documentation"

# Claude Flow Expert System Summary (file_claude-flow-expert-system-summary_md_210ce9ff)
EXPERT_SYSTEM_ENHANCED_TAGS='[
    "claude-flow", "expert-system", "integration-complete", "summary", "archon",
    "rag-enhanced-retrieval", "knowledge-management", "intelligent-query-processing",
    "code-examples", "comprehensive-guidance", "multi-agent-workflows", "system-optimization",
    "consultation-tool", "ai-expertise", "workflow-orchestration", "performance-guidance",
    "setup-optimization", "configuration-best-practices", "system-integration"
]'

update_source_tags "file_claude-flow-expert-system-summary_md_210ce9ff" "$EXPERT_SYSTEM_ENHANCED_TAGS" "Claude Flow Expert System Summary"

# Claude Flow Integration Guide (file_claude-flow-integration-guide_md_49b30f00)
INTEGRATION_GUIDE_ENHANCED_TAGS='[
    "claude-flow", "integration", "guide", "multi-agent", "sparc",
    "swarm-topologies", "agent-collaboration", "complex-problem-solving", "specialized-agents",
    "orchestration-framework", "neural-pattern-training", "performance-optimization",
    "knowledge-sharing", "progress-tracking", "result-management", "archon-integration",
    "structured-workflows", "coordination-patterns", "system-architecture"
]'

update_source_tags "file_claude-flow-integration-guide_md_49b30f00" "$INTEGRATION_GUIDE_ENHANCED_TAGS" "Claude Flow Integration Guide"

echo ""
echo "=" * 70
echo "🎯 Comprehensive tagging implementation completed!"
echo ""

# Verification
echo "🔍 Verifying implementation..."
echo "Getting updated knowledge sources to verify tagging..."

# Get knowledge items to verify
curl -s -X GET "$API_URL/knowledge-items?per_page=100" | jq -r '
.items[] | 
select(.source_id | 
    (. == "56cb969b4f4e75d5" or 
     . == "65516ba46d606b01" or 
     . == "92913be64b1ead25" or 
     . == "a51526d65470cb31" or 
     . == "ccbb49fd5eb8b6a3" or 
     . == "file_claude-flow-expert-agent_md_e532ca06" or 
     . == "file_claude-flow-expert-system-summary_md_210ce9ff" or 
     . == "file_claude-flow-integration-guide_md_49b30f00")
) | 
"  \(.source_id): \(.metadata.tags | length) tags applied"
' 2>/dev/null || echo "  ⚠️ Could not verify tagging - server may not be running"

echo ""
echo "🏁 Implementation completed successfully!"
echo "All comprehensive tag sets have been applied to improve knowledge base discoverability."