#!/usr/bin/env python3
"""
Comprehensive Knowledge Base Tagging Implementation

This script implements the comprehensive tagging strategy based on research analysis.
It applies optimal tag sets to improve discoverability and search accuracy across
all knowledge base sources using direct HTTP API calls.

Priority Implementation Order:
1. AWS Labs MCP (56cb969b4f4e75d5) - Infrastructure management platform
2. Claude Flow Wiki (65516ba46d606b01) - Enterprise AI orchestration  
3. Claude Code (92913be64b1ead25) - Terminal-integrated AI coding tool
4. PydanticAI (a51526d65470cb31) - AI framework for multi-agent applications
5. Archon Repository (ccbb49fd5eb8b6a3) - AI coding assistant operating system
6. Enhance existing tagged documents
"""

import asyncio
import json
import sys
import os
from pathlib import Path
import httpx


class ComprehensiveTaggingImplementer:
    """Implements comprehensive tagging for all knowledge base sources using HTTP API."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        
    # Comprehensive tag sets based on research analysis
    TAG_SETS = {
        # AWS Labs MCP (56cb969b4f4e75d5) - Infrastructure management platform
        "56cb969b4f4e75d5": [
            # Core Technology & Platform
            "aws-labs", "mcp-protocol", "model-context-protocol", "infrastructure-management", 
            "cloud-infrastructure", "aws-services", "infrastructure-automation", "cloud-management",
            
            # Implementation & Integration
            "python-implementation", "typescript-support", "nodejs-integration", "sdk-development",
            "api-integration", "client-libraries", "server-implementation", "protocol-implementation",
            
            # DevOps & Operations
            "devops-tools", "automation-tools", "infrastructure-as-code", "deployment-automation",
            "resource-provisioning", "configuration-management", "operational-efficiency",
            
            # Security & Compliance
            "security-validation", "token-based-auth", "access-control", "security-frameworks",
            "compliance-tools", "security-best-practices", "authentication-systems",
            
            # Documentation & Development
            "developer-tools", "documentation-tools", "code-examples", "best-practices",
            "integration-guides", "setup-documentation", "troubleshooting-guides",
            
            # Performance & Quality
            "test-coverage", "code-quality", "performance-optimization", "reliability",
            "scalability", "monitoring-tools", "observability",
            
            # Open Source & Community
            "open-source", "community-driven", "collaborative-development", "transparency",
            "version-control", "github-integration", "contribution-guidelines"
        ],
        
        # Claude Flow Wiki (65516ba46d606b01) - Enterprise AI orchestration
        "65516ba46d606b01": [
            # Core AI Orchestration
            "claude-flow", "ai-orchestration", "multi-agent-systems", "swarm-intelligence", 
            "agent-coordination", "distributed-ai", "neural-networks", "ai-workflows",
            
            # Enterprise Features
            "enterprise-ai", "enterprise-grade", "production-ready", "scalable-ai",
            "business-intelligence", "enterprise-workflows", "corporate-automation",
            
            # Advanced AI Capabilities
            "neural-pattern-recognition", "truth-verification", "pair-programming", 
            "collaborative-ai", "self-organizing-systems", "adaptive-learning", "cognitive-patterns",
            
            # Technical Architecture
            "87-mcp-tools", "real-time-processing", "output-chaining", "continuous-training",
            "automated-processes", "intelligent-routing", "performance-optimization",
            
            # Development & Integration
            "ai-development", "workflow-automation", "integration-platform", "api-framework",
            "development-tools", "programming-assistance", "code-generation", "testing-automation",
            
            # Robustness & Performance
            "high-performance", "fault-tolerance", "reliability", "system-resilience", 
            "performance-monitoring", "bottleneck-analysis", "resource-optimization",
            
            # Documentation & Support
            "comprehensive-documentation", "tutorials", "best-practices", "troubleshooting",
            "configuration-guides", "deployment-guides", "maintenance-procedures"
        ],
        
        # Claude Code (92913be64b1ead25) - Terminal-integrated AI coding tool
        "92913be64b1ead25": [
            # Core Functionality
            "claude-code", "ai-coding-assistant", "terminal-integration", "code-generation",
            "natural-language-programming", "ai-powered-development", "coding-automation",
            
            # Development Workflow
            "multi-file-editing", "project-context", "architectural-decisions", "code-review",
            "version-control-integration", "git-integration", "workflow-enhancement",
            
            # AI & Language Model
            "anthropic-claude", "large-language-model", "natural-language-interface",
            "context-understanding", "intelligent-suggestions", "code-completion",
            
            # Testing & Quality
            "automated-testing", "test-generation", "code-quality", "bug-detection",
            "performance-analysis", "code-optimization", "refactoring-assistance",
            
            # Developer Experience
            "developer-productivity", "streamlined-workflow", "efficiency-tools",
            "reliability-focused", "user-experience", "command-line-interface",
            
            # Technical Integration
            "terminal-tools", "shell-integration", "cross-platform", "development-environment",
            "ide-alternative", "lightweight-tool", "fast-execution",
            
            # Documentation & Learning
            "documentation-tools", "code-documentation", "learning-assistance", 
            "best-practices", "coding-standards", "development-guides"
        ],
        
        # PydanticAI (a51526d65470cb31) - AI framework for multi-agent applications
        "a51526d65470cb31": [
            # Core Framework
            "pydantic-ai", "ai-framework", "python-framework", "type-safe-ai", 
            "structured-data", "data-validation", "schema-validation", "model-framework",
            
            # Multi-Agent Systems
            "multi-agent-applications", "agent-orchestration", "distributed-agents",
            "agent-communication", "collaborative-ai", "swarm-coordination",
            
            # AI Integration
            "openai-integration", "google-genai", "multiple-llm-providers", "llm-abstraction",
            "ai-provider-management", "model-switching", "ai-model-integration",
            
            # Development Tools
            "python-development", "async-programming", "modern-python", "developer-tools",
            "api-development", "web-frameworks", "microservices", "backend-development",
            
            # Data & Validation
            "data-modeling", "input-validation", "output-parsing", "type-checking",
            "runtime-validation", "schema-design", "data-structures", "serialization",
            
            # Deployment & Production
            "production-deployment", "scalable-applications", "containerization", 
            "cloud-deployment", "performance-optimization", "monitoring-integration",
            
            # Security & Best Practices
            "security-focused", "input-sanitization", "safe-execution", "best-practices",
            "error-handling", "logging-integration", "debugging-tools", "testing-support"
        ],
        
        # Archon Repository (ccbb49fd5eb8b6a3) - AI coding assistant operating system
        "ccbb49fd5eb8b6a3": [
            # Core Platform
            "archon", "ai-operating-system", "coding-assistant-platform", "ai-development-framework",
            "integrated-platform", "comprehensive-solution", "development-environment",
            
            # AI Enhancement
            "ai-powered-workflows", "intelligent-automation", "development-acceleration",
            "ai-assisted-coding", "smart-suggestions", "context-aware-assistance",
            
            # Documentation Framework
            "docusaurus-2", "documentation-platform", "technical-documentation", "api-docs",
            "user-guides", "developer-documentation", "interactive-docs",
            
            # Industry Applications
            "cross-industry", "versatile-platform", "scalable-solutions", "enterprise-ready",
            "production-applications", "business-automation", "workflow-optimization",
            
            # Integration & Collaboration
            "seamless-integration", "team-collaboration", "shared-workflows", "version-control",
            "project-management", "task-automation", "collaborative-development",
            
            # Security & Deployment
            "security-focused", "secure-deployment", "access-control", "privacy-protection",
            "enterprise-security", "compliance-ready", "audit-trails",
            
            # Development Support
            "development-acceleration", "productivity-enhancement", "learning-resources",
            "troubleshooting-support", "community-support", "extensive-examples"
        ]
    }
    
    # Enhanced tags for existing tagged documents
    ENHANCED_TAGS = {
        # Claude Flow Expert Agent Documentation (file_claude-flow-expert-agent_md_e532ca06)
        "file_claude-flow-expert-agent_md_e532ca06": [
            # Original tags maintained
            "claude-flow", "expert-agent", "configuration", "ai-consultant", "sparc",
            # Additional comprehensive tags
            "multi-agent-orchestration", "workflow-automation", "agent-coordination", 
            "intelligent-automation", "enterprise-ai", "swarm-intelligence", "neural-patterns",
            "performance-optimization", "distributed-systems", "ai-development", "best-practices",
            "integration-guides", "configuration-management", "troubleshooting", "monitoring-tools"
        ],
        
        # Claude Flow Expert System Summary (file_claude-flow-expert-system-summary_md_210ce9ff)
        "file_claude-flow-expert-system-summary_md_210ce9ff": [
            # Original tags maintained
            "claude-flow", "expert-system", "integration-complete", "summary", "archon",
            # Additional comprehensive tags
            "rag-enhanced-retrieval", "knowledge-management", "intelligent-query-processing",
            "code-examples", "comprehensive-guidance", "multi-agent-workflows", "system-optimization",
            "consultation-tool", "ai-expertise", "workflow-orchestration", "performance-guidance",
            "setup-optimization", "configuration-best-practices", "system-integration"
        ],
        
        # Claude Flow Integration Guide (file_claude-flow-integration-guide_md_49b30f00)
        "file_claude-flow-integration-guide_md_49b30f00": [
            # Original tags maintained
            "claude-flow", "integration", "guide", "multi-agent", "sparc",
            # Additional comprehensive tags
            "swarm-topologies", "agent-collaboration", "complex-problem-solving", "specialized-agents",
            "orchestration-framework", "neural-pattern-training", "performance-optimization",
            "knowledge-sharing", "progress-tracking", "result-management", "archon-integration",
            "structured-workflows", "coordination-patterns", "system-architecture"
        ]
    }
    
    async def update_source_tags(self, source_id: str, tags: list[str]) -> bool:
        """Update tags for a specific source using HTTP API."""
        try:
            print(f"Updating tags for source {source_id}...")
            print(f"  Tags to apply: {len(tags)} tags")
            
            # Prepare the metadata update
            update_data = {
                "metadata": {
                    "tags": tags
                }
            }
            
            # Make HTTP PUT request to update the knowledge item
            url = f"{self.base_url}/api/knowledge-items/{source_id}"
            response = await self.client.put(url, json=update_data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✅ Successfully updated tags for {source_id}")
                return True
            else:
                error_msg = f"HTTP {response.status_code}"
                if response.status_code == 404:
                    error_msg += " - Source not found"
                elif response.status_code >= 400:
                    try:
                        error_data = response.json()
                        error_msg += f" - {error_data.get('error', {}).get('detail', 'Unknown error')}"
                    except:
                        error_msg += f" - {response.text}"
                        
                print(f"  ❌ Failed to update tags for {source_id}: {error_msg}")
                return False
                
        except Exception as e:
            print(f"  ❌ Exception updating tags for {source_id}: {str(e)}")
            return False
    
    async def implement_all_tagging(self):
        """Implement comprehensive tagging for all sources."""
        print("🏷️ Starting Comprehensive Knowledge Base Tagging Implementation")
        print("=" * 70)
        
        success_count = 0
        total_count = 0
        
        # Update main sources with comprehensive tag sets
        print("\n📚 Implementing tags for main knowledge sources...")
        for source_id, tags in self.TAG_SETS.items():
            total_count += 1
            if await self.update_source_tags(source_id, tags):
                success_count += 1
        
        # Enhance existing tagged documents
        print("\n🔧 Enhancing existing tagged documents...")
        for source_id, tags in self.ENHANCED_TAGS.items():
            total_count += 1
            if await self.update_source_tags(source_id, tags):
                success_count += 1
        
        # Summary
        print("\n" + "=" * 70)
        print(f"🎯 Tagging Implementation Complete!")
        print(f"   Successfully updated: {success_count}/{total_count} sources")
        
        if success_count == total_count:
            print("   ✅ All sources updated successfully!")
        else:
            print(f"   ⚠️ {total_count - success_count} sources had issues")
        
        return success_count, total_count
    
    async def verify_tagging(self):
        """Verify that tagging was applied correctly using HTTP API."""
        print("\n🔍 Verifying tagging implementation...")
        
        all_source_ids = list(self.TAG_SETS.keys()) + list(self.ENHANCED_TAGS.keys())
        
        # Get all knowledge items to verify
        try:
            url = f"{self.base_url}/api/knowledge-items?per_page=100"
            response = await self.client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])
                
                # Create a lookup map
                source_map = {item["source_id"]: item for item in items}
                
                for source_id in all_source_ids:
                    if source_id in source_map:
                        item = source_map[source_id]
                        metadata = item.get("metadata", {})
                        tags = metadata.get("tags", [])
                        print(f"  {source_id}: {len(tags)} tags applied")
                    else:
                        print(f"  ❌ {source_id}: Source not found")
            else:
                print(f"  ❌ Failed to retrieve knowledge items for verification: HTTP {response.status_code}")
        except Exception as e:
            print(f"  ❌ Error during verification: {str(e)}")
    
    async def close(self):
        """Clean up resources."""
        await self.client.aclose()


async def main():
    """Main execution function."""
    implementer = None
    try:
        implementer = ComprehensiveTaggingImplementer()
        
        # Implement comprehensive tagging
        success_count, total_count = await implementer.implement_all_tagging()
        
        # Verify the implementation
        await implementer.verify_tagging()
        
        print(f"\n🏁 Final Status: {success_count}/{total_count} sources successfully tagged")
        
        if success_count == total_count:
            return 0  # Success
        else:
            return 1  # Partial failure
            
    except Exception as e:
        print(f"❌ Critical error during tagging implementation: {str(e)}")
        import traceback
        traceback.print_exc()
        return 2  # Critical failure
    finally:
        if implementer:
            await implementer.close()


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)