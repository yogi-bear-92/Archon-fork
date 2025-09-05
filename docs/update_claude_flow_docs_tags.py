#!/usr/bin/env python3
"""
Script to update Claude Flow documentation repository source with comprehensive tags.

This script directly updates the knowledge base with proper tags for the Claude Flow
documentation repository, covering documentation type, agent orchestration capabilities,
deployment guides, development workflows, and operational guidance.
"""

import asyncio
import os
import sys
import json
from datetime import datetime

# Add the python src directory to path to import the services
sys.path.append('/Users/yogi/Projects/Archon-fork/python/src')

from server.services.source_management_service import SourceManagementService
from server.utils import get_supabase_client


async def update_claude_flow_docs_tags():
    """Update Claude Flow documentation repository source with comprehensive tags."""
    
    # Claude Flow documentation repository source ID
    source_id = "f276af742f2f3e44"
    
    # Comprehensive tags covering all requested areas
    comprehensive_tags = [
        # Documentation type
        "claude-flow-docs",
        "api-documentation", 
        "architecture-docs",
        "technical-documentation",
        "developer-documentation",
        "operational-guides",
        "reference-material",
        
        # Core focus - Agent orchestration
        "agent-orchestration",
        "multi-agent-systems",
        "ai-swarm-coordination",
        "intelligent-routing",
        "agent-collaboration",
        "distributed-ai",
        
        # Core focus - Deployment guides  
        "deployment-guides",
        "system-deployment",
        "containerization",
        "docker-compose",
        "microservices-deployment",
        "production-deployment",
        "scalability-guidance",
        
        # Core focus - Development workflows
        "development-workflows", 
        "sparc-methodology",
        "tdd-workflows",
        "agent-development",
        "workflow-automation",
        "development-patterns",
        "best-practices",
        
        # Content areas - Swarm coordination
        "swarm-coordination",
        "coordination-topologies",
        "mesh-networking",
        "hierarchical-coordination", 
        "consensus-mechanisms",
        "distributed-coordination",
        "fault-tolerance",
        
        # Content areas - Training pipeline
        "training-pipeline",
        "neural-patterns",
        "pattern-learning",
        "performance-optimization",
        "cognitive-patterns",
        "adaptive-learning",
        "model-training",
        
        # Content areas - Monitoring
        "monitoring",
        "performance-tracking",
        "system-health",
        "metrics-collection",
        "observability",
        "real-time-monitoring",
        "bottleneck-analysis",
        
        # Usage context - Developer documentation
        "setup-guides",
        "integration-guides",
        "code-examples",
        "tutorials",
        "troubleshooting",
        "debugging-guides",
        "configuration-management",
        
        # Usage context - Operational guides
        "system-administration",
        "maintenance-procedures",
        "scaling-strategies",
        "backup-recovery",
        "security-configuration",
        "environment-management",
        "disaster-recovery",
        
        # Technology and platform tags
        "fastapi",
        "pydantic-ai", 
        "socket-io",
        "postgresql",
        "supabase",
        "pgvector",
        "docker",
        "kubernetes",
        "react",
        "typescript",
        
        # Advanced capabilities
        "progressive-refinement",
        "rag-enhancement",
        "knowledge-management",
        "vector-search",
        "semantic-search",
        "context-management",
        "memory-management",
        
        # Integration aspects
        "mcp-protocol",
        "github-integration",
        "claude-integration",
        "ai-client-integration",
        "api-integration",
        "webhook-integration",
        
        # Performance and optimization
        "performance-benchmarks",
        "optimization-techniques",
        "resource-management",
        "load-balancing",
        "caching-strategies",
        "query-optimization",
        
        # Architectural patterns
        "microservices-architecture",
        "event-driven-architecture", 
        "layered-architecture",
        "separation-of-concerns",
        "design-patterns",
        "system-design"
    ]
    
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Create source management service
        source_service = SourceManagementService(supabase)
        
        print(f"🔄 Updating Claude Flow documentation repository ({source_id}) with comprehensive tags...")
        print(f"📋 Adding {len(comprehensive_tags)} tags covering:")
        print("   • Documentation type (claude-flow-docs, api-documentation, architecture-docs)")
        print("   • Core focus (agent-orchestration, deployment-guides, development-workflows)")  
        print("   • Content areas (swarm-coordination, training-pipeline, monitoring)")
        print("   • Usage context (developer-documentation, operational-guides, reference-material)")
        print("   • Technology stack and integration capabilities")
        print("   • Advanced features and architectural patterns")
        
        # Update the source metadata with the new tags
        success, result = source_service.update_source_metadata(
            source_id=source_id,
            tags=comprehensive_tags
        )
        
        if success:
            print(f"✅ Successfully updated Claude Flow documentation repository with {len(comprehensive_tags)} comprehensive tags!")
            print(f"📊 Updated fields: {result.get('updated_fields', [])}")
            
            # Verify the update by fetching the source details
            success_verify, details = source_service.get_source_details(source_id)
            if success_verify:
                source_data = details.get('source', {})
                metadata = source_data.get('metadata', {})
                current_tags = metadata.get('tags', [])
                
                print(f"🔍 Verification: Source now has {len(current_tags)} tags")
                print(f"📋 Tags applied: {', '.join(current_tags[:15])}{'...' if len(current_tags) > 15 else ''}")
                
                # Show tag categories for verification
                doc_type_tags = [tag for tag in current_tags if tag in ["claude-flow-docs", "api-documentation", "architecture-docs"]]
                core_focus_tags = [tag for tag in current_tags if tag in ["agent-orchestration", "deployment-guides", "development-workflows"]]
                content_area_tags = [tag for tag in current_tags if tag in ["swarm-coordination", "training-pipeline", "monitoring"]]
                
                print(f"🏷️  Documentation type tags: {', '.join(doc_type_tags)}")
                print(f"🎯 Core focus tags: {', '.join(core_focus_tags)}")
                print(f"📚 Content area tags: {', '.join(content_area_tags)}")
                
        else:
            print(f"❌ Failed to update Claude Flow documentation repository: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating Claude Flow documentation tags: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("🚀 Claude Flow Documentation Repository Tags Update Script")
    print("📖 Comprehensive Documentation Hub for Claude Flow Platform")
    print("=" * 70)
    
    # Run the async function
    success = asyncio.run(update_claude_flow_docs_tags())
    
    print("=" * 70)
    if success:
        print("✅ Claude Flow documentation repository updated successfully!")
        print("🏷️  The source now has comprehensive tags covering:")
        print("   📚 Documentation Types:")
        print("      • API documentation and technical references")
        print("      • Architecture documentation and system design")
        print("      • Developer guides and operational procedures")
        print("   🎯 Core Focus Areas:")
        print("      • Agent orchestration and multi-agent systems")
        print("      • Deployment guides and scalability strategies")  
        print("      • Development workflows and SPARC methodology")
        print("   📋 Content Areas:")
        print("      • Swarm coordination and topology management")
        print("      • Training pipeline and neural pattern learning")
        print("      • Monitoring, performance tracking, and observability")
        print("   🛠️  Usage Context:")
        print("      • Developer documentation with setup and integration guides")
        print("      • Operational guides for system administration")
        print("      • Reference material for troubleshooting and optimization")
        print("   🚀 Advanced Capabilities:")
        print("      • Progressive refinement and RAG enhancement")
        print("      • Microservices architecture and integration patterns")
        print("      • Performance optimization and resource management")
    else:
        print("❌ Failed to update Claude Flow documentation repository tags")
        sys.exit(1)