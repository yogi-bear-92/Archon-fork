#!/usr/bin/env python3
"""
Script to verify Claude Flow documentation repository tags.
This script checks the applied tags and provides a detailed breakdown.
"""

import asyncio
import sys

# Add the python src directory to path to import the services
sys.path.append('/Users/yogi/Projects/Archon-fork/python/src')

from server.services.source_management_service import SourceManagementService
from server.utils import get_supabase_client


async def verify_claude_flow_tags():
    """Verify Claude Flow documentation repository tags."""
    
    # Claude Flow documentation repository source ID
    source_id = "f276af742f2f3e44"
    
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Create source management service
        source_service = SourceManagementService(supabase)
        
        print("🔍 Verifying Claude Flow Documentation Repository Tags")
        print("=" * 60)
        
        # Fetch source details
        success, details = source_service.get_source_details(source_id)
        
        if not success:
            print(f"❌ Failed to fetch source details: {details.get('error')}")
            return False
            
        source_data = details.get('source', {})
        metadata = source_data.get('metadata', {})
        current_tags = metadata.get('tags', [])
        
        print(f"📊 Source ID: {source_id}")
        print(f"🏷️  Total Tags: {len(current_tags)}")
        print()
        
        # Categorize tags for verification
        tag_categories = {
            "Documentation Type": [
                "claude-flow-docs", "api-documentation", "architecture-docs", 
                "technical-documentation", "developer-documentation", 
                "operational-guides", "reference-material"
            ],
            "Core Focus": [
                "agent-orchestration", "deployment-guides", "development-workflows",
                "multi-agent-systems", "ai-swarm-coordination", "intelligent-routing"
            ],
            "Content Areas": [
                "swarm-coordination", "training-pipeline", "monitoring",
                "coordination-topologies", "neural-patterns", "performance-tracking"
            ],
            "Technology Stack": [
                "fastapi", "pydantic-ai", "socket-io", "postgresql", 
                "supabase", "pgvector", "docker", "kubernetes", "react", "typescript"
            ],
            "Advanced Capabilities": [
                "progressive-refinement", "rag-enhancement", "knowledge-management",
                "vector-search", "semantic-search", "context-management"
            ],
            "Integration": [
                "mcp-protocol", "github-integration", "claude-integration",
                "api-integration", "webhook-integration"
            ]
        }
        
        # Check each category
        for category, expected_tags in tag_categories.items():
            present_tags = [tag for tag in expected_tags if tag in current_tags]
            missing_tags = [tag for tag in expected_tags if tag not in current_tags]
            
            print(f"📚 {category}:")
            print(f"   ✅ Present ({len(present_tags)}): {', '.join(present_tags[:5])}{'...' if len(present_tags) > 5 else ''}")
            if missing_tags:
                print(f"   ❌ Missing ({len(missing_tags)}): {', '.join(missing_tags)}")
            print()
        
        # Show all tags in alphabetical order
        sorted_tags = sorted(current_tags)
        print("📋 All Applied Tags (Alphabetical):")
        for i, tag in enumerate(sorted_tags, 1):
            print(f"   {i:2d}. {tag}")
        
        print()
        print("✅ Tag verification complete!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error verifying tags: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Claude Flow Documentation Tags Verification")
    print()
    
    # Run the async function
    success = asyncio.run(verify_claude_flow_tags())
    
    if not success:
        sys.exit(1)