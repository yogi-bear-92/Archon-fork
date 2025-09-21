#!/usr/bin/env python3
"""
Script to update PydanticAI documentation source with comprehensive tags.

This script directly updates the knowledge base with proper tags for the PydanticAI
AI framework documentation, covering framework type, core capabilities, and features.
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


async def update_pydantic_ai_tags():
    """Update PydanticAI source with comprehensive tags."""
    
    # PydanticAI source ID
    source_id = "a51526d65470cb31"
    
    # Comprehensive tags covering all requested areas
    comprehensive_tags = [
        # Framework type
        "pydantic-ai",
        "ai-framework", 
        "python",
        
        # Core capabilities  
        "multi-agent",
        "llm-integration",
        "ai-providers",
        
        # Features
        "model-deployment",
        "testing-tools",
        "security-focused",
        
        # Technology focus
        "openai",
        "google-genai", 
        "language-models",
        
        # Additional technical tags
        "structured-responses",
        "type-hints",
        "graph-support",
        "system-prompts",
        "agent-dependencies",
        "temporal-integration",
        "fault-tolerant",
        "deterministic-workflows",
        "developer-tools",
        "ai-development",
        "ease-of-use",
        "pydantic",
        "python-framework"
    ]
    
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Create source management service
        source_service = SourceManagementService(supabase)
        
        print(f"🔄 Updating PydanticAI source ({source_id}) with comprehensive tags...")
        print(f"📋 Adding {len(comprehensive_tags)} tags covering:")
        print("   • Framework type (pydantic-ai, ai-framework, python)")
        print("   • Core capabilities (multi-agent, llm-integration, ai-providers)")
        print("   • Features (model-deployment, testing-tools, security-focused)")
        print("   • Technology focus (openai, google-genai, language-models)")
        print("   • Additional technical capabilities")
        
        # Update the source metadata with the new tags
        success, result = source_service.update_source_metadata(
            source_id=source_id,
            tags=comprehensive_tags
        )
        
        if success:
            print(f"✅ Successfully updated PydanticAI source with {len(comprehensive_tags)} comprehensive tags!")
            print(f"📊 Updated fields: {result.get('updated_fields', [])}")
            
            # Verify the update by fetching the source details
            success_verify, details = source_service.get_source_details(source_id)
            if success_verify:
                source_data = details.get('source', {})
                metadata = source_data.get('metadata', {})
                current_tags = metadata.get('tags', [])
                
                print(f"🔍 Verification: Source now has {len(current_tags)} tags")
                print(f"📋 Tags applied: {', '.join(current_tags[:10])}{'...' if len(current_tags) > 10 else ''}")
                
        else:
            print(f"❌ Failed to update PydanticAI source: {result.get('error')}")
            return False
            
    except Exception as e:
        print(f"❌ Error updating PydanticAI source tags: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    print("🚀 PydanticAI Documentation Tags Update Script")
    print("=" * 60)
    
    # Run the async function
    success = asyncio.run(update_pydantic_ai_tags())
    
    print("=" * 60)
    if success:
        print("✅ PydanticAI documentation source updated successfully!")
        print("🏷️  The source now has comprehensive tags covering:")
        print("   • Framework identification and type classification")
        print("   • Core AI development capabilities") 
        print("   • Security and deployment features")
        print("   • Provider and technology integrations")
        print("   • Developer experience enhancements")
    else:
        print("❌ Failed to update PydanticAI source tags")
        sys.exit(1)