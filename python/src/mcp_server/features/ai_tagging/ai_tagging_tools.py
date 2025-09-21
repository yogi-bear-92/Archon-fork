"""
AI Tagging MCP Tools

MCP tools for AI tag generation and management.
These tools expose the AI tagging functionality through the MCP protocol.
"""

import json
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import Context, FastMCP


def register_ai_tagging_tools(mcp: FastMCP):
    """Register AI tagging tools with the MCP server."""

    @mcp.tool()
    async def generate_ai_tags(
        ctx: Context,
        content: str,
        knowledge_type: str = "technical",
        source_url: str = None,
        existing_tags: str = None,
        max_tags: int = 10,
    ) -> str:
        """
        Generate AI tags for given content.
        
        Args:
            content: The content to analyze and generate tags for
            knowledge_type: Type of knowledge (technical, documentation, tutorial, etc.)
            source_url: Optional URL for context
            existing_tags: JSON string of existing tags to avoid duplicating
            max_tags: Maximum number of tags to generate (default: 10)
            
        Returns:
            JSON string with generated tags and metadata
        """
        try:
            import httpx
            
            # Parse existing tags if provided
            existing_tags_list = []
            if existing_tags:
                try:
                    existing_tags_list = json.loads(existing_tags)
                except json.JSONDecodeError:
                    existing_tags_list = []
            
            # Prepare request data
            request_data = {
                "content": content,
                "knowledge_type": knowledge_type,
                "source_url": source_url,
                "existing_tags": existing_tags_list,
                "max_tags": max_tags,
            }
            
            # Make HTTP request to AI tagging API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8181/api/ai-tagging/generate-tags",
                    json=request_data,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return json.dumps(result, indent=2)
                else:
                    error_detail = response.text
                    return json.dumps({
                        "success": False,
                        "error": f"API request failed with status {response.status_code}",
                        "detail": error_detail
                    }, indent=2)
                    
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to generate AI tags: {str(e)}"
            }, indent=2)

    @mcp.tool()
    async def update_source_with_ai_tags(
        ctx: Context,
        source_id: str,
        force_update: bool = False,
    ) -> str:
        """
        Update a specific source with AI-generated tags.
        
        Args:
            source_id: The source ID to update with AI tags
            force_update: Whether to update even if tags already exist
            
        Returns:
            JSON string with update results
        """
        try:
            import httpx
            
            # Prepare request data
            request_data = {
                "source_id": source_id,
                "force_update": force_update,
            }
            
            # Make HTTP request to AI tagging API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8181/api/ai-tagging/update-source-sync",
                    json=request_data,
                    timeout=60.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return json.dumps(result, indent=2)
                else:
                    error_detail = response.text
                    return json.dumps({
                        "success": False,
                        "error": f"API request failed with status {response.status_code}",
                        "detail": error_detail
                    }, indent=2)
                    
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to update source with AI tags: {str(e)}"
            }, indent=2)

    @mcp.tool()
    async def update_document_chunks_with_ai_tags(
        ctx: Context,
        source_id: str,
        batch_size: int = 10,
    ) -> str:
        """
        Update document chunks for a source with AI-generated tags.
        
        Args:
            source_id: The source ID to update chunks for
            batch_size: Number of chunks to process at once
            
        Returns:
            JSON string with update results
        """
        try:
            import httpx
            
            # Make HTTP request to AI tagging API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"http://localhost:8181/api/ai-tagging/update-chunks?source_id={source_id}&batch_size={batch_size}",
                    timeout=120.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return json.dumps(result, indent=2)
                else:
                    error_detail = response.text
                    return json.dumps({
                        "success": False,
                        "error": f"API request failed with status {response.status_code}",
                        "detail": error_detail
                    }, indent=2)
                    
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to update document chunks with AI tags: {str(e)}"
            }, indent=2)

    @mcp.tool()
    async def bulk_update_ai_tags(
        ctx: Context,
        limit: int = None,
        knowledge_types: str = None,
    ) -> str:
        """
        Update all sources with AI-generated tags.
        
        Args:
            limit: Maximum number of sources to process
            knowledge_types: JSON string of knowledge types to filter by
            
        Returns:
            JSON string with bulk update results
        """
        try:
            import httpx
            
            # Parse knowledge types if provided
            knowledge_types_list = None
            if knowledge_types:
                try:
                    knowledge_types_list = json.loads(knowledge_types)
                except json.JSONDecodeError:
                    knowledge_types_list = None
            
            # Prepare request data
            request_data = {
                "limit": limit,
                "knowledge_types": knowledge_types_list,
            }
            
            # Make HTTP request to AI tagging API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:8181/api/ai-tagging/bulk-update",
                    json=request_data,
                    timeout=300.0  # 5 minutes for bulk operations
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return json.dumps(result, indent=2)
                else:
                    error_detail = response.text
                    return json.dumps({
                        "success": False,
                        "error": f"API request failed with status {response.status_code}",
                        "detail": error_detail
                    }, indent=2)
                    
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to perform bulk AI tag update: {str(e)}"
            }, indent=2)

    @mcp.tool()
    async def get_sources_without_ai_tags(ctx: Context) -> str:
        """
        Get sources that don't have AI-generated tags.
        
        Returns:
            JSON string with list of sources needing AI tags
        """
        try:
            import httpx
            
            # Make HTTP request to AI tagging API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://localhost:8181/api/ai-tagging/sources-without-ai-tags",
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return json.dumps(result, indent=2)
                else:
                    error_detail = response.text
                    return json.dumps({
                        "success": False,
                        "error": f"API request failed with status {response.status_code}",
                        "detail": error_detail
                    }, indent=2)
                    
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to get sources without AI tags: {str(e)}"
            }, indent=2)

    @mcp.tool()
    async def get_ai_tagging_status(ctx: Context) -> str:
        """
        Get AI tagging system status and statistics.
        
        Returns:
            JSON string with AI tagging system status
        """
        try:
            import httpx
            
            # Make HTTP request to AI tagging API
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "http://localhost:8181/api/ai-tagging/status",
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return json.dumps(result, indent=2)
                else:
                    error_detail = response.text
                    return json.dumps({
                        "success": False,
                        "error": f"API request failed with status {response.status_code}",
                        "detail": error_detail
                    }, indent=2)
                    
        except Exception as e:
            return json.dumps({
                "success": False,
                "error": f"Failed to get AI tagging status: {str(e)}"
            }, indent=2)
