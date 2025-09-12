"""
AI Tagging API Routes

API endpoints for AI tag generation and management.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.server.config.logfire_config import get_logger, safe_logfire_error, safe_logfire_info
from src.server.services.ai_tagging_background_service import get_ai_tagging_background_service
from src.server.services.ai_tag_generation_service import get_ai_tag_service

logger = get_logger(__name__)

# Create router
ai_tagging_router = APIRouter(prefix="/api/ai-tagging", tags=["AI Tagging"])


class SourceUpdateRequest(BaseModel):
    source_id: str
    force_update: bool = False


class BulkUpdateRequest(BaseModel):
    limit: Optional[int] = None
    knowledge_types: Optional[List[str]] = None


class TagGenerationRequest(BaseModel):
    content: str
    knowledge_type: str = "technical"
    source_url: Optional[str] = None
    existing_tags: Optional[List[str]] = None
    max_tags: int = 10


@ai_tagging_router.post("/generate-tags")
async def generate_tags_for_content(request: TagGenerationRequest) -> Dict[str, Any]:
    """Generate AI tags for given content."""
    try:
        ai_tag_service = get_ai_tag_service()
        
        tags = await ai_tag_service.generate_tags_for_content(
            content=request.content,
            knowledge_type=request.knowledge_type,
            source_url=request.source_url,
            existing_tags=request.existing_tags,
            max_tags=request.max_tags,
        )
        
        safe_logfire_info(f"Generated {len(tags)} AI tags for content")
        
        return {
            "success": True,
            "tags": tags,
            "count": len(tags),
        }
        
    except Exception as e:
        logger.error(f"Tag generation failed: {e}", exc_info=True)
        safe_logfire_error(f"Tag generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@ai_tagging_router.post("/update-source")
async def update_source_with_ai_tags(
    request: SourceUpdateRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Update a specific source with AI-generated tags."""
    try:
        ai_tagging_service = get_ai_tagging_background_service()
        
        # Run in background to avoid blocking
        background_tasks.add_task(
            ai_tagging_service.update_source_tags_with_ai,
            request.source_id,
            request.force_update
        )
        
        safe_logfire_info(f"Started AI tag update for source {request.source_id}")
        
        return {
            "success": True,
            "message": f"AI tag update started for source {request.source_id}",
            "source_id": request.source_id,
        }
        
    except Exception as e:
        logger.error(f"Source update failed: {e}", exc_info=True)
        safe_logfire_error(f"Source update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@ai_tagging_router.post("/update-source-sync")
async def update_source_with_ai_tags_sync(request: SourceUpdateRequest) -> Dict[str, Any]:
    """Update a specific source with AI-generated tags (synchronous)."""
    try:
        ai_tagging_service = get_ai_tagging_background_service()
        
        result = await ai_tagging_service.update_source_tags_with_ai(
            request.source_id,
            request.force_update
        )
        
        if result["success"]:
            safe_logfire_info(f"Updated source {request.source_id} with AI tags")
        else:
            safe_logfire_error(f"Failed to update source {request.source_id}: {result.get('error')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Source update failed: {e}", exc_info=True)
        safe_logfire_error(f"Source update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@ai_tagging_router.post("/update-chunks")
async def update_document_chunks_with_ai_tags(
    source_id: str,
    batch_size: int = 10,
    background_tasks: BackgroundTasks = None
) -> Dict[str, Any]:
    """Update document chunks for a source with AI-generated tags."""
    try:
        ai_tagging_service = get_ai_tagging_background_service()
        
        if background_tasks:
            # Run in background
            background_tasks.add_task(
                ai_tagging_service.update_document_chunks_with_ai_tags,
                source_id,
                batch_size
            )
            
            return {
                "success": True,
                "message": f"AI tag update started for document chunks in source {source_id}",
                "source_id": source_id,
            }
        else:
            # Run synchronously
            result = await ai_tagging_service.update_document_chunks_with_ai_tags(
                source_id, batch_size
            )
            return result
        
    except Exception as e:
        logger.error(f"Chunk update failed: {e}", exc_info=True)
        safe_logfire_error(f"Chunk update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@ai_tagging_router.post("/bulk-update")
async def update_all_sources_with_ai_tags(
    request: BulkUpdateRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """Update all sources with AI-generated tags."""
    try:
        ai_tagging_service = get_ai_tagging_background_service()
        
        # Run in background as this can take a long time
        background_tasks.add_task(
            ai_tagging_service.update_all_sources_with_ai_tags,
            request.limit,
            request.knowledge_types
        )
        
        safe_logfire_info("Started bulk AI tag update for all sources")
        
        return {
            "success": True,
            "message": "Bulk AI tag update started",
            "limit": request.limit,
            "knowledge_types": request.knowledge_types,
        }
        
    except Exception as e:
        logger.error(f"Bulk update failed: {e}", exc_info=True)
        safe_logfire_error(f"Bulk update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@ai_tagging_router.get("/sources-without-ai-tags")
async def get_sources_without_ai_tags() -> Dict[str, Any]:
    """Get sources that don't have AI-generated tags."""
    try:
        ai_tagging_service = get_ai_tagging_background_service()
        
        sources = await ai_tagging_service.get_sources_without_ai_tags()
        
        return {
            "success": True,
            "sources": sources,
            "count": len(sources),
        }
        
    except Exception as e:
        logger.error(f"Failed to get sources without AI tags: {e}", exc_info=True)
        safe_logfire_error(f"Failed to get sources without AI tags: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@ai_tagging_router.get("/status")
async def get_ai_tagging_status() -> Dict[str, Any]:
    """Get AI tagging system status."""
    try:
        ai_tagging_service = get_ai_tagging_background_service()
        
        # Get sources without AI tags
        sources_without_ai = await ai_tagging_service.get_sources_without_ai_tags()
        
        # Get total sources count
        from src.server.services.client_manager import get_supabase_client
        supabase = get_supabase_client()
        total_sources_response = supabase.table("archon_sources").select("source_id", count="exact").execute()
        total_sources = total_sources_response.count or 0
        
        return {
            "success": True,
            "total_sources": total_sources,
            "sources_without_ai_tags": len(sources_without_ai),
            "sources_with_ai_tags": total_sources - len(sources_without_ai),
            "ai_tagging_enabled": True,
        }
        
    except Exception as e:
        logger.error(f"Failed to get AI tagging status: {e}", exc_info=True)
        safe_logfire_error(f"Failed to get AI tagging status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
