"""
AI Tagging Background Service

Handles background AI tag generation for existing knowledge entries.
This service can be used to retroactively add AI tags to content that was crawled before AI tagging was implemented.
"""

import asyncio
from typing import Any, Dict, List, Optional

from ..config.logfire_config import get_logger, safe_logfire_error, safe_logfire_info
from .ai_tag_generation_service import get_ai_tag_service
from .client_manager import get_supabase_client

logger = get_logger(__name__)


class AITaggingBackgroundService:
    """Service for background AI tag generation and updates."""

    def __init__(self):
        self.logger = logger
        self.supabase_client = get_supabase_client()

    async def update_source_tags_with_ai(
        self, 
        source_id: str, 
        force_update: bool = False
    ) -> Dict[str, Any]:
        """
        Update tags for a specific source using AI analysis.
        
        Args:
            source_id: The source ID to update
            force_update: Whether to update even if tags already exist
            
        Returns:
            Dict with update results
        """
        try:
            # Get the source record
            source_response = self.supabase_client.table("archon_sources").select("*").eq("source_id", source_id).execute()
            
            if not source_response.data:
                return {"success": False, "error": f"Source {source_id} not found"}
            
            source_data = source_response.data[0]
            existing_tags = source_data.get("metadata", {}).get("tags", [])
            
            # Skip if already has AI tags and not forcing update
            if existing_tags and not force_update:
                ai_tag_indicators = ["ai-generated", "auto-tagged", "machine-learning", "artificial-intelligence"]
                has_ai_tags = any(indicator in str(existing_tags).lower() for indicator in ai_tag_indicators)
                if has_ai_tags:
                    return {"success": True, "message": "Source already has AI tags", "skipped": True}
            
            # Get content for analysis
            content = source_data.get("content", "")
            if not content or len(content.strip()) < 100:
                return {"success": False, "error": "Insufficient content for AI analysis"}
            
            # Generate AI tags
            ai_tag_service = get_ai_tag_service()
            ai_tags = await ai_tag_service.generate_tags_for_source(
                source_id=source_id,
                content=content,
                knowledge_type=source_data.get("metadata", {}).get("knowledge_type", "technical"),
                source_url=source_data.get("source_url"),
                existing_tags=existing_tags,
            )
            
            if not ai_tags:
                return {"success": False, "error": "No AI tags generated"}
            
            # Combine with existing tags
            all_tags = list(set(existing_tags + ai_tags))
            
            # Update the source record
            update_data = {
                "metadata": {
                    **source_data.get("metadata", {}),
                    "tags": all_tags,
                    "ai_tags_generated": True,
                    "ai_tags_count": len(ai_tags),
                }
            }
            
            update_response = self.supabase_client.table("archon_sources").update(update_data).eq("source_id", source_id).execute()
            
            if update_response.data:
                safe_logfire_info(f"Updated source {source_id} with {len(ai_tags)} AI tags")
                return {
                    "success": True,
                    "source_id": source_id,
                    "ai_tags_generated": len(ai_tags),
                    "total_tags": len(all_tags),
                    "ai_tags": ai_tags,
                }
            else:
                return {"success": False, "error": "Failed to update source record"}
                
        except Exception as e:
            self.logger.error(f"Failed to update source {source_id} with AI tags: {e}", exc_info=True)
            safe_logfire_error(f"AI tag update failed for {source_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    async def update_document_chunks_with_ai_tags(
        self, 
        source_id: str, 
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Update individual document chunks with AI-generated tags.
        
        Args:
            source_id: The source ID to update chunks for
            batch_size: Number of chunks to process at once
            
        Returns:
            Dict with update results
        """
        try:
            # Get document chunks for this source
            chunks_response = self.supabase_client.table("archon_documents").select("*").eq("source_id", source_id).execute()
            
            if not chunks_response.data:
                return {"success": False, "error": f"No document chunks found for source {source_id}"}
            
            chunks = chunks_response.data
            updated_count = 0
            ai_tag_service = get_ai_tag_service()
            
            # Process chunks in batches
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                
                for chunk in batch:
                    try:
                        content = chunk.get("content", "")
                        if len(content) < 200:  # Skip very short chunks
                            continue
                        
                        existing_tags = chunk.get("metadata", {}).get("tags", [])
                        
                        # Generate AI tags for this chunk
                        ai_tags = await ai_tag_service.generate_tags_for_content(
                            content=content,
                            knowledge_type=chunk.get("metadata", {}).get("knowledge_type", "technical"),
                            source_url=chunk.get("url"),
                            existing_tags=existing_tags,
                            max_tags=3,  # Fewer tags per chunk
                        )
                        
                        if ai_tags:
                            # Combine with existing tags
                            all_tags = list(set(existing_tags + ai_tags))
                            
                            # Update the chunk
                            update_data = {
                                "metadata": {
                                    **chunk.get("metadata", {}),
                                    "tags": all_tags,
                                    "ai_tags_generated": True,
                                }
                            }
                            
                            self.supabase_client.table("archon_documents").update(update_data).eq("id", chunk["id"]).execute()
                            updated_count += 1
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to update chunk {chunk.get('id')}: {e}")
                        continue
                
                # Small delay between batches to avoid overwhelming the system
                await asyncio.sleep(0.1)
            
            safe_logfire_info(f"Updated {updated_count} document chunks for source {source_id}")
            return {
                "success": True,
                "source_id": source_id,
                "chunks_updated": updated_count,
                "total_chunks": len(chunks),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update document chunks for {source_id}: {e}", exc_info=True)
            safe_logfire_error(f"Document chunk AI tag update failed for {source_id}: {str(e)}")
            return {"success": False, "error": str(e)}

    async def update_all_sources_with_ai_tags(
        self, 
        limit: Optional[int] = None,
        knowledge_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Update all sources with AI-generated tags.
        
        Args:
            limit: Maximum number of sources to process
            knowledge_types: Only process sources with these knowledge types
            
        Returns:
            Dict with update results
        """
        try:
            # Build query
            query = self.supabase_client.table("archon_sources").select("source_id, metadata")
            
            if knowledge_types:
                query = query.in_("metadata->knowledge_type", knowledge_types)
            
            if limit:
                query = query.limit(limit)
            
            sources_response = query.execute()
            
            if not sources_response.data:
                return {"success": True, "message": "No sources found to update", "updated": 0}
            
            sources = sources_response.data
            updated_count = 0
            failed_count = 0
            results = []
            
            for source in sources:
                source_id = source["source_id"]
                result = await self.update_source_tags_with_ai(source_id)
                
                if result["success"]:
                    updated_count += 1
                else:
                    failed_count += 1
                
                results.append({
                    "source_id": source_id,
                    "success": result["success"],
                    "error": result.get("error"),
                })
                
                # Small delay between sources
                await asyncio.sleep(0.5)
            
            safe_logfire_info(f"AI tag update completed: {updated_count} successful, {failed_count} failed")
            return {
                "success": True,
                "updated_count": updated_count,
                "failed_count": failed_count,
                "total_processed": len(sources),
                "results": results,
            }
            
        except Exception as e:
            self.logger.error(f"Failed to update all sources with AI tags: {e}", exc_info=True)
            safe_logfire_error(f"Bulk AI tag update failed: {str(e)}")
            return {"success": False, "error": str(e)}

    async def get_sources_without_ai_tags(self) -> List[Dict[str, Any]]:
        """
        Get sources that don't have AI-generated tags.
        
        Returns:
            List of sources without AI tags
        """
        try:
            # Get all sources
            sources_response = self.supabase_client.table("archon_sources").select("source_id, metadata").execute()
            
            if not sources_response.data:
                return []
            
            sources_without_ai = []
            
            for source in sources_response.data:
                metadata = source.get("metadata", {})
                tags = metadata.get("tags", [])
                has_ai_tags = metadata.get("ai_tags_generated", False)
                
                # Check if has AI-generated tags
                if not has_ai_tags and not any("ai" in str(tag).lower() for tag in tags):
                    sources_without_ai.append({
                        "source_id": source["source_id"],
                        "current_tags": tags,
                        "tag_count": len(tags),
                    })
            
            return sources_without_ai
            
        except Exception as e:
            self.logger.error(f"Failed to get sources without AI tags: {e}", exc_info=True)
            return []


# Global instance
_ai_tagging_background_service: Optional[AITaggingBackgroundService] = None


def get_ai_tagging_background_service() -> AITaggingBackgroundService:
    """Get the global AI tagging background service instance."""
    global _ai_tagging_background_service
    if _ai_tagging_background_service is None:
        _ai_tagging_background_service = AITaggingBackgroundService()
    return _ai_tagging_background_service
