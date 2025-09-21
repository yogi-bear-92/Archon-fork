"""
AI Tag Generation Service

Generates relevant tags for crawled content using AI analysis.
This service analyzes content and suggests appropriate tags for better discoverability.
"""

import asyncio
from typing import Any, Dict, List, Optional

from src.server.config.logfire_config import get_logger, safe_logfire_error, safe_logfire_info
from .llm_provider_service import get_llm_client

logger = get_logger(__name__)


class AITagGenerationService:
    """Service for generating AI-powered tags from content analysis."""

    def __init__(self):
        self.logger = logger

    async def generate_tags_for_content(
        self,
        content: str,
        knowledge_type: str = "technical",
        source_url: Optional[str] = None,
        existing_tags: Optional[List[str]] = None,
        max_tags: int = 10,
        provider: Optional[str] = None,
    ) -> List[str]:
        """
        Generate relevant tags for content using AI analysis.

        Args:
            content: The content to analyze
            knowledge_type: Type of knowledge (technical, documentation, etc.)
            source_url: Optional URL for context
            existing_tags: Existing tags to avoid duplicating
            max_tags: Maximum number of tags to generate
            provider: Optional LLM provider override

        Returns:
            List of generated tags
        """
        if not content or len(content.strip()) < 100:
            self.logger.warning("Content too short for AI tag generation")
            return []

        try:
            # Limit content length to avoid token limits
            truncated_content = content[:15000] if len(content) > 15000 else content
            
            # Prepare context information
            context_info = self._build_context_info(knowledge_type, source_url, existing_tags)
            
            # Generate tags using LLM
            tags = await self._generate_tags_with_llm(
                content=truncated_content,
                context_info=context_info,
                max_tags=max_tags,
                provider=provider
            )
            
            # Clean and validate tags
            cleaned_tags = self._clean_and_validate_tags(tags, existing_tags)
            
            safe_logfire_info(
                f"Generated {len(cleaned_tags)} AI tags | content_length={len(content)} | knowledge_type={knowledge_type}"
            )
            
            return cleaned_tags[:max_tags]

        except Exception as e:
            self.logger.error(f"AI tag generation failed: {e}", exc_info=True)
            safe_logfire_error(f"AI tag generation failed: {str(e)}")
            return []

    def _build_context_info(
        self, 
        knowledge_type: str, 
        source_url: Optional[str], 
        existing_tags: Optional[List[str]]
    ) -> str:
        """Build context information for tag generation."""
        context_parts = []
        
        if knowledge_type:
            context_parts.append(f"Knowledge Type: {knowledge_type}")
        
        if source_url:
            # Extract domain and technology hints from URL
            domain = source_url.split("//")[-1].split("/")[0] if "//" in source_url else source_url
            context_parts.append(f"Source Domain: {domain}")
            
            # Add technology hints based on URL patterns
            if "github.com" in source_url:
                context_parts.append("Source: GitHub repository")
            elif "docs." in source_url:
                context_parts.append("Source: Documentation site")
            elif "api." in source_url:
                context_parts.append("Source: API documentation")
        
        if existing_tags:
            context_parts.append(f"Existing Tags: {', '.join(existing_tags[:5])}")
        
        return " | ".join(context_parts)

    async def _generate_tags_with_llm(
        self,
        content: str,
        context_info: str,
        max_tags: int,
        provider: Optional[str]
    ) -> List[str]:
        """Generate tags using LLM analysis."""
        try:
            async with get_llm_client(provider=provider) as client:
                # Determine model based on provider
                model_choice = "gpt-4o-mini"  # Default to cost-effective model
                
                prompt = f"""Analyze the following content and generate relevant, specific tags for a knowledge management system.

{context_info}

Content to analyze:
{content}

Generate {max_tags} relevant tags that would help users discover this content. Focus on:
- Technology names and frameworks
- Programming languages
- Concepts and methodologies
- Use cases and applications
- Difficulty levels (beginner, intermediate, advanced)
- Content types (tutorial, reference, guide, example)

Requirements:
- Use lowercase with hyphens for multi-word tags (e.g., "machine-learning", "react-hooks")
- Be specific and technical
- Avoid generic terms like "documentation" or "guide"
- Include both broad categories and specific details
- Each tag should be 1-3 words maximum

Return only the tags, one per line, no numbering or bullets."""

                response = await client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at analyzing technical content and generating relevant tags for knowledge management systems."
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.3,  # Lower temperature for more consistent results
                    max_tokens=500,
                )

                # Parse the response
                response_text = response.choices[0].message.content.strip()
                tags = [tag.strip() for tag in response_text.split('\n') if tag.strip()]
                
                return tags

        except Exception as e:
            self.logger.error(f"LLM tag generation failed: {e}", exc_info=True)
            raise

    def _clean_and_validate_tags(
        self, 
        tags: List[str], 
        existing_tags: Optional[List[str]] = None
    ) -> List[str]:
        """Clean and validate generated tags."""
        if not tags:
            return []
        
        existing_set = set(existing_tags or [])
        cleaned_tags = []
        
        for tag in tags:
            # Clean the tag
            cleaned_tag = tag.strip().lower()
            
            # Remove common prefixes and suffixes
            cleaned_tag = cleaned_tag.replace('tag:', '').replace('tags:', '').replace('-', ' ')
            cleaned_tag = '-'.join(cleaned_tag.split())  # Convert spaces to hyphens
            
            # Validate tag
            if (cleaned_tag and 
                len(cleaned_tag) > 2 and 
                len(cleaned_tag) < 50 and
                cleaned_tag not in existing_set and
                cleaned_tag not in cleaned_tags):
                cleaned_tags.append(cleaned_tag)
        
        return cleaned_tags

    async def generate_tags_for_source(
        self,
        source_id: str,
        content: str,
        knowledge_type: str = "technical",
        source_url: Optional[str] = None,
        existing_tags: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Generate tags specifically for a source (higher-level analysis).
        
        Args:
            source_id: The source identifier
            content: Combined content from the source
            knowledge_type: Type of knowledge
            source_url: Original URL
            existing_tags: Existing tags to avoid duplicating
            
        Returns:
            List of generated tags
        """
        # For source-level tagging, we can be more strategic about tag generation
        # Focus on broader categories and technology identification
        
        try:
            # Use a more focused prompt for source-level tagging
            truncated_content = content[:20000] if len(content) > 20000 else content
            
            async with get_llm_client() as client:
                prompt = f"""Analyze this content and generate high-level tags for a knowledge source.

Source ID: {source_id}
Knowledge Type: {knowledge_type}
Source URL: {source_url or 'Not provided'}

Content:
{truncated_content}

Generate 8-12 strategic tags that categorize this source at a high level. Focus on:
- Primary technology/framework
- Domain/industry
- Content type (docs, api, tutorial, etc.)
- Complexity level
- Key concepts covered

Return tags in lowercase with hyphens, one per line."""

                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at categorizing technical content sources for knowledge management."
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=300,
                )

                response_text = response.choices[0].message.content.strip()
                tags = [tag.strip().lower() for tag in response_text.split('\n') if tag.strip()]
                
                # Clean and validate
                cleaned_tags = self._clean_and_validate_tags(tags, existing_tags)
                
                safe_logfire_info(
                    f"Generated {len(cleaned_tags)} source-level tags for {source_id}"
                )
                
                return cleaned_tags

        except Exception as e:
            self.logger.error(f"Source-level tag generation failed for {source_id}: {e}", exc_info=True)
            return []


# Global instance
_ai_tag_service: Optional[AITagGenerationService] = None


def get_ai_tag_service() -> AITagGenerationService:
    """Get the global AI tag generation service instance."""
    global _ai_tag_service
    if _ai_tag_service is None:
        _ai_tag_service = AITagGenerationService()
    return _ai_tag_service
