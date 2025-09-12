"""
Intelligent URL Detection and Knowledge Base Auto-Addition Service

This service monitors external URL calls and intelligently suggests adding them
to the knowledge base using AI-powered decision making.
"""

import asyncio
import hashlib
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from src.server.config.logfire_config import get_logger, safe_span, safe_set_attribute
from .client_manager import get_supabase_client
from .mcp_service_client import MCPServiceClient

logger = get_logger(__name__)


class URLAnalysis(BaseModel):
    """Analysis result for a detected URL."""
    url: str
    domain: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    content_quality_score: float = Field(ge=0.0, le=1.0)
    domain_reputation_score: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    reasoning: str
    recommended_action: str  # "auto_add", "suggest", "ignore"
    detected_at: datetime = Field(default_factory=datetime.now)
    source_context: str = ""
    tags: List[str] = Field(default_factory=list)


class URLDetectionConfig(BaseModel):
    """Configuration for URL detection system."""
    enabled: bool = True
    auto_add_threshold: float = 0.85
    suggest_threshold: float = 0.6
    ignore_threshold: float = 0.3
    max_concurrent_analyses: int = 10
    cache_ttl_hours: int = 24
    excluded_domains: Set[str] = {
        "localhost", "127.0.0.1", "0.0.0.0",
        "example.com", "test.com", "internal.local"
    }
    preferred_domains: Set[str] = {
        "github.com", "docs.python.org", "stackoverflow.com",
        "developer.mozilla.org", "w3.org", "ietf.org"
    }


class URLDetectionService:
    """
    Service for detecting external URLs and managing intelligent auto-addition
    to the knowledge base.
    """
    
    def __init__(self):
        self.mcp_client = MCPServiceClient()
        self.config = URLDetectionConfig()
        self._analysis_cache: Dict[str, URLAnalysis] = {}
        self._processing_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._worker_tasks: List[asyncio.Task] = []
        self._url_patterns = [
            re.compile(r'https?://[^\s\])\'"<>]+', re.IGNORECASE),
            re.compile(r'www\.[^\s\])\'"<>]+', re.IGNORECASE),
        ]
        
    async def initialize(self):
        """Initialize the URL detection service."""
        try:
            with safe_span("url_detection_init") as span:
                # Start background workers
                for i in range(3):  # 3 worker processes
                    task = asyncio.create_task(self._background_worker(f"worker-{i}"))
                    self._worker_tasks.append(task)
                
                # Load configuration from database
                await self._load_configuration()
                
                logger.info("✅ URL Detection Service initialized successfully")
                safe_set_attribute(span, "workers_started", len(self._worker_tasks))
                
        except Exception as e:
            logger.error(f"❌ Failed to initialize URL Detection Service: {e}")
            raise
    
    async def shutdown(self):
        """Gracefully shutdown the service."""
        logger.info("🛑 Shutting down URL Detection Service...")
        
        # Cancel worker tasks
        for task in self._worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        
        logger.info("✅ URL Detection Service shutdown complete")
    
    async def detect_urls_in_text(
        self, 
        text: str, 
        context: str = "",
        auto_process: bool = True
    ) -> List[str]:
        """
        Detect URLs in text and optionally process them automatically.
        
        Args:
            text: Text to scan for URLs
            context: Context about where the text came from
            auto_process: Whether to automatically queue for processing
            
        Returns:
            List of detected URLs
        """
        with safe_span("detect_urls") as span:
            safe_set_attribute(span, "text_length", len(text))
            safe_set_attribute(span, "context", context)
            
            urls = []
            
            # Extract URLs using regex patterns
            for pattern in self._url_patterns:
                matches = pattern.findall(text)
                urls.extend(matches)
            
            # Clean and validate URLs
            cleaned_urls = []
            for url in urls:
                cleaned_url = self._clean_url(url)
                if cleaned_url and self._is_external_url(cleaned_url):
                    cleaned_urls.append(cleaned_url)
            
            # Remove duplicates while preserving order
            unique_urls = list(dict.fromkeys(cleaned_urls))
            
            safe_set_attribute(span, "urls_detected", len(unique_urls))
            
            if unique_urls:
                logger.info(f"🔍 Detected {len(unique_urls)} URLs in text: {unique_urls[:3]}...")
                
                if auto_process:
                    # Queue for intelligent processing
                    for url in unique_urls:
                        await self._queue_for_analysis(url, context)
            
            return unique_urls
    
    async def analyze_url(self, url: str, context: str = "") -> URLAnalysis:
        """
        Perform intelligent analysis of a URL to determine if it should be
        added to the knowledge base.
        """
        with safe_span("analyze_url") as span:
            safe_set_attribute(span, "url", url)
            safe_set_attribute(span, "context", context)
            
            # Check cache first
            url_hash = self._get_url_hash(url)
            if url_hash in self._analysis_cache:
                cached = self._analysis_cache[url_hash]
                if (datetime.now() - cached.detected_at).hours < self.config.cache_ttl_hours:
                    logger.debug(f"📋 Using cached analysis for {url}")
                    return cached
            
            # Perform new analysis
            analysis = await self._perform_url_analysis(url, context)
            
            # Cache the result
            self._analysis_cache[url_hash] = analysis
            
            # Store in database for persistence
            await self._store_analysis_result(analysis)
            
            safe_set_attribute(span, "overall_score", analysis.overall_score)
            safe_set_attribute(span, "recommended_action", analysis.recommended_action)
            
            return analysis
    
    async def process_agent_response(
        self, 
        response_text: str, 
        agent_context: str = ""
    ) -> Dict[str, any]:
        """
        Process agent response for URL detection and analysis.
        
        Returns:
            Dictionary with detected URLs and processing results
        """
        with safe_span("process_agent_response") as span:
            detected_urls = await self.detect_urls_in_text(
                response_text, 
                f"agent_response:{agent_context}"
            )
            
            if not detected_urls:
                return {"urls_detected": 0, "actions_taken": []}
            
            results = {
                "urls_detected": len(detected_urls),
                "urls": detected_urls,
                "actions_taken": [],
                "suggestions": []
            }
            
            # Analyze each URL
            for url in detected_urls:
                try:
                    analysis = await self.analyze_url(url, agent_context)
                    
                    if analysis.recommended_action == "auto_add":
                        # Automatically add to knowledge base
                        await self._auto_add_to_knowledge_base(url, analysis)
                        results["actions_taken"].append({
                            "action": "auto_added",
                            "url": url,
                            "score": analysis.overall_score
                        })
                        
                    elif analysis.recommended_action == "suggest":
                        # Add to suggestions for user review
                        results["suggestions"].append({
                            "url": url,
                            "score": analysis.overall_score,
                            "reasoning": analysis.reasoning,
                            "tags": analysis.tags
                        })
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error analyzing URL {url}: {e}")
            
            safe_set_attribute(span, "urls_processed", len(detected_urls))
            safe_set_attribute(span, "auto_added", len(results["actions_taken"]))
            safe_set_attribute(span, "suggestions", len(results["suggestions"]))
            
            return results
    
    async def get_url_suggestions(self, limit: int = 10) -> List[Dict[str, any]]:
        """Get pending URL suggestions for user review."""
        try:
            client = get_supabase_client()
            
            result = client.table("url_suggestions").select(
                "*"
            ).eq("status", "pending").order(
                "overall_score", desc=True
            ).limit(limit).execute()
            
            return result.data if result.data else []
            
        except Exception as e:
            logger.error(f"❌ Error fetching URL suggestions: {e}")
            return []
    
    async def approve_url_suggestion(self, suggestion_id: str) -> bool:
        """Approve a URL suggestion and add it to knowledge base."""
        try:
            client = get_supabase_client()
            
            # Get suggestion details
            result = client.table("url_suggestions").select(
                "*"
            ).eq("id", suggestion_id).single().execute()
            
            if not result.data:
                return False
            
            suggestion = result.data
            
            # Add to knowledge base
            await self.mcp_client.crawl_url(
                url=suggestion["url"],
                options={
                    "source_type": "url_suggestion",
                    "tags": suggestion.get("tags", [])
                }
            )
            
            # Update suggestion status
            client.table("url_suggestions").update({
                "status": "approved",
                "approved_at": datetime.now().isoformat()
            }).eq("id", suggestion_id).execute()
            
            logger.info(f"✅ Approved URL suggestion: {suggestion['url']}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error approving URL suggestion: {e}")
            return False
    
    # Private methods
    
    def _clean_url(self, url: str) -> Optional[str]:
        """Clean and normalize a URL."""
        try:
            # Remove common trailing characters
            url = url.rstrip('.,;:!?)"\']}>')
            
            # Add protocol if missing
            if url.startswith('www.'):
                url = f"https://{url}"
            
            # Validate URL format
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return None
            
            return url
            
        except Exception:
            return None
    
    def _is_external_url(self, url: str) -> bool:
        """Check if URL is external and not excluded."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remove port if present
            if ':' in domain:
                domain = domain.split(':')[0]
            
            return domain not in self.config.excluded_domains
            
        except Exception:
            return False
    
    def _get_url_hash(self, url: str) -> str:
        """Generate hash for URL caching."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]
    
    async def _queue_for_analysis(self, url: str, context: str):
        """Queue URL for background analysis."""
        try:
            await self._processing_queue.put((url, context))
        except asyncio.QueueFull:
            logger.warning(f"⚠️ Processing queue full, dropping URL: {url}")
    
    async def _background_worker(self, worker_id: str):
        """Background worker for processing URL analysis queue."""
        logger.info(f"🔧 Starting background worker: {worker_id}")
        
        while True:
            try:
                # Get URL from queue with timeout
                url, context = await asyncio.wait_for(
                    self._processing_queue.get(), 
                    timeout=5.0
                )
                
                # Process the URL
                await self.analyze_url(url, context)
                
            except asyncio.TimeoutError:
                # No items in queue, continue
                continue
            except asyncio.CancelledError:
                logger.info(f"🛑 Background worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in background worker {worker_id}: {e}")
    
    async def _perform_url_analysis(self, url: str, context: str) -> URLAnalysis:
        """Perform detailed AI-powered analysis of a URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Check if already in knowledge base
            already_crawled = await self._check_if_already_crawled(url)
            
            if already_crawled:
                return URLAnalysis(
                    url=url,
                    domain=domain,
                    relevance_score=0.0,
                    content_quality_score=0.0,
                    domain_reputation_score=0.0,
                    overall_score=0.0,
                    reasoning="URL already exists in knowledge base",
                    recommended_action="ignore",
                    source_context=context
                )
            
            # Calculate domain reputation score
            domain_score = self._calculate_domain_reputation(domain)
            
            # For now, use heuristic-based analysis
            # TODO: Integrate with AI model for content analysis
            relevance_score = self._calculate_relevance_score(url, context)
            quality_score = self._calculate_quality_score(url, domain)
            
            # Calculate overall score (weighted average)
            overall_score = (
                relevance_score * 0.4 +
                quality_score * 0.4 +
                domain_score * 0.2
            )
            
            # Determine recommended action
            if overall_score >= self.config.auto_add_threshold:
                action = "auto_add"
                reasoning = f"High confidence score ({overall_score:.2f}) indicates valuable content"
            elif overall_score >= self.config.suggest_threshold:
                action = "suggest"
                reasoning = f"Moderate score ({overall_score:.2f}) suggests potential value, user review recommended"
            else:
                action = "ignore"
                reasoning = f"Low score ({overall_score:.2f}) indicates limited relevance"
            
            # Generate tags based on URL analysis
            tags = self._generate_tags(url, domain, context)
            
            return URLAnalysis(
                url=url,
                domain=domain,
                relevance_score=relevance_score,
                content_quality_score=quality_score,
                domain_reputation_score=domain_score,
                overall_score=overall_score,
                reasoning=reasoning,
                recommended_action=action,
                source_context=context,
                tags=tags
            )
            
        except Exception as e:
            logger.error(f"❌ Error analyzing URL {url}: {e}")
            
            # Return safe default analysis
            return URLAnalysis(
                url=url,
                domain=parsed.netloc if parsed else "unknown",
                relevance_score=0.0,
                content_quality_score=0.0,
                domain_reputation_score=0.0,
                overall_score=0.0,
                reasoning=f"Analysis failed: {str(e)}",
                recommended_action="ignore",
                source_context=context
            )
    
    def _calculate_domain_reputation(self, domain: str) -> float:
        """Calculate domain reputation score."""
        if domain in self.config.preferred_domains:
            return 0.9
        
        # Check domain patterns
        if any(pattern in domain for pattern in ['docs.', 'developer.', 'api.']):
            return 0.8
        
        if domain.endswith('.org') or domain.endswith('.edu'):
            return 0.7
        
        if domain.endswith('.gov'):
            return 0.8
        
        # Default for unknown domains
        return 0.5
    
    def _calculate_relevance_score(self, url: str, context: str) -> float:
        """Calculate relevance score based on URL and context."""
        score = 0.5  # Base score
        
        # Check for technical documentation patterns
        if any(term in url.lower() for term in ['docs', 'documentation', 'guide', 'tutorial', 'api']):
            score += 0.3
        
        # Check for code repository patterns
        if 'github.com' in url.lower() or 'gitlab.com' in url.lower():
            score += 0.2
        
        # Check context relevance
        if context and any(term in context.lower() for term in ['task', 'project', 'research']):
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_quality_score(self, url: str, domain: str) -> float:
        """Calculate content quality score."""
        score = 0.5  # Base score
        
        # Favor known high-quality domains
        quality_indicators = [
            'stackoverflow.com', 'github.com', 'docs.python.org',
            'developer.mozilla.org', 'w3.org', 'ietf.org'
        ]
        
        if any(indicator in domain for indicator in quality_indicators):
            score += 0.4
        
        # Check URL structure quality
        if len(url.split('/')) > 3:  # Has path structure
            score += 0.1
        
        return min(score, 1.0)
    
    def _generate_tags(self, url: str, domain: str, context: str) -> List[str]:
        """Generate relevant tags for the URL."""
        tags = []
        
        # Domain-based tags
        if 'github.com' in domain:
            tags.extend(['code', 'repository', 'source'])
        elif 'docs.' in domain or 'documentation' in url:
            tags.extend(['documentation', 'guide'])
        elif 'api.' in domain or '/api/' in url:
            tags.extend(['api', 'reference'])
        
        # Content-based tags
        if any(term in url.lower() for term in ['python', 'py']):
            tags.append('python')
        elif any(term in url.lower() for term in ['javascript', 'js', 'node']):
            tags.append('javascript')
        elif any(term in url.lower() for term in ['react', 'vue', 'angular']):
            tags.append('frontend')
        
        # Context-based tags
        if 'agent' in context.lower():
            tags.append('agent-related')
        elif 'project' in context.lower():
            tags.append('project-resource')
        
        return list(set(tags))  # Remove duplicates
    
    async def _check_if_already_crawled(self, url: str) -> bool:
        """Check if URL is already in the knowledge base."""
        try:
            client = get_supabase_client()
            
            result = client.table("archon_sources").select("id").eq(
                "source_url", url
            ).limit(1).execute()
            
            return len(result.data) > 0 if result.data else False
            
        except Exception as e:
            logger.warning(f"⚠️ Error checking if URL already crawled: {e}")
            return False
    
    async def _auto_add_to_knowledge_base(self, url: str, analysis: URLAnalysis):
        """Automatically add URL to knowledge base."""
        try:
            logger.info(f"🤖 Auto-adding URL to knowledge base: {url}")
            
            await self.mcp_client.crawl_url(
                url=url,
                options={
                    "source_type": "auto_detected",
                    "tags": analysis.tags
                }
            )
            
            # Log the auto-addition
            await self._log_auto_addition(url, analysis)
            
        except Exception as e:
            logger.error(f"❌ Error auto-adding URL {url}: {e}")
    
    async def _store_analysis_result(self, analysis: URLAnalysis):
        """Store analysis result in database."""
        try:
            client = get_supabase_client()
            
            # Only store suggestions (not ignored or auto-added items)
            if analysis.recommended_action == "suggest":
                client.table("url_suggestions").insert({
                    "url": analysis.url,
                    "domain": analysis.domain,
                    "relevance_score": analysis.relevance_score,
                    "content_quality_score": analysis.content_quality_score,
                    "domain_reputation_score": analysis.domain_reputation_score,
                    "overall_score": analysis.overall_score,
                    "reasoning": analysis.reasoning,
                    "recommended_action": analysis.recommended_action,
                    "source_context": analysis.source_context,
                    "tags": analysis.tags,
                    "status": "pending",
                    "detected_at": analysis.detected_at.isoformat()
                }).execute()
            
        except Exception as e:
            logger.warning(f"⚠️ Error storing analysis result: {e}")
    
    async def _log_auto_addition(self, url: str, analysis: URLAnalysis):
        """Log automatic addition to knowledge base."""
        try:
            client = get_supabase_client()
            
            client.table("url_auto_additions").insert({
                "url": url,
                "overall_score": analysis.overall_score,
                "reasoning": analysis.reasoning,
                "tags": analysis.tags,
                "source_context": analysis.source_context,
                "added_at": datetime.now().isoformat()
            }).execute()
            
        except Exception as e:
            logger.warning(f"⚠️ Error logging auto addition: {e}")
    
    async def _load_configuration(self):
        """Load configuration from database or environment."""
        # For now, use defaults
        # TODO: Implement database-backed configuration
        logger.debug("📋 Using default URL detection configuration")


# Global instance
_url_detection_service: Optional[URLDetectionService] = None


def get_url_detection_service() -> URLDetectionService:
    """Get global URL detection service instance."""
    global _url_detection_service
    
    if _url_detection_service is None:
        _url_detection_service = URLDetectionService()
    
    return _url_detection_service