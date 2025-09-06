"""
FastAPI Middleware for URL Detection and Auto-Addition

This middleware intercepts outbound HTTP requests made by the application
and analyzes them for potential addition to the knowledge base.
"""

import asyncio
import json
import logging
import time
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..config.logfire_config import get_logger, safe_span, safe_set_attribute
from ..services.url_detection_service import get_url_detection_service

logger = get_logger(__name__)


class URLDetectionMiddleware(BaseHTTPMiddleware):
    """
    Middleware that detects external URLs in HTTP requests and responses,
    analyzing them for potential auto-addition to the knowledge base.
    """
    
    def __init__(self, app, enabled: bool = True):
        super().__init__(app)
        self.enabled = enabled
        self.url_detection_service = None
        self._request_id_counter = 0
        
        # URLs to ignore (internal endpoints)
        self.ignore_patterns = {
            "/health", "/api/health", "/docs", "/redoc", "/openapi.json",
            "/favicon.ico", "/static/", "/_next/", "/assets/"
        }
        
        # Content types to scan for URLs
        self.scannable_content_types = {
            "application/json", "text/plain", "text/html", "text/markdown",
            "application/x-www-form-urlencoded"
        }
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and response for URL detection."""
        
        if not self.enabled:
            return await call_next(request)
        
        # Initialize service if needed
        if self.url_detection_service is None:
            try:
                self.url_detection_service = get_url_detection_service()
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize URL detection service: {e}")
                return await call_next(request)
        
        # Generate request ID for tracking
        self._request_id_counter += 1
        request_id = f"req-{self._request_id_counter}"
        
        with safe_span("url_detection_middleware") as span:
            safe_set_attribute(span, "request_id", request_id)
            safe_set_attribute(span, "method", request.method)
            safe_set_attribute(span, "path", str(request.url.path))
            
            start_time = time.time()
            
            # Skip ignored endpoints
            if self._should_ignore_path(request.url.path):
                return await call_next(request)
            
            # Process request
            request_urls = await self._extract_urls_from_request(request, request_id)
            
            # Execute the request
            response = await call_next(request)
            
            # Process response
            response_urls = await self._extract_urls_from_response(response, request_id)
            
            # Combine and process all detected URLs
            all_urls = list(set(request_urls + response_urls))
            
            if all_urls:
                processing_time = (time.time() - start_time) * 1000
                safe_set_attribute(span, "urls_detected", len(all_urls))
                safe_set_attribute(span, "processing_time_ms", processing_time)
                
                # Process URLs in background to avoid blocking response
                asyncio.create_task(self._process_detected_urls(
                    all_urls, 
                    request_id, 
                    f"HTTP {request.method} {request.url.path}"
                ))
            
            return response
    
    async def _extract_urls_from_request(self, request: Request, request_id: str) -> List[str]:
        """Extract URLs from request body and headers."""
        urls = []
        
        try:
            # Check for URLs in request body (for POST/PUT requests)
            if request.method in ["POST", "PUT", "PATCH"]:
                # Read body (this consumes the stream, so we need to be careful)
                body = await self._safely_read_request_body(request)
                
                if body:
                    urls.extend(await self._extract_urls_from_text(
                        body, f"request_body:{request_id}"
                    ))
            
            # Check for URLs in query parameters
            if request.url.query:
                urls.extend(await self._extract_urls_from_text(
                    request.url.query, f"query_params:{request_id}"
                ))
            
            # Check for URLs in specific headers
            user_agent = request.headers.get("user-agent", "")
            if user_agent:
                urls.extend(await self._extract_urls_from_text(
                    user_agent, f"user_agent:{request_id}"
                ))
                
        except Exception as e:
            logger.warning(f"⚠️ Error extracting URLs from request {request_id}: {e}")
        
        return urls
    
    async def _extract_urls_from_response(self, response: Response, request_id: str) -> List[str]:
        """Extract URLs from response body and headers."""
        urls = []
        
        try:
            # Only scan certain content types
            content_type = response.headers.get("content-type", "").lower()
            if not any(ct in content_type for ct in self.scannable_content_types):
                return urls
            
            # Read response body
            if hasattr(response, 'body') and response.body:
                try:
                    # Decode response body
                    if isinstance(response.body, bytes):
                        body_text = response.body.decode('utf-8', errors='ignore')
                    else:
                        body_text = str(response.body)
                    
                    # Extract URLs
                    urls.extend(await self._extract_urls_from_text(
                        body_text, f"response_body:{request_id}"
                    ))
                    
                except Exception as e:
                    logger.debug(f"Could not decode response body for {request_id}: {e}")
            
            # Check response headers for URLs
            location = response.headers.get("location")
            if location:
                urls.extend(await self._extract_urls_from_text(
                    location, f"location_header:{request_id}"
                ))
                
        except Exception as e:
            logger.warning(f"⚠️ Error extracting URLs from response {request_id}: {e}")
        
        return urls
    
    async def _extract_urls_from_text(self, text: str, context: str) -> List[str]:
        """Extract URLs from text using the URL detection service."""
        try:
            if self.url_detection_service:
                return await self.url_detection_service.detect_urls_in_text(
                    text, context, auto_process=False
                )
        except Exception as e:
            logger.warning(f"⚠️ Error extracting URLs from text: {e}")
        
        return []
    
    async def _process_detected_urls(self, urls: List[str], request_id: str, context: str):
        """Process detected URLs in background."""
        try:
            if not urls or not self.url_detection_service:
                return
            
            logger.info(f"🔍 Processing {len(urls)} URLs detected in {context}")
            
            for url in urls:
                # Process each URL with the detection service
                await self.url_detection_service.analyze_url(url, f"http_middleware:{context}")
                
        except Exception as e:
            logger.error(f"❌ Error processing detected URLs: {e}")
    
    async def _safely_read_request_body(self, request: Request) -> Optional[str]:
        """Safely read request body without consuming the stream."""
        try:
            # Check if body has already been read
            if hasattr(request, '_body'):
                body = request._body
            else:
                # Read the body
                body = await request.body()
                request._body = body  # Cache for later use
            
            if body:
                # Try to decode as text
                if isinstance(body, bytes):
                    # Try UTF-8 first, fall back to latin-1
                    try:
                        return body.decode('utf-8')
                    except UnicodeDecodeError:
                        try:
                            return body.decode('latin-1')
                        except UnicodeDecodeError:
                            return body.decode('utf-8', errors='ignore')
                else:
                    return str(body)
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not read request body: {e}")
            return None
    
    def _should_ignore_path(self, path: str) -> bool:
        """Check if path should be ignored for URL detection."""
        return any(pattern in path for pattern in self.ignore_patterns)


class URLResponseTracker:
    """
    Helper class to track URLs in outbound HTTP responses made by the application.
    """
    
    def __init__(self):
        self.url_detection_service = get_url_detection_service()
        self._tracked_responses = {}
    
    async def track_outbound_request(
        self, 
        url: str, 
        method: str = "GET",
        context: str = "",
        response_data: Optional[Dict] = None
    ):
        """Track an outbound HTTP request made by the application."""
        try:
            with safe_span("track_outbound_request") as span:
                safe_set_attribute(span, "url", url)
                safe_set_attribute(span, "method", method)
                safe_set_attribute(span, "context", context)
                
                # Analyze the target URL
                await self.url_detection_service.analyze_url(
                    url, f"outbound_request:{context}"
                )
                
                # If response data contains URLs, analyze those too
                if response_data:
                    response_text = json.dumps(response_data) if isinstance(response_data, dict) else str(response_data)
                    await self.url_detection_service.detect_urls_in_text(
                        response_text, f"outbound_response:{context}"
                    )
                
        except Exception as e:
            logger.warning(f"⚠️ Error tracking outbound request to {url}: {e}")


# Global instance for tracking outbound requests
_url_response_tracker: Optional[URLResponseTracker] = None


def get_url_response_tracker() -> URLResponseTracker:
    """Get global URL response tracker instance."""
    global _url_response_tracker
    
    if _url_response_tracker is None:
        _url_response_tracker = URLResponseTracker()
    
    return _url_response_tracker


def create_url_detection_middleware(enabled: bool = True) -> URLDetectionMiddleware:
    """Factory function to create URL detection middleware."""
    return lambda app: URLDetectionMiddleware(app, enabled=enabled)