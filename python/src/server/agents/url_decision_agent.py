"""
AI-Powered URL Decision Agent

This agent uses AI to intelligently analyze URLs and determine whether they
should be automatically added to the knowledge base or require user review.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from ..config.logfire_config import get_logger, safe_span, safe_set_attribute
from ..services.llm_provider_service import get_llm_client
from ..services.url_detection_service import URLAnalysis

logger = get_logger(__name__)


class URLDecisionContext(BaseModel):
    """Context information for URL decision making."""
    url: str
    domain: str
    source_context: str = ""
    detected_in: str = ""  # "agent_response", "task_description", "mcp_call", etc.
    project_context: str = ""
    user_preferences: Dict[str, Any] = Field(default_factory=dict)


class URLDecisionResult(BaseModel):
    """Result of AI-powered URL decision analysis."""
    url: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    relevance_score: float = Field(ge=0.0, le=1.0)
    quality_score: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=1.0)
    recommended_action: str  # "auto_add", "suggest", "ignore", "block"
    reasoning: str
    key_factors: List[str] = Field(default_factory=list)
    suggested_tags: List[str] = Field(default_factory=list)
    estimated_value: str = ""  # "high", "medium", "low"
    content_type_prediction: str = ""  # "documentation", "code", "tutorial", "reference", etc.


class URLDecisionAgent:
    """
    AI agent that makes intelligent decisions about whether URLs should be
    added to the knowledge base.
    """
    
    def __init__(self):
        self._decision_cache: Dict[str, URLDecisionResult] = {}
        
        # Prompts for different types of analysis
        self.analysis_prompt = """
You are an intelligent URL analysis agent for a knowledge management system. Your job is to analyze URLs and determine if they should be automatically added to a knowledge base, suggested for user review, or ignored.

URL to analyze: {url}
Domain: {domain}
Source context: {source_context}
Detected in: {detected_in}
Project context: {project_context}

Please analyze this URL based on the following criteria:

1. **RELEVANCE** (0.0-1.0): How relevant is this URL to software development, documentation, learning, or project work?
   - High (0.8-1.0): Core documentation, official guides, essential references
   - Medium (0.4-0.7): Useful resources, tutorials, community content
   - Low (0.0-0.3): Unrelated content, personal blogs, marketing material

2. **QUALITY** (0.0-1.0): What is the expected quality and trustworthiness of the content?
   - High (0.8-1.0): Official documentation, reputable sources, well-maintained
   - Medium (0.4-0.7): Community resources, established sites, good reputation
   - Low (0.0-0.3): Unknown sources, poor reputation, outdated content

3. **RISK** (0.0-1.0): What are the risks of adding this URL?
   - Low risk (0.0-0.3): Trusted domains, official sources, stable content
   - Medium risk (0.4-0.7): Community content, dynamic pages, moderate trust
   - High risk (0.8-1.0): Unknown sources, potentially harmful, unreliable

4. **CONFIDENCE** (0.0-1.0): How confident are you in your assessment?
   - High (0.8-1.0): Clear indicators, well-known domain, obvious value/lack thereof
   - Medium (0.4-0.7): Some indicators, moderate certainty
   - Low (0.0-0.3): Uncertain domain, unclear value, need more information

Based on your analysis, recommend one of these actions:
- **auto_add**: High confidence, high relevance, low risk - add automatically
- **suggest**: Moderate confidence/relevance - suggest to user for review
- **ignore**: Low relevance or value - ignore silently
- **block**: High risk or inappropriate content - actively block

Also provide:
- Key factors that influenced your decision
- Suggested tags for categorizing the content
- Estimated value (high/medium/low)
- Predicted content type (documentation, code, tutorial, reference, etc.)
- Clear reasoning for your recommendation

Respond in JSON format:
{{
    "confidence_score": 0.0,
    "relevance_score": 0.0,
    "quality_score": 0.0,
    "risk_score": 0.0,
    "recommended_action": "action",
    "reasoning": "detailed explanation",
    "key_factors": ["factor1", "factor2"],
    "suggested_tags": ["tag1", "tag2"],
    "estimated_value": "high/medium/low",
    "content_type_prediction": "type"
}}
"""
    
    async def analyze_url_decision(
        self, 
        context: URLDecisionContext
    ) -> URLDecisionResult:
        """
        Perform AI-powered analysis of a URL to determine the best action.
        
        Args:
            context: URLDecisionContext with URL and surrounding information
            
        Returns:
            URLDecisionResult with detailed analysis and recommendation
        """
        with safe_span("ai_url_decision") as span:
            safe_set_attribute(span, "url", context.url)
            safe_set_attribute(span, "domain", context.domain)
            safe_set_attribute(span, "source_context", context.source_context)
            
            # Check cache first
            cache_key = self._get_cache_key(context)
            if cache_key in self._decision_cache:
                logger.debug(f"📋 Using cached decision for {context.url}")
                return self._decision_cache[cache_key]
            
            try:
                # Prepare the prompt
                prompt = self.analysis_prompt.format(
                    url=context.url,
                    domain=context.domain,
                    source_context=context.source_context or "Unknown",
                    detected_in=context.detected_in or "Unknown",
                    project_context=context.project_context or "General"
                )
                
                # Get AI analysis using LLM provider
                async with get_llm_client() as client:
                    response = await client.chat.completions.create(
                        model="gpt-4o-mini",  # Fast model for URL analysis
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=800,
                        temperature=0.1,  # Low temperature for consistent analysis
                        response_format={"type": "json_object"}
                    )
                    response_text = response.choices[0].message.content
                
                # Parse the response
                import json
                try:
                    analysis_data = json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.warning(f"⚠️ Failed to parse AI response as JSON: {e}")
                    # Fallback to heuristic analysis
                    return await self._fallback_heuristic_analysis(context)
                
                # Create result object
                result = URLDecisionResult(
                    url=context.url,
                    confidence_score=analysis_data.get("confidence_score", 0.5),
                    relevance_score=analysis_data.get("relevance_score", 0.5),
                    quality_score=analysis_data.get("quality_score", 0.5),
                    risk_score=analysis_data.get("risk_score", 0.5),
                    recommended_action=analysis_data.get("recommended_action", "suggest"),
                    reasoning=analysis_data.get("reasoning", "AI analysis completed"),
                    key_factors=analysis_data.get("key_factors", []),
                    suggested_tags=analysis_data.get("suggested_tags", []),
                    estimated_value=analysis_data.get("estimated_value", "medium"),
                    content_type_prediction=analysis_data.get("content_type_prediction", "unknown")
                )
                
                # Validate and adjust the result
                result = self._validate_and_adjust_result(result, context)
                
                # Cache the result
                self._decision_cache[cache_key] = result
                
                safe_set_attribute(span, "recommended_action", result.recommended_action)
                safe_set_attribute(span, "confidence_score", result.confidence_score)
                safe_set_attribute(span, "estimated_value", result.estimated_value)
                
                logger.info(f"🤖 AI analysis complete for {context.url}: {result.recommended_action} "
                          f"(confidence: {result.confidence_score:.2f})")
                
                return result
                
            except Exception as e:
                logger.error(f"❌ Error in AI URL analysis for {context.url}: {e}")
                
                # Fallback to heuristic analysis
                return await self._fallback_heuristic_analysis(context)
    
    async def batch_analyze_urls(
        self, 
        contexts: List[URLDecisionContext],
        max_concurrent: int = 5
    ) -> List[URLDecisionResult]:
        """
        Analyze multiple URLs concurrently with rate limiting.
        
        Args:
            contexts: List of URLDecisionContext objects
            max_concurrent: Maximum concurrent AI requests
            
        Returns:
            List of URLDecisionResult objects
        """
        with safe_span("batch_url_analysis") as span:
            safe_set_attribute(span, "url_count", len(contexts))
            safe_set_attribute(span, "max_concurrent", max_concurrent)
            
            # Use semaphore to limit concurrent AI requests
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def analyze_with_semaphore(context):
                async with semaphore:
                    return await self.analyze_url_decision(context)
            
            # Execute batch analysis
            tasks = [analyze_with_semaphore(ctx) for ctx in contexts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle any exceptions
            valid_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"❌ Error analyzing URL {contexts[i].url}: {result}")
                    # Create fallback result
                    valid_results.append(await self._fallback_heuristic_analysis(contexts[i]))
                else:
                    valid_results.append(result)
            
            safe_set_attribute(span, "successful_analyses", len(valid_results))
            
            return valid_results
    
    async def learn_from_user_feedback(
        self, 
        url: str, 
        ai_recommendation: str, 
        user_action: str,
        feedback_reason: str = ""
    ):
        """
        Learn from user feedback to improve future recommendations.
        
        Args:
            url: The URL that was analyzed
            ai_recommendation: What the AI recommended
            user_action: What the user actually did
            feedback_reason: Optional reason from user
        """
        with safe_span("learn_user_feedback") as span:
            safe_set_attribute(span, "url", url)
            safe_set_attribute(span, "ai_recommendation", ai_recommendation)
            safe_set_attribute(span, "user_action", user_action)
            
            # Log the feedback for future model improvements
            logger.info(f"📚 Learning from feedback - URL: {url}, "
                       f"AI: {ai_recommendation}, User: {user_action}")
            
            # TODO: Implement machine learning feedback loop
            # For now, just log the feedback
            
            # Could implement:
            # 1. Store feedback in database
            # 2. Periodically retrain models
            # 3. Adjust heuristic weights
            # 4. Update domain reputation scores
            
            safe_set_attribute(span, "feedback_logged", True)
    
    # Private methods
    
    def _get_cache_key(self, context: URLDecisionContext) -> str:
        """Generate cache key for URL decision."""
        import hashlib
        
        # Include URL and key context factors in cache key
        cache_data = f"{context.url}|{context.domain}|{context.source_context}"
        return hashlib.md5(cache_data.encode()).hexdigest()[:16]
    
    def _validate_and_adjust_result(
        self, 
        result: URLDecisionResult, 
        context: URLDecisionContext
    ) -> URLDecisionResult:
        """Validate AI result and apply safety checks."""
        
        # Ensure scores are within valid ranges
        result.confidence_score = max(0.0, min(1.0, result.confidence_score))
        result.relevance_score = max(0.0, min(1.0, result.relevance_score))
        result.quality_score = max(0.0, min(1.0, result.quality_score))
        result.risk_score = max(0.0, min(1.0, result.risk_score))
        
        # Ensure valid action
        if result.recommended_action not in ["auto_add", "suggest", "ignore", "block"]:
            result.recommended_action = "suggest"  # Safe default
        
        # Apply safety overrides
        parsed = urlparse(context.url)
        domain = parsed.netloc.lower()
        
        # Override for known safe domains
        if domain in ["docs.python.org", "github.com", "stackoverflow.com"]:
            if result.recommended_action == "block":
                result.recommended_action = "suggest"
                result.reasoning += " (Override: Known safe domain)"
        
        # Override for local/internal URLs
        if domain in ["localhost", "127.0.0.1", "0.0.0.0"] or domain.endswith(".local"):
            result.recommended_action = "ignore"
            result.reasoning += " (Override: Local/internal URL)"
        
        # Ensure high-risk URLs are not auto-added
        if result.risk_score > 0.7 and result.recommended_action == "auto_add":
            result.recommended_action = "suggest"
            result.reasoning += " (Safety: High risk score, requires review)"
        
        return result
    
    async def _fallback_heuristic_analysis(
        self, 
        context: URLDecisionContext
    ) -> URLDecisionResult:
        """Fallback heuristic analysis when AI is unavailable."""
        
        parsed = urlparse(context.url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        
        # Initialize scores
        relevance_score = 0.5
        quality_score = 0.5
        risk_score = 0.3
        confidence_score = 0.6
        
        key_factors = []
        suggested_tags = []
        
        # Domain-based scoring
        if domain in ["docs.python.org", "github.com", "stackoverflow.com", 
                     "developer.mozilla.org", "w3.org"]:
            quality_score += 0.3
            relevance_score += 0.3
            risk_score -= 0.2
            key_factors.append("Trusted domain")
            
        elif domain.endswith(".org") or domain.endswith(".edu"):
            quality_score += 0.2
            risk_score -= 0.1
            key_factors.append("Institutional domain")
            
        elif domain.endswith(".gov"):
            quality_score += 0.25
            risk_score -= 0.15
            key_factors.append("Government domain")
        
        # Path-based scoring
        if any(term in path for term in ["/docs/", "/documentation/", "/guide/", "/tutorial/"]):
            relevance_score += 0.2
            suggested_tags.append("documentation")
            key_factors.append("Documentation path")
            
        elif "/api/" in path:
            relevance_score += 0.15
            suggested_tags.append("api")
            key_factors.append("API reference")
        
        # Technology-specific tags
        if any(term in context.url.lower() for term in ["python", "/py/"]):
            suggested_tags.append("python")
        elif any(term in context.url.lower() for term in ["javascript", "/js/", "node"]):
            suggested_tags.append("javascript")
        
        # Normalize scores
        relevance_score = max(0.0, min(1.0, relevance_score))
        quality_score = max(0.0, min(1.0, quality_score))
        risk_score = max(0.0, min(1.0, risk_score))
        
        # Calculate overall recommendation
        if relevance_score > 0.8 and quality_score > 0.7 and risk_score < 0.3:
            action = "auto_add"
            value = "high"
        elif relevance_score > 0.5 and quality_score > 0.5:
            action = "suggest"
            value = "medium"
        else:
            action = "ignore"
            value = "low"
        
        return URLDecisionResult(
            url=context.url,
            confidence_score=confidence_score,
            relevance_score=relevance_score,
            quality_score=quality_score,
            risk_score=risk_score,
            recommended_action=action,
            reasoning=f"Heuristic analysis based on domain and path patterns. "
                     f"Key factors: {', '.join(key_factors) if key_factors else 'Basic pattern matching'}",
            key_factors=key_factors,
            suggested_tags=suggested_tags,
            estimated_value=value,
            content_type_prediction="documentation" if "docs" in path else "unknown"
        )


# Global instance
_url_decision_agent: Optional[URLDecisionAgent] = None


def get_url_decision_agent() -> URLDecisionAgent:
    """Get global URL decision agent instance."""
    global _url_decision_agent
    
    if _url_decision_agent is None:
        _url_decision_agent = URLDecisionAgent()
    
    return _url_decision_agent