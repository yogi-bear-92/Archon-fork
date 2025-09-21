"""
RAG Module for Archon MCP Server (HTTP-based version)

This module provides tools for:
- RAG query and search
- Source management
- Code example extraction and search

This version uses HTTP calls to the server service instead of importing
service modules directly, enabling true microservices architecture.
"""

import json
import logging
import os
from urllib.parse import urljoin

import httpx
from mcp.server.fastmcp import Context, FastMCP

# Import service discovery for HTTP communication
from src.server.config.service_discovery import get_api_url

logger = logging.getLogger(__name__)


def get_setting(key: str, default: str = "false") -> str:
    """Get a setting from environment variable."""
    return os.getenv(key, default)


def get_bool_setting(key: str, default: bool = False) -> bool:
    """Get a boolean setting from environment variable."""
    value = get_setting(key, "false" if not default else "true")
    return value.lower() in ("true", "1", "yes", "on")


def register_rag_tools(mcp: FastMCP):
    """Register all RAG tools with the MCP server."""

    @mcp.tool()
    async def rag_get_available_sources(ctx: Context) -> str:
        """
        Get list of available sources in the knowledge base.

        Returns:
            JSON string with structure:
            - success: bool - Operation success status
            - sources: list[dict] - Array of source objects
            - count: int - Number of sources
            - error: str - Error description if success=false
        """
        try:
            api_url = get_api_url()
            timeout = httpx.Timeout(30.0, connect=5.0)

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(urljoin(api_url, "/api/rag/sources"))

                if response.status_code == 200:
                    result = response.json()
                    sources = result.get("sources", [])

                    return json.dumps(
                        {"success": True, "sources": sources, "count": len(sources)}, indent=2
                    )
                else:
                    error_detail = response.text
                    return json.dumps(
                        {"success": False, "error": f"HTTP {response.status_code}: {error_detail}"},
                        indent=2,
                    )

        except Exception as e:
            logger.error(f"Error getting sources: {e}")
            return json.dumps({"success": False, "error": str(e)}, indent=2)

    @mcp.tool()
    async def rag_search_knowledge_base(
        ctx: Context, query: str, source_id: str | None = None, match_count: int = 5
    ) -> str:
        """
        Search knowledge base for relevant content using RAG.

        Args:
            query: Search query - Keep it SHORT and FOCUSED (2-5 keywords).
                   Good: "vector search", "authentication JWT", "React hooks"
                   Bad: "how to implement user authentication with JWT tokens in React with TypeScript and handle refresh tokens"
            source_id: Optional source ID filter from rag_get_available_sources().
                      This is the 'id' field from available sources, NOT a URL or domain name.
                      Example: "src_1234abcd" not "docs.anthropic.com"
            match_count: Max results (default: 5)

        Returns:
            JSON string with structure:
            - success: bool - Operation success status
            - results: list[dict] - Array of matching documents with content and metadata
            - reranked: bool - Whether results were reranked
            - error: str|null - Error description if success=false
        """
        try:
            api_url = get_api_url()
            timeout = httpx.Timeout(30.0, connect=5.0)

            async with httpx.AsyncClient(timeout=timeout) as client:
                request_data = {"query": query, "match_count": match_count}
                if source_id:
                    request_data["source"] = source_id

                response = await client.post(urljoin(api_url, "/api/rag/query"), json=request_data)

                if response.status_code == 200:
                    result = response.json()
                    return json.dumps(
                        {
                            "success": True,
                            "results": result.get("results", []),
                            "reranked": result.get("reranked", False),
                            "error": None,
                        },
                        indent=2,
                    )
                else:
                    error_detail = response.text
                    return json.dumps(
                        {
                            "success": False,
                            "results": [],
                            "error": f"HTTP {response.status_code}: {error_detail}",
                        },
                        indent=2,
                    )

        except Exception as e:
            logger.error(f"Error performing RAG query: {e}")
            return json.dumps({"success": False, "results": [], "error": str(e)}, indent=2)

    @mcp.tool()
    async def rag_search_code_examples(
        ctx: Context, query: str, source_id: str | None = None, match_count: int = 5
    ) -> str:
        """
        Search for relevant code examples in the knowledge base.

        Args:
            query: Search query - Keep it SHORT and FOCUSED (2-5 keywords).
                   Good: "React useState", "FastAPI middleware", "vector pgvector"
                   Bad: "React hooks useState useEffect useContext useReducer useMemo useCallback"
            source_id: Optional source ID filter from rag_get_available_sources().
                      This is the 'id' field from available sources, NOT a URL or domain name.
                      Example: "src_1234abcd" not "docs.anthropic.com"
            match_count: Max results (default: 5)

        Returns:
            JSON string with structure:
            - success: bool - Operation success status
            - results: list[dict] - Array of code examples with content and summaries
            - reranked: bool - Whether results were reranked
            - error: str|null - Error description if success=false
        """
        try:
            api_url = get_api_url()
            timeout = httpx.Timeout(30.0, connect=5.0)

            async with httpx.AsyncClient(timeout=timeout) as client:
                request_data = {"query": query, "match_count": match_count}
                if source_id:
                    request_data["source"] = source_id

                # Call the dedicated code examples endpoint
                response = await client.post(
                    urljoin(api_url, "/api/rag/code-examples"), json=request_data
                )

                if response.status_code == 200:
                    result = response.json()
                    return json.dumps(
                        {
                            "success": True,
                            "results": result.get("results", []),
                            "reranked": result.get("reranked", False),
                            "error": None,
                        },
                        indent=2,
                    )
                else:
                    error_detail = response.text
                    return json.dumps(
                        {
                            "success": False,
                            "results": [],
                            "error": f"HTTP {response.status_code}: {error_detail}",
                        },
                        indent=2,
                    )

        except Exception as e:
            logger.error(f"Error searching code examples: {e}")
            return json.dumps({"success": False, "results": [], "error": str(e)}, indent=2)

    # Log successful registration
    logger.info("✓ RAG tools registered (HTTP-based version)")
