"""
HTTP-based MCP Server for Docker containers

This creates an HTTP endpoint that serves MCP functionality for Docker environments
where stdio transport is not available.
"""

import json
import logging
import os
import sys
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load environment variables
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path, override=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger("archon-mcp-http")

# Server configuration
ARCHON_API_BASE = os.getenv("ARCHON_API_BASE", "http://archon-server:8181")
if "localhost" in ARCHON_API_BASE or "127.0.0.1" in ARCHON_API_BASE:
    ARCHON_API_BASE = "http://localhost:8181"

ARCHON_MCP_PORT = int(os.getenv("ARCHON_MCP_PORT", 8051))

app = FastAPI(title="Archon MCP HTTP Server", version="1.0.0")

async def make_api_call(endpoint: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Make an API call to the Archon server"""
    url = f"{ARCHON_API_BASE}/api{endpoint}"
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        if method == "POST":
            response = await client.post(url, json=data)
        elif method == "PUT":
            response = await client.put(url, json=data)
        elif method == "DELETE":
            response = await client.delete(url)
        else:  # GET
            response = await client.get(url)
        
        response.raise_for_status()
        return response.json()

@app.get("/")
async def root():
    return {"message": "Archon MCP HTTP Server", "status": "running"}

@app.get("/mcp")
async def mcp_info():
    return {
        "name": "archon-mcp",
        "version": "1.0.0",
        "description": "Archon MCP Server via HTTP",
        "transport": "http"
    }

@app.get("/health")
async def health():
    try:
        # Test connection to main Archon server
        health_result = await make_api_call("/health")
        return {
            "status": "healthy",
            "mcp_server": "running",
            "archon_api": health_result.get("status", "unknown"),
            "api_base": ARCHON_API_BASE
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "api_base": ARCHON_API_BASE
            }
        )

@app.post("/projects")
async def create_project(project_data: Dict[str, Any]):
    try:
        result = await make_api_call("/projects", method="POST", data=project_data)
        return result
    except Exception as e:
        logger.error(f"Create project failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects")
async def list_projects():
    try:
        result = await make_api_call("/projects")
        return result
    except Exception as e:
        logger.error(f"List projects failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/projects/{project_id}")
async def get_project(project_id: str):
    try:
        result = await make_api_call(f"/projects/{project_id}")
        return result
    except Exception as e:
        logger.error(f"Get project failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/projects/{project_id}/tasks")
async def create_task(project_id: str, task_data: Dict[str, Any]):
    try:
        result = await make_api_call(f"/projects/{project_id}/tasks", method="POST", data=task_data)
        return result
    except Exception as e:
        logger.error(f"Create task failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks")
async def list_tasks():
    try:
        result = await make_api_call("/tasks")
        return result
    except Exception as e:
        logger.error(f"List tasks failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag/query")
async def rag_query(query_data: Dict[str, Any]):
    try:
        result = await make_api_call("/rag/query", method="POST", data=query_data)
        return result
    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def main():
    """Main entry point for the HTTP MCP server"""
    logger.info("🚀 Starting Archon HTTP MCP Server")
    logger.info(f"📡 API Base: {ARCHON_API_BASE}")
    logger.info(f"🌐 HTTP Server Port: {ARCHON_MCP_PORT}")
    
    # Test API connectivity
    try:
        health = await make_api_call("/health")
        logger.info("✅ API connectivity confirmed")
    except Exception as e:
        logger.error(f"❌ API connectivity failed: {e}")
        
    # Run the HTTP server
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=ARCHON_MCP_PORT,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())