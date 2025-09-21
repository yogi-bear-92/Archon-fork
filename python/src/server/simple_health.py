#!/usr/bin/env python3
"""
Simple health check server for Docker development
Bypasses database connections and complex initialization
"""

import os
from fastapi import FastAPI

app = FastAPI(title="Archon Health Check Server")

@app.get("/health")
async def health_check():
    """Simple health check endpoint"""
    return {
        "status": "healthy",
        "service": "archon-server",
        "mode": "development",
        "port": os.getenv("ARCHON_SERVER_PORT", "8181")
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Archon Knowledge Engine API", 
        "status": "running",
        "mode": "development"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ARCHON_SERVER_PORT", "8181"))
    uvicorn.run(app, host="0.0.0.0", port=port)