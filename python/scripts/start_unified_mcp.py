#!/usr/bin/env python3
"""
Unified Archon MCP Server Startup Script

This script starts the unified MCP server that wraps all Archon services.
It can be used directly or through Claude Desktop MCP configuration.
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Set environment variables if not already set
if not os.getenv("ARCHON_BASE_URL"):
    os.environ["ARCHON_BASE_URL"] = "http://localhost:8080"

if not os.getenv("ARCHON_SERVER_PORT"):
    os.environ["ARCHON_SERVER_PORT"] = "8080"

# Import and run the unified MCP server
if __name__ == "__main__":
    from server.unified_archon_mcp import main
    import asyncio
    
    print("🚀 Starting Unified Archon MCP Server...")
    print("📍 Backend URL:", os.getenv("ARCHON_BASE_URL"))
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Archon MCP Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)