#!/usr/bin/env python3
"""
Stdio-based MCP server for Claude Code integration.
This wrapper runs the Archon MCP server in stdio mode instead of HTTP mode.
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Set environment to use local services (not Docker)
os.environ["ARCHON_BASE_URL"] = os.getenv("ARCHON_BASE_URL", "http://localhost:8181")
os.environ["ARCHON_SERVER_PORT"] = os.getenv("ARCHON_SERVER_PORT", "8181")
os.environ["ARCHON_MCP_PORT"] = os.getenv("ARCHON_MCP_PORT", "8051")

# Import and run the MCP server with stdio transport
from src.mcp_server.mcp_server import mcp, setup_logfire, logger, mcp_logger

def main():
    """Run MCP server in stdio mode for Claude Code."""
    try:
        # Initialize Logfire
        setup_logfire(service_name="archon-mcp-stdio")

        logger.info("🚀 Starting Archon MCP Server (stdio mode)")
        logger.info("   Mode: stdio")
        logger.info("   For: Claude Code integration")

        mcp_logger.info("🔥 Logfire initialized for stdio MCP server")
        mcp_logger.info("🌟 Starting MCP server in stdio mode")

        # Run in stdio mode
        mcp.run(transport="stdio")

    except Exception as e:
        mcp_logger.error(f"💥 Fatal error: {str(e)}")
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("👋 MCP server stopped by user")
    except Exception as e:
        logger.error(f"💥 Unhandled exception: {e}")
        sys.exit(1)