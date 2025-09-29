#!/usr/bin/env python3
"""
Simple stdio-to-HTTP adapter for Archon MCP Server

This script acts as a bridge between Claude Code's stdio transport
and Archon's HTTP-based MCP server running in Docker.
"""

import json
import sys
import httpx

MCP_BASE_URL = "http://localhost:8051"

def main():
    """Read from stdin, forward to HTTP server, write response to stdout"""
    try:
        # Read JSON-RPC request from stdin
        for line in sys.stdin:
            if not line.strip():
                continue

            try:
                request = json.loads(line)

                # Forward to HTTP MCP server
                with httpx.Client(timeout=30.0) as client:
                    response = client.post(
                        f"{MCP_BASE_URL}/mcp",
                        json=request,
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "application/json, text/event-stream"
                        }
                    )

                    # Write response to stdout
                    sys.stdout.write(response.text + "\n")
                    sys.stdout.flush()

            except json.JSONDecodeError as e:
                # Invalid JSON - send error response
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"}
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()

            except Exception as e:
                # Other errors
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32603, "message": f"Internal error: {e}"}
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        sys.stderr.write(f"Fatal error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()