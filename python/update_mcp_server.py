#!/usr/bin/env python3
"""
Script to update MCP server with CLI tools integration
"""

cli_tools_registration = '''
    # CLI Tools Integration (from removed MCP servers)
    try:
        from src.mcp_server.features.cli_tools_module import register_cli_tools
        
        register_cli_tools(mcp)
        modules_registered += 1
        logger.info("✓ CLI tools integration registered")
    except ImportError as e:
        logger.warning(f"⚠ CLI tools not available: {e}")
    except Exception as e:
        logger.error(f"✗ Failed to register CLI tools: {e}")
        logger.error(traceback.format_exc())

'''

print("CLI Tools Registration Code:")
print("=" * 50)
print(cli_tools_registration.strip())
print("=" * 50)
print("\nAdd this code after the Claude Flow Integration Tools section in the Docker container's MCP server.")