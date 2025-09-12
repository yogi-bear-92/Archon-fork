import asyncio
import json
import sys
from pathlib import Path
from server.services.cli_tool_discovery_service import cli_discovery_service
#!/usr/bin/env python3
"""
Validation script for CLI tools integration
Quick validation that all removed server functionality is available
"""

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def validate_removed_server_functionality():
    """Validate that removed server functionality is available through CLI tools"""
    print("🔍 Validating Removed Server Functionality Integration")
    print("=" * 65)

    # Expected functionality from removed servers
    expected_servers = {
        "claude-flow": {
            "commands": ["swarm", "agent", "sparc", "memory", "coordination", "analysis", "monitoring"],
            "description": "Enterprise AI agent orchestration platform"
        },
        "ruv-swarm": {
            "commands": ["init", "spawn", "coordinate"],
            "description": "Advanced swarm intelligence platform"
        },
        "flow-nexus": {
            "commands": ["deploy", "orchestrate"],
            "description": "AI-powered development platform"
        }
    }

    try:
        print("1. Starting CLI discovery service...")
        await cli_discovery_service.start()

        print("2. Analyzing discovered CLI tools...")
        cli_statuses = cli_discovery_service.get_tool_status()
        cli_commands = cli_discovery_service.get_discovered_commands()

        print(f"\n📊 Discovery Results:")
        print(f"   Total CLI tools configured: {len(cli_statuses)}")
        print(f"   Available CLI tools: {sum(1 for s in cli_statuses.values() if s.available)}")
        print(f"   Total commands discovered: {len(cli_commands)}")

        print(f"\n🔧 CLI Tool Status:")
        for name, status in cli_statuses.items():
            status_icon = "✅" if status.available else "❌"
            print(f"   {status_icon} {name}: {'Available' if status.available else 'Unavailable'}")
            if status.version and status.available:
                print(f"      Version: {status.version}")
            if status.error_message and not status.available:
                print(f"      Error: {status.error_message}")

        print(f"\n📋 Available Commands:")
        for cmd_name, cmd in cli_commands.items():
            print(f"   - {cmd_name}: {cmd.description}")

        # Validation against expected functionality
        print(f"\n🎯 Functionality Validation:")
        validation_results = {}

        for server, expected in expected_servers.items():
            available_commands = []

            # Find commands for this server
            for cmd_name, cmd in cli_commands.items():
                if cmd.tool == server:
                    # Extract command name (remove server prefix)
                    command_name = cmd_name.split("__")[-1]
                    available_commands.append(command_name)

            # Calculate coverage
            expected_commands = expected["commands"]
            covered_commands = [cmd for cmd in available_commands if cmd in expected_commands]
            coverage = len(covered_commands) / len(expected_commands) * 100 if expected_commands else 0

            validation_results[server] = {
                "expected": expected_commands,
                "available": available_commands,
                "covered": covered_commands,
                "coverage": coverage
            }

            # Display results
            status_icon = "✅" if coverage >= 75 else "⚠️" if coverage >= 50 else "❌"
            print(f"   {status_icon} {server}: {coverage:.0f}% coverage")
            print(f"      Expected: {expected_commands}")
            print(f"      Available: {available_commands}")
            if coverage < 100:
                missing = [cmd for cmd in expected_commands if cmd not in available_commands]
                print(f"      Missing: {missing}")

        # Overall assessment
        print(f"\n📈 Overall Assessment:")
        total_servers = len(expected_servers)
        successful_integrations = sum(1 for result in validation_results.values() if result["coverage"] >= 75)
        partial_integrations = sum(1 for result in validation_results.values() if 50 <= result["coverage"] < 75)
        failed_integrations = sum(1 for result in validation_results.values() if result["coverage"] < 50)

        print(f"   📊 Total removed servers: {total_servers}")
        print(f"   ✅ Successfully integrated: {successful_integrations}")
        print(f"   ⚠️ Partially integrated: {partial_integrations}")
        print(f"   ❌ Failed integration: {failed_integrations}")

        overall_success_rate = successful_integrations / total_servers * 100

        if overall_success_rate >= 75:
            print(f"\n🎉 SUCCESS: {overall_success_rate:.0f}% of removed servers successfully integrated!")
            print("   Archon can effectively replace the removed MCP servers.")
            return True
        elif overall_success_rate >= 50:
            print(f"\n⚠️ PARTIAL SUCCESS: {overall_success_rate:.0f}% integration achieved")
            print("   Most functionality is available, but some gaps remain.")
            return True
        else:
            print(f"\n❌ INTEGRATION INCOMPLETE: Only {overall_success_rate:.0f}% success rate")
            print("   Significant functionality is missing.")
            return False

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

    finally:
        await cli_discovery_service.stop()

async def test_specific_commands():
    """Test execution of specific commands to prove functionality"""
    print("\n🧪 Testing Specific Command Execution")
    print("=" * 40)

    try:
        await cli_discovery_service.start()
        await asyncio.sleep(2)  # Let discovery complete

        cli_commands = cli_discovery_service.get_discovered_commands()

        # Test commands that should work
        test_commands = [
            ("claude-flow__swarm", {"--help": True}),
            ("flow-nexus__deploy", {"--help": True}),
        ]

        for cmd_name, args in test_commands:
            if cmd_name in cli_commands:
                print(f"Testing {cmd_name}...")
                try:
                    result = await cli_discovery_service.execute_command(cmd_name, args)
                    if result.get("success"):
                        print(f"✅ {cmd_name} executed successfully")
                        stdout = result.get("stdout", "")
                        if "swarm" in stdout.lower() or "deploy" in stdout.lower():
                            print(f"   Output contains expected keywords")
                    else:
                        print(f"❌ {cmd_name} failed: {result.get('stderr', 'Unknown error')}")
                except Exception as e:
                    print(f"❌ {cmd_name} error: {e}")
            else:
                print(f"⚠️ {cmd_name} not found in discovered commands")

    except Exception as e:
        print(f"❌ Command testing failed: {e}")
    finally:
        await cli_discovery_service.stop()

async def main():
    """Main validation function"""
    success = await validate_removed_server_functionality()

    if success:
        await test_specific_commands()

    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
