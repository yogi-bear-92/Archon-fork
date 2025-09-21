#!/usr/bin/env python3
"""
Legendary Team Communication CLI
Command-line interface for agent-to-agent communication
"""

import asyncio
import sys
import json
from typing import List, Dict, Any
from legendary_agent_communication import LegendaryTeamCommunicator, MessageType, MessagePriority

class LegendaryCommunicationCLI:
    """CLI for Legendary Team Communication"""
    
    def __init__(self):
        self.communicator = LegendaryTeamCommunicator()
        self.running = False
    
    def show_help(self):
        """Show help information"""
        print("🏆 LEGENDARY TEAM COMMUNICATION CLI")
        print("=" * 50)
        print()
        print("USAGE:")
        print("  python legendary-communication-cli.py <command> [options]")
        print()
        print("COMMANDS:")
        print("  start                    - Start interactive communication mode")
        print("  coordinate <sender> <recipient> <task> - Coordinate task between agents")
        print("  share <sender> <recipient> <knowledge> - Share knowledge between agents")
        print("  solve <sender> <recipient> <problem> - Request problem solving")
        print("  discuss <sender> <division> <topic> - Start division discussion")
        print("  broadcast <sender> <message> - Broadcast message to all agents")
        print("  stats                    - Show communication statistics")
        print("  status                   - Show agent status")
        print("  history [limit]          - Show message history")
        print("  help                     - Show this help")
        print()
        print("AGENTS:")
        print("  @serena                  - Code Intelligence Empress")
        print("  @claude-flow             - Orchestration Emperor")
        print("  @rag-master              - Knowledge Oracle")
        print("  @performance-demon       - Speed Master")
        print("  @security-sentinel       - Cyber Guardian")
        print("  @document-architect      - Content Master")
        print("  @neural-architect        - AI Mastermind")
        print("  @integration-master      - Connector Emperor")
        print("  @challenge-conqueror     - Victory Master")
        print("  @browser-commander       - UI Emperor")
        print("  @data-oracle             - Insight Master")
        print("  @flow-nexus-guide        - Platform Master")
        print("  @flow-nexus-guide-v2     - Advanced Platform Master")
        print()
        print("DIVISIONS:")
        print("  @code-division           - Code Intelligence Division")
        print("  @orchestration-division  - Orchestration Division")
        print("  @knowledge-division      - Knowledge Division")
        print("  @performance-division    - Performance Division")
        print("  @security-division       - Security Division")
        print("  @integration-division    - Integration Division")
        print("  @content-division        - Content Division")
        print("  @ui-division             - UI Division")
        print("  @analytics-division      - Analytics Division")
        print()
        print("EXAMPLES:")
        print("  python legendary-communication-cli.py coordinate @serena @claude-flow 'Let\\'s analyze this code'")
        print("  python legendary-communication-cli.py share @rag-master @neural-architect 'AI research data'")
        print("  python legendary-communication-cli.py discuss @serena @code-division 'Refactoring strategy'")
        print("  python legendary-communication-cli.py broadcast @claude-flow 'Starting complex workflow'")
        print("  python legendary-communication-cli.py start")
    
    async def start_interactive_mode(self):
        """Start interactive communication mode"""
        print("🏆 LEGENDARY TEAM INTERACTIVE COMMUNICATION")
        print("=" * 50)
        print("Type 'help' for commands, 'quit' to exit")
        print()
        
        self.running = True
        
        while self.running:
            try:
                command = input("legendary> ").strip()
                
                if not command:
                    continue
                
                if command.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    self.running = False
                    break
                
                if command.lower() == 'help':
                    self.show_interactive_help()
                    continue
                
                if command.lower() == 'stats':
                    await self.show_stats()
                    continue
                
                if command.lower() == 'status':
                    await self.show_status()
                    continue
                
                if command.lower().startswith('history'):
                    parts = command.split()
                    limit = int(parts[1]) if len(parts) > 1 else 10
                    await self.show_history(limit)
                    continue
                
                # Parse command
                parts = command.split(' ', 2)
                if len(parts) < 2:
                    print("❌ Invalid command format. Use 'help' for examples.")
                    continue
                
                cmd_type = parts[0].lower()
                
                if cmd_type == 'coordinate' and len(parts) >= 3:
                    sender, recipient, task = parts[1], parts[2], parts[3] if len(parts) > 3 else ""
                    await self.coordinate_task(sender, recipient, task)
                
                elif cmd_type == 'share' and len(parts) >= 3:
                    sender, recipient, knowledge = parts[1], parts[2], parts[3] if len(parts) > 3 else ""
                    await self.share_knowledge(sender, recipient, knowledge)
                
                elif cmd_type == 'solve' and len(parts) >= 3:
                    sender, recipient, problem = parts[1], parts[2], parts[3] if len(parts) > 3 else ""
                    await self.solve_problem(sender, recipient, problem)
                
                elif cmd_type == 'discuss' and len(parts) >= 3:
                    sender, division, topic = parts[1], parts[2], parts[3] if len(parts) > 3 else ""
                    await self.discuss_division(sender, division, topic)
                
                elif cmd_type == 'broadcast' and len(parts) >= 2:
                    sender, message = parts[1], parts[2] if len(parts) > 2 else ""
                    await self.broadcast_message(sender, message)
                
                else:
                    print("❌ Unknown command. Use 'help' for available commands.")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                self.running = False
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_interactive_help(self):
        """Show help for interactive mode"""
        print("\n📋 INTERACTIVE COMMANDS:")
        print("  coordinate <sender> <recipient> <task> - Coordinate task")
        print("  share <sender> <recipient> <knowledge> - Share knowledge")
        print("  solve <sender> <recipient> <problem> - Solve problem")
        print("  discuss <sender> <division> <topic> - Division discussion")
        print("  broadcast <sender> <message> - Broadcast to all")
        print("  stats - Show statistics")
        print("  status - Show agent status")
        print("  history [limit] - Show message history")
        print("  help - Show this help")
        print("  quit - Exit")
        print()
    
    async def coordinate_task(self, sender: str, recipient: str, task: str):
        """Coordinate a task between agents"""
        if not task:
            print("❌ Task description required")
            return
        
        print(f"🎯 Coordinating task from {sender} to {recipient}...")
        success = await self.communicator.coordinate_task(sender, recipient, task)
        
        if success:
            print(f"✅ Task coordination sent: '{task}'")
        else:
            print("❌ Failed to send task coordination")
    
    async def share_knowledge(self, sender: str, recipient: str, knowledge: str):
        """Share knowledge between agents"""
        if not knowledge:
            print("❌ Knowledge content required")
            return
        
        print(f"📚 Sharing knowledge from {sender} to {recipient}...")
        success = await self.communicator.share_knowledge(sender, recipient, knowledge)
        
        if success:
            print(f"✅ Knowledge shared: '{knowledge}'")
        else:
            print("❌ Failed to share knowledge")
    
    async def solve_problem(self, sender: str, recipient: str, problem: str):
        """Request problem solving from another agent"""
        if not problem:
            print("❌ Problem description required")
            return
        
        print(f"🔧 Requesting problem solving from {recipient}...")
        success = await self.communicator.solve_problem(sender, recipient, problem)
        
        if success:
            print(f"✅ Problem solving requested: '{problem}'")
        else:
            print("❌ Failed to request problem solving")
    
    async def discuss_division(self, sender: str, division: str, topic: str):
        """Start a division discussion"""
        if not topic:
            print("❌ Discussion topic required")
            return
        
        print(f"🎭 Starting division discussion in {division}...")
        success = await self.communicator.discuss_in_division(sender, division, topic)
        
        if success:
            print(f"✅ Division discussion started: '{topic}'")
        else:
            print("❌ Failed to start division discussion")
    
    async def broadcast_message(self, sender: str, message: str):
        """Broadcast message to all agents"""
        if not message:
            print("❌ Message content required")
            return
        
        print(f"📢 Broadcasting message from {sender}...")
        success = await self.communicator.broadcast_to_all(sender, message)
        
        if success:
            print(f"✅ Message broadcasted: '{message}'")
        else:
            print("❌ Failed to broadcast message")
    
    async def show_stats(self):
        """Show communication statistics"""
        stats = self.communicator.get_stats()
        print("\n📊 COMMUNICATION STATISTICS:")
        print("=" * 30)
        for key, value in stats.items():
            print(f"  {key}: {value}")
        print()
    
    async def show_status(self):
        """Show agent status"""
        status = self.communicator.get_agent_status()
        print("\n🤖 AGENT STATUS:")
        print("=" * 20)
        for agent_id, agent_info in status.items():
            print(f"  {agent_id}: {agent_info['type']} - {agent_info['status']}")
        print()
    
    async def show_history(self, limit: int = 10):
        """Show message history"""
        history = self.communicator.comm_system.get_message_history(limit)
        print(f"\n📜 MESSAGE HISTORY (Last {len(history)} messages):")
        print("=" * 40)
        
        if not history:
            print("  No messages yet")
        else:
            for msg in history[-limit:]:
                print(f"  [{msg.timestamp:.1f}] {msg.sender_id} → {msg.recipient_id}")
                print(f"    {msg.message_type.value}: {msg.content}")
                print()
    
    async def run_command(self, args: List[str]):
        """Run a single command"""
        if not args:
            self.show_help()
            return
        
        command = args[0].lower()
        
        if command == "start":
            await self.start_interactive_mode()
        
        elif command == "coordinate" and len(args) >= 4:
            sender, recipient, task = args[1], args[2], " ".join(args[3:])
            await self.coordinate_task(sender, recipient, task)
        
        elif command == "share" and len(args) >= 4:
            sender, recipient, knowledge = args[1], args[2], " ".join(args[3:])
            await self.share_knowledge(sender, recipient, knowledge)
        
        elif command == "solve" and len(args) >= 4:
            sender, recipient, problem = args[1], args[2], " ".join(args[3:])
            await self.solve_problem(sender, recipient, problem)
        
        elif command == "discuss" and len(args) >= 4:
            sender, division, topic = args[1], args[2], " ".join(args[3:])
            await self.discuss_division(sender, division, topic)
        
        elif command == "broadcast" and len(args) >= 3:
            sender, message = args[1], " ".join(args[2:])
            await self.broadcast_message(sender, message)
        
        elif command == "stats":
            await self.show_stats()
        
        elif command == "status":
            await self.show_status()
        
        elif command == "history":
            limit = int(args[1]) if len(args) > 1 else 10
            await self.show_history(limit)
        
        elif command == "help":
            self.show_help()
        
        else:
            print("❌ Unknown command. Use 'help' for available commands.")

async def main():
    """Main entry point"""
    cli = LegendaryCommunicationCLI()
    await cli.run_command(sys.argv[1:])

if __name__ == "__main__":
    asyncio.run(main())
