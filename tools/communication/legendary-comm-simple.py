#!/usr/bin/env python3
"""
Simple Legendary Team Communication System
Standalone agent-to-agent communication
"""

import asyncio
import time
from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class MessageType(Enum):
    TASK_COORDINATION = "task_coordination"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    PROBLEM_SOLVING = "problem_solving"
    BROADCAST = "broadcast"
    DIVISION_DISCUSSION = "division_discussion"

@dataclass
class AgentMessage:
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    content: str
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

class SimpleLegendaryCommunicator:
    """Simple Legendary Team Communication System"""
    
    def __init__(self):
        self.agents = {
            "@serena": "Code Intelligence Empress",
            "@claude-flow": "Orchestration Emperor",
            "@rag-master": "Knowledge Oracle",
            "@performance-demon": "Speed Master",
            "@security-sentinel": "Cyber Guardian",
            "@document-architect": "Content Master",
            "@neural-architect": "AI Mastermind",
            "@integration-master": "Connector Emperor",
            "@challenge-conqueror": "Victory Master",
            "@browser-commander": "UI Emperor",
            "@data-oracle": "Insight Master",
            "@flow-nexus-guide": "Platform Master",
            "@flow-nexus-guide-v2": "Advanced Platform Master"
        }
        self.message_history = []
        self.stats = {
            'messages_sent': 0,
            'broadcasts_sent': 0,
            'direct_messages_sent': 0
        }
    
    async def send_message(self, sender: str, recipient: str, content: str, msg_type: MessageType = MessageType.TASK_COORDINATION):
        """Send a message between agents"""
        if sender not in self.agents:
            print(f"❌ Sender {sender} not found")
            return False
        
        if recipient != "*" and recipient not in self.agents:
            print(f"❌ Recipient {recipient} not found")
            return False
        
        message = AgentMessage(
            message_id=f"msg_{int(time.time() * 1000)}",
            sender_id=sender,
            recipient_id=recipient,
            message_type=msg_type,
            content=content
        )
        
        self.message_history.append(message)
        self.stats['messages_sent'] += 1
        
        if recipient == "*":
            self.stats['broadcasts_sent'] += 1
            print(f"📢 {sender} broadcasted: {content}")
        else:
            self.stats['direct_messages_sent'] += 1
            print(f"💬 {sender} → {recipient}: {content}")
        
        return True
    
    async def coordinate_task(self, sender: str, recipient: str, task: str):
        """Coordinate a task between agents"""
        return await self.send_message(sender, recipient, task, MessageType.TASK_COORDINATION)
    
    async def share_knowledge(self, sender: str, recipient: str, knowledge: str):
        """Share knowledge between agents"""
        return await self.send_message(sender, recipient, knowledge, MessageType.KNOWLEDGE_SHARING)
    
    async def solve_problem(self, sender: str, recipient: str, problem: str):
        """Request problem solving from another agent"""
        return await self.send_message(sender, recipient, problem, MessageType.PROBLEM_SOLVING)
    
    async def broadcast_to_all(self, sender: str, message: str):
        """Broadcast message to all agents"""
        return await self.send_message(sender, "*", message, MessageType.BROADCAST)
    
    def get_stats(self):
        """Get communication statistics"""
        return self.stats
    
    def get_agents(self):
        """Get list of agents"""
        return list(self.agents.keys())
    
    def get_message_history(self, limit: int = 10):
        """Get recent message history"""
        return self.message_history[-limit:]

async def demo():
    """Demonstrate agent communication"""
    comm = SimpleLegendaryCommunicator()
    
    print("🏆 LEGENDARY TEAM COMMUNICATION DEMO")
    print("=" * 50)
    
    # Show available agents
    print("\n🤖 Available Agents:")
    for agent in comm.get_agents():
        print(f"  {agent}")
    
    # Example communications
    print("\n💬 Communication Examples:")
    
    # Task coordination
    await comm.coordinate_task("@serena", "@claude-flow", "Let's analyze this React component together")
    
    # Knowledge sharing
    await comm.share_knowledge("@rag-master", "@neural-architect", "Here's the latest AI research data")
    
    # Problem solving
    await comm.solve_problem("@performance-demon", "@security-sentinel", "How do we optimize while maintaining security?")
    
    # Broadcast
    await comm.broadcast_to_all("@claude-flow", "Starting complex multi-agent workflow - all hands on deck!")
    
    # Show stats
    print(f"\n📊 Communication Stats: {comm.get_stats()}")
    
    # Show history
    print(f"\n📜 Recent Messages:")
    for msg in comm.get_message_history(5):
        print(f"  [{msg.timestamp:.1f}] {msg.sender_id} → {msg.recipient_id}: {msg.content}")

if __name__ == "__main__":
    asyncio.run(demo())
