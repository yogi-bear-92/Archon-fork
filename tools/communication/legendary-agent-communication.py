#!/usr/bin/env python3
"""
Legendary Team Agent Communication System
Enables agent-to-agent communication within Flow-Nexus swarms
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of agent messages"""
    TASK_COORDINATION = "task_coordination"
    KNOWLEDGE_SHARING = "knowledge_sharing"
    PROBLEM_SOLVING = "problem_solving"
    STATUS_UPDATE = "status_update"
    BROADCAST = "broadcast"
    DIVISION_DISCUSSION = "division_discussion"
    CROSS_SWARM = "cross_swarm"

class MessagePriority(Enum):
    """Message priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AgentMessage:
    """Agent message structure"""
    message_id: str
    sender_id: str
    recipient_id: str  # "*" for broadcast
    message_type: MessageType
    content: str
    priority: MessagePriority = MessagePriority.MEDIUM
    timestamp: float = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.metadata is None:
            self.metadata = {}

class AgentCommunicationSystem:
    """Agent communication system for Legendary Team"""
    
    def __init__(self, swarm_id: str):
        self.swarm_id = swarm_id
        self.agents = {}
        self.message_handlers = {}
        self.subscriptions = {}
        self.message_history = []
        self.communication_stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'broadcasts_sent': 0,
            'direct_messages_sent': 0,
            'start_time': time.time()
        }
    
    def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str]):
        """Register an agent in the communication system"""
        self.agents[agent_id] = {
            'type': agent_type,
            'capabilities': capabilities,
            'status': 'active',
            'last_seen': time.time()
        }
        logger.info(f"🤖 Registered agent {agent_id} ({agent_type})")
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler for specific message types"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
        logger.info(f"📝 Registered handler for {message_type.value} messages")
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message between agents"""
        try:
            # Validate sender
            if message.sender_id not in self.agents:
                logger.error(f"❌ Sender {message.sender_id} not registered")
                return False
            
            # Store message in history
            self.message_history.append(message)
            
            # Update stats
            self.communication_stats['messages_sent'] += 1
            if message.recipient_id == "*":
                self.communication_stats['broadcasts_sent'] += 1
            else:
                self.communication_stats['direct_messages_sent'] += 1
            
            # Process message
            if message.recipient_id == "*":
                # Broadcast message
                await self._broadcast_message(message)
            else:
                # Direct message
                await self._send_direct_message(message)
            
            logger.info(f"📤 Sent {message.message_type.value} message from {message.sender_id} to {message.recipient_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error sending message: {e}")
            return False
    
    async def _broadcast_message(self, message: AgentMessage):
        """Broadcast message to all agents"""
        for agent_id in self.agents:
            if agent_id != message.sender_id:
                await self._deliver_message(agent_id, message)
    
    async def _send_direct_message(self, message: AgentMessage):
        """Send direct message to specific agent"""
        if message.recipient_id in self.agents:
            await self._deliver_message(message.recipient_id, message)
        else:
            logger.warning(f"⚠️ Recipient {message.recipient_id} not found")
    
    async def _deliver_message(self, agent_id: str, message: AgentMessage):
        """Deliver message to specific agent"""
        try:
            # Update recipient's last seen
            if agent_id in self.agents:
                self.agents[agent_id]['last_seen'] = time.time()
            
            # Call registered handlers
            if message.message_type in self.message_handlers:
                for handler in self.message_handlers[message.message_type]:
                    try:
                        await handler(agent_id, message)
                    except Exception as e:
                        logger.error(f"❌ Error in message handler: {e}")
            
            self.communication_stats['messages_received'] += 1
            logger.debug(f"📨 Delivered message to {agent_id}")
            
        except Exception as e:
            logger.error(f"❌ Error delivering message to {agent_id}: {e}")
    
    async def send_task_coordination(self, sender_id: str, recipient_id: str, task: str, priority: MessagePriority = MessagePriority.MEDIUM):
        """Send task coordination message"""
        message = AgentMessage(
            message_id=f"task_{int(time.time() * 1000)}",
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.TASK_COORDINATION,
            content=task,
            priority=priority,
            metadata={'task_type': 'coordination'}
        )
        return await self.send_message(message)
    
    async def send_knowledge_sharing(self, sender_id: str, recipient_id: str, knowledge: str, knowledge_type: str = "general"):
        """Send knowledge sharing message"""
        message = AgentMessage(
            message_id=f"knowledge_{int(time.time() * 1000)}",
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.KNOWLEDGE_SHARING,
            content=knowledge,
            priority=MessagePriority.MEDIUM,
            metadata={'knowledge_type': knowledge_type}
        )
        return await self.send_message(message)
    
    async def send_problem_solving(self, sender_id: str, recipient_id: str, problem: str, urgency: str = "medium"):
        """Send problem solving request"""
        message = AgentMessage(
            message_id=f"problem_{int(time.time() * 1000)}",
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.PROBLEM_SOLVING,
            content=problem,
            priority=MessagePriority.HIGH if urgency == "high" else MessagePriority.MEDIUM,
            metadata={'urgency': urgency}
        )
        return await self.send_message(message)
    
    async def broadcast_to_division(self, sender_id: str, division: str, message_content: str):
        """Broadcast message to specific division"""
        # Get agents in division
        division_agents = self._get_division_agents(division)
        
        for agent_id in division_agents:
            if agent_id != sender_id:
                message = AgentMessage(
                    message_id=f"division_{int(time.time() * 1000)}",
                    sender_id=sender_id,
                    recipient_id=agent_id,
                    message_type=MessageType.DIVISION_DISCUSSION,
                    content=message_content,
                    priority=MessagePriority.MEDIUM,
                    metadata={'division': division}
                )
                await self.send_message(message)
    
    def _get_division_agents(self, division: str) -> List[str]:
        """Get agents in specific division"""
        division_mapping = {
            "code-division": ["@serena", "@document-architect", "@browser-commander"],
            "orchestration-division": ["@claude-flow", "@integration-master", "@flow-nexus-guide", "@flow-nexus-guide-v2"],
            "knowledge-division": ["@rag-master", "@neural-architect", "@data-oracle"],
            "performance-division": ["@performance-demon", "@challenge-conqueror"],
            "security-division": ["@security-sentinel"],
            "integration-division": ["@integration-master", "@claude-flow"],
            "content-division": ["@document-architect", "@rag-master"],
            "ui-division": ["@browser-commander", "@serena"],
            "analytics-division": ["@data-oracle", "@neural-architect"]
        }
        
        return division_mapping.get(division, [])
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        uptime = time.time() - self.communication_stats['start_time']
        return {
            **self.communication_stats,
            'uptime_seconds': uptime,
            'messages_per_minute': self.communication_stats['messages_sent'] / (uptime / 60),
            'active_agents': len([a for a in self.agents.values() if a['status'] == 'active']),
            'total_agents': len(self.agents)
        }
    
    def get_message_history(self, limit: int = 100) -> List[AgentMessage]:
        """Get recent message history"""
        return self.message_history[-limit:]
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            agent_id: {
                'type': agent['type'],
                'status': agent['status'],
                'last_seen': agent['last_seen'],
                'capabilities': agent['capabilities']
            }
            for agent_id, agent in self.agents.items()
        }

class LegendaryTeamCommunicator:
    """High-level interface for Legendary Team communication"""
    
    def __init__(self, swarm_id: str = "legendary-team"):
        self.comm_system = AgentCommunicationSystem(swarm_id)
        self._setup_legendary_agents()
        self._setup_message_handlers()
    
    def _setup_legendary_agents(self):
        """Setup all Legendary Team agents"""
        legendary_agents = {
            "@serena": {
                "type": "Code Intelligence Empress",
                "capabilities": ["semantic_code_intelligence", "ast_analysis", "refactoring", "architecture_analysis"]
            },
            "@claude-flow": {
                "type": "Orchestration Emperor", 
                "capabilities": ["multi_agent_orchestration", "swarm_coordination", "workflow_automation"]
            },
            "@rag-master": {
                "type": "Knowledge Oracle",
                "capabilities": ["knowledge_retrieval", "information_synthesis", "research_expertise"]
            },
            "@performance-demon": {
                "type": "Speed Master",
                "capabilities": ["system_optimization", "performance_benchmarking", "speed_optimization"]
            },
            "@security-sentinel": {
                "type": "Cyber Guardian",
                "capabilities": ["vulnerability_scanning", "security_auditing", "threat_detection"]
            },
            "@document-architect": {
                "type": "Content Master",
                "capabilities": ["content_creation", "documentation", "content_optimization"]
            },
            "@neural-architect": {
                "type": "AI Mastermind",
                "capabilities": ["ai_model_design", "neural_training", "pattern_recognition"]
            },
            "@integration-master": {
                "type": "Connector Emperor",
                "capabilities": ["system_integration", "api_integration", "platform_unification"]
            },
            "@challenge-conqueror": {
                "type": "Victory Master",
                "capabilities": ["problem_solving", "challenge_completion", "debugging"]
            },
            "@browser-commander": {
                "type": "UI Emperor",
                "capabilities": ["ui_testing", "automation", "user_experience"]
            },
            "@data-oracle": {
                "type": "Insight Master",
                "capabilities": ["data_analysis", "trend_prediction", "analytics"]
            },
            "@flow-nexus-guide": {
                "type": "Platform Master",
                "capabilities": ["platform_guidance", "feature_explanation", "troubleshooting"]
            },
            "@flow-nexus-guide-v2": {
                "type": "Advanced Platform Master",
                "capabilities": ["advanced_platform_expertise", "neural_guidance", "platform_mastery"]
            }
        }
        
        for agent_id, agent_info in legendary_agents.items():
            self.comm_system.register_agent(
                agent_id=agent_id,
                agent_type=agent_info["type"],
                capabilities=agent_info["capabilities"]
            )
    
    def _setup_message_handlers(self):
        """Setup message handlers for different message types"""
        
        async def handle_task_coordination(agent_id: str, message: AgentMessage):
            logger.info(f"🎯 {agent_id} received task coordination: {message.content}")
        
        async def handle_knowledge_sharing(agent_id: str, message: AgentMessage):
            logger.info(f"📚 {agent_id} received knowledge: {message.content}")
        
        async def handle_problem_solving(agent_id: str, message: AgentMessage):
            logger.info(f"🔧 {agent_id} received problem: {message.content}")
        
        async def handle_division_discussion(agent_id: str, message: AgentMessage):
            logger.info(f"🎭 {agent_id} received division message: {message.content}")
        
        # Register handlers
        self.comm_system.register_message_handler(MessageType.TASK_COORDINATION, handle_task_coordination)
        self.comm_system.register_message_handler(MessageType.KNOWLEDGE_SHARING, handle_knowledge_sharing)
        self.comm_system.register_message_handler(MessageType.PROBLEM_SOLVING, handle_problem_solving)
        self.comm_system.register_message_handler(MessageType.DIVISION_DISCUSSION, handle_division_discussion)
    
    async def coordinate_task(self, sender: str, recipient: str, task: str):
        """Coordinate a task between agents"""
        return await self.comm_system.send_task_coordination(sender, recipient, task)
    
    async def share_knowledge(self, sender: str, recipient: str, knowledge: str, knowledge_type: str = "general"):
        """Share knowledge between agents"""
        return await self.comm_system.send_knowledge_sharing(sender, recipient, knowledge, knowledge_type)
    
    async def solve_problem(self, sender: str, recipient: str, problem: str, urgency: str = "medium"):
        """Request problem solving from another agent"""
        return await self.comm_system.send_problem_solving(sender, recipient, problem, urgency)
    
    async def discuss_in_division(self, sender: str, division: str, topic: str):
        """Start a division discussion"""
        return await self.comm_system.broadcast_to_division(sender, division, topic)
    
    async def broadcast_to_all(self, sender: str, message: str):
        """Broadcast message to all agents"""
        broadcast_message = AgentMessage(
            message_id=f"broadcast_{int(time.time() * 1000)}",
            sender_id=sender,
            recipient_id="*",
            message_type=MessageType.BROADCAST,
            content=message,
            priority=MessagePriority.MEDIUM
        )
        return await self.comm_system.send_message(broadcast_message)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return self.comm_system.get_communication_stats()
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return self.comm_system.get_agent_status()

# Example usage
async def main():
    """Example usage of the Legendary Team Communication System"""
    
    # Initialize communicator
    communicator = LegendaryTeamCommunicator()
    
    print("🏆 LEGENDARY TEAM COMMUNICATION SYSTEM")
    print("=" * 50)
    
    # Example 1: Task coordination
    print("\n🎯 Task Coordination Example:")
    await communicator.coordinate_task(
        sender="@serena",
        recipient="@claude-flow", 
        task="Let's coordinate on analyzing this React component architecture"
    )
    
    # Example 2: Knowledge sharing
    print("\n📚 Knowledge Sharing Example:")
    await communicator.share_knowledge(
        sender="@rag-master",
        recipient="@neural-architect",
        knowledge="Here's the latest research on transformer architectures",
        knowledge_type="ai_research"
    )
    
    # Example 3: Division discussion
    print("\n🎭 Division Discussion Example:")
    await communicator.discuss_in_division(
        sender="@serena",
        division="code-division",
        topic="We need to refactor this legacy codebase - what's our strategy?"
    )
    
    # Example 4: Problem solving
    print("\n🔧 Problem Solving Example:")
    await communicator.solve_problem(
        sender="@performance-demon",
        recipient="@security-sentinel",
        problem="How do we optimize performance while maintaining security?",
        urgency="high"
    )
    
    # Example 5: Broadcast to all
    print("\n📢 Broadcast Example:")
    await communicator.broadcast_to_all(
        sender="@claude-flow",
        message="Starting complex multi-agent workflow - all hands on deck!"
    )
    
    # Show stats
    print("\n📊 Communication Statistics:")
    stats = communicator.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show agent status
    print("\n🤖 Agent Status:")
    agent_status = communicator.get_agent_status()
    for agent_id, status in agent_status.items():
        print(f"  {agent_id}: {status['type']} - {status['status']}")

if __name__ == "__main__":
    asyncio.run(main())
