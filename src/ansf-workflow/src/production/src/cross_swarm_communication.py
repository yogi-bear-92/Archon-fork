#!/usr/bin/env python3
"""
Cross-Swarm Communication System
Mock implementation for testing cross-swarm messaging
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    HEARTBEAT = "heartbeat"
    PERFORMANCE_DATA = "performance_data"
    RESOURCE_UPDATE = "resource_update"
    COORDINATION_REQUEST = "coordination_request"

class MessagePriority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class DeliveryMode(Enum):
    FIRE_AND_FORGET = "fire_and_forget"
    ACKNOWLEDGED = "acknowledged"
    CONFIRMED = "confirmed"

class SwarmMessage:
    def __init__(self, message_id: str, sender_id: str, recipient_id: str,
                 message_type: MessageType, priority: MessagePriority,
                 delivery_mode: DeliveryMode, payload: Dict[str, Any],
                 timestamp: Optional[datetime] = None):
        self.message_id = message_id
        self.sender_id = sender_id
        self.recipient_id = recipient_id
        self.message_type = message_type
        self.priority = priority
        self.delivery_mode = delivery_mode
        self.payload = payload
        self.timestamp = timestamp or datetime.now()
        self.delivery_attempts = 0
        self.acknowledged = False

class MessageRouter:
    def __init__(self):
        self.routing_table: Dict[str, str] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.delivery_stats = {
            'messages_routed': 0,
            'delivery_failures': 0,
            'average_delivery_time': 0.0
        }
    
    def add_route(self, swarm_id: str, endpoint: str):
        """Add routing entry for a swarm"""
        self.routing_table[swarm_id] = endpoint
    
    async def route_message(self, message: SwarmMessage) -> bool:
        """Route message to destination"""
        if message.recipient_id not in self.routing_table:
            logger.warning(f"No route found for swarm {message.recipient_id}")
            self.delivery_stats['delivery_failures'] += 1
            return False
        
        # Simulate message delivery
        await asyncio.sleep(0.01)  # Simulate network latency
        
        message.delivery_attempts += 1
        self.delivery_stats['messages_routed'] += 1
        
        return True
    
    def register_handler(self, message_type: MessageType, handler: Callable):
        """Register message handler"""
        self.message_handlers[message_type.value] = handler

class CrossSwarmCommunicator:
    def __init__(self, swarm_id: str):
        self.swarm_id = swarm_id
        self.message_router = MessageRouter()
        self.message_handlers: Dict[str, Callable] = {}
        self.peer_swarms: Dict[str, str] = {}
        self.communication_stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'connection_errors': 0,
            'successful_deliveries': 0
        }
        self.pending_acknowledgments: Dict[str, SwarmMessage] = {}
    
    def connect_to_swarm(self, swarm_id: str, endpoint: str):
        """Connect to another swarm"""
        self.peer_swarms[swarm_id] = endpoint
        self.message_router.add_route(swarm_id, endpoint)
        logger.info(f"Connected to swarm {swarm_id} at {endpoint}")
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register handler for specific message type"""
        self.message_handlers[message_type.value] = handler
        self.message_router.register_handler(message_type, handler)
    
    async def send_message(self, message: SwarmMessage) -> bool:
        """Send message to another swarm"""
        try:
            success = await self.message_router.route_message(message)
            
            if success:
                self.communication_stats['messages_sent'] += 1
                self.communication_stats['successful_deliveries'] += 1
                
                if message.delivery_mode in [DeliveryMode.ACKNOWLEDGED, DeliveryMode.CONFIRMED]:
                    self.pending_acknowledgments[message.message_id] = message
                
                return True
            else:
                self.communication_stats['connection_errors'] += 1
                return False
                
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            self.communication_stats['connection_errors'] += 1
            return False
    
    async def send_task_request(self, target_swarm_id: str, task_data: Dict[str, Any]) -> str:
        """Send task request to target swarm"""
        message_id = str(uuid.uuid4())
        
        message = SwarmMessage(
            message_id=message_id,
            sender_id=self.swarm_id,
            recipient_id=target_swarm_id,
            message_type=MessageType.TASK_REQUEST,
            priority=MessagePriority.HIGH,
            delivery_mode=DeliveryMode.CONFIRMED,
            payload=task_data
        )
        
        success = await self.send_message(message)
        return message_id if success else None
    
    async def send_heartbeat(self, target_swarm_id: str) -> bool:
        """Send heartbeat to target swarm"""
        message = SwarmMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.swarm_id,
            recipient_id=target_swarm_id,
            message_type=MessageType.HEARTBEAT,
            priority=MessagePriority.LOW,
            delivery_mode=DeliveryMode.FIRE_AND_FORGET,
            payload={'timestamp': datetime.now().isoformat(), 'status': 'active'}
        )
        
        return await self.send_message(message)
    
    async def broadcast_performance_data(self, performance_data: Dict[str, Any]):
        """Broadcast performance data to all connected swarms"""
        message_id = str(uuid.uuid4())
        
        broadcast_tasks = []
        for swarm_id in self.peer_swarms.keys():
            message = SwarmMessage(
                message_id=f"{message_id}_{swarm_id}",
                sender_id=self.swarm_id,
                recipient_id=swarm_id,
                message_type=MessageType.PERFORMANCE_DATA,
                priority=MessagePriority.NORMAL,
                delivery_mode=DeliveryMode.FIRE_AND_FORGET,
                payload=performance_data
            )
            broadcast_tasks.append(self.send_message(message))
        
        if broadcast_tasks:
            results = await asyncio.gather(*broadcast_tasks, return_exceptions=True)
            return all(result is True for result in results if not isinstance(result, Exception))
        
        return True
    
    async def handle_incoming_message(self, message: SwarmMessage) -> bool:
        """Handle incoming message"""
        try:
            self.communication_stats['messages_received'] += 1
            
            message_type = message.message_type.value
            if message_type in self.message_handlers:
                handler = self.message_handlers[message_type]
                await handler(message)
            
            # Send acknowledgment if required
            if message.delivery_mode in [DeliveryMode.ACKNOWLEDGED, DeliveryMode.CONFIRMED]:
                await self._send_acknowledgment(message)
            
            return True
            
        except Exception as e:
            logger.error(f"Error handling incoming message: {e}")
            return False
    
    async def _send_acknowledgment(self, original_message: SwarmMessage):
        """Send acknowledgment for received message"""
        ack_message = SwarmMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.swarm_id,
            recipient_id=original_message.sender_id,
            message_type=MessageType.TASK_RESPONSE,
            priority=MessagePriority.NORMAL,
            delivery_mode=DeliveryMode.FIRE_AND_FORGET,
            payload={
                'acknowledgment': True,
                'original_message_id': original_message.message_id,
                'status': 'received'
            }
        )
        
        await self.send_message(ack_message)
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        return {
            'swarm_id': self.swarm_id,
            'connected_swarms': len(self.peer_swarms),
            'communication_stats': self.communication_stats.copy(),
            'pending_acknowledgments': len(self.pending_acknowledgments),
            'router_stats': self.message_router.delivery_stats.copy()
        }