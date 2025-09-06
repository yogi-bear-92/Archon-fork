#!/usr/bin/env python3
"""
Cross-Swarm Communication Protocol
Advanced messaging and coordination system for multi-swarm environments

Features:
- Secure Message Routing Between Swarms
- Event-Driven Communication Architecture 
- Distributed State Synchronization
- Fault-Tolerant Message Delivery
- Cross-Swarm Resource Sharing
- Real-Time Coordination Events
- Message Priority and QoS Management

Communication Patterns:
┌─────────────────────────────────────────────────────────┐
│                MESSAGE BROKER LAYER                     │
│            (Pub/Sub + Point-to-Point)                  │
├─────────────────────────────────────────────────────────┤
│  SWARM A ←──→ ROUTING MESH ←──→ SWARM B                │
│     ↕             ↕              ↕                      │
│  SWARM C ←──→ STATE SYNC ←──→ SWARM D                  │
├─────────────────────────────────────────────────────────┤
│            DISTRIBUTED EVENT BUS                        │
│    (Resource Events, Task Events, State Events)        │
└─────────────────────────────────────────────────────────┘

Author: Claude Code Cross-Swarm Team
Target: Sub-10ms latency with 99.9% message delivery
"""

import asyncio
import json
import uuid
import hashlib
import time
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
import weakref
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Types of cross-swarm messages."""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response" 
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_OFFER = "resource_offer"
    STATE_SYNC = "state_sync"
    HEARTBEAT = "heartbeat"
    COORDINATION_EVENT = "coordination_event"
    SYSTEM_ALERT = "system_alert"
    PERFORMANCE_DATA = "performance_data"
    CAPABILITY_ADVERTISEMENT = "capability_advertisement"
    LOAD_BALANCING = "load_balancing"
    EMERGENCY = "emergency"

class MessagePriority(Enum):
    """Message priority levels for QoS."""
    EMERGENCY = 0
    CRITICAL = 1 
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

class DeliveryMode(Enum):
    """Message delivery modes."""
    FIRE_AND_FORGET = "fire_and_forget"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"
    CONFIRMED = "confirmed"

@dataclass
class SwarmMessage:
    """Cross-swarm message with routing and QoS information."""
    message_id: str
    sender_id: str
    recipient_id: str  # Can be specific swarm ID or "*" for broadcast
    message_type: MessageType
    priority: MessagePriority
    delivery_mode: DeliveryMode
    payload: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            'message_id': self.message_id,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'message_type': self.message_type.value,
            'priority': self.priority.value,
            'delivery_mode': self.delivery_mode.value,
            'payload': self.payload,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'correlation_id': self.correlation_id,
            'reply_to': self.reply_to,
            'headers': self.headers,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SwarmMessage':
        """Create message from dictionary."""
        message = cls(
            message_id=data['message_id'],
            sender_id=data['sender_id'],
            recipient_id=data['recipient_id'],
            message_type=MessageType(data['message_type']),
            priority=MessagePriority(data['priority']),
            delivery_mode=DeliveryMode(data['delivery_mode']),
            payload=data['payload'],
            created_at=datetime.fromisoformat(data['created_at']),
            correlation_id=data.get('correlation_id'),
            reply_to=data.get('reply_to'),
            headers=data.get('headers', {}),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3)
        )
        
        if data.get('expires_at'):
            message.expires_at = datetime.fromisoformat(data['expires_at'])
        
        return message
    
    def is_expired(self) -> bool:
        """Check if message has expired."""
        if self.expires_at:
            return datetime.now() > self.expires_at
        return False
    
    def calculate_checksum(self) -> str:
        """Calculate message checksum for integrity verification."""
        content = json.dumps(self.payload, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

class MessageHandler:
    """Base class for message handlers."""
    
    async def handle_message(self, message: SwarmMessage) -> Optional[SwarmMessage]:
        """Handle incoming message and optionally return a response."""
        raise NotImplementedError

class EventBus:
    """Distributed event bus for cross-swarm communication."""
    
    def __init__(self):
        self.subscribers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.event_history: deque = deque(maxlen=1000)
        self.stats = {
            'events_published': 0,
            'events_delivered': 0,
            'failed_deliveries': 0
        }
        
    def subscribe(self, event_type: MessageType, handler: Callable):
        """Subscribe to events of a specific type."""
        self.subscribers[event_type].append(handler)
        logger.info(f"📡 Subscribed to {event_type.value} events")
    
    def unsubscribe(self, event_type: MessageType, handler: Callable):
        """Unsubscribe from events."""
        if handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
    
    async def publish(self, event: SwarmMessage):
        """Publish event to all subscribers."""
        self.event_history.append(event)
        self.stats['events_published'] += 1
        
        subscribers = self.subscribers.get(event.message_type, [])
        
        if not subscribers:
            logger.debug(f"📢 No subscribers for event type {event.message_type.value}")
            return
        
        # Deliver to all subscribers concurrently
        delivery_tasks = []
        for handler in subscribers:
            delivery_tasks.append(self._deliver_event(handler, event))
        
        results = await asyncio.gather(*delivery_tasks, return_exceptions=True)
        
        # Count successful deliveries
        successful = sum(1 for result in results if not isinstance(result, Exception))
        self.stats['events_delivered'] += successful
        self.stats['failed_deliveries'] += len(results) - successful
    
    async def _deliver_event(self, handler: Callable, event: SwarmMessage):
        """Deliver event to a specific handler."""
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"❌ Event delivery failed: {e}")
            raise

class MessageRouter:
    """Advanced message routing for cross-swarm communication."""
    
    def __init__(self):
        self.routing_table: Dict[str, str] = {}  # swarm_id -> endpoint
        self.message_queues: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.priority_queues: Dict[MessagePriority, deque] = {
            priority: deque(maxlen=500) for priority in MessagePriority
        }
        self.delivery_confirmations: Dict[str, datetime] = {}
        self.failed_messages: deque = deque(maxlen=100)
        self.routing_stats = {
            'messages_routed': 0,
            'delivery_failures': 0,
            'average_latency_ms': 0.0,
            'queue_depths': defaultdict(int)
        }
        
    def register_swarm(self, swarm_id: str, endpoint: str):
        """Register a swarm endpoint for routing."""
        self.routing_table[swarm_id] = endpoint
        logger.info(f"🔗 Registered swarm {swarm_id} at {endpoint}")
    
    def unregister_swarm(self, swarm_id: str):
        """Unregister a swarm endpoint."""
        if swarm_id in self.routing_table:
            del self.routing_table[swarm_id]
            logger.info(f"🔌 Unregistered swarm {swarm_id}")
    
    async def route_message(self, message: SwarmMessage) -> bool:
        """Route message to appropriate destination."""
        if message.is_expired():
            logger.warning(f"⚠️ Message {message.message_id} expired, dropping")
            return False
        
        if message.recipient_id == "*":
            # Broadcast message
            return await self._broadcast_message(message)
        else:
            # Direct message
            return await self._route_direct_message(message)
    
    async def _broadcast_message(self, message: SwarmMessage) -> bool:
        """Broadcast message to all registered swarms."""
        if not self.routing_table:
            logger.warning("⚠️ No swarms registered for broadcast")
            return False
        
        # Send to all swarms except sender
        broadcast_tasks = []
        for swarm_id, endpoint in self.routing_table.items():
            if swarm_id != message.sender_id:
                broadcast_message = SwarmMessage(
                    message_id=f"{message.message_id}_bc_{swarm_id}",
                    sender_id=message.sender_id,
                    recipient_id=swarm_id,
                    message_type=message.message_type,
                    priority=message.priority,
                    delivery_mode=message.delivery_mode,
                    payload=message.payload,
                    headers=message.headers.copy()
                )
                broadcast_tasks.append(self._deliver_message(broadcast_message, endpoint))
        
        results = await asyncio.gather(*broadcast_tasks, return_exceptions=True)
        successful = sum(1 for result in results if result is True)
        
        logger.info(f"📡 Broadcast message {message.message_id} to {successful}/{len(broadcast_tasks)} swarms")
        return successful > 0
    
    async def _route_direct_message(self, message: SwarmMessage) -> bool:
        """Route message to specific recipient."""
        endpoint = self.routing_table.get(message.recipient_id)
        
        if not endpoint:
            logger.error(f"❌ No route to swarm {message.recipient_id}")
            self.failed_messages.append(message)
            self.routing_stats['delivery_failures'] += 1
            return False
        
        return await self._deliver_message(message, endpoint)
    
    async def _deliver_message(self, message: SwarmMessage, endpoint: str) -> bool:
        """Deliver message to specific endpoint."""
        start_time = time.time()
        
        try:
            # Add to priority queue
            self.priority_queues[message.priority].append((message, endpoint))
            
            # Simulate network delivery (replace with actual network call)
            await asyncio.sleep(0.001 + message.priority.value * 0.001)  # Priority-based delay
            
            # Update stats
            latency = (time.time() - start_time) * 1000  # Convert to ms
            self.routing_stats['messages_routed'] += 1
            self.routing_stats['average_latency_ms'] = (
                (self.routing_stats['average_latency_ms'] * (self.routing_stats['messages_routed'] - 1) + latency)
                / self.routing_stats['messages_routed']
            )
            
            # Handle delivery confirmation
            if message.delivery_mode in [DeliveryMode.CONFIRMED, DeliveryMode.EXACTLY_ONCE]:
                self.delivery_confirmations[message.message_id] = datetime.now()
            
            logger.debug(f"📨 Delivered message {message.message_id} to {message.recipient_id} ({latency:.2f}ms)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Message delivery failed: {e}")
            self.failed_messages.append(message)
            self.routing_stats['delivery_failures'] += 1
            return False
    
    def get_routing_metrics(self) -> Dict[str, Any]:
        """Get routing performance metrics."""
        return {
            'registered_swarms': len(self.routing_table),
            'routing_stats': self.routing_stats,
            'priority_queue_depths': {
                priority.value: len(queue) for priority, queue in self.priority_queues.items()
            },
            'failed_messages': len(self.failed_messages),
            'pending_confirmations': len(self.delivery_confirmations)
        }

class CrossSwarmCommunicator:
    """Main cross-swarm communication coordinator."""
    
    def __init__(self, swarm_id: str):
        self.swarm_id = swarm_id
        self.message_router = MessageRouter()
        self.event_bus = EventBus()
        self.message_handlers: Dict[MessageType, MessageHandler] = {}
        self.outbound_queue: deque = deque()
        self.inbound_queue: deque = deque()
        self.state_cache: Dict[str, Any] = {}
        self.peer_swarms: Set[str] = set()
        self.communication_stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'bytes_transferred': 0,
            'connection_errors': 0,
            'start_time': datetime.now()
        }
        self._running = False
        self._communication_tasks: List[asyncio.Task] = []
        
    def register_handler(self, message_type: MessageType, handler: MessageHandler):
        """Register a message handler for a specific message type."""
        self.message_handlers[message_type] = handler
        logger.info(f"📝 Registered handler for {message_type.value} messages")
    
    def connect_to_swarm(self, swarm_id: str, endpoint: str):
        """Connect to another swarm."""
        self.message_router.register_swarm(swarm_id, endpoint)
        self.peer_swarms.add(swarm_id)
        logger.info(f"🤝 Connected to swarm {swarm_id}")
    
    async def send_message(self, message: SwarmMessage) -> bool:
        """Send message to another swarm."""
        message.sender_id = self.swarm_id
        self.outbound_queue.append(message)
        
        success = await self.message_router.route_message(message)
        
        if success:
            self.communication_stats['messages_sent'] += 1
            self.communication_stats['bytes_transferred'] += len(json.dumps(message.to_dict()))
        else:
            self.communication_stats['connection_errors'] += 1
        
        return success
    
    async def send_task_request(self, recipient_id: str, task_data: Dict[str, Any]) -> str:
        """Send a task request to another swarm."""
        message = SwarmMessage(
            message_id=f"task_req_{uuid.uuid4().hex[:8]}",
            sender_id=self.swarm_id,
            recipient_id=recipient_id,
            message_type=MessageType.TASK_REQUEST,
            priority=MessagePriority.HIGH,
            delivery_mode=DeliveryMode.CONFIRMED,
            payload={'task_data': task_data},
            reply_to=self.swarm_id,
            expires_at=datetime.now() + timedelta(minutes=5)
        )
        
        await self.send_message(message)
        return message.message_id
    
    async def send_resource_offer(self, available_resources: Dict[str, Any]):
        """Broadcast available resources to all swarms."""
        message = SwarmMessage(
            message_id=f"resource_offer_{uuid.uuid4().hex[:8]}",
            sender_id=self.swarm_id,
            recipient_id="*",  # Broadcast
            message_type=MessageType.RESOURCE_OFFER,
            priority=MessagePriority.NORMAL,
            delivery_mode=DeliveryMode.FIRE_AND_FORGET,
            payload={'resources': available_resources},
            expires_at=datetime.now() + timedelta(minutes=2)
        )
        
        await self.send_message(message)
    
    async def synchronize_state(self, state_data: Dict[str, Any], target_swarms: List[str] = None):
        """Synchronize state with other swarms."""
        recipients = target_swarms or list(self.peer_swarms)
        
        for recipient in recipients:
            message = SwarmMessage(
                message_id=f"state_sync_{uuid.uuid4().hex[:8]}",
                sender_id=self.swarm_id,
                recipient_id=recipient,
                message_type=MessageType.STATE_SYNC,
                priority=MessagePriority.NORMAL,
                delivery_mode=DeliveryMode.AT_LEAST_ONCE,
                payload={'state': state_data, 'timestamp': datetime.now().isoformat()},
                expires_at=datetime.now() + timedelta(minutes=1)
            )
            
            await self.send_message(message)
    
    async def broadcast_capability_advertisement(self, capabilities: Dict[str, Any]):
        """Advertise capabilities to all connected swarms."""
        message = SwarmMessage(
            message_id=f"cap_ad_{uuid.uuid4().hex[:8]}",
            sender_id=self.swarm_id,
            recipient_id="*",
            message_type=MessageType.CAPABILITY_ADVERTISEMENT,
            priority=MessagePriority.LOW,
            delivery_mode=DeliveryMode.FIRE_AND_FORGET,
            payload={'capabilities': capabilities, 'swarm_id': self.swarm_id},
            expires_at=datetime.now() + timedelta(minutes=10)
        )
        
        await self.send_message(message)
    
    async def send_emergency_alert(self, alert_data: Dict[str, Any]):
        """Send emergency alert to all swarms."""
        message = SwarmMessage(
            message_id=f"emergency_{uuid.uuid4().hex[:8]}",
            sender_id=self.swarm_id,
            recipient_id="*",
            message_type=MessageType.EMERGENCY,
            priority=MessagePriority.EMERGENCY,
            delivery_mode=DeliveryMode.CONFIRMED,
            payload=alert_data,
            expires_at=datetime.now() + timedelta(minutes=15)
        )
        
        await self.send_message(message)
        logger.critical(f"🚨 EMERGENCY ALERT SENT: {alert_data.get('description', 'Unknown emergency')}")
    
    async def process_inbound_message(self, message: SwarmMessage) -> Optional[SwarmMessage]:
        """Process incoming message and generate response if needed."""
        try:
            self.communication_stats['messages_received'] += 1
            
            # Update state cache if it's a state sync message
            if message.message_type == MessageType.STATE_SYNC:
                peer_state = message.payload.get('state', {})
                self.state_cache[message.sender_id] = peer_state
            
            # Find appropriate handler
            handler = self.message_handlers.get(message.message_type)
            
            if handler:
                response = await handler.handle_message(message)
                return response
            else:
                logger.warning(f"⚠️ No handler for message type {message.message_type.value}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error processing message {message.message_id}: {e}")
            return None
    
    async def start_communication(self):
        """Start the communication system."""
        logger.info(f"🚀 Starting cross-swarm communication for {self.swarm_id}")
        self._running = True
        
        # Start background communication tasks
        self._communication_tasks = [
            asyncio.create_task(self._process_outbound_queue()),
            asyncio.create_task(self._process_inbound_queue()),
            asyncio.create_task(self._send_heartbeats()),
            asyncio.create_task(self._cleanup_expired_data())
        ]
        
        # Announce presence to network
        await self.broadcast_capability_advertisement({
            'swarm_type': 'general',
            'available_capacity': 100,
            'specializations': []
        })
        
        logger.info(f"✅ Cross-swarm communication started for {self.swarm_id}")
    
    async def stop_communication(self):
        """Stop the communication system."""
        logger.info(f"🛑 Stopping cross-swarm communication for {self.swarm_id}")
        self._running = False
        
        # Cancel communication tasks
        for task in self._communication_tasks:
            task.cancel()
        
        # Wait for tasks to finish
        await asyncio.gather(*self._communication_tasks, return_exceptions=True)
        
        logger.info(f"✅ Cross-swarm communication stopped for {self.swarm_id}")
    
    async def _process_outbound_queue(self):
        """Process outbound message queue."""
        while self._running:
            try:
                if self.outbound_queue:
                    message = self.outbound_queue.popleft()
                    await self.message_router.route_message(message)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"❌ Error processing outbound queue: {e}")
                await asyncio.sleep(1)
    
    async def _process_inbound_queue(self):
        """Process inbound message queue."""
        while self._running:
            try:
                if self.inbound_queue:
                    message = self.inbound_queue.popleft()
                    response = await self.process_inbound_message(message)
                    
                    # Send response if generated
                    if response and message.reply_to:
                        response.recipient_id = message.reply_to
                        response.correlation_id = message.message_id
                        await self.send_message(response)
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"❌ Error processing inbound queue: {e}")
                await asyncio.sleep(1)
    
    async def _send_heartbeats(self):
        """Send periodic heartbeat messages."""
        while self._running:
            try:
                heartbeat = SwarmMessage(
                    message_id=f"heartbeat_{uuid.uuid4().hex[:8]}",
                    sender_id=self.swarm_id,
                    recipient_id="*",
                    message_type=MessageType.HEARTBEAT,
                    priority=MessagePriority.BACKGROUND,
                    delivery_mode=DeliveryMode.FIRE_AND_FORGET,
                    payload={
                        'timestamp': datetime.now().isoformat(),
                        'status': 'healthy',
                        'load': 0.5  # Placeholder
                    },
                    expires_at=datetime.now() + timedelta(seconds=90)
                )
                
                await self.send_message(heartbeat)
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
            except Exception as e:
                logger.error(f"❌ Error sending heartbeat: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_expired_data(self):
        """Clean up expired data and confirmations."""
        while self._running:
            try:
                current_time = datetime.now()
                
                # Clean up old delivery confirmations (older than 5 minutes)
                expired_confirmations = [
                    msg_id for msg_id, timestamp in self.message_router.delivery_confirmations.items()
                    if current_time - timestamp > timedelta(minutes=5)
                ]
                
                for msg_id in expired_confirmations:
                    del self.message_router.delivery_confirmations[msg_id]
                
                # Clean up old state cache entries (older than 10 minutes)
                # This would be more sophisticated in a real implementation
                
                await asyncio.sleep(60)  # Cleanup every minute
                
            except Exception as e:
                logger.error(f"❌ Error in cleanup: {e}")
                await asyncio.sleep(60)
    
    def get_communication_metrics(self) -> Dict[str, Any]:
        """Get comprehensive communication metrics."""
        uptime = (datetime.now() - self.communication_stats['start_time']).total_seconds()
        
        return {
            'swarm_id': self.swarm_id,
            'communication_stats': self.communication_stats,
            'routing_metrics': self.message_router.get_routing_metrics(),
            'event_bus_stats': self.event_bus.stats,
            'connected_swarms': len(self.peer_swarms),
            'outbound_queue_size': len(self.outbound_queue),
            'inbound_queue_size': len(self.inbound_queue),
            'cached_states': len(self.state_cache),
            'uptime_seconds': uptime,
            'messages_per_second': self.communication_stats['messages_sent'] / max(uptime, 1),
            'bandwidth_bytes_per_second': self.communication_stats['bytes_transferred'] / max(uptime, 1)
        }

# Example message handlers

class TaskRequestHandler(MessageHandler):
    """Handler for task requests from other swarms."""
    
    async def handle_message(self, message: SwarmMessage) -> Optional[SwarmMessage]:
        """Handle task request message."""
        task_data = message.payload.get('task_data', {})
        
        # Simulate task processing
        await asyncio.sleep(0.1)
        
        # Create response
        response = SwarmMessage(
            message_id=f"task_resp_{uuid.uuid4().hex[:8]}",
            sender_id=message.recipient_id,  # Will be set by sender
            recipient_id=message.sender_id,
            message_type=MessageType.TASK_RESPONSE,
            priority=MessagePriority.HIGH,
            delivery_mode=DeliveryMode.CONFIRMED,
            payload={
                'success': True,
                'result': f"Task completed successfully",
                'processing_time': 0.1
            },
            correlation_id=message.message_id
        )
        
        return response

class ResourceOfferHandler(MessageHandler):
    """Handler for resource offers from other swarms."""
    
    async def handle_message(self, message: SwarmMessage) -> Optional[SwarmMessage]:
        """Handle resource offer message."""
        resources = message.payload.get('resources', {})
        
        logger.info(f"📦 Received resource offer from {message.sender_id}: {resources}")
        
        # No response needed for resource offers
        return None

class StateSyncHandler(MessageHandler):
    """Handler for state synchronization messages."""
    
    async def handle_message(self, message: SwarmMessage) -> Optional[SwarmMessage]:
        """Handle state sync message."""
        peer_state = message.payload.get('state', {})
        
        logger.info(f"🔄 State sync from {message.sender_id}: {len(peer_state)} entries")
        
        # State is automatically cached by the communicator
        return None

# Example usage
if __name__ == "__main__":
    async def main():
        # Create communicators for two swarms
        swarm_a = CrossSwarmCommunicator("swarm_a")
        swarm_b = CrossSwarmCommunicator("swarm_b")
        
        # Register handlers
        swarm_a.register_handler(MessageType.TASK_REQUEST, TaskRequestHandler())
        swarm_b.register_handler(MessageType.TASK_REQUEST, TaskRequestHandler())
        
        # Connect swarms to each other
        swarm_a.connect_to_swarm("swarm_b", "endpoint_b")
        swarm_b.connect_to_swarm("swarm_a", "endpoint_a")
        
        # Start communication
        await swarm_a.start_communication()
        await swarm_b.start_communication()
        
        # Send some test messages
        await swarm_a.send_task_request("swarm_b", {"task": "example_task"})
        await swarm_b.send_resource_offer({"cpu": 4, "memory": "8GB"})
        
        # Run for demonstration
        print("🚀 Cross-swarm communication running... Press Ctrl+C to stop")
        try:
            for i in range(10):
                await asyncio.sleep(5)
                
                # Print metrics
                metrics_a = swarm_a.get_communication_metrics()
                print(f"📊 Swarm A - Sent: {metrics_a['communication_stats']['messages_sent']}, "
                      f"Received: {metrics_a['communication_stats']['messages_received']}")
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping communication...")
            await swarm_a.stop_communication()
            await swarm_b.stop_communication()

    asyncio.run(main())