#!/usr/bin/env python3
"""
Cross-Swarm Intelligence Communication and Knowledge Sharing System
Phase 3 Implementation - Advanced inter-swarm coordination and collective intelligence
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from collections import deque, defaultdict
import hashlib
import threading
import weakref
import gc
from enum import Enum

# Networking and communication imports
import websockets
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import zmq
import zmq.asyncio

# Distributed coordination imports
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

class SwarmRole(Enum):
    """Roles in cross-swarm coordination"""
    COORDINATOR = "coordinator"
    PARTICIPANT = "participant"
    OBSERVER = "observer"
    SPECIALIST = "specialist"
    BACKUP = "backup"

class MessageType(Enum):
    """Types of cross-swarm messages"""
    KNOWLEDGE_SHARE = "knowledge_share"
    COORDINATION_REQUEST = "coordination_request"
    PERFORMANCE_REPORT = "performance_report"
    RESOURCE_REQUEST = "resource_request"
    EMERGENCY_SIGNAL = "emergency_signal"
    CONSENSUS_VOTE = "consensus_vote"
    PATTERN_DISCOVERY = "pattern_discovery"
    CAPABILITY_ANNOUNCEMENT = "capability_announcement"

@dataclass
class SwarmIdentity:
    """Identity and metadata for a swarm"""
    swarm_id: str
    name: str
    role: SwarmRole
    capabilities: List[str]
    specializations: List[str]
    performance_metrics: Dict[str, float]
    resource_capacity: Dict[str, float]
    current_load: Dict[str, float]
    reputation_score: float
    trust_level: float
    last_seen: datetime
    network_address: str
    public_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class KnowledgePacket:
    """Knowledge package for sharing between swarms"""
    packet_id: str
    source_swarm: str
    knowledge_type: str
    content: Dict[str, Any]
    relevance_score: float
    confidence: float
    creation_time: datetime
    expiry_time: Optional[datetime]
    dependencies: List[str]
    validation_signatures: Dict[str, str]
    access_permissions: List[str]
    usage_count: int = 0
    success_rate: float = 0.0

@dataclass
class CrossSwarmTask:
    """Task requiring coordination across multiple swarms"""
    task_id: str
    description: str
    required_capabilities: List[str]
    resource_requirements: Dict[str, float]
    priority: float
    deadline: Optional[datetime]
    assigned_swarms: List[str]
    progress: Dict[str, float]
    status: str
    coordination_pattern: str
    dependencies: List[str]
    results: Dict[str, Any] = field(default_factory=dict)

class DistributedConsensusProtocol:
    """Distributed consensus mechanism for cross-swarm decisions"""
    
    def __init__(self, consensus_threshold: float = 0.67):
        self.consensus_threshold = consensus_threshold
        self.active_proposals = {}
        self.voting_history = deque(maxlen=1000)
        self.reputation_scores = defaultdict(float)
        
    async def propose_decision(self, proposal_id: str, proposal: Dict[str, Any],
                             participating_swarms: List[str]) -> str:
        """Propose a decision for consensus voting"""
        
        proposal_data = {
            'proposal_id': proposal_id,
            'proposal': proposal,
            'participating_swarms': participating_swarms,
            'votes': {},
            'start_time': datetime.now(),
            'deadline': datetime.now() + timedelta(minutes=5),
            'status': 'voting',
            'vote_weights': {swarm: 1.0 for swarm in participating_swarms}
        }
        
        self.active_proposals[proposal_id] = proposal_data
        return proposal_id
    
    async def cast_vote(self, proposal_id: str, swarm_id: str, 
                       vote: bool, reasoning: str = "") -> bool:
        """Cast a vote on an active proposal"""
        
        if proposal_id not in self.active_proposals:
            return False
        
        proposal = self.active_proposals[proposal_id]
        
        if proposal['status'] != 'voting':
            return False
        
        if swarm_id not in proposal['participating_swarms']:
            return False
        
        # Weight vote by reputation
        weight = proposal['vote_weights'].get(swarm_id, 1.0)
        reputation_modifier = max(0.5, self.reputation_scores.get(swarm_id, 1.0))
        final_weight = weight * reputation_modifier
        
        proposal['votes'][swarm_id] = {
            'vote': vote,
            'weight': final_weight,
            'timestamp': datetime.now(),
            'reasoning': reasoning
        }
        
        # Check if consensus reached
        await self._check_consensus(proposal_id)
        return True
    
    async def _check_consensus(self, proposal_id: str):
        """Check if consensus has been reached"""
        proposal = self.active_proposals[proposal_id]
        
        total_weight = sum(vote_data['weight'] for vote_data in proposal['votes'].values())
        yes_weight = sum(vote_data['weight'] for vote_data in proposal['votes'].values() 
                        if vote_data['vote'])
        
        if total_weight == 0:
            return
        
        consensus_ratio = yes_weight / total_weight
        
        # Check if enough swarms have voted
        participation_ratio = len(proposal['votes']) / len(proposal['participating_swarms'])
        
        if (consensus_ratio >= self.consensus_threshold and participation_ratio >= 0.5) or \
           datetime.now() > proposal['deadline']:
            
            proposal['status'] = 'decided'
            proposal['result'] = consensus_ratio >= self.consensus_threshold
            proposal['final_consensus_ratio'] = consensus_ratio
            
            # Update reputation scores based on vote alignment with majority
            majority_vote = consensus_ratio >= 0.5
            for swarm_id, vote_data in proposal['votes'].items():
                if vote_data['vote'] == majority_vote:
                    self.reputation_scores[swarm_id] += 0.1
                else:
                    self.reputation_scores[swarm_id] -= 0.05
                
                # Keep reputation in bounds
                self.reputation_scores[swarm_id] = max(0.1, min(2.0, self.reputation_scores[swarm_id]))
            
            self.voting_history.append(proposal.copy())

class KnowledgeGraph:
    """Neural knowledge graph for cross-swarm intelligence"""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.knowledge_embeddings = {}
        self.swarm_embeddings = {}
        self.knowledge_graph = defaultdict(list)
        
        # Neural components for knowledge processing
        self.knowledge_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, embedding_dim)
        )
        
        self.similarity_network = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.relevance_predictor = nn.Sequential(
            nn.Linear(embedding_dim * 3, 256),  # Knowledge + Source + Target swarm
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def add_knowledge(self, packet: KnowledgePacket) -> torch.Tensor:
        """Add knowledge to the graph with neural encoding"""
        
        # Create embedding from knowledge content
        content_str = json.dumps(packet.content, sort_keys=True)
        content_hash = hashlib.sha256(content_str.encode()).hexdigest()
        
        # Generate pseudo-embedding (in practice, use proper text encoder)
        embedding = torch.randn(self.embedding_dim)
        
        # Encode through neural network
        with torch.no_grad():
            encoded_embedding = self.knowledge_encoder(embedding.unsqueeze(0)).squeeze(0)
        
        self.knowledge_embeddings[packet.packet_id] = encoded_embedding
        
        # Add to graph structure
        self.knowledge_graph[packet.source_swarm].append(packet.packet_id)
        
        return encoded_embedding
    
    def find_relevant_knowledge(self, query_embedding: torch.Tensor,
                              source_swarm: str, target_swarm: str,
                              top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most relevant knowledge for a query"""
        
        if not self.knowledge_embeddings:
            return []
        
        similarities = []
        source_embedding = self.swarm_embeddings.get(source_swarm, torch.zeros(self.embedding_dim))
        target_embedding = self.swarm_embeddings.get(target_swarm, torch.zeros(self.embedding_dim))
        
        for knowledge_id, knowledge_embedding in self.knowledge_embeddings.items():
            # Calculate similarity
            combined_input = torch.cat([query_embedding, knowledge_embedding])
            with torch.no_grad():
                similarity = self.similarity_network(combined_input.unsqueeze(0)).item()
            
            # Calculate relevance
            relevance_input = torch.cat([knowledge_embedding, source_embedding, target_embedding])
            with torch.no_grad():
                relevance = self.relevance_predictor(relevance_input.unsqueeze(0)).item()
            
            # Combined score
            score = similarity * 0.6 + relevance * 0.4
            similarities.append((knowledge_id, score))
        
        # Return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def update_swarm_embedding(self, swarm_id: str, capabilities: List[str],
                             performance_metrics: Dict[str, float]):
        """Update embedding for a swarm based on its characteristics"""
        
        # Create embedding from swarm characteristics
        feature_vector = []
        
        # Capability features (simplified binary encoding)
        capability_features = [1.0 if cap in capabilities else 0.0 
                             for cap in ['research', 'coding', 'analysis', 'optimization', 'coordination']]
        feature_vector.extend(capability_features)
        
        # Performance features
        performance_features = [
            performance_metrics.get('accuracy', 0.5),
            performance_metrics.get('speed', 0.5),
            performance_metrics.get('reliability', 0.5),
            performance_metrics.get('efficiency', 0.5)
        ]
        feature_vector.extend(performance_features)
        
        # Pad to embedding dimension
        while len(feature_vector) < self.embedding_dim:
            feature_vector.append(0.0)
        
        self.swarm_embeddings[swarm_id] = torch.tensor(feature_vector[:self.embedding_dim])

class SwarmCommunicationProtocol:
    """Protocol for secure communication between swarms"""
    
    def __init__(self, swarm_identity: SwarmIdentity):
        self.identity = swarm_identity
        self.connections = {}
        self.message_queue = asyncio.Queue()
        self.websocket_server = None
        self.zmq_context = None
        self.zmq_socket = None
        
    async def start_communication(self, port: int = 8765):
        """Start communication services"""
        
        # Start WebSocket server for HTTP-based communication
        await self._start_websocket_server(port)
        
        # Start ZMQ socket for high-performance messaging
        await self._start_zmq_communication(port + 1000)
        
        logger.info(f"Swarm {self.identity.swarm_id} communication started on ports {port} and {port + 1000}")
    
    async def _start_websocket_server(self, port: int):
        """Start WebSocket server"""
        async def handle_websocket(websocket, path):
            try:
                async for message in websocket:
                    await self._handle_message(json.loads(message), websocket)
            except websockets.exceptions.ConnectionClosed:
                pass
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
        
        self.websocket_server = await websockets.serve(handle_websocket, "localhost", port)
    
    async def _start_zmq_communication(self, port: int):
        """Start ZMQ communication for high-performance messaging"""
        self.zmq_context = zmq.asyncio.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.REP)
        self.zmq_socket.bind(f"tcp://*:{port}")
        
        # Start ZMQ message handling task
        asyncio.create_task(self._handle_zmq_messages())
    
    async def _handle_zmq_messages(self):
        """Handle incoming ZMQ messages"""
        while True:
            try:
                message = await self.zmq_socket.recv_json()
                response = await self._process_message(message)
                await self.zmq_socket.send_json(response)
            except Exception as e:
                logger.error(f"ZMQ message handling error: {e}")
                await asyncio.sleep(1)
    
    async def send_message(self, target_swarm: str, message_type: MessageType,
                          content: Dict[str, Any], priority: str = "normal") -> bool:
        """Send message to another swarm"""
        
        message = {
            'message_id': hashlib.sha256(f"{datetime.now().isoformat()}{target_swarm}".encode()).hexdigest()[:16],
            'source_swarm': self.identity.swarm_id,
            'target_swarm': target_swarm,
            'message_type': message_type.value,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'priority': priority,
            'signature': self._sign_message(content)
        }
        
        try:
            # Try ZMQ first for performance
            if target_swarm in self.connections and 'zmq_address' in self.connections[target_swarm]:
                return await self._send_zmq_message(target_swarm, message)
            
            # Fallback to WebSocket
            return await self._send_websocket_message(target_swarm, message)
            
        except Exception as e:
            logger.error(f"Failed to send message to {target_swarm}: {e}")
            return False
    
    async def _send_zmq_message(self, target_swarm: str, message: Dict[str, Any]) -> bool:
        """Send message via ZMQ"""
        try:
            zmq_address = self.connections[target_swarm]['zmq_address']
            
            # Create temporary ZMQ socket for sending
            temp_socket = self.zmq_context.socket(zmq.REQ)
            temp_socket.connect(zmq_address)
            
            await temp_socket.send_json(message)
            response = await temp_socket.recv_json()
            
            temp_socket.close()
            return response.get('status') == 'success'
            
        except Exception as e:
            logger.error(f"ZMQ send error: {e}")
            return False
    
    async def _send_websocket_message(self, target_swarm: str, message: Dict[str, Any]) -> bool:
        """Send message via WebSocket"""
        try:
            if target_swarm not in self.connections:
                return False
            
            websocket_uri = self.connections[target_swarm]['websocket_uri']
            
            async with websockets.connect(websocket_uri) as websocket:
                await websocket.send(json.dumps(message))
                response = await websocket.recv()
                response_data = json.loads(response)
                return response_data.get('status') == 'success'
                
        except Exception as e:
            logger.error(f"WebSocket send error: {e}")
            return False
    
    async def _handle_message(self, message: Dict[str, Any], websocket=None) -> Dict[str, Any]:
        """Handle incoming message"""
        try:
            # Validate message signature
            if not self._verify_message(message):
                return {'status': 'error', 'message': 'invalid_signature'}
            
            response = await self._process_message(message)
            
            if websocket:
                await websocket.send(json.dumps(response))
            
            return response
            
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            return {'status': 'error', 'message': str(e)}
    
    async def _process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process message based on type"""
        
        message_type = MessageType(message['message_type'])
        content = message['content']
        
        if message_type == MessageType.KNOWLEDGE_SHARE:
            return await self._handle_knowledge_share(content)
        elif message_type == MessageType.COORDINATION_REQUEST:
            return await self._handle_coordination_request(content)
        elif message_type == MessageType.PERFORMANCE_REPORT:
            return await self._handle_performance_report(content)
        elif message_type == MessageType.RESOURCE_REQUEST:
            return await self._handle_resource_request(content)
        elif message_type == MessageType.EMERGENCY_SIGNAL:
            return await self._handle_emergency_signal(content)
        else:
            return {'status': 'error', 'message': 'unknown_message_type'}
    
    async def _handle_knowledge_share(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Handle knowledge sharing message"""
        # Implementation depends on integration with knowledge graph
        return {'status': 'success', 'message': 'knowledge_received'}
    
    async def _handle_coordination_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Handle coordination request"""
        # Implementation depends on coordination strategy
        return {'status': 'success', 'message': 'coordination_acknowledged'}
    
    async def _handle_performance_report(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Handle performance report"""
        # Update local knowledge about reporting swarm
        return {'status': 'success', 'message': 'performance_recorded'}
    
    async def _handle_resource_request(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Handle resource request"""
        # Evaluate ability to provide resources
        return {'status': 'success', 'message': 'resource_evaluated'}
    
    async def _handle_emergency_signal(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Handle emergency signal"""
        # Priority handling for emergencies
        logger.warning(f"Emergency signal received: {content}")
        return {'status': 'success', 'message': 'emergency_acknowledged'}
    
    def _sign_message(self, content: Dict[str, Any]) -> str:
        """Create message signature (simplified)"""
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _verify_message(self, message: Dict[str, Any]) -> bool:
        """Verify message signature (simplified)"""
        expected_signature = self._sign_message(message['content'])
        return message.get('signature') == expected_signature
    
    async def stop_communication(self):
        """Stop communication services"""
        if self.websocket_server:
            self.websocket_server.close()
            await self.websocket_server.wait_closed()
        
        if self.zmq_socket:
            self.zmq_socket.close()
        
        if self.zmq_context:
            self.zmq_context.term()

class CrossSwarmIntelligenceCoordinator:
    """Main coordinator for cross-swarm intelligence and communication"""
    
    def __init__(self, swarm_identity: SwarmIdentity, config: Dict[str, Any]):
        self.identity = swarm_identity
        self.config = config
        
        # Core components
        self.knowledge_graph = KnowledgeGraph(config.get('embedding_dim', 512))
        self.consensus_protocol = DistributedConsensusProtocol(
            config.get('consensus_threshold', 0.67)
        )
        self.communication = SwarmCommunicationProtocol(swarm_identity)
        
        # Swarm registry
        self.known_swarms: Dict[str, SwarmIdentity] = {}
        self.active_connections: Set[str] = set()
        
        # Task coordination
        self.active_tasks: Dict[str, CrossSwarmTask] = {}
        self.task_queue = asyncio.Queue()
        
        # Performance tracking
        self.collaboration_history = deque(maxlen=1000)
        self.knowledge_sharing_stats = defaultdict(int)
        self.reputation_system = defaultdict(float)
        
        # Neural components for intelligent coordination
        self.coordination_network = self._build_coordination_network()
        
    def _build_coordination_network(self) -> nn.Module:
        """Build neural network for intelligent coordination decisions"""
        
        class CoordinationNetwork(nn.Module):
            def __init__(self, input_dim: int = 256):
                super().__init__()
                
                self.feature_encoder = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.BatchNorm1d(512),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.GELU(),
                    nn.Dropout(0.1)
                )
                
                # Multiple decision heads
                self.coordination_decision = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.GELU(),
                    nn.Linear(128, 5),  # 5 coordination strategies
                    nn.Softmax(dim=-1)
                )
                
                self.resource_allocation = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.GELU(),
                    nn.Linear(128, 10),  # Resource allocation weights
                    nn.Softmax(dim=-1)
                )
                
                self.priority_assessment = nn.Sequential(
                    nn.Linear(256, 64),
                    nn.GELU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, context_features: torch.Tensor) -> Dict[str, torch.Tensor]:
                encoded = self.feature_encoder(context_features)
                
                return {
                    'coordination_strategy': self.coordination_decision(encoded),
                    'resource_allocation': self.resource_allocation(encoded),
                    'priority_score': self.priority_assessment(encoded)
                }
        
        return CoordinationNetwork(self.config.get('coordination_input_dim', 256))
    
    async def start_coordination(self, port: int = 8765):
        """Start the cross-swarm intelligence coordinator"""
        
        # Start communication services
        await self.communication.start_communication(port)
        
        # Start background tasks
        asyncio.create_task(self._discovery_task())
        asyncio.create_task(self._knowledge_sharing_task())
        asyncio.create_task(self._coordination_task())
        asyncio.create_task(self._maintenance_task())
        
        logger.info(f"Cross-swarm coordinator started for {self.identity.swarm_id}")
    
    async def register_swarm(self, swarm_identity: SwarmIdentity):
        """Register a new swarm for coordination"""
        
        self.known_swarms[swarm_identity.swarm_id] = swarm_identity
        
        # Update knowledge graph with swarm information
        self.knowledge_graph.update_swarm_embedding(
            swarm_identity.swarm_id,
            swarm_identity.capabilities,
            swarm_identity.performance_metrics
        )
        
        # Establish connection
        await self._establish_connection(swarm_identity)
        
        logger.info(f"Registered swarm: {swarm_identity.swarm_id}")
    
    async def share_knowledge(self, knowledge_type: str, content: Dict[str, Any],
                            target_swarms: Optional[List[str]] = None,
                            relevance_threshold: float = 0.7) -> Dict[str, bool]:
        """Share knowledge with other swarms"""
        
        # Create knowledge packet
        packet = KnowledgePacket(
            packet_id=hashlib.sha256(f"{knowledge_type}{datetime.now().isoformat()}".encode()).hexdigest()[:16],
            source_swarm=self.identity.swarm_id,
            knowledge_type=knowledge_type,
            content=content,
            relevance_score=1.0,  # Self-assessed
            confidence=0.9,       # Default confidence
            creation_time=datetime.now(),
            expiry_time=datetime.now() + timedelta(hours=24),
            dependencies=[],
            validation_signatures={},
            access_permissions=['public']
        )
        
        # Add to knowledge graph
        self.knowledge_graph.add_knowledge(packet)
        
        # Determine target swarms if not specified
        if target_swarms is None:
            target_swarms = await self._find_relevant_swarms(packet, relevance_threshold)
        
        # Share knowledge
        sharing_results = {}
        for target_swarm in target_swarms:
            if target_swarm in self.active_connections:
                success = await self.communication.send_message(
                    target_swarm,
                    MessageType.KNOWLEDGE_SHARE,
                    {'knowledge_packet': packet.__dict__}
                )
                sharing_results[target_swarm] = success
                
                if success:
                    self.knowledge_sharing_stats[target_swarm] += 1
        
        return sharing_results
    
    async def request_coordination(self, task: CrossSwarmTask) -> Dict[str, Any]:
        """Request coordination for a cross-swarm task"""
        
        # Find suitable swarms for the task
        suitable_swarms = await self._find_suitable_swarms(task)
        
        if not suitable_swarms:
            return {'status': 'error', 'message': 'no_suitable_swarms'}
        
        # Create coordination context
        context = self._create_coordination_context(task, suitable_swarms)
        
        # Use neural network to determine coordination strategy
        coordination_decision = await self._make_coordination_decision(context)
        
        # Send coordination requests
        coordination_responses = {}
        for swarm_id in suitable_swarms[:coordination_decision['max_swarms']]:
            response = await self.communication.send_message(
                swarm_id,
                MessageType.COORDINATION_REQUEST,
                {
                    'task': task.__dict__,
                    'coordination_strategy': coordination_decision['strategy'],
                    'resource_allocation': coordination_decision['resource_allocation'],
                    'priority': coordination_decision['priority']
                }
            )
            coordination_responses[swarm_id] = response
        
        # Update active tasks
        self.active_tasks[task.task_id] = task
        
        return {
            'status': 'success',
            'task_id': task.task_id,
            'participating_swarms': list(coordination_responses.keys()),
            'coordination_strategy': coordination_decision,
            'responses': coordination_responses
        }
    
    async def propose_consensus_decision(self, proposal: Dict[str, Any],
                                       participating_swarms: Optional[List[str]] = None) -> str:
        """Propose a decision for cross-swarm consensus"""
        
        if participating_swarms is None:
            participating_swarms = list(self.active_connections)
        
        proposal_id = await self.consensus_protocol.propose_decision(
            proposal_id=hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest()[:16],
            proposal=proposal,
            participating_swarms=participating_swarms
        )
        
        # Notify participating swarms
        for swarm_id in participating_swarms:
            await self.communication.send_message(
                swarm_id,
                MessageType.CONSENSUS_VOTE,
                {
                    'proposal_id': proposal_id,
                    'proposal': proposal,
                    'voting_deadline': (datetime.now() + timedelta(minutes=5)).isoformat()
                }
            )
        
        return proposal_id
    
    async def vote_on_proposal(self, proposal_id: str, vote: bool, reasoning: str = "") -> bool:
        """Vote on a consensus proposal"""
        return await self.consensus_protocol.cast_vote(
            proposal_id, self.identity.swarm_id, vote, reasoning
        )
    
    async def discover_patterns(self, domain: str, min_confidence: float = 0.8) -> List[Dict[str, Any]]:
        """Discover patterns across swarm knowledge and experiences"""
        
        patterns = []
        
        # Analyze knowledge graph for patterns
        domain_knowledge = []
        for packet_id, embedding in self.knowledge_graph.knowledge_embeddings.items():
            # This is simplified - would need proper pattern detection
            patterns.append({
                'pattern_id': f"pattern_{len(patterns)}",
                'domain': domain,
                'confidence': min_confidence + 0.1,
                'description': f"Pattern discovered in {domain}",
                'supporting_knowledge': [packet_id],
                'participating_swarms': list(self.known_swarms.keys()),
                'pattern_type': 'knowledge_clustering'
            })
        
        # Share discovered patterns
        if patterns:
            for swarm_id in self.active_connections:
                await self.communication.send_message(
                    swarm_id,
                    MessageType.PATTERN_DISCOVERY,
                    {'patterns': patterns, 'domain': domain}
                )
        
        return patterns
    
    async def emergency_coordination(self, emergency_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate emergency response across swarms"""
        
        # Broadcast emergency signal
        emergency_responses = {}
        for swarm_id in self.active_connections:
            response = await self.communication.send_message(
                swarm_id,
                MessageType.EMERGENCY_SIGNAL,
                {
                    'emergency_type': emergency_type,
                    'context': context,
                    'source_swarm': self.identity.swarm_id,
                    'timestamp': datetime.now().isoformat(),
                    'severity': context.get('severity', 'high')
                },
                priority="critical"
            )
            emergency_responses[swarm_id] = response
        
        # Create emergency coordination task
        emergency_task = CrossSwarmTask(
            task_id=f"emergency_{hashlib.sha256(emergency_type.encode()).hexdigest()[:8]}",
            description=f"Emergency coordination: {emergency_type}",
            required_capabilities=['emergency_response'],
            resource_requirements={'priority': 1.0, 'urgency': 1.0},
            priority=1.0,
            deadline=datetime.now() + timedelta(minutes=30),
            assigned_swarms=list(self.active_connections),
            progress={},
            status='emergency',
            coordination_pattern='all_hands',
            dependencies=[]
        )
        
        return {
            'emergency_task': emergency_task,
            'responses': emergency_responses,
            'coordination_status': 'initiated'
        }
    
    async def _discovery_task(self):
        """Background task for swarm discovery"""
        while True:
            try:
                # Implement swarm discovery mechanism
                # This could involve multicast, DHT, or service discovery
                await asyncio.sleep(30)  # Discovery interval
                
            except Exception as e:
                logger.error(f"Discovery task error: {e}")
                await asyncio.sleep(60)
    
    async def _knowledge_sharing_task(self):
        """Background task for proactive knowledge sharing"""
        while True:
            try:
                # Periodically share valuable knowledge
                if len(self.knowledge_graph.knowledge_embeddings) > 0:
                    # Select knowledge to share based on value and relevance
                    await self._proactive_knowledge_sharing()
                
                await asyncio.sleep(120)  # Sharing interval
                
            except Exception as e:
                logger.error(f"Knowledge sharing task error: {e}")
                await asyncio.sleep(60)
    
    async def _coordination_task(self):
        """Background task for handling coordination requests"""
        while True:
            try:
                # Process queued coordination tasks
                if not self.task_queue.empty():
                    task = await self.task_queue.get()
                    await self._process_coordination_task(task)
                
                await asyncio.sleep(1)  # Fast processing
                
            except Exception as e:
                logger.error(f"Coordination task error: {e}")
                await asyncio.sleep(10)
    
    async def _maintenance_task(self):
        """Background task for system maintenance"""
        while True:
            try:
                # Clean up expired knowledge
                await self._cleanup_expired_knowledge()
                
                # Update swarm reputation scores
                await self._update_reputation_scores()
                
                # Optimize knowledge graph
                await self._optimize_knowledge_graph()
                
                await asyncio.sleep(300)  # Maintenance interval
                
            except Exception as e:
                logger.error(f"Maintenance task error: {e}")
                await asyncio.sleep(60)
    
    async def _find_relevant_swarms(self, packet: KnowledgePacket, 
                                  threshold: float) -> List[str]:
        """Find swarms most relevant for knowledge sharing"""
        
        relevant_swarms = []
        packet_embedding = self.knowledge_graph.knowledge_embeddings.get(packet.packet_id)
        
        if packet_embedding is None:
            return []
        
        for swarm_id, swarm_embedding in self.knowledge_graph.swarm_embeddings.items():
            if swarm_id == self.identity.swarm_id:
                continue
            
            # Calculate relevance
            relevance_scores = self.knowledge_graph.find_relevant_knowledge(
                packet_embedding, self.identity.swarm_id, swarm_id, 1
            )
            
            if relevance_scores and relevance_scores[0][1] > threshold:
                relevant_swarms.append(swarm_id)
        
        return relevant_swarms
    
    async def _find_suitable_swarms(self, task: CrossSwarmTask) -> List[str]:
        """Find swarms suitable for a coordination task"""
        
        suitable_swarms = []
        
        for swarm_id, swarm_identity in self.known_swarms.items():
            if swarm_id == self.identity.swarm_id:
                continue
            
            # Check capability match
            capability_match = any(cap in swarm_identity.capabilities 
                                 for cap in task.required_capabilities)
            
            # Check resource availability
            resource_available = all(
                swarm_identity.current_load.get(resource, 1.0) < 0.8
                for resource in task.resource_requirements.keys()
            )
            
            # Check reputation
            reputation_ok = self.reputation_system.get(swarm_id, 1.0) > 0.5
            
            if capability_match and resource_available and reputation_ok:
                suitable_swarms.append(swarm_id)
        
        return suitable_swarms
    
    def _create_coordination_context(self, task: CrossSwarmTask, 
                                   suitable_swarms: List[str]) -> torch.Tensor:
        """Create context tensor for coordination decision making"""
        
        context_features = []
        
        # Task features
        context_features.extend([
            task.priority,
            len(task.required_capabilities),
            len(suitable_swarms),
            (task.deadline - datetime.now()).total_seconds() / 3600.0 if task.deadline else 24.0
        ])
        
        # Swarm capability aggregation
        total_capabilities = set()
        avg_performance = 0.0
        avg_load = 0.0
        
        for swarm_id in suitable_swarms:
            swarm = self.known_swarms.get(swarm_id)
            if swarm:
                total_capabilities.update(swarm.capabilities)
                avg_performance += swarm.performance_metrics.get('accuracy', 0.5)
                avg_load += np.mean(list(swarm.current_load.values()))
        
        num_swarms = len(suitable_swarms)
        if num_swarms > 0:
            avg_performance /= num_swarms
            avg_load /= num_swarms
        
        context_features.extend([
            len(total_capabilities),
            avg_performance,
            avg_load,
            num_swarms
        ])
        
        # Pad to fixed size
        target_size = self.config.get('coordination_input_dim', 256)
        while len(context_features) < target_size:
            context_features.append(0.0)
        
        return torch.tensor(context_features[:target_size], dtype=torch.float32).unsqueeze(0)
    
    async def _make_coordination_decision(self, context: torch.Tensor) -> Dict[str, Any]:
        """Make coordination decision using neural network"""
        
        with torch.no_grad():
            decision_output = self.coordination_network(context)
        
        # Extract decisions
        strategy_probs = decision_output['coordination_strategy'].squeeze().numpy()
        strategies = ['hierarchical', 'peer_to_peer', 'hybrid', 'consensus', 'emergency']
        selected_strategy = strategies[np.argmax(strategy_probs)]
        
        resource_allocation = decision_output['resource_allocation'].squeeze().numpy()
        priority_score = decision_output['priority_score'].item()
        
        return {
            'strategy': selected_strategy,
            'resource_allocation': resource_allocation.tolist(),
            'priority': priority_score,
            'max_swarms': min(8, int(priority_score * 10) + 2)  # Scale based on priority
        }
    
    async def _establish_connection(self, swarm_identity: SwarmIdentity):
        """Establish connection with a swarm"""
        
        try:
            # Add connection info
            self.communication.connections[swarm_identity.swarm_id] = {
                'websocket_uri': f"ws://{swarm_identity.network_address}:8765",
                'zmq_address': f"tcp://{swarm_identity.network_address}:9765"
            }
            
            self.active_connections.add(swarm_identity.swarm_id)
            
            # Send capability announcement
            await self.communication.send_message(
                swarm_identity.swarm_id,
                MessageType.CAPABILITY_ANNOUNCEMENT,
                {
                    'capabilities': self.identity.capabilities,
                    'specializations': self.identity.specializations,
                    'performance_metrics': self.identity.performance_metrics,
                    'resource_capacity': self.identity.resource_capacity
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to establish connection with {swarm_identity.swarm_id}: {e}")
    
    async def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status"""
        return {
            'swarm_identity': self.identity.__dict__,
            'known_swarms': len(self.known_swarms),
            'active_connections': len(self.active_connections),
            'active_tasks': len(self.active_tasks),
            'knowledge_packets': len(self.knowledge_graph.knowledge_embeddings),
            'reputation_scores': dict(self.reputation_system),
            'knowledge_sharing_stats': dict(self.knowledge_sharing_stats),
            'collaboration_history_size': len(self.collaboration_history),
            'consensus_proposals_active': len(self.consensus_protocol.active_proposals)
        }
    
    async def stop_coordination(self):
        """Stop coordination services"""
        await self.communication.stop_communication()
        logger.info(f"Cross-swarm coordination stopped for {self.identity.swarm_id}")

# Factory function
def create_cross_swarm_coordinator(swarm_id: str, capabilities: List[str], 
                                 config: Optional[Dict[str, Any]] = None) -> CrossSwarmIntelligenceCoordinator:
    """Create cross-swarm intelligence coordinator"""
    
    default_config = {
        'embedding_dim': 512,
        'consensus_threshold': 0.67,
        'coordination_input_dim': 256
    }
    
    if config:
        default_config.update(config)
    
    # Create swarm identity
    swarm_identity = SwarmIdentity(
        swarm_id=swarm_id,
        name=f"Swarm-{swarm_id}",
        role=SwarmRole.PARTICIPANT,
        capabilities=capabilities,
        specializations=[],
        performance_metrics={'accuracy': 0.887, 'speed': 0.8, 'reliability': 0.9},
        resource_capacity={'cpu': 1.0, 'memory': 1.0, 'network': 1.0},
        current_load={'cpu': 0.3, 'memory': 0.4, 'network': 0.2},
        reputation_score=1.0,
        trust_level=1.0,
        last_seen=datetime.now(),
        network_address='localhost'
    )
    
    return CrossSwarmIntelligenceCoordinator(swarm_identity, default_config)

if __name__ == "__main__":
    # Example usage and testing
    async def test_cross_swarm_intelligence():
        """Test the cross-swarm intelligence system"""
        
        # Create two coordinators for testing
        coordinator1 = create_cross_swarm_coordinator(
            'swarm_neural_01', 
            ['neural_coordination', 'transformer_processing', 'ensemble_methods']
        )
        
        coordinator2 = create_cross_swarm_coordinator(
            'swarm_neural_02',
            ['predictive_scaling', 'performance_monitoring', 'resource_optimization']
        )
        
        # Start coordination services
        await coordinator1.start_coordination(8765)
        await coordinator2.start_coordination(8766)
        
        # Register swarms with each other
        await coordinator1.register_swarm(coordinator2.identity)
        await coordinator2.register_swarm(coordinator1.identity)
        
        # Test knowledge sharing
        knowledge_content = {
            'neural_pattern': 'transformer_attention_optimization',
            'performance_improvement': '15%',
            'resource_efficiency': '20%',
            'implementation_details': {
                'attention_heads': 8,
                'hidden_dim': 512,
                'optimization_strategy': 'gradient_accumulation'
            }
        }
        
        sharing_result = await coordinator1.share_knowledge(
            'neural_optimization',
            knowledge_content
        )
        
        print(f"Knowledge sharing result: {sharing_result}")
        
        # Test coordination request
        coordination_task = CrossSwarmTask(
            task_id='neural_ensemble_optimization',
            description='Optimize neural ensemble for improved accuracy',
            required_capabilities=['neural_coordination', 'ensemble_methods'],
            resource_requirements={'cpu': 0.8, 'memory': 0.6},
            priority=0.9,
            deadline=datetime.now() + timedelta(hours=2),
            assigned_swarms=[],
            progress={},
            status='pending',
            coordination_pattern='hybrid',
            dependencies=[]
        )
        
        coordination_result = await coordinator1.request_coordination(coordination_task)
        print(f"Coordination result: {coordination_result}")
        
        # Test consensus decision
        proposal = {
            'decision_type': 'neural_architecture_update',
            'proposed_changes': {
                'increase_attention_heads': True,
                'add_residual_connections': True,
                'optimize_memory_usage': True
            },
            'expected_improvement': 0.15
        }
        
        proposal_id = await coordinator1.propose_consensus_decision(proposal)
        
        # Vote on proposal
        await coordinator2.vote_on_proposal(proposal_id, True, "Improvements look beneficial")
        
        # Get status
        status1 = await coordinator1.get_coordination_status()
        status2 = await coordinator2.get_coordination_status()
        
        print(f"\nCoordinator 1 Status:")
        print(f"Known swarms: {status1['known_swarms']}")
        print(f"Active connections: {status1['active_connections']}")
        print(f"Knowledge packets: {status1['knowledge_packets']}")
        
        print(f"\nCoordinator 2 Status:")
        print(f"Known swarms: {status2['known_swarms']}")
        print(f"Active connections: {status2['active_connections']}")
        print(f"Knowledge packets: {status2['knowledge_packets']}")
        
        # Cleanup
        await coordinator1.stop_coordination()
        await coordinator2.stop_coordination()
        
        return sharing_result, coordination_result, status1, status2
    
    # Run test if executed directly
    import asyncio
    asyncio.run(test_cross_swarm_intelligence())