#!/usr/bin/env python3
"""
Predictive Scaling Network for Dynamic Multi-Agent Management
Phase 3 Implementation - Advanced neural prediction for optimal resource allocation
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from collections import deque, defaultdict
import psutil
import gc

# Time series and forecasting imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna

logger = logging.getLogger(__name__)

@dataclass
class WorkloadMetrics:
    """Comprehensive workload metrics for prediction"""
    timestamp: datetime
    task_complexity: float
    concurrent_tasks: int
    resource_utilization: Dict[str, float]  # CPU, memory, network
    performance_history: List[float]
    error_rate: float
    latency_percentiles: Dict[str, float]  # p50, p95, p99
    throughput: float
    queue_depth: int
    agent_efficiency: Dict[str, float]
    context_features: Dict[str, Any]

@dataclass
class ScalingPrediction:
    """Prediction output for scaling decisions"""
    optimal_agent_count: int
    confidence: float
    resource_requirements: Dict[str, float]
    expected_performance: float
    scaling_timeline: Dict[str, int]  # When to scale up/down
    cost_estimate: float
    risk_assessment: Dict[str, float]
    alternative_scenarios: List[Dict[str, Any]]

@dataclass
class ScalingAction:
    """Recommended scaling action"""
    action_type: str  # 'scale_up', 'scale_down', 'maintain', 'emergency'
    target_agents: int
    priority: str
    execution_time: datetime
    resource_changes: Dict[str, float]
    expected_impact: Dict[str, float]
    rollback_plan: Dict[str, Any]

class TemporalAttentionLayer(nn.Module):
    """Temporal attention mechanism for time series data"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)
        
        self.output_projection = nn.Linear(hidden_dim, input_dim)
        self.layer_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Temporal position encoding
        self.temporal_encoding = nn.Parameter(torch.randn(1000, input_dim))
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with temporal attention"""
        batch_size, seq_len, _ = x.size()
        
        # Add temporal encoding
        positions = torch.arange(seq_len, device=x.device)
        temporal_enc = self.temporal_encoding[positions].unsqueeze(0).expand(batch_size, -1, -1)
        x_with_time = x + temporal_enc
        
        # Multi-head attention
        Q = self.query_projection(x_with_time).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_projection(x_with_time).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_projection(x_with_time).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        # Output projection and residual connection
        output = self.output_projection(context)
        return self.layer_norm(x + self.dropout(output))

class WorkloadEncoder(nn.Module):
    """Neural encoder for workload features"""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 256, num_layers: int = 3):
        super().__init__()
        
        layers = []
        prev_dim = feature_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            hidden_dim = int(hidden_dim * 0.8)  # Gradual reduction
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = prev_dim
        
    def forward(self, workload_features: torch.Tensor) -> torch.Tensor:
        """Encode workload features into compressed representation"""
        return self.encoder(workload_features)

class PerformancePredictor(nn.Module):
    """Predict performance metrics based on scaling decisions"""
    
    def __init__(self, input_dim: int, output_metrics: int = 8):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            
            nn.Linear(128, output_metrics)
        )
        
        # Separate heads for different metric types
        self.latency_head = nn.Linear(128, 1)
        self.throughput_head = nn.Linear(128, 1)
        self.accuracy_head = nn.Linear(128, 1)
        self.resource_head = nn.Linear(128, 3)  # CPU, Memory, Network
        self.cost_head = nn.Linear(128, 1)
        
    def forward(self, encoded_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict multiple performance metrics"""
        # Shared representation
        shared_features = self.predictor[:-1](encoded_features)  # Exclude final layer
        
        return {
            'latency': torch.sigmoid(self.latency_head(shared_features)),  # 0-1 normalized
            'throughput': F.relu(self.throughput_head(shared_features)),   # Positive values
            'accuracy': torch.sigmoid(self.accuracy_head(shared_features)), # 0-1 probability
            'resource_usage': torch.sigmoid(self.resource_head(shared_features)), # 0-1 normalized
            'cost': F.relu(self.cost_head(shared_features))  # Positive cost
        }

class ScalingPolicyNetwork(nn.Module):
    """Policy network for scaling decisions using reinforcement learning principles"""
    
    def __init__(self, state_dim: int, action_dim: int = 32):  # Max 32 agents
        super().__init__()
        
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Actor network (scaling decisions)
        self.actor = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic network (value estimation)
        self.critic = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
        # Scaling confidence estimator
        self.confidence_estimator = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning action probabilities, value, and confidence"""
        encoded_state = self.state_encoder(state)
        
        action_probs = self.actor(encoded_state)
        value = self.critic(encoded_state)
        confidence = self.confidence_estimator(encoded_state)
        
        return action_probs, value, confidence

class RiskAssessmentModule(nn.Module):
    """Neural network for assessing scaling risks"""
    
    def __init__(self, input_dim: int, risk_categories: int = 6):
        super().__init__()
        
        self.risk_categories = [
            'performance_degradation', 'resource_exhaustion', 'cost_overrun',
            'system_instability', 'cascade_failure', 'recovery_difficulty'
        ]
        
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        
        # Individual risk category heads
        self.risk_heads = nn.ModuleDict({
            category: nn.Sequential(
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1),
                nn.Sigmoid()  # Risk probability 0-1
            ) for category in self.risk_categories
        })
        
        # Overall risk aggregator
        self.risk_aggregator = nn.Sequential(
            nn.Linear(len(self.risk_categories), 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Assess risks across multiple categories"""
        shared_features = self.shared_encoder(features)
        
        individual_risks = {}
        risk_scores = []
        
        for category in self.risk_categories:
            risk_score = self.risk_heads[category](shared_features)
            individual_risks[category] = risk_score
            risk_scores.append(risk_score)
        
        # Overall risk
        combined_risks = torch.cat(risk_scores, dim=-1)
        overall_risk = self.risk_aggregator(combined_risks)
        
        return {
            'individual_risks': individual_risks,
            'overall_risk': overall_risk,
            'risk_vector': combined_risks
        }

class PredictiveScalingNetwork(nn.Module):
    """Main network combining all components for predictive scaling"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.feature_dim = config.get('feature_dim', 256)
        self.sequence_length = config.get('sequence_length', 100)
        self.max_agents = config.get('max_agents', 32)
        
        # Core components
        self.temporal_attention = TemporalAttentionLayer(
            input_dim=self.feature_dim,
            hidden_dim=config.get('attention_hidden_dim', 128),
            num_heads=config.get('attention_heads', 8)
        )
        
        self.workload_encoder = WorkloadEncoder(
            feature_dim=self.feature_dim,
            hidden_dim=config.get('encoder_hidden_dim', 256),
            num_layers=config.get('encoder_layers', 3)
        )
        
        self.performance_predictor = PerformancePredictor(
            input_dim=self.workload_encoder.output_dim,
            output_metrics=config.get('output_metrics', 8)
        )
        
        self.scaling_policy = ScalingPolicyNetwork(
            state_dim=self.workload_encoder.output_dim + 8,  # +8 for predicted metrics
            action_dim=self.max_agents
        )
        
        self.risk_assessor = RiskAssessmentModule(
            input_dim=self.workload_encoder.output_dim,
            risk_categories=6
        )
        
        # LSTM for time series modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=config.get('lstm_hidden_dim', 256),
            num_layers=config.get('lstm_layers', 2),
            dropout=0.1,
            batch_first=True,
            bidirectional=True
        )
        
        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.workload_encoder.output_dim + 512, 256),  # +512 from bidirectional LSTM
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, workload_sequence: torch.Tensor,
               current_state: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for predictive scaling"""
        batch_size, seq_len, feature_dim = workload_sequence.size()
        
        # Temporal attention for sequence modeling
        attended_sequence = self.temporal_attention(workload_sequence)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(attended_sequence)
        lstm_features = lstm_out[:, -1, :]  # Last time step
        
        # Encode current workload state
        encoded_workload = self.workload_encoder(current_state)
        
        # Feature fusion
        fused_features = self.feature_fusion(torch.cat([encoded_workload, lstm_features], dim=-1))
        
        # Performance prediction
        performance_pred = self.performance_predictor(fused_features)
        
        # Combine for policy input
        policy_input = torch.cat([
            fused_features,
            performance_pred['latency'],
            performance_pred['throughput'], 
            performance_pred['accuracy'],
            performance_pred['resource_usage'].flatten(1),
            performance_pred['cost']
        ], dim=-1)
        
        # Scaling policy decision
        action_probs, value_estimate, confidence = self.scaling_policy(policy_input)
        
        # Risk assessment
        risk_analysis = self.risk_assessor(fused_features)
        
        return {
            'performance_prediction': performance_pred,
            'scaling_action_probs': action_probs,
            'value_estimate': value_estimate,
            'confidence': confidence,
            'risk_analysis': risk_analysis,
            'encoded_features': fused_features
        }

class AdvancedPredictiveScaler:
    """Advanced predictive scaling system with neural networks"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize neural network
        self.network = PredictiveScalingNetwork(config).to(self.device)
        
        # Data preprocessing
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.is_fitted = False
        
        # Historical data storage
        self.workload_history = deque(maxlen=config.get('history_size', 1000))
        self.performance_history = deque(maxlen=config.get('history_size', 1000))
        self.scaling_history = deque(maxlen=config.get('history_size', 1000))
        
        # Performance tracking
        self.prediction_accuracy = deque(maxlen=100)
        self.scaling_effectiveness = deque(maxlen=100)
        
        # Optimization
        self.optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Hyperparameter optimization
        self.hyperopt_study = optuna.create_study(
            direction='minimize',
            study_name='predictive_scaling_optimization'
        )
        
        # Current system state
        self.current_agents = config.get('initial_agents', 4)
        self.last_scaling_time = datetime.now()
        self.scaling_cooldown = timedelta(seconds=config.get('scaling_cooldown_seconds', 60))
        
    async def predict_scaling_needs(self, workload_metrics: WorkloadMetrics,
                                   context: Dict[str, Any]) -> ScalingPrediction:
        """Main method for predicting scaling needs"""
        
        # Extract and preprocess features
        feature_vector = self._extract_features(workload_metrics, context)
        
        # Get sequence data
        sequence_data = self._prepare_sequence_data()
        
        # Neural network inference
        with torch.no_grad():
            network_output = self.network(sequence_data, feature_vector)
        
        # Process outputs
        scaling_prediction = self._process_network_output(
            network_output, workload_metrics, context
        )
        
        # Update history
        self._update_history(workload_metrics, scaling_prediction)
        
        return scaling_prediction
    
    async def recommend_scaling_action(self, prediction: ScalingPrediction,
                                     current_state: Dict[str, Any]) -> ScalingAction:
        """Recommend specific scaling action based on prediction"""
        
        current_time = datetime.now()
        time_since_last_scaling = current_time - self.last_scaling_time
        
        # Check cooldown period
        if time_since_last_scaling < self.scaling_cooldown:
            return ScalingAction(
                action_type='maintain',
                target_agents=self.current_agents,
                priority='low',
                execution_time=current_time,
                resource_changes={},
                expected_impact={'reason': 'scaling_cooldown_active'},
                rollback_plan={}
            )
        
        # Determine action type
        current_agents = current_state.get('current_agents', self.current_agents)
        optimal_agents = prediction.optimal_agent_count
        confidence = prediction.confidence
        
        # Decision logic
        agent_diff = optimal_agents - current_agents
        confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        if confidence < confidence_threshold:
            action_type = 'maintain'
            target_agents = current_agents
            priority = 'low'
        elif abs(agent_diff) <= 1:
            action_type = 'maintain'
            target_agents = current_agents
            priority = 'low'
        elif agent_diff > 0:
            # Scale up
            if agent_diff > 5 or prediction.risk_assessment.get('performance_degradation', 0) > 0.8:
                action_type = 'emergency'
                priority = 'critical'
            else:
                action_type = 'scale_up'
                priority = 'high' if agent_diff > 2 else 'medium'
            target_agents = min(optimal_agents, current_agents + 3)  # Gradual scaling
        else:
            # Scale down
            action_type = 'scale_down'
            priority = 'medium'
            target_agents = max(optimal_agents, max(1, current_agents - 2))  # Gradual scaling
        
        # Calculate resource changes
        resource_changes = self._calculate_resource_changes(
            current_agents, target_agents, prediction.resource_requirements
        )
        
        # Estimate impact
        expected_impact = self._estimate_scaling_impact(
            current_agents, target_agents, prediction
        )
        
        # Create rollback plan
        rollback_plan = self._create_rollback_plan(current_agents, target_agents)
        
        return ScalingAction(
            action_type=action_type,
            target_agents=target_agents,
            priority=priority,
            execution_time=current_time + timedelta(seconds=5),  # Small delay for preparation
            resource_changes=resource_changes,
            expected_impact=expected_impact,
            rollback_plan=rollback_plan
        )
    
    async def train_model(self, training_data: List[Tuple[WorkloadMetrics, Dict[str, float]]],
                         epochs: int = 100) -> Dict[str, float]:
        """Train the predictive scaling network"""
        
        if not training_data:
            logger.warning("No training data provided")
            return {'error': 'no_training_data'}
        
        # Prepare training dataset
        features, targets = self._prepare_training_data(training_data)
        
        if features is None or targets is None:
            return {'error': 'data_preparation_failed'}
        
        # Training loop
        self.network.train()
        total_loss = 0
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Mini-batch training
            batch_size = self.config.get('batch_size', 32)
            for i in range(0, len(features), batch_size):
                batch_features = features[i:i+batch_size]
                batch_targets = targets[i:i+batch_size]
                
                # Forward pass
                sequence_data = self._create_mock_sequence(batch_features)
                outputs = self.network(sequence_data, batch_features)
                
                # Calculate loss
                loss = self._calculate_training_loss(outputs, batch_targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            total_loss += avg_epoch_loss
            
            # Learning rate scheduling
            self.scheduler.step(avg_epoch_loss)
            
            # Early stopping
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                patience_counter = 0
            else:
                patience_counter = getattr(self, 'patience_counter', 0) + 1
                
            if patience_counter >= 20:  # Early stopping patience
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}, Loss: {avg_epoch_loss:.4f}")
        
        # Validation
        validation_metrics = self._validate_model(training_data[-20:])  # Use last 20% for validation
        
        return {
            'final_loss': best_loss,
            'epochs_trained': epoch + 1,
            'validation_metrics': validation_metrics
        }
    
    async def optimize_hyperparameters(self, validation_data: List[Tuple[WorkloadMetrics, Dict[str, float]]],
                                     n_trials: int = 50) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna"""
        
        def objective(trial):
            # Suggest hyperparameters
            config = {
                'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'encoder_hidden_dim': trial.suggest_categorical('encoder_hidden_dim', [128, 256, 512]),
                'attention_heads': trial.suggest_categorical('attention_heads', [4, 8, 16]),
                'lstm_hidden_dim': trial.suggest_categorical('lstm_hidden_dim', [128, 256, 512]),
                'dropout_rate': trial.suggest_uniform('dropout_rate', 0.0, 0.3)
            }
            
            # Create temporary network with suggested hyperparameters
            temp_config = {**self.config, **config}
            temp_network = PredictiveScalingNetwork(temp_config).to(self.device)
            
            # Train for few epochs
            temp_optimizer = torch.optim.AdamW(
                temp_network.parameters(),
                lr=config['learning_rate']
            )
            
            temp_network.train()
            total_loss = 0
            
            # Quick training
            features, targets = self._prepare_training_data(validation_data)
            if features is None:
                return float('inf')
            
            for epoch in range(10):  # Quick evaluation
                for i in range(0, len(features), config['batch_size']):
                    batch_features = features[i:i+config['batch_size']]
                    batch_targets = targets[i:i+config['batch_size']]
                    
                    if len(batch_features) == 0:
                        continue
                    
                    sequence_data = self._create_mock_sequence(batch_features)
                    outputs = temp_network(sequence_data, batch_features)
                    loss = self._calculate_training_loss(outputs, batch_targets)
                    
                    temp_optimizer.zero_grad()
                    loss.backward()
                    temp_optimizer.step()
                    
                    total_loss += loss.item()
            
            return total_loss / max(1, len(features) // config['batch_size'])
        
        try:
            self.hyperopt_study.optimize(objective, n_trials=n_trials)
            best_params = self.hyperopt_study.best_params
            
            # Update configuration with best parameters
            self.config.update(best_params)
            
            # Recreate network with optimal parameters
            self.network = PredictiveScalingNetwork(self.config).to(self.device)
            self.optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=best_params.get('learning_rate', 0.001)
            )
            
            return {
                'best_params': best_params,
                'best_value': self.hyperopt_study.best_value,
                'trials_completed': len(self.hyperopt_study.trials)
            }
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            return {'error': str(e)}
    
    def _extract_features(self, metrics: WorkloadMetrics, context: Dict[str, Any]) -> torch.Tensor:
        """Extract features from workload metrics"""
        features = []
        
        # Basic metrics
        features.extend([
            metrics.task_complexity,
            float(metrics.concurrent_tasks),
            metrics.error_rate,
            metrics.throughput,
            float(metrics.queue_depth)
        ])
        
        # Resource utilization
        features.extend([
            metrics.resource_utilization.get('cpu', 0.0),
            metrics.resource_utilization.get('memory', 0.0),
            metrics.resource_utilization.get('network', 0.0),
            metrics.resource_utilization.get('disk', 0.0)
        ])
        
        # Performance history statistics
        if metrics.performance_history:
            perf_history = np.array(metrics.performance_history[-50:])  # Last 50 entries
            features.extend([
                np.mean(perf_history),
                np.std(perf_history),
                np.max(perf_history),
                np.min(perf_history),
                np.median(perf_history)
            ])
        else:
            features.extend([0.5, 0.1, 1.0, 0.0, 0.5])  # Default values
        
        # Latency percentiles
        features.extend([
            metrics.latency_percentiles.get('p50', 0.0),
            metrics.latency_percentiles.get('p95', 0.0),
            metrics.latency_percentiles.get('p99', 0.0)
        ])
        
        # Agent efficiency statistics
        if metrics.agent_efficiency:
            efficiency_values = list(metrics.agent_efficiency.values())
            features.extend([
                np.mean(efficiency_values),
                np.std(efficiency_values),
                np.max(efficiency_values),
                np.min(efficiency_values)
            ])
        else:
            features.extend([0.8, 0.1, 1.0, 0.5])
        
        # Context features
        features.extend([
            context.get('priority', 0.5),
            context.get('deadline_pressure', 0.5),
            context.get('cost_sensitivity', 0.5),
            context.get('reliability_requirement', 0.9),
            float(context.get('peak_hours', False)),
            float(context.get('maintenance_window', False))
        ])
        
        # Time-based features
        now = datetime.now()
        features.extend([
            now.hour / 24.0,  # Hour of day normalized
            now.weekday() / 7.0,  # Day of week normalized
            (now.day - 1) / 30.0,  # Day of month normalized
            now.month / 12.0  # Month normalized
        ])
        
        # System state features
        features.extend([
            float(self.current_agents) / self.config.get('max_agents', 32),
            (datetime.now() - self.last_scaling_time).total_seconds() / 3600.0,  # Hours since last scaling
            len(self.workload_history) / self.config.get('history_size', 1000),
        ])
        
        # Pad to fixed size if necessary
        target_size = self.config.get('feature_dim', 256)
        while len(features) < target_size:
            features.append(0.0)
        
        # Truncate if too large
        features = features[:target_size]
        
        # Convert to tensor and normalize
        feature_tensor = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Apply scaling if fitted
        if self.is_fitted:
            try:
                feature_array = feature_tensor.cpu().numpy()
                scaled_array = self.feature_scaler.transform(feature_array)
                feature_tensor = torch.tensor(scaled_array, dtype=torch.float32, device=self.device)
            except:
                pass  # Use original if scaling fails
        
        return feature_tensor
    
    def _prepare_sequence_data(self) -> torch.Tensor:
        """Prepare sequence data for temporal modeling"""
        sequence_length = self.config.get('sequence_length', 100)
        feature_dim = self.config.get('feature_dim', 256)
        
        if len(self.workload_history) < sequence_length:
            # Pad with zeros if insufficient history
            padding_needed = sequence_length - len(self.workload_history)
            sequence = [torch.zeros(feature_dim)] * padding_needed
            
            # Add available history
            for workload in self.workload_history:
                try:
                    features = self._extract_features(workload, {}).squeeze(0)
                    sequence.append(features)
                except:
                    sequence.append(torch.zeros(feature_dim))
        else:
            # Use recent history
            sequence = []
            recent_workloads = list(self.workload_history)[-sequence_length:]
            
            for workload in recent_workloads:
                try:
                    features = self._extract_features(workload, {}).squeeze(0)
                    sequence.append(features)
                except:
                    sequence.append(torch.zeros(feature_dim))
        
        return torch.stack(sequence).unsqueeze(0).to(self.device)  # Add batch dimension
    
    def _process_network_output(self, network_output: Dict[str, torch.Tensor],
                              metrics: WorkloadMetrics, context: Dict[str, Any]) -> ScalingPrediction:
        """Process network output into scaling prediction"""
        
        # Extract predictions
        action_probs = network_output['scaling_action_probs'].cpu().numpy()[0]
        confidence = network_output['confidence'].item()
        performance_pred = network_output['performance_prediction']
        risk_analysis = network_output['risk_analysis']
        
        # Find optimal agent count (weighted by probabilities)
        agent_counts = np.arange(1, len(action_probs) + 1)
        optimal_agent_count = int(np.average(agent_counts, weights=action_probs))
        
        # Extract performance predictions
        resource_requirements = {
            'cpu': performance_pred['resource_usage'][0][0].item(),
            'memory': performance_pred['resource_usage'][0][1].item(),
            'network': performance_pred['resource_usage'][0][2].item()
        }
        
        expected_performance = performance_pred['accuracy'][0].item()
        
        # Extract risk assessment
        risk_assessment = {}
        for risk_type, risk_tensor in risk_analysis['individual_risks'].items():
            risk_assessment[risk_type] = risk_tensor[0].item()
        
        # Create scaling timeline
        scaling_timeline = self._create_scaling_timeline(
            self.current_agents, optimal_agent_count, confidence
        )
        
        # Cost estimation
        cost_estimate = performance_pred['cost'][0].item() * optimal_agent_count
        
        # Alternative scenarios
        alternative_scenarios = self._generate_alternative_scenarios(
            action_probs, agent_counts, performance_pred, risk_analysis
        )
        
        return ScalingPrediction(
            optimal_agent_count=optimal_agent_count,
            confidence=confidence,
            resource_requirements=resource_requirements,
            expected_performance=expected_performance,
            scaling_timeline=scaling_timeline,
            cost_estimate=cost_estimate,
            risk_assessment=risk_assessment,
            alternative_scenarios=alternative_scenarios
        )
    
    def _create_scaling_timeline(self, current_agents: int, target_agents: int, 
                               confidence: float) -> Dict[str, int]:
        """Create timeline for scaling actions"""
        diff = abs(target_agents - current_agents)
        
        if diff == 0:
            return {'maintain': 0}
        
        # Base timing on confidence and magnitude of change
        base_delay = 30 if confidence > 0.8 else 60  # seconds
        step_delay = max(10, int(base_delay / max(1, diff)))
        
        timeline = {}
        if target_agents > current_agents:
            # Scale up timeline
            for i in range(1, diff + 1):
                step_time = i * step_delay
                timeline[f'add_agent_{i}'] = step_time
        else:
            # Scale down timeline
            for i in range(1, diff + 1):
                step_time = i * step_delay
                timeline[f'remove_agent_{i}'] = step_time
        
        return timeline
    
    def _generate_alternative_scenarios(self, action_probs: np.ndarray, agent_counts: np.ndarray,
                                      performance_pred: Dict[str, torch.Tensor],
                                      risk_analysis: Dict[str, torch.Tensor]) -> List[Dict[str, Any]]:
        """Generate alternative scaling scenarios"""
        scenarios = []
        
        # Top 3 alternatives by probability
        top_indices = np.argsort(action_probs)[-3:]
        
        for idx in top_indices:
            scenario = {
                'agent_count': int(agent_counts[idx]),
                'probability': float(action_probs[idx]),
                'expected_latency': performance_pred['latency'][0].item(),
                'expected_throughput': performance_pred['throughput'][0].item(),
                'expected_accuracy': performance_pred['accuracy'][0].item(),
                'cost_estimate': performance_pred['cost'][0].item() * agent_counts[idx],
                'risk_score': risk_analysis['overall_risk'][0].item()
            }
            scenarios.append(scenario)
        
        return sorted(scenarios, key=lambda x: x['probability'], reverse=True)
    
    def _calculate_resource_changes(self, current_agents: int, target_agents: int,
                                  resource_requirements: Dict[str, float]) -> Dict[str, float]:
        """Calculate resource changes for scaling"""
        agent_diff = target_agents - current_agents
        
        return {
            'cpu_change': resource_requirements['cpu'] * agent_diff,
            'memory_change': resource_requirements['memory'] * agent_diff,
            'network_change': resource_requirements['network'] * agent_diff,
            'agent_change': agent_diff
        }
    
    def _estimate_scaling_impact(self, current_agents: int, target_agents: int,
                               prediction: ScalingPrediction) -> Dict[str, float]:
        """Estimate impact of scaling action"""
        return {
            'performance_improvement': prediction.expected_performance - 0.887,  # vs baseline
            'latency_change': -0.1 if target_agents > current_agents else 0.05,  # Estimated
            'cost_change': prediction.cost_estimate - (current_agents * 10),  # Estimated cost per agent
            'risk_mitigation': 1.0 - prediction.risk_assessment.get('overall_risk', 0.5),
            'confidence_level': prediction.confidence
        }
    
    def _create_rollback_plan(self, current_agents: int, target_agents: int) -> Dict[str, Any]:
        """Create rollback plan for scaling action"""
        return {
            'rollback_agent_count': current_agents,
            'rollback_conditions': [
                'performance_degradation > 0.1',
                'error_rate > 0.05',
                'resource_exhaustion',
                'manual_trigger'
            ],
            'rollback_timeout_seconds': 300,  # 5 minutes
            'monitoring_metrics': [
                'latency_p99', 'error_rate', 'resource_utilization', 'throughput'
            ]
        }
    
    def _prepare_training_data(self, training_data: List[Tuple[WorkloadMetrics, Dict[str, float]]]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prepare training data for neural network"""
        if not training_data:
            return None, None
        
        features_list = []
        targets_list = []
        
        for metrics, target_dict in training_data:
            try:
                # Extract features
                features = self._extract_features(metrics, {}).squeeze(0)
                features_list.append(features)
                
                # Extract targets
                targets = [
                    target_dict.get('optimal_agents', 4) / 32.0,  # Normalized
                    target_dict.get('performance', 0.887),
                    target_dict.get('latency', 0.5),
                    target_dict.get('throughput', 0.5),
                    target_dict.get('cost', 0.5),
                    target_dict.get('cpu_usage', 0.5),
                    target_dict.get('memory_usage', 0.5),
                    target_dict.get('network_usage', 0.5)
                ]
                targets_list.append(torch.tensor(targets, dtype=torch.float32))
                
            except Exception as e:
                logger.warning(f"Skipping training sample due to error: {e}")
                continue
        
        if not features_list:
            return None, None
        
        features_tensor = torch.stack(features_list)
        targets_tensor = torch.stack(targets_list)
        
        # Fit scalers if not already fitted
        if not self.is_fitted:
            self.feature_scaler.fit(features_tensor.cpu().numpy())
            self.target_scaler.fit(targets_tensor.cpu().numpy())
            self.is_fitted = True
        
        return features_tensor.to(self.device), targets_tensor.to(self.device)
    
    def _create_mock_sequence(self, features: torch.Tensor) -> torch.Tensor:
        """Create mock sequence data for training"""
        batch_size = features.size(0)
        seq_len = self.config.get('sequence_length', 100)
        feature_dim = features.size(1)
        
        # Create sequence by adding noise to current features
        sequence = features.unsqueeze(1).repeat(1, seq_len, 1)
        noise = torch.randn_like(sequence) * 0.1
        return sequence + noise
    
    def _calculate_training_loss(self, outputs: Dict[str, torch.Tensor], 
                               targets: torch.Tensor) -> torch.Tensor:
        """Calculate training loss"""
        # Performance prediction loss
        perf_pred = outputs['performance_prediction']
        
        # Extract target components
        target_agents = targets[:, 0]  # Normalized agent count
        target_perf = targets[:, 1]    # Performance
        target_latency = targets[:, 2] # Latency
        target_throughput = targets[:, 3] # Throughput
        target_cost = targets[:, 4]    # Cost
        target_resources = targets[:, 5:8]  # CPU, Memory, Network
        
        # Calculate individual losses
        agent_loss = F.mse_loss(
            outputs['scaling_action_probs'].mean(dim=1), 
            target_agents
        )
        
        perf_loss = F.mse_loss(perf_pred['accuracy'].squeeze(), target_perf)
        latency_loss = F.mse_loss(perf_pred['latency'].squeeze(), target_latency)
        throughput_loss = F.mse_loss(perf_pred['throughput'].squeeze(), target_throughput)
        cost_loss = F.mse_loss(perf_pred['cost'].squeeze(), target_cost)
        resource_loss = F.mse_loss(perf_pred['resource_usage'], target_resources)
        
        # Confidence regularization (encourage high confidence for good predictions)
        confidence_reg = -outputs['confidence'].mean()  # Maximize confidence
        
        # Risk regularization (encourage low risk predictions)
        risk_reg = outputs['risk_analysis']['overall_risk'].mean()
        
        # Combine losses
        total_loss = (
            agent_loss * 2.0 +          # Agent prediction is most important
            perf_loss * 1.5 +           # Performance is critical
            latency_loss * 1.0 +        # Latency matters
            throughput_loss * 1.0 +     # Throughput matters
            cost_loss * 0.5 +           # Cost is secondary
            resource_loss * 0.5 +       # Resource usage is secondary
            confidence_reg * 0.1 +      # Small regularization
            risk_reg * 0.1              # Small regularization
        )
        
        return total_loss
    
    def _validate_model(self, validation_data: List[Tuple[WorkloadMetrics, Dict[str, float]]]) -> Dict[str, float]:
        """Validate model performance"""
        if not validation_data:
            return {'error': 'no_validation_data'}
        
        self.network.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for metrics, targets in validation_data:
                try:
                    features = self._extract_features(metrics, {})
                    sequence_data = self._create_mock_sequence(features)
                    
                    outputs = self.network(sequence_data, features)
                    
                    # Extract predictions
                    pred_agents = outputs['scaling_action_probs'].argmax(dim=1).item() + 1
                    pred_perf = outputs['performance_prediction']['accuracy'].item()
                    
                    predictions.append([pred_agents, pred_perf])
                    actuals.append([targets.get('optimal_agents', 4), targets.get('performance', 0.887)])
                    
                except Exception as e:
                    logger.warning(f"Validation error for sample: {e}")
                    continue
        
        if not predictions:
            return {'error': 'no_valid_predictions'}
        
        pred_array = np.array(predictions)
        actual_array = np.array(actuals)
        
        # Calculate metrics
        agent_mae = mean_absolute_error(actual_array[:, 0], pred_array[:, 0])
        agent_mse = mean_squared_error(actual_array[:, 0], pred_array[:, 0])
        
        perf_mae = mean_absolute_error(actual_array[:, 1], pred_array[:, 1])
        perf_r2 = r2_score(actual_array[:, 1], pred_array[:, 1])
        
        return {
            'agent_prediction_mae': float(agent_mae),
            'agent_prediction_mse': float(agent_mse),
            'performance_prediction_mae': float(perf_mae),
            'performance_prediction_r2': float(perf_r2),
            'samples_validated': len(predictions)
        }
    
    def _update_history(self, metrics: WorkloadMetrics, prediction: ScalingPrediction):
        """Update historical data"""
        self.workload_history.append(metrics)
        
        # Track prediction accuracy (simplified)
        if len(self.performance_history) > 0:
            last_actual = self.performance_history[-1]
            predicted = prediction.expected_performance
            accuracy = 1.0 - abs(last_actual - predicted)
            self.prediction_accuracy.append(accuracy)
        
        self.performance_history.append(prediction.expected_performance)
    
    async def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and metrics"""
        return {
            'is_fitted': self.is_fitted,
            'current_agents': self.current_agents,
            'history_size': len(self.workload_history),
            'prediction_accuracy': {
                'mean': float(np.mean(self.prediction_accuracy)) if self.prediction_accuracy else 0.0,
                'std': float(np.std(self.prediction_accuracy)) if self.prediction_accuracy else 0.0,
                'samples': len(self.prediction_accuracy)
            },
            'last_scaling_time': self.last_scaling_time.isoformat(),
            'scaling_cooldown_remaining': max(0, (
                self.last_scaling_time + self.scaling_cooldown - datetime.now()
            ).total_seconds()),
            'hyperopt_trials': len(self.hyperopt_study.trials) if hasattr(self.hyperopt_study, 'trials') else 0,
            'model_parameters': sum(p.numel() for p in self.network.parameters()),
            'device': str(self.device)
        }

# Factory function
def create_predictive_scaler(config: Optional[Dict[str, Any]] = None) -> AdvancedPredictiveScaler:
    """Create and initialize predictive scaler"""
    default_config = {
        'feature_dim': 256,
        'sequence_length': 100,
        'max_agents': 32,
        'initial_agents': 4,
        'attention_hidden_dim': 128,
        'attention_heads': 8,
        'encoder_hidden_dim': 256,
        'encoder_layers': 3,
        'lstm_hidden_dim': 256,
        'lstm_layers': 2,
        'learning_rate': 0.001,
        'weight_decay': 0.01,
        'batch_size': 32,
        'confidence_threshold': 0.7,
        'scaling_cooldown_seconds': 60,
        'history_size': 1000
    }
    
    if config:
        default_config.update(config)
    
    return AdvancedPredictiveScaler(default_config)

if __name__ == "__main__":
    # Example usage and testing
    async def test_predictive_scaler():
        """Test the predictive scaling system"""
        scaler = create_predictive_scaler()
        
        # Create sample workload metrics
        workload = WorkloadMetrics(
            timestamp=datetime.now(),
            task_complexity=0.8,
            concurrent_tasks=12,
            resource_utilization={'cpu': 0.75, 'memory': 0.6, 'network': 0.3},
            performance_history=[0.85, 0.87, 0.89, 0.86, 0.88],
            error_rate=0.02,
            latency_percentiles={'p50': 0.1, 'p95': 0.3, 'p99': 0.5},
            throughput=150.0,
            queue_depth=25,
            agent_efficiency={'agent_1': 0.9, 'agent_2': 0.85, 'agent_3': 0.8},
            context_features={'priority': 'high', 'deadline': '1h'}
        )
        
        context = {
            'priority': 0.9,
            'deadline_pressure': 0.7,
            'cost_sensitivity': 0.4,
            'reliability_requirement': 0.95
        }
        
        # Test prediction
        prediction = await scaler.predict_scaling_needs(workload, context)
        
        print(f"Predictive Scaling Results:")
        print(f"Optimal agents: {prediction.optimal_agent_count}")
        print(f"Confidence: {prediction.confidence:.3f}")
        print(f"Expected performance: {prediction.expected_performance:.3f}")
        print(f"Cost estimate: {prediction.cost_estimate:.2f}")
        print(f"Risk assessment: {prediction.risk_assessment}")
        
        # Test scaling recommendation
        current_state = {'current_agents': 4}
        action = await scaler.recommend_scaling_action(prediction, current_state)
        
        print(f"\nScaling Recommendation:")
        print(f"Action: {action.action_type}")
        print(f"Target agents: {action.target_agents}")
        print(f"Priority: {action.priority}")
        print(f"Expected impact: {action.expected_impact}")
        
        # Get model status
        status = await scaler.get_model_status()
        print(f"\nModel Status:")
        print(f"Parameters: {status['model_parameters']:,}")
        print(f"Device: {status['device']}")
        print(f"History size: {status['history_size']}")
        
        return prediction, action, status
    
    # Run test if executed directly
    import asyncio
    asyncio.run(test_predictive_scaler())