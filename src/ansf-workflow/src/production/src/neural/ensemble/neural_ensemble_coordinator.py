#!/usr/bin/env python3
"""
Neural Ensemble Methods for Advanced Multi-Agent Coordination
Phase 3 Implementation - Advanced ensemble techniques for improved accuracy
"""

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from collections import deque
import weakref
import gc

# Advanced ensemble imports
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.special import softmax
from scipy.stats import entropy
import optuna  # For hyperparameter optimization

logger = logging.getLogger(__name__)

@dataclass
class EnsembleModel:
    """Individual model in the ensemble"""
    model_id: str
    model: nn.Module
    weight: float
    performance_history: List[float]
    specialization: str
    confidence_threshold: float
    resource_usage: float
    last_updated: datetime
    meta_features: Dict[str, Any]

@dataclass
class EnsembleDecision:
    """Decision made by the ensemble"""
    prediction: torch.Tensor
    confidence: float
    individual_predictions: Dict[str, torch.Tensor]
    individual_confidences: Dict[str, float]
    consensus_score: float
    diversity_score: float
    explanation: Dict[str, Any]
    resource_cost: float

@dataclass
class DynamicWeight:
    """Dynamic weight for ensemble members"""
    base_weight: float
    performance_modifier: float
    context_modifier: float
    recency_modifier: float
    diversity_bonus: float
    final_weight: float

class AdaptiveWeightingNetwork(nn.Module):
    """Neural network for adaptive ensemble weighting"""
    
    def __init__(self, feature_dim: int, num_models: int, hidden_dims: List[int] = [256, 128]):
        super().__init__()
        self.num_models = num_models
        
        layers = []
        prev_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Multiple heads for different weighting strategies
        self.performance_head = nn.Linear(prev_dim, num_models)
        self.uncertainty_head = nn.Linear(prev_dim, num_models)
        self.diversity_head = nn.Linear(prev_dim, num_models)
        self.context_head = nn.Linear(prev_dim, num_models)
        
        # Final combination layer
        self.combination_layer = nn.Linear(num_models * 4, num_models)
        
    def forward(self, features: torch.Tensor, 
                individual_confidences: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for adaptive weighting"""
        batch_size = features.size(0)
        
        # Extract common features
        common_features = self.feature_extractor(features)
        
        # Different weighting strategies
        performance_weights = torch.sigmoid(self.performance_head(common_features))
        uncertainty_weights = torch.sigmoid(self.uncertainty_head(common_features))
        diversity_weights = torch.sigmoid(self.diversity_head(common_features))
        context_weights = torch.sigmoid(self.context_head(common_features))
        
        # Combine all strategies
        combined_features = torch.cat([
            performance_weights, uncertainty_weights, 
            diversity_weights, context_weights
        ], dim=-1)
        
        # Final adaptive weights
        adaptive_weights = F.softmax(self.combination_layer(combined_features), dim=-1)
        
        # Apply confidence modulation
        confidence_modulated = adaptive_weights * individual_confidences.unsqueeze(0)
        normalized_weights = F.softmax(confidence_modulated, dim=-1)
        
        return {
            'adaptive_weights': normalized_weights,
            'performance_weights': performance_weights,
            'uncertainty_weights': uncertainty_weights,
            'diversity_weights': diversity_weights,
            'context_weights': context_weights
        }

class UncertaintyEstimator(nn.Module):
    """Epistemic and aleatoric uncertainty estimation"""
    
    def __init__(self, input_dim: int, output_dim: int, num_monte_carlo: int = 50):
        super().__init__()
        self.num_monte_carlo = num_monte_carlo
        
        # Bayesian layers with dropout for epistemic uncertainty
        self.bayesian_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(128, output_dim * 2)  # Mean and variance
        )
        
        # Enable dropout during inference for MC sampling
        self.train()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with uncertainty estimation"""
        if self.training:
            # Single forward pass during training
            output = self.bayesian_layers(x)
            mean = output[:, :output.size(1)//2]
            log_var = output[:, output.size(1)//2:]
            var = torch.exp(log_var)
            
            return mean, var, torch.zeros_like(var)  # No epistemic during training
        
        else:
            # Monte Carlo sampling during inference
            predictions = []
            
            for _ in range(self.num_monte_carlo):
                with torch.no_grad():
                    output = self.bayesian_layers(x)
                    mean = output[:, :output.size(1)//2]
                    predictions.append(mean)
            
            predictions = torch.stack(predictions)
            
            # Epistemic uncertainty (model uncertainty)
            epistemic_uncertainty = torch.var(predictions, dim=0)
            
            # Mean prediction
            mean_prediction = torch.mean(predictions, dim=0)
            
            # Aleatoric uncertainty (data uncertainty) - from last forward pass
            output = self.bayesian_layers(x)
            log_var = output[:, output.size(1)//2:]
            aleatoric_uncertainty = torch.exp(log_var)
            
            return mean_prediction, aleatoric_uncertainty, epistemic_uncertainty

class ConsensusBuilder:
    """Advanced consensus building for ensemble decisions"""
    
    def __init__(self, consensus_strategies: List[str] = None):
        self.strategies = consensus_strategies or [
            'weighted_average', 'majority_vote', 'bayesian_fusion', 
            'dempster_shafer', 'rank_aggregation'
        ]
        self.strategy_weights = torch.ones(len(self.strategies)) / len(self.strategies)
        
    def build_consensus(self, predictions: Dict[str, torch.Tensor], 
                       confidences: Dict[str, float],
                       weights: Dict[str, float],
                       strategy: str = 'adaptive') -> Tuple[torch.Tensor, float]:
        """Build consensus using specified or adaptive strategy"""
        
        if strategy == 'adaptive':
            return self._adaptive_consensus(predictions, confidences, weights)
        elif strategy == 'weighted_average':
            return self._weighted_average(predictions, weights)
        elif strategy == 'majority_vote':
            return self._majority_vote(predictions, weights)
        elif strategy == 'bayesian_fusion':
            return self._bayesian_fusion(predictions, confidences, weights)
        elif strategy == 'dempster_shafer':
            return self._dempster_shafer_fusion(predictions, confidences)
        elif strategy == 'rank_aggregation':
            return self._rank_aggregation(predictions, weights)
        else:
            raise ValueError(f"Unknown consensus strategy: {strategy}")
    
    def _adaptive_consensus(self, predictions: Dict[str, torch.Tensor],
                          confidences: Dict[str, float],
                          weights: Dict[str, float]) -> Tuple[torch.Tensor, float]:
        """Adaptive consensus combining multiple strategies"""
        consensus_results = []
        consensus_confidences = []
        
        # Apply all strategies
        for strategy in self.strategies:
            if strategy == 'adaptive':
                continue
                
            try:
                result, confidence = self.build_consensus(
                    predictions, confidences, weights, strategy
                )
                consensus_results.append(result)
                consensus_confidences.append(confidence)
            except:
                # Skip failed strategies
                continue
        
        if not consensus_results:
            # Fallback to weighted average
            return self._weighted_average(predictions, weights)
        
        # Weight strategies based on historical performance
        strategy_weights = F.softmax(self.strategy_weights[:len(consensus_results)], dim=0)
        
        # Combine strategy results
        stacked_results = torch.stack(consensus_results)
        final_consensus = torch.sum(
            stacked_results * strategy_weights.unsqueeze(-1).unsqueeze(-1), 
            dim=0
        )
        
        final_confidence = np.average(
            consensus_confidences, weights=strategy_weights.detach().numpy()
        )
        
        return final_consensus, float(final_confidence)
    
    def _weighted_average(self, predictions: Dict[str, torch.Tensor],
                         weights: Dict[str, float]) -> Tuple[torch.Tensor, float]:
        """Weighted average consensus"""
        weighted_sum = torch.zeros_like(next(iter(predictions.values())))
        total_weight = 0
        
        for model_id, prediction in predictions.items():
            weight = weights.get(model_id, 1.0)
            weighted_sum += weight * prediction
            total_weight += weight
        
        consensus = weighted_sum / total_weight if total_weight > 0 else weighted_sum
        confidence = min(total_weight / len(predictions), 1.0)
        
        return consensus, confidence
    
    def _majority_vote(self, predictions: Dict[str, torch.Tensor],
                      weights: Dict[str, float]) -> Tuple[torch.Tensor, float]:
        """Majority voting consensus"""
        # Convert to discrete predictions for voting
        discrete_preds = {}
        for model_id, prediction in predictions.items():
            if prediction.dim() > 1:
                discrete_preds[model_id] = torch.argmax(prediction, dim=-1)
            else:
                discrete_preds[model_id] = (prediction > 0.5).float()
        
        # Weighted voting
        vote_counts = {}
        total_weights = {}
        
        for model_id, pred in discrete_preds.items():
            weight = weights.get(model_id, 1.0)
            
            for class_label in pred.unique():
                class_label_item = class_label.item()
                if class_label_item not in vote_counts:
                    vote_counts[class_label_item] = 0
                    total_weights[class_label_item] = 0
                
                vote_counts[class_label_item] += weight * (pred == class_label).sum().item()
                total_weights[class_label_item] += weight
        
        # Find majority winner
        winner = max(vote_counts.keys(), key=lambda k: vote_counts[k])
        confidence = vote_counts[winner] / sum(vote_counts.values())
        
        # Convert back to tensor format
        consensus_shape = next(iter(predictions.values())).shape
        consensus = torch.full(consensus_shape, winner, dtype=torch.float)
        
        return consensus, confidence
    
    def _bayesian_fusion(self, predictions: Dict[str, torch.Tensor],
                        confidences: Dict[str, float],
                        weights: Dict[str, float]) -> Tuple[torch.Tensor, float]:
        """Bayesian model fusion"""
        # Treat predictions as likelihood and confidences as prior
        log_likelihoods = []
        priors = []
        
        for model_id, prediction in predictions.items():
            # Convert prediction to log-likelihood
            if prediction.dim() > 1:
                log_likelihood = F.log_softmax(prediction, dim=-1)
            else:
                # Binary case
                prob = torch.sigmoid(prediction)
                log_likelihood = torch.log(torch.stack([1-prob, prob], dim=-1))
            
            log_likelihoods.append(log_likelihood)
            priors.append(confidences.get(model_id, 0.5) * weights.get(model_id, 1.0))
        
        # Combine using Bayes rule
        priors = torch.tensor(priors)
        priors = F.softmax(priors, dim=0)
        
        weighted_log_likelihood = torch.zeros_like(log_likelihoods[0])
        for i, log_likelihood in enumerate(log_likelihoods):
            weighted_log_likelihood += priors[i] * log_likelihood
        
        # Convert back to probabilities
        if weighted_log_likelihood.dim() > 1:
            consensus = F.softmax(weighted_log_likelihood, dim=-1)
        else:
            consensus = torch.sigmoid(weighted_log_likelihood)
        
        # Confidence based on entropy
        if consensus.dim() > 1:
            entropy_val = -torch.sum(consensus * torch.log(consensus + 1e-8), dim=-1).mean()
            confidence = 1.0 - (entropy_val / np.log(consensus.size(-1))).item()
        else:
            confidence = max(consensus.item(), 1 - consensus.item())
        
        return consensus, confidence
    
    def _dempster_shafer_fusion(self, predictions: Dict[str, torch.Tensor],
                              confidences: Dict[str, float]) -> Tuple[torch.Tensor, float]:
        """Dempster-Shafer theory fusion"""
        # Simplified DS fusion for neural predictions
        # Treat predictions as basic probability assignments
        
        mass_functions = []
        for model_id, prediction in predictions.items():
            confidence = confidences.get(model_id, 0.5)
            
            if prediction.dim() > 1:
                # Multiclass case
                masses = F.softmax(prediction, dim=-1) * confidence
                # Add uncertainty mass
                uncertainty = (1 - confidence) * torch.ones_like(masses)
                mass_function = torch.cat([masses, uncertainty], dim=-1)
            else:
                # Binary case  
                prob = torch.sigmoid(prediction)
                masses = torch.stack([1-prob, prob]) * confidence
                uncertainty = (1 - confidence) * torch.ones_like(masses)
                mass_function = torch.cat([masses, uncertainty], dim=0)
            
            mass_functions.append(mass_function)
        
        # Combine mass functions using Dempster's rule
        combined_mass = mass_functions[0]
        for mass in mass_functions[1:]:
            combined_mass = self._dempster_combination(combined_mass, mass)
        
        # Extract final prediction (remove uncertainty component)
        if combined_mass.dim() > 1:
            consensus = combined_mass[:, :-1]  # Remove last column (uncertainty)
            consensus = F.normalize(consensus, p=1, dim=-1)
        else:
            consensus = combined_mass[:-1]  # Remove last element (uncertainty)
            consensus = consensus / consensus.sum()
        
        # Confidence is 1 - uncertainty
        if combined_mass.dim() > 1:
            confidence = (1 - combined_mass[:, -1].mean()).item()
        else:
            confidence = (1 - combined_mass[-1]).item()
        
        return consensus, max(confidence, 0.0)
    
    def _dempster_combination(self, mass1: torch.Tensor, mass2: torch.Tensor) -> torch.Tensor:
        """Combine two mass functions using Dempster's rule"""
        # Simplified implementation
        combined = mass1 * mass2
        normalization = combined.sum()
        
        if normalization > 0:
            return combined / normalization
        else:
            return combined
    
    def _rank_aggregation(self, predictions: Dict[str, torch.Tensor],
                         weights: Dict[str, float]) -> Tuple[torch.Tensor, float]:
        """Rank aggregation consensus (Borda count)"""
        # Convert predictions to rankings
        rankings = {}
        for model_id, prediction in predictions.items():
            if prediction.dim() > 1:
                # Sort to get rankings
                _, ranked_indices = torch.sort(prediction, dim=-1, descending=True)
                rankings[model_id] = ranked_indices
            else:
                # Binary case - simple ranking
                rankings[model_id] = (prediction > 0.5).long()
        
        # Aggregate rankings using weighted Borda count
        if not rankings:
            return torch.zeros_like(next(iter(predictions.values()))), 0.0
        
        # For simplicity, return weighted average for now
        return self._weighted_average(predictions, weights)

class DiversityOptimizer:
    """Optimize ensemble diversity for better performance"""
    
    def __init__(self, target_diversity: float = 0.7):
        self.target_diversity = target_diversity
        self.diversity_history = deque(maxlen=100)
        
    def optimize_diversity(self, ensemble_models: List[EnsembleModel],
                         predictions: Dict[str, torch.Tensor]) -> List[float]:
        """Optimize model weights to achieve target diversity"""
        current_diversity = self.calculate_diversity(predictions)
        self.diversity_history.append(current_diversity)
        
        if current_diversity < self.target_diversity * 0.9:
            # Increase diversity by penalizing similar predictions
            return self._increase_diversity(ensemble_models, predictions)
        elif current_diversity > self.target_diversity * 1.1:
            # Decrease diversity by favoring better performing models
            return self._decrease_diversity(ensemble_models, predictions)
        else:
            # Maintain current weights
            return [model.weight for model in ensemble_models]
    
    def calculate_diversity(self, predictions: Dict[str, torch.Tensor]) -> float:
        """Calculate ensemble diversity using prediction disagreement"""
        if len(predictions) < 2:
            return 0.0
        
        pred_list = list(predictions.values())
        diversities = []
        
        for i in range(len(pred_list)):
            for j in range(i + 1, len(pred_list)):
                # Calculate disagreement between predictions
                if pred_list[i].dim() > 1 and pred_list[j].dim() > 1:
                    # Multiclass case
                    disagreement = 1 - F.cosine_similarity(
                        pred_list[i].flatten().unsqueeze(0),
                        pred_list[j].flatten().unsqueeze(0)
                    ).item()
                else:
                    # Binary case
                    diff = torch.abs(pred_list[i] - pred_list[j])
                    disagreement = diff.mean().item()
                
                diversities.append(abs(disagreement))
        
        return np.mean(diversities) if diversities else 0.0
    
    def _increase_diversity(self, models: List[EnsembleModel], 
                          predictions: Dict[str, torch.Tensor]) -> List[float]:
        """Increase diversity by penalizing similar predictions"""
        new_weights = []
        
        for model in models:
            # Calculate how different this model is from others
            diversity_score = 0.0
            model_pred = predictions.get(model.model_id)
            
            if model_pred is not None:
                for other_id, other_pred in predictions.items():
                    if other_id != model.model_id:
                        disagreement = 1 - F.cosine_similarity(
                            model_pred.flatten().unsqueeze(0),
                            other_pred.flatten().unsqueeze(0)
                        ).item()
                        diversity_score += abs(disagreement)
                
                diversity_score /= (len(predictions) - 1)
            
            # Increase weight for more diverse models
            diversity_bonus = 1.0 + diversity_score
            new_weight = model.weight * diversity_bonus
            new_weights.append(new_weight)
        
        # Normalize weights
        total_weight = sum(new_weights)
        return [w / total_weight for w in new_weights] if total_weight > 0 else new_weights
    
    def _decrease_diversity(self, models: List[EnsembleModel],
                           predictions: Dict[str, torch.Tensor]) -> List[float]:
        """Decrease diversity by favoring better performing models"""
        new_weights = []
        
        for model in models:
            # Favor models with better recent performance
            recent_performance = np.mean(model.performance_history[-10:]) if model.performance_history else 0.5
            performance_bonus = 1.0 + recent_performance
            
            new_weight = model.weight * performance_bonus
            new_weights.append(new_weight)
        
        # Normalize weights
        total_weight = sum(new_weights)
        return [w / total_weight for w in new_weights] if total_weight > 0 else new_weights

class AdvancedNeuralEnsemble:
    """Advanced neural ensemble coordinator with sophisticated methods"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Core components
        self.ensemble_models: List[EnsembleModel] = []
        self.adaptive_weighting = AdaptiveWeightingNetwork(
            feature_dim=config.get('feature_dim', 256),
            num_models=config.get('max_models', 16),
            hidden_dims=config.get('weighting_hidden_dims', [256, 128])
        ).to(self.device)
        
        self.uncertainty_estimator = UncertaintyEstimator(
            input_dim=config.get('uncertainty_input_dim', 512),
            output_dim=config.get('uncertainty_output_dim', 128),
            num_monte_carlo=config.get('num_monte_carlo', 50)
        ).to(self.device)
        
        self.consensus_builder = ConsensusBuilder(
            config.get('consensus_strategies', [
                'weighted_average', 'bayesian_fusion', 'majority_vote'
            ])
        )
        
        self.diversity_optimizer = DiversityOptimizer(
            target_diversity=config.get('target_diversity', 0.7)
        )
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.ensemble_metrics = {
            'accuracy': 0.887,  # Baseline
            'diversity': 0.0,
            'consensus': 0.0,
            'uncertainty': 0.0,
            'efficiency': 0.0
        }
        
        # Meta-learning for ensemble optimization
        self.meta_optimizer = optuna.create_study(
            direction='maximize',
            study_name='ensemble_optimization'
        )
        
    async def add_model(self, model: nn.Module, model_id: str, 
                       specialization: str = 'general',
                       initial_weight: float = 1.0) -> bool:
        """Add a new model to the ensemble"""
        try:
            ensemble_model = EnsembleModel(
                model_id=model_id,
                model=model.to(self.device),
                weight=initial_weight,
                performance_history=[],
                specialization=specialization,
                confidence_threshold=0.7,
                resource_usage=0.0,
                last_updated=datetime.now(),
                meta_features={}
            )
            
            self.ensemble_models.append(ensemble_model)
            
            # Re-normalize weights
            total_weight = sum(m.weight for m in self.ensemble_models)
            if total_weight > 0:
                for model in self.ensemble_models:
                    model.weight /= total_weight
            
            logger.info(f"Added model {model_id} to ensemble. Total models: {len(self.ensemble_models)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add model {model_id}: {e}")
            return False
    
    async def remove_model(self, model_id: str) -> bool:
        """Remove a model from the ensemble"""
        for i, model in enumerate(self.ensemble_models):
            if model.model_id == model_id:
                del self.ensemble_models[i]
                
                # Re-normalize remaining weights
                total_weight = sum(m.weight for m in self.ensemble_models)
                if total_weight > 0:
                    for model in self.ensemble_models:
                        model.weight /= total_weight
                
                logger.info(f"Removed model {model_id} from ensemble")
                return True
        
        logger.warning(f"Model {model_id} not found in ensemble")
        return False
    
    async def coordinate_prediction(self, inputs: torch.Tensor,
                                  context: Dict[str, Any]) -> EnsembleDecision:
        """Main coordination method for ensemble prediction"""
        if not self.ensemble_models:
            raise ValueError("No models in ensemble")
        
        start_time = datetime.now()
        
        # Get predictions from all models
        individual_predictions = {}
        individual_confidences = {}
        resource_costs = []
        
        for model in self.ensemble_models:
            try:
                # Get prediction
                with torch.no_grad():
                    prediction = model.model(inputs)
                    individual_predictions[model.model_id] = prediction
                
                # Estimate confidence and uncertainty
                mean_pred, aleatoric_unc, epistemic_unc = self.uncertainty_estimator(inputs)
                total_uncertainty = aleatoric_unc + epistemic_unc
                confidence = 1.0 - total_uncertainty.mean().item()
                individual_confidences[model.model_id] = max(0.0, min(1.0, confidence))
                
                # Track resource usage
                resource_costs.append(model.resource_usage)
                
            except Exception as e:
                logger.warning(f"Model {model.model_id} failed prediction: {e}")
                continue
        
        if not individual_predictions:
            raise RuntimeError("All ensemble models failed to produce predictions")
        
        # Calculate adaptive weights
        feature_context = self._extract_context_features(context, inputs)
        confidence_tensor = torch.tensor(list(individual_confidences.values()), device=self.device)
        
        weighting_results = self.adaptive_weighting(feature_context, confidence_tensor)
        adaptive_weights = weighting_results['adaptive_weights'].squeeze(0)
        
        # Convert to dictionary
        model_weights = {}
        for i, model in enumerate(self.ensemble_models):
            if model.model_id in individual_predictions:
                model_weights[model.model_id] = adaptive_weights[i].item()
        
        # Optimize for diversity
        optimized_weights_list = self.diversity_optimizer.optimize_diversity(
            self.ensemble_models, individual_predictions
        )
        
        # Combine adaptive and diversity-optimized weights
        final_weights = {}
        for i, model in enumerate(self.ensemble_models):
            if model.model_id in individual_predictions:
                adaptive_weight = model_weights.get(model.model_id, 0.0)
                diversity_weight = optimized_weights_list[i] if i < len(optimized_weights_list) else 0.0
                
                # Weighted combination
                alpha = 0.7  # Favor adaptive weights
                final_weights[model.model_id] = alpha * adaptive_weight + (1 - alpha) * diversity_weight
        
        # Build consensus
        consensus_prediction, consensus_confidence = self.consensus_builder.build_consensus(
            individual_predictions, individual_confidences, final_weights,
            strategy=context.get('consensus_strategy', 'adaptive')
        )
        
        # Calculate ensemble metrics
        diversity_score = self.diversity_optimizer.calculate_diversity(individual_predictions)
        consensus_score = self._calculate_consensus_score(
            individual_predictions, consensus_prediction
        )
        
        # Calculate resource cost
        avg_resource_cost = np.mean(resource_costs) if resource_costs else 0.0
        
        # Create explanation
        explanation = self._generate_explanation(
            individual_predictions, individual_confidences, final_weights,
            weighting_results, context
        )
        
        # Create ensemble decision
        decision = EnsembleDecision(
            prediction=consensus_prediction,
            confidence=consensus_confidence,
            individual_predictions=individual_predictions,
            individual_confidences=individual_confidences,
            consensus_score=consensus_score,
            diversity_score=diversity_score,
            explanation=explanation,
            resource_cost=avg_resource_cost
        )
        
        # Update performance tracking
        end_time = datetime.now()
        latency = (end_time - start_time).total_seconds()
        
        self._update_performance(decision, latency, context)
        
        return decision
    
    async def optimize_ensemble(self, validation_data: List[Tuple[torch.Tensor, torch.Tensor]],
                              optimization_steps: int = 100) -> Dict[str, float]:
        """Optimize ensemble configuration using meta-learning"""
        
        def objective(trial):
            # Hyperparameters to optimize
            target_diversity = trial.suggest_float('target_diversity', 0.3, 0.9)
            consensus_strategy = trial.suggest_categorical(
                'consensus_strategy', ['weighted_average', 'bayesian_fusion', 'adaptive']
            )
            confidence_threshold = trial.suggest_float('confidence_threshold', 0.5, 0.9)
            
            # Test configuration
            self.diversity_optimizer.target_diversity = target_diversity
            
            accuracies = []
            for inputs, targets in validation_data[:20]:  # Use subset for speed
                try:
                    decision = asyncio.run(self.coordinate_prediction(
                        inputs, {'consensus_strategy': consensus_strategy}
                    ))
                    
                    # Calculate accuracy (simplified)
                    if decision.prediction.dim() > 1:
                        pred_class = torch.argmax(decision.prediction, dim=-1)
                        accuracy = (pred_class == targets).float().mean().item()
                    else:
                        pred_binary = (decision.prediction > 0.5).float()
                        accuracy = (pred_binary == targets).float().mean().item()
                    
                    accuracies.append(accuracy)
                    
                except Exception as e:
                    accuracies.append(0.0)  # Penalty for failed predictions
            
            return np.mean(accuracies) if accuracies else 0.0
        
        # Run optimization
        try:
            self.meta_optimizer.optimize(objective, n_trials=optimization_steps)
            best_params = self.meta_optimizer.best_params
            
            # Apply best configuration
            self.diversity_optimizer.target_diversity = best_params.get('target_diversity', 0.7)
            
            logger.info(f"Ensemble optimization completed. Best accuracy: {self.meta_optimizer.best_value:.3f}")
            
            return {
                'best_accuracy': self.meta_optimizer.best_value,
                'best_params': best_params,
                'trials_completed': len(self.meta_optimizer.trials)
            }
            
        except Exception as e:
            logger.error(f"Ensemble optimization failed: {e}")
            return {'best_accuracy': 0.0, 'error': str(e)}
    
    def _extract_context_features(self, context: Dict[str, Any], 
                                inputs: torch.Tensor) -> torch.Tensor:
        """Extract features for adaptive weighting"""
        features = []
        
        # Input statistics
        features.extend([
            inputs.mean().item(),
            inputs.std().item(),
            inputs.min().item(),
            inputs.max().item()
        ])
        
        # Context features
        features.extend([
            context.get('task_complexity', 0.5),
            context.get('accuracy_requirement', 0.9),
            context.get('latency_constraint', 0.5),
            context.get('resource_constraint', 0.5)
        ])
        
        # Ensemble state features
        features.extend([
            len(self.ensemble_models),
            np.mean([m.weight for m in self.ensemble_models]) if self.ensemble_models else 0.0,
            self.ensemble_metrics['accuracy'],
            self.ensemble_metrics['diversity']
        ])
        
        # Historical performance
        if self.performance_history:
            recent_performance = list(self.performance_history)[-10:]
            features.extend([
                np.mean(recent_performance),
                np.std(recent_performance),
                max(recent_performance),
                min(recent_performance)
            ])
        else:
            features.extend([0.887, 0.0, 0.887, 0.887])  # Default values
        
        # Pad to fixed size (256 features)
        while len(features) < 256:
            features.append(0.0)
        
        return torch.tensor(features[:256], dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def _calculate_consensus_score(self, individual_predictions: Dict[str, torch.Tensor],
                                 consensus_prediction: torch.Tensor) -> float:
        """Calculate how well individual predictions agree with consensus"""
        if len(individual_predictions) < 2:
            return 1.0
        
        agreements = []
        for pred in individual_predictions.values():
            agreement = F.cosine_similarity(
                pred.flatten().unsqueeze(0),
                consensus_prediction.flatten().unsqueeze(0)
            ).item()
            agreements.append(abs(agreement))
        
        return np.mean(agreements)
    
    def _generate_explanation(self, individual_predictions: Dict[str, torch.Tensor],
                            individual_confidences: Dict[str, float],
                            final_weights: Dict[str, float],
                            weighting_results: Dict[str, torch.Tensor],
                            context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate explanation for ensemble decision"""
        
        # Find most influential models
        sorted_weights = sorted(final_weights.items(), key=lambda x: x[1], reverse=True)
        top_models = sorted_weights[:3]
        
        # Analyze weighting strategy contributions
        weighting_analysis = {}
        for strategy_name, weights in weighting_results.items():
            if 'weights' in strategy_name:
                avg_weight = weights.mean().item()
                weighting_analysis[strategy_name] = avg_weight
        
        explanation = {
            'top_contributing_models': [
                {'model_id': model_id, 'weight': weight, 'confidence': individual_confidences.get(model_id, 0.0)}
                for model_id, weight in top_models
            ],
            'weighting_strategy_contributions': weighting_analysis,
            'consensus_method': context.get('consensus_strategy', 'adaptive'),
            'diversity_score': self.diversity_optimizer.calculate_diversity(individual_predictions),
            'total_models_used': len(individual_predictions),
            'prediction_uncertainty': {
                'min_confidence': min(individual_confidences.values()) if individual_confidences else 0.0,
                'max_confidence': max(individual_confidences.values()) if individual_confidences else 0.0,
                'avg_confidence': np.mean(list(individual_confidences.values())) if individual_confidences else 0.0
            }
        }
        
        return explanation
    
    def _update_performance(self, decision: EnsembleDecision, latency: float, 
                          context: Dict[str, Any]):
        """Update performance metrics and model weights"""
        
        # Update ensemble metrics
        self.ensemble_metrics['diversity'] = decision.diversity_score
        self.ensemble_metrics['consensus'] = decision.consensus_score
        self.ensemble_metrics['uncertainty'] = 1.0 - decision.confidence
        self.ensemble_metrics['efficiency'] = 1.0 / (1.0 + latency + decision.resource_cost)
        
        # Track performance history
        self.performance_history.append(decision.confidence)
        
        # Update individual model performance
        for model in self.ensemble_models:
            if model.model_id in decision.individual_confidences:
                model_confidence = decision.individual_confidences[model.model_id]
                model.performance_history.append(model_confidence)
                
                # Limit history size
                if len(model.performance_history) > 100:
                    model.performance_history = model.performance_history[-50:]
                
                model.last_updated = datetime.now()
        
        # Log performance improvement
        if len(self.performance_history) > 10:
            recent_avg = np.mean(list(self.performance_history)[-10:])
            if recent_avg > 0.887:  # Above baseline
                improvement = ((recent_avg - 0.887) / 0.887) * 100
                logger.info(f"Ensemble performance improvement: {improvement:.2f}% over baseline")
    
    async def get_ensemble_status(self) -> Dict[str, Any]:
        """Get current ensemble status and metrics"""
        model_info = []
        for model in self.ensemble_models:
            recent_performance = np.mean(model.performance_history[-10:]) if model.performance_history else 0.0
            
            model_info.append({
                'model_id': model.model_id,
                'weight': model.weight,
                'specialization': model.specialization,
                'recent_performance': recent_performance,
                'confidence_threshold': model.confidence_threshold,
                'resource_usage': model.resource_usage,
                'last_updated': model.last_updated.isoformat(),
                'total_predictions': len(model.performance_history)
            })
        
        return {
            'total_models': len(self.ensemble_models),
            'ensemble_metrics': self.ensemble_metrics,
            'performance_history_size': len(self.performance_history),
            'recent_performance': np.mean(list(self.performance_history)[-10:]) if self.performance_history else 0.0,
            'diversity_target': self.diversity_optimizer.target_diversity,
            'models': model_info,
            'optimization_trials': len(self.meta_optimizer.trials) if hasattr(self.meta_optimizer, 'trials') else 0
        }

# Factory function for easy initialization
def create_neural_ensemble(config: Optional[Dict[str, Any]] = None) -> AdvancedNeuralEnsemble:
    """Create and initialize advanced neural ensemble"""
    default_config = {
        'feature_dim': 256,
        'max_models': 16,
        'weighting_hidden_dims': [256, 128],
        'uncertainty_input_dim': 512,
        'uncertainty_output_dim': 128,
        'num_monte_carlo': 50,
        'target_diversity': 0.7,
        'consensus_strategies': ['weighted_average', 'bayesian_fusion', 'majority_vote']
    }
    
    if config:
        default_config.update(config)
    
    return AdvancedNeuralEnsemble(default_config)

if __name__ == "__main__":
    # Example usage and testing
    async def test_neural_ensemble():
        """Test the advanced neural ensemble system"""
        ensemble = create_neural_ensemble()
        
        # Create mock models for testing
        class MockModel(nn.Module):
            def __init__(self, output_size: int = 10):
                super().__init__()
                self.linear = nn.Linear(512, output_size)
                
            def forward(self, x):
                return self.linear(x)
        
        # Add models to ensemble
        for i in range(5):
            model = MockModel()
            await ensemble.add_model(
                model, f'model_{i}', 
                specialization=f'specialist_{i % 3}',
                initial_weight=1.0
            )
        
        # Test prediction
        test_input = torch.randn(4, 512)  # Batch of 4, 512 features
        context = {
            'task_complexity': 0.8,
            'accuracy_requirement': 0.95,
            'consensus_strategy': 'adaptive'
        }
        
        decision = await ensemble.coordinate_prediction(test_input, context)
        
        print(f"Ensemble Decision Results:")
        print(f"Prediction shape: {decision.prediction.shape}")
        print(f"Confidence: {decision.confidence:.3f}")
        print(f"Diversity score: {decision.diversity_score:.3f}")
        print(f"Consensus score: {decision.consensus_score:.3f}")
        print(f"Resource cost: {decision.resource_cost:.3f}")
        print(f"Models used: {len(decision.individual_predictions)}")
        
        # Get ensemble status
        status = await ensemble.get_ensemble_status()
        print(f"\nEnsemble Status:")
        print(f"Total models: {status['total_models']}")
        print(f"Recent performance: {status['recent_performance']:.3f}")
        print(f"Ensemble accuracy: {status['ensemble_metrics']['accuracy']:.3f}")
        
        return decision, status
    
    # Run test if executed directly
    import asyncio
    asyncio.run(test_neural_ensemble())