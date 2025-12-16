"""
Ensemble Predictor for SCI-ARC.

Combines multiple inference strategies for maximum accuracy:
1. Augmentation Voting (existing)
2. Stochastic Sampling (new)
3. Test-Time Training (new)
4. Consistency Verification (new)

Designed for ablation studies - each component can be toggled.

Usage:
    predictor = EnsemblePredictor(model, config)
    predictions = predictor.predict(task)
    
    # Returns ranked list of candidates with confidence scores
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from .sampler import StochasticSampler, SamplingConfig, ConsistencyVerifier
from .ttt import TTTAdapter, TTTConfig


@dataclass
class EnsembleConfig:
    """Configuration for ensemble prediction."""
    
    # === COMPONENT TOGGLES (for ablation studies) ===
    
    # Use Test-Time Training
    use_ttt: bool = True
    
    # Use stochastic sampling (MC Dropout + Temperature)
    use_stochastic_sampling: bool = True
    
    # Use augmentation voting (dihedral transforms)
    use_augmentation_voting: bool = True
    
    # Use consistency verification for candidate ranking
    use_consistency_verification: bool = True
    
    # === SAMPLING PARAMETERS ===
    
    # Number of stochastic samples
    num_samples: int = 32
    
    # Sampling temperature
    temperature: float = 0.8
    
    # Number of dihedral transforms for voting
    num_dihedral: int = 8
    
    # === TTT PARAMETERS ===
    
    # TTT learning rate
    ttt_learning_rate: float = 1e-4
    
    # TTT steps
    ttt_steps: int = 20
    
    # Modules to fine-tune during TTT
    ttt_modules: List[str] = field(default_factory=lambda: [
        'grid_encoder', 
        'structural_encoder'
    ])
    
    # === OUTPUT ===
    
    # Number of top candidates to return (ARC allows 2-3 attempts)
    top_k: int = 3
    
    # Device
    device: str = 'cuda'
    
    # Verbose logging
    verbose: bool = False


class EnsemblePredictor:
    """
    Ensemble prediction combining multiple inference strategies.
    
    Pipeline:
    1. [Optional] TTT: Adapt model to task
    2. [Optional] Augmentation: Apply dihedral transforms
    3. [Optional] Sampling: Generate diverse candidates
    4. Aggregation: Vote across all candidates
    5. [Optional] Verification: Re-rank by consistency
    
    All components can be toggled for ablation studies.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[EnsembleConfig] = None,
    ):
        self.model = model
        self.config = config or EnsembleConfig()
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize components based on config
        self._setup_components()
    
    def _setup_components(self):
        """Initialize inference components."""
        # TTT
        if self.config.use_ttt:
            ttt_config = TTTConfig(
                enabled=True,
                num_steps=self.config.ttt_steps,
                learning_rate=self.config.ttt_learning_rate,
                finetune_modules=self.config.ttt_modules,
                device=self.config.device,
                verbose=self.config.verbose,
            )
            self.ttt = TTTAdapter(self.model, ttt_config)
        else:
            self.ttt = None
        
        # Stochastic sampler
        if self.config.use_stochastic_sampling:
            sampling_config = SamplingConfig(
                num_samples=self.config.num_samples,
                temperature=self.config.temperature,
                use_mc_dropout=True,
                device=self.config.device,
            )
            self.sampler = StochasticSampler(self.model, sampling_config)
        else:
            self.sampler = None
        
        # Consistency verifier
        if self.config.use_consistency_verification:
            self.verifier = ConsistencyVerifier(self.model, self.config.device)
        else:
            self.verifier = None
    
    def _apply_augmentation_voting(
        self,
        input_grids: torch.Tensor,
        output_grids: torch.Tensor,
        test_input: torch.Tensor,
    ) -> List[np.ndarray]:
        """
        Generate predictions using augmentation voting.
        
        Applies dihedral transforms and collects predictions.
        """
        from ..evaluation.voting import (
            dihedral_transform_torch, 
            inverse_dihedral_transform_np
        )
        
        predictions = []
        
        with torch.no_grad():
            for tid in range(self.config.num_dihedral):
                # Transform inputs
                aug_in = dihedral_transform_torch(input_grids, tid)
                aug_out = dihedral_transform_torch(output_grids, tid)
                aug_test = dihedral_transform_torch(test_input, tid)
                
                # Prepare batch
                aug_in = aug_in.unsqueeze(0).to(self.device)
                aug_out = aug_out.unsqueeze(0).to(self.device)
                aug_test = aug_test.unsqueeze(0).to(self.device)
                test_output_dummy = torch.zeros_like(aug_test)
                
                # Forward
                outputs = self.model.forward_training(
                    input_grids=aug_in,
                    output_grids=aug_out,
                    test_input=aug_test,
                    test_output=test_output_dummy,
                )
                
                pred = outputs['logits'][0].argmax(dim=-1).cpu().numpy()
                
                # Inverse transform
                pred_inv = inverse_dihedral_transform_np(pred, tid)
                predictions.append(pred_inv)
        
        return predictions
    
    def predict(
        self,
        input_grids: torch.Tensor,   # [N, H, W]
        output_grids: torch.Tensor,  # [N, H, W]
        test_input: torch.Tensor,    # [H, W]
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate ranked predictions for a task.
        
        Args:
            input_grids: Demo inputs [N, H, W]
            output_grids: Demo outputs [N, H, W]
            test_input: Test input [H, W]
            target_shape: Expected output shape
        
        Returns:
            List of dicts with keys:
                - 'prediction': np.ndarray [H, W]
                - 'confidence': float (frequency-based)
                - 'consistency': float (if verification enabled)
                - 'rank': int
        """
        input_grids = input_grids.to(self.device)
        output_grids = output_grids.to(self.device)
        test_input = test_input.to(self.device)
        
        if target_shape is None:
            target_shape = (test_input.shape[0], test_input.shape[1])
        
        all_predictions = []
        
        try:
            # Step 1: TTT Adaptation
            if self.ttt is not None:
                if self.config.verbose:
                    print("Running TTT adaptation...")
                self.ttt.adapt(input_grids.cpu(), output_grids.cpu())
            
            # Step 2: Augmentation Voting
            if self.config.use_augmentation_voting:
                if self.config.verbose:
                    print(f"Running augmentation voting ({self.config.num_dihedral} transforms)...")
                aug_preds = self._apply_augmentation_voting(
                    input_grids.cpu(), output_grids.cpu(), test_input.cpu()
                )
                all_predictions.extend(aug_preds)
            
            # Step 3: Stochastic Sampling
            if self.sampler is not None:
                if self.config.verbose:
                    print(f"Running stochastic sampling ({self.config.num_samples} samples)...")
                candidates, _ = self.sampler.generate_candidates(
                    input_grids.cpu(), output_grids.cpu(), test_input.cpu(), target_shape
                )
                all_predictions.extend(candidates)
            
            # Step 4: Aggregate and rank
            if self.config.verbose:
                print(f"Aggregating {len(all_predictions)} predictions...")
            
            ranked_candidates = self._aggregate_and_rank(
                all_predictions, input_grids.cpu(), output_grids.cpu(), test_input.cpu()
            )
            
            return ranked_candidates[:self.config.top_k]
        
        finally:
            # Always reset TTT
            if self.ttt is not None:
                self.ttt.reset()
    
    def _aggregate_and_rank(
        self,
        predictions: List[np.ndarray],
        input_grids: torch.Tensor,
        output_grids: torch.Tensor,
        test_input: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """
        Aggregate predictions and rank by confidence.
        
        Confidence = frequency + consistency bonus
        """
        if not predictions:
            return []
        
        # Deduplicate and count
        pred_to_count: Dict[bytes, int] = {}
        pred_to_array: Dict[bytes, np.ndarray] = {}
        
        for pred in predictions:
            key = pred.tobytes()
            pred_to_count[key] = pred_to_count.get(key, 0) + 1
            if key not in pred_to_array:
                pred_to_array[key] = pred.copy()
        
        # Compute confidence scores
        total_preds = len(predictions)
        results = []
        
        for key, count in pred_to_count.items():
            pred = pred_to_array[key]
            frequency = count / total_preds
            
            # Consistency verification
            consistency = 0.0
            if self.verifier is not None:
                consistency = self.verifier.compute_consistency_score(
                    pred, input_grids, output_grids, test_input
                )
            
            # Combined confidence: weighted average
            # Frequency is primary, consistency is secondary
            confidence = 0.7 * frequency + 0.3 * consistency
            
            results.append({
                'prediction': pred,
                'confidence': confidence,
                'frequency': frequency,
                'consistency': consistency,
                'count': count,
            })
        
        # Sort by confidence (descending)
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Add rank
        for i, r in enumerate(results):
            r['rank'] = i + 1
        
        return results
    
    def predict_task(
        self,
        task: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Predict for a task in standard ARC format.
        
        Args:
            task: Dict with 'train' and 'test' keys
        
        Returns:
            List of ranked predictions for each test case
        """
        # Parse demos
        train_inputs = [
            torch.tensor(p['input'], dtype=torch.long) 
            for p in task['train']
        ]
        train_outputs = [
            torch.tensor(p['output'], dtype=torch.long)
            for p in task['train']
        ]
        
        # Stack
        input_grids = torch.stack(train_inputs)
        output_grids = torch.stack(train_outputs)
        
        all_results = []
        
        for test_pair in task['test']:
            test_input = torch.tensor(test_pair['input'], dtype=torch.long)
            
            # Predict
            predictions = self.predict(
                input_grids, output_grids, test_input
            )
            
            result = {
                'predictions': predictions,
                'test_input': test_pair['input'],
            }
            
            # Check correctness if ground truth available
            if 'output' in test_pair:
                target = np.array(test_pair['output'])
                result['target'] = target
                result['correct'] = any(
                    np.array_equal(p['prediction'], target)
                    for p in predictions
                )
            
            all_results.append(result)
        
        return all_results


def evaluate_with_ablation(
    model: nn.Module,
    tasks: List[Dict[str, Any]],
    config: EnsembleConfig,
    ablation_mode: str = 'full',
) -> Dict[str, float]:
    """
    Evaluate model with ablation study support.
    
    Args:
        model: SCI-ARC model
        tasks: List of ARC tasks
        config: Base ensemble config
        ablation_mode: One of:
            - 'full': All components enabled
            - 'no_ttt': Disable TTT
            - 'no_sampling': Disable stochastic sampling
            - 'no_voting': Disable augmentation voting
            - 'no_consistency': Disable consistency verification
            - 'baseline': All disabled (just greedy decoding)
    
    Returns:
        Dict with accuracy metrics
    """
    # Apply ablation
    ablation_config = EnsembleConfig(
        use_ttt=config.use_ttt,
        use_stochastic_sampling=config.use_stochastic_sampling,
        use_augmentation_voting=config.use_augmentation_voting,
        use_consistency_verification=config.use_consistency_verification,
        num_samples=config.num_samples,
        temperature=config.temperature,
        num_dihedral=config.num_dihedral,
        ttt_learning_rate=config.ttt_learning_rate,
        ttt_steps=config.ttt_steps,
        ttt_modules=config.ttt_modules,
        top_k=config.top_k,
        device=config.device,
        verbose=config.verbose,
    )
    
    if ablation_mode == 'no_ttt':
        ablation_config.use_ttt = False
    elif ablation_mode == 'no_sampling':
        ablation_config.use_stochastic_sampling = False
    elif ablation_mode == 'no_voting':
        ablation_config.use_augmentation_voting = False
    elif ablation_mode == 'no_consistency':
        ablation_config.use_consistency_verification = False
    elif ablation_mode == 'baseline':
        ablation_config.use_ttt = False
        ablation_config.use_stochastic_sampling = False
        ablation_config.use_augmentation_voting = False
        ablation_config.use_consistency_verification = False
    
    # Create predictor
    predictor = EnsemblePredictor(model, ablation_config)
    
    # Evaluate
    correct = 0
    total = 0
    
    for task in tasks:
        results = predictor.predict_task(task)
        for result in results:
            if 'correct' in result:
                total += 1
                if result['correct']:
                    correct += 1
    
    accuracy = correct / max(1, total)
    
    return {
        'mode': ablation_mode,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'config': {
            'use_ttt': ablation_config.use_ttt,
            'use_stochastic_sampling': ablation_config.use_stochastic_sampling,
            'use_augmentation_voting': ablation_config.use_augmentation_voting,
            'use_consistency_verification': ablation_config.use_consistency_verification,
        }
    }
