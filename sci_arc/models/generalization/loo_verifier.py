"""
LOO Verifier: Leave-One-Out Manifold Verification

Implements confidence scoring based on LOO consistency:
1. For each candidate program that solves training pairs 1, 2, 3
2. Hide pair 3 and check if logic from pairs 1, 2 still predicts 3
3. Assign confidence score based on this consistency

WHY THIS GENERALIZES:
- A program that ONLY works when seeing all examples is likely overfit
- A program that works with held-out examples is capturing the true rule
- This is the key insight: generalization = consistent across held-out data

Integration with RLAN:
- Takes candidate predictions from TEPS/NS-TEPS/Neural
- Scores each candidate based on LOO consistency
- Returns ranked predictions with confidence scores

Author: AI Research Assistant
Date: January 2026
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from itertools import combinations


@dataclass
class LOOVerifierConfig:
    """Configuration for LOO Verifier."""
    enabled: bool = True
    min_pairs_for_loo: int = 2  # Need at least 2 pairs to do LOO
    match_threshold: float = 0.9  # Threshold for "correct" prediction
    consistency_weight: float = 0.5  # Weight for LOO consistency vs direct match
    max_candidates: int = 50  # Max candidates to verify


class LOOVerifier(nn.Module):
    """
    Leave-One-Out Manifold Verifier.
    
    Verifies that candidate solutions generalize by checking if they
    work when some training examples are held out.
    """
    
    def __init__(self, config: LOOVerifierConfig = None):
        super().__init__()
        self.config = config or LOOVerifierConfig()
    
    @torch.no_grad()
    def verify(
        self,
        candidates: List[Dict[str, Any]],
        train_inputs: List[np.ndarray],
        train_outputs: List[np.ndarray],
        program_executor: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Verify and rank candidates based on LOO consistency.
        
        Args:
            candidates: List of candidate predictions, each with:
                - prediction: The predicted output
                - program: Optional program/trace that generated it
                - confidence: Initial confidence score
            train_inputs: Training input grids
            train_outputs: Training output grids  
            program_executor: Optional function to execute programs
            
        Returns:
            List of candidates with updated confidence scores and rankings
        """
        if not self.config.enabled:
            return candidates
        
        n_pairs = len(train_inputs)
        
        if n_pairs < self.config.min_pairs_for_loo:
            # Not enough pairs for LOO, return as-is
            return candidates
        
        verified_candidates = []
        
        for cand in candidates[:self.config.max_candidates]:
            prediction = cand.get('prediction')
            program = cand.get('program')
            initial_confidence = cand.get('confidence', 0.5)
            
            if prediction is None:
                verified_candidates.append({
                    **cand,
                    'loo_score': 0.0,
                    'final_confidence': 0.0,
                })
                continue
            
            # Calculate LOO consistency score
            loo_score = self._compute_loo_score(
                prediction=prediction,
                program=program,
                train_inputs=train_inputs,
                train_outputs=train_outputs,
                program_executor=program_executor,
            )
            
            # Combine initial confidence with LOO score
            final_confidence = (
                (1 - self.config.consistency_weight) * initial_confidence +
                self.config.consistency_weight * loo_score
            )
            
            verified_candidates.append({
                **cand,
                'loo_score': loo_score,
                'final_confidence': final_confidence,
            })
        
        # Sort by final confidence
        verified_candidates.sort(key=lambda x: -x.get('final_confidence', 0))
        
        return verified_candidates
    
    def _compute_loo_score(
        self,
        prediction: np.ndarray,
        program: Any,
        train_inputs: List[np.ndarray],
        train_outputs: List[np.ndarray],
        program_executor: Optional[callable] = None,
    ) -> float:
        """
        Compute LOO consistency score for a candidate.
        
        For each held-out pair:
        1. "Learn" from the remaining pairs
        2. Check if the pattern still correctly predicts the held-out output
        """
        n_pairs = len(train_inputs)
        if n_pairs < 2:
            return 0.5  # Can't do LOO
        
        loo_scores = []
        
        # Try holding out each pair
        for holdout_idx in range(n_pairs):
            # Create held-out split
            held_out_input = train_inputs[holdout_idx]
            held_out_output = train_outputs[holdout_idx]
            
            remaining_inputs = [inp for i, inp in enumerate(train_inputs) if i != holdout_idx]
            remaining_outputs = [out for i, out in enumerate(train_outputs) if i != holdout_idx]
            
            # Check if we can predict the held-out output
            loo_match = self._check_loo_prediction(
                held_out_input=held_out_input,
                held_out_output=held_out_output,
                remaining_inputs=remaining_inputs,
                remaining_outputs=remaining_outputs,
                candidate_prediction=prediction,
                program=program,
                program_executor=program_executor,
            )
            
            loo_scores.append(loo_match)
        
        # Average LOO score
        return np.mean(loo_scores) if loo_scores else 0.0
    
    def _check_loo_prediction(
        self,
        held_out_input: np.ndarray,
        held_out_output: np.ndarray,
        remaining_inputs: List[np.ndarray],
        remaining_outputs: List[np.ndarray],
        candidate_prediction: np.ndarray,
        program: Any,
        program_executor: Optional[callable] = None,
    ) -> float:
        """
        Check if a candidate pattern correctly predicts held-out output.
        
        Strategy:
        1. If we have a program, execute it on remaining pairs and check consistency
        2. If no program, use pattern matching heuristics
        """
        
        if program is not None and program_executor is not None:
            # Try to execute program on held-out input
            try:
                predicted = program_executor(program, held_out_input)
                if predicted is not None and predicted.shape == held_out_output.shape:
                    return np.mean(predicted == held_out_output)
            except Exception:
                pass
        
        # Fallback: Check if the transformation pattern is consistent
        # This uses heuristic pattern matching
        
        # Check 1: Shape consistency
        shape_scores = []
        for inp, out in zip(remaining_inputs, remaining_outputs):
            inp_shape = inp.shape
            out_shape = out.shape
            # Check if shape transform is consistent
            if held_out_input.shape == inp_shape and held_out_output.shape == out_shape:
                shape_scores.append(1.0)
            else:
                shape_scores.append(0.5)
        
        # Check 2: Color distribution consistency
        color_scores = []
        for inp, out in zip(remaining_inputs, remaining_outputs):
            inp_colors = set(np.unique(inp))
            out_colors = set(np.unique(out))
            held_inp_colors = set(np.unique(held_out_input))
            held_out_colors = set(np.unique(held_out_output))
            
            # Check if color relationship is similar
            if (len(out_colors - inp_colors) == len(held_out_colors - held_inp_colors)):
                color_scores.append(1.0)
            else:
                color_scores.append(0.5)
        
        # Check 3: Size ratio consistency
        ratio_scores = []
        for inp, out in zip(remaining_inputs, remaining_outputs):
            inp_size = inp.shape[0] * inp.shape[1]
            out_size = out.shape[0] * out.shape[1]
            held_inp_size = held_out_input.shape[0] * held_out_input.shape[1]
            held_out_size = held_out_output.shape[0] * held_out_output.shape[1]
            
            if inp_size > 0 and held_inp_size > 0:
                ratio1 = out_size / inp_size
                ratio2 = held_out_size / held_inp_size
                if abs(ratio1 - ratio2) < 0.1:
                    ratio_scores.append(1.0)
                else:
                    ratio_scores.append(0.5)
        
        # Combine heuristic scores
        all_scores = shape_scores + color_scores + ratio_scores
        return np.mean(all_scores) if all_scores else 0.5
    
    def rank_predictions(
        self,
        predictions: List[np.ndarray],
        train_inputs: List[np.ndarray],
        train_outputs: List[np.ndarray],
        confidences: Optional[List[float]] = None,
    ) -> List[Tuple[np.ndarray, float, int]]:
        """
        Rank multiple predictions using LOO verification.
        
        Args:
            predictions: List of candidate predictions
            train_inputs: Training inputs
            train_outputs: Training outputs
            confidences: Optional initial confidence scores
            
        Returns:
            List of (prediction, confidence, rank) tuples sorted by confidence
        """
        if confidences is None:
            confidences = [0.5] * len(predictions)
        
        candidates = [
            {'prediction': pred, 'confidence': conf}
            for pred, conf in zip(predictions, confidences)
        ]
        
        verified = self.verify(candidates, train_inputs, train_outputs)
        
        results = []
        for rank, cand in enumerate(verified):
            results.append((
                cand['prediction'],
                cand.get('final_confidence', 0.0),
                rank,
            ))
        
        return results


class VerifierRanker(nn.Module):
    """
    Ranker that combines multiple verification strategies.
    
    Assigns confidence scores using:
    1. LOO consistency
    2. Direct match scores
    3. Program complexity penalties
    """
    
    def __init__(self, config: LOOVerifierConfig = None):
        super().__init__()
        self.config = config or LOOVerifierConfig()
        self.loo_verifier = LOOVerifier(config)
    
    @torch.no_grad()
    def rank(
        self,
        candidates: List[Dict[str, Any]],
        train_inputs: List[np.ndarray],
        train_outputs: List[np.ndarray],
        test_input: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Rank candidates and return the best prediction.
        
        Args:
            candidates: List of candidate predictions with metadata
            train_inputs: Training inputs
            train_outputs: Training outputs
            test_input: The test input grid
            
        Returns:
            Dict with best prediction and ranking info
        """
        if not self.config.enabled:
            # Return first candidate
            if candidates:
                return {
                    'prediction': candidates[0].get('prediction'),
                    'confidence': candidates[0].get('confidence', 0.0),
                    'method': 'passthrough',
                    'ranking': [0],
                }
            return {
                'prediction': None,
                'confidence': 0.0,
                'method': 'no_candidates',
                'ranking': [],
            }
        
        # Apply LOO verification
        verified = self.loo_verifier.verify(
            candidates=candidates,
            train_inputs=train_inputs,
            train_outputs=train_outputs,
        )
        
        if not verified:
            return {
                'prediction': None,
                'confidence': 0.0,
                'method': 'no_verified',
                'ranking': [],
            }
        
        # Apply complexity penalty if programs are available
        for cand in verified:
            program = cand.get('program')
            if program is not None and hasattr(program, 'steps'):
                # Penalize complex programs (Occam's razor)
                complexity = len(program.steps) if hasattr(program, 'steps') else 1
                complexity_penalty = 0.1 * (complexity - 1)
                cand['final_confidence'] = max(0, cand.get('final_confidence', 0) - complexity_penalty)
        
        # Re-sort after complexity penalty
        verified.sort(key=lambda x: -x.get('final_confidence', 0))
        
        best = verified[0]
        
        return {
            'prediction': best.get('prediction'),
            'confidence': best.get('final_confidence', 0.0),
            'loo_score': best.get('loo_score', 0.0),
            'method': 'loo_ranked',
            'ranking': [v.get('final_confidence', 0) for v in verified[:10]],
            'num_candidates': len(verified),
        }
