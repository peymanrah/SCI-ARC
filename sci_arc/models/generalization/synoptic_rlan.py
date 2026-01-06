"""
Synoptic-RLAN v3.0 (S-RLAN) Enhanced Inference Pipeline

Combines all generalization modules into a unified inference pipeline:
1. NS-TEPS: Neuro-symbolic program search
2. TEPS: Basic program synthesis
3. HASR: Test-time adaptation
4. LOO Verifier: Generalization verification
5. Base RLAN: Neural fallback

INFERENCE STRATEGY:
1. Run NS-TEPS and TEPS in parallel to find candidate programs
2. Collect successful programs as pseudo-labels for HASR
3. Run HASR adaptation if enough pseudo-labels
4. Get neural prediction from RLAN
5. Rank all candidates using LOO Verifier
6. Return best prediction with confidence

Author: AI Research Assistant
Date: January 2026
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

from .teps import TEPS, TEPSConfig
from .ns_teps import NSTEPS, NSTEPSConfig
from .hasr import HASR, HASRConfig
from .loo_verifier import LOOVerifier, LOOVerifierConfig, VerifierRanker
from .consistency_verifier import ConsistencyVerifier, ConsistencyConfig


@dataclass
class SRLANConfig:
    """Configuration for Synoptic-RLAN v3.0 inference."""
    # Master switch
    enabled: bool = True
    
    # Module switches
    use_ns_teps: bool = True
    use_teps: bool = True
    use_hasr: bool = True
    use_loo_verifier: bool = True
    use_consistency_verifier: bool = True
    
    # Module configs
    ns_teps_config: NSTEPSConfig = None
    teps_config: TEPSConfig = None
    hasr_config: HASRConfig = None
    loo_config: LOOVerifierConfig = None
    consistency_config: ConsistencyConfig = None
    
    # Strategy
    prefer_symbolic_over_neural: bool = True
    require_loo_verification: bool = False
    min_confidence_threshold: float = 0.3
    
    def __post_init__(self):
        if self.ns_teps_config is None:
            self.ns_teps_config = NSTEPSConfig()
        if self.teps_config is None:
            self.teps_config = TEPSConfig()
        if self.hasr_config is None:
            self.hasr_config = HASRConfig()
        if self.loo_config is None:
            self.loo_config = LOOVerifierConfig()
        if self.consistency_config is None:
            self.consistency_config = ConsistencyConfig()


class SynopticRLAN(nn.Module):
    """
    Synoptic-RLAN v3.0: Full generalization pipeline.
    
    This is the main integration point for all generalization modules.
    """
    
    def __init__(self, config: SRLANConfig = None):
        super().__init__()
        self.config = config or SRLANConfig()
        
        # Initialize modules
        if self.config.use_ns_teps:
            self.ns_teps = NSTEPS(self.config.ns_teps_config)
        else:
            self.ns_teps = None
            
        if self.config.use_teps:
            self.teps = TEPS(self.config.teps_config)
        else:
            self.teps = None
            
        if self.config.use_hasr:
            self.hasr = HASR(self.config.hasr_config)
        else:
            self.hasr = None
            
        if self.config.use_loo_verifier:
            self.loo_verifier = VerifierRanker(self.config.loo_config)
        else:
            self.loo_verifier = None
            
        if self.config.use_consistency_verifier:
            self.consistency_verifier = ConsistencyVerifier(self.config.consistency_config)
        else:
            self.consistency_verifier = None
    
    def predict(
        self,
        rlan_model: nn.Module,
        test_input: Union[torch.Tensor, np.ndarray],
        train_inputs: List[Union[torch.Tensor, np.ndarray]],
        train_outputs: List[Union[torch.Tensor, np.ndarray]],
        device: torch.device = None,
        **rlan_kwargs,
    ) -> Dict[str, Any]:
        """
        Run full S-RLAN inference pipeline.
        
        Returns:
            Dict with:
            - prediction: Best prediction
            - confidence: Confidence score
            - method: Method that produced the prediction
            - candidates: All candidate predictions
            - metadata: Additional debug info
        """
        if not self.config.enabled:
            # Fallback to base RLAN
            return self._neural_predict(rlan_model, test_input, train_inputs, train_outputs, device, **rlan_kwargs)
        
        # Convert inputs to numpy
        test_np = self._to_numpy(test_input)
        train_inputs_np = [self._to_numpy(t) for t in train_inputs]
        train_outputs_np = [self._to_numpy(t) for t in train_outputs]
        
        candidates = []
        program_results = []
        
        # ============================================
        # STEP 1: Run NS-TEPS (Object-level search)
        # ============================================
        if self.ns_teps is not None and self.config.use_ns_teps:
            ns_result = self.ns_teps.search(
                test_input=test_np,
                train_inputs=train_inputs_np,
                train_outputs=train_outputs_np,
            )
            
            if ns_result.get('prediction') is not None:
                candidates.append({
                    'prediction': ns_result['prediction'],
                    'program': ns_result.get('trace'),
                    'confidence': ns_result.get('confidence', 0.0),
                    'method': 'ns_teps',
                })
                program_results.append(ns_result)
        
        # ============================================
        # STEP 2: Run TEPS (Basic program search)
        # ============================================
        if self.teps is not None and self.config.use_teps:
            teps_result = self.teps.search(
                test_input=test_np,
                train_inputs=train_inputs_np,
                train_outputs=train_outputs_np,
            )
            
            if teps_result.get('prediction') is not None:
                candidates.append({
                    'prediction': teps_result['prediction'],
                    'program': teps_result.get('program'),
                    'confidence': teps_result.get('confidence', 0.0),
                    'method': 'teps',
                })
                program_results.append(teps_result)
        
        # ============================================
        # STEP 3: Run HASR (Test-time adaptation)
        # ============================================
        hasr_adapted = False
        if self.hasr is not None and self.config.use_hasr and program_results:
            # Collect pseudo-labels from successful programs
            pseudo_labels = self.hasr.collect_pseudo_labels(
                program_results=program_results,
                train_inputs=train_inputs_np,
                train_outputs=train_outputs_np,
            )
            
            if pseudo_labels:
                # Adapt the refinement module
                adapt_result = self.hasr.adapt(
                    pseudo_labels=pseudo_labels,
                    rlan_model=rlan_model,
                    device=device,
                )
                hasr_adapted = adapt_result.get('adapted', False)
        
        # ============================================
        # STEP 4: Get Neural Prediction
        # ============================================
        neural_result = self._neural_predict(
            rlan_model, test_input, train_inputs, train_outputs, device, **rlan_kwargs
        )
        neural_pred = neural_result.get('prediction')
        
        if neural_pred is not None:
            # Optionally refine with HASR
            if hasr_adapted and self.hasr is not None:
                refined = self.hasr.refine(
                    base_prediction=neural_pred,
                    input_grid=test_np,
                    rlan_model=rlan_model,
                    device=device,
                )
                candidates.append({
                    'prediction': refined['prediction'],
                    'program': None,
                    'confidence': 0.6 if refined.get('refined') else 0.5,
                    'method': 'neural_refined' if refined.get('refined') else 'neural',
                })
            else:
                candidates.append({
                    'prediction': neural_pred,
                    'program': None,
                    'confidence': 0.5,
                    'method': 'neural',
                })
        
        # ============================================
        # STEP 5: Rank with LOO Verifier
        # ============================================
        if not candidates:
            return {
                'prediction': None,
                'confidence': 0.0,
                'method': 'no_candidates',
                'candidates': [],
                'metadata': {'error': 'No predictions generated'},
            }
        
        if self.loo_verifier is not None and self.config.use_loo_verifier:
            ranked = self.loo_verifier.rank(
                candidates=candidates,
                train_inputs=train_inputs_np,
                train_outputs=train_outputs_np,
                test_input=test_np,
            )
            
            best_prediction = ranked.get('prediction')
            best_confidence = ranked.get('confidence', 0.0)
            best_method = ranked.get('method', 'loo_ranked')
        else:
            # Sort by confidence and pick best
            candidates.sort(key=lambda x: -x.get('confidence', 0))
            best = candidates[0]
            best_prediction = best['prediction']
            best_confidence = best['confidence']
            best_method = best['method']
        
        # ============================================
        # STEP 6: Apply confidence threshold
        # ============================================
        if best_confidence < self.config.min_confidence_threshold:
            # Low confidence - prefer neural as fallback
            for cand in candidates:
                if cand['method'] in ('neural', 'neural_refined'):
                    best_prediction = cand['prediction']
                    best_method = cand['method'] + '_fallback'
                    best_confidence = cand['confidence']
                    break
        
        return {
            'prediction': best_prediction,
            'confidence': best_confidence,
            'method': best_method,
            'candidates': candidates,
            'metadata': {
                'num_candidates': len(candidates),
                'hasr_adapted': hasr_adapted,
                'ns_teps_ran': self.ns_teps is not None and self.config.use_ns_teps,
                'teps_ran': self.teps is not None and self.config.use_teps,
            },
        }
    
    def _neural_predict(
        self,
        rlan_model: nn.Module,
        test_input: Union[torch.Tensor, np.ndarray],
        train_inputs: List[Union[torch.Tensor, np.ndarray]],
        train_outputs: List[Union[torch.Tensor, np.ndarray]],
        device: torch.device = None,
        **rlan_kwargs,
    ) -> Dict[str, Any]:
        """Get prediction from base RLAN model."""
        rlan_model.eval()
        
        if device is None:
            try:
                device = next(rlan_model.parameters()).device
            except StopIteration:
                device = torch.device('cpu')
        
        test_tensor = self._to_tensor(test_input, device)
        
        try:
            # Prepare inputs
            train_in_list = [self._to_tensor(g, device) for g in train_inputs]
            train_out_list = [self._to_tensor(g, device) for g in train_outputs]
            
            # Find max dimensions
            all_grids = train_in_list + train_out_list + [test_tensor]
            max_h = max(g.shape[0] for g in all_grids)
            max_w = max(g.shape[1] for g in all_grids)
            
            # Pad grids
            def pad_grid(g, h, w):
                padded = torch.full((h, w), 10, dtype=g.dtype, device=g.device)
                padded[:g.shape[0], :g.shape[1]] = g
                return padded
            
            train_in_padded = torch.stack([pad_grid(t, max_h, max_w) for t in train_in_list]).unsqueeze(0)
            train_out_padded = torch.stack([pad_grid(t, max_h, max_w) for t in train_out_list]).unsqueeze(0)
            test_padded = pad_grid(test_tensor, max_h, max_w).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                prediction = rlan_model.predict(
                    input_grid=test_padded,
                    train_inputs=train_in_padded,
                    train_outputs=train_out_padded,
                    **rlan_kwargs,
                )
            
            # Remove padding
            orig_h, orig_w = test_tensor.shape[0], test_tensor.shape[1]
            prediction = prediction.squeeze(0)[:orig_h, :orig_w].cpu().numpy()
            
            return {
                'success': True,
                'prediction': prediction,
                'confidence': 0.5,
            }
            
        except Exception as e:
            return {
                'success': False,
                'prediction': None,
                'error': str(e),
                'confidence': 0.0,
            }
    
    def _to_numpy(self, x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert to numpy array."""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.array(x)
    
    def _to_tensor(self, x: Union[torch.Tensor, np.ndarray], device: torch.device) -> torch.Tensor:
        """Convert to tensor on device."""
        if isinstance(x, torch.Tensor):
            return x.to(device)
        return torch.tensor(x, dtype=torch.long, device=device)


def run_srlan_inference(
    rlan_model: nn.Module,
    test_input: np.ndarray,
    train_inputs: List[np.ndarray],
    train_outputs: List[np.ndarray],
    config: SRLANConfig = None,
    device: torch.device = None,
) -> Dict[str, Any]:
    """
    Convenience function to run S-RLAN inference.
    """
    srlan = SynopticRLAN(config)
    if device is not None:
        srlan = srlan.to(device)
    return srlan.predict(rlan_model, test_input, train_inputs, train_outputs, device)
