"""
EnhancedInference: Combines TEPS, ConsistencyVerifier, and base RLAN

This is the main entry point for enhanced inference that attempts to
generalize to never-seen-before rules.

INFERENCE STRATEGY:
1. Try TEPS program search first (if enabled)
   - If found program explains all training pairs, use it
   
2. Get RLAN neural prediction

3. Verify RLAN prediction with ConsistencyVerifier
   - If consistent with training patterns, accept
   - If not, try TEPS partial match as alternative
   
4. Return best prediction with confidence

This module WRAPS the base RLAN - it doesn't modify it.

Author: AI Research Assistant
Date: January 2026
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

from .teps import TEPS, TEPSConfig
from .consistency_verifier import ConsistencyVerifier, ConsistencyConfig


@dataclass
class EnhancedInferenceConfig:
    """Configuration for enhanced inference."""
    # Master switch
    enabled: bool = True
    
    # TEPS settings
    use_teps: bool = True
    teps_config: TEPSConfig = None
    
    # Consistency verification settings
    use_verification: bool = True
    verification_config: ConsistencyConfig = None
    
    # Fallback behavior
    prefer_teps_over_neural: bool = True  # If TEPS succeeds, prefer it
    verification_rejection_threshold: float = 0.3  # Reject if below this
    
    def __post_init__(self):
        if self.teps_config is None:
            self.teps_config = TEPSConfig()
        if self.verification_config is None:
            self.verification_config = ConsistencyConfig()


class EnhancedInference(nn.Module):
    """
    Enhanced inference pipeline combining:
    - TEPS (program synthesis)
    - ConsistencyVerifier
    - Base RLAN neural prediction
    
    Usage:
        enhanced = EnhancedInference(config)
        
        result = enhanced.predict(
            rlan_model=model,
            test_input=test_grid,
            train_inputs=[in1, in2, ...],
            train_outputs=[out1, out2, ...],
        )
        
        prediction = result['prediction']
        method = result['method']  # 'teps', 'neural', 'fallback'
        confidence = result['confidence']
    """
    
    def __init__(self, config: EnhancedInferenceConfig = None):
        super().__init__()
        
        self.config = config or EnhancedInferenceConfig()
        
        # Initialize modules
        if self.config.use_teps:
            self.teps = TEPS(self.config.teps_config)
        else:
            self.teps = None
            
        if self.config.use_verification:
            self.verifier = ConsistencyVerifier(self.config.verification_config)
        else:
            self.verifier = None
    
    def _to_numpy(self, g: Union[torch.Tensor, np.ndarray, list]) -> np.ndarray:
        """Convert grid to numpy."""
        if isinstance(g, torch.Tensor):
            return g.detach().cpu().numpy()
        return np.array(g)
    
    def _to_tensor(
        self, 
        g: Union[torch.Tensor, np.ndarray, list], 
        device: torch.device = None,
    ) -> torch.Tensor:
        """Convert grid to tensor."""
        if isinstance(g, torch.Tensor):
            return g if device is None else g.to(device)
        t = torch.tensor(np.array(g), dtype=torch.long)
        return t if device is None else t.to(device)
    
    @torch.no_grad()
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
        Run enhanced inference.
        
        Args:
            rlan_model: The base RLAN model
            test_input: Test input grid
            train_inputs: List of training input grids
            train_outputs: List of training output grids
            device: Device for computation
            **rlan_kwargs: Additional args for RLAN predict method
            
        Returns:
            result: Dict with:
                - prediction: The final prediction (numpy array)
                - method: str - 'teps', 'neural', or 'fallback'
                - confidence: float - confidence score
                - teps_result: Optional TEPS result
                - verification_result: Optional verification result
        """
        if not self.config.enabled:
            # Just use base RLAN
            return self._rlan_predict(
                rlan_model, test_input, train_inputs, train_outputs, 
                device, **rlan_kwargs
            )
        
        # Convert inputs
        test_np = self._to_numpy(test_input)
        train_in_np = [self._to_numpy(g) for g in train_inputs]
        train_out_np = [self._to_numpy(g) for g in train_outputs]
        
        result = {
            'prediction': None,
            'method': None,
            'confidence': 0.0,
            'teps_result': None,
            'verification_result': None,
        }
        
        # Step 1: Try TEPS program search
        teps_prediction = None
        if self.teps is not None and self.config.use_teps:
            teps_result = self.teps.search(
                test_input=test_np,
                train_inputs=train_in_np,
                train_outputs=train_out_np,
            )
            result['teps_result'] = teps_result
            
            if teps_result['success']:
                teps_prediction = teps_result['prediction']
                if self.config.prefer_teps_over_neural:
                    result['prediction'] = teps_prediction
                    result['method'] = 'teps'
                    result['confidence'] = 1.0  # TEPS match is exact
                    return result
        
        # Step 2: Get RLAN neural prediction
        neural_result = self._rlan_predict(
            rlan_model, test_input, train_inputs, train_outputs,
            device, **rlan_kwargs
        )
        neural_prediction = neural_result['prediction']
        
        # Step 3: Verify RLAN prediction
        if self.verifier is not None and self.config.use_verification:
            test_tensor = self._to_tensor(test_input, device)
            pred_tensor = self._to_tensor(neural_prediction, device)
            train_in_tensors = [self._to_tensor(g, device) for g in train_inputs]
            train_out_tensors = [self._to_tensor(g, device) for g in train_outputs]
            
            verification = self.verifier.verify(
                test_input=test_tensor,
                prediction=pred_tensor,
                train_inputs=train_in_tensors,
                train_outputs=train_out_tensors,
            )
            result['verification_result'] = verification
            
            # Decide based on verification
            if verification['is_consistent']:
                result['prediction'] = neural_prediction
                result['method'] = 'neural'
                result['confidence'] = verification['score']
                return result
            elif verification['score'] < self.config.verification_rejection_threshold:
                # Neural prediction is rejected - try TEPS partial if available
                if teps_prediction is not None:
                    result['prediction'] = teps_prediction
                    result['method'] = 'teps_fallback'
                    result['confidence'] = 0.5
                    return result
        
        # Step 4: Return neural prediction (with low confidence if verification failed)
        result['prediction'] = neural_prediction
        result['method'] = 'neural'
        verification_result = result.get('verification_result')
        if verification_result is not None and isinstance(verification_result, dict):
            result['confidence'] = verification_result.get('score', 0.5)
        else:
            result['confidence'] = 0.5
        
        return result
    
    def _rlan_predict(
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
        
        # Prepare inputs for RLAN
        test_tensor = self._to_tensor(test_input, device)
        
        # RLAN expects specific input format - try different methods
        try:
            # Method 1: Use predict method if available
            if hasattr(rlan_model, 'predict'):
                # Format inputs as RLAN expects
                train_in_list = [self._to_tensor(g, device) for g in train_inputs]
                train_out_list = [self._to_tensor(g, device) for g in train_outputs]
                
                # Try calling predict with various input formats
                try:
                    prediction = rlan_model.predict(
                        input_grid=test_tensor.unsqueeze(0) if test_tensor.dim() == 2 else test_tensor,
                        train_inputs=torch.stack(train_in_list).unsqueeze(0),
                        train_outputs=torch.stack(train_out_list).unsqueeze(0),
                        **rlan_kwargs,
                    )
                except Exception:
                    # Try simpler format
                    prediction = rlan_model.predict(
                        test_tensor,
                        train_in_list,
                        train_out_list,
                        **rlan_kwargs,
                    )
                    
            # Method 2: Use forward method
            elif hasattr(rlan_model, 'forward'):
                train_in_list = [self._to_tensor(g, device) for g in train_inputs]
                train_out_list = [self._to_tensor(g, device) for g in train_outputs]
                
                prediction = rlan_model(
                    input_grid=test_tensor.unsqueeze(0) if test_tensor.dim() == 2 else test_tensor,
                    train_inputs=torch.stack(train_in_list).unsqueeze(0),
                    train_outputs=torch.stack(train_out_list).unsqueeze(0),
                    **rlan_kwargs,
                )
                
                # If output is logits, convert to class predictions
                if isinstance(prediction, dict):
                    prediction = prediction.get('logits', prediction.get('output'))
                if isinstance(prediction, torch.Tensor) and prediction.dim() >= 3:
                    if prediction.shape[-3] == 10:  # Logits with 10 classes
                        prediction = prediction.argmax(dim=-3)
            else:
                raise RuntimeError("RLAN model has neither predict nor forward method")
                
            # Convert to numpy
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.detach().cpu().squeeze().numpy()
            
            return {
                'prediction': prediction,
                'method': 'neural',
                'confidence': 0.5,
            }
            
        except Exception as e:
            # Fallback: return zeros
            print(f"Warning: RLAN prediction failed: {e}")
            return {
                'prediction': np.zeros_like(self._to_numpy(test_input)),
                'method': 'error',
                'confidence': 0.0,
                'error': str(e),
            }
    
    def forward(self, *args, **kwargs):
        """Forward calls predict()."""
        return self.predict(*args, **kwargs)


def run_enhanced_inference(
    rlan_model: nn.Module,
    test_input: Union[torch.Tensor, np.ndarray],
    train_inputs: List[Union[torch.Tensor, np.ndarray]],
    train_outputs: List[Union[torch.Tensor, np.ndarray]],
    config: EnhancedInferenceConfig = None,
    device: torch.device = None,
    **rlan_kwargs,
) -> Dict[str, Any]:
    """
    Convenience function for enhanced inference.
    
    Creates EnhancedInference instance and runs prediction.
    
    Args:
        rlan_model: The base RLAN model
        test_input: Test input grid
        train_inputs: List of training input grids
        train_outputs: List of training output grids
        config: Optional EnhancedInferenceConfig
        device: Device for computation
        **rlan_kwargs: Additional args for RLAN
        
    Returns:
        result: Dict with prediction and metadata
    """
    enhanced = EnhancedInference(config or EnhancedInferenceConfig())
    
    return enhanced.predict(
        rlan_model=rlan_model,
        test_input=test_input,
        train_inputs=train_inputs,
        train_outputs=train_outputs,
        device=device,
        **rlan_kwargs,
    )
