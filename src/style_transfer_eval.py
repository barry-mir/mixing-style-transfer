"""
ST-ITO Style Transfer Integration for Evaluation.

This module provides utilities for evaluating mixing representation embeddings
using ST-ITO (Style Transfer with Inference-Time Optimization) to test if
tracks with similar embeddings produce high-quality style transfers.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import soundfile as sf

# Add st-ito to path
STITO_PATH = Path(__file__).parent.parent / 'st-ito'
sys.path.insert(0, str(STITO_PATH))

try:
    from st_ito.utils import load_param_model, get_param_embeds
    from st_ito.style_transfer import run_es
    from st_ito import effects
    import pedalboard
except ImportError as e:
    print(f"Warning: ST-ITO imports failed: {e}")
    print(f"Make sure st-ito is installed: pip install -e {STITO_PATH}")


class StyleTransferEvaluator:
    """
    Evaluator for mixing representation using ST-ITO style transfer.

    Workflow:
    1. Extract embeddings from test tracks using trained model
    2. Retrieve similar tracks from training set
    3. Transfer style from retrieved track to test track using ST-ITO
    4. Measure quality of transfer (AFx-Rep similarity, audio features)
    """

    def __init__(
        self,
        device: str = 'cuda',
        sample_rate: int = 44100,
        max_iters: int = 25,
        population_size: int = 64
    ):
        """
        Args:
            device: Device for inference ('cuda' or 'cpu')
            sample_rate: Sample rate for audio processing
            max_iters: Maximum CMA-ES iterations
            population_size: CMA-ES population size
        """
        self.device = device
        self.sample_rate = sample_rate
        self.max_iters = max_iters
        self.population_size = population_size

        # Load AFx-Rep model
        print("Loading AFx-Rep model...")
        self.afx_model = load_param_model(use_gpu=(device == 'cuda'))
        print("AFx-Rep model loaded successfully")

    def create_effect_chain(self, chain_type: str = 'standard') -> List:
        """
        Create effect chain for style transfer.

        Args:
            chain_type: Type of effect chain
                - 'standard': EQ + Compressor + Reverb + Limiter
                - 'minimal': Compressor + EQ only
                - 'creative': Full chain with delay and distortion

        Returns:
            List of pedalboard effects with parameter ranges
        """
        if chain_type == 'standard':
            return [
                {
                    'effect': pedalboard.Compressor,
                    'params': {
                        'threshold_db': (-40.0, -5.0),
                        'ratio': (1.5, 10.0),
                        'attack_ms': (1.0, 50.0),
                        'release_ms': (50.0, 500.0)
                    }
                },
                {
                    'effect': pedalboard.HighpassFilter,
                    'params': {
                        'cutoff_frequency_hz': (20.0, 200.0)
                    }
                },
                {
                    'effect': pedalboard.LowpassFilter,
                    'params': {
                        'cutoff_frequency_hz': (8000.0, 20000.0)
                    }
                },
                {
                    'effect': pedalboard.Reverb,
                    'params': {
                        'room_size': (0.0, 0.8),
                        'damping': (0.2, 0.8),
                        'wet_level': (0.0, 0.3),
                        'dry_level': (0.7, 1.0)
                    }
                },
                {
                    'effect': pedalboard.Limiter,
                    'params': {
                        'threshold_db': (-10.0, -0.5),
                        'release_ms': (50.0, 200.0)
                    }
                }
            ]
        elif chain_type == 'minimal':
            return [
                {
                    'effect': pedalboard.Compressor,
                    'params': {
                        'threshold_db': (-40.0, -5.0),
                        'ratio': (1.5, 8.0)
                    }
                },
                {
                    'effect': pedalboard.HighpassFilter,
                    'params': {
                        'cutoff_frequency_hz': (20.0, 150.0)
                    }
                }
            ]
        else:
            raise ValueError(f"Unknown chain type: {chain_type}")

    def run_style_transfer(
        self,
        input_audio: torch.Tensor,
        target_audio: torch.Tensor,
        effect_chain: Optional[List] = None,
        verbose: bool = True
    ) -> Dict:
        """
        Run ST-ITO style transfer optimization.

        Args:
            input_audio: Input audio to transform (2, T) stereo tensor
            target_audio: Reference audio with desired style (2, T) stereo tensor
            effect_chain: List of effects with param ranges. If None, uses 'standard'
            verbose: Whether to print progress

        Returns:
            Dictionary with:
                - 'output_audio': Transferred audio (2, T)
                - 'input_embedding': AFx-Rep embedding of input
                - 'target_embedding': AFx-Rep embedding of target
                - 'output_embedding': AFx-Rep embedding of output
                - 'initial_distance': Cosine distance before transfer
                - 'final_distance': Cosine distance after transfer
                - 'improvement': Improvement in similarity
                - 'best_params': Optimized effect parameters
        """
        if effect_chain is None:
            effect_chain = self.create_effect_chain('standard')

        # Convert to numpy for ST-ITO
        input_np = input_audio.cpu().numpy()
        target_np = target_audio.cpu().numpy()

        # Extract initial embeddings
        with torch.no_grad():
            input_emb = get_param_embeds(input_np, self.afx_model, self.sample_rate)
            target_emb = get_param_embeds(target_np, self.afx_model, self.sample_rate)

        # Compute initial distance (use mid channel)
        input_mid = input_emb['mid']  # (512,)
        target_mid = target_emb['mid']
        initial_distance = 1.0 - F.cosine_similarity(
            torch.from_numpy(input_mid).unsqueeze(0),
            torch.from_numpy(target_mid).unsqueeze(0)
        ).item()

        if verbose:
            print(f"  Initial cosine distance: {initial_distance:.4f}")
            print(f"  Running CMA-ES optimization ({self.max_iters} iterations)...")

        # Run ST-ITO optimization
        result = run_es(
            input_audio=input_np,
            target_audio=target_np,
            sample_rate=self.sample_rate,
            plugins=effect_chain,
            model=self.afx_model,
            embed_func=get_param_embeds,
            max_iters=self.max_iters,
            population_size=self.population_size,
            use_gpu=(self.device == 'cuda'),
            verbose=verbose
        )

        # Extract output embedding
        output_audio = result['output_audio']
        with torch.no_grad():
            output_emb = get_param_embeds(output_audio, self.afx_model, self.sample_rate)

        # Compute final distance
        output_mid = output_emb['mid']
        final_distance = 1.0 - F.cosine_similarity(
            torch.from_numpy(output_mid).unsqueeze(0),
            torch.from_numpy(target_mid).unsqueeze(0)
        ).item()

        improvement = initial_distance - final_distance

        if verbose:
            print(f"  Final cosine distance: {final_distance:.4f}")
            print(f"  Improvement: {improvement:.4f}")

        return {
            'output_audio': torch.from_numpy(output_audio).float(),
            'input_embedding': input_emb,
            'target_embedding': target_emb,
            'output_embedding': output_emb,
            'initial_distance': initial_distance,
            'final_distance': final_distance,
            'improvement': improvement,
            'best_params': result.get('best_params', {})
        }

    def compute_audio_features(self, audio: torch.Tensor) -> Dict[str, float]:
        """
        Compute audio features for quality assessment.

        Args:
            audio: Stereo audio tensor (2, T)

        Returns:
            Dictionary of audio features
        """
        audio_np = audio.cpu().numpy()

        # LUFS (loudness)
        import pyloudnorm as pyln
        meter = pyln.Meter(self.sample_rate)
        loudness = meter.integrated_loudness(audio_np.T)  # (T, 2) format

        # RMS
        rms = np.sqrt(np.mean(audio_np ** 2))

        # Crest factor
        peak = np.max(np.abs(audio_np))
        crest_factor = peak / (rms + 1e-8)

        # Spectral centroid (using librosa)
        import librosa
        centroids = []
        for ch in range(2):
            centroid = librosa.feature.spectral_centroid(
                y=audio_np[ch],
                sr=self.sample_rate,
                n_fft=2048,
                hop_length=512
            )
            centroids.append(np.mean(centroid))
        spectral_centroid = np.mean(centroids)

        return {
            'loudness_lufs': float(loudness),
            'rms': float(rms),
            'crest_factor': float(crest_factor),
            'spectral_centroid': float(spectral_centroid)
        }

    def evaluate_transfer_quality(
        self,
        input_audio: torch.Tensor,
        target_audio: torch.Tensor,
        output_audio: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate quality of style transfer.

        Compares audio features between target and output to assess
        how well the style was transferred beyond just embedding similarity.

        Args:
            input_audio: Original input (2, T)
            target_audio: Target reference (2, T)
            output_audio: Transferred output (2, T)

        Returns:
            Dictionary of quality metrics
        """
        # Compute features
        target_features = self.compute_audio_features(target_audio)
        output_features = self.compute_audio_features(output_audio)

        # Compute feature distances
        loudness_diff = abs(target_features['loudness_lufs'] - output_features['loudness_lufs'])
        rms_diff = abs(target_features['rms'] - output_features['rms'])
        crest_diff = abs(target_features['crest_factor'] - output_features['crest_factor'])
        centroid_diff = abs(target_features['spectral_centroid'] - output_features['spectral_centroid'])

        # Normalize centroid diff
        centroid_diff_norm = centroid_diff / (target_features['spectral_centroid'] + 1e-8)

        return {
            'loudness_diff_lufs': loudness_diff,
            'rms_diff': rms_diff,
            'crest_factor_diff': crest_diff,
            'spectral_centroid_diff': centroid_diff,
            'spectral_centroid_diff_norm': centroid_diff_norm,
            'target_features': target_features,
            'output_features': output_features
        }

    def save_audio(self, audio: torch.Tensor, path: str):
        """Save audio tensor to file."""
        audio_np = audio.cpu().numpy().T  # (T, 2) for soundfile
        sf.write(path, audio_np, self.sample_rate)


def test_style_transfer():
    """Test style transfer evaluator."""
    print("Testing StyleTransferEvaluator...")

    # Create evaluator
    evaluator = StyleTransferEvaluator(
        device='cuda' if torch.cuda.is_available() else 'cpu',
        sample_rate=44100,
        max_iters=5  # Small for testing
    )

    # Create dummy audio (2, 44100*5) = 5 seconds stereo
    print("\nCreating dummy audio...")
    input_audio = torch.randn(2, 44100 * 5)
    target_audio = torch.randn(2, 44100 * 5)

    # Run style transfer
    print("\nRunning style transfer...")
    result = evaluator.run_style_transfer(
        input_audio=input_audio,
        target_audio=target_audio,
        verbose=True
    )

    print(f"\nResults:")
    print(f"  Initial distance: {result['initial_distance']:.4f}")
    print(f"  Final distance: {result['final_distance']:.4f}")
    print(f"  Improvement: {result['improvement']:.4f}")

    # Evaluate quality
    print("\nEvaluating transfer quality...")
    quality = evaluator.evaluate_transfer_quality(
        input_audio=input_audio,
        target_audio=target_audio,
        output_audio=result['output_audio']
    )

    print(f"  Loudness diff: {quality['loudness_diff_lufs']:.2f} LUFS")
    print(f"  Spectral centroid diff: {quality['spectral_centroid_diff_norm']:.2%}")


if __name__ == '__main__':
    test_style_transfer()
