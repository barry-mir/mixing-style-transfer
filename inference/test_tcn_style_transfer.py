"""
Differentiable style transfer using TCN mixer.

Optimizes TCN parameters to minimize embedding distance between
processed input and target using gradient descent.
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import json
import soundfile as sf
import random
from pathlib import Path
from tqdm import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import MixingStyleEncoder
from mixing_utils import MixingFeatureExtractor
from musdb_dataset import MUSDB18EmbeddingDataset
from tcn_mixer import create_tcn_mixer


def compute_mixing_embedding(mixture_audio, stems_dict, model, feature_extractor, device, requires_grad=False):
    """Compute embedding using mixing representation model."""
    if not requires_grad:
        # No gradients needed - use CPU and detach for efficiency
        with torch.no_grad():
            stems_cpu = {k: v.detach().cpu() if v.is_cuda else v for k, v in stems_dict.items()}
            mixture_cpu = mixture_audio.detach().cpu() if mixture_audio.is_cuda else mixture_audio

            # Extract features on CPU
            mixing_features = feature_extractor.extract_all_features(stems_cpu, mixture_cpu)

            # Move to device for model
            stems_device = {k: v.unsqueeze(0).to(device) for k, v in stems_cpu.items()}
            mixing_features = mixing_features.unsqueeze(0).to(device)

            embedding = model(stems_device, mixing_features)
    else:
        # WITH gradients - keep everything on device, no detach!
        # Move stems to device (without detaching to preserve gradients)
        stems_device_cpu = {k: v.cpu() for k, v in stems_dict.items()}
        mixture_cpu = mixture_audio.cpu()

        # Extract features WITH gradients (no torch.no_grad!)
        mixing_features = feature_extractor.extract_all_features(stems_device_cpu, mixture_cpu)

        # Move everything to device
        stems_device = {k: v.unsqueeze(0).to(device) for k, v in stems_dict.items()}  # Original stems with grad
        mixing_features = mixing_features.unsqueeze(0).to(device)

        # Compute embedding with gradients
        embedding = model(stems_device, mixing_features)

    return embedding.squeeze(0)  # (512,)


def load_mixing_model(checkpoint_path, device):
    """Load trained mixing representation model."""
    model = MixingStyleEncoder(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512,
        n_mels=80,
        split_size=16,
        overlap=8,
        channels=8,
        embed_dim=512,
        feature_dim=64
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded from epoch {checkpoint['epoch']}")
    return model


def optimize_tcn_style_transfer(
    tcn,
    stems_input,
    target_emb,
    mixing_model,
    feature_extractor,
    device,
    num_steps=200,
    lr=0.01,
    verbose=True
):
    """
    Optimize TCN parameters to match target embedding.

    Args:
        tcn: TCNMixer model
        stems_input: Dict of input stems
        target_emb: Target embedding (512,)
        mixing_model: Trained mixing model
        feature_extractor: Feature extractor
        device: Device
        num_steps: Number of optimization steps
        lr: Learning rate
        verbose: Print progress

    Returns:
        dict with:
            - processed_stems: Final processed stems
            - processed_mixture: Final processed mixture
            - distances: List of distances over optimization
            - final_distance: Final distance to target
    """
    # Normalize target embedding
    target_emb = F.normalize(target_emb, p=2, dim=0)

    # Stack stems for TCN: (8, T)
    stem_order = ['vocals', 'bass', 'drums', 'other']
    stems_list = [stems_input[name] for name in stem_order]
    stacked_stems = torch.cat(stems_list, dim=0).unsqueeze(0).to(device)  # (1, 8, T)
    stacked_stems.requires_grad = False

    # Initialize TCN
    tcn = tcn.to(device)
    tcn.train()

    # Optimizer
    optimizer = optim.Adam(tcn.parameters(), lr=lr)

    distances = []
    best_distance = float('inf')
    best_state = None

    # Debug: Check if TCN is initialized to identity
    if verbose:
        with torch.no_grad():
            test_processed = tcn(stacked_stems)
            diff = (test_processed - stacked_stems).abs().mean().item()
            print(f"Initial TCN output difference from input: {diff:.6f} (should be very small)")

    for step in range(num_steps):
        optimizer.zero_grad()

        # Process stems through TCN
        processed_stacked = tcn(stacked_stems)  # (1, 8, T)

        # Unstack processed stems
        processed_stems = {}
        for i, stem_name in enumerate(stem_order):
            processed_stems[stem_name] = processed_stacked[0, i*2:(i+1)*2, :]  # (2, T)

        # Sum to create mixture
        processed_mixture = sum(processed_stems.values())  # (2, T)

        # Compute embedding WITH gradients
        emb = compute_mixing_embedding(
            processed_mixture,
            processed_stems,
            mixing_model,
            feature_extractor,
            device,
            requires_grad=True  # Enable gradients for backprop
        )

        # Normalize
        emb = F.normalize(emb, p=2, dim=0)

        # Compute loss (negative cosine similarity)
        loss = 1.0 - F.cosine_similarity(emb.unsqueeze(0), target_emb.unsqueeze(0))

        # Compute distance for tracking (without grad)
        with torch.no_grad():
            distance = loss.item()
            distances.append(distance)

            # Track best
            if distance < best_distance:
                best_distance = distance
                best_state = {
                    'processed_stems': {k: v.detach().cpu() for k, v in processed_stems.items()},
                    'processed_mixture': processed_mixture.detach().cpu(),
                    'tcn_state': tcn.state_dict().copy()
                }

        # Backward pass - minimize distance
        loss.backward()
        optimizer.step()

        if verbose and (step % 10 == 0 or step == num_steps - 1):
            print(f"Step {step:3d}/{num_steps}: distance = {distance:.4f}, "
                  f"loss = {loss.item():.4f}, best = {best_distance:.4f}")

    return {
        'processed_stems': best_state['processed_stems'],
        'processed_mixture': best_state['processed_mixture'],
        'distances': distances,
        'final_distance': best_distance,
        'converged': best_distance < distances[0] * 0.8  # 20% improvement
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='TCN-based style transfer test')
    parser.add_argument('--checkpoint', type=str, default='/nas/mixing-representation/checkpoints_baseline/best_model.pt')
    parser.add_argument('--musdb_path', type=str, default='/nas/MUSDB18')
    parser.add_argument('--output_dir', type=str, default='tcn_style_transfer_results')
    parser.add_argument('--num_pairs', type=int, default=10)
    parser.add_argument('--num_steps', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--segment_duration', type=float, default=10.0)
    parser.add_argument('--receptive_field', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducible pair selection')
    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("TCN-Based Differentiable Style Transfer")
    print("="*80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Number of pairs: {args.num_pairs}")
    print(f"Optimization steps: {args.num_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Segment duration: {args.segment_duration}s")
    print(f"Device: {device}")
    print()

    # Load mixing model
    print("Loading mixing model...")
    mixing_model = load_mixing_model(args.checkpoint, device)

    # Initialize feature extractor
    print("Initializing mixing feature extractor...")
    feature_extractor = MixingFeatureExtractor(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512
    )

    # Load dataset
    print(f"\nLoading MUSDB18 train set...")
    dataset = MUSDB18EmbeddingDataset(
        data_path=args.musdb_path,
        split='train',
        segment_duration=args.segment_duration,
        segment_offset=0.0
    )
    print(f"Dataset size: {len(dataset)} tracks")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / 'audio'
    audio_dir.mkdir(exist_ok=True)

    # Pick random pairs
    print(f"\nPicking {args.num_pairs} random pairs...")
    all_indices = list(range(len(dataset)))
    pairs = []
    for _ in range(args.num_pairs):
        idx_a, idx_b = random.sample(all_indices, 2)
        pairs.append((idx_a, idx_b))
    print(f"Selected {len(pairs)} pairs")

    # Run style transfer on each pair
    results = []

    for pair_idx, (idx_input, idx_target) in enumerate(tqdm(pairs, desc="Processing pairs")):
        print(f"\n{'='*80}")
        print(f"Pair {pair_idx + 1}/{args.num_pairs}")
        print(f"{'='*80}")

        # Load tracks
        stems_input, mixture_input, name_input, _ = dataset[idx_input]
        stems_target, mixture_target, name_target, _ = dataset[idx_target]

        print(f"Input track: {name_input}")
        print(f"Target track: {name_target}")

        # Compute initial and target embeddings
        print("Computing embeddings...")
        input_emb = compute_mixing_embedding(mixture_input, stems_input, mixing_model, feature_extractor, device)
        target_emb = compute_mixing_embedding(mixture_target, stems_target, mixing_model, feature_extractor, device)

        # Normalize
        input_emb = F.normalize(input_emb, p=2, dim=0)
        target_emb = F.normalize(target_emb, p=2, dim=0)

        # Compute initial distance
        initial_distance = 1.0 - F.cosine_similarity(
            input_emb.unsqueeze(0),
            target_emb.unsqueeze(0)
        ).item()

        print(f"Initial embedding distance: {initial_distance:.4f}")

        # Create TCN mixer
        print(f"\nInitializing TCN mixer (receptive field: {args.receptive_field}s)...")
        tcn = create_tcn_mixer(receptive_field_seconds=args.receptive_field)

        # Optimize
        print(f"Optimizing for {args.num_steps} steps...")
        result = optimize_tcn_style_transfer(
            tcn,
            stems_input,
            target_emb,
            mixing_model,
            feature_extractor,
            device,
            num_steps=args.num_steps,
            lr=args.lr,
            verbose=True
        )

        final_distance = result['final_distance']
        improvement = initial_distance - final_distance
        improvement_pct = (improvement / initial_distance) * 100 if initial_distance > 0 else 0

        print(f"\n{'='*80}")
        print(f"Results for pair {pair_idx + 1}")
        print(f"{'='*80}")
        print(f"Initial distance: {initial_distance:.4f}")
        print(f"Final distance: {final_distance:.4f}")
        print(f"Improvement: {improvement:.4f} ({improvement_pct:.1f}%)")
        print(f"Converged: {'Yes' if result['converged'] else 'No'}")

        # Save audio files
        input_path = audio_dir / f"pair{pair_idx+1:02d}_input_{name_input}.wav"
        target_path = audio_dir / f"pair{pair_idx+1:02d}_target_{name_target}.wav"
        output_path = audio_dir / f"pair{pair_idx+1:02d}_output_{name_input}_to_{name_target}.wav"

        # Save as (T, 2) for soundfile
        sf.write(str(input_path), mixture_input.cpu().numpy().T, 44100)
        sf.write(str(target_path), mixture_target.cpu().numpy().T, 44100)
        sf.write(str(output_path), result['processed_mixture'].cpu().numpy().T, 44100)

        # Store results
        results.append({
            'pair_index': pair_idx + 1,
            'input_track': name_input,
            'target_track': name_target,
            'initial_distance': initial_distance,
            'final_distance': final_distance,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'converged': result['converged'],
            'distances': result['distances'],
            'output_file': str(output_path.name)
        })

    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print("Summary")
    print(f"{'='*80}")
    avg_improvement = np.mean([r['improvement'] for r in results])
    avg_improvement_pct = np.mean([r['improvement_pct'] for r in results])
    num_converged = sum([r['converged'] for r in results])

    print(f"Average improvement: {avg_improvement:.4f} ({avg_improvement_pct:.1f}%)")
    print(f"Converged: {num_converged}/{len(results)} pairs")
    print(f"\nResults saved to: {results_path}")
    print(f"Audio saved to: {audio_dir}")


if __name__ == "__main__":
    main()
