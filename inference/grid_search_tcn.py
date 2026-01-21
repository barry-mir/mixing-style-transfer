"""
Grid search for TCN-based style transfer hyperparameters.

Tests different combinations of:
- Optimizer: Adam, AdamW, SGD
- Learning rate: 0.0001, 0.0005, 0.001, 0.005
- Number of steps: 300, 500, 1000
- TCN hidden channels: 32, 64, 128
- Receptive field: 1.0s, 2.0s, 3.0s
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import random
from pathlib import Path
from tqdm import tqdm
import sys
import itertools
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import MixingStyleEncoder
from mixing_utils import MixingFeatureExtractor
from musdb_dataset import MUSDB18EmbeddingDataset
from tcn_mixer import TCNMixer


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
        stems_device_cpu = {k: v.cpu() for k, v in stems_dict.items()}
        mixture_cpu = mixture_audio.cpu()

        # Extract features WITH gradients (no torch.no_grad!)
        mixing_features = feature_extractor.extract_all_features(stems_device_cpu, mixture_cpu)

        # Move everything to device
        stems_device = {k: v.unsqueeze(0).to(device) for k, v in stems_dict.items()}
        mixing_features = mixing_features.unsqueeze(0).to(device)

        # Compute embedding with gradients
        embedding = model(stems_device, mixing_features)

    return embedding.squeeze(0)


def optimize_tcn_style_transfer(
    tcn,
    stems_input,
    target_emb,
    mixing_model,
    feature_extractor,
    device,
    optimizer_name='Adam',
    num_steps=500,
    lr=0.001,
    verbose=False
):
    """Optimize TCN parameters to match target embedding."""
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
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(tcn.parameters(), lr=lr)
    elif optimizer_name == 'AdamW':
        optimizer = optim.AdamW(tcn.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(tcn.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    distances = []
    best_distance = float('inf')

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
            requires_grad=True
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

        # Backward pass
        loss.backward()
        optimizer.step()

    return {
        'distances': distances,
        'final_distance': best_distance,
        'initial_distance': distances[0],
        'improvement': distances[0] - best_distance,
        'improvement_pct': ((distances[0] - best_distance) / distances[0]) * 100 if distances[0] > 0 else 0,
        'converged': best_distance < distances[0] * 0.8
    }


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

    return model


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Grid search for TCN hyperparameters')
    parser.add_argument('--checkpoint', type=str, default='/nas/mixing-representation/checkpoints_baseline/best_model.pt')
    parser.add_argument('--musdb_path', type=str, default='/nas/MUSDB18')
    parser.add_argument('--output_file', type=str, default='tcn_grid_search_results.json')
    parser.add_argument('--num_pairs', type=int, default=5, help='Number of test pairs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("TCN Hyperparameter Grid Search")
    print("="*80)
    print(f"Device: {device}")
    print(f"Test pairs: {args.num_pairs}")
    print(f"Random seed: {args.seed}")
    print()

    # Define grid
    grid = {
        'optimizer': ['Adam', 'AdamW'],
        'lr': [0.0005, 0.001, 0.002],
        'num_steps': [300, 500],
        'hidden_channels': [64, 128],
        'receptive_field': [1.5, 2.0, 3.0]
    }

    print("Grid search space:")
    for key, values in grid.items():
        print(f"  {key}: {values}")

    total_configs = np.prod([len(v) for v in grid.values()])
    print(f"\nTotal configurations: {total_configs}")
    print(f"Total experiments: {total_configs * args.num_pairs}")
    print()

    # Load mixing model
    print("Loading mixing model...")
    mixing_model = load_mixing_model(args.checkpoint, device)

    # Initialize feature extractor
    print("Initializing feature extractor...")
    feature_extractor = MixingFeatureExtractor(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512
    )

    # Load dataset
    print(f"Loading MUSDB18 train set...")
    dataset = MUSDB18EmbeddingDataset(
        data_path=args.musdb_path,
        split='train',
        segment_duration=10.0,
        segment_offset=0.0
    )

    # Pick test pairs
    print(f"Selecting {args.num_pairs} test pairs...")
    all_indices = list(range(len(dataset)))
    test_pairs = []
    for _ in range(args.num_pairs):
        idx_a, idx_b = random.sample(all_indices, 2)
        test_pairs.append((idx_a, idx_b))

    print(f"Test pairs:")
    for i, (idx_a, idx_b) in enumerate(test_pairs):
        _, _, name_a, _ = dataset[idx_a]
        _, _, name_b, _ = dataset[idx_b]
        print(f"  Pair {i+1}: {name_a} → {name_b}")
    print()

    # Precompute embeddings for all test pairs
    print("Precomputing embeddings for test pairs...")
    pair_data = []
    for idx_input, idx_target in test_pairs:
        stems_input, mixture_input, name_input, _ = dataset[idx_input]
        stems_target, mixture_target, name_target, _ = dataset[idx_target]

        input_emb = compute_mixing_embedding(mixture_input, stems_input, mixing_model, feature_extractor, device)
        target_emb = compute_mixing_embedding(mixture_target, stems_target, mixing_model, feature_extractor, device)

        pair_data.append({
            'stems_input': stems_input,
            'target_emb': target_emb,
            'name_input': name_input,
            'name_target': name_target
        })

    # Run grid search
    results = []
    config_id = 0

    for optimizer_name in grid['optimizer']:
        for lr in grid['lr']:
            for num_steps in grid['num_steps']:
                for hidden_channels in grid['hidden_channels']:
                    for receptive_field in grid['receptive_field']:
                        config_id += 1

                        config = {
                            'config_id': config_id,
                            'optimizer': optimizer_name,
                            'lr': lr,
                            'num_steps': num_steps,
                            'hidden_channels': hidden_channels,
                            'receptive_field': receptive_field
                        }

                        print(f"[{config_id}/{total_configs}] Testing: optimizer={optimizer_name}, lr={lr}, steps={num_steps}, hidden={hidden_channels}, rf={receptive_field}s")

                        pair_results = []

                        for pair_idx, pair in enumerate(pair_data):
                            # Create fresh TCN for this config
                            target_rf_samples = int(receptive_field * 44100)
                            kernel_size = 15
                            num_blocks = int(np.ceil(np.log2(target_rf_samples / kernel_size)))

                            tcn = TCNMixer(
                                in_channels=8,
                                hidden_channels=hidden_channels,
                                num_blocks=num_blocks,
                                kernel_size=kernel_size
                            )

                            # Run optimization
                            result = optimize_tcn_style_transfer(
                                tcn,
                                pair['stems_input'],
                                pair['target_emb'],
                                mixing_model,
                                feature_extractor,
                                device,
                                optimizer_name=optimizer_name,
                                num_steps=num_steps,
                                lr=lr,
                                verbose=False
                            )

                            pair_results.append({
                                'pair_idx': pair_idx + 1,
                                'input_track': pair['name_input'],
                                'target_track': pair['name_target'],
                                'initial_distance': result['initial_distance'],
                                'final_distance': result['final_distance'],
                                'improvement': result['improvement'],
                                'improvement_pct': result['improvement_pct'],
                                'converged': result['converged']
                            })

                        # Aggregate results
                        avg_improvement = np.mean([r['improvement'] for r in pair_results])
                        avg_improvement_pct = np.mean([r['improvement_pct'] for r in pair_results])
                        avg_final_distance = np.mean([r['final_distance'] for r in pair_results])
                        num_converged = sum([r['converged'] for r in pair_results])

                        config['avg_improvement'] = avg_improvement
                        config['avg_improvement_pct'] = avg_improvement_pct
                        config['avg_final_distance'] = avg_final_distance
                        config['num_converged'] = num_converged
                        config['pair_results'] = pair_results

                        results.append(config)

                        print(f"  → Avg improvement: {avg_improvement:.4f} ({avg_improvement_pct:.1f}%), Final dist: {avg_final_distance:.4f}, Converged: {num_converged}/{args.num_pairs}")
                        print()

    # Save results
    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'grid': grid,
            'num_pairs': args.num_pairs,
            'seed': args.seed,
            'results': results
        }, f, indent=2)

    # Print summary table
    print("\n" + "="*80)
    print("GRID SEARCH RESULTS SUMMARY")
    print("="*80)
    print(f"{'ID':<4} {'Opt':<6} {'LR':<7} {'Steps':<6} {'Hidden':<7} {'RF':<5} {'Avg Imp%':<10} {'Avg Final':<11} {'Conv':<5}")
    print("-"*80)

    # Sort by average improvement percentage (descending)
    sorted_results = sorted(results, key=lambda x: x['avg_improvement_pct'], reverse=True)

    for r in sorted_results:
        print(f"{r['config_id']:<4} {r['optimizer']:<6} {r['lr']:<7.4f} {r['num_steps']:<6} "
              f"{r['hidden_channels']:<7} {r['receptive_field']:<5.1f} "
              f"{r['avg_improvement_pct']:<10.2f} {r['avg_final_distance']:<11.4f} "
              f"{r['num_converged']}/{args.num_pairs}")

    print("-"*80)

    # Best configuration
    best = sorted_results[0]
    print(f"\nBEST CONFIGURATION (Config ID {best['config_id']}):")
    print(f"  Optimizer: {best['optimizer']}")
    print(f"  Learning rate: {best['lr']}")
    print(f"  Steps: {best['num_steps']}")
    print(f"  Hidden channels: {best['hidden_channels']}")
    print(f"  Receptive field: {best['receptive_field']}s")
    print(f"  Average improvement: {best['avg_improvement']:.4f} ({best['avg_improvement_pct']:.2f}%)")
    print(f"  Average final distance: {best['avg_final_distance']:.4f}")
    print(f"  Converged: {best['num_converged']}/{args.num_pairs}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
