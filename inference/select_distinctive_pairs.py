"""
Select distinctive track pairs from balanced MUSDB18 subset.

Computes embeddings for all tracks and selects pairs with lowest cosine similarity
(most distinctive mixing styles).
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import sys
from pathlib import Path
from tqdm import tqdm
import itertools

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import MixingStyleEncoder
from mixing_utils import MixingFeatureExtractor
from musdb_dataset import MUSDB18EmbeddingDataset


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


def compute_embedding(stems, mixture, model, feature_extractor, device):
    """Compute embedding for a track."""
    with torch.no_grad():
        # Extract features on CPU
        stems_cpu = {k: v.cpu() for k, v in stems.items()}
        mixture_cpu = mixture.cpu()
        mixing_features = feature_extractor.extract_all_features(stems_cpu, mixture_cpu)

        # Move to device for model
        stems_device = {k: v.unsqueeze(0).to(device) for k, v in stems_cpu.items()}
        mixing_features = mixing_features.unsqueeze(0).to(device)

        embedding = model(stems_device, mixing_features)
        embedding = F.normalize(embedding.squeeze(0), p=2, dim=0)

    return embedding


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Select distinctive pairs from balanced subset')
    parser.add_argument('--checkpoint', type=str, default='/nas/mixing-representation/checkpoints_baseline/best_model.pt')
    parser.add_argument('--musdb_path', type=str, default='/nas/MUSDB18_Balanced')
    parser.add_argument('--num_pairs', type=int, default=10)
    parser.add_argument('--output_file', type=str, default='distinctive_pairs.json')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("Selecting Distinctive Track Pairs")
    print("="*80)
    print(f"Balanced subset: {args.musdb_path}")
    print(f"Number of pairs: {args.num_pairs}")
    print(f"Device: {device}")
    print()

    # Load mixing model
    print("Loading mixing model...")
    model = load_mixing_model(args.checkpoint, device)

    # Initialize feature extractor
    print("Initializing feature extractor...")
    feature_extractor = MixingFeatureExtractor(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512
    )

    # Load balanced dataset
    print(f"\nLoading balanced dataset...")
    dataset = MUSDB18EmbeddingDataset(
        data_path=args.musdb_path,
        split='train',
        segment_duration=10.0,
        segment_offset=0.0
    )
    print(f"Dataset size: {len(dataset)} tracks")
    print()

    # Compute embeddings for all tracks
    print("Computing embeddings for all tracks...")
    embeddings = []
    track_names = []

    for i in tqdm(range(len(dataset)), desc="Computing embeddings"):
        stems, mixture, name, _ = dataset[i]
        emb = compute_embedding(stems, mixture, model, feature_extractor, device)
        embeddings.append(emb.cpu())
        track_names.append(name)

    embeddings = torch.stack(embeddings)  # (N, 512)
    print(f"Computed embeddings shape: {embeddings.shape}")
    print()

    # Compute pairwise cosine similarities
    print("Computing pairwise cosine similarities...")
    similarities = torch.mm(embeddings, embeddings.t())  # (N, N)

    # Create mask to exclude self-pairs and duplicates
    mask = torch.triu(torch.ones_like(similarities), diagonal=1).bool()

    # Get all valid pairs with their similarities
    valid_indices = torch.where(mask)
    pair_similarities = similarities[valid_indices].numpy()
    pair_indices = list(zip(valid_indices[0].numpy(), valid_indices[1].numpy()))

    # Sort by similarity (ascending = most distinctive first)
    sorted_pairs = sorted(zip(pair_similarities, pair_indices))

    print(f"Total possible pairs: {len(pair_indices)}")
    print()

    # Select top N most distinctive pairs
    print(f"Selecting {args.num_pairs} most distinctive pairs:")
    print(f"{'#':<4} {'Track A':<40} {'Track B':<40} {'Similarity':<12} {'Distance':<10}")
    print("-"*110)

    selected_pairs = []
    for i, (similarity, (idx_a, idx_b)) in enumerate(sorted_pairs[:args.num_pairs]):
        distance = 1.0 - similarity
        selected_pairs.append({
            'pair_id': i + 1,
            'index_a': int(idx_a),
            'index_b': int(idx_b),
            'track_a': track_names[idx_a],
            'track_b': track_names[idx_b],
            'cosine_similarity': float(similarity),
            'distance': float(distance)
        })

        print(f"{i+1:<4} {track_names[idx_a]:<40} {track_names[idx_b]:<40} "
              f"{similarity:<12.4f} {distance:<10.4f}")

    print("-"*110)
    print()

    # Save results
    output_path = Path(args.output_file)
    results = {
        'num_tracks': len(dataset),
        'num_pairs': args.num_pairs,
        'pairs': selected_pairs
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_path}")
    print()

    # Print statistics
    similarities_selected = [p['cosine_similarity'] for p in selected_pairs]
    print("Statistics for selected pairs:")
    print(f"  Mean similarity: {np.mean(similarities_selected):.4f}")
    print(f"  Min similarity: {np.min(similarities_selected):.4f}")
    print(f"  Max similarity: {np.max(similarities_selected):.4f}")
    print(f"  Mean distance: {np.mean([p['distance'] for p in selected_pairs]):.4f}")
    print()


if __name__ == "__main__":
    main()
