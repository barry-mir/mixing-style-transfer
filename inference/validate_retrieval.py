"""
Validation script for retrieval evaluation.

Evaluates:
1. In-domain: Top-k retrieval accuracy on validation set
2. Out-of-domain: Retrieval for test cases in /nas/music2preset_testcase/
"""

import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import shutil
import librosa
import soundfile as sf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from model import MixingStyleEncoder
from data import FMAContrastiveDataset, SCNetSeparator
from mixing_utils import MixingFeatureExtractor
from validation_utils import (
    compute_track_embedding,
    build_embedding_cache,
    retrieve_top_k,
    evaluate_retrieval_accuracy,
    save_cache,
    load_cache,
    save_metrics
)


def load_model(checkpoint_path, device):
    """Load model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")

    # Initialize model (match training configuration from train_baseline.sh)
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

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model loaded successfully from epoch {checkpoint['epoch']}")
    return model


def validate_in_domain(model, dataset, val_indices, feature_extractor, scnet, device, cache_dir, use_cache=True):
    """
    Validate on in-domain data (validation set).

    Args:
        model: Trained model
        dataset: FMAContrastiveDataset
        val_indices: List of validation indices
        feature_extractor: MixingFeatureExtractor
        scnet: SCNetSeparator
        device: torch device
        cache_dir: Directory to save/load caches
        use_cache: Whether to use cached embeddings

    Returns:
        metrics: Dict with retrieval accuracies
    """
    print("\n" + "=" * 80)
    print("IN-DOMAIN VALIDATION (Validation Set)")
    print("=" * 80)

    query_cache_path = os.path.join(cache_dir, 'val_queries.pt')
    pool_cache_path = os.path.join(cache_dir, 'val_pool.pt')

    # Build or load query cache (last 10 seconds)
    if use_cache and os.path.exists(query_cache_path):
        print("Loading query cache...")
        query_cache = load_cache(query_cache_path)
    else:
        print("\nBuilding query embeddings (last 10 seconds of each track)...")
        query_embeddings = []
        query_indices = []

        for idx in tqdm(val_indices, desc="Computing queries"):
            try:
                track_path = dataset.track_dirs[idx]

                # Get audio duration to compute offset for last 10 seconds
                # Load first stem to check duration
                vocals_path = os.path.join(track_path, 'vocals.mp3')
                import librosa
                duration = librosa.get_duration(path=vocals_path)

                # Use last 10 seconds (or full track if shorter)
                start_sec = max(0, duration - 10.0)

                embedding = compute_track_embedding(
                    track_path=track_path,
                    start_sec=start_sec,
                    duration_sec=10.0,
                    model=model,
                    feature_extractor=feature_extractor,
                    scnet=scnet,
                    device=device,
                    use_preseparated=True
                )

                query_embeddings.append(embedding)
                query_indices.append(idx)

            except Exception as e:
                print(f"\nError processing query {idx}: {e}")
                continue

        query_cache = {
            'embeddings': torch.stack(query_embeddings),
            'track_indices': query_indices
        }
        save_cache(query_cache, query_cache_path)

    # Build or load retrieval pool cache (first 10 seconds)
    if use_cache and os.path.exists(pool_cache_path):
        print("Loading retrieval pool cache...")
        pool_cache = load_cache(pool_cache_path)
    else:
        print("\nBuilding retrieval pool (first 10 seconds of each track)...")
        pool_cache = build_embedding_cache(
            dataset=dataset,
            indices=val_indices,
            model=model,
            feature_extractor=feature_extractor,
            scnet=scnet,
            device=device,
            query_duration=10.0,
            use_preseparated=True,
            desc="Computing pool"
        )
        save_cache(pool_cache, pool_cache_path)

    # Evaluate retrieval
    print("\nEvaluating retrieval performance...")
    metrics = evaluate_retrieval_accuracy(
        queries=query_cache['embeddings'],
        retrieval_pool=pool_cache['embeddings'],
        query_indices=query_cache['track_indices'],
        pool_indices=pool_cache['track_indices'],
        k_values=[1, 5]
    )

    print("\nIn-Domain Retrieval Results:")
    print(f"  Top-1 Accuracy: {metrics['top_1_accuracy']*100:.2f}%")
    print(f"  Top-5 Accuracy: {metrics['top_5_accuracy']*100:.2f}%")

    return metrics


def validate_out_of_domain(model, dataset, train_val_indices, feature_extractor, scnet, device,
                           test_dir, cache_dir, output_dir, use_cache=True):
    """
    Validate on out-of-domain test cases.

    Args:
        model: Trained model
        dataset: FMAContrastiveDataset
        train_val_indices: List of all train+val indices
        feature_extractor: MixingFeatureExtractor
        scnet: SCNetSeparator
        device: torch device
        test_dir: Directory with test MP3 files
        cache_dir: Directory to save/load caches
        output_dir: Directory to save results
        use_cache: Whether to use cached embeddings

    Returns:
        results: Dict with retrieval results
    """
    print("\n" + "=" * 80)
    print("OUT-OF-DOMAIN VALIDATION (Test Cases)")
    print("=" * 80)

    pool_cache_path = os.path.join(cache_dir, 'full_dataset_pool.pt')

    # Build or load full dataset retrieval pool
    if use_cache and os.path.exists(pool_cache_path):
        print("Loading full dataset pool cache...")
        pool_cache = load_cache(pool_cache_path)
    else:
        print("\nBuilding full dataset retrieval pool (train + val, first 10 seconds)...")
        pool_cache = build_embedding_cache(
            dataset=dataset,
            indices=train_val_indices,
            model=model,
            feature_extractor=feature_extractor,
            scnet=scnet,
            device=device,
            query_duration=10.0,
            use_preseparated=True,
            desc="Computing full pool"
        )
        save_cache(pool_cache, pool_cache_path)

    # Find test files
    test_files = list(Path(test_dir).glob('*.mp3'))
    print(f"\nFound {len(test_files)} test files in {test_dir}")

    if len(test_files) == 0:
        print("No test files found!")
        return {}

    # Process each test file
    results = {}
    retrieved_dir = os.path.join(output_dir, 'retrieved_audio')
    os.makedirs(retrieved_dir, exist_ok=True)

    print("\nProcessing test files...")
    for test_file in tqdm(test_files, desc="Test cases"):
        try:
            test_name = test_file.stem

            # Compute query embedding (first 10 seconds)
            # Note: test files are unseparated, so we use scnet
            print(f"\n  Processing: {test_name}")
            query_embedding = compute_track_embedding(
                track_path=str(test_file),
                start_sec=0.0,
                duration_sec=10.0,
                model=model,
                feature_extractor=feature_extractor,
                scnet=scnet,
                device=device,
                use_preseparated=False  # Need to separate on-the-fly
            )

            # Retrieve top-1 match
            top_indices, similarities = retrieve_top_k(
                query_embedding=query_embedding,
                retrieval_pool=pool_cache['embeddings'],
                k=1
            )

            # Get retrieved track info
            retrieved_idx = top_indices[0].item()
            retrieved_track_idx = pool_cache['track_indices'][retrieved_idx]
            retrieved_track_path = pool_cache['track_paths'][retrieved_idx]
            similarity = similarities[0].item()

            # Load all stems and mix them together
            try:
                stems_to_mix = []
                for stem_name in ['vocals', 'bass', 'drums', 'other']:
                    stem_path = os.path.join(retrieved_track_path, f"{stem_name}.mp3")
                    if os.path.exists(stem_path):
                        stem_audio, sr = librosa.load(stem_path, sr=44100, mono=False)
                        # Ensure stereo
                        if stem_audio.ndim == 1:
                            stem_audio = np.stack([stem_audio, stem_audio], axis=0)
                        elif stem_audio.shape[0] == 1:
                            stem_audio = np.repeat(stem_audio, 2, axis=0)
                        stems_to_mix.append(stem_audio)

                if stems_to_mix:
                    # Sum all stems to create full mix
                    mixed_audio = sum(stems_to_mix)

                    # Save as audio file
                    output_path = os.path.join(retrieved_dir, f"{test_name}_retrieved_mix.wav")
                    sf.write(output_path, mixed_audio.T, 44100)  # Transpose to (samples, channels)

            except Exception as e:
                print(f"    Warning: Could not create mix for {test_name}: {e}")

            # Store results
            results[test_name] = {
                'test_file': str(test_file),
                'retrieved_track_idx': retrieved_track_idx,
                'retrieved_track_path': retrieved_track_path,
                'similarity': similarity
            }

            print(f"    → Retrieved: Track {retrieved_track_idx} (similarity: {similarity:.4f})")

        except Exception as e:
            print(f"\n  Error processing {test_file}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save retrieval log
    log_path = os.path.join(output_dir, 'retrieval_log.json')
    save_metrics(results, log_path)

    print(f"\nOut-of-domain results saved to: {output_dir}")
    print(f"  Retrieval log: {log_path}")
    print(f"  Retrieved audio: {retrieved_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Validation script for retrieval evaluation')
    parser.add_argument('--checkpoint', type=str, default='/nas/mixing-representation/checkpoints_baseline/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default='/ssd2/barry/fma_full/',
                        help='Path to FMA dataset')
    parser.add_argument('--separated_path', type=str, default='/ssd2/barry/fma_separated_cropped/',
                        help='Path to pre-separated stems')
    parser.add_argument('--test_dir', type=str, default='/nas/music2preset_testcase/',
                        help='Directory with out-of-domain test files')
    parser.add_argument('--output_dir', type=str, default='validation_results/',
                        help='Output directory for results')
    parser.add_argument('--cache_dir', type=str, default='validation_results/embeddings_cache/',
                        help='Directory to save/load embedding caches')
    parser.add_argument('--use_cache', action='store_true', default=True,
                        help='Use cached embeddings if available')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for dataset split')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--scnet_model', type=str,
                        default='Music-Source-Separation-Training/model_scnet_masked_ep_111_sdr_9.8286.ckpt',
                        help='Path to SCNet model')
    parser.add_argument('--scnet_config', type=str,
                        default='Music-Source-Separation-Training/configs/config_musdb18_scnet_xl_ihf.yaml',
                        help='Path to SCNet config')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    # Load model
    model = load_model(args.checkpoint, device)

    # Initialize feature extractor (match training config)
    print("\nInitializing feature extractor...")
    feature_extractor = MixingFeatureExtractor(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512,
        n_mels=80
    )

    # Initialize SCNet separator (for out-of-domain data)
    print("Initializing SCNet separator...")
    scnet = SCNetSeparator(
        model_path=args.scnet_model,
        config_path=args.scnet_config,
        device=device
    )

    # Create dataset
    print("\nLoading dataset...")
    dataset = FMAContrastiveDataset(
        data_path=args.data_path,
        separated_path=args.separated_path,
        use_preseparated=True,
        scnet_separator=None,  # Not needed since using pre-separated
        clip_duration=10.0,
        sample_rate=44100,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        augment_prob=0.0,  # No augmentation for validation
        gain_range=0.0,
        num_songs_per_batch=1,
        num_mix_variants=1,
        num_segments=1
    )
    print(f"Dataset loaded: {len(dataset)} tracks")

    # Split dataset (same as training)
    dataset_size = len(dataset)
    val_size = int(args.val_split * dataset_size)
    train_size = dataset_size - val_size

    indices = list(range(dataset_size))
    np.random.seed(args.seed)
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_val_indices = indices  # All indices for out-of-domain pool

    print(f"\nDataset split:")
    print(f"  Training: {len(train_indices)} tracks")
    print(f"  Validation: {len(val_indices)} tracks")

    # Run in-domain validation
    in_domain_metrics = validate_in_domain(
        model=model,
        dataset=dataset,
        val_indices=val_indices,
        feature_extractor=feature_extractor,
        scnet=scnet,
        device=device,
        cache_dir=args.cache_dir,
        use_cache=args.use_cache
    )

    # Run out-of-domain validation
    out_domain_results = validate_out_of_domain(
        model=model,
        dataset=dataset,
        train_val_indices=train_val_indices,
        feature_extractor=feature_extractor,
        scnet=scnet,
        device=device,
        test_dir=args.test_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        use_cache=args.use_cache
    )

    # Save final metrics
    final_metrics = {
        'in_domain': in_domain_metrics,
        'out_of_domain': {
            'num_test_cases': len(out_domain_results),
            'test_cases': list(out_domain_results.keys())
        }
    }
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    save_metrics(final_metrics, metrics_path)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  Metrics: {metrics_path}")
    print(f"  In-domain Top-1 Accuracy: {in_domain_metrics['top_1_accuracy']*100:.2f}%")
    print(f"  In-domain Top-5 Accuracy: {in_domain_metrics['top_5_accuracy']*100:.2f}%")
    print(f"  Out-of-domain test cases processed: {len(out_domain_results)}")


if __name__ == '__main__':
    main()
