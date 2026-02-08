"""
Utility functions for validation and retrieval evaluation.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import librosa
from tqdm import tqdm
from pathlib import Path
import json


def load_audio_segment(audio_path, start_sec, duration_sec, sample_rate=44100):
    """
    Load a specific segment of audio file.

    Args:
        audio_path: Path to audio file
        start_sec: Start time in seconds
        duration_sec: Duration in seconds
        sample_rate: Target sample rate

    Returns:
        audio: Numpy array (2, samples) - stereo audio
    """
    # Calculate sample indices
    start_sample = int(start_sec * sample_rate)
    duration_samples = int(duration_sec * sample_rate)

    # Load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate, mono=False, offset=start_sec, duration=duration_sec)

    # Ensure stereo
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] == 1:
        audio = np.repeat(audio, 2, axis=0)

    # Pad if too short
    if audio.shape[1] < duration_samples:
        pad_length = duration_samples - audio.shape[1]
        audio = np.pad(audio, ((0, 0), (0, pad_length)), mode='constant')

    return audio


def load_stems_segment(track_dir, start_sec, duration_sec, sample_rate=44100):
    """
    Load a specific segment from pre-separated stems.

    Args:
        track_dir: Directory containing separated stems
        start_sec: Start time in seconds
        duration_sec: Duration in seconds
        sample_rate: Target sample rate

    Returns:
        stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                   Each value is numpy array (2, samples)
    """
    stems_dict = {}

    for stem_name in ['vocals', 'bass', 'drums', 'other']:
        stem_path = os.path.join(track_dir, f"{stem_name}.mp3")
        if not os.path.exists(stem_path):
            raise FileNotFoundError(f"Missing stem: {stem_path}")

        # Load segment
        audio = load_audio_segment(stem_path, start_sec, duration_sec, sample_rate)
        stems_dict[stem_name] = audio

    return stems_dict


def compute_embedding(stems_dict, mixing_features, model, device):
    """
    Compute embedding for a single sample.

    Args:
        stems_dict: Dict of stems (numpy arrays)
        mixing_features: Torch tensor of mixing features
        model: MixingStyleEncoder model
        device: torch device

    Returns:
        embedding: Torch tensor (768,)
    """
    # Convert stems to torch tensors
    stems_dict_torch = {}
    for stem_name, stem_audio in stems_dict.items():
        tensor = torch.from_numpy(stem_audio).float().unsqueeze(0)  # (1, 2, T)
        stems_dict_torch[stem_name] = tensor.to(device)

    # Move features to device
    mixing_features = mixing_features.unsqueeze(0).to(device)  # (1, feature_dim)

    # Compute embedding
    with torch.no_grad():
        embedding = model(stems_dict_torch, mixing_features)  # (1, 768)

    return embedding.squeeze(0).cpu()  # (768,)


def compute_track_embedding(track_path, start_sec, duration_sec, model, feature_extractor, scnet, device, use_preseparated=True):
    """
    Compute embedding for a track segment.

    Args:
        track_path: Path to track directory (if preseparated) or audio file
        start_sec: Start time in seconds
        duration_sec: Duration in seconds
        model: MixingStyleEncoder model
        feature_extractor: MixingFeatureExtractor
        scnet: SCNetSeparator (only needed if not preseparated)
        device: torch device
        use_preseparated: Whether to use pre-separated stems

    Returns:
        embedding: Torch tensor (768,)
    """
    sample_rate = 44100

    if use_preseparated:
        # Load from pre-separated stems
        stems_dict = load_stems_segment(track_path, start_sec, duration_sec, sample_rate)
    else:
        # Load audio and separate on-the-fly
        audio = load_audio_segment(track_path, start_sec, duration_sec, sample_rate)
        audio_torch = torch.from_numpy(audio).float().to(device)
        stems_dict_torch = scnet.separate(audio_torch)

        # Convert back to numpy for feature extraction
        stems_dict = {
            stem_name: stem_tensor.cpu().numpy()
            for stem_name, stem_tensor in stems_dict_torch.items()
        }

    # Compute mixing features
    mixing_features = feature_extractor.extract_all_features(
        {k: torch.from_numpy(v).float() for k, v in stems_dict.items()}
    )

    # Compute embedding
    embedding = compute_embedding(stems_dict, mixing_features, model, device)

    return embedding


def build_embedding_cache(dataset, indices, model, feature_extractor, scnet, device,
                          query_duration=1.0, use_preseparated=True, desc="Building cache"):
    """
    Pre-compute embeddings for a set of tracks.

    Args:
        dataset: FMAContrastiveDataset
        indices: List of dataset indices to process
        model: MixingStyleEncoder model
        feature_extractor: MixingFeatureExtractor
        scnet: SCNetSeparator
        device: torch device
        query_duration: Duration in seconds to use from each track
        use_preseparated: Whether to use pre-separated stems
        desc: Progress bar description

    Returns:
        cache: Dict with keys:
            'embeddings': Tensor (N, 768)
            'track_indices': List of track indices
            'track_paths': List of track paths
    """
    embeddings = []
    track_indices = []
    track_paths = []

    for idx in tqdm(indices, desc=desc):
        try:
            # Get track path
            if use_preseparated:
                track_path = dataset.track_dirs[idx]
            else:
                track_path = dataset.audio_files[idx]

            # Compute embedding for first query_duration seconds
            embedding = compute_track_embedding(
                track_path=track_path,
                start_sec=0.0,
                duration_sec=query_duration,
                model=model,
                feature_extractor=feature_extractor,
                scnet=scnet,
                device=device,
                use_preseparated=use_preseparated
            )

            embeddings.append(embedding)
            track_indices.append(idx)
            track_paths.append(track_path)

        except Exception as e:
            print(f"\nError processing track {idx}: {e}")
            continue

    # Stack embeddings
    embeddings_tensor = torch.stack(embeddings)  # (N, 768)

    cache = {
        'embeddings': embeddings_tensor,
        'track_indices': track_indices,
        'track_paths': track_paths
    }

    return cache


def retrieve_top_k(query_embedding, retrieval_pool, k=5):
    """
    Retrieve top-k most similar embeddings.

    Args:
        query_embedding: Tensor (768,)
        retrieval_pool: Tensor (N, 768)
        k: Number of retrievals

    Returns:
        indices: Tensor (k,) - indices of top-k matches
        similarities: Tensor (k,) - cosine similarities
    """
    # Normalize embeddings
    query_norm = F.normalize(query_embedding.unsqueeze(0), dim=1)  # (1, 768)
    pool_norm = F.normalize(retrieval_pool, dim=1)  # (N, 768)

    # Compute cosine similarity
    similarities = torch.matmul(query_norm, pool_norm.T).squeeze(0)  # (N,)

    # Get top-k
    top_k_similarities, top_k_indices = torch.topk(similarities, k=k, largest=True)

    return top_k_indices, top_k_similarities


def evaluate_retrieval_accuracy(queries, retrieval_pool, query_indices, pool_indices, k_values=[1, 5]):
    """
    Evaluate retrieval accuracy.

    Args:
        queries: Tensor (M, 768) - query embeddings
        retrieval_pool: Tensor (N, 768) - retrieval pool embeddings
        query_indices: List of M track indices for queries
        pool_indices: List of N track indices for pool
        k_values: List of k values to evaluate

    Returns:
        metrics: Dict with accuracy@k for each k
    """
    num_queries = queries.shape[0]
    correct_at_k = {k: 0 for k in k_values}

    for i in range(num_queries):
        query_idx = query_indices[i]
        query_emb = queries[i]

        # Retrieve top-k
        max_k = max(k_values)
        top_k_pool_idx, _ = retrieve_top_k(query_emb, retrieval_pool, k=max_k)

        # Convert pool indices to track indices
        retrieved_track_indices = [pool_indices[idx.item()] for idx in top_k_pool_idx]

        # Check if correct track is in top-k
        for k in k_values:
            if query_idx in retrieved_track_indices[:k]:
                correct_at_k[k] += 1

    # Compute accuracy
    metrics = {}
    for k in k_values:
        accuracy = correct_at_k[k] / num_queries
        metrics[f'top_{k}_accuracy'] = accuracy

    return metrics


def save_cache(cache, save_path):
    """Save embedding cache to disk."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(cache, save_path)
    print(f"Cache saved to: {save_path}")


def load_cache(cache_path):
    """Load embedding cache from disk."""
    cache = torch.load(cache_path, map_location='cpu')
    print(f"Cache loaded from: {cache_path}")
    return cache


def save_metrics(metrics, save_path):
    """Save metrics to JSON file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {save_path}")
