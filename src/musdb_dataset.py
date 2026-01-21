"""
MUSDB18 Dataset loader for mixing representation learning.

MUSDB18 contains professionally mixed music tracks with separated stems:
- 100 training tracks
- 50 test tracks
- Each track has: vocals, bass, drums, other, and mixture

Dataset structure:
/nas/MUSDB18/
├── train/
│   ├── <song_name>/
│   │   ├── mixture.wav
│   │   ├── vocals.wav
│   │   ├── bass.wav
│   │   ├── drums.wav
│   │   └── other.wav
│   └── ...
└── test/
    └── ...
"""

import os
import torch
import librosa
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from torch.utils.data import Dataset


class MUSDB18Dataset(Dataset):
    """
    MUSDB18 dataset for mixing representation learning.

    Loads separated stems and mixture for each track.
    Compatible with FMA dataset API for easy switching.
    """

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        sample_rate: int = 44100,
        clip_duration: Optional[float] = None,
        return_mixture: bool = True
    ):
        """
        Args:
            data_path: Path to MUSDB18 root directory (e.g., /nas/MUSDB18)
            split: 'train' or 'test'
            sample_rate: Target sample rate (default: 44100)
            clip_duration: If specified, load only first N seconds. If None, load full track.
            return_mixture: Whether to also return the mixture audio
        """
        self.data_path = Path(data_path)
        self.split = split
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.return_mixture = return_mixture

        # Find all track directories
        split_path = self.data_path / split
        if not split_path.exists():
            raise ValueError(f"Split path does not exist: {split_path}")

        self.track_dirs = sorted([
            d for d in split_path.iterdir()
            if d.is_dir()
        ])

        if len(self.track_dirs) == 0:
            raise ValueError(f"No tracks found in {split_path}")

        print(f"MUSDB18 {split} set: Found {len(self.track_dirs)} tracks")

    def __len__(self) -> int:
        return len(self.track_dirs)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor], str]:
        """
        Load stems and mixture for a track.

        Returns:
            stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                       Each value is tensor of shape (2, T) for stereo
            mixture: Mixture audio tensor (2, T) or None if return_mixture=False
            track_name: Name of the track
        """
        track_dir = self.track_dirs[idx]
        track_name = track_dir.name

        # Load all stems
        stems_dict = {}
        for stem_name in ['vocals', 'bass', 'drums', 'other']:
            stem_path = track_dir / f"{stem_name}.wav"

            if not stem_path.exists():
                raise FileNotFoundError(f"Missing stem: {stem_path}")

            # Load audio with librosa
            # duration parameter crops from start if specified
            audio, sr = librosa.load(
                stem_path,
                sr=self.sample_rate,
                mono=False,
                duration=self.clip_duration
            )

            # Ensure stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)
            elif audio.shape[0] == 1:
                audio = np.repeat(audio, 2, axis=0)

            stems_dict[stem_name] = torch.from_numpy(audio).float()

        # Load mixture if requested
        mixture = None
        if self.return_mixture:
            mixture_path = track_dir / "mixture.wav"

            if mixture_path.exists():
                audio, sr = librosa.load(
                    mixture_path,
                    sr=self.sample_rate,
                    mono=False,
                    duration=self.clip_duration
                )

                # Ensure stereo
                if audio.ndim == 1:
                    audio = np.stack([audio, audio], axis=0)
                elif audio.shape[0] == 1:
                    audio = np.repeat(audio, 2, axis=0)

                mixture = torch.from_numpy(audio).float()
            else:
                # If mixture doesn't exist, compute from stems
                mixture = sum(stems_dict.values())

        return stems_dict, mixture, track_name

    def get_track_paths(self) -> List[Path]:
        """Get list of all track directory paths."""
        return self.track_dirs

    def get_track_names(self) -> List[str]:
        """Get list of all track names."""
        return [d.name for d in self.track_dirs]


class MUSDB18EmbeddingDataset(Dataset):
    """
    Simplified MUSDB18 dataset for embedding extraction.

    Only loads what's needed for computing embeddings from a trained model.
    Useful for evaluation and retrieval tasks.
    """

    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        sample_rate: int = 44100,
        segment_duration: float = 10.0,
        segment_offset: float = 0.0
    ):
        """
        Args:
            data_path: Path to MUSDB18 root directory
            split: 'train' or 'test'
            sample_rate: Target sample rate
            segment_duration: Duration of segment to load in seconds
            segment_offset: Offset from start in seconds (e.g., 0.0 = beginning, 10.0 = skip first 10s)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.segment_offset = segment_offset

        # Find all tracks
        split_path = self.data_path / split
        if not split_path.exists():
            raise ValueError(f"Split path does not exist: {split_path}")

        self.track_dirs = sorted([
            d for d in split_path.iterdir()
            if d.is_dir()
        ])

        print(f"MUSDB18 {split} set: Found {len(self.track_dirs)} tracks")
        print(f"  Segment: {segment_duration}s starting at {segment_offset}s")

    def __len__(self) -> int:
        return len(self.track_dirs)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, str, Path]:
        """
        Load a segment of stems and mixture.

        Returns:
            stems_dict: Dict of stem audio tensors (2, T)
            mixture: Mixture audio tensor (2, T)
            track_name: Name of the track
            track_path: Path to track directory
        """
        track_dir = self.track_dirs[idx]
        track_name = track_dir.name

        # Calculate offset and duration for librosa.load
        offset = self.segment_offset
        duration = self.segment_duration

        # Load stems
        stems_dict = {}
        for stem_name in ['vocals', 'bass', 'drums', 'other']:
            stem_path = track_dir / f"{stem_name}.wav"

            if not stem_path.exists():
                raise FileNotFoundError(f"Missing stem: {stem_path}")

            audio, sr = librosa.load(
                stem_path,
                sr=self.sample_rate,
                mono=False,
                offset=offset,
                duration=duration
            )

            # Ensure stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)
            elif audio.shape[0] == 1:
                audio = np.repeat(audio, 2, axis=0)

            stems_dict[stem_name] = torch.from_numpy(audio).float()

        # Load mixture
        mixture_path = track_dir / "mixture.wav"

        if mixture_path.exists():
            audio, sr = librosa.load(
                mixture_path,
                sr=self.sample_rate,
                mono=False,
                offset=offset,
                duration=duration
            )

            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)
            elif audio.shape[0] == 1:
                audio = np.repeat(audio, 2, axis=0)

            mixture = torch.from_numpy(audio).float()
        else:
            # Compute from stems
            mixture = sum(stems_dict.values())

        return stems_dict, mixture, track_name, track_dir


def test_musdb18_dataset():
    """Test MUSDB18 dataset loading."""
    print("Testing MUSDB18Dataset...")

    # Test basic loading
    dataset = MUSDB18Dataset(
        data_path='/nas/MUSDB18',
        split='train',
        sample_rate=44100,
        clip_duration=10.0
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Track names: {dataset.get_track_names()[:5]}")

    # Load first track
    stems_dict, mixture, track_name = dataset[0]

    print(f"\nLoaded track: {track_name}")
    print(f"Stems:")
    for stem_name, audio in stems_dict.items():
        print(f"  {stem_name}: {audio.shape}")
    print(f"Mixture: {mixture.shape if mixture is not None else None}")

    # Test embedding dataset
    print("\n" + "="*80)
    print("Testing MUSDB18EmbeddingDataset...")

    emb_dataset = MUSDB18EmbeddingDataset(
        data_path='/nas/MUSDB18',
        split='test',
        sample_rate=44100,
        segment_duration=10.0,
        segment_offset=0.0
    )

    stems_dict, mixture, track_name, track_path = emb_dataset[0]
    print(f"\nLoaded track: {track_name}")
    print(f"Track path: {track_path}")
    print(f"Mixture shape: {mixture.shape}")


if __name__ == '__main__':
    test_musdb18_dataset()
