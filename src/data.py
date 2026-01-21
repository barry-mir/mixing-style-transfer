"""
Dataset for FMA with SCNet source separation and contrastive learning pairs.
"""

import os
import sys
import glob
import torch
import torchaudio
import librosa
import numpy as np
from torch.utils.data import Dataset
import yaml
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import threading

# Add Music-Source-Separation-Training to path
scnet_path = os.path.join(os.path.dirname(__file__), '..', 'Music-Source-Separation-Training')
sys.path.insert(0, scnet_path)
from utils.settings import get_model_from_config
from utils.model_utils import demix, load_start_checkpoint
from utils.audio_utils import normalize_audio, denormalize_audio

# Import from local mixing_utils (renamed to avoid conflict with SCNet utils)
from mixing_utils import MixingFeatureExtractor, AudioAugmenter


class SCNetSeparator:
    """Wrapper for SCNet source separation (no-grad, in-memory)."""

    def __init__(self, model_path, config_path, device='cuda'):
        """
        Initialize SCNet model for inference.

        Args:
            model_path: Path to SCNet checkpoint
            config_path: Path to SCNet config YAML
            device: Device for inference
        """
        self.device = device

        # Load model and config
        model, config = get_model_from_config('scnet_masked', config_path)
        checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')

        # Create dummy args for load_start_checkpoint
        class Args:
            model_type = 'scnet_masked'
            start_check_point = model_path

        args = Args()
        args.lora_checkpoint_loralib = None
        load_start_checkpoint(args, model, checkpoint, type_='inference')

        self.model = model.to(device)
        self.model.eval()
        self.config = config
        self.sample_rate = config.audio.get('sample_rate', 44100)

    @torch.no_grad()
    def separate(self, audio):
        """
        Separate audio into 4 stems (vocals, bass, drums, other).

        Args:
            audio: Stereo audio tensor (2, T) or numpy array

        Returns:
            stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                       Each value is tensor of shape (2, T)
        """
        # Convert to numpy if tensor
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)

        # Normalize
        if self.config.inference.get('normalize', False):
            audio, norm_params = normalize_audio(audio)
        else:
            norm_params = None

        # Separate using SCNet demix function
        with torch.no_grad():
            waveforms = demix(
                self.config,
                self.model,
                audio,
                self.device,
                model_type='scnet_masked',
                pbar=False
            )

        # Denormalize if needed
        if norm_params is not None:
            for stem in waveforms:
                waveforms[stem] = denormalize_audio(waveforms[stem], norm_params)

        # Convert to torch tensors
        stems_dict = {
            stem: torch.from_numpy(waveforms[stem]).float()
            for stem in ['vocals', 'bass', 'drums', 'other']
        }

        return stems_dict


class FMAContrastiveDataset(Dataset):
    """
    FMA dataset for two-axis contrastive learning of mixing style representations.

    Two-axis structure:
    - Axis 1 (Content): Same song, different segments → POSITIVE for content
    - Axis 2 (Mixing): Same song, different mix variants → NEGATIVE for mixing style

    Each sample returns (S × V × T) items where:
    - S songs
    - V mix variants per song (augmentations)
    - T temporal segments per variant

    This enables multi-positive, multi-negative InfoNCE loss.
    """

    def __init__(
        self,
        data_path,
        separated_path='/nas/FMA/fma_separated/',
        use_preseparated=True,
        scnet_separator=None,
        clip_duration=10.0,
        sample_rate=44100,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        augment_prob=0.5,
        gain_range=9.0,
        num_songs_per_batch=8,  # S: number of songs
        num_mix_variants=3,      # V: number of mix variants per song
        num_segments=2,          # T: number of temporal segments per variant
        min_audio_duration=25.0  # Minimum duration to get multiple clips
    ):
        """
        Args:
            data_path: Path to FMA dataset (original audio)
            separated_path: Path to pre-separated stems directory
            use_preseparated: If True, load pre-separated stems (recommended)
            scnet_separator: SCNetSeparator instance (only needed if use_preseparated=False)
            clip_duration: Duration of clips in seconds
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            augment_prob: Probability of applying augmentations
            gain_range: Gain augmentation range
            num_songs_per_batch: S - number of songs per batch
            num_mix_variants: V - number of mix variants per song
            num_segments: T - number of temporal segments per variant
            min_audio_duration: Minimum audio duration to use
        """
        self.data_path = data_path
        self.separated_path = separated_path
        self.use_preseparated = use_preseparated
        self.scnet = scnet_separator
        self.clip_duration = clip_duration
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.clip_samples = int(clip_duration * sample_rate)

        # Two-axis contrastive learning structure
        self.num_songs = num_songs_per_batch
        self.num_variants = num_mix_variants
        self.num_segments = num_segments
        self.items_per_song = num_mix_variants * num_segments  # V × T

        if use_preseparated:
            # Find pre-separated track directories
            if not os.path.exists(separated_path):
                raise ValueError(f"Separated stems directory not found: {separated_path}")

            self.track_dirs = [
                d for d in glob.glob(os.path.join(separated_path, '*'))
                if os.path.isdir(d)
            ]
            print(f"Found {len(self.track_dirs)} pre-separated tracks.")
        else:
            # Find all audio files (old behavior)
            self.audio_files = []
            for ext in ['*.mp3', '*.wav', '*.flac']:
                self.audio_files.extend(glob.glob(os.path.join(data_path, '**', ext), recursive=True))
            print(f"Found {len(self.audio_files)} audio files.")

            if scnet_separator is None:
                raise ValueError("scnet_separator required when use_preseparated=False")

        # Initialize feature extractor and augmenter
        self.feature_extractor = MixingFeatureExtractor(sample_rate, n_fft, hop_length, n_mels)
        self.augmenter = AudioAugmenter(sample_rate, gain_range, augment_prob)

    def __len__(self):
        if self.use_preseparated:
            return len(self.track_dirs)
        else:
            return len(self.audio_files)

    def _load_single_stem(self, stem_path):
        """Load a single stem file (for parallel loading)."""
        # Use torchaudio (much faster than librosa)
        try:
            audio, sr = torchaudio.load(stem_path)

            # Resample if needed (torchaudio is faster than librosa for this)
            if sr != self.sr:
                resampler = torchaudio.transforms.Resample(sr, self.sr)
                audio = resampler(audio)

            # Ensure stereo
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)

        except Exception as e:
            # Fallback to librosa if torchaudio fails
            audio, sr = librosa.load(stem_path, sr=self.sr, mono=False)
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)
            audio = torch.from_numpy(audio).float()

        return audio

    def _load_preseparated_stems(self, track_dir):
        """Load pre-separated stems from disk using parallel I/O."""
        stem_paths = {
            stem_name: os.path.join(track_dir, f"{stem_name}.mp3")
            for stem_name in ['vocals', 'bass', 'drums', 'other']
        }

        # Check all files exist
        for stem_name, stem_path in stem_paths.items():
            if not os.path.exists(stem_path):
                raise FileNotFoundError(f"Missing stem: {stem_path}")

        # Load all stems in parallel using ThreadPoolExecutor
        # This parallelizes MP3 decoding which is CPU-intensive
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                stem_name: executor.submit(self._load_single_stem, path)
                for stem_name, path in stem_paths.items()
            }
            stems_dict = {
                stem_name: future.result()
                for stem_name, future in futures.items()
            }

        return stems_dict

    def _load_audio(self, file_path):
        """Load and resample audio to target sample rate."""
        audio, sr = librosa.load(file_path, sr=self.sr, mono=False)

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)

        return torch.from_numpy(audio).float()

    def _random_crop(self, audio, duration_samples):
        """Randomly crop audio to specified duration."""
        total_samples = audio.shape[1]

        if total_samples <= duration_samples:
            # Pad if too short
            padding = duration_samples - total_samples
            audio = torch.nn.functional.pad(audio, (0, padding))
            start_idx = 0
        else:
            # Random crop
            max_start = total_samples - duration_samples
            start_idx = np.random.randint(0, max_start)

        return audio[:, start_idx:start_idx + duration_samples]

    def _random_crop_stems(self, stems_dict, duration_samples):
        """Randomly crop all stems to the same duration."""
        # Get total duration from any stem
        total_samples = list(stems_dict.values())[0].shape[1]

        if total_samples <= duration_samples:
            # Pad if too short
            padding = duration_samples - total_samples
            cropped_stems = {}
            for stem_name, stem_audio in stems_dict.items():
                cropped_stems[stem_name] = torch.nn.functional.pad(stem_audio, (0, padding))
        else:
            # Random crop
            max_start = total_samples - duration_samples
            start_idx = np.random.randint(0, max_start)

            cropped_stems = {}
            for stem_name, stem_audio in stems_dict.items():
                cropped_stems[stem_name] = stem_audio[:, start_idx:start_idx + duration_samples]

        return cropped_stems

    def __getitem__(self, idx):
        """
        Get a contrastive sample for one song.

        Two modes:
        1. Baseline mode (num_variants=1, num_segments=2):
           - Returns 2 random clips from same song (no augmentation)
           - Simple positive pairs: same song, different clips

        2. Two-axis mode (num_variants>1):
           - Returns V × T items with augmentations
           - Complex positive/negative structure

        Returns:
            Tuple of (stems_list, features_list, song_idx, variant_idxs, segment_idxs):
            - stems_list: List of items, each containing stems_dict with 4 stems (2, T)
            - features_list: List of mixing feature tensors (feature_dim,)
            - song_idx: Song index (for identifying positives)
            - variant_idxs: Tensor of variant indices (for mixing discrimination)
            - segment_idxs: Tensor of segment indices (for temporal discrimination)
        """
        if self.use_preseparated:
            # Load full song stems
            track_dir = self.track_dirs[idx]
            full_stems = self._load_preseparated_stems(track_dir)
        else:
            # Load audio and separate on-the-fly
            audio_file = self.audio_files[idx]
            audio = self._load_audio(audio_file)
            full_stems = self.scnet.separate(audio)

        stems_list = []
        features_list = []
        variant_idxs = []
        segment_idxs = []

        # Baseline mode: no augmentation, just different temporal segments
        if self.num_variants == 1:
            # Only use original stems, no augmentation
            for segment_idx in range(self.num_segments):
                # Random crop a segment
                segment_stems = self._random_crop_stems(full_stems, self.clip_samples)

                # Compute mixing features
                segment_mixture = sum(segment_stems.values())
                segment_features = self.feature_extractor.extract_all_features(
                    segment_stems, segment_mixture
                )

                stems_list.append(segment_stems)
                features_list.append(segment_features)
                variant_idxs.append(0)  # All same variant (original)
                segment_idxs.append(segment_idx)
        else:
            # Two-axis mode: Generate V mix variants
            for variant_idx in range(self.num_variants):
                # Create mix variant by augmenting full stems
                # Variant 0 = original, others = augmented
                if variant_idx == 0:
                    variant_stems = full_stems
                else:
                    variant_stems = self.augmenter.augment_stems(full_stems)

                # Generate T temporal segments from this variant
                for segment_idx in range(self.num_segments):
                    # Random crop a segment
                    segment_stems = self._random_crop_stems(variant_stems, self.clip_samples)

                    # Compute mixing features
                    segment_mixture = sum(segment_stems.values())
                    segment_features = self.feature_extractor.extract_all_features(
                        segment_stems, segment_mixture
                    )

                    stems_list.append(segment_stems)
                    features_list.append(segment_features)
                    variant_idxs.append(variant_idx)
                    segment_idxs.append(segment_idx)

        # Convert to tensors
        variant_idxs = torch.tensor(variant_idxs, dtype=torch.long)
        segment_idxs = torch.tensor(segment_idxs, dtype=torch.long)

        # Explicitly delete full_stems to free memory immediately
        del full_stems
        if self.num_variants > 1:
            # variant_stems might hold references
            import gc
            gc.collect()

        return stems_list, features_list, idx, variant_idxs, segment_idxs


def collate_fn(batch):
    """
    Custom collate function for two-axis contrastive learning.

    Args:
        batch: List of S tuples, each containing:
              (stems_list, features_list, song_idx, variant_idxs, segment_idxs)
              where each song contributes V×T items

    Returns:
        Tuple of (stems_dict, features, song_labels, variant_labels, segment_labels):
        - stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                     Each value is (S×V×T, 2, audio_samples)
        - features: (S×V×T, feature_dim)
        - song_labels: (S×V×T,) - song index for each item
        - variant_labels: (S×V×T,) - variant index within song
        - segment_labels: (S×V×T,) - segment index within variant

    These labels define the contrastive structure:
    - Same song_label + same variant_label + different segment_label → POSITIVE (content invariance)
    - Same song_label + different variant_label → NEGATIVE (mixing sensitivity)
    - Different song_label → NEGATIVE (hard negatives)
    """
    all_stems_list = []
    all_features = []
    all_song_labels = []
    all_variant_labels = []
    all_segment_labels = []

    for stems_list, features_list, song_idx, variant_idxs, segment_idxs in batch:
        # Each sample contains V×T items
        num_items = len(stems_list)

        for i in range(num_items):
            all_stems_list.append(stems_list[i])
            all_features.append(features_list[i])
            all_song_labels.append(song_idx)
            all_variant_labels.append(variant_idxs[i].item())
            all_segment_labels.append(segment_idxs[i].item())

    # Stack all stems
    stems_dict = {}
    for stem_name in ['vocals', 'bass', 'drums', 'other']:
        stem_tensors = [stems[stem_name] for stems in all_stems_list]
        stems_dict[stem_name] = torch.stack(stem_tensors, dim=0)  # (S×V×T, 2, T)

    # Stack features and labels
    features = torch.stack(all_features, dim=0)  # (S×V×T, feature_dim)
    song_labels = torch.tensor(all_song_labels, dtype=torch.long)  # (S×V×T,)
    variant_labels = torch.tensor(all_variant_labels, dtype=torch.long)  # (S×V×T,)
    segment_labels = torch.tensor(all_segment_labels, dtype=torch.long)  # (S×V×T,)

    return stems_dict, features, song_labels, variant_labels, segment_labels
