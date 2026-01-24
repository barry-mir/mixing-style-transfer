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


class FMABaselineDataset(Dataset):
    """
    Simple baseline FMA dataset for InfoNCE contrastive learning.

    No mixing variants - just loads pre-separated stems and creates positive
    pairs from different temporal segments of the same song.

    Positive: Same song, different temporal segments
    Negative: Different songs
    """

    def __init__(
        self,
        separated_path='/nas/FMA/fma_separated/',
        clip_duration=10.0,
        sample_rate=44100,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        num_segments=2,  # Number of temporal segments per song
        min_audio_duration=25.0  # Minimum duration to get multiple clips
    ):
        """
        Args:
            separated_path: Path to pre-separated stems directory
            clip_duration: Duration of clips in seconds
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            num_segments: Number of temporal segments per song (for positives)
            min_audio_duration: Minimum audio duration to use
        """
        self.separated_path = separated_path
        self.clip_duration = clip_duration
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.clip_samples = int(clip_duration * sample_rate)
        self.num_segments = num_segments
        self.min_duration = min_audio_duration

        # Find pre-separated track directories
        if not os.path.exists(separated_path):
            raise ValueError(f"Separated stems directory not found: {separated_path}")

        self.track_dirs = [
            d for d in glob.glob(os.path.join(separated_path, '*'))
            if os.path.isdir(d)
        ]

        # Filter by duration
        self.track_dirs = self._filter_by_duration()

        print(f"Found {len(self.track_dirs)} tracks with duration >= {min_audio_duration}s")

        # Initialize feature extractor
        self.feature_extractor = MixingFeatureExtractor(sample_rate, n_fft, hop_length, n_mels)

    def _filter_by_duration(self):
        """Filter tracks by minimum duration."""
        valid_tracks = []
        for track_dir in self.track_dirs:
            # Check vocals duration as proxy
            vocals_path = os.path.join(track_dir, 'vocals.wav')
            if os.path.exists(vocals_path):
                info = torchaudio.info(vocals_path)
                duration = info.num_frames / info.sample_rate
                if duration >= self.min_duration:
                    valid_tracks.append(track_dir)
        return valid_tracks

    def __len__(self):
        return len(self.track_dirs)

    def _load_single_stem(self, stem_path):
        """Load a single stem file."""
        audio, sr = torchaudio.load(stem_path)

        # Resample if needed
        if sr != self.sr:
            resampler = torchaudio.transforms.Resample(sr, self.sr)
            audio = resampler(audio)

        # Ensure stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2, :]

        return audio

    def _load_stems(self, track_dir):
        """Load all stems for a track."""
        stems = {}
        for stem_name in ['vocals', 'bass', 'drums', 'other']:
            stem_path = os.path.join(track_dir, f'{stem_name}.wav')
            if os.path.exists(stem_path):
                stems[stem_name] = self._load_single_stem(stem_path)
            else:
                # Silent stem if missing
                stems[stem_name] = torch.zeros(2, self.clip_samples)

        return stems

    def _extract_random_clips(self, stems, num_clips):
        """Extract random non-overlapping clips from stems."""
        # Get audio length (use vocals as reference)
        audio_length = stems['vocals'].shape[1]

        # Calculate valid start positions (avoid overlap)
        max_start = audio_length - self.clip_samples
        if max_start <= 0:
            # Audio too short, pad
            clips = []
            for _ in range(num_clips):
                clip = {}
                for stem_name, stem_audio in stems.items():
                    if stem_audio.shape[1] < self.clip_samples:
                        pad_length = self.clip_samples - stem_audio.shape[1]
                        stem_audio = torch.nn.functional.pad(stem_audio, (0, pad_length))
                    clip[stem_name] = stem_audio[:, :self.clip_samples]
                clips.append(clip)
            return clips

        # Sample non-overlapping start positions
        min_gap = self.clip_samples  # Minimum gap between clips
        starts = []
        for _ in range(num_clips):
            valid_starts = list(range(0, max_start, min_gap))
            # Filter out starts too close to existing ones
            for existing_start in starts:
                valid_starts = [s for s in valid_starts
                               if abs(s - existing_start) >= min_gap]
            if valid_starts:
                start = np.random.choice(valid_starts)
                starts.append(start)
            else:
                # Fallback: random start
                start = np.random.randint(0, max_start)
                starts.append(start)

        # Extract clips
        clips = []
        for start in starts:
            clip = {}
            for stem_name, stem_audio in stems.items():
                clip[stem_name] = stem_audio[:, start:start + self.clip_samples]
            clips.append(clip)

        return clips

    def __getitem__(self, idx):
        """
        Returns multiple temporal segments from the same song.

        Returns:
            Tuple of (stems_list, features_list, song_idx):
            - stems_list: List of num_segments stem dicts
            - features_list: List of num_segments feature vectors
            - song_idx: Song index (for contrastive learning)
        """
        track_dir = self.track_dirs[idx]

        # Load all stems
        stems_full = self._load_stems(track_dir)

        # Extract multiple random clips (temporal segments)
        clips = self._extract_random_clips(stems_full, self.num_segments)

        # Extract features for each clip
        stems_list = []
        features_list = []

        for clip_stems in clips:
            # Compute mixture
            mixture = sum(clip_stems.values())  # (2, T)

            # Extract mixing features
            features = self.feature_extractor.extract_all_features(clip_stems, mixture)

            stems_list.append(clip_stems)
            features_list.append(features)

        return stems_list, features_list, idx  # idx is the song label


def baseline_collate_fn(batch):
    """
    Collate function for baseline InfoNCE dataset.

    Args:
        batch: List of (stems_list, features_list, song_idx) tuples

    Returns:
        Tuple of (stems_dict, features, song_labels):
        - stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                     Each value is (N, 2, T) where N = batch_size * num_segments
        - features: (N, feature_dim)
        - song_labels: (N,) - song index for each segment
    """
    all_stems_list = []
    all_features = []
    all_song_labels = []

    for stems_list, features_list, song_idx in batch:
        for stems, features in zip(stems_list, features_list):
            all_stems_list.append(stems)
            all_features.append(features)
            all_song_labels.append(song_idx)

    # Stack all stems
    stems_dict = {}
    for stem_name in ['vocals', 'bass', 'drums', 'other']:
        stem_tensors = [stems[stem_name] for stems in all_stems_list]
        stems_dict[stem_name] = torch.stack(stem_tensors, dim=0)  # (N, 2, T)

    # Stack features and labels
    features = torch.stack(all_features, dim=0)  # (N, feature_dim)
    song_labels = torch.tensor(all_song_labels, dtype=torch.long)  # (N,)

    return stems_dict, features, song_labels



class StyleTransferDataset(Dataset):
    """
    Dataset for zero-shot mixing style transfer training.

    Creates pairs of (input_stems, target_stems, target_features) where:
    - input_stems: Original separated stems
    - target_stems: Augmented version with different mixing style
    - target_features: Pre-computed mixing features of target_stems

    Augmentation strategies:
    - Stem-level gain adjustments (±6dB per stem)
    - Relative loudness changes (boost vocals, reduce bass, etc.)
    - Creates diverse mixing styles from same audio content
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
        augment_prob=1.0,  # Always augment for style transfer
        gain_range=6.0,    # ±6dB gain range for creating mixing variations
        use_detailed_spectral=False,
        n_spectral_bins=32,
    ):
        """
        Args:
            data_path: Path to audio dataset
            separated_path: Path to pre-separated stems directory
            use_preseparated: If True, load pre-separated stems (recommended)
            scnet_separator: SCNetSeparator instance (only needed if use_preseparated=False)
            clip_duration: Duration of clips in seconds (default: 10.0s for encoder)
            sample_rate: Audio sample rate
            n_fft: FFT size
            hop_length: Hop length for STFT
            n_mels: Number of mel bands
            augment_prob: Probability of applying augmentations (default: 1.0)
            gain_range: Gain augmentation range in dB (default: ±6dB)
            use_detailed_spectral: If True, use detailed frequency curve
            n_spectral_bins: Number of spectral bins for detailed spectral
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

        if use_preseparated:
            # Find pre-separated track directories
            if not os.path.exists(separated_path):
                raise ValueError(f"Separated stems directory not found: {separated_path}")

            self.track_dirs = [
                d for d in glob.glob(os.path.join(separated_path, '*'))
                if os.path.isdir(d)
            ]
            print(f"[StyleTransferDataset] Found {len(self.track_dirs)} pre-separated tracks.")
        else:
            # Find all audio files
            self.audio_files = []
            for ext in ['*.mp3', '*.wav', '*.flac']:
                self.audio_files.extend(glob.glob(os.path.join(data_path, '**', ext), recursive=True))
            print(f"[StyleTransferDataset] Found {len(self.audio_files)} audio files.")

            if scnet_separator is None:
                raise ValueError("scnet_separator required when use_preseparated=False")

        # Initialize feature extractor and augmenter
        self.feature_extractor = MixingFeatureExtractor(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            use_detailed_spectral=use_detailed_spectral,
            n_spectral_bins=n_spectral_bins
        )
        self.augmenter = AudioAugmenter(sample_rate, gain_range, augment_prob)

    def __len__(self):
        if self.use_preseparated:
            return len(self.track_dirs)
        else:
            return len(self.audio_files)

    def _load_single_stem(self, stem_path):
        """Load a single stem file (for parallel loading)."""
        try:
            audio, sr = torchaudio.load(stem_path)

            # Resample if needed
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

        # Load all stems in parallel
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
        Get a style transfer training sample.

        Returns:
            Tuple of (input_stems, target_stems, target_features):
            - input_stems: Dict with keys ['vocals', 'bass', 'drums', 'other'], values (2, T)
            - target_stems: Augmented stems dict with same structure
            - target_features: Tensor of target mixing features (56,)
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

        # Random crop to clip duration (10 seconds by default)
        input_stems = self._random_crop_stems(full_stems, self.clip_samples)

        # Create target by augmenting stems (different mixing style)
        target_stems = self.augmenter.augment_stems(input_stems)

        # Compute target mixing features
        target_mixture = sum(target_stems.values())
        target_features = self.feature_extractor.extract_all_features(
            target_stems, target_mixture
        )

        # Clean up
        del full_stems

        return input_stems, target_stems, target_features


def style_transfer_collate_fn(batch):
    """
    Custom collate function for style transfer training.

    Args:
        batch: List of tuples (input_stems, target_stems, target_features)

    Returns:
        Tuple of (input_stems_dict, target_stems_dict, target_features):
        - input_stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                           Each value is (B, 2, T)
        - target_stems_dict: Dict with same structure
        - target_features: (B, 56)
    """
    input_stems_list = []
    target_stems_list = []
    target_features_list = []

    for input_stems, target_stems, target_features in batch:
        input_stems_list.append(input_stems)
        target_stems_list.append(target_stems)
        target_features_list.append(target_features)

    # Stack stems for each stem type
    input_stems_dict = {}
    target_stems_dict = {}
    for stem_name in ['vocals', 'bass', 'drums', 'other']:
        input_stems_dict[stem_name] = torch.stack(
            [stems[stem_name] for stems in input_stems_list], dim=0
        )  # (B, 2, T)
        target_stems_dict[stem_name] = torch.stack(
            [stems[stem_name] for stems in target_stems_list], dim=0
        )  # (B, 2, T)

    # Stack features
    target_features = torch.stack(target_features_list, dim=0)  # (B, 56)

    return input_stems_dict, target_stems_dict, target_features
