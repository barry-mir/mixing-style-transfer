"""
Fast preprocessing for FMA dataset with parallel data loading and batched inference.

Key optimizations:
1. Parallel data loading with DataLoader (multi-worker audio loading)
2. Batched GPU inference (process multiple tracks at once)
3. Async MP3 encoding (don't block GPU while encoding)
4. Increased inference batch_size in SCNet config
5. Pin memory for faster CPU->GPU transfer

Usage:
    python scripts/preprocess_fma_separation_fast.py \
        --batch_size 8 \
        --num_workers 8 \
        --inference_batch_size 16
"""

import os
import sys
import glob
import torch
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import subprocess
from torch.utils.data import Dataset, DataLoader
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading
import fcntl
import time

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Music-Source-Separation-Training'))

# Import after path setup
from utils.settings import get_model_from_config
from utils.model_utils import demix, load_start_checkpoint
from utils.audio_utils import normalize_audio, denormalize_audio


class FMADataset(Dataset):
    """Dataset for parallel loading of FMA audio files."""

    def __init__(self, audio_files, output_dir, skip_existing=True, lock_dir=None):
        self.audio_files = audio_files
        self.output_dir = output_dir
        self.skip_existing = skip_existing
        self.lock_dir = lock_dir or os.path.join(output_dir, '.locks')

        # Create lock directory
        os.makedirs(self.lock_dir, exist_ok=True)

        # Filter out already processed files
        if skip_existing:
            self.audio_files = [
                f for f in audio_files
                if not self._is_processed(f)
            ]

    def _is_processed(self, audio_path):
        """Check if track is already processed."""
        track_id = Path(audio_path).stem
        output_dir = os.path.join(self.output_dir, track_id)
        return all([
            os.path.exists(os.path.join(output_dir, f"{stem}.mp3"))
            for stem in ['vocals', 'bass', 'drums', 'other']
        ])

    def _try_acquire_lock(self, track_id):
        """Try to acquire exclusive lock for processing this track."""
        lock_file = os.path.join(self.lock_dir, f"{track_id}.lock")
        try:
            # Create lock file and acquire exclusive lock
            lock_fd = os.open(lock_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            return lock_fd, lock_file
        except FileExistsError:
            # Another process is already processing this track
            return None, None

    def _release_lock(self, lock_fd, lock_file):
        """Release lock after processing."""
        if lock_fd is not None:
            try:
                os.close(lock_fd)
                os.remove(lock_file)
            except:
                pass

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        """Load and return audio tensor."""
        audio_path = self.audio_files[idx]
        track_id = Path(audio_path).stem

        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=44100, mono=False)

            # Ensure stereo
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=0)

            # Convert to torch
            audio_torch = torch.from_numpy(audio).float()

            return audio_torch, track_id, audio_path

        except Exception as e:
            print(f"\n⚠️  Error loading {audio_path}: {e}")
            # Return None to signal skip
            return None, track_id, audio_path


def collate_fn(batch):
    """
    Custom collate to handle variable-length audio.
    Filters out None values (skipped tracks) - no cropping, process each track at full length.
    """
    # Filter out None values (skipped tracks)
    valid_batch = [(a, tid, path) for a, tid, path in batch if a is not None]

    # If all tracks were skipped, return empty batch
    if len(valid_batch) == 0:
        return None, None, None, None

    audios, track_ids, audio_paths = zip(*valid_batch)

    # Return as lists - each track will be processed individually at its full length
    original_lengths = [a.shape[1] for a in audios]

    return list(audios), original_lengths, list(track_ids), list(audio_paths)


class SCNetSeparatorBatched:
    """Batched SCNet separator for fast preprocessing."""

    def __init__(self, model_path, config_path, device='cuda', inference_batch_size=16):
        self.device = device

        # Load model and config
        model, config = get_model_from_config('scnet_masked', config_path)
        checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')

        # Create dummy args
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

        # Override inference batch size for faster processing
        self.config.inference.batch_size = inference_batch_size
        print(f"  → SCNet inference batch_size: {inference_batch_size}")

    @torch.no_grad()
    def separate_batch(self, audio_list):
        """
        Separate a list of audio tracks (each at full length).

        Args:
            audio_list: List of audio tensors, each (2, T) where T can vary

        Returns:
            List of stems_dict for each track
        """
        results = []

        for audio_tensor in audio_list:
            audio = audio_tensor.cpu().numpy()

            # Normalize
            if self.config.inference.get('normalize', False):
                audio, norm_params = normalize_audio(audio)
            else:
                norm_params = None

            # Separate using demix
            waveforms = demix(
                self.config,
                self.model,
                audio,
                self.device,
                model_type='scnet_masked',
                pbar=False
            )

            # Denormalize
            if norm_params is not None:
                for stem in waveforms:
                    waveforms[stem] = denormalize_audio(waveforms[stem], norm_params)

            # Convert to torch
            stems_dict = {
                stem: torch.from_numpy(waveforms[stem]).float()
                for stem in ['vocals', 'bass', 'drums', 'other']
            }

            results.append(stems_dict)

        return results


class AsyncMP3Encoder:
    """Async MP3 encoder to not block GPU processing."""

    def __init__(self, max_workers=4, bitrate='192k'):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.bitrate = bitrate
        self.futures = []

    def encode_stem(self, audio, output_path, sample_rate=44100):
        """Submit MP3 encoding job (non-blocking)."""
        future = self.executor.submit(
            self._encode_stem_sync, audio, output_path, sample_rate
        )
        self.futures.append(future)

    def _encode_stem_sync(self, audio, output_path, sample_rate):
        """Synchronous MP3 encoding (runs in thread pool)."""
        try:
            # Save as temp WAV
            temp_wav = output_path.replace('.mp3', '_temp.wav')
            audio_np = audio.cpu().numpy().T  # (T, 2)
            sf.write(temp_wav, audio_np, sample_rate)

            # Convert to MP3
            cmd = [
                'ffmpeg', '-y', '-i', temp_wav,
                '-codec:a', 'libmp3lame',
                '-b:a', self.bitrate,
                '-ar', str(sample_rate),
                output_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            # Remove temp WAV
            os.remove(temp_wav)
            return True

        except Exception as e:
            print(f"\n⚠️  MP3 encoding error for {output_path}: {e}")
            return False

    def wait_all(self):
        """Wait for all encoding jobs to complete."""
        for future in as_completed(self.futures):
            future.result()
        self.futures.clear()

    def shutdown(self):
        """Shutdown executor."""
        self.wait_all()
        self.executor.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser(description='Fast FMA preprocessing with batched inference')
    parser.add_argument('--input_dir', type=str, default='/nas/FMA/fma_full/',
                        help='Input FMA dataset directory')
    parser.add_argument('--output_dir', type=str, default='/nas/FMA/fma_separated/',
                        help='Output directory for separated stems')
    parser.add_argument('--scnet_model', type=str,
                        default='Music-Source-Separation-Training/model_scnet_masked_ep_111_sdr_9.8286.ckpt',
                        help='Path to SCNet checkpoint')
    parser.add_argument('--scnet_config', type=str,
                        default='Music-Source-Separation-Training/configs/config_musdb18_scnet_xl_ihf.yaml',
                        help='Path to SCNet config')
    parser.add_argument('--bitrate', type=str, default='192k',
                        help='MP3 bitrate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device for inference')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for data loading (number of tracks loaded in parallel)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of worker processes for data loading')
    parser.add_argument('--inference_batch_size', type=int, default=16,
                        help='Batch size for SCNet inference (chunks per forward pass)')
    parser.add_argument('--mp3_workers', type=int, default=8,
                        help='Number of threads for MP3 encoding')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip already processed tracks')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of tracks')

    args = parser.parse_args()

    print("="*80)
    print("Fast FMA Dataset Preprocessing: SCNet Source Separation")
    print("="*80)
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size (tracks): {args.batch_size}")
    print(f"Inference batch size (chunks): {args.inference_batch_size}")
    print(f"Data loading workers: {args.num_workers}")
    print(f"MP3 encoding workers: {args.mp3_workers}")
    print(f"MP3 Bitrate: {args.bitrate}")
    print(f"Skip existing: {args.skip_existing}")

    # Check ffmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("\n❌ ERROR: ffmpeg not found")
        return

    # Find audio files
    print("\n🔍 Finding audio files...")
    audio_files = []
    for ext in ['*.mp3', '*.wav', '*.flac']:
        audio_files.extend(glob.glob(os.path.join(args.input_dir, '**', ext), recursive=True))

    if args.limit:
        audio_files = audio_files[:args.limit]

    print(f"✓ Found {len(audio_files)} audio files")

    # Initialize SCNet
    print("\n📦 Loading SCNet model...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    scnet = SCNetSeparatorBatched(
        model_path=args.scnet_model,
        config_path=args.scnet_config,
        device=device,
        inference_batch_size=args.inference_batch_size
    )
    print(f"✓ SCNet loaded on {device}")

    # Create dataset and dataloader
    print("\n📂 Creating dataset and dataloader...")
    dataset = FMADataset(
        audio_files,
        args.output_dir,
        skip_existing=args.skip_existing
    )

    if len(dataset) == 0:
        print("✓ All tracks already processed!")
        return

    print(f"  → {len(dataset)} tracks to process")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True if args.num_workers > 0 else False
    )

    # Create async MP3 encoder
    mp3_encoder = AsyncMP3Encoder(max_workers=args.mp3_workers, bitrate=args.bitrate)

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all batches
    print("\n🎵 Processing tracks with batched inference...")
    print("="*80)

    success_count = 0
    error_count = 0
    skip_count = 0

    try:
        pbar = tqdm(dataloader, desc="Processing batches", total=len(dataloader))

        for audio_list, lengths, track_ids, audio_paths in pbar:
            # Skip empty batches (all tracks had errors)
            if audio_list is None:
                skip_count += args.batch_size
                continue

            # Try to acquire locks for all tracks in batch
            locks = []
            tracks_to_process = []
            tracks_to_skip = []
            audio_to_process = []

            for i, track_id in enumerate(track_ids):
                # Check if already processed (double-check)
                if dataset._is_processed(audio_paths[i]):
                    tracks_to_skip.append(track_id)
                    locks.append((None, None))
                    continue

                # Try to acquire lock
                lock_fd, lock_file = dataset._try_acquire_lock(track_id)
                if lock_fd is not None:
                    locks.append((lock_fd, lock_file))
                    tracks_to_process.append(i)
                    audio_to_process.append(audio_list[i].to(device, non_blocking=True))
                else:
                    # Another GPU is processing this track
                    tracks_to_skip.append(track_id)
                    locks.append((None, None))

            # Skip batch if nothing to process
            if not tracks_to_process:
                # Release any acquired locks
                for lock_fd, lock_file in locks:
                    dataset._release_lock(lock_fd, lock_file)
                continue

            try:
                # Separate tracks (GPU processing) - each at full length
                stems_list = scnet.separate_batch(audio_to_process)

                # Save stems (async, non-blocking)
                for idx, (stems_dict, i) in enumerate(zip(stems_list, tracks_to_process)):
                    track_id = track_ids[i]

                    output_dir = os.path.join(args.output_dir, track_id)
                    os.makedirs(output_dir, exist_ok=True)

                    for stem_name, stem_audio in stems_dict.items():
                        output_path = os.path.join(output_dir, f"{stem_name}.mp3")
                        mp3_encoder.encode_stem(stem_audio, output_path)

                    success_count += 1

                    # Release lock after MP3 encoding is submitted
                    dataset._release_lock(locks[i][0], locks[i][1])

                # Update progress
                skip_count += len(tracks_to_skip)
                pbar.set_postfix({
                    'success': success_count,
                    'errors': error_count,
                    'skipped': skip_count
                })

            except Exception as e:
                print(f"\n⚠️  Batch processing error: {e}")
                error_count += len(tracks_to_process)

                # Release locks on error
                for lock_fd, lock_file in locks:
                    dataset._release_lock(lock_fd, lock_file)

        pbar.close()

        # Wait for all MP3 encoding to finish
        print("\n⏳ Waiting for MP3 encoding to complete...")
        mp3_encoder.wait_all()

    finally:
        mp3_encoder.shutdown()

    # Summary
    print("\n" + "="*80)
    print("Processing Complete!")
    print("="*80)
    print(f"✓ Successfully processed: {success_count}")
    print(f"⊘ Skipped (error/locked/already processed): {skip_count}")
    print(f"✗ Errors: {error_count}")
    print(f"\n📁 Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
