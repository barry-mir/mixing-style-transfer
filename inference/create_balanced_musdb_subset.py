"""
Create a balanced subset of MUSDB18 dataset.

Selects tracks where all 4 stems (vocals, bass, drums, other) have significant energy
(each ≥10% of total energy). For each valid track, extracts the most balanced 10-second
segment and saves it in MUSDB18 format.

This ensures fair style transfer comparisons by avoiding tracks where one stem dominates
(e.g., piano-only tracks with all energy in "other").
"""

import os
import sys
import json
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

def compute_stem_energies(stems_dict):
    """
    Compute RMS energy for each stem.

    Args:
        stems_dict: Dict of {stem_name: audio_array (2, T)}

    Returns:
        energy_dict: Dict of {stem_name: rms_energy}
        energy_percentages: Dict of {stem_name: percentage}
    """
    energies = {}

    for stem_name, audio in stems_dict.items():
        # RMS energy: sqrt(mean(x^2))
        rms = np.sqrt(np.mean(audio ** 2))
        energies[stem_name] = rms

    # Compute percentages
    total_energy = sum(energies.values())
    percentages = {name: (energy / total_energy) * 100
                  for name, energy in energies.items()}

    return energies, percentages


def is_balanced(energy_percentages, min_percentage=10.0):
    """
    Check if all stems have at least min_percentage of total energy.

    Args:
        energy_percentages: Dict of {stem_name: percentage}
        min_percentage: Minimum percentage required for each stem

    Returns:
        is_valid: Boolean
        reason: String describing why invalid (if not valid)
    """
    for stem_name, percentage in energy_percentages.items():
        if percentage < min_percentage:
            return False, f"{stem_name} only {percentage:.1f}% (< {min_percentage}%)"

    return True, "All stems balanced"


def find_best_balanced_window(track_path, window_duration=10.0, stride=1.0,
                              sample_rate=44100, min_percentage=10.0):
    """
    Scan track to find the most balanced 10-second window.

    Args:
        track_path: Path to track directory
        window_duration: Window size in seconds
        stride: Stride for sliding window in seconds
        sample_rate: Audio sample rate
        min_percentage: Minimum percentage for each stem

    Returns:
        best_offset: Offset in seconds of best window (None if no valid window)
        best_info: Dict with energy stats for best window
    """
    stem_names = ['vocals', 'bass', 'drums', 'other']

    # Load full track to get duration
    mixture_path = track_path / 'mixture.wav'
    if not mixture_path.exists():
        return None, {'error': 'mixture.wav not found'}

    # Get duration without loading full audio
    duration = librosa.get_duration(path=str(mixture_path))

    if duration < window_duration:
        return None, {'error': f'Track too short: {duration:.1f}s < {window_duration}s'}

    # Scan with sliding window
    best_offset = None
    best_balance_score = float('inf')  # Lower is better (std dev of percentages)
    best_info = None

    num_windows = int((duration - window_duration) / stride) + 1

    for i in range(num_windows):
        offset = i * stride

        # Load 10-second window for all stems
        stems_dict = {}
        try:
            for stem_name in stem_names:
                stem_path = track_path / f'{stem_name}.wav'
                audio, sr = librosa.load(str(stem_path), sr=sample_rate,
                                       mono=False, offset=offset,
                                       duration=window_duration)

                # Ensure stereo
                if audio.ndim == 1:
                    audio = np.stack([audio, audio])

                stems_dict[stem_name] = audio

        except Exception as e:
            continue  # Skip this window on error

        # Compute energies
        energies, percentages = compute_stem_energies(stems_dict)

        # Check if balanced
        is_valid, reason = is_balanced(percentages, min_percentage)

        if is_valid:
            # Compute balance score (lower std dev = more balanced)
            balance_score = np.std(list(percentages.values()))

            if balance_score < best_balance_score:
                best_balance_score = balance_score
                best_offset = offset
                best_info = {
                    'offset': offset,
                    'energy_percentages': percentages,
                    'energies': energies,
                    'balance_score': balance_score,
                    'std_dev': balance_score
                }

    if best_offset is None:
        return None, {'error': 'No balanced window found', 'min_percentage': min_percentage}

    return best_offset, best_info


def extract_and_save_clip(track_path, offset, output_path, duration=10.0, sample_rate=44100):
    """
    Extract 10-second clip and save in MUSDB format.

    Args:
        track_path: Path to original track directory
        offset: Offset in seconds
        output_path: Path to output directory
        duration: Clip duration in seconds
        sample_rate: Audio sample rate
    """
    stem_names = ['vocals', 'bass', 'drums', 'other', 'mixture']

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract and save each stem
    for stem_name in stem_names:
        stem_file = track_path / f'{stem_name}.wav'

        if not stem_file.exists():
            print(f"  Warning: {stem_file} not found, skipping")
            continue

        # Load segment
        audio, sr = librosa.load(str(stem_file), sr=sample_rate, mono=False,
                                offset=offset, duration=duration)

        # Ensure stereo
        if audio.ndim == 1:
            audio = np.stack([audio, audio])

        # Save (transpose to (T, 2) for soundfile)
        output_file = output_path / f'{stem_name}.wav'
        sf.write(str(output_file), audio.T, sample_rate)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Create balanced MUSDB18 subset')
    parser.add_argument('--input_path', type=str, default='/nas/MUSDB18',
                       help='Path to MUSDB18 dataset')
    parser.add_argument('--output_path', type=str, default='/nas/MUSDB18_Balanced',
                       help='Path for balanced subset output')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'test'],
                       help='Dataset split to process')
    parser.add_argument('--min_percentage', type=float, default=10.0,
                       help='Minimum energy percentage required per stem')
    parser.add_argument('--window_duration', type=float, default=10.0,
                       help='Window duration in seconds')
    parser.add_argument('--stride', type=float, default=1.0,
                       help='Stride for sliding window in seconds')
    parser.add_argument('--sample_rate', type=int, default=44100,
                       help='Audio sample rate')
    args = parser.parse_args()

    input_dir = Path(args.input_path) / args.split
    output_dir = Path(args.output_path) / args.split

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    print("="*80)
    print("MUSDB18 Balanced Subset Creation")
    print("="*80)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Split: {args.split}")
    print(f"Min stem percentage: {args.min_percentage}%")
    print(f"Window: {args.window_duration}s, Stride: {args.stride}s")
    print()

    # Get all tracks
    track_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    print(f"Found {len(track_dirs)} tracks in {args.split} split")
    print()

    # Process each track
    results = []
    valid_tracks = 0
    skipped_tracks = 0

    for track_dir in tqdm(track_dirs, desc="Processing tracks"):
        track_name = track_dir.name

        # Find best balanced window
        best_offset, info = find_best_balanced_window(
            track_dir,
            window_duration=args.window_duration,
            stride=args.stride,
            sample_rate=args.sample_rate,
            min_percentage=args.min_percentage
        )

        if best_offset is None:
            # No valid window found
            skipped_tracks += 1
            results.append({
                'track_name': track_name,
                'valid': False,
                'reason': info.get('error', 'Unknown error')
            })
            tqdm.write(f"  ✗ {track_name}: {info.get('error', 'Unknown error')}")
        else:
            # Valid window found - extract and save
            valid_tracks += 1
            output_track_dir = output_dir / track_name

            try:
                extract_and_save_clip(
                    track_dir,
                    best_offset,
                    output_track_dir,
                    duration=args.window_duration,
                    sample_rate=args.sample_rate
                )

                results.append({
                    'track_name': track_name,
                    'valid': True,
                    'offset': best_offset,
                    'energy_percentages': info['energy_percentages'],
                    'balance_score': info['balance_score'],
                    'std_dev': info['std_dev']
                })

                percentages = info['energy_percentages']
                tqdm.write(f"  ✓ {track_name} @ {best_offset:.1f}s: "
                          f"V={percentages['vocals']:.1f}% "
                          f"B={percentages['bass']:.1f}% "
                          f"D={percentages['drums']:.1f}% "
                          f"O={percentages['other']:.1f}% "
                          f"(std={info['std_dev']:.2f})")

            except Exception as e:
                skipped_tracks += 1
                results.append({
                    'track_name': track_name,
                    'valid': False,
                    'reason': f'Extraction failed: {str(e)}'
                })
                tqdm.write(f"  ✗ {track_name}: Extraction failed - {str(e)}")

    # Save manifest
    manifest = {
        'config': {
            'input_path': str(input_dir),
            'output_path': str(output_dir),
            'split': args.split,
            'min_percentage': args.min_percentage,
            'window_duration': args.window_duration,
            'stride': args.stride,
            'sample_rate': args.sample_rate
        },
        'summary': {
            'total_tracks': len(track_dirs),
            'valid_tracks': valid_tracks,
            'skipped_tracks': skipped_tracks,
            'valid_percentage': (valid_tracks / len(track_dirs)) * 100
        },
        'tracks': results
    }

    manifest_path = output_dir / 'manifest.json'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total tracks processed: {len(track_dirs)}")
    print(f"Valid balanced tracks: {valid_tracks} ({(valid_tracks/len(track_dirs))*100:.1f}%)")
    print(f"Skipped tracks: {skipped_tracks} ({(skipped_tracks/len(track_dirs))*100:.1f}%)")
    print()
    print(f"Output saved to: {output_dir}")
    print(f"Manifest saved to: {manifest_path}")
    print()

    # Print some rejection statistics
    if skipped_tracks > 0:
        print("Rejection reasons:")
        rejection_reasons = {}
        for result in results:
            if not result['valid']:
                reason = result['reason']
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

        for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
            print(f"  {reason}: {count} tracks")

    print("="*80)


if __name__ == "__main__":
    main()
