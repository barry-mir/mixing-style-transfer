"""
End-to-End Style Transfer with Source Separation

Performs style transfer starting from raw audio clips (no pre-separated stems required).
Supports two optimization modes:
  1. embeddings: Optimize TCN to match target embedding (512-dim learned space)
  2. features: Optimize TCN directly on mixing features (56-dim hand-crafted space)
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import soundfile as sf
import librosa
import json
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from model import MixingStyleEncoder
from mixing_utils import MixingFeatureExtractor
from tcn_mixer import create_tcn_mixer
from data import SCNetSeparator


def load_audio(file_path, target_sr=44100, segment_duration=10.0, segment_offset=0.0):
    """
    Load audio file and ensure stereo format at target sample rate.

    Args:
        file_path: Path to audio file
        target_sr: Target sample rate (default 44100)
        segment_duration: Duration of segment to extract in seconds (default 10.0)
        segment_offset: Offset in seconds where to start extraction (default 0.0)

    Returns:
        audio: (2, T) tensor where T = segment_duration * target_sr
    """
    audio, sr = sf.read(file_path)

    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=target_sr).T

    # Ensure stereo (2, T)
    if audio.ndim == 1:
        audio = np.stack([audio, audio])  # Duplicate mono to stereo
    else:
        audio = audio.T  # (T, 2) → (2, T)

    # Ensure exactly 2 channels
    if audio.shape[0] != 2:
        audio = audio[:2, :]  # Take first 2 channels

    # Extract segment
    target_length = int(segment_duration * target_sr)
    start_sample = int(segment_offset * target_sr)
    end_sample = start_sample + target_length

    # Crop or pad to exact length
    if start_sample >= audio.shape[1]:
        # Offset beyond audio length, return zeros
        audio = np.zeros((2, target_length), dtype=np.float32)
    elif end_sample > audio.shape[1]:
        # Audio shorter than requested, pad with zeros
        segment = audio[:, start_sample:]
        padding = np.zeros((2, target_length - segment.shape[1]), dtype=np.float32)
        audio = np.concatenate([segment, padding], axis=1)
    else:
        # Normal case: extract segment
        audio = audio[:, start_sample:end_sample]

    return torch.from_numpy(audio).float()


def compute_mixing_embedding(mixture_audio, stems_dict, model, feature_extractor, device, requires_grad=False):
    """
    Compute mixing embedding using the mixing representation model.

    Args:
        mixture_audio: (2, T) tensor
        stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other'], values (2, T)
        model: MixingStyleEncoder instance
        feature_extractor: MixingFeatureExtractor instance
        device: torch device
        requires_grad: If True, allow gradients to flow through computation

    Returns:
        embedding: (512,) tensor
    """
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

    return embedding.squeeze(0)  # (512,)


def extract_mixing_features(mixture_audio, stems_dict, feature_extractor, requires_grad=False):
    """
    Extract mixing features directly (without embedding computation).

    Args:
        mixture_audio: (2, T) tensor
        stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other'], values (2, T)
        feature_extractor: MixingFeatureExtractor instance
        requires_grad: If True, allow gradients to flow through computation

    Returns:
        features: (56,) tensor
    """
    if not requires_grad:
        with torch.no_grad():
            stems_cpu = {k: v.detach().cpu() if v.is_cuda else v for k, v in stems_dict.items()}
            mixture_cpu = mixture_audio.detach().cpu() if mixture_audio.is_cuda else mixture_audio
            features = feature_extractor.extract_all_features(stems_cpu, mixture_cpu)
    else:
        # WITH gradients - no torch.no_grad()!
        stems_cpu = {k: v.cpu() for k, v in stems_dict.items()}
        mixture_cpu = mixture_audio.cpu()
        features = feature_extractor.extract_all_features(stems_cpu, mixture_cpu)

    return features


def optimize_tcn_embeddings(
    tcn,
    stems_input,
    target_emb,
    mixing_model,
    feature_extractor,
    device,
    num_steps=500,
    lr=0.001,
    verbose=True
):
    """
    Optimize TCN parameters to match target embedding.

    Returns:
        dict with processed_stems, processed_mixture, iteration_log, etc.
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

    iteration_log = []
    best_distance = float('inf')
    best_state = None

    for step in tqdm(range(num_steps), desc="Optimizing (embeddings)"):
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

        # Track metrics (no grad)
        with torch.no_grad():
            distance = loss.item()

            # Track best
            if distance < best_distance:
                best_distance = distance
                best_state = {
                    'processed_stems': {k: v.detach().cpu() for k, v in processed_stems.items()},
                    'processed_mixture': processed_mixture.detach().cpu(),
                    'tcn_state': {k: v.cpu().clone() for k, v in tcn.state_dict().items()}
                }

            iteration_log.append({
                'step': step,
                'embedding_distance': distance
            })

        # Backward pass
        loss.backward()
        optimizer.step()

        if verbose and (step % 50 == 0 or step == num_steps - 1):
            print(f"  Step {step:3d}/{num_steps}: embedding_distance = {distance:.4f}, best = {best_distance:.4f}")

    return {
        'processed_stems': best_state['processed_stems'],
        'processed_mixture': best_state['processed_mixture'],
        'iteration_log': iteration_log,
        'final_distance': best_distance,
        'converged': best_distance < iteration_log[0]['embedding_distance'] * 0.8
    }


def optimize_tcn_features(
    tcn,
    stems_input,
    target_features,
    feature_extractor,
    device,
    num_steps=500,
    lr=0.001,
    verbose=True
):
    """
    Optimize TCN parameters to match target features directly.

    Returns:
        dict with processed_stems, processed_mixture, iteration_log, etc.
    """
    # Normalize target features
    target_features = F.normalize(target_features, p=2, dim=0)

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

    iteration_log = []
    best_distance = float('inf')
    best_state = None

    for step in tqdm(range(num_steps), desc="Optimizing (features)"):
        optimizer.zero_grad()

        # Process stems through TCN
        processed_stacked = tcn(stacked_stems)  # (1, 8, T)

        # Unstack processed stems
        processed_stems = {}
        for i, stem_name in enumerate(stem_order):
            processed_stems[stem_name] = processed_stacked[0, i*2:(i+1)*2, :]  # (2, T)

        # Sum to create mixture
        processed_mixture = sum(processed_stems.values())  # (2, T)

        # Extract features WITH gradients
        features = extract_mixing_features(
            processed_mixture,
            processed_stems,
            feature_extractor,
            requires_grad=True
        )

        # Move to device and normalize
        features = features.to(device)
        features = F.normalize(features, p=2, dim=0)

        # Compute loss using mean squared error (MSE) between features
        loss = F.mse_loss(features, target_features.to(device))

        # Track metrics (no grad)
        with torch.no_grad():
            distance = loss.item()

            # Track best
            if distance < best_distance:
                best_distance = distance
                best_state = {
                    'processed_stems': {k: v.detach().cpu() for k, v in processed_stems.items()},
                    'processed_mixture': processed_mixture.detach().cpu(),
                    'tcn_state': {k: v.cpu().clone() for k, v in tcn.state_dict().items()}
                }

            iteration_log.append({
                'step': step,
                'feature_distance': distance
            })

        # Backward pass
        loss.backward()
        optimizer.step()

        if verbose and (step % 50 == 0 or step == num_steps - 1):
            print(f"  Step {step:3d}/{num_steps}: feature_distance = {distance:.4f}, best = {best_distance:.4f}")

    return {
        'processed_stems': best_state['processed_stems'],
        'processed_mixture': best_state['processed_mixture'],
        'iteration_log': iteration_log,
        'final_distance': best_distance,
        'converged': best_distance < iteration_log[0]['feature_distance'] * 0.8
    }


class MultiResolutionSTFTLoss(torch.nn.Module):
    """
    Multi-resolution STFT loss for audio quality preservation.
    Computes spectral loss at multiple FFT sizes.
    """
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[256, 512, 128],
                 win_sizes=[1024, 2048, 512], window='hann'):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        self.window = window

    def stft(self, x, fft_size, hop_size, win_size):
        """Compute STFT"""
        # x: (B, C, T) or (C, T)
        if x.ndim == 2:
            x = x.unsqueeze(0)  # (1, C, T)

        B, C, T = x.shape
        # Merge batch and channel for STFT
        x_flat = x.reshape(B * C, T)  # (B*C, T)

        # Create window
        window_tensor = torch.hann_window(win_size, device=x.device)

        # STFT
        spec = torch.stft(
            x_flat,
            n_fft=fft_size,
            hop_length=hop_size,
            win_length=win_size,
            window=window_tensor,
            return_complex=True
        )  # (B*C, freq, time)

        return spec

    def spectral_convergence(self, x_mag, y_mag):
        """Spectral convergence loss"""
        return torch.norm(y_mag - x_mag, p='fro') / torch.norm(y_mag, p='fro')

    def log_stft_magnitude(self, x_mag, y_mag):
        """Log STFT magnitude loss"""
        return F.l1_loss(torch.log(x_mag + 1e-5), torch.log(y_mag + 1e-5))

    def forward(self, x, y):
        """
        Args:
            x: predicted audio (B, C, T) or (C, T)
            y: target audio (B, C, T) or (C, T)

        Returns:
            loss: scalar
        """
        total_loss = 0.0

        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            # Compute STFT
            x_spec = self.stft(x, fft_size, hop_size, win_size)
            y_spec = self.stft(y, fft_size, hop_size, win_size)

            # Magnitude
            x_mag = torch.abs(x_spec)
            y_mag = torch.abs(y_spec)

            # Spectral convergence + log magnitude
            sc_loss = self.spectral_convergence(x_mag, y_mag)
            log_loss = self.log_stft_magnitude(x_mag, y_mag)

            total_loss += sc_loss + log_loss

        return total_loss / len(self.fft_sizes)


def optimize_tcn_cycle_consistency(
    tcn_forward,
    tcn_backward,
    stems_input,
    target_emb,
    mixing_model,
    feature_extractor,
    device,
    num_steps=500,
    lr=0.001,
    lambda_recon=10.0,
    lambda_waveform=1.0,
    verbose=True,
    optimize_target_type='embeddings'
):
    """
    Optimize TCN with cycle-consistency:
    - Forward: input -> TCN1 -> transferred (match target style)
    - Backward: transferred -> TCN2 -> reconstructed (match input audio)

    Args:
        tcn_forward: First TCN (input -> transferred)
        tcn_backward: Second TCN (transferred -> reconstructed)
        stems_input: Input stems dict
        target_emb: Target embedding or features
        mixing_model: Encoder for embeddings
        feature_extractor: For features
        device: torch device
        num_steps: Optimization steps
        lr: Learning rate
        lambda_recon: Weight for reconstruction MRSTFT loss
        lambda_waveform: Weight for waveform L1 loss
        verbose: Print progress
        optimize_target_type: 'embeddings' or 'features'

    Returns:
        dict with processed_stems, processed_mixture, iteration_log, etc.
    """
    # Stack stems for TCN: (8, T)
    stem_order = ['vocals', 'bass', 'drums', 'other']
    stems_list = [stems_input[name] for name in stem_order]
    stacked_stems = torch.cat(stems_list, dim=0).unsqueeze(0).to(device)  # (1, 8, T)
    stacked_stems.requires_grad = False

    # Initialize TCNs
    tcn_forward = tcn_forward.to(device)
    tcn_backward = tcn_backward.to(device)
    tcn_forward.train()
    tcn_backward.train()

    # MRSTFT loss for reconstruction
    mrstft_loss = MultiResolutionSTFTLoss().to(device)

    # Optimizer for both TCNs
    optimizer = optim.Adam(
        list(tcn_forward.parameters()) + list(tcn_backward.parameters()),
        lr=lr
    )

    # Normalize target
    if optimize_target_type == 'embeddings':
        target_emb = F.normalize(target_emb, p=2, dim=0)

    iteration_log = []
    best_distance = float('inf')
    best_state = None

    for step in tqdm(range(num_steps), desc="Optimizing (cycle-consistency)"):
        optimizer.zero_grad()

        # ===== FORWARD PASS: input -> transferred =====
        transferred_stacked = tcn_forward(stacked_stems)  # (1, 8, T)

        # Unstack transferred stems
        transferred_stems = {}
        for i, stem_name in enumerate(stem_order):
            transferred_stems[stem_name] = transferred_stacked[0, i*2:(i+1)*2, :]  # (2, T)

        # Sum to create mixture
        transferred_mixture = sum(transferred_stems.values())  # (2, T)

        # Compute style loss (embedding or feature)
        if optimize_target_type == 'embeddings':
            # Compute embedding WITH gradients
            emb = compute_mixing_embedding(
                transferred_mixture,
                transferred_stems,
                mixing_model,
                feature_extractor,
                device,
                requires_grad=True
            )
            emb = F.normalize(emb, p=2, dim=0)

            # Style loss: negative cosine similarity
            style_loss = 1.0 - F.cosine_similarity(emb.unsqueeze(0), target_emb.unsqueeze(0))
        else:
            # Feature-based style loss
            pred_features = feature_extractor.extract_all_features(
                transferred_stems,
                transferred_mixture
            )
            style_loss = F.mse_loss(pred_features, target_emb)

        # ===== BACKWARD PASS: transferred -> reconstructed =====
        # Detach transferred to prevent gradient flow from reconstruction to style
        # (Only TCN_backward should be trained for reconstruction)
        reconstructed_stacked = tcn_backward(transferred_stacked)  # (1, 8, T)

        # Reconstruction losses
        # 1. MRSTFT loss (spectral quality)
        recon_mrstft_loss = mrstft_loss(reconstructed_stacked, stacked_stems)

        # 2. Waveform L1 loss (time-domain)
        recon_l1_loss = F.l1_loss(reconstructed_stacked, stacked_stems)

        # ===== TOTAL LOSS =====
        loss = style_loss + lambda_recon * recon_mrstft_loss + lambda_waveform * recon_l1_loss

        # Track metrics (no grad)
        with torch.no_grad():
            style_distance = style_loss.item()
            mrstft_distance = recon_mrstft_loss.item()
            l1_distance = recon_l1_loss.item()
            total_distance = loss.item()

            # Track best based on style loss
            if style_distance < best_distance:
                best_distance = style_distance
                best_state = {
                    'processed_stems': {k: v.detach().cpu() for k, v in transferred_stems.items()},
                    'processed_mixture': transferred_mixture.detach().cpu(),
                    'tcn_forward_state': {k: v.cpu().clone() for k, v in tcn_forward.state_dict().items()},
                    'tcn_backward_state': {k: v.cpu().clone() for k, v in tcn_backward.state_dict().items()}
                }

            iteration_log.append({
                'step': step,
                'style_loss': style_distance,
                'recon_mrstft_loss': mrstft_distance,
                'recon_l1_loss': l1_distance,
                'total_loss': total_distance
            })

        # Backward pass
        loss.backward()
        optimizer.step()

        if verbose and (step % 50 == 0 or step == num_steps - 1):
            print(f"  Step {step:3d}/{num_steps}: style={style_distance:.4f}, "
                  f"mrstft={mrstft_distance:.4f}, l1={l1_distance:.4f}, "
                  f"total={total_distance:.4f}, best_style={best_distance:.4f}")

    return {
        'processed_stems': best_state['processed_stems'],
        'processed_mixture': best_state['processed_mixture'],
        'iteration_log': iteration_log,
        'final_distance': best_distance,
        'converged': best_distance < iteration_log[0]['style_loss'] * 0.8,
        'tcn_forward_state': best_state['tcn_forward_state'],
        'tcn_backward_state': best_state['tcn_backward_state']
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='End-to-End Style Transfer with Source Separation')
    parser.add_argument('--input_audio', type=str, required=True, help='Path to input audio file')
    parser.add_argument('--target_audio', type=str, required=True, help='Path to target audio file')
    parser.add_argument('--optimize_target', type=str, choices=['embeddings', 'features'], default='embeddings',
                        help='Optimization target: embeddings (512-dim) or features (56-dim)')
    parser.add_argument('--output_dir', type=str, default='e2e_style_transfer_output',
                        help='Output directory for results')
    parser.add_argument('--mixing_checkpoint', type=str,
                        default='/nas/mixing-representation/checkpoints_baseline/best_model.pt',
                        help='Path to trained mixing model checkpoint')
    parser.add_argument('--scnet_model', type=str,
                        default='../Music-Source-Separation-Training/model_scnet_masked_ep_111_sdr_9.8286.ckpt',
                        help='Path to SCNet model checkpoint')
    parser.add_argument('--scnet_config', type=str,
                        default='../Music-Source-Separation-Training/configs/config_musdb18_scnet_xl_ihf.yaml',
                        help='Path to SCNet config file')
    parser.add_argument('--num_steps', type=int, default=250, help='Number of optimization steps')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--receptive_field', type=float, default=5.2,
                        help='TCN receptive field in seconds (default: 5.2s, reference architecture)')
    parser.add_argument('--use_detailed_spectral', action='store_true', default=False,
                        help='Use detailed spectral features (frequency curve) instead of 3-band')
    parser.add_argument('--n_spectral_bins', type=int, default=32,
                        help='Number of spectral bins for detailed spectral features')
    parser.add_argument('--segment_duration', type=float, default=10.0, help='Audio segment duration in seconds (default: 10.0)')
    parser.add_argument('--segment_offset', type=float, default=0.0, help='Segment offset in seconds (default: 0.0 for beginning)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')

    # Cycle-consistency arguments
    parser.add_argument('--use_cycle_consistency', action='store_true', default=False,
                        help='Use cycle-consistency loss (two-stage: forward + reconstruction)')
    parser.add_argument('--lambda_recon', type=float, default=10.0,
                        help='Weight for reconstruction MRSTFT loss (default: 10.0)')
    parser.add_argument('--lambda_waveform', type=float, default=1.0,
                        help='Weight for waveform L1 loss (default: 1.0)')

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("="*80)
    print("End-to-End Style Transfer with Source Separation")
    print("="*80)
    print(f"Input audio: {args.input_audio}")
    print(f"Target audio: {args.target_audio}")
    print(f"Optimize target: {args.optimize_target}")
    print(f"Output directory: {args.output_dir}")
    print(f"Segment duration: {args.segment_duration}s (offset: {args.segment_offset}s)")
    print(f"Device: {device}")
    print()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_stems_dir = output_dir / 'input_stems'
    target_stems_dir = output_dir / 'target_stems'
    input_stems_dir.mkdir(exist_ok=True)
    target_stems_dir.mkdir(exist_ok=True)

    # Load audio files (extract segment)
    print(f"Loading audio files ({args.segment_duration}s segment from offset {args.segment_offset}s)...")
    input_audio = load_audio(args.input_audio, segment_duration=args.segment_duration, segment_offset=args.segment_offset)
    target_audio = load_audio(args.target_audio, segment_duration=args.segment_duration, segment_offset=args.segment_offset)
    print(f"  Input: {input_audio.shape} ({input_audio.shape[1]/44100:.2f}s)")
    print(f"  Target: {target_audio.shape} ({target_audio.shape[1]/44100:.2f}s)")
    print()

    # Save original audio
    sf.write(str(output_dir / 'input_original.wav'), input_audio.numpy().T, 44100)
    sf.write(str(output_dir / 'target_original.wav'), target_audio.numpy().T, 44100)

    # Initialize SCNet separator
    print("Initializing SCNet separator...")
    separator = SCNetSeparator(
        model_path=args.scnet_model,
        config_path=args.scnet_config,
        device=device
    )
    print()

    # Separate input audio
    print("Separating input audio...")
    input_stems = separator.separate(input_audio.to(device))
    print("  Extracted stems: vocals, bass, drums, other")

    # Save input stems
    for stem_name, stem_audio in input_stems.items():
        sf.write(str(input_stems_dir / f'{stem_name}.wav'), stem_audio.cpu().numpy().T, 44100)
    print()

    # Separate target audio
    print("Separating target audio...")
    target_stems = separator.separate(target_audio.to(device))
    print("  Extracted stems: vocals, bass, drums, other")

    # Save target stems
    for stem_name, stem_audio in target_stems.items():
        sf.write(str(target_stems_dir / f'{stem_name}.wav'), stem_audio.cpu().numpy().T, 44100)
    print()

    # Initialize mixing feature extractor
    print("Initializing mixing feature extractor...")
    feature_extractor = MixingFeatureExtractor(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512,
        use_detailed_spectral=args.use_detailed_spectral,
        n_spectral_bins=args.n_spectral_bins
    )
    print()

    # Compute target embedding and features for tracking
    print("Computing target representations...")
    target_features = extract_mixing_features(target_audio, target_stems, feature_extractor)
    target_features_norm = F.normalize(target_features, p=2, dim=0)
    print(f"  Target features: {target_features.shape}")

    if args.optimize_target == 'embeddings':
        # Load mixing model
        print("Loading mixing representation model...")
        mixing_model = MixingStyleEncoder(
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

        checkpoint = torch.load(args.mixing_checkpoint, map_location=device)
        mixing_model.load_state_dict(checkpoint['model_state_dict'])
        mixing_model.eval()
        print(f"  Model loaded from epoch {checkpoint['epoch']}")

        # Compute target embedding
        target_emb = compute_mixing_embedding(target_audio, target_stems, mixing_model, feature_extractor, device)
        target_emb_norm = F.normalize(target_emb, p=2, dim=0)
        print(f"  Target embedding: {target_emb.shape}")
        print()
    else:
        mixing_model = None
        target_emb_norm = None
        print()

    # Create TCN mixer(s)
    if args.use_cycle_consistency:
        print(f"Initializing TWO TCN mixers for cycle-consistency (receptive field: {args.receptive_field}s)...")
        tcn_forward = create_tcn_mixer(receptive_field_seconds=args.receptive_field)
        tcn_backward = create_tcn_mixer(receptive_field_seconds=args.receptive_field)
        print(f"  Forward TCN parameters: {sum(p.numel() for p in tcn_forward.parameters()):,}")
        print(f"  Backward TCN parameters: {sum(p.numel() for p in tcn_backward.parameters()):,}")
        print(f"  Total parameters: {sum(p.numel() for p in tcn_forward.parameters()) + sum(p.numel() for p in tcn_backward.parameters()):,}")
    else:
        print(f"Initializing TCN mixer (receptive field: {args.receptive_field}s)...")
        tcn = create_tcn_mixer(receptive_field_seconds=args.receptive_field)
    print()

    # Compute initial distances
    print("Computing initial distances...")
    initial_features = extract_mixing_features(input_audio, input_stems, feature_extractor)
    initial_features_norm = F.normalize(initial_features, p=2, dim=0)
    initial_feature_distance = (1.0 - F.cosine_similarity(
        initial_features_norm.unsqueeze(0), target_features_norm.unsqueeze(0)
    )).item()
    print(f"  Initial feature distance: {initial_feature_distance:.4f}")

    if args.optimize_target == 'embeddings':
        initial_emb = compute_mixing_embedding(input_audio, input_stems, mixing_model, feature_extractor, device)
        initial_emb_norm = F.normalize(initial_emb, p=2, dim=0)
        initial_embedding_distance = (1.0 - F.cosine_similarity(
            initial_emb_norm.unsqueeze(0), target_emb_norm.unsqueeze(0)
        )).item()
        print(f"  Initial embedding distance: {initial_embedding_distance:.4f}")
    else:
        initial_embedding_distance = None
    print()

    # Run optimization
    if args.use_cycle_consistency:
        print(f"Running cycle-consistency optimization ({args.num_steps} steps, lr={args.lr})...")
        print(f"  lambda_recon (MRSTFT): {args.lambda_recon}")
        print(f"  lambda_waveform (L1): {args.lambda_waveform}")
        print()

        # Select target based on optimize_target
        if args.optimize_target == 'embeddings':
            target = target_emb_norm
        else:
            target = target_features_norm

        result = optimize_tcn_cycle_consistency(
            tcn_forward,
            tcn_backward,
            input_stems,
            target,
            mixing_model,
            feature_extractor,
            device,
            num_steps=args.num_steps,
            lr=args.lr,
            lambda_recon=args.lambda_recon,
            lambda_waveform=args.lambda_waveform,
            verbose=True,
            optimize_target_type=args.optimize_target
        )
    else:
        print(f"Running optimization ({args.num_steps} steps, lr={args.lr})...")
        print()

        if args.optimize_target == 'embeddings':
            result = optimize_tcn_embeddings(
                tcn,
                input_stems,
                target_emb_norm,
                mixing_model,
                feature_extractor,
                device,
                num_steps=args.num_steps,
                lr=args.lr,
                verbose=True
            )
        else:  # features
            result = optimize_tcn_features(
                tcn,
                input_stems,
                target_features_norm,
                feature_extractor,
                device,
                num_steps=args.num_steps,
                lr=args.lr,
                verbose=True
            )

    print()
    print("Optimization complete!")
    print()

    # Compute final distances (both embedding and features)
    print("Computing final distances...")
    final_features = extract_mixing_features(
        result['processed_mixture'],
        result['processed_stems'],
        feature_extractor
    )
    final_features_norm = F.normalize(final_features, p=2, dim=0)
    final_feature_distance = (1.0 - F.cosine_similarity(
        final_features_norm.unsqueeze(0), target_features_norm.unsqueeze(0)
    )).item()
    print(f"  Final feature distance: {final_feature_distance:.4f}")

    if args.optimize_target == 'embeddings':
        final_emb = compute_mixing_embedding(
            result['processed_mixture'],
            result['processed_stems'],
            mixing_model,
            feature_extractor,
            device
        )
        final_emb_norm = F.normalize(final_emb, p=2, dim=0)
        final_embedding_distance = (1.0 - F.cosine_similarity(
            final_emb_norm.unsqueeze(0), target_emb_norm.unsqueeze(0)
        )).item()
        print(f"  Final embedding distance: {final_embedding_distance:.4f}")
    else:
        final_embedding_distance = None
    print()

    # Save output audio
    print("Saving output audio...")
    sf.write(str(output_dir / 'output_transferred.wav'),
             result['processed_mixture'].numpy().T, 44100)
    print(f"  Saved: {output_dir / 'output_transferred.wav'}")
    print()

    # Save metrics
    print("Saving metrics...")
    metrics = {
        'input_audio': args.input_audio,
        'target_audio': args.target_audio,
        'optimize_target': args.optimize_target,
        'hyperparameters': {
            'num_steps': args.num_steps,
            'lr': args.lr,
            'receptive_field': args.receptive_field
        },
        'initial_embedding_distance': initial_embedding_distance,
        'final_embedding_distance': final_embedding_distance,
        'initial_feature_distance': initial_feature_distance,
        'final_feature_distance': final_feature_distance,
        'converged': result['converged'],
        'iterations': result['iteration_log']
    }

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {output_dir / 'metrics.json'}")
    print()

    # Print summary
    print("="*80)
    print("Summary")
    print("="*80)
    if initial_embedding_distance is not None:
        improvement_emb = initial_embedding_distance - final_embedding_distance
        improvement_emb_pct = (improvement_emb / initial_embedding_distance) * 100
        print(f"Embedding distance: {initial_embedding_distance:.4f} → {final_embedding_distance:.4f} "
              f"({improvement_emb_pct:+.1f}%)")

    improvement_feat = initial_feature_distance - final_feature_distance
    improvement_feat_pct = (improvement_feat / initial_feature_distance) * 100
    print(f"Feature distance: {initial_feature_distance:.4f} → {final_feature_distance:.4f} "
          f"({improvement_feat_pct:+.1f}%)")
    print(f"Converged: {'Yes' if result['converged'] else 'No'}")
    print()
    print(f"All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
