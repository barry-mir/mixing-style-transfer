"""
Contrastive loss functions for mixing style representation learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Standard InfoNCE loss for contrastive learning.

    Simple baseline setup:
    - Positive pairs: Multiple clips from the same song
    - Negative pairs: Clips from different songs

    The loss encourages:
    - Similar embeddings for clips from the same song
    - Different embeddings for clips from different songs
    """

    def __init__(self, temperature=0.1):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, song_labels):
        """
        Compute InfoNCE loss.

        Args:
            embeddings: Tensor of shape (N, D) where N is batch size, D is embedding dim
            song_labels: Tensor of shape (N,) indicating which samples are from the same song
                        Samples with the same label are positives, different labels are negatives

        Returns:
            loss: Scalar contrastive loss value
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Check embeddings for NaN/inf
        if torch.isnan(embeddings).any():
            print(f"[InfoNCELoss] NaN in embeddings before normalization!")
            print(f"  NaN count: {torch.isnan(embeddings).sum()}")
        if torch.isinf(embeddings).any():
            print(f"[InfoNCELoss] Inf in embeddings before normalization!")
            print(f"  Inf count: {torch.isinf(embeddings).sum()}")

        # Normalize embeddings to unit sphere
        embeddings = F.normalize(embeddings, dim=1)

        if torch.isnan(embeddings).any():
            print(f"[InfoNCELoss] NaN in embeddings after normalization!")

        # Compute similarity matrix (N, N)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        if torch.isnan(similarity_matrix).any():
            print(f"[InfoNCELoss] NaN in similarity_matrix!")
        if torch.isinf(similarity_matrix).any():
            print(f"[InfoNCELoss] Inf in similarity_matrix! Max: {similarity_matrix.max()}, Min: {similarity_matrix.min()}")

        # Create mask for positive pairs (same song label, excluding self)
        song_labels = song_labels.unsqueeze(1)  # (N, 1)
        mask_positive = (song_labels == song_labels.T).float()  # (N, N)
        mask_positive.fill_diagonal_(0)  # Exclude self-similarity

        # Create mask for negative pairs (different song labels)
        mask_negative = (song_labels != song_labels.T).float()  # (N, N)
        mask_negative.fill_diagonal_(0)

        # For numerical stability, subtract max
        max_similarity = torch.max(similarity_matrix, dim=1, keepdim=True)[0]

        # Debug: Check if similarity_matrix has extreme values
        if torch.isnan(similarity_matrix).any() or torch.isinf(similarity_matrix).any():
            print(f"[InfoNCELoss] similarity_matrix has NaN/Inf BEFORE exp!")
            print(f"  NaN count: {torch.isnan(similarity_matrix).sum()}")
            print(f"  Inf count: {torch.isinf(similarity_matrix).sum()}")
            valid_mask = ~torch.isnan(similarity_matrix) & ~torch.isinf(similarity_matrix)
            if valid_mask.any():
                print(f"  Valid range: [{similarity_matrix[valid_mask].min()}, {similarity_matrix[valid_mask].max()}]")
            else:
                print(f"  ALL VALUES ARE NaN/Inf!")
                print(f"  embeddings norm: min={torch.norm(embeddings, dim=1).min()}, max={torch.norm(embeddings, dim=1).max()}")
                print(f"  temperature: {self.temperature}")

        similarity_matrix_centered = similarity_matrix - max_similarity

        # Debug: Check after centering
        if torch.isnan(similarity_matrix_centered).any() or torch.isinf(similarity_matrix_centered).any():
            print(f"[InfoNCELoss] similarity_matrix_centered has NaN/Inf!")
            print(f"  NaN count: {torch.isnan(similarity_matrix_centered).sum()}")
            print(f"  Inf count: {torch.isinf(similarity_matrix_centered).sum()}")

        similarity_matrix_exp = torch.exp(similarity_matrix_centered)

        # Debug: Check after exp
        if torch.isnan(similarity_matrix_exp).any() or torch.isinf(similarity_matrix_exp).any():
            print(f"[InfoNCELoss] similarity_matrix_exp has NaN/Inf AFTER exp!")
            print(f"  NaN count: {torch.isnan(similarity_matrix_exp).sum()}")
            print(f"  Inf count: {torch.isinf(similarity_matrix_exp).sum()}")
            print(f"  Input to exp range: [{similarity_matrix_centered.min()}, {similarity_matrix_centered.max()}]")

        # Compute loss for each anchor
        losses = []
        for i in range(batch_size):
            # Positive similarities for anchor i
            pos_sim = similarity_matrix_exp[i] * mask_positive[i]  # (N,)
            pos_sum = pos_sim.sum()

            # Negative similarities for anchor i
            neg_sim = similarity_matrix_exp[i] * mask_negative[i]  # (N,)
            neg_sum = neg_sim.sum()

            # InfoNCE loss for anchor i
            if pos_sum > 0:  # Only compute loss if there are positives
                loss_i = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
                losses.append(loss_i)

        # Average over batch
        if len(losses) == 0:
            raise RuntimeError(
                f"No positive pairs found in batch! "
                f"Batch size: {batch_size}, "
                f"Unique songs: {len(torch.unique(song_labels))}, "
                f"This likely means each song only appears once in the batch."
            )

        loss = torch.stack(losses).mean()
        return loss


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.

    Standard contrastive loss used in SimCLR and similar frameworks.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Compute NT-Xent loss for two views (positive pairs).

        Args:
            z_i: First view embeddings (B, D)
            z_j: Second view embeddings (B, D)

        Returns:
            loss: Scalar loss value
        """
        batch_size = z_i.shape[0]
        device = z_i.device

        # Normalize
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)

        # Create labels: i-th sample's positive is (i + B)-th sample
        labels = torch.cat([
            torch.arange(batch_size) + batch_size,
            torch.arange(batch_size)
        ]).to(device)  # (2B,)

        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=device).bool()
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # Compute cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class UncertaintyWeightedMSELoss(nn.Module):
    """
    Uncertainty-weighted MSE loss for mixing features with different scales.

    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene
    Geometry and Semantics" (Kendall et al., 2017).

    The loss formula per feature group:
        L = Σ(L_i / (2σ_i²)) + log(σ_i)

    where:
    - L_i is the MSE loss for feature group i
    - σ_i is a learned uncertainty parameter (automatically balances scales)
    - log(σ_i) prevents trivial solution σ → ∞

    Feature groups (default 56 features):
    - Dynamics: 24 features (4 stems × 6) - RMS, crest factor, loudness per stem
    - Spectral: 20 features (4 stems × 5) - Band energies, tilt, flatness per stem
    - Stereo: 12 features (4 stems × 3) - ILD, correlation, mid-side ratio per stem

    With detailed spectral (e.g., 32 bins):
    - Dynamics: 24 features (4 stems × 6)
    - Spectral: 136 features (4 stems × 34) - 32 freq bins + tilt + flatness
    - Stereo: 12 features (4 stems × 3)

    Feature scales (before weighting):
    - Dynamics: -60 to +10 dB
    - Spectral: -100 to 0 dB
    - Stereo: -40 to +40 dB / -1 to +1
    """

    def __init__(
        self,
        num_feature_groups=4,
        dynamics_dim_per_stem=6,
        spectral_dim_per_stem=5,
        stereo_dim_per_stem=3,
        global_dim=8
    ):
        """
        Args:
            num_feature_groups: Number of feature groups (default: 4)
            dynamics_dim_per_stem: Dynamics features per stem (default: 6)
            spectral_dim_per_stem: Spectral features per stem (default: 5 for old, 34 for new with 32 bins)
            stereo_dim_per_stem: Stereo features per stem (default: 3)
            global_dim: Global/relational features (default: 8 for 4 rel_loudness + 4 masking)
        """
        super().__init__()

        # Learnable log-variance parameters (one per feature group)
        # Initialize to 0 (σ = 1) for all groups
        self.log_sigma = nn.Parameter(torch.zeros(num_feature_groups))

        # Calculate feature group indices dynamically
        # 4 stems in order: vocals, bass, drums, other
        dynamics_total = 4 * dynamics_dim_per_stem
        spectral_total = 4 * spectral_dim_per_stem
        stereo_total = 4 * stereo_dim_per_stem

        self.group_indices = {
            'dynamics': list(range(0, dynamics_total)),
            'spectral': list(range(dynamics_total, dynamics_total + spectral_total)),
            'stereo': list(range(dynamics_total + spectral_total, dynamics_total + spectral_total + stereo_total)),
            'global': list(range(dynamics_total + spectral_total + stereo_total,
                                dynamics_total + spectral_total + stereo_total + global_dim)),
        }

        self.num_groups = num_feature_groups
        self.total_features = dynamics_total + spectral_total + stereo_total + global_dim

        assert num_feature_groups == len(self.group_indices), \
            f"num_feature_groups ({num_feature_groups}) must match number of feature groups ({len(self.group_indices)})"

        print(f"[UncertaintyWeightedMSELoss] Initialized with:")
        print(f"  Dynamics: indices {self.group_indices['dynamics'][0]}-{self.group_indices['dynamics'][-1]} ({dynamics_total} features)")
        print(f"  Spectral: indices {self.group_indices['spectral'][0]}-{self.group_indices['spectral'][-1]} ({spectral_total} features)")
        print(f"  Stereo: indices {self.group_indices['stereo'][0]}-{self.group_indices['stereo'][-1]} ({stereo_total} features)")
        print(f"  Global: indices {self.group_indices['global'][0]}-{self.group_indices['global'][-1]} ({global_dim} features)")
        print(f"  Total features: {self.total_features}")

    def forward(self, pred_features, target_features):
        """
        Compute uncertainty-weighted MSE loss.

        Args:
            pred_features: (B, total_features) predicted mixing features
            target_features: (B, total_features) target mixing features

        Returns:
            loss: Scalar uncertainty-weighted loss
            loss_dict: Dict with per-group losses and uncertainties for logging
        """
        # Validate input shapes
        if pred_features.shape[-1] != self.total_features:
            raise ValueError(f"Expected {self.total_features} features, got {pred_features.shape[-1]}")
        if target_features.shape[-1] != self.total_features:
            raise ValueError(f"Expected {self.total_features} features, got {target_features.shape[-1]}")

        # Convert log_sigma to sigma
        sigma = torch.exp(self.log_sigma)  # (num_groups,)

        total_loss = 0.0
        loss_dict = {}

        # Compute weighted loss for each feature group
        for group_idx, (group_name, indices) in enumerate(self.group_indices.items()):
            # Extract features for this group
            group_pred = pred_features[:, indices]       # (B, num_features_in_group)
            group_target = target_features[:, indices]   # (B, num_features_in_group)

            # Compute MSE for this group
            group_mse = F.mse_loss(group_pred, group_target, reduction='mean')

            # Apply uncertainty weighting: L / (2*sigma^2) + log(sigma)
            weighted_loss = group_mse / (2 * sigma[group_idx]**2) + self.log_sigma[group_idx]

            total_loss += weighted_loss

            # Store for logging
            loss_dict[f'{group_name}_mse'] = group_mse.item()
            loss_dict[f'{group_name}_sigma'] = sigma[group_idx].item()
            loss_dict[f'{group_name}_weighted'] = weighted_loss.item()

        # Store overall uncertainty parameters
        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def get_uncertainties(self):
        """
        Get current uncertainty parameters for each feature group.

        Returns:
            Dict mapping group names to sigma values
        """
        sigma = torch.exp(self.log_sigma)
        return {
            group_name: sigma[i].item()
            for i, group_name in enumerate(self.group_indices.keys())
        }


class MultiResolutionSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss for audio quality preservation.
    Computes spectral loss at multiple FFT sizes for better perceptual quality.

    Used for cycle-consistency training to ensure reconstructed audio matches input.
    """

    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[256, 512, 128],
                 win_sizes=[1024, 2048, 512], window='hann'):
        """
        Args:
            fft_sizes: List of FFT sizes (default: [1024, 2048, 512])
            hop_sizes: List of hop sizes (default: [256, 512, 128])
            win_sizes: List of window sizes (default: [1024, 2048, 512])
            window: Window type (default: 'hann')
        """
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        self.window = window

    def stft(self, x, fft_size, hop_size, win_size):
        """
        Compute STFT.

        Args:
            x: Input audio (B, C, T) or (C, T)
            fft_size: FFT size
            hop_size: Hop size
            win_size: Window size

        Returns:
            Complex STFT tensor (B*C, freq, time)
        """
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
        """
        Spectral convergence loss (Frobenius norm ratio).

        Args:
            x_mag: Predicted magnitude spectrogram
            y_mag: Target magnitude spectrogram

        Returns:
            Spectral convergence loss (scalar)
        """
        return torch.norm(y_mag - x_mag, p='fro') / (torch.norm(y_mag, p='fro') + 1e-8)

    def log_stft_magnitude(self, x_mag, y_mag):
        """
        Log STFT magnitude loss (L1 in log space).

        Args:
            x_mag: Predicted magnitude spectrogram
            y_mag: Target magnitude spectrogram

        Returns:
            Log magnitude loss (scalar)
        """
        return nn.functional.l1_loss(torch.log(x_mag + 1e-5), torch.log(y_mag + 1e-5))

    def forward(self, x, y):
        """
        Compute multi-resolution STFT loss.

        Args:
            x: Predicted audio (B, C, T) or (C, T)
            y: Target audio (B, C, T) or (C, T)

        Returns:
            loss: Scalar loss value
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
            log_mag_loss = self.log_stft_magnitude(x_mag, y_mag)

            total_loss += sc_loss + log_mag_loss

        # Average over resolutions
        total_loss = total_loss / len(self.fft_sizes)

        return total_loss
