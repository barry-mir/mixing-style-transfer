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
            print(f"pos_sum: {pos_sum}, neg_sum: {neg_sum}, at batch index {i}")

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


class ContrastiveLossSimple(nn.Module):
    """
    Simplified contrastive loss for pairs.

    This version assumes the batch is organized as:
    [anchor_1, positive_1, negative_1, anchor_2, positive_2, negative_2, ...]

    Where:
    - anchor_i and positive_i are from the same song (different clips)
    - anchor_i and negative_i are the same clip (augmented)
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings):
        """
        Compute contrastive loss assuming batch structure [anchor, pos, neg, ...].

        Args:
            embeddings: Tensor of shape (3*N, D) where N is number of triplets

        Returns:
            loss: Scalar contrastive loss
        """
        batch_size = embeddings.shape[0]
        assert batch_size % 3 == 0, "Batch size must be multiple of 3 (anchor, pos, neg)"

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Reshape into triplets
        embeddings = embeddings.view(-1, 3, embeddings.shape[1])  # (N, 3, D)
        anchors = embeddings[:, 0, :]  # (N, D)
        positives = embeddings[:, 1, :]  # (N, D)
        negatives = embeddings[:, 2, :]  # (N, D)

        # Compute similarities
        pos_sim = torch.sum(anchors * positives, dim=1) / self.temperature  # (N,)
        neg_sim = torch.sum(anchors * negatives, dim=1) / self.temperature  # (N,)

        # InfoNCE loss: -log(exp(pos_sim) / (exp(pos_sim) + exp(neg_sim)))
        loss = -torch.log(
            torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim) + 1e-8)
        )

        return loss.mean()


class TwoAxisInfoNCELoss(nn.Module):
    """
    Two-axis InfoNCE loss for mixing style representation learning.

    Correct structure:
    - Axis 1 (Content invariance): Same song, same variant, different segments → POSITIVE
      (Mixing style is shared across the whole song)
    - Axis 2 (Mixing sensitivity): Same song, different variants → NEGATIVE
      (Different mixes should have different representations)
    - Hard negatives: Different songs → NEGATIVE

    This encourages:
    - Content-invariant representations within same mix (temporal segments are similar)
    - Mixing-sensitive representations (different mixes have different embeddings)
    """

    def __init__(self, temperature=0.1):
        """
        Args:
            temperature: Temperature parameter for scaling similarities
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, song_labels, variant_labels, segment_labels):
        """
        Compute two-axis InfoNCE loss.

        Args:
            embeddings: (N, D) where N = S×V×T (songs × variants × segments)
            song_labels: (N,) - song index for each embedding
            variant_labels: (N,) - variant index within song
            segment_labels: (N,) - segment index within variant

        Returns:
            loss: Scalar contrastive loss
        """
        batch_size = embeddings.shape[0]
        device = embeddings.device

        # Check embeddings for NaN/inf
        if torch.isnan(embeddings).any():
            print(f"[TwoAxisInfoNCELoss] NaN in embeddings before normalization!")
        if torch.isinf(embeddings).any():
            print(f"[TwoAxisInfoNCELoss] Inf in embeddings before normalization!")

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        if torch.isnan(embeddings).any():
            print(f"[TwoAxisInfoNCELoss] NaN in embeddings after normalization!")

        # Compute similarity matrix (N, N)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        if torch.isnan(similarity_matrix).any():
            print(f"[TwoAxisInfoNCELoss] NaN in similarity_matrix!")
        if torch.isinf(similarity_matrix).any():
            print(f"[TwoAxisInfoNCELoss] Inf in similarity_matrix! Max: {similarity_matrix.max()}, Min: {similarity_matrix.min()}")

        # Create masks for different relationships
        song_labels = song_labels.unsqueeze(1)  # (N, 1)
        variant_labels = variant_labels.unsqueeze(1)  # (N, 1)

        # Axis 1: Same song + same variant (different segments only) → POSITIVE
        # This captures content invariance - all segments of same mix should be similar
        mask_same_song = (song_labels == song_labels.T).float()  # (N, N)
        mask_same_variant = (variant_labels == variant_labels.T).float()  # (N, N)
        mask_positive = mask_same_song * mask_same_variant  # Same song AND same variant
        mask_positive.fill_diagonal_(0)  # Exclude self-similarity

        # Axis 2: Same song + different variant → NEGATIVE (for mixing discrimination)
        # These are the key negatives that enforce mixing sensitivity
        mask_diff_variant = (song_labels == song_labels.T) & (variant_labels != variant_labels.T)
        mask_diff_variant = mask_diff_variant.float()  # (N, N)

        # Hard negatives: Different songs
        mask_diff_song = (song_labels != song_labels.T).float()  # (N, N)

        # All negatives: different variants of same song + different songs
        mask_negative = mask_diff_variant + mask_diff_song  # (N, N)

        # Exp of similarities for numerical stability
        max_similarity = torch.max(similarity_matrix, dim=1, keepdim=True)[0]
        similarity_matrix_centered = similarity_matrix - max_similarity

        if torch.isnan(similarity_matrix_centered).any() or torch.isinf(similarity_matrix_centered).any():
            print(f"[TwoAxisInfoNCELoss] similarity_matrix_centered has NaN/Inf!")

        similarity_matrix_exp = torch.exp(similarity_matrix_centered)

        if torch.isnan(similarity_matrix_exp).any() or torch.isinf(similarity_matrix_exp).any():
            print(f"[TwoAxisInfoNCELoss] similarity_matrix_exp has NaN/Inf AFTER exp!")
            print(f"  NaN count: {torch.isnan(similarity_matrix_exp).sum()}")
            print(f"  Inf count: {torch.isinf(similarity_matrix_exp).sum()}")
            print(f"  Input to exp range: [{similarity_matrix_centered.min()}, {similarity_matrix_centered.max()}]")

        # Compute InfoNCE loss
        losses = []
        for i in range(batch_size):
            # Positive similarities: same song, same variant, different segments
            pos_sim = similarity_matrix_exp[i] * mask_positive[i]
            pos_sum = pos_sim.sum()

            # Negative similarities: different variants + different songs
            neg_sim = similarity_matrix_exp[i] * mask_negative[i]
            neg_sum = neg_sim.sum()

            # InfoNCE loss
            if pos_sum > 0:  # Only if positives exist
                loss_i = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
                losses.append(loss_i)

        # Average over batch
        if len(losses) == 0:
            raise RuntimeError(
                f"No positive pairs found in batch! "
                f"Batch size: {batch_size}, "
                f"Unique (song, variant) pairs: {len(torch.unique(torch.stack([song_labels.squeeze(), variant_labels.squeeze()], dim=1), dim=0))}, "
                f"num_segments should be ≥2 to create positives within same variant."
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
