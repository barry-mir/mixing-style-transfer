"""
Training script for mixing style representation learning - Stage 1: Contrastive pretraining.
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import random

from params import get_params
from model import MixingStyleEncoder, SongIdentityDiscriminator
from data import SCNetSeparator, FMABaselineDataset, baseline_collate_fn
from loss import InfoNCELoss
from grl import GradientReversalLayer, compute_grl_lambda, compute_adversarial_lambda


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path, scheduler=None, discriminator=None, disc_optimizer=None, disc_scheduler=None):
    """Save model checkpoint with optional discriminator for adversarial training."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if scheduler is not None:
        checkpoint_dict['scheduler_state_dict'] = scheduler.state_dict()
    if discriminator is not None:
        checkpoint_dict['discriminator_state_dict'] = discriminator.state_dict()
    if disc_optimizer is not None:
        checkpoint_dict['disc_optimizer_state_dict'] = disc_optimizer.state_dict()
    if disc_scheduler is not None:
        checkpoint_dict['disc_scheduler_state_dict'] = disc_scheduler.state_dict()
    torch.save(checkpoint_dict, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path, scheduler=None, discriminator=None,
                    disc_optimizer=None, disc_scheduler=None, weights_only=False):
    """
    Load model checkpoint with optional discriminator for adversarial training.

    Args:
        model: Model to load weights into
        optimizer: Optimizer (state will be loaded unless weights_only=True)
        checkpoint_path: Path to checkpoint file
        scheduler: Optional learning rate scheduler
        discriminator: Optional discriminator for adversarial training
        disc_optimizer: Optional discriminator optimizer
        disc_scheduler: Optional discriminator scheduler
        weights_only: If True, only load model weights and reset training state (epoch=0)

    Returns:
        epoch: Starting epoch (0 if weights_only=True, otherwise from checkpoint)
        loss: Loss from checkpoint (or 0.0 if weights_only=True)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Try to load model state dict and report any issues
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"✓ Model state dict loaded successfully")
    except RuntimeError as e:
        print(f"✗ ERROR loading model state dict:")
        print(f"  {str(e)}")
        print(f"\nCheckpoint model keys:")
        for key in list(checkpoint['model_state_dict'].keys())[:10]:
            shape = checkpoint['model_state_dict'][key].shape if hasattr(checkpoint['model_state_dict'][key], 'shape') else 'scalar'
            print(f"  {key}: {shape}")
        print(f"  ... ({len(checkpoint['model_state_dict'])} total keys)")
        print(f"\nCurrent model keys:")
        for key in list(model.state_dict().keys())[:10]:
            shape = model.state_dict()[key].shape if hasattr(model.state_dict()[key], 'shape') else 'scalar'
            print(f"  {key}: {shape}")
        print(f"  ... ({len(model.state_dict())} total keys)")
        raise  # Re-raise the exception to stop execution

    if discriminator is not None:
        if 'discriminator_state_dict' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            print("  ✓ Discriminator state dict loaded successfully")
        else:
            print("  ⚠ WARNING: Discriminator requested but not found in checkpoint")
            print("     Starting discriminator from scratch")

    if weights_only:
        # Only load model weights, reset training state
        print(f"✓ Weights-only mode: Model loaded from {checkpoint_path}")
        print(f"  Training will start from epoch 0 with fresh optimizer/scheduler state")
        return 0, 0.0

    # Load optimizer and scheduler state for full resume
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("  Scheduler state loaded")

    if disc_optimizer is not None and 'disc_optimizer_state_dict' in checkpoint:
        disc_optimizer.load_state_dict(checkpoint['disc_optimizer_state_dict'])
        print("  ✓ Discriminator optimizer state loaded")

    if disc_scheduler is not None and 'disc_scheduler_state_dict' in checkpoint:
        disc_scheduler.load_state_dict(checkpoint['disc_scheduler_state_dict'])
        print("  ✓ Discriminator scheduler state loaded")

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch}, loss {loss:.4f})")
    return epoch, loss


def compute_adversarial_loss(
    mixing_embeddings, track_dirs, discriminator, grl_layer,
    song_id_embeddings, song_id_lookup, global_step, total_steps, args, device
):
    """
    Compute adversarial loss for removing song identity from mixing embeddings.

    Args:
        mixing_embeddings: Mixing embeddings from encoder, shape (N, embed_dim)
        track_dirs: List of track directory paths (N,)
        discriminator: Song identity discriminator network
        grl_layer: Gradient reversal layer
        song_id_embeddings: Pre-computed song identity embeddings, shape (M, 512)
        song_id_lookup: Dict mapping track_dir -> embedding index
        global_step: Current training step
        total_steps: Total number of training steps
        args: Training arguments
        device: torch device

    Returns:
        loss: Adversarial loss (scalar)
        lambda_param: Current GRL lambda value (for logging)
    """
    # Update GRL lambda: use fixed value or schedule
    if args.fixed_grl_lambda is not None:
        # Use fixed GRL lambda
        lambda_param = args.fixed_grl_lambda
    else:
        # Use DANN schedule (ramps from 0 to 1)
        lambda_param = compute_grl_lambda(
            current_step=global_step,
            total_steps=total_steps,
            warmup_steps=args.adversarial_warmup_steps
        )
    grl_layer.set_lambda(lambda_param)

    # Lookup song ID embeddings for each track_dir
    valid_indices = []
    target_song_ids = []

    for i, track_dir in enumerate(track_dirs):
        if track_dir in song_id_lookup:
            valid_indices.append(i)
            embedding_idx = song_id_lookup[track_dir]
            target_song_ids.append(song_id_embeddings[embedding_idx])

    # Skip if no valid samples (shouldn't happen normally)
    if len(valid_indices) == 0:
        print(f"WARNING: No valid samples found for adversarial training")
        return torch.tensor(0.0, device=device, requires_grad=True), lambda_param

    # Filter to valid samples
    valid_indices = torch.tensor(valid_indices, dtype=torch.long, device=device)
    mixing_emb_valid = mixing_embeddings[valid_indices]  # (K, embed_dim)
    target_song_id = torch.stack(target_song_ids, dim=0).to(device)  # (K, 512)

    # Add noise to embeddings to weaken discriminator (if enabled)
    if args.discriminator_noise > 0.0:
        noise = torch.randn_like(mixing_emb_valid) * args.discriminator_noise
        mixing_emb_valid = mixing_emb_valid + noise

    # Apply GRL + discriminator
    grl_output = grl_layer(mixing_emb_valid)  # (K, embed_dim)
    predicted_song_id = discriminator(grl_output)  # (K, 512)

    # Compute cosine similarity loss
    # Loss = 1 - cosine_similarity (we want to maximize similarity, which minimizes loss)
    # The discriminator learns to predict song ID (minimize loss)
    # The encoder learns to hide song ID (maximize loss via reversed gradients)
    pred_norm = nn.functional.normalize(predicted_song_id, dim=1)
    target_norm = nn.functional.normalize(target_song_id, dim=1)
    cosine_sim = (pred_norm * target_norm).sum(dim=1)  # (K,)
    loss = (1.0 - cosine_sim).mean()

    return loss, lambda_param


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, args, writer, scaler=None, scheduler=None,
                discriminator=None, grl_layer=None, song_id_embeddings=None, song_id_lookup=None, steps_per_epoch=None, total_steps=None,
                disc_optimizer=None, disc_scheduler=None):
    """Train for one epoch with optional adversarial training and separate discriminator optimizer."""
    model.train()
    if discriminator is not None:
        discriminator.train()

    total_loss = 0.0
    total_contrastive_loss = 0.0
    total_adversarial_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    use_amp = scaler is not None

    for batch_idx, batch_data in enumerate(pbar):
        # Unpack batch data (with or without track_dirs)
        if len(batch_data) == 4:
            stems_dict, mixing_features, song_labels, track_dirs = batch_data
        else:
            stems_dict, mixing_features, song_labels = batch_data
            track_dirs = None
        # Move stems to device
        stems_dict = {k: v.to(device) for k, v in stems_dict.items()}  # (N, 2, T)
        # Check all tensors for NaN values and print their shapes
        for k, v in stems_dict.items():
            if torch.isnan(v).any():
                print(f'WARNING: {k} contains NaN values!')

        if torch.isnan(mixing_features).any():
            print('WARNING: mixing_features contains NaN values!')

        if torch.isnan(song_labels.float()).any():
            print('WARNING: song_labels contains NaN values!')

        mixing_features = mixing_features.to(device)  # (N, feature_dim)
        song_labels = song_labels.to(device)  # (N,)

        optimizer.zero_grad()
        if disc_optimizer is not None:
            disc_optimizer.zero_grad()

        # Forward pass with mixed precision
        if use_amp:
            with torch.cuda.amp.autocast():
                embeddings = model(stems_dict, mixing_features)  # (N, embed_dim)

                # Compute contrastive loss (InfoNCE)
                loss_contrastive = criterion(embeddings, song_labels)

                # Compute adversarial loss if enabled
                loss_adversarial = torch.tensor(0.0, device=device)
                grl_lambda = 0.0
                adv_lambda = 0.0
                if args.use_adversarial and track_dirs is not None:
                    global_step = epoch * steps_per_epoch + batch_idx
                    loss_adversarial, grl_lambda = compute_adversarial_loss(
                        embeddings, track_dirs, discriminator, grl_layer,
                        song_id_embeddings, song_id_lookup, global_step, total_steps, args, device
                    )
                    # Compute current adversarial lambda weight
                    adv_lambda = compute_adversarial_lambda(
                        global_step, total_steps, args.adversarial_warmup_steps,
                        args.initial_adversarial_lambda, args.adversarial_lambda
                    )

                # Combined loss
                loss = loss_contrastive + adv_lambda * loss_adversarial if args.use_adversarial else loss_contrastive

            # Debug: check loss properties
            if batch_idx == 0 and epoch == 0:
                print(f'\nDebug info (first batch):')
                print(f'  Contrastive loss: {loss_contrastive.item():.4f}')
                if args.use_adversarial:
                    print(f'  Adversarial loss: {loss_adversarial.item():.4f}')
                    print(f'  GRL lambda: {grl_lambda:.4f}')
                    print(f'  Adversarial weight (λ_adv): {adv_lambda:.4f}')
                    if args.discriminator_noise > 0.0:
                        print(f'  Discriminator noise scale: {args.discriminator_noise}')
                print(f'  Total loss: {loss.item():.4f}')
                print(f'  Embeddings shape: {embeddings.shape}')
                print(f'  Unique songs: {len(torch.unique(song_labels))}')

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            if disc_optimizer is not None:
                scaler.step(disc_optimizer)
            scaler.update()
        else:
            # Standard FP32 training
            embeddings = model(stems_dict, mixing_features)

            # Compute contrastive loss (InfoNCE)
            loss_contrastive = criterion(embeddings, song_labels)

            # Compute adversarial loss if enabled
            loss_adversarial = torch.tensor(0.0, device=device)
            grl_lambda = 0.0
            adv_lambda = 0.0
            if args.use_adversarial and track_dirs is not None:
                global_step = epoch * steps_per_epoch + batch_idx
                loss_adversarial, grl_lambda = compute_adversarial_loss(
                    embeddings, track_dirs, discriminator, grl_layer,
                    song_id_embeddings, song_id_lookup, global_step, total_steps, args, device
                )
                # Compute current adversarial lambda weight
                adv_lambda = compute_adversarial_lambda(
                    global_step, total_steps, args.adversarial_warmup_steps,
                    args.initial_adversarial_lambda, args.adversarial_lambda
                )

            # Combined loss
            loss = loss_contrastive + adv_lambda * loss_adversarial if args.use_adversarial else loss_contrastive

            loss.backward()
            optimizer.step()
            if disc_optimizer is not None:
                disc_optimizer.step()

        # Update learning rate scheduler (per-step)
        if scheduler is not None:
            scheduler.step()
        if disc_scheduler is not None:
            disc_scheduler.step()

        # Update metrics - detach loss to avoid retaining graph
        loss_value = loss.detach().item()
        loss_contrastive_value = loss_contrastive.detach().item()
        loss_adversarial_value = loss_adversarial.detach().item() if isinstance(loss_adversarial, torch.Tensor) else 0.0

        total_loss += loss_value
        total_contrastive_loss += loss_contrastive_value
        total_adversarial_loss += loss_adversarial_value
        num_batches += 1

        # Get current learning rate for display
        current_lr = optimizer.param_groups[0]['lr']

        # Update progress bar
        if args.use_adversarial:
            pbar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'cont': f'{loss_contrastive_value:.4f}',
                'adv': f'{loss_adversarial_value:.4f}',
                'λ_grl': f'{grl_lambda:.3f}',
                'λ_adv': f'{adv_lambda:.3f}',
                'lr': f'{current_lr:.6f}'
            })
        else:
            pbar.set_postfix({'loss': f'{loss_value:.4f}', 'lr': f'{current_lr:.6f}'})

        # Log to tensorboard
        if batch_idx % args.log_interval == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/train_batch', loss_value, global_step)
            writer.add_scalar('Loss/contrastive', loss_contrastive_value, global_step)
            writer.add_scalar('Learning_Rate', current_lr, global_step)

            if args.use_adversarial:
                writer.add_scalar('Loss/adversarial', loss_adversarial_value, global_step)
                writer.add_scalar('Loss/total', loss_value, global_step)
                writer.add_scalar('Adversarial/grl_lambda', grl_lambda, global_step)
                writer.add_scalar('Adversarial/adv_lambda', adv_lambda, global_step)

        # Explicitly delete large tensors to free memory
        del stems_dict, mixing_features, embeddings, loss, loss_contrastive
        if args.use_adversarial:
            del loss_adversarial

    avg_loss = total_loss / num_batches
    avg_contrastive_loss = total_contrastive_loss / num_batches
    avg_adversarial_loss = total_adversarial_loss / num_batches

    if args.use_adversarial:
        return avg_loss, avg_contrastive_loss, avg_adversarial_loss
    else:
        return avg_loss


def validate_epoch(model, dataloader, criterion, device, epoch, args):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")

    with torch.no_grad():  # No gradients needed for validation
        for batch_idx, batch_data in enumerate(pbar):
            # Unpack batch data (with or without track_dirs)
            if len(batch_data) == 4:
                stems_dict, mixing_features, song_labels, track_dirs = batch_data
            else:
                stems_dict, mixing_features, song_labels = batch_data

            # Move data to device
            stems_dict = {k: v.to(device) for k, v in stems_dict.items()}
            mixing_features = mixing_features.to(device)
            song_labels = song_labels.to(device)

            # Forward pass (no AMP for validation to save memory)
            embeddings = model(stems_dict, mixing_features)

            # Compute InfoNCE loss
            loss = criterion(embeddings, song_labels)

            # Update metrics
            loss_value = loss.item()
            total_loss += loss_value
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'val_loss': f'{loss_value:.4f}'})

            # Cleanup
            del stems_dict, mixing_features, embeddings, loss

    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    """Main training function."""
    # Parse arguments
    args = get_params()

    # Set random seed
    set_seed(args.seed)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Initialize tensorboard writer
    writer = SummaryWriter(log_dir=args.log_dir)

    # Create dataset
    print("Creating baseline dataset...")

    # Use pre-separated stems (required for baseline)
    separated_path = getattr(args, 'separated_path', '/nas/FMA/fma_separated/')
    print(f"Using pre-separated stems from: {separated_path}")

    # Get baseline parameters
    num_segments = getattr(args, 'num_segments', 2)  # Number of clips per song for positives

    # Create full dataset
    full_dataset = FMABaselineDataset(
        separated_path=separated_path,
        clip_duration=args.clip_duration,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        num_segments=num_segments,
        min_audio_duration=25.0
    )
    print(f"Full dataset created with {len(full_dataset)} tracks")
    print(f"Baseline structure: {num_segments} temporal segments per song")
    print(f"Effective batch size per batch: batch_size × {num_segments} segments")

    # Split dataset into train (90%) and validation (10%)
    dataset_size = len(full_dataset)
    val_size = int(0.1 * dataset_size)
    train_size = dataset_size - val_size

    # Create indices for split
    indices = list(range(dataset_size))
    np.random.seed(args.seed)  # For reproducible split
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create subset datasets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    print(f"\nDataset split:")
    print(f"  Training samples: {len(train_dataset)} ({train_size/dataset_size*100:.1f}%)")
    print(f"  Validation samples: {len(val_dataset)} ({val_size/dataset_size*100:.1f}%)")

    # Create training dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=baseline_collate_fn,
        pin_memory=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=False,
        multiprocessing_context='fork' if args.num_workers > 0 else None
    )

    # Create validation dataloader
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No need to shuffle validation
        num_workers=args.num_workers,
        collate_fn=baseline_collate_fn,
        pin_memory=False,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=False,
        multiprocessing_context='fork' if args.num_workers > 0 else None
    )

    # Get feature dimension from dataset
    stems_list, features_list, song_idx, track_dir = full_dataset[0]
    feature_dim = features_list[0].shape[0]  # First item's feature dimension
    print(f"Mixing feature dimension: {feature_dim}")

    # Load song identity embeddings for adversarial training
    song_id_embeddings = None
    song_id_lookup = None
    if args.use_adversarial:
        print("\n" + "=" * 80)
        print("ADVERSARIAL TRAINING MODE")
        print("=" * 80)
        print(f"Loading song identity embeddings from: {args.song_id_cache_path}")
        cache = torch.load(args.song_id_cache_path, map_location='cpu')
        song_id_embeddings = cache['embeddings'].to(device)  # (N, 512)
        song_id_lookup = {path: idx for idx, path in enumerate(cache['track_paths'])}
        print(f"  Loaded {song_id_embeddings.shape[0]} song identity embeddings")
        print(f"  Embedding dimension: {song_id_embeddings.shape[1]}")
        print(f"  Adversarial lambda: {args.adversarial_lambda}")
        print(f"  Adversarial warmup steps: {args.adversarial_warmup_steps}")
        print("=" * 80)

    # Initialize model (now accepts raw audio)
    print("Initializing model...")
    model = MixingStyleEncoder(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        split_size=args.band_split_size,
        overlap=args.band_overlap,
        channels=8,  # 4 stems × 2 stereo
        embed_dim=args.encoder_dim,
        feature_dim=feature_dim
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Initialize discriminator and GRL for adversarial training
    discriminator = None
    grl_layer = None
    if args.use_adversarial:
        print("\nInitializing adversarial training components...")
        discriminator = SongIdentityDiscriminator(
            input_dim=args.encoder_dim,
            hidden_dim=args.discriminator_hidden_dim,
            output_dim=song_id_embeddings.shape[1],  # Match song ID embedding dimension
            dropout=args.discriminator_dropout
        ).to(device)
        grl_layer = GradientReversalLayer(init_lambda=0.0).to(device)

        disc_params = sum(p.numel() for p in discriminator.parameters())
        print(f"  Discriminator parameters: {disc_params:,}")
        print(f"  GRL initial lambda: 0.0 (will warm up to 1.0)")
        if args.discriminator_noise > 0.0:
            print(f"  Discriminator noise: {args.discriminator_noise} (adds Gaussian noise to embeddings)")

    # Initialize optimizer(s)
    disc_optimizer = None
    disc_scheduler = None

    if args.use_adversarial and args.discriminator_lr is not None:
        # Use separate optimizers for encoder and discriminator
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(),
            lr=args.discriminator_lr,
            weight_decay=args.weight_decay
        )
        print(f"\nUsing separate optimizers:")
        print(f"  Encoder LR: {args.learning_rate}")
        print(f"  Discriminator LR: {args.discriminator_lr}")
    elif args.use_adversarial:
        # Use single optimizer with same LR for both
        parameters = list(model.parameters()) + list(discriminator.parameters())
        optimizer = torch.optim.AdamW(
            parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        print(f"\nOptimizer includes encoder + discriminator parameters (same LR: {args.learning_rate})")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

    # Initialize learning rate scheduler: linear warmup + cosine decay
    # Calculate total steps
    steps_per_epoch = len(train_dataloader)
    total_steps = args.num_epochs * steps_per_epoch
    warmup_steps = 2000

    print(f"\nLearning rate schedule:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Initial LR: {args.learning_rate}")

    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(current_step):
        # Linear warmup for first 2000 steps
        if current_step < warmup_steps:
            return current_step / warmup_steps
        # Cosine decay after warmup
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Create scheduler for discriminator if it has separate optimizer
    if disc_optimizer is not None:
        disc_scheduler = LambdaLR(disc_optimizer, lr_lambda=lr_lambda)
        print(f"  Discriminator uses same schedule with initial LR: {args.discriminator_lr}")

    # Initialize InfoNCE loss function
    criterion = InfoNCELoss(temperature=args.temperature)
    print(f"Using InfoNCELoss with temperature={args.temperature}")
    print("  Positives: Same song, different temporal segments")
    print("  Negatives: Different songs")

    # Initialize mixed precision training (AMP) for VRAM reduction
    use_amp = getattr(args, 'use_amp', False) and device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        print("Using mixed precision training (FP16) for VRAM reduction")
    else:
        print("Using standard FP32 training")

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume is not None:
        start_epoch, _ = load_checkpoint(
            model, optimizer, args.resume, scheduler, discriminator,
            disc_optimizer, disc_scheduler,
            weights_only=args.weights_only
        )
        if not args.weights_only:
            start_epoch += 1  # Continue from next epoch (only if resuming, not if weights_only)

    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print("=" * 80)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.num_epochs):
        # Train one epoch
        train_result = train_epoch(
            model, train_dataloader, criterion, optimizer, device,
            epoch, args, writer, scaler, scheduler,
            discriminator, grl_layer, song_id_embeddings, song_id_lookup,
            steps_per_epoch, total_steps,
            disc_optimizer, disc_scheduler
        )

        # Unpack results
        if args.use_adversarial:
            train_loss, train_contrastive_loss, train_adversarial_loss = train_result
        else:
            train_loss = train_result

        # Validate one epoch
        val_loss = validate_epoch(
            model, val_dataloader, criterion, device,
            epoch, args
        )

        # Log epoch metrics
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss = {train_loss:.4f}")
        if args.use_adversarial:
            print(f"    Contrastive Loss = {train_contrastive_loss:.4f}")
            print(f"    Adversarial Loss = {train_adversarial_loss:.4f}")
        print(f"  Val Loss   = {val_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'checkpoint_epoch_{epoch+1}.pt'
            )
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, scheduler, discriminator, disc_optimizer, disc_scheduler)

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, best_checkpoint_path, scheduler, discriminator, disc_optimizer, disc_scheduler)
            print(f"  ✓ New best model saved! Val Loss: {best_val_loss:.4f}")

        print("=" * 80)

    # Save final model
    final_checkpoint_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, args.num_epochs - 1, val_loss, final_checkpoint_path, scheduler, discriminator, disc_optimizer, disc_scheduler)

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {final_checkpoint_path}")

    writer.close()


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method("spawn", force=True)
    main()
