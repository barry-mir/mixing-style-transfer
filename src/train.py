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
from model import MixingStyleEncoder
from data import SCNetSeparator, FMABaselineDataset, baseline_collate_fn
from loss import InfoNCELoss


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


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path, scheduler=None):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    checkpoint_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if scheduler is not None:
        checkpoint_dict['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint_dict, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path, scheduler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("  Scheduler state loaded")
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch})")
    return epoch, loss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, args, writer, scaler=None, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    use_amp = scaler is not None

    for batch_idx, (stems_dict, mixing_features, song_labels) in enumerate(pbar):
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

        # Forward pass with mixed precision
        if use_amp:
            with torch.cuda.amp.autocast():
                embeddings = model(stems_dict, mixing_features)  # (S×V×T, embed_dim)

                # Compute InfoNCE loss
                loss = criterion(embeddings, song_labels)

            # Debug: check loss properties
            if batch_idx == 0 and epoch == 0:
                print(f'\nDebug info (first batch):')
                print(f'  Loss value: {loss.item():.4f}')
                print(f'  Loss requires_grad: {loss.requires_grad}')
                print(f'  Loss grad_fn: {loss.grad_fn}')
                print(f'  Embeddings shape: {embeddings.shape}')
                print(f'  Unique songs: {len(torch.unique(song_labels))}')

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard FP32 training
            embeddings = model(stems_dict, mixing_features)

            # Compute InfoNCE loss
            loss = criterion(embeddings, song_labels)

            loss.backward()
            optimizer.step()

        # Update learning rate scheduler (per-step)
        if scheduler is not None:
            scheduler.step()

        # Update metrics - detach loss to avoid retaining graph
        loss_value = loss.detach().item()
        total_loss += loss_value
        num_batches += 1

        # Get current learning rate for display
        current_lr = optimizer.param_groups[0]['lr']

        # Update progress bar
        pbar.set_postfix({'loss': f'{loss_value:.4f}', 'lr': f'{current_lr:.6f}'})

        # Log to tensorboard
        if batch_idx % args.log_interval == 0:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar('Loss/train_batch', loss_value, global_step)
            writer.add_scalar('Learning_Rate', current_lr, global_step)

        # Explicitly delete large tensors to free memory
        del stems_dict, mixing_features, embeddings, loss

    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(model, dataloader, criterion, device, epoch, args):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")

    with torch.no_grad():  # No gradients needed for validation
        for batch_idx, (stems_dict, mixing_features, song_labels) in enumerate(pbar):
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
    stems_list, features_list, song_idx = full_dataset[0]
    feature_dim = features_list[0].shape[0]  # First item's feature dimension
    print(f"Mixing feature dimension: {feature_dim}")

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

    # Initialize optimizer
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
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, scheduler)
        start_epoch += 1

    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print("=" * 80)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.num_epochs):
        # Train one epoch
        train_loss = train_epoch(
            model, train_dataloader, criterion, optimizer, device,
            epoch, args, writer, scaler, scheduler
        )

        # Validate one epoch
        val_loss = validate_epoch(
            model, val_dataloader, criterion, device,
            epoch, args
        )

        # Log epoch metrics
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss = {train_loss:.4f}")
        print(f"  Val Loss   = {val_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)

        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f'checkpoint_epoch_{epoch+1}.pt'
            )
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, scheduler)

        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, val_loss, best_checkpoint_path, scheduler)
            print(f"  ✓ New best model saved! Val Loss: {best_val_loss:.4f}")

        print("=" * 80)

    # Save final model
    final_checkpoint_path = os.path.join(args.checkpoint_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, args.num_epochs - 1, val_loss, final_checkpoint_path, scheduler)

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {final_checkpoint_path}")

    writer.close()


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method("spawn", force=True)
    main()
