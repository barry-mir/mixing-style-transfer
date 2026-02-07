"""
Training script for zero-shot mixing style transfer.

Trains an end-to-end system where:
1. Encoder processes both input and target stems → embeddings
2. Concatenate input + target embeddings (512 + 512 = 1024)
3. FiLM Generator converts embeddings → FiLM parameters for TCN
4. TCN processes input stems with FiLM conditioning → output stems
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from model import MixingStyleEncoder
from tcn_mixer import TCNMixer, TCNFiLMGenerator
from loss import MultiResolutionSTFTLoss
from data import StyleTransferDataset, style_transfer_collate_fn
from mixing_utils import MixingFeatureExtractor


class StyleTransferTrainer:
    """
    Trainer for zero-shot mixing style transfer.
    """

    def __init__(
        self,
        encoder,
        tcn,
        film_generator,
        feature_extractor,
        optimizer,
        scheduler,
        device,
        output_dir,
        log_interval=10,
        lambda_cycle=0.1,  # Weight for cycle consistency loss (keep low to avoid identity collapse)
        gradient_accumulation_steps=32,  # Accumulate gradients over N steps
        use_cycle_consistency=True,  # Enable/disable cycle consistency loss
    ):
        self.encoder = encoder
        self.tcn = tcn
        self.film_generator = film_generator
        self.feature_extractor = feature_extractor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.log_interval = log_interval
        self.lambda_cycle = lambda_cycle if use_cycle_consistency else 0.0
        self.use_cycle_consistency = use_cycle_consistency
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Multi-resolution STFT loss for cycle consistency
        self.mrstft_loss = MultiResolutionSTFTLoss().to(device)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard')
        print(f"TensorBoard logs: {self.output_dir / 'tensorboard'}")

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def _stems_dict_to_batch(self, stems_dict):
        """
        Convert stems dict to batch tensor for encoder.

        Args:
            stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                       Each value is (B, 2, T)

        Returns:
            batch_stems: (B, 8, T) tensor
        """
        # Stack in order: vocals, bass, drums, other
        stem_order = ['vocals', 'bass', 'drums', 'other']
        stems_list = [stems_dict[name] for name in stem_order]

        # Concatenate: (B, 8, T)
        batch_stems = torch.cat(stems_list, dim=1)

        return batch_stems

    def train_step(self, input_stems_dict, target_stems_dict, target_features):
        """
        Single training step.

        Args:
            input_stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                             Each value is (B, 2, T)
            target_stems_dict: Same structure as input_stems_dict
            target_features: (B, 56) target mixing features

        Returns:
            loss: Scalar loss value
            loss_dict: Dict with detailed losses for logging
        """
        # Convert stems dicts to batch tensors
        input_batch = self._stems_dict_to_batch(input_stems_dict)  # (B, 8, T)
        target_batch = self._stems_dict_to_batch(target_stems_dict)  # (B, 8, T)

        # Move to device
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)
        target_features = target_features.to(self.device)

        # 1. Extract features for encoder conditioning (frozen, no gradients)
        with torch.no_grad():
            # Compute input mixing features
            input_mixture = input_batch.sum(dim=1, keepdim=True)  # (B, 2, T)
            input_stems_for_features = {
                name: input_stems_dict[name].to(self.device)
                for name in ['vocals', 'bass', 'drums', 'other']
            }
            input_features = torch.stack([
                self.feature_extractor.extract_all_features(
                    {name: input_stems_for_features[name][i] for name in input_stems_for_features},
                    input_mixture[i]
                )
                for i in range(input_batch.shape[0])
            ], dim=0)  # (B, 56)

            # Compute target mixing features for conditioning
            target_mixture = target_batch.sum(dim=1, keepdim=True)  # (B, 2, T)
            target_stems_for_features = {
                name: target_stems_dict[name].to(self.device)
                for name in ['vocals', 'bass', 'drums', 'other']
            }
            target_features_for_encoder = torch.stack([
                self.feature_extractor.extract_all_features(
                    {name: target_stems_for_features[name][i] for name in target_stems_for_features},
                    target_mixture[i]
                )
                for i in range(target_batch.shape[0])
            ], dim=0)  # (B, 56)

            # 2. Encode input and target stems (frozen encoder, no gradients)
            input_embedding = self.encoder(input_stems_for_features, input_features)  # (B, 512)
            target_embedding = self.encoder(target_stems_for_features, target_features_for_encoder)  # (B, 512)

        # 3. Concatenate embeddings
        concat_embedding = torch.cat([input_embedding, target_embedding], dim=1)  # (B, 1024)

        # 4. Generate FiLM parameters
        film_params = self.film_generator(concat_embedding)  # List of dicts

        # 5. Process input stems through TCN with FiLM
        output_batch = self.tcn(input_batch, film_params=film_params)  # (B, 8, T)

        # 6. Compute output embedding
        # Split output back into stems dict
        output_stems_dict = {}
        for i, stem_name in enumerate(['vocals', 'bass', 'drums', 'other']):
            output_stems_dict[stem_name] = output_batch[:, i*2:(i+1)*2, :]  # (B, 2, T)

        # Compute output mixture and features (no grad for feature extractor)
        output_mixture = output_batch.sum(dim=1)  # (B, 2, T) - no keepdim needed
        output_features = torch.stack([
            self.feature_extractor.extract_all_features(
                {name: output_stems_dict[name][i] for name in output_stems_dict},
                output_mixture[i]
            )
            for i in range(output_batch.shape[0])
        ], dim=0)  # (B, 56)

        # Encode output WITH gradients (encoder params frozen, but gradients flow through)
        # This allows gradients to flow back to output_batch and optimize TCN
        output_embedding = self.encoder(output_stems_dict, output_features)  # (B, 512)

        # 7. Compute style matching loss: 1 - cosine_similarity(output_emb, target_emb)
        # Normalize embeddings
        output_embedding_norm = nn.functional.normalize(output_embedding, p=2, dim=1)
        target_embedding_norm = nn.functional.normalize(target_embedding, p=2, dim=1)

        # Cosine similarity
        cos_sim = (output_embedding_norm * target_embedding_norm).sum(dim=1)  # (B,)
        style_loss = (1.0 - cos_sim).mean()  # Minimize distance

        loss_dict = {'style_loss': style_loss.item(), 'cos_sim': cos_sim.mean().item()}

        # 8. Cycle consistency: output -> reconstructed input (optional)
        if self.use_cycle_consistency:
            # Generate FiLM parameters for backward pass (output style -> input style)
            # Concatenate: output embedding + input embedding (reverse order)
            concat_embedding_backward = torch.cat([target_embedding, input_embedding], dim=1)  # (B, 1024)
            film_params_backward = self.film_generator(concat_embedding_backward)

            # Apply backward TCN: output -> reconstructed input
            reconstructed_batch = self.tcn(output_batch, film_params=film_params_backward)  # (B, 8, T)

            # Compute cycle consistency loss (MRSTFT)
            cycle_loss = self.mrstft_loss(reconstructed_batch, input_batch)

            # 9. Total loss: style matching + cycle consistency
            total_loss = style_loss + self.lambda_cycle * cycle_loss

            # Update loss dict
            loss_dict['cycle_loss'] = cycle_loss.item()
        else:
            # No cycle consistency - only style matching loss
            total_loss = style_loss
            loss_dict['cycle_loss'] = 0.0

        loss_dict['total_loss'] = total_loss.item()

        return total_loss, loss_dict

    def train_epoch(self, train_loader):
        """
        Train for one epoch with gradient accumulation.
        """
        self.encoder.eval()  # Keep encoder in eval mode (frozen)
        self.tcn.train()
        self.film_generator.train()

        epoch_losses = []
        epoch_loss_dicts = []

        # Zero gradients at the start
        self.optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")

        for batch_idx, (input_stems, target_stems, target_features) in enumerate(pbar):
            # Forward pass
            loss, loss_dict = self.train_step(input_stems, target_stems, target_features)

            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Perform optimizer step every gradient_accumulation_steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping (encoder is frozen, no gradients)
                torch.nn.utils.clip_grad_norm_(self.tcn.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.film_generator.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

            # Logging (log the unscaled loss)
            epoch_losses.append(loss.item() * self.gradient_accumulation_steps)
            epoch_loss_dicts.append(loss_dict)

            if batch_idx % self.log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    'style': f"{loss_dict.get('style_loss', 0):.4f}",
                    'cycle': f"{loss_dict.get('cycle_loss', 0):.4f}",
                    'cos_sim': f"{loss_dict.get('cos_sim', 0):.3f}",
                })

                # TensorBoard logging
                self.writer.add_scalar('Train/loss', loss.item() * self.gradient_accumulation_steps, self.global_step)
                self.writer.add_scalar('Train/style_loss', loss_dict.get('style_loss', 0), self.global_step)
                self.writer.add_scalar('Train/cycle_loss', loss_dict.get('cycle_loss', 0), self.global_step)
                self.writer.add_scalar('Train/cosine_similarity', loss_dict.get('cos_sim', 0), self.global_step)
                self.writer.add_scalar('Train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)

            self.global_step += 1

        # Final optimizer step for any remaining gradients
        if (batch_idx + 1) % self.gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.tcn.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.film_generator.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

        # Compute epoch averages
        avg_loss = np.mean(epoch_losses)
        avg_loss_dict = {
            key: np.mean([d[key] for d in epoch_loss_dicts])
            for key in epoch_loss_dicts[0].keys()
        }

        # Log epoch averages to TensorBoard
        self.writer.add_scalar('Epoch/train_loss', avg_loss, self.epoch)
        self.writer.add_scalar('Epoch/train_style_loss', avg_loss_dict.get('style_loss', 0), self.epoch)
        self.writer.add_scalar('Epoch/train_cycle_loss', avg_loss_dict.get('cycle_loss', 0), self.epoch)
        self.writer.add_scalar('Epoch/train_cosine_similarity', avg_loss_dict.get('cos_sim', 0), self.epoch)

        return avg_loss, avg_loss_dict

    @torch.no_grad()
    def validate(self, val_loader):
        """
        Validate on validation set.
        """
        self.encoder.eval()
        self.tcn.eval()
        self.film_generator.eval()

        val_losses = []
        val_loss_dicts = []

        pbar = tqdm(val_loader, desc="Validation")

        for input_stems, target_stems, target_features in pbar:
            # Forward pass
            loss, loss_dict = self.train_step(input_stems, target_stems, target_features)

            val_losses.append(loss.item())
            val_loss_dicts.append(loss_dict)

            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

        # Compute averages
        avg_val_loss = np.mean(val_losses)
        avg_val_loss_dict = {
            key: np.mean([d[key] for d in val_loss_dicts])
            for key in val_loss_dicts[0].keys()
        }

        # Log validation metrics to TensorBoard
        self.writer.add_scalar('Epoch/val_loss', avg_val_loss, self.epoch)
        self.writer.add_scalar('Epoch/val_style_loss', avg_val_loss_dict.get('style_loss', 0), self.epoch)
        self.writer.add_scalar('Epoch/val_cycle_loss', avg_val_loss_dict.get('cycle_loss', 0), self.epoch)
        self.writer.add_scalar('Epoch/val_cosine_similarity', avg_val_loss_dict.get('cos_sim', 0), self.epoch)

        return avg_val_loss, avg_val_loss_dict

    def save_checkpoint(self, filename="checkpoint.pt"):
        """
        Save training checkpoint.
        """
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'encoder_state_dict': self.encoder.state_dict(),
            'tcn_state_dict': self.tcn.state_dict(),
            'film_generator_state_dict': self.film_generator.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }

        checkpoint_path = self.output_dir / filename
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load training checkpoint.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.tcn.load_state_dict(checkpoint['tcn_state_dict'])
        self.film_generator.load_state_dict(checkpoint['film_generator_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Checkpoint loaded from {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train zero-shot mixing style transfer")

    # Data arguments
    parser.add_argument('--data_path', type=str, default='/nas/FMA/fma_large/',
                        help='Path to audio dataset')
    parser.add_argument('--separated_path', type=str, default='/nas/FMA/fma_separated/',
                        help='Path to pre-separated stems')
    parser.add_argument('--use_preseparated', action='store_true', default=True,
                        help='Use pre-separated stems')

    # Model arguments
    parser.add_argument('--encoder_checkpoint', type=str, default=None,
                        help='Path to pretrained encoder checkpoint (optional)')
    parser.add_argument('--hidden_channels', type=int, default=16,
                        help='TCN hidden channels (default: 128, reference architecture)')
    parser.add_argument('--num_blocks', type=int, default=14,
                        help='Number of TCN blocks (default: 14, reference architecture)')
    parser.add_argument('--kernel_size', type=int, default=15,
                        help='TCN kernel size (default: 15, reference architecture)')
    parser.add_argument('--causal', action='store_true', default=False,
                        help='Use causal TCN (default: False, reference is non-causal)')

    # Feature extraction arguments
    parser.add_argument('--use_detailed_spectral', action='store_true', default=False,
                        help='Use detailed spectral features (frequency curve) instead of 3-band')
    parser.add_argument('--n_spectral_bins', type=int, default=32,
                        help='Number of spectral bins for detailed spectral features')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr_encoder', type=float, default=1e-4,
                        help='Learning rate for encoder')
    parser.add_argument('--lr_tcn', type=float, default=2e-4,
                        help='Learning rate for TCN and FiLM generator')
    parser.add_argument('--warmup_steps', type=int, default=2000,
                        help='Warmup steps for scheduler')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Number of dataloader workers')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=32,
                        help='Accumulate gradients over N steps before optimizer step (default: 32)')

    # Loss weights
    parser.add_argument('--lambda_cycle', type=float, default=0.1,
                        help='Weight for cycle consistency loss (keep low to avoid identity collapse, default: 0.1)')
    parser.add_argument('--disable_cycle_consistency', action='store_true', default=False,
                        help='Disable cycle consistency loss (only use style matching loss)')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='outputs/style_transfer',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Logging interval')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Validation interval (epochs)')
    parser.add_argument('--save_interval', type=int, default=5,
                        help='Checkpoint save interval (epochs)')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Initialize models
    print("Initializing models...")

    # Feature extractor (create first to get feature dimensions)
    feature_extractor = MixingFeatureExtractor(
        sample_rate=44100,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        use_detailed_spectral=args.use_detailed_spectral,
        n_spectral_bins=args.n_spectral_bins
    )

    # Get feature dimensions
    feature_dim = feature_extractor.get_feature_dim()
    print(f"Feature dimension: {feature_dim}")
    if args.use_detailed_spectral:
        print(f"  Using detailed spectral with {args.n_spectral_bins} bins")
        spectral_dim_per_stem = args.n_spectral_bins + 2  # bins + tilt + flatness
    else:
        print(f"  Using 3-band spectral (old mode)")
        spectral_dim_per_stem = 5

    # Encoder
    encoder = MixingStyleEncoder(
        sample_rate=44100,
        n_fft=2048,
        hop_length=512,
        n_mels=80,
        split_size=16,
        overlap=8,
        channels=8,
        embed_dim=512,
        feature_dim=feature_dim
    ).to(device)

    # Load pretrained encoder if provided
    if args.encoder_checkpoint is not None:
        print(f"Loading pretrained encoder from {args.encoder_checkpoint}")
        checkpoint = torch.load(args.encoder_checkpoint, map_location=device)
        encoder.load_state_dict(checkpoint['model_state_dict'])

    # TCN with FiLM conditioning
    tcn = TCNMixer(
        in_channels=8,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        kernel_size=args.kernel_size,
        causal=args.causal,
        use_film=True
    ).to(device)

    # FiLM generator
    film_generator = TCNFiLMGenerator(
        embed_dim=1024,  # 512 + 512
        num_blocks=args.num_blocks,
        hidden_channels=args.hidden_channels
    ).to(device)

    # Freeze encoder (pretrained, no gradients)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    print("Encoder frozen (no gradients)")

    # Count parameters
    encoder_params = sum(p.numel() for p in encoder.parameters())
    tcn_params = sum(p.numel() for p in tcn.parameters())
    film_params = sum(p.numel() for p in film_generator.parameters())
    trainable_params = tcn_params + film_params
    print(f"Encoder parameters (frozen): {encoder_params:,}")
    print(f"TCN parameters (trainable): {tcn_params:,}")
    print(f"FiLM generator parameters (trainable): {film_params:,}")
    print(f"Total trainable parameters: {trainable_params:,}")
    print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    if args.disable_cycle_consistency:
        print(f"Cycle consistency: DISABLED (style loss only)")
    else:
        print(f"Cycle consistency: enabled (lambda={args.lambda_cycle})")

    # Optimizer - only TCN and FiLM generator (no criterion, using cosine similarity)
    optimizer = AdamW([
        {'params': tcn.parameters(), 'lr': args.lr_tcn},
        {'params': film_generator.parameters(), 'lr': args.lr_tcn},
    ], weight_decay=1e-4)

    # Scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )

    # Create datasets
    print("Creating datasets...")
    print(f"  data_path: {args.data_path}")
    print(f"  separated_path: {args.separated_path}")
    print(f"  use_preseparated: {args.use_preseparated}")

    train_dataset = StyleTransferDataset(
        data_path=args.data_path,
        separated_path=args.separated_path,
        use_preseparated=args.use_preseparated,
        clip_duration=10.0,
        sample_rate=44100,
        use_detailed_spectral=args.use_detailed_spectral,
        n_spectral_bins=args.n_spectral_bins,
    )

    # Use 90/10 train/val split
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=style_transfer_collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=style_transfer_collate_fn,
        pin_memory=True,
    )

    # Create trainer
    trainer = StyleTransferTrainer(
        encoder=encoder,
        tcn=tcn,
        film_generator=film_generator,
        feature_extractor=feature_extractor,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        log_interval=args.log_interval,
        lambda_cycle=args.lambda_cycle,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_cycle_consistency=not args.disable_cycle_consistency,
    )

    # Resume from checkpoint if provided
    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    # Training loop
    print("Starting training...")

    for epoch in range(trainer.epoch, args.num_epochs):
        trainer.epoch = epoch

        # Train
        train_loss, train_loss_dict = trainer.train_epoch(train_loader)

        print(f"\nEpoch {epoch} - Train Loss: {train_loss:.4f}")
        print(f"  Style Loss: {train_loss_dict.get('style_loss', 0):.4f}")
        print(f"  Cycle Loss: {train_loss_dict.get('cycle_loss', 0):.4f}")
        print(f"  Cosine Similarity: {train_loss_dict.get('cos_sim', 0):.3f}")

        # Validate
        if epoch % args.val_interval == 0:
            val_loss, val_loss_dict = trainer.validate(val_loader)

            print(f"\nEpoch {epoch} - Val Loss: {val_loss:.4f}")
            print(f"  Style Loss: {val_loss_dict.get('style_loss', 0):.4f}")
            print(f"  Cycle Loss: {val_loss_dict.get('cycle_loss', 0):.4f}")
            print(f"  Cosine Similarity: {val_loss_dict.get('cos_sim', 0):.3f}")

            # Save best model
            if val_loss < trainer.best_val_loss:
                trainer.best_val_loss = val_loss
                trainer.save_checkpoint("best_model.pt")
                print(f"New best validation loss: {val_loss:.4f}")

        # Save checkpoint periodically
        if epoch % args.save_interval == 0:
            trainer.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

        # Step scheduler
        trainer.scheduler.step()

    print("Training complete!")

    # Close TensorBoard writer
    trainer.writer.close()


if __name__ == '__main__':
    main()
