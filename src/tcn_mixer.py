"""
Temporal Convolutional Network for differentiable mixing style transfer.

Architecture:
- Input: 8 channels (4 stems × stereo)
- Output: 8 channels (processed stems)
- Receptive field: ~2 seconds at 44.1kHz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution with padding to maintain temporal dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )

    def forward(self, x):
        x = self.conv(x)
        # Remove future padding
        if self.padding > 0:
            x = x[:, :, :-self.padding]
        return x


class ResidualBlock(nn.Module):
    """
    TCN residual block with dilated convolutions.
    """
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(1, channels)
        self.norm2 = nn.GroupNorm(1, channels)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x + residual)
        return x


class TCNMixer(nn.Module):
    """
    TCN-based mixing processor for style transfer.

    Args:
        in_channels: Number of input channels (default: 8 for 4 stems × stereo)
        hidden_channels: Hidden layer channels (default: 64)
        num_blocks: Number of residual blocks (default: 10)
        kernel_size: Convolution kernel size (default: 3)

    Receptive field calculation:
        RF = 1 + sum(dilation_i * (kernel_size - 1) for each layer)

    With 10 blocks, dilations [1,2,4,8,16,32,64,128,256,512], kernel_size=3:
        RF = 1 + (1+2+4+8+16+32+64+128+256+512) * 2 = 2047 samples
        At 44.1kHz: 2047 / 44100 ≈ 0.046 seconds

    To get ~2 second receptive field, we need more layers or larger dilations.
    With 15 blocks and exponential dilations up to 2^14:
        RF ≈ 32768 samples ≈ 0.74 seconds

    With kernel_size=15 and 12 blocks:
        RF = 1 + (1+2+4+...+2048) * 14 = 57330 samples ≈ 1.3 seconds
    """
    def __init__(
        self,
        in_channels=8,
        hidden_channels=64,
        num_blocks=12,
        kernel_size=15
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # Input projection
        self.input_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)

        # TCN blocks with exponentially increasing dilation
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** i
            self.blocks.append(ResidualBlock(hidden_channels, kernel_size, dilation))

        # Output projection
        self.output_conv = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)

        # Initialize output layer to near-identity (small weights + bias=0)
        # This ensures TCN starts close to identity function via residual connection
        nn.init.normal_(self.output_conv.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.output_conv.bias)

        # Calculate receptive field
        self.receptive_field = 1 + sum(
            2 ** i * (kernel_size - 1) for i in range(num_blocks)
        )
        print(f"TCNMixer receptive field: {self.receptive_field} samples "
              f"({self.receptive_field / 44100:.3f} seconds at 44.1kHz)")

    def forward(self, x):
        """
        Args:
            x: (B, 8, T) tensor of stems [vocals_L, vocals_R, bass_L, bass_R, ...]

        Returns:
            Processed stems (B, 8, T)
        """
        # Project to hidden dimension
        h = self.input_conv(x)

        # Apply TCN blocks
        for block in self.blocks:
            h = block(h)

        # Project to output
        out = self.output_conv(h)

        # Residual connection with input
        out = out + x

        return out

    def process_stems_dict(self, stems_dict):
        """
        Convenience method to process stems dictionary.

        Args:
            stems_dict: Dict with keys ['vocals', 'bass', 'drums', 'other']
                       Each value is (2, T) tensor

        Returns:
            Processed stems dict with same structure
        """
        # Stack stems: (8, T)
        stem_order = ['vocals', 'bass', 'drums', 'other']
        stems_list = []
        for stem_name in stem_order:
            stem = stems_dict[stem_name]  # (2, T)
            stems_list.append(stem)

        stacked = torch.cat(stems_list, dim=0)  # (8, T)
        stacked = stacked.unsqueeze(0)  # (1, 8, T)

        # Process
        processed = self.forward(stacked)  # (1, 8, T)
        processed = processed.squeeze(0)  # (8, T)

        # Unstack
        processed_dict = {}
        for i, stem_name in enumerate(stem_order):
            processed_dict[stem_name] = processed[i*2:(i+1)*2, :]  # (2, T)

        return processed_dict


def create_tcn_mixer(receptive_field_seconds=2.0, sample_rate=44100):
    """
    Create TCN mixer with specified receptive field.

    Args:
        receptive_field_seconds: Desired receptive field in seconds
        sample_rate: Audio sample rate

    Returns:
        TCNMixer instance
    """
    target_rf_samples = int(receptive_field_seconds * sample_rate)

    # Calculate required number of blocks
    # RF = 1 + sum(2^i * (k-1)) for i in 0..n-1
    # With k=15: RF = 1 + 14 * (2^n - 1)
    # Solve for n: 2^n = (RF - 1) / 14 + 1

    kernel_size = 15
    n = math.ceil(math.log2((target_rf_samples - 1) / (kernel_size - 1) + 1))

    print(f"Target receptive field: {target_rf_samples} samples ({receptive_field_seconds}s)")
    print(f"Using {n} TCN blocks with kernel_size={kernel_size}")

    return TCNMixer(
        in_channels=8,
        hidden_channels=64,
        num_blocks=n,
        kernel_size=kernel_size
    )


if __name__ == "__main__":
    # Test TCN mixer
    tcn = create_tcn_mixer(receptive_field_seconds=2.0)

    # Test forward pass
    x = torch.randn(1, 8, 44100)  # 1 second of audio
    y = tcn(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Receptive field: {tcn.receptive_field} samples")

    # Count parameters
    total_params = sum(p.numel() for p in tcn.parameters())
    print(f"Total parameters: {total_params:,}")
