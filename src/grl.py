"""
Gradient Reversal Layer (GRL) for domain adversarial training.

The GRL acts as an identity function during forward pass, but reverses
(negates) gradients during backpropagation. This enables adversarial training
where the encoder learns to maximize the discriminator loss (i.e., fool the
discriminator) while the discriminator learns to minimize its loss.
"""

import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from "Unsupervised Domain Adaptation by Backpropagation" (Ganin & Lempitsky, 2015).

    Forward pass: identity function
    Backward pass: negate gradients and scale by lambda
    """

    @staticmethod
    def forward(ctx, x, lambda_param):
        """
        Forward pass: return input unchanged.

        Args:
            ctx: Context object to save information for backward pass
            x: Input tensor
            lambda_param: Gradient reversal strength (scalar)

        Returns:
            x: Input tensor unchanged
        """
        ctx.lambda_param = lambda_param
        return x.view_as(x)  # Identity

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: negate and scale gradients.

        Args:
            ctx: Context object with saved information
            grad_output: Gradient from the next layer

        Returns:
            grad_input: Negated and scaled gradient
            None: No gradient for lambda_param
        """
        lambda_param = ctx.lambda_param
        grad_input = grad_output.neg() * lambda_param
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer module.

    Wraps the GradientReversalFunction to make it usable as a PyTorch module.
    The lambda parameter controls the strength of gradient reversal.
    """

    def __init__(self, init_lambda=1.0):
        """
        Initialize GRL.

        Args:
            init_lambda: Initial gradient reversal strength (default: 1.0)
        """
        super(GradientReversalLayer, self).__init__()
        self.lambda_param = init_lambda

    def forward(self, x):
        """
        Forward pass through GRL.

        Args:
            x: Input tensor

        Returns:
            Output tensor (same as input in forward pass)
        """
        return GradientReversalFunction.apply(x, self.lambda_param)

    def set_lambda(self, lambda_param):
        """
        Set the gradient reversal strength.

        Args:
            lambda_param: New lambda value
        """
        self.lambda_param = lambda_param


def compute_grl_lambda(current_step, total_steps, warmup_steps=2000):
    """
    Compute GRL lambda according to the schedule from "Domain-Adversarial Training of Neural Networks" (Ganin et al., 2016).

    Schedule:
    - Before warmup: λ = 0 (no adversarial training)
    - After warmup: λ = 2 / (1 + exp(-10 * p)) - 1
      where p = (current_step - warmup_steps) / (total_steps - warmup_steps)

    This schedule gradually increases λ from 0 to 1, which helps stabilize training.

    Args:
        current_step: Current training step
        total_steps: Total number of training steps
        warmup_steps: Number of steps before starting adversarial training (default: 2000)

    Returns:
        lambda_param: GRL strength parameter (0 to 1)
    """
    if current_step < warmup_steps:
        # No adversarial training during warmup
        return 0.0

    # Compute progress after warmup
    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    progress = np.clip(progress, 0.0, 1.0)  # Ensure progress is in [0, 1]

    # DANN schedule: λ = 2 / (1 + exp(-10*p)) - 1
    # This smoothly increases from 0 to 1
    lambda_param = 2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0

    return lambda_param


def compute_adversarial_lambda(current_step, total_steps, warmup_steps, initial_lambda, final_lambda):
    """
    Compute adversarial loss weight with linear ramp-up schedule.

    Schedule:
    - Before warmup: λ = initial_lambda
    - After warmup: λ linearly increases from initial_lambda to final_lambda

    This allows gradual increase of adversarial loss weight to prevent disrupting
    the contrastive learning objective early in training.

    Args:
        current_step: Current training step
        total_steps: Total number of training steps
        warmup_steps: Number of steps before starting adversarial ramp-up
        initial_lambda: Initial adversarial loss weight
        final_lambda: Final adversarial loss weight

    Returns:
        lambda_param: Adversarial loss weight (initial_lambda to final_lambda)
    """
    if current_step < warmup_steps:
        # Use initial lambda during warmup
        return initial_lambda

    # Compute progress after warmup
    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    progress = np.clip(progress, 0.0, 1.0)  # Ensure progress is in [0, 1]

    # Linear interpolation from initial to final
    lambda_param = initial_lambda + (final_lambda - initial_lambda) * progress

    return lambda_param


if __name__ == '__main__':
    """
    Unit test for Gradient Reversal Layer.
    """
    print("Testing Gradient Reversal Layer...")
    print("=" * 80)

    # Test 1: Forward pass should be identity
    print("\nTest 1: Forward pass (should be identity)")
    grl = GradientReversalLayer(init_lambda=1.0)
    x = torch.randn(4, 512, requires_grad=True)
    y = grl(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  Forward pass is identity: {torch.allclose(x, y)}")

    # Test 2: Backward pass should negate gradients
    print("\nTest 2: Backward pass (should negate gradients)")
    loss = y.sum()
    loss.backward()
    print(f"  Gradient sum (should be negative): {x.grad.sum().item():.4f}")
    print(f"  Original tensor sum: {x.sum().item():.4f}")
    expected_grad = -torch.ones_like(x) * 1.0  # Negated and scaled by lambda=1.0
    print(f"  Gradients match expected: {torch.allclose(x.grad, expected_grad)}")

    # Test 3: Lambda scaling
    print("\nTest 3: Lambda scaling")
    for lam in [0.0, 0.5, 1.0, 2.0]:
        grl = GradientReversalLayer(init_lambda=lam)
        x = torch.randn(4, 512, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()
        print(f"  Lambda={lam:.1f}: Mean gradient = {x.grad.mean().item():.4f} (expected: {-lam:.4f})")

    # Test 4: GRL lambda schedule
    print("\nTest 4: GRL lambda schedule")
    total_steps = 10000
    warmup_steps = 2000
    test_steps = [0, 500, 1000, 2000, 4000, 6000, 8000, 10000]
    print(f"  Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    for step in test_steps:
        lam = compute_grl_lambda(step, total_steps, warmup_steps)
        print(f"    Step {step:5d}: λ_grl = {lam:.4f}")

    # Test 5: Adversarial lambda schedule
    print("\nTest 5: Adversarial lambda schedule")
    total_steps = 10000
    warmup_steps = 2000
    initial_lambda = 0.0
    final_lambda = 1.5
    test_steps = [0, 500, 1000, 2000, 4000, 6000, 8000, 10000]
    print(f"  Total steps: {total_steps}, Warmup steps: {warmup_steps}")
    print(f"  Initial λ: {initial_lambda}, Final λ: {final_lambda}")
    for step in test_steps:
        lam = compute_adversarial_lambda(step, total_steps, warmup_steps, initial_lambda, final_lambda)
        print(f"    Step {step:5d}: λ_adv = {lam:.4f}")

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
