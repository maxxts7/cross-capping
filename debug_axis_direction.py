#!/usr/bin/env python3
"""
Debug script to check if axis directions and thresholds are correct.

This will help us understand why relaxed thresholds made benign degradation worse.
"""

import torch
import logging
from crosscap_experiment import load_original_capping, load_warmup

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def debug_axis_directions():
    """Check axis directions and threshold signs."""
    model_name = "Qwen/Qwen3-32B"

    logger.info("=== Loading Original Capping Config ===")
    try:
        assistant_axes, assistant_taus, cap_layers = load_original_capping(model_name)

        logger.info(f"Cap layers: {cap_layers}")
        logger.info(f"Number of assistant axes: {len(assistant_axes)}")
        logger.info(f"Number of assistant thresholds: {len(assistant_taus)}")

        # Check a few representative layers
        sample_layers = [cap_layers[0], cap_layers[-1]]
        for layer_idx in sample_layers:
            axis = assistant_axes[layer_idx]
            tau = assistant_taus[layer_idx]
            logger.info(f"Layer {layer_idx}: threshold = {tau:.4f}, axis norm = {axis.norm().item():.4f}")

            # Show a few components of the axis to see the direction
            logger.info(f"  First 5 axis components: {axis[:5].tolist()}")
            logger.info(f"  Last 5 axis components: {axis[-5:].tolist()}")

    except Exception as e:
        logger.error(f"Failed to load original capping: {e}")
        return

    logger.info("\n=== Loading Warmup State (if available) ===")
    try:
        warmup_path = "p25/warmup.pt"
        state = torch.load(warmup_path, map_location="cpu", weights_only=False)

        # Check if we have compliance stats
        if "compliance_stats" in state:
            stats = state["compliance_stats"]
            sample_layer = cap_layers[0]
            if sample_layer in stats:
                layer_stats = stats[sample_layer]
                logger.info(f"Layer {sample_layer} compliance stats:")
                for key, value in layer_stats.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"  {key}: {value:.4f}")

        # Check compliance axes if available
        if "compliance_axes" in state:
            compliance_axes = state["compliance_axes"]
            sample_layer = cap_layers[0]
            if sample_layer in compliance_axes:
                comp_axis = compliance_axes[sample_layer]
                assist_axis = assistant_axes[sample_layer]

                # Check dot product to see if they're aligned or opposed
                dot_product = (comp_axis @ assist_axis).item()
                logger.info(f"Layer {sample_layer}: compliance vs assistant dot product = {dot_product:.4f}")

                if abs(dot_product) > 0.8:
                    if dot_product > 0:
                        logger.warning("Compliance and assistant axes are highly ALIGNED (same direction)")
                    else:
                        logger.warning("Compliance and assistant axes are highly OPPOSED (opposite directions)")
                else:
                    logger.info("Compliance and assistant axes are reasonably orthogonal")

    except FileNotFoundError:
        logger.warning("No warmup file found at p25/warmup.pt")
    except Exception as e:
        logger.error(f"Failed to load warmup state: {e}")

def analyze_threshold_logic():
    """Analyze the threshold comparison logic."""
    logger.info("\n=== Threshold Logic Analysis ===")

    logger.info("Current logic: if detect_proj < tau_detect -> trigger capping")
    logger.info("Assistant axis points 'toward assistant' (helpful behavior)")
    logger.info("Expected behavior:")
    logger.info("  - Benign prompts: HIGH projection on assistant axis -> SHOULD NOT trigger")
    logger.info("  - Jailbreak prompts: LOW projection on assistant axis -> SHOULD trigger")
    logger.info("  - Higher threshold = more permissive = fewer false positives on benign")

    logger.info("\nIf relaxing thresholds (making them 2x higher) made benign degradation WORSE,")
    logger.info("then either:")
    logger.info("  1. The axis direction is wrong (points away from assistant)")
    logger.info("  2. The threshold comparison is wrong (should be > not <)")
    logger.info("  3. The relaxation factor is applied in wrong direction")

if __name__ == "__main__":
    debug_axis_directions()
    analyze_threshold_logic()