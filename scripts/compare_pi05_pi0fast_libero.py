#!/usr/bin/env python3
"""Compare pi05_libero (Way1: flow-matching) vs pi05_libero with FAST detokenizer (Way2: AR generation).

Way1 (baseline):
    pi05_libero normal output using its native tokenizer + flow-matching denoiser chain,
    outputting continuous actions → one trajectory.

Way2 (alignment test):
    Use the pi05_libero VLM to autoregressively generate discrete tokens, then use the
    FAST detokenizer (from pi0_fast_libero) to decode those tokens into final actions → one trajectory.

Comparison metrics:
    - Token-level: token alignment, length, special tokens, EOS/stop analysis
    - Trajectory-level: MSE, max absolute error, per-step difference
"""

import dataclasses
import json
import logging
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import sentencepiece
import tyro

from openpi.models import model as _model
from openpi.models import tokenizer as _tokenizer
from openpi.models.pi0 import Pi0
from openpi.models.pi0 import make_attn_mask
from openpi.policies import libero_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
import openpi.transforms as transforms

logger = logging.getLogger("compare_pi05_pi0fast")

# PaliGemma vocabulary constants
PALIGEMMA_EOS_TOKEN = 1
PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Args:
    """Arguments for comparing pi05_libero (Way1) vs pi05+FAST-detokenizer (Way2)."""

    # ── Checkpoint paths ──────────────────────────────────────────────
    pi05_config_name: str = "pi05_libero"
    pi05_checkpoint_dir: str = "gs://openpi-assets/checkpoints/pi05_libero"

    # ── Way2: FAST detokenizer settings ───────────────────────────────
    # Action dimension expected by the FAST detokenizer (libero = 7)
    fast_action_dim: int = 7
    # Action horizon expected by the FAST detokenizer
    fast_action_horizon: int = 10
    # Max token length for FAST tokenizer
    fast_max_token_len: int = 180

    # ── Autoregressive generation settings ────────────────────────────
    ar_max_decode_steps: int = 256
    ar_temperature: float = 0.0

    # ── Flow-matching settings ────────────────────────────────────────
    flow_num_steps: int = 10

    # ── Output ────────────────────────────────────────────────────────
    output_dir: str = "data/compare_pi05_pi0fast"
    seed: int = 42

    # ── Data ──────────────────────────────────────────────────────────
    # If set, path to a .npy file with a saved observation dict
    obs_path: str | None = None


def create_dummy_observation() -> dict:
    """Create a dummy observation for testing without a real environment."""
    return libero_policy.make_libero_example()


def load_pi05_policy_and_model(args: Args):
    """Load the pi05_libero policy with all transforms and the raw model.

    Returns:
        Tuple of (policy, model, train_config).
    """
    train_config = _config.get_config(args.pi05_config_name)
    checkpoint_dir = _tokenizer.download.maybe_download(args.pi05_checkpoint_dir)

    # Load model directly
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

    # Create policy (this also loads model, but we need policy for Way1)
    policy = _policy_config.create_trained_policy(train_config, args.pi05_checkpoint_dir)

    return policy, model, train_config


def way1_flow_matching(policy, obs: dict) -> dict:
    """Way1: Normal pi05_libero flow-matching inference → trajectory.

    Uses the standard pipeline: tokenize prompt → embed → flow-matching denoising → actions.

    Returns:
        dict with 'actions' (trajectory) and timing info.
    """
    logger.info("=== Way1: Flow-matching inference ===")
    start = time.monotonic()
    result = policy.infer(obs)
    elapsed = time.monotonic() - start
    logger.info(f"Way1 inference time: {elapsed:.3f}s")
    logger.info(f"Way1 actions shape: {result['actions'].shape}")
    return result


def way2_ar_generation_with_fast_detokenizer(
    model: Pi0,
    train_config,
    obs: dict,
    args: Args,
) -> dict:
    """Way2: Use pi05 VLM for autoregressive token generation + FAST detokenizer.

    Steps:
    1. Prepare the observation using pi05 input transforms (same as Way1 up to tokenization)
    2. Use the pi05 PaliGemma model to autoregressively generate discrete tokens
    3. Decode the generated tokens using the FAST detokenizer
    4. Return decoded actions (trajectory)

    Returns:
        dict with 'actions' (trajectory), 'generated_tokens', 'decoded_text', and timing info.
    """
    logger.info("=== Way2: AR generation + FAST detokenizer ===")

    # ── Step 1: Build the FAST tokenizer (for detokenizing) ───────────
    fast_tokenizer = _tokenizer.FASTTokenizer(
        max_len=args.fast_max_token_len,
    )

    # ── Step 2: Prepare the observation using the *FAST-style* tokenizer ──
    # We need to tokenize the input using the FAST tokenizer format so the
    # autoregressive prefix contains "Task: ..., State: ...;\nAction: " format.
    # This ensures the generated tokens can be properly decoded.
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)

    # Build the input using data transforms (LiberoInputs) but with FAST tokenization
    inputs = dict(obs)  # shallow copy
    # Apply InjectDefaultPrompt
    inputs = transforms.InjectDefaultPrompt(None)(inputs)
    # Apply data transforms (LiberoInputs)
    for t in data_config.data_transforms.inputs:
        inputs = t(inputs)
    # Apply normalization
    norm_stats = data_config.norm_stats
    inputs = transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)(inputs)
    # Resize images
    inputs = transforms.ResizeImages(224, 224)(inputs)

    # Tokenize with the FAST tokenizer (no actions, inference mode)
    prompt = inputs.pop("prompt", None)
    if prompt is None:
        raise ValueError("Prompt is required for Way2")
    if not isinstance(prompt, str):
        prompt = prompt.item()

    state = inputs["state"]
    tokens, token_mask, ar_mask, loss_mask = fast_tokenizer.tokenize(prompt, state, actions=None)

    # The prefix tokens end at the "Action: " marker
    # For inference, the tokenizer only produces prefix tokens (actions=None)
    prefix_len = int(np.sum(token_mask))
    prefix_tokens = tokens[:prefix_len]

    logger.info(f"FAST tokenizer prefix length: {prefix_len}")
    logger.info(f"Prefix tokens (first 20): {prefix_tokens[:20].tolist()}")

    # ── Step 3: Autoregressive generation using pi05 PaliGemma ────────
    start = time.monotonic()

    # Build full observation for model
    # We reuse the pi05 model's image embedding and pass it through the PaliGemma LLM
    # for autoregressive generation

    # Prepare images from the processed inputs
    obs_for_model = dict(inputs)
    obs_for_model["tokenized_prompt"] = tokens
    obs_for_model["tokenized_prompt_mask"] = token_mask
    obs_for_model["token_ar_mask"] = ar_mask
    obs_for_model["token_loss_mask"] = loss_mask

    # Batch and convert to jax arrays
    obs_jax = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], obs_for_model)
    observation = _model.Observation.from_dict(obs_jax)

    # Run the autoregressive generation
    rng = jax.random.key(args.seed)
    generated_tokens = _ar_generate_with_pi05(
        model,
        observation,
        rng,
        max_steps=args.ar_max_decode_steps,
        temperature=args.ar_temperature,
    )

    ar_time = time.monotonic() - start
    logger.info(f"Way2 AR generation time: {ar_time:.3f}s")

    # Convert to numpy
    gen_tokens_np = np.asarray(generated_tokens[0])  # unbatch

    # Decode generated tokens back to text
    pg_tokenizer_path = _tokenizer.download.maybe_download(
        "gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}
    )
    with pg_tokenizer_path.open("rb") as f:
        sp_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    # Full token sequence = prefix + generated
    full_tokens = np.concatenate([prefix_tokens, gen_tokens_np[gen_tokens_np != 0]])
    decoded_text = sp_tokenizer.decode(full_tokens.astype(int).tolist())
    logger.info(f"Generated {len(gen_tokens_np[gen_tokens_np != 0])} tokens")
    logger.info(f"Decoded text: {decoded_text[:200]}...")

    # ── Step 4: Use FAST detokenizer to extract actions ───────────────
    actions = fast_tokenizer.extract_actions(
        full_tokens.astype(np.int32),
        action_horizon=args.fast_action_horizon,
        action_dim=args.fast_action_dim,
    )

    elapsed = time.monotonic() - start
    logger.info(f"Way2 total time: {elapsed:.3f}s")
    logger.info(f"Way2 actions shape: {actions.shape}")

    return {
        "actions": actions,
        "generated_tokens": gen_tokens_np,
        "full_tokens": full_tokens,
        "decoded_text": decoded_text,
        "ar_time_s": ar_time,
        "total_time_s": elapsed,
    }


def _ar_generate_with_pi05(
    model: Pi0,
    observation: _model.Observation,
    rng: jax.Array,
    *,
    max_steps: int = 256,
    temperature: float = 0.0,
) -> np.ndarray:
    """Autoregressively generate tokens using the pi05 model's PaliGemma backbone.

    This function uses only the first expert (PaliGemma, expert 0) of the pi05 model
    to do autoregressive generation, bypassing the action expert (expert 1) and
    flow-matching denoiser entirely.

    The process:
    1. Embed prefix (images + tokenized prompt) through image encoder and LLM embedding
    2. Run forward pass through the transformer with only expert 0
    3. Get logits from the embedder's decode method
    4. Sample next token and repeat

    Args:
        model: The Pi0 model (pi05 variant).
        observation: Batched observation.
        rng: Random key.
        max_steps: Maximum number of tokens to generate.
        temperature: Sampling temperature (0 = greedy).

    Returns:
        Generated token array of shape [batch, max_steps].
    """
    observation = _model.preprocess_observation(None, observation, train=False)

    # ── Embed prefix (images + language tokens) ──
    prefix_tokens_emb, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    batch_size = prefix_tokens_emb.shape[0]
    prefix_len = prefix_tokens_emb.shape[1]

    # ── Initial forward pass to get KV cache ──
    # We use only expert 0 (PaliGemma) by passing [prefix_tokens, None]
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1

    # Forward pass with only PaliGemma expert
    (prefix_out, _), kv_cache = model.PaliGemma.llm(
        [prefix_tokens_emb, None],
        positions=positions,
        mask=prefix_attn_mask,
    )

    # Get logits from the last prefix token
    # Access the embedder through the linen module
    llm_module = model.PaliGemma.llm.module
    last_hidden = prefix_out[:, -1:, :]
    last_logit = llm_module.embedder.decode(last_hidden)

    # ── Autoregressive decoding loop ──
    output_tokens = []
    current_logit = last_logit

    # Track the actual prefix length per batch element for position computation
    actual_prefix_len = jnp.sum(prefix_mask, axis=-1)  # [batch]

    for step in range(max_steps):
        # Sample token
        rng, step_rng = jax.random.split(rng)
        if temperature > 0:
            token = jax.random.categorical(step_rng, current_logit / temperature, axis=-1)
        else:
            token = jnp.argmax(current_logit, axis=-1)

        # token shape: [batch, 1]
        output_tokens.append(token[:, 0])

        # Check for EOS
        if jnp.all(token == PALIGEMMA_EOS_TOKEN):
            logger.info(f"EOS token generated at step {step}")
            break

        # Embed the new token using the PaliGemma embedder
        new_token_emb = llm_module.embedder.encode(token).astype(jnp.dtype(llm_module.embed_dtype))

        # Compute position for new token
        new_pos = (actual_prefix_len + step).astype(jnp.int32)[:, None]

        # Build attention mask: new token attends to all previous valid tokens
        # The KV cache grew from the initial prefix forward pass, which included padding.
        # After concat with new token, kv time dim = old_cache_time + 1.
        # Mask shape must be [batch, query_len=1, key_len=old_cache_time + 1]

        # The first `prefix_len` positions in cache correspond to the prefix
        # (some may be padding), then positions prefix_len..prefix_len+step are generated tokens
        prefix_valid = jnp.arange(prefix_len)[None, :] < actual_prefix_len[:, None]
        # All generated tokens (including the current one being added) are valid
        gen_valid = jnp.ones((batch_size, step + 1), dtype=jnp.bool_)
        full_valid = jnp.concatenate([prefix_valid, gen_valid], axis=1)
        new_mask = full_valid[:, None, :]

        # Forward pass with new token, using KV cache
        (new_out, _), kv_cache = model.PaliGemma.llm(
            [new_token_emb, None],
            positions=new_pos,
            mask=new_mask,
            kv_cache=kv_cache,
        )

        # Get logits for next token
        current_logit = llm_module.embedder.decode(new_out)

    # Stack all generated tokens
    if output_tokens:
        result = jnp.stack(output_tokens, axis=1)
        # Pad to max_steps
        if result.shape[1] < max_steps:
            padding = jnp.zeros((batch_size, max_steps - result.shape[1]), dtype=result.dtype)
            result = jnp.concatenate([result, padding], axis=1)
    else:
        result = jnp.zeros((batch_size, max_steps), dtype=jnp.int32)

    return result


def compare_results(way1_result: dict, way2_result: dict, args: Args) -> dict:
    """Compare Way1 and Way2 results.

    Compares:
    - Token-level metrics (if available from Way2)
    - Trajectory-level metrics (MSE, max error, per-step diff)

    Returns:
        Comparison metrics dict.
    """
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)

    metrics = {}

    # ── Trajectory comparison ─────────────────────────────────────────
    way1_actions = way1_result["actions"]
    way2_actions = way2_result["actions"]

    logger.info(f"\nWay1 actions shape: {way1_actions.shape}")
    logger.info(f"Way2 actions shape: {way2_actions.shape}")

    # Align dimensions for comparison
    min_horizon = min(way1_actions.shape[0], way2_actions.shape[0])
    min_dim = min(way1_actions.shape[1], way2_actions.shape[1])

    w1 = way1_actions[:min_horizon, :min_dim]
    w2 = way2_actions[:min_horizon, :min_dim]

    # Per-step L2 distance
    per_step_l2 = np.sqrt(np.sum((w1 - w2) ** 2, axis=-1))
    # MSE
    mse = np.mean((w1 - w2) ** 2)
    # Max absolute error
    max_abs_err = np.max(np.abs(w1 - w2))
    # Per-dimension statistics
    per_dim_mse = np.mean((w1 - w2) ** 2, axis=0)

    metrics["trajectory"] = {
        "mse": float(mse),
        "max_abs_error": float(max_abs_err),
        "per_step_l2": per_step_l2.tolist(),
        "per_dim_mse": per_dim_mse.tolist(),
        "mean_per_step_l2": float(np.mean(per_step_l2)),
    }

    logger.info("\n--- Trajectory Metrics ---")
    logger.info(f"MSE: {mse:.6f}")
    logger.info(f"Max absolute error: {max_abs_err:.6f}")
    logger.info(f"Mean per-step L2 distance: {np.mean(per_step_l2):.6f}")
    logger.info(f"Per-step L2: {per_step_l2}")
    logger.info(f"Per-dim MSE: {per_dim_mse}")

    # ── Token-level analysis (Way2 only) ──────────────────────────────
    if "generated_tokens" in way2_result:
        gen_tokens = way2_result["generated_tokens"]
        non_zero = gen_tokens[gen_tokens != 0]

        metrics["tokens"] = {
            "total_generated": len(non_zero),
            "max_token_id": int(np.max(non_zero)) if len(non_zero) > 0 else 0,
            "min_token_id": int(np.min(non_zero)) if len(non_zero) > 0 else 0,
            "contains_eos": bool(np.any(non_zero == PALIGEMMA_EOS_TOKEN)),
            "eos_position": int(np.argmax(non_zero == PALIGEMMA_EOS_TOKEN))
            if np.any(non_zero == PALIGEMMA_EOS_TOKEN)
            else -1,
            "unique_tokens": len(np.unique(non_zero)),
        }

        logger.info("\n--- Token Metrics (Way2) ---")
        logger.info(f"Total tokens generated: {len(non_zero)}")
        logger.info(
            f"Token ID range: [{np.min(non_zero) if len(non_zero) > 0 else 'N/A'}, "
            f"{np.max(non_zero) if len(non_zero) > 0 else 'N/A'}]"
        )
        logger.info(f"Contains EOS: {np.any(non_zero == PALIGEMMA_EOS_TOKEN)}")
        logger.info(f"Unique tokens: {len(np.unique(non_zero))}")

        if "decoded_text" in way2_result:
            logger.info(f"Decoded text preview: {way2_result['decoded_text'][:300]}")

    # ── Trajectory value analysis ─────────────────────────────────────
    logger.info("\n--- Trajectory Values ---")
    logger.info(f"Way1 actions (first step): {way1_actions[0]}")
    logger.info(f"Way2 actions (first step): {way2_actions[0]}")
    if min_horizon > 1:
        logger.info(f"Way1 actions (last step): {way1_actions[min_horizon - 1]}")
        logger.info(f"Way2 actions (last step): {way2_actions[min_horizon - 1]}")

    logger.info(
        f"\nWay1 action stats: mean={np.mean(w1):.4f}, std={np.std(w1):.4f}, min={np.min(w1):.4f}, max={np.max(w1):.4f}"
    )
    logger.info(
        f"Way2 action stats: mean={np.mean(w2):.4f}, std={np.std(w2):.4f}, min={np.min(w2):.4f}, max={np.max(w2):.4f}"
    )

    metrics["way1_stats"] = {
        "mean": float(np.mean(w1)),
        "std": float(np.std(w1)),
        "min": float(np.min(w1)),
        "max": float(np.max(w1)),
    }
    metrics["way2_stats"] = {
        "mean": float(np.mean(w2)),
        "std": float(np.std(w2)),
        "min": float(np.min(w2)),
        "max": float(np.max(w2)),
    }

    return metrics


def main(args: Args) -> None:
    """Main comparison pipeline."""
    logger.info("Starting pi05_libero vs pi05+FAST comparison")
    logger.info(f"Config: {args}")

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load observation ──────────────────────────────────────────────
    if args.obs_path is not None:
        logger.info(f"Loading observation from {args.obs_path}")
        obs = np.load(args.obs_path, allow_pickle=True).item()
    else:
        logger.info("Using dummy observation (no real data provided)")
        obs = create_dummy_observation()

    logger.info(f"Observation keys: {list(obs.keys())}")

    # ── Load pi05 policy ──────────────────────────────────────────────
    logger.info("Loading pi05_libero policy...")
    policy, model, train_config = load_pi05_policy_and_model(args)
    logger.info("Policy loaded successfully.")

    # ── Way1: Flow-matching ───────────────────────────────────────────
    way1_result = way1_flow_matching(policy, dict(obs))

    # ── Way2: AR + FAST detokenizer ───────────────────────────────────
    way2_result = way2_ar_generation_with_fast_detokenizer(model, train_config, dict(obs), args)

    # ── Compare ───────────────────────────────────────────────────────
    metrics = compare_results(way1_result, way2_result, args)

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "way1_actions": way1_result["actions"].tolist(),
        "way2_actions": way2_result["actions"].tolist(),
        "metrics": metrics,
        "args": dataclasses.asdict(args),
    }

    results_path = output_dir / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Save actions as numpy arrays
    np.save(output_dir / "way1_actions.npy", way1_result["actions"])
    np.save(output_dir / "way2_actions.npy", way2_result["actions"])

    if "generated_tokens" in way2_result:
        np.save(output_dir / "way2_tokens.npy", way2_result["generated_tokens"])
        np.save(output_dir / "way2_full_tokens.npy", way2_result["full_tokens"])

    logger.info("Comparison complete!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
