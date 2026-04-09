#!/usr/bin/env python3
"""eval_reconstruct.py — diagnostic: can the model reconstruct masked/corrupted text?

Supports multiple models and sampling methods. Uses model_eval.CHECKPOINTS
for model loading and provides reconstruction-adapted versions of all
DDPM-based samplers from model_eval.py.

Usage:
  python eval_reconstruct.py \\
    --model refine --sampler refine \\
    --input_file samples.jsonl --steps 128 \\
    --show_intermediate --show_frequency 5
"""
import argparse
import itertools
import json
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import omegaconf
import dataloader
import diffusion as diffusion_module
from diffusion import _sample_categorical
from model_eval import CHECKPOINTS, _register_resolvers


# ── Model loading ─────────────────────────────────────────────────────────────

def load_mdlm(model_name: str, device: str):
    _register_resolvers()
    ckpt_path = CHECKPOINTS[model_name]
    raw = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = raw['hyper_parameters']['config']
    if not isinstance(config, omegaconf.DictConfig):
        config = omegaconf.OmegaConf.create(config)
    with omegaconf.open_dict(config):
        config.trainer.num_nodes = 1
        config.trainer.devices = 1
        config.trainer.accumulate_grad_batches = 1
    tokenizer = dataloader.get_tokenizer(config)
    model = diffusion_module.Diffusion.load_from_checkpoint(
        ckpt_path, tokenizer=tokenizer, config=config, map_location=device)
    model.to(device).eval()
    if model.ema is not None:
        params = lambda: itertools.chain(
            model.backbone.parameters(), model.noise.parameters())
        model.ema.store(params())
        model.ema.copy_to(params())
    return model


# ── Step-data helpers for visualization ───────────────────────────────────────

def _get_correction_viz(step_num, x_before, x_after, log_probs, det_candidates,
                        mask_idx, content_mask=None, n_corrections_made=0):
    """Build step_data tuple for visualization (batch element 0 only)."""
    top_corrections = []
    eligible = (x_before[0] != mask_idx)
    if content_mask is not None:
        eligible = eligible & content_mask[0]
    unmasked_pos = eligible.nonzero(as_tuple=True)[0]
    if unmasked_pos.numel() > 0:
        changed = det_candidates[0][unmasked_pos] != x_before[0][unmasked_pos]
        cand_pos = unmasked_pos[changed]
        if cand_pos.numel() > 0:
            lp_new = log_probs[0][cand_pos, det_candidates[0][cand_pos]]
            lp_cur = log_probs[0][cand_pos, x_before[0][cand_pos]]
            ratios = lp_new - lp_cur
            k8 = min(8, cand_pos.numel())
            vals, idx8 = ratios.topk(k8, largest=True)
            top_corrections = [
                (vals[i].item(), x_before[0][cand_pos[idx8[i]]].item(),
                 det_candidates[0][cand_pos[idx8[i]]].item())
                for i in range(k8)
            ]
    n_unmasked_before = int((x_before[0] != mask_idx).sum().item())
    n_unmasked_after = int((x_after[0] != mask_idx).sum().item())
    return (step_num, x_before[0].cpu(), det_candidates[0].cpu(), x_after[0].cpu(),
            top_corrections, n_unmasked_after - n_unmasked_before, n_corrections_made)


def _compute_backbone_viz(model, step_num, x_before, x_after, sigma, mask_idx,
                          content_mask=None, n_corrections_made=0):
    """One backbone call on x_after to produce visualization data."""
    sigma_proc = model._process_sigma(sigma)
    with torch.cuda.amp.autocast(dtype=torch.float32):
        logits = model.backbone(x_after, sigma_proc)
    logits[:, :, mask_idx] += model.neg_infinity
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    det_candidates = log_probs.argmax(dim=-1)
    return _get_correction_viz(step_num, x_before, x_after, log_probs,
                               det_candidates, mask_idx, content_mask,
                               n_corrections_made)


# ── Reconstruction samplers ──────────────────────────────────────────────────

@torch.no_grad()
def reconstruct_ddpm(model, x_init, steps, device, collect_steps=False,
                     content_mask=None, n_correct_per_step=0, show_frequency=1):
    """DDPM reconstruction with optional per-step correction.

    n_correct_per_step=0 → vanilla DDPM (fills masks only, preserves unmasked).
    n_correct_per_step>0 → DDPM + top-k correction each step (used by vanilla with corrections).
    """
    mask_idx = model.mask_index
    eps = 1e-5
    x = x_init.clone().to(device)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    dt = (1 - eps) / steps
    step_data = []
    flip_counts = []  # per-step flip counts: list of [batch_size] tensors

    for step in range(steps):
        should_collect = collect_steps and (step % show_frequency == 0 or step == steps - 1)
        x_before = x.clone()
        t = timesteps[step] * torch.ones(x.shape[0], 1, device=device)

        sigma_t, _ = model.noise(t)
        sigma_s, _ = model.noise(t - dt)
        if sigma_t.ndim > 1: sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1: sigma_s = sigma_s.squeeze(-1)
        move_chance_t = (1 - torch.exp(-sigma_t))[:, None, None]
        move_chance_s = (1 - torch.exp(-sigma_s))[:, None, None]

        log_p_x0 = model.forward(x, sigma_t)
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        q_xs[:, :, mask_idx] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        copy_flag = (x != mask_idx).to(x.dtype)
        x = (copy_flag * x + (1 - copy_flag) * _x).long()

        # Correction phase
        log_probs = det_candidates = None
        n_corrections = 0
        if n_correct_per_step > 0:
            sigma = model._process_sigma(sigma_s)
            with torch.cuda.amp.autocast(dtype=torch.float32):
                logits = model.backbone(x, sigma)
            logits[:, :, mask_idx] += model.neg_infinity
            log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
            det_candidates = log_probs.argmax(dim=-1)
            for b in range(x.shape[0]):
                eligible = (x[b] != mask_idx)
                if content_mask is not None:
                    eligible = eligible & content_mask[b]
                unmasked_pos = eligible.nonzero(as_tuple=True)[0]
                if unmasked_pos.numel() == 0: continue
                changed = det_candidates[b][unmasked_pos] != x[b][unmasked_pos]
                cand_pos = unmasked_pos[changed]
                if cand_pos.numel() == 0: continue
                k = min(n_correct_per_step, cand_pos.numel())
                lp_new = log_probs[b][cand_pos, det_candidates[b][cand_pos]]
                lp_cur = log_probs[b][cand_pos, x[b][cand_pos]]
                _, top_idx = (lp_new - lp_cur).topk(k, largest=True)
                x[b, cand_pos[top_idx]] = det_candidates[b, cand_pos[top_idx]]
                if b == 0: n_corrections = top_idx.numel()

        # Count flipped unmasked tokens
        was_unmasked = (x_before != mask_idx)
        flipped = was_unmasked & (x != x_before)
        flip_counts.append(flipped.sum(dim=-1))  # [batch_size]

        if should_collect:
            if log_probs is not None:
                step_data.append(_get_correction_viz(
                    step, x_before, x, log_probs, det_candidates,
                    mask_idx, content_mask, n_corrections))
            else:
                step_data.append(_compute_backbone_viz(
                    model, step, x_before, x, sigma_s, mask_idx,
                    content_mask))

    # Final noise removal
    t_final = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
    x = model.forward(x, model.noise(t_final)[0]).argmax(dim=-1)
    return x, step_data, flip_counts


@torch.no_grad()
def reconstruct_ddpm_onepass(model, x_init, steps, device, collect_steps=False,
                             content_mask=None, n_correct_per_step=10,
                             show_frequency=1):
    """Single-pass reconstruction: one backbone call for both DDPM and correction.

    Inlines the DDPM update to capture raw logits before SUBS parameterization,
    then reuses those same logits for the correction phase.
    """
    mask_idx = model.mask_index
    eps = 1e-5
    x = x_init.clone().to(device)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    dt = (1 - eps) / steps
    step_data = []
    flip_counts = []  # per-step flip counts: list of [batch_size] tensors

    for step in range(steps):
        should_collect = collect_steps and (step % show_frequency == 0 or step == steps - 1)
        x_before = x.clone()
        t = timesteps[step] * torch.ones(x.shape[0], 1, device=device)

        n_correct_per_step = max(1, int(20 / math.sqrt(step + 1)))

        sigma_t, _ = model.noise(t)
        sigma_s, _ = model.noise(t - dt)
        if sigma_t.ndim > 1: sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1: sigma_s = sigma_s.squeeze(-1)
        move_chance_t = (1 - torch.exp(-sigma_t))[:, None, None]
        move_chance_s = (1 - torch.exp(-sigma_s))[:, None, None]

        # Single backbone call
        sigma = model._process_sigma(sigma_t)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            raw_logits = model.backbone(x, sigma)

        # SUBS parameterization for DDPM
        subs_logits = raw_logits.clone()
        subs_logits[:, :, mask_idx] += model.neg_infinity
        subs_logits = subs_logits - torch.logsumexp(subs_logits, dim=-1, keepdim=True)
        unmasked_indices = (x != mask_idx)
        subs_logits[unmasked_indices] = model.neg_infinity
        subs_logits[unmasked_indices, x[unmasked_indices]] = 0

        previously_unmasked = (x != mask_idx)

        # DDPM posterior sampling
        q_xs = subs_logits.exp() * (move_chance_t - move_chance_s)
        q_xs[:, :, mask_idx] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        copy_flag = previously_unmasked.to(x.dtype)
        x = (copy_flag * x + (1 - copy_flag) * _x).long()

        # Correction using same raw_logits (only on previously unmasked positions)
        corr_logits = raw_logits.clone()
        corr_logits[:, :, mask_idx] += model.neg_infinity
        log_probs = corr_logits - torch.logsumexp(corr_logits, dim=-1, keepdim=True)
        det_candidates = log_probs.argmax(dim=-1)
        n_corrections = 0

        if n_correct_per_step > 0:
            for b in range(x.shape[0]):
                prev_pos = previously_unmasked[b].nonzero(as_tuple=True)[0]
                if content_mask is not None:
                    mask_b = content_mask[b][prev_pos]
                    prev_pos = prev_pos[mask_b]
                if prev_pos.numel() == 0: continue
                changed = det_candidates[b][prev_pos] != x[b][prev_pos]
                cand_pos = prev_pos[changed]
                if cand_pos.numel() == 0: continue
                lp_new = log_probs[b][cand_pos, det_candidates[b][cand_pos]]
                lp_cur = log_probs[b][cand_pos, x[b][cand_pos]]
                k = min(n_correct_per_step, cand_pos.numel())
                _, top_idx = (lp_new - lp_cur).topk(k, largest=True)
                x[b, cand_pos[top_idx]] = det_candidates[b, cand_pos[top_idx]]
                if b == 0: n_corrections = top_idx.numel()

        # Count flipped unmasked tokens
        was_unmasked = (x_before != mask_idx)
        flipped = was_unmasked & (x != x_before)
        flip_counts.append(flipped.sum(dim=-1))  # [batch_size]

        if should_collect:
            step_data.append(_get_correction_viz(
                step, x_before, x, log_probs, det_candidates,
                mask_idx, content_mask, n_corrections))

    # Final noise removal
    t_final = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
    x = model.forward(x, model.noise(t_final)[0]).argmax(dim=-1)
    return x, step_data, flip_counts


@torch.no_grad()
def reconstruct_ddpm_atomic(model, x_init, steps, device, collect_steps=False,
                            content_mask=None, **kwargs):
    """Single forward pass: feed input tokens to the model, take argmax of logits.

    No sampling, no DDPM schedule, no iteration — purely observes the model's
    direct response to the input. Collects the same visualization data as
    reconstruct_ddpm_onepass.
    """
    mask_idx = model.mask_index
    eps = 1e-5
    x = x_init.clone().to(device)

    # Use t=1 (fully noised) so the model operates in its expected regime
    t = torch.ones(x.shape[0], 1, device=device)
    sigma_t, _ = model.noise(t)
    if sigma_t.ndim > 1:
        sigma_t = sigma_t.squeeze(-1)

    # Single backbone call
    sigma = model._process_sigma(sigma_t)
    with torch.cuda.amp.autocast(dtype=torch.float32):
        raw_logits = model.backbone(x, sigma)

    # Compute log-probs (suppress mask token)
    corr_logits = raw_logits.clone()
    corr_logits[:, :, mask_idx] += model.neg_infinity
    log_probs = corr_logits - torch.logsumexp(corr_logits, dim=-1, keepdim=True)
    det_candidates = log_probs.argmax(dim=-1)

    x_before = x.clone()
    x_out = det_candidates.clone()

    # Count flipped unmasked tokens within the content window only
    was_unmasked = (x_before != mask_idx)
    if content_mask is not None:
        was_unmasked = was_unmasked & content_mask
    flipped = was_unmasked & (x_out != x_before)
    flip_counts = [flipped.sum(dim=-1)]  # single-element list for API compatibility

    step_data = []
    if collect_steps:
        step_data.append(_get_correction_viz(
            0, x_before, x_out, log_probs, det_candidates,
            mask_idx, content_mask, n_corrections_made=0))

    return x_out, step_data, flip_counts, log_probs


@torch.no_grad()
def reconstruct_ddpm_stochastic_backloaded(model, x_init, steps, device, collect_steps=False,
                         content_mask=None, n_correct_per_step=10,
                         show_frequency=1, guidance_scale=1.0, 
                         proseco_budget=1, proseco_freq=1):
    """
    v11: Dual-Logit Guided ASRM for Reconstruction Eval.
    Integrates ASRM (Re-Masking instead of token swapping), Time-Annealed Draft, 
    and CFG Guidance while tracking unmasked flips and step data for visualization.
    """
    mask_idx = model.mask_index
    eps = 1e-5
    
    # --- NFE Parity Calculation ---
    S = proseco_budget
    omega = proseco_freq
    actual_steps = max(1, int(steps / (1 + (S / omega))))
    
    x = x_init.clone().to(device)
    timesteps = torch.linspace(1, eps, actual_steps + 1, device=device)
    dt = (1 - eps) / actual_steps
    
    step_data = []
    flip_counts = []

    for step in range(actual_steps):
        should_collect = collect_steps and (step % show_frequency == 0 or step == actual_steps - 1)
        x_before = x.clone()
        t = timesteps[step] * torch.ones(x.shape[0], 1, device=device)

        # ── 1. Pure Anchor Pass (Local Precision) ──────────────────
        sigma_t, _ = model.noise(t)
        sigma_s, _ = model.noise(t - dt)
        if sigma_t.ndim > 1: sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1: sigma_s = sigma_s.squeeze(-1)
        move_chance_t = (1 - torch.exp(-sigma_t))[:, None, None]
        move_chance_s = (1 - torch.exp(-sigma_s))[:, None, None]

        # Handle potentially different model signatures
        sigma = model._process_sigma(sigma_t) if hasattr(model, '_process_sigma') else sigma_t
        with torch.cuda.amp.autocast(dtype=torch.float32):
            anchor_logits = getattr(model, 'backbone', model.forward)(x, sigma)

        anchor_logits[:, :, mask_idx] += getattr(model, 'neg_infinity', -1e9)
        anchor_log_probs = anchor_logits - torch.logsumexp(anchor_logits, dim=-1, keepdim=True)
        candidates = anchor_log_probs.argmax(dim=-1)
        unmasked_before = (x != mask_idx)
        
        n_corrections = 0
        corr_log_probs = anchor_log_probs
        corr_det_candidates = candidates

        # ── 2. Re-Masking & Dual-Logit Draft ───────────────────────
        if (step + 1) % omega == 0 and S > 0 and n_correct_per_step > 0:
            
            # Step A: Identify Corruptions & Re-Mask (ASRM)
            for b in range(x.shape[0]):
                u_pos = unmasked_before[b].nonzero(as_tuple=True)[0]
                
                # IMPORTANT: Apply content_mask from reconstruct eval
                if content_mask is not None:
                    mask_b = content_mask[b][u_pos]
                    u_pos = u_pos[mask_b]
                    
                if u_pos.numel() == 0: continue
                
                changed = candidates[b][u_pos] != x[b][u_pos]
                cand_pos = u_pos[changed]
                if cand_pos.numel() == 0: continue
                
                lp_new = anchor_log_probs[b][cand_pos, candidates[b][cand_pos]]
                lp_cur = anchor_log_probs[b][cand_pos, x[b][cand_pos]]
                doubt = lp_new - lp_cur
                
                valid = doubt > 0.0
                if not valid.any(): continue
                
                valid_pos = cand_pos[valid]
                k = min(n_correct_per_step, valid_pos.numel())
                _, top_idx = doubt[valid].topk(k, largest=True)
                target_pos = valid_pos[top_idx]
                
                # THE FIX: Revert corrupted token to mask. Do not direct swap.
                x[b, target_pos] = mask_idx
                unmasked_before[b, target_pos] = False 
                if b == 0: n_corrections = top_idx.numel()

            # Step B: Time-Annealed Stochastic Draft
            t_frac = step / max(1, actual_steps - 1)
            masks_now = (x == mask_idx)
            
            anchor_probs = anchor_log_probs.exp()
            draft_candidates = _sample_categorical(anchor_probs)
            
            # Gradually allow draft to expand based on timestep
            fill_mask = masks_now & (torch.rand_like(x, dtype=torch.float) < t_frac)
            
            x_draft = x.clone()
            x_draft[fill_mask] = draft_candidates[fill_mask]
            
            # Step C: Constrained S-Pass Draft
            draft_logits = anchor_logits
            for _s in range(S):
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    draft_logits = getattr(model, 'backbone', model.forward)(x_draft, sigma)
                
                draft_logits[:, :, mask_idx] += getattr(model, 'neg_infinity', -1e9)
                
                if _s < S - 1:
                    new_cands = draft_logits.argmax(dim=-1)
                    x_draft[fill_mask] = new_cands[fill_mask]
            
            # Step D: CFG Guidance Blend
            guided_logits = anchor_logits + guidance_scale * (draft_logits - anchor_logits)
            log_x_theta = guided_logits - torch.logsumexp(guided_logits, dim=-1, keepdim=True)
            
            # Update visualization trackers to show the guided probabilities
            corr_log_probs = log_x_theta
            corr_det_candidates = log_x_theta.argmax(dim=-1)
            
            del x_draft, draft_logits
        else:
            log_x_theta = anchor_log_probs

        # ── 3. DDPM Unmasking ──────────────────────────────────────
        # Nullify unmasked log probs to strictly maintain them (unless we just re-masked them)
        subs_logits = log_x_theta.clone()
        subs_logits[unmasked_before] = getattr(model, 'neg_infinity', -1e9)
        subs_logits[unmasked_before, x[unmasked_before]] = 0

        p_x0 = subs_logits.exp()
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, mask_idx] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)

        copy_flag = unmasked_before.to(x.dtype)
        x = (copy_flag * x + (1 - copy_flag) * _x).long()

        # Track evaluation statistics
        was_unmasked = (x_before != mask_idx)
        flipped = was_unmasked & (x != x_before)
        flip_counts.append(flipped.sum(dim=-1))

        if should_collect:
            # Assumes _get_correction_viz is available in scope (standard in your eval file)
            step_data.append(_get_correction_viz(
                step, x_before, x, corr_log_probs, corr_det_candidates,
                mask_idx, content_mask, n_corrections))

    # Final noise removal
    t_final = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
    sigma_final = model._process_sigma(model.noise(t_final)[0]) if hasattr(model, '_process_sigma') else model.noise(t_final)[0]
    
    with torch.cuda.amp.autocast(dtype=torch.float32):
        final_logits = getattr(model, 'backbone', model.forward)(x, sigma_final)
        
    x = final_logits.argmax(dim=-1)
    
    return x, step_data, flip_counts


@torch.no_grad()
def _select_corrector_tokens(log_x_theta, corrector_x, method='argmax'):
    """Select tokens from corrector logits using the specified method.

    Args:
        log_x_theta: Log-probabilities of shape (batch, seq_len, vocab_size).
        corrector_x: Current corrector tokens of shape (batch, seq_len).
        method: 'argmax' — update all positions with argmax tokens.
                'topk'   — take argmax at every position, but only apply
                           at the k=100 positions with highest argmax
                           confidence; other positions keep their value
                           from corrector_x.
    """
    if method == 'argmax':
        return log_x_theta.argmax(dim=-1)
    elif method == 'topk':
        k = 100
        argmax_tokens = log_x_theta.argmax(dim=-1)
        argmax_conf = log_x_theta.max(dim=-1).values
        # Algorithm 5: zero out confidence where output == input
        # so only positions where the corrector *changed* the token compete
        unchanged = (argmax_tokens == corrector_x)
        argmax_conf = argmax_conf.masked_fill(unchanged, float('-inf'))
        seq_len = argmax_conf.shape[-1]
        k_clamped = min(k, seq_len)
        _, topk_pos = argmax_conf.topk(k_clamped, dim=-1)
        mask = torch.zeros_like(argmax_conf, dtype=torch.bool)
        mask.scatter_(-1, topk_pos, True)
        return torch.where(mask, argmax_tokens, corrector_x)
    else:
        raise ValueError(f"Unknown corrector_sampling method: {method}")


def reconstruct_proseco(model, x_init, steps, device, proseco_budget=3,
                        proseco_freq=1, corrector_sees_masks=False,
                        corrector_sampling='argmax',
                        collect_steps=False, content_mask=None, show_frequency=1):
    """ProSeCo reconstruction: DDPM unmasking with periodic self-correction.

    corrector_sees_masks=False → ProSeCo (corrector gets dense argmax sequence).
    corrector_sees_masks=True  → ProSeCo-M (corrector sees masked + unmasked tokens).
    """
    mask_idx = model.mask_index
    eps = 1e-5
    omega = proseco_freq
    S = proseco_budget
    actual_steps = max(int(steps / (1 + (S / omega))), 1)

    x = x_init.clone().to(device)
    timesteps = torch.linspace(1, eps, actual_steps + 1, device=device)
    dt = (1 - eps) / actual_steps
    step_data = []
    flip_counts = []  # per-step flip counts: list of [batch_size] tensors

    for i in range(actual_steps):
        should_collect = collect_steps and (i % show_frequency == 0 or i == actual_steps - 1)
        x_before = x.clone()
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)

        sigma_t, _ = model.noise(t)
        sigma_s, _ = model.noise(t - dt)
        if sigma_t.ndim > 1: sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1: sigma_s = sigma_s.squeeze(-1)
        move_chance_t = (1 - torch.exp(-sigma_t))[:, None, None]
        move_chance_s = (1 - torch.exp(-sigma_s))[:, None, None]

        log_x_theta = model.forward(x, sigma_t)
        use_f64 = getattr(model.config, 'sampling', None) and \
            getattr(model.config.sampling, 'use_float64', False)
        if use_f64:
            log_x_theta = log_x_theta.to(torch.float64)

        n_corrections = 0
        if (i + 1) % omega == 0:
            unmasked = (x != mask_idx)

            if corrector_sees_masks:
                corrector_x = x.clone()
                corrector_x = torch.where(unmasked, log_x_theta.argmax(dim=-1), corrector_x)
            else:
                corrector_x = log_x_theta.argmax(dim=-1)

            for _s in range(S):
                sigma = model._process_sigma(sigma_t)
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    corrector_logits = model.backbone(corrector_x, sigma)
                corrector_logits[..., mask_idx] += model.neg_infinity
                corrector_log_x_theta = corrector_logits.log_softmax(dim=-1)
                if use_f64:
                    corrector_log_x_theta = corrector_log_x_theta.to(torch.float64)
                refined_tokens = _select_corrector_tokens(corrector_log_x_theta, corrector_x, corrector_sampling)
                if corrector_sees_masks:
                    corrector_x = torch.where(unmasked, refined_tokens, corrector_x)
                else:
                    corrector_x = refined_tokens

            if should_collect:
                n_corrections = int((corrector_x[0][unmasked[0]] != x[0][unmasked[0]]).sum().item())

            x = torch.where(unmasked, corrector_x, x)
            log_x_theta = corrector_log_x_theta

        # DDPM posterior sampling
        x_theta = log_x_theta.exp()
        q_xs = x_theta * (move_chance_t - move_chance_s)
        q_xs[:, :, mask_idx] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)
        copy_flag = (x != mask_idx)
        x = torch.where(copy_flag, x, _x)

        # Count flipped unmasked tokens
        was_unmasked = (x_before != mask_idx)
        flipped = was_unmasked & (x != x_before)
        flip_counts.append(flipped.sum(dim=-1))  # [batch_size]

        if should_collect:
            step_data.append(_compute_backbone_viz(
                model, i, x_before, x, sigma_s, mask_idx,
                content_mask, n_corrections))

    # Final denoising pass
    t_final = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
    unet_conditioning = model.noise(t_final)[0]
    x = model.forward(x, unet_conditioning).argmax(dim=-1)

    return x, step_data, flip_counts


RECON_SAMPLERS = ['vanilla', 'refine', 'proseco', 'proseco_m']


def run_reconstruction(sampler_name, model, x_init, steps, device, **kwargs):
    """Dispatch to the appropriate reconstruction sampler."""
    if sampler_name == 'vanilla':
        kwargs.pop('n_correct_per_step', None)
        kwargs.pop('proseco_budget', None)
        kwargs.pop('proseco_freq', None)
        kwargs.pop('corrector_sampling', None)
        return reconstruct_ddpm(model, x_init, steps, device,
                                n_correct_per_step=0, **kwargs)
    elif sampler_name == 'refine':
        kwargs.pop('proseco_budget', None)
        kwargs.pop('proseco_freq', None)
        kwargs.pop('corrector_sampling', None)
        x_out, step_data, flip_counts = reconstruct_ddpm_stochastic_backloaded(
            model, x_init, steps, device, **kwargs)
        return x_out, step_data, flip_counts
    elif sampler_name == 'proseco':
        proseco_budget = kwargs.pop('proseco_budget', 3)
        proseco_freq = kwargs.pop('proseco_freq', 1)
        corrector_sampling = kwargs.pop('corrector_sampling', 'argmax')
        kwargs.pop('n_correct_per_step', None)
        return reconstruct_proseco(model, x_init, steps, device,
                                   proseco_budget=proseco_budget,
                                   proseco_freq=proseco_freq,
                                   corrector_sees_masks=False,
                                   corrector_sampling=corrector_sampling, **kwargs)
    elif sampler_name == 'proseco_m':
        proseco_budget = kwargs.pop('proseco_budget', 3)
        proseco_freq = kwargs.pop('proseco_freq', 1)
        corrector_sampling = kwargs.pop('corrector_sampling', 'argmax')
        kwargs.pop('n_correct_per_step', None)
        return reconstruct_proseco(model, x_init, steps, device,
                                   proseco_budget=proseco_budget,
                                   proseco_freq=proseco_freq,
                                   corrector_sees_masks=True,
                                   corrector_sampling=corrector_sampling, **kwargs)
    else:
        raise ValueError(f'Unknown sampler: {sampler_name}')


# ── Visualization ────────────────────────────────────────────────────────────

def visualize_step(tok, step, x_before, candidates, x_after, mask_idx, display_len,
                   top_corrections=None, n_to_unmask=None, n_to_correct=None, chunk=50):
    """Print model proposals vs what was actually selected for one sampler step."""
    try:
        from rich.console import Console
        from rich.text import Text
        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    counts = ''
    if n_to_unmask is not None:
        counts += f'  unmask={n_to_unmask}'
    if n_to_correct is not None:
        counts += f'  correct={n_to_correct}'
    print(f'\n  -- Step {step} --{counts}')

    if top_corrections:
        print('  Top confidence-ratio corrections:')
        for rank, (log_ratio, orig_id, new_id) in enumerate(top_corrections, 1):
            orig_str = tok.decode([orig_id])
            new_str  = tok.decode([new_id])
            ratio_str = f'{log_ratio:+.3f}'
            if use_rich:
                line = Text(f'    {rank}. ')
                line.append(repr(orig_str), style='red')
                line.append(' -> ')
                line.append(repr(new_str), style='green')
                line.append(f'  log_ratio={ratio_str}', style='dim')
                console.print(line)
            else:
                print(f'    {rank}. {repr(orig_str)} -> {repr(new_str)}  log_ratio={ratio_str}')
        print()
    x_before = x_before[:display_len]
    candidates = candidates[:display_len]
    x_after = x_after[:display_len]
    L = display_len

    for start in range(0, L, chunk):
        end = min(start + chunk, L)
        before_toks = [tok.decode([t]) for t in x_before[start:end].tolist()]
        cand_toks   = [tok.decode([t]) for t in candidates[start:end].tolist()]
        after_toks  = [tok.decode([t]) for t in x_after[start:end].tolist()]

        if use_rich:
            before_line = Text("  BEFORE:    ")
            cand_line   = Text("  PROPOSED:  ")
            after_line  = Text("  SELECTED:  ")
            for i, pos in enumerate(range(start, end)):
                b_tok, c_tok, a_tok = before_toks[i], cand_toks[i], after_toks[i]
                if x_before[pos].item() == mask_idx:
                    before_line.append("[M] ", style="yellow")
                else:
                    before_line.append(b_tok + " ")
                if x_before[pos].item() == mask_idx:
                    cand_line.append(c_tok + " ", style="cyan")
                elif c_tok != b_tok:
                    cand_line.append(c_tok + " ", style="magenta")
                else:
                    cand_line.append(c_tok + " ", style="dim")
                if a_tok != b_tok:
                    after_line.append(a_tok + " ", style="green")
                else:
                    after_line.append(a_tok + " ", style="dim")
            console.print(before_line)
            console.print(cand_line)
            console.print(after_line)
            console.print()
        else:
            def fmt_before(i, pos):
                return "[M]" if x_before[pos].item() == mask_idx else before_toks[i]
            def fmt_after(i, pos):
                return f"[+{after_toks[i]}]" if after_toks[i] != before_toks[i] else after_toks[i]
            print("  BEFORE:  ", " ".join(fmt_before(i, p) for i, p in enumerate(range(start, end))))
            print("  PROPOSED:", " ".join(cand_toks))
            print("  SELECTED:", " ".join(fmt_after(i, p) for i, p in enumerate(range(start, end))))
            print()


def visualize(tok, x_orig, x_init, x_out, masked_pos, corrupt_pos, mask_idx, chunk=50):
    try:
        from rich.console import Console
        from rich.text import Text
        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    L = x_orig.shape[0]
    for start in range(0, L, chunk):
        end = min(start + chunk, L)
        orig_toks = [tok.decode([t]) for t in x_orig[start:end].tolist()]
        init_toks = [tok.decode([t]) for t in x_init[start:end].tolist()]
        out_toks  = [tok.decode([t]) for t in x_out[start:end].tolist()]

        if use_rich:
            orig_line = Text("ORIGINAL: ")
            init_line = Text("INPUT:    ")
            out_line  = Text("OUTPUT:   ")
            for i, pos in enumerate(range(start, end)):
                orig_line.append(orig_toks[i] + " ")
                if masked_pos[pos]:
                    init_line.append("[M] ", style="yellow")
                elif corrupt_pos[pos]:
                    init_line.append(f"[C:{init_toks[i]}] ", style="red")
                else:
                    init_line.append(init_toks[i] + " ")
                correct = (x_out[pos] == x_orig[pos]).item()
                is_special = masked_pos[pos] or corrupt_pos[pos]
                if is_special and correct:
                    out_line.append(out_toks[i] + " ", style="green")
                elif not correct:
                    out_line.append(f"[{out_toks[i]}] ", style="red")
                else:
                    out_line.append(out_toks[i] + " ")
            console.print(orig_line)
            console.print(init_line)
            console.print(out_line)
            console.print()
        else:
            init_str = " ".join(
                "[M]" if masked_pos[p] else f"[C:{init_toks[i]}]" if corrupt_pos[p] else init_toks[i]
                for i, p in enumerate(range(start, end)))
            out_str = " ".join(
                f"[OK:{out_toks[i]}]" if (x_out[pos] == x_orig[pos]).item() else f"[X:{out_toks[i]}]"
                for i, pos in enumerate(range(start, end)))
            print("ORIG:", " ".join(orig_toks))
            print("IN:  ", init_str)
            print("OUT: ", out_str)
            print()


def visualize_atomic(tok, x_orig, x_init, x_out, target_corrupt_pos, scenario,
                     mask_idx, content_len, chunk=50):
    """Visualize atomic A/B scenario inputs and outputs.

    INPUT legend:
      [T:tok]  — target corrupt (the 25% we evaluate)   bold red
      [F:tok]  — filler corrupt (scenario B extra)       red
      [M]      — filler masked  (scenario A)             yellow
      tok      — clean                                   default
    OUTPUT legend:
      tok      — correct                                 green
      [X:tok]  — wrong                                   red
      (T) / (F) suffix indicates the position type so errors are easy to attribute.
    """
    try:
        from rich.console import Console
        from rich.text import Text
        console = Console()
        use_rich = True
    except ImportError:
        use_rich = False

    display_len = content_len
    content_range = torch.zeros(x_orig.shape[0], dtype=torch.bool, device=x_orig.device)
    content_range[:content_len] = True
    if scenario == 'A':
        filler_pos = (x_init == mask_idx) & content_range
    else:
        filler_pos = ~target_corrupt_pos & (x_init != x_orig) & content_range

    for start in range(0, display_len, chunk):
        end = min(start + chunk, display_len)
        orig_toks = [tok.decode([t]) for t in x_orig[start:end].tolist()]
        init_toks = [tok.decode([t]) for t in x_init[start:end].tolist()]
        out_toks  = [tok.decode([t]) for t in x_out[start:end].tolist()]

        if use_rich:
            orig_line = Text("ORIGINAL: ")
            init_line = Text("INPUT:    ")
            out_line  = Text("OUTPUT:   ")
            for i, pos in enumerate(range(start, end)):
                orig_line.append(orig_toks[i] + " ")
                is_target  = target_corrupt_pos[pos].item()
                is_filler_mask   = (scenario == 'A') and (x_init[pos] == mask_idx).item()
                is_filler_corrupt = (scenario == 'B') and filler_pos[pos].item()
                if is_target:
                    init_line.append(f"[T:{init_toks[i]}] ", style="bold red")
                elif is_filler_mask:
                    init_line.append("[M] ", style="yellow")
                elif is_filler_corrupt:
                    init_line.append(f"[F:{init_toks[i]}] ", style="red")
                else:
                    init_line.append(init_toks[i] + " ")
                correct = (x_out[pos] == x_orig[pos]).item()
                tag = "(T)" if is_target else ("(F)" if is_filler_corrupt else "")
                if correct and (is_target or is_filler_corrupt or is_filler_mask):
                    out_line.append(out_toks[i] + tag + " ", style="green")
                elif not correct:
                    out_line.append(f"[X:{out_toks[i]}]{tag} ", style="bold red")
                else:
                    out_line.append(out_toks[i] + " ")
            console.print(orig_line)
            console.print(init_line)
            console.print(out_line)
            console.print()
        else:
            def _in(i, pos):
                if target_corrupt_pos[pos].item():
                    return f"[T:{init_toks[i]}]"
                if scenario == 'A' and (x_init[pos] == mask_idx).item():
                    return "[M]"
                if scenario == 'B' and filler_pos[pos].item():
                    return f"[F:{init_toks[i]}]"
                return init_toks[i]
            def _out(i, pos):
                correct = (x_out[pos] == x_orig[pos]).item()
                tag = "(T)" if target_corrupt_pos[pos].item() else (
                    "(F)" if (scenario == 'B' and filler_pos[pos].item()) else "")
                return out_toks[i] + tag if correct else f"[X:{out_toks[i]}]{tag}"
            print("ORIG:", " ".join(orig_toks))
            print("IN:  ", " ".join(_in(i, p) for i, p in enumerate(range(start, end))))
            print("OUT: ", " ".join(_out(i, p) for i, p in enumerate(range(start, end))))
            print()


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='MDLM reconstruction evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--model', choices=list(CHECKPOINTS.keys()), required=True,
                   help='Which model checkpoint to load')
    p.add_argument('--sampler', choices=RECON_SAMPLERS, default='refine',
                   help='Sampling method for reconstruction')
    p.add_argument('--input_file', required=True)
    p.add_argument('--n_samples', type=int, default=10)
    p.add_argument('--atomic_scenario', choices=['A', 'B'], default=None,
                   help=(
                       'Atomic comparison mode (uses --sampler refine implicitly). '
                       'A: 50%% clean / 25%% corrupt / 25%% masked (UUUUCCCCMMMM). '
                       'B: 50%% clean / 25%% corrupt / 25%% extra-corrupt (UUUUCCCCCCCC). '
                       'Evaluates correction of the target 25%% corrupt positions only.'))
    p.add_argument('--mask_frac', type=str, default='0.5',
                   help='Fraction of content to mask, or "random" for per-sample random values')
    p.add_argument('--corrupt_frac', type=str, default='0.15',
                   help='Fraction of unmasked content to corrupt, or "random" for per-sample random values')
    p.add_argument('--steps', type=int, default=128)
    p.add_argument('--verbose', action='store_true', default=True)
    p.add_argument('--show_intermediate', action='store_true', default=False)
    p.add_argument('--show_frequency', type=int, default=1,
                   help='Show intermediate outputs every N steps (1=every step)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_correct_per_step', type=int, default=10,
                   help='Max corrections per DDPM step (used by refine sampler)')
    p.add_argument('--proseco_budget', type=int, default=3,
                   help='S: corrector forward passes per correction round (proseco only)')
    p.add_argument('--proseco_freq', type=int, default=1,
                   help='omega: run corrector every omega steps (proseco only)')
    p.add_argument('--corrector_sampling', choices=['argmax', 'topk'], default='argmax',
                   help='Token selection method in corrector phase (topk uses k=100)')
    p.add_argument('--eval_batch_size', type=int, default=1,
                   help='Batch size for reconstruction inference')
    p.add_argument('--save_gen_dir', type=str, default=None,
                   help='Directory to save reconstructed output text as JSONL')
    p.add_argument('--save_res_dir', type=str, default=None,
                   help='Directory to save evaluation results as JSON')
    return p.parse_args()


# ── Random frac helpers ───────────────────────────────────────────────────────

# Range for random mask/corrupt fractions (easily adjustable)
RANDOM_MASK_FRAC_RANGE = (0.1, 0.9)
RANDOM_CORRUPT_FRAC_RANGE = (0.1, 0.9)


def parse_frac_arg(value: str):
    """Parse a frac arg: returns a float if numeric, or 'random' if random."""
    if value.strip().lower() == 'random':
        return 'random'
    return float(value)


def sample_frac(frac_val, rng, frac_range):
    """Return a float frac: either the fixed value or a random sample."""
    if frac_val == 'random':
        lo, hi = frac_range
        return lo + (hi - lo) * rng.random()
    return frac_val


# ── Prepare a single sample (masking + corruption) ───────────────────────────

def prepare_sample(text, tok, L, mask_idx, vocab_size, mask_frac_val, corrupt_frac_val,
                   rng, device):
    """Tokenize, mask, and corrupt a single text sample.

    Returns: x_orig, x_init, masked_pos, corrupt_pos, content_len, used_mask_frac, used_corrupt_frac
    """
    mf = sample_frac(mask_frac_val, rng, RANDOM_MASK_FRAC_RANGE)
    cf = sample_frac(corrupt_frac_val, rng, RANDOM_CORRUPT_FRAC_RANGE)

    ids = tok.encode(text, add_special_tokens=False)
    ids = ids[:L] + [tok.eos_token_id or 0] * max(0, L - len(ids))
    x_orig = torch.tensor(ids[:L], dtype=torch.long, device=device)

    eos = tok.eos_token_id or 0
    content_len = int((x_orig != eos).sum().item())
    n_mask = int(mf * content_len)

    # Use a torch Generator seeded from the Python rng for reproducibility
    torch_seed = rng.randint(0, 2**63 - 1)
    g = torch.Generator(device=device)
    g.manual_seed(torch_seed)

    perm = torch.randperm(content_len, device=device, generator=g)
    masked_pos = torch.zeros(L, dtype=torch.bool, device=device)
    masked_pos[perm[:n_mask]] = True

    x_masked = x_orig.clone()
    x_masked[masked_pos] = mask_idx

    # Corrupt unmasked content positions
    unmasked_idx = (~masked_pos).nonzero(as_tuple=True)[0]
    unmasked_idx = unmasked_idx[unmasked_idx < content_len]
    n_corrupt = int(cf * len(unmasked_idx))
    corrupt_perm = torch.randperm(len(unmasked_idx), device=device, generator=g)[:n_corrupt]
    corrupt_positions_idx = unmasked_idx[corrupt_perm]
    corrupt_pos = torch.zeros(L, dtype=torch.bool, device=device)
    corrupt_pos[corrupt_positions_idx] = True

    x_init = x_masked.clone()
    rand_tokens = torch.randint(0, vocab_size - 1, (n_corrupt,), device=device, generator=g)
    rand_tokens[rand_tokens >= mask_idx] += 1
    x_init[corrupt_positions_idx] = rand_tokens

    return x_orig, x_init, masked_pos, corrupt_pos, content_len, mf, cf


def prepare_atomic_sample(text, tok, L, mask_idx, vocab_size, scenario, seed, device,
                          corrupt_frac=0.25, filler_frac=0.25):
    """Prepare a sample for the atomic A/B comparison experiment.

    corrupt_frac  — fraction of content to make target corrupt (--corrupt_frac)
    filler_frac   — fraction of content for filler noise (--mask_frac):
                    masked in scenario A, extra-corrupt in scenario B
    Remainder     — clean (unmodified)

    Returns:
        x_orig          — original token ids  [L]
        x_init          — model input         [L]
        target_corrupt_pos — bool mask [L], the target corrupt positions
        content_len     — number of real (non-EOS) tokens
    """
    ids = tok.encode(text, add_special_tokens=False)
    ids = ids[:L] + [tok.eos_token_id or 0] * max(0, L - len(ids))
    x_orig = torch.tensor(ids[:L], dtype=torch.long, device=device)

    eos = tok.eos_token_id or 0
    content_len = int((x_orig != eos).sum().item())

    g = torch.Generator(device=device)
    g.manual_seed(seed)

    perm = torch.randperm(content_len, device=device, generator=g)
    n_corrupt = int(corrupt_frac * content_len)
    n_filler  = int(filler_frac  * content_len)
    corrupt_idx = perm[:n_corrupt]
    filler_idx  = perm[n_corrupt: n_corrupt + n_filler]

    target_corrupt_pos = torch.zeros(L, dtype=torch.bool, device=device)
    target_corrupt_pos[corrupt_idx] = True

    x_init = x_orig.clone()

    # Corrupt target positions
    def _rand_tokens(n):
        toks = torch.randint(0, vocab_size - 1, (n,), device=device, generator=g)
        toks[toks >= mask_idx] += 1
        return toks

    x_init[corrupt_idx] = _rand_tokens(corrupt_idx.numel())

    if scenario == 'A':
        x_init[filler_idx] = mask_idx
    else:  # scenario B
        x_init[filler_idx] = _rand_tokens(filler_idx.numel())

    return x_orig, x_init, target_corrupt_pos, content_len


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import random
    import statistics
    from tqdm import tqdm

    args = parse_args()
    torch.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Parse frac args
    mask_frac_val = parse_frac_arg(args.mask_frac)
    corrupt_frac_val = parse_frac_arg(args.corrupt_frac)

    # Auto-disable verbose/show_intermediate for batch mode
    batch_size = args.eval_batch_size
    if batch_size > 1:
        if args.verbose:
            print('Note: --verbose disabled (eval_batch_size > 1)')
            args.verbose = False
        if args.show_intermediate:
            print('Note: --show_intermediate disabled (eval_batch_size > 1)')
            args.show_intermediate = False

    # Create save dirs if specified
    if args.save_gen_dir:
        os.makedirs(args.save_gen_dir, exist_ok=True)
    if args.save_res_dir:
        os.makedirs(args.save_res_dir, exist_ok=True)

    print(f'[Loading model: {args.model}]')
    model = load_mdlm(args.model, device)
    L = model.config.model.length
    mask_idx = model.mask_index
    vocab_size = model.vocab_size
    tok = model.tokenizer
    print(f'Loaded model  L={L}  vocab={vocab_size}  mask_index={mask_idx}')
    print(f'Sampler: {args.sampler}  steps={args.steps}  batch_size={batch_size}')
    print(f'mask_frac={args.mask_frac}  corrupt_frac={args.corrupt_frac}')

    with open(args.input_file) as f:
        texts = [json.loads(l) for l in f][:args.n_samples]

    # Seeded RNG for reproducible random fracs
    rng = random.Random(args.seed)

    # ── Atomic A/B experiment ─────────────────────────────────────────────────
    if args.atomic_scenario is not None:
        filler_type = "masked" if args.atomic_scenario == "A" else "extra-corrupt"
        print(f'\n=== Atomic scenario {args.atomic_scenario} ===')
        print(f'Split: corrupt_frac={args.corrupt_frac} (target C) / '
              f'mask_frac={args.mask_frac} ({filler_type}) / '
              f'remainder clean')
        all_x_orig_ab, all_x_init_ab, all_target_pos, all_content_lens_ab = [], [], [], []
        for sample_idx, text in enumerate(texts):
            x_orig, x_init, target_corrupt_pos, content_len = prepare_atomic_sample(
                text, tok, L, mask_idx, vocab_size, args.atomic_scenario,
                args.seed + sample_idx, device,
                corrupt_frac=float(args.corrupt_frac),
                filler_frac=float(args.mask_frac))
            all_x_orig_ab.append(x_orig)
            all_x_init_ab.append(x_init)
            all_target_pos.append(target_corrupt_pos)
            all_content_lens_ab.append(content_len)

        agg_ab = {
            'corr_rate': [],       # fraction of target corrupt recovered correctly
            'n_target':  [],       # total target corrupt tokens
            'n_flipped': [],       # target corrupt tokens the model changed
            'flip_prec': [],       # of all content flips, fraction that hit target corrupt
            'ce_loss':   [],       # cross-entropy loss over target corrupt positions
        }
        for idx in range(len(texts)):
            x_orig = all_x_orig_ab[idx]
            x_init_1 = all_x_init_ab[idx]
            content_mask = (x_orig != (tok.eos_token_id or 0)).unsqueeze(0)
            x_out, _, _, log_probs = reconstruct_ddpm_atomic(
                model, x_init_1.unsqueeze(0), steps=None, device=device,
                content_mask=content_mask)
            x_out = x_out[0]
            log_probs = log_probs[0]  # [L, vocab]
            tp = all_target_pos[idx]

            # Correction rate on target corrupt positions
            n_target = tp.sum().item()
            corr_rate = (x_out[tp] == x_orig[tp]).float().mean().item() if n_target > 0 else float('nan')

            # How many target corrupt tokens did the model actually change?
            n_flipped = (tp & (x_out != x_init_1)).sum().item()

            # Of all flips on non-filler positions (clean + target corrupt), what fraction
            # landed on target corrupt? Filler (masked in A, extra-corrupt in B) is excluded
            # so it doesn't inflate the denominator.
            content_range = content_mask[0]
            if args.atomic_scenario == 'A':
                filler_pos = (x_init_1 == mask_idx) & content_range
            else:
                filler_pos = ~tp & (x_init_1 != x_orig) & content_range
            non_filler_flips = (content_range & ~filler_pos & (x_out != x_init_1)).sum().item()
            flip_prec = n_flipped / non_filler_flips if non_filler_flips > 0 else float('nan')

            # Cross-entropy loss over target corrupt positions
            if n_target > 0:
                ce_loss = -log_probs[tp, x_orig[tp]].mean().item()
            else:
                ce_loss = float('nan')

            agg_ab['corr_rate'].append(corr_rate)
            agg_ab['n_target'].append(n_target)
            agg_ab['n_flipped'].append(n_flipped)
            agg_ab['flip_prec'].append(flip_prec)
            agg_ab['ce_loss'].append(ce_loss)

            if args.verbose:
                print(f'\n-- Sample {idx+1}/{len(texts)} --')
                print(f'  Target corrupt tokens : {n_target}')
                print(f'  Target tokens flipped : {n_flipped} / {n_target}  '
                      f'({n_flipped/n_target:.1%} of targets)' if n_target > 0 else '')
                print(f'  Correction rate       : {corr_rate:.1%}  '
                      f'(flipped to correct / total target)')
                print(f'  Flip precision        : {flip_prec:.1%}  '
                      f'(target flips / non-filler flips = {n_flipped} / {non_filler_flips})')
                print(f'  CE loss (target)      : {ce_loss:.4f}')
                visualize_atomic(tok, x_orig, x_init_1, x_out, tp,
                                 args.atomic_scenario, mask_idx,
                                 all_content_lens_ab[idx])

        def _avg(lst):
            valid = [v for v in lst if v == v]
            return sum(valid) / len(valid) if valid else float('nan')

        print(f'\n=== Scenario {args.atomic_scenario} summary (n={len(texts)}) ===')
        print(f'  Correction rate    : {_avg(agg_ab["corr_rate"]):.1%}  '
              f'(target corrupt recovered correctly)')
        print(f'  Avg targets flipped: {_avg(agg_ab["n_flipped"]):.1f} / '
              f'{_avg(agg_ab["n_target"]):.1f}  (model changed the token)')
        print(f'  Flip precision     : {_avg(agg_ab["flip_prec"]):.1%}  '
              f'(of flips on clean+target positions, fraction on target corrupt)')
        print(f'  CE loss (target)   : {_avg(agg_ab["ce_loss"]):.4f}  '
              f'(cross-entropy over target corrupt positions)')
        return

    # Prepare all samples
    all_x_orig, all_x_init = [], []
    all_masked_pos, all_corrupt_pos = [], []
    all_content_lens = []
    all_mask_fracs, all_corrupt_fracs = [], []

    for text in texts:
        x_orig, x_init, masked_pos, corrupt_pos, content_len, mf, cf = prepare_sample(
            text, tok, L, mask_idx, vocab_size, mask_frac_val, corrupt_frac_val,
            rng, device)
        all_x_orig.append(x_orig)
        all_x_init.append(x_init)
        all_masked_pos.append(masked_pos)
        all_corrupt_pos.append(corrupt_pos)
        all_content_lens.append(content_len)
        all_mask_fracs.append(mf)
        all_corrupt_fracs.append(cf)

    # Run reconstruction in batches
    all_x_out = []
    all_flip_stats = []  # per-sample: dict with avg, std, min, max
    n_total = len(texts)
    gen_texts = []

    for batch_start in tqdm(range(0, n_total, batch_size), desc='Reconstructing', disable=(n_total <= batch_size)):
        batch_end = min(batch_start + batch_size, n_total)
        bs = batch_end - batch_start

        x_init_batch = torch.stack(all_x_init[batch_start:batch_end])  # [bs, L]
        x_orig_batch = torch.stack(all_x_orig[batch_start:batch_end])
        eos = tok.eos_token_id or 0
        content_mask = (x_orig_batch != eos)  # [bs, L]

        x_out_batch, step_data, flip_counts = run_reconstruction(
            args.sampler, model, x_init_batch, args.steps, device,
            collect_steps=args.show_intermediate,
            content_mask=content_mask,
            n_correct_per_step=args.n_correct_per_step,
            proseco_budget=args.proseco_budget,
            proseco_freq=args.proseco_freq,
            corrector_sampling=args.corrector_sampling,
            show_frequency=args.show_frequency,
        )

        # Compute per-sample flip stats: flip_counts is list of [bs] tensors
        if flip_counts:
            fc = torch.stack(flip_counts).float()  # [n_steps, bs]
            for i in range(bs):
                col = fc[:, i]
                all_flip_stats.append({
                    'avg': col.mean().item(),
                    'std': col.std().item() if col.numel() > 1 else 0.0,
                    'min': col.min().item(),
                    'max': col.max().item(),
                })
        else:
            for i in range(bs):
                all_flip_stats.append({'avg': 0.0, 'std': 0.0, 'min': 0, 'max': 0})

        for i in range(bs):
            all_x_out.append(x_out_batch[i])
            gen_texts.append(tok.decode(x_out_batch[i].tolist()))

        # Visualization for batch_size==1 only
        if bs == 1 and (args.show_intermediate or args.verbose):
            idx = batch_start
            x_orig = all_x_orig[idx]
            x_init = all_x_init[idx]
            x_out = x_out_batch[0]
            masked_pos = all_masked_pos[idx]
            corrupt_pos = all_corrupt_pos[idx]
            content_len = all_content_lens[idx]
            display_len = content_len

            if args.show_intermediate:
                for (step_num, xb, cand, xa, top_corr, n_unmask, n_corr) in step_data:
                    visualize_step(tok, step_num + 1, xb, cand, xa, mask_idx, display_len,
                                   top_corrections=top_corr,
                                   n_to_unmask=n_unmask, n_to_correct=n_corr)

            if args.verbose:
                visualize(tok, x_orig[:display_len], x_init[:display_len], x_out[:display_len],
                          masked_pos[:display_len], corrupt_pos[:display_len], mask_idx)

    # Compute per-sample metrics
    agg = {
        'masked_rec': [], 'corrupt_corr': [], 'clean_pres': [],
        'overall': [], 'corrupt_revised': [], 'edit_precision': [],
    }

    def fmt(v): return f'{v:.1%}' if v == v else 'N/A'

    for idx in range(n_total):
        x_orig = all_x_orig[idx]
        x_init = all_x_init[idx]
        x_out = all_x_out[idx]
        masked_pos = all_masked_pos[idx]
        corrupt_pos = all_corrupt_pos[idx]
        clean_pos = ~masked_pos & ~corrupt_pos

        def rate(mask):
            return (x_out[mask] == x_orig[mask]).float().mean().item() if mask.sum() > 0 else float('nan')

        mr = rate(masked_pos)
        cr = rate(corrupt_pos)
        cp = rate(clean_pos)
        ov = (x_out == x_orig).float().mean().item()
        cv = (x_out[corrupt_pos] != x_init[corrupt_pos]).float().mean().item() if corrupt_pos.sum() > 0 else float('nan')

        # Edit precision: of unmasked tokens changed by model, what fraction were corrupt
        unmasked = ~masked_pos
        changed_unmasked = unmasked & (x_out != x_init)
        n_changed = changed_unmasked.sum().item()
        if n_changed > 0:
            ep = (changed_unmasked & corrupt_pos).sum().item() / n_changed
        else:
            ep = float('nan')

        agg['masked_rec'].append(mr)
        agg['corrupt_corr'].append(cr)
        agg['clean_pres'].append(cp)
        agg['overall'].append(ov)
        agg['corrupt_revised'].append(cv)
        agg['edit_precision'].append(ep)

        if batch_size == 1:
            fs = all_flip_stats[idx]
            print(f'\n-- Sample {idx+1}/{n_total} ---------------------------------')
            print(f'  Masked recovery:      {fmt(mr)}  |  Corrupt correction: {fmt(cr)}')
            print(f'  Corrupt revised:      {fmt(cv)}  |  Clean preservation: {fmt(cp)}')
            print(f'  Edit precision:       {fmt(ep)}  |  Overall accuracy:   {fmt(ov)}')
            print(f'  Unmasked flips/step:  avg={fs["avg"]:.2f}  std={fs["std"]:.2f}  min={fs["min"]:.0f}  max={fs["max"]:.0f}')

    # Aggregate summary
    def avg(lst):
        valid = [x for x in lst if x == x]
        return statistics.mean(valid) if valid else float('nan')

    mr_avg = avg(agg['masked_rec'])
    cr_avg = avg(agg['corrupt_corr'])
    cp_avg = avg(agg['clean_pres'])
    ov_avg = avg(agg['overall'])
    cv_avg = avg(agg['corrupt_revised'])
    ep_avg = avg(agg['edit_precision'])

    # Aggregate flip stats across samples
    flip_avgs = [fs['avg'] for fs in all_flip_stats]
    flip_stds = [fs['std'] for fs in all_flip_stats]
    flip_mins = [fs['min'] for fs in all_flip_stats]
    flip_maxs = [fs['max'] for fs in all_flip_stats]
    flip_avg_of_avg = avg(flip_avgs)
    flip_avg_of_std = avg(flip_stds)
    flip_global_min = min(flip_mins) if flip_mins else 0
    flip_global_max = max(flip_maxs) if flip_maxs else 0

    def fmt_avg(v): return f'{v:.2%}' if v == v else 'N/A'
    print('\n' + '=' * 55)
    print('  AGGREGATE SUMMARY')
    print('=' * 55)
    print(f'  Masked recovery (masked correct / masked total):          {fmt_avg(mr_avg)}')
    print(f'  Corrupt correction (corrupt correct / corrupt total):     {fmt_avg(cr_avg)}')
    print(f'  Corrupt revision (corrupt changed / corrupt total):       {fmt_avg(cv_avg)}')
    print(f'  Edit precision (corrupt changed / unmasked changed):      {fmt_avg(ep_avg)}')
    print(f'  Clean preservation (clean unchanged / clean total):       {fmt_avg(cp_avg)}')
    print(f'  Overall accuracy (all correct / all total):               {fmt_avg(ov_avg)}')
    print(f'  Unmasked flips/step:  avg={flip_avg_of_avg:.2f}  std={flip_avg_of_std:.2f}  min={flip_global_min:.0f}  max={flip_global_max:.0f}')
    print(f'  Model={args.model}  Sampler={args.sampler}  Steps={args.steps}')
    print(f'  n_correct_per_step={args.n_correct_per_step}  '
          f'mask_frac={args.mask_frac}  corrupt_frac={args.corrupt_frac}')
    print(f'  n_samples={n_total}  eval_batch_size={batch_size}')
    if 'proseco' in args.sampler:
        print(f'  proseco_budget={args.proseco_budget}  proseco_freq={args.proseco_freq}  '
              f'corrector_sampling={args.corrector_sampling}')

    # Interpretation hint
    def safe_lt(v, threshold): return v == v and v < threshold
    def safe_gt(v, threshold): return v == v and v > threshold
    if safe_lt(cp_avg, 0.9):
        hint = "WARNING: low clean preservation -- sampler may be mangling good tokens"
    elif safe_gt(cv_avg, 0.5) and safe_lt(cr_avg, 0.3):
        hint = "INFO: model detects corruptions (high revision rate) but fixes them incorrectly"
    elif safe_lt(cv_avg, 0.2) and safe_lt(cr_avg, 0.2):
        hint = "INFO: model does not detect/revise corruptions -- try a correction-enabled sampler"
    elif safe_lt(cr_avg, 0.3) and safe_gt(mr_avg, 0.5):
        hint = "INFO: high masked recovery but low corrupt correction -- model may not use context to fix errors"
    elif safe_lt(mr_avg, 0.3):
        hint = "INFO: low masked recovery -- model may not be reconstructing well from context"
    else:
        hint = "OK: rates look reasonable"
    print(f'\n  Hint: {hint}')
    print('=' * 55)

    # Save generated outputs
    config_key = f'{args.model}_{args.sampler}_steps{args.steps}'
    if args.save_gen_dir:
        fname = f'{config_key}.jsonl'
        fpath = os.path.join(args.save_gen_dir, fname)
        with open(fpath, 'w') as f:
            for t in gen_texts:
                f.write(json.dumps(t) + '\n')
        print(f'\nSaved {len(gen_texts)} outputs → {fpath}')

    # Save results
    if args.save_res_dir:
        results = {
            'config': {
                'model': args.model,
                'sampler': args.sampler,
                'steps': args.steps,
                'mask_frac': args.mask_frac,
                'corrupt_frac': args.corrupt_frac,
                'n_samples': n_total,
                'eval_batch_size': batch_size,
                'n_correct_per_step': args.n_correct_per_step,
                'seed': args.seed,
            },
            'aggregate': {
                'masked_rec': mr_avg,
                'corrupt_corr': cr_avg,
                'corrupt_revised': cv_avg,
                'edit_precision': ep_avg,
                'clean_pres': cp_avg,
                'overall': ov_avg,
                'unmasked_flips_per_step': {
                    'avg': flip_avg_of_avg,
                    'std': flip_avg_of_std,
                    'min': flip_global_min,
                    'max': flip_global_max,
                },
            },
            'per_sample': {
                'masked_rec': agg['masked_rec'],
                'corrupt_corr': agg['corrupt_corr'],
                'corrupt_revised': agg['corrupt_revised'],
                'edit_precision': agg['edit_precision'],
                'clean_pres': agg['clean_pres'],
                'overall': agg['overall'],
                'unmasked_flips': [fs for fs in all_flip_stats],
            },
        }
        fname = f'{config_key}.json'
        fpath = os.path.join(args.save_res_dir, fname)
        with open(fpath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Saved results → {fpath}')


if __name__ == '__main__':
    main()
