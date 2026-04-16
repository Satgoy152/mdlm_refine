#!/usr/bin/env python3
"""eval_reconstruct.py — diagnostic: can the MDM reconstruct masked/corrupted responses?

Mirrors mdlm/eval_reconstruct.py but for the diffusion-vs-ar repo. Loads a
trained MDM via `load_model_and_tokenizer`, reads a jsonl task file with
input/output fields (the standard diffusion-vs-ar dataset format), and
evaluates reconstruction after masking + corrupting a fraction of the
response tokens.

Sampling methods (vanilla, refine, proseco) mirror the structure of
`CustomDiffusionTrainer.generate_samples_*` in
src/llmtuner/tuner/mdm/trainer.py, adapted to start from a partially
masked+corrupted state instead of a fully-masked one.

Example:
  python eval_reconstruct.py \\
    --model_name_or_path model_config_tiny \\
    --checkpoint_dir /nfs/.../sudoku/refine \\
    --dataset sudoku_test \\
    --cutoff_len 164 \\
    --sampler refine --diffusion_steps 4 \\
    --mask_frac 0.25 --corrupt_frac 0.25 \\
    --n_samples 100
"""
import argparse
import json
import os
import sys
import random
import statistics

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from llmtuner.tuner.core import load_model_and_tokenizer
from llmtuner.hparams import ModelArguments, DiffusionArguments, FinetuningArguments


# ── Sample preparation ───────────────────────────────────────────────────────

def prepare_sample(prompt_text, response_text, tok, L, rng, mask_frac, corrupt_frac, device):
    """Build (x_orig, x_init, src_mask, content_mask, masked_pos, corrupt_pos).

    src_mask   : True for prompt + sep + pad (positions we never touch).
    content_mask : True for response tokens + eos (~src_mask & within L).
    masked_pos : subset of content_mask, set to [MASK] in x_init.
    corrupt_pos: subset of content_mask \ masked_pos, set to random tokens.
    """
    mask_id = tok.mask_token_id
    sep_id = tok.sep_token_id
    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id
    vocab_size = tok.vocab_size

    src_ids = tok.encode(prompt_text)
    tgt_ids = tok.encode(response_text)
    if L is not None:
        tgt_ids = tgt_ids[:max(0, L - 2)]
        src_ids = src_ids[-(L - 2 - len(tgt_ids)):]

    input_ids = src_ids + [sep_id] + tgt_ids + [eos_id]
    src_len = len(src_ids) + 1  # src + sep
    tgt_start = src_len
    tgt_end = tgt_start + len(tgt_ids) + 1  # include eos

    # pad to L
    if len(input_ids) < L:
        input_ids = input_ids + [pad_id] * (L - len(input_ids))
    else:
        input_ids = input_ids[:L]
        tgt_end = min(tgt_end, L)

    x_orig = torch.tensor(input_ids, dtype=torch.long, device=device)

    src_mask = torch.ones(L, dtype=torch.bool, device=device)
    content_mask = torch.zeros(L, dtype=torch.bool, device=device)
    content_mask[tgt_start:tgt_end] = True
    src_mask[tgt_start:tgt_end] = False  # only response tokens are "target"

    content_len = int(content_mask.sum().item())
    n_mask = int(mask_frac * content_len)

    torch_seed = rng.randint(0, 2**63 - 1)
    g = torch.Generator(device=device)
    g.manual_seed(torch_seed)

    content_idx = content_mask.nonzero(as_tuple=True)[0]
    perm = content_idx[torch.randperm(content_len, device=device, generator=g)]
    mask_positions = perm[:n_mask]

    masked_pos = torch.zeros(L, dtype=torch.bool, device=device)
    masked_pos[mask_positions] = True

    unmasked_content = perm[n_mask:]
    n_corrupt = int(corrupt_frac * unmasked_content.numel())
    corrupt_positions = unmasked_content[torch.randperm(unmasked_content.numel(), device=device, generator=g)[:n_corrupt]]

    corrupt_pos = torch.zeros(L, dtype=torch.bool, device=device)
    corrupt_pos[corrupt_positions] = True

    x_init = x_orig.clone()
    x_init[masked_pos] = mask_id
    # Corrupt: sample random tokens in [0, vocab_size) but skip mask_id
    rand_tokens = torch.randint(0, vocab_size - 1, (n_corrupt,), device=device, generator=g)
    rand_tokens[rand_tokens >= mask_id] += 1
    x_init[corrupt_positions] = rand_tokens

    return x_orig, x_init, src_mask, content_mask, masked_pos, corrupt_pos


# ── Reconstruction samplers (mirror trainer.py but start from x_init) ────────

def _forward_logits(model, xt, t_tensor, attention_mask):
    """Replicates trainer.py's forward + right-shift."""
    logits = model(xt, t_tensor, attention_mask=attention_mask)
    logits = torch.cat([logits[:, 0:1], logits[:, :-1]], dim=1)
    return logits


def _topk_masking(scores, cutoff_len, stochastic=False, temp=1.0):
    if stochastic:
        gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-8) + 1e-8)
        _scores = scores + temp * gumbel
    else:
        _scores = scores
    sorted_index = _scores.sort(-1)[0]
    cutoff = sorted_index.gather(dim=-1, index=cutoff_len)
    return _scores < cutoff


def _topk_decoding(x0, x0_scores, decoding_strategy, init_maskable_mask, t, max_step, mask_id):
    topk_mode, schedule = decoding_strategy.split("-")
    if schedule == "linear":
        rate = t / max_step
    elif schedule == "cosine":
        rate = np.cos((max_step - t) / max_step * np.pi * 0.5)
    else:
        raise NotImplementedError
    cutoff_len = (init_maskable_mask.sum(1, keepdim=True) * rate).long()
    _scores_for_topk = x0_scores.masked_fill(~init_maskable_mask, 1000.0)
    if topk_mode.startswith("stochastic"):
        noise_scale = float(topk_mode.replace("stochastic", ""))
        lowest_k = _topk_masking(_scores_for_topk, cutoff_len, stochastic=True, temp=noise_scale * rate)
    elif topk_mode == "deterministic":
        lowest_k = _topk_masking(_scores_for_topk, cutoff_len, stochastic=False)
    else:
        raise NotImplementedError
    return x0.masked_fill(lowest_k, mask_id)


def _verbose_print(tag, tok, xt, content_mask):
    """Print just the content window of batch element 0."""
    idx = content_mask[0].nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return
    start, end = idx.min().item(), idx.max().item() + 1
    piece = xt[0, start:end].tolist()
    print(f"{tag}", tok.decode(piece))


@torch.no_grad()
def reconstruct(sampler, model, x_init, src_mask, content_mask, *, args, tok, verbose=False):
    """Dispatch to the chosen reconstruction sampler."""
    if sampler == 'vanilla':
        return _recon_vanilla(model, x_init, src_mask, content_mask, args=args, tok=tok, verbose=verbose)
    elif sampler == 'refine':
        return _recon_refine(model, x_init, src_mask, content_mask, args=args, tok=tok, verbose=verbose)
    elif sampler == 'proseco':
        return _recon_proseco(model, x_init, src_mask, content_mask, args=args, tok=tok, verbose=verbose)
    else:
        raise ValueError(f'Unknown sampler: {sampler}')


@torch.no_grad()
def _recon_vanilla(model, x_init, src_mask, content_mask, *, args, tok, verbose):
    """Vanilla: only originally-masked positions get filled; corrupt/clean untouched."""
    mask_id = tok.mask_token_id
    T = args.diffusion_steps
    device = x_init.device
    bs = x_init.size(0)

    xt = x_init.clone()
    attention_mask = torch.ones_like(xt)

    # Only currently-masked (within target region) positions are maskable.
    init_maskable_mask = maskable_mask = (xt == mask_id) & (~src_mask)

    for t in range(T - 1, -1, -1):
        if verbose:
            _verbose_print(f"t={t+1}(in):", tok, xt, content_mask)

        t_tensor = torch.full((bs,), t, device=device)
        logits = _forward_logits(model, xt, t_tensor, attention_mask)
        scores = torch.log_softmax(logits, dim=-1)
        scores[:, :, tok.vocab_size:] = -1000
        x0_scores, x0 = scores.max(-1)
        x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])

        if verbose:
            _verbose_print(f"t={t+1}(out):", tok, x0, content_mask)

        if t > 0:
            if args.topk_decoding:
                xt = _topk_decoding(x0, x0_scores, args.decoding_strategy,
                                    init_maskable_mask, t, T, mask_id)
            else:
                unmask_prob = 1 / (t + 1)
                mask_to_x0 = torch.rand(xt.shape, device=device) < unmask_prob
                mask_to_x0 = mask_to_x0 & maskable_mask
                xt[mask_to_x0] = x0[mask_to_x0]
                maskable_mask = maskable_mask & ~mask_to_x0
        else:
            xt = x0

    return xt


@torch.no_grad()
def _recon_refine(model, x_init, src_mask, content_mask, *, args, tok, verbose):
    """Refine: vanilla unmasking + per-step correction of already-unmasked tokens.

    Correction targets "previously unmasked" positions inside the target region
    — i.e. clean + corrupt + any positions filled in earlier steps.
    """
    mask_id = tok.mask_token_id
    T = args.diffusion_steps
    device = x_init.device
    bs = x_init.size(0)

    xt = x_init.clone()
    attention_mask = torch.ones_like(xt)

    init_maskable_mask = maskable_mask = (xt == mask_id) & (~src_mask)
    # Everything inside content that is not maskable = already "unmasked" target tokens.
    target_region = (~src_mask)  # includes content + any pad-in-src-mask-region

    n_correct_per_step = args.n_correct_per_step
    correct_mode = args.correct_mode
    correct_threshold = args.correct_threshold

    for t in range(T - 1, -1, -1):
        previously_unmasked = (xt != mask_id) & target_region

        if verbose:
            _verbose_print(f"t={t+1}(in):", tok, xt, content_mask)

        t_tensor = torch.full((bs,), t, device=device)
        logits = _forward_logits(model, xt, t_tensor, attention_mask)
        scores = torch.log_softmax(logits, dim=-1)
        scores[:, :, tok.vocab_size:] = -1000
        x0_scores, x0 = scores.max(-1)
        x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])

        if verbose:
            _verbose_print(f"t={t+1}(out):", tok, x0, content_mask)

        if t > 0:
            if args.topk_decoding:
                xt = _topk_decoding(x0, x0_scores, args.decoding_strategy,
                                    init_maskable_mask, t, T, mask_id)
            else:
                unmask_prob = 1 / (t + 1)
                mask_to_x0 = torch.rand(xt.shape, device=device) < unmask_prob
                mask_to_x0 = mask_to_x0 & maskable_mask
                xt[mask_to_x0] = x0[mask_to_x0]
                maskable_mask = maskable_mask & ~mask_to_x0
        else:
            xt = x0

        # ── Correction phase (reuses logits from this step) ──
        if n_correct_per_step > 0:
            log_probs = scores  # already log_softmax
            det_candidates = log_probs.argmax(dim=-1)
            for b in range(bs):
                unmasked_pos = previously_unmasked[b].nonzero(as_tuple=True)[0]
                if unmasked_pos.numel() == 0:
                    continue
                changed = det_candidates[b][unmasked_pos] != xt[b][unmasked_pos]
                cand_pos = unmasked_pos[changed]
                if cand_pos.numel() == 0:
                    continue
                log_p_new = log_probs[b][cand_pos, det_candidates[b][cand_pos]]
                log_p_cur = log_probs[b][cand_pos, xt[b][cand_pos]]
                log_ratio = log_p_new - log_p_cur

                if correct_mode == 'topk':
                    k = min(n_correct_per_step, cand_pos.numel())
                    _, top_idx = log_ratio.topk(k, largest=True)
                    accept_pos = cand_pos[top_idx]
                elif correct_mode == 'threshold':
                    accept_pos = cand_pos[log_ratio > correct_threshold]
                else:
                    raise ValueError(f'Unknown correct_mode: {correct_mode}')

                xt[b, accept_pos] = det_candidates[b, accept_pos]

    return xt


@torch.no_grad()
def _recon_proseco(model, x_init, src_mask, content_mask, *, args, tok, verbose):
    """ProSeCo: vanilla unmasking with periodic dense-corrector passes."""
    mask_id = tok.mask_token_id
    T = args.diffusion_steps
    omega = args.proseco_freq
    S = args.proseco_budget
    device = x_init.device
    bs = x_init.size(0)

    xt = x_init.clone()
    attention_mask = torch.ones_like(xt)

    init_maskable_mask = maskable_mask = (xt == mask_id) & (~src_mask)
    target_region = (~src_mask)

    actual_steps = max(int(T * omega / (omega + S)), 1)

    for step_i in range(actual_steps):
        t = actual_steps - 1 - step_i
        if verbose:
            _verbose_print(f"t={t+1}(in):", tok, xt, content_mask)

        t_tensor = torch.full((bs,), t, device=device)
        logits = _forward_logits(model, xt, t_tensor, attention_mask)
        scores = torch.log_softmax(logits, dim=-1)
        scores[:, :, tok.vocab_size:] = -1000
        x0_scores, x0 = scores.max(-1)
        x0 = xt.masked_scatter(maskable_mask, x0[maskable_mask])

        if verbose:
            _verbose_print(f"t={t+1}(out):", tok, x0, content_mask)

        # ProSeCo corrector every omega steps
        if (step_i + 1) % omega == 0 and t > 0:
            corrector_x = x0.clone()
            for _s in range(S):
                corr_logits = _forward_logits(model, corrector_x, t_tensor, attention_mask)
                corr_scores = torch.log_softmax(corr_logits, dim=-1)
                corr_scores[:, :, tok.vocab_size:] = -1000
                corr_preds = corr_scores.argmax(dim=-1)
                # Only refresh target-region positions
                corrector_x = xt.masked_scatter(target_region, corr_preds[target_region])

            already_unmasked = (xt != mask_id) & target_region
            xt = torch.where(already_unmasked, corrector_x, xt)
            x0 = corrector_x
            x0_scores = corr_scores.max(-1)[0]

        if t > 0:
            if args.topk_decoding:
                xt = _topk_decoding(x0, x0_scores, args.decoding_strategy,
                                    init_maskable_mask, t, actual_steps, mask_id)
            else:
                unmask_prob = 1 / (t + 1)
                mask_to_x0 = torch.rand(xt.shape, device=device) < unmask_prob
                mask_to_x0 = mask_to_x0 & maskable_mask
                xt[mask_to_x0] = x0[mask_to_x0]
                maskable_mask = maskable_mask & ~mask_to_x0
        else:
            xt = x0

    return xt


# ── Model loading ────────────────────────────────────────────────────────────

def load_mdm(args, device):
    model_args = ModelArguments(
        model_name_or_path=args.model_name_or_path,
        checkpoint_dir=args.checkpoint_dir,
        cache_dir=args.cache_dir,
    )
    finetuning_args = FinetuningArguments(stage='mdm', finetuning_type='full')
    diffusion_args = DiffusionArguments(
        diffusion_steps=args.diffusion_steps,
        decoding_strategy=args.decoding_strategy,
        topk_decoding=args.topk_decoding,
        sampling_method=args.sampler,
        n_correct_per_step=args.n_correct_per_step,
        correct_mode=args.correct_mode,
        correct_threshold=args.correct_threshold,
        proseco_budget=args.proseco_budget,
        proseco_freq=args.proseco_freq,
    )
    model, tok = load_model_and_tokenizer(
        model_args, finetuning_args, is_trainable=False, diffusion_args=diffusion_args)
    model.to(device).eval()
    return model, tok


# ── Data loading ─────────────────────────────────────────────────────────────

def load_task_jsonl(path, n_samples):
    samples = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = obj.get('input') or obj.get('prompt') or ''
            response = obj.get('output') or obj.get('response') or ''
            samples.append((prompt, response))
            if len(samples) >= n_samples:
                break
    return samples


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='MDM reconstruction evaluation (diffusion-vs-ar)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Model
    p.add_argument('--model_name_or_path', required=True,
                   help='Base model config dir (e.g. model_config_tiny)')
    p.add_argument('--checkpoint_dir', required=True,
                   help='Path to trained MDM checkpoint dir (contains pytorch_model.bin)')
    p.add_argument('--cache_dir', default=None)
    # Dataset
    p.add_argument('--input_file', required=True,
                   help='Path to jsonl task file with {input, output} fields')
    p.add_argument('--dataset', default=None,
                   help='Optional dataset tag used in output filenames')
    p.add_argument('--cutoff_len', type=int, required=True)
    p.add_argument('--n_samples', type=int, default=100)
    # Sampling
    p.add_argument('--sampler', choices=['vanilla', 'refine', 'proseco'], default='refine')
    p.add_argument('--diffusion_steps', type=int, default=4)
    p.add_argument('--topk_decoding', action='store_true', default=False)
    p.add_argument('--decoding_strategy', default='stochastic0.5-linear')
    p.add_argument('--n_correct_per_step', type=int, default=5)
    p.add_argument('--correct_mode', choices=['topk', 'threshold'], default='topk')
    p.add_argument('--correct_threshold', type=float, default=0.01)
    p.add_argument('--proseco_budget', type=int, default=1)
    p.add_argument('--proseco_freq', type=int, default=1)
    # Reconstruction
    p.add_argument('--mask_frac', type=float, default=0.25)
    p.add_argument('--corrupt_frac', type=float, default=0.25)
    p.add_argument('--eval_batch_size', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--verbose', action='store_true', default=False,
                   help='Print intermediate decoding steps (batch_size=1 only)')
    p.add_argument('--save_gen_dir', type=str, default=None)
    p.add_argument('--save_res_dir', type=str, default=None)
    return p.parse_args()


# ── Metrics + visualization ──────────────────────────────────────────────────

def visualize(tok, x_orig, x_init, x_out, content_mask, masked_pos, corrupt_pos):
    idx = content_mask.nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        return
    start, end = idx.min().item(), idx.max().item() + 1
    mask_id = tok.mask_token_id

    def row(xs, tag_pos=None):
        out = []
        for i in range(start, end):
            tok_str = tok.decode([xs[i].item()])
            if tag_pos is not None and tag_pos[i]:
                out.append(f'[{tok_str}]')
            elif xs[i].item() == mask_id:
                out.append('[M]')
            else:
                out.append(tok_str)
        return ' '.join(out)

    print('ORIG:', row(x_orig))
    print('IN:  ', row(x_init, masked_pos | corrupt_pos))
    print('OUT: ', row(x_out))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = args.eval_batch_size
    if batch_size > 1 and args.verbose:
        print('Note: --verbose disabled (eval_batch_size > 1)')
        args.verbose = False

    if args.save_gen_dir:
        os.makedirs(args.save_gen_dir, exist_ok=True)
    if args.save_res_dir:
        os.makedirs(args.save_res_dir, exist_ok=True)

    print(f'[Loading model: {args.model_name_or_path}  ckpt={args.checkpoint_dir}]')
    model, tok = load_mdm(args, device)
    L = args.cutoff_len
    print(f'L={L}  vocab={tok.vocab_size}  mask_id={tok.mask_token_id}')
    print(f'Sampler={args.sampler}  steps={args.diffusion_steps}  batch_size={batch_size}')
    print(f'mask_frac={args.mask_frac}  corrupt_frac={args.corrupt_frac}')

    samples = load_task_jsonl(args.input_file, args.n_samples)
    print(f'Loaded {len(samples)} samples from {args.input_file}')

    rng = random.Random(args.seed)

    # Prepare all samples
    all_x_orig, all_x_init = [], []
    all_src_mask, all_content_mask = [], []
    all_masked_pos, all_corrupt_pos = [], []
    for prompt, response in samples:
        x_orig, x_init, src_mask, content_mask, masked_pos, corrupt_pos = prepare_sample(
            prompt, response, tok, L, rng, args.mask_frac, args.corrupt_frac, device)
        all_x_orig.append(x_orig)
        all_x_init.append(x_init)
        all_src_mask.append(src_mask)
        all_content_mask.append(content_mask)
        all_masked_pos.append(masked_pos)
        all_corrupt_pos.append(corrupt_pos)

    # Batched reconstruction
    all_x_out = []
    gen_texts = []
    n_total = len(samples)

    for batch_start in tqdm(range(0, n_total, batch_size), desc='Reconstructing',
                            disable=(n_total <= batch_size)):
        batch_end = min(batch_start + batch_size, n_total)
        x_init_batch = torch.stack(all_x_init[batch_start:batch_end])
        src_mask_batch = torch.stack(all_src_mask[batch_start:batch_end])
        content_mask_batch = torch.stack(all_content_mask[batch_start:batch_end])

        x_out = reconstruct(
            args.sampler, model, x_init_batch, src_mask_batch, content_mask_batch,
            args=args, tok=tok, verbose=args.verbose)

        for i in range(x_out.size(0)):
            all_x_out.append(x_out[i])
            cm = content_mask_batch[i].nonzero(as_tuple=True)[0]
            if cm.numel() > 0:
                s, e = cm.min().item(), cm.max().item() + 1
                gen_texts.append(tok.decode(x_out[i, s:e].tolist()))
            else:
                gen_texts.append('')

    # Metrics
    agg = {'masked_rec': [], 'corrupt_corr': [], 'clean_pres': [],
           'overall': [], 'corrupt_revised': [], 'edit_precision': []}

    def fmt(v): return f'{v:.1%}' if v == v else 'N/A'

    for idx in range(n_total):
        x_orig = all_x_orig[idx]
        x_init = all_x_init[idx]
        x_out = all_x_out[idx]
        content_mask = all_content_mask[idx]
        masked_pos = all_masked_pos[idx]
        corrupt_pos = all_corrupt_pos[idx]
        clean_pos = content_mask & ~masked_pos & ~corrupt_pos

        def rate(m):
            return (x_out[m] == x_orig[m]).float().mean().item() if m.sum() > 0 else float('nan')

        mr = rate(masked_pos)
        cr = rate(corrupt_pos)
        cp = rate(clean_pos)
        ov = rate(content_mask)
        cv = (x_out[corrupt_pos] != x_init[corrupt_pos]).float().mean().item() if corrupt_pos.sum() > 0 else float('nan')

        unmasked = content_mask & ~masked_pos
        changed = unmasked & (x_out != x_init)
        n_changed = changed.sum().item()
        ep = (changed & corrupt_pos).sum().item() / n_changed if n_changed > 0 else float('nan')

        agg['masked_rec'].append(mr)
        agg['corrupt_corr'].append(cr)
        agg['clean_pres'].append(cp)
        agg['overall'].append(ov)
        agg['corrupt_revised'].append(cv)
        agg['edit_precision'].append(ep)

        if batch_size == 1 and args.verbose:
            print(f'\n-- Sample {idx+1}/{n_total} --')
            print(f'  Masked recovery: {fmt(mr)}  |  Corrupt correction: {fmt(cr)}')
            print(f'  Corrupt revised: {fmt(cv)}  |  Clean preservation: {fmt(cp)}')
            print(f'  Edit precision:  {fmt(ep)}  |  Overall accuracy:   {fmt(ov)}')
            visualize(tok, x_orig, x_init, x_out, content_mask, masked_pos, corrupt_pos)

    def avg(lst):
        valid = [v for v in lst if v == v]
        return statistics.mean(valid) if valid else float('nan')

    mr_avg = avg(agg['masked_rec'])
    cr_avg = avg(agg['corrupt_corr'])
    cp_avg = avg(agg['clean_pres'])
    ov_avg = avg(agg['overall'])
    cv_avg = avg(agg['corrupt_revised'])
    ep_avg = avg(agg['edit_precision'])

    def fmt_avg(v): return f'{v:.2%}' if v == v else 'N/A'
    print('\n' + '=' * 55)
    print('  AGGREGATE SUMMARY')
    print('=' * 55)
    print(f'  Masked recovery:     {fmt_avg(mr_avg)}')
    print(f'  Corrupt correction:  {fmt_avg(cr_avg)}')
    print(f'  Corrupt revision:    {fmt_avg(cv_avg)}')
    print(f'  Edit precision:      {fmt_avg(ep_avg)}')
    print(f'  Clean preservation:  {fmt_avg(cp_avg)}')
    print(f'  Overall accuracy:    {fmt_avg(ov_avg)}')
    print(f'  Sampler={args.sampler}  steps={args.diffusion_steps}')
    print(f'  mask_frac={args.mask_frac}  corrupt_frac={args.corrupt_frac}')
    print(f'  n_samples={n_total}  eval_batch_size={batch_size}')
    if args.sampler == 'proseco':
        print(f'  proseco_budget={args.proseco_budget}  proseco_freq={args.proseco_freq}')
    print('=' * 55)

    tag = args.dataset or os.path.splitext(os.path.basename(args.input_file))[0]
    config_key = f'{tag}_{args.sampler}_steps{args.diffusion_steps}'

    if args.save_gen_dir:
        fpath = os.path.join(args.save_gen_dir, f'{config_key}.jsonl')
        with open(fpath, 'w') as f:
            for t in gen_texts:
                f.write(json.dumps(t) + '\n')
        print(f'Saved {len(gen_texts)} outputs → {fpath}')

    if args.save_res_dir:
        results = {
            'config': {
                'model_name_or_path': args.model_name_or_path,
                'checkpoint_dir': args.checkpoint_dir,
                'dataset': tag,
                'sampler': args.sampler,
                'diffusion_steps': args.diffusion_steps,
                'mask_frac': args.mask_frac,
                'corrupt_frac': args.corrupt_frac,
                'n_samples': n_total,
                'eval_batch_size': batch_size,
                'n_correct_per_step': args.n_correct_per_step,
                'proseco_budget': args.proseco_budget,
                'proseco_freq': args.proseco_freq,
                'seed': args.seed,
            },
            'aggregate': {
                'masked_rec': mr_avg,
                'corrupt_corr': cr_avg,
                'corrupt_revised': cv_avg,
                'edit_precision': ep_avg,
                'clean_pres': cp_avg,
                'overall': ov_avg,
            },
            'per_sample': agg,
        }
        fpath = os.path.join(args.save_res_dir, f'{config_key}.json')
        with open(fpath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Saved results → {fpath}')


if __name__ == '__main__':
    main()
