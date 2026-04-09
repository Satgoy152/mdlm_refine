# Evaluation

## model_eval.py

Main file for generating text from models and computing metrics. Supports loading refine, vanilla, and proseco models from the hardcoded checkpoint directories at the top of the file (`CHECKPOINTS` dict).

You can select any combination of models and samplers and evaluate them across multiple metrics.

### Models

Available models: `vanilla`, `refine`, `proseco`

### Samplers

Defined in the file under the `SAMPLERS` dict:

- `vanilla` — vanilla MDLM ancestral sampler using DDPM updates
- `refine` — DDPM unmasking + top-k correction of unmasked positions each step (currently)
- `proseco` — MDLM unmasking with periodic self-correction
- `proseco_m` — ProSeCo variant where correction phase sees masked + unmasked tokens

### Metrics

- **gen_ppl** — generative perplexity. Use `--gen_model` to select which model to evaluate with. Available models: `gpt2`, `llama3b`, `llama1b`.

  **Note:** Running llama models requires the eval venv (`.eval_venv`), which doesn't support inference. So the workflow is: first generate samples with the original venv, then compute llama gen_ppl with the eval venv using `--skip_inference`.

  Step 1 — generate samples with the original venv:
  ```bash
  source .venv/bin/activate

  python model_eval.py \
    --models refine \
    --samplers refine \
    --num_samples 128 \
    --num_steps 128 \
    --gen_len 1024 \
    --metrics gen_ppl \
    --gen_model gpt2 \
    --save_gen_dir ./eval_samples \
    --save_res_dir ./eval_results
  ```

  Step 2 — get llama gen_ppl with the eval venv:
  ```bash
  source .eval_venv/bin/activate

  python model_eval.py \
    --models refine \
    --samplers refine \
    --num_samples 128 \
    --num_steps 128 \
    --gen_len 1024 \
    --metrics gen_ppl \
    --gen_model llama1b \
    --save_gen_dir ./eval_samples \
    --save_res_dir ./eval_results \
    --skip_inference
  ```

- **mauve** — you need at least 5000 samples for accurate mauve results. Requires `--mauve_dataset` (e.g. `openwebtext-valid`, `openwebtext-split`, `wikitext103`).

- **entropy** — Shannon entropy over token distribution.

- **rep_n** — repeated n-gram rate. Change n with `--rep_n`:
  ```bash
  python model_eval.py \
    --models vanilla \
    --samplers vanilla \
    --num_samples 128 \
    --num_steps 128 \
    --gen_len 1024 \
    --metrics rep_n \
    --rep_n 3 \
    --save_res_dir ./eval_results
  ```

### Saving outputs

- **Samples** are saved to `--save_gen_dir`. If no dir is provided, samples won't be saved.
- **Results** (metrics) are saved to `--save_res_dir`. If none is provided, they won't be saved.

### File naming convention

Generated samples are saved as JSONL files:

```
{model}_{sampler}_steps{steps}.jsonl
```

For example: `refine_refine_steps128.jsonl`, `vanilla_proseco_steps256.jsonl`

Results are saved as a single timestamped JSON:

```
results_{YYYYMMDD_HHMMSS}.json
```

Inside the results JSON, keys follow the same `{model}_{sampler}_steps{steps}` pattern.

### Sampler-specific arguments

Different samplers accept additional arguments. For example, using the proseco sampler with a topk corrector:

```bash
python model_eval.py \
  --models vanilla \
  --samplers proseco \
  --num_samples 100 \
  --num_steps 128 256 512 1024 \
  --gen_len 1024 \
  --metrics gen_ppl \
  --gen_model gpt2 \
  --save_gen_dir ./eval_samples \
  --save_res_dir ./eval_results \
  --eval_batch_size 10 \
  --corrector_sampling topk
```

Available sampler arguments:

- `--proseco_budget` — number of corrector forward passes (default: 3)
- `--proseco_freq` — run corrector every N steps (default: 1)
- `--corrector_sampling` — token selection method: `argmax` or `topk` (default: `argmax`)
- `--correction_ratio` — fraction of steps used for correction in refine sampler (default: 1.0)
- `--fill_ratio` — fraction of masked positions to fill in refine correction (default: 0.75)

### Skip inference

If you already have generated samples saved, you can skip inference and just run metrics:

```bash
python model_eval.py \
  --models proseco \
  --samplers proseco \
  --num_samples 100 \
  --num_steps 128 \
  --gen_len 1024 \
  --metrics mauve \
  --gen_model llama1b \
  --mauve_dataset openwebtext-split \
  --save_gen_dir ./eval_samples \
  --save_res_dir ./eval_results \
  --skip_inference
```

This loads samples from `save_gen_dir` using the same `{model}_{sampler}_steps{steps}.jsonl` naming convention.

---

## eval_reconstruct.py

Diagnostic tool for testing whether a model can reconstruct masked/corrupted text. Uses the same model checkpoints as `model_eval.py` and provides reconstruction-adapted versions of all samplers.

### Basic usage

```bash
python eval_reconstruct.py \
  --model refine \
  --sampler refine \
  --input_file sample_wikitext.jsonl \
  --n_samples 100 \
  --steps 128 \
  --mask_frac 0.25 \
  --corrupt_frac 0.25 \
  --eval_batch_size 10 \
  --n_correct_per_step 10 \
  --save_gen_dir ./eval_reconstruct_samples \
  --save_res_dir ./eval_results
```

### Arguments

- `--model` — which model checkpoint to load (`vanilla`, `refine`, `proseco`)
- `--sampler` — sampling method: `vanilla`, `refine`, `proseco`, `proseco_m`
- `--input_file` — JSONL file containing source texts to reconstruct
- `--mask_frac` — fraction of content to mask, or `random` for per-sample random values (default: 0.5)
- `--corrupt_frac` — fraction of unmasked content to corrupt, or `random` for per-sample random values (default: 0.15)
- `--steps` — number of diffusion steps (default: 128)
- `--n_correct_per_step` — max corrections per DDPM step for refine sampler (default: 10)
- `--show_intermediate` — show intermediate outputs during reconstruction
- `--show_frequency` — show intermediate outputs every N steps (default: 1)
- `--atomic_scenario` — atomic comparison mode (`A` or `B`). A: 50% clean / 25% corrupt / 25% masked. B: 50% clean / 25% corrupt / 25% extra-corrupt.

### File naming convention

Same pattern as `model_eval.py`:

- Reconstructed samples: `{model}_{sampler}_steps{steps}.jsonl` in `save_gen_dir`
- Results: `{model}_{sampler}_steps{steps}.json` in `save_res_dir`
