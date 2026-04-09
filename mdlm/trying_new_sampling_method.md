# Trying a New Sampling Method

As of now, we should implement a new method in the following two places:

1. **`model_eval.py`** — replace `sample_refine_v2` (generation from scratch)
2. **`eval_reconstruct.py`** — replace `reconstruct_ddpm_stochastic_backloaded` (reconstruction of masked/corrupted text)

---

## model_eval.py — `sample_refine_v2`

Located at [model_eval.py:622](model_eval.py#L622). This is the generation sampler — it generates text from a fully masked sequence.

### Function signature

```python
def sample_refine_v2(model, n_samples: int, steps: int, gen_len: int,
                     device: str, n_correct_per_step: int = 50,
                     correct_mode: str = 'topk',
                     correct_threshold: float = 0.01,
                     correction_ratio: float = 1.0,
                     fill_ratio: float = 0.75, **_):
```

### Inputs

- `model` — loaded MDLM model (already on device, in eval mode, with EMA applied)
- `n_samples` — number of sequences to generate in this batch
- `steps` — number of diffusion steps (NFE budget)
- `gen_len` — sequence length to generate
- `device` — `'cuda'` or `'cpu'`
- `**_` — catches any extra kwargs passed by the harness (proseco args, etc.)

The remaining args (`n_correct_per_step`, `correct_mode`, etc.) are specific to the current refine_v2 implementation. Replace them with whatever your method needs.

### Expected output

Return a tensor of shape `(n_samples, gen_len)` containing token IDs. The tensor should be on the same device.

### Where it's referenced

- Registered in the `SAMPLERS` dict at [model_eval.py:841](model_eval.py#L841):
  ```python
  SAMPLERS = {
      'vanilla': sample_vanilla,
      'refine':  sample_refine_v2,   # <-- your function here
      'proseco': sample_proseco,
      'proseco_m': sample_proseco_m,
  }
  ```
- Called by `generate_samples()` at [model_eval.py:850](model_eval.py#L850), which handles batching and passes kwargs from the command line args.

---

## eval_reconstruct.py — `reconstruct_ddpm_stochastic_backloaded`

Located at [eval_reconstruct.py:327](eval_reconstruct.py#L327). This is the reconstruction sampler — it takes a partially masked/corrupted sequence and tries to reconstruct it.

### Function signature

```python
def reconstruct_ddpm_stochastic_backloaded(model, x_init, steps, device,
                         collect_steps=False, content_mask=None,
                         n_correct_per_step=10, show_frequency=1,
                         guidance_scale=1.0, proseco_budget=1, proseco_freq=1):
```

### Inputs

- `model` — loaded MDLM model
- `x_init` — input tensor of shape `(batch, seq_len)` with some positions masked/corrupted
- `steps` — number of diffusion steps
- `device` — `'cuda'` or `'cpu'`
- `collect_steps` — whether to collect per-step visualization data
- `content_mask` — boolean tensor of shape `(batch, seq_len)`, marks which positions are content (vs padding/special tokens). Corrections should only apply to positions where this is `True`.
- `show_frequency` — collect visualization data every N steps

### Expected output

Return a tuple of `(x_out, step_data, flip_counts)`:

- `x_out` — tensor of shape `(batch, seq_len)` with reconstructed token IDs
- `step_data` — list of per-step visualization tuples (can be an empty list `[]` if `collect_steps` is `False`)
- `flip_counts` — list of per-step flip count tensors (can be an empty list `[]`)

### Where it's referenced

- Called by `run_reconstruction()` at [eval_reconstruct.py:631](eval_reconstruct.py#L631) under the `'refine'` sampler branch:
  ```python
  elif sampler_name == 'refine':
      kwargs.pop('proseco_budget', None)
      kwargs.pop('proseco_freq', None)
      kwargs.pop('corrector_sampling', None)
      x_out, step_data, flip_counts = reconstruct_ddpm_stochastic_backloaded(
          model, x_init, steps, device, **kwargs)
      return x_out, step_data, flip_counts
  ```

---

## Adding command line arguments

### model_eval.py

Add your argument in `parse_args()` at [model_eval.py:1268](model_eval.py#L1268):

```python
p.add_argument('--my_new_arg', type=float, default=0.5,
               help='Description of what it does')
```

Then pass it through `generate_samples()` at [model_eval.py:850](model_eval.py#L850). Add it to the function signature and forward it to the sampler call at line 861:

```python
tokens = sampler_fn(
    model=model, n_samples=bs, steps=steps, gen_len=gen_len,
    device=device, proseco_budget=proseco_budget, proseco_freq=proseco_freq,
    corrector_sampling=corrector_sampling,
    correction_ratio=correction_ratio, fill_ratio=fill_ratio,
    my_new_arg=my_new_arg,  # add here
)
```

Your sampler function picks it up as a keyword argument. The `**_` in the signature catches any kwargs your function doesn't use, so other samplers won't break.

### eval_reconstruct.py

Add your argument in `parse_args()` at [eval_reconstruct.py:884](eval_reconstruct.py#L884):

```python
p.add_argument('--my_new_arg', type=float, default=0.5,
               help='Description of what it does')
```

It gets passed through `run_reconstruction()` at [eval_reconstruct.py:618](eval_reconstruct.py#L618) via `**kwargs`. If you need to explicitly handle it (instead of letting it pass through), pop it from kwargs in the `'refine'` branch like the other samplers do.

---

## Adding a new sampler to the command line selection

If you want to add a new sampler name instead of replacing `refine`:

1. In `model_eval.py`, add your function to the `SAMPLERS` dict at [line 841](model_eval.py#L841):
   ```python
   SAMPLERS = {
       'vanilla': sample_vanilla,
       'refine':  sample_refine_v2,
       'proseco': sample_proseco,
       'proseco_m': sample_proseco_m,
       'my_sampler': my_sampler_fn,
   }
   ```

2. Add the name to the `--samplers` choices at [line 1269](model_eval.py#L1269):
   ```python
   p.add_argument('--samplers', nargs='+',
       choices=['vanilla', 'refine', 'proseco', 'proseco_m', 'my_sampler'], required=True)
   ```

3. In `eval_reconstruct.py`, add your name to `RECON_SAMPLERS` at [line 615](eval_reconstruct.py#L615) and add a branch in `run_reconstruction()` at [line 618](eval_reconstruct.py#L618).
