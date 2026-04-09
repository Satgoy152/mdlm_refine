"""Extract random samples from a dataset and save to JSONL."""

import argparse
import json
import random

import omegaconf
import torch

import dataloader


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset config name (e.g. wikitext103)')
  parser.add_argument('--output', type=str, required=True,
                      help='Output JSONL file path')
  parser.add_argument('--num_samples', type=int, required=True,
                      help='Number of examples to save')
  parser.add_argument('--len', type=int, default=1024,
                      dest='max_len',
                      help='Max token length (default: 1024)')
  parser.add_argument('--cache_dir', type=str, default=None,
                      help='Cache dir for dataset (overrides config)')
  args = parser.parse_args()

  # Load dataset config
  data_cfg = omegaconf.OmegaConf.load(
    f'configs/data/{args.dataset}.yaml')

  if args.cache_dir is not None:
    data_cfg.cache_dir = args.cache_dir

  tokenizer = dataloader.get_tokenizer(
    omegaconf.OmegaConf.create({'data': data_cfg}))

  ds = dataloader.get_dataset(
    dataset_name=data_cfg.train,
    tokenizer=tokenizer,
    wrap=data_cfg.wrap,
    mode='train',
    cache_dir=data_cfg.cache_dir,
    block_size=args.max_len)

  # Random sample indices
  n = len(ds)
  k = min(args.num_samples, n)
  indices = random.sample(range(n), k)

  with open(args.output, 'w') as f:
    for idx in indices:
      tokens = ds[idx]['input_ids']
      text = tokenizer.decode(tokens, skip_special_tokens=True)
      f.write(json.dumps(text) + '\n')

  print(f'Saved {k} samples to {args.output}')


if __name__ == '__main__':
  main()
