#!/bin/bash
python extract_samples.py \
  --dataset wikitext103 \
  --output sample_wikitext.jsonl \
  --num_samples 100 \
  --cache_dir /nfs/turbo/coe-jjparkcv-medium/satyam/.cache/datasets
