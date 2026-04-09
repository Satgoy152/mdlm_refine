#!/bin/bash
#SBATCH --job-name=sweep_0.75_fr
#SBATCH --output=slurm_output/sweep_0.75_fr.out
#SBATCH --error=slurm_output/sweep_0.75_fr.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=24:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=16

# Sweep over correction_ratio and fill_ratio for refine sampler
source /home/sagoyal/research/mdlm/.venv/bin/activate

CORRECTION_RATIOS="0.75"
FILL_RATIOS="0.25 0.5 0.75 1.0"

CSV_FILE="./sweep_results/sweep_0.75_fr.csv"

echo "correction_ratio,fill_ratio,gen_ppl,entropy,rep_n" > "$CSV_FILE"

for CR in $CORRECTION_RATIOS; do
  for FR in $FILL_RATIOS; do
    echo "============================================"
    echo "Running: correction_ratio=$CR  fill_ratio=$FR"
    echo "============================================"

    RES_DIR="./eval_results/cr${CR}_fr${FR}"
    mkdir -p "$RES_DIR"

    python model_eval.py \
      --models refine \
      --samplers refine \
      --num_samples 100 \
      --num_steps 128 \
      --gen_len 1024 \
      --metrics gen_ppl entropy rep_n \
      --gen_model gpt2 \
      --save_gen_dir ./eval_reconstruct_samples \
      --save_res_dir "$RES_DIR" \
      --corrector_sampling 'topk' \
      --eval_batch_size 10 \
      --correction_ratio "$CR" \
      --fill_ratio "$FR"

    # Extract metrics from the most recent results JSON
    RESULTS_JSON=$(ls -t "$RES_DIR"/results_*.json 2>/dev/null | head -1)
    if [ -n "$RESULTS_JSON" ]; then
      GEN_PPL=$(python -c "
import json, sys
with open('$RESULTS_JSON') as f:
    data = json.load(f)
for k, v in data.items():
    gp = v.get('gen_ppl', {}).get('gpt2', 'NA')
    ent = v.get('entropy', 'NA')
    rn = v.get('rep_n', 'NA')
    print(f'{gp},{ent},{rn}')
    break
")
      echo "${CR},${FR},${GEN_PPL}" >> "$CSV_FILE"
      echo "  -> Recorded: CR=$CR, FR=$FR, metrics=$GEN_PPL"
    else
      echo "${CR},${FR},NA,NA,NA" >> "$CSV_FILE"
      echo "  -> WARNING: No results JSON found for CR=$CR, FR=$FR"
    fi

    echo ""
  done
done

echo ""
echo "=========================================="
echo "  SWEEP COMPLETE — Results Summary"
echo "=========================================="
echo ""
column -t -s',' "$CSV_FILE"
echo ""
echo "Full results saved to: $CSV_FILE"
