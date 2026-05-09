#!/bin/bash
#SBATCH --job-name=sweep_eval
#SBATCH --output=slurm_output/sweep_eval.out
#SBATCH --error=slurm_output/sweep_eval.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=spgpu2
#SBATCH --account=jjparkcv_owned1
#SBATCH --time=24:00:00
#SBATCH --mem=40GB
#SBATCH --cpus-per-task=8

# Sweeps over task x model x sampling_method x diffusion_steps,
# parses predict_acc and predict_loss from each log, and writes a CSV.
source /home/sagoyal/research/diffusion-vs-ar/.venv/bin/activate

export WANDB_DISABLED=true
export WANDB_PROJECT="diffusion-vs-ar"

# ---- Sweep configuration ----
TASKS=(3-sat cd sudoku)
MODELS=(refine_3 proseco vanilla)
SAMPLING_METHODS=(vanilla)
DIFFUSION_STEPS=(1 2 5 10 20)

# Per-sampling hyperparameters (override here as needed).
TOPK_DECODING=False            # used by vanilla and refine
N_CORRECT_PER_STEP=8          # refine only
CORRECT_MODE=topk             # refine only
PROSECO_BUDGET=1              # proseco only
PROSECO_FREQ=1                # proseco only
DECODING_STRATEGY_DEFAULT=deterministic-linear   # vanilla / refine
DECODING_STRATEGY_PROSECO=deterministic-linear   # proseco

CACHE_DIR=/nfs/turbo/coe-jjparkcv-medium/satyam/.cache
EXP_ROOT=/nfs/turbo/coe-jjparkcv-medium/satyam/sr
EVAL_ROOT=/home/sagoyal/research/mdlm_refine/diffusion-vs-ar/eval_results/sweep
mkdir -p "$EVAL_ROOT"

CSV_OUT="$EVAL_ROOT/sweep_results.csv"
echo "task,model,sampling_method,diffusion_steps,predict_acc,predict_loss,extra,log_path" > "$CSV_OUT"

# ---- Per-task settings ----
# Returns: <ckpt_task_dir> <dataset_name> <model_config> <cutoff_len>
task_settings() {
    case "$1" in
        3-sat)
            echo "sat 3sat9_test model_config_tiny 325"
            ;;
        sudoku)
            echo "sudoku sudoku_test model_config_tiny 164"
            ;;
        cd)
            echo "cd5 cd5_test model_config 64"
            ;;
        *)
            echo "" ; return 1 ;;
    esac
}

# Parse "predict_acc" and "predict_loss" from a log file.
parse_metric() {
    local log_path="$1"
    local key="$2"
    grep -E "^\s*${key}\s*=" "$log_path" | tail -n 1 | awk -F'=' '{print $2}' | tr -d ' '
}

# Resolve a usable checkpoint dir. Some training runs (e.g. sat/refine_3,
# cd5/refine_3) have not been consolidated to the top level yet, so the
# tokenizer/model files only live inside the latest checkpoint-* subdir.
resolve_ckpt_dir() {
    local base="$1"
    if [[ -f "$base/tokenizer_config.json" ]]; then
        echo "$base"
        return 0
    fi
    local latest
    latest="$(ls -d "$base"/checkpoint-* 2>/dev/null | sort -V | tail -n 1)"
    if [[ -n "$latest" && -f "$latest/tokenizer_config.json" ]]; then
        echo "$latest"
        return 0
    fi
    echo ""
    return 1
}

# ---- Main loop ----
for task in "${TASKS[@]}"; do
    read -r ckpt_dir dataset model_cfg cutoff_len <<< "$(task_settings "$task")"
    if [[ -z "$ckpt_dir" ]]; then
        echo "[skip] unknown task: $task"
        continue
    fi

    for model in "${MODELS[@]}"; do
        exp_base="$EXP_ROOT/$ckpt_dir/$model"
        if [[ ! -d "$exp_base" ]]; then
            echo "[skip] missing checkpoint dir: $exp_base"
            continue
        fi
        exp="$(resolve_ckpt_dir "$exp_base")"
        if [[ -z "$exp" ]]; then
            echo "[skip] no usable checkpoint (no tokenizer_config.json) under: $exp_base"
            continue
        fi
        if [[ "$exp" != "$exp_base" ]]; then
            echo "[info] using checkpoint subdir: $exp"
        fi

        for sampling in "${SAMPLING_METHODS[@]}"; do
            for steps in "${DIFFUSION_STEPS[@]}"; do
                # Build the per-sampling argument set and a tag for filenames/CSV.
                extra_csv=""
                case "$sampling" in
                    vanilla)
                        sampling_args=(
                            --decoding_strategy "$DECODING_STRATEGY_DEFAULT"
                            --topk_decoding "$TOPK_DECODING"
                            --sampling_method vanilla
                            --n_correct_per_step 0
                            --correct_mode "$CORRECT_MODE"
                            --proseco_budget 1
                            --proseco_freq 1
                        )
                        tag="vanilla-topk${TOPK_DECODING}"
                        extra_csv="topk=${TOPK_DECODING}"
                        ;;
                    refine)
                        sampling_args=(
                            --decoding_strategy "$DECODING_STRATEGY_DEFAULT"
                            --topk_decoding "$TOPK_DECODING"
                            --sampling_method refine
                            --n_correct_per_step "$N_CORRECT_PER_STEP"
                            --correct_mode "$CORRECT_MODE"
                            --proseco_budget 1
                            --proseco_freq 1
                        )
                        tag="refine-n${N_CORRECT_PER_STEP}-topk${TOPK_DECODING}"
                        extra_csv="n_correct=${N_CORRECT_PER_STEP};topk=${TOPK_DECODING}"
                        ;;
                    proseco)
                        sampling_args=(
                            --decoding_strategy "$DECODING_STRATEGY_PROSECO"
                            --topk_decoding "$TOPK_DECODING"
                            --sampling_method proseco
                            --n_correct_per_step 0
                            --correct_mode "$CORRECT_MODE"
                            --proseco_budget "$PROSECO_BUDGET"
                            --proseco_freq "$PROSECO_FREQ"
                        )
                        tag="proseco-b${PROSECO_BUDGET}-f${PROSECO_FREQ}"
                        extra_csv="budget=${PROSECO_BUDGET};freq=${PROSECO_FREQ}"
                        ;;
                esac

                eval_dir="$EVAL_ROOT/$task/$model/$sampling/T${steps}_${tag}"
                mkdir -p "$eval_dir"
                log_path="$eval_dir/eval.log"

                echo "[run] task=$task model=$model sampling=$sampling steps=$steps -> $log_path"

                CUDA_VISIBLE_DEVICES=0 \
                python3 -u src/train_bash.py \
                    --stage mdm --overwrite_output_dir \
                    --cache_dir "$CACHE_DIR" \
                    --model_name_or_path "$model_cfg" \
                    --do_predict \
                    --cutoff_len "$cutoff_len" \
                    --dataset "$dataset" \
                    --finetuning_type full \
                    --diffusion_steps "$steps" \
                    --output_dir "$eval_dir" \
                    --checkpoint_dir "$exp" \
                    --remove_unused_columns False \
                    "${sampling_args[@]}" \
                    > "$log_path" 2>&1

                acc="$(parse_metric "$log_path" predict_acc)"
                loss="$(parse_metric "$log_path" predict_loss)"
                acc="${acc:-NA}"
                loss="${loss:-NA}"

                echo "${task},${model},${sampling},${steps},${acc},${loss},${extra_csv},${log_path}" >> "$CSV_OUT"
                echo "  -> acc=$acc loss=$loss"
            done
        done
    done
done

echo "Done. CSV written to: $CSV_OUT"
