#!/bin/bash

#SBATCH --job-name=dcoy_seh_sel
#SBATCH --partition=L40
#SBATCH -N 1
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40:1
#SBATCH --output=seh_del_deepcoy_select_100_%j.out
#SBATCH --error=seh_del_deepcoy_select_100_%j.err
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/share/home/u25511/wangyan/DeepCoy

set -eo pipefail

source /share/home/u25511/miniforge3/bin/activate /share/home/u25511/wangyan/deepcoy_env

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""

BASE_DIR="seh_del_1to10"
RUN_DIR="${BASE_DIR}/deepcoy_dude_100_array8"
MERGED_FILE="${RUN_DIR}/sEH_DeepCoy_generated_100_merged.txt"
SELECT_INPUT_DIR="${RUN_DIR}/selection_input"
SELECT_OUT_DIR="${RUN_DIR}/selection_output"
FINAL_OUT_DIR="/share/home/u25511/wangyan/HitScreen/data/finetune_DEL_samples_filtered/sEH_DeepCoy_1to10"

mkdir -p "${SELECT_INPUT_DIR}" "${SELECT_OUT_DIR}" "${FINAL_OUT_DIR}"

echo "Start: $(date)"
echo "Merging chunk outputs..."

rm -f "${MERGED_FILE}"
for f in "${RUN_DIR}"/chunk_*/sEH_DeepCoy_generated_100_chunk_*.txt; do
    if [ ! -s "$f" ]; then
        echo "Missing or empty chunk output: $f" >&2
        exit 1
    fi
    cat "$f" >> "${MERGED_FILE}"
done

echo "Merged candidate lines:"
wc -l "${MERGED_FILE}"

cp "${MERGED_FILE}" "${SELECT_INPUT_DIR}/generated_decoys.txt"

echo "Selecting DeepCoy decoys: $(date)"

python /share/home/u25511/wangyan/all9_comprehensive_finetune_analysis/public_data_control/select_deepcoy_mw_matched_del_decoys.py \
    --reference-train /share/home/u25511/wangyan/HitScreen/data/finetune_DEL_samples_filtered/sEH/train.csv \
    --reference-val /share/home/u25511/wangyan/HitScreen/data/finetune_DEL_samples_filtered/sEH/val.csv \
    --generated-pairs "${MERGED_FILE}" \
    --test-csv /share/home/u25511/wangyan/CHEMBL_with_DUDE_csv_20260715/sEH_CHEMBL_with_DUDE.csv \
    --outdir "${FINAL_OUT_DIR}" \
    --target sEH_DeepCoy_MWmatched_1to10 \
    --ratio 10 \
    --seed 20260820 \
    --max-tanimoto 0.3 \
    --mw-tolerance 50 \
    --logp-tolerance 1.0 \
    --rb-tolerance 4 \
    > "${RUN_DIR}/sEH_DeepCoy_MWmatched_selection_1to10_log.txt"

echo "Selected files:"
find "${SELECT_OUT_DIR}" "${FINAL_OUT_DIR}" \
    -maxdepth 1 \
    -type f \
    -printf '%s %p\n' | sort -n

echo "Done: $(date)"
