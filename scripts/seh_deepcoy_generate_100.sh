#!/bin/bash

#SBATCH --job-name=dcoy_seh_gen
#SBATCH --partition=L40
#SBATCH -N 1
#SBATCH --cpus-per-task=7
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40:1
#SBATCH --array=0-7
#SBATCH --output=seh_del_deepcoy_generate_100_%A_%a.out
#SBATCH --error=seh_del_deepcoy_generate_100_%A_%a.err
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/share/home/u25511/wangyan/DeepCoy

set -eo pipefail

source /share/home/u25511/miniforge3/bin/activate /share/home/u25511/wangyan/deepcoy_env

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export TF_CPP_MIN_LOG_LEVEL=1

CHUNK_ID="$(printf "%02d" "${SLURM_ARRAY_TASK_ID}")"
DATASET_NAME="seh_del_chunk_${CHUNK_ID}"
BASE_DIR="seh_del_1to10"
CHUNK_SMI="${BASE_DIR}/sEH_active_chunk_${CHUNK_ID}.smi"
RUN_DIR="${BASE_DIR}/deepcoy_dude_100_array8/chunk_${CHUNK_ID}"
JSON_FILE="${BASE_DIR}/molecules_${DATASET_NAME}.json"
GEN_FILE="${RUN_DIR}/sEH_DeepCoy_generated_100_chunk_${CHUNK_ID}.txt"

mkdir -p "${RUN_DIR}"

echo "Start: $(date)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST:-local}"
echo "Chunk input: ${CHUNK_SMI}"
echo "Generated output: ${GEN_FILE}"

if [ ! -s "${JSON_FILE}" ]; then
    echo "Preparing DeepCoy JSON: $(date)"
    (cd data && python -u prepare_data.py \
        --data_path "../${CHUNK_SMI}" \
        --dataset_name "${DATASET_NAME}" \
        --save_dir "../${BASE_DIR}/")
fi

if [ ! -s "${GEN_FILE}" ]; then
    echo "Generating candidate decoys on CPU: $(date)"
    python -u DeepCoy.py \
        --restore models/DeepCoy_DUDE_model_e09.pickle \
        --dataset zinc \
        --config "{\"generation\": true, \"number_of_generation_per_valid\": 100, \"batch_size\": 1, \"train_file\": \"${JSON_FILE}\", \"valid_file\": \"${JSON_FILE}\", \"output_name\": \"${GEN_FILE}\", \"use_subgraph_freqs\": false}"
fi

echo "Generated candidate lines:"
wc -l "${GEN_FILE}"
echo "Done: $(date)"
