set -e

echo "--------------set VLLM API begin------------------"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
if [ ! -d '/cache/output' ]; then
  mkdir -p /cache/output
fi
nohup vllm serve /cache/pretrained/Qwen3-VL-4B-Instruct \
    --port 18900 \
    --gpu-memory-utilization 0.1 \
    --tensor-parallel-size 8 \
    --served-model-name "Qwen3-VL-4B" \
    --trust-remote-code \
    --disable-log-requests > /cache/output/vllm_api.log 2>&1 &

until curl -sf http://127.0.0.1:18900/v1/models >/dev/null; do
      echo "Waiting for Judge API..."
      sleep 5
done
echo "--------------set VLLM API end------------------"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -x

export PYTHONUNBUFFERED=1

OUTPUT_NAME=BEE-RL-4B

unset LD_PRELOAD
unset NCCL_TOPO_FILE
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export RAY_WORKER_REGISTER_TIMEOUT_SECONDS=120
export VLLM_NO_USAGE_STATS=1
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DASHBOARD=1
export RAY_DASHBOARD_ENABLED=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_NUM_CPUS=1
export RAY_NUM_GPUS=8
export USE_RAY_LOCAL=1
export RAY_ADDRESS=local
export RAY_METRICS_EXPORT_PORT=0
export RAY_LOG_TO_STDERR=0
export RAY_LOCAL_MODE=0
export RAY_task_exit_on_oom=1
export RAY_SPILL_DIR=/cache/ray_spill
export RAY_TMPDIR=/cache/ray_tmp
mkdir -p /cache/ray_spill /cache/ray_tmp
export RAY_WORKER_REGISTER_TIMEOUT_SECONDS=300

MODEL_PATH=/cache/pretrained/Qwen3-VL-4B-Instruct-BEE-SFT-4b
SAVE_PATH=/cache/output/$OUTPUT_NAME
IMAGE_DIR=/cache/data/Thyme-RL
ROLLOUT_N=8
TEMPERATURE=1.0
GPU_UTILIZATION=0.6
ORI_BSZ=512
N_GPUS_PER_NODE=8
TENSOR_PARALLEL_SIZE=1
MAX_PROMPT_LENGTH=25952
MAX_RESPONSE_LENGTH=2048
LR=1e-6
TOTAL_EPOCHS=1
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://127.0.0.1:18900/v1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "--------------train begin------------------"
python -m verl.trainer.main \
    config=examples/config_bee.yaml \
    data.train_files=/path \
    data.val_files=/path \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=Epochs${TOTAL_EPOCHS}_LR${LR}_Rollout${ROLLOUT_N}_temp${TEMPERATURE} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    worker.rollout.tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
    worker.actor.optim.strategy=adamw \
    worker.actor.optim.lr=${LR} \
    data.mini_rollout_batch_size=128 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.3 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=False \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.temperature=${TEMPERATURE} \
    worker.rollout.gpu_memory_utilization=${GPU_UTILIZATION} \
    worker.rollout.enable_chunked_prefill=true \
    data.image_dir=${IMAGE_DIR} \
    data.rollout_batch_size=${ORI_BSZ} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=2 \
    data.min_pixels=262144 \
    data.max_pixels=16777216 \
    worker.rollout.max_num_batched_tokens=28000
echo "--------------train done------------------"

trap 'kill "$JUDGE_PID" 2>/dev/null || true' EXIT