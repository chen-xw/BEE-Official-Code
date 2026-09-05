
NPROC_PER_NODE=8

MODEL_NAME=qwen3vl_4b
DATASET_NAME=BEE_SFT
CONFIG=configs/train_full/$MODEL_NAME/$DATASET_NAME.yaml

echo "--------------------------------train begin--------------------------------"
torchrun \
    --nnodes=1 \
    --nproc_per_node=$NPROC_PER_NODE \
    src/train.py \
    $CONFIG
echo "--------------------------------train end--------------------------------"