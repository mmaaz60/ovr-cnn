OUTPUT_DIR="$HOME/runs/vltrain/121"
CONFIG_FILE="configs/mmss_v07.yaml"
DISTRIBUTED_ADDRESS="10.127.30.42"
PORT=52155
NUM_MACHINES=8
GPUS_PER_MACHINE=1
MACHINE_RANK=$1

python -m torch.distributed.launch --nproc_per_node="$GPUS_PER_MACHINE" --nnodes="$NUM_MACHINES" --node_rank="$MACHINE_RANK" --master_addr="$DISTRIBUTED_ADDRESS" --master_port="$PORT" tools/train_net.py --config-file "$CONFIG_FILE" --skip-test OUTPUT_DIR "$OUTPUT_DIR"