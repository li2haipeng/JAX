python3 -m torch.distributed.launch \
    --nproc_per_node=8 --nnodes=2 --node_rank=1 \
    --master_addr="172.31.34.171" --master_port=1234 --use-env bert.py 512 40
