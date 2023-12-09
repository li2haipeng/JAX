python3 -m torch.distributed.launch \
    --nproc_per_node=8 --nnodes=2 --node_rank=0 \
    --master_addr="172.31.45.102" --master_port=2244 --use-env bert.py 512 16

# python3 -m torch.distributed.launch \
#     --nproc_per_node=8 --use-env bert.py 512 32
