import os
import jax

# this is in slurm fwiw
# os.environ['FI_EFA_FORK_SAFE'] = '1'
# os.environ['FI_EFA_USE_DEVICE_RDMA'] = '1'
# os.environ['FI_LOG_LEVEL'] = 'INFO'
os.environ['NCCL_DEBUG'] = 'TRACE'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
# local_device_ids=[int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')]

# print(local_device_ids)
jax.distributed.initialize(coordinator_address='172.31.45.102:2234', num_processes=2, process_id=0) 
# if slurm and one gpu per process then do
# jax.distributed.initialize()

print(jax.devices())
print(jax.process_count())

xs = jax.numpy.ones(jax.local_device_count())
r = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs)
print(r)