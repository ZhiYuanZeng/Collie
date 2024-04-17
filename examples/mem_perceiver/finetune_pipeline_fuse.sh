export MASTER_PORT="12346"
python=/remote-home/zyzeng/miniconda3/envs/collie/bin/python
srun -p moss --gres=gpu:1 -u -n1 ${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py --perceiver pipeline_fuse