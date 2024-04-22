export MASTER_PORT="12347"
python=/remote-home/zyzeng/miniconda3/envs/collie/bin/python
srun -p moss --gres=gpu:1 -u -n1 ${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py --compress_type chunk_prefix --do_eval