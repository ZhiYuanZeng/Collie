export MASTER_PORT="12346"
python=/remote-home/zyzeng/miniconda3/envs/collie/bin/python
srun -p moss --gres=gpu:1 -u -n1 ${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
--pruner_type h2o \
--do_eval \
--lr 0.00002 \
--memory_type write_new_compressed_read_new_compressed

srun -p moss --gres=gpu:1 -u -n1 ${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
--pruner_type h2o \
--do_eval \
--lr 0.00002 \
--memory_type update_incremental_compressed_read_all_compressed
