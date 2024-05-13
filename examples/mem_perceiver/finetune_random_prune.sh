export MASTER_PORT="12348"
python=/remote-home/zyzeng/miniconda3/envs/collie/bin/python
list=('write_new_compressed_read_new_compressed' 'increment_compressed_read_all_compressed' 'update_incremental_compressed_read_all_compressed' 'increment_all_read_retrieved' 'increment_compressed_read_retrieved')
for item in "${list[@]}"; do
  echo run $item
  srun -p moss --gres=gpu:1 -u -n1 ${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
    --pruner_type random_prune \
    --do_eval \
    --lr 0.00002 \
    --memory_type $item
done
# --perceiver_path /remote-home/zyzeng/collie/ckpts/parallel_sparse/epoch_1/