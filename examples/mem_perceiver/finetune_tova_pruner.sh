export MASTER_PORT="12348"
python=/remote-home/zyzeng/miniconda3/envs/collie/bin/python
memory_type='dynamic_incremental_double_compress'
pruner_type='tova'


srun -p moss -n1 --gres=gpu:1 -u ${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
  --pruner_type $pruner_type \
  --do_eval \
  --lr 0.00002 \
  --memory_type $memory_type \
  --llm_model llama2_7b \
  --num_train_samples 15000 \
  --num_eval_samples 128 \
  --chunk_size 512 \
  --query_len 64 \
  --use_flash \
  --num_gpus 1 \
  --compressed_chunk_size 64 \
  --temperature 1.0 \