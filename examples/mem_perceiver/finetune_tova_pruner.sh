export MASTER_PORT="12347"
python=/remote-home/zyzeng/miniconda3/envs/collie/bin/python
memory_type='Incremental_Chunk_Streaming_Dynamic_History'
pruner_type='tova'


srun -p a800 -n1 --gres=gpu:1 -u ${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
  --pruner_type $pruner_type \
  --do_eval \
  --lr 0.00002 \
  --memory_type $memory_type \
  --llm_model llama2_7b \
  --num_train_samples 4000 \
  --num_eval_samples 128 \
  --chunk_size 1024 \
  --query_len 128