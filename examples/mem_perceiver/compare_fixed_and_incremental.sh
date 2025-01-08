export MASTER_PORT="12348"
python=/remote-home/zyzeng/miniconda3/envs/collie/bin/python
dynamic_incremental='Incremental_Chunk_Streaming_Dynamic_History'
fixed_incremental='Incremental_Chunk_Streaming_Fixed_History'
double_incremental="dynamic_incremental_double_compress"
chunk_streaming='Chunk_Streaming'

pruner_types=('no_compress')
eval_len=32768
chunk_sizes=(8192)
num_gpus=1
memory_types=($chunk_streaming)
memory_sizes=(8192)

for mt in "${memory_types[@]}"; do
for chunk_size in "${chunk_sizes[@]}"; do
for pruner_type in "${pruner_types[@]}"; do
for ms in "${memory_sizes[@]}"; do
# srun -n${num_gpus} -p a800 --gres=gpu:${num_gpus} -u ${python} 
${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
  --pruner_type $pruner_type \
  --do_eval \
  --lr 0.00002 \
  --memory_type $mt \
  --llm_model llama2_7b \
  --num_train_samples 15000 \
  --num_eval_samples 8 \
  --chunk_size $chunk_size \
  --use_flash \
  --num_gpus ${num_gpus} \
  --eval_len $eval_len \
  --memory_size_limit $ms \
  --tag compare_fixed_and_incremental_v2
done
done
done
done
