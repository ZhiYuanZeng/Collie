export MASTER_PORT="12349"
python=/remote-home/zyzeng/miniconda3/envs/collie/bin/python
dynamic_incremental='Incremental_Chunk_Streaming_Dynamic_History'
fixed_incremental='Incremental_Chunk_Streaming_Fixed_History'
double_incremental="dynamic_incremental_double_compress"
dual_memory="dual_memory"
chunk_streaming='Chunk_Streaming'

tova="tova"
conv="conv"

memory_sizes=(512)
pruner_types=("streaming_llm")
memory_types=($dynamic_incremental $chunk_streaming)
eval_len=16384
num_gpus=1
chunk_size=4096
incremental_types=('decremental_chunk')
# incremental_types=('linear' 'sqrt' 'square' 'inverse_concave' 'inverse_convex' 'square_sqrt' 'double_inverse' 'adaptive')

llm_model="llama2_7b"
for it in ${incremental_types[@]}; do
for ms in "${memory_sizes[@]}"; do
for memory_type in "${memory_types[@]}"; do
for pruner_type in "${pruner_types[@]}"; do
num_chunks=$(expr $eval_len / $chunk_size)
echo $eval_len $memory_type $pruner_type
# srun -p a800 -n${num_gpus} --gres=gpu:${num_gpus} --mem-per-cpu=4G -u 
${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
  --pruner_type $pruner_type \
  --do_eval \
  --lr 0.00002 \
  --memory_type $memory_type \
  --llm_model ${llm_model} \
  --num_train_samples 15000 \
  --num_eval_samples 128 \
  --chunk_size ${chunk_size} \
  --use_flash \
  --num_gpus ${num_gpus} \
  --eval_len $eval_len \
  --report_keep_memory_rate \
  --memory_size_limit ${ms} \
  --incremental_type ${it}
  # --tag 
done
done
done
done