export MASTER_PORT="12348"
python=/remote-home/zyzeng/miniconda3/envs/collie/bin/python
dynamic_incremental='Incremental_Chunk_Streaming_Dynamic_History'
fixed_incremental='Incremental_Chunk_Streaming_Fixed_History'
double_incremental="dynamic_incremental_double_compress"
dual_memory="dual_memory"
chunk_streaming='Chunk_Streaming'

pruner_types=("tova")
memory_types=($chunk_streaming)
eval_lens=(16384)
num_gpus=4
compress_ratio=0.125

for eval_len in "${eval_lens[@]}"; do
for memory_type in "${memory_types[@]}"; do
for pruner_type in "${pruner_types[@]}"; do
echo $eval_len $memory_type $pruner_type
srun -p moss -n${num_gpus} --gres=gpu:${num_gpus} -u ${python} /remote-home/zyzeng/collie/examples/mem_perceiver/finetune_mem_perceive.py \
  --pruner_type $pruner_type \
  --do_eval \
  --lr 0.00002 \
  --memory_type $memory_type \
  --llm_model llama2_7b \
  --num_train_samples 15000 \
  --num_eval_samples 128 \
  --chunk_size 512 \
  --use_flash \
  --num_gpus ${num_gpus} \
  --eval_len $eval_len \
  --compressed_chunk_size 512 \
  --compress_ratio ${compress_ratio}
done
done
done